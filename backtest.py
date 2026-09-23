"""
歷史回測（baseline）：對現行評分規則做回測，作為之後規則調整的比較基準。

設計原則
- 不重寫任何評分／特徵邏輯：直接呼叫 stock_analyzer.get_stock_data()、
  score_stock()、calc_trade_levels()、classify_stock()。做法是把
  stock_analyzer._finmind_request 換成「只回傳 as_of 當天（含）以前資料」
  的本機歷史快取，並把 stock_analyzer.datetime.today() 固定成當次模擬的
  執行日——get_stock_data 看到的資料窗口（90 天／430 天）與實盤完全一致。
- 不偷看未來：
  - 股價／法人／除權息：只取 date ≤ as_of。
  - 月營收：FinMind 沒有公告日欄位（date 是營收月份的次月 1 日，
    create_time 只有 2026-04 之後才有值且是爬取時間），一律假設 M 月營收
    在 M+1 月 10 日「之後」才可取得（as_of ≥ 11 日）。這比實盤保守：實盤
    若公司在 1～10 日提早公告，當天就會用到。

用法
  python backtest.py fetch     # 下載 13 檔完整歷史＋台指期到 backtest_cache/
  python backtest.py parity    # 用 data/signals_log.csv 比對實盤紀錄
  python backtest.py spread    # spread 欄位交叉比對未還原事件
  python backtest.py run --start 2024-01-01 --end 2026-06-22
  python backtest.py slices    # 法人連買天數切片（需先跑過 run，只讀快取）
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

import stock_analyzer as sa

# stock_analyzer 匯入時會設定 root logger（寫 analyzer.log + stdout）；回測
# 會呼叫 get_stock_data 數萬次，不能把 analyzer.log 灌爆，這裡改成只把
# WARNING 以上印到 stderr。
_root = logging.getLogger()
for _h in list(_root.handlers):
    _root.removeHandler(_h)
logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                    format="%(levelname)s %(message)s", force=True)
sa.log.setLevel(logging.ERROR)

_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR  = os.path.join(_DIR, "backtest_cache")
OUTPUT_DIR = os.path.join(_DIR, "backtest_output")

DATASETS = [
    "TaiwanStockPrice",
    "TaiwanStockInstitutionalInvestorsBuySell",
    "TaiwanStockMonthRevenue",
    "TaiwanStockDividendResult",
]

# 月營收可取得日：M 月營收（FinMind date = M+1 月 1 日）在 M+1 月的這一天
# 「之後」才可用。法定公告期限為次月 10 日。
REVENUE_PUBLISH_DAY = 10

FUTURES_DATASET = "TaiwanFuturesDaily"
FUTURES_ID      = "TX"
FUTURES_START_YEAR = 2017  # 夜盤 2017-05-15 開始

# 最後 3 個月保留為驗證期：開發期的回測一律截斷在這天之前，連出場用的
# 股價也不看驗證期的資料。
VALIDATION_START = "2026-06-23"

# 資料不可信期間：這些日期以前的訊號一律排除。6669 從上市到 2019-03，
# TaiwanStockPrice 的 spread 與前一日收盤對不上共 291 天，且多次出現超過
# 10% 漲跌幅限制的單日變動（見 python backtest.py spread）。
EXCLUDE_SIGNALS_BEFORE = {"6669": "2019-04-01"}

# 夜盤閘門開始有資料的日期；有閘門／無閘門的對照一律在這天之後的同一段期間比較
GATE_START = "2017-05-16"

PORTFOLIO_SEEDS = 40


# ══════════════════════════════════════════════════════════════
# 資料層：完整歷史快取
# ══════════════════════════════════════════════════════════════

def _cache_path(dataset: str, stock_id: str) -> str:
    return os.path.join(CACHE_DIR, f"{dataset}_{stock_id}.pkl")


def fetch_history(stock_ids, refresh: bool = False) -> None:
    """下載 FinMind 可取得的最長歷史（start_date=1990-01-01）到本機快取"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    for sid in stock_ids:
        for ds in DATASETS:
            path = _cache_path(ds, sid)
            if os.path.exists(path) and not refresh:
                continue
            resp = requests.get(sa.FINMIND_BASE, params={
                "dataset": ds, "data_id": sid,
                "start_date": "1990-01-01", "token": sa.FINMIND_TOKEN,
            }, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != 200:
                raise RuntimeError(f"FinMind {ds} ({sid}): {data.get('msg')}")
            pd.DataFrame(data.get("data") or []).to_pickle(path)
            print(f"  cached {ds} {sid}: {len(data.get('data') or [])} rows")
            time.sleep(0.5)

    path = _cache_path(FUTURES_DATASET, FUTURES_ID)
    if os.path.exists(path) and not refresh:
        return
    parts = []
    for y in range(FUTURES_START_YEAR, datetime.today().year + 1):
        resp = requests.get(sa.FINMIND_BASE, params={
            "dataset": FUTURES_DATASET, "data_id": FUTURES_ID,
            "start_date": f"{y}-01-01", "end_date": f"{y}-12-31",
            "token": sa.FINMIND_TOKEN,
        }, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != 200:
            raise RuntimeError(f"FinMind {FUTURES_DATASET} ({y}): {data.get('msg')}")
        parts.append(pd.DataFrame(data.get("data") or []))
        time.sleep(0.5)
    df = pd.concat(parts, ignore_index=True)
    df.to_pickle(path)
    print(f"  cached {FUTURES_DATASET} {FUTURES_ID}: {len(df)} rows")


class HistoryStore:
    """讀取本機快取，依 as_of 回傳「當時可取得」的資料切片"""

    def __init__(self):
        self._data = {}

    def _load(self, dataset: str, stock_id: str) -> pd.DataFrame:
        key = (dataset, stock_id)
        if key not in self._data:
            path = _cache_path(dataset, stock_id)
            if not os.path.exists(path):
                raise FileNotFoundError(f"缺少快取 {path}，請先執行 python backtest.py fetch")
            df = pd.read_pickle(path)
            if not df.empty:
                df["date"] = df["date"].astype(str)
            self._data[key] = df
        return self._data[key]

    def query(self, dataset: str, stock_id: str, start_date: str, as_of: str,
              night_as_of: "str | None" = None) -> pd.DataFrame:
        df = self._load(dataset, stock_id)
        if df.empty:
            return df
        if dataset == FUTURES_DATASET:
            # FinMind 把夜盤（D 日 15:00～D+1 日 05:00）標在「它所屬的下一個
            # 交易日」。實盤於 UTC 22:33（台灣 D+1 日 06:33）執行，這時 D 日
            # 夜盤已收盤，所以夜盤可取到 night_as_of（D 之後的下一個交易日），
            # 日盤只能取到 D。
            night_as_of = night_as_of or as_of
            is_night = df["trading_session"] == "after_market"
            out = df[(df["date"] >= start_date) & (
                (~is_night & (df["date"] <= as_of)) | (is_night & (df["date"] <= night_as_of))
            )]
            return out.reset_index(drop=True)
        out = df[(df["date"] >= start_date) & (df["date"] <= as_of)]
        if dataset == "TaiwanStockMonthRevenue":
            # date 是次月 1 日；可取得日 = 次月 REVENUE_PUBLISH_DAY 日之後
            avail = out["date"].str[:8] + f"{REVENUE_PUBLISH_DAY:02d}"
            out = out[avail < as_of]
        return out.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
# 單日訊號重建：呼叫實盤的 get_stock_data / score_stock / ...
# ══════════════════════════════════════════════════════════════

class _FrozenDatetime(datetime):
    _today = None

    @classmethod
    def today(cls):
        return cls._today

    @classmethod
    def now(cls, tz=None):
        return cls._today


def _patched(store: HistoryStore, as_of: str, run_date: str, night_as_of: "str | None" = None):
    """回傳 (enter, exit)：暫時把 stock_analyzer 的資料來源與「今天」換成回測版本"""
    def _fake_request(dataset, sid, start_date):
        return store.query(dataset, sid, start_date, as_of, night_as_of), None

    orig = (sa._finmind_request, sa.datetime, sa.time.sleep)

    def enter():
        _FrozenDatetime._today = datetime.strptime(run_date, "%Y-%m-%d")
        sa._finmind_request = _fake_request
        sa.datetime = _FrozenDatetime
        sa.time.sleep = lambda *_: None

    def exit_():
        sa._finmind_request, sa.datetime, sa.time.sleep = orig

    return enter, exit_


def compute_gate(store: HistoryStore, as_of: str, night_as_of: str) -> dict:
    """重建訊號日 as_of 當晚的台指期夜盤閘門：直接呼叫 sa.get_futures_gate()"""
    enter, exit_ = _patched(store, as_of, as_of, night_as_of)
    enter()
    try:
        return sa.get_futures_gate()
    finally:
        exit_()


def compute_signal(store: HistoryStore, stock_id: str, as_of: str, run_date: "str | None" = None) -> dict:
    """重建 stock_id 在 as_of（訊號日）收盤後的實盤輸出。

    run_date：實盤 datetime.today() 的日期，決定 get_stock_data 的 90 天／
    430 天窗口起點。實盤在 GitHub Actions（UTC）22:33 執行，UTC 日期即
    data_date，所以預設 run_date = as_of。

    回傳 get_stock_data 的 result，再併上 score_stock、calc_trade_levels、
    classify_stock 的輸出與 price_series。
    """
    run_date = run_date or as_of
    enter, exit_ = _patched(store, as_of, run_date)
    enter()
    try:
        raw, price_series = sa.get_stock_data(stock_id)
    finally:
        exit_()

    if raw.get("error"):
        return {**raw, "price_series": price_series}

    levels = sa.calc_trade_levels(raw.get("close"), raw.get("atr"))
    fs = {**raw, **sa.score_stock(raw), **levels}
    fs["bucket"] = sa.classify_stock(fs)
    fs["price_series"] = price_series
    return fs


# ══════════════════════════════════════════════════════════════
# 一致性測試：比對 data/signals_log.csv 的實盤紀錄
# ══════════════════════════════════════════════════════════════

# (欄位名稱, signals_log 欄位, 由回測結果取值的函式, 容許誤差；None＝需完全相等)
PARITY_FIELDS = [
    ("K",                 "K",                 lambda f: f.get("K"),                          0.01),
    ("D",                 "D",                 lambda f: f.get("D"),                          0.01),
    ("above_ma5",         "above_ma5",         lambda f: f.get("above_ma5"),                  None),
    ("above_ma20",        "above_ma20",        lambda f: f.get("above_ma20"),                 None),
    ("ma5_dev_pct",       "ma5_dev_pct",       lambda f: sa._dev_pct(f.get("close"), f.get("ma5")),  0.01),
    ("ma20_dev_pct",      "ma20_dev_pct",      lambda f: sa._dev_pct(f.get("close"), f.get("ma20")), 0.01),
    ("foreign_consec",    "foreign_consec",    lambda f: f.get("foreign_consec"),             None),
    ("trust_consec",      "trust_consec",      lambda f: f.get("trust_consec"),               None),
    ("fundamental_score", "fundamental_score", lambda f: f.get("fundamental_score"),          None),
    ("chip_score",        "chip_score",        lambda f: f.get("chip_score"),                 None),
    ("technical_score",   "technical_score",   lambda f: f.get("technical_score"),            None),
    ("total_score",       "total_score",       lambda f: f.get("total_score"),                None),
    # 以下為輔助欄位：主要欄位不符時用來判斷原因
    ("close",             "close",             lambda f: f.get("close"),                      0.001),
    ("price_chg_pct",     "price_chg_pct",     lambda f: f.get("price_chg_pct"),              0.01),
    ("volume_ratio",      "volume_ratio",      lambda f: f.get("volume_ratio"),               0.01),
    ("atr",               "atr",               lambda f: f.get("atr"),                        0.01),
    ("foreign_net_lots",  "foreign_net_lots",  lambda f: f.get("foreign_net"),                None),
    ("trust_net_lots",    "trust_net_lots",    lambda f: f.get("trust_net"),                  None),
    ("stop_loss",         "stop_loss",         lambda f: sa._safe_float(f.get("stop_loss")),  0.001),
    ("target1",           "target1",           lambda f: sa._safe_float(f.get("target1")),    0.001),
    ("target2",           "target2",           lambda f: sa._safe_float(f.get("target2")),    0.001),
    ("bucket",            "bucket",            lambda f: f.get("bucket"),                     None),
]


def _norm(v):
    """CSV 讀回來的值與回測值統一型別後再比較"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if isinstance(v, bool):
        return v
    try:
        return float(s)
    except ValueError:
        return s


def _match(live, bt, tol) -> bool:
    a, b = _norm(live), _norm(bt)
    if a is None or b is None:
        return a is None and b is None
    if tol is not None and isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= tol + 1e-9
    return a == b


def run_parity(log_path: str = sa.SIGNALS_LOG_PATH) -> pd.DataFrame:
    df_log = sa.read_signals_log(log_path)
    store = HistoryStore()
    rows = []
    for _, r in df_log.iterrows():
        sid = str(r["stock_id"])
        data_date = str(r["data_date"])
        run_date = str(r["generated_at"])[:10]  # 實盤 datetime.today()（UTC）
        fs = compute_signal(store, sid, data_date, run_date)
        bt_date = (str(fs["price_series"]["date"].iloc[-1])
                   if not fs["price_series"].empty else None)
        for name, col, getter, tol in PARITY_FIELDS:
            live, bt = r[col], getter(fs)
            rows.append({
                "data_date": data_date, "stock_id": sid, "name": r["name"],
                "field": name, "live": _norm(live), "backtest": _norm(bt),
                "match": _match(live, bt, tol),
                "bt_last_price_date": bt_date,
            })
    out = pd.DataFrame(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out.to_csv(os.path.join(OUTPUT_DIR, "parity_check.csv"), index=False)
    return out


# ══════════════════════════════════════════════════════════════
# 回測：訊號序列、閘門、交易模擬
# ══════════════════════════════════════════════════════════════

FEE_RATE = 0.001425   # 手續費（買、賣各一次），依券商折扣調整
TAX_RATE = 0.003      # 證交稅（賣出時）
HOLD_DAYS = 20        # 進場日為 T+0，最晚 T+19 收盤出場
SCENARIOS = ("stop_first", "target_first")
SIGNAL_BUCKET = "推薦"


def adjusted_prices(store: HistoryStore, stock_id: str, end: str) -> pd.DataFrame:
    """end 當天以前的完整股價，套用 end 以前所有除權息事件（sa.apply_price_adjustment）。

    用來算持有期間的損益：以還原價計算的報酬已包含股利（等同股利再投入同
    一檔）。訊號日的真實報價價位（限價、停損、目標）用 _scale 換算到這個
    基準，作法同 signals_log 的 _local_scale_factor。
    """
    px = store.query("TaiwanStockPrice", stock_id, "1900-01-01", end).sort_values("date")
    for col in ["close", "max", "min", "open", "Trading_Volume"]:
        px[col] = pd.to_numeric(px[col], errors="coerce")
    # 價格為 0 的列（例：3037 2022-02-22 全為 0、成交量 0，疑似暫停交易）視為
    # 非交易日，只影響交易模擬；訊號端仍走實盤 get_stock_data，原樣保留。
    px = px[(px[["open", "max", "min", "close"]] > 0).all(axis=1)]
    div = store.query("TaiwanStockDividendResult", stock_id, "1900-01-01", end).copy()
    if not div.empty:
        for col in ["before_price", "after_price"]:
            div[col] = pd.to_numeric(div[col], errors="coerce")
        div = div.dropna(subset=["before_price", "after_price"])
        div = div[div["before_price"] > 0].sort_values("date")
    adj = sa.apply_price_adjustment(px.reset_index(drop=True), div)
    adj["raw_close"] = px["close"].values
    return adj.reset_index(drop=True)


def build_signals(store: HistoryStore, stock_ids, start: str, end: str) -> pd.DataFrame:
    """逐檔逐日呼叫 compute_signal，得到 [start, end] 每個交易日收盤後的實盤輸出"""
    rows = []
    for sid in stock_ids:
        dates = store.query("TaiwanStockPrice", sid, start, end)["date"].sort_values()
        for d in dates:
            fs = compute_signal(store, sid, d)
            if fs.get("error"):
                continue
            rows.append({
                "date": d, "stock_id": sid,
                "total_score": fs["total_score"],
                "fundamental_score": fs["fundamental_score"],
                "chip_score": fs["chip_score"],
                "technical_score": fs["technical_score"],
                "bucket": fs["bucket"],
                "close": fs["close"], "atr": fs["atr"],
                "stop_loss": sa._safe_float(fs["stop_loss"]),
                "target1": sa._safe_float(fs["target1"]),
                "target2": sa._safe_float(fs["target2"]),
                "data_warning": ",".join(sorted({w.get("type", "") for w in fs["data_warning"]})),
            })
        print(f"  signals {sid}: {len(dates)} days", file=sys.stderr)
    return pd.DataFrame(rows)


def build_gates(store: HistoryStore, dates) -> dict:
    """每個訊號日當晚的夜盤閘門狀態（green/yellow/red）；夜盤開始前為 None"""
    fut = store._load(FUTURES_DATASET, FUTURES_ID)
    night_dates = sorted(fut.loc[fut["trading_session"] == "after_market", "date"].unique())
    first_night = night_dates[0]
    out = {}
    for d in dates:
        nxt = [x for x in night_dates if x > d][:1]
        if d < first_night or not nxt:
            out[d] = None
            continue
        g = compute_gate(store, d, nxt[0])
        out[d] = None if g.get("error") else g["status"]
    return out


def simulate_trade(adj: pd.DataFrame, idx: int, sig: dict, scenario: str) -> dict:
    """模擬一筆訊號的進出場（單位部位＝1），回傳進場、分批出場與報酬。

    - 進場：T+0＝訊號日下一個交易日，限價＝訊號日收盤；最低價 ≤ 限價才成交，
      成交價 = min(開盤, 限價)。
    - T+0：只檢查停損（成交前後順序無法得知，觸及停損即標 ambiguous；
      stop_first 情境以 min(停損, 成交價) 出場，target_first 情境視為未觸發）。
      T+0 不檢查目標價（無法得知最高價發生在成交前或後）。
    - T+1～T+19：開盤跳空穿過停損／目標以開盤價成交；盤中觸及停損以停損價、
      觸及目標以目標價成交；目標1 出一半，停損維持原價；目標2 全出。同日同時
      觸及停損與目標 → ambiguous，依 scenario 決定先後。
    - T+19 收盤仍有部位 → 收盤價出場。
    - 資料不足 T+19：status="open"，exits 只含已發生的部分。
    """
    n = len(adj)
    e = idx + 1
    base = {"stock_id": sig["stock_id"], "signal_date": sig["date"], "scenario": scenario,
            "total_score": sig["total_score"], "entry_date": None, "entry_px": None,
            "exits": [], "ambiguous": False, "exit_date": None, "exit_reason": None,
            "hold_days": None, "ret_gross": None, "ret_net": None}
    if e >= n:
        return {**base, "status": "pending"}
    s = adj.at[idx, "close"] / sig["close"]
    lim = sig["close"] * s
    stop, t1, t2 = (sig[k] * s if sig[k] is not None else None
                    for k in ("stop_loss", "target1", "target2"))
    base["entry_date"] = adj.at[e, "date"]
    if not adj.at[e, "min"] <= lim:
        return {**base, "status": "no_fill"}
    fill = min(adj.at[e, "open"], lim)
    base["entry_px"] = fill

    exits, rem, t1_done, amb = [], 1.0, False, False

    def out(i, frac, px, reason):
        nonlocal rem
        exits.append((adj.at[i, "date"], frac, px, reason, i))
        rem = round(rem - frac, 10)

    if adj.at[e, "min"] <= stop:
        amb = True
        if scenario == "stop_first":
            out(e, rem, min(stop, fill), "stop_T0")

    last = min(e + HOLD_DAYS - 1, n - 1)
    for i in range(e + 1, last + 1):
        if rem <= 0:
            break
        o, h, l = adj.at[i, "open"], adj.at[i, "max"], adj.at[i, "min"]
        if o <= stop:
            out(i, rem, o, "stop_gap"); break
        if o >= t2:
            out(i, rem, o, "t2_gap"); break
        if not t1_done and o >= t1:
            out(i, 0.5, o, "t1_gap"); t1_done = True
        hit_s = l <= stop
        hit_t1 = (not t1_done) and h >= t1
        hit_t2 = h >= t2
        if hit_s and (hit_t1 or hit_t2):
            amb = True
            if scenario == "stop_first":
                out(i, rem, stop, "stop"); break
            if hit_t1:
                out(i, 0.5, t1, "t1"); t1_done = True
            if hit_t2:
                out(i, rem, t2, "t2")
            else:
                out(i, rem, stop, "stop")
            break
        if hit_s:
            out(i, rem, stop, "stop"); break
        if hit_t1:
            out(i, 0.5, t1, "t1"); t1_done = True
        if hit_t2:
            out(i, rem, t2, "t2"); break

    complete = rem <= 0 or e + HOLD_DAYS - 1 <= n - 1
    if rem > 0 and e + HOLD_DAYS - 1 <= n - 1:
        out(e + HOLD_DAYS - 1, rem, adj.at[e + HOLD_DAYS - 1, "close"], "time")

    base.update(exits=exits, ambiguous=amb, window_complete=e + HOLD_DAYS - 1 <= n - 1)
    if not complete:
        return {**base, "status": "open"}
    proceeds = sum(f * px for _, f, px, _, _ in exits)
    base.update(
        status="closed",
        exit_date=exits[-1][0],
        exit_reason=exits[-1][3],
        hold_days=exits[-1][4] - e,
        ret_gross=proceeds / fill - 1,
        ret_net=proceeds * (1 - FEE_RATE - TAX_RATE) / (fill * (1 + FEE_RATE)) - 1,
    )
    return base


def build_outcomes(signals: pd.DataFrame, adj_by_sid: dict, buckets=(SIGNAL_BUCKET,)) -> dict:
    """每筆訊號 × 兩種 ambiguous 情境的交易結果；key = (stock_id, date, scenario)。
    buckets 預設只算推薦；分數分析另外把觀望也算進來當參考。"""
    out = {}
    rec = signals[signals["bucket"].isin(buckets)]
    for sid, grp in rec.groupby("stock_id"):
        adj = adj_by_sid[sid]
        pos = {d: i for i, d in enumerate(adj["date"])}
        for sig in grp.to_dict("records"):
            if sig["date"] not in pos:
                print(f"  略過 {sid} {sig['date']}：訊號日不是有效交易日", file=sys.stderr)
                continue
            for sc in SCENARIOS:
                out[(sid, sig["date"], sc)] = simulate_trade(adj, pos[sig["date"]], sig, sc)
    # 兩情境任一出現 ambiguous，整筆交易都標 ambiguous
    for (sid, d, sc), tr in out.items():
        other = out[(sid, d, "target_first" if sc == "stop_first" else "stop_first")]
        tr["ambiguous_any"] = tr["ambiguous"] or other["ambiguous"]
    return out


# ── A. 單筆交易層：每檔同時最多一筆部位，不受資金限制 ─────────────

def trade_sequence(signals: pd.DataFrame, outcomes: dict, scenario: str,
                   gates: "dict | None" = None) -> "tuple[pd.DataFrame, dict]":
    """依時間逐檔走訊號：持有中（訊號日 < 前一筆出場日）的新訊號略過。
    gates 不為 None 時，夜盤紅燈當晚不下單（沒有閘門資料的日期不過濾）。"""
    trades, cnt = [], {"signals": 0, "skipped_holding": 0, "skipped_gate": 0,
                       "orders": 0, "no_fill": 0, "open_at_end": 0}
    rec = signals[signals["bucket"] == SIGNAL_BUCKET].sort_values("date")
    for sid, grp in rec.groupby("stock_id"):
        busy_until = ""
        for d in grp["date"]:
            cnt["signals"] += 1
            if d < busy_until:
                cnt["skipped_holding"] += 1
                continue
            if gates is not None and gates.get(d) == "red":
                cnt["skipped_gate"] += 1
                continue
            tr = outcomes.get((sid, d, scenario))
            if tr is None:
                continue
            if tr["status"] == "pending":
                continue
            cnt["orders"] += 1
            if tr["status"] == "no_fill":
                cnt["no_fill"] += 1
                continue
            # 資料末端不足 T+19 的交易一律不計（即使已提早出場），避免只留下
            # 「很快停損／停利」的那些，造成期末選擇偏誤
            if tr["status"] == "open" or not tr["window_complete"]:
                cnt["open_at_end"] += 1
                busy_until = "9999-12-31"
                continue
            trades.append(tr)
            busy_until = tr["exit_date"]
    return pd.DataFrame(trades), cnt


def trade_stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0}
    r = t["ret_net"]
    wins, losses = r[r > 0], r[r <= 0]
    return {
        "trades": len(t),
        "win_rate": len(wins) / len(t),
        "avg_win": wins.mean() if len(wins) else 0.0,
        "avg_loss": losses.mean() if len(losses) else 0.0,
        "profit_factor": wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf"),
        "expectancy": r.mean(),
        "median": r.median(),
        "avg_hold_days": t["hold_days"].mean(),
        "ambiguous": int(t["ambiguous_any"].sum()),
    }


# ── B. 投資組合層：有限資金、最多同時 N 檔、允許零股 ──────────────

def simulate_portfolio(signals: pd.DataFrame, outcomes: dict, adj_by_sid: dict,
                       n_slots: int, scenario: str, gates: "dict | None" = None,
                       capital: float = 1_000_000, tiebreak_seed: "int | None" = None) -> dict:
    """每日：先處理當日進場（前一晚的掛單）→ 當日出場 → 收盤市值 → 當晚下單。
    下單：當晚推薦訊號中，排除已持有或已掛單的股票，依 total_score 由高到低、
    stock_id 由小到大（tiebreak_seed 不為 None 時，同分改為隨機順序，用來
    檢查結果對排序的敏感度），取空位數量；每筆預留 min(權益/N, 可用現金)，未成交
    則隔日釋放。股數不限整張（零股）。"""
    calendar = sorted(set().union(*[set(a["date"]) for a in adj_by_sid.values()]))
    calendar = [d for d in calendar if d >= signals["date"].min()]
    close_by = {sid: dict(zip(a["date"], a["close"])) for sid, a in adj_by_sid.items()}
    rec = signals[signals["bucket"] == SIGNAL_BUCKET]
    if tiebreak_seed is not None:
        rec = rec.assign(_tb=np.random.default_rng(tiebreak_seed).random(len(rec)))
    else:
        rec = rec.assign(_tb=rec["stock_id"].astype(int))
    sig_by_date = {d: g.sort_values(["total_score", "_tb"], ascending=[False, True])
                   for d, g in rec.groupby("date")}

    cash, reserved = capital, 0.0
    positions, pending, last_close = {}, {}, {}
    curve, closed = [], []
    for t in calendar:
        for sid, od in list(pending.items()):
            tr = od["trade"]
            if tr["entry_date"] != t:
                continue
            del pending[sid]
            reserved -= od["alloc"]
            if tr["status"] == "no_fill":
                continue
            shares = od["alloc"] / (tr["entry_px"] * (1 + FEE_RATE))
            cash -= od["alloc"]
            positions[sid] = {"shares": shares, "rem": 1.0, "trade": tr, "cost": od["alloc"],
                              "proceeds": 0.0}
        for sid, pos in list(positions.items()):
            for d, frac, px, _, _ in pos["trade"]["exits"]:
                if d == t:
                    amt = pos["shares"] * frac * px * (1 - FEE_RATE - TAX_RATE)
                    cash += amt
                    pos["proceeds"] += amt
                    pos["rem"] = round(pos["rem"] - frac, 10)
            if pos["rem"] <= 0:
                closed.append({"stock_id": sid, "signal_date": pos["trade"]["signal_date"],
                               "ret_net": pos["proceeds"] / pos["cost"] - 1})
                del positions[sid]
        for sid, cb in close_by.items():
            if t in cb:
                last_close[sid] = cb[t]
        equity = cash + sum(p["shares"] * p["rem"] * last_close[s] for s, p in positions.items())
        curve.append((t, equity, len(positions)))

        if t not in sig_by_date or (gates is not None and gates.get(t) == "red"):
            continue
        free = n_slots - len(positions) - len(pending)
        for sig in sig_by_date[t].to_dict("records"):
            if free <= 0:
                break
            sid = sig["stock_id"]
            if sid in positions or sid in pending:
                continue
            tr = outcomes.get((sid, t, scenario))
            if tr is None:
                continue
            if tr["status"] == "pending":
                continue
            alloc = min(equity / n_slots, cash - reserved)
            if alloc <= 1:
                break
            pending[sid] = {"alloc": alloc, "trade": tr}
            reserved += alloc
            free -= 1

    eq = pd.DataFrame(curve, columns=["date", "equity", "n_pos"]).set_index("date")
    ct = pd.DataFrame(closed)
    years = (pd.Timestamp(eq.index[-1]) - pd.Timestamp(eq.index[0])).days / 365.25
    dd = eq["equity"] / eq["equity"].cummax() - 1
    r = ct["ret_net"] if not ct.empty else pd.Series(dtype=float)
    return {
        "total_return": eq["equity"].iloc[-1] / capital - 1,
        "cagr": (eq["equity"].iloc[-1] / capital) ** (1 / years) - 1 if years > 0 else None,
        "max_drawdown": dd.min(),
        "mdd_date": dd.idxmin(),
        "trades": len(ct),
        "win_rate": (r > 0).mean() if len(r) else None,
        "profit_factor": r[r > 0].sum() / -r[r <= 0].sum() if (r <= 0).any() else float("inf"),
        "avg_slots_used": eq["n_pos"].mean() / n_slots,
        "curve": eq,
    }


def buy_and_hold(adj_by_sid: dict, start: str) -> dict:
    """同期等權重買進持有（還原價、不含成本；只含期初已上市者），僅供對照選股偏誤"""
    series = []
    for a in adj_by_sid.values():
        s = a.set_index("date")["close"]
        s = s[s.index >= start]
        # 只納入期初就已上市的股票，避免中途加入造成隱性再平衡
        if len(s) and s.index[0] <= (pd.Timestamp(start) + pd.Timedelta(days=10)).strftime("%Y-%m-%d"):
            series.append(s / s.iloc[0])
    eq = pd.concat(series, axis=1).sort_index().ffill().mean(axis=1)
    dd = eq / eq.cummax() - 1
    years = (pd.Timestamp(eq.index[-1]) - pd.Timestamp(eq.index[0])).days / 365.25
    return {"total_return": eq.iloc[-1] - 1, "cagr": eq.iloc[-1] ** (1 / years) - 1,
            "max_drawdown": dd.min()}


# ══════════════════════════════════════════════════════════════
# 資料檢查：spread 欄位交叉比對未還原事件
# ══════════════════════════════════════════════════════════════

def spread_check(store: HistoryStore, stock_ids, tol: float = 0.002) -> pd.DataFrame:
    """TaiwanStockPrice 的 spread＝收盤 − 當日參考價。參考價 ≠ 前一日收盤
    即代表當天有「參考價調整事件」（除權息、減資、分割、面額變更等）。
    列出所有這類事件，標註是否在 TaiwanStockDividendResult 裡（有就會被
    apply_price_adjustment 還原；沒有就不會）。"""
    rows = []
    for sid in stock_ids:
        px = store.query("TaiwanStockPrice", sid, "1900-01-01", "9999-12-31").sort_values("date")
        px = px[pd.to_numeric(px["close"], errors="coerce") > 0].reset_index(drop=True)
        close = pd.to_numeric(px["close"], errors="coerce")
        spread = pd.to_numeric(px["spread"], errors="coerce")
        prev = close.shift(1)
        ref = close - spread
        ratio = ref / prev
        div_dates = set(store.query("TaiwanStockDividendResult", sid, "1900-01-01", "9999-12-31")["date"])
        for i in px.index[1:]:
            if pd.isna(ratio[i]) or abs(ratio[i] - 1) <= tol:
                continue
            rows.append({
                "stock_id": sid, "date": px.at[i, "date"],
                "ref_vs_prev_pct": round((ratio[i] - 1) * 100, 2),
                "raw_chg_pct": round((close[i] / prev[i] - 1) * 100, 2),
                "in_dividend_result": px.at[i, "date"] in div_dates,
            })
    return pd.DataFrame(rows)


def _fmt_pct(x):
    return "—" if x is None or pd.isna(x) else f"{x * 100:.2f}%"


def _pctiles(vals, qs=(10, 50, 90)):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    return [np.percentile(v, q) if len(v) else np.nan for q in qs]


def portfolio_distribution(signals, outcomes, adj_by_sid, n_slots, scenario, gates, seeds):
    """同分訊號隨機排序 × seeds 次，回傳各指標的 10／50／90 百分位"""
    runs = [simulate_portfolio(signals, outcomes, adj_by_sid, n_slots, scenario, gates,
                               tiebreak_seed=k) for k in range(seeds)]
    return {m: _pctiles([r[m] for r in runs])
            for m in ("total_return", "cagr", "max_drawdown", "win_rate", "profit_factor", "trades")}


def score_table(signals: pd.DataFrame, outcomes: dict, scenario: str = "stop_first") -> pd.DataFrame:
    """依 total_score 統計「每一個訊號各自獨立進場」的結果（訊號之間會重疊，
    同一段行情可能被連續幾天的訊號重複計入）"""
    rows = []
    for (sid, d, sc), tr in outcomes.items():
        if sc != scenario:
            continue
        rows.append({"total_score": tr["total_score"], "status": tr["status"],
                     "window_complete": tr.get("window_complete", True), "ret_net": tr["ret_net"]})
    df = pd.DataFrame(rows)
    out = []
    for score, g in df.groupby("total_score"):
        decided = g[g["status"].isin(["no_fill", "closed", "open"])]
        closed = g[(g["status"] == "closed") & g["window_complete"]]
        r = closed["ret_net"]
        out.append({
            "total_score": score, "signals": len(g),
            "fill_rate": (decided["status"] != "no_fill").mean() if len(decided) else np.nan,
            "trades": len(closed), "win_rate": (r > 0).mean(), "avg_ret": r.mean(),
            "profit_factor": r[r > 0].sum() / -r[r <= 0].sum() if (r <= 0).any() else np.inf,
        })
    return pd.DataFrame(out)


def seq_score_table(trades: pd.DataFrame, cnt_by_score: dict, seed: int = 0) -> pd.DataFrame:
    """單筆交易層（不重疊）依 total_score 統計，附平均報酬的 bootstrap 95% 信賴區間"""
    rng = np.random.default_rng(seed)
    out = []
    for score, g in trades.groupby("total_score"):
        r = g["ret_net"].to_numpy()
        boots = [rng.choice(r, len(r)).mean() for _ in range(2000)]
        c = cnt_by_score.get(score, {})
        out.append({
            "total_score": score, "orders": c.get("orders", 0),
            "fill_rate": 1 - c.get("no_fill", 0) / c["orders"] if c.get("orders") else np.nan,
            "trades": len(r), "win_rate": (r > 0).mean(), "avg_ret": r.mean(),
            "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5),
            "profit_factor": r[r > 0].sum() / -r[r <= 0].sum() if (r <= 0).any() else np.inf,
        })
    return pd.DataFrame(out)


def orders_by_score(signals, outcomes, scenario, gates=None) -> dict:
    """重跑 trade_sequence 的下單邏輯，依分數統計下單數與未成交數"""
    cnt = {}
    rec = signals[signals["bucket"] == SIGNAL_BUCKET].sort_values("date")
    for sid, grp in rec.groupby("stock_id"):
        busy_until = ""
        for sig in grp.to_dict("records"):
            d = sig["date"]
            if d < busy_until or (gates is not None and gates.get(d) == "red"):
                continue
            tr = outcomes.get((sid, d, scenario))
            if tr is None:
                continue
            if tr["status"] == "pending":
                continue
            if tr["status"] == "open" or not tr.get("window_complete", True):
                busy_until = "9999-12-31"
                continue
            c = cnt.setdefault(sig["total_score"], {"orders": 0, "no_fill": 0})
            c["orders"] += 1
            if tr["status"] == "no_fill":
                c["no_fill"] += 1
                continue
            busy_until = tr["exit_date"]
    return cnt


def _p(x, digits=2):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.{digits}f}%"


def _f(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.2f}"


def run_backtest(start: str, end: str, seeds: int = PORTFOLIO_SEEDS) -> None:
    if end >= VALIDATION_START:
        raise SystemExit(f"end 必須早於驗證期起點 {VALIDATION_START}")
    store = HistoryStore()
    sids = list(sa.STOCKS)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tag = f"{start}_{end}"

    sig_path = os.path.join(OUTPUT_DIR, f"signals_{tag}.pkl")
    if os.path.exists(sig_path):
        signals = pd.read_pickle(sig_path)
    else:
        signals = build_signals(store, sids, start, end)
        signals.to_pickle(sig_path)
    gate_path = os.path.join(OUTPUT_DIR, f"gates_{tag}.pkl")
    if os.path.exists(gate_path):
        gates = pd.read_pickle(gate_path)
    else:
        gates = build_gates(store, sorted(signals["date"].unique()))
        pd.to_pickle(gates, gate_path)

    n_before = len(signals)
    for sid, cutoff in EXCLUDE_SIGNALS_BEFORE.items():
        signals = signals[~((signals["stock_id"] == sid) & (signals["date"] < cutoff))]
    excluded = n_before - len(signals)

    adj_by_sid = {sid: adjusted_prices(store, sid, end) for sid in sids}
    outcomes = build_outcomes(signals, adj_by_sid)
    outcomes_ref = build_outcomes(signals, adj_by_sid, buckets=(SIGNAL_BUCKET, "觀望"))

    P = print
    P(f"# 回測結果 {start} ～ {end}（驗證期 {VALIDATION_START} 起未使用）\n")
    P(f"排除 6669 資料不可信期間的股日：{excluded}")
    P("分類股日數：", signals["bucket"].value_counts().to_dict())
    gs = pd.Series({d: gates.get(d) for d in signals["date"].unique()})
    P("夜盤閘門（訊號日）：", gs.value_counts(dropna=False).to_dict(), "\n")

    # ── A. 單筆交易層 ──
    P("## A. 單筆交易層（每檔同時一筆、不限資金、已扣成本）\n")
    P("| 期間 | 版本 | 情境 | 推薦訊號 | 持有中略過 | 閘門略過 | 下單 | 成交率 | 交易 | 勝率 | 平均賺 | 平均賠 | 獲利因子 | 期望值 | 中位數 | 平均持有 | ambiguous | 排除ambiguous期望值 |")
    P("|" + "---|" * 18)
    all_trades = []
    periods = [("全期", signals, None), (f"{GATE_START}起", signals[signals["date"] >= GATE_START], None),
               (f"{GATE_START}起", signals[signals["date"] >= GATE_START], gates)]
    for label, sg, gmap in periods:
        for sc in SCENARIOS:
            t, cnt = trade_sequence(sg, outcomes, sc, gmap)
            st = trade_stats(t)
            nt = trade_stats(t[~t["ambiguous_any"]])
            fr = (cnt["orders"] - cnt["no_fill"]) / cnt["orders"]
            P(f"| {label} | {'有閘門' if gmap else '無閘門'} | {sc} | {cnt['signals']} | {cnt['skipped_holding']} | "
              f"{cnt['skipped_gate']} | {cnt['orders']} | {_p(fr,1)} | {st['trades']} | {_p(st['win_rate'],1)} | "
              f"{_p(st['avg_win'])} | {_p(st['avg_loss'])} | {_f(st['profit_factor'])} | {_p(st['expectancy'])} | "
              f"{_p(st['median'])} | {st['avg_hold_days']:.1f} | {st['ambiguous']} | {_p(nt['expectancy'])} |")
            all_trades.append(t.drop(columns=["exits"]).assign(period=label, gate=bool(gmap)))
    trades_all = pd.concat(all_trades)
    trades_all.to_csv(os.path.join(OUTPUT_DIR, f"trades_{tag}.csv"), index=False)

    base = trades_all[(trades_all["period"] == "全期") & (trades_all["scenario"] == "stop_first")]
    P("\n出場原因（全期、無閘門、stop_first）：", base["exit_reason"].value_counts().to_dict())
    P("\n### 逐年（全期、無閘門、stop_first）\n")
    P("| 年 | 交易 | 勝率 | 期望值 | 獲利因子 |")
    P("|---|---|---|---|---|")
    for y, g in base.groupby(base["signal_date"].str[:4]):
        st = trade_stats(g)
        P(f"| {y} | {st['trades']} | {_p(st['win_rate'],1)} | {_p(st['expectancy'])} | {_f(st['profit_factor'])} |")
    P("\n### 逐檔（全期、無閘門、stop_first）\n")
    P("| 股票 | 交易 | 勝率 | 期望值 | 獲利因子 |")
    P("|---|---|---|---|---|")
    for sid, g in base.groupby("stock_id"):
        st = trade_stats(g)
        P(f"| {sid} {sa.STOCKS[sid]} | {st['trades']} | {_p(st['win_rate'],1)} | {_p(st['expectancy'])} | {_f(st['profit_factor'])} |")

    # ── 分數分析 ──
    P("\n## 依 total_score（全期、無閘門）\n")
    for sc in SCENARIOS:
        P(f"### (1) 單筆交易層（不重疊，{sc}）\n")
        P("| 分數 | 下單 | 成交率 | 交易 | 勝率 | 平均報酬 | 95% CI | 獲利因子 |")
        P("|---|---|---|---|---|---|---|---|")
        tt = base if sc == "stop_first" else trades_all[(trades_all["period"] == "全期") & (trades_all["scenario"] == sc)]
        for r in seq_score_table(tt, orders_by_score(signals, outcomes, sc)).itertuples():
            P(f"| {r.total_score} | {r.orders} | {_p(r.fill_rate,1)} | {r.trades} | {_p(r.win_rate,1)} | "
              f"{_p(r.avg_ret)} | {_p(r.ci_low)}～{_p(r.ci_high)} | {_f(r.profit_factor)} |")
        P(f"\n### (2) 每個訊號各自獨立進場（會重疊；5–6 分為觀望，僅供參考，{sc}）\n")
        P("| 分數 | 訊號 | 成交率 | 交易 | 勝率 | 平均報酬 | 獲利因子 |")
        P("|---|---|---|---|---|---|---|")
        for r in score_table(signals, outcomes_ref, sc).itertuples():
            P(f"| {r.total_score} | {r.signals} | {_p(r.fill_rate,1)} | {r.trades} | {_p(r.win_rate,1)} | "
              f"{_p(r.avg_ret)} | {_f(r.profit_factor)} |")
        P()

    # ── B. 投資組合層 ──
    P(f"## B. 投資組合層（初始 100 萬、零股、每檔 = 權益/N；同分隨機排序 × {seeds} 種子，10／50／90 百分位）\n")
    P("| 期間 | 版本 | N | 情境 | 總報酬 | 年化 | 最大回撤 | 勝率 | 獲利因子 | 交易數 |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for label, sg, gmap in periods:
        for n in (3, 5):
            for sc in SCENARIOS:
                d = portfolio_distribution(sg, outcomes, adj_by_sid, n, sc, gmap, seeds)
                fmt = lambda k, f: " / ".join(f(x) for x in d[k])
                P(f"| {label} | {'有閘門' if gmap else '無閘門'} | {n} | {sc} | {fmt('total_return', lambda x: _p(x,0))} | "
                  f"{fmt('cagr', lambda x: _p(x,1))} | {fmt('max_drawdown', lambda x: _p(x,1))} | "
                  f"{fmt('win_rate', lambda x: _p(x,1))} | {fmt('profit_factor', _f)} | "
                  f"{fmt('trades', lambda x: f'{x:.0f}')} |")
                sys.stdout.flush()
    for label, st_ in (("全期", start), (f"{GATE_START}起", GATE_START)):
        bh = buy_and_hold(adj_by_sid, st_)
        P(f"\n對照（{label}）等權買進持有（期初已上市者）：總報酬 {_p(bh['total_return'],0)}、年化 {_p(bh['cagr'],1)}、最大回撤 {_p(bh['max_drawdown'],1)}")


# ══════════════════════════════════════════════════════════════
# 切片分析：法人連買天數、分數分層的叢集信賴區間（只用快取，不重跑回測）
# ══════════════════════════════════════════════════════════════

def _consec_bucket(x):
    return "≥3" if x >= 3 else str(int(x))


def _cluster_ci(values, clusters, n_boot=2000, seed=0):
    """以 (stock_id, 年月) 為叢集的 bootstrap 95% 信賴區間：同一檔同一個月
    的重疊訊號一起抽，避免把連續幾天的同一段行情當成獨立樣本"""
    df = pd.DataFrame({"v": values, "c": clusters}).dropna()
    if df.empty:
        return np.nan, np.nan
    g = df.groupby("c")["v"].agg(["sum", "count"])
    sums, cnts = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def run_slices(start: str, end: str) -> None:
    store = HistoryStore()
    signals = pd.read_pickle(os.path.join(OUTPUT_DIR, f"signals_{start}_{end}.pkl"))
    for sid, cutoff in EXCLUDE_SIGNALS_BEFORE.items():
        signals = signals[~((signals["stock_id"] == sid) & (signals["date"] < cutoff))]
    adj_by_sid = {sid: adjusted_prices(store, sid, end) for sid in sa.STOCKS}
    outcomes = build_outcomes(signals, adj_by_sid, buckets=(SIGNAL_BUCKET, "觀望"))

    # 推薦訊號逐筆補上法人連買天數（用實盤 get_stock_data 重建，同一份快取）
    rec = signals[signals["bucket"] == SIGNAL_BUCKET]
    rows = []
    for sig in rec.to_dict("records"):
        tr = outcomes.get((sig["stock_id"], sig["date"], "stop_first"))
        if tr is None or tr["status"] in ("pending", "no_fill"):
            continue
        fs = compute_signal(store, sig["stock_id"], sig["date"])
        adj = adj_by_sid[sig["stock_id"]]
        e = int(adj.index[adj["date"] == tr["entry_date"]][0])

        def ret(k):
            return adj.at[e + k, "close"] / tr["entry_px"] - 1 if e + k < len(adj) else np.nan

        rows.append({
            "stock_id": sig["stock_id"], "date": sig["date"],
            "cluster": f"{sig['stock_id']}-{sig['date'][:7]}",
            "foreign": _consec_bucket(fs["foreign_consec"]),
            "trust": _consec_bucket(fs["trust_consec"]),
            "t0_stop": any(x[3] == "stop_T0" for x in tr["exits"]),
            "ret_t1": ret(1), "ret_t5": ret(5),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, f"slices_{start}_{end}.csv"), index=False)

    P = print
    P(f"# 切片分析 {start} ～ {end}（推薦訊號、已成交、每個訊號各自獨立進場）\n")
    P("報酬定義同 signals_log 的 ret_t1_pct／ret_t5_pct：(T+1 或 T+5 收盤) / 成交價 − 1，"
      "還原價、未扣成本、不考慮中途停損停利。95% CI 為 (股票, 年月) 叢集 bootstrap。\n")
    for who, label in (("foreign", "外資"), ("trust", "投信")):
        P(f"## {label}連買天數\n")
        P("| 連買天數 | 成交筆數 | 進場當日觸及停損 | T+1 平均報酬 | T+1 95% CI | T+5 平均報酬 | T+5 95% CI |")
        P("|---|---|---|---|---|---|---|")
        for b in ("0", "1", "2", "≥3"):
            g = df[df[who] == b]
            if g.empty:
                P(f"| {b} | 0 | — | — | — | — | — |")
                continue
            c1 = _cluster_ci(g["ret_t1"], g["cluster"])
            c5 = _cluster_ci(g["ret_t5"], g["cluster"])
            ct = _cluster_ci(g["t0_stop"].astype(float), g["cluster"])
            P(f"| {b} | {len(g)} | {_p(g['t0_stop'].mean(),1)}（{_p(ct[0],1)}～{_p(ct[1],1)}） | "
              f"{_p(g['ret_t1'].mean())} | {_p(c1[0])}～{_p(c1[1])} | "
              f"{_p(g['ret_t5'].mean())} | {_p(c5[0])}～{_p(c5[1])} |")
        P()

    # 分數分層表（每個訊號各自獨立進場）補上叢集信賴區間
    P("## 分數分層：每個訊號各自獨立進場，改用叢集 bootstrap 95% CI（stop_first）\n")
    P("| 分數 | 交易 | 叢集數 | 平均報酬 | 叢集 95% CI |")
    P("|---|---|---|---|---|")
    rr = [{"score": tr["total_score"], "ret": tr["ret_net"], "cluster": f"{sid}-{d[:7]}"}
          for (sid, d, sc), tr in outcomes.items()
          if sc == "stop_first" and tr["status"] == "closed" and tr.get("window_complete", True)]
    rr = pd.DataFrame(rr)
    for score, g in rr.groupby("score"):
        lo, hi = _cluster_ci(g["ret"], g["cluster"])
        P(f"| {score} | {len(g)} | {g['cluster'].nunique()} | {_p(g['ret'].mean())} | {_p(lo)}～{_p(hi)} |")


def main():
    ap = argparse.ArgumentParser(description="stock_analyzer baseline 回測")
    ap.add_argument("cmd", choices=["fetch", "parity", "spread", "run", "slices"])
    ap.add_argument("--refresh", action="store_true", help="fetch：重新下載已存在的快取")
    ap.add_argument("--start", default="2012-08-01")
    ap.add_argument("--seeds", type=int, default=PORTFOLIO_SEEDS)
    ap.add_argument("--end", default=(datetime.strptime(VALIDATION_START, "%Y-%m-%d")
                                      - timedelta(days=1)).strftime("%Y-%m-%d"))
    args = ap.parse_args()

    if args.cmd == "fetch":
        fetch_history(list(sa.STOCKS), refresh=args.refresh)
    elif args.cmd == "parity":
        out = run_parity()
        summary = out.groupby("field", sort=False)["match"].agg(["sum", "count"])
        print(summary.to_string())
        bad = out[~out["match"]]
        print(f"\n不相符：{len(bad)} / {len(out)} 格")
        if not bad.empty:
            print(bad.to_string(index=False))
    elif args.cmd == "spread":
        df = spread_check(HistoryStore(), list(sa.STOCKS))
        df.to_csv(os.path.join(OUTPUT_DIR, "spread_check.csv"), index=False)
        print(f"參考價調整事件 {len(df)} 筆，其中不在 TaiwanStockDividendResult：")
        print(df[~df["in_dividend_result"]].to_string(index=False))
    elif args.cmd == "run":
        run_backtest(args.start, args.end, args.seeds)
    elif args.cmd == "slices":
        run_slices(args.start, args.end)


if __name__ == "__main__":
    main()
