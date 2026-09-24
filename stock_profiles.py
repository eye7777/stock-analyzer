"""
個股體檢：13 檔股票的股價波動特性與現行系統在各檔的表現（只描述，不找規則）。

資料
- 只用開發期（DEV_START ～ DEV_END）；驗證期（backtest.VALIDATION_START 起）
  不進入任何計算：個股快取一律經 HistoryStore.query(as_of=DEV_END) 截斷，
  新下載的加權指數與美股只下載到 DEV_END。ATR14 與 60 日標準差的暖身期
  會用到 DEV_START 之前（WARMUP_START 起）的股價。
- 6669 在 backtest.EXCLUDE_SIGNALS_BEFORE 之前的股價整段不用（所有面向）。
- 系統適配沿用既有回測輸出（backtest_output/trades_*.csv、signals_*.pkl）。

用法
  python stock_profiles.py fetch    # 下載加權指數與美股日線到 backtest_cache/
  python stock_profiles.py report   # 計算並寫出 docs/stock_profiles_report.md
"""

import os
import sys
import math
import time
import argparse

import numpy as np
import pandas as pd
import requests

import backtest as bt
import stock_analyzer as sa

DEV_START = "2012-08-01"
DEV_END   = "2026-06-22"
WARMUP_START = "2012-01-01"
assert DEV_END < bt.VALIDATION_START

INDEX_DATASET, INDEX_ID = "TaiwanStockPrice", "TAIEX"
US_DATASET = "USStockPrice"
US_IDS = ("^SOX", "NVDA", "^GSPC")
MAX_REQUESTS = 100

SPLIT_DATE = "2024-01-01"      # 系統適配拆成此日前／後（依訊號日）
ATR_N = 14
HIT_HORIZON = 19               # 先碰到哪個價位：D+1 ～ D+19
HIT_UP = (2.0, 3.0)
HIT_DN = 1.5
GAP_MIN = 0.03
Z_WINDOW, Z_TAIL = 60, 0.05
FWD_DAYS = (1, 5, 10)
REV_FWD = 20
DIV_WIN = 5
CASH_DIV_TYPES = {"息", "除息"}
LIMIT_CHANGE_DATE = "2015-06-01"   # 漲跌幅限制 ±7% → ±10%
N_BOOT = 2000

REPORT_PATH = os.path.join(bt._DIR, "docs", "stock_profiles_report.md")
TAG = f"{DEV_START}_{DEV_END}"


# ══════════════════════════════════════════════════════════════
# 下載：加權指數與美股（每檔一次請求，只到 DEV_END）
# ══════════════════════════════════════════════════════════════

def fetch() -> None:
    jobs = [(INDEX_DATASET, INDEX_ID)] + [(US_DATASET, u) for u in US_IDS]
    n = 0
    for ds, sid in jobs:
        path = bt._cache_path(ds, sid)
        if os.path.exists(path):
            continue
        if n >= MAX_REQUESTS:
            raise SystemExit(f"已達請求上限 {MAX_REQUESTS} 次，停止下載")
        resp = requests.get(sa.FINMIND_BASE, params={
            "dataset": ds, "data_id": sid,
            "start_date": WARMUP_START, "end_date": DEV_END,
            "token": sa.FINMIND_TOKEN,
        }, timeout=120)
        n += 1
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != 200:
            raise RuntimeError(f"FinMind {ds} ({sid}): {data.get('msg')}")
        df = pd.DataFrame(data.get("data") or [])
        df.to_pickle(path)
        print(f"  cached {ds} {sid}: {len(df)} rows "
              f"({df['date'].min() if len(df) else '-'} ～ {df['date'].max() if len(df) else '-'})")
        time.sleep(0.5)
    print(f"本次請求 {n} 次")


# ══════════════════════════════════════════════════════════════
# 信賴區間：(股票, 年月) 叢集 bootstrap
# ══════════════════════════════════════════════════════════════

# 報告中「平均值／差值／迴歸係數對零」的 CI 格；比例類指標不列入
TESTED = set()


def _cluster(sid: str, dates) -> np.ndarray:
    return np.array([f"{sid}-{d[:7]}" for d in dates])


def mean_ci(values, clusters, key=None) -> tuple:
    """(平均, 下緣, 上緣)；key 不為 None 時計入多重比較格數"""
    v = pd.Series(np.asarray(values, dtype=float))
    ok = v.notna().to_numpy()
    if ok.sum() == 0:
        return (np.nan, np.nan, np.nan)
    lo, hi = bt._cluster_ci(v[ok].to_numpy(), np.asarray(clusters)[ok], n_boot=N_BOOT)
    if key is not None:
        TESTED.add(key)
    return (v[ok].mean(), lo, hi)


def diff_ci(values, clusters, mask, key=None) -> tuple:
    """子集平均 − 全體平均；兩者在同一次叢集重抽中一起計算"""
    df = pd.DataFrame({"v": np.asarray(values, dtype=float), "c": clusters,
                       "m": np.asarray(mask, dtype=bool)}).dropna(subset=["v"])
    if df["m"].sum() == 0:
        return (np.nan, np.nan, np.nan)
    df["vm"] = df["v"] * df["m"]
    g = df.groupby("c").agg(s=("v", "sum"), n=("v", "count"), sm=("vm", "sum"), nm=("m", "sum"))
    s, n, sm, nm = (g[k].to_numpy(dtype=float) for k in ("s", "n", "sm", "nm"))
    idx = np.random.default_rng(0).integers(0, len(g), size=(N_BOOT, len(g)))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = sm[idx].sum(1) / nm[idx].sum(1) - s[idx].sum(1) / n[idx].sum(1)
    d = d[~np.isnan(d)]
    if key is not None:
        TESTED.add(key)
    point = df.loc[df["m"], "v"].mean() - df["v"].mean()
    return (point, np.percentile(d, 2.5), np.percentile(d, 97.5))


def reg_ci(x, y, clusters, key=None) -> dict:
    """y 對 x 的 OLS 斜率與相關係數；CI 以叢集充分統計量重抽"""
    df = pd.DataFrame({"x": x, "y": y, "c": clusters}).dropna()
    df["xx"], df["yy"], df["xy"] = df["x"] ** 2, df["y"] ** 2, df["x"] * df["y"]
    g = df.groupby("c")[["x", "y", "xx", "yy", "xy"]].sum()
    g["n"] = df.groupby("c").size()
    a = g.to_numpy(dtype=float)

    def stats(t):
        sx, sy, sxx, syy, sxy, n = (t[..., i] for i in range(6))
        cxy, cxx, cyy = sxy - sx * sy / n, sxx - sx ** 2 / n, syy - sy ** 2 / n
        return cxy / cxx, cxy / np.sqrt(cxx * cyy)

    b, r = stats(a.sum(0))
    idx = np.random.default_rng(0).integers(0, len(g), size=(N_BOOT, len(g)))
    bb, rr = stats(a[idx].sum(1))
    if key is not None:
        TESTED.add(key)
    return {"beta": (b, *np.percentile(bb, [2.5, 97.5])),
            "corr": (r, *np.percentile(rr, [2.5, 97.5])), "n": len(df)}


# ══════════════════════════════════════════════════════════════
# 股價：還原價（報酬、ATR、跳空）＋原始價（漲跌停）
# ══════════════════════════════════════════════════════════════

def _tick(p: float) -> float:
    for bound, t in ((10, 0.01), (50, 0.05), (100, 0.1), (500, 0.5), (1000, 1.0)):
        if p < bound:
            return t
    return 5.0


def limit_prices(ref: float, date: str) -> tuple:
    """(漲停價, 跌停價)：參考價 ×(1 ± 限制)，依升降單位向內取整"""
    if not ref > 0:
        return (np.nan, np.nan)
    pct = 0.07 if date < LIMIT_CHANGE_DATE else 0.10
    up, dn = ref * (1 + pct), ref * (1 - pct)
    tu, td = _tick(up), _tick(dn)
    return (math.floor(round(up / tu, 6)) * tu, math.ceil(round(dn / td, 6)) * td)


def load_prices(store: bt.HistoryStore, sid: str) -> pd.DataFrame:
    """WARMUP_START ～ DEV_END 的逐日資料（6669 從 EXCLUDE_SIGNALS_BEFORE 起）"""
    adj = bt.adjusted_prices(store, sid, DEV_END)
    start = max(WARMUP_START, bt.EXCLUDE_SIGNALS_BEFORE.get(sid, WARMUP_START))
    df = adj[adj["date"] >= start].reset_index(drop=True)

    # 原始價：還原因子對 OHLC 一致，原始價 = 還原價 × (原始收盤 / 還原收盤)
    f = df["raw_close"] / df["close"]
    raw = {c: df[c] * f for c in ("open", "max", "min")}
    div = store.query("TaiwanStockDividendResult", sid, "1900-01-01", DEV_END)
    ref_px = dict(zip(div["date"], pd.to_numeric(div["reference_price"], errors="coerce")))
    ref = df["raw_close"].shift(1)
    ref = pd.Series([ref_px.get(d, r) for d, r in zip(df["date"], ref)], index=df.index)
    lim = [limit_prices(r, d) for r, d in zip(ref, df["date"])]
    df["limit_up"] = [x[0] for x in lim]
    df["limit_dn"] = [x[1] for x in lim]
    eps = 1e-6
    has = df["limit_up"].notna()
    df["up_close"] = np.where(has, df["raw_close"] >= df["limit_up"] - eps, np.nan)
    df["up_touch"] = np.where(has, raw["max"] >= df["limit_up"] - eps, np.nan)
    df["dn_close"] = np.where(has, df["raw_close"] <= df["limit_dn"] + eps, np.nan)
    df["dn_touch"] = np.where(has, raw["min"] <= df["limit_dn"] + eps, np.nan)

    prev = df["close"].shift(1)
    df["ret"] = df["close"] / prev - 1
    df["gap"] = df["open"] / prev - 1
    tr = pd.concat([df["max"] - df["min"], (df["max"] - prev).abs(), (df["min"] - prev).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_N).mean()
    df["atr_pct"] = df["atr"] / df["close"]
    df["z"] = df["ret"] / df["ret"].shift(1).rolling(Z_WINDOW).std()
    for k in FWD_DAYS + (REV_FWD,):
        df[f"fwd{k}"] = df["close"].shift(-k) / df["close"] - 1
    df["year"] = df["date"].str[:4]
    df["cluster"] = _cluster(sid, df["date"])
    return df


def load_index(store: bt.HistoryStore) -> pd.Series:
    px = store.query(INDEX_DATASET, INDEX_ID, WARMUP_START, DEV_END)
    return pd.to_numeric(px.set_index("date")["close"], errors="coerce").sort_index()


def load_us(store: bt.HistoryStore, uid: str) -> pd.Series:
    px = store.query(US_DATASET, uid, WARMUP_START, DEV_END)
    return pd.to_numeric(px.set_index("date")["Adj_Close"], errors="coerce").sort_index()


# ══════════════════════════════════════════════════════════════
# 1. 波動特性
# ══════════════════════════════════════════════════════════════

def first_hit(df: pd.DataFrame, up_mult: float) -> pd.Series:
    """從 D 收盤出發，D+1～D+19 先碰到 +up_mult×ATR 或 −1.5×ATR（ATR 取 D 當日）。
    回傳 up／down／both（同一天兩者都碰到）／none；窗口不足或無 ATR 為 NaN"""
    hi, lo = df["max"].to_numpy(), df["min"].to_numpy()
    c, a = df["close"].to_numpy(), df["atr"].to_numpy()
    n, out = len(df), [np.nan] * len(df)
    for i in range(n):
        if i + HIT_HORIZON >= n or not a[i] > 0:
            continue
        h = hi[i + 1:i + HIT_HORIZON + 1] >= c[i] + up_mult * a[i]
        l = lo[i + 1:i + HIT_HORIZON + 1] <= c[i] - HIT_DN * a[i]
        iu = h.argmax() if h.any() else HIT_HORIZON
        il = l.argmax() if l.any() else HIT_HORIZON
        out[i] = ("none" if iu == il == HIT_HORIZON else "both" if iu == il
                  else "up" if iu < il else "down")
    return pd.Series(out, index=df.index)


def volatility(df: pd.DataFrame) -> dict:
    d = df[df["date"] >= DEV_START].copy()
    d["gap_up"] = (d["gap"] >= GAP_MIN).astype(float).where(d["gap"].notna())
    d["gap_dn"] = (d["gap"] <= -GAP_MIN).astype(float).where(d["gap"].notna())
    freq_cols = ["gap_up", "gap_dn", "up_close", "up_touch", "dn_close", "dn_touch"]
    yearly = d.groupby("year").agg(days=("date", "size"), atr_pct=("atr_pct", "median"),
                                   ret_sd=("ret", "std"), **{c: (c, "mean") for c in freq_cols})
    full = {"atr_pct": d["atr_pct"].median(), "ret_sd": d["ret"].std(),
            "atr_pct_post": d.loc[d["date"] >= SPLIT_DATE, "atr_pct"].median(),
            "ret_sd_post": d.loc[d["date"] >= SPLIT_DATE, "ret"].std()}
    for c in freq_cols:
        full[c] = mean_ci(d[c], d["cluster"])
    hits = {}
    for m in HIT_UP:
        fh = first_hit(df, m)[d.index]
        ok = fh.notna()
        hits[m] = {"n": int(ok.sum()),
                   **{k: mean_ci((fh[ok] == k).astype(float), d.loc[ok, "cluster"])
                      for k in ("up", "down", "both", "none")}}
    return {"yearly": yearly, "full": full, "hits": hits}


# ══════════════════════════════════════════════════════════════
# 2. 大漲大跌之後
# ══════════════════════════════════════════════════════════════

def big_moves(sid: str, df: pd.DataFrame) -> dict:
    d = df[(df["date"] >= DEV_START) & df["z"].notna()]
    hi, lo = d["z"].quantile(1 - Z_TAIL), d["z"].quantile(Z_TAIL)
    allday = df[df["date"] >= DEV_START]
    out = {"n_z": len(d), "z_hi": hi, "z_lo": lo,
           "n_top": int((d["z"] >= hi).sum()), "n_bot": int((d["z"] <= lo).sum()), "rows": []}
    for k in FWD_DAYS:
        col = f"fwd{k}"
        top = allday.index.isin(d.index[d["z"] >= hi])
        bot = allday.index.isin(d.index[d["z"] <= lo])
        out["rows"].append({
            "k": k,
            "all": mean_ci(allday[col], allday["cluster"], key=(sid, "big", "all", k)),
            "top": mean_ci(allday.loc[top, col], allday.loc[top, "cluster"], key=(sid, "big", "top", k)),
            "bot": mean_ci(allday.loc[bot, col], allday.loc[bot, "cluster"], key=(sid, "big", "bot", k)),
            "top_diff": diff_ci(allday[col], allday["cluster"], top, key=(sid, "big", "top_d", k)),
            "bot_diff": diff_ci(allday[col], allday["cluster"], bot, key=(sid, "big", "bot_d", k)),
        })
    return out


# ══════════════════════════════════════════════════════════════
# 3. 事件前後：月營收年增率、除權息
# ══════════════════════════════════════════════════════════════

def revenue_events(store: bt.HistoryStore, sid: str, df: pd.DataFrame, idx: pd.Series) -> dict:
    """M 月營收可取得日 = 次月 11 日起第一個交易日（同 HistoryStore 的規則）；
    20 日報酬 = 可取得日收盤 → 20 個交易日後收盤"""
    rev = store.query("TaiwanStockMonthRevenue", sid, "1900-01-01", DEV_END)
    rev = rev.assign(revenue=pd.to_numeric(rev["revenue"], errors="coerce"))
    by_ym = {(int(y), int(m)): v for y, m, v in zip(rev["revenue_year"], rev["revenue_month"], rev["revenue"])}
    dates = df["date"].to_numpy()
    rows = []
    for r in rev.itertuples(index=False):
        prev = by_ym.get((int(r.revenue_year) - 1, int(r.revenue_month)))
        if not (prev and prev > 0 and r.revenue > 0):
            continue
        avail = r.date[:8] + f"{bt.REVENUE_PUBLISH_DAY:02d}"
        i = int(np.searchsorted(dates, avail, side="right"))
        # i == 0：可取得日早於載入的股價起點（例：6669 排除期），不是真的事件日
        if i == 0 or i >= len(df) or df.at[i, "date"] < DEV_START or pd.isna(df.at[i, f"fwd{REV_FWD}"]):
            continue
        d0, d1 = df.at[i, "date"], df.at[i + REV_FWD, "date"]
        ix = idx.get(d1) / idx.get(d0) - 1 if d0 in idx and d1 in idx else np.nan
        ret = df.at[i, f"fwd{REV_FWD}"]
        rows.append({"date": d0, "yoy": r.revenue / prev - 1, "ret": ret, "excess": ret - ix})
    ev = pd.DataFrame(rows)
    ev["grp"] = pd.qcut(ev["yoy"], 3, labels=["低", "中", "高"])
    ev["cluster"] = _cluster(sid, ev["date"])
    out = {"n": len(ev), "groups": []}
    for g in ("低", "中", "高", "全部"):
        e = ev if g == "全部" else ev[ev["grp"] == g]
        out["groups"].append({
            "grp": g, "n": len(e), "yoy_range": (e["yoy"].min(), e["yoy"].max()),
            "ret": mean_ci(e["ret"], e["cluster"], key=(sid, "rev", g, "ret")),
            "excess": mean_ci(e["excess"], e["cluster"], key=(sid, "rev", g, "ex")),
        })
    return out


def dividend_events(store: bt.HistoryStore, sid: str, df: pd.DataFrame) -> dict:
    """現金股利除息日為第 0 天；第 k 天的累積報酬 = 第 −6 天收盤 → 第 k 天收盤（還原價）"""
    start = max(DEV_START, bt.EXCLUDE_SIGNALS_BEFORE.get(sid, DEV_START))
    div = store.query("TaiwanStockDividendResult", sid, start, DEV_END)
    cash = div[div["stock_or_cache_dividend"].isin(CASH_DIV_TYPES)]
    other = div[~div["stock_or_cache_dividend"].isin(CASH_DIV_TYPES)]
    pos = {d: i for i, d in enumerate(df["date"])}
    paths, skipped = [], 0
    for d in cash["date"]:
        i = pos.get(d)
        if i is None or i - DIV_WIN - 1 < 0 or i + DIV_WIN >= len(df):
            skipped += 1
            continue
        base = df.at[i - DIV_WIN - 1, "close"]
        paths.append({"date": d, **{k: df.at[i + k, "close"] / base - 1 for k in range(-DIV_WIN, DIV_WIN + 1)}})
    p = pd.DataFrame(paths)
    cl = _cluster(sid, p["date"]) if len(p) else np.array([])
    return {"n_cash": len(p), "skipped": skipped,
            "other": other["stock_or_cache_dividend"].value_counts().to_dict(),
            "path": {k: mean_ci(p[k], cl, key=(sid, "div", k)) if len(p) else (np.nan,) * 3
                     for k in range(-DIV_WIN, DIV_WIN + 1)}}


# ══════════════════════════════════════════════════════════════
# 4. 連動：加權指數、美股前一晚
# ══════════════════════════════════════════════════════════════

def index_link(sid: str, df: pd.DataFrame, idx: pd.Series) -> dict:
    d = df[df["date"] >= DEV_START].copy()
    # 指數報酬要對應個股前一個有效交易日（個股停牌日時兩者區間不同，改用收盤比）
    prev = df["date"].shift(1)[d.index]
    d["idx_ret"] = [idx.get(a) / idx.get(b) - 1 if a in idx and b in idx else np.nan
                    for a, b in zip(d["date"], prev)]
    yearly = {}
    for y, g in d.groupby("year"):
        x = g[["ret", "idx_ret"]].dropna()
        cxy = np.cov(x["ret"], x["idx_ret"])
        yearly[y] = {"beta": cxy[0, 1] / cxy[1, 1], "corr": x["ret"].corr(x["idx_ret"]), "n": len(x)}
    full = reg_ci(d["idx_ret"], d["ret"], d["cluster"], key=(sid, "beta"))
    return {"yearly": yearly, "full": full}


def us_link(sid: str, df: pd.DataFrame, us: dict) -> dict:
    """台股第 D 日（前一個交易日為 P）對應美股日期 P ≤ u < D 的各場收盤，
    休市期間多場美股報酬連乘；美股這段沒有開盤則不列入"""
    d = df[df["date"] >= DEV_START]
    prev = df["date"].shift(1)[d.index].fillna("").to_numpy()
    out = {}
    for uid, px in us.items():
        lp = np.log(px.dropna())
        udates = lp.index.to_numpy()
        ia = np.searchsorted(udates, d["date"].to_numpy(), side="left") - 1   # 最後一場 u < D
        ib = np.searchsorted(udates, prev, side="left") - 1                   # 最後一場 u < P
        ok = (ia > ib) & (ib >= 0)
        ur = np.where(ok, np.exp(lp.to_numpy()[ia] - lp.to_numpy()[np.maximum(ib, 0)]) - 1, np.nan)
        out[uid] = {m: reg_ci(ur, d[m], d["cluster"], key=(sid, "us", uid, m)) for m in ("ret", "gap")}
    return out


# ══════════════════════════════════════════════════════════════
# 5. 系統適配（既有基準回測輸出）
# ══════════════════════════════════════════════════════════════

def _stop_dist(t: pd.DataFrame, signals: pd.DataFrame, adj_by_sid: dict) -> pd.Series:
    """進場時的停損距離（同 risk_factors.attach_outcomes）：停損價換算到還原價基準，相對成交價"""
    sig = signals.set_index(["stock_id", "date"])[["close", "stop_loss"]]
    out = []
    for r in t.itertuples(index=False):
        s = sig.loc[(r.stock_id, r.signal_date)]
        adj = adj_by_sid[r.stock_id]
        c = adj.loc[adj["date"] == r.signal_date, "close"].iloc[0]
        out.append(1 - s["stop_loss"] * c / s["close"] / r.entry_px)
    return pd.Series(out, index=t.index)


def fit_stats(sid: str, t: pd.DataFrame, table: str) -> dict:
    """t：該檔已完成的交易（ret_net、exit_reason、stop_dist、signal_date）"""
    res = {}
    for label, g in (("全期", t), (f"{SPLIT_DATE[:4]} 前", t[t["signal_date"] < SPLIT_DATE]),
                     (f"{SPLIT_DATE[:4]} 起", t[t["signal_date"] >= SPLIT_DATE])):
        cl = _cluster(sid, g["signal_date"])
        rok = g["stop_dist"] > 0
        res[label] = {
            "n": len(g),
            "win": mean_ci((g["ret_net"] > 0).astype(float), cl),
            "exp": mean_ci(g["ret_net"], cl, key=(sid, table, label, "exp")),
            "r": mean_ci(g.loc[rok, "ret_net"] / g.loc[rok, "stop_dist"], cl[rok.to_numpy()],
                         key=(sid, table, label, "r")),
            "r_excluded": int((~rok).sum()),
            "t0": mean_ci((g["exit_reason"] == "stop_T0").astype(float), cl),
            "reasons": g["exit_reason"].value_counts().to_dict(),
        }
    return res


def system_fit(store: bt.HistoryStore) -> "tuple[dict, dict]":
    signals = pd.read_pickle(os.path.join(bt.OUTPUT_DIR, f"signals_{TAG}.pkl"))
    for sid, cutoff in bt.EXCLUDE_SIGNALS_BEFORE.items():
        signals = signals[~((signals["stock_id"] == sid) & (signals["date"] < cutoff))]
    adj_by_sid = {sid: bt.adjusted_prices(store, sid, DEV_END) for sid in sa.STOCKS}

    # 主表：不重疊交易（現行系統的真實行為＝無閘門），stop_first
    tr = pd.read_csv(os.path.join(bt.OUTPUT_DIR, f"trades_{TAG}.csv"), dtype={"stock_id": str})
    tr = tr[(tr["period"] == "全期") & (~tr["gate"]) & (tr["scenario"] == "stop_first")].copy()
    tr["stop_dist"] = _stop_dist(tr, signals, adj_by_sid)

    # 附表：每個推薦訊號各自獨立進場（已成交、完整 T+19 窗口）
    outcomes = bt.build_outcomes(signals, adj_by_sid)
    ind = pd.DataFrame([o for (_, _, sc), o in outcomes.items()
                        if sc == "stop_first" and o["status"] == "closed" and o.get("window_complete")])
    ind["stop_dist"] = _stop_dist(ind, signals, adj_by_sid)

    main = {sid: fit_stats(sid, tr[tr["stock_id"] == sid], "fit") for sid in sa.STOCKS}
    appx = {sid: fit_stats(sid, ind[ind["stock_id"] == sid], "fit_ind") for sid in sa.STOCKS}
    return main, appx


def limit_check(store: bt.HistoryStore) -> dict:
    """驗算漲跌停價：開發期除權息日，自行計算 vs TaiwanStockDividendResult 的 max_price／min_price"""
    n, bad = 0, []
    for sid in sa.STOCKS:
        div = store.query("TaiwanStockDividendResult", sid, DEV_START, DEV_END)
        for r in div.itertuples(index=False):
            up, dn = limit_prices(float(r.reference_price), r.date)
            n += 1
            up_ok, dn_ok = abs(up - float(r.max_price)) < 1e-6, abs(dn - float(r.min_price)) < 1e-6
            if not (up_ok and dn_ok):
                bad.append({"type": r.stock_or_cache_dividend, "up_ok": up_ok, "dn_ok": dn_ok})
    return {"n": n, "bad": pd.DataFrame(bad, columns=["type", "up_ok", "dn_ok"])}


# ══════════════════════════════════════════════════════════════
# 報告
# ══════════════════════════════════════════════════════════════

def _pc(x, d=2):
    return "—" if x is None or pd.isna(x) else f"{x * 100:+.{d}f}%"


def _ci(t, d=2):
    m, lo, hi = t
    return "—" if pd.isna(m) else f"{_pc(m, d)}（{_pc(lo, d)}～{_pc(hi, d)}）"


def _pr(t):
    """比例（附 CI）"""
    m, lo, hi = t if isinstance(t, tuple) else (t, np.nan, np.nan)
    if pd.isna(m):
        return "—"
    return f"{m * 100:.1f}%" + ("" if pd.isna(lo) else f"（{lo * 100:.1f}～{hi * 100:.1f}）")


def _p1(x):
    return "—" if pd.isna(x) else f"{x * 100:.2f}%"


def _num(t, d=2):
    m, lo, hi = t
    return f"{m:+.{d}f}（{lo:+.{d}f}～{hi:+.{d}f}）"


REASONS = ("stop", "stop_gap", "stop_T0", "t2", "t2_gap", "time")


def _fit_rows(P, fit):
    P("| 期間 | 交易 | 勝率 | 期望值（ret_net） | R 倍數 | R 排除 | 進場當日停損（stop_T0） | 出場原因 " +
      "／".join(REASONS) + " |")
    P("|---|---|---|---|---|---|---|---|")
    for label, f in fit.items():
        if f["n"] == 0:
            P(f"| {label} | 0 | — | — | — | — | — | — |")
            continue
        rs = "／".join(str(f["reasons"].get(k, 0)) for k in REASONS)
        r = "—" if pd.isna(f["r"][0]) else _num(f["r"])
        P(f"| {label} | {f['n']} | {_pr(f['win'])} | {_ci(f['exp'])} | {r} | {f['r_excluded']} | "
          f"{_pr(f['t0'])} | {rs} |")


def write_report(res: dict, fit_main: dict, fit_appx: dict, lim: dict) -> None:
    lines = []
    P = lambda s="": lines.append(s)
    n_tests = len(TESTED)
    bad = lim["bad"]

    P(f"# 個股體檢報告（{DEV_START} ～ {DEV_END}）\n")
    P(f"> **本報告只做描述，不做顯著性結論，也不主張任何一檔「有某種規律」。** 報告中共有 **{n_tests} 格**"
      f"「平均值／差值／迴歸係數」附 95% 信賴區間（比例類指標不計入）。即使每一格的真實值都是零，"
      f"在 95% 信賴水準下，預期仍約有 **{n_tests * 0.05:.0f} 格**純屬巧合地排除零；因此任何單一格排除零，"
      f"都不應解讀為該檔有穩定特性。\n")
    P(f"由 `python stock_profiles.py report` 產生。驗證期（{bt.VALIDATION_START} 起）未進入任何計算。\n")

    P("## 資料與定義\n")
    P(f"- **期間**：統計日為 {DEV_START} ～ {DEV_END} 的交易日。ATR14 與 60 日標準差的暖身期使用 "
      f"{WARMUP_START} 起的股價；往後看的報酬窗口超出 {DEV_END} 者不計。個股快取經 "
      f"`HistoryStore.query(as_of={DEV_END})` 截斷；加權指數與美股只下載到 {DEV_END}。")
    P(f"- **6669**：{bt.EXCLUDE_SIGNALS_BEFORE['6669']} 以前的股價在所有面向都不使用（沿用 backtest 的資料不可信期間）。")
    P("- **價格**：報酬、ATR、跳空、除權息走勢一律用還原價（`backtest.adjusted_prices`，含股利再投入）；"
      "漲跌停用原始價。")
    P("- **信賴區間**：95% CI 為 (股票, 年月) 叢集 bootstrap（2000 次）。")
    P("")
    P("**1. 波動特性**")
    P(f"- ATR% = ATR14（真實區間的 14 日簡單平均，同 `sa.calc_atr`）÷ 收盤；各年取中位數。日報酬標準差為還原收盤對收盤。")
    P(f"- 開盤跳空：開盤 ÷ 前一日收盤 − 1，≥ +{GAP_MIN:.0%}（上）或 ≤ −{GAP_MIN:.0%}（下），分開計頻率。")
    P(f"- 漲跌停：參考價 = 前一日原始收盤；除權息日改用 `reference_price`。限制 {LIMIT_CHANGE_DATE} 以前 ±7%、之後 ±10%，"
      f"依升降單位向內取整。「收盤」＝收盤價等於漲（跌）停價；「觸及」＝盤中最高（低）價等於漲（跌）停價。"
      f"驗算：開發期 {lim['n']} 個除權息日，自行計算的漲跌停價與 FinMind（`max_price`／`min_price`）不一致 "
      f"{len(bad)} 筆" + ("" if bad.empty else
      f"，類型 {'、'.join(f'{k} {v} 筆' for k, v in bad['type'].value_counts().items())}；"
      f"其中漲停價不一致 {int((~bad['up_ok']).sum())} 筆、跌停價不一致 {int((~bad['dn_ok']).sum())} 筆"
      "（這幾天沿用自行計算的價格，未另外修正）") + "。")
    P(f"- 先碰到哪個價位（基準，不限訊號日）：每個交易日 D 以 D 收盤與 D 當日 ATR 為基準，看 D+1 ～ D+{HIT_HORIZON} 的"
      f"最高／最低價，比較先碰到 +2×ATR（或 +3×ATR）還是 −{HIT_DN}×ATR；同一天兩者都碰到另列「同日」，"
      f"{HIT_HORIZON} 日內都沒碰到列「未碰到」。窗口不足 {HIT_HORIZON} 日的日子不計。")
    P("")
    P("**2. 大漲大跌之後**")
    P(f"- 標準化報酬 = 當日報酬 ÷ 前 {Z_WINDOW} 個交易日（不含當日）日報酬標準差；前 {Z_WINDOW} 日資料不足的日子不列入。"
      f"每檔取自己開發期標準化報酬的前 {Z_TAIL:.0%}（大漲日）與後 {Z_TAIL:.0%}（大跌日）。")
    P(f"- 之後 k 日報酬 = D 收盤 → D+k 收盤（k = {', '.join(map(str, FWD_DAYS))}），與該股開發期全部交易日的平均比較；"
      "差值 CI 以同一次叢集重抽計算子集與全體的平均。")
    P("")
    P("**3. 事件前後**")
    P(f"- 月營收年增率 = 當月營收 ÷ 去年同月營收 − 1。可取得日沿用回測規則（M 月營收於次月 {bt.REVENUE_PUBLISH_DAY} 日之後、"
      f"即 11 日起第一個交易日可用）；{REV_FWD} 日報酬 = 可取得日收盤 → {REV_FWD} 個交易日後收盤。"
      "超額報酬 = 個股報酬 − 加權指數同期報酬。每檔依自己的年增率分三等分（低／中／高）。")
    P(f"- 除權息：只分析現金股利（`stock_or_cache_dividend` 為「息」或「除息」），以除息日為第 0 天，"
      f"第 k 天的累積報酬 = 第 −{DIV_WIN + 1} 天收盤 → 第 k 天收盤（還原價，已扣除除息缺口）。"
      "含股票股利的事件（權、權息、除權、除權息）只列筆數。減資：`TaiwanStockDividendResult` 的事件類型只有"
      "息／除息／權／權息／除權／除權息，沒有減資類別；本報告未另外查找減資事件。")
    P("- 法說會：本機快取沒有日期資料，依指示未向 FinMind 試探，**本面向略過**。")
    P("")
    P("**4. 連動**")
    P("- 加權指數：`TaiwanStockPrice` 的 TAIEX（價格指數，不含股息，對日報酬影響很小）。beta = 個股日報酬對指數日報酬的 OLS 斜率；"
      "分年列點估計，全期附 CI。")
    P(f"- 美股（{', '.join(US_IDS)}，`USStockPrice` 的 `Adj_Close`）：台股第 D 日（前一個交易日為 P）對應美股日期 P ≤ u < D 的"
      "各場（即台股 D 日開盤前已收盤的美股），休市期間多場連乘；這段期間美股沒有開盤的日子不列入。"
      "台股指標分「收盤對收盤」與「開盤跳空」（開盤 ÷ P 收盤 − 1），列相關係數（附 CI）與斜率。")
    P("")
    P("**5. 系統適配**")
    P(f"- 主表：基準回測 `trades_{TAG}.csv` 的不重疊交易（全期、無閘門＝現行系統真實行為、stop_first），依訊號日拆成 "
      f"{SPLIT_DATE} 前／後。附表：每個推薦訊號各自獨立進場（`backtest.build_outcomes`，已成交、完整 T+19 窗口），交易會重疊。")
    P("- 勝率 = ret_net > 0 的比例；期望值 = 平均 ret_net（已扣成本）；R 倍數 = ret_net ÷ 進場時停損距離"
      "（同 risk_factors：停損價換算到還原價基準，相對成交價；距離 ≤ 0 者排除，列於「R 排除」）；"
      "進場當日停損 = 出場原因為 stop_T0 的比例（stop_first 情境下 T+0 觸及停損即全數出場）。"
      "出場原因為最後一筆出場的原因（目標1 出半不另列）。")
    P("")

    # ── 總覽 ──
    P("## 13 檔總覽\n")
    P("比例欄位不附 CI；期望值與 R 倍數附 95% CI。ATR%、日報酬標準差為「全期／2024 起」。系統欄位取自主表（不重疊交易、全期）。\n")
    P("| 股票 | 交易日 | ATR% 中位數 | 日報酬標準差 | 跳空≥3% 上／下 | 收盤漲停／跌停 | 先到 +2ATR／−1.5ATR／同日 | "
      "beta（TAIEX） | 與 ^SOX 前晚相關：收盤／開盤 | 交易 | 勝率 | 期望值 | R 倍數 | 進場當日停損 |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for sid, r in res.items():
        v, full, h = r["vol"], r["vol"]["full"], r["vol"]["hits"][2.0]
        sox = r["us"]["^SOX"]
        f = fit_main[sid]["全期"]
        P(f"| {sid} {sa.STOCKS[sid]} | {r['days']} | {_p1(full['atr_pct'])}／{_p1(full['atr_pct_post'])} | "
          f"{_p1(full['ret_sd'])}／{_p1(full['ret_sd_post'])} | {_pr(full['gap_up'][0])}／{_pr(full['gap_dn'][0])} | "
          f"{_pr(full['up_close'][0])}／{_pr(full['dn_close'][0])} | "
          f"{_pr(h['up'][0])}／{_pr(h['down'][0])}／{_pr(h['both'][0])} | {r['idx']['full']['beta'][0]:.2f} | "
          f"{sox['ret']['corr'][0]:.2f}／{sox['gap']['corr'][0]:.2f} | {f['n']} | {_pr(f['win'][0])} | "
          f"{_ci(f['exp'])} | {_num(f['r'])} | {_pr(f['t0'][0])} |")
    P()

    # ── 每檔 ──
    for sid, r in res.items():
        P(f"## {sid} {sa.STOCKS[sid]}\n")
        cut = bt.EXCLUDE_SIGNALS_BEFORE.get(sid)
        P(f"統計交易日 {r['days']}（{r['first']} ～ {r['last']}）" + (f"；{cut} 以前排除。" if cut else "。") + "\n")

        v = r["vol"]
        P("### 1. 波動特性\n")
        P("| 年 | 交易日 | ATR% 中位數 | 日報酬標準差 | 跳空≥3% 上 | 跳空≥3% 下 | 收盤漲停 | 觸及漲停 | 收盤跌停 | 觸及跌停 |")
        P("|---|---|---|---|---|---|---|---|---|---|")
        for y, row in v["yearly"].iterrows():
            P(f"| {y} | {int(row['days'])} | {_p1(row['atr_pct'])} | {_p1(row['ret_sd'])} | " +
              " | ".join(_pr(row[c]) for c in ("gap_up", "gap_dn", "up_close", "up_touch", "dn_close", "dn_touch")) + " |")
        fu = v["full"]
        P(f"| 全期 | {r['days']} | {_p1(fu['atr_pct'])} | {_p1(fu['ret_sd'])} | " +
          " | ".join(_pr(fu[c]) for c in ("gap_up", "gap_dn", "up_close", "up_touch", "dn_close", "dn_touch")) + " |")
        P()
        P(f"先碰到哪個價位（D+1 ～ D+{HIT_HORIZON}，比例附 95% CI）：\n")
        P(f"| 上方價位 | 起點日數 | 先到上方 | 先到 −{HIT_DN}×ATR | 同日 | 未碰到 |")
        P("|---|---|---|---|---|---|")
        for m, h in v["hits"].items():
            P(f"| +{m:g}×ATR | {h['n']} | {_pr(h['up'])} | {_pr(h['down'])} | {_pr(h['both'])} | {_pr(h['none'])} |")
        P()

        b = r["big"]
        P("### 2. 大漲大跌之後\n")
        P(f"標準化報酬有效日 {b['n_z']}；大漲日門檻 ≥ {b['z_hi']:.2f}（{b['n_top']} 天），大跌日門檻 ≤ {b['z_lo']:.2f}（{b['n_bot']} 天）。\n")
        P("| 之後 | 全部交易日 | 大漲日 | 大漲日 − 全部 | 大跌日 | 大跌日 − 全部 |")
        P("|---|---|---|---|---|---|")
        for row in b["rows"]:
            P(f"| {row['k']} 日 | {_ci(row['all'])} | {_ci(row['top'])} | {_ci(row['top_diff'])} | "
              f"{_ci(row['bot'])} | {_ci(row['bot_diff'])} |")
        P()

        P("### 3. 事件前後\n")
        rv = r["rev"]
        P(f"月營收年增率三等分（事件 {rv['n']} 個），可取得日起 {REV_FWD} 日報酬：\n")
        P("| 組 | 事件 | 年增率範圍 | 20 日報酬 | 20 日超額報酬（− 加權指數） |")
        P("|---|---|---|---|---|")
        for g in rv["groups"]:
            lo, hi = g["yoy_range"]
            P(f"| {g['grp']} | {g['n']} | {lo * 100:+.1f}% ～ {hi * 100:+.1f}% | {_ci(g['ret'])} | {_ci(g['excess'])} |")
        P()
        dv = r["div"]
        other = "、".join(f"{k} {n} 筆" for k, n in dv["other"].items()) or "無"
        P(f"除息（現金股利）事件 {dv['n_cash']} 個" + (f"（窗口不足略過 {dv['skipped']} 個）" if dv["skipped"] else "") +
          f"；含股票股利的事件（未分析）：{other}。\n")
        if dv["n_cash"]:
            P("| 天 | " + " | ".join(str(k) for k in dv["path"]) + " |")
            P("|---|" + "---|" * len(dv["path"]))
            P("| 平均累積報酬 | " + " | ".join(_pc(t[0]) for t in dv["path"].values()) + " |")
            P("| 95% CI | " + " | ".join(f"{_pc(t[1])}～{_pc(t[2])}" for t in dv["path"].values()) + " |")
            P()

        P("### 4. 連動\n")
        ix = r["idx"]
        P(f"對加權指數（全期，n={ix['full']['n']}）：beta {_num(ix['full']['beta'])}，相關係數 {ix['full']['corr'][0]:.2f}。\n")
        P("| 年 | " + " | ".join(ix["yearly"]) + " |")
        P("|---|" + "---|" * len(ix["yearly"]))
        P("| beta | " + " | ".join(f"{x['beta']:.2f}" for x in ix["yearly"].values()) + " |")
        P("| 相關 | " + " | ".join(f"{x['corr']:.2f}" for x in ix["yearly"].values()) + " |")
        P()
        P("與美股前一晚報酬：\n")
        P("| 美股 | 台股指標 | n | 相關係數 | 斜率 |")
        P("|---|---|---|---|---|")
        for uid, u in r["us"].items():
            for m, lab in (("ret", "收盤對收盤"), ("gap", "開盤跳空")):
                P(f"| {uid} | {lab} | {u[m]['n']} | {_num(u[m]['corr'])} | {u[m]['beta'][0]:.2f} |")
        P()

        P("### 5. 系統適配\n")
        P("主表（不重疊交易、無閘門、stop_first）：\n")
        _fit_rows(P, fit_main[sid])
        P()
        P("附表（每個推薦訊號各自獨立進場，交易重疊）：\n")
        _fit_rows(P, fit_appx[sid])
        P()

    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"寫入 {REPORT_PATH}（CI 格數 {n_tests}）")


def run_report() -> None:
    store = bt.HistoryStore()
    idx = load_index(store)
    us = {u: load_us(store, u) for u in US_IDS}
    res = {}
    for sid in sa.STOCKS:
        df = load_prices(store, sid)
        dev = df[df["date"] >= DEV_START]
        res[sid] = {
            "days": len(dev), "first": dev["date"].iloc[0], "last": dev["date"].iloc[-1],
            "vol": volatility(df),
            "big": big_moves(sid, df),
            "rev": revenue_events(store, sid, df, idx),
            "div": dividend_events(store, sid, df),
            "idx": index_link(sid, df, idx),
            "us": us_link(sid, df, us),
        }
        print(f"  {sid} done", file=sys.stderr)
    fit_main, fit_appx = system_fit(store)
    write_report(res, fit_main, fit_appx, limit_check(store))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fetch", "report"])
    args = ap.parse_args()
    if args.cmd == "fetch":
        fetch()
    else:
        run_report()


if __name__ == "__main__":
    main()
