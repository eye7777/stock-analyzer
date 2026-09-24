"""
風險因素切片（只驗證、不改評分規則）：訊號日收盤後可取得的特徵，與進場後
被套（進場當日觸及停損、T+1／T+5 報酬、依現行出場規則的每筆期望值）的關係。

設計原則
- 不重寫任何評分／交易邏輯：訊號、特徵一律由 backtest.compute_signal()（實盤
  get_stock_data／score_stock／classify_stock／calc_trade_levels）重建；交易
  結果一律用 backtest.build_outcomes()／simulate_trade()。
- 只用訊號日收盤後可取得的資料（HistoryStore 只回傳 date ≤ 訊號日）。
- 驗證期（backtest.VALIDATION_START 起）不使用：股價一律只讀到 end。
- 分組、主要對比、檢定族在看結果前已定下（見 CONTRASTS 與 docs/risk_factor_report.md）。

用法
  python risk_factors.py build    # 逐筆重建特徵，寫入 backtest_output/（需先跑過 backtest.py run）
  python risk_factors.py report   # 分組結果、主要對比與 BH 調整（markdown 印到 stdout）
  python risk_factors.py posthoc  # 事後檢查（不在事先計畫內、不列入檢定）
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

import backtest as bt
import stock_analyzer as sa

BUCKETS = (bt.SIGNAL_BUCKET, "觀望")
SAMPLES = (("推薦", bt.SIGNAL_BUCKET), ("觀望（5～6 分）", "觀望"))

# 因素 1 主要對比：嚴格版需在推薦樣本有 ≥ F1_MIN_N 筆已成交，否則退回寬鬆版
F1_MIN_N = 100
SHADOW_MIN, AMP_MIN, VR_MIN = 50.0, 3.0, 1.5
CHG_MIN = 6.0
QUINTILE_VARS = ("ma5_dev_pct", "ma20_dev_pct", "return_20d_pct")
INST_VARS = ("foreign_net", "foreign_pct_1d", "foreign_pct_5d",
             "trust_net", "trust_pct_1d", "trust_pct_5d")
METRICS = (("t0_stop", "進場當日觸及停損"), ("ret_t1", "T+1 報酬"),
           ("ret_t5", "T+5 報酬"), ("exp_sf", "期望值（stop_first）"))
FDR_Q = 0.10
N_BOOT_DIFF = 10_000


def _tag(start, end):
    return f"{start}_{end}"


def _load_signals(start: str, end: str) -> pd.DataFrame:
    signals = pd.read_pickle(os.path.join(bt.OUTPUT_DIR, f"signals_{_tag(start, end)}.pkl"))
    for sid, cutoff in bt.EXCLUDE_SIGNALS_BEFORE.items():
        signals = signals[~((signals["stock_id"] == sid) & (signals["date"] < cutoff))]
    return signals


def _inst_last_dates(store: bt.HistoryStore, sid: str) -> dict:
    """{法人名稱: 已排序的日期陣列}，用來查訊號日當下法人序列的最後一筆日期"""
    df = store._load("TaiwanStockInstitutionalInvestorsBuySell", sid)
    return {k: np.array(sorted(df.loc[df["name"] == k, "date"].unique()))
            for k in ("Foreign_Investor", "Investment_Trust")}


def _last_on_or_before(dates: np.ndarray, d: str):
    i = np.searchsorted(dates, d, side="right")
    return dates[i - 1] if i else None


def build_factors(start: str, end: str) -> pd.DataFrame:
    """推薦＋觀望訊號逐筆重建特徵，並附上成交狀態（不含任何報酬）"""
    if end >= bt.VALIDATION_START:
        raise SystemExit(f"end 必須早於驗證期起點 {bt.VALIDATION_START}")
    store = bt.HistoryStore()
    signals = _load_signals(start, end)
    adj_by_sid = {sid: bt.adjusted_prices(store, sid, end) for sid in sa.STOCKS}
    outcomes = bt.build_outcomes(signals, adj_by_sid, buckets=BUCKETS)

    rows = []
    sig = signals[signals["bucket"].isin(BUCKETS)]
    for sid, grp in sig.groupby("stock_id"):
        inst_dates = _inst_last_dates(store, sid)
        for s in grp.to_dict("records"):
            tr = outcomes.get((sid, s["date"], "stop_first"))
            if tr is None:
                continue
            fs = bt.compute_signal(store, sid, s["date"])
            vol = fs.get("volume")
            rows.append({
                "stock_id": sid, "date": s["date"], "bucket": s["bucket"],
                "total_score": s["total_score"],
                "cluster": f"{sid}-{s['date'][:7]}",
                "status": tr["status"],
                "window_complete": bool(tr.get("window_complete", False)),
                # 1. 訊號日 K 棒形狀與量比
                "price_chg_pct": fs.get("price_chg_pct"),
                "volume_ratio": fs.get("volume_ratio"),
                "amplitude_pct": fs.get("prev_day_amplitude_pct"),
                "upper_shadow_pct": fs.get("prev_day_upper_shadow_pct"),
                # 2／4. 法人
                "foreign_net": fs.get("foreign_net"),
                "trust_net": fs.get("trust_net"),
                "volume": vol,
                "foreign_net_5d": fs.get("foreign_net_5d"),
                "trust_net_5d": fs.get("trust_net_5d"),
                "foreign_pct_5d": fs.get("foreign_net_pct_of_volume"),
                "trust_pct_5d": fs.get("trust_net_pct_of_volume"),
                "foreign_pct_1d": sa.calc_net_pct_of_volume(fs.get("foreign_net"), vol),
                "trust_pct_1d": sa.calc_net_pct_of_volume(fs.get("trust_net"), vol),
                "foreign_last_date": _last_on_or_before(inst_dates["Foreign_Investor"], s["date"]),
                "trust_last_date": _last_on_or_before(inst_dates["Investment_Trust"], s["date"]),
                # 3. 乖離與近 20 日漲幅
                "ma5_dev_pct": sa._dev_pct(fs.get("close"), fs.get("ma5")),
                "ma20_dev_pct": sa._dev_pct(fs.get("close"), fs.get("ma20")),
                "return_20d_pct": sa.calc_20d_return_pct(fs["price_series"]),
            })
        print(f"  factors {sid}: {len(grp)}", file=sys.stderr)
    df = pd.DataFrame(rows)
    df.to_pickle(os.path.join(bt.OUTPUT_DIR, f"risk_factors_{_tag(start, end)}.pkl"))
    return df


# ══════════════════════════════════════════════════════════════
# 結果：併上交易結果（沿用 build_outcomes），分組與主要對比
# ══════════════════════════════════════════════════════════════

def attach_outcomes(fac: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """每筆訊號補上：是否成交、進場當日觸及停損、T+1／T+5 報酬（未扣成本，
    定義同 backtest.run_slices）、兩情境的 ret_net（完整 T+19 窗口才計），
    以及事後檢查用的 ATR%（訊號日 atr / close）與進場時的停損距離"""
    store = bt.HistoryStore()
    signals = _load_signals(start, end)
    adj_by_sid = {sid: bt.adjusted_prices(store, sid, end) for sid in sa.STOCKS}
    outcomes = bt.build_outcomes(signals, adj_by_sid, buckets=BUCKETS)
    sig_by_key = signals.set_index(["stock_id", "date"])[["close", "atr", "stop_loss"]]
    rows = []
    for r in fac[["stock_id", "date"]].itertuples(index=False):
        sf = outcomes[(r.stock_id, r.date, "stop_first")]
        tf = outcomes[(r.stock_id, r.date, "target_first")]
        sig = sig_by_key.loc[(r.stock_id, r.date)]
        filled = sf["status"] in ("closed", "open")
        rec = {"filled": filled if sf["status"] != "pending" else np.nan,
               "atr_pct": sig["atr"] / sig["close"] * 100 if sig["atr"] else np.nan}
        if filled:
            adj = adj_by_sid[r.stock_id]
            e = int(adj.index[adj["date"] == sf["entry_date"]][0])
            # 進場時的停損距離：停損價換算到還原價基準（同 simulate_trade 的 _scale），
            # 相對成交價；成交價已在停損之下（開盤跳空）時距離 ≤ 0，R 倍數不定義
            stop_adj = sig["stop_loss"] * adj.at[e - 1, "close"] / sig["close"]
            rec["stop_dist"] = 1 - stop_adj / sf["entry_px"]

            def ret(k):
                return adj.at[e + k, "close"] / sf["entry_px"] - 1 if e + k < len(adj) else np.nan

            rec.update(t0_stop=float(any(x[3] == "stop_T0" for x in sf["exits"])),
                       ret_t1=ret(1), ret_t5=ret(5))
            for key, tr in (("exp_sf", sf), ("exp_tf", tf)):
                ok = tr["status"] == "closed" and tr.get("window_complete", False)
                rec[key] = tr["ret_net"] if ok else np.nan
        rows.append(rec)
    return pd.concat([fac.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def _quintile(x: pd.Series, cuts) -> pd.Series:
    return pd.Series(np.searchsorted(cuts, x.to_numpy(), side="right") + 1, index=x.index)


def define_groups(df: pd.DataFrame) -> "tuple[pd.DataFrame, dict]":
    """依事先定下的規則加上分組欄位。分位切點一律由「推薦、已成交」樣本算出，
    觀望樣本套用同一組數值切點。回傳 (df, meta)"""
    df = df.copy()
    rec_f = df[(df["bucket"] == bt.SIGNAL_BUCKET) & (df["filled"] == True)]  # noqa: E712
    meta = {"cuts": {}}

    sh = df["upper_shadow_pct"] >= SHADOW_MIN
    vr = df["volume_ratio"] >= VR_MIN
    amp = df["amplitude_pct"] >= AMP_MIN
    strict = sh & vr & amp
    n_strict = int(strict[rec_f.index].sum())
    meta["f1_n_strict"] = n_strict
    meta["f1_strict"] = n_strict >= F1_MIN_N
    df["f1_primary"] = strict if meta["f1_strict"] else (sh & vr)
    df["f1_cell"] = np.select([sh & vr, sh & ~vr, ~sh & vr],
                              ["爆量長上影", "長上影非爆量", "爆量非長上影"], "其他")

    for who in ("foreign", "trust"):
        x = df[f"{who}_net_5d"]
        df[f"{who}_5d_neg"] = x < 0
        df[f"{who}_5d_sign"] = np.select([x < 0, x == 0], ["< 0", "= 0"], "> 0")

    for v in QUINTILE_VARS + INST_VARS:
        cuts = rec_f[v].quantile([.2, .4, .6, .8]).to_numpy()
        meta["cuts"][v] = cuts
        df[f"{v}_q"] = _quintile(df[v], cuts)
    for v in QUINTILE_VARS:
        df[f"{v}_top"] = df[f"{v}_q"] == 5

    df["chg_hi"] = df["price_chg_pct"] >= CHG_MIN
    df["chg_grp"] = pd.cut(df["price_chg_pct"], [-np.inf, 0, 3, 6, 9.5, np.inf], right=False,
                           labels=["< 0", "0～3", "3～6", "6～9.5", "≥ 9.5"]).astype(str)
    return df, meta


def contrasts(meta: dict):
    """7 個主要對比：(代號, 名稱, 風險組欄位, 風險組標籤, 對照組標籤)"""
    f1 = (f"上影 ≥ {SHADOW_MIN:.0f}% 且振幅 ≥ {AMP_MIN:.0f}% 且量比 ≥ {VR_MIN}"
          if meta["f1_strict"] else f"上影 ≥ {SHADOW_MIN:.0f}% 且量比 ≥ {VR_MIN}")
    c = meta["cuts"]
    return [
        ("C1", "爆量長上影", "f1_primary", f1, "其他"),
        ("C2", "外資近 5 日合計", "foreign_5d_neg", "< 0", "≥ 0"),
        ("C3", "投信近 5 日合計", "trust_5d_neg", "< 0", "≥ 0"),
        ("C4", "MA5 乖離", "ma5_dev_pct_top", f"Q5（≥ {c['ma5_dev_pct'][3]:.2f}%）", "Q1～Q4"),
        ("C5", "MA20 乖離", "ma20_dev_pct_top", f"Q5（≥ {c['ma20_dev_pct'][3]:.2f}%）", "Q1～Q4"),
        ("C6", "近 20 日漲幅", "return_20d_pct_top", f"Q5（≥ {c['return_20d_pct'][3]:.2f}%）", "Q1～Q4"),
        ("C7", "訊號日漲幅", "chg_hi", f"≥ {CHG_MIN:.0f}%", f"< {CHG_MIN:.0f}%"),
    ]


def cluster_diff(values, clusters, flag, n_boot=N_BOOT_DIFF, seed=0):
    """風險組平均 − 對照組平均，以 (股票, 年月) 叢集為單位整體重抽（同一叢集
    內兩組的訊號一起抽）。回傳 (差值, CI 下緣, CI 上緣, 雙尾 bootstrap p 值)"""
    d = pd.DataFrame({"v": values, "c": clusters, "f": flag}).dropna(subset=["v"])
    d["f"] = d["f"].astype(bool)
    point = d.loc[d["f"], "v"].mean() - d.loc[~d["f"], "v"].mean()
    g = d.groupby(["c", "f"])["v"].agg(["sum", "count"]).unstack("f", fill_value=0)
    sA, cA = g[("sum", True)].to_numpy(), g[("count", True)].to_numpy()
    sB, cB = g[("sum", False)].to_numpy(), g[("count", False)].to_numpy()
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot // 1000):
        idx = rng.integers(0, len(g), size=(1000, len(g)))
        with np.errstate(invalid="ignore", divide="ignore"):
            diffs.append(sA[idx].sum(1) / cA[idx].sum(1) - sB[idx].sum(1) / cB[idx].sum(1))
    diffs = np.concatenate(diffs)
    diffs = diffs[np.isfinite(diffs)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    p = max(p, 1 / len(diffs))
    return point, lo, hi, p


def bh_adjust(p) -> np.ndarray:
    """Benjamini-Hochberg 調整後 p 值（q 值）"""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def run_contrasts(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    rows = []
    for label, bucket in SAMPLES:
        s = df[(df["bucket"] == bucket) & (df["filled"] == True)]  # noqa: E712
        for code, name, col, a, b in contrasts(meta):
            for m, mname in METRICS + (("exp_tf", "期望值（target_first，敏感度）"),):
                pt, lo, hi, p = cluster_diff(s[m], s["cluster"], s[col])
                va = s.loc[s[col], m].dropna()
                vb = s.loc[~s[col], m].dropna()
                rows.append({"sample": label, "code": code, "factor": name, "risk": a, "ref": b,
                             "metric": m, "metric_name": mname,
                             "n_a": len(va), "cl_a": s.loc[va.index, "cluster"].nunique(),
                             "n_b": len(vb), "mean_a": va.mean(), "mean_b": vb.mean(),
                             "diff": pt, "lo": lo, "hi": hi, "p": p,
                             "in_family": m != "exp_tf"})
    out = pd.DataFrame(rows)
    out["q"] = np.nan
    for label, _ in SAMPLES:
        fam = (out["sample"] == label) & out["in_family"]
        out.loc[fam, "q"] = bh_adjust(out.loc[fam, "p"])
    return out


# ── 輸出 ──────────────────────────────────────────────────────

def _p(x, digits=2):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.{digits}f}%"


def _pp(x):
    """百分點差值，帶正負號"""
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.2f}"


def _pv(x):
    return "—" if x is None or not np.isfinite(x) else ("<0.001" if x < 0.001 else f"{x:.3f}")


def _ci_cell(v, c, digits=2):
    v = v.dropna()
    if v.empty:
        return "—"
    lo, hi = bt._cluster_ci(v, c[v.index])
    return f"{_p(v.mean(), digits)}（{_p(lo, digits)}～{_p(hi, digits)}）"


def group_table(s: pd.DataFrame, key: str, order) -> None:
    """描述表：成交率、四個指標（附叢集 95% CI）與 target_first 期望值"""
    print("| 組別 | 下單 | 成交率 | 成交 | 叢集 | 進場當日觸及停損 | T+1 報酬 | T+5 報酬 | 期望值 stop_first | 期望值 target_first |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for gv in order:
        g = s[s[key] == gv]
        orders = g[g["filled"].notna()]
        f = orders[orders["filled"] == True]  # noqa: E712
        if f.empty:
            print(f"| {gv} | {len(orders)} | — | 0 | 0 | — | — | — | — | — |")
            continue
        print(f"| {gv} | {len(orders)} | {_p(len(f) / len(orders), 1)} | {len(f)} | {f['cluster'].nunique()} | "
              f"{_ci_cell(f['t0_stop'], f['cluster'], 1)} | {_ci_cell(f['ret_t1'], f['cluster'])} | "
              f"{_ci_cell(f['ret_t5'], f['cluster'])} | {_ci_cell(f['exp_sf'], f['cluster'])} | "
              f"{_p(f['exp_tf'].mean())} |")
    print()


def print_report(df: pd.DataFrame, meta: dict, res: pd.DataFrame) -> None:
    P = print
    P("# 風險因素切片（自動輸出）\n")
    P(f"因素 1 嚴格定義（上影 ≥ {SHADOW_MIN:.0f}% 且振幅 ≥ {AMP_MIN:.0f}% 且量比 ≥ {VR_MIN}）"
      f"在推薦樣本的已成交筆數：{meta['f1_n_strict']}（門檻 {F1_MIN_N}）→ 採用"
      f"{'嚴格' if meta['f1_strict'] else '寬鬆（上影 ≥ 50% 且量比 ≥ 1.5）'}定義\n")
    P("分位切點（推薦、已成交樣本）：")
    for v, c in meta["cuts"].items():
        P(f"- {v}：" + "／".join(f"{x:.2f}" for x in c))
    P()

    for label, _ in SAMPLES:
        r = res[res["sample"] == label]
        P(f"## 主要對比：{label}（風險組 − 對照組；差值單位：百分點；BH 族 = 28 次）\n")
        P("| 代號 | 因素 | 風險組 | 對照組 | 指標 | 風險組 n（叢集） | 對照組 n | 風險組 | 對照組 | 差值 | 叢集 95% CI | p | BH q |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for x in r.itertuples():
            q = f"{_pv(x.q)}{' ✱' if x.q <= FDR_Q else ''}" if x.in_family else "（不檢定）"
            P(f"| {x.code} | {x.factor} | {x.risk} | {x.ref} | {x.metric_name} | {x.n_a}（{x.cl_a}） | {x.n_b} | "
              f"{_p(x.mean_a)} | {_p(x.mean_b)} | {_pp(x.diff)} | {_pp(x.lo)}～{_pp(x.hi)} | {_pv(x.p)} | {q} |")
        P(f"\n✱：BH 調整後 q ≤ {FDR_Q}\n")

    P("## 判定（推薦通過 BH 且觀望差值同號）\n")
    P("| 代號 | 指標 | 推薦差值 | 推薦 q | 觀望差值 | 觀望 q | 判定 |")
    P("|---|---|---|---|---|---|---|")
    fam = res[res["in_family"]]
    for (code, m), g in fam.groupby(["code", "metric"], sort=False):
        a = g[g["sample"] == SAMPLES[0][0]].iloc[0]
        b = g[g["sample"] == SAMPLES[1][0]].iloc[0]
        ok = a.q <= FDR_Q and np.sign(a["diff"]) == np.sign(b["diff"])
        P(f"| {code} | {a.metric_name} | {_pp(a['diff'])} | {_pv(a.q)} | {_pp(b['diff'])} | {_pv(b.q)} | "
          f"{'有證據' if ok else '探索性'} |")
    P()

    for label, bucket in SAMPLES:
        s = df[df["bucket"] == bucket]
        P(f"## 描述：{label}\n")
        P("### 因素 1：主要對比定義\n")
        group_table(s.assign(_g=np.where(s["f1_primary"], "符合", "其他")), "_g", ["符合", "其他"])
        P(f"### 因素 1：2×2（上影 ≥ {SHADOW_MIN:.0f}% × 量比 ≥ {VR_MIN}）\n")
        group_table(s, "f1_cell", ["爆量長上影", "長上影非爆量", "爆量非長上影", "其他"])
        for who, name in (("foreign", "外資"), ("trust", "投信")):
            P(f"### 因素 2：{name}近 5 日合計\n")
            group_table(s, f"{who}_5d_sign", ["< 0", "= 0", "> 0"])
        for v, name in (("ma5_dev_pct", "MA5 乖離"), ("ma20_dev_pct", "MA20 乖離"),
                        ("return_20d_pct", "近 20 日漲幅")):
            P(f"### 因素 3：{name}（五分位）\n")
            group_table(s, f"{v}_q", [1, 2, 3, 4, 5])
        for v in INST_VARS:
            P(f"### 因素 4（只描述）：{v}（五分位）\n")
            group_table(s, f"{v}_q", [1, 2, 3, 4, 5])
        P("### 因素 5：訊號日漲幅\n")
        group_table(s, "chg_grp", ["< 0", "0～3", "3～6", "6～9.5", "≥ 9.5"])

    rec_f = df[(df["bucket"] == bt.SIGNAL_BUCKET) & (df["filled"] == True)]  # noqa: E712
    P("## 因素 4 各五分位的股票組成（推薦、已成交筆數）\n")
    for v in INST_VARS:
        ct = pd.crosstab(rec_f["stock_id"], rec_f[f"{v}_q"]).reindex(columns=[1, 2, 3, 4, 5], fill_value=0)
        P(f"### {v}\n")
        P("| 股票 | Q1 | Q2 | Q3 | Q4 | Q5 |")
        P("|---|---|---|---|---|---|")
        for sid, row in ct.iterrows():
            P(f"| {sid} {sa.STOCKS[sid]} | " + " | ".join(str(int(x)) for x in row) + " |")
        P()


def run_report(start: str, end: str) -> None:
    if end >= bt.VALIDATION_START:
        raise SystemExit(f"end 必須早於驗證期起點 {bt.VALIDATION_START}")
    path = os.path.join(bt.OUTPUT_DIR, f"risk_factors_{_tag(start, end)}.pkl")
    fac = pd.read_pickle(path) if os.path.exists(path) else build_factors(start, end)
    df = attach_outcomes(fac, start, end)
    df, meta = define_groups(df)
    res = run_contrasts(df, meta)
    res.to_csv(os.path.join(bt.OUTPUT_DIR, f"risk_contrasts_{_tag(start, end)}.csv"), index=False)
    print_report(df, meta, res)


# ══════════════════════════════════════════════════════════════
# 事後檢查：看過主要結果後才加的，只作解讀參考，不算 p 值、不列入 BH
# ══════════════════════════════════════════════════════════════

POSTHOC_CONTRASTS = ("C1", "C5", "C6", "C7")
R_CONTRASTS = ("C5", "C6", "C7")
PERIOD_SPLIT = "2024-01-01"
OVERLAP_COLS = (("ma5_dev_pct_top", "C4 MA5 乖離 Q5"), ("ma20_dev_pct_top", "C5 MA20 乖離 Q5"),
                ("return_20d_pct_top", "C6 近 20 日漲幅 Q5"), ("chg_hi", "C7 漲幅 ≥ 6%"),
                ("f1_primary", "C1 爆量長上影"))
SPEARMAN_VARS = ("price_chg_pct", "volume_ratio", "ma5_dev_pct", "ma20_dev_pct", "return_20d_pct")


def print_posthoc(df: pd.DataFrame, meta: dict) -> None:
    P = print
    rec = df[(df["bucket"] == bt.SIGNAL_BUCKET) & (df["filled"] == True)]  # noqa: E712
    con = {c[0]: c for c in contrasts(meta)}
    late = rec["date"] >= PERIOD_SPLIT

    P("## 事後檢查（不在事先計畫內，只作解讀參考，不列入檢定）\n")
    P(f"### 時間與波動度（推薦、已成交；期望值 = stop_first ret_net；以 {PERIOD_SPLIT} 切分）\n")
    P(f"推薦、已成交整體：{PERIOD_SPLIT[:4]} 年起占 {_p(late.mean(), 1)}\n")
    P("| 風險組 | 2024～2026 年占比（風險組／對照組） | ATR% 中位數（風險組／對照組） | "
      "2024 年前：風險組／對照組期望值（風險組 n） | 差值 | 2024 年起：風險組／對照組期望值（風險組 n） | 差值 |")
    P("|---|---|---|---|---|---|---|")
    for code in POSTHOC_CONTRASTS:
        _, name, col, _, _ = con[code]
        a, b = rec[rec[col]], rec[~rec[col]]
        cells = []
        for m in (~late, late):
            s = rec[m]
            ea, eb = s.loc[s[col], "exp_sf"].mean(), s.loc[~s[col], "exp_sf"].mean()
            cells.append(f"{_p(ea)}／{_p(eb)}（{int(s[col].sum())}） | {_pp(ea - eb)}")
        P(f"| {code} {name} | {_p((a['date'] >= PERIOD_SPLIT).mean(), 1)}／{_p((b['date'] >= PERIOD_SPLIT).mean(), 1)} | "
          f"{a['atr_pct'].median():.2f}／{b['atr_pct'].median():.2f} | {cells[0]} | {cells[1]} |")
    P()

    P("### 因素重疊（推薦、已成交；列＝符合 A 的訊號中，也符合 B 的比例）\n")
    P("| A ＼ B | " + " | ".join(n for _, n in OVERLAP_COLS) + " |")
    P("|---|" + "---|" * len(OVERLAP_COLS))
    for ca, na in OVERLAP_COLS:
        g = rec[rec[ca]]
        P(f"| {na}（n={len(g)}） | " + " | ".join(
            "—" if cb == ca else _p(g[cb].mean(), 0) for cb, _ in OVERLAP_COLS) + " |")
    P()
    corr = rec[list(SPEARMAN_VARS)].corr(method="spearman")
    P("Spearman 相關係數（推薦、已成交）：\n")
    P("| | " + " | ".join(SPEARMAN_VARS) + " |")
    P("|---|" + "---|" * len(SPEARMAN_VARS))
    for v in SPEARMAN_VARS:
        P(f"| {v} | " + " | ".join(f"{corr.at[v, w]:.2f}" for w in SPEARMAN_VARS) + " |")
    P()

    P("### 敏感度：每筆期望值改用 R 倍數（C5～C7，推薦、已成交）\n")
    P("R 倍數 = stop_first 的 ret_net ÷ 進場時的停損距離%（(成交價 − 停損價) / 成交價，停損價換算到還原價基準）。"
      "只計完整 T+19 窗口的交易；成交價已在停損之下（停損距離 ≤ 0）的交易 R 不定義、排除。"
      "95% CI 為 (股票, 年月) 叢集 bootstrap；不算 p 值、不做 BH。\n")
    ok = rec["exp_sf"].notna()
    bad = ok & (rec["stop_dist"] <= 0)
    P(f"排除（停損距離 ≤ 0）：{int(bad.sum())} 筆／{int(ok.sum())} 筆\n")
    r = rec[ok & ~bad].assign(r_mult=lambda x: x["exp_sf"] / x["stop_dist"])
    P("| 對比 | 組別 | n | 叢集 | 停損距離% 中位數 | 期望值（%） | 期望值（R） | 差值（R，風險組 − 對照組） |")
    P("|---|---|---|---|---|---|---|---|")
    for code in R_CONTRASTS:
        _, name, col, a_lab, b_lab = con[code]
        means = {}
        for flag, lab in ((True, a_lab), (False, b_lab)):
            g = r[r[col] == flag]
            lo, hi = bt._cluster_ci(g["r_mult"], g["cluster"])
            means[flag] = g["r_mult"].mean()
            diff = f"{means[True] - means[False]:+.3f}" if not flag else ""
            P(f"| {code if flag else ''} {name if flag else ''} | {lab} | {len(g)} | {g['cluster'].nunique()} | "
              f"{g['stop_dist'].median() * 100:.2f}% | {_ci_cell(g['exp_sf'], g['cluster'])} | "
              f"{g['r_mult'].mean():+.3f}R（{lo:+.3f}～{hi:+.3f}） | {diff} |")
    P()


def run_posthoc(start: str, end: str) -> None:
    if end >= bt.VALIDATION_START:
        raise SystemExit(f"end 必須早於驗證期起點 {bt.VALIDATION_START}")
    path = os.path.join(bt.OUTPUT_DIR, f"risk_factors_{_tag(start, end)}.pkl")
    fac = pd.read_pickle(path) if os.path.exists(path) else build_factors(start, end)
    df, meta = define_groups(attach_outcomes(fac, start, end))
    print_posthoc(df, meta)


def main():
    ap = argparse.ArgumentParser(description="風險因素切片")
    ap.add_argument("cmd", choices=["build", "report", "posthoc"])
    ap.add_argument("--start", default="2012-08-01")
    ap.add_argument("--end", default="2026-06-22")
    args = ap.parse_args()
    if args.cmd == "build":
        build_factors(args.start, args.end)
    elif args.cmd == "report":
        run_report(args.start, args.end)
    elif args.cmd == "posthoc":
        run_posthoc(args.start, args.end)


if __name__ == "__main__":
    main()
