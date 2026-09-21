#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
台股 AI 每日分析系統
每天早上 7:30 自動執行，分析 12 檔台股並寄送 Email 報告
"""

import os
import sys
import json
import time
import logging
import subprocess
import traceback
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd
import requests
from dotenv import load_dotenv
import anthropic

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    # yfinance 未安裝時自動安裝，避免排程執行時因環境缺套件而跳過美股數據
    print("⚠️  偵測到 yfinance 未安裝，嘗試自動安裝...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "yfinance>=0.2.40"],
            check=True,
        )
        import yfinance as yf
        HAS_YFINANCE = True
        print("✅ yfinance 自動安裝完成")
    except Exception as _install_err:
        print(f"⚠️  yfinance 自動安裝失敗，跳過美股數據：{_install_err}")
        HAS_YFINANCE = False

# ══════════════════════════════════════════════════════════════
# 載入環境變數
# ══════════════════════════════════════════════════════════════
# 找到 .env 的絕對路徑（不管從哪裡執行都能找到）
_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_DIR, ".env"))

FINMIND_TOKEN   = os.getenv("FINMIND_TOKEN", "")
ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
GMAIL_USER      = os.getenv("GMAIL_USER", "")
GMAIL_PASSWORD  = os.getenv("GMAIL_PASSWORD", "")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL", "")

# ══════════════════════════════════════════════════════════════
# 追蹤股票清單
# ══════════════════════════════════════════════════════════════
STOCKS = {
    "3017": "奇鋐",       # 散熱
    "3324": "雙鴻",       # 散熱
    "3653": "健策",       # 散熱均熱片
    "3533": "嘉澤",       # 連接器
    "6669": "緯穎",       # AI伺服器
    "2356": "英業達",     # AI伺服器
    "2345": "智邦",       # 網通
    "2383": "台光電",     # CCL
    "3037": "欣興",       # ABF載板
    "2301": "光寶科",     # 電源/BBU
    "3211": "順達",       # BBU
    "2059": "川湖",       # 伺服器滑軌
    "1513": "中興電",     # 重電
}

# ══════════════════════════════════════════════════════════════
# 事件旗標設定
# ══════════════════════════════════════════════════════════════
# 手動維護的個股風險備註（司法調查、處置股等），有設定的股票會在報告中以
# 醒目文字顯示。
#
# 處置股為何在此手動維護：TWSE OpenAPI 有免費、可程式取得的處置股清單
# （https://openapi.twse.com.tw/v1/announcement/punish，集中市場公布處置股票，
# 免 token），但 FinMind 對應的 TaiwanStockDispositionSecuritiesPeriod
# 資料集目前 token 等級（register）打不到（回傳 400 需升級付費層），
# 尚未串接自動偵測，故先以此手動 config 頂著。
EVENT_FLAGS = {
    "3037": "司法調查中（8/28 起）",
}

# 除權息事件距今幾個「交易日」內視為「近期」，報告會標示提醒旗標
RECENT_DIVIDEND_WINDOW_DAYS = 20

# 還原後的收盤價序列，近幾個交易日內若仍出現單日變動超過此百分比，視為疑似
# 價格斷層（台股漲跌限制 ±10%，超過代表還原可能不完整，或有未被
# TaiwanStockDividendResult 涵蓋的事件，例如股票分割）
PRICE_GAP_LOOKBACK_DAYS = 60
PRICE_GAP_THRESHOLD_PCT = 11.0

# 訊號日誌（用於事後回測，repo 為 public，欄位內容一律可公開）
SIGNALS_LOG_DIR  = os.path.join(_DIR, "data")
SIGNALS_LOG_PATH = os.path.join(SIGNALS_LOG_DIR, "signals_log.csv")

# ══════════════════════════════════════════════════════════════
# 日誌設定
# ══════════════════════════════════════════════════════════════
LOG_FILE = os.path.join(_DIR, "analyzer.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# FinMind API 工具函數
# ══════════════════════════════════════════════════════════════
FINMIND_BASE = "https://api.finmindtrade.com/api/v4/data"


def _finmind_request(dataset: str, stock_id: str, start_date: str) -> "tuple[pd.DataFrame, str | None]":
    """FinMind API 的底層請求，回傳 (DataFrame, error_msg)

    error_msg 為 None 代表 API 呼叫成功——即使 data 是空陣列，也視為成功的
    「查無資料」，不是失敗。error_msg 非 None 代表呼叫本身失敗（HTTP 例外、
    或 FinMind 回傳 status != 200），此時 DataFrame 必為空。
    呼叫端如果需要區分「查無資料」與「API 呼叫失敗」（例如除權息這種缺資料
    會導致還原完全沒做、卻又不會出錯的情境），必須用這個函式而不是
    finmind_get，否則兩種情況都會被誤判成同一種空結果。
    """
    try:
        params = {
            "dataset": dataset,
            "data_id": stock_id,
            "start_date": start_date,
            "token": FINMIND_TOKEN,
        }
        resp = requests.get(FINMIND_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != 200:
            return pd.DataFrame(), data.get("msg", "未知錯誤")

        if not data.get("data"):
            return pd.DataFrame(), None

        return pd.DataFrame(data["data"]), None

    except Exception as e:
        return pd.DataFrame(), str(e)


def finmind_get(dataset: str, stock_id: str, start_date: str) -> pd.DataFrame:
    """呼叫 FinMind API 取得資料，失敗時回傳空 DataFrame

    沿用舊行為給大多數呼叫端使用：呼叫失敗與「查無資料」在這裡一律回傳空
    DataFrame，差異只留在 log。需要區分兩者時改用 _finmind_request。
    """
    df, err = _finmind_request(dataset, stock_id, start_date)
    if err:
        log.warning(f"FinMind {dataset} ({stock_id}): {err}")
    return df


# ══════════════════════════════════════════════════════════════
# 美股數據（yfinance）
# ══════════════════════════════════════════════════════════════

US_SYMBOLS = {
    "SOX":  ("^SOX",  "費半"),
    "IXIC": ("^IXIC", "納斯達克"),
    "NVDA": ("NVDA",  "NVIDIA"),
    "AMD":  ("AMD",   "AMD"),
}


def get_us_market_data() -> dict:
    """使用 yfinance 抓取美股前一交易日收盤數據"""
    log.info("抓取美股數據（yfinance）...")

    if not HAS_YFINANCE:
        log.warning("⚠️  yfinance 未安裝，跳過美股數據")
        return {}

    result = {}
    for key, (ticker, label) in US_SYMBOLS.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if len(hist) < 2:
                raise ValueError("資料筆數不足")
            prev  = float(hist["Close"].iloc[-2])
            last  = float(hist["Close"].iloc[-1])
            chg_p = (last - prev) / prev * 100 if prev else 0.0
            result[key] = {
                "label":   label,
                "close":   round(last, 2),
                "chg_pct": round(chg_p, 2),
                "chg_str": f"{'+' if chg_p >= 0 else ''}{chg_p:.2f}%",
            }
            log.info(f"  ✅ {label}（{ticker}）：{result[key]['close']}  {result[key]['chg_str']}")
        except Exception as e:
            log.warning(f"  ⚠️  yfinance 抓取 {ticker} 失敗：{e}")
            result[key] = {"label": label, "close": "N/A", "chg_pct": 0.0, "chg_str": "N/A"}

    return result


# ══════════════════════════════════════════════════════════════
# 美股新聞分析（Claude web_search）
# ══════════════════════════════════════════════════════════════

_SEARCH_KEYWORDS = [
    "AI 伺服器 台股",
    "半導體 重大消息",
    "NVIDIA AMD 最新",
    "SpaceX Anthropic AI",
]


def get_us_news_analysis(us_data: dict) -> dict:
    """兩步驟：① Claude web_search 搜尋新聞；② 結構化輸出影響分析"""
    log.info("使用 Claude web_search 搜尋最新財經新聞...")

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    stocks_list = "、".join(f"{n}({sid})" for sid, n in STOCKS.items())
    us_summary  = "、".join(
        f"{v['label']} {v['chg_str']}" for v in us_data.values() if isinstance(v, dict)
    ) or "（美股數據未取得）"

    # ── 步驟一：web_search 搜尋 ────────────────────────────────
    keywords_str = "\n".join(f'{i+1}. "{kw}"' for i, kw in enumerate(_SEARCH_KEYWORDS))
    search_prompt = (
        f"請搜尋以下關鍵字的今日最新財經新聞：\n{keywords_str}\n\n"
        f"昨夜美股：{us_summary}\n\n"
        "搜尋後請整理出最重要的 2-3 則新聞標題與摘要。"
    )

    news_text = ""
    try:
        search_resp = client.beta.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=3000,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": search_prompt}],
            betas=["web-search-2025-03-05"],
        )
        for block in search_resp.content:
            if hasattr(block, "text"):
                news_text += block.text
        log.info(f"  web_search 完成，取得 {len(news_text)} 字")
    except Exception as e:
        log.warning(f"  Claude web_search 失敗：{e}")
        news_text = f"新聞搜尋失敗（{e}）"

    # ── 步驟二：結構化分析 ──────────────────────────────────────
    analysis_prompt = (
        f"根據以下新聞內容和美股數據，分析對台股的影響。\n\n"
        f"【昨夜美股】{us_summary}\n\n"
        f"【新聞內容】\n{news_text[:3000]}\n\n"
        f"請呼叫 output_news_analysis 工具，輸出：\n"
        f"1. 2-3 則最重要的新聞，說明各自對哪些台股標的有影響（{stocks_list}）\n"
        f"2. 今日開盤前一句話提示（30 字以內）"
    )

    output_tools = [{
        "name": "output_news_analysis",
        "description": "輸出結構化新聞分析結果",
        "input_schema": {
            "type": "object",
            "properties": {
                "news_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title":           {"type": "string", "description": "新聞標題（25字內）"},
                            "summary":         {"type": "string", "description": "新聞摘要（60字內）"},
                            "affected_stocks": {"type": "string", "description": "影響的台股標的及說明"},
                            "impact":          {"type": "string", "enum": ["正面", "負面", "中性"]},
                        },
                        "required": ["title", "summary", "affected_stocks", "impact"],
                    },
                },
                "key_alert": {"type": "string", "description": "今日開盤前最重要提示（30字內）"},
            },
            "required": ["news_items", "key_alert"],
        },
    }]

    try:
        analysis_resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            tools=output_tools,
            tool_choice={"type": "tool", "name": "output_news_analysis"},
            messages=[{"role": "user", "content": analysis_prompt}],
        )
        for block in analysis_resp.content:
            if block.type == "tool_use" and block.name == "output_news_analysis":
                result = block.input
                log.info(f"  ✅ 新聞分析完成：{len(result.get('news_items', []))} 則")
                return result
    except Exception as e:
        log.warning(f"  新聞結構化分析失敗：{e}")

    return {
        "news_items": [],
        "key_alert": "新聞分析暫時無法取得，請自行留意盤前消息",
    }


# ══════════════════════════════════════════════════════════════
# 技術指標計算
# ══════════════════════════════════════════════════════════════

def calc_kd(df: pd.DataFrame, n: int = 9) -> tuple[float, float]:
    """
    計算 KD 值
    df 必須包含 max（最高）、min（最低）、close（收盤）欄位
    回傳 (K值, D值)
    """
    if len(df) < n:
        return 50.0, 50.0

    df = df.copy()
    df["low_n"]  = df["min"].rolling(n).min()
    df["high_n"] = df["max"].rolling(n).max()
    df["rsv"] = (
        (df["close"] - df["low_n"]) /
        (df["high_n"] - df["low_n"] + 1e-8) * 100
    )

    k_val, d_val = 50.0, 50.0
    for rsv in df["rsv"].dropna():
        k_val = 2 / 3 * k_val + 1 / 3 * rsv
        d_val = 2 / 3 * d_val + 1 / 3 * k_val

    return round(k_val, 2), round(d_val, 2)


def calc_atr(df: pd.DataFrame, n: int = 14) -> "float | None":
    """計算 ATR(n)（Average True Range，平均真實區間）

    df 必須包含 max（最高）、min（最低）、close（收盤）欄位，且已依日期排序。
    True Range = max(當日高低差, |當日高 − 前收|, |當日低 − 前收|)
    ATR = 最近 n 日 True Range 的簡單移動平均。

    資料筆數不足（< n + 1，因為需要前一日收盤）時回傳 None，
    由呼叫端 fallback 回固定比例停損停利。
    """
    need = {"max", "min", "close"}
    if not need.issubset(df.columns) or len(df) < n + 1:
        return None

    high  = pd.to_numeric(df["max"],   errors="coerce")
    low   = pd.to_numeric(df["min"],   errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(n).mean().iloc[-1]
    if pd.isna(atr) or atr <= 0:
        return None
    return float(round(atr, 2))


def calc_prev_day_shape(df_price: pd.DataFrame) -> dict:
    """算「前一日」（＝目前資料最後一個交易日）的振幅%與上影線占振幅比例

    振幅% = (最高－最低) / 前一交易日收盤 × 100（標準振幅定義）
    上影線占振幅比例 = (最高－max(開盤,收盤)) / (最高－最低) × 100

    只讀 df_price（用還原後的 max/min/open/close），不修改任何既有欄位。
    """
    out = {"prev_day_amplitude_pct": None, "prev_day_upper_shadow_pct": None}
    if len(df_price) < 2:
        return out

    last = df_price.iloc[-1]
    prev = df_price.iloc[-2]
    try:
        high, low   = float(last["max"]), float(last["min"])
        open_, close = float(last["open"]), float(last["close"])
        prev_close  = float(prev["close"])
    except (KeyError, TypeError, ValueError):
        return out

    if any(pd.isna(x) for x in (high, low, open_, close, prev_close)) or not prev_close:
        return out

    rng = high - low
    out["prev_day_amplitude_pct"] = float(round(rng / prev_close * 100, 2))
    out["prev_day_upper_shadow_pct"] = (
        float(round((high - max(open_, close)) / rng * 100, 2)) if rng > 0 else 0.0
    )
    return out


def calc_net_pct_of_volume(net_lots: "int | None", volume_shares: "float | None") -> "float | None":
    """net_lots 單位「張」（1 張＝1000 股），volume_shares 單位「股」——兩者
    在 FinMind 原始資料裡單位不同，這裡先把 net_lots 換算成股再取比例。
    呼叫端須確保 net_lots 與 volume_shares 是同一個觀察期間（例如都用近 5
    日）加總後的值，否則比例沒有意義。
    """
    if net_lots is None or not volume_shares:
        return None
    return float(round(net_lots * 1000 / volume_shares * 100, 3))


def calc_institutional_5d_aligned(df_sub: pd.DataFrame, df_price: pd.DataFrame) -> dict:
    """把單一法人（外資或投信）的日買賣超序列（date, net）跟 df_price
    （date, Trading_Volume）依日期合併，取雙方都有資料的最近 5 個交易日，
    用同一組日期分別加總 net 與成交量——避免法人序列跟股價序列各自缺漏
    的日期不同，導致「近5日」實際上指的是不同的日期範圍。

    回傳 {"net_5d": ..., "pct_of_volume": ...}：兩者口徑一致，都要求合併
    後剛好湊滿 5 個共同交易日才計算；不足 5 天時「近5日」本來就不成立，
    net_5d 與 pct_of_volume 一起留 None，不會出現「net_5d 是 3 天的合計，
    卻標成近5日」這種名不符實的情況。
    """
    if df_sub is None or df_sub.empty:
        return {"net_5d": None, "pct_of_volume": None}

    merged = pd.merge(
        df_sub[["date", "net"]],
        df_price[["date", "Trading_Volume"]],
        on="date", how="inner",
    ).sort_values("date")

    if len(merged) < 5:
        return {"net_5d": None, "pct_of_volume": None}

    tail = merged.tail(5)
    net_5d = int(round(tail["net"].sum()))
    vol_5d = float(tail["Trading_Volume"].sum())
    return {"net_5d": net_5d, "pct_of_volume": calc_net_pct_of_volume(net_5d, vol_5d)}


# ══════════════════════════════════════════════════════════════
# 除權息還原（避免 MA/KD/ATR 被未還原股價的價格斷層扭曲）
# ══════════════════════════════════════════════════════════════

def get_dividend_events(stock_id: str, start_date: str) -> "tuple[pd.DataFrame, str | None]":
    """取得除權息事件（現金股利／股票股利，含盈餘或公積轉增資）

    回傳 (events_df, error_msg)：
    - error_msg 為 None 代表 API 呼叫成功，即使 events_df 是空的（這檔股票
      在查詢區間內單純沒有除權息）也算成功，不是失敗。
    - error_msg 非 None 代表重試 2 次（間隔 1 秒，共 3 次嘗試）後 API 呼叫
      仍然失敗（HTTP 例外或 FinMind status != 200），此時 events_df 必為
      空。呼叫端不可把這種情況誤判成「沒有除權息」而靜默略過還原——必須
      另外標示 data_warning。

    FinMind TaiwanStockDividendResult 的 date 欄位即為實際除權息交易日，
    before_price/after_price 是當天真正生效的除權息前後基準價，可直接拿來
    反推還原因子（after_price / before_price）。stock_or_cache_dividend
    欄位標示事件類型："息"＝現金股利除息，"權"＝股票股利（含盈餘/公積轉
    增資）除權——不是現金增資（現金增資另有 CashIncreaseSubscriptionRate
    等欄位，在 TaiwanStockDividend 資料集裡，且與此欄位無關）。
    """
    df, err = pd.DataFrame(), None
    max_attempts = 3  # 1 次原始嘗試 + 2 次重試
    for attempt in range(1, max_attempts + 1):
        df, err = _finmind_request("TaiwanStockDividendResult", stock_id, start_date)
        if err is None:
            break
        if attempt < max_attempts:
            log.warning(
                f"FinMind TaiwanStockDividendResult ({stock_id}) "
                f"第 {attempt} 次失敗：{err}，1 秒後重試"
            )
            time.sleep(1)

    if err:
        log.warning(f"FinMind TaiwanStockDividendResult ({stock_id})：重試 {max_attempts - 1} 次後仍失敗：{err}")
        return df, err

    if df.empty:
        return df, None

    df = df.copy()
    df["date"] = df["date"].astype(str)
    for col in ["before_price", "after_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["before_price", "after_price"])
    df = df[df["before_price"] > 0]

    return df.sort_values("date").reset_index(drop=True), None


def apply_price_adjustment(df_price: pd.DataFrame, div_events: pd.DataFrame) -> pd.DataFrame:
    """用除權息事件的 before/after 基準價，把股價還原成連續序列（向前還原法）

    以「最新一筆價格為基準」：對每一筆除權息事件，事件日「之前」的所有歷史
    價格都乘上 after_price / before_price 的比例；事件日當天與之後的價格
    維持原始成交價不變。因此今日（最新）收盤價、進場區間、停損價永遠等於
    實際盤面報價，不會與券商報價脫鉤——只有歷史區間被壓縮，藉此消除除權息
    造成的價格斷層，讓 MA/KD/ATR 能在連續價格上計算。
    """
    df = df_price.copy()
    if div_events is None or div_events.empty:
        return df

    factor = pd.Series(1.0, index=df.index)
    for _, ev in div_events.iterrows():
        ratio = ev["after_price"] / ev["before_price"]
        if pd.isna(ratio) or ratio <= 0:
            continue
        factor.loc[df["date"] < ev["date"]] *= ratio

    for col in ["close", "max", "min", "open"]:
        if col in df.columns:
            df[col] = df[col] * factor

    return df


def get_recent_dividend_flag(
    df_price: pd.DataFrame, div_events: pd.DataFrame, n: int = RECENT_DIVIDEND_WINDOW_DAYS
) -> "dict | None":
    """判斷最近一筆除權息事件是否落在近 n 個交易日內，供報告顯示提醒旗標"""
    if div_events is None or div_events.empty or df_price.empty:
        return None

    dates = df_price["date"].reset_index(drop=True)
    latest_event = div_events.iloc[-1]
    ev_date = latest_event["date"]

    matches = dates[dates == ev_date]
    if matches.empty:
        return None

    trading_days_ago = (len(dates) - 1) - matches.index[-1]
    if trading_days_ago < 0 or trading_days_ago > n:
        return None

    ratio = float(latest_event["after_price"]) / float(latest_event["before_price"])
    return {
        "date": ev_date,
        "type": str(latest_event.get("stock_or_cache_dividend", "") or "除權息"),
        "before_price": float(latest_event["before_price"]),
        "after_price": float(latest_event["after_price"]),
        "chg_pct": float(round((ratio - 1) * 100, 2)),
        "trading_days_ago": int(trading_days_ago),
    }


def detect_price_gaps(
    df_price: pd.DataFrame,
    lookback: int = PRICE_GAP_LOOKBACK_DAYS,
    threshold_pct: float = PRICE_GAP_THRESHOLD_PCT,
) -> list:
    """在還原後的收盤價序列上，掃描近 lookback 個交易日內是否仍有單日變動
    超過 threshold_pct% 的斷層

    只做標示（data_warning），不改變 score_stock / calc_trade_levels 的計算
    邏輯——分數與停損停利仍照常算出、寫入 result。但報告端（
    compose_email_html）看到 data_warning 後，會把該股票整檔從推薦進場／
    觀望／暫不關注清單中排除，且不顯示分數與進場區間、停損停利，只留昨收、
    漲跌幅與警示文字讓人工判斷——不是「算出來但加註不可信」，而是報告畫面
    直接不顯示這些可能失真的數字。
    """
    if df_price.empty or len(df_price) < 2:
        return []

    tail = df_price.tail(lookback + 1).reset_index(drop=True)
    closes = pd.to_numeric(tail["close"], errors="coerce")
    pct_chg = closes.pct_change() * 100

    gaps = []
    for i in range(1, len(tail)):
        p = pct_chg.iloc[i]
        if pd.notna(p) and abs(p) > threshold_pct:
            gaps.append({
                "date": str(tail["date"].iloc[i]),
                "pct": float(round(p, 2)),
            })
    return gaps


# ══════════════════════════════════════════════════════════════
# 評分與交易價位（Python 端計算，取代 LLM 算術）
# ══════════════════════════════════════════════════════════════

def score_stock(s: dict) -> dict:
    """依固定規則在 Python 端直接計算評分，避免 LLM 算術誤差（滿分 10 分）

    規則與原本寫在 Claude prompt 內的完全一致：
      基本面（max 2）：avg_yoy≥30%→2，15–30%→1，其他→0
      籌碼面（max 4）：外資買超 +1，外資連買≥3日再 +1，投信買超 +1，投信連買≥3日再 +1
      技術面（max 4）：站上MA5 +1，站上MA20 +1，KD黃金交叉且K<80再 +1，
                      量比>1.2且當日收紅再 +1
      total_score = 三項加總；recommend = total_score ≥ 7
    """
    # ── 基本面（max 2）──
    avg_yoy = s.get("avg_yoy")
    if avg_yoy is None:
        fundamental = 0
    elif avg_yoy >= 30:
        fundamental = 2
    elif avg_yoy >= 15:
        fundamental = 1
    else:
        fundamental = 0

    # ── 籌碼面（max 4）──
    chip = 0
    if s.get("foreign_net", 0) > 0:
        chip += 1
        if s.get("foreign_consec", 0) >= 3:
            chip += 1
    if s.get("trust_net", 0) > 0:
        chip += 1
        if s.get("trust_consec", 0) >= 3:
            chip += 1

    # ── 技術面（max 4）──
    technical = 0
    if s.get("above_ma5"):
        technical += 1
    if s.get("above_ma20"):
        technical += 1
    if s.get("kd_cross") and s.get("K", 50.0) < 80:
        technical += 1
    if s.get("volume_ratio", 1.0) > 1.2 and (s.get("price_chg_pct") or 0) > 0:
        technical += 1

    total = fundamental + chip + technical
    return {
        "fundamental_score": int(fundamental),
        "chip_score":        int(chip),
        "technical_score":   int(technical),
        "total_score":       int(total),
        "recommend":         bool(total >= 7),
    }


def calc_trade_levels(close, atr) -> dict:
    """依 ATR 計算停損／停利價位（取一位小數）

      停損   = close − 1.5 × ATR
      target1 = close + 2 × ATR（先出一半）
      target2 = close + 3 × ATR（全出）

    ATR 為 None 或非正值（資料不足）時 fallback 回原本固定比例：
      停損 close×0.95、target1 close×1.08、target2 close×1.12
    """
    if close is None or close <= 0:
        return {
            "stop_loss": "N/A", "target1": "N/A", "target2": "N/A",
            "level_basis": "無收盤價",
        }

    if atr and atr > 0:
        stop = close - 1.5 * atr
        t1   = close + 2.0 * atr
        t2   = close + 3.0 * atr
        basis = f"ATR={atr:.2f}（停損 −1.5×、目標 +2×/+3×）"
    else:
        stop = close * 0.95
        t1   = close * 1.08
        t2   = close * 1.12
        basis = "固定比例 −5%/+8%/+12%（ATR 資料不足）"

    return {
        "stop_loss":   f"{stop:.1f}",
        "target1":     f"{t1:.1f}",
        "target2":     f"{t2:.1f}",
        "level_basis": basis,
    }


def classify_stock(s: dict) -> str:
    """依 data_warning 與 total_score，回傳這檔股票所屬分類：
    「警示」／「推薦」／「觀望」／「暫不關注」

    email（compose_email_html）與訊號日誌（signals_log）共用這個函式，
    確保兩邊口徑一致。門檻與原本寫在 compose_email_html 裡的完全相同
    （≥7 推薦、5–6 觀望、<5 暫不關注，有 data_warning 一律歸警示），只是
    把判斷抽成共用函式，沒有改變任何規則，也沒有動 score_stock 本身。
    """
    if s.get("data_warning"):
        return "警示"
    total = s.get("total_score", 0)
    if total >= 7:
        return "推薦"
    elif total >= 5:
        return "觀望"
    else:
        return "暫不關注"


# ══════════════════════════════════════════════════════════════
# 資料抓取
# ══════════════════════════════════════════════════════════════

def get_futures_gate() -> dict:
    """抓取台指期（TX）近月合約，比較夜盤收盤 vs 前一交易日結算價，回傳閘門狀態

    FinMind 的 TaiwanFuturesDaily 每個交易日會回傳多筆資料：
    - 多個到期月合約（近月、遠月）以及跨月價差合約（contract_date 含 "/"）
    - 每個合約再拆成 trading_session = "position"（日盤）與 "after_market"（夜盤）
    - settlement_price（結算價）只有日盤（position）那筆會有值，夜盤固定是 0
    因此必須先篩選出「近月合約」，分別取最新一筆夜盤收盤，以及在夜盤日期
    之前最近一個交易日的日盤結算價，兩者相減才是正確的夜盤閘門判斷。
    """
    log.info("抓取台指期（TX）夜盤資料...")

    start = (datetime.today() - timedelta(days=10)).strftime("%Y-%m-%d")
    df = finmind_get("TaiwanFuturesDaily", "TX", start)

    result = {
        "status":      "green",
        "emoji":       "🟢",
        "label":       "綠燈",
        "action":      "可依評分正常操作",
        "night_close": None,
        "settlement":  None,
        "diff_pts":    None,
        "diff_pct":    None,
        "error":       None,
    }

    if df.empty:
        result["error"] = "無法取得台指期資料，閘門預設綠燈"
        log.warning("⚠️  台指期（TX）無資料，夜盤閘門預設綠燈")
        return result

    required_cols = {"date", "contract_date", "close", "settlement_price", "trading_session"}
    if not required_cols.issubset(df.columns):
        result["error"] = f"台指期資料缺少欄位（現有：{list(df.columns)}），閘門預設綠燈"
        log.warning(f"⚠️  {result['error']}")
        return result

    df = df.copy()
    df["contract_date"] = df["contract_date"].astype(str)
    # 排除跨月價差合約，只保留單一到期月的合約
    df = df[~df["contract_date"].str.contains("/")]
    for col in ["close", "settlement_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df_after    = df[(df["trading_session"] == "after_market") & (df["close"] > 0)]
    df_position = df[(df["trading_session"] == "position") & (df["settlement_price"] > 0)]

    if df_after.empty or df_position.empty:
        result["error"] = "台指期收盤價或結算價無效，閘門預設綠燈"
        log.warning(f"⚠️  {result['error']}")
        return result

    # 夜盤：取近月合約（contract_date 最小）最新一筆有效收盤
    after_date = df_after["date"].max()
    after_row  = df_after[df_after["date"] == after_date].sort_values("contract_date").iloc[0]
    night_close = float(after_row["close"])

    # 結算價：取夜盤日期「之前」最近一個交易日的日盤結算（找不到就退而求其次取最新一筆）
    df_position_prior = df_position[df_position["date"] < after_date]
    if df_position_prior.empty:
        df_position_prior = df_position
    settlement_date = df_position_prior["date"].max()
    position_row = df_position_prior[df_position_prior["date"] == settlement_date].sort_values("contract_date").iloc[0]
    settlement = float(position_row["settlement_price"])

    if not night_close or not settlement:
        result["error"] = "台指期收盤價或結算價無效，閘門預設綠燈"
        log.warning(f"⚠️  {result['error']}")
        return result

    diff_pts    = round(night_close - settlement, 0)
    diff_pct    = round((night_close - settlement) / settlement * 100, 2)

    result["night_close"] = night_close
    result["settlement"]  = settlement
    result["diff_pts"]    = diff_pts
    result["diff_pct"]    = diff_pct

    if diff_pts <= -200:
        result.update({"status": "red",    "emoji": "🔴", "label": "紅燈", "action": "今日暫停所有新進場"})
    elif diff_pts <= -100:
        result.update({"status": "yellow", "emoji": "🟡", "label": "黃燈", "action": "降低倉位"})
    else:
        result.update({"status": "green",  "emoji": "🟢", "label": "綠燈", "action": "可依評分正常操作"})

    sign = "+" if diff_pts >= 0 else ""
    log.info(
        f"  ✅ 台指期夜盤收盤：{night_close:.0f}  結算價：{settlement:.0f}"
        f"  差距：{sign}{diff_pts:.0f}點（{sign}{diff_pct:.2f}%）→ {result['emoji']} {result['label']}"
    )
    return result


def get_market_data() -> dict:
    """抓取台灣加權指數（大盤）資料
    FinMind 正確 data_id 為 'TAIEX'（非 Y9999）
    """
    log.info("抓取大盤指數資料...")
    start = (datetime.today() - timedelta(days=14)).strftime("%Y-%m-%d")
    df = finmind_get("TaiwanStockPrice", "TAIEX", start)

    result = {"close": "N/A", "change": "N/A", "change_pct": "N/A", "date": "N/A"}
    if df.empty:
        log.warning("⚠️  大盤指數（TAIEX）無資料")
        return result

    df = df.sort_values("date").reset_index(drop=True)
    for col in ["close", "spread"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    last_row = df.iloc[-1]
    close_val = float(last_row["close"])
    # FinMind 的 spread 欄位就是漲跌點數（已含正負號）
    spread_val = float(last_row.get("spread", 0) or 0)
    chg_p = round(spread_val / (close_val - spread_val) * 100, 2) if (close_val - spread_val) else 0

    result["date"]       = str(last_row["date"])
    result["close"]      = f"{close_val:,.2f}"
    result["change"]     = f"{'+' if spread_val >= 0 else ''}{spread_val:,.2f}"
    result["change_pct"] = f"{'+' if chg_p >= 0 else ''}{chg_p:.2f}%"

    log.info(f"  ✅ 大盤：{result['close']} 點  漲跌：{result['change']}（{result['change_pct']}）")
    return result


def get_stock_data(stock_id: str) -> "tuple[dict, pd.DataFrame]":
    """
    抓取單一股票所需的所有資料：
    - 昨日收盤價與成交量
    - MA5 / MA20 / KD
    - 外資、投信買賣超
    - 近三個月營收年增率

    回傳 (result, price_series)：
    - result：原本的欄位字典，供 analyze_with_claude／compose_email_html 使用。
    - price_series：這次抓到、已還原的股價序列（date/open/max/min/close），
      只給 signals_log 的 backfill_outcomes() 用來回填舊訊號的後續報酬，
      不併入 result，也不會被送進 Claude 提示詞或 email（呼叫端不傳這個值
      給 analyze_with_claude／compose_email_html 就自然保證這點）。資料不足
      或發生錯誤時回傳空的 DataFrame。
    """
    name  = STOCKS[stock_id]
    log.info(f"  抓取 {name}（{stock_id}）...")

    today    = datetime.today()
    start_90 = (today - timedelta(days=90)).strftime("%Y-%m-%d")   # 技術指標 / 法人用
    # 營收需要抓到去年同期：最新3個月 + 去年同3個月 ≈ 14個月，取 430 天保證涵蓋
    start_14m = (today - timedelta(days=430)).strftime("%Y-%m-%d")

    result = {
        "stock_id": stock_id,
        "name": name,
        "close": None,
        "volume": None,
        "volume_ratio": 1.0,
        "price_chg": None,
        "price_chg_pct": None,
        "ma5": None,
        "ma20": None,
        "above_ma5": False,
        "above_ma20": False,
        "K": 50.0,
        "D": 50.0,
        "kd_cross": False,
        "atr": None,
        "foreign_net": 0,
        "trust_net": 0,
        "foreign_consec": 0,
        "trust_consec": 0,
        "foreign_net_5d": None,
        "trust_net_5d": None,
        "foreign_net_pct_of_volume": None,
        "trust_net_pct_of_volume": None,
        "prev_day_amplitude_pct": None,
        "prev_day_upper_shadow_pct": None,
        "revenue_yoy_list": [],
        "avg_yoy": None,
        "dividend_flag": None,
        "data_warning": [],
        "error": None,
    }

    # ── 1. 股價資料 ────────────────────────────────────────────────
    df_price = finmind_get("TaiwanStockPrice", stock_id, start_90)
    if df_price.empty:
        result["error"] = "無法取得股價資料"
        return result, pd.DataFrame()

    df_price = df_price.sort_values("date").reset_index(drop=True)
    for col in ["close", "max", "min", "open", "Trading_Volume"]:
        if col in df_price.columns:
            df_price[col] = pd.to_numeric(df_price[col], errors="coerce")

    if len(df_price) < 2:
        result["error"] = "股價資料筆數不足"
        return result, pd.DataFrame()

    # ── 除權息還原 ────────────────────────────────────────────────
    # TaiwanStockPrice 是未還原股價，除權息當天會出現價格斷層，直接拿來算
    # MA/KD/ATR 會嚴重失真（例：6669 緯穎 2026/09/02 股票股利（盈餘/公積
    # 轉增資）除權，FinMind stock_or_cache_dividend="權"，未還原股價單日
    # 「跌」66.5%，MA20/ATR 因此完全脫離現價；不是現金增資）。用
    # TaiwanStockDividendResult 的 before/after 基準價反推還原因子，對事件
    # 日之前的歷史價格做「向前還原」——今日收盤價維持原始報價不變。
    time.sleep(0.3)  # 避免 API rate limit
    div_events, div_err = get_dividend_events(stock_id, start_90)
    if div_err:
        result["data_warning"].append({
            "type": "dividend_fetch_failed",
            "message": "除權息資料取得失敗，指標可能未還原",
            "detail": div_err,
        })
    result["dividend_flag"] = get_recent_dividend_flag(df_price, div_events)
    df_price = apply_price_adjustment(df_price, div_events)

    # 還原完成後再檢查一次：近 60 個交易日內若仍有單日變動超過 11%，代表還原
    # 可能不完整（或有其他未涵蓋的事件），只標示不改分數。
    gaps = detect_price_gaps(df_price)
    if gaps:
        result["data_warning"].append({
            "type": "price_gap",
            "message": "疑似價格斷層，技術面分數與停損停利不可信",
            "gaps": gaps,
        })

    last_row = df_price.iloc[-1]
    prev_row = df_price.iloc[-2]

    result["close"]  = float(last_row["close"])
    result["volume"] = int(last_row.get("Trading_Volume", 0) or 0)

    prev_close = float(prev_row["close"])
    result["price_chg"]     = float(round(result["close"] - prev_close, 2))
    result["price_chg_pct"] = float(round(
        (result["close"] - prev_close) / prev_close * 100, 2
    )) if prev_close else 0.0

    # 量比（今日 vs 前5日均量）—— 明確轉為 Python float
    if len(df_price) >= 6:
        avg5 = float(df_price["Trading_Volume"].iloc[-6:-1].mean())
        result["volume_ratio"] = float(round(
            result["volume"] / avg5, 2
        )) if avg5 and avg5 > 0 else 1.0

    # MA5 / MA20（明確轉換為 Python float/bool，避免 numpy 型別 JSON 序列化問題）
    closes = df_price["close"].dropna()
    if len(closes) >= 5:
        result["ma5"]       = float(round(float(closes.tail(5).mean()), 2))
        result["above_ma5"] = bool(result["close"] > result["ma5"])
    if len(closes) >= 20:
        result["ma20"]       = float(round(float(closes.tail(20).mean()), 2))
        result["above_ma20"] = bool(result["close"] > result["ma20"])

    # KD（明確轉換 bool）
    if {"max", "min", "close"}.issubset(df_price.columns):
        k, d = calc_kd(df_price)
        result["K"]        = float(k)
        result["D"]        = float(d)
        result["kd_cross"] = bool(k > d)

    # ATR(14)：供 ATR 動態停損停利使用；資料不足時為 None，由下游 fallback 回固定比例
    result["atr"] = calc_atr(df_price)

    # 前一日振幅% / 上影線占振幅比例（只讀 df_price，供 signals_log 用）。
    # 這個計算只是為了寫日誌，出錯不可讓報告本身的流程跟著掛掉，抓到例外
    # 就留 None、記警告，繼續往下跑。
    try:
        result.update(calc_prev_day_shape(df_price))
    except Exception as e:
        log.warning(f"    [{name}] calc_prev_day_shape 失敗（不影響報告，供 signals_log 用）：{e}")
        result["prev_day_amplitude_pct"] = None
        result["prev_day_upper_shadow_pct"] = None

    # ── 2. 三大法人（外資 + 投信）─────────────────────────────────
    time.sleep(0.3)  # 避免 API rate limit
    # 正確的 FinMind 資料集名稱（注意末尾有 BuySell）
    df_inst = finmind_get("TaiwanStockInstitutionalInvestorsBuySell", stock_id, start_90)

    if not df_inst.empty:
        df_inst = df_inst.sort_values("date")

        for inst_name, key in [
            ("Foreign_Investor", "foreign"),
            ("Investment_Trust", "trust"),
        ]:
            df_sub = df_inst[df_inst["name"] == inst_name].copy()
            if df_sub.empty:
                continue

            df_sub["buy"]  = pd.to_numeric(df_sub.get("buy",  0), errors="coerce").fillna(0)
            df_sub["sell"] = pd.to_numeric(df_sub.get("sell", 0), errors="coerce").fillna(0)
            # FinMind 買賣數量單位為「股」，除以 1000 轉換為「張」
            df_sub["net"]  = (df_sub["buy"] - df_sub["sell"]) / 1000

            result[f"{key}_net"] = int(round(df_sub["net"].iloc[-1]))

            # 近5日買賣超與近5日成交量占比：法人序列跟 df_price 依日期合併
            # 後取共同的最近 5 個交易日，兩邊「5天」指同一組日期，不是各自
            # tail(5)——法人資料偶爾有缺漏日，各自 tail(5) 可能對不上同一段
            # 期間。df_sub 這次呼叫本來就抓了 90 天的每日序列，不用多打 API。
            # 這也只是為了寫日誌，出錯不可影響報告本身的流程。
            try:
                aligned = calc_institutional_5d_aligned(df_sub, df_price)
            except Exception as e:
                log.warning(
                    f"    [{name}] calc_institutional_5d_aligned（{key}）失敗"
                    f"（不影響報告，供 signals_log 用）：{e}"
                )
                aligned = {"net_5d": None, "pct_of_volume": None}
            result[f"{key}_net_5d"] = aligned["net_5d"]
            result[f"{key}_net_pct_of_volume"] = aligned["pct_of_volume"]

            # 計算連續買超天數（從最新往回數）
            consec = 0
            for net in reversed(df_sub["net"].values):
                if net > 0:
                    consec += 1
                else:
                    break
            result[f"{key}_consec"] = consec

    # ── 3. 月營收年增率（近三個月）────────────────────────────────
    time.sleep(0.3)
    df_rev = finmind_get("TaiwanStockMonthRevenue", stock_id, start_14m)

    if not df_rev.empty:
        df_rev = df_rev.sort_values("date").reset_index(drop=True)
        for col in ["revenue", "revenue_year", "revenue_month"]:
            if col in df_rev.columns:
                df_rev[col] = pd.to_numeric(df_rev[col], errors="coerce")

        recent3 = df_rev.tail(3)
        yoy_list = []

        for _, row in recent3.iterrows():
            yr  = int(row.get("revenue_year",  0) or 0)
            mo  = int(row.get("revenue_month", 0) or 0)
            rev = float(row.get("revenue", 0) or 0)

            # 找去年同月
            same_mo_prev = df_rev[
                (df_rev["revenue_year"]  == yr - 1) &
                (df_rev["revenue_month"] == mo)
            ]
            if not same_mo_prev.empty:
                prev_rev = float(same_mo_prev["revenue"].values[0] or 0)
                yoy = round((rev - prev_rev) / prev_rev * 100, 2) if prev_rev else None
            else:
                yoy = None

            yoy_list.append({
                "month": f"{yr}/{mo:02d}",
                "yoy":   yoy,
            })

        result["revenue_yoy_list"] = yoy_list
        valid_yoy = [x["yoy"] for x in yoy_list if x["yoy"] is not None]
        result["avg_yoy"] = round(sum(valid_yoy) / len(valid_yoy), 2) if valid_yoy else None

    # ── 除錯：印出本檔股票關鍵資料摘要 ────────────────────────────
    rev_str = " | ".join(
        f"{x['month']} YoY={x['yoy']}%" if x["yoy"] is not None else f"{x['month']} N/A"
        for x in result.get("revenue_yoy_list", [])
    ) or "無"
    log.info(
        f"    [{name}] 收盤={result['close']} 漲跌={result['price_chg_pct']}%"
        f" MA5={result['ma5']} MA20={result['ma20']}"
        f" K={result['K']} D={result['D']} ATR={result['atr']}"
        f" 外資={result['foreign_net']:+d}張(連{result['foreign_consec']}日)"
        f" 投信={result['trust_net']:+d}張(連{result['trust_consec']}日)"
        f" 營收YoY: {rev_str}"
        f" 均YoY={result['avg_yoy']}"
    )
    if result["dividend_flag"]:
        df_ = result["dividend_flag"]
        log.info(
            f"    [{name}] ⚡ 近期除權息：{df_['date']}（{df_['trading_days_ago']} 個交易日前）"
            f" {df_['type']} {df_['before_price']}→{df_['after_price']}"
            f"（{df_['chg_pct']:+.2f}%），MA/KD/ATR 已依還原股價計算"
        )
    if stock_id in EVENT_FLAGS:
        log.warning(f"    [{name}] ⚠️  個股警示：{EVENT_FLAGS[stock_id]}")
    for w in result["data_warning"]:
        gap_str = ""
        if w.get("gaps"):
            gap_str = "；" + "、".join(f"{g['date']} {g['pct']:+.1f}%" for g in w["gaps"])
        log.warning(f"    [{name}] 🚧 資料品質警示：{w['message']}{gap_str}")

    price_series = df_price[["date", "open", "max", "min", "close"]].copy()
    return result, price_series


# ══════════════════════════════════════════════════════════════
# Claude AI 分析
# ══════════════════════════════════════════════════════════════

def analyze_with_claude(stocks_data: list, market_data: dict) -> dict:
    """
    評分與交易價位在 Python 端直接計算（score_stock / calc_trade_levels），
    Claude API 只負責產生文字說明（各面向理由、進場區間、風險提示等），
    最後由本函式把系統計算的數值覆蓋回 Claude 的輸出，確保分數與價位精確。
    """
    log.info("計算評分（Python）＋呼叫 Claude API 產生文字說明...")

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    # 準備給 Claude 的資料摘要（全部轉為 Python 原生型別）
    stocks_summary = []
    scored_by_id = {}   # stock_id -> {評分欄位..., stop_loss, target1, target2, level_basis}
    flagged_sids = set()  # data_warning 非空的股票代號：不送進 Claude prompt，
                           # 避免 market_summary／overall_foreign_trend／risk_warning
                           # 這幾個跨股票彙總欄位引用到可能失真的分數或指標
    for s in stocks_data:
        if s.get("error"):
            stocks_summary.append({
                "stock_id": s["stock_id"],
                "name": s["name"],
                "error": s["error"],
            })
            continue

        sid = str(s["stock_id"])
        if s.get("data_warning"):
            flagged_sids.add(sid)

        rev_str = "、".join([
            f"{x['month']} YoY={x['yoy']}%" if x["yoy"] is not None else f"{x['month']} YoY=N/A"
            for x in s.get("revenue_yoy_list", [])
        ]) or "無資料"

        if not s.get("data_warning"):
            stocks_summary.append({
                "stock_id":            sid,
                "name":                str(s["name"]),
                "close":               float(s["close"]) if s["close"] is not None else None,
                "price_chg_pct":       float(s["price_chg_pct"]) if s["price_chg_pct"] is not None else 0.0,
                "volume_ratio":        float(s["volume_ratio"]) if s["volume_ratio"] is not None else 1.0,
                "ma5":                 float(s["ma5"]) if s["ma5"] is not None else None,
                "ma20":                float(s["ma20"]) if s["ma20"] is not None else None,
                "above_ma5":           bool(s["above_ma5"]),
                "above_ma20":          bool(s["above_ma20"]),
                "K":                   float(s["K"]),
                "D":                   float(s["D"]),
                "kd_cross":            bool(s["kd_cross"]),
                "atr":                 float(s["atr"]) if s.get("atr") is not None else None,
                "foreign_net_lots":    int(s["foreign_net"]),
                "trust_net_lots":      int(s["trust_net"]),
                "foreign_consec_days": int(s["foreign_consec"]),
                "trust_consec_days":   int(s["trust_consec"]),
                "revenue_3m":          str(rev_str),
                "avg_yoy_pct":         float(s["avg_yoy"]) if s["avg_yoy"] is not None else None,
            })

        # ── Python 端直接計算評分與 ATR 停損停利（取代 LLM 算術）── 對每一檔都要
        # 執行，不受 data_warning 影響：final_stocks／stock_card 仍需要這些數字，
        # 只是報告端（compose_email_html）依 data_warning 決定顯不顯示，不是這裡。
        levels = calc_trade_levels(
            float(s["close"]) if s.get("close") is not None else None,
            s.get("atr"),
        )
        scored_by_id[sid] = {**score_stock(s), **levels}

    # 傳給 Claude 的評分結果（只含數字，供其撰寫理由時對照）——排除 data_warning
    # 股票，理由同上：避免彙總文字引用到可能失真的分數。
    scores_for_prompt = {
        sid: {k: v[k] for k in (
            "fundamental_score", "chip_score", "technical_score",
            "total_score", "recommend",
        )}
        for sid, v in scored_by_id.items()
        if sid not in flagged_sids
    }

    prompt = f"""你是台股量化分析師，請用繁體中文為以下股票撰寫分析說明，並呼叫 output_analysis 工具。

大盤：加權指數 {market_data.get('close','N/A')} 點，漲跌 {market_data.get('change','N/A')}（{market_data.get('change_pct','N/A')}）

股票資料（JSON）：
{json.dumps(stocks_summary, ensure_ascii=False, indent=2)}

【評分結果 —— 已由系統依固定規則計算完成，數字不可更改，請依這些分數撰寫理由】
{json.dumps(scores_for_prompt, ensure_ascii=False, indent=2)}

評分規則（僅供你撰寫理由時對照，實際分數以上方系統計算為準，滿分10分）：
基本面（max 2）：avg_yoy_pct≥30%→2分，15-30%→1分，其他→0分
籌碼面（max 4）：外資買超+1，外資連買≥3日再+1，投信買超+1，投信連買≥3日再+1
技術面（max 4）：above_ma5=true+1，above_ma20=true+1，kd_cross=true且K<80再+1，volume_ratio>1.2且price_chg_pct>0再+1

【輸出要求】
- 逐檔輸出 fundamental_reason / chip_reason / technical_reason，內容需與系統給的分數一致，各限50字以內
- entry_range：建議進場區間（參考昨收價，例如「95.0–97.5」）
- summary限30字以內；watch_conditions：未達推薦門檻（總分<7）時要觀察的改善條件
- 停損停利價位由系統以 ATR 計算，你不需輸出
- market_summary/risk_warning/overall_foreign_trend各限60字以內"""

    # ── 使用 Tool Use 確保輸出合法 JSON（最可靠方式）────────────────
    tools = [
        {
            "name": "output_analysis",
            "description": "輸出台股 AI 分析結果的結構化資料",
            "input_schema": {
                "type": "object",
                "properties": {
                    "stocks": {
                        "type": "array",
                        "description": "每檔股票的分析結果",
                        "items": {
                            "type": "object",
                            "properties": {
                                "stock_id":           {"type": "string"},
                                "name":               {"type": "string"},
                                "fundamental_reason": {"type": "string", "description": "基本面說明（50字內），須與系統分數一致"},
                                "chip_reason":        {"type": "string", "description": "籌碼面說明（50字內），須與系統分數一致"},
                                "technical_reason":   {"type": "string", "description": "技術面說明（50字內），須與系統分數一致"},
                                "entry_range":        {"type": "string", "description": "建議進場區間，參考昨收價"},
                                "summary":            {"type": "string", "description": "一句話總結（30字內）"},
                                "watch_conditions":   {"type": "string", "description": "未達推薦門檻時的改善觀察條件"},
                            },
                            "required": [
                                "stock_id", "name",
                                "fundamental_reason", "chip_reason", "technical_reason",
                                "entry_range", "summary", "watch_conditions",
                            ],
                        },
                    },
                    "market_summary":        {"type": "string", "description": "大盤概況說明（50字內）"},
                    "overall_foreign_trend": {"type": "string", "description": "外資在這12檔的整體動向"},
                    "risk_warning":          {"type": "string", "description": "今日操作風險提示（100字內）"},
                },
                "required": ["stocks", "market_summary", "overall_foreign_trend", "risk_warning"],
            },
        }
    ]

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8000,          # ← 從 4096 提升到 8000，防止 8 檔股票詳細分析被截斷
        tools=tools,
        tool_choice={"type": "tool", "name": "output_analysis"},
        messages=[{"role": "user", "content": prompt}],
    )

    # 診斷：若仍然 max_tokens，至少留下警告
    if message.stop_reason == "max_tokens":
        log.warning("⚠️  Claude 回傳被 max_tokens 截斷，部分資料可能不完整")

    # 從 tool_use block 中取出結構化結果
    result = None
    for block in message.content:
        if block.type == "tool_use" and block.name == "output_analysis":
            result = block.input
            break
    if result is None:
        raise ValueError("Claude 未回傳 output_analysis 工具呼叫結果")

    # ── 以 stock_id 對回 Claude 產生的文字說明 ──────────────────
    narr_by_id = {str(s.get("stock_id", "")): s for s in result.get("stocks", [])}

    # ── 重建 stocks：數值一律用 Python 計算結果，文字用 Claude 輸出 ──
    # 分數、推薦、收盤價、ATR 停損停利皆由系統填入，AI 無法覆蓋；
    # 公司名稱一律以 STOCKS 字典為準（Claude 有時會產生錯誤名稱）。
    final_stocks = []
    for s in stocks_data:
        if s.get("error"):
            continue
        sid  = str(s["stock_id"])
        sc   = scored_by_id.get(sid, {})
        narr = narr_by_id.get(sid, {})
        final_stocks.append({
            "stock_id":           sid,
            "name":               STOCKS.get(sid, s.get("name", "")),
            "fundamental_score":  sc.get("fundamental_score", 0),
            "fundamental_reason": narr.get("fundamental_reason", ""),
            "chip_score":         sc.get("chip_score", 0),
            "chip_reason":        narr.get("chip_reason", ""),
            "technical_score":    sc.get("technical_score", 0),
            "technical_reason":   narr.get("technical_reason", ""),
            "total_score":        sc.get("total_score", 0),
            "recommend":          sc.get("recommend", False),
            "close":              float(s["close"]) if s.get("close") is not None else None,
            "entry_range":        narr.get("entry_range", "N/A"),
            "stop_loss":          sc.get("stop_loss", "N/A"),
            "target1":            sc.get("target1", "N/A"),
            "target2":            sc.get("target2", "N/A"),
            "summary":            narr.get("summary", ""),
            "watch_conditions":   narr.get("watch_conditions", ""),
            "level_basis":        sc.get("level_basis", ""),
            "dividend_flag":      s.get("dividend_flag"),
            "event_note":         EVENT_FLAGS.get(sid),
            "data_warning":       s.get("data_warning") or [],
        })

    result["stocks"] = final_stocks

    log.info(f"  評分完成（Python 計算）：{len(final_stocks)} 檔")
    for s in final_stocks:
        log.info(
            f"    {s['name']}({s['stock_id']}): "
            f"總={s['total_score']} 基={s['fundamental_score']} "
            f"籌={s['chip_score']} 技={s['technical_score']} "
            f"推薦={s['recommend']} 停損={s['stop_loss']} 目標={s['target1']}/{s['target2']} "
            f"[{s['level_basis']}]"
        )
    return result


# ══════════════════════════════════════════════════════════════
# 訊號日誌（data/signals_log.csv）與回填（backfill_outcomes）
# ══════════════════════════════════════════════════════════════
#
# 目的：每天把 13 檔的評分與價位存成一列，供事後檢驗系統訊號的實際表現。
# 不影響 score_stock／calc_trade_levels／報告內容——這裡只讀已經算好的
# 結果，寫進 CSV，不會回頭改動評分或報告顯示的任何數字。
#
# repo 目前是 public，這份 CSV 會跟著公開：等於公開這 13 檔每天的評分、
# 進場價位、停損停利、法人籌碼細節、回測報酬。沒有個資/密鑰疑慮，但等於
# 把交易邏輯的具體門檻與歷史表現攤在陽光下，之後如果不想讓策略細節被看到
# 需要重新考慮。

SIGNALS_LOG_COLUMNS = [
    # 基本
    "data_date", "generated_at", "code_version", "stock_id", "name",
    # 結果
    "total_score", "fundamental_score", "chip_score", "technical_score",
    "bucket", "recommend",
    # 報價（一律真實報價，不存還原價）
    "close", "price_chg_pct", "entry_low", "entry_high",
    "stop_loss", "target1", "target2", "atr", "stop_pct", "target1_pct",
    # 籌碼
    "foreign_net_lots", "trust_net_lots", "foreign_consec", "trust_consec",
    "foreign_net_5d", "trust_net_5d",
    "foreign_net_pct_of_volume", "trust_net_pct_of_volume",
    # 技術
    "volume_ratio", "K", "D", "above_ma5", "above_ma20",
    "ma5_dev_pct", "ma20_dev_pct", "return_20d_pct",
    "prev_day_amplitude_pct", "prev_day_upper_shadow_pct",
    # 旗標
    "dividend_flag", "event_note", "data_warning_types",
    # 回填（backfill_outcomes 補）
    "entry_filled", "entry_price", "entry_day_low_hit_stop",
    "ret_t0_pct", "ret_t1_pct", "ret_t5_pct", "ret_t10_pct",
    "period_high", "period_low",
    "hit_target1", "hit_target2", "hit_stop", "ambiguous",
    "backfill_complete",
]

# 上面「回填」那一段的欄位名稱，讀 CSV 回來後要轉成 object dtype（見
# read_signals_log），避免布林值（entry_filled 等）之後要寫回浮點欄位
# （ret_t1_pct 等）或空欄位時被 pandas 的型別推斷擋下來。
SIGNALS_LOG_BACKFILL_COLUMNS = [
    "entry_filled", "entry_price", "entry_day_low_hit_stop",
    "ret_t0_pct", "ret_t1_pct", "ret_t5_pct", "ret_t10_pct",
    "period_high", "period_low",
    "hit_target1", "hit_target2", "hit_stop", "ambiguous",
    "backfill_complete",
]


def get_code_version() -> str:
    """執行當下的程式版本標記：GITHUB_SHA 前 7 碼；本機執行（沒有這個環境
    變數，例如手動測試）時填 "local"，方便事後分辨這一列是哪個版本的程式
    產生的。
    """
    sha = os.environ.get("GITHUB_SHA", "")
    return sha[:7] if sha else "local"


def _safe_float(x) -> "float | None":
    """字串/None/NaN 一律轉成 float 或 None，不拋例外"""
    try:
        if x is None:
            return None
        if isinstance(x, float) and pd.isna(x):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _dev_pct(close, ma) -> "float | None":
    """乖離% = (close - ma) / ma × 100"""
    if close is None or not ma:
        return None
    return float(round((close - ma) / ma * 100, 2))


def calc_20d_return_pct(price_series: pd.DataFrame) -> "float | None":
    """近 20 個交易日報酬%，用還原後的收盤序列算（避免除權息造成假報酬）。
    這是相對報酬率，不是報價，跟「報價一律存真實報價」的規則不衝突——
    只有絕對價格欄位（close/entry_low/entry_high/stop_loss/target1/target2）
    才要求存真實報價。
    """
    if price_series is None or price_series.empty:
        return None
    closes = price_series["close"].dropna()
    if len(closes) < 21:
        return None
    base = float(closes.iloc[-21])
    if not base:
        return None
    return float(round((float(closes.iloc[-1]) - base) / base * 100, 2))


def build_signal_row(
    fs: dict, raw: dict, data_date: str, generated_at: str, code_version: str
) -> dict:
    """把單一股票的訊號組成 signals_log 的一列。

    fs：analyze_with_claude() 輸出的 final_stocks 項目（total_score／
        stop_loss/target1/target2、dividend_flag、event_note、data_warning）。
    raw：get_stock_data() 回傳的 result（K/D/MA/籌碼/振幅等技術與籌碼欄位）。
    只讀 fs／raw，不修改它們既有欄位。return_20d_pct 由呼叫端另外填入
    （需要 price_series，不在 fs／raw 裡）。

    entry_high 固定等於 close（昨收）——跟報告「進場區間」的上緣同一個
    數字，backfill 的成交假設要跟報告一致，不能自己另外發明一個更寬鬆的
    上緣。entry_low = close − 0.5×ATR（ATR 拿不到時退回 close 本身），只
    是參考用的下緣，backfill 的成交判斷完全不看這個欄位。這兩個欄位都跟
    email 裡 Claude 寫的 entry_range 文字是兩回事，是獨立的 Python 決定論
    算法，不影響報告顯示，也不是 calc_trade_levels 的一部分。
    """
    close = _safe_float(fs.get("close"))
    atr   = _safe_float(raw.get("atr"))

    entry_high = close
    entry_low  = None
    if close is not None:
        entry_low = round(close - 0.5 * atr, 2) if atr else close

    def _pct_from_close(level_str):
        lv = _safe_float(level_str)
        if lv is not None and close:
            return round((lv - close) / close * 100, 2)
        return None

    dw_types = ",".join(sorted({
        w.get("type", "") for w in (fs.get("data_warning") or []) if w.get("type")
    }))

    div_flag = raw.get("dividend_flag")
    div_flag_str = f"{div_flag.get('type','')}@{div_flag.get('date','')}" if div_flag else ""

    return {
        "data_date": data_date,
        "generated_at": generated_at,
        "code_version": code_version,
        "stock_id": fs["stock_id"],
        "name": fs["name"],

        "total_score": fs.get("total_score"),
        "fundamental_score": fs.get("fundamental_score"),
        "chip_score": fs.get("chip_score"),
        "technical_score": fs.get("technical_score"),
        "bucket": classify_stock(fs),
        "recommend": fs.get("recommend"),

        "close": close,
        "price_chg_pct": raw.get("price_chg_pct"),
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop_loss": _safe_float(fs.get("stop_loss")),
        "target1": _safe_float(fs.get("target1")),
        "target2": _safe_float(fs.get("target2")),
        "atr": atr,
        "stop_pct": _pct_from_close(fs.get("stop_loss")),
        "target1_pct": _pct_from_close(fs.get("target1")),

        "foreign_net_lots": raw.get("foreign_net"),
        "trust_net_lots": raw.get("trust_net"),
        "foreign_consec": raw.get("foreign_consec"),
        "trust_consec": raw.get("trust_consec"),
        "foreign_net_5d": raw.get("foreign_net_5d"),
        "trust_net_5d": raw.get("trust_net_5d"),
        "foreign_net_pct_of_volume": raw.get("foreign_net_pct_of_volume"),
        "trust_net_pct_of_volume": raw.get("trust_net_pct_of_volume"),

        "volume_ratio": raw.get("volume_ratio"),
        "K": raw.get("K"),
        "D": raw.get("D"),
        "above_ma5": raw.get("above_ma5"),
        "above_ma20": raw.get("above_ma20"),
        "ma5_dev_pct": _dev_pct(close, raw.get("ma5")),
        "ma20_dev_pct": _dev_pct(close, raw.get("ma20")),
        "return_20d_pct": None,  # 呼叫端填入
        "prev_day_amplitude_pct": raw.get("prev_day_amplitude_pct"),
        "prev_day_upper_shadow_pct": raw.get("prev_day_upper_shadow_pct"),

        "dividend_flag": div_flag_str,
        "event_note": fs.get("event_note") or "",
        "data_warning_types": dw_types,

        "entry_filled": None,
        "entry_price": None,
        "entry_day_low_hit_stop": None,
        "ret_t0_pct": None,
        "ret_t1_pct": None,
        "ret_t5_pct": None,
        "ret_t10_pct": None,
        "period_high": None,
        "period_low": None,
        "hit_target1": None,
        "hit_target2": None,
        "hit_stop": None,
        "ambiguous": None,
        "backfill_complete": False,
    }


def read_signals_log(path: str = SIGNALS_LOG_PATH) -> pd.DataFrame:
    """讀取現有的 signals_log.csv；檔案不存在時回傳空的、schema 正確的 DataFrame

    回填欄位讀回來後一律轉成 object dtype：這欄位可能同時存放 True/False
    （entry_filled 等）跟浮點數（ret_t1_pct 等）跟 NaN，如果讓 pandas 依
    CSV 內容自行推斷成 float64（例如某欄目前全空，或全是 0/1），之後
    backfill_outcomes 把 bool 寫進去會被 pandas 的型別檢查擋下來或悄悄
    轉型，object dtype 才能讓每一格各自存放它該有的型別。
    """
    if os.path.exists(path):
        df = pd.read_csv(path, dtype={"stock_id": str})
        for col in SIGNALS_LOG_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[SIGNALS_LOG_COLUMNS]
        for col in SIGNALS_LOG_BACKFILL_COLUMNS:
            df[col] = df[col].astype(object)
        return df
    return pd.DataFrame(columns=SIGNALS_LOG_COLUMNS)


def _is_true(v) -> bool:
    """CSV 讀回來的布林值常變成字串，統一判斷"""
    return str(v).strip().lower() in ("true", "1", "1.0")


def _local_scale_factor(row, series: pd.DataFrame) -> "float | None":
    """把 CSV 裡存的固定價位（entry_high/stop_loss/target1/target2，都是
    真實報價），換算成跟這次抓到的（以「今天」為錨點的）還原序列同一個
    基準，兩者才能互相比較。

    做法：series 裡 data_date 那一列的值，除以 CSV 當時存的真實收盤價，
    就是「data_date 到回填當下這段期間發生了多少除權息稀釋」的比例，拿
    這個比例去縮放 CSV 裡的固定價位。

    前提：data_date 到回填當下這段期間內只發生一次還原事件；如果同一檔
    股票在同一個 T+10 觀察窗內連續發生兩次除權息，這個簡化算法會有殘差
    誤差（機率極低，先不處理，只在這裡註明）。
    """
    data_date = str(row["data_date"])
    dates = series["date"].astype(str).reset_index(drop=True)
    matches = dates[dates == data_date]
    if matches.empty:
        return None
    idx = matches.index[-1]
    today_anchored_close = _safe_float(series.iloc[idx]["close"])
    raw_close = _safe_float(row.get("close"))
    if not raw_close or today_anchored_close is None:
        return None
    return today_anchored_close / raw_close


def _compute_outcome(row, series: pd.DataFrame) -> "dict | None":
    """對單一列訊號，用 series（該檔還原後的股價序列，date 遞增排序）漸進
    式算出進場結果與 T+0/T+1/T+5/T+10 報酬率——不是全有全無：每個欄位只要
    它需要的那天資料到手，這次就先寫進去，不等 T+10 全部到齊才一次寫完。

    - 進場日相關欄位（entry_filled/entry_price/entry_day_low_hit_stop）與
      ret_t0_pct／ret_t1_pct／ret_t5_pct：只要這次執行的 series 涵蓋得到
      對應日期，每次都重新計算並覆寫舊值——不是「算過一次就不再動」。這
      是刻意的：_local_scale_factor 的縮放係數每次執行都用「當次」的
      series 重算，如果只在第一次寫入時算好 entry_price 就鎖住，後面
      T+10 用另一次執行、另一個縮放係數算出來的收盤價去除，兩邊基準可能
      不一致（例如兩次執行之間又發生一次除權息）。每次都重算，才能保證
      同一列所有欄位永遠是同一個（當次執行的）還原基準算出來的。
      entry_filled=False（no_fill）時，這三個報酬率一律明確設回 None，
      不留上一次執行（可能是成交狀態還沒確定，或狀態改變前）殘留的舊值。
    - period_high/period_low/hit_target1/hit_target2/hit_stop/ambiguous/
      ret_t10_pct：固定要等 T+10 整個 10 天視窗都到手才一起算，同時把
      backfill_complete 設 True，之後這一列就不會再被處理、不會再被
      覆寫。

    只有連進場日都還沒到（entry_idx 超出 series 範圍）才整列回傳 None，
    這次完全沒東西可寫。

    比較用的 entry_high/stop_loss/target1/target2 会先用 _local_scale_factor
    換算到跟 series 同一個還原基準，避免除權息造成的假觸發／假報酬。
    entry_price/period_high/period_low 因此也是在這個還原基準下算出來的
    ——如果 data_date 到回填當下之間有除權息事件，這幾個回填欄位就不等於
    當時的 literal 報價，而是換算後的一致基準值；CSV 裡「報價」區的原始
    欄位（close/entry_low/entry_high/stop_loss/target1/target2）完全不受
    影響，永遠是真實報價。
    """
    dates = series["date"].astype(str).reset_index(drop=True)
    data_date = str(row["data_date"])
    matches = dates[dates == data_date]
    if matches.empty:
        return None
    data_idx = matches.index[-1]
    entry_idx = data_idx + 1
    if entry_idx >= len(series):
        return None  # 連進場日都還沒到，這次完全沒東西可寫

    scale = _local_scale_factor(row, series)
    if scale is None:
        return None

    entry_high = _safe_float(row.get("entry_high"))
    stop_loss  = _safe_float(row.get("stop_loss"))
    target1    = _safe_float(row.get("target1"))
    target2    = _safe_float(row.get("target2"))
    entry_high = entry_high * scale if entry_high is not None else None
    stop_loss  = stop_loss  * scale if stop_loss  is not None else None
    target1    = target1    * scale if target1    is not None else None
    target2    = target2    * scale if target2    is not None else None

    out = {}

    # ── 進場日相關欄位：每次都用這次執行的還原基準重新算、覆寫舊值 ──
    entry_row     = series.iloc[entry_idx]
    entry_open    = _safe_float(entry_row["open"])
    entry_day_low = _safe_float(entry_row["min"])

    entry_filled = False
    entry_price  = None
    if entry_high is not None and entry_day_low is not None and entry_open is not None:
        if entry_day_low <= entry_high:
            entry_filled = True
            entry_price  = round(min(entry_open, entry_high), 2)

    entry_day_low_hit_stop = None
    if stop_loss is not None and entry_day_low is not None:
        entry_day_low_hit_stop = bool(entry_day_low <= stop_loss)

    out["entry_filled"] = entry_filled
    out["entry_price"] = entry_price
    out["entry_day_low_hit_stop"] = entry_day_low_hit_stop

    def _close_at(offset):
        i = entry_idx + offset
        return _safe_float(series.iloc[i]["close"]) if i < len(series) else None

    # ── T+0／T+1／T+5 報酬：對應日期到手就（重新）算，同樣覆寫舊值 ────
    # 未成交（entry_filled=False）時這三個報酬率一律明確設回 None，不留
    # 上一次執行（可能是成交狀態改變前）殘留下來的舊值。
    if entry_filled and entry_price:
        c0 = _close_at(0)
        if c0 is not None:
            out["ret_t0_pct"] = round((c0 - entry_price) / entry_price * 100, 2)
        c1 = _close_at(1)
        if c1 is not None:
            out["ret_t1_pct"] = round((c1 - entry_price) / entry_price * 100, 2)
        c5 = _close_at(5)
        if c5 is not None:
            out["ret_t5_pct"] = round((c5 - entry_price) / entry_price * 100, 2)
    else:
        out["ret_t0_pct"] = None
        out["ret_t1_pct"] = None
        out["ret_t5_pct"] = None

    # ── T+10 整包：滿了才一次算完，同時把 backfill_complete 設 True ──
    t10_idx = entry_idx + 10
    if t10_idx < len(series):
        window = series.iloc[entry_idx: t10_idx + 1]  # 進場日到 T+10（含）
        period_high = float(window["max"].max())
        period_low  = float(window["min"].min())

        ret_t10 = None
        if entry_filled and entry_price:
            c10 = _close_at(10)
            if c10 is not None:
                ret_t10 = round((c10 - entry_price) / entry_price * 100, 2)

        hit_target1 = bool(target1   is not None and period_high >= target1)
        hit_target2 = bool(target2   is not None and period_high >= target2)
        hit_stop    = bool(stop_loss is not None and period_low  <= stop_loss)

        # 同一天同時觸及停損與任一目標，日K線看不出先後順序，標 ambiguous
        ambiguous = False
        for _, day in window.iterrows():
            day_low, day_high = _safe_float(day["min"]), _safe_float(day["max"])
            day_hit_stop = stop_loss is not None and day_low  is not None and day_low  <= stop_loss
            day_hit_t1   = target1   is not None and day_high is not None and day_high >= target1
            day_hit_t2   = target2   is not None and day_high is not None and day_high >= target2
            if day_hit_stop and (day_hit_t1 or day_hit_t2):
                ambiguous = True
                break

        out["ret_t10_pct"]  = ret_t10
        out["period_high"]  = round(period_high, 2)
        out["period_low"]   = round(period_low, 2)
        out["hit_target1"]  = hit_target1
        out["hit_target2"]  = hit_target2
        out["hit_stop"]     = hit_stop
        out["ambiguous"]    = ambiguous
        out["backfill_complete"] = True

    return out  # entry_idx 檢查通過後，out 一定至少有進場日相關欄位


def backfill_outcomes(df_log: pd.DataFrame, price_cache: dict) -> pd.DataFrame:
    """補填舊訊號的後續結果（T+1/5/10 報酬率、是否觸及停損停利等）。

    只處理 backfill_complete 還不是 True 的列；只用 price_cache（這次執行
    已經抓到的還原序列）計算，不會為了回填另外呼叫 FinMind——涵蓋不到的
    列這次先跳過，留到之後執行、資料視窗往前推進後自然補齊。
    """
    if df_log.empty:
        return df_log

    df_log = df_log.copy()
    pending_mask = ~df_log["backfill_complete"].apply(_is_true)

    for idx in df_log.index[pending_mask]:
        row = df_log.loc[idx]
        sid = str(row["stock_id"])
        series = price_cache.get(sid)
        if series is None or series.empty:
            continue

        outcome = _compute_outcome(row, series)
        if outcome is None:
            continue

        for col, val in outcome.items():
            df_log.at[idx, col] = val

    return df_log


def write_signals_log(
    stocks_data: list,
    final_stocks: list,
    price_cache: dict,
    generated_at: "str | None" = None,
    path: str = SIGNALS_LOG_PATH,
) -> pd.DataFrame:
    """把這次執行 13 檔的訊號寫進 CSV，並回填舊訊號的後續結果。

    主鍵 (data_date, stock_id)：同一鍵重跑會覆寫該列，不會重複新增。
    13 檔全部都寫，不只推薦股；有 data_warning 的股票也照寫（bucket 會是
    「警示」），方便事後統計時自行排除。
    """
    if generated_at is None:
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    code_version = get_code_version()

    raw_by_id = {s["stock_id"]: s for s in stocks_data}
    new_rows = []
    for fs in final_stocks:
        sid = str(fs["stock_id"])
        raw = raw_by_id.get(sid, {})
        series = price_cache.get(sid)
        if series is None or series.empty:
            log.warning(f"  ⚠️  {fs.get('name', sid)}（{sid}）無股價序列，signals_log 跳過這檔")
            continue
        data_date = str(series["date"].iloc[-1])
        row = build_signal_row(fs, raw, data_date, generated_at, code_version)
        row["return_20d_pct"] = calc_20d_return_pct(series)
        new_rows.append(row)

    df_log = read_signals_log(path)

    if new_rows:
        df_new = pd.DataFrame(new_rows, columns=SIGNALS_LOG_COLUMNS)
        for col in SIGNALS_LOG_BACKFILL_COLUMNS:
            df_new[col] = df_new[col].astype(object)

        if df_log.empty:
            # 空的（新檔案或剛好被濾光）DataFrame 直接 concat 在新版 pandas
            # 會噴 FutureWarning（全 NA 欄位的型別推斷即將改變行為），這裡
            # 沒有舊資料可合併，直接用 df_new 取代，不必真的呼叫 concat。
            df_log = df_new
        else:
            key_new = set(zip(df_new["data_date"].astype(str), df_new["stock_id"].astype(str)))
            mask_keep = ~df_log.apply(
                lambda r: (str(r["data_date"]), str(r["stock_id"])) in key_new, axis=1
            )
            df_log = df_log[mask_keep]
            df_log = pd.concat([df_log, df_new], ignore_index=True)
        log.info(f"  📝 signals_log：本次寫入/覆寫 {len(df_new)} 列")
    else:
        log.warning("  ⚠️  signals_log：本次沒有任何可寫入的列")

    df_log = backfill_outcomes(df_log, price_cache)
    df_log = df_log.sort_values(["data_date", "stock_id"]).reset_index(drop=True)

    os.makedirs(SIGNALS_LOG_DIR, exist_ok=True)
    df_log.to_csv(path, index=False)
    log.info(f"  ✅ signals_log 已更新：{path}（共 {len(df_log)} 列）")
    return df_log


# ══════════════════════════════════════════════════════════════
# Email 組合與寄送
# ══════════════════════════════════════════════════════════════

def _build_gate_banner(gate: dict) -> str:
    """組裝台指期夜盤閘門大型橫幅（顯示於報告最頂端）"""
    status = gate.get("status", "green")
    bg_map = {"red": "#c0392b", "yellow": "#d68910", "green": "#1e8449"}
    bg     = bg_map.get(status, "#1e8449")

    if gate.get("error"):
        detail = f'<span style="font-size:13px;opacity:0.8;">（{gate["error"]}）</span>'
    else:
        nc   = gate.get("night_close", 0)
        st   = gate.get("settlement",  0)
        pts  = gate.get("diff_pts",    0)
        pct  = gate.get("diff_pct",    0)
        sign = "+" if pts >= 0 else ""
        detail = (
            f'<span style="font-size:14px;opacity:0.9;">'
            f'台指期夜盤收盤：<strong>{nc:,.0f}</strong> ｜ '
            f'結算價：<strong>{st:,.0f}</strong> ｜ '
            f'差距：<strong>{sign}{pts:.0f}點（{sign}{pct:.2f}%）</strong>'
            f'</span>'
        )

    return f"""
  <!-- 台指期夜盤閘門 -->
  <div style="background:{bg};padding:22px 30px;text-align:center;">
    <div style="font-size:52px;line-height:1.1;">{gate.get("emoji","🟢")}</div>
    <div style="font-size:26px;font-weight:bold;color:white;margin:6px 0;">
      台指期夜盤閘門：{gate.get("label","綠燈")} — {gate.get("action","")}
    </div>
    <div style="margin-top:6px;color:rgba(255,255,255,0.85);">{detail}</div>
  </div>
"""


def _build_us_section(us_market: dict, news_analysis: dict) -> str:
    """組裝「零、昨夜美股與重大消息」HTML 區塊"""
    GREEN = "#27ae60"
    RED   = "#e74c3c"
    GRAY  = "#888888"

    # ── 美股數據格子 ──────────────────────────────────────────
    us_cells = ""
    for key in ("SOX", "IXIC", "NVDA", "AMD"):
        info = us_market.get(key, {})
        label   = info.get("label", key)
        chg_str = info.get("chg_str", "N/A")
        chg_p   = info.get("chg_pct", 0.0)
        color   = GREEN if chg_p > 0 else (RED if chg_p < 0 else GRAY)
        us_cells += (
            f'<div style="padding:8px;background:white;border-radius:5px;">'
            f'<span style="font-size:12px;color:#666;">{label}</span><br>'
            f'<strong style="color:{color};font-size:15px;">{chg_str}</strong>'
            f'</div>'
        )

    us_block = (
        '<div style="background:#f0f4ff;border-radius:8px;padding:14px;margin-bottom:16px;">'
        '<p style="margin:0 0 10px;font-weight:bold;font-size:14px;color:#1a237e;">📊 美股指數（昨夜收盤）</p>'
        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;">{us_cells}</div>'
        '</div>'
    ) if us_market else '<p style="color:#aaa;font-size:13px;">美股數據未取得</p>'

    # ── 新聞項目 ──────────────────────────────────────────────
    impact_color = {"正面": GREEN, "負面": RED, "中性": GRAY}
    news_items_html = ""
    for item in news_analysis.get("news_items", []):
        ic = impact_color.get(item.get("impact", "中性"), GRAY)
        badge = (
            f'<span style="background:{ic};color:white;padding:2px 8px;'
            f'border-radius:10px;font-size:11px;margin-left:8px;">'
            f'{item.get("impact","中性")}</span>'
        )
        news_items_html += (
            f'<div style="margin:10px 0;padding:12px;background:#f9f9f9;'
            f'border-left:3px solid {ic};border-radius:4px;">'
            f'<p style="margin:0 0 4px;font-size:13px;font-weight:bold;">'
            f'📌 {item.get("title","")}{badge}</p>'
            f'<p style="margin:0 0 4px;font-size:13px;color:#444;">{item.get("summary","")}</p>'
            f'<p style="margin:0;font-size:12px;color:#666;">影響標的：{item.get("affected_stocks","")}</p>'
            f'</div>'
        )
    if not news_items_html:
        news_items_html = '<p style="color:#aaa;font-size:13px;">暫無重大消息</p>'

    # ── 一句話提示 ────────────────────────────────────────────
    key_alert = news_analysis.get("key_alert", "")
    alert_block = ""
    if key_alert:
        alert_block = (
            '<div style="background:#fffde7;border-left:4px solid #f9a825;'
            'padding:12px;border-radius:4px;margin-top:12px;">'
            f'⚡ <strong>今日開盤前注意：</strong>{key_alert}'
            '</div>'
        )

    return (
        '<div class="section">'
        '<h2 class="blue">零、🌙 昨夜美股與重大消息</h2>'
        f'{us_block}'
        '<p style="font-weight:bold;font-size:14px;margin:0 0 6px;">📰 重大消息與影響</p>'
        f'{news_items_html}'
        f'{alert_block}'
        '</div>'
    )


def compose_email_html(
    analysis: dict,
    stocks_data: list,
    today_str: str,
    market_data: dict = None,
    us_market: dict = None,
    news_analysis: dict = None,
    futures_gate: dict = None,
) -> str:
    """組合 HTML 格式的完整 Email 內容"""

    if market_data is None:
        market_data = {}
    if us_market is None:
        us_market = {}
    if news_analysis is None:
        news_analysis = {}
    if futures_gate is None:
        futures_gate = {"status": "green", "emoji": "🟢", "label": "綠燈", "action": "可依評分正常操作", "error": "未取得資料"}

    raw_by_id = {s["stock_id"]: s for s in stocks_data}
    all_stocks = analysis.get("stocks", [])

    # 有 data_warning（價格資料疑似失真）的股票，不參與進場清單／觀望／暫不
    # 關注的分類，也不顯示分數與進場區間、停損停利——這些數字很可能是用失真
    # 資料算出來的，混在正常清單裡反而誤導。獨立另闢一區只顯示警示文字。
    # 分類邏輯統一由 classify_stock() 判斷（signals_log 也用同一個函式）。
    buckets = {"警示": [], "推薦": [], "觀望": [], "暫不關注": []}
    for s in all_stocks:
        buckets[classify_stock(s)].append(s)
    flagged     = buckets["警示"]
    recommended = buckets["推薦"]
    watchlist   = buckets["觀望"]
    weak        = buckets["暫不關注"]

    # ── 色彩常數 ──
    GREEN  = "#27ae60"
    YELLOW = "#f39c12"
    RED    = "#e74c3c"
    BLUE   = "#2980b9"

    def score_badge(score, max_score, color):
        return (
            f'<span style="background:{color};color:white;'
            f'padding:2px 8px;border-radius:12px;font-size:13px;">'
            f'{score}/{max_score}</span>'
        )

    def _secondary_flags_html(s):
        """個股警示（EVENT_FLAGS 手動備註）與近期除權息旗標

        flag_banner_html（一般股票卡片）與 _flagged_card（data_warning
        股票）共用同一份樣式，確保兩邊顯示一致。
        """
        html = ""
        event_note = s.get("event_note")
        if event_note:
            html += (
                f'<div style="background:#fdecea;border-left:4px solid {RED};'
                f'padding:6px 10px;border-radius:4px;margin-bottom:8px;'
                f'font-size:12px;color:#c0392b;font-weight:bold;">'
                f'⚠️ 個股警示：{event_note}'
                f'</div>'
            )
        div_flag = s.get("dividend_flag")
        if div_flag:
            html += (
                f'<div style="background:#eef2ff;border-left:4px solid #5c6bc0;'
                f'padding:6px 10px;border-radius:4px;margin-bottom:8px;'
                f'font-size:12px;color:#3949ab;">'
                f'🔔 近期除權息（{div_flag.get("type","")}）：{div_flag.get("date","")}'
                f'（{div_flag.get("trading_days_ago",0)} 個交易日前）　'
                f'除權息前收盤 {div_flag.get("before_price","")} → 參考價 {div_flag.get("after_price","")}'
                f'（{div_flag.get("chg_pct",0):+.2f}%）｜指標已依還原股價計算'
                f'</div>'
            )
        return html

    def flag_banner_html(s):
        """資料品質警示（data_warning）、個股警示（EVENT_FLAGS 手動備註）與近期除權息旗標"""
        html = ""
        for w in s.get("data_warning") or []:
            gap_str = "、".join(
                f"{g['date']} {g['pct']:+.1f}%" for g in (w.get("gaps") or [])
            )
            detail = f"（{gap_str}）" if gap_str else ""
            html += (
                '<div style="background:#fff3e0;border-left:4px solid #e67e22;'
                'padding:6px 10px;border-radius:4px;margin-bottom:8px;'
                'font-size:12px;color:#a04000;font-weight:bold;">'
                f'🚧 資料品質警示：{w.get("message","")}{detail}'
                '</div>'
            )
        html += _secondary_flags_html(s)
        return html

    def stock_card(s, border_color, show_trade=True):
        raw  = raw_by_id.get(s["stock_id"], {})
        vrat = raw.get("volume_ratio", 1.0)
        chg  = raw.get("price_chg_pct", 0)
        chg_color = GREEN if (chg or 0) >= 0 else RED
        chg_str   = f"+{chg}%" if (chg or 0) >= 0 else f"{chg}%"

        card = f"""
        <div style="background:#fafafa;border-left:5px solid {border_color};
                    padding:16px;margin:12px 0;border-radius:6px;
                    box-shadow:0 1px 4px rgba(0,0,0,0.08);">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <h3 style="margin:0;font-size:17px;color:#2c3e50;">
              {s['name']}（{s['stock_id']}）
            </h3>
            <div>
              {score_badge(s.get('fundamental_score',0), 2, '#8e44ad')}
              &nbsp;{score_badge(s.get('chip_score',0), 4, '#2980b9')}
              &nbsp;{score_badge(s.get('technical_score',0), 4, '#16a085')}
              &nbsp;<span style="background:{border_color};color:white;padding:3px 12px;
                      border-radius:12px;font-weight:bold;font-size:14px;">
                總分 {s.get('total_score',0)}/10</span>
            </div>
          </div>

          {flag_banner_html(s)}

          <table style="width:100%;font-size:13px;color:#555;margin-bottom:8px;">
            <tr>
              <td>📈 昨收：<strong>{s.get('close','N/A')}</strong></td>
              <td>漲跌：<strong style="color:{chg_color}">{chg_str}</strong></td>
              <td>成交量比：<strong>{vrat}x</strong></td>
            </tr>
          </table>

          <p style="margin:4px 0;font-size:13px;">
            <span style="color:#8e44ad">◆ 基本面</span>：{s.get('fundamental_reason','')}
          </p>
          <p style="margin:4px 0;font-size:13px;">
            <span style="color:#2980b9">◆ 籌碼面</span>：{s.get('chip_reason','')}
          </p>
          <p style="margin:4px 0;font-size:13px;">
            <span style="color:#16a085">◆ 技術面</span>：{s.get('technical_reason','')}
          </p>
        """

        if show_trade and s.get("recommend"):
            def _pct(level_str):
                """由實際價位與昨收算出漲跌幅字串（ATR 停損停利的幅度非固定）"""
                try:
                    lv = float(level_str)
                    c  = float(s.get("close") or 0)
                    if c > 0:
                        p = (lv - c) / c * 100
                        return f"{'+' if p >= 0 else ''}{p:.1f}%"
                except (TypeError, ValueError):
                    pass
                return "N/A"

            basis = s.get("level_basis", "")
            basis_html = (
                f'<br><span style="font-size:11px;color:#888;">依據：{basis}</span>'
                if basis else ""
            )
            card += f"""
          <div style="background:#e8f5e9;padding:10px 14px;border-radius:4px;
                      margin-top:10px;font-size:13px;">
            💰 <strong>交易策略</strong>：
            進場區間 <strong>{s.get('entry_range','N/A')}</strong> ｜
            停損 <strong style="color:{RED}">{s.get('stop_loss','N/A')}</strong>（{_pct(s.get('stop_loss'))}）<br>
            🎯 第一目標：<strong style="color:{GREEN}">{s.get('target1','N/A')}</strong>（{_pct(s.get('target1'))} 出一半）｜
               第二目標：<strong style="color:{GREEN}">{s.get('target2','N/A')}</strong>（{_pct(s.get('target2'))} 全出）
            {basis_html}
          </div>
            """

        if not s.get("recommend") and s.get("watch_conditions"):
            card += f"""
          <p style="margin:6px 0 0;font-size:12px;color:#888;">
            ⏳ 待改善：{s.get('watch_conditions','')}
          </p>
            """

        card += "</div>"
        return card

    # ── 組裝完整 HTML ──
    rec_section = "".join(stock_card(s, GREEN) for s in recommended) or (
        "<p style='color:#aaa;'>今日無符合條件（≥7分）的推薦標的</p>"
    )
    watch_section = "".join(stock_card(s, YELLOW, show_trade=False) for s in watchlist) or (
        "<p style='color:#aaa;'>無觀望標的</p>"
    )
    def _weak_flags_html(s):
        # weak 只包含 clean（無 data_warning）的股票，data_warning 走 flagged_section
        parts = []
        if s.get("event_note"):
            parts.append(f"<span style='color:{RED};'>⚠️ {s['event_note']}</span>")
        if s.get("dividend_flag"):
            parts.append("<span style='color:#3949ab;'>🔔 近期除權息</span>")
        return f"<br>{'　'.join(parts)}" if parts else ""

    def _flagged_card(s):
        """data_warning 股票：顯示原始昨收／漲跌供人工判斷，仍隱藏分數、
        進場區間、停損停利（這些數字用的是同一份可能失真的資料算出來的）"""
        raw  = raw_by_id.get(s["stock_id"], {})
        close = raw.get("close", s.get("close"))
        chg   = raw.get("price_chg_pct", 0) or 0
        chg_color = GREEN if chg >= 0 else RED
        chg_str   = f"+{chg}%" if chg >= 0 else f"{chg}%"
        msgs = "；".join(w.get("message", "") for w in (s.get("data_warning") or []))
        return f"""
        <div style="background:#fff3e0;border-left:5px solid #e67e22;
                    padding:16px;margin:12px 0;border-radius:6px;">
          <h3 style="margin:0 0 6px;font-size:16px;color:#a04000;">
            {s['name']}（{s['stock_id']}）
          </h3>
          <p style="margin:0 0 6px;font-size:13px;color:#555;">
            📈 昨收（原始報價）：<strong>{close if close is not None else 'N/A'}</strong>
            &nbsp;漲跌：<strong style="color:{chg_color}">{chg_str}</strong>
            &nbsp;<span style="color:#a04000;font-weight:bold;">請手動判斷</span>
          </p>
          <p style="margin:0;font-size:13px;color:#a04000;font-weight:bold;">
            🚧 {msgs}
          </p>
          {_secondary_flags_html(s)}
          <p style="margin:6px 0 0;font-size:12px;color:#888;">
            分數、進場區間、停損停利已隱藏，不列入任何清單。
          </p>
        </div>
        """

    flagged_section = "".join(_flagged_card(s) for s in flagged)
    flagged_block = ""
    if flagged:
        flagged_block = f"""
  <!-- 資料異常警示（data_warning，不計分不列入任何清單） -->
  <div class="section">
    <h2 style="font-size:16px;margin:0 0 14px;color:#e67e22;border-left:4px solid #e67e22;padding-left:10px;">
      🚧 資料異常警示（共 {len(flagged)} 檔，不計分、不列入任何清單）
    </h2>
    {flagged_section}
  </div>
"""

    weak_rows = "".join(
        f"<tr><td style='padding:6px;'>{s['name']}（{s['stock_id']}）{_weak_flags_html(s)}</td>"
        f"<td style='padding:6px;text-align:center;'>{s.get('total_score',0)}/10</td>"
        f"<td style='padding:6px;color:#888;'>{s.get('summary','')}</td></tr>"
        for s in weak
    )
    weak_section = (
        f"""<table style="width:100%;border-collapse:collapse;font-size:13px;">
          <tr style="background:#f5f5f5;">
            <th style="padding:6px;text-align:left;">股票</th>
            <th style="padding:6px;">評分</th>
            <th style="padding:6px;text-align:left;">說明</th>
          </tr>
          {weak_rows}
        </table>"""
        if weak_rows else "<p style='color:#aaa;'>無</p>"
    )

    html = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head><meta charset="UTF-8">
<style>
  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; background:#f0f2f5;
         margin:0; padding:20px; color:#333; }}
  .container {{ max-width:780px; margin:0 auto; background:white;
               border-radius:10px; box-shadow:0 2px 12px rgba(0,0,0,0.1);
               overflow:hidden; }}
  .header {{ background:linear-gradient(135deg,#1a237e,#1565c0);
             color:white; padding:24px 30px; }}
  .header h1 {{ margin:0; font-size:22px; letter-spacing:1px; }}
  .header p  {{ margin:6px 0 0; opacity:0.85; font-size:14px; }}
  .section {{ padding:20px 30px; border-bottom:1px solid #eee; }}
  .section h2 {{ margin:0 0 14px; font-size:16px; border-left:4px solid;
                padding-left:10px; }}
  .section h2.green  {{ border-color:{GREEN}; color:{GREEN}; }}
  .section h2.yellow {{ border-color:{YELLOW}; color:{YELLOW}; }}
  .section h2.red    {{ border-color:{RED}; color:{RED}; }}
  .section h2.blue   {{ border-color:{BLUE}; color:{BLUE}; }}
  .footer {{ padding:14px 30px; background:#f8f9fa; font-size:11px; color:#aaa; }}
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div class="header">
    <h1>📊 台股 AI 每日分析報告</h1>
    <p>報告日期：{today_str} ｜ 分析模型：Claude AI ｜ 資料來源：FinMind</p>
  </div>

  <!-- 台指期夜盤閘門 -->
  {_build_gate_banner(futures_gate)}

  <!-- 零、昨夜美股與重大消息 -->
  {_build_us_section(us_market, news_analysis)}

  <!-- 一、大盤概況 -->
  <div class="section">
    <h2 class="blue">一、大盤概況</h2>
    <p style="font-size:15px;">
      🏦 加權指數（昨收）：<strong>{market_data.get('close','N/A')}</strong>
      &nbsp;漲跌：<strong>{market_data.get('change','N/A')}</strong>
      （<strong>{market_data.get('change_pct','N/A')}</strong>）
    </p>
    <p style="font-size:14px;color:#555;">{analysis.get('market_summary','')}</p>
  </div>

  {flagged_block}

  <!-- 二、推薦進場清單 -->
  <div class="section">
    <h2 class="green">二、✅ 推薦進場清單（評分 ≥ 7 分，共 {len(recommended)} 檔）</h2>
    {rec_section}
  </div>

  <!-- 三、觀望清單 -->
  <div class="section">
    <h2 class="yellow">三、👀 觀望清單（評分 5–6 分，共 {len(watchlist)} 檔）</h2>
    {watch_section}
  </div>

  <!-- 評分偏低 -->
  <div class="section">
    <h2 style="font-size:16px;margin:0 0 14px;color:#999;border-left:4px solid #ccc;padding-left:10px;">
      暫不關注（評分 &lt; 5 分）
    </h2>
    {weak_section}
  </div>

  <!-- 四、風險提示 -->
  <div class="section">
    <h2 class="red">四、⚠️ 今日風險提示</h2>
    <p style="font-size:14px;"><strong>外資動向：</strong>{analysis.get('overall_foreign_trend','')}</p>
    <div style="background:#fff5f5;border-left:4px solid {RED};padding:12px;
                border-radius:4px;font-size:14px;">
      {analysis.get('risk_warning','')}
    </div>
  </div>

  <!-- Footer -->
  <div class="footer">
    ⚠️ 本報告由 AI 自動生成，僅供參考，不構成投資建議。投資有風險，請自行判斷。<br>
    產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ｜
    分析引擎：Anthropic Claude Sonnet
  </div>

</div>
</body>
</html>"""

    return html


def send_email(subject: str, html_body: str) -> None:
    """透過 Gmail SMTP 寄出 HTML Email"""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_USER
    msg["To"]      = RECIPIENT_EMAIL

    msg.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(GMAIL_USER, GMAIL_PASSWORD)
        smtp.sendmail(GMAIL_USER, RECIPIENT_EMAIL, msg.as_string())

    log.info(f"✅ Email 寄出成功 → {RECIPIENT_EMAIL}")


def send_error_email(error_details: str) -> None:
    """當程式發生嚴重錯誤時，寄送錯誤通知"""
    try:
        today_str = datetime.now().strftime("%Y/%m/%d %H:%M")
        subject   = f"❌ 台股 AI 分析系統發生錯誤 — {today_str}"
        body = f"""
        <html><body style="font-family:Arial;padding:20px;">
        <h2 style="color:red;">台股 AI 分析系統發生錯誤</h2>
        <p><strong>時間：</strong>{today_str}</p>
        <p><strong>錯誤詳情：</strong></p>
        <pre style="background:#f5f5f5;padding:15px;border-radius:5px;
                    overflow:auto;font-size:12px;">{error_details}</pre>
        <p style="color:#666;font-size:12px;">請檢查程式或手動執行排查問題。</p>
        </body></html>
        """
        send_email(subject, body)
    except Exception as e:
        log.error(f"連錯誤通知信也無法寄出：{e}")


# ══════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════

def main():
    today      = datetime.now()
    today_str  = today.strftime("%Y/%m/%d")
    log.info(f"{'='*60}")
    log.info(f"台股 AI 每日分析系統啟動 — {today_str}")
    log.info(f"{'='*60}")

    # ── 檢查必要設定 ───────────────────────────────────────────
    missing = [
        k for k, v in {
            "FINMIND_TOKEN":   FINMIND_TOKEN,
            "ANTHROPIC_KEY":   ANTHROPIC_KEY,
            "GMAIL_USER":      GMAIL_USER,
            "GMAIL_PASSWORD":  GMAIL_PASSWORD,
            "RECIPIENT_EMAIL": RECIPIENT_EMAIL,
        }.items() if not v
    ]
    if missing:
        err = f"缺少必要環境變數：{missing}"
        log.error(err)
        send_error_email(err)
        sys.exit(1)

    try:
        # ── Step 0: 台指期夜盤閘門 ──────────────────────────────
        futures_gate = get_futures_gate()

        # ── Step 1: 美股數據 + 新聞分析 ─────────────────────────
        us_market     = get_us_market_data()
        news_analysis = get_us_news_analysis(us_market)

        # ── Step 2: 大盤資料 ────────────────────────────────────
        market_data = get_market_data()

        # ── Step 3: 逐一抓取股票資料 ────────────────────────────
        log.info(f"開始抓取 {len(STOCKS)} 檔股票資料...")
        stocks_data = []
        price_cache = {}  # stock_id -> 還原股價序列，只給 backfill_outcomes 用
        for stock_id in STOCKS:
            try:
                data, price_series = get_stock_data(stock_id)
                stocks_data.append(data)
                price_cache[stock_id] = price_series
                time.sleep(0.5)  # 避免 API 頻率限制
            except Exception as e:
                log.error(f"抓取 {stock_id} 失敗：{e}")
                stocks_data.append({
                    "stock_id": stock_id,
                    "name": STOCKS[stock_id],
                    "error": str(e),
                })
                price_cache[stock_id] = pd.DataFrame()

        # ── Step 4: Claude AI 分析 ───────────────────────────────
        analysis = analyze_with_claude(stocks_data, market_data)

        # ── Step 5: 組合並寄送 Email ────────────────────────────
        gate_emoji = futures_gate.get("emoji", "📊")
        subject    = f"{gate_emoji} 台股 AI 每日分析報告 — {today_str}"
        html_body = compose_email_html(
            analysis, stocks_data, today_str, market_data,
            us_market=us_market, news_analysis=news_analysis,
            futures_gate=futures_gate,
        )
        send_email(subject, html_body)

        log.info("🎉 今日分析完成！")

        # ── Step 6: 更新訊號日誌（data/signals_log.csv）─────────
        # 放在寄信之後：這裡萬一出錯，不影響報告已經寄出這件事，只記警告、
        # 不中斷、不觸發錯誤通知信。
        try:
            write_signals_log(stocks_data, analysis.get("stocks", []), price_cache)
        except Exception as e:
            log.error(f"⚠️  signals_log 更新失敗（不影響報告寄送）：{e}\n{traceback.format_exc()}")
            print("::warning::signals_log 更新失敗，本次日誌未寫入")

    except json.JSONDecodeError as e:
        err = f"Claude API 回傳格式錯誤（非 JSON）：{e}\n{traceback.format_exc()}"
        log.error(err)
        send_error_email(err)
        sys.exit(1)

    except Exception as e:
        err = f"程式執行發生錯誤：{e}\n{traceback.format_exc()}"
        log.error(err)
        send_error_email(err)
        sys.exit(1)


if __name__ == "__main__":
    main()
