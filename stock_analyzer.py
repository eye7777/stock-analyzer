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


# ══════════════════════════════════════════════════════════════
# 除權息還原（避免 MA/KD/ATR 被未還原股價的價格斷層扭曲）
# ══════════════════════════════════════════════════════════════

def get_dividend_events(stock_id: str, start_date: str) -> "tuple[pd.DataFrame, str | None]":
    """取得除權息事件（現金股利／股票股利，含盈餘或公積轉增資）

    回傳 (events_df, error_msg)：
    - error_msg 為 None 代表 API 呼叫成功，即使 events_df 是空的（這檔股票
      在查詢區間內單純沒有除權息）也算成功，不是失敗。
    - error_msg 非 None 代表 API 呼叫本身失敗（HTTP 例外或 FinMind
      status != 200），此時 events_df 必為空。呼叫端不可把這種情況誤判成
      「沒有除權息」而靜默略過還原——必須另外標示 data_warning。

    FinMind TaiwanStockDividendResult 的 date 欄位即為實際除權息交易日，
    before_price/after_price 是當天真正生效的除權息前後基準價，可直接拿來
    反推還原因子（after_price / before_price）。stock_or_cache_dividend
    欄位標示事件類型："息"＝現金股利除息，"權"＝股票股利（含盈餘/公積轉
    增資）除權——不是現金增資（現金增資另有 CashIncreaseSubscriptionRate
    等欄位，在 TaiwanStockDividend 資料集裡，且與此欄位無關）。
    """
    df, err = _finmind_request("TaiwanStockDividendResult", stock_id, start_date)
    if err:
        log.warning(f"FinMind TaiwanStockDividendResult ({stock_id}): {err}")
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

    只做標示（data_warning），完全不影響 score_stock / calc_trade_levels
    的計算結果——分數與停損停利照舊輸出，報告會另外提示「不可信」讓人判斷。
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


def get_stock_data(stock_id: str) -> dict:
    """
    抓取單一股票所需的所有資料：
    - 昨日收盤價與成交量
    - MA5 / MA20 / KD
    - 外資、投信買賣超
    - 近三個月營收年增率
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
        return result

    df_price = df_price.sort_values("date").reset_index(drop=True)
    for col in ["close", "max", "min", "open", "Trading_Volume"]:
        if col in df_price.columns:
            df_price[col] = pd.to_numeric(df_price[col], errors="coerce")

    if len(df_price) < 2:
        result["error"] = "股價資料筆數不足"
        return result

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

    return result


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
    for s in stocks_data:
        if s.get("error"):
            stocks_summary.append({
                "stock_id": s["stock_id"],
                "name": s["name"],
                "error": s["error"],
            })
            continue

        rev_str = "、".join([
            f"{x['month']} YoY={x['yoy']}%" if x["yoy"] is not None else f"{x['month']} YoY=N/A"
            for x in s.get("revenue_yoy_list", [])
        ]) or "無資料"

        stocks_summary.append({
            "stock_id":            str(s["stock_id"]),
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

        # ── Python 端直接計算評分與 ATR 停損停利（取代 LLM 算術）──
        sid = str(s["stock_id"])
        levels = calc_trade_levels(
            float(s["close"]) if s.get("close") is not None else None,
            s.get("atr"),
        )
        scored_by_id[sid] = {**score_stock(s), **levels}

    # 傳給 Claude 的評分結果（只含數字，供其撰寫理由時對照）
    scores_for_prompt = {
        sid: {k: v[k] for k in (
            "fundamental_score", "chip_score", "technical_score",
            "total_score", "recommend",
        )}
        for sid, v in scored_by_id.items()
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
    flagged     = [s for s in all_stocks if s.get("data_warning")]
    clean       = [s for s in all_stocks if not s.get("data_warning")]

    recommended = [s for s in clean if s.get("total_score", 0) >= 7]
    watchlist   = [s for s in clean if 5 <= s.get("total_score", 0) <= 6]
    weak        = [s for s in clean if s.get("total_score", 0) < 5]

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
        """data_warning 股票只顯示警示文字，不顯示分數、進場區間、停損停利"""
        msgs = "；".join(w.get("message", "") for w in (s.get("data_warning") or []))
        return f"""
        <div style="background:#fff3e0;border-left:5px solid #e67e22;
                    padding:16px;margin:12px 0;border-radius:6px;">
          <h3 style="margin:0 0 6px;font-size:16px;color:#a04000;">
            {s['name']}（{s['stock_id']}）
          </h3>
          <p style="margin:0;font-size:13px;color:#a04000;font-weight:bold;">
            🚧 {msgs}
          </p>
          <p style="margin:6px 0 0;font-size:12px;color:#888;">
            資料可能失真，本檔本次不計分、不列入任何清單，僅顯示警示。
          </p>
        </div>
        """

    flagged_section = "".join(_flagged_card(s) for s in flagged) or (
        "<p style='color:#aaa;'>無</p>"
    )

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

  <!-- 資料異常警示（data_warning，不計分不列入任何清單） -->
  <div class="section">
    <h2 style="font-size:16px;margin:0 0 14px;color:#e67e22;border-left:4px solid #e67e22;padding-left:10px;">
      🚧 資料異常警示（共 {len(flagged)} 檔，不計分、不列入任何清單）
    </h2>
    {flagged_section}
  </div>

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
        for stock_id in STOCKS:
            try:
                data = get_stock_data(stock_id)
                stocks_data.append(data)
                time.sleep(0.5)  # 避免 API 頻率限制
            except Exception as e:
                log.error(f"抓取 {stock_id} 失敗：{e}")
                stocks_data.append({
                    "stock_id": stock_id,
                    "name": STOCKS[stock_id],
                    "error": str(e),
                })

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
