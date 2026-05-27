#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
台股 AI 每日分析系統
每天早上 7:30 自動執行，分析 8 檔台股並寄送 Email 報告
"""

import os
import sys
import json
import time
import logging
import traceback
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd
import requests
from dotenv import load_dotenv
import anthropic

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
    "3017": "奇鋐",
    "3324": "雙鴻",
    "3037": "欣興",
    "6669": "緯穎",
    "3533": "嘉澤",
    "8210": "勤誠",
    "3013": "晟銘電",
    "2356": "英業達",
}

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


def finmind_get(dataset: str, stock_id: str, start_date: str) -> pd.DataFrame:
    """呼叫 FinMind API 取得資料，失敗時回傳空 DataFrame"""
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
            log.warning(f"FinMind {dataset} ({stock_id}): {data.get('msg', '未知錯誤')}")
            return pd.DataFrame()

        if not data.get("data"):
            return pd.DataFrame()

        return pd.DataFrame(data["data"])

    except Exception as e:
        log.warning(f"FinMind API 呼叫失敗 ({dataset}, {stock_id}): {e}")
        return pd.DataFrame()


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


# ══════════════════════════════════════════════════════════════
# 資料抓取
# ══════════════════════════════════════════════════════════════

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
        "foreign_net": 0,
        "trust_net": 0,
        "foreign_consec": 0,
        "trust_consec": 0,
        "revenue_yoy_list": [],
        "avg_yoy": None,
        "error": None,
    }

    # ── 1. 股價資料 ────────────────────────────────────────────────
    df_price = finmind_get("TaiwanStockPrice", stock_id, start_90)
    if df_price.empty:
        result["error"] = "無法取得股價資料"
        return result

    df_price = df_price.sort_values("date").reset_index(drop=True)
    for col in ["close", "max", "min", "Trading_Volume"]:
        if col in df_price.columns:
            df_price[col] = pd.to_numeric(df_price[col], errors="coerce")

    if len(df_price) < 2:
        result["error"] = "股價資料筆數不足"
        return result

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
        f" K={result['K']} D={result['D']}"
        f" 外資={result['foreign_net']:+d}張(連{result['foreign_consec']}日)"
        f" 投信={result['trust_net']:+d}張(連{result['trust_consec']}日)"
        f" 營收YoY: {rev_str}"
        f" 均YoY={result['avg_yoy']}"
    )

    return result


# ══════════════════════════════════════════════════════════════
# Claude AI 分析
# ══════════════════════════════════════════════════════════════

def analyze_with_claude(stocks_data: list, market_data: dict) -> dict:
    """
    將所有股票資料送給 Claude API（使用 Tool Use 強制輸出合法 JSON）
    取得評分、建議、進出場價位等結構化分析
    """
    log.info("呼叫 Claude API 進行 AI 分析...")

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    # 準備給 Claude 的資料摘要（全部轉為 Python 原生型別）
    stocks_summary = []
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
            "foreign_net_lots":    int(s["foreign_net"]),
            "trust_net_lots":      int(s["trust_net"]),
            "foreign_consec_days": int(s["foreign_consec"]),
            "trust_consec_days":   int(s["trust_consec"]),
            "revenue_3m":          str(rev_str),
            "avg_yoy_pct":         float(s["avg_yoy"]) if s["avg_yoy"] is not None else None,
        })

    prompt = f"""你是台股量化分析師，請用繁體中文分析以下股票並呼叫 output_analysis 工具。

大盤：加權指數 {market_data.get('close','N/A')} 點，漲跌 {market_data.get('change','N/A')}（{market_data.get('change_pct','N/A')}）

股票資料（JSON）：
{json.dumps(stocks_summary, ensure_ascii=False, indent=2)}

【評分規則，嚴格依數據執行，滿分10分】
基本面（max 2）：avg_yoy_pct≥30%→2分，15-30%→1分，其他→0分
籌碼面（max 4）：外資買超+1，外資連買≥3日再+1，投信買超+1，投信連買≥3日再+1
技術面（max 4）：above_ma5=true+1，above_ma20=true+1，kd_cross=true且K<80再+1，volume_ratio>1.2且price_chg_pct>0再+1

【輸出要求】
- total_score = 三項分數加總（必須精確）
- recommend：total_score≥7才填true
- 進出場：stop_loss=close×0.95，target1=close×1.08，target2=close×1.12（均取一位小數）
- 各reason欄位限50字以內
- summary限30字以內
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
                                "total_score":        {"type": "integer", "minimum": 0, "maximum": 10},
                                "fundamental_score":  {"type": "integer", "minimum": 0, "maximum": 2},
                                "fundamental_reason": {"type": "string"},
                                "chip_score":         {"type": "integer", "minimum": 0, "maximum": 4},
                                "chip_reason":        {"type": "string"},
                                "technical_score":    {"type": "integer", "minimum": 0, "maximum": 4},
                                "technical_reason":   {"type": "string"},
                                "recommend":          {"type": "boolean"},
                                "close":              {"type": "number"},
                                "entry_range":        {"type": "string"},
                                "stop_loss":          {"type": "string"},
                                "target1":            {"type": "string"},
                                "target2":            {"type": "string"},
                                "summary":            {"type": "string"},
                                "watch_conditions":   {"type": "string"},
                            },
                            "required": [
                                "stock_id", "name", "total_score",
                                "fundamental_score", "fundamental_reason",
                                "chip_score", "chip_reason",
                                "technical_score", "technical_reason",
                                "recommend", "close",
                                "entry_range", "stop_loss", "target1", "target2",
                                "summary", "watch_conditions",
                            ],
                        },
                    },
                    "market_summary":        {"type": "string", "description": "大盤概況說明（50字內）"},
                    "overall_foreign_trend": {"type": "string", "description": "外資在這8檔的整體動向"},
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
    for block in message.content:
        if block.type == "tool_use" and block.name == "output_analysis":
            result = block.input
            # ── 強制用 STOCKS 字典覆蓋公司名稱 ──────────────────────
            # Claude 有時會產生錯誤名稱（如「田勝明」→「晟銘電」），
            # 以 stock_id 為 key 強制寫入正確名稱，AI 無法覆蓋
            for s in result.get("stocks", []):
                sid = s.get("stock_id", "")
                if sid in STOCKS:
                    s["name"] = STOCKS[sid]   # ← 永遠用這裡的正確名稱
            log.info(f"  Claude 回傳 {len(result.get('stocks', []))} 檔評分結果")
            for s in result.get("stocks", []):
                log.info(f"    {s.get('name','?')}({s.get('stock_id','?')}): "
                         f"總={s.get('total_score','?')} 基={s.get('fundamental_score','?')} "
                         f"籌={s.get('chip_score','?')} 技={s.get('technical_score','?')} "
                         f"推薦={s.get('recommend','?')}")
            return result

    raise ValueError("Claude 未回傳 output_analysis 工具呼叫結果")


# ══════════════════════════════════════════════════════════════
# Email 組合與寄送
# ══════════════════════════════════════════════════════════════

def compose_email_html(analysis: dict, stocks_data: list, today_str: str, market_data: dict = None) -> str:
    """組合 HTML 格式的完整 Email 內容"""

    if market_data is None:
        market_data = {}

    raw_by_id = {s["stock_id"]: s for s in stocks_data}
    all_stocks = analysis.get("stocks", [])

    recommended = [s for s in all_stocks if s.get("total_score", 0) >= 7]
    watchlist   = [s for s in all_stocks if 5 <= s.get("total_score", 0) <= 6]
    weak        = [s for s in all_stocks if s.get("total_score", 0) < 5]

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

    def stock_card(s, border_color, show_trade=True):
        raw  = raw_by_id.get(s["stock_id"], {})
        vol  = raw.get("volume", 0)
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
            card += f"""
          <div style="background:#e8f5e9;padding:10px 14px;border-radius:4px;
                      margin-top:10px;font-size:13px;">
            💰 <strong>交易策略</strong>：
            進場區間 <strong>{s.get('entry_range','N/A')}</strong> ｜
            停損 <strong style="color:{RED}">{s.get('stop_loss','N/A')}</strong>（-5%）<br>
            🎯 第一目標：<strong style="color:{GREEN}">{s.get('target1','N/A')}</strong>（+8% 出一半）｜
               第二目標：<strong style="color:{GREEN}">{s.get('target2','N/A')}</strong>（+12% 全出）
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
    weak_rows = "".join(
        f"<tr><td style='padding:6px;'>{s['name']}（{s['stock_id']}）</td>"
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
        # ── Step 1: 大盤資料 ────────────────────────────────────
        market_data = get_market_data()

        # ── Step 2: 逐一抓取股票資料 ────────────────────────────
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

        # ── Step 3: Claude AI 分析 ───────────────────────────────
        analysis = analyze_with_claude(stocks_data, market_data)

        # ── Step 4: 組合並寄送 Email ────────────────────────────
        subject  = f"📊 台股 AI 每日分析報告 — {today_str}"
        html_body = compose_email_html(analysis, stocks_data, today_str, market_data)
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
