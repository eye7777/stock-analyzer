# 📊 台股 AI 每日分析系統

每天台灣時間早上 7:30 自動執行，分析 8 檔台股，產生 AI 評分報告並寄送 Email。

## 追蹤股票
奇鋐(3017)、雙鴻(3324)、欣興(3037)、緯穎(6669)、嘉澤(3533)、勤誠(8210)、晟銘電(3013)、英業達(2356)

## 分析維度
- **基本面（2分）**：近三個月營收年增率
- **籌碼面（4分）**：外資、投信買賣超張數與連買天數
- **技術面（4分）**：MA5/MA20/KD/量增價漲

## 使用技術
- Python 3.11 / FinMind API / Anthropic Claude API / Gmail SMTP
- 自動排程：GitHub Actions（每週一至週五 07:30 台灣時間）

## 必要的 GitHub Secrets
| Secret 名稱 | 說明 |
|---|---|
| `FINMIND_TOKEN` | FinMind API Token |
| `ANTHROPIC_API_KEY` | Claude AI API Key |
| `GMAIL_USER` | 寄件 Gmail 帳號 |
| `GMAIL_PASSWORD` | Gmail 應用程式密碼 |
| `RECIPIENT_EMAIL` | 收件 Email |

> ⚠️ 本系統僅供個人學習與參考，不構成投資建議。
