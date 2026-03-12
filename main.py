import json
import logging
import math
import os
import random
import re
import time
from datetime import datetime
from io import StringIO
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Set, Tuple

import gspread
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from google.oauth2.service_account import Credentials
from scipy.optimize import bisect

JST = ZoneInfo("Asia/Tokyo")
APP_CONFIG_ENV = "APP_CONFIG_JSON"
BENCHMARK_TICKER = "1306.T"  # TOPIX ETF as fallback market proxy
SHEET_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
SBI_JP10Y_URL = "https://www.sbisec.co.jp/ETGate/?_ControlID=WPLETmgR001Control&_PageID=WPLETmgR001Mdtl20&_DataStoreID=DSWPLETmgR001Control&_ActionID=DefaultAID&burl=iris_indexDetail&cat1=market&cat2=index&dir=tl1-idxdtl%7Ctl2-JP10YT%3DXX%7Ctl5-jpn&file=index.html&getFlg=on"
SBI_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:135.0) Gecko/20100101 Firefox/135.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Safari/605.1.15",
]
_RF_RATE_CACHE: Optional[float] = None
_RF_RATE_SOURCE = ""

EVAL_HEADERS = [
    "適正株価",                # E
    "買い上限株価",            # F
    "現在株価との差異率",      # G
    "買い上限との差異率",      # H
    "総合判定",                # I
    "減衰EP株価",              # J
    "利益アンカー株価",        # K
    "純資産アンカー株価",      # L
    "保守PBR株価",             # M
    "配当割引株価",            # N
    "金融利益アンカー株価",    # O
    "金融業種フラグ",          # P
    "モデル信頼度",            # Q
]

DB_HEADERS = [
    # 7.1 基本識別
    "ticker_yf",
    "market",
    "currency",
    "quote_type",
    "sector_raw",
    "industry_raw",
    "financial_flag",
    "financial_flag_override",
    "data_status",
    "last_db_update_jst",
    # 7.2 株価・株式数
    "current_price",
    "market_cap",
    "shares_outstanding",
    "enterprise_value",
    "beta",
    "dividend_yield",
    "trailing_annual_dividend_rate",
    # 7.3 損益
    "revenue_ttm",
    "ebit_ttm",
    "ebitda_ttm",
    "net_income_ttm",
    "nopat_ttm",
    # 7.4 貸借対照表
    "total_assets",
    "total_equity",
    "cash_and_equivalents",
    "total_debt",
    "net_debt",
    "invested_capital",
    # 7.5 キャッシュフロー
    "operating_cf_ttm",
    "capex_ttm",
    "fcf_ttm",
    "fcfe_ttm",
    # 7.6 1株指標
    "eps_ttm",
    "bps",
    "dps_ttm",
    "payout_ratio",
    "pb_now",
    "pe_now",
    # 7.7 収益性
    "roe_1y",
    "roe_3y_avg",
    "roic_1y",
    "roic_3y_avg",
    "roe_normalized",
    "roic_normalized",
    # 7.8 資本コスト系
    "rf_rate",
    "erp",
    "country_risk_premium",
    "size_premium",
    "wacc",
    "coe",
    "cod_estimate",
    "tax_rate_estimate",
    # 7.9 成長前提
    "growth_base",
    "growth_floor",
    "growth_cap",
    "terminal_growth",
    "gap_year_default",
    # 7.10 Reverse DCF 用
    "implied_growth_rate",
    "implied_gap_years",
    # 7.11 金融専用
    "financial_roe_avg",
    "financial_payout_avg",
    "financial_justified_pbr",
    "financial_implied_roe",
    # 7.12 エラー追跡
    "missing_fields",
    "calc_error",
    "notes",
]

FINANCIAL_KEYWORDS = [
    "bank", "banks", "insurance", "capital markets", "financial services",
    "asset management", "securities", "broker", "brokerage",
    "銀行", "保険", "証券", "金融", "アセットマネジメント", "資産運用",
]

LABELS = {
    "revenue": ["Total Revenue", "Operating Revenue", "Revenue"],
    "ebit": ["EBIT", "Operating Income", "Operating Profit"],
    "ebitda": ["EBITDA"],
    "net_income": ["Net Income", "Net Income Common Stockholders", "Net Income Including Noncontrolling Interests"],
    "assets": ["Total Assets"],
    "equity": ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest", "Total Equity"],
    "cash": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash And Short Term Investments"],
    "debt": ["Total Debt", "Long Term Debt And Capital Lease Obligation", "Current Debt And Capital Lease Obligation"],
    "operating_cf": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities", "Net Cash Provided By Operating Activities"],
    "capex": ["Capital Expenditure", "Capital Expenditures"],
    "pretax_income": ["Pretax Income", "Pre-Tax Income", "Pretax Earnings"],
    "tax_provision": ["Tax Provision", "Provision For Income Taxes", "Income Tax Expense"],
}

logging.basicConfig(
    level=logging.ERROR,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


def error_guidance_message(exc: Exception) -> str:
    msg = str(exc)
    lower_msg = msg.lower()

    if "grid limits" in lower_msg or "max columns" in lower_msg or "max rows" in lower_msg:
        return "修正方針: スプレッドシートの行数・列数が不足しています。対象シートのグリッドを拡張してください。"
    if "permission" in lower_msg or "forbidden" in lower_msg or "403" in lower_msg:
        return "修正方針: サービスアカウントに対象スプレッドシートの編集権限を付与してください。"
    if "not found" in lower_msg or "404" in lower_msg:
        return "修正方針: スプレッドシートURL、シート名、銘柄コードの指定を確認してください。"
    if "429" in lower_msg or "quota" in lower_msg or "rate limit" in lower_msg:
        return "修正方針: API呼び出し頻度を下げるか、待機時間を増やしてください。"
    if "app_config_json" in lower_msg or "credential" in lower_msg or "auth" in lower_msg or "secrets" in lower_msg:
        return "修正方針: APP_CONFIG_JSON と GCP サービスアカウント鍵の設定を確認してください。"

    return "修正方針: エラー文を確認し、APP_CONFIG_JSON・シート設定・入力銘柄コードの順に切り分けてください。"


def log_error_with_guidance(exc: Exception) -> None:
    logger.error("ERROR: %s: %s", type(exc).__name__, exc)
    logger.error(error_guidance_message(exc))


def load_config() -> Dict[str, Any]:
    raw = os.environ.get(APP_CONFIG_ENV)
    if not raw:
        raise RuntimeError(f"{APP_CONFIG_ENV} is not set.")
    config = json.loads(raw)
    required = ["spreadsheet_url", "sheet_name", "gcp_service_account"]
    missing = [k for k in required if not config.get(k)]
    if missing:
        raise RuntimeError(f"Missing config keys: {', '.join(missing)}")
    return config


def get_client(config: Dict[str, Any]) -> gspread.Client:
    credentials = Credentials.from_service_account_info(
        config["gcp_service_account"], scopes=SHEET_SCOPES
    )
    return gspread.authorize(credentials)


def column_letter(col_num: int) -> str:
    result = ""
    while col_num > 0:
        col_num, rem = divmod(col_num - 1, 26)
        result = chr(65 + rem) + result
    return result


def safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except Exception:
        return None


def safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den == 0:
        return None
    return num / den


def clip(value: Optional[float], low: float, high: float) -> Optional[float]:
    if value is None:
        return None
    return max(low, min(high, value))


def average(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def median_or_single(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None and v > 0]
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    return float(np.median(vals))


def normalize_code(code: Any) -> str:
    text = str(code).strip()
    return text if text.endswith(".T") else f"{text}.T"


def parse_datetime_jst(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(text, fmt)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=JST)
            return dt.astimezone(JST)
        except Exception:
            continue
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=JST)
        return dt.astimezone(JST)
    except Exception:
        return None


def should_refresh_db(existing_db: Dict[str, Any], force: bool = False) -> bool:
    if force:
        return True
    last_updated = parse_datetime_jst(existing_db.get("last_db_update_jst"))
    if not last_updated:
        return True
    now = datetime.now(JST)
    age_days = (now - last_updated).days
    return age_days >= 7


def market_cap_size_premium(market_cap: Optional[float]) -> float:
    if market_cap is None:
        return 0.02
    if market_cap >= 1_000_000_000_000:  # 1兆円
        return 0.0
    if market_cap >= 300_000_000_000:    # 3000億円
        return 0.015
    return 0.02


def first_matching_row(df: Optional[pd.DataFrame], labels: List[str]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    for label in labels:
        if label in df.index:
            return df.loc[label]
    lower_map = {str(idx).lower(): idx for idx in df.index}
    for label in labels:
        idx = lower_map.get(label.lower())
        if idx is not None:
            return df.loc[idx]
    return None


def _sorted_numeric_series(row: pd.Series, newest_first: bool = True) -> pd.Series:
    cleaned = pd.to_numeric(row, errors="coerce")
    try:
        cleaned.index = pd.to_datetime(cleaned.index)
        cleaned = cleaned.sort_index(ascending=not newest_first)
    except Exception:
        pass
    return cleaned.dropna()


def latest_series_value(df: Optional[pd.DataFrame], labels: List[str]) -> Optional[float]:
    row = first_matching_row(df, labels)
    if row is None:
        return None
    cleaned = _sorted_numeric_series(row, newest_first=True)
    if cleaned.empty:
        return None
    return safe_float(cleaned.iloc[0])


def sum_recent_quarters(df: Optional[pd.DataFrame], labels: List[str], limit: int = 4) -> Optional[float]:
    row = first_matching_row(df, labels)
    if row is None:
        return None
    cleaned = _sorted_numeric_series(row, newest_first=True)
    if cleaned.empty:
        return None
    try:
        idx = pd.to_datetime(cleaned.index)
        if len(idx) > 0:
            latest_dt = idx[0]
            cutoff = latest_dt - pd.Timedelta(days=400)
            cleaned = cleaned[idx >= cutoff]
    except Exception:
        pass
    if len(cleaned) < limit:
        return None
    return safe_float(cleaned.iloc[:limit].sum())


def latest_point_in_time(df: Optional[pd.DataFrame], labels: List[str]) -> Optional[float]:
    return latest_series_value(df, labels)


def get_annual_values(df: Optional[pd.DataFrame], labels: List[str], periods: int = 3) -> List[Optional[float]]:
    row = first_matching_row(df, labels)
    if row is None:
        return []
    cleaned = _sorted_numeric_series(row, newest_first=True)
    values = []
    for v in cleaned.iloc[:periods]:
        values.append(safe_float(v))
    return values


def normalize_capex(raw_capex: Optional[float]) -> Optional[float]:
    if raw_capex is None:
        return None
    return -abs(raw_capex)


def compute_cagr(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    last = vals[0]
    max_years = min(3, len(vals) - 1)
    base = vals[max_years]
    years = max_years
    if base is None or last is None or base <= 0 or last <= 0 or years <= 0:
        return None
    try:
        return (last / base) ** (1 / years) - 1
    except Exception:
        return None


def get_optional_config_rate(config: Dict[str, Any], key: str, default: float) -> float:
    return safe_float(config.get(key)) or default


def parse_missing_fields(value: Any) -> Set[str]:
    if value is None:
        return set()
    text = str(value).strip()
    if not text:
        return set()
    return {part.strip() for part in text.split(",") if part.strip()}


def compute_annual_roe_series(income_stmt: Optional[pd.DataFrame], balance_sheet: Optional[pd.DataFrame]) -> List[Optional[float]]:
    ni = get_annual_values(income_stmt, LABELS["net_income"], periods=4)
    eq = get_annual_values(balance_sheet, LABELS["equity"], periods=4)
    results: List[Optional[float]] = []
    max_periods = min(len(ni), len(eq))
    for i in range(max_periods):
        curr_eq = eq[i]
        next_eq = eq[i + 1] if i + 1 < len(eq) else None
        avg_eq = average([curr_eq, next_eq]) if next_eq is not None else curr_eq
        results.append(safe_div(ni[i], avg_eq))
    return results[:3]


def compute_annual_roic_series(income_stmt: Optional[pd.DataFrame], balance_sheet: Optional[pd.DataFrame], tax_rate: float) -> List[Optional[float]]:
    ebit = get_annual_values(income_stmt, LABELS["ebit"], periods=4)
    eq = get_annual_values(balance_sheet, LABELS["equity"], periods=4)
    cash = get_annual_values(balance_sheet, LABELS["cash"], periods=4)
    debt = get_annual_values(balance_sheet, LABELS["debt"], periods=4)

    results: List[Optional[float]] = []
    max_periods = min(len(ebit), len(eq), len(cash), len(debt))
    for i in range(max_periods):
        nopat = None if ebit[i] is None else ebit[i] * (1 - tax_rate)
        invested = None
        if eq[i] is not None or debt[i] is not None or cash[i] is not None:
            invested = (eq[i] or 0.0) + (debt[i] or 0.0) - (cash[i] or 0.0)
        if invested is not None and invested <= 0:
            results.append(None)
        else:
            results.append(safe_div(nopat, invested))
    return results[:3]


def compute_two_year_weekly_beta(ticker: str, benchmark: str = BENCHMARK_TICKER) -> Optional[float]:
    try:
        prices = yf.download(
            [ticker, benchmark],
            period="2y",
            interval="1wk",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if prices.empty:
            return None
        closes = prices["Close"] if isinstance(prices.columns, pd.MultiIndex) else prices
        if ticker not in closes.columns or benchmark not in closes.columns:
            return None
        returns = closes[[ticker, benchmark]].pct_change().dropna()
        if len(returns) < 52:
            return None
        cov = np.cov(returns[ticker], returns[benchmark])[0, 1]
        var = np.var(returns[benchmark])
        if var == 0:
            return None
        beta_raw = cov / var
        beta = 0.67 * beta_raw + 0.33 * 1.0
        beta = clip(beta, 0.30, 2.00)
        return safe_float(beta)
    except Exception:
        return None


def detect_financial_flag(sector_raw: Optional[str], industry_raw: Optional[str], quote_type: Optional[str]) -> int:
    if str(quote_type or "").upper() in {"ETF", "MUTUALFUND"}:
        return 0
    text = f"{sector_raw or ''} {industry_raw or ''}".lower()
    if any(keyword in text for keyword in ["reit", "real estate investment trust", "不動産投資信託"]):
        return 0
    return 1 if any(keyword.lower() in text for keyword in FINANCIAL_KEYWORDS) else 0


def with_note(notes: List[str], message: str) -> None:
    if message not in notes:
        notes.append(message)


def get_info_value(info: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        value = info.get(key)
        if value not in (None, "", "None"):
            return value
    return None


def _extract_percent_from_text(text: str) -> Optional[float]:
    patterns = [
        r"JP10YT=XX.{0,300}?(-?\d+(?:\.\d+)?)\s*%",
        r"(?:日本\s*10年|日本10年|10年国債|長期金利).{0,200}?(-?\d+(?:\.\d+)?)\s*%",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            value = safe_float(m.group(1))
            if value is not None:
                rate = value / 100.0
                if 0.0 <= rate <= 0.10:
                    return rate
    return None


def fetch_rf_rate_japan_from_sbi() -> float:
    global _RF_RATE_CACHE, _RF_RATE_SOURCE
    if _RF_RATE_CACHE is not None:
        return _RF_RATE_CACHE

    for attempt in range(3):
        headers = {
            "User-Agent": random.choice(SBI_USER_AGENTS),
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
        try:
            response = requests.get(SBI_JP10Y_URL, headers=headers, timeout=10)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or response.encoding
            html = response.text

            rate = _extract_percent_from_text(html)
            if rate is None:
                try:
                    tables = pd.read_html(StringIO(html))
                except Exception:
                    tables = []
                for table in tables:
                    for _, row in table.astype(str).iterrows():
                        row_text = " ".join(row.tolist())
                        if any(label in row_text for label in ["JP10YT=XX", "日本 10年", "日本10年", "10年国債", "長期金利"]):
                            match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", row_text)
                            if match:
                                value = safe_float(match.group(1))
                                if value is not None:
                                    candidate = value / 100.0
                                    if 0.0 <= candidate <= 0.10:
                                        rate = candidate
                                        break
                    if rate is not None:
                        break
            if rate is None:
                for label in ["日本 10年", "日本10年", "10年国債", "長期金利"]:
                    match = re.search(label + r".{0,200}?(-?\d+(?:\.\d+)?)\s*%", html, flags=re.IGNORECASE | re.DOTALL)
                    if match:
                        value = safe_float(match.group(1))
                        if value is not None:
                            candidate = value / 100.0
                            if 0.0 <= candidate <= 0.10:
                                rate = candidate
                                break

            if rate is not None:
                _RF_RATE_CACHE = rate
                _RF_RATE_SOURCE = "SBI"
                return rate
        except Exception:
            pass

        if attempt < 2:
            time.sleep(2 ** attempt)

    _RF_RATE_CACHE = 0.015
    _RF_RATE_SOURCE = "FALLBACK"
    return _RF_RATE_CACHE


def fetch_ticker_data(ticker: str, refresh_full: bool, config: Dict[str, Any]) -> Dict[str, Any]:
    tk = yf.Ticker(ticker)

    info: Dict[str, Any] = {}
    missing_fields: List[str] = []
    notes: List[str] = []
    calc_error = ""

    try:
        info = tk.info or {}
    except Exception as exc:
        info = {}
        with_note(notes, f"info取得失敗: {exc}")

    try:
        fast_info = dict(tk.fast_info) if getattr(tk, "fast_info", None) else {}
    except Exception:
        fast_info = {}

    try:
        price_hist = tk.history(period="5d", auto_adjust=False)
    except Exception:
        price_hist = pd.DataFrame()

    if refresh_full:
        try:
            income_stmt = tk.income_stmt
        except Exception:
            income_stmt = pd.DataFrame()
        try:
            quarterly_income_stmt = tk.quarterly_income_stmt
        except Exception:
            quarterly_income_stmt = pd.DataFrame()
        try:
            balance_sheet = tk.balance_sheet
        except Exception:
            balance_sheet = pd.DataFrame()
        try:
            quarterly_balance_sheet = tk.quarterly_balance_sheet
        except Exception:
            quarterly_balance_sheet = pd.DataFrame()
        try:
            cashflow = tk.cashflow
        except Exception:
            cashflow = pd.DataFrame()
        try:
            quarterly_cashflow = tk.quarterly_cashflow
        except Exception:
            quarterly_cashflow = pd.DataFrame()
        try:
            dividends = tk.dividends
        except Exception:
            dividends = pd.Series(dtype="float64")
    else:
        income_stmt = quarterly_income_stmt = balance_sheet = quarterly_balance_sheet = pd.DataFrame()
        cashflow = quarterly_cashflow = pd.DataFrame()
        dividends = pd.Series(dtype="float64")

    current_price = None
    if not price_hist.empty and "Close" in price_hist.columns:
        current_price = safe_float(price_hist["Close"].dropna().iloc[-1]) if not price_hist["Close"].dropna().empty else None
    if current_price is None:
        current_price = safe_float(
            get_info_value(fast_info, ["lastPrice", "regularMarketPrice", "last_price"])
            or get_info_value(info, ["currentPrice", "regularMarketPrice", "previousClose"])
        )
    market_cap = safe_float(
        get_info_value(fast_info, ["marketCap"])
        or get_info_value(info, ["marketCap"])
    )
    shares = safe_float(
        get_info_value(fast_info, ["shares"])
        or get_info_value(info, ["sharesOutstanding", "impliedSharesOutstanding"])
    )
    enterprise_value = safe_float(get_info_value(info, ["enterpriseValue"]))
    sector_raw = get_info_value(info, ["sector", "sectorDisp"])
    industry_raw = get_info_value(info, ["industry", "industryDisp"])
    quote_type = get_info_value(info, ["quoteType"])
    market = get_info_value(info, ["exchange", "fullExchangeName", "market"])
    currency = get_info_value(info, ["currency"])
    trailing_dividend_rate = safe_float(get_info_value(info, ["trailingAnnualDividendRate", "dividendRate"]))
    dividend_yield = safe_float(get_info_value(info, ["dividendYield"]))
    raw_beta = safe_float(get_info_value(info, ["beta"]))

    if refresh_full:
        revenue_ttm = sum_recent_quarters(quarterly_income_stmt, LABELS["revenue"])
        if revenue_ttm is None:
            revenue_ttm = latest_series_value(income_stmt, LABELS["revenue"])

        ebit_ttm = sum_recent_quarters(quarterly_income_stmt, LABELS["ebit"])
        if ebit_ttm is None:
            ebit_ttm = latest_series_value(income_stmt, LABELS["ebit"])

        ebitda_ttm = sum_recent_quarters(quarterly_income_stmt, LABELS["ebitda"])
        if ebitda_ttm is None:
            ebitda_ttm = latest_series_value(income_stmt, LABELS["ebitda"])

        net_income_ttm = sum_recent_quarters(quarterly_income_stmt, LABELS["net_income"])
        if net_income_ttm is None:
            net_income_ttm = latest_series_value(income_stmt, LABELS["net_income"])

        total_assets = latest_point_in_time(quarterly_balance_sheet, LABELS["assets"]) or latest_point_in_time(balance_sheet, LABELS["assets"])
        total_equity = latest_point_in_time(quarterly_balance_sheet, LABELS["equity"]) or latest_point_in_time(balance_sheet, LABELS["equity"])
        cash_and_equivalents = latest_point_in_time(quarterly_balance_sheet, LABELS["cash"]) or latest_point_in_time(balance_sheet, LABELS["cash"])
        total_debt = latest_point_in_time(quarterly_balance_sheet, LABELS["debt"]) or latest_point_in_time(balance_sheet, LABELS["debt"])

        operating_cf_ttm = sum_recent_quarters(quarterly_cashflow, LABELS["operating_cf"])
        if operating_cf_ttm is None:
            operating_cf_ttm = latest_series_value(cashflow, LABELS["operating_cf"])

        capex_ttm = sum_recent_quarters(quarterly_cashflow, LABELS["capex"])
        if capex_ttm is None:
            capex_ttm = latest_series_value(cashflow, LABELS["capex"])
        capex_ttm = normalize_capex(capex_ttm)

        fcf_ttm = None
        if operating_cf_ttm is not None and capex_ttm is not None:
            fcf_ttm = operating_cf_ttm + capex_ttm

        dps_ttm = None
        if isinstance(dividends, pd.Series) and not dividends.empty:
            recent = dividends[dividends.index >= (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=370))]
            if not recent.empty:
                dps_ttm = safe_float(recent.sum())
        if dps_ttm is None:
            dps_ttm = trailing_dividend_rate

        pretax_income_ttm = sum_recent_quarters(quarterly_income_stmt, LABELS["pretax_income"])
        tax_provision_ttm = sum_recent_quarters(quarterly_income_stmt, LABELS["tax_provision"])
        tax_rate_estimate = 0.30
        if pretax_income_ttm is not None and tax_provision_ttm is not None and pretax_income_ttm > 0 and tax_provision_ttm >= 0:
            tax_rate_estimate = clip(tax_provision_ttm / pretax_income_ttm, 0.20, 0.35) or 0.30

        nopat_ttm = None if ebit_ttm is None else ebit_ttm * (1 - tax_rate_estimate)
        net_debt = None
        if total_debt is not None or cash_and_equivalents is not None:
            net_debt = (total_debt or 0.0) - (cash_and_equivalents or 0.0)
        invested_capital = None
        if total_equity is not None or total_debt is not None or cash_and_equivalents is not None:
            invested_capital = (total_equity or 0.0) + (total_debt or 0.0) - (cash_and_equivalents or 0.0)
            if invested_capital <= 0:
                invested_capital = None
        with_note(notes, "Invested Capitalは Total Equity + Total Debt - Cash の簡略定義。持合株・投資有価証券は未調整。")

        eps_ttm = safe_float(get_info_value(info, ["trailingEps"]))
        if eps_ttm is None and shares:
            eps_ttm = safe_div(net_income_ttm, shares)
        bps = None
        if total_equity is not None and shares:
            bps = safe_div(total_equity, shares)
        payout_ratio = safe_float(get_info_value(info, ["payoutRatio"]))
        if payout_ratio is None and dps_ttm is not None and eps_ttm not in (None, 0):
            payout_ratio = safe_div(dps_ttm, eps_ttm)
        if payout_ratio is not None and payout_ratio < 0:
            payout_ratio = None
        if payout_ratio is not None and payout_ratio > 1.2:
            payout_ratio = 1.2

        pb_now = safe_div(current_price, bps)
        pe_now = safe_div(current_price, eps_ttm)

        roe_series = compute_annual_roe_series(income_stmt, balance_sheet)
        roic_series = compute_annual_roic_series(income_stmt, balance_sheet, tax_rate_estimate)
        roe_1y = roe_series[0] if len(roe_series) >= 1 else None
        roe_3y_avg = average(roe_series[:3])
        roic_1y = roic_series[0] if len(roic_series) >= 1 else None
        roic_3y_avg = average(roic_series[:3])

        roe_normalized = roe_3y_avg or average(roe_series[:2]) or roe_1y
        roic_normalized = roic_3y_avg or average(roic_series[:2]) or roic_1y

        beta = compute_two_year_weekly_beta(ticker) or raw_beta or 1.0
        beta = clip(beta, 0.30, 2.00) or 1.0
        if raw_beta is None:
            with_note(notes, "βは2年週次推定を優先し、取得不能時はinfoのbeta→1.0で代替。業種平均フォールバックは未実装。")

        rf_rate = fetch_rf_rate_japan_from_sbi()
        if _RF_RATE_SOURCE == "SBI":
            with_note(notes, "rf_rateはSBIの日本10年国債利回りを使用")
        else:
            with_note(notes, "rf_rateはSBI取得失敗のため固定値0.015を使用")
        erp = get_optional_config_rate(config, "erp_override", 0.055)
        country_risk_premium = get_optional_config_rate(config, "country_risk_premium_override", 0.0)
        size_premium = market_cap_size_premium(market_cap)
        coe = rf_rate + beta * erp + size_premium + country_risk_premium
        cod_estimate = 0.02 if total_debt and total_debt > 0 else 0.015
        debt_for_weight = total_debt or 0.0
        equity_for_weight = total_equity or 0.0
        total_capital = debt_for_weight + equity_for_weight
        if total_capital > 0:
            wacc = (
                coe * (equity_for_weight / total_capital)
                + cod_estimate * (1 - tax_rate_estimate) * (debt_for_weight / total_capital)
            )
        else:
            wacc = coe

        growth_floor = -0.02
        growth_cap = 0.15
        terminal_growth = 0.01
        gap_year_default = 5

        growth_candidates: List[Optional[float]] = []
        revenue_annual = get_annual_values(income_stmt, LABELS["revenue"], periods=4)
        revenue_cagr = compute_cagr(revenue_annual)
        if revenue_cagr is not None:
            growth_candidates.append(clip(revenue_cagr, -0.02, growth_cap))

        ebit_annual = get_annual_values(income_stmt, LABELS["ebit"], periods=4)
        nopat_annual = [None if v is None else v * (1 - tax_rate_estimate) for v in ebit_annual]
        nopat_cagr = compute_cagr(nopat_annual)
        if nopat_cagr is not None:
            growth_candidates.append(clip(nopat_cagr, -0.02, growth_cap))

        if roe_normalized is not None:
            payout_use = min(max(payout_ratio if payout_ratio is not None else 0.5, 0.0), 1.0)
            retention_growth = roe_normalized * (1 - payout_use)
            growth_candidates.append(clip(retention_growth, -0.02, growth_cap))

        analyst_growth = safe_float(get_info_value(info, ["earningsGrowth", "revenueGrowth"]))
        if analyst_growth is not None and -0.20 <= analyst_growth <= 0.20:
            growth_candidates.append(clip(analyst_growth, -0.02, growth_cap))
            with_note(notes, "analyst growthは補助候補のみ採用")

        growth_values = [v for v in growth_candidates if v is not None]
        growth_base = float(np.median(growth_values)) if growth_values else 0.03
        growth_base = clip(growth_base, growth_floor, growth_cap)

        financial_flag = detect_financial_flag(sector_raw, industry_raw, quote_type)
        with_note(notes, "fcfe_ttmは未算出（fcf_ttmとの同一視を禁止）")

        data = {
            "ticker_yf": ticker,
            "market": market,
            "currency": currency,
            "quote_type": quote_type,
            "sector_raw": sector_raw,
            "industry_raw": industry_raw,
            "financial_flag": financial_flag,
            "data_status": "OK",
            "last_db_update_jst": datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S"),
            "current_price": current_price,
            "market_cap": market_cap,
            "shares_outstanding": shares,
            "enterprise_value": enterprise_value,
            "beta": beta,
            "dividend_yield": dividend_yield,
            "trailing_annual_dividend_rate": trailing_dividend_rate,
            "revenue_ttm": revenue_ttm,
            "ebit_ttm": ebit_ttm,
            "ebitda_ttm": ebitda_ttm,
            "net_income_ttm": net_income_ttm,
            "nopat_ttm": nopat_ttm,
            "total_assets": total_assets,
            "total_equity": total_equity,
            "cash_and_equivalents": cash_and_equivalents,
            "total_debt": total_debt,
            "net_debt": net_debt,
            "invested_capital": invested_capital,
            "operating_cf_ttm": operating_cf_ttm,
            "capex_ttm": capex_ttm,
            "fcf_ttm": fcf_ttm,
            "fcfe_ttm": None,
            "eps_ttm": eps_ttm,
            "bps": bps,
            "dps_ttm": dps_ttm,
            "payout_ratio": payout_ratio,
            "pb_now": pb_now,
            "pe_now": pe_now,
            "roe_1y": roe_1y,
            "roe_3y_avg": roe_3y_avg,
            "roic_1y": roic_1y,
            "roic_3y_avg": roic_3y_avg,
            "roe_normalized": roe_normalized,
            "roic_normalized": roic_normalized,
            "rf_rate": rf_rate,
            "erp": erp,
            "country_risk_premium": country_risk_premium,
            "size_premium": size_premium,
            "wacc": wacc,
            "coe": coe,
            "cod_estimate": cod_estimate,
            "tax_rate_estimate": tax_rate_estimate,
            "growth_base": growth_base,
            "growth_floor": growth_floor,
            "growth_cap": growth_cap,
            "terminal_growth": terminal_growth,
            "gap_year_default": gap_year_default,
            "financial_roe_avg": roe_normalized if financial_flag == 1 else None,
            "financial_payout_avg": payout_ratio if financial_flag == 1 else None,
            "missing_fields": "",
            "calc_error": "",
            "notes": " | ".join(notes),
        }

        if financial_flag == 1:
            required_keys = [
                "current_price", "bps", "roe_normalized", "coe"
            ]
        else:
            required_keys = [
                "current_price", "market_cap", "shares_outstanding", "bps",
                "roic_normalized", "wacc", "nopat_ttm"
            ]
        missing_fields = [k for k in required_keys if data.get(k) is None]
    else:
        data = {
            "ticker_yf": ticker,
            "current_price": current_price,
            "market_cap": market_cap,
            "enterprise_value": enterprise_value,
            "beta": raw_beta,
            "sector_raw": sector_raw,
            "industry_raw": industry_raw,
            "market": market,
            "currency": currency,
            "quote_type": quote_type,
            "dividend_yield": dividend_yield,
            "trailing_annual_dividend_rate": trailing_dividend_rate,
            "data_status": "OK",
        }

    data["missing_fields"] = ",".join(missing_fields)
    data["calc_error"] = calc_error
    if not data.get("data_status"):
        data["data_status"] = "OK"
    return data


def merge_db(existing_db: Dict[str, Any], fresh: Dict[str, Any], refresh_full: bool) -> Dict[str, Any]:
    merged = {key: existing_db.get(key) for key in DB_HEADERS}
    for key, value in fresh.items():
        if value is not None and value != "":
            merged[key] = value
        elif refresh_full and key in DB_HEADERS and key not in ("financial_flag_override",):
            merged[key] = value

    current_price = safe_float(merged.get("current_price"))
    bps = safe_float(merged.get("bps"))
    eps_ttm = safe_float(merged.get("eps_ttm"))
    merged["pb_now"] = safe_div(current_price, bps)
    merged["pe_now"] = safe_div(current_price, eps_ttm)

    defaults = {
        "growth_floor": -0.02,
        "growth_cap": 0.15,
        "terminal_growth": 0.01,
        "gap_year_default": 5,
        "rf_rate": 0.015,
        "erp": 0.055,
        "country_risk_premium": 0.0,
        "tax_rate_estimate": 0.30,
    }
    for key, value in defaults.items():
        if merged.get(key) in (None, ""):
            merged[key] = value

    if merged.get("data_status") in (None, ""):
        merged["data_status"] = "OK"
    return merged


# 既存ロジックは流用可能箇所として残すが、新表示列の算出では使用しない
def compute_nonfinancial_fair_price(db: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], List[str]]:
    notes: List[str] = []
    roic = safe_float(db.get("roic_normalized"))
    wacc = safe_float(db.get("wacc"))
    invested_capital = safe_float(db.get("invested_capital"))
    shares = safe_float(db.get("shares_outstanding"))
    net_debt = safe_float(db.get("net_debt")) or 0.0
    nopat = safe_float(db.get("nopat_ttm"))
    g = safe_float(db.get("growth_base"))
    g_floor = safe_float(db.get("growth_floor")) or 0.0
    g_cap = safe_float(db.get("growth_cap")) or 0.15
    g_term = safe_float(db.get("terminal_growth")) or 0.01
    target_ev = safe_float(db.get("enterprise_value"))
    market_cap = safe_float(db.get("market_cap"))

    if target_ev is None and market_cap is not None:
        target_ev = market_cap + net_debt

    ep_price = None
    vdf_price = None
    implied_growth = None
    implied_gap_years = None

    if all(v is not None for v in [roic, wacc, invested_capital, shares]) and shares and wacc > g_term:
        try:
            ev = invested_capital
            ic = invested_capital
            spread = roic - wacc
            for year in range(1, 6):
                ic = ic * (1 + (g or 0.03))
                ep = spread * ic
                ev += ep / ((1 + wacc) ** year)
            terminal_ic = ic * (1 + g_term)
            terminal_ep = spread * terminal_ic
            ev += (terminal_ep / (wacc - g_term)) / ((1 + wacc) ** 5)
            equity_value = ev - net_debt
            ep_price = safe_div(equity_value, shares)
        except Exception as exc:
            with_note(notes, f"Economic Profit計算失敗: {exc}")

    if all(v is not None for v in [roic, wacc, nopat, shares]) and shares:
        try:
            if g is not None and roic > g and wacc > g:
                ev = nopat * (1 - g / roic) / (wacc - g)
                equity_value = ev - net_debt
                vdf_price = safe_div(equity_value, shares)
                if safe_float(db.get("cash_and_equivalents")) and ev is not None and safe_float(db.get("cash_and_equivalents")) > ev:
                    with_note(notes, "Cash > EV のキャッシュリッチ状態。Value Driver評価が過大になり得ます。")
            else:
                with_note(notes, "Value Driver Formulaは ROIC <= g または WACC <= g のため算出不能。")
        except Exception as exc:
            with_note(notes, f"Value Driver計算失敗: {exc}")

    if all(v is not None for v in [roic, wacc, nopat, target_ev]) and target_ev and wacc > 0:
        def ev_by_growth(growth: float) -> float:
            if roic <= growth or wacc <= growth:
                return -1e18
            return nopat * (1 - growth / roic) / (wacc - growth)

        upper = min(g_cap, roic - 1e-4, wacc - 1e-4)
        if upper > g_floor + 1e-4:
            try:
                f_low = ev_by_growth(g_floor) - target_ev
                f_high = ev_by_growth(upper) - target_ev
                if math.isnan(f_low) or math.isnan(f_high):
                    implied_growth = None
                elif f_low == 0:
                    implied_growth = g_floor
                elif f_low * f_high < 0:
                    implied_growth = bisect(lambda x: ev_by_growth(x) - target_ev, g_floor, upper, maxiter=100)
            except Exception as exc:
                with_note(notes, f"Reverse DCF成長率計算失敗: {exc}")

    if all(v is not None for v in [roic, wacc, invested_capital, target_ev]) and wacc > 0:
        try:
            growth_assumption = g or 0.03

            def ev_with_gap_years(n_years: float) -> float:
                n_full = int(math.floor(max(n_years, 0)))
                frac = max(n_years - n_full, 0)
                ev = invested_capital
                ic = invested_capital
                spread = roic - wacc
                for year in range(1, n_full + 1):
                    ic = ic * (1 + growth_assumption)
                    ep = spread * ic
                    ev += ep / ((1 + wacc) ** year)
                if frac > 0:
                    ic = ic * (1 + growth_assumption * frac)
                    ep = spread * ic * frac
                    ev += ep / ((1 + wacc) ** (n_full + frac))
                return ev

            values = [(x, ev_with_gap_years(float(x))) for x in range(0, 41)]
            greater = [years for years, ev in values if ev >= target_ev]
            implied_gap_years = float(greater[0]) if greater else None
        except Exception as exc:
            with_note(notes, f"Reverse DCF GAP年数計算失敗: {exc}")

    return ep_price, vdf_price, implied_growth, implied_gap_years, notes


def compute_financial_fair_price(db: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], str, List[str]]:
    notes: List[str] = []
    current_price = safe_float(db.get("current_price"))
    bps = safe_float(db.get("bps"))
    payout = safe_float(db.get("payout_ratio"))
    roe = safe_float(db.get("roe_normalized"))
    coe = safe_float(db.get("coe"))
    terminal_growth = safe_float(db.get("terminal_growth")) or 0.01
    pb_now = safe_float(db.get("pb_now"))

    if payout is not None and payout > 1.2:
        payout = 1.2
    if payout is not None and payout < 0:
        payout = None

    justified_pbr = None
    fair_price = None
    implied_roe = None
    spread = None
    judgement = "算出不能"

    if bps is None:
        with_note(notes, "金融算出不能: BPS不足。")
    if roe is None:
        with_note(notes, "金融算出不能: 平準化ROE不足。")
    if coe is None:
        with_note(notes, "金融算出不能: CoE不足。")

    if roe is not None and coe is not None:
        spread = roe - coe

    if all(v is not None for v in [roe, coe]):
        g = roe * (1 - (payout if payout is not None else 0.5))
        if coe > g:
            justified_pbr = safe_div((roe - g), (coe - g))
            if justified_pbr is not None and justified_pbr < 0:
                justified_pbr = None
        else:
            with_note(notes, "金融算出不能: CoE <= g のため正当PBR算出不能。")
            with_note(notes, "金融正当PBRは CoE <= g のため算出不能。")

    if justified_pbr is not None and bps is not None:
        fair_price = bps * justified_pbr

    if all(v is not None for v in [pb_now, coe]):
        implied_roe = pb_now * (coe - terminal_growth) + terminal_growth

    if fair_price is None or current_price is None or roe is None or coe is None:
        judgement = "算出不能"
    elif roe <= coe:
        judgement = "改善待ち"
    elif implied_roe is not None and implied_roe > max(roe + 0.03, roe * 1.25):
        judgement = "市場期待過大"
    elif current_price < fair_price * 0.8:
        judgement = "割安候補"
    elif current_price <= fair_price * 1.2:
        judgement = "妥当"
    else:
        judgement = "期待先行"

    return coe, roe, spread, justified_pbr, fair_price, implied_roe, judgement, notes


def compute_decay_ep_price(db: Dict[str, Any]) -> Optional[float]:
    roic_normalized = safe_float(db.get("roic_normalized"))
    roic_1y = safe_float(db.get("roic_1y"))
    wacc = safe_float(db.get("wacc"))
    coe = safe_float(db.get("coe"))
    invested_capital = safe_float(db.get("invested_capital"))
    shares = safe_float(db.get("shares_outstanding"))
    net_debt = safe_float(db.get("net_debt")) or 0.0
    growth_base = safe_float(db.get("growth_base"))

    roic_candidates = [v for v in [roic_normalized, roic_1y] if v is not None]
    if not roic_candidates or invested_capital in (None, 0) or shares in (None, 0):
        return None
    roic_use = min(roic_candidates)

    discount_candidates = [v for v in [
        wacc,
        (coe - 0.01) if coe is not None else None,
        0.07,
    ] if v is not None]
    if not discount_candidates:
        return None
    discount_use = max(discount_candidates)

    spread = roic_use - discount_use
    if spread > 0.15:
        decay_years = 7
    elif spread > 0.08:
        decay_years = 5
    else:
        decay_years = 3

    growth_candidates = [
        clip(growth_base, -0.02, 0.04),
        clip(0.03, -0.02, 0.04),
    ]
    growth_vals = [v for v in growth_candidates if v is not None]
    if not growth_vals:
        return None
    growth_use = float(np.median(growth_vals))

    ev = invested_capital
    ic = invested_capital

    for year in range(1, decay_years + 1):
        ic = ic * (1 + growth_use)
        decay_factor = (decay_years + 1 - year) / decay_years
        ep = spread * decay_factor * ic
        ev += ep / ((1 + discount_use) ** year)

    equity_value = ev - net_debt
    if equity_value <= 0:
        return None
    return safe_div(equity_value, shares)


def compute_profit_anchor_price(db: Dict[str, Any]) -> Optional[float]:
    eps_ttm = safe_float(db.get("eps_ttm"))
    roic_normalized = safe_float(db.get("roic_normalized"))
    wacc = safe_float(db.get("wacc"))
    growth_base = safe_float(db.get("growth_base"))
    if eps_ttm is None or roic_normalized is None or wacc is None:
        return None
    spread = max(roic_normalized - wacc, 0.0)
    growth_use = clip(max(growth_base or 0.0, 0.0), 0.0, 0.15) or 0.0
    pe_target = clip(10 + 40 * spread + 25 * growth_use, 8, 25)
    price = eps_ttm * pe_target
    if price <= 0:
        return None
    return price


def compute_asset_anchor_price(db: Dict[str, Any]) -> Optional[float]:
    bps = safe_float(db.get("bps"))
    roe_normalized = safe_float(db.get("roe_normalized"))
    roe_1y = safe_float(db.get("roe_1y"))
    coe = safe_float(db.get("coe"))
    if bps is None:
        return None

    roe_use = roe_normalized if roe_normalized is not None else roe_1y
    if roe_use is not None and coe is not None:
        spread = max(roe_use - coe, 0.0)
        pb_target = clip(0.8 + 8 * spread, 0.8, 2.5)
        price = bps * pb_target
        if price > 0:
            return price

    roic_normalized = safe_float(db.get("roic_normalized"))
    wacc = safe_float(db.get("wacc"))
    if roic_normalized is None or wacc is None:
        return None
    spread = max(roic_normalized - wacc, 0.0)
    pb_target = clip(0.8 + 8 * spread, 0.8, 2.5)
    price = bps * pb_target
    if price <= 0:
        return None
    return price


def compute_conservative_pbr_price(db: Dict[str, Any]) -> Optional[float]:
    roe_normalized = safe_float(db.get("roe_normalized"))
    roe_1y = safe_float(db.get("roe_1y"))
    coe = safe_float(db.get("coe"))
    bps = safe_float(db.get("bps"))
    payout_ratio = safe_float(db.get("payout_ratio"))

    roe_candidates = [v for v in [roe_normalized, roe_1y] if v is not None]
    if not roe_candidates or bps is None or coe is None or payout_ratio is None:
        return None
    if payout_ratio < 0:
        return None
    payout_use = min(payout_ratio, 1.0)
    roe_use = min(roe_candidates)
    coe_use = max(coe, 0.08)

    retained_growth = roe_use * (1 - payout_use)
    g_candidates = [v for v in [retained_growth, 0.02, coe_use - 0.02] if v is not None]
    if not g_candidates:
        return None
    g_use = clip(min(g_candidates), 0.0, 0.02)
    if coe_use <= g_use:
        return None

    pbr_raw = safe_div((roe_use - g_use), (coe_use - g_use))
    if pbr_raw is None or pbr_raw < 0:
        return None
    pbr_target = clip(pbr_raw, 0.6, 2.2)
    price = bps * pbr_target
    if price <= 0:
        return None
    return price


def compute_dividend_discount_price(db: Dict[str, Any]) -> Optional[float]:
    dps_ttm = safe_float(db.get("dps_ttm"))
    coe = safe_float(db.get("coe"))
    payout_ratio = safe_float(db.get("payout_ratio"))
    roe_normalized = safe_float(db.get("roe_normalized"))

    if dps_ttm is None or coe is None or payout_ratio is None or roe_normalized is None:
        return None
    if payout_ratio < 0:
        return None
    payout_use = min(payout_ratio, 1.0)
    coe_use = max(coe, 0.08)
    retained_growth = roe_normalized * (1 - payout_use)
    g_candidates = [v for v in [retained_growth, 0.015, coe_use - 0.02] if v is not None]
    if not g_candidates:
        return None
    g_div = clip(min(g_candidates), 0.0, 0.015)
    if coe_use <= g_div:
        return None

    price = safe_div(dps_ttm, (coe_use - g_div))
    if price is None or price <= 0:
        return None
    return price


def compute_financial_profit_anchor_price(db: Dict[str, Any]) -> Optional[float]:
    eps_ttm = safe_float(db.get("eps_ttm"))
    roe_normalized = safe_float(db.get("roe_normalized"))
    coe = safe_float(db.get("coe"))
    if eps_ttm is None or roe_normalized is None or coe is None:
        return None
    spread = max(roe_normalized - coe, 0.0)
    pe_target = clip(7 + 35 * spread, 6, 12)
    price = eps_ttm * pe_target
    if price <= 0:
        return None
    return price


def compute_model_confidence(candidate_prices: List[Optional[float]], db: Dict[str, Any]) -> str:
    vals = [v for v in candidate_prices if v is not None and v > 0]
    if not vals:
        return "低"

    override = str(db.get("financial_flag_override") or "").strip()
    auto_financial_flag = safe_int(db.get("financial_flag")) or 0
    if override in {"0", "1"}:
        financial_flag = int(override)
    else:
        financial_flag = auto_financial_flag

    missing = parse_missing_fields(db.get("missing_fields"))
    if financial_flag == 1:
        critical = {"current_price", "bps", "roe_normalized", "coe"}
    else:
        critical = {"current_price", "bps", "roic_normalized", "wacc", "nopat_ttm"}
    quality_penalty = len(critical & missing)

    last_update = parse_datetime_jst(db.get("last_db_update_jst"))
    stale = True if last_update is None else (datetime.now(JST) - last_update).days > 30

    if quality_penalty >= 2 or len(vals) == 1:
        return "低"

    cv = None
    if len(vals) >= 2:
        mean_val = float(np.mean(vals))
        if mean_val > 0:
            cv = float(np.std(vals) / mean_val)

    if len(vals) >= 3 and cv is not None and cv <= 0.20 and not stale and quality_penalty == 0:
        return "高"
    if len(vals) >= 2 and cv is not None and cv <= 0.40 and quality_penalty <= 1:
        return "中"
    return "低"


def compute_outputs(db: Dict[str, Any]) -> Dict[str, Any]:
    current_price = safe_float(db.get("current_price"))
    override = str(db.get("financial_flag_override") or "").strip()
    auto_financial_flag = safe_int(db.get("financial_flag")) or 0
    if override in {"0", "1"}:
        financial_flag = int(override)
    else:
        financial_flag = auto_financial_flag

    decay_ep_price = None
    profit_anchor_price = None
    asset_anchor_price = None
    conservative_pbr_price = None
    dividend_discount_price = None
    financial_profit_anchor_price = None

    if financial_flag == 1:
        conservative_pbr_price = compute_conservative_pbr_price(db)
        dividend_discount_price = compute_dividend_discount_price(db)
        financial_profit_anchor_price = compute_financial_profit_anchor_price(db)
        candidate_prices = [conservative_pbr_price, dividend_discount_price, financial_profit_anchor_price]
    else:
        decay_ep_price = compute_decay_ep_price(db)
        profit_anchor_price = compute_profit_anchor_price(db)
        asset_anchor_price = compute_asset_anchor_price(db)
        candidate_prices = [decay_ep_price, profit_anchor_price, asset_anchor_price]

    fair_price = median_or_single(candidate_prices)
    buy_limit_price = None
    if fair_price is not None:
        buy_limit_price = fair_price * (0.85 if financial_flag == 1 else 0.80)

    diff_rate = safe_div((current_price - fair_price), fair_price) if current_price is not None and fair_price else None
    buy_limit_diff_rate = safe_div((current_price - buy_limit_price), buy_limit_price) if current_price is not None and buy_limit_price else None
    confidence = compute_model_confidence(candidate_prices, db)

    roe_normalized = safe_float(db.get("roe_normalized"))
    coe = safe_float(db.get("coe"))
    roic_normalized = safe_float(db.get("roic_normalized"))
    wacc = safe_float(db.get("wacc"))

    if fair_price is None or current_price is None:
        overall_judgement = "算出不能"
    elif financial_flag == 1 and (roe_normalized is None or coe is None or roe_normalized <= coe):
        overall_judgement = "見送り"
    elif financial_flag == 0 and (roic_normalized is None or wacc is None or roic_normalized <= wacc):
        overall_judgement = "見送り"
    elif buy_limit_price is None:
        overall_judgement = "算出不能"
    elif current_price <= buy_limit_price * 0.90:
        overall_judgement = "強い割安"
    elif current_price <= buy_limit_price:
        overall_judgement = "割安"
    elif current_price <= fair_price * 0.90:
        overall_judgement = "やや割安"
    elif current_price <= fair_price * 1.10:
        overall_judgement = "妥当"
    elif current_price <= fair_price * 1.30:
        overall_judgement = "やや割高"
    elif current_price <= fair_price * 1.60:
        overall_judgement = "割高"
    else:
        overall_judgement = "かなり割高"

    return {
        "適正株価": fair_price,
        "買い上限株価": buy_limit_price,
        "現在株価との差異率": diff_rate,
        "買い上限との差異率": buy_limit_diff_rate,
        "総合判定": overall_judgement,
        "減衰EP株価": decay_ep_price,
        "利益アンカー株価": profit_anchor_price,
        "純資産アンカー株価": asset_anchor_price,
        "保守PBR株価": conservative_pbr_price,
        "配当割引株価": dividend_discount_price,
        "金融利益アンカー株価": financial_profit_anchor_price,
        "金融業種フラグ": financial_flag,
        "モデル信頼度": confidence,
    }


def ensure_headers(ws: gspread.Worksheet) -> Tuple[Dict[str, int], Dict[str, int]]:
    ws.update("E1:Q1", [EVAL_HEADERS], value_input_option="USER_ENTERED")
    db_start_col = 27  # AA
    db_end_col = db_start_col + len(DB_HEADERS) - 1
    db_range = f"AA1:{column_letter(db_end_col)}1"
    ws.update(db_range, [DB_HEADERS], value_input_option="USER_ENTERED")

    header_row = ws.row_values(1)
    header_map = {name: idx + 1 for idx, name in enumerate(header_row) if name}

    eval_positions = {header: header_map[header] for header in EVAL_HEADERS if header in header_map}
    db_positions = {header: header_map[header] for header in DB_HEADERS if header in header_map}
    return eval_positions, db_positions


def row_to_db_dict(header_row: List[str], row_values: List[str]) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for idx, value in enumerate(row_values):
        if idx < len(header_row):
            header = header_row[idx]
            if header:
                data[header] = value
    return data


def serialize_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if math.isnan(value) or math.isinf(value):
            return ""
        return float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return str(value)


def main() -> None:
    config = load_config()
    gc = get_client(config)
    spreadsheet = gc.open_by_url(config["spreadsheet_url"])
    ws = spreadsheet.worksheet(config["sheet_name"])

    ensure_headers(ws)
    header_row = ws.row_values(1)

    input_rows = ws.get("A2:D")
    if not input_rows:
        return

    last_row = len(input_rows) + 1
    db_last_col_letter = column_letter(27 + len(DB_HEADERS) - 1)
    existing_full_rows = ws.get(f"A2:{db_last_col_letter}{last_row}")

    output_matrix: List[List[Any]] = []
    db_matrix: List[List[Any]] = []

    force_db_refresh = bool(config.get("force_db_refresh", False))

    for row_idx, row in enumerate(input_rows, start=2):
        full_row = existing_full_rows[row_idx - 2] if row_idx - 2 < len(existing_full_rows) else []
        existing_db = row_to_db_dict(header_row, full_row)

        code = str(row[0]).strip() if len(row) >= 1 else ""
        if not code:
            output_matrix.append([""] * len(EVAL_HEADERS))
            db_matrix.append([""] * len(DB_HEADERS))
            continue

        ticker = normalize_code(code)
        refresh_full = should_refresh_db(existing_db, force=force_db_refresh)

        try:
            fresh = fetch_ticker_data(ticker, refresh_full=refresh_full, config=config)
            if not refresh_full:
                fresh["last_db_update_jst"] = existing_db.get("last_db_update_jst")
                if not fresh.get("financial_flag"):
                    fresh["financial_flag"] = existing_db.get("financial_flag")
                if not fresh.get("missing_fields"):
                    fresh["missing_fields"] = existing_db.get("missing_fields", "")
                if not fresh.get("notes"):
                    fresh["notes"] = existing_db.get("notes", "")
            db = merge_db(existing_db, fresh, refresh_full=refresh_full)

            if db.get("ticker_yf") in (None, ""):
                db["ticker_yf"] = ticker
            if db.get("financial_flag_override") in (None, ""):
                db["financial_flag_override"] = existing_db.get("financial_flag_override", "")

            outputs = compute_outputs(db)
        except Exception as exc:
            log_error_with_guidance(exc)
            db = {key: existing_db.get(key, "") for key in DB_HEADERS}
            db["ticker_yf"] = ticker
            db["data_status"] = "ERROR"
            db["calc_error"] = str(exc)
            outputs = {key: "" for key in EVAL_HEADERS}
            outputs["総合判定"] = "算出不能"

        output_matrix.append([serialize_cell(outputs.get(h)) for h in EVAL_HEADERS])
        db_matrix.append([serialize_cell(db.get(h)) for h in DB_HEADERS])

    eval_end_col = column_letter(5 + len(EVAL_HEADERS) - 1)  # E
    ws.update(
        f"E2:{eval_end_col}{last_row}",
        output_matrix,
        value_input_option="USER_ENTERED",
    )

    db_end_col = column_letter(27 + len(DB_HEADERS) - 1)  # AA
    ws.update(
        f"AA2:{db_end_col}{last_row}",
        db_matrix,
        value_input_option="USER_ENTERED",
    )


if __name__ == "__main__":
    main()
