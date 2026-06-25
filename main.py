import json
import logging
import math
import os
import re
import time
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Set, Tuple

import gspread
import numpy as np
import pandas as pd
import yfinance as yf
from google.oauth2.service_account import Credentials
from scipy.optimize import bisect

JST = ZoneInfo("Asia/Tokyo")
SCRIPT_VERSION = "20260625-v4-diagnostic"
APP_CONFIG_ENV = "APP_CONFIG_JSON"
BENCHMARK_TICKER = "1306.T"  # TOPIX ETF as fallback market proxy
SHEET_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
MOF_JGB_CSV_URL = "https://www.mof.go.jp/jgbs/reference/interest_rate/jgbcm.csv"
_RF_RATE_CACHE: Optional[float] = None
_RF_RATE_SOURCE = ""

EVAL_HEADERS = [
    "現在株価",                # E
    "適正株価",                # F
    "買い上限株価",            # G
    "現在株価との差異率",      # H
    "買い上限との差異率",      # I
    "総合判定",                # J
    "減衰EP株価",              # K
    "利益アンカー株価",        # L
    "純資産アンカー株価",      # M
    "保守PBR株価",             # N
    "配当割引株価",            # O
    "金融利益アンカー株価",    # P
    "金融業種フラグ",          # Q
    "モデル信頼度",            # R
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
    level=logging.INFO,
    format="%(asctime)s JST %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.Formatter.converter = lambda *args: datetime.now(JST).timetuple()
logger = logging.getLogger(__name__)


def diag(message: str, *args: Any) -> None:
    logger.info("DIAG " + message, *args)


def diag_warn(message: str, *args: Any) -> None:
    logger.warning("DIAG " + message, *args)


def sample_values(values: List[Any], limit: int = 20) -> List[Any]:
    return list(values[:limit])


def safe_len(value: Any) -> int:
    try:
        return len(value)
    except Exception:
        return 0


def error_guidance_message(exc: Exception) -> str:
    msg = str(exc)
    lower_msg = msg.lower()

    if "rf_rate取得失敗" in msg:
        return "修正方針: 財務省CSV（jgbcm.csv）の取得可否、列名（基準日・10年）、文字コード（shift-jis）を確認してください。"
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


def should_refresh_db(existing_db: Dict[str, Any], force: bool = False, refresh_days: int = 7) -> bool:
    if force:
        return True
    last_updated = parse_datetime_jst(existing_db.get("last_db_update_jst"))
    if not last_updated:
        return True
    now = datetime.now(JST)
    age_days = (now - last_updated).days
    return age_days >= refresh_days


def get_config_int(config: Dict[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    try:
        return int(value)
    except Exception:
        return default


def get_config_bool(config: Dict[str, Any], key: str, default: bool = False) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def normalize_data_status(value: Any) -> str:
    return str(value or "").strip().upper()


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
    value = safe_float(config.get(key))
    return default if value is None else value


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


def parse_japanese_era_date(text: Any) -> Optional[pd.Timestamp]:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None

    m = re.match(r"^R\s*(\d+)[\./-](\d{1,2})[\./-](\d{1,2})$", s, flags=re.IGNORECASE)
    if not m:
        return None

    try:
        year = 2018 + int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3))
        return pd.Timestamp(year=year, month=month, day=day)
    except Exception:
        return None


def fetch_rf_rate_japan_from_mof() -> float:
    global _RF_RATE_CACHE, _RF_RATE_SOURCE
    if _RF_RATE_CACHE is not None:
        return _RF_RATE_CACHE

    try:
        df = pd.read_csv(MOF_JGB_CSV_URL, skiprows=1, encoding="shift-jis")
    except Exception as exc:
        raise RuntimeError(f"rf_rate取得失敗: 財務省CSVの取得に失敗: {exc}") from exc

    df.columns = [str(col).strip() for col in df.columns]

    if "基準日" not in df.columns:
        raise RuntimeError("rf_rate取得失敗: 財務省CSVに 基準日 列が存在しない")
    if "10年" not in df.columns:
        raise RuntimeError("rf_rate取得失敗: 財務省CSVに 10年 列が存在しない")

    work = df[["基準日", "10年"]].copy()
    work["基準日_parsed"] = work["基準日"].apply(parse_japanese_era_date)
    work["10年_numeric"] = pd.to_numeric(work["10年"], errors="coerce")
    work = work.dropna(subset=["基準日_parsed", "10年_numeric"])

    if work.empty:
        raise RuntimeError("rf_rate取得失敗: 有効な基準日・10年利回り行が存在しない")

    work = work.sort_values("基準日_parsed")
    latest = work.iloc[-1]
    rate_percent = safe_float(latest["10年_numeric"])

    if rate_percent is None:
        raise RuntimeError("rf_rate取得失敗: 最新行の10年利回りが数値化できない")

    rf_rate = rate_percent / 100.0
    if not (0.0 <= rf_rate <= 0.10):
        raise RuntimeError(f"rf_rate取得失敗: 10年利回りの値が想定範囲外: {rate_percent}")

    _RF_RATE_CACHE = rf_rate
    _RF_RATE_SOURCE = "MOF"
    return rf_rate


def fetch_ticker_data(ticker: str, refresh_full: bool, config: Dict[str, Any], rf_rate: Optional[float] = None) -> Dict[str, Any]:
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

        if rf_rate is None:
            raise RuntimeError("rf_rate未設定: 財務省CSVの取得結果を渡してください")
        with_note(notes, "rf_rateは財務省CSVの10年国債利回りを使用")
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
        "現在株価": current_price,
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


def build_db_by_ticker(header_row: List[str], existing_full_rows: List[List[str]]) -> Dict[str, Dict[str, Any]]:
    db_by_ticker: Dict[str, Dict[str, Any]] = {}
    for row in existing_full_rows:
        row_db = row_to_db_dict(header_row, row)
        ticker_raw = str(row_db.get("ticker_yf") or "").strip()

        # 旧データや手動編集で ticker_yf が空でも、同じ行のA列コードから復旧できるようにする。
        # ただし、DBが空の行もここに入るため、後段で last_db_update_jst 空として初回取得対象にする。
        if not ticker_raw and len(row) >= 1:
            ticker_raw = str(row[0] or "").strip()

        if not ticker_raw:
            continue

        ticker = normalize_code(ticker_raw)
        existing = db_by_ticker.get(ticker)
        if existing is None:
            db_by_ticker[ticker] = row_db
            continue

        # 同一tickerのDB行が複数ある場合は、更新日時が新しいものを優先する。
        existing_dt = parse_datetime_jst(existing.get("last_db_update_jst"))
        row_dt = parse_datetime_jst(row_db.get("last_db_update_jst"))
        if existing_dt is None or (row_dt is not None and row_dt > existing_dt):
            db_by_ticker[ticker] = row_db

    return db_by_ticker


def extract_input_records(input_rows: List[List[Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for row_index, row in enumerate(input_rows, start=2):
        code = str(row[0]).strip() if len(row) >= 1 else ""
        if not code:
            records.append({
                "row_index": row_index,
                "code": "",
                "ticker": "",
                "is_blank": True,
                "is_duplicate": False,
            })
            continue

        # 同一銘柄を複数行で保有するケースは正常。
        # API取得は後段でticker単位に重複排除し、出力は各保有行に同じDBを使い回す。
        records.append({
            "row_index": row_index,
            "code": code,
            "ticker": normalize_code(code),
            "is_blank": False,
            "is_duplicate": False,
        })

    return records


def latest_close_from_download(data: pd.DataFrame, ticker: str) -> Optional[float]:
    if data is None or data.empty:
        return None

    try:
        if isinstance(data.columns, pd.MultiIndex):
            close_data = None
            level0 = [str(v) for v in data.columns.get_level_values(0)]
            level1 = [str(v) for v in data.columns.get_level_values(1)]

            if "Close" in level0:
                close_data = data["Close"]
            elif "Close" in level1:
                close_data = data.xs("Close", axis=1, level=1)

            if close_data is None:
                return None

            if isinstance(close_data, pd.Series):
                series = close_data.dropna()
            elif ticker in close_data.columns:
                series = close_data[ticker].dropna()
            else:
                return None
        else:
            if "Close" not in data.columns:
                return None
            series = data["Close"].dropna()

        if series.empty:
            return None
        return safe_float(series.iloc[-1])
    except Exception:
        return None


def chunk_list(values: List[str], size: int) -> List[List[str]]:
    if size <= 0:
        size = 100
    return [values[i:i + size] for i in range(0, len(values), size)]


def unique_tickers(values: List[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for value in values:
        ticker = str(value or "").strip()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        result.append(ticker)
    return result


def fetch_latest_price_individual(ticker: str) -> Optional[float]:
    try:
        hist = yf.Ticker(ticker).history(period="5d", auto_adjust=False)
        if hist.empty or "Close" not in hist.columns:
            return None
        close = hist["Close"].dropna()
        if close.empty:
            return None
        return safe_float(close.iloc[-1])
    except Exception as exc:
        logger.error("ERROR: 株価個別取得失敗 %s: %s", ticker, exc)
        return None


def fetch_latest_prices_for_chunk(tickers: List[str]) -> Dict[str, float]:
    if not tickers:
        return {}

    try:
        data = yf.download(
            tickers if len(tickers) > 1 else tickers[0],
            period="5d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        logger.error("ERROR: 株価chunk取得失敗 size=%s: %s", len(tickers), exc)
        return {}

    prices: Dict[str, float] = {}
    for ticker in tickers:
        price = latest_close_from_download(data, ticker)
        if price is not None:
            prices[ticker] = price
    return prices


def fetch_latest_prices(
    tickers: List[str],
    chunk_size: int = 100,
    retry_chunk_size: int = 25,
    individual_fallback_limit: int = 50,
) -> Dict[str, float]:
    unique_tickers = list(dict.fromkeys([ticker for ticker in tickers if ticker]))
    diag(
        "PRICE_FETCH_START total_unique=%s chunk_size=%s retry_chunk_size=%s individual_fallback_limit=%s sample=%s",
        len(unique_tickers),
        chunk_size,
        retry_chunk_size,
        individual_fallback_limit,
        sample_values(unique_tickers, 15),
    )
    if not unique_tickers:
        diag("PRICE_FETCH_END total_unique=0 fetched=0 missing=0")
        return {}

    prices: Dict[str, float] = {}
    missing_after_retry: List[str] = []

    chunks = chunk_list(unique_tickers, chunk_size)
    for chunk_no, chunk in enumerate(chunks, start=1):
        before = len(prices)
        chunk_prices = fetch_latest_prices_for_chunk(chunk)
        prices.update(chunk_prices)

        missing = [ticker for ticker in chunk if ticker not in chunk_prices]
        diag(
            "PRICE_CHUNK chunk_no=%s/%s size=%s fetched=%s missing=%s missing_sample=%s",
            chunk_no,
            len(chunks),
            len(chunk),
            len(prices) - before,
            len(missing),
            sample_values(missing, 10),
        )
        if not missing:
            continue

        retry_chunks = chunk_list(missing, retry_chunk_size)
        for retry_no, retry_chunk in enumerate(retry_chunks, start=1):
            retry_before = len(prices)
            retry_prices = fetch_latest_prices_for_chunk(retry_chunk)
            prices.update(retry_prices)
            retry_missing = [ticker for ticker in retry_chunk if ticker not in retry_prices]
            missing_after_retry.extend(retry_missing)
            diag(
                "PRICE_RETRY parent_chunk=%s retry_no=%s/%s size=%s fetched=%s missing=%s missing_sample=%s",
                chunk_no,
                retry_no,
                len(retry_chunks),
                len(retry_chunk),
                len(prices) - retry_before,
                len(retry_missing),
                sample_values(retry_missing, 10),
            )

    fallback_targets = list(dict.fromkeys(missing_after_retry))[:max(0, individual_fallback_limit)]
    diag(
        "PRICE_FALLBACK_START targets=%s skipped_by_limit=%s sample=%s",
        len(fallback_targets),
        max(0, len(set(missing_after_retry)) - len(fallback_targets)),
        sample_values(fallback_targets, 20),
    )
    fallback_ok = 0
    for ticker in fallback_targets:
        if ticker in prices:
            continue
        price = fetch_latest_price_individual(ticker)
        if price is not None:
            prices[ticker] = price
            fallback_ok += 1

    missing_final = [ticker for ticker in unique_tickers if ticker not in prices]
    diag(
        "PRICE_FETCH_END total_unique=%s fetched=%s fallback_ok=%s missing=%s missing_sample=%s",
        len(unique_tickers),
        len(prices),
        fallback_ok,
        len(missing_final),
        sample_values(missing_final, 30),
    )
    return prices


def full_refresh_priority(
    ticker: str,
    existing_db: Dict[str, Any],
    force_db_refresh: bool,
    refresh_days: int,
) -> Optional[Tuple[int, datetime]]:
    has_existing_ticker = bool(str(existing_db.get("ticker_yf") or "").strip())
    last_updated = parse_datetime_jst(existing_db.get("last_db_update_jst"))
    data_status = normalize_data_status(existing_db.get("data_status"))

    if not has_existing_ticker:
        return 0, datetime.min.replace(tzinfo=JST)
    if last_updated is None:
        return 1, datetime.min.replace(tzinfo=JST)
    if data_status == "ERROR":
        return 2, last_updated
    if should_refresh_db(existing_db, force=False, refresh_days=refresh_days):
        return 3, last_updated
    if force_db_refresh:
        return 4, last_updated
    return None


def has_required_db_cache(existing_db: Dict[str, Any]) -> bool:
    return bool(str(existing_db.get("ticker_yf") or "").strip()) and parse_datetime_jst(existing_db.get("last_db_update_jst")) is not None


def select_full_refresh_tickers(
    records: List[Dict[str, Any]],
    db_by_ticker: Dict[str, Dict[str, Any]],
    force_db_refresh: bool,
    refresh_days: int,
    max_full_refresh_per_run: int,
    max_initial_full_refresh_per_run: int,
) -> Tuple[Set[str], Set[str]]:
    # 初回・DBキャッシュなしは必ず今回取得する。
    # ここを上限制御すると、保有一覧の途中から評価が空になる。
    initial_candidates: List[Tuple[int, str]] = []
    regular_candidates: List[Tuple[int, datetime, int, str]] = []
    reason_by_ticker: Dict[str, str] = {}
    seen: Set[str] = set()

    for order, record in enumerate(records):
        if record.get("is_blank"):
            continue
        ticker = str(record.get("ticker") or "")
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)

        existing_db = db_by_ticker.get(ticker, {})
        ticker_yf = str(existing_db.get("ticker_yf") or "").strip()
        last_updated = parse_datetime_jst(existing_db.get("last_db_update_jst"))
        data_status = normalize_data_status(existing_db.get("data_status"))

        if not ticker_yf:
            initial_candidates.append((order, ticker))
            reason_by_ticker[ticker] = "initial_no_ticker_yf"
            continue
        if last_updated is None:
            initial_candidates.append((order, ticker))
            reason_by_ticker[ticker] = "initial_no_last_db_update_jst"
            continue
        if data_status == "PENDING":
            initial_candidates.append((order, ticker))
            reason_by_ticker[ticker] = "initial_pending"
            continue

        if data_status == "ERROR":
            regular_candidates.append((0, last_updated, order, ticker))
            reason_by_ticker[ticker] = "regular_error_retry"
            continue

        if should_refresh_db(existing_db, force=False, refresh_days=refresh_days):
            regular_candidates.append((1, last_updated, order, ticker))
            reason_by_ticker[ticker] = "regular_stale_ttl"
            continue

        if force_db_refresh:
            regular_candidates.append((2, last_updated, order, ticker))
            reason_by_ticker[ticker] = "regular_force"

    initial_candidates.sort(key=lambda item: item[0])
    regular_candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    selected: Set[str] = {ticker for _, ticker in initial_candidates}
    regular_selected = regular_candidates[:max(0, max_full_refresh_per_run)] if max_full_refresh_per_run > 0 else []
    if regular_selected:
        selected.update(ticker for _, _, _, ticker in regular_selected)
    elif force_db_refresh and max_full_refresh_per_run <= 0:
        selected.update(ticker for _, _, _, ticker in regular_candidates)

    all_regular = {ticker for _, _, _, ticker in regular_candidates}
    deferred = all_regular - selected

    selected_reasons = Counter(reason_by_ticker.get(ticker, "unknown") for ticker in selected)
    deferred_reasons = Counter(reason_by_ticker.get(ticker, "unknown") for ticker in deferred)
    diag(
        "FULL_REFRESH_SELECT unique_records=%s db_cache=%s initial_candidates=%s regular_candidates=%s selected=%s deferred=%s force=%s refresh_days=%s max_regular=%s max_initial_config=%s",
        len(seen),
        len(db_by_ticker),
        len(initial_candidates),
        len(regular_candidates),
        len(selected),
        len(deferred),
        force_db_refresh,
        refresh_days,
        max_full_refresh_per_run,
        max_initial_full_refresh_per_run,
    )
    diag("FULL_REFRESH_SELECTED_REASONS %s", dict(selected_reasons))
    diag("FULL_REFRESH_DEFERRED_REASONS %s", dict(deferred_reasons))
    diag("FULL_REFRESH_SELECTED_SAMPLE %s", sample_values(sorted(selected), 50))
    diag("FULL_REFRESH_DEFERRED_SAMPLE %s", sample_values(sorted(deferred), 50))

    return selected, deferred

def duplicate_outputs() -> Dict[str, Any]:
    # 同一銘柄を複数行で保有するケースは正常。通常はこの関数を使わない。
    outputs = {key: "" for key in EVAL_HEADERS}
    outputs["総合判定"] = "算出不能"
    outputs["モデル信頼度"] = "低"
    return outputs

def pending_outputs() -> Dict[str, Any]:
    # 通常運用では使わない。未取得DBは必ずフル取得対象に入れる。
    outputs = {key: "" for key in EVAL_HEADERS}
    outputs["総合判定"] = "算出不能"
    outputs["モデル信頼度"] = "低"
    return outputs

def generated_tail_ranges(start_row: int, end_row: int, db_end_col_letter: str) -> List[str]:
    if end_row < start_row:
        return []
    return [
        f"E{start_row}:R{end_row}",
        f"AA{start_row}:{db_end_col_letter}{end_row}",
    ]


def ensure_headers(ws: gspread.Worksheet) -> Tuple[Dict[str, int], Dict[str, int]]:
    ws.update(
        values=[EVAL_HEADERS],
        range_name="E1:R1",
        value_input_option="USER_ENTERED",
    )

    db_start_col = 27  # AA
    db_end_col = db_start_col + len(DB_HEADERS) - 1
    db_range = f"AA1:{column_letter(db_end_col)}1"
    ws.update(
        values=[DB_HEADERS],
        range_name=db_range,
        value_input_option="USER_ENTERED",
    )

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
    started = time.monotonic()
    diag("RUN_START version=%s event=%s python=%s cwd=%s", SCRIPT_VERSION, os.environ.get("GITHUB_EVENT_NAME", ""), os.sys.version.split()[0], os.getcwd())

    config = load_config()
    safe_config_keys = sorted([key for key in config.keys() if key != "gcp_service_account"])
    diag("CONFIG_LOADED keys=%s max_initial=%s max_regular=%s price_chunk=%s retry_chunk=%s fallback_limit=%s force=%s db_refresh_days=%s",
         safe_config_keys,
         config.get("max_initial_full_refresh_per_run", "default=1000"),
         config.get("max_full_refresh_per_run", "default=30"),
         config.get("price_chunk_size", "default=100"),
         config.get("price_retry_chunk_size", "default=25"),
         config.get("price_individual_fallback_limit", "default=50"),
         config.get("force_db_refresh", "default=false"),
         config.get("db_refresh_days", "default=7"))

    gc = get_client(config)
    spreadsheet = gc.open_by_url(config["spreadsheet_url"])
    ws = spreadsheet.worksheet(config["sheet_name"])
    diag("SHEET_OPEN spreadsheet_title=%s worksheet_title=%s rows=%s cols=%s", getattr(spreadsheet, "title", ""), ws.title, ws.row_count, ws.col_count)

    ensure_headers(ws)
    header_row = ws.row_values(1)
    header_map = {name: idx + 1 for idx, name in enumerate(header_row) if name}
    missing_eval_headers = [h for h in EVAL_HEADERS if h not in header_map]
    missing_db_headers = [h for h in DB_HEADERS if h not in header_map]
    diag("HEADERS header_cells=%s missing_eval=%s missing_db=%s", len(header_row), missing_eval_headers, missing_db_headers)

    db_last_col_letter = column_letter(27 + len(DB_HEADERS) - 1)

    input_rows = ws.get("A2:D")
    existing_db_key_rows = ws.get("AA2:AA")

    current_last_row = len(input_rows) + 1 if input_rows else 1
    existing_last_row = len(existing_db_key_rows) + 1 if existing_db_key_rows else 1
    clear_until_row = max(current_last_row, existing_last_row)
    diag("SHEET_READ input_rows=%s current_last_row=%s existing_db_key_rows=%s existing_last_row=%s clear_until_row=%s db_last_col=%s",
         len(input_rows), current_last_row, len(existing_db_key_rows), existing_last_row, clear_until_row, db_last_col_letter)

    existing_full_rows: List[List[str]] = []
    if clear_until_row >= 2:
        existing_full_rows = ws.get(f"A2:{db_last_col_letter}{clear_until_row}")
    diag("SHEET_READ_FULL rows=%s cols_max=%s", len(existing_full_rows), max([len(r) for r in existing_full_rows], default=0))

    if not input_rows:
        tail_ranges = generated_tail_ranges(2, clear_until_row, db_last_col_letter)
        diag_warn("NO_INPUT_ROWS clear_ranges=%s", tail_ranges)
        if tail_ranges:
            ws.batch_clear(tail_ranges)
        diag("RUN_END version=%s status=no_input elapsed_sec=%.2f", SCRIPT_VERSION, time.monotonic() - started)
        return

    records = extract_input_records(input_rows)
    blank_rows = sum(1 for r in records if r.get("is_blank"))
    nonblank_records = [r for r in records if not r.get("is_blank") and r.get("ticker")]
    active_tickers = unique_tickers([str(record["ticker"]) for record in nonblank_records])
    ticker_row_counts = Counter(str(record["ticker"]) for record in nonblank_records)
    duplicate_tickers = sorted([ticker for ticker, count in ticker_row_counts.items() if count > 1])
    diag("INPUT_PARSE total_records=%s blank_rows=%s nonblank_rows=%s unique_tickers=%s duplicate_tickers=%s duplicate_sample=%s first_rows=%s",
         len(records), blank_rows, len(nonblank_records), len(active_tickers), len(duplicate_tickers), sample_values(duplicate_tickers, 30), sample_values([r.get("ticker") for r in nonblank_records], 30))

    db_by_ticker = build_db_by_ticker(header_row, existing_full_rows)
    db_status_counts = Counter(normalize_data_status(db.get("data_status")) for db in db_by_ticker.values())
    db_missing_last_update = [ticker for ticker, db in db_by_ticker.items() if parse_datetime_jst(db.get("last_db_update_jst")) is None]
    diag("DB_CACHE_BUILD unique_db_tickers=%s status_counts=%s missing_last_update=%s missing_last_update_sample=%s db_sample=%s",
         len(db_by_ticker), dict(db_status_counts), len(db_missing_last_update), sample_values(sorted(db_missing_last_update), 30), sample_values(sorted(db_by_ticker.keys()), 30))

    price_chunk_size = get_config_int(config, "price_chunk_size", 100)
    price_retry_chunk_size = get_config_int(config, "price_retry_chunk_size", 25)
    price_individual_fallback_limit = get_config_int(config, "price_individual_fallback_limit", 50)
    latest_prices = fetch_latest_prices(
        active_tickers,
        chunk_size=price_chunk_size,
        retry_chunk_size=price_retry_chunk_size,
        individual_fallback_limit=price_individual_fallback_limit,
    )
    price_missing = [ticker for ticker in active_tickers if ticker not in latest_prices]
    diag("PRICE_RESULT active=%s fetched=%s missing=%s missing_sample=%s", len(active_tickers), len(latest_prices), len(price_missing), sample_values(price_missing, 50))

    force_db_refresh = get_config_bool(config, "force_db_refresh", False)
    refresh_days = get_config_int(config, "db_refresh_days", 7)
    max_full_refresh_per_run = get_config_int(config, "max_full_refresh_per_run", 30)
    max_initial_full_refresh_per_run = get_config_int(config, "max_initial_full_refresh_per_run", 1000)

    full_refresh_tickers, deferred_refresh_tickers = select_full_refresh_tickers(
        records=records,
        db_by_ticker=db_by_ticker,
        force_db_refresh=force_db_refresh,
        refresh_days=refresh_days,
        max_full_refresh_per_run=max_full_refresh_per_run,
        max_initial_full_refresh_per_run=max_initial_full_refresh_per_run,
    )

    # 最終安全策：DBキャッシュなし・last_db_update_jst空の銘柄は必ず今回のフル取得対象に入れる。
    forced_by_safety: List[str] = []
    for ticker in active_tickers:
        existing_db = db_by_ticker.get(ticker, {})
        if not str(existing_db.get("ticker_yf") or "").strip() or parse_datetime_jst(existing_db.get("last_db_update_jst")) is None:
            if ticker not in full_refresh_tickers:
                forced_by_safety.append(ticker)
            full_refresh_tickers.add(ticker)
            deferred_refresh_tickers.discard(ticker)
    diag("FULL_REFRESH_AFTER_SAFETY selected=%s deferred=%s forced_by_safety=%s forced_sample=%s selected_sample=%s",
         len(full_refresh_tickers), len(deferred_refresh_tickers), len(forced_by_safety), sample_values(forced_by_safety, 50), sample_values(sorted(full_refresh_tickers), 50))

    rf_rate: Optional[float] = None
    rf_rate_ok = False
    if full_refresh_tickers:
        try:
            rf_rate = fetch_rf_rate_japan_from_mof()
            rf_rate_ok = True
            diag("RF_RATE_OK value=%s source=%s full_refresh_count=%s", rf_rate, _RF_RATE_SOURCE, len(full_refresh_tickers))
        except Exception as exc:
            logger.error("ERROR: %s", exc)
            logger.error(error_guidance_message(exc))
            diag_warn("RF_RATE_FAIL full_refresh_skipped=%s reason=%s", len(full_refresh_tickers), exc)
            deferred_refresh_tickers.update(full_refresh_tickers)
            full_refresh_tickers = set()
    else:
        diag("RF_RATE_SKIP reason=no_full_refresh")

    output_matrix: List[List[Any]] = []
    db_matrix: List[List[Any]] = []
    row_status_counts: Counter[str] = Counter()
    output_judgement_counts: Counter[str] = Counter()
    full_refresh_success: List[str] = []
    full_refresh_fail: List[str] = []
    existing_cache_no_refresh: List[str] = []
    no_cache_not_refreshed: List[str] = []
    banned_outputs: List[Tuple[int, str, str]] = []

    for record in records:
        row_index = int(record.get("row_index") or 0)
        if record.get("is_blank"):
            output_matrix.append([""] * len(EVAL_HEADERS))
            db_matrix.append([""] * len(DB_HEADERS))
            row_status_counts["blank"] += 1
            continue

        ticker = str(record.get("ticker") or "")
        existing_db = db_by_ticker.get(ticker, {})

        refresh_full = ticker in full_refresh_tickers
        deferred_refresh = ticker in deferred_refresh_tickers
        db_base = {key: existing_db.get(key, "") for key in DB_HEADERS}
        db_base["financial_flag_override"] = existing_db.get("financial_flag_override", "")

        try:
            if refresh_full:
                diag("ROW_FULL_REFRESH_START row=%s ticker=%s has_existing_ticker=%s last_update=%s data_status=%s price_available=%s",
                     row_index, ticker, bool(existing_db.get("ticker_yf")), existing_db.get("last_db_update_jst"), existing_db.get("data_status"), ticker in latest_prices)
                fresh = fetch_ticker_data(ticker, refresh_full=True, config=config, rf_rate=rf_rate)
                full_refresh_success.append(ticker)
                row_status_counts["full_refresh"] += 1
                diag("ROW_FULL_REFRESH_OK row=%s ticker=%s missing_fields=%s data_status=%s current_price=%s last_update=%s",
                     row_index, ticker, fresh.get("missing_fields"), fresh.get("data_status"), fresh.get("current_price"), fresh.get("last_db_update_jst"))
            else:
                fresh = {
                    "ticker_yf": ticker,
                }
                latest_price = latest_prices.get(ticker)
                if latest_price is not None:
                    fresh["current_price"] = latest_price

                if not existing_db.get("ticker_yf"):
                    fresh["data_status"] = "ERROR"
                    fresh["calc_error"] = "DBキャッシュなし。フル取得対象に漏れています"
                    no_cache_not_refreshed.append(ticker)
                    row_status_counts["no_cache_not_refreshed"] += 1
                    diag_warn("ROW_NO_CACHE_NOT_REFRESHED row=%s ticker=%s deferred=%s rf_rate_ok=%s", row_index, ticker, deferred_refresh, rf_rate_ok)
                else:
                    fresh["data_status"] = existing_db.get("data_status") or "OK"
                    existing_cache_no_refresh.append(ticker)
                    row_status_counts["existing_cache_no_refresh"] += 1

            db = merge_db(db_base, fresh, refresh_full=refresh_full)

            if db.get("ticker_yf") in (None, ""):
                db["ticker_yf"] = ticker
            if db.get("financial_flag_override") in (None, ""):
                db["financial_flag_override"] = existing_db.get("financial_flag_override", "")

            outputs = compute_outputs(db)
        except Exception as exc:
            log_error_with_guidance(exc)
            full_refresh_fail.append(ticker)
            row_status_counts["exception"] += 1
            diag_warn("ROW_EXCEPTION row=%s ticker=%s refresh_full=%s deferred=%s exc_type=%s exc=%s", row_index, ticker, refresh_full, deferred_refresh, type(exc).__name__, exc)

            db = {key: existing_db.get(key, "") for key in DB_HEADERS}
            db["ticker_yf"] = ticker
            db["financial_flag_override"] = existing_db.get("financial_flag_override", "")
            db["data_status"] = "ERROR"
            db["calc_error"] = str(exc)
            outputs = {key: "" for key in EVAL_HEADERS}
            outputs["総合判定"] = "算出不能"

        judgement = str(outputs.get("総合判定") or "")
        output_judgement_counts[judgement] += 1
        if judgement in {"DB更新待ち", "重複銘柄", "DB未取得"}:
            banned_outputs.append((row_index, ticker, judgement))
            diag_warn("BANNED_OUTPUT_DETECTED row=%s ticker=%s judgement=%s version=%s", row_index, ticker, judgement, SCRIPT_VERSION)

        output_matrix.append([serialize_cell(outputs.get(h)) for h in EVAL_HEADERS])
        db_matrix.append([serialize_cell(db.get(h)) for h in DB_HEADERS])

    diag("PROCESS_SUMMARY row_status_counts=%s", dict(row_status_counts))
    diag("PROCESS_REFRESH full_success_unique=%s full_success_rows=%s full_fail=%s no_cache_not_refreshed=%s existing_cache_no_refresh_rows=%s",
         len(set(full_refresh_success)), len(full_refresh_success), len(full_refresh_fail), len(no_cache_not_refreshed), len(existing_cache_no_refresh))
    diag("PROCESS_REFRESH_SUCCESS_SAMPLE %s", sample_values(full_refresh_success, 50))
    diag("PROCESS_REFRESH_FAIL_SAMPLE %s", sample_values(full_refresh_fail, 50))
    diag("PROCESS_NO_CACHE_NOT_REFRESHED_SAMPLE %s", sample_values(no_cache_not_refreshed, 50))
    diag("OUTPUT_JUDGEMENT_COUNTS %s", dict(output_judgement_counts))
    if banned_outputs:
        diag_warn("BANNED_OUTPUT_SUMMARY count=%s sample=%s", len(banned_outputs), sample_values(banned_outputs, 50))
    else:
        diag("BANNED_OUTPUT_SUMMARY count=0")

    last_output_row = len(records) + 1
    eval_end_col = column_letter(5 + len(EVAL_HEADERS) - 1)  # E
    db_end_col = column_letter(27 + len(DB_HEADERS) - 1)  # AA

    diag("SHEET_WRITE_EVAL range=%s rows=%s cols=%s", f"E2:{eval_end_col}{last_output_row}", len(output_matrix), len(EVAL_HEADERS))
    ws.update(
        values=output_matrix,
        range_name=f"E2:{eval_end_col}{last_output_row}",
        value_input_option="USER_ENTERED",
    )
    diag("SHEET_WRITE_DB range=%s rows=%s cols=%s", f"AA2:{db_end_col}{last_output_row}", len(db_matrix), len(DB_HEADERS))
    ws.update(
        values=db_matrix,
        range_name=f"AA2:{db_end_col}{last_output_row}",
        value_input_option="USER_ENTERED",
    )

    tail_start_row = last_output_row + 1
    tail_ranges = generated_tail_ranges(tail_start_row, clear_until_row, db_last_col_letter)
    if tail_ranges:
        diag("SHEET_CLEAR_TAIL ranges=%s", tail_ranges)
        ws.batch_clear(tail_ranges)
    else:
        diag("SHEET_CLEAR_TAIL ranges=[]")

    diag("RUN_END version=%s status=ok elapsed_sec=%.2f records=%s unique_tickers=%s full_refresh=%s banned_outputs=%s", SCRIPT_VERSION, time.monotonic() - started, len(records), len(active_tickers), len(set(full_refresh_success)), len(banned_outputs))


if __name__ == "__main__":
    main()
