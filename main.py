"""
日本株ポートフォリオ 適正株価シート 更新スクリプト
仕様書 v0.2.1 準拠

Phase 1-a: 入力読込 → yfinance 取得 → DB 列書き込み → エラー記録
Phase 1-b: 金融判定 / E / F / G / I 列
Phase 1-c: J / K / L / U 列（統合・判定・乖離率）
Phase 1-d: M〜R / T 列（金融専用）
Phase 2 : H / S 列
"""

import json
import math
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import gspread
import numpy as np
import pandas as pd
import yfinance as yf
from google.oauth2.service_account import Credentials
from scipy import optimize

# ─── 定数 ───────────────────────────────────────────────
JST = timezone(timedelta(hours=9))
SECRETS_PATH = "secrets.json"

# yfinance レート制限回避のウェイト（秒）
YFINANCE_WAIT = 1.5

# 資本コスト前提値（DB 列への初期書き込み用）
DEFAULT_RF_RATE = 0.015          # 10年国債利回り近辺
DEFAULT_ERP = 0.055              # Damodaran 日本推計
DEFAULT_COUNTRY_RISK_PREMIUM = 0.0
DEFAULT_COD_ESTIMATE = 0.018     # 固定近似
DEFAULT_TAX_RATE = 0.30          # 日本実効税率
DEFAULT_TERMINAL_GROWTH = 0.01   # 永続成長率
DEFAULT_GROWTH_FLOOR = 0.0
DEFAULT_GROWTH_CAP = 0.15
DEFAULT_GAP_YEAR_DEFAULT = 5

# サイズプレミアム（時価総額閾値）
SIZE_PREMIUM_LARGE_CAP = 500_000_000_000  # 5000億円
SIZE_PREMIUM_LARGE = 0.0
SIZE_PREMIUM_SMALL = 0.015

# 金融判定キーワード（yfinance sector/industry 用）
FINANCIAL_KEYWORDS = {
    "bank", "insurance", "capital markets", "financial services",
    "securities", "broker", "asset management", "diversified financials",
    "consumer finance", "mortgage", "thrift", "credit services",
    "銀行", "保険", "証券", "金融", "信金", "信組", "信託",
}

# 列インデックス（0始まり）: A=0, B=1, ... Z=25, AA=26, AB=27, ...
COL = {
    # ユーザー入力
    "ticker_code": 0,   # A
    "company_name": 1,  # B
    "shares_held": 2,   # C
    "cost_price": 3,    # D
    # 評価列
    "E": 4,   # Economic Profit 株価
    "F": 5,   # Value Driver Formula 株価
    "G": 6,   # Reverse DCF 市場織込成長率
    "H": 7,   # Reverse DCF GAP 年数
    "I": 8,   # ROIC-WACC スプレッド
    "J": 9,   # 統合適正株価
    "K": 10,  # 現在株価との差異率
    "L": 11,  # 総合判定
    # 金融専用
    "M": 12,  # 金融業種フラグ
    "N": 13,  # 金融_CoE
    "O": 14,  # 金融_平準化ROE
    "P": 15,  # 金融_ROE-CoEスプレッド
    "Q": 16,  # 金融_正当PBR
    "R": 17,  # 金融_適正株価
    "S": 18,  # 金融_市場織込ROE
    "T": 19,  # 金融_判定
    # E/F乖離率
    "U": 20,
    # DB 列 AA(26) 以降
    "AA": 26,  # ticker_yf
}

# DB列名リスト（AA以降の順番）
DB_COLS = [
    "ticker_yf", "market", "currency", "quote_type",
    "sector_raw", "industry_raw", "financial_flag", "financial_flag_override",
    "data_status", "last_db_update_jst",
    # 株価・株式数
    "current_price", "market_cap", "shares_outstanding", "enterprise_value",
    "beta", "dividend_yield", "trailing_annual_dividend_rate",
    # 損益
    "revenue_ttm", "ebit_ttm", "ebitda_ttm", "net_income_ttm", "nopat_ttm",
    # 貸借対照表
    "total_assets", "total_equity", "cash_and_equivalents",
    "total_debt", "net_debt", "invested_capital",
    # CF
    "operating_cf_ttm", "capex_ttm", "fcf_ttm", "fcfe_ttm",
    # 1株指標
    "eps_ttm", "bps", "dps_ttm", "payout_ratio", "pb_now", "pe_now",
    # 収益性
    "roe_1y", "roe_3y_avg", "roic_1y", "roic_3y_avg",
    "roe_normalized", "roic_normalized",
    # 資本コスト
    "rf_rate", "erp", "country_risk_premium", "size_premium",
    "wacc", "coe", "cod_estimate", "tax_rate_estimate",
    # 成長前提
    "growth_base", "growth_floor", "growth_cap", "terminal_growth", "gap_year_default",
    # Reverse DCF
    "implied_growth_rate", "implied_gap_years",
    # 金融専用
    "financial_roe_avg", "financial_payout_avg",
    "financial_justified_pbr", "financial_implied_roe",
    # エラー追跡
    "missing_fields", "calc_error", "notes",
]

TOTAL_COLS = COL["AA"] + len(DB_COLS)  # シート全列数


# ─── ユーティリティ ─────────────────────────────────────

def safe(val, default=None):
    """NaN / None / inf を default に変換"""
    if val is None:
        return default
    try:
        if math.isnan(val) or math.isinf(val):
            return default
    except TypeError:
        pass
    return val


def pct_fmt(val) -> str:
    """小数→パーセント文字列（例: 0.123 → '12.3%'）"""
    if val is None:
        return ""
    return f"{val * 100:.1f}%"


def _get_fiscal_annual(df: pd.DataFrame, row_key: str, n: int = 3):
    """
    財務三表 DataFrame から年度値を最大 n 期取得する。
    TTM 列（最新の 12ヶ月ローリング）を除外し、年度決算列のみを使う。
    """
    if df is None or df.empty:
        return []
    try:
        if row_key not in df.index:
            return []
        row = df.loc[row_key].dropna()
        # 列が Timestamp の場合、月末以外（＝中間決算）を除外
        annual_cols = []
        for col in row.index:
            try:
                dt = pd.Timestamp(col)
                # 3/31, 6/30, 9/30, 12/31 を年度末とみなす（月末判定）
                import calendar
                last_day = calendar.monthrange(dt.year, dt.month)[1]
                if dt.day == last_day:
                    annual_cols.append(col)
            except Exception:
                annual_cols.append(col)
        row = row[annual_cols].sort_index(ascending=False)
        return row.iloc[:n].tolist()
    except Exception:
        return []


def _avg(values: list) -> Optional[float]:
    """有効値の平均（空リストは None）"""
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


# ─── Google Sheets 接続 ─────────────────────────────────

def connect_sheet(secrets: dict):
    sa_info = secrets["gcp_service_account"]
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_url(secrets["spreadsheet_url"])
    ws = sh.worksheet(secrets["sheet_name"])
    return ws


# ─── 銘柄コード処理 ─────────────────────────────────────

def to_yf_ticker(code: str) -> str:
    """4桁数字 or 英字混じりコード → XXXX.T"""
    code = str(code).strip()
    if not code.endswith(".T"):
        code = code + ".T"
    return code


# ─── yfinance データ取得 ─────────────────────────────────

def fetch_yf_data(ticker_code: str) -> dict:
    """
    yfinance から必要なデータを取得してフラットな dict で返す。
    欠損は None。エラーは例外ではなく空値で返す。
    """
    result = {k: None for k in DB_COLS}
    missing = []
    errors = []
    notes_list = []

    ticker_yf = to_yf_ticker(ticker_code)
    result["ticker_yf"] = ticker_yf

    try:
        tk = yf.Ticker(ticker_yf)
        info = {}
        try:
            info = tk.info or {}
        except Exception as e:
            errors.append(f"info: {e}")

        # ── 基本識別 ──
        result["market"] = safe(info.get("exchange"))
        result["currency"] = safe(info.get("currency"))
        result["quote_type"] = safe(info.get("quoteType"))
        result["sector_raw"] = safe(info.get("sector"))
        result["industry_raw"] = safe(info.get("industry"))

        # ── 株価・株式数 ──
        current_price = safe(info.get("currentPrice") or info.get("regularMarketPrice"))
        result["current_price"] = current_price

        market_cap = safe(info.get("marketCap"))
        result["market_cap"] = market_cap

        shares = safe(info.get("sharesOutstanding") or info.get("impliedSharesOutstanding"))
        result["shares_outstanding"] = shares

        ev = safe(info.get("enterpriseValue"))
        result["enterprise_value"] = ev

        beta = None
        try:
            # 2年週次βを優先
            hist = tk.history(period="2y", interval="1wk")
            if hist is not None and len(hist) > 30:
                returns = hist["Close"].pct_change().dropna()
                # 日経225代替としてTOPIX ETFを使うのは困難なため、
                # yfinance betaをフォールバックとして採用
                beta = safe(info.get("beta"))
            else:
                beta = safe(info.get("beta"))
        except Exception:
            beta = safe(info.get("beta"))
        result["beta"] = beta

        result["dividend_yield"] = safe(info.get("dividendYield"))
        result["trailing_annual_dividend_rate"] = safe(info.get("trailingAnnualDividendRate"))

        # ── 財務三表 ──
        income_stmt = None
        balance_sheet = None
        cashflow = None
        try:
            income_stmt = tk.financials          # 年度 income statement
        except Exception as e:
            errors.append(f"financials: {e}")
        try:
            balance_sheet = tk.balance_sheet
        except Exception as e:
            errors.append(f"balance_sheet: {e}")
        try:
            cashflow = tk.cashflow
        except Exception as e:
            errors.append(f"cashflow: {e}")

        # ── 損益 ──
        revenue_vals = _get_fiscal_annual(income_stmt, "Total Revenue")
        ebit_vals = _get_fiscal_annual(income_stmt, "EBIT")
        ebitda_vals = _get_fiscal_annual(income_stmt, "EBITDA")
        ni_vals = _get_fiscal_annual(income_stmt, "Net Income")

        result["revenue_ttm"] = safe(revenue_vals[0]) if revenue_vals else None
        result["ebit_ttm"] = safe(ebit_vals[0]) if ebit_vals else None
        result["ebitda_ttm"] = safe(ebitda_vals[0]) if ebitda_vals else None
        result["net_income_ttm"] = safe(ni_vals[0]) if ni_vals else None

        ebit1 = result["ebit_ttm"]
        nopat = ebit1 * (1 - DEFAULT_TAX_RATE) if ebit1 is not None else None
        result["nopat_ttm"] = nopat

        # ── 貸借対照表 ──
        total_equity_vals = _get_fiscal_annual(balance_sheet, "Stockholders Equity")
        total_debt_vals = _get_fiscal_annual(balance_sheet, "Total Debt")
        cash_vals = _get_fiscal_annual(balance_sheet, "Cash And Cash Equivalents")
        total_assets_vals = _get_fiscal_annual(balance_sheet, "Total Assets")

        total_equity = safe(total_equity_vals[0]) if total_equity_vals else None
        total_debt = safe(total_debt_vals[0]) if total_debt_vals else None
        cash = safe(cash_vals[0]) if cash_vals else None

        result["total_assets"] = safe(total_assets_vals[0]) if total_assets_vals else None
        result["total_equity"] = total_equity
        result["cash_and_equivalents"] = cash
        result["total_debt"] = total_debt

        net_debt = None
        invested_capital = None
        if total_debt is not None and cash is not None:
            net_debt = total_debt - cash
        if total_equity is not None and total_debt is not None and cash is not None:
            invested_capital = total_equity + total_debt - cash
            notes_list.append("Invested Capital = TotalEquity+TotalDebt-Cash（持合株除外）")

        result["net_debt"] = net_debt
        result["invested_capital"] = invested_capital

        # ── CF ──
        op_cf_vals = _get_fiscal_annual(cashflow, "Operating Cash Flow")
        capex_vals = _get_fiscal_annual(cashflow, "Capital Expenditure")

        op_cf = safe(op_cf_vals[0]) if op_cf_vals else None
        capex = safe(capex_vals[0]) if capex_vals else None
        result["operating_cf_ttm"] = op_cf
        result["capex_ttm"] = capex

        fcf = None
        if op_cf is not None and capex is not None:
            fcf = op_cf - abs(capex)
        result["fcf_ttm"] = fcf

        # FCFE ≈ FCF（簡略）
        result["fcfe_ttm"] = fcf

        # ── 1株指標 ──
        eps = safe(info.get("trailingEps"))
        result["eps_ttm"] = eps

        bps = safe(info.get("bookValue"))
        result["bps"] = bps

        dps = safe(info.get("trailingAnnualDividendRate"))
        result["dps_ttm"] = dps

        payout = safe(info.get("payoutRatio"))
        if payout is not None:
            if payout < 0:
                payout = None
            elif payout > 1.2:
                payout = 1.2
        result["payout_ratio"] = payout

        pb = safe(info.get("priceToBook"))
        result["pb_now"] = pb

        pe = safe(info.get("trailingPE"))
        result["pe_now"] = pe

        # ── 収益性（ROE / ROIC）──
        # 年度ROEの計算: NI / Equity
        ni_annual = _get_fiscal_annual(income_stmt, "Net Income", n=3)
        eq_annual = _get_fiscal_annual(balance_sheet, "Stockholders Equity", n=3)
        ebit_annual = _get_fiscal_annual(income_stmt, "EBIT", n=3)
        ic_annual = []

        roe_list = []
        roic_list = []
        for i in range(min(len(ni_annual), len(eq_annual))):
            ni_v = safe(ni_annual[i])
            eq_v = safe(eq_annual[i])
            if ni_v is not None and eq_v is not None and eq_v > 0:
                roe_list.append(ni_v / eq_v)

        # 投下資本の年度値
        td_annual = _get_fiscal_annual(balance_sheet, "Total Debt", n=3)
        ca_annual = _get_fiscal_annual(balance_sheet, "Cash And Cash Equivalents", n=3)
        for i in range(min(len(ebit_annual), len(eq_annual), len(td_annual), len(ca_annual))):
            eb = safe(ebit_annual[i])
            eq = safe(eq_annual[i])
            td = safe(td_annual[i])
            ca = safe(ca_annual[i])
            if all(v is not None for v in [eb, eq, td, ca]):
                nopat_v = eb * (1 - DEFAULT_TAX_RATE)
                ic_v = eq + td - ca
                if ic_v > 0:
                    roic_list.append(nopat_v / ic_v)

        result["roe_1y"] = roe_list[0] if len(roe_list) >= 1 else None
        result["roe_3y_avg"] = _avg(roe_list[:3]) if len(roe_list) >= 2 else None
        result["roic_1y"] = roic_list[0] if len(roic_list) >= 1 else None
        result["roic_3y_avg"] = _avg(roic_list[:3]) if len(roic_list) >= 2 else None

        # 平準化（3y_avg > 2y_avg > 1y > None）
        def normalized(vals):
            if len(vals) >= 3:
                return _avg(vals[:3])
            elif len(vals) == 2:
                return _avg(vals[:2])
            elif len(vals) == 1:
                return vals[0]
            return None

        result["roe_normalized"] = normalized(roe_list)
        result["roic_normalized"] = normalized(roic_list)

        # ── 資本コスト ──
        result["rf_rate"] = DEFAULT_RF_RATE
        result["erp"] = DEFAULT_ERP
        result["country_risk_premium"] = DEFAULT_COUNTRY_RISK_PREMIUM
        result["cod_estimate"] = DEFAULT_COD_ESTIMATE
        result["tax_rate_estimate"] = DEFAULT_TAX_RATE

        # サイズプレミアム
        size_p = SIZE_PREMIUM_LARGE
        if market_cap is not None and market_cap < SIZE_PREMIUM_LARGE_CAP:
            size_p = SIZE_PREMIUM_SMALL
        result["size_premium"] = size_p

        # CoE
        beta_use = beta if beta is not None else 1.0  # フォールバック: β=1.0
        coe = DEFAULT_RF_RATE + beta_use * DEFAULT_ERP + size_p + DEFAULT_COUNTRY_RISK_PREMIUM
        result["coe"] = coe

        # WACC
        wacc = None
        if total_equity is not None and total_debt is not None:
            total_cap = total_equity + total_debt
            if total_cap > 0:
                cod_after_tax = DEFAULT_COD_ESTIMATE * (1 - DEFAULT_TAX_RATE)
                wacc = coe * (total_equity / total_cap) + cod_after_tax * (total_debt / total_cap)
        result["wacc"] = wacc

        # ── 成長前提 ──
        # growth_base: ROICのフォールバックとして単純 ROIC×留保率
        roic_n = result["roic_normalized"]
        roe_n = result["roe_normalized"]
        payout = result["payout_ratio"]
        retention = (1 - payout) if payout is not None else 0.5
        growth_base = None
        if roic_n is not None:
            growth_base = roic_n * retention
            growth_base = max(DEFAULT_GROWTH_FLOOR, min(DEFAULT_GROWTH_CAP, growth_base))
        result["growth_base"] = growth_base
        result["growth_floor"] = DEFAULT_GROWTH_FLOOR
        result["growth_cap"] = DEFAULT_GROWTH_CAP
        result["terminal_growth"] = DEFAULT_TERMINAL_GROWTH
        result["gap_year_default"] = DEFAULT_GAP_YEAR_DEFAULT

        # ── 欠損チェック ──
        critical_fields = ["current_price", "shares_outstanding", "roic_normalized", "wacc"]
        for f in critical_fields:
            if result.get(f) is None:
                missing.append(f)

        # ── 金融専用 ──
        fin_roe_avg = result["roe_normalized"]
        result["financial_roe_avg"] = fin_roe_avg
        result["financial_payout_avg"] = result["payout_ratio"]

        # financial_flag_override は既存値を上書きしない（後で処理）
        result["financial_flag_override"] = None  # 新規行のみ空欄

    except Exception as e:
        errors.append(f"{type(e).__name__}: {e}")

    result["missing_fields"] = ", ".join(missing) if missing else ""
    result["calc_error"] = "; ".join(errors) if errors else ""
    result["notes"] = "; ".join(notes_list) if notes_list else ""
    result["last_db_update_jst"] = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    result["data_status"] = "ok" if not errors else "error"

    return result


# ─── 金融判定 ────────────────────────────────────────────

def is_financial(db: dict, override_val) -> bool:
    """金融企業フラグを返す。override が 0/1 ならそれを優先"""
    if override_val is not None and str(override_val).strip() in ("0", "1"):
        return str(override_val).strip() == "1"
    sector = (db.get("sector_raw") or "").lower()
    industry = (db.get("industry_raw") or "").lower()
    combined = sector + " " + industry
    return any(kw in combined for kw in FINANCIAL_KEYWORDS)


# ─── 評価計算 ────────────────────────────────────────────

def calc_ep_price(db: dict) -> Optional[float]:
    """
    E列: Economic Profit 株価
    EP = (ROIC - WACC) × IC
    企業価値 = IC + Σ[EP/(1+WACC)^t] + TV
    TV = EP_5 / WACC (永続)
    """
    roic = db.get("roic_normalized")
    wacc = db.get("wacc")
    ic = db.get("invested_capital")
    shares = db.get("shares_outstanding")
    net_debt = db.get("net_debt")
    g = db.get("growth_base") or DEFAULT_TERMINAL_GROWTH

    if any(v is None for v in [roic, wacc, ic, shares]):
        return None
    if wacc <= 0 or ic <= 0 or shares <= 0:
        return None

    ep = (roic - wacc) * ic
    # 5年明示期間
    pv_ep = sum(ep * ((1 + g) ** t) / ((1 + wacc) ** t) for t in range(1, 6))
    # 終価（第5年末 EP を WACC で永続）
    ep5 = ep * ((1 + g) ** 5)
    tv = ep5 / wacc if wacc > 0 else 0
    tv_pv = tv / ((1 + wacc) ** 5)

    firm_value = ic + pv_ep + tv_pv
    equity_value = firm_value - (net_debt or 0)
    if equity_value <= 0:
        return None
    return equity_value / shares


def calc_vdf_price(db: dict) -> Optional[float]:
    """
    F列: Value Driver Formula 株価
    EV = NOPAT × (1 - g/ROIC) / (WACC - g)
    """
    nopat = db.get("nopat_ttm")
    roic = db.get("roic_normalized")
    wacc = db.get("wacc")
    shares = db.get("shares_outstanding")
    net_debt = db.get("net_debt")
    g = db.get("growth_base") or DEFAULT_TERMINAL_GROWTH

    if any(v is None for v in [nopat, roic, wacc, shares]):
        return None
    if roic <= g or wacc <= g or shares <= 0:
        return None

    ev = nopat * (1 - g / roic) / (wacc - g)
    equity_value = ev - (net_debt or 0)

    # キャッシュリッチ過剰警告
    cash = db.get("cash_and_equivalents") or 0
    if cash > ev:
        notes = db.get("notes") or ""
        db["notes"] = notes + "; WARNING: Cash>EV キャッシュリッチ過剰"

    if equity_value <= 0:
        return None
    return equity_value / shares


def calc_implied_growth(db: dict) -> Optional[float]:
    """
    G列: Reverse DCF 市場織込成長率
    現在の時価総額を再現する g_implied を二分法で逆算
    """
    roic = db.get("roic_normalized")
    wacc = db.get("wacc")
    ic = db.get("invested_capital")
    shares = db.get("shares_outstanding")
    net_debt = db.get("net_debt")
    market_cap = db.get("market_cap")

    if any(v is None for v in [roic, wacc, ic, shares, market_cap]):
        return None
    if wacc <= 0 or ic <= 0 or shares <= 0 or market_cap <= 0:
        return None

    target_equity = market_cap  # 時価総額 ≈ 株主価値

    def equity_from_g(g):
        if wacc <= g or roic <= g:
            return None
        ep = (roic - wacc) * ic
        pv_ep = sum(ep * ((1 + g) ** t) / ((1 + wacc) ** t) for t in range(1, 6))
        ep5 = ep * ((1 + g) ** 5)
        tv = ep5 / wacc if wacc > 0 else 0
        tv_pv = tv / ((1 + wacc) ** 5)
        firm_val = ic + pv_ep + tv_pv
        return firm_val - (net_debt or 0)

    try:
        lo, hi = DEFAULT_GROWTH_FLOOR, DEFAULT_GROWTH_CAP - 0.001
        if equity_from_g(lo) is None or equity_from_g(hi) is None:
            return None
        f_lo = equity_from_g(lo) - target_equity
        f_hi = equity_from_g(hi) - target_equity
        if f_lo * f_hi > 0:
            return None  # 解なし
        result = optimize.brentq(
            lambda g: (equity_from_g(g) or 0) - target_equity,
            lo, hi, xtol=1e-6, maxiter=100
        )
        return result
    except Exception:
        return None


def calc_gap_years(db: dict) -> Optional[float]:
    """
    H列: Reverse DCF GAP 年数
    成長率を固定し、超過収益を何年維持すれば現在時価総額を説明できるか逆算
    """
    roic = db.get("roic_normalized")
    wacc = db.get("wacc")
    ic = db.get("invested_capital")
    shares = db.get("shares_outstanding")
    net_debt = db.get("net_debt")
    market_cap = db.get("market_cap")
    g = db.get("growth_base") or DEFAULT_TERMINAL_GROWTH

    if any(v is None for v in [roic, wacc, ic, shares, market_cap]):
        return None
    if wacc <= g or roic <= g or ic <= 0 or shares <= 0 or market_cap <= 0:
        return None

    target_equity = market_cap

    def equity_from_years(n):
        n = int(round(n))
        ep = (roic - wacc) * ic
        pv_ep = sum(ep * ((1 + g) ** t) / ((1 + wacc) ** t) for t in range(1, n + 1))
        ep_n = ep * ((1 + g) ** n)
        tv = ep_n / wacc if wacc > 0 else 0
        tv_pv = tv / ((1 + wacc) ** n)
        firm_val = ic + pv_ep + tv_pv
        return firm_val - (net_debt or 0)

    try:
        lo, hi = 1.0, 50.0
        f_lo = equity_from_years(lo) - target_equity
        f_hi = equity_from_years(hi) - target_equity
        if f_lo * f_hi > 0:
            return None
        result = optimize.brentq(
            lambda n: equity_from_years(n) - target_equity,
            lo, hi, xtol=0.1, maxiter=100
        )
        return result
    except Exception:
        return None


# ─── 金融専用計算 ─────────────────────────────────────────

def calc_financial_cols(db: dict, current_price: Optional[float]) -> dict:
    """M〜T 列の金融専用指標を計算して返す"""
    out = {k: None for k in ["coe_fin", "roe_norm", "roe_coe_spread",
                               "justified_pbr", "fair_price_fin",
                               "implied_roe", "judgment_fin"]}
    coe = db.get("coe")
    roe_n = db.get("roe_normalized")
    bps = db.get("bps")
    payout = db.get("payout_ratio")
    g_terminal = DEFAULT_TERMINAL_GROWTH

    out["coe_fin"] = coe
    out["roe_norm"] = roe_n

    if coe is None or roe_n is None:
        out["judgment_fin"] = "算出不能"
        return out

    out["roe_coe_spread"] = roe_n - coe

    # Q列: Justified PBR
    # g = ROE × (1 - payout)
    if payout is not None and 0 <= payout <= 1.2:
        g_q = roe_n * (1 - payout)
    else:
        g_q = None

    justified_pbr = None
    if g_q is not None and coe > g_q:
        justified_pbr = (roe_n - g_q) / (coe - g_q)
    out["justified_pbr"] = justified_pbr

    # R列: 金融_適正株価
    fair_price_fin = None
    if bps is not None and justified_pbr is not None:
        fair_price_fin = bps * justified_pbr
    out["fair_price_fin"] = fair_price_fin

    # S列: 市場織込ROE
    # ROE_implied = PB_now × (CoE - g_terminal) + g_terminal
    pb_now = db.get("pb_now")
    implied_roe = None
    if pb_now is not None and coe is not None:
        implied_roe = pb_now * (coe - g_terminal) + g_terminal
    out["implied_roe"] = implied_roe

    # T列: 金融_判定
    if fair_price_fin is None:
        out["judgment_fin"] = "算出不能"
    elif roe_n <= coe:
        out["judgment_fin"] = "改善待ち"
    elif current_price is not None:
        if implied_roe is not None and implied_roe > roe_n * 1.3:
            out["judgment_fin"] = "市場期待過大"
        elif current_price < fair_price_fin * 0.8:
            out["judgment_fin"] = "割安候補"
        elif current_price > fair_price_fin * 1.2:
            out["judgment_fin"] = "期待先行"
        else:
            out["judgment_fin"] = "妥当"
    else:
        out["judgment_fin"] = "算出不能"

    return out


# ─── メイン処理 ──────────────────────────────────────────

def build_row(existing_row: list, db: dict, financial_flag: bool,
              fin: dict, ep: Optional[float], vdf: Optional[float],
              g_implied: Optional[float], gap_yrs: Optional[float],
              current_price: Optional[float]) -> list:
    """
    シートの1行分（TOTAL_COLS 長）のリストを組み立てる。
    A〜D列はユーザー入力なので書き込まない（元の値を保持）。
    """
    row = list(existing_row) + [""] * (TOTAL_COLS - len(existing_row))

    def set_col(col_idx, val):
        if col_idx < TOTAL_COLS:
            row[col_idx] = "" if val is None else val

    # E列: Economic Profit 株価
    set_col(COL["E"], round(ep, 1) if ep is not None else None)

    # F列: VDF株価
    set_col(COL["F"], round(vdf, 1) if vdf is not None else None)

    # G列: 市場織込成長率
    set_col(COL["G"], pct_fmt(g_implied) if g_implied is not None else None)

    # H列: GAP年数
    set_col(COL["H"], round(gap_yrs, 1) if gap_yrs is not None else None)

    # I列: ROIC-WACC スプレッド
    roic_n = db.get("roic_normalized")
    wacc = db.get("wacc")
    spread = None
    if roic_n is not None and wacc is not None:
        spread = roic_n - wacc
    set_col(COL["I"], pct_fmt(spread) if spread is not None else None)

    # J列: 統合適正株価
    j_val = None
    if financial_flag:
        j_val = fin.get("fair_price_fin")
    else:
        if ep is not None and vdf is not None:
            j_val = (ep + vdf) / 2
        elif ep is not None:
            j_val = ep
        elif vdf is not None:
            j_val = vdf
    set_col(COL["J"], round(j_val, 1) if j_val is not None else None)

    # K列: 差異率
    k_val = None
    if j_val is not None and current_price is not None and j_val != 0:
        k_val = (current_price - j_val) / j_val
    set_col(COL["K"], pct_fmt(k_val) if k_val is not None else None)

    # L列: 総合判定
    if financial_flag:
        set_col(COL["L"], fin.get("judgment_fin") or "算出不能")
    else:
        if j_val is None:
            set_col(COL["L"], "算出不能")
        elif spread is not None and spread <= 0:
            set_col(COL["L"], "改善待ち")
        elif k_val is not None and k_val <= -0.20 and spread is not None and spread > 0:
            # G < 15%, H < 20年
            g_ok = g_implied is None or g_implied < 0.15
            h_ok = gap_yrs is None or gap_yrs < 20
            if g_ok and h_ok:
                set_col(COL["L"], "割安候補")
            else:
                set_col(COL["L"], "期待先行")
        elif k_val is not None and abs(k_val) < 0.20:
            set_col(COL["L"], "妥当")
        elif k_val is not None and (k_val >= 0.20 or
                                     (g_implied is not None and g_implied >= 0.15) or
                                     (gap_yrs is not None and gap_yrs >= 20)):
            set_col(COL["L"], "期待先行")
        else:
            set_col(COL["L"], "算出不能")

    # M列: 金融フラグ
    set_col(COL["M"], 1 if financial_flag else 0)

    # N〜T 列（金融専用）
    set_col(COL["N"], pct_fmt(fin.get("coe_fin")))
    set_col(COL["O"], pct_fmt(fin.get("roe_norm")))
    set_col(COL["P"], pct_fmt(fin.get("roe_coe_spread")))
    set_col(COL["Q"], round(fin.get("justified_pbr"), 3) if fin.get("justified_pbr") is not None else None)
    set_col(COL["R"], round(fin.get("fair_price_fin"), 1) if fin.get("fair_price_fin") is not None else None)
    set_col(COL["S"], pct_fmt(fin.get("implied_roe")))
    set_col(COL["T"], fin.get("judgment_fin") or "")

    # U列: E/F乖離率
    u_val = None
    if ep is not None and vdf is not None:
        denom = (ep + vdf) / 2
        if denom != 0:
            u_val = abs(ep - vdf) / denom
    set_col(COL["U"], pct_fmt(u_val) if u_val is not None else None)

    # DB列 (AA以降)
    for i, col_name in enumerate(DB_COLS):
        col_idx = COL["AA"] + i
        val = db.get(col_name)
        if val is None:
            val = ""
        # financial_flag_override は既存シートの値を保持
        if col_name == "financial_flag_override":
            existing_val = row[col_idx] if col_idx < len(row) else ""
            val = existing_val if str(existing_val).strip() in ("0", "1") else ""
        # financial_flag は計算結果で上書き
        if col_name == "financial_flag":
            val = 1 if financial_flag else 0
        set_col(col_idx, val)

    return row


def error_guidance_message(e: Exception) -> str:
    msg = str(e)
    lower_msg = msg.lower()

    if "grid limits" in lower_msg or "max columns" in lower_msg or "max rows" in lower_msg:
        return "修正方針: スプレッドシートの行数・列数が不足しています。対象シートのグリッドを拡張してください。"
    if "permission" in lower_msg or "forbidden" in lower_msg or "403" in lower_msg:
        return "修正方針: サービスアカウントに対象スプレッドシートの編集権限を付与してください。"
    if "not found" in lower_msg or "404" in lower_msg:
        return "修正方針: スプレッドシートURL、シート名、銘柄コードの指定を確認してください。"
    if "429" in lower_msg or "quota" in lower_msg or "rate limit" in lower_msg:
        return "修正方針: API呼び出し頻度を下げるか、待機時間を増やしてください。"
    if "secrets" in lower_msg or "credential" in lower_msg or "auth" in lower_msg:
        return "修正方針: secrets.json の内容と GCP サービスアカウント鍵を確認してください。"

    return "修正方針: エラー文を確認し、secrets・シート設定・入力銘柄コードの順に切り分けてください。"


def log_error_with_guidance(e: Exception):
    print(f"ERROR: {type(e).__name__}: {e}")
    print(error_guidance_message(e))


def main():

    # secrets 読み込み
    with open(SECRETS_PATH, "r", encoding="utf-8") as f:
        secrets = json.load(f)

    # シート接続
    ws = connect_sheet(secrets)

    # 全データ取得
    all_values = ws.get_all_values()
    if not all_values:
        return

    header_row = 0  # 0行目をヘッダーとして扱う（1行目がデータ）
    data_start = 1  # データ開始行インデックス

    # ヘッダー行を確認・作成
    if all_values and all_values[0] and str(all_values[0][0]).strip().upper() in ("A", "銘柄コード", ""):
        # ヘッダーがある場合
        data_rows = all_values[data_start:]
        header_offset = 2  # シート上の行番号（1始まり）でデータが2行目から
    else:
        # ヘッダーなし（1行目からデータ）
        data_rows = all_values
        header_offset = 1

    updated_rows = []

    for row_idx, raw_row in enumerate(data_rows):
        sheet_row_num = row_idx + header_offset

        # A列: 銘柄コード（文字列として扱う）
        ticker_code = str(raw_row[0]).strip() if raw_row else ""
        if not ticker_code or ticker_code.lower() in ("none", "nan", "a", "銘柄コード"):
            continue

        try:
            # yfinanceデータ取得
            db = fetch_yf_data(ticker_code)
            time.sleep(YFINANCE_WAIT)

            current_price = db.get("current_price")

            # financial_flag_override の既存値を取得
            override_col = COL["AA"] + DB_COLS.index("financial_flag_override")
            override_val = raw_row[override_col] if override_col < len(raw_row) else ""

            # 金融判定
            fin_flag = is_financial(db, override_val)

            # 評価計算
            if fin_flag:
                ep_price = None
                vdf_price = None
                g_implied = None
                gap_yrs = None
                fin_cols = calc_financial_cols(db, current_price)
            else:
                ep_price = calc_ep_price(db)
                vdf_price = calc_vdf_price(db)
                g_implied = calc_implied_growth(db)
                gap_yrs = calc_gap_years(db)
                fin_cols = {k: None for k in ["coe_fin", "roe_norm", "roe_coe_spread",
                                               "justified_pbr", "fair_price_fin",
                                               "implied_roe", "judgment_fin"]}

            # DB: Reverse DCF 結果を保存
            db["implied_growth_rate"] = g_implied
            db["implied_gap_years"] = gap_yrs
            if fin_flag:
                db["financial_justified_pbr"] = fin_cols.get("justified_pbr")
                db["financial_implied_roe"] = fin_cols.get("implied_roe")

            # 行データ組み立て
            new_row = build_row(
                raw_row, db, fin_flag, fin_cols,
                ep_price, vdf_price, g_implied, gap_yrs, current_price
            )

            updated_rows.append((sheet_row_num, new_row))

        except Exception as e:
            log_error_with_guidance(e)

    # シートへの書き込み（バッチ更新）
    if updated_rows:
        for sheet_row_num, new_row in updated_rows:
            # E列以降のみ更新（A〜D列はユーザー入力を保護）
            start_col_letter = "E"
            write_data = new_row[COL["E"]:]
            cell_range = f"{start_col_letter}{sheet_row_num}"
            try:
                ws.update(
                    range_name=cell_range,
                    values=[write_data]
                )
                time.sleep(0.3)  # API制限回避
            except Exception as e:
                log_error_with_guidance(e)


if __name__ == "__main__":
    main()
