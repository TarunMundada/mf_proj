from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import time
import math
from dateutil.relativedelta import relativedelta

app = FastAPI(title="MF Backtest (mfapi every-request)")

MFAPI_BASE = "https://api.mfapi.in/mf"
REQUEST_TIMEOUT = 15
RETRIES = 3
RETRY_BACKOFF = 1.0  # seconds

# ---------- request models ----------
class SIPParams(BaseModel):
    amount_per_month: float
    day_of_month: int = 10  # invest on 10th by default

class LumpsumParams(BaseModel):
    amount: float

class BacktestRequest(BaseModel):
    # Either provide scheme_code OR scheme_query (name) - we accept both
    scheme_code: Optional[int] = None
    scheme_query: Optional[str] = None
    start_date: date
    end_date: date
    strategy: Literal["SIP", "LUMPSUM"]
    sip: Optional[SIPParams] = None
    lumpsum: Optional[LumpsumParams] = None
    risk_free_rate_annual: float = 0.06  # default 6% annual

# ---------- helper: mfapi fetch ----------
def mfapi_get_json(path: str, params=None):
    url = MFAPI_BASE + path
    last_err = None
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(RETRY_BACKOFF * (2 ** attempt))
    raise last_err

def resolve_scheme_code_from_query(q: str) -> int:
    # calls /search?q=... and returns first schemeCode
    res = mfapi_get_json("/search", params={"q": q})
    if not res:
        raise HTTPException(status_code=404, detail=f"No scheme found for query: {q}")
    first = res[0]
    return int(first["schemeCode"])

def fetch_nav_df(scheme_code: int) -> pd.DataFrame:
    # fetch full history on every request (mfapi returns full history for /{code})
    payload = mfapi_get_json(f"/{scheme_code}")
    if "data" not in payload:
        raise HTTPException(status_code=502, detail="mfapi returned unexpected payload")
    df = pd.DataFrame(payload["data"])
    # mfapi returns dates as DD-MM-YYYY
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df["nav"] = df["nav"].astype(float)
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "nav"]]

# ---------- helpers: finance metrics ----------
def compute_max_drawdown(portfolio_values: pd.Series) -> float:
    # portfolio_values indexed by date
    cummax = portfolio_values.cummax()
    drawdown = (portfolio_values - cummax) / cummax
    return float(drawdown.min())  # negative number

def annualized_return(total_return: float, days: int) -> float:
    years = days / 365.25
    if years <= 0:
        return 0.0
    return total_return ** (1 / years) - 1

def annualized_volatility(daily_returns: pd.Series) -> float:
    # daily_returns like pct change series
    if daily_returns.shape[0] < 2 or daily_returns.std() == 0:
        return 0.0
    return float(daily_returns.std() * (252 ** 0.5))

def sharpe_ratio(annual_ret: float, annual_vol: float, rf: float):
    if annual_vol == 0:
        return None
    return (annual_ret - rf) / annual_vol

# ---------- helpers: portfolio simulations ----------
def find_nav_on_or_after(df: pd.DataFrame, target_date: pd.Timestamp):
    # find first available NAV on or after the target_date
    sel = df[df["date"] >= target_date]
    if sel.empty:
        return None
    return float(sel.iloc[0]["nav"]), sel.iloc[0]["date"]

def run_lumpsum(df: pd.DataFrame, start_date: date, end_date: date, amount: float):
    # buy on first available nav >= start_date
    buy = find_nav_on_or_after(df, pd.Timestamp(start_date))
    if buy is None:
        raise HTTPException(status_code=400, detail="No NAV available on/after start_date")
    buy_nav, buy_date = buy
    units = amount / buy_nav
    # build portfolio daily series from buy_date to end_date
    mask = (df["date"] >= buy_date) & (df["date"] <= pd.Timestamp(end_date))
    dfp = df.loc[mask].copy().reset_index(drop=True)
    if dfp.empty:
        raise HTTPException(status_code=400, detail="No NAVs in requested date range after buy date")
    dfp["units"] = units
    dfp["portfolio_value"] = dfp["units"] * dfp["nav"]
    dfp["cashflow"] = 0.0
    dfp.loc[dfp.index[0], "cashflow"] = amount  # initial investment on buy date row
    dfp["cumulative_invested"] = dfp["cashflow"].cumsum()
    return dfp

def month_iter(start: date, end: date, day_of_month: int):
    cur = start
    # align first month: if start.day <= day_of_month then invest in that month else next month
    if start.day <= day_of_month:
        invest_date = date(start.year, start.month, min(day_of_month, 28))  # min with 28 safe for feb
    else:
        nextm = start + relativedelta(months=1)
        invest_date = date(nextm.year, nextm.month, min(day_of_month, 28))
    while invest_date <= end:
        yield invest_date
        invest_date = (pd.Timestamp(invest_date) + relativedelta(months=1)).date()

def run_sip(df: pd.DataFrame, start_date: date, end_date: date, amount_per_month: float, day_of_month: int):
    # For each monthly investment date, find first available NAV on/after that date and buy units
    # We'll accumulate units and produce daily portfolio from first purchase date to end_date
    purchases = []
    for invest_dt in month_iter(start_date, end_date, day_of_month):
        found = find_nav_on_or_after(df, pd.Timestamp(invest_dt))
        if found is None:
            # if no NAV after this invest date, skip this month's purchase
            continue
        nav, actual_date = found
        units = amount_per_month / nav
        purchases.append({"date": pd.Timestamp(actual_date), "amount": amount_per_month, "units": units, "nav": nav})

    if not purchases:
        raise HTTPException(status_code=400, detail="No successful SIP purchases in requested range")

    first_buy_date = purchases[0]["date"]
    mask = (df["date"] >= first_buy_date) & (df["date"] <= pd.Timestamp(end_date))
    dfp = df.loc[mask].copy().reset_index(drop=True)
    dfp["units"] = 0.0
    dfp["cashflow"] = 0.0

    # apply purchases: add units on the exact purchase date rows (or the first row that equals the purchase date)
    for p in purchases:
        # find row index in dfp with date == p.date
        idx = dfp.index[dfp["date"] == p["date"]]
        if len(idx) == 0:
            # should not happen as we matched df on >= purchase dates
            continue
        i = idx[0]
        dfp.at[i, "cashflow"] += p["amount"]
        dfp.at[i, "units"] += p["units"]

    # forward-fill total units held
    dfp["units"] = dfp["units"].cumsum()
    dfp["portfolio_value"] = dfp["units"] * dfp["nav"]
    dfp["cumulative_invested"] = dfp["cashflow"].cumsum()
    return dfp

# ---------- endpoint ----------
@app.post("/backtest")
def backtest(req: BacktestRequest):
    # Resolve scheme_code
    scheme_code = req.scheme_code
    if scheme_code is None:
        if not req.scheme_query:
            raise HTTPException(status_code=400, detail="Provide scheme_code or scheme_query")
        scheme_code = resolve_scheme_code_from_query(req.scheme_query)

    # fetch NAVs every request
    df_nav = fetch_nav_df(int(scheme_code))

    # limit date range to available NAVs (we'll let simulation handle errors)
    if req.start_date > req.end_date:
        raise HTTPException(status_code=400, detail="start_date must be <= end_date")

    # run chosen simulation
    if req.strategy == "LUMPSUM":
        if not req.lumpsum:
            raise HTTPException(status_code=400, detail="lumpsum params required for LUMPSUM strategy")
        dfp = run_lumpsum(df_nav, req.start_date, req.end_date, req.lumpsum.amount)

    elif req.strategy == "SIP":
        if not req.sip:
            raise HTTPException(status_code=400, detail="sip params required for SIP strategy")
        dfp = run_sip(df_nav, req.start_date, req.end_date, req.sip.amount_per_month, req.sip.day_of_month)

    else:
        raise HTTPException(status_code=400, detail="Unsupported strategy")

    # compute metrics
    pv = dfp["portfolio_value"]
    invested = dfp["cumulative_invested"].iloc[-1]
    final_value = float(pv.iloc[-1])
    total_return_abs = final_value - invested
    total_return_mul = (final_value / invested) if invested > 0 else 0.0

    days = (dfp["date"].iloc[-1] - dfp["date"].iloc[0]).days
    ann_ret = annualized_return(total_return_mul, days) if invested > 0 else 0.0

    daily_ret = pv.pct_change().fillna(0)
    ann_vol = annualized_volatility(daily_ret)
    max_dd = compute_max_drawdown(pv)

    sharpe = sharpe_ratio(ann_ret, ann_vol, req.risk_free_rate_annual)

    # prepare response (time series as list of objects)
    timeseries = dfp[["date", "nav", "units", "cashflow", "cumulative_invested", "portfolio_value"]].copy()
    timeseries["date"] = timeseries["date"].dt.strftime("%Y-%m-%d")
    ts_list = timeseries.to_dict(orient="records")

    resp = {
        "scheme_code": int(scheme_code),
        "start_date": req.start_date.isoformat(),
        "end_date": req.end_date.isoformat(),
        "strategy": req.strategy,
        "metrics": {
            "final_value": final_value,
            "invested": float(invested),
            "total_return_abs": float(total_return_abs),
            "total_return_mul": float(total_return_mul),
            "annualized_return": float(ann_ret),
            "annualized_volatility": float(ann_vol),
            "max_drawdown": float(max_dd),
            "sharpe_ratio": None if sharpe is None else float(sharpe)
        },
        "timeseries": ts_list
    }
    return resp