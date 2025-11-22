# analytics/metrics.py
import pandas as pd
import numpy as np
from datetime import datetime
from math import sqrt

def _ensure_df(df):
    df = df.copy()
    if 'date' not in df.columns or 'portfolio_value' not in df.columns:
        raise ValueError("portfolio_daily must contain 'date' and 'portfolio_value' columns")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['portfolio_value'] = pd.to_numeric(df['portfolio_value'], errors='coerce')
    return df

def final_and_invested(portfolio_daily):
    """
    Returns final_value, cumulative_invested (sum of positive cashflows).
    """
    df = _ensure_df(portfolio_daily)
    final_value = float(df['portfolio_value'].iloc[-1])
    invested = 0.0
    if 'cashflow' in df.columns:
        # sum of positive cashflows (investments)
        invested = float(df.loc[df['cashflow'] > 0, 'cashflow'].sum())
    return final_value, invested

def total_return_pct(portfolio_daily):
    """
    (final / invested) - 1 ; returns None if invested == 0
    """
    final, invested = final_and_invested(portfolio_daily)
    if invested == 0:
        return None
    return (final / invested) - 1.0

def cagr(portfolio_daily):
    """
    Compound Annual Growth Rate based on portfolio_value and invested timing.
    Uses start value = portfolio_value at first row (or invested if you prefer).
    This computes CAGR = (ending / starting)^(1/years) - 1, where years = days/365
    """
    df = _ensure_df(portfolio_daily)
    start_val = float(df['portfolio_value'].iloc[0])
    end_val = float(df['portfolio_value'].iloc[-1])
    days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
    if days <= 0 or start_val <= 0:
        return None
    years = days / 365.0
    return (end_val / start_val) ** (1.0 / years) - 1.0

def daily_returns(portfolio_daily):
    df = _ensure_df(portfolio_daily)
    returns = df['portfolio_value'].pct_change().fillna(0.0)
    returns.index = df['date']
    return returns

def volatility_annualized(portfolio_daily, freq=252):
    """
    Annualized volatility from daily returns (default 252 trading days)
    """
    ret = daily_returns(portfolio_daily)
    # use sample std (ddof=1)
    vol = np.nanstd(ret, ddof=1)
    return vol * np.sqrt(freq)

def sharpe_ratio(portfolio_daily, risk_free_rate_annual=0.06, freq=252):
    """
    Sharpe ratio (annualized).
    risk_free_rate_annual: e.g. 0.06 for 6% p.a.
    Uses arithmetic mean of daily returns annualized.
    """
    ret = daily_returns(portfolio_daily)
    mean_daily = np.nanmean(ret)
    ann_return = (1 + mean_daily) ** freq - 1.0
    vol_ann = volatility_annualized(portfolio_daily, freq=freq)
    if vol_ann == 0:
        return None
    return (ann_return - risk_free_rate_annual) / vol_ann

def max_drawdown(portfolio_daily):
    """
    Returns max drawdown in fractional terms (negative value, e.g., -0.2242).
    Also returns the (peak_date, trough_date).
    """
    df = _ensure_df(portfolio_daily)
    pv = df['portfolio_value']
    cummax = pv.cummax()
    drawdown = (pv - cummax) / cummax
    min_dd = float(drawdown.min())
    trough_idx = int(drawdown.idxmin())
    trough_date = df.loc[trough_idx, 'date']
    # find last peak before trough
    peak_idx = df.loc[:trough_idx, 'portfolio_value'].idxmax()
    peak_date = df.loc[peak_idx, 'date']
    return min_dd, (peak_date, trough_date)

# XIRR implementation (useful when you want IRR of cashflows)
def _year_frac(d0, d1):
    return (d1 - d0).days / 365.0

def xirr_from_portfolio(portfolio_daily):
    """
    Build cashflow list from portfolio_daily and compute XIRR.
    Convention used: investments are negative cashflows (outflows), final portfolio value positive inflow.
    Returns (xirr_decimal, cashflows_list)
    """
    df = _ensure_df(portfolio_daily)
    if 'cashflow' not in df.columns:
        raise ValueError("portfolio_daily must include 'cashflow' column to compute XIRR")
    # build cashflows: investments as negative numbers
    cflows = []
    for _, r in df.loc[df['cashflow'] != 0, ['date', 'cashflow']].iterrows():
        cflows.append((pd.to_datetime(r['date']).date(), -float(r['cashflow'])))
    # final positive inflow
    final = float(df['portfolio_value'].iloc[-1])
    final_date = pd.to_datetime(df['date'].iloc[-1]).date()
    cflows.append((final_date, final))
    # sort
    cflows = sorted(cflows, key=lambda x: x[0])

    # simple root finding based on bisection over a wide range
    def npv(rate):
        d0 = cflows[0][0]
        s = 0.0
        for d, a in cflows:
            yrs = _year_frac(d0, d)
            s += a / ((1.0 + rate) ** yrs)
        return s

    low, high = -0.9999, 10.0
    for _ in range(100):
        mid = (low + high) / 2.0
        try:
            val = npv(mid)
        except Exception:
            low = mid
            continue
        if val > 0:
            low = mid
        else:
            high = mid
    x = (low + high) / 2.0
    return x, cflows

def summary_metrics(portfolio_daily, risk_free_rate_annual=0.06, freq=252):
    """
    Return a dictionary with common metrics:
      - final_value
      - invested
      - total_return_pct
      - cagr_pct
      - max_drawdown_pct
      - volatility_pct (annualized)
      - sharpe
      - xirr_pct
    """
    final, invested = final_and_invested(portfolio_daily)
    total_ret = None if invested == 0 else (final / invested - 1.0)
    cagr_v = cagr(portfolio_daily)
    dd, (peak, trough) = max_drawdown(portfolio_daily)
    vol = volatility_annualized(portfolio_daily, freq=freq)
    sharpe = sharpe_ratio(portfolio_daily, risk_free_rate_annual=risk_free_rate_annual, freq=freq)

    xirr_val = None
    try:
        xirr_val, _ = xirr_from_portfolio(portfolio_daily)
    except Exception:
        xirr_val = None

    return {
        'final_value': final,
        'invested': invested,
        'total_return_pct': None if total_ret is None else total_ret * 100.0,
        'cagr_pct': None if cagr_v is None else cagr_v * 100.0,
        'max_drawdown_pct': dd * 100.0,
        'volatility_pct': vol * 100.0,
        'sharpe': sharpe,
        'xirr_pct': None if xirr_val is None else xirr_val * 100.0
    }