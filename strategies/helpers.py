import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def _ensure_nav_df(nav_df):
    if 'date' not in nav_df.columns or 'nav' not in nav_df.columns:
        raise ValueError("nav_df must contain 'date' and 'nav' columns")
    df = nav_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
    df = df.dropna(subset=['nav'])
    if df.empty:
        raise ValueError("NAV dataframe contains no valid rows after cleaning")
    return df

def _next_trading_day_in_month(df, year, month, day):
    try:
        candidate = datetime(year, month, day).date()
    except ValueError:
        # day may be invalid (e.g., 31 for some months) -> pick last day of month
        last_day = (datetime(year, month, 28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        candidate = last_day.date()

    # all available dates for that month
    month_mask = (df['date'].dt.year == year) & (df['date'].dt.month == month)
    available = df.loc[month_mask, 'date'].dt.date.sort_values()
    if available.empty:
        return None

    # if exact candidate present, pick it
    if candidate in set(available):
        return candidate

    # next available on/after candidate
    nexts = [d for d in available if d >= candidate]
    if nexts:
        return nexts[0]

    # fallback: previous available
    prevs = [d for d in available if d < candidate]
    if prevs:
        return prevs[-1]

    return None