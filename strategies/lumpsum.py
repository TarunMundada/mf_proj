import pandas as pd
from strategies.helpers import _ensure_nav_df

def simulate_lumpsum(nav_df, amount, start_date, end_date=None):
    """
    Lumpsum simulation.
    nav_df: DataFrame with date (datetime) and nav (float)
    amount: float, total invested at start_date
    start_date: string or date-like (investment date)
    end_date: optional, string/date. If None, simulate until last available NAV.
    Returns: portfolio_daily DataFrame
    """
    df = _ensure_nav_df(nav_df)
    start_date = pd.to_datetime(start_date).date()
    if end_date:
        end_date = pd.to_datetime(end_date).date()
    else:
        end_date = df['date'].dt.date.max()

    # find first trading day on/after start_date
    cand = df.loc[df['date'].dt.date >= start_date].head(1)
    if cand.empty:
        raise ValueError("No NAV available on or after start_date within provided NAV data")
    invest_date = cand.iloc[0]['date'].date()
    invest_nav = float(cand.iloc[0]['nav'])
    if invest_nav <= 0:
        raise ValueError("Invalid NAV on invest date")

    units_bought = amount / invest_nav
    total_units = 0.0

    # prepare output rows: all trading days between invest_date and end_date inclusive
    out_mask = (df['date'].dt.date >= invest_date) & (df['date'].dt.date <= end_date)
    out_df = df.loc[out_mask, ['date', 'nav']].copy().reset_index(drop=True)

    rows = []
    total_units = 0.0
    cumulative_invested = 0.0
    for idx, row in out_df.iterrows():
        d = row['date'].date()
        nav = float(row['nav'])
        cashflow = 0.0
        units_today = 0.0
        # invest on invest_date only
        if d == invest_date:
            cashflow = float(amount)
            units_today = units_bought
            total_units += units_today
            cumulative_invested += cashflow

        portfolio_value = total_units * nav
        rows.append({
            'date': row['date'],
            'nav': nav,
            'cashflow': cashflow,
            'units_bought': units_today,
            'total_units': total_units,
            'portfolio_value': portfolio_value,
            'cumulative_invested': cumulative_invested
        })

    return pd.DataFrame(rows)