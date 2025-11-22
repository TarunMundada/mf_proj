import pandas as pd
from strategies.helpers import _ensure_nav_df, _next_trading_day_in_month

def simulate_sip(nav_df, monthly_amount, sip_day, start_date, end_date, initial_amount=0.0):
    """
    SIP simulation with optional initial lumpsum.

    Parameters
    ----------
    nav_df : DataFrame with columns ['date','nav'] (date is datetime)
    monthly_amount : float   -> amount invested each month on SIP date
    sip_day : int            -> target day of month for SIP (1-31). function falls back to next trading day in month
    start_date : str/date    -> inclusive
    end_date : str/date      -> inclusive
    initial_amount : float   -> optional initial lumpsum invested on first trading day >= start_date

    Returns
    -------
    DataFrame with columns:
      date, nav, cashflow, units_bought, total_units, portfolio_value, cumulative_invested
    """
    df = _ensure_nav_df(nav_df)
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()

    # restrict navs to window for efficiency, but helpers use full df for month lookups
    full_df = df  # for monthly lookup we use the full cleaned df
    window_mask = (df['date'].dt.date >= start) & (df['date'].dt.date <= end)
    window_df = df.loc[window_mask].copy().reset_index(drop=True)
    if window_df.empty:
        raise ValueError("No NAV data in requested date range")

    # --- find monthly invest dates between start and end ---
    months = []
    cur = start.replace(day=1)
    while cur <= end:
        months.append((cur.year, cur.month))
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)

    invest_dates = []
    for (y, m) in months:
        target = _next_trading_day_in_month(full_df, y, m, sip_day)
        if target is None:
            continue
        if target < start or target > end:
            continue
        invest_dates.append(target)
    invest_dates = sorted(set(invest_dates))

    # --- find initial lumpsum invest date (first trading day on/after start) if requested ---
    initial_invest_date = None
    if initial_amount and initial_amount > 0:
        cand = full_df.loc[full_df['date'].dt.date >= start].head(1)
        if cand.empty:
            raise ValueError("No NAV available on or after start_date to perform initial investment")
        initial_invest_date = cand.iloc[0]['date'].date()

    invest_dates_set = set(invest_dates)  # for fast membership checks

    # --- build daily portfolio rows across trading days in window_df ---
    rows = []
    total_units = 0.0
    cumulative_invested = 0.0

    for _, row in window_df.iterrows():
        d = row['date'].date()
        nav = float(row['nav'])
        cashflow = 0.0
        units_today = 0.0

        # initial lumpsum (if this is the first available trading day >= start)
        if initial_invest_date is not None and d == initial_invest_date:
            if nav > 0:
                cashflow += float(initial_amount)
                units_init = float(initial_amount) / nav
                units_today += units_init
                total_units += units_init
                cumulative_invested += float(initial_amount)
            else:
                # skip initial if nav invalid (shouldn't happen after cleaning)
                pass

        # monthly SIP on invest dates
        if d in invest_dates_set:
            if nav > 0:
                cashflow += float(monthly_amount)
                units_sip = float(monthly_amount) / nav
                units_today += units_sip
                total_units += units_sip
                cumulative_invested += float(monthly_amount)
            else:
                # skip this month's SIP if nav invalid
                pass

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