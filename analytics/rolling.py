import pandas as pd
import numpy as np

def rolling_periodic_returns(portfolio_daily, window_days):
    """
    Compute simple rolling returns over a window of `window_days` trading days.
    Returns a DataFrame with columns: date, rolling_return (fractional)
    Note: requires at least window_days + 1 rows; NaN where not available.
    """
    df = portfolio_daily.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['rolling_return'] = df['portfolio_value'].pct_change(periods=window_days)
    return df[['date', 'rolling_return']]

def rolling_annualized_return(portfolio_daily, window_days, annualize_by=252):
    """
    Convert rolling periodic returns into annualized returns:
      annualized = (1 + periodic_return) ** (annualize_by / window_days) - 1
    window_days: number of trading days over which the periodic return was computed (e.g., 252 for 1-year)
    """
    r = rolling_periodic_returns(portfolio_daily, window_days)
    r['annualized_return'] = r['rolling_return'].apply(
        lambda x: (1 + x) ** (annualize_by / window_days) - 1 if pd.notnull(x) else np.nan
    )
    return r[['date', 'annualized_return']]

def rolling_years(portfolio_daily, years=1, trading_days_per_year=252):
    """
    Convenience wrapper to compute rolling annualized returns for `years` (1, 3, 5 etc).
    Uses window_days = years * trading_days_per_year (approx).
    Returns DataFrame with date and annualized_return.
    """
    window_days = int(years * trading_days_per_year)
    return rolling_annualized_return(portfolio_daily, window_days, annualize_by=trading_days_per_year)

def rolling_multi_years(portfolio_daily, years_list=[1,3,5], trading_days_per_year=252):
    """
    Compute rolling annualized returns for multiple year windows.
    Returns a DataFrame with date and columns annualized_{n}yr
    """
    df = pd.DataFrame({'date': pd.to_datetime(portfolio_daily['date'])})
    df = df.sort_values('date').reset_index(drop=True)
    for y in years_list:
        r = rolling_years(portfolio_daily, years=y, trading_days_per_year=trading_days_per_year)
        col = f'annualized_{y}yr'
        df = df.merge(r.rename(columns={'annualized_return': col}), on='date', how='left')
    return df

# quick example helper
def last_n_rolling_summary(portfolio_daily, years=1, trading_days_per_year=252):
    """
    Return simple stats for the rolling returns of the last n years:
      mean, median, min, max for the rolling annualized returns
    """
    r = rolling_years(portfolio_daily, years=years, trading_days_per_year=trading_days_per_year)
    arr = r['annualized_return'].dropna()
    if arr.empty:
        return {}
    return {
        'mean_pct': float(arr.mean() * 100.0),
        'median_pct': float(arr.median() * 100.0),
        'min_pct': float(arr.min() * 100.0),
        'max_pct': float(arr.max() * 100.0),
        'count': int(arr.count())
    }