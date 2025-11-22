import pandas as pd
from strategies.lumpsum import simulate_lumpsum
from analytics.rolling import rolling_years, rolling_periodic_returns

def tiny_nav_df():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    navs = list(range(10, 30))   # perfectly linear
    return pd.DataFrame({"date": dates, "nav": navs})

def test_rolling():
    df = tiny_nav_df()
    p = simulate_lumpsum(df, 1000, "2024-01-01", "2024-01-20")

    # 5-day rolling return
    r = rolling_periodic_returns(p, 5)
    assert len(r) == len(p)
    assert 'rolling_return' in r.columns

    # now test annualized 1-year (252 days approx)
    r2 = rolling_years(p, years=1)
    assert 'annualized_return' in r2.columns