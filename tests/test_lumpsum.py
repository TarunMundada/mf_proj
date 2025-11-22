import pandas as pd
from strategies.lumpsum import simulate_lumpsum

def tiny_nav_df():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    navs = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    return pd.DataFrame({"date": dates, "nav": navs})

def test_lumpsum():
    df = tiny_nav_df()
    p = simulate_lumpsum(
        nav_df=df,
        amount=1000,
        start_date="2024-01-01",
        end_date="2024-01-10"
    )

    first = p.iloc[0]
    assert first['cashflow'] == 1000
    # initial NAV = 10 → units = 1000 / 10 = 100
    assert abs(first['units_bought'] - 100) < 1e-6
    assert abs(first['total_units'] - 100) < 1e-6

    # final value = 100 * last NAV (19) = 1900
    assert abs(p.iloc[-1]['portfolio_value'] - 1900) < 1e-6