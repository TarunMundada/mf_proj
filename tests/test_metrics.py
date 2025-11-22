import pandas as pd
from strategies.lumpsum import simulate_lumpsum
from analytics.metrics import summary_metrics, xirr_from_portfolio

def tiny_nav_df():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    navs = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]   # linear uptrend
    return pd.DataFrame({"date": dates, "nav": navs})

def test_metrics():
    df = tiny_nav_df()
    p = simulate_lumpsum(df, 1000, "2024-01-01", "2024-01-10")

    m = summary_metrics(p, risk_free_rate_annual=0)

    # invested = 1000
    assert abs(m['invested'] - 1000) < 1e-6
    # final value = units(100)*19 = 1900
    assert abs(m['final_value'] - 1900) < 1e-6
    # total return = +90%
    assert abs(m['total_return_pct'] - 90) < 1e-3

    # Sharpe > 0 since returns are always positive
    assert m['sharpe'] > 0

    # XIRR should be close to the CAGR of this period (~ around 196% annualized because 10 days)
    xirr, _ = xirr_from_portfolio(p)
    assert xirr > 0