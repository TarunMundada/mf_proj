import pandas as pd
from strategies.sip import simulate_sip

def tiny_nav_df():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    navs = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    return pd.DataFrame({"date": dates, "nav": navs})

def test_simple_sip():
    df = tiny_nav_df()
    # SIP on day=2, monthly_amount=100
    portfolio = simulate_sip(
        nav_df=df,
        monthly_amount=100,
        sip_day=2,
        start_date="2024-01-01",
        end_date="2024-01-10",
        initial_amount=0
    )

    # The SIP should trigger on Jan 2 → NAV=11 → units=100/11
    first_tx = portfolio.loc[portfolio['cashflow'] > 0].iloc[0]
    assert abs(first_tx['units_bought'] - (100/11)) < 1e-6
    assert first_tx['cashflow'] == 100

    # Total units after first tx
    assert abs(portfolio['total_units'].max() - (100/11)) < 1e-6