from service.data_fetcher import get_nav_history
from service.parser import nav_json_to_df

from strategies.lumpsum import simulate_lumpsum
from strategies.sip import simulate_sip

from analytics.metrics import summary_metrics, xirr_from_portfolio
from analytics.rolling import rolling_multi_years, last_n_rolling_summary

def main():
    scheme_code = 125497  # example fund

    # Step 1: Fetch from mfapi
    json_data = get_nav_history(scheme_code)

    # Step 2: Convert to DataFrame
    df = nav_json_to_df(json_data)

    # print(df.head())
    
    # portfolio_ls = simulate_lumpsum(nav_df=df,
    #                             amount=100000,            # ₹1,00,000 lumpsum
    #                             start_date="2022-11-22",
    #                             end_date="2025-11-22")
    # print(portfolio_ls.tail())

# SIP example
    portfolio_daily = simulate_sip(nav_df=df,
                                monthly_amount=10000,    # ₹10,000 per month
                                sip_day=22,              # invest on 22th each month (or next trading day)
                                start_date="2022-11-22",
                                end_date="2025-11-22",
                                initial_amount=50000)    # ₹50,000 initial lumpsum
    print(portfolio_daily.tail())

    metrics = summary_metrics(portfolio_daily, risk_free_rate_annual=0.06)
    print(metrics)

    # XIRR and cashflows
    xirr_val, cashflows = xirr_from_portfolio(portfolio_daily)
    print("XIRR %:", xirr_val * 100.0)

    # 1-year and 3-year rolling annualized return series
    rolling = rolling_multi_years(portfolio_daily, years_list=[1,3])
    print(rolling.tail())

    # stats for 1-year rolling returns
    print(last_n_rolling_summary(portfolio_daily, years=1))

if __name__ == "__main__":
    main()

