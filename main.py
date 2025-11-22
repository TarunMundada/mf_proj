from service.data_fetcher import get_nav_history
from service.parser import nav_json_to_df

def main():
    scheme_code = 125497  # example fund

    # Step 1: Fetch from mfapi
    json_data = get_nav_history(scheme_code)

    # Step 2: Convert to DataFrame
    df = nav_json_to_df(json_data)

    print(df.head())

if __name__ == "__main__":
    main()