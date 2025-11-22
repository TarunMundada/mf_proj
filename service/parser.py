import pandas as pd

def nav_json_to_df(json_data):
    df = pd.DataFrame(json_data["data"])
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df["nav"] = df["nav"].astype(float)
    df = df.sort_values("date").reset_index(drop=True)
    return df