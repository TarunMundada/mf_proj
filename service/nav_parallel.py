from concurrent.futures import ThreadPoolExecutor, as_completed
from service.data_fetcher import get_nav_history
from service.parser import nav_json_to_df

def fetch_nav_for_one(asset):
    """
    asset: {"id": "...", "scheme_code": 12345}
    returns (id, nav_df)
    """
    aid = asset["id"]
    code = asset["scheme_code"]

    try:
        raw = get_nav_history(code)           # mfapi call
        nav_df = nav_json_to_df(raw)          # convert to df
        return (aid, nav_df, None)
    except Exception as e:
        return (aid, None, e)

def fetch_navs_parallel(asset_inputs, max_workers=8):
    """
    asset_inputs: list of {"id": "...", "scheme_code": 12345}
    returns dict: {id: nav_df}
    """
    out = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_nav_for_one, asset) for asset in asset_inputs]

        for f in as_completed(futures):
            aid, nav_df, err = f.result()
            if err is not None:
                print(f"[ERROR] Failed fetching NAV for {aid}: {err}")
                raise err
            out[aid] = nav_df

    return out