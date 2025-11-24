import pandas as pd
from typing import List, Dict
from service.data_fetcher import get_nav_history
from service.parser import nav_json_to_df
from strategies.sip import simulate_sip
from strategies.lumpsum import simulate_lumpsum

def _simulate_one_asset_from_api(scheme_code, monthly_amount, sip_day, start_date, end_date, initial_amount=0.0):
    """
    Fetch NAV from mfapi and run simulate_sip (with optional initial).
    Returns the asset's portfolio_daily DataFrame (date, nav, cashflow, units_bought, total_units, portfolio_value, cumulative_invested)
    """
    j = get_nav_history(scheme_code)
    df_nav = nav_json_to_df(j)     # expects 'date' and 'nav' cols
    # ensure df spans requested range (simulate_sip will validate)
    return simulate_sip(df_nav, monthly_amount, sip_day, start_date, end_date, initial_amount=initial_amount)

def merge_asset_series(asset_series_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    asset_series_map: dict asset_id -> per-asset df returned by simulate_sip/simulate_lumpsum
    Output: merged portfolio_daily DataFrame with:
      date (datetime),
      nav_{asset}, units_bought_{asset}, total_units_{asset}, asset_value_{asset}, cuminved_{asset}, cashflow_{asset} ...
      portfolio_value, cashflow (sum), cumulative_invested (sum)
    """
    # collect all date frames and merge on date (outer union)
    dfs = []
    for aid, df in asset_series_map.items():
        d = df.copy()
        # rename columns to prefix with asset id
        rename_map = {
            'nav': f'nav_{aid}',
            'cashflow': f'cashflow_{aid}',
            'units_bought': f'units_bought_{aid}',
            'total_units': f'total_units_{aid}',
            'portfolio_value': f'asset_value_{aid}',
            'cumulative_invested': f'cuminv_{aid}'
        }
        d = d.rename(columns=rename_map)
        # keep only relevant columns
        cols_keep = ['date'] + list(rename_map.values())
        d = d[cols_keep]
        dfs.append(d)

    if not dfs:
        return pd.DataFrame(columns=['date'])

    merged = dfs[0]
    for other in dfs[1:]:
        merged = pd.merge(merged, other, on='date', how='outer')

    # sort and forward-fill numeric columns sensibly
    merged = merged.sort_values('date').reset_index(drop=True)

    # forward-fill navs & units/asset_value/cuminv as appropriate; fill missing with 0
    # Identify asset_value and cuminvs
    asset_value_cols = [c for c in merged.columns if c.startswith('asset_value_')]
    cuminvs = [c for c in merged.columns if c.startswith('cuminv_')]
    cashflow_cols = [c for c in merged.columns if c.startswith('cashflow_')]

    # forward fill navs and total_units where appropriate
    merged = merged.sort_values('date').reset_index(drop=True)
    merged[asset_value_cols] = merged[asset_value_cols].ffill().fillna(0.0)
    merged[cuminvs] = merged[cuminvs].ffill().fillna(0.0)
    merged[cashflow_cols] = merged[cashflow_cols].fillna(0.0)

    # compute portfolio-level aggregates
    merged['portfolio_value'] = merged[asset_value_cols].sum(axis=1)
    merged['cashflow'] = merged[cashflow_cols].sum(axis=1)
    merged['cumulative_invested'] = merged[cuminvs].sum(axis=1)

    # ensure types
    merged['date'] = pd.to_datetime(merged['date'])
    numeric_cols = asset_value_cols + cuminvs + cashflow_cols + ['portfolio_value','cumulative_invested','cashflow']
    for c in numeric_cols:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors='coerce').fillna(0.0)

    return merged

def simulate_portfolio_fixed_split(asset_inputs: List[Dict], start_date: str, end_date: str):
    """
    asset_inputs: list of dicts:
      {
        "id": "ICICI_TECH",
        "scheme_code": 125497,
        "monthly_amount": 15000,
        "sip_day": 10,
        "initial_amount": 50000   # optional
      }
    start_date, end_date: strings "YYYY-MM-DD"

    Returns:
      {
        "portfolio_daily": merged_df,
        "asset_series": {id: df},
        "transactions": combined transactions DataFrame (date,asset,cashflow,units_bought,asset_value)
      }
    """
    asset_series = {}
    tx_frames = []
    for asset in asset_inputs:
        aid = asset['id']
        code = asset['scheme_code']
        monthly = asset.get('monthly_amount', 0.0)
        sip_day = asset.get('sip_day', 10)
        initial = asset.get('initial_amount', 0.0)

        df_asset = _simulate_one_asset_from_api(code, monthly, sip_day, start_date, end_date, initial_amount=initial)
        asset_series[aid] = df_asset

        # extract transactions for this asset
        tx = df_asset.loc[df_asset['cashflow'] != 0, ['date','cashflow','units_bought','total_units','portfolio_value','cumulative_invested']].copy()
        if not tx.empty:
            tx = tx.rename(columns={
                'cashflow':'cashflow',
                'units_bought':'units_bought',
                'total_units':'total_units',
                'portfolio_value':'asset_value',
                'cumulative_invested':'cumulative_invested'
            })
            tx['asset'] = aid
            tx_frames.append(tx[['date','asset','cashflow','units_bought','asset_value','cumulative_invested']])

    merged = merge_asset_series(asset_series)

    transactions = pd.concat(tx_frames, ignore_index=True) if tx_frames else pd.DataFrame(columns=['date','asset','cashflow','units_bought','asset_value','cumulative_invested'])
    transactions = transactions.sort_values('date').reset_index(drop=True)

    return {
        'portfolio_daily': merged,
        'asset_series': asset_series,
        'transactions': transactions
    }