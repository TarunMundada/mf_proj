import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import uvicorn

# --- import your existing modules (ensure these paths/names match your project) ---
from service.data_fetcher import search_funds    # implement this in service/data_fetcher.py
from service.nav_parallel import fetch_navs_parallel
from strategies.sip import simulate_sip, _ensure_nav_df
from strategies.portfolio_fixed_split import merge_asset_series
from analytics.metrics import summary_metrics

# ---- FastAPI app ----
app = FastAPI(title="MF Backtester API")

# Dev-time: allow all origins. In prod lock this down.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for blocking IO (network + pandas)
executor = ThreadPoolExecutor(max_workers=8)


# ---- Pydantic request models ----
class AssetInput(BaseModel):
    id: Optional[str] = None
    scheme_code: int
    monthly_amount: float = 0.0
    sip_day: int = 10
    initial_amount: float = 0.0

class PortfolioRequest(BaseModel):
    assets: List[AssetInput]
    start_date: str
    end_date: str


# ---- Search endpoint (typeahead) ----
@app.get("/api/search")
async def api_search(q: str):
    """
    Wrapper around your service.data_fetcher.search_funds(q).
    Must return a list of objects with at least schemeCode / schemeName keys.
    """
    if not q or len(q.strip()) < 1:
        return []
    loop = asyncio.get_event_loop()
    try:
        # run blocking search in threadpool if search_funds is synchronous
        results = await loop.run_in_executor(executor, lambda: search_funds(q))
        # normalize keys if necessary in your search_funds implementation
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- Portfolio backtest endpoint ----
@app.post("/portfolio/backtest")
async def portfolio_backtest(req: PortfolioRequest):
    """
    Runs the fixed-split portfolio backtest (Mode A).
    Workflow:
      1. Normalize asset inputs (assign id if missing)
      2. Fetch NAVs in parallel via fetch_navs_parallel
      3. Simulate each asset with simulate_sip (run in threadpool)
      4. Merge per-asset series (merge_asset_series) and compute metrics
      5. Return JSON: portfolio_daily, transactions, metrics
    """
    if not req.assets:
        raise HTTPException(status_code=400, detail="No assets provided")

    # normalize assets, ensure id unique
    asset_inputs = []
    for i, a in enumerate(req.assets):
        aid = str(a.scheme_code) if (not a.id) else str(a.id)
        asset_inputs.append({
            "id": aid,
            "scheme_code": int(a.scheme_code),
            "monthly_amount": float(a.monthly_amount),
            "sip_day": int(a.sip_day),
            "initial_amount": float(a.initial_amount)
        })

    loop = asyncio.get_event_loop()

    # 1) fetch NAVs in parallel (blocking) -- runs in threadpool
    try:
        nav_map = await loop.run_in_executor(executor, lambda: fetch_navs_parallel(asset_inputs, max_workers=6))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch NAVs: {e}")

    # 2) simulate each asset using simulate_sip (offload to threadpool)
    asset_series = {}
    tx_frames = []
    for a in asset_inputs:
        aid = a["id"]
        nav_df = nav_map.get(aid)
        if nav_df is None:
            # If fetch_navs_parallel returned schemes keyed differently, try scheme_code string
            nav_df = nav_map.get(str(a["scheme_code"]))
        if nav_df is None:
            raise HTTPException(status_code=502, detail=f"NAV missing for asset {aid}")

        # clean nav_df
        nav_df = _ensure_nav_df(nav_df)

        # run simulate_sip in threadpool (blocking)
        df_asset = await loop.run_in_executor(
            executor,
            lambda nav_df=nav_df, a=a, sd=req.start_date, ed=req.end_date: simulate_sip(nav_df, a['monthly_amount'], a['sip_day'], sd, ed, initial_amount=a['initial_amount'])
        )

        asset_series[aid] = df_asset

        # extract transaction rows for response
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

    # 3) merge per-asset series (this is CPU-bound pandas work, run in threadpool)
    try:
        merged = await loop.run_in_executor(executor, lambda: merge_asset_series(asset_series))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to merge series: {e}")

    # 4) compute portfolio-level metrics (can be CPU-bound)
    try:
        metrics = await loop.run_in_executor(executor, lambda: summary_metrics(merged))
    except Exception as e:
        # keep going but show that metrics failed
        metrics = {"error": str(e)}

    # 5) prepare transactions and serializable outputs
    transactions = (pd.concat(tx_frames, ignore_index=True).sort_values('date').reset_index(drop=True) if tx_frames else pd.DataFrame())

    # convert dates to ISO strings for JSON
    merged_serial = merged.copy()
    merged_serial['date'] = pd.to_datetime(merged_serial['date']).dt.strftime('%Y-%m-%d')
    portfolio_daily = merged_serial.to_dict(orient='records')

    tx_json = transactions.copy()
    if not tx_json.empty:
        tx_json['date'] = pd.to_datetime(tx_json['date']).dt.strftime('%Y-%m-%d')
        tx_json = tx_json.to_dict(orient='records')
    else:
        tx_json = []

    return {
        "portfolio_daily": portfolio_daily,
        "transactions": tx_json,
        "metrics": metrics,
        "meta": {"generated_by": "mf_backtester_api"}
    }


# ---- Optional: serve built frontend (if you want static + api on same origin) ----
# Put your built UI in ui/dist (run `cd ui && npm run build`) and uncomment this block.
# app.mount("/", StaticFiles(directory="ui/dist", html=True), name="static")


# ---- run with: python -m uvicorn main:app --reload (or uvicorn main:app --reload) ----
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)