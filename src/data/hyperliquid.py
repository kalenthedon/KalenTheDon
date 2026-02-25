import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

HYPERLIQUID_URL = "https://api.hyperliquid.xyz/info"

def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def fetch_candles_window(symbol: str, interval: str, start_dt: datetime, end_dt: datetime) -> list:
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": symbol,
            "interval": interval,
            "startTime": _to_ms(start_dt),
            "endTime": _to_ms(end_dt),
        },
    }
    r = requests.post(HYPERLIQUID_URL, headers={"Content-Type": "application/json"}, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "candles" in data:
        return data["candles"]
    if isinstance(data, list):
        return data
    return []

def candles_to_df(candles: list) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame([{
        "timestamp": datetime.fromtimestamp(c["t"]/1000, tz=timezone.utc),
        "Open": c["o"],
        "High": c["h"],
        "Low":  c["l"],
        "Close": c["c"],
        "Volume": c["v"],
    } for c in candles])

    df = df.set_index("timestamp").sort_index()

    # ensure numeric
    for col in ["Open","High","Low","Close","Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    return df

def fetch_history_paginated(
    symbol: str,
    interval: str,
    lookback_weeks: int,
    out_csv_path: str,
    max_per_call: int = 5000,
    sleep_s: float = 1.0,
) -> pd.DataFrame:
    """
    Paginates backwards in time by time windows.
    For 1m, a 5000-candle window ~= 5000 minutes.
    For 1h, 5000 hours.
    """
    end_dt = datetime.now(timezone.utc)
    start_target = end_dt - timedelta(weeks=lookback_weeks)

    all_chunks = []
    seen_first_ts = None

    # determine candle duration
    if interval.endswith("m"):
        minutes = int(interval[:-1])
        window = timedelta(minutes=minutes * (max_per_call - 1))
    elif interval.endswith("h"):
        hours = int(interval[:-1])
        window = timedelta(hours=hours * (max_per_call - 1))
    else:
        raise ValueError("interval must look like '1m', '5m', '1h', etc.")

    iter_count = 0
    while end_dt > start_target:
        iter_count += 1
        start_dt = max(start_target, end_dt - window)

        print(f"[{iter_count}] Fetch {symbol} {interval} {start_dt} -> {end_dt}")

        candles = fetch_candles_window(symbol, interval, start_dt, end_dt)
        df = candles_to_df(candles)

        if df.empty:
            print("No more data returned by API (empty). Stopping.")
            break

        # Detect “stuck pagination”: if earliest timestamp doesn't move backward, stop
        first_ts = df.index.min()
        if seen_first_ts is not None and first_ts >= seen_first_ts:
            print("Pagination appears stuck (API returning same window). Stopping to avoid infinite loop.")
            break
        seen_first_ts = first_ts

        all_chunks.append(df)

        # move end backwards with a tiny buffer to prevent overlap duplicates
        end_dt = first_ts - timedelta(seconds=1)

        time.sleep(sleep_s)

    if not all_chunks:
        return pd.DataFrame()

    full = pd.concat(all_chunks).sort_index()
    full = full[~full.index.duplicated(keep="first")]

    # cache
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    full.to_csv(out_csv_path)
    print(f"Saved {len(full)} rows to {out_csv_path} ({full.index.min()} -> {full.index.max()})")
    return full

def load_or_fetch(symbol: str, interval: str, lookback_weeks: int, data_dir: str = "centralized_data") -> pd.DataFrame:
    path = os.path.join(data_dir, f"{symbol}-{interval}-{lookback_weeks}w.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = df.set_index("timestamp").sort_index()
        for col in ["Open","High","Low","Close","Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna()
        print(f"Loaded {len(df)} rows from {path} ({df.index.min()} -> {df.index.max()})")
        return df

    return fetch_history_paginated(symbol, interval, lookback_weeks, out_csv_path=path)
