import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

BINANCE_FAPI_URL = "https://fapi.binance.com"


def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _interval_to_timedelta(interval: str) -> timedelta:
    if interval.endswith("m"):
        return timedelta(minutes=int(interval[:-1]))
    if interval.endswith("h"):
        return timedelta(hours=int(interval[:-1]))
    if interval.endswith("d"):
        return timedelta(days=int(interval[:-1]))
    if interval.endswith("w"):
        return timedelta(weeks=int(interval[:-1]))
    raise ValueError(f"Unsupported interval: {interval}")


def _floor_dt_to_interval(dt: datetime, interval: str) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)

    if interval.endswith("m"):
        minutes = int(interval[:-1])
        floored_minute = (dt.minute // minutes) * minutes
        return dt.replace(minute=floored_minute, second=0, microsecond=0)

    if interval.endswith("h"):
        hours = int(interval[:-1])
        floored_hour = (dt.hour // hours) * hours
        return dt.replace(hour=floored_hour, minute=0, second=0, microsecond=0)

    if interval.endswith("d"):
        days = int(interval[:-1])
        floored_day = ((dt.day - 1) // days) * days + 1
        return dt.replace(day=floored_day, hour=0, minute=0, second=0, microsecond=0)

    if interval.endswith("w"):
        weeks = int(interval[:-1])
        # floor to Monday 00:00 UTC, then align by week block
        base = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        monday = base - timedelta(days=base.weekday())
        epoch_monday = datetime(1970, 1, 5, tzinfo=timezone.utc)
        weeks_since_epoch = (monday - epoch_monday).days // 7
        floored_weeks = (weeks_since_epoch // weeks) * weeks
        return epoch_monday + timedelta(weeks=floored_weeks)

    raise ValueError(f"Unsupported interval: {interval}")


def _binance_symbol(symbol: str) -> str:
    symbol = symbol.upper()
    if symbol.endswith("USDT"):
        return symbol
    return f"{symbol}USDT"


def fetch_exchange_info(session: Optional[requests.Session] = None) -> dict:
    client = session or requests
    r = client.get(f"{BINANCE_FAPI_URL}/fapi/v1/exchangeInfo", timeout=30)
    r.raise_for_status()
    return r.json()


def validate_symbol_exists(symbol: str, session: Optional[requests.Session] = None) -> None:
    exchange_info = fetch_exchange_info(session=session)
    available = {s["symbol"] for s in exchange_info.get("symbols", [])}
    if symbol not in available:
        raise ValueError(
            f"Binance USD-M futures symbol not found: {symbol}. "
            f"Sample available symbols: {sorted(list(available))[:20]}"
        )


def fetch_klines_window(
    symbol: str,
    interval: str,
    start_dt: datetime,
    end_dt: datetime,
    limit: int = 1500,
    session: Optional[requests.Session] = None,
) -> list:
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": _to_ms(start_dt),
        "endTime": _to_ms(end_dt),
        "limit": limit,
    }

    client = session or requests
    r = client.get(f"{BINANCE_FAPI_URL}/fapi/v1/klines", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if not isinstance(data, list):
        raise ValueError(f"Unexpected Binance klines response: {data}")

    return data


def klines_to_df(klines: list) -> pd.DataFrame:
    if not klines:
        return pd.DataFrame()

    rows = []
    for k in klines:
        rows.append(
            {
                "timestamp": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
                "Open": k[1],
                "High": k[2],
                "Low": k[3],
                "Close": k[4],
                "Volume": k[5],
            }
        )

    df = pd.DataFrame(rows).set_index("timestamp").sort_index()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    df = df[~df.index.duplicated(keep="first")]
    return df


def _is_cache_sufficient(df: pd.DataFrame, lookback_weeks: int, interval: str) -> bool:
    if df.empty:
        return False

    bar_delta = _interval_to_timedelta(interval)
    now_floor = _floor_dt_to_interval(datetime.now(timezone.utc), interval)
    target_start = now_floor - timedelta(weeks=lookback_weeks)

    actual_start = df.index.min()
    actual_end = df.index.max()

    covers_start = actual_start <= (target_start + bar_delta)
    recent_enough_end = actual_end >= (now_floor - 2 * bar_delta)

    return covers_start and recent_enough_end


def fetch_history_paginated(
    symbol: str,
    interval: str,
    lookback_weeks: int,
    out_csv_path: str,
    max_per_call: int = 1500,
    sleep_s: float = 0.20,
) -> pd.DataFrame:
    if max_per_call > 1500:
        raise ValueError("Binance USD-M futures klines limit cannot exceed 1500.")

    symbol = _binance_symbol(symbol)
    bar_delta = _interval_to_timedelta(interval)
    end_dt = _floor_dt_to_interval(datetime.now(timezone.utc), interval)
    start_target = end_dt - timedelta(weeks=lookback_weeks)
    window = bar_delta * (max_per_call - 1)

    all_chunks = []
    seen_earliest_ts = None

    session = requests.Session()
    validate_symbol_exists(symbol, session=session)

    iter_count = 0
    while end_dt >= start_target:
        iter_count += 1
        start_dt = max(start_target, end_dt - window)

        print(f"[{iter_count}] Fetch Binance {symbol} {interval} {start_dt} -> {end_dt}")

        klines = fetch_klines_window(
            symbol=symbol,
            interval=interval,
            start_dt=start_dt,
            end_dt=end_dt,
            limit=max_per_call,
            session=session,
        )
        df = klines_to_df(klines)

        if df.empty:
            print("No more data returned by Binance (empty). Stopping.")
            break

        first_ts = df.index.min()
        last_ts = df.index.max()

        if seen_earliest_ts is not None and first_ts >= seen_earliest_ts:
            raise RuntimeError(
                f"Pagination stuck: earliest timestamp did not move backward. "
                f"prev={seen_earliest_ts}, current={first_ts}"
            )

        expected_max_rows = int((end_dt - start_dt) / bar_delta) + 1
        print(
            f"    returned rows={len(df)} range=({first_ts} -> {last_ts}) "
            f"expected_max~{expected_max_rows}"
        )

        all_chunks.append(df)
        seen_earliest_ts = first_ts
        end_dt = first_ts - bar_delta

        if end_dt < start_target:
            break

        time.sleep(sleep_s)

    session.close()

    if not all_chunks:
        return pd.DataFrame()

    full = pd.concat(all_chunks).sort_index()
    full = full[~full.index.duplicated(keep="first")]

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    full.to_csv(out_csv_path)
    print(f"Saved {len(full)} rows to {out_csv_path} ({full.index.min()} -> {full.index.max()})")
    return full


def load_or_fetch(
    symbol: str,
    interval: str,
    lookback_weeks: int,
    data_dir: str = "centralized_data",
) -> pd.DataFrame:
    normalized_symbol = _binance_symbol(symbol)
    path = os.path.join(data_dir, f"binance-{normalized_symbol}-{interval}-{lookback_weeks}w.csv")

    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = df.set_index("timestamp").sort_index()

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna()
        df = df[~df.index.duplicated(keep="first")]

        print(f"Loaded {len(df)} rows from {path} ({df.index.min()} -> {df.index.max()})")

        if _is_cache_sufficient(df, lookback_weeks, interval):
            print("Cache coverage OK.")
            return df

        print("Cache coverage insufficient for requested lookback -> refetching.")
        os.remove(path)

    return fetch_history_paginated(
        symbol=symbol,
        interval=interval,
        lookback_weeks=lookback_weeks,
        out_csv_path=path,
    )