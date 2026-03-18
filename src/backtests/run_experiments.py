import itertools
from datetime import datetime

import pandas as pd

from src.backtests.run_ml_backtest import run

EXP_PATH = "experiments/experiment_log.csv"


def print_run_snapshot():
    try:
        df = pd.read_csv(EXP_PATH)
    except Exception as e:
        print(f"⚠️ Could not read experiment log for snapshot: {e}")
        return

    if df.empty:
        print("⚠️ Experiment log is empty.")
        return

    latest = df.sort_values("timestamp", ascending=False).iloc[0]

    auc = pd.to_numeric(latest.get("oos_auc"), errors="coerce")
    inv_auc = pd.to_numeric(latest.get("oos_auc_inverted"), errors="coerce")
    logloss = pd.to_numeric(latest.get("oos_logloss"), errors="coerce")
    trades = pd.to_numeric(latest.get("trades"), errors="coerce")
    ret = pd.to_numeric(latest.get("return_pct"), errors="coerce")
    dd = pd.to_numeric(latest.get("max_drawdown_pct"), errors="coerce")

    gap = auc - inv_auc if pd.notna(auc) and pd.notna(inv_auc) else float("nan")

    print("\n--- RUN SNAPSHOT ---")
    print(f"timestamp      : {latest.get('timestamp')}")
    print(f"source/model   : {latest.get('data_source')} / {latest.get('model')}")
    print(f"market         : {latest.get('symbol')} {latest.get('interval')}")
    print(f"horizon/thresh : {latest.get('horizon_bars')} / {latest.get('threshold')}")
    print(f"OOS AUC        : {auc:.4f}" if pd.notna(auc) else "OOS AUC        : nan")
    print(f"OOS invAUC     : {inv_auc:.4f}" if pd.notna(inv_auc) else "OOS invAUC     : nan")
    print(f"AUC gap        : {gap:.4f}" if pd.notna(gap) else "AUC gap        : nan")
    print(f"OOS LogLoss    : {logloss:.6f}" if pd.notna(logloss) else "OOS LogLoss    : nan")
    print(f"Trades         : {int(trades)}" if pd.notna(trades) else "Trades         : nan")
    print(f"Return %       : {ret:.2f}" if pd.notna(ret) else "Return %       : nan")
    print(f"Max DD %       : {dd:.2f}" if pd.notna(dd) else "Max DD %       : nan")
    print("--------------------\n")


def run_grid():
    """
    Runs a focused confirmation grid around the current promising region.
    Diagnostics first. Do NOT optimize on PnL.
    """

    symbol = "ETH"
    interval = "1h"
    data_source = "binance_futures"
    lookback_weeks = 250

    horizons = [12]
    thresholds = [0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040]
    model_types = ["lightgbm", "logreg"]

    grid = list(itertools.product(model_types, horizons, thresholds))
    total = len(grid)

    print(f"\nRunning focused confirmation grid: {total} runs\n")

    for run_idx, (model_type, horizon, threshold) in enumerate(grid, start=1):
        print("\n" + "=" * 60)
        print(f"Run {run_idx}/{total}")
        print(f"model_type={model_type}, horizon={horizon}, threshold={threshold}")
        print("=" * 60)

        try:
            run(
                symbol=symbol,
                interval=interval,
                lookback_weeks=lookback_weeks,
                data_source=data_source,
                model_type=model_type,
                override_horizon_bars=horizon,
                override_threshold=threshold,
            )
            print_run_snapshot()
        except Exception as e:
            print(f"❌ Run failed: {e}")


if __name__ == "__main__":
    start = datetime.utcnow()
    print(f"Started at {start}")

    run_grid()

    end = datetime.utcnow()
    print(f"\nFinished at {end}")
    print(f"Total duration: {end - start}")