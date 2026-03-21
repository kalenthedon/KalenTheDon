import itertools
from datetime import datetime

import pandas as pd

from src.backtests.run_ml_backtest import run

EXP_PATH = "experiments/experiment_log.csv"


def safe_num(x):
    return pd.to_numeric(x, errors="coerce")


def load_experiment_log() -> pd.DataFrame:
    try:
        df = pd.read_csv(EXP_PATH)
    except Exception as e:
        print(f"⚠️ Could not read experiment log: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    numeric_cols = [
        "horizon_bars",
        "threshold",
        "oos_auc",
        "oos_auc_inverted",
        "oos_logloss",
        "trades",
        "return_pct",
        "max_drawdown_pct",
        "fold_count",
        "fold_auc_mean",
        "fold_auc_median",
        "fold_auc_std",
        "fold_auc_min",
        "fold_auc_max",
        "fold_auc_gt_0_5",
        "fold_auc_lt_0_5",
        "fold_auc_lt_0_45",
        "bull_auc",
        "bear_auc",
        "sideways_auc",
        "bull_n",
        "bear_n",
        "sideways_n",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = safe_num(df[col])

    if "oos_auc" in df.columns and "oos_auc_inverted" in df.columns:
        df["auc_gap"] = df["oos_auc"] - df["oos_auc_inverted"]

    return df


def print_run_snapshot():
    df = load_experiment_log()
    if df.empty:
        print("⚠️ Experiment log is empty.")
        return

    latest = df.sort_values("timestamp", ascending=False).iloc[0]

    auc = latest.get("oos_auc")
    inv_auc = latest.get("oos_auc_inverted")
    logloss = latest.get("oos_logloss")
    trades = latest.get("trades")
    ret = latest.get("return_pct")
    dd = latest.get("max_drawdown_pct")
    fold_count = latest.get("fold_count")
    fold_auc_gt_0_5 = latest.get("fold_auc_gt_0_5")
    fold_auc_lt_0_5 = latest.get("fold_auc_lt_0_5")
    fold_auc_lt_0_45 = latest.get("fold_auc_lt_0_45")
    fold_auc_std = latest.get("fold_auc_std")
    gap = latest.get("auc_gap", float("nan"))

    bull_auc = latest.get("bull_auc", float("nan"))
    bear_auc = latest.get("bear_auc", float("nan"))
    sideways_auc = latest.get("sideways_auc", float("nan"))
    bull_n = latest.get("bull_n", float("nan"))
    bear_n = latest.get("bear_n", float("nan"))
    sideways_n = latest.get("sideways_n", float("nan"))

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
    print(f"Folds          : {int(fold_count)}" if pd.notna(fold_count) else "Folds          : nan")
    print(f"Fold AUC std   : {fold_auc_std:.4f}" if pd.notna(fold_auc_std) else "Fold AUC std   : nan")
    print(f"Fold AUC > 0.5 : {int(fold_auc_gt_0_5)}" if pd.notna(fold_auc_gt_0_5) else "Fold AUC > 0.5 : nan")
    print(f"Fold AUC < 0.5 : {int(fold_auc_lt_0_5)}" if pd.notna(fold_auc_lt_0_5) else "Fold AUC < 0.5 : nan")
    print(f"Fold AUC <0.45 : {int(fold_auc_lt_0_45)}" if pd.notna(fold_auc_lt_0_45) else "Fold AUC <0.45 : nan")
    print(f"Bull AUC       : {bull_auc:.4f}" if pd.notna(bull_auc) else "Bull AUC       : nan")
    print(f"Bear AUC       : {bear_auc:.4f}" if pd.notna(bear_auc) else "Bear AUC       : nan")
    print(f"Sideways AUC   : {sideways_auc:.4f}" if pd.notna(sideways_auc) else "Sideways AUC   : nan")
    print(f"Bull n         : {int(bull_n)}" if pd.notna(bull_n) else "Bull n         : nan")
    print(f"Bear n         : {int(bear_n)}" if pd.notna(bear_n) else "Bear n         : nan")
    print(f"Sideways n     : {int(sideways_n)}" if pd.notna(sideways_n) else "Sideways n     : nan")
    print("--------------------\n")


def make_research_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    x = df.copy()

    # practical trust filters
    if "fold_count" in x.columns:
        x = x[x["fold_count"].fillna(0) >= 3].copy()

    if "trades" in x.columns:
        x = x[x["trades"].fillna(0) >= 30].copy()

    if x.empty:
        return x

    # rank by robustness, not PnL
    sort_cols = [
        "oos_auc",
        "auc_gap",
        "sideways_auc",
        "oos_logloss",
        "fold_auc_std",
        "fold_auc_lt_0_45",
        "trades",
    ]
    ascending = [False, False, False, True, True, True, False]

    existing_sort_cols = [c for c in sort_cols if c in x.columns]
    existing_ascending = [ascending[i] for i, c in enumerate(sort_cols) if c in x.columns]

    x = x.sort_values(existing_sort_cols, ascending=existing_ascending)

    keep_cols = [
        "timestamp",
        "data_source",
        "model",
        "symbol",
        "interval",
        "horizon_bars",
        "threshold",
        "oos_auc",
        "oos_auc_inverted",
        "auc_gap",
        "oos_logloss",
        "fold_count",
        "fold_auc_std",
        "fold_auc_lt_0_45",
        "bull_auc",
        "bear_auc",
        "sideways_auc",
        "bull_n",
        "bear_n",
        "sideways_n",
        "trades",
        "return_pct",
        "max_drawdown_pct",
    ]

    keep_cols = [c for c in keep_cols if c in x.columns]
    return x[keep_cols].head(20)


def print_research_leaderboard(df: pd.DataFrame) -> None:
    print("\n=== RESEARCH LEADERBOARD ===")
    if df.empty:
        print("No rows pass robustness filters.")
        print("============================\n")
        return

    for _, row in df.iterrows():
        print(
            f"{row.get('timestamp')} | "
            f"h={int(row['horizon_bars'])} thr={row['threshold']} | "
            f"auc={row['oos_auc']:.4f} | "
            f"inv={row['oos_auc_inverted']:.4f} | "
            f"gap={row['auc_gap']:.4f} | "
            f"ll={row['oos_logloss']:.6f} | "
            f"sideways_auc={row['sideways_auc']:.4f} | "
            f"bull_auc={row['bull_auc']:.4f} | "
            f"trades={int(row['trades'])} | "
            f"fold_std={row['fold_auc_std']:.4f} | "
            f"ret={row['return_pct']:.2f}% | "
            f"dd={row['max_drawdown_pct']:.2f}%"
        )
    print("============================\n")


def print_label_comparison(df: pd.DataFrame) -> None:
    print("\n=== LABEL COMPARISON SUMMARY ===")
    if df.empty:
        print("No rows available.")
        print("================================\n")
        return

    required = ["horizon_bars", "threshold"]
    for col in required:
        if col not in df.columns:
            print(f"Missing required column: {col}")
            print("================================\n")
            return

    grouped = (
        df.groupby(["horizon_bars", "threshold"], dropna=False)
        .agg(
            runs=("timestamp", "count"),
            auc_mean=("oos_auc", "mean"),
            inv_mean=("oos_auc_inverted", "mean"),
            gap_mean=("auc_gap", "mean"),
            ll_mean=("oos_logloss", "mean"),
            sideways_auc_mean=("sideways_auc", "mean"),
            bull_auc_mean=("bull_auc", "mean"),
            trades_mean=("trades", "mean"),
            fold_std_mean=("fold_auc_std", "mean"),
            ret_mean=("return_pct", "mean"),
            dd_mean=("max_drawdown_pct", "mean"),
        )
        .reset_index()
        .sort_values(
            ["auc_mean", "gap_mean", "sideways_auc_mean", "ll_mean"],
            ascending=[False, False, False, True],
        )
    )

    for _, row in grouped.iterrows():
        print(
            f"h={int(row['horizon_bars'])} thr={row['threshold']} | "
            f"runs={int(row['runs'])} | "
            f"auc_mean={row['auc_mean']:.4f} | "
            f"inv_mean={row['inv_mean']:.4f} | "
            f"gap_mean={row['gap_mean']:.4f} | "
            f"ll_mean={row['ll_mean']:.6f} | "
            f"sideways_auc_mean={row['sideways_auc_mean']:.4f} | "
            f"bull_auc_mean={row['bull_auc_mean']:.4f} | "
            f"trades_mean={row['trades_mean']:.1f} | "
            f"fold_std_mean={row['fold_std_mean']:.4f} | "
            f"ret_mean={row['ret_mean']:.2f}% | "
            f"dd_mean={row['dd_mean']:.2f}%"
        )
    print("================================\n")


def run_grid():
    """
    Runs a small label comparison grid.
    Diagnostics first. Do NOT optimize on PnL.
    """

    symbol = "ETH"
    interval = "1h"
    data_source = "binance_futures"
    lookback_weeks = 250

    # current primary + secondary label candidates
    horizons = [8, 12]
    thresholds = [0.005, 0.004]
    model_types = ["logreg"]

    # explicit candidate set only
    candidate_pairs = {(8, 0.005), (12, 0.004)}
    grid = [
        (model_type, horizon, threshold)
        for model_type, horizon, threshold in itertools.product(model_types, horizons, thresholds)
        if (horizon, threshold) in candidate_pairs
    ]

    total = len(grid)
    print(f"\nRunning label comparison grid: {total} runs\n")

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

    df = load_experiment_log()
    leaderboard = make_research_leaderboard(df)
    print_research_leaderboard(leaderboard)
    print_label_comparison(df)


if __name__ == "__main__":
    start = datetime.utcnow()
    print(f"Started at {start}")

    run_grid()

    end = datetime.utcnow()
    print(f"\nFinished at {end}")
    print(f"Total duration: {end - start}")