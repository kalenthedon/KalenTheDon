import os
import csv
import subprocess
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from datetime import datetime

import pandas as pd
from backtesting import Backtest
from lightgbm import LGBMClassifier

from src.data.hyperliquid import load_or_fetch as load_or_fetch_hyperliquid
from src.data.binance_futures import load_or_fetch as load_or_fetch_binance_futures
from src.ml.features import make_features, make_labels
from src.ml.walkforward_train import WalkForwardRunConfig, walk_forward_train_predict
from src.strategies.ml_signal_strategy import MLSignalStrategy
from src.strategies.ml_signal_strategy_short import MLSignalStrategyShort


def get_git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def classify_regime(buy_hold_return_pct: float) -> str:
    if buy_hold_return_pct >= 10:
        return "bull"
    if buy_hold_return_pct <= -10:
        return "bear"
    return "sideways"

def classify_market_regime(df: pd.DataFrame, lookback_bars: int = 24, bull_thr: float = 0.02, bear_thr: float = -0.02) -> pd.DataFrame:
    x = df.copy()

    if "Close" not in x.columns:
        raise ValueError("classify_market_regime expected 'Close' column")

    x["market_ret_lb"] = x["Close"].pct_change(lookback_bars)

    x["market_regime"] = "sideways"
    x.loc[x["market_ret_lb"] >= bull_thr, "market_regime"] = "bull"
    x.loc[x["market_ret_lb"] <= bear_thr, "market_regime"] = "bear"
    x.loc[x["market_ret_lb"].isna(), "market_regime"] = "unknown"

    return x


def summarize_market_regime_signal(merged: pd.DataFrame) -> pd.DataFrame:
    if merged.empty:
        return pd.DataFrame(columns=[
            "market_regime", "n", "pos_rate", "auc", "auc_inverted", "auc_gap", "logloss"
        ])

    proba_col = None
    for c in merged.columns:
        if str(c).startswith("proba_"):
            proba_col = c
            break

    if proba_col is None:
        raise ValueError("No probability column found in merged dataframe")

    required = ["y", "market_regime", proba_col]
    missing = [c for c in required if c not in merged.columns]
    if missing:
        raise ValueError(f"Missing columns for regime signal summary: {missing}")

    rows = []

    for regime in ["bull", "bear", "sideways", "unknown"]:
        g = merged.loc[merged["market_regime"] == regime, ["y", proba_col]].dropna().copy()

        n = int(len(g))
        pos_rate = float(g["y"].mean()) if n > 0 else float("nan")

        auc = float("nan")
        auc_inverted = float("nan")
        ll = float("nan")

        if n >= 2 and g["y"].nunique() >= 2:
            auc = float(roc_auc_score(g["y"], g[proba_col]))
            auc_inverted = float(roc_auc_score(g["y"], 1.0 - g[proba_col]))
            ll = float(log_loss(g["y"], g[proba_col], labels=[0, 1]))

        rows.append({
            "market_regime": regime,
            "n": n,
            "pos_rate": pos_rate,
            "auc": auc,
            "auc_inverted": auc_inverted,
            "auc_gap": auc - auc_inverted if pd.notna(auc) and pd.notna(auc_inverted) else float("nan"),
            "logloss": ll,
        })

    return pd.DataFrame(rows)


def print_market_regime_signal_summary(regime_df: pd.DataFrame) -> None:
    print("\n=== MARKET REGIME SIGNAL SUMMARY ===")

    if regime_df.empty:
        print("No regime summary rows.")
        print("===================================\n")
        return

    for _, row in regime_df.iterrows():
        auc = row["auc"]
        inv_auc = row["auc_inverted"]
        gap = row["auc_gap"]
        ll = row["logloss"]
        pos_rate = row["pos_rate"]

        print(
            f"{row['market_regime']:>8} | "
            f"n={int(row['n']):>5} | "
            f"pos_rate={pos_rate:.4f} | "
            f"auc={auc:.4f} | "
            f"invAUC={inv_auc:.4f} | "
            f"gap={gap:.4f} | "
            f"logloss={ll:.6f}"
        )

    print("===================================\n")

def print_prediction_regime_trade_diagnostic(merged: pd.DataFrame) -> None:
    print("\n=== PREDICTION REGIME TRADE DIAGNOSTIC ===")

    proba_col = None
    for c in merged.columns:
        if str(c).startswith("proba_"):
            proba_col = c
            break

    if proba_col is None:
        print("No probability column found.")
        print("=========================================\n")
        return

    required = ["market_regime", "y", proba_col]
    missing = [c for c in required if c not in merged.columns]
    if missing:
        print(f"Missing columns: {missing}")
        print("=========================================\n")
        return

    x = merged[required].dropna().copy()

    for regime in ["bull", "bear", "sideways", "unknown"]:
        g = x[x["market_regime"] == regime].copy()
        if g.empty:
            print(f"{regime:>8} | no rows")
            continue

        avg_p = float(g[proba_col].mean())
        hit_rate = float(g["y"].mean())
        p90 = float(g[proba_col].quantile(0.90))
        p95 = float(g[proba_col].quantile(0.95))
        p99 = float(g[proba_col].quantile(0.99))

        print(
            f"{regime:>8} | "
            f"rows={len(g):>5} | "
            f"avg_proba={avg_p:.4f} | "
            f"label_rate={hit_rate:.4f} | "
            f"p90={p90:.4f} | "
            f"p95={p95:.4f} | "
            f"p99={p99:.4f}"
        )

    print("=========================================\n")

def log_market_regime_diags(
    regime_df: pd.DataFrame,
    run_id: str,
    git_commit: str,
    data_source: str,
    symbol: str,
    interval: str,
    log_path: str = "experiments/market_regime_diagnostics.csv",
) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    fieldnames = [
        "run_id",
        "timestamp",
        "git_commit",
        "data_source",
        "symbol",
        "interval",
        "market_regime",
        "n",
        "pos_rate",
        "auc",
        "auc_inverted",
        "auc_gap",
        "logloss",
    ]

    if regime_df.empty:
        return

    out = regime_df.copy()
    out["run_id"] = run_id
    out["timestamp"] = datetime.utcnow().isoformat()
    out["git_commit"] = git_commit
    out["data_source"] = data_source
    out["symbol"] = symbol
    out["interval"] = interval
    out = out[fieldnames]

    if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
        try:
            existing = pd.read_csv(log_path, nrows=1)
            if list(existing.columns) != fieldnames:
                print("⚠️ Market regime schema mismatch -> resetting regime log")
                os.remove(log_path)
        except Exception:
            print("⚠️ Corrupt market regime CSV -> resetting")
            os.remove(log_path)

    file_exists = os.path.isfile(log_path) and os.path.getsize(log_path) > 0

    out.to_csv(
        log_path,
        mode="a",
        header=not file_exists,
        index=False,
    )

def print_run_snapshot(config: dict, metrics: dict) -> None:
    auc = metrics.get("oos_auc", float("nan"))
    inv_auc = metrics.get("oos_auc_inverted", float("nan"))
    logloss = metrics.get("oos_logloss", float("nan"))
    trades = metrics.get("trades", float("nan"))
    ret = metrics.get("return_pct", float("nan"))
    dd = metrics.get("max_drawdown_pct", float("nan"))

    gap = float("nan")
    if pd.notna(auc) and pd.notna(inv_auc):
        gap = auc - inv_auc

    print("\n--- RUN SNAPSHOT ---")
    print(f"source/model   : {config.get('data_source')} / {config.get('model')}")
    print(f"market         : {config.get('symbol')} {config.get('interval')}")
    print(f"lookback_weeks : {config.get('lookback_weeks')}")
    print(f"horizon/thresh : {config.get('horizon_bars')} / {config.get('threshold')}")
    print(f"OOS AUC        : {auc:.4f}" if pd.notna(auc) else "OOS AUC        : nan")
    print(f"OOS invAUC     : {inv_auc:.4f}" if pd.notna(inv_auc) else "OOS invAUC     : nan")
    print(f"AUC gap        : {gap:.4f}" if pd.notna(gap) else "AUC gap        : nan")
    print(f"OOS LogLoss    : {logloss:.6f}" if pd.notna(logloss) else "OOS LogLoss    : nan")
    print(f"Trades         : {int(trades)}" if pd.notna(trades) else "Trades         : nan")
    print(f"Return %       : {ret:.2f}" if pd.notna(ret) else "Return %       : nan")
    print(f"Max DD %       : {dd:.2f}" if pd.notna(dd) else "Max DD %       : nan")
    print("--------------------\n")

def summarize_bt_stats(name: str, strategy_name: str, stats) -> dict:
    return {
        "variant": name,
        "strategy_name": strategy_name,
        "return_pct": float(stats["Return [%]"]),
        "max_drawdown_pct": float(stats["Max. Drawdown [%]"]),
        "sharpe": float(stats["Sharpe Ratio"]),
        "profit_factor": float(stats["Profit Factor"]),
        "trades": int(stats["# Trades"]),
        "exposure_pct": float(stats["Exposure Time [%]"]),
        "buy_hold_return_pct": float(stats["Buy & Hold Return [%]"]),
    }

def print_strategy_ablation_summary(rows: list) -> None:
    print("\n=== STRATEGY ABLATION SUMMARY ===")
    if not rows:
        print("No ablation rows.")
        print("================================\n")
        return

    for r in rows:
        print(
            f"{r['variant']:<28} | "
            f"strategy={r['strategy_name']:<20} | "
            f"ret={r['return_pct']:>8.2f}% | "
            f"dd={r['max_drawdown_pct']:>8.2f}% | "
            f"sharpe={r['sharpe']:>7.3f} | "
            f"pf={r['profit_factor']:>6.3f} | "
            f"trades={r['trades']:>4d} | "
            f"exposure={r['exposure_pct']:>7.2f}%"
        )
    print("================================\n")

def make_sideways_probability_buckets(
    merged: pd.DataFrame,
    n_buckets: int = 10,
) -> pd.DataFrame:
    proba_col = None
    for c in merged.columns:
        if str(c).startswith("proba_"):
            proba_col = c
            break

    if proba_col is None:
        raise ValueError("No probability column found for sideways probability buckets")

    required = ["market_regime", "y", "Close", proba_col]
    missing = [c for c in required if c not in merged.columns]
    if missing:
        raise ValueError(f"Missing columns for sideways probability buckets: {missing}")

    x = merged.loc[merged["market_regime"] == "sideways", ["y", "Close", proba_col]].copy()
    x = x.dropna().copy()

    if x.empty:
        return pd.DataFrame()

    # realized forward return aligned with current label horizon
    x["fwd_ret"] = x["Close"].shift(-1) / x["Close"] - 1.0

    # rank probabilities into quantile buckets
    x["bucket"] = pd.qcut(
        x[proba_col],
        q=n_buckets,
        labels=False,
        duplicates="drop",
    )

    out = (
        x.groupby("bucket")
        .agg(
            n=("y", "size"),
            avg_proba=(proba_col, "mean"),
            pos_rate=("y", "mean"),
            mean_1bar_ret=("fwd_ret", "mean"),
        )
        .reset_index()
        .sort_values("bucket")
    )

    out["bucket"] = out["bucket"].astype(int)
    return out


def print_sideways_probability_buckets(bucket_df: pd.DataFrame) -> None:
    print("\n=== SIDEWAYS PROBABILITY BUCKETS ===")
    if bucket_df.empty:
        print("No sideways bucket rows.")
        print("===================================\n")
        return

    for _, row in bucket_df.iterrows():
        print(
            f"bucket={int(row['bucket']):>2d} | "
            f"n={int(row['n']):>5d} | "
            f"avg_proba={row['avg_proba']:.4f} | "
            f"pos_rate={row['pos_rate']:.4f} | "
            f"mean_1bar_ret={row['mean_1bar_ret']:.6f}"
        )
    print("===================================\n")       

def summarize_fold_diags(fold_diags: list) -> dict:
    if not fold_diags:
        return {
            "fold_count": 0,
            "fold_auc_mean": float("nan"),
            "fold_auc_median": float("nan"),
            "fold_auc_std": float("nan"),
            "fold_auc_min": float("nan"),
            "fold_auc_max": float("nan"),
            "fold_auc_gt_0_5": 0,
            "fold_auc_lt_0_5": 0,
            "fold_auc_lt_0_45": 0,
            "fold_invauc_mean": float("nan"),
            "fold_invauc_median": float("nan"),
            "fold_invauc_std": float("nan"),
            "fold_invauc_min": float("nan"),
            "fold_invauc_max": float("nan"),
            "fold_logloss_mean": float("nan"),
            "fold_logloss_median": float("nan"),
        }

    x = pd.DataFrame(fold_diags).copy()
    x["auc"] = pd.to_numeric(x["auc"], errors="coerce")
    x["auc_inverted"] = pd.to_numeric(x["auc_inverted"], errors="coerce")
    x["logloss"] = pd.to_numeric(x["logloss"], errors="coerce")

    return {
        "fold_count": int(len(x)),
        "fold_auc_mean": float(x["auc"].mean()),
        "fold_auc_median": float(x["auc"].median()),
        "fold_auc_std": float(x["auc"].std()),
        "fold_auc_min": float(x["auc"].min()),
        "fold_auc_max": float(x["auc"].max()),
        "fold_auc_gt_0_5": int((x["auc"] > 0.5).sum()),
        "fold_auc_lt_0_5": int((x["auc"] < 0.5).sum()),
        "fold_auc_lt_0_45": int((x["auc"] < 0.45).sum()),
        "fold_invauc_mean": float(x["auc_inverted"].mean()),
        "fold_invauc_median": float(x["auc_inverted"].median()),
        "fold_invauc_std": float(x["auc_inverted"].std()),
        "fold_invauc_min": float(x["auc_inverted"].min()),
        "fold_invauc_max": float(x["auc_inverted"].max()),
        "fold_logloss_mean": float(x["logloss"].mean()),
        "fold_logloss_median": float(x["logloss"].median()),
    }
def print_fold_summary(fold_summary: dict) -> None:
    print("\n=== FOLD STABILITY SUMMARY ===")
    print(f"Folds           : {fold_summary.get('fold_count')}")
    print(f"AUC mean        : {fold_summary.get('fold_auc_mean', float('nan')):.4f}")
    print(f"AUC median      : {fold_summary.get('fold_auc_median', float('nan')):.4f}")
    print(f"AUC std         : {fold_summary.get('fold_auc_std', float('nan')):.4f}")
    print(f"AUC min/max     : {fold_summary.get('fold_auc_min', float('nan')):.4f} / {fold_summary.get('fold_auc_max', float('nan')):.4f}")
    print(f"AUC > 0.5       : {fold_summary.get('fold_auc_gt_0_5')}")
    print(f"AUC < 0.5       : {fold_summary.get('fold_auc_lt_0_5')}")
    print(f"AUC < 0.45      : {fold_summary.get('fold_auc_lt_0_45')}")
    print(f"invAUC mean     : {fold_summary.get('fold_invauc_mean', float('nan')):.4f}")
    print(f"invAUC median   : {fold_summary.get('fold_invauc_median', float('nan')):.4f}")
    print(f"invAUC std      : {fold_summary.get('fold_invauc_std', float('nan')):.4f}")
    print(f"LogLoss mean    : {fold_summary.get('fold_logloss_mean', float('nan')):.6f}")
    print(f"LogLoss median  : {fold_summary.get('fold_logloss_median', float('nan')):.6f}")
    print("==============================\n")

def interval_to_pandas_freq(interval: str) -> str:
    unit = interval[-1]
    value = int(interval[:-1])

    if unit == "m":
        return f"{value}min"
    if unit == "h":
        return f"{value}h"
    if unit == "d":
        return f"{value}D"

    raise ValueError(f"Unsupported interval for timestamp validation: {interval}")


def validate_time_index(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if df.empty:
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Expected DatetimeIndex on loaded OHLCV data.")

    if not df.index.is_monotonic_increasing:
        print("Index not sorted -> sorting by timestamp")
        df = df.sort_index(kind="stable")

    duplicate_mask = df.index.duplicated(keep=False)
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count > 0:
        dupes = df.index[duplicate_mask]
        sample_dupes = list(dict.fromkeys(str(ts) for ts in dupes[:10]))
        raise ValueError(
            f"Duplicate timestamps detected: {duplicate_count} rows across "
            f"{df.index.nunique()} unique timestamps. Sample: {sample_dupes}"
        )

    expected_freq = interval_to_pandas_freq(interval)
    expected_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=expected_freq)
    missing_index = expected_index.difference(df.index)
    missing_count = int(len(missing_index))

    if missing_count > 0:
        sample_missing = [str(ts) for ts in missing_index[:10]]
        raise ValueError(
            f"Missing timestamps detected: {missing_count}. Sample: {sample_missing}"
        )

    print(
        f"Timestamp integrity OK -> rows={len(df)}, duplicates=0, missing=0, "
        f"range=({df.index.min()} -> {df.index.max()}), freq={expected_freq}"
    )
    return df


def load_dataset(data_source: str, symbol: str, interval: str, lookback_weeks: int) -> pd.DataFrame:
    if data_source == "hyperliquid":
        return load_or_fetch_hyperliquid(symbol, interval, lookback_weeks)

    if data_source == "binance_futures":
        return load_or_fetch_binance_futures(symbol, interval, lookback_weeks)

    raise ValueError(
        f"Unsupported data_source={data_source}. "
        f"Expected one of: ['hyperliquid', 'binance_futures']"
    )


def log_experiment(config: dict, metrics: dict, log_path: str = "experiments/experiment_log.csv") -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    fieldnames = [
        "timestamp",
        "git_commit",
        "data_source",
        "symbol",
        "interval",
        "lookback_weeks",
        "model",
        "commission",
        "horizon_bars",
        "threshold",
        "train_size",
        "test_size",
        "step_size",
        "purge_size",
        "embargo_size",
        "calibrate",
        "calibrator_method",
        "calibrator_val_frac",
        "rows_backtest",
        "oos_auc",
        "oos_auc_inverted",
        "oos_logloss",
        "fold_count",
        "fold_auc_mean",
        "fold_auc_median",
        "fold_auc_std",
        "fold_auc_min",
        "fold_auc_max",
        "fold_auc_gt_0_5",
        "fold_auc_lt_0_5",
        "fold_auc_lt_0_45",
        "fold_invauc_mean",
        "fold_invauc_median",
        "fold_invauc_std",
        "fold_invauc_min",
        "fold_invauc_max",
        "fold_logloss_mean",
        "fold_logloss_median",
        "buy_hold_return_pct",
        "return_over_bh",
        "score_ret_dd",
        "regime_tag",
        "return_pct",
        "max_drawdown_pct",
        "sharpe",
        "profit_factor",
        "trades",
        "exposure_pct",
        "commissions_paid",
        "equity_final",
        "bull_auc",
        "bear_auc",
        "sideways_auc",
        "bull_n",
        "bear_n",
        "sideways_n",
        "ablation_no_gate_return_pct",
        "ablation_no_gate_dd_pct",
        "ablation_no_gate_sharpe",
        "ablation_no_gate_pf",
        "ablation_no_gate_trades",
        "ablation_no_gate_exposure_pct",

        "ablation_block_bull_return_pct",
        "ablation_block_bull_dd_pct",
        "ablation_block_bull_sharpe",
        "ablation_block_bull_pf",
        "ablation_block_bull_trades",
        "ablation_block_bull_exposure_pct",

        "ablation_sideways_only_return_pct",
        "ablation_sideways_only_dd_pct",
        "ablation_sideways_only_sharpe",
        "ablation_sideways_only_pf",
        "ablation_sideways_only_trades",
        "ablation_sideways_only_exposure_pct",
        
    ]

    row = {k: "" for k in fieldnames}
    row.update({
        "timestamp": datetime.utcnow().isoformat(),
        **config,
        **metrics,
    })

    if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
        try:
            existing = pd.read_csv(log_path)
            if list(existing.columns) != fieldnames:
                print("⚠️ Schema mismatch detected -> resetting experiment log")
                os.remove(log_path)
        except Exception:
            print("⚠️ Corrupt CSV detected -> resetting experiment log")
            os.remove(log_path)

    file_exists = os.path.isfile(log_path) and os.path.getsize(log_path) > 0

    with open(log_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def log_fold_diags(fold_diags: list, run_id: str, log_path: str = "experiments/fold_diagnostics.csv") -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    fieldnames = [
        "run_id",
        "timestamp",
        "git_commit",
        "data_source",
        "symbol",
        "interval",
        "fold_id",
        "n",
        "pos_rate",
        "proba_col",
        "auc",
        "auc_inverted",
        "logloss",
        "start",
        "end",
    ]

    rows = []

    for r in fold_diags:
        row = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "git_commit": r.get("git_commit", ""),
            "data_source": r.get("data_source", ""),
            "symbol": r.get("symbol", ""),
            "interval": r.get("interval", ""),
            "fold_id": r.get("fold_id"),
            "n": r.get("n"),
            "pos_rate": r.get("pos_rate"),
            "proba_col": r.get("proba_col"),
            "auc": r.get("auc"),
            "auc_inverted": r.get("auc_inverted"),
            "logloss": r.get("logloss"),
            "start": r.get("start"),
            "end": r.get("end"),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    # 🔒 enforce strict column order (prevents corruption)
    df = df[fieldnames]

    # 🔒 drop any fully empty rows (safety)
    df = df.dropna(how="all")

    # 🔒 reset file if schema mismatch
    if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
        try:
            existing = pd.read_csv(log_path, nrows=1)
            if list(existing.columns) != fieldnames:
                print("⚠️ Fold schema mismatch -> resetting fold log")
                os.remove(log_path)
        except Exception:
            print("⚠️ Corrupt fold CSV -> resetting")
            os.remove(log_path)

    file_exists = os.path.isfile(log_path) and os.path.getsize(log_path) > 0

    df.to_csv(
        log_path,
        mode="a",
        header=not file_exists,
        index=False,
    )


def make_model() -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=5,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

def make_logreg_model() -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        random_state=42,
    )
    

def get_model_factory(model_type: str):
    if model_type == "lightgbm":
        return make_model

    if model_type == "logreg":
        return make_logreg_model

    raise ValueError(
        f"Unsupported model_type={model_type}. "
        f"Expected one of: ['lightgbm', 'logreg']"
    )

def make_sideways_threshold_diagnostic(
    merged: pd.DataFrame,
    horizon_bars: int,
    thresholds: list = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = [0.55, 0.60, 0.65, 0.70]

    proba_col = None
    for c in merged.columns:
        if str(c).startswith("proba_"):
            proba_col = c
            break

    if proba_col is None:
        raise ValueError("No probability column found for sideways threshold diagnostic")

    required = ["market_regime", "y", "Close", proba_col]
    missing = [c for c in required if c not in merged.columns]
    if missing:
        raise ValueError(f"Missing columns for sideways threshold diagnostic: {missing}")

    x = merged.loc[merged["market_regime"] == "sideways", ["Close", "y", proba_col]].copy()
    x = x.dropna().copy()

    if x.empty:
        return pd.DataFrame()

    # realized forward return aligned to the current label horizon
    x["fwd_ret_h"] = x["Close"].shift(-horizon_bars) / x["Close"] - 1.0
    x = x.dropna().copy()

    rows = []

    for thr in thresholds:
        g = x.loc[x[proba_col] > thr].copy()
        n = int(len(g))

        if n == 0:
            rows.append({
                "threshold": thr,
                "n": 0,
                "avg_proba": float("nan"),
                "pos_rate": float("nan"),
                "mean_fwd_ret_h": float("nan"),
                "median_fwd_ret_h": float("nan"),
                "hit_rate_ret_gt_0": float("nan"),
                "hit_rate_ret_gt_label": float("nan"),
            })
            continue

        rows.append({
            "threshold": thr,
            "n": n,
            "avg_proba": float(g[proba_col].mean()),
            "pos_rate": float(g["y"].mean()),
            "mean_fwd_ret_h": float(g["fwd_ret_h"].mean()),
            "median_fwd_ret_h": float(g["fwd_ret_h"].median()),
            "hit_rate_ret_gt_0": float((g["fwd_ret_h"] > 0).mean()),
            "hit_rate_ret_gt_label": float((g["fwd_ret_h"] >= 0.005).mean()),
        })

    return pd.DataFrame(rows)


def print_sideways_threshold_diagnostic(threshold_df: pd.DataFrame) -> None:
    print("\n=== SIDEWAYS THRESHOLD DIAGNOSTIC ===")

    if threshold_df.empty:
        print("No sideways threshold rows.")
        print("====================================\n")
        return

    for _, row in threshold_df.iterrows():
        n = int(row["n"]) if pd.notna(row["n"]) else 0

        if n == 0:
            print(f"thr={row['threshold']:.2f} | n=0")
            continue

        print(
            f"thr={row['threshold']:.2f} | "
            f"n={n:>5d} | "
            f"avg_proba={row['avg_proba']:.4f} | "
            f"pos_rate={row['pos_rate']:.4f} | "
            f"mean_fwd_ret_h={row['mean_fwd_ret_h']:.6f} | "
            f"median_fwd_ret_h={row['median_fwd_ret_h']:.6f} | "
            f"hit_ret>0={row['hit_rate_ret_gt_0']:.4f} | "
            f"hit_ret>label={row['hit_rate_ret_gt_label']:.4f}"
        )

    print("====================================\n")

def label_return_diagnostic(merged: pd.DataFrame, horizon_bars: int):
    print("\n=== LABEL RETURN DIAGNOSTIC ===")

    if "Close" not in merged.columns or "y" not in merged.columns:
        print("Missing required columns.")
        print("================================\n")
        return

    x = merged[["Close", "y"]].copy()
    x["fwd_ret_h"] = x["Close"].shift(-horizon_bars) / x["Close"] - 1.0
    x = x.dropna()

    pos = x[x["y"] == 1]
    neg = x[x["y"] == 0]

    print(f"Positive label mean return : {pos['fwd_ret_h'].mean():.6f}")
    print(f"Negative label mean return : {neg['fwd_ret_h'].mean():.6f}")
    print(f"Positive median return     : {pos['fwd_ret_h'].median():.6f}")
    print(f"Negative median return     : {neg['fwd_ret_h'].median():.6f}")

    print("================================\n")

def run(
    symbol: str = "ETH",
    interval: str = "1h",
    lookback_weeks: int = 250,
    data_source: str = "binance_futures",
    model_type: str = "logreg",
    override_horizon_bars: int = None,
    override_threshold: float = None,
) -> None:
    df = load_dataset(data_source, symbol, interval, lookback_weeks)
    if df.empty:
        print("No data loaded.")
        return

    actual_start = df.index.min()
    actual_end = df.index.max()
    actual_span = actual_end - actual_start
    actual_weeks = actual_span / pd.Timedelta(weeks=1)
    coverage_ratio = actual_span / pd.Timedelta(weeks=lookback_weeks)

    print(
        f"Coverage check -> source={data_source}, requested={lookback_weeks}w, "
        f"actual_start={actual_start}, actual_end={actual_end}, "
        f"coverage_ratio={coverage_ratio:.3f}"
    )
    print(
        f"Actual loaded span -> {actual_weeks:.2f} weeks "
        f"({actual_start} -> {actual_end})"
    )

    if coverage_ratio < 0.95:
        raise RuntimeError(
            f"Insufficient history for requested lookback_weeks={lookback_weeks}. "
            f"Only {coverage_ratio:.1%} of requested span is available."
        )

    df = validate_time_index(df, interval)

    # =========================
    # 🔹 Defaults
    # =========================
    horizon_bars = 8
    train_size = 50_000
    test_size = 10_000
    step_size = 10_000

    if interval.endswith("h"):
        # practical hourly defaults that actually produce folds
        train_size = 24 * 7 * 20   # 3360
        test_size = 24 * 7 * 4     # 672
        step_size = 24 * 7 * 4     # 672

    if override_horizon_bars is not None:
        horizon_bars = override_horizon_bars

    # =========================
    # 🔹 Label config
    # =========================
    label_threshold = 0.005
    if override_threshold is not None:
        label_threshold = override_threshold

    commission = 0.001

    # =========================
    # 🔹 Features + labels
    # =========================
    feat_df = make_features(df)
    feat_df["y"] = make_labels(
        feat_df,
        horizon=horizon_bars,
        threshold=label_threshold,
    )
    feat_df = feat_df.dropna()

    ohlcv_cols = {"Open", "High", "Low", "Close", "Volume"}
    feature_cols = [c for c in feat_df.columns if c not in ohlcv_cols and c != "y"]
    
    # =========================
    # 🔹 Walk-forward config
    # =========================
    cfg = WalkForwardRunConfig(
        feature_cols=feature_cols,
        label_col="y",
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        purge_size=horizon_bars,
        embargo_size=0,
        enable_early_stopping=True,
        early_stopping_rounds=100,
        early_stop_val_frac=0.15,
        calibrate=True,
        calibrator_method="sigmoid",
        calibrator_val_frac=0.15,
        save_models=False,
    )

    # =========================
    # 🔹 Train
    # =========================
    model_factory = get_model_factory(model_type)
    print(f"Model selected -> {model_type}")

    pred_df, diag = walk_forward_train_predict(
        feat_df,
        model_factory,
        cfg,
        time_col=None,
    )

    oos_auc = float(diag.get("oos_auc", float("nan")))
    oos_auc_inverted = float(diag.get("oos_auc_inverted", float("nan")))
    oos_logloss = float(diag.get("oos_logloss", float("nan")))
    fold_diags = diag.get("fold_diags", [])
    fold_summary = summarize_fold_diags(fold_diags)
    print_fold_summary(fold_summary)

    merged = feat_df.join(pred_df, how="inner")

    print(f"Merged rows for backtest: {len(merged)} ({merged.index.min()} -> {merged.index.max()})")

    # diagnostics-only regime tagging happens AFTER OOS predictions are produced
    merged = classify_market_regime(merged, lookback_bars=24, bull_thr=0.02, bear_thr=-0.02)

    regime_signal_df = summarize_market_regime_signal(merged)
    regime_lookup = (
        regime_signal_df.set_index("market_regime")[["auc", "n"]].to_dict("index")
        if not regime_signal_df.empty
        else {}
    )
    print_market_regime_signal_summary(regime_signal_df)

    print_prediction_regime_trade_diagnostic(merged)

    sideways_bucket_df = make_sideways_probability_buckets(merged, n_buckets=10)
    print_sideways_probability_buckets(sideways_bucket_df)

    sideways_threshold_df = make_sideways_threshold_diagnostic(
        merged,
        horizon_bars=horizon_bars,
    )
    print_sideways_threshold_diagnostic(sideways_threshold_df)

    label_return_diagnostic(merged, horizon_bars)
    
    # =========================
    # 🔹 Backtest ablation
    # =========================
    ablation_runs = [
    {
        "name": "short_only_no_gate",
        "strategy_cls": MLSignalStrategyShort,
        "strategy_name": "MLSignalStrategyShort",
        "params": {
            "block_bull_regime": False,
            "sideways_only": False,
        },
    },
    {
        "name": "short_only_block_bull",
        "strategy_cls": MLSignalStrategyShort,
        "strategy_name": "MLSignalStrategyShort",
        "params": {
            "block_bull_regime": True,
            "sideways_only": False,
        },
    },
    {
        "name": "short_only_sideways_only",
        "strategy_cls": MLSignalStrategyShort,
        "strategy_name": "MLSignalStrategyShort",
        "params": {
            "block_bull_regime": False,
            "sideways_only": True,
        },
    },
]
    ablation_rows = []
    ablation_stats = {}

    for run_cfg in ablation_runs:
        name = run_cfg["name"]
        strategy_cls = run_cfg["strategy_cls"]
        strategy_name = run_cfg["strategy_name"]
        params = run_cfg["params"]

        print(f"\nRunning strategy variant -> {name}")

        bt = Backtest(
            merged,
            strategy_cls,
            cash=100_000,
            commission=commission,
            exclusive_orders=True,
            finalize_trades=True,
        )

        stats_variant = bt.run(**params)
        ablation_stats[name] = {
            "stats": stats_variant,
            "strategy_cls": strategy_cls,
            "strategy_name": strategy_name,
            "params": params,
        }
        ablation_rows.append(summarize_bt_stats(name, strategy_name, stats_variant))

        print(stats_variant)

    print_strategy_ablation_summary(ablation_rows)

    # choose one canonical variant for logging + plotting
    # current preferred diagnostic: long_only_block_bull
    selected_variant = "long_only_block_bull"
    selected = ablation_stats[selected_variant]
    stats = selected["stats"]

    print(f"\nSelected variant for logging/plotting -> {selected_variant}")

    bt = Backtest(
        merged,
        selected["strategy_cls"],
        cash=100_000,
        commission=commission,
        exclusive_orders=True,
        finalize_trades=True,
    )
    bt.run(**selected["params"])
    bt.plot()

    # =========================
    # 🔹 Metrics
    # =========================
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    git_commit = get_git_commit_hash()

    return_pct = float(stats["Return [%]"])
    max_dd_pct = float(stats["Max. Drawdown [%]"])
    buy_hold_return_pct = float(stats["Buy & Hold Return [%]"])

    config = {
        "git_commit": git_commit,
        "data_source": data_source,
        "symbol": symbol,
        "interval": interval,
        "lookback_weeks": lookback_weeks,
        "model": model_type,
        "commission": commission,
        "horizon_bars": horizon_bars,
        "threshold": label_threshold,
        "train_size": train_size,
        "test_size": test_size,
        "step_size": step_size,
        "purge_size": cfg.purge_size,
        "embargo_size": cfg.embargo_size,
        "calibrate": cfg.calibrate,
        "calibrator_method": cfg.calibrator_method,
        "calibrator_val_frac": cfg.calibrator_val_frac,
    }

    ablation_lookup = {r["variant"]: r for r in ablation_rows}

    metrics = {
        "rows_backtest": int(len(merged)),
        "oos_auc": oos_auc,
        "oos_auc_inverted": oos_auc_inverted,
        "oos_logloss": oos_logloss,
        "buy_hold_return_pct": buy_hold_return_pct,
        "return_over_bh": return_pct - buy_hold_return_pct,
        "score_ret_dd": return_pct / abs(max_dd_pct) if max_dd_pct != 0 else float("nan"),
        "regime_tag": classify_regime(buy_hold_return_pct),
        "return_pct": return_pct,
        "max_drawdown_pct": max_dd_pct,
        "sharpe": float(stats["Sharpe Ratio"]),
        "profit_factor": float(stats["Profit Factor"]),
        "trades": int(stats["# Trades"]),
        "exposure_pct": float(stats["Exposure Time [%]"]),
        "commissions_paid": float(stats["Commissions [$]"]),
        "equity_final": float(stats["Equity Final [$]"]),
        "fold_count": fold_summary["fold_count"],
        "fold_auc_mean": fold_summary["fold_auc_mean"],
        "fold_auc_median": fold_summary["fold_auc_median"],
        "fold_auc_std": fold_summary["fold_auc_std"],
        "fold_auc_min": fold_summary["fold_auc_min"],
        "fold_auc_max": fold_summary["fold_auc_max"],
        "fold_auc_gt_0_5": fold_summary["fold_auc_gt_0_5"],
        "fold_auc_lt_0_5": fold_summary["fold_auc_lt_0_5"],
        "fold_auc_lt_0_45": fold_summary["fold_auc_lt_0_45"],
        "fold_invauc_mean": fold_summary["fold_invauc_mean"],
        "fold_invauc_median": fold_summary["fold_invauc_median"],
        "fold_invauc_std": fold_summary["fold_invauc_std"],
        "fold_invauc_min": fold_summary["fold_invauc_min"],
        "fold_invauc_max": fold_summary["fold_invauc_max"],
        "fold_logloss_mean": fold_summary["fold_logloss_mean"],
        "fold_logloss_median": fold_summary["fold_logloss_median"],
        "bull_auc": float(regime_lookup.get("bull", {}).get("auc", float("nan"))),
        "bear_auc": float(regime_lookup.get("bear", {}).get("auc", float("nan"))),
        "sideways_auc": float(regime_lookup.get("sideways", {}).get("auc", float("nan"))),
        "bull_n": float(regime_lookup.get("bull", {}).get("n", float("nan"))),
        "bear_n": float(regime_lookup.get("bear", {}).get("n", float("nan"))),
        "sideways_n": float(regime_lookup.get("sideways", {}).get("n", float("nan"))),
        "ablation_no_gate_return_pct": float(ablation_lookup.get("long_only_no_gate", {}).get("return_pct", float("nan"))),
        "ablation_no_gate_dd_pct": float(ablation_lookup.get("long_only_no_gate", {}).get("max_drawdown_pct", float("nan"))),
        "ablation_no_gate_sharpe": float(ablation_lookup.get("long_only_no_gate", {}).get("sharpe", float("nan"))),
        "ablation_no_gate_pf": float(ablation_lookup.get("long_only_no_gate", {}).get("profit_factor", float("nan"))),
        "ablation_no_gate_trades": float(ablation_lookup.get("long_only_no_gate", {}).get("trades", float("nan"))),
        "ablation_no_gate_exposure_pct": float(ablation_lookup.get("long_only_no_gate", {}).get("exposure_pct", float("nan"))),

        "ablation_block_bull_return_pct": float(ablation_lookup.get("long_only_block_bull", {}).get("return_pct", float("nan"))),
        "ablation_block_bull_dd_pct": float(ablation_lookup.get("long_only_block_bull", {}).get("max_drawdown_pct", float("nan"))),
        "ablation_block_bull_sharpe": float(ablation_lookup.get("long_only_block_bull", {}).get("sharpe", float("nan"))),
        "ablation_block_bull_pf": float(ablation_lookup.get("long_only_block_bull", {}).get("profit_factor", float("nan"))),
        "ablation_block_bull_trades": float(ablation_lookup.get("long_only_block_bull", {}).get("trades", float("nan"))),
        "ablation_block_bull_exposure_pct": float(ablation_lookup.get("long_only_block_bull", {}).get("exposure_pct", float("nan"))),

        "ablation_sideways_only_return_pct": float(ablation_lookup.get("long_only_sideways_only", {}).get("return_pct", float("nan"))),
        "ablation_sideways_only_dd_pct": float(ablation_lookup.get("long_only_sideways_only", {}).get("max_drawdown_pct", float("nan"))),
        "ablation_sideways_only_sharpe": float(ablation_lookup.get("long_only_sideways_only", {}).get("sharpe", float("nan"))),
        "ablation_sideways_only_pf": float(ablation_lookup.get("long_only_sideways_only", {}).get("profit_factor", float("nan"))),
        "ablation_sideways_only_trades": float(ablation_lookup.get("long_only_sideways_only", {}).get("trades", float("nan"))),
        "ablation_sideways_only_exposure_pct": float(ablation_lookup.get("long_only_sideways_only", {}).get("exposure_pct", float("nan"))),
    }

    if int(metrics["trades"]) < 30:
        print(f"⚠️ Skipping experiment log: insufficient trades ({int(metrics['trades'])})")
        return
    
        # =========================
    # 🔹 Logging
    # =========================
    if int(metrics["trades"]) < 30:
        print(f"⚠️ Skipping experiment log: insufficient trades ({int(metrics['trades'])})")
    else:
        log_experiment(config, metrics)
        print("Experiment logged -> experiments/experiment_log.csv")

    for r in fold_diags:
        r["git_commit"] = git_commit
        r["data_source"] = data_source
        r["symbol"] = symbol
        r["interval"] = interval

    log_fold_diags(fold_diags, run_id=run_id)
    print("Fold diagnostics logged -> experiments/fold_diagnostics.csv")

    log_market_regime_diags(
        regime_signal_df,
        run_id=run_id,
        git_commit=git_commit,
        data_source=data_source,
        symbol=symbol,
        interval=interval,
    )
    print("Market regime diagnostics logged -> experiments/market_regime_diagnostics.csv")

    # =========================
    # 🔹 Snapshot (KEY OUTPUT)
    # =========================
    print_run_snapshot(config, metrics)

if __name__ == "__main__":
    run()