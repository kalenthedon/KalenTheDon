import os
import csv
import subprocess
from sklearn.linear_model import LogisticRegression
from datetime import datetime

import pandas as pd
from backtesting import Backtest
from lightgbm import LGBMClassifier

from src.data.hyperliquid import load_or_fetch as load_or_fetch_hyperliquid
from src.data.binance_futures import load_or_fetch as load_or_fetch_binance_futures
from src.ml.features import make_features, make_labels
from src.ml.walkforward_train import WalkForwardRunConfig, walk_forward_train_predict
from src.strategies.ml_signal_strategy import MLSignalStrategy


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
        max_iter=1000,
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
    horizon_bars = 60
    train_size = 50_000
    test_size = 10_000
    step_size = 10_000

    if interval.endswith("h"):
        horizon_bars = 24

    if override_horizon_bars is not None:
        horizon_bars = override_horizon_bars
        train_size = 24 * 7 * 20
        test_size = 24 * 7 * 4
        step_size = 24 * 7 * 4

    # =========================
    # 🔹 Label config
    # =========================
    label_threshold = 0.001
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

    merged = feat_df.join(pred_df, how="inner")

    print(f"Merged rows for backtest: {len(merged)} ({merged.index.min()} -> {merged.index.max()})")

    # =========================
    # 🔹 Backtest
    # =========================
    bt = Backtest(
        merged,
        MLSignalStrategy,
        cash=100_000,
        commission=commission,
        exclusive_orders=True,
        finalize_trades=True,
    )

    stats = bt.run()
    print(stats)
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
    }

    # =========================
    # 🔹 Logging
    # =========================
    log_experiment(config, metrics)
    print("Experiment logged -> experiments/experiment_log.csv")

    for r in fold_diags:
        r["git_commit"] = git_commit
        r["data_source"] = data_source
        r["symbol"] = symbol
        r["interval"] = interval

    log_fold_diags(fold_diags, run_id=run_id)
    print("Fold diagnostics logged -> experiments/fold_diagnostics.csv")

    # =========================
    # 🔹 Snapshot (KEY OUTPUT)
    # =========================
    print_run_snapshot(config, metrics)