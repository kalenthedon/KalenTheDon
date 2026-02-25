from backtesting import Backtest


from src.data.hyperliquid import load_or_fetch
from src.ml.features import make_features, make_labels
from src.ml.walkforward_train import WalkForwardRunConfig, walk_forward_train_predict
from src.strategies.ml_signal_strategy import MLSignalStrategy


def make_model():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        objective="binary",
        n_estimators=2000,          # use early stopping in future; for now keep high
        learning_rate=0.03,
        num_leaves=31,              # keep small to reduce overfit
        max_depth=5,                # cap depth
        min_child_samples=100,      # IMPORTANT: raise for 1m data (prevents memorizing noise)
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
    )


def run(symbol="ETH", interval="1h", lookback_weeks=100):
    df = load_or_fetch(symbol, interval, lookback_weeks)
    if df.empty:
        print("No data loaded.")
        return

    # --- Choose horizon in BARS (this also becomes purge_size) ---
    if interval.endswith("h"):
        horizon_bars = 24   # 24 hours ahead on 1h bars
        train_size = 24 * 7 * 20  # 20 weeks
        test_size  = 24 * 7 * 4   # 4 weeks
        step_size  = 24 * 7 * 4   # 4 weeks
    else:
        horizon_bars = 60   # 60 minutes ahead on 1m bars
        train_size = 50_000
        test_size  = 10_000
        step_size  = 10_000

    label_threshold = 0.001

    # --- Build features + labels ---
    feat_df = make_features(df)
    feat_df["y"] = make_labels(feat_df, horizon=horizon_bars, threshold=label_threshold)

    # Important: drop the tail rows that have NaN labels due to shift(-horizon)
    feat_df = feat_df.dropna()

    # Define feature columns (everything we engineered; exclude OHLCV + y)
    ohlcv_cols = {"Open", "High", "Low", "Close", "Volume"}
    feature_cols = [c for c in feat_df.columns if c not in ohlcv_cols and c != "y"]

    # --- Purged walk-forward config ---
    cfg = WalkForwardRunConfig(
    feature_cols=feature_cols,
    label_col="y",
    train_size=train_size,
    test_size=test_size,
    step_size=step_size,
    purge_size=horizon_bars,

    enable_early_stopping=True,
    early_stopping_rounds=100,
    early_stop_val_frac=0.15,

    calibrate=True,
    calibrator_method="sigmoid",
    calibrator_val_frac=0.15,
)

    # --- Walk-forward train + OOS predictions ---
    pred_df = walk_forward_train_predict(feat_df, make_model, cfg, time_col=None)

    # Merge predictions into data for backtesting.py
    merged = feat_df.join(pred_df, how="inner")

    print(f"Merged rows for backtest: {len(merged)} ({merged.index.min()} -> {merged.index.max()})")
    print("Pred columns:", list(pred_df.columns))

    bt = Backtest(
    merged,
    MLSignalStrategy,
    cash=100_000,
    commission=0.001,
    exclusive_orders=True,
    finalize_trades=True,
)

    stats = bt.run()
    print(stats)
    bt.plot()


if __name__ == "__main__":
    run()
