import numpy as np
import pandas as pd
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


# =========================
# 🔧 FEATURE TOGGLES
# =========================
USE_MULTI_RETURNS = False
USE_ZSCORE = False
USE_VOL = False
USE_TREND = True
USE_INTERACTIONS = False


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Controlled feature builder for ablation testing.
    Toggle features above to test impact.
    """
    out = df.copy()

    close = out["Close"]
    high = out["High"]
    low = out["Low"]

    # =========================
    # 🔹 BASELINE (DO NOT TOUCH)
    # =========================
    out["log_ret_1"] = np.log(close).diff()
    out["ret_1"] = close.pct_change()

    out["ema_20"] = EMAIndicator(close, window=20).ema_indicator()
    out["ema_50"] = EMAIndicator(close, window=50).ema_indicator()
    out["ema_diff"] = (out["ema_20"] - out["ema_50"]) / close

    out["rsi_14"] = RSIIndicator(close, window=14).rsi() / 100.0

    out["atr_14"] = AverageTrueRange(high, low, close, window=14).average_true_range() / close

    out["vol_60"] = out["log_ret_1"].rolling(60).std()

    # =========================
    # 🔹 OPTIONAL FEATURES
    # =========================

    if USE_MULTI_RETURNS:
        out["ret_3"] = close.pct_change(3)
        out["ret_6"] = close.pct_change(6)
        out["ret_12"] = close.pct_change(12)
        out["ret_accel"] = out["ret_3"] - out["ret_12"]

    if USE_TREND:
        out["ema_200"] = EMAIndicator(close, window=200).ema_indicator()
        out["trend"] = (out["ema_50"] > out["ema_200"]).astype(int)
        out["price_vs_ema50"] = (close - out["ema_50"]) / close

    if USE_VOL:
        out["vol_24"] = out["log_ret_1"].rolling(24).std()
        out["vol_168"] = out["log_ret_1"].rolling(168).std()
        out["vol_ratio"] = out["vol_24"] / out["vol_168"]

    if USE_ZSCORE:
        mean_50 = close.rolling(50).mean()
        std_50 = close.rolling(50).std()
        out["zscore_50"] = (close - mean_50) / std_50

    if USE_INTERACTIONS:
        if USE_MULTI_RETURNS and USE_VOL:
            out["mom_x_vol"] = out["ret_12"] * out["vol_24"]
        if USE_TREND:
            out["trend_x_rsi"] = out.get("trend", 0) * out["rsi_14"]
        if USE_ZSCORE and USE_VOL:
            out["zscore_x_vol"] = out["zscore_50"] * out["vol_24"]

    # =========================
    # 🔹 TIME FEATURES
    # =========================
    out["hour"] = out.index.hour / 23.0
    out["dow"] = out.index.dayofweek / 6.0

    out = out.dropna()
    return out


def make_labels(df_feat: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.Series:
    future_ret = df_feat["Close"].shift(-horizon) / df_feat["Close"] - 1.0
    y = (future_ret < -threshold).astype(int)
    return y