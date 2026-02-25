import numpy as np
import pandas as pd
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates features WITHOUT future leakage.
    """
    out = df.copy()

    close = out["Close"]
    high = out["High"]
    low = out["Low"]

    out["log_ret_1"] = np.log(close).diff()
    out["ret_1"] = close.pct_change()

    out["ema_20"] = EMAIndicator(close, window=20).ema_indicator()
    out["ema_50"] = EMAIndicator(close, window=50).ema_indicator()
    out["ema_diff"] = (out["ema_20"] - out["ema_50"]) / close

    out["rsi_14"] = RSIIndicator(close, window=14).rsi() / 100.0

    out["atr_14"] = AverageTrueRange(high, low, close, window=14).average_true_range() / close

    # volatility proxy
    out["vol_60"] = out["log_ret_1"].rolling(60).std()

    # time features (optional but helpful)
    out["hour"] = out.index.hour / 23.0
    out["dow"] = out.index.dayofweek / 6.0

    out = out.dropna()
    return out

def make_labels(df_feat: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.Series:
    """
    Binary label: 1 if future return over `horizon` bars > threshold else 0.
    Horizon is in bars (so on 1m, horizon=30 = 30 minutes).
    """
    future_ret = df_feat["Close"].shift(-horizon) / df_feat["Close"] - 1.0
    y = (future_ret < -threshold).astype(int)
    return y
