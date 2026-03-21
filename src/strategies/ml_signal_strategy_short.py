import numpy as np
from backtesting import Strategy


class MLSignalStrategyShort(Strategy):
    """
    SHORT-only strategy aligned to a downside label.

    Current label:
      y = 1 when future return over horizon is below -threshold

    Therefore:
      high probability => downside risk => short bias
    """

    short_thr = 0.48
    exit_thr = 0.50
    use_prev_bar_signal = True

    block_bull_regime = False
    sideways_only = False

    def init(self):
        df = self.data.df
        self.prob_col = "proba_cal" if "proba_cal" in df.columns else "proba_raw"
        self.has_market_regime = "market_regime" in df.columns

    def next(self):
        df = self.data.df

        idx = -2 if self.use_prev_bar_signal else -1
        if len(df) < 2 and idx == -2:
            return

        p = float(df[self.prob_col].iloc[idx])
        if np.isnan(p):
            return

        regime = None
        if self.has_market_regime:
            regime = str(df["market_regime"].iloc[idx]).lower()

        # exits always allowed
        if self.position:
            if self.position.is_short and p < self.exit_thr:
                self.position.close()
            return

        if (self.block_bull_regime or self.sideways_only) and regime is None:
            return

        if self.sideways_only and regime != "sideways":
            return

        if self.block_bull_regime and regime == "bull":
            return

        # short-only entry: high downside probability
        if 0.48 < p < 0.62:
            self.sell()