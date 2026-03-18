import numpy as np
from backtesting import Strategy


class MLSignalStrategy(Strategy):
    """
    Uses out-of-sample predicted probabilities:
      - prefers 'proba_cal'
      - falls back to 'proba_raw'

    Current research mode:
    - Uses previous bar signal for realism
    - Inverts probability because global diagnostics showed inverted skill
      (this should eventually be replaced by label alignment once data depth is improved)
    """

    buy_thr = 0.55
    sell_thr = 0.45
    use_prev_bar_signal = True

    def init(self):
        df = self.data.df
        self.prob_col = "proba_cal" if "proba_cal" in df.columns else "proba_raw"

    def next(self):
        df = self.data.df

        idx = -2 if self.use_prev_bar_signal else -1
        if len(df) < 2 and idx == -2:
            return

        p = float(df[self.prob_col].iloc[idx])

        if np.isnan(p):
            return

        # Current research hack: invert probability because global AUC inversion tested better
        p = 1.0 - p

        if not self.position:
            if p > self.buy_thr:
                self.buy()
            elif p < self.sell_thr:
                self.sell()
        else:
            if self.position.is_long and p < 0.50:
                self.position.close()
            elif self.position.is_short and p > 0.50:
                self.position.close()