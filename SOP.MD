# KalenTheDon — ML Trading Research SOP

---

# 0) Mission

Build an institutional-grade ML trading system for Hyperliquid perpetuals that:

- Avoids lookahead bias
- Avoids label leakage
- Uses purged walk-forward validation
- Evaluates model skill before PnL
- Is reproducible and version-controlled
- Does NOT optimize based solely on backtest returns

Non-Negotiables:
- No threshold tuning based on PnL alone
- No hyperparameter tuning without out-of-sample validation
- Every change must pass a robustness test

---

# 1) Current System Snapshot (UPDATE EVERY SESSION)

Date:
Symbol:
Interval:
Lookback Weeks:
Rows Loaded:

## Label Definition
Horizon (bars):
Threshold:
Label Type:
- [ ] Up move (future_ret > threshold)
- [ ] Down move (future_ret < -threshold)

## Feature Set
- log_ret_1
- ret_1
- ema_20
- ema_50
- ema_diff
- rsi_14
- atr_14
- vol_60
- hour
- dow

## Validation Setup
Train Size:
Test Size:
Step Size:
Purge Size:
Embargo:
Calibration:
Early Stopping:

## Model
Model Type:
Key Params:
Notes:

## Strategy Mapping
Probability Column:
Thresholds:
Commission:
Use Prev Bar Signal:

---

# 2) Latest Experiment Results

## OOS Diagnostics
OOS AUC:
OOS LogLoss:
OOS AUC (inverted):
Positive Rate:

## Backtest Summary
Return:
Max Drawdown:
Sharpe:
Profit Factor:
Trades:
Exposure:
Commission Used:

---

# 3) Verified Findings (Permanent)

### Signal Inversion Discovery
- Inverted probability produced materially higher AUC.
- Indicates model predicts opposite direction of current label.
- Requires label alignment cleanup.

### Cost Sensitivity
- Doubling commission significantly altered performance.
- Edge currently fragile to friction.

---

# 4) Professional Roadmap (Do Not Reorder Without Justification)

D) Fold-by-fold stability diagnostics  
B) Extend historical depth  
A) Clean label alignment (remove inversion hack)  
C) Redesign execution mapping (ranking / sizing / regime filter)

---

# 5) Robustness Tests (Required Before Any Claim)

- Commission doubled
- Horizon ±25%
- Train window ±25%
- Different symbol test
- Regime split (bull vs bear)
- Feature importance stability
- Fold stability

---

# 6) Change Log (Append Only)

YYYY-MM-DD:
- Description of change
- Reason
- Result summary

---

# 7) How To Use This File

At the start of every research session:
1. Pull latest repo
2. Run backtest
3. Record diagnostics
4. Update Snapshot + Results
5. Commit changes

## Run Backtest Cmd
python -m src.backtests.run_ml_backtest

Never skip logging.