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
Signal Inversion Active:

---

# 2) Latest Experiment Results

## OOS Diagnostics
OOS AUC:
OOS LogLoss:
OOS AUC (inverted):
Positive Rate:

## Backtest Summary
Return:
Buy & Hold:
Return - Buy & Hold:
Max Drawdown:
Return / |DD|:
Sharpe:
Profit Factor:
Trades:
Exposure:
Commission Used:
Regime Tag:

---

# 3) Verified Findings (Permanent)

### Signal Inversion Discovery
- Global OOS inverted AUC materially exceeded raw AUC in recent tests.
- Fold-level stability is mixed, so inversion is not yet permanent truth.
- Current inversion in strategy is a temporary research device, not final production logic.

### Cost Sensitivity
- Performance meaningfully changes under higher commission.
- Current edge is still fragile to friction.

### Data Depth Constraint
- Current 1h dataset has produced too few folds.
- Must extend historical coverage before making strong conclusions.

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

## Start Enviorment
source .venv/bin/activate
python -m src.backtests.run_ml_backtest

Never skip logging.python -m py_compile src/ml/walkforward_train.py# KalenTheDon — ML Trading Research SOP

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
Signal Inversion Active:

---

# 2) Latest Experiment Results

## OOS Diagnostics
OOS AUC:
OOS LogLoss:
OOS AUC (inverted):
Positive Rate:

## Backtest Summary
Return:
Buy & Hold:
Return - Buy & Hold:
Max Drawdown:
Return / |DD|:
Sharpe:
Profit Factor:
Trades:
Exposure:
Commission Used:
Regime Tag:

---

# 3) Verified Findings (Permanent)

### Signal Inversion Discovery
- Global OOS inverted AUC materially exceeded raw AUC in recent tests.
- Fold-level stability is mixed, so inversion is not yet permanent truth.
- Current inversion in strategy is a temporary research device, not final production logic.

### Cost Sensitivity
- Performance meaningfully changes under higher commission.
- Current edge is still fragile to friction.

### Data Depth Constraint
- Current 1h dataset has produced too few folds.
- Must extend historical coverage before making strong conclusions.

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

Never skip logging.