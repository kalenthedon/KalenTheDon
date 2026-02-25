# src/ml/walkforward_train.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss

from src.validation.purged_cv import (
    PurgedWalkForwardConfig,
    PurgedWalkForwardSplitter,
    assert_time_sorted,
)


@dataclass(frozen=True)
class WalkForwardRunConfig:
    feature_cols: List[str]
    label_col: str

    # Purged WF
    train_size: int
    test_size: int
    step_size: int
    purge_size: int
    embargo_size: int = 0

    # Early stopping (recommended for boosted trees)
    enable_early_stopping: bool = True
    early_stopping_rounds: int = 100
    early_stop_val_frac: float = 0.15  # slice off last X% of TRAIN for early stopping validation

    # Calibration (recommended for trading thresholds)
    calibrate: bool = True
    calibrator_method: str = "sigmoid"   # "sigmoid" or "isotonic"
    calibrator_val_frac: float = 0.15    # slice off last X% of TRAIN for calibration

    # Persistence
    save_models: bool = False
    model_dir: str = "models"


def _is_lightgbm_model(model: BaseEstimator) -> bool:
    name = model.__class__.__name__.lower()
    mod = (model.__class__.__module__ or "").lower()
    return ("lgbm" in name) or ("lightgbm" in mod)


def _is_xgboost_model(model: BaseEstimator) -> bool:
    name = model.__class__.__name__.lower()
    mod = (model.__class__.__module__ or "").lower()
    return ("xgb" in name) or ("xgboost" in mod)


def _split_train_for_earlystop_and_calibration(
    train_idx: np.ndarray,
    early_stop_frac: float,
    calibrator_frac: float,
    enable_early_stopping: bool,
    enable_calibration: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns:
      fit_idx: indices used to fit the base model
      es_idx: indices used for early stopping eval_set (or None)
      cal_idx: indices used to fit the calibrator (or None)

    Order is time-consistent: [ ... fit ... | ... earlystop ... | ... calibrate ... ]
    """
    if enable_early_stopping:
        if not (0.0 < early_stop_frac < 0.5):
            raise ValueError("early_stop_val_frac should be between (0, 0.5)")
    if enable_calibration:
        if not (0.0 < calibrator_frac < 0.5):
            raise ValueError("calibrator_val_frac should be between (0, 0.5)")

    total_tail = 0.0
    if enable_early_stopping:
        total_tail += early_stop_frac
    if enable_calibration:
        total_tail += calibrator_frac

    # We need enough room for fit data; keep tail slices under 50% of train
    if total_tail >= 0.5:
        raise ValueError(
            "early_stop_val_frac + calibrator_val_frac must be < 0.5 "
            "(need enough data to fit the model)."
        )

    n = len(train_idx)
    # compute slice sizes
    n_cal = int(n * calibrator_frac) if enable_calibration else 0
    n_es = int(n * early_stop_frac) if enable_early_stopping else 0

    # ensure at least 1 sample if enabled
    if enable_calibration:
        n_cal = max(1, n_cal)
    if enable_early_stopping:
        n_es = max(1, n_es)

    # If both enabled, calibration gets the very last chunk
    # early-stop chunk sits just before it
    end = n
    cal_idx = None
    es_idx = None

    if enable_calibration:
        cal_idx = train_idx[end - n_cal : end]
        end = end - n_cal

    if enable_early_stopping:
        es_idx = train_idx[end - n_es : end]
        end = end - n_es

    fit_idx = train_idx[:end]

    if len(fit_idx) < 50:
        # too small to fit anything meaningfully
        raise ValueError("Train window too small after splits (fit_idx < 50). Reduce val fracs or increase train_size.")

    return fit_idx, es_idx, cal_idx


def _fit_with_optional_early_stopping(
    model: BaseEstimator,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_es: Optional[pd.DataFrame],
    y_es: Optional[pd.Series],
    cfg: WalkForwardRunConfig,
) -> BaseEstimator:
    """
    Fits model. If model is LightGBM/XGBoost and early-stopping data is provided, uses it.
    Otherwise fits normally.
    """
    use_es = (
        cfg.enable_early_stopping
        and X_es is not None
        and y_es is not None
        and len(X_es) >= 10
        and (_is_lightgbm_model(model) or _is_xgboost_model(model))
    )

    if not use_es:
        model.fit(X_fit, y_fit)
        return model

    # Try common sklearn-style early stopping args.
    # If unsupported (TypeError), fall back to normal fit.
    try:
        model.fit(
            X_fit,
            y_fit,
            eval_set=[(X_es, y_es)],
            early_stopping_rounds=cfg.early_stopping_rounds,
            verbose=False,
        )
    except TypeError:
        # Some wrappers use callbacks or different params; safest fallback:
        model.fit(X_fit, y_fit)

    return model


def walk_forward_train_predict(
    df: pd.DataFrame,
    model_factory: Callable[[], BaseEstimator],
    cfg: WalkForwardRunConfig,
    time_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Purged walk-forward training and OOS prediction with optional:
      - Early stopping (LightGBM/XGBoost)
      - Probability calibration (Platt/Isotonic)

    Returns a DataFrame of out-of-sample predictions.
    """
    assert_time_sorted(df, time_col=time_col)

    X = df[cfg.feature_cols].astype(float)
    y = df[cfg.label_col].astype(int)

    n = len(df)
    splitter = PurgedWalkForwardSplitter(
        PurgedWalkForwardConfig(
            train_size=cfg.train_size,
            test_size=cfg.test_size,
            step_size=cfg.step_size,
            purge_size=cfg.purge_size,
            embargo_size=cfg.embargo_size,
        )
    )

    out_index = df[time_col] if time_col is not None else df.index
    preds_out: List[pd.DataFrame] = []

    os.makedirs(cfg.model_dir, exist_ok=True)

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(n)):
        # Remove NaNs within fold
        train_mask = np.isfinite(X.iloc[train_idx].to_numpy()).all(axis=1) & np.isfinite(y.iloc[train_idx].to_numpy())
        test_mask = np.isfinite(X.iloc[test_idx].to_numpy()).all(axis=1) & np.isfinite(y.iloc[test_idx].to_numpy())

        train_idx_eff = train_idx[train_mask]
        test_idx_eff = test_idx[test_mask]

        if len(train_idx_eff) < 200 or len(test_idx_eff) < 20:
            continue

        model = model_factory()

        X_test = X.iloc[test_idx_eff]
        y_test = y.iloc[test_idx_eff]

        # Split training into: fit | earlystop | calibrate (time-consistent)
        try:
            fit_idx, es_idx, cal_idx = _split_train_for_earlystop_and_calibration(
                train_idx=train_idx_eff,
                early_stop_frac=cfg.early_stop_val_frac,
                calibrator_frac=cfg.calibrator_val_frac,
                enable_early_stopping=cfg.enable_early_stopping,
                enable_calibration=cfg.calibrate,
            )
        except ValueError:
            # If splitting fails due to small windows, fall back to fitting on full train
            fit_idx = train_idx_eff
            es_idx = None
            cal_idx = None

        X_fit = X.iloc[fit_idx]
        y_fit = y.iloc[fit_idx]

        X_es = X.iloc[es_idx] if es_idx is not None else None
        y_es = y.iloc[es_idx] if es_idx is not None else None

        # Fit base model (optionally with early stopping)
        model = _fit_with_optional_early_stopping(model, X_fit, y_fit, X_es, y_es, cfg)

        # Raw probabilities
        proba_raw = model.predict_proba(X_test)[:, 1]

        if cfg.calibrate and cal_idx is not None and len(cal_idx) >= 20:
            X_cal = X.iloc[cal_idx]
            y_cal = y.iloc[cal_idx]

            calibrator = CalibratedClassifierCV(model, method=cfg.calibrator_method, cv="prefit")
            calibrator.fit(X_cal, y_cal)
            proba_cal = calibrator.predict_proba(X_test)[:, 1]

            if cfg.save_models:
                joblib.dump(
                    {"model": model, "calibrator": calibrator, "cfg": cfg, "fold_id": fold_id},
                    os.path.join(cfg.model_dir, f"fold_{fold_id:03d}.joblib"),
                )

            fold_df = pd.DataFrame(
                {
                    "fold_id": fold_id,
                    "proba_raw": proba_raw,
                    "proba_cal": proba_cal,
                    "y_true": y_test.to_numpy(),
                },
                index=out_index[test_idx_eff],
            )
        else:
            if cfg.save_models:
                joblib.dump(
                    {"model": model, "cfg": cfg, "fold_id": fold_id},
                    os.path.join(cfg.model_dir, f"fold_{fold_id:03d}.joblib"),
                )

            fold_df = pd.DataFrame(
                {
                    "fold_id": fold_id,
                    "proba_raw": proba_raw,
                    "y_true": y_test.to_numpy(),
                },
                index=out_index[test_idx_eff],
            )

        preds_out.append(fold_df)

    if not preds_out:
        raise RuntimeError("No folds produced predictions. Check window sizes / NaNs / label availability.")

    pred_df = pd.concat(preds_out).sort_index()

    # --- Global OOS diagnostics ---
    try:
        from sklearn.metrics import roc_auc_score, log_loss

        proba_col = "proba_cal" if "proba_cal" in pred_df.columns else "proba_raw"

        auc = roc_auc_score(pred_df["y_true"], pred_df[proba_col])
        ll = log_loss(pred_df["y_true"], pred_df[proba_col])

        print("\n=== OOS Classification Diagnostics ===")
        print(f"OOS AUC: {auc:.4f}")
        print(f"OOS LogLoss: {ll:.6f}")
        inv_auc = roc_auc_score(pred_df["y_true"], 1.0 - pred_df[proba_col])
        print(f"OOS AUC (inverted probs): {inv_auc:.4f}")
    except Exception as e:
        print(f"AUC calculation skipped: {e}")
        
    return pred_df