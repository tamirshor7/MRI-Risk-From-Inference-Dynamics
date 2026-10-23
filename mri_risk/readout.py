from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .trajectory import FEATURE_NAMES

MATCHED_ALPHA_GRID = np.logspace(-6, 6, 25)
XPDNET_CV_SALT = "xpdnet-source-ridge-cv-v1"

@dataclass
class SourceRidge:
    protocol: str
    alpha: float
    model: object
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def predict(self, features: np.ndarray) -> np.ndarray:
        X = np.asarray(features, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError("Input feature dimension does not match the trained model")
        if self.protocol == "matched":
            return np.asarray(self.model.predict(X), dtype=np.float64)
        return _predict_xpdnet(self.model, X, self.feature_names)

def _validated_inputs(X, y, patients):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    patients = np.asarray(patients, dtype=str)
    if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Expected (N, {len(FEATURE_NAMES)}) input features")
    if y.shape != (len(X),) or patients.shape != (len(X),):
        raise ValueError("Targets and patient IDs must be 1-D and row-aligned")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Training data contain non-finite values")
    if len(np.unique(patients)) < 5:
        raise ValueError("Five patient-disjoint CV folds are required")
    return X, y, patients

def _fit_matched(X, y, alpha):
    estimator = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=float(alpha))),
    ])
    return estimator.fit(X, y)

def select_matched_alpha(X, y, patients):

    X, y, patients = _validated_inputs(X, y, patients)
    splitter = GroupKFold(n_splits=5)
    cv = []
    for alpha in MATCHED_ALPHA_GRID:
        fold_rmse = []
        for train, valid in splitter.split(X, y, patients):
            pred = _fit_matched(X[train], y[train], alpha).predict(X[valid])
            fold_rmse.append(float(np.sqrt(np.mean((y[valid] - pred) ** 2))))
        cv.append({
            "alpha": float(alpha),
            "mean_rmse": float(np.mean(fold_rmse)),
            "fold_rmse": fold_rmse,
        })
    best = min(cv, key=lambda result: (result["mean_rmse"], result["alpha"]))
    return float(best["alpha"]), cv

def _xpdnet_patient_fold(patient: str) -> int:
    digest = hashlib.sha256((XPDNET_CV_SALT + ":" + patient).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % 5

def _fit_xpdnet(X, y, alpha, feature_names):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=0)
    active = sigma > 1e-10
    if not np.any(active):
        raise ValueError("All source features were discarded")
    names = [name for name, keep in zip(feature_names, active) if keep]
    Z = (X[:, active] - mu[active]) / sigma[active]
    target_mean = float(np.mean(y))
    gram = Z.T @ Z + float(alpha) * np.eye(Z.shape[1], dtype=np.float64)
    weights = np.linalg.solve(gram, Z.T @ (y - target_mean))
    return {
        "feature_names": names,
        "discarded_features": [name for name, keep in zip(feature_names, active) if not keep],
        "mean": mu[active], "std": sigma[active], "weights": weights,
        "target_mean": target_mean, "lambda": float(alpha),
    }

def select_xpdnet_alpha(X, y, patients, alpha_grid):

    X, y, patients = _validated_inputs(X, y, patients)
    alpha_grid = [float(value) for value in alpha_grid]
    if not alpha_grid or any(not np.isfinite(a) or a <= 0 for a in alpha_grid):
        raise ValueError("Supply the positive lambda grid from the original XPDNet protocol JSON")
    folds = np.asarray([_xpdnet_patient_fold(patient) for patient in patients])
    if set(folds.tolist()) != set(range(5)):
        raise ValueError("Not all five XPDNet hash folds are populated")
    candidates = []
    for alpha in alpha_grid:
        oof = np.full(len(y), np.nan, dtype=np.float64)
        for fold in range(5):
            train, valid = folds != fold, folds == fold
            model = _fit_xpdnet(X[train], y[train], alpha, FEATURE_NAMES)
            oof[valid] = _predict_xpdnet(model, X[valid], FEATURE_NAMES)
        if not np.isfinite(oof).all():
            raise RuntimeError("Missing XPDNet out-of-fold predictions")
        candidates.append({
            "alpha": alpha, "oof_rmse": float(np.sqrt(np.mean((y - oof) ** 2))),
        })
    selected = sorted(candidates, key=lambda x: (round(x["oof_rmse"], 12), -x["alpha"]))[0]
    return selected["alpha"], candidates

def _predict_xpdnet(model, X, names):
    index = {name: i for i, name in enumerate(names)}
    columns = [index[name] for name in model["feature_names"]]
    standardized = (X[:, columns] - model["mean"]) / model["std"]
    return model["target_mean"] + standardized @ model["weights"]

def train_source_ridge(X, y, patients, *, protocol: str, xpdnet_alpha_grid=None):

    X, y, patients = _validated_inputs(X, y, patients)
    if protocol == "matched":
        alpha, cv = select_matched_alpha(X, y, patients)
        model = _fit_matched(X, y, alpha)
    elif protocol == "xpdnet":
        if xpdnet_alpha_grid is None:
            raise ValueError("XPDNet requires its frozen source protocol lambda grid")
        alpha, cv = select_xpdnet_alpha(X, y, patients, xpdnet_alpha_grid)
        model = _fit_xpdnet(X, y, alpha, FEATURE_NAMES)
    else:
        raise ValueError(f"Unknown readout protocol: {protocol}")
    return SourceRidge(protocol, alpha, model), cv
