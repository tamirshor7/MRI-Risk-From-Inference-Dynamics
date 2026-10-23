import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .trajectory import FEATURE_NAMES

def partition_matched_data(source: pd.DataFrame, target: pd.DataFrame):

    if len(source) != 3676 or len(target) != 400:
        raise ValueError("Expected 3676 source and 400 M4Raw slice rows")
    for name, frame in (("source", source), ("target", target)):
        missing = (set(FEATURE_NAMES) | {"log_nmse", "patient_id"}) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing {name} columns: {sorted(missing)}")
    if "source_split" not in source.columns:
        raise ValueError("Source feature table lacks source_split")
    split = source.source_split.astype(str).str.lower()
    fit = source.loc[split == "fit"].reset_index(drop=True)
    holdout = source.loc[split == "holdout"].reset_index(drop=True)
    target = target.reset_index(drop=True)
    if len(fit) != 2944 or len(holdout) != 732:
        raise ValueError("Source split must contain 2944 fit and 732 holdout slices")
    fit_patients = set(fit.patient_id.astype(str))
    holdout_patients = set(holdout.patient_id.astype(str))
    if fit_patients & holdout_patients:
        raise ValueError("Source patient leakage")
    if len(fit_patients) != 736 or len(holdout_patients) != 183:
        raise ValueError("Source fit/holdout patient counts differ from the paper")
    for frame in (fit, holdout, target):
        if not np.isfinite(frame.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=float)).all():
            raise ValueError("Feature table contains non-finite values")
    return fit, holdout, target

def prediction_frame(frame: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    def first_column(possibilities):
        for name in possibilities:
            if name in frame.columns:
                return name
        raise ValueError(f"Missing identity column from {possibilities}")
    filename = first_column(("fname", "filename", "file"))
    sl = first_column(("slice", "slice_num", "slice_index", "slice_idx"))
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.shape != (len(frame),):
        raise ValueError("Prediction count mismatch")
    return pd.DataFrame({
        "fname": frame[filename].astype(str).to_numpy(),
        "patient_id": frame.patient_id.astype(str).to_numpy(),
        "slice": frame[sl].astype(int).to_numpy(),
        "target": frame.log_nmse.to_numpy(dtype=np.float64),
        "prediction": predictions,
    })

def scan_means(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"fname", "patient_id", "target", "prediction"}
    if not required.issubset(frame.columns):
        raise ValueError("Missing fields needed for scan aggregation")
    if (frame.groupby("fname").patient_id.nunique() != 1).any():
        raise ValueError("A scan belongs to multiple patients")
    return frame.groupby(["fname", "patient_id"], as_index=False).agg(
        target=("target", "mean"), prediction=("prediction", "mean")
    )

def risk_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    y = frame.target.to_numpy(dtype=np.float64)
    pred = frame.prediction.to_numpy(dtype=np.float64)
    if len(y) < 3 or np.std(y) == 0 or np.std(pred) == 0:
        raise ValueError("Correlation undefined for this prediction set")
    return {
        "n": len(frame),
        "pearson": float(pearsonr(y, pred).statistic),
        "spearman": float(spearmanr(y, pred).statistic),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
    }
