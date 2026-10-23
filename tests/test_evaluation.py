import numpy as np
import pandas as pd
import pytest

from mri_risk.evaluation import partition_matched_data, prediction_frame, risk_metrics, scan_means
from mri_risk.trajectory import FEATURE_NAMES


def _frames():
    rng = np.random.default_rng(17)
    n = 3676
    source = pd.DataFrame(rng.normal(size=(n, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    source["source_split"] = ["fit"] * 2944 + ["holdout"] * 732
    source["patient_id"] = [f"p{i // 4}" for i in range(n)]
    source["fname"] = [f"f{i // 4}" for i in range(n)]
    source["slice_num"] = np.tile(np.arange(4), n // 4)
    source["log_nmse"] = rng.normal(size=n)
    target = pd.DataFrame(rng.normal(size=(400, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    target["patient_id"] = [f"t{i // 16}" for i in range(400)]
    target["fname"] = [f"m{i // 4}" for i in range(400)]
    target["slice_num"] = np.tile(np.arange(4), 100)
    target["log_nmse"] = rng.normal(size=400)
    return source, target


def test_matched_partition_and_aggregation():
    source, target = _frames()
    fit, holdout, target = partition_matched_data(source, target)
    assert (len(fit), len(holdout), len(target)) == (2944, 732, 400)
    predicted = target.log_nmse.to_numpy() + np.linspace(-0.5, 0.5, 400)
    slices = prediction_frame(target, predicted)
    scans = scan_means(slices)
    assert len(scans) == 100
    assert scans.patient_id.nunique() == 25
    assert np.isfinite(list(risk_metrics(scans).values())).all()
    np.testing.assert_allclose(scans.target.to_numpy(), target.groupby(["fname", "patient_id"]).log_nmse.mean().to_numpy())


def test_patient_leakage_is_rejected():
    source, target = _frames()
    source.loc[2944, "patient_id"] = source.loc[0, "patient_id"]
    with pytest.raises(ValueError, match="leakage"):
        partition_matched_data(source, target)


def test_missing_features_are_rejected():
    source, target = _frames()
    with pytest.raises(ValueError, match="Missing"):
        partition_matched_data(source.drop(columns=[FEATURE_NAMES[0]]), target)
