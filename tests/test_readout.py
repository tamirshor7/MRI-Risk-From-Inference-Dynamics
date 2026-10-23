import numpy as np

from mri_risk.readout import train_source_ridge

def _data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 16))
    y = X @ rng.normal(size=16) + rng.normal(size=80) * 0.2
    patients = np.asarray([f"patient-{i // 4:03d}" for i in range(80)])
    return X, y, patients

def test_matched_readout():
    X, y, patients = _data()
    predictor, cv = train_source_ridge(X, y, patients, protocol="matched")
    assert len(cv) == 25
    assert np.isfinite(predictor.predict(X)).all()
    assert predictor.alpha in [row["alpha"] for row in cv]

def test_xpdnet_readout():
    X, y, patients = _data()
    predictor, cv = train_source_ridge(
        X, y, patients, protocol="xpdnet", xpdnet_alpha_grid=[0.001, 0.01, 0.1, 1.0]
    )
    assert len(cv) == 4
    assert predictor.protocol == "xpdnet"
    assert predictor.predict(X).shape == (80,)
