import numpy as np
import pytest

from mri_risk.trajectory import FEATURE_NAMES, trajectory_features, project_xpdnet_primal_states

def test_feature_schema_and_lengths():
    rng = np.random.default_rng(7)
    for length in (5, 13, 26):
        q = np.abs(rng.normal(size=(length, 8, 9)).astype(np.float32))
        features = trajectory_features(q)
        assert tuple(features) == FEATURE_NAMES
        assert np.isfinite(list(features.values())).all()

def test_constant_trajectory():
    q = np.ones((13, 5, 6), dtype=np.float32)
    features = trajectory_features(q)
    assert features["trajectory_path_length"] == 0.0
    assert features["trajectory_endpoint_displacement"] == 0.0
    assert features["trajectory_consecutive_delta_cos_mean"] == 0.0

def test_xpdnet_projector():
    q = np.arange(5 * 5 * 6, dtype=np.float32).reshape(5, 5, 6)
    native = np.zeros((5, 1, 5, 6, 5), dtype=np.float32)
    native[:, 0, ..., 0] = q
    projected = project_xpdnet_primal_states(native)
    np.testing.assert_array_equal(projected, q)

def test_shape_checks():
    with pytest.raises(ValueError):
        trajectory_features(np.ones((3, 6, 6)))
    with pytest.raises(ValueError):
        trajectory_features([np.ones((8, 8)), np.ones((9, 9))] * 3)

def test_frozen_golden_vectors():
    import json
    from pathlib import Path

    golden = json.loads((Path(__file__).parent / 'golden_trajectory_features.json').read_text())
    for length in (5, 13, 26):
        base = np.arange(7 * 9, dtype=np.float32).reshape(7, 9) / 15
        images = [
            (base + np.float32(.07 * t) * np.sin(
                np.arange(7 * 9, dtype=np.float32).reshape(7, 9) + t * .25
            )).astype(np.float32)
            for t in range(length)
        ]
        actual = trajectory_features(images)
        np.testing.assert_allclose(
            [actual[name] for name in FEATURE_NAMES],
            [golden[str(length)][name] for name in FEATURE_NAMES],
            atol=1e-13, rtol=0,
        )
