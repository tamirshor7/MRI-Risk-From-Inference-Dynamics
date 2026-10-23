from collections.abc import Sequence

import numpy as np

EPS = 1e-12

FEATURE_NAMES = (
    "trajectory_path_length",
    "trajectory_endpoint_displacement",
    "trajectory_path_efficiency",
    "trajectory_step_mean",
    "trajectory_step_std",
    "trajectory_step_max",
    "trajectory_step_final",
    "trajectory_step_late_over_early",
    "trajectory_log_step_slope",
    "trajectory_late_log_step_slope",
    "trajectory_endpoint_distance_auc",
    "trajectory_late_endpoint_distance_auc",
    "trajectory_endpoint_monotone_fraction",
    "trajectory_consecutive_delta_cos_mean",
    "trajectory_consecutive_delta_cos_std",
    "trajectory_late_delta_cos_mean",
)

def _rms(value: np.ndarray) -> float:
    value = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(value * value)))

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denominator <= EPS else float(np.dot(a, b) / denominator)

def _slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    centered_x = x - np.mean(x)
    centered_y = y - np.mean(y)
    denominator = float(np.dot(centered_x, centered_x))
    return 0.0 if denominator <= EPS else float(np.dot(centered_x, centered_y) / denominator)

def trajectory_features(images: Sequence[np.ndarray] | np.ndarray) -> dict[str, float]:

    images = list(images)
    if not 5 <= len(images) <= 26:
        raise ValueError("A trajectory must contain between 5 and 26 states")
    shapes = [np.shape(image) for image in images]
    if any(len(shape) != 2 or shape != shapes[0] for shape in shapes):
        raise ValueError("Every trajectory state must be an aligned 2-D image")

    q = [np.asarray(np.abs(np.asarray(image)), dtype=np.float64) for image in images]
    if not all(np.all(np.isfinite(image)) for image in q):
        raise ValueError("Trajectory contains non-finite values")

    total_steps = len(q) - 1
    initial_scale = _rms(q[0]) + EPS
    deltas = [q[t] - q[t - 1] for t in range(1, total_steps + 1)]
    steps = np.asarray([_rms(delta) / initial_scale for delta in deltas], dtype=np.float64)

    path_length = float(np.sum(steps))
    displacement = _rms(q[-1] - q[0]) / initial_scale
    window = max(1, total_steps // 3)
    indices = np.arange(1, total_steps + 1, dtype=np.float64)
    log_steps = np.log(steps + EPS)

    final_scale = _rms(q[-1]) + EPS
    remaining = np.asarray(
        [_rms(q[-1] - image) / final_scale for image in q], dtype=np.float64
    )
    consecutive_cosines = np.asarray(
        [_cosine(deltas[i], deltas[i + 1]) for i in range(total_steps - 1)],
        dtype=np.float64,
    )
    cosine_window = max(1, len(consecutive_cosines) // 3)

    values = (
        path_length,
        displacement,
        float(displacement / (path_length + EPS)),
        float(np.mean(steps)),
        float(np.std(steps)),
        float(np.max(steps)),
        float(steps[-1]),
        float(np.mean(steps[-window:]) / (np.mean(steps[:window]) + EPS)),
        _slope(indices, log_steps),
        _slope(indices[-window:], log_steps[-window:]),
        float(np.mean(remaining)),
        float(np.mean(remaining[-(window + 1):])),
        float(np.mean(np.diff(remaining) <= 0)),
        float(np.mean(consecutive_cosines)),
        float(np.std(consecutive_cosines)),
        float(np.mean(consecutive_cosines[-cosine_window:])),
    )
    if not np.isfinite(values).all():
        raise ValueError("Non-finite trajectory descriptor")
    return dict(zip(FEATURE_NAMES, values, strict=True))

def project_xpdnet_primal_states(states: Sequence[np.ndarray]) -> list[np.ndarray]:

    images = []
    for state in states:
        state = np.asarray(state)
        if state.ndim != 4 or state.shape[0] != 1 or state.shape[-1] != 5:
            raise ValueError(f"Unexpected XPDNet primal-state shape: {state.shape}")
        images.append(np.asarray(np.abs(state[0, ..., 0]), dtype=np.float64))
    return images
