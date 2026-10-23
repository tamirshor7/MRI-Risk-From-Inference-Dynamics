import hashlib
import math

import numpy as np

from mri_risk.trajectory import project_xpdnet_primal_states, trajectory_features


def source_mask(reference_kspace, filename):
    import tensorflow as tf
    from fastmri_recon.data.utils.masking.gen_mask_tf import gen_mask_equidistant_tf

    digest = hashlib.sha256(filename.encode("utf-8")).digest()
    seed = (20260811 + int.from_bytes(digest[:4], "little")) % (2**31 - 1)
    tf.random.set_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    reference = tf.convert_to_tensor(
        np.asarray(reference_kspace, dtype=np.complex64)[None]
    )
    mask = gen_mask_equidistant_tf(
        reference, accel_factor=8, multicoil=True, mask_type="fake"
    )
    binary = np.abs(np.asarray(mask)) > 0
    lines = int(binary.sum())
    width = int(binary.shape[-1])
    if lines == 0 or not 7.5 <= width / lines <= 8.5:
        raise ValueError("Unexpected source AF8 mask")
    mask_sha = hashlib.sha256(np.ascontiguousarray(binary).tobytes()).hexdigest()
    return mask, seed, mask_sha, lines


def reconstruct_xpdnet_source(model, capture, kspace, target, mask):
    import tensorflow as tf
    from fastmri_recon.data.utils.multicoil.smap_extract import extract_smaps

    raw = np.asarray(kspace, dtype=np.complex64)
    truth = np.asarray(target, dtype=np.float32)
    if raw.ndim != 3 or truth.ndim != 2:
        raise ValueError("Expected 3D multicoil k-space and a 2D target")
    height, width = truth.shape
    if height > raw.shape[-2] or width > raw.shape[-1]:
        raise ValueError("Target exceeds native k-space geometry")

    tensor = tf.convert_to_tensor(raw[None], dtype=tf.complex64)
    smaps = extract_smaps(tensor, low_freq_percentage=4)
    masked = tf.cast(mask, tensor.dtype) * tensor
    inputs = [
        (masked * 1e6)[..., None],
        mask,
        smaps,
        tf.constant([[height, width]], dtype=tf.int32),
    ]

    capture["states"].clear()
    capture["smaps"] = None
    capture["enabled"] = True
    try:
        prediction = model(inputs, training=False)
    finally:
        capture["enabled"] = False

    if isinstance(prediction, (tuple, list)):
        prediction = prediction[-1]
    prediction = np.asarray(prediction.numpy()).squeeze()
    if np.iscomplexobj(prediction):
        prediction = np.abs(prediction)
    if prediction.shape != truth.shape:
        raise ValueError("Unexpected XPDNet source prediction shape")

    captured = [np.asarray(state.numpy()) for state in capture["states"]]
    if len(captured) != 26:
        raise ValueError(f"Expected 26 states, found {len(captured)}")

    cropped = []
    for state in captured:
        if state.ndim != 4 or state.shape[0] != 1 or state.shape[-1] != 5:
            raise ValueError(f"Unexpected primal shape: {state.shape}")
        h, w = state.shape[1:3]
        if h < height or w < width:
            raise ValueError("Primal state smaller than target")
        top, left = (h - height) // 2, (w - width) // 2
        cropped.append(state[:, top:top + height, left:left + width, :])

    images = np.stack(project_xpdnet_primal_states(cropped))
    scaled_truth = truth.astype(np.float64) * 1e6
    pred = np.asarray(prediction, dtype=np.float64)
    nmse = float(np.sum((pred - scaled_truth)**2) / np.sum(scaled_truth**2))
    endpoint_rel = float(
        np.linalg.norm(pred - images[-1]) / max(np.linalg.norm(pred), 1e-30)
    )
    if not np.isfinite(nmse) or nmse <= 0 or endpoint_rel > 5e-6:
        raise RuntimeError("Invalid source reconstruction or endpoint")

    return {
        "states": images,
        "features": trajectory_features(images),
        "nmse": nmse,
        "log_nmse": math.log(nmse),
        "endpoint_rel": endpoint_rel,
    }
