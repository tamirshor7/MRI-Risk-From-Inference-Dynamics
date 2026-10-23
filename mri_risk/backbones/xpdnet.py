import hashlib
from pathlib import Path

import numpy as np

from mri_risk.trajectory import project_xpdnet_primal_states, trajectory_features


def load_xpdnet(checkpoint):
    import tensorflow as tf
    from fastmri_recon.models.subclassed_models.xpdnet import XPDNet
    from fastmri_recon.models.subclassed_models.denoisers.proposed_params import get_model_specs

    tf.config.run_functions_eagerly(True)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    specifications = [
        (model_fun, kwargs, n_scales, res)
        for name, size, model_fun, kwargs, _, n_scales, res in get_model_specs(
            n_primal=5, force_res=False
        )
        if name == "MWCNN" and size == "medium"
    ]
    if len(specifications) != 1:
        raise RuntimeError("Expected one medium MWCNN specification")

    model_fun, kwargs, n_scales, res = specifications[0]
    kwargs = dict(kwargs, use_bias=False)

    capture = {"enabled": False, "states": [], "smaps": None}
    base = XPDNet.__mro__[1]
    original = base.i_domain_correction

    def pick(value):
        if tf.is_tensor(value) and value.shape.rank == 4 and value.shape[-1] == 5:
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                state = pick(item)
                if state is not None:
                    return state
        return None

    def intercept(self, i_domain, image_buffer, kspace_buffer, mask, smaps, *args):
        if capture["enabled"] and not capture["states"]:
            initial = pick(image_buffer)
            if initial is None:
                raise RuntimeError("Missing initial primal buffer")
            capture["states"].append(tf.identity(initial))
        if capture["enabled"] and capture["smaps"] is None:
            capture["smaps"] = tf.identity(smaps)
        output = original(self, i_domain, image_buffer, kspace_buffer, mask, smaps, *args)
        if capture["enabled"]:
            state = pick(output)
            if state is None:
                raise RuntimeError("Missing updated primal buffer")
            capture["states"].append(tf.identity(state))
        return output

    base.i_domain_correction = intercept
    model = XPDNet(
        model_fun, kwargs, n_primal=5, multicoil=True,
        n_scales=n_scales, refine_smaps=True, refine_big=True,
        res=res, output_shape_spec=True, n_iter=25,
    )
    model([
        tf.zeros((1, 1, 320, 320, 1), dtype=tf.complex64),
        tf.zeros((1, 1, 320, 320), dtype=tf.complex64),
        tf.zeros((1, 1, 320, 320), dtype=tf.complex64),
        tf.constant([[320, 320]], dtype=tf.int32),
    ], training=False)
    model.load_weights(str(Path(checkpoint)))
    return model, capture


def m4raw_mask(kspace, filename):
    import tensorflow as tf
    from fastmri_recon.data.utils.masking.gen_mask_tf import gen_mask_equidistant_tf

    digest = hashlib.sha256(filename.encode("utf-8")).digest()
    seed = (20260809 + int.from_bytes(digest[:4], "little")) % (2**31 - 1)
    tf.random.set_seed(seed)
    np.random.seed(seed % (2**32 - 1))

    probe = (np.asarray(kspace, dtype=np.complex64)[None] * 1e-6).astype(np.complex64)
    mask = gen_mask_equidistant_tf(
        tf.convert_to_tensor(probe, dtype=tf.complex64),
        accel_factor=8, multicoil=True, mask_type="fake",
    )
    binary = np.abs(np.asarray(mask)) > 0
    digest = hashlib.sha256(np.ascontiguousarray(binary).tobytes()).hexdigest()
    return mask, seed, digest


def reconstruct_xpdnet(model, capture, kspace, target, mask):
    import tensorflow as tf
    from fastmri_recon.data.utils.multicoil.smap_extract import extract_smaps

    raw = np.asarray(kspace, dtype=np.complex64)
    truth = np.asarray(target, dtype=np.float32).squeeze()
    if raw.ndim != 3 or truth.shape != (256, 256):
        raise ValueError("Expected multicoil k-space and 256x256 target")

    stored = tf.convert_to_tensor((raw[None] * 1e-6).astype(np.complex64))
    smaps = extract_smaps(stored, low_freq_percentage=4)
    masked = tf.cast(mask, stored.dtype) * stored

    inputs = [
        (masked * 1e6)[..., None],
        mask,
        smaps,
        tf.constant([[256, 256]], dtype=tf.int32),
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

    states = [np.asarray(state.numpy()) for state in capture["states"]]
    if len(states) != 26:
        raise RuntimeError(f"Expected 26 primal states, got {len(states)}")

    images = np.stack(project_xpdnet_primal_states(states))
    if images.shape[1:] != truth.shape or prediction.shape != truth.shape:
        raise ValueError("Reconstruction shape mismatch")

    nmse = float(
        np.sum((prediction.astype(np.float64) - truth) ** 2)
        / max(np.sum(truth.astype(np.float64) ** 2), 1e-30)
    )

    return {
        "states": images,
        "features": trajectory_features(images),
        "nmse": nmse,
        "log_nmse": float(np.log(nmse + 1e-12)),
        "endpoint_rel": float(
            np.linalg.norm(prediction - images[-1])
            / max(np.linalg.norm(prediction), 1e-30)
        ),
    }
