import math
from pathlib import Path

import numpy as np

from mri_risk.trajectory import trajectory_features


def load_promptmr(checkpoint, device):
    import torch
    from models import PromptMR

    saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    h = saved["hyper_parameters"]

    names = (
        "num_cascades", "num_adj_slices", "n_feat0", "feature_dim",
        "prompt_dim", "sens_n_feat0", "sens_feature_dim",
        "sens_prompt_dim", "len_prompt", "prompt_size", "n_enc_cab",
        "n_dec_cab", "n_skip_cab", "n_bottleneck_cab", "no_use_ca",
        "learnable_prompt", "adaptive_input", "n_buffer", "n_history",
        "use_sens_adj",
    )

    if h["model_version"] != "promptmr_v2":
        raise ValueError("Unexpected PromptMR version")
    if int(h["num_cascades"]) != 12 or int(h["num_adj_slices"]) != 5:
        raise ValueError("Unexpected PromptMR architecture")

    model = PromptMR(**{name: h[name] for name in names}).to(device)

    for cascade in model.cascades:
        value = int(cascade.model.n_buffer)
        if value != int(h["n_buffer"]):
            raise ValueError("PromptMR n_buffer mismatch")
        cascade.n_buffer = value

    state = {
        name[len("promptmr."):]: value
        for name, value in saved["state_dict"].items()
        if name.startswith("promptmr.")
    }
    model.load_state_dict(state, strict=True)

    return model.eval()


def adjacent_slices(center, count):
    return [min(max(center + offset, 0), count - 1) for offset in range(-2, 3)]


def _crop(image, height, width):
    h, w = image.shape[-2:]
    if h < height or w < width:
        raise ValueError("Cannot crop reconstruction to target")
    top = (h - height) // 2
    left = (w - width) // 2
    return image[..., top:top + height, left:left + width]


def reconstruct_promptmr(model, kspace_stack, target, mask_array, device):
    import torch
    import fastmri
    from models.promptmr_v2 import sens_reduce, sens_expand

    kspace = torch.view_as_real(
        torch.from_numpy(np.asarray(kspace_stack, dtype=np.complex64))
    )

    if kspace.shape[-3] > 384:
        image = fastmri.ifft2c(kspace)
        top = (image.shape[-3] - 384) // 2
        kspace = fastmri.fft2c(image[..., top:top + 384, :, :])

    kspace = kspace.unsqueeze(0).to(device)
    mask = torch.from_numpy(np.asarray(mask_array, dtype=bool)).to(
        device=device, dtype=torch.bool
    ).unsqueeze(-1)

    if int(mask.shape[-2]) != int(kspace.shape[-2]):
        raise ValueError("Mask and k-space width differ")

    masked = kspace * mask
    num_low = int(round(int(kspace.shape[-2]) * 0.04))
    low_tensor = torch.tensor([num_low], device=device, dtype=torch.long)

    with torch.inference_mode():
        sensitivity = model.sens_net(
            masked, mask, low_tensor, ("cartesian",), False
        )
        zero_image = sens_reduce(masked, sensitivity, 5)
        prediction = zero_image.clone()
        latent = zero_image.clone()
        history = None

        center_kspace = torch.chunk(masked, 5, dim=1)[2]
        initialization = fastmri.rss(
            fastmri.complex_abs(fastmri.ifft2c(center_kspace)), dim=1
        )
        states = [initialization[0].detach().cpu().numpy()]

        for cascade in model.cascades:
            prediction, latent, history = cascade(
                prediction, zero_image, latent, mask, sensitivity, history
            )

            center_image = torch.chunk(prediction, 5, dim=1)[2]
            center_sensitivity = torch.chunk(sensitivity, 5, dim=1)[2]
            magnitude = fastmri.rss(
                fastmri.complex_abs(
                    fastmri.complex_mul(center_image, center_sensitivity)
                ),
                dim=1,
            )
            states.append(magnitude[0].detach().cpu().numpy())

        final_kspace = sens_expand(prediction, sensitivity, 5)
        residual = torch.where(
            mask,
            final_kspace - masked,
            torch.zeros(1, 1, 1, 1, 1, device=device, dtype=final_kspace.dtype),
        )
        dc_error = float((
            torch.linalg.vector_norm(residual)
            / (torch.linalg.vector_norm(masked) + 1e-12)
        ).item())

    if len(states) != 13:
        raise ValueError("Expected 13 reconstruction states")

    target = np.asarray(target, dtype=np.float32).squeeze()
    if target.ndim != 2:
        raise ValueError("Expected a 2-D target")

    output_height, output_width = states[-1].shape[-2:]
    height, width = target.shape

    if output_width < width:
        height = width = output_width
    if output_height < height:
        raise ValueError("PromptMR output height is smaller than target")

    images = np.stack([_crop(image, height, width) for image in states])
    truth = _crop(target, height, width)

    final = np.asarray(images[-1], dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    energy = float(np.sum(truth64 ** 2))

    if energy <= 0:
        raise ValueError("Nonpositive target energy")

    score = float(np.sum((final - truth64) ** 2) / energy)

    return {
        "states": images,
        "target": truth,
        "features": trajectory_features(images),
        "nmse": score,
        "log_nmse": float(math.log(max(score, 1e-300))),
        "num_low_frequencies": num_low,
        "input_height": int(kspace.shape[-3]),
        "input_width": int(kspace.shape[-2]),
        "final_dc_rel_l2": dc_error,
    }
