import math
from pathlib import Path

import numpy as np

from mri_risk.trajectory import trajectory_features


def load_varnet(checkpoint, device):
    import torch
    from fastmri.models.varnet import VarNet

    model = VarNet(
        num_cascades=12,
        pools=4,
        chans=18,
        sens_pools=4,
        sens_chans=8,
    ).to(device)

    weights = torch.load(
        Path(checkpoint),
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(weights, dict) and "state_dict" in weights:
        weights = weights["state_dict"]

    model.load_state_dict(weights)
    model.eval()
    return model


def reconstruct_varnet(model, kspace_np, target_np, mask_np, device):
    import fastmri
    import torch

    target = np.asarray(target_np, dtype=np.float32).squeeze()
    if target.ndim != 2:
        raise ValueError(f"Expected a 2-D target, got {target.shape}")

    kspace = torch.view_as_real(
        torch.from_numpy(np.asarray(kspace_np, dtype=np.complex64))
    ).unsqueeze(0).to(device)

    mask = torch.from_numpy(np.asarray(mask_np, dtype=bool)).to(
        device=device, dtype=torch.bool
    ).unsqueeze(-1)

    if int(mask.shape[-2]) != int(kspace.shape[-2]):
        raise ValueError("Mask width and k-space width differ")

    masked = kspace * mask
    low_frequencies = int(round(int(kspace.shape[-2]) * 0.04))

    with torch.inference_mode():
        initialization = fastmri.rss(
            fastmri.complex_abs(fastmri.ifft2c(masked)),
            dim=1,
        )
        states = [initialization[0].detach().cpu().numpy()]

        sensitivity = model.sens_net(
            masked, mask, low_frequencies
        )
        current = masked.clone()

        for cascade in model.cascades:
            current = cascade(
                current, masked, mask, sensitivity
            )
            image = fastmri.rss(
                fastmri.complex_abs(fastmri.ifft2c(current)),
                dim=1,
            )
            states.append(image[0].detach().cpu().numpy())

        zero = torch.zeros(
            1, 1, 1, 1, 1,
            device=device,
            dtype=current.dtype,
        )
        residual = torch.where(
            mask, current - masked, zero
        )
        dc_error = float((
            torch.linalg.vector_norm(residual)
            / (torch.linalg.vector_norm(masked) + 1e-12)
        ).item())

    if len(states) != 13:
        raise RuntimeError(f"Expected 13 states, got {len(states)}")

    height, width = target.shape
    cropped = []

    for state in states:
        h, w = state.shape
        if h < height or w < width:
            raise ValueError("Reconstruction is smaller than target")

        top = (h - height) // 2
        left = (w - width) // 2
        cropped.append(
            state[top:top + height, left:left + width]
        )

    final = np.asarray(cropped[-1], dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    energy = float(np.sum(truth ** 2))

    if energy <= 0:
        raise ValueError("Nonpositive target energy")

    score = float(np.sum((final - truth) ** 2) / energy)

    return {
        "states": np.stack(cropped),
        "features": trajectory_features(cropped),
        "nmse": score,
        "log_nmse": float(math.log(max(score, 1e-300))),
        "num_low_frequencies": low_frequencies,
        "final_dc_rel_l2": dc_error,
    }
