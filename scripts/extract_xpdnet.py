import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from mri_risk.backbones.xpdnet import load_xpdnet, m4raw_mask, reconstruct_xpdnet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw-h5", type=Path, required=True)
    parser.add_argument("--slice", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with h5py.File(args.raw_h5, "r") as data:
        kspace = np.asarray(data["kspace"][args.slice], dtype=np.complex64)
        probe = np.asarray(data["kspace"][7], dtype=np.complex64)
        target = np.asarray(data["reconstruction_rss"][args.slice], dtype=np.float32)

    model, capture = load_xpdnet(args.checkpoint)
    mask, seed, mask_sha = m4raw_mask(probe, args.raw_h5.name)
    result = reconstruct_xpdnet(model, capture, kspace, target, mask)

    states = result.pop("states")
    result.update({"seed": int(seed), "mask_sha256": mask_sha})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "states.npz", states=states)
    (args.output_dir / "features.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
