import argparse
import json
from pathlib import Path

import numpy as np

from mri_risk.backbones.promptmr import (
    adjacent_slices,
    load_promptmr,
    reconstruct_promptmr,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw-h5", type=Path, required=True)
    parser.add_argument("--slice", type=int, required=True)
    parser.add_argument("--mask-bank", type=Path, required=True)
    parser.add_argument("--mask-key", required=True)
    parser.add_argument("--target-npz", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import h5py

    with h5py.File(args.raw_h5, "r") as data:
        count = len(data["kspace"])

        if not 0 <= args.slice < count:
            raise ValueError("Slice index outside volume")

        indices = adjacent_slices(args.slice, count)
        stack = np.concatenate(
            [np.asarray(data["kspace"][i], dtype=np.complex64) for i in indices],
            axis=0,
        )

        if args.target_npz is None:
            target = np.asarray(
                data["reconstruction_rss"][args.slice], dtype=np.float32
            )

    if args.target_npz is not None:
        with np.load(args.target_npz, allow_pickle=False) as data:
            target = np.asarray(data["target"], dtype=np.float32)

    with np.load(args.mask_bank, allow_pickle=False) as data:
        mask = np.asarray(data[args.mask_key], dtype=bool)

    model = load_promptmr(args.checkpoint, args.device)
    result = reconstruct_promptmr(model, stack, target, mask, args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        args.output_dir / "states.npz",
        states=result.pop("states"),
        target=result.pop("target"),
    )

    (args.output_dir / "features.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    print(json.dumps(result))


if __name__ == "__main__":
    main()
