import argparse
import csv
import hashlib
import json
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from mri_risk.trajectory import FEATURE_NAMES


def _load_backbone(name, checkpoint, device):
    if name == "varnet":
        from mri_risk.backbones.varnet import load_varnet, reconstruct_varnet
        return load_varnet(checkpoint, device), reconstruct_varnet
    from mri_risk.backbones.promptmr import load_promptmr, reconstruct_promptmr
    return load_promptmr(checkpoint, device), reconstruct_promptmr


def _run(model, reconstruct, backbone, dataset, center, target, mask, device):
    kspace = dataset["kspace"]
    if backbone == "varnet":
        raw = np.asarray(kspace[center], dtype=np.complex64)
    else:
        from mri_risk.backbones.promptmr import adjacent_slices
        indices = adjacent_slices(center, len(kspace))
        raw = np.concatenate(
            [np.asarray(kspace[index], dtype=np.complex64) for index in indices],
            axis=0,
        )
    return reconstruct(model, raw, target, mask, device)


def _store_images(output, result, target, scale):
    images = np.asarray(result["states"], dtype=np.float32)
    target = np.asarray(result.get("target", target), dtype=np.float32)
    if images.shape[1:] != target.shape:
        raise ValueError("Trajectory and target dimensions disagree")
    np.savez_compressed(
        output,
        b0=np.asarray(images[0] * scale, dtype=np.float32),
        b12=np.asarray(images[-1] * scale, dtype=np.float32),
        target=np.asarray(target * scale, dtype=np.float32),
    )



def _complete_fragment(fragment, selections, filename, image_dir):
    if not fragment.is_file():
        return False

    try:
        frame = pd.read_csv(fragment)
        required = {"filename", "slice", "static_cache_path"}

        if not required.issubset(frame.columns):
            return False
        if len(frame) != len(selections):
            return False
        if not frame["filename"].astype(str).eq(filename).all():
            return False

        expected = {int(item["slice"]) for item in selections}
        observed = frame["slice"].astype(int).tolist()

        if len(set(observed)) != len(observed) or set(observed) != expected:
            return False

        for row in frame.itertuples(index=False):
            name = hashlib.sha256(
                f"{filename}:{int(row.slice)}".encode()
            ).hexdigest() + ".npz"

            if str(row.static_cache_path) != str(Path("images") / name):
                return False

            image_path = image_dir / name
            if not image_path.is_file():
                return False

            with np.load(image_path, allow_pickle=False) as arrays:
                if not {"b0", "b12", "target"}.issubset(arrays.files):
                    return False

                values = [
                    np.asarray(arrays[key])
                    for key in ("b0", "b12", "target")
                ]

                if any(value.ndim != 2 for value in values):
                    return False
                if len({value.shape for value in values}) != 1:
                    return False
                if not all(np.isfinite(value).all() for value in values):
                    return False

        return True

    except (
        OSError,
        ValueError,
        KeyError,
        EOFError,
        zipfile.BadZipFile,
    ):
        return False


def _process_volume(args, model, reconstruct, item, bank, raw_path, fragment_dir, image_dir):
    filename = item["filename"]
    fragment = fragment_dir / (Path(filename).stem + ".csv")
    selections = item["selections"]

    if _complete_fragment(fragment, selections, filename, image_dir):
        return False

    with h5py.File(raw_path, "r") as dataset:
        rows = []
        for selection in selections:
            center = int(selection["slice"])
            mask = np.asarray(bank[selection["mask_key"]], dtype=bool)

            if args.domain == "source":
                target = np.asarray(dataset["reconstruction_rss"][center], dtype=np.float32)
                scale = 1e6
            else:
                target = np.asarray(
                    dataset["reconstruction_rss"][center], dtype=np.float32
                )
                scale = 1.0

            result = _run(
                model, reconstruct, args.backbone, dataset,
                center, target, mask, args.device
            )

            cache_name = hashlib.sha256(
                f"{filename}:{center}".encode()
            ).hexdigest() + ".npz"
            _store_images(image_dir / cache_name, result, target, scale)

            acquisition = item["acquisition"] or dataset.attrs.get("acquisition", "")
            if isinstance(acquisition, bytes):
                acquisition = acquisition.decode("utf-8")

            row = {
                "filename": filename,
                "patient_id": str(item["patient_id"]),
                "acquisition": str(acquisition),
                "slice": center,
                "nmse": float(result["nmse"]),
                "log_nmse": float(result["log_nmse"]),
                "final_dc_rel_l2": float(result["final_dc_rel_l2"]),
                "num_low_frequencies": int(result["num_low_frequencies"]),
                "cache_scale": scale,
                "static_cache_path": str(Path("images") / cache_name),
                **{
                    name: float(result["features"][name])
                    for name in FEATURE_NAMES
                },
            }

            if args.domain == "source":
                row["source_batch"] = args.source_batch
                row["source_split"] = str(item["source_split"])

            rows.append(row)

    temporary = fragment.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, fragment)
    return True


def _read_manifest(args):
    if args.domain == "source":
        with args.manifest.open(newline="") as handle:
            candidates = [
                row for row in csv.DictReader(handle)
                if row["source_batch"] == args.source_batch
            ]
        return {
            row["filename"]: {
                "filename": row["filename"],
                "patient_id": str(row["patient_id"]),
                "acquisition": "",
                "source_split": row["source_split"],
                "selections": [
                    {"slice": int(index), "mask_key": row["mask_bank_key"]}
                    for index in row["selected_slices"].split(",")
                ],
            }
            for row in candidates
        }

    volumes = json.loads(args.manifest.read_text())
    if len(volumes) != 100:
        raise ValueError("Expected 100 target scans")

    return {
        volume["fname"]: {
            "filename": volume["fname"],
            "patient_id": str(volume["patient_id"]),
            "acquisition": "AXFLAIR",
            "selections": [
                {
                    "slice": int(entry["slice_num"]),
                    "mask_key": entry["mask_bank_key"],
                }
                for entry in volume["slices"]
            ],
        }
        for volume in volumes
    }


def _aggregate(fragment_dir, output, n_volumes, expected_rows, complete):
    fragments = sorted(fragment_dir.glob("*.csv"))
    frames = [
        pd.read_csv(path, dtype={"patient_id": str})
        for path in fragments
    ]

    if complete and len(fragments) != n_volumes:
        raise ValueError(
            f"Expected {n_volumes} volume fragments, found {len(fragments)}"
        )

    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if complete and len(table) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} slices, found {len(table)}"
        )

    if not table.empty:
        if table.duplicated(["filename", "slice"]).any():
            raise ValueError("Duplicate slice identities")
        table.sort_values(
            ["filename", "slice"]
        ).to_csv(output, index=False)

    print(
        f"VOLUMES={len(fragments)}/{n_volumes} "
        f"SLICES={len(table)}/{expected_rows}",
        flush=True,
    )
    print(
        "COHORT_STATUS="
        + ("PASS" if complete and len(fragments) == n_volumes else "PARTIAL"),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=("varnet", "promptmr"), required=True)
    parser.add_argument("--domain", choices=("source", "m4"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mask-bank", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-batch", choices=("val_batch_0", "val_batch_1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-volumes", type=int, default=0)
    args = parser.parse_args()

    if args.domain == "source" and args.source_batch is None:
        parser.error("Source requires --source-batch")
    if args.max_volumes < 0:
        parser.error("--max-volumes must be nonnegative")

    manifest = _read_manifest(args)
    if not manifest:
        raise ValueError("Empty manifest")
    if any(len(item["selections"]) != 4 for item in manifest.values()):
        raise ValueError("Expected four selected slices per volume")

    output = args.output_dir

    def identity(path):
        path = path.resolve(strict=True)
        stat = path.stat()
        return [str(path), stat.st_size, stat.st_mtime_ns]

    run = {
        "backbone": args.backbone,
        "domain": args.domain,
        "checkpoint": identity(args.checkpoint),
        "manifest": identity(args.manifest),
        "mask_bank": identity(args.mask_bank),
        "input": str(args.input.resolve(strict=True)),
        "source_batch": args.source_batch,
    }

    run_file = output / "run.json"

    if run_file.is_file():
        if json.loads(run_file.read_text()) != run:
            raise RuntimeError("Output directory belongs to another run")
    elif (output / "fragments").exists() and any(
        (output / "fragments").glob("*.csv")
    ):
        raise RuntimeError("Existing fragments have no run configuration")
    else:
        output.mkdir(parents=True, exist_ok=True)
        run_file.write_text(json.dumps(run, indent=2) + "\n")

    fragment_dir = output / "fragments"
    image_dir = output / "images"
    staging_dir = output / "staging"

    for directory in (fragment_dir, image_dir, staging_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model, reconstruct = _load_backbone(
        args.backbone, args.checkpoint, args.device
    )
    remaining = args.max_volumes or len(manifest)
    done = 0

    with np.load(args.mask_bank, allow_pickle=False) as bank:
        needed = {
            selection["mask_key"]
            for item in manifest.values()
            for selection in item["selections"]
        }
        missing = needed - set(bank.files)
        if missing:
            raise ValueError(f"Mask bank is missing {len(missing)} keys")

        if args.domain == "m4":
            for filename, item in manifest.items():
                if done >= remaining:
                    break
                raw_path = args.input / filename
                if not raw_path.is_file():
                    raise FileNotFoundError(raw_path)
                processed = _process_volume(
                    args, model, reconstruct, item, bank,
                    raw_path, fragment_dir, image_dir
                )
                print(
                    f"VOLUME={filename} PROCESSED={int(processed)}",
                    flush=True,
                )
                done += 1
        else:
            with tarfile.open(args.input, mode="r|xz") as archive:
                for member in archive:
                    if done >= remaining:
                        break
                    if not member.isfile() or not member.name.endswith(".h5"):
                        continue
                    filename = Path(member.name).name
                    if filename not in manifest:
                        continue

                    item = manifest[filename]
                    fragment = fragment_dir / (Path(filename).stem + ".csv")

                    if _complete_fragment(
                        fragment, item["selections"], filename, image_dir
                    ):
                        print(f"VOLUME={filename} PROCESSED=0", flush=True)
                        done += 1
                        continue

                    staging = staging_dir / filename
                    try:
                        source = archive.extractfile(member)
                        if source is None:
                            raise RuntimeError(f"Cannot extract {member.name}")
                        with source, staging.open("wb") as destination:
                            shutil.copyfileobj(
                                source, destination, length=8 * 1024 * 1024
                            )
                        processed = _process_volume(
                            args, model, reconstruct, manifest[filename],
                            bank, staging, fragment_dir, image_dir
                        )
                        print(
                            f"VOLUME={filename} PROCESSED={int(processed)}",
                            flush=True,
                        )
                        done += 1
                    finally:
                        staging.unlink(missing_ok=True)

    _aggregate(
        fragment_dir,
        output / "features.csv",
        len(manifest),
        len(manifest) * 4,
        args.max_volumes == 0,
    )


if __name__ == "__main__":
    main()
