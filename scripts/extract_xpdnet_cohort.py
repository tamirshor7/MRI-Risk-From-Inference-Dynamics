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


def split_patient(patient_id):
    data = ("xpdnet-fastmri-brain-source-v1:" + patient_id).encode("utf-8")
    fraction = int.from_bytes(hashlib.sha256(data).digest()[:8], "big") / 2**64
    return "fit" if fraction < 0.8 else "holdout"


def decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_manifest(args):
    if args.domain == "source":
        with args.manifest.open(newline="") as handle:
            rows = [
                row for row in csv.DictReader(handle)
                if row["source_batch"] == args.batch
            ]
        items = {}
        for row in rows:
            fname = row["filename"]
            selected = [int(index) for index in row["selected_slices"].split(",")]
            n = int(row["n_slices"])
            if selected != list(range(n // 2 - 2, n // 2 + 2)):
                raise ValueError(f"Unexpected source slice selection: {fname}")
            if fname in items:
                raise ValueError(f"Duplicate volume: {fname}")
            items[fname] = {
                "slices": selected,
                "patient_id": row["patient_id"],
            }
        if len(items) not in (460, 459):
            raise ValueError(f"Unexpected source volume count: {len(items)}")
    else:
        volumes = json.loads(args.manifest.read_text())
        items = {}
        for volume in volumes:
            fname = volume["fname"]
            selected = [int(x["slice_num"]) for x in volume["slices"]]
            if selected != [7, 8, 9, 10]:
                raise ValueError(f"Unexpected M4 slice selection: {fname}")
            if fname in items:
                raise ValueError(f"Duplicate volume: {fname}")
            items[fname] = {
                "slices": selected,
                "patient_id": str(volume["patient_id"]),
            }
        if len(items) != 100:
            raise ValueError(f"Unexpected M4 volume count: {len(items)}")
    return items


def image_name(filename, index):
    return hashlib.sha256(f"{filename}:{index}".encode()).hexdigest() + ".npz"


def fragment_complete(fragment, image_dir, fname, selected):
    if not fragment.is_file():
        return False
    try:
        rows = pd.read_csv(
            fragment, dtype={"filename": str, "patient_id": str}
        )
        if len(rows) != 4 or not rows.filename.eq(fname).all():
            return False
        if sorted(rows["slice"].astype(int).tolist()) != sorted(selected):
            return False

        required = set(FEATURE_NAMES) | {
            "log_nmse", "nmse", "mask_sha256", "static_cache_path"
        }
        if not required.issubset(rows.columns):
            return False

        for row in rows.itertuples(index=False):
            rel = Path("images") / image_name(fname, int(row.slice))
            if str(row.static_cache_path) != str(rel):
                return False
            with np.load(image_dir / rel.name, allow_pickle=False) as data:
                names = ("b0", "b12", "b25", "target")
                if not set(names).issubset(data.files):
                    return False
                arrays = [np.asarray(data[key]) for key in names]
                if len({array.shape for array in arrays}) != 1:
                    return False
                if any(
                    array.ndim != 2 or not np.isfinite(array).all()
                    for array in arrays
                ):
                    return False
        return True
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return False


def process_volume(
    args, fname, item, path, model, capture, fragment_dir, image_dir
):
    from mri_risk.backbones.xpdnet import m4raw_mask, reconstruct_xpdnet
    from mri_risk.backbones.xpdnet_source import (
        source_mask, reconstruct_xpdnet_source
    )

    fragment = fragment_dir / (Path(fname).stem + ".csv")
    selected = item["slices"]

    if fragment_complete(fragment, image_dir, fname, selected):
        return False

    with h5py.File(path, "r") as data:
        if len(data["kspace"]) <= max(selected):
            raise ValueError(f"Insufficient slices: {fname}")

        patient_id = decode(data.attrs["patient_id"])
        acquisition = decode(data.attrs.get("acquisition", ""))
        if patient_id != item["patient_id"]:
            raise ValueError(f"Patient mismatch: {fname}")

        reference = np.asarray(
            data["kspace"][selected[0]], dtype=np.complex64
        )

        if args.domain == "source":
            mask, seed, mask_sha, lines = source_mask(reference, fname)
        else:
            mask, seed, mask_sha = m4raw_mask(reference, fname)
            lines = int(np.sum(np.abs(np.asarray(mask)) > 0))

        rows = []
        pending = []

        for index in selected:
            raw = np.asarray(data["kspace"][index], dtype=np.complex64)
            target = np.asarray(
                data["reconstruction_rss"][index], dtype=np.float32
            )

            if args.domain == "source":
                result = reconstruct_xpdnet_source(
                    model, capture, raw, target, mask
                )
            else:
                result = reconstruct_xpdnet(
                    model, capture, raw, target, mask
                )

            states = result.pop("states")
            if len(states) != 26 or states.shape[1:] != target.shape:
                raise ValueError(f"Invalid trajectory shape: {fname} {index}")
            if set(result["features"]) != set(FEATURE_NAMES):
                raise ValueError("Unexpected feature columns")

            image_rel = Path("images") / image_name(fname, index)
            pending.append((
                image_dir / image_rel.name,
                {
                    "b0": np.asarray(states[0], dtype=np.float32),
                    "b12": np.asarray(states[12], dtype=np.float32),
                    "b25": np.asarray(states[25], dtype=np.float32),
                    "target": target,
                },
            ))

            rows.append({
                "filename": fname,
                "patient_id": patient_id,
                "acquisition": acquisition,
                "slice": int(index),
                "source_batch": args.batch if args.domain == "source" else "",
                "source_split": (
                    split_patient(patient_id)
                    if args.domain == "source" else ""
                ),
                "seed": int(seed),
                "mask_sha256": mask_sha,
                "mask_lines": lines,
                "nmse": result["nmse"],
                "log_nmse": result["log_nmse"],
                "b25_output_rel_error": result["endpoint_rel"],
                "static_cache_path": str(image_rel),
                **result["features"],
            })

            print(
                f"SLICE={fname}:{index} NMSE={result['nmse']:.9e}",
                flush=True,
            )

    for destination, payload in pending:
        temporary = destination.with_name(
            destination.name + f".tmp.{os.getpid()}"
        )
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        os.replace(temporary, destination)

    temporary_csv = fragment.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(temporary_csv, index=False)
    os.replace(temporary_csv, fragment)
    return True


def write_features(args, items, fragment_dir):
    fragments = [
        fragment_dir / (Path(fname).stem + ".csv")
        for fname in items
    ]
    existing = [path for path in fragments if path.is_file()]
    if not existing:
        raise RuntimeError("No completed volumes")

    table = pd.concat(
        [
            pd.read_csv(path, dtype={"patient_id": str})
            for path in existing
        ],
        ignore_index=True,
    )

    if table[["filename", "slice"]].duplicated().any():
        raise RuntimeError("Duplicate slice identity")

    table.sort_values(["filename", "slice"], inplace=True)
    output = args.output_dir / "features.csv"
    temporary = output.with_suffix(".csv.tmp")
    table.to_csv(temporary, index=False)
    os.replace(temporary, output)

    print(
        f"VOLUMES={len(existing)}/{len(items)} "
        f"SLICES={len(table)}/{4 * len(items)}",
        flush=True,
    )

    if len(existing) == len(items) and len(table) == 4 * len(items):
        print("XPDNET_COHORT_STATUS=PASS", flush=True)
    else:
        print("XPDNET_COHORT_STATUS=PARTIAL", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("source", "m4"), required=True)
    parser.add_argument("--batch", choices=("val_batch_0", "val_batch_1"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-volumes", type=int, default=0)
    args = parser.parse_args()

    if (args.domain == "source") != (args.batch is not None):
        parser.error("Specify --batch for source only")
    if args.max_volumes < 0:
        parser.error("--max-volumes must be nonnegative")

    items = load_manifest(args)

    def identity(path):
        path = path.resolve(strict=True)
        stat = path.stat()
        return [str(path), stat.st_size, stat.st_mtime_ns]

    run = {
        "domain": args.domain,
        "batch": args.batch,
        "checkpoint": identity(args.checkpoint),
        "manifest": identity(args.manifest),
        "input": str(args.input.resolve(strict=True)),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = args.output_dir / "run.json"

    if config.exists():
        if json.loads(config.read_text()) != run:
            raise RuntimeError("Output directory belongs to a different run")
    elif any(args.output_dir.iterdir()):
        raise RuntimeError("Unidentified files in output directory")
    else:
        config.write_text(json.dumps(run, indent=2) + "\n")

    fragment_dir = args.output_dir / "fragments"
    image_dir = args.output_dir / "images"
    stage_dir = args.output_dir / "staging"

    for folder in (fragment_dir, image_dir, stage_dir):
        folder.mkdir(parents=True, exist_ok=True)

    from mri_risk.backbones.xpdnet import load_xpdnet

    model, capture = load_xpdnet(args.checkpoint)
    limit = args.max_volumes or len(items)
    visited = 0

    if args.domain == "m4":
        for fname, item in items.items():
            if visited >= limit:
                break

            fragment = fragment_dir / (Path(fname).stem + ".csv")
            if not fragment_complete(
                fragment, image_dir, fname, item["slices"]
            ):
                changed = process_volume(
                    args, fname, item, args.input / fname,
                    model, capture, fragment_dir, image_dir
                )
            else:
                changed = False

            visited += 1
            print(
                f"VOLUME={fname} PROCESSED={int(changed)}",
                flush=True,
            )
    else:
        with tarfile.open(args.input, mode="r|xz") as archive:
            for member in archive:
                if visited >= limit:
                    break
                if not member.isfile() or not member.name.endswith(".h5"):
                    continue

                fname = Path(member.name).name
                if fname not in items:
                    continue

                item = items[fname]
                fragment = fragment_dir / (Path(fname).stem + ".csv")

                if fragment_complete(
                    fragment, image_dir, fname, item["slices"]
                ):
                    changed = False
                else:
                    stage = stage_dir / fname
                    try:
                        source = archive.extractfile(member)
                        if source is None:
                            raise RuntimeError(f"Cannot extract {fname}")

                        with source, stage.open("wb") as dest:
                            shutil.copyfileobj(
                                source, dest, length=8 * 1024 * 1024
                            )

                        changed = process_volume(
                            args, fname, item, stage,
                            model, capture, fragment_dir, image_dir
                        )
                    finally:
                        stage.unlink(missing_ok=True)

                visited += 1
                print(
                    f"VOLUME={fname} PROCESSED={int(changed)}",
                    flush=True,
                )

    write_features(args, items, fragment_dir)


if __name__ == "__main__":
    main()
