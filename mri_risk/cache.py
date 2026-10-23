import csv
from pathlib import Path

import numpy as np


def read_feature_rows(csv_paths, cache_root=None):
    rows = []
    root = Path(cache_root).resolve() if cache_root else None

    for csv_path in csv_paths:
        csv_path = Path(csv_path).resolve(strict=True)

        with csv_path.open(newline="", encoding="utf-8") as handle:
            batch = list(csv.DictReader(handle))

        for row in batch:
            recorded = Path(row["static_cache_path"])

            candidates = []

            if root is not None:
                candidates.append(root / recorded.name)

            candidates.extend((
                recorded,
                csv_path.parent.parent / "cache" / recorded.name,
                csv_path.parent / "cache" / recorded.name,
            ))

            resolved = next(
                (path.resolve() for path in candidates if path.is_file()),
                None,
            )

            if resolved is None:
                raise FileNotFoundError(
                    f"Missing cache for {csv_path.name}: {recorded.name}"
                )

            row["static_cache_path"] = str(resolved)
            rows.append(row)

    return rows


def load_cache(row, backbone):
    key = "b25" if backbone == "xpdnet" else "b12"

    with np.load(row["static_cache_path"], allow_pickle=False) as data:
        reconstruction = np.asarray(data[key], dtype=np.float32)
        target = np.asarray(data["target"], dtype=np.float32)

    if reconstruction.ndim != 2 or reconstruction.shape != target.shape:
        raise ValueError("Invalid reconstruction cache")

    if not (
        np.isfinite(reconstruction).all()
        and np.isfinite(target).all()
    ):
        raise ValueError("Nonfinite cache values")

    return reconstruction, target
