import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mri_risk.evaluation import (
    partition_matched_data, prediction_frame, risk_metrics, scan_means,
)
from mri_risk.readout import train_source_ridge
from mri_risk.trajectory import FEATURE_NAMES

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, choices=("matched", "xpdnet"))
    parser.add_argument("--source-csv", action="append", required=True, type=Path)
    parser.add_argument("--target-csv", required=True, type=Path)
    parser.add_argument("--xpdnet-protocol", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    source = pd.concat([pd.read_csv(path) for path in args.source_csv], ignore_index=True)
    target = pd.read_csv(args.target_csv)
    fit, holdout, target = partition_matched_data(source, target)

    xpdnet_grid = None
    if args.protocol == "xpdnet":
        if args.xpdnet_protocol is None:
            parser.error("--xpdnet-protocol is required with protocol=xpdnet")
        protocol = json.loads(args.xpdnet_protocol.read_text(encoding="utf-8"))
        if tuple(protocol["primary_family"]["features"]) != FEATURE_NAMES:
            raise ValueError("XPDNet source protocol does not use the shared 16-D descriptor")
        xpdnet_grid = protocol["ridge"]["lambda_grid"]

    model, cv = train_source_ridge(
        fit.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=np.float64),
        fit.log_nmse.to_numpy(dtype=np.float64),
        fit.patient_id.astype(str).to_numpy(),
        protocol=args.protocol, xpdnet_alpha_grid=xpdnet_grid,
    )
    holdout_scores = prediction_frame(
        holdout, model.predict(holdout.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=np.float64))
    )
    target_scores = prediction_frame(
        target, model.predict(target.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=np.float64))
    )
    target_scans = scan_means(target_scores)
    if len(target_scans) != 100 or target_scans.patient_id.nunique() != 25:
        raise ValueError("M4Raw requires 100 scans from 25 patients")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    holdout_scores.to_csv(output / "trajectory_source_holdout_slice.csv", index=False)
    target_scores.to_csv(output / "trajectory_m4_slice.csv", index=False)
    target_scans.to_csv(output / "trajectory_m4_scan.csv", index=False)
    summary = {
        "protocol": args.protocol,
        "selected_alpha": model.alpha,
        "features": list(FEATURE_NAMES),
        "source_holdout_slice": risk_metrics(holdout_scores),
        "m4_scan": risk_metrics(target_scans),
        "source_only": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "cv.json").write_text(json.dumps(cv, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
