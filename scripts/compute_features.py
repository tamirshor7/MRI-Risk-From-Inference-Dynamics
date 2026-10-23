import argparse
import json
from pathlib import Path

import numpy as np

from mri_risk.trajectory import trajectory_features, project_xpdnet_primal_states

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states-npz", required=True, type=Path)
    parser.add_argument("--states-key", default="states")
    parser.add_argument("--input-type", choices=("magnitude", "xpdnet-native"), default="magnitude")
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    with np.load(args.states_npz, allow_pickle=False) as arrays:
        states = arrays[args.states_key]
    if args.input_type == "xpdnet-native":
        states = project_xpdnet_primal_states(states)
    features = trajectory_features(states)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(features, indent=2) + "\n", encoding="utf-8")
    print(f"FEATURE_COUNT={len(features)}")
    print(f"OUTPUT={args.output_json}")

if __name__ == "__main__":
    main()
