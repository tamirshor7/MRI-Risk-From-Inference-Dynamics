# MRI Reconstruction Risk Estimation from Inference Dynamics

**Tamir Shor, Alex Bronstein**

This repository contains the official implementation of the **WACV 2027** paper [MRI Reconstruction Risk Estimation from Inference Dynamics](https://arxiv.org/abs/2610.26558).

Iterative MRI reconstruction models produce a sequence of intermediate estimates before returning the final image. We use this sequence to estimate reconstruction error, without requiring a reference image at inference time. The method extracts 16 trajectory features and learns a ridge regression predictor on source-domain data, while keeping the reconstruction model frozen.

Our implementation supports XPDNet, VarNet, and PromptMR+ as reconstruction backbones.

## Installation

Create the environment, run all commands below from the repository root:

    conda env create -f environment.yml
    conda activate mri-inference-risk
    pip install -e .

## Data

Our experiments use [fastMRI brain](https://fastmri.med.nyu.edu/) as the source dataset and [M4Raw](https://github.com/mylyu/M4Raw) for cross-domain evaluation.

The extraction scripts take dataset directories, model checkpoints, sampling masks, and measurement manifests as arguments.

For XPDNet and PromptMR+:

- [XPDNet / fastMRI Reproducible Benchmark](https://github.com/zaccharieramzi/fastmri-reproducible-benchmark)
- [PromptMR+](https://github.com/hellopipu/PromptMR-plus)


## Running the experiments

The reconstruction scripts extract intermediate model states and compute trajectory descriptors. The readout is trained on source-domain features and evaluated on the requested dataset.

### Reconstruction trajectories

VarNet:

    python scripts/extract_cohort.py \
        --backbone varnet \
        --domain m4 \
        --checkpoint /path/to/varnet_checkpoint.pt \
        --manifest /path/to/m4_manifest.json \
        --mask-bank /path/to/m4_masks.npz \
        --input /path/to/m4raw/multicoil_test \
        --device cuda:0 \
        --output-dir results/varnet/m4

For PromptMR+, use the same interface with `--backbone promptmr` and the corresponding checkpoint.

For XPDNet:

    MRI_RISK_XPDNET_VENDOR=/path/to/fastmri-reproducible-benchmark \
    python scripts/xpdnet_compat.py scripts/extract_xpdnet_cohort.py \
        --domain m4 \
        --checkpoint /path/to/xpdnet_weights.h5 \
        --manifest /path/to/m4_manifest.json \
        --input /path/to/m4raw/multicoil_test \
        --output-dir results/xpdnet/m4

The extraction scripts also support source-domain processing with the corresponding fastMRI inputs and measurement protocol.

### Risk prediction

After extracting the source and target features, fit the source-trained readout:

    python scripts/reproduce_readout.py \
        --protocol matched \
        --source-csv results/varnet/source/features.csv \
        --target-csv results/varnet/m4/features.csv \
        --output-dir results/varnet/readout


## Citation

@article{shor2026mri,
  title={MRI Reconstruction Risk Estimation from Inference Dynamics},
  author={Shor, Tamir and Bronstein, Alex},
  journal={arXiv preprint arXiv:2610.26558},
  year={2026}
}
