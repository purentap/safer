# SAFER:  Incorporating Visual Context for Multitask Failure Detection in VLA Models

Official code release for [SAFER: Incorporating Visual Context for Multitask Failure Detection in VLA Models (IJCNN-2026)](http://linklings.s3.amazonaws.com/organizations/WCCI/wcci2026/submissions/stype114/5oRpz-ijcnn_pap5374s2.pdf). This codebase is built upon the SAFE codebase.

## Contents

- `train.py` — Main training pipeline.
- `datasets/` — loaders for OpenVLA, OpenVLA-WidowX, Pi0, Pi0-FAST, PI0FAST-DROID, and SIMPLER rollouts.
- `model.py` — LSTM and fusion-LSTM detector implementations.
- `conf/` — Hydra defaults, model presets, and benchmark settings.
- `scripts/` — paper sweep presets.
- `embedding_visualization.py` — t-SNE plots of rollout image embeddings.
- `analyze_wandb_results.py` — aggregate completed W&B sweep runs.

## Installation

<strong>Requirements</strong>: Python 3.10+, NVIDIA GPU with CUDA 12.8 recommended.



```bash
conda create -n safer python=3.10 -y
conda activate safer

python -m pip install --upgrade pip

#install pytorch
pip install torch --index-url https://download.pytorch.org/whl/cu128

#install project dependencies
pip install -r requirements.txt
```

Weights & Biases (W&B) is enabled by default. Log in before using it, or pass
`wandb.enabled=false` to run offline:

```bash
wandb login
```

## Data setup

We do not provide the datasets. Trajectories can be collected using [SAFE](https://github.com/vla-safe) implementation, with image embeddings extracted during the collection process. Each benchmark requires
prepared rollout records containing policy embeddings (hidden features and image embeddings) and episode metadata.

Set the paths to the rollout datasets in
`scripts/local_env.sh`:

```bash
#!/usr/bin/env bash

export OPENVLA_ROLLOUT_PATH="/path/to/openvla/rollouts"
export OPENVLA_WIDOWX_ROLLOUT_PATH="/path/to/openvla_widowx/rollouts"
export PI0_ROLLOUT_PATH="/path/to/pi0_libero/rollouts"
export PI0_FAST_ROLLOUT_PATH="/path/to/pi0fast_libero/rollouts"
export PI0_DROID_ROLLOUT_PATH="/path/to/pi0fast_droid/rollouts"
export OPENPIZERO_FRACTAL_ROLLOUT_PATH="/path/to/openpi0_fractal/rollouts"
export OPENPIZERO_BRIDGE_ROLLOUT_PATH="/path/to/openpi0_bridge/rollouts"
```

Then, load the environment variables before running the benchmarks: 
```bash
source scripts/local_env.sh
```

See [the dataset guide](conf/dataset/README.md) for the expected directory
layouts and record fields. Dataset-specific feature names and dimensions are
defined in `conf/dataset/*.yaml`.

## Training


The `scripts/` directory contains the hyperparameter and seed choices used for
benchmark sweeps. Run commands from the repository root.

```bash
source scripts/local_env.sh

bash scripts/***.py
```


The model factory currently registers `FusionLSTMModel_v2` and `LSTMModel`.
Use `model=fusion_lstmv2` for the paper results.

## W&B sweep analysis

`analyze_wandb_results.py` aggregates completed grouped W&B runs. It selects
groups by mean validation-seen AUC and reports validation-unseen AUC measured
at the best validation-seen epoch.

```bash
python analyze_wandb_results.py wandb_username/project_name \
  --output outputs/wandb_group_summary.csv
```

## Visualization

`embedding_visualization.py` loads rollouts with the same dataset handlers as
training, projects **image embeddings** to 2D with t-SNE, and writes two PNG
figures: one colored by episode success/failure and one by task ID. The base code is taken from the SAFE work. 

Loader field names and benchmark-specific options come from `conf/dataset/<name>.yaml`;
paths and plot settings are passed on the command line.

Run from the repository root. Either set rollout paths via
`scripts/local_env.sh` or pass `--data-root` explicitly:

```bash
source scripts/local_env.sh

python embedding_visualization.py --dataset openvla

python embedding_visualization.py \
  --dataset pi0_libero \
  --data-root "$PI0_ROLLOUT_PATH" \
  --output-dir outputs/pi0_libero_tsne \
  --stride 5 \
  --perplexity 40 \
```

for projection in the paper, run: 
```bash
python embedding_visualization.py -d openvla -o visualizations/openvla_tsne
```

| Flag | Default | Description |
| --- | --- | --- |
| `--dataset` / `-d` | *(required)* | Preset: `openvla`, `openvla_widowx`, `pi0_libero`, `pi0fast_libero`, `pi0fast_droid`, `open_pi0_simpler_bridge`, `open_pi0_simpler_fractal` |
| `--data-root` | env from preset YAML | Rollout root directory |
| `--output-dir` / `-o` | `visualizations` | Output directory for PNGs |
| `--stride` | `1` | Use every N-th timestep when collecting embeddings |
| `--perplexity` | `30` | t-SNE perplexity (capped automatically for small samples) |

Outputs (default directory `visualizations/`):

- `<dataset>_feats_vis_skip<stride>-succ.png` — cool/warm colors by failure label
- `<dataset>_feats_vis_skip<stride>-taskid.png` — color by task ID

## Citation

If you find this work useful, please cite:

```
@inproceedings{tap2026safer,
title={SAFER: Incorporating Visual Context for Multitask Failure Detection in VLA Models},
author={Tap, Püren and Abdurrahman, Rafi Putra and Sarıel, Sanem and Kalkan, Sinan},
booktitle={International Joint Conference on Neural Networks (IJCNN)},
year={2026}
}
```