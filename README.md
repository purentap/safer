# SAFER: Incorporating Visual Context for Multitask Failure Detection in VLA Models

Official code release for **[Paper title — TBD]** ([arXiv / project page — TBD]).


## Setup

**Requirements:** Python 3.10+, NVIDIA GPU with CUDA 12.8 recommended.

Create an environment (Conda or `venv`), upgrade `pip`, then install **PyTorch first**, then the rest of the dependencies.

### 1. Create environment

**Conda:**

```bash
conda create -n safer python=3.10 -y
conda activate safer
```

### 2. Install dependencies

```bash
python -m pip install -U pip

# GPU (CUDA 12.8)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Project dependencies (Hydra, W&B, sklearn, …)
pip install -r requirements.txt
```

**Weights & Biases:** training configs default to `wandb.enabled=true`. Either run `wandb login` or disable logging:

```bash
wandb login
```

**Analysis extras (optional):** OpenCV / imageio for CSV and video analysis helpers:

```bash
pip install opencv-python imageio
# or, when added: pip install -r requirements-analysis.txt
```

## Data

Rollouts are directories of pickled policy records (image and action embeddings plus episode metadata). Layout and paths are configured per benchmark under [`conf/dataset/`](conf/dataset/).

1. Download or generate embeddings for your benchmark (see paper / supplementary material).
2. Set `dataset.path` to the root of your rollout tree (override on the CLI or edit the YAML).
3. Adjust `dataset.data_path` if you use a subset of tasks.

Example override:

```bash
python train.py dataset=pi0fast_droid dataset.path=/path/to/pi0fast_droid_0510_all
```

## Training

Default config: [`conf/config.yaml`](conf/config.yaml) (model + dataset composed via Hydra).

Single run:

```bash
python train.py model=fusion_lstmv2 dataset=pi0fast_droid seed=0 wandb.enabled=false
```

Hyperparameter sweep (matches [`scripts/pi0fast_droid.bash`](scripts/pi0fast_droid.bash)):

```bash
bash scripts/pi0fast_droid.bash
```

More config examples: [`conf/README.md`](conf/README.md).

## Project layout

| Path | Description |
|------|-------------|
| `train.py` | Training entry point (Hydra) |
| `model.py` | Detector architectures (MLP, LSTM, fusion) |
| `datasets/` | Per-benchmark dataset loaders and splits |
| `conformal/` | Functional conformal prediction utilities |
| `metrics_utils.py` | ROC-AUC and threshold evaluation |
| `scripts/` | Shell wrappers for paper experiments |
| `conf/` | Hydra configs (model, dataset, training) |

## Citation

```bibtex
@article{yourpaper2026,
  title   = {TBD},
  author  = {TBD},
  journal = {TBD},
  year    = {2026}
}
```

## License

TBD
