# Hydra configuration guide

`conf/config.yaml` is the composed root configuration. It selects a model and
a dataset preset, then provides training, optimizer, scheduler, checkpoint,
W&B, and visualization settings.

```yaml
defaults:
  - model: fusion_lstmv2
  - dataset: pi0fast_droid
  - _self_
```

## Select a preset

```bash
python train.py \
  model=fusion_lstmv2 \
  dataset=pi0_libero \
  data_root="$PI0_ROLLOUT_ROOT"
```

Available dataset presets are:

```text
openvla
openvla_widowx
pi0_libero
pi0fast_libero
pi0fast_droid
open_pi0_simpler_bridge
open_pi0_simpler_fractal
```

The supported model-factory presets are `fusion_lstmv2` and `lstm`.
`fusion_lstmv2` derives image and action dimensions from the chosen dataset.

## Override values


```bash
python train.py \
  dataset=pi0_libero \
  data_root="$PI0_ROLLOUT_ROOT" \
  training.n_epochs=200 \
  training.batch_size=32 \
  training.learning_rate=3e-4 \
  training.lambda_reg=1e-3 \
  model.params.lstm_hidden_dim=128 \
  wandb.enabled=false
```

For multiruns, use `-m` and comma-separated values:

```bash
python train.py -m \
  dataset=pi0_libero \
  data_root="$PI0_ROLLOUT_ROOT" \
  seed=0,1,2 \
  training.learning_rate=1e-4,3e-4
```

## Main sections

| Section | Purpose |
| --- | --- |
| `data_root` | Mandatory root directory for the chosen rollout dataset. |
| `dataset` | Loader name, embedding field names/dimensions, split ratios, and benchmark-specific options. |
| `model` | Detector implementation and architecture parameters. |
| `training` | Epochs, batch size, learning rate, regularization, and scheduler toggle. |
| `optimizer` | Optimizer type and weight decay. The current factory implements `Adam`. |
| `scheduler` | Scheduler configuration used when `training.use_scheduler=true`. |
| `checkpoint` | Best-model checkpoint setting and destination. |
| `wandb` | W&B project, grouping, run name, and enable switch. |
<!-- | `visualization` | t-SNE output directory, timestep stride, perplexity, and point size. | -->

