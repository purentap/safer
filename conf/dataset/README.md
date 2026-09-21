# Dataset guide

Each dataset preset is selected with `dataset=<preset>`. Dataset paths are loaded as environment variables under scripts/local_env.sh. 

```bash
python train.py dataset=pi0_libero data_root="$PI0_ROLLOUT_ROOT"
```

Rollout data must come from the corresponding policy/data pipeline. Expected
layouts:

| Hydra preset | Environment variable (`path` in YAML) | Expected root layout |
| --- | --- | --- |
| `openvla` | `OPENVLA_ROLLOUT_PATH` | Paired `*.csv` and `*.pkl` rollout files directly in the root. |
| `openvla_widowx` | `OPENVLA_WIDOWX_ROLLOUT_PATH` | CSV/PKL rollout files under the configured root (glob on `*.csv`). |
| `pi0_libero` | `PI0_ROLLOUT_PATH` | `env_records/` and `policy_records/` subdirectories. |
| `pi0fast_libero` | `PI0_FAST_ROLLOUT_PATH` | `env_records/` and `policy_records/` subdirectories. |
| `pi0fast_droid` | `PI0_DROID_ROLLOUT_PATH` | Root plus relative paths in `data_path`; each listed directory contains `env_records/` and `img_embed_extracted/`. |
| `open_pi0_simpler_bridge` | `OPENPIZERO_BRIDGE_ROLLOUT_PATH` | Task directories with `*_meta.json` and matching `.pkl` rollout records. |
| `open_pi0_simpler_fractal` | `OPENPIZERO_FRACTAL_ROLLOUT_PATH` | Same as bridge. |


## Common rollout assumptions

Loaders construct `RolloutData` objects with:

- image embeddings, one vector per timestep;
- action/hidden-state embeddings, one vector per timestep;
- an episode success label (`1` success, `0` failure);
- task and episode identifiers.

Field names in the rollout files are configured in each preset via
`img_embedding_name` and `hidden_feat_name`. Expected embedding
dimensions are listed in the YAML; keep them aligned with the prepared records.

Other preset fields (train/val split ratios, DROID success-rate caps, and so
on) are documented inline in the corresponding YAML files.
