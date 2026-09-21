"""t-SNE visualization of image embeddings for a rollout dataset preset.

Examples:
  source scripts/local_env.sh
  python embedding_visualization.py --dataset pi0_libero --data-root "$PI0_ROLLOUT_ROOT"
  python embedding_visualization.py -d openvla -o visualizations/openvla_tsne
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationResolutionError
from sklearn.manifold import TSNE

import utils
from datasets import get_dataset_handler

REPO_ROOT = Path(__file__).resolve().parent

DATASET_CHOICES = [
    "openvla",
    "openvla_widowx",
    "pi0_libero",
    "pi0fast_libero",
    "pi0fast_droid",
    "open_pi0_simpler_bridge",
    "open_pi0_simpler_fractal",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project rollout image embeddings with t-SNE and save scatter plots.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        choices=DATASET_CHOICES,
        required=True,
        help="Dataset loader preset (matches conf/dataset/<name>.yaml).",
    )
    parser.add_argument(
        "--data-root",
        help=(
            "Root directory containing rollout records. "
            "If omitted, uses the path from conf/dataset/<name>.yaml "
            "(requires the matching env var from scripts/local_env.sh)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="visualizations",
        help="Directory for PNG outputs (default: visualizations).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Subsample every N timesteps when collecting embeddings (default: 1).",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (default: 30).",
    )

    return parser.parse_args()


def load_dataset_cfg(dataset_name: str, data_root: str | None) -> DictConfig:
    yaml_path = REPO_ROOT / "conf" / "dataset" / f"{dataset_name}.yaml"
    if not yaml_path.is_file():
        print(f"Missing dataset preset: {yaml_path}", file=sys.stderr)
        sys.exit(1)

    dataset = OmegaConf.load(yaml_path)

    if data_root is None:
        try:
            OmegaConf.resolve(dataset)
        except InterpolationResolutionError as exc:
            print(
                "Could not resolve dataset path from the environment.\n"
                "Either pass --data-root /path/to/rollouts or run:\n"
                "  source scripts/local_env.sh",
                file=sys.stderr,
            )
            print(f"Details: {exc}", file=sys.stderr)
            sys.exit(1)
        data_root = dataset.path

    name = dataset.get("name", dataset_name)
    return OmegaConf.create(
        {
            "data_root": data_root,
            "dataset": OmegaConf.merge(
                dataset,
                {"name": name, "path": data_root},
            ),
        }
    )


def main(args: argparse.Namespace) -> None:
    save_folder = args.output_dir
    feat_skip = args.stride
    perplexity = args.perplexity
    dataset_name = args.dataset

    os.makedirs(save_folder, exist_ok=True)
    #seed for data loading
    utils.seed_everything(0)

    cfg = load_dataset_cfg(dataset_name, args.data_root)
    DatasetHandler = get_dataset_handler(dataset_name)
    dataset_handler = DatasetHandler(cfg)
    rollouts = dataset_handler.load_rollouts()
    rollouts = sorted(rollouts, key=lambda x: (x.get_task_id(), x.get_episode_idx()))
    print(f"Loaded {len(rollouts)} rollouts from {cfg.data_root}")
    print("Data loaded successfully")

    utils.seed_everything(0)

    feats, rollout_indices = [], []
    for i, r in enumerate(rollouts):
        feat = r.get_img_embeddings()[::feat_skip]
        feats.append(feat)
        rollout_indices.append(np.ones(feat.shape[0]) * i)

    feats = np.concatenate(feats, axis=0)
    rollout_indices = np.concatenate(rollout_indices, axis=0)

    projector = TSNE(n_components=2, perplexity=tsne_perplexity)
    feats_projected = projector.fit_transform(feats)
    print(f"feats_projected: {feats_projected.shape} {feats_projected.dtype}")

    task_ids = []
    colors_by_success = []
    for i, r in enumerate(rollouts):
        feat_proj = feats_projected[rollout_indices == i]
        task_ids.append(np.ones(feat_proj.shape[0]) * r.get_task_id())
        if r.get_episode_success() == 0:
            colors_by_success.append(np.linspace(0.6, 1, feat_proj.shape[0]))
        else:
            colors_by_success.append(np.zeros(feat_proj.shape[0]))
    colors_by_success = np.concatenate(colors_by_success, axis=0)
    task_ids = np.concatenate(task_ids, axis=0)

    succ_path = f"{save_folder}/{dataset_name}_feats_vis_skip{feat_skip}-succ.png"
    plt.figure(dpi=200)
    plt.scatter(
        feats_projected[:, 0],
        feats_projected[:, 1],
        c=colors_by_success,
        cmap="coolwarm",
        s=0.5,
        alpha=0.5,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(succ_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {succ_path}")

    custom_colors = [
        "#98df8a",
        "#c5b0d5",
        "#8c564b",
        "#ff7f0e",
        "#9467bd",
        "#bcbd22",
        "#7f7f7f",
        "#e377c2",
        "#2ca02c",
        "#c49c94",
    ]
    cmap_task_ids = mpl.colors.ListedColormap(custom_colors)

    task_path = f"{save_folder}/{dataset_name}_feats_vis_skip{feat_skip}-taskid.png"
    plt.figure(dpi=200)
    plt.scatter(
        feats_projected[:, 0],
        feats_projected[:, 1],
        c=task_ids,
        cmap=cmap_task_ids,
        s=0.5,
        alpha=0.7,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(task_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {task_path}")


if __name__ == "__main__":
    main(parse_args())
