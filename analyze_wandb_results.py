"""Summarize completed Weights & Biases runs by hyperparameter group.

Example:
    python analyze_wandb_results.py entity/project --output results.csv

The output CSV contains one row per W&B group:

    group: W&B group name shared by the sweep runs.
    n_finished_runs: Number of included runs in the group.
    learning_rate, lambda_reg: Hyperparameters read from the first run's config.
    use_scheduler, scheduler_type, scheduler_eta_min: Scheduler settings.
    selection_metric: Metric used to rank groups.
    selection_mean, selection_std: Mean and sample standard deviation across
        runs that logged the selection metric.
    n_selection_metric: Number of runs contributing to those selection values.
    report_metric: Secondary metric reported alongside the selected one.
    report_mean, report_std: Mean and sample standard deviation across runs
        that logged the report metric.
    n_report_metric: Number of runs contributing to those report values.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb


SELECTION_METRIC = "auc_by_min_task_step/val_seen"
REPORT_METRIC = "auc_by_min_task_step/val_unseen_at_best_val_seen"
SUMMARY_FIELDS = ("max", "best", "value", "mean", "median", "last")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate W&B summary metrics across runs in each group."
    )
    parser.add_argument(
        "project",
        help="W&B project path, usually 'entity/project' (or 'project' for the default entity).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("wandb_group_summary.csv"),
        help="CSV destination (default: %(default)s).",
    )
    parser.add_argument(
        "--selection-metric",
        default=SELECTION_METRIC,
        help="Metric used to select the best hyperparameters.",
    )
    parser.add_argument(
        "--report-metric",
        default=REPORT_METRIC,
        help="Secondary metric reported for the selected hyperparameters.",
    )
    parser.add_argument(
        "--include-running",
        action="store_true",
        help="Include runs whose W&B state is not 'finished'.",
    )
    return parser.parse_args()


def as_scalar(value: Any) -> float | None:
    """Return a numeric W&B summary value, including nested summary objects."""
    if isinstance(value, (int, float, np.number)):
        return float(value)

    if not isinstance(value, Mapping):
        try:
            value = dict(value)
        except (TypeError, ValueError):
            return None

    for field in SUMMARY_FIELDS:
        nested_value = value.get(field)
        if isinstance(nested_value, (int, float, np.number)):
            return float(nested_value)

    return next(
        (
            float(nested_value)
            for nested_value in value.values()
            if isinstance(nested_value, (int, float, np.number))
        ),
        None,
    )


def summary_metric_value(summary: Mapping[str, Any], metric_name: str) -> float | None:
    """Read a metric from W&B, including values stored under a ``.max`` key.

    ``wandb.define_metric(..., summary="max")`` commonly exposes its aggregate
    as ``<metric_name>.max`` in a run summary.  Accepting the base name keeps
    the command-line interface concise while still supporting that convention.
    """
    aggregate_names = (f"{metric_name}.max", f"{metric_name}.best")
    for name in (*aggregate_names, metric_name):
        value = as_scalar(summary.get(name))
        if value is not None:
            return value
    return None


def nested_value(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Read a nested config value without raising on incomplete old runs."""
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping):
            return default
        value = value.get(key, default)
    return value


def extract_hyperparameters(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the experiment settings that identify a sweep group."""
    use_scheduler = nested_value(config, "training", "use_scheduler", default=False)
    return {
        "learning_rate": nested_value(
            config,
            "training",
            "learning_rate",
            default=nested_value(config, "training", "lr"),
        ),
        "lambda_reg": nested_value(config, "training", "lambda_reg"),
        "use_scheduler": use_scheduler,
        "scheduler_type": nested_value(config, "scheduler", "type") if use_scheduler else None,
        "scheduler_eta_min": nested_value(config, "scheduler", "eta_min") if use_scheduler else None,
    }


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    """Compute a sample standard deviation, using zero for one completed run."""
    count = len(values)
    return {
        "mean": float(np.mean(values)) if values else np.nan,
        "std": float(np.std(values, ddof=1)) if count > 1 else 0.0 if count == 1 else np.nan,
        "n": count,
    }


def group_runs(runs: Sequence[Any], include_running: bool) -> dict[str, list[Any]]:
    """Keep only grouped runs, optionally excluding unfinished experiments."""
    groups: dict[str, list[Any]] = defaultdict(list)
    for run in runs:
        if not run.group or (not include_running and run.state != "finished"):
            continue
        groups[run.group].append(run)
    return groups


def get_group_statistics(
    api: wandb.Api,
    project: str,
    selection_metric: str,
    report_metric: str,
    include_running: bool = False,
) -> pd.DataFrame:
    """Return one row per W&B group with metrics across its available runs."""
    groups = group_runs(api.runs(project), include_running)
    rows: list[dict[str, Any]] = []

    for group_name, runs in groups.items():
        selection_values = [
            value
            for run in runs
            if (value := summary_metric_value(run.summary, selection_metric)) is not None
        ]
        report_values = [
            value
            for run in runs
            if (value := summary_metric_value(run.summary, report_metric)) is not None
        ]
        selection_summary = summarize(selection_values)
        report_summary = summarize(report_values)

        rows.append(
            {
                "group": group_name,
                "n_finished_runs": len(runs),
                **extract_hyperparameters(runs[0].config),
                "selection_metric": selection_metric,
                "selection_mean": selection_summary["mean"],
                "selection_std": selection_summary["std"],
                "n_selection_metric": selection_summary["n"],
                "report_metric": report_metric,
                "report_mean": report_summary["mean"],
                "report_std": report_summary["std"],
                "n_report_metric": report_summary["n"],
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["selection_mean", "learning_rate", "lambda_reg"],
        ascending=[False, True, True],
        na_position="last",
    )


def print_results_table(frame: pd.DataFrame) -> None:
    """Print the fields needed to inspect a hyperparameter sweep."""
    if frame.empty:
        print("No grouped runs with the requested state were found.")
        return

    display_columns = [
        "group",
        "learning_rate",
        "lambda_reg",
        "selection_mean",
        "selection_std",
        "n_selection_metric",
        "report_mean",
        "report_std",
        "n_report_metric",
    ]
    print(frame[display_columns].to_string(index=False, float_format="{:.4f}".format))


def main() -> None:
    args = parse_args()
    frame = get_group_statistics(
        api=wandb.Api(),
        project=args.project,
        selection_metric=args.selection_metric,
        report_metric=args.report_metric,
        include_running=args.include_running,
    )
    print_results_table(frame)

    if frame.empty:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    groups_with_selection_metric = frame.dropna(subset=["selection_mean"])
    if groups_with_selection_metric.empty:
        print(
            f"\nWrote {len(frame)} groups to {args.output}, but none contained "
            f"the selection metric '{args.selection_metric}'."
        )
        return

    best = groups_with_selection_metric.iloc[0]
    print(f"\nBest group by {args.selection_metric}: {best['group']}")
    print(f"Wrote {len(frame)} groups to {args.output}")


if __name__ == "__main__":
    main()
