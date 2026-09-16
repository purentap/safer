"""
Script to extract mean and std of success metrics from WandB runs grouped by hyperparameters.

This script queries WandB API to get aggregated statistics for each hyperparameter combination.
"""

import wandb
import pandas as pd
import numpy as np
from collections import defaultdict
from datasets import get_dataset_handler

# Initialize WandB API
api = wandb.Api()

# Set your project name
PROJECT_NAME = "*******"  # Update this to match your project
POLICY = "PI0"

def _extract_scalar_from_summary_value(value):
    """
    W&B summaries may return nested SummarySubDict objects for metrics,
    especially when historical aggregation or nested keys are involved.
    This helper attempts to coerce such values into a single float.
    """
    # Direct numeric
    if isinstance(value, (int, float, np.number)):
        return float(value)
    # SummarySubDict or dict-like
    try:
        # Convert to a plain dict if possible
        as_dict = dict(value)
    except Exception:
        as_dict = value if isinstance(value, dict) else None
    if isinstance(as_dict, dict):
        # Prefer common numeric fields if present
        for key in ("max", "best", "value", "mean", "median", "last"):
            if key in as_dict and isinstance(as_dict[key], (int, float, np.number)):
                return float(as_dict[key])
        # Fallback: first numeric found
        for v in as_dict.values():
            if isinstance(v, (int, float, np.number)):
                return float(v)
    # Could not coerce
    return None

def get_group_statistics(project_name, metric_name="auc_by_min_task_step/val_seen"):
    """
    Get mean and std for a metric across all runs in each group.
    
    Args:
        project_name: WandB project name
        metric_name: Name of the metric to analyze (e.g., "auc_by_min_task_step/val_seen")
    
    Returns:
        DataFrame with columns: group, lr, lambda_reg, mean, std, n_runs
    """
    runs = api.runs(project_name)
    
    # Group runs by their group name
    groups = defaultdict(list)
    for run in runs:
        if run.group:  # Only process runs that have a group
            groups[run.group].append(run)
    
    results = []
    
    for group_name, group_runs in groups.items():
        # Extract hyperparameters from config
        try:
            # Get from first run's config
            first_run = group_runs[0]
            config = first_run.config
            
            # Handle nested config structure
            if "training" in config:
                lr = config["training"].get("learning_rate", config["training"].get("lr", "N/A"))
                lambda_reg = config["training"].get("lambda_reg", "N/A")
                use_scheduler = config["training"].get("use_scheduler", False)

            if use_scheduler:
                scheduler_type = config["scheduler"].get("type", "N/A")
                eta_min = config["scheduler"].get("eta_min", "N/A")
            else:
                scheduler_type = "None"
                eta_min = "N/A"
        except Exception:
            lr = "N/A"
            lambda_reg = "N/A"
            use_scheduler = False
            scheduler_type = "N/A"
            eta_min = "N/A"
        # Extract metric values from summary (best value)
        metric_values_seen = []
        metric_values_unseen_at_best_val_seen = []
        for run in group_runs:
            # Try to get metric from summary
            summary = run.summary

            for metric_name in metrics:
                if metric_name in summary:
                    coerced = _extract_scalar_from_summary_value(summary[metric_name])
                    if coerced is not None:
                        if metric_name == "auc_by_min_task_step/val_seen":
                        #if metric_name == "auc_seen":
                            metric_values_seen.append(coerced)
                        elif metric_name == "auc_by_min_task_step/val_unseen_at_best_val_seen":
                        #elif metric_name == "auc_unseen":
                            metric_values_unseen_at_best_val_seen.append(coerced)
                
        
        if metric_values_seen and metric_values_unseen_at_best_val_seen:
            mean_val_seen = np.mean(metric_values_seen)
            std_val_seen = np.std(metric_values_seen)
            n_runs_seen = len(metric_values_seen)
            mean_val_unseen_at_best_val_seen = np.mean(metric_values_unseen_at_best_val_seen)
            std_val_unseen_at_best_val_seen = np.std(metric_values_unseen_at_best_val_seen)
            n_runs_unseen_at_best_val_seen = len(metric_values_unseen_at_best_val_seen)
            
            results.append({
                "group": group_name,
                "lr": lr,
                "lambda_reg": lambda_reg,
                "use_scheduler": use_scheduler,
                "scheduler_type": scheduler_type,
                "eta_min": eta_min,
                "mean_val_seen": mean_val_seen,
                "std_val_seen": std_val_seen,
                "n_runs_seen": n_runs_seen,
                "values_seen": metric_values_seen,
                "mean_val_unseen_at_best_val_seen": mean_val_unseen_at_best_val_seen,
                "std_val_unseen_at_best_val_seen": std_val_unseen_at_best_val_seen,
                "n_runs_unseen_at_best_val_seen": n_runs_unseen_at_best_val_seen,
                "values_unseen_at_best_val_seen": metric_values_unseen_at_best_val_seen
            })
    
    df = pd.DataFrame(results)
    return df.sort_values(["lr", "lambda_reg"])


def print_results_table(df, metric_name="auc_by_min_task_step/val_seen"):
    """Print a nicely formatted results table."""
    print(f"\n{'='*80}")
    print(f"Results for metric: {metric_name}")
    print(f"{'='*80}")
    print(f"\n{'LR':<12} {'Lambda Reg':<12} {'Mean':<10} {'Std':<10} {'N Runs':<8}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        print(f"{row['lr']:<12.6f} {row['lambda_reg']:<12.6f} "
              f"{row['mean']:<10.4f} {row['std']:<10.4f} {int(row['n_runs']):<8}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    metrics = ["auc_by_min_task_step/val_seen", "auc_by_min_task_step/val_unseen_at_best_val_seen"]
    #metrics = ["auc_seen", "auc_unseen"]
    # Create a summary table with all metrics
    print("\n" + "="*80)
    print("SUMMARY: Best hyperparameter settings")
    print("="*80)
    
    try:
        df_seen = get_group_statistics(PROJECT_NAME, metrics)
        csv_filename = f"{POLICY}_{PROJECT_NAME}_results.csv"

        if not df_seen.empty:
            df_seen.to_csv(csv_filename, index=False)
            best_idx = df_seen['mean_val_seen'].idxmax()
            best_row = df_seen.loc[best_idx]
            print("\nBest val_seen AUC:")
            print(f"  Group: {best_row['group']}")
            print(f"  LR: {best_row['lr']:.6f}")
            print(f"  Lambda Reg: {best_row['lambda_reg']:.6f}")
            print(f"  Use Scheduler: {best_row['use_scheduler']}")
            print(f"  Scheduler Type: {best_row['scheduler_type']}")
            print(f"  Eta Min: {best_row['eta_min']}")
            print(f"  Seen Mean AUC: {best_row['mean_val_seen']:.4f} ± {best_row['std_val_seen']:.4f}")
            print(f"  Unseen at Best Val Seen Mean AUC: {best_row['mean_val_unseen_at_best_val_seen']:.4f} ± {best_row['std_val_unseen_at_best_val_seen']:.4f}")
    except Exception as e:
        print(f"Error creating summary: {e}")

