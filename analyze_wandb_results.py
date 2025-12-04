"""
Script to extract mean and std of success metrics from WandB runs grouped by hyperparameters.

This script queries WandB API to get aggregated statistics for each hyperparameter combination.
"""

import wandb
import pandas as pd
import numpy as np
from collections import defaultdict

# Initialize WandB API
api = wandb.Api()

# Set your project name
PROJECT_NAME = "deneme"  # Update this to match your project

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
            else:
                lr = config.get("learning_rate", config.get("lr", "N/A"))
                lambda_reg = config.get("lambda_reg", "N/A")
        except:
            lr = "N/A"
            lambda_reg = "N/A"
        
        # Extract metric values from summary (best value)
        metric_values = []
        for run in group_runs:
            # Try to get metric from summary
            summary = run.summary
            if metric_name in summary:
                metric_values.append(summary[metric_name])
            else:
                # Try alternative names
                alt_name = metric_name.replace("/", "_")
                if alt_name in summary:
                    metric_values.append(summary[alt_name])
        
        if metric_values:
            mean_val = np.mean(metric_values)
            std_val = np.std(metric_values)
            n_runs = len(metric_values)
            
            results.append({
                "group": group_name,
                "lr": lr,
                "lambda_reg": lambda_reg,
                "mean": mean_val,
                "std": std_val,
                "n_runs": n_runs,
                "values": metric_values
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
    # Analyze different metrics
    metrics_to_analyze = [
        "auc_by_min_task_step/val_seen",
        "auc_by_min_task_step/val_unseen",
        "auc_by_min_task_step/val_unseen_at_best_val_seen"
    ]
    
    for metric in metrics_to_analyze:
        try:
            df = get_group_statistics(PROJECT_NAME, metric)
            if not df.empty:
                print_results_table(df, metric)
                
                # Save to CSV
                csv_filename = f"results_{metric.replace('/', '_')}.csv"
                df.to_csv(csv_filename, index=False)
                print(f"Saved results to {csv_filename}\n")
            else:
                print(f"No data found for metric: {metric}\n")
        except Exception as e:
            print(f"Error analyzing {metric}: {e}\n")
    
    # Create a summary table with all metrics
    print("\n" + "="*80)
    print("SUMMARY: Best hyperparameter settings")
    print("="*80)
    
    try:
        df_seen = get_group_statistics(PROJECT_NAME, "auc_by_min_task_step/val_seen")
        if not df_seen.empty:
            best_idx = df_seen['mean'].idxmax()
            best_row = df_seen.loc[best_idx]
            print(f"\nBest val_seen AUC:")
            print(f"  LR: {best_row['lr']:.6f}")
            print(f"  Lambda Reg: {best_row['lambda_reg']:.6f}")
            print(f"  Mean AUC: {best_row['mean']:.4f} ± {best_row['std']:.4f}")
    except Exception as e:
        print(f"Error creating summary: {e}")

