"""
Script to pull WandB experiments and analyze performance by group.
For each group, computes the max AUC value for each seed, then averages across seeds.
Identifies the best performing groups for both val_seen and val_unseen metrics.
"""

import wandb
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# Initialize WandB API
api = wandb.Api()

# Set your project name - update this to match your project
PROJECT_NAME = "img_embeddings_auc_sweep"  # Update this if needed
POLICY = "OPENVLA"

def get_max_metric_from_history(run, metric_name):
    """
    Get the maximum value of a metric from a run's history.
    
    Args:
        run: WandB run object
        metric_name: Name of the metric (e.g., "auc_by_min_task_step/val_seen")
    
    Returns:
        Maximum value of the metric (as a scalar), or None if not found
    """
    try:
        # First try to get from summary (which should have max if defined correctly)
        if metric_name in run.summary:
            summary_val = run.summary[metric_name]
            # Handle case where summary might be a dict-like object (SummarySubdict) with "max" key
            # Check if it's a dict-like object (dict or SummarySubdict)
            type_name = type(summary_val).__name__
            if type_name in ['dict', 'SummarySubdict'] or hasattr(summary_val, '__getitem__'):
                try:
                    # Try to access "max" key
                    if "max" in summary_val:
                        return summary_val["max"]
                except (TypeError, KeyError):
                    # If "max" not found or not accessible, try get() method
                    try:
                        if hasattr(summary_val, 'get'):
                            max_val = summary_val.get("max")
                            if max_val is not None:
                                return max_val
                    except:
                        pass
            # If no "max" key, return the value directly (it's already a scalar)
            return summary_val
        
        # If not in summary, fetch from history
        history = run.history(keys=[metric_name])
        if not history.empty and metric_name in history.columns:
            max_val = history[metric_name].max()
            if not pd.isna(max_val):
                return float(max_val)
        
        # Try alternative metric name format
        alt_name = metric_name.replace("/", "_")
        if alt_name in run.summary:
            summary_val = run.summary[alt_name]
            type_name = type(summary_val).__name__
            if type_name in ['dict', 'SummarySubdict'] or hasattr(summary_val, '__getitem__'):
                try:
                    if "max" in summary_val:
                        return summary_val["max"]
                except (TypeError, KeyError):
                    try:
                        if hasattr(summary_val, 'get'):
                            max_val = summary_val.get("max")
                            if max_val is not None:
                                return max_val
                    except:
                        pass
            return summary_val
            
    except Exception as e:
        print(f"Warning: Could not get {metric_name} from run {run.name}: {e}")
    
    return None


def analyze_group_performance(project_name):
    """
    Analyze performance for all groups in a WandB project.
    
    For each group:
    - Finds runs with seed_0, seed_1, seed_2
    - Gets max auc_by_min_task_step/val_seen for each seed
    - Averages the max values across seeds
    - Does the same for auc_by_min_task_step/val_unseen
    
    Returns:
        DataFrame with group performance metrics
    """
    print(f"Fetching runs from project: {project_name}")
    runs = api.runs(project_name)
    print(f"Found {len(runs)} total runs")
    
    # Group runs by their group name
    groups = defaultdict(list)
    for run in runs:
        if run.group:  # Only process runs that have a group
            groups[run.group].append(run)
    
    print(f"Found {len(groups)} groups")
    
    results = []
    
    for group_name, group_runs in tqdm(groups.items(), desc="Processing groups"):
        # Extract hyperparameters from config
        try:
            first_run = group_runs[0]
            config = first_run.config
            
            # Handle nested config structure
            if "training" in config:
                lr = config["training"].get("learning_rate", config["training"].get("lr", "N/A"))
                lambda_reg = config["training"].get("lambda_reg", "N/A")
                use_scheduler = config["training"].get("use_scheduler", False)
            else:
                lr = config.get("learning_rate", config.get("lr", "N/A"))
                lambda_reg = config.get("lambda_reg", "N/A")
                use_scheduler = config.get("use_scheduler", False)
            
            # Get scheduler info if available
            if "scheduler" in config and use_scheduler:
                scheduler_type = config["scheduler"].get("type", "N/A")
                eta_min = config["scheduler"].get("eta_min", "N/A")
            else:
                scheduler_type = "None"
                eta_min = "N/A"
                
        except Exception as e:
            print(f"Warning: Could not extract config for group {group_name}: {e}")
            lr = "N/A"
            lambda_reg = "N/A"
            scheduler_type = "N/A"
            eta_min = "N/A"
        
        # Organize runs by seed
        runs_by_seed = {}
        for run in group_runs:
            run_name = run.name
            # Extract seed from run name (e.g., "seed_0", "seed_1", "seed_2")
            if "seed_" in run_name:
                try:
                    seed = int(run_name.split("seed_")[1].split("_")[0])
                    runs_by_seed[seed] = run
                except:
                    pass
        
        # Get max values for each seed
        val_seen_maxes = []
        val_unseen_maxes = []
        
        for seed in [0, 1, 2]:
            if seed in runs_by_seed:
                run = runs_by_seed[seed]
                
                # Get max val_seen
                max_val_seen = get_max_metric_from_history(run, "auc_by_min_task_step/val_seen")
                if max_val_seen is not None:
                    val_seen_maxes.append(max_val_seen)
                
                # Get max val_unseen
                max_val_unseen = get_max_metric_from_history(run, "auc_by_min_task_step/val_unseen")
                if max_val_unseen is not None:
                    val_unseen_maxes.append(max_val_unseen)
        
        # Calculate averages
        avg_val_seen = np.mean(val_seen_maxes) if val_seen_maxes else None
        std_val_seen = np.std(val_seen_maxes) if len(val_seen_maxes) > 1 else 0.0
        avg_val_unseen = np.mean(val_unseen_maxes) if val_unseen_maxes else None
        std_val_unseen = np.std(val_unseen_maxes) if len(val_unseen_maxes) > 1 else 0.0
        
        # Only add to results if we have at least one valid metric
        if avg_val_seen is not None or avg_val_unseen is not None:
            results.append({
                "group": group_name,
                "lr": lr,
                "lambda_reg": lambda_reg,
                "scheduler": scheduler_type,
                "eta_min": eta_min,
                "avg_val_seen": avg_val_seen,
                "std_val_seen": std_val_seen,
                "n_seeds_seen": len(val_seen_maxes),
                "avg_val_unseen": avg_val_unseen,
                "std_val_unseen": std_val_unseen,
                "n_seeds_unseen": len(val_unseen_maxes),
                "val_seen_maxes": val_seen_maxes,
                "val_unseen_maxes": val_unseen_maxes,
            })
    
    df = pd.DataFrame(results)
    return df


def print_results_table(df):
    """Print a nicely formatted results table."""
    print(f"\n{'='*120}")
    print("GROUP PERFORMANCE SUMMARY")
    print(f"{'='*120}")
    print(f"\n{'Group':<40} {'LR':<10} {'Lambda':<10} {'Avg Val_Seen':<15} {'Avg Val_Unseen':<15} {'Seeds':<8}")
    print("-" * 120)
    
    # Sort by average val_seen (descending)
    df_sorted = df.sort_values("avg_val_seen", ascending=False, na_position='last')
    
    for _, row in df_sorted.iterrows():
        group_short = row['group'][:37] + "..." if len(row['group']) > 40 else row['group']
        lr_str = f"{row['lr']:.2e}" if isinstance(row['lr'], (int, float)) else str(row['lr'])
        lambda_str = f"{row['lambda_reg']:.2e}" if isinstance(row['lambda_reg'], (int, float)) else str(row['lambda_reg'])
        
        seen_str = f"{row['avg_val_seen']:.4f}±{row['std_val_seen']:.4f}" if row['avg_val_seen'] is not None else "N/A"
        unseen_str = f"{row['avg_val_unseen']:.4f}±{row['std_val_unseen']:.4f}" if row['avg_val_unseen'] is not None else "N/A"
        seeds_str = f"{int(row['n_seeds_seen'])}/{int(row['n_seeds_unseen'])}"
        
        print(f"{group_short:<40} {lr_str:<10} {lambda_str:<10} {seen_str:<15} {unseen_str:<15} {seeds_str:<8}")
    
    print(f"\n{'='*120}\n")


def find_best_groups(df):
    """Find and display the best performing groups."""
    print("\n" + "="*120)
    print("BEST PERFORMING GROUPS")
    print("="*120)
    
    # Best by val_seen
    df_seen = df[df['avg_val_seen'].notna()].copy()
    if not df_seen.empty:
        best_seen_idx = df_seen['avg_val_seen'].idxmax()
        best_seen = df_seen.loc[best_seen_idx]
        
        print(f"\n🏆 Best val_seen AUC:")
        print(f"   Group: {best_seen['group']}")
        print(f"   LR: {best_seen['lr']}")
        print(f"   Lambda Reg: {best_seen['lambda_reg']}")
        print(f"   Scheduler: {best_seen['scheduler']}")
        if best_seen['eta_min'] != "N/A":
            print(f"   Eta Min: {best_seen['eta_min']}")
        print(f"   Val Seen AUC: {best_seen['avg_val_seen']:.4f} ± {best_seen['std_val_seen']:.4f}")
        print(f"   Individual Val Seen values: {best_seen['val_seen_maxes']}")
        if best_seen['avg_val_unseen'] is not None:
            print(f"   Val Unseen AUC: {best_seen['avg_val_unseen']:.4f} ± {best_seen['std_val_unseen']:.4f}")
    


if __name__ == "__main__":
    # Analyze the project
    df = analyze_group_performance(PROJECT_NAME)
    
    
    if df.empty:
        print("No results found. Please check:")
        print(f"  1. Project name is correct: {PROJECT_NAME}")
        print("  2. Runs have group names assigned")
        print("  3. Runs have seed_0, seed_1, seed_2 naming")
        print("  4. Metrics are logged as 'auc_by_min_task_step/val_seen' and 'auc_by_min_task_step/val_unseen'")
    else:
        # Print results table
        print_results_table(df)
        
        # Find best groups
        find_best_groups(df)
        
        # Save to CSV
        csv_filename = f"{POLICY}_{PROJECT_NAME}_wandb_group_analysis.csv"
        # Remove list columns before saving
        df_to_save = df.drop(columns=['val_seen_maxes', 'val_unseen_maxes'])
        df_to_save.to_csv(csv_filename, index=False)
        print(f"✅ Results saved to {csv_filename}")
    