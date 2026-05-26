import wandb
import pandas as pd
import numpy as np
import natsort
import re
from collections import defaultdict

# Initialize WandB API
api = wandb.Api()

# Set your project name
PROJECT_NAME = "pi0fast_droid_task_analiz"  # Update this to match your project

def get_group_statistics(project_name, metric_name="auc_by_min_task_step/val_seen"):
    runs = api.runs(project_name)
    
    # Group runs by their group name
    groups = defaultdict(list)
    for run in runs:
        if run.group:  # Only process runs that have a group
            groups[run.group].append(run)
    
    print(groups)
    for group_name, group_runs in groups.items():
        for run in group_runs:
            name = run.name
            print(name)
            summary = run.summary
            best_epoch = summary.get("best_epoch", None)
            #best_epoch += 1 #epoch is 0-indexed in training code
            print(best_epoch)
            #extract key information
            keys = []
            for key, item in summary.items():
                if key.startswith("auc_by_min_task_step/")  and re.search(r'_task_\d+$', key):
                    keys.append(key)
            keys = natsort.natsorted(keys)
            print(keys)
            history = run.history(keys=keys)
            print(history.iloc[best_epoch])
            #task_metrics = history.iloc[[best_epoch]]
            task_metrics = history.iloc[best_epoch].to_frame().T
            #print(history.iloc[best_epoch])

            #save as csv 
            task_metrics.to_csv(f"./task_analysis/{name}.csv", index=False)
            

if __name__ == "__main__":
    get_group_statistics(PROJECT_NAME)