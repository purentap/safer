import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import cv2
import math
import imageio
import os
import wandb
import ast
import json
from matplotlib.ticker import StrMethodFormatter


def read_wandb_table(project_name):
    api = wandb.Api()
    runs = api.runs(project_name)
    print(runs)
    eval_types = ["at_earliest_stop", "by_earliest_stop", "by_final_end"]
    all_dfs=[]
    for run in runs:
        for artifact in run.logged_artifacts():
            if "classify_split_cp" in artifact.name: #   classify_fixed_threshold: for our code with t =0.5
                                                            # best_fixed_threshold_classification: for safe reprod with t= 0.5 
                                                            # classify_best_threshold: for our code with t obtained by youden's j statistics.
                latest_name = artifact.name.split(":")[0] + ":latest"  # strip version, pin to latest
                artifact = api.artifact(f"{run.entity}/{run.project}/{latest_name}")
                table = artifact.get("classify_split_cp")
                df = pd.DataFrame(data=table.data, columns=table.columns)
                all_dfs.append(df)
                break

        
    stacked_df = pd.concat(all_dfs, ignore_index=True)
    filtered_df = stacked_df[(stacked_df["calib on"]== "neg") & (stacked_df["time"] == "by earliest stop")]
    print(filtered_df)
    # Mean over the 5 runs, grouped by split + eval_time
    metric_cols = [ "acc", "bal_acc", "f1", "weighted-acc"]
    mean_std_df = (
        stacked_df
        .groupby(["alpha"])[metric_cols] #eval_time for safer or time for safe
        .agg(["mean", "std"])
        .reset_index()
    )
    mean_std_df.columns = [
    f"{col}_{stat}" if stat else col 
    for col, stat in mean_std_df.columns
    ]   
    print(mean_std_df.columns)
    mean_std_df["bal_acc_pct"] = mean_std_df.apply(
    lambda r: f"{r['bal_acc_mean']*100:.2f} ± {r['bal_acc_std']*100:.2f}", axis=1
)
    mean_std_df["f1_pct"] = mean_std_df.apply(
    lambda r: f"{r['f1_mean']*100:.2f} ± {r['f1_std']*100:.2f}", axis=1
    )

    print(mean_std_df)

    bal_acc_means = mean_std_df["bal_acc_mean"].values
    alphas = mean_std_df["alpha"].values

    print(bal_acc_means)
    print(alphas)

        # Create plot
    plt.figure(figsize=(7, 5))

    plt.plot(
        alphas,
        bal_acc_means,
        marker="o",
        linewidth=2,
        markersize=6
    )

    # Labels
    plt.xlabel(r"$\alpha$", fontsize=13)
    plt.ylabel("Balanced Accuracy", fontsize=13)

    # Optional: show values above points
    for x, y in zip(alphas, bal_acc_means):
        plt.text(
            x,
            y + 0.01,
            f"{y:.2f}",
            ha="center",
            fontsize=8
        )

    # Grid
    plt.grid(True, alpha=0.3)

    # Keep accuracy in a sensible range
    plt.ylim(0.5, 1.0)

    plt.tight_layout()
    plt.show()
    plt.savefig(f"split_cp.pdf", bbox_inches="tight")

if __name__ == "__main__":
    project_name = "split_cp_pi0fast_droid"
    read_wandb_table(project_name)