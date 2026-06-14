import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import cv2
import math
import imageio
import os
import wandb
def predict_by_threshold(scores, labels,rollouts, split, threshold, cfg):
    print(split, "scores shape:", len(scores))
    n_test_samples = len(scores)
    labels = np.asarray(labels)
    #earliest_stop = np.array([r.task_min_step for r in rollouts])

    max_scores = [s[:r.task_min_step].max() for s, r in zip(scores, rollouts)]

    #max_length = max(len(s) for s in scores)

    #pad scores to the same length
    #for i, s in enumerate(scores):
    #    scores[i] = np.pad(s, (0, max_length - len(s)), mode='edge')
    
    detection_mask = max_scores >= threshold

    #by earliest stop
    #lengths = earliest_stop # (N,)
    # After the earliest stop, no more detection is possible. 
    #for i in range(len(scores)):
    #    detection_mask[i, lengths[i]:] = False

    #has_detection = np.any(detection_mask, axis=1) # (N,)
    #has_detection = np.any(detection_mask)
    has_detection = detection_mask

    pos_mask = labels == 1 # (N,)
    predicted = has_detection # (N,)
    tp = (predicted & pos_mask).sum()
    fn = (~predicted & pos_mask).sum()
    fp = (predicted & ~pos_mask).sum()
    tn = (~predicted & ~pos_mask).sum()

    # Safe division for metrics
    with np.errstate(divide='ignore', invalid='ignore'):
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        acc = (tp + tn) / n_test_samples
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        bal_acc = (tpr + tnr) / 2

    print(f"************** {split} **************")
    print(f"tp: {tp}, fn: {fn}, fp: {fp}, tn: {tn}")
    print(f"tpr: {tpr}, tnr: {tnr}, fpr: {fpr}, fnr: {fnr}")
    print(f"acc: {acc}, f1: {f1}, bal_acc: {bal_acc}")

    #build confusion matrix
    cm = np.array([[tp, fn], [fp, tn]])

    #save confusion matrix
    plt.figure()
    plt.imshow(cm, interpolation='nearest')

    plt.title("Confusion Matrix")

    plt.xticks([0, 1], ["Pred Positive", "Pred Negative"])
    plt.yticks([0, 1], ["Actual Positive", "Actual Negative"])

    # Add text labels inside cells
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j],
                    ha="center", va="center")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.colorbar()

    plt.savefig(f"{split}_confusion_matrix.png", dpi=300, bbox_inches="tight")

    #find misclassified samples
    misclassified_sample_ids = []
    for i in range(len(max_scores)):
        if predicted[i] != pos_mask[i]:
            misclassified_sample_ids.append(i)
    print(f"misclassified samples: {misclassified_sample_ids}")
    #find false positive samples 
    false_positive_sample_ids = []
    for i in range(len(max_scores)):
        if predicted[i] == 1 and pos_mask[i] == 0:
            false_positive_sample_ids.append(i)
    #print(f"false positive samples: {false_positive_sample_ids}")
    #find false negative samples
    false_negative_sample_ids = []
    for i in range(len(max_scores)):
        if predicted[i] == 0 and pos_mask[i] == 1:
            false_negative_sample_ids.append(i)
    #print(f"false negative samples: {false_negative_sample_ids}")
    # get task ids of misclassified samples
    task_ids = [rollouts[i].task_id for i in misclassified_sample_ids]
    print(f"task ids of misclassified samples: {task_ids}")
    false_positive_task_ids = [rollouts[i].task_id for i in false_positive_sample_ids]
    false_negative_task_ids = [rollouts[i].task_id for i in false_negative_sample_ids]
    #task id counts
    task_id_counts = Counter(task_ids)
    task_id_counts = dict(sorted(task_id_counts.items(), key=lambda x: x[0]))
    false_positive_task_id_counts = Counter(false_positive_task_ids)
    false_positive_task_id_counts = dict(sorted(false_positive_task_id_counts.items(), key=lambda x: x[0]))
    false_negative_task_id_counts = Counter(false_negative_task_ids)
    false_negative_task_id_counts = dict(sorted(false_negative_task_id_counts.items(), key=lambda x: x[0]))
    print(f"task id counts: {task_id_counts}")
    print(f"false positive task id counts: {false_positive_task_id_counts}")
    print(f"false negative task id counts: {false_negative_task_id_counts}")

    misclassified_samples=[]
    fp_samples=[]
    fn_samples=[]
    for i in misclassified_sample_ids:
        path = rollouts[i].mp4_path
        rollouts_score = scores[i]
        rollouts_max_score = max_scores[i]
        rollouts_label = labels[i]
        predicted_label = predicted[i]
        task_min_step = rollouts[i].task_min_step
        task_description = rollouts[i].task_description

        data = {
            "episode_idx": rollouts[i].episode_idx,
            "task_id": rollouts[i].task_id,
            "mp4_path": path,
            "rollouts_score": rollouts_score,
            "rollouts_max_score": rollouts_max_score,
            "rollouts_label": rollouts_label,
            "predicted_label": predicted_label,
            "task_min_step": task_min_step,
            "threshold": threshold,
            "task_description": task_description,
        }
        #check if fp 
        if predicted_label == 1 and rollouts_label == 0:
            fp_samples.append(data)
        #check if fn
        if predicted_label == 0 and rollouts_label == 1:
            fn_samples.append(data)

        misclassified_samples.append(data)
    
    df_all = pd.DataFrame(misclassified_samples)
    df_all.to_csv(f"./analiz_csv/seed{cfg.seed}_{split}_misclassified_samples.csv", index=False)
    df_fp = pd.DataFrame(fp_samples)
    df_fp.to_csv(f"./analiz_csv/seed{cfg.seed}_{split}_false_positive_samples.csv", index=False)
    df_fn = pd.DataFrame(fn_samples)
    df_fn.to_csv(f"./analiz_csv/seed{cfg.seed}_{split}_false_negative_samples.csv", index=False)



def save_predictions_to_csv(all_scores, failure_labels, rollouts, split_name, threshold):
    
    if split_name == "train":
        pass
    else:
        print(split_name, "scores shape:", len(all_scores))

        if isinstance(failure_labels, list):
            failure_labels = np.array(failure_labels)

        
        max_scores = [s[:r.task_min_step].max() for s, r in zip(all_scores, rollouts)]

        preds = (max_scores >= threshold).astype(int)

        pos_mask = failure_labels == 1 # (N,)

        print("failure labels: ", failure_labels)
        print("pos mask: " , pos_mask)
        print("preds", preds)
        #find misclassified samples
        tp_indices, tn_indices = [], []
        fn_indices, fp_indices = [],[]
        misclassified_sample_indices, correct_classified_sample_indices=[], []
        for i in range(len(max_scores)):
            # if pos_mask[i] == 1 and  pos_mask[i] == preds[i]: #TP
            #     tp_indices.append(i)
            # elif pos_mask[i] == 0 and pos_mask[i] == preds[i]: #TN
            #     tn_indices.append(i)
            
            # elif pos_mask[i]== 0 and preds[i] == 1: #FP
            #     fp_indices.append[i]
            
            # elif pos_mask[i] == 1 and preds[i] == 0: #FN 
            #     fn_indices.append(i)
            if pos_mask[i] != preds[i]:
                misclassified_sample_indices.append(i)

            else: 
                correct_classified_sample_indices.append(i)

        misclassified_samples=[]
        correct_classified_samples=[]
        for i in misclassified_sample_indices:
            path = rollouts[i].mp4_path
            rollouts_score = all_scores[i]
            rollouts_max_score = max_scores[i]
            rollouts_label = failure_labels[i]
            predicted_label = preds[i]
            task_min_step = rollouts[i].task_min_step
            task_description = rollouts[i].task_description

            data = {
                "episode_idx": rollouts[i].episode_idx,
                "task_id": rollouts[i].task_id,
                "mp4_path": path,
                "all_scores": rollouts_score,
                "rollouts_max_score": rollouts_max_score,
                "rollouts_failure_label": rollouts_label,
                "predicted_failure_label": predicted_label,
                "task_min_step": task_min_step,
                "threshold": threshold,
                "task_description": task_description,
            }
            misclassified_samples.append(data)
        
        for i in correct_classified_sample_indices:
            path = rollouts[i].mp4_path
            rollouts_score = all_scores[i]
            rollouts_max_score = max_scores[i]
            rollouts_label = failure_labels[i]
            predicted_label = preds[i]
            task_min_step = rollouts[i].task_min_step
            task_description = rollouts[i].task_description

            data = {
                "episode_idx": rollouts[i].episode_idx,
                "task_id": rollouts[i].task_id,
                "mp4_path": path,
                "all_scores": rollouts_score,
                "rollouts_max_score": rollouts_max_score,
                "rollouts_failure_label": rollouts_label,
                "predicted_failure_label": predicted_label,
                "task_min_step": task_min_step,
                "threshold": threshold,
                "task_description": task_description,
            }
            correct_classified_samples.append(data)
        
        df_misclassified = pd.DataFrame(misclassified_samples)
        df_correctly_classified = pd.DataFrame(correct_classified_samples)
        df_misclassified.to_csv(f"./analiz_csv/tez_latest/seed{1}_{split_name}_misclassified_samples.csv", index=False)
        df_correctly_classified.to_csv(f"./analiz_csv/tez_latest/seed{1}_{split_name}_correctly_classified.csv", index=False)
def read_wandb_table(project_name):
    api = wandb.Api()
    runs = api.runs(project_name)
    print(runs)
    eval_types = ["at_earliest_stop", "by_earliest_stop", "by_final_end"]
    all_dfs=[]
    for run in runs:
        #run_summary = run.summary.get("classify_fixed_threshold/")
        #print(run_summary)
        #summary_dict = dict(run_summary)

        # artifact_path = summary_dict["_latest_artifact_path"]
        # print(artifact_path)
        #artifact = api.artifact(artifact_path)
        #print(artifact)
        for artifact in run.logged_artifacts():
            if "classify_best_threshold" in artifact.name: #   classify_fixed_threshold: for our code with t =0.5
                                                            # best_fixed_threshold_classification: for safe reprod with t= 0.5 
                                                            # classify_best_threshold: for our code with t obtained by youden's j statistics.
                latest_name = artifact.name.split(":")[0] + ":latest"  # strip version, pin to latest
                artifact = api.artifact(f"{run.entity}/{run.project}/{latest_name}")
                table = artifact.get("classify_best_threshold")
                df = pd.DataFrame(data=table.data, columns=table.columns)
                all_dfs.append(df)
                break

        
    stacked_df = pd.concat(all_dfs, ignore_index=True)
    # Mean over the 5 runs, grouped by split + eval_time
    metric_cols = [ "acc", "bal_acc", "f1", "weighted-acc"]
    mean_std_df = (
        stacked_df
        .groupby(["split", "eval_time"])[metric_cols] #eval_time for safer or time for safe
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
def read_frames_from_path(mp4_path):
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {mp4_path}")
        return None
    
    frame_count = 0 
    frames= []
    while True: 
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read the frame.")
            break
        
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #cv2.imwrite(f"./analiz_videos/frame_{frame_count}.jpg", frame)
        frames.append(frame)
        frame_count += 1 


    cap.release()
    return frames

def create_videos_with_red_border(csv_path, output_path, sub_type, seed):
    df = pd.read_csv(csv_path)
    for index, row in df.iterrows():
        mp4_path = row["mp4_path"]
        print(mp4_path)
        

        frames= read_frames_from_path(mp4_path)
        #print(len(frames))

        # read the model output probabilties 
        model_scores = row["all_scores"]

        model_scores = np.fromstring(model_scores.strip('[]'), dtype=float, sep=' ').tolist()
        print(len(model_scores) * 8)
        print(len(frames))
        #dataset has 1 extra frame, we need to remove it
        if math.ceil( (len(frames)-1) /8) == len(model_scores):
            frames = frames[:-1]
        

        threshold = row["threshold"]
        mp4_path = mp4_path.split("/")[-4:]
        mp4_path = "/".join(mp4_path)
        print(mp4_path)
        # For each frame in the raw video, create a figure that shows:
        # - The current RGB frame.
        # - A plot of the predicted score up to that time step.
        # Taken from the original SAFE repository
        exec_horizon = 8
        frames_to_plot = []
        for j in range(len(frames)):
            fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=120)
            # Left subplot: RGB frame
            ax = axes[0]
            ax.imshow(frames[j][:, :, ::-1])
            ax.axis("off")
            ax.set_title(f"RGB obs frame {j}")

            # Right subplot: score progression
            ax = axes[1]
            # Calculate how many scores to plot: one per exec_horizon steps.
            score_plot_end = j // exec_horizon + 1

            ax.plot(model_scores[:score_plot_end], label="Current rollout", color="blue", lw=2)
            ax.set_xlim(0, len(model_scores))
            #show the threshold line
            ax.axhline(threshold, color='red', linestyle='--', label='threshold')
            ax.legend()
            ax.set_title("Predicted failure score")
            ax.set_xlabel("Time step")
            ax.set_ylabel("Score")

            # draw a vertical line
            ax.axvline(row["task_min_step"], color='black', linestyle='--', label='earliest termination')

            # fig.suptitle(
            # f"{r.task_description}\n , Succ {r.episode_success} Final score {model_scores[-1]:.2f}"
            # )
            fig.suptitle(f"{row['task_description']}\n Task id: {row['task_id']}\n Label: {row['rollouts_failure_label']}\n Predicted: {row['predicted_failure_label']}\n Max score: {row['rollouts_max_score']:.2f}")
            fig.tight_layout()
            # Draw the canvas and convert the figure to a numpy array.
            fig.canvas.draw()
            plot_img = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)
            frames_to_plot.append(plot_img)

        # Save the list of frames as an mp4 video.

        save_path = os.path.join(
            output_path, f"seed{seed}",sub_type,
            f"Task{row['task_id']}_ep{row['episode_idx']}_succ{row['rollouts_failure_label']}_pred{row['predicted_failure_label']}.mp4",
        )
        #save_path = "./single_example.mp4"
        imageio.mimsave(save_path, frames_to_plot, fps=10)
        # for i in range(len(model_scores)):
        #     if model_scores[i] >= threshold:
        #         frames[i] = cv2.rectangle(frames[i], (0, 0), (640, 480), (0, 0, 255), 2)
        #     else:
        #         frames[i] = cv2.rectangle(frames[i], (0, 0), (640, 480), (0, 255, 0), 2)
        
        # #save as a video 
        # out_video_path = f"{output_path}/{row['task_id']}.mp4"
        # out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frames[0].shape[1], frames[0].shape[0]))
        # for frame in frames:
        #     out.write(frame)
        # out.release()
        

if __name__ == "__main__":
    path = "/home/ai/puren/research/fail-detection-embeddings/analiz_csv/tez_latest/seed1_val_unseen_correctly_classified.csv"
    out_path = "./analiz_videos/tez_latest"
    #sub_type = "unseen/fn"
    sub_type="val_unseen/correctly_classified"
    seed = 1
    project = "pi0fast_droid_fusion_lstmv2_youdensj"
    #read_wandb_table(project)
    create_videos_with_red_border(path, out_path, sub_type, seed)