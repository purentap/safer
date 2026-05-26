from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.metrics import roc_auc_score
from conformal.functional_predictor import (RegressionType,
    ModulationType, FunctionalPredictor)
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import analiz_utils


def eval_roc_auc(scores_by_split_name, rollouts_by_split_name, debug=False, cfg=None):
    roc_curves_data = []
    time_quantiles = [0.25, 0.5, 0.75, 1.0] 
    auc_by_time = {}
    auc_by_min_task_step = {}
    logs = {}

    for split, rollouts in rollouts_by_split_name.items():
        scores = scores_by_split_name[split]
        labels = [1-r.episode_success for r in rollouts]

        task_ids = sorted(list[int](set([rollout.task_id for rollout in rollouts])))

        # Compute ROC curves and AUC by time quantiles and by minimum task step. 
        auc_by_time_quantiles, _, _  = compute_roc_by_time_quantile(scores, rollouts, time_quantiles)
        auc_by_min_task_steps, best_threshold = compute_roc_by_min_task_step(scores, rollouts, labels,split,debug=True, threshold=True)
        if debug:
            analiz_utils.predict_by_threshold(scores, labels,rollouts, split, best_threshold, cfg)
        #compute auc by task id
        for task_id in task_ids:
            indices_task = [i for i, r in enumerate(rollouts) if r.task_id == task_id]
            rollouts_task = [rollouts[i] for i in indices_task]
            scores_task = [scores[i] for i in indices_task]
            labels_task = [1-r.episode_success for r in rollouts_task]
            auc_by_task_id = compute_roc_by_min_task_step(scores_task, rollouts_task, labels_task)
            logs[f"auc_by_min_task_step/{split}_task_{task_id}"] = auc_by_task_id
            
        auc_by_time[split] = auc_by_time_quantiles
        auc_by_min_task_step[split] = auc_by_min_task_steps
        for k, v in auc_by_time_quantiles.items():
            logs[f"auc_by_time_quantile_{k}/{split}"] = v
        
        logs[f"auc_by_min_task_step/{split}"] = auc_by_min_task_steps
    return logs

def compute_roc_by_min_task_step(scores, rollouts, labels, split=None,threshold=False, debug=False):
    #return a scalar
    scores = [s[:r.task_min_step].max() for s, r in zip(scores, rollouts)]
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
        

    if threshold:
        # Plot
        plt.figure()
        plt.plot(fpr, tpr, label=f"{split} ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle='--')  # Random classifier line

        #plot best threshold according to Youden's J
        plt.scatter(fpr[best_idx], tpr[best_idx])
        plt.text(fpr[best_idx] + 0.02, tpr[best_idx] - 0.05,    f"Best threshold = {best_threshold:.3f}", fontsize=10)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{split} ROC Curve")
        plt.legend(loc="lower right")

        plt.savefig(f"{split}_roc_curve.png", dpi=300, bbox_inches="tight")
    return roc_auc, best_threshold
def compute_roc_by_time_quantile(scores, rollouts, time_quantiles):
    fpr_by_time = {}
    tpr_by_time = {}
    auc_by_time = {}

    for q in time_quantiles:
        success_scores = [
            s[ round((r.task_min_step - 1) * q) ] 
            for s, r in zip(scores, rollouts)
            if r.episode_success == 1
        ]
        fail_scores = [
            s[ round((r.task_min_step - 1) * q) ] 
            for s, r in zip(scores, rollouts)
            if r.episode_success == 0
        ]
        
        fpr, tpr, roc_auc = compute_roc(success_scores, fail_scores)
        
        fpr_by_time[q] = fpr
        tpr_by_time[q] = tpr
        auc_by_time[q] = roc_auc
    
    return auc_by_time, fpr_by_time, tpr_by_time

# Doing failure detection, 1 means failure, 0 means success
def compute_roc(success_scores, fail_scores):
    y_true = [1] * len(fail_scores) + [0] * len(success_scores)
    y_score = fail_scores + success_scores
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc

def eval_functional_conformal(scores_by_split_name,
                                rollouts_by_split_name,
                                calib_split_names = ["val_seen"],
                                test_split_names = ["val_unseen"],
                                align_method = "extend"
                                ):
    alphas = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    classification_logs = []

    calibration_rollouts = sum([rollouts_by_split_name[split] for split in calib_split_names], [])
    test_rollouts = sum([rollouts_by_split_name[split] for split in test_split_names], [])

    calibration_scores = sum([scores_by_split_name[split] for split in calib_split_names], [])
    test_scores = sum([scores_by_split_name[split] for split in test_split_names], [])
    
    test_labels_all = np.asarray([1-r.episode_success for r in test_rollouts])

 
    test_earliest_stop = np.array([r.task_min_step for r in test_rollouts]) # (N,)

    if align_method == "extend":
        # Extend the early-stoping scores with the last value
        max_length = max(len(s) for s in calibration_scores + test_scores)
        for i, s in enumerate(calibration_scores):
            calibration_scores[i] = np.pad(s, (0, max_length - len(s)), mode='edge')
        for i, s in enumerate(test_scores):
            test_scores[i] = np.pad(s, (0, max_length - len(s)), mode='edge')
    
    for calib_on in ['neg']: # Calibration on the successful rollouts
        lower_bound = False
        cal_scores_used = [s for s, r in zip(calibration_scores, calibration_rollouts) if r.episode_success == 1]

        # Split the cal scores evenly randomly into two set (for regression and modulation respectively)
        cal_scores_used = np.array(cal_scores_used)

        np.random.shuffle(cal_scores_used)
        n_cal_1 = int(len(cal_scores_used) * 0.3) # 30% according to Chen's implementation
        cal_scores_1 = cal_scores_used[:n_cal_1]
        cal_scores_2 = cal_scores_used[n_cal_1:]

        # Compute the conformal prediction band
        test_scores_all = np.array(test_scores) # (N, T)
        n_test_samples = len(test_scores_all)
        
        cp_bands_by_alpha = {}
        
        for eval_time in ['by final end', 'by earliest stop']:
            for alpha in alphas:
                predictor = FunctionalPredictor(ModulationType.Tfunc, RegressionType.Mean)
                cp_band = predictor.get_one_sided_prediction_band(
                    cal_scores_1, cal_scores_2, alpha, lower_bound=lower_bound)
                
                cp_bands_by_alpha[alpha] = cp_band

                # Flag is raised if the test score is out of the band. 
                if lower_bound: detection_mask = test_scores_all <= cp_band # (N, T)
                else:           detection_mask = test_scores_all >= cp_band # (N, T)

                # Handle different evaluation time modes    
                if eval_time == "by final end":
                    lengths = test_scores_all.shape[1] # scalar, T
                
                elif eval_time == "by earliest stop":
                    lengths = test_earliest_stop # (N,)
                    # After the earliest stop, no more detection is possible. 
                    for i in range(len(test_scores_all)):
                        detection_mask[i, lengths[i]:] = False
                
                            
                has_detection = np.any(detection_mask, axis=1) # (N,)
                first_detection = np.argmax(detection_mask, axis=1) # (N,)
                detection_times = np.where(has_detection, first_detection, lengths) # (N,)
                relative_detection_times = detection_times / lengths # (N,)

                # Compute detection time and classification metrics
                pos_mask = test_labels_all == 1 # (N,)
                avg_det_time = np.mean(relative_detection_times[pos_mask])
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

                classification_logs.append({
                    "alpha" : alpha,
                    "time": eval_time,
                    "avg_det_time": avg_det_time,
                    "tpr": tpr,
                    "tnr": tnr,
                    "fpr": fpr,
                    "fnr": fnr,
                   "acc": acc,
                   "f1": f1,
                   "bal_acc": bal_acc
                })
    df = pd.DataFrame(classification_logs)

    return df