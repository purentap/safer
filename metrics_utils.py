from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.metrics import roc_auc_score
from conformal.functional_predictor import (RegressionType,
    ModulationType, FunctionalPredictor)
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import torch

EVAL_TIMES = [
    "at earliest stop",
    "by earliest stop",
    "by final end",
]

def eval_roc_auc(scores_by_split_name, rollouts_by_split_name, debug=False, cfg=None):
    roc_curves_data = []
    time_quantiles = [0.25, 0.5, 0.75, 1.0] 
    auc_by_time = {}
    auc_by_min_task_step = {}
    logs = {}
    best_thresholds_by_split = {}
    for split, rollouts in rollouts_by_split_name.items():
        scores = scores_by_split_name[split]
        labels = [1-r.episode_success for r in rollouts]

        task_ids = sorted(list[int](set([rollout.task_id for rollout in rollouts])))

        # Compute ROC curves and AUC by time quantiles and by minimum task step. 
        auc_by_time_quantiles, _, _  = compute_roc_by_time_quantile(scores, rollouts, time_quantiles)
        auc_by_min_task_steps, best_threshold = compute_roc_by_min_task_step(scores, rollouts, labels,split,debug=True, threshold=True)
            
        auc_by_time[split] = auc_by_time_quantiles
        auc_by_min_task_step[split] = auc_by_min_task_steps
        for k, v in auc_by_time_quantiles.items():
            logs[f"auc_by_time_quantile_{k}/{split}"] = v
        
        logs[f"auc_by_min_task_step/{split}"] = auc_by_min_task_steps
        logs[f"best_threshold_by_min_task_step/{split}"] = best_threshold
        best_thresholds_by_split[split] = best_threshold
    return logs, best_thresholds_by_split

def compute_roc_by_min_task_step(scores, rollouts, labels, split=None,threshold=True, debug=False):
    #return a scalar
    scores = [s[:r.task_min_step].max() for s, r in zip(scores, rollouts)]
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    #find the best threshold according to Youden's J statistic
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

        plt.savefig(f"{split}_roc_curve.pdf", bbox_inches="tight")
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

    # calibration_rollouts = sum([rollouts_by_split_name[split] for split in calib_split_names], [])
    # test_rollouts = sum([rollouts_by_split_name[split] for split in test_split_names], [])

    # calibration_scores = sum([scores_by_split_name[split] for split in calib_split_names], [])
    # test_scores = sum([scores_by_split_name[split] for split in test_split_names], [])
    
    # test_labels_all = np.asarray([1-r.episode_success for r in test_rollouts])
    cal_rollouts, cal_scores_all = [], []
    for split_name in calib_split_names:
        cal_rollouts.extend(rollouts_by_split_name[split_name])
        cal_scores_all.extend(scores_by_split_name[split_name])
    cal_labels_all = np.asarray([1-r.episode_success for r in cal_rollouts])

    test_rollouts, test_scores_all = [], []
    for split_name in test_split_names:
        test_rollouts.extend(rollouts_by_split_name[split_name])
        test_scores_all.extend(scores_by_split_name[split_name])
    test_labels_all = np.asarray([1-r.episode_success for r in test_rollouts])
 
    # test_earliest_stop = np.array([r.task_min_step for r in test_rollouts]) # (N,)

    # if align_method == "extend":
    #     # Extend the early-stoping scores with the last value
    #     max_length = max(len(s) for s in calibration_scores + test_scores)
    #     for i, s in enumerate(calibration_scores):
    #         calibration_scores[i] = np.pad(s, (0, max_length - len(s)), mode='edge')
    #     for i, s in enumerate(test_scores):
    #         test_scores[i] = np.pad(s, (0, max_length - len(s)), mode='edge')
    
    test_earliest_stop = np.array([r.task_min_step for r in test_rollouts]) # (N,)
    if align_method == "extend":
        # Extend the early-stoping scores with the last value
        max_length = max(len(s) for s in cal_scores_all + test_scores_all)
        for i, s in enumerate(cal_scores_all):
            cal_scores_all[i] = np.pad(s, (0, max_length - len(s)), mode='edge')
        for i, s in enumerate(test_scores_all):
            test_scores_all[i] = np.pad(s, (0, max_length - len(s)), mode='edge')

    for calib_on in ['neg']: # Calibration on the successful rollouts
        lower_bound = False
        cal_scores_used = [s for s, r in zip(cal_scores_all, cal_rollouts) if r.episode_success == 1]

        # Split the cal scores evenly randomly into two set (for regression and modulation respectively)
        cal_scores_used = np.array(cal_scores_used)

        np.random.seed(42) # For reproducibility
        np.random.shuffle(cal_scores_used)
        n_cal_1 = int(len(cal_scores_used) * 0.3) # 30% according to Chen's implementation
        cal_scores_1 = cal_scores_used[:n_cal_1]
        cal_scores_2 = cal_scores_used[n_cal_1:]

        # Compute the conformal prediction band
        test_scores_all = np.array(test_scores_all) # (N, T)
        n_test_samples = len(test_scores_all)
        
        cp_bands_by_alpha = {}
        
        for eval_time in ['by earliest stop']:
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
                    print(f"eval_time: {eval_time}, lengths: {lengths}")
                
                elif eval_time == "by earliest stop":
                    lengths = test_earliest_stop # (N,)
                    print(f"eval_time: {eval_time}, lengths: {lengths}")
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
                    "cal split": f"{'+'.join(calib_split_names)}",
                    "test split": f"{'+'.join(test_split_names)}",
                    "calib on": calib_on,
                    "alpha" : alpha,
                    "time": eval_time,
                    "avg_det_time": avg_det_time,
                    "thresh_method" : "functional CP",
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

def eval_binary_classification(scores, labels, threshold):
    #taken from SAFE
    '''
    Compute the metrics for a binary classification task.
    Compute TPR, TNR, Accuracy, F1 Score based on the given threshold.
    Also compute the ROC AUC and PRC AUC, which are agnostic to the threshold.
    Properly handle the case where there is only one class in the labels.
    
    Args:
        scores: classifier scores, shape (n_samples,), higher score means more likely to be positive.
        labels: GT labels, shape (n_samples,), 1 means positive, 0 means negative.
        threshold: The threshold for the binary classification.
    
    Returns:
        dict: A dictionary of the computed metrics, with keys {tpr, tnr, accuracy, f1, roc_auc, prc_auc}.
    '''
    if isinstance(scores, list):
        scores = np.array(scores)
    if isinstance(labels, list):
        labels = np.array(labels)
    
    pos_freq = np.sum(labels) / len(labels)
    neg_freq = 1 - pos_freq

    # Generate binary predictions using the threshold.
    preds = (scores >= threshold).astype(int)

    # Calculate confusion matrix components.
    TP = np.sum((preds == 1) & (labels == 1))
    FP = np.sum((preds == 1) & (labels == 0))
    TN = np.sum((preds == 0) & (labels == 0))
    FN = np.sum((preds == 0) & (labels == 1))
    
    # Compute TPR (Recall) and TNR.
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    tnr = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    
    # Compute Accuracy.
    acc = (TP + TN) / len(labels) if len(labels) > 0 else 0.0
    bal_acc = (tpr + tnr) / 2
    weighted_acc = (tpr * neg_freq + tnr * pos_freq) # Weighted by the inverse class frequency
    
    # Compute Precision.
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    # Compute F1 Score.
    f1 = (2 * precision * tpr / (precision + tpr)) if (precision + tpr) > 0 else 0.0

    # Return the computed metrics.
    return {
        "tpr": tpr,
        "tnr": tnr,
        "fpr": fpr,
        "fnr": fnr,
        "acc": acc,
        "bal_acc": bal_acc,
        "f1": f1,
        "weighted-acc": weighted_acc,
        #"roc_auc": roc_auc,
        #"prc_auc": prc_auc,
    }
def eval_fixed_threshold(scores_by_split_name, rollouts_by_split_name, gate_vectors_by_split_name={}, thresholds=[0.5]):
    # classification with fixed threshold, 0.5
    classification_logs = []
    classification_per_task = {}
    for split_name in rollouts_by_split_name:
        rollouts = rollouts_by_split_name[split_name]
        scores_all = scores_by_split_name[split_name]
        gate_vectors = gate_vectors_by_split_name[split_name]
        labels = [1-r.episode_success for r in rollouts]
        for eval_time in EVAL_TIMES:
            if eval_time == "at earliest stop":
                scores = [s[-1] for s, r in zip(scores_all, rollouts)]
            elif eval_time == "by earliest stop":
                scores = [s[:r.task_min_step].max() for s, r in zip(scores_all, rollouts)]
            elif eval_time == "by final end":
                scores = [s[:len(r.action_embeddings)].max() for s, r in zip(scores_all, rollouts)]
            else:
                raise ValueError(f"Unknown eval_time: {eval_time}")
            
            if isinstance(thresholds, dict):
                thresh = thresholds[split_name]
                result = eval_binary_classification(scores, labels, thresh)
                classification_logs.append({
                    "split": split_name,
                    "eval_time": eval_time,
                    "threshold_method": "youdens_j",
                    "threshold": thresh,
                    **result
                })

            else: 
                for thresh in thresholds:
                    result = eval_binary_classification(scores, labels, thresh)
                    classification_logs.append({
                        "split": split_name,
                        "eval_time": eval_time,
                        "threshold_method": "fixed",
                        "threshold": thresh,
                        **result
                    })
    
    df = pd.DataFrame(classification_logs)
    return df, classification_per_task

def eval_split_conformal(rollouts_by_split_name, scores_by_split_name, method_name, alphas=None, calib_split_names = ["val_seen"], test_split_names = ["val_unseen"]):
    #taken from SAFE

    if alphas is None:
        alphas = [0.02] + [0.05 * i for i in range(1, 10)] + [0.5, 0.6, 0.7, 0.8, 0.9]

    classification_logs = []
    # Construct data for calibration and test sets
    cal_rollouts, cal_scores_all = [], []
    for split_name in calib_split_names:
        cal_rollouts.extend(rollouts_by_split_name[split_name])
        cal_scores_all.extend(scores_by_split_name[split_name])
    cal_labels = [1-r.episode_success for r in cal_rollouts]
    test_rollouts, test_scores_all = [], []
    for split_name in test_split_names:
        test_rollouts.extend(rollouts_by_split_name[split_name])
        test_scores_all.extend(scores_by_split_name[split_name])
    test_labels = [1-r.episode_success for r in test_rollouts]

    for eval_time in EVAL_TIMES:
        if eval_time == "at earliest stop":
            cal_scores = [s[r.task_min_step - 1] for s, r in zip(cal_scores_all, cal_rollouts)]
            test_scores = [s[r.task_min_step - 1] for s, r in zip(test_scores_all, test_rollouts)]
        elif eval_time == "by earliest stop":
            cal_scores = [s[:r.task_min_step].max() for s, r in zip(cal_scores_all, cal_rollouts)]
            test_scores = [s[:r.task_min_step].max() for s, r in zip(test_scores_all, test_rollouts)]
        elif eval_time == "by final end":
            cal_scores = [s[:len(r.action_embeddings)].max() for s, r in zip(cal_scores_all, cal_rollouts)]
            test_scores = [s[:len(r.action_embeddings)].max() for s, r in zip(test_scores_all, test_rollouts)]

        for alpha in alphas:
            thresholds = split_conformal_binary(cal_scores, cal_labels, test_scores, alpha)

            for calib_label in ['pos', 'neg']:
                if calib_label == 'pos': 
                    thresh_pos = 1 - thresholds[1]
                else: 
                    thresh_pos = thresholds[0]
                
                result = eval_binary_classification(test_scores, test_labels, thresh_pos)
                classification_logs.append({
                    "detect_method": method_name,
                    "cal split": f"{'+'.join(calib_split_names)}",
                    "test split": f"{'+'.join(test_split_names)}",
                    "calib on": calib_label,
                    "task": "all",
                    "thresh_method": f"split CP, cal on {'+'.join(calib_split_names)}",
                    "alpha": alpha,
                    "time": eval_time,
                    **result,
                    "threshold": thresh_pos,
                })
    classification_logs = pd.DataFrame(classification_logs)

    return classification_logs
        

def split_conformal_binary(cal_scores, cal_labels, test_scores, alpha):
    """
    Performs split conformal prediction for binary classification.
    
    For each calibration example, a nonconformity score is computed according to:
      - If the true label is 1 (positive):  alpha = 1 - s(x)
      - If the true label is 0 (negative):  alpha = s(x)
    
    Then, for each candidate label, a threshold is computed from the calibration set.
    For a test example with score s, the nonconformity scores are:
      - For candidate label 1: 1 - s
      - For candidate label 0: s
    
    The prediction set for the test example includes a label if its test nonconformity score is below the corresponding threshold.
    
    Args:
        cal_scores: 1D tensor of shape (N_cal,) containing scores (from e.g. a sigmoid) for calibration examples.
        cal_labels: 1D tensor of shape (N_cal,) containing true binary labels (0 or 1) for calibration examples.
        test_scores: 1D tensor of shape (N_test,) containing scores for test examples.
        alpha: Significance level (e.g., 0.1 for 90% coverage).
        
    Returns:
        A list of length N_test, where each element is a set containing one or both of the candidate labels (0 and/or 1).
    """
    #taken from SAFE
    if isinstance(cal_scores, list):
        cal_scores = torch.tensor(cal_scores)
    if isinstance(cal_labels, list):
        cal_labels = torch.tensor(cal_labels)
    if isinstance(test_scores, list):
        test_scores = torch.tensor(test_scores)
    
    # Compute thresholds for each candidate label
    thresholds = {}

    # For positive class (label 1): use nonconformity score = 1 - score.
    pos_mask = (cal_labels == 1)
    if pos_mask.sum() > 0:
        cal_pos_scores = cal_scores[pos_mask]
        # Compute nonconformity values for positive examples.
        cal_pos_nconf = 1 - cal_pos_scores
        threshold_pos = quantile_threshold(cal_pos_nconf, alpha)
        #print("threshold pos: " , threshold_pos)
        thresholds[1] = threshold_pos.item()
    else:
        thresholds[1] = float('inf')

        
    # For negative class (label 0): use nonconformity score = score.
    neg_mask = (cal_labels == 0)
    if neg_mask.sum() > 0:
        cal_neg_scores = cal_scores[neg_mask]
        # Nonconformity values for negative examples.
        cal_neg_nconf = cal_neg_scores
        threshold_neg = quantile_threshold(cal_neg_nconf, alpha)
        #print("threshold neg: " , threshold_neg)
        thresholds[0] = threshold_neg.item()
    else:
        thresholds[0] = float('inf')
    
    return thresholds


def quantile_threshold(scores, alpha):
    """
    Computes the threshold as the ceil((N+1)*(1 - alpha))-th smallest value of the provided scores.
    
    Args:
        scores: 1D tensor of nonconformity scores (for a given class).
        alpha: significance level (e.g., 0.1 means we want at least 90% coverage).
        
    Returns:
        A scalar tensor representing the threshold.
    """
    N = scores.numel()
    # Calculate rank: note that we need to use 1-indexing for the quantile
    k = int(torch.ceil(torch.tensor((N + 1) * (1 - alpha), dtype=torch.float)))
    k = np.clip(k, 1, N)  # Ensure k is within bounds
    sorted_scores, _ = torch.sort(scores)
    threshold = sorted_scores[k - 1]  # k-1 because of 0-indexing
    return threshold
