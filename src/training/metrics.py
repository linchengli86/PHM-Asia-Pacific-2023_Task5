"""Evaluation metrics for all 5 tasks."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score,
)


def compute_all_metrics(preds, trues):
    """Compute all task metrics.

    Args:
        preds: dict with keys task1-5, numpy arrays
        trues: dict with keys task1-5, numpy arrays

    Returns:
        metrics: flat dict of scalar values
    """
    m = {}

    # task1: binary
    yt = trues["task1"].squeeze().astype(int)
    yp = preds["task1"].squeeze().astype(int)
    m["task1_acc"] = accuracy_score(yt, yp)
    m["task1_f1"] = f1_score(yt, yp, average="binary", zero_division=0)
    m["task1_prec"] = precision_score(yt, yp, average="binary", zero_division=0)
    m["task1_rec"] = recall_score(yt, yp, average="binary", zero_division=0)

    # task2: 4-class
    yt = trues["task2"].squeeze().astype(int)
    yp = preds["task2"].squeeze().astype(int)
    m["task2_acc"] = accuracy_score(yt, yp)
    m["task2_f1"] = f1_score(yt, yp, average="macro", zero_division=0)

    # task3: 9-class
    yt = trues["task3"].squeeze().astype(int)
    yp = preds["task3"].squeeze().astype(int)
    m["task3_acc"] = accuracy_score(yt, yp)
    m["task3_f1"] = f1_score(yt, yp, average="macro", zero_division=0)

    # task4: 5-class
    yt = trues["task4"].squeeze().astype(int)
    yp = preds["task4"].squeeze().astype(int)
    m["task4_acc"] = accuracy_score(yt, yp)
    m["task4_f1"] = f1_score(yt, yp, average="macro", zero_division=0)

    # task5: regression
    yt = trues["task5"].squeeze()
    yp = preds["task5"].squeeze()
    m["task5_mae"] = mean_absolute_error(yt, yp)
    m["task5_rmse"] = np.sqrt(mean_squared_error(yt, yp))
    m["task5_r2"] = r2_score(yt, yp)

    return m


def summarize_folds(fold_metrics_list):
    """Compute mean ± std across folds."""
    summary = {}
    for key in fold_metrics_list[0]:
        vals = [m[key] for m in fold_metrics_list]
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return summary
