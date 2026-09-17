"""Raw data loading — no augmentation, no leakage.

CRITICAL: Test data (answer.csv) is NEVER loaded during training.
It is only loaded by evaluate.py when explicitly authorized.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path


def load_single_csv(case_id, data_dir):
    """Load one Case CSV, return (T, 7) float32 array."""
    fname = f"Case{int(case_id):03d}.csv"
    df = pd.read_csv(os.path.join(data_dir, fname))
    return df[["P1", "P2", "P3", "P4", "P5", "P6", "P7"]].values.astype(np.float32)


def load_raw_data(data_root, train_ids, test_ids):
    """Load all train and test time series.

    Args:
        data_root: path to dataset/ folder
        train_ids: list of ints, e.g. range(1, 178)
        test_ids: list of ints, e.g. range(178, 224)

    Returns:
        X_train: (N_train, 1201, 7)
        X_test: (N_test, 1201, 7)
    """
    train_dir = os.path.join(data_root, "train", "data")
    test_dir = os.path.join(data_root, "test", "data")

    X_train = np.stack([load_single_csv(cid, train_dir) for cid in train_ids], axis=0)
    X_test = np.stack([load_single_csv(cid, test_dir) for cid in test_ids], axis=0)

    return X_train.astype(np.float32), X_test.astype(np.float32)


def normalize_series(X_train, X_test):
    """Subtract initial value per sample — handles 2.0 vs 3.0 baseline shift."""
    train_init = X_train[:, :1, :]
    test_init = X_test[:, :1, :]
    return X_train - train_init, X_test - test_init


def load_test_answer(answer_path):
    """Load the ground truth answer.csv. ONLY call during final evaluation.

    Args:
        answer_path: path to test data/answer.csv

    Returns:
        DataFrame with columns: Spacecraft No., ID, task1-5, Test condition
    """
    if not os.path.exists(answer_path):
        raise FileNotFoundError(f"Answer file not found: {answer_path}")
    return pd.read_csv(answer_path)
