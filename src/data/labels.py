"""Label generation from train/labels.xlsx.

Produces task1-task5 labels for all 177 training cases.
Test labels (answer.csv) are NEVER touched during training.
"""

import numpy as np
import pandas as pd


def generate_labels(labels_path):
    """Parse labels.xlsx and return 5-task label arrays.

    Args:
        labels_path: path to train/labels.xlsx

    Returns:
        y: dict with keys task1-task5, each (177,) numpy array
    """
    raw = pd.read_excel(labels_path, header=None)
    data = raw.iloc[2:].copy()
    data.columns = [
        "Case", "Spacecraft", "Condition",
        "SV1", "SV2", "SV3", "SV4",
        "BP1", "BP2", "BP3", "BP4", "BP5", "BP6", "BP7", "BV1",
    ]
    data["Case"] = data["Case"].astype(int)
    data["Spacecraft"] = data["Spacecraft"].astype(int)
    for col in ["SV1", "SV2", "SV3", "SV4"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    n = len(data)
    bp_cols = ["BP1", "BP2", "BP3", "BP4", "BP5", "BP6", "BP7", "BV1"]
    sv_cols = ["SV1", "SV2", "SV3", "SV4"]

    # task1: binary anomaly detection (0=Normal, 1=Abnormal)
    task1 = (data["Condition"] != "Normal").astype(int).values.astype(np.float32)

    # task2: anomaly subtype
    task2 = np.zeros(n, dtype=np.int64)
    task2[data["Condition"] == "Normal"] = 0
    task2[data["Condition"] == "Anomaly"] = 2   # bubble anomaly
    task2[data["Condition"] == "Fault"] = 3      # SV fault
    # class 1 ("Other" / "Unknown anomaly") does not appear in training

    # task3: bubble position (9-class)
    task3 = np.zeros(n, dtype=np.int64)
    for i, col in enumerate(bp_cols):
        mask = (data[col].values == "Yes")
        task3[mask] = i + 1

    # task4: faulty valve position (5-class)
    task4 = np.zeros(n, dtype=np.int64)
    for i, col in enumerate(sv_cols):
        mask = (data[col].values != 100)
        task4[mask] = i + 1

    # task5: valve opening ratio (regression)
    task5 = np.full(n, 100.0, dtype=np.float32)
    sv_vals = data[sv_cols].values.astype(np.float32)
    for p in range(4):
        mask = (sv_vals[:, p] != 100)
        task5[mask] = sv_vals[mask, p]

    return {
        "task1": task1,
        "task2": task2,
        "task3": task3,
        "task4": task4,
        "task5": task5,
    }
