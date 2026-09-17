"""PyTorch Dataset for windowed time series + labels."""

import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """Simple dataset: (x, task1, task2, task3, task4, task5).

    Returns dict for clean batching.
    """

    def __init__(self, X, y, indices=None):
        if indices is not None:
            X = X[indices]
            y = {k: v[indices] for k, v in y.items()}

        self.X = torch.from_numpy(X).float()
        self.y = {}
        for k in ["task1", "task2", "task3", "task4", "task5"]:
            v = y[k]
            if k == "task1" or k == "task5":
                self.y[k] = torch.from_numpy(v).float().unsqueeze(1) if v.ndim == 1 else torch.from_numpy(v).float()
            else:
                self.y[k] = torch.from_numpy(v).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "x": self.X[idx],
            "task1": self.y["task1"][idx],
            "task2": self.y["task2"][idx],
            "task3": self.y["task3"][idx],
            "task4": self.y["task4"][idx],
            "task5": self.y["task5"][idx],
        }
