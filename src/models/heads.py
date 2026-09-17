"""Standard task heads: Linear → Activation."""

import torch
import torch.nn as nn
from ..utils.config import get_task_config


class MultiTaskHeads(nn.Module):
    """5 task heads on top of shared encoder output.

    task1: Linear(embed→32→1) + Sigmoid → BCE
    task2: Linear(embed→32→4) → CE
    task3: Linear(embed→32→9) → CE  (or ProtoTaskHeads)
    task4: Linear(embed→32→5) → CE
    task5: Linear(embed→32→1) + ReLU → MSE
    """

    def __init__(self, embed_dim, cfg):
        super().__init__()
        tasks = cfg["tasks"]
        hidden = 32

        self.head1 = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, tasks["task1"]["num_classes"]),
        )
        self.head2 = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, tasks["task2"]["num_classes"]),
        )
        self.head3 = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, tasks["task3"]["num_classes"]),
        )
        self.head4 = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, tasks["task4"]["num_classes"]),
        )
        self.head5 = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, tasks["task5"]["num_classes"]),
        )

    def forward(self, x):
        o1 = self.head1(x)
        o2 = self.head2(x)
        o3 = self.head3(x)
        o4 = self.head4(x)
        o5 = torch.relu(self.head5(x))
        return o1, o2, o3, o4, o5
