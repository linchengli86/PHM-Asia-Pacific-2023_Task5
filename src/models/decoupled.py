"""Task-Specific Decoupled Encoders.

Three independent pathways from the raw window:
  Encoder A (CNN 128dim): task1 (binary), task2 (4-class)
  Encoder B (CNN 128dim): task3 (9-class), task4 (5-class) — metric space
  Encoder C (CNN 64dim):  task5 (regression)

No shared feature space. No SSL needed. Each path learns what its task needs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import CNNEncoder


class DecoupledModel(nn.Module):
    """Three encoders, three pathways, one forward pass."""

    def __init__(self, in_channels=7):
        super().__init__()
        # Encoder A: classification tasks 1&2 (discriminative)
        self.enc_a = CNNEncoder(in_channels, channels=[64, 128, 128],
                                kernels=[7, 5, 3], dropout=0.3, output_dim=128)
        # Encoder B: few-shot tasks 3&4 (metric learning)
        self.enc_b = CNNEncoder(in_channels, channels=[64, 128, 128],
                                kernels=[7, 5, 3], dropout=0.3, output_dim=128)
        # Encoder C: regression task 5 (value-sensitive)
        self.enc_c = CNNEncoder(in_channels, channels=[32, 64, 64],
                                kernels=[7, 5, 3], dropout=0.2, output_dim=64)

        # Heads
        self.head1 = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head2 = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 4))

        # Proto embedding for task3/4
        self.proto3 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32))
        self.proto4 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32))

        self.head5 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        h_a = self.enc_a(x)       # (B, 128) — discriminative
        h_b = self.enc_b(x)       # (B, 128) — metric
        h_c = self.enc_c(x)       # (B, 64)  — value-sensitive

        o1 = self.head1(h_a)
        o2 = self.head2(h_a)
        e3 = F.normalize(self.proto3(h_b), dim=1)  # (B, 32)
        e4 = F.normalize(self.proto4(h_b), dim=1)  # (B, 32)
        o5 = torch.relu(self.head5(h_c))

        return o1, o2, e3, e4, o5


class DecoupledLoss(nn.Module):
    """Per-path losses: A→BCE+CE, B→Proto, C→MSE."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.ce2 = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        o1, o2, e3, e4, o5 = outputs
        # Path A: standard classification
        l1 = self.bce(o1, targets["task1"])
        l2 = self.ce2(o2, targets["task2"])

        # Path B: prototypical (in-batch)
        l3 = self._proto_loss(e3, targets["task3"])
        l4 = self._proto_loss(e4, targets["task4"])

        # Path C: regression
        l5 = F.mse_loss(o5, targets["task5"])

        return [l1, l2 * 1.5, l3, l4, l5 * 2.0]

    def _proto_loss(self, embeddings, labels):
        """In-batch prototypical loss."""
        device = embeddings.device
        unique = torch.unique(labels)
        if len(unique) < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        protos, cls = [], []
        for c in unique:
            m = (labels == c)
            protos.append(embeddings[m].mean(0))
            cls.append(c)
        protos = torch.stack(protos)
        dist = torch.cdist(embeddings, protos) ** 2
        logits = -dist

        target_idx = torch.zeros(len(embeddings), dtype=torch.long, device=device)
        for i, c in enumerate(unique):
            target_idx[labels == c] = i

        return F.cross_entropy(logits, target_idx)
