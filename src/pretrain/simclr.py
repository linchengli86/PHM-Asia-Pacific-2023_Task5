"""SimCLR contrastive pretraining — System 1 (fast thinking)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def augment_simclr(x, strength="medium"):
    """Random augmentation for contrastive pair generation.

    Returns two augmented views of the same batch.
    """
    B, T, C = x.shape
    device = x.device

    # Augmentation 1: jitter
    sigma = 0.01 if strength == "weak" else 0.03
    x1 = x + torch.randn_like(x) * sigma * np.random.uniform(0.5, 1.5)

    # Augmentation 2: jitter + scaling
    x2 = x + torch.randn_like(x) * sigma * np.random.uniform(0.5, 1.5)
    factor = 0.85 + torch.rand(B, 1, 1).to(device) * 0.3
    x2 = x2 * factor

    return x1, x2


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class SimCLRPretrainer:
    """SimCLR pretraining wrapper."""

    def __init__(self, encoder, temperature=0.1, proj_dim=64):
        self.encoder = encoder
        self.projection = ProjectionHead(encoder.output_dim, 128, proj_dim)
        self.temperature = temperature

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.projection.parameters())

    def train(self):
        self.encoder.train()
        self.projection.train()

    def eval(self):
        self.encoder.eval()
        self.projection.eval()

    def to(self, device):
        self.encoder.to(device)
        self.projection.to(device)

    def compute_loss(self, x):
        """NT-Xent loss on a batch."""
        x1, x2 = augment_simclr(x)
        z1 = self.projection(self.encoder(x1))
        z2 = self.projection(self.encoder(x2))

        B = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.T) / self.temperature

        labels = torch.arange(B, device=z.device)
        labels = torch.cat([labels + B, labels], dim=0)

        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -float("inf"))

        return F.cross_entropy(sim, labels)
