"""Prototypical Networks for few-shot classification (task3, task4).

Snell et al., NeurIPS 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ProtoEmbedding(nn.Module):
    """Projects encoder output to normalized embedding space for prototypical distance."""

    def __init__(self, embed_dim, proto_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, proto_dim),
            nn.ReLU(),
            nn.Linear(proto_dim, proto_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class ProtoTaskHeads(nn.Module):
    """Mixed heads: task1/2/5 standard, task3/4 prototypical."""

    def __init__(self, embed_dim, cfg):
        super().__init__()
        tasks = cfg["tasks"]
        hidden = 32

        self.head1 = nn.Sequential(nn.Linear(embed_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, tasks["task1"]["num_classes"]))
        self.head2 = nn.Sequential(nn.Linear(embed_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, tasks["task2"]["num_classes"]))
        self.head5 = nn.Sequential(nn.Linear(embed_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, tasks["task5"]["num_classes"]))

        proto_dim = cfg.get("proto", {}).get("embed_dim", 32)
        self.proto_embed3 = ProtoEmbedding(embed_dim, proto_dim)
        self.proto_embed4 = ProtoEmbedding(embed_dim, proto_dim)

    def forward(self, x, return_all=False):
        o1 = self.head1(x)
        o2 = self.head2(x)
        e3 = self.proto_embed3(x)
        e4 = self.proto_embed4(x)
        o5 = torch.relu(self.head5(x))
        if return_all:
            return o1, o2, e3, e4, o5
        return o1, o2, e3, e4, o5


class ProtoLoss(nn.Module):
    """Prototypical loss: CE on negative Euclidean distances to class prototypes."""

    def __init__(self, n_way):
        super().__init__()
        self.n_way = n_way

    def forward(self, embeddings, labels):
        device = embeddings.device
        unique_classes = torch.unique(labels)

        if len(unique_classes) < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        prototypes = []
        proto_cls = []
        for c in unique_classes:
            mask = (labels == c)
            prototypes.append(embeddings[mask].mean(0))
            proto_cls.append(c)

        prototypes = torch.stack(prototypes)
        dist2 = torch.cdist(embeddings, prototypes) ** 2
        logits = -dist2

        target_idx = torch.zeros(len(embeddings), dtype=torch.long, device=device)
        for i, c in enumerate(unique_classes):
            target_idx[labels == c] = i

        return F.cross_entropy(logits, target_idx)


class EMAPrototypes:
    """Exponential Moving Average prototype cache (锦上添花 4).

    Smoothes prototype estimates across batches — critical for minority classes
    with only 3 samples where batch-level estimates are extremely noisy.
    """

    def __init__(self, n_classes, embed_dim, alpha=0.9, warmup=0.5, total_epochs=300):
        self.prototypes = None
        self.classes = None
        self.alpha_init = warmup
        self.alpha_final = alpha
        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.total_epochs = total_epochs

    def get_alpha(self, epoch):
        """Cosine anneal alpha from warmup to final."""
        progress = min(epoch / self.total_epochs, 1.0)
        return self.alpha_final - (self.alpha_final - self.alpha_init) * (1 + np.cos(progress * np.pi)) / 2

    def update(self, embeddings, labels, epoch):
        """Update EMA prototypes with current batch's class means.

        Args:
            embeddings: (B, D) normalized embeddings
            labels: (B,) class labels
            epoch: current epoch (for alpha scheduling)
        """
        alpha = self.get_alpha(epoch)
        batch_protos, batch_classes = build_prototypes(embeddings, labels, self.n_classes)

        if batch_protos is None:
            return

        if self.prototypes is None:
            self.prototypes = batch_protos
            self.classes = batch_classes
        else:
            # Merge: update existing, add new
            for i, c in enumerate(batch_classes):
                c = int(c.item())
                if c in self.classes:
                    idx = (self.classes == c).nonzero(as_tuple=True)[0][0]
                    self.prototypes[idx] = alpha * self.prototypes[idx] + (1 - alpha) * batch_protos[i]
                else:
                    self.prototypes = torch.cat([self.prototypes, batch_protos[i:i+1]], dim=0)
                    self.classes = torch.cat([self.classes, batch_classes[i:i+1]], dim=0)

    def get_prototypes(self):
        return self.prototypes, self.classes


def build_prototypes(embeddings, labels, n_classes):
    """Compute class prototypes from all samples.

    Returns:
        prototypes: (K, D) where K <= n_classes
        proto_classes: (K,) class index array
    """
    device = embeddings.device
    proto_list, class_list = [], []
    for c in range(n_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            proto_list.append(embeddings[mask].mean(0))
            class_list.append(c)

    if len(proto_list) == 0:
        return None, None
    return torch.stack(proto_list), torch.tensor(class_list, device=device)


def compute_proto_logits(embeddings, prototypes):
    """Negative squared Euclidean distance as logits."""
    return -(torch.cdist(embeddings, prototypes) ** 2)
