"""Manifold Mixup for time series embeddings (锦上添花 1).

Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
Verma et al., "Manifold Mixup", ICML 2019

Applied at the embedding level: interpolate encoder outputs and labels,
creating smooth transitions between classes.
"""

import torch
import numpy as np


def mixup_data(x, y, alpha=0.5):
    """Standard input-level mixup.

    Args:
        x: (B, T, C) time series batch
        y: dict of label arrays, each (B,) or (B, 1)
        alpha: Beta distribution parameter

    Returns:
        mixed_x, mixed_y, lam
    """
    if alpha <= 0:
        return x, y, 1.0

    B = x.shape[0]
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)

    index = torch.randperm(B, device=x.device)

    # Mix input
    mixed_x = lam * x + (1 - lam) * x[index]

    # Mix labels (for classification tasks: use mixed targets in CE)
    mixed_y = {}
    for k, v in y.items():
        if k in ["task1", "task5"]:
            mixed_y[k] = lam * v + (1 - lam) * v[index]
        else:
            # Multi-class: store both targets + lambda
            mixed_y[k] = (v, v[index], lam)

    return mixed_x, mixed_y, lam


def manifold_mixup(embeddings, y, alpha=0.5):
    """Mixup at the embedding level (Manifold Mixup).

    More effective than input mixup for time series because:
    - Time series have strong temporal structure that input mixup can break
    - Embedding space is smoother and interpolation is more meaningful

    Args:
        embeddings: (B, D) encoded representations
        y: dict of label arrays
        alpha: Beta parameter

    Returns:
        mixed_emb, mixed_y, lam
    """
    if alpha <= 0:
        return embeddings, y, 1.0

    B = embeddings.shape[0]
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)

    index = torch.randperm(B, device=embeddings.device)
    mixed_emb = lam * embeddings + (1 - lam) * embeddings[index]

    mixed_y = {}
    for k, v in y.items():
        if k in ["task1", "task5"]:
            mixed_y[k] = lam * v + (1 - lam) * v[index]
        else:
            mixed_y[k] = (v, v[index], lam)

    return mixed_emb, mixed_y, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss: weighted sum of losses against two targets.

    Usage:
        For multi-class tasks (task2/3/4) with manifold mixup:
            y_a, y_b, lam = mixed_y["task2"]
            loss = lam * CE(pred, y_a) + (1-lam) * CE(pred, y_b)
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
