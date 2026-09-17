"""Fast-Slow Dual Channel Model — Collaborative (Idea 4 / GR00T N1).

System 1 (Fast CNN): "Big picture" — binary anomaly detection, output h₁.
System 2 (Slow Transformer): "Detailed reasoning" — receives h₁ as context.

Key: S1's features h₁ are fused with S2's features h₂ before task2-5 heads.
The fast channel's "intuition" directly informs the slow channel's "analysis".
"""

import torch
import torch.nn as nn
from .encoder import CNNEncoder, TransformerEncoder


class System1(nn.Module):
    """Fast anomaly detector: CNN → h₁ (128-dim fast features) + task1 logit."""

    def __init__(self, in_channels, cnn_cfg):
        super().__init__()
        self.encoder = CNNEncoder(
            in_channels=in_channels,
            channels=cnn_cfg["channels"],
            kernels=cnn_cfg["kernels"],
            dropout=cnn_cfg["dropout"],
            output_dim=cnn_cfg["output_dim"],
        )
        embed_dim = cnn_cfg["output_dim"]
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        h = self.encoder(x)              # (B, 128) fast features
        return h, self.head(h)           # h₁, task1 logit


class System2(nn.Module):
    """Slow diagnostic engine: Transformer → h₂ → [h₁ ⊕ h₂] → task2-5.

    Receives fast features h₁ from System 1 and fuses them with its own
    representation before classification. The fusion lets System 2 know
    "what System 1 already saw".
    """

    def __init__(self, in_channels, tf_cfg, task_cfg, fast_dim):
        super().__init__()
        self.encoder = TransformerEncoder(
            in_channels=in_channels,
            d_model=tf_cfg["d_model"],
            nhead=tf_cfg["nhead"],
            num_layers=tf_cfg["num_layers"],
            dim_feedforward=tf_cfg["dim_feedforward"],
            dropout=tf_cfg["dropout"],
            max_len=tf_cfg["max_len"],
        )
        slow_dim = tf_cfg["d_model"]
        fused_dim = slow_dim + fast_dim  # h₁ ⊕ h₂

        # Fusion layer: project concatenated features
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # Task heads on fused representation
        from .heads import MultiTaskHeads
        task_cfg_copy = dict(task_cfg)
        self.heads = MultiTaskHeads(128, task_cfg_copy)

    def forward(self, x, h_fast):
        """x: raw input (B, T, C), h_fast: S1 features (B, fast_dim)

        Returns: (o2, o3, o4, o5) — task2-5 predictions only.
        """
        h_slow = self.encoder(x)                # (B, slow_dim)
        h_fused = self.fusion(torch.cat([h_slow, h_fast], dim=1))  # (B, 128)
        # MultiTaskHeads returns 5 outputs; we only use task2-5
        _, o2, o3, o4, o5 = self.heads(h_fused)
        return o2, o3, o4, o5


class DualChannelModel(nn.Module):
    """Collaborative fast-slow dual channel.

    Training:
      Stage 1: Train System 1 on all data (task1 only). Target high recall.
      Stage 2: Freeze System 1. Train System 2 with S1 features as input.

    Inference:
      h₁, task1_logit = S1(x)
      task2-5 = S2(x, h₁)          ← h₁ always informs S2
      if task1 == 0 (Normal):       ← gate still works for efficiency
          task2-5 → default values
    """

    def __init__(self, cfg):
        super().__init__()
        data_cfg = cfg["data"]
        cnn_cfg = cfg["encoder"]["cnn"]
        tf_cfg = cfg["encoder"]["transformer"]

        self.system1 = System1(data_cfg["n_params"], cnn_cfg)
        self.system2 = System2(data_cfg["n_params"], tf_cfg, cfg,
                               fast_dim=cnn_cfg["output_dim"])
        self.threshold = 0.5

    def forward(self, x, stage=1):
        if stage == 1:
            h1, logit1 = self.system1(x)
            return logit1, None, h1
        elif stage == 2:
            # Stage 2: S1 runs but is frozen, h₁ passed to S2
            with torch.no_grad():
                h1, _ = self.system1(x)
            return self.system2(x, h1)  # (o2,o3,o4,o5)
        else:
            # Full inference
            h1, logit1 = self.system1(x)
            prob1 = torch.sigmoid(logit1)
            is_abnormal = (prob1 > self.threshold).squeeze(-1)
            o2, o3, o4, o5 = self.system2(x, h1)
            return logit1, o2, o3, o4, o5, is_abnormal
