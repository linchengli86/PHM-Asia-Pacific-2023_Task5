"""BERT-style MLM pretraining for time series.

Key design:
  - Continuous segment masking (block of 20-40 steps → [MASK]=0)
  - Encoder: 3-layer 1D CNN (preserves spatial dims, no pooling)
  - Decoder: 2-layer MLP → reconstruct masked positions
  - Loss: MSE on masked positions only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BERTEncoder(nn.Module):
    """CNN encoder that preserves temporal resolution (no pooling in last stage).

    Returns both pooled embedding h₁ and full temporal embeddings h_all.
    """

    def __init__(self, in_channels=7, d_model=128):
        super().__init__()
        # Block 1: stride=2
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, 7, padding=3), nn.BatchNorm1d(64),
            nn.ReLU(inplace=True), nn.MaxPool1d(2), nn.Dropout(0.2))
        # Block 2: stride=2
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True), nn.MaxPool1d(2), nn.Dropout(0.2))
        # Block 3: stride=1 (preserve resolution)
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, d_model, 3, padding=1), nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True), nn.Dropout(0.2))

        self.d_model = d_model
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, T, C) → (B, C, T)
        x = x.transpose(1, 2)
        x = self.conv1(x)   # (B, 64, T/2)
        x = self.conv2(x)   # (B, 128, T/4)
        h_all = self.conv3(x)  # (B, d_model, T/4) — temporal features

        h_pool = self.gap(h_all).squeeze(-1)  # (B, d_model)
        h_all = h_all.transpose(1, 2)          # (B, T/4, d_model)
        return h_pool, h_all


class MLMDecoder(nn.Module):
    """Reconstruct masked positions from latent + visible context."""

    def __init__(self, d_model=128, n_params=7, target_len=50, hidden=256):
        super().__init__()
        self.target_len = target_len
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, target_len * n_params),
        )

    def forward(self, h_pool):
        # h_pool: (B, d_model) → reconstruct full temporal window
        out = self.net(h_pool)  # (B, T/4 * n_params)
        return out.view(-1, self.target_len, 7)


class BERTPretrainer:
    """BERT-style MLM training."""

    def __init__(self, encoder, n_params=7, mask_len=(20, 40), t_downsample=4):
        self.encoder = encoder
        self.decoder = MLMDecoder(encoder.d_model, n_params,
                                   target_len=200 // t_downsample)
        self.mask_len = mask_len
        self.n_params = n_params
        self.t_downsample = t_downsample

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def to(self, device):
        self.encoder.to(device)
        self.decoder.to(device)

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def compute_loss(self, x):
        """Mask a random continuous segment, encode, reconstruct masked steps."""
        B, T, C = x.shape
        device = x.device

        # Random continuous mask per sample
        x_masked = x.clone()
        mask_positions = []
        for i in range(B):
            seg_len = np.random.randint(*self.mask_len)
            t0 = np.random.randint(0, max(1, T - seg_len))
            x_masked[i, t0:t0+seg_len, :] = 0.0
            mask_positions.append((t0, t0 + seg_len))

        # Encode
        h_pool, h_all = self.encoder(x_masked)

        # Decode → reconstruct at downsampled resolution
        x_recon = self.decoder(h_pool)  # (B, T//4, C)

        # MSE on masked positions only (downsampled)
        loss = 0.0
        n_masked = 0
        for i in range(B):
            t0, t1 = mask_positions[i]
            t0_ds = t0 // self.t_downsample
            t1_ds = max(t0_ds + 1, t1 // self.t_downsample)
            if t1_ds > t0_ds:
                x_orig_ds = x[i, ::self.t_downsample, :][t0_ds:t1_ds]
                x_recon_ds = x_recon[i, t0_ds:t1_ds]
                loss += F.mse_loss(x_recon_ds, x_orig_ds)
                n_masked += 1

        return loss / max(n_masked, 1)


class MultiHeadCrossAttention(nn.Module):
    """Cross-attention: Q from previous stage, KV from BERT temporal embeddings.

    Projects query to kv_dim first, then standard cross-attention.
    """

    def __init__(self, q_dim, kv_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(q_dim, kv_dim)  # project query to KV space
        self.attn = nn.MultiheadAttention(kv_dim, n_heads, dropout=dropout,
                                           batch_first=True)

    def forward(self, query, key_value):
        q = self.q_proj(query).unsqueeze(1)  # (B, 1, kv_dim)
        out, _ = self.attn(q, key_value, key_value)
        return out.squeeze(1)  # (B, kv_dim)


class StageHead(nn.Module):
    """One stage of the progressive chain: cross-attn + small CNN + class head."""

    def __init__(self, prev_dim, bert_dim, stage_dim=64, n_classes=4):
        super().__init__()
        self.cross_attn = MultiHeadCrossAttention(prev_dim, bert_dim)
        self.cnn = nn.Sequential(
            nn.Conv1d(7, 32, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        combined = stage_dim + bert_dim + 32  # h_prev + c_attn + raw_cnn
        self.head = nn.Sequential(
            nn.Linear(combined, 64), nn.ReLU(), nn.Linear(64, n_classes))
        self.stage_dim = stage_dim + bert_dim  # output feature dim

    def forward(self, h_prev, h_all, raw):
        # Cross-attention
        c = self.cross_attn(h_prev, h_all)  # (B, bert_dim)
        # Raw CNN features
        r = self.cnn(raw.transpose(1, 2)).squeeze(-1)  # (B, 32)
        # Combine
        h_stage = torch.cat([h_prev, c], dim=1)  # (B, prev_dim+bert_dim)
        out = self.head(torch.cat([h_stage, r], dim=1))
        return out, h_stage
