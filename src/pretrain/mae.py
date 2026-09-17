"""MAE (Masked Autoencoder) pretraining — System 2 (slow thinking).

BERT-style: mask random time steps, reconstruct them from context.
Only works with CNNEncoder (output_dim must match decoder input).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MAEDecoder(nn.Module):
    """Lightweight linear decoder — works with any encoder type.

    Directly projects latent to (target_len × n_params) via MLP,
    avoiding ConvTranspose shape mismatches with different encoders.
    """

    def __init__(self, latent_dim, n_params, target_len=200, hidden_dim=256):
        super().__init__()
        self.target_len = target_len
        self.n_params = n_params
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_params * target_len),
        )

    def forward(self, latent):
        x = self.net(latent)                            # (B, C*T)
        return x.view(-1, self.target_len, self.n_params)  # (B, T, C)


class MAEPretrainer:
    """MAE pretraining wrapper.

    Masks random time steps, encodes visible steps, decodes to reconstruct masked steps.
    Only CNNEncoder is supported (transformer needs special mask handling).
    """

    def __init__(self, encoder, n_params=7, mask_ratio=0.5, target_len=200, decoder_dim=64):
        self.encoder = encoder
        self.decoder = MAEDecoder(encoder.output_dim, n_params, target_len, decoder_dim)
        self.mask_ratio = mask_ratio
        self.target_len = target_len
        self.n_params = n_params

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def to(self, device):
        self.encoder.to(device)
        self.decoder.to(device)

    def compute_loss(self, x):
        """Mask and reconstruct. Loss only on masked positions."""
        B, T, C = x.shape
        device = x.device

        # Create random mask (time-step level, applied to all parameters)
        n_mask = int(T * self.mask_ratio)
        mask = torch.zeros(B, T, device=device)
        for i in range(B):
            idx = torch.randperm(T, device=device)[:n_mask]
            mask[i, idx] = 1

        # Encode visible steps
        x_visible = x.clone()
        x_visible[mask.bool()] = 0
        latent = self.encoder(x_visible)

        # Decode to full sequence
        x_recon = self.decoder(latent)

        # Resize if needed
        if x_recon.shape[1] != T:
            if x_recon.shape[1] > T:
                x_recon = x_recon[:, :T, :]
            else:
                pad = torch.zeros(B, T - x_recon.shape[1], C, device=device)
                x_recon = torch.cat([x_recon, pad], dim=1)

        # MSE only on masked positions
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, C)
        loss = F.mse_loss(x_recon[mask_expanded.bool()], x[mask_expanded.bool()])
        return loss
