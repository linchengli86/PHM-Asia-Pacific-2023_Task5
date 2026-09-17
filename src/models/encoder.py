"""Temporal encoders: CNN and lightweight Transformer."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """3-block 1D-CNN with global average pooling.

    Output: (B, output_dim) embedding vector.
    """

    def __init__(self, in_channels=7, channels=(64, 128, 128), kernels=(7, 5, 3),
                 dropout=0.3, output_dim=128):
        super().__init__()
        layers = []
        c_in = in_channels
        for c_out, k in zip(channels, kernels):
            layers.extend([
                nn.Conv1d(c_in, c_out, k, padding=k // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            ])
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.output_dim = output_dim

    def forward(self, x):
        # x: (B, T, C) → (B, C, T)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.gap(x).squeeze(-1)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class TransformerEncoder(nn.Module):
    """Lightweight Transformer for time series (Idea 3).

    d_model=64, 3 layers, 4 heads — ~150K params.
    """

    def __init__(self, in_channels=7, d_model=64, nhead=4, num_layers=3,
                 dim_feedforward=128, dropout=0.2, max_len=200):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = d_model

    def forward(self, x):
        # x: (B, T, C)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return x


def build_encoder(cfg):
    """Factory: build encoder from config."""
    enc_cfg = cfg["encoder"]
    enc_type = enc_cfg.get("type", "cnn")

    if enc_type == "cnn":
        c = enc_cfg["cnn"]
        return CNNEncoder(in_channels=cfg["data"]["n_params"],
                          channels=c["channels"], kernels=c["kernels"],
                          dropout=c["dropout"], output_dim=c["output_dim"])
    elif enc_type == "transformer":
        c = enc_cfg["transformer"]
        return TransformerEncoder(in_channels=cfg["data"]["n_params"],
                                  d_model=c["d_model"], nhead=c["nhead"],
                                  num_layers=c["num_layers"],
                                  dim_feedforward=c["dim_feedforward"],
                                  dropout=c["dropout"], max_len=c["max_len"])
    else:
        raise ValueError(f"Unknown encoder type: {enc_type}")
