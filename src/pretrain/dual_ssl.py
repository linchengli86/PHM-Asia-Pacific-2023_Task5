"""Dual-channel SSL pretraining (Idea 2+).

SimCLR (fast) + MAE (slow), sharing the same encoder.

L_total = alpha_simclr * L_simclr + alpha_mae * L_mae
"""

from .simclr import SimCLRPretrainer
from .mae import MAEPretrainer


class DualSSLPretrainer:
    """Combined SimCLR + MAE pretraining with shared encoder."""

    def __init__(self, encoder, cfg):
        pretrain_cfg = cfg["pretrain"]["dual"]
        simclr_cfg = cfg["pretrain"]["simclr"]
        mae_cfg = cfg["pretrain"]["mae"]

        self.encoder = encoder
        self.simclr = SimCLRPretrainer(encoder, simclr_cfg["temperature"],
                                       simclr_cfg.get("projection_dim", 64))
        self.mae = MAEPretrainer(encoder, cfg["data"]["n_params"],
                                 mae_cfg["mask_ratio"], 200, mae_cfg.get("decoder_dim", 64))
        self.alpha_simclr = pretrain_cfg["alpha_simclr"]
        self.alpha_mae = pretrain_cfg["alpha_mae"]

    def parameters(self):
        return list(self.encoder.parameters()) + \
               list(self.simclr.projection.parameters()) + \
               list(self.mae.decoder.parameters())

    def train(self):
        self.encoder.train()
        self.simclr.projection.train()
        self.mae.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.simclr.projection.eval()
        self.mae.decoder.eval()

    def to(self, device):
        self.encoder.to(device)
        self.simclr.projection.to(device)
        self.mae.decoder.to(device)

    def compute_loss(self, x):
        loss_simclr = self.simclr.compute_loss(x)
        loss_mae = self.mae.compute_loss(x)
        return {
            "simclr": loss_simclr,
            "mae": loss_mae,
            "total": self.alpha_simclr * loss_simclr + self.alpha_mae * loss_mae,
        }
