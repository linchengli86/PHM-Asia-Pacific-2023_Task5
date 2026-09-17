"""MixupTrainer: Training with manifold mixup (锦上添花 1).

Adds embedding-level mixup to the standard training loop.
Drop-in replacement for Trainer — just swap the import.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from .losses import MultiTaskLoss, PCGradWrapper
from .metrics import compute_all_metrics


class MixupTrainer:
    """Trainer with manifold mixup on encoder outputs."""

    def __init__(self, model, cfg, device):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.loss_fn = MultiTaskLoss(cfg)
        self.pcgrad = PCGradWrapper()
        self.mixup_alpha = cfg.get("mixup", {}).get("alpha", 0.5)
        self.use_pcgrad = cfg["training"].get("use_pcgrad", False)

        encoder_params = []
        head_params = []
        for name, p in model.named_parameters():
            if "encoder" in name:
                encoder_params.append(p)
            else:
                head_params.append(p)
        self.optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": cfg["training"]["lr"] * 0.1},
            {"params": head_params, "lr": cfg["training"]["lr"]},
        ], weight_decay=cfg["training"]["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=cfg["training"]["lr_factor"],
            patience=cfg["training"]["lr_patience"])
        self.early_stop = cfg["training"]["early_stop_patience"]

    def _mixup_loss(self, outputs, y_a, y_b, lam):
        """Mixup loss: weighted sum of losses against two target sets."""
        losses_a = self.loss_fn(outputs, {k: a for k, a in y_a.items()})
        losses_b = self.loss_fn(outputs, {k: b for k, b in y_b.items()})
        return [lam * la + (1 - lam) * lb for la, lb in zip(losses_a, losses_b)]

    def train_epoch(self, loader):
        self.model.train()
        total_loss, n = 0, 0
        for batch in loader:
            x = batch["x"].to(self.device)
            tgt = {k: batch[k].to(self.device) for k in
                   ["task1", "task2", "task3", "task4", "task5"]}

            # Get encoder output
            h = self.model.encoder(x)

            # Manifold mixup
            B = h.shape[0]
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha > 0 else 1.0
            lam = max(lam, 1 - lam)
            index = torch.randperm(B, device=self.device)

            h_mixed = lam * h + (1 - lam) * h[index]
            outputs = self.model.heads(h_mixed)

            # Mixup targets
            y_a = tgt
            y_b = {k: v[index] for k, v in tgt.items()}
            losses = self._mixup_loss(outputs, y_a, y_b, lam)

            if self.use_pcgrad:
                self.pcgrad.step(losses, self.model, self.optimizer)
            else:
                total = sum(losses)
                self.optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += sum(l.item() for l in losses)
            n += 1
        return total_loss / max(n, 1)

    @torch.no_grad()
    def validate_epoch(self, loader):
        self.model.eval()
        total_loss, n = 0, 0
        preds = {f"task{i}": [] for i in range(1, 6)}
        trues = {f"task{i}": [] for i in range(1, 6)}

        for batch in loader:
            x = batch["x"].to(self.device)
            tgt = {k: batch[k].to(self.device) for k in
                   ["task1", "task2", "task3", "task4", "task5"]}
            for k in trues:
                trues[k].append(tgt[k].cpu().numpy())

            outputs = self.model(x)
            losses = self.loss_fn(outputs, tgt)
            total_loss += sum(l.item() for l in losses)

            preds["task1"].append((torch.sigmoid(outputs[0]) > 0.5).int().cpu().numpy())
            preds["task2"].append(outputs[1].argmax(1).cpu().numpy())
            preds["task3"].append(outputs[2].argmax(1).cpu().numpy())
            preds["task4"].append(outputs[3].argmax(1).cpu().numpy())
            preds["task5"].append(outputs[4].cpu().numpy())
            n += 1

        for k in preds:
            preds[k] = np.concatenate(preds[k])
            trues[k] = np.concatenate(trues[k])
        return total_loss / max(n, 1), preds, trues

    def run(self, train_loader, val_loader, max_epochs):
        best_loss, best_metrics, patience = float("inf"), None, 0
        for epoch in range(max_epochs):
            tr_loss = self.train_epoch(train_loader)
            val_loss, preds, trues = self.validate_epoch(val_loader)
            self.scheduler.step(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_metrics = compute_all_metrics(preds, trues)
                best_metrics["epoch"] = epoch + 1
                patience = 0
            else:
                patience += 1
            if patience >= self.early_stop:
                break
        return best_metrics
