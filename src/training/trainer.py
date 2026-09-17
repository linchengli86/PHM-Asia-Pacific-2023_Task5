"""Base trainer class with PCGrad support."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from .losses import MultiTaskLoss, PCGradWrapper
from .metrics import compute_all_metrics


class Trainer:
    """Handles one fold of training + validation."""

    def __init__(self, model, cfg, device):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.loss_fn = MultiTaskLoss(cfg)
        self.pcgrad = PCGradWrapper()

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
            patience=cfg["training"]["lr_patience"],
        )
        self.use_pcgrad = cfg["training"].get("use_pcgrad", True)
        self.early_stop = cfg["training"]["early_stop_patience"]

    def train_epoch(self, loader):
        self.model.train()
        total_loss, n = 0, 0
        for batch in loader:
            x = batch["x"].to(self.device)
            tgt = {k: batch[k].to(self.device) for k in
                   ["task1", "task2", "task3", "task4", "task5"]}
            outputs = self.model(x)
            losses = self.loss_fn(outputs, tgt)

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
        return total_loss / n

    @torch.no_grad()
    def validate_epoch(self, loader):
        self.model.eval()
        total_loss = 0
        n = 0
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

        return total_loss / n, preds, trues

    def run(self, train_loader, val_loader, max_epochs):
        best_loss = float("inf")
        best_metrics = None
        patience = 0

        for epoch in range(max_epochs):
            tr_loss = self.train_epoch(train_loader)
            val_loss, preds, trues = self.validate_epoch(val_loader)
            self.scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_metrics = compute_all_metrics(preds, trues)
                best_metrics["epoch"] = epoch + 1
                best_metrics["val_loss"] = float(val_loss)
                patience = 0
            else:
                patience += 1

            if patience >= self.early_stop:
                break

        return best_metrics
