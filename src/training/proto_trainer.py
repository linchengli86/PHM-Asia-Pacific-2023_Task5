"""ProtoTrainer: Training + validation with prototypical heads for task3/4.

Extends the standard Trainer to handle:
  - ProtoLoss for task3/4 training
  - Prototype construction + nearest-prototype classification for validation
  - EMA prototype cache (锦上添花 4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from .losses import PCGradWrapper
from .metrics import compute_all_metrics
from ..models.prototypical import ProtoLoss, build_prototypes, compute_proto_logits, EMAPrototypes


class ProtoTrainer:
    """Trainer with prototypical network support for task3/task4."""

    def __init__(self, model, cfg, device):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.use_pcgrad = cfg["training"].get("use_pcgrad", False)
        self.pcgrad = PCGradWrapper()

        tasks_cfg = cfg["tasks"]
        proto_cfg = cfg.get("proto", {})
        self.proto_loss3 = ProtoLoss(n_way=tasks_cfg["task3"]["num_classes"])
        self.proto_loss4 = ProtoLoss(n_way=tasks_cfg["task4"]["num_classes"])
        self.use_ema = proto_cfg.get("use_ema", False)
        if self.use_ema:
            self.ema3 = EMAPrototypes(tasks_cfg["task3"]["num_classes"],
                                      proto_cfg.get("embed_dim", 32),
                                      proto_cfg.get("ema_alpha", 0.9),
                                      proto_cfg.get("ema_warmup", 0.5),
                                      cfg["training"]["max_epochs"])
            self.ema4 = EMAPrototypes(tasks_cfg["task4"]["num_classes"],
                                      proto_cfg.get("embed_dim", 32),
                                      proto_cfg.get("ema_alpha", 0.9),
                                      proto_cfg.get("ema_warmup", 0.5),
                                      cfg["training"]["max_epochs"])

        # Optimizer: separate lr for encoder and heads
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

        self.w1 = tasks_cfg["task1"]["weight"]
        self.w2 = tasks_cfg["task2"]["weight"]
        self.w3 = tasks_cfg["task3"]["weight"]
        self.w4 = tasks_cfg["task4"]["weight"]
        self.w5 = tasks_cfg["task5"]["weight"]

    def _compute_losses(self, outputs, targets):
        o1, o2, e3, e4, o5 = outputs
        l1 = F.binary_cross_entropy_with_logits(o1, targets["task1"])
        l2 = F.cross_entropy(o2, targets["task2"])
        l3 = self.proto_loss3(e3, targets["task3"])
        l4 = self.proto_loss4(e4, targets["task4"])
        l5 = F.mse_loss(torch.relu(o5), targets["task5"])
        return [self.w1 * l1, self.w2 * l2, self.w3 * l3, self.w4 * l4, self.w5 * l5]

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss, n = 0, 0
        for batch in loader:
            x = batch["x"].to(self.device)
            tgt = {k: batch[k].to(self.device) for k in
                   ["task1", "task2", "task3", "task4", "task5"]}
            outputs = self.model(x)
            losses = self._compute_losses(outputs, tgt)

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

        # Update EMA prototypes
        if self.use_ema:
            emb3_list, lab3_list = [], []
            emb4_list, lab4_list = [], []
            with torch.no_grad():
                for batch in loader:
                    x = batch["x"].to(self.device)
                    _, _, e3, e4, _ = self.model(x)
                    emb3_list.append(e3)
                    lab3_list.append(batch["task3"])
                    emb4_list.append(e4)
                    lab4_list.append(batch["task4"])
            self.ema3.update(torch.cat(emb3_list), torch.cat(lab3_list), epoch)
            self.ema4.update(torch.cat(emb4_list), torch.cat(lab4_list), epoch)

        return total_loss / max(n, 1)

    @torch.no_grad()
    def validate_epoch(self, loader, train_loader=None, epoch=0):
        self.model.eval()
        total_loss, n = 0, 0
        preds = {f"task{i}": [] for i in range(1, 6)}
        trues = {f"task{i}": [] for i in range(1, 6)}

        # Build prototypes from training data
        if self.use_ema:
            proto3, cls3 = self.ema3.get_prototypes()
            proto4, cls4 = self.ema4.get_prototypes()
        else:
            emb3_tr, lab3_tr = [], []
            emb4_tr, lab4_tr = [], []
            for batch in train_loader:
                x = batch["x"].to(self.device)
                _, _, e3, e4, _ = self.model(x)
                emb3_tr.append(e3)
                lab3_tr.append(batch["task3"])
                emb4_tr.append(e4)
                lab4_tr.append(batch["task4"])
            proto3, cls3 = build_prototypes(torch.cat(emb3_tr), torch.cat(lab3_tr),
                                            self.cfg["tasks"]["task3"]["num_classes"])
            proto4, cls4 = build_prototypes(torch.cat(emb4_tr), torch.cat(lab4_tr),
                                            self.cfg["tasks"]["task4"]["num_classes"])

        for batch in loader:
            x = batch["x"].to(self.device)
            tgt = {k: batch[k].to(self.device) for k in
                   ["task1", "task2", "task3", "task4", "task5"]}
            for k in trues:
                trues[k].append(tgt[k].cpu().numpy())

            outputs = self.model(x)
            losses = self._compute_losses(outputs, tgt)
            total_loss += sum(l.item() for l in losses)

            o1, o2, e3, e4, o5 = outputs
            preds["task1"].append((torch.sigmoid(o1) > 0.5).int().cpu().numpy())
            preds["task2"].append(o2.argmax(1).cpu().numpy())
            # Proto classification
            log3 = compute_proto_logits(e3, proto3)
            log4 = compute_proto_logits(e4, proto4)
            preds["task3"].append(cls3[log3.argmax(1)].cpu().numpy())
            preds["task4"].append(cls4[log4.argmax(1)].cpu().numpy())
            preds["task5"].append(o5.cpu().numpy())
            n += 1

        for k in preds:
            preds[k] = np.concatenate(preds[k])
            trues[k] = np.concatenate(trues[k])

        return total_loss / max(n, 1), preds, trues

    def run(self, train_loader, val_loader, max_epochs):
        best_loss = float("inf")
        best_metrics = None
        patience = 0

        for epoch in range(max_epochs):
            tr_loss = self.train_epoch(train_loader, epoch)
            val_loss, preds, trues = self.validate_epoch(val_loader, train_loader, epoch)
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
