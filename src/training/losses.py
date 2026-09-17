"""Multi-task loss and PCGrad gradient surgery."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiTaskLoss(nn.Module):
    """Weighted multi-task loss with optional class weights."""

    def __init__(self, cfg, class_weights=None):
        super().__init__()
        tasks = cfg["tasks"]
        self.bce = nn.BCEWithLogitsLoss()
        self.ce2 = nn.CrossEntropyLoss(weight=class_weights.get("task2") if class_weights else None)
        self.ce3 = nn.CrossEntropyLoss(weight=class_weights.get("task3") if class_weights else None)
        self.ce4 = nn.CrossEntropyLoss(weight=class_weights.get("task4") if class_weights else None)
        self.mse = nn.MSELoss()
        self.w1 = tasks["task1"]["weight"]
        self.w2 = tasks["task2"]["weight"]
        self.w3 = tasks["task3"]["weight"]
        self.w4 = tasks["task4"]["weight"]
        self.w5 = tasks["task5"]["weight"]

    def forward(self, outputs, targets):
        o1, o2, o3, o4, o5 = outputs
        losses = [
            self.w1 * self.bce(o1, targets["task1"]),
            self.w2 * self.ce2(o2, targets["task2"]),
            self.w3 * self.ce3(o3, targets["task3"]),
            self.w4 * self.ce4(o4, targets["task4"]),
            self.w5 * self.mse(o5, targets["task5"]),
        ]
        return losses


class PCGradWrapper:
    """Wrap per-task losses and apply PCGrad.

    Usage:
        wrapper = PCGradWrapper()
        losses = loss_fn(outputs, targets)
        wrapper.step(losses, model, optimizer)
    """

    def step(self, losses, model, optimizer, scaler=None):
        """Compute per-task gradients, project conflicts, update."""
        n_tasks = len(losses)
        task_grads = []

        # Compute per-task gradients
        for i, loss in enumerate(losses):
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            grads = []
            for p in model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.clone())
                else:
                    grads.append(torch.zeros_like(p))
            task_grads.append(grads)

        optimizer.zero_grad()

        # Project conflicting gradients
        projected = [list(g) for g in task_grads]
        for i in range(n_tasks):
            for j in range(i + 1, n_tasks):
                dot, norm_i, norm_j = 0.0, 0.0, 0.0
                for k in range(len(task_grads[i])):
                    dot += (task_grads[i][k] * task_grads[j][k]).sum().item()
                    norm_i += (task_grads[i][k] ** 2).sum().item()
                    norm_j += (task_grads[j][k] ** 2).sum().item()
                cos = dot / (np.sqrt(norm_i) * np.sqrt(norm_j) + 1e-10)

                if cos < 0:
                    for k in range(len(task_grads[i])):
                        proj_ij = (task_grads[i][k] * task_grads[j][k]).sum() / (norm_j + 1e-10)
                        projected[i][k] = task_grads[i][k] - proj_ij * task_grads[j][k]
                        proj_ji = (task_grads[j][k] * task_grads[i][k]).sum() / (norm_i + 1e-10)
                        projected[j][k] = task_grads[j][k] - proj_ji * task_grads[i][k]

        # Apply summed projected gradients
        for k, p in enumerate(model.parameters()):
            if p.requires_grad:
                p.grad = sum(projected[i][k] for i in range(n_tasks))

        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
