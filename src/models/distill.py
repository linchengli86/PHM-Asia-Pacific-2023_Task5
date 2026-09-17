"""Task-to-Task Knowledge Distillation (锦上添花 3).

Hinton et al., "Distilling the Knowledge in a Neural Network", 2015

Core idea: Use the simple task (task1: anomaly detection, lots of data) as a
teacher to guide the difficult tasks (task3/4: fine-grained localization, few samples).

Teacher's soft labels contain richer information than hard one-hot labels:
  hard: [0, 0, 1, 0, 0, 0, 0, 0, 0]  ← just "class 2"
  soft: [0.02, 0.05, 0.70, 0.15, 0.03, 0.02, 0.01, 0.01, 0.01]  ← class 2, but class 3 is plausible
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """Combined hard label + soft distillation loss.

    L = (1-alpha) * L_CE(y_true, y_pred) + alpha * T^2 * KL(softmax(teacher/T) || softmax(student/T))
    """

    def __init__(self, temperature=3.0, alpha=0.3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, hard_labels):
        """Compute distillation loss.

        Args:
            student_logits: (B, K) from student model
            teacher_logits: (B, K) from teacher model (no_grad)
            hard_labels: (B,) integer labels

        Returns:
            combined_loss
        """
        # Hard label loss
        hard_loss = F.cross_entropy(student_logits, hard_labels)

        # Soft distillation loss
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        distill_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean")

        return (1 - self.alpha) * hard_loss + self.alpha * (self.temperature ** 2) * distill_loss


class TaskDistiller:
    """Distill knowledge from task1 to task3/4.

    Strategy:
      Stage 1: Train teacher (full 5-task model) on all data
      Stage 2: Freeze teacher. Train student — task3/4 heads receive
               extra distillation loss from teacher's task3/4 logits.
    """

    def __init__(self, teacher_model, cfg):
        self.teacher = teacher_model
        distill_cfg = cfg.get("distill", {})
        self.loss_fn = DistillationLoss(
            temperature=distill_cfg.get("temperature", 3.0),
            alpha=distill_cfg.get("alpha", 0.3),
        )
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def get_teacher_logits(self, x):
        return self.teacher(x)

    def compute_student_loss(self, student_outputs, teacher_logits, hard_labels):
        """Loss for task3 and task4 heads with distillation."""
        o3_student, o4_student = student_outputs[2], student_outputs[3]
        o3_teacher, o4_teacher = teacher_logits[2], teacher_logits[3]

        loss3 = self.loss_fn(o3_student, o3_teacher, hard_labels["task3"])
        loss4 = self.loss_fn(o4_student, o4_teacher, hard_labels["task4"])
        return loss3 + loss4
