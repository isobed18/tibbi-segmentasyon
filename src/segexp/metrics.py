from __future__ import annotations

import torch


def binary_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


def dice_iou_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> tuple[float, float]:
    pred = (binary_probs(logits) >= threshold).float()
    target = (target > 0.5).float()
    dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dim=dims)
    pred_sum = pred.sum(dim=dims)
    target_sum = target.sum(dim=dims)
    union = pred_sum + target_sum - intersection
    eps = 1e-7
    dice = ((2 * intersection + eps) / (pred_sum + target_sum + eps)).mean().item()
    iou = ((intersection + eps) / (union + eps)).mean().item()
    return dice, iou


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)

