from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from segexp.data import SyntheticShapesDataset
from segexp.metrics import dice_iou_from_logits
from segexp.models import build_model


def test_synthetic_dataset_shapes() -> None:
    ds = SyntheticShapesDataset(length=2, image_size=64, in_channels=1, seed=1)
    image, mask = ds[0]
    assert image.shape == (1, 64, 64)
    assert mask.shape == (1, 64, 64)
    assert image.dtype == torch.float32


def test_unet_forward_and_metric() -> None:
    model = build_model("unet", in_channels=1, image_size=64)
    x = torch.randn(2, 1, 64, 64)
    y = torch.zeros(2, 1, 64, 64)
    logits = model(x)
    assert logits.shape == y.shape
    dice, iou = dice_iou_from_logits(logits, y)
    assert 0 <= dice <= 1
    assert 0 <= iou <= 1

