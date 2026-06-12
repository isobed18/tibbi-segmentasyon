from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from monai.losses import DiceLoss, FocalLoss, TverskyLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .data import CachedArrayDataset, SegmentationDataset, SyntheticShapesDataset, cache_dir_for, read_manifest, split_rows
from .metrics import AverageMeter, dice_iou_from_logits
from .models import build_model, profile_model
from .paths import CACHE_ROOT, CHECKPOINTS_DIR, MANIFESTS_DIR, RUNS_DIR
from .utils import now_stamp, seed_everything, write_json


def _make_loaders(
    dataset_name: str,
    image_size: int,
    in_channels: int,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
    seed: int,
    max_train_items: int | None,
    max_val_items: int | None,
    use_cache: bool = False,
    sampler_name: str = "none",
) -> tuple[DataLoader, DataLoader, int]:
    if dataset_name == "synthetic":
        train_ds = SyntheticShapesDataset(max_train_items or 64, image_size, in_channels=in_channels, seed=seed)
        val_ds = SyntheticShapesDataset(max_val_items or 16, image_size, in_channels=in_channels, seed=seed + 10_000)
        total = len(train_ds) + len(val_ds)
    else:
        rows = read_manifest(MANIFESTS_DIR / f"{dataset_name}.csv")
        if len(rows) < 4:
            raise RuntimeError(f"Manifest for {dataset_name} has too few samples: {len(rows)}")
        train_rows, val_rows = split_rows(rows, val_fraction, seed)
        if max_train_items:
            train_rows = train_rows[:max_train_items]
        if max_val_items:
            val_rows = val_rows[:max_val_items]
        cache_dir = cache_dir_for(CACHE_ROOT, dataset_name, image_size, seed, max_train_items, max_val_items)
        if use_cache and (cache_dir / "train_images.npy").exists() and (cache_dir / "val_images.npy").exists():
            train_ds = CachedArrayDataset(cache_dir, "train", augment=True)
            val_ds = CachedArrayDataset(cache_dir, "val", augment=False)
        else:
            train_ds = SegmentationDataset(train_rows, image_size, in_channels=in_channels, augment=True)
            val_ds = SegmentationDataset(val_rows, image_size, in_channels=in_channels, augment=False)
        total = len(rows)
    train_sampler = None
    if sampler_name == "foreground":
        weights = _foreground_sampler_weights(train_ds)
        train_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    elif sampler_name != "none":
        raise ValueError(f"Unknown sampler: {sampler_name}")
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, total


def _foreground_sampler_weights(dataset) -> torch.DoubleTensor:
    if isinstance(dataset, CachedArrayDataset):
        masks = np.asarray(dataset.masks)
        foreground = (masks.reshape(masks.shape[0], -1) > 0).mean(axis=1)
    elif isinstance(dataset, SegmentationDataset):
        foreground = []
        for row in dataset.rows:
            mask = SegmentationDataset._read_mask(Path(row["mask"]))
            foreground.append(float((mask > 0).mean()))
        foreground = np.asarray(foreground, dtype=np.float64)
    else:
        foreground = np.ones(len(dataset), dtype=np.float64)
    weights = np.sqrt(np.maximum(foreground, 1e-6))
    weights = weights / max(float(weights.mean()), 1e-6)
    return torch.as_tensor(weights, dtype=torch.double)


def _build_criterion(loss_name: str):
    loss_name = loss_name.lower()
    bce_loss = torch.nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    focal_loss = FocalLoss(to_onehot_y=False, gamma=2.0, reduction="mean")
    tversky_loss = TverskyLoss(sigmoid=True, alpha=0.3, beta=0.7, reduction="mean")

    if loss_name == "bce_dice":
        return lambda logits, mask: bce_loss(logits, mask) + dice_loss(logits, mask)
    if loss_name == "dice_focal":
        return lambda logits, mask: dice_loss(logits, mask) + focal_loss(logits, mask)
    if loss_name == "tversky_focal":
        return lambda logits, mask: tversky_loss(logits, mask) + 0.5 * focal_loss(logits, mask)
    raise ValueError(f"Unknown loss: {loss_name}")


def _run_epoch(model, loader, optimizer, scaler, criterion, device, amp: bool, train: bool) -> dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    start = time.perf_counter()
    for image, mask in tqdm(loader, leave=False):
        image = image.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(image)
                loss = criterion(logits, mask)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        dice, iou = dice_iou_from_logits(logits.detach(), mask.detach())
        batch_n = image.shape[0]
        loss_meter.update(loss.item(), batch_n)
        dice_meter.update(dice, batch_n)
        iou_meter.update(iou, batch_n)
    elapsed = time.perf_counter() - start
    return {
        "loss": loss_meter.avg,
        "dice": dice_meter.avg,
        "iou": iou_meter.avg,
        "seconds": elapsed,
        "samples_per_second": len(loader.dataset) / max(elapsed, 1e-6),
    }


def benchmark_inference(model, loader, device, amp: bool, warmup_batches: int = 2, max_batches: int = 12) -> dict[str, float]:
    model.eval()
    times: list[float] = []
    samples = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for idx, (image, _) in enumerate(loader):
            if idx >= warmup_batches + max_batches:
                break
            image = image.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                _ = model(image)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            if idx >= warmup_batches:
                times.append(elapsed)
                samples += image.shape[0]
    total = sum(times)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return {
        "inference_seconds": total,
        "inference_samples": samples,
        "latency_ms_per_batch": 1000 * total / max(1, len(times)),
        "latency_ms_per_image": 1000 * total / max(1, samples),
        "fps": samples / max(total, 1e-6),
        "peak_memory_mb": round(peak / 1024**2, 2),
    }


def train_one_experiment(
    dataset_name: str,
    model_name: str,
    in_channels: int,
    image_size: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    amp: bool,
    num_workers: int,
    val_fraction: float,
    max_train_items: int | None,
    max_val_items: int | None,
    seed: int,
    run_root: Path | None = None,
    use_cache: bool = False,
    loss_name: str = "bce_dice",
    sampler_name: str = "none",
) -> dict[str, Any]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, total_items = _make_loaders(
        dataset_name,
        image_size,
        in_channels,
        batch_size,
        num_workers,
        val_fraction,
        seed,
        max_train_items,
        max_val_items,
        use_cache,
        sampler_name,
    )
    model = build_model(model_name, in_channels=in_channels, image_size=image_size).to(device)
    profile = profile_model(model, in_channels, image_size, device)
    criterion = _build_criterion(loss_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    run_id = f"{now_stamp()}_{dataset_name}_{model_name}"
    run_dir = (run_root or RUNS_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_dice = -1.0
    best_path = CHECKPOINTS_DIR / f"{run_id}_best.pt"
    start_all = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, scaler, criterion, device, amp, train=True)
        val_metrics = _run_epoch(model, val_loader, optimizer, scaler, criterion, device, amp, train=False)
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save({"model": model.state_dict(), "config": {"dataset": dataset_name, "model": model_name}}, best_path)
    inference = benchmark_inference(model, val_loader, device, amp)
    elapsed_all = time.perf_counter() - start_all
    summary = {
        "run_id": run_id,
        "dataset": dataset_name,
        "model": model_name,
        "device": str(device),
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "use_cache": use_cache,
        "loss_name": loss_name,
        "sampler_name": sampler_name,
        "total_manifest_items": total_items,
        "train_items": len(train_loader.dataset),
        "val_items": len(val_loader.dataset),
        "best_val_dice": best_dice,
        "final_val_dice": history[-1]["val_dice"],
        "final_val_iou": history[-1]["val_iou"],
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "total_seconds": elapsed_all,
        "checkpoint": str(best_path),
        **profile,
        **inference,
    }
    write_json(run_dir / "summary.json", summary)
    return summary
