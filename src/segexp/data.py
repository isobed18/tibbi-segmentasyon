from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _norm_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "mask", "dataset", "case_id"])
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit is not None:
        rows = rows[:limit]
    return rows


def scan_isic2018(root: Path) -> list[dict[str, str]]:
    candidates = list(root.rglob("ISIC2018_Task1-2_Training_Input"))
    image_dir = candidates[0] if candidates else root
    mask_candidates = list(root.rglob("ISIC2018_Task1_Training_GroundTruth"))
    mask_dir = mask_candidates[0] if mask_candidates else root
    rows: list[dict[str, str]] = []
    for image_path in sorted(image_dir.glob("ISIC_*.jpg")):
        stem = image_path.stem
        possible = [
            mask_dir / f"{stem}_segmentation.png",
            mask_dir / f"{stem}.png",
        ]
        mask_path = next((p for p in possible if p.exists()), None)
        if mask_path:
            rows.append({"image": _norm_path(image_path), "mask": _norm_path(mask_path), "dataset": "isic2018", "case_id": stem})
    return rows


def _mask_score(path: Path) -> int:
    name = path.name.lower()
    score = 0
    for token in ["mask", "seg", "label", "annotation", "ground", "gt", "lesion"]:
        if token in name:
            score += 1
    return score


def scan_generic_pairs(root: Path, dataset: str) -> list[dict[str, str]]:
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    mask_files = [p for p in files if _mask_score(p) > 0 or any(_mask_score(parent) > 0 for parent in p.parents)]
    image_files = [p for p in files if p not in set(mask_files)]
    mask_by_stem = {p.stem.lower().replace("_mask", "").replace("_segmentation", "").replace("-mask", ""): p for p in mask_files}
    rows: list[dict[str, str]] = []
    for image_path in sorted(image_files):
        stem = image_path.stem.lower()
        match = mask_by_stem.get(stem)
        if match is None:
            for key, mask_path in mask_by_stem.items():
                if key in stem or stem in key:
                    match = mask_path
                    break
        if match is not None:
            rows.append(
                {
                    "image": _norm_path(image_path),
                    "mask": _norm_path(match),
                    "dataset": dataset,
                    "case_id": image_path.stem,
                }
            )
    return rows


def scan_lits_png(root: Path, target: str = "lesion", dataset: str | None = None) -> list[dict[str, str]]:
    if target not in {"lesion", "liver"}:
        raise ValueError("target must be 'lesion' or 'liver'")
    dataset = dataset or f"lits_{target}"
    files = [p for p in root.rglob("*.png")]
    images: dict[tuple[str, str], Path] = {}
    liver_masks: dict[tuple[str, str], Path] = {}
    lesion_masks: dict[tuple[str, str], Path] = {}
    for path in files:
        stem = path.stem.lower()
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        prefix, slice_id = parts[0], parts[-1]
        if "-" not in prefix:
            continue
        kind, case_id = prefix.split("-", 1)
        key = (case_id, slice_id)
        if kind == "volume":
            images[key] = path
        elif kind == "segmentation" and "lesionmask" in stem:
            lesion_masks[key] = path
        elif kind == "segmentation" and "livermask" in stem:
            liver_masks[key] = path
    rows: list[dict[str, str]] = []
    for key, image_path in sorted(images.items()):
        mask_path = lesion_masks.get(key) if target == "lesion" else liver_masks.get(key)
        if mask_path:
            rows.append(
                {
                    "image": _norm_path(image_path),
                    "mask": _norm_path(mask_path),
                    "dataset": dataset,
                    "case_id": f"{key[0]}_{key[1]}",
                }
            )
    if rows:
        return rows
    return scan_generic_pairs(root, dataset)


class SegmentationDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], image_size: int, in_channels: int, augment: bool = False) -> None:
        self.rows = rows
        self.image_size = image_size
        self.in_channels = in_channels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        image = self._read_image(row["image"])
        mask = self._read_mask(Path(row["mask"]))
        if self.augment:
            image, mask = self._augment(image, mask)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if self.in_channels == 1 and image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if self.in_channels == 3 and image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        image = image.astype(np.float32) / 255.0
        mask = (mask.astype(np.float32) > 0).astype(np.float32)
        if image.ndim == 2:
            image = image[None, ...]
        else:
            image = image.transpose(2, 0, 1)
        mask = mask[None, ...]
        return torch.from_numpy(image), torch.from_numpy(mask)

    @staticmethod
    def _read_image(path: str | Path) -> np.ndarray:
        text = str(path)
        if "|" in text:
            channels = []
            for part in text.split("|"):
                arr = SegmentationDataset._read_single_image(Path(part))
                if arr.ndim == 3:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                channels.append(arr)
            return np.stack(channels, axis=2)
        return SegmentationDataset._read_single_image(Path(text))

    @staticmethod
    def _read_single_image(path: Path) -> np.ndarray:
        with Image.open(path) as im:
            if im.mode in {"I;16", "I"}:
                arr = np.array(im)
                arr = arr.astype(np.float32)
                lo, hi = np.percentile(arr, [1, 99])
                arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
                return (arr * 255).astype(np.uint8)
            return np.array(im.convert("RGB"))

    @staticmethod
    def _read_mask(path: Path) -> np.ndarray:
        with Image.open(path) as im:
            return np.array(im.convert("L"))

    @staticmethod
    def _augment(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if random.random() < 0.5:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        if random.random() < 0.25:
            k = random.choice([1, 2, 3])
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()
        return image, mask


class CachedArrayDataset(Dataset):
    def __init__(self, cache_dir: Path, split: str, augment: bool = False) -> None:
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.augment = augment
        self.images = np.load(self.cache_dir / f"{split}_images.npy", mmap_mode="r")
        self.masks = np.load(self.cache_dir / f"{split}_masks.npy", mmap_mode="r")

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = np.asarray(self.images[idx]).copy()
        mask = np.asarray(self.masks[idx]).copy()
        if self.augment:
            image, mask = self._augment_chw(image, mask)
        image = image.astype(np.float32) / 255.0
        mask = (mask.astype(np.float32) > 0).astype(np.float32)
        return torch.from_numpy(image), torch.from_numpy(mask)

    @staticmethod
    def _augment_chw(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=2).copy()
        if random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if random.random() < 0.25:
            k = random.choice([1, 2, 3])
            image = np.rot90(image, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(1, 2)).copy()
        return image, mask


def cache_name(dataset_name: str, image_size: int, seed: int, max_train_items: int | None, max_val_items: int | None) -> str:
    train_part = "all" if max_train_items is None else str(max_train_items)
    val_part = "all" if max_val_items is None else str(max_val_items)
    stable_seed = int(seed) % (2**32 - 1)
    return f"{dataset_name}_s{image_size}_seed{stable_seed}_tr{train_part}_va{val_part}"


def cache_dir_for(
    cache_root: Path,
    dataset_name: str,
    image_size: int,
    seed: int,
    max_train_items: int | None,
    max_val_items: int | None,
) -> Path:
    return Path(cache_root) / cache_name(dataset_name, image_size, seed, max_train_items, max_val_items)


def preprocess_row(row: dict[str, str], image_size: int, in_channels: int) -> tuple[np.ndarray, np.ndarray]:
    image = SegmentationDataset._read_image(row["image"])
    mask = SegmentationDataset._read_mask(Path(row["mask"]))
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    if in_channels == 1 and image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if in_channels == 3 and image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim == 2:
        image = image[None, ...]
    else:
        image = image.transpose(2, 0, 1)
    mask = (mask > 0).astype(np.uint8)[None, ...] * 255
    return image.astype(np.uint8), mask


class SyntheticShapesDataset(Dataset):
    def __init__(self, length: int, image_size: int, in_channels: int = 1, seed: int = 0) -> None:
        self.length = length
        self.image_size = image_size
        self.in_channels = in_channels
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng((self.seed + idx) % (2**32 - 1))
        size = self.image_size
        image = rng.normal(0.08, 0.025, size=(size, size)).astype(np.float32)
        mask = np.zeros((size, size), dtype=np.float32)
        center = (rng.integers(size // 4, 3 * size // 4), rng.integers(size // 4, 3 * size // 4))
        axes = (rng.integers(size // 10, size // 4), rng.integers(size // 10, size // 4))
        angle = int(rng.integers(0, 180))
        cv2.ellipse(mask, center, axes, angle, 0, 360, 1.0, -1)
        image += mask * rng.uniform(0.45, 0.8)
        image = np.clip(image, 0, 1)
        if self.in_channels == 3:
            image = np.repeat(image[..., None], 3, axis=2).transpose(2, 0, 1)
        else:
            image = image[None, ...]
        return torch.from_numpy(image.astype(np.float32)), torch.from_numpy(mask[None, ...].astype(np.float32))


def split_rows(rows: list[dict[str, str]], val_fraction: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    shuffled = rows[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    val_count = max(1, int(math.ceil(len(shuffled) * val_fraction)))
    return shuffled[val_count:], shuffled[:val_count]
