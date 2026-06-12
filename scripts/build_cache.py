from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from segexp.data import cache_dir_for, preprocess_row, read_manifest, split_rows
from segexp.paths import CACHE_ROOT, MANIFESTS_DIR, ensure_project_dirs
from segexp.utils import load_yaml, write_json


def dataset_channels(dataset_name: str) -> int:
    return 3 if dataset_name == "isic2018" or dataset_name.endswith("_25d") else 1


def build_split(rows: list[dict[str, str]], image_size: int, in_channels: int) -> tuple[np.ndarray, np.ndarray]:
    images = np.empty((len(rows), in_channels, image_size, image_size), dtype=np.uint8)
    masks = np.empty((len(rows), 1, image_size, image_size), dtype=np.uint8)
    for idx, row in enumerate(tqdm(rows, leave=False)):
        image, mask = preprocess_row(row, image_size, in_channels)
        images[idx] = image
        masks[idx] = mask
    return images, masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--preset", default="standard", choices=["smoke", "quick", "standard"])
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ensure_project_dirs()
    cfg = load_yaml(args.config)
    base = cfg["training"]
    preset = cfg["presets"][args.preset]
    image_size = int(preset.get("image_size", base["image_size"]))
    max_train_items = int(preset.get("max_train_items", base["max_train_items"]))
    max_val_items = int(preset.get("max_val_items", base["max_val_items"]))
    seed = int(cfg["project"]["seed"])
    val_fraction = float(base["val_fraction"])
    results = []
    for dataset_name in args.dataset:
        start = time.perf_counter()
        rows = read_manifest(MANIFESTS_DIR / f"{dataset_name}.csv")
        train_rows, val_rows = split_rows(rows, val_fraction, seed)
        train_rows = train_rows[:max_train_items]
        val_rows = val_rows[:max_val_items]
        cache_dir = cache_dir_for(CACHE_ROOT, dataset_name, image_size, seed, max_train_items, max_val_items)
        done = cache_dir / "cache_meta.json"
        if done.exists() and not args.force:
            results.append({"dataset": dataset_name, "status": "exists", "cache_dir": str(cache_dir)})
            continue
        cache_dir.mkdir(parents=True, exist_ok=True)
        in_channels = dataset_channels(dataset_name)
        train_images, train_masks = build_split(train_rows, image_size, in_channels)
        val_images, val_masks = build_split(val_rows, image_size, in_channels)
        np.save(cache_dir / "train_images.npy", train_images)
        np.save(cache_dir / "train_masks.npy", train_masks)
        np.save(cache_dir / "val_images.npy", val_images)
        np.save(cache_dir / "val_masks.npy", val_masks)
        meta = {
            "dataset": dataset_name,
            "image_size": image_size,
            "in_channels": in_channels,
            "seed": seed % (2**32 - 1),
            "train_items": len(train_rows),
            "val_items": len(val_rows),
            "seconds": time.perf_counter() - start,
            "cache_dir": str(cache_dir),
            "format": "uint8 NCHW images, uint8 N1HW masks",
        }
        write_json(done, meta)
        results.append({"dataset": dataset_name, "status": "built", **meta})
    print(results)


if __name__ == "__main__":
    main()
