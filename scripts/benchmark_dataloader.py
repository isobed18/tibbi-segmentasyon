from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from segexp.data import CachedArrayDataset, SegmentationDataset, cache_dir_for, read_manifest, split_rows
from segexp.paths import CACHE_ROOT, MANIFESTS_DIR, WORKSPACE_DIR, ensure_project_dirs
from segexp.utils import load_yaml


def dataset_channels(dataset_name: str) -> int:
    return 3 if dataset_name == "isic2018" else 1


def bench(loader: DataLoader, max_batches: int) -> dict[str, float]:
    samples = 0
    start = time.perf_counter()
    first_batch_seconds = None
    for idx, (image, mask) in enumerate(loader):
        if idx == 0:
            first_batch_seconds = time.perf_counter() - start
        samples += int(image.shape[0])
        if idx + 1 >= max_batches:
            break
    elapsed = time.perf_counter() - start
    return {
        "samples": samples,
        "batches": min(max_batches, len(loader)),
        "seconds": elapsed,
        "first_batch_seconds": first_batch_seconds or elapsed,
        "samples_per_second": samples / max(elapsed, 1e-6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(WORKSPACE_DIR / "configs" / "experiments.yaml"))
    parser.add_argument("--preset", default="standard", choices=["quick", "standard"])
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--max-batches", type=int, default=40)
    args = parser.parse_args()
    ensure_project_dirs()
    cfg = load_yaml(args.config)
    base = cfg["training"]
    preset = cfg["presets"][args.preset]
    image_size = int(preset.get("image_size", base["image_size"]))
    batch_size = int(preset.get("batch_size", base["batch_size"]))
    num_workers = int(base["num_workers"])
    seed = int(cfg["project"]["seed"])
    max_train_items = int(preset.get("max_train_items", base["max_train_items"]))
    max_val_items = int(preset.get("max_val_items", base["max_val_items"]))
    results = []
    for dataset_name in args.dataset:
        rows = read_manifest(MANIFESTS_DIR / f"{dataset_name}.csv")
        train_rows, _ = split_rows(rows, float(base["val_fraction"]), seed)
        train_rows = train_rows[:max_train_items]
        in_channels = dataset_channels(dataset_name)
        raw_ds = SegmentationDataset(train_rows, image_size, in_channels=in_channels, augment=True)
        raw_loader = DataLoader(raw_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        raw = bench(raw_loader, args.max_batches)
        raw.update({"dataset": dataset_name, "mode": "raw"})
        results.append(raw)

        cache_dir = cache_dir_for(CACHE_ROOT, dataset_name, image_size, seed, max_train_items, max_val_items)
        if (cache_dir / "train_images.npy").exists():
            cached_ds = CachedArrayDataset(cache_dir, "train", augment=True)
            cached_loader = DataLoader(cached_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
            cached = bench(cached_loader, args.max_batches)
            cached.update({"dataset": dataset_name, "mode": "cache"})
            results.append(cached)
    df = pd.DataFrame(results)
    out = WORKSPACE_DIR / "reports" / "tables" / "dataloader_benchmark.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(out)


if __name__ == "__main__":
    main()

