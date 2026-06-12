from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from segexp.paths import RUNS_DIR, WORKSPACE_DIR, ensure_project_dirs
from segexp.train import train_one_experiment
from segexp.utils import env_summary, load_yaml, now_stamp, write_json


def dataset_channels(dataset_name: str) -> int:
    return 3 if dataset_name == "isic2018" or dataset_name.endswith("_25d") else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(WORKSPACE_DIR / "configs" / "experiments.yaml"))
    parser.add_argument("--preset", default="smoke", choices=["smoke", "quick", "standard"])
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--model", action="append")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--loss", default="bce_dice", choices=["bce_dice", "dice_focal", "tversky_focal"])
    parser.add_argument("--sampler", default="none", choices=["none", "foreground"])
    parser.add_argument("--run-label", default=None)
    args = parser.parse_args()
    ensure_project_dirs()
    cfg = load_yaml(args.config)
    base = cfg["training"]
    preset = cfg["presets"][args.preset]
    datasets = args.dataset or preset["datasets"]
    models = args.model or preset["models"]
    label = args.run_label or args.preset
    run_root = RUNS_DIR / f"{now_stamp()}_{label}"
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "environment.json", env_summary())
    summaries = []
    for dataset_name in datasets:
        for model_name in models:
            try:
                summary = train_one_experiment(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    in_channels=dataset_channels(dataset_name),
                    image_size=int(preset.get("image_size", base["image_size"])),
                    batch_size=int(preset.get("batch_size", base["batch_size"])),
                    epochs=int(preset["epochs"]),
                    lr=float(base["learning_rate"]),
                    weight_decay=float(base["weight_decay"]),
                    amp=bool(base["amp"]),
                    num_workers=int(base["num_workers"]),
                    val_fraction=float(base["val_fraction"]),
                    max_train_items=int(preset.get("max_train_items", base["max_train_items"])),
                    max_val_items=int(preset.get("max_val_items", base["max_val_items"])),
                    seed=int(cfg["project"]["seed"]),
                    run_root=run_root,
                    use_cache=args.cache,
                    loss_name=args.loss,
                    sampler_name=args.sampler,
                )
                summaries.append(summary)
            except Exception as exc:
                summaries.append({"dataset": dataset_name, "model": model_name, "status": "failed", "error": repr(exc)})
                write_json(run_root / f"failed_{dataset_name}_{model_name}.json", summaries[-1])
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(run_root / "summary.csv", index=False)
    write_json(run_root / "summary.json", summaries)
    latest = RUNS_DIR / "latest_summary.csv"
    summary_df.to_csv(latest, index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
