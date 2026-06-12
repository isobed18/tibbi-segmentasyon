from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from segexp.data import read_manifest, write_manifest
from segexp.paths import MANIFESTS_DIR, ensure_project_dirs


VOLUME_RE = re.compile(r"^(volume-\d+)_([0-9]+)\.png$", re.IGNORECASE)


def neighbor_triplet(image_path: Path) -> str:
    match = VOLUME_RE.match(image_path.name)
    if not match:
        return "|".join([str(image_path)] * 3)
    prefix, slice_text = match.groups()
    slice_idx = int(slice_text)
    parts = []
    for offset in (-1, 0, 1):
        candidate = image_path.with_name(f"{prefix}_{slice_idx + offset}.png")
        parts.append(str(candidate if candidate.exists() else image_path))
    return "|".join(parts).replace("\\", "/")


def build_25d_rows(source_name: str, target_name: str) -> list[dict[str, str]]:
    rows = read_manifest(MANIFESTS_DIR / f"{source_name}.csv")
    out = []
    for row in rows:
        out.append(
            {
                "image": neighbor_triplet(Path(row["image"])),
                "mask": row["mask"],
                "dataset": target_name,
                "case_id": row["case_id"],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="lits_lesion_positive")
    parser.add_argument("--target", default="lits_lesion_positive_25d")
    args = parser.parse_args()
    ensure_project_dirs()
    rows = build_25d_rows(args.source, args.target)
    out_path = MANIFESTS_DIR / f"{args.target}.csv"
    write_manifest(out_path, rows)

    summary_path = MANIFESTS_DIR / "manifest_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        summary = summary[summary["dataset"] != args.target]
    else:
        summary = pd.DataFrame(columns=["dataset", "pairs", "manifest"])
    summary = pd.concat(
        [
            summary,
            pd.DataFrame([{"dataset": args.target, "pairs": len(rows), "manifest": str(out_path)}]),
        ],
        ignore_index=True,
    )
    summary.to_csv(summary_path, index=False)
    print({"dataset": args.target, "pairs": len(rows), "manifest": str(out_path)})


if __name__ == "__main__":
    main()
