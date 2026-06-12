from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from segexp.data import scan_isic2018, scan_lits_png, write_manifest
from segexp.paths import MANIFESTS_DIR, RAW_ROOT, ensure_project_dirs
from segexp.utils import write_json


def positive_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    kept = []
    for row in rows:
        try:
            mask = np.array(Image.open(row["mask"]).convert("L"))
        except Exception:
            continue
        if bool((mask > 0).any()):
            kept.append(row)
    return kept


def main() -> None:
    ensure_project_dirs()
    summaries = []
    liver_rows = scan_lits_png(RAW_ROOT / "lits_png", target="liver", dataset="lits_liver")
    lesion_rows = scan_lits_png(RAW_ROOT / "lits_png", target="lesion", dataset="lits_lesion")
    scans = {
        "isic2018": scan_isic2018(RAW_ROOT / "isic2018"),
        "lits_liver": liver_rows,
        "lits_liver_positive": positive_rows(liver_rows),
        "lits_lesion": lesion_rows,
        "lits_lesion_positive": positive_rows(lesion_rows),
    }
    for name, rows in scans.items():
        path = MANIFESTS_DIR / f"{name}.csv"
        write_manifest(path, rows)
        summaries.append({"dataset": name, "pairs": len(rows), "manifest": str(path)})
    summary_path = MANIFESTS_DIR / "manifest_summary.json"
    write_json(summary_path, summaries)
    pd.DataFrame(summaries).to_csv(MANIFESTS_DIR / "manifest_summary.csv", index=False)
    print(summaries)


if __name__ == "__main__":
    main()
