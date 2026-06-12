from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from segexp.paths import RAW_ROOT, ensure_project_dirs
from segexp.utils import now_stamp, write_json


ISIC_URLS = {
    "ISIC2018_Task1-2_Training_Input.zip": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip",
    "ISIC2018_Task1_Training_GroundTruth.zip": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip",
}

KAGGLE_DATASETS = {
    "lits_png": "andrewmvd/lits-png",
    "stroke_teknofest_colorized": "shuvokumarbasakbd/brain-stroke-dataset-colorized-teknofest-2021",
}

ACCESS_NOTES = {
    "brats2025": {
        "url": "https://www.synapse.org/brats2025",
        "status": "requires_synapse_login_or_token",
        "note": "No C:/Users/ishak/.synapseConfig was present at setup time. Script records this and continues.",
    },
    "luna16": {
        "url": "https://luna16.grand-challenge.org/",
        "status": "official_grand_challenge_dataset_large",
        "note": "Handled as a documented target unless direct archive links are provided or mirrored locally.",
    },
    "mammosightr": {
        "url": "https://acikveri.saglik.gov.tr/Home/DataSetDetail/3",
        "status": "national_open_data_portal_manual_access_likely",
        "note": "Portal access may require web session or manual approval; script records access state.",
    },
    "inme_bt": {
        "url": "https://acikveri.saglik.gov.tr/Home/DataSetDetail/1",
        "status": "national_open_data_portal_manual_access_likely",
        "note": "Official dataset is a project target; Kaggle colorized derivative can be downloaded separately.",
    },
}


def unzip_if_needed(zip_path: Path, dest: Path) -> None:
    marker = dest / f".unzipped_{zip_path.stem}"
    if marker.exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    marker.write_text(now_stamp(), encoding="utf-8")


def download_isic2018() -> dict:
    dest = RAW_ROOT / "isic2018"
    dest.mkdir(parents=True, exist_ok=True)
    result = {"dataset": "isic2018", "files": [], "status": "ok"}
    for filename, url in ISIC_URLS.items():
        zip_path = dest / filename
        if not zip_path.exists():
            urlretrieve(url, zip_path)
        unzip_if_needed(zip_path, dest)
        result["files"].append({"path": str(zip_path), "bytes": zip_path.stat().st_size})
    return result


def download_kaggle(name: str) -> dict:
    dataset = KAGGLE_DATASETS[name]
    dest = RAW_ROOT / name
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(dest),
        "--unzip",
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    return {
        "dataset": name,
        "kaggle_ref": dataset,
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "path": str(dest),
    }


def record_access_notes() -> dict:
    notes = {"datasets": ACCESS_NOTES, "synapse_config_present": Path.home().joinpath(".synapseConfig").exists()}
    path = RAW_ROOT / "access_notes.json"
    write_json(path, notes)
    return {"dataset": "access_notes", "status": "ok", "path": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--isic2018", action="store_true")
    parser.add_argument("--lits-png", action="store_true")
    parser.add_argument("--stroke-kaggle", action="store_true")
    args = parser.parse_args()
    ensure_project_dirs()
    selected = {
        "isic2018": args.all or args.isic2018,
        "lits_png": args.all or args.lits_png,
        "stroke_teknofest_colorized": args.all or args.stroke_kaggle,
    }
    results = [record_access_notes()]
    if selected["isic2018"]:
        try:
            results.append(download_isic2018())
        except Exception as exc:
            results.append({"dataset": "isic2018", "status": "failed", "error": repr(exc)})
    for name in ["lits_png", "stroke_teknofest_colorized"]:
        if selected[name]:
            try:
                results.append(download_kaggle(name))
            except Exception as exc:
                results.append({"dataset": name, "status": "failed", "error": repr(exc)})
    write_json(RAW_ROOT / f"download_log_{now_stamp()}.json", results)
    print(results)


if __name__ == "__main__":
    main()
