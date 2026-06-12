from __future__ import annotations

from pathlib import Path


WORKSPACE_DIR = Path("C:/Users/ishak/tibbi-segmentasyon")
ARTIFACT_ROOT = Path("D:/tibbi-segmentasyon")
RAW_ROOT = ARTIFACT_ROOT / "data" / "raw"
PROCESSED_ROOT = ARTIFACT_ROOT / "data" / "processed"
MANIFESTS_DIR = PROCESSED_ROOT / "manifests"
CACHE_ROOT = PROCESSED_ROOT / "cache"
RUNS_DIR = ARTIFACT_ROOT / "runs"
CHECKPOINTS_DIR = ARTIFACT_ROOT / "checkpoints"
REPORTS_DIR = ARTIFACT_ROOT / "reports"
TMP_DIR = ARTIFACT_ROOT / "tmp"


def ensure_project_dirs() -> None:
    for path in [
        RAW_ROOT,
        PROCESSED_ROOT,
        MANIFESTS_DIR,
        CACHE_ROOT,
        RUNS_DIR,
        CHECKPOINTS_DIR,
        REPORTS_DIR,
        TMP_DIR,
        WORKSPACE_DIR / "reports" / "figures",
        WORKSPACE_DIR / "reports" / "tables",
        WORKSPACE_DIR / "logs",
    ]:
        path.mkdir(parents=True, exist_ok=True)
