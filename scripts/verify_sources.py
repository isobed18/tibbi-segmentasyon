from __future__ import annotations

import csv
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from scripts.generate_ieee_report import REFERENCES
from segexp.paths import WORKSPACE_DIR, ensure_project_dirs


def check_url(url: str, timeout: int = 20) -> dict[str, str | int]:
    headers = {"User-Agent": "CodexSourceVerifier/1.0"}
    for method in ("HEAD", "GET"):
        request = urllib.request.Request(url, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return {
                    "http_ok": "true",
                    "method": method,
                    "status_code": int(response.status),
                    "final_url": response.geturl(),
                    "error": "",
                }
        except urllib.error.HTTPError as exc:
            if method == "HEAD" and exc.code in {403, 405}:
                continue
            return {
                "http_ok": "false",
                "method": method,
                "status_code": int(exc.code),
                "final_url": url,
                "error": str(exc.reason),
            }
        except Exception as exc:
            last_error = repr(exc)
            if method == "HEAD":
                continue
            return {
                "http_ok": "false",
                "method": method,
                "status_code": "",
                "final_url": url,
                "error": last_error,
            }
    return {"http_ok": "false", "method": "", "status_code": "", "final_url": url, "error": "unreachable"}


def main() -> None:
    ensure_project_dirs()
    out = WORKSPACE_DIR / "reports" / "tables" / "source_http_check.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for ref in REFERENCES:
        result = check_url(ref["url"])
        rows.append({**ref, **result})
        time.sleep(0.2)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
