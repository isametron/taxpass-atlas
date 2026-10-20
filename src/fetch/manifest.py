"""Record every raw download in data/raw/MANIFEST.json (committed; the data is not).

Each entry: path -> {url, params, fetched_utc, sha256, bytes}. Re-fetching a file
overwrites its entry, so the manifest always describes what is on disk.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.forecast.lock import sha256_file
from src.paths import DATA_RAW, ROOT

MANIFEST = DATA_RAW / "MANIFEST.json"


def _load() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {}


def record(path: Path, url: str, params: dict | None = None) -> None:
    entries = _load()
    entries[path.relative_to(ROOT).as_posix()] = {
        "url": url,
        "params": {k: v for k, v in (params or {}).items() if k.lower() not in {"appid", "token"}},
        "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(entries, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    MANIFEST.write_bytes(text.encode("utf-8"))


def write_json(path: Path, obj, url: str, params: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    record(path, url, params)
    return path
