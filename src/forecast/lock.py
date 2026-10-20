"""Hash-lock the Japan 2027 forecast and verify it later.

The manifest records SHA-256 of the raw bytes of each locked file, so anyone can check
it with `sha256sum`. Locked files are written with LF line endings and marked `-text`
in .gitattributes, so git never rewrites them on checkout.

The code tree hash is internal (it ties the forecast to the code that made it). It
normalises CRLF to LF so it is stable across Windows and Unix checkouts.

Time-stamping uses OpenTimestamps (`ots stamp`), which sends only the manifest's hash
to public calendar servers. It is never run implicitly.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.paths import ROOT

MANIFEST_VERSION = 1
CODE_GLOBS = ("src/**/*.py", "config/*.yaml")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_tree_sha256(root: Path = ROOT, globs: tuple[str, ...] = CODE_GLOBS) -> str:
    """Hash over (relative path, LF-normalised content) of every code/config file."""
    files = sorted({p for g in globs for p in root.glob(g) if p.is_file()})
    h = hashlib.sha256()
    for p in files:
        rel = p.relative_to(root).as_posix()
        content = p.read_bytes().replace(b"\r\n", b"\n")
        h.update(rel.encode() + b"\0" + hashlib.sha256(content).digest())
    return h.hexdigest()


def build_manifest(locked_files: list[Path], extra: dict | None = None, root: Path = ROOT) -> dict:
    return {
        "manifest_version": MANIFEST_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(locked_files)},
        "code_tree_sha256": code_tree_sha256(root),
        **(extra or {}),
    }


def write_manifest(manifest: dict, path: Path) -> str:
    """Write the manifest deterministically (sorted keys, LF) and return its SHA-256."""
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_bytes(text.encode("utf-8"))
    digest = sha256_file(path)
    path.with_name(path.name + ".sha256").write_bytes(f"{digest}  {path.name}\n".encode())
    return digest


def verify_manifest(path: Path, root: Path = ROOT) -> list[str]:
    """Return a list of problems; empty means every locked file matches."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.exists():
        expected = sidecar.read_text().split()[0]
        if sha256_file(path) != expected:
            problems.append(f"{path.name}: manifest itself was modified after locking")
    for rel, expected in manifest["files"].items():
        f = root / rel
        if not f.exists():
            problems.append(f"{rel}: missing")
        elif sha256_file(f) != expected:
            problems.append(f"{rel}: hash mismatch")
    return problems


def ots_stamp(path: Path) -> Path:
    """Submit `path`'s hash to OpenTimestamps calendars; returns the .ots receipt path."""
    exe = shutil.which("ots")
    if exe is None:
        raise RuntimeError("`ots` not found; install opentimestamps-client in the active environment")
    subprocess.run([exe, "stamp", str(path)], check=True)
    return path.with_name(path.name + ".ots")
