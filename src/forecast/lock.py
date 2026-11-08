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
import os
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


CALENDARS = ("https://alice.btc.calendar.opentimestamps.org",
             "https://bob.btc.calendar.opentimestamps.org",
             "https://finney.calendar.eternitywall.com")
MIN_CALENDARS = 2


def ots_stamp(path: Path, calendar_cls=None, calendars: tuple[str, ...] = CALENDARS, timeout: int = 30) -> Path:
    """Submit a salted hash of `path` to OpenTimestamps calendars; returns the .ots receipt path.

    Uses the opentimestamps library directly because the `ots` command crashes at import on
    Windows (python-bitcoinlib looks for an OpenSSL DLL that is absent). The receipt format is
    the same, so `ots upgrade` and `ots verify` work on it from any machine with a working client.
    """
    from opentimestamps.calendar import RemoteCalendar
    from opentimestamps.core.op import OpAppend, OpSHA256
    from opentimestamps.core.serialize import StreamSerializationContext
    from opentimestamps.core.timestamp import DetachedTimestampFile

    calendar_cls = calendar_cls or RemoteCalendar
    with open(path, "rb") as f:
        stamp = DetachedTimestampFile.from_fd(OpSHA256(), f)
    salted = stamp.timestamp.ops.add(OpAppend(os.urandom(16)))
    tip = salted.ops.add(OpSHA256())
    accepted = 0
    errors = []
    for url in calendars:
        try:
            tip.merge(calendar_cls(url).submit(tip.msg, timeout=timeout))
            accepted += 1
        except Exception as e:  # a calendar being down is expected; the threshold below decides
            errors.append(f"{url}: {type(e).__name__}")
    if accepted < MIN_CALENDARS:
        raise RuntimeError(f"only {accepted} calendars accepted the stamp: {errors}")
    out = path.with_name(path.name + ".ots")
    with open(out, "wb") as f:
        stamp.serialize(StreamSerializationContext(f))
    return out
