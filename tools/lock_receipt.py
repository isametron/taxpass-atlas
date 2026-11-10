"""Verify, time-stamp or upgrade the Japan 2027 forecast lock.

  python -m tools.lock_receipt verify    check every locked file against the manifest
  python -m tools.lock_receipt stamp     submit a salted hash of the manifest to the calendars
  python -m tools.lock_receipt upgrade   after Bitcoin confirms (hours later), add the proof

This lives outside src/ on purpose: the lock manifest records a hash over src/**/*.py, so
adding code there after the lock would make the current tree stop matching it.

`upgrade` does the job of `ots upgrade`, which crashes on Windows (python-bitcoinlib looks
for an OpenSSL DLL that is absent). The receipt format is unchanged, so `ots verify` works
on it from any machine with a working client.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.forecast import lock
from src.paths import FORECASTS


def _pending_nodes(timestamp):
    """Yield (node, calendar url) for every pending attestation in a timestamp tree."""
    from opentimestamps.core.notary import PendingAttestation
    for att in list(timestamp.attestations):
        if isinstance(att, PendingAttestation):
            yield timestamp, att.uri
    for child in list(timestamp.ops.values()):
        yield from _pending_nodes(child)


def bitcoin_attestations(timestamp) -> list[dict]:
    """Bitcoin block heights in a receipt, each with the merkle root a block explorer shows.

    The attested message is the merkle root in internal byte order; explorers print it reversed.
    """
    from opentimestamps.core.notary import BitcoinBlockHeaderAttestation
    return sorted(({"height": att.height, "merkle_root": msg[::-1].hex()}
                   for msg, att in timestamp.all_attestations()
                   if isinstance(att, BitcoinBlockHeaderAttestation)), key=lambda r: r["height"])


def ots_upgrade(receipt: Path, calendar_cls=None, calendars: tuple[str, ...] = lock.CALENDARS,
                timeout: int = 30) -> dict:
    """Fetch the Bitcoin proof for a pending .ots receipt and rewrite it in place.

    Only calendars in `calendars` are contacted, never a URL read from the receipt. The file is
    rewritten (atomically) only when a calendar returns new attestations. Returns per-calendar
    status and any Bitcoin attestations.
    """
    from opentimestamps.calendar import CommitmentNotFoundError, RemoteCalendar
    from opentimestamps.core.serialize import StreamDeserializationContext, StreamSerializationContext
    from opentimestamps.core.timestamp import DetachedTimestampFile

    calendar_cls = calendar_cls or RemoteCalendar
    with open(receipt, "rb") as f:
        stamp = DetachedTimestampFile.deserialize(StreamDeserializationContext(f))
    before = set(stamp.timestamp.all_attestations())
    status = {}
    for node, url in list(_pending_nodes(stamp.timestamp)):
        if url not in calendars:
            status[url] = "skipped: not a known calendar"
            continue
        try:
            node.merge(calendar_cls(url).get_timestamp(node.msg, timeout=timeout))
            status[url] = "upgraded"
        except CommitmentNotFoundError:
            status[url] = "still pending"
        except Exception as e:  # a calendar being down just leaves its entry pending
            status[url] = f"error: {type(e).__name__}"
    changed = set(stamp.timestamp.all_attestations()) != before
    if changed:
        tmp = receipt.with_name(receipt.name + ".tmp")
        with open(tmp, "wb") as f:
            stamp.serialize(StreamSerializationContext(f))
        os.replace(tmp, receipt)
    return {"changed": changed, "calendars": status, "bitcoin": bitcoin_attestations(stamp.timestamp)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("verify", "stamp", "upgrade"))
    ap.add_argument("manifest", nargs="?", type=Path, default=FORECASTS / "LOCK_MANIFEST.json")
    args = ap.parse_args(argv)
    if args.command == "verify":
        problems = lock.verify_manifest(args.manifest)
        print("\n".join(problems) or "all locked files match")
        raise SystemExit(1 if problems else 0)
    if args.command == "stamp":
        print("wrote", lock.ots_stamp(args.manifest))
        return
    res = ots_upgrade(args.manifest.with_name(args.manifest.name + ".ots"))
    for url, s in res["calendars"].items():
        print(f"{url}: {s}")
    for b in res["bitcoin"]:
        print(f"Bitcoin block {b['height']}, merkle root {b['merkle_root']}")
    print("receipt updated" if res["changed"] else "receipt unchanged")


if __name__ == "__main__":
    main()
