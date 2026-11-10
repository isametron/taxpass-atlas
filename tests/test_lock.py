import json

from src.forecast import lock
from tools import lock_receipt as lr


def _make_locked(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "config").mkdir()
    (tmp_path / "src" / "m.py").write_text("x = 1\n")
    (tmp_path / "config" / "p.yaml").write_text("a: 1\n")
    csv = tmp_path / "forecast.csv"
    csv.write_bytes(b"item,mu,sigma\n1001,0.8,0.2\n")
    manifest = lock.build_manifest([csv], root=tmp_path)
    path = tmp_path / "LOCK_MANIFEST.json"
    lock.write_manifest(manifest, path)
    return csv, path, manifest


def test_untouched_lock_verifies(tmp_path):
    _, path, _ = _make_locked(tmp_path)
    assert lock.verify_manifest(path, root=tmp_path) == []


def test_tampered_forecast_is_rejected(tmp_path):
    csv, path, _ = _make_locked(tmp_path)
    csv.write_bytes(b"item,mu,sigma\n1001,0.9,0.2\n")
    assert lock.verify_manifest(path, root=tmp_path) == ["forecast.csv: hash mismatch"]


def test_tampered_manifest_is_rejected(tmp_path):
    _, path, _ = _make_locked(tmp_path)
    m = json.loads(path.read_text())
    m["files"]["forecast.csv"] = "0" * 64
    path.write_text(json.dumps(m))
    problems = lock.verify_manifest(path, root=tmp_path)
    assert any("manifest itself was modified" in p for p in problems)


def test_code_tree_hash_ignores_line_endings(tmp_path):
    _make_locked(tmp_path)
    before = lock.code_tree_sha256(tmp_path)
    (tmp_path / "src" / "m.py").write_bytes(b"x = 1\r\n")
    assert lock.code_tree_sha256(tmp_path) == before
    (tmp_path / "src" / "m.py").write_bytes(b"x = 2\n")
    assert lock.code_tree_sha256(tmp_path) != before


def test_manifest_is_written_with_lf_only(tmp_path):
    _, path, _ = _make_locked(tmp_path)
    assert b"\r\n" not in path.read_bytes()


class _FakeCalendar:
    calls = 0

    def __init__(self, url):
        self.url = url

    def submit(self, msg, timeout=None):
        from opentimestamps.core.notary import PendingAttestation
        from opentimestamps.core.timestamp import Timestamp
        if "down" in self.url:
            raise OSError("calendar unavailable")
        _FakeCalendar.calls += 1
        ts = Timestamp(msg)
        ts.attestations.add(PendingAttestation(self.url))
        return ts


def test_ots_stamp_writes_receipt_for_the_file_hash(tmp_path):
    from opentimestamps.core.op import OpSHA256
    from opentimestamps.core.serialize import StreamDeserializationContext
    from opentimestamps.core.timestamp import DetachedTimestampFile
    f = tmp_path / "m.json"
    f.write_bytes(b"{}\n")
    out = lock.ots_stamp(f, _FakeCalendar, ("https://a", "https://b", "https://down"))
    with open(out, "rb") as fh:
        stamp = DetachedTimestampFile.deserialize(StreamDeserializationContext(fh))
    assert isinstance(stamp.file_hash_op, OpSHA256)
    assert stamp.file_digest.hex() == lock.sha256_file(f)


class _UpgradeCalendar:
    """alice has a Bitcoin proof, bob is still pending, finney is down."""
    contacted = []

    def __init__(self, url):
        self.url = url
        _UpgradeCalendar.contacted.append(url)

    def get_timestamp(self, commitment, timeout=None):
        from opentimestamps.calendar import CommitmentNotFoundError
        from opentimestamps.core.notary import BitcoinBlockHeaderAttestation
        from opentimestamps.core.op import OpSHA256
        from opentimestamps.core.timestamp import Timestamp
        if "bob" in self.url:
            raise CommitmentNotFoundError("pending")
        if "finney" in self.url:
            raise OSError("calendar unavailable")
        ts = Timestamp(commitment)
        ts.ops.add(OpSHA256()).attestations.add(BitcoinBlockHeaderAttestation(900000))
        return ts


def _stamped(tmp_path, calendars):
    f = tmp_path / "m.json"
    f.write_bytes(b"{}\n")
    return f, lock.ots_stamp(f, _FakeCalendar, calendars)


def test_ots_upgrade_adds_bitcoin_proof(tmp_path):
    import hashlib
    alice, bob, finney = (lock.CALENDARS[0], lock.CALENDARS[1], lock.CALENDARS[2])
    f, receipt = _stamped(tmp_path, (alice, bob, finney))
    before = receipt.read_bytes()
    _UpgradeCalendar.contacted = []
    res = lr.ots_upgrade(receipt, _UpgradeCalendar)
    assert res["changed"] and receipt.read_bytes() != before
    assert res["calendars"] == {alice: "upgraded", bob: "still pending", finney: "error: OSError"}
    [btc] = res["bitcoin"]
    assert btc["height"] == 900000
    # Proof survives a round trip and still commits to the manifest's own hash.
    again = lr.ots_upgrade(receipt, _UpgradeCalendar)
    assert again["bitcoin"] == res["bitcoin"] and not again["changed"]
    assert len(btc["merkle_root"]) == 64 and btc["merkle_root"] != hashlib.sha256(f.read_bytes()).hexdigest()


def test_ots_upgrade_ignores_unknown_calendars(tmp_path):
    _, receipt = _stamped(tmp_path, ("https://a.example", "https://b.example"))
    before = receipt.read_bytes()
    _UpgradeCalendar.contacted = []
    res = lr.ots_upgrade(receipt, _UpgradeCalendar)
    assert _UpgradeCalendar.contacted == []
    assert not res["changed"] and res["bitcoin"] == []
    assert set(res["calendars"].values()) == {"skipped: not a known calendar"}
    assert receipt.read_bytes() == before


def test_ots_stamp_needs_two_calendars(tmp_path):
    import pytest
    f = tmp_path / "m.json"
    f.write_bytes(b"{}\n")
    with pytest.raises(RuntimeError, match="only 1 calendars"):
        lock.ots_stamp(f, _FakeCalendar, ("https://a", "https://down", "https://down2"))
    assert not (tmp_path / "m.json.ots").exists()
