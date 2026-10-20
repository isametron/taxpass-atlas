import json

from src.forecast import lock


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
