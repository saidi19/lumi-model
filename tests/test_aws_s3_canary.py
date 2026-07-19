from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from lumi.aws_s3_canary import (
    _bucket_names,
    _validate_run_id,
    _verify_download,
    _write_payload,
)


def test_bucket_names_are_bounded_and_do_not_expose_account_id() -> None:
    data, logs = _bucket_names("123456789012", "20260719-120000-deadbeef")
    assert data.startswith("lumi-corpus-canary-")
    assert logs.startswith("lumi-corpus-audit-")
    assert "123456789012" not in data
    assert "123456789012" not in logs
    assert len(data) <= 63
    assert len(logs) <= 63


def test_run_id_validation_rejects_paths_and_shell_syntax() -> None:
    assert _validate_run_id("20260719-120000-deadbeef") == "20260719-120000-deadbeef"
    for unsafe in ("../escape", "bad;command", "UPPERCASE", "short"):
        with pytest.raises(ValueError, match="run ID"):
            _validate_run_id(unsafe)


def test_deterministic_payload_and_download_verification(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    digest = _write_payload(first, 1024 * 1024 + 7, "fixture")
    assert _write_payload(second, 1024 * 1024 + 7, "fixture") == digest
    assert first.read_bytes() == second.read_bytes()
    _verify_download(second, digest, first.stat().st_size)
    second.write_bytes(second.read_bytes() + b"x")
    with pytest.raises(RuntimeError, match="failed size or SHA-256"):
        _verify_download(second, digest, first.stat().st_size)


def test_payload_digest_is_sha256(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    observed = _write_payload(path, 17, "tiny")
    assert observed == hashlib.sha256(path.read_bytes()).hexdigest()
