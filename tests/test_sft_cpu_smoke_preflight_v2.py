from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from lumi import posttraining_sft_cpu_smoke_preflight_v2 as preflight


def _plan_shape() -> dict:
    bindings = {
        name: copy.deepcopy(value) for name, value in preflight.EXPECTED_EXTERNAL_BINDINGS.items()
    }
    bindings.update(
        {
            name: {"path": path, "sha256": "1" * 64}
            for name, path in preflight.EXPECTED_CODE_PATHS.items()
        }
    )
    bindings.update(
        {
            name: {"path": path, "sha256": "2" * 64}
            for name, path in preflight.EXPECTED_SUCCESSOR_CODE_PATHS.items()
        }
    )
    return {
        "authorization": copy.deepcopy(preflight.EXPECTED_AUTHORIZATION),
        "authorization_bytes": preflight.EXPECTED_AUTHORIZATION_BYTES,
        "authorization_sha256": preflight.EXPECTED_AUTHORIZATION_SHA256,
        "authorization_text": preflight.EXPECTED_AUTHORIZATION_TEXT,
        "bindings": bindings,
        "input": copy.deepcopy(preflight.EXPECTED_INPUT),
        "limits": copy.deepcopy(preflight.EXPECTED_LIMITS),
        "observed_at": preflight.EXPECTED_OBSERVED_AT,
        "output": copy.deepcopy(preflight.EXPECTED_OUTPUT),
        "schema_version": preflight.PLAN_SCHEMA_VERSION,
        "status": preflight.PLAN_STATUS,
        "successor": copy.deepcopy(preflight.EXPECTED_SUCCESSOR),
        "trust_boundary": copy.deepcopy(preflight.EXPECTED_TRUST),
    }


def _synthetic_private_inventory(root: Path) -> dict:
    encoded = root / preflight.EXPECTED_INPUT["encoded_directory"]
    encoded.mkdir(parents=True)
    chain = [
        root / "artifacts/lumi-sft-token-consistent-shards-v0.2",
        root / "artifacts/lumi-sft-token-consistent-shards-v0.2/attempt-001",
        root / "artifacts/lumi-sft-token-consistent-shards-v0.2/attempt-001/private",
        encoded,
    ]
    for directory in chain:
        directory.chmod(0o700)
    declared = {
        "train-00000.tokens.u32le": b"\x01\x00\x00\x00",
        "train-00000.labels.i32le": b"\x01\x00\x00\x00",
        "train-00000.index.jsonl": b"{}\n",
    }
    files = []
    for name, payload in declared.items():
        path = encoded / name
        path.write_bytes(payload)
        path.chmod(0o600)
        files.append(
            {
                "bytes": len(payload),
                "path": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = {
        "shards": [
            {
                "files": files,
                "index": 0,
                "records": 1,
                "split": "train",
                "supervised_tokens": 1,
                "tokens": 1,
            }
        ]
    }
    manifest_path = encoded / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_path.chmod(0o600)
    return manifest


def test_authorization_binding_is_exact() -> None:
    payload = preflight.EXPECTED_AUTHORIZATION_TEXT.encode("utf-8")
    assert len(payload) == 12
    assert hashlib.sha256(payload).hexdigest() == preflight.EXPECTED_AUTHORIZATION_SHA256


def test_strict_json_rejects_duplicate_keys_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        preflight.strict_json_bytes(b'{"a":1,"a":2}')
    with pytest.raises(ValueError, match="nonfinite"):
        preflight.strict_json_bytes(b'{"a":NaN}')


def test_raw_text_guard_rejects_content_fields() -> None:
    with pytest.raises(ValueError, match="raw training text"):
        preflight.assert_raw_text_free({"nested": [{"content": "forbidden"}]})


def test_plan_shape_rejects_any_boundary_drift() -> None:
    plan = _plan_shape()
    preflight._validate_plan_shape(plan)
    plan["limits"]["maximum_encoded_index_row_reads"] = 1
    with pytest.raises(ValueError, match="plan contract changed"):
        preflight._validate_plan_shape(plan)


def test_pending_successor_hash_is_fail_closed(tmp_path: Path) -> None:
    binding = {
        "path": preflight.EXPECTED_SUCCESSOR_CODE_PATHS["successor_runtime"],
        "sha256": preflight.PENDING_SHA256,
    }
    with pytest.raises(ValueError, match="not frozen"):
        preflight._resolve_binding(tmp_path, "successor_runtime", binding)


def test_internal_hash_validation_is_exact() -> None:
    value = {"answer": 42}
    value["verification_sha256"] = preflight.sha256_value(value)
    assert preflight._verify_internal(value, "verification_sha256") == value["verification_sha256"]
    value["answer"] = 43
    with pytest.raises(ValueError, match="internal hash changed"):
        preflight._verify_internal(value, "verification_sha256")


def test_inventory_is_lstat_only_and_digest_is_deterministic(tmp_path: Path, monkeypatch) -> None:
    manifest = _synthetic_private_inventory(tmp_path)
    original_open = os.open

    def guarded_open(path, *args, **kwargs):
        name = os.fspath(path)
        if name.endswith((".index.jsonl", ".tokens.u32le", ".labels.i32le")):
            raise AssertionError("encoded content must not be opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", guarded_open)
    first, first_digest = preflight.inventory_metadata(tmp_path, manifest)
    second, second_digest = preflight.inventory_metadata(tmp_path, manifest)
    assert len(first) == 4
    assert first == second
    assert first_digest == second_digest


def test_inventory_rejects_nonprivate_file_mode(tmp_path: Path) -> None:
    manifest = _synthetic_private_inventory(tmp_path)
    path = tmp_path / preflight.EXPECTED_INPUT["encoded_directory"] / "train-00000.index.jsonl"
    path.chmod(0o644)
    with pytest.raises(ValueError, match="file metadata changed"):
        preflight.inventory_metadata(tmp_path, manifest)


def test_report_validator_requires_zero_encoded_content() -> None:
    report = {
        "bindings": {"plan_file_sha256": "a" * 64},
        "decision": {
            "first_four_record_lengths_known": False,
            "legacy_runtime_reusable": False,
        },
        "production_eligible": False,
        "raw_training_text_retained": False,
        "resource_accounting": {
            "encoded_binary_bytes_read": 0,
            "encoded_index_row_reads": 0,
            "training_actions": 0,
        },
        "schema_version": preflight.REPORT_SCHEMA_VERSION,
        "status": preflight.REPORT_STATUS,
        "trust_boundary": copy.deepcopy(preflight.EXPECTED_TRUST),
    }
    preflight.validate_report(report, plan_file_sha256="a" * 64)
    report["resource_accounting"]["encoded_binary_bytes_read"] = 1
    with pytest.raises(ValueError, match="report changed"):
        preflight.validate_report(report, plan_file_sha256="a" * 64)
