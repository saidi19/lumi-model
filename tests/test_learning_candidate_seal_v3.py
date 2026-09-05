from __future__ import annotations

import copy
import hashlib
import json
import stat
import time
from pathlib import Path

import pytest

import lumi.posttraining_learning_candidate_seal_v3 as seal_v3
from lumi.posttraining_learning_candidate_seal_v3 import (
    EXPECTED_CONTRACT_BOUNDARY,
    EXPECTED_HYGIENE,
    EXPECTED_INPUT,
    EXPECTED_MANIFEST_AUTHORIZATION_BOUNDARY,
    EXPECTED_MANIFEST_BINDINGS,
    EXPECTED_MANIFEST_MARKER,
    EXPECTED_MANIFEST_NEXT_GATE,
    EXPECTED_NEXT_GATE,
    EXPECTED_OBSERVED_AT,
    EXPECTED_PASS_ONE_RESOURCES,
    EXPECTED_PRODUCTION_BLOCKERS,
    EXPECTED_SOURCE_PATHS,
    EXPECTED_VERIFICATION_MARKER,
    MANIFEST_SCHEMA_VERSION,
    MANIFEST_STATUS,
    _assert_raw_text_free,
    build_verification,
    canonical_json,
    finish_marker,
    load_plan,
    materialize_sources,
    open_private_output,
    sha256_value,
    start_marker,
    strict_json_bytes,
    validate_manifest,
    validate_verification,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-learning-candidate-seal-v0.3.json"


@pytest.fixture(scope="module")
def plan() -> dict:
    value = json.loads(PLAN.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def manifest(plan: dict) -> dict:
    value = {
        "authorization_boundary": copy.deepcopy(EXPECTED_MANIFEST_AUTHORIZATION_BOUNDARY),
        "bindings": copy.deepcopy(EXPECTED_MANIFEST_BINDINGS),
        "contract_boundary": copy.deepcopy(EXPECTED_CONTRACT_BOUNDARY),
        "hygiene": copy.deepcopy(EXPECTED_HYGIENE),
        "input": copy.deepcopy(EXPECTED_INPUT),
        "next_gate": copy.deepcopy(EXPECTED_MANIFEST_NEXT_GATE),
        "observed_at": EXPECTED_OBSERVED_AT,
        "ordering_contract": "source_id_lexical_then_existing_file_order_v0.1",
        "output": {
            "candidate_input_bytes": EXPECTED_INPUT["candidate_input_bytes"],
            "candidate_row_reads": EXPECTED_INPUT["records"],
            "input_record_id_sequence_sha256": "1" * 64,
            "source_metrics": {
                source["source_id"]: {
                    "by_split": copy.deepcopy(source["by_split"]),
                    "candidate_bytes": source["bytes"],
                    "candidate_sha256": source["candidate_sha256"],
                    "records": source["records"],
                }
                for source in plan["sources"]
            },
            "splits": {
                "train": {
                    "bytes": 1,
                    "record_id_sequence_sha256": "2" * 64,
                    "records": EXPECTED_INPUT["train_records"],
                    "serialized_tokens": EXPECTED_INPUT["train_serialized_tokens"],
                    "sha256": "3" * 64,
                    "supervised_tokens": EXPECTED_INPUT["train_supervised_tokens"],
                },
                "validation": {
                    "bytes": 1,
                    "record_id_sequence_sha256": "4" * 64,
                    "records": EXPECTED_INPUT["validation_records"],
                    "serialized_tokens": EXPECTED_INPUT["validation_serialized_tokens"],
                    "sha256": "5" * 64,
                    "supervised_tokens": EXPECTED_INPUT["validation_supervised_tokens"],
                },
            },
            "unique_record_ids": EXPECTED_INPUT["records"],
        },
        "plan_file_sha256": "6" * 64,
        "production_blockers_preserved": list(EXPECTED_PRODUCTION_BLOCKERS),
        "production_eligible": False,
        "raw_training_text_retained_in_manifest": False,
        "resource_accounting": copy.deepcopy(EXPECTED_PASS_ONE_RESOURCES),
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": MANIFEST_STATUS,
        "token_accounting_contract": (
            "exact_candidate_sha256_plus_frozen_v0_18_per_source_token_ledger"
        ),
    }
    value["seal_sha256"] = sha256_value(value)
    validate_manifest(value, plan=plan)
    return value


def _resign_manifest(value: dict) -> None:
    value.pop("seal_sha256", None)
    value["seal_sha256"] = sha256_value(value)


def _synthetic_audit(plan: dict, manifest: dict) -> dict:
    return {
        "candidate_files_opened": 0,
        "output_audit_row_reads": EXPECTED_INPUT["records"],
        "source_split_records": {
            source["source_id"]: {
                split: source["by_split"][split]["records"] for split in ("train", "validation")
            }
            for source in plan["sources"]
        },
        "splits": {
            split: {
                field: manifest["output"]["splits"][split][field]
                for field in (
                    "bytes",
                    "record_id_sequence_sha256",
                    "records",
                    "sha256",
                )
            }
            for split in ("train", "validation")
        },
        "unique_record_ids": EXPECTED_INPUT["records"],
    }


def test_plan_binds_exact_twenty_source_paths_without_row_reads(plan: dict) -> None:
    assert [source["source_id"] for source in plan["sources"]] == sorted(EXPECTED_SOURCE_PATHS)
    assert {source["source_id"]: source["path"] for source in plan["sources"]} == (
        EXPECTED_SOURCE_PATHS
    )
    assert sum(source["records"] for source in plan["sources"]) == 329_200
    assert sum(source["bytes"] for source in plan["sources"]) == 529_207_367


def test_real_plan_metadata_chain_loads_without_candidate_rows() -> None:
    loaded = load_plan(PLAN, repository_root=ROOT, check_candidate_files=True)
    assert loaded["input"] == EXPECTED_INPUT


def test_materializer_hashes_and_writes_each_synthetic_row_once(tmp_path: Path) -> None:
    rows = [
        {
            "messages": [
                {"content": "u", "role": "user"},
                {"content": "a", "role": "assistant"},
            ],
            "record_id": "r1",
            "source": {"source_id": "synthetic"},
            "split": "train",
        },
        {
            "messages": [
                {"content": "u2", "role": "user"},
                {"content": "a2", "role": "assistant"},
            ],
            "record_id": "r2",
            "source": {"source_id": "synthetic"},
            "split": "validation",
        },
    ]
    candidate_payload = b"".join(canonical_json(row) + b"\n" for row in rows)
    candidate = tmp_path / "candidate.jsonl"
    candidate.write_bytes(candidate_payload)
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    train = private / "train.jsonl"
    validation = private / "validation.jsonl"
    source = {
        "by_split": {
            "train": {
                "records": 1,
                "serialized_tokens": 10,
                "supervised_tokens": 4,
            },
            "validation": {
                "records": 1,
                "serialized_tokens": 11,
                "supervised_tokens": 5,
            },
        },
        "bytes": len(candidate_payload),
        "candidate_sha256": hashlib.sha256(candidate_payload).hexdigest(),
        "path": "candidate.jsonl",
        "records": 2,
        "source_id": "synthetic",
    }
    with (
        open_private_output(train) as train_handle,
        open_private_output(validation) as validation_handle,
    ):
        observed = materialize_sources(
            sources=[source],
            repository_root=tmp_path,
            train_handle=train_handle,
            validation_handle=validation_handle,
            started=time.monotonic(),
            maximum_candidate_rows=2,
            maximum_data_write_bytes=10_000,
        )
    assert observed["candidate_row_reads"] == 2
    assert observed["unique_record_ids"] == 2
    assert observed["source_metrics"]["synthetic"]["candidate_sha256"] == source["candidate_sha256"]
    assert train.read_bytes() == canonical_json(rows[0]) + b"\n"
    assert validation.read_bytes() == canonical_json(rows[1]) + b"\n"
    assert stat.S_IMODE(train.stat().st_mode) == 0o600
    assert stat.S_IMODE(validation.stat().st_mode) == 0o600


def _synthetic_source(path: Path, source_id: str, rows: list[dict]) -> dict:
    payload = b"".join(canonical_json(row) + b"\n" for row in rows)
    path.write_bytes(payload)
    by_split = {
        split: {
            "records": sum(row["split"] == split for row in rows),
            "serialized_tokens": sum(row["split"] == split for row in rows),
            "supervised_tokens": sum(row["split"] == split for row in rows),
        }
        for split in ("train", "validation")
    }
    return {
        "by_split": by_split,
        "bytes": len(payload),
        "candidate_sha256": hashlib.sha256(payload).hexdigest(),
        "path": path.name,
        "records": len(rows),
        "source_id": source_id,
    }


def test_materializer_never_scans_bytes_after_final_bound_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = {
        "messages": [{"content": "first", "role": "user"}],
        "record_id": "first",
        "source": {"source_id": "synthetic"},
        "split": "train",
    }
    forbidden_tail = {
        "messages": [{"content": "must-not-parse", "role": "user"}],
        "record_id": "tail",
        "source": {"source_id": "synthetic"},
        "split": "train",
    }
    candidate = tmp_path / "candidate.jsonl"
    payload = canonical_json(first) + b"\n" + canonical_json(forbidden_tail) + b"\n"
    candidate.write_bytes(payload)
    source = {
        "by_split": {
            "train": {"records": 1, "serialized_tokens": 1, "supervised_tokens": 1},
            "validation": {
                "records": 0,
                "serialized_tokens": 0,
                "supervised_tokens": 0,
            },
        },
        "bytes": len(payload),
        "candidate_sha256": hashlib.sha256(payload).hexdigest(),
        "path": candidate.name,
        "records": 1,
        "source_id": "synthetic",
    }
    parsed: list[bytes] = []
    original = seal_v3.strict_json_bytes

    def recording_parser(value: bytes) -> object:
        parsed.append(value)
        return original(value)

    monkeypatch.setattr(seal_v3, "strict_json_bytes", recording_parser)
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    with (
        open_private_output(private / "train.jsonl") as train_handle,
        open_private_output(private / "validation.jsonl") as validation_handle,
        pytest.raises(ValueError, match="bytes after final bound row"),
    ):
        materialize_sources(
            sources=[source],
            repository_root=tmp_path,
            train_handle=train_handle,
            validation_handle=validation_handle,
            started=time.monotonic(),
            maximum_candidate_rows=1,
            maximum_data_write_bytes=10_000,
        )
    assert parsed == [canonical_json(first) + b"\n"]


def test_materializer_enforces_global_record_id_uniqueness(tmp_path: Path) -> None:
    sources = []
    for source_id in ("a", "b"):
        row = {
            "messages": [{"content": source_id, "role": "user"}],
            "record_id": "shared-id",
            "source": {"source_id": source_id},
            "split": "train",
        }
        sources.append(_synthetic_source(tmp_path / f"{source_id}.jsonl", source_id, [row]))
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    with (
        open_private_output(private / "train.jsonl") as train_handle,
        open_private_output(private / "validation.jsonl") as validation_handle,
        pytest.raises(ValueError, match="duplicate learning-candidate record ID"),
    ):
        materialize_sources(
            sources=sources,
            repository_root=tmp_path,
            train_handle=train_handle,
            validation_handle=validation_handle,
            started=time.monotonic(),
            maximum_candidate_rows=2,
            maximum_data_write_bytes=10_000,
        )


def test_materializer_checks_write_cap_before_violating_write(tmp_path: Path) -> None:
    row = {
        "messages": [{"content": "private-sentinel", "role": "user"}],
        "record_id": "r1",
        "source": {"source_id": "synthetic"},
        "split": "train",
    }
    source = _synthetic_source(tmp_path / "candidate.jsonl", "synthetic", [row])
    row_bytes = canonical_json(row) + b"\n"
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    train = private / "train.jsonl"
    with (
        open_private_output(train) as train_handle,
        open_private_output(private / "validation.jsonl") as validation_handle,
        pytest.raises(ValueError, match="private-write cap"),
    ):
        materialize_sources(
            sources=[source],
            repository_root=tmp_path,
            train_handle=train_handle,
            validation_handle=validation_handle,
            started=time.monotonic(),
            maximum_candidate_rows=1,
            maximum_data_write_bytes=len(row_bytes) - 1,
        )
    assert train.read_bytes() == b""


def test_materializer_evidence_is_raw_text_free(tmp_path: Path) -> None:
    sentinel = "RAW-PRIVATE-SENTINEL-9b307f"
    row = {
        "messages": [{"content": sentinel, "role": "user"}],
        "record_id": "r1",
        "source": {"source_id": "synthetic"},
        "split": "train",
    }
    source = _synthetic_source(tmp_path / "candidate.jsonl", "synthetic", [row])
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    with (
        open_private_output(private / "train.jsonl") as train_handle,
        open_private_output(private / "validation.jsonl") as validation_handle,
    ):
        observed = materialize_sources(
            sources=[source],
            repository_root=tmp_path,
            train_handle=train_handle,
            validation_handle=validation_handle,
            started=time.monotonic(),
            maximum_candidate_rows=1,
            maximum_data_write_bytes=10_000,
        )
    assert sentinel.encode() not in canonical_json(observed)


def test_durable_markers_make_started_passes_nonretryable(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    identity = start_marker(manifest_path, EXPECTED_MANIFEST_MARKER)
    finish_marker(manifest_path, EXPECTED_MANIFEST_MARKER, b"{}\n", identity)
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        start_marker(manifest_path, EXPECTED_MANIFEST_MARKER)

    verification_path = tmp_path / "verification.json"
    start_marker(verification_path, EXPECTED_VERIFICATION_MARKER)
    with pytest.raises(FileExistsError):
        start_marker(verification_path, EXPECTED_VERIFICATION_MARKER)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("input", "records"), 329_199),
        (("output", "unique_record_ids"), 329_199),
        (("output", "splits", "train", "serialized_tokens"), 33_596_488),
        (("output", "splits", "validation", "supervised_tokens"), 939_292),
        (("authorization_boundary", "production_dataset_seal"), True),
        (("contract_boundary", "production_sft_allowed"), True),
        (("bindings", "source_registry_sha256"), "a" * 64),
        (("next_gate", "candidate_files_reopened"), True),
        (("resource_accounting", "output_audit_row_reads"), 1),
        (("production_eligible",), True),
    ],
)
def test_manifest_rejects_resigned_boundary_drift(
    plan: dict,
    manifest: dict,
    path: tuple[str, ...],
    value: object,
) -> None:
    changed = copy.deepcopy(manifest)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _resign_manifest(changed)
    with pytest.raises(ValueError):
        validate_manifest(changed, plan=plan)


def test_verification_rejects_resigned_audit_drift(plan: dict, manifest: dict) -> None:
    audit = _synthetic_audit(plan, manifest)
    verification = build_verification(
        plan=plan,
        manifest=manifest,
        manifest_file_sha256="7" * 64,
        audit=audit,
    )
    validate_verification(verification, plan=plan, manifest=manifest)
    changed = copy.deepcopy(verification)
    changed["observed_outputs"]["candidate_files_opened"] = 1
    changed.pop("verification_sha256")
    changed["verification_sha256"] = sha256_value(changed)
    with pytest.raises(ValueError):
        validate_verification(changed, plan=plan, manifest=manifest)


def test_strict_json_rejects_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        strict_json_bytes(b'{"a":1,"a":2}\n')


@pytest.mark.parametrize(
    "raw_key",
    [
        "assistant_response",
        "content",
        "messages",
        "prompt",
        "raw_text",
        "record_id",
        "response",
        "text",
        "user_prompt",
    ],
)
def test_raw_text_keys_fail_closed(raw_key: str) -> None:
    with pytest.raises(ValueError, match="raw candidate key"):
        _assert_raw_text_free({"nested": [{raw_key: "forbidden"}]})


def test_next_gate_is_preflight_only_and_non_gpu() -> None:
    assert EXPECTED_NEXT_GATE == {
        "action": "authorize_private_nonproduction_sft_encoding_preflight",
        "authorization_required": True,
        "gpu_required": False,
        "production_sft": False,
    }
