from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

import lumi.posttraining_private_sft_encoding_preflight_v1 as preflight
from lumi.posttraining_private_sft_encoding_preflight_v1 import (
    EXPECTED_AUTHORIZATION,
    EXPECTED_AUTHORIZATION_BYTES,
    EXPECTED_AUTHORIZATION_SHA256,
    EXPECTED_AUTHORIZATION_TEXT,
    EXPECTED_ENCODING_CONTRACT,
    EXPECTED_INPUT,
    EXPECTED_INPUT_METADATA,
    EXPECTED_LIMITS,
    EXPECTED_NEXT_GATE,
    EXPECTED_OUTPUT_ABSENCE,
    EXPECTED_PROJECTION,
    EXPECTED_REPORT_MARKER,
    EXPECTED_RESOURCE_PASS_ONE,
    EXPECTED_RESOURCE_PASS_TWO,
    EXPECTED_SUCCESSOR,
    EXPECTED_VERIFICATION_MARKER,
    _assert_raw_text_free,
    _project_shards,
    build_report,
    build_verification,
    finish_marker,
    load_plan,
    sha256_value,
    start_marker,
    strict_json_bytes,
    validate_report,
    validate_verification,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-private-nonproduction-encoding-preflight-v0.1.json"


@pytest.fixture(scope="module")
def plan() -> dict:
    value = json.loads(PLAN.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.fixture()
def report() -> dict:
    return build_report(
        plan={
            "_runtime_observed": {
                "input_metadata": copy.deepcopy(EXPECTED_INPUT_METADATA),
                "output_absence": copy.deepcopy(EXPECTED_OUTPUT_ABSENCE),
            }
        },
        plan_file_sha256="a" * 64,
        repository_root=ROOT,
    )


def _resign(value: dict, field: str) -> None:
    value.pop(field, None)
    value[field] = sha256_value(value)


def test_authorization_is_exact_and_metadata_only() -> None:
    assert len(EXPECTED_AUTHORIZATION_TEXT.encode()) == EXPECTED_AUTHORIZATION_BYTES == 22
    assert hashlib.sha256(EXPECTED_AUTHORIZATION_TEXT.encode()).hexdigest() == (
        EXPECTED_AUTHORIZATION_SHA256
    )
    assert EXPECTED_AUTHORIZATION["metadata_only_encoding_preflight_authorized"] is True
    assert EXPECTED_AUTHORIZATION["private_jsonl_row_access_authorized"] is False
    assert EXPECTED_AUTHORIZATION["encoded_shard_materialization_authorized"] is False
    assert EXPECTED_AUTHORIZATION["sft_training_authorized"] is False
    assert EXPECTED_AUTHORIZATION["candidate_mutation_authorized"] is False
    assert EXPECTED_AUTHORIZATION["source_mutation_authorized"] is False
    assert EXPECTED_AUTHORIZATION["private_input_or_predecessor_mutation_authorized"] is False


def test_real_plan_load_never_opens_private_jsonl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protected = {ROOT / split["path"] for split in EXPECTED_INPUT["splits"].values()}
    original_open = Path.open
    original_os_open = os.open

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if path.absolute() in protected:
            raise AssertionError("private JSONL must remain unopened in preflight")
        return original_open(path, *args, **kwargs)

    def guarded_os_open(path: str | bytes | os.PathLike[str] | os.PathLike[bytes], *args: object):
        candidate = Path(os.fsdecode(path)).absolute()
        if candidate in protected:
            raise AssertionError("private JSONL must remain unopened in preflight")
        return original_os_open(path, *args)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(os, "open", guarded_os_open)
    loaded = load_plan(PLAN, repository_root=ROOT)
    assert loaded["input"] == EXPECTED_INPUT


def test_preflight_and_successor_read_budgets_are_exactly_bounded() -> None:
    assert EXPECTED_LIMITS["maximum_factual_passes"] == 2
    assert EXPECTED_LIMITS["maximum_private_jsonl_row_reads"] == 0
    assert EXPECTED_LIMITS["maximum_candidate_row_reads"] == 0
    assert EXPECTED_LIMITS["maximum_source_row_reads"] == 0
    assert EXPECTED_LIMITS["maximum_test_object_reads"] == 0
    assert EXPECTED_LIMITS["maximum_candidate_or_source_mutations"] == 0
    assert EXPECTED_LIMITS["maximum_private_input_or_predecessor_mutations"] == 0
    assert EXPECTED_RESOURCE_PASS_ONE["factual_passes"] == 1
    assert EXPECTED_RESOURCE_PASS_TWO["factual_passes"] == 2
    successor = EXPECTED_SUCCESSOR["limits"]
    assert successor["maximum_input_jsonl_row_reads"] == 329_200
    assert successor["maximum_encoded_index_row_reads"] == 329_200
    assert successor["maximum_total_row_reads"] == 658_400
    assert successor["maximum_candidate_row_reads"] == 0
    assert successor["maximum_input_or_predecessor_mutations"] == 0


def test_exact_shard_and_write_projection_is_proved() -> None:
    assert _project_shards(33_596_489, 4_095, 4_194_304) == (9, 9)
    assert _project_shards(2_295_902, 4_095, 4_194_304) == (1, 1)
    assert EXPECTED_PROJECTION == {
        "binary_token_and_label_bytes": 287_139_128,
        "encoded_index_maximum_bytes": 57_610_000,
        "encoded_index_maximum_bytes_per_row": 175,
        "evidence_and_manifest_reserve_bytes": 4_194_304,
        "maximum_projected_write_bytes": 348_943_432,
        "shards": 10,
        "train_shards": 9,
        "validation_shards": 1,
    }
    assert EXPECTED_PROJECTION["maximum_projected_write_bytes"] < 536_870_912


def test_encoding_contract_fits_binary_types_and_context() -> None:
    assert EXPECTED_ENCODING_CONTRACT["maximum_record_tokens"] == 4_095
    assert EXPECTED_ENCODING_CONTRACT["context_window"] == 4_096
    assert EXPECTED_ENCODING_CONTRACT["tokenizer_vocab_size"] == 65_536
    assert EXPECTED_ENCODING_CONTRACT["tokenizer_unique_id_count"] == 65_536
    assert EXPECTED_ENCODING_CONTRACT["tokenizer_id_minimum"] == 0
    assert EXPECTED_ENCODING_CONTRACT["tokenizer_id_maximum"] == 65_535
    assert EXPECTED_ENCODING_CONTRACT["token_ids_fit_uint32_and_int32"] is True
    assert EXPECTED_ENCODING_CONTRACT["encoded_directory_mode"] == "0700"
    assert EXPECTED_ENCODING_CONTRACT["encoded_file_mode"] == "0600"


def test_private_file_metadata_validator_is_lstat_only(tmp_path: Path) -> None:
    for expected in EXPECTED_INPUT["splits"].values():
        path = tmp_path / expected["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.parent.chmod(0o700)
        path.touch(mode=0o600)
        path.chmod(0o600)
        os.truncate(path, expected["bytes"])
    assert preflight._validate_input_metadata(tmp_path) == EXPECTED_INPUT_METADATA


@pytest.mark.parametrize("fault", ["mode", "hardlink", "symlink", "size"])
def test_private_file_metadata_validator_rejects_alias_or_drift(tmp_path: Path, fault: str) -> None:
    paths = []
    for expected in EXPECTED_INPUT["splits"].values():
        path = tmp_path / expected["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.parent.chmod(0o700)
        path.touch(mode=0o600)
        path.chmod(0o600)
        os.truncate(path, expected["bytes"])
        paths.append(path)
    target = paths[0]
    if fault == "mode":
        target.chmod(0o640)
    elif fault == "hardlink":
        os.link(target, target.with_name("alias.jsonl"))
    elif fault == "symlink":
        real = target.with_name("real.jsonl")
        target.rename(real)
        target.symlink_to(real.name)
    else:
        os.truncate(target, target.stat().st_size - 1)
    with pytest.raises(ValueError, match="metadata changed"):
        preflight._validate_input_metadata(tmp_path)


def test_report_is_raw_text_free_and_requires_fresh_successor(report: dict) -> None:
    validate_report(report)
    assert report["input_metadata"] == EXPECTED_INPUT_METADATA
    assert report["output_absence"] == EXPECTED_OUTPUT_ABSENCE
    assert report["next_gate"] == EXPECTED_NEXT_GATE
    assert report["decision"]["encoded_shards_created"] is False
    assert report["successor"]["historical_registry_v0_17_executor_reusable"] is False
    assert report["successor"]["generic_shard_writer_reusable"] is False
    hardening = report["successor"]["required_hardening"]
    assert hardening["input_hash_verified_on_same_stream_used_for_encoding"] is True
    assert hardening["input_open_nofollow"] is True
    assert hardening["input_same_fd_regular_mode_0600_nlink_one"] is True
    assert hardening["input_owner_matches_process"] is True
    assert hardening["private_parent_chain_anchored_without_symlinks"] is True
    assert hardening["input_record_sequence_and_split_totals_verified"] is True
    assert hardening["exclusive_nofollow_output_creation"] is True
    assert hardening["encoded_directory_mode"] == "0700"
    assert hardening["encoded_file_mode"] == "0600"
    assert report["production_eligible"] is False
    assert b"messages" not in preflight.canonical_json(report)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("input", "records"), 329_199),
        (("input", "maximum_record_tokens"), 4_096),
        (("input_metadata", "train", "row_reads"), 1),
        (("projection", "train_shards"), 8),
        (("projection", "maximum_projected_write_bytes"), 536_870_913),
        (("decision", "encoded_shards_created"), True),
        (("successor", "historical_registry_v0_17_executor_reusable"), True),
        (("successor", "limits", "maximum_input_jsonl_row_reads"), 329_201),
        (("next_gate", "authorization_required"), False),
        (("production_eligible",), True),
    ],
)
def test_report_rejects_resigned_boundary_drift(
    report: dict, path: tuple[str, ...], value: object
) -> None:
    changed = copy.deepcopy(report)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _resign(changed, "report_sha256")
    with pytest.raises(ValueError):
        validate_report(changed)


def test_verification_requires_exact_second_pass(report: dict) -> None:
    verification = build_verification(report=report, report_file_sha256="b" * 64)
    validate_verification(verification)
    changed = copy.deepcopy(verification)
    changed["resource_accounting"]["factual_passes"] = 1
    _resign(changed, "verification_sha256")
    with pytest.raises(ValueError):
        validate_verification(changed)


def test_markers_are_private_same_inode_and_nonretryable(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    identity = start_marker(report_path, EXPECTED_REPORT_MARKER)
    finish_marker(report_path, EXPECTED_REPORT_MARKER, b"{}\n", identity)
    assert stat.S_IMODE(report_path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        start_marker(report_path, EXPECTED_REPORT_MARKER)

    verification_path = tmp_path / "verification.json"
    start_marker(verification_path, EXPECTED_VERIFICATION_MARKER)
    with pytest.raises(FileExistsError):
        start_marker(verification_path, EXPECTED_VERIFICATION_MARKER)


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
def test_raw_training_text_keys_fail_closed(raw_key: str) -> None:
    with pytest.raises(ValueError, match="raw training text"):
        _assert_raw_text_free({"nested": [{raw_key: "forbidden"}]})
