#!/usr/bin/env python3
"""Run one raw-text-free stage-localization replay for failed attempt 005."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

import lumi.posttraining_bedstc2_banglish_conversation_curation_v5 as v5
import scripts.curate_sft_bedstc2_banglish_conversation_v5 as runner_v5
import scripts.verify_sft_bedstc2_banglish_conversation_v5 as verifier_v5
from lumi.posttraining_candidate_mix_preflight import file_sha256

AUTHORIZATION_TEXT = (
    "I approve one bounded local-only Be-DSTC2 attempt-005 raw-text-free "
    "stage-localization diagnostic. Bind v0.6 plan SHA-256 "
    "43c6f895870cb0a364fbb759d5e7dc6bf9dff34064dd0742751db1e7e378256c, "
    "implementation SHA-256 "
    "75a140d0de68b0a956c090f4677acefd929d9135130d279fb79909f119e08cdf, "
    "runner SHA-256 "
    "80159335cd14e29e668f46bd57a050008b6582aa5355e3aa9206903854df2555, "
    "verifier SHA-256 "
    "ae6424a92e5e59f1f59f3135be059e2881824b8d188145c2660bf20d260c59b2, "
    "test SHA-256 "
    "52442f05001dfbfc83d90a0755fc20df3b0d4d445ca82e5fd7d732d91a09c5e9, "
    "and exact 47-byte attempt-005 marker SHA-256 "
    "237ab60e7c92aaeb713c8c447d1f0c3367bbc038fb90e46f543897ea43d72b89. "
    "Preserve all prior evidence byte-for-byte. Authorize exactly one fresh "
    "diagnostic pass retaining only allowlisted stage codes and aggregate counts, "
    "rejection counts, and supervised-token totals—no source, candidate, ID, or "
    "exception text. Cap it at 113,718 source-line reads, 314,716 existing-candidate "
    "reads, 20,000 new-candidate reads, 334,716 total candidate-audit reads, 30 "
    "runtime-minutes, 1,048,576 private-write bytes, and zero network/GPU/inference. "
    "Create only a versioned diagnostic plan, implementation, synthetic tests, "
    "raw-text-free report, and structural verification. No candidate or owner "
    "artifacts, policy/floor change, retry, source or registry mutation, promotion, "
    "dataset seal, training, remote access, or unrelated action."
)
AUTHORIZATION_TEXT_SHA256 = (
    "c9083b5502f4cbdcee4276fb33543edca704f3b73619be73b20456ac94bcc064"
)

PLAN_SCHEMA = "lumi.sft-bedstc2-attempt-005-stage-localization-plan.v0.9"
REPORT_SCHEMA = "lumi.sft-bedstc2-attempt-005-stage-localization-report.v0.9"
VERIFICATION_SCHEMA = (
    "lumi.sft-bedstc2-attempt-005-stage-localization-verification.v0.9"
)
PLAN_PATH = "configs/lumi-4b-sft-bedstc2-attempt-005-stage-localization-diagnostic-v0.9.json"
IMPLEMENTATION_PATH = "scripts/diagnose_sft_bedstc2_attempt_005_stage_localization_v1.py"
TEST_PATH = "tests/test_bedstc2_attempt_005_stage_localization_v1.py"
REPORT_PATH = "evidence/lumi-4b-sft-bedstc2-attempt-005-stage-localization-diagnostic-v0.9.json"
VERIFICATION_PATH = (
    "evidence/lumi-4b-sft-bedstc2-attempt-005-stage-localization-diagnostic-"
    "verification-v0.9.json"
)
REPORT_MARKER = b'{"started":"bedstc2_attempt_005_stage_localization_v0.9"}\n'

PREDECESSOR_BINDINGS = {
    "predecessor_implementation": (
        "src/lumi/posttraining_bedstc2_banglish_conversation_curation_v5.py",
        "75a140d0de68b0a956c090f4677acefd929d9135130d279fb79909f119e08cdf",
    ),
    "predecessor_plan": (
        "configs/lumi-4b-sft-bedstc2-banglish-conversation-curation-recovery-v0.6.json",
        "43c6f895870cb0a364fbb759d5e7dc6bf9dff34064dd0742751db1e7e378256c",
    ),
    "predecessor_runner": (
        "scripts/curate_sft_bedstc2_banglish_conversation_v5.py",
        "80159335cd14e29e668f46bd57a050008b6582aa5355e3aa9206903854df2555",
    ),
    "predecessor_runner_marker": (
        "evidence/lumi-4b-sft-bedstc2-banglish-conversation-curation-v0.6.json",
        "237ab60e7c92aaeb713c8c447d1f0c3367bbc038fb90e46f543897ea43d72b89",
    ),
    "predecessor_test": (
        "tests/test_bedstc2_banglish_conversation_curation_v5.py",
        "52442f05001dfbfc83d90a0755fc20df3b0d4d445ca82e5fd7d732d91a09c5e9",
    ),
    "predecessor_verifier": (
        "scripts/verify_sft_bedstc2_banglish_conversation_v5.py",
        "ae6424a92e5e59f1f59f3135be059e2881824b8d188145c2660bf20d260c59b2",
    ),
}
EXPECTED_BINDING_PATHS = {
    **{name: value[0] for name, value in PREDECESSOR_BINDINGS.items()},
    "diagnostic_implementation": IMPLEMENTATION_PATH,
    "diagnostic_test": TEST_PATH,
}
EXPECTED_LIMITS = {
    "maximum_existing_candidate_row_reads": 314_716,
    "maximum_gpu_use": 0,
    "maximum_inference_calls": 0,
    "maximum_new_candidate_row_reads": 20_000,
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 1_048_576,
    "maximum_runtime_minutes": 30,
    "maximum_source_line_reads": 113_718,
    "maximum_total_candidate_audit_row_reads": 334_716,
    "passes": 1,
}
EXPECTED_HISTORICAL_FAILURE = {
    "candidate_or_owner_outputs_materialized": 0,
    "existing_candidate_row_reads_conservative_upper_bound": 314_716,
    "new_candidate_row_reads_conservative_upper_bound": 20_000,
    "observed_wall_seconds": 87.55,
    "private_write_bytes": 47,
    "runner_passes_completed": 0,
    "runner_passes_started": 1,
    "source_line_reads_conservative_upper_bound": 113_718,
    "total_candidate_audit_row_reads_conservative_upper_bound": 334_716,
    "verifier_invoked": False,
}
EXPECTED_OUTPUT_PATHS = {"report": REPORT_PATH, "verification": VERIFICATION_PATH}
EXPECTED_PROHIBITED_ACTIONS = {
    "candidate_or_owner_artifact_write": False,
    "dataset_seal": False,
    "gpu_use": False,
    "inference": False,
    "network": False,
    "policy_or_floor_change": False,
    "promotion": False,
    "registry_mutation": False,
    "remote_or_instance_action": False,
    "retry": False,
    "source_mutation": False,
    "training": False,
}

STAGE_ORDER = (
    "raw_object_verification",
    "source_parse_train",
    "source_parse_dev",
    "source_inventory_or_malformed_inventory",
    "vocab_load",
    "candidate_construction",
    "runtime_initialization",
    "new_generic_filter",
    "baseline_evidence_or_inventory",
    "baseline_first_selection",
    "owner_artifacts_in_memory",
    "serialization_in_memory",
    "combined_audit",
    "report_validation",
    "build_returned",
    "post_build_artifact_validation",
    "post_build_telemetry_validation",
    "post_build_rejection_validation",
    "post_build_owner_validation",
    "prospective_verification",
    "prospective_write_cap",
    "predecessor_marker_integrity",
    "post_build_checks_complete",
)
GATE_TO_STAGE = {
    "raw_object_binding_or_content": "raw_object_verification",
    "source_parse_train": "source_parse_train",
    "source_parse_dev": "source_parse_dev",
    "source_inventory": "source_inventory_or_malformed_inventory",
    "malformed_source_inventory": "source_inventory_or_malformed_inventory",
    "vocab_inventory": "vocab_load",
    "vocab_item_contract": "vocab_load",
    "vocab_empty_placeholder_count": "vocab_load",
    "vocab_lexical_or_anchor_contract": "vocab_load",
    "candidate_dialogue_identity": "candidate_construction",
    "candidate_inventory_cap": "candidate_construction",
    "runtime_registry_boundary": "runtime_initialization",
    "new_generic_filter": "new_generic_filter",
    "baseline_internal_evidence": "baseline_evidence_or_inventory",
    "baseline_registry_source": "baseline_evidence_or_inventory",
    "baseline_rows_or_token_ledger": "baseline_evidence_or_inventory",
    "baseline_excluded_source": "baseline_evidence_or_inventory",
    "selection_input_cap": "baseline_first_selection",
    "selection_baseline_exact_duplicate": "baseline_first_selection",
    "selection_baseline_minhash": "baseline_first_selection",
    "selection_train_token_floor": "baseline_first_selection",
    "selection_retained_cap": "baseline_first_selection",
    "owner_packet_contract": "owner_artifacts_in_memory",
    "serialization_in_memory": "serialization_in_memory",
    "combined_generic_gate": "combined_audit",
    "combined_exact_gate": "combined_audit",
    "combined_minhash_gate": "combined_audit",
    "combined_token_ledger": "combined_audit",
    "combined_family_cap": "combined_audit",
    "combined_task_or_language_floor": "combined_audit",
    "report_validation": "report_validation",
    "post_build_artifact_binding": "post_build_artifact_validation",
    "post_build_telemetry": "post_build_telemetry_validation",
    "post_build_rejection": "post_build_rejection_validation",
    "post_build_owner": "post_build_owner_validation",
    "prospective_verification": "prospective_verification",
    "prospective_write_cap": "prospective_write_cap",
    "predecessor_marker_integrity": "predecessor_marker_integrity",
    "unexpected_v5_output_appearance": "predecessor_marker_integrity",
    "prior_failure_not_reproduced": "post_build_checks_complete",
    "unclassified_value_error": "raw_object_verification",
    "unclassified_non_value_error": "raw_object_verification",
}
ALLOWED_GATE_CODES = tuple(GATE_TO_STAGE)

AGGREGATE_KEYS = {
    "baseline_rows_loaded",
    "baseline_sources_loaded",
    "candidate_payload_bytes",
    "candidate_rows_proposed",
    "candidate_rows_rejected_before_selection",
    "dev_dialogues",
    "dev_source_lines",
    "dialogues_total",
    "generic_accepted_rows",
    "generic_rejected_rows",
    "malformed_source_rows",
    "new_rows_processed_by_selection",
    "owner_html_bytes",
    "owner_packet_bytes",
    "rejection_payload_bytes",
    "selected_rows",
    "selected_train_rows",
    "selected_validation_rows",
    "source_lines_total",
    "telemetry_payload_bytes",
    "train_dialogues",
    "train_source_lines",
    "train_supervised_tokens",
    "validation_supervised_tokens",
    "vocab_english_unique_tokens",
    "vocab_native_unique_tokens",
    "vocab_others_unique_tokens",
}
AGGREGATE_EARLIEST_STAGE = {
    "baseline_rows_loaded": "baseline_evidence_or_inventory",
    "baseline_sources_loaded": "baseline_evidence_or_inventory",
    "candidate_payload_bytes": "serialization_in_memory",
    "candidate_rows_proposed": "candidate_construction",
    "candidate_rows_rejected_before_selection": "candidate_construction",
    "dev_dialogues": "source_parse_dev",
    "dev_source_lines": "source_parse_dev",
    "dialogues_total": "source_parse_dev",
    "generic_accepted_rows": "new_generic_filter",
    "generic_rejected_rows": "new_generic_filter",
    "malformed_source_rows": "source_inventory_or_malformed_inventory",
    "new_rows_processed_by_selection": "baseline_first_selection",
    "owner_html_bytes": "owner_artifacts_in_memory",
    "owner_packet_bytes": "owner_artifacts_in_memory",
    "rejection_payload_bytes": "serialization_in_memory",
    "selected_rows": "baseline_first_selection",
    "selected_train_rows": "baseline_first_selection",
    "selected_validation_rows": "baseline_first_selection",
    "source_lines_total": "source_parse_dev",
    "telemetry_payload_bytes": "serialization_in_memory",
    "train_dialogues": "source_parse_train",
    "train_source_lines": "source_parse_train",
    "train_supervised_tokens": "baseline_first_selection",
    "validation_supervised_tokens": "baseline_first_selection",
    "vocab_english_unique_tokens": "vocab_load",
    "vocab_native_unique_tokens": "vocab_load",
    "vocab_others_unique_tokens": "vocab_load",
}
EXPECTED_VOCAB_TOKEN_CARDINALITIES = {
    "english": 319,
    "native": 450,
    "others": 281,
}
REJECTION_CODES = (
    "api_call_only_assistant_response",
    "assistant_not_dual_anchor",
    "current_user_not_dual_anchor",
    "exact_duplicate_of_existing_or_selected",
    "generated_metadata",
    "kb_only_record",
    "minhash_0_9_duplicate_of_existing_or_selected",
    "safety_pattern",
    "silence_or_unknown",
    "source_native_punctuation_integrity",
    "source_non_kb_numbered_row",
    "source_outer_trailing_whitespace",
    "source_pair_user_trailing_whitespace",
    "split_supervised_token_cap",
    "tokenizer_or_4096_fit",
    "ungrounded_entity_or_factual_response",
)

_EXACT_LINE_GATES = {
    ("_verify_raw_objects", 1631): "raw_object_binding_or_content",
    ("_verify_raw_objects", 1641): "raw_object_binding_or_content",
    ("parse_dialogues", 836): "source_parse_train",
    ("parse_dialogues", 842): "source_parse_train",
    ("parse_dialogues", 846): "source_parse_train",
    ("parse_dialogues", 858): "source_parse_train",
    ("parse_dialogues", 866): "source_parse_train",
    ("parse_dialogues", 892): "source_parse_train",
    ("parse_dialogues", 905): "source_parse_train",
    ("_normalized_vocab_tokens", 935): "vocab_inventory",
    ("_normalized_vocab_tokens", 940): "vocab_item_contract",
    ("_normalized_vocab_tokens", 946): "vocab_empty_placeholder_count",
    ("_normalized_vocab_tokens", 948): "vocab_lexical_or_anchor_contract",
    ("load_vocab", 956): "vocab_inventory",
    ("load_vocab", 969): "vocab_lexical_or_anchor_contract",
    ("candidate_rows", 1024): "candidate_dialogue_identity",
    ("_runtime", 1168): "runtime_registry_boundary",
    ("_load_baseline", 1191): "baseline_internal_evidence",
    ("_load_baseline", 1213): "baseline_registry_source",
    ("_load_baseline", 1237): "baseline_rows_or_token_ledger",
    ("_load_baseline", 1240): "baseline_excluded_source",
    ("_load_baseline", 1242): "baseline_excluded_source",
    ("_select_baseline_first", 1264): "selection_input_cap",
    ("_select_baseline_first", 1277): "selection_baseline_exact_duplicate",
    ("_select_baseline_first", 1280): "selection_baseline_minhash",
    ("_select_baseline_first", 1345): "selection_train_token_floor",
    ("_select_baseline_first", 1347): "selection_retained_cap",
    ("render_owner_packet", 1376): "owner_packet_contract",
    ("render_owner_packet", 1388): "owner_packet_contract",
    ("render_owner_packet", 1391): "owner_packet_contract",
    ("render_owner_packet", 1396): "owner_packet_contract",
    ("render_owner_packet", 1413): "owner_packet_contract",
    ("render_owner_packet", 1420): "owner_packet_contract",
    ("_audit_combined", 1496): "combined_generic_gate",
    ("_audit_combined", 1511): "combined_exact_gate",
    ("_audit_combined", 1513): "combined_minhash_gate",
    ("_audit_combined", 1546): "combined_token_ledger",
    ("_audit_combined", 1575): "combined_family_cap",
    ("_audit_combined", 1583): "combined_task_or_language_floor",
    ("build_artifacts", 2055): "source_inventory",
    ("build_artifacts", 2073): "malformed_source_inventory",
    ("build_artifacts", 2084): "candidate_inventory_cap",
}

_FUNCTION_STAGES = {
    "_audit_combined": "combined_audit",
    "_load_baseline": "baseline_evidence_or_inventory",
    "_normalized_vocab_tokens": "vocab_load",
    "_runtime": "runtime_initialization",
    "_select_baseline_first": "baseline_first_selection",
    "_verify_raw_objects": "raw_object_verification",
    "build_owner_artifacts": "owner_artifacts_in_memory",
    "candidate_rows": "candidate_construction",
    "load_vocab": "vocab_load",
    "parse_dialogues": "source_parse_train",
    "render_owner_packet": "owner_artifacts_in_memory",
    "validate_report": "report_validation",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _internal_sha(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return _sha256_bytes(_canonical_bytes(unsigned))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _finish_marker(path: Path, payload: bytes) -> None:
    if path.read_bytes() != REPORT_MARKER:
        raise ValueError("diagnostic marker changed")
    with path.open("r+b") as handle:
        handle.seek(0)
        handle.truncate()
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _runtime_alarm(_signum: int, _frame: object) -> None:
    raise TimeoutError("diagnostic runtime cap exceeded")


def _arm_runtime_alarm() -> None:
    signal.signal(signal.SIGALRM, _runtime_alarm)
    signal.setitimer(signal.ITIMER_REAL, EXPECTED_LIMITS["maximum_runtime_minutes"] * 60)


def _cancel_runtime_alarm() -> None:
    signal.setitimer(signal.ITIMER_REAL, 0)


def _validate_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("diagnostic JSON root changed")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("diagnostic path escaped repository root")
    return path


def _validate_predecessor_state(root: Path) -> dict[str, Any]:
    predecessor_plan_path = root / PREDECESSOR_BINDINGS["predecessor_plan"][0]
    predecessor_plan = v5.load_plan(predecessor_plan_path, repository_root=root)
    paths = {
        name: root / relative for name, relative in v5.EXPECTED_OUTPUT_PATHS.items()
    }
    marker = paths["report"]
    if (
        marker.stat().st_size != len(runner_v5.RUNNER_MARKER)
        or marker.read_bytes() != runner_v5.RUNNER_MARKER
        or file_sha256(marker)
        != PREDECESSOR_BINDINGS["predecessor_runner_marker"][1]
    ):
        raise ValueError("failed attempt-005 marker changed")
    if any(path.exists() for name, path in paths.items() if name != "report"):
        raise ValueError("failed attempt-005 absent-output inventory changed")
    if runner_v5._attempt_private_inventory(root):
        raise ValueError("failed attempt-005 private inventory changed")
    return predecessor_plan


def load_diagnostic_plan(path: Path, *, repository_root: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    expected_path = (root / PLAN_PATH).resolve()
    if path.resolve() != expected_path:
        raise ValueError("diagnostic plan path changed")
    plan = _load_json(expected_path)
    if set(plan) != {
        "allowed_gate_codes",
        "authorization_text",
        "authorization_text_sha256",
        "bindings",
        "historical_failure",
        "limits",
        "observed_at",
        "output_paths",
        "prohibited_actions",
        "retention",
        "schema_version",
        "stage_order",
        "status",
    }:
        raise ValueError("diagnostic plan inventory changed")
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("status") != "authorized_one_pass_raw_text_free_stage_localization"
        or plan.get("observed_at") != "2026-08-16"
        or plan.get("authorization_text") != AUTHORIZATION_TEXT
        or plan.get("authorization_text_sha256") != AUTHORIZATION_TEXT_SHA256
        or _sha256_bytes(AUTHORIZATION_TEXT.encode()) != AUTHORIZATION_TEXT_SHA256
        or plan.get("limits") != EXPECTED_LIMITS
        or plan.get("historical_failure") != EXPECTED_HISTORICAL_FAILURE
        or plan.get("output_paths") != EXPECTED_OUTPUT_PATHS
        or plan.get("prohibited_actions") != EXPECTED_PROHIBITED_ACTIONS
        or plan.get("stage_order") != list(STAGE_ORDER)
        or plan.get("allowed_gate_codes") != list(ALLOWED_GATE_CODES)
        or plan.get("retention")
        != {
            "aggregate_counts_only": True,
            "candidate_or_owner_payloads": False,
            "exception_text": False,
            "record_or_row_ids": False,
            "source_or_candidate_text": False,
        }
    ):
        raise ValueError("diagnostic plan boundary changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(EXPECTED_BINDING_PATHS):
        raise ValueError("diagnostic binding inventory changed")
    for name, relative in EXPECTED_BINDING_PATHS.items():
        binding = bindings[name]
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"bytes", "path", "sha256"}
            or binding.get("path") != relative
            or not isinstance(binding.get("bytes"), int)
            or binding["bytes"] <= 0
            or not _validate_sha(binding.get("sha256"))
        ):
            raise ValueError("diagnostic binding changed")
        live = _resolve(root, relative)
        if live.stat().st_size != binding["bytes"] or file_sha256(live) != binding["sha256"]:
            raise ValueError("diagnostic live binding changed")
        predecessor = PREDECESSOR_BINDINGS.get(name)
        if predecessor is not None and binding["sha256"] != predecessor[1]:
            raise ValueError("diagnostic predecessor hash changed")
    _validate_predecessor_state(root)
    return plan


def _preflight_outputs(root: Path) -> dict[str, Path]:
    paths = {name: root / relative for name, relative in EXPECTED_OUTPUT_PATHS.items()}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("diagnostic output already exists")
    return paths


def _reason_counts(rows: Any) -> tuple[dict[str, int], int]:
    counts = Counter({reason: 0 for reason in REJECTION_CODES})
    unknown = 0
    if not isinstance(rows, list):
        return dict(counts), unknown
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        reasons = row.get("reason_codes")
        if not isinstance(reasons, list):
            continue
        for reason in reasons:
            if reason in counts:
                counts[reason] += 1
            else:
                unknown += 1
    return dict(counts), unknown


def _put_count(aggregates: dict[str, int | None], key: str, value: Any) -> None:
    if key in AGGREGATE_KEYS and isinstance(value, int) and value >= 0:
        aggregates[key] = value


def _extract_safe_aggregates(
    tb: TracebackType | None,
) -> tuple[dict[str, int | None], dict[str, int], int]:
    aggregates: dict[str, int | None] = {key: None for key in sorted(AGGREGATE_KEYS)}
    reason_counts = {reason: 0 for reason in REJECTION_CODES}
    unknown_reasons = 0
    current = tb
    while current is not None:
        frame = current.tb_frame
        if Path(frame.f_code.co_filename).resolve() != Path(v5.__file__).resolve():
            current = current.tb_next
            continue
        name = frame.f_code.co_name
        line = current.tb_lineno
        local = frame.f_locals
        if name == "parse_dialogues":
            object_name = local.get("object_name")
            lines = local.get("lines")
            dialogues = local.get("dialogues")
            if object_name == "dialog-dstc2-trn.txt":
                _put_count(aggregates, "train_source_lines", lines)
                if isinstance(dialogues, list):
                    _put_count(aggregates, "train_dialogues", len(dialogues))
            elif object_name == "dialog-dstc2-dev.txt":
                _put_count(aggregates, "dev_source_lines", lines)
                if isinstance(dialogues, list):
                    _put_count(aggregates, "dev_dialogues", len(dialogues))
        elif name == "candidate_rows":
            records = local.get("records")
            rejected = local.get("rejected")
            if isinstance(records, list):
                _put_count(aggregates, "candidate_rows_proposed", len(records))
            if isinstance(rejected, list):
                _put_count(aggregates, "candidate_rows_rejected_before_selection", len(rejected))
                reason_counts, unknown_reasons = _reason_counts(rejected)
        elif name == "_load_baseline":
            rows = local.get("rows")
            source_metrics = local.get("source_metrics")
            if isinstance(rows, list):
                _put_count(aggregates, "baseline_rows_loaded", len(rows))
            if isinstance(source_metrics, dict):
                _put_count(aggregates, "baseline_sources_loaded", len(source_metrics))
        elif name == "_select_baseline_first":
            baseline = local.get("baseline")
            proposed = local.get("proposed")
            selected = local.get("selected")
            totals = local.get("totals")
            rejected = local.get("rejected")
            index = local.get("index")
            if isinstance(baseline, Sequence):
                _put_count(aggregates, "baseline_rows_loaded", len(baseline))
            if isinstance(proposed, Sequence):
                _put_count(aggregates, "generic_accepted_rows", len(proposed))
            if isinstance(selected, list):
                _put_count(aggregates, "selected_rows", len(selected))
                _put_count(
                    aggregates,
                    "selected_train_rows",
                    sum(
                        row.get("split") == "train"
                        for row in selected
                        if isinstance(row, Mapping)
                    ),
                )
                _put_count(
                    aggregates,
                    "selected_validation_rows",
                    sum(
                        row.get("split") == "validation"
                        for row in selected
                        if isinstance(row, Mapping)
                    ),
                )
            if isinstance(totals, Mapping):
                _put_count(aggregates, "train_supervised_tokens", totals.get("train", 0))
                _put_count(
                    aggregates,
                    "validation_supervised_tokens",
                    totals.get("validation", 0),
                )
            if isinstance(proposed, Sequence):
                if line <= 1283:
                    # The first loop indexes the already-loaded baseline; its
                    # index must never be charged as a new-candidate read.
                    processed = 0
                elif line == 1287:
                    processed = 0
                elif 1288 <= line <= 1343 and isinstance(index, int):
                    processed = min(len(proposed), index + 1)
                elif line >= 1344:
                    processed = len(proposed)
                else:
                    processed = None
                _put_count(aggregates, "new_rows_processed_by_selection", processed)
            if isinstance(rejected, list):
                reason_counts, unknown_reasons = _reason_counts(rejected)
        elif name == "build_artifacts":
            train = local.get("train")
            dev = local.get("dev")
            train_lines = local.get("train_lines")
            dev_lines = local.get("dev_lines")
            malformed = local.get("malformed_source_rows")
            proposed = local.get("proposed")
            generic_new = local.get("generic_new")
            generic = local.get("generic")
            baseline = local.get("baseline")
            selected = local.get("selected")
            totals = local.get("totals")
            rejected = local.get("rejected")
            for key, value in (
                ("train_source_lines", train_lines),
                ("dev_source_lines", dev_lines),
                ("malformed_source_rows", malformed),
            ):
                _put_count(aggregates, key, value)
            if isinstance(train, list):
                _put_count(aggregates, "train_dialogues", len(train))
            if isinstance(dev, list):
                _put_count(aggregates, "dev_dialogues", len(dev))
            if isinstance(proposed, list):
                _put_count(aggregates, "candidate_rows_proposed", len(proposed))
            if isinstance(generic_new, list):
                _put_count(aggregates, "generic_accepted_rows", len(generic_new))
            if isinstance(generic, Mapping):
                _put_count(aggregates, "generic_rejected_rows", generic.get("rejected"))
            if hasattr(baseline, "rows"):
                _put_count(aggregates, "baseline_rows_loaded", len(baseline.rows))
            if hasattr(baseline, "source_metrics"):
                _put_count(aggregates, "baseline_sources_loaded", len(baseline.source_metrics))
            if isinstance(selected, list):
                _put_count(aggregates, "selected_rows", len(selected))
                _put_count(
                    aggregates,
                    "selected_train_rows",
                    sum(
                        row.get("split") == "train"
                        for row in selected
                        if isinstance(row, Mapping)
                    ),
                )
                _put_count(
                    aggregates,
                    "selected_validation_rows",
                    sum(
                        row.get("split") == "validation"
                        for row in selected
                        if isinstance(row, Mapping)
                    ),
                )
                if line >= 2115 and isinstance(generic_new, Sequence):
                    _put_count(
                        aggregates,
                        "new_rows_processed_by_selection",
                        len(generic_new),
                    )
            if isinstance(totals, Mapping):
                _put_count(aggregates, "train_supervised_tokens", totals.get("train", 0))
                _put_count(
                    aggregates,
                    "validation_supervised_tokens",
                    totals.get("validation", 0),
                )
            if isinstance(rejected, list):
                _put_count(aggregates, "candidate_rows_rejected_before_selection", len(rejected))
                reason_counts, unknown_reasons = _reason_counts(rejected)
            english = local.get("english")
            native = local.get("native")
            others = local.get("others")
            if isinstance(english, frozenset):
                _put_count(aggregates, "vocab_english_unique_tokens", len(english))
            if isinstance(native, frozenset):
                _put_count(aggregates, "vocab_native_unique_tokens", len(native))
            if isinstance(others, frozenset):
                _put_count(aggregates, "vocab_others_unique_tokens", len(others))
            for key, payload in (
                ("candidate_payload_bytes", local.get("candidate_bytes")),
                ("owner_html_bytes", local.get("html_bytes")),
                ("owner_packet_bytes", local.get("packet_bytes")),
                ("rejection_payload_bytes", local.get("rejection_bytes")),
                ("telemetry_payload_bytes", local.get("telemetry_bytes")),
            ):
                if isinstance(payload, bytes):
                    _put_count(aggregates, key, len(payload))
        current = current.tb_next
    train_lines = aggregates["train_source_lines"]
    dev_lines = aggregates["dev_source_lines"]
    if isinstance(train_lines, int) and isinstance(dev_lines, int):
        aggregates["source_lines_total"] = train_lines + dev_lines
    train_dialogues = aggregates["train_dialogues"]
    dev_dialogues = aggregates["dev_dialogues"]
    if isinstance(train_dialogues, int) and isinstance(dev_dialogues, int):
        aggregates["dialogues_total"] = train_dialogues + dev_dialogues
    return aggregates, reason_counts, unknown_reasons


def _build_line_gate(line: int) -> tuple[str, str] | None:
    ranges = (
        (2033, 2035, "raw_object_verification", "raw_object_binding_or_content"),
        (2036, 2040, "source_parse_train", "source_parse_train"),
        (2041, 2045, "source_parse_dev", "source_parse_dev"),
        (
            2046,
            2074,
            "source_inventory_or_malformed_inventory",
            "source_inventory",
        ),
        (2075, 2075, "vocab_load", "vocab_inventory"),
        (2076, 2084, "candidate_construction", "candidate_inventory_cap"),
        (2085, 2085, "runtime_initialization", "runtime_registry_boundary"),
        (2086, 2097, "new_generic_filter", "new_generic_filter"),
        (
            2098,
            2106,
            "baseline_evidence_or_inventory",
            "baseline_rows_or_token_ledger",
        ),
        (2107, 2115, "baseline_first_selection", "unclassified_value_error"),
        (2116, 2116, "owner_artifacts_in_memory", "owner_packet_contract"),
        (2117, 2131, "serialization_in_memory", "serialization_in_memory"),
        (2132, 2144, "combined_audit", "unclassified_value_error"),
        (2145, 2251, "report_validation", "report_validation"),
    )
    for start, end, stage, gate in ranges:
        if start <= line <= end:
            return stage, gate
    return None


def _classify_failure(error: BaseException) -> tuple[str, str]:
    stage = "raw_object_verification"
    gate = (
        "unclassified_value_error"
        if isinstance(error, ValueError)
        else "unclassified_non_value_error"
    )
    exact: tuple[str, str] | None = None
    current = error.__traceback__
    while current is not None:
        frame = current.tb_frame
        name = frame.f_code.co_name
        line = current.tb_lineno
        if Path(frame.f_code.co_filename).resolve() == Path(v5.__file__).resolve():
            if name == "build_artifacts":
                build_match = _build_line_gate(line)
                if build_match is not None:
                    stage, gate = build_match
            fallback = _FUNCTION_STAGES.get(name)
            if fallback is not None:
                stage = fallback
            mapped = _EXACT_LINE_GATES.get((name, line))
            if mapped is not None:
                exact = (GATE_TO_STAGE[mapped], mapped)
        current = current.tb_next
    if exact is not None:
        stage, gate = exact
    if gate in {"source_parse_train", "source_parse_dev"}:
        # The outer build call is the only source-object discriminator.
        build_frames: list[int] = []
        current = error.__traceback__
        while current is not None:
            if (
                Path(current.tb_frame.f_code.co_filename).resolve()
                == Path(v5.__file__).resolve()
                and current.tb_frame.f_code.co_name == "build_artifacts"
            ):
                build_frames.append(current.tb_lineno)
            current = current.tb_next
        if any(2041 <= line <= 2045 for line in build_frames):
            stage = gate = "source_parse_dev"
        else:
            stage = gate = "source_parse_train"
    return stage, gate


def _stage_results(localized_stage: str, *, build_returned: bool) -> dict[str, str]:
    if localized_stage not in STAGE_ORDER:
        localized_stage = STAGE_ORDER[0]
    failed_index = STAGE_ORDER.index(localized_stage)
    results: dict[str, str] = {}
    for index, stage in enumerate(STAGE_ORDER):
        if build_returned and stage == "build_returned":
            results[stage] = "completed"
        elif index < failed_index:
            results[stage] = "completed"
        elif index == failed_index:
            results[stage] = (
                "completed" if localized_stage == "post_build_checks_complete" else "failed"
            )
        else:
            results[stage] = "not_run"
    return results


def _fill_success_aggregates(
    aggregates: dict[str, int | None], artifacts: v5.BuildArtifacts
) -> None:
    report = artifacts.report
    candidate = report["candidate"]
    curation = report["curation"]
    resources = report["resource_accounting"]
    source_token_ledger = report["audit"]["accepted_metrics"][
        "source_train_supervised_tokens"
    ]
    values = {
        "baseline_rows_loaded": resources["existing_candidate_row_reads"],
        "baseline_sources_loaded": len(source_token_ledger) - 1,
        "candidate_payload_bytes": len(artifacts.candidate_bytes),
        "candidate_rows_proposed": curation["constructed_candidate_rows"],
        "candidate_rows_rejected_before_selection": curation[
            "raw_text_free_rejection_records"
        ],
        "generic_accepted_rows": curation["generic_accepted_rows"],
        "generic_rejected_rows": curation["generic_rejected_rows"],
        "malformed_source_rows": curation["parsed_malformed_source_rows_rejected"],
        "new_rows_processed_by_selection": curation["generic_accepted_rows"],
        "owner_html_bytes": len(artifacts.html_bytes),
        "owner_packet_bytes": len(artifacts.owner_packet_bytes),
        "rejection_payload_bytes": len(artifacts.rejection_bytes),
        "selected_rows": candidate["records"],
        "selected_train_rows": candidate["train_records"],
        "selected_validation_rows": candidate["validation_records"],
        "source_lines_total": resources["source_line_reads"],
        "telemetry_payload_bytes": len(artifacts.telemetry_bytes),
        "train_dialogues": v5.EXPECTED_DIALOGUES["train"],
        "train_source_lines": v5.EXPECTED_LINES["train"],
        "train_supervised_tokens": candidate["train_supervised_tokens"],
        "validation_supervised_tokens": candidate["validation_supervised_tokens"],
        "vocab_english_unique_tokens": EXPECTED_VOCAB_TOKEN_CARDINALITIES["english"],
        "vocab_native_unique_tokens": EXPECTED_VOCAB_TOKEN_CARDINALITIES["native"],
        "vocab_others_unique_tokens": EXPECTED_VOCAB_TOKEN_CARDINALITIES["others"],
        "dev_dialogues": v5.EXPECTED_DIALOGUES["dev"],
        "dev_source_lines": v5.EXPECTED_LINES["dev"],
        "dialogues_total": v5.EXPECTED_DIALOGUES["total"],
    }
    for key, value in values.items():
        _put_count(aggregates, key, value)


def _post_build_checks(
    *, artifacts: v5.BuildArtifacts, root: Path, predecessor_plan_path: Path
) -> tuple[str, str, dict[str, int | None]]:
    aggregates = {key: None for key in sorted(AGGREGATE_KEYS)}
    _fill_success_aggregates(aggregates, artifacts)
    payloads: dict[str, bytes] | None = None
    report_payload: bytes | None = None
    verification_payload: bytes | None = None
    try:
        payloads = runner_v5._payloads(artifacts)
        for name, payload in payloads.items():
            evidence = artifacts.report["artifacts"][name]
            if len(payload) != evidence["bytes"] or _sha256_bytes(payload) != evidence["sha256"]:
                return "post_build_artifact_validation", "post_build_artifact_binding", aggregates
        try:
            v5.validate_report(artifacts.report)
        except Exception:
            return "post_build_artifact_validation", "post_build_artifact_binding", aggregates
        report_payload = v5.json_bytes(artifacts.report)
        try:
            telemetry_ids = verifier_v5._validate_telemetry(
                payloads["telemetry"], artifacts.report
            )
        except Exception:
            return "post_build_telemetry_validation", "post_build_telemetry", aggregates
        try:
            verifier_v5._validate_rejections(payloads["rejection_ledger"], artifacts.report)
        except Exception:
            return "post_build_rejection_validation", "post_build_rejection", aggregates
        try:
            verifier_v5._validate_owner(
                packet_payload=payloads["owner_review_packet"],
                html_payload=payloads["owner_review_html"],
                report=artifacts.report,
                telemetry_ids=telemetry_ids,
            )
        except Exception:
            return "post_build_owner_validation", "post_build_owner", aggregates
        try:
            prospective = v5.build_verification(
                report=artifacts.report,
                report_file_sha256=_sha256_bytes(report_payload),
            )
            v5.validate_verification(prospective)
            verification_payload = v5.json_bytes(prospective)
        except Exception:
            return "prospective_verification", "prospective_verification", aggregates
        try:
            total = runner_v5._prospective_total_write_bytes(
                root=root,
                plan_path=predecessor_plan_path,
                payloads=payloads,
                report=report_payload,
                verification=verification_payload,
            )
            if total > v5.EXPECTED_COMBINED_SUCCESSOR_LIMITS[
                "maximum_private_write_bytes"
            ]:
                return "prospective_write_cap", "prospective_write_cap", aggregates
        except Exception:
            return "prospective_write_cap", "prospective_write_cap", aggregates
        predecessor_paths = {
            name: root / relative for name, relative in v5.EXPECTED_OUTPUT_PATHS.items()
        }
        if predecessor_paths["report"].read_bytes() != runner_v5.RUNNER_MARKER:
            return "predecessor_marker_integrity", "predecessor_marker_integrity", aggregates
        if any(
            path.exists() for name, path in predecessor_paths.items() if name != "report"
        ) or runner_v5._attempt_private_inventory(root):
            return (
                "predecessor_marker_integrity",
                "unexpected_v5_output_appearance",
                aggregates,
            )
        return "post_build_checks_complete", "prior_failure_not_reproduced", aggregates
    finally:
        if payloads is not None:
            payloads.clear()
        report_payload = None
        verification_payload = None


def _resource_accounting(
    *, aggregates: Mapping[str, int | None], runtime_seconds: float
) -> dict[str, Any]:
    source = aggregates.get("source_lines_total")
    baseline = aggregates.get("baseline_rows_loaded")
    new_rows = aggregates.get("new_rows_processed_by_selection")
    if new_rows is None and aggregates.get("selected_rows") is not None:
        new_rows = aggregates["selected_rows"]
    total = baseline + new_rows if isinstance(baseline, int) and isinstance(new_rows, int) else None
    return {
        "build_artifacts_calls": 1,
        "candidate_or_owner_output_bytes": 0,
        "existing_candidate_row_reads": baseline,
        "existing_candidate_row_reads_upper_bound": EXPECTED_LIMITS[
            "maximum_existing_candidate_row_reads"
        ],
        "gpu_use": 0,
        "inference_calls": 0,
        "network_bytes": 0,
        "new_candidate_row_reads": new_rows,
        "new_candidate_row_reads_upper_bound": EXPECTED_LIMITS[
            "maximum_new_candidate_row_reads"
        ],
        "runtime_seconds": round(runtime_seconds, 6),
        "source_line_reads": source,
        "source_line_reads_upper_bound": EXPECTED_LIMITS["maximum_source_line_reads"],
        "total_candidate_audit_row_reads": total,
        "total_candidate_audit_row_reads_upper_bound": EXPECTED_LIMITS[
            "maximum_total_candidate_audit_row_reads"
        ],
    }


def build_report(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    localized_stage: str,
    localized_gate: str,
    build_returned: bool,
    post_build_checks_completed: bool,
    aggregates: Mapping[str, int | None],
    rejection_counts: Mapping[str, int],
    unknown_rejection_reason_occurrences: int,
    runtime_seconds: float,
) -> dict[str, Any]:
    status = (
        "STOP_prior_failure_not_reproduced"
        if localized_gate == "prior_failure_not_reproduced"
        else (
            "PASS_localized_failure_gate"
            if localized_gate
            not in {"unclassified_value_error", "unclassified_non_value_error"}
            else "STOP_unclassified_failure"
        )
    )
    report: dict[str, Any] = {
        "aggregates": dict(aggregates),
        "authorization_text_sha256": AUTHORIZATION_TEXT_SHA256,
        "bindings": {
            name: value["sha256"] for name, value in sorted(plan["bindings"].items())
        },
        "historical_failure": EXPECTED_HISTORICAL_FAILURE,
        "output_inventory": {
            "candidate_or_owner_outputs_created": 0,
            "report": REPORT_PATH,
            "verification": VERIFICATION_PATH,
        },
        "plan_file_sha256": plan_file_sha256,
        "prohibited_actions": EXPECTED_PROHIBITED_ACTIONS,
        "raw_text_retained": False,
        "rejection_counts": {
            **{reason: int(rejection_counts.get(reason, 0)) for reason in REJECTION_CODES},
            "unknown_reason_code_occurrences": unknown_rejection_reason_occurrences,
        },
        "replay": {
            "build_artifacts_calls": 1,
            "build_returned": build_returned,
            "localized_gate": localized_gate,
            "localized_stage": localized_stage,
            "post_build_checks_completed": post_build_checks_completed,
            "prior_failure_reproduced": localized_gate != "prior_failure_not_reproduced",
        },
        "resource_accounting": _resource_accounting(
            aggregates=aggregates, runtime_seconds=runtime_seconds
        ),
        "schema_version": REPORT_SCHEMA,
        "stage_results": _stage_results(localized_stage, build_returned=build_returned),
        "status": status,
    }
    report["report_sha256"] = _internal_sha(report, "report_sha256")
    validate_report(
        report,
        expected_bindings={
            name: value["sha256"] for name, value in sorted(plan["bindings"].items())
        },
        expected_plan_file_sha256=plan_file_sha256,
    )
    return report


def _validate_aggregate_resource_consistency(
    *,
    aggregates: Mapping[str, int | None],
    replay: Mapping[str, Any],
    resources: Mapping[str, Any],
) -> None:
    if set(AGGREGATE_EARLIEST_STAGE) != AGGREGATE_KEYS:
        raise ValueError("diagnostic aggregate stage inventory changed")
    localized_index = STAGE_ORDER.index(replay["localized_stage"])
    for key, earliest_stage in AGGREGATE_EARLIEST_STAGE.items():
        earliest_index = STAGE_ORDER.index(earliest_stage)
        value = aggregates[key]
        if localized_index < earliest_index and value is not None:
            raise ValueError("diagnostic aggregate appeared before its stage")
        if localized_index > earliest_index and value is None:
            raise ValueError("diagnostic completed-stage aggregate is absent")

    for total_key, left_key, right_key in (
        ("source_lines_total", "train_source_lines", "dev_source_lines"),
        ("dialogues_total", "train_dialogues", "dev_dialogues"),
    ):
        total = aggregates[total_key]
        left = aggregates[left_key]
        right = aggregates[right_key]
        if total is None:
            if left is not None and right is not None:
                raise ValueError("diagnostic aggregate total is absent")
        elif left is None or right is None or total != left + right:
            raise ValueError("diagnostic aggregate sum changed")

    selected = aggregates["selected_rows"]
    selected_train = aggregates["selected_train_rows"]
    selected_validation = aggregates["selected_validation_rows"]
    if selected is None:
        if selected_train is not None or selected_validation is not None:
            raise ValueError("diagnostic selected split aggregate is partial")
    elif (
        selected_train is None
        or selected_validation is None
        or selected != selected_train + selected_validation
    ):
        raise ValueError("diagnostic selected split aggregate changed")

    for lesser_key, greater_key in (
        ("generic_accepted_rows", "candidate_rows_proposed"),
        ("selected_rows", "generic_accepted_rows"),
        ("selected_rows", "new_rows_processed_by_selection"),
    ):
        lesser = aggregates[lesser_key]
        greater = aggregates[greater_key]
        if lesser is not None and greater is not None and lesser > greater:
            raise ValueError("diagnostic aggregate ordering changed")

    expected_new = aggregates["new_rows_processed_by_selection"]
    if expected_new is None and replay["build_returned"]:
        expected_new = aggregates["selected_rows"]
    expected_existing = aggregates["baseline_rows_loaded"]
    expected_source = aggregates["source_lines_total"]
    expected_total = (
        expected_existing + expected_new
        if isinstance(expected_existing, int) and isinstance(expected_new, int)
        else None
    )
    if (
        resources["source_line_reads"] != expected_source
        or resources["existing_candidate_row_reads"] != expected_existing
        or resources["new_candidate_row_reads"] != expected_new
        or resources["total_candidate_audit_row_reads"] != expected_total
    ):
        raise ValueError("diagnostic aggregate/resource linkage changed")


def validate_report(
    report: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, str] | None = None,
    expected_plan_file_sha256: str | None = None,
) -> None:
    if set(report) != {
        "aggregates",
        "authorization_text_sha256",
        "bindings",
        "historical_failure",
        "output_inventory",
        "plan_file_sha256",
        "prohibited_actions",
        "raw_text_retained",
        "rejection_counts",
        "replay",
        "report_sha256",
        "resource_accounting",
        "schema_version",
        "stage_results",
        "status",
    }:
        raise ValueError("diagnostic report inventory changed")
    replay = report.get("replay")
    aggregates = report.get("aggregates")
    resources = report.get("resource_accounting")
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or report.get("authorization_text_sha256") != AUTHORIZATION_TEXT_SHA256
        or report.get("historical_failure") != EXPECTED_HISTORICAL_FAILURE
        or report.get("output_inventory")
        != {
            "candidate_or_owner_outputs_created": 0,
            "report": REPORT_PATH,
            "verification": VERIFICATION_PATH,
        }
        or report.get("prohibited_actions") != EXPECTED_PROHIBITED_ACTIONS
        or report.get("raw_text_retained") is not False
        or not _validate_sha(report.get("plan_file_sha256"))
        or not isinstance(report.get("bindings"), Mapping)
        or set(report["bindings"]) != set(EXPECTED_BINDING_PATHS)
        or not all(_validate_sha(value) for value in report["bindings"].values())
        or (
            expected_bindings is not None
            and dict(report["bindings"]) != dict(expected_bindings)
        )
        or (
            expected_plan_file_sha256 is not None
            and report.get("plan_file_sha256") != expected_plan_file_sha256
        )
        or not isinstance(replay, Mapping)
        or set(replay)
        != {
            "build_artifacts_calls",
            "build_returned",
            "localized_gate",
            "localized_stage",
            "post_build_checks_completed",
            "prior_failure_reproduced",
        }
        or replay.get("build_artifacts_calls") != 1
        or replay.get("localized_gate") not in ALLOWED_GATE_CODES
        or (
            replay.get("localized_gate")
            not in {"unclassified_value_error", "unclassified_non_value_error"}
            and replay.get("localized_stage") != GATE_TO_STAGE[replay["localized_gate"]]
        )
        or (
            replay.get("localized_gate")
            in {"unclassified_value_error", "unclassified_non_value_error"}
            and replay.get("localized_stage") not in STAGE_ORDER
        )
        or not all(
            isinstance(replay.get(key), bool)
            for key in (
                "build_returned",
                "post_build_checks_completed",
                "prior_failure_reproduced",
            )
        )
        or replay["prior_failure_reproduced"]
        != (replay["localized_gate"] != "prior_failure_not_reproduced")
        or replay["post_build_checks_completed"]
        != (replay["localized_gate"] == "prior_failure_not_reproduced")
        or replay["build_returned"]
        != (
            STAGE_ORDER.index(replay["localized_stage"])
            >= STAGE_ORDER.index("build_returned")
        )
    ):
        raise ValueError("diagnostic report boundary changed")
    expected_status = (
        "STOP_prior_failure_not_reproduced"
        if replay["localized_gate"] == "prior_failure_not_reproduced"
        else (
            "PASS_localized_failure_gate"
            if replay["localized_gate"]
            not in {"unclassified_value_error", "unclassified_non_value_error"}
            else "STOP_unclassified_failure"
        )
    )
    if report.get("status") != expected_status:
        raise ValueError("diagnostic status changed")
    if (
        not isinstance(aggregates, Mapping)
        or set(aggregates) != AGGREGATE_KEYS
        or any(
            value is not None and (not isinstance(value, int) or value < 0)
            for value in aggregates.values()
        )
    ):
        raise ValueError("diagnostic aggregate inventory changed")
    counts = report.get("rejection_counts")
    if (
        not isinstance(counts, Mapping)
        or set(counts) != set(REJECTION_CODES) | {"unknown_reason_code_occurrences"}
        or any(not isinstance(value, int) or value < 0 for value in counts.values())
    ):
        raise ValueError("diagnostic rejection counts changed")
    stages = report.get("stage_results")
    if (
        not isinstance(stages, Mapping)
        or set(stages) != set(STAGE_ORDER)
        or any(value not in {"completed", "failed", "not_run"} for value in stages.values())
        or stages != _stage_results(
            replay["localized_stage"], build_returned=replay["build_returned"]
        )
    ):
        raise ValueError("diagnostic stage transitions changed")
    if (
        not isinstance(resources, Mapping)
        or set(resources)
        != {
            "build_artifacts_calls",
            "candidate_or_owner_output_bytes",
            "existing_candidate_row_reads",
            "existing_candidate_row_reads_upper_bound",
            "gpu_use",
            "inference_calls",
            "network_bytes",
            "new_candidate_row_reads",
            "new_candidate_row_reads_upper_bound",
            "runtime_seconds",
            "source_line_reads",
            "source_line_reads_upper_bound",
            "total_candidate_audit_row_reads",
            "total_candidate_audit_row_reads_upper_bound",
        }
        or resources["build_artifacts_calls"] != 1
        or resources["candidate_or_owner_output_bytes"] != 0
        or any(resources[key] != 0 for key in ("gpu_use", "inference_calls", "network_bytes"))
        or resources["source_line_reads_upper_bound"]
        != EXPECTED_LIMITS["maximum_source_line_reads"]
        or resources["existing_candidate_row_reads_upper_bound"]
        != EXPECTED_LIMITS["maximum_existing_candidate_row_reads"]
        or resources["new_candidate_row_reads_upper_bound"]
        != EXPECTED_LIMITS["maximum_new_candidate_row_reads"]
        or resources["total_candidate_audit_row_reads_upper_bound"]
        != EXPECTED_LIMITS["maximum_total_candidate_audit_row_reads"]
        or not isinstance(resources["runtime_seconds"], (int, float))
        or not 0 <= resources["runtime_seconds"] <= 30 * 60
    ):
        raise ValueError("diagnostic resource accounting changed")
    for observed, maximum in (
        (resources["source_line_reads"], resources["source_line_reads_upper_bound"]),
        (
            resources["existing_candidate_row_reads"],
            resources["existing_candidate_row_reads_upper_bound"],
        ),
        (resources["new_candidate_row_reads"], resources["new_candidate_row_reads_upper_bound"]),
        (
            resources["total_candidate_audit_row_reads"],
            resources["total_candidate_audit_row_reads_upper_bound"],
        ),
    ):
        if observed is not None and (not isinstance(observed, int) or not 0 <= observed <= maximum):
            raise ValueError("diagnostic observed resource count changed")
    _validate_aggregate_resource_consistency(
        aggregates=aggregates,
        replay=replay,
        resources=resources,
    )
    if report.get("report_sha256") != _internal_sha(report, "report_sha256"):
        raise ValueError("diagnostic report internal seal changed")


def build_structural_verification(
    *,
    report: Mapping[str, Any],
    report_payload: bytes,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
) -> dict[str, Any]:
    validate_report(
        report,
        expected_bindings={
            name: value["sha256"] for name, value in sorted(plan["bindings"].items())
        },
        expected_plan_file_sha256=plan_file_sha256,
    )
    verification: dict[str, Any] = {
        "candidate_or_baseline_payload_opened": False,
        "checks": {
            "aggregate_and_resource_bounds_verified": True,
            "authorization_and_binding_inventory_verified": True,
            "closed_stage_and_gate_mapping_verified": True,
            "no_candidate_or_owner_output_verified": True,
            "raw_text_boundary_verified": True,
            "report_file_and_internal_hashes_verified": True,
            "stage_transition_legality_verified": True,
        },
        "plan_file_sha256": plan_file_sha256,
        "report_file_sha256": _sha256_bytes(report_payload),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA,
        "source_replay_performed": False,
        "status": "PASS_structurally_verified_stage_localization",
    }
    verification["verification_sha256"] = _internal_sha(
        verification, "verification_sha256"
    )
    validate_structural_verification(verification, report=report, report_payload=report_payload)
    return verification


def validate_structural_verification(
    verification: Mapping[str, Any], *, report: Mapping[str, Any], report_payload: bytes
) -> None:
    if set(verification) != {
        "candidate_or_baseline_payload_opened",
        "checks",
        "plan_file_sha256",
        "report_file_sha256",
        "report_internal_sha256",
        "schema_version",
        "source_replay_performed",
        "status",
        "verification_sha256",
    }:
        raise ValueError("diagnostic verification inventory changed")
    if (
        verification.get("schema_version") != VERIFICATION_SCHEMA
        or verification.get("status") != "PASS_structurally_verified_stage_localization"
        or verification.get("candidate_or_baseline_payload_opened") is not False
        or verification.get("source_replay_performed") is not False
        or verification.get("plan_file_sha256") != report.get("plan_file_sha256")
        or verification.get("report_file_sha256") != _sha256_bytes(report_payload)
        or verification.get("report_internal_sha256") != report.get("report_sha256")
        or not isinstance(verification.get("checks"), Mapping)
        or set(verification["checks"])
        != {
            "aggregate_and_resource_bounds_verified",
            "authorization_and_binding_inventory_verified",
            "closed_stage_and_gate_mapping_verified",
            "no_candidate_or_owner_output_verified",
            "raw_text_boundary_verified",
            "report_file_and_internal_hashes_verified",
            "stage_transition_legality_verified",
        }
        or not all(value is True for value in verification["checks"].values())
        or verification.get("verification_sha256")
        != _internal_sha(verification, "verification_sha256")
    ):
        raise ValueError("diagnostic verification boundary changed")


def _verification_with_write_accounting(
    *,
    verification: dict[str, Any],
    root: Path,
    plan_path: Path,
    report_payload: bytes,
) -> tuple[dict[str, Any], bytes, int]:
    # The exact physical write accounting includes final plan/code/test bytes,
    # the durable marker write, final report rewrite, and verification write.
    static_bytes = sum(
        path.stat().st_size
        for path in (plan_path, root / IMPLEMENTATION_PATH, root / TEST_PATH)
    )
    candidate = dict(verification)
    for _ in range(8):
        candidate.pop("verification_sha256", None)
        candidate["verification_sha256"] = _internal_sha(
            candidate, "verification_sha256"
        )
        payload = _json_bytes(candidate)
        total = static_bytes + len(REPORT_MARKER) + len(report_payload) + len(payload)
        # Total is checked externally and intentionally not embedded, avoiding a
        # self-referential byte-length field.
        if total <= EXPECTED_LIMITS["maximum_private_write_bytes"]:
            return candidate, payload, total
    raise ValueError("diagnostic verification accounting did not converge")


def _execute_once(
    *, root: Path, plan_path: Path, plan: Mapping[str, Any]
) -> tuple[dict[str, Any], bytes]:
    started = time.monotonic()
    predecessor_plan_path = root / PREDECESSOR_BINDINGS["predecessor_plan"][0]
    predecessor_plan = _load_json(predecessor_plan_path)
    aggregates = {key: None for key in sorted(AGGREGATE_KEYS)}
    rejection_counts = {reason: 0 for reason in REJECTION_CODES}
    unknown_reasons = 0
    build_returned = False
    post_build_checks_completed = False
    artifacts: v5.BuildArtifacts | None = None
    try:
        artifacts = v5.build_artifacts(
            plan=predecessor_plan,
            plan_file_sha256=file_sha256(predecessor_plan_path),
            repository_root=root,
            started=started,
        )
        build_returned = True
        _fill_success_aggregates(aggregates, artifacts)
        localized_stage, localized_gate, post_aggregates = _post_build_checks(
            artifacts=artifacts,
            root=root,
            predecessor_plan_path=predecessor_plan_path,
        )
        aggregates.update(
            {key: value for key, value in post_aggregates.items() if value is not None}
        )
        post_build_checks_completed = localized_gate == "prior_failure_not_reproduced"
    except BaseException as error:
        tb = error.__traceback__
        localized_stage, localized_gate = _classify_failure(error)
        aggregates, rejection_counts, unknown_reasons = _extract_safe_aggregates(tb)
        if tb is not None:
            traceback.clear_frames(tb)
        del tb
        del error
    finally:
        artifacts = None
    report = build_report(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        localized_stage=localized_stage,
        localized_gate=localized_gate,
        build_returned=build_returned,
        post_build_checks_completed=post_build_checks_completed,
        aggregates=aggregates,
        rejection_counts=rejection_counts,
        unknown_rejection_reason_occurrences=unknown_reasons,
        runtime_seconds=time.monotonic() - started,
    )
    return report, _json_bytes(report)


def main() -> int:
    process_started = time.monotonic()
    _arm_runtime_alarm()
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_diagnostic_plan(plan_path, repository_root=root)
    paths = _preflight_outputs(root)
    if time.monotonic() - process_started > EXPECTED_LIMITS["maximum_runtime_minutes"] * 60:
        raise TimeoutError("diagnostic preflight exceeded runtime cap")
    _write_exclusive(paths["report"], REPORT_MARKER)
    report, report_payload = _execute_once(root=root, plan_path=plan_path, plan=plan)
    validate_report(report)
    verification = build_structural_verification(
        report=report,
        report_payload=report_payload,
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
    )
    verification, verification_payload, total_write_bytes = _verification_with_write_accounting(
        verification=verification,
        root=root,
        plan_path=plan_path,
        report_payload=report_payload,
    )
    validate_structural_verification(
        verification, report=report, report_payload=report_payload
    )
    if total_write_bytes > EXPECTED_LIMITS["maximum_private_write_bytes"]:
        raise ValueError("diagnostic private-write cap exceeded")
    predecessor_paths = {
        name: root / relative for name, relative in v5.EXPECTED_OUTPUT_PATHS.items()
    }
    if predecessor_paths["report"].read_bytes() != runner_v5.RUNNER_MARKER:
        raise ValueError("failed attempt-005 marker changed before publication")
    if any(
        path.exists() for name, path in predecessor_paths.items() if name != "report"
    ) or runner_v5._attempt_private_inventory(root):
        raise ValueError("failed attempt-005 output appeared during diagnostic")
    _finish_marker(paths["report"], report_payload)
    _write_exclusive(paths["verification"], verification_payload)
    _cancel_runtime_alarm()
    print(
        json.dumps(
            {
                "localized_gate": report["replay"]["localized_gate"],
                "report_file_sha256": _sha256_bytes(report_payload),
                "status": report["status"],
                "verification_file_sha256": _sha256_bytes(verification_payload),
            },
            sort_keys=True,
        )
    )
    return 0


def _entrypoint() -> int:
    try:
        return main()
    except BaseException:
        _cancel_runtime_alarm()
        print(json.dumps({"status": "FAIL_CLOSED"}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
