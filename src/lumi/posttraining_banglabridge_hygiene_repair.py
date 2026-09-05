"""Deterministic, result-preserving BanglaBridge closure-candidate repair."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .data_quality import BenchmarkRegistry, normalize_for_matching, sha256_file
from .posttraining_closure_canary_curation import (
    PurePythonNearDuplicateIndex,
    _custom_quality_reasons,
    _record_text,
    load_direct_tokenizer,
)
from .posttraining_data import evaluate_sft_record, load_sft_policy
from .posttraining_format import encode_sft_messages, load_chat_template

PLAN_SCHEMA_VERSION = "lumi.sft-banglabridge-hygiene-repair.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-banglabridge-hygiene-repair-report.v0.1"
MANIFEST_SCHEMA_VERSION = "lumi.sft-banglabridge-hygiene-repair-manifest.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-banglabridge-hygiene-repair-verification.v0.1"
PLAN_STATUS = "authorized_cpu_only_private_nonproduction_repair"
REPORT_STATUS = "repaired_private_nonproduction_candidate_registry_unchanged"

EXPECTED_AUTHORIZATION = {
    "cpu_only": True,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "gpu_authorized": False,
    "human_authorship_claim_authorized": False,
    "network_access_authorized": False,
    "production_promotion_authorized": False,
    "registry_change_authorized": False,
    "sft_training_authorized": False,
}
EXPECTED_LIMITS = {
    "candidate_input_records": 561,
    "maximum_private_write_bytes": 16_777_216,
    "maximum_runtime_minutes": 60,
    "maximum_source_rows": 608,
    "maximum_train_supervised_tokens": 50_000,
}
EXPECTED_BINDINGS = {
    "benchmark_registry",
    "chat_template",
    "data_contract",
    "data_quality",
    "implementation",
    "input_candidate",
    "owner_review_receipt",
    "owner_review_sample",
    "owner_review_verification",
    "preflight_report",
    "preflight_verification",
    "runner",
    "source_rows",
    "test",
    "tokenizer",
    "verifier",
}

EXPECTED_PREFLIGHT_REPORT_FILE_SHA256 = (
    "4c3cd6acc971fe9d5686fb4026f79133426b037fdd424bfe7a324b60fb6de17a"
)
EXPECTED_PREFLIGHT_VERIFICATION_FILE_SHA256 = (
    "8e94531bf52db58da97091d581e3db62f137f4369130869f6a32ee8c32834131"
)
EXPECTED_INPUT_CANDIDATE_SHA256 = (
    "b75c3517be8dba9295d36f93e3275f06d1287da7e2fe5b60fdfadbc4f27e9900"
)
EXPECTED_SOURCE_ROWS_SHA256 = (
    "313e5efb38760d8de09050162b6d8c5a2db76b9107cb2c24d828012113a7a73d"
)
EXPECTED_TOKENIZER_SHA256 = (
    "1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee"
)
EXPECTED_BENCHMARK_SHA256 = (
    "1ffb5e938ab28c2fc876b9efd6c6ed59059f1e85aed2c396c024ae5925a4745e"
)
SOURCE_ID = "banglabridge_instructions_closure_canary"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REGION_SPECIFIC_EMERGENCY = re.compile(r"(?<!\d)(?:108|1930|14416)(?!\d)")
_PROVENANCE_FIELDS = (
    "source_adapted_instruction_sha256",
    "source_adapted_output_sha256",
    "source_original_instruction_sha256",
    "source_original_output_sha256",
)


def sha256_value(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def write_bytes_atomic(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    write_bytes_atomic(path, _json_bytes(value))


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(row)
    return rows


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def resolve_binding(
    repository_root: str | Path,
    specification: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    root = Path(repository_root).resolve()
    relative = specification.get("path")
    expected = _require_sha256(specification.get("sha256"), f"bindings.{name}.sha256")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"invalid BanglaBridge repair binding: {name}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.parent.resolve()) or not path.is_file():
        raise ValueError(f"BanglaBridge repair binding missing: {name}")
    if sha256_file(path) != expected:
        raise ValueError(f"BanglaBridge repair binding changed: {name}")
    return path


def load_repair_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    plan = _read_object(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported BanglaBridge hygiene-repair plan")
    if plan.get("status") != PLAN_STATUS:
        raise ValueError("BanglaBridge hygiene-repair status changed")
    if plan.get("observed_at") != "2026-08-10":
        raise ValueError("BanglaBridge hygiene-repair date changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("BanglaBridge hygiene-repair authorization changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("BanglaBridge hygiene-repair limits changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("BanglaBridge hygiene-repair bindings changed")
    for name, specification in bindings.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"invalid BanglaBridge hygiene-repair binding: {name}")
        resolve_binding(repository_root, specification, name=name)
    if bindings["preflight_report"]["sha256"] != EXPECTED_PREFLIGHT_REPORT_FILE_SHA256:
        raise ValueError("BanglaBridge repair preflight report changed")
    if (
        bindings["preflight_verification"]["sha256"]
        != EXPECTED_PREFLIGHT_VERIFICATION_FILE_SHA256
    ):
        raise ValueError("BanglaBridge repair preflight verification changed")
    if bindings["input_candidate"]["sha256"] != EXPECTED_INPUT_CANDIDATE_SHA256:
        raise ValueError("BanglaBridge input candidate changed")
    if bindings["source_rows"]["sha256"] != EXPECTED_SOURCE_ROWS_SHA256:
        raise ValueError("BanglaBridge source rows changed")
    if bindings["tokenizer"]["sha256"] != EXPECTED_TOKENIZER_SHA256:
        raise ValueError("BanglaBridge tokenizer changed")
    if bindings["benchmark_registry"]["sha256"] != EXPECTED_BENCHMARK_SHA256:
        raise ValueError("BanglaBridge benchmark registry changed")
    output = plan.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("BanglaBridge hygiene-repair output is missing")
    expected_output = {
        "candidate_path": (
            "../lumi-artifacts/4b-sft-banglabridge-hygiene-repair-v0.1/"
            "4c3cd6acc971/attempt-001/private/banglabridge-candidate.jsonl"
        ),
        "manifest_path": (
            "../lumi-artifacts/4b-sft-banglabridge-hygiene-repair-v0.1/"
            "4c3cd6acc971/attempt-001/private-manifest.json"
        ),
        "report_path": "evidence/lumi-4b-sft-banglabridge-hygiene-repair-v0.1.json",
        "verification_path": (
            "evidence/lumi-4b-sft-banglabridge-hygiene-repair-verification-v0.1.json"
        ),
    }
    if output != expected_output:
        raise ValueError("BanglaBridge hygiene-repair outputs changed")
    return plan


def _validate_internal_seal(value: Mapping[str, Any], field: str, label: str) -> None:
    unsigned = {key: item for key, item in value.items() if key != field}
    if value.get(field) != sha256_value(unsigned):
        raise ValueError(f"{label} internal seal changed")


def _verify_authorization_bindings(root: Path, bindings: Mapping[str, Any]) -> None:
    preflight = _read_object(
        resolve_binding(root, bindings["preflight_report"], name="preflight_report")
    )
    _validate_internal_seal(preflight, "report_sha256", "promotion preflight")
    if (
        preflight.get("status")
        != "blocked_repair_required_registry_successor_not_projected"
    ):
        raise ValueError("BanglaBridge repair preflight status changed")
    blocker_codes = {
        blocker.get("code")
        for blocker in preflight.get("blockers", [])
        if isinstance(blocker, Mapping)
    }
    if not {
        "banglabridge_train_token_ceiling_exceeded",
        "banglabridge_original_and_adapted_provenance_hashes_missing",
    }.issubset(blocker_codes):
        raise ValueError("BanglaBridge repair blockers are not bound")
    verification = _read_object(
        resolve_binding(
            root,
            bindings["preflight_verification"],
            name="preflight_verification",
        )
    )
    _validate_internal_seal(
        verification, "verification_sha256", "promotion preflight verification"
    )
    if verification.get("status") != (
        "verified_blocked_preflight_registry_successor_not_projected"
    ):
        raise ValueError("BanglaBridge repair preflight verification status changed")


def _owner_reviewed_rows(
    candidate_by_id: Mapping[str, Mapping[str, Any]],
    sample: list[dict[str, Any]],
    receipt: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> tuple[set[str], dict[str, Any]]:
    _validate_internal_seal(receipt, "receipt_sha256", "owner-review receipt")
    _validate_internal_seal(
        verification, "verification_sha256", "owner-review verification"
    )
    if (
        receipt.get("all_sampled_rows_accepted") is not True
        or receipt.get("accepted_records") != 32
        or receipt.get("rejected_records") != 0
        or receipt.get("production_eligible") is not False
        or sha256_value(sample) != receipt.get("sample_sha256")
    ):
        raise ValueError("owner-review evidence changed")
    bridge_rows = [
        row
        for row in sample
        if isinstance(row.get("skill_id"), str)
        and str(row["skill_id"]).startswith(f"{SOURCE_ID}:")
    ]
    if len(bridge_rows) != 8:
        raise ValueError("BanglaBridge owner-review sample count changed")
    reviewed: set[str] = set()
    for sample_row in bridge_rows:
        record_id = sample_row.get("record_id")
        if not isinstance(record_id, str) or record_id not in candidate_by_id:
            raise ValueError("owner-reviewed BanglaBridge row is missing")
        candidate = candidate_by_id[record_id]
        if (
            candidate.get("messages") != sample_row.get("messages")
            or candidate.get("split") != sample_row.get("split")
        ):
            raise ValueError("owner-reviewed BanglaBridge visible row changed")
        reviewed.add(record_id)
    return reviewed, {
        "bridge_sample_records": 8,
        "owner_review_receipt_sha256": receipt["receipt_sha256"],
        "sample_sha256": receipt["sample_sha256"],
        "visible_messages_preserved": True,
    }


def _repair_row(
    row: Mapping[str, Any],
    source_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    source = row.get("source")
    if not isinstance(source, Mapping) or source.get("source_id") != SOURCE_ID:
        raise ValueError("BanglaBridge candidate source changed")
    document_id = source.get("document_id")
    if not isinstance(document_id, str) or not document_id.startswith("row-"):
        raise ValueError("BanglaBridge source document ID changed")
    try:
        index = int(document_id.removeprefix("row-"))
    except ValueError as error:
        raise ValueError("BanglaBridge source document ID is invalid") from error
    if index < 0 or index >= len(source_rows):
        raise ValueError("BanglaBridge source document ID escaped source rows")
    original = source_rows[index]
    required = ("instruction", "output", "orig_instruction", "orig_output")
    if any(not isinstance(original.get(field), str) for field in required):
        raise ValueError("BanglaBridge original/adapted provenance fields are missing")
    expected_messages = [
        {"role": "user", "content": str(original["instruction"]).strip()},
        {"role": "assistant", "content": str(original["output"]).strip()},
    ]
    if row.get("messages") != expected_messages:
        raise ValueError("BanglaBridge visible messages changed from adapted source")
    visible = "\n".join(message["content"] for message in expected_messages)
    if _REGION_SPECIFIC_EMERGENCY.search(visible):
        raise ValueError("region-specific emergency number reached repair candidate")
    repaired = copy.deepcopy(dict(row))
    provenance = repaired.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("BanglaBridge provenance object is missing")
    values = {
        "source_adapted_instruction_sha256": hashlib.sha256(
            str(original["instruction"]).encode("utf-8")
        ).hexdigest(),
        "source_adapted_output_sha256": hashlib.sha256(
            str(original["output"]).encode("utf-8")
        ).hexdigest(),
        "source_original_instruction_sha256": hashlib.sha256(
            str(original["orig_instruction"]).encode("utf-8")
        ).hexdigest(),
        "source_original_output_sha256": hashlib.sha256(
            str(original["orig_output"]).encode("utf-8")
        ).hexdigest(),
    }
    if any(field in provenance for field in _PROVENANCE_FIELDS):
        raise ValueError("BanglaBridge provenance repair would overwrite existing fields")
    provenance.update(values)
    if repaired.get("messages") != row.get("messages"):
        raise ValueError("BanglaBridge repair changed visible messages")
    return repaired


def _priority(record_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"banglabridge-hygiene-v0.1\0{record_id}".encode("utf-8")
    ).hexdigest()
    return digest, record_id


def _audit_and_measure(
    rows: list[dict[str, Any]],
    *,
    repository_root: Path,
    bindings: Mapping[str, Any],
) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    tokenizer = load_direct_tokenizer(
        resolve_binding(repository_root, bindings["tokenizer"], name="tokenizer")
    )
    template = load_chat_template(
        resolve_binding(repository_root, bindings["chat_template"], name="chat_template")
    )
    policy = load_sft_policy(
        resolve_binding(repository_root, bindings["data_contract"], name="data_contract")
    )
    quality_path = resolve_binding(
        repository_root, bindings["data_quality"], name="data_quality"
    )
    quality = _read_object(quality_path)
    if (
        quality.get("benchmark_decontamination", {}).get("registry_sha256")
        != EXPECTED_BENCHMARK_SHA256
    ):
        raise ValueError("BanglaBridge benchmark policy changed")
    benchmark = BenchmarkRegistry(
        quality["benchmark_decontamination"], repository_root
    )
    metrics: dict[str, dict[str, int]] = {}
    rejection_counts: Counter[str] = Counter()
    benchmark_matches = 0
    for row in rows:
        reasons = list(_custom_quality_reasons(row))
        decision = evaluate_sft_record(
            row,
            policy=policy,
            benchmark_registry=benchmark,
            source_registry=None,
        )
        reasons.extend(decision.reasons)
        if any("benchmark" in reason for reason in reasons):
            benchmark_matches += 1
        if reasons:
            rejection_counts.update(dict.fromkeys(reasons, 1))
            continue
        encoded = encode_sft_messages(
            row["messages"], tokenizer=tokenizer, template=template
        )
        metrics[str(row["record_id"])] = {
            "serialized_tokens": len(encoded.input_ids),
            "supervised_tokens": encoded.supervised_tokens,
        }
    if rejection_counts or len(metrics) != len(rows):
        raise ValueError(
            f"BanglaBridge repaired candidate failed machine gates: {dict(rejection_counts)}"
        )
    return metrics, {
        "benchmark_matches": benchmark_matches,
        "machine_accepted_records": len(metrics),
        "machine_rejected_records": 0,
    }


def _verify_dedup(rows: list[dict[str, Any]]) -> dict[str, int]:
    exact_seen: set[str] = set()
    near = PurePythonNearDuplicateIndex()
    exact_duplicates = 0
    near_duplicates = 0
    for row in sorted(rows, key=lambda item: str(item["record_id"])):
        exact = sha256_value(
            [
                {
                    "content": normalize_for_matching(str(message["content"])),
                    "role": message["role"],
                }
                for message in row["messages"]
            ]
        )
        if exact in exact_seen:
            exact_duplicates += 1
            continue
        exact_seen.add(exact)
        duplicate_of, _ = near.duplicate_of(str(row["record_id"]), _record_text(row))
        if duplicate_of is not None:
            near_duplicates += 1
    if exact_duplicates or near_duplicates:
        raise ValueError("BanglaBridge repaired candidate introduced duplicate rows")
    return {
        "exact_duplicates": exact_duplicates,
        "near_duplicates_at_0_82": near_duplicates,
    }


def build_repair(
    plan: Mapping[str, Any],
    *,
    plan_file_sha256: str,
    repository_root: str | Path,
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    root = Path(repository_root).resolve()
    bindings = plan["bindings"]
    _verify_authorization_bindings(root, bindings)
    input_rows = _read_jsonl(
        resolve_binding(root, bindings["input_candidate"], name="input_candidate")
    )
    source_rows = _read_jsonl(
        resolve_binding(root, bindings["source_rows"], name="source_rows")
    )
    if len(input_rows) != 561 or len(source_rows) != 608:
        raise ValueError("BanglaBridge repair input bounds changed")
    candidate_by_id: dict[str, dict[str, Any]] = {}
    for row in input_rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or record_id in candidate_by_id:
            raise ValueError("BanglaBridge candidate record IDs changed")
        candidate_by_id[record_id] = row
    sample = json.loads(
        resolve_binding(root, bindings["owner_review_sample"], name="owner_review_sample")
        .read_text(encoding="utf-8")
    )
    if not isinstance(sample, list):
        raise ValueError("owner-review sample root changed")
    receipt = _read_object(
        resolve_binding(root, bindings["owner_review_receipt"], name="owner_review_receipt")
    )
    owner_verification = _read_object(
        resolve_binding(
            root,
            bindings["owner_review_verification"],
            name="owner_review_verification",
        )
    )
    reviewed_ids, owner_review = _owner_reviewed_rows(
        candidate_by_id, sample, receipt, owner_verification
    )

    repaired_rows = [_repair_row(row, source_rows) for row in input_rows]
    repaired_by_id = {str(row["record_id"]): row for row in repaired_rows}
    metrics, machine_gate = _audit_and_measure(
        repaired_rows,
        repository_root=root,
        bindings=bindings,
    )
    reviewed_train = sorted(
        (
            repaired_by_id[record_id]
            for record_id in reviewed_ids
            if repaired_by_id[record_id]["split"] == "train"
        ),
        key=lambda row: str(row["record_id"]),
    )
    reviewed_train_tokens = sum(
        metrics[str(row["record_id"])]["supervised_tokens"] for row in reviewed_train
    )
    ceiling = int(plan["limits"]["maximum_train_supervised_tokens"])
    if reviewed_train_tokens > ceiling:
        raise ValueError("owner-reviewed BanglaBridge train rows exceed token ceiling")
    selected_train = list(reviewed_train)
    selected_ids = {str(row["record_id"]) for row in selected_train}
    train_tokens = reviewed_train_tokens
    remaining_train = sorted(
        (
            row
            for row in repaired_rows
            if row["split"] == "train" and row["record_id"] not in selected_ids
        ),
        key=lambda row: _priority(str(row["record_id"])),
    )
    fit_skipped = 0
    for row in remaining_train:
        row_tokens = metrics[str(row["record_id"])]["supervised_tokens"]
        if train_tokens + row_tokens > ceiling:
            fit_skipped += 1
            continue
        selected_train.append(row)
        selected_ids.add(str(row["record_id"]))
        train_tokens += row_tokens
    validation_rows = sorted(
        (row for row in repaired_rows if row["split"] == "validation"),
        key=lambda row: str(row["record_id"]),
    )
    selected = sorted(
        selected_train + validation_rows,
        key=lambda row: (str(row["split"]), str(row["record_id"])),
    )
    if not reviewed_ids.issubset({str(row["record_id"]) for row in selected}):
        raise ValueError("repair dropped an owner-reviewed BanglaBridge row")
    if train_tokens > ceiling:
        raise ValueError("BanglaBridge repaired train candidate exceeded ceiling")
    validation_tokens = sum(
        metrics[str(row["record_id"])]["supervised_tokens"] for row in validation_rows
    )
    serialized_train_tokens = sum(
        metrics[str(row["record_id"])]["serialized_tokens"] for row in selected_train
    )
    serialized_validation_tokens = sum(
        metrics[str(row["record_id"])]["serialized_tokens"] for row in validation_rows
    )
    dedup = _verify_dedup(selected)
    candidate_payload = _jsonl_bytes(selected)
    candidate_sha256 = hashlib.sha256(candidate_payload).hexdigest()
    ledger = [
        {
            "record_id": row["record_id"],
            "serialized_tokens": metrics[str(row["record_id"])]["serialized_tokens"],
            "split": row["split"],
            "supervised_tokens": metrics[str(row["record_id"])]["supervised_tokens"],
        }
        for row in selected
    ]
    manifest: dict[str, Any] = {
        "candidate": {
            "bytes": len(candidate_payload),
            "records": len(selected),
            "sha256": candidate_sha256,
            "splits": {
                "train": {
                    "records": len(selected_train),
                    "serialized_tokens": serialized_train_tokens,
                    "supervised_tokens": train_tokens,
                },
                "validation": {
                    "records": len(validation_rows),
                    "serialized_tokens": serialized_validation_tokens,
                    "supervised_tokens": validation_tokens,
                },
            },
        },
        "input_candidate_sha256": EXPECTED_INPUT_CANDIDATE_SHA256,
        "owner_review": owner_review,
        "provenance_fields_added": list(_PROVENANCE_FIELDS),
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_id": SOURCE_ID,
        "status": "private_nonproduction_repaired_candidate",
        "token_ledger_sha256": sha256_value(ledger),
    }
    manifest["manifest_sha256"] = sha256_value(manifest)
    manifest_file_sha256 = hashlib.sha256(_json_bytes(manifest)).hexdigest()
    private_write_bytes = len(candidate_payload) + len(_json_bytes(manifest))
    if private_write_bytes > int(plan["limits"]["maximum_private_write_bytes"]):
        raise ValueError("BanglaBridge repair exceeded private-write limit")
    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "bindings": {
            "input_candidate_sha256": EXPECTED_INPUT_CANDIDATE_SHA256,
            "owner_review_receipt_sha256": receipt["receipt_sha256"],
            "preflight_report_file_sha256": EXPECTED_PREFLIGHT_REPORT_FILE_SHA256,
            "preflight_verification_file_sha256": (
                EXPECTED_PREFLIGHT_VERIFICATION_FILE_SHA256
            ),
            "source_rows_sha256": EXPECTED_SOURCE_ROWS_SHA256,
            "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        },
        "checks": {
            "all_input_rows_replayed": True,
            "all_retained_rows_have_four_provenance_hashes": True,
            "benchmark_matches": machine_gate["benchmark_matches"],
            "dataset_sealed": False,
            "exact_duplicates": dedup["exact_duplicates"],
            "machine_rejected_records": machine_gate["machine_rejected_records"],
            "near_duplicates_at_0_82": dedup["near_duplicates_at_0_82"],
            "network_used": False,
            "owner_reviewed_rows_retained": len(reviewed_ids),
            "region_specific_emergency_number_rows": 0,
            "registry_changed": False,
            "sft_training_started": False,
            "visible_messages_changed": 0,
        },
        "input": {
            "candidate_records": len(input_rows),
            "source_rows": len(source_rows),
            "train_supervised_tokens": 251_756,
            "validation_supervised_tokens": 4_995,
        },
        "observed_at": plan["observed_at"],
        "output": {
            "candidate_file_sha256": candidate_sha256,
            "manifest_file_sha256": manifest_file_sha256,
            "manifest_internal_sha256": manifest["manifest_sha256"],
            "private_write_bytes": private_write_bytes,
            "records": len(selected),
            "train_records": len(selected_train),
            "train_serialized_tokens": serialized_train_tokens,
            "train_supervised_tokens": train_tokens,
            "validation_records": len(validation_rows),
            "validation_serialized_tokens": serialized_validation_tokens,
            "validation_supervised_tokens": validation_tokens,
        },
        "owner_review": owner_review,
        "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
        "production_eligible": False,
        "raw_training_text_in_report": False,
        "result_preservation": {
            "input_candidate_modified": False,
            "owner_review_visible_rows_preserved": True,
            "predecessor_registry_modified": False,
            "validation_rows_preserved": len(validation_rows) == 17,
        },
        "schema_version": REPORT_SCHEMA_VERSION,
        "selection": {
            "algorithm": "owner_review_lock_then_sha256_priority_greedy_fit_v0.1",
            "fit_skipped_train_records": fit_skipped,
            "owner_reviewed_train_records": len(reviewed_train),
            "owner_reviewed_train_supervised_tokens": reviewed_train_tokens,
            "train_ceiling_supervised_tokens": ceiling,
        },
        "status": REPORT_STATUS,
    }
    report["report_sha256"] = sha256_value(report)
    validate_report(report)
    return candidate_payload, manifest, report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported BanglaBridge hygiene-repair report")
    if report.get("status") != REPORT_STATUS:
        raise ValueError("BanglaBridge hygiene repair did not complete")
    _validate_internal_seal(report, "report_sha256", "BanglaBridge hygiene repair")
    checks = report.get("checks")
    output = report.get("output")
    if not isinstance(checks, Mapping) or not isinstance(output, Mapping):
        raise ValueError("BanglaBridge hygiene-repair checks are missing")
    if (
        checks.get("machine_rejected_records") != 0
        or checks.get("benchmark_matches") != 0
        or checks.get("exact_duplicates") != 0
        or checks.get("near_duplicates_at_0_82") != 0
        or checks.get("owner_reviewed_rows_retained") != 8
        or checks.get("visible_messages_changed") != 0
        or checks.get("registry_changed") is not False
        or checks.get("sft_training_started") is not False
        or output.get("train_supervised_tokens", 50_001) > 50_000
        or output.get("validation_records") != 17
    ):
        raise ValueError("BanglaBridge hygiene-repair gates changed")
