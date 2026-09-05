"""Truthful dataset-level owner spot-check provenance for Lumi AI v0.3 rows."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .posttraining_candidate_mix_preflight import (
    _load_json,
    _require_sha256,
    _verify_internal_sha,
    file_sha256,
    sha256_value,
)
from .posttraining_data import filter_sft_records, load_sft_policy, validate_message_sequence
from .posttraining_first_party import canonical_json_bytes
from .posttraining_format import load_chat_template, verify_chat_tokenizer
from .posttraining_sources import load_sft_source_registry
from .tokenizer_gate import encode_untrusted_text

PLAN_SCHEMA_VERSION = "lumi.sft-ai-v3-provenance-materialization-plan.v0.1"
MANIFEST_SCHEMA_VERSION = "lumi.sft-ai-v3-provenance-manifest.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-ai-v3-provenance-materialization-report.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-ai-v3-provenance-materialization-verification.v0.1"
)

EXPECTED_AUTHORIZATION = {
    "candidate_metadata_repair_authorized": True,
    "candidate_message_modification_authorized": False,
    "candidate_source_or_split_modification_authorized": False,
    "dataset_acquisition_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_promotion_authorized": False,
    "production_sft_authorized": False,
    "purpose": "ai_v0_3_dataset_level_owner_spot_check_provenance_only",
    "source_registry_change_authorized": False,
    "unrelated_data_access_authorized": False,
}
EXPECTED_LIMITS = {
    "candidate_records": 320,
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 16 * 1024 * 1024,
    "maximum_runtime_minutes": 60,
    "owner_spot_check_records": 32,
}
EXPECTED_OUTPUT = {
    "candidate_path": (
        "../lumi-artifacts/4b-sft-ai-v3-provenance-v0.1/"
        "7961c081dc57/attempt-001/ai-v3-candidate.jsonl"
    ),
    "manifest_path": (
        "../lumi-artifacts/4b-sft-ai-v3-provenance-v0.1/"
        "7961c081dc57/attempt-001/provenance-manifest.json"
    ),
    "production_dataset_promotions": 0,
    "raw_training_text_in_repo_evidence": False,
    "report_path": "evidence/lumi-4b-sft-ai-v3-provenance-materialization-v0.1.json",
    "verification_path": (
        "evidence/lumi-4b-sft-ai-v3-provenance-materialization-verification-v0.1.json"
    ),
}
EXPECTED_BINDINGS = {
    "candidate_mix_report",
    "candidate_mix_verification",
    "chat_template",
    "data_contract",
    "implementation",
    "owner_review_receipt",
    "owner_review_sample",
    "pilot_config",
    "source_candidate",
    "source_promotion",
    "source_promotion_verification",
    "source_registry",
    "test",
    "tokenizer_config",
    "tokenizer_json",
    "runner",
    "verifier",
}

SOURCE_ID = "lumi_ai_assisted_sft_v0_3"
SOURCE_REVISION = "7961c081dc57339613fd95961adea65d02d22348b8a69217cc0af3949fe6a7c6"
SOURCE_REGISTRY_SHA256 = (
    "d20c89ca34a284fc120a115d0ac9578c7e03ae3c915452d3f0ea144928e9a519"
)
EXPECTED_BY_SPLIT = {
    "train": {"records": 240, "supervised_tokens": 9_314},
    "validation": {"records": 80, "supervised_tokens": 2_138},
}
OWNER_REVIEW_CLAIM = "ai_generated_plus_owner_reviewed_not_human_written"
OWNER_REVIEW_SCOPE = "dataset_level_32_row_spot_check_not_individual_320_row_review"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(value)
    return rows


def _candidate_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def resolve_binding(
    repository_root: str | Path,
    binding: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    root = Path(repository_root).resolve()
    workspace = root.parent.resolve()
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"invalid AI v0.3 provenance binding path: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"AI v0.3 provenance binding escapes workspace: {name}")
    if not resolved.is_file():
        raise ValueError(f"AI v0.3 provenance binding is missing: {name}")
    expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
    if file_sha256(resolved) != expected:
        raise ValueError(f"AI v0.3 provenance binding changed: {name}")
    return resolved


def resolve_output(repository_root: str | Path, relative: str, *, name: str) -> Path:
    root = Path(repository_root).resolve()
    workspace = root.parent.resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"AI v0.3 provenance output escapes workspace: {name}")
    return resolved


def load_ai_v3_provenance_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "expected": {
            "by_split": EXPECTED_BY_SPLIT,
            "owner_review_claim": OWNER_REVIEW_CLAIM,
            "owner_review_scope": OWNER_REVIEW_SCOPE,
            "source_id": SOURCE_ID,
            "source_revision": SOURCE_REVISION,
        },
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-07",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_cpu_only_ai_v0_3_provenance_materialization",
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"AI v0.3 provenance plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("AI v0.3 provenance binding set changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid AI v0.3 provenance binding: {name}")
        resolve_binding(repository_root, binding, name=name)
    if bindings["source_registry"]["sha256"] != SOURCE_REGISTRY_SHA256:
        raise ValueError("AI v0.3 provenance registry binding changed")
    for name in ("candidate_path", "manifest_path", "report_path", "verification_path"):
        resolve_output(repository_root, plan["output"][name], name=name)
    return plan


def _validate_owner_review(
    receipt: Mapping[str, Any],
    sample_rows: Sequence[Mapping[str, Any]],
) -> set[str]:
    _verify_internal_sha(receipt, "receipt_sha256")
    expected = {
        "accepted_records": 32,
        "all_sampled_rows_accepted": True,
        "owner_review_claim": OWNER_REVIEW_CLAIM,
        "production_eligible": False,
        "rejected_records": 0,
        "review_scope": "sample",
        "sample_records": 32,
        "status": "owner_spot_check_passed_pilot_still_not_production_dataset",
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ValueError(f"AI v0.3 owner-review receipt changed: {field}")
    if bytes_sha256(canonical_json_bytes(list(sample_rows))) != receipt.get(
        "sample_sha256"
    ):
        raise ValueError("AI v0.3 owner-review sample fingerprint changed")
    sample_ids = [row.get("record_id") for row in sample_rows]
    if any(not isinstance(record_id, str) or not record_id for record_id in sample_ids):
        raise ValueError("AI v0.3 owner-review sample record IDs changed")
    if len(sample_ids) != 32 or len(set(sample_ids)) != 32:
        raise ValueError("AI v0.3 owner-review sample cardinality changed")
    return set(sample_ids)


def _measure_conversation(
    messages: Sequence[Mapping[str, str]],
    *,
    tokenizer: Any,
    template: Any,
) -> int:
    validate_message_sequence(messages)
    total_tokens = 2
    supervised_tokens = 1
    for message in messages:
        content_ids = encode_untrusted_text(tokenizer, message["content"])
        if not content_ids:
            raise ValueError("AI v0.3 message encoded to zero tokens")
        total_tokens += len(content_ids) + 2
        if message["role"] in template.loss_roles:
            supervised_tokens += len(content_ids) + 1
    if total_tokens > template.context_window:
        raise ValueError("AI v0.3 conversation exceeds frozen context")
    if supervised_tokens < 2:
        raise ValueError("AI v0.3 conversation has no supervised response")
    return supervised_tokens


def _validate_bound_evidence(resolved: Mapping[str, Path]) -> None:
    mix_report = _load_json(resolved["candidate_mix_report"])
    mix_verification = _load_json(resolved["candidate_mix_verification"])
    promotion = _load_json(resolved["source_promotion"])
    promotion_verification = _load_json(resolved["source_promotion_verification"])
    _verify_internal_sha(mix_report, "report_sha256")
    _verify_internal_sha(mix_verification, "verification_sha256")
    _verify_internal_sha(promotion, "receipt_sha256")
    _verify_internal_sha(promotion_verification, "verification_sha256")
    if mix_report.get("status") != (
        "verified_registry_v0_8_candidate_mix_not_ready_for_seal_or_gpu_sft"
    ):
        raise ValueError("AI v0.3 predecessor mix report changed")
    if mix_verification.get("status") != (
        "verified_registry_v0_8_candidate_mix_reaudit_nonproduction"
    ):
        raise ValueError("AI v0.3 predecessor mix verification changed")
    if promotion.get("source", {}).get("source_id") != SOURCE_ID:
        raise ValueError("AI v0.3 source promotion changed")
    if promotion_verification.get("status") != (
        "verified_ai_v0_3_candidate_registry_promotion_nonproduction"
    ):
        raise ValueError("AI v0.3 source promotion verification changed")


def build_ai_v3_provenance_outputs(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    from transformers import AutoTokenizer

    root = Path(repository_root).resolve()
    resolved = {
        name: resolve_binding(root, binding, name=name)
        for name, binding in plan["bindings"].items()
    }
    _validate_bound_evidence(resolved)
    owner_receipt = _load_json(resolved["owner_review_receipt"])
    owner_sample = _load_jsonl(resolved["owner_review_sample"])
    sample_ids = _validate_owner_review(owner_receipt, owner_sample)
    source_rows = _load_jsonl(resolved["source_candidate"])
    if len(source_rows) != 320:
        raise ValueError("AI v0.3 candidate row count changed")

    policy = load_sft_policy(resolved["data_contract"])
    registry = load_sft_source_registry(
        resolved["source_registry"],
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
        expected_sha256=SOURCE_REGISTRY_SHA256,
    )
    registry_source = registry.by_id.get(SOURCE_ID)
    if registry_source is None or registry_source.revision != SOURCE_REVISION:
        raise ValueError("AI v0.3 source registry binding changed")
    if "per_record_owner_review_provenance_not_materialized" not in (
        registry_source.production_blockers
    ):
        raise ValueError("AI v0.3 provenance blocker unexpectedly disappeared")
    template = load_chat_template(resolved["chat_template"])
    if resolved["tokenizer_json"].parent != resolved["tokenizer_config"].parent:
        raise ValueError("AI v0.3 frozen tokenizer files are not colocated")
    tokenizer = AutoTokenizer.from_pretrained(
        resolved["tokenizer_json"].parent,
        local_files_only=True,
    )
    verify_chat_tokenizer(tokenizer, template)

    seen_ids: set[str] = set()
    output_rows: list[dict[str, Any]] = []
    split_records: Counter[str] = Counter()
    split_tokens: Counter[str] = Counter()
    message_bindings: list[dict[str, str]] = []
    sampled_rows = 0
    for row in source_rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in seen_ids:
            raise ValueError("AI v0.3 record IDs changed")
        seen_ids.add(record_id)
        source = row.get("source")
        provenance = row.get("provenance")
        messages = row.get("messages")
        split = row.get("split")
        if not isinstance(source, Mapping):
            raise ValueError("AI v0.3 source metadata changed")
        if source.get("source_id") != SOURCE_ID or source.get("revision") != SOURCE_REVISION:
            raise ValueError("AI v0.3 source binding changed")
        if split not in {"train", "validation"}:
            raise ValueError("AI v0.3 split changed")
        if not isinstance(provenance, Mapping):
            raise ValueError("AI v0.3 provenance changed")
        if provenance.get("human_review_status") != "owner_spot_check_pending":
            raise ValueError("AI v0.3 predecessor owner-review state changed")
        if not isinstance(messages, list):
            raise ValueError("AI v0.3 messages changed")

        sampled = record_id in sample_ids
        sampled_rows += int(sampled)
        rematerialized = copy.deepcopy(row)
        repaired_provenance = dict(rematerialized["provenance"])
        prior_transform = _require_sha256(
            repaired_provenance.get("transform_sha256"),
            "ai_v3.provenance.transform_sha256",
        )
        transform_binding = {
            "operation": "bind_dataset_level_owner_spot_check_v0.1",
            "owner_review_receipt_file_sha256": file_sha256(
                resolved["owner_review_receipt"]
            ),
            "owner_review_receipt_sha256": owner_receipt["receipt_sha256"],
            "owner_review_sampled_record": sampled,
            "owner_review_scope": OWNER_REVIEW_SCOPE,
            "prior_transform_sha256": prior_transform,
            "record_id": record_id,
        }
        repaired_provenance.update(
            {
                "human_review_status": "owner_spot_check_passed",
                "owner_review_claim": OWNER_REVIEW_CLAIM,
                "owner_review_receipt_file_sha256": file_sha256(
                    resolved["owner_review_receipt"]
                ),
                "owner_review_receipt_sha256": owner_receipt["receipt_sha256"],
                "owner_review_sample_records": owner_receipt["sample_records"],
                "owner_review_sampled_record": sampled,
                "owner_review_scope": OWNER_REVIEW_SCOPE,
                "transform_sha256": sha256_value(transform_binding),
            }
        )
        rematerialized["provenance"] = repaired_provenance
        for field in rematerialized:
            if field != "provenance" and rematerialized[field] != row[field]:
                raise ValueError(f"AI v0.3 non-provenance field changed: {field}")
        supervised_tokens = _measure_conversation(
            messages,
            tokenizer=tokenizer,
            template=template,
        )
        split_records[split] += 1
        split_tokens[split] += supervised_tokens
        message_bindings.append(
            {
                "content_sha256": str(provenance["content_sha256"]),
                "messages_sha256": sha256_value(messages),
                "record_id": record_id,
            }
        )
        output_rows.append(rematerialized)

    if sampled_rows != 32 or not sample_ids.issubset(seen_ids):
        raise ValueError("AI v0.3 reviewed sample is not an exact candidate subset")
    by_split = {
        split: {
            "records": split_records[split],
            "supervised_tokens": split_tokens[split],
        }
        for split in ("train", "validation")
    }
    if by_split != EXPECTED_BY_SPLIT:
        raise ValueError("AI v0.3 frozen-token totals changed")
    accepted, generic_report = filter_sft_records(
        output_rows,
        policy=policy,
        source_registry=registry,
    )
    if len(accepted) != 320 or generic_report["rejected"] != 0:
        raise ValueError(
            "AI v0.3 repaired candidate failed generic validation: "
            f"{generic_report['rejection_counts']}"
        )

    candidate_bytes = _candidate_bytes(output_rows)
    if len(candidate_bytes) > plan["limits"]["maximum_private_write_bytes"]:
        raise ValueError("AI v0.3 private candidate exceeds write ceiling")
    candidate_sha256 = bytes_sha256(candidate_bytes)
    manifest: dict[str, Any] = {
        "artifacts": {
            "candidate": {
                "bytes": len(candidate_bytes),
                "path": plan["output"]["candidate_path"],
                "sha256": candidate_sha256,
            }
        },
        "bindings": {
            "owner_review_receipt_file_sha256": file_sha256(
                resolved["owner_review_receipt"]
            ),
            "owner_review_receipt_sha256": owner_receipt["receipt_sha256"],
            "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
            "source_candidate_sha256": file_sha256(resolved["source_candidate"]),
            "source_registry_sha256": SOURCE_REGISTRY_SHA256,
        },
        "by_split": by_split,
        "message_and_content_binding_sha256": sha256_value(message_bindings),
        "owner_review": {
            "claim": OWNER_REVIEW_CLAIM,
            "individually_reviewed_all_rows_claimed": False,
            "sample_records": 32,
            "sampled_record_ids_sha256": sha256_value(sorted(sample_ids)),
            "scope": OWNER_REVIEW_SCOPE,
            "unsampled_records": 288,
        },
        "records": 320,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_id": SOURCE_ID,
        "source_revision": SOURCE_REVISION,
        "status": "private_ai_v0_3_provenance_candidate_materialized_nonproduction",
    }
    manifest["manifest_sha256"] = sha256_value(manifest)
    manifest_file_sha256 = bytes_sha256(json_file_bytes(manifest))
    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "bindings": {
            "candidate_file_sha256": candidate_sha256,
            "manifest_file_sha256": manifest_file_sha256,
            "manifest_internal_sha256": manifest["manifest_sha256"],
            "owner_review_receipt_file_sha256": file_sha256(
                resolved["owner_review_receipt"]
            ),
            "owner_review_receipt_sha256": owner_receipt["receipt_sha256"],
            "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
            "source_candidate_sha256": file_sha256(resolved["source_candidate"]),
            "source_registry_sha256": SOURCE_REGISTRY_SHA256,
        },
        "generic_validation": {
            "accepted": generic_report["accepted"],
            "accepted_dataset_sha256": generic_report["accepted_dataset_sha256"],
            "raw_text_retained": generic_report["raw_text_retained"],
            "rejected": generic_report["rejected"],
            "rejection_counts": generic_report["rejection_counts"],
            "report_sha256": generic_report["report_sha256"],
        },
        "observed_at": plan["observed_at"],
        "production_dataset_promotions": 0,
        "production_eligible": False,
        "provenance_materialization": {
            "by_split": by_split,
            "candidate_file_sha256": candidate_sha256,
            "individually_reviewed_all_rows_claimed": False,
            "message_and_content_binding_sha256": sha256_value(message_bindings),
            "records": 320,
            "rows_with_owner_review_receipt_binding": 320,
            "sampled_records": sampled_rows,
            "source_revision_unchanged": True,
            "unsampled_records": 320 - sampled_rows,
        },
        "raw_training_text_retained": False,
        "remaining_blockers": [
            "immutable_source_registry_successor_not_yet_bound_to_repaired_candidate",
            "accepted_train_supervised_tokens_below_10m_floor",
            "task_language_and_source_balance_not_closed",
            "within_split_near_duplicate_canonical_selection_not_materialized",
            "active_production_data_contract_not_updated",
            "dataset_seal_cpu_canary_and_sft_not_authorized",
        ],
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_registry_changed": False,
        "status": "verified_ai_v0_3_owner_spot_check_provenance_materialized_nonproduction",
    }
    report["report_sha256"] = sha256_value(report)
    validate_ai_v3_provenance_report(report)
    return candidate_bytes, manifest, report


def validate_ai_v3_provenance_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported AI v0.3 provenance report schema")
    _verify_internal_sha(report, "report_sha256")
    if report.get("production_eligible") is not False:
        raise ValueError("AI v0.3 provenance output cannot be production eligible")
    if report.get("production_dataset_promotions") != 0:
        raise ValueError("AI v0.3 provenance output cannot promote a dataset")
    if report.get("source_registry_changed") is not False:
        raise ValueError("AI v0.3 provenance output cannot change the registry")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("AI v0.3 provenance report must remain raw-text-free")
    metrics = report.get("provenance_materialization")
    if not isinstance(metrics, Mapping):
        raise ValueError("AI v0.3 provenance metrics are missing")
    expected = {
        "by_split": EXPECTED_BY_SPLIT,
        "individually_reviewed_all_rows_claimed": False,
        "records": 320,
        "rows_with_owner_review_receipt_binding": 320,
        "sampled_records": 32,
        "source_revision_unchanged": True,
        "unsampled_records": 288,
    }
    for field, value in expected.items():
        if metrics.get(field) != value:
            raise ValueError(f"AI v0.3 provenance report changed: {field}")
    generic = report.get("generic_validation")
    if not isinstance(generic, Mapping):
        raise ValueError("AI v0.3 generic validation is missing")
    if generic.get("accepted") != 320 or generic.get("rejected") != 0:
        raise ValueError("AI v0.3 repaired candidate validation changed")
    if generic.get("rejection_counts") != {}:
        raise ValueError("AI v0.3 repaired candidate has rejection reasons")


def build_ai_v3_provenance_verification(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
) -> dict[str, Any]:
    validate_ai_v3_provenance_report(report)
    verification: dict[str, Any] = {
        "checks": {
            "all_320_rows_bind_exact_owner_receipt": True,
            "all_320_rows_pass_generic_schema_and_registry_validation": True,
            "candidate_messages_sources_splits_and_content_hashes_unchanged": True,
            "dataset_level_32_row_spot_check_not_misrepresented_as_320_row_review": True,
            "frozen_train_and_validation_token_counts_unchanged": True,
            "no_network_api_gpu_registry_seal_sft_or_production_promotion": True,
            "original_candidate_unchanged": True,
            "private_candidate_and_manifest_reconstructed": True,
            "raw_training_text_absent_from_repo_evidence": True,
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "report_file_sha256": _require_sha256(
            report_file_sha256, "report_file_sha256"
        ),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_ai_v0_3_provenance_materialization_nonproduction",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification
