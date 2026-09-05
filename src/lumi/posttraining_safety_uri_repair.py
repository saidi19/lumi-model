"""Bounded first-party URI repair for the reviewed Lumi safety/tool candidate."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .data_quality import BenchmarkRegistry
from .posttraining_data import (
    filter_sft_records,
    load_sft_policy,
    validate_message_sequence,
)
from .posttraining_format import load_chat_template, verify_chat_tokenizer
from .posttraining_sources import load_sft_source_registry
from .tokenizer_gate import encode_untrusted_text

PLAN_SCHEMA_VERSION = "lumi.sft-safety-uri-repair-plan.v0.1"
MANIFEST_SCHEMA_VERSION = "lumi.sft-safety-uri-repair-manifest.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-safety-uri-repair-report.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-safety-uri-repair-verification.v0.1"

SAFETY_SOURCE_ID = "lumi_safety_tool_sft_v0_1"
SAFETY_DATASET_ID = "Afnan-Technologies-Inc/lumi-safety-tool-sft-v0.1"
EXPECTED_REGISTRY_SHA256 = (
    "8cd8cc0b20ccd7da2f9083b0903178b98f59e3279c6777ebf8fee4866069d573"
)
EXPECTED_PREDECESSOR_SHA256 = (
    "ffc4a91613dd228282d2bc8ef2b93a6d508e31fa90e2925d91de837c1b4bd0e1"
)
EXPECTED_SAFETY_TOKENS = {
    "train": {"records": 240, "supervised_tokens": 18_636},
    "validation": {"records": 80, "supervised_tokens": 6_388},
}
EXPECTED_AUTHORIZATION = {
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_promotion_authorized": False,
    "production_sft_authorized": False,
    "purpose": "safety_tool_deterministic_first_party_uri_and_validation_only",
    "source_uri_repair_authorized": True,
    "unrelated_data_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_network_bytes": 0,
    "maximum_runtime_minutes": 60,
    "safety_candidate_records": 320,
    "source_registry_entries_changed": 0,
}
EXPECTED_OUTPUT = {
    "artifact_manifest_path": (
        "../lumi-artifacts/4b-sft-safety-uri-repair-v0.1/"
        "8cd8cc0b20cc/attempt-001/safety-tool-uri-repair-manifest.json"
    ),
    "private_safety_candidate_path": (
        "../lumi-artifacts/4b-sft-safety-uri-repair-v0.1/"
        "8cd8cc0b20cc/attempt-001/safety-tool-candidate.jsonl"
    ),
    "production_dataset_promotions": 0,
    "raw_training_text_in_repo_evidence": False,
    "report_path": "evidence/lumi-4b-sft-safety-uri-repair-v0.1.json",
    "verification_path": (
        "evidence/lumi-4b-sft-safety-uri-repair-verification-v0.1.json"
    ),
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RECORD_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_file_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


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


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256")
    return value


def _verify_internal_sha(value: Mapping[str, Any], field: str) -> str:
    seal = _require_sha256(value.get(field), field)
    unsigned = {key: item for key, item in value.items() if key != field}
    if sha256_value(unsigned) != seal:
        raise ValueError(f"{field} changed")
    return seal


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
        raise ValueError(f"invalid safety URI repair binding path: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"safety URI repair binding escapes workspace: {name}")
    if not resolved.is_file():
        raise ValueError(f"safety URI repair binding is missing: {name}")
    expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
    if file_sha256(resolved) != expected:
        raise ValueError(f"safety URI repair binding changed: {name}")
    return resolved


def resolve_output(repository_root: str | Path, relative: str, *, name: str) -> Path:
    root = Path(repository_root).resolve()
    workspace = root.parent.resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"safety URI repair output escapes workspace: {name}")
    return resolved


def load_safety_uri_repair_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    plan = _load_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-06",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_bounded_cpu_only_safety_uri_repair",
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"safety URI repair plan changed: {field}")
    bindings = plan.get("bindings")
    required = {
        "benchmark_registry",
        "chat_template",
        "data_contract",
        "implementation",
        "posttraining_data_implementation",
        "predecessor_candidate",
        "predecessor_hygiene_manifest",
        "predecessor_hygiene_report",
        "predecessor_hygiene_verification",
        "runner",
        "safety_owner_review_receipt",
        "source_registry",
        "source_registry_implementation",
        "test",
        "tokenizer_config",
        "tokenizer_json",
        "verifier",
    }
    if not isinstance(bindings, Mapping) or set(bindings) != required:
        raise ValueError("safety URI repair binding set changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid safety URI repair binding: {name}")
        resolve_binding(repository_root, binding, name=name)
    if bindings["predecessor_candidate"]["sha256"] != EXPECTED_PREDECESSOR_SHA256:
        raise ValueError("safety URI predecessor binding changed")
    if bindings["source_registry"]["sha256"] != EXPECTED_REGISTRY_SHA256:
        raise ValueError("source registry binding changed")
    for name, relative in EXPECTED_OUTPUT.items():
        if name.endswith("_path") and isinstance(relative, str):
            resolve_output(repository_root, relative, name=name)
    return plan


def _validate_predecessor_chain(
    *,
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> None:
    _verify_internal_sha(manifest, "manifest_sha256")
    _verify_internal_sha(report, "report_sha256")
    _verify_internal_sha(verification, "verification_sha256")
    if manifest.get("status") != "safety_metadata_rematerialized_nonproduction":
        raise ValueError("predecessor hygiene manifest status changed")
    if manifest.get("records") != 320:
        raise ValueError("predecessor hygiene manifest row count changed")
    if manifest.get("output_candidate_file_sha256") != EXPECTED_PREDECESSOR_SHA256:
        raise ValueError("predecessor hygiene manifest candidate binding changed")
    if report.get("status") != "verified_scoped_candidate_hygiene_repair_nonproduction":
        raise ValueError("predecessor hygiene report status changed")
    if report.get("bindings", {}).get(
        "private_safety_candidate_file_sha256"
    ) != EXPECTED_PREDECESSOR_SHA256:
        raise ValueError("predecessor hygiene report candidate binding changed")
    if "safety_source_uri_missing_from_320_rows_outside_this_scoped_repair" not in report.get(
        "remaining_blockers", []
    ):
        raise ValueError("predecessor URI blocker changed")
    if verification.get("status") != "verified_scoped_candidate_hygiene_outputs_nonproduction":
        raise ValueError("predecessor hygiene verification status changed")


def _validate_owner_receipt(receipt: Mapping[str, Any]) -> None:
    _verify_internal_sha(receipt, "receipt_sha256")
    if receipt.get("receipt_sha256") != (
        "369a3110a46de6189a7311eee946daa7092cf04cb9676dfb8ea67cfb41e838fb"
    ):
        raise ValueError("safety owner-review receipt changed")
    if receipt.get("status") != (
        "owner_spot_check_passed_source_registry_promotion_still_separate"
    ):
        raise ValueError("safety owner-review status changed")


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
            raise ValueError("safety message encoded to zero tokens")
        total_tokens += len(content_ids) + 2
        if message["role"] in template.loss_roles:
            supervised_tokens += len(content_ids) + 1
    if total_tokens > template.context_window:
        raise ValueError("safety conversation exceeds the frozen context window")
    if supervised_tokens < 2:
        raise ValueError("safety conversation has no supervised response")
    return supervised_tokens


def _deterministic_uri(record_id: str) -> str:
    if _RECORD_ID.fullmatch(record_id) is None:
        raise ValueError("safety record ID is not URI-safe")
    return f"private://{SAFETY_DATASET_ID}/{record_id}"


def _repair_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], bytes, dict[str, Any]]:
    if len(rows) != 320:
        raise ValueError("safety candidate must contain exactly 320 rows")
    output_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_uris: set[str] = set()
    content_bindings: list[dict[str, str]] = []
    transform_bindings: list[dict[str, str]] = []

    for row in rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in seen_ids:
            raise ValueError("safety record IDs changed")
        seen_ids.add(record_id)
        if row.get("split") not in {"train", "validation"}:
            raise ValueError("safety top-level split changed")
        source = row.get("source")
        provenance = row.get("provenance")
        messages = row.get("messages")
        if not isinstance(source, Mapping):
            raise ValueError("safety source metadata changed")
        if source.get("source_id") != SAFETY_SOURCE_ID:
            raise ValueError("safety source ID changed")
        if source.get("dataset_id") != SAFETY_DATASET_ID:
            raise ValueError("safety dataset ID changed")
        if source.get("document_id") != record_id:
            raise ValueError("safety document binding changed")
        if source.get("split") != row["split"]:
            raise ValueError("safety split binding changed")
        if "uri" in source:
            raise ValueError("predecessor safety row unexpectedly has a source URI")
        if not isinstance(provenance, Mapping):
            raise ValueError("safety provenance changed")
        if provenance.get("human_review_status") != "owner_spot_check_passed":
            raise ValueError("safety owner-review state changed")
        if not isinstance(messages, list):
            raise ValueError("safety messages changed")

        prior_transform = _require_sha256(
            provenance.get("transform_sha256"),
            "safety.provenance.transform_sha256",
        )
        uri = _deterministic_uri(record_id)
        if uri in seen_uris:
            raise ValueError("duplicate safety source URI")
        seen_uris.add(uri)
        repaired = copy.deepcopy(row)
        repaired["source"]["uri"] = uri
        transform_binding = {
            "operation": "add_deterministic_first_party_source_uri_v0.1",
            "prior_transform_sha256": prior_transform,
            "source_uri": uri,
        }
        repaired["provenance"]["transform_sha256"] = sha256_value(transform_binding)

        comparison = copy.deepcopy(repaired)
        del comparison["source"]["uri"]
        comparison["provenance"]["transform_sha256"] = prior_transform
        if comparison != row:
            raise ValueError("URI repair changed content or unrelated metadata")
        content_sha256 = _require_sha256(
            provenance.get("content_sha256"),
            "safety.provenance.content_sha256",
        )
        content_bindings.append(
            {
                "content_sha256": content_sha256,
                "messages_sha256": sha256_value(messages),
                "record_id": record_id,
            }
        )
        transform_bindings.append(
            {
                "new_transform_sha256": repaired["provenance"]["transform_sha256"],
                "prior_transform_sha256": prior_transform,
                "record_id": record_id,
                "source_uri_sha256": bytes_sha256(uri.encode("utf-8")),
            }
        )
        output_rows.append(repaired)

    candidate_bytes = b"".join(
        canonical_json_bytes(row) + b"\n" for row in output_rows
    )
    metrics = {
        "candidate_file_sha256": bytes_sha256(candidate_bytes),
        "content_and_messages_binding_sha256": sha256_value(content_bindings),
        "records": len(output_rows),
        "records_with_deterministic_source_uri": len(seen_uris),
        "records_with_only_authorized_metadata_delta": len(output_rows),
        "transform_chain_binding_sha256": sha256_value(transform_bindings),
        "unique_source_uris": len(seen_uris),
        "uri_set_sha256": sha256_value(sorted(seen_uris)),
    }
    return output_rows, candidate_bytes, metrics


def rebuild_safety_uri_repair_outputs(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    root = Path(repository_root).resolve()
    resolved = {
        name: resolve_binding(root, binding, name=name)
        for name, binding in plan["bindings"].items()
    }
    predecessor_manifest = _load_json(resolved["predecessor_hygiene_manifest"])
    predecessor_report = _load_json(resolved["predecessor_hygiene_report"])
    predecessor_verification = _load_json(
        resolved["predecessor_hygiene_verification"]
    )
    _validate_predecessor_chain(
        manifest=predecessor_manifest,
        report=predecessor_report,
        verification=predecessor_verification,
    )
    owner_receipt = _load_json(resolved["safety_owner_review_receipt"])
    _validate_owner_receipt(owner_receipt)

    policy = load_sft_policy(resolved["data_contract"])
    benchmark_registry = BenchmarkRegistry(policy.benchmark_decontamination, root)
    if benchmark_registry.sha256 != plan["bindings"]["benchmark_registry"]["sha256"]:
        raise ValueError("benchmark registry binding changed")
    source_registry = load_sft_source_registry(
        resolved["source_registry"],
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
        expected_sha256=EXPECTED_REGISTRY_SHA256,
    )

    rows, candidate_bytes, repair_metrics = _repair_rows(
        _load_jsonl(resolved["predecessor_candidate"])
    )
    accepted, generic_report = filter_sft_records(
        rows,
        policy=policy,
        benchmark_registry=benchmark_registry,
        source_registry=source_registry,
    )
    if len(accepted) != 320 or generic_report["rejected"] != 0:
        raise ValueError(
            "generic SFT validation rejected URI-repaired safety records: "
            f"{generic_report['rejection_counts']}"
        )
    if generic_report["rejection_counts"]:
        raise ValueError("generic SFT validation emitted rejection reasons")

    template = load_chat_template(resolved["chat_template"])
    if resolved["tokenizer_json"].parent != resolved["tokenizer_config"].parent:
        raise ValueError("frozen tokenizer files are not colocated")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(
        resolved["tokenizer_json"].parent,
        local_files_only=True,
    )
    verify_chat_tokenizer(tokenizer, template)
    split_records: Counter[str] = Counter()
    split_tokens: Counter[str] = Counter()
    for row in rows:
        split = row["split"]
        split_records[split] += 1
        split_tokens[split] += _measure_conversation(
            row["messages"], tokenizer=tokenizer, template=template
        )
    by_split = {
        split: {
            "records": split_records[split],
            "supervised_tokens": split_tokens[split],
        }
        for split in ("train", "validation")
    }
    if by_split != EXPECTED_SAFETY_TOKENS:
        raise ValueError("safety frozen-token totals changed")

    manifest: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "benchmark_registry_sha256": benchmark_registry.sha256,
        "content_and_messages_binding_sha256": repair_metrics[
            "content_and_messages_binding_sha256"
        ],
        "generic_validation": generic_report,
        "input_candidate_file_sha256": EXPECTED_PREDECESSOR_SHA256,
        "output_candidate_file_sha256": repair_metrics["candidate_file_sha256"],
        "raw_training_text_retained": False,
        "records": repair_metrics["records"],
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_registry_sha256": source_registry.registry_sha256,
        "split_metrics": by_split,
        "status": "safety_source_uris_repaired_and_validated_nonproduction",
        "transform_chain_binding_sha256": repair_metrics[
            "transform_chain_binding_sha256"
        ],
        "uri_set_sha256": repair_metrics["uri_set_sha256"],
    }
    manifest["manifest_sha256"] = sha256_value(manifest)
    manifest_bytes = json_file_bytes(manifest)

    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "bindings": {
            "artifact_manifest_file_sha256": bytes_sha256(manifest_bytes),
            "benchmark_registry_sha256": benchmark_registry.sha256,
            "generic_validation_report_sha256": generic_report["report_sha256"],
            "output_candidate_file_sha256": repair_metrics["candidate_file_sha256"],
            "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
            "predecessor_candidate_file_sha256": EXPECTED_PREDECESSOR_SHA256,
            "source_registry_sha256": source_registry.registry_sha256,
        },
        "frozen_token_verification": {
            "by_split": by_split,
            "chat_template_sha256": plan["bindings"]["chat_template"]["sha256"],
            "context_window": template.context_window,
            "tokenizer_json_sha256": plan["bindings"]["tokenizer_json"]["sha256"],
        },
        "generic_sft_validation": {
            "accepted": generic_report["accepted"],
            "benchmark_contamination_rejections": sum(
                count
                for reason, count in generic_report["rejection_counts"].items()
                if reason.startswith("benchmark_contamination:")
            ),
            "exact_normalized_duplicates": generic_report["rejection_counts"].get(
                "duplicate_normalized_conversation", 0
            ),
            "raw_text_retained": generic_report["raw_text_retained"],
            "rejected": generic_report["rejected"],
            "rejection_counts": generic_report["rejection_counts"],
            "report_sha256": generic_report["report_sha256"],
            "source_registry_rejections": sum(
                count
                for reason, count in generic_report["rejection_counts"].items()
                if reason.startswith("source_registry")
            ),
            "total": generic_report["total"],
        },
        "metadata_repair": repair_metrics,
        "observed_at": plan["observed_at"],
        "production_dataset_promotions": 0,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "remaining_blockers": [
            "accepted_train_supervised_tokens_below_10m_floor",
            "task_language_and_source_balance_not_closed",
            "full_mix_cross_source_near_deduplication_not_complete",
            "active_production_data_contract_not_updated",
            "dataset_seal_cpu_canary_and_sft_not_authorized",
        ],
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "verified_safety_source_uri_repair_nonproduction",
    }
    report["report_sha256"] = sha256_value(report)
    validate_safety_uri_repair_report(report)
    return {
        "artifact_manifest_bytes": manifest_bytes,
        "private_safety_candidate_bytes": candidate_bytes,
        "report": report,
    }


def validate_safety_uri_repair_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported safety URI repair report schema")
    _verify_internal_sha(report, "report_sha256")
    if report.get("production_eligible") is not False:
        raise ValueError("safety URI repair cannot be production eligible")
    if report.get("production_dataset_promotions") != 0:
        raise ValueError("safety URI repair cannot promote a production dataset")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("safety URI repair report must remain raw-text-free")
    metadata = report.get("metadata_repair")
    if not isinstance(metadata, Mapping):
        raise ValueError("safety URI repair metrics are missing")
    for field in (
        "records",
        "records_with_deterministic_source_uri",
        "records_with_only_authorized_metadata_delta",
        "unique_source_uris",
    ):
        if metadata.get(field) != 320:
            raise ValueError(f"safety URI repair metric changed: {field}")
    generic = report.get("generic_sft_validation")
    if not isinstance(generic, Mapping):
        raise ValueError("generic SFT validation summary is missing")
    if generic.get("accepted") != 320 or generic.get("rejected") != 0:
        raise ValueError("not every safety record passed generic SFT validation")
    if generic.get("rejection_counts") != {}:
        raise ValueError("generic SFT validation has rejection reasons")


def build_safety_uri_repair_verification(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
) -> dict[str, Any]:
    validate_safety_uri_repair_report(report)
    verification: dict[str, Any] = {
        "checks": {
            "all_320_rows_have_unique_deterministic_first_party_uris": True,
            "all_320_rows_pass_generic_sft_schema_and_registry_validation": True,
            "benchmark_decontamination_passed": True,
            "frozen_4k_token_counts_unchanged": True,
            "messages_content_and_unrelated_metadata_unchanged": True,
            "no_network_api_gpu_seal_sft_or_production_promotion": True,
            "predecessor_candidate_and_registry_unchanged": True,
            "raw_training_text_absent_from_repo_evidence": True,
            "transform_chain_rebound_for_uri_metadata_delta": True,
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "report_file_sha256": _require_sha256(
            report_file_sha256, "report_file_sha256"
        ),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_safety_source_uri_repair_outputs_nonproduction",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification
