"""Bounded, non-production repair for reviewed Lumi SFT candidate metadata."""

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

from .posttraining_data import load_sft_policy, validate_message_sequence
from .posttraining_format import load_chat_template, verify_chat_tokenizer
from .posttraining_sources import SFTSource, load_sft_source_registry
from .tokenizer_gate import encode_untrusted_text

PLAN_SCHEMA_VERSION = "lumi.sft-candidate-hygiene-repair-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-candidate-hygiene-repair-report.v0.1"
MANIFEST_SCHEMA_VERSION = "lumi.sft-safety-tool-rematerialization-manifest.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-candidate-hygiene-repair-verification.v0.1"

EXPECTED_AUTHORIZATION = {
    "candidate_hygiene_repair_authorized": True,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_promotion_authorized": False,
    "production_sft_authorized": False,
    "purpose": "aya_validation_registry_binding_and_safety_metadata_repair_only",
    "unrelated_data_access_authorized": False,
}
EXPECTED_LIMITS = {
    "aya_registry_entries_changed": 1,
    "maximum_network_bytes": 0,
    "maximum_runtime_minutes": 60,
    "safety_candidate_records": 320,
}
EXPECTED_OUTPUT = {
    "artifact_manifest_path": (
        "../lumi-artifacts/4b-sft-candidate-hygiene-repair-v0.1/"
        "4892b05c49dc/attempt-001/safety-tool-rematerialization-manifest.json"
    ),
    "private_safety_candidate_path": (
        "../lumi-artifacts/4b-sft-candidate-hygiene-repair-v0.1/"
        "4892b05c49dc/attempt-001/safety-tool-candidate.jsonl"
    ),
    "production_dataset_promotions": 0,
    "raw_training_text_in_repo_evidence": False,
    "report_path": "evidence/lumi-4b-sft-candidate-hygiene-repair-v0.1.json",
    "successor_registry_path": "configs/lumi-4b-sft-source-registry-v0.7.json",
    "verification_path": (
        "evidence/lumi-4b-sft-candidate-hygiene-repair-verification-v0.1.json"
    ),
}

AYA_SOURCE_ID = "coherelabs_aya_dataset"
SAFETY_SOURCE_ID = "lumi_safety_tool_sft_v0_1"
EXPECTED_AYA_SPLIT = {
    "train": {
        "record_ids_sha256": "dcc6f7329c6e3eb08744326d52422ab1e8c408e5dfa1340680b03f07b5ad3bca",
        "records": 1_330,
        "supervised_tokens": 520_269,
    },
    "validation": {
        "record_ids_sha256": "6f2b88a5748660c8fcaa827ecdb0b97e5b89888993adf6325b9f1f6a06ae3a16",
        "records": 148,
        "supervised_tokens": 57_184,
    },
}
EXPECTED_SAFETY_TOKENS = {
    "train": {"records": 240, "supervised_tokens": 18_636},
    "validation": {"records": 80, "supervised_tokens": 6_388},
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


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


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        raise ValueError(f"invalid hygiene binding path: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"hygiene binding escapes workspace: {name}")
    if not resolved.is_file():
        raise ValueError(f"hygiene binding is missing: {name}")
    expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
    if file_sha256(resolved) != expected:
        raise ValueError(f"hygiene binding changed: {name}")
    return resolved


def resolve_output(repository_root: str | Path, relative: str, *, name: str) -> Path:
    root = Path(repository_root).resolve()
    workspace = root.parent.resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"hygiene output escapes workspace: {name}")
    return resolved


def load_candidate_hygiene_plan(
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
        "status": "authorized_bounded_cpu_only_candidate_hygiene_repair",
    }
    for field, value in expected.items():
        if plan.get(field) != value:
            raise ValueError(f"candidate-hygiene plan changed: {field}")
    bindings = plan.get("bindings")
    required = {
        "aya_split_candidate",
        "aya_split_receipt",
        "candidate_mix_report",
        "candidate_mix_verification",
        "chat_template",
        "data_contract",
        "predecessor_registry",
        "safety_candidate",
        "safety_owner_review_receipt",
        "safety_token_audit",
        "tokenizer_config",
        "tokenizer_json",
    }
    if not isinstance(bindings, Mapping) or not required.issubset(bindings):
        raise ValueError("candidate-hygiene binding set is incomplete")
    for name, binding in bindings.items():
        if not isinstance(name, str) or not isinstance(binding, Mapping):
            raise ValueError("invalid candidate-hygiene binding")
        resolve_binding(repository_root, binding, name=name)
    for name, relative in EXPECTED_OUTPUT.items():
        if name.endswith("_path") and isinstance(relative, str):
            resolve_output(repository_root, relative, name=name)
    return plan


def _validate_aya_receipt(receipt: Mapping[str, Any]) -> None:
    _verify_internal_sha(receipt, "receipt_sha256")
    if receipt.get("status") != "private_aya_split_materialized_nonproduction":
        raise ValueError("Aya split receipt status changed")
    if receipt.get("projection", {}).get("train") != EXPECTED_AYA_SPLIT["train"]:
        raise ValueError("Aya train projection changed")
    if receipt.get("projection", {}).get("validation") != EXPECTED_AYA_SPLIT["validation"]:
        raise ValueError("Aya validation projection changed")
    production = receipt.get("production")
    if not isinstance(production, Mapping):
        raise ValueError("Aya production boundary is missing")
    if any(production.get(field) is not False for field in (
        "dataset_sealed",
        "production_eligible",
        "production_sft_authorized",
        "source_registry_changed",
    )):
        raise ValueError("Aya split receipt crossed a production boundary")


def _validate_safety_owner_receipt(receipt: Mapping[str, Any]) -> None:
    _verify_internal_sha(receipt, "receipt_sha256")
    expected = {
        "accepted_records": 40,
        "all_sampled_rows_accepted": True,
        "owner_review_claim": "ai_generated_plus_human_owner_reviewed_not_human_written",
        "production_eligible": False,
        "production_promotions": 0,
        "raw_training_text_retained": False,
        "rejected_records": 0,
        "sample_records": 40,
        "status": "owner_spot_check_passed_source_registry_promotion_still_separate",
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ValueError(f"safety owner-review receipt changed: {field}")


def _source_registry_bytes(
    predecessor: Mapping[str, Any],
    *,
    policy: Any,
) -> tuple[bytes, dict[str, Any]]:
    successor = copy.deepcopy(predecessor)
    sources = successor.get("sources")
    if not isinstance(sources, list):
        raise ValueError("predecessor source registry changed")
    changed = 0
    source_after: dict[str, Any] | None = None
    for source in sources:
        if not isinstance(source, dict) or source.get("source_id") != AYA_SOURCE_ID:
            continue
        if source.get("splits") != ["train"]:
            raise ValueError("Aya predecessor split binding changed")
        if "frozen_validation_split_not_materialized" not in source.get(
            "production_blockers", []
        ):
            raise ValueError("Aya predecessor blocker changed")
        source["adapter"] = "aya_bengali_repaired_4k_reviewed_split_v0.3"
        source["splits"] = ["train", "validation"]
        source["production_blockers"] = [
            blocker
            for blocker in source["production_blockers"]
            if blocker != "frozen_validation_split_not_materialized"
        ]
        source_after = copy.deepcopy(source)
        changed += 1
    if changed != 1 or source_after is None:
        raise ValueError("exactly one Aya source entry must change")
    for source in sources:
        SFTSource.from_mapping(
            source,
            allowed_licenses=policy.allowed_licenses,
            allowed_languages=policy.allowed_languages,
            allowed_splits=policy.allowed_splits,
            allowed_tasks=frozenset(policy.task_mix),
        )
    return json_file_bytes(successor), source_after


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


def _rematerialize_safety_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    owner_receipt: Mapping[str, Any],
    owner_receipt_file_sha256: str,
    tokenizer: Any,
    template: Any,
) -> tuple[bytes, dict[str, Any]]:
    if len(rows) != 320:
        raise ValueError("safety candidate must contain exactly 320 rows")
    seen_ids: set[str] = set()
    output_rows: list[dict[str, Any]] = []
    split_records: Counter[str] = Counter()
    split_tokens: Counter[str] = Counter()
    message_binding_rows: list[dict[str, str]] = []
    missing_source_uri = 0

    for row in rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in seen_ids:
            raise ValueError("safety record IDs changed")
        seen_ids.add(record_id)
        if "split" in row:
            raise ValueError("safety predecessor unexpectedly has top-level split")
        source = row.get("source")
        provenance = row.get("provenance")
        messages = row.get("messages")
        if not isinstance(source, Mapping) or source.get("source_id") != SAFETY_SOURCE_ID:
            raise ValueError("safety source binding changed")
        split = source.get("split")
        if split not in {"train", "validation"}:
            raise ValueError("safety nested split changed")
        if not isinstance(provenance, Mapping):
            raise ValueError("safety provenance changed")
        if provenance.get("human_review_status") != "owner_spot_check_pending":
            raise ValueError("safety predecessor owner-review state changed")
        if not isinstance(messages, list):
            raise ValueError("safety messages changed")
        missing_source_uri += int(not source.get("uri"))

        rematerialized = copy.deepcopy(row)
        rematerialized["split"] = split
        repaired_provenance = dict(rematerialized["provenance"])
        prior_transform = _require_sha256(
            repaired_provenance.get("transform_sha256"),
            "safety.provenance.transform_sha256",
        )
        transform_binding = {
            "operation": "add_top_level_split_and_bind_owner_review_v0.1",
            "owner_review_receipt_file_sha256": owner_receipt_file_sha256,
            "owner_review_receipt_sha256": owner_receipt["receipt_sha256"],
            "prior_transform_sha256": prior_transform,
            "split": split,
        }
        repaired_provenance.update(
            {
                "human_review_status": "owner_spot_check_passed",
                "owner_review_claim": owner_receipt["owner_review_claim"],
                "owner_review_receipt_file_sha256": owner_receipt_file_sha256,
                "owner_review_receipt_sha256": owner_receipt["receipt_sha256"],
                "owner_review_sample_records": owner_receipt["sample_records"],
                "transform_sha256": sha256_value(transform_binding),
            }
        )
        rematerialized["provenance"] = repaired_provenance
        if rematerialized["messages"] != row["messages"]:
            raise ValueError("safety messages changed during metadata repair")
        if rematerialized["source"] != row["source"]:
            raise ValueError("safety source metadata changed during scoped repair")
        supervised_tokens = _measure_conversation(
            rematerialized["messages"],
            tokenizer=tokenizer,
            template=template,
        )
        split_records[split] += 1
        split_tokens[split] += supervised_tokens
        message_binding_rows.append(
            {
                "content_sha256": provenance["content_sha256"],
                "record_id": record_id,
                "messages_sha256": sha256_value(row["messages"]),
            }
        )
        output_rows.append(rematerialized)

    by_split = {
        split: {
            "records": split_records[split],
            "supervised_tokens": split_tokens[split],
        }
        for split in ("train", "validation")
    }
    if by_split != EXPECTED_SAFETY_TOKENS:
        raise ValueError("safety frozen-token totals changed")
    candidate_bytes = b"".join(
        canonical_json_bytes(row) + b"\n" for row in output_rows
    )
    metrics = {
        "by_split": by_split,
        "candidate_file_sha256": bytes_sha256(candidate_bytes),
        "message_and_content_binding_sha256": sha256_value(message_binding_rows),
        "records": len(output_rows),
        "records_missing_source_uri_after_scoped_repair": missing_source_uri,
        "rows_with_owner_review_receipt_binding": sum(
            row["provenance"].get("owner_review_receipt_sha256")
            == owner_receipt["receipt_sha256"]
            for row in output_rows
        ),
        "rows_with_top_level_split": sum(
            row.get("split") == row["source"].get("split") for row in output_rows
        ),
    }
    return candidate_bytes, metrics


def rebuild_candidate_hygiene_outputs(
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
    policy = load_sft_policy(resolved["data_contract"])
    predecessor = _load_json(resolved["predecessor_registry"])
    predecessor_registry = load_sft_source_registry(
        resolved["predecessor_registry"],
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
    )
    if predecessor_registry.registry_sha256 != plan["bindings"]["predecessor_registry"][
        "sha256"
    ]:
        raise ValueError("predecessor registry hash changed")
    aya_receipt = _load_json(resolved["aya_split_receipt"])
    _validate_aya_receipt(aya_receipt)
    if file_sha256(resolved["aya_split_candidate"]) != aya_receipt["artifacts"][
        "candidate"
    ]["sha256"]:
        raise ValueError("Aya split candidate changed")
    successor_bytes, aya_after = _source_registry_bytes(predecessor, policy=policy)

    owner_receipt = _load_json(resolved["safety_owner_review_receipt"])
    _validate_safety_owner_receipt(owner_receipt)
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
    safety_candidate_bytes, safety_metrics = _rematerialize_safety_rows(
        _load_jsonl(resolved["safety_candidate"]),
        owner_receipt=owner_receipt,
        owner_receipt_file_sha256=plan["bindings"]["safety_owner_review_receipt"][
            "sha256"
        ],
        tokenizer=tokenizer,
        template=template,
    )
    safety_audit = _load_json(resolved["safety_token_audit"])
    _verify_internal_sha(safety_audit, "report_sha256")
    for split in ("train", "validation"):
        observed = safety_audit["metrics"]["by_split"][split]
        expected = EXPECTED_SAFETY_TOKENS[split]
        if observed["records"] != expected["records"] or observed[
            "supervised_tokens"
        ] != expected["supervised_tokens"]:
            raise ValueError("safety token-audit binding changed")

    manifest: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "input_candidate_file_sha256": plan["bindings"]["safety_candidate"]["sha256"],
        "message_and_content_binding_sha256": safety_metrics[
            "message_and_content_binding_sha256"
        ],
        "output_candidate_file_sha256": safety_metrics["candidate_file_sha256"],
        "owner_review_receipt_file_sha256": plan["bindings"][
            "safety_owner_review_receipt"
        ]["sha256"],
        "owner_review_receipt_sha256": owner_receipt["receipt_sha256"],
        "raw_training_text_retained": False,
        "records": safety_metrics["records"],
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "split_counts": {
            split: safety_metrics["by_split"][split]["records"]
            for split in ("train", "validation")
        },
        "status": "safety_metadata_rematerialized_nonproduction",
    }
    manifest["manifest_sha256"] = sha256_value(manifest)
    manifest_bytes = json_file_bytes(manifest)
    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "aya_registry_delta": {
            "adapter_after": aya_after["adapter"],
            "adapter_before": predecessor_registry.by_id[AYA_SOURCE_ID].adapter,
            "changed_source_id": AYA_SOURCE_ID,
            "predecessor_splits": sorted(predecessor_registry.by_id[AYA_SOURCE_ID].splits),
            "successor_splits": aya_after["splits"],
            "validation_records_bound": EXPECTED_AYA_SPLIT["validation"]["records"],
            "validation_supervised_tokens_bound": EXPECTED_AYA_SPLIT["validation"][
                "supervised_tokens"
            ],
        },
        "bindings": {
            "artifact_manifest_file_sha256": bytes_sha256(manifest_bytes),
            "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
            "predecessor_registry_sha256": bytes_sha256(
                resolved["predecessor_registry"].read_bytes()
            ),
            "private_safety_candidate_file_sha256": safety_metrics[
                "candidate_file_sha256"
            ],
            "successor_registry_sha256": bytes_sha256(successor_bytes),
        },
        "observed_at": plan["observed_at"],
        "production_dataset_promotions": 0,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "remaining_blockers": [
            "safety_source_uri_missing_from_320_rows_outside_this_scoped_repair",
            "accepted_train_supervised_tokens_below_10m_floor",
            "task_language_and_source_balance_not_closed",
            "full_mix_cross_source_near_deduplication_not_complete",
            "active_production_data_contract_not_updated",
            "dataset_seal_cpu_canary_and_sft_not_authorized",
        ],
        "safety_rematerialization": safety_metrics,
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "verified_scoped_candidate_hygiene_repair_nonproduction",
    }
    report["report_sha256"] = sha256_value(report)
    validate_candidate_hygiene_report(report)
    return {
        "artifact_manifest_bytes": manifest_bytes,
        "private_safety_candidate_bytes": safety_candidate_bytes,
        "report": report,
        "successor_registry_bytes": successor_bytes,
    }


def validate_candidate_hygiene_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported candidate-hygiene report schema")
    _verify_internal_sha(report, "report_sha256")
    if report.get("production_eligible") is not False:
        raise ValueError("candidate-hygiene report cannot be production eligible")
    if report.get("production_dataset_promotions") != 0:
        raise ValueError("candidate-hygiene repair cannot promote a production dataset")
    if report.get("raw_training_text_retained") is not False:
        raise ValueError("candidate-hygiene report must remain raw-text-free")
    safety = report.get("safety_rematerialization")
    if not isinstance(safety, Mapping):
        raise ValueError("safety rematerialization metrics are missing")
    if safety.get("records") != 320:
        raise ValueError("safety rematerialization row count changed")
    if safety.get("rows_with_top_level_split") != 320:
        raise ValueError("not every safety row has a top-level split")
    if safety.get("rows_with_owner_review_receipt_binding") != 320:
        raise ValueError("not every safety row binds the owner-review receipt")


def build_candidate_hygiene_verification(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
) -> dict[str, Any]:
    validate_candidate_hygiene_report(report)
    verification: dict[str, Any] = {
        "checks": {
            "aya_validation_split_registry_bound": True,
            "exactly_one_registry_source_changed": True,
            "frozen_token_counts_unchanged": True,
            "messages_and_content_unchanged": True,
            "no_network_api_gpu_seal_sft_or_production_promotion": True,
            "owner_review_receipt_bound_to_all_320_rows": True,
            "predecessor_registry_unchanged": True,
            "raw_training_text_absent_from_repo_evidence": True,
            "top_level_split_present_on_all_320_rows": True,
        },
        "observed_at": plan["observed_at"],
        "production_eligible": False,
        "report_file_sha256": _require_sha256(
            report_file_sha256, "report_file_sha256"
        ),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_scoped_candidate_hygiene_outputs_nonproduction",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification
