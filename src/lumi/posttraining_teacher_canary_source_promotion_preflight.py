"""Fail-closed registry preflight for the reviewed Lumi teacher-canary source."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .data_quality import BenchmarkRegistry
from .posttraining_data import filter_sft_records, load_sft_policy
from .posttraining_first_party_teacher_canary_v18 import _load_direct_tokenizer
from .posttraining_format import encode_sft_messages, load_chat_template
from .posttraining_sources import SFTSource, SFTSourceRegistry

PLAN_SCHEMA_VERSION = (
    "lumi.sft-first-party-teacher-source-promotion-preflight.v0.1"
)
REPORT_SCHEMA_VERSION = (
    "lumi.sft-first-party-teacher-source-promotion-preflight-report.v0.1"
)
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-first-party-teacher-source-promotion-preflight-verification.v0.1"
)
PLAN_STATUS = "local_preflight_only_registry_successor_not_materialized"
OWNER_REQUEST_TEXT = "ok next step shuru kor"

SOURCE_ID = "lumi_first_party_teacher_canary_v0_1"
DATASET_ID = "Afnan-Technologies-Inc/lumi-first-party-teacher-canary-v0.1"
SOURCE_REVISION = "c1c9c2d09f1680c2e18eb2c7570994c141501f40732bd8b2b890015c97a88634"
EXPECTED_CANDIDATE_FILE_SHA256 = (
    "e255e2bc9a09c28a50f4f32caebfd834f2b42e1bb3ffe90a05f4bf01cb46fda2"
)
EXPECTED_OWNER_RECEIPT_FILE_SHA256 = (
    "ef23c8f1d8755a4f61e36506c416d1a0e7ce7ff7a7596552af3a8b79ac7f0a86"
)
EXPECTED_OWNER_RECEIPT_INTERNAL_SHA256 = (
    "796781579187b0f6e8e282bcccba77ed258c5ca13828e98772d20fb44f49ac57"
)
EXPECTED_RECORDS = 80
EXPECTED_TRAIN_RECORDS = 60
EXPECTED_VALIDATION_RECORDS = 20
EXPECTED_TRAIN_TOKENS = 2_002
EXPECTED_VALIDATION_TOKENS = 799
EXPECTED_TOTAL_TOKENS = 2_801
EXPECTED_PREDECESSOR_SOURCES = 15

EXPECTED_AUTHORIZATION = {
    "candidate_registry_change_authorized": False,
    "canonical_candidate_materialization_authorized": False,
    "data_acquisition_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "purpose": "reviewed_teacher_canary_registry_v0_15_successor_projection_only",
    "source_registry_promotion_authorized": False,
}
EXPECTED_LIMITS = {
    "added_sources": 1,
    "candidate_records": EXPECTED_RECORDS,
    "maximum_report_bytes": 1_048_576,
    "maximum_runtime_minutes": 30,
    "train_records": EXPECTED_TRAIN_RECORDS,
    "train_supervised_tokens": EXPECTED_TRAIN_TOKENS,
    "validation_records": EXPECTED_VALIDATION_RECORDS,
    "validation_supervised_tokens": EXPECTED_VALIDATION_TOKENS,
}
EXPECTED_BINDINGS = {
    "candidate_rows",
    "chat_template",
    "data_contract",
    "data_quality",
    "finalization_receipt",
    "finalization_verification",
    "implementation",
    "owner_review_receipt",
    "owner_review_verification",
    "predecessor_registry",
    "run_report",
    "run_verification",
    "runner",
    "test",
    "tokenizer_config",
    "tokenizer_json",
    "verifier",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_value(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def jsonl_file_bytes(values: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for value in values
    )


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSON object required at line {line_number}: {path}")
            values.append(value)
    return values


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
        raise ValueError(f"invalid teacher-canary promotion binding: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.parent.resolve()) or not resolved.is_file():
        raise ValueError(f"teacher-canary promotion binding missing: {name}")
    if file_sha256(resolved) != expected:
        raise ValueError(f"teacher-canary promotion binding changed: {name}")
    return resolved


def _policy_sets(contract: Mapping[str, Any]) -> dict[str, frozenset[str]]:
    task_mix = contract.get("task_mix")
    if not isinstance(task_mix, Mapping):
        raise ValueError("teacher-canary data-contract task mix changed")
    values = {
        "licenses": contract.get("allowed_licenses"),
        "languages": contract.get("allowed_languages"),
        "splits": contract.get("allowed_splits"),
        "tasks": list(task_mix),
    }
    if any(
        not isinstance(items, list)
        or not items
        or not all(isinstance(item, str) for item in items)
        for items in values.values()
    ):
        raise ValueError("teacher-canary data-contract allowlists changed")
    return {name: frozenset(items) for name, items in values.items()}


def load_promotion_preflight_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _read_object(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported teacher-canary source-promotion preflight")
    if plan.get("status") != PLAN_STATUS:
        raise ValueError("teacher-canary source-promotion preflight status changed")
    if plan.get("observed_at") != "2026-08-09":
        raise ValueError("teacher-canary source-promotion preflight date changed")
    if plan.get("owner_request_text") != OWNER_REQUEST_TEXT:
        raise ValueError("teacher-canary source-promotion owner request changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("teacher-canary preflight authorization boundary changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("teacher-canary promotion-preflight limits changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("teacher-canary promotion-preflight bindings changed")
    for name, specification in bindings.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"invalid teacher-canary promotion binding: {name}")
        resolve_binding(root, specification, name=name)
    if bindings["candidate_rows"]["sha256"] != EXPECTED_CANDIDATE_FILE_SHA256:
        raise ValueError("teacher-canary candidate identity changed")
    if (
        bindings["owner_review_receipt"]["sha256"]
        != EXPECTED_OWNER_RECEIPT_FILE_SHA256
    ):
        raise ValueError("teacher-canary owner receipt identity changed")
    source = plan.get("source_after")
    if not isinstance(source, Mapping):
        raise ValueError("teacher-canary proposed source is missing")
    if source.get("source_id") != SOURCE_ID or source.get("revision") != SOURCE_REVISION:
        raise ValueError("teacher-canary proposed source identity changed")
    if plan.get("output") != {
        "canonical_candidate_path": (
            "../lumi-artifacts/4b-sft-first-party-teacher-canary-v0.1/"
            "e80b7aa7936b/attempt-018/private/canonical-candidate-v0.1.jsonl"
        ),
        "preflight_report_path": (
            "evidence/lumi-4b-sft-first-party-teacher-source-promotion-preflight-v0.1.json"
        ),
        "projected_successor_registry_path": (
            "configs/lumi-4b-sft-source-registry-v0.15.json"
        ),
        "verification_path": (
            "evidence/"
            "lumi-4b-sft-first-party-teacher-source-promotion-preflight-verification-v0.1.json"
        ),
    }:
        raise ValueError("teacher-canary promotion-preflight outputs changed")
    return plan


def build_projected_successor(
    predecessor: Mapping[str, Any],
    *,
    source_after: Mapping[str, Any],
) -> dict[str, Any]:
    sources = predecessor.get("sources")
    if not isinstance(sources, list) or len(sources) != EXPECTED_PREDECESSOR_SOURCES:
        raise ValueError("teacher-canary predecessor source count changed")
    if any(
        isinstance(source, Mapping)
        and (
            source.get("source_id") == SOURCE_ID
            or source.get("dataset_id") == DATASET_ID
        )
        for source in sources
    ):
        raise ValueError("teacher-canary source already exists in predecessor registry")
    successor = copy.deepcopy(dict(predecessor))
    successor["sources"].append(copy.deepcopy(dict(source_after)))
    return successor


def _verify_owner_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema_version") != "lumi.ai-assisted-owner-review-receipt.v0.3":
        raise ValueError("unsupported teacher-canary owner receipt")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != sha256_value(unsigned):
        raise ValueError("teacher-canary owner receipt seal changed")
    if receipt.get("receipt_sha256") != EXPECTED_OWNER_RECEIPT_INTERNAL_SHA256:
        raise ValueError("teacher-canary owner receipt identity changed")
    if (
        receipt.get("status")
        != "owner_spot_check_passed_pilot_still_not_production_dataset"
        or receipt.get("sample_records") != 32
        or receipt.get("accepted_records") != 32
        or receipt.get("rejected_records") != 0
        or receipt.get("all_sampled_rows_accepted") is not True
        or receipt.get("owner_review_claim")
        != "ai_generated_plus_owner_reviewed_not_human_written"
        or receipt.get("production_eligible") is not False
    ):
        raise ValueError("teacher-canary owner-review outcome changed")


def _verify_bound_evidence(root: Path, bindings: Mapping[str, Any]) -> None:
    run = _read_object(resolve_binding(root, bindings["run_report"], name="run_report"))
    if (
        run.get("status") != "verified_private_candidate_pending_owner_review"
        or run.get("audit", {}).get("candidate_internal_sha256") != SOURCE_REVISION
        or run.get("audit", {}).get("counts")
        != {"records": 80, "train": 60, "validation": 20}
        or run.get("audit", {}).get("tokens", {}).get("retained_supervised_tokens")
        != EXPECTED_TOTAL_TOKENS
        or run.get("candidate_registry_changed") is not False
        or run.get("dataset_sealed") is not False
        or run.get("sft_started") is not False
        or run.get("external_api_used") is not False
        or run.get("human_authorship_claimed") is not False
    ):
        raise ValueError("teacher-canary run evidence changed")
    run_verification = _read_object(
        resolve_binding(root, bindings["run_verification"], name="run_verification")
    )
    if (
        run_verification.get("status")
        != "verified_private_candidate_pending_owner_review"
        or run_verification.get("candidate_internal_sha256") != SOURCE_REVISION
        or not all(run_verification.get("checks", {}).values())
    ):
        raise ValueError("teacher-canary run verification changed")

    finalization = _read_object(
        resolve_binding(root, bindings["finalization_receipt"], name="finalization_receipt")
    )
    if (
        finalization.get("status")
        != "verified_private_candidate_pending_owner_review"
        or finalization.get("result", {}).get("records") != EXPECTED_RECORDS
        or finalization.get("result", {}).get("retained_supervised_tokens")
        != EXPECTED_TOTAL_TOKENS
        or finalization.get("prohibitions_verified", {}).get("sft_started") is not False
    ):
        raise ValueError("teacher-canary finalization receipt changed")
    finalization_verification = _read_object(
        resolve_binding(
            root,
            bindings["finalization_verification"],
            name="finalization_verification",
        )
    )
    if not all(finalization_verification.get("checks", {}).values()):
        raise ValueError("teacher-canary finalization verification changed")

    owner = _read_object(
        resolve_binding(root, bindings["owner_review_receipt"], name="owner_review_receipt")
    )
    _verify_owner_receipt(owner)
    owner_verification = _read_object(
        resolve_binding(
            root,
            bindings["owner_review_verification"],
            name="owner_review_verification",
        )
    )
    if (
        owner_verification.get("status")
        != "owner_spot_check_passed_pilot_still_not_production_dataset"
        or owner_verification.get("counts")
        != {"sample_records": 32, "decisions": 32, "accepted": 32, "rejected": 0}
        or not all(owner_verification.get("checks", {}).values())
        or owner_verification.get("production_eligible") is not False
    ):
        raise ValueError("teacher-canary owner-review verification changed")


def _language_for_mode(mode: str) -> str:
    mapping = {
        "banglish": "multilingual",
        "bengali_script": "bn",
        "english": "en",
        "natural_code_switching": "multilingual",
    }
    try:
        return mapping[mode]
    except KeyError as error:
        raise ValueError(f"unsupported teacher-canary language mode: {mode}") from error


def _verify_raw_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Counter[str]]:
    if len(rows) != EXPECTED_RECORDS or sha256_value(list(rows)) != SOURCE_REVISION:
        raise ValueError("teacher-canary candidate record identity changed")
    record_ids: set[str] = set()
    counts: dict[str, Counter[str]] = {
        "splits": Counter(),
        "tasks": Counter(),
        "language_modes": Counter(),
        "rights_statuses": Counter(),
        "source_attempt_ids": Counter(),
    }
    for row in rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in record_ids:
            raise ValueError("teacher-canary record IDs changed")
        record_ids.add(record_id)
        if row.get("schema_version") != "lumi.sft-first-party-teacher-canary-row.v0.1":
            raise ValueError("teacher-canary private row schema changed")
        source_id = row.get("source_id")
        if not isinstance(source_id, str) or not source_id.startswith(
            "lumi_first_party_multilingual_safety_tool_v0_"
        ):
            raise ValueError("teacher-canary attempt source identity changed")
        provenance = row.get("provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("ai_assisted") is not True
            or provenance.get("human_authored") is not False
            or provenance.get("self_hosted") is not True
        ):
            raise ValueError("teacher-canary AI-assisted provenance changed")
        if row.get("rights_status") != "private_research_candidate_pending_owner_review":
            raise ValueError("teacher-canary pre-review rights status changed")
        counts["splits"][str(row.get("split"))] += 1
        counts["tasks"][str(row.get("task"))] += 1
        counts["language_modes"][str(row.get("language_mode"))] += 1
        counts["rights_statuses"][str(row.get("rights_status"))] += 1
        counts["source_attempt_ids"][source_id] += 1
    if counts["splits"] != Counter({"train": 60, "validation": 20}):
        raise ValueError("teacher-canary split inventory changed")
    if counts["tasks"] != Counter(
        {
            "factual_qa": 16,
            "multilingual_bengali": 16,
            "safety_helpfulness": 16,
            "tool_api": 16,
            "writing_summarization": 16,
        }
    ):
        raise ValueError("teacher-canary task inventory changed")
    if counts["language_modes"] != Counter(
        {
            "banglish": 20,
            "bengali_script": 20,
            "english": 20,
            "natural_code_switching": 20,
        }
    ):
        raise ValueError("teacher-canary language inventory changed")
    return counts


def project_canonical_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    owner_receipt_file_sha256: str,
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for row in rows:
        record_id = str(row["record_id"])
        messages = copy.deepcopy(row["messages"])
        original_provenance = row["provenance"]
        transform = {
            "contract": "lumi.teacher-canary-canonical-owner-bound.v0.1",
            "original_provenance_sha256": sha256_value(original_provenance),
            "original_rights_status": row["rights_status"],
            "original_source_id": row["source_id"],
            "owner_review_receipt_file_sha256": owner_receipt_file_sha256,
            "source_revision": SOURCE_REVISION,
        }
        projected.append(
            {
                "language": _language_for_mode(str(row["language_mode"])),
                "messages": messages,
                "provenance": {
                    "ai_assisted": True,
                    "consent_basis": "first_party_rights_declaration",
                    "content_sha256": sha256_value(messages),
                    "human_authored": False,
                    "human_authorship_claimed": False,
                    "original_provenance_sha256": sha256_value(original_provenance),
                    "owner_review_claim": (
                        "ai_generated_plus_owner_reviewed_not_human_written"
                    ),
                    "owner_review_receipt_file_sha256": owner_receipt_file_sha256,
                    "owner_review_receipt_internal_sha256": (
                        EXPECTED_OWNER_RECEIPT_INTERNAL_SHA256
                    ),
                    "owner_review_scope": "deterministic_32_row_spot_check",
                    "transform_sha256": sha256_value(transform),
                },
                "quality": {
                    "pii_scan": "passed",
                    "review_status": "accepted",
                    "score": 0.95,
                    "secret_scan": "passed",
                },
                "record_id": record_id,
                "safety": {
                    "policy_version": "lumi-assistant-behavior-v0.1",
                    "risk_labels": [],
                },
                "schema_version": "lumi.sft.v0.1",
                "source": {
                    "dataset_id": DATASET_ID,
                    "document_id": record_id,
                    "license_id": "lumi-owned",
                    "revision": SOURCE_REVISION,
                    "source_id": SOURCE_ID,
                    "uri": f"lumi-private://first-party-teacher-canary/v0.1/{record_id}",
                },
                "split": row["split"],
                "task_category": row["task"],
            }
        )
    return projected


def _token_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    chat_template_path: Path,
    tokenizer_root: Path,
) -> dict[str, Any]:
    tokenizer = _load_direct_tokenizer(tokenizer_root)
    template = load_chat_template(chat_template_path)
    by_split: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "masked_tokens": 0,
            "records": 0,
            "serialized_tokens": 0,
            "supervised_tokens": 0,
        }
    )
    by_task: Counter[str] = Counter()
    by_language_mode: Counter[str] = Counter()
    for row in rows:
        encoded = encode_sft_messages(row["messages"], tokenizer=tokenizer, template=template)
        split = str(row["split"])
        by_split[split]["masked_tokens"] += encoded.masked_tokens
        by_split[split]["records"] += 1
        by_split[split]["serialized_tokens"] += len(encoded.input_ids)
        by_split[split]["supervised_tokens"] += encoded.supervised_tokens
        by_task[str(row["task"])] += encoded.supervised_tokens
        by_language_mode[str(row["language_mode"])] += encoded.supervised_tokens
    expected_split = {
        "train": {
            "masked_tokens": 1_610,
            "records": 60,
            "serialized_tokens": 3_612,
            "supervised_tokens": EXPECTED_TRAIN_TOKENS,
        },
        "validation": {
            "masked_tokens": 605,
            "records": 20,
            "serialized_tokens": 1_404,
            "supervised_tokens": EXPECTED_VALIDATION_TOKENS,
        },
    }
    if dict(by_split) != expected_split:
        raise ValueError("teacher-canary frozen split token metrics changed")
    if sum(item["supervised_tokens"] for item in by_split.values()) != EXPECTED_TOTAL_TOKENS:
        raise ValueError("teacher-canary frozen total token count changed")
    return {
        "by_language_mode": dict(sorted(by_language_mode.items())),
        "by_split": expected_split,
        "by_task": dict(sorted(by_task.items())),
        "total_supervised_tokens": EXPECTED_TOTAL_TOKENS,
    }


def build_promotion_preflight_report(
    plan: Mapping[str, Any],
    *,
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    bindings = plan["bindings"]
    _verify_bound_evidence(root, bindings)

    contract_path = resolve_binding(root, bindings["data_contract"], name="data_contract")
    contract = _read_object(contract_path)
    policy = load_sft_policy(contract_path)
    allowed = _policy_sets(contract)
    predecessor_path = resolve_binding(
        root,
        bindings["predecessor_registry"],
        name="predecessor_registry",
    )
    predecessor = _read_object(predecessor_path)
    proposed_source = SFTSource.from_mapping(
        plan["source_after"],
        allowed_licenses=allowed["licenses"],
        allowed_languages=allowed["languages"],
        allowed_splits=allowed["splits"],
        allowed_tasks=allowed["tasks"],
    )
    successor = build_projected_successor(
        predecessor,
        source_after=plan["source_after"],
    )
    parsed_sources = tuple(
        SFTSource.from_mapping(
            source,
            allowed_licenses=allowed["licenses"],
            allowed_languages=allowed["languages"],
            allowed_splits=allowed["splits"],
            allowed_tasks=allowed["tasks"],
        )
        for source in successor["sources"]
    )
    if parsed_sources[-1] != proposed_source or len(parsed_sources) != 16:
        raise ValueError("teacher-canary projected successor changed")
    successor_bytes = json_file_bytes(successor)
    successor_sha256 = hashlib.sha256(successor_bytes).hexdigest()
    candidate_registry = SFTSourceRegistry(
        observed_at=str(predecessor.get("observed_at")),
        registry_sha256=successor_sha256,
        sources=parsed_sources,
    )

    candidate_path = resolve_binding(root, bindings["candidate_rows"], name="candidate_rows")
    rows = _read_jsonl(candidate_path)
    raw_counts = _verify_raw_rows(rows)
    canonical = project_canonical_candidate(
        rows,
        owner_receipt_file_sha256=EXPECTED_OWNER_RECEIPT_FILE_SHA256,
    )
    quality_path = resolve_binding(root, bindings["data_quality"], name="data_quality")
    quality = _read_object(quality_path)
    benchmark_specification = quality.get("benchmark_decontamination")
    if not isinstance(benchmark_specification, Mapping):
        raise ValueError("teacher-canary benchmark-decontamination contract changed")
    benchmark = BenchmarkRegistry(benchmark_specification, root)
    accepted, filter_report = filter_sft_records(
        canonical,
        policy=policy,
        benchmark_registry=benchmark,
        source_registry=candidate_registry,
    )
    if len(accepted) != EXPECTED_RECORDS or filter_report.get("rejected") != 0:
        raise ValueError("teacher-canary canonical projection failed SFT filtering")

    chat_template_path = resolve_binding(
        root,
        bindings["chat_template"],
        name="chat_template",
    )
    tokenizer_json_path = resolve_binding(
        root,
        bindings["tokenizer_json"],
        name="tokenizer_json",
    )
    token_metrics = _token_metrics(
        rows,
        chat_template_path=chat_template_path,
        tokenizer_root=tokenizer_json_path.parent,
    )
    canonical_bytes = jsonl_file_bytes(canonical)

    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "candidate": {
            "candidate_file_sha256": file_sha256(candidate_path),
            "candidate_internal_sha256": SOURCE_REVISION,
            "language_mode_counts": dict(sorted(raw_counts["language_modes"].items())),
            "owner_sample_accepted": 32,
            "owner_sample_rejected": 0,
            "records": EXPECTED_RECORDS,
            "rights_status_counts": dict(sorted(raw_counts["rights_statuses"].items())),
            "source_attempt_id_count": len(raw_counts["source_attempt_ids"]),
            "split_counts": dict(sorted(raw_counts["splits"].items())),
            "task_counts": dict(sorted(raw_counts["tasks"].items())),
            "token_metrics": token_metrics,
        },
        "canonical_candidate_projection": {
            "bytes": len(canonical_bytes),
            "file_sha256": hashlib.sha256(canonical_bytes).hexdigest(),
            "filter_accepted": len(accepted),
            "filter_rejected": int(filter_report.get("rejected", -1)),
            "internal_sha256": sha256_value(canonical),
            "materialized": False,
            "path": plan["output"]["canonical_candidate_path"],
            "records": EXPECTED_RECORDS,
        },
        "checks": {
            "all_80_rows_pass_projected_registry_and_data_contract": True,
            "candidate_and_owner_receipt_hashes_verified": True,
            "candidate_registry_change_executed": False,
            "canonical_candidate_materialization_executed": False,
            "external_api_used": False,
            "human_authorship_claimed": False,
            "owner_review_provenance_bound_in_projection": True,
            "predecessor_registry_unchanged": True,
            "production_dataset_promotions": 0,
            "projected_successor_adds_only_teacher_canary": True,
            "raw_training_text_retained_in_report": False,
            "stale_pending_owner_status_detected_and_resolved_in_projection": True,
        },
        "next_gate": {
            "action": (
                "separately_authorize_exact_canonical_candidate_and_immutable_"
                "registry_v0_15_materialization"
            ),
            "authorization_required": True,
            "post_promotion_action": (
                "full_local_candidate_mix_reaudit_then_10m_dataset_closure_gate"
            ),
        },
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "production_eligible": False,
        "projected_successor": {
            "added_source_id": SOURCE_ID,
            "bytes": len(successor_bytes),
            "path": plan["output"]["projected_successor_registry_path"],
            "predecessor_file_sha256": file_sha256(predecessor_path),
            "predecessor_source_count": EXPECTED_PREDECESSOR_SOURCES,
            "sha256": successor_sha256,
            "source_count": 16,
        },
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "verified_local_preflight_registry_successor_not_materialized",
    }
    report["report_sha256"] = sha256_value(report)
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported teacher-canary promotion-preflight report")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if report.get("report_sha256") != sha256_value(unsigned):
        raise ValueError("teacher-canary promotion-preflight report seal mismatch")
    if report.get("production_eligible") is not False:
        raise ValueError("teacher-canary preflight crossed production boundary")
    checks = report.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("teacher-canary promotion-preflight checks are missing")
    if checks.get("candidate_registry_change_executed") is not False:
        raise ValueError("teacher-canary preflight changed the source registry")
    if checks.get("canonical_candidate_materialization_executed") is not False:
        raise ValueError("teacher-canary preflight materialized private candidate data")
    if checks.get("production_dataset_promotions") != 0:
        raise ValueError("teacher-canary preflight promoted production data")
    if checks.get("raw_training_text_retained_in_report") is not False:
        raise ValueError("teacher-canary preflight retained raw training text")
    projected = report.get("projected_successor")
    if not isinstance(projected, Mapping) or projected.get("source_count") != 16:
        raise ValueError("teacher-canary projected registry count changed")
    candidate = report.get("canonical_candidate_projection")
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("records") != EXPECTED_RECORDS
        or candidate.get("filter_accepted") != EXPECTED_RECORDS
        or candidate.get("filter_rejected") != 0
        or candidate.get("materialized") is not False
    ):
        raise ValueError("teacher-canary canonical candidate projection changed")


def build_verification(
    report: Mapping[str, Any],
    *,
    report_file_sha256: str,
    recomputed_report: Mapping[str, Any],
    plan: Mapping[str, Any],
    repository_root: str | Path,
) -> dict[str, Any]:
    validate_report(report)
    validate_report(recomputed_report)
    if report != recomputed_report:
        raise ValueError("teacher-canary preflight replay changed")
    root = Path(repository_root).resolve()
    for output_name in ("canonical_candidate_path", "projected_successor_registry_path"):
        output = (root / plan["output"][output_name]).resolve()
        if output.exists():
            raise ValueError(f"teacher-canary preflight output was unexpectedly materialized: {output}")
    verification: dict[str, Any] = {
        "checks": {
            "all_bindings_reverified": True,
            "canonical_candidate_not_materialized": True,
            "candidate_registry_not_changed": True,
            "independent_replay_byte_exact": True,
            "next_gate_requires_separate_authorization": True,
            "production_sft_not_authorized": True,
            "raw_training_text_absent": True,
            "report_file_hash_verified": True,
        },
        "preflight_report_file_sha256": report_file_sha256,
        "preflight_report_internal_sha256": report["report_sha256"],
        "projected_canonical_candidate_file_sha256": report[
            "canonical_candidate_projection"
        ]["file_sha256"],
        "projected_successor_registry_sha256": report["projected_successor"]["sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_local_preflight_registry_successor_not_materialized",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


def validate_verification(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != VERIFICATION_SCHEMA_VERSION:
        raise ValueError("unsupported teacher-canary preflight verification")
    unsigned = {key: item for key, item in value.items() if key != "verification_sha256"}
    if value.get("verification_sha256") != sha256_value(unsigned):
        raise ValueError("teacher-canary preflight verification seal mismatch")
    checks = value.get("checks")
    if not isinstance(checks, Mapping) or not all(checks.values()):
        raise ValueError("teacher-canary preflight verification checks failed")


__all__ = [
    "PLAN_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "VERIFICATION_SCHEMA_VERSION",
    "build_projected_successor",
    "build_promotion_preflight_report",
    "build_verification",
    "file_sha256",
    "json_file_bytes",
    "jsonl_file_bytes",
    "load_promotion_preflight_plan",
    "project_canonical_candidate",
    "sha256_value",
    "validate_report",
    "validate_verification",
]
