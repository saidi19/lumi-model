"""Promote the owner-accepted ProsocialDialog candidate into registry v0.19."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_prosocial_dialog_safety_curation_v1 import file_sha256, sha256_value
from .posttraining_sources import load_sft_source_registry
from .posttraining_data import load_sft_policy

PLAN_SCHEMA_VERSION = "lumi.sft-prosocial-dialog-source-promotion-plan.v0.19"
RECEIPT_SCHEMA_VERSION = "lumi.sft-prosocial-dialog-source-promotion-receipt.v0.19"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-prosocial-dialog-source-promotion-verification.v0.19"
)
DRAFT_SCHEMA_VERSION = "lumi.sft-candidate-mix-reaudit-plan.v0.13-draft"

SOURCE_ID = "allenai_prosocial_dialog_safety_candidate"
DATASET_ID = "allenai/prosocial-dialog"
REVISION = "d77d7ad3c624c51030f2f32c83e892b3d620b3d4"
LICENSE_ID = "cc-by-4.0"
EXPECTED_PREDECESSOR_REGISTRY_SHA256 = (
    "b9ba6a6c551fd7e4da76bc6e0df5e347d0fcdf47f31e12599e4ddc4665dab01d"
)
EXPECTED_SUCCESSOR_REGISTRY_SHA256 = (
    "d88ef71863c38c277260172dd6b3b5ce5e2330726370329bc438b7750f43ec9c"
)
EXPECTED_SUCCESSOR_REGISTRY_BYTES = 26_895
EXPECTED_CANDIDATE_SHA256 = (
    "9f9f8dfdbd3bc667c0329936a9ef312f3a05b9547cfe061f5d3e66d751d02829"
)
EXPECTED_CANDIDATE_BYTES = 46_770_099
EXPECTED_RECORDS = 29_264
EXPECTED_TRAIN_RECORDS = 27_463
EXPECTED_VALIDATION_RECORDS = 1_801
EXPECTED_TRAIN_TOKENS = 899_998
EXPECTED_VALIDATION_TOKENS = 59_999
EXPECTED_OWNER_RECEIPT_SHA256 = (
    "3323bcc0dc1895c8595339fc7dcadc8da06fa16b571b6b95ff625e2b2f1e574d"
)
EXPECTED_OWNER_VERIFICATION_SHA256 = (
    "a622a396676531c5d2ea80f8629937aa95e26d44971a40e237eb328d2532e0c8"
)
EXPECTED_CURATION_REPORT_SHA256 = (
    "fbafcbfdf82348ef1de6748640dfa62d03e9909712898d0b571c933bcb5d1ea7"
)
EXPECTED_CURATION_VERIFICATION_SHA256 = (
    "31df85c19fc2ac2580485dc3bd1e31c31b595923356ee4c754900fbc06214574"
)
EXPECTED_CROSS_SOURCE_REPORT_SHA256 = (
    "0b170105caf5713e58363f39270a6efe2bbca8b0c4641badf4d3c79b5e767429"
)
EXPECTED_CROSS_SOURCE_VERIFICATION_SHA256 = (
    "35f4aceb519b4681e611ed104b30ee53dfed6de4f23a02eaa4b730d340b3f43b"
)
EXPECTED_AUTHORIZATION = {
    "candidate_mutation_authorized": False,
    "candidate_registry_change_authorized": True,
    "data_contract_change_authorized": False,
    "dataset_seal_authorized": False,
    "download_authorized": False,
    "external_api_authorized": False,
    "full_candidate_mix_reaudit_authorized": False,
    "gpu_authorized": False,
    "network_authorized": False,
    "predecessor_registry_mutation_authorized": False,
    "production_promotion_authorized": False,
    "remote_access_authorized": False,
    "sft_training_authorized": False,
    "source_registry_promotion_authorized": True,
    "unrelated_action_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 2_097_152,
    "maximum_runtime_minutes": 30,
}


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object: {path}:{line_number}")
            rows.append(value)
    return rows


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite ProsocialDialog promotion output: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes_atomic(path, json_file_bytes(value))


def source_metadata() -> dict[str, Any]:
    return {
        "access": "public",
        "adapter": "prosocial_dialog_mixed_provenance_safety_owner_bound_v0.19",
        "blockers": [],
        "dataset_id": DATASET_ID,
        "evidence_urls": [
            f"https://huggingface.co/datasets/{DATASET_ID}/tree/{REVISION}"
        ],
        "languages": ["en"],
        "license_id": LICENSE_ID,
        "origin": "mixed",
        "production_blockers": [
            "owner_review_is_32_row_deterministic_sample_not_full_record_review",
            "full_registry_v0_19_cross_source_hygiene_audit_not_complete",
            "production_dataset_seal_and_sft_authorization_not_complete",
        ],
        "production_eligible": False,
        "repository_url": f"https://huggingface.co/datasets/{DATASET_ID}",
        "revision": REVISION,
        "source_id": SOURCE_ID,
        "splits": ["train", "validation"],
        "status": "canary_eligible",
        "task_categories": ["safety_helpfulness"],
    }


def load_plan(path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("ProsocialDialog promotion plan schema changed")
    if plan.get("status") != "authorized_registry_v0_19_promotion_no_reaudit":
        raise ValueError("ProsocialDialog promotion is not authorized")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("ProsocialDialog promotion authorization changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("ProsocialDialog promotion limits changed")
    text = plan.get("authorization_text")
    if not isinstance(text, str) or hashlib.sha256(text.encode()).hexdigest() != plan.get(
        "authorization_text_sha256"
    ):
        raise ValueError("ProsocialDialog authorization text binding changed")
    if plan.get("source_after") != source_metadata():
        raise ValueError("ProsocialDialog promoted source metadata changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("ProsocialDialog promotion bindings missing")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid ProsocialDialog promotion binding: {name}")
        resolved = (root / str(binding.get("path"))).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError(f"ProsocialDialog promotion binding missing: {name}")
        if file_sha256(resolved) != binding.get("sha256"):
            raise ValueError(f"ProsocialDialog promotion binding changed: {name}")
    exact = {
        "candidate": EXPECTED_CANDIDATE_SHA256,
        "curation_report": EXPECTED_CURATION_REPORT_SHA256,
        "curation_verification": EXPECTED_CURATION_VERIFICATION_SHA256,
        "owner_review_receipt": EXPECTED_OWNER_RECEIPT_SHA256,
        "owner_review_verification": EXPECTED_OWNER_VERIFICATION_SHA256,
        "predecessor_registry": EXPECTED_PREDECESSOR_REGISTRY_SHA256,
    }
    for name, expected in exact.items():
        if bindings.get(name, {}).get("sha256") != expected:
            raise ValueError(f"ProsocialDialog exact promotion binding changed: {name}")
    return plan


def project_successor_registry(predecessor: Mapping[str, Any]) -> dict[str, Any]:
    sources = predecessor.get("sources")
    if not isinstance(sources, list) or len(sources) != 22:
        raise ValueError("source registry v0.18 inventory changed")
    if any(source.get("source_id") == SOURCE_ID for source in sources):
        raise ValueError("ProsocialDialog source already exists")
    successor = dict(predecessor)
    successor["sources"] = [*sources, source_metadata()]
    encoded = json_file_bytes(successor)
    if len(encoded) != EXPECTED_SUCCESSOR_REGISTRY_BYTES:
        raise ValueError("registry v0.19 projected byte size changed")
    if hashlib.sha256(encoded).hexdigest() != EXPECTED_SUCCESSOR_REGISTRY_SHA256:
        raise ValueError("registry v0.19 projected SHA-256 changed")
    return successor


def validate_candidate_and_evidence(
    plan: Mapping[str, Any], repository_root: Path
) -> dict[str, Any]:
    bindings = plan["bindings"]
    candidate_path = repository_root / bindings["candidate"]["path"]
    if candidate_path.stat().st_size != EXPECTED_CANDIDATE_BYTES:
        raise ValueError("ProsocialDialog candidate byte size changed")
    rows = _read_jsonl(candidate_path)
    if len(rows) != EXPECTED_RECORDS:
        raise ValueError("ProsocialDialog candidate row count changed")
    ids: set[str] = set()
    by_split: Counter[str] = Counter()
    for row in rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or record_id in ids:
            raise ValueError("ProsocialDialog candidate record ID changed or repeated")
        ids.add(record_id)
        by_split[str(row.get("split"))] += 1
        source = row.get("source")
        provenance = row.get("provenance")
        if not isinstance(source, Mapping) or (
            source.get("source_id") != SOURCE_ID
            or source.get("dataset_id") != DATASET_ID
            or source.get("revision") != REVISION
            or source.get("license_id") != LICENSE_ID
        ):
            raise ValueError("ProsocialDialog candidate source binding changed")
        if not isinstance(provenance, Mapping) or (
            provenance.get("origin_label")
            != "mixed_human_ai_assisted_existing_public_dataset"
            or provenance.get("prompt_origin")
            != "gpt_3_generated_or_extended_seed_source"
            or provenance.get("assistant_response_origin") != "human_crowdworker"
        ):
            raise ValueError("ProsocialDialog candidate provenance changed")
        if row.get("task_category") != "safety_helpfulness" or row.get("language") != "en":
            raise ValueError("ProsocialDialog candidate task or language changed")
    if by_split != Counter(train=EXPECTED_TRAIN_RECORDS, validation=EXPECTED_VALIDATION_RECORDS):
        raise ValueError("ProsocialDialog candidate split counts changed")

    report = _load_json(repository_root / bindings["curation_report"]["path"])
    verification = _load_json(
        repository_root / bindings["curation_verification"]["path"]
    )
    owner = _load_json(repository_root / bindings["owner_review_receipt"]["path"])
    owner_verification = _load_json(
        repository_root / bindings["owner_review_verification"]["path"]
    )
    if (
        report.get("status") != "PASS_nonproduction_candidate_owner_review_pending"
        or report.get("tokens", {}).get("accepted_records") != EXPECTED_RECORDS
        or report.get("tokens", {}).get("train_supervised_tokens") != EXPECTED_TRAIN_TOKENS
        or report.get("tokens", {}).get("validation_supervised_tokens")
        != EXPECTED_VALIDATION_TOKENS
        or verification.get("status")
        != "PASS_independent_nonproduction_candidate_verification"
        or owner.get("status") != "owner_accepted_all_32_prosocial_dialog_rows"
        or owner.get("accepted_records") != 32
        or owner.get("rejected_records") != 0
        or owner_verification.get("status")
        != "PASS_owner_accepted_all_32_prosocial_dialog_rows"
    ):
        raise ValueError("ProsocialDialog candidate evidence changed")
    return {
        "file_sha256": EXPECTED_CANDIDATE_SHA256,
        "internal_sha256": sha256_value(rows),
        "records": len(rows),
        "train_records": by_split["train"],
        "train_supervised_tokens": EXPECTED_TRAIN_TOKENS,
        "validation_records": by_split["validation"],
        "validation_supervised_tokens": EXPECTED_VALIDATION_TOKENS,
    }


def build_receipt(
    plan: Mapping[str, Any], *, plan_file_sha256: str, repository_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = validate_candidate_and_evidence(plan, repository_root)
    predecessor_path = repository_root / plan["bindings"]["predecessor_registry"]["path"]
    predecessor = _load_json(predecessor_path)
    successor = project_successor_registry(predecessor)
    receipt: dict[str, Any] = {
        "authorization": dict(EXPECTED_AUTHORIZATION),
        "authorization_text": plan["authorization_text"],
        "authorization_text_sha256": plan["authorization_text_sha256"],
        "bindings": {
            "candidate_file_sha256": EXPECTED_CANDIDATE_SHA256,
            "curation_report_file_sha256": EXPECTED_CURATION_REPORT_SHA256,
            "curation_verification_file_sha256": EXPECTED_CURATION_VERIFICATION_SHA256,
            "owner_review_receipt_file_sha256": EXPECTED_OWNER_RECEIPT_SHA256,
            "owner_review_verification_file_sha256": EXPECTED_OWNER_VERIFICATION_SHA256,
            "plan_file_sha256": plan_file_sha256,
            "predecessor_registry_file_sha256": EXPECTED_PREDECESSOR_REGISTRY_SHA256,
        },
        "candidate": {
            **candidate,
            "bytes": EXPECTED_CANDIDATE_BYTES,
            "path": plan["bindings"]["candidate"]["path"],
            "reused_without_regeneration_or_rematerialization": True,
        },
        "checks": {
            "candidate_mutated": False,
            "data_contract_changed": False,
            "dataset_sealed": False,
            "external_api_used": False,
            "full_candidate_mix_reaudit_run": False,
            "gpu_used": False,
            "network_used": False,
            "predecessor_registry_mutated": False,
            "production_promoted": False,
            "sft_started": False,
            "source_registry_promotions": 1,
            "truthful_mixed_provenance_preserved": True,
        },
        "next_candidate_mix_projection": {
            "included_candidate_sources": 16,
            "input_records": 309_838,
            "train_supervised_tokens": 9_553_473,
            "validation_supervised_tokens": 578_618,
        },
        "observed_at": "2026-08-13",
        "production_eligible": False,
        "registry": {
            "added_source_id": SOURCE_ID,
            "predecessor_path": plan["bindings"]["predecessor_registry"]["path"],
            "predecessor_sha256": EXPECTED_PREDECESSOR_REGISTRY_SHA256,
            "source_count": 23,
            "successor_bytes": EXPECTED_SUCCESSOR_REGISTRY_BYTES,
            "successor_path": plan["output"]["successor_registry_path"],
            "successor_sha256": EXPECTED_SUCCESSOR_REGISTRY_SHA256,
        },
        "remaining_blockers": [
            "full_registry_v0_19_cross_source_hygiene_audit_not_complete",
            "accepted_train_supervised_tokens_below_10m_floor",
            "remaining_task_and_language_floors_not_met",
            "production_dataset_seal_and_sft_authorization_not_complete",
        ],
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "verified_prosocial_dialog_candidate_and_registry_v0_19_nonproduction",
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    return receipt, successor


def build_verification(
    *,
    plan: Mapping[str, Any],
    receipt: Mapping[str, Any],
    receipt_file_sha256: str,
    repository_root: Path,
) -> dict[str, Any]:
    predecessor_path = repository_root / plan["bindings"]["predecessor_registry"]["path"]
    successor_path = repository_root / plan["output"]["successor_registry_path"]
    predecessor = _load_json(predecessor_path)
    successor = _load_json(successor_path)
    if successor.get("sources", [])[:-1] != predecessor.get("sources"):
        raise ValueError("registry v0.18 sources were not preserved exactly")
    if successor.get("sources", [])[-1] != source_metadata():
        raise ValueError("registry v0.19 appended source changed")
    if file_sha256(successor_path) != EXPECTED_SUCCESSOR_REGISTRY_SHA256:
        raise ValueError("registry v0.19 file hash changed")
    unsigned = dict(receipt)
    internal = unsigned.pop("receipt_sha256", None)
    if internal != sha256_value(unsigned):
        raise ValueError("ProsocialDialog promotion receipt seal changed")

    policy = load_sft_policy(repository_root / "configs/lumi-4b-sft-data-contract-v0.1.json")
    registry = load_sft_source_registry(
        successor_path,
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
        expected_sha256=EXPECTED_SUCCESSOR_REGISTRY_SHA256,
    )
    if len(registry.sources) != 23 or registry.sources[-1].source_id != SOURCE_ID:
        raise ValueError("registry v0.19 source inventory changed")

    candidate = validate_candidate_and_evidence(plan, repository_root)
    verification: dict[str, Any] = {
        "candidate": {
            **candidate,
            "bytes": EXPECTED_CANDIDATE_BYTES,
        },
        "checks": {
            "candidate_and_evidence_rebound": True,
            "candidate_reused_without_copy_or_mutation": True,
            "data_contract_unchanged": True,
            "dataset_remains_unsealed": True,
            "no_external_api_network_gpu_remote_or_sft_action": True,
            "predecessor_registry_unchanged": True,
            "promotion_receipt_internal_seal_verified": True,
            "raw_training_text_absent_from_receipt_and_verification": True,
            "successor_is_exact_append_only_23_source_registry": True,
            "truthful_mixed_provenance_preserved": True,
        },
        "observed_at": "2026-08-13",
        "production_eligible": False,
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_internal_sha256": internal,
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "PASS_verified_prosocial_dialog_candidate_and_registry_v0_19_nonproduction",
        "successor_registry": {
            "bytes": successor_path.stat().st_size,
            "file_sha256": file_sha256(successor_path),
            "source_count": len(registry.sources),
        },
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


def build_reaudit_draft(
    *,
    plan: Mapping[str, Any],
    receipt_file_sha256: str,
    verification_file_sha256: str,
) -> dict[str, Any]:
    bindings = plan["bindings"]
    return {
        "authorization": {
            "candidate_mutation_authorized": False,
            "dataset_seal_authorized": False,
            "external_api_authorized": False,
            "full_local_reaudit_authorized": False,
            "gpu_authorized": False,
            "network_authorized": False,
            "production_promotion_authorized": False,
            "registry_change_authorized": False,
            "sft_training_authorized": False,
        },
        "bindings": {
            "benchmark_registry": {
                "path": "data/benchmark-signatures-v0.3.jsonl.gz",
                "sha256": "1ffb5e938ab28c2fc876b9efd6c6ed59059f1e85aed2c396c024ae5925a4745e",
            },
            "predecessor_cross_source_report": {
                "path": bindings["cross_source_report"]["path"],
                "sha256": EXPECTED_CROSS_SOURCE_REPORT_SHA256,
            },
            "predecessor_cross_source_verification": {
                "path": bindings["cross_source_verification"]["path"],
                "sha256": EXPECTED_CROSS_SOURCE_VERIFICATION_SHA256,
            },
            "prosocial_candidate": {
                "path": bindings["candidate"]["path"],
                "sha256": EXPECTED_CANDIDATE_SHA256,
            },
            "prosocial_promotion_receipt": {
                "path": plan["output"]["receipt_path"],
                "sha256": receipt_file_sha256,
            },
            "prosocial_promotion_verification": {
                "path": plan["output"]["verification_path"],
                "sha256": verification_file_sha256,
            },
            "source_registry": {
                "path": plan["output"]["successor_registry_path"],
                "sha256": EXPECTED_SUCCESSOR_REGISTRY_SHA256,
            },
        },
        "expected": {
            "included_candidate_sources": 16,
            "input_records": 309_838,
            "input_train_supervised_tokens": 9_553_473,
            "input_validation_supervised_tokens": 578_618,
            "registry_sources": 23,
        },
        "limits": {
            "maximum_candidate_records_per_pass": 309_838,
            "maximum_full_audit_passes": 2,
            "maximum_network_bytes": 0,
            "maximum_private_write_bytes": 2_097_152,
            "maximum_runtime_minutes": 180,
            "maximum_total_candidate_row_reads": 619_676,
        },
        "observed_at": "2026-08-13",
        "output": {
            "production_promotion_permitted": False,
            "raw_training_text_in_outputs": False,
            "report_path": "evidence/lumi-4b-sft-candidate-mix-reaudit-v0.13.json",
            "verification_path": (
                "evidence/lumi-4b-sft-candidate-mix-reaudit-verification-v0.13.json"
            ),
        },
        "schema_version": DRAFT_SCHEMA_VERSION,
        "status": "draft_requires_exact_owner_authorization",
    }


def promote(plan_path: str | Path, repository_root: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(repository_root).resolve()
    plan = load_plan(plan_path, root)
    receipt, successor = build_receipt(
        plan, plan_file_sha256=file_sha256(plan_path), repository_root=root
    )
    successor_path = root / plan["output"]["successor_registry_path"]
    receipt_path = root / plan["output"]["receipt_path"]
    verification_path = root / plan["output"]["verification_path"]
    draft_path = root / plan["output"]["reaudit_draft_path"]
    _write_bytes_atomic(successor_path, json_file_bytes(successor))
    _write_json_atomic(receipt_path, receipt)
    verification = build_verification(
        plan=plan,
        receipt=receipt,
        receipt_file_sha256=file_sha256(receipt_path),
        repository_root=root,
    )
    _write_json_atomic(verification_path, verification)
    draft = build_reaudit_draft(
        plan=plan,
        receipt_file_sha256=file_sha256(receipt_path),
        verification_file_sha256=file_sha256(verification_path),
    )
    _write_json_atomic(draft_path, draft)
    written_bytes = sum(
        path.stat().st_size
        for path in (successor_path, receipt_path, verification_path, draft_path)
    )
    if written_bytes > EXPECTED_LIMITS["maximum_private_write_bytes"]:
        raise ValueError("ProsocialDialog promotion private write cap exceeded")
    return receipt, verification, draft


__all__ = [
    "build_reaudit_draft",
    "build_receipt",
    "build_verification",
    "json_file_bytes",
    "load_plan",
    "project_successor_registry",
    "promote",
    "source_metadata",
    "validate_candidate_and_evidence",
]
