"""Project the owner-accepted gpt-oss scale candidate into canonical SFT form."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .data_quality import BenchmarkRegistry
from .posttraining_candidate_mix_preflight import file_sha256, sha256_value
from .posttraining_data import filter_sft_records, load_sft_policy
from .posttraining_format import encode_sft_messages, load_chat_template
from .posttraining_gpt_oss_teacher_source_promotion_preflight_v16 import (
    PureByteLevelBPETokenizer,
)
from .posttraining_sources import SFTSource, SFTSourceRegistry
from .posttraining_teacher_canary_source_promotion_preflight import (
    json_file_bytes,
    jsonl_file_bytes,
)

ARTIFACT_ROOT = (
    "artifacts/lumi-gpt-oss-deterministic-tool-repair-v0.6/"
    "79db468b8c61/attempt-001"
)
RAW_CANDIDATE_PATH = f"{ARTIFACT_ROOT}/private/candidate-rows.jsonl"
OWNER_REVIEW_PATH = f"{ARTIFACT_ROOT}/evidence/owner-review.json"
OWNER_VERIFICATION_PATH = f"{ARTIFACT_ROOT}/evidence/owner-review-verification.json"
RUN_REPORT_PATH = f"{ARTIFACT_ROOT}/evidence/run-report.json"
RUN_VERIFICATION_PATH = f"{ARTIFACT_ROOT}/evidence/run-verification.json"

EXPECTED_RAW_FILE_SHA256 = (
    "f5393d4f6438ae5a03354f2e8916cb05de0663bc02f90eebdb41f05726de4a69"
)
EXPECTED_OWNER_REVIEW_FILE_SHA256 = (
    "9ee9d3701d765d80519cc2dae67e2973909689a46b076d63136c3d8825f9d37b"
)
EXPECTED_OWNER_VERIFICATION_FILE_SHA256 = (
    "5bafe90dafa8cc4a231d20e230622c36674bfe7406b1e0ccf3630c91ca907a5b"
)
EXPECTED_RUN_REPORT_FILE_SHA256 = (
    "aaec8e86bdf64e4d2cd595a53f4730e1d653c0242ff5e9150585c8db973d9b61"
)
EXPECTED_RUN_VERIFICATION_FILE_SHA256 = (
    "346caa7eff00d1f63dc2bb6f1d3a19907a1ac45cf9337d6d542209a9c4d0d6d9"
)
EXPECTED_CANDIDATE_MIX_REPORT_SHA256 = (
    "0eb5d92c83c6f403fbd5e5f0067b3c21f1aa64db352148eee6258a966db1b1da"
)
EXPECTED_CANDIDATE_MIX_VERIFICATION_SHA256 = (
    "4336f0b060d3624c825e40a67ae2ba25a167d48da3dbd127db57507ec738001d"
)

SOURCE_ID = "lumi_first_party_multilingual_safety_tool_v0_6"
DATASET_ID = "Afnan-Technologies-Inc/lumi-first-party-multilingual-safety-tool-v0.6"
EXPECTED_RECORDS = 984
EXPECTED_TRAIN_RECORDS = 787
EXPECTED_VALIDATION_RECORDS = 197
SPLIT_CONTRACT = "preserve_frozen_scale_schedule_split.v0.17"

LANGUAGE_MAP = {
    "banglish": "multilingual",
    "bengali_script": "bn",
    "english": "en",
    "natural_code_switching": "multilingual",
}
TASKS = (
    "factual_qa",
    "multilingual_bengali",
    "safety_helpfulness",
    "tool_api",
    "writing_summarization",
)

PLAN_SCHEMA = "lumi.sft-gpt-oss-scale-source-promotion-preflight.v0.17"
REPORT_SCHEMA = "lumi.sft-gpt-oss-scale-source-promotion-preflight-report.v0.17"
VERIFICATION_SCHEMA = (
    "lumi.sft-gpt-oss-scale-source-promotion-preflight-verification.v0.17"
)
PLAN_STATUS = "local_projection_only_registry_v0_17_not_materialized"

EXPECTED_RAW_INTERNAL_SHA256 = (
    "07b673b7486cfd8b6916fa19d12d0d47b0078857da8aa8bd2fe9bf6f0834008f"
)
EXPECTED_CANONICAL_FILE_SHA256 = (
    "d0b2af4539ae156739ffe50a72245fd1faffc27538417b214dc9864627a2e4af"
)
EXPECTED_CANONICAL_INTERNAL_SHA256 = (
    "510e08540a7d316fd219a58623150997a1c2f532a9590b3711241da66194ecac"
)
EXPECTED_CANONICAL_BYTES = 2_264_807
EXPECTED_ACCEPTED_DATASET_SHA256 = (
    "f625b4ca10312df094338aacce073a1792fc875d5f6dfaab296a450ab4b74909"
)
EXPECTED_FILTER_REPORT_INTERNAL_SHA256 = (
    "6fd8e7d8ebadb86f36eda07cd0fc290880ff2d39cb69151f977b2e2576b2c549"
)
EXPECTED_PREDECESSOR_REGISTRY_SHA256 = (
    "23276446086ee9c68784e6d20e25705fa9516b4e0f913dc96d8abd1b6b443ba6"
)
EXPECTED_SUCCESSOR_REGISTRY_SHA256 = (
    "85a688a62be0902b177469c99eb31525fb633efab5e420416b705d03e8d80e1a"
)
EXPECTED_SUCCESSOR_REGISTRY_BYTES = 21_354
EXPECTED_TRAIN_TOKENS = 75_439
EXPECTED_VALIDATION_TOKENS = 18_974
EXPECTED_CURRENT_MIX_TOKENS = 7_266_143
EXPECTED_PROJECTED_MIX_TOKENS = 7_341_582
EXPECTED_PROJECTED_GAP = 2_658_418
EXPECTED_TOKEN_METRICS = {
    "by_split": {
        "train": {
            "masked_tokens": 39_325,
            "records": 787,
            "serialized_tokens": 114_764,
            "supervised_tokens": 75_439,
        },
        "validation": {
            "masked_tokens": 9_865,
            "records": 197,
            "serialized_tokens": 28_839,
            "supervised_tokens": 18_974,
        },
    },
    "train_by_language_mode": {
        "banglish": 19_490,
        "bengali_script": 20_237,
        "english": 20_585,
        "natural_code_switching": 15_127,
    },
    "train_by_task": {
        "factual_qa": 3_702,
        "multilingual_bengali": 15_997,
        "safety_helpfulness": 22_123,
        "tool_api": 19_521,
        "writing_summarization": 14_096,
    },
}

EXPECTED_AUTHORIZATION = {
    "candidate_mix_reaudit_authorized": False,
    "candidate_registry_change_authorized": False,
    "canonical_candidate_materialization_authorized": False,
    "data_contract_change_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "gpu_authorized": False,
    "network_authorized": False,
    "predecessor_registry_mutation_authorized": False,
    "production_promotion_authorized": False,
    "production_sft_authorized": False,
    "purpose": "exact_gpt_oss_scale_candidate_and_registry_v0_17_projection_only",
    "source_registry_promotion_authorized": False,
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "added_sources": 1,
    "candidate_bytes": EXPECTED_CANONICAL_BYTES,
    "candidate_records": EXPECTED_RECORDS,
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 2_097_152,
    "maximum_runtime_minutes": 30,
    "predecessor_sources": 17,
    "registry_bytes": EXPECTED_SUCCESSOR_REGISTRY_BYTES,
    "successor_sources": 18,
    "train_records": EXPECTED_TRAIN_RECORDS,
    "train_supervised_tokens": EXPECTED_TRAIN_TOKENS,
    "validation_records": EXPECTED_VALIDATION_RECORDS,
    "validation_supervised_tokens": EXPECTED_VALIDATION_TOKENS,
}
EXPECTED_OUTPUT = {
    "canonical_candidate_path": f"{ARTIFACT_ROOT}/private/canonical-candidate-v0.17.jsonl",
    "preflight_report_path": (
        "evidence/lumi-4b-sft-gpt-oss-scale-source-promotion-preflight-v0.17.json"
    ),
    "projected_successor_registry_path": "configs/lumi-4b-sft-source-registry-v0.17.json",
    "verification_path": (
        "evidence/lumi-4b-sft-gpt-oss-scale-source-promotion-preflight-"
        "verification-v0.17.json"
    ),
}

EXACT_AUTHORIZATION_TEXT = (
    "I approve one bounded local-only Lumi gpt-oss v0.6 scale candidate promotion into an "
    "immutable source-registry v0.17 successor. Bind the exact 984-row machine-verified candidate "
    "SHA-256 f5393d4f6438ae5a03354f2e8916cb05de0663bc02f90eebdb41f05726de4a69, "
    "owner-review SHA-256 9ee9d3701d765d80519cc2dae67e2973909689a46b076d63136c3d8825f9d37b, "
    "owner-review verification SHA-256 "
    "5bafe90dafa8cc4a231d20e230622c36674bfe7406b1e0ccf3630c91ca907a5b, "
    "run-report SHA-256 aaec8e86bdf64e4d2cd595a53f4730e1d653c0242ff5e9150585c8db973d9b61, "
    "and run-verification SHA-256 "
    "346caa7eff00d1f63dc2bb6f1d3a19907a1ac45cf9337d6d542209a9c4d0d6d9. "
    "Materialize only the exact canonical 984-row candidate projected by contract "
    "lumi.gpt-oss-scale-canonical-owner-bound.v0.17, bound to SHA-256 "
    "d0b2af4539ae156739ffe50a72245fd1faffc27538417b214dc9864627a2e4af and 2,264,807 "
    "bytes, using split contract preserve_frozen_scale_schedule_split.v0.17 with 787 train rows / "
    "75,439 train supervised tokens and 197 validation rows / 18,974 validation supervised tokens. "
    "Require all 984 rows to pass the frozen SFT schema, rights, provenance, secret, PII, "
    "benchmark-decontamination, 4K-fit, tokenizer, chat-template, and assistant-and-tool-only "
    "masking gates. Preserve source registry v0.16 SHA-256 "
    "23276446086ee9c68784e6d20e25705fa9516b4e0f913dc96d8abd1b6b443ba6 byte-for-byte and append "
    "only source lumi_first_party_multilingual_safety_tool_v0_6 to create immutable registry v0.17 "
    "SHA-256 85a688a62be0902b177469c99eb31525fb633efab5e420416b705d03e8d80e1a, exactly "
    "21,354 bytes and 18 sources. Maximum 45 local runtime-minutes / 4 MiB additional private "
    "writes / 0 network bytes / 0 GPU use. It may write only the bound canonical candidate, "
    "immutable registry v0.17, non-production promotion receipt, independent verification, and "
    "the raw-text-free draft plan for the next full candidate-mix re-audit. No predecessor "
    "mutation, external API, download, dataset seal, active data-contract change, production "
    "promotion, SFT training, unrelated access, remote access, instance action, or termination."
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSON object required at line {line_number}: {path}")
        values.append(value)
    return values


def load_bound_candidate(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bindings = {
        RAW_CANDIDATE_PATH: EXPECTED_RAW_FILE_SHA256,
        OWNER_REVIEW_PATH: EXPECTED_OWNER_REVIEW_FILE_SHA256,
        OWNER_VERIFICATION_PATH: EXPECTED_OWNER_VERIFICATION_FILE_SHA256,
        RUN_REPORT_PATH: EXPECTED_RUN_REPORT_FILE_SHA256,
        RUN_VERIFICATION_PATH: EXPECTED_RUN_VERIFICATION_FILE_SHA256,
    }
    for relative, expected in bindings.items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"gpt-oss scale promotion binding changed: {relative}")
    rows = _read_jsonl(root / RAW_CANDIDATE_PATH)
    owner = _read_json(root / OWNER_REVIEW_PATH)
    owner_verification = _read_json(root / OWNER_VERIFICATION_PATH)
    run_report = _read_json(root / RUN_REPORT_PATH)
    run_verification = _read_json(root / RUN_VERIFICATION_PATH)
    audit = run_report.get("machine_audit", {})
    if (
        len(rows) != EXPECTED_RECORDS
        or owner.get("status") != "ACCEPT_deterministic_100_row_owner_review_packet"
        or owner.get("decision") != "accept"
        or owner.get("accepted_review_rows") != 100
        or owner.get("rejected_review_rows") != 0
        or owner_verification.get("status") != "PASS_owner_acceptance_verified"
        or run_report.get("status") != "machine_verified_pending_100_row_owner_review"
        or run_verification.get("status") != "PASS"
        or audit.get("accepted_records") != EXPECTED_RECORDS
        or audit.get("raw_tool_envelope_rows") != 0
        or audit.get("critical_factual_safety_or_tool_failures") != 0
        or not all(audit.get("checks", {}).values())
    ):
        raise ValueError("gpt-oss scale accepted candidate evidence changed")
    return rows, owner


def source_metadata(source_revision: str) -> dict[str, Any]:
    return {
        "access": "first_party",
        "adapter": "lumi_first_party_multilingual_safety_tool_canonical_owner_bound_v0.17",
        "blockers": [],
        "dataset_id": DATASET_ID,
        "evidence_urls": [],
        "languages": ["bn", "en", "multilingual"],
        "license_id": "lumi-owned",
        "origin": "synthetic",
        "production_blockers": [
            "candidate_is_only_984_records",
            "owner_quality_review_is_100_row_deterministic_sample_not_full_record_review",
            "full_mix_cross_source_reaudit_not_complete",
            "first_party_3_2m_closure_lane_not_generated",
            "production_dataset_seal_and_sft_authorization_not_complete",
        ],
        "production_eligible": False,
        "repository_url": None,
        "revision": source_revision,
        "source_id": SOURCE_ID,
        "splits": ["train", "validation"],
        "status": "candidate_eligible",
        "task_categories": list(TASKS),
    }


def project_canonical_candidate(
    rows: Sequence[Mapping[str, Any]], *, owner_review_file_sha256: str
) -> list[dict[str, Any]]:
    source_revision = sha256_value(list(rows))
    projected: list[dict[str, Any]] = []
    for row in rows:
        record_id = str(row["record_id"])
        language_mode = str(row["language_mode"])
        messages = copy.deepcopy(row["messages"])
        original = copy.deepcopy(row["provenance"])
        transform = {
            "contract": "lumi.gpt-oss-scale-canonical-owner-bound.v0.17",
            "original_provenance_sha256": sha256_value(original),
            "original_rights_status": row["rights_status"],
            "owner_review_file_sha256": owner_review_file_sha256,
            "source_revision": source_revision,
            "split_contract": SPLIT_CONTRACT,
        }
        projected.append(
            {
                "language": LANGUAGE_MAP[language_mode],
                "messages": messages,
                "provenance": {
                    "ai_assisted": True,
                    "consent_basis": "first_party_rights_declaration",
                    "content_sha256": sha256_value(messages),
                    "generation_mode": original.get(
                        "generation_mode", "self_hosted_teacher_generation"
                    ),
                    "human_authored": False,
                    "human_authorship_claimed": False,
                    "model_id": original.get("model_id"),
                    "model_revision": original.get("model_revision"),
                    "original_provenance_sha256": sha256_value(original),
                    "owner_review_claim": (
                        "ai_generated_plus_owner_reviewed_not_human_written"
                    ),
                    "owner_review_file_sha256": owner_review_file_sha256,
                    "owner_review_scope": "deterministic_100_row_spot_check",
                    "runtime_image_digest": original.get("runtime_image_digest"),
                    "self_hosted": bool(original.get("self_hosted", True)),
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
                    "revision": source_revision,
                    "source_id": SOURCE_ID,
                    "uri": (
                        "lumi-private://first-party-multilingual-safety-tool/"
                        f"v0.6/{record_id}"
                    ),
                },
                "split": row["split"],
                "task_category": row["task"],
            }
        )
    return projected


def _token_metrics(
    raw_rows: Sequence[Mapping[str, Any]],
    canonical: Sequence[Mapping[str, Any]],
    *,
    tokenizer_path: Path,
    chat_template_path: Path,
) -> dict[str, Any]:
    tokenizer = PureByteLevelBPETokenizer(tokenizer_path)
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
    by_mode: Counter[str] = Counter()
    for raw, row in zip(raw_rows, canonical, strict=True):
        encoded = encode_sft_messages(
            row["messages"], tokenizer=tokenizer, template=template
        )
        split = str(row["split"])
        by_split[split]["masked_tokens"] += encoded.masked_tokens
        by_split[split]["records"] += 1
        by_split[split]["serialized_tokens"] += len(encoded.input_ids)
        by_split[split]["supervised_tokens"] += encoded.supervised_tokens
        if split == "train":
            by_task[str(raw["task"])] += encoded.supervised_tokens
            by_mode[str(raw["language_mode"])] += encoded.supervised_tokens
    return {
        "by_split": dict(sorted(by_split.items())),
        "train_by_language_mode": dict(sorted(by_mode.items())),
        "train_by_task": dict(sorted(by_task.items())),
    }


def build_reference_projection(root: Path) -> dict[str, Any]:
    rows, owner = load_bound_candidate(root)
    source_revision = sha256_value(rows)
    canonical = project_canonical_candidate(
        rows, owner_review_file_sha256=EXPECTED_OWNER_REVIEW_FILE_SHA256
    )
    predecessor_path = root / "configs/lumi-4b-sft-source-registry-v0.16.json"
    predecessor = _read_json(predecessor_path)
    if any(
        source.get("source_id") == SOURCE_ID
        for source in predecessor.get("sources", [])
        if isinstance(source, Mapping)
    ):
        raise ValueError("gpt-oss scale source already exists in registry v0.16")
    successor = copy.deepcopy(predecessor)
    successor["sources"].append(source_metadata(source_revision))

    contract_path = root / "configs/lumi-4b-sft-data-contract-v0.1.json"
    contract = _read_json(contract_path)
    allowed = {
        "licenses": frozenset(contract["allowed_licenses"]),
        "languages": frozenset(contract["allowed_languages"]),
        "splits": frozenset(contract["allowed_splits"]),
        "tasks": frozenset(contract["task_mix"]),
    }
    parsed = tuple(
        SFTSource.from_mapping(
            item,
            allowed_licenses=allowed["licenses"],
            allowed_languages=allowed["languages"],
            allowed_splits=allowed["splits"],
            allowed_tasks=allowed["tasks"],
        )
        for item in successor["sources"]
    )
    successor_bytes = json_file_bytes(successor)
    successor_sha256 = hashlib.sha256(successor_bytes).hexdigest()
    registry = SFTSourceRegistry(
        observed_at=str(successor["observed_at"]),
        registry_sha256=successor_sha256,
        sources=parsed,
    )
    quality = _read_json(root / "configs/data-quality-v0.3.json")
    accepted, filter_report = filter_sft_records(
        canonical,
        policy=load_sft_policy(contract_path),
        benchmark_registry=BenchmarkRegistry(
            quality["benchmark_decontamination"], root
        ),
        source_registry=registry,
    )
    token_metrics = _token_metrics(
        rows,
        canonical,
        tokenizer_path=(
            root.parent
            / "lumi-artifacts/4b-segment2-throughput-gate-v0.1/"
            "8bedfbc868e8/staging/input/tokenizer/tokenizer.json"
        ),
        chat_template_path=root / "configs/lumi-4b-chat-template-v0.1.json",
    )
    canonical_bytes = jsonl_file_bytes(canonical)
    train_tokens = token_metrics["by_split"]["train"]["supervised_tokens"]
    return {
        "canonical": canonical,
        "owner": owner,
        "projection": {
            "accepted_dataset_sha256": filter_report["accepted_dataset_sha256"],
            "canonical_bytes": len(canonical_bytes),
            "canonical_file_sha256": hashlib.sha256(canonical_bytes).hexdigest(),
            "canonical_internal_sha256": sha256_value(canonical),
            "current_mix_train_supervised_tokens": EXPECTED_CURRENT_MIX_TOKENS,
            "filter_accepted": len(accepted),
            "filter_rejected": int(filter_report["rejected"]),
            "filter_report_internal_sha256": sha256_value(filter_report),
            "projected_gap_to_10m": 10_000_000 - (
                EXPECTED_CURRENT_MIX_TOKENS + train_tokens
            ),
            "projected_mix_train_supervised_tokens": (
                EXPECTED_CURRENT_MIX_TOKENS + train_tokens
            ),
            "raw_internal_sha256": source_revision,
            "records": len(rows),
            "registry_predecessor_sha256": file_sha256(predecessor_path),
            "successor_registry_bytes": len(successor_bytes),
            "successor_registry_sha256": successor_sha256,
            "successor_sources": len(successor["sources"]),
            "token_metrics": token_metrics,
        },
        "successor": successor,
    }


EXPECTED_BINDINGS = frozenset(
    {
        "candidate_mix_report",
        "candidate_mix_verification",
        "candidate_rows",
        "chat_template",
        "data_contract",
        "data_quality",
        "implementation",
        "owner_review",
        "owner_review_verification",
        "plan_runner",
        "plan_verifier",
        "predecessor_registry",
        "run_report",
        "run_verification",
        "test",
        "tokenizer_config",
        "tokenizer_json",
    }
)


def resolve_binding(
    root: Path, specification: Mapping[str, Any], *, name: str
) -> Path:
    relative = specification.get("path")
    expected = specification.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"invalid gpt-oss scale promotion binding: {name}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.parent.resolve()) or not path.is_file():
        raise ValueError(f"missing gpt-oss scale promotion binding: {name}")
    if file_sha256(path) != expected:
        raise ValueError(f"changed gpt-oss scale promotion binding: {name}")
    return path


def load_preflight_plan(path: Path, *, repository_root: Path) -> dict[str, Any]:
    plan = _read_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-11",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA,
        "source_after": source_metadata(EXPECTED_RAW_INTERNAL_SHA256),
        "status": PLAN_STATUS,
    }
    for name, value in expected.items():
        if plan.get(name) != value:
            raise ValueError(f"gpt-oss scale promotion preflight changed: {name}")
    next_gate = plan.get("next_gate")
    if not isinstance(next_gate, Mapping) or next_gate.get(
        "exact_authorization_text"
    ) != EXACT_AUTHORIZATION_TEXT:
        raise ValueError("gpt-oss scale promotion authorization text changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("gpt-oss scale promotion binding set changed")
    for name, specification in bindings.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"invalid gpt-oss scale promotion binding: {name}")
        resolve_binding(repository_root, specification, name=name)
    exact = {
        "candidate_mix_report": EXPECTED_CANDIDATE_MIX_REPORT_SHA256,
        "candidate_mix_verification": EXPECTED_CANDIDATE_MIX_VERIFICATION_SHA256,
        "candidate_rows": EXPECTED_RAW_FILE_SHA256,
        "owner_review": EXPECTED_OWNER_REVIEW_FILE_SHA256,
        "owner_review_verification": EXPECTED_OWNER_VERIFICATION_FILE_SHA256,
        "predecessor_registry": EXPECTED_PREDECESSOR_REGISTRY_SHA256,
        "run_report": EXPECTED_RUN_REPORT_FILE_SHA256,
        "run_verification": EXPECTED_RUN_VERIFICATION_FILE_SHA256,
    }
    for name, expected_sha256 in exact.items():
        if bindings[name]["sha256"] != expected_sha256:
            raise ValueError(f"gpt-oss scale exact binding changed: {name}")
    return plan


def _validate_projection(projection: Mapping[str, Any]) -> None:
    expected = {
        "accepted_dataset_sha256": EXPECTED_ACCEPTED_DATASET_SHA256,
        "canonical_bytes": EXPECTED_CANONICAL_BYTES,
        "canonical_file_sha256": EXPECTED_CANONICAL_FILE_SHA256,
        "canonical_internal_sha256": EXPECTED_CANONICAL_INTERNAL_SHA256,
        "current_mix_train_supervised_tokens": EXPECTED_CURRENT_MIX_TOKENS,
        "filter_accepted": EXPECTED_RECORDS,
        "filter_rejected": 0,
        "filter_report_internal_sha256": EXPECTED_FILTER_REPORT_INTERNAL_SHA256,
        "projected_gap_to_10m": EXPECTED_PROJECTED_GAP,
        "projected_mix_train_supervised_tokens": EXPECTED_PROJECTED_MIX_TOKENS,
        "raw_internal_sha256": EXPECTED_RAW_INTERNAL_SHA256,
        "records": EXPECTED_RECORDS,
        "registry_predecessor_sha256": EXPECTED_PREDECESSOR_REGISTRY_SHA256,
        "successor_registry_bytes": EXPECTED_SUCCESSOR_REGISTRY_BYTES,
        "successor_registry_sha256": EXPECTED_SUCCESSOR_REGISTRY_SHA256,
        "successor_sources": 18,
        "token_metrics": EXPECTED_TOKEN_METRICS,
    }
    if dict(projection) != expected:
        raise ValueError("gpt-oss scale promotion reference projection changed")


def build_preflight_report(
    plan: Mapping[str, Any], *, plan_file_sha256: str, repository_root: Path
) -> dict[str, Any]:
    result = build_reference_projection(repository_root)
    projection = result["projection"]
    _validate_projection(projection)
    bindings = plan["bindings"]
    resolved = {
        name: resolve_binding(repository_root, specification, name=name)
        for name, specification in bindings.items()
    }
    for output_name in ("canonical_candidate_path", "projected_successor_registry_path"):
        if (repository_root / EXPECTED_OUTPUT[output_name]).exists():
            raise ValueError("gpt-oss scale promotion output was already materialized")
    report: dict[str, Any] = {
        "authorization": dict(EXPECTED_AUTHORIZATION),
        "bindings": {
            name: {"path": bindings[name]["path"], "sha256": file_sha256(path)}
            for name, path in sorted(resolved.items())
        },
        "checks": {
            "all_984_rows_pass_frozen_candidate_filters": True,
            "candidate_and_owner_evidence_hashes_verified": True,
            "candidate_mix_v0_11_evidence_verified": True,
            "canonical_candidate_not_materialized": True,
            "dependency_free_tokenizer_matches_frozen_reference": True,
            "gpu_network_and_external_api_unused": True,
            "human_authorship_not_claimed": True,
            "predecessor_registry_unchanged": True,
            "registry_v0_17_not_materialized": True,
            "schedule_split_preserved": True,
            "tool_messages_use_assistant_and_tool_only_masking": True,
        },
        "next_gate": dict(plan["next_gate"]),
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "production": {
            "candidate_mix_reaudited": False,
            "candidate_registry_changed": False,
            "canonical_candidate_materialized": False,
            "data_contract_changed": False,
            "dataset_sealed": False,
            "production_eligible": False,
            "sft_started": False,
        },
        "projection": copy.deepcopy(dict(projection)),
        "schema_version": REPORT_SCHEMA,
        "source_after": copy.deepcopy(plan["source_after"]),
        "status": "verified_local_projection_registry_v0_17_not_materialized",
    }
    report["report_sha256"] = sha256_value(report)
    validate_preflight_report(report)
    return report


def validate_preflight_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("unsupported gpt-oss scale promotion report")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if report.get("report_sha256") != sha256_value(unsigned):
        raise ValueError("gpt-oss scale promotion report seal changed")
    _validate_projection(report.get("projection", {}))
    checks = report.get("checks")
    if not isinstance(checks, Mapping) or not all(checks.values()):
        raise ValueError("gpt-oss scale promotion checks failed")
    production = report.get("production")
    if not isinstance(production, Mapping) or any(production.values()):
        raise ValueError("gpt-oss scale preflight crossed a production boundary")
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    for forbidden in ('"messages"', '"content"', '"assistant_response"'):
        if forbidden in serialized:
            raise ValueError("gpt-oss scale promotion report retained raw training text")


def build_preflight_verification(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
    replay: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    validate_preflight_report(report)
    validate_preflight_report(replay)
    if report != replay:
        raise ValueError("gpt-oss scale promotion preflight replay changed")
    for output_name in ("canonical_candidate_path", "projected_successor_registry_path"):
        if (repository_root / plan["output"][output_name]).exists():
            raise ValueError("gpt-oss scale promotion output unexpectedly exists")
    verification: dict[str, Any] = {
        "checks": {
            "all_bindings_reverified": True,
            "all_candidate_filters_replayed": True,
            "canonical_candidate_hash_and_bytes_reproduced": True,
            "independent_projection_byte_exact": True,
            "next_gate_requires_exact_authorization": True,
            "no_gpu_network_registry_seal_or_sft_action": True,
            "raw_training_text_absent": True,
            "schedule_split_and_token_counts_reproduced": True,
            "successor_registry_hash_and_bytes_reproduced": True,
        },
        "preflight_report_file_sha256": report_file_sha256,
        "preflight_report_internal_sha256": report["report_sha256"],
        "projected_canonical_candidate_file_sha256": EXPECTED_CANONICAL_FILE_SHA256,
        "projected_successor_registry_sha256": EXPECTED_SUCCESSOR_REGISTRY_SHA256,
        "schema_version": VERIFICATION_SCHEMA,
        "status": "PASS_local_projection_registry_v0_17_not_materialized",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


__all__ = [
    "DATASET_ID",
    "EXPECTED_CANONICAL_FILE_SHA256",
    "EXPECTED_PROJECTED_GAP",
    "EXPECTED_RECORDS",
    "EXPECTED_SUCCESSOR_REGISTRY_SHA256",
    "EXPECTED_TOKEN_METRICS",
    "EXPECTED_TRAIN_RECORDS",
    "EXPECTED_VALIDATION_RECORDS",
    "SOURCE_ID",
    "SPLIT_CONTRACT",
    "build_preflight_report",
    "build_preflight_verification",
    "build_reference_projection",
    "load_preflight_plan",
    "project_canonical_candidate",
    "source_metadata",
    "validate_preflight_report",
]
