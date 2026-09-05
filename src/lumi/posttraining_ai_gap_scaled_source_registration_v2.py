"""Register the scaled AI-assisted gap lane in immutable registry v0.24."""

from __future__ import annotations

import copy
import hashlib
import json
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import posttraining_ai_assisted_gap_closure_scaled_v2 as scaled
from .posttraining_data import load_sft_policy
from .posttraining_sources import SFTSource, load_sft_source_registry


PLAN_SCHEMA = "lumi.sft-ai-gap-scaled-source-registration-plan.v0.2"
RECEIPT_SCHEMA = "lumi.sft-ai-gap-scaled-source-registration-receipt.v0.2"
VERIFICATION_SCHEMA = "lumi.sft-ai-gap-scaled-source-registration-verification.v0.2"
PLAN_STATUS = "authorized_machine_only_candidate_registry_v0_24_successor"
RECEIPT_STATUS = "PASS_ai_gap_scaled_candidate_registered_in_v0_24_nonproduction"
VERIFICATION_STATUS = "PASS_verified_ai_gap_scaled_candidate_registry_v0_24_nonproduction"

AUTHORIZATION_TEXT = "ok next step shuru de"
AUTHORIZATION_BYTES = 21
AUTHORIZATION_SHA256 = "7a2317268ba301ceb4a30bb9aa558fad1796139ccc3f5e87722f1e944c4ddcd3"

SOURCE_ID = scaled.SOURCE_ID
SOURCE_REVISION = "ec06420bfa968df23fc287d9372549b6244c6bd00c917047613d2b05a3c79a93"
DATASET_ID = scaled.DATASET_ID
PREDECESSOR_SOURCES = 26
SUCCESSOR_SOURCES = 27

EXPECTED_CODE_PATHS = {
    "implementation": "src/lumi/posttraining_ai_gap_scaled_source_registration_v2.py",
    "runner": "scripts/register_sft_ai_gap_scaled_source_v2.py",
    "verifier": "scripts/verify_sft_ai_gap_scaled_source_registration_v2.py",
    "test": "tests/test_ai_gap_scaled_source_registration_v2.py",
}
EXPECTED_STATIC_BINDINGS = {
    "data_contract": {
        "path": "configs/lumi-4b-sft-data-contract-v0.1.json",
        "sha256": "9a283d9333d29deb7b516fa45eb4eec50232be9925f4ee4aa3a8f624213b2f06",
    },
    "predecessor_registry": {
        "path": "configs/lumi-4b-sft-source-registry-v0.23.json",
        "sha256": "81009e0c47fda396384264dd285370530cf4b25ef0e55c0aad69a5525b484e94",
    },
    "scaled_implementation": {
        "path": "src/lumi/posttraining_ai_assisted_gap_closure_scaled_v2.py",
        "sha256": "d6c6b440a2217398d29195e2f3de41b669a1db7fc78b32f4ea9b70217343e9fc",
    },
    "scaled_plan": {
        "path": "configs/lumi-4b-sft-ai-assisted-gap-closure-scaled-v0.2.json",
        "sha256": "8ba9bc6c1ca03bd5cbee5f24a5d6a31341e16fa6de5ed18067813f7ee0ecad3c",
    },
    "scaled_report": {
        "path": "evidence/lumi-4b-sft-ai-assisted-gap-closure-scaled-v0.2.json",
        "sha256": "8459fa5f535ca3e2bc16b26fcfd51710e83e78cc4fb75809596b97179f962d14",
    },
    "scaled_verification": {
        "path": "evidence/lumi-4b-sft-ai-assisted-gap-closure-scaled-verification-v0.2.json",
        "sha256": "c973b300a1ce0dee259d13f9bf2496f9d1a6c9440cc4509d2768e6e1c836ce18",
    },
    "source_registry_implementation": {
        "path": "src/lumi/posttraining_sources.py",
        "sha256": "10aa8e195bcab2613f3941f22f5303fc1cc8264403b4d7f20d04230bfcb40e50",
    },
}
EXPECTED_AUTHORIZATION = {
    "candidate_source_registration_authorized": True,
    "immutable_registry_successor_authorized": True,
    "machine_only_nonproduction_policy_authorized": True,
    "candidate_row_reads_authorized": False,
    "candidate_row_writes_authorized": False,
    "dataset_seal_authorized": False,
    "external_model_api_authorized": False,
    "gpu_authorized": False,
    "network_authorized": False,
    "predecessor_registry_mutation_authorized": False,
    "production_data_contract_change_authorized": False,
    "production_promotion_authorized": False,
    "remote_access_authorized": False,
    "sft_training_authorized": False,
    "unrelated_action_authorized": False,
}
EXPECTED_LIMITS = {
    "added_sources": 1,
    "changed_predecessor_sources": 0,
    "candidate_row_reads": 0,
    "candidate_row_writes": 0,
    "factual_verification_passes": 2,
    "maximum_evidence_write_bytes": 2_097_152,
    "maximum_external_model_api_calls": 0,
    "maximum_gpu_actions": 0,
    "maximum_network_bytes": 0,
    "maximum_runtime_minutes": 10,
    "maximum_training_actions": 0,
    "predecessor_sources": PREDECESSOR_SOURCES,
    "registry_successor_writes": 1,
    "removed_sources": 0,
    "successor_sources": SUCCESSOR_SOURCES,
}
EXPECTED_SOURCE = {
    "access": "first_party",
    "adapter": "lumi_ai_assisted_gap_closure_deterministic_renderer_v0.2",
    "blockers": [],
    "dataset_id": DATASET_ID,
    "evidence_urls": [],
    "languages": ["multilingual"],
    "license_id": "lumi-owned",
    "origin": "synthetic",
    "production_blockers": [
        "candidate_is_machine_only_ai_assisted_scaled_lane_pending_full_mix_reaudit",
        "full_registry_v0_24_cross_source_hygiene_and_floor_reaudit_not_complete",
        "production_data_contract_dataset_seal_and_sft_authorization_not_complete",
    ],
    "production_eligible": False,
    "repository_url": None,
    "revision": SOURCE_REVISION,
    "source_id": SOURCE_ID,
    "splits": ["train", "validation"],
    "status": "candidate_eligible",
    "task_categories": [
        "conversation", "factual_qa", "multilingual_bengali",
        "safety_helpfulness", "tool_api", "writing_summarization",
    ],
}
EXPECTED_OUTPUT = {
    "active_production_data_contract_update_permitted": False,
    "candidate_rows_read": False,
    "predecessor_registry_mutation_permitted": False,
    "production_dataset_promotions": 0,
    "raw_candidate_data_in_evidence": False,
    "receipt_path": "evidence/lumi-4b-sft-ai-gap-scaled-source-registration-v0.2.json",
    "successor_registry_path": "configs/lumi-4b-sft-source-registry-v0.24.json",
    "verification_path": "evidence/lumi-4b-sft-ai-gap-scaled-source-registration-verification-v0.2.json",
}
_FORBIDDEN_EVIDENCE_KEYS = {
    "assistant_response", "content", "messages", "prompt", "response",
    "source_text", "text", "tool_result", "user_prompt",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"binding escapes repository: {value}")
    return path


def resolve_binding(root: Path, binding: Mapping[str, Any], label: str) -> Path:
    if set(binding) != {"path", "sha256"}:
        raise ValueError(f"{label} binding shape changed")
    path = _resolve(root, str(binding.get("path")))
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} must be a regular file")
    if file_sha256(path) != binding.get("sha256"):
        raise ValueError(f"{label} binding changed")
    return path


def _verify_internal(value: Mapping[str, Any], field: str, label: str) -> None:
    unsigned = {key: item for key, item in value.items() if key != field}
    if value.get(field) != canonical_sha256(unsigned):
        raise ValueError(f"{label} internal seal changed")


def _assert_raw_text_free(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_EVIDENCE_KEYS:
                raise ValueError(f"raw candidate key in evidence: {'.'.join(path + (str(key),))}")
            _assert_raw_text_free(item, path + (str(key),))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_raw_text_free(item, path + (str(index),))


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan_path = Path(path).resolve()
    if plan_path != (root / "configs/lumi-4b-sft-ai-gap-scaled-source-registration-v0.2.json").resolve():
        raise ValueError("scaled source-registration plan path changed")
    plan = read_object(plan_path)
    expected_keys = {
        "authorization", "authorization_bytes", "authorization_sha256",
        "authorization_text", "bindings", "limits", "observed_at", "output",
        "schema_version", "source_after", "status",
    }
    if (
        set(plan) != expected_keys
        or plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("status") != PLAN_STATUS
        or plan.get("authorization_text") != AUTHORIZATION_TEXT
        or plan.get("authorization_bytes") != AUTHORIZATION_BYTES
        or plan.get("authorization_sha256") != AUTHORIZATION_SHA256
        or hashlib.sha256(AUTHORIZATION_TEXT.encode()).hexdigest() != AUTHORIZATION_SHA256
        or plan.get("authorization") != EXPECTED_AUTHORIZATION
        or plan.get("limits") != EXPECTED_LIMITS
        or plan.get("source_after") != EXPECTED_SOURCE
        or plan.get("output") != EXPECTED_OUTPUT
        or plan.get("observed_at") != "2026-09-02"
    ):
        raise ValueError("scaled source-registration plan boundary changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("scaled source-registration bindings missing")
    expected_binding_keys = set(EXPECTED_STATIC_BINDINGS) | set(EXPECTED_CODE_PATHS)
    if set(bindings) != expected_binding_keys:
        raise ValueError("scaled source-registration binding inventory changed")
    for label, expected in EXPECTED_STATIC_BINDINGS.items():
        if bindings.get(label) != expected:
            raise ValueError(f"{label} static binding changed")
        resolve_binding(root, expected, label)
    for label, expected_path in EXPECTED_CODE_PATHS.items():
        binding = bindings.get(label)
        if not isinstance(binding, Mapping) or binding.get("path") != expected_path:
            raise ValueError(f"{label} code binding changed")
        resolve_binding(root, binding, label)
    return plan


def validate_scaled_evidence(
    report: Mapping[str, Any], verification: Mapping[str, Any]
) -> dict[str, Any]:
    if report.get("schema_version") != scaled.REPORT_SCHEMA or report.get("status") != scaled.REPORT_STATUS:
        raise ValueError("scaled gap report schema or status changed")
    _verify_internal(report, "report_sha256", "scaled gap report")
    if verification.get("schema_version") != scaled.VERIFICATION_SCHEMA or verification.get("status") != scaled.VERIFICATION_STATUS:
        raise ValueError("scaled gap verification schema or status changed")
    _verify_internal(verification, "verification_sha256", "scaled gap verification")
    candidate = report.get("candidate")
    audit = report.get("audit")
    if not isinstance(candidate, Mapping) or not isinstance(audit, Mapping):
        raise ValueError("scaled candidate or audit aggregate missing")
    first, second = audit.get("pass_1"), audit.get("pass_2")
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        raise ValueError("scaled two-pass audit missing")
    if canonical_sha256(first) != canonical_sha256(second):
        raise ValueError("scaled audit passes differ")
    checks = first.get("checks")
    verification_checks = verification.get("checks")
    execution = report.get("execution")
    expected_verification_checks = {
        "authorization_and_code_bindings_verified",
        "candidate_exists_and_size_matches_without_third_content_read",
        "gap_floor_contributions_verified",
        "human_authorship_not_claimed",
        "prior_canary_cross_source_near_duplicate_free",
        "production_remains_false",
        "raw_candidate_data_absent_from_evidence",
        "two_sequential_audit_pass_summaries_equal",
        "zero_network_external_api_runtime_model_gpu_registry_training",
    }
    if (
        candidate.get("records") != scaled.EXPECTED_RECORDS
        or candidate.get("source_revision") != SOURCE_REVISION
        or candidate.get("candidate_file_sha256") != "9642885685993343148d531b00cf8f5e65042ae7b4eed72bae967a41d53b27c2"
        or candidate.get("candidate_file_bytes") != 28_324_248
        or candidate.get("raw_candidate_data_in_evidence") is not False
        or first.get("accepted_records") != scaled.EXPECTED_RECORDS
        or first.get("acceptance_rate") != 1.0
        or first.get("counts", {}).get("by_split") != {"train": 10_736, "validation": 2_684}
        or first.get("supervised_tokens", {}).get("by_split")
        != {"train": scaled.EXPECTED_TRAIN_SUPERVISED_TOKENS, "validation": scaled.EXPECTED_VALIDATION_SUPERVISED_TOKENS}
        or first.get("supervised_tokens", {}).get("train_by_language_mode") != scaled.EXPECTED_TRAIN_TOKENS_BY_MODE
        or first.get("supervised_tokens", {}).get("train_by_task") != scaled.EXPECTED_TRAIN_TOKENS_BY_TASK
        or first.get("projected_remaining_gaps")
        != {"banglish": 0, "factual_qa": 0, "natural_code_switching": 0, "strict_lower_bound": 0}
        or not isinstance(checks, Mapping) or not checks or not all(value is True for value in checks.values())
        or audit.get("passes_equal_sha256") != canonical_sha256(first)
        or not isinstance(execution, Mapping)
        or set(execution)
        != {
            "candidate_file_audit_passes", "candidate_file_row_reads",
            "candidate_writes", "elapsed_seconds", "external_model_api_calls",
            "gpu_actions", "network_bytes", "registry_writes",
            "runtime_model_calls", "sequential_passes", "training_actions",
        }
        or execution
        != {
            **execution,
            "candidate_file_audit_passes": 2,
            "candidate_file_row_reads": 26_840,
            "candidate_writes": 13_420,
            "sequential_passes": True,
            "network_bytes": 0,
            "external_model_api_calls": 0,
            "runtime_model_calls": 0,
            "gpu_actions": 0,
            "registry_writes": 0,
            "training_actions": 0,
        }
        or report.get("production")
        != {
            "dataset_sealed": False,
            "production_eligible": False,
            "sft_training_started": False,
            "source_registry_changed": False,
        }
        or verification.get("production_eligible") is not False
        or verification.get("report_file_sha256") != EXPECTED_STATIC_BINDINGS["scaled_report"]["sha256"]
        or verification.get("report_internal_sha256") != report.get("report_sha256")
        or not isinstance(verification_checks, Mapping)
        or set(verification_checks) != expected_verification_checks
        or not verification_checks
        or not all(value is True for value in verification_checks.values())
        or verification.get("candidate_file_sha256_from_bound_two_pass_report") != candidate.get("candidate_file_sha256")
    ):
        raise ValueError("scaled candidate frozen evidence changed")
    _assert_raw_text_free(report)
    _assert_raw_text_free(verification)
    return {
        "accepted_records": scaled.EXPECTED_RECORDS,
        "train_records": scaled.EXPECTED_TRAIN_RECORDS,
        "validation_records": scaled.EXPECTED_VALIDATION_RECORDS,
        "candidate_file_sha256": candidate["candidate_file_sha256"],
        "source_revision": SOURCE_REVISION,
        "train_supervised_tokens": scaled.EXPECTED_TRAIN_SUPERVISED_TOKENS,
        "validation_supervised_tokens": scaled.EXPECTED_VALIDATION_SUPERVISED_TOKENS,
    }


def build_successor(
    predecessor: Mapping[str, Any], *, source_after: Mapping[str, Any], observed_at: str
) -> dict[str, Any]:
    sources = predecessor.get("sources")
    if not isinstance(sources, list) or len(sources) != PREDECESSOR_SOURCES:
        raise ValueError("predecessor registry source count changed")
    if any(isinstance(source, Mapping) and source.get("source_id") == SOURCE_ID for source in sources):
        raise ValueError("scaled gap source already exists in predecessor registry")
    successor = copy.deepcopy(dict(predecessor))
    successor["observed_at"] = observed_at
    successor["sources"] = copy.deepcopy(sources) + [copy.deepcopy(dict(source_after))]
    return successor


def validate_successor(
    predecessor: Mapping[str, Any], successor: Mapping[str, Any], *,
    data_contract_path: str | Path,
) -> None:
    before, after = predecessor.get("sources"), successor.get("sources")
    if not isinstance(before, list) or not isinstance(after, list):
        raise ValueError("registry sources must be arrays")
    if len(before) != PREDECESSOR_SOURCES or len(after) != SUCCESSOR_SOURCES:
        raise ValueError("registry delta source counts changed")
    if after[:PREDECESSOR_SOURCES] != before or after[-1] != EXPECTED_SOURCE:
        raise ValueError("registry prefix or appended source changed")
    if successor.get("schema_version") != predecessor.get("schema_version") or successor.get("authorization") != predecessor.get("authorization"):
        raise ValueError("registry schema or authorization changed")
    policy = load_sft_policy(data_contract_path)
    parsed = [
        SFTSource.from_mapping(
            source,
            allowed_licenses=policy.allowed_licenses,
            allowed_languages=policy.allowed_languages,
            allowed_splits=policy.allowed_splits,
            allowed_tasks=frozenset(policy.task_mix),
        )
        for source in after
    ]
    if parsed[-1].source_id != SOURCE_ID or parsed[-1].status != "candidate_eligible" or parsed[-1].production_eligible:
        raise ValueError("parsed scaled source admissibility changed")


def build_receipt(
    *, plan: Mapping[str, Any], plan_file_sha256: str,
    predecessor_file_sha256: str, successor_file_sha256: str,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": RECEIPT_STATUS,
        "observed_at": plan["observed_at"],
        "authorization": copy.deepcopy(plan["authorization"]),
        "bindings": {
            "plan_file_sha256": plan_file_sha256,
            "predecessor_registry_sha256": predecessor_file_sha256,
            "successor_registry_sha256": successor_file_sha256,
            "scaled_report_sha256": plan["bindings"]["scaled_report"]["sha256"],
            "scaled_verification_sha256": plan["bindings"]["scaled_verification"]["sha256"],
        },
        "registry_delta": {
            "added_source_id": SOURCE_ID,
            "added_sources": 1,
            "changed_predecessor_sources": 0,
            "predecessor_sources": PREDECESSOR_SOURCES,
            "removed_sources": 0,
            "status_after": "candidate_eligible",
            "successor_sources": SUCCESSOR_SOURCES,
        },
        "candidate": {
            "records": candidate["accepted_records"],
            "train_records": candidate["train_records"],
            "validation_records": candidate["validation_records"],
            "source_revision": candidate["source_revision"],
            "candidate_file_sha256": candidate["candidate_file_sha256"],
            "train_supervised_tokens": candidate["train_supervised_tokens"],
            "validation_supervised_tokens": candidate["validation_supervised_tokens"],
            "candidate_rows_read": 0,
            "candidate_rows_modified": 0,
            "raw_candidate_data_retained": False,
        },
        "policy": {
            "machine_only_candidate_registration": True,
            "human_authorship_claimed": False,
            "human_review_claimed": False,
            "active_production_data_contract_updated": False,
            "production_dataset_promotions": 0,
            "production_eligible": False,
        },
        "execution": {
            "factual_verification_pass": 1,
            "network_bytes": 0,
            "external_model_api_calls": 0,
            "gpu_actions": 0,
            "training_actions": 0,
        },
        "remaining_blockers": list(EXPECTED_SOURCE["production_blockers"]),
        "next_gate": "registry_v0_24_full_candidate_mix_cross_source_reaudit",
        "production_eligible": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _assert_raw_text_free(receipt)
    return receipt


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    payload = pretty_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def run_registration(
    *, plan_path: str | Path, repository_root: str | Path,
    successor_path: str | Path, receipt_path: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan_file = Path(plan_path).resolve()
    successor_file = Path(successor_path).resolve()
    receipt_file = Path(receipt_path).resolve()
    plan = load_plan(plan_file, repository_root=root)
    if successor_file != (root / EXPECTED_OUTPUT["successor_registry_path"]).resolve() or receipt_file != (root / EXPECTED_OUTPUT["receipt_path"]).resolve():
        raise ValueError("scaled registration output path changed")
    bindings = plan["bindings"]
    predecessor_path = resolve_binding(root, bindings["predecessor_registry"], "predecessor_registry")
    data_contract_path = resolve_binding(root, bindings["data_contract"], "data_contract")
    candidate = validate_scaled_evidence(
        read_object(resolve_binding(root, bindings["scaled_report"], "scaled_report")),
        read_object(resolve_binding(root, bindings["scaled_verification"], "scaled_verification")),
    )
    predecessor = read_object(predecessor_path)
    successor = build_successor(predecessor, source_after=plan["source_after"], observed_at=plan["observed_at"])
    validate_successor(predecessor, successor, data_contract_path=data_contract_path)
    if successor_file.exists() or receipt_file.exists():
        raise FileExistsError("v0.24 registration outputs already exist; retry is forbidden")
    _write_once(successor_file, successor)
    successor_sha256 = file_sha256(successor_file)
    policy = load_sft_policy(data_contract_path)
    parsed = load_sft_source_registry(
        successor_file,
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
        expected_sha256=successor_sha256,
    )
    if len(parsed.sources) != SUCCESSOR_SOURCES or parsed.sources[-1].source_id != SOURCE_ID:
        raise ValueError("materialized v0.24 registry parsing changed")
    receipt = build_receipt(
        plan=plan,
        plan_file_sha256=file_sha256(plan_file),
        predecessor_file_sha256=file_sha256(predecessor_path),
        successor_file_sha256=successor_sha256,
        candidate=candidate,
    )
    if len(pretty_json_bytes(receipt)) > EXPECTED_LIMITS["maximum_evidence_write_bytes"]:
        raise ValueError("scaled registration evidence write cap exceeded")
    _write_once(receipt_file, receipt)
    return receipt


def verify_registration(
    *, plan_path: str | Path, repository_root: str | Path,
    successor_path: str | Path, receipt_path: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan_file = Path(plan_path).resolve()
    successor_file = Path(successor_path).resolve()
    receipt_file = Path(receipt_path).resolve()
    plan = load_plan(plan_file, repository_root=root)
    if successor_file != (root / EXPECTED_OUTPUT["successor_registry_path"]).resolve() or receipt_file != (root / EXPECTED_OUTPUT["receipt_path"]).resolve():
        raise ValueError("scaled verification input path changed")
    bindings = plan["bindings"]
    predecessor_path = resolve_binding(root, bindings["predecessor_registry"], "predecessor_registry")
    data_contract_path = resolve_binding(root, bindings["data_contract"], "data_contract")
    predecessor = read_object(predecessor_path)
    expected_successor = build_successor(predecessor, source_after=plan["source_after"], observed_at=plan["observed_at"])
    observed_successor = read_object(successor_file)
    if observed_successor != expected_successor:
        raise ValueError("materialized v0.24 registry differs from reconstruction")
    validate_successor(predecessor, observed_successor, data_contract_path=data_contract_path)
    candidate = validate_scaled_evidence(
        read_object(resolve_binding(root, bindings["scaled_report"], "scaled_report")),
        read_object(resolve_binding(root, bindings["scaled_verification"], "scaled_verification")),
    )
    expected_receipt = build_receipt(
        plan=plan,
        plan_file_sha256=file_sha256(plan_file),
        predecessor_file_sha256=file_sha256(predecessor_path),
        successor_file_sha256=file_sha256(successor_file),
        candidate=candidate,
    )
    receipt = read_object(receipt_file)
    if receipt != expected_receipt:
        raise ValueError("scaled registration receipt differs from reconstruction")
    _verify_internal(receipt, "receipt_sha256", "scaled registration receipt")
    verification: dict[str, Any] = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": VERIFICATION_STATUS,
        "observed_at": plan["observed_at"],
        "receipt_file_sha256": file_sha256(receipt_file),
        "receipt_internal_sha256": receipt["receipt_sha256"],
        "successor_registry_sha256": file_sha256(successor_file),
        "execution": {
            "factual_verification_pass": 2,
            "candidate_row_reads": 0,
            "network_bytes": 0,
            "external_model_api_calls": 0,
            "gpu_actions": 0,
            "training_actions": 0,
        },
        "checks": {
            "candidate_rows_not_read_or_modified": True,
            "exactly_one_candidate_source_added": True,
            "human_authorship_or_review_not_claimed": True,
            "predecessor_registry_unchanged": True,
            "production_contract_unchanged": True,
            "production_promotion_count_zero": True,
            "scaled_evidence_reconstructed": True,
            "successor_registry_reconstructed": True,
            "two_zero_network_factual_passes_complete": True,
        },
        "production_eligible": False,
        "next_gate": "registry_v0_24_full_candidate_mix_cross_source_reaudit",
    }
    verification["verification_sha256"] = canonical_sha256(verification)
    _assert_raw_text_free(verification)
    return verification


__all__ = [
    "AUTHORIZATION_BYTES", "AUTHORIZATION_SHA256", "AUTHORIZATION_TEXT",
    "EXPECTED_SOURCE", "SOURCE_ID", "_assert_raw_text_free",
    "build_successor", "canonical_sha256", "file_sha256", "load_plan",
    "run_registration", "validate_scaled_evidence", "validate_successor",
    "verify_registration",
]
