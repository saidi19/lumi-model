"""Materialize the human-owner review for the BanglaBridge expanded delta."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_banglabridge_expanded_delta_recovery_v1 import (
    STATUS as RECOVERY_STATUS,
)
from .posttraining_banglabridge_expanded_delta_recovery_v1 import (
    VERIFICATION_STATUS as RECOVERY_VERIFICATION_STATUS,
)
from .posttraining_banglabridge_expanded_delta_recovery_v1 import (
    file_sha256,
    sha256_value,
)
from .posttraining_banglabridge_expanded_delta_recovery_v1 import (
    validate_report as validate_recovery_report,
)
from .posttraining_banglabridge_expanded_delta_recovery_v1 import (
    validate_verification as validate_recovery_verification,
)
from .posttraining_first_party import canonical_json_bytes
from .posttraining_owner_review import (
    DECISIONS_SCHEMA_VERSION,
    validate_owner_decisions,
)

PLAN_SCHEMA_VERSION = (
    "lumi.sft-banglabridge-expanded-delta-owner-review-plan.v0.1"
)
RECEIPT_SCHEMA_VERSION = (
    "lumi.sft-banglabridge-expanded-delta-owner-review-receipt.v0.1"
)
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-banglabridge-expanded-delta-owner-review-verification.v0.1"
)
RECEIPT_STATUS = "owner_accepted_all_32_banglabridge_expanded_delta_rows"
VERIFICATION_STATUS = (
    "PASS_owner_accepted_all_32_banglabridge_expanded_delta_rows"
)
EXPECTED_OWNER_REQUEST = "done mama and file"
EXPECTED_OWNER_REQUEST_SHA256 = (
    "6c6239d6b66e38c1c31494c659e4f637211528fe6a720471a95bba7565b3987e"
)
EXPECTED_SAMPLE_CONTENT_SHA256 = (
    "e6f43e874d9d2dcb3a791ffae82594f9186fbd14dcaa91020667aec06c340f65"
)
EXPECTED_SAMPLE_FILE_SHA256 = (
    "68f392628e6f5c4a56ea54279a981ca76cc9cedbefd11854f9f1731a52a254ca"
)
EXPECTED_DECISIONS_FILE_SHA256 = (
    "6e269de778d801f7dc450ac4d938e41cd84c372937e198e5ae9819c030a093bf"
)
EXPECTED_DECISIONS_CONTENT_SHA256 = (
    "2c641dc09fe69dd072d6c7330a501426f66bed57b20239a8e7e6c23b089354f5"
)
EXPECTED_REVIEWED_IDS_SHA256 = (
    "6a8cba8e2a91dccace8784205eaf91bd6287d9659c4c967ef1e1bcd98e254cbf"
)
EXPECTED_AUTHORIZATION = {
    "candidate_or_source_mutation_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_or_download_authorized": False,
    "gpu_or_inference_authorized": False,
    "human_owner_decision_materialization_authorized": True,
    "network_authorized": False,
    "production_promotion_authorized": False,
    "registry_mutation_authorized": False,
    "sft_training_authorized": False,
}
EXPECTED_BINDING_SHA256 = {
    "candidate": "5cf0b5f4ad4029f7a9e25b49ac7785f43607fb3196baf8c9f5dd272da0551058",
    "decisions": EXPECTED_DECISIONS_FILE_SHA256,
    "delta_ledger": "53aee6b837f5dd5cd6d763011b054019338e6cda42fa7b209ac97734d196614f",
    "owner_sample": EXPECTED_SAMPLE_FILE_SHA256,
    "recovery_plan": "eb3ddcbd5bd5d0b72d2b9ee5c129f334ab1cf401d7b2470eee21eebc32a9cb8f",
    "recovery_report": "66c8bce030779bf5ab01176deb95ba924a915b13759751b23097be17b03190ab",
    "recovery_verification": "d61ebba98bfb2758753c1ff60e4982390849b1daddaeae96e74716a5dc1764f3",
}
EXPECTED_CODE_PATHS = {
    "implementation": (
        "src/lumi/posttraining_banglabridge_expanded_delta_owner_review_v1.py"
    ),
    "materializer": (
        "scripts/materialize_sft_banglabridge_expanded_delta_owner_receipt_v1.py"
    ),
    "test": "tests/test_banglabridge_expanded_delta_owner_review_v1.py",
    "verifier": (
        "scripts/verify_sft_banglabridge_expanded_delta_owner_review_v1.py"
    ),
}
EXPECTED_PLAN_KEYS = {
    "authorization",
    "bindings",
    *EXPECTED_CODE_PATHS,
    "owner_request_sha256",
    "owner_request_text",
    "output",
    "schema_version",
    "status",
}
EXPECTED_CANDIDATE = {
    "records": 561,
    "train_records": 544,
    "train_supervised_tokens": 250_441,
    "validation_records": 17,
    "validation_supervised_tokens": 4_960,
}
EXPECTED_DELTA = {"records": 445, "train_supervised_tokens": 200_685}
EXPECTED_NEXT_GATE = {
    "action": "separate_bounded_registry_successor_promotion_then_full_mix_reaudit",
    "dataset_seal_or_sft_authorized": False,
    "owner_review_complete": True,
    "registry_change_authorized": False,
    "separate_authorization_required": True,
}
EXPECTED_VERIFICATION_CHECKS = {
    "all_32_owner_decisions_are_accept_and_threshold_met": True,
    "candidate_delta_packet_and_recovery_evidence_bound": True,
    "decision_and_packet_content_and_file_hash_domains_reproduced": True,
    "no_raw_training_text_in_receipt_or_verification": True,
    "no_registry_seal_inference_gpu_network_sft_or_promotion": True,
    "owner_review_is_human_claim_not_automated_claim": True,
    "source_provenance_and_row_authorship_not_overclaimed": True,
}
EXPECTED_RECEIPT_KEYS = {
    "acceptance_rate",
    "accepted_records",
    "all_sampled_rows_accepted",
    "authorization",
    "bindings",
    "candidate",
    "delta",
    "next_gate",
    "owner_review_claim",
    "production_eligible",
    "raw_training_text_retained",
    "receipt_sha256",
    "rejected_record_ids",
    "rejected_records",
    "reviewed_at",
    "reviewer_alias",
    "sample_records",
    "schema_version",
    "status",
    "threshold",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite BanglaBridge owner review: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_plan(path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _load_json(path)
    if (
        not isinstance(plan, dict)
        or set(plan) != EXPECTED_PLAN_KEYS
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status") != "authorized_human_owner_review_finalization"
        or plan.get("authorization") != EXPECTED_AUTHORIZATION
        or plan.get("owner_request_text") != EXPECTED_OWNER_REQUEST
        or plan.get("owner_request_sha256") != EXPECTED_OWNER_REQUEST_SHA256
        or hashlib.sha256(plan["owner_request_text"].encode()).hexdigest()
        != EXPECTED_OWNER_REQUEST_SHA256
        or plan.get("output")
        != {
            "receipt_path": (
                "evidence/lumi-4b-sft-banglabridge-expanded-delta-owner-review-"
                "v0.1.json"
            ),
            "verification_path": (
                "evidence/lumi-4b-sft-banglabridge-expanded-delta-owner-review-"
                "verification-v0.1.json"
            ),
        }
    ):
        raise ValueError("BanglaBridge owner-review plan changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(
        EXPECTED_BINDING_SHA256
    ):
        raise ValueError("BanglaBridge owner-review binding inventory changed")
    for name, expected_sha256 in EXPECTED_BINDING_SHA256.items():
        binding = bindings[name]
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
            raise ValueError(f"invalid BanglaBridge owner-review binding: {name}")
        resolved = (root / str(binding.get("path"))).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError(f"missing BanglaBridge owner-review binding: {name}")
        if binding.get("sha256") != expected_sha256 or file_sha256(resolved) != expected_sha256:
            raise ValueError(f"BanglaBridge owner-review binding changed: {name}")
    for name, expected_path in EXPECTED_CODE_PATHS.items():
        binding = plan[name]
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256"}
            or binding.get("path") != expected_path
            or not isinstance(binding.get("sha256"), str)
            or _SHA256.fullmatch(binding["sha256"]) is None
        ):
            raise ValueError(f"invalid BanglaBridge owner-review code binding: {name}")
        resolved = (root / expected_path).resolve()
        if (
            not resolved.is_relative_to(root)
            or not resolved.is_file()
            or file_sha256(resolved) != binding["sha256"]
        ):
            raise ValueError(f"BanglaBridge owner-review code changed: {name}")
    return plan


def _validate_recovery_evidence(
    *, plan: Mapping[str, Any], repository_root: Path
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    bindings = plan["bindings"]
    report = _load_json(repository_root / bindings["recovery_report"]["path"])
    verification = _load_json(
        repository_root / bindings["recovery_verification"]["path"]
    )
    validate_recovery_report(report)
    validate_recovery_verification(verification)
    if (
        report.get("status") != RECOVERY_STATUS
        or report.get("report_sha256")
        != "8dab0c08ff0e51a343624929c6f2c6dc86282bd3bdb587aa1c563abea5d90730"
        or verification.get("status") != RECOVERY_VERIFICATION_STATUS
        or verification.get("verification_sha256")
        != "7bd7fe589e529de8a8a2cca3a6c46a2e36940f4c81e76979aedafac19f758691"
        or verification.get("report_file_sha256")
        != EXPECTED_BINDING_SHA256["recovery_report"]
        or report.get("successor")
        != {
            "bytes": 1_967_432,
            "records": 561,
            "sha256": EXPECTED_BINDING_SHA256["candidate"],
            "train_records": 544,
            "train_supervised_tokens": 250_441,
            "validation_records": 17,
            "validation_supervised_tokens": 4_960,
        }
        or report.get("delta", {}).get("records") != 445
        or report.get("delta", {}).get("train_supervised_tokens") != 200_685
        or report.get("owner_review", {}).get("packet_content_sha256")
        != EXPECTED_SAMPLE_CONTENT_SHA256
        or report.get("owner_review", {}).get("packet_file_sha256")
        != EXPECTED_SAMPLE_FILE_SHA256
        or report.get("owner_review", {}).get("minimum_accepts_required_for_later_promotion")
        != 29
    ):
        raise ValueError("BanglaBridge recovery evidence changed")
    return report, verification


def build_receipt(
    plan: Mapping[str, Any], *, plan_file_sha256: str, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    bindings = plan["bindings"]
    sample = _load_json(root / bindings["owner_sample"]["path"])
    if (
        not isinstance(sample, list)
        or len(sample) != 32
        or hashlib.sha256(canonical_json_bytes(sample)).hexdigest()
        != EXPECTED_SAMPLE_CONTENT_SHA256
    ):
        raise ValueError("BanglaBridge owner sample changed")
    sample_ids = {str(row.get("record_id")) for row in sample}
    if len(sample_ids) != 32:
        raise ValueError("BanglaBridge owner sample IDs changed")

    decisions = _load_json(root / bindings["decisions"]["path"])
    if (
        not isinstance(decisions, Mapping)
        or decisions.get("schema_version") != DECISIONS_SCHEMA_VERSION
        or hashlib.sha256(canonical_json_bytes(decisions)).hexdigest()
        != EXPECTED_DECISIONS_CONTENT_SHA256
    ):
        raise ValueError("BanglaBridge owner decisions changed")
    generic = validate_owner_decisions(
        decisions,
        sample_rows=sample,
        sample_sha256=EXPECTED_SAMPLE_CONTENT_SHA256,
    )
    decision_rows = decisions["decisions"]
    decision_ids = {str(row["record_id"]) for row in decision_rows}
    if (
        generic.get("accepted_records") != 32
        or generic.get("rejected_records") != 0
        or decision_ids != sample_ids
        or any(row.get("accepted") is not True or row.get("note") != "" for row in decision_rows)
        or sha256_value(sorted(decision_ids)) != EXPECTED_REVIEWED_IDS_SHA256
    ):
        raise ValueError("BanglaBridge owner outcome changed")
    report, verification = _validate_recovery_evidence(plan=plan, repository_root=root)

    receipt: dict[str, Any] = {
        "acceptance_rate": 1.0,
        "accepted_records": 32,
        "all_sampled_rows_accepted": True,
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            "candidate_file_sha256": EXPECTED_BINDING_SHA256["candidate"],
            "decisions_content_sha256": EXPECTED_DECISIONS_CONTENT_SHA256,
            "decisions_file_sha256": EXPECTED_DECISIONS_FILE_SHA256,
            "delta_ledger_file_sha256": EXPECTED_BINDING_SHA256["delta_ledger"],
            "owner_request_sha256": EXPECTED_OWNER_REQUEST_SHA256,
            "owner_sample_content_sha256": EXPECTED_SAMPLE_CONTENT_SHA256,
            "owner_sample_file_sha256": EXPECTED_SAMPLE_FILE_SHA256,
            "plan_file_sha256": plan_file_sha256,
            "recovery_plan_file_sha256": EXPECTED_BINDING_SHA256["recovery_plan"],
            "recovery_report_file_sha256": EXPECTED_BINDING_SHA256["recovery_report"],
            "recovery_report_internal_sha256": report["report_sha256"],
            "recovery_verification_file_sha256": EXPECTED_BINDING_SHA256[
                "recovery_verification"
            ],
            "recovery_verification_internal_sha256": verification[
                "verification_sha256"
            ],
            "reviewed_record_ids_sha256": EXPECTED_REVIEWED_IDS_SHA256,
        },
        "candidate": EXPECTED_CANDIDATE,
        "delta": EXPECTED_DELTA,
        "next_gate": EXPECTED_NEXT_GATE,
        "owner_review_claim": (
            "banglabridge_cc_by_4_0_source_derived_or_deterministically_transformed_"
            "owner_spot_checked_32_row_level_human_authorship_not_established"
        ),
        "production_eligible": False,
        "raw_training_text_retained": False,
        "rejected_record_ids": [],
        "rejected_records": 0,
        "reviewed_at": decisions["reviewed_at"],
        "reviewer_alias": decisions["reviewer_id"],
        "sample_records": 32,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": RECEIPT_STATUS,
        "threshold": {"minimum_accepts": 29, "satisfied": True},
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    validate_receipt(receipt)
    return receipt


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    unsigned = dict(receipt)
    internal = unsigned.pop("receipt_sha256", None)
    if (
        set(receipt) != EXPECTED_RECEIPT_KEYS
        or internal != sha256_value(unsigned)
        or receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != RECEIPT_STATUS
        or receipt.get("authorization") != EXPECTED_AUTHORIZATION
        or receipt.get("accepted_records") != 32
        or receipt.get("rejected_records") != 0
        or receipt.get("rejected_record_ids") != []
        or receipt.get("acceptance_rate") != 1.0
        or receipt.get("all_sampled_rows_accepted") is not True
        or receipt.get("sample_records") != 32
        or receipt.get("threshold") != {"minimum_accepts": 29, "satisfied": True}
        or receipt.get("production_eligible") is not False
        or receipt.get("raw_training_text_retained") is not False
        or receipt.get("reviewer_alias") != "owner-afnan"
        or receipt.get("reviewed_at") != "2026-08-13T13:35:40Z"
        or receipt.get("candidate") != EXPECTED_CANDIDATE
        or receipt.get("delta") != EXPECTED_DELTA
        or receipt.get("owner_review_claim")
        != (
            "banglabridge_cc_by_4_0_source_derived_or_deterministically_"
            "transformed_owner_spot_checked_32_row_level_human_authorship_"
            "not_established"
        )
        or receipt.get("next_gate") != EXPECTED_NEXT_GATE
        or not isinstance(receipt.get("bindings"), Mapping)
        or set(receipt["bindings"])
        != {
            "candidate_file_sha256",
            "decisions_content_sha256",
            "decisions_file_sha256",
            "delta_ledger_file_sha256",
            "owner_request_sha256",
            "owner_sample_content_sha256",
            "owner_sample_file_sha256",
            "plan_file_sha256",
            "recovery_plan_file_sha256",
            "recovery_report_file_sha256",
            "recovery_report_internal_sha256",
            "recovery_verification_file_sha256",
            "recovery_verification_internal_sha256",
            "reviewed_record_ids_sha256",
        }
        or receipt["bindings"].get("candidate_file_sha256")
        != EXPECTED_BINDING_SHA256["candidate"]
        or receipt["bindings"].get("decisions_content_sha256")
        != EXPECTED_DECISIONS_CONTENT_SHA256
        or receipt["bindings"].get("decisions_file_sha256")
        != EXPECTED_DECISIONS_FILE_SHA256
        or receipt["bindings"].get("delta_ledger_file_sha256")
        != EXPECTED_BINDING_SHA256["delta_ledger"]
        or receipt["bindings"].get("owner_request_sha256")
        != EXPECTED_OWNER_REQUEST_SHA256
        or receipt["bindings"].get("owner_sample_content_sha256")
        != EXPECTED_SAMPLE_CONTENT_SHA256
        or receipt["bindings"].get("owner_sample_file_sha256")
        != EXPECTED_SAMPLE_FILE_SHA256
        or receipt["bindings"].get("recovery_plan_file_sha256")
        != EXPECTED_BINDING_SHA256["recovery_plan"]
        or receipt["bindings"].get("recovery_report_file_sha256")
        != EXPECTED_BINDING_SHA256["recovery_report"]
        or receipt["bindings"].get("recovery_report_internal_sha256")
        != "8dab0c08ff0e51a343624929c6f2c6dc86282bd3bdb587aa1c563abea5d90730"
        or receipt["bindings"].get("recovery_verification_file_sha256")
        != EXPECTED_BINDING_SHA256["recovery_verification"]
        or receipt["bindings"].get("recovery_verification_internal_sha256")
        != "7bd7fe589e529de8a8a2cca3a6c46a2e36940f4c81e76979aedafac19f758691"
        or receipt["bindings"].get("reviewed_record_ids_sha256")
        != EXPECTED_REVIEWED_IDS_SHA256
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in receipt["bindings"].values()
        )
    ):
        raise ValueError("BanglaBridge owner-review receipt changed")
    serialized = json.dumps(receipt, ensure_ascii=False, sort_keys=True)
    for forbidden in ('"messages"', '"content"', '"user_prompt"', '"assistant_response"'):
        if forbidden in serialized:
            raise ValueError("BanglaBridge owner receipt retained raw text")


def build_verification(
    receipt: Mapping[str, Any], *, receipt_file_sha256: str
) -> dict[str, Any]:
    validate_receipt(receipt)
    if _SHA256.fullmatch(receipt_file_sha256) is None:
        raise ValueError("BanglaBridge owner receipt file hash changed")
    verification: dict[str, Any] = {
        "checks": EXPECTED_VERIFICATION_CHECKS,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_internal_sha256": receipt["receipt_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": VERIFICATION_STATUS,
    }
    verification["verification_sha256"] = sha256_value(verification)
    validate_verification(verification)
    return verification


def validate_verification(verification: Mapping[str, Any]) -> None:
    unsigned = dict(verification)
    internal = unsigned.pop("verification_sha256", None)
    if (
        set(verification)
        != {
            "checks",
            "production_eligible",
            "raw_training_text_retained",
            "receipt_file_sha256",
            "receipt_internal_sha256",
            "schema_version",
            "status",
            "verification_sha256",
        }
        or internal != sha256_value(unsigned)
        or verification.get("schema_version") != VERIFICATION_SCHEMA_VERSION
        or verification.get("status") != VERIFICATION_STATUS
        or verification.get("production_eligible") is not False
        or verification.get("raw_training_text_retained") is not False
        or verification.get("checks") != EXPECTED_VERIFICATION_CHECKS
        or any(
            not isinstance(verification.get(field), str)
            or _SHA256.fullmatch(verification[field]) is None
            for field in ("receipt_file_sha256", "receipt_internal_sha256")
        )
    ):
        raise ValueError("BanglaBridge owner verification changed")


def finalize(
    plan_path: str | Path, repository_root: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(repository_root).resolve()
    plan = load_plan(plan_path, root)
    receipt = build_receipt(
        plan, plan_file_sha256=file_sha256(plan_path), repository_root=root
    )
    receipt_path = root / plan["output"]["receipt_path"]
    verification_path = root / plan["output"]["verification_path"]
    _write_json_atomic(receipt_path, receipt)
    verification = build_verification(
        receipt, receipt_file_sha256=file_sha256(receipt_path)
    )
    _write_json_atomic(verification_path, verification)
    return receipt, verification


__all__ = [
    "build_receipt",
    "build_verification",
    "finalize",
    "load_plan",
    "validate_receipt",
    "validate_verification",
]
