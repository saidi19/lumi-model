"""Finalize the bounded Be-DSTC2 attempt-006 human-owner spot check.

This module reads only the exact owner decision export, the exact 32-row owner
packet, and the already raw-text-free v0.7 report and verification.  It never
opens the candidate, source objects, baseline, registry, or evaluation objects.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from .posttraining_bedstc2_banglish_conversation_curation_v6 import (
    validate_report as validate_curation_report,
)
from .posttraining_bedstc2_banglish_conversation_curation_v6 import (
    validate_verification as validate_curation_verification,
)

PLAN_SCHEMA_VERSION = (
    "lumi.sft-bedstc2-banglish-conversation-owner-review-plan.v0.1"
)
RECEIPT_SCHEMA_VERSION = (
    "lumi.sft-bedstc2-banglish-conversation-owner-review-receipt.v0.1"
)
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-bedstc2-banglish-conversation-owner-review-verification.v0.1"
)
DECISIONS_SCHEMA_VERSION = "lumi.ai-assisted-owner-decisions.v0.3"
PLAN_STATUS = "authorized_bedstc2_human_owner_review_finalization"
RECEIPT_STATUS = (
    "owner_accepted_all_32_bedstc2_rows_nonproduction_rights_pending"
)
VERIFICATION_STATUS = (
    "PASS_verified_owner_accepted_all_32_bedstc2_rows_nonproduction_rights_pending"
)

EXPECTED_PLAN_PATH = Path(
    "configs/lumi-4b-sft-bedstc2-banglish-conversation-owner-review-v0.1.json"
)
EXPECTED_AUTHORIZATION_TEXT = "ok next step shruu kor"
EXPECTED_AUTHORIZATION_SHA256 = (
    "2fa197f0571b03a6ed07c2636c661b114dd1654dc4332e1a4a4f43ec3992bbc3"
)
EXPECTED_EXTERNAL_DECISIONS_PATH = Path(
    "/Users/afnan/Downloads/lumi-owner-review-decisions-v0.3 (3).json"
)
EXPECTED_CODE_PATHS = {
    "implementation": (
        "src/lumi/posttraining_bedstc2_banglish_conversation_owner_review_v1.py"
    ),
    "finalizer": (
        "scripts/finalize_sft_bedstc2_banglish_conversation_owner_review_v1.py"
    ),
    "verifier": (
        "scripts/verify_sft_bedstc2_banglish_conversation_owner_review_v1.py"
    ),
    "test": "tests/test_bedstc2_banglish_conversation_owner_review_v1.py",
}
EXPECTED_OUTPUT = {
    "receipt_path": (
        "evidence/lumi-4b-sft-bedstc2-banglish-conversation-owner-review-v0.1.json"
    ),
    "verification_path": (
        "evidence/lumi-4b-sft-bedstc2-banglish-conversation-owner-review-"
        "verification-v0.1.json"
    ),
}
EXPECTED_INPUTS = {
    "decisions": {
        "path": str(EXPECTED_EXTERNAL_DECISIONS_PATH),
        "bytes": 4_021,
        "sha256": "68c5802a0e44937166bf1db16463de058513b5992d71da99d93d8144bd99b5a6",
        "canonical_sha256": (
            "86bd11d6b40da24a0fee6a1570b41e94e145fed58d2e0244787661a15fecc918"
        ),
    },
    "owner_packet": {
        "path": (
            "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-006/"
            "private/owner-review-32.json"
        ),
        "bytes": 49_495,
        "sha256": "e9872362d98e2897bd519830e89f480492bc224380bf795e97ae5451b4355545",
        "canonical_sha256": (
            "4e6a9e73b3f4b38ebb2a9ba579b93c8b87e0c50a0cc4a27e33360eaec9a4a65d"
        ),
    },
    "curation_report": {
        "path": (
            "evidence/lumi-4b-sft-bedstc2-banglish-conversation-curation-v0.7.json"
        ),
        "bytes": 26_182,
        "sha256": "81548913d2a9fe1d5cdf9ecbc2f927e64f20033a34931425f12821a15b7305df",
        "canonical_sha256": (
            "bf690b11e0018b1b96e996c37cf511db80eb60e5b6cadfface82326f68fc296b"
        ),
    },
    "curation_verification": {
        "path": (
            "evidence/lumi-4b-sft-bedstc2-banglish-conversation-curation-"
            "verification-v0.7.json"
        ),
        "bytes": 2_914,
        "sha256": "d5b87bd7882111b882f5d6fc33401e58a1ec694d5f801f750ab8ed3bc6720a61",
        "canonical_sha256": (
            "ba8597ab5fa6b1468b9ef1fb466b4082fddcd679a322a15d71a807fb63519241"
        ),
    },
}
EXPECTED_LIMITS = {
    "passes": 2,
    "decision_reads": 2,
    "decision_rows": 64,
    "owner_packet_reads": 2,
    "owner_packet_rows": 64,
    "curation_report_reads": 2,
    "curation_verification_reads": 2,
    "source_line_reads": 0,
    "candidate_row_reads": 0,
    "baseline_row_reads": 0,
    "test_object_reads": 0,
    "runtime_seconds_total": 300,
    "runtime_seconds_per_process": 150,
    "private_write_bytes": 1_048_576,
    "network_bytes": 0,
    "gpu_use": 0,
    "inference_calls": 0,
}
EXPECTED_AUTHORIZATION = {
    "human_owner_decision_materialization_authorized": True,
    "candidate_or_source_mutation_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_or_download_authorized": False,
    "gpu_or_inference_authorized": False,
    "network_authorized": False,
    "production_promotion_authorized": False,
    "registry_mutation_authorized": False,
    "sft_training_authorized": False,
    "source_candidate_or_baseline_replay_authorized": False,
    "test_object_access_authorized": False,
    "unrelated_action_authorized": False,
}
EXPECTED_CANDIDATE = {
    "file_sha256_attested_not_opened": (
        "f90378fa0791279d84db6da1654f2f0ea2b334d2c56de0a062184636a075fc3d"
    ),
    "records": 2_865,
    "train_records": 2_729,
    "train_supervised_tokens": 53_147,
    "validation_records": 136,
    "validation_supervised_tokens": 2_539,
}
EXPECTED_PROVENANCE = (
    "human_in_house_annotated_source_derived_bengali_code_mixed_surface_wording_"
    "with_underlying_dstc2_authorship_and_origin_not_established"
)
EXPECTED_OWNER_REVIEW_CLAIM = (
    "source_derived_bedstc2_banglish_rows_human_owner_spot_checked_32_"
    "row_level_human_authorship_and_identity_not_established"
)
EXPECTED_RIGHTS = {
    "license": "apache-2.0",
    "upstream_dstc2_rights_chain_independently_verified": False,
}
EXPECTED_EVALUATION_PROHIBITION = {
    "bedstc2_dev_claimed_evaluation": "permanently_prohibited",
    "bedstc2_test_claimed_evaluation": "permanently_prohibited",
    "bedstc2_test_object": {
        "accessed": False,
        "bytes": 2_283_372,
        "git_blob_sha1": "85a9a4a43f3d1a9c54772c84d4a43872c9589ae6",
        "path": "data/bengali/dialog-dstc2-tst.txt",
    },
    "billsum_ca_test_claimed_evaluation": "permanently_prohibited",
    "billsum_test_claimed_evaluation": "permanently_prohibited",
    "reason": "BillSum train split used for SFT candidate construction",
    "reason_bedstc2": (
        "official train and dev objects used for SFT candidate construction"
    ),
}
EXPECTED_REMAINING_BLOCKERS = [
    "upstream_dstc2_rights_chain_not_independently_verified",
    "source_promotion_and_full_cross_source_reaudit_not_authorized",
    "production_dataset_seal_and_sft_authorization_not_complete",
]
EXPECTED_NEXT_GATE = {
    "action": "separate_bounded_upstream_dstc2_rights_review",
    "owner_review_complete": True,
    "production_promotion_authorized": False,
    "separate_authorization_required": True,
}
EXPECTED_REVIEWED_IDS_DOMAIN_SHA256 = (
    "00b4770f6f33db397fc2b00c76b5b4741b4cf5567a02a44a61e527cc361521b9"
)
EXPECTED_REVIEWED_AT = "2026-08-16T16:06:54Z"
EXPECTED_RECEIPT_MARKER = (
    b'{"started":"bedstc2_owner_review_receipt_v0.1"}\n'
)
EXPECTED_VERIFICATION_MARKER = (
    b'{"started":"bedstc2_owner_review_verification_v0.1"}\n'
)
EXPECTED_CHECKS = {
    "all_32_owner_decisions_accept_and_29_threshold_met": True,
    "candidate_source_and_baseline_payloads_not_opened": True,
    "curation_report_and_verification_bound": True,
    "decision_and_packet_file_and_canonical_hash_domains_reproduced": True,
    "decision_notes_reviewer_alias_record_ids_and_training_text_not_retained": True,
    "human_owner_review_claim_is_self_attested_not_identity_verified": True,
    "no_registry_seal_training_promotion_network_gpu_or_inference": True,
    "source_provenance_rights_and_evaluation_limits_preserved": True,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RECORD_ID_RE = re.compile(r"^bedstc2-banglish-[0-9a-f]{24}$")
_REVIEWER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_AI_ALIAS_MARKERS = {
    "automated",
    "bot",
    "chatgpt",
    "claude",
    "codex",
    "gemini",
    "openai",
    "lumi-deterministic-ai-assisted-guard-v0.3",
}
_FORBIDDEN_RECEIPT_KEYS = {
    "assistant_response",
    "content",
    "decisions",
    "messages",
    "note",
    "record_id",
    "reviewer_alias",
    "reviewer_id",
    "user_prompt",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_value(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_: str) -> Any:
    raise ValueError("non-finite JSON number")


def _assert_no_surrogates(value: Any) -> None:
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("invalid Unicode string") from error
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_surrogates(key)
            _assert_no_surrogates(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_surrogates(item)


def strict_json_bytes(payload: bytes) -> Any:
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid strict JSON") from error
    _assert_no_surrogates(value)
    return value


def _resolve_repo_path(root: Path, path: str) -> Path:
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("repository path escaped")
    return resolved


def _assert_regular_metadata(path: Path, expected_bytes: int) -> None:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError("bound input is not a regular file")
    if info.st_size != expected_bytes:
        raise ValueError("bound input size changed")


def read_bound_bytes(path: Path, *, expected_bytes: int, expected_sha256: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size != expected_bytes:
            raise ValueError("bound input metadata changed")
        chunks: list[bytes] = []
        remaining = expected_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(payload) != expected_bytes or sha256_bytes(payload) != expected_sha256:
        raise ValueError("bound input content changed")
    return payload


def _validate_packet_shape(value: Any) -> list[str]:
    if not isinstance(value, list) or len(value) != 32:
        raise ValueError("owner packet inventory changed")
    record_ids: list[str] = []
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {
            "language_mode",
            "messages",
            "record_id",
            "skill_id",
            "split",
        }:
            raise ValueError("owner packet row fields changed")
        record_id = row.get("record_id")
        messages = row.get("messages")
        if (
            not isinstance(record_id, str)
            or _RECORD_ID_RE.fullmatch(record_id) is None
            or row.get("language_mode") != "banglish"
            or row.get("skill_id") != "conversation"
            or row.get("split") != "train"
            or not isinstance(messages, list)
            or len(messages) != 2
        ):
            raise ValueError("owner packet row contract changed")
        for message, role in zip(messages, ("user", "assistant"), strict=True):
            if (
                not isinstance(message, Mapping)
                or set(message) != {"content", "role"}
                or message.get("role") != role
                or not isinstance(message.get("content"), str)
                or not message["content"]
            ):
                raise ValueError("owner packet message contract changed")
        record_ids.append(record_id)
    if len(set(record_ids)) != 32:
        raise ValueError("owner packet record IDs changed")
    return record_ids


def _validate_reviewer_alias(value: Any) -> None:
    if not isinstance(value, str) or _REVIEWER_RE.fullmatch(value) is None:
        raise ValueError("owner reviewer alias contract changed")
    folded = value.casefold()
    if any(marker in folded for marker in _AI_ALIAS_MARKERS):
        raise ValueError("automated reviewer alias prohibited")


def _validate_timestamp(value: Any) -> None:
    if not isinstance(value, str) or len(value) != 20:
        raise ValueError("owner timestamp contract changed")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError("owner timestamp contract changed") from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ValueError("owner timestamp contract changed")


def _validate_decisions_shape(
    value: Any,
    *,
    expected_record_ids: Sequence[str],
    expected_sample_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "decisions",
        "reviewed_at",
        "reviewer_id",
        "sample_sha256",
        "schema_version",
    }:
        raise ValueError("owner decision root fields changed")
    if (
        value.get("schema_version") != DECISIONS_SCHEMA_VERSION
        or value.get("sample_sha256") != expected_sample_sha256
    ):
        raise ValueError("owner decision binding changed")
    _validate_reviewer_alias(value.get("reviewer_id"))
    _validate_timestamp(value.get("reviewed_at"))
    decisions = value.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != len(expected_record_ids):
        raise ValueError("owner decision inventory changed")
    accepted = 0
    rejected = 0
    nonempty_notes = 0
    observed: list[str] = []
    for decision in decisions:
        if not isinstance(decision, Mapping) or set(decision) != {
            "accepted",
            "note",
            "record_id",
        }:
            raise ValueError("owner decision row fields changed")
        record_id = decision.get("record_id")
        accepted_value = decision.get("accepted")
        note = decision.get("note")
        if (
            not isinstance(record_id, str)
            or _RECORD_ID_RE.fullmatch(record_id) is None
            or type(accepted_value) is not bool
            or not isinstance(note, str)
            or len(note) > 2_000
        ):
            raise ValueError("owner decision row contract changed")
        observed.append(record_id)
        accepted += int(accepted_value)
        rejected += int(not accepted_value)
        nonempty_notes += int(bool(note))
    if observed != list(expected_record_ids) or len(set(observed)) != len(observed):
        raise ValueError("owner decision record order changed")
    if accepted < 29:
        raise ValueError("owner decision threshold not met")
    return {
        "accepted": accepted,
        "rejected": rejected,
        "nonempty_notes": nonempty_notes,
        "reviewed_at": value["reviewed_at"],
    }


def _reviewed_ids_domain_sha256(record_ids: Sequence[str]) -> str:
    return sha256_bytes(
        b"bedstc2-owner-reviewed-record-ids-v0.1\0"
        + canonical_json_bytes(sorted(record_ids))
    )


def _assert_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _FORBIDDEN_RECEIPT_KEYS:
                raise ValueError("raw or private owner-review key retained")
            _assert_no_forbidden_keys(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_forbidden_keys(item)


def _input_path(root: Path, name: str) -> Path:
    binding = EXPECTED_INPUTS[name]
    if name == "decisions":
        path = Path(str(binding["path"])).resolve()
        if path != EXPECTED_EXTERNAL_DECISIONS_PATH:
            raise ValueError("external decision path changed")
        return path
    return _resolve_repo_path(root, str(binding["path"]))


def _read_json_binding(root: Path, name: str) -> tuple[Any, bytes]:
    binding = EXPECTED_INPUTS[name]
    payload = read_bound_bytes(
        _input_path(root, name),
        expected_bytes=int(binding["bytes"]),
        expected_sha256=str(binding["sha256"]),
    )
    value = strict_json_bytes(payload)
    validate_binding_hash_domain(name, value, binding)
    return value, payload


def validate_binding_hash_domain(
    name: str, value: Any, binding: Mapping[str, Any]
) -> None:
    if name in {"curation_report", "curation_verification"}:
        if not isinstance(value, Mapping):
            raise ValueError("bound sealed JSON root changed")
        seal_key = (
            "report_sha256" if name == "curation_report" else "verification_sha256"
        )
        unsigned = {key: item for key, item in value.items() if key != seal_key}
        if (
            value.get(seal_key) != binding["canonical_sha256"]
            or sha256_value(unsigned) != binding["canonical_sha256"]
        ):
            raise ValueError("bound internal JSON seal changed")
        return
    if sha256_value(value) != binding["canonical_sha256"]:
        raise ValueError("bound canonical JSON changed")


def _validate_upstream(
    report: Mapping[str, Any], verification: Mapping[str, Any]
) -> None:
    validate_curation_report(report)
    validate_curation_verification(verification)
    if (
        report.get("schema_version")
        != "lumi.sft-bedstc2-banglish-conversation-curation-report.v0.7"
        or report.get("status")
        != "PASS_nonproduction_bedstc2_candidate_owner_review_pending"
        or report.get("report_sha256")
        != EXPECTED_INPUTS["curation_report"]["canonical_sha256"]
        or verification.get("schema_version")
        != "lumi.sft-bedstc2-banglish-conversation-curation-verification.v0.7"
        or verification.get("status")
        != "PASS_verified_nonproduction_bedstc2_candidate_owner_review_pending"
        or verification.get("verification_sha256")
        != EXPECTED_INPUTS["curation_verification"]["canonical_sha256"]
        or verification.get("report_file_sha256")
        != EXPECTED_INPUTS["curation_report"]["sha256"]
        or report.get("candidate")
        != {
            key: value
            for key, value in EXPECTED_CANDIDATE.items()
            if key != "file_sha256_attested_not_opened"
        }
        or report.get("artifacts", {}).get("candidate", {}).get("sha256")
        != EXPECTED_CANDIDATE["file_sha256_attested_not_opened"]
        or report.get("owner_review")
        != {
            "minimum_accepts": 29,
            "packet_file_sha256": EXPECTED_INPUTS["owner_packet"]["sha256"],
            "packet_records": 32,
            "packet_semantic_sha256": EXPECTED_INPUTS["owner_packet"][
                "canonical_sha256"
            ],
            "status": "PENDING_OWNER_REVIEW",
        }
        or report.get("provenance") != EXPECTED_PROVENANCE
        or report.get("rights")
        != {
            "license": "apache-2.0",
            "upstream_dstc2_rights_chain_independently_verified": False,
        }
        or report.get("production_blockers")
        != [
            "upstream_dstc2_rights_chain_not_independently_verified",
            "owner_quality_review_not_complete",
            "source_promotion_and_full_cross_source_reaudit_not_authorized",
            "production_dataset_seal_and_sft_authorization_not_complete",
        ]
        or report.get("production_eligible") is not False
        or verification.get("production_eligible") is not False
    ):
        raise ValueError("upstream Be-DSTC2 evidence changed")


def load_plan(path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resolved_plan = Path(path).resolve()
    expected_plan = (root / EXPECTED_PLAN_PATH).resolve()
    if resolved_plan != expected_plan:
        raise ValueError("owner-review plan path changed")
    plan = strict_json_bytes(resolved_plan.read_bytes())
    expected_keys = {
        "authorization",
        "authorization_sha256",
        "authorization_text",
        "finalizer",
        "implementation",
        "inputs",
        "limits",
        "output",
        "owner_review_contract",
        "schema_version",
        "status",
        "test",
        "verifier",
    }
    if (
        not isinstance(plan, dict)
        or set(plan) != expected_keys
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status") != PLAN_STATUS
        or plan.get("authorization_text") != EXPECTED_AUTHORIZATION_TEXT
        or plan.get("authorization_sha256") != EXPECTED_AUTHORIZATION_SHA256
        or sha256_bytes(plan["authorization_text"].encode())
        != EXPECTED_AUTHORIZATION_SHA256
        or plan.get("authorization") != EXPECTED_AUTHORIZATION
        or plan.get("limits") != EXPECTED_LIMITS
        or plan.get("inputs") != EXPECTED_INPUTS
        or plan.get("output") != EXPECTED_OUTPUT
        or plan.get("owner_review_contract")
        != {
            "accepted_records": 32,
            "decision_schema_version": DECISIONS_SCHEMA_VERSION,
            "minimum_accepts": 29,
            "nonempty_notes": 0,
            "receipt_marker_bytes": len(EXPECTED_RECEIPT_MARKER),
            "receipt_marker_sha256": sha256_bytes(EXPECTED_RECEIPT_MARKER),
            "rejected_records": 0,
            "reviewed_at_claimed_utc": EXPECTED_REVIEWED_AT,
            "reviewed_record_ids_domain_sha256": EXPECTED_REVIEWED_IDS_DOMAIN_SHA256,
            "sample_records": 32,
            "verification_marker_bytes": len(EXPECTED_VERIFICATION_MARKER),
            "verification_marker_sha256": sha256_bytes(
                EXPECTED_VERIFICATION_MARKER
            ),
        }
    ):
        raise ValueError("owner-review plan contract changed")
    for name, expected_path in EXPECTED_CODE_PATHS.items():
        binding = plan.get(name)
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256"}
            or binding.get("path") != expected_path
            or not isinstance(binding.get("sha256"), str)
            or _SHA256_RE.fullmatch(binding["sha256"]) is None
        ):
            raise ValueError("owner-review code binding changed")
        code_path = _resolve_repo_path(root, expected_path)
        if not code_path.is_file() or file_sha256(code_path) != binding["sha256"]:
            raise ValueError("owner-review code hash changed")
    for name, binding in EXPECTED_INPUTS.items():
        _assert_regular_metadata(_input_path(root, name), int(binding["bytes"]))
    for output_path in EXPECTED_OUTPUT.values():
        resolved = _resolve_repo_path(root, output_path)
        if resolved.parent != (root / "evidence").resolve():
            raise ValueError("owner-review output path changed")
    return plan


def build_receipt(
    plan: Mapping[str, Any], *, plan_file_sha256: str, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    packet, _ = _read_json_binding(root, "owner_packet")
    packet_ids = _validate_packet_shape(packet)
    decisions, _ = _read_json_binding(root, "decisions")
    outcome = _validate_decisions_shape(
        decisions,
        expected_record_ids=packet_ids,
        expected_sample_sha256=EXPECTED_INPUTS["owner_packet"]["canonical_sha256"],
    )
    report, _ = _read_json_binding(root, "curation_report")
    verification, _ = _read_json_binding(root, "curation_verification")
    if not isinstance(report, Mapping) or not isinstance(verification, Mapping):
        raise ValueError("upstream evidence root changed")
    _validate_upstream(report, verification)
    reviewed_ids_sha256 = _reviewed_ids_domain_sha256(packet_ids)
    if (
        reviewed_ids_sha256 != EXPECTED_REVIEWED_IDS_DOMAIN_SHA256
        or outcome
        != {
            "accepted": 32,
            "rejected": 0,
            "nonempty_notes": 0,
            "reviewed_at": EXPECTED_REVIEWED_AT,
        }
    ):
        raise ValueError("bound owner-review outcome changed")
    receipt: dict[str, Any] = {
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            "authorization_sha256": EXPECTED_AUTHORIZATION_SHA256,
            "curation_report_file_sha256": EXPECTED_INPUTS["curation_report"][
                "sha256"
            ],
            "curation_report_internal_sha256": EXPECTED_INPUTS[
                "curation_report"
            ]["canonical_sha256"],
            "curation_verification_file_sha256": EXPECTED_INPUTS[
                "curation_verification"
            ]["sha256"],
            "curation_verification_internal_sha256": EXPECTED_INPUTS[
                "curation_verification"
            ]["canonical_sha256"],
            "decisions_bytes": EXPECTED_INPUTS["decisions"]["bytes"],
            "decisions_canonical_sha256": EXPECTED_INPUTS["decisions"][
                "canonical_sha256"
            ],
            "decisions_file_sha256": EXPECTED_INPUTS["decisions"]["sha256"],
            "owner_packet_bytes": EXPECTED_INPUTS["owner_packet"]["bytes"],
            "owner_packet_file_sha256": EXPECTED_INPUTS["owner_packet"]["sha256"],
            "owner_packet_semantic_sha256": EXPECTED_INPUTS["owner_packet"][
                "canonical_sha256"
            ],
            "plan_file_sha256": plan_file_sha256,
            "reviewed_record_ids_domain_sha256": reviewed_ids_sha256,
        },
        "candidate": EXPECTED_CANDIDATE,
        "decision_notes_retained": False,
        "evaluation_prohibition": EXPECTED_EVALUATION_PROHIBITION,
        "human_owner_review_claim_self_attested": True,
        "next_gate": EXPECTED_NEXT_GATE,
        "owner_review": {
            "accepted_records": outcome["accepted"],
            "all_sampled_rows_accepted": outcome["rejected"] == 0,
            "automated_reviewer_markers_absent": True,
            "rejected_records": outcome["rejected"],
            "reviewed_at_claimed_utc": outcome["reviewed_at"],
            "sample_records": 32,
            "threshold": {
                "minimum_accepts": 29,
                "observed_accepts": outcome["accepted"],
                "satisfied": outcome["accepted"] >= 29,
            },
        },
        "owner_review_claim": EXPECTED_OWNER_REVIEW_CLAIM,
        "production_eligible": False,
        "provenance": EXPECTED_PROVENANCE,
        "raw_training_text_retained": False,
        "receipt_sha256": "",
        "record_ids_retained": False,
        "remaining_blockers": EXPECTED_REMAINING_BLOCKERS,
        "reviewer_alias_retained": False,
        "rights": EXPECTED_RIGHTS,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": RECEIPT_STATUS,
    }
    receipt["receipt_sha256"] = sha256_value(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    validate_receipt(receipt)
    return receipt


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    expected_keys = {
        "authorization",
        "bindings",
        "candidate",
        "decision_notes_retained",
        "evaluation_prohibition",
        "human_owner_review_claim_self_attested",
        "next_gate",
        "owner_review",
        "owner_review_claim",
        "production_eligible",
        "provenance",
        "raw_training_text_retained",
        "receipt_sha256",
        "record_ids_retained",
        "remaining_blockers",
        "reviewer_alias_retained",
        "rights",
        "schema_version",
        "status",
    }
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    bindings = receipt.get("bindings")
    if (
        set(receipt) != expected_keys
        or receipt.get("receipt_sha256") != sha256_value(unsigned)
        or receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != RECEIPT_STATUS
        or receipt.get("authorization") != EXPECTED_AUTHORIZATION
        or receipt.get("candidate") != EXPECTED_CANDIDATE
        or receipt.get("evaluation_prohibition") != EXPECTED_EVALUATION_PROHIBITION
        or receipt.get("next_gate") != EXPECTED_NEXT_GATE
        or receipt.get("owner_review_claim") != EXPECTED_OWNER_REVIEW_CLAIM
        or receipt.get("provenance") != EXPECTED_PROVENANCE
        or receipt.get("rights") != EXPECTED_RIGHTS
        or receipt.get("remaining_blockers") != EXPECTED_REMAINING_BLOCKERS
        or receipt.get("production_eligible") is not False
        or receipt.get("raw_training_text_retained") is not False
        or receipt.get("decision_notes_retained") is not False
        or receipt.get("record_ids_retained") is not False
        or receipt.get("reviewer_alias_retained") is not False
        or receipt.get("human_owner_review_claim_self_attested") is not True
        or receipt.get("owner_review")
        != {
            "accepted_records": 32,
            "all_sampled_rows_accepted": True,
            "automated_reviewer_markers_absent": True,
            "rejected_records": 0,
            "reviewed_at_claimed_utc": EXPECTED_REVIEWED_AT,
            "sample_records": 32,
            "threshold": {
                "minimum_accepts": 29,
                "observed_accepts": 32,
                "satisfied": True,
            },
        }
        or not isinstance(bindings, Mapping)
        or set(bindings)
        != {
            "authorization_sha256",
            "curation_report_file_sha256",
            "curation_report_internal_sha256",
            "curation_verification_file_sha256",
            "curation_verification_internal_sha256",
            "decisions_bytes",
            "decisions_canonical_sha256",
            "decisions_file_sha256",
            "owner_packet_bytes",
            "owner_packet_file_sha256",
            "owner_packet_semantic_sha256",
            "plan_file_sha256",
            "reviewed_record_ids_domain_sha256",
        }
        or bindings.get("authorization_sha256") != EXPECTED_AUTHORIZATION_SHA256
        or bindings.get("curation_report_file_sha256")
        != EXPECTED_INPUTS["curation_report"]["sha256"]
        or bindings.get("curation_report_internal_sha256")
        != EXPECTED_INPUTS["curation_report"]["canonical_sha256"]
        or bindings.get("curation_verification_file_sha256")
        != EXPECTED_INPUTS["curation_verification"]["sha256"]
        or bindings.get("curation_verification_internal_sha256")
        != EXPECTED_INPUTS["curation_verification"]["canonical_sha256"]
        or bindings.get("decisions_bytes") != EXPECTED_INPUTS["decisions"]["bytes"]
        or bindings.get("decisions_file_sha256")
        != EXPECTED_INPUTS["decisions"]["sha256"]
        or bindings.get("decisions_canonical_sha256")
        != EXPECTED_INPUTS["decisions"]["canonical_sha256"]
        or bindings.get("owner_packet_bytes")
        != EXPECTED_INPUTS["owner_packet"]["bytes"]
        or bindings.get("owner_packet_file_sha256")
        != EXPECTED_INPUTS["owner_packet"]["sha256"]
        or bindings.get("owner_packet_semantic_sha256")
        != EXPECTED_INPUTS["owner_packet"]["canonical_sha256"]
        or bindings.get("reviewed_record_ids_domain_sha256")
        != EXPECTED_REVIEWED_IDS_DOMAIN_SHA256
        or not isinstance(bindings.get("plan_file_sha256"), str)
        or _SHA256_RE.fullmatch(bindings["plan_file_sha256"]) is None
    ):
        raise ValueError("Be-DSTC2 owner-review receipt changed")
    _assert_no_forbidden_keys(receipt)


def build_verification(
    receipt: Mapping[str, Any],
    *,
    receipt_file_sha256: str,
    receipt_file_bytes: int,
    plan_file_sha256: str,
) -> dict[str, Any]:
    validate_receipt(receipt)
    if (
        _SHA256_RE.fullmatch(receipt_file_sha256) is None
        or receipt_file_bytes <= 0
        or receipt["bindings"]["plan_file_sha256"] != plan_file_sha256
    ):
        raise ValueError("receipt file binding changed")
    verification: dict[str, Any] = {
        "checks": EXPECTED_CHECKS,
        "independent_replay": {
            "baseline_row_reads": 0,
            "candidate_row_reads": 0,
            "curation_report_reads": 1,
            "curation_verification_reads": 1,
            "decision_reads": 1,
            "decision_rows": 32,
            "owner_packet_reads": 1,
            "owner_packet_rows": 32,
            "receipt_reads": 1,
            "source_line_reads": 0,
            "test_object_reads": 0,
        },
        "plan_file_sha256": plan_file_sha256,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "receipt_file_bytes": receipt_file_bytes,
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_internal_sha256": receipt["receipt_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": VERIFICATION_STATUS,
        "verification_sha256": "",
        "zero_use": {"gpu_use": 0, "inference_calls": 0, "network_bytes": 0},
    }
    verification["verification_sha256"] = sha256_value(
        {
            key: value
            for key, value in verification.items()
            if key != "verification_sha256"
        }
    )
    validate_verification(verification)
    return verification


def validate_verification(verification: Mapping[str, Any]) -> None:
    expected_keys = {
        "checks",
        "independent_replay",
        "plan_file_sha256",
        "production_eligible",
        "raw_training_text_retained",
        "receipt_file_bytes",
        "receipt_file_sha256",
        "receipt_internal_sha256",
        "schema_version",
        "status",
        "verification_sha256",
        "zero_use",
    }
    unsigned = {
        key: value for key, value in verification.items() if key != "verification_sha256"
    }
    replay = verification.get("independent_replay")
    if (
        set(verification) != expected_keys
        or verification.get("verification_sha256") != sha256_value(unsigned)
        or verification.get("schema_version") != VERIFICATION_SCHEMA_VERSION
        or verification.get("status") != VERIFICATION_STATUS
        or verification.get("checks") != EXPECTED_CHECKS
        or verification.get("production_eligible") is not False
        or verification.get("raw_training_text_retained") is not False
        or verification.get("zero_use")
        != {"gpu_use": 0, "inference_calls": 0, "network_bytes": 0}
        or replay
        != {
            "baseline_row_reads": 0,
            "candidate_row_reads": 0,
            "curation_report_reads": 1,
            "curation_verification_reads": 1,
            "decision_reads": 1,
            "decision_rows": 32,
            "owner_packet_reads": 1,
            "owner_packet_rows": 32,
            "receipt_reads": 1,
            "source_line_reads": 0,
            "test_object_reads": 0,
        }
        or not isinstance(verification.get("receipt_file_bytes"), int)
        or verification["receipt_file_bytes"] <= 0
        or any(
            not isinstance(verification.get(field), str)
            or _SHA256_RE.fullmatch(verification[field]) is None
            for field in (
                "plan_file_sha256",
                "receipt_file_sha256",
                "receipt_internal_sha256",
            )
        )
    ):
        raise ValueError("Be-DSTC2 owner-review verification changed")
    _assert_no_forbidden_keys(verification)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def start_marker(path: Path, marker: bytes) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("owner-review marker inode changed")
        written = os.write(descriptor, marker)
        if written != len(marker):
            raise OSError("short owner-review marker write")
        os.fsync(descriptor)
        identity = (info.st_dev, info.st_ino)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return identity


def finish_marker(
    path: Path,
    marker: bytes,
    payload: bytes,
    identity: tuple[int, int],
) -> None:
    flags = os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or (info.st_dev, info.st_ino) != identity
        ):
            raise ValueError("owner-review marker inode changed")
        current = os.read(descriptor, len(marker) + 1)
        if current != marker:
            raise ValueError("owner-review marker changed")
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short owner-review final write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def preflight_finalizer(plan: Mapping[str, Any], root: Path) -> tuple[Path, Path]:
    receipt_path = _resolve_repo_path(root, plan["output"]["receipt_path"])
    verification_path = _resolve_repo_path(root, plan["output"]["verification_path"])
    if receipt_path.exists() or verification_path.exists():
        raise FileExistsError("owner-review finalization already started")
    return receipt_path, verification_path


def preflight_verifier(plan: Mapping[str, Any], root: Path) -> tuple[Path, Path]:
    receipt_path = _resolve_repo_path(root, plan["output"]["receipt_path"])
    verification_path = _resolve_repo_path(root, plan["output"]["verification_path"])
    if not receipt_path.is_file() or verification_path.exists():
        raise FileExistsError("owner-review verification preflight changed")
    if receipt_path.stat().st_size <= len(EXPECTED_RECEIPT_MARKER):
        raise ValueError("owner-review receipt is an incomplete marker")
    return receipt_path, verification_path


def prospective_private_write_bytes(
    *,
    plan: Mapping[str, Any],
    root: Path,
    receipt_bytes: bytes,
    verification_bytes: bytes,
) -> int:
    static = (root / EXPECTED_PLAN_PATH).stat().st_size
    for name in EXPECTED_CODE_PATHS:
        static += _resolve_repo_path(root, plan[name]["path"]).stat().st_size
    return (
        static
        + len(EXPECTED_RECEIPT_MARKER)
        + len(receipt_bytes)
        + len(EXPECTED_VERIFICATION_MARKER)
        + len(verification_bytes)
    )


__all__ = [
    "EXPECTED_AUTHORIZATION_SHA256",
    "EXPECTED_CHECKS",
    "EXPECTED_INPUTS",
    "EXPECTED_LIMITS",
    "EXPECTED_OUTPUT",
    "EXPECTED_PLAN_PATH",
    "EXPECTED_RECEIPT_MARKER",
    "EXPECTED_VERIFICATION_MARKER",
    "RECEIPT_SCHEMA_VERSION",
    "RECEIPT_STATUS",
    "VERIFICATION_SCHEMA_VERSION",
    "VERIFICATION_STATUS",
    "_read_json_binding",
    "_reviewed_ids_domain_sha256",
    "_validate_decisions_shape",
    "_validate_packet_shape",
    "_validate_upstream",
    "build_receipt",
    "build_verification",
    "canonical_json_bytes",
    "file_sha256",
    "finish_marker",
    "json_bytes",
    "load_plan",
    "preflight_finalizer",
    "preflight_verifier",
    "prospective_private_write_bytes",
    "read_bound_bytes",
    "sha256_bytes",
    "sha256_value",
    "start_marker",
    "strict_json_bytes",
    "validate_binding_hash_domain",
    "validate_receipt",
    "validate_verification",
]
