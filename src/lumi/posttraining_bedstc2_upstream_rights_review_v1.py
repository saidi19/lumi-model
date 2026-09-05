"""Bounded, local-only Be-DSTC2 upstream-rights evidence review.

The review reads exactly seven hash-bound local files.  It records only
allowlisted booleans, counts, identifiers, and digests; document text and
training rows are never retained.  Repository-level Apache-2.0 text is not
treated as proof of the underlying DSTC2 rights chain.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PLAN_SCHEMA_VERSION = "lumi.sft-bedstc2-upstream-rights-review-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-bedstc2-upstream-rights-review-report.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-bedstc2-upstream-rights-review-verification.v0.1"
)
PLAN_STATUS = "authorized_local_only_bedstc2_upstream_rights_review"
REPORT_STATUS = "STOP_local_evidence_insufficient_upstream_dstc2_rights_unresolved"
VERIFICATION_STATUS = (
    "PASS_verified_STOP_local_evidence_insufficient_upstream_dstc2_rights_unresolved"
)

EXPECTED_PLAN_PATH = Path(
    "configs/lumi-4b-sft-bedstc2-upstream-rights-review-v0.1.json"
)
EXPECTED_AUTHORIZATION_TEXT = "ok next step shuru de"
EXPECTED_AUTHORIZATION_SHA256 = (
    "7a2317268ba301ceb4a30bb9aa558fad1796139ccc3f5e87722f1e944c4ddcd3"
)
EXPECTED_CODE_PATHS = {
    "implementation": "src/lumi/posttraining_bedstc2_upstream_rights_review_v1.py",
    "runner": "scripts/review_sft_bedstc2_upstream_rights_v1.py",
    "verifier": "scripts/verify_sft_bedstc2_upstream_rights_v1.py",
    "test": "tests/test_bedstc2_upstream_rights_review_v1.py",
}
EXPECTED_OUTPUT = {
    "report_path": "evidence/lumi-4b-sft-bedstc2-upstream-rights-review-v0.1.json",
    "verification_path": (
        "evidence/lumi-4b-sft-bedstc2-upstream-rights-review-verification-v0.1.json"
    ),
}
EXPECTED_INPUTS: dict[str, dict[str, Any]] = {
    "acquisition_receipt": {
        "path": "evidence/lumi-4b-sft-bedstc2-acquisition-receipt-v0.1.json",
        "bytes": 2_246,
        "sha256": "50573eee3db949169ac5bd7cf38667e5455a1c651b30e314296ffb2097112c4e",
        "internal_sha256": "a7f0f14bed39b109237c845c66dec021c10c6f81877ae615461a9b8c02c8e5b4",
        "seal_key": "receipt_sha256",
    },
    "repository_readme": {
        "path": (
            "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-001/"
            "private/raw/README.md"
        ),
        "bytes": 5_978,
        "sha256": "b6733216e7b23677b1b724c45b224f362eb259bf0e5686521d43eaf9a74cf2e6",
    },
    "repository_license": {
        "path": (
            "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-001/"
            "private/raw/LICENSE"
        ),
        "bytes": 11_357,
        "sha256": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    },
    "curation_report": {
        "path": "evidence/lumi-4b-sft-bedstc2-banglish-conversation-curation-v0.7.json",
        "bytes": 26_182,
        "sha256": "81548913d2a9fe1d5cdf9ecbc2f927e64f20033a34931425f12821a15b7305df",
        "internal_sha256": "bf690b11e0018b1b96e996c37cf511db80eb60e5b6cadfface82326f68fc296b",
        "seal_key": "report_sha256",
    },
    "curation_verification": {
        "path": (
            "evidence/lumi-4b-sft-bedstc2-banglish-conversation-curation-"
            "verification-v0.7.json"
        ),
        "bytes": 2_914,
        "sha256": "d5b87bd7882111b882f5d6fc33401e58a1ec694d5f801f750ab8ed3bc6720a61",
        "internal_sha256": "ba8597ab5fa6b1468b9ef1fb466b4082fddcd679a322a15d71a807fb63519241",
        "seal_key": "verification_sha256",
    },
    "owner_receipt": {
        "path": (
            "evidence/lumi-4b-sft-bedstc2-banglish-conversation-owner-review-v0.1.json"
        ),
        "bytes": 4_430,
        "sha256": "d2ff312c30a102c4996eb8129d0dcadfac615b56c4858788dc386098b8dc1e6f",
        "internal_sha256": "99eacbcd1f2ab4aa8c5ea91867e6e5aca2e411639c88880b4208bcbb22621035",
        "seal_key": "receipt_sha256",
    },
    "owner_verification": {
        "path": (
            "evidence/lumi-4b-sft-bedstc2-banglish-conversation-owner-review-"
            "verification-v0.1.json"
        ),
        "bytes": 1_677,
        "sha256": "7695df6012e277d0f3c914c4a58488015fb8b93eaebb91f980e6439b3e539bea",
        "internal_sha256": "0b5dbfa7fe07dd259ce1848b8a93eacaa735c5ba4d69a7b978a132f5377762cb",
        "seal_key": "verification_sha256",
    },
}
EXPECTED_LIMITS = {
    "passes": 2,
    "bound_input_files_per_pass": 7,
    "bound_input_bytes_per_pass": 54_784,
    "bound_input_file_reads_total": 14,
    "bound_input_bytes_total": 109_568,
    "acquisition_receipt_reads": 2,
    "repository_readme_reads": 2,
    "repository_license_reads": 2,
    "curation_report_reads": 2,
    "curation_verification_reads": 2,
    "owner_receipt_reads": 2,
    "owner_verification_reads": 2,
    "generated_report_reads": 1,
    "source_line_reads": 0,
    "candidate_row_reads": 0,
    "baseline_row_reads": 0,
    "test_object_reads": 0,
    "runtime_seconds_per_process": 150,
    "runtime_seconds_total": 300,
    "private_write_bytes": 1_048_576,
    "network_bytes": 0,
    "gpu_use": 0,
    "inference_calls": 0,
}
EXPECTED_AUTHORIZATION = {
    "local_upstream_rights_review_authorized": True,
    "authoritative_upstream_document_acquisition_authorized": False,
    "candidate_or_source_mutation_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_or_download_authorized": False,
    "gpu_or_inference_authorized": False,
    "legal_review_authorized": False,
    "network_authorized": False,
    "production_promotion_authorized": False,
    "registry_mutation_authorized": False,
    "sft_training_authorized": False,
    "source_candidate_or_baseline_replay_authorized": False,
    "test_object_access_authorized": False,
    "unrelated_action_authorized": False,
}

EXPECTED_SOURCE_CONTRACT = {
    "dataset_id": "sumanbanerjee1/Code-Mixed-Dialog",
    "license_id": "apache-2.0",
    "origin": "mixed",
    "revision": "9df1d4dc800548a883f8bc1a9ce4116c77aebc02",
}
EXPECTED_PROVENANCE = (
    "human_in_house_annotated_source_derived_bengali_code_mixed_surface_wording_"
    "with_underlying_dstc2_authorship_and_origin_not_established"
)
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
EXPECTED_OWNER_REVIEW = {
    "accepted_records": 32,
    "minimum_accepts": 29,
    "rejected_records": 0,
    "sample_records": 32,
    "threshold_satisfied": True,
}
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
EXPECTED_DOCUMENT_OBSERVATIONS = {
    "repository_apache_2_0_text_present": True,
    "readme_identifies_code_mixed_dialog": True,
    "readme_says_dstc2_text_was_adapted": True,
    "filled_repository_copyright_notice_present": False,
}
EXPECTED_RIGHTS_CRITERIA = {
    "upstream_dstc2_licensor_identified": False,
    "upstream_dstc2_license_or_permission_bound": False,
    "exact_bengali_object_scope_bound": False,
    "derivative_and_commercial_use_grant_bound": False,
    "attribution_notice_duties_for_upstream_bound": False,
}
EXPECTED_REASON_CODES = [
    "repository_license_not_proof_of_upstream_dstc2_rights",
    "upstream_licensor_or_rightsholder_not_identified",
    "applicable_upstream_terms_not_locally_bound",
    "derivative_and_commercial_use_permission_not_established",
    "exact_object_scope_and_attribution_duties_not_established",
]
EXPECTED_NEXT_GATE = {
    "action": (
        "separately_authorized_authoritative_upstream_dstc2_rights_document_"
        "acquisition_and_legal_review"
    ),
    "authoritative_document_acquisition_authorized": False,
    "legal_review_authorized": False,
    "network_authorized": False,
    "separate_authorization_required": True,
}
EXPECTED_CHECKS = {
    "all_seven_bound_inputs_hash_verified": True,
    "curation_and_owner_evidence_internal_seals_verified": True,
    "owner_review_completion_preserved": True,
    "repository_license_not_treated_as_upstream_dstc2_permission": True,
    "source_candidate_baseline_and_test_objects_not_opened": True,
    "unresolved_rights_fail_closed": True,
    "provenance_evaluation_and_blockers_preserved": True,
    "no_promotion_seal_training_registry_network_gpu_or_inference": True,
}

README_MARKERS = (
    "# Code-Mixed-Dialog",
    "A Dataset for Building Code-Mixed Goal Oriented Conversation Systems.",
    (
        "Specifically, we take the text from the DSTC2 restaurant reservation "
        "dataset and create code-mixed versions of it"
    ),
    "The dataset and baseline implementations are provided here.",
)
LICENSE_MARKERS = (
    "Apache License",
    "Version 2.0, January 2004",
    "Copyright [yyyy] [name of copyright owner]",
)
EXPECTED_REPORT_MARKER = b'{"started":"bedstc2_upstream_rights_report_v0.1"}\n'
EXPECTED_VERIFICATION_MARKER = (
    b'{"started":"bedstc2_upstream_rights_verification_v0.1"}\n'
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_REPORT_KEYS = {
    "assistant_response",
    "content",
    "document_text",
    "excerpt",
    "messages",
    "prompt",
    "raw_text",
    "snippet",
    "source_text",
    "training_text",
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
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
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
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid strict JSON") from error
    _assert_no_surrogates(value)
    return value


def resolve_repo_path(root: Path, path: str | Path) -> Path:
    """Resolve lexically under ``root`` without following any path component.

    ``Path.resolve`` cannot be used here: it follows a symlink leaf before the
    caller gets a chance to reject it.  The returned path is normalized only
    with ``abspath``/``normpath`` and every component that already exists is
    checked with ``lstat``.  A missing leaf is allowed for fresh output paths.
    """

    root = Path(os.path.abspath(os.fspath(root)))
    raw = Path(path)
    joined = raw if raw.is_absolute() else root / raw
    candidate = Path(os.path.abspath(os.path.normpath(os.fspath(joined))))
    if not candidate.is_relative_to(root):
        raise ValueError("repository path escaped")
    root_info = os.lstat(root)
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ValueError("repository root metadata changed")
    current = root
    parts = candidate.relative_to(root).parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("repository path contains a symlink")
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise ValueError("repository path parent is not a directory")
    return candidate


def read_bound_bytes(path: Path, *, expected_bytes: int, expected_sha256: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size != expected_bytes
        ):
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


def _input_path(root: Path, name: str) -> Path:
    return resolve_repo_path(root, str(EXPECTED_INPUTS[name]["path"]))


def _assert_input_metadata(root: Path, name: str) -> None:
    binding = EXPECTED_INPUTS[name]
    path = _input_path(root, name)
    info = path.lstat()
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_size != binding["bytes"]
    ):
        raise ValueError("bound input metadata changed")


def validate_internal_seal(name: str, value: Any) -> None:
    binding = EXPECTED_INPUTS[name]
    if not isinstance(value, Mapping) or "seal_key" not in binding:
        raise ValueError("sealed input contract changed")
    seal_key = str(binding["seal_key"])
    unsigned = {key: item for key, item in value.items() if key != seal_key}
    if (
        value.get(seal_key) != binding["internal_sha256"]
        or sha256_value(unsigned) != binding["internal_sha256"]
    ):
        raise ValueError("sealed input internal hash changed")


def _read_json_input(root: Path, name: str) -> Mapping[str, Any]:
    binding = EXPECTED_INPUTS[name]
    payload = read_bound_bytes(
        _input_path(root, name),
        expected_bytes=int(binding["bytes"]),
        expected_sha256=str(binding["sha256"]),
    )
    value = strict_json_bytes(payload)
    validate_internal_seal(name, value)
    return value


def _read_text_input(root: Path, name: str) -> str:
    binding = EXPECTED_INPUTS[name]
    payload = read_bound_bytes(
        _input_path(root, name),
        expected_bytes=int(binding["bytes"]),
        expected_sha256=str(binding["sha256"]),
    )
    try:
        return payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("rights document is not strict UTF-8") from error


def inspect_rights_documents(readme: str, license_text: str) -> dict[str, bool]:
    if not all(marker in readme for marker in README_MARKERS):
        raise ValueError("bound README rights markers changed")
    if not all(marker in license_text for marker in LICENSE_MARKERS):
        raise ValueError("bound LICENSE rights markers changed")
    return dict(EXPECTED_DOCUMENT_OBSERVATIONS)


def _validate_acquisition(value: Mapping[str, Any]) -> None:
    expected_objects = [
        {
            "bytes": 11_357,
            "git_blob_sha1": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64",
            "path": (
                "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/"
                "attempt-001/private/raw/LICENSE"
            ),
            "sha256": EXPECTED_INPUTS["repository_license"]["sha256"],
        },
        {
            "bytes": 5_978,
            "git_blob_sha1": "ef54e017c4bf440b89cf3644a76acef22cdf3981",
            "path": (
                "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/"
                "attempt-001/private/raw/README.md"
            ),
            "sha256": EXPECTED_INPUTS["repository_readme"]["sha256"],
        },
        {
            "bytes": 4_085_040,
            "git_blob_sha1": "4683420d14bf546c31e7698ce9ecd799ba0f88d3",
            "path": (
                "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/"
                "attempt-001/private/raw/data/bengali/dialog-dstc2-trn.txt"
            ),
            "sha256": "79644ed579c831626b1f2a2dfef1f804b835c9e3da19f93fe493c9fbbb3b591c",
        },
        {
            "bytes": 1_361_043,
            "git_blob_sha1": "543eb0aa02d9ec7f1af65516017a1bfcace1f636",
            "path": (
                "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/"
                "attempt-001/private/raw/data/bengali/dialog-dstc2-dev.txt"
            ),
            "sha256": "a46b98e4f3ff3c9575240fe42fee2d735fa61ec41aacd53d73627b33b7147116",
        },
        {
            "bytes": 19_767,
            "git_blob_sha1": "39bba927098a8792ff3b4f611bf35362eb0b8e4f",
            "path": (
                "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/"
                "attempt-001/private/raw/data/bengali/vocab_splits.json"
            ),
            "sha256": "f55a71418e699ece420b0206c74422f1889fbc9da24be4aea19c15d6912a6870",
        },
    ]
    if (
        value.get("schema_version") != "lumi.sft-bedstc2-acquisition-receipt.v0.1"
        or value.get("status") != "PASS_exact_five_object_acquisition_only"
        or value.get("revision") != EXPECTED_SOURCE_CONTRACT["revision"]
        or value.get("objects") != expected_objects
        or value.get("raw_training_text_in_receipt") is not False
        or value.get("checks", {}).get("test_object_accessed") is not False
    ):
        raise ValueError("Be-DSTC2 acquisition evidence changed")


def _validate_curation(
    report: Mapping[str, Any], verification: Mapping[str, Any]
) -> None:
    expected_candidate_without_hash = {
        key: value
        for key, value in EXPECTED_CANDIDATE.items()
        if key != "file_sha256_attested_not_opened"
    }
    if (
        report.get("schema_version")
        != "lumi.sft-bedstc2-banglish-conversation-curation-report.v0.7"
        or report.get("status")
        != "PASS_nonproduction_bedstc2_candidate_owner_review_pending"
        or report.get("candidate") != expected_candidate_without_hash
        or report.get("artifacts", {}).get("candidate", {}).get("sha256")
        != EXPECTED_CANDIDATE["file_sha256_attested_not_opened"]
        or report.get("provenance") != EXPECTED_PROVENANCE
        or report.get("rights") != EXPECTED_RIGHTS
        or report.get("evaluation_prohibition") != EXPECTED_EVALUATION_PROHIBITION
        or report.get("production_eligible") is not False
        or verification.get("schema_version")
        != "lumi.sft-bedstc2-banglish-conversation-curation-verification.v0.7"
        or verification.get("status")
        != "PASS_verified_nonproduction_bedstc2_candidate_owner_review_pending"
        or verification.get("report_file_sha256")
        != EXPECTED_INPUTS["curation_report"]["sha256"]
        or verification.get("production_eligible") is not False
    ):
        raise ValueError("Be-DSTC2 curation evidence changed")


def _validate_owner(
    receipt: Mapping[str, Any], verification: Mapping[str, Any]
) -> None:
    owner_review = receipt.get("owner_review", {})
    if (
        receipt.get("schema_version")
        != "lumi.sft-bedstc2-banglish-conversation-owner-review-receipt.v0.1"
        or receipt.get("status")
        != "owner_accepted_all_32_bedstc2_rows_nonproduction_rights_pending"
        or receipt.get("candidate") != EXPECTED_CANDIDATE
        or receipt.get("provenance") != EXPECTED_PROVENANCE
        or receipt.get("rights") != EXPECTED_RIGHTS
        or receipt.get("evaluation_prohibition") != EXPECTED_EVALUATION_PROHIBITION
        or receipt.get("remaining_blockers") != EXPECTED_REMAINING_BLOCKERS
        or receipt.get("production_eligible") is not False
        or owner_review.get("accepted_records") != 32
        or owner_review.get("rejected_records") != 0
        or owner_review.get("sample_records") != 32
        or owner_review.get("threshold", {}).get("minimum_accepts") != 29
        or owner_review.get("threshold", {}).get("satisfied") is not True
        or verification.get("schema_version")
        != "lumi.sft-bedstc2-banglish-conversation-owner-review-verification.v0.1"
        or verification.get("status")
        != "PASS_verified_owner_accepted_all_32_bedstc2_rows_nonproduction_rights_pending"
        or verification.get("receipt_file_sha256")
        != EXPECTED_INPUTS["owner_receipt"]["sha256"]
        or verification.get("production_eligible") is not False
    ):
        raise ValueError("Be-DSTC2 owner-review evidence changed")


def read_inputs_and_assess(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    acquisition = _read_json_input(root, "acquisition_receipt")
    readme = _read_text_input(root, "repository_readme")
    license_text = _read_text_input(root, "repository_license")
    curation_report = _read_json_input(root, "curation_report")
    curation_verification = _read_json_input(root, "curation_verification")
    owner_receipt = _read_json_input(root, "owner_receipt")
    owner_verification = _read_json_input(root, "owner_verification")
    _validate_acquisition(acquisition)
    _validate_curation(curation_report, curation_verification)
    _validate_owner(owner_receipt, owner_verification)
    observations = inspect_rights_documents(readme, license_text)
    return {
        "document_observations": observations,
        "rights_criteria": dict(EXPECTED_RIGHTS_CRITERIA),
        "criteria_satisfied": sum(EXPECTED_RIGHTS_CRITERIA.values()),
        "criteria_unresolved": sum(not value for value in EXPECTED_RIGHTS_CRITERIA.values()),
    }


def _report_bindings(plan_file_sha256: str) -> dict[str, Any]:
    bindings: dict[str, Any] = {"plan_file_sha256": plan_file_sha256}
    for name, item in EXPECTED_INPUTS.items():
        binding = {"bytes": item["bytes"], "file_sha256": item["sha256"]}
        if "internal_sha256" in item:
            binding["internal_sha256"] = item["internal_sha256"]
        bindings[name] = binding
    return bindings


def assemble_report(plan_file_sha256: str, assessment: Mapping[str, Any]) -> dict[str, Any]:
    if _SHA256_RE.fullmatch(plan_file_sha256) is None or assessment != {
        "document_observations": EXPECTED_DOCUMENT_OBSERVATIONS,
        "rights_criteria": EXPECTED_RIGHTS_CRITERIA,
        "criteria_satisfied": 0,
        "criteria_unresolved": 5,
    }:
        raise ValueError("rights assessment changed")
    report: dict[str, Any] = {
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": _report_bindings(plan_file_sha256),
        "candidate": EXPECTED_CANDIDATE,
        "checks": EXPECTED_CHECKS,
        "document_observations": EXPECTED_DOCUMENT_OBSERVATIONS,
        "document_snippets_retained": False,
        "document_text_retained": False,
        "evaluation_prohibition": EXPECTED_EVALUATION_PROHIBITION,
        "input_accounting": {
            "bound_input_bytes": 54_784,
            "bound_input_file_reads": 7,
            "bound_input_files": 7,
            "source_line_reads": 0,
            "candidate_row_reads": 0,
            "baseline_row_reads": 0,
            "test_object_reads": 0,
        },
        "next_gate": EXPECTED_NEXT_GATE,
        "owner_review": EXPECTED_OWNER_REVIEW,
        "production_eligible": False,
        "provenance": EXPECTED_PROVENANCE,
        "raw_training_text_retained": False,
        "reason_codes": EXPECTED_REASON_CODES,
        "remaining_blockers": EXPECTED_REMAINING_BLOCKERS,
        "report_sha256": "",
        "rights": EXPECTED_RIGHTS,
        "rights_assessment": {
            "criteria": EXPECTED_RIGHTS_CRITERIA,
            "criteria_satisfied": 0,
            "criteria_unresolved": 5,
        },
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_contract": EXPECTED_SOURCE_CONTRACT,
        "status": REPORT_STATUS,
    }
    report["report_sha256"] = sha256_value(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    validate_report(report)
    return report


def build_report(*, repository_root: str | Path, plan_file_sha256: str) -> dict[str, Any]:
    return assemble_report(plan_file_sha256, read_inputs_and_assess(repository_root))


def _assert_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _FORBIDDEN_REPORT_KEYS:
                raise ValueError("raw text key retained")
            _assert_no_forbidden_keys(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_forbidden_keys(item)


def validate_report(report: Mapping[str, Any]) -> None:
    expected_keys = {
        "authorization",
        "bindings",
        "candidate",
        "checks",
        "document_observations",
        "document_snippets_retained",
        "document_text_retained",
        "evaluation_prohibition",
        "input_accounting",
        "next_gate",
        "owner_review",
        "production_eligible",
        "provenance",
        "raw_training_text_retained",
        "reason_codes",
        "remaining_blockers",
        "report_sha256",
        "rights",
        "rights_assessment",
        "schema_version",
        "source_contract",
        "status",
    }
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    bindings = report.get("bindings")
    plan_hash = bindings.get("plan_file_sha256") if isinstance(bindings, Mapping) else None
    if (
        set(report) != expected_keys
        or report.get("report_sha256") != sha256_value(unsigned)
        or report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("status") != REPORT_STATUS
        or report.get("authorization") != EXPECTED_AUTHORIZATION
        or not isinstance(plan_hash, str)
        or _SHA256_RE.fullmatch(plan_hash) is None
        or bindings != _report_bindings(plan_hash)
        or report.get("candidate") != EXPECTED_CANDIDATE
        or report.get("checks") != EXPECTED_CHECKS
        or report.get("document_observations") != EXPECTED_DOCUMENT_OBSERVATIONS
        or report.get("document_snippets_retained") is not False
        or report.get("document_text_retained") is not False
        or report.get("evaluation_prohibition") != EXPECTED_EVALUATION_PROHIBITION
        or report.get("next_gate") != EXPECTED_NEXT_GATE
        or report.get("owner_review") != EXPECTED_OWNER_REVIEW
        or report.get("production_eligible") is not False
        or report.get("provenance") != EXPECTED_PROVENANCE
        or report.get("raw_training_text_retained") is not False
        or report.get("reason_codes") != EXPECTED_REASON_CODES
        or report.get("remaining_blockers") != EXPECTED_REMAINING_BLOCKERS
        or report.get("rights") != EXPECTED_RIGHTS
        or report.get("rights_assessment")
        != {
            "criteria": EXPECTED_RIGHTS_CRITERIA,
            "criteria_satisfied": 0,
            "criteria_unresolved": 5,
        }
        or report.get("source_contract") != EXPECTED_SOURCE_CONTRACT
        or report.get("input_accounting")
        != {
            "bound_input_bytes": 54_784,
            "bound_input_file_reads": 7,
            "bound_input_files": 7,
            "source_line_reads": 0,
            "candidate_row_reads": 0,
            "baseline_row_reads": 0,
            "test_object_reads": 0,
        }
    ):
        raise ValueError("Be-DSTC2 upstream-rights report changed")
    _assert_no_forbidden_keys(report)


def build_verification(
    report: Mapping[str, Any],
    *,
    report_file_sha256: str,
    report_file_bytes: int,
    plan_file_sha256: str,
) -> dict[str, Any]:
    validate_report(report)
    if (
        _SHA256_RE.fullmatch(report_file_sha256) is None
        or report_file_bytes <= 0
        or report["bindings"]["plan_file_sha256"] != plan_file_sha256
    ):
        raise ValueError("rights report file binding changed")
    verification: dict[str, Any] = {
        "checks": {
            "independent_report_reconstruction_matches": True,
            "all_seven_bound_inputs_replayed": True,
            "raw_text_absent": True,
            "remaining_blockers_preserved": True,
            "stop_disposition_and_production_false_preserved": True,
            "zero_external_or_training_actions": True,
        },
        "independent_replay": {
            "acquisition_receipt_reads": 1,
            "repository_readme_reads": 1,
            "repository_license_reads": 1,
            "curation_report_reads": 1,
            "curation_verification_reads": 1,
            "owner_receipt_reads": 1,
            "owner_verification_reads": 1,
            "bound_input_file_reads": 7,
            "bound_input_bytes": 54_784,
            "generated_report_reads": 1,
            "source_line_reads": 0,
            "candidate_row_reads": 0,
            "baseline_row_reads": 0,
            "test_object_reads": 0,
        },
        "plan_file_sha256": plan_file_sha256,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "report_file_bytes": report_file_bytes,
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
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
        "report_file_bytes",
        "report_file_sha256",
        "report_internal_sha256",
        "schema_version",
        "status",
        "verification_sha256",
        "zero_use",
    }
    unsigned = {
        key: value for key, value in verification.items() if key != "verification_sha256"
    }
    if (
        set(verification) != expected_keys
        or verification.get("verification_sha256") != sha256_value(unsigned)
        or verification.get("schema_version") != VERIFICATION_SCHEMA_VERSION
        or verification.get("status") != VERIFICATION_STATUS
        or verification.get("checks")
        != {
            "independent_report_reconstruction_matches": True,
            "all_seven_bound_inputs_replayed": True,
            "raw_text_absent": True,
            "remaining_blockers_preserved": True,
            "stop_disposition_and_production_false_preserved": True,
            "zero_external_or_training_actions": True,
        }
        or verification.get("independent_replay")
        != {
            "acquisition_receipt_reads": 1,
            "repository_readme_reads": 1,
            "repository_license_reads": 1,
            "curation_report_reads": 1,
            "curation_verification_reads": 1,
            "owner_receipt_reads": 1,
            "owner_verification_reads": 1,
            "bound_input_file_reads": 7,
            "bound_input_bytes": 54_784,
            "generated_report_reads": 1,
            "source_line_reads": 0,
            "candidate_row_reads": 0,
            "baseline_row_reads": 0,
            "test_object_reads": 0,
        }
        or verification.get("production_eligible") is not False
        or verification.get("raw_training_text_retained") is not False
        or verification.get("zero_use")
        != {"gpu_use": 0, "inference_calls": 0, "network_bytes": 0}
        or not isinstance(verification.get("report_file_bytes"), int)
        or verification["report_file_bytes"] <= 0
        or any(
            not isinstance(verification.get(field), str)
            or _SHA256_RE.fullmatch(verification[field]) is None
            for field in (
                "plan_file_sha256",
                "report_file_sha256",
                "report_internal_sha256",
            )
        )
    ):
        raise ValueError("Be-DSTC2 upstream-rights verification changed")
    _assert_no_forbidden_keys(verification)


def load_plan(path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resolved = resolve_repo_path(root, path)
    if resolved != resolve_repo_path(root, EXPECTED_PLAN_PATH):
        raise ValueError("upstream-rights plan path changed")
    info = resolved.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError("upstream-rights plan metadata changed")
    plan = strict_json_bytes(resolved.read_bytes())
    expected_keys = {
        "authorization",
        "authorization_sha256",
        "authorization_text",
        "implementation",
        "inputs",
        "limits",
        "output",
        "rights_review_contract",
        "runner",
        "schema_version",
        "status",
        "test",
        "verifier",
    }
    contract = {
        "document_observations": EXPECTED_DOCUMENT_OBSERVATIONS,
        "rights_criteria": EXPECTED_RIGHTS_CRITERIA,
        "reason_codes": EXPECTED_REASON_CODES,
        "report_marker_bytes": len(EXPECTED_REPORT_MARKER),
        "report_marker_sha256": sha256_bytes(EXPECTED_REPORT_MARKER),
        "verification_marker_bytes": len(EXPECTED_VERIFICATION_MARKER),
        "verification_marker_sha256": sha256_bytes(EXPECTED_VERIFICATION_MARKER),
        "expected_status": REPORT_STATUS,
        "production_eligible": False,
    }
    if (
        not isinstance(plan, dict)
        or set(plan) != expected_keys
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status") != PLAN_STATUS
        or plan.get("authorization_text") != EXPECTED_AUTHORIZATION_TEXT
        or plan.get("authorization_sha256") != EXPECTED_AUTHORIZATION_SHA256
        or sha256_bytes(plan["authorization_text"].encode("utf-8"))
        != EXPECTED_AUTHORIZATION_SHA256
        or plan.get("authorization") != EXPECTED_AUTHORIZATION
        or plan.get("inputs") != EXPECTED_INPUTS
        or plan.get("limits") != EXPECTED_LIMITS
        or plan.get("output") != EXPECTED_OUTPUT
        or plan.get("rights_review_contract") != contract
    ):
        raise ValueError("upstream-rights plan contract changed")
    for name, expected_path in EXPECTED_CODE_PATHS.items():
        binding = plan.get(name)
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256"}
            or binding.get("path") != expected_path
            or not isinstance(binding.get("sha256"), str)
            or _SHA256_RE.fullmatch(binding["sha256"]) is None
        ):
            raise ValueError("upstream-rights code binding changed")
        code_path = resolve_repo_path(root, expected_path)
        if not code_path.is_file() or file_sha256(code_path) != binding["sha256"]:
            raise ValueError("upstream-rights code hash changed")
    if sum(int(item["bytes"]) for item in EXPECTED_INPUTS.values()) != 54_784:
        raise ValueError("bound input byte total changed")
    for name in EXPECTED_INPUTS:
        _assert_input_metadata(root, name)
    for output_path in EXPECTED_OUTPUT.values():
        if resolve_repo_path(root, output_path).parent != resolve_repo_path(
            root, "evidence"
        ):
            raise ValueError("upstream-rights output path changed")
    return plan


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
            raise ValueError("upstream-rights marker inode changed")
        offset = 0
        while offset < len(marker):
            written = os.write(descriptor, marker[offset:])
            if written <= 0:
                raise OSError("short upstream-rights marker write")
            offset += written
        os.fsync(descriptor)
        identity = (info.st_dev, info.st_ino)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return identity


def finish_marker(
    path: Path, marker: bytes, payload: bytes, identity: tuple[int, int]
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
            raise ValueError("upstream-rights marker inode changed")
        if os.read(descriptor, len(marker) + 1) != marker:
            raise ValueError("upstream-rights marker changed")
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short upstream-rights final write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def preflight_runner(plan: Mapping[str, Any], root: Path) -> tuple[Path, Path]:
    report = resolve_repo_path(root, plan["output"]["report_path"])
    verification = resolve_repo_path(root, plan["output"]["verification_path"])
    if report.exists() or verification.exists():
        raise FileExistsError("upstream-rights review already started")
    return report, verification


def preflight_verifier(plan: Mapping[str, Any], root: Path) -> tuple[Path, Path]:
    report = resolve_repo_path(root, plan["output"]["report_path"])
    verification = resolve_repo_path(root, plan["output"]["verification_path"])
    if not report.is_file() or verification.exists():
        raise FileExistsError("upstream-rights verification preflight changed")
    if report.stat().st_size <= len(EXPECTED_REPORT_MARKER):
        raise ValueError("upstream-rights report is an incomplete marker")
    return report, verification


def prospective_private_write_bytes(
    *,
    plan: Mapping[str, Any],
    root: Path,
    report_bytes: bytes,
    verification_bytes: bytes,
) -> int:
    static = (root / EXPECTED_PLAN_PATH).stat().st_size
    for name in EXPECTED_CODE_PATHS:
        static += resolve_repo_path(root, plan[name]["path"]).stat().st_size
    return (
        static
        + len(EXPECTED_REPORT_MARKER)
        + len(report_bytes)
        + len(EXPECTED_VERIFICATION_MARKER)
        + len(verification_bytes)
    )


__all__ = [
    "EXPECTED_AUTHORIZATION",
    "EXPECTED_AUTHORIZATION_SHA256",
    "EXPECTED_AUTHORIZATION_TEXT",
    "EXPECTED_CHECKS",
    "EXPECTED_CODE_PATHS",
    "EXPECTED_DOCUMENT_OBSERVATIONS",
    "EXPECTED_EVALUATION_PROHIBITION",
    "EXPECTED_INPUTS",
    "EXPECTED_LIMITS",
    "EXPECTED_NEXT_GATE",
    "EXPECTED_OUTPUT",
    "EXPECTED_PLAN_PATH",
    "EXPECTED_REASON_CODES",
    "EXPECTED_REMAINING_BLOCKERS",
    "EXPECTED_REPORT_MARKER",
    "EXPECTED_RIGHTS",
    "EXPECTED_RIGHTS_CRITERIA",
    "EXPECTED_SOURCE_CONTRACT",
    "EXPECTED_VERIFICATION_MARKER",
    "REPORT_SCHEMA_VERSION",
    "REPORT_STATUS",
    "VERIFICATION_SCHEMA_VERSION",
    "VERIFICATION_STATUS",
    "assemble_report",
    "build_report",
    "build_verification",
    "canonical_json_bytes",
    "file_sha256",
    "finish_marker",
    "inspect_rights_documents",
    "json_bytes",
    "load_plan",
    "preflight_runner",
    "preflight_verifier",
    "prospective_private_write_bytes",
    "read_bound_bytes",
    "read_inputs_and_assess",
    "resolve_repo_path",
    "sha256_bytes",
    "sha256_value",
    "start_marker",
    "strict_json_bytes",
    "validate_internal_seal",
    "validate_report",
    "validate_verification",
]
