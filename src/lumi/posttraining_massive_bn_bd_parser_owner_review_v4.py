"""Finalize the local MASSIVE bn-BD parser-canary owner review.

The two-pass successor reads only exact, already-local v0.3 artifacts and the
human decision export.  Public outputs retain hashes and aggregate counts only;
reviewer aliases, notes, record IDs, messages, and training text are excluded.
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

PLAN_SCHEMA_VERSION = "lumi.sft-massive-bn-bd-parser-owner-review-plan.v0.4"
RECEIPT_SCHEMA_VERSION = "lumi.sft-massive-bn-bd-parser-owner-review-receipt.v0.4"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-massive-bn-bd-parser-owner-review-verification.v0.4"
)
DECISIONS_SCHEMA_VERSION = "lumi.ai-assisted-owner-decisions.v0.3"

PLAN_STATUS = "authorized_local_only_massive_bn_bd_owner_review_nonproduction"
RECEIPT_STATUS = (
    "PASS_owner_accepted_64_massive_bn_bd_parser_rows_nonproduction_"
    "cross_source_preflight_pending"
)
VERIFICATION_STATUS = (
    "PASS_verified_owner_accepted_64_massive_bn_bd_parser_rows_nonproduction_"
    "cross_source_preflight_pending"
)

EXPECTED_PLAN_PATH = Path(
    "configs/lumi-4b-sft-massive-bn-bd-parser-owner-review-v0.4.json"
)
EXPECTED_AUTHORIZATION_TEXT = "done"
EXPECTED_AUTHORIZATION_SHA256 = (
    "a4c3ed04a95a3da14a9d235c83d868bed7c0f45cf7f3faa751ee8f50598d2211"
)
EXPECTED_EXTERNAL_DECISIONS_PATH = Path(
    "/Users/afnan/Downloads/lumi-owner-review-decisions-v0.3 (4).json"
)
_PRIVATE_ROOT = Path(
    "../lumi-artifacts/4b-sft-massive-bn-bd-parser-canary-v0.3/"
    "38be2e88eabf/attempt-001/private"
)
EXPECTED_CODE_PATHS = {
    "implementation": "src/lumi/posttraining_massive_bn_bd_parser_owner_review_v4.py",
    "finalizer": "scripts/finalize_sft_massive_bn_bd_parser_owner_review_v4.py",
    "verifier": "scripts/verify_sft_massive_bn_bd_parser_owner_review_v4.py",
    "test": "tests/test_massive_bn_bd_parser_owner_review_v4.py",
}
EXPECTED_OUTPUT = {
    "receipt_path": "evidence/lumi-4b-sft-massive-bn-bd-parser-owner-review-v0.4.json",
    "verification_path": (
        "evidence/lumi-4b-sft-massive-bn-bd-parser-owner-review-"
        "verification-v0.4.json"
    ),
}
EXPECTED_INPUTS = {
    "predecessor_plan": {
        "path": "configs/lumi-4b-sft-massive-bn-bd-parser-canary-v0.3.json",
        "bytes": 12_558,
        "sha256": "4e2ecde03115cffe2d0dbffa78c83b1165b7be291912642a9ed1e287998f521a",
        "internal_sha256": (
            "c4935ac5a934f091539269f212ca41048d72baae8ff9fb78dd85f309784d73dd"
        ),
    },
    "candidate": {
        "path": str(_PRIVATE_ROOT / "candidate-rows.jsonl"),
        "bytes": 140_522,
        "sha256": "8e107adb12839baf9e0aca0d9eebe0b058a2e69d2e9e9b5f4985b69c143e243c",
        "sample_sha256": (
            "b3dfbf766ef6772864fd380abff5b5f2b35b15535e6e07561bac543c230e01a2"
        ),
    },
    "owner_review_html": {
        "path": str(_PRIVATE_ROOT / "review/owner-review-64.html"),
        "bytes": 74_188,
        "sha256": "21a96d6da2b8bb952e29d02cac6610407ec28baf934acf7c8687cc5cc2930705",
    },
    "start_marker": {
        "path": str(_PRIVATE_ROOT / ".parser-canary-started-v0.3"),
        "bytes": 47,
        "sha256": "a9e55cfc4fddbd1ec0ad01c73693578439a70cdb9453abfb068efd3b7c5a5c17",
    },
    "private_manifest": {
        "path": str(_PRIVATE_ROOT / "private-manifest.json"),
        "bytes": 902,
        "sha256": "b6816ce7661d3f4d43375ee1f28211894a6ecd019e24684edab5bd559c748c1e",
        "internal_sha256": (
            "17726026ab133ce1dd6c216bd83a27b794aea1e234dcc81275183a314d0e5a1b"
        ),
    },
    "curation_report": {
        "path": "evidence/lumi-4b-sft-massive-bn-bd-parser-canary-v0.3.json",
        "bytes": 9_416,
        "sha256": "81072f7d69df0b82737e7b4869474ba28785e89d0e68aaf269c2201b25861400",
        "internal_sha256": (
            "5396dc4516254efe58e8c82f63613a8b6cb3e4e8c8c245ba7a8a441e8ea626ee"
        ),
    },
    "curation_verification": {
        "path": (
            "evidence/lumi-4b-sft-massive-bn-bd-parser-canary-"
            "verification-v0.3.json"
        ),
        "bytes": 2_590,
        "sha256": "f9876d58faffe4554f4adf7a388dbb01d4485acfd9ee657f2130bc270fcaa849",
        "internal_sha256": (
            "8e265356f07f7f32e5cdb07923a89e998b5116b9aa8b0da958041ca768d7b249"
        ),
    },
    "decisions": {
        "path": str(EXPECTED_EXTERNAL_DECISIONS_PATH),
        "bytes": 8_053,
        "sha256": "96f597fd5a4a0920d0888197b9c3ed4469929b47b4234239681decd557b36251",
        "canonical_sha256": (
            "5cc8cded6c1466143c3a86e7d69306333fdd6402a20768eaca8f55bdc63676a6"
        ),
    },
}
EXPECTED_LIMITS = {
    "passes": 2,
    "candidate_reads": 2,
    "candidate_row_reads": 128,
    "decision_reads": 2,
    "decision_row_reads": 128,
    "predecessor_plan_reads": 2,
    "owner_review_html_reads": 2,
    "start_marker_reads": 2,
    "private_manifest_reads": 2,
    "curation_report_reads": 2,
    "curation_verification_reads": 2,
    "receipt_reads": 1,
    "source_line_reads": 0,
    "baseline_row_reads": 0,
    "test_object_reads": 0,
    "registry_reads": 0,
    "runtime_seconds_total": 300,
    "runtime_seconds_per_process": 150,
    "private_write_bytes": 1_048_576,
    "network_bytes": 0,
    "gpu_use": 0,
    "inference_calls": 0,
    "training_actions": 0,
}
EXPECTED_AUTHORIZATION = {
    "human_owner_decision_materialization_authorized": True,
    "candidate_or_source_mutation_authorized": False,
    "source_or_baseline_replay_authorized": False,
    "registry_mutation_authorized": False,
    "production_promotion_authorized": False,
    "dataset_seal_authorized": False,
    "sft_training_authorized": False,
    "network_gpu_or_inference_authorized": False,
    "test_object_access_authorized": False,
    "unrelated_action_authorized": False,
}
EXPECTED_REVIEWED_AT = "2026-08-31T05:33:53Z"
EXPECTED_REVIEWED_IDS_DOMAIN_SHA256 = (
    "6a57ca232c624c52b6e39d304992f002b3d115ccc8807312b620cb7b67316dd3"
)
EXPECTED_RECEIPT_MARKER = (
    b'{"started":"massive_bn_bd_parser_owner_review_receipt_v0.4"}\n'
)
EXPECTED_VERIFICATION_MARKER = (
    b'{"started":"massive_bn_bd_parser_owner_review_verification_v0.4"}\n'
)
EXPECTED_CHECKS = {
    "all_64_owner_decisions_accept_and_61_threshold_met": True,
    "candidate_projection_and_decision_sample_hash_reproduced": True,
    "decision_candidate_report_html_manifest_marker_and_plan_bound": True,
    "decision_notes_reviewer_alias_record_ids_and_training_text_not_retained": True,
    "human_owner_review_is_self_attested_not_identity_verified": True,
    "no_source_baseline_test_registry_network_gpu_inference_training_or_promotion": True,
    "predecessor_evidence_preserved_and_nonproduction": True,
    "receipt_is_raw_text_free_and_exactly_reproducible": True,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RECORD_ID_RE = re.compile(r"^massive-bn-bd-parser-[0-9a-f]{24}$")
_REVIEWER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_AI_ALIAS_MARKERS = ("automated", "bot", "chatgpt", "claude", "codex", "gemini", "openai")
_FORBIDDEN_PUBLIC_KEYS = {
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
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid strict JSON") from error
    _assert_no_surrogates(value)
    return value


def _resolve_repo_path(root: Path, path: str) -> Path:
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.parent):
        raise ValueError("bound repository-adjacent path escaped")
    return resolved


def _input_path(root: Path, name: str) -> Path:
    if name == "decisions":
        path = Path(str(EXPECTED_INPUTS[name]["path"])).resolve()
        if path != EXPECTED_EXTERNAL_DECISIONS_PATH:
            raise ValueError("external decision path changed")
        return path
    return _resolve_repo_path(root, str(EXPECTED_INPUTS[name]["path"]))


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


def _read_binding(root: Path, name: str) -> bytes:
    binding = EXPECTED_INPUTS[name]
    return read_bound_bytes(
        _input_path(root, name),
        expected_bytes=int(binding["bytes"]),
        expected_sha256=str(binding["sha256"]),
    )


def _validate_internal_seal(value: Any, binding: Mapping[str, Any], key: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("sealed input root changed")
    unsigned = {name: item for name, item in value.items() if name != key}
    expected = binding["internal_sha256"]
    if value.get(key) != expected or sha256_value(unsigned) != expected:
        raise ValueError("sealed input hash changed")


def _candidate_sample(payload: bytes) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[Mapping[str, Any]] = []
    for line in payload.splitlines():
        if not line:
            continue
        value = strict_json_bytes(line)
        if not isinstance(value, Mapping):
            raise ValueError("candidate row root changed")
        rows.append(value)
    if len(rows) != 64:
        raise ValueError("candidate row count changed")
    expected_keys = {
        "language",
        "messages",
        "provenance",
        "quality",
        "record_id",
        "safety",
        "schema_version",
        "source",
        "split",
        "task_category",
    }
    sample: list[dict[str, Any]] = []
    for row in rows:
        record_id = row.get("record_id")
        messages = row.get("messages")
        if (
            set(row) != expected_keys
            or not isinstance(record_id, str)
            or _RECORD_ID_RE.fullmatch(record_id) is None
            or row.get("split") != "train"
            or not isinstance(messages, list)
            or len(messages) != 4
        ):
            raise ValueError("candidate row contract changed")
        for message, role in zip(
            messages,
            ("user", "tool", "tool_result", "assistant"),
            strict=True,
        ):
            if (
                not isinstance(message, Mapping)
                or set(message) != {"content", "role"}
                or message.get("role") != role
                or not isinstance(message.get("content"), str)
                or not message["content"]
            ):
                raise ValueError("candidate message contract changed")
        sample.append(
            {
                "record_id": record_id,
                "skill_id": "amazon_massive_bn_bd_tool_canary:pure_parser",
                "language_mode": "bn_bd_source_utterance",
                "split": "train",
                "messages": messages,
            }
        )
    sample.sort(key=lambda item: item["record_id"])
    ids = [str(item["record_id"]) for item in sample]
    if len(set(ids)) != 64 or sha256_value(sample) != EXPECTED_INPUTS["candidate"]["sample_sha256"]:
        raise ValueError("candidate owner sample changed")
    return sample, ids


def _validate_timestamp(value: Any) -> None:
    if not isinstance(value, str) or len(value) != 20:
        raise ValueError("owner timestamp contract changed")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError("owner timestamp contract changed") from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ValueError("owner timestamp contract changed")


def _validate_decisions(value: Any, expected_ids: Sequence[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "decisions",
        "reviewed_at",
        "reviewer_id",
        "sample_sha256",
        "schema_version",
    }:
        raise ValueError("owner decision root changed")
    reviewer = value.get("reviewer_id")
    if (
        value.get("schema_version") != DECISIONS_SCHEMA_VERSION
        or value.get("sample_sha256") != EXPECTED_INPUTS["candidate"]["sample_sha256"]
        or not isinstance(reviewer, str)
        or _REVIEWER_RE.fullmatch(reviewer) is None
        or any(marker in reviewer.casefold() for marker in _AI_ALIAS_MARKERS)
    ):
        raise ValueError("owner decision binding changed")
    _validate_timestamp(value.get("reviewed_at"))
    decisions = value.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != 64:
        raise ValueError("owner decision inventory changed")
    observed: list[str] = []
    accepted = rejected = nonempty_notes = 0
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
    if observed != list(expected_ids) or len(set(observed)) != 64 or accepted < 61:
        raise ValueError("owner decision coverage or threshold changed")
    outcome = {
        "accepted": accepted,
        "rejected": rejected,
        "nonempty_notes": nonempty_notes,
        "reviewed_at": value["reviewed_at"],
    }
    if outcome != {
        "accepted": 64,
        "rejected": 0,
        "nonempty_notes": 0,
        "reviewed_at": EXPECTED_REVIEWED_AT,
    }:
        raise ValueError("bound owner-review outcome changed")
    return outcome


def _reviewed_ids_sha256(record_ids: Sequence[str]) -> str:
    return sha256_bytes(
        b"massive-bn-bd-parser-owner-reviewed-record-ids-v0.4\0"
        + canonical_json_bytes(sorted(record_ids))
    )


def _read_and_validate_inputs(root: Path) -> dict[str, Any]:
    predecessor_plan = strict_json_bytes(_read_binding(root, "predecessor_plan"))
    _validate_internal_seal(
        predecessor_plan,
        EXPECTED_INPUTS["predecessor_plan"],
        "plan_sha256",
    )
    if (
        predecessor_plan.get("schema_version")
        != "lumi.sft-massive-bn-bd-parser-canary-plan.v0.3"
        or predecessor_plan.get("status")
        != "authorized_local_only_collision_aware_parser_canary_nonproduction"
    ):
        raise ValueError("predecessor plan changed")

    candidate_payload = _read_binding(root, "candidate")
    _, record_ids = _candidate_sample(candidate_payload)
    _read_binding(root, "owner_review_html")
    _read_binding(root, "start_marker")

    manifest = strict_json_bytes(_read_binding(root, "private_manifest"))
    _validate_internal_seal(
        manifest,
        EXPECTED_INPUTS["private_manifest"],
        "manifest_sha256",
    )
    if (
        manifest.get("schema_version")
        != "lumi.sft-massive-bn-bd-parser-canary-private-manifest.v0.3"
        or manifest.get("candidate_rows") != 64
        or manifest.get("candidate_file_sha256") != EXPECTED_INPUTS["candidate"]["sha256"]
        or manifest.get("owner_review_rows") != 64
        or manifest.get("owner_review_sample_sha256")
        != EXPECTED_INPUTS["candidate"]["sample_sha256"]
        or manifest.get("owner_review_status") != "pending"
        or manifest.get("active_tool_api_credit_rows") != 0
        or manifest.get("production_eligible") is not False
    ):
        raise ValueError("private manifest changed")

    report = strict_json_bytes(_read_binding(root, "curation_report"))
    _validate_internal_seal(report, EXPECTED_INPUTS["curation_report"], "report_sha256")
    if (
        report.get("schema_version")
        != "lumi.sft-massive-bn-bd-parser-canary-report.v0.3"
        or report.get("status")
        != "PASS_private_64_row_parser_canary_pending_owner_review_nonproduction"
        or report.get("owner_review")
        != {
            "completed": False,
            "minimum_acceptance_basis_points": 9_500,
            "rows": 64,
            "sample_sha256": EXPECTED_INPUTS["candidate"]["sample_sha256"],
            "status": "pending",
        }
        or report.get("production_eligible") is not False
    ):
        raise ValueError("curation report changed")

    verification = strict_json_bytes(_read_binding(root, "curation_verification"))
    _validate_internal_seal(
        verification,
        EXPECTED_INPUTS["curation_verification"],
        "verification_sha256",
    )
    if (
        verification.get("schema_version")
        != "lumi.sft-massive-bn-bd-parser-canary-verification.v0.3"
        or verification.get("status")
        != "PASS_zero_source_parser_canary_verification_nonproduction"
        or verification.get("report_file_sha256")
        != EXPECTED_INPUTS["curation_report"]["sha256"]
        or verification.get("report_internal_sha256")
        != EXPECTED_INPUTS["curation_report"]["internal_sha256"]
        or verification.get("production_eligible") is not False
    ):
        raise ValueError("curation verification changed")

    decisions = strict_json_bytes(_read_binding(root, "decisions"))
    if sha256_value(decisions) != EXPECTED_INPUTS["decisions"]["canonical_sha256"]:
        raise ValueError("owner decision canonical hash changed")
    outcome = _validate_decisions(decisions, record_ids)
    reviewed_ids_sha256 = _reviewed_ids_sha256(record_ids)
    if reviewed_ids_sha256 != EXPECTED_REVIEWED_IDS_DOMAIN_SHA256:
        raise ValueError("reviewed record-ID domain changed")
    return {"outcome": outcome, "reviewed_ids_sha256": reviewed_ids_sha256}


def _assert_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _FORBIDDEN_PUBLIC_KEYS:
                raise ValueError("raw or private owner-review key retained")
            _assert_no_forbidden_keys(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_forbidden_keys(item)


def load_plan(path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resolved_plan = Path(path).resolve()
    if resolved_plan != (root / EXPECTED_PLAN_PATH).resolve():
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
    expected_contract = {
        "accepted_records": 64,
        "cryptographic_signature_present": False,
        "decision_schema_version": DECISIONS_SCHEMA_VERSION,
        "minimum_acceptance_basis_points": 9_500,
        "minimum_accepts": 61,
        "nonempty_notes": 0,
        "receipt_marker_bytes": len(EXPECTED_RECEIPT_MARKER),
        "receipt_marker_sha256": sha256_bytes(EXPECTED_RECEIPT_MARKER),
        "rejected_records": 0,
        "reviewed_at_claimed_utc": EXPECTED_REVIEWED_AT,
        "reviewed_record_ids_domain_sha256": EXPECTED_REVIEWED_IDS_DOMAIN_SHA256,
        "reviewer_identity_verified": False,
        "sample_records": 64,
        "verification_marker_bytes": len(EXPECTED_VERIFICATION_MARKER),
        "verification_marker_sha256": sha256_bytes(EXPECTED_VERIFICATION_MARKER),
    }
    if (
        not isinstance(plan, dict)
        or set(plan) != expected_keys
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status") != PLAN_STATUS
        or plan.get("authorization_text") != EXPECTED_AUTHORIZATION_TEXT
        or plan.get("authorization_sha256") != EXPECTED_AUTHORIZATION_SHA256
        or sha256_bytes(EXPECTED_AUTHORIZATION_TEXT.encode())
        != EXPECTED_AUTHORIZATION_SHA256
        or plan.get("authorization") != EXPECTED_AUTHORIZATION
        or plan.get("inputs") != EXPECTED_INPUTS
        or plan.get("limits") != EXPECTED_LIMITS
        or plan.get("output") != EXPECTED_OUTPUT
        or plan.get("owner_review_contract") != expected_contract
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
        code_path = (root / expected_path).resolve()
        if not code_path.is_relative_to(root) or file_sha256(code_path) != binding["sha256"]:
            raise ValueError("owner-review code hash changed")
    for name, binding in EXPECTED_INPUTS.items():
        info = _input_path(root, name).lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_size != binding["bytes"]:
            raise ValueError("bound input metadata changed")
    for output in EXPECTED_OUTPUT.values():
        resolved = (root / output).resolve()
        if not resolved.is_relative_to(root / "evidence"):
            raise ValueError("owner-review output path changed")
    return plan


def build_receipt(
    plan: Mapping[str, Any], *, plan_file_sha256: str, repository_root: str | Path
) -> dict[str, Any]:
    del plan
    if _SHA256_RE.fullmatch(plan_file_sha256) is None:
        raise ValueError("plan file hash changed")
    outcome = _read_and_validate_inputs(Path(repository_root).resolve())
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": RECEIPT_STATUS,
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            "authorization_sha256": EXPECTED_AUTHORIZATION_SHA256,
            "candidate_bytes": EXPECTED_INPUTS["candidate"]["bytes"],
            "candidate_file_sha256": EXPECTED_INPUTS["candidate"]["sha256"],
            "curation_report_file_sha256": EXPECTED_INPUTS["curation_report"]["sha256"],
            "curation_report_internal_sha256": EXPECTED_INPUTS["curation_report"]["internal_sha256"],
            "curation_verification_file_sha256": EXPECTED_INPUTS["curation_verification"]["sha256"],
            "curation_verification_internal_sha256": EXPECTED_INPUTS["curation_verification"]["internal_sha256"],
            "decisions_bytes": EXPECTED_INPUTS["decisions"]["bytes"],
            "decisions_canonical_sha256": EXPECTED_INPUTS["decisions"]["canonical_sha256"],
            "decisions_file_sha256": EXPECTED_INPUTS["decisions"]["sha256"],
            "owner_review_html_file_sha256": EXPECTED_INPUTS["owner_review_html"]["sha256"],
            "owner_review_sample_sha256": EXPECTED_INPUTS["candidate"]["sample_sha256"],
            "plan_file_sha256": plan_file_sha256,
            "predecessor_manifest_file_sha256": EXPECTED_INPUTS["private_manifest"]["sha256"],
            "predecessor_manifest_internal_sha256": EXPECTED_INPUTS["private_manifest"]["internal_sha256"],
            "predecessor_plan_file_sha256": EXPECTED_INPUTS["predecessor_plan"]["sha256"],
            "predecessor_plan_internal_sha256": EXPECTED_INPUTS["predecessor_plan"]["internal_sha256"],
            "predecessor_start_marker_sha256": EXPECTED_INPUTS["start_marker"]["sha256"],
            "reviewed_record_ids_domain_sha256": outcome["reviewed_ids_sha256"],
        },
        "owner_review": {
            "accepted_records": outcome["outcome"]["accepted"],
            "all_sampled_rows_accepted": True,
            "automated_reviewer_markers_absent": True,
            "cryptographic_signature_present": False,
            "rejected_records": outcome["outcome"]["rejected"],
            "reviewed_at_claimed_utc": outcome["outcome"]["reviewed_at"],
            "reviewer_alias_present": True,
            "reviewer_identity_verified": False,
            "sample_records": 64,
            "threshold": {
                "minimum_acceptance_basis_points": 9_500,
                "minimum_accepts": 61,
                "observed_acceptance_basis_points": 10_000,
                "observed_accepts": 64,
                "satisfied": True,
            },
        },
        "candidate": {
            "records": 64,
            "train_records": 64,
            "intent_coverage_rows": 60,
            "deterministic_extra_rows": 4,
            "four_role_pure_parser_rows": 64,
            "external_action_performed_rows": 0,
        },
        "rights": {
            "license": "cc-by-4.0",
            "source_revision": "d1792d0e4f32a74d43725cc1b901f600dad990dd",
        },
        "owner_review_claim": (
            "source_derived_machine_curated_massive_bn_bd_parser_rows_"
            "human_owner_reviewed_not_human_written"
        ),
        "human_owner_review_claim_self_attested": True,
        "reviewer_alias_retained": False,
        "decision_notes_retained": False,
        "record_ids_retained": False,
        "raw_training_text_retained": False,
        "active_tool_api_credit_rows": 0,
        "remaining_blockers": [
            "cross_source_dedup_and_registry_preflight_not_authorized",
            "source_registry_promotion_not_authorized",
            "production_dataset_seal_and_sft_authorization_not_complete",
        ],
        "next_gate": {
            "action": "separate_cross_source_dedup_and_registry_preflight",
            "owner_review_complete": True,
            "separate_authorization_required": True,
            "production_promotion_authorized": False,
        },
        "production_eligible": False,
        "receipt_sha256": "",
    }
    receipt["receipt_sha256"] = sha256_value(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    validate_receipt(receipt)
    return receipt


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    expected_keys = {
        "active_tool_api_credit_rows",
        "authorization",
        "bindings",
        "candidate",
        "decision_notes_retained",
        "human_owner_review_claim_self_attested",
        "next_gate",
        "owner_review",
        "owner_review_claim",
        "production_eligible",
        "raw_training_text_retained",
        "receipt_sha256",
        "record_ids_retained",
        "remaining_blockers",
        "reviewer_alias_retained",
        "rights",
        "schema_version",
        "status",
    }
    bindings = receipt.get("bindings")
    expected_binding_values = {
        "authorization_sha256": EXPECTED_AUTHORIZATION_SHA256,
        "candidate_bytes": EXPECTED_INPUTS["candidate"]["bytes"],
        "candidate_file_sha256": EXPECTED_INPUTS["candidate"]["sha256"],
        "curation_report_file_sha256": EXPECTED_INPUTS["curation_report"]["sha256"],
        "curation_report_internal_sha256": EXPECTED_INPUTS["curation_report"]["internal_sha256"],
        "curation_verification_file_sha256": EXPECTED_INPUTS["curation_verification"]["sha256"],
        "curation_verification_internal_sha256": EXPECTED_INPUTS["curation_verification"]["internal_sha256"],
        "decisions_bytes": EXPECTED_INPUTS["decisions"]["bytes"],
        "decisions_canonical_sha256": EXPECTED_INPUTS["decisions"]["canonical_sha256"],
        "decisions_file_sha256": EXPECTED_INPUTS["decisions"]["sha256"],
        "owner_review_html_file_sha256": EXPECTED_INPUTS["owner_review_html"]["sha256"],
        "owner_review_sample_sha256": EXPECTED_INPUTS["candidate"]["sample_sha256"],
        "predecessor_manifest_file_sha256": EXPECTED_INPUTS["private_manifest"]["sha256"],
        "predecessor_manifest_internal_sha256": EXPECTED_INPUTS["private_manifest"]["internal_sha256"],
        "predecessor_plan_file_sha256": EXPECTED_INPUTS["predecessor_plan"]["sha256"],
        "predecessor_plan_internal_sha256": EXPECTED_INPUTS["predecessor_plan"]["internal_sha256"],
        "predecessor_start_marker_sha256": EXPECTED_INPUTS["start_marker"]["sha256"],
        "reviewed_record_ids_domain_sha256": EXPECTED_REVIEWED_IDS_DOMAIN_SHA256,
    }
    expected_owner_review = {
        "accepted_records": 64,
        "all_sampled_rows_accepted": True,
        "automated_reviewer_markers_absent": True,
        "cryptographic_signature_present": False,
        "rejected_records": 0,
        "reviewed_at_claimed_utc": EXPECTED_REVIEWED_AT,
        "reviewer_alias_present": True,
        "reviewer_identity_verified": False,
        "sample_records": 64,
        "threshold": {
            "minimum_acceptance_basis_points": 9_500,
            "minimum_accepts": 61,
            "observed_acceptance_basis_points": 10_000,
            "observed_accepts": 64,
            "satisfied": True,
        },
    }
    if (
        set(receipt) != expected_keys
        or receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != RECEIPT_STATUS
        or receipt.get("receipt_sha256") != sha256_value(unsigned)
        or receipt.get("authorization") != EXPECTED_AUTHORIZATION
        or receipt.get("production_eligible") is not False
        or receipt.get("active_tool_api_credit_rows") != 0
        or receipt.get("reviewer_alias_retained") is not False
        or receipt.get("decision_notes_retained") is not False
        or receipt.get("record_ids_retained") is not False
        or receipt.get("raw_training_text_retained") is not False
        or receipt.get("human_owner_review_claim_self_attested") is not True
        or receipt.get("owner_review") != expected_owner_review
        or receipt.get("candidate")
        != {
            "records": 64,
            "train_records": 64,
            "intent_coverage_rows": 60,
            "deterministic_extra_rows": 4,
            "four_role_pure_parser_rows": 64,
            "external_action_performed_rows": 0,
        }
        or receipt.get("rights")
        != {
            "license": "cc-by-4.0",
            "source_revision": "d1792d0e4f32a74d43725cc1b901f600dad990dd",
        }
        or receipt.get("owner_review_claim")
        != (
            "source_derived_machine_curated_massive_bn_bd_parser_rows_"
            "human_owner_reviewed_not_human_written"
        )
        or receipt.get("remaining_blockers")
        != [
            "cross_source_dedup_and_registry_preflight_not_authorized",
            "source_registry_promotion_not_authorized",
            "production_dataset_seal_and_sft_authorization_not_complete",
        ]
        or receipt.get("next_gate")
        != {
            "action": "separate_cross_source_dedup_and_registry_preflight",
            "owner_review_complete": True,
            "separate_authorization_required": True,
            "production_promotion_authorized": False,
        }
        or not isinstance(bindings, Mapping)
        or set(bindings) != set(expected_binding_values) | {"plan_file_sha256"}
        or any(bindings.get(key) != value for key, value in expected_binding_values.items())
        or not isinstance(bindings.get("plan_file_sha256"), str)
        or _SHA256_RE.fullmatch(bindings["plan_file_sha256"]) is None
    ):
        raise ValueError("MASSIVE owner-review receipt changed")
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
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": VERIFICATION_STATUS,
        "checks": EXPECTED_CHECKS,
        "independent_replay": {
            "candidate_reads": 1,
            "candidate_row_reads": 64,
            "decision_reads": 1,
            "decision_row_reads": 64,
            "predecessor_plan_reads": 1,
            "owner_review_html_reads": 1,
            "start_marker_reads": 1,
            "private_manifest_reads": 1,
            "curation_report_reads": 1,
            "curation_verification_reads": 1,
            "receipt_reads": 1,
            "source_line_reads": 0,
            "baseline_row_reads": 0,
            "test_object_reads": 0,
            "registry_reads": 0,
        },
        "plan_file_sha256": plan_file_sha256,
        "receipt_file_bytes": receipt_file_bytes,
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_internal_sha256": receipt["receipt_sha256"],
        "raw_training_text_retained": False,
        "production_eligible": False,
        "zero_use": {
            "network_bytes": 0,
            "gpu_use": 0,
            "inference_calls": 0,
            "training_actions": 0,
        },
        "verification_sha256": "",
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
    unsigned = {
        key: value
        for key, value in verification.items()
        if key != "verification_sha256"
    }
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
    expected_replay = {
        "candidate_reads": 1,
        "candidate_row_reads": 64,
        "decision_reads": 1,
        "decision_row_reads": 64,
        "predecessor_plan_reads": 1,
        "owner_review_html_reads": 1,
        "start_marker_reads": 1,
        "private_manifest_reads": 1,
        "curation_report_reads": 1,
        "curation_verification_reads": 1,
        "receipt_reads": 1,
        "source_line_reads": 0,
        "baseline_row_reads": 0,
        "test_object_reads": 0,
        "registry_reads": 0,
    }
    if (
        set(verification) != expected_keys
        or verification.get("schema_version") != VERIFICATION_SCHEMA_VERSION
        or verification.get("status") != VERIFICATION_STATUS
        or verification.get("verification_sha256") != sha256_value(unsigned)
        or verification.get("checks") != EXPECTED_CHECKS
        or verification.get("production_eligible") is not False
        or verification.get("raw_training_text_retained") is not False
        or verification.get("zero_use")
        != {
            "network_bytes": 0,
            "gpu_use": 0,
            "inference_calls": 0,
            "training_actions": 0,
        }
        or verification.get("independent_replay") != expected_replay
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
        raise ValueError("MASSIVE owner-review verification changed")
    _assert_no_forbidden_keys(verification)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def start_marker(path: Path, marker: bytes) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("owner-review marker inode changed")
        if os.write(descriptor, marker) != len(marker):
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
            or os.read(descriptor, len(marker) + 1) != marker
        ):
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
    receipt = (root / str(plan["output"]["receipt_path"])).resolve()
    verification = (root / str(plan["output"]["verification_path"])).resolve()
    if receipt.exists() or verification.exists():
        raise FileExistsError("owner-review finalization already started")
    return receipt, verification


def preflight_verifier(plan: Mapping[str, Any], root: Path) -> tuple[Path, Path]:
    receipt = (root / str(plan["output"]["receipt_path"])).resolve()
    verification = (root / str(plan["output"]["verification_path"])).resolve()
    if not receipt.is_file() or verification.exists():
        raise FileExistsError("owner-review verification preflight changed")
    if receipt.stat().st_size <= len(EXPECTED_RECEIPT_MARKER):
        raise ValueError("owner-review receipt is an incomplete marker")
    return receipt, verification


def prospective_private_write_bytes(
    *,
    plan: Mapping[str, Any],
    root: Path,
    receipt_bytes: bytes,
    verification_bytes: bytes,
) -> int:
    static = (root / EXPECTED_PLAN_PATH).stat().st_size
    for name in EXPECTED_CODE_PATHS:
        static += (root / str(plan[name]["path"])).stat().st_size
    return (
        static
        + len(EXPECTED_RECEIPT_MARKER)
        + len(receipt_bytes)
        + len(EXPECTED_VERIFICATION_MARKER)
        + len(verification_bytes)
    )


__all__ = [
    "EXPECTED_LIMITS",
    "EXPECTED_RECEIPT_MARKER",
    "EXPECTED_VERIFICATION_MARKER",
    "build_receipt",
    "build_verification",
    "file_sha256",
    "finish_marker",
    "json_bytes",
    "load_plan",
    "preflight_finalizer",
    "preflight_verifier",
    "prospective_private_write_bytes",
    "sha256_bytes",
    "start_marker",
    "strict_json_bytes",
    "validate_receipt",
    "validate_verification",
]
