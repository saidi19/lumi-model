"""Read-only BigCodeBench re-audit of the reviewed OpenCodeInstruct candidate."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .data_quality import BenchmarkRegistry, sha256_file
from .posttraining_bigcodebench_signatures import sha256_value

PLAN_SCHEMA_VERSION = "lumi.sft-opencodeinstruct-bigcodebench-reaudit-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-opencodeinstruct-bigcodebench-reaudit.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-opencodeinstruct-bigcodebench-reaudit-verification.v0.1"
)

OWNER_REQUEST_TEXT = "ok next step shuru kor ar cc mane ki ogula copyright issue?"
EXPECTED_CANDIDATE_RECORDS = 11_100
EXPECTED_CANDIDATE_SHA256 = (
    "9be50e3b6078fdc50d3b887e7850ffc4e57f9a634f72fb928b62908da854446e"
)
EXPECTED_REGISTRY_SHA256 = (
    "4a9e17b75c0447b19b01c845d5f22cb117cb6c68bcb70b94a7393620ac5a4873"
)
EXPECTED_QUALITY_SHA256 = (
    "b2dfabd7e7ba6d6b2bd6600a3f65e43a72a551b02857b6eae42ddbd90187f816"
)
EXPECTED_AUTHORIZATION = {
    "candidate_modification_authorized": False,
    "candidate_registry_promotion_authorized": False,
    "cpu_only_local_reaudit_authorized": True,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "gpu_authorized": False,
    "livecodebench_access_authorized": False,
    "network_access_authorized": False,
    "sft_training_authorized": False,
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_candidate_records": EXPECTED_CANDIDATE_RECORDS,
    "maximum_report_bytes": 1_048_576,
    "maximum_runtime_minutes": 30,
}
EXPECTED_BINDINGS = {
    "bigcodebench_promotion_report",
    "bigcodebench_promotion_verification",
    "candidate_records",
    "data_quality_v0_2",
    "implementation",
    "owner_review_receipt",
    "registry_v0_2",
    "runner",
    "source_curation_report",
    "verifier",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


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
        raise ValueError(f"invalid OpenCodeInstruct re-audit binding: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.parent.resolve()) or not resolved.is_file():
        raise ValueError(f"OpenCodeInstruct re-audit binding missing: {name}")
    if sha256_file(resolved) != expected:
        raise ValueError(f"OpenCodeInstruct re-audit binding changed: {name}")
    return resolved


def load_reaudit_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _read_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported OpenCodeInstruct BigCodeBench re-audit plan")
    if plan.get("status") != "requested_cpu_only_read_only_preflight":
        raise ValueError("OpenCodeInstruct BigCodeBench re-audit status changed")
    if plan.get("observed_at") != "2026-08-07":
        raise ValueError("OpenCodeInstruct BigCodeBench re-audit date changed")
    if plan.get("owner_request_text") != OWNER_REQUEST_TEXT:
        raise ValueError("OpenCodeInstruct BigCodeBench owner request changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("OpenCodeInstruct BigCodeBench authorization changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("OpenCodeInstruct BigCodeBench limits changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("OpenCodeInstruct BigCodeBench bindings changed")
    for name, specification in bindings.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"invalid OpenCodeInstruct re-audit binding: {name}")
        resolve_binding(root, specification, name=name)
    if bindings["candidate_records"]["sha256"] != EXPECTED_CANDIDATE_SHA256:
        raise ValueError("OpenCodeInstruct candidate identity changed")
    if bindings["registry_v0_2"]["sha256"] != EXPECTED_REGISTRY_SHA256:
        raise ValueError("BigCodeBench registry identity changed")
    if bindings["data_quality_v0_2"]["sha256"] != EXPECTED_QUALITY_SHA256:
        raise ValueError("BigCodeBench data-quality identity changed")
    if plan.get("output") != {
        "report_path": (
            "evidence/lumi-4b-sft-opencodeinstruct-bigcodebench-reaudit-v0.1.json"
        ),
        "verification_path": (
            "evidence/"
            "lumi-4b-sft-opencodeinstruct-bigcodebench-reaudit-verification-v0.1.json"
        ),
    }:
        raise ValueError("OpenCodeInstruct BigCodeBench outputs changed")
    return plan


def _validate_owner_review(receipt: Mapping[str, Any]) -> dict[str, Any]:
    if receipt.get("schema_version") != "lumi.sft-opencodeinstruct-owner-review-receipt.v0.1":
        raise ValueError("OpenCodeInstruct owner-review schema changed")
    if receipt.get("all_sampled_rows_accepted") is not True:
        raise ValueError("OpenCodeInstruct owner review did not pass")
    if receipt.get("accepted_records") != 32 or receipt.get("rejected_records") != 0:
        raise ValueError("OpenCodeInstruct owner-review decisions changed")
    candidate = receipt.get("candidate")
    if not isinstance(candidate, Mapping) or candidate.get("records") != (
        EXPECTED_CANDIDATE_RECORDS
    ):
        raise ValueError("OpenCodeInstruct owner-review candidate count changed")
    bindings = receipt.get("bindings")
    if not isinstance(bindings, Mapping) or bindings.get(
        "accepted_records_file_sha256"
    ) != EXPECTED_CANDIDATE_SHA256:
        raise ValueError("OpenCodeInstruct owner review binds another candidate")
    return {
        "accepted_records": 32,
        "all_sampled_rows_accepted": True,
        "receipt_internal_sha256": receipt.get("receipt_sha256"),
        "reviewer_id": receipt.get("reviewer_id"),
        "sample_records": receipt.get("sample_records"),
    }


def scan_candidate(
    plan: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    bindings = plan["bindings"]
    candidate_path = resolve_binding(root, bindings["candidate_records"], name="candidate_records")
    quality_path = resolve_binding(root, bindings["data_quality_v0_2"], name="data_quality_v0_2")
    registry_path = resolve_binding(root, bindings["registry_v0_2"], name="registry_v0_2")
    owner_review_path = resolve_binding(
        root,
        bindings["owner_review_receipt"],
        name="owner_review_receipt",
    )
    quality = _read_json(quality_path)
    benchmark = BenchmarkRegistry(quality["benchmark_decontamination"], root)
    if benchmark.path.resolve() != registry_path:
        raise ValueError("OpenCodeInstruct re-audit registry resolved elsewhere")
    if "bigcodebench" not in benchmark.benchmarks:
        raise ValueError("BigCodeBench is absent from the promoted registry")
    if "livecodebench" in benchmark.benchmarks:
        raise ValueError("LiveCodeBench unexpectedly entered this bounded re-audit")

    matched_counts: Counter[str] = Counter()
    matched_record_ids: list[str] = []
    record_ids: list[str] = []
    split_counts: Counter[str] = Counter()
    with candidate_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid OpenCodeInstruct candidate JSON at line {line_number}"
                ) from error
            if not isinstance(record, Mapping):
                raise ValueError("OpenCodeInstruct candidate row must be an object")
            record_id = record.get("record_id")
            messages = record.get("messages")
            split = record.get("split")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError("OpenCodeInstruct candidate record ID changed")
            if not isinstance(messages, list) or not messages:
                raise ValueError("OpenCodeInstruct candidate messages changed")
            contents: list[str] = []
            for message in messages:
                if not isinstance(message, Mapping) or not isinstance(
                    message.get("content"), str
                ):
                    raise ValueError("OpenCodeInstruct candidate message changed")
                contents.append(message["content"])
            if split not in {"train", "validation"}:
                raise ValueError("OpenCodeInstruct candidate split changed")
            record_ids.append(record_id)
            split_counts[str(split)] += 1
            match = benchmark.match("\n".join(contents))
            if match is not None:
                matched_counts[match[0]] += 1
                matched_record_ids.append(record_id)

    if len(record_ids) != EXPECTED_CANDIDATE_RECORDS:
        raise ValueError("OpenCodeInstruct candidate row count changed")
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("OpenCodeInstruct candidate record IDs are duplicated")
    owner_review = _validate_owner_review(_read_json(owner_review_path))
    return {
        "candidate": {
            "file_sha256": sha256_file(candidate_path),
            "record_ids_sha256": hashlib.sha256(
                "\n".join(sorted(record_ids)).encode("utf-8")
            ).hexdigest(),
            "records": len(record_ids),
            "split_counts": dict(sorted(split_counts.items())),
        },
        "owner_review": owner_review,
        "registry": {
            "benchmarks": sorted(benchmark.benchmarks),
            "bigcodebench_present": True,
            "entries": len(benchmark.required_matches),
            "file_sha256": benchmark.sha256,
            "livecodebench_present": False,
            "unique_exact_hashes": len(benchmark.exact),
        },
        "results": {
            "match_counts": dict(sorted(matched_counts.items())),
            "matched_record_ids_sha256": hashlib.sha256(
                "\n".join(sorted(matched_record_ids)).encode("utf-8")
            ).hexdigest(),
            "matched_records": len(matched_record_ids),
            "scan_unit": "full_conversation_message_content_v0.1",
        },
    }


def build_report(
    plan: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    scan = scan_candidate(plan, repository_root=repository_root)
    report = {
        "authorization": plan["authorization"],
        "boundaries": {
            "candidate_modified": False,
            "candidate_registry_promoted": False,
            "dataset_sealed": False,
            "gpu_or_sft_used": False,
            "livecodebench_accessed": False,
            "network_or_external_api_used": False,
        },
        "candidate": scan["candidate"],
        "observed_at": "2026-08-07",
        "owner_review": scan["owner_review"],
        "production_eligible": False,
        "registry": scan["registry"],
        "remaining_blockers": [
            "livecodebench_rights_or_evaluation_only_replacement_unresolved",
            "opencodeinstruct_candidate_registry_promotion_not_authorized",
            "dataset_seal_and_sft_not_authorized",
        ],
        "results": scan["results"],
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": (
            "bigcodebench_reaudit_passed_no_matches_livecodebench_still_unresolved"
            if scan["results"]["matched_records"] == 0
            else "bigcodebench_reaudit_found_matches_candidate_repair_required"
        ),
    }
    report["report_sha256"] = sha256_value(report)
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported OpenCodeInstruct BigCodeBench re-audit report")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if report.get("report_sha256") != sha256_value(unsigned):
        raise ValueError("OpenCodeInstruct BigCodeBench report seal is invalid")
    if report.get("production_eligible") is not False:
        raise ValueError("OpenCodeInstruct BigCodeBench re-audit crossed production boundary")
    boundaries = report.get("boundaries")
    if not isinstance(boundaries, Mapping) or any(boundaries.values()):
        raise ValueError("OpenCodeInstruct BigCodeBench re-audit crossed a boundary")
    candidate = report.get("candidate")
    if not isinstance(candidate, Mapping) or candidate.get("records") != (
        EXPECTED_CANDIDATE_RECORDS
    ):
        raise ValueError("OpenCodeInstruct BigCodeBench report candidate changed")
    if candidate.get("file_sha256") != EXPECTED_CANDIDATE_SHA256:
        raise ValueError("OpenCodeInstruct BigCodeBench report candidate hash changed")
    registry = report.get("registry")
    if not isinstance(registry, Mapping) or registry.get("file_sha256") != (
        EXPECTED_REGISTRY_SHA256
    ):
        raise ValueError("OpenCodeInstruct BigCodeBench report registry changed")


__all__ = [
    "PLAN_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "VERIFICATION_SCHEMA_VERSION",
    "build_report",
    "load_reaudit_plan",
    "scan_candidate",
    "validate_report",
]
