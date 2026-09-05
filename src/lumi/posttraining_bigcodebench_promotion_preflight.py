"""Raw-text-free promotion preflight for the BigCodeBench signature candidate."""

from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .data_quality import sha256_file
from .posttraining_bigcodebench_signatures import (
    PREDECESSOR_ENTRIES,
    PROHIBITED_FIELDS,
    PROMPT_FIELDS,
    SOURCE_REVISION,
    load_registry,
    records_sha256,
    sha256_value,
)
from .posttraining_bigcodebench_signatures import (
    validate_report as validate_acquisition_report,
)

PLAN_SCHEMA_VERSION = "lumi.bigcodebench-signature-promotion-preflight-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.bigcodebench-signature-promotion-preflight.v0.1"
PROMOTION_PLAN_SCHEMA_VERSION = "lumi.bigcodebench-signature-promotion-plan.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.bigcodebench-signature-promotion-preflight-verification.v0.1"
)

OWNER_AUTHORIZATION_STATEMENT = (
    "I approve one bounded CPU-only BigCodeBench benchmark-signature candidate "
    "promotion preflight, bound to candidate SHA-256 "
    "3e00fe6f8960bc3dd328b17814ad87b7e69750447cc7fdbd4c14b7f4adfb2b26, "
    "acquisition report SHA-256 "
    "89b0071af48698931e1f5d94550c9313e4d1c628320e039ac9d4e1b744ae820c, "
    "verification SHA-256 "
    "2f986073a8b037a5a020081b849193e74d02c4035c4ce54338809f6faac0e423, "
    "and predecessor registry SHA-256 "
    "be02bdbd7f2c0e807c86cf9678eff86d711aaa34ffdc8e6ebb687a26220ffcbb, "
    "maximum 30 minutes. It may audit the exact 50,419-entry candidate and produce "
    "a raw-text-free immutable-v0.2 promotion plan and verification receipt. No "
    "network/API/GPU, active-registry or config mutation, promotion, dataset seal, "
    "SFT training, LiveCodeBench access, unrelated data, or external distribution."
)

CANDIDATE_SHA256 = "3e00fe6f8960bc3dd328b17814ad87b7e69750447cc7fdbd4c14b7f4adfb2b26"
ACQUISITION_REPORT_INTERNAL_SHA256 = (
    "89b0071af48698931e1f5d94550c9313e4d1c628320e039ac9d4e1b744ae820c"
)
ACQUISITION_VERIFICATION_INTERNAL_SHA256 = (
    "2f986073a8b037a5a020081b849193e74d02c4035c4ce54338809f6faac0e423"
)
PREDECESSOR_SHA256 = "be02bdbd7f2c0e807c86cf9678eff86d711aaa34ffdc8e6ebb687a26220ffcbb"
CANDIDATE_ENTRIES = 50_419
APPENDED_ENTRIES = 3_420
EXPECTED_UNIQUE_EXACT_HASHES = 50_249
EXPECTED_BIGCODEBENCH_UNIQUE_EXACT_HASHES = 3_387

EXPECTED_AUTHORIZATION = {
    "active_registry_mutation_authorized": False,
    "candidate_promotion_authorized": False,
    "config_mutation_authorized": False,
    "cpu_only_preflight_authorized": True,
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
    "maximum_candidate_entries": CANDIDATE_ENTRIES,
    "maximum_runtime_minutes": 30,
}
EXPECTED_BINDINGS = {
    "acquisition_report",
    "acquisition_verification",
    "candidate_registry",
    "current_data_quality_config",
    "implementation",
    "predecessor_registry",
    "runner",
    "verifier",
}
ENTRY_FIELDS = {
    "anchor_sha256",
    "benchmark",
    "example_id",
    "exact_sha256",
    "minimum_anchor_matches",
    "split",
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
        raise ValueError(f"invalid BigCodeBench promotion-preflight binding: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.parent.resolve()) or not resolved.is_file():
        raise ValueError(f"BigCodeBench promotion-preflight binding missing: {name}")
    if sha256_file(resolved) != expected:
        raise ValueError(f"BigCodeBench promotion-preflight binding changed: {name}")
    return resolved


def load_preflight_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _read_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported BigCodeBench promotion-preflight plan")
    if plan.get("status") != "authorized_cpu_only_promotion_preflight_no_promotion":
        raise ValueError("BigCodeBench promotion-preflight status changed")
    if plan.get("observed_at") != "2026-08-07":
        raise ValueError("BigCodeBench promotion-preflight date changed")
    if plan.get("owner_authorization_statement") != OWNER_AUTHORIZATION_STATEMENT:
        raise ValueError("BigCodeBench promotion-preflight authorization text changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("BigCodeBench promotion-preflight authorization changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("BigCodeBench promotion-preflight limits changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("BigCodeBench promotion-preflight bindings changed")
    for name, specification in bindings.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"invalid BigCodeBench promotion-preflight binding: {name}")
        resolve_binding(root, specification, name=name)
    if bindings["candidate_registry"]["sha256"] != CANDIDATE_SHA256:
        raise ValueError("BigCodeBench candidate binding changed")
    if bindings["predecessor_registry"]["sha256"] != PREDECESSOR_SHA256:
        raise ValueError("BigCodeBench predecessor binding changed")
    report = _read_json(
        resolve_binding(root, bindings["acquisition_report"], name="acquisition_report")
    )
    validate_acquisition_report(report)
    if report.get("report_sha256") != ACQUISITION_REPORT_INTERNAL_SHA256:
        raise ValueError("BigCodeBench acquisition report seal changed")
    verification = _read_json(
        resolve_binding(
            root,
            bindings["acquisition_verification"],
            name="acquisition_verification",
        )
    )
    if verification.get("verification_sha256") != (
        ACQUISITION_VERIFICATION_INTERNAL_SHA256
    ):
        raise ValueError("BigCodeBench acquisition verification seal changed")
    output = plan.get("output")
    if not isinstance(output, Mapping) or set(output) != {
        "promotion_plan_path",
        "raw_benchmark_text_in_git",
        "report_path",
        "verification_path",
    }:
        raise ValueError("BigCodeBench promotion-preflight output changed")
    if output.get("raw_benchmark_text_in_git") is not False:
        raise ValueError("BigCodeBench promotion-preflight cannot retain raw text")
    return plan


def _validate_entry(entry: Mapping[str, Any], *, line_number: int) -> None:
    if set(entry) != ENTRY_FIELDS:
        raise ValueError(f"benchmark signature schema changed at entry {line_number}")
    if {"text", "prompt", *PROHIBITED_FIELDS}.intersection(entry):
        raise ValueError(f"benchmark signature retained raw text at entry {line_number}")
    if not isinstance(entry.get("benchmark"), str) or not isinstance(
        entry.get("example_id"), str
    ):
        raise ValueError(f"benchmark signature identity changed at entry {line_number}")
    _require_sha256(entry.get("exact_sha256"), f"entry[{line_number}].exact_sha256")
    anchors = entry.get("anchor_sha256")
    if not isinstance(anchors, list) or len(anchors) > 2:
        raise ValueError(f"benchmark signature anchors changed at entry {line_number}")
    for index, anchor in enumerate(anchors):
        _require_sha256(anchor, f"entry[{line_number}].anchor_sha256[{index}]")
    if entry.get("minimum_anchor_matches") != min(2, len(anchors)):
        raise ValueError(f"benchmark signature match floor changed at entry {line_number}")
    if not isinstance(entry.get("split"), str) or not entry["split"]:
        raise ValueError(f"benchmark signature split changed at entry {line_number}")


def deterministic_registry_bytes(
    metadata: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as gzip_handle:
        with io.TextIOWrapper(gzip_handle, encoding="utf-8") as text_handle:
            text_handle.write(
                json.dumps(
                    {"_meta": dict(metadata)},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
            for entry in entries:
                text_handle.write(
                    json.dumps(
                        entry,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
    return buffer.getvalue()


def _proposed_release_metadata(candidate_metadata: Mapping[str, Any]) -> dict[str, Any]:
    metadata = copy.deepcopy(dict(candidate_metadata))
    if metadata.pop("candidate_not_active", None) is not True:
        raise ValueError("BigCodeBench candidate activation marker changed")
    metadata["release"] = {
        "immutable_path": "data/benchmark-signatures-v0.2.jsonl.gz",
        "source_candidate_sha256": CANDIDATE_SHA256,
        "status": "immutable_registry_successor",
    }
    return metadata


def _proposed_quality_config(current: Mapping[str, Any], registry_sha256: str) -> dict[str, Any]:
    successor = copy.deepcopy(dict(current))
    benchmark = successor.get("benchmark_decontamination")
    if not isinstance(benchmark, dict) or benchmark != {
        "anchor_ngram_tokens": 8,
        "minimum_anchor_matches": 2,
        "registry": "data/benchmark-signatures-v0.1.jsonl.gz",
        "registry_sha256": PREDECESSOR_SHA256,
    }:
        raise ValueError("active data-quality benchmark contract changed")
    benchmark["registry"] = "data/benchmark-signatures-v0.2.jsonl.gz"
    benchmark["registry_sha256"] = registry_sha256
    return successor


def audit_and_build_outputs(
    plan: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(repository_root).resolve()
    bindings = plan["bindings"]
    predecessor_path = resolve_binding(
        root,
        bindings["predecessor_registry"],
        name="predecessor_registry",
    )
    candidate_path = resolve_binding(
        root,
        bindings["candidate_registry"],
        name="candidate_registry",
    )
    quality_path = resolve_binding(
        root,
        bindings["current_data_quality_config"],
        name="current_data_quality_config",
    )
    predecessor_metadata, predecessor_entries = load_registry(predecessor_path)
    candidate_metadata, candidate_entries = load_registry(candidate_path)
    if len(predecessor_entries) != PREDECESSOR_ENTRIES:
        raise ValueError("BigCodeBench predecessor entry count changed")
    if len(candidate_entries) != CANDIDATE_ENTRIES:
        raise ValueError("BigCodeBench candidate entry count changed")
    for line_number, entry in enumerate(candidate_entries, start=1):
        _validate_entry(entry, line_number=line_number)
    if candidate_entries[:PREDECESSOR_ENTRIES] != predecessor_entries:
        raise ValueError("BigCodeBench candidate predecessor sequence changed")
    appended = candidate_entries[PREDECESSOR_ENTRIES:]
    if len(appended) != APPENDED_ENTRIES:
        raise ValueError("BigCodeBench appended entry count changed")
    if any(
        entry.get("benchmark") != "bigcodebench" or entry.get("split") != "v0.1.4"
        for entry in appended
    ):
        raise ValueError("BigCodeBench appended source identity changed")
    field_counts = Counter(
        str(entry["example_id"]).rsplit(":", maxsplit=1)[-1] for entry in appended
    )
    if field_counts != Counter({field: 1_140 for field in PROMPT_FIELDS}):
        raise ValueError("BigCodeBench appended field coverage changed")
    identities = [(entry["benchmark"], entry["example_id"]) for entry in candidate_entries]
    if len(set(identities)) != CANDIDATE_ENTRIES:
        raise ValueError("BigCodeBench candidate identities are duplicated")
    unique_exact = len({entry["exact_sha256"] for entry in candidate_entries})
    unique_bigcodebench_exact = len({entry["exact_sha256"] for entry in appended})
    if unique_exact != EXPECTED_UNIQUE_EXACT_HASHES:
        raise ValueError("BigCodeBench candidate exact-hash inventory changed")
    if unique_bigcodebench_exact != EXPECTED_BIGCODEBENCH_UNIQUE_EXACT_HASHES:
        raise ValueError("BigCodeBench appended exact-hash inventory changed")
    if candidate_metadata.get("predecessor") != {
        "entries": PREDECESSOR_ENTRIES,
        "entries_sha256": records_sha256(predecessor_entries),
    }:
        raise ValueError("BigCodeBench candidate predecessor metadata changed")
    if candidate_metadata.get("appended_source", {}).get("revision") != SOURCE_REVISION:
        raise ValueError("BigCodeBench candidate source revision changed")
    if candidate_metadata.get("entries") != CANDIDATE_ENTRIES:
        raise ValueError("BigCodeBench candidate metadata count changed")
    if candidate_metadata.get("raw_text_retained") is not False:
        raise ValueError("BigCodeBench candidate raw-text policy changed")

    release_metadata = _proposed_release_metadata(candidate_metadata)
    proposed_registry_bytes = deterministic_registry_bytes(release_metadata, candidate_entries)
    proposed_registry_sha256 = hashlib.sha256(proposed_registry_bytes).hexdigest()
    current_quality = _read_json(quality_path)
    proposed_quality = _proposed_quality_config(current_quality, proposed_registry_sha256)
    proposed_quality_bytes = (
        json.dumps(proposed_quality, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    promotion_plan: dict[str, Any] = {
        "authorization": {
            "active_contract_update_authorized": False,
            "promotion_authorized": False,
        },
        "candidate": {
            "entries": CANDIDATE_ENTRIES,
            "file_sha256": CANDIDATE_SHA256,
            "path": "data/benchmark-signatures-v0.2.candidate.jsonl.gz",
        },
        "livecodebench": {
            "included": False,
            "status": "blocked_pending_dataset_content_license_clarification",
        },
        "observed_at": "2026-08-07",
        "predecessor": {
            "entries": PREDECESSOR_ENTRIES,
            "file_sha256": PREDECESSOR_SHA256,
            "path": "data/benchmark-signatures-v0.1.jsonl.gz",
        },
        "proposed_data_quality_successor": {
            "bytes": len(proposed_quality_bytes),
            "content": proposed_quality,
            "file_sha256": hashlib.sha256(proposed_quality_bytes).hexdigest(),
            "path": "configs/data-quality-v0.2.json",
        },
        "proposed_registry_successor": {
            "bytes": len(proposed_registry_bytes),
            "entries": CANDIDATE_ENTRIES,
            "entries_sha256": records_sha256(candidate_entries),
            "file_sha256": proposed_registry_sha256,
            "metadata": release_metadata,
            "path": "data/benchmark-signatures-v0.2.jsonl.gz",
            "raw_text_retained": False,
            "transformation": (
                "preserve all candidate entries in order; replace candidate_not_active=true "
                "with deterministic immutable release metadata"
            ),
        },
        "schema_version": PROMOTION_PLAN_SCHEMA_VERSION,
        "status": "proposed_not_authorized",
    }
    promotion_plan["plan_sha256"] = sha256_value(promotion_plan)

    report: dict[str, Any] = {
        "authorization": plan["authorization"],
        "audit": {
            "appended_entries": len(appended),
            "appended_field_counts": dict(sorted(field_counts.items())),
            "appended_records_sha256": records_sha256(appended),
            "candidate_entries": len(candidate_entries),
            "candidate_entries_sha256": records_sha256(candidate_entries),
            "candidate_exact_hash_reuses": CANDIDATE_ENTRIES - unique_exact,
            "candidate_unique_exact_hashes": unique_exact,
            "identity_collisions": 0,
            "new_exact_hash_reuses": APPENDED_ENTRIES - unique_bigcodebench_exact,
            "new_unique_exact_hashes": unique_bigcodebench_exact,
            "predecessor_entries_preserved_in_order": True,
            "raw_text_or_prohibited_fields_present": False,
        },
        "boundaries": {
            "active_registry_mutated": False,
            "candidate_promoted": False,
            "config_mutated": False,
            "gpu_or_sft_used": False,
            "livecodebench_accessed": False,
            "network_used": False,
            "raw_benchmark_text_in_git": False,
        },
        "observed_at": "2026-08-07",
        "production_eligible": False,
        "promotion_plan_internal_sha256": promotion_plan["plan_sha256"],
        "remaining_blockers": [
            "immutable_registry_promotion_not_authorized",
            "data_quality_successor_materialization_not_authorized",
            "livecodebench_rights_or_replacement_unresolved",
            "opencodeinstruct_candidate_reaudit_not_run",
            "dataset_seal_and_sft_not_authorized",
        ],
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "verified_promotion_preflight_no_promotion",
    }
    report["report_sha256"] = sha256_value(report)
    return report, promotion_plan


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported BigCodeBench promotion-preflight report")
    seal = _require_sha256(report.get("report_sha256"), "report_sha256")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if sha256_value(unsigned) != seal:
        raise ValueError("BigCodeBench promotion-preflight report seal is invalid")
    if report.get("production_eligible") is not False:
        raise ValueError("BigCodeBench promotion preflight crossed production boundary")
    boundaries = report.get("boundaries", {})
    if any(boundaries.get(key) is not False for key in boundaries):
        raise ValueError("BigCodeBench promotion preflight crossed mutation boundary")


def validate_promotion_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != PROMOTION_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported BigCodeBench promotion plan")
    seal = _require_sha256(plan.get("plan_sha256"), "plan_sha256")
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if sha256_value(unsigned) != seal:
        raise ValueError("BigCodeBench promotion plan seal is invalid")
    if plan.get("authorization") != {
        "active_contract_update_authorized": False,
        "promotion_authorized": False,
    }:
        raise ValueError("BigCodeBench promotion plan crossed authorization boundary")


__all__ = [
    "OWNER_AUTHORIZATION_STATEMENT",
    "PLAN_SCHEMA_VERSION",
    "PROMOTION_PLAN_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "VERIFICATION_SCHEMA_VERSION",
    "audit_and_build_outputs",
    "deterministic_registry_bytes",
    "load_preflight_plan",
    "validate_promotion_plan",
    "validate_report",
]
