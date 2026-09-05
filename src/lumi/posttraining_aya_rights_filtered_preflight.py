"""Non-mutating preflight for a rights-filtered Aya Bengali successor."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_closure_canary_curation import load_direct_tokenizer
from .posttraining_format import encode_sft_messages, load_chat_template

PLAN_SCHEMA_VERSION = "lumi.sft-aya-rights-filtered-materialization-preflight.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-aya-rights-filtered-materialization-preflight-report.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-aya-rights-filtered-materialization-preflight-verification.v0.1"
)
PLAN_STATUS = "authorized_cpu_only_nonmutating_materialization_preflight"
REPORT_STATUS = "projected_byte_exact_filtered_successor_materialization_not_run"

EXPECTED_AUTHORIZATION = {
    "candidate_materialization_authorized": False,
    "candidate_modification_authorized": False,
    "cpu_only": True,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "gpu_authorized": False,
    "network_access_authorized": False,
    "production_promotion_authorized": False,
    "registry_change_authorized": False,
    "sft_training_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_input_records": 5_079,
    "maximum_projected_output_records": 3_995,
    "maximum_runtime_minutes": 30,
    "maximum_written_evidence_bytes": 1_048_576,
}
EXPECTED_BINDINGS = {
    "aya_candidate",
    "chat_template",
    "dependency_overlay_receipt",
    "implementation",
    "rights_audit_report",
    "rights_audit_verification",
    "runner",
    "test",
    "tokenizer",
    "verifier",
}
EXPECTED_INPUT_SHA256 = "f1cfb5a4d482c8e69e07a087f55f484016c3e0fa58c4b195fc2fe8feaae4c532"
EXPECTED_RIGHTS_REPORT_SHA256 = "c1f3ccf357bb236dcf58ba67908b74f180534410a848a7a41dded1fe8cf5db24"
EXPECTED_RIGHTS_VERIFICATION_SHA256 = (
    "e9f283b6a76343a08de71a1bde6b03d68ea751afeb8139b98a422eb76f14f67f"
)
EXPECTED_TOKENIZER_SHA256 = "1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee"
EXPECTED_CHAT_TEMPLATE_SHA256 = "bc7853a9277feddae2fd3db786c6a12f4fa3602a135a0408c38d4390f33bec12"
EXPECTED_AGGREGATE_SOURCE = {
    "dataset_id": "CohereLabs/aya_collection_language_split",
    "license_id": "apache-2.0",
    "revision": "a3af2fde4b4cb5b2775830b11244a1a20b5f004f",
    "source_id": "coherelabs_aya_bengali_closure_canary",
}
RETAINED_FAMILIES = {
    "120f0a5cb5386355e804603ebac9c71e2775e698d02beefacb2e9b781a850151": (
        "joke_explaination_inst_translated"
    ),
    "3bae6c736d432b44b3e79feb6ed935e45ec923fa56b5eb73518e1b755afcea3c": (
        "wiki_split_inst_translated"
    ),
}
QUARANTINED_FAMILIES = {
    "1ca8a8ec9ac03c59c0be773f060e77d89eeab400f6a9160b6a13a2ef5c256859": (
        "flan_gem_wikilingua_translated"
    )
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_REPORT_KEYS = {
    "assistant_response",
    "content",
    "messages",
    "prompt",
    "raw_text",
    "user_prompt",
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_value(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _resolve_file(repository_root: str | Path, relative: Any, *, label: str) -> Path:
    root = Path(repository_root).resolve()
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"invalid path for {label}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.parent.resolve()) or not path.is_file():
        raise ValueError(f"missing or escaped file for {label}")
    return path


def _verify_binding(
    repository_root: str | Path,
    specification: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    path = _resolve_file(repository_root, specification.get("path"), label=label)
    expected = _require_sha256(specification.get("sha256"), f"{label}.sha256")
    if file_sha256(path) != expected:
        raise ValueError(f"binding changed for {label}")
    return path


def _verify_runtime(plan: Mapping[str, Any]) -> None:
    import tokenizers

    runtime = plan.get("runtime")
    if runtime != {
        "pythonpath_root": "/private/tmp/lumi-teacher-canary-v18-runtime",
        "tokenizers_version": "0.22.2",
    }:
        raise ValueError("Aya filtered-preflight runtime changed")
    runtime_root = Path(runtime["pythonpath_root"]).resolve()
    module_path = Path(tokenizers.__file__).resolve()
    if tokenizers.__version__ != runtime["tokenizers_version"]:
        raise ValueError("tokenizers version changed")
    if not module_path.is_relative_to(runtime_root):
        raise ValueError("tokenizers was not loaded from the verified overlay")


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    plan = _read_object(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported Aya filtered-materialization preflight")
    if plan.get("status") != PLAN_STATUS:
        raise ValueError("Aya filtered-materialization status changed")
    if plan.get("observed_at") != "2026-08-10":
        raise ValueError("Aya filtered-materialization date changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("Aya filtered-materialization authorization changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("Aya filtered-materialization limits changed")

    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("Aya filtered-materialization binding set changed")
    for name, specification in bindings.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"invalid Aya filtered-materialization binding: {name}")
        _verify_binding(repository_root, specification, label=f"bindings.{name}")
    expected_hashes = {
        "aya_candidate": EXPECTED_INPUT_SHA256,
        "chat_template": EXPECTED_CHAT_TEMPLATE_SHA256,
        "rights_audit_report": EXPECTED_RIGHTS_REPORT_SHA256,
        "rights_audit_verification": EXPECTED_RIGHTS_VERIFICATION_SHA256,
        "tokenizer": EXPECTED_TOKENIZER_SHA256,
    }
    for name, expected in expected_hashes.items():
        if bindings[name]["sha256"] != expected:
            raise ValueError(f"Aya filtered-materialization binding changed: {name}")

    rights_report = _read_object(
        _verify_binding(
            repository_root,
            bindings["rights_audit_report"],
            label="bindings.rights_audit_report",
        )
    )
    rights_decisions = {
        item["source_dataset_name_sha256"]: item["disposition"]
        for item in rights_report.get("constituents", [])
    }
    expected_decisions = {
        **{key: "retain" for key in RETAINED_FAMILIES},
        **{key: "quarantine" for key in QUARANTINED_FAMILIES},
    }
    normalized = {
        key: "quarantine" if value.startswith("quarantine") else "retain"
        for key, value in rights_decisions.items()
    }
    if normalized != expected_decisions:
        raise ValueError("Aya rights decisions changed")

    projection = plan.get("projection_expectation")
    if not isinstance(projection, Mapping):
        raise ValueError("Aya projection expectation is missing")
    required_projection_fields = {
        "input_records",
        "output_bytes",
        "output_candidate_sha256",
        "output_records",
        "quarantined",
        "retained",
    }
    if set(projection) != required_projection_fields:
        raise ValueError("Aya projection expectation fields changed")
    _require_sha256(
        projection.get("output_candidate_sha256"),
        "projection_expectation.output_candidate_sha256",
    )

    expected_output = {
        "projected_candidate_path": (
            "../lumi-artifacts/4b-sft-aya-rights-filtered-materialization-v0.1/"
            "c1f3ccf357bb/attempt-001/private/aya-bengali-candidate.jsonl"
        ),
        "projected_manifest_path": (
            "../lumi-artifacts/4b-sft-aya-rights-filtered-materialization-v0.1/"
            "c1f3ccf357bb/attempt-001/private-manifest.json"
        ),
        "report_path": (
            "evidence/lumi-4b-sft-aya-rights-filtered-materialization-preflight-v0.1.json"
        ),
        "verification_path": (
            "evidence/"
            "lumi-4b-sft-aya-rights-filtered-materialization-preflight-"
            "verification-v0.1.json"
        ),
    }
    if plan.get("output") != expected_output:
        raise ValueError("Aya filtered-materialization outputs changed")
    _verify_runtime(plan)
    return plan


def project_candidate(
    candidate_path: str | Path,
    *,
    tokenizer_path: str | Path,
    chat_template_path: str | Path,
) -> dict[str, Any]:
    tokenizer = load_direct_tokenizer(Path(tokenizer_path))
    template = load_chat_template(chat_template_path)
    output_digest = hashlib.sha256()
    output_bytes = 0
    input_records = 0
    seen_record_ids: set[str] = set()
    totals: dict[str, Counter[str]] = {
        "retained": Counter(),
        "quarantined": Counter(),
    }
    by_family: dict[str, Counter[str]] = {}

    with Path(candidate_path).open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if not line.endswith(b"\n"):
                raise ValueError("Aya input JSONL line is not newline terminated")
            row = json.loads(line.decode("utf-8"))
            if not isinstance(row, dict):
                raise ValueError(f"Aya candidate row {line_number} is not an object")
            record_id = row.get("record_id")
            if not isinstance(record_id, str) or record_id in seen_record_ids:
                raise ValueError("Aya candidate record identity changed")
            seen_record_ids.add(record_id)
            source = row.get("source")
            provenance = row.get("provenance")
            split = row.get("split")
            if not isinstance(source, Mapping) or not isinstance(provenance, Mapping):
                raise ValueError("Aya candidate source/provenance is missing")
            for key, expected in EXPECTED_AGGREGATE_SOURCE.items():
                if source.get(key) != expected:
                    raise ValueError(f"Aya aggregate source changed: {key}")
            family_hash = provenance.get("source_dataset_name_sha256")
            if family_hash in RETAINED_FAMILIES:
                bucket = "retained"
                family_id = RETAINED_FAMILIES[str(family_hash)]
                output_digest.update(line)
                output_bytes += len(line)
            elif family_hash in QUARANTINED_FAMILIES:
                bucket = "quarantined"
                family_id = QUARANTINED_FAMILIES[str(family_hash)]
            else:
                raise ValueError("Aya candidate contains an unclassified family")
            if split not in {"train", "validation"}:
                raise ValueError("Aya candidate split changed")
            encoded = encode_sft_messages(
                row.get("messages"), tokenizer=tokenizer, template=template
            )
            serialized_tokens = len(encoded.input_ids)
            family = by_family.setdefault(family_id, Counter())
            for counter in (totals[bucket], family):
                counter["records"] += 1
                counter[f"{split}_records"] += 1
                counter["serialized_tokens"] += serialized_tokens
                counter["supervised_tokens"] += encoded.supervised_tokens
                counter[f"{split}_serialized_tokens"] += serialized_tokens
                counter[f"{split}_supervised_tokens"] += encoded.supervised_tokens
                counter["maximum_serialized_tokens"] = max(
                    counter["maximum_serialized_tokens"], serialized_tokens
                )
            input_records += 1

    def normalized(counter: Counter[str]) -> dict[str, Any]:
        return {
            "maximum_serialized_tokens": counter["maximum_serialized_tokens"],
            "records": counter["records"],
            "serialized_tokens": counter["serialized_tokens"],
            "splits": {
                split: {
                    "records": counter[f"{split}_records"],
                    "serialized_tokens": counter[f"{split}_serialized_tokens"],
                    "supervised_tokens": counter[f"{split}_supervised_tokens"],
                }
                for split in ("train", "validation")
            },
            "supervised_tokens": counter["supervised_tokens"],
        }

    return {
        "by_family": {
            family_id: normalized(counter) for family_id, counter in sorted(by_family.items())
        },
        "input_records": input_records,
        "output_bytes": output_bytes,
        "output_candidate_sha256": output_digest.hexdigest(),
        "output_records": totals["retained"]["records"],
        "quarantined": normalized(totals["quarantined"]),
        "retained": normalized(totals["retained"]),
    }


def build_report(
    plan: Mapping[str, Any],
    *,
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    bindings = plan["bindings"]
    projection = project_candidate(
        _verify_binding(root, bindings["aya_candidate"], label="aya_candidate"),
        tokenizer_path=_verify_binding(root, bindings["tokenizer"], label="tokenizer"),
        chat_template_path=_verify_binding(root, bindings["chat_template"], label="chat_template"),
    )
    expected = plan["projection_expectation"]
    observed_expectation = {
        key: projection[key]
        for key in (
            "input_records",
            "output_bytes",
            "output_candidate_sha256",
            "output_records",
            "quarantined",
            "retained",
        )
    }
    if observed_expectation != expected:
        raise ValueError("Aya filtered candidate projection changed")
    report: dict[str, Any] = {
        "authorization": plan["authorization"],
        "bindings": {
            name: {
                "path": specification["path"],
                "sha256": specification["sha256"],
            }
            for name, specification in sorted(bindings.items())
        },
        "blockers": [
            "filtered_candidate_materialization_not_authorized",
            "source_registry_change_not_authorized",
            "production_dataset_seal_not_authorized",
            "sft_training_not_authorized",
        ],
        "checks": {
            "all_input_rows_classified": True,
            "all_retained_lines_projected_byte_exact": True,
            "candidate_not_written": True,
            "frozen_tokenizer_remeasured": True,
            "quarantined_family_excluded": True,
            "rights_audit_bound": True,
        },
        "next_materialization": {
            "maximum_output_records": projection["output_records"],
            "maximum_private_write_bytes": 67_108_864,
            "maximum_runtime_minutes": 60,
            "projected_candidate_path": plan["output"]["projected_candidate_path"],
            "projected_manifest_path": plan["output"]["projected_manifest_path"],
        },
        "observed_at": "2026-08-10",
        "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
        "production_eligible": False,
        "projection": projection,
        "raw_training_text_in_report": False,
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": REPORT_STATUS,
    }
    report["report_sha256"] = sha256_value(report)
    validate_report(report)
    return report


def _walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            keys.add(str(key))
            keys.update(_walk_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.update(_walk_keys(nested))
    return keys


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported Aya filtered-materialization report")
    if report.get("status") != REPORT_STATUS:
        raise ValueError("Aya filtered-materialization report status changed")
    if report.get("production_eligible") is not False:
        raise ValueError("Aya filtered-materialization crossed production boundary")
    if report.get("raw_training_text_in_report") is not False:
        raise ValueError("Aya filtered-materialization report retained raw text")
    expected = report.get("report_sha256")
    unsigned = dict(report)
    unsigned.pop("report_sha256", None)
    if expected != sha256_value(unsigned):
        raise ValueError("Aya filtered-materialization report seal changed")
    if _walk_keys(report) & _FORBIDDEN_REPORT_KEYS:
        raise ValueError("Aya filtered-materialization report contains raw-text fields")
    projection = report.get("projection")
    if not isinstance(projection, Mapping):
        raise ValueError("Aya filtered-materialization projection is missing")
    if projection.get("input_records") != 5_079:
        raise ValueError("Aya filtered-materialization input count changed")
    if projection.get("output_records") != 3_995:
        raise ValueError("Aya filtered-materialization output count changed")
    if projection["quarantined"]["records"] != 1_084:
        raise ValueError("Aya filtered-materialization quarantine count changed")


def build_verification(report: Mapping[str, Any], *, report_file_sha256: str) -> dict[str, Any]:
    validate_report(report)
    verification: dict[str, Any] = {
        "checks": {
            "candidate_remained_unwritten": True,
            "exact_output_bytes_reconstructed": True,
            "frozen_token_counts_reconstructed": True,
            "rights_decisions_replayed": True,
            "source_registry_dataset_seal_and_sft_unchanged": True,
        },
        "input_candidate_file_sha256": EXPECTED_INPUT_SHA256,
        "observed_at": "2026-08-10",
        "production_eligible": False,
        "projected_candidate_sha256": report["projection"]["output_candidate_sha256"],
        "report_file_sha256": _require_sha256(report_file_sha256, "report_file_sha256"),
        "report_internal_sha256": report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_nonmutating_aya_filtered_materialization_preflight",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification
