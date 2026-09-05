"""Byte-exact private materialization of the rights-filtered Aya candidate."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_aya_rights_filtered_preflight import (
    EXPECTED_AGGREGATE_SOURCE,
    QUARANTINED_FAMILIES,
    RETAINED_FAMILIES,
    file_sha256,
    project_candidate,
    sha256_value,
)
from .posttraining_aya_rights_filtered_preflight import (
    validate_report as validate_preflight_report,
)

PLAN_SCHEMA_VERSION = "lumi.sft-aya-rights-filtered-materialization.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-aya-rights-filtered-materialization-report.v0.1"
MANIFEST_SCHEMA_VERSION = "lumi.sft-aya-rights-filtered-candidate-manifest.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-aya-rights-filtered-materialization-verification.v0.1"
PLAN_STATUS = "authorized_cpu_only_byte_exact_private_materialization"
REPORT_STATUS = "materialized_private_nonproduction_aya_rights_filtered_candidate"

OWNER_AUTHORIZATION = (
    "I approve one bounded CPU-only byte-exact Aya rights-filtered candidate "
    "materialization, bound to preflight report file SHA-256 "
    "76fd3d1e1a013517eb815a93efce71adf028428563b7c2a504dcae200be47e5e and "
    "verification file SHA-256 "
    "c7fe627e1c835061cb99f0fa2abff6ca0f3d57dbf453f01fd3aeaf87b0b98bb6, "
    "maximum 5,079 input records / 3,995 output records / 67,108,864 private "
    "write bytes / 60 minutes. It may preserve the projected 3,995 retained "
    "JSONL lines byte-exact, retain only the WikiSplit and Joke families, "
    "quarantine all 1,084 WikiLingua rows, write the private candidate and "
    "raw-text-free manifest, and independently verify frozen-token counts and "
    "hashes. No network/API/GPU, registry change, dataset seal, SFT training, "
    "production promotion, unrelated access, or external distribution."
)
OWNER_AUTHORIZATION_SHA256 = hashlib.sha256(OWNER_AUTHORIZATION.encode()).hexdigest()

EXPECTED_AUTHORIZATION = {
    "candidate_materialization_authorized": True,
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
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_input_records": 5_079,
    "maximum_output_records": 3_995,
    "maximum_private_write_bytes": 67_108_864,
    "maximum_runtime_minutes": 60,
}
EXPECTED_BINDINGS = {
    "aya_candidate",
    "chat_template",
    "data_contract",
    "dependency_overlay_receipt",
    "implementation",
    "preflight_plan",
    "preflight_report",
    "preflight_verification",
    "runner",
    "source_registry",
    "test",
    "tokenizer",
    "verifier",
}
EXPECTED_HASHES = {
    "aya_candidate": "f1cfb5a4d482c8e69e07a087f55f484016c3e0fa58c4b195fc2fe8feaae4c532",
    "chat_template": "bc7853a9277feddae2fd3db786c6a12f4fa3602a135a0408c38d4390f33bec12",
    "data_contract": "9a283d9333d29deb7b516fa45eb4eec50232be9925f4ee4aa3a8f624213b2f06",
    "dependency_overlay_receipt": (
        "72b047f333348a1844d30a1112631a0eac8bfaa4805e3473f4f2b089555dd9c1"
    ),
    "preflight_plan": "c3b243baaff55ba39fee9d11647c217cfd8d653b15beaad60c1d99d356f3d89f",
    "preflight_report": "76fd3d1e1a013517eb815a93efce71adf028428563b7c2a504dcae200be47e5e",
    "preflight_verification": ("c7fe627e1c835061cb99f0fa2abff6ca0f3d57dbf453f01fd3aeaf87b0b98bb6"),
    "source_registry": "de81a51f0288b4851ff5ea4ca0598ddcf4936b2bf2c4127ab2b1c0d6deb1a4de",
    "tokenizer": "1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee",
}
EXPECTED_OUTPUT_CANDIDATE_SHA256 = (
    "48795a958a86a4b4df06f50a1f1185c07f4772fba1cc48f2b673166bc0c7b9aa"
)
EXPECTED_OUTPUT_BYTES = 9_251_318
EXPECTED_OUTPUT_RECORDS = 3_995
EXPECTED_QUARANTINED_RECORDS = 1_084
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_RAW_KEYS = {
    "assistant_response",
    "content",
    "messages",
    "prompt",
    "raw_text",
    "user_prompt",
}


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


def _resolve_binding(
    repository_root: str | Path,
    specification: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    root = Path(repository_root).resolve()
    relative = specification.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"invalid Aya materialization binding: {name}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.parent.resolve()) or not path.is_file():
        raise ValueError(f"missing or escaped Aya materialization binding: {name}")
    expected = _require_sha256(specification.get("sha256"), f"bindings.{name}.sha256")
    if file_sha256(path) != expected:
        raise ValueError(f"Aya materialization binding changed: {name}")
    return path


def _verify_runtime(plan: Mapping[str, Any]) -> None:
    import tokenizers

    expected = {
        "pythonpath_root": "/private/tmp/lumi-teacher-canary-v18-runtime",
        "tokenizers_version": "0.22.2",
    }
    if plan.get("runtime") != expected:
        raise ValueError("Aya materialization runtime changed")
    if tokenizers.__version__ != expected["tokenizers_version"]:
        raise ValueError("Aya materialization tokenizers version changed")
    if (
        not Path(tokenizers.__file__)
        .resolve()
        .is_relative_to(Path(expected["pythonpath_root"]).resolve())
    ):
        raise ValueError("Aya materialization tokenizers escaped verified overlay")


def _validate_internal_seal(value: Mapping[str, Any], field: str, label: str) -> None:
    unsigned = dict(value)
    observed = unsigned.pop(field, None)
    if observed != sha256_value(unsigned):
        raise ValueError(f"{label} internal seal changed")


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    plan = _read_object(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported Aya rights-filtered materialization plan")
    if plan.get("status") != PLAN_STATUS or plan.get("observed_at") != "2026-08-10":
        raise ValueError("Aya rights-filtered materialization status/date changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("Aya rights-filtered materialization authorization changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("Aya rights-filtered materialization limits changed")
    if plan.get("owner_authorization_sha256") != OWNER_AUTHORIZATION_SHA256:
        raise ValueError("Aya rights-filtered owner authorization changed")

    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("Aya rights-filtered materialization bindings changed")
    for name, specification in bindings.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"invalid Aya materialization binding: {name}")
        _resolve_binding(repository_root, specification, name=name)
    for name, expected in EXPECTED_HASHES.items():
        if bindings[name]["sha256"] != expected:
            raise ValueError(f"Aya fixed materialization binding changed: {name}")

    preflight = _read_object(
        _resolve_binding(repository_root, bindings["preflight_report"], name="preflight_report")
    )
    validate_preflight_report(preflight)
    projection = preflight.get("projection")
    if not isinstance(projection, Mapping):
        raise ValueError("Aya approved preflight projection is missing")
    if (
        projection.get("input_records") != EXPECTED_LIMITS["maximum_input_records"]
        or projection.get("output_records") != EXPECTED_OUTPUT_RECORDS
        or projection.get("output_bytes") != EXPECTED_OUTPUT_BYTES
        or projection.get("output_candidate_sha256") != EXPECTED_OUTPUT_CANDIDATE_SHA256
        or projection.get("quarantined", {}).get("records") != EXPECTED_QUARANTINED_RECORDS
    ):
        raise ValueError("Aya approved preflight projection changed")

    verification = _read_object(
        _resolve_binding(
            repository_root,
            bindings["preflight_verification"],
            name="preflight_verification",
        )
    )
    _validate_internal_seal(
        verification,
        "verification_sha256",
        "Aya materialization preflight verification",
    )
    if (
        verification.get("status") != "verified_nonmutating_aya_filtered_materialization_preflight"
        or verification.get("report_file_sha256") != EXPECTED_HASHES["preflight_report"]
        or verification.get("projected_candidate_sha256") != EXPECTED_OUTPUT_CANDIDATE_SHA256
    ):
        raise ValueError("Aya materialization preflight verification changed")

    expected_output = {
        "candidate_path": (
            "../lumi-artifacts/4b-sft-aya-rights-filtered-materialization-v0.1/"
            "c1f3ccf357bb/attempt-001/private/aya-bengali-candidate.jsonl"
        ),
        "manifest_path": (
            "../lumi-artifacts/4b-sft-aya-rights-filtered-materialization-v0.1/"
            "c1f3ccf357bb/attempt-001/private-manifest.json"
        ),
        "report_path": ("evidence/lumi-4b-sft-aya-rights-filtered-materialization-v0.1.json"),
        "verification_path": (
            "evidence/lumi-4b-sft-aya-rights-filtered-materialization-verification-v0.1.json"
        ),
    }
    if plan.get("output") != expected_output:
        raise ValueError("Aya rights-filtered materialization outputs changed")
    _verify_runtime(plan)
    return plan


def _extract_candidate(input_path: str | Path) -> tuple[bytes, dict[str, Any]]:
    payload = bytearray()
    input_records = 0
    retained_ids = hashlib.sha256()
    quarantined_ids = hashlib.sha256()
    by_bucket: dict[str, Counter[str]] = {
        "retained": Counter(),
        "quarantined": Counter(),
    }
    by_family: dict[str, Counter[str]] = {}
    seen_ids: set[str] = set()

    with Path(input_path).open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if not line.endswith(b"\n"):
                raise ValueError("Aya candidate contains a non-terminated JSONL line")
            row = json.loads(line.decode("utf-8"))
            if not isinstance(row, dict):
                raise ValueError(f"Aya candidate row {line_number} is not an object")
            record_id = row.get("record_id")
            source = row.get("source")
            provenance = row.get("provenance")
            split = row.get("split")
            if not isinstance(record_id, str) or record_id in seen_ids:
                raise ValueError("Aya candidate record identity changed")
            seen_ids.add(record_id)
            if not isinstance(source, Mapping) or not isinstance(provenance, Mapping):
                raise ValueError("Aya candidate source/provenance is missing")
            if any(source.get(key) != value for key, value in EXPECTED_AGGREGATE_SOURCE.items()):
                raise ValueError("Aya aggregate source binding changed")
            if split not in {"train", "validation"}:
                raise ValueError("Aya candidate split changed")
            family_hash = provenance.get("source_dataset_name_sha256")
            if family_hash in RETAINED_FAMILIES:
                bucket = "retained"
                family_id = RETAINED_FAMILIES[str(family_hash)]
                payload.extend(line)
                retained_ids.update(record_id.encode())
                retained_ids.update(b"\n")
            elif family_hash in QUARANTINED_FAMILIES:
                bucket = "quarantined"
                family_id = QUARANTINED_FAMILIES[str(family_hash)]
                quarantined_ids.update(record_id.encode())
                quarantined_ids.update(b"\n")
            else:
                raise ValueError("Aya candidate contains an unclassified family")
            by_bucket[bucket]["records"] += 1
            by_bucket[bucket][f"{split}_records"] += 1
            family = by_family.setdefault(family_id, Counter())
            family["records"] += 1
            family[f"{split}_records"] += 1
            input_records += 1

    digest = hashlib.sha256(payload).hexdigest()
    if (
        input_records != EXPECTED_LIMITS["maximum_input_records"]
        or by_bucket["retained"]["records"] != EXPECTED_OUTPUT_RECORDS
        or by_bucket["quarantined"]["records"] != EXPECTED_QUARANTINED_RECORDS
        or len(payload) != EXPECTED_OUTPUT_BYTES
        or digest != EXPECTED_OUTPUT_CANDIDATE_SHA256
    ):
        raise ValueError("Aya byte-exact materialization projection changed")
    return bytes(payload), {
        "by_family": {
            family_id: {
                "records": counter["records"],
                "splits": {
                    split: {"records": counter[f"{split}_records"]}
                    for split in ("train", "validation")
                },
            }
            for family_id, counter in sorted(by_family.items())
        },
        "input_records": input_records,
        "quarantined_record_ids_sha256": quarantined_ids.hexdigest(),
        "retained_record_ids_sha256": retained_ids.hexdigest(),
    }


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


def build_materialization(
    plan: Mapping[str, Any],
    *,
    plan_file_sha256: str,
    repository_root: str | Path,
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    root = Path(repository_root).resolve()
    bindings = plan["bindings"]
    input_path = _resolve_binding(root, bindings["aya_candidate"], name="aya_candidate")
    preflight = _read_object(
        _resolve_binding(root, bindings["preflight_report"], name="preflight_report")
    )
    projection = project_candidate(
        input_path,
        tokenizer_path=_resolve_binding(root, bindings["tokenizer"], name="tokenizer"),
        chat_template_path=_resolve_binding(root, bindings["chat_template"], name="chat_template"),
    )
    if projection != preflight["projection"]:
        raise ValueError("Aya frozen-token projection changed before materialization")
    candidate_payload, membership = _extract_candidate(input_path)

    retained_families = []
    for family_hash, family_id in sorted(RETAINED_FAMILIES.items(), key=lambda item: item[1]):
        retained_families.append(
            {
                "family_id": family_id,
                "metrics": projection["by_family"][family_id],
                "source_dataset_name_sha256": family_hash,
            }
        )
    quarantined_families = []
    for family_hash, family_id in sorted(QUARANTINED_FAMILIES.items(), key=lambda item: item[1]):
        quarantined_families.append(
            {
                "family_id": family_id,
                "metrics": projection["by_family"][family_id],
                "source_dataset_name_sha256": family_hash,
            }
        )

    manifest: dict[str, Any] = {
        "authorization_sha256": OWNER_AUTHORIZATION_SHA256,
        "candidate": {
            "bytes": len(candidate_payload),
            "maximum_serialized_tokens": projection["retained"]["maximum_serialized_tokens"],
            "records": projection["output_records"],
            "serialized_tokens": projection["retained"]["serialized_tokens"],
            "sha256": hashlib.sha256(candidate_payload).hexdigest(),
            "splits": projection["retained"]["splits"],
            "supervised_tokens": projection["retained"]["supervised_tokens"],
        },
        "checks": {
            "input_lines_preserved_byte_exact": True,
            "quarantined_rows_excluded": True,
            "raw_training_text_in_manifest": False,
            "rights_audit_bound": True,
        },
        "input_candidate_sha256": EXPECTED_HASHES["aya_candidate"],
        "membership": membership,
        "preflight_report_file_sha256": EXPECTED_HASHES["preflight_report"],
        "preflight_verification_file_sha256": EXPECTED_HASHES["preflight_verification"],
        "production_eligible": False,
        "quarantined_families": quarantined_families,
        "retained_families": retained_families,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_registry_sha256": EXPECTED_HASHES["source_registry"],
        "status": "private_byte_exact_rights_filtered_candidate",
    }
    manifest["manifest_sha256"] = sha256_value(manifest)
    manifest_payload = canonical_json_bytes(manifest)

    report: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "bindings": {
            "input_candidate_file_sha256": EXPECTED_HASHES["aya_candidate"],
            "owner_authorization_sha256": OWNER_AUTHORIZATION_SHA256,
            "preflight_report_file_sha256": EXPECTED_HASHES["preflight_report"],
            "preflight_verification_file_sha256": EXPECTED_HASHES["preflight_verification"],
            "source_registry_file_sha256": EXPECTED_HASHES["source_registry"],
            "tokenizer_file_sha256": EXPECTED_HASHES["tokenizer"],
        },
        "checks": {
            "all_input_rows_classified": True,
            "candidate_lines_preserved_byte_exact": True,
            "dataset_sealed": False,
            "network_api_or_gpu_used": False,
            "quarantined_rows_written": 0,
            "registry_changed": False,
            "sft_training_started": False,
        },
        "input_records": projection["input_records"],
        "observed_at": "2026-08-10",
        "output": {
            "candidate_bytes": len(candidate_payload),
            "candidate_file_sha256": hashlib.sha256(candidate_payload).hexdigest(),
            "manifest_file_sha256": hashlib.sha256(manifest_payload).hexdigest(),
            "manifest_internal_sha256": manifest["manifest_sha256"],
            "private_write_bytes": len(candidate_payload) + len(manifest_payload),
            "records": projection["output_records"],
            "train_records": projection["retained"]["splits"]["train"]["records"],
            "train_supervised_tokens": projection["retained"]["splits"]["train"][
                "supervised_tokens"
            ],
            "validation_records": projection["retained"]["splits"]["validation"]["records"],
            "validation_supervised_tokens": projection["retained"]["splits"]["validation"][
                "supervised_tokens"
            ],
        },
        "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
        "production_eligible": False,
        "quarantined": projection["quarantined"],
        "raw_training_text_in_report": False,
        "retained": projection["retained"],
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": REPORT_STATUS,
    }
    report["report_sha256"] = sha256_value(report)
    validate_report(report)
    total_bytes = len(candidate_payload) + len(manifest_payload) + len(canonical_json_bytes(report))
    if total_bytes > int(plan["limits"]["maximum_private_write_bytes"]):
        raise ValueError("Aya rights-filtered materialization exceeded write limit")
    return candidate_payload, manifest, report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported Aya rights-filtered materialization report")
    if report.get("status") != REPORT_STATUS or report.get("production_eligible") is not False:
        raise ValueError("Aya rights-filtered materialization crossed production boundary")
    _validate_internal_seal(report, "report_sha256", "Aya materialization report")
    if report.get("raw_training_text_in_report") is not False:
        raise ValueError("Aya materialization report retained raw training text")
    if _walk_keys(report) & _FORBIDDEN_RAW_KEYS:
        raise ValueError("Aya materialization report contains raw-text fields")
    output = report.get("output")
    checks = report.get("checks")
    if not isinstance(output, Mapping) or not isinstance(checks, Mapping):
        raise ValueError("Aya materialization report is incomplete")
    if (
        output.get("records") != EXPECTED_OUTPUT_RECORDS
        or output.get("candidate_bytes") != EXPECTED_OUTPUT_BYTES
        or output.get("candidate_file_sha256") != EXPECTED_OUTPUT_CANDIDATE_SHA256
        or report.get("quarantined", {}).get("records") != EXPECTED_QUARANTINED_RECORDS
        or checks.get("quarantined_rows_written") != 0
        or checks.get("registry_changed") is not False
        or checks.get("sft_training_started") is not False
    ):
        raise ValueError("Aya rights-filtered materialization gates changed")


def verify_materialized_outputs(
    plan: Mapping[str, Any],
    *,
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    expected_candidate, expected_manifest, expected_report = build_materialization(
        plan,
        plan_file_sha256=plan_file_sha256,
        repository_root=root,
    )
    candidate_path = (root / plan["output"]["candidate_path"]).resolve()
    manifest_path = (root / plan["output"]["manifest_path"]).resolve()
    report_path = (root / plan["output"]["report_path"]).resolve()
    if candidate_path.read_bytes() != expected_candidate:
        raise ValueError("Aya materialized candidate bytes do not reproduce")
    if manifest_path.read_bytes() != canonical_json_bytes(expected_manifest):
        raise ValueError("Aya materialized manifest does not reproduce")
    if report_path.read_bytes() != canonical_json_bytes(expected_report):
        raise ValueError("Aya materialization report does not reproduce")

    output_projection = project_candidate(
        candidate_path,
        tokenizer_path=_resolve_binding(root, plan["bindings"]["tokenizer"], name="tokenizer"),
        chat_template_path=_resolve_binding(
            root, plan["bindings"]["chat_template"], name="chat_template"
        ),
    )
    approved_projection = _read_object(
        _resolve_binding(root, plan["bindings"]["preflight_report"], name="preflight_report")
    )["projection"]
    if (
        output_projection["input_records"] != EXPECTED_OUTPUT_RECORDS
        or output_projection["output_records"] != EXPECTED_OUTPUT_RECORDS
        or output_projection["output_candidate_sha256"] != EXPECTED_OUTPUT_CANDIDATE_SHA256
        or output_projection["output_bytes"] != EXPECTED_OUTPUT_BYTES
        or output_projection["retained"] != approved_projection["retained"]
        or output_projection["quarantined"]["records"] != 0
        or set(output_projection["by_family"]) != set(RETAINED_FAMILIES.values())
    ):
        raise ValueError("Aya materialized output token or family replay changed")
    if (
        file_sha256(
            _resolve_binding(root, plan["bindings"]["source_registry"], name="source_registry")
        )
        != EXPECTED_HASHES["source_registry"]
    ):
        raise ValueError("Aya materialization changed source registry")

    verification: dict[str, Any] = {
        "checks": {
            "candidate_bytes_reconstructed_from_approved_input": True,
            "candidate_contains_only_two_retained_families": True,
            "frozen_token_counts_reconstructed_from_materialized_output": True,
            "manifest_and_report_reproduced_byte_exact": True,
            "no_network_api_gpu_registry_dataset_seal_or_sft": True,
            "wiki_lingua_records_in_output": 0,
        },
        "manifest_file_sha256": file_sha256(manifest_path),
        "manifest_internal_sha256": expected_manifest["manifest_sha256"],
        "observed_at": "2026-08-10",
        "output_candidate_file_sha256": file_sha256(candidate_path),
        "production_eligible": False,
        "report_file_sha256": file_sha256(report_path),
        "report_internal_sha256": expected_report["report_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "source_registry_file_sha256": EXPECTED_HASHES["source_registry"],
        "status": "verified_private_nonproduction_aya_rights_filtered_materialization",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification
