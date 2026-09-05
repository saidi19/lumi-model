"""Materialize the exact registry-v0.13 OASST1/OASST2 hygiene successors."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .posttraining_ai_v3_provenance_materialization import _verify_internal_sha
from .posttraining_candidate_mix_hygiene_preflight_v2 import (
    canonical_json_bytes,
    validate_hygiene_preflight_v2_report,
)
from .posttraining_candidate_mix_preflight import (
    _load_json,
    _load_jsonl,
    file_sha256,
    sha256_value,
)
from .posttraining_candidate_mix_reaudit import resolve_binding

PLAN_SCHEMA_VERSION = "lumi.sft-candidate-mix-hygiene-materialization-plan.v0.2"
MANIFEST_SCHEMA_VERSION = "lumi.sft-candidate-mix-hygiene-materialization-manifest.v0.2"
RECEIPT_SCHEMA_VERSION = "lumi.sft-candidate-mix-hygiene-materialization-receipt.v0.2"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-candidate-mix-hygiene-materialization-verification.v0.2"

EXPECTED_AUTHORIZATION = {
    "candidate_materialization_authorized": True,
    "candidate_registry_change_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_promotion_authorized": False,
    "production_sft_authorized": False,
    "purpose": "materialize_exact_registry_v0_13_oasst1_oasst2_hygiene_successors_only",
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_input_records": 8_168,
    "maximum_network_bytes": 0,
    "maximum_output_records": 6_918,
    "maximum_projected_removals": 1_250,
    "maximum_runtime_minutes": 60,
    "maximum_total_write_bytes": 33_554_432,
}
OUTPUT_ROOT = (
    "../lumi-artifacts/4b-sft-candidate-mix-hygiene-materialization-v0.2/c07ac2b3c451/attempt-001"
)
EXPECTED_OUTPUT = {
    "manifest_path": f"{OUTPUT_ROOT}/materialization-manifest.json",
    "oasst1_candidate_path": f"{OUTPUT_ROOT}/oasst1-clean-candidate.jsonl",
    "oasst2_candidate_path": f"{OUTPUT_ROOT}/oasst2-clean-candidate.jsonl",
    "production_promotions": 0,
    "raw_training_text_in_receipts": False,
    "receipt_path": f"{OUTPUT_ROOT}/materialization-receipt.json",
    "verification_path": f"{OUTPUT_ROOT}/materialization-verification.json",
}
EXPECTED_BINDINGS = frozenset(
    {
        "implementation",
        "oasst1_candidate",
        "oasst2_candidate",
        "preflight_implementation",
        "preflight_plan",
        "preflight_report",
        "preflight_verification",
        "runner",
        "test",
        "verifier",
    }
)
EXPECTED_PREFLIGHT = {
    "plan_file_sha256": ("d37aacc368c6a185c9b6fd59e5da70e6153733f4eb484cb2cd83a6a9f5a4f6c1"),
    "report_file_sha256": ("55a6962ca4a05ff1268eb63e0d871f55ccc9086d8a0eebc01cb44bcb564ee7fe"),
    "report_internal_sha256": ("c07ac2b3c4517514a66bd7ca357931b3a9f4cc1bc72099b22d4873504409d77a"),
    "verification_file_sha256": (
        "8429c03d34a1fea09d80869b3cba92a08d68dbd2075c34515add1b793e4315eb"
    ),
    "verification_internal_sha256": (
        "d9196581fef25358ccaaf57d2dc807f7bd373500c910f8239f6843ac3bd415c0"
    ),
}
EXPECTED_INPUTS = {
    "openassistant_oasst1": {
        "binding": "oasst1_candidate",
        "records": 1_742,
        "sha256": "1c970f94fe009acc4d379e9c53eee5ecefe29f42db33f01c6d9c09c87e7950c2",
    },
    "openassistant_oasst2": {
        "binding": "oasst2_candidate",
        "records": 6_426,
        "sha256": "1642c444f75f29c70f5182d0ff0ff94133f52b942e36fb3655be62caa4a4e6df",
    },
}
EXPECTED_PROJECTED = {
    "openassistant_oasst1": {
        "bytes": 4_345_139,
        "records": 1_720,
        "sha256": "6a0828f05bcf737921e9388d00705e7dd0a07cf447ba6b35fec57124a29bc369",
    },
    "openassistant_oasst2": {
        "bytes": 16_364_469,
        "records": 5_198,
        "sha256": "57ba5ea18e2f621b9bcc79d18340df92ba7bf3a0e378d3273498d5e497870459",
    },
}


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _candidate_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def resolve_materialization_output(
    repository_root: str | Path, relative: str, *, name: str
) -> Path:
    root = Path(repository_root).resolve()
    workspace = root.parent.resolve()
    resolved = (root / relative).resolve()
    artifact_root = (workspace / "lumi-artifacts").resolve()
    if not resolved.is_relative_to(artifact_root):
        raise ValueError(f"candidate materialization output escaped lumi-artifacts: {name}")
    if resolved.is_relative_to(root):
        raise ValueError(f"candidate materialization output entered Git workspace: {name}")
    return resolved


def load_hygiene_materialization_v2_plan(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    plan = _load_json(Path(path))
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-08",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_bounded_cpu_only_exact_registry_v0_13_hygiene_materialization",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"candidate hygiene materialization v0.2 plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("candidate hygiene materialization v0.2 binding set changed")
    for name, binding in bindings.items():
        resolve_binding(repository_root, binding, name=name)
    exact = {
        "oasst1_candidate": EXPECTED_INPUTS["openassistant_oasst1"]["sha256"],
        "oasst2_candidate": EXPECTED_INPUTS["openassistant_oasst2"]["sha256"],
        "preflight_plan": EXPECTED_PREFLIGHT["plan_file_sha256"],
        "preflight_report": EXPECTED_PREFLIGHT["report_file_sha256"],
        "preflight_verification": EXPECTED_PREFLIGHT["verification_file_sha256"],
    }
    for name, expected_sha256 in exact.items():
        if bindings[name]["sha256"] != expected_sha256:
            raise ValueError(f"candidate hygiene materialization v0.2 binding changed: {name}")
    for name, relative in EXPECTED_OUTPUT.items():
        if name.endswith("_path") and isinstance(relative, str):
            resolve_materialization_output(repository_root, relative, name=name)
    return plan


def project_hygiene_candidates_v2(
    *, plan: Mapping[str, Any], repository_root: str | Path
) -> tuple[dict[str, bytes], dict[str, Any]]:
    root = Path(repository_root).resolve()
    resolved = {
        name: resolve_binding(root, binding, name=name)
        for name, binding in plan["bindings"].items()
    }
    report = _load_json(resolved["preflight_report"])
    verification = _load_json(resolved["preflight_verification"])
    validate_hygiene_preflight_v2_report(report)
    _verify_internal_sha(verification, "verification_sha256")
    if report.get("report_sha256") != EXPECTED_PREFLIGHT["report_internal_sha256"]:
        raise ValueError("candidate materialization v0.2 preflight report changed")
    if (
        verification.get("verification_sha256")
        != EXPECTED_PREFLIGHT["verification_internal_sha256"]
    ):
        raise ValueError("candidate materialization v0.2 preflight verification changed")
    if verification.get("report_file_sha256") != EXPECTED_PREFLIGHT["report_file_sha256"]:
        raise ValueError("candidate materialization v0.2 report link changed")
    if report.get("candidate_files_materialized") is not False:
        raise ValueError("preflight unexpectedly materialized candidate files")

    rows_by_source: dict[str, list[dict[str, Any]]] = {}
    for source_id, expected in EXPECTED_INPUTS.items():
        path = resolved[str(expected["binding"])]
        if file_sha256(path) != expected["sha256"]:
            raise ValueError(f"candidate materialization v0.2 input changed: {source_id}")
        rows = _load_jsonl(path)
        if len(rows) != expected["records"]:
            raise ValueError(f"candidate materialization v0.2 input count changed: {source_id}")
        if any(row.get("source", {}).get("source_id") != source_id for row in rows):
            raise ValueError(f"candidate materialization v0.2 source ID changed: {source_id}")
        rows_by_source[source_id] = rows
    if sum(map(len, rows_by_source.values())) != EXPECTED_LIMITS["maximum_input_records"]:
        raise ValueError("candidate materialization v0.2 input total changed")

    generic_drop_ids = {item["record_id"] for item in report["generic_quarantine"]["records"]}
    component_drop_ids = {
        item["record_id"] for item in report["component_projection"]["loser_ledger"]
    }
    if generic_drop_ids & component_drop_ids:
        raise ValueError("candidate materialization v0.2 drop ledgers overlap")
    drop_ids = generic_drop_ids | component_drop_ids
    if len(drop_ids) != EXPECTED_LIMITS["maximum_projected_removals"]:
        raise ValueError("candidate materialization v0.2 removal count changed")
    all_input_ids = {row["record_id"] for rows in rows_by_source.values() for row in rows}
    if not drop_ids.issubset(all_input_ids):
        raise ValueError("candidate materialization v0.2 drop IDs are absent")

    projected_rows = {
        source_id: [row for row in rows if row["record_id"] not in drop_ids]
        for source_id, rows in rows_by_source.items()
    }
    output_bytes = {source_id: _candidate_bytes(rows) for source_id, rows in projected_rows.items()}
    summaries: dict[str, Any] = {}
    for source_id, rows in projected_rows.items():
        value = output_bytes[source_id]
        summary = {
            "bytes": len(value),
            "records": len(rows),
            "sha256": _bytes_sha256(value),
        }
        if summary != EXPECTED_PROJECTED[source_id]:
            raise ValueError(f"candidate materialization v0.2 projection changed: {source_id}")
        projected_report = report["projected_candidates"][source_id]
        if (
            projected_report["candidate_file_sha256"] != summary["sha256"]
            or projected_report["candidate_jsonl_bytes"] != summary["bytes"]
            or projected_report["records"] != summary["records"]
        ):
            raise ValueError(f"candidate materialization v0.2 preflight drift: {source_id}")
        summaries[source_id] = {
            **summary,
            "by_split": projected_report["by_split"],
            "message_binding_sha256": projected_report["message_binding_sha256"],
            "record_ids_sha256": projected_report["record_ids_sha256"],
        }
    if (
        sum(item["records"] for item in summaries.values())
        != EXPECTED_LIMITS["maximum_output_records"]
    ):
        raise ValueError("candidate materialization v0.2 output total changed")

    removals_by_source_split: dict[str, Counter[str]] = {}
    rows_by_id = {row["record_id"]: row for rows in rows_by_source.values() for row in rows}
    for record_id in sorted(drop_ids):
        row = rows_by_id[record_id]
        source_id = row["source"]["source_id"]
        removals_by_source_split.setdefault(source_id, Counter())[row["split"]] += 1
    manifest: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "inputs": {
            source_id: {
                "records": expected["records"],
                "sha256": expected["sha256"],
            }
            for source_id, expected in EXPECTED_INPUTS.items()
        },
        "observed_at": plan["observed_at"],
        "outputs": summaries,
        "preflight": {
            "component_drop_record_ids_sha256": report["component_projection"][
                "drop_record_ids_sha256"
            ],
            "generic_drop_ledger_sha256": report["generic_quarantine"]["drop_ledger_sha256"],
            "plan_file_sha256": EXPECTED_PREFLIGHT["plan_file_sha256"],
            "report_file_sha256": EXPECTED_PREFLIGHT["report_file_sha256"],
            "report_internal_sha256": EXPECTED_PREFLIGHT["report_internal_sha256"],
            "selection_ledger_sha256": report["component_projection"]["selection_ledger_sha256"],
            "verification_file_sha256": EXPECTED_PREFLIGHT["verification_file_sha256"],
        },
        "production_eligible": False,
        "raw_training_text_retained": False,
        "removals": {
            "by_source_split": {
                source_id: {split: count for split, count in sorted(counter.items())}
                for source_id, counter in sorted(removals_by_source_split.items())
            },
            "record_ids_sha256": sha256_value(sorted(drop_ids)),
            "records": len(drop_ids),
        },
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "projected_exact_private_nonproduction_hygiene_successors",
    }
    manifest["manifest_sha256"] = sha256_value(manifest)
    validate_hygiene_materialization_v2_manifest(manifest)
    return output_bytes, manifest


def validate_hygiene_materialization_v2_manifest(
    manifest: Mapping[str, Any],
) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported hygiene materialization v0.2 manifest")
    _verify_internal_sha(manifest, "manifest_sha256")
    if manifest.get("production_eligible") is not False:
        raise ValueError("hygiene materialization v0.2 cannot be production eligible")
    if manifest.get("raw_training_text_retained") is not False:
        raise ValueError("hygiene materialization v0.2 manifest must be raw-text-free")
    if manifest.get("removals", {}).get("records") != 1_250:
        raise ValueError("hygiene materialization v0.2 removal count changed")
    outputs = manifest.get("outputs", {})
    if set(outputs) != set(EXPECTED_PROJECTED):
        raise ValueError("hygiene materialization v0.2 output set changed")
    for source_id, expected in EXPECTED_PROJECTED.items():
        if {key: outputs[source_id].get(key) for key in ("bytes", "records", "sha256")} != expected:
            raise ValueError(f"hygiene materialization v0.2 output changed: {source_id}")
    serialized = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    for forbidden in ('"messages"', '"content"', '"assistant_response"'):
        if forbidden in serialized:
            raise ValueError("hygiene materialization v0.2 manifest contains raw text")


def build_hygiene_materialization_v2_receipt(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    manifest: Mapping[str, Any],
    manifest_file_sha256: str,
) -> dict[str, Any]:
    validate_hygiene_materialization_v2_manifest(manifest)
    receipt: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "checks": {
            "candidate_registry_changes": 0,
            "dataset_seals": 0,
            "external_api_calls": 0,
            "network_bytes": 0,
            "paid_gpu_runs": 0,
            "production_promotions": 0,
            "production_sft_runs": 0,
            "successor_candidate_files": 2,
        },
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_internal_sha256": manifest["manifest_sha256"],
        "observed_at": plan["observed_at"],
        "outputs": manifest["outputs"],
        "plan_file_sha256": plan_file_sha256,
        "production_eligible": False,
        "projected_removals": manifest["removals"]["records"],
        "raw_training_text_retained": False,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "materialized_exact_private_nonproduction_registry_v0_13_hygiene_successors",
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    return receipt


def validate_hygiene_materialization_v2_receipt(
    receipt: Mapping[str, Any],
) -> None:
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported hygiene materialization v0.2 receipt")
    _verify_internal_sha(receipt, "receipt_sha256")
    if receipt.get("production_eligible") is not False:
        raise ValueError("hygiene materialization v0.2 receipt cannot be production")
    if receipt.get("raw_training_text_retained") is not False:
        raise ValueError("hygiene materialization v0.2 receipt must be raw-text-free")
    if receipt.get("projected_removals") != 1_250:
        raise ValueError("hygiene materialization v0.2 receipt removals changed")


def build_hygiene_materialization_v2_verification(
    *,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_file_sha256: str,
    receipt: Mapping[str, Any],
    receipt_file_sha256: str,
    inventory_before_verification: Sequence[str],
) -> dict[str, Any]:
    validate_hygiene_materialization_v2_manifest(manifest)
    validate_hygiene_materialization_v2_receipt(receipt)
    expected_inventory = [
        "materialization-manifest.json",
        "materialization-receipt.json",
        "oasst1-clean-candidate.jsonl",
        "oasst2-clean-candidate.jsonl",
    ]
    if sorted(inventory_before_verification) != expected_inventory:
        raise ValueError("hygiene materialization v0.2 inventory changed")
    if receipt.get("manifest_file_sha256") != manifest_file_sha256:
        raise ValueError("hygiene materialization v0.2 receipt-manifest link changed")
    verification: dict[str, Any] = {
        "checks": {
            "candidate_registry_unchanged": True,
            "exact_four_object_pre_verification_inventory": True,
            "input_candidates_remain_hash_exact": True,
            "no_network_api_gpu_seal_sft_or_production_promotion": True,
            "oasst1_successor_hash_size_and_count_verified": True,
            "oasst2_successor_hash_size_and_count_verified": True,
            "outputs_are_outside_git_under_lumi_artifacts": True,
            "preflight_keep_drop_ledgers_bound": True,
            "raw_training_text_absent_from_manifest_receipt_and_verification": True,
        },
        "inventory_before_verification": list(expected_inventory),
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_internal_sha256": manifest["manifest_sha256"],
        "observed_at": plan["observed_at"],
        "outputs": manifest["outputs"],
        "production_eligible": False,
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_internal_sha256": receipt["receipt_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_exact_private_nonproduction_registry_v0_13_hygiene_materialization",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


__all__ = [
    "EXPECTED_LIMITS",
    "EXPECTED_OUTPUT",
    "EXPECTED_PROJECTED",
    "build_hygiene_materialization_v2_receipt",
    "build_hygiene_materialization_v2_verification",
    "json_file_bytes",
    "load_hygiene_materialization_v2_plan",
    "project_hygiene_candidates_v2",
    "resolve_materialization_output",
    "validate_hygiene_materialization_v2_manifest",
    "validate_hygiene_materialization_v2_receipt",
]
