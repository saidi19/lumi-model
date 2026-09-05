"""Materialize the exact preflighted Lumi SFT candidate hygiene projection."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .posttraining_candidate_mix_hygiene_preflight import (
    canonical_json_bytes,
    validate_hygiene_preflight_report,
)
from .posttraining_candidate_mix_preflight import (
    _load_json,
    _load_jsonl,
    file_sha256,
    sha256_value,
)
from .posttraining_candidate_mix_reaudit import resolve_binding

PLAN_SCHEMA_VERSION = "lumi.sft-candidate-mix-hygiene-materialization-plan.v0.1"
MANIFEST_SCHEMA_VERSION = "lumi.sft-candidate-mix-hygiene-materialization-manifest.v0.1"
RECEIPT_SCHEMA_VERSION = "lumi.sft-candidate-mix-hygiene-materialization-receipt.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-candidate-mix-hygiene-materialization-verification.v0.1"
)

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
    "purpose": "materialize_exact_oasst1_and_aya_hygiene_successors_only",
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_candidate_records_read": 3_223,
    "maximum_output_candidate_records": 3_200,
    "maximum_projected_removals": 23,
    "maximum_runtime_minutes": 60,
    "maximum_total_write_bytes": 16_777_216,
    "maximum_network_bytes": 0,
}
EXPECTED_OUTPUT = {
    "aya_candidate_path": (
        "../lumi-artifacts/4b-sft-candidate-mix-hygiene-materialization-v0.1/"
        "58348effb427/attempt-001/aya-clean-candidate.jsonl"
    ),
    "manifest_path": (
        "../lumi-artifacts/4b-sft-candidate-mix-hygiene-materialization-v0.1/"
        "58348effb427/attempt-001/materialization-manifest.json"
    ),
    "oasst1_candidate_path": (
        "../lumi-artifacts/4b-sft-candidate-mix-hygiene-materialization-v0.1/"
        "58348effb427/attempt-001/oasst1-clean-candidate.jsonl"
    ),
    "production_promotions": 0,
    "raw_training_text_in_repo_evidence": False,
    "receipt_path": (
        "evidence/lumi-4b-sft-candidate-mix-hygiene-materialization-v0.1.json"
    ),
    "verification_path": (
        "evidence/lumi-4b-sft-candidate-mix-hygiene-materialization-"
        "verification-v0.1.json"
    ),
}
EXPECTED_BINDINGS = frozenset(
    {
        "aya_candidate",
        "implementation",
        "oasst1_candidate",
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
    "plan_file_sha256": (
        "09cfd706e2c2ff850e8a4696d9165a2c72276277229d8e73b478565e7016193c"
    ),
    "report_file_sha256": (
        "95cdb80fc77eacc18757063cd1ff7ab6821e4ea80d10f17b5257614db2c3d0fe"
    ),
    "report_internal_sha256": (
        "58348effb4279bd12117b340ee8ad8c1d2d01dad1a07b983e1da5384cfa21592"
    ),
    "verification_file_sha256": (
        "30be5377fe3684085e811eeae196a48116fd216f2a46ba30d297bfff037fbfb5"
    ),
    "verification_internal_sha256": (
        "fd5b8a962179a5e0cc7ece8727fdce70c3e39f99e549b7c204e454f6decfa784"
    ),
}
EXPECTED_INPUT_HASHES = {
    "coherelabs_aya_dataset": (
        "b9a9154565d769b001fc6a595699bff6bd63587aa66c5c53070257716dad143b"
    ),
    "openassistant_oasst1": (
        "15bdb511c4011fd4680cdf87c84de1cbcc979ce5e3792709e16ed3bef25e7b98"
    ),
}
EXPECTED_PROJECTED = {
    "coherelabs_aya_dataset": {
        "bytes": 4_866_000,
        "records": 1_458,
        "sha256": (
            "6e601c9eee1cc3f90647cd4701a8bf24667dd36e25a42f49f7ef4059b9683558"
        ),
    },
    "openassistant_oasst1": {
        "bytes": 4_395_648,
        "records": 1_742,
        "sha256": (
            "1c970f94fe009acc4d379e9c53eee5ecefe29f42db33f01c6d9c09c87e7950c2"
        ),
    },
}


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _verify_internal_sha(value: Mapping[str, Any], field: str) -> str:
    observed = value.get(field)
    if not isinstance(observed, str) or len(observed) != 64:
        raise ValueError(f"missing internal SHA-256: {field}")
    unsigned = {key: item for key, item in value.items() if key != field}
    if sha256_value(unsigned) != observed:
        raise ValueError(f"internal SHA-256 changed: {field}")
    return observed


def _resolve_output(repository_root: str | Path, relative: str, *, name: str) -> Path:
    root = Path(repository_root).resolve()
    workspace = root.parent.resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"candidate materialization output escapes workspace: {name}")
    return resolved


def load_hygiene_materialization_plan(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    plan = _load_json(Path(path))
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-07",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "authorized_bounded_cpu_only_exact_hygiene_materialization",
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"candidate hygiene materialization plan changed: {field}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("candidate hygiene materialization binding set changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid candidate materialization binding: {name}")
        resolve_binding(repository_root, binding, name=name)
    if bindings["preflight_plan"]["sha256"] != EXPECTED_PREFLIGHT[
        "plan_file_sha256"
    ]:
        raise ValueError("candidate materialization preflight-plan binding changed")
    if bindings["preflight_report"]["sha256"] != EXPECTED_PREFLIGHT[
        "report_file_sha256"
    ]:
        raise ValueError("candidate materialization preflight-report binding changed")
    if bindings["preflight_verification"]["sha256"] != EXPECTED_PREFLIGHT[
        "verification_file_sha256"
    ]:
        raise ValueError("candidate materialization preflight verification changed")
    for name, relative in EXPECTED_OUTPUT.items():
        if name.endswith("_path") and isinstance(relative, str):
            _resolve_output(repository_root, relative, name=name)
    return plan


def _candidate_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _project_candidates(
    *,
    plan: Mapping[str, Any],
    repository_root: str | Path,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    root = Path(repository_root).resolve()
    resolved = {
        name: resolve_binding(root, binding, name=name)
        for name, binding in plan["bindings"].items()
    }
    report = _load_json(resolved["preflight_report"])
    verification = _load_json(resolved["preflight_verification"])
    validate_hygiene_preflight_report(report)
    _verify_internal_sha(verification, "verification_sha256")
    if report.get("report_sha256") != EXPECTED_PREFLIGHT[
        "report_internal_sha256"
    ]:
        raise ValueError("candidate materialization preflight report changed")
    if verification.get("verification_sha256") != EXPECTED_PREFLIGHT[
        "verification_internal_sha256"
    ]:
        raise ValueError("candidate materialization preflight verification changed")
    if verification.get("report_file_sha256") != EXPECTED_PREFLIGHT[
        "report_file_sha256"
    ]:
        raise ValueError("candidate materialization report-verification link changed")
    if report.get("candidate_files_materialized") is not False:
        raise ValueError("preflight unexpectedly materialized candidate files")

    oasst1_rows = _load_jsonl(resolved["oasst1_candidate"])
    aya_rows = _load_jsonl(resolved["aya_candidate"])
    if len(oasst1_rows) + len(aya_rows) != EXPECTED_LIMITS[
        "maximum_candidate_records_read"
    ]:
        raise ValueError("candidate materialization input record count changed")
    if file_sha256(resolved["oasst1_candidate"]) != EXPECTED_INPUT_HASHES[
        "openassistant_oasst1"
    ]:
        raise ValueError("candidate materialization OASST1 source changed")
    if file_sha256(resolved["aya_candidate"]) != EXPECTED_INPUT_HASHES[
        "coherelabs_aya_dataset"
    ]:
        raise ValueError("candidate materialization Aya source changed")

    oasst1_drop_ids = {
        item["record_id"]
        for item in report["oasst1_benchmark_quarantine"]["records"]
    }
    aya_drop_ids = {
        item["drop"]["record_id"]
        for item in report["near_duplicate_projection"]["components"]
    }
    if len(oasst1_drop_ids) != 3 or len(aya_drop_ids) != 20:
        raise ValueError("candidate materialization drop ledger changed")
    oasst1_input_ids = {row["record_id"] for row in oasst1_rows}
    aya_input_ids = {row["record_id"] for row in aya_rows}
    if not oasst1_drop_ids.issubset(oasst1_input_ids):
        raise ValueError("candidate materialization OASST1 drop IDs are absent")
    if not aya_drop_ids.issubset(aya_input_ids):
        raise ValueError("candidate materialization Aya drop IDs are absent")

    projected_rows = {
        "openassistant_oasst1": [
            row for row in oasst1_rows if row["record_id"] not in oasst1_drop_ids
        ],
        "coherelabs_aya_dataset": [
            row for row in aya_rows if row["record_id"] not in aya_drop_ids
        ],
    }
    output_bytes = {
        source_id: _candidate_bytes(rows)
        for source_id, rows in projected_rows.items()
    }
    summaries: dict[str, Any] = {}
    for source_id, rows in projected_rows.items():
        expected = EXPECTED_PROJECTED[source_id]
        value = output_bytes[source_id]
        projected_report = report["projected_candidates"][source_id]
        summary = {
            "bytes": len(value),
            "records": len(rows),
            "sha256": _bytes_sha256(value),
        }
        if summary != expected:
            raise ValueError(f"candidate materialization projection changed: {source_id}")
        if (
            projected_report["candidate_file_sha256"] != summary["sha256"]
            or projected_report["candidate_jsonl_bytes"] != summary["bytes"]
            or projected_report["records"] != summary["records"]
        ):
            raise ValueError(f"candidate materialization preflight drift: {source_id}")
        summaries[source_id] = {
            **summary,
            "by_split": projected_report["by_split"],
            "record_ids_sha256": projected_report["record_ids_sha256"],
            "message_binding_sha256": projected_report["message_binding_sha256"],
        }
    if sum(item["records"] for item in summaries.values()) != EXPECTED_LIMITS[
        "maximum_output_candidate_records"
    ]:
        raise ValueError("candidate materialization projected record count changed")
    return output_bytes, {
        "drop_ledger": {
            "aya_drop_record_ids_sha256": report["near_duplicate_projection"][
                "drop_record_ids_sha256"
            ],
            "aya_records": len(aya_drop_ids),
            "oasst1_drop_ledger_sha256": report["oasst1_benchmark_quarantine"][
                "drop_ledger_sha256"
            ],
            "oasst1_records": len(oasst1_drop_ids),
            "total_records": len(aya_drop_ids) + len(oasst1_drop_ids),
        },
        "outputs": summaries,
        "preflight": {
            "report_file_sha256": EXPECTED_PREFLIGHT["report_file_sha256"],
            "report_internal_sha256": EXPECTED_PREFLIGHT[
                "report_internal_sha256"
            ],
            "verification_file_sha256": EXPECTED_PREFLIGHT[
                "verification_file_sha256"
            ],
            "verification_internal_sha256": EXPECTED_PREFLIGHT[
                "verification_internal_sha256"
            ],
        },
    }


def _write_exact(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != value:
            raise ValueError(f"refusing to overwrite changed materialization output: {path}")
        return
    path.write_bytes(value)


def materialize_hygiene_candidates(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output_bytes, projection = _project_candidates(
        plan=plan,
        repository_root=root,
    )
    output_paths = {
        "coherelabs_aya_dataset": _resolve_output(
            root, plan["output"]["aya_candidate_path"], name="aya_candidate_path"
        ),
        "openassistant_oasst1": _resolve_output(
            root,
            plan["output"]["oasst1_candidate_path"],
            name="oasst1_candidate_path",
        ),
    }
    manifest: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "candidate_outputs": {
            source_id: {
                **projection["outputs"][source_id],
                "path": plan["output"][
                    "aya_candidate_path"
                    if source_id == "coherelabs_aya_dataset"
                    else "oasst1_candidate_path"
                ],
            }
            for source_id in sorted(projection["outputs"])
        },
        "drop_ledger": projection["drop_ledger"],
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "preflight": projection["preflight"],
        "production_eligible": False,
        "raw_training_text_retained": False,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_candidates_immutable": True,
        "status": "private_nonproduction_hygiene_candidates_materialized",
    }
    manifest["manifest_sha256"] = sha256_value(manifest)
    manifest_bytes = json_file_bytes(manifest)

    receipt: dict[str, Any] = {
        "authorization": dict(plan["authorization"]),
        "candidate_outputs": manifest["candidate_outputs"],
        "drop_ledger": projection["drop_ledger"],
        "manifest": {
            "bytes": len(manifest_bytes),
            "path": plan["output"]["manifest_path"],
            "sha256": _bytes_sha256(manifest_bytes),
            "internal_sha256": manifest["manifest_sha256"],
        },
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "preflight": projection["preflight"],
        "production_dataset_promotions": 0,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "source_candidate_files_modified": False,
        "status": "verified_private_nonproduction_hygiene_materialization",
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    receipt_bytes = json_file_bytes(receipt)
    total_write_bytes = (
        sum(len(value) for value in output_bytes.values())
        + len(manifest_bytes)
        + len(receipt_bytes)
    )
    if total_write_bytes > plan["limits"]["maximum_total_write_bytes"]:
        raise ValueError("candidate materialization write ceiling exceeded")

    for source_id, path in output_paths.items():
        _write_exact(path, output_bytes[source_id])
    _write_exact(
        _resolve_output(root, plan["output"]["manifest_path"], name="manifest_path"),
        manifest_bytes,
    )
    _write_exact(
        _resolve_output(root, plan["output"]["receipt_path"], name="receipt_path"),
        receipt_bytes,
    )
    return receipt


def validate_hygiene_materialization_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported candidate materialization receipt schema")
    _verify_internal_sha(receipt, "receipt_sha256")
    if receipt.get("source_candidate_files_modified") is not False:
        raise ValueError("candidate materialization changed a source candidate")
    if receipt.get("production_eligible") is not False:
        raise ValueError("candidate materialization cannot be production eligible")
    if receipt.get("production_dataset_promotions") != 0:
        raise ValueError("candidate materialization cannot promote a dataset")
    if receipt.get("raw_training_text_retained") is not False:
        raise ValueError("candidate materialization receipt must be raw-text-free")
    outputs = receipt.get("candidate_outputs", {})
    for source_id, expected in EXPECTED_PROJECTED.items():
        observed = outputs.get(source_id, {})
        for field in ("bytes", "records", "sha256"):
            if observed.get(field) != expected[field]:
                raise ValueError(f"candidate materialization output changed: {source_id}")
    if receipt.get("drop_ledger", {}).get("total_records") != 23:
        raise ValueError("candidate materialization drop count changed")
    serialized = json.dumps(receipt, ensure_ascii=False, sort_keys=True)
    for forbidden in ('"messages"', '"content"', '"assistant_response"'):
        if forbidden in serialized:
            raise ValueError("candidate materialization receipt contains raw training text")


def build_hygiene_materialization_verification(
    *,
    plan: Mapping[str, Any],
    receipt: Mapping[str, Any],
    receipt_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    validate_hygiene_materialization_receipt(receipt)
    expected_bytes, projection = _project_candidates(
        plan=plan,
        repository_root=root,
    )
    actual_paths = {
        "coherelabs_aya_dataset": _resolve_output(
            root, plan["output"]["aya_candidate_path"], name="aya_candidate_path"
        ),
        "openassistant_oasst1": _resolve_output(
            root,
            plan["output"]["oasst1_candidate_path"],
            name="oasst1_candidate_path",
        ),
    }
    for source_id, path in actual_paths.items():
        if not path.is_file() or path.read_bytes() != expected_bytes[source_id]:
            raise ValueError(f"candidate materialization replay mismatch: {source_id}")
        if file_sha256(path) != projection["outputs"][source_id]["sha256"]:
            raise ValueError(f"candidate materialization output hash changed: {source_id}")
    manifest_path = _resolve_output(
        root, plan["output"]["manifest_path"], name="manifest_path"
    )
    manifest = _load_json(manifest_path)
    _verify_internal_sha(manifest, "manifest_sha256")
    if file_sha256(manifest_path) != receipt["manifest"]["sha256"]:
        raise ValueError("candidate materialization manifest changed")
    written_bytes = (
        sum(path.stat().st_size for path in actual_paths.values())
        + manifest_path.stat().st_size
        + _resolve_output(
            root, plan["output"]["receipt_path"], name="receipt_path"
        ).stat().st_size
    )
    if written_bytes > plan["limits"]["maximum_total_write_bytes"]:
        raise ValueError("candidate materialization verified write ceiling exceeded")
    verification: dict[str, Any] = {
        "checks": {
            "all_23_preflighted_removals_replayed": True,
            "candidate_output_bytes_match_exact_projection": True,
            "candidate_source_files_remain_hash_identical": True,
            "manifest_and_receipt_are_raw_text_free": True,
            "no_network_api_gpu_registry_seal_sft_or_production_promotion": True,
            "output_record_counts_and_hashes_verified": True,
            "private_write_ceiling_respected": True,
        },
        "observed_at": plan["observed_at"],
        "output_candidate_hashes": {
            source_id: projection["outputs"][source_id]["sha256"]
            for source_id in sorted(projection["outputs"])
        },
        "production_eligible": False,
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_internal_sha256": receipt["receipt_sha256"],
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_exact_private_nonproduction_hygiene_materialization",
        "verified_written_bytes_excluding_verification_receipt": written_bytes,
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification

