"""Bounded private materialization for the approved Aya Bengali split."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .posttraining_aya_validation import (
    EXPECTED_PROJECTION,
    build_near_duplicate_components,
    build_split_projection,
    load_aya_validation_plan,
    resolve_validation_binding,
    select_validation_ids,
    validate_validation_preflight_report,
    verify_validation_upstream,
)
from .posttraining_first_party import canonical_json_bytes, file_sha256
from .posttraining_format import ChatTemplate, ChatTokenizer, encode_sft_messages
from .posttraining_oasst1_full_run import load_runtime_components

RECEIPT_SCHEMA_VERSION = "lumi.sft-aya-bengali-validation-materialization-receipt.v0.1"
MANIFEST_SCHEMA_VERSION = "lumi.sft-aya-bengali-split-manifest-row.v0.1"
ATTRIBUTION_BINDING_SCHEMA_VERSION = "lumi.sft-aya-bengali-attribution-binding.v0.1"

OWNER_AUTHORIZATION = (
    "I approve one bounded CPU-only materialization of the exact Aya Bengali split "
    "projected by preflight receipt "
    "48b21e9f5b1ce07290beab67c3f77a33572a0ab887b871e19fcebc5b99fc8628: "
    "1,330 train records / 520,269 supervised tokens and 148 validation records / "
    "57,184 supervised tokens, maximum 1,478 records / 16 MiB private writes / 60 "
    "minutes. It may write only the private split candidate, raw-text-free split "
    "manifest, attribution binding, and verification receipt outside Git. No "
    "network/API, GPU, dataset seal, SFT training, source-registry change, production "
    "promotion, unrelated data access, or external distribution."
)
OWNER_AUTHORIZATION_SHA256 = hashlib.sha256(OWNER_AUTHORIZATION.encode()).hexdigest()

EXPECTED_PREFLIGHT_PLAN_FILE_SHA256 = (
    "d2a92c89ede411cddd4c6d32335b41e978b1dad08fe946b36200d0e0263ad724"
)
EXPECTED_PREFLIGHT_REPORT_FILE_SHA256 = (
    "3f021e617592cee83d83addcf6bf30cbf0eac31690fa3af3a3ea8044f1cbf2ef"
)
EXPECTED_PREFLIGHT_RECEIPT_SHA256 = (
    "48b21e9f5b1ce07290beab67c3f77a33572a0ab887b871e19fcebc5b99fc8628"
)
EXPECTED_RECORDS = 1_478
EXPECTED_SUPERVISED_TOKENS = 577_453
MAXIMUM_PRIVATE_WRITE_BYTES = 16 * 1024 * 1024
MAXIMUM_RUNTIME_MINUTES = 60
PRIVATE_ROOT_RELATIVE = (
    "../lumi-artifacts/4b-sft-aya-bengali-validation-materialization-v0.1/"
    "48b21e9f5b1c/attempt-001"
)
OUTPUT_FILENAMES = {
    "attribution_binding": "attribution-binding.json",
    "candidate": "split-candidate.jsonl",
    "manifest": "split-manifest.jsonl",
    "receipt": "verification-receipt.json",
}
EXPECTED_AUTHORIZATION = {
    "candidate_split_materialization_authorized": True,
    "dataset_seal_authorized": False,
    "external_api_spend_authorized": False,
    "external_distribution_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_dataset_promotion_authorized": False,
    "production_sft_authorized": False,
    "source_registry_change_authorized": False,
    "unrelated_data_access_authorized": False,
    "purpose": "exact_private_aya_bengali_train_validation_split_only",
}


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def json_file_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def jsonl_file_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if any(not isinstance(value, dict) for value in values):
        raise ValueError(f"JSONL rows must be objects: {path}")
    return values


def resolve_private_root(repository_root: str | Path) -> Path:
    root = Path(repository_root).resolve()
    workspace = root.parent.resolve()
    private_root = (root / PRIVATE_ROOT_RELATIVE).resolve()
    if private_root.is_relative_to(root):
        raise ValueError("Aya split materialization entered the Git repository")
    if not private_root.is_relative_to(workspace):
        raise ValueError("Aya split materialization escaped the workspace")
    return private_root


def load_exact_materialization_inputs(
    repository_root: str | Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    ChatTemplate,
    ChatTokenizer,
]:
    """Load only the exact hash-bound candidate approved by the split preflight."""

    root = Path(repository_root).resolve()
    plan_path = root / "configs/lumi-4b-sft-aya-bengali-validation-preflight-v0.1.json"
    report_path = root / "evidence/lumi-4b-sft-aya-bengali-validation-preflight-v0.1.json"
    if file_sha256(plan_path) != EXPECTED_PREFLIGHT_PLAN_FILE_SHA256:
        raise ValueError("Aya validation preflight plan changed after approval")
    if file_sha256(report_path) != EXPECTED_PREFLIGHT_REPORT_FILE_SHA256:
        raise ValueError("Aya validation preflight report changed after approval")
    plan = load_aya_validation_plan(plan_path, repository_root=root)
    report = _read_json(report_path)
    validate_validation_preflight_report(report)
    if (
        report.get("report_sha256") != EXPECTED_PREFLIGHT_RECEIPT_SHA256
        or report.get("plan_file_sha256") != EXPECTED_PREFLIGHT_PLAN_FILE_SHA256
        or report.get("projection") != EXPECTED_PROJECTION
    ):
        raise ValueError("Aya validation preflight receipt differs from owner approval")

    bindings = plan["bindings"]
    verify_validation_upstream(
        materialization_report=_read_json(
            resolve_validation_binding(root, bindings["materialization_report"])
        ),
        owner_review_receipt=_read_json(
            resolve_validation_binding(root, bindings["owner_review_receipt"])
        ),
        source_promotion_receipt=_read_json(
            resolve_validation_binding(root, bindings["source_promotion_receipt"])
        ),
    )
    rows = _read_jsonl(resolve_validation_binding(root, bindings["candidate_records"]))
    attribution = _read_jsonl(
        resolve_validation_binding(root, bindings["attribution_manifest"])
    )
    _policy, _registry, _benchmark, template, tokenizer = load_runtime_components(
        plan,
        repository_root=root,
    )
    return plan, report, rows, attribution, template, tokenizer


def _build_attribution_binding(
    *,
    plan: Mapping[str, Any],
    attribution_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    record_ids = [str(row.get("record_id")) for row in attribution_rows]
    if len(record_ids) != EXPECTED_RECORDS or len(set(record_ids)) != EXPECTED_RECORDS:
        raise ValueError("Aya attribution membership changed")
    fields = {
        "dataset_id": {str(row.get("dataset_id")) for row in attribution_rows},
        "license_id": {str(row.get("license_id")) for row in attribution_rows},
        "revision": {str(row.get("revision")) for row in attribution_rows},
        "source_id": {str(row.get("source_id")) for row in attribution_rows},
    }
    if any(len(values) != 1 for values in fields.values()):
        raise ValueError("Aya attribution source binding is not unique")
    value: dict[str, Any] = {
        "attribution_manifest_file_sha256": plan["bindings"]["attribution_manifest"][
            "sha256"
        ],
        "attribution_record_ids_sha256": sha256_value(sorted(record_ids)),
        "attribution_rows": len(attribution_rows),
        "candidate_input_file_sha256": plan["bindings"]["candidate_records"]["sha256"],
        "dataset_id": next(iter(fields["dataset_id"])),
        "license_id": next(iter(fields["license_id"])),
        "owner_review_receipt_file_sha256": plan["bindings"]["owner_review_receipt"][
            "sha256"
        ],
        "revision": next(iter(fields["revision"])),
        "schema_version": ATTRIBUTION_BINDING_SCHEMA_VERSION,
        "source_id": next(iter(fields["source_id"])),
        "source_promotion_receipt_file_sha256": plan["bindings"][
            "source_promotion_receipt"
        ]["sha256"],
    }
    value["binding_sha256"] = sha256_value(value)
    return value


def build_materialized_split(
    rows: Sequence[Mapping[str, Any]],
    attribution_rows: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    template: ChatTemplate,
    tokenizer: ChatTokenizer,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Reproduce the exact projection and return the private candidate and raw-free ledgers."""

    projection = build_split_projection(
        rows,
        attribution_rows,
        plan=plan,
        template=template,
        tokenizer=tokenizer,
    )
    components = build_near_duplicate_components(rows, policy=plan["policy"])
    attribution_by_id = {str(row["record_id"]): row for row in attribution_rows}
    validation_ids, _component_count = select_validation_ids(
        components,
        attribution_by_id=attribution_by_id,
        plan=plan,
    )

    candidate: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for source_row in sorted(rows, key=lambda row: str(row["record_id"])):
        record_id = str(source_row["record_id"])
        split = "validation" if record_id in validation_ids else "train"
        row = {**copy.deepcopy(dict(source_row)), "split": split}
        encoded = encode_sft_messages(row["messages"], tokenizer=tokenizer, template=template)
        candidate.append(row)
        manifest.append(
            {
                "attribution_row_sha256": sha256_value(attribution_by_id[record_id]),
                "masked_tokens": encoded.masked_tokens,
                "record_id": record_id,
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "serialized_tokens": len(encoded.input_ids),
                "split": split,
                "supervised_tokens": encoded.supervised_tokens,
            }
        )

    by_split = {
        split: {
            "records": sum(row["split"] == split for row in manifest),
            "supervised_tokens": sum(
                int(row["supervised_tokens"]) for row in manifest if row["split"] == split
            ),
        }
        for split in ("train", "validation")
    }
    if by_split != {
        "train": {"records": 1330, "supervised_tokens": 520269},
        "validation": {"records": 148, "supervised_tokens": 57184},
    }:
        raise ValueError("Aya materialized split metrics changed")
    attribution_binding = _build_attribution_binding(
        plan=plan,
        attribution_rows=attribution_rows,
    )
    return candidate, manifest, attribution_binding, projection


def build_artifact_entry(
    *,
    logical_path: Path,
    payload: bytes,
    repository_root: Path,
    contains_raw_training_text: bool,
) -> dict[str, Any]:
    return {
        "bytes": len(payload),
        "contains_raw_training_text": contains_raw_training_text,
        "path": logical_path.relative_to(repository_root.parent).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def build_materialization_receipt(
    *,
    plan: Mapping[str, Any],
    projection: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    implementation_file_sha256: str,
    runner_file_sha256: str,
) -> dict[str, Any]:
    if projection != EXPECTED_PROJECTION:
        raise ValueError("Aya receipt projection changed")
    if set(artifacts) != {"attribution_binding", "candidate", "manifest"}:
        raise ValueError("Aya materialization artifact inventory changed")
    receipt: dict[str, Any] = {
        "artifacts": {key: dict(value) for key, value in sorted(artifacts.items())},
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {
            "attribution_manifest_file_sha256": plan["bindings"]["attribution_manifest"][
                "sha256"
            ],
            "candidate_input_file_sha256": plan["bindings"]["candidate_records"]["sha256"],
            "implementation_file_sha256": implementation_file_sha256,
            "owner_review_receipt_file_sha256": plan["bindings"]["owner_review_receipt"][
                "sha256"
            ],
            "preflight_plan_file_sha256": EXPECTED_PREFLIGHT_PLAN_FILE_SHA256,
            "preflight_report_file_sha256": EXPECTED_PREFLIGHT_REPORT_FILE_SHA256,
            "preflight_receipt_sha256": EXPECTED_PREFLIGHT_RECEIPT_SHA256,
            "runner_file_sha256": runner_file_sha256,
            "source_promotion_receipt_file_sha256": plan["bindings"][
                "source_promotion_receipt"
            ]["sha256"],
        },
        "candidate": {
            "materialized": True,
            "records": EXPECTED_RECORDS,
            "supervised_tokens": EXPECTED_SUPERVISED_TOKENS,
        },
        "limits": {
            "maximum_private_write_bytes": MAXIMUM_PRIVATE_WRITE_BYTES,
            "maximum_records": EXPECTED_RECORDS,
            "maximum_runtime_minutes": MAXIMUM_RUNTIME_MINUTES,
        },
        "observed_at": "2026-08-06",
        "owner_authorization_sha256": OWNER_AUTHORIZATION_SHA256,
        "production": {
            "dataset_sealed": False,
            "production_dataset_promotions": 0,
            "production_eligible": False,
            "production_sft_authorized": False,
            "source_registry_changed": False,
        },
        "projection": copy.deepcopy(dict(projection)),
        "raw_training_text_in_receipt": False,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "private_aya_split_materialized_nonproduction",
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    return receipt


def validate_materialization_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported Aya split materialization receipt")
    seal = receipt.get("receipt_sha256")
    if not isinstance(seal, str) or sha256_value(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    ) != seal:
        raise ValueError("Aya split materialization receipt seal changed")
    if (
        receipt.get("authorization") != EXPECTED_AUTHORIZATION
        or receipt.get("owner_authorization_sha256") != OWNER_AUTHORIZATION_SHA256
        or receipt.get("projection") != EXPECTED_PROJECTION
        or receipt.get("candidate")
        != {
            "materialized": True,
            "records": EXPECTED_RECORDS,
            "supervised_tokens": EXPECTED_SUPERVISED_TOKENS,
        }
        or receipt.get("production")
        != {
            "dataset_sealed": False,
            "production_dataset_promotions": 0,
            "production_eligible": False,
            "production_sft_authorized": False,
            "source_registry_changed": False,
        }
        or receipt.get("raw_training_text_in_receipt") is not False
    ):
        raise ValueError("Aya split materialization boundary changed")


def verify_materialized_output(
    private_root: Path,
    *,
    repository_root: Path,
    implementation_path: Path,
    runner_path: Path,
) -> dict[str, Any]:
    """Independently rebuild and verify every private materialized artifact."""

    expected_names = set(OUTPUT_FILENAMES.values())
    observed_names = {path.name for path in private_root.iterdir() if path.is_file()}
    if observed_names != expected_names:
        raise ValueError("Aya private output inventory changed")
    plan, _report, rows, attribution, template, tokenizer = load_exact_materialization_inputs(
        repository_root
    )
    expected_candidate, expected_manifest, expected_binding, projection = (
        build_materialized_split(
            rows,
            attribution,
            plan=plan,
            template=template,
            tokenizer=tokenizer,
        )
    )
    candidate_path = private_root / OUTPUT_FILENAMES["candidate"]
    manifest_path = private_root / OUTPUT_FILENAMES["manifest"]
    binding_path = private_root / OUTPUT_FILENAMES["attribution_binding"]
    receipt_path = private_root / OUTPUT_FILENAMES["receipt"]
    if _read_jsonl(candidate_path) != expected_candidate:
        raise ValueError("Aya private split candidate changed")
    if _read_jsonl(manifest_path) != expected_manifest:
        raise ValueError("Aya raw-text-free split manifest changed")
    if _read_json(binding_path) != expected_binding:
        raise ValueError("Aya attribution binding changed")

    artifacts = {
        "attribution_binding": build_artifact_entry(
            logical_path=binding_path,
            payload=binding_path.read_bytes(),
            repository_root=repository_root,
            contains_raw_training_text=False,
        ),
        "candidate": build_artifact_entry(
            logical_path=candidate_path,
            payload=candidate_path.read_bytes(),
            repository_root=repository_root,
            contains_raw_training_text=True,
        ),
        "manifest": build_artifact_entry(
            logical_path=manifest_path,
            payload=manifest_path.read_bytes(),
            repository_root=repository_root,
            contains_raw_training_text=False,
        ),
    }
    expected_receipt = build_materialization_receipt(
        plan=plan,
        projection=projection,
        artifacts=artifacts,
        implementation_file_sha256=file_sha256(implementation_path),
        runner_file_sha256=file_sha256(runner_path),
    )
    receipt = _read_json(receipt_path)
    if receipt != expected_receipt:
        raise ValueError("Aya verification receipt changed")
    validate_materialization_receipt(receipt)
    total_bytes = sum(
        (private_root / filename).stat().st_size for filename in OUTPUT_FILENAMES.values()
    )
    if total_bytes > MAXIMUM_PRIVATE_WRITE_BYTES:
        raise ValueError("Aya split materialization exceeded the private-write bound")
    return {
        "candidate_file_sha256": file_sha256(candidate_path),
        "manifest_file_sha256": file_sha256(manifest_path),
        "receipt_file_sha256": file_sha256(receipt_path),
        "receipt_sha256": receipt["receipt_sha256"],
        "total_private_bytes": total_bytes,
    }
