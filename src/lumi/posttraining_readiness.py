"""Fail-closed readiness binding for Lumi's sealed 80B-token base model."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_data import SFTPolicy
from .posttraining_evaluation import validate_evaluation_plan
from .posttraining_format import ChatTemplate

BASE_BINDING_SCHEMA_VERSION = "lumi.posttraining-base-binding.v0.1"
READINESS_SCHEMA_VERSION = "lumi.posttraining-readiness.v0.1"
FINAL_SOURCE_COMMIT = "418a113021daddb0ea8e364e2a20c3eabc4ab9bf"
FINAL_RUN_FINGERPRINT = "578feadb13558d48ad537a9fb843dd9e3c9baf07079d14f32f601c9d3be5f768"
FINAL_CHECKPOINT_MANIFEST_SHA256 = (
    "73c852095441a4246cea7bb97f7831c364137869e00e29c815b7e22a5972f8de"
)
FINAL_HANDOFF_SHA256 = "8258d67fa0cd53a9b1075c4f5f0c17e14e40fb928ef2a53f36a523f55763de4b"
FINAL_HANDOFF_FILE_SHA256 = (
    "63788dfec4199ddcf908227db1bd6f6852da3500df1a4e4d8092e149579d4afa"
)
FINAL_EVALUATION_SHA256 = "3047ca1c426bf7d815715fe1accaa30d06abe34c4b774f6da707e2e2baf35bd0"
FINAL_TOKENIZER_MANIFEST_SHA256 = (
    "282d7caa5a9a952c8af66fb28679ee34fa3c844c28e8922f4b4e297ca0fdaab9"
)
FINAL_BACKUP_PLAN_SHA256 = "6e2fd6dccccac816832b3d98aed20862a7d3cd501c63233f8888fab08d60dedc"
FINAL_BACKUP_RECEIPT_SHA256 = (
    "b826123e42ac71c3cb15f3a67d5e057507781286331f147fc0ab0fb584059ea5"
)
FINAL_BACKUP_VERIFICATION_SHA256 = (
    "d42475846f8ecb23d87f9be7964d769f285b034bebdd7112716ebbbea456c1f8"
)
FINAL_BACKUP_BUCKET = "lumi-corpus-prod-27777c02b29b76efcac3"
FINAL_BACKUP_PREFIX = (
    "lumi-corpus/v0.1/control/backups/4b-80b-production-v0.1/"
    f"{FINAL_RUN_FINGERPRINT}/segment-3/{FINAL_HANDOFF_SHA256}"
)
FINAL_STEP = 76_294
FINAL_TOKENS = 80_000_000_000
FINAL_SEQUENCE_CURSOR = 19_531_250
EXPECTED_TOKENIZER_FILES = {
    "tokenizer/lumi_tokenizer_metadata.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def tokenizer_manifest_sha256(files: Any) -> str:
    if not isinstance(files, list) or len(files) != len(EXPECTED_TOKENIZER_FILES):
        raise ValueError("tokenizer file inventory changed")
    normalized = []
    for item in files:
        if not isinstance(item, Mapping):
            raise ValueError("tokenizer file entry must be an object")
        relative_path = item.get("relative_path")
        size = item.get("bytes")
        digest = item.get("sha256")
        if relative_path not in EXPECTED_TOKENIZER_FILES:
            raise ValueError("unexpected tokenizer file")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("tokenizer file size is invalid")
        if not _valid_sha256(digest):
            raise ValueError("tokenizer file SHA-256 is invalid")
        normalized.append(
            {"bytes": size, "relative_path": relative_path, "sha256": digest}
        )
    if {item["relative_path"] for item in normalized} != EXPECTED_TOKENIZER_FILES:
        raise ValueError("tokenizer file inventory is incomplete")
    return _sha256(sorted(normalized, key=lambda item: item["relative_path"]))


def validate_final_base_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != BASE_BINDING_SCHEMA_VERSION:
        raise ValueError("unsupported final base binding schema")
    if value.get("model_id") != "lumi-4b-base" or value.get("checkpoint_stage") != "base":
        raise ValueError("final base binding targets the wrong model stage")
    if value.get("source_commit") != FINAL_SOURCE_COMMIT or _GIT_SHA.fullmatch(
        str(value.get("source_commit", ""))
    ) is None:
        raise ValueError("final base source commit changed")
    if value.get("run_fingerprint") != FINAL_RUN_FINGERPRINT:
        raise ValueError("final base run fingerprint changed")

    checkpoint = value.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint binding is required")
    expected_checkpoint = {
        "manifest_sha256": FINAL_CHECKPOINT_MANIFEST_SHA256,
        "sequence_cursor": FINAL_SEQUENCE_CURSOR,
        "step": FINAL_STEP,
        "training_tokens": FINAL_TOKENS,
        "world_size": 8,
    }
    if dict(checkpoint) != expected_checkpoint:
        raise ValueError("final checkpoint boundary or identity changed")

    handoff = value.get("handoff")
    if not isinstance(handoff, Mapping) or handoff.get("sha256") != FINAL_HANDOFF_SHA256:
        raise ValueError("final handoff changed")
    if handoff.get("file_sha256") != FINAL_HANDOFF_FILE_SHA256:
        raise ValueError("final handoff file SHA-256 changed")

    evaluation = value.get("final_evaluation")
    if not isinstance(evaluation, Mapping) or evaluation.get("sha256") != FINAL_EVALUATION_SHA256:
        raise ValueError("final frozen evaluation binding changed")
    if evaluation.get("relative_path") != (
        "evidence/evaluations/step-00076294/evaluation-report.json"
    ):
        raise ValueError("final frozen evaluation path changed")

    tokenizer = value.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise ValueError("tokenizer binding is required")
    derived_tokenizer_hash = tokenizer_manifest_sha256(tokenizer.get("files"))
    if (
        tokenizer.get("manifest_sha256") != derived_tokenizer_hash
        or derived_tokenizer_hash != FINAL_TOKENIZER_MANIFEST_SHA256
    ):
        raise ValueError("tokenizer manifest SHA-256 does not match its files")

    backup = value.get("s3_backup")
    if not isinstance(backup, Mapping) or backup.get("status") != "verified":
        raise ValueError("S3 backup is not verified")
    expected_backup_hashes = {
        "plan_sha256": FINAL_BACKUP_PLAN_SHA256,
        "receipt_sha256": FINAL_BACKUP_RECEIPT_SHA256,
        "verification_report_sha256": FINAL_BACKUP_VERIFICATION_SHA256,
    }
    if any(backup.get(field) != digest for field, digest in expected_backup_hashes.items()):
        raise ValueError("S3 backup evidence identity changed")
    if backup.get("archive_objects") != 52 or backup.get("archive_bytes") != 48_844_509_979:
        raise ValueError("S3 archive inventory changed")
    if backup.get("inventory_objects") != 53 or backup.get("inventory_bytes") != 48_844_545_022:
        raise ValueError("S3 receipt-inclusive inventory changed")
    if backup.get("bucket") != FINAL_BACKUP_BUCKET or backup.get("prefix") != FINAL_BACKUP_PREFIX:
        raise ValueError("S3 backup location changed")
    return json.loads(json.dumps(value))


def load_final_base_binding(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("final base binding must be a JSON object")
    return validate_final_base_binding(value)


def build_posttraining_readiness_report(
    *,
    base_binding: Mapping[str, Any],
    chat_template: Mapping[str, Any],
    evaluation_plan: Mapping[str, Any],
    sft_policy: Mapping[str, Any],
) -> dict[str, Any]:
    binding = validate_final_base_binding(base_binding)
    ChatTemplate.from_mapping(chat_template)
    validate_evaluation_plan(evaluation_plan)
    SFTPolicy.from_mapping(sft_policy)
    report = {
        "base_binding_sha256": _sha256(binding),
        "base_checkpoint_ready": True,
        "base_checkpoint_manifest_sha256": FINAL_CHECKPOINT_MANIFEST_SHA256,
        "blockers": [
            "sft_dataset:not_curated_or_sealed",
            "sft_training:not_authorized",
            "posttraining_benchmarks:not_run",
            "safety_evaluation:not_run",
        ],
        "chat_template_sha256": _sha256(chat_template),
        "chat_template_ready": True,
        "evaluation_plan_sha256": _sha256(evaluation_plan),
        "next_gate": "curate_filter_deduplicate_and_seal_sft_dataset",
        "paid_gpu_authorized": False,
        "schema_version": READINESS_SCHEMA_VERSION,
        "sft_data_contract_sha256": _sha256(sft_policy),
        "sft_data_contract_ready": True,
        "tokenizer_manifest_sha256": binding["tokenizer"]["manifest_sha256"],
    }
    report["report_sha256"] = _sha256(report)
    return report
