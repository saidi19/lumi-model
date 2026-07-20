"""Fail-closed final seal for Lumi's 120B production candidate corpus."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .emr_production_shard_writer import (
    dedup_result_fingerprint,
    shard_writer_job_token,
    validate_global_dedup_result,
)
from .emr_serverless_production import (
    validate_paid_production_authorization,
    validate_production_checkpoint,
)

SCHEMA_VERSION = 1


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{label} is outside its allowed range")
    return value


def _safe_key(value: Any, *, namespace: str, label: str) -> str:
    if not isinstance(value, str) or not value.startswith(f"{namespace}/"):
        raise ValueError(f"{label} left the production namespace")
    if value.startswith("/") or "//" in value or "/../" in value:
        raise ValueError(f"{label} is unsafe")
    return value


def _normalize_baseline_shard(
    shard: Mapping[str, Any], *, namespace: str
) -> dict[str, Any]:
    category = shard.get("category")
    if not isinstance(category, str):
        raise ValueError("baseline shard category is invalid")
    return {
        "category": category,
        "clean_bytes": None,
        "clean_object_key": _safe_key(
            shard.get("clean_object_key"), namespace=namespace, label="baseline clean object"
        ),
        "clean_sha256": _sha256(shard.get("clean_sha256"), "baseline clean object"),
        "documents": _positive_int(shard.get("documents"), "baseline shard documents"),
        "manifest_object_key": _safe_key(
            shard.get("manifest_object_key"),
            namespace=namespace,
            label="baseline manifest object",
        ),
        "manifest_sha256": _sha256(
            shard.get("manifest_sha256"), "baseline manifest"
        ),
        "origin": "independently_verified_resume_prefix",
        "provenance_bytes": None,
        "provenance_object_key": _safe_key(
            shard.get("provenance_object_key"),
            namespace=namespace,
            label="baseline provenance object",
        ),
        "provenance_sha256": _sha256(
            shard.get("provenance_sha256"), "baseline provenance object"
        ),
        "sequence": _positive_int(
            shard.get("sequence"), "baseline shard sequence", allow_zero=True
        ),
        "shard_id": str(shard.get("shard_id")),
        "text_bytes": _positive_int(shard.get("text_bytes"), "baseline text bytes"),
        "token_bytes": _positive_int(shard.get("tokens"), "baseline tokens") * 2,
        "token_object_key": _safe_key(
            shard.get("token_object_key"),
            namespace=namespace,
            label="baseline token object",
        ),
        "token_sha256": _sha256(shard.get("token_sha256"), "baseline token object"),
        "tokens": _positive_int(shard.get("tokens"), "baseline shard tokens"),
    }


def _normalize_production_shard(
    shard: Mapping[str, Any],
    *,
    category: str,
    namespace: str,
    maximum_tokens: int,
) -> dict[str, Any]:
    if shard.get("category") != category:
        raise ValueError("shard-writer output crosses categories")
    tokens = _positive_int(shard.get("tokens"), "production shard tokens")
    if tokens > maximum_tokens:
        raise ValueError("production shard exceeds its bounded document overshoot")
    token_bytes = _positive_int(shard.get("token_object_bytes"), "token object bytes")
    if token_bytes != tokens * 2:
        raise ValueError("production shard token-byte conservation failed")
    return {
        "category": category,
        "clean_bytes": _positive_int(shard.get("object_bytes"), "clean object bytes"),
        "clean_object_key": _safe_key(
            shard.get("object_key"), namespace=namespace, label="clean object"
        ),
        "clean_sha256": _sha256(shard.get("object_sha256"), "clean object"),
        "documents": _positive_int(shard.get("documents"), "production shard documents"),
        "manifest_object_key": _safe_key(
            shard.get("manifest_object_key"), namespace=namespace, label="manifest object"
        ),
        "manifest_sha256": _sha256(shard.get("manifest_sha256"), "manifest"),
        "origin": "emr_global_dedup_shard_writer",
        "provenance_bytes": _positive_int(
            shard.get("provenance_index_bytes"), "provenance object bytes"
        ),
        "provenance_object_key": _safe_key(
            shard.get("provenance_index_key"),
            namespace=namespace,
            label="provenance object",
        ),
        "provenance_sha256": _sha256(
            shard.get("provenance_index_sha256"), "provenance object"
        ),
        "sequence": _positive_int(
            shard.get("sequence"), "production shard sequence", allow_zero=True
        ),
        "shard_id": str(shard.get("shard_id")),
        "text_bytes": _positive_int(
            shard.get("accepted_text_utf8_bytes"), "accepted UTF-8 bytes"
        ),
        "token_bytes": token_bytes,
        "token_object_key": _safe_key(
            shard.get("token_object_key"), namespace=namespace, label="token object"
        ),
        "token_sha256": _sha256(shard.get("token_object_sha256"), "token object"),
        "tokens": tokens,
    }


def _validate_writer_result(
    result: Mapping[str, Any],
    *,
    category_row: Mapping[str, Any],
    plan: Mapping[str, Any],
    accepted_checkpoint: Mapping[str, Any],
    authorization: Mapping[str, Any],
    dedup_result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    category = str(category_row["category"])
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported shard-writer result schema")
    if result.get("gate") != "lumi_emr_production_shard_writer_result_v0.1":
        raise ValueError("unexpected shard-writer result gate")
    if result.get("status") != "complete" or result.get("category") != category:
        raise ValueError("shard-writer result is incomplete or miscategorized")
    if result.get("plan_fingerprint") != plan["plan_fingerprint"]:
        raise ValueError("shard-writer result targets another plan")
    if result.get("authorization_run_token") != authorization["run_token"]:
        raise ValueError("shard-writer authorization changed")
    if result.get("dedup_result_sha256") != dedup_result_fingerprint(dedup_result):
        raise ValueError("shard-writer global-dedup result changed")
    if result.get("job_token") != shard_writer_job_token(
        authorization, dedup_result, category
    ):
        raise ValueError("shard-writer job token changed")
    selected_tokens = _positive_int(
        result.get("selected_tokens"), "selected tokens", allow_zero=True
    )
    selected_documents = _positive_int(
        result.get("selected_documents"), "selected documents", allow_zero=True
    )
    if selected_tokens != int(category_row["selected_tokens"]):
        raise ValueError("shard-writer selected token total changed")
    if selected_documents != int(category_row["selected_documents"]):
        raise ValueError("shard-writer selected document total changed")
    partial = accepted_checkpoint.get("partial_shard")
    expected_carried_tokens = 0
    expected_carried_documents = 0
    if isinstance(partial, Mapping) and partial.get("category") == category:
        expected_carried_tokens = int(partial["tokens"])
        expected_carried_documents = int(partial["documents"])
    if result.get("carried_partial_tokens") != expected_carried_tokens:
        raise ValueError("shard-writer carried partial token total changed")
    if result.get("carried_partial_documents") != expected_carried_documents:
        raise ValueError("shard-writer carried partial document total changed")
    shards = result.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("shard-writer produced no sealed shards")
    maximum_tokens = int(plan["limits"]["clean_shard_target_tokens"]) + int(
        plan["limits"]["maximum_category_overshoot_tokens"]
    )
    normalized = [
        _normalize_production_shard(
            shard,
            category=category,
            namespace=str(plan["storage"]["namespace"]),
            maximum_tokens=maximum_tokens,
        )
        for shard in shards
    ]
    if sum(row["tokens"] for row in normalized) != (
        selected_tokens + expected_carried_tokens
    ):
        raise ValueError("shard-writer physical token conservation failed")
    if sum(row["documents"] for row in normalized) != (
        selected_documents + expected_carried_documents
    ):
        raise ValueError("shard-writer physical document conservation failed")
    first_sequence = int(partial["sequence"]) if expected_carried_tokens else 0
    sequences = [row["sequence"] for row in normalized]
    if sequences != list(range(first_sequence, first_sequence + len(normalized))):
        raise ValueError("shard-writer sequence is not contiguous")
    if result.get("next_sequence") != first_sequence + len(normalized):
        raise ValueError("shard-writer next sequence changed")
    return normalized


def final_corpus_seal_fingerprint(seal: Mapping[str, Any]) -> str:
    material = dict(seal)
    material.pop("seal_sha256", None)
    return _sha256_bytes(_canonical_json_bytes(material))


def build_final_corpus_seal(
    plan: Mapping[str, Any],
    accepted_checkpoint: Mapping[str, Any],
    authorization: Mapping[str, Any],
    dedup_result: Mapping[str, Any],
    shard_writer_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind every selected document to one immutable, conservation-checked seal."""

    validate_production_checkpoint(accepted_checkpoint, plan)
    validate_paid_production_authorization(authorization, plan, accepted_checkpoint)
    validate_global_dedup_result(dedup_result, plan, authorization)
    category_results = {str(row["category"]): row for row in dedup_result["categories"]}
    writers: dict[str, Mapping[str, Any]] = {}
    for result in shard_writer_results:
        if not isinstance(result, Mapping):
            raise ValueError("shard-writer result is invalid")
        category = str(result.get("category"))
        if category in writers:
            raise ValueError("final seal repeats a shard-writer category")
        writers[category] = result
    if set(writers) != set(category_results):
        raise ValueError("final seal is missing one or more shard-writer categories")

    namespace = str(plan["storage"]["namespace"])
    baseline = [
        _normalize_baseline_shard(row, namespace=namespace)
        for row in accepted_checkpoint["sealed_shards"]
    ]
    production: list[dict[str, Any]] = []
    for plan_category in plan["categories"]:
        category = str(plan_category["category"])
        production.extend(
            _validate_writer_result(
                writers[category],
                category_row=category_results[category],
                plan=plan,
                accepted_checkpoint=accepted_checkpoint,
                authorization=authorization,
                dedup_result=dedup_result,
            )
        )
    shards = [*baseline, *production]
    manifest_hashes: set[str] = set()
    object_keys: set[str] = set()
    for shard in shards:
        if shard["manifest_sha256"] in manifest_hashes:
            raise ValueError("final corpus repeats a shard manifest")
        manifest_hashes.add(shard["manifest_sha256"])
        for name in (
            "clean_object_key",
            "manifest_object_key",
            "provenance_object_key",
            "token_object_key",
        ):
            key = shard[name]
            if key in object_keys:
                raise ValueError("final corpus repeats an S3 object key")
            object_keys.add(key)

    total_tokens = sum(int(row["tokens"]) for row in shards)
    total_documents = sum(int(row["documents"]) for row in shards)
    expected_tokens = sum(int(row["final_tokens"]) for row in category_results.values())
    expected_documents = sum(
        int(row["final_documents"]) for row in category_results.values()
    )
    if total_tokens != expected_tokens:
        raise ValueError("final corpus physical token conservation failed")
    if total_documents != expected_documents:
        raise ValueError("final corpus physical document conservation failed")
    if total_tokens < int(plan["limits"]["candidate_total_floor_tokens"]):
        raise ValueError("final corpus remains below the 120B candidate floor")
    if total_tokens > int(plan["limits"]["candidate_hard_maximum_tokens"]):
        raise ValueError("final corpus exceeds the 150B hard token cap")
    production_bytes = sum(
        int(row["clean_bytes"])
        + int(row["token_bytes"])
        + int(row["provenance_bytes"])
        for row in production
    )
    conservative_logical_bytes = int(
        accepted_checkpoint["logical_storage_bytes"]
    ) + production_bytes
    if conservative_logical_bytes > int(plan["limits"]["hard_logical_storage_bytes"]):
        raise ValueError("final corpus exceeds the 2TB logical storage cap")
    categories = [
        {
            "accepted_documents": int(category_results[row["category"]]["final_documents"]),
            "accepted_tokens": int(category_results[row["category"]]["final_tokens"]),
            "category": row["category"],
            "floor_tokens": int(row["floor_tokens"]),
        }
        for row in plan["categories"]
    ]
    seal = {
        "accepted_root_checkpoint_sha256": accepted_checkpoint["checkpoint_sha256"],
        "authorization_sha256": _sha256_bytes(_canonical_json_bytes(authorization)),
        "authorization_run_token": authorization["run_token"],
        "bucket": plan["storage"]["bucket"],
        "candidate_hard_maximum_tokens": int(
            plan["limits"]["candidate_hard_maximum_tokens"]
        ),
        "candidate_total_floor_tokens": int(
            plan["limits"]["candidate_total_floor_tokens"]
        ),
        "categories": categories,
        "conservative_logical_storage_bytes": conservative_logical_bytes,
        "dedup_result_sha256": dedup_result_fingerprint(dedup_result),
        "final_training_target_tokens": int(plan["limits"]["final_training_tokens"]),
        "gate": "lumi_emr_final_candidate_corpus_seal_v0.1",
        "namespace": namespace,
        "plan_fingerprint": plan["plan_fingerprint"],
        "schema_version": SCHEMA_VERSION,
        "shards": shards,
        "source_commit": plan["source_commit"],
        "status": "sealed_ready_for_80b_sampling",
        "total_accepted_documents": total_documents,
        "total_accepted_tokens": total_tokens,
    }
    seal["seal_sha256"] = final_corpus_seal_fingerprint(seal)
    return seal


def write_final_corpus_seal(seal: Mapping[str, Any], directory: Path) -> dict[str, Any]:
    """Write canonical seal bytes plus a standard local SHA-256 manifest."""

    if seal.get("seal_sha256") != final_corpus_seal_fingerprint(seal):
        raise ValueError("final corpus seal SHA-256 changed")
    directory.mkdir(parents=True, exist_ok=False)
    seal_path = directory / "final-candidate-corpus.json"
    seal_path.write_bytes(_canonical_json_bytes(seal))
    digest = _sha256_bytes(seal_path.read_bytes())
    manifest_path = directory / "manifest.sha256"
    manifest_path.write_text(f"{digest}  {seal_path.name}\n", encoding="utf-8")
    return {
        "manifest_path": str(manifest_path),
        "seal_path": str(seal_path),
        "seal_sha256": digest,
    }
