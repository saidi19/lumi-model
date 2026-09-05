"""Append-only registry-v0.18 projection for approved existing candidates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PREDECESSOR_SHA256 = "85a688a62be0902b177469c99eb31525fb633efab5e420416b705d03e8d80e1a"
OWNER_RECEIPT_SHA256 = "772f891805ac00b080fe1ab2af82fb020302837251a8848a3138d1f807df7ce0"

ADDITIONS: tuple[dict[str, Any], ...] = (
    {
        "access": "public",
        "adapter": "existing_candidate_salvage_aya_mit_owner_bound_v0.18",
        "blockers": [],
        "dataset_id": "CohereLabs/aya_collection_language_split",
        "evidence_urls": [
            "https://huggingface.co/datasets/CohereLabs/aya_collection_language_split/tree/a3af2fde4b4cb5b2775830b11244a1a20b5f004f",
        ],
        "languages": ["bn"],
        "license_id": "apache-2.0",
        "origin": "human",
        "production_blockers": [
            "only_the_exact_673_row_constituent_mit_subset_is_authorized",
            "full_registry_v0_18_cross_source_hygiene_audit_not_complete",
            "production_dataset_seal_and_sft_authorization_not_complete",
        ],
        "production_eligible": False,
        "repository_url": "https://huggingface.co/datasets/CohereLabs/aya_collection_language_split",
        "revision": "a3af2fde4b4cb5b2775830b11244a1a20b5f004f",
        "source_id": "coherelabs_aya_bengali_closure_canary",
        "splits": ["train", "validation"],
        "status": "canary_eligible",
        "task_categories": ["multilingual_bengali"],
    },
    {
        "access": "public",
        "adapter": "existing_candidate_salvage_nemotron_cap_safe_v0.18",
        "blockers": [],
        "dataset_id": "nvidia/Nemotron-Post-Training-Dataset-v1",
        "evidence_urls": [
            "https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1/tree/74e23eb6f830fef4a9e96a92f6f6262214cbb9a8",
        ],
        "languages": ["en"],
        "license_id": "cc-by-4.0",
        "origin": "synthetic",
        "production_blockers": [
            "only_the_exact_owner_sample_preserving_1228_row_subset_is_authorized",
            "full_registry_v0_18_cross_source_hygiene_audit_not_complete",
            "production_dataset_seal_and_sft_authorization_not_complete",
        ],
        "production_eligible": False,
        "repository_url": "https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1",
        "revision": "74e23eb6f830fef4a9e96a92f6f6262214cbb9a8",
        "source_id": "nvidia_nemotron_post_training_tool_canary",
        "splits": ["train", "validation"],
        "status": "canary_eligible",
        "task_categories": ["tool_api"],
    },
    {
        "access": "public",
        "adapter": "banglabridge_hygiene_repair_byte_exact_v0.18",
        "blockers": [],
        "dataset_id": "subhajitmahata84/banglabridge-instructions",
        "evidence_urls": [
            "https://huggingface.co/datasets/subhajitmahata84/banglabridge-instructions/tree/b5ace9b2e56702cfd3aa5fe2e8c3870f67f57300",
        ],
        "languages": ["bn", "multilingual"],
        "license_id": "cc-by-4.0",
        "origin": "human",
        "production_blockers": [
            "full_registry_v0_18_cross_source_hygiene_audit_not_complete",
            "production_dataset_seal_and_sft_authorization_not_complete",
        ],
        "production_eligible": False,
        "repository_url": "https://huggingface.co/datasets/subhajitmahata84/banglabridge-instructions",
        "revision": "b5ace9b2e56702cfd3aa5fe2e8c3870f67f57300",
        "source_id": "banglabridge_instructions_closure_canary",
        "splits": ["train", "validation"],
        "status": "canary_eligible",
        "task_categories": ["multilingual_bengali"],
    },
    {
        "access": "public",
        "adapter": "banglatlit_auxiliary_byte_exact_v0.18",
        "blockers": [],
        "dataset_id": "aplycaebous/BanglaTLit",
        "evidence_urls": [
            "https://huggingface.co/datasets/aplycaebous/BanglaTLit/tree/22cacbaac72bf6ee828f9d6be1e28bfd7a84843f",
        ],
        "languages": ["multilingual"],
        "license_id": "mit",
        "origin": "human",
        "production_blockers": [
            "auxiliary_transliteration_only_zero_behavior_floor_credit",
            "full_registry_v0_18_cross_source_hygiene_audit_not_complete",
            "production_dataset_seal_and_sft_authorization_not_complete",
        ],
        "production_eligible": False,
        "repository_url": "https://huggingface.co/datasets/aplycaebous/BanglaTLit",
        "revision": "22cacbaac72bf6ee828f9d6be1e28bfd7a84843f",
        "source_id": "banglatlit_auxiliary_closure_canary",
        "splits": ["train", "validation"],
        "status": "canary_eligible",
        "task_categories": ["multilingual_bengali"],
    },
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_value(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def build_registry(predecessor_path: str | Path) -> dict[str, Any]:
    path = Path(predecessor_path)
    if file_sha256(path) != PREDECESSOR_SHA256:
        raise ValueError("registry v0.17 hash changed")
    predecessor = json.loads(path.read_text(encoding="utf-8"))
    if predecessor.get("schema_version") != "lumi.sft.sources.v0.1":
        raise ValueError("registry schema changed")
    sources = predecessor.get("sources")
    if not isinstance(sources, list) or len(sources) != 18:
        raise ValueError("registry v0.17 source inventory changed")
    existing_ids = {source.get("source_id") for source in sources}
    addition_ids = {source["source_id"] for source in ADDITIONS}
    if existing_ids & addition_ids:
        raise ValueError("registry v0.18 source already exists")
    return {
        "authorization": {
            "bulk_data_download_authorized": False,
            "paid_gpu_authorized": False,
            "production_sft_authorized": False,
            "purpose": "metadata_and_source_admissibility_audit_only",
        },
        "observed_at": "2026-08-13",
        "schema_version": predecessor["schema_version"],
        "sources": [*sources, *ADDITIONS],
    }


def build_receipt(
    *, predecessor_path: str | Path, successor_path: str | Path, owner_receipt_path: str | Path
) -> dict[str, Any]:
    predecessor = json.loads(Path(predecessor_path).read_text(encoding="utf-8"))
    successor = json.loads(Path(successor_path).read_text(encoding="utf-8"))
    if successor["sources"][:-len(ADDITIONS)] != predecessor["sources"]:
        raise ValueError("registry v0.17 sources were not preserved exactly")
    if file_sha256(owner_receipt_path) != OWNER_RECEIPT_SHA256:
        raise ValueError("Aya owner-review receipt changed")
    receipt: dict[str, Any] = {
        "authorization_boundary": {
            "dataset_seal_authorized": False,
            "gpu_or_network_used": False,
            "predecessor_mutated": False,
            "production_promotion_authorized": False,
            "sft_training_authorized": False,
        },
        "added_source_ids": [source["source_id"] for source in ADDITIONS],
        "owner_review_receipt_file_sha256": OWNER_RECEIPT_SHA256,
        "predecessor_registry_file_sha256": PREDECESSOR_SHA256,
        "predecessor_source_count": len(predecessor["sources"]),
        "production_eligible": False,
        "schema_version": "lumi.sft-existing-candidate-registry-projection-receipt.v0.18",
        "status": "PASS_immutable_registry_v0_18_projected_nonproduction",
        "successor_registry_file_sha256": file_sha256(successor_path),
        "successor_registry_internal_sha256": sha256_value(successor),
        "successor_source_count": len(successor["sources"]),
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    return receipt
