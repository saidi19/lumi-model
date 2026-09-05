"""Deterministic message-native human authoring packets for Lumi SFT."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .posttraining_authoring_workflow import AuthoringPacket
from .posttraining_first_party import canonical_json_bytes, file_sha256
from .posttraining_first_party_v2 import audit_first_party_authoring_rows_v2

WORKFLOW_SCHEMA_VERSION = "lumi.first-party-authoring-workflow.v0.2"
MANIFEST_SCHEMA_VERSION = "lumi.first-party-authoring-workflow-manifest.v0.2"
EXPECTED_AUTHORIZATION = {
    "external_distribution_authorized": False,
    "human_authoring_workflow_authorized": True,
    "model_generated_training_text_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "raw_training_text_in_git": False,
}
EXPECTED_AUTHOR_SLOTS = [f"author-slot-{index:02d}" for index in range(1, 5)]
EXPECTED_REVIEWER_SLOTS = [f"reviewer-slot-{index:02d}" for index in range(1, 5)]


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_authoring_workflow_v2_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Validate the v0.2 private workflow and its exact authoring-plan binding."""

    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or value.get("schema_version") != WORKFLOW_SCHEMA_VERSION:
        raise ValueError("unsupported first-party v0.2 workflow")
    if value.get("status") != (
        "message_native_blank_human_packets_ready_identity_assignment_pending"
    ):
        raise ValueError("first-party v0.2 workflow status changed")
    if value.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("first-party v0.2 workflow authorization changed")
    binding = value.get("authoring_plan_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("first-party v0.2 workflow binding is missing")
    if binding.get("path") != "configs/lumi-4b-first-party-authoring-pilot-v0.2.json":
        raise ValueError("first-party v0.2 workflow plan path changed")
    if file_sha256(Path(repository_root) / str(binding["path"])) != binding.get(
        "sha256"
    ):
        raise ValueError("first-party v0.2 workflow binding changed")

    workflow = value.get("workflow")
    if not isinstance(workflow, Mapping):
        raise ValueError("first-party v0.2 workflow configuration is missing")
    if workflow.get("assignment_strategy") != (
        "skill_language_sorted_round_robin_per_split_v0.2"
    ):
        raise ValueError("first-party v0.2 assignment strategy changed")
    if workflow.get("author_slots") != EXPECTED_AUTHOR_SLOTS:
        raise ValueError("first-party v0.2 author slots changed")
    if workflow.get("reviewer_slots") != EXPECTED_REVIEWER_SLOTS:
        raise ValueError("first-party v0.2 reviewer slots changed")
    if workflow.get("packet_count") != 16 or workflow.get("rows_per_packet") != 20:
        raise ValueError("first-party v0.2 packet bounds changed")
    if workflow.get("minimum_distinct_humans") != 2:
        raise ValueError("first-party v0.2 minimum human count changed")
    if workflow.get("author_and_reviewer_human_identity_must_differ") is not True:
        raise ValueError("first-party v0.2 independent review is required")
    if workflow.get("slot_names_are_not_human_identity_evidence") is not True:
        raise ValueError("first-party v0.2 slot identity boundary changed")
    if value.get("storage_policy") != {
        "completed_rows_must_remain_outside_git": True,
        "private_encrypted_storage_required": True,
        "raw_text_allowed_in_progress_reports": False,
    }:
        raise ValueError("first-party v0.2 storage policy changed")
    return value


def build_authoring_packets_v2(
    *,
    authoring_plan: Mapping[str, Any],
    coverage: Mapping[str, Any],
    blank_rows: list[Mapping[str, Any]],
    workflow_plan: Mapping[str, Any],
) -> tuple[AuthoringPacket, ...]:
    """Build balanced v0.2 packets only from the exact blank template."""

    audit = audit_first_party_authoring_rows_v2(
        blank_rows,
        plan=authoring_plan,
        coverage=coverage,
    )
    if audit["status_counts"] != {"accepted": 0, "blank": 320, "invalid": 0}:
        raise ValueError("v0.2 packets require the exact blank template")
    workflow = workflow_plan["workflow"]
    packet_count = int(workflow["packet_count"])
    packet_rows: list[list[dict[str, Any]]] = [[] for _ in range(packet_count)]
    for split_index, split in enumerate(("train", "validation")):
        ordered = sorted(
            (row for row in blank_rows if row["split"] == split),
            key=lambda row: (row["skill_id"], row["language"], row["record_id"]),
        )
        offset = split_index * (packet_count // 2)
        for index, row in enumerate(ordered):
            packet_rows[(index + offset) % packet_count].append(
                json.loads(json.dumps(row, ensure_ascii=False))
            )
    packets = tuple(
        AuthoringPacket(
            author_slot=workflow["author_slots"][
                index % len(workflow["author_slots"])
            ],
            packet_id=f"authoring-v2-packet-{index + 1:02d}",
            reviewer_slot=workflow["reviewer_slots"][
                (index + 1) % len(workflow["reviewer_slots"])
            ],
            rows=tuple(rows),
        )
        for index, rows in enumerate(packet_rows)
    )
    if any(len(packet.rows) != 20 for packet in packets):
        raise AssertionError("first-party v0.2 packet row balance changed")
    if len({row["record_id"] for packet in packets for row in packet.rows}) != 320:
        raise AssertionError("first-party v0.2 packet membership changed")
    return packets


def build_authoring_workflow_v2_manifest(
    *,
    authoring_plan: Mapping[str, Any],
    packets: tuple[AuthoringPacket, ...],
    workflow_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a raw-text-free v0.2 private packet inventory."""

    packet_entries = []
    for packet in packets:
        packet_entries.append(
            {
                "author_slot": packet.author_slot,
                "jsonl_sha256": hashlib.sha256(packet.jsonl_payload).hexdigest(),
                "language_counts": dict(
                    sorted(Counter(row["language"] for row in packet.rows).items())
                ),
                "packet_id": packet.packet_id,
                "record_ids": sorted(row["record_id"] for row in packet.rows),
                "reviewer_slot": packet.reviewer_slot,
                "rows": len(packet.rows),
                "skill_counts": dict(
                    sorted(Counter(row["skill_id"] for row in packet.rows).items())
                ),
                "split_counts": dict(
                    sorted(Counter(row["split"] for row in packet.rows).items())
                ),
            }
        )
    manifest: dict[str, Any] = {
        "authorization": {
            "model_generated_training_text_authorized": False,
            "paid_gpu_authorized": False,
            "production_sft_authorized": False,
        },
        "authoring_plan_canonical_sha256": _fingerprint(authoring_plan),
        "human_identity_assignment_complete": False,
        "minimum_distinct_humans": 2,
        "packet_count": len(packets),
        "packets": packet_entries,
        "raw_training_text_retained": False,
        "records": sum(len(packet.rows) for packet in packets),
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "message_native_blank_packets_waiting_for_human_identity_assignment",
        "workflow_plan_canonical_sha256": _fingerprint(workflow_plan),
    }
    manifest["manifest_sha256"] = _fingerprint(manifest)
    return manifest


def validate_authoring_workflow_v2_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the raw-text-free v0.2 packet inventory and its digest."""

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported first-party v0.2 workflow manifest")
    if manifest.get("authorization") != {
        "model_generated_training_text_authorized": False,
        "paid_gpu_authorized": False,
        "production_sft_authorized": False,
    }:
        raise ValueError("first-party v0.2 manifest authorization changed")
    if manifest.get("status") != (
        "message_native_blank_packets_waiting_for_human_identity_assignment"
    ):
        raise ValueError("first-party v0.2 manifest status changed")
    if manifest.get("raw_training_text_retained") is not False:
        raise ValueError("first-party v0.2 manifest may not retain raw text")
    if manifest.get("human_identity_assignment_complete") is not False:
        raise ValueError("first-party v0.2 manifest cannot claim human identities")
    if manifest.get("minimum_distinct_humans") != 2:
        raise ValueError("first-party v0.2 minimum human count changed")

    entries = manifest.get("packets")
    if not isinstance(entries, list) or len(entries) != 16:
        raise ValueError("first-party v0.2 manifest packet count changed")
    packet_ids: list[str] = []
    record_ids: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("first-party v0.2 manifest packet entry is invalid")
        packet_id = entry.get("packet_id")
        if not isinstance(packet_id, str):
            raise ValueError("first-party v0.2 manifest packet ID is invalid")
        packet_ids.append(packet_id)
        if entry.get("author_slot") not in EXPECTED_AUTHOR_SLOTS:
            raise ValueError("first-party v0.2 manifest author slot changed")
        if entry.get("reviewer_slot") not in EXPECTED_REVIEWER_SLOTS:
            raise ValueError("first-party v0.2 manifest reviewer slot changed")
        if entry.get("author_slot", "").split("-")[-1] == entry.get(
            "reviewer_slot", ""
        ).split("-")[-1]:
            raise ValueError("first-party v0.2 manifest review slot is not independent")
        entry_ids = entry.get("record_ids")
        if not isinstance(entry_ids, list) or len(entry_ids) != 20:
            raise ValueError("first-party v0.2 manifest packet membership changed")
        if any(not isinstance(record_id, str) for record_id in entry_ids):
            raise ValueError("first-party v0.2 manifest record ID is invalid")
        if entry.get("rows") != 20 or entry.get("split_counts") != {
            "train": 15,
            "validation": 5,
        }:
            raise ValueError("first-party v0.2 manifest packet balance changed")
        digest = entry.get("jsonl_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("first-party v0.2 manifest packet digest is invalid")
        record_ids.extend(entry_ids)
    if len(set(packet_ids)) != 16:
        raise ValueError("first-party v0.2 manifest packet IDs are duplicated")
    if len(record_ids) != 320 or len(set(record_ids)) != 320:
        raise ValueError("first-party v0.2 manifest record membership is invalid")
    if manifest.get("packet_count") != 16 or manifest.get("records") != 320:
        raise ValueError("first-party v0.2 manifest totals changed")

    expected_sha256 = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if expected_sha256 != _fingerprint(unsigned):
        raise ValueError("first-party v0.2 manifest SHA-256 changed")
    return json.loads(json.dumps(manifest, ensure_ascii=False))


def validate_packet_rows_against_v2_manifest(
    *,
    manifest: Mapping[str, Any],
    packet_id: str,
    rows: list[Mapping[str, Any]],
) -> None:
    """Allow human edits only when a packet keeps its exact record membership."""

    validated = validate_authoring_workflow_v2_manifest(manifest)
    matches = [
        entry for entry in validated["packets"] if entry.get("packet_id") == packet_id
    ]
    if len(matches) != 1:
        raise ValueError("first-party v0.2 packet is absent or duplicated")
    expected_ids = matches[0]["record_ids"]
    observed_ids = [row.get("record_id") for row in rows]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("first-party v0.2 packet contains duplicate record IDs")
    if set(observed_ids) != set(expected_ids):
        raise ValueError("first-party v0.2 packet membership changed")
