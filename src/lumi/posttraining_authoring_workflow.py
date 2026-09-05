"""Deterministic private work packets for Lumi-owned human SFT authoring."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .posttraining_first_party import (
    audit_first_party_authoring_rows,
    canonical_json_bytes,
    file_sha256,
)

AUTHORING_WORKFLOW_SCHEMA_VERSION = "lumi.first-party-authoring-workflow.v0.1"
AUTHORING_WORKFLOW_MANIFEST_SCHEMA_VERSION = (
    "lumi.first-party-authoring-workflow-manifest.v0.1"
)
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


def _jsonl_payload(rows: tuple[dict[str, Any], ...]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


@dataclass(frozen=True)
class AuthoringPacket:
    author_slot: str
    packet_id: str
    reviewer_slot: str
    rows: tuple[dict[str, Any], ...]

    @property
    def jsonl_payload(self) -> bytes:
        return _jsonl_payload(self.rows)


def load_authoring_workflow_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Validate the non-authorizing packet workflow and its authoring-plan binding."""

    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("authoring workflow plan must be an object")
    if value.get("schema_version") != AUTHORING_WORKFLOW_SCHEMA_VERSION:
        raise ValueError("unsupported authoring workflow schema")
    if value.get("status") != "blank_human_authoring_packets_ready_no_authored_records":
        raise ValueError("authoring workflow status changed")
    if value.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("authoring workflow authorization boundary changed")

    binding = value.get("authoring_plan_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("authoring workflow binding is missing")
    if binding.get("path") != "configs/lumi-4b-first-party-authoring-pilot-v0.1.json":
        raise ValueError("authoring workflow plan path changed")
    if file_sha256(Path(repository_root) / binding["path"]) != binding.get("sha256"):
        raise ValueError("authoring workflow plan binding changed")

    workflow = value.get("workflow")
    if not isinstance(workflow, Mapping):
        raise ValueError("authoring workflow configuration is missing")
    if workflow.get("assignment_strategy") != (
        "skill_language_sorted_round_robin_per_split_v0.1"
    ):
        raise ValueError("authoring packet assignment strategy changed")
    if workflow.get("author_slots") != EXPECTED_AUTHOR_SLOTS:
        raise ValueError("authoring slots changed")
    if workflow.get("reviewer_slots") != EXPECTED_REVIEWER_SLOTS:
        raise ValueError("reviewer slots changed")
    if workflow.get("packet_count") != 16 or workflow.get("rows_per_packet") != 20:
        raise ValueError("authoring packet bounds changed")
    if workflow.get("author_and_reviewer_human_identity_must_differ") is not True:
        raise ValueError("independent human review is required")
    if workflow.get("slot_names_are_not_human_identity_evidence") is not True:
        raise ValueError("authoring slot identity boundary changed")

    storage = value.get("storage_policy")
    if storage != {
        "completed_rows_must_remain_outside_git": True,
        "private_encrypted_storage_required": True,
        "raw_text_allowed_in_progress_reports": False,
    }:
        raise ValueError("authoring workflow storage policy changed")
    return value


def build_authoring_packets(
    *,
    authoring_plan: Mapping[str, Any],
    blank_rows: list[Mapping[str, Any]],
    workflow_plan: Mapping[str, Any],
) -> tuple[AuthoringPacket, ...]:
    """Create balanced blank packets without generating or transforming text."""

    audit = audit_first_party_authoring_rows(blank_rows, plan=authoring_plan)
    if audit["status_counts"] != {
        "accepted": 0,
        "blank": 320,
        "in_progress": 0,
        "invalid": 0,
    }:
        raise ValueError("authoring packets may only be built from the exact blank template")

    workflow = workflow_plan["workflow"]
    packet_count = workflow["packet_count"]
    packet_rows: list[list[dict[str, Any]]] = [[] for _ in range(packet_count)]
    for split_index, split in enumerate(("train", "validation")):
        ordered_rows = sorted(
            (row for row in blank_rows if row["split"] == split),
            key=lambda row: (
                row["skill_id"],
                row["language"],
                row["record_id"],
            ),
        )
        offset = split_index * (packet_count // 2)
        for index, row in enumerate(ordered_rows):
            packet_rows[(index + offset) % packet_count].append(
                json.loads(json.dumps(row, ensure_ascii=False))
            )

    packets = tuple(
        AuthoringPacket(
            author_slot=workflow["author_slots"][index % len(workflow["author_slots"])],
            packet_id=f"authoring-packet-{index + 1:02d}",
            reviewer_slot=workflow["reviewer_slots"][
                (index + 1) % len(workflow["reviewer_slots"])
            ],
            rows=tuple(rows),
        )
        for index, rows in enumerate(packet_rows)
    )
    if any(len(packet.rows) != workflow["rows_per_packet"] for packet in packets):
        raise AssertionError("authoring packet row balance changed")
    if {row["record_id"] for packet in packets for row in packet.rows} != {
        row["record_id"] for row in blank_rows
    }:
        raise AssertionError("authoring packet membership changed")
    return packets


def build_authoring_workflow_manifest(
    *,
    authoring_plan: Mapping[str, Any],
    packets: tuple[AuthoringPacket, ...],
    workflow_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a raw-text-free packet inventory for private human operations."""

    packet_entries = []
    all_rows: list[dict[str, Any]] = []
    for packet in packets:
        all_rows.extend(packet.rows)
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
        "packet_count": len(packets),
        "packets": packet_entries,
        "raw_training_text_retained": False,
        "records": len(all_rows),
        "schema_version": AUTHORING_WORKFLOW_MANIFEST_SCHEMA_VERSION,
        "status": "blank_packets_ready_waiting_for_human_identity_assignment",
        "workflow_plan_canonical_sha256": _fingerprint(workflow_plan),
    }
    manifest["manifest_sha256"] = _fingerprint(manifest)
    return manifest


def validate_authoring_workflow_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Verify packet inventory integrity without accepting any authored text."""

    if manifest.get("schema_version") != AUTHORING_WORKFLOW_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported authoring workflow manifest")
    if manifest.get("authorization") != {
        "model_generated_training_text_authorized": False,
        "paid_gpu_authorized": False,
        "production_sft_authorized": False,
    }:
        raise ValueError("authoring workflow manifest authorization changed")
    if manifest.get("status") != (
        "blank_packets_ready_waiting_for_human_identity_assignment"
    ):
        raise ValueError("authoring workflow manifest status changed")
    if manifest.get("raw_training_text_retained") is not False:
        raise ValueError("authoring workflow manifest may not retain raw text")
    if manifest.get("human_identity_assignment_complete") is not False:
        raise ValueError("slot assignment may not claim completed human identities")

    entries = manifest.get("packets")
    if not isinstance(entries, list) or len(entries) != 16:
        raise ValueError("authoring workflow manifest packet count changed")
    packet_ids = [entry.get("packet_id") for entry in entries if isinstance(entry, Mapping)]
    if len(packet_ids) != len(entries) or len(set(packet_ids)) != len(packet_ids):
        raise ValueError("authoring workflow manifest packet IDs are invalid")
    record_ids = [
        record_id
        for entry in entries
        for record_id in entry.get("record_ids", [])
    ]
    if len(record_ids) != 320 or len(set(record_ids)) != 320:
        raise ValueError("authoring workflow manifest record membership is invalid")
    if manifest.get("packet_count") != 16 or manifest.get("records") != 320:
        raise ValueError("authoring workflow manifest totals changed")

    expected_sha256 = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if expected_sha256 != _fingerprint(unsigned):
        raise ValueError("authoring workflow manifest SHA-256 changed")
    return json.loads(json.dumps(manifest, ensure_ascii=False))


def validate_packet_rows_against_manifest(
    *,
    manifest: Mapping[str, Any],
    packet_id: str,
    rows: list[Mapping[str, Any]],
) -> None:
    """Require exact packet membership while allowing humans to fill text/evidence fields."""

    validated_manifest = validate_authoring_workflow_manifest(manifest)
    entries = validated_manifest["packets"]
    matches = [entry for entry in entries if entry.get("packet_id") == packet_id]
    if len(matches) != 1:
        raise ValueError("authoring packet is absent or duplicated in manifest")
    expected_ids = matches[0].get("record_ids")
    observed_ids = [row.get("record_id") for row in rows]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("authoring packet contains duplicate record IDs")
    if set(observed_ids) != set(expected_ids):
        raise ValueError("authoring packet membership changed")
