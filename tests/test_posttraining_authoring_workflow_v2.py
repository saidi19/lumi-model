from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from lumi.posttraining_authoring_workflow_v2 import (
    build_authoring_packets_v2,
    build_authoring_workflow_v2_manifest,
    load_authoring_workflow_v2_plan,
    validate_authoring_workflow_v2_manifest,
    validate_packet_rows_against_v2_manifest,
)
from lumi.posttraining_first_party_v2 import (
    build_blank_authoring_template_v2,
    load_first_party_authoring_v2_plan,
)

ROOT = Path(__file__).resolve().parents[1]
AUTHORING_PLAN_PATH = ROOT / "configs/lumi-4b-first-party-authoring-pilot-v0.2.json"
WORKFLOW_PLAN_PATH = ROOT / "configs/lumi-4b-first-party-authoring-workflow-v0.2.json"


def _build() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    tuple,
]:
    authoring, coverage = load_first_party_authoring_v2_plan(
        AUTHORING_PLAN_PATH,
        repository_root=ROOT,
    )
    workflow = load_authoring_workflow_v2_plan(
        WORKFLOW_PLAN_PATH,
        repository_root=ROOT,
    )
    packets = build_authoring_packets_v2(
        authoring_plan=authoring,
        coverage=coverage,
        blank_rows=build_blank_authoring_template_v2(authoring, coverage),
        workflow_plan=workflow,
    )
    return authoring, coverage, workflow, packets


def test_v2_workflow_builds_exact_message_native_packets() -> None:
    authoring, _, workflow, packets = _build()
    assert len(packets) == 16
    assert {len(packet.rows) for packet in packets} == {20}
    assert {
        (
            sum(row["split"] == "train" for row in packet.rows),
            sum(row["split"] == "validation" for row in packet.rows),
        )
        for packet in packets
    } == {(15, 5)}
    assert len({row["record_id"] for packet in packets for row in packet.rows}) == 320
    assert all(row["messages"] == [] for packet in packets for row in packet.rows)

    manifest = build_authoring_workflow_v2_manifest(
        authoring_plan=authoring,
        packets=packets,
        workflow_plan=workflow,
    )
    assert manifest["records"] == 320
    assert manifest["human_identity_assignment_complete"] is False
    assert manifest["raw_training_text_retained"] is False
    assert "messages" not in json.dumps(manifest)
    assert validate_authoring_workflow_v2_manifest(manifest) == manifest


def test_v2_packet_membership_allows_only_content_field_progress() -> None:
    authoring, _, workflow, packets = _build()
    manifest = build_authoring_workflow_v2_manifest(
        authoring_plan=authoring,
        packets=packets,
        workflow_plan=workflow,
    )
    rows = [copy.deepcopy(row) for row in packets[0].rows]
    rows[0]["messages"] = [
        {"content": "Human work may be filled here.", "role": "user"}
    ]
    validate_packet_rows_against_v2_manifest(
        manifest=manifest,
        packet_id=packets[0].packet_id,
        rows=rows,
    )

    with pytest.raises(ValueError, match="membership changed"):
        validate_packet_rows_against_v2_manifest(
            manifest=manifest,
            packet_id=packets[0].packet_id,
            rows=rows[:-1],
        )


def test_v2_workflow_rejects_nonblank_source_and_model_authorization(
    tmp_path: Path,
) -> None:
    authoring, coverage, workflow, _ = _build()
    rows = build_blank_authoring_template_v2(authoring, coverage)
    rows[0]["messages"] = [{"content": "Already authored", "role": "user"}]
    with pytest.raises(ValueError, match="exact blank template"):
        build_authoring_packets_v2(
            authoring_plan=authoring,
            coverage=coverage,
            blank_rows=rows,
            workflow_plan=workflow,
        )

    changed = copy.deepcopy(workflow)
    changed["authorization"]["model_generated_training_text_authorized"] = True
    path = tmp_path / "changed-workflow.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization changed"):
        load_authoring_workflow_v2_plan(path, repository_root=ROOT)


def test_v2_manifest_rejects_membership_and_hash_drift() -> None:
    authoring, _, workflow, packets = _build()
    manifest = build_authoring_workflow_v2_manifest(
        authoring_plan=authoring,
        packets=packets,
        workflow_plan=workflow,
    )
    changed = copy.deepcopy(manifest)
    changed["packets"][0]["record_ids"].pop()
    with pytest.raises(ValueError, match="packet membership changed"):
        validate_authoring_workflow_v2_manifest(changed)

    changed = copy.deepcopy(manifest)
    changed["packets"][0]["author_slot"] = "changed"
    with pytest.raises(ValueError, match="author slot changed"):
        validate_authoring_workflow_v2_manifest(changed)
