from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.posttraining_authoring_workflow import (
    build_authoring_packets,
    build_authoring_workflow_manifest,
    load_authoring_workflow_plan,
    validate_authoring_workflow_manifest,
    validate_packet_rows_against_manifest,
)
from lumi.posttraining_first_party import (
    build_blank_authoring_template,
    load_first_party_authoring_plan,
)

ROOT = Path(__file__).parents[1]
AUTHORING_PLAN_PATH = ROOT / "configs" / "lumi-4b-first-party-authoring-pilot-v0.1.json"
WORKFLOW_PLAN_PATH = (
    ROOT / "configs" / "lumi-4b-first-party-authoring-workflow-v0.1.json"
)


def _build():
    authoring = load_first_party_authoring_plan(AUTHORING_PLAN_PATH)
    workflow = load_authoring_workflow_plan(WORKFLOW_PLAN_PATH, repository_root=ROOT)
    packets = build_authoring_packets(
        authoring_plan=authoring,
        blank_rows=build_blank_authoring_template(authoring),
        workflow_plan=workflow,
    )
    return authoring, workflow, packets


def test_authoring_workflow_balances_exact_blank_rows_without_text_manifest() -> None:
    authoring, workflow, packets = _build()
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
    assert all(
        packet.author_slot.split("-")[-1] != packet.reviewer_slot.split("-")[-1]
        for packet in packets
    )

    manifest = build_authoring_workflow_manifest(
        authoring_plan=authoring,
        packets=packets,
        workflow_plan=workflow,
    )
    assert manifest["records"] == 320
    assert manifest["packet_count"] == 16
    assert manifest["human_identity_assignment_complete"] is False
    assert manifest["raw_training_text_retained"] is False
    assert "user_message" not in json.dumps(manifest)
    assert "assistant_response" not in json.dumps(manifest)
    assert validate_authoring_workflow_manifest(manifest) == manifest


def test_authoring_packets_require_exact_membership_but_allow_human_progress() -> None:
    authoring, workflow, packets = _build()
    manifest = build_authoring_workflow_manifest(
        authoring_plan=authoring,
        packets=packets,
        workflow_plan=workflow,
    )
    rows = [copy.deepcopy(row) for row in packets[0].rows]
    rows[0]["user_message"] = "Human work may be filled after packet creation."
    validate_packet_rows_against_manifest(
        manifest=manifest,
        packet_id=packets[0].packet_id,
        rows=rows,
    )

    with pytest.raises(ValueError, match="membership changed"):
        validate_packet_rows_against_manifest(
            manifest=manifest,
            packet_id=packets[0].packet_id,
            rows=rows[:-1],
        )


def test_authoring_workflow_rejects_nonblank_source_or_authorization_drift(
    tmp_path: Path,
) -> None:
    authoring, workflow, _ = _build()
    rows = build_blank_authoring_template(authoring)
    rows[0]["user_message"] = "Already authored"
    with pytest.raises(ValueError, match="exact blank template"):
        build_authoring_packets(
            authoring_plan=authoring,
            blank_rows=rows,
            workflow_plan=workflow,
        )

    changed = copy.deepcopy(workflow)
    changed["authorization"]["model_generated_training_text_authorized"] = True
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization boundary"):
        load_authoring_workflow_plan(path, repository_root=ROOT)


def test_authoring_workflow_manifest_rejects_membership_or_hash_drift() -> None:
    authoring, workflow, packets = _build()
    manifest = build_authoring_workflow_manifest(
        authoring_plan=authoring,
        packets=packets,
        workflow_plan=workflow,
    )
    changed = copy.deepcopy(manifest)
    changed["packets"][0]["record_ids"].pop()
    with pytest.raises(ValueError, match="record membership"):
        validate_authoring_workflow_manifest(changed)

    changed = copy.deepcopy(manifest)
    changed["packets"][0]["author_slot"] = "changed"
    with pytest.raises(ValueError, match="SHA-256 changed"):
        validate_authoring_workflow_manifest(changed)
