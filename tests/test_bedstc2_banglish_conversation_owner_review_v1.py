from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

import lumi.posttraining_bedstc2_banglish_conversation_owner_review_v1 as owner

ROOT = Path(__file__).resolve().parents[1]


def _packet() -> list[dict[str, object]]:
    return [
        {
            "language_mode": "banglish",
            "messages": [
                {"content": f"synthetic context {index}", "role": "user"},
                {"content": f"synthetic answer {index}", "role": "assistant"},
            ],
            "record_id": f"bedstc2-banglish-{index:024x}",
            "skill_id": "conversation",
            "split": "train",
        }
        for index in range(32)
    ]


def _decisions(*, accepts: int = 32) -> dict[str, object]:
    packet = _packet()
    return {
        "schema_version": owner.DECISIONS_SCHEMA_VERSION,
        "sample_sha256": owner.sha256_value(packet),
        "reviewer_id": "owner-synthetic",
        "reviewed_at": owner.EXPECTED_REVIEWED_AT,
        "decisions": [
            {
                "accepted": index < accepts,
                "note": "",
                "record_id": row["record_id"],
            }
            for index, row in enumerate(packet)
        ],
    }


def _receipt() -> dict[str, object]:
    receipt: dict[str, object] = {
        "authorization": owner.EXPECTED_AUTHORIZATION,
        "bindings": {
            "authorization_sha256": owner.EXPECTED_AUTHORIZATION_SHA256,
            "curation_report_file_sha256": owner.EXPECTED_INPUTS["curation_report"][
                "sha256"
            ],
            "curation_report_internal_sha256": owner.EXPECTED_INPUTS[
                "curation_report"
            ]["canonical_sha256"],
            "curation_verification_file_sha256": owner.EXPECTED_INPUTS[
                "curation_verification"
            ]["sha256"],
            "curation_verification_internal_sha256": owner.EXPECTED_INPUTS[
                "curation_verification"
            ]["canonical_sha256"],
            "decisions_bytes": owner.EXPECTED_INPUTS["decisions"]["bytes"],
            "decisions_canonical_sha256": owner.EXPECTED_INPUTS["decisions"][
                "canonical_sha256"
            ],
            "decisions_file_sha256": owner.EXPECTED_INPUTS["decisions"]["sha256"],
            "owner_packet_bytes": owner.EXPECTED_INPUTS["owner_packet"]["bytes"],
            "owner_packet_file_sha256": owner.EXPECTED_INPUTS["owner_packet"][
                "sha256"
            ],
            "owner_packet_semantic_sha256": owner.EXPECTED_INPUTS["owner_packet"][
                "canonical_sha256"
            ],
            "plan_file_sha256": "a" * 64,
            "reviewed_record_ids_domain_sha256": owner.EXPECTED_REVIEWED_IDS_DOMAIN_SHA256,
        },
        "candidate": owner.EXPECTED_CANDIDATE,
        "decision_notes_retained": False,
        "evaluation_prohibition": owner.EXPECTED_EVALUATION_PROHIBITION,
        "human_owner_review_claim_self_attested": True,
        "next_gate": owner.EXPECTED_NEXT_GATE,
        "owner_review": {
            "accepted_records": 32,
            "all_sampled_rows_accepted": True,
            "automated_reviewer_markers_absent": True,
            "rejected_records": 0,
            "reviewed_at_claimed_utc": owner.EXPECTED_REVIEWED_AT,
            "sample_records": 32,
            "threshold": {
                "minimum_accepts": 29,
                "observed_accepts": 32,
                "satisfied": True,
            },
        },
        "owner_review_claim": owner.EXPECTED_OWNER_REVIEW_CLAIM,
        "production_eligible": False,
        "provenance": owner.EXPECTED_PROVENANCE,
        "raw_training_text_retained": False,
        "receipt_sha256": "",
        "record_ids_retained": False,
        "remaining_blockers": owner.EXPECTED_REMAINING_BLOCKERS,
        "reviewer_alias_retained": False,
        "rights": owner.EXPECTED_RIGHTS,
        "schema_version": owner.RECEIPT_SCHEMA_VERSION,
        "status": owner.RECEIPT_STATUS,
    }
    receipt["receipt_sha256"] = owner.sha256_value(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    return receipt


def _reseal_receipt(value: dict[str, object]) -> None:
    value["receipt_sha256"] = owner.sha256_value(
        {key: item for key, item in value.items() if key != "receipt_sha256"}
    )


def _reseal_verification(value: dict[str, object]) -> None:
    value["verification_sha256"] = owner.sha256_value(
        {key: item for key, item in value.items() if key != "verification_sha256"}
    )


def test_strict_json_rejects_duplicate_keys_and_nonfinite_numbers() -> None:
    with pytest.raises(ValueError):
        owner.strict_json_bytes(b'{"a":1,"a":2}')
    with pytest.raises(ValueError):
        owner.strict_json_bytes(b'{"a":NaN}')


def test_strict_json_rejects_unpaired_surrogate() -> None:
    with pytest.raises(ValueError):
        owner.strict_json_bytes(b'{"a":"\\ud800"}')


def test_binding_hash_domains_distinguish_full_canonical_and_internal_seal() -> None:
    packet = _packet()
    owner.validate_binding_hash_domain(
        "owner_packet",
        packet,
        {"canonical_sha256": owner.sha256_value(packet)},
    )
    unsigned = {"schema_version": "synthetic", "value": 1}
    sealed = {**unsigned, "report_sha256": owner.sha256_value(unsigned)}
    owner.validate_binding_hash_domain(
        "curation_report",
        sealed,
        {"canonical_sha256": sealed["report_sha256"]},
    )
    sealed["value"] = 2
    with pytest.raises(ValueError):
        owner.validate_binding_hash_domain(
            "curation_report",
            sealed,
            {"canonical_sha256": sealed["report_sha256"]},
        )


def test_synthetic_packet_contract_and_id_domain() -> None:
    packet = _packet()
    ids = owner._validate_packet_shape(packet)
    assert len(ids) == 32
    assert len(set(ids)) == 32
    assert len(owner._reviewed_ids_domain_sha256(ids)) == 64


def test_packet_rejects_wrong_role_or_duplicate_id() -> None:
    packet = _packet()
    packet[1]["record_id"] = packet[0]["record_id"]
    with pytest.raises(ValueError):
        owner._validate_packet_shape(packet)
    packet = _packet()
    packet[0]["messages"][0]["role"] = "assistant"  # type: ignore[index]
    with pytest.raises(ValueError):
        owner._validate_packet_shape(packet)


def test_decisions_accept_exact_29_row_threshold() -> None:
    packet = _packet()
    outcome = owner._validate_decisions_shape(
        _decisions(accepts=29),
        expected_record_ids=[str(row["record_id"]) for row in packet],
        expected_sample_sha256=owner.sha256_value(packet),
    )
    assert outcome["accepted"] == 29
    assert outcome["rejected"] == 3


def test_decisions_reject_28_row_threshold() -> None:
    packet = _packet()
    with pytest.raises(ValueError):
        owner._validate_decisions_shape(
            _decisions(accepts=28),
            expected_record_ids=[str(row["record_id"]) for row in packet],
            expected_sample_sha256=owner.sha256_value(packet),
        )


@pytest.mark.parametrize(
    "reviewer",
    ["", " owner", "openai-owner", "codex", "automated-reviewer", "💡owner"],
)
def test_decisions_reject_invalid_or_automated_alias(reviewer: str) -> None:
    packet = _packet()
    decisions = _decisions()
    decisions["reviewer_id"] = reviewer
    with pytest.raises(ValueError):
        owner._validate_decisions_shape(
            decisions,
            expected_record_ids=[str(row["record_id"]) for row in packet],
            expected_sample_sha256=owner.sha256_value(packet),
        )


def test_decisions_reject_invalid_calendar_timestamp() -> None:
    packet = _packet()
    decisions = _decisions()
    decisions["reviewed_at"] = "2026-02-31T10:00:00Z"
    with pytest.raises(ValueError):
        owner._validate_decisions_shape(
            decisions,
            expected_record_ids=[str(row["record_id"]) for row in packet],
            expected_sample_sha256=owner.sha256_value(packet),
        )


def test_decisions_reject_order_change_unknown_id_and_oversize_note() -> None:
    packet = _packet()
    ids = [str(row["record_id"]) for row in packet]
    decisions = _decisions()
    decisions["decisions"][0], decisions["decisions"][1] = (  # type: ignore[index]
        decisions["decisions"][1],  # type: ignore[index]
        decisions["decisions"][0],  # type: ignore[index]
    )
    with pytest.raises(ValueError):
        owner._validate_decisions_shape(
            decisions,
            expected_record_ids=ids,
            expected_sample_sha256=owner.sha256_value(packet),
        )
    decisions = _decisions()
    decisions["decisions"][0]["record_id"] = "bedstc2-banglish-ffffffffffffffffffffffff"  # type: ignore[index]
    with pytest.raises(ValueError):
        owner._validate_decisions_shape(
            decisions,
            expected_record_ids=ids,
            expected_sample_sha256=owner.sha256_value(packet),
        )
    decisions = _decisions()
    decisions["decisions"][0]["note"] = "x" * 2_001  # type: ignore[index]
    with pytest.raises(ValueError):
        owner._validate_decisions_shape(
            decisions,
            expected_record_ids=ids,
            expected_sample_sha256=owner.sha256_value(packet),
        )


def test_receipt_and_verification_validate() -> None:
    receipt = _receipt()
    owner.validate_receipt(receipt)
    verification = owner.build_verification(
        receipt,
        receipt_file_sha256="b" * 64,
        receipt_file_bytes=len(owner.json_bytes(receipt)),
        plan_file_sha256="a" * 64,
    )
    owner.validate_verification(verification)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("production_eligible",), True),
        (("rights", "upstream_dstc2_rights_chain_independently_verified"), True),
        (("owner_review", "accepted_records"), 31),
        (("owner_review", "threshold", "minimum_accepts"), 28),
        (("remaining_blockers",), []),
        (("reviewer_alias_retained",), True),
    ],
)
def test_resealed_receipt_tampering_is_rejected(
    path: tuple[str, ...], value: object
) -> None:
    receipt = copy.deepcopy(_receipt())
    cursor: object = receipt
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = value  # type: ignore[index]
    _reseal_receipt(receipt)
    with pytest.raises(ValueError):
        owner.validate_receipt(receipt)


def test_receipt_rejects_raw_or_private_keys() -> None:
    receipt = _receipt()
    receipt["messages"] = ["synthetic sentinel"]
    _reseal_receipt(receipt)
    with pytest.raises(ValueError):
        owner.validate_receipt(receipt)


def test_resealed_verification_tampering_is_rejected() -> None:
    receipt = _receipt()
    verification = owner.build_verification(
        receipt,
        receipt_file_sha256="b" * 64,
        receipt_file_bytes=len(owner.json_bytes(receipt)),
        plan_file_sha256="a" * 64,
    )
    verification["independent_replay"]["candidate_row_reads"] = 1  # type: ignore[index]
    _reseal_verification(verification)
    with pytest.raises(ValueError):
        owner.validate_verification(verification)


def test_marker_is_exclusive_and_finalized_in_place(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    marker = owner.EXPECTED_RECEIPT_MARKER
    identity = owner.start_marker(path, marker)
    with pytest.raises(FileExistsError):
        owner.start_marker(path, marker)
    payload = owner.json_bytes(_receipt())
    owner.finish_marker(path, marker, payload, identity)
    assert path.read_bytes() == payload


def test_finish_marker_rejects_wrong_marker(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    path.write_bytes(b"wrong")
    with pytest.raises(ValueError):
        owner.finish_marker(
            path,
            owner.EXPECTED_RECEIPT_MARKER,
            b"payload",
            (path.stat().st_dev, path.stat().st_ino),
        )


def test_finish_marker_rejects_inode_swap(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    identity = owner.start_marker(path, owner.EXPECTED_RECEIPT_MARKER)
    path.rename(tmp_path / "original-marker.json")
    path.write_bytes(owner.EXPECTED_RECEIPT_MARKER)
    with pytest.raises(ValueError):
        owner.finish_marker(
            path,
            owner.EXPECTED_RECEIPT_MARKER,
            b"payload",
            identity,
        )


def test_live_plan_load_is_metadata_only(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("payload read before marker")

    monkeypatch.setattr(owner, "read_bound_bytes", forbidden)
    plan = owner.load_plan(ROOT / owner.EXPECTED_PLAN_PATH, ROOT)
    assert plan["limits"]["source_line_reads"] == 0
    assert plan["limits"]["candidate_row_reads"] == 0
    assert plan["limits"]["baseline_row_reads"] == 0


def test_plan_has_exact_input_and_output_scope() -> None:
    plan = json.loads((ROOT / owner.EXPECTED_PLAN_PATH).read_text(encoding="utf-8"))
    assert set(plan["inputs"]) == {
        "decisions",
        "owner_packet",
        "curation_report",
        "curation_verification",
    }
    assert plan["output"] == owner.EXPECTED_OUTPUT
    serialized = json.dumps(plan, sort_keys=True)
    for forbidden in ("candidate.jsonl", "dialog-dstc2-trn", "dialog-dstc2-dev", "v0.15"):
        assert forbidden not in serialized


def test_entrypoints_have_marker_before_one_build_and_independent_verifier() -> None:
    finalizer_path = ROOT / owner.EXPECTED_CODE_PATHS["finalizer"]
    verifier_path = ROOT / owner.EXPECTED_CODE_PATHS["verifier"]
    finalizer_source = finalizer_path.read_text(encoding="utf-8")
    verifier_source = verifier_path.read_text(encoding="utf-8")
    ast.parse(finalizer_source)
    ast.parse(verifier_source)
    assert finalizer_source.count("build_receipt(") == 1
    assert finalizer_source.index("start_marker(") < finalizer_source.index(
        "build_receipt("
    )
    assert "contract.build_receipt" not in verifier_source
    assert verifier_source.index("contract.start_marker(") < verifier_source.rindex(
        "_independent_expected_receipt("
    )


def test_authorization_limits_and_marker_hashes_are_frozen() -> None:
    assert owner.sha256_bytes(owner.EXPECTED_AUTHORIZATION_TEXT.encode()) == (
        owner.EXPECTED_AUTHORIZATION_SHA256
    )
    assert owner.EXPECTED_LIMITS == {
        "passes": 2,
        "decision_reads": 2,
        "decision_rows": 64,
        "owner_packet_reads": 2,
        "owner_packet_rows": 64,
        "curation_report_reads": 2,
        "curation_verification_reads": 2,
        "source_line_reads": 0,
        "candidate_row_reads": 0,
        "baseline_row_reads": 0,
        "test_object_reads": 0,
        "runtime_seconds_total": 300,
        "runtime_seconds_per_process": 150,
        "private_write_bytes": 1_048_576,
        "network_bytes": 0,
        "gpu_use": 0,
        "inference_calls": 0,
    }
    assert owner.sha256_bytes(owner.EXPECTED_RECEIPT_MARKER) == (
        "4a48f55a061035a65bcb401d82e61a249eb8c09f19f6d23eaf2bb5c76eadeb96"
    )
    assert owner.sha256_bytes(owner.EXPECTED_VERIFICATION_MARKER) == (
        "40f70a10366fc046cc14963457a847dcd675d5cdd98410e989ec6e795092d532"
    )
