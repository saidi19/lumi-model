from __future__ import annotations

import copy
import hashlib
import json
import sys
import types
from unittest.mock import patch

import pytest

import lumi.posttraining_massive_bn_bd_nfc_recovery_v2 as recovery


def _expectations(
    *,
    rows: int,
    non_nfc_utterance: int = 0,
    non_nfc_annotated: int = 0,
    annotations: int = 0,
    annotated_rows: int = 0,
    maximum_annotations: int = 0,
    slot_labels: int = 0,
    scenarios: int = 1,
    intents: int = 1,
) -> recovery.RecoveryExpectations:
    return recovery.RecoveryExpectations(
        rows=rows,
        raw_non_nfc_utterances=non_nfc_utterance,
        raw_non_nfc_annotated_utterances=non_nfc_annotated,
        total_annotations=annotations,
        rows_with_annotations=annotated_rows,
        maximum_annotations_per_row=maximum_annotations,
        slot_label_cardinality=slot_labels,
        scenario_cardinality=scenarios,
        intent_cardinality=intents,
    )


def _row(
    utterance: str,
    annotated: str,
    *,
    scenario: int = 5,
    intent: int = 0,
    locale: str = "bn-BD",
    partition: str = "train",
) -> dict[str, object]:
    return {
        "locale": locale,
        "partition": partition,
        "scenario": scenario,
        "intent": intent,
        "utt": utterance,
        "annot_utt": annotated,
        # Sensitive values may exist on an input row but the recovery core never reads them.
        "id": "synthetic-id-must-not-escape",
        "worker_id": "synthetic-worker-must-not-escape",
        "judgments": {"synthetic": "payload-must-not-escape"},
    }


def test_fixed_metadata_and_label_map_bindings() -> None:
    assert recovery.AUDIT_COLUMNS == (
        "locale",
        "partition",
        "scenario",
        "intent",
        "utt",
        "annot_utt",
    )
    assert len(recovery.SCENARIO_LABELS) == 18
    assert len(recovery.INTENT_LABELS) == 60
    assert recovery._class_label_sha256(recovery.SCENARIO_LABELS) == (
        "75badc0873f3cee75962a95d553bda51ce11e8f7089bce25f91a9b3f9419411f"
    )
    assert recovery._class_label_sha256(recovery.INTENT_LABELS) == (
        "c1d704291fc472572de507c7ff40487d73cc70a97bd29a9954b14d609350a012"
    )
    assert recovery.HUGGINGFACE_METADATA_BYTES == 2_192
    assert recovery.HUGGINGFACE_METADATA_SHA256 == (
        "f2e7b189ab9dc9c95ce9fdab5ef7d36522771451fa663fee11af08c8612c245e"
    )
    assert recovery.HUGGINGFACE_CANONICAL_FEATURES_SHA256 == (
        "c68f119d6e7737f35447f34b703e617cda3f69c85cce718f0022381f5d5c2164"
    )


def test_production_expectations_remain_exact() -> None:
    assert recovery.PRODUCTION_EXPECTATIONS == recovery.RecoveryExpectations(
        rows=11_514,
        raw_non_nfc_utterances=1_495,
        raw_non_nfc_annotated_utterances=1_495,
        total_annotations=11_287,
        rows_with_annotations=7_754,
        maximum_annotations_per_row=7,
        slot_label_cardinality=55,
        scenario_cardinality=18,
        intent_cardinality=60,
    )


def test_valid_nfc_projection_is_aggregate_only_and_reconciles() -> None:
    rows = [
        _row("Cafe\u0301 tomorrow", "[date : Cafe\u0301] tomorrow"),
        _row("lights blue", "lights [color : blue]", scenario=8, intent=1),
        _row("ticket", "ticket", scenario=1, intent=2),
    ]
    expected = _expectations(
        rows=3,
        non_nfc_utterance=1,
        non_nfc_annotated=1,
        annotations=2,
        annotated_rows=2,
        maximum_annotations=1,
        slot_labels=2,
        scenarios=3,
        intents=3,
    )
    first, second, reconciled = recovery.audit_and_reconcile_two_passes(
        iter(rows), iter(rows), expectations=expected
    )

    assert first == second
    assert first["decision"] == {
        "nfc_recovery_gate_pass": True,
        "scalar_argument_adapter_eligible": True,
        "fresh_adapter_contract_successor_eligible": True,
        "candidate_construction_eligible": False,
        "candidate_rows_constructed": 0,
        "active_tool_api_credit_rows": 0,
    }
    assert first["aggregate"]["nfc_intersection"] == {
        "utterance_only": 0,
        "annotated_utterance_only": 0,
        "both": 1,
        "neither": 2,
    }
    assert first["aggregate"]["annotations"]["ordered_signature_mismatch_rows"] == 0
    assert first["aggregate"]["annotations"]["literal_segment_mismatch_rows"] == 0
    assert first["aggregate"]["annotations"]["deannotation_mismatch_rows"] == 0
    assert reconciled["decision"]["nfc_recovery_gate_pass"] is True
    serialized = json.dumps(first, ensure_ascii=False)
    for forbidden in (
        "Cafe",
        "tomorrow",
        "synthetic-id-must-not-escape",
        "synthetic-worker-must-not-escape",
        "payload-must-not-escape",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    "annotated",
    (
        "[date : tomorrow",
        "tomorrow]",
        "[[date : tomorrow]]",
        "[ : tomorrow]",
        "[date : ]",
        "[date tomorrow]",
    ),
)
def test_strict_parser_fails_closed_on_malformed_brackets(annotated: str) -> None:
    result = recovery.audit_recovery_pass(
        [_row("tomorrow", annotated)], expectations=_expectations(rows=1)
    )
    assert result["aggregate"]["annotations"]["malformed_rows"] == 1
    assert result["decision"]["nfc_recovery_gate_pass"] is False


def test_preexisting_duplicate_slot_labels_do_not_fail_nfc_recovery() -> None:
    result = recovery.audit_recovery_pass(
        [_row("today and tomorrow", "[date : today] and [date : tomorrow]")],
        expectations=_expectations(
            rows=1,
            annotations=2,
            annotated_rows=1,
            maximum_annotations=2,
            slot_labels=1,
        ),
    )
    annotations = result["aggregate"]["annotations"]
    assert annotations["preexisting_duplicate_slot_label_rows"] == 1
    assert annotations["introduced_nfc_duplicate_slot_label_rows"] == 0
    assert annotations["post_nfc_duplicate_slot_label_rows"] == 1
    assert result["decision"]["nfc_recovery_gate_pass"] is True
    assert result["decision"]["scalar_argument_adapter_eligible"] is False
    assert result["decision"]["fresh_adapter_contract_successor_eligible"] is True
    assert result["decision"]["candidate_construction_eligible"] is False


def test_nfc_introduced_slot_label_collision_fails_recovery() -> None:
    result = recovery.audit_recovery_pass(
        [_row("today and tomorrow", "[café : today] and [cafe\u0301 : tomorrow]")],
        expectations=_expectations(
            rows=1,
            non_nfc_annotated=1,
            annotations=2,
            annotated_rows=1,
            maximum_annotations=2,
            slot_labels=1,
        ),
    )
    annotations = result["aggregate"]["annotations"]
    assert annotations["preexisting_duplicate_slot_label_rows"] == 0
    assert annotations["introduced_nfc_duplicate_slot_label_rows"] == 1
    assert result["decision"]["nfc_recovery_gate_pass"] is False


def test_nfc_introduced_content_collisions_fail_for_all_three_scopes() -> None:
    result = recovery.audit_recovery_pass(
        [_row("Café", "Café"), _row("Cafe\u0301", "Cafe\u0301")],
        expectations=_expectations(
            rows=2,
            non_nfc_utterance=1,
            non_nfc_annotated=1,
        ),
    )
    duplicates = result["aggregate"]["duplicates"]
    for scope in ("utterance", "annotated_utterance", "semantic_tuple"):
        assert duplicates[scope]["preexisting_duplicate_groups"] == 0
        assert duplicates[scope]["introduced_normalization_collision_groups"] == 1
        assert duplicates[scope]["introduced_normalization_collision_affected_rows"] == 2
    assert result["decision"]["nfc_recovery_gate_pass"] is False


def test_preexisting_exact_duplicates_are_counted_but_do_not_fail_recovery() -> None:
    result = recovery.audit_recovery_pass(
        [_row("same", "same"), _row("same", "same")],
        expectations=_expectations(rows=2),
    )
    for summary in result["aggregate"]["duplicates"].values():
        assert summary["preexisting_duplicate_groups"] == 1
        assert summary["preexisting_duplicate_affected_rows"] == 2
        assert summary["introduced_normalization_collision_groups"] == 0
    assert result["decision"]["nfc_recovery_gate_pass"] is True


def test_deannotation_scope_and_intent_prefix_are_fail_closed() -> None:
    mismatch = recovery.audit_recovery_pass(
        [_row("tomorrow", "[date : today]")],
        expectations=_expectations(
            rows=1,
            annotations=1,
            annotated_rows=1,
            maximum_annotations=1,
            slot_labels=1,
        ),
    )
    assert mismatch["aggregate"]["annotations"]["deannotation_mismatch_rows"] == 1
    assert mismatch["decision"]["nfc_recovery_gate_pass"] is False

    wrong_semantics = recovery.audit_recovery_pass(
        [_row("hello", "hello", scenario=0, intent=0)],
        expectations=_expectations(rows=1),
    )
    assert wrong_semantics["aggregate"]["semantics"]["intent_prefix_mismatch_rows"] == 1
    assert wrong_semantics["decision"]["nfc_recovery_gate_pass"] is False

    wrong_scope = recovery.audit_recovery_pass(
        [_row("hello", "hello", locale="en-US")],
        expectations=_expectations(rows=1),
    )
    assert wrong_scope["checks"]["all_rows_are_bn_bd_train"] is False
    assert wrong_scope["decision"]["nfc_recovery_gate_pass"] is False


def test_raw_text_free_guard_and_reconciliation_mismatch() -> None:
    good = recovery.audit_recovery_pass([_row("same", "same")], expectations=_expectations(rows=1))
    bad = recovery.audit_recovery_pass(
        [_row("different", "different", scenario=0, intent=0)],
        expectations=_expectations(rows=1),
    )
    reconciled = recovery.reconcile_recovery_passes(good, bad)
    assert reconciled["checks"]["pass_results_identical"] is False
    assert reconciled["decision"]["nfc_recovery_gate_pass"] is False

    contaminated = dict(good)
    contaminated["utt"] = "must not be retained"
    with pytest.raises(ValueError):
        recovery.validate_raw_text_free_result(contaminated)

    for key in ("message", "note"):
        contaminated = dict(good)
        contaminated[key] = "SOURCE PAYLOAD"
        with pytest.raises(ValueError):
            recovery.validate_raw_text_free_result(contaminated)

    contaminated = dict(good)
    contaminated["decision"] = dict(good["decision"])
    contaminated["decision"]["note"] = "SOURCE PAYLOAD"
    with pytest.raises(ValueError):
        recovery.validate_raw_text_free_result(contaminated)

    logically_tampered = copy.deepcopy(good)
    logically_tampered["aggregate"]["rows"] += 1
    with pytest.raises(ValueError):
        recovery.validate_raw_text_free_result(logically_tampered)


def test_observed_class_label_order_is_directly_bound() -> None:
    valid = {
        "scenario": {"_type": "ClassLabel", "names": list(recovery.SCENARIO_LABELS)},
        "intent": {"_type": "ClassLabel", "names": list(recovery.INTENT_LABELS)},
    }
    recovery._validate_observed_class_label_maps(valid)
    altered = dict(valid)
    altered["scenario"] = {
        "_type": "ClassLabel",
        "names": list(reversed(recovery.SCENARIO_LABELS)),
    }
    with pytest.raises(ValueError):
        recovery._validate_observed_class_label_maps(altered)


def test_parquet_wrapper_streams_exact_allowlisted_columns_once() -> None:
    rows = [_row("same", "same")]
    features = {
        "scenario": {"_type": "ClassLabel", "names": list(recovery.SCENARIO_LABELS)},
        "intent": {"_type": "ClassLabel", "names": list(recovery.INTENT_LABELS)},
        "synthetic": {"dtype": "string", "_type": "Value"},
    }
    raw_metadata = json.dumps(
        {"info": {"features": features}},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    captured: dict[str, object] = {"calls": 0}

    class _Field:
        def __init__(self, name: str, field_type: str, nullable: bool) -> None:
            self.name = name
            self.type = field_type
            self.nullable = nullable

    class _Schema:
        metadata = {b"huggingface": raw_metadata}

        def __iter__(self):
            return iter(_Field(*descriptor) for descriptor in recovery.EXPECTED_TOP_LEVEL_SCHEMA)

    class _Batch:
        def to_pydict(self):
            return {key: [rows[0][key]] for key in recovery.AUDIT_COLUMNS}

    class _ParquetFile:
        metadata = types.SimpleNamespace(num_rows=1)
        schema_arrow = _Schema()

        def __init__(self, path, *, pre_buffer, memory_map) -> None:
            assert path == "synthetic.parquet"
            assert pre_buffer is False
            assert memory_map is False

        def iter_batches(self, *, batch_size, columns, use_threads):
            captured["calls"] = int(captured["calls"]) + 1
            captured["columns"] = columns
            assert batch_size == 1_024
            assert use_threads is False
            yield _Batch()

    parquet_module = types.ModuleType("pyarrow.parquet")
    parquet_module.ParquetFile = _ParquetFile
    pyarrow_module = types.ModuleType("pyarrow")
    pyarrow_module.parquet = parquet_module
    feature_hash = hashlib.sha256(recovery._canonical_json_bytes(features)).hexdigest()
    with (
        patch.dict(
            sys.modules,
            {"pyarrow": pyarrow_module, "pyarrow.parquet": parquet_module},
        ),
        patch.object(recovery, "HUGGINGFACE_METADATA_BYTES", len(raw_metadata)),
        patch.object(
            recovery,
            "HUGGINGFACE_METADATA_SHA256",
            hashlib.sha256(raw_metadata).hexdigest(),
        ),
        patch.object(recovery, "HUGGINGFACE_CANONICAL_FEATURES_SHA256", feature_hash),
    ):
        result = recovery.aggregate_parquet_pass(
            "synthetic.parquet", expectations=_expectations(rows=1)
        )

    assert captured == {"calls": 1, "columns": list(recovery.AUDIT_COLUMNS)}
    assert result["decision"]["nfc_recovery_gate_pass"] is True
    assert result["decision"]["candidate_construction_eligible"] is False
