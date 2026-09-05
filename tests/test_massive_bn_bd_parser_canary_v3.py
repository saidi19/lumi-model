from __future__ import annotations

import copy
import hashlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

import lumi.posttraining_massive_bn_bd_parser_canary_v3 as canary
from lumi.posttraining_data import evaluate_sft_record, load_sft_policy


def _row(
    utterance: str,
    annotated: str,
    *,
    scenario: int = 9,
    intent: int = 5,
) -> dict[str, object]:
    return {
        "locale": "bn-BD",
        "partition": "train",
        "scenario": scenario,
        "intent": intent,
        "utt": utterance,
        "annot_utt": annotated,
        "id": "forbidden-source-id",
        "worker_id": "forbidden-worker-id",
        "slot_method": {"forbidden": "payload"},
        "judgments": {"forbidden": "payload"},
    }


def _semantic(
    utterance: str = "আজ কী খবর",
    *,
    scenario: str = "general",
    intent: str = "general_greet",
    slots: tuple[canary.Slot, ...] = (),
) -> canary.SemanticTuple:
    return canary.SemanticTuple(
        utterance=utterance,
        scenario=scenario,
        intent=intent,
        intent_index=canary.predecessor.INTENT_LABELS.index(intent),
        slots=slots,
    )


def test_fixed_scope_marker_and_collision_bindings() -> None:
    assert canary.SOURCE_COLUMNS == (
        "locale",
        "partition",
        "scenario",
        "intent",
        "utt",
        "annot_utt",
    )
    assert len(canary.START_MARKER) == 47
    assert canary.START_MARKER_SHA256 == (
        "a9e55cfc4fddbd1ec0ad01c73693578439a70cdb9453abfb068efd3b7c5a5c17"
    )
    assert all(
        value["introduced_normalization_collision_groups"] == 8
        and value["introduced_normalization_collision_affected_rows"] == 22
        for value in canary.EXPECTED_COLLISIONS.values()
    )
    assert canary.EXPECTED_LIMITS["exact_source_passes"] == 1
    assert canary.EXPECTED_LIMITS["maximum_source_row_reads"] == 11_514
    assert canary.EXPECTED_LIMITS["maximum_baseline_candidate_reads"] == 0


def test_nfc_equivalent_full_semantics_collapse_and_repeated_slots_survive() -> None:
    rows = [
        _row(
            "Café today and tomorrow",
            "[place:Café] [date:today] and [date:tomorrow]",
            scenario=2,
            intent=50,
        ),
        _row(
            "Cafe\u0301 today and tomorrow",
            "[place:Cafe\u0301] [date:today] and [date:tomorrow]",
            scenario=2,
            intent=50,
        ),
    ]
    result = canary.analyze_rows(rows, maximum_rows=2)

    assert result.aggregate["curation"] == {
        "unique_normalized_semantic_tuples": 1,
        "duplicate_source_rows_collapsed": 1,
        "ambiguous_normalized_utterance_groups": 0,
        "ambiguous_source_rows": 0,
        "ambiguous_unique_semantic_tuples": 0,
        "nonambiguous_unique_semantic_tuples": 1,
        "quarantine_ledger_sha256": canary.sha256_value([]),
    }
    semantic = result.eligible_semantics[0]
    assert [slot.name for slot in semantic.slots] == ["place", "date", "date"]
    assert [slot.value for slot in semantic.slots] == ["Café", "today", "tomorrow"]
    for scope in ("utterance", "annotated_utterance", "semantic_tuple"):
        collision = result.aggregate["duplicates"][scope]
        assert collision["introduced_normalization_collision_groups"] == 1
        assert collision["introduced_normalization_collision_affected_rows"] == 2
    serialized = json.dumps(result.aggregate, ensure_ascii=False)
    assert "Café" not in serialized
    assert "forbidden-source-id" not in serialized
    assert "forbidden-worker-id" not in serialized


def test_multi_projection_normalized_utterance_is_quarantined_as_a_whole() -> None:
    result = canary.analyze_rows(
        [
            _row("same request", "same request", scenario=9, intent=5),
            _row("same request", "same request", scenario=9, intent=25),
        ],
        maximum_rows=2,
    )
    curation = result.aggregate["curation"]
    assert curation["unique_normalized_semantic_tuples"] == 2
    assert curation["ambiguous_normalized_utterance_groups"] == 1
    assert curation["ambiguous_source_rows"] == 2
    assert curation["ambiguous_unique_semantic_tuples"] == 2
    assert curation["nonambiguous_unique_semantic_tuples"] == 0
    assert result.eligible_semantics == ()


def test_candidate_is_canonical_four_role_parser_with_no_side_effect_claim() -> None:
    semantic = _semantic(
        slots=(
            canary.Slot(name="date", value="আজ"),
            canary.Slot(name="date", value="কাল"),
        )
    )
    record = canary.build_candidate(semantic)
    replay = canary.parse_and_validate_candidate(record)

    assert replay == semantic
    assert [message["role"] for message in record["messages"]] == [
        "user",
        "tool",
        "tool_result",
        "assistant",
    ]
    tool = json.loads(record["messages"][1]["content"])
    result = json.loads(record["messages"][2]["content"])
    assert tool["name"] == "massive.parse_intent"
    assert [item["name"] for item in tool["arguments"]["slots"]] == ["date", "date"]
    assert result["status"] == "parsed_only"
    assert result["external_action_performed"] is False
    assert record["record_id"].endswith(semantic.digest[:24])
    assert semantic.digest in record["source"]["document_id"]


def test_candidate_rejects_any_external_action_tamper() -> None:
    record = canary.build_candidate(_semantic())
    tampered = copy.deepcopy(record)
    result = json.loads(tampered["messages"][2]["content"])
    result["external_action_performed"] = True
    tampered["messages"][2]["content"] = canary.canonical_json_text(result)
    with pytest.raises(ValueError):
        canary.parse_and_validate_candidate(tampered)


def test_candidate_passes_frozen_generic_schema_without_registry_credit() -> None:
    root = Path(__file__).resolve().parents[1]
    policy = load_sft_policy(root / "configs/lumi-4b-sft-data-contract-v0.1.json")
    record = canary.build_candidate(_semantic())
    decision = evaluate_sft_record(
        record, policy=policy, benchmark_registry=None, source_registry=None
    )
    assert decision.accepted is True
    assert record["provenance"]["source_registry_status"] == "pending_separate_promotion"


def test_selection_is_60_intent_covering_plus_four_deterministic_extras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measured: list[canary.MeasuredCandidate] = []
    scenario_by_prefix = {
        label: index for index, label in enumerate(canary.predecessor.SCENARIO_LABELS)
    }
    for intent_index, intent in enumerate(canary.predecessor.INTENT_LABELS):
        scenario = intent.partition("_")[0]
        assert scenario in scenario_by_prefix
        semantic = canary.SemanticTuple(
            utterance=f"synthetic request {intent_index}",
            scenario=scenario,
            intent=intent,
            intent_index=intent_index,
            slots=(),
        )
        measured.append(
            canary.MeasuredCandidate(
                semantic=semantic,
                record=canary.build_candidate(semantic),
                total_tokens=20,
                supervised_tokens=10,
                masked_tokens=10,
            )
        )
    for intent_index in range(4):
        intent = canary.predecessor.INTENT_LABELS[intent_index]
        semantic = canary.SemanticTuple(
            utterance=f"extra synthetic request {intent_index}",
            scenario=intent.partition("_")[0],
            intent=intent,
            intent_index=intent_index,
            slots=(),
        )
        measured.append(
            canary.MeasuredCandidate(
                semantic=semantic,
                record=canary.build_candidate(semantic),
                total_tokens=20,
                supervised_tokens=10,
                masked_tokens=10,
            )
        )

    class _NoDuplicates:
        def signature(self, text: str) -> str:
            return text

        def duplicate_of(self, signature: str) -> None:
            return None

        def add(self, document_id: str, signature: str) -> None:
            assert document_id and signature

    monkeypatch.setattr(canary, "_near_index", _NoDuplicates)
    selected, summary = canary.select_canary(measured)
    assert len(selected) == 64
    assert summary["intent_coverage_rows"] == 60
    assert summary["missing_intents"] == 0
    assert summary["deterministic_extra_rows"] == 4
    assert len({item.semantic.intent_index for item in selected}) == 60


def test_owner_review_packet_is_private_source_derived_and_action_safe() -> None:
    record = canary.build_candidate(_semantic())
    payload, digest = canary.owner_review_html([record])
    text = payload.decode("utf-8")
    assert digest == canary.sha256_value(canary._owner_sample([record]))
    assert "source-derived and machine-curated" in text
    assert "does not make them human-written or authorize external actions" in text
    assert record["record_id"] in text


def _minimal_public_report() -> dict[str, object]:
    report: dict[str, object] = {
        "schema_version": canary.REPORT_SCHEMA,
        "status": canary.STOP_STATUS,
        "observed_at": "2026-08-31",
        "authorization_sha256": canary.AUTHORIZATION_SHA256,
        "plan_file_sha256": "0" * 64,
        "plan_internal_sha256": "1" * 64,
        "bindings": {},
        "scan": {},
        "scan_checks": {},
        "curation": {},
        "machine_gate": {},
        "checks": {},
        "private_artifacts": {},
        "owner_review": {},
        "decision": {},
        "resource_accounting": {},
        "raw_text_retained": False,
        "production_eligible": False,
    }
    canary._seal(report, "report_sha256")
    return report


def test_public_report_guard_rejects_unknown_or_nested_payload_keys() -> None:
    good = _minimal_public_report()
    canary.validate_public_report(good)

    unknown = copy.deepcopy(good)
    unknown["note"] = "SOURCE PAYLOAD"
    canary._seal(unknown, "report_sha256")
    with pytest.raises(ValueError):
        canary.validate_public_report(unknown)

    nested = copy.deepcopy(good)
    nested["bindings"] = {"messages": ["SOURCE PAYLOAD"]}
    canary._seal(nested, "report_sha256")
    with pytest.raises(ValueError):
        canary.validate_public_report(nested)


def test_real_shaped_stop_report_is_raw_text_free_and_structurally_valid() -> None:
    scan = canary.analyze_rows([_row("hello", "hello")], maximum_rows=1)
    empty_hash = canary.sha256_value([])
    machine_gate = {
        "generic_filter": {
            "total": 0,
            "accepted": 0,
            "rejected": 0,
            "rejection_counts": {},
            "accepted_dataset_sha256": empty_hash,
        },
        "tokenization": {
            "total_tokens": 0,
            "supervised_tokens": 0,
            "masked_tokens": 0,
            "maximum_conversation_tokens": 0,
            "context_window": 4096,
            "assistant_and_tool_only_loss": True,
        },
        "deduplication": {
            "exact_normalized_conversation_duplicates": 0,
            "near_duplicate_groups": 0,
            "near_duplicate_affected_rows": 0,
            "near_duplicate_ledger_sha256": empty_hash,
            "semantic_selection_sha256": empty_hash,
        },
        "parser_contract": {
            "rows": 0,
            "four_role_trajectory_rows": 0,
            "parsed_only_rows": 0,
            "external_action_performed_rows": 0,
            "ordered_slot_list_rows": 0,
            "unique_intents": 0,
        },
    }
    report = canary._build_report(
        plan={
            "plan_sha256": "0" * 64,
            "predecessor_bindings": {},
            "contract_bindings": {},
            "code_bindings": {},
        },
        plan_file_sha256="1" * 64,
        scan=scan,
        scan_checks={"collision_aggregates_reproduced": False},
        measurement={
            "evaluated": 0,
            "eligible": 0,
            "rejected": 0,
            "rejection_counts": {},
        },
        selection={
            "intent_coverage_rows": 0,
            "missing_intents": 60,
            "deterministic_extra_rows": 0,
            "near_duplicate_skips": 0,
            "selected_rows": 0,
            "selection_sha256": empty_hash,
        },
        machine_gate=machine_gate,
        private_artifacts=canary._empty_private_artifacts(),
        private_write_bytes=len(canary.START_MARKER),
        elapsed_seconds=0.0,
    )
    assert report["status"] == canary.STOP_STATUS
    canary.validate_public_report(report)


def test_parquet_wrapper_requests_only_six_columns_in_one_source_pass() -> None:
    row = _row("hello", "hello")
    features = {
        "scenario": {
            "_type": "ClassLabel",
            "names": list(canary.predecessor.SCENARIO_LABELS),
        },
        "intent": {
            "_type": "ClassLabel",
            "names": list(canary.predecessor.INTENT_LABELS),
        },
    }
    raw_metadata = json.dumps(
        {"info": {"features": features}},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    observed: dict[str, object] = {"iter_calls": 0}

    class _Field:
        def __init__(self, name: str, field_type: str, nullable: bool) -> None:
            self.name = name
            self.type = field_type
            self.nullable = nullable

    class _Schema:
        metadata = {b"huggingface": raw_metadata}

        def __iter__(self):
            return iter(
                _Field(*descriptor)
                for descriptor in canary.predecessor.EXPECTED_TOP_LEVEL_SCHEMA
            )

    class _Batch:
        def to_pydict(self):
            return {key: [row[key]] for key in canary.SOURCE_COLUMNS}

    class _ParquetFile:
        metadata = types.SimpleNamespace(num_rows=1)
        schema_arrow = _Schema()

        def __init__(self, path, *, pre_buffer, memory_map) -> None:
            assert path == "synthetic.parquet"
            assert pre_buffer is False
            assert memory_map is False

        def iter_batches(self, *, batch_size, columns, use_threads):
            observed["iter_calls"] = int(observed["iter_calls"]) + 1
            observed["columns"] = columns
            assert batch_size == 1_024
            assert use_threads is False
            yield _Batch()

    parquet_module = types.ModuleType("pyarrow.parquet")
    parquet_module.ParquetFile = _ParquetFile
    pyarrow_module = types.ModuleType("pyarrow")
    pyarrow_module.parquet = parquet_module
    with (
        patch.dict(
            sys.modules,
            {"pyarrow": pyarrow_module, "pyarrow.parquet": parquet_module},
        ),
        patch.dict(canary.EXPECTED_LIMITS, {"maximum_source_row_reads": 1}),
        patch.object(canary.predecessor, "HUGGINGFACE_METADATA_BYTES", len(raw_metadata)),
        patch.object(
            canary.predecessor,
            "HUGGINGFACE_METADATA_SHA256",
            hashlib.sha256(raw_metadata).hexdigest(),
        ),
        patch.object(
            canary.predecessor,
            "HUGGINGFACE_CANONICAL_FEATURES_SHA256",
            hashlib.sha256(canary.canonical_json_bytes(features)).hexdigest(),
        ),
    ):
        result = canary.scan_bound_parquet("synthetic.parquet")

    assert observed == {"iter_calls": 1, "columns": list(canary.SOURCE_COLUMNS)}
    assert len(result.eligible_semantics) == 1
