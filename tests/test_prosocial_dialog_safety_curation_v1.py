from __future__ import annotations

from lumi.posttraining_prosocial_dialog_safety_curation_v1 import (
    ORIGIN_LABEL,
    RESPONSE_ORIGIN,
    RETAINED_SAFETY_LABELS,
    _candidate_from_raw,
    custom_quality_reasons,
    dialogue_split,
)


def _row(*, dialogue_id: int = 17, response_id: int = 3) -> dict[str, object]:
    return {
        "context": "A person is being treated unfairly by a neighbor.",
        "dialogue_id": dialogue_id,
        "response": "Please support them and seek help from a trusted local service.",
        "response_id": response_id,
        "safety_label": "__needs_caution__",
        "source": "socialchemistry",
    }


def test_dialogue_split_is_stable_and_dialogue_disjoint() -> None:
    assert dialogue_split(17) == dialogue_split(17)
    record_a, attribution_a = _candidate_from_raw(_row(response_id=1), object_name="train.json")
    record_b, attribution_b = _candidate_from_raw(_row(response_id=2), object_name="valid.json")
    assert record_a["split"] == record_b["split"]
    assert attribution_a["dialogue_id"] == attribution_b["dialogue_id"] == 17


def test_candidate_preserves_truthful_mixed_provenance() -> None:
    record, _ = _candidate_from_raw(_row(), object_name="train.json")
    assert record["provenance"]["origin_label"] == ORIGIN_LABEL
    assert record["provenance"]["assistant_response_origin"] == RESPONSE_ORIGIN
    assert record["safety"]["risk_labels"][0] in RETAINED_SAFETY_LABELS
    assert custom_quality_reasons(record) == ()


def test_custom_gate_rejects_unsafe_or_unpunctuated_response() -> None:
    unsafe = _row()
    unsafe["response"] = "You should attack them"
    record, _ = _candidate_from_raw(unsafe, object_name="train.json")
    assert "critical_factual_safety_pattern" in custom_quality_reasons(record)
    assert "assistant_terminal_punctuation" in custom_quality_reasons(record)
