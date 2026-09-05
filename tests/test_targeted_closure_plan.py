from __future__ import annotations

from lumi.posttraining_targeted_closure_plan import (
    _matrix_totals,
    sha256_value,
)

MATRIX = {
    "safety_helpfulness": {
        "banglish": 430_000,
        "natural_code_switching": 260_000,
        "bengali_script": 60_000,
        "english": 100_000,
    },
    "writing_summarization": {
        "banglish": 170_000,
        "natural_code_switching": 100_000,
        "bengali_script": 80_000,
        "english": 0,
    },
    "multilingual_bengali": {
        "banglish": 180_000,
        "natural_code_switching": 70_000,
        "bengali_script": 0,
        "english": 0,
    },
    "factual_qa": {
        "banglish": 70_000,
        "natural_code_switching": 20_000,
        "bengali_script": 10_000,
        "english": 50_000,
    },
}


def test_target_matrix_sums_exactly() -> None:
    tasks, languages = _matrix_totals(MATRIX)
    assert tasks == {
        "safety_helpfulness": 850_000,
        "writing_summarization": 350_000,
        "multilingual_bengali": 250_000,
        "factual_qa": 150_000,
    }
    assert languages == {
        "banglish": 850_000,
        "natural_code_switching": 450_000,
        "bengali_script": 150_000,
        "english": 150_000,
    }
    assert sum(tasks.values()) == 1_600_000
    assert sum(languages.values()) == 1_600_000


def test_internal_hash_is_deterministic() -> None:
    assert sha256_value({"b": 2, "a": 1}) == sha256_value({"a": 1, "b": 2})
