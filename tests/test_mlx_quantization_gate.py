from __future__ import annotations

import pytest

from lumi.mlx_quantization_gate import _select_stratified_sequences


def _stratum(identifier: str, start: int, end: int) -> dict:
    return {
        "category": f"category-{identifier}",
        "component": f"component-{identifier}",
        "id": identifier,
        "sequence_end": end,
        "sequence_start": start,
    }


def test_stratified_probe_selects_centers_of_equal_bins() -> None:
    selected = _select_stratified_sequences(
        [_stratum("a", 0, 8), _stratum("b", 20, 32)],
        sequences_per_stratum=2,
    )

    assert selected[0]["sequence_indices"] == [2, 6]
    assert selected[1]["sequence_indices"] == [23, 29]
    assert [item["id"] for item in selected] == ["a", "b"]


def test_stratified_probe_rejects_an_undersized_stratum() -> None:
    with pytest.raises(ValueError, match="cannot select 2"):
        _select_stratified_sequences(
            [_stratum("tiny", 4, 5)],
            sequences_per_stratum=2,
        )


def test_stratified_probe_rejects_nonpositive_count() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _select_stratified_sequences(
            [_stratum("a", 0, 8)],
            sequences_per_stratum=0,
        )
