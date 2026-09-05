from __future__ import annotations

from pathlib import Path

from scripts.diagnose_sft_bedstc2_row_shapes_v2 import scan


def test_shape_diagnostic_emits_only_structural_aggregates(tmp_path: Path) -> None:
    path = tmp_path / "rows.txt"
    payload = (
        b"1 user \t assistant\n"
        b"2 entity R_value result extra\n"
        b"3 entity  R_value result\n\n"
    )
    path.write_bytes(payload)
    result = scan(path, expected={"bytes": len(payload), "lines": 4})
    assert result["counts"]["pair_user_trailing_whitespace"] == 1
    assert result["counts"]["pair_assistant_leading_whitespace"] == 1
    assert result["histograms"]["non_tab_token_counts"] == {"4": 2}
    assert "result extra" not in str(result)
