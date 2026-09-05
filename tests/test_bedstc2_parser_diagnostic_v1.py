from __future__ import annotations

import io

from scripts.diagnose_sft_bedstc2_parser_contract_v1 import scan_stream


def _scan(value: bytes) -> dict[str, object]:
    return scan_stream(io.BytesIO(value), maximum_lines=value.count(b"\n") + 1, size=len(value))


def test_detects_implicit_ordinal_one_boundary_without_raw_text() -> None:
    result = _scan(b"1 user\tanswer\n2 entity R_value result\n1 next\treply\n")
    assert result["first_v0_3_failure_code"] == "ordinal_one_reset_without_blank"
    assert result["dialogues_reset_aware"] == 2
    assert result["counts"]["implicit_ordinal_one_boundaries"] == 1
    assert "user" not in str(result)


def test_preserves_explicit_blank_boundaries() -> None:
    result = _scan(b"1 user\tanswer\n\n1 next\treply\n")
    assert result["first_v0_3_failure_code"] is None
    assert result["dialogues_reset_aware"] == 2
    assert result["counts"]["explicit_dialogue_boundaries"] == 1


def test_classifies_api_kb_and_malformed_without_text() -> None:
    result = _scan(
        b"1 user\tapi_call restaurant\n2 entity R_post-code result\n3 malformed row\n"
    )
    assert result["counts"]["api_rows"] == 1
    assert result["counts"]["kb_rows"] == 1
    assert result["counts"]["malformed_non_tab_rows"] == 1
    assert "restaurant" not in str(result)
