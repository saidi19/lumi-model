from __future__ import annotations

import hashlib
import io

import pytest

from scripts.diagnose_sft_bedstc2_malformed_intersections_v3 import (
    EXPECTED_COUNTS,
    EXPECTED_INTERSECTIONS,
    _diagnostic_confirmed,
    _scan_stream,
    _seal,
)


def _expected(payload: bytes, lines: int) -> dict[str, object]:
    git = hashlib.sha1(  # noqa: S324 - canonical Git blob identity.
        f"blob {len(payload)}\0".encode(), usedforsecurity=False
    )
    git.update(payload)
    return {
        "bytes": len(payload),
        "git_blob_sha1": git.hexdigest(),
        "lines": lines,
        "path": "synthetic.txt",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_intersection_scan_is_aggregate_only_and_disjoint() -> None:
    payload = (
        b"1 entity relation value\n"
        b"2 entity R_value result \n"
        b"3 user \tassistant\n"
        b"4 user\tassistant\n\n"
    )
    binding, counts, intersections = _scan_stream(
        io.BytesIO(payload), expected=_expected(payload, 5), total_budget=[0]
    )
    assert binding["lines"] == 5
    assert counts == {
        "non_kb_numbered_rows": 1,
        "outer_trailing_whitespace_rows": 1,
        "pair_user_trailing_whitespace_rows": 1,
        "unique_malformed_rows": 3,
    }
    assert intersections == EXPECTED_INTERSECTIONS
    assert "entity relation value" not in str((binding, counts, intersections))


def test_intersection_scan_counts_overlap_once() -> None:
    payload = b"1 entity relation value \n\n"
    _, counts, intersections = _scan_stream(
        io.BytesIO(payload), expected=_expected(payload, 2), total_budget=[0]
    )
    assert counts["non_kb_numbered_rows"] == 1
    assert counts["outer_trailing_whitespace_rows"] == 1
    assert counts["unique_malformed_rows"] == 1
    assert intersections["non_kb_and_outer"] == 1


def test_exact_authorized_counts_are_required() -> None:
    result = {
        "counts": EXPECTED_COUNTS,
        "intersections": EXPECTED_INTERSECTIONS,
        "source_line_reads": 113_718,
    }
    assert _diagnostic_confirmed(result)
    changed = {**result, "counts": {**EXPECTED_COUNTS, "unique_malformed_rows": 474}}
    assert not _diagnostic_confirmed(changed)


def test_seal_is_canonical_and_raw_free() -> None:
    value = _seal({"counts": EXPECTED_COUNTS, "raw_text_retained": False}, "report_sha256")
    assert len(value["report_sha256"]) == 64
    assert "messages" not in value


@pytest.mark.parametrize(
    "payload",
    [
        b"1 entity R_\n\n",
        b"1 user\tassistant\textra\n\n",
        b"2 user\tassistant\n\n",
    ],
)
def test_unobserved_shapes_remain_fail_closed(payload: bytes) -> None:
    with pytest.raises(ValueError):
        _scan_stream(
            io.BytesIO(payload), expected=_expected(payload, 2), total_budget=[0]
        )
