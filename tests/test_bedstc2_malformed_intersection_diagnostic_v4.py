from __future__ import annotations

import hashlib
import io

import pytest

from scripts.diagnose_sft_bedstc2_malformed_intersections_v4 import (
    AUTHORIZATION_TEXT_SHA256,
    EXPECTED_COUNTS,
    EXPECTED_INTERSECTIONS,
    EXPECTED_OBJECTS,
    EXPECTED_PREDECESSOR_BINDINGS,
    MAXIMUM_DIAGNOSTIC_WRITE_BYTES,
    MAXIMUM_RUNTIME_SECONDS,
    PRIOR_AUTHORIZATION_TEXT_SHA256,
    SCHEMA_VERSION,
    _diagnostic_confirmed,
    _json_bytes,
    _scan_stream,
    _seal,
    build_verification,
    validate_report,
    validate_verification,
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


@pytest.mark.parametrize(
    "payload",
    [
        b"1 user\tassistant \n\n",
        b"1 user\tassistant\t\n\n",
        b"1 entity R_rel value \n\n",
        b"1 entity R_rel value\t\n\n",
    ],
)
def test_outer_suffix_is_classified_from_trimmed_body(payload: bytes) -> None:
    _, counts, intersections = _scan_stream(
        io.BytesIO(payload), expected=_expected(payload, 2), total_budget=[0]
    )
    assert counts == {
        "non_kb_numbered_rows": 0,
        "outer_trailing_whitespace_rows": 1,
        "pair_user_trailing_whitespace_rows": 0,
        "unique_malformed_rows": 1,
    }
    assert intersections == EXPECTED_INTERSECTIONS


def test_pair_user_and_outer_intersection_is_counted() -> None:
    payload = b"1 user \tassistant \n\n"
    _, counts, intersections = _scan_stream(
        io.BytesIO(payload), expected=_expected(payload, 2), total_budget=[0]
    )
    assert counts["pair_user_trailing_whitespace_rows"] == 1
    assert counts["outer_trailing_whitespace_rows"] == 1
    assert counts["unique_malformed_rows"] == 1
    assert intersections["outer_and_pair_user"] == 1


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


def test_complete_expected_report_and_verification_validate() -> None:
    predecessor = {
        name: {
            "bytes": int(expected.get("bytes", 1)),
            "path": expected["path"],
            "sha256": expected["sha256"],
        }
        for name, expected in EXPECTED_PREDECESSOR_BINDINGS.items()
    }
    sources = {
        name: {
            "bytes": expected["bytes"],
            "git_blob_sha1": expected["git_blob_sha1"],
            "lines": expected["lines"],
            "path": expected["path"],
            "sha256": expected["sha256"],
        }
        for name, expected in EXPECTED_OBJECTS.items()
    }
    report = _seal(
        {
            "authorization_text_sha256": AUTHORIZATION_TEXT_SHA256,
            "bindings": {
                "diagnostic_code": {
                    "script": {
                        "bytes": 1,
                        "path": "scripts/diagnose_sft_bedstc2_malformed_intersections_v4.py",
                        "sha256": "1" * 64,
                    },
                    "test": {
                        "bytes": 1,
                        "path": "tests/test_bedstc2_malformed_intersection_diagnostic_v4.py",
                        "sha256": "2" * 64,
                    },
                },
                "predecessor": predecessor,
                "source_objects": sources,
            },
            "diagnostic": {
                "counts": EXPECTED_COUNTS,
                "intersections": EXPECTED_INTERSECTIONS,
                "source_line_reads": 113_718,
            },
            "limits": {
                "baseline_row_reads": 0,
                "candidate_row_reads": 0,
                "maximum_runtime_seconds": MAXIMUM_RUNTIME_SECONDS,
                "maximum_write_bytes": MAXIMUM_DIAGNOSTIC_WRITE_BYTES,
                "network_bytes": 0,
                "source_line_reads": 113_718,
            },
            "prior_authorization_text_sha256": PRIOR_AUTHORIZATION_TEXT_SHA256,
            "prior_consumption": {
                "baseline_row_reads": 0,
                "candidate_row_reads": 0,
                "diagnostic_passes_completed": 0,
                "diagnostic_passes_started": 1,
                "gpu_use": 0,
                "inference_calls": 0,
                "network_bytes": 0,
                "private_write_bytes": 61,
                "runtime_seconds": 0.07,
                "source_line_reads_upper_bound": 113_718,
                "test_object_accessed": False,
                "verification_invoked": False,
            },
            "prohibited_actions": {
                "gpu_used": False,
                "inference_called": False,
                "network_used": False,
                "registry_mutated": False,
                "source_mutated": False,
                "test_object_accessed": False,
            },
            "raw_text_retained": False,
            "schema_version": SCHEMA_VERSION,
            "status": "PASS_confirmed_421_17_37_unique_475_disjoint",
        },
        "report_sha256",
    )
    validate_report(report)
    verification = build_verification(report, _json_bytes(report))
    validate_verification(verification)
    assert verification["conditional_recovery_authorized"] is True


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
