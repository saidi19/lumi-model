from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest

import scripts.diagnose_sft_bedstc2_vocab_contract_v5 as diagnostic


def _payload(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def test_current_loader_contract_is_recognized() -> None:
    result = diagnostic.analyze_payload(
        _payload(
            {
                "english_language_vocab": ["hello", "need-help"],
                "native_language_vocab": ["ami", "chai"],
                "others_vocab": ["restaurant"],
            }
        )
    )
    loader = result["current_loader_contract"]
    assert loader == {
        "expected_top_level_keys": list(diagnostic.EXPECTED_KEYS),
        "logical_role_mapping_unambiguous": True,
        "mismatch_codes": [],
        "passes": True,
        "repairable_key_mismatch": False,
    }
    assert result["logical_role_mapping"] == {
        "english": "english_language_vocab",
        "native": "native_language_vocab",
        "others": "others_vocab",
    }


def test_clean_three_key_alias_contract_is_conditionally_repairable() -> None:
    result = diagnostic.analyze_payload(
        _payload(
            {
                "bn_vocab": ["ami", "chai"],
                "en_vocab": ["hello", "need"],
                "universal_vocab": ["restaurant"],
            }
        )
    )
    loader = result["current_loader_contract"]
    assert loader["passes"] is False
    assert loader["repairable_key_mismatch"] is True
    assert loader["mismatch_codes"] == ["top_level_keys_mismatch"]
    assert result["top_level_keys"] == ["bn_vocab", "en_vocab", "universal_vocab"]
    assert result["logical_role_mapping"] == {
        "english": "en_vocab",
        "native": "bn_vocab",
        "others": "universal_vocab",
    }


def test_arbitrary_clean_keys_never_authorize_semantic_role_guessing() -> None:
    result = diagnostic.analyze_payload(
        _payload(
            {
                "first": ["alpha"],
                "second": ["beta"],
                "third": ["gamma"],
            }
        )
    )
    loader = result["current_loader_contract"]
    assert loader["repairable_key_mismatch"] is False
    assert loader["logical_role_mapping_unambiguous"] is False
    assert "logical_role_mapping_ambiguous_or_unknown" in loader["mismatch_codes"]
    assert result["logical_role_mapping"] == {
        "english": None,
        "native": None,
        "others": None,
    }


def test_ambiguous_or_incomplete_alias_mapping_stops() -> None:
    result = diagnostic.analyze_payload(
        _payload(
            {
                "en_vocab": ["hello"],
                "english_vocab": ["help"],
                "bn_vocab": ["ami"],
            }
        )
    )
    loader = result["current_loader_contract"]
    assert loader["repairable_key_mismatch"] is False
    assert loader["logical_role_mapping_unambiguous"] is False
    assert "logical_role_mapping_ambiguous_or_unknown" in loader["mismatch_codes"]


def test_others_need_not_have_an_exclusive_anchor() -> None:
    result = diagnostic.analyze_payload(
        _payload(
            {
                "bn_vocab": ["shared", "ami"],
                "en_vocab": ["shared", "hello"],
                "universal_vocab": ["shared"],
            }
        )
    )
    assert result["exclusive_normalized_token_cardinalities"]["universal_vocab"] == 0
    assert result["current_loader_contract"]["repairable_key_mismatch"] is True


@pytest.mark.parametrize(
    "value, expected_code",
    [
        (["not", "an", "object"], "root_type_mismatch"),
        (
            {
                "first": ["clean"],
                "second": ["   "],
                "third": ["other"],
            },
            "field_shape_mismatch",
        ),
        (
            {
                "first": ["clean"],
                "second": [7],
                "third": ["other"],
            },
            "field_shape_mismatch",
        ),
        (
            {
                "first": ["clean"],
                "second": {"nested": "value"},
                "third": ["other"],
            },
            "field_shape_mismatch",
        ),
    ],
)
def test_unsupported_shapes_never_authorize_repair(value: object, expected_code: str) -> None:
    result = diagnostic.analyze_payload(_payload(value))
    loader = result["current_loader_contract"]
    assert loader["passes"] is False
    assert loader["repairable_key_mismatch"] is False
    assert expected_code in loader["mismatch_codes"]


def test_duplicate_root_keys_are_detected_before_collapse() -> None:
    result = diagnostic.analyze_payload(
        b'{"first":["one"],"first":["two"],"second":["three"],"third":["four"]}'
    )
    assert result["duplicate_top_level_key_count"] == 1
    assert "duplicate_top_level_keys" in result["current_loader_contract"]["mismatch_codes"]
    assert result["current_loader_contract"]["repairable_key_mismatch"] is False


def test_overlap_and_exclusive_cardinalities_use_frozen_tokenization() -> None:
    result = diagnostic.analyze_payload(
        _payload(
            {
                "first": ["Shared", "alpha-beta", "can't"],
                "second": ["shared", "gamma"],
                "third": ["shared", "delta"],
            }
        )
    )
    pairwise = {
        (item["left_schema_key"], item["right_schema_key"]): item[
            "intersection_cardinality"
        ]
        for item in result["pairwise_intersections"]
    }
    assert pairwise == {
        ("first", "second"): 1,
        ("first", "third"): 1,
        ("second", "third"): 1,
    }
    assert result["all_field_intersection_cardinality"] == 1
    assert result["exclusive_normalized_token_cardinalities"] == {
        "first": 2,
        "second": 1,
        "third": 1,
    }


def test_aggregate_evidence_never_serializes_fixture_lexical_values() -> None:
    lexical_values = ["privatealpha", "privatebeta", "privategamma"]
    result = diagnostic.analyze_payload(
        _payload(
            {
                "first": [lexical_values[0]],
                "second": [lexical_values[1]],
                "third": [lexical_values[2]],
            }
        )
    )
    serialized = json.dumps(result, sort_keys=True)
    assert all(value not in serialized for value in lexical_values)


def test_marker_is_durable_and_publication_is_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    diagnostic._write_exclusive(path, diagnostic.REPORT_MARKER)
    with pytest.raises(FileExistsError):
        diagnostic._write_exclusive(path, b"second")
    diagnostic._finish_marker(path, b'{"status":"PASS"}\n')
    assert path.read_bytes() == b'{"status":"PASS"}\n'


def test_marker_precedes_the_only_vocab_read_in_main() -> None:
    source = inspect.getsource(diagnostic.main)
    assert source.count("_read_vocab_once") == 1
    assert source.index("_write_exclusive(report_path, REPORT_MARKER)") < source.index(
        "_read_vocab_once"
    )
    opener = inspect.getsource(diagnostic._read_vocab_once)
    assert opener.count('.open("rb")') == 1
    assert opener.count("handle.read()") == 1


def test_authorization_hash_and_predecessor_marker_are_exact() -> None:
    assert diagnostic._sha256_bytes(diagnostic.AUTHORIZATION_TEXT.encode()) == (
        diagnostic.AUTHORIZATION_TEXT_SHA256
    )
    marker_path, marker_sha = diagnostic.PREDECESSOR_BINDINGS["runner_marker"]
    assert marker_path.endswith("curation-v0.5.json")
    assert marker_sha == "a0f713fcb3ad48e68c857a3481a94e7cfdde83f3a49a3a761c8ce927aa070142"


def _valid_report() -> dict[str, object]:
    aggregate = diagnostic.analyze_payload(
        _payload(
            {
                "bn_vocab": ["ami", "chai"],
                "en_vocab": ["hello", "need"],
                "universal_vocab": ["shared"],
            }
        )
    )
    return diagnostic.build_report(aggregate, diagnostic.build_bindings())


def _reseal_report(report: dict[str, object]) -> None:
    report["report_sha256"] = diagnostic._internal_sha(report, "report_sha256")


def test_report_rejects_resealed_predecessor_binding_tamper() -> None:
    report = copy.deepcopy(_valid_report())
    report["bindings"]["predecessor"]["runner_marker"]["sha256"] = "0" * 64
    _reseal_report(report)
    with pytest.raises(ValueError, match="predecessor binding"):
        diagnostic.validate_report(report)


def test_report_recomputes_mapping_and_conditional_gate() -> None:
    report = copy.deepcopy(_valid_report())
    report["diagnostic"]["current_loader_contract"]["repairable_key_mismatch"] = False
    report["conditional_recovery_authorized"] = False
    report["status"] = "STOP_existing_or_unsupported_vocab_contract"
    _reseal_report(report)
    with pytest.raises(ValueError, match="loader evidence"):
        diagnostic.validate_report(report)


def test_report_rejects_resealed_logical_mapping_tamper() -> None:
    report = copy.deepcopy(_valid_report())
    report["diagnostic"]["logical_role_mapping"]["english"] = "bn_vocab"
    _reseal_report(report)
    with pytest.raises(ValueError, match="logical-role mapping"):
        diagnostic.validate_report(report)


def test_verification_is_linked_to_exact_report_payload_and_gate() -> None:
    report = _valid_report()
    report_file_sha = diagnostic._sha256_bytes(diagnostic._json_bytes(report))
    verification = diagnostic.build_verification(report, report_file_sha)
    diagnostic.validate_verification(verification, report, report_file_sha)

    tampered = copy.deepcopy(verification)
    tampered["conditional_recovery_authorized"] = False
    tampered["verification_sha256"] = diagnostic._internal_sha(
        tampered, "verification_sha256"
    )
    with pytest.raises(ValueError):
        diagnostic.validate_verification(tampered, report, report_file_sha)

    with pytest.raises(ValueError):
        diagnostic.validate_verification(verification, report, "0" * 64)
