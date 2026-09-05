from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

import lumi.posttraining_bedstc2_banglish_conversation_curation_v5 as recovery_v5
import scripts.curate_sft_bedstc2_banglish_conversation_v5 as runner_v5
import scripts.verify_sft_bedstc2_banglish_conversation_v5 as verifier_v5
from lumi.posttraining_bedstc2_banglish_conversation_curation_v5 import (
    EXPECTED_AUTHORIZATION,
    EXPECTED_BASE_RECORDS,
    EXPECTED_BASE_TRAIN_TOKENS,
    EXPECTED_BASE_VALIDATION_TOKENS,
    EXPECTED_BINDING_PATHS,
    EXPECTED_COMBINED_SUCCESSOR_LIMITS,
    EXPECTED_DIALOGUES,
    EXPECTED_EMPTY_LEDGER_SHA256,
    EXPECTED_EVALUATION_PROHIBITION,
    EXPECTED_LINES,
    EXPECTED_MALFORMED_REASON_OCCURRENCES,
    EXPECTED_MALFORMED_SOURCE_ROWS_RANGE,
    EXPECTED_NEAR_POLICY,
    EXPECTED_PRIOR_CONSUMPTION,
    EXPECTED_VOCAB_CONTRACT_RECOVERY,
    FAMILY,
    ParsedDialogue,
    ParsedEntry,
    _assert_test_object_absent,
    candidate_rows,
    dual_anchor,
    json_bytes,
    load_vocab,
    parse_dialogues,
    render_owner_packet,
    sha256_value,
    validate_report,
    validate_verification,
)
from scripts.curate_sft_bedstc2_banglish_conversation_v5 import (
    RUNNER_MARKER,
    _start_marker,
    _write_exclusive,
    preflight_runner,
)
from scripts.verify_sft_bedstc2_banglish_conversation_v5 import (
    _preflight_then_replay,
    _validate_rejections,
    _validate_telemetry,
    preflight_first_pass_outputs,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _owner_packet() -> list[dict[str, object]]:
    return [
        {
            "language_mode": "banglish",
            "messages": [
                {
                    "content": (
                        "ami need <title>Lumi owner review</title> & </script> help please"
                    ),
                    "role": "user",
                },
                {"content": "ami can help please", "role": "assistant"},
            ],
            "record_id": f"record-{index:02d}",
            "skill_id": "conversation",
            "split": "train",
        }
        for index in range(32)
    ]


def _near(records: int) -> dict[str, object]:
    return {
        "comparison_unit": EXPECTED_NEAR_POLICY["comparison_unit"],
        "component_ledger_sha256": EXPECTED_EMPTY_LEDGER_SHA256,
        "components": records,
        "cross_source_components": 0,
        "duplicate_extra_records": 0,
        "edge_count": 0,
        "largest_component_records": 1,
        "multi_record_components": 0,
        "per_source_multi_record_components": {},
        "records": records,
        "size_histogram": {},
        "train_validation_components": 0,
        "train_validation_records": 0,
    }


def _valid_report(*, artifacts: dict[str, dict[str, object]] | None = None) -> dict:
    records = 32
    train_tokens = 60_000
    combined = EXPECTED_BASE_RECORDS + records
    candidate_hash = "1" * 64
    owner_file_hash = "2" * 64
    if artifacts is None:
        artifacts = {
            "candidate": {
                "bytes": 1,
                "contains_raw_training_text": True,
                "sha256": candidate_hash,
            },
            "owner_review_html": {
                "bytes": 1,
                "contains_raw_training_text": True,
                "sha256": "3" * 64,
            },
            "owner_review_packet": {
                "bytes": 1,
                "contains_raw_training_text": True,
                "sha256": owner_file_hash,
            },
            "rejection_ledger": {
                "bytes": 1,
                "contains_raw_training_text": False,
                "sha256": "4" * 64,
            },
            "telemetry": {
                "bytes": 1,
                "contains_raw_training_text": False,
                "sha256": "5" * 64,
            },
        }
    else:
        candidate_hash = str(artifacts["candidate"]["sha256"])
        owner_file_hash = str(artifacts["owner_review_packet"]["sha256"])
    language_gaps = {
        "banglish": 712_114,
        "bengali_script": 0,
        "english": 0,
        "natural_code_switching": 365_420,
    }
    task_gaps = {
        "coding": 0,
        "conversation": 0,
        "factual_qa": 114_306,
        "multilingual_bengali": 0,
        "reasoning_math_science": 0,
        "safety_helpfulness": 0,
        "tool_api": 2,
        "writing_summarization": 0,
    }
    report = {
        "artifacts": artifacts,
        "audit": {
            "accepted_metrics": {
                "family_train_supervised_tokens": {FAMILY: train_tokens},
                "language_train_supervised_tokens_accounting": {"banglish": train_tokens},
                "language_train_supervised_tokens_behavior_safe": {"banglish": train_tokens},
                "source_train_supervised_tokens": {"synthetic": train_tokens},
                "task_train_supervised_tokens_accounting": {"conversation": train_tokens},
                "task_train_supervised_tokens_behavior_safe": {"conversation": train_tokens},
                "train_supervised_tokens": EXPECTED_BASE_TRAIN_TOKENS + train_tokens,
                "validation_supervised_tokens": EXPECTED_BASE_VALIDATION_TOKENS,
            },
            "exact": {
                "cross_source_duplicate_groups": 0,
                "duplicate_extra_records": 0,
                "duplicate_groups": 0,
                "train_validation_overlap_groups": 0,
            },
            "generic": {
                "accepted": combined,
                "accepted_dataset_sha256": "6" * 64,
                "raw_text_retained": False,
                "rejected": 0,
                "rejection_counts": {},
                "report_sha256": "7" * 64,
                "schema_version": "lumi.sft.v0.1",
                "total": combined,
            },
            "near": _near(combined),
            "new_source_metrics": {
                "auxiliary_only_behavior_credit": True,
                "by_split": {
                    "train": {
                        "records": records,
                        "serialized_tokens": 64_000,
                        "supervised_tokens": train_tokens,
                    },
                    "validation": {
                        "records": 0,
                        "serialized_tokens": 0,
                        "supervised_tokens": 0,
                    },
                },
                "candidate_file_sha256": candidate_hash,
                "language_mode_policy": "language_field_multilingual_as_banglish",
                "maximum_record_tokens": 2_000,
                "records": records,
                "rows_with_pending_owner_review_provenance": records,
                "source_family": FAMILY,
                "train_by_language_mode": {"banglish": train_tokens},
                "train_by_task": {"conversation": train_tokens},
            },
            "task_language_family": {
                "family_cap_dilution_gap": 0,
                "language_floor_gaps_behavior_safe": language_gaps,
                "numeric_gap_to_10m": 0,
                "source_family_cap": 0.35,
                "source_family_cap_violations": [],
                "source_family_shares": {
                    FAMILY: round(train_tokens / (EXPECTED_BASE_TRAIN_TOKENS + train_tokens), 8)
                },
                "strict_targeted_gap_lower_bound": sum(language_gaps.values()),
                "task_floor_gaps_behavior_safe": task_gaps,
            },
        },
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {name: "8" * 64 for name in EXPECTED_BINDING_PATHS},
        "candidate": {
            "records": records,
            "train_records": records,
            "train_supervised_tokens": train_tokens,
            "validation_records": 0,
            "validation_supervised_tokens": 0,
        },
        "curation": {
            "constructed_candidate_rows": records,
            "generic_accepted_rows": records,
            "generic_rejected_rows": 0,
            "parsed_malformed_source_rows_rejected": EXPECTED_MALFORMED_SOURCE_ROWS_RANGE[0],
            "raw_text_free_rejection_records": 1,
            "retained_rows": records,
        },
        "evaluation_prohibition": EXPECTED_EVALUATION_PROHIBITION,
        "frozen_gate_results": {
            "assistant_only_masking_and_4k_fit": True,
            "benchmark_decontamination": True,
            "classified_body_malformed_intersection_recovery_reproduced": True,
            "exact_duplicate_and_split_overlap": True,
            "factual_grounding": True,
            "minhash_0_9": True,
            "observed_malformed_numbered_rows_rejected_without_normalization": True,
            "parser_contract_syntax_only_kb_classification": True,
            "pii_secret_and_safety": True,
            "punctuation_integrity": True,
            "rights_and_truthful_provenance": True,
            "roman_script_dual_language_anchor": True,
            "schema": True,
            "source_family_cap": True,
            "task_and_language_floors_measured": True,
            "vocab_exact_empty_placeholder_recovery_reproduced": True,
        },
        "owner_review": {
            "minimum_accepts": 29,
            "packet_file_sha256": owner_file_hash,
            "packet_records": 32,
            "packet_semantic_sha256": "9" * 64,
            "status": "PENDING_OWNER_REVIEW",
        },
        "parser_shape_rejections": {
            "reason_occurrences": EXPECTED_MALFORMED_REASON_OCCURRENCES,
            "source_text_normalized": False,
            "unique_numbered_rows": EXPECTED_MALFORMED_SOURCE_ROWS_RANGE[0],
        },
        "plan_file_sha256": "a" * 64,
        "prior_consumption": EXPECTED_PRIOR_CONSUMPTION,
        "production_blockers": [
            "upstream_dstc2_rights_chain_not_independently_verified",
            "owner_quality_review_not_complete",
            "source_promotion_and_full_cross_source_reaudit_not_authorized",
            "production_dataset_seal_and_sft_authorization_not_complete",
        ],
        "production_eligible": False,
        "provenance": (
            "human_in_house_annotated_source_derived_bengali_code_mixed_surface_"
            "wording_with_underlying_dstc2_authorship_and_origin_not_established"
        ),
        "raw_text_boundary": {
            "candidate": True,
            "owner_review_html": True,
            "owner_review_packet": True,
            "rejection_ledger": False,
            "report": False,
            "telemetry": False,
            "tests_and_logs": False,
            "verification": False,
        },
        "resource_accounting": {
            "existing_candidate_row_reads": EXPECTED_BASE_RECORDS,
            "full_passes": 1,
            "new_candidate_row_reads": records,
            "source_line_reads": EXPECTED_LINES["total"],
            "total_candidate_audit_row_reads": combined,
        },
        "rights": {
            "license": "apache-2.0",
            "upstream_dstc2_rights_chain_independently_verified": False,
        },
        "schema_version": "lumi.sft-bedstc2-banglish-conversation-curation-report.v0.6",
        "source_inventory": EXPECTED_DIALOGUES,
        "status": "PASS_nonproduction_bedstc2_candidate_owner_review_pending",
        "vocab_contract_recovery": EXPECTED_VOCAB_CONTRACT_RECOVERY,
    }
    report["report_sha256"] = sha256_value(report)
    return report


def _reseal(report: dict) -> None:
    report.pop("report_sha256", None)
    report["report_sha256"] = sha256_value(report)


def test_parser_classifies_kb_by_syntax_without_api_adjacency(tmp_path: Path) -> None:
    valid = tmp_path / "valid.txt"
    valid.write_text(
        "1 ami need please\tami can help please\n"
        "2 ami restaurant chai\tapi_call restaurant cheap\n"
        "3 rest R_cuisine bangla\n"
        "4 ami result chai\tami found it please\n\n",
        encoding="utf-8",
    )
    dialogues, lines = parse_dialogues(valid, object_name="synthetic.txt")
    assert len(dialogues) == 1
    assert lines == 5
    assert dialogues[0].entries[2].kind == "kb"
    assert dialogues[0].entries[2].body == "rest R_cuisine bangla"
    _, rejected = candidate_rows(
        dialogues,
        english=frozenset({"need", "please", "restaurant", "result", "found", "it"}),
        native=frozenset({"ami", "chai"}),
        others=frozenset({"rest"}),
    )
    reasons = {reason for item in rejected for reason in item["reason_codes"]}
    assert "api_call_only_assistant_response" in reasons
    assert "kb_only_record" in reasons

    non_adjacent = tmp_path / "non-adjacent.txt"
    non_adjacent.write_text(
        "1 ami need please\tapi_call restaurant cheap\n"
        "2 ami result chai\tami found it please\n"
        "3 rest R_post-code cb30qz\n\n",
        encoding="utf-8",
    )
    parsed, _ = parse_dialogues(non_adjacent, object_name="synthetic.txt")
    assert parsed[0].entries[-1].kind == "kb"
    assert parsed[0].entries[-1].body == "rest R_post-code cb30qz"

    diagnosed = tmp_path / "diagnosed.txt"
    diagnosed.write_text(
        "1 entity relation value\n2 ami need \tami can help please\n3 rest R_cuisine bangla \n\n",
        encoding="utf-8",
    )
    parsed, _ = parse_dialogues(diagnosed, object_name="synthetic.txt")
    assert [entry.kind for entry in parsed[0].entries] == [
        "malformed",
        "malformed",
        "malformed",
    ]
    assert parsed[0].entries[0].reason_codes == ("source_non_kb_numbered_row",)
    assert parsed[0].entries[1].reason_codes == ("source_pair_user_trailing_whitespace",)
    assert parsed[0].entries[2].reason_codes == ("source_outer_trailing_whitespace",)
    records, rejected = candidate_rows(
        parsed,
        english=frozenset({"need", "can", "help", "please"}),
        native=frozenset({"ami"}),
        others=frozenset({"rest"}),
    )
    assert records == []
    assert {reason for item in rejected for reason in item["reason_codes"]} == {
        "source_non_kb_numbered_row",
        "source_outer_trailing_whitespace",
        "source_pair_user_trailing_whitespace",
    }

    classified = tmp_path / "classified-body.txt"
    classified.write_text(
        "1 ami need please\tami can help please\t\n2 rest R_cuisine bangla\t\n\n",
        encoding="utf-8",
    )
    parsed, _ = parse_dialogues(classified, object_name="synthetic.txt")
    assert [entry.kind for entry in parsed[0].entries] == ["malformed", "malformed"]
    assert all(
        entry.reason_codes == ("source_outer_trailing_whitespace",) for entry in parsed[0].entries
    )
    assert parsed[0].entries[0].body.endswith("\t")
    assert parsed[0].entries[1].body.endswith("\t")

    invalid = tmp_path / "invalid.txt"
    invalid.write_text("1 entity R_\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed non-tab"):
        parse_dialogues(invalid, object_name="synthetic.txt")


def test_future_kb_cannot_ground_prior_response() -> None:
    dialogue = ParsedDialogue(
        entries=(
            ParsedEntry("ami need\tfoo ami please", "pair", 1, "ami need", "foo ami please"),
            ParsedEntry(
                "ami restaurant chai\tapi_call restaurant",
                "api",
                2,
                "ami restaurant chai",
                "api_call restaurant",
            ),
            ParsedEntry("foo R_cuisine bangla", "kb", 3),
        ),
        object_name="synthetic.txt",
        source_index=0,
    )
    records, rejected = candidate_rows(
        [dialogue],
        english=frozenset({"need", "please", "restaurant"}),
        native=frozenset({"ami", "chai"}),
        others=frozenset({"foo"}),
    )
    assert records == []
    assert any("ungrounded_entity_or_factual_response" in item["reason_codes"] for item in rejected)


def test_dual_anchor_is_bilateral_and_roman_only() -> None:
    english = frozenset({"need"})
    native = frozenset({"ami"})
    assert dual_anchor("ami need", english=english, native=native)
    assert not dual_anchor("ami", english=english, native=native)
    assert not dual_anchor("আমি ami need", english=english, native=native)


def _write_vocab(tmp_path: Path, value: object) -> Path:
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _synthetic_vocab(*, english: object, native: object, others: object) -> dict[str, object]:
    return {
        "english_language_vocab": english,
        "native_language_vocab": native,
        "others_vocab": others,
    }


def test_vocab_loader_skips_exactly_one_others_empty_placeholder(tmp_path: Path) -> None:
    path = _write_vocab(
        tmp_path,
        _synthetic_vocab(
            english=["Hello", "..."],
            native=["ami"],
            others=["", "shared", "!!!"],
        ),
    )
    english, native, others = load_vocab(path)
    assert english == frozenset({"hello"})
    assert native == frozenset({"ami"})
    assert others == frozenset({"shared"})


@pytest.mark.parametrize("others", [["shared"], ["", "", "shared"]])
def test_vocab_loader_requires_exactly_one_others_empty_placeholder(
    tmp_path: Path, others: list[str]
) -> None:
    path = _write_vocab(
        tmp_path,
        _synthetic_vocab(english=["hello"], native=["ami"], others=others),
    )
    with pytest.raises(ValueError, match="empty-placeholder count"):
        load_vocab(path)


@pytest.mark.parametrize(
    "english,native",
    [(["", "hello"], ["ami"]), (["hello"], ["", "ami"])],
)
def test_vocab_loader_rejects_empty_english_or_native(
    tmp_path: Path, english: list[str], native: list[str]
) -> None:
    path = _write_vocab(
        tmp_path,
        _synthetic_vocab(english=english, native=native, others=["", "shared"]),
    )
    with pytest.raises(ValueError, match="empty-placeholder count"):
        load_vocab(path)


@pytest.mark.parametrize(
    "english,native,others",
    [
        ([" hello"], ["ami"], ["", "shared"]),
        (["hello"], ["ami"], ["", "shared "]),
        (["hello"], [7], ["", "shared"]),
        (["..."], ["ami"], ["", "!!!"]),
    ],
)
def test_vocab_loader_preserves_other_fail_closed_item_gates(
    tmp_path: Path, english: object, native: object, others: object
) -> None:
    path = _write_vocab(
        tmp_path,
        _synthetic_vocab(english=english, native=native, others=others),
    )
    with pytest.raises(ValueError):
        load_vocab(path)


def test_vocab_loader_rejects_missing_extra_or_alias_keys(tmp_path: Path) -> None:
    for value in (
        {"english_language_vocab": ["hello"], "native_language_vocab": ["ami"]},
        {
            **_synthetic_vocab(english=["hello"], native=["ami"], others=["", "shared"]),
            "extra": [],
        },
        {"en_vocab": ["hello"], "bn_vocab": ["ami"], "others_vocab": ["", "shared"]},
    ):
        with pytest.raises(ValueError, match="inventory"):
            load_vocab(_write_vocab(tmp_path, value))


def test_unicode_owner_hash_domains_and_static_html_integrity() -> None:
    packet = _owner_packet()
    packet_payload, html_payload, semantic, file_hash = render_owner_packet(packet)
    assert semantic != file_hash
    assert _sha(packet_payload) == file_hash
    assert json.loads(packet_payload) == packet
    html = html_payload.decode()
    assert "lumi.ai-assisted-owner-decisions.v0.3" in html
    assert "lumi-owner-review-decisions-v0.3.json" in html
    assert "<title>Lumi Be-DSTC2 owner review</title>" in html
    assert "<title>Lumi owner review</title> & </script>" not in html
    assert "\\u003ctitle\\u003eLumi owner review\\u003c/title\\u003e" in html


def test_report_and_verification_are_strict_and_hash_domains_distinct() -> None:
    report = _valid_report()
    validate_report(report)
    broken = copy.deepcopy(report)
    broken["frozen_gate_results"]["factual_grounding"] = False
    _reseal(broken)
    with pytest.raises(ValueError, match="gate"):
        validate_report(broken)
    broken = copy.deepcopy(report)
    broken["owner_review"]["packet_semantic_sha256"] = broken["owner_review"]["packet_file_sha256"]
    _reseal(broken)
    with pytest.raises(ValueError, match="owner"):
        validate_report(broken)

    from lumi.posttraining_bedstc2_banglish_conversation_curation_v5 import (
        build_verification,
    )

    verification = build_verification(report=report, report_file_sha256="b" * 64)
    validate_verification(verification)
    verification["resource_accounting"]["new_candidate_row_reads"] += 1
    verification.pop("verification_sha256")
    verification["verification_sha256"] = sha256_value(verification)
    with pytest.raises(ValueError, match="resource"):
        validate_verification(verification)


def test_existing_output_or_marker_blocks_runner_before_replay(tmp_path: Path) -> None:
    paths = {name: tmp_path / name for name in ("candidate", "report", "verification")}
    preflight_runner(root=tmp_path, paths=paths)
    _start_marker(paths["report"], payload=RUNNER_MARKER)
    with pytest.raises(FileExistsError):
        preflight_runner(root=tmp_path, paths=paths)


def test_authorized_final_publication_is_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "authorized-output.json"
    _write_exclusive(path, b"first\n")
    with pytest.raises(FileExistsError):
        _write_exclusive(path, b"second\n")
    assert path.read_bytes() == b"first\n"


def test_telemetry_preflight_counts_without_candidate_row_parse() -> None:
    telemetry_rows = [
        {
            "record_id": "r1",
            "serialized_tokens": 10,
            "split": "train",
            "supervised_tokens": 4,
        }
    ]
    report = {
        "candidate": {
            "records": 1,
            "train_records": 1,
            "validation_records": 0,
            "train_supervised_tokens": 4,
            "validation_supervised_tokens": 0,
        }
    }
    telemetry_ids = _validate_telemetry(
        b"".join(canonical_line(row) for row in telemetry_rows), report
    )
    assert telemetry_ids == {"r1": "train"}
    telemetry_rows[0]["split"] = "validation"
    with pytest.raises(ValueError):
        _validate_telemetry(b"".join(canonical_line(row) for row in telemetry_rows), report)


def test_rejection_preflight_binds_diagnosed_reason_occurrences() -> None:
    rows = [
        {
            "reason_codes": ["source_non_kb_numbered_row"],
            "record_id": f"non-kb-{index:03d}",
        }
        for index in range(421)
    ]
    rows.extend(
        {
            "reason_codes": ["source_outer_trailing_whitespace"],
            "record_id": f"outer-{index:03d}",
        }
        for index in range(17)
    )
    rows.extend(
        {
            "reason_codes": ["source_pair_user_trailing_whitespace"],
            "record_id": f"pair-{index:03d}",
        }
        for index in range(37)
    )
    report = {
        "curation": {"raw_text_free_rejection_records": 475},
        "parser_shape_rejections": {
            "reason_occurrences": EXPECTED_MALFORMED_REASON_OCCURRENCES,
            "source_text_normalized": False,
            "unique_numbered_rows": 475,
        },
    }
    payload = b"".join(canonical_line(row) for row in rows)
    _validate_rejections(payload, report)
    rows[0]["reason_codes"] = ["different_reason"]
    with pytest.raises(ValueError, match="parser-shape"):
        _validate_rejections(b"".join(canonical_line(row) for row in rows), report)


def test_verifier_preflight_failure_never_calls_replay(tmp_path: Path) -> None:
    packet_payload, html_payload, _, _ = render_owner_packet(_owner_packet())
    telemetry_rows = [
        {
            "record_id": f"record-{index:02d}",
            "serialized_tokens": 2_000,
            "split": "train",
            "supervised_tokens": 1_875,
        }
        for index in range(32)
    ]
    telemetry_payload = b"".join(canonical_line(row) for row in telemetry_rows)
    rejection_payload = canonical_line(
        {"reason_codes": ["synthetic_rejection"], "record_id": "rejected-00"}
    )
    payloads = {
        "candidate": b"private-candidate\n",
        "owner_review_html": html_payload,
        "owner_review_packet": packet_payload,
        "rejection_ledger": rejection_payload,
        "telemetry": telemetry_payload,
    }
    artifact_evidence = {
        name: {
            "bytes": len(payload),
            "contains_raw_training_text": name
            in {"candidate", "owner_review_html", "owner_review_packet"},
            "sha256": _sha(payload),
        }
        for name, payload in payloads.items()
    }
    report = _valid_report(artifacts=artifact_evidence)
    plan = {
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {name: {"sha256": "8" * 64} for name in EXPECTED_BINDING_PATHS},
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}\n", encoding="utf-8")
    report["plan_file_sha256"] = _sha(plan_path.read_bytes())
    report["owner_review"]["packet_file_sha256"] = _sha(packet_payload)
    report["owner_review"]["packet_semantic_sha256"] = hashlib.sha256(
        json.dumps(
            _owner_packet(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode()
    ).hexdigest()
    _reseal(report)
    paths = {name: tmp_path / name for name in (*payloads, "report", "verification")}
    for name, payload in payloads.items():
        paths[name].write_bytes(payload)
    paths["report"].write_bytes(json_bytes(report))

    paths["owner_review_packet"].write_bytes(packet_payload + b" ")
    replay_called = False

    def replay() -> None:
        nonlocal replay_called
        replay_called = True

    with pytest.raises(ValueError):
        _preflight_then_replay(
            lambda: preflight_first_pass_outputs(plan=plan, plan_path=plan_path, paths=paths),
            replay,
        )
    assert replay_called is False


def canonical_line(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        + b"\n"
    )


def test_corrected_inventory_and_two_pass_caps_are_exact() -> None:
    assert EXPECTED_DIALOGUES == {"dev": 500, "total": 2_118, "train": 1_618}
    assert EXPECTED_LINES == {"dev": 28_617, "total": 113_718, "train": 85_101}
    assert 2 * EXPECTED_LINES["total"] == 227_436
    assert 2 * EXPECTED_BASE_RECORDS == 629_432
    assert EXPECTED_COMBINED_SUCCESSOR_LIMITS == {
        "maximum_all_candidate_audit_row_reads": 669_432,
        "maximum_existing_candidate_row_reads": 629_432,
        "maximum_gpu_use": 0,
        "maximum_inference_calls": 0,
        "maximum_new_candidate_row_reads": 40_000,
        "maximum_network_bytes": 0,
        "maximum_private_write_bytes": 134_283_264,
        "maximum_runtime_minutes": 65,
        "maximum_source_line_reads": 227_436,
    }
    assert EXPECTED_PRIOR_CONSUMPTION[
        "v0_7_malformed_intersection_diagnostic_success"
    ]["source_line_reads"] == 113_718
    assert "source_line_reads_conservative_upper_bound" not in EXPECTED_PRIOR_CONSUMPTION[
        "v0_7_malformed_intersection_diagnostic_success"
    ]
    assert EXPECTED_PRIOR_CONSUMPTION["v0_5_vocab_contract_failure"][
        "source_line_reads_conservative_upper_bound"
    ] == 113_718
    assert "source_line_reads" not in EXPECTED_PRIOR_CONSUMPTION[
        "v0_5_vocab_contract_failure"
    ]


def test_exact_vocab_override_authorization_and_markers_are_frozen() -> None:
    text = recovery_v5.EXPECTED_VOCAB_RECOVERY_AUTHORIZATION_TEXT
    assert len(text.encode()) == 2_278
    assert _sha(text.encode()) == recovery_v5.EXPECTED_VOCAB_RECOVERY_AUTHORIZATION_TEXT_SHA256
    assert recovery_v5.EXPECTED_VOCAB_CONTRACT_RECOVERY["empty_string_items_by_role"] == {
        "english": 0,
        "native": 0,
        "others": 1,
    }
    assert RUNNER_MARKER == b'{"started":"v0.6_runner_pass_one_attempt_005"}\n'
    assert _sha(RUNNER_MARKER) == (
        "237ab60e7c92aaeb713c8c447d1f0c3367bbc038fb90e46f543897ea43d72b89"
    )
    assert runner_v5.VERIFIER_MARKER == (b'{"started":"v0.6_verifier_pass_two_attempt_005"}\n')
    assert _sha(runner_v5.VERIFIER_MARKER) == (
        "b06d0f508ac52dfe7315dbc52d3db7b9d7a7e52dd295b218c42284e2f7f90732"
    )


def test_vocab_diagnostic_stop_false_override_validation_is_strict(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / recovery_v5.EXPECTED_BINDING_PATHS["vocab_diagnostic_report"]
    verification_path = (
        tmp_path / recovery_v5.EXPECTED_BINDING_PATHS["vocab_diagnostic_verification"]
    )
    report_path.parent.mkdir(parents=True)
    report = {
        "authorization_text_sha256": (
            "b9f2d20769d3c247954d4eeea32bf81114f0a18c6d2185f4839730c4542b7f64"
        ),
        "conditional_recovery_authorized": False,
        "diagnostic": {
            "all_field_intersection_cardinality": 1,
            "current_loader_contract": {
                "expected_top_level_keys": [
                    "english_language_vocab",
                    "native_language_vocab",
                    "others_vocab",
                ],
                "logical_role_mapping_unambiguous": True,
                "mismatch_codes": ["field_shape_mismatch"],
                "passes": False,
                "repairable_key_mismatch": False,
            },
            "exclusive_normalized_token_cardinalities": {
                "english_language_vocab": 319,
                "native_language_vocab": 450,
                "others_vocab": 217,
            },
            "field_evidence": [
                {
                    "empty_string_items": 0,
                    "item_count": 360,
                    "item_type_counts": {"string": 360},
                    "json_type": "array",
                    "no_lexical_token_string_items": 1,
                    "normalized_token_occurrences": 363,
                    "normalized_unique_token_cardinality": 357,
                    "outer_whitespace_string_items": 0,
                    "schema_key": "english_language_vocab",
                },
                {
                    "empty_string_items": 0,
                    "item_count": 477,
                    "item_type_counts": {"string": 477},
                    "json_type": "array",
                    "no_lexical_token_string_items": 0,
                    "normalized_token_occurrences": 477,
                    "normalized_unique_token_cardinality": 477,
                    "outer_whitespace_string_items": 0,
                    "schema_key": "native_language_vocab",
                },
                {
                    "empty_string_items": 1,
                    "item_count": 535,
                    "item_type_counts": {"string": 535},
                    "json_type": "array",
                    "no_lexical_token_string_items": 4,
                    "normalized_token_occurrences": 1_589,
                    "normalized_unique_token_cardinality": 281,
                    "outer_whitespace_string_items": 0,
                    "schema_key": "others_vocab",
                },
            ],
            "logical_role_mapping": EXPECTED_VOCAB_CONTRACT_RECOVERY["role_mapping"],
            "pairwise_intersections": [
                {
                    "intersection_cardinality": 1,
                    "left_schema_key": "english_language_vocab",
                    "right_schema_key": "native_language_vocab",
                },
                {
                    "intersection_cardinality": 38,
                    "left_schema_key": "english_language_vocab",
                    "right_schema_key": "others_vocab",
                },
                {
                    "intersection_cardinality": 27,
                    "left_schema_key": "native_language_vocab",
                    "right_schema_key": "others_vocab",
                },
            ],
            "root_json_type": "object",
            "top_level_keys": [
                "english_language_vocab",
                "native_language_vocab",
                "others_vocab",
            ],
        },
        "raw_text_retained": False,
        "schema_version": "lumi.sft-bedstc2-vocab-contract-diagnostic.v0.8",
        "status": "STOP_existing_or_unsupported_vocab_contract",
    }
    verification = {
        "checks": {"aggregate_contract_verified": True},
        "conditional_recovery_authorized": False,
        "report_file_sha256": recovery_v5.EXPECTED_STABLE_BINDING_SHA256[
            "vocab_diagnostic_report"
        ],
        "report_internal_sha256": recovery_v5.EXPECTED_INTERNAL_BINDINGS[
            "vocab_diagnostic_report"
        ][1],
        "schema_version": (
            "lumi.sft-bedstc2-vocab-contract-diagnostic-verification.v0.8"
        ),
        "status": "STOP_structurally_verified_no_conditional_recovery",
    }
    bindings = {
        "vocab_diagnostic_report": {
            "path": recovery_v5.EXPECTED_BINDING_PATHS["vocab_diagnostic_report"]
        },
        "vocab_diagnostic_verification": {
            "path": recovery_v5.EXPECTED_BINDING_PATHS[
                "vocab_diagnostic_verification"
            ]
        },
    }

    def write_evidence(current_report: dict, current_verification: dict) -> None:
        report_path.write_bytes(json_bytes(current_report))
        verification_path.write_bytes(json_bytes(current_verification))

    write_evidence(report, verification)
    recovery_v5._verify_vocab_diagnostic(tmp_path, bindings)

    for mutate in (
        lambda value: value.update({"conditional_recovery_authorized": True}),
        lambda value: value.update({"status": "PASS"}),
        lambda value: value["diagnostic"]["field_evidence"][2].update(
            {"empty_string_items": 0}
        ),
        lambda value: value["diagnostic"].update(
            {"logical_role_mapping": {"others_vocab": "english"}}
        ),
    ):
        tampered_report = copy.deepcopy(report)
        mutate(tampered_report)
        write_evidence(tampered_report, verification)
        with pytest.raises(ValueError):
            recovery_v5._verify_vocab_diagnostic(tmp_path, bindings)

    tampered_verification = copy.deepcopy(verification)
    tampered_verification["conditional_recovery_authorized"] = True
    write_evidence(report, tampered_verification)
    with pytest.raises(ValueError):
        recovery_v5._verify_vocab_diagnostic(tmp_path, bindings)


def test_runner_and_verifier_each_call_build_once_with_fresh_paths() -> None:
    runner_source = inspect.getsource(runner_v5.main)
    verifier_source = inspect.getsource(verifier_v5.main)
    assert runner_source.count("build_artifacts(") == 1
    assert verifier_source.count("build_artifacts(") == 1
    assert runner_source.index("_start_marker") < runner_source.index("build_artifacts(")
    assert verifier_source.index("preflight_first_pass_outputs") < verifier_source.index(
        "_start_marker"
    )
    assert all(
        "attempt-005" in path or "v0.6.json" in path
        for path in recovery_v5.EXPECTED_OUTPUT_PATHS.values()
    )


def test_parser_uses_classified_body_for_tab_shape() -> None:
    source = inspect.getsource(recovery_v5.parse_dialogues)
    assert 'tabs = classified_body.count("\\t")' in source
    assert 'user, assistant = classified_body.split("\\t")' in source
    assert 'tabs = body.count("\\t")' not in source


def test_prohibited_test_object_presence_fails_metadata_only(tmp_path: Path) -> None:
    _assert_test_object_absent(tmp_path)
    prohibited = tmp_path / "data/bengali/dialog-dstc2-tst.txt"
    prohibited.parent.mkdir(parents=True)
    prohibited.touch()
    with pytest.raises(ValueError, match="test object"):
        _assert_test_object_absent(tmp_path)


def test_raw_content_verification_is_after_durable_runner_marker() -> None:
    load_source = inspect.getsource(recovery_v5.load_plan)
    build_source = inspect.getsource(recovery_v5.build_artifacts)
    runner_source = inspect.getsource(runner_v5.main)
    assert "_verify_raw_objects" not in load_source
    assert "_verify_raw_object_metadata" in load_source
    assert "_verify_raw_objects" in build_source
    assert runner_source.index("_start_marker") < runner_source.index("build_artifacts")


@pytest.mark.parametrize("script", [runner_v5, verifier_v5])
def test_entrypoint_sanitizes_exception_text(monkeypatch, capsys, script) -> None:
    def fail() -> int:
        raise ValueError("PRIVATE_SOURCE_SENTINEL")

    monkeypatch.setattr(script, "main", fail)
    assert script._entrypoint() == 1
    captured = capsys.readouterr()
    assert "PRIVATE_SOURCE_SENTINEL" not in captured.out + captured.err
    assert '"status": "FAIL_CLOSED"' in captured.err
