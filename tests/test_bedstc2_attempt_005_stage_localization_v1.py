from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import traceback
from pathlib import Path

import pytest

import scripts.diagnose_sft_bedstc2_attempt_005_stage_localization_v1 as diagnostic


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _synthetic_plan() -> dict:
    return {
        "bindings": {
            name: {"bytes": 1, "path": path, "sha256": "a" * 64}
            for name, path in diagnostic.EXPECTED_BINDING_PATHS.items()
        }
    }


def _empty_aggregates() -> dict[str, int | None]:
    return {key: None for key in sorted(diagnostic.AGGREGATE_KEYS)}


def _empty_rejections() -> dict[str, int]:
    return {reason: 0 for reason in diagnostic.REJECTION_CODES}


def _selection_floor_aggregates() -> dict[str, int | None]:
    values = _empty_aggregates()
    values.update(
        {
            "baseline_rows_loaded": 314_716,
            "baseline_sources_loaded": 17,
            "candidate_rows_proposed": 1_000,
            "candidate_rows_rejected_before_selection": 575,
            "dev_dialogues": 500,
            "dev_source_lines": 28_617,
            "dialogues_total": 2_118,
            "generic_accepted_rows": 900,
            "generic_rejected_rows": 100,
            "malformed_source_rows": 475,
            "new_rows_processed_by_selection": 900,
            "selected_rows": 100,
            "selected_train_rows": 95,
            "selected_validation_rows": 5,
            "source_lines_total": 113_718,
            "train_dialogues": 1_618,
            "train_source_lines": 85_101,
            "train_supervised_tokens": 59_999,
            "validation_supervised_tokens": 1_000,
            "vocab_english_unique_tokens": 319,
            "vocab_native_unique_tokens": 450,
            "vocab_others_unique_tokens": 281,
        }
    )
    return values


def _localized_report() -> dict:
    return diagnostic.build_report(
        plan=_synthetic_plan(),
        plan_file_sha256="b" * 64,
        localized_stage="baseline_first_selection",
        localized_gate="selection_train_token_floor",
        build_returned=False,
        post_build_checks_completed=False,
        aggregates=_selection_floor_aggregates(),
        rejection_counts=_empty_rejections(),
        unknown_rejection_reason_occurrences=0,
        runtime_seconds=1.0,
    )


def _raw_failure_report() -> dict:
    return diagnostic.build_report(
        plan=_synthetic_plan(),
        plan_file_sha256="b" * 64,
        localized_stage="raw_object_verification",
        localized_gate="raw_object_binding_or_content",
        build_returned=False,
        post_build_checks_completed=False,
        aggregates=_empty_aggregates(),
        rejection_counts=_empty_rejections(),
        unknown_rejection_reason_occurrences=0,
        runtime_seconds=1.0,
    )


def _synthetic_v5_error(*, callee: str, build_call_line: int) -> BaseException:
    lines = [
        f"def {callee}():",
        "    raise ValueError('raw-secret-sentinel')",
        "",
        "def build_artifacts():",
    ]
    while len(lines) < build_call_line - 1:
        lines.append("")
    lines.append(f"    {callee}()")
    lines.extend(("", "build_artifacts()"))
    namespace: dict[str, object] = {}
    try:
        exec(compile("\n".join(lines), diagnostic.v5.__file__, "exec"), namespace)
    except ValueError as error:
        return error
    raise AssertionError("synthetic failure did not run")


def test_authorization_hash_paths_marker_and_caps_are_exact() -> None:
    assert len(diagnostic.AUTHORIZATION_TEXT.encode()) == 1_376
    assert _sha(diagnostic.AUTHORIZATION_TEXT.encode()) == (
        "c9083b5502f4cbdcee4276fb33543edca704f3b73619be73b20456ac94bcc064"
    )
    assert diagnostic.AUTHORIZATION_TEXT_SHA256 == _sha(
        diagnostic.AUTHORIZATION_TEXT.encode()
    )
    assert diagnostic.EXPECTED_OUTPUT_PATHS == {
        "report": (
            "evidence/lumi-4b-sft-bedstc2-attempt-005-stage-localization-"
            "diagnostic-v0.9.json"
        ),
        "verification": (
            "evidence/lumi-4b-sft-bedstc2-attempt-005-stage-localization-"
            "diagnostic-verification-v0.9.json"
        ),
    }
    assert diagnostic.EXPECTED_LIMITS == {
        "maximum_existing_candidate_row_reads": 314_716,
        "maximum_gpu_use": 0,
        "maximum_inference_calls": 0,
        "maximum_new_candidate_row_reads": 20_000,
        "maximum_network_bytes": 0,
        "maximum_private_write_bytes": 1_048_576,
        "maximum_runtime_minutes": 30,
        "maximum_source_line_reads": 113_718,
        "maximum_total_candidate_audit_row_reads": 334_716,
        "passes": 1,
    }
    assert diagnostic.REPORT_MARKER.endswith(b"\n")


def test_predecessor_hashes_and_historical_failure_are_frozen() -> None:
    assert diagnostic.PREDECESSOR_BINDINGS == {
        "predecessor_implementation": (
            "src/lumi/posttraining_bedstc2_banglish_conversation_curation_v5.py",
            "75a140d0de68b0a956c090f4677acefd929d9135130d279fb79909f119e08cdf",
        ),
        "predecessor_plan": (
            "configs/lumi-4b-sft-bedstc2-banglish-conversation-curation-recovery-v0.6.json",
            "43c6f895870cb0a364fbb759d5e7dc6bf9dff34064dd0742751db1e7e378256c",
        ),
        "predecessor_runner": (
            "scripts/curate_sft_bedstc2_banglish_conversation_v5.py",
            "80159335cd14e29e668f46bd57a050008b6582aa5355e3aa9206903854df2555",
        ),
        "predecessor_runner_marker": (
            "evidence/lumi-4b-sft-bedstc2-banglish-conversation-curation-v0.6.json",
            "237ab60e7c92aaeb713c8c447d1f0c3367bbc038fb90e46f543897ea43d72b89",
        ),
        "predecessor_test": (
            "tests/test_bedstc2_banglish_conversation_curation_v5.py",
            "52442f05001dfbfc83d90a0755fc20df3b0d4d445ca82e5fd7d732d91a09c5e9",
        ),
        "predecessor_verifier": (
            "scripts/verify_sft_bedstc2_banglish_conversation_v5.py",
            "ae6424a92e5e59f1f59f3135be059e2881824b8d188145c2660bf20d260c59b2",
        ),
    }
    assert diagnostic.EXPECTED_HISTORICAL_FAILURE[
        "source_line_reads_conservative_upper_bound"
    ] == 113_718
    assert diagnostic.EXPECTED_HISTORICAL_FAILURE["private_write_bytes"] == 47
    assert diagnostic.EXPECTED_HISTORICAL_FAILURE["verifier_invoked"] is False


def test_exactly_one_build_call_and_marker_precedes_replay() -> None:
    replay_source = inspect.getsource(diagnostic._execute_once)
    main_source = inspect.getsource(diagnostic.main)
    assert replay_source.count("v5.build_artifacts(") == 1
    replay_tree = ast.parse(replay_source)
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "build_artifacts"
        for node in ast.walk(replay_tree)
    ) == 1
    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(replay_tree))
    assert "runner_v5.main(" not in replay_source
    assert "verifier_v5.main(" not in replay_source
    assert main_source.index("_write_exclusive") < main_source.index("_execute_once")
    assert main_source.index("_finish_marker") < main_source.index(
        '_write_exclusive(paths["verification"]'
    )


def test_closed_line_map_localizes_selection_floor() -> None:
    assert diagnostic._EXACT_LINE_GATES[("_select_baseline_first", 1345)] == (
        "selection_train_token_floor"
    )
    assert diagnostic.GATE_TO_STAGE["selection_train_token_floor"] == (
        "baseline_first_selection"
    )
    assert diagnostic._build_line_gate(2107) == (
        "baseline_first_selection",
        "unclassified_value_error",
    )
    assert diagnostic._build_line_gate(2041) == ("source_parse_dev", "source_parse_dev")
    assert diagnostic._build_line_gate(2130) == (
        "serialization_in_memory",
        "serialization_in_memory",
    )
    assert diagnostic._build_line_gate(2132) == (
        "combined_audit",
        "unclassified_value_error",
    )
    assert diagnostic._build_line_gate(2243) == ("report_validation", "report_validation")


@pytest.mark.parametrize(
    ("callee", "build_line", "expected"),
    [
        ("parse_dialogues", 2041, ("source_parse_dev", "source_parse_dev")),
        (
            "_jsonl_bytes",
            2130,
            ("serialization_in_memory", "serialization_in_memory"),
        ),
        (
            "unknown_generic_helper",
            2086,
            ("new_generic_filter", "new_generic_filter"),
        ),
    ],
)
def test_full_traceback_classifier_uses_bound_build_call_line(
    callee: str, build_line: int, expected: tuple[str, str]
) -> None:
    error = _synthetic_v5_error(callee=callee, build_call_line=build_line)
    tb = error.__traceback__
    try:
        assert diagnostic._classify_failure(error) == expected
    finally:
        if tb is not None:
            traceback.clear_frames(tb)


def test_baseline_loop_index_is_not_charged_as_new_candidate_progress() -> None:
    padding = [""] * (1_277 - 8)
    source = "\n".join(
        [
            "def _select_baseline_first():",
            "    baseline = [{}, {}, {}]",
            "    proposed = [{}, {}]",
            "    index = 2",
            "    selected = None",
            "    totals = None",
            "    rejected = []",
            *padding,
            "    raise ValueError('raw-secret-sentinel')",
            "",
            "_select_baseline_first()",
        ]
    )
    namespace: dict[str, object] = {}
    try:
        exec(compile(source, diagnostic.v5.__file__, "exec"), namespace)
    except ValueError as error:
        tb = error.__traceback__
        try:
            aggregates, _, _ = diagnostic._extract_safe_aggregates(tb)
            assert aggregates["baseline_rows_loaded"] == 3
            assert aggregates["new_rows_processed_by_selection"] == 0
        finally:
            if tb is not None:
                traceback.clear_frames(tb)
    else:
        raise AssertionError("synthetic selection failure did not run")


def test_traceback_aggregate_extraction_retains_no_ids_or_exception_text() -> None:
    prefix = [
        "def _select_baseline_first():",
        "    baseline = [{'record_id': 'baseline-secret'}] * 3",
        "    proposed = [{'record_id': 'candidate-secret', 'split': 'train'}] * 2",
        "    selected = [{'record_id': 'selected-secret', 'split': 'train'}]",
        "    totals = {'train': 59999, 'validation': 7}",
        "    rejected = [{'reason_codes': ['tokenizer_or_4096_fit', "
        "'unseen-secret-reason'], 'record_id': 'rejected-secret'}]",
        "    index = 1",
    ]
    source = "\n".join(
        [
            *prefix,
            *([""] * (1_300 - len(prefix) - 1)),
            "    raise ValueError('raw-source-sentinel')",
            "",
            "_select_baseline_first()",
        ]
    )
    namespace: dict[str, object] = {}
    try:
        exec(compile(source, diagnostic.v5.__file__, "exec"), namespace)
    except ValueError as error:
        tb = error.__traceback__
        aggregates, reasons, unknown = diagnostic._extract_safe_aggregates(tb)
        serialized = json.dumps(
            {"aggregates": aggregates, "reasons": reasons, "unknown": unknown},
            sort_keys=True,
        )
        assert aggregates["baseline_rows_loaded"] == 3
        assert aggregates["generic_accepted_rows"] == 2
        assert aggregates["selected_rows"] == 1
        assert aggregates["train_supervised_tokens"] == 59_999
        assert aggregates["validation_supervised_tokens"] == 7
        assert aggregates["new_rows_processed_by_selection"] == 2
        assert reasons["tokenizer_or_4096_fit"] == 1
        assert unknown == 1
        assert "secret" not in serialized
        assert "sentinel" not in serialized
        if tb is not None:
            traceback.clear_frames(tb)
    else:
        raise AssertionError("synthetic proposed-loop failure did not run")


def test_rejection_counter_never_serializes_unknown_reason_names() -> None:
    counts, unknown = diagnostic._reason_counts(
        [
            {
                "reason_codes": ["kb_only_record", "raw-secret-reason"],
                "record_id": "raw-secret-id",
            }
        ]
    )
    payload = json.dumps({"counts": counts, "unknown": unknown})
    assert counts["kb_only_record"] == 1
    assert unknown == 1
    assert "raw-secret" not in payload


def test_localized_report_is_raw_free_and_strict() -> None:
    report = _localized_report()
    diagnostic.validate_report(report)
    payload = diagnostic._json_bytes(report)
    diagnostic.validate_report(json.loads(payload))
    assert b"record_id" not in payload
    assert b"messages" not in payload
    assert b"exception" not in payload
    assert report["status"] == "PASS_localized_failure_gate"
    assert report["replay"]["localized_gate"] == "selection_train_token_floor"
    assert report["stage_results"]["baseline_first_selection"] == "failed"
    assert report["stage_results"]["owner_artifacts_in_memory"] == "not_run"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("raw_text_retained",), True),
        (("replay", "localized_gate"), "prior_failure_not_reproduced"),
        (("aggregates", "selected_rows"), -1),
        (("aggregates", "selected_rows"), 101),
        (("aggregates", "source_lines_total"), 113_717),
        (("resource_accounting", "source_line_reads_upper_bound"), 113_719),
        (("stage_results", "owner_artifacts_in_memory"), "completed"),
    ],
)
def test_report_tamper_fails_even_after_reseal(
    path: tuple[str, ...], value: object
) -> None:
    report = copy.deepcopy(_localized_report())
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    report["report_sha256"] = diagnostic._internal_sha(report, "report_sha256")
    with pytest.raises(ValueError):
        diagnostic.validate_report(report)


def test_resealed_future_stage_aggregate_is_rejected() -> None:
    report = _raw_failure_report()
    report["aggregates"]["selected_rows"] = 5
    report["report_sha256"] = diagnostic._internal_sha(report, "report_sha256")
    with pytest.raises(ValueError):
        diagnostic.validate_report(report)


def test_structural_verification_never_replays_source(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _localized_report()
    payload = diagnostic._json_bytes(report)
    replay_called = False

    def forbidden_replay(*_args: object, **_kwargs: object) -> None:
        nonlocal replay_called
        replay_called = True
        raise AssertionError("source replay forbidden")

    monkeypatch.setattr(diagnostic.v5, "build_artifacts", forbidden_replay)
    verification = diagnostic.build_structural_verification(
        report=report,
        report_payload=payload,
        plan=_synthetic_plan(),
        plan_file_sha256=report["plan_file_sha256"],
    )
    diagnostic.validate_structural_verification(
        verification, report=report, report_payload=payload
    )
    assert replay_called is False
    assert verification["source_replay_performed"] is False
    assert verification["candidate_or_baseline_payload_opened"] is False


def test_structural_verification_hash_tamper_fails() -> None:
    report = _localized_report()
    payload = diagnostic._json_bytes(report)
    verification = diagnostic.build_structural_verification(
        report=report,
        report_payload=payload,
        plan=_synthetic_plan(),
        plan_file_sha256=report["plan_file_sha256"],
    )
    verification["report_file_sha256"] = "0" * 64
    verification["verification_sha256"] = diagnostic._internal_sha(
        verification, "verification_sha256"
    )
    with pytest.raises(ValueError):
        diagnostic.validate_structural_verification(
            verification, report=report, report_payload=payload
        )


def test_structural_verification_rejects_resealed_binding_tamper() -> None:
    report = copy.deepcopy(_localized_report())
    report["bindings"]["predecessor_plan"] = "0" * 64
    report["report_sha256"] = diagnostic._internal_sha(report, "report_sha256")
    payload = diagnostic._json_bytes(report)
    with pytest.raises(ValueError):
        diagnostic.build_structural_verification(
            report=report,
            report_payload=payload,
            plan=_synthetic_plan(),
            plan_file_sha256=report["plan_file_sha256"],
        )


def test_output_preflight_is_no_clobber(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        diagnostic,
        "EXPECTED_OUTPUT_PATHS",
        {"report": "report.json", "verification": "verification.json"},
    )
    paths = diagnostic._preflight_outputs(tmp_path)
    paths["report"].write_bytes(b"marker")
    with pytest.raises(FileExistsError):
        diagnostic._preflight_outputs(tmp_path)


def test_marker_finish_requires_exact_current_bytes(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    diagnostic._write_exclusive(path, diagnostic.REPORT_MARKER)
    diagnostic._finish_marker(path, b"{}\n")
    assert path.read_bytes() == b"{}\n"
    with pytest.raises(ValueError):
        diagnostic._finish_marker(path, b"replacement\n")


def test_stage_order_and_gate_inventory_are_closed() -> None:
    assert len(diagnostic.STAGE_ORDER) == len(set(diagnostic.STAGE_ORDER))
    assert len(diagnostic.ALLOWED_GATE_CODES) == len(set(diagnostic.ALLOWED_GATE_CODES))
    assert set(diagnostic.GATE_TO_STAGE.values()).issubset(diagnostic.STAGE_ORDER)
    assert diagnostic.EXPECTED_PROHIBITED_ACTIONS == {
        "candidate_or_owner_artifact_write": False,
        "dataset_seal": False,
        "gpu_use": False,
        "inference": False,
        "network": False,
        "policy_or_floor_change": False,
        "promotion": False,
        "registry_mutation": False,
        "remote_or_instance_action": False,
        "retry": False,
        "source_mutation": False,
        "training": False,
    }
