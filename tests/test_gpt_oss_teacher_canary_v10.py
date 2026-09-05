from __future__ import annotations

from pathlib import Path

from lumi.posttraining_gpt_oss_teacher_canary_v10 import (
    PREDECESSOR_FAILURE,
    _BASE_V8_CHAT_COMPLETION_PAYLOAD,
    _BASE_V8_VALIDATE_PROMPT_QUALITY,
    _patched_v8,
    chat_completion_payload,
    patch_context_regression_gate,
    validate_prompt_quality,
    v8,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-gpt-oss-teacher-runner-repair-resume-v0.10.json"


def test_base_functions_are_captured_before_patch() -> None:
    assert _BASE_V8_CHAT_COMPLETION_PAYLOAD is not chat_completion_payload
    assert _BASE_V8_VALIDATE_PROMPT_QUALITY is not validate_prompt_quality


def test_patch_context_restores_v8_symbols_even_on_error() -> None:
    original = v8.chat_completion_payload
    try:
        with _patched_v8():
            assert v8.chat_completion_payload is chat_completion_payload
            raise RuntimeError("deliberate test error")
    except RuntimeError:
        pass
    assert v8.chat_completion_payload is original


def test_v09_failure_consumed_no_model_calls() -> None:
    assert PREDECESSOR_FAILURE["additional_inference_calls"] == 0
    assert PREDECESSOR_FAILURE["additional_runtime_retries"] == 0


def test_patch_context_executes_plan_and_payload_without_recursion() -> None:
    checks = patch_context_regression_gate(PLAN, ROOT)
    assert all(checks.values())
