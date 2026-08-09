from __future__ import annotations

from pathlib import Path

from lumi import posttraining_stronger_teacher_ab as paired
from lumi import posttraining_stronger_teacher_ab_gpu_completion as completion

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ARTIFACT = (
    ROOT.parent
    / "lumi-artifacts"
    / "4b-sft-stronger-teacher-ab-v0.1"
    / "c52325bacdb8"
    / "attempt-001"
)


def test_completion_is_narrow_and_mathematically_sufficient() -> None:
    assert completion.CORRECTION["maximum_response_tokens_before"] == 256
    assert completion.CORRECTION["maximum_response_tokens_after"] == 512
    assert completion.LIMITS["maximum_network_bytes"] == 0
    assert completion.LIMITS["remaining_valid_candidate_outputs"] == 79
    assert completion.LIMITS["maximum_additional_inference_attempts"] == 100
    assert completion.LIMITS["cumulative_inference_attempts"] == 282
    assert completion.LIMITS["cumulative_runtime_retries"] == 81
    assert all(completion.PROHIBITIONS.values())


def test_corrected_retry_payload_changes_only_the_response_ceiling() -> None:
    slot = next(
        item
        for item in completion.preflight.selected_schedule()
        if item.record_id == completion.FAILED_RECORD_ID
    )
    original = paired._payload(  # noqa: SLF001
        slot,
        model_key="candidate",
        exemplars={},
        prompt_variant="runtime_retry_v2",
    )
    corrected = paired._payload(  # noqa: SLF001
        slot,
        model_key="candidate",
        exemplars={},
        prompt_variant="runtime_retry_v2",
        maximum_response_tokens=512,
    )
    assert original["options"]["num_predict"] == 256
    assert corrected["options"]["num_predict"] == 512
    original_prompt = original["prompt"].replace("256 tokens", "<limit> tokens")
    corrected_prompt = corrected["prompt"].replace("512 tokens", "<limit> tokens")
    assert original_prompt == corrected_prompt
    original_options = dict(original["options"])
    corrected_options = dict(corrected["options"])
    original_options.pop("num_predict")
    corrected_options.pop("num_predict")
    assert original_options == corrected_options


def test_failure_boundary_is_exact() -> None:
    boundary = completion.current_boundary(
        artifact=SOURCE_ARTIFACT, exact_failure_boundary=True
    )
    assert boundary["candidate"]["valid_finals"] == 21
    assert boundary["candidate"]["incomplete_final_record_ids"] == [
        completion.FAILED_RECORD_ID
    ]
    assert boundary["cumulative_inference_attempts"] == 182
    assert boundary["cumulative_runtime_retries"] == 60
