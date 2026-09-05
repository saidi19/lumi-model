from __future__ import annotations

from lumi.posttraining_first_party_teacher_canary import LANGUAGE_MODES, SCENARIOS
from lumi.posttraining_first_party_teacher_canary_v2 import (
    STRICT_BANGLISH_INSTRUCTION,
    build_retry_prompt,
    retry_payload,
)


def test_retry_prompt_strengthens_only_banglish() -> None:
    scenario = SCENARIOS[0]
    prompt = build_retry_prompt(scenario, "banglish")
    assert STRICT_BANGLISH_INSTRUCTION in prompt
    assert "English-only text is forbidden" in prompt
    assert "Each field" in prompt
    for language in set(LANGUAGE_MODES) - {"banglish"}:
        assert STRICT_BANGLISH_INSTRUCTION not in build_retry_prompt(scenario, language)


def test_retry_payload_keeps_exact_generation_limits() -> None:
    scenario = SCENARIOS[0]
    record_id = "lumi-teacher-factual_qa-banglish-00"
    payload = retry_payload(record_id, scenario, "banglish")
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"]["num_ctx"] == 4096
    assert payload["options"]["num_predict"] == 384
    assert payload["options"]["temperature"] == 0.7
    assert payload["options"]["top_k"] == 20
    assert payload["options"]["top_p"] == 0.8
