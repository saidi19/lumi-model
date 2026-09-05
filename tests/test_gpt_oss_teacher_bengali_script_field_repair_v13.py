from __future__ import annotations

import json

from lumi.posttraining_gpt_oss_teacher_bengali_script_field_repair_v13 import (
    EXACT_ASSISTANT_RESPONSE,
    EXACT_AUTHORIZATION_TEXT,
    EXACT_FIELD_INSTRUCTION,
    EXACT_USER_PROMPT,
    LIMITS,
    RECORD_ID,
    REQUIRED_LITERALS,
    _reference_row_passes,
    repair_manifest,
    repair_payload,
    validate_exact_bengali_fields,
)


def test_repair_is_one_row_and_bounded() -> None:
    assert RECORD_ID == "lumi-gptoss-factual_qa-bengali_script-05"
    assert LIMITS["exact_repair_records"] == 1
    assert LIMITS["maximum_inference_calls"] == 2
    assert LIMITS["maximum_runtime_retries"] == 1
    assert LIMITS["maximum_network_bytes"] == 0


def test_manifest_attempts_are_distinct() -> None:
    manifest = repair_manifest()
    assert len(manifest["attempts"]) == 2
    assert len({item["seed"] for item in manifest["attempts"]}) == 2
    assert len({item["request_sha256"] for item in manifest["attempts"]}) == 2


def test_payload_binds_exact_bengali_fields() -> None:
    payload = repair_payload(1)
    text = json.dumps(payload["messages"], ensure_ascii=False)
    assert EXACT_FIELD_INSTRUCTION in text
    assert EXACT_USER_PROMPT in text
    assert EXACT_ASSISTANT_RESPONSE in text
    for literal in REQUIRED_LITERALS:
        assert text.count(literal) >= 2
    assert payload["chat_template_kwargs"] == {"reasoning_effort": "low"}
    assert payload["max_tokens"] == 1536


def test_reference_row_passes_frozen_and_repair_gates() -> None:
    assert _reference_row_passes()


def test_gate_rejects_any_field_deviation() -> None:
    try:
        validate_exact_bengali_fields([
            {"role": "user", "content": EXACT_USER_PROMPT},
            {"role": "assistant", "content": "তিন চতুর্থাংশের দশমিক মান 0.75।"},
        ])
    except ValueError as error:
        assert "assistant_response is not the exact reviewed text" in str(error)
        return
    raise AssertionError("one-sided literal omission passed the v0.13 gate")


def test_authorization_cannot_resume_full_canary() -> None:
    assert "cannot satisfy the 16-cell gate" in EXACT_AUTHORIZATION_TEXT
    assert "No external API, public port, model or runtime download" in EXACT_AUTHORIZATION_TEXT
