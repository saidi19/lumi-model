from collections import Counter

from lumi import posttraining_first_party_scale_canary as scale
from lumi import posttraining_gpt_oss_closure_generation_preflight_v1 as preflight
from lumi import posttraining_gpt_oss_closure_generation_v1 as closure


def test_schedule_is_unique_and_has_exact_five_percent_holdout_pattern() -> None:
    schedule = closure.build_schedule()
    assert len({slot.record_id for slot in schedule}) == len(schedule)
    assert len({slot.user_prompt for slot in schedule}) == len(schedule)
    counts = Counter((slot.task, slot.language_mode, slot.split) for slot in schedule)
    for task in scale.TASKS:
        for language in scale.LANGUAGE_MODES:
            target = preflight.CELL_TARGETS[task][language]
            train = counts[(task, language, "train")]
            validation = counts[(task, language, "validation")]
            assert train == closure._cell_max_train_slots(target)
            assert validation == train // 19


def test_payload_is_response_only_and_authorized() -> None:
    slot = next(slot for slot in closure.build_schedule() if slot.task != "tool_api")
    first = closure.chat_completion_payload(slot, 1, None)
    second = closure.chat_completion_payload(slot, 2, "tokens:supervised_below_64")
    assert first["max_tokens"] == 2_048
    assert first["chat_template_kwargs"] == {"reasoning_effort": "low"}
    assert first["response_format"]["json_schema"]["schema"]["required"] == ["assistant_response"]
    assert first["seed"] != second["seed"]
    assert preflight.EXPECTED_SCHEDULE_SPEC_SHA256 == scale.sha256_value(preflight.SCHEDULE_SPEC)
