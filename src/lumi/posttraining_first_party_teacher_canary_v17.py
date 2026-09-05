"""Deterministic first-party completion for the four Banglish writing rows."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import posttraining_first_party_teacher_canary as base
from . import posttraining_first_party_teacher_canary_v4 as schema_v4
from . import posttraining_first_party_teacher_canary_v11 as terminal_v11
from . import posttraining_first_party_teacher_canary_v15 as task_v15
from .posttraining_first_party import file_sha256

CONTINUATION_CONTRACT = {
    "additional_inference_attempts": 40,
    "deterministic_template_records": 4,
    "maximum_attempts_per_record": 3,
    "maximum_local_inference_calls": 36,
    "maximum_network_bytes": 0,
    "maximum_retained_new_records": 16,
    "maximum_retained_supervised_tokens_total": 24_000,
    "maximum_runtime_minutes": 120,
    "total_retained_records": 80,
}
CARRY_RECORDS = 64
NEW_RECORDS = 16
DETERMINISTIC_TEMPLATE_RECORDS = 4
NON_TARGET_RECORDS = 12
MAXIMUM_LOCAL_INFERENCE_CALLS = 36
TEMPLATE_CONTRACT = "lumi.first-party-ai-assisted-template.v0.1"
TEMPLATE_METHOD = "frozen_scenario_bound_ai_assisted_template_v1"

_V15_GENERATE = task_v15.generate_continuation_attempt
_V15_MAKE_ROW = task_v15._make_row  # noqa: SLF001

FROZEN_WRITING_TEMPLATES: dict[str, dict[str, str]] = {
    "meeting_summary": {
        "user_prompt": (
            "Eta ektu short summary koro: Meeting 10:00 e Room B te hobe, ar ID niye "
            "aste hobe."
        ),
        "assistant_response": (
            "Ami short summary dichhi: 10:00 e Room B te meeting hobe, tai ID niye aste "
            "hobe."
        ),
    },
    "release_summary": {
        "user_prompt": (
            "Eta ektu short summary koro: Version 2.4 e dark mode ar offline search add "
            "kora hoyeche."
        ),
        "assistant_response": (
            "Ami short summary dichhi: 2.4 version e dark mode ar offline search jog "
            "hoyeche."
        ),
    },
    "plant_care_summary": {
        "user_prompt": (
            "Eta ektu short summary koro: Gache 3 times/week pani dite hobe, indirect "
            "light e rakhte hobe, ar Friday mati check korte hobe."
        ),
        "assistant_response": (
            "Ami short summary dichhi: 3 times/week pani dao, indirect light e rakho, ar "
            "Friday mati check koro."
        ),
    },
    "project_summary": {
        "user_prompt": (
            "Eta ektu short summary koro: Phase Alpha er deadline 12 May, ar review er "
            "dayitto Nadia er."
        ),
        "assistant_response": (
            "Ami short summary dichhi: Phase Alpha 12 May er moddhe shesh korte hobe, ar "
            "review korbe Nadia."
        ),
    },
}


def _targeted(scenario: base.Scenario, language_mode: str) -> bool:
    return language_mode == "banglish" and scenario.scenario_id in FROZEN_WRITING_TEMPLATES


def deterministic_template_output(scenario: base.Scenario) -> dict[str, Any]:
    template = FROZEN_WRITING_TEMPLATES.get(scenario.scenario_id)
    if template is None:
        raise ValueError("scenario is not bound to a deterministic writing template")
    return {
        "assistant_response": template["assistant_response"],
        "safety_label": scenario.safety_label,
        "scenario_id": scenario.scenario_id,
        "tool_call": None,
        "tool_result": None,
        "user_prompt": template["user_prompt"],
    }


def _make_template_row(
    record_id: str,
    scenario: base.Scenario,
    *,
    attempt_number: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = deterministic_template_output(scenario)
    task_messages = task_v15.validate_task_anchored_banglish(
        output,
        scenario=scenario,
        language_mode="banglish",
    )
    normalized, messages, terminal_composition = terminal_v11.compose_banglish_terminal_output(
        output,
        scenario=scenario,
    )
    if normalized != output or messages != task_messages:
        raise ValueError("frozen writing template was modified during validation")
    if any(
        terminal_composition[f"{field}_action"] != "preserve_existing_terminal"
        for field in ("user_prompt", "assistant_response")
    ):
        raise ValueError("frozen writing template terminal punctuation changed")

    row = _V15_MAKE_ROW(
        record_id,
        scenario,
        "banglish",
        messages,
        attempt_number=attempt_number,
        terminal_composition=None,
        banglish_composition=terminal_composition,
    )
    output_sha256 = base.sha256_value(output)
    row["source_id"] = "lumi_first_party_multilingual_safety_tool_v0_17"
    provenance = row["provenance"]
    provenance["prompt_contract"] = TEMPLATE_CONTRACT
    provenance["deterministic_template_method"] = TEMPLATE_METHOD
    provenance["deterministic_template_internal_sha256"] = output_sha256
    provenance["generated_by_local_teacher_inference"] = False
    provenance["human_authorship_claimed"] = False

    request = {
        "method": TEMPLATE_METHOD,
        "record_id": record_id,
        "scenario_id": scenario.scenario_id,
        "template_internal_sha256": output_sha256,
    }
    telemetry = {
        "attempt_number": attempt_number,
        "banglish_terminal_composition": terminal_composition,
        "deterministic_template_internal_sha256": output_sha256,
        "deterministic_template_method": TEMPLATE_METHOD,
        "elapsed_seconds": 0.0,
        "eval_count": 0,
        "eval_duration_ns": 0,
        "load_duration_ns": 0,
        "local_inference_used": False,
        "prompt_eval_count": 0,
        "prompt_eval_duration_ns": 0,
        "record_id": record_id,
        "request_sha256": base.sha256_value(request),
        "response_envelope_sha256": output_sha256,
        "terminal_composition": None,
    }
    return row, telemetry


def generate_continuation_attempt(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
    *,
    attempt_number: int,
    endpoint: str,
    rejection_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _targeted(scenario, language_mode):
        return _make_template_row(
            record_id,
            scenario,
            attempt_number=attempt_number,
        )
    return _V15_GENERATE(
        record_id,
        scenario,
        language_mode,
        attempt_number=attempt_number,
        endpoint=endpoint,
        rejection_path=rejection_path,
    )


def _validate_continuation_plan(plan: dict[str, Any], *, artifact: Path) -> None:
    if plan.get("schema_completion") != CONTINUATION_CONTRACT:
        raise ValueError("deterministic template continuation contract changed")
    carry = plan.get("carry_forward")
    if not isinstance(carry, dict):
        raise ValueError("deterministic template carry-forward binding is missing")
    family_root = artifact.parent
    expected = {
        "candidate_rows_path": str(
            family_root / "attempt-016" / "private" / "candidate-rows.partial.jsonl"
        ),
        "candidate_rows_sha256": (
            "ba6650bf74a20f6e4dc06a1e9a6a0f3330d8e0b865316406d7d273d205e84fd1"
        ),
        "failure_report_path": str(
            family_root / "attempt-016" / "evidence" / "failure-report.json"
        ),
        "failure_report_sha256": (
            "5d7692a82ffce304e2192e6e8027e06ffc41d9f296a3915fab31b05b4a3aa1de"
        ),
        "failure_verification_path": str(
            family_root / "attempt-016" / "evidence" / "failure-verification.json"
        ),
        "failure_verification_sha256": (
            "8b8ed644436d35cc24645dfc2f425c5e526dad47cbd06873bd23ebd2fde3fa7f"
        ),
        "records": CARRY_RECORDS,
        "telemetry_path": str(
            family_root / "attempt-016" / "private" / "generation-telemetry.partial.jsonl"
        ),
        "telemetry_sha256": ("02c8393fc9e2c710260838d2c86f0758b64b45ce21820e3c0f135ba723aebaa2"),
    }
    if carry != expected:
        raise ValueError("deterministic template carry-forward contract changed")
    for field in ("candidate_rows", "failure_report", "failure_verification", "telemetry"):
        if file_sha256(carry[f"{field}_path"]) != carry[f"{field}_sha256"]:
            raise ValueError(f"deterministic template carry-forward changed: {field}")


def _initialize_continuation(
    plan: dict[str, Any],
    *,
    artifact: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    private = artifact / "private"
    partial_rows = private / "candidate-rows.partial.jsonl"
    partial_telemetry = private / "generation-telemetry.partial.jsonl"
    ledger_path = private / "additional-attempt-ledger.jsonl"
    present = (partial_rows.exists(), partial_telemetry.exists())
    if any(present) and not all(present):
        raise ValueError("deterministic template partial row/telemetry pair is incomplete")
    carry_rows = base._read_jsonl(  # noqa: SLF001
        Path(plan["carry_forward"]["candidate_rows_path"])
    )
    carry_telemetry = base._read_jsonl(  # noqa: SLF001
        Path(plan["carry_forward"]["telemetry_path"])
    )
    if len(carry_rows) != CARRY_RECORDS or len(carry_telemetry) != CARRY_RECORDS:
        raise ValueError("deterministic template carry-forward count changed")
    if not any(present):
        base.write_jsonl(partial_rows, carry_rows)
        base.write_jsonl(partial_telemetry, carry_telemetry)
    rows = base._read_jsonl(partial_rows)  # noqa: SLF001
    telemetry = base._read_jsonl(partial_telemetry)  # noqa: SLF001
    ledger = base._read_jsonl(ledger_path)  # noqa: SLF001
    if rows[:CARRY_RECORDS] != carry_rows or telemetry[:CARRY_RECORDS] != carry_telemetry:
        raise ValueError("deterministic template exact carry-forward changed")
    expected_ids = base.expected_record_ids()
    if [str(row.get("record_id")) for row in rows] != list(expected_ids[: len(rows)]):
        raise ValueError("deterministic template rows are not an exact expected prefix")
    if len(rows) != len(telemetry) or not CARRY_RECORDS <= len(rows) <= 80:
        raise ValueError("deterministic template row and telemetry counts differ")
    if len(ledger) > CONTINUATION_CONTRACT["additional_inference_attempts"]:
        raise ValueError("deterministic template continuation ledger exceeds authorization")
    return rows, telemetry, ledger


@contextmanager
def _patched_v4_runtime() -> Iterator[None]:
    original_contract = schema_v4.SCHEMA_COMPLETION_CONTRACT
    original_validate = schema_v4._validate_schema_completion_plan
    original_initialize = schema_v4._initialize_or_load
    original_generate = schema_v4.generate_schema_completion_attempt
    schema_v4.SCHEMA_COMPLETION_CONTRACT = CONTINUATION_CONTRACT
    schema_v4._validate_schema_completion_plan = _validate_continuation_plan
    schema_v4._initialize_or_load = _initialize_continuation
    schema_v4.generate_schema_completion_attempt = generate_continuation_attempt
    try:
        yield
    finally:
        schema_v4.SCHEMA_COMPLETION_CONTRACT = original_contract
        schema_v4._validate_schema_completion_plan = original_validate
        schema_v4._initialize_or_load = original_initialize
        schema_v4.generate_schema_completion_attempt = original_generate


def run_deterministic_template_continuation(
    *,
    plan_path: str | Path,
    repository_root: str | Path,
    artifact_root: str | Path,
    endpoint: str = "http://127.0.0.1:11434/api/generate",
) -> dict[str, Any]:
    with _patched_v4_runtime():
        report = schema_v4.run_schema_completion(
            plan_path=plan_path,
            repository_root=repository_root,
            artifact_root=artifact_root,
            endpoint=endpoint,
        )
    artifact = Path(artifact_root)
    rows = base._read_jsonl(artifact / "private" / "candidate-rows.jsonl")  # noqa: SLF001
    telemetry = base._read_jsonl(  # noqa: SLF001
        artifact / "private" / "generation-telemetry.jsonl"
    )
    ledger = base._read_jsonl(  # noqa: SLF001
        artifact / "private" / "additional-attempt-ledger.jsonl"
    )
    new_rows = rows[CARRY_RECORDS:]
    new_telemetry = telemetry[CARRY_RECORDS:]
    template_records = sum(
        item.get("local_inference_used") is False for item in new_telemetry
    )
    local_inference_calls = len(ledger) - template_records
    report["carry_forward"]["records"] = CARRY_RECORDS
    report["schema_completion"]["additional_records_retained"] = NEW_RECORDS
    report["schema_completion"]["deterministic_template_method"] = TEMPLATE_METHOD
    report["schema_completion"]["deterministic_template_records"] = template_records
    report["schema_completion"]["local_inference_calls"] = local_inference_calls
    report["schema_completion"]["non_target_records"] = sum(
        row.get("source_id") == "lumi_first_party_multilingual_safety_tool_v0_15"
        for row in new_rows
    )
    base.write_json(artifact / "evidence" / "run-report.json", report)
    return report


def _verify_template_row(
    row: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    *,
    scenario: base.Scenario,
) -> None:
    expected_output = deterministic_template_output(scenario)
    natural = {
        message["role"]: str(message["content"])
        for message in row.get("messages", [])
        if message.get("role") in {"user", "assistant"}
    }
    if natural != {
        "user": expected_output["user_prompt"],
        "assistant": expected_output["assistant_response"],
    }:
        raise ValueError("frozen deterministic writing template changed")
    expected_sha256 = base.sha256_value(expected_output)
    provenance = row.get("provenance", {})
    if row.get("source_id") != "lumi_first_party_multilingual_safety_tool_v0_17":
        raise ValueError("deterministic template source binding changed")
    if provenance.get("prompt_contract") != TEMPLATE_CONTRACT:
        raise ValueError("deterministic template contract changed")
    if provenance.get("deterministic_template_method") != TEMPLATE_METHOD:
        raise ValueError("deterministic template provenance method changed")
    if provenance.get("deterministic_template_internal_sha256") != expected_sha256:
        raise ValueError("deterministic template provenance hash changed")
    if provenance.get("generated_by_local_teacher_inference") is not False:
        raise ValueError("deterministic template inference provenance changed")
    if provenance.get("human_authorship_claimed") is not False:
        raise ValueError("deterministic template human-authorship provenance changed")
    if telemetry.get("local_inference_used") is not False:
        raise ValueError("deterministic template telemetry claims inference")
    if telemetry.get("deterministic_template_method") != TEMPLATE_METHOD:
        raise ValueError("deterministic template telemetry method changed")
    if telemetry.get("deterministic_template_internal_sha256") != expected_sha256:
        raise ValueError("deterministic template telemetry hash changed")
    if telemetry.get("response_envelope_sha256") != expected_sha256:
        raise ValueError("deterministic template response hash changed")
    if terminal_v11._verify_banglish_composition(row, telemetry) != 0:  # noqa: SLF001
        raise ValueError("deterministic template punctuation changed")


def verify_deterministic_template_continuation(
    *,
    plan_path: str | Path,
    repository_root: str | Path,
    artifact_root: str | Path,
) -> dict[str, Any]:
    artifact = Path(artifact_root).resolve()
    verification = base.verify_run(
        plan_path=plan_path,
        repository_root=repository_root,
        artifact_root=artifact,
    )
    plan = base.load_canary_plan(plan_path, repository_root=repository_root)
    _validate_continuation_plan(plan, artifact=artifact)
    report = base._json_object(artifact / "evidence" / "run-report.json")  # noqa: SLF001
    rows = base._read_jsonl(artifact / "private" / "candidate-rows.jsonl")  # noqa: SLF001
    telemetry = base._read_jsonl(  # noqa: SLF001
        artifact / "private" / "generation-telemetry.jsonl"
    )
    ledger_path = artifact / "private" / "additional-attempt-ledger.jsonl"
    ledger = base._read_jsonl(ledger_path)  # noqa: SLF001
    carry_rows = base._read_jsonl(  # noqa: SLF001
        Path(plan["carry_forward"]["candidate_rows_path"])
    )
    carry_telemetry = base._read_jsonl(  # noqa: SLF001
        Path(plan["carry_forward"]["telemetry_path"])
    )
    if (
        rows[:CARRY_RECORDS] != carry_rows
        or telemetry[:CARRY_RECORDS] != carry_telemetry
        or len(rows) != 80
        or len(telemetry) != 80
    ):
        raise ValueError("deterministic template carry-forward or final counts changed")
    if not NEW_RECORDS <= len(ledger) <= CONTINUATION_CONTRACT["additional_inference_attempts"]:
        raise ValueError("deterministic template continuation attempt count is invalid")
    if sum(item.get("status") == "accepted" for item in ledger) != NEW_RECORDS:
        raise ValueError("deterministic template accepted-operation count changed")
    if base.sha256_value(ledger) != report.get("schema_completion", {}).get(
        "attempt_ledger_internal_sha256"
    ):
        raise ValueError("deterministic template continuation ledger changed")
    if file_sha256(ledger_path) != report.get("attempt_ledger_file_sha256"):
        raise ValueError("deterministic template continuation ledger file changed")

    scenarios = {scenario.scenario_id: scenario for scenario in base.SCENARIOS}
    template_records = 0
    non_target_records = 0
    code_switch_records = 0
    for index, (row, item_telemetry) in enumerate(zip(rows, telemetry, strict=True)):
        if index < CARRY_RECORDS:
            continue
        scenario = scenarios[str(row["skill_id"])]
        language_mode = str(row["language_mode"])
        if _targeted(scenario, language_mode):
            _verify_template_row(row, item_telemetry, scenario=scenario)
            template_records += 1
            continue
        if row.get("source_id") != "lumi_first_party_multilingual_safety_tool_v0_15":
            raise ValueError("non-target predecessor source binding changed")
        if item_telemetry.get("local_inference_used") is False:
            raise ValueError("non-target row incorrectly claims zero inference")
        if language_mode == "natural_code_switching":
            task_v15._verify_new_code_switch_terminal(  # noqa: SLF001
                row,
                item_telemetry,
                scenario=scenario,
            )
            code_switch_records += 1
        non_target_records += 1

    local_inference_calls = len(ledger) - template_records
    if template_records != DETERMINISTIC_TEMPLATE_RECORDS:
        raise ValueError("deterministic template record count changed")
    if non_target_records != NON_TARGET_RECORDS:
        raise ValueError("non-target record count changed")
    if code_switch_records != 4:
        raise ValueError("remaining code-switch record count changed")
    if not NON_TARGET_RECORDS <= local_inference_calls <= MAXIMUM_LOCAL_INFERENCE_CALLS:
        raise ValueError("local inference call count exceeds deterministic template contract")
    schema_report = report.get("schema_completion", {})
    if report.get("carry_forward", {}).get("records") != CARRY_RECORDS:
        raise ValueError("deterministic template report carry count changed")
    if schema_report.get("deterministic_template_records") != template_records:
        raise ValueError("deterministic template report count changed")
    if schema_report.get("non_target_records") != non_target_records:
        raise ValueError("non-target report count changed")
    if schema_report.get("local_inference_calls") != local_inference_calls:
        raise ValueError("local inference report count changed")

    verification["checks"]["carry_forward_exact_sixty_four_rows"] = True
    verification["checks"]["four_frozen_Banglish_writing_templates_verified"] = True
    verification["checks"]["templates_are_AI_assisted_not_human_authored"] = True
    verification["checks"]["templates_used_zero_local_inference"] = True
    verification["checks"]["all_twelve_non_target_paths_preserved_v0_15"] = True
    verification["attempt_ledger_file_sha256"] = file_sha256(ledger_path)
    verification["deterministic_template_records_verified"] = template_records
    verification["non_target_records_verified"] = non_target_records
    verification["local_inference_calls_verified"] = local_inference_calls
    base.write_json(artifact / "evidence" / "verification-report.json", verification)
    return verification
