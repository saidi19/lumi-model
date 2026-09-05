"""Fully deterministic completion of the final Lumi teacher-canary writing rows."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import posttraining_first_party_teacher_canary as base
from . import posttraining_first_party_teacher_canary_v15 as task_v15
from . import posttraining_first_party_teacher_canary_v17 as template_v17
from .posttraining_first_party import file_sha256
from .posttraining_owner_review import build_owner_review_html

DETERMINISTIC_COMPLETION_CONTRACT = {
    "deterministic_template_records": 15,
    "maximum_local_inference_calls": 0,
    "maximum_network_bytes": 0,
    "maximum_operations": 15,
    "maximum_retained_new_records": 15,
    "maximum_retained_supervised_tokens_total": 24_000,
    "maximum_runtime_minutes": 60,
    "total_retained_records": 80,
}
CARRY_RECORDS = 65
NEW_RECORDS = 15
MAXIMUM_LOCAL_INFERENCE_CALLS = 0
TEMPLATE_CONTRACT = "lumi.first-party-ai-assisted-deterministic-multilingual-template.v0.1"
TEMPLATE_METHOD = "frozen_scenario_language_ai_assisted_template_v1"
SOURCE_ID = "lumi_first_party_multilingual_safety_tool_v0_18"


FROZEN_WRITING_TEMPLATES: dict[str, dict[str, dict[str, str]]] = {
    "meeting_summary": {
        "banglish": template_v17.FROZEN_WRITING_TEMPLATES["meeting_summary"],
        "bengali_script": {
            "user_prompt": (
                "নিচের তথ্য সংক্ষিপ্ত করুন: সভা হবে 10:00-এ, স্থান Room B; প্রবেশের "
                "জন্য ID সঙ্গে আনুন।"
            ),
            "assistant_response": (
                "সংক্ষেপ: 10:00-এ Room B-তে সভা হবে; প্রবেশের সময় ID সঙ্গে রাখতে হবে।"
            ),
        },
        "english": {
            "user_prompt": (
                "Please summarize this note: The meeting is at 10:00 in Room B, and "
                "attendees must bring ID."
            ),
            "assistant_response": (
                "Summary: The meeting is at 10:00 in Room B, and attendees must bring ID."
            ),
        },
        "natural_code_switching": {
            "user_prompt": (
                "একটি short update বানান—meeting 10:00-এ Room B-তে, আর check-in-এর "
                "জন্য ID আনতে হবে।"
            ),
            "assistant_response": (
                "Quick update: 10:00-এ Room B-তে meeting হবে; check-in-এর সময় ID প্রয়োজন।"
            ),
        },
    },
    "release_summary": {
        "banglish": template_v17.FROZEN_WRITING_TEMPLATES["release_summary"],
        "bengali_script": {
            "user_prompt": (
                "রিলিজ নোটটি সংক্ষেপ করুন: Version 2.4-এ dark mode ও offline search "
                "যোগ হয়েছে।"
            ),
            "assistant_response": (
                "সংক্ষেপে, Version 2.4-এ dark mode এবং offline search—দুটিই যোগ হয়েছে।"
            ),
        },
        "english": {
            "user_prompt": (
                "Condense this release note: Version 2.4 introduces dark mode and "
                "offline search."
            ),
            "assistant_response": (
                "Version 2.4 adds two features: dark mode and offline search."
            ),
        },
        "natural_code_switching": {
            "user_prompt": (
                "এই release note থেকে একটি short summary দিন: Version 2.4 এখন dark mode "
                "ও offline search support করে।"
            ),
            "assistant_response": (
                "Release summary: Version 2.4-এ নতুন dark mode ও offline search পাওয়া যাবে।"
            ),
        },
    },
    "plant_care_summary": {
        "banglish": template_v17.FROZEN_WRITING_TEMPLATES["plant_care_summary"],
        "bengali_script": {
            "user_prompt": (
                "পরিচর্যার নোটটি সংক্ষেপ করুন: গাছে 3 times/week পানি দিন, indirect "
                "light-এ রাখুন এবং Friday মাটি পরীক্ষা করুন।"
            ),
            "assistant_response": (
                "সংক্ষেপ: 3 times/week পানি দিতে হবে, indirect light বজায় রাখতে হবে এবং "
                "Friday মাটি দেখতে হবে।"
            ),
        },
        "english": {
            "user_prompt": (
                "Summarize the care instructions: Water 3 times/week, provide indirect "
                "light, and check the soil on Friday."
            ),
            "assistant_response": (
                "Care summary: Water 3 times/week, keep the plant in indirect light, and "
                "check its soil on Friday."
            ),
        },
        "natural_code_switching": {
            "user_prompt": (
                "এই plant care note-টি ছোট করুন: 3 times/week পানি, indirect light, আর "
                "Friday soil check দরকার।"
            ),
            "assistant_response": (
                "Care update: গাছে 3 times/week পানি দিন, indirect light-এ রাখুন এবং "
                "Friday soil check করুন।"
            ),
        },
    },
    "project_summary": {
        "banglish": template_v17.FROZEN_WRITING_TEMPLATES["project_summary"],
        "bengali_script": {
            "user_prompt": (
                "প্রকল্পের নোটটি সংক্ষেপ করুন: Phase Alpha-এর শেষ সময় 12 May এবং Nadia "
                "রিভিউয়ের দায়িত্বে আছেন।"
            ),
            "assistant_response": (
                "সংক্ষেপ: Phase Alpha 12 May-এর মধ্যে শেষ হবে, আর রিভিউ করবেন Nadia।"
            ),
        },
        "english": {
            "user_prompt": (
                "Summarize this project note: Phase Alpha is due on 12 May, and Nadia "
                "owns the review."
            ),
            "assistant_response": (
                "Project summary: Phase Alpha is due 12 May, with Nadia responsible for "
                "the review."
            ),
        },
        "natural_code_switching": {
            "user_prompt": (
                "এই project update সংক্ষেপ করুন: Phase Alpha-এর deadline 12 May, আর "
                "review owner হলেন Nadia।"
            ),
            "assistant_response": (
                "Project update: Phase Alpha 12 May-এর মধ্যে শেষ করতে হবে; review করবেন "
                "Nadia।"
            ),
        },
    },
}


def deterministic_template_output(
    scenario: base.Scenario,
    language_mode: str,
) -> dict[str, Any]:
    try:
        template = FROZEN_WRITING_TEMPLATES[scenario.scenario_id][language_mode]
    except KeyError as error:
        raise ValueError("scenario-language pair lacks a frozen writing template") from error
    return {
        "assistant_response": template["assistant_response"],
        "safety_label": scenario.safety_label,
        "scenario_id": scenario.scenario_id,
        "tool_call": None,
        "tool_result": None,
        "user_prompt": template["user_prompt"],
    }


def validate_deterministic_template(
    output: Mapping[str, Any],
    *,
    scenario: base.Scenario,
    language_mode: str,
) -> list[dict[str, str]]:
    if language_mode == "banglish":
        messages = task_v15.validate_task_anchored_banglish(
            output,
            scenario=scenario,
            language_mode=language_mode,
        )
    else:
        messages = base._validate_output(  # noqa: SLF001
            output,
            scenario=scenario,
            language_mode=language_mode,
        )
    for field in ("user_prompt", "assistant_response"):
        text = output.get(field)
        if not isinstance(text, str) or text != text.strip():
            raise ValueError(f"frozen template {field} has surrounding whitespace")
        if text[-1] not in ".!?।":
            raise ValueError(f"frozen template {field} lacks terminal punctuation")
        for literal in scenario.required_literals:
            if literal not in text:
                raise ValueError(f"frozen template {field} lacks literal: {literal}")
    return messages


def make_deterministic_template_row(
    record_id: str,
    scenario: base.Scenario,
    language_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = deterministic_template_output(scenario, language_mode)
    messages = validate_deterministic_template(
        output,
        scenario=scenario,
        language_mode=language_mode,
    )
    template_sha256 = base.sha256_value(output)
    row = {
        "schema_version": base.ROW_SCHEMA_VERSION,
        "record_id": record_id,
        "source_id": SOURCE_ID,
        "task": scenario.task,
        "skill_id": scenario.scenario_id,
        "language_mode": language_mode,
        "split": "validation" if scenario.variant == 3 else "train",
        "messages": messages,
        "provenance": {
            "ai_assisted": True,
            "authoring_assistant_family": "codex",
            "deterministic_template_internal_sha256": template_sha256,
            "deterministic_template_method": TEMPLATE_METHOD,
            "freeform_manual_rewriting": False,
            "generated_by_local_teacher_inference": False,
            "human_authored": False,
            "human_authorship_claimed": False,
            "inference_location": "none_deterministic_materialization",
            "language_mode_binding": language_mode,
            "prompt_contract": TEMPLATE_CONTRACT,
            "required_literals": list(scenario.required_literals),
            "runtime_external_api_used": False,
            "scenario_id_binding": scenario.scenario_id,
            "self_hosted": True,
        },
        "rights_status": "private_research_candidate_pending_owner_review",
    }
    request = {
        "language_mode": language_mode,
        "method": TEMPLATE_METHOD,
        "record_id": record_id,
        "scenario_id": scenario.scenario_id,
        "template_internal_sha256": template_sha256,
    }
    telemetry = {
        "attempt_number": 1,
        "deterministic_template_internal_sha256": template_sha256,
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
        "response_envelope_sha256": template_sha256,
    }
    return row, telemetry


def _validate_plan(plan: dict[str, Any], *, artifact: Path) -> None:
    if plan.get("deterministic_completion") != DETERMINISTIC_COMPLETION_CONTRACT:
        raise ValueError("deterministic multilingual completion contract changed")
    carry = plan.get("carry_forward")
    if not isinstance(carry, dict):
        raise ValueError("deterministic multilingual carry-forward binding is missing")
    family_root = artifact.parent
    expected = {
        "candidate_rows_path": str(
            family_root / "attempt-017" / "private" / "candidate-rows.partial.jsonl"
        ),
        "candidate_rows_sha256": (
            "fb7b7580525df1ca0ecbb977e81e062dac59a225f3d89fe0d5f10039a92f7a9f"
        ),
        "failure_report_path": str(
            family_root / "attempt-017" / "evidence" / "failure-report.json"
        ),
        "failure_report_sha256": (
            "514bc7053a4c23b8c767af94b4634480a4fac0a2c0ce41c0c7f96dfccab8cd06"
        ),
        "failure_verification_path": str(
            family_root / "attempt-017" / "evidence" / "failure-verification.json"
        ),
        "failure_verification_sha256": (
            "9037ebe458d41a706e0e8e8df557f9d74520d6c5a486c90f6a9a46c07cc07a3f"
        ),
        "records": CARRY_RECORDS,
        "telemetry_path": str(
            family_root / "attempt-017" / "private" / "generation-telemetry.partial.jsonl"
        ),
        "telemetry_sha256": (
            "41b5fe05aa86ede83de02dc1ce9be20b95df6514d263ae312db8aff03fdb4873"
        ),
    }
    if carry != expected:
        raise ValueError("deterministic multilingual carry-forward contract changed")
    for field in ("candidate_rows", "failure_report", "failure_verification", "telemetry"):
        if file_sha256(carry[f"{field}_path"]) != carry[f"{field}_sha256"]:
            raise ValueError(f"deterministic multilingual carry-forward changed: {field}")


def _initialize(
    plan: dict[str, Any],
    *,
    artifact: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    private = artifact / "private"
    partial_rows = private / "candidate-rows.partial.jsonl"
    partial_telemetry = private / "generation-telemetry.partial.jsonl"
    ledger_path = private / "deterministic-operation-ledger.jsonl"
    carry_rows = base._read_jsonl(  # noqa: SLF001
        Path(plan["carry_forward"]["candidate_rows_path"])
    )
    carry_telemetry = base._read_jsonl(  # noqa: SLF001
        Path(plan["carry_forward"]["telemetry_path"])
    )
    if len(carry_rows) != CARRY_RECORDS or len(carry_telemetry) != CARRY_RECORDS:
        raise ValueError("deterministic multilingual carry-forward count changed")
    present = (partial_rows.exists(), partial_telemetry.exists())
    if any(present) and not all(present):
        raise ValueError("deterministic multilingual partial pair is incomplete")
    if not any(present):
        base.write_jsonl(partial_rows, carry_rows)
        base.write_jsonl(partial_telemetry, carry_telemetry)
    rows = base._read_jsonl(partial_rows)  # noqa: SLF001
    telemetry = base._read_jsonl(partial_telemetry)  # noqa: SLF001
    ledger = base._read_jsonl(ledger_path)  # noqa: SLF001
    if rows[:CARRY_RECORDS] != carry_rows or telemetry[:CARRY_RECORDS] != carry_telemetry:
        raise ValueError("deterministic multilingual exact carry-forward changed")
    if len(rows) != len(telemetry) or not CARRY_RECORDS <= len(rows) <= 80:
        raise ValueError("deterministic multilingual row and telemetry counts differ")
    if len(ledger) != len(rows) - CARRY_RECORDS or len(ledger) > NEW_RECORDS:
        raise ValueError("deterministic multilingual operation ledger changed")
    if [str(row.get("record_id")) for row in rows] != list(
        base.expected_record_ids()[: len(rows)]
    ):
        raise ValueError("deterministic multilingual rows are not an exact prefix")
    scenarios = {(item.task, item.variant): item for item in base.SCENARIOS}
    for index in range(CARRY_RECORDS, len(rows)):
        record_id = base.expected_record_ids()[index]
        task, language_mode, variant_text = record_id.removeprefix("lumi-teacher-").rsplit(
            "-", 2
        )
        expected_row, expected_telemetry = make_deterministic_template_row(
            record_id,
            scenarios[(task, int(variant_text))],
            language_mode,
        )
        if rows[index] != expected_row or telemetry[index] != expected_telemetry:
            raise ValueError("existing deterministic multilingual partial changed")
    return rows, telemetry, ledger


class _DirectTokenizerAdapter:
    def __init__(self, tokenizer: Any, special_ids: list[int]) -> None:
        self._tokenizer = tokenizer
        self.all_special_ids = special_ids

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        split_special_tokens: bool,
    ) -> list[int]:
        del split_special_tokens
        return list(self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids)

    def get_vocab(self) -> Mapping[str, int]:
        return self._tokenizer.get_vocab()


def _load_direct_tokenizer(path: str | Path) -> _DirectTokenizerAdapter:
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError(
            "exact tokenizers runtime is absent; install only the separately authorized "
            "hashed dependency overlay"
        ) from error
    root = Path(path)
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    special_ids = sorted(
        int(token["id"])
        for token in json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))[
            "added_tokens"
        ]
        if token.get("special") is True
    )
    return _DirectTokenizerAdapter(tokenizer, special_ids)


@contextmanager
def _direct_tokenizer_runtime() -> Iterator[None]:
    original = base._load_tokenizer  # noqa: SLF001
    base._load_tokenizer = _load_direct_tokenizer  # noqa: SLF001
    try:
        yield
    finally:
        base._load_tokenizer = original  # noqa: SLF001


def run_deterministic_multilingual_completion(
    *,
    plan_path: str | Path,
    repository_root: str | Path,
    artifact_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    artifact = Path(artifact_root).resolve()
    plan = base.load_canary_plan(plan_path, repository_root=root)
    _validate_plan(plan, artifact=artifact)
    started_wall = time.time()
    started_monotonic = time.monotonic()
    rows, telemetry, ledger = _initialize(plan, artifact=artifact)
    private = artifact / "private"
    ledger_path = private / "deterministic-operation-ledger.jsonl"
    expected_ids = base.expected_record_ids()
    scenarios = {(item.task, item.variant): item for item in base.SCENARIOS}

    while len(rows) < DETERMINISTIC_COMPLETION_CONTRACT["total_retained_records"]:
        if (time.monotonic() - started_monotonic) / 60 > DETERMINISTIC_COMPLETION_CONTRACT[
            "maximum_runtime_minutes"
        ]:
            raise RuntimeError("deterministic multilingual completion exceeded runtime")
        record_id = expected_ids[len(rows)]
        task, language_mode, variant_text = record_id.removeprefix("lumi-teacher-").rsplit(
            "-", 2
        )
        scenario = scenarios[(task, int(variant_text))]
        row, item_telemetry = make_deterministic_template_row(
            record_id,
            scenario,
            language_mode,
        )
        rows.append(row)
        telemetry.append(item_telemetry)
        ledger.append(
            {
                "operation_number": len(ledger) + 1,
                "record_id": record_id,
                "request_sha256": item_telemetry["request_sha256"],
                "response_envelope_sha256": item_telemetry["response_envelope_sha256"],
                "status": "accepted_deterministic_zero_inference",
            }
        )
        base.write_jsonl(private / "candidate-rows.partial.jsonl", rows)
        base.write_jsonl(private / "generation-telemetry.partial.jsonl", telemetry)
        base.write_jsonl(ledger_path, ledger)

    base.write_jsonl(private / "candidate-rows.jsonl", rows)
    base.write_jsonl(private / "generation-telemetry.jsonl", telemetry)
    bindings = plan["bindings"]
    with _direct_tokenizer_runtime():
        audit = base.audit_rows(
            rows,
            repository_root=root,
            quality_config_path=root / str(bindings["data_quality"]["path"]),
            chat_template_path=root / str(bindings["chat_template"]["path"]),
            tokenizer_path=str(plan["tokenizer"]["path"]),
            maximum_supervised_tokens=DETERMINISTIC_COMPLETION_CONTRACT[
                "maximum_retained_supervised_tokens_total"
            ],
        )
    owner_sample = base.select_owner_sample(rows)
    owner_sample_sha256 = base.sha256_value(owner_sample)
    base.write_jsonl(private / "owner-review-sample-32.jsonl", owner_sample)
    review = artifact / "review"
    review.mkdir(parents=True, exist_ok=True)
    (review / "owner-review-32.html").write_text(
        build_owner_review_html(owner_sample, sample_sha256=owner_sample_sha256),
        encoding="utf-8",
    )
    report = {
        "schema_version": base.REPORT_SCHEMA_VERSION,
        "status": "verified_private_candidate_pending_owner_review",
        "production_eligible": False,
        "candidate_registry_changed": False,
        "dataset_sealed": False,
        "external_api_used": False,
        "human_authorship_claimed": False,
        "sft_started": False,
        "authorization_file_sha256": file_sha256(artifact / "authorization.json"),
        "plan_file_sha256": file_sha256(plan_path),
        "carry_forward": {
            "records": CARRY_RECORDS,
            "source_candidate_rows_file_sha256": plan["carry_forward"][
                "candidate_rows_sha256"
            ],
            "source_failure_report_sha256": plan["carry_forward"][
                "failure_report_sha256"
            ],
        },
        "deterministic_completion": {
            "additional_records_retained": NEW_RECORDS,
            "deterministic_operations": len(ledger),
            "local_inference_calls": 0,
            "operation_ledger_internal_sha256": base.sha256_value(ledger),
            "template_contract": TEMPLATE_CONTRACT,
            "template_method": TEMPLATE_METHOD,
        },
        "model": {
            "file_sha256": base.MODEL_SHA256,
            "id": base.MODEL_ID,
            "revision": base.MODEL_REVISION,
            "runtime": "not_used_for_attempt_018_deterministic_completion",
        },
        "audit": audit,
        "generation": {
            "calls": 0,
            "deterministic_materializations": NEW_RECORDS,
            "telemetry_internal_sha256": base.sha256_value(telemetry),
        },
        "owner_review": {
            "records": len(owner_sample),
            "sample_internal_sha256": owner_sample_sha256,
            "status": "pending",
        },
        "runtime": {
            "elapsed_seconds": round(time.monotonic() - started_monotonic, 6),
            "finished_unix": int(time.time()),
            "started_unix": int(started_wall),
        },
        "private_artifacts": {
            "candidate_rows_file_sha256": file_sha256(private / "candidate-rows.jsonl"),
            "generation_telemetry_file_sha256": file_sha256(
                private / "generation-telemetry.jsonl"
            ),
            "owner_review_html_file_sha256": file_sha256(review / "owner-review-32.html"),
            "owner_review_sample_file_sha256": file_sha256(
                private / "owner-review-sample-32.jsonl"
            ),
        },
        "operation_ledger_file_sha256": file_sha256(ledger_path),
    }
    base.write_json(artifact / "evidence" / "run-report.json", report)
    return report


def _verify_deterministic_row(
    row: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    *,
    scenario: base.Scenario,
    language_mode: str,
) -> None:
    expected_row, expected_telemetry = make_deterministic_template_row(
        str(row.get("record_id")),
        scenario,
        language_mode,
    )
    if row != expected_row or telemetry != expected_telemetry:
        raise ValueError("frozen deterministic multilingual row changed")


def verify_deterministic_multilingual_completion(
    *,
    plan_path: str | Path,
    repository_root: str | Path,
    artifact_root: str | Path,
) -> dict[str, Any]:
    artifact = Path(artifact_root).resolve()
    with _direct_tokenizer_runtime():
        verification = base.verify_run(
            plan_path=plan_path,
            repository_root=repository_root,
            artifact_root=artifact,
        )
    plan = base.load_canary_plan(plan_path, repository_root=repository_root)
    _validate_plan(plan, artifact=artifact)
    report = base._json_object(artifact / "evidence" / "run-report.json")  # noqa: SLF001
    rows = base._read_jsonl(artifact / "private" / "candidate-rows.jsonl")  # noqa: SLF001
    telemetry = base._read_jsonl(  # noqa: SLF001
        artifact / "private" / "generation-telemetry.jsonl"
    )
    ledger_path = artifact / "private" / "deterministic-operation-ledger.jsonl"
    ledger = base._read_jsonl(ledger_path)  # noqa: SLF001
    carry_rows = base._read_jsonl(  # noqa: SLF001
        Path(plan["carry_forward"]["candidate_rows_path"])
    )
    carry_telemetry = base._read_jsonl(  # noqa: SLF001
        Path(plan["carry_forward"]["telemetry_path"])
    )
    if rows[:CARRY_RECORDS] != carry_rows or telemetry[:CARRY_RECORDS] != carry_telemetry:
        raise ValueError("deterministic multilingual final carry-forward changed")
    if len(rows) != 80 or len(telemetry) != 80 or len(ledger) != NEW_RECORDS:
        raise ValueError("deterministic multilingual final counts changed")
    scenarios = {(item.task, item.variant): item for item in base.SCENARIOS}
    for index in range(CARRY_RECORDS, 80):
        record_id = base.expected_record_ids()[index]
        task, language_mode, variant_text = record_id.removeprefix("lumi-teacher-").rsplit(
            "-", 2
        )
        _verify_deterministic_row(
            rows[index],
            telemetry[index],
            scenario=scenarios[(task, int(variant_text))],
            language_mode=language_mode,
        )
    if any(item.get("status") != "accepted_deterministic_zero_inference" for item in ledger):
        raise ValueError("deterministic multilingual operation status changed")
    completion = report.get("deterministic_completion", {})
    if completion.get("deterministic_operations") != NEW_RECORDS:
        raise ValueError("deterministic multilingual report operation count changed")
    if completion.get("local_inference_calls") != 0:
        raise ValueError("deterministic multilingual report claims local inference")
    if completion.get("operation_ledger_internal_sha256") != base.sha256_value(ledger):
        raise ValueError("deterministic multilingual operation ledger changed")
    if report.get("operation_ledger_file_sha256") != file_sha256(ledger_path):
        raise ValueError("deterministic multilingual operation ledger file changed")
    verification["checks"]["carry_forward_exact_sixty_five_rows"] = True
    verification["checks"]["fifteen_frozen_multilingual_templates_verified"] = True
    verification["checks"]["zero_local_teacher_inference"] = True
    verification["checks"]["templates_are_AI_assisted_not_human_authored"] = True
    verification["checks"]["all_required_literals_byte_exact"] = True
    verification["deterministic_template_records_verified"] = NEW_RECORDS
    verification["local_inference_calls_verified"] = 0
    verification["operation_ledger_file_sha256"] = file_sha256(ledger_path)
    base.write_json(artifact / "evidence" / "verification-report.json", verification)
    return verification
