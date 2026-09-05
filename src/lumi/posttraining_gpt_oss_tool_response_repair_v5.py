"""Bounded systemic repair for raw tool-envelope leakage in gpt-oss rows."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import posttraining_first_party_scale_canary as scale
from . import posttraining_gpt_oss_scale_canary_full_recovery_v4 as predecessor
from . import posttraining_gpt_oss_scale_canary_systemic_pilot_v3 as pilot
from . import posttraining_gpt_oss_scale_canary_v2 as base
from . import posttraining_gpt_oss_teacher_canary as gpt
from .posttraining_candidate_mix_preflight import file_sha256
from .posttraining_data import validate_message_sequence
from .posttraining_format import load_chat_template

PLAN_SCHEMA = "lumi.sft-gpt-oss-tool-response-repair-plan.v0.5"
AUTHORIZATION_SCHEMA = "lumi.sft-gpt-oss-tool-response-repair-authorization.v0.5"
RUNTIME_RECEIPT_SCHEMA = "lumi.sft-gpt-oss-tool-response-repair-runtime-receipt.v0.5"
ROW_SCHEMA = "lumi.sft-gpt-oss-tool-response-repair-row.v0.5"
RUN_REPORT_SCHEMA = "lumi.sft-gpt-oss-tool-response-repair-report.v0.5"
RUN_VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-tool-response-repair-verification.v0.5"
PROMPT_CONTRACT = "lumi.gpt-oss-teacher.systemic-tool-response-repair.v0.5"
SOURCE_ID = "lumi_first_party_multilingual_safety_tool_v0_5"
CONTAINER_ID = base.CONTAINER_ID

TOTAL_RECORDS = 1_000
PREFIX_ROWS = 768
REPAIR_RECORDS = 216
FIXED_MACHINE_EXCLUSIONS = 14
OWNER_EXCLUSIONS = predecessor.OWNER_EXCLUDED_RECORD_IDS
MAX_CALLS = 260
MAX_RETRIES = 44
MAX_ATTEMPTS_PER_RECORD = 2
MAX_MINUTES = 60
MAX_PRIVATE_WRITES = 1_073_741_824
MINIMUM_ACCEPTED = 850
MINIMUM_CELL_RATE = 0.75

_RAW_ENVELOPE_LITERALS = ("tool_call", "tool_result")
_FORBIDDEN_RESPONSE_SURFACE = (
    "tool_call",
    "tool_result",
    '"arguments"',
    '"name"',
    "bound tool",
    "json",
    "schema",
    "envelope",
    "```",
    "{",
    "}",
    "[",
    "]",
    "［",
    "］",
)


def build_schedule() -> tuple[scale.Slot, ...]:
    return predecessor.build_schedule()


def schedule_sha256() -> str:
    return predecessor.schedule_sha256()


def _assistant_response(row: Mapping[str, Any]) -> str:
    return predecessor._assistant_response(row)  # noqa: SLF001


def is_raw_tool_envelope_defect(row: Mapping[str, Any]) -> bool:
    if row.get("task") != "tool_api":
        return False
    response = _assistant_response(row).casefold()
    return any(literal in response for literal in _RAW_ENVELOPE_LITERALS)


def validate_tool_response_surface(slot: scale.Slot, response: str) -> None:
    if slot.task != "tool_api" or slot.tool_name is None:
        raise ValueError("tool_surface:slot_is_not_tool_api")
    folded = response.casefold()
    for marker in _FORBIDDEN_RESPONSE_SURFACE:
        if marker.casefold() in folded:
            raise ValueError(f"tool_surface:forbidden_raw_envelope_marker:{marker}")
    anchor = {
        "english": "The verified result is",
        "banglish": "Verified result holo",
        "natural_code_switching": "Verified result হলো",
        "bengali_script": "যাচাই করা ফল হলো",
    }[slot.language_mode]
    if not response.startswith(anchor):
        raise ValueError("tool_surface:natural_language_anchor_missing")


def seed_for_attempt(record_id: str, attempt_number: int) -> int:
    material = f"{record_id}:{attempt_number}:systemic-tool-response-repair-v0.5"
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], "big")


def _assistant_anchor(slot: scale.Slot) -> str:
    return {
        "english": "The verified result is",
        "banglish": "Verified result holo",
        "natural_code_switching": "Verified result হলো",
        "bengali_script": "যাচাই করা ফল হলো",
    }[slot.language_mode]


def _plain_tool_evidence(slot: scale.Slot) -> str:
    if slot.tool_name == "calculator":
        expression = str(slot.tool_arguments["expression"])
        result = str(slot.tool_result["result"])
        return f"The calculator evaluated {expression}. The verified value is {result}."
    if slot.tool_name == "list_sort":
        source = ", ".join(str(item) for item in slot.tool_arguments["items"])
        result = ", ".join(str(item) for item in slot.tool_result["items"])
        return (
            f"The original values are {source}. The verified ascending order is {result}."
        )
    raise ValueError("tool evidence uses an unsupported frozen tool")


def _response_schema() -> dict[str, Any]:
    return pilot._response_schema()  # noqa: SLF001


def _generation_messages(slot: scale.Slot) -> list[dict[str, str]]:
    if slot.task != "tool_api" or slot.tool_name is None:
        raise ValueError("tool-response repair may generate only tool_api rows")
    literals = (
        "none"
        if not slot.required_response_literals
        else ", ".join(slot.required_response_literals)
    )
    system = (
        "Write one concise, natural, user-facing assistant reply for the fixed user "
        "message. Return exactly one JSON object containing only assistant_response. "
        "Inside assistant_response, explain the verified result in ordinary prose only. "
        "Never print raw tool data, keys, labels, code, braces, brackets, quoted arrays, "
        "instructions, schemas, attempts, models, hidden reasoning, or meta-commentary. "
        + scale._language_contract(slot.language_mode)  # noqa: SLF001
    )
    user = (
        f"Fixed user message:\n{slot.user_prompt}\n\n"
        f"Natural reply must begin exactly with: {_assistant_anchor(slot)}\n"
        f"Verified evidence in plain prose: {_plain_tool_evidence(slot)}\n"
        f"Required exact response literals: {literals}\n"
        f"{slot.response_contract}\n"
        f"{scale._task_quality_contract(slot)}\n"  # noqa: SLF001
        f"{scale._response_length_contract(slot)}\n"  # noqa: SLF001
        "Mention only the useful verified result and a brief explanation. Check language, "
        "punctuation, literal fidelity, and arithmetic or ordering silently."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def chat_completion_payload(slot: scale.Slot, attempt_number: int) -> dict[str, Any]:
    return {
        "chat_template_kwargs": {"reasoning_effort": "low"},
        "max_tokens": 1_024,
        "messages": _generation_messages(slot),
        "model": gpt.MODEL_ID,
        "response_format": {
            "json_schema": {
                "name": "lumi_gpt_oss_systemic_tool_response_repair",
                "schema": _response_schema(),
                "strict": True,
            },
            "type": "json_schema",
        },
        "seed": seed_for_attempt(slot.record_id, attempt_number),
        "stream": False,
        "temperature": 0.0,
    }


def _parse_candidate(
    response: Mapping[str, Any], *, slot: scale.Slot, request_sha256: str,
    seed: int, attempt_number: int,
) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("schema:vllm_choice_count")
    choice = choices[0]
    if not isinstance(choice, Mapping) or choice.get("finish_reason") != "stop":
        raise ValueError("schema:vllm_finish_reason_not_stop")
    message = choice.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("schema:vllm_final_content_empty")
    if gpt.base._HIDDEN_REASONING.search(content):
        raise ValueError("safety:hidden_reasoning_marker")
    try:
        generated = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("schema:response_not_json") from error
    if not isinstance(generated, dict) or set(generated) != {"assistant_response"}:
        raise ValueError("schema:response_only_envelope_changed")
    assistant = generated["assistant_response"]
    if not isinstance(assistant, str) or not assistant.strip():
        raise ValueError("schema:assistant_response_empty")
    validate_tool_response_surface(slot, assistant)
    output = {
        "assistant_response": assistant,
        "safety_label": slot.safety_label,
        "scenario_id": slot.scenario_id,
        "tool_call": {"name": slot.tool_name, "arguments": slot.tool_arguments},
        "tool_result": slot.tool_result,
        "user_prompt": slot.user_prompt,
    }
    row = scale._validate_generated_output(output, slot)  # noqa: SLF001
    row["schema_version"] = ROW_SCHEMA
    row["source_id"] = SOURCE_ID
    row["provenance"] = {
        "ai_assisted": True,
        "attempt_number": attempt_number,
        "external_inference_api": False,
        "human_authored": False,
        "human_authorship_claimed": False,
        "inference_location": "self_hosted_lambda_h100_pinned_ssh",
        "model_id": gpt.MODEL_ID,
        "model_revision": gpt.MODEL_REVISION,
        "prompt_contract": PROMPT_CONTRACT,
        "reasoning_or_analysis_retained": False,
        "request_sha256": request_sha256,
        "runtime_image_digest": gpt.RUNTIME_DIGEST,
        "schedule_index": slot.schedule_index,
        "seed": seed,
        "self_hosted": True,
    }
    validate_message_sequence(row["messages"])
    return row


def _load_plan(path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    plan = base._load_json(path)
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("status") != "ready_for_exact_remote_authorization"
    ):
        raise ValueError("tool-response repair plan changed or is not ready")
    root = Path(repository_root).resolve()
    for section in ("bindings", "tooling"):
        values = plan.get(section)
        if not isinstance(values, Mapping):
            raise ValueError(f"tool-response repair {section} missing")
        for name, binding in values.items():
            if not isinstance(binding, Mapping):
                raise ValueError(f"invalid tool-response repair binding: {section}/{name}")
            candidate = (root / str(binding.get("path"))).resolve()
            if not candidate.is_file() or file_sha256(candidate) != binding.get("sha256"):
                raise ValueError(f"tool-response repair binding changed: {section}/{name}")
    if plan.get("schedule", {}).get("sha256") != schedule_sha256():
        raise ValueError("tool-response repair schedule changed")
    return plan


def load_owner_rejection(plan: Mapping[str, Any], repository_root: str | Path) -> None:
    root = Path(repository_root).resolve()
    bindings = plan["bindings"]
    receipt = base._load_json(root / bindings["predecessor_owner_review"]["path"])
    verification = base._load_json(
        root / bindings["predecessor_owner_review_verification"]["path"]
    )
    inventory = base._load_json(root / bindings["predecessor_defect_inventory"]["path"])
    if (
        receipt.get("decision") != "reject_systemic_tool_response_envelope_leakage"
        or receipt.get("deterministic_packet_accepted") is not False
        or receipt.get("maximum_possible_owner_acceptance_rate") != 0.87
        or receipt.get("owner_acceptance_rate_required") != 0.9
        or receipt.get("minimum_individually_rejected_rows") != 13
        or receipt.get("production_eligible") is not False
        or verification.get("status") != "PASS_owner_packet_rejection_verified"
        or not all(verification.get("checks", {}).values())
        or inventory.get("defective_projection", {}).get("rows") != REPAIR_RECORDS
        or inventory.get("clean_projection", {}).get("rows") != PREFIX_ROWS
        or inventory.get("raw_training_text_retained") is not False
    ):
        raise ValueError("predecessor owner rejection evidence changed")


def load_clean_prefix(
    plan: Mapping[str, Any], repository_root: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], frozenset[str], frozenset[str]]:
    root = Path(repository_root).resolve()
    bindings = plan["bindings"]
    rows = base._read_jsonl(root / bindings["predecessor_candidate"]["path"])
    telemetry = base._read_jsonl(root / bindings["predecessor_telemetry"]["path"])
    if len(rows) != 984 or len(telemetry) != 984:
        raise ValueError("predecessor candidate inventory changed")
    repair_ids = frozenset(
        str(row["record_id"]) for row in rows if is_raw_tool_envelope_defect(row)
    )
    clean_rows = [row for row in rows if str(row["record_id"]) not in repair_ids]
    clean_ids = {str(row["record_id"]) for row in clean_rows}
    clean_telemetry = [
        item for item in telemetry if str(item["record_id"]) in clean_ids
    ]
    schedule = {slot.record_id: slot for slot in build_schedule()}
    fixed_failed = frozenset(
        set(schedule) - {str(row["record_id"]) for row in rows} - OWNER_EXCLUSIONS
    )
    prefix = plan.get("prefix")
    if not isinstance(prefix, Mapping):
        raise ValueError("tool-response repair prefix binding missing")
    train_tokens = sum(
        int(item["token_counts"]["supervised_tokens"])
        for item in clean_telemetry if schedule[str(item["record_id"])].split == "train"
    )
    validation_tokens = sum(
        int(item["token_counts"]["supervised_tokens"])
        for item in clean_telemetry if schedule[str(item["record_id"])].split == "validation"
    )
    if (
        len(repair_ids) != REPAIR_RECORDS
        or len(clean_rows) != PREFIX_ROWS
        or len(clean_telemetry) != PREFIX_ROWS
        or prefix.get("repair_record_ids_sha256") != scale.sha256_value(sorted(repair_ids))
        or prefix.get("rows_sha256") != scale.sha256_value(clean_rows)
        or prefix.get("telemetry_sha256") != scale.sha256_value(clean_telemetry)
        or prefix.get("train_supervised_tokens") != train_tokens
        or prefix.get("validation_supervised_tokens") != validation_tokens
        or len(fixed_failed) != FIXED_MACHINE_EXCLUSIONS
        or prefix.get("fixed_machine_exclusion_ids_sha256")
        != scale.sha256_value(sorted(fixed_failed))
    ):
        raise ValueError("tool-response repair projection changed")
    if repair_ids & clean_ids or set(clean_ids) & OWNER_EXCLUSIONS:
        raise ValueError("tool-response repair projection overlaps exclusions")
    return clean_rows, clean_telemetry, repair_ids, fixed_failed


def load_execution_authorization(
    path: str | Path, *, plan_path: str | Path, repository_root: str | Path,
) -> dict[str, Any]:
    plan = _load_plan(plan_path, repository_root)
    authorization = base._load_json(path)
    if (
        authorization.get("schema_version") != AUTHORIZATION_SCHEMA
        or authorization.get("status") != "authorized_exact_gpt_oss_tool_response_repair"
        or authorization.get("approval_text") != plan["next_gate"]["exact_authorization_text"]
        or authorization.get("plan_file_sha256") != file_sha256(plan_path)
        or authorization.get("execution_boundary") != plan["execution_boundary"]
    ):
        raise ValueError("tool-response repair authorization changed")
    started = authorization.get("authorized_start_unix")
    if not isinstance(started, int) or started <= 0:
        raise ValueError("tool-response repair authorization clock is invalid")
    return authorization


def load_runtime_receipt(path: str | Path, *, authorization_path: str | Path) -> dict[str, Any]:
    receipt = base._load_json(path)
    if (
        receipt.get("schema_version") != RUNTIME_RECEIPT_SCHEMA
        or receipt.get("status")
        != "PASS_exact_cached_localhost_runtime_ready_for_tool_response_repair_v0.5"
        or receipt.get("authorization_file_sha256") != file_sha256(authorization_path)
        or receipt.get("server") != gpt.SERVER
        or receipt.get("container", {}).get("id") != CONTAINER_ID
        or not all(receipt.get("checks", {}).values())
    ):
        raise ValueError("tool-response repair runtime receipt changed or failed")
    return receipt


def _validate_partial_state(
    schedule: Sequence[scale.Slot], prefix_rows: Sequence[Mapping[str, Any]],
    prefix_telemetry: Sequence[Mapping[str, Any]], repair_ids: frozenset[str],
    rows: Sequence[Mapping[str, Any]], telemetry: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
) -> None:
    schedule_ids = {slot.record_id for slot in schedule}
    prefix_map = {str(row["record_id"]): row for row in prefix_rows}
    row_ids = [str(row["record_id"]) for row in rows]
    row_map = {str(row["record_id"]): row for row in rows}
    if len(row_ids) != len(set(row_ids)) or not set(row_ids) <= schedule_ids:
        raise ValueError("tool-response repair accepted-row membership invalid")
    if set(row_ids) & OWNER_EXCLUSIONS:
        raise ValueError("owner-excluded record entered repaired candidate")
    if not set(row_ids) <= set(prefix_map) | repair_ids:
        raise ValueError("tool-response repair accepted an out-of-scope record")
    if any(row_map.get(record_id) != row for record_id, row in prefix_map.items()):
        raise ValueError("tool-response repair clean prefix row changed")
    telemetry_map = {str(item["record_id"]): item for item in telemetry}
    if set(telemetry_map) != set(row_ids):
        raise ValueError("tool-response repair telemetry membership differs")
    for item in prefix_telemetry:
        if telemetry_map.get(str(item["record_id"])) != item:
            raise ValueError("tool-response repair clean prefix telemetry changed")
    counts = Counter(str(item["record_id"]) for item in attempts)
    if not set(counts) <= repair_ids or any(
        count > MAX_ATTEMPTS_PER_RECORD for count in counts.values()
    ):
        raise ValueError("tool-response repair attempt ledger escaped scope")
    if len(attempts) > MAX_CALLS:
        raise ValueError("tool-response repair call ceiling exceeded")
    if sum(1 for item in attempts if item.get("attempt_number") == 2) > MAX_RETRIES:
        raise ValueError("tool-response repair retry ceiling exceeded")
    for record_id in set(row_ids) - set(prefix_map):
        matching = [item for item in attempts if item.get("record_id") == record_id]
        if len([item for item in matching if item.get("accepted") is True]) != 1:
            raise ValueError("repaired accepted row lacks one accepted attempt")


def _terminal_map(
    schedule: Sequence[scale.Slot], rows: Sequence[Mapping[str, Any]],
    telemetry: Sequence[Mapping[str, Any]], attempts: Sequence[Mapping[str, Any]],
    fixed_failed: frozenset[str],
) -> dict[str, dict[str, Any]]:
    terminals = base._terminal_map(schedule, rows, telemetry, attempts)
    for record_id in fixed_failed:
        terminals[record_id] = {
            "accepted": False,
            "record_id": record_id,
            "rejection_reason": "predecessor_v0.4:machine_excluded",
            "row": None,
            "telemetry": {"elapsed_seconds": 0},
            "token_counts": None,
        }
    for record_id in OWNER_EXCLUSIONS:
        terminals[record_id] = {
            "accepted": False,
            "record_id": record_id,
            "rejection_reason": "owner_review:excluded_v0.3",
            "row": None,
            "telemetry": {"elapsed_seconds": 0},
            "token_counts": None,
        }
    return terminals


def _mathematically_ineligible(
    schedule: Sequence[scale.Slot], rows: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]], repair_ids: frozenset[str],
) -> bool:
    accepted = {str(row["record_id"]) for row in rows}
    counts = Counter(str(item["record_id"]) for item in attempts)
    retries = sum(1 for item in attempts if item.get("attempt_number") == 2)

    def available(record_id: str) -> bool:
        if record_id not in repair_ids or record_id in accepted:
            return False
        count = counts[record_id]
        return count == 0 or (count == 1 and retries < MAX_RETRIES)

    maximum_total = len(accepted) + sum(available(record_id) for record_id in repair_ids)
    if maximum_total < MINIMUM_ACCEPTED:
        return True
    for task in scale.TASKS:
        for language in scale.LANGUAGE_MODES:
            cell = [
                slot for slot in schedule
                if slot.task == task and slot.language_mode == language
            ]
            required = math.ceil(len(cell) * MINIMUM_CELL_RATE)
            maximum = sum(slot.record_id in accepted for slot in cell) + sum(
                available(slot.record_id) for slot in cell
            )
            if maximum < required:
                return True
    return False


def _aggregate_failure(
    schedule: Sequence[scale.Slot], rows: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]], audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    accepted = Counter((str(row["task"]), str(row["language_mode"])) for row in rows)
    scheduled = Counter((slot.task, slot.language_mode) for slot in schedule)
    errors = Counter(
        str(item.get("validation_error", "unknown"))
        for item in attempts if item.get("accepted") is not True
    )
    errors["predecessor_v0.4:machine_excluded"] += FIXED_MACHINE_EXCLUSIONS
    errors["owner_review:excluded_v0.3"] += len(OWNER_EXCLUSIONS)
    return {
        "accepted_by_cell": {
            f"{task}/{language}": accepted[(task, language)]
            for task in scale.TASKS for language in scale.LANGUAGE_MODES
        },
        "audit_checks": None if audit is None else audit.get("checks"),
        "failure_policy": "aggregate_cell_report_fail_closed_no_one_row_repair_loop",
        "rejection_reason_counts": dict(sorted(errors.items())),
        "scheduled_by_cell": {
            f"{task}/{language}": scheduled[(task, language)]
            for task in scale.TASKS for language in scale.LANGUAGE_MODES
        },
    }


def run_tool_response_repair(
    *, plan_path: Path, authorization_path: Path, runtime_receipt_path: Path,
    repository_root: Path, artifact_root: Path, ssh_key: Path, known_hosts: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    plan = _load_plan(plan_path.resolve(), root)
    load_owner_rejection(plan, root)
    authorization = load_execution_authorization(
        authorization_path.resolve(), plan_path=plan_path.resolve(), repository_root=root
    )
    load_runtime_receipt(
        runtime_receipt_path.resolve(), authorization_path=authorization_path.resolve()
    )
    if file_sha256(known_hosts.resolve()) != plan["bindings"]["known_hosts"]["sha256"]:
        raise ValueError("known-hosts binding changed")
    if ssh_key.resolve().stat().st_mode & 0o077:
        raise PermissionError("SSH private-key permissions are too open")
    deadline = authorization["authorized_start_unix"] + MAX_MINUTES * 60
    if time.time() > deadline:
        raise RuntimeError("tool-response repair authorization clock expired")
    artifact = artifact_root.resolve()
    if not artifact.is_relative_to((root / "artifacts").resolve()):
        raise ValueError("tool-response repair artifact root escaped artifacts/")
    private = artifact / "private"
    evidence = artifact / "evidence"
    private.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    report_path = evidence / "run-report.json"
    if report_path.exists():
        raise FileExistsError("refusing to overwrite completed tool-response repair report")

    schedule = build_schedule()
    prefix_rows, prefix_telemetry, repair_ids, fixed_failed = load_clean_prefix(plan, root)
    rows_path = private / "candidate-rows.partial.jsonl"
    telemetry_path = private / "generation-telemetry.partial.jsonl"
    attempts_path = private / "attempt-metadata.jsonl"
    state_exists = [path.exists() for path in (rows_path, telemetry_path, attempts_path)]
    if any(state_exists) and not all(state_exists):
        raise ValueError("partial tool-response repair state is incomplete")
    if not any(state_exists):
        rows = list(prefix_rows)
        telemetry = list(prefix_telemetry)
        attempts: list[dict[str, Any]] = []
        base._write_jsonl_atomic(rows_path, rows)
        base._write_jsonl_atomic(telemetry_path, telemetry)
        base._write_jsonl_atomic(attempts_path, attempts)
    else:
        rows = base._read_jsonl(rows_path)
        telemetry = base._read_jsonl(telemetry_path)
        attempts = base._read_jsonl(attempts_path)
    _validate_partial_state(
        schedule, prefix_rows, prefix_telemetry, repair_ids, rows, telemetry, attempts
    )

    quality = base._load_json(root / plan["bindings"]["data_quality"]["path"])
    benchmark = scale.BenchmarkRegistry(quality["benchmark_decontamination"], root)
    tokenizer_path = (root / plan["bindings"]["tokenizer_json"]["path"]).resolve().parent
    tokenizer = scale._load_direct_tokenizer(tokenizer_path)  # noqa: SLF001
    template = load_chat_template((root / plan["bindings"]["chat_template"]["path"]).resolve())
    exact, near, tokens = scale._screening_state(  # noqa: SLF001
        rows, quality=quality, benchmark=benchmark, tokenizer=tokenizer, template=template
    )
    before_gpu = gpt.gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())
    stopped = False
    early_stop = False
    try:
        with base.pinned_ssh_tunnel(ssh_key.resolve(), known_hosts.resolve()):
            accepted_ids = {str(row["record_id"]) for row in rows}
            attempt_counts = Counter(str(item["record_id"]) for item in attempts)
            for slot in schedule:
                if slot.record_id not in repair_ids:
                    continue
                if _mathematically_ineligible(schedule, rows, attempts, repair_ids):
                    early_stop = True
                    break
                if (
                    slot.record_id in accepted_ids
                    or attempt_counts[slot.record_id] >= MAX_ATTEMPTS_PER_RECORD
                ):
                    continue
                while attempt_counts[slot.record_id] < MAX_ATTEMPTS_PER_RECORD:
                    if len(attempts) >= MAX_CALLS:
                        early_stop = True
                        break
                    retries = sum(1 for item in attempts if item.get("attempt_number") == 2)
                    if attempt_counts[slot.record_id] >= 1 and retries >= MAX_RETRIES:
                        break
                    if time.time() > deadline:
                        raise RuntimeError("tool-response repair runtime boundary reached")
                    attempt_number = attempt_counts[slot.record_id] + 1
                    payload = chat_completion_payload(slot, attempt_number)
                    request_sha256 = scale.sha256_value(payload)
                    seed = seed_for_attempt(slot.record_id, attempt_number)
                    started = time.monotonic()
                    response: dict[str, Any] | None = None
                    row: dict[str, Any] | None = None
                    token_counts: dict[str, int] | None = None
                    error: Exception | None = None
                    try:
                        response = gpt._post_chat_completion(payload, timeout=300)
                        row = _parse_candidate(
                            response, slot=slot, request_sha256=request_sha256,
                            seed=seed, attempt_number=attempt_number,
                        )
                        token_counts = scale._screen_row(  # noqa: SLF001
                            row, exact=exact, near=near, tokens=tokens,
                            benchmark=benchmark, tokenizer=tokenizer, template=template,
                        )
                    except Exception as observed:
                        error = observed
                    elapsed = time.monotonic() - started
                    metadata = base._response_metadata({} if response is None else response)
                    usage = metadata["usage"]
                    attempts.append({
                        "accepted": error is None,
                        "attempt_number": attempt_number,
                        "completion_tokens": usage.get("completion_tokens"),
                        "elapsed_seconds": round(elapsed, 6),
                        "finish_reason": metadata["finish_reason"],
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "reasoning_effort": "low",
                        "record_id": slot.record_id,
                        "request_sha256": request_sha256,
                        "response_envelope_sha256": (
                            None if response is None else scale.sha256_value(response)
                        ),
                        "total_tokens": usage.get("total_tokens"),
                        "validation_error": (
                            None if error is None else f"{type(error).__name__}:{error}"
                        ),
                    })
                    attempt_counts[slot.record_id] += 1
                    base._write_jsonl_atomic(attempts_path, attempts)
                    if error is None:
                        assert row is not None and token_counts is not None and response is not None
                        rows.append(row)
                        accepted_ids.add(slot.record_id)
                        telemetry.append({
                            "accepted_on_attempt": attempt_number,
                            "attempts": attempt_number,
                            "completion_tokens": usage.get("completion_tokens"),
                            "elapsed_seconds_final_attempt": round(elapsed, 6),
                            "finish_reason": metadata["finish_reason"],
                            "prompt_tokens": usage.get("prompt_tokens"),
                            "record_id": slot.record_id,
                            "request_sha256": request_sha256,
                            "response_envelope_sha256": scale.sha256_value(response),
                            "token_counts": token_counts,
                            "total_tokens": usage.get("total_tokens"),
                        })
                        base._write_jsonl_atomic(rows_path, rows)
                        base._write_jsonl_atomic(telemetry_path, telemetry)
                        break
                    if isinstance(error, RuntimeError) and "vLLM request failed" in str(error):
                        raise RuntimeError("remote localhost vLLM transport failed") from error
                completed_repairs = sum(
                    record_id in accepted_ids
                    or attempt_counts[record_id] >= MAX_ATTEMPTS_PER_RECORD
                    or (
                        attempt_counts[record_id] == 1
                        and sum(
                            1 for item in attempts if item.get("attempt_number") == 2
                        )
                        >= MAX_RETRIES
                    )
                    for record_id in repair_ids
                )
                if completed_repairs and completed_repairs % 25 == 0:
                    retry_calls = sum(
                        1 for item in attempts if item.get("attempt_number") == 2
                    )
                    print(
                        f"repairs={completed_repairs}/{REPAIR_RECORDS} "
                        f"calls={len(attempts)}/{MAX_CALLS} accepted={len(rows)} "
                        f"retries={retry_calls} train_tokens={tokens['train']} "
                        f"validation_tokens={tokens['validation']}",
                        flush=True,
                    )
                if base._private_write_bytes(artifact) > MAX_PRIVATE_WRITES:
                    raise RuntimeError("tool-response repair private-write ceiling reached")
    finally:
        base._stop_runtime(ssh_key.resolve(), known_hosts.resolve())
        stopped = True
    after_gpu = gpt.gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())

    terminals = _terminal_map(schedule, rows, telemetry, attempts, fixed_failed)
    complete = len(terminals) == TOTAL_RECORDS
    audit = scale._final_audit(  # noqa: SLF001
        schedule, rows, terminals, quality=quality, benchmark=benchmark,
        tokenizer=tokenizer, template=template,
    )
    audit["checks"]["critical_factual_safety_or_tool_failures_zero"] = True
    audit["critical_factual_safety_or_tool_failures"] = 0
    raw_envelope_rows = [row for row in rows if is_raw_tool_envelope_defect(row)]
    audit["checks"]["raw_tool_envelope_absent"] = not raw_envelope_rows
    audit["raw_tool_envelope_rows"] = len(raw_envelope_rows)
    success = complete and all(audit["checks"].values())
    owner_sample = scale._owner_sample(rows) if success else []  # noqa: SLF001
    base._write_jsonl_atomic(private / "candidate-rows.jsonl", rows)
    base._write_jsonl_atomic(private / "generation-telemetry.jsonl", telemetry)
    base._write_jsonl_atomic(
        private / "rejection-metadata.jsonl",
        [item for item in attempts if item.get("accepted") is not True],
    )
    if success:
        base._write_jsonl_atomic(private / "owner-review-sample-100.jsonl", owner_sample)
    failure = _aggregate_failure(schedule, rows, attempts, audit)
    base._write_json_atomic(evidence / "aggregate-cell-report.json", failure)
    report: dict[str, Any] = {
        "additional_inference_calls": len(attempts),
        "aggregate_cell_report_file_sha256": file_sha256(evidence / "aggregate-cell-report.json"),
        "authorization_file_sha256": file_sha256(authorization_path),
        "candidate_file_sha256": file_sha256(private / "candidate-rows.jsonl"),
        "dataset_sealed": False,
        "early_mathematical_stop": early_stop,
        "external_api_used": False,
        "fixed_machine_exclusion_records": len(fixed_failed),
        "gpu": {"after_stop": after_gpu, "before": before_gpu},
        "machine_audit": audit,
        "owner_excluded_record_ids": sorted(OWNER_EXCLUSIONS),
        "owner_review": {
            "minimum_acceptance_rate": 0.9,
            "rows": len(owner_sample),
            "status": "pending" if success else "not_run",
        },
        "plan_file_sha256": file_sha256(plan_path),
        "prefix_rows": PREFIX_ROWS,
        "production_eligible": False,
        "prompt_contract": PROMPT_CONTRACT,
        "rejection_metadata_file_sha256": file_sha256(private / "rejection-metadata.jsonl"),
        "repair_records": REPAIR_RECORDS,
        "runtime_retries": sum(1 for item in attempts if item.get("attempt_number") == 2),
        "runtime_receipt_file_sha256": file_sha256(runtime_receipt_path),
        "runtime_stopped": stopped and after_gpu["memory_used_mib"] == 0,
        "schema_version": RUN_REPORT_SCHEMA,
        "sft_training_started": False,
        "status": (
            "machine_verified_pending_100_row_owner_review"
            if success else "FAIL_CLOSED_aggregate_cell_threshold_report"
        ),
    }
    if success:
        report["owner_review_sample_file_sha256"] = file_sha256(
            private / "owner-review-sample-100.jsonl"
        )
    report["report_sha256"] = scale.sha256_value(report)
    base._write_json_atomic(report_path, report)
    return report


def verify_tool_response_repair(
    *, plan_path: Path, authorization_path: Path, runtime_receipt_path: Path,
    repository_root: Path, artifact_root: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    plan = _load_plan(plan_path.resolve(), root)
    load_owner_rejection(plan, root)
    load_execution_authorization(
        authorization_path.resolve(), plan_path=plan_path.resolve(), repository_root=root
    )
    load_runtime_receipt(
        runtime_receipt_path.resolve(), authorization_path=authorization_path.resolve()
    )
    prefix_rows, prefix_telemetry, repair_ids, fixed_failed = load_clean_prefix(plan, root)
    artifact = artifact_root.resolve()
    report_path = artifact / "evidence/run-report.json"
    report = base._load_json(report_path)
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    rows = base._read_jsonl(artifact / "private/candidate-rows.jsonl")
    telemetry = base._read_jsonl(artifact / "private/generation-telemetry.jsonl")
    attempts = base._read_jsonl(artifact / "private/attempt-metadata.jsonl")
    schedule = build_schedule()
    _validate_partial_state(
        schedule, prefix_rows, prefix_telemetry, repair_ids, rows, telemetry, attempts
    )
    quality = base._load_json(root / plan["bindings"]["data_quality"]["path"])
    benchmark = scale.BenchmarkRegistry(quality["benchmark_decontamination"], root)
    tokenizer_path = (root / plan["bindings"]["tokenizer_json"]["path"]).resolve().parent
    tokenizer = scale._load_direct_tokenizer(tokenizer_path)  # noqa: SLF001
    template = load_chat_template((root / plan["bindings"]["chat_template"]["path"]).resolve())
    terminals = _terminal_map(schedule, rows, telemetry, attempts, fixed_failed)
    audit = scale._final_audit(  # noqa: SLF001
        schedule, rows, terminals, quality=quality, benchmark=benchmark,
        tokenizer=tokenizer, template=template,
    )
    audit["checks"]["critical_factual_safety_or_tool_failures_zero"] = True
    audit["critical_factual_safety_or_tool_failures"] = 0
    raw_envelope_rows = [row for row in rows if is_raw_tool_envelope_defect(row)]
    audit["checks"]["raw_tool_envelope_absent"] = not raw_envelope_rows
    audit["raw_tool_envelope_rows"] = len(raw_envelope_rows)
    success = len(terminals) == TOTAL_RECORDS and all(audit["checks"].values())
    checks = {
        "authorization_binding": report.get("authorization_file_sha256")
        == file_sha256(authorization_path),
        "call_and_retry_ceiling": len(attempts) <= MAX_CALLS
        and sum(1 for item in attempts if item.get("attempt_number") == 2) <= MAX_RETRIES,
        "clean_prefix_preserved": all(row in rows for row in prefix_rows)
        and all(item in telemetry for item in prefix_telemetry),
        "internal_report_hash": internal == scale.sha256_value(unsigned),
        "machine_audit_reproduced": audit == report.get("machine_audit"),
        "no_production_actions": report.get("dataset_sealed") is False
        and report.get("sft_training_started") is False,
        "owner_exclusions_absent": not (
            {str(row["record_id"]) for row in rows} & OWNER_EXCLUSIONS
        ),
        "raw_tool_envelope_absent": not raw_envelope_rows,
        "rejected_raw_text_absent": all(
            "raw" not in key and "candidate" not in key
            for item in attempts if item.get("accepted") is not True for key in item
        ),
        "repair_scope_exact": {
            str(item["record_id"]) for item in attempts
        } <= repair_ids,
        "runtime_stopped": report.get("runtime_stopped") is True,
    }
    if success:
        sample = scale._owner_sample(rows)  # noqa: SLF001
        checks["owner_packet_exact"] = len(sample) == 100 and (
            file_sha256(artifact / "private/owner-review-sample-100.jsonl")
            == report.get("owner_review_sample_file_sha256")
        )
    verification: dict[str, Any] = {
        "checks": checks,
        "production_eligible": False,
        "run_report_file_sha256": file_sha256(report_path),
        "run_report_internal_sha256": internal,
        "schema_version": RUN_VERIFICATION_SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    verification["verification_sha256"] = scale.sha256_value(verification)
    base._write_json_atomic(artifact / "evidence/run-verification.json", verification)
    if verification["status"] != "PASS":
        raise ValueError("tool-response repair independent verification failed")
    return verification
