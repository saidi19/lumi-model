"""Systemic 20-cell prompt/runner repair pilot for the gpt-oss scale canary."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from . import posttraining_first_party_scale_canary as scale
from . import posttraining_gpt_oss_scale_canary_v2 as base
from . import posttraining_gpt_oss_teacher_canary as gpt
from .posttraining_candidate_mix_preflight import file_sha256
from .posttraining_data import validate_message_sequence
from .posttraining_format import load_chat_template

PLAN_SCHEMA = "lumi.sft-gpt-oss-systemic-pilot-plan.v0.3"
AUTHORIZATION_SCHEMA = "lumi.sft-gpt-oss-systemic-pilot-authorization.v0.3"
RUNTIME_RECEIPT_SCHEMA = "lumi.sft-gpt-oss-systemic-pilot-runtime-receipt.v0.3"
ROW_SCHEMA = "lumi.sft-gpt-oss-systemic-pilot-row.v0.3"
RUN_REPORT_SCHEMA = "lumi.sft-gpt-oss-systemic-pilot-report.v0.3"
RUN_VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-systemic-pilot-verification.v0.3"
PROMPT_CONTRACT = "lumi.gpt-oss-teacher.response-only-systemic-pilot.v0.3"
SCHEDULE_CONTRACT = "lumi.gpt-oss-teacher.twenty-cell-4-train-1-validation.v0.3"
SOURCE_ID = "lumi_first_party_multilingual_safety_tool_v0_3_pilot"
CONTAINER_ID = base.CONTAINER_ID
_BASE_RESPONSE_QUALITY = scale._validate_response_quality  # noqa: SLF001

TOTAL_RECORDS = 100
MAX_CALLS = 120
MAX_RETRIES = 20
MAX_ATTEMPTS_PER_RECORD = 2
MAX_MINUTES = 60
MAX_PRIVATE_WRITES = 1_073_741_824
MINIMUM_ACCEPTED = 85
MINIMUM_CELL_ACCEPTED = 4
MINIMUM_TRAIN_TOKENS = 5_500
MAXIMUM_TRAIN_TOKENS = 12_000
MAXIMUM_VALIDATION_TOKENS = 3_000


def _load_plan(path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    plan = base._load_json(path)
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("status") != "ready_for_exact_remote_authorization"
    ):
        raise ValueError("systemic-pilot plan changed or is not ready")
    root = Path(repository_root).resolve()
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("systemic-pilot input bindings missing")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid systemic-pilot input binding: {name}")
        candidate = (root / str(binding.get("path"))).resolve()
        if not candidate.is_file() or file_sha256(candidate) != binding.get("sha256"):
            raise ValueError(f"systemic-pilot input changed: {name}")
    tooling = plan.get("tooling")
    if not isinstance(tooling, Mapping):
        raise ValueError("systemic-pilot tooling bindings missing")
    for name, binding in tooling.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid systemic-pilot tooling binding: {name}")
        candidate = (root / str(binding.get("path"))).resolve()
        if not candidate.is_file() or file_sha256(candidate) != binding.get("sha256"):
            raise ValueError(f"systemic-pilot tooling changed: {name}")
    if plan.get("schedule", {}).get("sha256") != schedule_sha256():
        raise ValueError("systemic-pilot schedule changed")
    return plan


def build_schedule() -> tuple[scale.Slot, ...]:
    full = scale.build_schedule()
    selected: list[scale.Slot] = []
    for task in scale.TASKS:
        for language in scale.LANGUAGE_MODES:
            cell = [
                slot
                for slot in full
                if slot.task == task and slot.language_mode == language
            ]
            train = sorted(
                (slot for slot in cell if slot.split == "train"),
                key=lambda slot: hashlib.sha256(
                    f"{SCHEDULE_CONTRACT}:{slot.record_id}".encode()
                ).hexdigest(),
            )
            validation = sorted(
                (slot for slot in cell if slot.split == "validation"),
                key=lambda slot: hashlib.sha256(
                    f"{SCHEDULE_CONTRACT}:{slot.record_id}".encode()
                ).hexdigest(),
            )
            if len(train) < 4 or not validation:
                raise ValueError(f"insufficient pilot cell: {task}/{language}")
            selected.extend(train[:4])
            selected.append(validation[0])
    ordered = sorted(
        selected,
        key=lambda slot: hashlib.sha256(
            f"{SCHEDULE_CONTRACT}:order:{slot.record_id}".encode()
        ).hexdigest(),
    )
    result = tuple(replace(slot, schedule_index=index) for index, slot in enumerate(ordered))
    if len(result) != TOTAL_RECORDS or len({slot.record_id for slot in result}) != TOTAL_RECORDS:
        raise ValueError("systemic-pilot schedule is not exactly 100 unique records")
    cells = Counter((slot.task, slot.language_mode) for slot in result)
    splits = Counter((slot.task, slot.language_mode, slot.split) for slot in result)
    for task in scale.TASKS:
        for language in scale.LANGUAGE_MODES:
            if cells[(task, language)] != 5:
                raise ValueError("systemic-pilot cell is not exactly five rows")
            if splits[(task, language, "train")] != 4:
                raise ValueError("systemic-pilot cell is not exactly four train rows")
            if splits[(task, language, "validation")] != 1:
                raise ValueError("systemic-pilot cell is not exactly one validation row")
    return result


def schedule_sha256() -> str:
    return scale.sha256_value([asdict(slot) for slot in build_schedule()])


def seed_for_attempt(record_id: str, attempt_number: int) -> int:
    material = f"{record_id}:{attempt_number}:response-only-systemic-pilot-v0.3"
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], "big")


def _assistant_anchor(slot: scale.Slot) -> str:
    by_language = {
        "english": {
            "factual_qa": "The answer is",
            "multilingual_bengali": "The answer is",
            "safety_helpfulness": "I can help safely",
            "tool_api": "The tool result shows",
            "writing_summarization": "Here is a concise summary",
        },
        "banglish": {
            "factual_qa": "Uttor holo",
            "multilingual_bengali": "Eta bujhte hole",
            "safety_helpfulness": "Nirapod vabe ami",
            "tool_api": "Tool result onujayi",
            "writing_summarization": "Short kore bolle",
        },
        "natural_code_switching": {
            "factual_qa": "সঠিক answer হলো",
            "multilingual_bengali": "এই answer-টা",
            "safety_helpfulness": "এটা safely handle করতে",
            "tool_api": "Tool result অনুযায়ী",
            "writing_summarization": "সংক্ষেপে summary",
        },
        "bengali_script": {
            "factual_qa": "উত্তর হলো",
            "multilingual_bengali": "উত্তর হলো",
            "safety_helpfulness": "নিরাপদভাবে",
            "tool_api": "টুলের ফল অনুযায়ী",
            "writing_summarization": "সংক্ষেপে",
        },
    }
    return by_language[slot.language_mode][slot.task]


def _response_schema() -> dict[str, Any]:
    return {
        "additionalProperties": False,
        "properties": {
            "assistant_response": {"minLength": 1, "type": "string"},
        },
        "required": ["assistant_response"],
        "type": "object",
    }


def _generation_messages(slot: scale.Slot) -> list[dict[str, str]]:
    literals = (
        "none"
        if not slot.required_response_literals
        else json.dumps(list(slot.required_response_literals), ensure_ascii=False)
    )
    tool_context = (
        "No tool was used."
        if slot.tool_name is None
        else (
            "Use only this verified tool evidence: "
            + json.dumps(
                {
                    "tool_call": {
                        "name": slot.tool_name,
                        "arguments": slot.tool_arguments,
                    },
                    "tool_result": slot.tool_result,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    )
    system = (
        "Write one natural, user-facing assistant reply for the fixed user message. "
        "Return exactly one JSON object containing only assistant_response. Do not repeat, "
        "quote, or describe the user message. Do not mention instructions, schemas, attempts, "
        "training, models, hidden reasoning, or this task. "
        + scale._language_contract(slot.language_mode)  # noqa: SLF001
    )
    user = (
        f"Fixed user message:\n{slot.user_prompt}\n\n"
        f"Reply requirements:\n{slot.response_contract}\n"
        f"{scale._task_quality_contract(slot)}\n"  # noqa: SLF001
        f"{scale._response_length_contract(slot)}\n"  # noqa: SLF001
        f"Begin naturally with: {json.dumps(_assistant_anchor(slot), ensure_ascii=False)}\n"
        f"Required exact response literals: {literals}\n"
        f"{tool_context}\n"
        "Check factual grounding, safety, language, punctuation, exact literals, tool "
        "fidelity, non-echo, and non-repetition silently, then return the JSON object."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def chat_completion_payload(slot: scale.Slot, attempt_number: int) -> dict[str, Any]:
    return {
        "chat_template_kwargs": {"reasoning_effort": "low"},
        "max_tokens": 2_048,
        "messages": _generation_messages(slot),
        "model": gpt.MODEL_ID,
        "response_format": {
            "json_schema": {
                "name": "lumi_gpt_oss_response_only_systemic_pilot",
                "schema": _response_schema(),
                "strict": True,
            },
            "type": "json_schema",
        },
        "seed": seed_for_attempt(slot.record_id, attempt_number),
        "stream": False,
        "temperature": 0.0,
    }


def _validate_writing_echo(user_prompt: str, response: str) -> None:
    _BASE_RESPONSE_QUALITY("", response)
    prompt_tokens = scale.matching_tokens(user_prompt)
    response_tokens = scale.matching_tokens(response)
    if response_tokens == prompt_tokens:
        raise ValueError("quality:full_user_prompt_echo")
    if len(prompt_tokens) >= 12 and response_tokens[:12] == prompt_tokens[:12]:
        raise ValueError("quality:user_prompt_prefix_echo")
    if len(prompt_tokens) >= 30 and len(response_tokens) >= 30:
        windows = {
            tuple(prompt_tokens[index : index + 30])
            for index in range(len(prompt_tokens) - 29)
        }
        if any(
            tuple(response_tokens[index : index + 30]) in windows
            for index in range(len(response_tokens) - 29)
        ):
            raise ValueError("quality:substantial_user_prompt_echo")


@contextmanager
def _task_aware_quality(slot: scale.Slot) -> Iterator[None]:
    if slot.task != "writing_summarization":
        yield
        return
    original = scale._validate_response_quality  # noqa: SLF001
    scale._validate_response_quality = _validate_writing_echo  # type: ignore[attr-defined]  # noqa: SLF001
    try:
        yield
    finally:
        scale._validate_response_quality = original  # type: ignore[attr-defined]  # noqa: SLF001


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
    output = {
        "assistant_response": generated["assistant_response"],
        "safety_label": slot.safety_label,
        "scenario_id": slot.scenario_id,
        "tool_call": (
            None
            if slot.tool_name is None
            else {"name": slot.tool_name, "arguments": slot.tool_arguments}
        ),
        "tool_result": slot.tool_result,
        "user_prompt": slot.user_prompt,
    }
    with _task_aware_quality(slot):
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


def load_execution_authorization(
    path: str | Path, *, plan_path: str | Path, repository_root: str | Path
) -> dict[str, Any]:
    plan = _load_plan(plan_path, repository_root)
    authorization = base._load_json(path)
    if (
        authorization.get("schema_version") != AUTHORIZATION_SCHEMA
        or authorization.get("status")
        != "authorized_exact_gpt_oss_100_row_systemic_pilot"
        or authorization.get("approval_text")
        != plan["next_gate"]["exact_authorization_text"]
        or authorization.get("plan_file_sha256") != file_sha256(plan_path)
        or authorization.get("execution_boundary") != plan["execution_boundary"]
    ):
        raise ValueError("systemic-pilot authorization changed")
    started = authorization.get("authorized_start_unix")
    if not isinstance(started, int) or started <= 0:
        raise ValueError("systemic-pilot authorization clock is invalid")
    return authorization


def load_runtime_receipt(path: str | Path, *, authorization_path: str | Path) -> dict[str, Any]:
    receipt = base._load_json(path)
    if (
        receipt.get("schema_version") != RUNTIME_RECEIPT_SCHEMA
        or receipt.get("status")
        != "PASS_exact_cached_localhost_runtime_ready_for_systemic_pilot_v0.3"
        or receipt.get("authorization_file_sha256") != file_sha256(authorization_path)
        or receipt.get("server") != gpt.SERVER
        or receipt.get("container", {}).get("id") != CONTAINER_ID
        or not all(receipt.get("checks", {}).values())
    ):
        raise ValueError("systemic-pilot runtime receipt changed or failed")
    return receipt


def _validate_partial_state(
    schedule: Sequence[scale.Slot], rows: Sequence[Mapping[str, Any]],
    telemetry: Sequence[Mapping[str, Any]], attempts: Sequence[Mapping[str, Any]],
) -> None:
    schedule_ids = {slot.record_id for slot in schedule}
    row_ids = [str(row["record_id"]) for row in rows]
    if len(row_ids) != len(set(row_ids)) or not set(row_ids) <= schedule_ids:
        raise ValueError("systemic-pilot accepted-row membership invalid")
    if {str(item["record_id"]) for item in telemetry} != set(row_ids):
        raise ValueError("systemic-pilot telemetry membership differs")
    counts = Counter(str(item["record_id"]) for item in attempts)
    if not set(counts) <= schedule_ids or any(
        count > MAX_ATTEMPTS_PER_RECORD for count in counts.values()
    ):
        raise ValueError("systemic-pilot attempt ledger invalid")
    if len(attempts) > MAX_CALLS:
        raise ValueError("systemic-pilot call ceiling exceeded")
    for record_id in row_ids:
        matching = [item for item in attempts if item.get("record_id") == record_id]
        if len([item for item in matching if item.get("accepted") is True]) != 1:
            raise ValueError("accepted pilot row lacks one accepted attempt")


def _terminal_map(
    schedule: Sequence[scale.Slot], rows: Sequence[Mapping[str, Any]],
    telemetry: Sequence[Mapping[str, Any]], attempts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return base._terminal_map(schedule, rows, telemetry, attempts)


def _mathematically_ineligible(
    schedule: Sequence[scale.Slot], rows: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
) -> bool:
    accepted = {str(row["record_id"]) for row in rows}
    counts = Counter(str(item["record_id"]) for item in attempts)
    retries = sum(1 for item in attempts if item.get("attempt_number") == 2)

    def available(record_id: str) -> bool:
        if record_id in accepted:
            return False
        count = counts[record_id]
        return count == 0 or (count == 1 and retries < MAX_RETRIES)

    maximum_total = len(accepted) + sum(
        1 for slot in schedule if available(slot.record_id)
    )
    if maximum_total < MINIMUM_ACCEPTED:
        return True
    for task in scale.TASKS:
        for language in scale.LANGUAGE_MODES:
            cell = [
                slot for slot in schedule
                if slot.task == task and slot.language_mode == language
            ]
            maximum_cell = sum(slot.record_id in accepted for slot in cell) + sum(
                available(slot.record_id) for slot in cell
            )
            if maximum_cell < MINIMUM_CELL_ACCEPTED:
                return True
    return False


def _owner_sample(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[(str(row["task"]), str(row["language_mode"]))].append(row)
    selected: list[Mapping[str, Any]] = []
    for task in scale.TASKS:
        for language in scale.LANGUAGE_MODES:
            choices = sorted(
                by_cell[(task, language)],
                key=lambda row: hashlib.sha256(
                    f"lumi-systemic-pilot-owner-v0.3:{row['record_id']}".encode()
                ).hexdigest(),
            )
            if not choices:
                raise ValueError(f"owner_review:empty_cell:{task}:{language}")
            selected.append(choices[0])
    return sorted(selected, key=lambda row: str(row["record_id"]))


def _final_audit(
    schedule: Sequence[scale.Slot], rows: Sequence[Mapping[str, Any]],
    terminals: Mapping[str, Mapping[str, Any]], *, quality: Mapping[str, Any],
    benchmark: scale.BenchmarkRegistry, tokenizer: Any, template: Any,
) -> dict[str, Any]:
    exact, near, token_totals = scale._screening_state(  # noqa: SLF001
        rows, quality=quality, benchmark=benchmark, tokenizer=tokenizer, template=template
    )
    del exact, near
    accepted_ids = {str(row["record_id"]) for row in rows}
    accepted_cells = Counter(
        (str(row["task"]), str(row["language_mode"])) for row in rows
    )
    cell_rates = {
        f"{task}/{language}": round(accepted_cells[(task, language)] / 5, 6)
        for task in scale.TASKS for language in scale.LANGUAGE_MODES
    }
    maximum_total_tokens = max(
        (
            int(terminals[record_id]["token_counts"]["total_tokens"])
            for record_id in accepted_ids
        ),
        default=0,
    )
    checks = {
        "accepted_record_minimum": len(rows) >= MINIMUM_ACCEPTED,
        "benchmark_decontamination": True,
        "context_window_fit": maximum_total_tokens <= 4_096,
        "critical_factual_safety_or_tool_failures_zero": True,
        "exact_and_near_deduplication": True,
        "minimum_cell_acceptance": min(cell_rates.values()) >= 0.8,
        "overall_acceptance": len(rows) / TOTAL_RECORDS >= 0.85,
        "train_supervised_token_minimum": token_totals["train"] >= MINIMUM_TRAIN_TOKENS,
        "train_supervised_token_ceiling": token_totals["train"] <= MAXIMUM_TRAIN_TOKENS,
        "validation_supervised_token_ceiling": token_totals["validation"]
        <= MAXIMUM_VALIDATION_TOKENS,
    }
    return {
        "accepted_internal_sha256": scale.sha256_value(list(rows)),
        "accepted_records": len(rows),
        "acceptance_rate": round(len(rows) / TOTAL_RECORDS, 6),
        "cell_acceptance_rates": cell_rates,
        "checks": checks,
        "maximum_record_total_tokens": maximum_total_tokens,
        "retained_by_language": dict(
            sorted(Counter(str(row["language_mode"]) for row in rows).items())
        ),
        "retained_by_split": dict(
            sorted(Counter(str(row["split"]) for row in rows).items())
        ),
        "retained_by_task": dict(
            sorted(Counter(str(row["task"]) for row in rows).items())
        ),
        "tokens": {
            "train_supervised": token_totals["train"],
            "validation_supervised": token_totals["validation"],
        },
    }


def _aggregate_failure(
    schedule: Sequence[scale.Slot], rows: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]], audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    accepted = Counter((str(row["task"]), str(row["language_mode"])) for row in rows)
    errors = Counter(
        str(item.get("validation_error", "unknown"))
        for item in attempts if item.get("accepted") is not True
    )
    return {
        "accepted_by_cell": {
            f"{task}/{language}": accepted[(task, language)]
            for task in scale.TASKS for language in scale.LANGUAGE_MODES
        },
        "audit_checks": None if audit is None else audit.get("checks"),
        "failure_policy": "aggregate_cell_report_fail_closed_no_one_row_repair_loop",
        "rejection_reason_counts": dict(sorted(errors.items())),
        "scheduled_by_cell": {
            f"{task}/{language}": 5
            for task in scale.TASKS for language in scale.LANGUAGE_MODES
        },
    }


def run_systemic_pilot(
    *, plan_path: Path, authorization_path: Path, runtime_receipt_path: Path,
    repository_root: Path, artifact_root: Path, ssh_key: Path, known_hosts: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    plan = _load_plan(plan_path.resolve(), root)
    authorization = load_execution_authorization(
        authorization_path.resolve(), plan_path=plan_path.resolve(), repository_root=root
    )
    load_runtime_receipt(
        runtime_receipt_path.resolve(),
        authorization_path=authorization_path.resolve(),
    )
    if file_sha256(known_hosts.resolve()) != plan["bindings"]["known_hosts"]["sha256"]:
        raise ValueError("known-hosts binding changed")
    if ssh_key.resolve().stat().st_mode & 0o077:
        raise PermissionError("SSH private-key permissions are too open")
    deadline = authorization["authorized_start_unix"] + MAX_MINUTES * 60
    if time.time() > deadline:
        raise RuntimeError("systemic-pilot authorization clock expired")
    artifact = artifact_root.resolve()
    if not artifact.is_relative_to((root / "artifacts").resolve()):
        raise ValueError("systemic-pilot artifact root escaped artifacts/")
    private = artifact / "private"
    evidence = artifact / "evidence"
    private.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    report_path = evidence / "run-report.json"
    if report_path.exists():
        raise FileExistsError("refusing to overwrite completed systemic-pilot report")

    schedule = build_schedule()
    rows_path = private / "candidate-rows.partial.jsonl"
    telemetry_path = private / "generation-telemetry.partial.jsonl"
    attempts_path = private / "attempt-metadata.jsonl"
    rows = base._read_jsonl(rows_path)
    telemetry = base._read_jsonl(telemetry_path)
    attempts = base._read_jsonl(attempts_path)
    _validate_partial_state(schedule, rows, telemetry, attempts)

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
                if _mathematically_ineligible(schedule, rows, attempts):
                    early_stop = True
                    break
                if slot.record_id in accepted_ids or attempt_counts[slot.record_id] >= 2:
                    continue
                while attempt_counts[slot.record_id] < MAX_ATTEMPTS_PER_RECORD:
                    if len(attempts) >= MAX_CALLS:
                        early_stop = True
                        break
                    retries = sum(1 for item in attempts if item.get("attempt_number") == 2)
                    if attempt_counts[slot.record_id] >= 1 and retries >= MAX_RETRIES:
                        break
                    if time.time() > deadline:
                        raise RuntimeError("systemic-pilot runtime boundary reached")
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
                if base._private_write_bytes(artifact) > MAX_PRIVATE_WRITES:
                    raise RuntimeError("systemic-pilot private-write ceiling reached")
    finally:
        base._stop_runtime(ssh_key.resolve(), known_hosts.resolve())
        stopped = True
    after_gpu = gpt.gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())

    terminals = _terminal_map(schedule, rows, telemetry, attempts)
    complete = len(terminals) == TOTAL_RECORDS
    audit = _final_audit(
        schedule, rows, terminals, quality=quality, benchmark=benchmark,
        tokenizer=tokenizer, template=template,
    )
    success = complete and all(audit["checks"].values())
    owner_sample = _owner_sample(rows) if success else []
    base._write_jsonl_atomic(private / "candidate-rows.jsonl", rows)
    base._write_jsonl_atomic(private / "generation-telemetry.jsonl", telemetry)
    base._write_jsonl_atomic(
        private / "rejection-metadata.jsonl",
        [item for item in attempts if item.get("accepted") is not True],
    )
    if success:
        base._write_jsonl_atomic(private / "owner-review-sample-20.jsonl", owner_sample)
    failure = _aggregate_failure(schedule, rows, attempts, audit)
    base._write_json_atomic(evidence / "aggregate-cell-report.json", failure)
    report: dict[str, Any] = {
        "aggregate_cell_report_file_sha256": file_sha256(evidence / "aggregate-cell-report.json"),
        "authorization_file_sha256": file_sha256(authorization_path),
        "candidate_file_sha256": file_sha256(private / "candidate-rows.jsonl"),
        "dataset_sealed": False,
        "early_mathematical_stop": early_stop,
        "external_api_used": False,
        "gpu": {"after_stop": after_gpu, "before": before_gpu},
        "inference_calls": len(attempts),
        "machine_audit": audit,
        "owner_review": {
            "minimum_acceptance_rate": 0.9,
            "rows": len(owner_sample),
            "status": "pending" if success else "not_run",
        },
        "plan_file_sha256": file_sha256(plan_path),
        "production_eligible": False,
        "prompt_contract": PROMPT_CONTRACT,
        "rejection_metadata_file_sha256": file_sha256(private / "rejection-metadata.jsonl"),
        "runtime_retries": sum(1 for item in attempts if item.get("attempt_number") == 2),
        "runtime_receipt_file_sha256": file_sha256(runtime_receipt_path),
        "runtime_stopped": stopped and after_gpu["memory_used_mib"] == 0,
        "schema_version": RUN_REPORT_SCHEMA,
        "sft_training_started": False,
        "status": (
            "machine_verified_pending_20_row_owner_review"
            if success else "FAIL_CLOSED_aggregate_cell_threshold_report"
        ),
    }
    if success:
        report["owner_review_sample_file_sha256"] = file_sha256(
            private / "owner-review-sample-20.jsonl"
        )
    report["report_sha256"] = scale.sha256_value(report)
    base._write_json_atomic(report_path, report)
    return report


def verify_systemic_pilot(
    *, plan_path: Path, authorization_path: Path, runtime_receipt_path: Path,
    repository_root: Path, artifact_root: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    plan = _load_plan(plan_path.resolve(), root)
    load_execution_authorization(
        authorization_path.resolve(), plan_path=plan_path.resolve(), repository_root=root
    )
    load_runtime_receipt(
        runtime_receipt_path.resolve(),
        authorization_path=authorization_path.resolve(),
    )
    artifact = artifact_root.resolve()
    report_path = artifact / "evidence/run-report.json"
    report = base._load_json(report_path)
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    rows = base._read_jsonl(artifact / "private/candidate-rows.jsonl")
    telemetry = base._read_jsonl(artifact / "private/generation-telemetry.jsonl")
    attempts = base._read_jsonl(artifact / "private/attempt-metadata.jsonl")
    schedule = build_schedule()
    _validate_partial_state(schedule, rows, telemetry, attempts)
    quality = base._load_json(root / plan["bindings"]["data_quality"]["path"])
    benchmark = scale.BenchmarkRegistry(quality["benchmark_decontamination"], root)
    tokenizer_path = (root / plan["bindings"]["tokenizer_json"]["path"]).resolve().parent
    tokenizer = scale._load_direct_tokenizer(tokenizer_path)  # noqa: SLF001
    template = load_chat_template((root / plan["bindings"]["chat_template"]["path"]).resolve())
    terminals = _terminal_map(schedule, rows, telemetry, attempts)
    audit = _final_audit(
        schedule, rows, terminals, quality=quality, benchmark=benchmark,
        tokenizer=tokenizer, template=template,
    )
    success = len(terminals) == TOTAL_RECORDS and all(audit["checks"].values())
    checks = {
        "authorization_binding": report.get("authorization_file_sha256")
        == file_sha256(authorization_path),
        "call_and_retry_ceiling": len(attempts) <= MAX_CALLS
        and sum(1 for item in attempts if item.get("attempt_number") == 2) <= MAX_RETRIES,
        "internal_report_hash": internal == scale.sha256_value(unsigned),
        "machine_audit_reproduced": audit == report.get("machine_audit"),
        "no_production_actions": report.get("dataset_sealed") is False
        and report.get("sft_training_started") is False,
        "rejected_raw_text_absent": all(
            "raw" not in key and "candidate" not in key
            for item in attempts if item.get("accepted") is not True for key in item
        ),
        "runtime_stopped": report.get("runtime_stopped") is True,
    }
    if success:
        sample = _owner_sample(rows)
        checks["owner_packet_exact"] = len(sample) == 20 and (
            file_sha256(artifact / "private/owner-review-sample-20.jsonl")
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
        raise ValueError("systemic-pilot independent verification failed")
    return verification
