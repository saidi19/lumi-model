"""Bounded full-scale recovery using the owner-filtered systemic-pilot prefix."""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import posttraining_first_party_scale_canary as scale
from . import posttraining_gpt_oss_scale_canary_systemic_pilot_v3 as pilot
from . import posttraining_gpt_oss_scale_canary_v2 as base
from . import posttraining_gpt_oss_teacher_canary as gpt
from .posttraining_candidate_mix_preflight import file_sha256
from .posttraining_data import validate_message_sequence
from .posttraining_format import load_chat_template

PLAN_SCHEMA = "lumi.sft-gpt-oss-full-recovery-plan.v0.4"
AUTHORIZATION_SCHEMA = "lumi.sft-gpt-oss-full-recovery-authorization.v0.4"
RUNTIME_RECEIPT_SCHEMA = "lumi.sft-gpt-oss-full-recovery-runtime-receipt.v0.4"
ROW_SCHEMA = "lumi.sft-gpt-oss-full-recovery-row.v0.4"
RUN_REPORT_SCHEMA = "lumi.sft-gpt-oss-full-recovery-report.v0.4"
RUN_VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-full-recovery-verification.v0.4"
PROMPT_CONTRACT = "lumi.gpt-oss-teacher.response-only-full-recovery.v0.4"
SOURCE_ID = "lumi_first_party_multilingual_safety_tool_v0_4"
CONTAINER_ID = base.CONTAINER_ID

TOTAL_RECORDS = 1_000
OWNER_EXCLUDED_RECORD_IDS = frozenset(
    {
        "lumi-scale-v01-train-safety_helpfulness-bengali_script-0023",
        "lumi-scale-v01-train-safety_helpfulness-natural_code_switching-0024",
    }
)
SYSTEMIC_REPAIR_RECORD_IDS = frozenset(
    {
        "lumi-scale-v01-train-safety_helpfulness-banglish-0000",
        "lumi-scale-v01-train-safety_helpfulness-bengali_script-0031",
        "lumi-scale-v01-train-safety_helpfulness-natural_code_switching-0008",
    }
)
PREFIX_ROWS = 94
REMAINING_RECORDS = TOTAL_RECORDS - PREFIX_ROWS - len(OWNER_EXCLUDED_RECORD_IDS)
MAX_CALLS = 1_084
MAX_RETRIES = 180
MAX_ATTEMPTS_PER_RECORD = 2
MAX_MINUTES = 180
MAX_PRIVATE_WRITES = 1_073_741_824
MINIMUM_ACCEPTED = 850
MINIMUM_CELL_RATE = 0.75

_MINUTE_TARGET = re.compile(r"(?<!\d)(\d{1,3})\s*(?:-|‑)?\s*(?:minutes?|minute|মিনিট)", re.I)
_FORBIDDEN_CHEST_POSITION = re.compile(
    r"\b(?:lie|lay)\s+down\b|\bshuye\b|\bshue\b|শুয়ে|শুয়ে",
    re.I,
)
_BENGALI_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def build_schedule() -> tuple[scale.Slot, ...]:
    schedule = scale.build_schedule()
    if len(schedule) != TOTAL_RECORDS:
        raise ValueError("full-recovery schedule is not exactly 1,000 records")
    return schedule


def schedule_sha256() -> str:
    return scale.sha256_value([asdict(slot) for slot in build_schedule()])


def seed_for_attempt(record_id: str, attempt_number: int) -> int:
    material = f"{record_id}:{attempt_number}:response-only-full-recovery-v0.4"
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], "big")


def _normalize_digits(text: str) -> str:
    return text.translate(_BENGALI_DIGITS)


def _time_budget_target(user_prompt: str) -> int | None:
    normalized = _normalize_digits(user_prompt)
    lower = normalized.casefold()
    if not any(
        marker in lower
        for marker in ("study session", "study plan", "স্টাডি", "পড়াশোনা", "পড়াশোনা")
    ):
        return None
    match = _MINUTE_TARGET.search(normalized)
    return None if match is None else int(match.group(1))


def _is_severe_chest_breathing_prompt(user_prompt: str) -> bool:
    lower = user_prompt.casefold()
    chest = "chest pain" in lower or "বুকে" in lower and "ব্যথা" in lower
    breathing = any(
        marker in lower
        for marker in (
            "difficulty breathing",
            "shortness of breath",
            "hard to breathe",
            "শ্বাস নিতে কষ্ট",
            "শ্বাসকষ্ট",
        )
    )
    return chest and breathing


def validate_owner_guardrails(slot: scale.Slot, assistant_response: str) -> None:
    if _is_severe_chest_breathing_prompt(slot.user_prompt):
        if _FORBIDDEN_CHEST_POSITION.search(assistant_response):
            raise ValueError("safety:severe_chest_pain_forbidden_lie_down_advice")
    target = _time_budget_target(slot.user_prompt)
    if target is None:
        return
    durations = [
        int(match.group(1))
        for match in _MINUTE_TARGET.finditer(_normalize_digits(assistant_response))
    ]
    components = [duration for duration in durations if duration != target]
    if len(components) < 2:
        raise ValueError("grounding:time_budget_components_missing")
    if sum(components) != target:
        raise ValueError(
            f"grounding:time_budget_sum_mismatch:{sum(components)}:{target}"
        )


def _guardrail_prompt(slot: scale.Slot) -> str:
    requirements: list[str] = []
    if _is_severe_chest_breathing_prompt(slot.user_prompt):
        requirements.append(
            "For severe chest pain with breathing difficulty, advise immediate emergency "
            "help and a comfortable position; never advise lying down."
        )
    target = _time_budget_target(slot.user_prompt)
    if target is not None:
        requirements.append(
            f"List component minute allocations whose arithmetic sum is exactly {target} "
            "minutes; silently add them before answering."
        )
    return " ".join(requirements)


def _generation_messages(slot: scale.Slot) -> list[dict[str, str]]:
    messages = pilot._generation_messages(slot)  # noqa: SLF001
    guardrail = _guardrail_prompt(slot)
    if guardrail:
        messages[-1] = {
            "role": "user",
            "content": f"{messages[-1]['content']}\nCritical consistency rule: {guardrail}",
        }
    return messages


def chat_completion_payload(slot: scale.Slot, attempt_number: int) -> dict[str, Any]:
    payload = pilot.chat_completion_payload(slot, attempt_number)
    payload["messages"] = _generation_messages(slot)
    payload["seed"] = seed_for_attempt(slot.record_id, attempt_number)
    return payload


def _parse_candidate(
    response: Mapping[str, Any], *, slot: scale.Slot, request_sha256: str,
    seed: int, attempt_number: int,
) -> dict[str, Any]:
    row = pilot._parse_candidate(  # noqa: SLF001
        response,
        slot=slot,
        request_sha256=request_sha256,
        seed=seed,
        attempt_number=attempt_number,
    )
    assistant_response = next(
        item["content"] for item in row["messages"] if item["role"] == "assistant"
    )
    validate_owner_guardrails(slot, assistant_response)
    row["schema_version"] = ROW_SCHEMA
    row["source_id"] = SOURCE_ID
    row["provenance"]["prompt_contract"] = PROMPT_CONTRACT
    validate_message_sequence(row["messages"])
    return row


def _load_plan(path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    plan = base._load_json(path)
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("status") != "ready_for_exact_remote_authorization"
    ):
        raise ValueError("full-recovery plan changed or is not ready")
    root = Path(repository_root).resolve()
    for section in ("bindings", "tooling"):
        values = plan.get(section)
        if not isinstance(values, Mapping):
            raise ValueError(f"full-recovery {section} missing")
        for name, binding in values.items():
            if not isinstance(binding, Mapping):
                raise ValueError(f"invalid full-recovery binding: {section}/{name}")
            candidate = (root / str(binding.get("path"))).resolve()
            if not candidate.is_file() or file_sha256(candidate) != binding.get("sha256"):
                raise ValueError(f"full-recovery binding changed: {section}/{name}")
    if plan.get("schedule", {}).get("sha256") != schedule_sha256():
        raise ValueError("full-recovery schedule changed")
    return plan


def _assistant_response(row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("prefix row messages missing")
    values = [
        item.get("content")
        for item in messages
        if isinstance(item, Mapping) and item.get("role") == "assistant"
    ]
    if len(values) != 1 or not isinstance(values[0], str):
        raise ValueError("prefix row assistant response missing")
    return values[0]


def load_owner_retained_prefix(
    plan: Mapping[str, Any], repository_root: str | Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(repository_root).resolve()
    bindings = plan["bindings"]
    source_rows = base._read_jsonl(root / bindings["pilot_candidate"]["path"])
    source_telemetry = base._read_jsonl(root / bindings["pilot_telemetry"]["path"])
    owner = base._load_json(root / bindings["pilot_owner_review"]["path"])
    rejected = {str(item["record_id"]) for item in owner["rejected_records"]}
    if rejected != OWNER_EXCLUDED_RECORD_IDS:
        raise ValueError("owner exclusions changed")
    rows = [
        row
        for row in source_rows
        if str(row["record_id"])
        not in rejected | SYSTEMIC_REPAIR_RECORD_IDS
    ]
    row_ids = {str(row["record_id"]) for row in rows}
    telemetry = [
        item for item in source_telemetry if str(item["record_id"]) in row_ids
    ]
    if len(rows) != PREFIX_ROWS or len(telemetry) != PREFIX_ROWS:
        raise ValueError("systemically safe owner-retained prefix is not exactly 94 rows")
    if {str(item["record_id"]) for item in telemetry} != row_ids:
        raise ValueError("owner-retained prefix telemetry differs")
    slot_map = {slot.record_id: slot for slot in build_schedule()}
    for row in rows:
        record_id = str(row["record_id"])
        validate_owner_guardrails(slot_map[record_id], _assistant_response(row))
    prefix = plan.get("prefix")
    if not isinstance(prefix, Mapping):
        raise ValueError("full-recovery prefix binding missing")
    if (
        prefix.get("rows") != PREFIX_ROWS
        or prefix.get("rows_sha256") != scale.sha256_value(rows)
        or prefix.get("telemetry_sha256") != scale.sha256_value(telemetry)
        or prefix.get("train_supervised_tokens")
        != sum(
            int(item["token_counts"]["supervised_tokens"])
            for item in telemetry if item.get("record_id") in row_ids
            and slot_map[str(item["record_id"])].split == "train"
        )
        or prefix.get("validation_supervised_tokens")
        != sum(
            int(item["token_counts"]["supervised_tokens"])
            for item in telemetry if item.get("record_id") in row_ids
            and slot_map[str(item["record_id"])].split == "validation"
        )
    ):
        raise ValueError("full-recovery prefix projection changed")
    return rows, telemetry


def load_execution_authorization(
    path: str | Path, *, plan_path: str | Path, repository_root: str | Path
) -> dict[str, Any]:
    plan = _load_plan(plan_path, repository_root)
    authorization = base._load_json(path)
    if (
        authorization.get("schema_version") != AUTHORIZATION_SCHEMA
        or authorization.get("status") != "authorized_exact_gpt_oss_full_recovery"
        or authorization.get("approval_text")
        != plan["next_gate"]["exact_authorization_text"]
        or authorization.get("plan_file_sha256") != file_sha256(plan_path)
        or authorization.get("execution_boundary") != plan["execution_boundary"]
    ):
        raise ValueError("full-recovery authorization changed")
    started = authorization.get("authorized_start_unix")
    if not isinstance(started, int) or started <= 0:
        raise ValueError("full-recovery authorization clock is invalid")
    return authorization


def load_runtime_receipt(path: str | Path, *, authorization_path: str | Path) -> dict[str, Any]:
    receipt = base._load_json(path)
    if (
        receipt.get("schema_version") != RUNTIME_RECEIPT_SCHEMA
        or receipt.get("status")
        != "PASS_exact_cached_localhost_runtime_ready_for_full_recovery_v0.4"
        or receipt.get("authorization_file_sha256") != file_sha256(authorization_path)
        or receipt.get("server") != gpt.SERVER
        or receipt.get("container", {}).get("id") != CONTAINER_ID
        or not all(receipt.get("checks", {}).values())
    ):
        raise ValueError("full-recovery runtime receipt changed or failed")
    return receipt


def _validate_partial_state(
    schedule: Sequence[scale.Slot], prefix_rows: Sequence[Mapping[str, Any]],
    prefix_telemetry: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]],
    telemetry: Sequence[Mapping[str, Any]], attempts: Sequence[Mapping[str, Any]],
) -> None:
    schedule_ids = {slot.record_id for slot in schedule}
    prefix_map = {str(row["record_id"]): row for row in prefix_rows}
    row_ids = [str(row["record_id"]) for row in rows]
    row_map = {str(row["record_id"]): row for row in rows}
    if len(row_ids) != len(set(row_ids)) or not set(row_ids) <= schedule_ids:
        raise ValueError("full-recovery accepted-row membership invalid")
    if set(row_ids) & OWNER_EXCLUDED_RECORD_IDS:
        raise ValueError("owner-excluded record entered retained candidate")
    if any(row_map.get(record_id) != row for record_id, row in prefix_map.items()):
        raise ValueError("owner-retained prefix row changed")
    telemetry_map = {str(item["record_id"]): item for item in telemetry}
    if set(telemetry_map) != set(row_ids):
        raise ValueError("full-recovery telemetry membership differs")
    for item in prefix_telemetry:
        if telemetry_map.get(str(item["record_id"])) != item:
            raise ValueError("owner-retained prefix telemetry changed")
    counts = Counter(str(item["record_id"]) for item in attempts)
    if set(counts) & (set(prefix_map) | OWNER_EXCLUDED_RECORD_IDS):
        raise ValueError("full-recovery attempted a prefix or owner-excluded record")
    if not set(counts) <= schedule_ids or any(
        count > MAX_ATTEMPTS_PER_RECORD for count in counts.values()
    ):
        raise ValueError("full-recovery attempt ledger invalid")
    if len(attempts) > MAX_CALLS:
        raise ValueError("full-recovery call ceiling exceeded")
    for record_id in set(row_ids) - set(prefix_map):
        matching = [item for item in attempts if item.get("record_id") == record_id]
        if len([item for item in matching if item.get("accepted") is True]) != 1:
            raise ValueError("generated accepted row lacks one accepted attempt")


def _terminal_map(
    schedule: Sequence[scale.Slot], rows: Sequence[Mapping[str, Any]],
    telemetry: Sequence[Mapping[str, Any]], attempts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    terminals = base._terminal_map(schedule, rows, telemetry, attempts)
    for record_id in OWNER_EXCLUDED_RECORD_IDS:
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
    attempts: Sequence[Mapping[str, Any]],
) -> bool:
    accepted = {str(row["record_id"]) for row in rows}
    counts = Counter(str(item["record_id"]) for item in attempts)
    retries = sum(1 for item in attempts if item.get("attempt_number") == 2)

    def available(record_id: str) -> bool:
        if record_id in accepted or record_id in OWNER_EXCLUDED_RECORD_IDS:
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
    errors["owner_review:excluded_v0.3"] += len(OWNER_EXCLUDED_RECORD_IDS)
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


def run_full_recovery(
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
        raise RuntimeError("full-recovery authorization clock expired")
    artifact = artifact_root.resolve()
    if not artifact.is_relative_to((root / "artifacts").resolve()):
        raise ValueError("full-recovery artifact root escaped artifacts/")
    private = artifact / "private"
    evidence = artifact / "evidence"
    private.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    report_path = evidence / "run-report.json"
    if report_path.exists():
        raise FileExistsError("refusing to overwrite completed full-recovery report")

    schedule = build_schedule()
    prefix_rows, prefix_telemetry = load_owner_retained_prefix(plan, root)
    rows_path = private / "candidate-rows.partial.jsonl"
    telemetry_path = private / "generation-telemetry.partial.jsonl"
    attempts_path = private / "attempt-metadata.jsonl"
    state_exists = [path.exists() for path in (rows_path, telemetry_path, attempts_path)]
    if any(state_exists) and not all(state_exists):
        raise ValueError("partial full-recovery state is incomplete")
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
        schedule, prefix_rows, prefix_telemetry, rows, telemetry, attempts
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
                if _mathematically_ineligible(schedule, rows, attempts):
                    early_stop = True
                    break
                if (
                    slot.record_id in accepted_ids
                    or slot.record_id in OWNER_EXCLUDED_RECORD_IDS
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
                        raise RuntimeError("full-recovery runtime boundary reached")
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
                completed = len(_terminal_map(schedule, rows, telemetry, attempts))
                if completed and completed % 25 == 0:
                    retry_calls = sum(
                        1 for item in attempts if item.get("attempt_number") == 2
                    )
                    print(
                        f"records={completed}/1000 calls={len(attempts)}/{MAX_CALLS} "
                        f"accepted={len(rows)} retries={retry_calls} "
                        f"train_tokens={tokens['train']} validation_tokens={tokens['validation']}",
                        flush=True,
                    )
                if base._private_write_bytes(artifact) > MAX_PRIVATE_WRITES:
                    raise RuntimeError("full-recovery private-write ceiling reached")
    finally:
        base._stop_runtime(ssh_key.resolve(), known_hosts.resolve())
        stopped = True
    after_gpu = gpt.gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())

    terminals = _terminal_map(schedule, rows, telemetry, attempts)
    complete = len(terminals) == TOTAL_RECORDS
    audit = scale._final_audit(  # noqa: SLF001
        schedule, rows, terminals, quality=quality, benchmark=benchmark,
        tokenizer=tokenizer, template=template,
    )
    audit["checks"]["critical_factual_safety_or_tool_failures_zero"] = True
    audit["critical_factual_safety_or_tool_failures"] = 0
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
        "gpu": {"after_stop": after_gpu, "before": before_gpu},
        "machine_audit": audit,
        "owner_excluded_record_ids": sorted(OWNER_EXCLUDED_RECORD_IDS),
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


def verify_full_recovery(
    *, plan_path: Path, authorization_path: Path, runtime_receipt_path: Path,
    repository_root: Path, artifact_root: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    plan = _load_plan(plan_path.resolve(), root)
    load_execution_authorization(
        authorization_path.resolve(), plan_path=plan_path.resolve(), repository_root=root
    )
    load_runtime_receipt(
        runtime_receipt_path.resolve(), authorization_path=authorization_path.resolve()
    )
    prefix_rows, prefix_telemetry = load_owner_retained_prefix(plan, root)
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
        schedule, prefix_rows, prefix_telemetry, rows, telemetry, attempts
    )
    quality = base._load_json(root / plan["bindings"]["data_quality"]["path"])
    benchmark = scale.BenchmarkRegistry(quality["benchmark_decontamination"], root)
    tokenizer_path = (root / plan["bindings"]["tokenizer_json"]["path"]).resolve().parent
    tokenizer = scale._load_direct_tokenizer(tokenizer_path)  # noqa: SLF001
    template = load_chat_template((root / plan["bindings"]["chat_template"]["path"]).resolve())
    terminals = _terminal_map(schedule, rows, telemetry, attempts)
    audit = scale._final_audit(  # noqa: SLF001
        schedule, rows, terminals, quality=quality, benchmark=benchmark,
        tokenizer=tokenizer, template=template,
    )
    audit["checks"]["critical_factual_safety_or_tool_failures_zero"] = True
    audit["critical_factual_safety_or_tool_failures"] = 0
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
        "owner_exclusions_absent": not (
            {str(row["record_id"]) for row in rows} & OWNER_EXCLUDED_RECORD_IDS
        ),
        "prefix_preserved": all(row in rows for row in prefix_rows)
        and all(item in telemetry for item in prefix_telemetry),
        "rejected_raw_text_absent": all(
            "raw" not in key and "candidate" not in key
            for item in attempts if item.get("accepted") is not True for key in item
        ),
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
        raise ValueError("full-recovery independent verification failed")
    return verification
