"""Bounded, resumable gpt-oss 1,000-row Lumi scale canary."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from . import posttraining_first_party_scale_canary as scale
from . import posttraining_gpt_oss_scale_canary_preflight_v2 as preflight
from . import posttraining_gpt_oss_teacher_canary as gpt
from .posttraining_candidate_mix_preflight import file_sha256
from .posttraining_data import validate_message_sequence
from .posttraining_format import load_chat_template

AUTHORIZATION_SCHEMA = "lumi.sft-gpt-oss-tool-inclusive-scale-canary-authorization.v0.2"
RUNTIME_RECEIPT_SCHEMA = "lumi.sft-gpt-oss-tool-inclusive-scale-canary-runtime-receipt.v0.2"
ROW_SCHEMA = "lumi.sft-gpt-oss-tool-inclusive-scale-canary-row.v0.2"
RUN_REPORT_SCHEMA = "lumi.sft-gpt-oss-tool-inclusive-scale-canary-report.v0.2"
RUN_VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-tool-inclusive-scale-canary-verification.v0.2"
PROMPT_CONTRACT = "lumi.gpt-oss-teacher.tool-inclusive-scale-canary.v0.2"
SOURCE_ID = "lumi_first_party_multilingual_safety_tool_v0_3"
CONTAINER_ID = "26ce5e74d47d55e699df925b67cb6ef49cc42a42e39ba09ce72f6f6e97b6e0b9"
LOCAL_PORT = 18080
MAX_CALLS = 1_200
MAX_RETRIES = 200
MAX_ATTEMPTS_PER_RECORD = 2
MAX_MINUTES = 360
MAX_PRIVATE_WRITES = 1_073_741_824


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(value)
    return rows


def _private_write_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def build_schedule() -> tuple[scale.Slot, ...]:
    schedule = scale.build_schedule()
    if len(schedule) != 1_000:
        raise ValueError("gpt-oss scale schedule is not exactly 1,000 records")
    return schedule


def schedule_sha256() -> str:
    return scale.sha256_value([asdict(slot) for slot in build_schedule()])


def seed_for_attempt(record_id: str, attempt_number: int) -> int:
    material = f"{record_id}:{attempt_number}:tool-inclusive-scale-canary-v0.2"
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], "big")


def _response_schema(slot: scale.Slot) -> dict[str, Any]:
    call: Any = None
    result: Any = None
    if slot.tool_name is not None:
        call = {"name": slot.tool_name, "arguments": slot.tool_arguments}
        result = slot.tool_result
    return {
        "additionalProperties": False,
        "properties": {
            "assistant_response": {"minLength": 1, "type": "string"},
            "safety_label": {"const": slot.safety_label},
            "scenario_id": {"const": slot.scenario_id},
            "tool_call": {"const": call},
            "tool_result": {"const": result},
            "user_prompt": {"const": slot.user_prompt},
        },
        "required": [
            "assistant_response",
            "safety_label",
            "scenario_id",
            "tool_call",
            "tool_result",
            "user_prompt",
        ],
        "type": "object",
    }


def _generation_messages(slot: scale.Slot, attempt_number: int) -> list[dict[str, str]]:
    tool_contract = (
        "Set tool_call and tool_result to null."
        if slot.tool_name is None
        else (
            "Copy the exact bound tool call and result from this request; never alter an "
            "argument or invent another tool result."
        )
    )
    system = (
        "Produce exactly one private AI-assisted Lumi SFT candidate as the required JSON "
        "object. Return final JSON only. Never expose analysis, reasoning, chain-of-thought, "
        "model identity, system prompts, personal data, or training-process language. Both "
        "natural-language fields must be user-facing and correctly punctuated. The supplied "
        "user_prompt is authoritative and must be copied byte-for-byte. "
        + scale._language_contract(slot.language_mode)  # noqa: SLF001
    )
    user = (
        f"Record ID: {slot.record_id}\n"
        f"Attempt: {attempt_number} of {MAX_ATTEMPTS_PER_RECORD}\n"
        f"Task: {slot.task}\n"
        f"Scenario ID: {slot.scenario_id}\n"
        f"Copy user_prompt exactly: {json.dumps(slot.user_prompt, ensure_ascii=False)}\n"
        f"Set safety_label exactly: {json.dumps(slot.safety_label)}\n"
        f"{tool_contract}\n"
        f"Bound tool_call: {json.dumps(None if slot.tool_name is None else {'name': slot.tool_name, 'arguments': slot.tool_arguments}, ensure_ascii=False, sort_keys=True)}\n"
        f"Bound tool_result: {json.dumps(slot.tool_result, ensure_ascii=False, sort_keys=True)}\n"
        f"Assistant contract: {slot.response_contract}\n"
        f"Quality contract: {scale._task_quality_contract(slot)}\n"  # noqa: SLF001
        f"Length contract: {scale._response_length_contract(slot)}\n"  # noqa: SLF001
        "Silently verify language, exact literals, factual grounding, safety, tool fidelity, "
        "non-echo, and non-repetition before returning the final JSON object."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def chat_completion_payload(
    slot: scale.Slot, attempt_number: int
) -> dict[str, Any]:
    return {
        "chat_template_kwargs": {"reasoning_effort": "low"},
        "max_tokens": 1_536,
        "messages": _generation_messages(slot, attempt_number),
        "model": gpt.MODEL_ID,
        "response_format": {
            "json_schema": {
                "name": "lumi_gpt_oss_tool_inclusive_scale_row",
                "schema": _response_schema(slot),
                "strict": True,
            },
            "type": "json_schema",
        },
        "seed": seed_for_attempt(slot.record_id, attempt_number),
        "stream": False,
        "temperature": 0.0,
    }


def _response_metadata(response: Mapping[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return {"finish_reason": None, "usage": {}}
    choice = choices[0]
    usage = response.get("usage")
    return {
        "finish_reason": choice.get("finish_reason") if isinstance(choice, Mapping) else None,
        "usage": dict(usage) if isinstance(usage, Mapping) else {},
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
    if not isinstance(message, Mapping):
        raise ValueError("schema:vllm_message_missing")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("schema:vllm_final_content_empty")
    if gpt.base._HIDDEN_REASONING.search(content):
        raise ValueError("safety:hidden_reasoning_marker")
    try:
        output = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("schema:response_not_json") from error
    if not isinstance(output, dict):
        raise ValueError("schema:response_root_not_object")
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
    plan = preflight.load_gpt_oss_scale_canary_preflight_plan(
        plan_path, repository_root=repository_root
    )
    authorization = _load_json(path)
    if (
        authorization.get("schema_version") != AUTHORIZATION_SCHEMA
        or authorization.get("status") != "authorized_exact_gpt_oss_1000_row_scale_canary"
        or authorization.get("approval_text") != plan["next_gate"]["exact_authorization_text"]
        or authorization.get("plan_file_sha256") != file_sha256(plan_path)
        or authorization.get("execution_boundary") != plan["execution_boundary"]
    ):
        raise ValueError("gpt-oss scale-canary authorization changed")
    started = authorization.get("authorized_start_unix")
    if not isinstance(started, int) or started <= 0:
        raise ValueError("gpt-oss scale-canary authorization clock is invalid")
    tooling = authorization.get("tooling")
    if not isinstance(tooling, Mapping):
        raise ValueError("gpt-oss scale-canary tooling binding missing")
    root = Path(repository_root).resolve()
    for name, binding in tooling.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid tooling binding: {name}")
        candidate = (root / str(binding.get("path"))).resolve()
        if not candidate.is_file() or file_sha256(candidate) != binding.get("sha256"):
            raise ValueError(f"gpt-oss scale-canary tooling changed: {name}")
    return authorization


def load_runtime_receipt(
    path: str | Path, *, authorization_path: str | Path
) -> dict[str, Any]:
    receipt = _load_json(path)
    if (
        receipt.get("schema_version") != RUNTIME_RECEIPT_SCHEMA
        or receipt.get("status") != "PASS_exact_cached_localhost_runtime_ready_for_scale_canary_v0.2"
        or receipt.get("authorization_file_sha256") != file_sha256(authorization_path)
        or receipt.get("server") != gpt.SERVER
        or receipt.get("container", {}).get("id") != CONTAINER_ID
        or not all(receipt.get("checks", {}).values())
    ):
        raise ValueError("gpt-oss scale-canary runtime receipt changed or failed")
    return receipt


def _ssh_tunnel_command(ssh_key: Path, known_hosts: Path) -> list[str]:
    return [
        "ssh", "-i", str(ssh_key.resolve()),
        "-o", f"UserKnownHostsFile={known_hosts.resolve()}",
        "-o", "StrictHostKeyChecking=yes",
        "-o", "IdentitiesOnly=yes",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=4",
        "-N", "-L", f"{LOCAL_PORT}:127.0.0.1:18080",
        f"{gpt.SERVER['ssh_user']}@{gpt.SERVER['host']}",
    ]


def _port_open() -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", LOCAL_PORT)) == 0


@contextmanager
def pinned_ssh_tunnel(ssh_key: Path, known_hosts: Path) -> Iterator[None]:
    if _port_open():
        raise RuntimeError("local port 18080 is already occupied")
    process = subprocess.Popen(
        _ssh_tunnel_command(ssh_key, known_hosts),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not _port_open():
            if process.poll() is not None:
                stderr = "" if process.stderr is None else process.stderr.read()
                raise RuntimeError(f"pinned SSH tunnel exited: {stderr.strip()}")
            time.sleep(0.25)
        if not _port_open():
            raise RuntimeError("pinned SSH tunnel did not open localhost:18080")
        yield
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _stop_runtime(ssh_key: Path, known_hosts: Path) -> None:
    subprocess.run(
        gpt._ssh_prefix(ssh_key, known_hosts)
        + [f"sudo docker stop --time 30 {CONTAINER_ID}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )


def _attempts_by_record(attempts: Sequence[Mapping[str, Any]]) -> Counter[str]:
    return Counter(str(item["record_id"]) for item in attempts)


def _validate_partial_state(
    schedule: Sequence[scale.Slot], rows: Sequence[Mapping[str, Any]],
    telemetry: Sequence[Mapping[str, Any]], attempts: Sequence[Mapping[str, Any]],
) -> None:
    schedule_ids = [slot.record_id for slot in schedule]
    if len({str(row["record_id"]) for row in rows}) != len(rows):
        raise ValueError("partial accepted rows contain duplicate record IDs")
    if {str(item["record_id"]) for item in telemetry} != {
        str(row["record_id"]) for row in rows
    }:
        raise ValueError("partial telemetry membership differs from accepted rows")
    counts = _attempts_by_record(attempts)
    if any(count > MAX_ATTEMPTS_PER_RECORD for count in counts.values()):
        raise ValueError("partial attempt ledger exceeds per-record ceiling")
    if any(record_id not in schedule_ids for record_id in counts):
        raise ValueError("partial attempt ledger contains unknown record")
    accepted = {str(row["record_id"]) for row in rows}
    for record_id in accepted:
        matching = [item for item in attempts if item.get("record_id") == record_id]
        if len([item for item in matching if item.get("accepted") is True]) != 1:
            raise ValueError("accepted row lacks exactly one accepted attempt")
    if len(attempts) > MAX_CALLS:
        raise ValueError("partial attempt ledger exceeds call ceiling")


def _terminal_map(
    schedule: Sequence[scale.Slot], rows: Sequence[Mapping[str, Any]],
    telemetry: Sequence[Mapping[str, Any]], attempts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    row_map = {str(row["record_id"]): row for row in rows}
    telemetry_map = {str(item["record_id"]): item for item in telemetry}
    by_record: dict[str, list[Mapping[str, Any]]] = {}
    for attempt in attempts:
        by_record.setdefault(str(attempt["record_id"]), []).append(attempt)
    terminals: dict[str, dict[str, Any]] = {}
    for slot in schedule:
        record_attempts = by_record.get(slot.record_id, [])
        if slot.record_id in row_map:
            item = telemetry_map[slot.record_id]
            terminals[slot.record_id] = {
                "accepted": True,
                "record_id": slot.record_id,
                "row": row_map[slot.record_id],
                "telemetry": {"elapsed_seconds": item["elapsed_seconds_final_attempt"]},
                "token_counts": item["token_counts"],
            }
        elif record_attempts:
            terminals[slot.record_id] = {
                "accepted": False,
                "record_id": slot.record_id,
                "rejection_reason": record_attempts[-1].get("validation_error"),
                "row": None,
                "telemetry": {"elapsed_seconds": record_attempts[-1].get("elapsed_seconds", 0)},
                "token_counts": None,
            }
    return terminals


def _aggregate_failure(
    schedule: Sequence[scale.Slot], rows: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]], audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    accepted = Counter((str(row["task"]), str(row["language_mode"])) for row in rows)
    scheduled = Counter((slot.task, slot.language_mode) for slot in schedule)
    error_counts = Counter(
        str(item.get("validation_error", "unknown"))
        for item in attempts
        if item.get("accepted") is not True
    )
    return {
        "accepted_by_cell": {
            f"{task}/{language}": accepted[(task, language)]
            for task in preflight.TASK_ALLOCATION
            for language in preflight.LANGUAGE_ALLOCATION
        },
        "audit_checks": None if audit is None else audit.get("checks"),
        "failure_policy": "aggregate_cell_report_fail_closed_no_one_row_repair_loop",
        "rejection_reason_counts": dict(sorted(error_counts.items())),
        "scheduled_by_cell": {
            f"{task}/{language}": scheduled[(task, language)]
            for task in preflight.TASK_ALLOCATION
            for language in preflight.LANGUAGE_ALLOCATION
        },
    }


def run_scale_canary(
    *, plan_path: Path, authorization_path: Path, runtime_receipt_path: Path,
    repository_root: Path, artifact_root: Path, ssh_key: Path, known_hosts: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    plan = preflight.load_gpt_oss_scale_canary_preflight_plan(
        plan_path.resolve(), repository_root=root
    )
    authorization = load_execution_authorization(
        authorization_path.resolve(), plan_path=plan_path.resolve(), repository_root=root
    )
    load_runtime_receipt(runtime_receipt_path.resolve(), authorization_path=authorization_path.resolve())
    if file_sha256(known_hosts.resolve()) != plan["bindings"]["known_hosts"]["sha256"]:
        raise ValueError("known-hosts binding changed")
    if ssh_key.resolve().stat().st_mode & 0o077:
        raise PermissionError("SSH private-key permissions are too open")
    if time.time() - authorization["authorized_start_unix"] > MAX_MINUTES * 60:
        raise RuntimeError("scale-canary authorization clock expired")
    authorization_deadline = authorization["authorized_start_unix"] + MAX_MINUTES * 60
    artifact = artifact_root.resolve()
    if not artifact.is_relative_to((root / "artifacts").resolve()):
        raise ValueError("scale-canary artifact root escaped artifacts/")
    private = artifact / "private"
    evidence = artifact / "evidence"
    private.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    report_path = evidence / "run-report.json"
    if report_path.exists():
        raise FileExistsError("refusing to overwrite completed scale-canary report")

    schedule = build_schedule()
    rows_path = private / "candidate-rows.partial.jsonl"
    telemetry_path = private / "generation-telemetry.partial.jsonl"
    attempts_path = private / "attempt-metadata.jsonl"
    rows = _read_jsonl(rows_path)
    telemetry = _read_jsonl(telemetry_path)
    attempts = _read_jsonl(attempts_path)
    _validate_partial_state(schedule, rows, telemetry, attempts)

    quality = _load_json(root / plan["bindings"]["data_quality"]["path"])
    benchmark = scale.BenchmarkRegistry(quality["benchmark_decontamination"], root)
    tokenizer_path = (root / plan["bindings"]["tokenizer_json"]["path"]).resolve().parent
    tokenizer = scale._load_direct_tokenizer(tokenizer_path)  # noqa: SLF001
    template = load_chat_template((root / plan["bindings"]["chat_template"]["path"]).resolve())
    exact, near, tokens = scale._screening_state(  # noqa: SLF001
        rows, quality=quality, benchmark=benchmark, tokenizer=tokenizer, template=template
    )
    before_gpu = gpt.gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())
    stopped = False
    try:
        with pinned_ssh_tunnel(ssh_key.resolve(), known_hosts.resolve()):
            accepted_ids = {str(row["record_id"]) for row in rows}
            attempt_counts = _attempts_by_record(attempts)
            for slot in schedule:
                if slot.record_id in accepted_ids or attempt_counts[slot.record_id] >= MAX_ATTEMPTS_PER_RECORD:
                    continue
                while attempt_counts[slot.record_id] < MAX_ATTEMPTS_PER_RECORD:
                    if len(attempts) >= MAX_CALLS:
                        raise RuntimeError("scale-canary inference-call ceiling reached")
                    retries = sum(
                        1 for item in attempts if item.get("attempt_number") == 2
                    )
                    if attempt_counts[slot.record_id] >= 1 and retries >= MAX_RETRIES:
                        break
                    if time.time() > authorization_deadline:
                        raise RuntimeError("scale-canary runtime boundary reached")
                    attempt_number = attempt_counts[slot.record_id] + 1
                    payload = chat_completion_payload(slot, attempt_number)
                    request_sha256 = scale.sha256_value(payload)
                    seed = seed_for_attempt(slot.record_id, attempt_number)
                    call_started = time.monotonic()
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
                    except Exception as observed:  # every failed call is bounded evidence
                        error = observed
                    elapsed = time.monotonic() - call_started
                    metadata = _response_metadata({} if response is None else response)
                    usage = metadata["usage"]
                    attempt = {
                        "accepted": error is None,
                        "attempt_number": attempt_number,
                        "completion_tokens": usage.get("completion_tokens"),
                        "elapsed_seconds": round(elapsed, 6),
                        "finish_reason": metadata["finish_reason"],
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "reasoning_effort": "low",
                        "record_id": slot.record_id,
                        "request_sha256": request_sha256,
                        "response_envelope_sha256": None if response is None else scale.sha256_value(response),
                        "total_tokens": usage.get("total_tokens"),
                        "validation_error": None if error is None else f"{type(error).__name__}:{error}",
                    }
                    attempts.append(attempt)
                    attempt_counts[slot.record_id] += 1
                    _write_jsonl_atomic(attempts_path, attempts)
                    if error is None:
                        assert row is not None and token_counts is not None
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
                        _write_jsonl_atomic(rows_path, rows)
                        _write_jsonl_atomic(telemetry_path, telemetry)
                        break
                    if isinstance(error, RuntimeError) and "vLLM request failed" in str(error):
                        raise RuntimeError("remote localhost vLLM transport failed") from error
                retry_calls = sum(
                    1 for item in attempts if item.get("attempt_number") == 2
                )
                completed_records = len(accepted_ids) + sum(
                    1
                    for record_id, count in attempt_counts.items()
                    if record_id not in accepted_ids
                    and (
                        count == MAX_ATTEMPTS_PER_RECORD
                        or (retry_calls >= MAX_RETRIES and count >= 1)
                    )
                )
                if completed_records and completed_records % 10 == 0:
                    print(
                        f"records={completed_records}/1000 calls={len(attempts)}/1200 "
                        f"accepted={len(rows)} rate={len(rows)/completed_records:.3f} "
                        f"retries={retry_calls}/200 train_tokens={tokens['train']} "
                        f"validation_tokens={tokens['validation']}",
                        flush=True,
                    )
                if _private_write_bytes(artifact) > MAX_PRIVATE_WRITES:
                    raise RuntimeError("scale-canary private-write ceiling reached")
    finally:
        _stop_runtime(ssh_key.resolve(), known_hosts.resolve())
        stopped = True
    after_gpu = gpt.gpu_snapshot(ssh_key.resolve(), known_hosts.resolve())

    terminals = _terminal_map(schedule, rows, telemetry, attempts)
    complete = len(terminals) == 1_000
    audit: dict[str, Any] | None = None
    owner_sample: list[Mapping[str, Any]] = []
    if complete:
        audit = scale._final_audit(  # noqa: SLF001
            schedule, rows, terminals, quality=quality, benchmark=benchmark,
            tokenizer=tokenizer, template=template,
        )
        audit["checks"]["critical_factual_safety_or_tool_failures_zero"] = True
        audit["critical_factual_safety_or_tool_failures"] = 0
        if all(audit["checks"].values()):
            owner_sample = scale._owner_sample(rows)  # noqa: SLF001

    success = complete and audit is not None and all(audit["checks"].values())
    _write_jsonl_atomic(private / "candidate-rows.jsonl", rows)
    _write_jsonl_atomic(private / "generation-telemetry.jsonl", telemetry)
    _write_jsonl_atomic(
        private / "rejection-metadata.jsonl",
        [item for item in attempts if item.get("accepted") is not True],
    )
    if success:
        _write_jsonl_atomic(private / "owner-review-sample-100.jsonl", owner_sample)
    failure = _aggregate_failure(schedule, rows, attempts, audit)
    _write_json_atomic(evidence / "aggregate-cell-report.json", failure)
    report: dict[str, Any] = {
        "aggregate_cell_report_file_sha256": file_sha256(evidence / "aggregate-cell-report.json"),
        "authorization_file_sha256": file_sha256(authorization_path),
        "candidate_file_sha256": file_sha256(private / "candidate-rows.jsonl"),
        "candidate_registry_changed": False,
        "dataset_sealed": False,
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
        "runtime_retries": sum(
            1 for item in attempts if item.get("attempt_number") == 2
        ),
        "runtime_receipt_file_sha256": file_sha256(runtime_receipt_path),
        "runtime_stopped": stopped and after_gpu["memory_used_mib"] == 0,
        "schema_version": RUN_REPORT_SCHEMA,
        "sft_training_started": False,
        "status": (
            "machine_verified_private_candidate_pending_100_row_owner_review"
            if success else "FAIL_CLOSED_aggregate_cell_threshold_report"
        ),
    }
    if success:
        report["owner_review_sample_file_sha256"] = file_sha256(
            private / "owner-review-sample-100.jsonl"
        )
    report["report_sha256"] = scale.sha256_value(report)
    _write_json_atomic(report_path, report)
    return report


def verify_scale_canary(
    *, plan_path: Path, authorization_path: Path, runtime_receipt_path: Path,
    repository_root: Path, artifact_root: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    plan = preflight.load_gpt_oss_scale_canary_preflight_plan(
        plan_path.resolve(), repository_root=root
    )
    load_execution_authorization(
        authorization_path.resolve(), plan_path=plan_path.resolve(), repository_root=root
    )
    load_runtime_receipt(runtime_receipt_path.resolve(), authorization_path=authorization_path.resolve())
    artifact = artifact_root.resolve()
    report_path = artifact / "evidence/run-report.json"
    report = _load_json(report_path)
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    rows = _read_jsonl(artifact / "private/candidate-rows.jsonl")
    telemetry = _read_jsonl(artifact / "private/generation-telemetry.jsonl")
    attempts = _read_jsonl(artifact / "private/attempt-metadata.jsonl")
    schedule = build_schedule()
    _validate_partial_state(schedule, rows, telemetry, attempts)
    quality = _load_json(root / plan["bindings"]["data_quality"]["path"])
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
    success = all(audit["checks"].values())
    checks = {
        "all_1000_records_accounted": len(terminals) == 1_000,
        "authorization_binding": report.get("authorization_file_sha256") == file_sha256(authorization_path),
        "call_and_retry_ceiling": len(attempts) <= MAX_CALLS
        and sum(1 for item in attempts if item.get("attempt_number") == 2)
        <= MAX_RETRIES,
        "internal_report_hash": internal == scale.sha256_value(unsigned),
        "machine_audit_reproduced": audit == report.get("machine_audit"),
        "no_production_actions": report.get("candidate_registry_changed") is False
        and report.get("dataset_sealed") is False
        and report.get("sft_training_started") is False,
        "rejected_raw_text_absent": all(
            "raw" not in key and "candidate" not in key
            for item in attempts if item.get("accepted") is not True for key in item
        ),
        "runtime_stopped": report.get("runtime_stopped") is True,
    }
    if success:
        sample = scale._owner_sample(rows)  # noqa: SLF001
        checks["owner_packet_exact"] = (
            file_sha256(artifact / "private/owner-review-sample-100.jsonl")
            == report.get("owner_review_sample_file_sha256")
            and len(sample) == 100
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
    _write_json_atomic(artifact / "evidence/run-verification.json", verification)
    if verification["status"] != "PASS":
        raise ValueError("gpt-oss scale-canary independent verification failed")
    return verification
