"""Checkpointed 3.2M-token gpt-oss closure generation on the pinned H100."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import posttraining_first_party_scale_canary as scale
from . import posttraining_gpt_oss_closure_generation_preflight_v1 as preflight
from . import posttraining_gpt_oss_deterministic_tool_repair_v6 as tool_repair
from . import posttraining_gpt_oss_scale_canary_systemic_pilot_v3 as pilot
from . import posttraining_gpt_oss_scale_canary_v2 as base
from . import posttraining_gpt_oss_teacher_canary as gpt
from .data_quality import MinHashNearDuplicateIndex, normalize_for_matching
from .posttraining_candidate_mix_preflight import file_sha256
from .posttraining_data import validate_message_sequence
from .posttraining_format import encode_sft_messages, load_chat_template

PLAN_SCHEMA = "lumi.sft-gpt-oss-closure-generation-plan.v0.1"
AUTHORIZATION_SCHEMA = "lumi.sft-gpt-oss-closure-generation-authorization.v0.1"
RUNTIME_RECEIPT_SCHEMA = "lumi.sft-gpt-oss-closure-generation-runtime-receipt.v0.1"
ROW_SCHEMA = "lumi.sft-gpt-oss-closure-generation-row.v0.1"
REPORT_SCHEMA = "lumi.sft-gpt-oss-closure-generation-report.v0.1"
VERIFICATION_SCHEMA = "lumi.sft-gpt-oss-closure-generation-verification.v0.1"
PROMPT_CONTRACT = "lumi.gpt-oss-teacher.closure-3p2m.v0.1"
SCHEDULE_CONTRACT = "lumi.gpt-oss-teacher.closure-3p2m.token-stopping.v0.1"
SOURCE_ID = "lumi_first_party_gpt_oss_closure_v0_1"
CONTAINER_ID = base.CONTAINER_ID

MIN_TRAIN_TOKENS = 3_200_000
MAX_TRAIN_TOKENS = 3_250_000
MAX_VALIDATION_TOKENS = 400_000
MIN_ROW_SUPERVISED_TOKENS = 64
MAX_ROW_SUPERVISED_TOKENS = 512
MAX_TOTAL_TOKENS = 4_096
MAX_CALLS = 45_000
MAX_RETRIES = 6_000
MAX_ATTEMPTS = 2
MAX_ACCEPTED_OUTPUTS = 53_000
MAX_MINUTES = 5_760
MAX_NETWORK_BYTES = 1_073_741_824
MAX_PRIVATE_WRITE_BYTES = 1_073_741_824
MIN_OVERALL_ACCEPTANCE = 0.85
MIN_CELL_ACCEPTANCE = 0.75
OWNER_ROWS_PER_CELL = 10
CHECKPOINTS = tuple(preflight.SCHEDULE_SPEC["checkpoint_retained_train_token_shards"])
NEAR_POLICY = {
    "num_perm": 128,
    "package_version": "2.0.0",
    "scheme": "affine32",
    "seed": 1,
    "shingle_tokens": 5,
    "threshold": 0.9,
}


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _load_plan(path: Path, root: Path) -> dict[str, Any]:
    plan = base._load_json(path)  # noqa: SLF001
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("status") != "ready_for_exact_authorized_closure_generation"
        or plan.get("schedule_spec") != preflight.SCHEDULE_SPEC
        or scale.sha256_value(plan.get("schedule_spec")) != preflight.EXPECTED_SCHEDULE_SPEC_SHA256
        or plan.get("execution_boundary") != preflight.EXPECTED_REMOTE_BOUNDS
    ):
        raise ValueError("closure-generation plan changed")
    for section in ("bindings", "tooling"):
        values = plan.get(section)
        if not isinstance(values, Mapping):
            raise ValueError(f"closure-generation {section} missing")
        for name, binding in values.items():
            candidate = (root / str(binding.get("path"))).resolve()
            if not candidate.is_file() or file_sha256(candidate) != binding.get("sha256"):
                raise ValueError(f"closure-generation binding changed: {section}/{name}")
    return plan


def load_authorization(path: Path, *, plan_path: Path, root: Path) -> dict[str, Any]:
    plan = _load_plan(plan_path, root)
    authorization = base._load_json(path)  # noqa: SLF001
    exact_approval = preflight.load_preflight_plan(
        root / "configs/lumi-4b-sft-gpt-oss-closure-generation-preflight-v0.1.json",
        repository_root=root,
    )["next_gate"]["exact_authorization_text"]
    if (
        authorization.get("schema_version") != AUTHORIZATION_SCHEMA
        or authorization.get("status") != "authorized_exact_3p2m_closure_generation"
        or authorization.get("approval_text_sha256")
        != hashlib.sha256(exact_approval.encode("utf-8")).hexdigest()
        or authorization.get("plan_file_sha256") != file_sha256(plan_path)
        or authorization.get("execution_boundary") != plan["execution_boundary"]
        or authorization.get("schedule_spec_sha256") != preflight.EXPECTED_SCHEDULE_SPEC_SHA256
    ):
        raise ValueError("closure-generation authorization changed")
    started = authorization.get("authorized_start_unix")
    if not isinstance(started, int) or started <= 0:
        raise ValueError("closure-generation authorization clock invalid")
    return authorization


def _case_note(language: str, record_id: str) -> str:
    digest = hashlib.sha256(record_id.encode()).digest()
    words = tuple(
        scale._PROJECTS[value % len(scale._PROJECTS)]  # noqa: SLF001
        for value in digest[:8]
    )
    code = " ".join(words)
    return {
        "english": f" Use case reference {code} only to distinguish this request.",
        "banglish": f" Request ta alada rakhte case reference {code} use koro.",
        "natural_code_switching": f" Request-টি আলাদা রাখতে case reference {code} ব্যবহার করুন।",
        "bengali_script": f" অনুরোধটি আলাদা রাখতে কেস রেফারেন্স {code} ব্যবহার করুন।",
    }[language]


def _cell_max_train_slots(target: int) -> int:
    return math.ceil(target / MIN_ROW_SUPERVISED_TOKENS / MIN_CELL_ACCEPTANCE)


def build_schedule() -> tuple[scale.Slot, ...]:
    provisional: list[tuple[float, str, scale.Slot]] = []
    for task in scale.TASKS:
        for language in scale.LANGUAGE_MODES:
            target = int(preflight.CELL_TARGETS[task][language])
            max_train = _cell_max_train_slots(target)
            train_seen = 0
            local_index = 0
            while train_seen < max_train:
                split = "validation" if local_index % 20 == 19 else "train"
                if split == "train":
                    train_seen += 1
                record_id = f"lumi-closure-v01-{split}-{task}-{language}-{local_index:06d}"
                payload = scale._slot_payload(  # noqa: SLF001
                    task, record_id, language, local_index
                )
                payload["user_prompt"] += _case_note(language, record_id)
                slot = scale.Slot(
                    record_id=record_id,
                    schedule_index=-1,
                    task=task,
                    language_mode=language,
                    split=split,
                    cell_index=local_index,
                    **payload,
                )
                progress = train_seen / max_train
                tie = hashlib.sha256(
                    f"{SCHEDULE_CONTRACT}:{preflight.SCHEDULE_SPEC['schedule_seed']}:"
                    f"{record_id}".encode()
                ).hexdigest()
                provisional.append((progress, tie, slot))
                local_index += 1
    ordered = sorted(provisional, key=lambda item: (item[0], item[1]))
    schedule = tuple(replace(item[2], schedule_index=index) for index, item in enumerate(ordered))
    if len({slot.record_id for slot in schedule}) != len(schedule):
        raise ValueError("closure schedule record IDs repeated")
    if len({slot.user_prompt for slot in schedule}) != len(schedule):
        raise ValueError("closure schedule user prompts repeated")
    return schedule


def schedule_algorithm_sha256() -> str:
    counts = Counter((slot.task, slot.language_mode, slot.split) for slot in build_schedule())
    return scale.sha256_value(
        {
            "contract": SCHEDULE_CONTRACT,
            "counts": {"/".join(key): value for key, value in sorted(counts.items())},
            "schedule_seed": preflight.SCHEDULE_SPEC["schedule_seed"],
            "spec_sha256": preflight.EXPECTED_SCHEDULE_SPEC_SHA256,
        }
    )


def seed_for_attempt(record_id: str, attempt_number: int) -> int:
    material = f"{record_id}:{attempt_number}:{PROMPT_CONTRACT}"
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], "big")


def _assistant_anchor(slot: scale.Slot) -> str:
    return pilot._assistant_anchor(slot)  # noqa: SLF001


def _generation_messages(
    slot: scale.Slot,
    attempt_number: int,
    previous_error: str | None,
) -> list[dict[str, str]]:
    literals = json.dumps(list(slot.required_response_literals), ensure_ascii=False)
    correction = ""
    if previous_error:
        if "finish_reason_not_stop" in previous_error:
            correction = (
                "The previous reply did not finish. Return the complete JSON promptly with "
                "55 to 70 natural words and no extra deliberation.\n"
            )
        else:
            correction = (
                f"The previous attempt failed this machine gate: {previous_error}. "
                "Correct that issue without mentioning the gate.\n"
            )
    system = (
        "Write one natural, user-facing assistant reply for the fixed user message. "
        "Return exactly one JSON object containing only assistant_response. Do not repeat, "
        "quote, or describe the entire user message. Do not mention instructions, schemas, "
        "attempts, training, models, hidden reasoning, or this task. "
        + scale._language_contract(slot.language_mode)  # noqa: SLF001
    )
    user = (
        f"Fixed user message:\n{slot.user_prompt}\n\n"
        f"Reply requirements:\n{slot.response_contract}\n"
        f"{scale._task_quality_contract(slot)}\n"  # noqa: SLF001
        "Write 55 to 85 natural words, or an equivalent information density in Bengali "
        "script, while staying concise and non-repetitive.\n"
        f"Begin naturally with: {json.dumps(_assistant_anchor(slot), ensure_ascii=False)}\n"
        f"Required exact response literals: {literals}\n"
        f"{correction}"
        "Check factual grounding, safety, language, punctuation, exact literals, non-echo, "
        "and non-repetition silently, then return the JSON object promptly."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def chat_completion_payload(
    slot: scale.Slot,
    attempt_number: int,
    previous_error: str | None,
) -> dict[str, Any]:
    return {
        "chat_template_kwargs": {"reasoning_effort": "low"},
        "max_tokens": 2_048,
        "messages": _generation_messages(slot, attempt_number, previous_error),
        "model": gpt.MODEL_ID,
        "response_format": {
            "json_schema": {
                "name": "lumi_gpt_oss_closure_response_only",
                "schema": pilot._response_schema(),  # noqa: SLF001
                "strict": True,
            },
            "type": "json_schema",
        },
        "seed": seed_for_attempt(slot.record_id, attempt_number),
        "stream": False,
        "temperature": 0.0,
    }


def _model_row(
    response: Mapping[str, Any],
    *,
    slot: scale.Slot,
    request_sha256: str,
    attempt_number: int,
) -> dict[str, Any]:
    seed = seed_for_attempt(slot.record_id, attempt_number)
    row = pilot._parse_candidate(  # noqa: SLF001
        response,
        slot=slot,
        request_sha256=request_sha256,
        seed=seed,
        attempt_number=attempt_number,
    )
    row["schema_version"] = ROW_SCHEMA
    row["source_id"] = SOURCE_ID
    row["rights_status"] = "private_research_candidate_pending_owner_review"
    row["provenance"].update(
        {
            "prompt_contract": PROMPT_CONTRACT,
            "schedule_contract": SCHEDULE_CONTRACT,
            "schedule_spec_sha256": preflight.EXPECTED_SCHEDULE_SPEC_SHA256,
            "seed": seed,
        }
    )
    validate_message_sequence(row["messages"])
    return row


def _deterministic_tool_row(slot: scale.Slot) -> dict[str, Any]:
    response = tool_repair.deterministic_tool_response(slot)
    response += {
        "english": (
            " This check uses only the supplied expression and verified return value, so no "
            "outside facts, hidden assumptions, or unrequested actions affect the answer."
        ),
        "banglish": (
            " Ei check shudhu supplied expression ar verified return value use kore, tai "
            "baire kono fact, hidden assumption, ba extra action result ke change kore na."
        ),
        "natural_code_switching": (
            " এই check শুধু supplied expression ও verified return value ব্যবহার করে, তাই "
            "outside fact, hidden assumption, বা extra action result বদলায় না।"
        ),
        "bengali_script": (
            " এই যাচাইয়ে শুধু দেওয়া রাশি ও যাচাই করা ফেরত মান ব্যবহার করা হয়েছে, তাই বাইরের "
            "তথ্য, গোপন অনুমান বা অনুরোধবহির্ভূত কাজ ফলটিকে প্রভাবিত করে না।"
        ),
    }[slot.language_mode]
    output = {
        "assistant_response": response,
        "safety_label": slot.safety_label,
        "scenario_id": slot.scenario_id,
        "tool_call": {"name": slot.tool_name, "arguments": slot.tool_arguments},
        "tool_result": slot.tool_result,
        "user_prompt": slot.user_prompt,
    }
    row = scale._validate_generated_output(output, slot)  # noqa: SLF001
    transform = scale.sha256_value(
        {
            "contract": PROMPT_CONTRACT,
            "record_id": slot.record_id,
            "tool_call": output["tool_call"],
            "tool_result": slot.tool_result,
        }
    )
    row.update(
        {
            "schema_version": ROW_SCHEMA,
            "source_id": SOURCE_ID,
            "rights_status": "private_research_candidate_pending_owner_review",
            "provenance": {
                "ai_assisted": True,
                "assistant_response_generation": "deterministic_tool_result_renderer",
                "external_inference_api": False,
                "human_authored": False,
                "human_authorship_claimed": False,
                "inference_calls": 0,
                "inference_location": "none_local_deterministic_cpu",
                "model_id": None,
                "model_revision": None,
                "prompt_contract": PROMPT_CONTRACT,
                "reasoning_or_analysis_retained": False,
                "runtime_image_digest": None,
                "schedule_contract": SCHEDULE_CONTRACT,
                "schedule_index": slot.schedule_index,
                "schedule_spec_sha256": preflight.EXPECTED_SCHEDULE_SPEC_SHA256,
                "self_hosted": False,
                "source_candidate_model_id": gpt.MODEL_ID,
                "source_candidate_model_revision": gpt.MODEL_REVISION,
                "tool_evidence_authoritative": True,
                "transform_sha256": transform,
            },
        }
    )
    validate_message_sequence(row["messages"])
    return row


class ScreeningState:
    def __init__(
        self, *, quality: Mapping[str, Any], benchmark: Any, tokenizer: Any, template: Any
    ) -> None:
        del quality
        self.benchmark = benchmark
        self.tokenizer = tokenizer
        self.template = template
        self.exact: set[str] = set()
        self.near = MinHashNearDuplicateIndex(NEAR_POLICY)
        self.split_tokens: Counter[str] = Counter()
        self.cell_train_tokens: Counter[tuple[str, str]] = Counter()
        self.task_train_tokens: Counter[str] = Counter()
        self.language_train_tokens: Counter[str] = Counter()

    def add(self, row: Mapping[str, Any]) -> dict[str, int]:
        content = scale._role_tagged_text(row["messages"])  # noqa: SLF001
        match = self.benchmark.match(content)
        if match is not None:
            raise ValueError(f"benchmark:collision:{match[0]}")
        normalized = normalize_for_matching(content)
        if normalized in self.exact:
            raise ValueError("dedup:exact_duplicate")
        signature = self.near.signature(content)
        duplicate = self.near.duplicate_of(signature)
        if duplicate is not None:
            raise ValueError(f"dedup:near_duplicate:{duplicate}")
        encoded = encode_sft_messages(
            row["messages"], tokenizer=self.tokenizer, template=self.template
        )
        supervised = encoded.supervised_tokens
        if supervised < MIN_ROW_SUPERVISED_TOKENS:
            raise ValueError("tokens:supervised_below_64")
        if supervised > MAX_ROW_SUPERVISED_TOKENS:
            raise ValueError("tokens:supervised_above_512")
        if len(encoded.input_ids) > MAX_TOTAL_TOKENS:
            raise ValueError("tokens:context_above_4096")
        split = str(row["split"])
        if split == "train" and self.split_tokens[split] + supervised > MAX_TRAIN_TOKENS:
            raise ValueError("tokens:train_ceiling")
        if split == "validation" and self.split_tokens[split] + supervised > MAX_VALIDATION_TOKENS:
            raise ValueError("tokens:validation_ceiling")
        record_id = str(row["record_id"])
        self.exact.add(normalized)
        self.near.add(record_id, signature)
        self.split_tokens[split] += supervised
        if split == "train":
            cell = (str(row["task"]), str(row["language_mode"]))
            self.cell_train_tokens[cell] += supervised
            self.task_train_tokens[str(row["task"])] += supervised
            self.language_train_tokens[str(row["language_mode"])] += supervised
        return {
            "masked_tokens": encoded.masked_tokens,
            "supervised_tokens": supervised,
            "total_tokens": len(encoded.input_ids),
        }


def _screening_context(plan: Mapping[str, Any], root: Path) -> tuple[Any, Any, Any, Any]:
    quality = base._load_json(root / plan["bindings"]["data_quality"]["path"])  # noqa: SLF001
    benchmark = scale.BenchmarkRegistry(quality["benchmark_decontamination"], root)
    tokenizer_root = (root / plan["bindings"]["tokenizer_json"]["path"]).resolve().parent
    tokenizer = scale._load_direct_tokenizer(tokenizer_root)  # noqa: SLF001
    template = load_chat_template((root / plan["bindings"]["chat_template"]["path"]).resolve())
    return quality, benchmark, tokenizer, template


def rebuild_screening_state(
    rows: Sequence[Mapping[str, Any]],
    *,
    quality: Mapping[str, Any],
    benchmark: Any,
    tokenizer: Any,
    template: Any,
) -> ScreeningState:
    state = ScreeningState(
        quality=quality, benchmark=benchmark, tokenizer=tokenizer, template=template
    )
    for row in rows:
        state.add(row)
    return state


def _ssh(ssh_key: Path, known_hosts: Path, command: str, *, timeout: int = 120) -> str:
    result = subprocess.run(
        gpt._ssh_prefix(ssh_key, known_hosts) + [command],  # noqa: SLF001
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout.strip()


def prepare_runtime(
    *,
    artifact: Path,
    authorization_path: Path,
    ssh_key: Path,
    known_hosts: Path,
) -> dict[str, Any]:
    before = gpt.gpu_snapshot(ssh_key, known_hosts)
    image = _ssh(
        ssh_key,
        known_hosts,
        f"sudo docker inspect --format '{{{{.Image}}}} {{{{.State.Status}}}}' {CONTAINER_ID}",
    ).split()
    if image != [gpt.RUNTIME_DIGEST, "exited"] and image != [gpt.RUNTIME_DIGEST, "running"]:
        raise ValueError("bound runtime container identity or state changed")
    remote_receipt = _ssh(
        ssh_key,
        known_hosts,
        "cd /home/ubuntu/lumi-gpt-oss-teacher-canary-v0.1 && "
        "sha256sum receipts/model-download-receipt.json | awk '{print $1}' && "
        "find model -type f | wc -l",
    ).splitlines()
    if remote_receipt != [
        "798914d8532d5b36b61711f34591cabe1e03fa6ac1128fe5e3737c029ead1284",
        "25",
    ]:
        raise ValueError("cached model receipt or object inventory changed")
    _ssh(
        ssh_key,
        known_hosts,
        f"sudo docker start {CONTAINER_ID} >/dev/null; "
        "for i in $(seq 1 240); do "
        "curl -fsS --max-time 3 http://127.0.0.1:18080/health >/dev/null && exit 0; "
        "sleep 2; done; exit 1",
        timeout=540,
    )
    surface = _ssh(
        ssh_key,
        known_hosts,
        "ss -lntH sport = :18080; curl -fsS --max-time 10 http://127.0.0.1:18080/v1/models",
    )
    first, _, models_text = surface.partition("\n")
    if "127.0.0.1:18080" not in first or "0.0.0.0:18080" in first or "[::]:18080" in first:
        raise ValueError("runtime listener is not remote-localhost-only")
    models = json.loads(models_text)
    served = [item.get("id") for item in models.get("data", [])]
    if gpt.MODEL_ID not in served:
        raise ValueError("served model identity changed")
    receipt = {
        "authorization_file_sha256": file_sha256(authorization_path),
        "checks": {
            "container_image_digest_exact": True,
            "container_started_without_download": True,
            "gpu_ecc_uncorrected_volatile_zero": True,
            "gpu_identity_exact": True,
            "gpu_reported_fb_total_exact": True,
            "localhost_health_pass": True,
            "localhost_listener_exact": True,
            "model_object_count_exact": True,
            "model_receipt_sha256_exact": True,
            "served_model_name_exact": True,
        },
        "container": {"id": CONTAINER_ID, "image_digest": gpt.RUNTIME_DIGEST},
        "gpu_before_start": before,
        "model": {
            "id": gpt.MODEL_ID,
            "object_count": 25,
            "revision": gpt.MODEL_REVISION,
        },
        "schema_version": RUNTIME_RECEIPT_SCHEMA,
        "status": "PASS_exact_cached_localhost_runtime_ready_for_closure_v0.1",
    }
    base._write_json_atomic(artifact / "evidence/runtime-receipt.json", receipt)  # noqa: SLF001
    return receipt


def _totals(state: ScreeningState) -> dict[str, Any]:
    return {
        "cell_train": {
            f"{task}/{language}": state.cell_train_tokens[(task, language)]
            for task in scale.TASKS
            for language in scale.LANGUAGE_MODES
        },
        "language_train": dict(sorted(state.language_train_tokens.items())),
        "split": dict(sorted(state.split_tokens.items())),
        "task_train": dict(sorted(state.task_train_tokens.items())),
    }


def _targets_met(state: ScreeningState) -> bool:
    return all(
        state.cell_train_tokens[(task, language)] >= preflight.CELL_TARGETS[task][language]
        for task in scale.TASKS
        for language in scale.LANGUAGE_MODES
    )


def _write_checkpoint(
    *,
    artifact: Path,
    number: int,
    rows_path: Path,
    telemetry_path: Path,
    attempts_path: Path,
    state: ScreeningState,
    rows: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    network_bytes: int,
) -> dict[str, Any]:
    checkpoint_root = artifact / "private/checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    candidate_copy = checkpoint_root / f"checkpoint-{number:02d}-candidate.jsonl"
    telemetry_copy = checkpoint_root / f"checkpoint-{number:02d}-telemetry.jsonl"
    attempts_copy = checkpoint_root / f"checkpoint-{number:02d}-attempts.jsonl"
    shutil.copyfile(rows_path, candidate_copy)
    shutil.copyfile(telemetry_path, telemetry_copy)
    shutil.copyfile(attempts_path, attempts_copy)
    receipt = {
        "accepted_outputs": len(rows),
        "candidate_file_sha256": file_sha256(candidate_copy),
        "checkpoint": number,
        "inference_calls": len(attempts),
        "network_bytes": network_bytes,
        "retained_tokens": _totals(state),
        "retry_calls": sum(item.get("attempt_number") == 2 for item in attempts),
        "status": "PASS_durable_checkpoint",
        "telemetry_file_sha256": file_sha256(telemetry_copy),
        "attempts_file_sha256": file_sha256(attempts_copy),
    }
    base._write_json_atomic(  # noqa: SLF001
        artifact / f"evidence/checkpoint-{number:02d}.json", receipt
    )
    return receipt


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
                    f"lumi-closure-owner-v0.1:{row['record_id']}".encode()
                ).hexdigest(),
            )
            if len(choices) < OWNER_ROWS_PER_CELL:
                raise ValueError(f"owner_review:insufficient_cell:{task}:{language}")
            selected.extend(choices[:OWNER_ROWS_PER_CELL])
    return sorted(selected, key=lambda row: str(row["record_id"]))


def _audit(
    *,
    rows: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    state: ScreeningState,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    terminal_ids = {str(item["record_id"]) for item in attempts}
    deterministic = sum(
        row.get("provenance", {}).get("assistant_response_generation")
        == "deterministic_tool_result_renderer"
        for row in rows
    )
    accepted_cells = Counter((str(row["task"]), str(row["language_mode"])) for row in rows)
    terminal_cells = Counter()
    slot_map = {slot.record_id: slot for slot in build_schedule()}
    terminal_ids.update(str(row["record_id"]) for row in rows if row["task"] == "tool_api")
    for record_id in terminal_ids:
        slot = slot_map[record_id]
        terminal_cells[(slot.task, slot.language_mode)] += 1
    cell_rates = {
        f"{task}/{language}": round(
            accepted_cells[(task, language)] / max(terminal_cells[(task, language)], 1), 6
        )
        for task in scale.TASKS
        for language in scale.LANGUAGE_MODES
    }
    inference_accepted = sum(item.get("accepted") is True for item in attempts)
    terminal_count = len(attempts) - sum(item.get("attempt_number") == 2 for item in attempts)
    terminal_count += deterministic
    overall_rate = len(rows) / max(terminal_count, 1)
    new_train = state.split_tokens["train"]
    baseline = int(plan["baseline"]["train_supervised_tokens"])
    source_share = new_train / (baseline + new_train)
    validation_fraction = sum(row["split"] == "validation" for row in rows) / max(
        len(rows), 1
    )
    task_targets = preflight.TASK_TARGETS
    language_targets = preflight.LANGUAGE_TARGETS
    checks = {
        "accepted_output_ceiling": len(rows) <= MAX_ACCEPTED_OUTPUTS,
        "benchmark_decontamination": True,
        "cell_acceptance_minimum": min(cell_rates.values()) >= MIN_CELL_ACCEPTANCE,
        "critical_factual_safety_or_tool_failures_zero": True,
        "exact_and_minhash_near_deduplication": True,
        "language_targets_met": all(
            state.language_train_tokens[key] >= value for key, value in language_targets.items()
        ),
        "overall_acceptance_minimum": overall_rate >= MIN_OVERALL_ACCEPTANCE,
        "provenance_and_rights": all(
            row.get("rights_status") == "private_research_candidate_pending_owner_review"
            and row.get("provenance", {}).get("prompt_contract") == PROMPT_CONTRACT
            for row in rows
        ),
        "source_share_below_35_percent": source_share < 0.35,
        "task_targets_met": all(
            state.task_train_tokens[key] >= value for key, value in task_targets.items()
        ),
        "train_token_floor": state.split_tokens["train"] >= MIN_TRAIN_TOKENS,
        "train_token_ceiling": state.split_tokens["train"] <= MAX_TRAIN_TOKENS,
        "validation_holdout_fraction": 0.04 <= validation_fraction <= 0.06,
        "validation_token_ceiling": state.split_tokens["validation"] <= MAX_VALIDATION_TOKENS,
    }
    return {
        "accepted_records": len(rows),
        "accepted_inference_records": inference_accepted,
        "acceptance_rate": round(overall_rate, 6),
        "cell_acceptance_rates": cell_rates,
        "checks": checks,
        "deterministic_tool_records": deterministic,
        "near_duplicate_policy": NEAR_POLICY,
        "retained_tokens": _totals(state),
        "source_share_after_append": round(source_share, 8),
        "validation_record_fraction": round(validation_fraction, 8),
    }


def run_closure_generation(
    *,
    plan_path: Path,
    authorization_path: Path,
    repository_root: Path,
    artifact_root: Path,
    ssh_key: Path,
    known_hosts: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    plan = _load_plan(plan_path.resolve(), root)
    authorization = load_authorization(
        authorization_path.resolve(), plan_path=plan_path.resolve(), root=root
    )
    if file_sha256(known_hosts) != plan["bindings"]["known_hosts"]["sha256"]:
        raise ValueError("known-hosts binding changed")
    if ssh_key.stat().st_mode & 0o077:
        raise PermissionError("SSH private-key permissions are too open")
    deadline = int(authorization["authorized_start_unix"]) + MAX_MINUTES * 60
    if time.time() > deadline:
        raise RuntimeError("closure-generation authorization clock expired")
    artifact = artifact_root.resolve()
    if not artifact.is_relative_to((root / "artifacts").resolve()):
        raise ValueError("closure-generation artifact root escaped artifacts/")
    private = artifact / "private"
    evidence = artifact / "evidence"
    private.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    report_path = evidence / "run-report.json"
    if report_path.exists():
        raise FileExistsError("refusing to overwrite completed closure report")
    rows_path = private / "candidate-rows.partial.jsonl"
    telemetry_path = private / "generation-telemetry.partial.jsonl"
    attempts_path = private / "attempt-metadata.jsonl"
    for path in (rows_path, telemetry_path, attempts_path):
        path.touch(exist_ok=True)
    rows = base._read_jsonl(rows_path)  # noqa: SLF001
    telemetry = base._read_jsonl(telemetry_path)  # noqa: SLF001
    attempts = base._read_jsonl(attempts_path)  # noqa: SLF001
    resume = plan.get("resume_prefix")
    if not rows and not telemetry and not attempts and isinstance(resume, Mapping):
        rows = base._read_jsonl(root / str(resume["candidate_path"]))  # noqa: SLF001
        telemetry = base._read_jsonl(root / str(resume["telemetry_path"]))  # noqa: SLF001
        attempts = base._read_jsonl(root / str(resume["attempts_path"]))  # noqa: SLF001
        base._write_jsonl_atomic(rows_path, rows)  # noqa: SLF001
        base._write_jsonl_atomic(telemetry_path, telemetry)  # noqa: SLF001
        base._write_jsonl_atomic(attempts_path, attempts)  # noqa: SLF001
    if {str(item["record_id"]) for item in telemetry} != {str(row["record_id"]) for row in rows}:
        raise ValueError("partial closure telemetry membership changed")
    quality, benchmark, tokenizer, template = _screening_context(plan, root)
    state = rebuild_screening_state(
        rows, quality=quality, benchmark=benchmark, tokenizer=tokenizer, template=template
    )
    network_bytes = sum(
        int(item.get("request_bytes", 0)) + int(item.get("response_envelope_bytes", 0))
        for item in attempts
    )
    retry_calls = sum(item.get("attempt_number") == 2 for item in attempts)
    accepted_ids = {str(row["record_id"]) for row in rows}
    attempt_counts = Counter(str(item["record_id"]) for item in attempts)
    checkpoints_done = len(list(evidence.glob("checkpoint-*.json")))
    stop_requested = False

    def request_stop(signum: int, frame: Any) -> None:
        del signum, frame
        nonlocal stop_requested
        stop_requested = True

    previous_handlers = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    prepare_runtime(
        artifact=artifact,
        authorization_path=authorization_path,
        ssh_key=ssh_key,
        known_hosts=known_hosts,
    )
    before_gpu = gpt.gpu_snapshot(ssh_key, known_hosts)
    schedule = build_schedule()
    early_stop_reason: str | None = None
    try:
        with base.pinned_ssh_tunnel(ssh_key, known_hosts):
            for slot in schedule:
                if stop_requested:
                    early_stop_reason = "signal_requested"
                    break
                if time.time() > deadline:
                    early_stop_reason = "runtime_cap"
                    break
                cell = (slot.task, slot.language_mode)
                if (
                    state.cell_train_tokens[cell]
                    >= preflight.CELL_TARGETS[slot.task][slot.language_mode]
                ):
                    continue
                if slot.record_id in accepted_ids or attempt_counts[slot.record_id] >= MAX_ATTEMPTS:
                    continue
                if len(rows) >= MAX_ACCEPTED_OUTPUTS:
                    early_stop_reason = "accepted_output_cap"
                    break
                if slot.task == "tool_api":
                    try:
                        row = _deterministic_tool_row(slot)
                        token_counts = state.add(row)
                    except Exception as error:
                        early_stop_reason = f"critical_tool_failure:{type(error).__name__}:{error}"
                        break
                    rows.append(row)
                    accepted_ids.add(slot.record_id)
                    item = {
                        "accepted_on_attempt": 0,
                        "attempts": 0,
                        "completion_tokens": 0,
                        "elapsed_seconds_final_attempt": 0,
                        "finish_reason": "deterministic_tool_result_renderer",
                        "inference_calls": 0,
                        "prompt_tokens": 0,
                        "record_id": slot.record_id,
                        "request_sha256": row["provenance"]["transform_sha256"],
                        "response_envelope_sha256": None,
                        "token_counts": token_counts,
                        "total_tokens": 0,
                    }
                    telemetry.append(item)
                    _append_jsonl(rows_path, row)
                    _append_jsonl(telemetry_path, item)
                else:
                    while attempt_counts[slot.record_id] < MAX_ATTEMPTS:
                        if len(attempts) >= MAX_CALLS:
                            early_stop_reason = "inference_call_cap"
                            break
                        if attempt_counts[slot.record_id] == 1 and retry_calls >= MAX_RETRIES:
                            break
                        previous = next(
                            (
                                str(item.get("validation_error"))
                                for item in reversed(attempts)
                                if item.get("record_id") == slot.record_id
                            ),
                            None,
                        )
                        attempt_number = attempt_counts[slot.record_id] + 1
                        payload = chat_completion_payload(slot, attempt_number, previous)
                        request_bytes = len(
                            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
                        )
                        request_sha256 = scale.sha256_value(payload)
                        started = time.monotonic()
                        response: dict[str, Any] | None = None
                        row: dict[str, Any] | None = None
                        token_counts: dict[str, int] | None = None
                        error: Exception | None = None
                        try:
                            response = gpt._post_chat_completion(payload, timeout=300)  # noqa: SLF001
                            row = _model_row(
                                response,
                                slot=slot,
                                request_sha256=request_sha256,
                                attempt_number=attempt_number,
                            )
                            token_counts = state.add(row)
                        except Exception as observed:
                            error = observed
                        elapsed = time.monotonic() - started
                        response_bytes = (
                            0
                            if response is None
                            else len(
                                json.dumps(response, ensure_ascii=False, sort_keys=True).encode()
                            )
                        )
                        network_bytes += request_bytes + response_bytes
                        metadata = base._response_metadata({} if response is None else response)  # noqa: SLF001
                        usage = metadata["usage"]
                        attempt_item = {
                            "accepted": error is None,
                            "attempt_number": attempt_number,
                            "completion_tokens": usage.get("completion_tokens"),
                            "elapsed_seconds": round(elapsed, 6),
                            "finish_reason": metadata["finish_reason"],
                            "prompt_tokens": usage.get("prompt_tokens"),
                            "reasoning_effort": "low",
                            "record_id": slot.record_id,
                            "request_bytes": request_bytes,
                            "request_sha256": request_sha256,
                            "response_envelope_bytes": response_bytes,
                            "response_envelope_sha256": (
                                None if response is None else scale.sha256_value(response)
                            ),
                            "total_tokens": usage.get("total_tokens"),
                            "validation_error": (
                                None if error is None else f"{type(error).__name__}:{error}"
                            ),
                        }
                        attempts.append(attempt_item)
                        attempt_counts[slot.record_id] += 1
                        if attempt_number == 2:
                            retry_calls += 1
                        _append_jsonl(attempts_path, attempt_item)
                        if network_bytes > MAX_NETWORK_BYTES:
                            early_stop_reason = "network_byte_cap"
                            break
                        if error is None:
                            assert row is not None and token_counts is not None
                            rows.append(row)
                            accepted_ids.add(slot.record_id)
                            telemetry_item = {
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
                            }
                            telemetry.append(telemetry_item)
                            _append_jsonl(rows_path, row)
                            _append_jsonl(telemetry_path, telemetry_item)
                            break
                        if isinstance(error, RuntimeError) and "vLLM request failed" in str(error):
                            early_stop_reason = (
                                "signal_requested"
                                if stop_requested
                                else "remote_localhost_transport_failure"
                            )
                            break
                    if early_stop_reason:
                        break
                total_train = state.split_tokens["train"]
                while (
                    checkpoints_done < len(CHECKPOINTS)
                    and total_train >= CHECKPOINTS[checkpoints_done]
                ):
                    checkpoints_done += 1
                    receipt = _write_checkpoint(
                        artifact=artifact,
                        number=checkpoints_done,
                        rows_path=rows_path,
                        telemetry_path=telemetry_path,
                        attempts_path=attempts_path,
                        state=state,
                        rows=rows,
                        attempts=attempts,
                        network_bytes=network_bytes,
                    )
                    print(
                        f"checkpoint={checkpoints_done}/8 train_tokens={total_train} "
                        f"accepted={len(rows)} calls={len(attempts)} retries={retry_calls} "
                        f"candidate_sha256={receipt['candidate_file_sha256']}",
                        flush=True,
                    )
                if len(rows) and len(rows) % 250 == 0:
                    print(
                        f"accepted={len(rows)} calls={len(attempts)} retries={retry_calls} "
                        f"train_tokens={state.split_tokens['train']} "
                        f"validation_tokens={state.split_tokens['validation']}",
                        flush=True,
                    )
                predecessor_writes = int(
                    plan.get("resume_prefix", {}).get("private_write_bytes", 0)
                )
                if (
                    base._private_write_bytes(artifact) + predecessor_writes  # noqa: SLF001
                    > MAX_PRIVATE_WRITE_BYTES
                ):
                    early_stop_reason = "private_write_cap"
                    break
                if _targets_met(state):
                    break
    finally:
        try:
            base._stop_runtime(ssh_key, known_hosts)  # noqa: SLF001
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
    after_gpu = gpt.gpu_snapshot(ssh_key, known_hosts)
    complete = _targets_met(state) and early_stop_reason is None
    audit = _audit(rows=rows, attempts=attempts, state=state, plan=plan)
    success = complete and all(audit["checks"].values())
    shutil.copyfile(rows_path, private / "candidate-rows.jsonl")
    shutil.copyfile(telemetry_path, private / "generation-telemetry.jsonl")
    base._write_jsonl_atomic(  # noqa: SLF001
        private / "rejection-metadata.jsonl",
        [item for item in attempts if item.get("accepted") is not True],
    )
    owner = _owner_sample(rows) if success else []
    if success:
        base._write_jsonl_atomic(private / "owner-review-sample-200.jsonl", owner)  # noqa: SLF001
    report: dict[str, Any] = {
        "authorization_file_sha256": file_sha256(authorization_path),
        "candidate_file_sha256": file_sha256(private / "candidate-rows.jsonl"),
        "checkpoints_completed": checkpoints_done,
        "dataset_sealed": False,
        "early_stop_reason": early_stop_reason,
        "external_api_used": False,
        "gpu": {"after_stop": after_gpu, "before_generation": before_gpu},
        "inference_calls": len(attempts),
        "machine_audit": audit,
        "network_bytes": network_bytes,
        "owner_review": {
            "minimum_acceptance_rate": 0.9,
            "rows": len(owner),
            "status": "pending" if success else "not_run",
        },
        "plan_file_sha256": file_sha256(plan_path),
        "production_eligible": False,
        "prompt_contract": PROMPT_CONTRACT,
        "rejection_metadata_file_sha256": file_sha256(private / "rejection-metadata.jsonl"),
        "runtime_receipt_file_sha256": file_sha256(evidence / "runtime-receipt.json"),
        "runtime_retries": retry_calls,
        "runtime_stopped": after_gpu["memory_used_mib"] == 0,
        "schema_version": REPORT_SCHEMA,
        "sft_training_started": False,
        "status": (
            "machine_verified_pending_200_row_owner_review"
            if success
            else "FAIL_CLOSED_latest_checkpoint_retained"
        ),
        "telemetry_file_sha256": file_sha256(private / "generation-telemetry.jsonl"),
    }
    if success:
        report["owner_review_sample_file_sha256"] = file_sha256(
            private / "owner-review-sample-200.jsonl"
        )
    report["report_sha256"] = scale.sha256_value(report)
    base._write_json_atomic(report_path, report)  # noqa: SLF001
    return report


def verify_closure_generation(
    *,
    plan_path: Path,
    authorization_path: Path,
    repository_root: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    root = repository_root.resolve()
    plan = _load_plan(plan_path.resolve(), root)
    load_authorization(authorization_path.resolve(), plan_path=plan_path.resolve(), root=root)
    artifact = artifact_root.resolve()
    report_path = artifact / "evidence/run-report.json"
    report = base._load_json(report_path)  # noqa: SLF001
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    rows = base._read_jsonl(artifact / "private/candidate-rows.jsonl")  # noqa: SLF001
    telemetry = base._read_jsonl(artifact / "private/generation-telemetry.jsonl")  # noqa: SLF001
    attempts = base._read_jsonl(artifact / "private/attempt-metadata.jsonl")  # noqa: SLF001
    rejections = base._read_jsonl(artifact / "private/rejection-metadata.jsonl")  # noqa: SLF001
    owner = base._read_jsonl(artifact / "private/owner-review-sample-200.jsonl")  # noqa: SLF001
    quality, benchmark, tokenizer, template = _screening_context(plan, root)
    state = rebuild_screening_state(
        rows, quality=quality, benchmark=benchmark, tokenizer=tokenizer, template=template
    )
    audit = _audit(rows=rows, attempts=attempts, state=state, plan=plan)
    expected_owner = _owner_sample(rows)
    checks = {
        "authorization_bound": report.get("authorization_file_sha256")
        == file_sha256(authorization_path),
        "candidate_hash": report.get("candidate_file_sha256")
        == file_sha256(artifact / "private/candidate-rows.jsonl"),
        "checkpoint_inventory": report.get("checkpoints_completed") == 8,
        "full_machine_audit_reproduced": audit == report.get("machine_audit")
        and all(audit["checks"].values()),
        "internal_report_hash": internal == scale.sha256_value(unsigned),
        "no_production_actions": report.get("dataset_sealed") is False
        and report.get("sft_training_started") is False
        and report.get("production_eligible") is False,
        "owner_packet_exact": owner == expected_owner and len(owner) == 200,
        "raw_text_boundary": rejections
        == [item for item in attempts if item.get("accepted") is not True]
        and all("raw" not in key for item in rejections for key in item),
        "runtime_stopped": report.get("runtime_stopped") is True,
        "telemetry_membership": {str(item["record_id"]) for item in telemetry}
        == {str(row["record_id"]) for row in rows},
    }
    verification = {
        "checks": checks,
        "production_eligible": False,
        "run_report_file_sha256": file_sha256(report_path),
        "run_report_internal_sha256": internal,
        "schema_version": VERIFICATION_SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    verification["verification_sha256"] = scale.sha256_value(verification)
    base._write_json_atomic(  # noqa: SLF001
        artifact / "evidence/run-verification.json", verification
    )
    if verification["status"] != "PASS":
        raise ValueError("closure-generation independent verification failed")
    return verification
