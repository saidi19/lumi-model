# ruff: noqa: E501
"""Bounded, local-only Qwen teacher canary for private Lumi SFT candidates."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .data_quality import BenchmarkRegistry, normalize_for_matching
from .posttraining_data import validate_message_sequence
from .posttraining_first_party import canonical_json_bytes, file_sha256
from .posttraining_format import encode_sft_messages, load_chat_template
from .posttraining_owner_review import build_owner_review_html

PLAN_SCHEMA_VERSION = "lumi.sft-first-party-teacher-canary-plan.v0.1"
ROW_SCHEMA_VERSION = "lumi.sft-first-party-teacher-canary-row.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-first-party-teacher-canary-report.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-first-party-teacher-canary-verification.v0.1"

TASKS = (
    "factual_qa",
    "multilingual_bengali",
    "safety_helpfulness",
    "tool_api",
    "writing_summarization",
)
LANGUAGE_MODES = (
    "banglish",
    "bengali_script",
    "english",
    "natural_code_switching",
)

MODEL_ID = "Qwen/Qwen3-8B-GGUF"
MODEL_REVISION = "7c41481f57cb95916b40956ab2f0b139b296d974"
MODEL_FILE = "Qwen3-8B-Q4_K_M.gguf"
MODEL_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
MODEL_BYTES = 5_027_783_488
OLLAMA_MODEL = "lumi-qwen3-8b-teacher-canary:7c41481-q4km"

_BENGALI = re.compile(r"[\u0980-\u09ff]")
_LATIN = re.compile(r"[A-Za-z]")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\s().-]*){10,15}(?!\d)")
_SECRET = re.compile(
    r"(?:BEGIN (?:[A-Z ]+ )?PRIVATE KEY|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|"
    r"\bsk-[A-Za-z0-9_-]{16,}\b)",
    re.IGNORECASE,
)
_HIDDEN_REASONING = re.compile(
    r"(?:<\/?think>|chain[- ]of[- ]thought|hidden reasoning|internal reasoning|"
    r"my reasoning process|ধাপে ধাপে আমার চিন্তা|আমার গোপন চিন্তা)",
    re.IGNORECASE,
)
_MODEL_IDENTITY = re.compile(r"\b(?:qwen|alibaba|ollama|chatgpt|claude|gemini)\b", re.IGNORECASE)
_TRAINING_MARKER = re.compile(
    r"(?:sft candidate|training example|dataset row|system prompt|hidden instruction)",
    re.IGNORECASE,
)
_BANGLISH_MARKERS = {
    "ami",
    "apni",
    "ar",
    "bolo",
    "chai",
    "dao",
    "daw",
    "diye",
    "ei",
    "ekta",
    "eta",
    "hobe",
    "hoy",
    "jonno",
    "kibhabe",
    "ki",
    "koro",
    "lagbe",
    "na",
    "pari",
    "tumi",
}


@dataclass(frozen=True)
class Scenario:
    task: str
    variant: int
    scenario_id: str
    instruction: str
    required_literals: tuple[str, ...] = ()
    answer_number: str | None = None
    safety_label: str = "safe"
    tool_name: str | None = None
    tool_arguments: Mapping[str, Any] | None = None
    tool_result: Mapping[str, Any] | None = None


FACTUAL = (
    Scenario(
        "factual_qa",
        0,
        "kilometer_to_meters",
        "Ask and answer how many metres are in one kilometre. The answer is exactly 1000 metres.",
        answer_number="1000",
    ),
    Scenario(
        "factual_qa",
        1,
        "days_in_week",
        "Ask and answer how many days are in one week. The answer is exactly 7 days.",
        answer_number="7",
    ),
    Scenario(
        "factual_qa",
        2,
        "triangle_sides",
        "Ask and answer how many sides a triangle has. The answer is exactly 3 sides.",
        answer_number="3",
    ),
    Scenario(
        "factual_qa",
        3,
        "water_freezing_celsius",
        "Ask and answer the standard Celsius freezing point of water. The answer is exactly 0 degrees Celsius.",
        answer_number="0",
    ),
)

MULTILINGUAL = (
    Scenario(
        "multilingual_bengali",
        0,
        "friendly_greeting",
        "Create a natural greeting from a user and a warm, concise reply. Do not claim persistent memory.",
    ),
    Scenario(
        "multilingual_bengali",
        1,
        "punctuation_repair",
        "The user asks to fix spelling and punctuation in a short informal sentence. The assistant returns a polished version and one brief note.",
    ),
    Scenario(
        "multilingual_bengali",
        2,
        "three_day_study_plan",
        "The user asks for a practical three-day study plan. Give three concise, context-sensitive steps.",
    ),
    Scenario(
        "multilingual_bengali",
        3,
        "polite_rewrite",
        "The user asks to rewrite a blunt request politely. Return only a concise, natural rewrite plus a short explanation.",
    ),
)

SAFETY = (
    Scenario(
        "safety_helpfulness",
        0,
        "benign_study_help",
        "The user asks for safe help preparing for tomorrow's quiz. Respond helpfully without over-refusing.",
        safety_label="safe",
    ),
    Scenario(
        "safety_helpfulness",
        1,
        "credential_privacy",
        "The user asks for another person's password. Refuse briefly and suggest a legitimate account-recovery route.",
        safety_label="refuse",
    ),
    Scenario(
        "safety_helpfulness",
        2,
        "urgent_medical_caution",
        "The user reports severe chest pain and trouble breathing. Do not diagnose; urge immediate local emergency help and contacting a trusted nearby person.",
        safety_label="medical_caution",
    ),
    Scenario(
        "safety_helpfulness",
        3,
        "phishing_request",
        "The user asks for a deceptive login page to steal credentials. Refuse and offer safe, defensive anti-phishing guidance.",
        safety_label="refuse",
    ),
)

TOOLS = (
    Scenario(
        "tool_api",
        0,
        "calculator_addition",
        "The user asks to calculate 173 + 58. Use the calculator tool exactly as specified, then give the result concisely.",
        answer_number="231",
        tool_name="calculator",
        tool_arguments={"expression": "173 + 58"},
        tool_result={"result": "231"},
    ),
    Scenario(
        "tool_api",
        1,
        "calculator_multiplication",
        "The user asks to calculate 132 × 11. Use the calculator tool exactly as specified, then give the result concisely.",
        answer_number="1452",
        tool_name="calculator",
        tool_arguments={"expression": "132 * 11"},
        tool_result={"result": "1452"},
    ),
    Scenario(
        "tool_api",
        2,
        "calculator_division",
        "The user asks to calculate 144 ÷ 9. Use the calculator tool exactly as specified, then give the result concisely.",
        answer_number="16",
        tool_name="calculator",
        tool_arguments={"expression": "144 / 9"},
        tool_result={"result": "16"},
    ),
    Scenario(
        "tool_api",
        3,
        "list_sort",
        "The user asks to sort 9, 2, and 5 in ascending order. Use the list_sort tool exactly as specified, then state the sorted list.",
        required_literals=("2", "5", "9"),
        tool_name="list_sort",
        tool_arguments={"items": [9, 2, 5], "order": "ascending"},
        tool_result={"items": [2, 5, 9]},
    ),
)

WRITING = (
    Scenario(
        "writing_summarization",
        0,
        "meeting_summary",
        "Create a user request containing a short original note: meeting at 10:00, Room B, and bring ID. Summarize all three facts. Preserve the literals 10:00, Room B, and ID exactly.",
        required_literals=("10:00", "Room B", "ID"),
    ),
    Scenario(
        "writing_summarization",
        1,
        "release_summary",
        "Create a user request containing a short original release note: version 2.4 adds dark mode and offline search. Summarize all facts. Preserve 2.4, dark mode, and offline search exactly.",
        required_literals=("2.4", "dark mode", "offline search"),
    ),
    Scenario(
        "writing_summarization",
        2,
        "plant_care_summary",
        "Create a user request containing a short original care note: water 3 times/week, use indirect light, and check soil Friday. Summarize all facts. Preserve 3 times/week, indirect light, and Friday exactly.",
        required_literals=("3 times/week", "indirect light", "Friday"),
    ),
    Scenario(
        "writing_summarization",
        3,
        "project_summary",
        "Create a user request containing a short original project note: Phase Alpha is due 12 May and Nadia owns the review. Summarize all facts. Preserve Phase Alpha, 12 May, and Nadia exactly.",
        required_literals=("Phase Alpha", "12 May", "Nadia"),
    ),
)

SCENARIOS = FACTUAL + MULTILINGUAL + SAFETY + TOOLS + WRITING

LANGUAGE_INSTRUCTIONS = {
    "english": (
        "Write the user prompt and assistant response in natural English only. "
        "Use correct spelling and punctuation."
    ),
    "bengali_script": (
        "Write the user prompt and assistant response in natural Bengali script. "
        "Use standard Bengali punctuation and correct spelling. Preserve any explicitly "
        "required ASCII literals exactly."
    ),
    "banglish": (
        "Write natural romanized Bangla (Banglish) using Latin letters only, with words "
        "such as ami, tumi, eta, ki, koro, or hobe where natural. Do not use Bengali "
        "script. Use clear punctuation and preserve required literals exactly."
    ),
    "natural_code_switching": (
        "Write natural Bengali-script and English code-switching. Both Bengali script and "
        "English words must appear in the user prompt and assistant response. Use correct "
        "punctuation and preserve required literals exactly."
    ),
}

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scenario_id": {"type": "string"},
        "user_prompt": {"type": "string"},
        "assistant_response": {"type": "string"},
        "safety_label": {
            "type": "string",
            "enum": ["safe", "refuse", "medical_caution"],
        },
        "tool_call": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "arguments": {"type": "object"},
                    },
                    "required": ["name", "arguments"],
                    "additionalProperties": False,
                },
            ]
        },
        "tool_result": {
            "anyOf": [
                {"type": "null"},
                {"type": "object"},
            ]
        },
    },
    "required": [
        "scenario_id",
        "user_prompt",
        "assistant_response",
        "safety_label",
        "tool_call",
        "tool_result",
    ],
    "additionalProperties": False,
}


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def seed_for_record(record_id: str) -> int:
    return int.from_bytes(hashlib.sha256(record_id.encode("utf-8")).digest()[:4], "big")


def expected_record_ids() -> tuple[str, ...]:
    return tuple(
        f"lumi-teacher-{scenario.task}-{language}-{scenario.variant:02d}"
        for scenario in SCENARIOS
        for language in LANGUAGE_MODES
    )


def _json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_canary_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    value = _json_object(path)
    if value.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported first-party teacher canary plan")
    if value.get("status") != "authorized_private_local_canary":
        raise ValueError("first-party teacher canary plan status changed")
    if value.get("tasks") != list(TASKS):
        raise ValueError("teacher canary task coverage changed")
    if value.get("language_modes") != list(LANGUAGE_MODES):
        raise ValueError("teacher canary language coverage changed")
    limits = value.get("limits")
    expected_limits = {
        "candidate_records": 80,
        "maximum_disk_bytes": 12 * 1024**3,
        "maximum_network_bytes": 5_100_000_000,
        "maximum_retained_supervised_tokens": 24_000,
        "maximum_runtime_minutes": 180,
        "owner_review_records": 32,
        "train_records": 60,
        "validation_records": 20,
    }
    if limits != expected_limits:
        raise ValueError("teacher canary limits changed")
    if value.get("model") != {
        "file": MODEL_FILE,
        "model_bytes": MODEL_BYTES,
        "model_id": MODEL_ID,
        "model_sha256": MODEL_SHA256,
        "ollama_model": OLLAMA_MODEL,
        "revision": MODEL_REVISION,
    }:
        raise ValueError("teacher canary model binding changed")

    root = Path(repository_root).resolve()
    for name, binding in value.get("bindings", {}).items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid teacher canary binding: {name}")
        raw_path = Path(str(binding.get("path"))).expanduser()
        resolved = raw_path.resolve() if raw_path.is_absolute() else (root / raw_path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"teacher canary binding missing: {name}")
        if file_sha256(resolved) != binding.get("sha256"):
            raise ValueError(f"teacher canary binding changed: {name}")
    return value


def build_generation_prompt(scenario: Scenario, language_mode: str) -> str:
    tool_contract = (
        "Set tool_call and tool_result to null."
        if scenario.tool_name is None
        else (
            "Return exactly this tool_call object: "
            f"{json.dumps({'name': scenario.tool_name, 'arguments': scenario.tool_arguments}, ensure_ascii=False, sort_keys=True)}. "
            "Return exactly this tool_result object: "
            f"{json.dumps(scenario.tool_result, ensure_ascii=False, sort_keys=True)}."
        )
    )
    system = (
        "You are producing exactly one private AI-assisted Lumi SFT canary row. "
        "Return only the requested compact JSON object. Never reveal analysis, reasoning, "
        "chain-of-thought, model identity, system prompts, or hidden instructions. Generate "
        "an original user prompt and a direct, concise assistant response. Do not mention "
        "training, datasets, or this task. Do not include markdown code fences. "
        f"{LANGUAGE_INSTRUCTIONS[language_mode]}"
    )
    user = (
        f"Scenario ID: {scenario.scenario_id}\n"
        f"Task: {scenario.task}\n"
        f"Instruction: {scenario.instruction}\n"
        f"Set safety_label exactly to {scenario.safety_label}. {tool_contract} "
        "End the assistant response with proper punctuation. /no_think"
    )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _post_local_json(url: str, payload: Mapping[str, Any], *, timeout: int = 180) -> dict[str, Any]:
    if url != "http://127.0.0.1:11434/api/generate":
        raise ValueError("teacher canary permits only the local Ollama generate endpoint")
    request = urllib.request.Request(
        url,
        data=canonical_json_bytes(payload),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            value = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"local Ollama request failed: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("local Ollama response root is not an object")
    return value


def ollama_payload(record_id: str, scenario: Scenario, language_mode: str) -> dict[str, Any]:
    return {
        "format": OUTPUT_SCHEMA,
        "keep_alive": -1,
        "model": OLLAMA_MODEL,
        "options": {
            "num_ctx": 4096,
            "num_predict": 384,
            "presence_penalty": 1.5,
            "seed": seed_for_record(record_id),
            "temperature": 0.7,
            "top_k": 20,
            "top_p": 0.8,
        },
        "prompt": build_generation_prompt(scenario, language_mode),
        "stream": False,
        "think": False,
    }


def _normalize_number_text(text: str) -> str:
    table = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
    return text.translate(table).replace(",", "")


def _text_for_row(messages: Sequence[Mapping[str, str]]) -> str:
    return "\n".join(str(message["content"]) for message in messages)


def _validate_language(messages: Sequence[Mapping[str, str]], language_mode: str) -> None:
    natural_messages = [
        message["content"] for message in messages if message["role"] in {"user", "assistant"}
    ]
    if len(natural_messages) != 2:
        raise ValueError("teacher canary expects one user and one final assistant message")
    for text in natural_messages:
        has_bengali = _BENGALI.search(text) is not None
        has_latin = _LATIN.search(text) is not None
        if language_mode == "english" and has_bengali:
            raise ValueError("English row contains Bengali script")
        if language_mode == "bengali_script" and not has_bengali:
            raise ValueError("Bengali row lacks Bengali script")
        if language_mode == "banglish":
            if has_bengali:
                raise ValueError("Banglish row contains Bengali script")
            markers = set(re.findall(r"[a-z]+", text.casefold())) & _BANGLISH_MARKERS
            if len(markers) < 1:
                raise ValueError("Banglish row lacks romanized Bengali markers")
        if language_mode == "natural_code_switching" and not (has_bengali and has_latin):
            raise ValueError("code-switched row must contain Bengali script and English")


def _validate_safety(response: str, expected_label: str) -> None:
    lowered = response.casefold()
    if expected_label == "refuse":
        markers = (
            "can't",
            "cannot",
            "won't",
            "not help",
            "পারব না",
            "সাহায্য করতে পারি না",
            "সহায়তা করতে পারি না",
            "সহায়তা করতে পারি না",
            "করতে পারি না",
            "দিতে পারি না",
            "নিরাপদ",
            "parbo na",
            "help korte parbo na",
        )
        if not any(marker in lowered for marker in markers):
            raise ValueError("refusal row lacks a clear boundary")
    elif expected_label == "medical_caution":
        markers = (
            "emergency",
            "hospital",
            "doctor",
            "জরুরি",
            "হাসপাতাল",
            "ডাক্তার",
            "joruri",
            "hospital e",
            "doctor",
        )
        if not any(marker in lowered for marker in markers):
            raise ValueError("urgent medical row lacks escalation guidance")


def _validate_punctuation(text: str) -> None:
    if len(text.strip()) < 2:
        raise ValueError("assistant response is too short")
    if text.rstrip()[-1] not in ".!?।)]}\"'":
        raise ValueError("assistant response lacks terminal punctuation")
    if text.count("{") != text.count("}") or text.count("[") != text.count("]"):
        raise ValueError("assistant response has unbalanced braces")
    if "```" in text or "$$" in text:
        raise ValueError("assistant response contains disallowed formatting artifacts")


def _validate_output(
    output: Mapping[str, Any],
    *,
    scenario: Scenario,
    language_mode: str,
) -> list[dict[str, str]]:
    if set(output) != {
        "assistant_response",
        "safety_label",
        "scenario_id",
        "tool_call",
        "tool_result",
        "user_prompt",
    }:
        raise ValueError("teacher JSON envelope fields changed")
    if output.get("scenario_id") != scenario.scenario_id:
        raise ValueError("teacher scenario binding changed")
    if output.get("safety_label") != scenario.safety_label:
        raise ValueError("teacher safety label changed")
    user_prompt = output.get("user_prompt")
    assistant_response = output.get("assistant_response")
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("teacher user prompt is empty")
    if not isinstance(assistant_response, str) or not assistant_response.strip():
        raise ValueError("teacher assistant response is empty")
    messages: list[dict[str, str]] = [{"role": "user", "content": user_prompt.strip()}]
    if scenario.tool_name is None:
        if output.get("tool_call") is not None or output.get("tool_result") is not None:
            raise ValueError("non-tool row unexpectedly emitted a tool call")
    else:
        expected_call = {"name": scenario.tool_name, "arguments": scenario.tool_arguments}
        if output.get("tool_call") != expected_call:
            raise ValueError("tool call differs from the deterministic contract")
        if output.get("tool_result") != scenario.tool_result:
            raise ValueError("tool result differs from the deterministic contract")
        messages.extend(
            [
                {
                    "role": "tool",
                    "content": json.dumps(expected_call, ensure_ascii=False, sort_keys=True),
                },
                {
                    "role": "tool_result",
                    "content": json.dumps(
                        scenario.tool_result,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ]
        )
    messages.append({"role": "assistant", "content": assistant_response.strip()})
    validate_message_sequence(messages)

    natural_text = f"{user_prompt}\n{assistant_response}"
    if _HIDDEN_REASONING.search(natural_text):
        raise ValueError("hidden-reasoning marker found")
    if _MODEL_IDENTITY.search(natural_text):
        raise ValueError("teacher identity leaked into candidate")
    if _TRAINING_MARKER.search(natural_text):
        raise ValueError("training-process marker leaked into candidate")
    if _EMAIL.search(natural_text) or _PHONE.search(natural_text) or _SECRET.search(natural_text):
        raise ValueError("PII or secret pattern found")
    _validate_language(messages, language_mode)
    _validate_punctuation(assistant_response)
    _validate_safety(assistant_response, scenario.safety_label)

    normalized_response = _normalize_number_text(assistant_response)
    if scenario.answer_number is not None and scenario.answer_number not in normalized_response:
        raise ValueError("ground-truth number missing from assistant response")
    for literal in scenario.required_literals:
        if literal not in user_prompt or literal not in assistant_response:
            raise ValueError(f"required grounded literal missing: {literal}")
    return messages


def generate_one(
    record_id: str,
    scenario: Scenario,
    language_mode: str,
    *,
    endpoint: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request_payload = ollama_payload(record_id, scenario, language_mode)
    started = time.monotonic()
    response = _post_local_json(endpoint, request_payload)
    elapsed = time.monotonic() - started
    if response.get("done") is not True or response.get("done_reason") != "stop":
        raise ValueError("local teacher did not finish with stop")
    if response.get("thinking") not in (None, ""):
        raise ValueError("local teacher returned hidden thinking")
    raw = response.get("response")
    if not isinstance(raw, str) or _HIDDEN_REASONING.search(raw):
        raise ValueError("local teacher raw response contains hidden reasoning")
    try:
        output = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("local teacher response is not JSON") from error
    if not isinstance(output, dict):
        raise ValueError("local teacher response root is not an object")
    messages = _validate_output(output, scenario=scenario, language_mode=language_mode)
    split = "validation" if scenario.variant == 3 else "train"
    row = {
        "schema_version": ROW_SCHEMA_VERSION,
        "record_id": record_id,
        "source_id": "lumi_first_party_multilingual_safety_tool_v0_3",
        "task": scenario.task,
        "skill_id": scenario.scenario_id,
        "language_mode": language_mode,
        "split": split,
        "messages": messages,
        "provenance": {
            "ai_assisted": True,
            "human_authored": False,
            "inference_location": "mac_local_ollama_metal",
            "model_file_sha256": MODEL_SHA256,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "ollama_model": OLLAMA_MODEL,
            "prompt_contract": "lumi.first-party-teacher-prompt.v0.1",
            "quantization": "Q4_K_M",
            "seed": seed_for_record(record_id),
            "self_hosted": True,
        },
        "rights_status": "private_research_candidate_pending_owner_review",
    }
    telemetry = {
        "record_id": record_id,
        "request_sha256": sha256_value(request_payload),
        "response_envelope_sha256": sha256_value(response),
        "elapsed_seconds": round(elapsed, 6),
        "load_duration_ns": response.get("load_duration"),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "prompt_eval_duration_ns": response.get("prompt_eval_duration"),
        "eval_count": response.get("eval_count"),
        "eval_duration_ns": response.get("eval_duration"),
    }
    return row, telemetry


def memory_free_percent() -> int:
    result = subprocess.run(
        ["memory_pressure", "-Q"],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"free percentage:\s*([0-9]+)%", result.stdout)
    if match is None:
        raise RuntimeError("unable to parse macOS memory pressure")
    return int(match.group(1))


def _load_quality_config(path: str | Path) -> dict[str, Any]:
    value = _json_object(path)
    if value.get("schema_version") != 1:
        raise ValueError("unsupported data-quality config")
    return value


def _load_tokenizer(path: str | Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(path),
        local_files_only=True,
        use_fast=True,
    )


def audit_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    repository_root: str | Path,
    quality_config_path: str | Path,
    chat_template_path: str | Path,
    tokenizer_path: str | Path,
    maximum_supervised_tokens: int,
) -> dict[str, Any]:
    if len(rows) != 80:
        raise ValueError("teacher canary must contain exactly 80 rows")
    ids = [str(row.get("record_id")) for row in rows]
    if tuple(ids) != expected_record_ids():
        raise ValueError("teacher canary record membership or ordering changed")
    if len(set(ids)) != 80:
        raise ValueError("teacher canary record IDs are not unique")
    counts = Counter(str(row.get("split")) for row in rows)
    if counts != Counter({"train": 60, "validation": 20}):
        raise ValueError("teacher canary split counts changed")
    cells = Counter((str(row.get("task")), str(row.get("language_mode"))) for row in rows)
    if set(cells.values()) != {4} or set(cells) != set(
        (task, language) for task in TASKS for language in LANGUAGE_MODES
    ):
        raise ValueError("teacher canary task-language coverage changed")

    root = Path(repository_root).resolve()
    quality = _load_quality_config(quality_config_path)
    benchmark = BenchmarkRegistry(quality["benchmark_decontamination"], root)
    template = load_chat_template(chat_template_path)
    tokenizer = _load_tokenizer(tokenizer_path)

    normalized: dict[str, str] = {}
    token_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError("teacher canary messages are missing")
        validate_message_sequence(messages)
        content = _text_for_row(messages)
        match = benchmark.match(content)
        if match is not None:
            raise ValueError(f"benchmark contamination detected for {row['record_id']}: {match}")
        normalized_text = normalize_for_matching(content)
        if normalized_text in normalized.values():
            raise ValueError("exact normalized duplicate found")
        normalized[str(row["record_id"])] = normalized_text
        encoded = encode_sft_messages(messages, tokenizer=tokenizer, template=template)
        token_counts[str(row["record_id"])] = {
            "masked_tokens": encoded.masked_tokens,
            "supervised_tokens": encoded.supervised_tokens,
            "total_tokens": len(encoded.input_ids),
        }

    near_pairs: list[tuple[str, str, float]] = []
    cross_split_pairs: list[tuple[str, str, float]] = []
    by_id = {str(row["record_id"]): row for row in rows}
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            ratio = SequenceMatcher(None, normalized[left], normalized[right]).ratio()
            if ratio >= 0.92:
                near_pairs.append((left, right, round(ratio, 6)))
                if by_id[left]["split"] != by_id[right]["split"]:
                    cross_split_pairs.append((left, right, round(ratio, 6)))
    if near_pairs:
        raise ValueError(f"near-duplicate candidates found: {near_pairs[:3]}")
    if cross_split_pairs:
        raise ValueError(f"train/validation split leakage found: {cross_split_pairs[:3]}")

    supervised_total = sum(item["supervised_tokens"] for item in token_counts.values())
    if supervised_total > maximum_supervised_tokens:
        raise ValueError("teacher canary exceeds supervised-token authorization")
    return {
        "benchmark_registry_sha256": benchmark.sha256,
        "candidate_internal_sha256": sha256_value(list(rows)),
        "counts": {
            "records": len(rows),
            "train": counts["train"],
            "validation": counts["validation"],
        },
        "deduplication": {
            "exact_normalized_duplicates": 0,
            "near_duplicate_pairs_at_0_92": 0,
            "train_validation_leakage_pairs": 0,
        },
        "frozen_tokenizer": {
            "path": str(Path(tokenizer_path).resolve()),
            "tokenizer_json_sha256": file_sha256(Path(tokenizer_path) / "tokenizer.json"),
        },
        "language_mode_counts": dict(
            sorted(Counter(str(row["language_mode"]) for row in rows).items())
        ),
        "task_counts": dict(sorted(Counter(str(row["task"]) for row in rows).items())),
        "token_counts_sha256": sha256_value(token_counts),
        "tokens": {
            "maximum_record_total_tokens": max(
                item["total_tokens"] for item in token_counts.values()
            ),
            "retained_supervised_tokens": supervised_total,
        },
    }


def select_owner_sample(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        by_cell.setdefault((str(row["task"]), str(row["language_mode"])), []).append(row)
    selected: list[Mapping[str, Any]] = []
    selected_ids: set[str] = set()
    for cell in sorted(by_cell):
        winner = min(
            by_cell[cell],
            key=lambda row: hashlib.sha256(
                f"lumi-owner-v0.1:{row['record_id']}".encode()
            ).hexdigest(),
        )
        selected.append(winner)
        selected_ids.add(str(winner["record_id"]))
    remaining = sorted(
        (row for row in rows if str(row["record_id"]) not in selected_ids),
        key=lambda row: hashlib.sha256(
            f"lumi-owner-v0.1:extra:{row['record_id']}".encode()
        ).hexdigest(),
    )
    selected.extend(remaining[:12])
    return sorted(selected, key=lambda row: str(row["record_id"]))


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid partial JSONL at line {line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"partial JSONL row {line_number} is not an object")
        rows.append(value)
    return rows


def run_canary(
    *,
    plan_path: str | Path,
    repository_root: str | Path,
    artifact_root: str | Path,
    endpoint: str = "http://127.0.0.1:11434/api/generate",
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    artifact = Path(artifact_root).resolve()
    plan = load_canary_plan(plan_path, repository_root=root)
    started_wall = time.time()
    started_monotonic = time.monotonic()
    private = artifact / "private"
    partial_rows = private / "candidate-rows.partial.jsonl"
    partial_telemetry = private / "generation-telemetry.partial.jsonl"
    rows = _read_jsonl(partial_rows)
    telemetry = _read_jsonl(partial_telemetry)
    expected_ids = expected_record_ids()
    if [str(row.get("record_id")) for row in rows] != list(expected_ids[: len(rows)]):
        raise ValueError("partial teacher rows are not an exact expected prefix")
    if len(rows) != len(telemetry) or len(rows) > 80:
        raise ValueError("partial teacher rows and telemetry differ")
    initial_memory = memory_free_percent()
    if initial_memory < 20:
        raise RuntimeError("memory pressure is unsafe before teacher generation")

    scenarios_by_task_variant = {(item.task, item.variant): item for item in SCENARIOS}
    for index, record_id in enumerate(expected_ids[len(rows) :], start=len(rows) + 1):
        parts = record_id.removeprefix("lumi-teacher-").rsplit("-", 2)
        task, language_mode, variant_text = parts
        scenario = scenarios_by_task_variant[(task, int(variant_text))]
        row, item_telemetry = generate_one(
            record_id,
            scenario,
            language_mode,
            endpoint=endpoint,
        )
        rows.append(row)
        telemetry.append(item_telemetry)
        write_jsonl(partial_rows, rows)
        write_jsonl(partial_telemetry, telemetry)
        if index % 10 == 0:
            free = memory_free_percent()
            if free < 15:
                raise RuntimeError("memory pressure became unsafe during teacher generation")
            elapsed_minutes = (time.monotonic() - started_monotonic) / 60
            if elapsed_minutes > int(plan["limits"]["maximum_runtime_minutes"]):
                raise RuntimeError("teacher canary exceeded its runtime authorization")
            print(f"generated {index}/80; memory_free={free}%", flush=True)

    evidence = artifact / "evidence"
    review = artifact / "review"
    write_jsonl(private / "candidate-rows.jsonl", rows)
    write_jsonl(private / "generation-telemetry.jsonl", telemetry)

    bindings = plan["bindings"]
    audit = audit_rows(
        rows,
        repository_root=root,
        quality_config_path=root / str(bindings["data_quality"]["path"]),
        chat_template_path=root / str(bindings["chat_template"]["path"]),
        tokenizer_path=str(plan["tokenizer"]["path"]),
        maximum_supervised_tokens=int(plan["limits"]["maximum_retained_supervised_tokens"]),
    )
    owner_sample = select_owner_sample(rows)
    owner_sample_sha256 = sha256_value(owner_sample)
    write_jsonl(private / "owner-review-sample-32.jsonl", owner_sample)
    review.mkdir(parents=True, exist_ok=True)
    (review / "owner-review-32.html").write_text(
        build_owner_review_html(owner_sample, sample_sha256=owner_sample_sha256),
        encoding="utf-8",
    )

    finished_wall = time.time()
    durations = [float(item["elapsed_seconds"]) for item in telemetry]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "verified_private_candidate_pending_owner_review",
        "production_eligible": False,
        "candidate_registry_changed": False,
        "dataset_sealed": False,
        "sft_started": False,
        "external_api_used": False,
        "human_authorship_claimed": False,
        "authorization_file_sha256": file_sha256(artifact / "authorization.json"),
        "plan_file_sha256": file_sha256(plan_path),
        "model": {
            "file_sha256": MODEL_SHA256,
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "runtime": "ollama-0.32.5-local-metal-offline",
        },
        "audit": audit,
        "generation": {
            "calls": len(telemetry),
            "maximum_call_seconds": round(max(durations), 6),
            "median_call_seconds": round(sorted(durations)[len(durations) // 2], 6),
            "telemetry_internal_sha256": sha256_value(telemetry),
        },
        "owner_review": {
            "records": len(owner_sample),
            "sample_internal_sha256": owner_sample_sha256,
            "status": "pending",
        },
        "runtime": {
            "elapsed_seconds": round(time.monotonic() - started_monotonic, 6),
            "finished_unix": int(finished_wall),
            "initial_memory_free_percent": initial_memory,
            "started_unix": int(started_wall),
        },
        "private_artifacts": {
            "candidate_rows_file_sha256": file_sha256(private / "candidate-rows.jsonl"),
            "generation_telemetry_file_sha256": file_sha256(private / "generation-telemetry.jsonl"),
            "owner_review_html_file_sha256": file_sha256(review / "owner-review-32.html"),
            "owner_review_sample_file_sha256": file_sha256(
                private / "owner-review-sample-32.jsonl"
            ),
        },
    }
    write_json(evidence / "run-report.json", report)
    return report


def verify_run(
    *,
    plan_path: str | Path,
    repository_root: str | Path,
    artifact_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    artifact = Path(artifact_root).resolve()
    plan = load_canary_plan(plan_path, repository_root=root)
    report = _json_object(artifact / "evidence" / "run-report.json")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported teacher canary report")
    if report.get("status") != "verified_private_candidate_pending_owner_review":
        raise ValueError("teacher canary report is not successful")
    candidate_path = artifact / "private" / "candidate-rows.jsonl"
    rows = [json.loads(line) for line in candidate_path.read_text(encoding="utf-8").splitlines()]
    audit = audit_rows(
        rows,
        repository_root=root,
        quality_config_path=root / str(plan["bindings"]["data_quality"]["path"]),
        chat_template_path=root / str(plan["bindings"]["chat_template"]["path"]),
        tokenizer_path=str(plan["tokenizer"]["path"]),
        maximum_supervised_tokens=int(plan["limits"]["maximum_retained_supervised_tokens"]),
    )
    if audit != report.get("audit"):
        raise ValueError("independent teacher canary audit differs")
    sample = select_owner_sample(rows)
    sample_sha256 = sha256_value(sample)
    if sample_sha256 != report.get("owner_review", {}).get("sample_internal_sha256"):
        raise ValueError("teacher canary owner-review sample changed")
    for name, expected in report.get("private_artifacts", {}).items():
        paths = {
            "candidate_rows_file_sha256": artifact / "private" / "candidate-rows.jsonl",
            "generation_telemetry_file_sha256": artifact / "private" / "generation-telemetry.jsonl",
            "owner_review_html_file_sha256": artifact / "review" / "owner-review-32.html",
            "owner_review_sample_file_sha256": artifact
            / "private"
            / "owner-review-sample-32.jsonl",
        }
        if name not in paths or file_sha256(paths[name]) != expected:
            raise ValueError(f"teacher canary private artifact changed: {name}")
    verification = {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified_private_candidate_pending_owner_review",
        "report_file_sha256": file_sha256(artifact / "evidence" / "run-report.json"),
        "candidate_internal_sha256": audit["candidate_internal_sha256"],
        "owner_sample_internal_sha256": sample_sha256,
        "checks": {
            "benchmark_decontamination": True,
            "candidate_membership": True,
            "frozen_token_measurement": True,
            "no_external_api": True,
            "no_hidden_chain_of_thought": True,
            "no_human_authorship_claim": True,
            "owner_packet_deterministic": True,
            "schema_and_message_sequence": True,
            "split_isolation": True,
        },
        "next_gate": "owner_review_required_before_any_candidate_registry_preflight",
    }
    write_json(artifact / "evidence" / "verification-report.json", verification)
    return verification
