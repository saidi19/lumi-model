"""CPU-only curation of the pinned Lumi SFT closure-canary sources."""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data_quality import BenchmarkRegistry, matching_tokens, normalize_for_matching
from .posttraining_data import evaluate_sft_record, load_sft_policy, validate_message_sequence
from .posttraining_format import (
    ChatTemplate,
    ChatTokenizer,
    encode_sft_messages,
    load_chat_template,
)
from .posttraining_owner_review import build_owner_review_html

SCHEMA_VERSION = "lumi.sft-closure-canary-curation-report.v0.1"
MANIFEST_SCHEMA_VERSION = "lumi.sft-closure-canary-private-manifest.v0.1"
PLAN_SHA256 = "31568114288e46c1c8613e20d9167af952050e5ed2a2556a64722bcec9da488a"
TOKENIZER_SHA256 = "1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee"
BENCHMARK_SHA256 = "1ffb5e938ab28c2fc876b9efd6c6ed59059f1e85aed2c396c024ae5925a4745e"
PYARROW_VERSION = "25.0.0"
PYARROW_WHEEL_SHA256 = (
    "8831a3ba52fa7cdb78d368d968b1dcd06171e6dff5461e16d90de91d371e47bc"
)
MAXIMUM_PRIVATE_WRITES = 4_294_967_296
MAXIMUM_TRAIN_TOKENS = 3_200_000
MAXIMUM_GENERAL_TRAIN_TOKENS = 2_900_000
MAXIMUM_AUXILIARY_TRAIN_TOKENS = 300_000
CURRENT_ACCEPTED_TRAIN_TOKENS = 7_256_670
TEN_MILLION_TOKENS = 10_000_000
VALIDATION_DENOMINATOR = 20

SOURCE_SPECS = {
    "nemotron_tool": {
        "source_id": "nvidia_nemotron_post_training_tool_canary",
        "dataset_id": "nvidia/Nemotron-Post-Training-Dataset-v1",
        "revision": "74e23eb6f830fef4a9e96a92f6f6262214cbb9a8",
        "license_id": "cc-by-4.0",
        "uri": (
            "https://huggingface.co/datasets/nvidia/"
            "Nemotron-Post-Training-Dataset-v1/tree/"
            "74e23eb6f830fef4a9e96a92f6f6262214cbb9a8"
        ),
    },
    "aya_bengali": {
        "source_id": "coherelabs_aya_bengali_closure_canary",
        "dataset_id": "CohereLabs/aya_collection_language_split",
        "revision": "a3af2fde4b4cb5b2775830b11244a1a20b5f004f",
        "license_id": "apache-2.0",
        "uri": (
            "https://huggingface.co/datasets/CohereLabs/"
            "aya_collection_language_split/tree/"
            "a3af2fde4b4cb5b2775830b11244a1a20b5f004f"
        ),
    },
    "banglatlit": {
        "source_id": "banglatlit_auxiliary_closure_canary",
        "dataset_id": "aplycaebous/BanglaTLit",
        "revision": "22cacbaac72bf6ee828f9d6be1e28bfd7a84843f",
        "license_id": "mit",
        "uri": (
            "https://huggingface.co/datasets/aplycaebous/BanglaTLit/tree/"
            "22cacbaac72bf6ee828f9d6be1e28bfd7a84843f"
        ),
    },
    "banglabridge": {
        "source_id": "banglabridge_instructions_closure_canary",
        "dataset_id": "subhajitmahata84/banglabridge-instructions",
        "revision": "b5ace9b2e56702cfd3aa5fe2e8c3870f67f57300",
        "license_id": "cc-by-4.0",
        "uri": (
            "https://huggingface.co/datasets/subhajitmahata84/"
            "banglabridge-instructions/tree/"
            "b5ace9b2e56702cfd3aa5fe2e8c3870f67f57300"
        ),
    },
}

LANE_TARGETS = {
    "nemotron_tool": 1_300_000,
    "aya_simplification": 500_000,
    "aya_summarization": 400_000,
    "aya_generation": 400_000,
    "banglatlit": 300_000,
}
RESERVOIR_LIMITS = {
    "nemotron_tool": 16_000,
    "aya_simplification": 22_000,
    "aya_summarization": 18_000,
    "aya_generation": 18_000,
    "banglatlit": 45_000,
    "banglabridge": 1_000,
}
AYA_DATASET_LANES = {
    "Wiki-split-inst (T)": ("aya_simplification", "writing_summarization"),
    "Flan-GEM-wiki-lingua (T)": ("aya_summarization", "writing_summarization"),
    "Joke-explaination-inst (T)": ("aya_generation", "multilingual_bengali"),
}
SOURCE_FILE_NAMES = {
    "nemotron_tool": "nemotron-tool-candidate.jsonl",
    "aya_bengali": "aya-bengali-candidate.jsonl",
    "banglatlit": "banglatlit-auxiliary-candidate.jsonl",
    "banglabridge": "banglabridge-candidate.jsonl",
}

RAW_OBJECTS = {
    "CohereLabs_aya_collection_language_split/README.md": (
        126_536,
        "453f2533dff0f575876bd5919e695db17f0b9c1d1b76b7d35e85ee0e9845bed1",
    ),
    "CohereLabs_aya_collection_language_split/bengali/train-00000-of-00003.parquet": (
        442_809_026,
        "8d4079c0f6e329280208cdb50d7c42164dd5c02781aab78cdf99780a606c9810",
    ),
    "aplycaebous_BanglaTLit/BanglaTLit_train.csv": (
        35_427_904,
        "b0c8b7bff33b6cb5edb20902b933a6d8a980173370cf2ac05b283ad3ae8f23da",
    ),
    "aplycaebous_BanglaTLit/README.md": (
        1_719,
        "812d5b605626923150b12493c2feaccf153c79906b9e6dd8581fef53e0c049df",
    ),
    "nvidia_Nemotron-Post-Training-Dataset-v1/README.md": (
        11_540,
        "4a06db710f8482a0dac092128bfad9368d0b082452126e578648b82036a0d941",
    ),
    "nvidia_Nemotron-Post-Training-Dataset-v1/data/chat-00000-of-00008.parquet": (
        149_037_549,
        "e7dc33d8e42829a47e29f1729c14a3ac12918405f1f5a4cf504d2f65b0bdf0c4",
    ),
    "nvidia_Nemotron-Post-Training-Dataset-v1/data/tool-00000-of-00013.parquet": (
        212_135_247,
        "661402cf61925e05b5b718031de3162fa854a220ace43e0135b3054b38cafcd5",
    ),
    "subhajitmahata84_banglabridge-instructions/README.md": (
        4_063,
        "f97019bf4d561e5c399418181347eec9448984fc34e181f13fca53ac1a7b5850",
    ),
    "subhajitmahata84_banglabridge-instructions/banglish_finetune.adapted.jsonl": (
        1_350_638,
        "313e5efb38760d8de09050162b6d8c5a2db76b9107cb2c24d828012113a7a73d",
    ),
}

_BENGALI = re.compile(r"[\u0980-\u09ff]")
_LATIN = re.compile(r"[A-Za-z]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HIDDEN_REASONING = re.compile(r"</?(?:think|analysis|reasoning)>|\bchain[- ]of[- ]thought\b", re.I)
_LEADING_PRIVATE_REASONING = re.compile(
    r"\A\s*<(?P<tag>think|analysis|reasoning)>\s*.*?\s*</(?P=tag)>\s*"
    r"(?P<visible>.+?)\s*\Z",
    re.I | re.S,
)
_UNSAFE_OPERATIONAL = re.compile(
    r"\b(?:credential\s*steal|keylogger|ransomware|phishing\s*kit|reverse\s*shell|"
    r"build\s+(?:a\s+)?bomb|suicide\s+method|child\s+sexual|credit\s+card\s+fraud)\b",
    re.I,
)
_BANGLISH_MARKERS = re.compile(
    r"\b(?:ami|amar|apni|ase|ache|bhalo|bol|chai|diye|eita|eta|hobe|jonno|"
    r"kibhabe|kivabe|ki|koro|korbo|nai|na|tumi|tomar)\b",
    re.I,
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_raw_objects(raw_root: Path) -> list[dict[str, Any]]:
    discovered = {
        str(path.relative_to(raw_root)): path
        for path in raw_root.rglob("*")
        if path.is_file()
    }
    if set(discovered) != set(RAW_OBJECTS):
        missing = sorted(set(RAW_OBJECTS) - set(discovered))
        unexpected = sorted(set(discovered) - set(RAW_OBJECTS))
        raise ValueError(
            f"closure-canary raw inventory changed: missing={missing}, "
            f"unexpected={unexpected}"
        )
    inventory: list[dict[str, Any]] = []
    for relative, (expected_bytes, expected_sha256) in sorted(RAW_OBJECTS.items()):
        path = discovered[relative]
        observed_bytes = path.stat().st_size
        observed_sha256 = file_sha256(path)
        if observed_bytes != expected_bytes or observed_sha256 != expected_sha256:
            raise ValueError(f"closure-canary raw object changed: {relative}")
        inventory.append(
            {
                "bytes": observed_bytes,
                "path": relative,
                "sha256": observed_sha256,
            }
        )
    return inventory


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


def load_direct_tokenizer(path: Path) -> _DirectTokenizerAdapter:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    value = json.loads(path.read_text(encoding="utf-8"))
    special_ids = sorted(
        int(token["id"])
        for token in value["added_tokens"]
        if token.get("special") is True
    )
    return _DirectTokenizerAdapter(tokenizer, special_ids)


def _source_record(
    *,
    lane: str,
    document_id: str,
    messages: list[dict[str, str]],
    language: str,
    task_category: str,
    quality_score: float,
    transform: str,
    extra_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_lane = "aya_bengali" if lane.startswith("aya_") else lane
    spec = SOURCE_SPECS[source_lane]
    content_identity = {
        "language": language,
        "messages": messages,
        "task_category": task_category,
    }
    content_sha256 = sha256_value(content_identity)
    document_digest = hashlib.sha256(document_id.encode()).hexdigest()[:24]
    record_id = f"closure-{spec['source_id']}-{document_digest}"
    split_digest = hashlib.sha256(f"closure-split-v0.1\0{record_id}".encode()).digest()
    split = (
        "validation"
        if int.from_bytes(split_digest[:8], "big") % VALIDATION_DENOMINATOR == 0
        else "train"
    )
    provenance: dict[str, Any] = {
        "consent_basis": "source_level_open_license",
        "content_sha256": content_sha256,
        "owner_review_status": "pending_deterministic_spot_check",
        "transform_sha256": sha256_value(
            {"adapter": transform, "revision": spec["revision"], "source": spec["source_id"]}
        ),
    }
    if extra_provenance:
        provenance.update(extra_provenance)
    return {
        "schema_version": "lumi.sft.v0.1",
        "record_id": record_id,
        "split": split,
        "language": language,
        "task_category": task_category,
        "messages": messages,
        "source": {
            "source_id": spec["source_id"],
            "dataset_id": spec["dataset_id"],
            "document_id": document_id,
            "license_id": spec["license_id"],
            "revision": spec["revision"],
            "uri": spec["uri"],
        },
        "provenance": provenance,
        "quality": {
            "score": quality_score,
            "review_method": "deterministic_machine_gates_owner_review_pending",
            "review_status": "accepted",
            "pii_scan": "passed",
            "secret_scan": "passed",
        },
        "safety": {"policy_version": "lumi-assistant-behavior-v0.1", "risk_labels": []},
    }


def _priority(record: Mapping[str, Any]) -> int:
    return int.from_bytes(
        hashlib.sha256(f"closure-priority-v0.1\0{record['record_id']}".encode()).digest()[:8],
        "big",
    )


def _reservoir_add(
    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]],
    lane: str,
    record: dict[str, Any],
) -> None:
    limit = RESERVOIR_LIMITS[lane]
    priority = _priority(record)
    item = (-priority, str(record["record_id"]), record)
    heap = heaps.setdefault(lane, [])
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def _text_quality_reasons(text: str) -> list[str]:
    reasons: list[str] = []
    stripped = text.strip()
    if not stripped:
        reasons.append("empty_text")
        return reasons
    if _CONTROL.search(stripped):
        reasons.append("control_character")
    if _HIDDEN_REASONING.search(stripped):
        reasons.append("hidden_reasoning_marker")
    if _UNSAFE_OPERATIONAL.search(stripped):
        reasons.append("unsafe_operational_content")
    if any(stripped.count(character) > 2_000 for character in ("{", "}", "[", "]")):
        reasons.append("pathological_punctuation")
    if re.search(r"([!?.,।])\1{5,}", stripped):
        reasons.append("repeated_punctuation")
    words = matching_tokens(stripped)
    if len(words) >= 80 and len(set(words)) / len(words) < 0.12:
        reasons.append("low_lexical_diversity")
    return reasons


def _language_reasons(record: Mapping[str, Any]) -> list[str]:
    text = "\n".join(str(message["content"]) for message in record["messages"])
    bengali = len(_BENGALI.findall(text))
    latin = len(_LATIN.findall(text))
    language = record["language"]
    if language == "bn" and bengali < 8:
        return ["bengali_script_signal_too_low"]
    if language == "en" and (latin < 20 or bengali > 0):
        return ["english_script_contract_failed"]
    if language == "multilingual" and latin < 8:
        return ["multilingual_latin_signal_too_low"]
    return []


def _custom_quality_reasons(record: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    for message in record["messages"]:
        reasons.extend(_text_quality_reasons(str(message["content"])))
    reasons.extend(_language_reasons(record))
    return list(dict.fromkeys(reasons))


def _visible_assistant_content(value: str) -> tuple[str | None, bool]:
    content = value.strip()
    if _HIDDEN_REASONING.search(content) is None:
        return content, False
    match = _LEADING_PRIVATE_REASONING.fullmatch(content)
    if match is None:
        return None, False
    visible = match.group("visible").strip()
    if not visible or _HIDDEN_REASONING.search(visible) is not None:
        return None, False
    return visible, True


def _tool_record(row: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    if (
        row.get("license") != "CC BY 4.0"
        or row.get("category") != "tool_calling"
        or not isinstance(row.get("uuid"), str)
    ):
        return None, "source_contract"
    try:
        metadata = json.loads(str(row.get("metadata") or ""))
    except json.JSONDecodeError:
        return None, "metadata_json"
    tools = metadata.get("tools") if isinstance(metadata, dict) else None
    if not isinstance(tools, list) or not tools:
        return None, "tool_schema_missing"
    tools_json = json.dumps(tools, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(tools_json) > 32_000:
        return None, "tool_schema_too_long"
    source_messages = row.get("messages")
    if not isinstance(source_messages, list) or not 2 <= len(source_messages) <= 64:
        return None, "message_count"
    messages: list[dict[str, str]] = []
    discarded_reasoning_blocks = 0
    tool_calls_seen = 0
    index = 0
    while index < len(source_messages):
        message = source_messages[index]
        role = message.get("role")
        content = message.get("content")
        calls = message.get("tool_calls") or []
        if role == "user":
            if not isinstance(content, str) or not content.strip():
                return None, "empty_user"
            if not messages:
                content = (
                    f"Available tools (JSON):\n{tools_json}\n\n"
                    f"User request:\n{content.strip()}"
                )
            messages.append({"role": "user", "content": content.strip()})
            index += 1
            continue
        if role == "assistant" and calls:
            if not isinstance(calls, list):
                return None, "tool_calls_schema"
            results: list[Mapping[str, Any]] = []
            cursor = index + 1
            while cursor < len(source_messages) and source_messages[cursor].get("role") == "tool":
                results.append(source_messages[cursor])
                cursor += 1
            if len(results) != len(calls):
                return None, "tool_call_result_count_mismatch"
            for call, result in zip(calls, results, strict=True):
                function = call.get("function") if isinstance(call, Mapping) else None
                if not isinstance(function, Mapping):
                    return None, "tool_function_schema"
                name = function.get("name")
                arguments = function.get("arguments")
                if not isinstance(name, str) or not name.strip() or not isinstance(arguments, str):
                    return None, "tool_function_schema"
                try:
                    argument_value = json.loads(arguments)
                except json.JSONDecodeError:
                    return None, "tool_arguments_json"
                result_content = result.get("content")
                if not isinstance(result_content, str) or not result_content.strip():
                    return None, "empty_tool_result"
                call_content = json.dumps(
                    {"arguments": argument_value, "name": name.strip()},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                messages.append({"role": "tool", "content": call_content})
                messages.append({"role": "tool_result", "content": result_content.strip()})
                tool_calls_seen += 1
            index = cursor
            continue
        if role == "assistant":
            if not isinstance(content, str) or not content.strip():
                return None, "empty_assistant"
            visible, discarded = _visible_assistant_content(content)
            if visible is None:
                return None, "unextractable_hidden_reasoning"
            messages.append({"role": "assistant", "content": visible})
            discarded_reasoning_blocks += int(discarded)
            index += 1
            continue
        return None, "unexpected_tool_trajectory_role"
    if tool_calls_seen < 1:
        return None, "no_tool_call"
    try:
        validate_message_sequence(messages)
    except ValueError as error:
        return None, f"normalized_sequence:{error}"
    return (
        _source_record(
            lane="nemotron_tool",
            document_id=str(row["uuid"]),
            messages=messages,
            language="en",
            task_category="tool_api",
            quality_score=0.85,
            transform="nemotron_structured_tool_to_lumi_roles_v0.1",
            extra_provenance={
                "discarded_unverified_reasoning_fields": True,
                "discarded_tagged_private_reasoning_blocks": discarded_reasoning_blocks,
                "structured_tool_calls_preserved": tool_calls_seen,
            },
        ),
        "eligible",
    )


def _aya_record(row: Mapping[str, Any]) -> tuple[str, dict[str, Any] | None, str]:
    dataset_name = row.get("dataset_name")
    lane_spec = AYA_DATASET_LANES.get(str(dataset_name))
    if lane_spec is None:
        return "aya_excluded", None, "dataset_family_reserved_or_not_selected"
    lane, task_category = lane_spec
    if row.get("language") != "ben" or row.get("script") != "Beng" or row.get("split") != "train":
        return lane, None, "language_or_split_contract"
    prompt = row.get("inputs")
    response = row.get("targets")
    if not isinstance(prompt, str) or not isinstance(response, str):
        return lane, None, "text_schema"
    prompt = prompt.strip()
    response = response.strip()
    if not prompt or not response:
        return lane, None, "empty_text"
    if len(prompt) > 20_000 or len(response) > 20_000:
        return lane, None, "character_limit"
    if normalize_for_matching(prompt) == normalize_for_matching(response):
        return lane, None, "copy_target"
    return (
        lane,
        _source_record(
            lane=lane,
            document_id=str(row["id"]),
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            language="bn",
            task_category=task_category,
            quality_score=0.80,
            transform="aya_bengali_direct_instruction_response_v0.1",
            extra_provenance={
                "source_dataset_name_sha256": hashlib.sha256(
                    str(dataset_name).encode()
                ).hexdigest(),
                "source_task_type": str(row.get("task_type")),
            },
        ),
        "eligible",
    )


def _banglatlit_record(row: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    bengali = str(row.get("text_bengali") or "").strip()
    transliterated = str(row.get("text_transliterated") or "").strip()
    if not bengali or not transliterated:
        return None, "empty_pair"
    if len(_BENGALI.findall(bengali)) < 3:
        return None, "bengali_source_signal"
    if _BENGALI.search(transliterated) or len(_LATIN.findall(transliterated)) < 3:
        return None, "transliteration_script_contract"
    if not 0.15 <= len(transliterated) / max(len(bengali), 1) <= 6.0:
        return None, "length_ratio"
    document_id = str(row.get("id") or sha256_value([bengali, transliterated]))
    prompt = f"বাংলা বাক্যটি Latin-script Banglish-এ লিখুন: {bengali}"
    return (
        _source_record(
            lane="banglatlit",
            document_id=document_id,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": transliterated},
            ],
            language="multilingual",
            task_category="multilingual_bengali",
            quality_score=0.80,
            transform="banglatlit_bengali_to_latin_auxiliary_v0.1",
            extra_provenance={"auxiliary_transliteration_only": True},
        ),
        "eligible",
    )


def _banglabridge_record(
    row: Mapping[str, Any], index: int
) -> tuple[dict[str, Any] | None, str]:
    prompt = row.get("instruction")
    response = row.get("output")
    if not isinstance(prompt, str) or not isinstance(response, str):
        return None, "text_schema"
    prompt = prompt.strip()
    response = response.strip()
    if not prompt or not response:
        return None, "empty_text"
    combined = f"{prompt}\n{response}"
    bengali = len(_BENGALI.findall(combined))
    latin = len(_LATIN.findall(combined))
    if bengali < 3 and (latin < 12 or _BANGLISH_MARKERS.search(combined) is None):
        return None, "bangla_or_banglish_signal"
    language = "bn" if bengali >= max(latin, 10) else "multilingual"
    return (
        _source_record(
            lane="banglabridge",
            document_id=f"row-{index:06d}",
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            language=language,
            task_category="multilingual_bengali",
            quality_score=0.78,
            transform="banglabridge_instruction_output_projection_v0.1",
        ),
        "eligible",
    )


def _load_reservoirs(raw_root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    import pyarrow.parquet as pq

    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = {}
    rejections: defaultdict[str, Counter[str]] = defaultdict(Counter)
    rows_seen: Counter[str] = Counter()

    chat_path = raw_root / (
        "nvidia_Nemotron-Post-Training-Dataset-v1/"
        "data/chat-00000-of-00008.parquet"
    )
    chat = pq.ParquetFile(chat_path)
    for batch in chat.iter_batches(batch_size=4096, columns=["messages"]):
        for row in batch.to_pylist():
            rows_seen["nemotron_chat"] += 1
            messages = row.get("messages") or []
            if (
                not messages
                or not isinstance(messages[0].get("content"), str)
                or not messages[0]["content"].strip()
            ):
                rejections["nemotron_chat"]["missing_user_prompt"] += 1
            else:
                rejections["nemotron_chat"]["unexpected_usable_row_not_curated"] += 1

    tool_path = raw_root / (
        "nvidia_Nemotron-Post-Training-Dataset-v1/"
        "data/tool-00000-of-00013.parquet"
    )
    for batch in pq.ParquetFile(tool_path).iter_batches(
        batch_size=1024,
        columns=["uuid", "license", "category", "messages", "metadata"],
    ):
        for row in batch.to_pylist():
            rows_seen["nemotron_tool"] += 1
            record, reason = _tool_record(row)
            if record is None:
                rejections["nemotron_tool"][reason] += 1
            else:
                _reservoir_add(heaps, "nemotron_tool", record)

    aya_path = raw_root / (
        "CohereLabs_aya_collection_language_split/"
        "bengali/train-00000-of-00003.parquet"
    )
    for batch in pq.ParquetFile(aya_path).iter_batches(
        batch_size=8192,
        columns=[
            "id",
            "inputs",
            "targets",
            "dataset_name",
            "task_type",
            "language",
            "script",
            "split",
        ],
    ):
        for row in batch.to_pylist():
            rows_seen["aya_bengali"] += 1
            lane, record, reason = _aya_record(row)
            if record is None:
                rejections["aya_bengali"][reason] += 1
            else:
                _reservoir_add(heaps, lane, record)

    tlit_path = raw_root / "aplycaebous_BanglaTLit/BanglaTLit_train.csv"
    with tlit_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "text_transliterated", "text_bengali"]:
            raise ValueError("BanglaTLit schema changed during curation")
        for row in reader:
            rows_seen["banglatlit"] += 1
            record, reason = _banglatlit_record(row)
            if record is None:
                rejections["banglatlit"][reason] += 1
            else:
                _reservoir_add(heaps, "banglatlit", record)

    bridge_path = (
        raw_root
        / "subhajitmahata84_banglabridge-instructions/banglish_finetune.adapted.jsonl"
    )
    with bridge_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            rows_seen["banglabridge"] += 1
            row = json.loads(line)
            record, reason = _banglabridge_record(row, index)
            if record is None:
                rejections["banglabridge"][reason] += 1
            else:
                _reservoir_add(heaps, "banglabridge", record)

    reservoirs = {
        lane: [item[2] for item in sorted(heap, key=lambda item: (-item[0], item[1]))]
        for lane, heap in heaps.items()
    }
    return reservoirs, {
        "reservoir_records": {lane: len(rows) for lane, rows in sorted(reservoirs.items())},
        "rejection_counts": {
            lane: dict(sorted(counter.items())) for lane, counter in sorted(rejections.items())
        },
        "rows_seen": dict(sorted(rows_seen.items())),
    }


@dataclass(frozen=True)
class MeasuredRecord:
    record: dict[str, Any]
    serialized_tokens: int
    supervised_tokens: int
    priority: int


def _measure_reservoirs(
    reservoirs: Mapping[str, Sequence[dict[str, Any]]],
    *,
    policy: Any,
    benchmark: BenchmarkRegistry,
    tokenizer: ChatTokenizer,
    template: ChatTemplate,
) -> tuple[dict[str, list[MeasuredRecord]], dict[str, Any]]:
    measured: dict[str, list[MeasuredRecord]] = {}
    rejections: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for lane, records in reservoirs.items():
        accepted: list[MeasuredRecord] = []
        for record in records:
            reasons = _custom_quality_reasons(record)
            decision = evaluate_sft_record(
                record,
                policy=policy,
                benchmark_registry=benchmark,
                source_registry=None,
            )
            reasons.extend(decision.reasons)
            if reasons:
                rejections[lane].update(dict.fromkeys(reasons, 1))
                continue
            try:
                encoded = encode_sft_messages(
                    record["messages"], tokenizer=tokenizer, template=template
                )
            except ValueError as error:
                reason = "over_4k" if "maximum is 4096" in str(error) else f"encoding:{error}"
                rejections[lane][reason] += 1
                continue
            accepted.append(
                MeasuredRecord(
                    record=record,
                    serialized_tokens=len(encoded.input_ids),
                    supervised_tokens=encoded.supervised_tokens,
                    priority=_priority(record),
                )
            )
        measured[lane] = sorted(
            accepted,
            key=lambda item: (item.priority, item.record["record_id"]),
        )
    return measured, {
        "machine_accepted": {lane: len(rows) for lane, rows in sorted(measured.items())},
        "rejection_counts": {
            lane: dict(sorted(counter.items())) for lane, counter in sorted(rejections.items())
        },
    }


def _mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


class PurePythonNearDuplicateIndex:
    """Eight-minhash candidate lookup with exact shingle-Jaccard confirmation."""

    def __init__(self, threshold: float = 0.82, shingle_tokens: int = 5) -> None:
        self.threshold = threshold
        self.shingle_tokens = shingle_tokens
        self.buckets: defaultdict[tuple[int, int], set[str]] = defaultdict(set)
        self.shingles: dict[str, frozenset[int]] = {}

    def _shingles(self, text: str) -> frozenset[int]:
        tokens = matching_tokens(text)
        if len(tokens) < self.shingle_tokens:
            windows = [" ".join(tokens)]
        else:
            windows = [
                " ".join(tokens[index : index + self.shingle_tokens])
                for index in range(len(tokens) - self.shingle_tokens + 1)
            ]
        return frozenset(
            int.from_bytes(hashlib.sha256(window.encode()).digest()[:8], "big")
            for window in windows
            if window
        )

    def _signature(self, shingles: frozenset[int]) -> tuple[int, ...]:
        if not shingles:
            return (0,) * 8
        return tuple(min(_mix64(value ^ salt) for value in shingles) for salt in range(8))

    def duplicate_of(self, record_id: str, text: str) -> tuple[str | None, str]:
        shingles = self._shingles(text)
        signature = self._signature(shingles)
        candidates: set[str] = set()
        for index, value in enumerate(signature):
            candidates.update(self.buckets[(index, value)])
        for candidate_id in sorted(candidates):
            other = self.shingles[candidate_id]
            union = len(shingles | other)
            similarity = len(shingles & other) / union if union else 1.0
            if similarity >= self.threshold:
                return candidate_id, sha256_value(signature)
        self.shingles[record_id] = shingles
        for index, value in enumerate(signature):
            self.buckets[(index, value)].add(record_id)
        return None, sha256_value(signature)


def _record_text(record: Mapping[str, Any]) -> str:
    return "\n".join(
        f"{message['role']} {normalize_for_matching(str(message['content']))}"
        for message in record["messages"]
    )


def _select_records(
    measured: Mapping[str, Sequence[MeasuredRecord]],
) -> tuple[list[MeasuredRecord], dict[str, Any]]:
    selected: list[MeasuredRecord] = []
    selected_ids: set[str] = set()
    exact_seen: dict[str, str] = {}
    near = PurePythonNearDuplicateIndex()
    dedup = Counter()
    lane_metrics: dict[str, Any] = {}
    ordered_lanes = [
        "nemotron_tool",
        "aya_simplification",
        "aya_summarization",
        "aya_generation",
        "banglabridge",
        "banglatlit",
    ]
    for lane in ordered_lanes:
        rows = list(measured.get(lane, ()))
        target_train = LANE_TARGETS.get(lane)
        target_validation = (
            math.ceil(target_train / (VALIDATION_DENOMINATOR - 1))
            if target_train is not None
            else None
        )
        tokens = Counter()
        counts = Counter()
        for split in ("train", "validation"):
            target = target_train if split == "train" else target_validation
            for item in rows:
                if item.record["split"] != split:
                    continue
                if target is not None and tokens[split] >= target:
                    break
                if (
                    target is not None
                    and tokens[split] + item.supervised_tokens > target
                ):
                    dedup["target_fit_skipped"] += 1
                    continue
                record_id = str(item.record["record_id"])
                if record_id in selected_ids:
                    dedup["duplicate_record_id"] += 1
                    continue
                exact = sha256_value(
                    [
                        {
                            "content": normalize_for_matching(message["content"]),
                            "role": message["role"],
                        }
                        for message in item.record["messages"]
                    ]
                )
                if exact in exact_seen:
                    dedup["exact_duplicate"] += 1
                    continue
                duplicate_of, _ = near.duplicate_of(record_id, _record_text(item.record))
                if duplicate_of is not None:
                    dedup["near_duplicate"] += 1
                    continue
                exact_seen[exact] = record_id
                selected_ids.add(record_id)
                selected.append(item)
                counts[split] += 1
                tokens[split] += item.supervised_tokens
        lane_metrics[lane] = {
            "records": dict(counts),
            "supervised_tokens": dict(tokens),
            "target_train_supervised_tokens": target_train,
            "target_validation_supervised_tokens": target_validation,
        }
    selected.sort(
        key=lambda item: (
            item.record["source"]["source_id"],
            item.record["split"],
            item.record["record_id"],
        )
    )
    return selected, {
        "algorithm": "eight_minhash_candidates_exact_5_shingle_jaccard_v0.1",
        "dedup_rejections": dict(sorted(dedup.items())),
        "lane_metrics": lane_metrics,
        "threshold": 0.82,
    }


def _write_candidates(
    selected: Sequence[MeasuredRecord], private_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    private_directory = private_root / "private"
    private_directory.mkdir(parents=True, exist_ok=True)
    by_source: defaultdict[str, list[MeasuredRecord]] = defaultdict(list)
    for item in selected:
        by_source[str(item.record["source"]["source_id"])].append(item)
    source_key_by_id = {
        spec["source_id"]: source_key for source_key, spec in SOURCE_SPECS.items()
    }
    inventories: list[dict[str, Any]] = []
    token_ledger: list[dict[str, Any]] = []
    for source_id, rows in sorted(by_source.items()):
        source_key = source_key_by_id[source_id]
        output = private_directory / SOURCE_FILE_NAMES[source_key]
        with output.open("w", encoding="utf-8") as handle:
            for item in rows:
                handle.write(
                    json.dumps(
                        item.record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                token_ledger.append(
                    {
                        "record_id": item.record["record_id"],
                        "serialized_tokens": item.serialized_tokens,
                        "split": item.record["split"],
                        "supervised_tokens": item.supervised_tokens,
                    }
                )
        by_split: defaultdict[str, Counter[str]] = defaultdict(Counter)
        for item in rows:
            bucket = by_split[str(item.record["split"])]
            bucket["records"] += 1
            bucket["serialized_tokens"] += item.serialized_tokens
            bucket["supervised_tokens"] += item.supervised_tokens
        inventories.append(
            {
                "bytes": output.stat().st_size,
                "file": str(output.relative_to(private_root)),
                "records": len(rows),
                "sha256": file_sha256(output),
                "source_id": source_id,
                "splits": {key: dict(value) for key, value in sorted(by_split.items())},
            }
        )
    return inventories, {
        "entries": len(token_ledger),
        "ledger_sha256": sha256_value(sorted(token_ledger, key=lambda row: row["record_id"])),
    }


def _owner_sample(selected: Sequence[MeasuredRecord]) -> list[dict[str, Any]]:
    by_source: defaultdict[str, list[MeasuredRecord]] = defaultdict(list)
    for item in selected:
        by_source[str(item.record["source"]["source_id"])].append(item)
    sample: list[dict[str, Any]] = []
    for source_id, rows in sorted(by_source.items()):
        ranked = sorted(
            rows,
            key=lambda item: hashlib.sha256(
                f"closure-owner-sample-v0.1\0{item.record['record_id']}".encode()
            ).hexdigest(),
        )
        for item in ranked[:8]:
            language = str(item.record["language"])
            if language == "bn":
                language_mode = "bengali_script"
            elif language == "en":
                language_mode = "english"
            elif source_id == SOURCE_SPECS["banglatlit"]["source_id"]:
                language_mode = "banglish"
            else:
                language_mode = "natural_code_switching"
            sample.append(
                {
                    "record_id": item.record["record_id"],
                    "skill_id": f"{source_id}:{item.record['task_category']}",
                    "language_mode": language_mode,
                    "split": item.record["split"],
                    "messages": item.record["messages"],
                }
            )
    if len(sample) < 32:
        existing = {row["record_id"] for row in sample}
        remainder = sorted(
            (item for item in selected if item.record["record_id"] not in existing),
            key=lambda item: hashlib.sha256(
                f"closure-owner-fill-v0.1\0{item.record['record_id']}".encode()
            ).hexdigest(),
        )
        for item in remainder[: 32 - len(sample)]:
            sample.append(
                {
                    "record_id": item.record["record_id"],
                    "skill_id": (
                        f"{item.record['source']['source_id']}:"
                        f"{item.record['task_category']}"
                    ),
                    "language_mode": (
                        "bengali_script" if item.record["language"] == "bn" else "english"
                    ),
                    "split": item.record["split"],
                    "messages": item.record["messages"],
                }
            )
    sample.sort(key=lambda row: row["record_id"])
    return sample[:32]


def _write_owner_review(selected: Sequence[MeasuredRecord], private_root: Path) -> dict[str, Any]:
    review_directory = private_root / "review"
    review_directory.mkdir(parents=True, exist_ok=True)
    sample = _owner_sample(selected)
    if len(sample) != 32:
        raise ValueError("closure-canary owner sample did not reach 32 rows")
    sample_sha256 = sha256_value(sample)
    sample_path = review_directory / "owner-review-sample-32.json"
    sample_path.write_text(
        json.dumps(sample, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    html = build_owner_review_html(sample, sample_sha256=sample_sha256).replace(
        (
            "These rows are AI-generated and owner-reviewed. "
            "Approving them does not make them human-written."
        ),
        (
            "These rows are source-derived and machine-curated. "
            "Approving them does not make them human-written."
        ),
    )
    html_path = review_directory / "owner-review-32.html"
    html_path.write_text(html, encoding="utf-8")
    return {
        "html_bytes": html_path.stat().st_size,
        "html_file": str(html_path.relative_to(private_root)),
        "html_sha256": file_sha256(html_path),
        "owner_review_status": "pending",
        "sample_bytes": sample_path.stat().st_size,
        "sample_file": str(sample_path.relative_to(private_root)),
        "sample_records": len(sample),
        "sample_sha256": sample_sha256,
    }


def _summarize_lengths(selected: Sequence[MeasuredRecord]) -> dict[str, Any]:
    values = [item.serialized_tokens for item in selected]
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "maximum": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "p95": ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)],
    }


def run_curation(
    *,
    repository_root: str | Path,
    raw_root: str | Path,
    private_root: str | Path,
    plan_path: str | Path,
    tokenizer_path: str | Path,
    chat_template_path: str | Path,
    data_contract_path: str | Path,
    data_quality_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    repository = Path(repository_root).resolve()
    raw = Path(raw_root).resolve()
    private = Path(private_root).resolve()
    plan = Path(plan_path).resolve()
    tokenizer_file = Path(tokenizer_path).resolve()
    if file_sha256(plan) != PLAN_SHA256:
        raise ValueError("closure-canary acquisition plan changed")
    if file_sha256(tokenizer_file) != TOKENIZER_SHA256:
        raise ValueError("closure-canary tokenizer changed")
    quality = json.loads(Path(data_quality_path).read_text(encoding="utf-8"))
    if quality["benchmark_decontamination"]["registry_sha256"] != BENCHMARK_SHA256:
        raise ValueError("closure-canary benchmark registry changed")
    import pyarrow

    if pyarrow.__version__ != PYARROW_VERSION:
        raise ValueError("closure-canary PyArrow version changed")
    tokenizer = load_direct_tokenizer(tokenizer_file)
    template = load_chat_template(chat_template_path)
    policy = load_sft_policy(data_contract_path)
    benchmark = BenchmarkRegistry(quality["benchmark_decontamination"], repository)

    raw_inventory = _verify_raw_objects(raw)
    reservoirs, source_gate = _load_reservoirs(raw)
    measured, machine_gate = _measure_reservoirs(
        reservoirs,
        policy=policy,
        benchmark=benchmark,
        tokenizer=tokenizer,
        template=template,
    )
    selected, dedup = _select_records(measured)
    inventories, token_ledger = _write_candidates(selected, private)
    owner_review = _write_owner_review(selected, private)

    by_split = defaultdict(Counter)
    source_tokens = Counter()
    auxiliary_train_tokens = 0
    for item in selected:
        split = str(item.record["split"])
        by_split[split]["records"] += 1
        by_split[split]["serialized_tokens"] += item.serialized_tokens
        by_split[split]["supervised_tokens"] += item.supervised_tokens
        if split == "train":
            source_id = str(item.record["source"]["source_id"])
            source_tokens[source_id] += item.supervised_tokens
            if item.record["provenance"].get("auxiliary_transliteration_only") is True:
                auxiliary_train_tokens += item.supervised_tokens
    train_tokens = by_split["train"]["supervised_tokens"]
    validation_tokens = by_split["validation"]["supervised_tokens"]
    general_train_tokens = train_tokens - auxiliary_train_tokens
    projected_total = CURRENT_ACCEPTED_TRAIN_TOKENS + train_tokens
    if train_tokens > MAXIMUM_TRAIN_TOKENS:
        raise ValueError("closure-canary train token ceiling exceeded")
    if general_train_tokens > MAXIMUM_GENERAL_TRAIN_TOKENS:
        raise ValueError("closure-canary general train token ceiling exceeded")
    if auxiliary_train_tokens > MAXIMUM_AUXILIARY_TRAIN_TOKENS:
        raise ValueError("closure-canary auxiliary token ceiling exceeded")
    raw_bytes = sum(item["bytes"] for item in raw_inventory)

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "private_nonproduction_candidates_owner_review_pending",
        "candidate_files": inventories,
        "owner_review": owner_review,
        "token_ledger": token_ledger,
    }
    manifest["manifest_sha256"] = sha256_value(manifest)
    manifest_path = private / "private-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    private_bytes = sum(
        path.stat().st_size for path in private.rglob("*") if path.is_file()
    )
    if private_bytes + raw_bytes > MAXIMUM_PRIVATE_WRITES:
        raise ValueError("closure-canary private write ceiling exceeded")

    blockers = [
        "owner_review_pending",
        "candidate_registry_promotion_not_authorized",
        "production_dataset_seal_not_authorized",
        "sft_training_not_authorized",
    ]
    if source_gate["rejection_counts"].get("nemotron_chat", {}).get(
        "missing_user_prompt"
    ) == source_gate["rows_seen"].get("nemotron_chat"):
        blockers.append("nemotron_chat_lane_unusable_all_user_prompts_missing")
    if projected_total < TEN_MILLION_TOKENS:
        blockers.append("projected_train_tokens_remain_below_10m")
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "curated_private_nonproduction_owner_review_pending",
        "authorization": {
            "acquisition_and_curation_plan_sha256": PLAN_SHA256,
            "cpu_only": True,
            "dataset_seal_authorized": False,
            "gpu_authorized": False,
            "registry_change_authorized": False,
            "sft_training_authorized": False,
        },
        "bindings": {
            "benchmark_registry_sha256": benchmark.sha256,
            "chat_template_sha256": file_sha256(chat_template_path),
            "data_contract_sha256": file_sha256(data_contract_path),
            "data_quality_sha256": file_sha256(data_quality_path),
            "private_manifest_file_sha256": file_sha256(manifest_path),
            "private_manifest_internal_sha256": manifest["manifest_sha256"],
            "pyarrow_version": PYARROW_VERSION,
            "pyarrow_wheel_sha256": PYARROW_WHEEL_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
        },
        "blockers": blockers,
        "candidate": {
            "by_split": {key: dict(value) for key, value in sorted(by_split.items())},
            "general_train_supervised_tokens": general_train_tokens,
            "auxiliary_train_supervised_tokens": auxiliary_train_tokens,
            "records": len(selected),
            "serialized_length": _summarize_lengths(selected),
            "source_train_supervised_tokens": dict(sorted(source_tokens.items())),
            "train_supervised_tokens": train_tokens,
            "validation_supervised_tokens": validation_tokens,
        },
        "closure": {
            "current_accepted_train_supervised_tokens": CURRENT_ACCEPTED_TRAIN_TOKENS,
            "gap_before_canary": TEN_MILLION_TOKENS - CURRENT_ACCEPTED_TRAIN_TOKENS,
            "projected_train_supervised_tokens": projected_total,
            "ten_million_floor_reached_by_projection": projected_total >= TEN_MILLION_TOKENS,
        },
        "deduplication": dedup,
        "machine_gate": machine_gate,
        "owner_review": owner_review,
        "private_writes": {
            "candidate_and_evidence_bytes": private_bytes,
            "raw_source_bytes": raw_bytes,
            "total_bytes": private_bytes + raw_bytes,
        },
        "raw_training_text_in_report": False,
        "raw_object_inventory": raw_inventory,
        "runtime_seconds": round(time.monotonic() - started, 3),
        "source_gate": source_gate,
    }
    report["report_sha256"] = sha256_value(report)
    return report, manifest


def write_report(path: str | Path, report: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "BENCHMARK_SHA256",
    "MANIFEST_SCHEMA_VERSION",
    "PLAN_SHA256",
    "PYARROW_VERSION",
    "PYARROW_WHEEL_SHA256",
    "SCHEMA_VERSION",
    "TOKENIZER_SHA256",
    "PurePythonNearDuplicateIndex",
    "canonical_json_bytes",
    "file_sha256",
    "load_direct_tokenizer",
    "run_curation",
    "sha256_value",
    "write_report",
]
