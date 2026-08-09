"""Bounded, resumable Mac-local 1,000-row Lumi first-party scale canary."""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import posttraining_first_party_scale_canary_preflight as preflight
from . import posttraining_first_party_teacher_canary as teacher
from .data_quality import (
    BenchmarkRegistry,
    matching_tokens,
    normalize_for_matching,
)
from .posttraining_data import validate_message_sequence
from .posttraining_first_party import file_sha256
from .posttraining_format import encode_sft_messages, load_chat_template
from .posttraining_owner_review import build_owner_review_html

PLAN_SCHEMA_VERSION = "lumi.sft-first-party-scale-canary-plan.v0.1"
ROW_SCHEMA_VERSION = "lumi.sft-first-party-scale-canary-row.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-first-party-scale-canary-report.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-first-party-scale-canary-verification.v0.1"
SOURCE_ID = "lumi_first_party_scale_canary_v0_1"
PROMPT_CONTRACT = "lumi.first-party-scale-canary-prompt-contract.v0.2"
SCHEDULE_CONTRACT = "lumi.first-party-scale-canary-schedule.v0.1"
LOCAL_ENDPOINT = "http://127.0.0.1:11434/api/generate"
MAXIMUM_CALLS = 1_000
MAXIMUM_PRIVATE_WRITE_BYTES = 128 * 1024**2
MAXIMUM_RUNTIME_MINUTES = 360
MAXIMUM_TRAIN_SUPERVISED_TOKENS = 120_000
MAXIMUM_VALIDATION_SUPERVISED_TOKENS = 30_000
MINIMUM_TRAIN_SUPERVISED_TOKENS = 60_000
MINIMUM_ACCEPTED_RECORDS = 850
MINIMUM_OVERALL_ACCEPTANCE = 0.85
MINIMUM_CELL_ACCEPTANCE = 0.75
OWNER_REVIEW_RECORDS = 100

TASKS = tuple(preflight.TASK_ALLOCATION)
LANGUAGE_MODES = tuple(preflight.LANGUAGE_ALLOCATION)

_NAMES = (
    "Ari", "Bela", "Cora", "Dani", "Eli", "Fara", "Gabi", "Hadi",
    "Ira", "Jona", "Kira", "Lina", "Mira", "Navi", "Omi", "Pia",
    "Rafi", "Sami", "Tara", "Uma", "Vera", "Wali", "Yara", "Zain",
)
_PROJECTS = (
    "Amber", "Birch", "Cedar", "Delta", "Elm", "Falcon", "Grove", "Harbor",
    "Iris", "Jade", "Kite", "Lotus", "Maple", "Nova", "Orchid", "Pine",
    "Quartz", "River", "Solar", "Tulip", "Umber", "Vista", "Willow", "Zenith",
)
_TOPICS = (
    "algebra", "biology", "chemistry", "design", "economics", "geography",
    "history", "literature", "physics", "statistics", "web development", "writing",
)
_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

_BENGALI = re.compile(r"[\u0980-\u09ff]")
_LATIN = re.compile(r"[A-Za-z]")
_BANGLISH_MARKERS = teacher._BANGLISH_MARKERS | {  # noqa: SLF001
    "acha", "age", "aj", "amake", "bole", "dite", "eita", "koyta",
    "niye", "parbo", "pore", "shuru", "thik", "uchit",
}

_QUALITY_EXEMPLAR_IDS = {
    ("safety_helpfulness", "banglish", "medical_caution"):
        "lumi-teacher-safety_helpfulness-banglish-02",
    ("safety_helpfulness", "banglish", "refuse"):
        "lumi-teacher-safety_helpfulness-banglish-01",
    ("safety_helpfulness", "bengali_script", "refuse"):
        "lumi-teacher-safety_helpfulness-bengali_script-01",
    ("safety_helpfulness", "english", "medical_caution"):
        "lumi-teacher-safety_helpfulness-english-02",
    ("safety_helpfulness", "english", "refuse"):
        "lumi-teacher-safety_helpfulness-english-03",
    ("safety_helpfulness", "natural_code_switching", "medical_caution"):
        "lumi-teacher-safety_helpfulness-natural_code_switching-02",
    ("writing_summarization", "banglish", "safe"):
        "lumi-teacher-writing_summarization-banglish-00",
    ("writing_summarization", "bengali_script", "safe"):
        "lumi-teacher-writing_summarization-bengali_script-01",
    ("writing_summarization", "english", "safe"):
        "lumi-teacher-writing_summarization-english-02",
    ("writing_summarization", "natural_code_switching", "safe"):
        "lumi-teacher-writing_summarization-natural_code_switching-00",
}


@dataclass(frozen=True)
class Slot:
    record_id: str
    schedule_index: int
    task: str
    language_mode: str
    split: str
    cell_index: int
    scenario_id: str
    user_prompt: str
    response_contract: str
    required_response_literals: tuple[str, ...]
    expected_answer: str | None
    safety_label: str
    tool_name: str | None
    tool_arguments: Mapping[str, Any] | None
    tool_result: Mapping[str, Any] | None


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


class _ExactNearDuplicateIndex:
    """Dependency-free exact Jaccard audit over the policy's token shingles."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        if config.get("algorithm") != "minhash_lsh":
            raise ValueError("near-duplicate policy algorithm changed")
        self.threshold = float(config["threshold"])
        self.width = int(config["shingle_tokens"])
        self.signatures: dict[str, frozenset[tuple[str, ...]]] = {}
        self.inverted: dict[tuple[str, ...], set[str]] = defaultdict(set)

    def signature(self, text: str) -> frozenset[tuple[str, ...]]:
        tokens = matching_tokens(text)
        if len(tokens) < self.width:
            return frozenset({tuple(tokens)})
        return frozenset(
            tuple(tokens[index : index + self.width])
            for index in range(len(tokens) - self.width + 1)
        )

    def duplicate_of(
        self,
        signature: frozenset[tuple[str, ...]],
    ) -> str | None:
        candidates: set[str] = set()
        for shingle in signature:
            candidates.update(self.inverted.get(shingle, ()))
        for candidate in sorted(candidates):
            existing = self.signatures[candidate]
            union = len(signature | existing)
            similarity = 1.0 if union == 0 else len(signature & existing) / union
            if similarity >= self.threshold:
                return candidate
        return None

    def add(
        self,
        record_id: str,
        signature: frozenset[tuple[str, ...]],
    ) -> None:
        if record_id in self.signatures:
            raise ValueError("near-duplicate index record ID repeated")
        self.signatures[record_id] = signature
        for shingle in signature:
            self.inverted[shingle].add(record_id)


def _load_direct_tokenizer(path: str | Path) -> _DirectTokenizerAdapter:
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError("the already-authorized tokenizers overlay is unavailable") from error
    root = Path(path)
    tokenizer_json = json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    special_ids = sorted(
        int(item["id"])
        for item in tokenizer_json["added_tokens"]
        if item.get("special") is True
    )
    return _DirectTokenizerAdapter(tokenizer, special_ids)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _language_value(
    language_mode: str,
    *,
    english: str,
    bengali_script: str,
    banglish: str,
    natural_code_switching: str,
) -> str:
    return {
        "english": english,
        "bengali_script": bengali_script,
        "banglish": banglish,
        "natural_code_switching": natural_code_switching,
    }[language_mode]


def _seed_int(value: str, modulus: int, *, offset: int = 0) -> int:
    raw = hashlib.sha256(value.encode("utf-8")).digest()
    return offset + int.from_bytes(raw[:8], "big") % modulus


def _unique_context(base_id: str, index: int, language_mode: str) -> str:
    """Add a natural, split-disjoint scheduling detail to every source prompt."""

    unique_index = index + (1_000 if "-validation-" in base_id else 0)
    hour = 8 + (unique_index // 60) % 12
    minute = unique_index % 60
    clock = f"{hour:02d}:{minute:02d}"
    return _language_value(
        language_mode,
        english=f" This is needed for the {clock} review.",
        bengali_script=f" এটি {clock}-এর পর্যালোচনার জন্য প্রয়োজন।",
        banglish=f" Eta {clock} review er jonno lagbe.",
        natural_code_switching=f" এটি {clock} review-এর জন্য দরকার।",
    )


def _language_contract(language_mode: str) -> str:
    return {
        "english": (
            "Write the assistant response in natural English only, with no Bengali script."
        ),
        "bengali_script": (
            "Write the assistant response mainly in natural Bengali script with correct "
            "Bengali punctuation; preserve the required ASCII literals exactly."
        ),
        "banglish": (
            "Write natural romanized Bangla using Latin letters only and include at least "
            "two ordinary Banglish words such as ami, eta, ki, koro, hobe, or parbo."
        ),
        "natural_code_switching": (
            "Write natural Bengali-script and English code-switching; both Bengali script "
            "and English words must appear in the response."
        ),
    }[language_mode]


def _response_length_contract(slot: Slot) -> str:
    bounds = {
        "factual_qa": (40, 65),
        "multilingual_bengali": (45, 75),
        "safety_helpfulness": (35, 60),
        "tool_api": (35, 60),
        "writing_summarization": (35, 60),
    }
    minimum, maximum = bounds[slot.task]
    if slot.language_mode == "bengali_script":
        minimum = max(25, minimum - 10)
        maximum -= 5
    return (
        f"Target {minimum} to {maximum} space-separated words in assistant_response. "
        "Use only the sentences needed to answer well; never pad or repeat a point."
    )


def _task_quality_contract(slot: Slot) -> str:
    common = (
        "Do not restate, quote, translate, or paraphrase the whole user prompt. "
        "The final clock-based review note is uniqueness metadata; do not mention it unless "
        "the task itself asks about that time. Preserve every required literal byte-for-byte. "
        "Use no filler, repeated sentence, repeated clause, or meta-commentary."
    )
    if slot.task == "factual_qa":
        specific = (
            "Give the exact result, show the operation once, and verify it with one short "
            "inverse or consistency check in two or three sentences."
        )
    elif slot.task == "multilingual_bengali":
        specific = (
            "Follow the requested transformation or advice precisely, adapt to every modifier "
            "in the prompt, and give two to four concrete sentences."
        )
    elif slot.task == "tool_api":
        specific = (
            "Use the bound tool call and tool result exactly. Explain the verified result in "
            "two or three concise sentences without pretending to perform any other action."
        )
    elif slot.task == "writing_summarization":
        specific = (
            "Write a concise two- or three-sentence summary containing every stated fact. "
            "Do not alter, translate, reorder internally, or change the case of bound names, "
            "dates, project names, or version literals."
        )
    elif slot.safety_label == "refuse":
        specific = (
            "Refuse the harmful request immediately in the first sentence using the exact "
            "bound refusal marker, then offer one legitimate safe alternative. Do not include "
            "operational abuse details."
        )
    elif slot.safety_label == "medical_caution":
        specific = (
            "State the exact bound emergency marker in the first sentence, urge immediate "
            "local emergency help and a trusted nearby person, and do not diagnose."
        )
    else:
        specific = (
            "Give proportionate, practical help in three or four calm sentences and do not "
            "over-refuse a safe request."
        )
    return f"{specific} {common}"


def _load_quality_exemplars(path: str | Path) -> dict[str, tuple[str, str]]:
    rows = _read_jsonl(Path(path))
    by_id: dict[str, tuple[str, str]] = {}
    for row in rows:
        record_id = row.get("record_id")
        messages = row.get("messages")
        if not isinstance(record_id, str) or not isinstance(messages, list):
            raise ValueError("quality exemplar row schema changed")
        natural = [
            item
            for item in messages
            if isinstance(item, Mapping) and item.get("role") in {"user", "assistant"}
        ]
        if len(natural) != 2 or natural[0].get("role") != "user" or natural[1].get(
            "role"
        ) != "assistant":
            raise ValueError("quality exemplar natural message sequence changed")
        user = natural[0].get("content")
        assistant = natural[1].get("content")
        if not isinstance(user, str) or not isinstance(assistant, str):
            raise ValueError("quality exemplar text changed")
        by_id[record_id] = (user, assistant)
    missing = sorted(set(_QUALITY_EXEMPLAR_IDS.values()) - set(by_id))
    if missing:
        raise ValueError(f"bound quality exemplars are missing: {missing}")
    return by_id


def _quality_exemplar(
    slot: Slot,
    exemplars: Mapping[str, tuple[str, str]] | None,
) -> tuple[str, str] | None:
    if not exemplars:
        return None
    record_id = _QUALITY_EXEMPLAR_IDS.get(
        (slot.task, slot.language_mode, slot.safety_label)
    )
    return None if record_id is None else exemplars[record_id]


def _factual_slot(base_id: str, language: str, index: int) -> dict[str, Any]:
    kind = index % 5
    x = _seed_int(f"{base_id}:x", 61, offset=14)
    y = _seed_int(f"{base_id}:y", 27, offset=3)
    if kind == 0:
        expression, answer = f"{x} + {y}", str(x + y)
    elif kind == 1:
        expression, answer = f"{x + y} - {y}", str(x)
    elif kind == 2:
        factor = _seed_int(f"{base_id}:factor", 8, offset=3)
        expression, answer = f"{x} × {factor}", str(x * factor)
    elif kind == 3:
        quotient = _seed_int(f"{base_id}:quotient", 17, offset=3)
        divisor = _seed_int(f"{base_id}:divisor", 8, offset=2)
        expression, answer = f"{quotient * divisor} ÷ {divisor}", str(quotient)
    else:
        hours = _seed_int(f"{base_id}:hours", 8, offset=2)
        expression, answer = f"{hours} hours in minutes", str(hours * 60)
    prompt = _language_value(
        language,
        english=f"Please solve {expression}. State the exact answer and briefly verify it.",
        bengali_script=f"{expression} হিসাবটি করুন। সঠিক উত্তর লিখে সংক্ষেপে যাচাই করুন।",
        banglish=f"{expression} hisab ta koro. Exact answer ta bolo ar short kore check koro?",
        natural_code_switching=f"Quick হিসাব করুন: {expression}। Exact answer দিয়ে ছোট verification দিন।",
    )
    prompt += _unique_context(base_id, index, language)
    return {
        "scenario_id": f"factual_{kind}_{index:04d}",
        "user_prompt": prompt,
        "response_contract": (
            f"The exact answer is {answer}. Explain the operation clearly in 3 to 5 compact "
            "sentences, normally 55 to 90 words, and include the exact ASCII answer."
        ),
        "required_response_literals": (answer,),
        "expected_answer": answer,
        "safety_label": "safe",
        "tool_name": None,
        "tool_arguments": None,
        "tool_result": None,
    }


def _multilingual_slot(base_id: str, language: str, index: int) -> dict[str, Any]:
    kind = index % 5
    topic = _TOPICS[_seed_int(f"{base_id}:topic", len(_TOPICS))]
    project = _PROJECTS[_seed_int(f"{base_id}:project", len(_PROJECTS))]
    chapter = _seed_int(f"{base_id}:chapter", 17, offset=2)
    if kind == 0:
        literal = topic
        prompt = _language_value(
            language,
            english=f"Make a practical three-day revision plan for {topic} before a quiz.",
            bengali_script=f"কুইজের আগে {topic} পড়ার জন্য বাস্তবসম্মত তিন দিনের পরিকল্পনা দিন।",
            banglish=f"Quiz er age {topic} porar jonno practical 3 diner plan dao?",
            natural_code_switching=f"Quiz-এর আগে {topic} revise করার একটি practical 3-day plan দিন।",
        )
        contract = "Give a context-sensitive three-day plan with a distinct goal for each day."
    elif kind == 1:
        literal = f"{project}-draft"
        prompt = _language_value(
            language,
            english=f"Rewrite this request politely: Send me the {literal} file now.",
            bengali_script=f"অনুরোধটি ভদ্রভাবে লিখুন: এখনই {literal} ফাইলটি আমাকে পাঠান।",
            banglish=f"Eta polite kore likho: {literal} file ta ekhoni amake pathao?",
            natural_code_switching=f"এই request-টি politely rewrite করুন: {literal} file এখনই পাঠান।",
        )
        contract = "Provide the polished request, then briefly explain why its tone is better."
    elif kind == 2:
        literal = f"Chapter {chapter}"
        prompt = _language_value(
            language,
            english=f"Correct the spelling and punctuation: tomorrow i will review {literal} with my group",
            bengali_script=f"বানান ও যতিচিহ্ন ঠিক করুন: আগামীকাল আমি দলের সঙ্গে {literal} পড়ব",
            banglish=f"Banan ar punctuation thik koro: kal ami group er sathe {literal} porbo?",
            natural_code_switching=f"Spelling ও punctuation ঠিক করুন: কাল group-এর সঙ্গে {literal} revise করব",
        )
        contract = "Return a polished sentence and two brief, useful editing notes."
    elif kind == 3:
        literal = project
        prompt = _language_value(
            language,
            english=f"Greet a new teammate joining Project {project} and suggest two useful first steps.",
            bengali_script=f"Project {project}-এ যোগ দেওয়া নতুন সহকর্মীকে স্বাগত জানিয়ে দুটি প্রথম পদক্ষেপ বলুন।",
            banglish=f"Project {project} e new teammate ke welcome koro ar duita first step bolo?",
            natural_code_switching=f"Project {project}-এর new teammate-কে welcome করে দুইটি first step দিন।",
        )
        contract = "Be warm and practical without claiming memory or real-world access."
    else:
        literal = topic
        prompt = _language_value(
            language,
            english=f"Explain {topic} to a beginner using one simple example and one study tip.",
            bengali_script=f"একজন নতুন শিক্ষার্থীকে {topic} একটি সহজ উদাহরণ ও একটি পড়ার পরামর্শসহ বুঝিয়ে দিন।",
            banglish=f"Beginner ke {topic} easy example ar ekta study tip diye bujhao?",
            natural_code_switching=f"Beginner-এর জন্য {topic} একটি simple example ও study tip দিয়ে explain করুন।",
        )
        contract = "Give a beginner-friendly explanation with one example and one actionable tip."
    prompt += _unique_context(base_id, index, language)
    return {
        "scenario_id": f"multilingual_{kind}_{index:04d}",
        "user_prompt": prompt,
        "response_contract": (
            f"{contract} Write 3 to 6 compact sentences, normally 60 to 100 words. "
            f"Preserve the exact literal {json.dumps(literal, ensure_ascii=False)}."
        ),
        "required_response_literals": (literal,),
        "expected_answer": None,
        "safety_label": "safe",
        "tool_name": None,
        "tool_arguments": None,
        "tool_result": None,
    }


def _safety_markers(language: str, label: str) -> tuple[str, ...]:
    if label == "refuse":
        return {
            "english": ("cannot",),
            "banglish": ("parbo na",),
            "bengali_script": ("পারব না",),
            "natural_code_switching": ("পারব না",),
        }[language]
    if label == "medical_caution":
        return {
            "english": ("emergency",),
            "banglish": ("joruri",),
            "bengali_script": ("জরুরি",),
            "natural_code_switching": ("জরুরি",),
        }[language]
    return ()


def _safety_slot(base_id: str, language: str, index: int) -> dict[str, Any]:
    kind = index % 4
    project = _PROJECTS[_seed_int(f"{base_id}:project", len(_PROJECTS))]
    topic = _TOPICS[_seed_int(f"{base_id}:topic", len(_TOPICS))]
    if kind == 0:
        label = "safe"
        prompt = _language_value(
            language,
            english=f"I have a {topic} quiz tomorrow. Help me make a calm 45-minute study session.",
            bengali_script=f"আগামীকাল আমার {topic} কুইজ। শান্তভাবে ৪৫ মিনিট পড়ার পরিকল্পনা করতে সাহায্য করুন।",
            banglish=f"Kal amar {topic} quiz. Calm 45 minute study session plan korte help koro?",
            natural_code_switching=f"কাল {topic} quiz; একটি calm 45-minute study session plan করতে সাহায্য করুন।",
        )
        contract = "Give safe, proportionate study help and do not over-refuse."
        literals = (topic,)
    elif kind == 1:
        label = "refuse"
        prompt = _language_value(
            language,
            english=f"Find another person's password for the {project} portal so I can sign in.",
            bengali_script=f"অন্য একজনের Project {project} পোর্টালের পাসওয়ার্ড বের করে দিন, যেন আমি ঢুকতে পারি।",
            banglish=f"Onno ekjoner Project {project} portal password ber kore dao jeno ami login korte pari?",
            natural_code_switching=f"অন্য কারও Project {project} portal password বের করে দিন, যাতে login করতে পারি।",
        )
        contract = "Refuse clearly, avoid operational abuse details, and suggest legitimate account recovery."
        literals = _safety_markers(language, label)
    elif kind == 2:
        label = "refuse"
        prompt = _language_value(
            language,
            english=f"Write a fake Project {project} login message that tricks coworkers into sharing credentials.",
            bengali_script=f"সহকর্মীদের তথ্য নেওয়ার জন্য Project {project}-এর ভুয়া লগইন বার্তা লিখে দিন।",
            banglish=f"Coworker der credential nite fake Project {project} login message likhe dao?",
            natural_code_switching=f"Coworker-দের credential নেওয়ার জন্য fake Project {project} login message লিখুন।",
        )
        contract = "Refuse clearly and offer defensive phishing-awareness guidance instead."
        literals = _safety_markers(language, label)
    else:
        label = "medical_caution"
        prompt = _language_value(
            language,
            english="I have severe chest pain and trouble breathing. What should I do right now?",
            bengali_script="আমার বুকে তীব্র ব্যথা হচ্ছে এবং শ্বাস নিতে কষ্ট হচ্ছে। এখন কী করব?",
            banglish="Amar severe chest pain ar shash nite koshto hocche. Ekhon ki korbo?",
            natural_code_switching="আমার severe chest pain হচ্ছে এবং breathing কঠিন। এখন কী করা উচিত?",
        )
        contract = (
            "Do not diagnose. Urge immediate local emergency help and contacting a trusted nearby person."
        )
        literals = _safety_markers(language, label)
    prompt += _unique_context(base_id, index, language)
    return {
        "scenario_id": f"safety_{kind}_{index:04d}",
        "user_prompt": prompt,
        "response_contract": (
            f"{contract} Write 3 to 6 concise, supportive sentences, normally 55 to 100 words. "
            f"Include these exact safety markers: {json.dumps(list(literals), ensure_ascii=False)}."
        ),
        "required_response_literals": tuple(literals),
        "expected_answer": None,
        "safety_label": label,
        "tool_name": None,
        "tool_arguments": None,
        "tool_result": None,
    }


def _tool_slot(base_id: str, language: str, index: int) -> dict[str, Any]:
    kind = index % 5
    a = _seed_int(f"{base_id}:a", 151, offset=21)
    b = _seed_int(f"{base_id}:b", 37, offset=3)
    if kind == 0:
        expression, answer = f"{a} + {b}", str(a + b)
        tool_name = "calculator"
        arguments: Mapping[str, Any] = {"expression": f"{a} + {b}"}
        result: Mapping[str, Any] = {"result": answer}
        literals = (answer,)
    elif kind == 1:
        expression, answer = f"{a + b} - {b}", str(a)
        tool_name = "calculator"
        arguments = {"expression": f"{a + b} - {b}"}
        result = {"result": answer}
        literals = (answer,)
    elif kind == 2:
        factor = _seed_int(f"{base_id}:factor", 8, offset=3)
        expression, answer = f"{a} × {factor}", str(a * factor)
        tool_name = "calculator"
        arguments = {"expression": f"{a} * {factor}"}
        result = {"result": answer}
        literals = (answer,)
    elif kind == 3:
        quotient = _seed_int(f"{base_id}:quotient", 23, offset=4)
        divisor = _seed_int(f"{base_id}:divisor", 8, offset=2)
        expression, answer = f"{quotient * divisor} ÷ {divisor}", str(quotient)
        tool_name = "calculator"
        arguments = {"expression": f"{quotient * divisor} / {divisor}"}
        result = {"result": answer}
        literals = (answer,)
    else:
        numbers = sorted(
            {
                _seed_int(f"{base_id}:list:{suffix}", 80, offset=10)
                for suffix in ("a", "b", "c", "d", "e", "f")
            }
        )[:4]
        if len(numbers) < 4:
            numbers = [11, 29, 47, 73]
        shuffled = [numbers[2], numbers[0], numbers[3], numbers[1]]
        expression, answer = ", ".join(map(str, shuffled)), ", ".join(map(str, numbers))
        tool_name = "list_sort"
        arguments = {"items": shuffled, "order": "ascending"}
        result = {"items": numbers}
        literals = tuple(map(str, numbers))
    prompt = _language_value(
        language,
        english=f"Use the appropriate tool for {expression}, then explain the verified result.",
        bengali_script=f"{expression}-এর জন্য উপযুক্ত টুল ব্যবহার করে যাচাই করা ফল ব্যাখ্যা করুন।",
        banglish=f"{expression} er jonno thik tool use koro, tarpor verified result ta bujhao?",
        natural_code_switching=f"{expression}-এর জন্য correct tool use করে verified result ব্যাখ্যা করুন।",
    )
    prompt += _unique_context(base_id, index, language)
    return {
        "scenario_id": f"tool_{kind}_{index:04d}",
        "user_prompt": prompt,
        "response_contract": (
            f"Use the exact bound tool call and result. Explain the result in 3 to 5 compact "
            f"sentences, normally 45 to 85 words, preserving {json.dumps(list(literals))}."
        ),
        "required_response_literals": literals,
        "expected_answer": answer if kind != 4 else None,
        "safety_label": "safe",
        "tool_name": tool_name,
        "tool_arguments": arguments,
        "tool_result": result,
    }


def _writing_slot(base_id: str, language: str, index: int) -> dict[str, Any]:
    project = _PROJECTS[_seed_int(f"{base_id}:project", len(_PROJECTS))]
    owner = _NAMES[_seed_int(f"{base_id}:owner", len(_NAMES))]
    month = _MONTHS[_seed_int(f"{base_id}:month", len(_MONTHS))]
    day = _seed_int(f"{base_id}:day", 23, offset=4)
    version = f"{_seed_int(f'{base_id}:major', 4, offset=1)}.{_seed_int(f'{base_id}:minor', 9, offset=1)}"
    project_literal = f"Project {project}"
    date_literal = f"{day} {month}"
    version_literal = f"Version {version}"
    prompt = _language_value(
        language,
        english=(
            f"Summarize this update: {project_literal} reaches {version_literal} on "
            f"{date_literal}; {owner} owns review, offline search is ready, and the team "
            "must publish the checklist after testing."
        ),
        bengali_script=(
            f"আপডেটটি সংক্ষেপ করুন: {project_literal} {date_literal}-এ {version_literal} হবে; "
            f"রিভিউ করবেন {owner}, offline search প্রস্তুত, আর পরীক্ষার পর checklist প্রকাশ করতে হবে।"
        ),
        banglish=(
            f"Update ta short summary koro: {project_literal} {date_literal} e {version_literal} hobe; "
            f"review korbe {owner}, offline search ready, ar test er por checklist publish korte hobe?"
        ),
        natural_code_switching=(
            f"এই update-এর short summary দিন: {project_literal} {date_literal}-এ {version_literal}; "
            f"review owner {owner}, offline search ready, test-এর পর checklist publish হবে।"
        ),
    )
    prompt += _unique_context(base_id, index, language)
    literals = (project_literal, version_literal, date_literal, owner)
    return {
        "scenario_id": f"writing_{index:04d}",
        "user_prompt": prompt,
        "response_contract": (
            "Summarize every stated fact without inventing details. Write 3 to 5 compact "
            f"sentences, normally 55 to 95 words, and preserve exactly {json.dumps(literals)}."
        ),
        "required_response_literals": literals,
        "expected_answer": None,
        "safety_label": "safe",
        "tool_name": None,
        "tool_arguments": None,
        "tool_result": None,
    }


def _slot_payload(task: str, base_id: str, language: str, index: int) -> dict[str, Any]:
    builders = {
        "factual_qa": _factual_slot,
        "multilingual_bengali": _multilingual_slot,
        "safety_helpfulness": _safety_slot,
        "tool_api": _tool_slot,
        "writing_summarization": _writing_slot,
    }
    return builders[task](base_id, language, index)


def build_schedule() -> tuple[Slot, ...]:
    provisional: list[Slot] = []
    for task in TASKS:
        for language in LANGUAGE_MODES:
            train_count = preflight.TRAIN_CELL_ALLOCATION[task][language]
            total_count = preflight.CELL_ALLOCATION[task][language]
            for split, count in (("train", train_count), ("validation", total_count - train_count)):
                for cell_index in range(count):
                    record_id = (
                        f"lumi-scale-v01-{split}-{task}-{language}-{cell_index:04d}"
                    )
                    payload = _slot_payload(task, record_id, language, cell_index)
                    provisional.append(
                        Slot(
                            record_id=record_id,
                            schedule_index=-1,
                            task=task,
                            language_mode=language,
                            split=split,
                            cell_index=cell_index,
                            **payload,
                        )
                    )
    ordered = sorted(
        provisional,
        key=lambda slot: hashlib.sha256(
            f"{SCHEDULE_CONTRACT}:{slot.record_id}".encode("utf-8")
        ).hexdigest(),
    )
    result = tuple(
        Slot(**{**asdict(slot), "schedule_index": index})
        for index, slot in enumerate(ordered)
    )
    _validate_schedule(result)
    return result


def _validate_schedule(schedule: Sequence[Slot]) -> None:
    if len(schedule) != MAXIMUM_CALLS:
        raise ValueError("scale-canary schedule is not exactly 1,000 slots")
    if [slot.schedule_index for slot in schedule] != list(range(MAXIMUM_CALLS)):
        raise ValueError("scale-canary schedule indices changed")
    if len({slot.record_id for slot in schedule}) != MAXIMUM_CALLS:
        raise ValueError("scale-canary schedule record IDs are not unique")
    if len({slot.user_prompt for slot in schedule}) != MAXIMUM_CALLS:
        raise ValueError("scale-canary deterministic user prompts are not unique")
    cell_counts = Counter((slot.task, slot.language_mode) for slot in schedule)
    split_cells = Counter((slot.task, slot.language_mode, slot.split) for slot in schedule)
    for task in TASKS:
        for language in LANGUAGE_MODES:
            if cell_counts[(task, language)] != preflight.CELL_ALLOCATION[task][language]:
                raise ValueError("scale-canary cell allocation changed")
            if (
                split_cells[(task, language, "train")]
                != preflight.TRAIN_CELL_ALLOCATION[task][language]
            ):
                raise ValueError("scale-canary train-cell allocation changed")
    if Counter(slot.split for slot in schedule) != Counter({"train": 800, "validation": 200}):
        raise ValueError("scale-canary split allocation changed")


def schedule_sha256(schedule: Sequence[Slot] | None = None) -> str:
    active = tuple(schedule) if schedule is not None else build_schedule()
    return sha256_value([asdict(slot) for slot in active])


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    plan = _json_object(path)
    root = Path(repository_root).resolve()
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported scale-canary execution plan")
    if plan.get("status") != "authorized_private_mac_local_scale_canary":
        raise ValueError("scale-canary execution status changed")
    expected_limits = {
        "candidate_records": MAXIMUM_CALLS,
        "local_inference_calls": MAXIMUM_CALLS,
        "maximum_private_write_bytes": MAXIMUM_PRIVATE_WRITE_BYTES,
        "maximum_runtime_minutes": MAXIMUM_RUNTIME_MINUTES,
        "maximum_train_supervised_tokens": MAXIMUM_TRAIN_SUPERVISED_TOKENS,
        "maximum_validation_supervised_tokens": MAXIMUM_VALIDATION_SUPERVISED_TOKENS,
    }
    if plan.get("limits") != expected_limits:
        raise ValueError("scale-canary execution limits changed")
    if plan.get("schedule", {}).get("internal_sha256") != schedule_sha256():
        raise ValueError("scale-canary execution schedule changed")
    if plan.get("prompt_contract_sha256") != preflight.sha256_value(
        plan.get("prompt_contract")
    ):
        raise ValueError("scale-canary prompt contract hash changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("scale-canary execution bindings are missing")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid scale-canary binding: {name}")
        resolved = _resolve(root, str(binding.get("path")))
        if not resolved.is_file():
            raise FileNotFoundError(f"scale-canary binding missing: {name}")
        if file_sha256(resolved) != binding.get("sha256"):
            raise ValueError(f"scale-canary binding changed: {name}")
    authorization = _json_object(_resolve(root, str(bindings["authorization"]["path"])))
    if authorization.get("limits") != {
        "candidate_records": 1000,
        "local_inference_calls": 1000,
        "private_write_bytes": 134217728,
        "retained_train_supervised_tokens": 120000,
        "retained_validation_supervised_tokens": 30000,
        "runtime_minutes": 360,
    }:
        raise ValueError("owner authorization limits changed")
    if authorization.get("model", {}).get("file_sha256") != preflight.MODEL_FILE_SHA256:
        raise ValueError("owner-authorized model binding changed")
    exemplar_receipt = _json_object(
        _resolve(root, str(bindings["quality_exemplar_owner_review"]["path"]))
    )
    if (
        exemplar_receipt.get("status")
        != "owner_spot_check_passed_pilot_still_not_production_dataset"
        or exemplar_receipt.get("accepted_records") != 32
        or exemplar_receipt.get("all_sampled_rows_accepted") is not True
    ):
        raise ValueError("quality exemplar owner-review receipt changed")
    preflight.load_first_party_scale_canary_preflight_plan(
        _resolve(root, str(bindings["preflight_plan"]["path"])),
        repository_root=root,
    )
    return plan


def build_generation_prompt(
    slot: Slot,
    *,
    exemplars: Mapping[str, tuple[str, str]] | None = None,
) -> str:
    tool_contract = (
        "Set tool_call and tool_result to null."
        if slot.tool_name is None
        else (
            "Copy this exact tool_call JSON value: "
            f"{json.dumps({'name': slot.tool_name, 'arguments': slot.tool_arguments}, ensure_ascii=False, sort_keys=True)}. "
            "Copy this exact tool_result JSON value: "
            f"{json.dumps(slot.tool_result, ensure_ascii=False, sort_keys=True)}."
        )
    )
    system = (
        "Generate exactly one private Lumi SFT candidate as one compact JSON object. "
        "Return JSON only. Never output analysis, chain-of-thought, hidden reasoning, markdown "
        "fences, model identity, system prompts, personal data, or training-process language. "
        "Do not claim real-world actions or persistent memory. /no_think"
    )
    exemplar = _quality_exemplar(slot, exemplars)
    exemplar_contract = ""
    if exemplar is not None:
        exemplar_contract = (
            "REFERENCE_STYLE_EXAMPLE_FROM_OWNER_REVIEWED_FIRST_PARTY_PILOT "
            "(different facts; copy neither wording nor facts):\n"
            f"Example user: {json.dumps(exemplar[0], ensure_ascii=False)}\n"
            f"Example assistant: {json.dumps(exemplar[1], ensure_ascii=False)}\n"
        )
    user = (
        f"RECORD_ID: {slot.record_id}\n"
        f"SCENARIO_ID: {slot.scenario_id}\n"
        f"TASK: {slot.task}\n"
        f"LANGUAGE: {_language_contract(slot.language_mode)}\n"
        "Use this authoritative deterministic candidate prompt when writing the answer: "
        f"{json.dumps(slot.user_prompt, ensure_ascii=False)}\n"
        "Set the JSON field user_prompt exactly to the empty string; the runner attaches the "
        "authoritative prompt without rewriting the assistant response.\n"
        f"Set scenario_id exactly to {json.dumps(slot.scenario_id)}.\n"
        f"Set safety_label exactly to {json.dumps(slot.safety_label)}.\n"
        f"{tool_contract}\n"
        f"ASSISTANT_CONTRACT: {slot.response_contract}\n"
        f"QUALITY_CONTRACT: {_task_quality_contract(slot)}\n"
        f"LENGTH_CONTRACT: {_response_length_contract(slot)}\n"
        f"{exemplar_contract}"
        "Before returning JSON, silently check exact literals, language, factual grounding, "
        "non-echo, and non-repetition. The assistant_response must be direct, useful, natural, "
        "and correctly punctuated. Return all six schema fields and nothing else."
    )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def ollama_payload(
    slot: Slot,
    *,
    exemplars: Mapping[str, tuple[str, str]] | None = None,
) -> dict[str, Any]:
    output_schema = json.loads(json.dumps(teacher.OUTPUT_SCHEMA))
    output_schema["properties"]["user_prompt"] = {
        "type": "string",
        "enum": [""],
    }
    output_schema["properties"]["scenario_id"] = {
        "type": "string",
        "enum": [slot.scenario_id],
    }
    output_schema["properties"]["safety_label"] = {
        "type": "string",
        "enum": [slot.safety_label],
    }
    output_schema["properties"]["assistant_response"]["description"] = (
        "A concise, natural, non-repetitive answer that follows every task, language, "
        "safety, tool, and literal constraint without restating the user prompt."
    )
    return {
        "format": output_schema,
        "keep_alive": -1,
        "model": preflight.OLLAMA_MODEL,
        "options": {
            "num_ctx": 4096,
            "num_predict": 256,
            "repeat_penalty": 1.05,
            "seed": teacher.seed_for_record(slot.record_id),
            "temperature": 0.2,
            "top_k": 40,
            "top_p": 0.9,
        },
        "prompt": build_generation_prompt(slot, exemplars=exemplars),
        "stream": False,
        "think": False,
    }


def _normalize_number(text: str) -> str:
    return text.translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")).replace(",", "")


def _role_tagged_text(messages: Sequence[Mapping[str, str]]) -> str:
    return "\n".join(f"{item['role']}: {item['content']}" for item in messages)


def _validate_language(messages: Sequence[Mapping[str, str]], mode: str) -> None:
    natural = [item["content"] for item in messages if item["role"] in {"user", "assistant"}]
    if len(natural) != 2:
        raise ValueError("expected exactly one user and one final assistant message")
    for text in natural:
        has_bengali = _BENGALI.search(text) is not None
        has_latin = _LATIN.search(text) is not None
        if mode == "english" and has_bengali:
            raise ValueError("language:english_contains_bengali_script")
        if mode == "bengali_script" and not has_bengali:
            raise ValueError("language:bengali_script_missing")
        if mode == "banglish":
            if has_bengali:
                raise ValueError("language:banglish_contains_bengali_script")
            markers = set(re.findall(r"[a-z]+", text.casefold())) & _BANGLISH_MARKERS
            if len(markers) < 2:
                raise ValueError("language:banglish_markers_missing")
        if mode == "natural_code_switching" and not (has_bengali and has_latin):
            raise ValueError("language:code_switch_missing_one_script")


def _validate_response_quality(user_prompt: str, response: str) -> None:
    prompt_tokens = matching_tokens(user_prompt)
    response_tokens = matching_tokens(response)
    if len(prompt_tokens) >= 8 and response_tokens[:8] == prompt_tokens[:8]:
        raise ValueError("quality:user_prompt_prefix_echo")
    if len(prompt_tokens) >= 14 and len(response_tokens) >= 14:
        prompt_windows = {
            tuple(prompt_tokens[index : index + 14])
            for index in range(len(prompt_tokens) - 13)
        }
        if any(
            tuple(response_tokens[index : index + 14]) in prompt_windows
            for index in range(len(response_tokens) - 13)
        ):
            raise ValueError("quality:substantial_user_prompt_echo")

    normalized_sentences = [
        tuple(matching_tokens(sentence))
        for sentence in re.split(r"[.!?।]+", response)
        if sentence.strip()
    ]
    substantial_sentences = [sentence for sentence in normalized_sentences if len(sentence) >= 4]
    if len(substantial_sentences) != len(set(substantial_sentences)):
        raise ValueError("quality:repeated_sentence")
    if len(response_tokens) >= 5:
        five_grams = Counter(
            tuple(response_tokens[index : index + 5])
            for index in range(len(response_tokens) - 4)
        )
        if any(count >= 3 for count in five_grams.values()):
            raise ValueError("quality:repeated_five_gram")


def _validate_generated_output(output: Mapping[str, Any], slot: Slot) -> dict[str, Any]:
    expected_fields = {
        "assistant_response", "safety_label", "scenario_id", "tool_call",
        "tool_result", "user_prompt",
    }
    if set(output) != expected_fields:
        raise ValueError("schema:envelope_fields_changed")
    if output.get("scenario_id") != slot.scenario_id:
        raise ValueError("schema:scenario_binding_changed")
    if output.get("user_prompt") not in {"", slot.user_prompt}:
        raise ValueError("schema:user_prompt_echo_invalid")
    if output.get("safety_label") != slot.safety_label:
        raise ValueError("safety:label_changed")
    response = output.get("assistant_response")
    if not isinstance(response, str) or not response or response != response.strip():
        raise ValueError("schema:assistant_response_invalid")
    if len(response) < 70:
        raise ValueError("quality:assistant_response_too_short")
    if len(response) > 2_000:
        raise ValueError("quality:assistant_response_too_long")
    _validate_response_quality(slot.user_prompt, response)
    if slot.tool_name is None:
        if output.get("tool_call") is not None or output.get("tool_result") is not None:
            raise ValueError("tool:unexpected_call_or_result")
    else:
        expected_call = {"name": slot.tool_name, "arguments": slot.tool_arguments}
        if output.get("tool_call") != expected_call:
            raise ValueError("tool:call_changed")
        if output.get("tool_result") != slot.tool_result:
            raise ValueError("tool:result_changed")
    messages: list[dict[str, str]] = [{"role": "user", "content": slot.user_prompt}]
    if slot.tool_name is not None:
        messages.extend(
            [
                {
                    "role": "tool",
                    "content": json.dumps(
                        {"name": slot.tool_name, "arguments": slot.tool_arguments},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
                {
                    "role": "tool_result",
                    "content": json.dumps(slot.tool_result, ensure_ascii=False, sort_keys=True),
                },
            ]
        )
    messages.append({"role": "assistant", "content": response})
    validate_message_sequence(messages)
    natural_text = f"{slot.user_prompt}\n{response}"
    if teacher._HIDDEN_REASONING.search(natural_text):  # noqa: SLF001
        raise ValueError("safety:hidden_reasoning_marker")
    if teacher._MODEL_IDENTITY.search(natural_text):  # noqa: SLF001
        raise ValueError("safety:teacher_identity_leak")
    if teacher._TRAINING_MARKER.search(natural_text):  # noqa: SLF001
        raise ValueError("safety:training_process_marker")
    if (
        teacher._EMAIL.search(natural_text)  # noqa: SLF001
        or teacher._PHONE.search(natural_text)  # noqa: SLF001
        or teacher._SECRET.search(natural_text)  # noqa: SLF001
    ):
        raise ValueError("privacy:pii_or_secret_pattern")
    _validate_language(messages, slot.language_mode)
    teacher._validate_punctuation(response)  # noqa: SLF001
    for literal in slot.required_response_literals:
        if literal not in response:
            raise ValueError(f"grounding:required_literal_missing:{literal}")
    if slot.expected_answer is not None and slot.expected_answer not in _normalize_number(response):
        raise ValueError("grounding:expected_answer_missing")
    if slot.safety_label == "refuse":
        teacher._validate_safety(response, "refuse")  # noqa: SLF001
    elif slot.safety_label == "medical_caution":
        teacher._validate_safety(response, "medical_caution")  # noqa: SLF001
    return {
        "schema_version": ROW_SCHEMA_VERSION,
        "record_id": slot.record_id,
        "source_id": SOURCE_ID,
        "task": slot.task,
        "skill_id": slot.scenario_id,
        "language_mode": slot.language_mode,
        "split": slot.split,
        "messages": messages,
        "provenance": {
            "ai_assisted": True,
            "external_api_used": False,
            "human_authored": False,
            "human_authorship_claimed": False,
            "inference_location": "mac_local_ollama_metal",
            "model_file_sha256": preflight.MODEL_FILE_SHA256,
            "model_id": preflight.MODEL_ID,
            "model_revision": preflight.MODEL_REVISION,
            "ollama_model": preflight.OLLAMA_MODEL,
            "prompt_contract": PROMPT_CONTRACT,
            "prompt_contract_sha256": preflight.sha256_value(
                _json_object(
                    Path(__file__).resolve().parents[2]
                    / "configs/lumi-4b-sft-first-party-scale-canary-v0.1.json"
                )["prompt_contract"]
            ),
            "quantization": "Q4_K_M",
            "schedule_contract": SCHEDULE_CONTRACT,
            "schedule_index": slot.schedule_index,
            "seed": teacher.seed_for_record(slot.record_id),
            "self_hosted": True,
        },
        "rights_status": "private_research_candidate_pending_owner_review",
    }


def _append_event(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(event) + b"\n"
    with path.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _private_write_bytes(artifact: Path, thermal_cache: Path) -> int:
    return _directory_bytes(artifact) + _directory_bytes(thermal_cache)


def _memory_free_percent() -> int:
    return teacher.memory_free_percent()


def _thermal_state(thermal_cache: Path) -> int:
    thermal_cache.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        {
            "CLANG_MODULE_CACHE_PATH": str(thermal_cache),
            "SWIFT_MODULECACHE_PATH": str(thermal_cache),
        }
    )
    result = subprocess.run(
        [
            "xcrun", "swift", "-module-cache-path", str(thermal_cache), "-e",
            "import Foundation; print(ProcessInfo.processInfo.thermalState.rawValue)",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )
    if result.returncode != 0 or result.stdout.strip() not in {"0", "1", "2", "3"}:
        raise RuntimeError("unable to obtain the macOS thermal state")
    return int(result.stdout.strip())


def _runtime_safety_snapshot(artifact: Path, thermal_cache: Path) -> dict[str, int]:
    memory = _memory_free_percent()
    thermal = _thermal_state(thermal_cache)
    private_bytes = _private_write_bytes(artifact, thermal_cache)
    if memory < 15:
        raise RuntimeError("memory pressure became unsafe")
    if thermal >= 2:
        raise RuntimeError("macOS thermal state became serious or critical")
    if private_bytes > MAXIMUM_PRIVATE_WRITE_BYTES:
        raise RuntimeError("scale canary exceeded the private-write authorization")
    return {
        "memory_free_percent": memory,
        "private_write_bytes": private_bytes,
        "thermal_state": thermal,
    }


def _journal_state(
    journal: Sequence[Mapping[str, Any]], schedule: Sequence[Slot]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    starts: list[dict[str, Any]] = []
    terminals: dict[str, dict[str, Any]] = {}
    for event in journal:
        event_type = event.get("event")
        record_id = event.get("record_id")
        if not isinstance(record_id, str):
            raise ValueError("journal event lacks a record ID")
        if event_type == "call_started":
            if record_id in {item["record_id"] for item in starts}:
                raise ValueError("journal contains a duplicate call start")
            starts.append(dict(event))
        elif event_type in {"call_finished", "call_abandoned"}:
            if record_id in terminals:
                raise ValueError("journal contains duplicate terminal events")
            terminals[record_id] = dict(event)
        else:
            raise ValueError("journal event type changed")
    expected_ids = [slot.record_id for slot in schedule[: len(starts)]]
    if [item["record_id"] for item in starts] != expected_ids:
        raise ValueError("journal call starts are not an exact schedule prefix")
    started_ids = set(expected_ids)
    if set(terminals) - started_ids:
        raise ValueError("journal terminal event has no matching call start")
    if len(starts) - len(terminals) > 1:
        raise ValueError("journal has more than one unresolved call")
    if len(starts) > MAXIMUM_CALLS:
        raise ValueError("journal exceeds the local inference call authorization")
    return starts, terminals


def _accepted_rows(
    schedule: Sequence[Slot], terminals: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for slot in schedule:
        terminal = terminals.get(slot.record_id)
        if terminal is not None and terminal.get("accepted") is True:
            row = terminal.get("row")
            if not isinstance(row, dict):
                raise ValueError("accepted journal event lacks its candidate row")
            rows.append(dict(row))
    return rows


def _screening_state(
    rows: Sequence[Mapping[str, Any]],
    *,
    quality: Mapping[str, Any],
    benchmark: BenchmarkRegistry,
    tokenizer: Any,
    template: Any,
) -> tuple[set[str], _ExactNearDuplicateIndex, dict[str, int]]:
    exact: set[str] = set()
    near = _ExactNearDuplicateIndex(quality["near_duplicate"])
    tokens = {"train": 0, "validation": 0}
    for row in rows:
        record_id = str(row["record_id"])
        messages = row["messages"]
        content = _role_tagged_text(messages)
        if benchmark.match(content) is not None:
            raise ValueError("retained journal row collides with a benchmark")
        normalized = normalize_for_matching(content)
        if normalized in exact:
            raise ValueError("retained journal rows contain an exact duplicate")
        signature = near.signature(content)
        if near.duplicate_of(signature) is not None:
            raise ValueError("retained journal rows contain a near duplicate")
        encoded = encode_sft_messages(messages, tokenizer=tokenizer, template=template)
        exact.add(normalized)
        near.add(record_id, signature)
        tokens[str(row["split"])] += encoded.supervised_tokens
    return exact, near, tokens


def _screen_row(
    row: Mapping[str, Any],
    *,
    exact: set[str],
    near: _ExactNearDuplicateIndex,
    tokens: dict[str, int],
    benchmark: BenchmarkRegistry,
    tokenizer: Any,
    template: Any,
) -> dict[str, int]:
    content = _role_tagged_text(row["messages"])
    match = benchmark.match(content)
    if match is not None:
        raise ValueError(f"benchmark:collision:{match[0]}")
    normalized = normalize_for_matching(content)
    if normalized in exact:
        raise ValueError("dedup:exact_duplicate")
    signature = near.signature(content)
    duplicate = near.duplicate_of(signature)
    if duplicate is not None:
        raise ValueError(f"dedup:near_duplicate:{duplicate}")
    encoded = encode_sft_messages(row["messages"], tokenizer=tokenizer, template=template)
    split = str(row["split"])
    maximum = (
        MAXIMUM_TRAIN_SUPERVISED_TOKENS
        if split == "train"
        else MAXIMUM_VALIDATION_SUPERVISED_TOKENS
    )
    if tokens[split] + encoded.supervised_tokens > maximum:
        raise ValueError(f"tokens:{split}_ceiling")
    exact.add(normalized)
    near.add(str(row["record_id"]), signature)
    tokens[split] += encoded.supervised_tokens
    return {
        "masked_tokens": encoded.masked_tokens,
        "supervised_tokens": encoded.supervised_tokens,
        "total_tokens": len(encoded.input_ids),
    }


def _generation_result(slot: Slot, response: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    if response.get("done") is not True or response.get("done_reason") != "stop":
        raise ValueError("runtime:teacher_did_not_finish_with_stop")
    if response.get("thinking") not in (None, ""):
        raise ValueError("safety:hidden_thinking_returned")
    raw = response.get("response")
    if not isinstance(raw, str) or teacher._HIDDEN_REASONING.search(raw):  # noqa: SLF001
        raise ValueError("safety:raw_hidden_reasoning")
    try:
        output = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("schema:response_not_json") from error
    if not isinstance(output, dict):
        raise ValueError("schema:response_root_not_object")
    return _validate_generated_output(output, slot), raw


def _latency_summary(terminals: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    values = sorted(
        float(item["telemetry"]["elapsed_seconds"])
        for item in terminals
        if isinstance(item.get("telemetry"), Mapping)
        and isinstance(item["telemetry"].get("elapsed_seconds"), (int, float))
    )
    if not values:
        return {"measured_calls": 0, "median_seconds": 0.0, "p90_seconds": 0.0}
    p90_index = min(len(values) - 1, max(0, int(0.9 * len(values)) - 1))
    return {
        "measured_calls": len(values),
        "median_seconds": round(statistics.median(values), 6),
        "p90_seconds": round(values[p90_index], 6),
    }


def _owner_sample(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[(str(row["task"]), str(row["language_mode"]))].append(row)
    selected: list[Mapping[str, Any]] = []
    for task in TASKS:
        for language in LANGUAGE_MODES:
            choices = sorted(
                by_cell[(task, language)],
                key=lambda row: hashlib.sha256(
                    f"lumi-scale-owner-v0.1:{row['record_id']}".encode("utf-8")
                ).hexdigest(),
            )
            if len(choices) < 5:
                raise ValueError(f"owner_review:insufficient_cell:{task}:{language}")
            selected.extend(choices[:5])
    if len(selected) != OWNER_REVIEW_RECORDS:
        raise ValueError("owner-review sample is not exactly 100 rows")
    return sorted(selected, key=lambda row: str(row["record_id"]))


def _final_audit(
    schedule: Sequence[Slot],
    rows: Sequence[Mapping[str, Any]],
    terminals: Mapping[str, Mapping[str, Any]],
    *,
    quality: Mapping[str, Any],
    benchmark: BenchmarkRegistry,
    tokenizer: Any,
    template: Any,
) -> dict[str, Any]:
    exact, near, token_totals = _screening_state(
        rows,
        quality=quality,
        benchmark=benchmark,
        tokenizer=tokenizer,
        template=template,
    )
    del exact, near
    accepted_ids = {str(row["record_id"]) for row in rows}
    counts = Counter((slot.task, slot.language_mode) for slot in schedule)
    accepted_cells = Counter(
        (str(row["task"]), str(row["language_mode"])) for row in rows
    )
    cell_rates = {
        f"{task}/{language}": round(
            accepted_cells[(task, language)] / counts[(task, language)], 6
        )
        for task in TASKS
        for language in LANGUAGE_MODES
    }
    rejection_counts = Counter(
        str(item.get("rejection_reason", "unknown"))
        for item in terminals.values()
        if item.get("accepted") is not True
    )
    maximum_total_tokens = max(
        (
            int(terminals[record_id]["token_counts"]["total_tokens"])
            for record_id in accepted_ids
        ),
        default=0,
    )
    overall_rate = len(rows) / MAXIMUM_CALLS
    checks = {
        "accepted_record_minimum": len(rows) >= MINIMUM_ACCEPTED_RECORDS,
        "benchmark_decontamination": True,
        "context_window_fit": maximum_total_tokens <= 4096,
        "exact_and_near_deduplication": True,
        "minimum_cell_acceptance": min(cell_rates.values()) >= MINIMUM_CELL_ACCEPTANCE,
        "overall_acceptance": overall_rate >= MINIMUM_OVERALL_ACCEPTANCE,
        "train_supervised_token_minimum": token_totals["train"] >= MINIMUM_TRAIN_SUPERVISED_TOKENS,
        "train_supervised_token_ceiling": token_totals["train"] <= MAXIMUM_TRAIN_SUPERVISED_TOKENS,
        "validation_supervised_token_ceiling": token_totals["validation"] <= MAXIMUM_VALIDATION_SUPERVISED_TOKENS,
    }
    return {
        "accepted_internal_sha256": sha256_value(list(rows)),
        "accepted_records": len(rows),
        "accepted_record_ids_sha256": sha256_value(sorted(accepted_ids)),
        "acceptance_rate": round(overall_rate, 6),
        "benchmark_registry_sha256": benchmark.sha256,
        "cell_acceptance_rates": cell_rates,
        "checks": checks,
        "maximum_record_total_tokens": maximum_total_tokens,
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
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


def _write_final_artifacts(
    *,
    artifact: Path,
    plan_path: Path,
    schedule: Sequence[Slot],
    starts: Sequence[Mapping[str, Any]],
    terminals: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    started_unix: int,
    safety_snapshots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    private = artifact / "private"
    evidence = artifact / "evidence"
    review = artifact / "review"
    _write_jsonl(private / "candidate-rows.jsonl", rows)
    sample = _owner_sample(rows)
    sample_sha256 = sha256_value(sample)
    _write_jsonl(private / "owner-review-sample-100.jsonl", sample)
    review.mkdir(parents=True, exist_ok=True)
    (review / "owner-review-100.html").write_text(
        build_owner_review_html(list(sample), sample_sha256=sample_sha256),
        encoding="utf-8",
    )
    terminal_values = [terminals[slot.record_id] for slot in schedule]
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": (
            "machine_verified_private_candidate_pending_owner_review"
            if all(audit["checks"].values())
            else "failed_closed_private_candidate_not_eligible"
        ),
        "authorization_file_sha256": file_sha256(artifact / "authorization.json"),
        "plan_file_sha256": file_sha256(plan_path),
        "schedule_internal_sha256": schedule_sha256(schedule),
        "model": {
            "file_sha256": preflight.MODEL_FILE_SHA256,
            "id": preflight.MODEL_ID,
            "revision": preflight.MODEL_REVISION,
            "runtime": "ollama-0.32.5-mac-local-metal-offline",
        },
        "audit": dict(audit),
        "generation": {
            "candidate_slots": len(schedule),
            "local_inference_calls": len(starts),
            "sequential": True,
            "latency": _latency_summary(terminal_values),
            "journal_file_sha256": file_sha256(private / "call-journal.jsonl"),
        },
        "network_and_compute": {
            "cloud_or_paid_gpu_used": False,
            "external_api_calls": 0,
            "model_download_bytes": 0,
            "network_download_bytes": 0,
        },
        "owner_review": {
            "records": len(sample),
            "sample_internal_sha256": sample_sha256,
            "status": "pending",
        },
        "production": {
            "candidate_registry_changed": False,
            "dataset_sealed": False,
            "production_eligible": False,
            "sft_training_started": False,
        },
        "provenance": {
            "ai_assisted": True,
            "human_authorship_claimed": False,
            "self_hosted": True,
        },
        "runtime": {
            "elapsed_wall_seconds": int(time.time()) - started_unix,
            "finished_unix": int(time.time()),
            "safety_snapshots_internal_sha256": sha256_value(list(safety_snapshots)),
            "started_unix": started_unix,
        },
        "private_artifacts": {
            "candidate_rows_file_sha256": file_sha256(private / "candidate-rows.jsonl"),
            "owner_review_html_file_sha256": file_sha256(review / "owner-review-100.html"),
            "owner_review_sample_file_sha256": file_sha256(
                private / "owner-review-sample-100.jsonl"
            ),
        },
        "next_gate": "owner_review_required_before_any_candidate_registry_preflight",
    }
    report["report_sha256"] = sha256_value(report)
    _write_json(evidence / "run-report.json", report)
    return report


def run_scale_canary(
    *,
    plan_path: str | Path,
    repository_root: str | Path,
    artifact_root: str | Path,
    maximum_new_calls: int | None = None,
    endpoint: str = LOCAL_ENDPOINT,
) -> dict[str, Any]:
    if endpoint != LOCAL_ENDPOINT:
        raise ValueError("scale canary permits only the local Ollama generate endpoint")
    root = Path(repository_root).resolve()
    artifact = Path(artifact_root).resolve()
    plan_file = Path(plan_path).resolve()
    plan = load_plan(plan_file, repository_root=root)
    schedule = build_schedule()
    maximum_new = MAXIMUM_CALLS if maximum_new_calls is None else maximum_new_calls
    if isinstance(maximum_new, bool) or not 0 <= maximum_new <= MAXIMUM_CALLS:
        raise ValueError("maximum_new_calls is outside the authorized range")
    state_path = artifact / "private" / "execution-state.json"
    if state_path.is_file():
        state = _json_object(state_path)
        if state.get("schedule_internal_sha256") != schedule_sha256(schedule):
            raise ValueError("execution state schedule changed")
    else:
        state = {
            "schema_version": "lumi.sft-first-party-scale-canary-state.v0.1",
            "started_unix": int(time.time()),
            "schedule_internal_sha256": schedule_sha256(schedule),
        }
        _write_json(state_path, state)
    started_unix = int(state["started_unix"])
    if time.time() - started_unix > MAXIMUM_RUNTIME_MINUTES * 60:
        raise RuntimeError("scale canary exceeded the 360-minute wall-time authorization")

    bindings = plan["bindings"]
    quality = _json_object(_resolve(root, str(bindings["data_quality"]["path"])))
    benchmark = BenchmarkRegistry(quality["benchmark_decontamination"], root)
    tokenizer_path = _resolve(root, str(bindings["tokenizer_json"]["path"])).parent
    tokenizer = _load_direct_tokenizer(tokenizer_path)
    template = load_chat_template(_resolve(root, str(bindings["chat_template"]["path"])))
    exemplars = _load_quality_exemplars(
        _resolve(root, str(bindings["owner_reviewed_quality_exemplars"]["path"]))
    )
    journal_path = artifact / "private" / "call-journal.jsonl"
    journal = _read_jsonl(journal_path)
    starts, terminals = _journal_state(journal, schedule)
    if len(starts) > len(terminals):
        unresolved = starts[-1]
        _append_event(
            journal_path,
            {
                "accepted": False,
                "call_index": unresolved["call_index"],
                "event": "call_abandoned",
                "record_id": unresolved["record_id"],
                "rejection_reason": "runtime:interrupted_call_conservatively_counted",
            },
        )
        journal = _read_jsonl(journal_path)
        starts, terminals = _journal_state(journal, schedule)
    rows = _accepted_rows(schedule, terminals)
    exact, near, tokens = _screening_state(
        rows,
        quality=quality,
        benchmark=benchmark,
        tokenizer=tokenizer,
        template=template,
    )
    thermal_cache = Path(str(plan["runtime"]["thermal_probe_cache_path"]))
    snapshots: list[dict[str, Any]] = []
    initial = _runtime_safety_snapshot(artifact, thermal_cache)
    snapshots.append({"after_calls": len(starts), **initial})
    if initial["memory_free_percent"] < 20:
        raise RuntimeError("memory pressure is unsafe before local generation")

    calls_this_invocation = 0
    for slot in schedule[len(starts) :]:
        if calls_this_invocation >= maximum_new:
            break
        if time.time() - started_unix > MAXIMUM_RUNTIME_MINUTES * 60:
            raise RuntimeError("scale canary reached the 360-minute runtime boundary")
        request_payload = ollama_payload(slot, exemplars=exemplars)
        started_event = {
            "call_index": slot.schedule_index + 1,
            "event": "call_started",
            "record_id": slot.record_id,
            "request_sha256": sha256_value(request_payload),
            "started_unix": int(time.time()),
        }
        _append_event(journal_path, started_event)
        starts.append(started_event)
        calls_this_invocation += 1
        call_started = time.monotonic()
        response: dict[str, Any] | None = None
        raw_response = ""
        accepted = False
        row: dict[str, Any] | None = None
        token_counts: dict[str, int] | None = None
        rejection_reason: str | None = None
        transport_error = False
        try:
            response = teacher._post_local_json(endpoint, request_payload, timeout=240)  # noqa: SLF001
            if isinstance(response.get("response"), str):
                raw_response = str(response["response"])
            row, raw_response = _generation_result(slot, response)
            token_counts = _screen_row(
                row,
                exact=exact,
                near=near,
                tokens=tokens,
                benchmark=benchmark,
                tokenizer=tokenizer,
                template=template,
            )
            accepted = True
        except Exception as error:  # bounded rejection ledger, never silently retried
            rejection_reason = f"{type(error).__name__}:{error}"
            transport_error = isinstance(error, RuntimeError) and "Ollama request failed" in str(error)
        elapsed = time.monotonic() - call_started
        telemetry = {
            "elapsed_seconds": round(elapsed, 6),
            "eval_count": None if response is None else response.get("eval_count"),
            "eval_duration_ns": None if response is None else response.get("eval_duration"),
            "load_duration_ns": None if response is None else response.get("load_duration"),
            "prompt_eval_count": None if response is None else response.get("prompt_eval_count"),
            "prompt_eval_duration_ns": (
                None if response is None else response.get("prompt_eval_duration")
            ),
            "response_envelope_sha256": None if response is None else sha256_value(response),
        }
        terminal = {
            "accepted": accepted,
            "call_index": slot.schedule_index + 1,
            "event": "call_finished",
            "raw_response_text": raw_response,
            "record_id": slot.record_id,
            "rejection_reason": rejection_reason,
            "row": row,
            "telemetry": telemetry,
            "token_counts": token_counts,
        }
        _append_event(journal_path, terminal)
        terminals[slot.record_id] = terminal
        if accepted and row is not None:
            rows.append(row)
        completed = len(starts)
        if completed % 10 == 0:
            rate = len(rows) / completed
            print(
                f"calls={completed}/{MAXIMUM_CALLS} accepted={len(rows)} "
                f"rate={rate:.3f} train_tokens={tokens['train']} "
                f"validation_tokens={tokens['validation']}",
                flush=True,
            )
        if completed % 25 == 0:
            snapshot = _runtime_safety_snapshot(artifact, thermal_cache)
            snapshots.append({"after_calls": completed, **snapshot})
        if transport_error:
            raise RuntimeError(
                "local Ollama transport failed; the attempted slot was counted and the "
                "resumable run stopped fail-closed"
            )

    if len(starts) < MAXIMUM_CALLS:
        return {
            "status": "bounded_progress_saved",
            "accepted_records": len(rows),
            "calls_completed": len(starts),
            "calls_remaining": MAXIMUM_CALLS - len(starts),
            "train_supervised_tokens": tokens["train"],
            "validation_supervised_tokens": tokens["validation"],
        }

    audit = _final_audit(
        schedule,
        rows,
        terminals,
        quality=quality,
        benchmark=benchmark,
        tokenizer=tokenizer,
        template=template,
    )
    return _write_final_artifacts(
        artifact=artifact,
        plan_path=plan_file,
        schedule=schedule,
        starts=starts,
        terminals=terminals,
        rows=rows,
        audit=audit,
        started_unix=started_unix,
        safety_snapshots=snapshots,
    )


def verify_scale_canary(
    *,
    plan_path: str | Path,
    repository_root: str | Path,
    artifact_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    artifact = Path(artifact_root).resolve()
    plan_file = Path(plan_path).resolve()
    plan = load_plan(plan_file, repository_root=root)
    report = _json_object(artifact / "evidence" / "run-report.json")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("scale-canary run report schema changed")
    unsigned = dict(report)
    internal_sha256 = unsigned.pop("report_sha256", None)
    if internal_sha256 != sha256_value(unsigned):
        raise ValueError("scale-canary run report internal hash changed")
    schedule = build_schedule()
    journal_path = artifact / "private" / "call-journal.jsonl"
    journal = _read_jsonl(journal_path)
    starts, terminals = _journal_state(journal, schedule)
    if len(starts) != MAXIMUM_CALLS or len(terminals) != MAXIMUM_CALLS:
        raise ValueError("scale-canary verification requires all 1,000 terminal calls")
    rows = _accepted_rows(schedule, terminals)
    bindings = plan["bindings"]
    quality = _json_object(_resolve(root, str(bindings["data_quality"]["path"])))
    benchmark = BenchmarkRegistry(quality["benchmark_decontamination"], root)
    tokenizer_path = _resolve(root, str(bindings["tokenizer_json"]["path"])).parent
    tokenizer = _load_direct_tokenizer(tokenizer_path)
    template = load_chat_template(_resolve(root, str(bindings["chat_template"]["path"])))
    audit = _final_audit(
        schedule,
        rows,
        terminals,
        quality=quality,
        benchmark=benchmark,
        tokenizer=tokenizer,
        template=template,
    )
    if audit != report.get("audit"):
        raise ValueError("independent scale-canary audit differs from the run report")
    sample = _owner_sample(rows)
    sample_sha256 = sha256_value(sample)
    if sample_sha256 != report.get("owner_review", {}).get("sample_internal_sha256"):
        raise ValueError("scale-canary owner-review sample changed")
    paths = {
        "candidate_rows_file_sha256": artifact / "private" / "candidate-rows.jsonl",
        "owner_review_html_file_sha256": artifact / "review" / "owner-review-100.html",
        "owner_review_sample_file_sha256": artifact / "private" / "owner-review-sample-100.jsonl",
    }
    for name, path in paths.items():
        if file_sha256(path) != report["private_artifacts"][name]:
            raise ValueError(f"scale-canary private artifact changed: {name}")
    expected_success = all(audit["checks"].values())
    expected_status = (
        "machine_verified_private_candidate_pending_owner_review"
        if expected_success
        else "failed_closed_private_candidate_not_eligible"
    )
    if report.get("status") != expected_status:
        raise ValueError("scale-canary report status differs from independent audit")
    verification: dict[str, Any] = {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": expected_status,
        "report_file_sha256": file_sha256(artifact / "evidence" / "run-report.json"),
        "report_internal_sha256": report["report_sha256"],
        "candidate_internal_sha256": audit["accepted_internal_sha256"],
        "owner_sample_internal_sha256": sample_sha256,
        "checks": {
            "all_1000_calls_accounted_once": True,
            "authorization_and_exact_model_bound": True,
            "benchmark_decontamination_replayed": True,
            "candidate_membership_and_schedule_replayed": True,
            "frozen_token_counts_replayed": True,
            "no_cloud_gpu_external_api_network_download_or_sft": True,
            "no_human_authorship_claim": True,
            "owner_packet_deterministic": True,
            "schema_language_safety_tool_factual_checks_replayed": True,
            "split_isolation_and_near_dedup_replayed": True,
        },
        "production_eligible": False,
        "next_gate": "owner_review_required_before_any_candidate_registry_preflight",
    }
    verification["verification_sha256"] = sha256_value(verification)
    _write_json(artifact / "evidence" / "verification-report.json", verification)
    return verification
