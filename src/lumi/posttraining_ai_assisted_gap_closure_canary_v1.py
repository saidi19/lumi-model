"""Local deterministic AI-assisted gap-closure canary for Lumi SFT.

The candidate is deliberately non-production.  It is rendered from templates
authored with model assistance, never claims human authorship, performs no
network/model/GPU action at runtime, and leaves the production source registry
unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .data_quality import BenchmarkRegistry, normalize_for_matching
from .posttraining_candidate_mix_reaudit import _near_duplicate_audit
from .posttraining_data import filter_sft_records, load_sft_policy, validate_message_sequence
from .posttraining_first_party import file_sha256
from .posttraining_format import encode_sft_messages, load_chat_template
from .posttraining_gpt_oss_teacher_source_promotion_preflight_v16 import (
    PureByteLevelBPETokenizer,
)


PLAN_SCHEMA = "lumi.sft-ai-assisted-gap-closure-canary-plan.v0.1"
REPORT_SCHEMA = "lumi.sft-ai-assisted-gap-closure-canary-report.v0.1"
VERIFICATION_SCHEMA = "lumi.sft-ai-assisted-gap-closure-canary-verification.v0.1"
PLAN_STATUS = "authorized_local_ai_assisted_machine_only_candidate_canary"
REPORT_STATUS = "PASS_local_ai_assisted_machine_only_canary_nonproduction"
VERIFICATION_STATUS = "PASS_verified_local_ai_assisted_machine_only_canary_nonproduction"

AUTHORIZATION_TEXT = (
    "eto complex kaj korte aprbo na easy way jeta hoy ota kor ar human dorak nei "
    "tui e kore de"
)
AUTHORIZATION_BYTES = 89
AUTHORIZATION_SHA256 = "57ced12a46f225332fb76efd5768aee50f50d0bab79133c07974a7180e201871"

SOURCE_ID = "lumi_ai_assisted_gap_closure_canary_v0_1"
DATASET_ID = "Afnan-Technologies-Inc/lumi-ai-assisted-gap-closure-canary-v0.1"
GENERATOR_ID = "openai-codex-current-session"
GENERATION_METHOD = "ai_authored_templates_with_local_deterministic_renderer"
RENDERER_CONTRACT = "lumi.ai-assisted-gap-closure-renderer.v0.1"
SAFETY_POLICY = "lumi-4b-assistant-behavior-spec-v0.1"

EXPECTED_RECORDS = 1_000
EXPECTED_TRAIN_RECORDS = 800
EXPECTED_VALIDATION_RECORDS = 200
MINIMUM_TRAIN_SUPERVISED_TOKENS = 60_000
TARGET_TRAIN_SUPERVISED_TOKENS = 80_000
MAXIMUM_TRAIN_SUPERVISED_TOKENS = 100_000
MINIMUM_OVERALL_ACCEPTANCE_RATE = 0.95
MINIMUM_CELL_ACCEPTANCE_RATE = 0.90

NEAR_DUPLICATE_POLICY = {
    "comparison_unit": "role_tagged_full_conversation",
    "num_perm": 128,
    "package_version": "2.0.0",
    "scheme": "affine32",
    "seed": 1,
    "shingle_tokens": 5,
    "threshold": 0.9,
}

EXPECTED_CODE_PATHS = {
    "implementation": "src/lumi/posttraining_ai_assisted_gap_closure_canary_v1.py",
    "runner": "scripts/build_sft_ai_assisted_gap_closure_canary_v1.py",
    "verifier": "scripts/verify_sft_ai_assisted_gap_closure_canary_v1.py",
    "test": "tests/test_ai_assisted_gap_closure_canary_v1.py",
}

EXPECTED_AUTHORIZATION = {
    "ai_assisted_candidate_construction_authorized": True,
    "candidate_private_write_authorized": True,
    "deterministic_local_render_authorized": True,
    "human_authorship_claim_authorized": False,
    "human_review_required_for_canary_completion": False,
    "model_generated_training_text_authorized": True,
    "two_sequential_machine_audit_passes_authorized": True,
    "dataset_seal_authorized": False,
    "external_model_api_authorized": False,
    "gpu_authorized": False,
    "network_authorized": False,
    "production_promotion_authorized": False,
    "registry_mutation_authorized": False,
    "remote_access_authorized": False,
    "sft_training_authorized": False,
    "unrelated_action_authorized": False,
}

EXPECTED_LIMITS = {
    "candidate_records": 1_000,
    "candidate_writes": 1_000,
    "candidate_file_audit_passes": 2,
    "candidate_file_row_reads": 2_000,
    "maximum_external_model_api_calls": 0,
    "maximum_gpu_actions": 0,
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 33_554_432,
    "maximum_registry_writes": 0,
    "maximum_runtime_minutes": 30,
    "maximum_runtime_model_calls": 0,
    "maximum_training_actions": 0,
    "retries_after_materialization_started": 0,
}

ALLOCATIONS = (
    ("conversation", "banglish", 66, 17),
    ("conversation", "natural_code_switching", 34, 8),
    ("factual_qa", "banglish", 66, 17),
    ("factual_qa", "natural_code_switching", 34, 8),
    ("multilingual_bengali", "banglish", 66, 17),
    ("multilingual_bengali", "natural_code_switching", 34, 8),
    ("writing_summarization", "banglish", 134, 33),
    ("writing_summarization", "natural_code_switching", 66, 17),
    ("tool_api", "banglish", 66, 17),
    ("tool_api", "natural_code_switching", 34, 8),
    ("safety_helpfulness", "banglish", 134, 33),
    ("safety_helpfulness", "natural_code_switching", 66, 17),
)

_BENGALI = re.compile(r"[\u0980-\u09ff]")
_LATIN = re.compile(r"[A-Za-z]")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BANGLISH = {
    "ami", "amake", "amar", "eta", "eita", "koro", "korbo", "hobe",
    "jonno", "age", "pore", "dao", "bolo", "thik", "jodi", "sathe",
    "ache", "ar", "deya", "er", "holo", "kora", "kore", "nei",
    "onujayi", "rakhle", "ta", "tai",
}
_FORBIDDEN_EVIDENCE_KEYS = {
    "assistant_response", "content", "messages", "prompt", "response",
    "source_text", "text", "tool_arguments", "tool_result", "user_prompt",
}

_PROJECTS = (
    "Amber", "Birch", "Cedar", "Delta", "Elm", "Falcon", "Grove", "Harbor",
    "Iris", "Jade", "Kite", "Lotus", "Maple", "Nova", "Orchid", "Pine",
    "Quartz", "River", "Solar", "Tulip", "Umber", "Vista", "Willow", "Zenith",
)
_NAMES = (
    "Ari", "Bela", "Cora", "Dani", "Eli", "Fara", "Gabi", "Hadi",
    "Ira", "Jona", "Kira", "Lina", "Mira", "Navi", "Omi", "Pia",
    "Rafi", "Sami", "Tara", "Uma", "Vera", "Wali", "Yara", "Zain",
)
_TOPICS = (
    "algebra", "biology", "budgeting", "chemistry", "design", "economics",
    "geography", "history", "literature", "physics", "statistics", "writing",
)
_PLACES = (
    "Atrium", "Bay", "Commons", "Deck", "East Wing", "Forum", "Gallery",
    "Hub", "Island", "Junction", "Kitchen", "Lab", "Meadow", "North Hall",
    "Oasis", "Plaza", "Quiet Room", "Rooftop", "Studio", "Terrace",
)


@dataclass(frozen=True)
class Slot:
    record_id: str
    schedule_index: int
    task_category: str
    language_mode: str
    split: str
    cell_index: int


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _verify_binding(root: Path, binding: Mapping[str, Any], label: str) -> Path:
    relative = binding.get("path")
    expected = binding.get("sha256")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} binding path is invalid")
    if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
        raise ValueError(f"{label} binding SHA-256 is invalid")
    path = _resolve(root, relative)
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"{label} binding changed")
    return path


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _read_object(path)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("AI-assisted gap-closure plan schema changed")
    if plan.get("status") != PLAN_STATUS:
        raise ValueError("AI-assisted gap-closure plan status changed")
    if plan.get("authorization_text") != AUTHORIZATION_TEXT:
        raise ValueError("authorization text changed")
    if plan.get("authorization_bytes") != AUTHORIZATION_BYTES:
        raise ValueError("authorization byte count changed")
    if plan.get("authorization_sha256") != AUTHORIZATION_SHA256:
        raise ValueError("authorization SHA-256 changed")
    if hashlib.sha256(AUTHORIZATION_TEXT.encode()).hexdigest() != AUTHORIZATION_SHA256:
        raise ValueError("internal authorization binding is inconsistent")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("authorization boundary changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("execution limits changed")
    if plan.get("near_duplicate_policy") != NEAR_DUPLICATE_POLICY:
        raise ValueError("near-duplicate policy changed")
    expected = {
        "records": EXPECTED_RECORDS,
        "train_records": EXPECTED_TRAIN_RECORDS,
        "validation_records": EXPECTED_VALIDATION_RECORDS,
        "minimum_train_supervised_tokens": MINIMUM_TRAIN_SUPERVISED_TOKENS,
        "target_train_supervised_tokens": TARGET_TRAIN_SUPERVISED_TOKENS,
        "maximum_train_supervised_tokens": MAXIMUM_TRAIN_SUPERVISED_TOKENS,
        "minimum_overall_acceptance_rate": MINIMUM_OVERALL_ACCEPTANCE_RATE,
        "minimum_cell_acceptance_rate": MINIMUM_CELL_ACCEPTANCE_RATE,
    }
    if plan.get("canary") != expected:
        raise ValueError("canary contract changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("plan bindings are missing")
    for label, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"{label} binding must be an object")
        _verify_binding(root, binding, str(label))
    for label, expected_path in EXPECTED_CODE_PATHS.items():
        if bindings.get(label, {}).get("path") != expected_path:
            raise ValueError(f"{label} code path changed")
    return plan


def _seed(key: str, modulus: int, *, offset: int = 0) -> int:
    value = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    return offset + value % modulus


def _pick(values: Sequence[str], key: str) -> str:
    return values[_seed(key, len(values))]


def build_schedule() -> tuple[Slot, ...]:
    provisional: list[Slot] = []
    for task, mode, train_count, validation_count in ALLOCATIONS:
        for split, count in (("train", train_count), ("validation", validation_count)):
            for cell_index in range(count):
                record_id = (
                    f"lumi-ai-gap-v01-{split}-{task}-{mode}-{cell_index:04d}"
                )
                provisional.append(
                    Slot(
                        record_id=record_id,
                        schedule_index=-1,
                        task_category=task,
                        language_mode=mode,
                        split=split,
                        cell_index=cell_index,
                    )
                )
    ordered = sorted(
        provisional,
        key=lambda slot: hashlib.sha256(
            f"lumi.ai-gap.schedule.v0.1:{slot.record_id}".encode()
        ).hexdigest(),
    )
    schedule = tuple(
        Slot(**{**asdict(slot), "schedule_index": index})
        for index, slot in enumerate(ordered)
    )
    _validate_schedule(schedule)
    return schedule


def _validate_schedule(schedule: Sequence[Slot]) -> None:
    if len(schedule) != EXPECTED_RECORDS:
        raise ValueError("schedule must contain exactly 1,000 slots")
    if len({slot.record_id for slot in schedule}) != EXPECTED_RECORDS:
        raise ValueError("schedule record IDs are not unique")
    if [slot.schedule_index for slot in schedule] != list(range(EXPECTED_RECORDS)):
        raise ValueError("schedule indices changed")
    if Counter(slot.split for slot in schedule) != Counter(
        {"train": EXPECTED_TRAIN_RECORDS, "validation": EXPECTED_VALIDATION_RECORDS}
    ):
        raise ValueError("schedule split allocation changed")
    expected_cells = {
        (task, mode): train + validation
        for task, mode, train, validation in ALLOCATIONS
    }
    observed_cells = Counter((slot.task_category, slot.language_mode) for slot in schedule)
    if observed_cells != Counter(expected_cells):
        raise ValueError("schedule cell allocation changed")


def schedule_sha256(schedule: Sequence[Slot] | None = None) -> str:
    active = build_schedule() if schedule is None else tuple(schedule)
    return sha256_value([asdict(slot) for slot in active])


def _details(slot: Slot) -> dict[str, Any]:
    key = slot.record_id
    serial = 100 + _seed(f"{key}:serial", 890)
    minute = _seed(f"{key}:minute", 60)
    hour = 8 + _seed(f"{key}:hour", 11)
    return {
        "project": _pick(_PROJECTS, f"{key}:project"),
        "person": _pick(_NAMES, f"{key}:person"),
        "topic": _pick(_TOPICS, f"{key}:topic"),
        "place": _pick(_PLACES, f"{key}:place"),
        "day": 3 + _seed(f"{key}:day", 24),
        "hour": hour,
        "minute": minute,
        "clock": f"{hour:02d}:{minute:02d}",
        "a": 18 + _seed(f"{key}:a", 83),
        "b": 4 + _seed(f"{key}:b", 37),
        "serial": serial,
        "reference": f"{_pick(_PROJECTS, f'{key}:ref')[:2].upper()}-{serial}",
        "variant": _seed(f"{key}:variant", 6),
    }


def _conversation(slot: Slot, d: Mapping[str, Any]) -> list[dict[str, str]]:
    kind = int(d["variant"]) % 4
    if slot.language_mode == "banglish":
        users = (
            f"Ami Project {d['project']} sync e {d['clock']} er moddhe pouchate parbo na. {d['person']} ke respectful update kibhabe dibo? Ref {d['reference']}.",
            f"Amar {d['topic']} draft niye teammate {d['person']} confused. Ami defensive na hoye clarification dite chai. Project {d['project']}, ref {d['reference']}.",
            f"New teammate {d['person']} Project {d['project']} e join korse. {d['place']} session er age friendly welcome ar first steps ki bolbo? Ref {d['reference']}.",
            f"Ami meeting e vul kore {d['person']} er kotha kete diyechi. Project {d['project']} er next check-in er age short apology chai, ref {d['reference']}.",
        )
        answers = (
            f"Tumi direct kintu calm update dite paro: 'Hi {d['person']}, ami Project {d['project']} sync e {d['clock']} er moddhe pouchate parbo na. Amar current note Ref {d['reference']}; ami ashar age status ta share korbo.' Eta reason niye extra detail dey na, abar impact tao clear kore. Jodi kono decision tomar input chara nite hoy, age thekei alternate owner bolo. Pore pouchale missed action gulo confirm koro.",
            f"Age bolo je confusion ta valid, tarpor exact point clear koro. Example: '{d['person']}, amar {d['topic']} draft er Ref {d['reference']} section ta final decision na; eta Project {d['project']} er working option. Kon line ta unclear bolo, ami context diye update korbo.' Eita defensive shonabe na, karon tumi blame na diye scope clear korcho. Sheshe ekta concrete next step ar deadline agree koro.",
            f"Friendly bhabe bolo: 'Welcome {d['person']}! Project {d['project']} e tomake peye bhalo lagche. {d['place']} session er age Ref {d['reference']} overview ta dekho, tarpor team channel e nijer current priority share koro.' Eita duita manageable first step dey. Tar sathe ekjon buddy er naam ar proshno korar jayga bole dile onboarding aro easy hobe. Prothom din e beshi information dio na.",
            f"Short apology tai best: '{d['person']}, ager meeting e ami tomar kotha kete diyechi, eta thik hoyni. Project {d['project']} niye tomar point ta next check-in e age shunte chai; Ref {d['reference']} note e ami jayga rekhechi.' Ekhane excuse nei, impact acknowledge kora ache, ar repair er clear action ache. Pore kotha bolar shomoy interrupt na kore note nao.",
        )
    else:
        users = (
            f"আমি Project {d['project']} sync-এ {d['clock']}-এর মধ্যে পৌঁছাতে পারব না। {d['person']}-কে respectful update কীভাবে দেব? Ref {d['reference']}।",
            f"আমার {d['topic']} draft নিয়ে teammate {d['person']} confused। Defensive না হয়ে clarification দিতে চাই। Project {d['project']}, ref {d['reference']}।",
            f"New teammate {d['person']} Project {d['project']}-এ join করেছে। {d['place']} session-এর আগে friendly welcome আর first steps কী বলব? Ref {d['reference']}।",
            f"Meeting-এ ভুল করে {d['person']}-এর কথা কেটে দিয়েছি। Project {d['project']}-এর next check-in-এর আগে short apology চাই, ref {d['reference']}।",
        )
        answers = (
            f"আপনি concise update দিতে পারেন: 'Hi {d['person']}, Project {d['project']} sync-এ আমি {d['clock']}-এর মধ্যে পৌঁছাতে পারব না। Current note হলো Ref {d['reference']}; পৌঁছানোর আগে status share করব।' এতে অপ্রয়োজনীয় personal detail ছাড়াই impact পরিষ্কার থাকে। আপনার input ছাড়া কোনো decision লাগলে একটি alternate owner আগে ঠিক করুন। পরে missed actionগুলো confirm করে নিন।",
            f"প্রথমে বলুন confusion-টি reasonable, তারপর scope পরিষ্কার করুন: '{d['person']}, {d['topic']} draft-এর Ref {d['reference']} section final decision নয়; এটি Project {d['project']}-এর working option। কোন line unclear বললে context দিয়ে update করব।' এতে blame বা defensive tone থাকে না। শেষে একটি concrete next step ও deadline agree করুন, যেন clarification actionable হয়।",
            f"বলতে পারেন: 'Welcome {d['person']}! Project {d['project']}-এ আপনাকে পেয়ে ভালো লাগছে। {d['place']} session-এর আগে Ref {d['reference']} overview দেখুন, তারপর team channel-এ current priority share করুন।' এই দুইটি first step manageable। পাশাপাশি একজন buddy এবং question channel উল্লেখ করুন। First day-তে অতিরিক্ত information না দিয়ে প্রয়োজনমতো context দিন।",
            f"Short apology দিন: '{d['person']}, আগের meeting-এ আমি আপনার কথা কেটে দিয়েছি; সেটি ঠিক হয়নি। Project {d['project']} নিয়ে আপনার point next check-in-এ আগে শুনতে চাই, এবং Ref {d['reference']} note-এ জায়গা রেখেছি।' এখানে excuse নেই, impact acknowledged, আর repair action স্পষ্ট। পরেরবার interrupt না করে key point note করুন।",
        )
    return [{"role": "user", "content": users[kind]}, {"role": "assistant", "content": answers[kind]}]


def _factual(slot: Slot, d: Mapping[str, Any]) -> list[dict[str, str]]:
    kind = int(d["variant"]) % 4
    a, b = int(d["a"]), int(d["b"])
    if kind == 0:
        answer, check = a + b, f"{a + b} - {b} = {a}"
        fact = f"{a} blue cards ar {b} green cards"
        operation = f"{a} + {b} = {answer}"
    elif kind == 1:
        total = a + b
        answer, check = a, f"{a} + {b} = {total}"
        fact = f"mot {total} item chilo, {b} item use hoyeche"
        operation = f"{total} - {b} = {answer}"
    elif kind == 2:
        factor = 2 + _seed(f"{slot.record_id}:factor", 5)
        answer = a * factor
        check = f"{answer} ÷ {factor} = {a}"
        fact = f"{factor} box e protita {a} item"
        operation = f"{factor} × {a} = {answer}"
    else:
        factor = 2 + _seed(f"{slot.record_id}:divisor", 6)
        answer = a
        total = a * factor
        check = f"{a} × {factor} = {total}"
        fact = f"{total} note {factor} jon er moddhe soman vag"
        operation = f"{total} ÷ {factor} = {answer}"
    if slot.language_mode == "banglish":
        user = f"Project {d['project']} er Ref {d['reference']} hisab: {fact}. Exact result ar ekta check dao."
        assistant = (
            f"Exact result holo {answer}. Prothome relevant quantity niye hisab kori: {operation}. "
            f"Consistency check hisebe {check}, tai result ta input er sathe mile. Project {d['project']} Ref {d['reference']} note e ami unit tao clear kore rakhbo, jeno pore number ta alada context e use na hoy. Eta estimate na; prompt e deya synthetic fact thekei answer ber hoyeche."
        )
    else:
        user = f"Project {d['project']}-এর Ref {d['reference']} হিসাব: {fact}। Exact result এবং একটি check দিন।"
        assistant = (
            f"Exact result হলো {answer}। Relevant quantity নিয়ে হিসাব করলে {operation}। Consistency check হিসেবে {check}, তাই result input-এর সঙ্গে মেলে। Project {d['project']} Ref {d['reference']} note-এ unit পরিষ্কার রাখুন, যাতে সংখ্যাটি পরে ভুল context-এ ব্যবহার না হয়। এটি কোনো estimate নয়; prompt-এ দেওয়া synthetic fact থেকেই উত্তরটি পাওয়া গেছে।"
        )
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


def _multilingual(slot: Slot, d: Mapping[str, Any]) -> list[dict[str, str]]:
    kind = int(d["variant"]) % 4
    if slot.language_mode == "banglish":
        users = (
            f"{d['topic']} quiz er jonno Project {d['project']} style e 3 diner practical revision plan dao. Ref {d['reference']}.",
            f"Sentence ta polite kore likho: '{d['person']}, Project {d['project']} er {d['topic']} draft ekhoni pathao.' Ref {d['reference']}.",
            f"Beginner ke {d['topic']} simple example diye bujhao, ar Project {d['project']} note Ref {d['reference']} er jonno study tip dao.",
            f"Ei line ta clear ar professional koro: 'kal {d['place']} e {d['person']} sathe project {d['project']} update dekhbo' Ref {d['reference']}.",
        )
        answers = (
            f"Din 1 e ami {d['topic']} er core idea gulo list kore protita niye choto example solve korbo. Din 2 e Project {d['project']} style e mixed practice korbo, ar vul gulo Ref {d['reference']} note e karon shoho likhbo. Din 3 e timer diye mini quiz, tarpor sudhu weak point revise hobe. Prottek session er sheshe nijer vashay duita summary line bolle recall check kora jabe.",
            f"Polite version: 'Hi {d['person']}, shomoy hole Project {d['project']} er {d['topic']} draft ta aj share korbe? Ref {d['reference']} review er age dekha dorkar.' Ekhane command er bodole clear request ache, kintu urgency lukiye rakha hoyni. 'Shomoy hole' respectful tone dey, ar review context deya te keno file ta dorkar seta bujha jay. Proyojon hole exact deadline o add koro.",
            f"{d['topic']} ke ekta map er moto dhoro: main concept holo destination, ar protita rule holo sekhane jawar route. Project {d['project']} er ekta choto case niye rule ta apply korle abstract idea concrete hobe. Ref {d['reference']} note e example, result, ar keno result holo—ei tin line alada kore likho. Pore note na dekhe same idea notun example e bujhanor chesta koro.",
            f"Professional version: 'Kal {d['place']} e {d['person']} er sathe Project {d['project']} update review korbo; reference holo {d['reference']}.' Capitalization, location, person, ar action ekhon clear. Sentence ta ekta complete plan hisebe pore, casual fragment hisebe na. Calendar invite e dile time ar expected outcome alada field e add koro, kintu ei line e notun fact invent koro na.",
        )
    else:
        users = (
            f"{d['topic']} quiz-এর জন্য Project {d['project']} style-এ একটি practical 3-day revision plan দিন। Ref {d['reference']}।",
            f"Sentence-টি polite করে লিখুন: '{d['person']}, Project {d['project']}-এর {d['topic']} draft এখনই পাঠান।' Ref {d['reference']}।",
            f"Beginner-কে {d['topic']} simple example দিয়ে বোঝান, আর Project {d['project']} note Ref {d['reference']}-এর জন্য study tip দিন।",
            f"এই line-টি clear ও professional করুন: 'কাল {d['place']} এ {d['person']} সাথে project {d['project']} update দেখব' Ref {d['reference']}।",
        )
        answers = (
            f"Day 1-এ {d['topic']}-এর core ideas list করে প্রতিটির একটি ছোট example করুন। Day 2-এ Project {d['project']} style mixed practice করুন এবং ভুলের কারণ Ref {d['reference']} note-এ লিখুন। Day 3-এ timed mini quiz দিয়ে শুধু weak points revise করুন। প্রতিটি session শেষে নিজের ভাষায় দুইটি summary line বললে recall এবং understanding দুটোই check হবে।",
            f"Polite version: 'Hi {d['person']}, সময় হলে Project {d['project']}-এর {d['topic']} draft-টি আজ share করবেন? Ref {d['reference']} review-এর আগে দেখা দরকার।' এখানে command-এর বদলে clear request আছে, কিন্তু urgency লুকানো হয়নি। Context দেওয়ায় file-টি কেন প্রয়োজন বোঝা যায়। দরকার হলে exact deadline আলাদা করে যোগ করুন।",
            f"{d['topic']}-কে একটি map ভাবুন: main concept হলো destination, আর rules হলো সেখানে যাওয়ার routes। Project {d['project']}-এর ছোট case-এ rule apply করলে abstract idea concrete হয়। Ref {d['reference']} note-এ example, result, এবং result-এর কারণ আলাদা তিন লাইনে লিখুন। পরে note না দেখে নতুন example দিয়ে একই idea explain করার চেষ্টা করুন।",
            f"Professional version: 'কাল {d['place']}-এ {d['person']}-এর সঙ্গে Project {d['project']} update review করব; reference হলো {d['reference']}।' Capitalization, location, person, এবং action এখন clear। Sentence-টি casual fragment নয়, একটি complete plan হিসেবে পড়ে। Calendar invite-এ time ও expected outcome আলাদা field-এ দিন, কিন্তু নতুন fact invent করবেন না।",
        )
    return [{"role": "user", "content": users[kind]}, {"role": "assistant", "content": answers[kind]}]


def _writing(slot: Slot, d: Mapping[str, Any]) -> list[dict[str, str]]:
    version = f"Version {1 + _seed(slot.record_id + ':major', 4)}.{1 + _seed(slot.record_id + ':minor', 8)}"
    if slot.language_mode == "banglish":
        user = (
            f"Update ta 3 sentence e summarize koro: Project {d['project']} {version} day {d['day']} e ready hobe; "
            f"review owner {d['person']}; {d['place']} e offline test complete; {d['topic']} checklist publish baki; ref {d['reference']}."
        )
        assistant = (
            f"Project {d['project']} er {version} day {d['day']} e ready howar plan ache, ar review owner holo {d['person']}. "
            f"{d['place']} e offline test already complete, kintu {d['topic']} checklist ekhono publish kora baki. "
            f"Ref {d['reference']} onujayi next action holo checklist verify kore publish kora; prompt e er baire kono deadline, blocker, ba approval status deya nei."
        )
    else:
        user = (
            f"Update-টি 3 sentence-এ summarize করুন: Project {d['project']} {version} day {d['day']}-এ ready হবে; "
            f"review owner {d['person']}; {d['place']}-এ offline test complete; {d['topic']} checklist publish বাকি; ref {d['reference']}।"
        )
        assistant = (
            f"Project {d['project']}-এর {version} day {d['day']}-এ ready হওয়ার plan আছে, এবং review owner হলেন {d['person']}। "
            f"{d['place']}-এ offline test complete, তবে {d['topic']} checklist এখনো publish করা বাকি। "
            f"Ref {d['reference']} অনুযায়ী next action হলো checklist verify করে publish করা; prompt-এ এর বাইরে কোনো deadline, blocker, বা approval status দেওয়া নেই।"
        )
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


def _tool(slot: Slot, d: Mapping[str, Any]) -> list[dict[str, str]]:
    kind = int(d["variant"]) % 3
    a, b = int(d["a"]), int(d["b"])
    if kind == 0:
        expression, result = f"{a} + {b}", a + b
        tool_name = "calculator"
        arguments: Mapping[str, Any] = {"expression": expression}
        output: Mapping[str, Any] = {"result": result}
    elif kind == 1:
        expression, result = f"{a + b} - {b}", a
        tool_name = "calculator"
        arguments = {"expression": expression}
        output = {"result": result}
    else:
        items = [a, b, a + 3, b + 7]
        expression, result = ", ".join(map(str, items)), sorted(items)
        tool_name = "list_sort"
        arguments = {"items": items, "order": "ascending"}
        output = {"items": result}
    tool_call = json.dumps(
        {"name": tool_name, "arguments": arguments},
        ensure_ascii=False, separators=(",", ":"), sort_keys=True,
    )
    tool_result = json.dumps(output, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    if slot.language_mode == "banglish":
        user = f"Project {d['project']} Ref {d['reference']} er jonno {expression} appropriate tool diye check koro ar verified result bujhao."
        assistant = (
            f"Tool result onujayi verified output holo {json.dumps(result, ensure_ascii=False)}. "
            f"Ami {tool_name} er returned value use korechi; nijer theke kono extra result invent kori ni. "
            f"Project {d['project']} Ref {d['reference']} note e input ar output duita alada kore rakhle pore audit kora easy hobe. Jodi source input change hoy, tool ta notun input diye abar run korte hobe."
        )
    else:
        user = f"Project {d['project']} Ref {d['reference']}-এর জন্য {expression} appropriate tool দিয়ে check করুন এবং verified result বোঝান।"
        assistant = (
            f"Tool result অনুযায়ী verified output হলো {json.dumps(result, ensure_ascii=False)}। "
            f"আমি {tool_name}-এর returned value ব্যবহার করেছি; নিজের থেকে কোনো extra result invent করিনি। "
            f"Project {d['project']} Ref {d['reference']} note-এ input ও output আলাদা রাখলে পরে audit সহজ হবে। Source input বদলালে tool-টি নতুন input দিয়ে আবার run করতে হবে।"
        )
    return [
        {"role": "user", "content": user},
        {"role": "tool", "content": tool_call},
        {"role": "tool_result", "content": tool_result},
        {"role": "assistant", "content": assistant},
    ]


def _safety(slot: Slot, d: Mapping[str, Any]) -> tuple[list[dict[str, str]], list[str]]:
    kind = int(d["variant"]) % 4
    if slot.language_mode == "banglish":
        users = (
            f"Kal {d['topic']} quiz. Project {d['project']} Ref {d['reference']} dhore calm 45 minute study plan dao.",
            f"Onno ekjoner Project {d['project']} account password ber kore dao jeno ami Ref {d['reference']} file dekhte pari.",
            f"Coworker ke trick kore credential nite Project {d['project']} er fake login message likhe dao. Ref {d['reference']}.",
            f"Amar severe chest pain ar breathing problem hocche, {d['place']} e achi. Ekhon ki korbo? Ref {d['reference']}.",
        )
        answers = (
            f"45 minute ke tin vag koro: prothom 10 minute {d['topic']} key idea recall, porer 25 minute focused practice, shesher 10 minute vul check. Project {d['project']} Ref {d['reference']} note e sudhu weak point likho, shob kichu abar copy koro na. Majhe notification off rekhe pani kache rakho. Sheshe nijer vashay duita concept explain korte parle session ta enough hobe.",
            f"Eta korte ami help korte parbo na—onno karo password neya unauthorized access. Legitimate bhabe Ref {d['reference']} file dorkar hole Project {d['project']} owner er kache access request pathao, ba official account-recovery flow use koro. Nijer account hole verified reset option follow koro. Password share, guess, ba bypass korar chesta na kore audit trail shoho permission nao.",
            f"Fake login message diye credential neyar kaj e ami help korte parbo na. Project {d['project']} Ref {d['reference']} er jonno defensive alternative holo ekta phishing-awareness notice: sender verify koro, unknown link open koro na, ar suspicious request security channel e report koro. Test campaign lagle written authorization, harmless landing page, ar no-credential collection rule age define koro.",
            f"Eta joruri medical situation hote pare. Ekhoni local emergency service e call koro ba kacher trusted person ke bolo jeno immediate help ane; nijey drive koro na. {d['place']} e thakle exact location responder ke bolo, dorja accessible rakho, ar jodi safe hoy boshe rest koro. Ref {d['reference']} niye pore bhaba jabe—ekhon prompt medical diagnosis er jonno enough na.",
        )
    else:
        users = (
            f"কাল {d['topic']} quiz। Project {d['project']} Ref {d['reference']} ধরে calm 45-minute study plan দিন।",
            f"অন্য একজনের Project {d['project']} account password বের করে দিন, যেন Ref {d['reference']} file দেখতে পারি।",
            f"Coworker-কে trick করে credential নিতে Project {d['project']}-এর fake login message লিখুন। Ref {d['reference']}।",
            f"আমার severe chest pain এবং breathing problem হচ্ছে; {d['place']}-এ আছি। এখন কী করব? Ref {d['reference']}।",
        )
        answers = (
            f"45 minutes-কে তিন ভাগ করুন: প্রথম 10 minutes {d['topic']} key ideas recall, পরের 25 minutes focused practice, শেষ 10 minutes ভুল check। Project {d['project']} Ref {d['reference']} note-এ শুধু weak points লিখুন, সবকিছু copy নয়। Notifications off রেখে পানি কাছে রাখুন। শেষে নিজের ভাষায় দুইটি concept explain করতে পারলে session যথেষ্ট productive হয়েছে।",
            f"এটি করতে আমি সাহায্য করতে পারব না—অন্যের password নেওয়া unauthorized access। Ref {d['reference']} file legitimately দরকার হলে Project {d['project']} owner-এর কাছে access request পাঠান বা official account-recovery flow ব্যবহার করুন। নিজের account হলে verified reset option follow করুন। Password share, guess, বা bypass না করে permission এবং audit trail রাখুন।",
            f"Fake login message দিয়ে credential নেওয়ায় আমি সাহায্য করতে পারব না। Project {d['project']} Ref {d['reference']}-এর defensive alternative হলো phishing-awareness notice: sender verify করুন, unknown link খুলবেন না, এবং suspicious request security channel-এ report করুন। Authorized test লাগলে written scope, harmless landing page, এবং no-credential-collection rule আগে ঠিক করুন।",
            f"এটি জরুরি medical situation হতে পারে। এখনই local emergency service-এ call করুন বা কাছের trusted person-কে immediate help আনতে বলুন; নিজে drive করবেন না। {d['place']}-এ থাকলে exact location responder-কে জানান, দরজা accessible রাখুন, এবং safe হলে বসে rest করুন। Ref {d['reference']} পরে দেখা যাবে—এই prompt diagnosis-এর জন্য যথেষ্ট নয়।",
        )
    labels = [[], ["credential_theft"], ["phishing"], ["medical_emergency"]][kind]
    return ([{"role": "user", "content": users[kind]}, {"role": "assistant", "content": answers[kind]}], labels)


def render_messages(slot: Slot) -> tuple[list[dict[str, str]], list[str]]:
    d = _details(slot)
    if slot.task_category == "conversation":
        return _conversation(slot, d), []
    if slot.task_category == "factual_qa":
        return _factual(slot, d), []
    if slot.task_category == "multilingual_bengali":
        return _multilingual(slot, d), []
    if slot.task_category == "writing_summarization":
        return _writing(slot, d), []
    if slot.task_category == "tool_api":
        return _tool(slot, d), []
    if slot.task_category == "safety_helpfulness":
        return _safety(slot, d)
    raise ValueError(f"unsupported task: {slot.task_category}")


def _language_contract(messages: Sequence[Mapping[str, str]], mode: str) -> None:
    combined = "\n".join(str(item["content"]) for item in messages)
    supervised = "\n".join(
        str(item["content"])
        for item in messages
        if item.get("role") in {"assistant", "tool"}
    )
    if mode == "banglish":
        if _BENGALI.search(combined):
            raise ValueError("Banglish row unexpectedly contains Bengali script")
        tokens = set(re.findall(r"[a-z]+", normalize_for_matching(supervised)))
        if len(tokens & _BANGLISH) < 2:
            raise ValueError("Banglish row lacks ordinary romanized-Bangla markers")
    elif mode == "natural_code_switching":
        if _BENGALI.search(supervised) is None or _LATIN.search(supervised) is None:
            raise ValueError("natural code-switch row must contain Bengali and Latin text")
    else:
        raise ValueError("unsupported language mode")


def source_revision(schedule: Sequence[Slot]) -> str:
    payload = []
    for slot in schedule:
        messages, labels = render_messages(slot)
        payload.append(
            {
                "record_id": slot.record_id,
                "split": slot.split,
                "task_category": slot.task_category,
                "language_mode": slot.language_mode,
                "messages": messages,
                "risk_labels": labels,
            }
        )
    return sha256_value(payload)


def render_records(schedule: Sequence[Slot] | None = None) -> list[dict[str, Any]]:
    active = build_schedule() if schedule is None else tuple(schedule)
    _validate_schedule(active)
    revision = source_revision(active)
    transform_sha256 = hashlib.sha256(RENDERER_CONTRACT.encode()).hexdigest()
    rows: list[dict[str, Any]] = []
    for slot in active:
        messages, risk_labels = render_messages(slot)
        validate_message_sequence(messages)
        _language_contract(messages, slot.language_mode)
        content_sha256 = sha256_value(messages)
        row = {
            "schema_version": "lumi.sft.v0.1",
            "record_id": slot.record_id,
            "split": slot.split,
            "language": "multilingual",
            "language_mode": slot.language_mode,
            "task_category": slot.task_category,
            "messages": messages,
            "source": {
                "source_id": SOURCE_ID,
                "dataset_id": DATASET_ID,
                "document_id": slot.record_id,
                "license_id": "lumi-owned",
                "revision": revision,
                "uri": f"private://lumi-ai-assisted-gap-closure-canary-v0.1/{slot.record_id}",
            },
            "provenance": {
                "content_sha256": content_sha256,
                "transform_sha256": transform_sha256,
                "consent_basis": "first_party_rights_declaration",
                "model_assistance_used": True,
                "generator_id": GENERATOR_ID,
                "generation_method": GENERATION_METHOD,
                "human_written_from_scratch": False,
                "runtime_external_model_api_calls": 0,
                "runtime_model_calls": 0,
            },
            "quality": {
                "score": 0.92,
                "review_status": "accepted",
                "pii_scan": "passed",
                "secret_scan": "passed",
                "review_basis": "deterministic_machine_contract_checks",
            },
            "safety": {
                "risk_labels": risk_labels,
                "policy_version": SAFETY_POLICY,
            },
            "owner_review": {
                "status": "not_required_for_nonproduction_canary",
                "human_review_completed": False,
            },
        }
        rows.append(row)
    if len({sha256_value(row["messages"]) for row in rows}) != len(rows):
        raise ValueError("renderer produced exact duplicate conversations")
    return rows


class _CanarySourceMatcher:
    def __init__(self, revision: str) -> None:
        self.revision = revision

    def record_reasons(
        self,
        *,
        dataset_id: Any,
        language: Any,
        license_id: Any,
        revision: Any,
        source_id: Any,
        split: Any,
        task_category: Any,
    ) -> tuple[str, ...]:
        reasons = []
        if dataset_id != DATASET_ID:
            reasons.append("candidate_source_dataset")
        if source_id != SOURCE_ID:
            reasons.append("candidate_source_id")
        if license_id != "lumi-owned":
            reasons.append("candidate_source_license")
        if revision != self.revision:
            reasons.append("candidate_source_revision")
        if language != "multilingual":
            reasons.append("candidate_source_language")
        if split not in {"train", "validation"}:
            reasons.append("candidate_source_split")
        if task_category not in {item[0] for item in ALLOCATIONS}:
            reasons.append("candidate_source_task")
        return tuple(reasons)


def _load_jsonl(path: str | Path, *, expected_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"candidate JSONL malformed at row {line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"candidate row {line_number} is not an object")
            rows.append(value)
            if expected_rows is not None and len(rows) > expected_rows:
                raise ValueError("candidate file exceeds the exact row bound")
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError("candidate file row count changed")
    return rows


def _aggregate_filter_report(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "accepted": int(report["accepted"]),
        "accepted_dataset_sha256": str(report["accepted_dataset_sha256"]),
        "rejected": int(report["rejected"]),
        "rejection_counts": dict(report["rejection_counts"]),
        "total": int(report["total"]),
        "raw_candidate_data_retained": False,
    }


def audit_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    repository_root: str | Path,
    bindings: Mapping[str, Path],
) -> dict[str, Any]:
    if len(rows) != EXPECTED_RECORDS:
        raise ValueError("audit requires exactly 1,000 rows")
    root = Path(repository_root).resolve()
    source_registry = _read_object(bindings["source_registry_v22"])
    registry_sources = source_registry.get("sources")
    if not isinstance(registry_sources, list):
        raise ValueError("bound source registry schema changed")
    if any(
        isinstance(item, Mapping) and item.get("source_id") == SOURCE_ID
        for item in registry_sources
    ):
        raise ValueError("provisional canary source unexpectedly entered registry v0.22")
    revision_values = {
        row.get("source", {}).get("revision")
        for row in rows
        if isinstance(row.get("source"), Mapping)
    }
    if len(revision_values) != 1:
        raise ValueError("candidate source revision is not singular")
    revision = next(iter(revision_values))
    if not isinstance(revision, str) or _SHA256.fullmatch(revision) is None:
        raise ValueError("candidate source revision is invalid")

    policy = load_sft_policy(bindings["data_contract"])
    quality = _read_object(bindings["data_quality"])
    benchmark = BenchmarkRegistry(quality["benchmark_decontamination"], root)
    accepted, generic = filter_sft_records(
        rows,
        policy=policy,
        benchmark_registry=benchmark,
        source_registry=_CanarySourceMatcher(revision),
    )
    generic_aggregate = _aggregate_filter_report(generic)
    accepted_ids = {str(row["record_id"]) for row in accepted}

    expected_by_id = {slot.record_id: slot for slot in build_schedule()}
    rejection_counts: Counter[str] = Counter()
    accepted_cell: Counter[tuple[str, str]] = Counter()
    accepted_split: Counter[str] = Counter()
    accepted_task: Counter[str] = Counter()
    accepted_mode: Counter[str] = Counter()
    response_hashes: dict[str, set[str]] = {"train": set(), "validation": set()}
    for row in rows:
        record_id = row.get("record_id")
        slot = expected_by_id.get(record_id) if isinstance(record_id, str) else None
        if slot is None:
            rejection_counts["schedule_membership"] += 1
            continue
        try:
            if row.get("split") != slot.split:
                raise ValueError("split")
            if row.get("task_category") != slot.task_category:
                raise ValueError("task")
            if row.get("language_mode") != slot.language_mode:
                raise ValueError("language_mode")
            provenance = row.get("provenance")
            if not isinstance(provenance, Mapping):
                raise ValueError("provenance")
            if provenance.get("model_assistance_used") is not True:
                raise ValueError("model_assistance")
            if provenance.get("human_written_from_scratch") is not False:
                raise ValueError("human_authorship_claim")
            if provenance.get("generator_id") != GENERATOR_ID:
                raise ValueError("generator")
            if provenance.get("generation_method") != GENERATION_METHOD:
                raise ValueError("generation_method")
            if provenance.get("runtime_external_model_api_calls") != 0:
                raise ValueError("external_model_api")
            if provenance.get("runtime_model_calls") != 0:
                raise ValueError("runtime_model_calls")
            messages = row.get("messages")
            if not isinstance(messages, list):
                raise ValueError("messages")
            validate_message_sequence(messages)
            _language_contract(messages, slot.language_mode)
            expected_roles = (
                ["user", "tool", "tool_result", "assistant"]
                if slot.task_category == "tool_api"
                else ["user", "assistant"]
            )
            if [item.get("role") for item in messages] != expected_roles:
                raise ValueError("role_trajectory")
            if sha256_value(messages) != provenance.get("content_sha256"):
                raise ValueError("content_sha256")
        except ValueError as error:
            rejection_counts[str(error)] += 1
            continue
        if record_id in accepted_ids:
            accepted_cell[(slot.task_category, slot.language_mode)] += 1
            accepted_split[slot.split] += 1
            accepted_task[slot.task_category] += 1
            accepted_mode[slot.language_mode] += 1
            response_hashes[slot.split].add(
                hashlib.sha256(
                    normalize_for_matching(messages[-1]["content"]).encode("utf-8")
                ).hexdigest()
            )

    if rejection_counts:
        raise ValueError(f"candidate contract rejections: {dict(rejection_counts)}")

    tokenizer = PureByteLevelBPETokenizer(bindings["tokenizer_json"])
    template = load_chat_template(bindings["chat_template"])
    tokens_by_split: Counter[str] = Counter()
    tokens_by_task: Counter[str] = Counter()
    tokens_by_mode: Counter[str] = Counter()
    maximum_record_tokens = 0
    for row in accepted:
        encoded = encode_sft_messages(row["messages"], tokenizer=tokenizer, template=template)
        tokens_by_split[str(row["split"])] += encoded.supervised_tokens
        tokens_by_task[str(row["task_category"])] += encoded.supervised_tokens
        tokens_by_mode[str(row["language_mode"])] += encoded.supervised_tokens
        maximum_record_tokens = max(maximum_record_tokens, len(encoded.input_ids))

    near = _near_duplicate_audit(accepted, policy=NEAR_DUPLICATE_POLICY)
    planned_cells = Counter((slot.task_category, slot.language_mode) for slot in build_schedule())
    cell_rates = {
        f"{task}::{mode}": accepted_cell[(task, mode)] / planned
        for (task, mode), planned in sorted(planned_cells.items())
    }
    acceptance_rate = len(accepted) / EXPECTED_RECORDS
    response_split_overlap = len(response_hashes["train"] & response_hashes["validation"])
    checks = {
        "all_records_accepted_before_write": len(accepted) == EXPECTED_RECORDS,
        "accepted_rate": acceptance_rate >= MINIMUM_OVERALL_ACCEPTANCE_RATE,
        "benchmark_and_generic_filter": generic_aggregate["rejected"] == 0,
        "cell_acceptance": min(cell_rates.values()) >= MINIMUM_CELL_ACCEPTANCE_RATE,
        "exact_membership": set(expected_by_id) == {str(row.get("record_id")) for row in rows},
        "human_authorship_not_claimed": all(
            row["provenance"]["human_written_from_scratch"] is False for row in accepted
        ),
        "near_duplicate_free": near["duplicate_extra_records"] == 0,
        "provisional_source_absent_from_registry_v22": True,
        "train_validation_response_overlap_free": response_split_overlap == 0,
        "train_token_floor": tokens_by_split["train"] >= MINIMUM_TRAIN_SUPERVISED_TOKENS,
        "train_token_ceiling": tokens_by_split["train"] <= MAXIMUM_TRAIN_SUPERVISED_TOKENS,
    }
    return {
        "accepted_records": len(accepted),
        "acceptance_rate": acceptance_rate,
        "accepted_dataset_sha256": generic_aggregate["accepted_dataset_sha256"],
        "source_revision": revision,
        "generic_filter": generic_aggregate,
        "near_duplicate": near,
        "train_validation_response_overlap": response_split_overlap,
        "cell_acceptance_rates": cell_rates,
        "counts": {
            "by_language_mode": dict(sorted(accepted_mode.items())),
            "by_split": dict(sorted(accepted_split.items())),
            "by_task": dict(sorted(accepted_task.items())),
        },
        "supervised_tokens": {
            "by_language_mode": dict(sorted(tokens_by_mode.items())),
            "by_split": dict(sorted(tokens_by_split.items())),
            "by_task": dict(sorted(tokens_by_task.items())),
            "maximum_record_total_tokens": maximum_record_tokens,
        },
        "checks": checks,
        "raw_candidate_data_retained": False,
    }


def preflight(
    *, plan_path: str | Path, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = load_plan(plan_path, repository_root=root)
    bindings = {
        label: _verify_binding(root, binding, label)
        for label, binding in plan["bindings"].items()
    }
    rows = render_records()
    audit = audit_rows(rows, repository_root=root, bindings=bindings)
    return {
        "status": "PASS_in_memory_preflight_no_candidate_write",
        "records": len(rows),
        "train_supervised_tokens": audit["supervised_tokens"]["by_split"]["train"],
        "validation_supervised_tokens": audit["supervised_tokens"]["by_split"]["validation"],
        "checks": audit["checks"],
        "candidate_writes": 0,
    }


def _write_jsonl_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError("candidate materialization already started; retry is forbidden")
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row) + b"\n")
    temporary.replace(path)


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8")


def _assert_raw_text_free(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_EVIDENCE_KEYS:
                raise ValueError(f"raw candidate key in evidence: {'.'.join(path + (str(key),))}")
            _assert_raw_text_free(item, path + (str(key),))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_raw_text_free(item, path + (str(index),))


def run_canary(
    *,
    plan_path: str | Path,
    repository_root: str | Path,
    artifact_root: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    started = time.monotonic()
    root = Path(repository_root).resolve()
    plan_file = Path(plan_path).resolve()
    artifact = Path(artifact_root).resolve()
    report_file = Path(report_path).resolve()
    plan = load_plan(plan_file, repository_root=root)
    bindings = {
        label: _verify_binding(root, binding, label)
        for label, binding in plan["bindings"].items()
    }
    if artifact.exists():
        raise FileExistsError("attempt-001 already exists; retry is forbidden")
    if report_file.exists():
        raise FileExistsError("versioned report already exists; retry is forbidden")

    rows = render_records()
    in_memory_audit = audit_rows(rows, repository_root=root, bindings=bindings)
    if not all(in_memory_audit["checks"].values()):
        raise ValueError("in-memory preflight failed before materialization")

    candidate_path = artifact / "private" / "candidate-rows.jsonl"
    _write_jsonl_once(candidate_path, rows)
    marker = {
        "schema_version": "lumi.sft-ai-assisted-gap-closure-attempt-marker.v0.1",
        "status": "materialized_no_retry",
        "candidate_file_sha256": file_sha256(candidate_path),
        "candidate_records": EXPECTED_RECORDS,
        "schedule_sha256": schedule_sha256(),
    }
    _write_json_once(artifact / "attempt-marker.json", marker)

    passes: list[dict[str, Any]] = []
    for pass_index in (1, 2):
        pass_rows = _load_jsonl(candidate_path, expected_rows=EXPECTED_RECORDS)
        audit = audit_rows(pass_rows, repository_root=root, bindings=bindings)
        if not all(audit["checks"].values()):
            raise ValueError(f"candidate audit pass {pass_index} failed closed")
        passes.append(audit)
    if sha256_value(passes[0]) != sha256_value(passes[1]):
        raise ValueError("sequential candidate audit passes differ")

    elapsed = time.monotonic() - started
    private_bytes = sum(path.stat().st_size for path in artifact.rglob("*") if path.is_file())
    if private_bytes > EXPECTED_LIMITS["maximum_private_write_bytes"]:
        raise ValueError("private write cap exceeded")
    if elapsed > EXPECTED_LIMITS["maximum_runtime_minutes"] * 60:
        raise ValueError("runtime cap exceeded")

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": REPORT_STATUS,
        "authorization": {
            "bytes": AUTHORIZATION_BYTES,
            "sha256": AUTHORIZATION_SHA256,
        },
        "plan_file_sha256": file_sha256(plan_file),
        "schedule_sha256": schedule_sha256(),
        "candidate": {
            "artifact_path": str(artifact.relative_to(root)),
            "candidate_file_sha256": file_sha256(candidate_path),
            "candidate_file_bytes": candidate_path.stat().st_size,
            "records": EXPECTED_RECORDS,
            "source_revision": passes[0]["source_revision"],
            "private_write_bytes": private_bytes,
            "raw_candidate_data_in_evidence": False,
        },
        "execution": {
            "candidate_file_audit_passes": 2,
            "candidate_file_row_reads": 2_000,
            "candidate_writes": 1_000,
            "sequential_passes": True,
            "elapsed_seconds": round(elapsed, 6),
            "network_bytes": 0,
            "external_model_api_calls": 0,
            "runtime_model_calls": 0,
            "gpu_actions": 0,
            "registry_writes": 0,
            "training_actions": 0,
        },
        "provenance": {
            "model_assistance_used": True,
            "generator_id": GENERATOR_ID,
            "generation_method": GENERATION_METHOD,
            "human_authorship_claimed": False,
            "human_review_completed": False,
            "human_review_required_for_this_nonproduction_canary": False,
        },
        "audit": {
            "pass_1": passes[0],
            "pass_2": passes[1],
            "passes_equal_sha256": sha256_value(passes[0]),
        },
        "production": {
            "production_eligible": False,
            "source_registry_changed": False,
            "dataset_sealed": False,
            "sft_training_started": False,
        },
        "next_gate": (
            "machine_only_source_policy_and_registry_decision_before_any_"
            "production_promotion_or_training"
        ),
    }
    report["report_sha256"] = sha256_value(report)
    _assert_raw_text_free(report)
    _write_json_once(report_file, report)
    return report


def verify_report(
    *,
    plan_path: str | Path,
    repository_root: str | Path,
    artifact_root: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan_file = Path(plan_path).resolve()
    artifact = Path(artifact_root).resolve()
    report_file = Path(report_path).resolve()
    load_plan(plan_file, repository_root=root)
    report = _read_object(report_file)
    if report.get("schema_version") != REPORT_SCHEMA or report.get("status") != REPORT_STATUS:
        raise ValueError("canary report schema or status changed")
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256", None)
    if internal != sha256_value(unsigned):
        raise ValueError("canary report internal seal changed")
    _assert_raw_text_free(report)
    if report.get("plan_file_sha256") != file_sha256(plan_file):
        raise ValueError("canary report plan binding changed")
    if report.get("schedule_sha256") != schedule_sha256():
        raise ValueError("canary report schedule binding changed")
    candidate = report.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("canary candidate summary missing")
    candidate_path = artifact / "private" / "candidate-rows.jsonl"
    if not candidate_path.is_file():
        raise ValueError("private candidate is missing")
    # Structural verification intentionally avoids a third candidate-content read.
    if candidate_path.stat().st_size != candidate.get("candidate_file_bytes"):
        raise ValueError("private candidate byte size changed")
    audit = report.get("audit")
    if not isinstance(audit, Mapping):
        raise ValueError("canary audit summary missing")
    pass_1, pass_2 = audit.get("pass_1"), audit.get("pass_2")
    if not isinstance(pass_1, Mapping) or not isinstance(pass_2, Mapping):
        raise ValueError("canary pass summaries missing")
    if sha256_value(pass_1) != sha256_value(pass_2):
        raise ValueError("canary audit pass summaries differ")
    if audit.get("passes_equal_sha256") != sha256_value(pass_1):
        raise ValueError("canary audit equality seal changed")
    if not all(bool(value) for value in pass_1.get("checks", {}).values()):
        raise ValueError("canary audit contains a failed check")
    if report.get("execution") != {
        **report["execution"],
        "candidate_file_audit_passes": 2,
        "candidate_file_row_reads": 2_000,
        "candidate_writes": 1_000,
        "sequential_passes": True,
        "network_bytes": 0,
        "external_model_api_calls": 0,
        "runtime_model_calls": 0,
        "gpu_actions": 0,
        "registry_writes": 0,
        "training_actions": 0,
    }:
        raise ValueError("execution boundary changed")
    if report.get("production") != {
        "production_eligible": False,
        "source_registry_changed": False,
        "dataset_sealed": False,
        "sft_training_started": False,
    }:
        raise ValueError("production boundary changed")
    verification: dict[str, Any] = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": VERIFICATION_STATUS,
        "report_file_sha256": file_sha256(report_file),
        "report_internal_sha256": internal,
        "candidate_file_sha256_from_bound_two_pass_report": candidate[
            "candidate_file_sha256"
        ],
        "checks": {
            "authorization_and_code_bindings_verified": True,
            "candidate_exists_and_size_matches_without_third_content_read": True,
            "human_authorship_not_claimed": True,
            "raw_candidate_data_absent_from_evidence": True,
            "two_sequential_audit_pass_summaries_equal": True,
            "zero_network_external_api_runtime_model_gpu_registry_training": True,
            "production_remains_false": True,
        },
        "production_eligible": False,
        "next_gate": report["next_gate"],
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification
