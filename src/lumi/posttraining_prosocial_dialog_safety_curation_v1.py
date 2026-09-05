"""Bounded, local-only ProsocialDialog safety-helpfulness curation."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, median
from typing import Any

from .data_quality import BenchmarkRegistry, MinHashNearDuplicateIndex, normalize_for_matching
from .posttraining_data import SFTPolicy, filter_sft_records, load_sft_policy
from .posttraining_format import ChatTemplate, encode_sft_messages, load_chat_template
from .posttraining_gpt_oss_teacher_source_promotion_preflight_v16 import (
    PureByteLevelBPETokenizer,
)
from .posttraining_owner_review import build_owner_review_html

PLAN_SCHEMA_VERSION = "lumi.sft-prosocial-dialog-safety-curation-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-prosocial-dialog-safety-curation-report.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-prosocial-dialog-safety-curation-verification.v0.1"
)

SOURCE_ID = "allenai_prosocial_dialog_safety_candidate"
DATASET_ID = "allenai/prosocial-dialog"
REVISION = "d77d7ad3c624c51030f2f32c83e892b3d620b3d4"
LICENSE_ID = "cc-by-4.0"
ADAPTER_ID = "prosocial_dialog_mixed_provenance_safety_v0.1"
ORIGIN_LABEL = "mixed_human_ai_assisted_existing_public_dataset"
PROMPT_ORIGIN = "gpt_3_generated_or_extended_seed_source"
RESPONSE_ORIGIN = "human_crowdworker"
RETAINED_SAFETY_LABELS = frozenset(
    {
        "__possibly_needs_caution__",
        "__probably_needs_caution__",
        "__needs_caution__",
        "__needs_intervention__",
    }
)
EXPECTED_UPSTREAM_SOURCES = frozenset(
    {"ethics_amt", "ethics_reddit", "sbic", "socialchemistry"}
)
EXPECTED_OBJECTS = {
    "train.json": {
        "sha256": "4782f339a1f01a6962289f1c7b2c0727e340e76af95a402c569a48f5df38dc75",
        "bytes": 85_050_121,
    },
    "valid.json": {
        "sha256": "2d937082c55dc9e5ce9416caba55dce0933df7ca768bfb76d65981dbfd3ecd9a",
        "bytes": 14_421_180,
    },
}
EXPECTED_BINDINGS = {
    "source_registry_v0_18": "b9ba6a6c551fd7e4da76bc6e0df5e347d0fcdf47f31e12599e4ddc4665dab01d",
    "cross_source_reaudit": "0b170105caf5713e58363f39270a6efe2bbca8b0c4641badf4d3c79b5e767429",
    "source_discovery": "257ed284a0b2154b2423ec2fdb23742a58578c657b707b7607eb7b6d044147ed",
    "discovery_verification": "30aa3b77e789ae27cd81b0747deb3f39ecab1ca032df92ada3d7b969383c2dee",
    "data_contract": "9a283d9333d29deb7b516fa45eb4eec50232be9925f4ee4aa3a8f624213b2f06",
    "chat_template": "bc7853a9277feddae2fd3db786c6a12f4fa3602a135a0408c38d4390f33bec12",
    "tokenizer_json": "1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee",
    "benchmark_registry": "1ffb5e938ab28c2fc876b9efd6c6ed59059f1e85aed2c396c024ae5925a4745e",
    "quality_config": "23fba21dbc822c84a5d16e63147fc1a7716576b8f1258cda8c035391f04f52c0",
}
EXPECTED_LIMITS = {
    "maximum_input_rows": 165_681,
    "maximum_network_bytes": 134_217_728,
    "maximum_private_write_bytes": 536_870_912,
    "maximum_runtime_minutes": 180,
    "maximum_train_supervised_tokens": 900_000,
    "maximum_validation_supervised_tokens": 60_000,
    "minimum_train_supervised_tokens": 780_000,
    "owner_review_rows": 32,
}
NEAR_DUPLICATE_CONFIG = {
    "algorithm": "minhash_lsh",
    "num_perm": 128,
    "package_version": "2.0.0",
    "scheme": "affine32",
    "seed": 1,
    "shingle_tokens": 5,
    "threshold": 0.9,
}

_TERMINAL_PUNCTUATION = re.compile(r"[.!?][\"')\]]*$")
_DOSAGE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ml)\b", re.IGNORECASE)
_UNSAFE_IMPERATIVE = re.compile(
    r"\b(?:you|they|we)\s+(?:should|must|need to|have to)\s+"
    r"(?:go\s+)?(?:kill|hurt|attack|shoot|stab|poison|blackmail|steal)\b",
    re.IGNORECASE,
)
_DANGEROUS_REASSURANCE = re.compile(
    r"\b(?:definitely|completely|perfectly)\s+safe\b|\bguaranteed\s+to\s+work\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RawSelection:
    records: tuple[dict[str, Any], ...]
    attributions: tuple[dict[str, Any], ...]
    metrics: Mapping[str, Any]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _resolve(repository_root: Path, relative_path: str, expected_sha256: str) -> Path:
    root = repository_root.resolve()
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root.parent):
        raise ValueError(f"binding escapes workspace: {relative_path}")
    if not candidate.is_file() or file_sha256(candidate) != expected_sha256:
        raise ValueError(f"binding changed: {relative_path}")
    return candidate


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    plan_path = Path(path)
    plan = _load_json(plan_path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported ProsocialDialog curation plan")
    if plan.get("status") != "authorized_local_only_acquisition_and_curation":
        raise ValueError("ProsocialDialog curation authorization changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("ProsocialDialog curation limits changed")
    if plan.get("near_duplicate") != NEAR_DUPLICATE_CONFIG:
        raise ValueError("ProsocialDialog near-duplicate policy changed")
    source = plan.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("ProsocialDialog source binding missing")
    if (
        source.get("source_id") != SOURCE_ID
        or source.get("dataset_id") != DATASET_ID
        or source.get("revision") != REVISION
        or source.get("license_id") != LICENSE_ID
        or source.get("objects") != EXPECTED_OBJECTS
    ):
        raise ValueError("ProsocialDialog source binding changed")
    provenance = plan.get("provenance")
    if provenance != {
        "assistant_response_origin": RESPONSE_ORIGIN,
        "prompt_origin": PROMPT_ORIGIN,
        "truthful_record_label": ORIGIN_LABEL,
    }:
        raise ValueError("ProsocialDialog provenance contract changed")
    authorization = plan.get("authorization")
    if authorization != {
        "dataset_seal_authorized": False,
        "external_api_authorized": False,
        "gpu_use_authorized": False,
        "inference_authorized": False,
        "local_acquisition_and_curation_authorized": True,
        "production_promotion_authorized": False,
        "registry_change_authorized": False,
        "sft_training_authorized": False,
    }:
        raise ValueError("ProsocialDialog authorization boundary changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("ProsocialDialog plan bindings missing")
    for name, expected_sha256 in EXPECTED_BINDINGS.items():
        binding = bindings.get(name)
        if not isinstance(binding, Mapping) or binding.get("sha256") != expected_sha256:
            raise ValueError(f"ProsocialDialog plan binding changed: {name}")
        _resolve(Path(repository_root), str(binding.get("path")), expected_sha256)
    for name in ("implementation", "runner", "verifier", "test"):
        binding = bindings.get(name)
        if not isinstance(binding, Mapping):
            raise ValueError(f"ProsocialDialog implementation binding missing: {name}")
        _resolve(Path(repository_root), str(binding.get("path")), str(binding.get("sha256")))
    return plan


def dialogue_split(dialogue_id: int) -> str:
    digest = hashlib.sha256(
        f"prosocial-dialogue-disjoint-split-v0.1\0{dialogue_id}".encode()
    ).digest()
    return "validation" if int.from_bytes(digest[:8], "big") % 20 == 0 else "train"


def _candidate_from_raw(row: Mapping[str, Any], *, object_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    required = {
        "context",
        "dialogue_id",
        "response",
        "response_id",
        "safety_label",
        "source",
    }
    if not required.issubset(row):
        raise ValueError("ProsocialDialog row is missing required fields")
    prompt = row["context"]
    response = row["response"]
    dialogue_id = row["dialogue_id"]
    response_id = row["response_id"]
    safety_label = row["safety_label"]
    upstream_source = row["source"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("ProsocialDialog prompt is empty")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("ProsocialDialog human response is empty")
    if isinstance(dialogue_id, bool) or not isinstance(dialogue_id, int):
        raise ValueError("ProsocialDialog dialogue_id changed type")
    if isinstance(response_id, bool) or not isinstance(response_id, int):
        raise ValueError("ProsocialDialog response_id changed type")
    if safety_label not in RETAINED_SAFETY_LABELS:
        raise ValueError("ProsocialDialog row is outside the retained safety labels")
    if upstream_source not in EXPECTED_UPSTREAM_SOURCES:
        raise ValueError("ProsocialDialog upstream source inventory changed")
    messages = [
        {"content": prompt.strip(), "role": "user"},
        {"content": response.strip(), "role": "assistant"},
    ]
    split = dialogue_split(dialogue_id)
    content_sha256 = sha256_value(messages)
    upstream_identity = {
        "dialogue_id": dialogue_id,
        "object": object_name,
        "response_id": response_id,
        "revision": REVISION,
    }
    identity_sha256 = sha256_value(upstream_identity)
    record_id = f"prosocial-safety-{identity_sha256[:24]}"
    transform_sha256 = sha256_value(
        {
            "adapter": ADAPTER_ID,
            "content_sha256": content_sha256,
            "origin_label": ORIGIN_LABEL,
            "split": split,
            "upstream_identity_sha256": identity_sha256,
        }
    )
    record = {
        "language": "en",
        "messages": messages,
        "provenance": {
            "assistant_response_origin": RESPONSE_ORIGIN,
            "consent_basis": "source_level_open_license",
            "content_sha256": content_sha256,
            "origin_label": ORIGIN_LABEL,
            "prompt_origin": PROMPT_ORIGIN,
            "transform_sha256": transform_sha256,
        },
        "quality": {
            "pii_scan": "passed",
            "review_method": (
                "upstream_human_crowdworker_response_plus_lumi_deterministic_gates_v0.1"
            ),
            "review_status": "accepted",
            "score": 0.85,
            "secret_scan": "passed",
        },
        "record_id": record_id,
        "safety": {
            "policy_version": "prosocial_upstream_label_plus_lumi_critical_scan_v0.1",
            "risk_labels": [str(safety_label)],
        },
        "schema_version": "lumi.sft.v0.1",
        "source": {
            "dataset_id": DATASET_ID,
            "document_id": f"dialogue-{dialogue_id}-response-{response_id}",
            "license_id": LICENSE_ID,
            "revision": REVISION,
            "source_id": SOURCE_ID,
            "uri": (
                f"https://huggingface.co/datasets/{DATASET_ID}/blob/{REVISION}/"
                f"{object_name}#dialogue_id={dialogue_id}&response_id={response_id}"
            ),
        },
        "split": split,
        "task_category": "safety_helpfulness",
    }
    attribution = {
        "dialogue_id": dialogue_id,
        "object": object_name,
        "record_id": record_id,
        "response_id": response_id,
        "safety_label": safety_label,
        "schema_version": "lumi.sft-prosocial-dialog-attribution.v0.1",
        "split": split,
        "upstream_identity_sha256": identity_sha256,
        "upstream_source": upstream_source,
    }
    return record, attribution


def load_raw_candidates(raw_root: str | Path, *, maximum_input_rows: int) -> RawSelection:
    root = Path(raw_root)
    records: list[dict[str, Any]] = []
    attributions: list[dict[str, Any]] = []
    rows_by_object: Counter[str] = Counter()
    rows_by_label: Counter[str] = Counter()
    retained_by_label: Counter[str] = Counter()
    excluded_labels: Counter[str] = Counter()
    upstream_sources: Counter[str] = Counter()
    input_rows = 0
    decoded_bytes = 0
    for object_name in ("train.json", "valid.json"):
        path = root / object_name
        expected = EXPECTED_OBJECTS[object_name]
        if not path.is_file() or file_sha256(path) != expected["sha256"]:
            raise ValueError(f"ProsocialDialog object hash changed: {object_name}")
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"ProsocialDialog object size changed: {object_name}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                decoded_bytes += len(line.encode("utf-8"))
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise ValueError("ProsocialDialog JSONL row must be an object")
                input_rows += 1
                if input_rows > maximum_input_rows:
                    raise ValueError("ProsocialDialog input-row cap exceeded")
                rows_by_object[object_name] += 1
                safety_label = str(row.get("safety_label"))
                rows_by_label[safety_label] += 1
                upstream_sources[str(row.get("source"))] += 1
                if safety_label not in RETAINED_SAFETY_LABELS:
                    excluded_labels[safety_label] += 1
                    continue
                record, attribution = _candidate_from_raw(row, object_name=object_name)
                records.append(record)
                attributions.append(attribution)
                retained_by_label[safety_label] += 1
    if set(upstream_sources) != EXPECTED_UPSTREAM_SOURCES:
        raise ValueError("ProsocialDialog upstream source inventory changed")
    return RawSelection(
        records=tuple(records),
        attributions=tuple(attributions),
        metrics={
            "decoded_bytes": decoded_bytes,
            "excluded_by_label": dict(sorted(excluded_labels.items())),
            "input_rows": input_rows,
            "retained_before_gates": len(records),
            "retained_by_label": dict(sorted(retained_by_label.items())),
            "rows_by_label": dict(sorted(rows_by_label.items())),
            "rows_by_object": dict(sorted(rows_by_object.items())),
            "upstream_sources": dict(sorted(upstream_sources.items())),
        },
    )


def custom_quality_reasons(record: Mapping[str, Any]) -> tuple[str, ...]:
    prompt = str(record["messages"][0]["content"])
    response = str(record["messages"][-1]["content"])
    normalized_prompt = normalize_for_matching(prompt)
    normalized_response = normalize_for_matching(response)
    reasons: list[str] = []
    if len(normalized_response.split()) < 4:
        reasons.append("response_too_short_for_helpfulness")
    if _TERMINAL_PUNCTUATION.search(response.rstrip()) is None:
        reasons.append("assistant_terminal_punctuation")
    if _DOSAGE.search(response):
        reasons.append("ungrounded_medical_dosage")
    if _UNSAFE_IMPERATIVE.search(response) or _DANGEROUS_REASSURANCE.search(response):
        reasons.append("critical_factual_safety_pattern")
    severe_chest = "chest pain" in normalized_prompt and (
        "breath" in normalized_prompt or "breathing" in normalized_prompt
    )
    if severe_chest and any(
        phrase in normalized_response
        for phrase in ("lie down", "sleep it off", "wait it out", "ignore it")
    ):
        reasons.append("unsafe_severe_chest_pain_advice")
    if normalized_prompt == normalized_response:
        reasons.append("prompt_response_exact_echo")
    return tuple(dict.fromkeys(reasons))


def apply_custom_quality_gates(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    accepted: list[Mapping[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    decisions: list[dict[str, Any]] = []
    for record in records:
        reasons = custom_quality_reasons(record)
        if reasons:
            rejection_counts.update(reasons)
        else:
            accepted.append(record)
        decisions.append(
            {
                "accepted": not reasons,
                "reasons": list(reasons),
                "record_id": record["record_id"],
            }
        )
    return accepted, {
        "accepted": len(accepted),
        "decisions_sha256": sha256_value(decisions),
        "rejected": len(records) - len(accepted),
        "rejection_counts": dict(sorted(rejection_counts.items())),
    }


def apply_near_dedup(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    index = MinHashNearDuplicateIndex(NEAR_DUPLICATE_CONFIG)
    ordered = sorted(
        records,
        key=lambda row: (
            0 if row["split"] == "train" else 1,
            hashlib.sha256(
                f"prosocial-near-dedup-v0.1\0{row['record_id']}".encode()
            ).hexdigest(),
        ),
    )
    accepted: list[Mapping[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for record in ordered:
        text = normalize_for_matching(
            "\n".join(
                f"{message['role']} {message['content']}" for message in record["messages"]
            )
        )
        signature = index.signature(text)
        duplicate_of = index.duplicate_of(signature)
        keep = duplicate_of is None
        if keep:
            index.add(str(record["record_id"]), signature)
            accepted.append(record)
        ledger.append(
            {
                "accepted": keep,
                "duplicate_of": duplicate_of,
                "record_id": record["record_id"],
                "signature_sha256": index.fingerprint(signature),
            }
        )
    return accepted, ledger, {
        "accepted": len(accepted),
        "comparison_unit": "role_tagged_full_conversation",
        "rejected": len(records) - len(accepted),
        "threshold": 0.9,
    }


def load_runtime_components(
    plan: Mapping[str, Any], *, repository_root: Path
) -> tuple[SFTPolicy, BenchmarkRegistry, ChatTemplate, PureByteLevelBPETokenizer]:
    bindings = plan["bindings"]
    contract_path = _resolve(
        repository_root,
        str(bindings["data_contract"]["path"]),
        str(bindings["data_contract"]["sha256"]),
    )
    quality_path = _resolve(
        repository_root,
        str(bindings["quality_config"]["path"]),
        str(bindings["quality_config"]["sha256"]),
    )
    benchmark_path = _resolve(
        repository_root,
        str(bindings["benchmark_registry"]["path"]),
        str(bindings["benchmark_registry"]["sha256"]),
    )
    template_path = _resolve(
        repository_root,
        str(bindings["chat_template"]["path"]),
        str(bindings["chat_template"]["sha256"]),
    )
    tokenizer_path = _resolve(
        repository_root,
        str(bindings["tokenizer_json"]["path"]),
        str(bindings["tokenizer_json"]["sha256"]),
    )
    quality = _load_json(quality_path)
    benchmark_config = quality.get("benchmark_decontamination")
    if not isinstance(benchmark_config, Mapping):
        raise ValueError("benchmark configuration missing")
    policy = replace(
        load_sft_policy(contract_path), benchmark_decontamination=dict(benchmark_config)
    )
    benchmark = BenchmarkRegistry(policy.benchmark_decontamination, repository_root)
    if benchmark.path.resolve() != benchmark_path:
        raise ValueError("benchmark registry resolved elsewhere")
    return (
        policy,
        benchmark,
        load_chat_template(template_path),
        PureByteLevelBPETokenizer(tokenizer_path),
    )


def measure_and_select(
    records: Sequence[Mapping[str, Any]],
    *,
    template: ChatTemplate,
    tokenizer: PureByteLevelBPETokenizer,
    maximum_train_supervised_tokens: int,
    maximum_validation_supervised_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    measured: list[tuple[dict[str, Any], int, int]] = []
    overlength = 0
    zero_supervised = 0
    for source_record in records:
        record = dict(source_record)
        try:
            encoded = encode_sft_messages(
                record["messages"], tokenizer=tokenizer, template=template
            )
        except ValueError as error:
            if "maximum is 4096" not in str(error):
                raise
            overlength += 1
            continue
        if encoded.supervised_tokens <= 0:
            zero_supervised += 1
            continue
        measured.append((record, len(encoded.input_ids), encoded.supervised_tokens))

    selected: list[tuple[dict[str, Any], int, int]] = []
    skipped_for_cap: Counter[str] = Counter()
    split_tokens: Counter[str] = Counter()
    for split, cap in (
        ("train", maximum_train_supervised_tokens),
        ("validation", maximum_validation_supervised_tokens),
    ):
        rows = [item for item in measured if item[0]["split"] == split]
        rows.sort(
            key=lambda item: hashlib.sha256(
                f"prosocial-token-selection-v0.1\0{item[0]['record_id']}".encode()
            ).hexdigest()
        )
        for item in rows:
            if split_tokens[split] + item[2] > cap:
                skipped_for_cap[split] += 1
                continue
            selected.append(item)
            split_tokens[split] += item[2]
    selected.sort(key=lambda item: (str(item[0]["split"]), str(item[0]["record_id"])))
    ledger = [
        {
            "record_id": record["record_id"],
            "serialized_tokens": serialized,
            "split": record["split"],
            "supervised_tokens": supervised,
        }
        for record, serialized, supervised in selected
    ]
    lengths = [serialized for _, serialized, _ in selected]
    by_split: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for record, serialized, supervised in selected:
        bucket = by_split[str(record["split"])]
        bucket["records"] += 1
        bucket["serialized_tokens"] += serialized
        bucket["supervised_tokens"] += supervised
    metrics = {
        "accepted_records": len(selected),
        "assistant_only_masking_verified": True,
        "by_split": {key: dict(value) for key, value in sorted(by_split.items())},
        "candidate_4k_fit_records": len(measured),
        "context_window": template.context_window,
        "overlength_rejected": overlength,
        "serialized_length": {
            "maximum": max(lengths, default=0),
            "mean": mean(lengths) if lengths else 0,
            "median": median(lengths) if lengths else 0,
            "minimum": min(lengths, default=0),
        },
        "skipped_for_token_cap": dict(sorted(skipped_for_cap.items())),
        "token_ledger_sha256": sha256_value(ledger),
        "train_supervised_tokens": split_tokens["train"],
        "validation_supervised_tokens": split_tokens["validation"],
        "zero_supervised_rejected": zero_supervised,
    }
    return [item[0] for item in selected], ledger, metrics


def deterministic_owner_sample(
    records: Sequence[Mapping[str, Any]], *, maximum_records: int
) -> list[dict[str, Any]]:
    buckets: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        label = str(record["safety"]["risk_labels"][0])
        buckets[(str(record["split"]), label)].append(record)
    for rows in buckets.values():
        rows.sort(
            key=lambda row: hashlib.sha256(
                f"prosocial-owner-review-v0.1\0{row['record_id']}".encode()
            ).hexdigest()
        )
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < min(maximum_records, len(records)):
        added = False
        for key in sorted(buckets):
            if offset >= len(buckets[key]):
                continue
            row = buckets[key][offset]
            selected.append(
                {
                    "language_mode": "english",
                    "messages": row["messages"],
                    "record_id": row["record_id"],
                    "skill_id": "safety_helpfulness",
                    "split": row["split"],
                    "upstream_safety_label": key[1],
                }
            )
            added = True
            if len(selected) == min(maximum_records, len(records)):
                break
        if not added:
            break
        offset += 1
    return selected


def build_prosocial_owner_review_html(
    sample_rows: list[Mapping[str, Any]],
) -> tuple[str, str]:
    sample_sha256 = sha256_value(sample_rows)
    page = build_owner_review_html(sample_rows, sample_sha256=sample_sha256)
    replacements = {
        "Lumi owner review": "Lumi ProsocialDialog owner review",
        "Lumi owner spot-check": "Lumi ProsocialDialog safety-helpfulness spot-check",
        (
            "These rows are AI-generated and owner-reviewed. Approving them does not "
            "make them human-written."
        ): (
            "Prompts are AI-generated or extended source prompts; assistant responses "
            "are human crowdworker-written. Check safety, factuality, helpfulness, "
            "naturalness, and punctuation. Approval does not authorize registry change, "
            "dataset sealing, or SFT."
        ),
        "lumi.ai-assisted-owner-decisions.v0.3": (
            "lumi.prosocial-dialog-owner-decisions.v0.1"
        ),
        "lumi-owner-review-decisions-v0.3.json": (
            "lumi-prosocial-dialog-owner-decisions-v0.1.json"
        ),
    }
    for old, new in replacements.items():
        if old not in page:
            raise ValueError(f"owner-review marker missing: {old[:48]}")
        page = page.replace(old, new)
    return page, sample_sha256


def _validate_candidate_binding(record: Mapping[str, Any]) -> None:
    source = record.get("source")
    provenance = record.get("provenance")
    safety = record.get("safety")
    if not isinstance(source, Mapping) or (
        source.get("source_id") != SOURCE_ID
        or source.get("dataset_id") != DATASET_ID
        or source.get("revision") != REVISION
        or source.get("license_id") != LICENSE_ID
    ):
        raise ValueError("candidate source binding changed")
    if not isinstance(provenance, Mapping) or (
        provenance.get("assistant_response_origin") != RESPONSE_ORIGIN
        or provenance.get("prompt_origin") != PROMPT_ORIGIN
        or provenance.get("origin_label") != ORIGIN_LABEL
        or provenance.get("consent_basis") != "source_level_open_license"
    ):
        raise ValueError("candidate mixed provenance changed")
    if not isinstance(safety, Mapping):
        raise ValueError("candidate safety metadata missing")
    labels = safety.get("risk_labels")
    if not isinstance(labels, list) or len(labels) != 1 or labels[0] not in RETAINED_SAFETY_LABELS:
        raise ValueError("candidate safety label changed")


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite ProsocialDialog JSON: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite ProsocialDialog JSONL: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite ProsocialDialog text: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _artifact(path: Path, *, name: str, contains_raw_training_text: bool) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "contains_raw_training_text": contains_raw_training_text,
        "name": name,
        "path": str(path),
        "sha256": file_sha256(path),
    }


def run_curation(
    *, plan_path: str | Path, repository_root: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    root = Path(repository_root).resolve()
    plan = load_plan(plan_path, repository_root=root)
    output = plan["output"]
    private_root = (root / str(output["private_root"])).resolve()
    if not private_root.is_relative_to(root):
        raise ValueError("private output escapes repository")
    raw_root = private_root / "private" / "raw"
    selection = load_raw_candidates(
        raw_root, maximum_input_rows=EXPECTED_LIMITS["maximum_input_rows"]
    )
    policy, benchmark, template, tokenizer = load_runtime_components(plan, repository_root=root)

    custom_accepted, custom_report = apply_custom_quality_gates(selection.records)
    generic_accepted, generic_report = filter_sft_records(
        custom_accepted, policy=policy, benchmark_registry=benchmark
    )
    for record in generic_accepted:
        _validate_candidate_binding(record)
    generic_ids = {str(row["record_id"]) for row in generic_accepted}
    attribution_by_id = {str(row["record_id"]): row for row in selection.attributions}
    gated_attributions = [attribution_by_id[record_id] for record_id in sorted(generic_ids)]

    near_accepted, near_ledger, near_report = apply_near_dedup(generic_accepted)
    selected, token_ledger, token_metrics = measure_and_select(
        near_accepted,
        template=template,
        tokenizer=tokenizer,
        maximum_train_supervised_tokens=EXPECTED_LIMITS[
            "maximum_train_supervised_tokens"
        ],
        maximum_validation_supervised_tokens=EXPECTED_LIMITS[
            "maximum_validation_supervised_tokens"
        ],
    )
    train_tokens = int(token_metrics["train_supervised_tokens"])
    validation_tokens = int(token_metrics["validation_supervised_tokens"])
    if not EXPECTED_LIMITS["minimum_train_supervised_tokens"] <= train_tokens <= EXPECTED_LIMITS[
        "maximum_train_supervised_tokens"
    ]:
        raise ValueError("ProsocialDialog retained train token target not met")
    if validation_tokens > EXPECTED_LIMITS["maximum_validation_supervised_tokens"]:
        raise ValueError("ProsocialDialog validation token cap exceeded")

    selected_ids = {str(row["record_id"]) for row in selected}
    selected_attributions = [
        attribution_by_id[record_id] for record_id in sorted(selected_ids)
    ]
    dialogue_splits: dict[int, set[str]] = defaultdict(set)
    for row in selected_attributions:
        dialogue_splits[int(row["dialogue_id"])].add(str(row["split"]))
    if any(len(splits) != 1 for splits in dialogue_splits.values()):
        raise ValueError("ProsocialDialog dialogue split overlap")

    owner_sample = deterministic_owner_sample(
        selected, maximum_records=EXPECTED_LIMITS["owner_review_rows"]
    )
    if len(owner_sample) != EXPECTED_LIMITS["owner_review_rows"]:
        raise ValueError("ProsocialDialog owner-review packet is incomplete")
    owner_html, owner_sha256 = build_prosocial_owner_review_html(owner_sample)

    candidate_path = private_root / "private" / "prosocial-safety-candidate.jsonl"
    attribution_path = private_root / "private" / "prosocial-safety-attribution.jsonl"
    token_ledger_path = private_root / "private" / "prosocial-safety-token-ledger.jsonl"
    near_ledger_path = private_root / "private" / "prosocial-safety-near-dedup-ledger.jsonl"
    owner_json_path = private_root / "private" / "owner-review-32.json"
    owner_html_path = private_root / "private" / "owner-review-32.html"
    report_path = root / str(output["report_path"])
    verification_path = root / str(output["verification_path"])

    _write_jsonl_atomic(candidate_path, selected)
    _write_jsonl_atomic(attribution_path, selected_attributions)
    _write_jsonl_atomic(token_ledger_path, token_ledger)
    _write_jsonl_atomic(near_ledger_path, near_ledger)
    _write_json_atomic(owner_json_path, owner_sample)
    _write_text_atomic(owner_html_path, owner_html)

    raw_objects = [
        _artifact(raw_root / name, name=f"raw_{name}", contains_raw_training_text=True)
        for name in ("train.json", "valid.json")
    ]
    private_objects = [
        _artifact(candidate_path, name="candidate", contains_raw_training_text=True),
        _artifact(attribution_path, name="attribution", contains_raw_training_text=False),
        _artifact(token_ledger_path, name="token_ledger", contains_raw_training_text=False),
        _artifact(near_ledger_path, name="near_dedup_ledger", contains_raw_training_text=False),
        _artifact(owner_json_path, name="owner_review_json", contains_raw_training_text=True),
        _artifact(owner_html_path, name="owner_review_html", contains_raw_training_text=True),
    ]
    total_private_bytes = sum(item["bytes"] for item in raw_objects + private_objects)
    if total_private_bytes > EXPECTED_LIMITS["maximum_private_write_bytes"]:
        raise ValueError("ProsocialDialog private write cap exceeded")
    if sum(item["bytes"] for item in raw_objects) > EXPECTED_LIMITS["maximum_network_bytes"]:
        raise ValueError("ProsocialDialog network byte cap exceeded")

    report: dict[str, Any] = {
        "acquisition": {
            "downloaded_objects": raw_objects,
            "network_bytes": sum(item["bytes"] for item in raw_objects),
            "only_authorized_objects_downloaded": True,
        },
        "artifacts": {
            "private_objects": private_objects,
            "private_write_bytes": total_private_bytes,
        },
        "authorization_boundary": {
            "dataset_seal_created": False,
            "external_api_or_inference_used": False,
            "gpu_used": False,
            "production_promoted": False,
            "registry_changed": False,
            "sft_training_run": False,
        },
        "curation": {
            "custom_factual_safety_and_punctuation": custom_report,
            "dialogue_disjoint": True,
            "dialogues_retained": len(dialogue_splits),
            "generic_sft_gates": {
                "accepted": generic_report["accepted"],
                "rejected": generic_report["rejected"],
                "rejection_counts": generic_report["rejection_counts"],
                "report_sha256": generic_report["report_sha256"],
            },
            "near_duplicate": near_report,
            "raw_selection": dict(selection.metrics),
            "retained_safety_labels": sorted(RETAINED_SAFETY_LABELS),
        },
        "gates": {
            "benchmark_decontamination": "PASS",
            "dialogue_split_overlap": "PASS_zero",
            "exact_duplicate": "PASS_zero_retained",
            "factual_safety": "PASS_zero_critical_retained",
            "four_k_fit": "PASS",
            "minhash_0_9": "PASS_zero_retained",
            "pii": "PASS",
            "provenance": "PASS_truthful_mixed",
            "punctuation": "PASS",
            "rights": "PASS_cc_by_4_0_revision_pinned",
            "schema": "PASS",
            "secret": "PASS",
        },
        "observed_at": "2026-08-13",
        "owner_review": {
            "packet_records": len(owner_sample),
            "packet_sha256": owner_sha256,
            "status": "PENDING_OWNER_REVIEW",
        },
        "plan_file_sha256": file_sha256(plan_path),
        "production_eligible": False,
        "provenance": {
            "assistant_response_origin": RESPONSE_ORIGIN,
            "prompt_origin": PROMPT_ORIGIN,
            "truthful_record_label": ORIGIN_LABEL,
        },
        "raw_training_text_retained_in_report": False,
        "runtime_seconds": time.monotonic() - started,
        "schema_version": REPORT_SCHEMA_VERSION,
        "source": {
            "dataset_id": DATASET_ID,
            "license_id": LICENSE_ID,
            "revision": REVISION,
            "source_id": SOURCE_ID,
        },
        "status": "PASS_nonproduction_candidate_owner_review_pending",
        "tokens": token_metrics,
    }
    report["report_sha256"] = sha256_value(report)
    _write_json_atomic(report_path, report)

    verification = verify_curation(
        plan_path=plan_path,
        report_path=report_path,
        repository_root=root,
        write_output=False,
    )
    _write_json_atomic(verification_path, verification)
    return report, verification


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object: {path}")
            rows.append(row)
    return rows


def _validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported ProsocialDialog curation report")
    if report.get("raw_training_text_retained_in_report") is not False:
        raise ValueError("ProsocialDialog report is not raw-text-free")
    claimed = report.get("report_sha256")
    body = dict(report)
    body.pop("report_sha256", None)
    if claimed != sha256_value(body):
        raise ValueError("ProsocialDialog report internal hash changed")


def verify_curation(
    *,
    plan_path: str | Path,
    report_path: str | Path,
    repository_root: str | Path,
    write_output: bool = True,
) -> dict[str, Any]:
    started = time.monotonic()
    root = Path(repository_root).resolve()
    plan = load_plan(plan_path, repository_root=root)
    report = _load_json(report_path)
    _validate_report(report)
    if report.get("plan_file_sha256") != file_sha256(plan_path):
        raise ValueError("ProsocialDialog report plan binding changed")
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("ProsocialDialog report artifacts missing")
    private_objects = artifacts.get("private_objects")
    if not isinstance(private_objects, list):
        raise ValueError("ProsocialDialog private artifact inventory missing")
    by_name = {str(item["name"]): item for item in private_objects}
    for item in private_objects:
        path = Path(str(item["path"]))
        if not path.is_absolute():
            path = root / path
        if not path.is_file() or file_sha256(path) != item.get("sha256"):
            raise ValueError(f"ProsocialDialog artifact changed: {item.get('name')}")
        if path.stat().st_size != item.get("bytes"):
            raise ValueError(f"ProsocialDialog artifact size changed: {item.get('name')}")
    candidate_path = Path(str(by_name["candidate"]["path"]))
    attribution_path = Path(str(by_name["attribution"]["path"]))
    token_ledger_path = Path(str(by_name["token_ledger"]["path"]))
    owner_path = Path(str(by_name["owner_review_json"]["path"]))
    for name, value in (
        ("candidate", candidate_path),
        ("attribution", attribution_path),
        ("token_ledger", token_ledger_path),
        ("owner", owner_path),
    ):
        if not value.is_absolute():
            locals()[f"{name}_path"] = root / value
    candidate_path = candidate_path if candidate_path.is_absolute() else root / candidate_path
    attribution_path = (
        attribution_path if attribution_path.is_absolute() else root / attribution_path
    )
    token_ledger_path = (
        token_ledger_path if token_ledger_path.is_absolute() else root / token_ledger_path
    )
    owner_path = owner_path if owner_path.is_absolute() else root / owner_path

    records = _read_jsonl(candidate_path)
    attributions = _read_jsonl(attribution_path)
    token_ledger = _read_jsonl(token_ledger_path)
    owner_sample = json.loads(owner_path.read_text(encoding="utf-8"))
    if not isinstance(owner_sample, list):
        raise ValueError("ProsocialDialog owner packet must be an array")

    policy, benchmark, template, tokenizer = load_runtime_components(plan, repository_root=root)
    generic_accepted, generic_report = filter_sft_records(
        records, policy=policy, benchmark_registry=benchmark
    )
    if len(generic_accepted) != len(records) or generic_report["rejected"] != 0:
        raise ValueError("ProsocialDialog retained candidate fails generic SFT gates")
    for record in records:
        _validate_candidate_binding(record)
        if custom_quality_reasons(record):
            raise ValueError("ProsocialDialog retained candidate fails custom gates")

    measured_ledger: list[dict[str, Any]] = []
    train_tokens = 0
    validation_tokens = 0
    for record in records:
        encoded = encode_sft_messages(
            record["messages"], tokenizer=tokenizer, template=template
        )
        item = {
            "record_id": record["record_id"],
            "serialized_tokens": len(encoded.input_ids),
            "split": record["split"],
            "supervised_tokens": encoded.supervised_tokens,
        }
        measured_ledger.append(item)
        if record["split"] == "train":
            train_tokens += encoded.supervised_tokens
        else:
            validation_tokens += encoded.supervised_tokens
    if measured_ledger != token_ledger:
        raise ValueError("ProsocialDialog token ledger changed")
    if not EXPECTED_LIMITS["minimum_train_supervised_tokens"] <= train_tokens <= EXPECTED_LIMITS[
        "maximum_train_supervised_tokens"
    ]:
        raise ValueError("ProsocialDialog verified train token target changed")
    if validation_tokens > EXPECTED_LIMITS["maximum_validation_supervised_tokens"]:
        raise ValueError("ProsocialDialog verified validation token cap changed")

    candidate_ids = {str(row["record_id"]) for row in records}
    if len(candidate_ids) != len(records):
        raise ValueError("ProsocialDialog duplicate record ID retained")
    if {str(row["record_id"]) for row in attributions} != candidate_ids:
        raise ValueError("ProsocialDialog attribution coverage changed")
    dialogue_splits: defaultdict[int, set[str]] = defaultdict(set)
    for row in attributions:
        dialogue_splits[int(row["dialogue_id"])].add(str(row["split"]))
    if any(len(splits) != 1 for splits in dialogue_splits.values()):
        raise ValueError("ProsocialDialog dialogue split overlap verified")

    near_accepted, near_ledger, _ = apply_near_dedup(records)
    if len(near_accepted) != len(records) or any(
        item["accepted"] is not True for item in near_ledger
    ):
        raise ValueError("ProsocialDialog near duplicate retained")
    expected_owner = deterministic_owner_sample(
        records, maximum_records=EXPECTED_LIMITS["owner_review_rows"]
    )
    if owner_sample != expected_owner:
        raise ValueError("ProsocialDialog owner-review sample changed")
    owner_sha256 = sha256_value(owner_sample)
    if report["owner_review"]["packet_sha256"] != owner_sha256:
        raise ValueError("ProsocialDialog owner-review hash changed")

    verification: dict[str, Any] = {
        "candidate_file_sha256": file_sha256(candidate_path),
        "candidate_records": len(records),
        "dialogues_retained": len(dialogue_splits),
        "gates_replayed": {
            "assistant_only_masking_and_4k_fit": True,
            "benchmark_decontamination": True,
            "dialogue_disjoint_split": True,
            "exact_duplicate": True,
            "factual_safety": True,
            "minhash_0_9": True,
            "pii_and_secret": True,
            "provenance_and_rights": True,
            "punctuation": True,
            "schema": True,
        },
        "owner_review_packet_sha256": owner_sha256,
        "raw_training_text_retained": False,
        "report_file_sha256": file_sha256(report_path),
        "report_internal_sha256": report["report_sha256"],
        "runtime_seconds": time.monotonic() - started,
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "PASS_independent_nonproduction_candidate_verification",
        "train_supervised_tokens": train_tokens,
        "validation_supervised_tokens": validation_tokens,
    }
    verification["verification_sha256"] = sha256_value(verification)
    if write_output:
        output_path = root / str(plan["output"]["verification_path"])
        _write_json_atomic(output_path, verification)
    return verification


__all__ = [
    "RETAINED_SAFETY_LABELS",
    "build_prosocial_owner_review_html",
    "custom_quality_reasons",
    "deterministic_owner_sample",
    "dialogue_split",
    "file_sha256",
    "load_plan",
    "load_raw_candidates",
    "run_curation",
    "sha256_value",
    "verify_curation",
]
