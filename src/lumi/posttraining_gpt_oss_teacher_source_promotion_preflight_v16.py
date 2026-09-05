"""Project the owner-accepted gpt-oss teacher canary into canonical SFT form."""

from __future__ import annotations

import copy
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .data_quality import BenchmarkRegistry
from .posttraining_candidate_mix_preflight import file_sha256, sha256_value
from .posttraining_data import filter_sft_records, load_sft_policy
from .posttraining_format import encode_sft_messages, load_chat_template
from .posttraining_sources import SFTSource, SFTSourceRegistry
from .posttraining_teacher_canary_source_promotion_preflight import (
    json_file_bytes,
    jsonl_file_bytes,
)

RAW_CANDIDATE_PATH = (
    "artifacts/lumi-gpt-oss-teacher-per-cell-recovery-v0.15/"
    "05f56935922e/attempt-001/private/candidate-rows.jsonl"
)
OWNER_REVIEW_PATH = (
    "artifacts/lumi-gpt-oss-teacher-per-cell-recovery-v0.15/"
    "05f56935922e/attempt-001/evidence/owner-review.json"
)
OWNER_VERIFICATION_PATH = (
    "artifacts/lumi-gpt-oss-teacher-per-cell-recovery-v0.15/"
    "05f56935922e/attempt-001/evidence/owner-review-verification.json"
)
RUN_REPORT_PATH = (
    "artifacts/lumi-gpt-oss-teacher-per-cell-recovery-v0.15/"
    "05f56935922e/attempt-001/evidence/run-report.json"
)
RUN_VERIFICATION_PATH = (
    "artifacts/lumi-gpt-oss-teacher-per-cell-recovery-v0.15/"
    "05f56935922e/attempt-001/evidence/run-verification.json"
)

EXPECTED_RAW_FILE_SHA256 = (
    "b6ac425ed995d94ee46255ddf674ca1b4510c3363e35c93ca138f871162413b6"
)
EXPECTED_OWNER_REVIEW_FILE_SHA256 = (
    "2151d94ecd702a5c82171628be2dd0de2a122f969a3df12d3a3cba33c5859b87"
)
EXPECTED_OWNER_VERIFICATION_FILE_SHA256 = (
    "27953ccea1a3a1d0e75d773686a4d932d73236d96eaaa75aa8f995f09cddf83e"
)
EXPECTED_RUN_REPORT_FILE_SHA256 = (
    "2423cfc3516cf347dd824cf83f1828caedaffc051c1d65dbc5e76548faeb6911"
)
EXPECTED_RUN_VERIFICATION_FILE_SHA256 = (
    "0f96ac5fe82315a17fb88e6236dd29b24314c411829a0f54e269022857af37f0"
)

SOURCE_ID = "lumi_first_party_gpt_oss_teacher_canary_v0_1"
DATASET_ID = "Afnan-Technologies-Inc/lumi-first-party-gpt-oss-teacher-canary-v0.1"
EXPECTED_RECORDS = 160
EXPECTED_TRAIN_RECORDS = 128
EXPECTED_VALIDATION_RECORDS = 32
SPLIT_CONTRACT = "scenario_variant_00_07_train_08_09_validation.v0.16"

LANGUAGE_MAP = {
    "banglish": "multilingual",
    "bengali_script": "bn",
    "english": "en",
    "natural_code_switching": "multilingual",
}
TASKS = (
    "factual_qa",
    "multilingual_bengali",
    "safety_helpfulness",
    "writing_summarization",
)

PLAN_SCHEMA = "lumi.sft-gpt-oss-teacher-source-promotion-preflight.v0.16"
REPORT_SCHEMA = "lumi.sft-gpt-oss-teacher-source-promotion-preflight-report.v0.16"
VERIFICATION_SCHEMA = (
    "lumi.sft-gpt-oss-teacher-source-promotion-preflight-verification.v0.16"
)
PLAN_STATUS = "local_projection_only_registry_v0_16_not_materialized"

EXPECTED_RAW_INTERNAL_SHA256 = (
    "4a308f5cf454305b9097c1172f91e5889b4074e464f90eae154b45de7589277e"
)
EXPECTED_CANONICAL_FILE_SHA256 = (
    "264e9fa3cc56e53ad77b9ebf4079fc4142ddb4d4f0b5b512f039535df7ba305b"
)
EXPECTED_CANONICAL_INTERNAL_SHA256 = (
    "3a28d9040a05cb806c2b3a48d901934c76c4dd399171231364519b02fbb1921e"
)
EXPECTED_CANONICAL_BYTES = 329_808
EXPECTED_PREDECESSOR_REGISTRY_SHA256 = (
    "de81a51f0288b4851ff5ea4ca0598ddcf4936b2bf2c4127ab2b1c0d6deb1a4de"
)
EXPECTED_SUCCESSOR_REGISTRY_SHA256 = (
    "23276446086ee9c68784e6d20e25705fa9516b4e0f913dc96d8abd1b6b443ba6"
)
EXPECTED_SUCCESSOR_REGISTRY_BYTES = 20_093
EXPECTED_TRAIN_TOKENS = 9_473
EXPECTED_VALIDATION_TOKENS = 1_906
EXPECTED_PROJECTED_MIX_TOKENS = 7_266_143
EXPECTED_PROJECTED_GAP = 2_733_857
EXPECTED_TOKEN_METRICS = {
    "by_split": {
        "train": {
            "masked_tokens": 4_694,
            "records": 128,
            "serialized_tokens": 14_167,
            "supervised_tokens": 9_473,
        },
        "validation": {
            "masked_tokens": 1_000,
            "records": 32,
            "serialized_tokens": 2_906,
            "supervised_tokens": 1_906,
        },
    },
    "train_by_language_mode": {
        "banglish": 1_648,
        "bengali_script": 3_560,
        "english": 1_454,
        "natural_code_switching": 2_811,
    },
    "train_by_task": {
        "factual_qa": 535,
        "multilingual_bengali": 3_082,
        "safety_helpfulness": 3_237,
        "writing_summarization": 2_619,
    },
}

EXPECTED_AUTHORIZATION = {
    "candidate_registry_change_authorized": False,
    "canonical_candidate_materialization_authorized": False,
    "data_contract_change_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "gpu_authorized": False,
    "network_authorized": False,
    "predecessor_registry_mutation_authorized": False,
    "production_promotion_authorized": False,
    "production_sft_authorized": False,
    "purpose": "exact_gpt_oss_teacher_candidate_and_registry_v0_16_projection_only",
    "source_registry_promotion_authorized": False,
    "unrelated_access_authorized": False,
}
EXPECTED_LIMITS = {
    "added_sources": 1,
    "candidate_bytes": EXPECTED_CANONICAL_BYTES,
    "candidate_records": EXPECTED_RECORDS,
    "maximum_network_bytes": 0,
    "maximum_private_write_bytes": 1_048_576,
    "maximum_runtime_minutes": 30,
    "predecessor_sources": 16,
    "registry_bytes": EXPECTED_SUCCESSOR_REGISTRY_BYTES,
    "successor_sources": 17,
    "train_records": EXPECTED_TRAIN_RECORDS,
    "train_supervised_tokens": EXPECTED_TRAIN_TOKENS,
    "validation_records": EXPECTED_VALIDATION_RECORDS,
    "validation_supervised_tokens": EXPECTED_VALIDATION_TOKENS,
}
EXPECTED_OUTPUT = {
    "canonical_candidate_path": (
        "artifacts/lumi-gpt-oss-teacher-per-cell-recovery-v0.15/"
        "05f56935922e/attempt-001/private/canonical-candidate-v0.16.jsonl"
    ),
    "preflight_report_path": (
        "evidence/lumi-4b-sft-gpt-oss-teacher-source-promotion-preflight-v0.16.json"
    ),
    "projected_successor_registry_path": (
        "configs/lumi-4b-sft-source-registry-v0.16.json"
    ),
    "verification_path": (
        "evidence/"
        "lumi-4b-sft-gpt-oss-teacher-source-promotion-preflight-verification-v0.16.json"
    ),
}

EXACT_AUTHORIZATION_TEXT = (
    "I approve one bounded local-only Lumi gpt-oss teacher candidate promotion into an immutable "
    "source-registry v0.16 successor. Bind the exact 160-row machine-verified candidate SHA-256 "
    "b6ac425ed995d94ee46255ddf674ca1b4510c3363e35c93ca138f871162413b6, owner-review SHA-256 "
    "2151d94ecd702a5c82171628be2dd0de2a122f969a3df12d3a3cba33c5859b87, owner-review verification "
    "SHA-256 27953ccea1a3a1d0e75d773686a4d932d73236d96eaaa75aa8f995f09cddf83e, run-report SHA-256 "
    "2423cfc3516cf347dd824cf83f1828caedaffc051c1d65dbc5e76548faeb6911, and run-verification "
    "SHA-256 0f96ac5fe82315a17fb88e6236dd29b24314c411829a0f54e269022857af37f0. Materialize only the "
    "exact canonical 160-row candidate projected by contract lumi.gpt-oss-teacher-canonical-owner-"
    "bound.v0.16, bound to SHA-256 264e9fa3cc56e53ad77b9ebf4079fc4142ddb4d4f0b5b512f039535df7ba305b "
    "and 329,808 bytes, using scenario-level split contract scenario_variant_00_07_train_08_09_"
    "validation.v0.16 with 128 train rows / 9,473 train supervised tokens and 32 validation rows / "
    "1,906 validation supervised tokens. Require all 160 rows to pass the frozen SFT schema, "
    "rights, provenance, secret, PII, benchmark-decontamination, 4K-fit, tokenizer, chat-template, "
    "and assistant-only masking gates. Preserve source registry v0.15 SHA-256 "
    "de81a51f0288b4851ff5ea4ca0598ddcf4936b2bf2c4127ab2b1c0d6deb1a4de byte-for-byte and append "
    "only source lumi_first_party_gpt_oss_teacher_canary_v0_1 to create immutable registry v0.16 "
    "SHA-256 23276446086ee9c68784e6d20e25705fa9516b4e0f913dc96d8abd1b6b443ba6, exactly 20,093 bytes "
    "and 17 sources. Maximum 30 local runtime-minutes / 1 MiB additional private writes / 0 network "
    "bytes / 0 GPU use. It may write only the bound canonical candidate, immutable registry v0.16, "
    "non-production promotion receipt, and independent verification. No predecessor mutation, "
    "external API, download, dataset seal, active data-contract change, production promotion, SFT "
    "training, unrelated access, remote access, instance action, or termination."
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSON object required at line {line_number}: {path}")
        values.append(value)
    return values


def _byte_encoder() -> dict[int, str]:
    values = [
        *range(ord("!"), ord("~") + 1),
        *range(ord("¡"), ord("¬") + 1),
        *range(ord("®"), ord("ÿ") + 1),
    ]
    codepoints = list(values)
    offset = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            codepoints.append(256 + offset)
            offset += 1
    return dict(zip(values, map(chr, codepoints), strict=True))


def _category(value: str) -> str:
    category = unicodedata.category(value)
    if category.startswith("L"):
        return "letter"
    if category.startswith("N"):
        return "number"
    if value.isspace():
        return "space"
    return "other"


def _byte_level_pretokens(text: str) -> list[str]:
    """Reproduce the frozen Hugging Face ByteLevel GPT-2 split expression."""

    values: list[str] = []
    index = 0
    contractions = ("'ll", "'re", "'ve", "'s", "'t", "'m", "'d")
    while index < len(text):
        contraction = next(
            (item for item in contractions if text.startswith(item, index)),
            None,
        )
        if contraction is not None:
            values.append(contraction)
            index += len(contraction)
            continue

        prefix = ""
        start = index
        if text[index] == " " and index + 1 < len(text):
            next_category = _category(text[index + 1])
            if next_category in {"letter", "number", "other"}:
                prefix = " "
                index += 1
        category = _category(text[index])
        if category in {"letter", "number", "other"}:
            end = index + 1
            while end < len(text) and _category(text[end]) == category:
                end += 1
            values.append(prefix + text[index:end])
            index = end
            continue
        if prefix:
            values.append(prefix)
            continue

        end = start + 1
        while end < len(text) and _category(text[end]) == "space":
            end += 1
        values.append(text[start:end])
        index = end
    return values


class PureByteLevelBPETokenizer:
    """Dependency-free reader for the exact frozen ByteLevel BPE tokenizer."""

    def __init__(self, tokenizer_path: Path) -> None:
        value = _read_json(tokenizer_path)
        model = value.get("model")
        if not isinstance(model, Mapping) or model.get("type") != "BPE":
            raise ValueError("frozen tokenizer is not ByteLevel BPE")
        vocab = model.get("vocab")
        merges = model.get("merges")
        if not isinstance(vocab, Mapping) or not isinstance(merges, list):
            raise ValueError("frozen tokenizer vocabulary or merges changed")
        self._vocab = {str(key): int(item) for key, item in vocab.items()}
        self._merges = {
            (str(pair[0]), str(pair[1])): rank
            for rank, pair in enumerate(merges)
            if isinstance(pair, list) and len(pair) == 2
        }
        self._bytes = _byte_encoder()
        self._cache: dict[str, tuple[str, ...]] = {}
        added = value.get("added_tokens")
        if not isinstance(added, list):
            raise ValueError("frozen tokenizer added tokens changed")
        self.all_special_ids = sorted(
            int(item["id"])
            for item in added
            if isinstance(item, Mapping) and item.get("special") is True
        )

    def get_vocab(self) -> Mapping[str, int]:
        return self._vocab

    def _bpe(self, word: str) -> tuple[str, ...]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        symbols = tuple(word)
        while len(symbols) > 1:
            pairs = {(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)}
            candidate = min(pairs, key=lambda pair: self._merges.get(pair, 10**12))
            if candidate not in self._merges:
                break
            first, second = candidate
            merged: list[str] = []
            index = 0
            while index < len(symbols):
                if (
                    index + 1 < len(symbols)
                    and symbols[index] == first
                    and symbols[index + 1] == second
                ):
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(symbols[index])
                    index += 1
            symbols = tuple(merged)
        self._cache[word] = symbols
        return symbols

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        split_special_tokens: bool,
    ) -> list[int]:
        if add_special_tokens or not split_special_tokens:
            raise ValueError("pure tokenizer only supports untrusted-text encoding")
        normalized = unicodedata.normalize("NFC", text)
        token_ids: list[int] = []
        for piece in _byte_level_pretokens(normalized):
            encoded = "".join(self._bytes[byte] for byte in piece.encode("utf-8"))
            for token in self._bpe(encoded):
                try:
                    token_ids.append(self._vocab[token])
                except KeyError as error:
                    raise ValueError("frozen tokenizer emitted an unknown BPE token") from error
        return token_ids


def _variant(record_id: str) -> int:
    try:
        value = int(record_id.rsplit("-", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError("gpt-oss record variant is invalid") from error
    if not 0 <= value <= 9:
        raise ValueError("gpt-oss record variant is outside 00-09")
    return value


def _split(record_id: str) -> str:
    return "validation" if _variant(record_id) >= 8 else "train"


def load_bound_candidate(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bindings = {
        RAW_CANDIDATE_PATH: EXPECTED_RAW_FILE_SHA256,
        OWNER_REVIEW_PATH: EXPECTED_OWNER_REVIEW_FILE_SHA256,
        OWNER_VERIFICATION_PATH: EXPECTED_OWNER_VERIFICATION_FILE_SHA256,
        RUN_REPORT_PATH: EXPECTED_RUN_REPORT_FILE_SHA256,
        RUN_VERIFICATION_PATH: EXPECTED_RUN_VERIFICATION_FILE_SHA256,
    }
    for relative, expected in bindings.items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"gpt-oss promotion binding changed: {relative}")
    rows = _read_jsonl(root / RAW_CANDIDATE_PATH)
    owner = _read_json(root / OWNER_REVIEW_PATH)
    owner_verification = _read_json(root / OWNER_VERIFICATION_PATH)
    run_report = _read_json(root / RUN_REPORT_PATH)
    run_verification = _read_json(root / RUN_VERIFICATION_PATH)
    if (
        len(rows) != EXPECTED_RECORDS
        or owner.get("status") != "ACCEPT_deterministic_32_row_owner_review_packet"
        or owner_verification.get("status") != "PASS_owner_acceptance_verified"
        or run_report.get("status")
        != "machine_verified_160_row_canary_pending_32_row_owner_review"
        or run_verification.get("status") != "PASS"
        or not all(run_report.get("machine_audit", {}).get("checks", {}).values())
    ):
        raise ValueError("gpt-oss accepted candidate evidence changed")
    return rows, owner


def source_metadata(source_revision: str) -> dict[str, Any]:
    return {
        "access": "first_party",
        "adapter": "lumi_first_party_gpt_oss_teacher_canary_canonical_owner_bound_v0.16",
        "blockers": [],
        "dataset_id": DATASET_ID,
        "evidence_urls": [],
        "languages": ["bn", "en", "multilingual"],
        "license_id": "lumi-owned",
        "origin": "synthetic",
        "production_blockers": [
            "candidate_is_only_160_records",
            "owner_quality_review_is_32_row_deterministic_sample_not_full_record_review",
            "full_mix_cross_source_reaudit_not_complete",
            "first_party_3_2m_closure_lane_not_generated",
            "production_dataset_seal_and_sft_authorization_not_complete",
        ],
        "production_eligible": False,
        "repository_url": None,
        "revision": source_revision,
        "source_id": SOURCE_ID,
        "splits": ["train", "validation"],
        "status": "candidate_eligible",
        "task_categories": list(TASKS),
    }


def project_canonical_candidate(
    rows: Sequence[Mapping[str, Any]], *, owner_review_file_sha256: str
) -> list[dict[str, Any]]:
    source_revision = sha256_value(list(rows))
    projected: list[dict[str, Any]] = []
    for row in rows:
        record_id = str(row["record_id"])
        language_mode = str(row["language_mode"])
        messages = copy.deepcopy(row["messages"])
        original = copy.deepcopy(row["provenance"])
        transform = {
            "contract": "lumi.gpt-oss-teacher-canonical-owner-bound.v0.16",
            "original_provenance_sha256": sha256_value(original),
            "original_rights_status": row["rights_status"],
            "owner_review_file_sha256": owner_review_file_sha256,
            "source_revision": source_revision,
            "split_contract": SPLIT_CONTRACT,
        }
        projected.append(
            {
                "language": LANGUAGE_MAP[language_mode],
                "messages": messages,
                "provenance": {
                    "ai_assisted": True,
                    "consent_basis": "first_party_rights_declaration",
                    "content_sha256": sha256_value(messages),
                    "human_authored": False,
                    "human_authorship_claimed": False,
                    "model_id": original["model_id"],
                    "model_revision": original["model_revision"],
                    "original_provenance_sha256": sha256_value(original),
                    "owner_review_claim": (
                        "ai_generated_plus_owner_reviewed_not_human_written"
                    ),
                    "owner_review_file_sha256": owner_review_file_sha256,
                    "owner_review_scope": "deterministic_32_row_spot_check",
                    "runtime_image_digest": original["runtime_image_digest"],
                    "self_hosted": True,
                    "transform_sha256": sha256_value(transform),
                },
                "quality": {
                    "pii_scan": "passed",
                    "review_status": "accepted",
                    "score": 0.95,
                    "secret_scan": "passed",
                },
                "record_id": record_id,
                "safety": {
                    "policy_version": "lumi-assistant-behavior-v0.1",
                    "risk_labels": [],
                },
                "schema_version": "lumi.sft.v0.1",
                "source": {
                    "dataset_id": DATASET_ID,
                    "document_id": record_id,
                    "license_id": "lumi-owned",
                    "revision": source_revision,
                    "source_id": SOURCE_ID,
                    "uri": (
                        "lumi-private://first-party-gpt-oss-teacher-canary/"
                        f"v0.1/{record_id}"
                    ),
                },
                "split": _split(record_id),
                "task_category": row["task"],
            }
        )
    return projected


def _token_metrics(
    raw_rows: Sequence[Mapping[str, Any]],
    canonical: Sequence[Mapping[str, Any]],
    *,
    tokenizer_path: Path,
    chat_template_path: Path,
) -> dict[str, Any]:
    tokenizer = PureByteLevelBPETokenizer(tokenizer_path)
    template = load_chat_template(chat_template_path)
    by_split: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "masked_tokens": 0,
            "records": 0,
            "serialized_tokens": 0,
            "supervised_tokens": 0,
        }
    )
    by_task: Counter[str] = Counter()
    by_mode: Counter[str] = Counter()
    for raw, row in zip(raw_rows, canonical, strict=True):
        encoded = encode_sft_messages(
            row["messages"], tokenizer=tokenizer, template=template
        )
        split = str(row["split"])
        by_split[split]["masked_tokens"] += encoded.masked_tokens
        by_split[split]["records"] += 1
        by_split[split]["serialized_tokens"] += len(encoded.input_ids)
        by_split[split]["supervised_tokens"] += encoded.supervised_tokens
        if split == "train":
            by_task[str(raw["task"])] += encoded.supervised_tokens
            by_mode[str(raw["language_mode"])] += encoded.supervised_tokens
    return {
        "by_split": dict(sorted(by_split.items())),
        "train_by_language_mode": dict(sorted(by_mode.items())),
        "train_by_task": dict(sorted(by_task.items())),
    }


def build_reference_projection(root: Path) -> dict[str, Any]:
    rows, owner = load_bound_candidate(root)
    source_revision = sha256_value(rows)
    canonical = project_canonical_candidate(
        rows, owner_review_file_sha256=EXPECTED_OWNER_REVIEW_FILE_SHA256
    )
    predecessor_path = root / "configs/lumi-4b-sft-source-registry-v0.15.json"
    predecessor = _read_json(predecessor_path)
    if any(
        source.get("source_id") == SOURCE_ID
        for source in predecessor.get("sources", [])
        if isinstance(source, Mapping)
    ):
        raise ValueError("gpt-oss teacher source already exists in registry v0.15")
    successor = copy.deepcopy(predecessor)
    successor["sources"].append(source_metadata(source_revision))

    contract_path = root / "configs/lumi-4b-sft-data-contract-v0.1.json"
    contract = _read_json(contract_path)
    allowed = {
        "licenses": frozenset(contract["allowed_licenses"]),
        "languages": frozenset(contract["allowed_languages"]),
        "splits": frozenset(contract["allowed_splits"]),
        "tasks": frozenset(contract["task_mix"]),
    }
    parsed = tuple(
        SFTSource.from_mapping(
            item,
            allowed_licenses=allowed["licenses"],
            allowed_languages=allowed["languages"],
            allowed_splits=allowed["splits"],
            allowed_tasks=allowed["tasks"],
        )
        for item in successor["sources"]
    )
    successor_bytes = json_file_bytes(successor)
    successor_sha256 = hashlib.sha256(successor_bytes).hexdigest()
    registry = SFTSourceRegistry(
        observed_at=str(successor["observed_at"]),
        registry_sha256=successor_sha256,
        sources=parsed,
    )
    quality = _read_json(root / "configs/data-quality-v0.3.json")
    accepted, filter_report = filter_sft_records(
        canonical,
        policy=load_sft_policy(contract_path),
        benchmark_registry=BenchmarkRegistry(
            quality["benchmark_decontamination"], root
        ),
        source_registry=registry,
    )
    token_metrics = _token_metrics(
        rows,
        canonical,
        tokenizer_path=(
            root.parent
            / "lumi-artifacts/4b-segment2-throughput-gate-v0.1/"
            "8bedfbc868e8/staging/input/tokenizer/tokenizer.json"
        ),
        chat_template_path=root / "configs/lumi-4b-chat-template-v0.1.json",
    )
    canonical_bytes = jsonl_file_bytes(canonical)
    train_tokens = token_metrics["by_split"]["train"]["supervised_tokens"]
    return {
        "canonical": canonical,
        "owner": owner,
        "projection": {
            "canonical_bytes": len(canonical_bytes),
            "canonical_file_sha256": hashlib.sha256(canonical_bytes).hexdigest(),
            "canonical_internal_sha256": sha256_value(canonical),
            "current_mix_train_supervised_tokens": 7_256_670,
            "filter_accepted": len(accepted),
            "filter_rejected": int(filter_report["rejected"]),
            "projected_gap_to_10m": 10_000_000 - (7_256_670 + train_tokens),
            "projected_mix_train_supervised_tokens": 7_256_670 + train_tokens,
            "raw_internal_sha256": source_revision,
            "records": len(rows),
            "registry_predecessor_sha256": file_sha256(predecessor_path),
            "successor_registry_bytes": len(successor_bytes),
            "successor_registry_sha256": successor_sha256,
            "successor_sources": len(successor["sources"]),
            "token_metrics": token_metrics,
        },
        "successor": successor,
    }


EXPECTED_BINDINGS = frozenset(
    {
        "candidate_rows",
        "chat_template",
        "data_contract",
        "data_quality",
        "implementation",
        "owner_review",
        "owner_review_verification",
        "plan_runner",
        "plan_verifier",
        "predecessor_registry",
        "run_report",
        "run_verification",
        "test",
        "tokenizer_config",
        "tokenizer_json",
    }
)


def resolve_binding(
    root: Path, specification: Mapping[str, Any], *, name: str
) -> Path:
    relative = specification.get("path")
    expected = specification.get("sha256")
    if (
        not isinstance(relative, str)
        or not isinstance(expected, str)
        or len(expected) != 64
    ):
        raise ValueError(f"invalid gpt-oss promotion binding: {name}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.parent.resolve()) or not path.is_file():
        raise ValueError(f"missing gpt-oss promotion binding: {name}")
    if file_sha256(path) != expected:
        raise ValueError(f"changed gpt-oss promotion binding: {name}")
    return path


def load_preflight_plan(path: Path, *, repository_root: Path) -> dict[str, Any]:
    plan = _read_json(path)
    expected = {
        "authorization": EXPECTED_AUTHORIZATION,
        "limits": EXPECTED_LIMITS,
        "observed_at": "2026-08-11",
        "output": EXPECTED_OUTPUT,
        "schema_version": PLAN_SCHEMA,
        "source_after": source_metadata(EXPECTED_RAW_INTERNAL_SHA256),
        "status": PLAN_STATUS,
    }
    for name, value in expected.items():
        if plan.get(name) != value:
            raise ValueError(f"gpt-oss promotion preflight changed: {name}")
    next_gate = plan.get("next_gate")
    if not isinstance(next_gate, Mapping) or next_gate.get(
        "exact_authorization_text"
    ) != EXACT_AUTHORIZATION_TEXT:
        raise ValueError("gpt-oss promotion authorization text changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("gpt-oss promotion binding set changed")
    for name, specification in bindings.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"invalid gpt-oss promotion binding: {name}")
        resolve_binding(repository_root, specification, name=name)
    exact = {
        "candidate_rows": EXPECTED_RAW_FILE_SHA256,
        "owner_review": EXPECTED_OWNER_REVIEW_FILE_SHA256,
        "owner_review_verification": EXPECTED_OWNER_VERIFICATION_FILE_SHA256,
        "predecessor_registry": EXPECTED_PREDECESSOR_REGISTRY_SHA256,
        "run_report": EXPECTED_RUN_REPORT_FILE_SHA256,
        "run_verification": EXPECTED_RUN_VERIFICATION_FILE_SHA256,
    }
    for name, expected_sha256 in exact.items():
        if bindings[name]["sha256"] != expected_sha256:
            raise ValueError(f"gpt-oss exact binding changed: {name}")
    return plan


def _validate_projection(projection: Mapping[str, Any]) -> None:
    expected = {
        "canonical_bytes": EXPECTED_CANONICAL_BYTES,
        "canonical_file_sha256": EXPECTED_CANONICAL_FILE_SHA256,
        "canonical_internal_sha256": EXPECTED_CANONICAL_INTERNAL_SHA256,
        "current_mix_train_supervised_tokens": 7_256_670,
        "filter_accepted": EXPECTED_RECORDS,
        "filter_rejected": 0,
        "projected_gap_to_10m": EXPECTED_PROJECTED_GAP,
        "projected_mix_train_supervised_tokens": EXPECTED_PROJECTED_MIX_TOKENS,
        "raw_internal_sha256": EXPECTED_RAW_INTERNAL_SHA256,
        "records": EXPECTED_RECORDS,
        "registry_predecessor_sha256": EXPECTED_PREDECESSOR_REGISTRY_SHA256,
        "successor_registry_bytes": EXPECTED_SUCCESSOR_REGISTRY_BYTES,
        "successor_registry_sha256": EXPECTED_SUCCESSOR_REGISTRY_SHA256,
        "successor_sources": 17,
        "token_metrics": EXPECTED_TOKEN_METRICS,
    }
    if dict(projection) != expected:
        raise ValueError("gpt-oss promotion reference projection changed")


def build_preflight_report(
    plan: Mapping[str, Any], *, plan_file_sha256: str, repository_root: Path
) -> dict[str, Any]:
    result = build_reference_projection(repository_root)
    projection = result["projection"]
    _validate_projection(projection)
    bindings = plan["bindings"]
    resolved = {
        name: resolve_binding(repository_root, specification, name=name)
        for name, specification in bindings.items()
    }
    canonical_output = (repository_root / EXPECTED_OUTPUT["canonical_candidate_path"]).resolve()
    registry_output = (
        repository_root / EXPECTED_OUTPUT["projected_successor_registry_path"]
    ).resolve()
    if canonical_output.exists() or registry_output.exists():
        raise ValueError("gpt-oss promotion output was already materialized")
    report: dict[str, Any] = {
        "authorization": dict(EXPECTED_AUTHORIZATION),
        "bindings": {
            name: {"path": bindings[name]["path"], "sha256": file_sha256(path)}
            for name, path in sorted(resolved.items())
        },
        "checks": {
            "all_160_rows_pass_frozen_candidate_filters": True,
            "canonical_candidate_not_materialized": True,
            "candidate_and_owner_evidence_hashes_verified": True,
            "dependency_free_tokenizer_matches_frozen_reference": True,
            "gpu_network_and_external_api_unused": True,
            "human_authorship_not_claimed": True,
            "predecessor_registry_unchanged": True,
            "registry_v0_16_not_materialized": True,
            "scenario_level_split_prevents_cross_language_variant_leakage": True,
        },
        "next_gate": dict(plan["next_gate"]),
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "production": {
            "candidate_registry_changed": False,
            "canonical_candidate_materialized": False,
            "data_contract_changed": False,
            "dataset_sealed": False,
            "production_eligible": False,
            "sft_started": False,
        },
        "projection": copy.deepcopy(dict(projection)),
        "schema_version": REPORT_SCHEMA,
        "source_after": copy.deepcopy(plan["source_after"]),
        "status": "verified_local_projection_registry_v0_16_not_materialized",
    }
    report["report_sha256"] = sha256_value(report)
    validate_preflight_report(report)
    return report


def validate_preflight_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("unsupported gpt-oss promotion report")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if report.get("report_sha256") != sha256_value(unsigned):
        raise ValueError("gpt-oss promotion report seal changed")
    _validate_projection(report.get("projection", {}))
    checks = report.get("checks")
    if not isinstance(checks, Mapping) or not all(checks.values()):
        raise ValueError("gpt-oss promotion checks failed")
    production = report.get("production")
    if not isinstance(production, Mapping) or any(production.values()):
        raise ValueError("gpt-oss preflight crossed a production boundary")
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    for forbidden in ('"messages"', '"content"', '"assistant_response"'):
        if forbidden in serialized:
            raise ValueError("gpt-oss promotion report retained raw training text")


def build_preflight_verification(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    report_file_sha256: str,
    replay: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    validate_preflight_report(report)
    validate_preflight_report(replay)
    if report != replay:
        raise ValueError("gpt-oss promotion preflight replay changed")
    for output_name in ("canonical_candidate_path", "projected_successor_registry_path"):
        if (repository_root / plan["output"][output_name]).exists():
            raise ValueError("gpt-oss promotion output unexpectedly exists")
    verification: dict[str, Any] = {
        "checks": {
            "all_bindings_reverified": True,
            "all_candidate_filters_replayed": True,
            "canonical_candidate_hash_and_bytes_reproduced": True,
            "independent_projection_byte_exact": True,
            "next_gate_requires_exact_authorization": True,
            "no_gpu_network_registry_seal_or_sft_action": True,
            "raw_training_text_absent": True,
            "scenario_level_split_and_token_counts_reproduced": True,
            "successor_registry_hash_and_bytes_reproduced": True,
        },
        "preflight_report_file_sha256": report_file_sha256,
        "preflight_report_internal_sha256": report["report_sha256"],
        "projected_canonical_candidate_file_sha256": (
            EXPECTED_CANONICAL_FILE_SHA256
        ),
        "projected_successor_registry_sha256": (
            EXPECTED_SUCCESSOR_REGISTRY_SHA256
        ),
        "schema_version": VERIFICATION_SCHEMA,
        "status": "PASS_local_projection_registry_v0_16_not_materialized",
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification


__all__ = [
    "DATASET_ID",
    "EXPECTED_RECORDS",
    "EXPECTED_TRAIN_RECORDS",
    "EXPECTED_VALIDATION_RECORDS",
    "PureByteLevelBPETokenizer",
    "SOURCE_ID",
    "SPLIT_CONTRACT",
    "build_reference_projection",
    "build_preflight_report",
    "build_preflight_verification",
    "load_preflight_plan",
    "project_canonical_candidate",
    "source_metadata",
    "validate_preflight_report",
]
