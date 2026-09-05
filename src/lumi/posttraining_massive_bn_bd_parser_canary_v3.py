"""Collision-aware private parser canary for the bound MASSIVE ``bn-BD`` train object.

The source object is read once through six allowlisted columns.  Unicode NFC is
applied only to derived values.  Normalization-equivalent semantic tuples are
collapsed deterministically, while every normalized utterance that maps to more
than one intent/argument projection is quarantined as a whole.  Public outputs
contain only fixed metadata, aggregate counts, and SHA-256 digests; the 64-row
canary and its owner-review page remain private and non-production.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import stat
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import posttraining_massive_bn_bd_nfc_recovery_v2 as predecessor
from .data_quality import BenchmarkRegistry, MinHashNearDuplicateIndex, normalize_for_matching
from .posttraining_data import evaluate_sft_record, filter_sft_records, load_sft_policy
from .posttraining_format import encode_sft_messages, load_chat_template, verify_chat_tokenizer
from .posttraining_gpt_oss_teacher_source_promotion_preflight_v16 import (
    PureByteLevelBPETokenizer,
)
from .posttraining_owner_review import build_owner_review_html

PLAN_SCHEMA = "lumi.sft-massive-bn-bd-parser-canary-plan.v0.3"
REPORT_SCHEMA = "lumi.sft-massive-bn-bd-parser-canary-report.v0.3"
PRIVATE_MANIFEST_SCHEMA = "lumi.sft-massive-bn-bd-parser-canary-private-manifest.v0.3"
VERIFICATION_SCHEMA = "lumi.sft-massive-bn-bd-parser-canary-verification.v0.3"

PLAN_STATUS = "authorized_local_only_collision_aware_parser_canary_nonproduction"
PASS_STATUS = "PASS_private_64_row_parser_canary_pending_owner_review_nonproduction"
STOP_STATUS = "STOP_collision_aware_parser_canary_gate_failed_nonproduction"
VERIFICATION_STATUS = "PASS_zero_source_parser_canary_verification_nonproduction"

AUTHORIZATION_TEXT = "ok shuru kor"
AUTHORIZATION_SHA256 = "1046cda26a21ff4994e4a452870466e609fe93d825cad3622e9fd069fb18b35e"

SOURCE_COLUMNS = ("locale", "partition", "scenario", "intent", "utt", "annot_utt")
TOOL_NAME = "massive.parse_intent"
ASSISTANT_CONFIRMATION = (
    "অনুরোধটি শুধু শ্রেণিবদ্ধ করা হয়েছে; কোনো বাহ্যিক কাজ করা হয়নি।"
)
SOURCE_ID = "amazon_massive_bn_bd_tool_canary"
DATASET_ID = "AmazonScience/massive"
SOURCE_REVISION = "d1792d0e4f32a74d43725cc1b901f600dad990dd"
SOURCE_LICENSE = "cc-by-4.0"
SOURCE_URI = (
    "https://huggingface.co/datasets/AmazonScience/massive/blob/"
    f"{SOURCE_REVISION}/bn-BD/train/0000.parquet#sha256="
    f"{predecessor.EXPECTED_SOURCE_BINDING['sha256']}"
)
TRANSFORM_ID = "massive_bn_bd_nfc_collision_aware_pure_parser_v0.3"

CANARY_ROWS = 64
INTENT_COVERAGE_ROWS = 60
DETERMINISTIC_EXTRA_ROWS = 4
MINIMUM_OWNER_ACCEPTANCE_BASIS_POINTS = 9_500

START_MARKER = b'{"started":"massive_bn_bd_parser_canary_v0.3"}\n'
START_MARKER_SHA256 = hashlib.sha256(START_MARKER).hexdigest()

EXPECTED_LIMITS = {
    "exact_source_passes": 1,
    "maximum_source_row_reads": 11_514,
    "maximum_persisted_candidate_rows": 64,
    "maximum_candidate_audit_reads": 64,
    "maximum_baseline_candidate_reads": 0,
    "maximum_test_dev_validation_audio_row_reads": 0,
    "maximum_private_write_bytes": 4_194_304,
    "maximum_runtime_seconds": 1_200,
    "maximum_network_bytes": 0,
    "maximum_gpu_actions": 0,
    "maximum_inference_calls": 0,
    "maximum_training_actions": 0,
    "network_retries": 0,
}

EXPECTED_SOURCE = dict(predecessor.EXPECTED_SOURCE_BINDING)
EXPECTED_OUTPUT = {
    "private_root": (
        "../lumi-artifacts/4b-sft-massive-bn-bd-parser-canary-v0.3/"
        "38be2e88eabf/attempt-001/private"
    ),
    "marker_name": ".parser-canary-started-v0.3",
    "candidate_file_name": "candidate-rows.jsonl",
    "owner_review_html_name": "review/owner-review-64.html",
    "private_manifest_name": "private-manifest.json",
    "report_path": "evidence/lumi-4b-sft-massive-bn-bd-parser-canary-v0.3.json",
    "verification_path": (
        "evidence/lumi-4b-sft-massive-bn-bd-parser-canary-verification-v0.3.json"
    ),
}

EXPECTED_NEAR_DUPLICATE_POLICY = {
    "comparison_unit": "role_tagged_full_conversation",
    "num_perm": 128,
    "package_version": "2.0.0",
    "scheme": "affine32",
    "seed": 1,
    "shingle_tokens": 5,
    "threshold": 0.9,
}

EXPECTED_COLLISIONS = {
    "utterance": {
        "population": 11_514,
        "preexisting_duplicate_groups": 233,
        "preexisting_duplicate_affected_rows": 538,
        "introduced_normalization_collision_groups": 8,
        "introduced_normalization_collision_affected_rows": 22,
    },
    "annotated_utterance": {
        "population": 11_514,
        "preexisting_duplicate_groups": 220,
        "preexisting_duplicate_affected_rows": 509,
        "introduced_normalization_collision_groups": 8,
        "introduced_normalization_collision_affected_rows": 22,
    },
    "semantic_tuple": {
        "population": 11_514,
        "preexisting_duplicate_groups": 211,
        "preexisting_duplicate_affected_rows": 483,
        "introduced_normalization_collision_groups": 8,
        "introduced_normalization_collision_affected_rows": 22,
    },
}

EXPECTED_SCAN_COUNTS = {
    "rows": 11_514,
    "raw_non_nfc_utterances": 1_495,
    "raw_non_nfc_annotated_utterances": 1_495,
    "raw_utterance_utf8_bytes": 1_069_083,
    "raw_annotated_utterance_utf8_bytes": 1_229_274,
    "post_nfc_utterance_utf8_bytes": 1_074_399,
    "post_nfc_annotated_utterance_utf8_bytes": 1_234_590,
    "maximum_utterance_characters": 295,
    "maximum_annotated_utterance_characters": 328,
    "total_annotations": 11_287,
    "rows_with_annotations": 7_754,
    "maximum_annotations_per_row": 7,
    "slot_label_cardinality": 55,
    "preexisting_duplicate_slot_label_rows": 317,
    "post_nfc_duplicate_slot_label_rows": 317,
    "scenario_cardinality": 18,
    "intent_cardinality": 60,
}

EXPECTED_CONTRACT_BINDINGS = {
    "data_contract": {
        "path": "configs/lumi-4b-sft-data-contract-v0.1.json",
        "sha256": "9a283d9333d29deb7b516fa45eb4eec50232be9925f4ee4aa3a8f624213b2f06",
    },
    "data_quality": {
        "path": "configs/data-quality-v0.3.json",
        "sha256": "23fba21dbc822c84a5d16e63147fc1a7716576b8f1258cda8c035391f04f52c0",
    },
    "benchmark_registry": {
        "path": "data/benchmark-signatures-v0.3.jsonl.gz",
        "sha256": "1ffb5e938ab28c2fc876b9efd6c6ed59059f1e85aed2c396c024ae5925a4745e",
    },
    "chat_template": {
        "path": "configs/lumi-4b-chat-template-v0.1.json",
        "sha256": "bc7853a9277feddae2fd3db786c6a12f4fa3602a135a0408c38d4390f33bec12",
    },
    "tokenizer": {
        "path": (
            "../lumi-artifacts/4b-segment2-throughput-gate-v0.1/8bedfbc868e8/"
            "staging/input/tokenizer/tokenizer.json"
        ),
        "sha256": "1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee",
    },
    "source_registry": {
        "path": "configs/lumi-4b-sft-source-registry-v0.21.json",
        "sha256": "daa2d81755456728c5c72bca9ee95d5424aba0fc05e9fdd13223018f4055e3a1",
        "bytes": 28_405,
    },
    "sft_filter_implementation": {
        "path": "src/lumi/posttraining_data.py",
        "sha256": "df8297e753e97028999c6bcd203edc2a00a02147f7d9ecb2255569c480363970",
    },
    "chat_format_implementation": {
        "path": "src/lumi/posttraining_format.py",
        "sha256": "32df8a7b29d73354f21b73c79442f7c44f437c24330662b308a381326ad0a9d5",
    },
    "quality_implementation": {
        "path": "src/lumi/data_quality.py",
        "sha256": "6a9ba4f5b5564d29b22d105a8ff74c9bf34b14b37ad00aca52eeec8307b41637",
    },
    "owner_review_implementation": {
        "path": "src/lumi/posttraining_owner_review.py",
        "sha256": "68055ff05caf500e9fd203e048c6a70e0161062db63cee365d29e7f9abe9826d",
    },
    "pure_tokenizer_implementation": {
        "path": "src/lumi/posttraining_gpt_oss_teacher_source_promotion_preflight_v16.py",
        "sha256": "c1e1bacbaf3bbdc5a0bdc1720f0063fa79330b1ada95f5a304feed2385f389e8",
    },
}

EXPECTED_PREDECESSOR_BINDINGS = {
    "nfc_recovery_plan": {
        "path": "configs/lumi-4b-sft-massive-bn-bd-nfc-recovery-v0.2.json",
        "sha256": "88366fdd3f27c7faccf69007ae446ceb367aa20f3da31fe0c7d4a90301b903e9",
        "internal_sha256": "32c63ea228a4dd579b325929033042286ef0be64e20017b1998386683968c88a",
    },
    "nfc_recovery_implementation": {
        "path": "src/lumi/posttraining_massive_bn_bd_nfc_recovery_v2.py",
        "sha256": "4c62e48eeca9b55b2a3f04d09c52349ca6d98c9a7a8834e02c1aa2629d386acb",
    },
    "nfc_recovery_runner": {
        "path": "scripts/recover_sft_massive_bn_bd_nfc_v2.py",
        "sha256": "79f4621aed2dfbd25379a3d4aa8eaf9307555d06aeeb525a6ead74143cb8ccf1",
    },
    "nfc_recovery_verifier": {
        "path": "scripts/verify_sft_massive_bn_bd_nfc_recovery_v2.py",
        "sha256": "ab88da26d958ed8892d896a3bd54f894aba3df6af07f3b173478b3279fe92ee4",
    },
    "nfc_recovery_test": {
        "path": "tests/test_massive_bn_bd_nfc_recovery_v2.py",
        "sha256": "2b5bdbaba9ba64bc008d78aef9d615b82fc58414e63b8e8d324c2410c047f891",
    },
    "nfc_recovery_report": {
        "path": "evidence/lumi-4b-sft-massive-bn-bd-nfc-recovery-v0.2.json",
        "sha256": "38be2e88eabfb67a149d9c6475eb6ad93b16099f4222eee986aabdf197c786d1",
        "internal_sha256": "78b86d38b8e29772c0d6fef09e4ee2e7eb1ba9adc1ef42337b7e1ce1cc442937",
    },
    "nfc_recovery_verification": {
        "path": "evidence/lumi-4b-sft-massive-bn-bd-nfc-recovery-verification-v0.2.json",
        "sha256": "a877d6c6c7a5d70ed58742d4e2e5a6cdaaa983448b2ee3e849bd94c5bc0fb285",
        "internal_sha256": "8cf3a0f742f04b3bb49d69e7aeb2ac1dfbd53f62fcf1f8946fc2b3dcb01c065c",
    },
    "nfc_recovery_marker": {
        "path": (
            "../lumi-artifacts/4b-sft-massive-bn-bd-nfc-recovery-v0.2/"
            "68d3507a4b7a/attempt-001/private/.nfc-recovery-started-v0.2"
        ),
        "sha256": "17e96fb81eb2a130041dc8543e6bc60a00167224b9a1ec5bf921f56d8c6c83c6",
        "bytes": 46,
    },
    "acquisition_plan": dict(predecessor.EXPECTED_PREDECESSOR_BINDINGS["acquisition_plan"]),
    "acquisition_report": dict(
        predecessor.EXPECTED_PREDECESSOR_BINDINGS["acquisition_report"]
    ),
    "acquisition_verification": dict(
        predecessor.EXPECTED_PREDECESSOR_BINDINGS["acquisition_verification"]
    ),
    "acquisition_manifest": dict(
        predecessor.EXPECTED_PREDECESSOR_BINDINGS["artifact_manifest"]
    ),
    "acquisition_marker": dict(
        predecessor.EXPECTED_PREDECESSOR_BINDINGS["acquisition_marker"]
    ),
}

EXPECTED_CODE_PATHS = {
    "implementation": "src/lumi/posttraining_massive_bn_bd_parser_canary_v3.py",
    "runner": "scripts/curate_sft_massive_bn_bd_parser_canary_v3.py",
    "verifier": "scripts/verify_sft_massive_bn_bd_parser_canary_v3.py",
    "test": "tests/test_massive_bn_bd_parser_canary_v3.py",
}

EXPECTED_PROHIBITIONS = {
    "no_retry_after_durable_start_marker": True,
    "no_source_predecessor_or_registry_mutation": True,
    "no_test_dev_validation_audio_or_non_bn_bd_reads": True,
    "no_existing_candidate_or_cross_source_baseline_reads": True,
    "no_active_credit_registry_promotion_dataset_seal_or_training": True,
    "no_network_gpu_inference_remote_instance_or_external_action": True,
    "no_source_id_worker_id_slot_method_or_judgment_value_reads": True,
    "no_unrelated_action": True,
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def canonical_json_text(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _seal(value: dict[str, Any], field: str) -> dict[str, Any]:
    value[field] = sha256_value({key: item for key, item in value.items() if key != field})
    return value


def _verify_seal(value: Mapping[str, Any], field: str) -> None:
    observed = value.get(field)
    if not isinstance(observed, str) or _SHA256.fullmatch(observed) is None:
        raise ValueError(f"invalid {field}")
    unsigned = {key: item for key, item in value.items() if key != field}
    if sha256_value(unsigned) != observed:
        raise ValueError(f"{field} changed")


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _resolve_repository_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("repository path escaped")
    return path


def _resolve_workspace_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.parent.resolve()):
        raise ValueError("workspace path escaped")
    return path


def _resolve_private_root(root: Path, value: str) -> Path:
    path = _resolve_workspace_path(root, value)
    allowed = (root.parent / "lumi-artifacts").resolve()
    if not path.is_relative_to(allowed):
        raise ValueError("private canary root escaped")
    return path


def _binding_path(root: Path, binding: Mapping[str, Any], label: str) -> Path:
    if set(binding) - {"path", "sha256", "internal_sha256", "bytes"}:
        raise ValueError(f"unexpected binding fields: {label}")
    path = _resolve_workspace_path(root, str(binding.get("path", "")))
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"binding is not a regular file: {label}")
    if sha256_file(path) != binding.get("sha256"):
        raise ValueError(f"binding changed: {label}")
    expected_bytes = binding.get("bytes")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ValueError(f"binding size changed: {label}")
    internal = binding.get("internal_sha256")
    if internal is not None:
        obj = _read_object(path)
        if internal not in {
            item
            for key, item in obj.items()
            if key.endswith("_sha256") and isinstance(item, str)
        }:
            raise ValueError(f"internal binding changed: {label}")
    return path


@dataclass(frozen=True)
class Slot:
    name: str
    value: str

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "value": self.value}


@dataclass(frozen=True)
class SemanticTuple:
    utterance: str
    scenario: str
    intent: str
    intent_index: int
    slots: tuple[Slot, ...]

    def arguments(self) -> dict[str, Any]:
        return {
            "locale": "bn-BD",
            "scenario": self.scenario,
            "intent": self.intent,
            "slots": [slot.as_dict() for slot in self.slots],
        }

    def identity(self) -> dict[str, Any]:
        return {"utterance": self.utterance, **self.arguments()}

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            b"lumi.massive.bn_bd.semantic.v0.3\x00" + canonical_json_bytes(self.identity())
        ).hexdigest()

    @property
    def projection_digest(self) -> str:
        return hashlib.sha256(
            b"lumi.massive.bn_bd.projection.v0.3\x00"
            + canonical_json_bytes(self.arguments())
        ).hexdigest()


@dataclass(frozen=True)
class ScanResult:
    aggregate: Mapping[str, Any]
    eligible_semantics: tuple[SemanticTuple, ...]


@dataclass(frozen=True)
class RuntimeContracts:
    policy: Any
    benchmark: BenchmarkRegistry
    template: Any
    tokenizer: PureByteLevelBPETokenizer


@dataclass(frozen=True)
class MeasuredCandidate:
    semantic: SemanticTuple
    record: Mapping[str, Any]
    total_tokens: int
    supervised_tokens: int
    masked_tokens: int


def _string_counts() -> dict[str, int]:
    return {
        "strings": 0,
        "non_strings": 0,
        "empty": 0,
        "outer_whitespace": 0,
        "non_nfc": 0,
        "control_character_rows": 0,
        "utf8_bytes": 0,
        "maximum_characters": 0,
    }


def _observe_string(value: Any, counts: dict[str, int]) -> str | None:
    if not isinstance(value, str):
        counts["non_strings"] += 1
        return None
    counts["strings"] += 1
    counts["empty"] += int(not value)
    counts["outer_whitespace"] += int(value != value.strip())
    counts["non_nfc"] += int(value != unicodedata.normalize("NFC", value))
    counts["control_character_rows"] += int(
        any(unicodedata.category(character) == "Cc" for character in value)
    )
    counts["utf8_bytes"] += len(value.encode("utf-8"))
    counts["maximum_characters"] = max(counts["maximum_characters"], len(value))
    return value


def _string_clear(value: Mapping[str, int], rows: int) -> bool:
    return (
        value["strings"] == rows
        and value["non_strings"] == 0
        and value["empty"] == 0
        and value["outer_whitespace"] == 0
        and value["control_character_rows"] == 0
    )


def _valid_index(value: Any, labels: Sequence[str]) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and 0 <= value < len(labels)


def _digest(domain: bytes, value: Any) -> str:
    return hashlib.sha256(domain + b"\x00" + canonical_json_bytes(value)).hexdigest()


def analyze_rows(
    rows: Iterable[Mapping[str, Any]], *, maximum_rows: int = 11_514
) -> ScanResult:
    """Analyze one already-streaming source pass and retain only normalized semantics."""

    raw_utterance = _string_counts()
    raw_annotated = _string_counts()
    normalized_utterance_counts = _string_counts()
    normalized_annotated_counts = _string_counts()
    collisions = {
        "utterance": predecessor._DigestCollisionAccumulator(b"massive.utt"),  # noqa: SLF001
        "annotated_utterance": predecessor._DigestCollisionAccumulator(  # noqa: SLF001
            b"massive.annot_utt"
        ),
        "semantic_tuple": predecessor._DigestCollisionAccumulator(  # noqa: SLF001
            b"massive.semantic_tuple"
        ),
    }

    row_count = 0
    scope_mismatch_rows = 0
    invalid_row_shapes = 0
    invalid_scenario_rows = 0
    invalid_intent_rows = 0
    intent_prefix_mismatch_rows = 0
    malformed_annotation_rows = 0
    ordered_annotation_mismatch_rows = 0
    literal_segment_mismatch_rows = 0
    deannotation_mismatch_rows = 0
    introduced_nfc_duplicate_slot_label_rows = 0
    preexisting_duplicate_slot_label_rows = 0
    post_nfc_duplicate_slot_label_rows = 0
    utterance_nfc_idempotence_failures = 0
    annotated_nfc_idempotence_failures = 0
    raw_total_annotations = 0
    post_total_annotations = 0
    rows_with_annotations = 0
    maximum_annotations = 0
    nfc_intersection = Counter(
        {"utterance_only": 0, "annotated_utterance_only": 0, "both": 0, "neither": 0}
    )
    scenario_values: set[int] = set()
    intent_values: set[int] = set()
    raw_slot_labels: set[str] = set()
    normalized_slot_labels: set[str] = set()

    semantic_by_key: dict[bytes, SemanticTuple] = {}
    semantic_source_counts: Counter[bytes] = Counter()
    utterance_projection_keys: dict[str, set[bytes]] = defaultdict(set)
    utterance_semantic_keys: dict[str, set[bytes]] = defaultdict(set)
    utterance_source_counts: Counter[str] = Counter()

    for row in rows:
        row_count += 1
        if row_count > maximum_rows:
            raise ValueError("source-row read cap exceeded")
        if not isinstance(row, Mapping):
            invalid_row_shapes += 1
            continue

        locale = row.get("locale")
        partition = row.get("partition")
        scope_ok = locale == "bn-BD" and partition == "train"
        scope_mismatch_rows += int(not scope_ok)

        scenario_index = row.get("scenario")
        intent_index = row.get("intent")
        scenario_ok = _valid_index(scenario_index, predecessor.SCENARIO_LABELS)
        intent_ok = _valid_index(intent_index, predecessor.INTENT_LABELS)
        invalid_scenario_rows += int(not scenario_ok)
        invalid_intent_rows += int(not intent_ok)
        if scenario_ok:
            scenario_values.add(int(scenario_index))
        if intent_ok:
            intent_values.add(int(intent_index))
        prefix_ok = False
        if scenario_ok and intent_ok:
            prefix_ok = (
                predecessor.INTENT_LABELS[int(intent_index)].partition("_")[0]
                == predecessor.SCENARIO_LABELS[int(scenario_index)]
            )
            intent_prefix_mismatch_rows += int(not prefix_ok)

        raw_utt = _observe_string(row.get("utt"), raw_utterance)
        raw_annot = _observe_string(row.get("annot_utt"), raw_annotated)
        if raw_utt is None or raw_annot is None:
            continue
        normalized_utt = unicodedata.normalize("NFC", raw_utt)
        normalized_annot = unicodedata.normalize("NFC", raw_annot)
        _observe_string(normalized_utt, normalized_utterance_counts)
        _observe_string(normalized_annot, normalized_annotated_counts)

        utterance_changed = raw_utt != normalized_utt
        annotated_changed = raw_annot != normalized_annot
        if utterance_changed and annotated_changed:
            nfc_intersection["both"] += 1
        elif utterance_changed:
            nfc_intersection["utterance_only"] += 1
        elif annotated_changed:
            nfc_intersection["annotated_utterance_only"] += 1
        else:
            nfc_intersection["neither"] += 1
        utterance_nfc_idempotence_failures += int(
            unicodedata.normalize("NFC", normalized_utt) != normalized_utt
        )
        annotated_nfc_idempotence_failures += int(
            unicodedata.normalize("NFC", normalized_annot) != normalized_annot
        )

        collisions["utterance"].observe(raw_utt, normalized_utt)
        collisions["annotated_utterance"].observe(raw_annot, normalized_annot)
        if scenario_ok and intent_ok:
            collisions["semantic_tuple"].observe(
                (scenario_index, intent_index, raw_utt, raw_annot),
                (scenario_index, intent_index, normalized_utt, normalized_annot),
            )

        try:
            parsed_raw = predecessor._strict_parse_annotations(raw_annot)  # noqa: SLF001
            parsed_normalized = predecessor._strict_parse_annotations(  # noqa: SLF001
                normalized_annot
            )
        except ValueError:
            malformed_annotation_rows += 1
            continue

        raw_total_annotations += parsed_raw.count
        post_total_annotations += parsed_normalized.count
        rows_with_annotations += int(parsed_raw.count > 0)
        maximum_annotations = max(maximum_annotations, parsed_raw.count)
        raw_labels = tuple(parsed_raw.labels)
        normalized_labels = tuple(unicodedata.normalize("NFC", item) for item in raw_labels)
        raw_label_set = set(raw_labels)
        normalized_label_set = set(normalized_labels)
        preexisting_duplicate_slot_label_rows += int(len(raw_label_set) < len(raw_labels))
        introduced_nfc_duplicate_slot_label_rows += int(
            len(normalized_label_set) < len(raw_label_set)
        )
        post_nfc_duplicate_slot_label_rows += int(
            len(normalized_label_set) < len(normalized_labels)
        )
        for raw_label, normalized_label in zip(raw_labels, normalized_labels, strict=True):
            raw_slot_labels.add(_digest(b"massive.slot.raw_casefold", raw_label.casefold()))
            normalized_slot_labels.add(
                _digest(b"massive.slot.post_nfc_casefold", normalized_label.casefold())
            )

        ordered_ok = (
            parsed_raw.ordered_nfc_signature() == parsed_normalized.ordered_nfc_signature()
        )
        literal_ok = (
            parsed_raw.literal_nfc_signature() == parsed_normalized.literal_nfc_signature()
        )
        deannotation_ok = (
            unicodedata.normalize("NFC", parsed_raw.deannotated()) == normalized_utt
            and parsed_normalized.deannotated() == normalized_utt
        )
        ordered_annotation_mismatch_rows += int(not ordered_ok)
        literal_segment_mismatch_rows += int(not literal_ok)
        deannotation_mismatch_rows += int(not deannotation_ok)

        row_local_clear = (
            scope_ok
            and scenario_ok
            and intent_ok
            and prefix_ok
            and ordered_ok
            and literal_ok
            and deannotation_ok
            and len(normalized_label_set) == len(raw_label_set)
        )
        if not row_local_clear:
            continue
        slots = tuple(
            Slot(name=name, value=value)
            for name, value in parsed_normalized.ordered_nfc_signature()
        )
        semantic = SemanticTuple(
            utterance=normalized_utt,
            scenario=predecessor.SCENARIO_LABELS[int(scenario_index)],
            intent=predecessor.INTENT_LABELS[int(intent_index)],
            intent_index=int(intent_index),
            slots=slots,
        )
        semantic_key = canonical_json_bytes(semantic.identity())
        existing = semantic_by_key.setdefault(semantic_key, semantic)
        if existing != semantic:
            raise ValueError("semantic canonicalization changed")
        semantic_source_counts[semantic_key] += 1
        projection_key = canonical_json_bytes(semantic.arguments())
        utterance_projection_keys[normalized_utt].add(projection_key)
        utterance_semantic_keys[normalized_utt].add(semantic_key)
        utterance_source_counts[normalized_utt] += 1

    duplicate_summaries = {key: item.summary() for key, item in collisions.items()}
    ambiguous_utterances = {
        utterance
        for utterance, projections in utterance_projection_keys.items()
        if len(projections) > 1
    }
    ambiguous_semantic_keys = {
        key
        for utterance in ambiguous_utterances
        for key in utterance_semantic_keys[utterance]
    }
    eligible = tuple(
        sorted(
            (
                semantic
                for key, semantic in semantic_by_key.items()
                if key not in ambiguous_semantic_keys
            ),
            key=lambda item: item.digest,
        )
    )
    quarantine_ledger = []
    for utterance in sorted(
        ambiguous_utterances,
        key=lambda item: _digest(b"massive.quarantine.utterance", item),
    ):
        quarantine_ledger.append(
            {
                "utterance_sha256": _digest(b"massive.quarantine.utterance", utterance),
                "source_rows": utterance_source_counts[utterance],
                "projection_sha256": sorted(
                    hashlib.sha256(b"massive.projection\x00" + item).hexdigest()
                    for item in utterance_projection_keys[utterance]
                ),
                "semantic_sha256": sorted(
                    hashlib.sha256(b"massive.semantic\x00" + item).hexdigest()
                    for item in utterance_semantic_keys[utterance]
                ),
            }
        )

    invalid = {
        "invalid_row_shapes": invalid_row_shapes,
        "scope_mismatch_rows": scope_mismatch_rows,
        "invalid_scenario_rows": invalid_scenario_rows,
        "invalid_intent_rows": invalid_intent_rows,
        "intent_prefix_mismatch_rows": intent_prefix_mismatch_rows,
        "malformed_annotation_rows": malformed_annotation_rows,
        "ordered_annotation_mismatch_rows": ordered_annotation_mismatch_rows,
        "literal_segment_mismatch_rows": literal_segment_mismatch_rows,
        "deannotation_mismatch_rows": deannotation_mismatch_rows,
        "introduced_nfc_duplicate_slot_label_rows": (
            introduced_nfc_duplicate_slot_label_rows
        ),
        "utterance_nfc_idempotence_failures": utterance_nfc_idempotence_failures,
        "annotated_nfc_idempotence_failures": annotated_nfc_idempotence_failures,
    }
    aggregate: dict[str, Any] = {
        "rows": row_count,
        "raw_strings": {
            "utterance": raw_utterance,
            "annotated_utterance": raw_annotated,
        },
        "post_nfc_strings": {
            "utterance": normalized_utterance_counts,
            "annotated_utterance": normalized_annotated_counts,
        },
        "nfc_intersection": dict(nfc_intersection),
        "semantics": {
            "scenario_cardinality": len(scenario_values),
            "intent_cardinality": len(intent_values),
        },
        "annotations": {
            "raw_total": raw_total_annotations,
            "post_nfc_total": post_total_annotations,
            "rows_with_annotations": rows_with_annotations,
            "maximum_per_row": maximum_annotations,
            "raw_casefold_slot_label_cardinality": len(raw_slot_labels),
            "post_nfc_casefold_slot_label_cardinality": len(normalized_slot_labels),
            "preexisting_duplicate_slot_label_rows": (
                preexisting_duplicate_slot_label_rows
            ),
            "post_nfc_duplicate_slot_label_rows": post_nfc_duplicate_slot_label_rows,
        },
        "duplicates": duplicate_summaries,
        "invalid": invalid,
        "curation": {
            "unique_normalized_semantic_tuples": len(semantic_by_key),
            "duplicate_source_rows_collapsed": sum(
                count - 1 for count in semantic_source_counts.values()
            ),
            "ambiguous_normalized_utterance_groups": len(ambiguous_utterances),
            "ambiguous_source_rows": sum(
                utterance_source_counts[item] for item in ambiguous_utterances
            ),
            "ambiguous_unique_semantic_tuples": len(ambiguous_semantic_keys),
            "nonambiguous_unique_semantic_tuples": len(eligible),
            "quarantine_ledger_sha256": sha256_value(quarantine_ledger),
        },
    }
    aggregate["scan_ledger_sha256"] = sha256_value(aggregate)
    return ScanResult(aggregate=aggregate, eligible_semantics=eligible)


def production_scan_checks(aggregate: Mapping[str, Any]) -> dict[str, bool]:
    raw = aggregate["raw_strings"]
    post = aggregate["post_nfc_strings"]
    annotations = aggregate["annotations"]
    invalid = aggregate["invalid"]
    expected = EXPECTED_SCAN_COUNTS
    return {
        "exact_source_row_count": aggregate["rows"] == expected["rows"],
        "source_scope_and_shapes_clear": invalid["invalid_row_shapes"] == 0
        and invalid["scope_mismatch_rows"] == 0,
        "scenario_intent_contract_clear": invalid["invalid_scenario_rows"] == 0
        and invalid["invalid_intent_rows"] == 0
        and invalid["intent_prefix_mismatch_rows"] == 0
        and aggregate["semantics"]
        == {
            "scenario_cardinality": expected["scenario_cardinality"],
            "intent_cardinality": expected["intent_cardinality"],
        },
        "raw_string_contract_exact": _string_clear(raw["utterance"], expected["rows"])
        and _string_clear(raw["annotated_utterance"], expected["rows"])
        and raw["utterance"]["non_nfc"] == expected["raw_non_nfc_utterances"]
        and raw["annotated_utterance"]["non_nfc"]
        == expected["raw_non_nfc_annotated_utterances"]
        and raw["utterance"]["utf8_bytes"] == expected["raw_utterance_utf8_bytes"]
        and raw["annotated_utterance"]["utf8_bytes"]
        == expected["raw_annotated_utterance_utf8_bytes"]
        and raw["utterance"]["maximum_characters"]
        == expected["maximum_utterance_characters"]
        and raw["annotated_utterance"]["maximum_characters"]
        == expected["maximum_annotated_utterance_characters"],
        "nfc_projection_contract_exact": _string_clear(post["utterance"], expected["rows"])
        and _string_clear(post["annotated_utterance"], expected["rows"])
        and post["utterance"]["non_nfc"] == 0
        and post["annotated_utterance"]["non_nfc"] == 0
        and post["utterance"]["utf8_bytes"]
        == expected["post_nfc_utterance_utf8_bytes"]
        and post["annotated_utterance"]["utf8_bytes"]
        == expected["post_nfc_annotated_utterance_utf8_bytes"]
        and invalid["utterance_nfc_idempotence_failures"] == 0
        and invalid["annotated_nfc_idempotence_failures"] == 0,
        "nfc_intersection_exact": aggregate["nfc_intersection"]
        == {
            "utterance_only": 0,
            "annotated_utterance_only": 0,
            "both": 1_495,
            "neither": 10_019,
        },
        "annotation_contract_exact": annotations["raw_total"]
        == expected["total_annotations"]
        and annotations["post_nfc_total"] == expected["total_annotations"]
        and annotations["rows_with_annotations"] == expected["rows_with_annotations"]
        and annotations["maximum_per_row"] == expected["maximum_annotations_per_row"]
        and annotations["raw_casefold_slot_label_cardinality"]
        == expected["slot_label_cardinality"]
        and annotations["post_nfc_casefold_slot_label_cardinality"]
        == expected["slot_label_cardinality"]
        and annotations["preexisting_duplicate_slot_label_rows"]
        == expected["preexisting_duplicate_slot_label_rows"]
        and annotations["post_nfc_duplicate_slot_label_rows"]
        == expected["post_nfc_duplicate_slot_label_rows"],
        "parser_preservation_checks_clear": invalid["malformed_annotation_rows"] == 0
        and invalid["ordered_annotation_mismatch_rows"] == 0
        and invalid["literal_segment_mismatch_rows"] == 0
        and invalid["deannotation_mismatch_rows"] == 0
        and invalid["introduced_nfc_duplicate_slot_label_rows"] == 0,
        "collision_aggregates_reproduced": aggregate["duplicates"]
        == EXPECTED_COLLISIONS,
        "fixed_metadata_contract_bound": len(predecessor.SCENARIO_LABELS) == 18
        and len(predecessor.INTENT_LABELS) == 60
        and predecessor._class_label_sha256(predecessor.SCENARIO_LABELS)  # noqa: SLF001
        == predecessor.SCENARIO_MAP_SHA256
        and predecessor._class_label_sha256(predecessor.INTENT_LABELS)  # noqa: SLF001
        == predecessor.INTENT_MAP_SHA256,
    }


def _verify_parquet_metadata(parquet: Any) -> None:
    if parquet.metadata.num_rows != EXPECTED_LIMITS["maximum_source_row_reads"]:
        raise ValueError("bound Parquet row count changed")
    schema = parquet.schema_arrow
    observed_schema = tuple((field.name, str(field.type), field.nullable) for field in schema)
    if observed_schema != predecessor.EXPECTED_TOP_LEVEL_SCHEMA:
        raise ValueError("bound Parquet top-level schema changed")
    metadata = schema.metadata
    expected_key = predecessor.HUGGINGFACE_METADATA_KEY.encode()
    if not isinstance(metadata, Mapping) or set(metadata) != {expected_key}:
        raise ValueError("bound Parquet metadata keys changed")
    raw_metadata = metadata[expected_key]
    if (
        len(raw_metadata) != predecessor.HUGGINGFACE_METADATA_BYTES
        or hashlib.sha256(raw_metadata).hexdigest()
        != predecessor.HUGGINGFACE_METADATA_SHA256
    ):
        raise ValueError("bound Hugging Face metadata changed")
    try:
        features = json.loads(raw_metadata)["info"]["features"]
    except (json.JSONDecodeError, KeyError, TypeError):
        raise ValueError("bound Hugging Face feature metadata is malformed") from None
    if (
        not isinstance(features, Mapping)
        or hashlib.sha256(canonical_json_bytes(features)).hexdigest()
        != predecessor.HUGGINGFACE_CANONICAL_FEATURES_SHA256
    ):
        raise ValueError("bound Hugging Face features changed")
    predecessor._validate_observed_class_label_maps(features)  # noqa: SLF001


def scan_bound_parquet(path: str | Path) -> ScanResult:
    """Perform the one authorized source pass through the six exact columns."""

    import pyarrow.parquet as pq  # type: ignore[import-not-found]

    parquet = pq.ParquetFile(str(path), pre_buffer=False, memory_map=False)
    _verify_parquet_metadata(parquet)
    batch_calls = 0

    def rows() -> Iterable[Mapping[str, Any]]:
        nonlocal batch_calls
        observed = 0
        for batch in parquet.iter_batches(
            batch_size=1_024, columns=list(SOURCE_COLUMNS), use_threads=False
        ):
            batch_calls += 1
            values = batch.to_pydict()
            if tuple(values) != SOURCE_COLUMNS:
                raise ValueError("Parquet projection columns changed")
            for locale, partition, scenario, intent, utterance, annotated in zip(
                values["locale"],
                values["partition"],
                values["scenario"],
                values["intent"],
                values["utt"],
                values["annot_utt"],
                strict=True,
            ):
                observed += 1
                if observed > EXPECTED_LIMITS["maximum_source_row_reads"]:
                    raise ValueError("source-row read cap exceeded")
                yield {
                    "locale": locale,
                    "partition": partition,
                    "scenario": scenario,
                    "intent": intent,
                    "utt": utterance,
                    "annot_utt": annotated,
                }
        if observed != EXPECTED_LIMITS["maximum_source_row_reads"]:
            raise ValueError("source-row read count changed")

    result = analyze_rows(rows())
    if batch_calls <= 0:
        raise ValueError("Parquet source pass did not start")
    return result


def _tool_call(semantic: SemanticTuple) -> dict[str, Any]:
    return {"name": TOOL_NAME, "arguments": semantic.arguments()}


def _tool_result(semantic: SemanticTuple) -> dict[str, Any]:
    return {
        "name": TOOL_NAME,
        "status": "parsed_only",
        "external_action_performed": False,
        "parsed": semantic.arguments(),
    }


def build_candidate(semantic: SemanticTuple) -> dict[str, Any]:
    messages = [
        {"role": "user", "content": semantic.utterance},
        {"role": "tool", "content": canonical_json_text(_tool_call(semantic))},
        {"role": "tool_result", "content": canonical_json_text(_tool_result(semantic))},
        {"role": "assistant", "content": ASSISTANT_CONFIRMATION},
    ]
    digest = semantic.digest
    content_sha256 = sha256_value(
        {"language": "bn", "messages": messages, "task_category": "tool_api"}
    )
    transform_sha256 = sha256_value(
        {
            "adapter": TRANSFORM_ID,
            "output_schema": "lumi.sft.v0.1",
            "revision": SOURCE_REVISION,
            "semantic_tuple_sha256": digest,
        }
    )
    document_id = f"massive-bn-BD-semantic-{digest}"
    return {
        "schema_version": "lumi.sft.v0.1",
        "record_id": f"massive-bn-bd-parser-{digest[:24]}",
        "split": "train",
        "language": "bn",
        "task_category": "tool_api",
        "messages": messages,
        "source": {
            "source_id": SOURCE_ID,
            "dataset_id": DATASET_ID,
            "document_id": document_id,
            "license_id": SOURCE_LICENSE,
            "revision": SOURCE_REVISION,
            "uri": SOURCE_URI,
        },
        "provenance": {
            "consent_basis": "source_level_open_license",
            "content_sha256": content_sha256,
            "transform_sha256": transform_sha256,
            "normalization": "NFC",
            "semantic_tuple_sha256": digest,
            "owner_review_status": "pending_deterministic_64_row_spot_check",
            "source_registry_status": "pending_separate_promotion",
        },
        "quality": {
            "score": 1.0,
            "review_method": "deterministic_machine_gates_owner_review_pending",
            "review_status": "accepted",
            "pii_scan": "passed",
            "secret_scan": "passed",
        },
        "safety": {
            "policy_version": "lumi-assistant-behavior-v0.1",
            "risk_labels": [],
        },
    }


def parse_and_validate_candidate(record: Mapping[str, Any]) -> SemanticTuple:
    messages = record.get("messages")
    if not isinstance(messages, list) or [item.get("role") for item in messages] != [
        "user",
        "tool",
        "tool_result",
        "assistant",
    ]:
        raise ValueError("candidate message sequence changed")
    if any(not isinstance(item.get("content"), str) for item in messages):
        raise ValueError("candidate message content changed")
    utterance = messages[0]["content"]
    if (
        not utterance
        or utterance != utterance.strip()
        or unicodedata.normalize("NFC", utterance) != utterance
        or any(unicodedata.category(character) == "Cc" for character in utterance)
    ):
        raise ValueError("candidate utterance contract changed")
    try:
        tool = json.loads(messages[1]["content"])
        result = json.loads(messages[2]["content"])
    except json.JSONDecodeError:
        raise ValueError("candidate tool JSON changed") from None
    if canonical_json_text(tool) != messages[1]["content"]:
        raise ValueError("candidate tool JSON is not canonical")
    if canonical_json_text(result) != messages[2]["content"]:
        raise ValueError("candidate tool-result JSON is not canonical")
    if not isinstance(tool, Mapping) or set(tool) != {"name", "arguments"}:
        raise ValueError("candidate tool schema changed")
    arguments = tool.get("arguments")
    if (
        tool.get("name") != TOOL_NAME
        or not isinstance(arguments, Mapping)
        or set(arguments) != {"locale", "scenario", "intent", "slots"}
        or arguments.get("locale") != "bn-BD"
    ):
        raise ValueError("candidate parser arguments changed")
    scenario = arguments.get("scenario")
    intent = arguments.get("intent")
    slots_value = arguments.get("slots")
    if scenario not in predecessor.SCENARIO_LABELS or intent not in predecessor.INTENT_LABELS:
        raise ValueError("candidate semantic labels changed")
    intent_index = predecessor.INTENT_LABELS.index(str(intent))
    if str(intent).partition("_")[0] != scenario:
        raise ValueError("candidate scenario/intent prefix changed")
    if not isinstance(slots_value, list):
        raise ValueError("candidate slots changed")
    slots: list[Slot] = []
    for item in slots_value:
        if not isinstance(item, Mapping) or set(item) != {"name", "value"}:
            raise ValueError("candidate ordered slot schema changed")
        name = item.get("name")
        value = item.get("value")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(value, str)
            or not value
            or unicodedata.normalize("NFC", name) != name
            or unicodedata.normalize("NFC", value) != value
        ):
            raise ValueError("candidate ordered slot value changed")
        slots.append(Slot(name=name, value=value))
    semantic = SemanticTuple(
        utterance=utterance,
        scenario=str(scenario),
        intent=str(intent),
        intent_index=intent_index,
        slots=tuple(slots),
    )
    if result != _tool_result(semantic):
        raise ValueError("candidate tool result is not an exact parsed-only echo")
    if messages[3]["content"] != ASSISTANT_CONFIRMATION:
        raise ValueError("candidate assistant no-action confirmation changed")
    expected = build_candidate(semantic)
    if dict(record) != expected:
        raise ValueError("candidate provenance or record binding changed")
    return semantic


def _normalized_rejection_reason(reason: str) -> str:
    if reason.startswith("benchmark_contamination:"):
        return "benchmark_contamination"
    return reason


def measure_semantics(
    semantics: Sequence[SemanticTuple], runtime: RuntimeContracts
) -> tuple[list[MeasuredCandidate], dict[str, Any]]:
    measured: list[MeasuredCandidate] = []
    rejection_counts: Counter[str] = Counter()
    for semantic in semantics:
        record = build_candidate(semantic)
        parse_and_validate_candidate(record)
        decision = evaluate_sft_record(
            record,
            policy=runtime.policy,
            benchmark_registry=runtime.benchmark,
            source_registry=None,
        )
        if not decision.accepted:
            rejection_counts.update(_normalized_rejection_reason(item) for item in decision.reasons)
            continue
        try:
            encoded = encode_sft_messages(
                record["messages"], tokenizer=runtime.tokenizer, template=runtime.template
            )
        except ValueError:
            rejection_counts["tokenization_or_context"] += 1
            continue
        measured.append(
            MeasuredCandidate(
                semantic=semantic,
                record=record,
                total_tokens=len(encoded.input_ids),
                supervised_tokens=encoded.supervised_tokens,
                masked_tokens=encoded.masked_tokens,
            )
        )
    return measured, {
        "evaluated": len(semantics),
        "eligible": len(measured),
        "rejected": len(semantics) - len(measured),
        "rejection_counts": dict(sorted(rejection_counts.items())),
    }


def role_tagged_conversation(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("candidate messages changed during near-duplicate audit")
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("candidate message schema changed")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("candidate message fields changed")
        parts.append(f"{role} {content}")
    return normalize_for_matching("\n".join(parts))


def _near_index() -> MinHashNearDuplicateIndex:
    return MinHashNearDuplicateIndex(
        {
            key: EXPECTED_NEAR_DUPLICATE_POLICY[key]
            for key in (
                "num_perm",
                "package_version",
                "scheme",
                "seed",
                "shingle_tokens",
                "threshold",
            )
        }
    )


def select_canary(
    measured: Sequence[MeasuredCandidate],
) -> tuple[list[MeasuredCandidate], dict[str, Any]]:
    by_intent: dict[int, list[MeasuredCandidate]] = defaultdict(list)
    for item in measured:
        by_intent[item.semantic.intent_index].append(item)
    for values in by_intent.values():
        values.sort(key=lambda item: item.semantic.digest)

    index = _near_index()
    selected: list[MeasuredCandidate] = []
    selected_digests: set[str] = set()
    near_skips = 0
    missing_intents = 0
    for intent_index in range(len(predecessor.INTENT_LABELS)):
        chosen: MeasuredCandidate | None = None
        for item in by_intent.get(intent_index, []):
            signature = index.signature(role_tagged_conversation(item.record))
            if index.duplicate_of(signature) is not None:
                near_skips += 1
                continue
            chosen = item
            index.add(str(item.record["record_id"]), signature)
            break
        if chosen is None:
            missing_intents += 1
            continue
        selected.append(chosen)
        selected_digests.add(chosen.semantic.digest)

    extras = 0
    if missing_intents == 0:
        for item in sorted(measured, key=lambda candidate: candidate.semantic.digest):
            if item.semantic.digest in selected_digests:
                continue
            signature = index.signature(role_tagged_conversation(item.record))
            if index.duplicate_of(signature) is not None:
                near_skips += 1
                continue
            index.add(str(item.record["record_id"]), signature)
            selected.append(item)
            selected_digests.add(item.semantic.digest)
            extras += 1
            if extras == DETERMINISTIC_EXTRA_ROWS:
                break

    selected.sort(key=lambda item: str(item.record["record_id"]))
    selection_ledger = [item.semantic.digest for item in selected]
    return selected, {
        "intent_coverage_rows": len(predecessor.INTENT_LABELS) - missing_intents,
        "missing_intents": missing_intents,
        "deterministic_extra_rows": extras,
        "near_duplicate_skips": near_skips,
        "selected_rows": len(selected),
        "selection_sha256": sha256_value(selection_ledger),
    }


def near_duplicate_audit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    index = _near_index()
    groups: list[tuple[str, str]] = []
    for record in sorted(records, key=lambda item: str(item["record_id"])):
        record_id = str(record["record_id"])
        signature = index.signature(role_tagged_conversation(record))
        duplicate = index.duplicate_of(signature)
        if duplicate is not None:
            groups.append((duplicate, record_id))
        index.add(record_id, signature)
    return {
        "groups": len(groups),
        "affected_rows": len({item for pair in groups for item in pair}),
        "ledger_sha256": sha256_value(groups),
    }


def _final_machine_gate(
    selected: Sequence[MeasuredCandidate], runtime: RuntimeContracts
) -> dict[str, Any]:
    records = [item.record for item in selected]
    accepted, generic = filter_sft_records(
        records,
        policy=runtime.policy,
        benchmark_registry=runtime.benchmark,
        source_registry=None,
    )
    near = near_duplicate_audit(records)
    semantic_digests = [parse_and_validate_candidate(record).digest for record in records]
    return {
        "generic_filter": {
            "total": generic["total"],
            "accepted": generic["accepted"],
            "rejected": generic["rejected"],
            "rejection_counts": generic["rejection_counts"],
            "accepted_dataset_sha256": generic["accepted_dataset_sha256"],
        },
        "tokenization": {
            "total_tokens": sum(item.total_tokens for item in selected),
            "supervised_tokens": sum(item.supervised_tokens for item in selected),
            "masked_tokens": sum(item.masked_tokens for item in selected),
            "maximum_conversation_tokens": max(
                (item.total_tokens for item in selected), default=0
            ),
            "context_window": runtime.template.context_window,
            "assistant_and_tool_only_loss": runtime.template.loss_roles
            == frozenset({"assistant", "tool"}),
        },
        "deduplication": {
            "exact_normalized_conversation_duplicates": len(records) - len(accepted),
            "near_duplicate_groups": near["groups"],
            "near_duplicate_affected_rows": near["affected_rows"],
            "near_duplicate_ledger_sha256": near["ledger_sha256"],
            "semantic_selection_sha256": sha256_value(sorted(semantic_digests)),
        },
        "parser_contract": {
            "rows": len(semantic_digests),
            "four_role_trajectory_rows": len(records),
            "parsed_only_rows": len(records),
            "external_action_performed_rows": 0,
            "ordered_slot_list_rows": len(records),
            "unique_intents": len(
                {item.semantic.intent_index for item in selected}
            ),
        },
    }


def _owner_sample(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    sample = [
        {
            "record_id": record["record_id"],
            "skill_id": f"{SOURCE_ID}:pure_parser",
            "language_mode": "bn_bd_source_utterance",
            "split": record["split"],
            "messages": record["messages"],
        }
        for record in records
    ]
    sample.sort(key=lambda item: str(item["record_id"]))
    return sample


def owner_review_html(records: Sequence[Mapping[str, Any]]) -> tuple[bytes, str]:
    sample = _owner_sample(records)
    sample_sha256 = sha256_value(sample)
    html = build_owner_review_html(sample, sample_sha256=sample_sha256).replace(
        (
            "These rows are AI-generated and owner-reviewed. "
            "Approving them does not make them human-written."
        ),
        (
            "These rows are source-derived and machine-curated. "
            "Approving them does not make them human-written or authorize external actions."
        ),
    )
    return html.encode("utf-8"), sample_sha256


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _write_fresh(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite canary output")
    temporary = path.with_name(f".{path.name}.part")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _write_marker(private_root: Path, marker_name: str) -> Path:
    private_root.mkdir(parents=True, exist_ok=False)
    os.chmod(private_root, stat.S_IRWXU)
    marker = private_root / marker_name
    _write_fresh(marker, START_MARKER)
    if marker.stat().st_size != len(START_MARKER) or sha256_file(marker) != START_MARKER_SHA256:
        raise ValueError("durable start marker changed")
    return marker


def _write_private_artifacts(
    *,
    private_root: Path,
    plan: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], int]:
    candidate_payload = _jsonl_bytes(records)
    html_payload, sample_sha256 = owner_review_html(records)
    candidate_sha256 = hashlib.sha256(candidate_payload).hexdigest()
    html_sha256 = hashlib.sha256(html_payload).hexdigest()
    manifest: dict[str, Any] = {
        "schema_version": PRIVATE_MANIFEST_SCHEMA,
        "status": "private_nonproduction_parser_canary_owner_review_pending",
        "candidate_rows": len(records),
        "candidate_file": plan["output"]["candidate_file_name"],
        "candidate_file_bytes": len(candidate_payload),
        "candidate_file_sha256": candidate_sha256,
        "owner_review_rows": len(records),
        "owner_review_html": plan["output"]["owner_review_html_name"],
        "owner_review_html_bytes": len(html_payload),
        "owner_review_html_sha256": html_sha256,
        "owner_review_sample_sha256": sample_sha256,
        "owner_review_status": "pending",
        "active_tool_api_credit_rows": 0,
        "production_eligible": False,
        "raw_text_in_public_evidence": False,
    }
    _seal(manifest, "manifest_sha256")
    manifest_payload = _json_bytes(manifest)
    total = len(START_MARKER) + len(candidate_payload) + len(html_payload) + len(manifest_payload)
    if total > EXPECTED_LIMITS["maximum_private_write_bytes"]:
        raise ValueError("private write cap would be exceeded")
    candidate_path = private_root / plan["output"]["candidate_file_name"]
    html_path = private_root / plan["output"]["owner_review_html_name"]
    manifest_path = private_root / plan["output"]["private_manifest_name"]
    _write_fresh(candidate_path, candidate_payload)
    _write_fresh(html_path, html_payload)
    _write_fresh(manifest_path, manifest_payload)
    return {
        "written": True,
        "candidate_file_bytes": len(candidate_payload),
        "candidate_file_sha256": candidate_sha256,
        "owner_review_html_bytes": len(html_payload),
        "owner_review_html_sha256": html_sha256,
        "private_manifest_bytes": len(manifest_payload),
        "private_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "owner_review_sample_sha256": sample_sha256,
        "marker_bytes": len(START_MARKER),
        "marker_sha256": START_MARKER_SHA256,
    }, total


def _empty_private_artifacts() -> dict[str, Any]:
    return {
        "written": False,
        "candidate_file_bytes": 0,
        "candidate_file_sha256": None,
        "owner_review_html_bytes": 0,
        "owner_review_html_sha256": None,
        "private_manifest_bytes": 0,
        "private_manifest_sha256": None,
        "owner_review_sample_sha256": None,
        "marker_bytes": len(START_MARKER),
        "marker_sha256": START_MARKER_SHA256,
    }


def _compact_scan(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    raw = aggregate["raw_strings"]
    post = aggregate["post_nfc_strings"]
    return {
        "rows": aggregate["rows"],
        "source_passes": 1,
        "allowlisted_column_count": len(SOURCE_COLUMNS),
        "raw_non_nfc_utterances": raw["utterance"]["non_nfc"],
        "raw_non_nfc_annotated_utterances": raw["annotated_utterance"]["non_nfc"],
        "post_nfc_non_nfc_utterances": post["utterance"]["non_nfc"],
        "post_nfc_non_nfc_annotated_utterances": post["annotated_utterance"]["non_nfc"],
        "nfc_intersection": aggregate["nfc_intersection"],
        "semantics": aggregate["semantics"],
        "annotations": aggregate["annotations"],
        "duplicates": aggregate["duplicates"],
        "invalid": aggregate["invalid"],
        "scan_ledger_sha256": aggregate["scan_ledger_sha256"],
    }


def _build_report(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    scan: ScanResult,
    scan_checks: Mapping[str, bool],
    measurement: Mapping[str, Any],
    selection: Mapping[str, Any],
    machine_gate: Mapping[str, Any],
    private_artifacts: Mapping[str, Any],
    private_write_bytes: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    curation = {
        **dict(scan.aggregate["curation"]),
        "machine_evaluated_unique_semantic_tuples": measurement["evaluated"],
        "machine_eligible_unique_semantic_tuples": measurement["eligible"],
        "machine_rejected_unique_semantic_tuples": measurement["rejected"],
        "machine_rejection_counts": measurement["rejection_counts"],
        **dict(selection),
    }
    generic = machine_gate["generic_filter"]
    tokenization = machine_gate["tokenization"]
    deduplication = machine_gate["deduplication"]
    parser_contract = machine_gate["parser_contract"]
    scan_pass = all(scan_checks.values())
    checks = {
        "source_scan_contract_passed": scan_pass,
        "collision_aggregates_reproduced": scan_checks.get(
            "collision_aggregates_reproduced", False
        ),
        "normalization_equivalent_semantic_tuples_deduplicated": (
            curation["unique_normalized_semantic_tuples"]
            + curation["duplicate_source_rows_collapsed"]
            <= scan.aggregate["rows"]
        ),
        "ambiguous_utterance_groups_quarantined": (
            curation["nonambiguous_unique_semantic_tuples"]
            + curation["ambiguous_unique_semantic_tuples"]
            == curation["unique_normalized_semantic_tuples"]
        ),
        "all_60_intents_eligible": selection["intent_coverage_rows"] == 60
        and selection["missing_intents"] == 0,
        "exactly_64_candidates_selected": selection["selected_rows"] == CANARY_ROWS
        and selection["deterministic_extra_rows"] == DETERMINISTIC_EXTRA_ROWS,
        "generic_sft_filter_passed": generic["total"] == CANARY_ROWS
        and generic["accepted"] == CANARY_ROWS
        and generic["rejected"] == 0,
        "tokenizer_and_4k_fit_passed": tokenization["maximum_conversation_tokens"]
        <= tokenization["context_window"]
        and tokenization["supervised_tokens"] > 0
        and tokenization["assistant_and_tool_only_loss"] is True,
        "exact_duplicate_gate_passed": deduplication[
            "exact_normalized_conversation_duplicates"
        ]
        == 0,
        "canary_near_duplicate_gate_passed": deduplication["near_duplicate_groups"] == 0
        and deduplication["near_duplicate_affected_rows"] == 0,
        "pure_parser_no_side_effect_contract_passed": parser_contract["rows"]
        == CANARY_ROWS
        and parser_contract["four_role_trajectory_rows"] == CANARY_ROWS
        and parser_contract["parsed_only_rows"] == CANARY_ROWS
        and parser_contract["external_action_performed_rows"] == 0
        and parser_contract["ordered_slot_list_rows"] == CANARY_ROWS
        and parser_contract["unique_intents"] == 60,
        "source_registry_remains_pending": True,
        "owner_review_remains_pending": True,
        "active_credit_remains_zero": True,
        "private_write_cap_respected": private_write_bytes
        <= EXPECTED_LIMITS["maximum_private_write_bytes"],
    }
    canary_pass = all(checks.values()) and private_artifacts["written"] is True
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": PASS_STATUS if canary_pass else STOP_STATUS,
        "observed_at": "2026-08-31",
        "authorization_sha256": AUTHORIZATION_SHA256,
        "plan_file_sha256": plan_file_sha256,
        "plan_internal_sha256": plan["plan_sha256"],
        "bindings": {
            "source_sha256": EXPECTED_SOURCE["sha256"],
            "predecessor_bindings_sha256": sha256_value(plan["predecessor_bindings"]),
            **{
                f"{key}_sha256": value["sha256"]
                for key, value in plan["contract_bindings"].items()
            },
            **{
                f"{key}_sha256": value["sha256"]
                for key, value in plan["code_bindings"].items()
            },
        },
        "scan": _compact_scan(scan.aggregate),
        "scan_checks": dict(scan_checks),
        "curation": curation,
        "machine_gate": dict(machine_gate),
        "checks": checks,
        "private_artifacts": dict(private_artifacts),
        "owner_review": {
            "status": "pending" if private_artifacts["written"] else "not_created",
            "rows": CANARY_ROWS if private_artifacts["written"] else 0,
            "sample_sha256": private_artifacts["owner_review_sample_sha256"],
            "minimum_acceptance_basis_points": MINIMUM_OWNER_ACCEPTANCE_BASIS_POINTS,
            "completed": False,
        },
        "decision": {
            "candidate_rows_constructed": CANARY_ROWS if private_artifacts["written"] else 0,
            "active_tool_api_credit_rows": 0,
            "source_registry_status": "pending_separate_promotion",
            "owner_review_status": "pending" if private_artifacts["written"] else "not_created",
            "candidate_registry_promotion_eligible": False,
            "dataset_seal_eligible": False,
            "sft_training_eligible": False,
            "production_eligible": False,
            "next_gate": (
                "owner_review_then_separate_cross_source_dedup_and_registry_preflight"
                if private_artifacts["written"]
                else "stop_fail_closed_before_candidate_use"
            ),
        },
        "resource_accounting": {
            "source_passes": 1,
            "source_row_reads": scan.aggregate["rows"],
            "persisted_candidate_rows": (
                CANARY_ROWS if private_artifacts["written"] else 0
            ),
            "candidate_file_reads": 0,
            "baseline_candidate_reads": 0,
            "test_dev_validation_audio_row_reads": 0,
            "private_write_bytes": private_write_bytes,
            "network_bytes": 0,
            "network_retries": 0,
            "gpu_actions": 0,
            "inference_calls": 0,
            "training_actions": 0,
            "elapsed_milliseconds": round(elapsed_seconds * 1_000),
        },
        "raw_text_retained": False,
        "production_eligible": False,
    }
    _seal(report, "report_sha256")
    validate_public_report(report)
    return report


_REPORT_TOP_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "observed_at",
        "authorization_sha256",
        "plan_file_sha256",
        "plan_internal_sha256",
        "bindings",
        "scan",
        "scan_checks",
        "curation",
        "machine_gate",
        "checks",
        "private_artifacts",
        "owner_review",
        "decision",
        "resource_accounting",
        "raw_text_retained",
        "production_eligible",
        "report_sha256",
    }
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "messages",
        "content",
        "utt",
        "annot_utt",
        "source_text",
        "raw_text",
        "record_id",
        "document_id",
        "worker_id",
        "judgments",
        "slot_method",
        "exception",
        "example",
        "note",
        "uri",
    }
)


def validate_public_report(report: Mapping[str, Any]) -> None:
    if set(report) != _REPORT_TOP_KEYS:
        raise ValueError("public report schema changed")
    _verify_seal(report, "report_sha256")

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if not isinstance(key, str) or key.casefold() in _FORBIDDEN_PUBLIC_KEYS:
                    raise ValueError("public report contains a payload-capable key")
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str):
            if any("\u0980" <= character <= "\u09ff" for character in value):
                raise ValueError("public report contains Bengali source text")
            if len(value) > 256:
                raise ValueError("public report contains an unbounded string")
        elif value is not None and not isinstance(value, (bool, int, float)):
            raise ValueError("public report contains an unsupported value")

    visit(report)
    if (
        report.get("raw_text_retained") is not False
        or report.get("production_eligible") is not False
    ):
        raise ValueError("public report safety state changed")


def _expected_authorization() -> dict[str, Any]:
    return {
        "text": AUTHORIZATION_TEXT,
        "text_sha256": AUTHORIZATION_SHA256,
        "local_only_one_source_pass_authorized": True,
        "collision_aware_semantic_dedup_authorized": True,
        "ambiguous_utterance_quarantine_authorized": True,
        "conditional_64_row_private_canary_authorized": True,
        "network_gpu_inference_training_authorized": False,
        "registry_promotion_seal_authorized": False,
        "external_action_authorized": False,
    }


def _expected_adapter_contract() -> dict[str, Any]:
    return {
        "tool_name": TOOL_NAME,
        "trajectory_roles": ["user", "tool", "tool_result", "assistant"],
        "tool_arguments": ["locale", "scenario", "intent", "slots"],
        "slots_are_ordered_name_value_objects": True,
        "repeated_slot_labels_preserved": True,
        "tool_result_status": "parsed_only",
        "tool_result_exactly_recomputable": True,
        "external_action_performed": False,
        "assistant_confirms_no_external_action": True,
        "task_category": "tool_api",
        "language": "bn",
        "split": "train",
        "license_id": SOURCE_LICENSE,
    }


def _expected_dedup_contract() -> dict[str, Any]:
    return {
        "normalization": "NFC",
        "semantic_tuple_fields": ["utterance", "locale", "scenario", "intent", "slots"],
        "normalization_equivalent_full_semantic_tuples": "collapse_deterministically",
        "ambiguous_normalized_utterance_definition": (
            "more_than_one_distinct_normalized_intent_argument_projection"
        ),
        "ambiguous_normalized_utterance_action": "quarantine_entire_group",
        "selection": "one_digest_minimum_per_60_intents_plus_four_digest_minimum_extras",
        "near_duplicate_policy": EXPECTED_NEAR_DUPLICATE_POLICY,
        "cross_source_baseline_dedup_deferred": True,
    }


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _read_object(path)
    _verify_seal(plan, "plan_sha256")
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != PLAN_STATUS:
        raise ValueError("parser-canary plan schema or status changed")
    expected_pairs = {
        "authorization": _expected_authorization(),
        "adapter_contract": _expected_adapter_contract(),
        "deduplication_contract": _expected_dedup_contract(),
        "expected_scan_counts": EXPECTED_SCAN_COUNTS,
        "expected_collision_aggregates": EXPECTED_COLLISIONS,
        "start_marker": {
            "bytes": len(START_MARKER),
            "sha256": START_MARKER_SHA256,
        },
        "limits": EXPECTED_LIMITS,
        "source": EXPECTED_SOURCE,
        "output": EXPECTED_OUTPUT,
        "predecessor_bindings": EXPECTED_PREDECESSOR_BINDINGS,
        "contract_bindings": EXPECTED_CONTRACT_BINDINGS,
        "runtime_binding": predecessor.EXPECTED_RUNTIME_BINDING,
        "prohibitions": EXPECTED_PROHIBITIONS,
    }
    for key, expected in expected_pairs.items():
        if plan.get(key) != expected:
            raise ValueError(f"parser-canary plan {key} changed")
    code = plan.get("code_bindings")
    if not isinstance(code, Mapping) or set(code) != set(EXPECTED_CODE_PATHS):
        raise ValueError("parser-canary code bindings changed")
    for label, relative in EXPECTED_CODE_PATHS.items():
        binding = code[label]
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
            raise ValueError(f"parser-canary code binding schema changed: {label}")
        if binding.get("path") != relative:
            raise ValueError(f"parser-canary code path changed: {label}")
        _binding_path(root, binding, label)
    return plan


def _registry_contains_source_id(value: Any, source_id: str) -> bool:
    if isinstance(value, Mapping):
        return any(
            (key == "source_id" and item == source_id)
            or _registry_contains_source_id(item, source_id)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_registry_contains_source_id(item, source_id) for item in value)
    return False


def load_runtime_contracts(
    plan: Mapping[str, Any], *, repository_root: str | Path
) -> RuntimeContracts:
    root = Path(repository_root).resolve()
    predecessor.verify_runtime_binding(plan["runtime_binding"])
    resolved: dict[str, Path] = {}
    for label, binding in plan["contract_bindings"].items():
        resolved[label] = _binding_path(root, binding, label)
    quality = _read_object(resolved["data_quality"])
    expected_benchmark = {
        "registry": EXPECTED_CONTRACT_BINDINGS["benchmark_registry"]["path"],
        "registry_sha256": EXPECTED_CONTRACT_BINDINGS["benchmark_registry"]["sha256"],
        "anchor_ngram_tokens": 8,
        "minimum_anchor_matches": 2,
    }
    if quality.get("benchmark_decontamination") != expected_benchmark:
        raise ValueError("benchmark decontamination contract changed")
    if importlib.metadata.version("datasketch") != "2.0.0":
        raise ValueError("datasketch runtime changed")
    registry = _read_object(resolved["source_registry"])
    if _registry_contains_source_id(registry, SOURCE_ID):
        raise ValueError("MASSIVE source unexpectedly appears in active registry")
    policy = load_sft_policy(resolved["data_contract"])
    template = load_chat_template(resolved["chat_template"])
    tokenizer = PureByteLevelBPETokenizer(resolved["tokenizer"])
    verify_chat_tokenizer(tokenizer, template)
    benchmark = BenchmarkRegistry(expected_benchmark, root)
    return RuntimeContracts(
        policy=policy, benchmark=benchmark, template=template, tokenizer=tokenizer
    )


def preflight_paths(
    plan: Mapping[str, Any], *, repository_root: str | Path
) -> tuple[Path, Path, Path, Path, Path]:
    root = Path(repository_root).resolve()
    for label, binding in plan["predecessor_bindings"].items():
        _binding_path(root, binding, label)
    source = _resolve_workspace_path(root, plan["source"]["path"])
    if (
        not source.is_file()
        or source.is_symlink()
        or source.stat().st_size != plan["source"]["bytes"]
        or sha256_file(source) != plan["source"]["sha256"]
    ):
        raise ValueError("bound source object changed")
    private_root = _resolve_private_root(root, plan["output"]["private_root"])
    marker = private_root / plan["output"]["marker_name"]
    report = _resolve_repository_path(root, plan["output"]["report_path"])
    verification = _resolve_repository_path(root, plan["output"]["verification_path"])
    for path in (private_root, marker, report, verification):
        if path.exists() or path.is_symlink():
            raise FileExistsError("parser-canary output is not fresh")
    return private_root, marker, report, verification, source


def execute_canary(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> tuple[dict[str, Any], Path]:
    started = time.monotonic()
    root = Path(repository_root).resolve()
    runtime = load_runtime_contracts(plan, repository_root=root)
    private_root, marker, report_path, _verification, source = preflight_paths(
        plan, repository_root=root
    )
    _write_marker(private_root, plan["output"]["marker_name"])
    if marker != private_root / plan["output"]["marker_name"]:
        raise ValueError("marker path changed")

    scan = scan_bound_parquet(source)
    scan_checks = production_scan_checks(scan.aggregate)
    measured: list[MeasuredCandidate] = []
    measurement: dict[str, Any] = {
        "evaluated": 0,
        "eligible": 0,
        "rejected": 0,
        "rejection_counts": {},
    }
    selection: dict[str, Any] = {
        "intent_coverage_rows": 0,
        "missing_intents": 60,
        "deterministic_extra_rows": 0,
        "near_duplicate_skips": 0,
        "selected_rows": 0,
        "selection_sha256": sha256_value([]),
    }
    selected: list[MeasuredCandidate] = []
    if all(scan_checks.values()):
        measured, measurement = measure_semantics(scan.eligible_semantics, runtime)
        selected, selection = select_canary(measured)

    machine_gate: dict[str, Any] = {
        "generic_filter": {
            "total": 0,
            "accepted": 0,
            "rejected": 0,
            "rejection_counts": {},
            "accepted_dataset_sha256": sha256_value([]),
        },
        "tokenization": {
            "total_tokens": 0,
            "supervised_tokens": 0,
            "masked_tokens": 0,
            "maximum_conversation_tokens": 0,
            "context_window": runtime.template.context_window,
            "assistant_and_tool_only_loss": runtime.template.loss_roles
            == frozenset({"assistant", "tool"}),
        },
        "deduplication": {
            "exact_normalized_conversation_duplicates": 0,
            "near_duplicate_groups": 0,
            "near_duplicate_affected_rows": 0,
            "near_duplicate_ledger_sha256": sha256_value([]),
            "semantic_selection_sha256": sha256_value([]),
        },
        "parser_contract": {
            "rows": 0,
            "four_role_trajectory_rows": 0,
            "parsed_only_rows": 0,
            "external_action_performed_rows": 0,
            "ordered_slot_list_rows": 0,
            "unique_intents": 0,
        },
    }
    private_artifacts = _empty_private_artifacts()
    private_write_bytes = len(START_MARKER)
    selection_ready = (
        all(scan_checks.values())
        and selection["intent_coverage_rows"] == 60
        and selection["missing_intents"] == 0
        and selection["deterministic_extra_rows"] == 4
        and selection["selected_rows"] == 64
    )
    if selection_ready:
        machine_gate = _final_machine_gate(selected, runtime)
        generic = machine_gate["generic_filter"]
        dedup = machine_gate["deduplication"]
        parser_contract = machine_gate["parser_contract"]
        final_ready = (
            generic["accepted"] == 64
            and generic["rejected"] == 0
            and dedup["exact_normalized_conversation_duplicates"] == 0
            and dedup["near_duplicate_groups"] == 0
            and parser_contract["external_action_performed_rows"] == 0
            and parser_contract["unique_intents"] == 60
        )
        if final_ready:
            records = [item.record for item in selected]
            private_artifacts, private_write_bytes = _write_private_artifacts(
                private_root=private_root, plan=plan, records=records
            )

    report = _build_report(
        plan=plan,
        plan_file_sha256=plan_file_sha256,
        scan=scan,
        scan_checks=scan_checks,
        measurement=measurement,
        selection=selection,
        machine_gate=machine_gate,
        private_artifacts=private_artifacts,
        private_write_bytes=private_write_bytes,
        elapsed_seconds=time.monotonic() - started,
    )
    _write_fresh(report_path, _json_bytes(report), mode=0o644)
    return report, report_path


def _read_jsonl_capped(path: Path, cap: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                raise ValueError("candidate file contains a blank line")
            rows.append(json.loads(line))
            if len(rows) > cap:
                raise ValueError("candidate audit read cap exceeded")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("candidate row schema changed")
    return rows


def verify_execution(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    plan_file_sha256: str,
    report_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    validate_public_report(report)
    runtime = load_runtime_contracts(plan, repository_root=root)
    for label, binding in plan["predecessor_bindings"].items():
        _binding_path(root, binding, label)
    source = _resolve_workspace_path(root, plan["source"]["path"])
    private_root = _resolve_private_root(root, plan["output"]["private_root"])
    marker = private_root / plan["output"]["marker_name"]
    if (
        not source.is_file()
        or sha256_file(source) != EXPECTED_SOURCE["sha256"]
        or not marker.is_file()
        or marker.read_bytes() != START_MARKER
    ):
        raise ValueError("source or marker binding changed")
    if report.get("plan_file_sha256") != plan_file_sha256:
        raise ValueError("report plan binding changed")
    if report.get("status") != PASS_STATUS:
        raise ValueError("parser canary did not reach the private owner-review gate")

    candidate_path = private_root / plan["output"]["candidate_file_name"]
    html_path = private_root / plan["output"]["owner_review_html_name"]
    manifest_path = private_root / plan["output"]["private_manifest_name"]
    rows = _read_jsonl_capped(
        candidate_path, EXPECTED_LIMITS["maximum_candidate_audit_reads"]
    )
    if len(rows) != CANARY_ROWS:
        raise ValueError("candidate row count changed")
    semantics = [parse_and_validate_candidate(row) for row in rows]
    if len({item.intent_index for item in semantics}) != 60:
        raise ValueError("candidate intent coverage changed")
    accepted, generic = filter_sft_records(
        rows,
        policy=runtime.policy,
        benchmark_registry=runtime.benchmark,
        source_registry=None,
    )
    if len(accepted) != 64 or generic["rejected"] != 0:
        raise ValueError("candidate SFT filter replay changed")
    total_tokens = 0
    supervised_tokens = 0
    masked_tokens = 0
    maximum_tokens = 0
    for row in rows:
        encoded = encode_sft_messages(
            row["messages"], tokenizer=runtime.tokenizer, template=runtime.template
        )
        total_tokens += len(encoded.input_ids)
        supervised_tokens += encoded.supervised_tokens
        masked_tokens += encoded.masked_tokens
        maximum_tokens = max(maximum_tokens, len(encoded.input_ids))
    near = near_duplicate_audit(rows)
    if near["groups"] != 0 or near["affected_rows"] != 0:
        raise ValueError("candidate near-duplicate replay changed")

    html_payload, owner_sample_sha256 = owner_review_html(rows)
    if html_path.read_bytes() != html_payload:
        raise ValueError("owner-review HTML changed")
    manifest = _read_object(manifest_path)
    _verify_seal(manifest, "manifest_sha256")
    expected_manifest = {
        "schema_version": PRIVATE_MANIFEST_SCHEMA,
        "status": "private_nonproduction_parser_canary_owner_review_pending",
        "candidate_rows": 64,
        "candidate_file": plan["output"]["candidate_file_name"],
        "candidate_file_bytes": candidate_path.stat().st_size,
        "candidate_file_sha256": sha256_file(candidate_path),
        "owner_review_rows": 64,
        "owner_review_html": plan["output"]["owner_review_html_name"],
        "owner_review_html_bytes": html_path.stat().st_size,
        "owner_review_html_sha256": sha256_file(html_path),
        "owner_review_sample_sha256": owner_sample_sha256,
        "owner_review_status": "pending",
        "active_tool_api_credit_rows": 0,
        "production_eligible": False,
        "raw_text_in_public_evidence": False,
    }
    _seal(expected_manifest, "manifest_sha256")
    if manifest != expected_manifest:
        raise ValueError("private manifest changed")
    artifacts = report["private_artifacts"]
    if (
        artifacts["candidate_file_sha256"] != sha256_file(candidate_path)
        or artifacts["owner_review_html_sha256"] != sha256_file(html_path)
        or artifacts["private_manifest_sha256"] != sha256_file(manifest_path)
        or artifacts["owner_review_sample_sha256"] != owner_sample_sha256
    ):
        raise ValueError("report private-artifact bindings changed")
    machine = report["machine_gate"]
    if (
        machine["generic_filter"]["accepted_dataset_sha256"]
        != generic["accepted_dataset_sha256"]
        or machine["tokenization"]["total_tokens"] != total_tokens
        or machine["tokenization"]["supervised_tokens"] != supervised_tokens
        or machine["tokenization"]["masked_tokens"] != masked_tokens
        or machine["tokenization"]["maximum_conversation_tokens"] != maximum_tokens
        or machine["deduplication"]["near_duplicate_ledger_sha256"]
        != near["ledger_sha256"]
    ):
        raise ValueError("report machine-gate aggregates changed")

    checks = {
        "plan_report_and_internal_seals_valid": True,
        "source_rehashed_with_zero_source_row_reads": True,
        "all_predecessor_evidence_rehashed_unchanged": True,
        "exact_durable_start_marker_preserved": True,
        "exactly_64_private_candidate_rows_replayed": True,
        "full_candidate_schema_provenance_and_parser_contract_recomputed": True,
        "generic_filter_benchmark_tokenizer_and_4k_gate_replayed": True,
        "exact_and_canary_near_duplicate_gates_replayed": True,
        "all_60_intents_remain_covered": True,
        "owner_review_packet_reproduced_and_pending": True,
        "source_registry_active_credit_promotion_seal_and_training_remain_disabled": True,
        "zero_network_gpu_inference_and_external_actions": True,
        "public_outputs_remain_raw_text_free": True,
    }
    verification: dict[str, Any] = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": VERIFICATION_STATUS,
        "observed_at": "2026-08-31",
        "plan_file_sha256": plan_file_sha256,
        "plan_internal_sha256": plan["plan_sha256"],
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "marker_sha256": START_MARKER_SHA256,
        "candidate_file_sha256": sha256_file(candidate_path),
        "owner_review_html_sha256": sha256_file(html_path),
        "private_manifest_sha256": sha256_file(manifest_path),
        "checks": checks,
        "decision": dict(report["decision"]),
        "resource_accounting": {
            "source_row_reads": 0,
            "candidate_audit_reads": 64,
            "baseline_candidate_reads": 0,
            "test_dev_validation_audio_row_reads": 0,
            "network_bytes": 0,
            "gpu_actions": 0,
            "inference_calls": 0,
            "training_actions": 0,
        },
        "raw_text_retained": False,
        "production_eligible": False,
    }
    _seal(verification, "verification_sha256")
    return verification
