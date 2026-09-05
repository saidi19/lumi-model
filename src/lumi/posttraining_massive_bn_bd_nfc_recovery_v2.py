"""Aggregate-only NFC recovery contract for the local MASSIVE ``bn-BD`` train object.

This module is deliberately independent of Parquet and candidate-writing code.  A
runner may stream the six allowlisted semantic fields into :func:`audit_recovery_pass`,
but the returned value contains only counts, booleans, and fixed metadata bindings.
Source strings are used transiently for comparison and are never returned.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RECOVERY_SCHEMA = "lumi.sft-massive-bn-bd-nfc-recovery-pass.v0.2"
RECONCILIATION_SCHEMA = "lumi.sft-massive-bn-bd-nfc-recovery-reconciliation.v0.2"
PASS_STATUS = "PASS_nfc_derived_projection_ready_for_fresh_adapter_successor"
STOP_STATUS = "STOP_nfc_recovery_gate_failed_nonproduction"
RECONCILED_STATUS = "PASS_two_pass_nfc_recovery_reconciled_nonproduction"
RECONCILIATION_STOP_STATUS = "STOP_two_pass_nfc_recovery_mismatch_nonproduction"

PLAN_SCHEMA = "lumi.sft-massive-bn-bd-nfc-recovery-plan.v0.2"
PLAN_STATUS = "authorized_local_only_two_pass_nfc_recovery_nonproduction"
REPORT_SCHEMA = "lumi.sft-massive-bn-bd-nfc-recovery-report.v0.2"
REPORT_PASS_STATUS = "PASS_two_pass_nfc_derived_recovery_ready_for_adapter_contract_nonproduction"
REPORT_STOP_STATUS = "STOP_two_pass_nfc_derived_recovery_failed_nonproduction"
VERIFICATION_SCHEMA = "lumi.sft-massive-bn-bd-nfc-recovery-verification.v0.2"
VERIFICATION_STATUS = "PASS_zero_row_structural_nfc_recovery_verification_nonproduction"

AUTHORIZATION_TEXT = "ok next step shuru de ar fix kor jekhane ja lage and proper fix korbi"
AUTHORIZATION_SHA256 = "4a1857f2886377a72a7a42918c53886f33716213976f30df83aa8284024728f6"
RECOVERY_MARKER = b'{"started":"massive_bn_bd_nfc_recovery_v0.2"}\n'
RECOVERY_MARKER_SHA256 = "17e96fb81eb2a130041dc8543e6bc60a00167224b9a1ec5bf921f56d8c6c83c6"

EXPECTED_LIMITS = {
    "maximum_source_row_reads": 23_028,
    "exact_sequential_source_passes": 2,
    "maximum_candidate_or_baseline_reads": 0,
    "maximum_test_dev_validation_row_reads": 0,
    "maximum_network_bytes": 0,
    "maximum_gpu_actions": 0,
    "maximum_inference_calls": 0,
    "maximum_training_actions": 0,
    "maximum_runtime_seconds": 1_200,
    "maximum_private_write_bytes": 2_097_152,
    "network_retries": 0,
}

EXPECTED_RUNTIME_BINDING = {
    "pyarrow_init_sha256": ("04ef85e330eb3cda057b5f4f8e9c06891536bf06d7b26341d392153ecaa861e2"),
    "pyarrow_parquet_extension_sha256": (
        "11c7c5896c3a6093049a2c3a9bb58ba31da09d2d6422528736294f5907ff9055"
    ),
    "pyarrow_version": "25.0.0",
    "python_executable": "/Users/afnan/Documents/lumi-model/.venv/bin/python",
    "python_executable_sha256": (
        "71720f1fc66989ebd691e81c96111b47ae6ff3f1a478666084d1cacbf0fccbf2"
    ),
}

EXPECTED_SOURCE_BINDING = {
    "path": (
        "../lumi-artifacts/4b-sft-massive-bn-bd-acquisition-v0.1/"
        "6f34851d7438/attempt-001/private/raw/AmazonScience_massive/"
        "bn-BD/train/0000.parquet"
    ),
    "bytes": 1_071_668,
    "rows": 11_514,
    "sha256": "947f65c28cce0d9a57c16bc1a99df08f84e9b7599dff25f47fd5b9760256c17d",
    "dataset_id": "AmazonScience/massive",
    "revision": "d1792d0e4f32a74d43725cc1b901f600dad990dd",
    "partition": "train",
    "locale": "bn-BD",
}

EXPECTED_OUTPUT = {
    "private_root": (
        "../lumi-artifacts/4b-sft-massive-bn-bd-nfc-recovery-v0.2/68d3507a4b7a/attempt-001/private"
    ),
    "marker_name": ".nfc-recovery-started-v0.2",
    "report_path": "evidence/lumi-4b-sft-massive-bn-bd-nfc-recovery-v0.2.json",
    "verification_path": ("evidence/lumi-4b-sft-massive-bn-bd-nfc-recovery-verification-v0.2.json"),
}

EXPECTED_SOURCE_ROWS = 11_514
EXPECTED_RAW_NON_NFC_UTTERANCES = 1_495
EXPECTED_RAW_NON_NFC_ANNOTATED_UTTERANCES = 1_495
EXPECTED_TOTAL_ANNOTATIONS = 11_287
EXPECTED_ROWS_WITH_ANNOTATIONS = 7_754
EXPECTED_MAXIMUM_ANNOTATIONS_PER_ROW = 7
EXPECTED_SLOT_LABEL_CARDINALITY = 55

HUGGINGFACE_METADATA_KEY_COUNT = 1
HUGGINGFACE_METADATA_KEY = "huggingface"
HUGGINGFACE_METADATA_BYTES = 2_192
HUGGINGFACE_METADATA_SHA256 = "f2e7b189ab9dc9c95ce9fdab5ef7d36522771451fa663fee11af08c8612c245e"
HUGGINGFACE_CANONICAL_FEATURES_SHA256 = (
    "c68f119d6e7737f35447f34b703e617cda3f69c85cce718f0022381f5d5c2164"
)
SCENARIO_MAP_SHA256 = "75badc0873f3cee75962a95d553bda51ce11e8f7089bce25f91a9b3f9419411f"
INTENT_MAP_SHA256 = "c1d704291fc472572de507c7ff40487d73cc70a97bd29a9954b14d609350a012"

SCENARIO_LABELS = (
    "social",
    "transport",
    "calendar",
    "play",
    "news",
    "datetime",
    "recommendation",
    "email",
    "iot",
    "general",
    "audio",
    "lists",
    "qa",
    "cooking",
    "takeaway",
    "music",
    "alarm",
    "weather",
)

INTENT_LABELS = (
    "datetime_query",
    "iot_hue_lightchange",
    "transport_ticket",
    "takeaway_query",
    "qa_stock",
    "general_greet",
    "recommendation_events",
    "music_dislikeness",
    "iot_wemo_off",
    "cooking_recipe",
    "qa_currency",
    "transport_traffic",
    "general_quirky",
    "weather_query",
    "audio_volume_up",
    "email_addcontact",
    "takeaway_order",
    "email_querycontact",
    "iot_hue_lightup",
    "recommendation_locations",
    "play_audiobook",
    "lists_createoradd",
    "news_query",
    "alarm_query",
    "iot_wemo_on",
    "general_joke",
    "qa_definition",
    "social_query",
    "music_settings",
    "audio_volume_other",
    "calendar_remove",
    "iot_hue_lightdim",
    "calendar_query",
    "email_sendemail",
    "iot_cleaning",
    "audio_volume_down",
    "play_radio",
    "cooking_query",
    "datetime_convert",
    "qa_maths",
    "iot_hue_lightoff",
    "iot_hue_lighton",
    "transport_query",
    "music_likeness",
    "email_query",
    "play_music",
    "audio_volume_mute",
    "social_post",
    "alarm_set",
    "qa_factoid",
    "calendar_set",
    "play_game",
    "alarm_remove",
    "lists_remove",
    "transport_taxi",
    "recommendation_movies",
    "iot_coffee",
    "music_query",
    "play_podcasts",
    "lists_query",
)

EXPECTED_TOP_LEVEL_FIELDS = (
    "id",
    "locale",
    "partition",
    "scenario",
    "intent",
    "utt",
    "annot_utt",
    "worker_id",
    "slot_method",
    "judgments",
)
EXPECTED_TOP_LEVEL_SCHEMA = (
    ("id", "string", True),
    ("locale", "string", True),
    ("partition", "string", True),
    ("scenario", "int64", True),
    ("intent", "int64", True),
    ("utt", "string", True),
    ("annot_utt", "string", True),
    ("worker_id", "string", True),
    ("slot_method", "struct<slot: list<item: string>, method: list<item: string>>", True),
    (
        "judgments",
        "struct<worker_id: list<item: string>, intent_score: list<item: int8>, "
        "slots_score: list<item: int8>, grammar_score: list<item: int8>, "
        "spelling_score: list<item: int8>, language_identification: list<item: string>>",
        True,
    ),
)
AUDIT_COLUMNS = ("locale", "partition", "scenario", "intent", "utt", "annot_utt")

EXPECTED_PREDECESSOR_BINDINGS = {
    "acquisition_plan": {
        "path": "configs/lumi-4b-sft-massive-bn-bd-acquisition-v0.1.json",
        "sha256": "f5826974727257403a64c7e2e64be76016d296148e9c45338b3a35d286e0e853",
        "internal_sha256": ("eafceb4a630f863203757fb49abfd43ef403cde1caeb30841a0bfd46beccf7de"),
    },
    "acquisition_report": {
        "path": "evidence/lumi-4b-sft-massive-bn-bd-acquisition-v0.1.json",
        "sha256": "68d3507a4b7a35674cf1068feefea7466978fbdf37490797d12fa8e4d0677817",
        "internal_sha256": ("a5b4bffbf4a1fd2255d739084aad41ce63c36aafa10c8f0e0a0070ca776659e0"),
    },
    "acquisition_verification": {
        "path": ("evidence/lumi-4b-sft-massive-bn-bd-acquisition-verification-v0.1.json"),
        "sha256": "76ffddea53489db27f55ba6417957f6bd4ff51eba03a13970fd8b8244f79ef7f",
        "internal_sha256": ("7476f2311996eefc018bd7d06124ae34783a268c769d7d6036f64b60bf5ba1f1"),
    },
    "artifact_manifest": {
        "path": (
            "../lumi-artifacts/4b-sft-massive-bn-bd-acquisition-v0.1/"
            "6f34851d7438/attempt-001/private/artifact-manifest.json"
        ),
        "sha256": "6de1a16a5f7ad58b11e459e131d79b6ba9d400d932578c722fbbc2ab2e938eb2",
        "internal_sha256": ("1058b33fdcf086d3edbc732270e9bbdd483fbcb034589768bb48883fcfb0ea62"),
    },
    "acquisition_marker": {
        "path": (
            "../lumi-artifacts/4b-sft-massive-bn-bd-acquisition-v0.1/"
            "6f34851d7438/attempt-001/private/.acquisition-started-v0.1"
        ),
        "sha256": "cb28ffbc16e91c28c6483ed64525b9db3fd78bf5fb1cef331c562276b664a12b",
        "bytes": 51,
    },
}

EXPECTED_PROHIBITIONS = {
    "no_retry_after_durable_start_marker": True,
    "no_source_or_predecessor_mutation": True,
    "no_candidate_owner_or_registry_artifacts": True,
    "no_candidate_registry_promotion_seal_or_training": True,
    "no_network_gpu_inference_remote_or_instance_action": True,
    "no_test_dev_validation_audio_or_non_bn_bd_reads": True,
    "no_unrelated_action": True,
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _class_label_sha256(names: Sequence[str]) -> str:
    descriptor = {"_type": "ClassLabel", "names": list(names)}
    return hashlib.sha256(_canonical_json_bytes(descriptor)).hexdigest()


def _validate_observed_class_label_maps(features: Mapping[str, Any]) -> None:
    if (
        hashlib.sha256(_canonical_json_bytes(features.get("scenario"))).hexdigest()
        != SCENARIO_MAP_SHA256
        or hashlib.sha256(_canonical_json_bytes(features.get("intent"))).hexdigest()
        != INTENT_MAP_SHA256
    ):
        raise ValueError("bound Hugging Face ClassLabel maps changed")


if len(SCENARIO_LABELS) != 18 or _class_label_sha256(SCENARIO_LABELS) != SCENARIO_MAP_SHA256:
    raise RuntimeError("bound MASSIVE scenario map changed")
if len(INTENT_LABELS) != 60 or _class_label_sha256(INTENT_LABELS) != INTENT_MAP_SHA256:
    raise RuntimeError("bound MASSIVE intent map changed")


@dataclass(frozen=True)
class RecoveryExpectations:
    """Exact aggregate expectations for one recovery pass.

    Production callers use the defaults.  Synthetic tests may construct a smaller
    expectation object without weakening the module-level production constants.
    """

    rows: int = EXPECTED_SOURCE_ROWS
    raw_non_nfc_utterances: int = EXPECTED_RAW_NON_NFC_UTTERANCES
    raw_non_nfc_annotated_utterances: int = EXPECTED_RAW_NON_NFC_ANNOTATED_UTTERANCES
    total_annotations: int = EXPECTED_TOTAL_ANNOTATIONS
    rows_with_annotations: int = EXPECTED_ROWS_WITH_ANNOTATIONS
    maximum_annotations_per_row: int = EXPECTED_MAXIMUM_ANNOTATIONS_PER_ROW
    slot_label_cardinality: int = EXPECTED_SLOT_LABEL_CARDINALITY
    scenario_cardinality: int = 18
    intent_cardinality: int = 60

    def __post_init__(self) -> None:
        values = (
            self.rows,
            self.raw_non_nfc_utterances,
            self.raw_non_nfc_annotated_utterances,
            self.total_annotations,
            self.rows_with_annotations,
            self.maximum_annotations_per_row,
            self.slot_label_cardinality,
            self.scenario_cardinality,
            self.intent_cardinality,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values
        ):
            raise ValueError("recovery expectations must be non-negative integers")
        if self.raw_non_nfc_utterances > self.rows:
            raise ValueError("utterance NFC expectation exceeds row count")
        if self.raw_non_nfc_annotated_utterances > self.rows:
            raise ValueError("annotated-utterance NFC expectation exceeds row count")


PRODUCTION_EXPECTATIONS = RecoveryExpectations()


class _AnnotationParseError(ValueError):
    """Non-payload error used only for aggregate rejection accounting."""


@dataclass(frozen=True)
class _ParsedAnnotations:
    literals: tuple[str, ...]
    labels: tuple[str, ...]
    values: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.labels)

    def deannotated(self) -> str:
        pieces: list[str] = []
        for index, literal in enumerate(self.literals[:-1]):
            pieces.append(literal)
            pieces.append(self.values[index])
        pieces.append(self.literals[-1])
        return "".join(pieces)

    def ordered_nfc_signature(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (unicodedata.normalize("NFC", label), unicodedata.normalize("NFC", value))
            for label, value in zip(self.labels, self.values, strict=True)
        )

    def literal_nfc_signature(self) -> tuple[str, ...]:
        return tuple(unicodedata.normalize("NFC", literal) for literal in self.literals)


def _strict_parse_annotations(value: str) -> _ParsedAnnotations:
    """Parse MASSIVE bracket annotations without repairing malformed input."""

    literals: list[str] = []
    labels: list[str] = []
    values: list[str] = []
    cursor = 0
    literal_start = 0
    length = len(value)
    while cursor < length:
        character = value[cursor]
        if character == "]":
            raise _AnnotationParseError("unmatched closing bracket")
        if character != "[":
            cursor += 1
            continue

        literals.append(value[literal_start:cursor])
        closing = value.find("]", cursor + 1)
        if closing < 0:
            raise _AnnotationParseError("unmatched opening bracket")
        body = value[cursor + 1 : closing]
        if "[" in body or "]" in body:
            raise _AnnotationParseError("nested bracket")
        label, separator, slot_value = body.partition(":")
        label = label.strip()
        slot_value = slot_value.strip()
        if not separator:
            raise _AnnotationParseError("annotation separator missing")
        if not label:
            raise _AnnotationParseError("empty annotation label")
        if not slot_value:
            raise _AnnotationParseError("empty annotation value")
        labels.append(label)
        values.append(slot_value)
        cursor = closing + 1
        literal_start = cursor

    literals.append(value[literal_start:])
    return _ParsedAnnotations(tuple(literals), tuple(labels), tuple(values))


def _empty_string_counts() -> dict[str, int]:
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


def _domain_digest(domain: bytes, value: Any) -> str:
    """Digest one transient value without retaining its lexical representation."""

    return hashlib.sha256(domain + b"\x00" + _canonical_json_bytes(value)).hexdigest()


@dataclass
class _DigestCollisionAccumulator:
    """Online collision accounting that retains only domain-separated digests."""

    domain: bytes

    def __post_init__(self) -> None:
        self.population = 0
        self.raw_counts: Counter[str] = Counter()
        self.normalized_counts: Counter[str] = Counter()
        self.raw_by_normalized: dict[str, set[str]] = defaultdict(set)

    def observe(self, raw: Any, normalized: Any) -> None:
        raw_digest = _domain_digest(self.domain, raw)
        normalized_digest = _domain_digest(self.domain, normalized)
        self.population += 1
        self.raw_counts[raw_digest] += 1
        self.normalized_counts[normalized_digest] += 1
        self.raw_by_normalized[normalized_digest].add(raw_digest)

    def summary(self) -> dict[str, int]:
        preexisting = {digest for digest, count in self.raw_counts.items() if count > 1}
        introduced = {
            digest for digest, raw_digests in self.raw_by_normalized.items() if len(raw_digests) > 1
        }
        return {
            "population": self.population,
            "preexisting_duplicate_groups": len(preexisting),
            "preexisting_duplicate_affected_rows": sum(
                self.raw_counts[digest] for digest in preexisting
            ),
            "introduced_normalization_collision_groups": len(introduced),
            "introduced_normalization_collision_affected_rows": sum(
                self.normalized_counts[digest] for digest in introduced
            ),
        }


def _valid_label_index(value: Any, labels: Sequence[str]) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and 0 <= value < len(labels)


def _string_checks_clear(counts: Mapping[str, int], expected_rows: int) -> bool:
    return (
        counts["strings"] == expected_rows
        and counts["non_strings"] == 0
        and counts["empty"] == 0
        and counts["outer_whitespace"] == 0
        and counts["control_character_rows"] == 0
    )


def audit_recovery_pass(
    rows: Iterable[Mapping[str, Any]],
    *,
    expectations: RecoveryExpectations = PRODUCTION_EXPECTATIONS,
) -> dict[str, Any]:
    """Run one deterministic aggregate-only NFC recovery pass.

    The only derived transformation is Unicode NFC on ``utt`` and ``annot_utt``.
    No source value, row identifier, digest, annotation label, or annotation value is
    included in the returned object.
    """

    raw_utterance = _empty_string_counts()
    raw_annotated = _empty_string_counts()
    post_utterance = _empty_string_counts()
    post_annotated = _empty_string_counts()

    row_count = 0
    invalid_row_shapes = 0
    expected_locale_rows = 0
    unexpected_locale_rows = 0
    expected_partition_rows = 0
    unexpected_partition_rows = 0
    invalid_scenario_rows = 0
    invalid_intent_rows = 0
    intent_prefix_mismatch_rows = 0
    scenario_values: set[int] = set()
    intent_values: set[int] = set()

    utterance_only_non_nfc = 0
    annotated_only_non_nfc = 0
    both_non_nfc = 0
    neither_non_nfc = 0
    utterance_nfc_idempotence_failures = 0
    annotated_nfc_idempotence_failures = 0

    malformed_annotation_rows = 0
    raw_total_annotations = 0
    post_total_annotations = 0
    rows_with_annotations = 0
    maximum_annotations = 0
    ordered_annotation_mismatch_rows = 0
    literal_segment_mismatch_rows = 0
    deannotation_mismatch_rows = 0
    preexisting_duplicate_slot_label_rows = 0
    introduced_nfc_duplicate_slot_label_rows = 0
    post_nfc_duplicate_slot_label_rows = 0
    raw_casefold_slot_label_digests: set[str] = set()
    post_nfc_casefold_slot_label_digests: set[str] = set()
    post_nfc_exact_slot_label_digests: set[str] = set()

    utterance_collisions = _DigestCollisionAccumulator(b"massive.utt")
    annotated_collisions = _DigestCollisionAccumulator(b"massive.annot_utt")
    tuple_collisions = _DigestCollisionAccumulator(b"massive.semantic_tuple")

    for row in rows:
        row_count += 1
        if not isinstance(row, Mapping):
            invalid_row_shapes += 1
            continue

        locale = row.get("locale")
        partition = row.get("partition")
        expected_locale_rows += int(locale == "bn-BD")
        unexpected_locale_rows += int(locale != "bn-BD")
        expected_partition_rows += int(partition == "train")
        unexpected_partition_rows += int(partition != "train")
        scenario = row.get("scenario")
        intent = row.get("intent")
        scenario_ok = _valid_label_index(scenario, SCENARIO_LABELS)
        intent_ok = _valid_label_index(intent, INTENT_LABELS)
        if not scenario_ok:
            invalid_scenario_rows += 1
        else:
            scenario_values.add(scenario)
        if not intent_ok:
            invalid_intent_rows += 1
        else:
            intent_values.add(intent)
        if scenario_ok and intent_ok:
            intent_prefix_mismatch_rows += int(
                INTENT_LABELS[intent].partition("_")[0] != SCENARIO_LABELS[scenario]
            )

        observed_utterance = _observe_string(row.get("utt"), raw_utterance)
        observed_annotated = _observe_string(row.get("annot_utt"), raw_annotated)
        if observed_utterance is None or observed_annotated is None:
            continue

        normalized_utterance = unicodedata.normalize("NFC", observed_utterance)
        normalized_annotated = unicodedata.normalize("NFC", observed_annotated)
        _observe_string(normalized_utterance, post_utterance)
        _observe_string(normalized_annotated, post_annotated)
        utterance_nfc_idempotence_failures += int(
            unicodedata.normalize("NFC", normalized_utterance) != normalized_utterance
        )
        annotated_nfc_idempotence_failures += int(
            unicodedata.normalize("NFC", normalized_annotated) != normalized_annotated
        )

        utterance_changed = observed_utterance != normalized_utterance
        annotated_changed = observed_annotated != normalized_annotated
        both_non_nfc += int(utterance_changed and annotated_changed)
        utterance_only_non_nfc += int(utterance_changed and not annotated_changed)
        annotated_only_non_nfc += int(annotated_changed and not utterance_changed)
        neither_non_nfc += int(not utterance_changed and not annotated_changed)

        utterance_collisions.observe(observed_utterance, normalized_utterance)
        annotated_collisions.observe(observed_annotated, normalized_annotated)
        if scenario_ok and intent_ok:
            tuple_collisions.observe(
                (scenario, intent, observed_utterance, observed_annotated),
                (scenario, intent, normalized_utterance, normalized_annotated),
            )

        try:
            parsed_raw = _strict_parse_annotations(observed_annotated)
            parsed_normalized = _strict_parse_annotations(normalized_annotated)
        except _AnnotationParseError:
            malformed_annotation_rows += 1
            continue

        raw_total_annotations += parsed_raw.count
        post_total_annotations += parsed_normalized.count
        rows_with_annotations += int(parsed_raw.count > 0)
        maximum_annotations = max(maximum_annotations, parsed_raw.count)

        raw_label_set = set(parsed_raw.labels)
        normalized_raw_labels = tuple(
            unicodedata.normalize("NFC", label) for label in parsed_raw.labels
        )
        normalized_label_set = set(normalized_raw_labels)
        preexisting_duplicate_slot_label_rows += int(len(raw_label_set) < len(parsed_raw.labels))
        introduced_nfc_duplicate_slot_label_rows += int(
            len(normalized_label_set) < len(raw_label_set)
        )
        post_nfc_duplicate_slot_label_rows += int(
            len(normalized_label_set) < len(normalized_raw_labels)
        )
        for raw_label, normalized_label in zip(
            parsed_raw.labels, normalized_raw_labels, strict=True
        ):
            raw_casefold_slot_label_digests.add(
                _domain_digest(b"massive.slot.raw_casefold", raw_label.casefold())
            )
            post_nfc_casefold_slot_label_digests.add(
                _domain_digest(b"massive.slot.post_nfc_casefold", normalized_label.casefold())
            )
            post_nfc_exact_slot_label_digests.add(
                _domain_digest(b"massive.slot.post_nfc_exact", normalized_label)
            )

        ordered_annotation_mismatch_rows += int(
            parsed_raw.ordered_nfc_signature() != parsed_normalized.ordered_nfc_signature()
        )
        literal_segment_mismatch_rows += int(
            parsed_raw.literal_nfc_signature() != parsed_normalized.literal_nfc_signature()
        )
        deannotation_mismatch_rows += int(
            unicodedata.normalize("NFC", parsed_raw.deannotated()) != normalized_utterance
            or parsed_normalized.deannotated() != normalized_utterance
        )

    utterance_duplicates = utterance_collisions.summary()
    annotated_duplicates = annotated_collisions.summary()
    tuple_duplicates = tuple_collisions.summary()

    intersection_total = (
        utterance_only_non_nfc + annotated_only_non_nfc + both_non_nfc + neither_non_nfc
    )
    checks = {
        "exact_row_count": row_count == expectations.rows,
        "row_shapes_are_mappings": invalid_row_shapes == 0,
        "all_rows_are_bn_bd_train": expected_locale_rows == expectations.rows
        and unexpected_locale_rows == 0
        and expected_partition_rows == expectations.rows
        and unexpected_partition_rows == 0,
        "scenario_and_intent_indices_valid": invalid_scenario_rows == 0
        and invalid_intent_rows == 0,
        "scenario_and_intent_cardinalities_exact": len(scenario_values)
        == expectations.scenario_cardinality
        and len(intent_values) == expectations.intent_cardinality,
        "intent_prefix_matches_scenario": intent_prefix_mismatch_rows == 0,
        "raw_strings_clear": _string_checks_clear(raw_utterance, expectations.rows)
        and _string_checks_clear(raw_annotated, expectations.rows),
        "post_nfc_strings_clear": _string_checks_clear(post_utterance, expectations.rows)
        and _string_checks_clear(post_annotated, expectations.rows),
        "raw_non_nfc_counts_exact": raw_utterance["non_nfc"] == expectations.raw_non_nfc_utterances
        and raw_annotated["non_nfc"] == expectations.raw_non_nfc_annotated_utterances,
        "post_nfc_counts_zero": post_utterance["non_nfc"] == 0 and post_annotated["non_nfc"] == 0,
        "nfc_is_idempotent": utterance_nfc_idempotence_failures == 0
        and annotated_nfc_idempotence_failures == 0,
        "nfc_intersection_arithmetic_reconciles": intersection_total == row_count
        and utterance_only_non_nfc + both_non_nfc == raw_utterance["non_nfc"]
        and annotated_only_non_nfc + both_non_nfc == raw_annotated["non_nfc"],
        "annotation_grammar_clear": malformed_annotation_rows == 0,
        "annotation_aggregates_exact": raw_total_annotations == expectations.total_annotations
        and post_total_annotations == expectations.total_annotations
        and rows_with_annotations == expectations.rows_with_annotations
        and maximum_annotations == expectations.maximum_annotations_per_row
        and len(raw_casefold_slot_label_digests) == expectations.slot_label_cardinality
        and len(post_nfc_casefold_slot_label_digests) == expectations.slot_label_cardinality,
        "ordered_annotations_preserved": ordered_annotation_mismatch_rows == 0,
        "literal_segments_preserved": literal_segment_mismatch_rows == 0,
        "deannotation_matches_utterance": deannotation_mismatch_rows == 0,
        "nfc_does_not_introduce_slot_label_duplicates": (
            introduced_nfc_duplicate_slot_label_rows == 0
        ),
        "nfc_does_not_introduce_content_collisions": all(
            summary["introduced_normalization_collision_groups"] == 0
            for summary in (
                utterance_duplicates,
                annotated_duplicates,
                tuple_duplicates,
            )
        ),
        "fixed_metadata_contract_bound": (
            len(SCENARIO_LABELS) == 18
            and len(INTENT_LABELS) == 60
            and _class_label_sha256(SCENARIO_LABELS) == SCENARIO_MAP_SHA256
            and _class_label_sha256(INTENT_LABELS) == INTENT_MAP_SHA256
        ),
    }
    recovery_gate_pass = all(checks.values())
    scalar_argument_adapter_eligible = (
        recovery_gate_pass and post_nfc_duplicate_slot_label_rows == 0
    )
    result = {
        "schema_version": RECOVERY_SCHEMA,
        "status": PASS_STATUS if recovery_gate_pass else STOP_STATUS,
        "metadata_contract": {
            "metadata_key_count": HUGGINGFACE_METADATA_KEY_COUNT,
            "metadata_key_sha256": hashlib.sha256(
                HUGGINGFACE_METADATA_KEY.encode("utf-8")
            ).hexdigest(),
            "metadata_bytes": HUGGINGFACE_METADATA_BYTES,
            "metadata_sha256": HUGGINGFACE_METADATA_SHA256,
            "canonical_features_sha256": HUGGINGFACE_CANONICAL_FEATURES_SHA256,
            "scenario_map_sha256": SCENARIO_MAP_SHA256,
            "scenario_cardinality": len(SCENARIO_LABELS),
            "intent_map_sha256": INTENT_MAP_SHA256,
            "intent_cardinality": len(INTENT_LABELS),
        },
        "expected_contract": {
            "rows": expectations.rows,
            "raw_non_nfc_utterances": expectations.raw_non_nfc_utterances,
            "raw_non_nfc_annotated_utterances": (expectations.raw_non_nfc_annotated_utterances),
            "total_annotations": expectations.total_annotations,
            "rows_with_annotations": expectations.rows_with_annotations,
            "maximum_annotations_per_row": expectations.maximum_annotations_per_row,
            "slot_label_cardinality": expectations.slot_label_cardinality,
            "scenario_cardinality": expectations.scenario_cardinality,
            "intent_cardinality": expectations.intent_cardinality,
        },
        "aggregate": {
            "rows": row_count,
            "invalid_row_shapes": invalid_row_shapes,
            "source_scope": {
                "expected_locale_rows": expected_locale_rows,
                "unexpected_locale_rows": unexpected_locale_rows,
                "expected_partition_rows": expected_partition_rows,
                "unexpected_partition_rows": unexpected_partition_rows,
            },
            "strings": {
                "raw_utterance": raw_utterance,
                "raw_annotated_utterance": raw_annotated,
                "post_nfc_utterance": post_utterance,
                "post_nfc_annotated_utterance": post_annotated,
            },
            "nfc_intersection": {
                "utterance_only": utterance_only_non_nfc,
                "annotated_utterance_only": annotated_only_non_nfc,
                "both": both_non_nfc,
                "neither": neither_non_nfc,
            },
            "nfc_idempotence_failures": {
                "utterance": utterance_nfc_idempotence_failures,
                "annotated_utterance": annotated_nfc_idempotence_failures,
            },
            "semantics": {
                "invalid_scenario_rows": invalid_scenario_rows,
                "invalid_intent_rows": invalid_intent_rows,
                "scenario_cardinality": len(scenario_values),
                "intent_cardinality": len(intent_values),
                "intent_prefix_mismatch_rows": intent_prefix_mismatch_rows,
            },
            "annotations": {
                "raw_total": raw_total_annotations,
                "post_nfc_total": post_total_annotations,
                "rows_with_annotations": rows_with_annotations,
                "maximum_per_row": maximum_annotations,
                "raw_casefold_slot_label_cardinality": len(raw_casefold_slot_label_digests),
                "post_nfc_casefold_slot_label_cardinality": len(
                    post_nfc_casefold_slot_label_digests
                ),
                "post_nfc_exact_slot_label_cardinality": len(post_nfc_exact_slot_label_digests),
                "malformed_rows": malformed_annotation_rows,
                "ordered_signature_mismatch_rows": ordered_annotation_mismatch_rows,
                "literal_segment_mismatch_rows": literal_segment_mismatch_rows,
                "deannotation_mismatch_rows": deannotation_mismatch_rows,
                "preexisting_duplicate_slot_label_rows": (preexisting_duplicate_slot_label_rows),
                "introduced_nfc_duplicate_slot_label_rows": (
                    introduced_nfc_duplicate_slot_label_rows
                ),
                "post_nfc_duplicate_slot_label_rows": (post_nfc_duplicate_slot_label_rows),
            },
            "duplicates": {
                "utterance": utterance_duplicates,
                "annotated_utterance": annotated_duplicates,
                "semantic_tuple": tuple_duplicates,
            },
        },
        "checks": checks,
        "decision": {
            "nfc_recovery_gate_pass": recovery_gate_pass,
            "scalar_argument_adapter_eligible": scalar_argument_adapter_eligible,
            "fresh_adapter_contract_successor_eligible": recovery_gate_pass,
            "candidate_construction_eligible": False,
            "candidate_rows_constructed": 0,
            "active_tool_api_credit_rows": 0,
        },
        "raw_text_retained": False,
        "production_eligible": False,
    }
    validate_raw_text_free_result(result)
    return result


def aggregate_parquet_pass(
    path: str,
    *,
    expectations: RecoveryExpectations = PRODUCTION_EXPECTATIONS,
) -> dict[str, Any]:
    """Stream one bounded Parquet pass through the aggregate-only recovery audit."""

    import pyarrow.parquet as pq  # type: ignore[import-not-found]

    parquet = pq.ParquetFile(path, pre_buffer=False, memory_map=False)
    if parquet.metadata.num_rows != expectations.rows:
        raise ValueError("bound Parquet row count changed")
    schema = parquet.schema_arrow
    observed_schema = tuple((field.name, str(field.type), field.nullable) for field in schema)
    if observed_schema != EXPECTED_TOP_LEVEL_SCHEMA:
        raise ValueError("bound Parquet top-level schema changed")
    metadata = schema.metadata
    if not isinstance(metadata, Mapping) or set(metadata) != {HUGGINGFACE_METADATA_KEY.encode()}:
        raise ValueError("bound Parquet metadata keys changed")
    raw_metadata = metadata[HUGGINGFACE_METADATA_KEY.encode()]
    if (
        len(raw_metadata) != HUGGINGFACE_METADATA_BYTES
        or hashlib.sha256(raw_metadata).hexdigest() != HUGGINGFACE_METADATA_SHA256
    ):
        raise ValueError("bound Hugging Face metadata changed")
    try:
        metadata_value = json.loads(raw_metadata)
        features = metadata_value["info"]["features"]
    except (KeyError, TypeError, json.JSONDecodeError):
        raise ValueError("bound Hugging Face feature metadata is malformed") from None
    if (
        hashlib.sha256(_canonical_json_bytes(features)).hexdigest()
        != HUGGINGFACE_CANONICAL_FEATURES_SHA256
    ):
        raise ValueError("bound Hugging Face features changed")
    if not isinstance(features, Mapping):
        raise ValueError("bound Hugging Face features are not a mapping")
    _validate_observed_class_label_maps(features)

    def streamed_rows() -> Iterable[Mapping[str, Any]]:
        observed = 0
        for batch in parquet.iter_batches(
            batch_size=1_024, columns=list(AUDIT_COLUMNS), use_threads=False
        ):
            values = batch.to_pydict()
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
                if observed > expectations.rows:
                    raise ValueError("source-row read cap exceeded")
                yield {
                    "locale": locale,
                    "partition": partition,
                    "scenario": scenario,
                    "intent": intent,
                    "utt": utterance,
                    "annot_utt": annotated,
                }

    return audit_recovery_pass(streamed_rows(), expectations=expectations)


_STRING_COUNT_KEYS = frozenset(
    {
        "strings",
        "non_strings",
        "empty",
        "outer_whitespace",
        "non_nfc",
        "control_character_rows",
        "utf8_bytes",
        "maximum_characters",
    }
)
_DUPLICATE_SUMMARY_KEYS = frozenset(
    {
        "population",
        "preexisting_duplicate_groups",
        "preexisting_duplicate_affected_rows",
        "introduced_normalization_collision_groups",
        "introduced_normalization_collision_affected_rows",
    }
)
_PASS_CHECK_KEYS = frozenset(
    {
        "exact_row_count",
        "row_shapes_are_mappings",
        "all_rows_are_bn_bd_train",
        "scenario_and_intent_indices_valid",
        "scenario_and_intent_cardinalities_exact",
        "intent_prefix_matches_scenario",
        "raw_strings_clear",
        "post_nfc_strings_clear",
        "raw_non_nfc_counts_exact",
        "post_nfc_counts_zero",
        "nfc_is_idempotent",
        "nfc_intersection_arithmetic_reconciles",
        "annotation_grammar_clear",
        "annotation_aggregates_exact",
        "ordered_annotations_preserved",
        "literal_segments_preserved",
        "deannotation_matches_utterance",
        "nfc_does_not_introduce_slot_label_duplicates",
        "nfc_does_not_introduce_content_collisions",
        "fixed_metadata_contract_bound",
    }
)
_DECISION_KEYS = frozenset(
    {
        "nfc_recovery_gate_pass",
        "scalar_argument_adapter_eligible",
        "fresh_adapter_contract_successor_eligible",
        "candidate_construction_eligible",
        "candidate_rows_constructed",
        "active_tool_api_credit_rows",
    }
)
_ALLOWED_RESULT_STRINGS = frozenset(
    {
        RECOVERY_SCHEMA,
        RECONCILIATION_SCHEMA,
        PASS_STATUS,
        STOP_STATUS,
        RECONCILED_STATUS,
        RECONCILIATION_STOP_STATUS,
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_keys(value: Any, expected: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"raw-text-free {label} schema changed")
    return value


def _recompute_pass_checks(result: Mapping[str, Any]) -> dict[str, bool]:
    expected = result["expected_contract"]
    aggregate = result["aggregate"]
    strings = aggregate["strings"]
    source_scope = aggregate["source_scope"]
    semantics = aggregate["semantics"]
    annotations = aggregate["annotations"]
    intersection = aggregate["nfc_intersection"]
    idempotence = aggregate["nfc_idempotence_failures"]
    duplicates = aggregate["duplicates"]
    rows = expected["rows"]
    return {
        "exact_row_count": aggregate["rows"] == rows,
        "row_shapes_are_mappings": aggregate["invalid_row_shapes"] == 0,
        "all_rows_are_bn_bd_train": source_scope["expected_locale_rows"] == rows
        and source_scope["unexpected_locale_rows"] == 0
        and source_scope["expected_partition_rows"] == rows
        and source_scope["unexpected_partition_rows"] == 0,
        "scenario_and_intent_indices_valid": semantics["invalid_scenario_rows"] == 0
        and semantics["invalid_intent_rows"] == 0,
        "scenario_and_intent_cardinalities_exact": semantics["scenario_cardinality"]
        == expected["scenario_cardinality"]
        and semantics["intent_cardinality"] == expected["intent_cardinality"],
        "intent_prefix_matches_scenario": semantics["intent_prefix_mismatch_rows"] == 0,
        "raw_strings_clear": _string_checks_clear(strings["raw_utterance"], rows)
        and _string_checks_clear(strings["raw_annotated_utterance"], rows),
        "post_nfc_strings_clear": _string_checks_clear(strings["post_nfc_utterance"], rows)
        and _string_checks_clear(strings["post_nfc_annotated_utterance"], rows),
        "raw_non_nfc_counts_exact": strings["raw_utterance"]["non_nfc"]
        == expected["raw_non_nfc_utterances"]
        and strings["raw_annotated_utterance"]["non_nfc"]
        == expected["raw_non_nfc_annotated_utterances"],
        "post_nfc_counts_zero": strings["post_nfc_utterance"]["non_nfc"] == 0
        and strings["post_nfc_annotated_utterance"]["non_nfc"] == 0,
        "nfc_is_idempotent": idempotence["utterance"] == 0
        and idempotence["annotated_utterance"] == 0,
        "nfc_intersection_arithmetic_reconciles": sum(intersection.values()) == aggregate["rows"]
        and intersection["utterance_only"] + intersection["both"]
        == strings["raw_utterance"]["non_nfc"]
        and intersection["annotated_utterance_only"] + intersection["both"]
        == strings["raw_annotated_utterance"]["non_nfc"],
        "annotation_grammar_clear": annotations["malformed_rows"] == 0,
        "annotation_aggregates_exact": annotations["raw_total"] == expected["total_annotations"]
        and annotations["post_nfc_total"] == expected["total_annotations"]
        and annotations["rows_with_annotations"] == expected["rows_with_annotations"]
        and annotations["maximum_per_row"] == expected["maximum_annotations_per_row"]
        and annotations["raw_casefold_slot_label_cardinality"] == expected["slot_label_cardinality"]
        and annotations["post_nfc_casefold_slot_label_cardinality"]
        == expected["slot_label_cardinality"],
        "ordered_annotations_preserved": annotations["ordered_signature_mismatch_rows"] == 0,
        "literal_segments_preserved": annotations["literal_segment_mismatch_rows"] == 0,
        "deannotation_matches_utterance": annotations["deannotation_mismatch_rows"] == 0,
        "nfc_does_not_introduce_slot_label_duplicates": annotations[
            "introduced_nfc_duplicate_slot_label_rows"
        ]
        == 0,
        "nfc_does_not_introduce_content_collisions": all(
            summary["introduced_normalization_collision_groups"] == 0
            for summary in duplicates.values()
        ),
        "fixed_metadata_contract_bound": result["metadata_contract"]
        == {
            "metadata_key_count": HUGGINGFACE_METADATA_KEY_COUNT,
            "metadata_key_sha256": hashlib.sha256(
                HUGGINGFACE_METADATA_KEY.encode("utf-8")
            ).hexdigest(),
            "metadata_bytes": HUGGINGFACE_METADATA_BYTES,
            "metadata_sha256": HUGGINGFACE_METADATA_SHA256,
            "canonical_features_sha256": HUGGINGFACE_CANONICAL_FEATURES_SHA256,
            "scenario_map_sha256": SCENARIO_MAP_SHA256,
            "scenario_cardinality": len(SCENARIO_LABELS),
            "intent_map_sha256": INTENT_MAP_SHA256,
            "intent_cardinality": len(INTENT_LABELS),
        },
    }


def validate_raw_text_free_result(result: Mapping[str, Any]) -> None:
    """Reject unknown fields, arbitrary strings, or credit-bearing results."""

    schema = result.get("schema_version")
    if schema == RECOVERY_SCHEMA:
        _require_keys(
            result,
            frozenset(
                {
                    "schema_version",
                    "status",
                    "metadata_contract",
                    "expected_contract",
                    "aggregate",
                    "checks",
                    "decision",
                    "raw_text_retained",
                    "production_eligible",
                }
            ),
            "pass result",
        )
        _require_keys(
            result["metadata_contract"],
            frozenset(
                {
                    "metadata_key_count",
                    "metadata_key_sha256",
                    "metadata_bytes",
                    "metadata_sha256",
                    "canonical_features_sha256",
                    "scenario_map_sha256",
                    "scenario_cardinality",
                    "intent_map_sha256",
                    "intent_cardinality",
                }
            ),
            "metadata contract",
        )
        _require_keys(
            result["expected_contract"],
            frozenset(
                {
                    "rows",
                    "raw_non_nfc_utterances",
                    "raw_non_nfc_annotated_utterances",
                    "total_annotations",
                    "rows_with_annotations",
                    "maximum_annotations_per_row",
                    "slot_label_cardinality",
                    "scenario_cardinality",
                    "intent_cardinality",
                }
            ),
            "expected contract",
        )
        aggregate = _require_keys(
            result["aggregate"],
            frozenset(
                {
                    "rows",
                    "invalid_row_shapes",
                    "source_scope",
                    "strings",
                    "nfc_intersection",
                    "nfc_idempotence_failures",
                    "semantics",
                    "annotations",
                    "duplicates",
                }
            ),
            "aggregate",
        )
        _require_keys(
            aggregate["source_scope"],
            frozenset(
                {
                    "expected_locale_rows",
                    "unexpected_locale_rows",
                    "expected_partition_rows",
                    "unexpected_partition_rows",
                }
            ),
            "source scope",
        )
        strings = _require_keys(
            aggregate["strings"],
            frozenset(
                {
                    "raw_utterance",
                    "raw_annotated_utterance",
                    "post_nfc_utterance",
                    "post_nfc_annotated_utterance",
                }
            ),
            "string aggregates",
        )
        for name, counts in strings.items():
            _require_keys(counts, _STRING_COUNT_KEYS, f"{name} string counts")
        _require_keys(
            aggregate["nfc_intersection"],
            frozenset({"utterance_only", "annotated_utterance_only", "both", "neither"}),
            "NFC intersection",
        )
        _require_keys(
            aggregate["nfc_idempotence_failures"],
            frozenset({"utterance", "annotated_utterance"}),
            "NFC idempotence",
        )
        _require_keys(
            aggregate["semantics"],
            frozenset(
                {
                    "invalid_scenario_rows",
                    "invalid_intent_rows",
                    "scenario_cardinality",
                    "intent_cardinality",
                    "intent_prefix_mismatch_rows",
                }
            ),
            "semantic aggregates",
        )
        _require_keys(
            aggregate["annotations"],
            frozenset(
                {
                    "raw_total",
                    "post_nfc_total",
                    "rows_with_annotations",
                    "maximum_per_row",
                    "raw_casefold_slot_label_cardinality",
                    "post_nfc_casefold_slot_label_cardinality",
                    "post_nfc_exact_slot_label_cardinality",
                    "malformed_rows",
                    "ordered_signature_mismatch_rows",
                    "literal_segment_mismatch_rows",
                    "deannotation_mismatch_rows",
                    "preexisting_duplicate_slot_label_rows",
                    "introduced_nfc_duplicate_slot_label_rows",
                    "post_nfc_duplicate_slot_label_rows",
                }
            ),
            "annotation aggregates",
        )
        duplicates = _require_keys(
            aggregate["duplicates"],
            frozenset({"utterance", "annotated_utterance", "semantic_tuple"}),
            "duplicate aggregates",
        )
        for name, summary in duplicates.items():
            _require_keys(summary, _DUPLICATE_SUMMARY_KEYS, f"{name} duplicates")
        _require_keys(result["checks"], _PASS_CHECK_KEYS, "pass checks")
        _require_keys(result["decision"], _DECISION_KEYS, "pass decision")
        recomputed_checks = _recompute_pass_checks(result)
        recovery_pass = all(recomputed_checks.values())
        scalar_eligible = (
            recovery_pass and aggregate["annotations"]["post_nfc_duplicate_slot_label_rows"] == 0
        )
        expected_decision = {
            "nfc_recovery_gate_pass": recovery_pass,
            "scalar_argument_adapter_eligible": scalar_eligible,
            "fresh_adapter_contract_successor_eligible": recovery_pass,
            "candidate_construction_eligible": False,
            "candidate_rows_constructed": 0,
            "active_tool_api_credit_rows": 0,
        }
        if (
            result["checks"] != recomputed_checks
            or result["status"] != (PASS_STATUS if recovery_pass else STOP_STATUS)
            or result["decision"] != expected_decision
        ):
            raise ValueError("recovery pass logical reconciliation changed")
    elif schema == RECONCILIATION_SCHEMA:
        _require_keys(
            result,
            frozenset(
                {
                    "schema_version",
                    "status",
                    "checks",
                    "decision",
                    "raw_text_retained",
                    "production_eligible",
                }
            ),
            "reconciliation",
        )
        _require_keys(
            result["checks"],
            frozenset(
                {
                    "pass_results_identical",
                    "both_nfc_recovery_gates_pass",
                    "both_passes_raw_text_free",
                    "candidate_credit_remains_zero",
                }
            ),
            "reconciliation checks",
        )
        _require_keys(result["decision"], _DECISION_KEYS, "reconciliation decision")
        reconciliation_pass = all(result["checks"].values())
        if (
            result["status"]
            != (RECONCILED_STATUS if reconciliation_pass else RECONCILIATION_STOP_STATUS)
            or result["decision"]["nfc_recovery_gate_pass"] is not reconciliation_pass
            or result["decision"]["fresh_adapter_contract_successor_eligible"]
            is not reconciliation_pass
        ):
            raise ValueError("recovery reconciliation logic changed")
    else:
        raise ValueError("unsupported raw-text-free recovery schema")

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ValueError("raw-text-free recovery key type changed")
                walk(child)
        elif isinstance(value, str):
            if value not in _ALLOWED_RESULT_STRINGS and _SHA256_RE.fullmatch(value) is None:
                raise ValueError("arbitrary string prohibited in recovery result")
        elif isinstance(value, bool):
            return
        elif isinstance(value, int) and not isinstance(value, bool):
            return
        else:
            raise ValueError("raw-text-free recovery value contract violated")

    walk(result)
    if result.get("raw_text_retained") is not False:
        raise ValueError("recovery result cannot retain raw text")
    if result.get("production_eligible") is not False:
        raise ValueError("recovery result cannot be production eligible")
    decision = result.get("decision")
    if isinstance(decision, Mapping):
        if decision.get("candidate_construction_eligible") is not False:
            raise ValueError("recovery result cannot authorize candidate construction")
        if decision.get("candidate_rows_constructed") != 0:
            raise ValueError("recovery result cannot construct candidates")
        if decision.get("active_tool_api_credit_rows") != 0:
            raise ValueError("recovery result cannot activate task credit")


def reconcile_recovery_passes(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, Any]:
    """Reconcile two independently streamed pass results without source access."""

    validate_raw_text_free_result(first)
    validate_raw_text_free_result(second)
    identical = first == second
    both_pass = (
        first.get("decision", {}).get("nfc_recovery_gate_pass") is True
        and second.get("decision", {}).get("nfc_recovery_gate_pass") is True
    )
    scalar_adapter_both = (
        first.get("decision", {}).get("scalar_argument_adapter_eligible") is True
        and second.get("decision", {}).get("scalar_argument_adapter_eligible") is True
    )
    reconciliation_pass = identical and both_pass
    result = {
        "schema_version": RECONCILIATION_SCHEMA,
        "status": RECONCILED_STATUS if reconciliation_pass else RECONCILIATION_STOP_STATUS,
        "checks": {
            "pass_results_identical": identical,
            "both_nfc_recovery_gates_pass": both_pass,
            "both_passes_raw_text_free": True,
            "candidate_credit_remains_zero": (
                first.get("decision", {}).get("active_tool_api_credit_rows") == 0
                and second.get("decision", {}).get("active_tool_api_credit_rows") == 0
            ),
        },
        "decision": {
            "nfc_recovery_gate_pass": reconciliation_pass,
            "scalar_argument_adapter_eligible": reconciliation_pass and scalar_adapter_both,
            "fresh_adapter_contract_successor_eligible": reconciliation_pass,
            "candidate_construction_eligible": False,
            "candidate_rows_constructed": 0,
            "active_tool_api_credit_rows": 0,
        },
        "raw_text_retained": False,
        "production_eligible": False,
    }
    validate_raw_text_free_result(result)
    return result


def audit_and_reconcile_two_passes(
    first_rows: Iterable[Mapping[str, Any]],
    second_rows: Iterable[Mapping[str, Any]],
    *,
    expectations: RecoveryExpectations = PRODUCTION_EXPECTATIONS,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Convenience helper that keeps both source passes explicitly sequential."""

    first = audit_recovery_pass(first_rows, expectations=expectations)
    second = audit_recovery_pass(second_rows, expectations=expectations)
    return first, second, reconcile_recovery_passes(first, second)


def sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _seal(value: dict[str, Any], key: str) -> dict[str, Any]:
    value[key] = sha256_value({name: item for name, item in value.items() if name != key})
    return value


def _verify_seal(value: Mapping[str, Any], key: str) -> None:
    internal = value.get(key)
    if not isinstance(internal, str) or _SHA256_RE.fullmatch(internal) is None:
        raise ValueError(f"invalid {key}")
    unsigned = {name: item for name, item in value.items() if name != key}
    if sha256_value(unsigned) != internal:
        raise ValueError(f"{key} changed")


def _json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json_fresh(path: str | Path, value: Mapping[str, Any]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("refusing to overwrite recovery evidence")
    payload = _json_file_bytes(value)
    temporary = destination.with_name(f".{destination.name}.part")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short recovery evidence write")
            offset += written
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    os.close(descriptor)
    os.link(temporary, destination, follow_symlinks=False)
    temporary.unlink()
    return len(payload)


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
        raise ValueError("private recovery root escaped")
    return path


def _validate_file_binding(root: Path, binding: Mapping[str, Any], label: str) -> Path:
    if set(binding) - {"path", "sha256", "internal_sha256", "bytes"}:
        raise ValueError(f"unexpected binding fields: {label}")
    path = _resolve_workspace_path(root, str(binding.get("path", "")))
    if not path.is_file() or sha256_file(path) != binding.get("sha256"):
        raise ValueError(f"binding changed: {label}")
    expected_bytes = binding.get("bytes")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ValueError(f"binding size changed: {label}")
    internal = binding.get("internal_sha256")
    if internal is not None:
        value = _read_object(path)
        if internal not in {
            item
            for name, item in value.items()
            if name.endswith("_sha256") and isinstance(item, str)
        }:
            raise ValueError(f"internal binding changed: {label}")
    return path


def verify_runtime_binding(runtime: Mapping[str, Any]) -> None:
    if dict(runtime) != EXPECTED_RUNTIME_BINDING:
        raise ValueError("runtime binding changed")
    import pyarrow  # type: ignore[import-not-found]
    import pyarrow._parquet  # type: ignore[import-not-found]

    expected_python = Path(EXPECTED_RUNTIME_BINDING["python_executable"]).resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise ValueError("bound Python executable changed")
    if pyarrow.__version__ != EXPECTED_RUNTIME_BINDING["pyarrow_version"]:
        raise ValueError("bound PyArrow version changed")
    runtime_files = {
        "python_executable": expected_python,
        "pyarrow_init": Path(pyarrow.__file__).resolve(),
        "pyarrow_parquet_extension": Path(pyarrow._parquet.__file__).resolve(),
    }
    for name, path in runtime_files.items():
        if not path.is_file() or sha256_file(path) != EXPECTED_RUNTIME_BINDING[f"{name}_sha256"]:
            raise ValueError(f"runtime file changed: {name}")


def _expected_authorization() -> dict[str, Any]:
    return {
        "text": AUTHORIZATION_TEXT,
        "text_sha256": AUTHORIZATION_SHA256,
        "local_only_authorized": True,
        "derived_nfc_audit_authorized": True,
        "candidate_construction_authorized": False,
        "network_gpu_inference_training_authorized": False,
        "registry_promotion_seal_authorized": False,
        "test_dev_validation_audio_authorized": False,
    }


def _expected_audit_contract() -> dict[str, Any]:
    return {
        "allowlisted_columns": list(AUDIT_COLUMNS),
        "expected_top_level_fields": list(EXPECTED_TOP_LEVEL_FIELDS),
        "raw_source_candidate_id_or_exception_text_retained": False,
        "read_id_worker_id_slot_method_or_judgments_values": False,
        "derived_normalization": "NFC",
        "source_mutated": False,
        "aggregate_only": True,
        "pass_results_must_be_identical": True,
    }


def _expected_counts_contract() -> dict[str, Any]:
    return {
        "rows": EXPECTED_SOURCE_ROWS,
        "raw_non_nfc_utterances": EXPECTED_RAW_NON_NFC_UTTERANCES,
        "raw_non_nfc_annotated_utterances": (EXPECTED_RAW_NON_NFC_ANNOTATED_UTTERANCES),
        "total_annotations": EXPECTED_TOTAL_ANNOTATIONS,
        "annotated_rows": EXPECTED_ROWS_WITH_ANNOTATIONS,
        "maximum_annotations": EXPECTED_MAXIMUM_ANNOTATIONS_PER_ROW,
        "slot_labels": EXPECTED_SLOT_LABEL_CARDINALITY,
        "scenarios": 18,
        "intents": 60,
        "candidate_rows_constructed": 0,
        "active_tool_api_credit_rows": 0,
        "production_eligible": False,
    }


def _expected_metadata_binding() -> dict[str, Any]:
    return {
        "metadata_key_count": HUGGINGFACE_METADATA_KEY_COUNT,
        "metadata_bytes": HUGGINGFACE_METADATA_BYTES,
        "metadata_sha256": HUGGINGFACE_METADATA_SHA256,
        "canonical_features_sha256": HUGGINGFACE_CANONICAL_FEATURES_SHA256,
        "scenario_map_sha256": SCENARIO_MAP_SHA256,
        "scenario_cardinality": 18,
        "intent_map_sha256": INTENT_MAP_SHA256,
        "intent_cardinality": 60,
    }


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _read_object(path)
    if set(plan) != {
        "schema_version",
        "status",
        "observed_at",
        "authorization",
        "limits",
        "audit_contract",
        "expected",
        "metadata_binding",
        "runtime_binding",
        "predecessor_bindings",
        "code_bindings",
        "source",
        "output",
        "prohibitions",
        "plan_sha256",
    }:
        raise ValueError("recovery plan fields changed")
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != PLAN_STATUS:
        raise ValueError("recovery plan identity changed")
    _verify_seal(plan, "plan_sha256")
    if (
        hashlib.sha256(AUTHORIZATION_TEXT.encode("utf-8")).hexdigest() != AUTHORIZATION_SHA256
        or plan.get("authorization") != _expected_authorization()
        or plan.get("limits") != EXPECTED_LIMITS
        or plan.get("audit_contract") != _expected_audit_contract()
        or plan.get("expected") != _expected_counts_contract()
        or plan.get("metadata_binding") != _expected_metadata_binding()
        or plan.get("runtime_binding") != EXPECTED_RUNTIME_BINDING
        or plan.get("predecessor_bindings") != EXPECTED_PREDECESSOR_BINDINGS
        or plan.get("source") != EXPECTED_SOURCE_BINDING
        or plan.get("output") != EXPECTED_OUTPUT
        or plan.get("prohibitions") != EXPECTED_PROHIBITIONS
        or plan.get("observed_at") != "2026-08-31"
    ):
        raise ValueError("recovery plan contract changed")

    code = plan.get("code_bindings")
    expected_code_paths = {
        "implementation": "src/lumi/posttraining_massive_bn_bd_nfc_recovery_v2.py",
        "runner": "scripts/recover_sft_massive_bn_bd_nfc_v2.py",
        "verifier": "scripts/verify_sft_massive_bn_bd_nfc_recovery_v2.py",
        "test": "tests/test_massive_bn_bd_nfc_recovery_v2.py",
    }
    if not isinstance(code, Mapping) or set(code) != set(expected_code_paths):
        raise ValueError("recovery code bindings changed")
    for name, expected_path in expected_code_paths.items():
        binding = code[name]
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256"}
            or binding.get("path") != expected_path
        ):
            raise ValueError(f"recovery code binding shape changed: {name}")
        code_path = _resolve_repository_path(root, expected_path)
        if not code_path.is_file() or sha256_file(code_path) != binding.get("sha256"):
            raise ValueError(f"recovery code binding changed: {name}")

    for name, binding in EXPECTED_PREDECESSOR_BINDINGS.items():
        _validate_file_binding(root, binding, name)
    source = _resolve_workspace_path(root, EXPECTED_SOURCE_BINDING["path"])
    if (
        not source.is_file()
        or source.stat().st_size != EXPECTED_SOURCE_BINDING["bytes"]
        or sha256_file(source) != EXPECTED_SOURCE_BINDING["sha256"]
    ):
        raise ValueError("bound MASSIVE source changed")
    _resolve_private_root(root, EXPECTED_OUTPUT["private_root"])
    for name in ("report_path", "verification_path"):
        evidence = _resolve_repository_path(root, EXPECTED_OUTPUT[name])
        if evidence.parent != (root / "evidence").resolve():
            raise ValueError("recovery evidence path changed")
    verify_runtime_binding(plan["runtime_binding"])
    return plan


def preflight_paths(
    plan: Mapping[str, Any], *, repository_root: str | Path
) -> tuple[Path, Path, Path, Path]:
    root = Path(repository_root).resolve()
    output = plan["output"]
    private_root = _resolve_private_root(root, str(output["private_root"]))
    marker = private_root / str(output["marker_name"])
    report = _resolve_repository_path(root, str(output["report_path"]))
    verification = _resolve_repository_path(root, str(output["verification_path"]))
    source = _resolve_workspace_path(root, str(plan["source"]["path"]))
    for path in (marker, report, verification):
        if path.exists() or path.is_symlink():
            raise FileExistsError("fresh recovery output already exists")
    return marker, report, verification, source


def start_marker(path: Path) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.parent.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("private recovery directory changed")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        written = os.write(descriptor, RECOVERY_MARKER)
        if written != len(RECOVERY_MARKER):
            raise OSError("short recovery marker write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if path.stat().st_size != len(RECOVERY_MARKER) or sha256_file(path) != RECOVERY_MARKER_SHA256:
        raise ValueError("recovery marker changed")


def _report_resource_accounting() -> dict[str, Any]:
    return {
        "sequential_source_passes": 2,
        "source_row_reads": 23_028,
        "candidate_or_baseline_reads": 0,
        "test_dev_validation_row_reads": 0,
        "private_write_bytes": len(RECOVERY_MARKER),
        "network_bytes": 0,
        "gpu_actions": 0,
        "inference_calls": 0,
        "training_actions": 0,
        "network_retries": 0,
    }


def build_report(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    first_pass: Mapping[str, Any],
    second_pass: Mapping[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    reconciliation = reconcile_recovery_passes(first_pass, second_pass)
    gate_pass = reconciliation["decision"]["nfc_recovery_gate_pass"] is True
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": REPORT_PASS_STATUS if gate_pass else REPORT_STOP_STATUS,
        "observed_at": plan["observed_at"],
        "authorization_sha256": AUTHORIZATION_SHA256,
        "plan_file_sha256": plan_file_sha256,
        "plan_internal_sha256": plan["plan_sha256"],
        "predecessor_bindings_sha256": sha256_value(plan["predecessor_bindings"]),
        "source": dict(EXPECTED_SOURCE_BINDING),
        "derived_transformation": {
            "normalization": "NFC",
            "fields": ["utt", "annot_utt"],
            "source_mutated": False,
            "raw_or_normalized_lexical_values_retained_after_each_row": False,
            "collision_state_retains_domain_separated_sha256_only": True,
        },
        "first_pass": dict(first_pass),
        "second_pass": dict(second_pass),
        "reconciliation": reconciliation,
        "resource_accounting": _report_resource_accounting(),
        "elapsed_seconds_millis": int(elapsed_seconds * 1_000),
        "decision": dict(reconciliation["decision"]),
        "next_gate": {
            "pure_non_side_effecting_parser_contract_required": True,
            "minimum_owner_review_rows": 64,
            "minimum_owner_acceptance_rate_basis_points": 9_500,
            "candidate_construction_authorized_here": False,
        },
        "raw_text_retained": False,
        "production_eligible": False,
    }
    _seal(report, "report_sha256")
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if set(report) != {
        "schema_version",
        "status",
        "observed_at",
        "authorization_sha256",
        "plan_file_sha256",
        "plan_internal_sha256",
        "predecessor_bindings_sha256",
        "source",
        "derived_transformation",
        "first_pass",
        "second_pass",
        "reconciliation",
        "resource_accounting",
        "elapsed_seconds_millis",
        "decision",
        "next_gate",
        "raw_text_retained",
        "production_eligible",
        "report_sha256",
    }:
        raise ValueError("recovery report fields changed")
    _verify_seal(report, "report_sha256")
    validate_raw_text_free_result(report["first_pass"])
    validate_raw_text_free_result(report["second_pass"])
    validate_raw_text_free_result(report["reconciliation"])
    expected_pass_contract = {
        "rows": EXPECTED_SOURCE_ROWS,
        "raw_non_nfc_utterances": EXPECTED_RAW_NON_NFC_UTTERANCES,
        "raw_non_nfc_annotated_utterances": (EXPECTED_RAW_NON_NFC_ANNOTATED_UTTERANCES),
        "total_annotations": EXPECTED_TOTAL_ANNOTATIONS,
        "rows_with_annotations": EXPECTED_ROWS_WITH_ANNOTATIONS,
        "maximum_annotations_per_row": EXPECTED_MAXIMUM_ANNOTATIONS_PER_ROW,
        "slot_label_cardinality": EXPECTED_SLOT_LABEL_CARDINALITY,
        "scenario_cardinality": 18,
        "intent_cardinality": 60,
    }
    for name in ("first_pass", "second_pass"):
        if (
            report[name].get("expected_contract") != expected_pass_contract
            or report[name].get("aggregate", {}).get("rows") != EXPECTED_SOURCE_ROWS
        ):
            raise ValueError("recovery report pass scope changed")
    rebuilt = reconcile_recovery_passes(report["first_pass"], report["second_pass"])
    gate_pass = rebuilt["decision"]["nfc_recovery_gate_pass"] is True
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or report.get("status") != (REPORT_PASS_STATUS if gate_pass else REPORT_STOP_STATUS)
        or report.get("observed_at") != "2026-08-31"
        or report.get("authorization_sha256") != AUTHORIZATION_SHA256
        or any(
            not isinstance(report.get(name), str)
            or _SHA256_RE.fullmatch(str(report.get(name))) is None
            for name in (
                "plan_file_sha256",
                "plan_internal_sha256",
                "predecessor_bindings_sha256",
                "report_sha256",
            )
        )
        or report.get("source") != EXPECTED_SOURCE_BINDING
        or report.get("derived_transformation")
        != {
            "normalization": "NFC",
            "fields": ["utt", "annot_utt"],
            "source_mutated": False,
            "raw_or_normalized_lexical_values_retained_after_each_row": False,
            "collision_state_retains_domain_separated_sha256_only": True,
        }
        or report.get("reconciliation") != rebuilt
        or report.get("resource_accounting") != _report_resource_accounting()
        or not isinstance(report.get("elapsed_seconds_millis"), int)
        or isinstance(report.get("elapsed_seconds_millis"), bool)
        or int(report["elapsed_seconds_millis"]) < 0
        or int(report["elapsed_seconds_millis"])
        > EXPECTED_LIMITS["maximum_runtime_seconds"] * 1_000
        or report.get("decision") != rebuilt["decision"]
        or report.get("next_gate")
        != {
            "pure_non_side_effecting_parser_contract_required": True,
            "minimum_owner_review_rows": 64,
            "minimum_owner_acceptance_rate_basis_points": 9_500,
            "candidate_construction_authorized_here": False,
        }
        or report.get("raw_text_retained") is not False
        or report.get("production_eligible") is not False
    ):
        raise ValueError("recovery report contract changed")
    if report["decision"].get("candidate_construction_eligible") is not False:
        raise ValueError("recovery report cannot authorize candidate construction")


def execute_recovery(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
) -> tuple[dict[str, Any], Path]:
    marker, report_path, _verification, source = preflight_paths(
        plan, repository_root=repository_root
    )
    start_marker(marker)
    started = time.monotonic()
    first = aggregate_parquet_pass(str(source))
    if time.monotonic() - started > EXPECTED_LIMITS["maximum_runtime_seconds"]:
        raise TimeoutError("NFC recovery runtime cap reached")
    second = aggregate_parquet_pass(str(source))
    elapsed = time.monotonic() - started
    if elapsed > EXPECTED_LIMITS["maximum_runtime_seconds"]:
        raise TimeoutError("NFC recovery runtime cap reached")
    report = build_report(
        plan=plan,
        plan_file_sha256=plan_file_sha256,
        first_pass=first,
        second_pass=second,
        elapsed_seconds=elapsed,
    )
    write_json_fresh(report_path, report)
    return report, report_path


def _validate_marker(path: Path) -> None:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != len(RECOVERY_MARKER)
        or sha256_file(path) != RECOVERY_MARKER_SHA256
        or path.read_bytes() != RECOVERY_MARKER
    ):
        raise ValueError("recovery marker changed")


def build_verification(
    *,
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    plan_file_sha256: str,
    report_file_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    validate_report(report)
    if (
        report.get("plan_file_sha256") != plan_file_sha256
        or report.get("plan_internal_sha256") != plan.get("plan_sha256")
        or report.get("predecessor_bindings_sha256") != sha256_value(plan["predecessor_bindings"])
        or report_file_sha256
        != sha256_file(_resolve_repository_path(root, str(plan["output"]["report_path"])))
    ):
        raise ValueError("recovery report or plan binding changed")
    private_root = _resolve_private_root(root, str(plan["output"]["private_root"]))
    marker = private_root / str(plan["output"]["marker_name"])
    _validate_marker(marker)
    checks = {
        "plan_and_all_bindings_rehashed": True,
        "source_file_rehashed_with_zero_source_row_reads": True,
        "exact_marker_preserved": True,
        "first_pass_is_raw_text_free": True,
        "second_pass_is_raw_text_free": True,
        "embedded_passes_reconcile": (
            report["reconciliation"]
            == reconcile_recovery_passes(report["first_pass"], report["second_pass"])
        ),
        "resource_accounting_exact": (
            report["resource_accounting"] == _report_resource_accounting()
        ),
        "candidate_and_active_credit_remain_zero": (
            report["decision"]["candidate_rows_constructed"] == 0
            and report["decision"]["active_tool_api_credit_rows"] == 0
            and report["decision"]["candidate_construction_eligible"] is False
        ),
        "prior_evidence_and_source_remain_immutable": True,
        "no_network_gpu_inference_training_registry_seal_or_remote_action": True,
    }
    if not all(checks.values()):
        raise ValueError("recovery structural verification failed")
    verification = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": VERIFICATION_STATUS,
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "plan_internal_sha256": plan["plan_sha256"],
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": report["report_sha256"],
        "marker_sha256": RECOVERY_MARKER_SHA256,
        "checks": checks,
        "resource_accounting": {
            "source_row_reads": 0,
            "candidate_or_baseline_reads": 0,
            "test_dev_validation_row_reads": 0,
            "network_bytes": 0,
            "gpu_actions": 0,
            "inference_calls": 0,
            "training_actions": 0,
        },
        "decision": dict(report["decision"]),
        "raw_text_retained": False,
        "production_eligible": False,
    }
    _seal(verification, "verification_sha256")
    validate_verification(verification)
    return verification


def validate_verification(verification: Mapping[str, Any]) -> None:
    if set(verification) != {
        "schema_version",
        "status",
        "observed_at",
        "plan_file_sha256",
        "plan_internal_sha256",
        "report_file_sha256",
        "report_internal_sha256",
        "marker_sha256",
        "checks",
        "resource_accounting",
        "decision",
        "raw_text_retained",
        "production_eligible",
        "verification_sha256",
    }:
        raise ValueError("recovery verification fields changed")
    _verify_seal(verification, "verification_sha256")
    expected_resource = {
        "source_row_reads": 0,
        "candidate_or_baseline_reads": 0,
        "test_dev_validation_row_reads": 0,
        "network_bytes": 0,
        "gpu_actions": 0,
        "inference_calls": 0,
        "training_actions": 0,
    }
    if (
        verification.get("schema_version") != VERIFICATION_SCHEMA
        or verification.get("status") != VERIFICATION_STATUS
        or verification.get("observed_at") != "2026-08-31"
        or verification.get("marker_sha256") != RECOVERY_MARKER_SHA256
        or any(
            not isinstance(verification.get(name), str)
            or _SHA256_RE.fullmatch(str(verification.get(name))) is None
            for name in (
                "plan_file_sha256",
                "plan_internal_sha256",
                "report_file_sha256",
                "report_internal_sha256",
                "verification_sha256",
            )
        )
        or not isinstance(verification.get("checks"), Mapping)
        or not verification["checks"]
        or not all(value is True for value in verification["checks"].values())
        or verification.get("resource_accounting") != expected_resource
        or not isinstance(verification.get("decision"), Mapping)
        or verification["decision"].get("candidate_construction_eligible") is not False
        or verification["decision"].get("candidate_rows_constructed") != 0
        or verification["decision"].get("active_tool_api_credit_rows") != 0
        or verification.get("raw_text_retained") is not False
        or verification.get("production_eligible") is not False
    ):
        raise ValueError("recovery verification contract changed")


__all__ = [
    "AUTHORIZATION_SHA256",
    "AUTHORIZATION_TEXT",
    "EXPECTED_MAXIMUM_ANNOTATIONS_PER_ROW",
    "EXPECTED_RAW_NON_NFC_ANNOTATED_UTTERANCES",
    "EXPECTED_RAW_NON_NFC_UTTERANCES",
    "EXPECTED_ROWS_WITH_ANNOTATIONS",
    "EXPECTED_SLOT_LABEL_CARDINALITY",
    "EXPECTED_SOURCE_ROWS",
    "EXPECTED_TOTAL_ANNOTATIONS",
    "HUGGINGFACE_CANONICAL_FEATURES_SHA256",
    "HUGGINGFACE_METADATA_BYTES",
    "HUGGINGFACE_METADATA_SHA256",
    "INTENT_LABELS",
    "INTENT_MAP_SHA256",
    "PRODUCTION_EXPECTATIONS",
    "RECOVERY_MARKER_SHA256",
    "RecoveryExpectations",
    "SCENARIO_LABELS",
    "SCENARIO_MAP_SHA256",
    "aggregate_parquet_pass",
    "audit_and_reconcile_two_passes",
    "audit_recovery_pass",
    "build_verification",
    "execute_recovery",
    "load_plan",
    "preflight_paths",
    "reconcile_recovery_passes",
    "sha256_file",
    "validate_report",
    "validate_raw_text_free_result",
    "validate_verification",
    "write_json_fresh",
]
