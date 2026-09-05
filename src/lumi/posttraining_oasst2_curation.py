"""Bounded, fail-closed curation for the pinned OASST2 ready-tree export."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, median
from typing import Any

from .data_quality import (
    BenchmarkRegistry,
    MinHashNearDuplicateIndex,
    normalize_for_matching,
    validate_quality_config,
)
from .posttraining_data import SFTPolicy, load_sft_policy
from .posttraining_format import ChatTemplate, encode_sft_messages, load_chat_template
from .posttraining_owner_review import build_owner_review_html

PLAN_SCHEMA_VERSION = "lumi.sft-oasst2-ready-tree-curation-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-oasst2-ready-tree-curation-report.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-oasst2-ready-tree-curation-verification.v0.1"
ATTRIBUTION_SCHEMA_VERSION = "lumi.sft-oasst2-attribution.v0.1"

SOURCE_ID = "openassistant_oasst2"
SOURCE_DATASET_ID = "OpenAssistant/oasst2"
SOURCE_REVISION = "179dd21fc55192153d94adb0e0ce8f69e222bf75"
SOURCE_LICENSE = "apache-2.0"
TREE_OBJECT = "2023-11-05_oasst2_ready.trees.jsonl.gz"
ADAPTER_ID = "oasst2_ready_tree_ranked_paths_v0.1"

EXPECTED_OBJECTS = (
    {
        "git_blob_sha1": "ce94775b146d09acf5c5ea31992a9f322e12de76",
        "path": "README.md",
        "size_bytes": 10_606,
    },
    {
        "path": TREE_OBJECT,
        "sha256": "7a886a16ccfc1173c4f00a6897523e3c95b2785a86ee44a18a98f4f2807ee29b",
        "size_bytes": 54_370_156,
    },
)
EXPECTED_AUTHORIZATION = {
    "candidate_registry_promotion_authorized": False,
    "cpu_only_acquisition_and_curation_authorized": True,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "network_access_authorized": True,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "unrelated_dataset_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_decoded_bytes": 536_870_912,
    "maximum_input_messages": 135_174,
    "maximum_input_trees": 13_854,
    "maximum_network_bytes": 67_108_864,
    "maximum_retained_train_supervised_tokens": 2_800_000,
    "maximum_runtime_minutes": 180,
    "owner_sample_records": 32,
    "target_retained_train_supervised_tokens": 2_600_000,
    "validation_fraction": 0.05,
}
EXPECTED_POLICY = {
    "allowed_languages": ["en"],
    "assistant_rank_tiers": [0, 1, 2],
    "assistant_quality_minimum": 0.75,
    "assistant_helpfulness_minimum": 0.75,
    "assistant_fails_task_maximum": 0.0,
    "hard_risk_label_maximum": 0.0,
    "hard_risk_labels": [
        "hate_speech",
        "lang_mismatch",
        "not_appropriate",
        "pii",
        "sexual_content",
        "spam",
    ],
    "minimum_review_count": 3,
    "near_duplicate": {
        "comparison_unit": "role_tagged_full_conversation",
        "num_perm": 128,
        "package_version": "2.0.0",
        "scheme": "affine32",
        "seed": 1,
        "shingle_tokens": 5,
        "threshold": 0.9,
    },
    "prompt_quality_minimum": 0.5,
    "required_tree_state": "ready_for_export",
    "split_assignment": "sha256_tree_id_modulo_20_validation_v0.1",
    "target_task_train_supervised_tokens": {
        "conversation": 330_000,
        "factual_qa": 1_150_000,
        "writing_summarization": 1_120_000,
    },
}
EXPECTED_OUTPUT = {
    "private_root": (
        "../lumi-artifacts/4b-sft-oasst2-ready-tree-curation-v0.1/16394e94d0eb/attempt-001"
    ),
    "raw_text_free_report_path": ("evidence/lumi-4b-sft-oasst2-ready-tree-curation-v0.1.json"),
    "raw_text_free_verification_path": (
        "evidence/lumi-4b-sft-oasst2-ready-tree-curation-verification-v0.1.json"
    ),
    "raw_training_text_in_git": False,
}
EXPECTED_BINDING_PATHS = {
    "benchmark_registry": "data/benchmark-signatures-v0.3.jsonl.gz",
    "chat_template": "configs/lumi-4b-chat-template-v0.1.json",
    "curation_implementation": "src/lumi/posttraining_oasst2_curation.py",
    "curation_runner": "scripts/run_oasst2_ready_tree_curation.py",
    "curation_verifier": "scripts/verify_oasst2_ready_tree_curation.py",
    "data_contract": "configs/lumi-4b-sft-data-contract-v0.1.json",
    "data_quality_implementation": "src/lumi/data_quality.py",
    "encoding_implementation": "src/lumi/posttraining_format.py",
    "filtering_implementation": "src/lumi/posttraining_data.py",
    "mix_closure_report": ("evidence/lumi-4b-sft-10m-mix-closure-source-selection-v0.1.json"),
    "mix_closure_verification": (
        "evidence/lumi-4b-sft-10m-mix-closure-source-selection-verification-v0.1.json"
    ),
    "owner_review_implementation": "src/lumi/posttraining_owner_review.py",
    "quality_config": "configs/data-quality-v0.3.json",
    "source_registry": "configs/lumi-4b-sft-source-registry-v0.12.json",
    "test": "tests/test_oasst2_ready_tree_curation.py",
    "tokenizer_config": (
        "../lumi-artifacts/4b-segment2-throughput-gate-v0.1/8bedfbc868e8/"
        "staging/input/tokenizer/tokenizer_config.json"
    ),
    "tokenizer_json": (
        "../lumi-artifacts/4b-segment2-throughput-gate-v0.1/8bedfbc868e8/"
        "staging/input/tokenizer/tokenizer.json"
    ),
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_NODE_FIELDS = frozenset(
    {
        "created_date",
        "deleted",
        "detoxify",
        "emojis",
        "events",
        "labels",
        "lang",
        "message_id",
        "message_tree_id",
        "model_name",
        "parent_id",
        "rank",
        "replies",
        "review_count",
        "review_result",
        "role",
        "synthetic",
        "text",
        "tree_state",
        "user_id",
    }
)
_WRAPPER_FIELDS = frozenset({"message_tree_id", "origin", "prompt", "tree_state"})


@dataclass(frozen=True)
class TreeSelection:
    records: tuple[dict[str, Any], ...]
    attributions: tuple[dict[str, Any], ...]
    metadata: Mapping[str, Mapping[str, Any]]
    metrics: Mapping[str, Any]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: str | Path) -> str:
    value = Path(path).read_bytes()
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value).hexdigest()  # noqa: S324 - Git identity.


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def resolve_binding(
    repository_root: str | Path,
    binding: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    root = Path(repository_root).resolve()
    workspace = root.parent.resolve()
    if binding.get("path") != EXPECTED_BINDING_PATHS.get(name):
        raise ValueError(f"OASST2 binding path changed: {name}")
    expected = binding.get("sha256")
    if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
        raise ValueError(f"OASST2 binding SHA-256 is invalid: {name}")
    resolved = (root / str(binding["path"])).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"OASST2 binding escapes workspace: {name}")
    if not resolved.is_file() or file_sha256(resolved) != expected:
        raise ValueError(f"OASST2 binding changed: {name}")
    return resolved


def load_oasst2_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    plan = _load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported OASST2 curation plan")
    if plan.get("status") != "authorized_cpu_only_ready_tree_curation_no_promotion":
        raise ValueError("OASST2 plan status changed")
    for field, expected in (
        ("authorization", EXPECTED_AUTHORIZATION),
        ("limits", EXPECTED_LIMITS),
        ("policy", EXPECTED_POLICY),
        ("output", EXPECTED_OUTPUT),
    ):
        if plan.get(field) != expected:
            raise ValueError(f"OASST2 plan {field} changed")
    source = plan.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("OASST2 source is missing")
    if source.get("source_id") != SOURCE_ID or source.get("dataset_id") != SOURCE_DATASET_ID:
        raise ValueError("OASST2 source identity changed")
    if source.get("revision") != SOURCE_REVISION or source.get("license_id") != SOURCE_LICENSE:
        raise ValueError("OASST2 revision or license changed")
    if source.get("objects") != list(EXPECTED_OBJECTS):
        raise ValueError("OASST2 exact object inventory changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(EXPECTED_BINDING_PATHS):
        raise ValueError("OASST2 binding set changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"OASST2 binding must be an object: {name}")
        resolve_binding(repository_root, binding, name=name)
    return plan


def verify_raw_objects(raw_root: str | Path) -> dict[str, Any]:
    root = Path(raw_root)
    observed: list[dict[str, Any]] = []
    for specification in EXPECTED_OBJECTS:
        path = root / specification["path"]
        if not path.is_file() or path.stat().st_size != specification["size_bytes"]:
            raise ValueError(f"OASST2 object size changed: {specification['path']}")
        item = {
            "bytes": path.stat().st_size,
            "object_path": specification["path"],
            "sha256": file_sha256(path),
        }
        if "sha256" in specification and item["sha256"] != specification["sha256"]:
            raise ValueError(f"OASST2 object SHA-256 changed: {specification['path']}")
        if "git_blob_sha1" in specification:
            item["git_blob_sha1"] = git_blob_sha1(path)
            if item["git_blob_sha1"] != specification["git_blob_sha1"]:
                raise ValueError(f"OASST2 Git blob changed: {specification['path']}")
        observed.append(item)
    if sum(item["bytes"] for item in observed) != 54_380_762:
        raise ValueError("OASST2 exact object byte total changed")
    readme = (root / "README.md").read_text(encoding="utf-8")
    if "apache-2.0" not in readme.casefold() and "apache 2.0" not in readme.casefold():
        raise ValueError("OASST2 Apache-2.0 documentation evidence changed")
    return {
        "exact_authorized_objects_only": True,
        "network_bytes": sum(item["bytes"] for item in observed),
        "object_count": len(observed),
        "objects": observed,
    }


def label_measurement(node: Mapping[str, Any], name: str) -> tuple[float, int] | None:
    labels = node.get("labels")
    if not isinstance(labels, Mapping):
        return None
    item = labels.get(name)
    if not isinstance(item, Mapping):
        return None
    value, count = item.get("value"), item.get("count")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
    ):
        return None
    return float(value), count


def _node_reasons(node: Mapping[str, Any], *, role: str) -> tuple[str, ...]:
    reasons: list[str] = []
    if set(node) != _NODE_FIELDS:
        reasons.append("node_schema")
    if node.get("role") != role:
        reasons.append("role")
    if node.get("deleted") is not False:
        reasons.append("deleted")
    if node.get("synthetic") is not False or node.get("model_name") is not None:
        reasons.append("synthetic_or_model_generated")
    if node.get("review_result") is not True:
        reasons.append("review_result")
    reviews = node.get("review_count")
    if isinstance(reviews, bool) or not isinstance(reviews, int) or reviews < 3:
        reasons.append("review_count")
    if node.get("lang") != "en":
        reasons.append("language")
    if not isinstance(node.get("text"), str) or not node["text"].strip():
        reasons.append("empty_text")
    quality = label_measurement(node, "quality")
    floor = 0.75 if role == "assistant" else 0.5
    if quality is None or quality[0] < floor:
        reasons.append("quality")
    for label in EXPECTED_POLICY["hard_risk_labels"]:
        measurement = label_measurement(node, label)
        if measurement is None or measurement[0] > 0:
            reasons.append(f"risk_label:{label}")
    if role == "assistant":
        helpfulness = label_measurement(node, "helpfulness")
        fails_task = label_measurement(node, "fails_task")
        if helpfulness is None or helpfulness[0] < 0.75:
            reasons.append("helpfulness")
        if fails_task is None or fails_task[0] > 0:
            reasons.append("fails_task")
        rank = node.get("rank")
        if isinstance(rank, bool) or rank not in {0, 1, 2}:
            reasons.append("assistant_rank")
    return tuple(dict.fromkeys(reasons))


def task_category(prompt: str) -> str:
    normalized = f" {normalize_for_matching(prompt)} "
    coding = (
        " code ",
        " coding ",
        " function ",
        " javascript ",
        " python ",
        " regex ",
        " sql ",
        " program ",
        " algorithm ",
        "```",
    )
    writing = (
        " summar",
        " rewrite",
        " rephrase",
        " proofread",
        " edit ",
        " write ",
        " draft ",
        " compose",
        " story ",
        " poem ",
        " essay ",
        " email ",
        " letter ",
        " blog ",
        " article ",
        " paragraph ",
        " title ",
        " caption ",
        " outline ",
        " translate",
        " creative ",
        " generate a ",
        " create a ",
    )
    conversation = (
        " hello ",
        " hi ",
        " hey ",
        " chat ",
        " advice ",
        " recommend",
        " opinion ",
        " roleplay",
        " pretend ",
        " feel ",
        " should i ",
        " help me decide",
        " what do you think",
        " talk about",
    )
    factual = (
        " what ",
        " who ",
        " when ",
        " where ",
        " why ",
        " how ",
        " explain",
        " define",
        " describe",
        " compare",
        " difference",
        " list ",
        " tell me",
        " which ",
        " history",
        " meaning",
    )
    if any(marker in normalized for marker in coding):
        return "coding"
    if any(marker in normalized for marker in writing):
        return "writing_summarization"
    if any(marker in normalized for marker in conversation):
        return "conversation"
    if any(marker in normalized for marker in factual):
        return "factual_qa"
    return "conversation"


def tree_split(message_tree_id: str) -> str:
    digest = hashlib.sha256(f"oasst2-tree-split-v0.1\0{message_tree_id}".encode()).digest()
    return "validation" if int.from_bytes(digest[:8], "big") % 20 == 0 else "train"


def _quality_value(node: Mapping[str, Any], name: str) -> float:
    measurement = label_measurement(node, name)
    if measurement is None:
        raise ValueError(f"missing validated OASST2 label: {name}")
    return measurement[0]


def _candidate(
    *,
    nodes: Sequence[Mapping[str, Any]],
    endpoint: Mapping[str, Any],
    tree_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    messages = [
        {
            "content": str(node["text"]),
            "role": "user" if node["role"] == "prompter" else "assistant",
        }
        for node in nodes
    ]
    split = tree_split(tree_id)
    category = task_category(messages[0]["content"])
    content_sha256 = sha256_value(messages)
    endpoint_id = str(endpoint["message_id"])
    record_digest = sha256_value({"endpoint": endpoint_id, "revision": SOURCE_REVISION})
    record_id = f"{SOURCE_ID}-{record_digest[:24]}"
    assistant_nodes = [node for node in nodes if node["role"] == "assistant"]
    assistant_ranks = [int(node["rank"]) for node in assistant_nodes]
    score = min(_quality_value(node, "quality") for node in assistant_nodes)
    transform_sha256 = sha256_value(
        {
            "adapter": ADAPTER_ID,
            "content_sha256": content_sha256,
            "endpoint": endpoint_id,
            "revision": SOURCE_REVISION,
            "split": split,
        }
    )
    attribution = {
        "assistant_ranks": assistant_ranks,
        "content_sha256": content_sha256,
        "dataset_id": SOURCE_DATASET_ID,
        "endpoint_message_id": endpoint_id,
        "language_tags": [str(node["lang"]) for node in nodes],
        "license_id": SOURCE_LICENSE,
        "message_ids": [str(node["message_id"]) for node in nodes],
        "message_tree_id": tree_id,
        "record_id": record_id,
        "revision": SOURCE_REVISION,
        "schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "source_id": SOURCE_ID,
        "source_object_path": TREE_OBJECT,
        "source_object_sha256": EXPECTED_OBJECTS[1]["sha256"],
        "split": split,
        "task_category": category,
        "transform_sha256": transform_sha256,
    }
    record = {
        "language": "en",
        "messages": messages,
        "provenance": {
            "consent_basis": "source_level_open_license",
            "content_sha256": content_sha256,
            "transform_sha256": transform_sha256,
        },
        "quality": {
            "pii_scan": "passed",
            "review_method": "oasst2_upstream_human_rank_and_quality_plus_lumi_filters_v0.1",
            "review_status": "accepted",
            "score": score,
            "secret_scan": "passed",
        },
        "record_id": record_id,
        "safety": {
            "policy_version": "lumi-oasst2-upstream-labels-plus-pattern-scan-v0.1",
            "risk_labels": [],
        },
        "schema_version": "lumi.sft.v0.1",
        "source": {
            "dataset_id": SOURCE_DATASET_ID,
            "document_id": endpoint_id,
            "license_id": SOURCE_LICENSE,
            "revision": SOURCE_REVISION,
            "source_id": SOURCE_ID,
            "uri": (
                f"https://huggingface.co/datasets/{SOURCE_DATASET_ID}/blob/"
                f"{SOURCE_REVISION}/{TREE_OBJECT}#message_id={endpoint_id}"
            ),
        },
        "split": split,
        "task_category": category,
    }
    metadata = {
        "depth": len(messages),
        "endpoint_quality": _quality_value(endpoint, "quality"),
        "maximum_assistant_rank": max(assistant_ranks),
        "message_tree_id": tree_id,
        "terminal": not bool(endpoint.get("replies")),
    }
    return record, attribution, metadata


def load_tree_candidates(
    path: str | Path,
    *,
    maximum_decoded_bytes: int,
    maximum_messages: int,
    maximum_trees: int,
) -> TreeSelection:
    records: list[dict[str, Any]] = []
    attributions: list[dict[str, Any]] = []
    metadata: dict[str, Mapping[str, Any]] = {}
    node_rejections: Counter[str] = Counter()
    candidates_by_rank: Counter[str] = Counter()
    candidates_by_split: Counter[str] = Counter()
    candidates_by_task: Counter[str] = Counter()
    message_ids: set[str] = set()
    decoded_bytes = 0
    messages_seen = 0
    trees_seen = 0

    with gzip.open(path, "rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            decoded_bytes += len(raw_line)
            if decoded_bytes > maximum_decoded_bytes:
                raise ValueError("OASST2 decoded byte limit exceeded")
            wrapper = json.loads(raw_line)
            if not isinstance(wrapper, Mapping) or set(wrapper) != _WRAPPER_FIELDS:
                raise ValueError(f"OASST2 wrapper schema changed at line {line_number}")
            trees_seen += 1
            if trees_seen > maximum_trees:
                raise ValueError("OASST2 tree limit exceeded")
            tree_id = wrapper.get("message_tree_id")
            if not isinstance(tree_id, str) or not tree_id:
                raise ValueError("OASST2 tree ID changed")
            if wrapper.get("tree_state") != "ready_for_export":
                raise ValueError("OASST2 non-ready wrapper encountered")
            prompt = wrapper.get("prompt")
            if not isinstance(prompt, Mapping):
                raise ValueError("OASST2 prompt tree changed")

            def visit(
                node: Mapping[str, Any],
                *,
                ancestors: tuple[Mapping[str, Any], ...],
                path_clean: bool,
                expected_role: str,
                _tree_id: str = tree_id,
            ) -> None:
                nonlocal messages_seen
                messages_seen += 1
                if messages_seen > maximum_messages:
                    raise ValueError("OASST2 message limit exceeded")
                message_id = node.get("message_id")
                if not isinstance(message_id, str) or not message_id:
                    raise ValueError("OASST2 message ID changed")
                if message_id in message_ids:
                    raise ValueError("OASST2 duplicate message ID encountered")
                message_ids.add(message_id)
                if node.get("message_tree_id") is not None:
                    raise ValueError("OASST2 ready-tree node tree pointer changed")
                reasons = _node_reasons(node, role=expected_role)
                node_rejections.update(reasons)
                clean = path_clean and not reasons
                path = (*ancestors, node)
                if clean and expected_role == "assistant":
                    record, attribution, row_metadata = _candidate(
                        nodes=path, endpoint=node, tree_id=_tree_id
                    )
                    records.append(record)
                    attributions.append(attribution)
                    metadata[record["record_id"]] = row_metadata
                    candidates_by_rank[str(row_metadata["maximum_assistant_rank"])] += 1
                    candidates_by_split[record["split"]] += 1
                    candidates_by_task[record["task_category"]] += 1
                replies = node.get("replies")
                if not isinstance(replies, list):
                    raise ValueError("OASST2 replies changed")
                next_role = "assistant" if expected_role == "prompter" else "prompter"
                for reply in replies:
                    if not isinstance(reply, Mapping):
                        raise ValueError("OASST2 reply changed")
                    if reply.get("parent_id") != message_id:
                        raise ValueError("OASST2 parent linkage changed")
                    visit(
                        reply,
                        ancestors=path,
                        path_clean=clean,
                        expected_role=next_role,
                    )

            if prompt.get("parent_id") is not None:
                raise ValueError("OASST2 root prompt parent changed")
            visit(prompt, ancestors=(), path_clean=True, expected_role="prompter")

    metrics = {
        "candidate_records": len(records),
        "candidates_by_maximum_assistant_rank": dict(sorted(candidates_by_rank.items())),
        "candidates_by_split": dict(sorted(candidates_by_split.items())),
        "candidates_by_task": dict(sorted(candidates_by_task.items())),
        "decoded_bytes": decoded_bytes,
        "input_messages": messages_seen,
        "input_trees": trees_seen,
        "node_rejection_counts": dict(sorted(node_rejections.items())),
        "tree_state": "ready_for_export",
    }
    return TreeSelection(
        records=tuple(records),
        attributions=tuple(attributions),
        metadata=metadata,
        metrics=metrics,
    )


def apply_conversation_near_dedup(
    records: Sequence[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Mapping[str, Any]],
    near_duplicate_config: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    index = MinHashNearDuplicateIndex(near_duplicate_config)
    ordered = sorted(
        records,
        key=lambda row: (
            int(metadata[str(row["record_id"])]["maximum_assistant_rank"]),
            -float(row["quality"]["score"]),
            not bool(metadata[str(row["record_id"])]["terminal"]),
            hashlib.sha256(str(row["record_id"]).encode()).hexdigest(),
        ),
    )
    accepted: list[Mapping[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for record in ordered:
        record_id = str(record["record_id"])
        text = normalize_for_matching(
            "\n".join(f"{message['role']} {message['content']}" for message in record["messages"])
        )
        signature = index.signature(text)
        duplicate_of = index.duplicate_of(signature)
        keep = duplicate_of is None
        if keep:
            index.add(record_id, signature)
            accepted.append(record)
        ledger.append(
            {
                "accepted": keep,
                "duplicate_of": duplicate_of,
                "record_id": record_id,
                "signature_sha256": index.fingerprint(signature),
            }
        )
    return (
        accepted,
        ledger,
        {
            "accepted": len(accepted),
            "algorithm": "minhash_lsh_with_exact_jaccard_confirmation_v0.1",
            "comparison_unit": "role_tagged_full_conversation",
            "rejected": len(records) - len(accepted),
            "threshold": float(near_duplicate_config["threshold"]),
        },
    )


def _nearest_rank(values: Sequence[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def measure_and_select(
    records: Sequence[Mapping[str, Any]],
    *,
    maximum_train_supervised_tokens: int,
    metadata: Mapping[str, Mapping[str, Any]],
    target_task_train_supervised_tokens: Mapping[str, int],
    target_train_supervised_tokens: int,
    template: ChatTemplate,
    tokenizer: Any,
    validation_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    measured: list[tuple[dict[str, Any], int, int]] = []
    overlength = 0
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
        measured.append((record, len(encoded.input_ids), encoded.supervised_tokens))

    def priority(item: tuple[dict[str, Any], int, int]) -> tuple[Any, ...]:
        row = item[0]
        meta = metadata[str(row["record_id"])]
        return (
            int(meta["maximum_assistant_rank"]),
            -float(row["quality"]["score"]),
            not bool(meta["terminal"]),
            hashlib.sha256(str(row["record_id"]).encode()).hexdigest(),
        )

    train_by_task: defaultdict[str, list[tuple[dict[str, Any], int, int]]] = defaultdict(list)
    validation: list[tuple[dict[str, Any], int, int]] = []
    for item in measured:
        if item[0]["task_category"] not in target_task_train_supervised_tokens:
            continue
        if item[0]["split"] == "train":
            train_by_task[str(item[0]["task_category"])].append(item)
        else:
            validation.append(item)
    for rows in train_by_task.values():
        rows.sort(key=priority)
    validation.sort(key=priority)

    selected_train: list[tuple[dict[str, Any], int, int]] = []
    selected_ids: set[str] = set()
    train_tokens = 0
    task_tokens: Counter[str] = Counter()
    for task in sorted(target_task_train_supervised_tokens):
        target = int(target_task_train_supervised_tokens[task])
        for item in train_by_task[task]:
            if task_tokens[task] >= target:
                break
            if train_tokens + item[2] > maximum_train_supervised_tokens:
                continue
            selected_train.append(item)
            selected_ids.add(str(item[0]["record_id"]))
            train_tokens += item[2]
            task_tokens[task] += item[2]

    remainder = sorted(
        (
            item
            for rows in train_by_task.values()
            for item in rows
            if str(item[0]["record_id"]) not in selected_ids
        ),
        key=priority,
    )
    for item in remainder:
        if train_tokens >= target_train_supervised_tokens:
            break
        if train_tokens + item[2] > maximum_train_supervised_tokens:
            continue
        selected_train.append(item)
        selected_ids.add(str(item[0]["record_id"]))
        train_tokens += item[2]
        task_tokens[str(item[0]["task_category"])] += item[2]

    validation_target = math.ceil(train_tokens * validation_fraction / (1.0 - validation_fraction))
    selected_validation: list[tuple[dict[str, Any], int, int]] = []
    validation_tokens = 0
    for item in validation:
        if validation_tokens >= validation_target:
            break
        selected_validation.append(item)
        validation_tokens += item[2]

    selected = selected_train + selected_validation
    selected.sort(key=lambda item: (str(item[0]["split"]), str(item[0]["record_id"])))
    token_ledger = [
        {
            "record_id": item[0]["record_id"],
            "serialized_tokens": item[1],
            "split": item[0]["split"],
            "supervised_tokens": item[2],
            "task_category": item[0]["task_category"],
        }
        for item in selected
    ]
    lengths = [item[1] for item in selected]
    by_split: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_task: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for record, serialized, supervised in selected:
        for bucket in (by_split[str(record["split"])], by_task[str(record["task_category"])]):
            bucket["records"] += 1
            bucket["serialized_tokens"] += serialized
            bucket["supervised_tokens"] += supervised
            bucket["masked_tokens"] += serialized - supervised
    task_shortfalls = {
        task: max(0, int(target) - int(task_tokens[task]))
        for task, target in sorted(target_task_train_supervised_tokens.items())
    }
    metrics = {
        "accepted_records": len(selected),
        "assistant_only_loss_verified": True,
        "by_split": {key: dict(value) for key, value in sorted(by_split.items())},
        "by_task": {key: dict(value) for key, value in sorted(by_task.items())},
        "candidate_4k_fit_records": len(measured),
        "context_window": template.context_window,
        "overlength_rejected": overlength,
        "planning_target_reached": train_tokens >= target_train_supervised_tokens,
        "serialized_length": {
            "maximum": max(lengths, default=0),
            "mean": mean(lengths) if lengths else 0,
            "median": median(lengths) if lengths else 0,
            "minimum": min(lengths, default=0),
            "p95": _nearest_rank(lengths, 0.95),
        },
        "serialized_tokens": sum(item[1] for item in selected),
        "supervised_tokens": sum(item[2] for item in selected),
        "task_target_shortfalls": task_shortfalls,
        "token_ledger_sha256": sha256_value(token_ledger),
        "train_supervised_tokens": train_tokens,
        "validation_supervised_tokens": validation_tokens,
        "validation_target_supervised_tokens": validation_target,
    }
    return [item[0] for item in selected], metrics, token_ledger


def deterministic_owner_sample(
    records: Sequence[Mapping[str, Any]], *, maximum_records: int
) -> list[dict[str, Any]]:
    buckets: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[(str(record["split"]), str(record["task_category"]))].append(record)
    for rows in buckets.values():
        rows.sort(key=lambda row: hashlib.sha256(str(row["record_id"]).encode()).hexdigest())
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < min(maximum_records, len(records)):
        added = False
        for key in sorted(buckets):
            rows = buckets[key]
            if offset < len(rows):
                row = rows[offset]
                selected.append(
                    {
                        "language_mode": "english",
                        "messages": row["messages"],
                        "record_id": row["record_id"],
                        "skill_id": row["task_category"],
                        "split": row["split"],
                    }
                )
                added = True
                if len(selected) == min(maximum_records, len(records)):
                    break
        if not added:
            break
        offset += 1
    return selected


def build_oasst2_owner_review_html(
    sample_rows: list[Mapping[str, Any]],
) -> tuple[str, str]:
    sample_sha256 = sha256_value(sample_rows)
    html = build_owner_review_html(sample_rows, sample_sha256=sample_sha256)
    replacements = {
        "Lumi owner review": "Lumi OASST2 owner review",
        "Lumi owner spot-check": "Lumi OASST2 conversation spot-check",
        (
            "These rows are AI-generated and owner-reviewed. Approving them does not "
            "make them human-written."
        ): (
            "These are upstream crowdsourced OpenAssistant conversations selected from "
            "reviewed ranked trees. Check correctness, naturalness, context, safety, and "
            "formatting. Approval does not authorize registry promotion, dataset sealing, "
            "or SFT."
        ),
        "lumi.ai-assisted-owner-decisions.v0.3": "lumi.oasst2-owner-decisions.v0.1",
        "lumi-owner-review-decisions-v0.3.json": "lumi-oasst2-owner-decisions-v0.1.json",
    }
    for old, new in replacements.items():
        if old not in html:
            raise ValueError(f"OASST2 owner-review marker missing: {old[:48]}")
        html = html.replace(old, new)
    return html, sample_sha256


def load_runtime_components(
    plan: Mapping[str, Any],
    *,
    repository_root: Path,
) -> tuple[SFTPolicy, BenchmarkRegistry, Mapping[str, Any], ChatTemplate, Any]:
    from transformers import AutoTokenizer

    bindings = plan["bindings"]
    contract_path = resolve_binding(
        repository_root, bindings["data_contract"], name="data_contract"
    )
    quality_path = resolve_binding(
        repository_root, bindings["quality_config"], name="quality_config"
    )
    benchmark_path = resolve_binding(
        repository_root, bindings["benchmark_registry"], name="benchmark_registry"
    )
    template_path = resolve_binding(
        repository_root, bindings["chat_template"], name="chat_template"
    )
    tokenizer_json = resolve_binding(
        repository_root, bindings["tokenizer_json"], name="tokenizer_json"
    )
    tokenizer_config = resolve_binding(
        repository_root, bindings["tokenizer_config"], name="tokenizer_config"
    )
    if tokenizer_json.parent != tokenizer_config.parent:
        raise ValueError("frozen Lumi tokenizer files are not colocated")
    quality = _load_json(quality_path)
    validate_quality_config(quality)
    if Path(str(quality["benchmark_decontamination"]["registry"])) != Path(
        EXPECTED_BINDING_PATHS["benchmark_registry"]
    ):
        raise ValueError("active OASST2 benchmark registry path changed")
    policy = replace(
        load_sft_policy(contract_path),
        benchmark_decontamination=dict(quality["benchmark_decontamination"]),
    )
    benchmark = BenchmarkRegistry(policy.benchmark_decontamination, repository_root)
    if benchmark.path.resolve() != benchmark_path:
        raise ValueError("OASST2 benchmark registry resolved elsewhere")
    near = dict(plan["policy"]["near_duplicate"])
    template = load_chat_template(template_path)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_json.parent, local_files_only=True)
    return policy, benchmark, near, template, tokenizer


def artifact_inventory(
    paths: Mapping[str, Path], *, raw_text_artifacts: Iterable[str]
) -> dict[str, Any]:
    raw = set(raw_text_artifacts)
    objects = [
        {
            "bytes": path.stat().st_size,
            "contains_raw_training_text": name in raw,
            "name": name,
            "path": path.name,
            "sha256": file_sha256(path),
        }
        for name, path in sorted(paths.items())
    ]
    return {"objects": objects, "total_bytes": sum(item["bytes"] for item in objects)}


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite OASST2 JSON: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite OASST2 JSONL: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite OASST2 text: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_raw_text_free_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported OASST2 report")
    if report.get("raw_training_text_present") is not False:
        raise ValueError("OASST2 report must remain raw-text-free")
    claimed = report.get("report_sha256")
    if not isinstance(claimed, str) or _SHA256.fullmatch(claimed) is None:
        raise ValueError("OASST2 report SHA-256 is invalid")
    unsealed = {key: value for key, value in report.items() if key != "report_sha256"}
    if sha256_value(unsealed) != claimed:
        raise ValueError("OASST2 report internal SHA-256 failed")


__all__ = [
    "ATTRIBUTION_SCHEMA_VERSION",
    "EXPECTED_LIMITS",
    "EXPECTED_OBJECTS",
    "REPORT_SCHEMA_VERSION",
    "SOURCE_ID",
    "SOURCE_REVISION",
    "TREE_OBJECT",
    "VERIFICATION_SCHEMA_VERSION",
    "apply_conversation_near_dedup",
    "artifact_inventory",
    "build_oasst2_owner_review_html",
    "deterministic_owner_sample",
    "file_sha256",
    "load_oasst2_plan",
    "load_runtime_components",
    "load_tree_candidates",
    "measure_and_select",
    "sha256_value",
    "task_category",
    "tree_split",
    "validate_raw_text_free_report",
    "verify_raw_objects",
    "write_json_atomic",
    "write_jsonl_atomic",
    "write_text_atomic",
]
