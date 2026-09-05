"""Fail-closed local preflight for an isolated Aya Bengali validation split."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .data_quality import MinHashNearDuplicateIndex, normalize_for_matching
from .posttraining_first_party import canonical_json_bytes, file_sha256
from .posttraining_format import ChatTemplate, ChatTokenizer
from .posttraining_oasst1_full_run import measure_token_fit

PLAN_SCHEMA_VERSION = "lumi.sft-aya-bengali-validation-preflight-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-aya-bengali-validation-preflight-report.v0.1"

EXPECTED_AUTHORIZATION = {
    "candidate_split_materialization_authorized": False,
    "dataset_seal_authorized": False,
    "external_api_spend_authorized": False,
    "network_access_authorized": False,
    "paid_gpu_authorized": False,
    "production_sft_authorized": False,
    "split_preflight_authorized": True,
    "purpose": "local_raw_text_free_aya_validation_split_projection_only",
}
EXPECTED_LIMITS = {
    "candidate_records": 1478,
    "candidate_supervised_tokens": 577453,
    "maximum_runtime_minutes": 60,
    "train_records": 1330,
    "validation_records": 148,
}
EXPECTED_POLICY = {
    "annotator_concentration_maximum_per_split": 0.5,
    "cluster_fields": ["user_prompt", "assistant_response"],
    "component_assignment": "sha256_ranked_subset_sum_v0.1",
    "dominant_annotator_validation_records": 74,
    "exact_cross_split_overlap_maximum": 0,
    "minhash": {
        "num_perm": 128,
        "package_version": "2.0.0",
        "scheme": "affine32",
        "seed": 1,
        "shingle_tokens": 5,
        "threshold": 0.9,
    },
    "near_duplicate_cross_split_records_maximum": 0,
    "selection_domain": "lumi-aya-validation-cluster-v0.1",
}
EXPECTED_OUTPUT = {
    "candidate_records_rewritten": False,
    "private_split_manifest_materialized": False,
    "production_dataset_promotions": 0,
    "raw_training_text_in_report": False,
    "report_path": "evidence/lumi-4b-sft-aya-bengali-validation-preflight-v0.1.json",
}
EXPECTED_PROJECTION = {
    "annotators": {
        "train_maximum_fraction": 0.5,
        "train_maximum_records": 665,
        "train_unique": 16,
        "validation_maximum_fraction": 0.5,
        "validation_maximum_records": 74,
        "validation_unique": 6,
    },
    "clusters": {
        "components": 1340,
        "largest_component_records": 6,
        "multi_record_components": 121,
        "validation_components": 135,
    },
    "leakage": {
        "exact_prompt_overlap": 0,
        "exact_response_overlap": 0,
        "prompt_near_duplicate_validation_records": 0,
        "response_near_duplicate_validation_records": 0,
    },
    "split_assignment_sha256": ("641dd62eac559c7b551986a45d352e7b43a2ecd8f6aab16d9aa1720db50f3113"),
    "train": {
        "record_ids_sha256": ("dcc6f7329c6e3eb08744326d52422ab1e8c408e5dfa1340680b03f07b5ad3bca"),
        "records": 1330,
        "supervised_tokens": 520269,
    },
    "validation": {
        "record_ids_sha256": ("6f2b88a5748660c8fcaa827ecdb0b97e5b89888993adf6325b9f1f6a06ae3a16"),
        "records": 148,
        "supervised_tokens": 57184,
    },
}
EXPECTED_BINDINGS = frozenset(
    {
        "attribution_manifest",
        "benchmark_registry",
        "candidate_records",
        "chat_template",
        "data_contract",
        "data_quality_implementation",
        "materialization_plan",
        "materialization_report",
        "owner_review_receipt",
        "preflight_implementation",
        "preflight_runner",
        "runtime_implementation",
        "source_promotion_receipt",
        "source_registry",
        "tokenizer_config",
        "tokenizer_json",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256")
    return value


def _verify_internal_seal(value: Mapping[str, Any], field: str) -> str:
    seal = _require_sha256(value.get(field), field)
    payload = {key: item for key, item in value.items() if key != field}
    if sha256_value(payload) != seal:
        raise ValueError(f"{field} changed")
    return seal


def resolve_validation_binding(
    repository_root: str | Path,
    binding: Mapping[str, Any],
) -> Path:
    root = Path(repository_root).resolve()
    workspace = root.parent.resolve()
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("Aya validation binding path is invalid")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError("Aya validation binding escapes workspace")
    return resolved


def load_aya_validation_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Load and hash-check the exact non-materializing validation preflight."""

    plan = _load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported Aya validation preflight plan")
    if plan.get("status") != "authorized_local_split_preflight_no_materialization":
        raise ValueError("Aya validation preflight status changed")
    if plan.get("observed_at") != "2026-08-06":
        raise ValueError("Aya validation preflight date changed")
    for field, expected in (
        ("authorization", EXPECTED_AUTHORIZATION),
        ("limits", EXPECTED_LIMITS),
        ("policy", EXPECTED_POLICY),
        ("expected_projection", EXPECTED_PROJECTION),
        ("output", EXPECTED_OUTPUT),
    ):
        if plan.get(field) != expected:
            raise ValueError(f"Aya validation preflight {field} changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("Aya validation preflight bindings changed")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"Aya validation binding is invalid: {name}")
        expected = _require_sha256(binding.get("sha256"), f"bindings.{name}.sha256")
        resolved = resolve_validation_binding(repository_root, binding)
        if not resolved.is_file() or file_sha256(resolved) != expected:
            raise ValueError(f"Aya validation binding changed: {name}")
    return plan


def verify_validation_upstream(
    *,
    materialization_report: Mapping[str, Any],
    owner_review_receipt: Mapping[str, Any],
    source_promotion_receipt: Mapping[str, Any],
) -> None:
    if _verify_internal_seal(materialization_report, "report_sha256") != (
        "a2f08b1594a2b6fd9def1f43a48e072cbe0f107f4bc7ea43b391d6066ddc6ade"
    ):
        raise ValueError("Aya materialization seal changed")
    if materialization_report.get("candidate") != {
        "materialized": True,
        "records": 1478,
        "supervised_tokens": 577453,
    }:
        raise ValueError("Aya materialization metrics changed")
    if _verify_internal_seal(owner_review_receipt, "receipt_sha256") != (
        "e23851af5718d10acdd120ca1e0c200672fb5c794ad503ed4d1f7f1650cc685f"
    ):
        raise ValueError("Aya owner-review seal changed")
    if (
        owner_review_receipt.get("accepted_records") != 32
        or owner_review_receipt.get("rejected_records") != 0
        or owner_review_receipt.get("all_sampled_rows_accepted") is not True
    ):
        raise ValueError("Aya owner review no longer passes")
    if _verify_internal_seal(source_promotion_receipt, "receipt_sha256") != (
        "66c03b268ab5544e7273d81e034be4792be78aa0956f2d73b1496f4950071446"
    ):
        raise ValueError("Aya source-promotion seal changed")
    if (
        source_promotion_receipt.get("candidate_source_registry_promotions") != 1
        or source_promotion_receipt.get("production_dataset_promotions") != 0
        or source_promotion_receipt.get("production_eligible") is not False
    ):
        raise ValueError("Aya source-promotion boundary changed")


def _new_near_duplicate_index(policy: Mapping[str, Any]) -> MinHashNearDuplicateIndex:
    return MinHashNearDuplicateIndex(policy["minhash"])


class _UnionFind:
    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def _message_text(row: Mapping[str, Any], index: int) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("Aya validation row message shape changed")
    if [message.get("role") for message in messages] != ["user", "assistant"]:
        raise ValueError("Aya validation row roles changed")
    content = messages[index].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Aya validation row content changed")
    return normalize_for_matching(content)


def build_near_duplicate_components(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
) -> list[list[str]]:
    """Connect any rows whose prompt or response has MinHash Jaccard >= 0.9."""

    record_ids = [str(row.get("record_id")) for row in rows]
    if any(not record_id or record_id == "None" for record_id in record_ids):
        raise ValueError("Aya validation record ID changed")
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Aya validation record IDs are not unique")
    union_find = _UnionFind(record_ids)
    for field_index in (0, -1):
        index = _new_near_duplicate_index(policy)
        for row in sorted(rows, key=lambda item: str(item["record_id"])):
            record_id = str(row["record_id"])
            signature = index.signature(_message_text(row, field_index))
            for candidate in sorted(index.index.query(signature)):
                if signature.jaccard(index.signatures[candidate]) >= float(
                    policy["minhash"]["threshold"]
                ):
                    union_find.union(record_id, candidate)
            index.add(record_id, signature)
    by_root: dict[str, list[str]] = {}
    for record_id in sorted(record_ids):
        by_root.setdefault(union_find.find(record_id), []).append(record_id)
    return list(by_root.values())


def _component_rank(component: Sequence[str], *, domain: str) -> str:
    joined = "\n".join(component)
    return hashlib.sha256(f"{domain}:{joined}".encode()).hexdigest()


def select_validation_ids(
    components: Sequence[Sequence[str]],
    *,
    attribution_by_id: Mapping[str, Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> tuple[set[str], int]:
    """Select whole clusters while preserving the exact 50% annotator cap."""

    annotators = Counter(
        str(attribution_by_id[record_id]["annotator_sha256"])
        for component in components
        for record_id in component
    )
    dominant_annotator = min(
        (annotator for annotator, count in annotators.items() if count == max(annotators.values()))
    )
    policy = plan["policy"]
    target_records = plan["limits"]["validation_records"]
    target_dominant = policy["dominant_annotator_validation_records"]
    ranked = sorted(
        (list(component) for component in components),
        key=lambda component: _component_rank(
            component,
            domain=policy["selection_domain"],
        ),
    )
    reachable: dict[tuple[int, int], list[list[str]]] = {(0, 0): []}
    for component in ranked:
        size = len(component)
        dominant_size = sum(
            attribution_by_id[record_id]["annotator_sha256"] == dominant_annotator
            for record_id in component
        )
        for total, dominant_total in sorted(list(reachable), reverse=True):
            key = (total + size, dominant_total + dominant_size)
            if key[0] <= target_records and key[1] <= target_dominant and key not in reachable:
                reachable[key] = [*reachable[(total, dominant_total)], component]
    selected = reachable.get((target_records, target_dominant))
    if selected is None:
        raise ValueError("Aya validation cluster assignment cannot satisfy exact bounds")
    return ({record_id for component in selected for record_id in component}, len(selected))


def _near_validation_records(
    train_values: set[str],
    validation_values: set[str],
    *,
    policy: Mapping[str, Any],
) -> int:
    index = _new_near_duplicate_index(policy)
    for offset, value in enumerate(sorted(train_values)):
        index.add(f"train-{offset:04d}", index.signature(value))
    return sum(
        index.duplicate_of(index.signature(value)) is not None for value in validation_values
    )


def _split_concentration(
    rows: Sequence[Mapping[str, Any]],
    *,
    attribution_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    counts = Counter(
        str(attribution_by_id[str(row["record_id"])]["annotator_sha256"]) for row in rows
    )
    maximum = max(counts.values())
    return {
        "maximum_fraction": maximum / len(rows),
        "maximum_records": maximum,
        "unique": len(counts),
    }


def build_split_projection(
    rows: Sequence[Mapping[str, Any]],
    attribution_rows: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    template: ChatTemplate,
    tokenizer: ChatTokenizer,
) -> dict[str, Any]:
    """Build and verify the exact in-memory split; do not write raw records."""

    limits = plan["limits"]
    if len(rows) != limits["candidate_records"] or len(attribution_rows) != len(rows):
        raise ValueError("Aya validation input count changed")
    if any(
        row.get("split") != "train"
        or row.get("language") != "bn"
        or row.get("task_category") != "multilingual_bengali"
        for row in rows
    ):
        raise ValueError("Aya validation candidate metadata changed")
    attribution_by_id = {str(row.get("record_id")): row for row in attribution_rows}
    record_ids = {str(row.get("record_id")) for row in rows}
    if set(attribution_by_id) != record_ids or len(attribution_by_id) != len(rows):
        raise ValueError("Aya validation attribution membership changed")

    components = build_near_duplicate_components(rows, policy=plan["policy"])
    validation_ids, validation_component_count = select_validation_ids(
        components,
        attribution_by_id=attribution_by_id,
        plan=plan,
    )
    train = [
        {**copy.deepcopy(dict(row)), "split": "train"}
        for row in rows
        if row["record_id"] not in validation_ids
    ]
    validation = [
        {**copy.deepcopy(dict(row)), "split": "validation"}
        for row in rows
        if row["record_id"] in validation_ids
    ]
    train_fit, train_metrics = measure_token_fit(
        train,
        template=template,
        tokenizer=tokenizer,
    )
    validation_fit, validation_metrics = measure_token_fit(
        validation,
        template=template,
        tokenizer=tokenizer,
    )
    if len(train_fit) != len(train) or len(validation_fit) != len(validation):
        raise ValueError("Aya validation split lost 4K-fit records")

    train_prompts = {_message_text(row, 0) for row in train}
    validation_prompts = {_message_text(row, 0) for row in validation}
    train_responses = {_message_text(row, -1) for row in train}
    validation_responses = {_message_text(row, -1) for row in validation}
    leakage = {
        "exact_prompt_overlap": len(train_prompts & validation_prompts),
        "exact_response_overlap": len(train_responses & validation_responses),
        "prompt_near_duplicate_validation_records": _near_validation_records(
            train_prompts,
            validation_prompts,
            policy=plan["policy"],
        ),
        "response_near_duplicate_validation_records": _near_validation_records(
            train_responses,
            validation_responses,
            policy=plan["policy"],
        ),
    }
    train_concentration = _split_concentration(
        train,
        attribution_by_id=attribution_by_id,
    )
    validation_concentration = _split_concentration(
        validation,
        attribution_by_id=attribution_by_id,
    )
    projection = {
        "annotators": {
            "train_maximum_fraction": train_concentration["maximum_fraction"],
            "train_maximum_records": train_concentration["maximum_records"],
            "train_unique": train_concentration["unique"],
            "validation_maximum_fraction": validation_concentration["maximum_fraction"],
            "validation_maximum_records": validation_concentration["maximum_records"],
            "validation_unique": validation_concentration["unique"],
        },
        "clusters": {
            "components": len(components),
            "largest_component_records": max(map(len, components)),
            "multi_record_components": sum(len(component) > 1 for component in components),
            "validation_components": validation_component_count,
        },
        "leakage": leakage,
        "split_assignment_sha256": sha256_value(
            {
                str(row["record_id"]): row["split"]
                for row in sorted([*train, *validation], key=lambda item: item["record_id"])
            }
        ),
        "train": {
            "record_ids_sha256": sha256_value(sorted(str(row["record_id"]) for row in train)),
            "records": len(train),
            "supervised_tokens": train_metrics["supervised_tokens"],
        },
        "validation": {
            "record_ids_sha256": sha256_value(sorted(validation_ids)),
            "records": len(validation),
            "supervised_tokens": validation_metrics["supervised_tokens"],
        },
    }
    if projection != plan["expected_projection"]:
        raise ValueError("Aya validation split projection changed")
    return projection


def build_validation_preflight_report(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    _require_sha256(plan_file_sha256, "plan_file_sha256")
    if projection != EXPECTED_PROJECTION:
        raise ValueError("Aya validation report projection changed")
    report: dict[str, Any] = {
        "authorization": EXPECTED_AUTHORIZATION,
        "bindings": {name: binding["sha256"] for name, binding in sorted(plan["bindings"].items())},
        "candidate_split_materialized": False,
        "dataset_seal_authorized": False,
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "production_dataset_promotions": 0,
        "production_eligible": False,
        "production_sft_authorized": False,
        "projection": copy.deepcopy(dict(projection)),
        "raw_training_text_retained": False,
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "local_split_preflight_passed_materialization_requires_authorization",
    }
    report["report_sha256"] = sha256_value(report)
    return report


def validate_validation_preflight_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("Aya validation preflight report schema changed")
    _verify_internal_seal(report, "report_sha256")
    if (
        report.get("status")
        != "local_split_preflight_passed_materialization_requires_authorization"
        or report.get("projection") != EXPECTED_PROJECTION
        or report.get("candidate_split_materialized") is not False
        or report.get("production_dataset_promotions") != 0
        or report.get("production_eligible") is not False
        or report.get("production_sft_authorized") is not False
        or report.get("dataset_seal_authorized") is not False
        or report.get("raw_training_text_retained") is not False
    ):
        raise ValueError("Aya validation preflight report boundary changed")
