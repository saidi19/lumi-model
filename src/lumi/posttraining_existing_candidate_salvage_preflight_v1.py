"""Token-consistent preflight for locally reusable Lumi SFT candidates."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .posttraining_format import encode_sft_messages, load_chat_template
from .posttraining_gpt_oss_teacher_source_promotion_preflight_v16 import (
    PureByteLevelBPETokenizer,
)

PLAN_SCHEMA_VERSION = "lumi.sft-existing-candidate-salvage-preflight-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-existing-candidate-salvage-preflight-report.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-existing-candidate-salvage-preflight-verification.v0.1"
)

EXPECTED_AUTHORIZATION = {
    "candidate_materialization_authorized": False,
    "candidate_registry_change_authorized": False,
    "dataset_seal_authorized": False,
    "download_authorized": False,
    "external_api_authorized": False,
    "gpu_authorized": False,
    "inference_authorized": False,
    "network_authorized": False,
    "predecessor_mutation_authorized": False,
    "production_promotion_authorized": False,
    "remote_access_authorized": False,
    "sft_training_authorized": False,
}

CURRENT_TRAIN_TOKENS = 7_341_582
CURRENT_TASKS = {
    "coding": 2_153_012,
    "conversation": 500_617,
    "factual_qa": 1_404_106,
    "multilingual_bengali": 537_242,
    "reasoning_math_science": 2_269_713,
    "safety_helpfulness": 36_255,
    "tool_api": 29_335,
    "writing_summarization": 411_302,
}
CURRENT_LANGUAGES = {
    "banglish": 27_890,
    "bengali_script": 549_652,
    "english": 6_740_757,
    "natural_code_switching": 23_283,
}
CURRENT_FAMILIES = {
    "coherelabs_aya_family": 513_379,
    "deepmind_mathematics_family": 2_221_113,
    "dmath_family": 48_480,
    "lumi_first_party_family": 114_864,
    "nvidia_family": 2_100_077,
    "openassistant_family": 2_343_669,
}
TASK_FLOORS = {
    "factual_qa": 1_500_000,
    "multilingual_bengali": 1_000_000,
    "safety_helpfulness": 800_000,
    "tool_api": 800_000,
    "writing_summarization": 1_200_000,
}
LANGUAGE_FLOORS = {
    "banglish": 800_000,
    "bengali_script": 1_000_000,
    "english": 7_800_000,
    "natural_code_switching": 400_000,
}
AYA_MIT_FAMILY_SHA256 = (
    "120f0a5cb5386355e804603ebac9c71e2775e698d02beefacb2e9b781a850151"
)
NEMOTRON_TRAIN_TOKEN_CEILING = 770_665


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_value(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[tuple[dict[str, Any], bytes]]:
    rows: list[tuple[dict[str, Any], bytes]] = []
    with path.open("rb") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSONL object: {path}")
            canonical = (
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
            if line != canonical:
                raise ValueError(f"candidate line is not canonical: {path}")
            rows.append((value, line))
    return rows


def load_plan(path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    workspace = root.parent.resolve()
    plan = _load_json(Path(path).resolve())
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("salvage preflight plan schema changed")
    if plan.get("status") != "authorized_local_only_token_consistent_salvage_preflight":
        raise ValueError("salvage preflight is not authorized")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("salvage preflight authorization widened")
    if plan.get("policy") != {
        "include_existing_nemotron": True,
        "include_unresolved_share_alike": False,
        "new_model_inference": False,
        "preserve_auxiliary_only_credit": True,
    }:
        raise ValueError("salvage preflight policy changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, dict) or len(bindings) != 12:
        raise ValueError("salvage preflight bindings changed")
    for name, binding in bindings.items():
        if not isinstance(binding, dict):
            raise ValueError(f"invalid binding: {name}")
        resolved = (root / str(binding.get("path"))).resolve()
        if not resolved.is_relative_to(workspace):
            raise ValueError(f"binding escaped workspace: {name}")
        if not resolved.is_file():
            raise ValueError(f"bound file is missing: {name}")
        if file_sha256(resolved) != binding.get("sha256"):
            raise ValueError(f"binding hash mismatch: {name}")
    return plan


def _measure(
    rows: Sequence[tuple[dict[str, Any], bytes]],
    tokenizer: PureByteLevelBPETokenizer,
    template: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    measured: list[dict[str, Any]] = []
    by_split: dict[str, Counter[str]] = {}
    for row, line in rows:
        encoded = encode_sft_messages(row["messages"], tokenizer=tokenizer, template=template)
        split = str(row["split"])
        bucket = by_split.setdefault(split, Counter())
        bucket["records"] += 1
        bucket["serialized_tokens"] += len(encoded.input_ids)
        bucket["supervised_tokens"] += encoded.supervised_tokens
        measured.append(
            {
                "record": row,
                "line": line,
                "serialized_tokens": len(encoded.input_ids),
                "supervised_tokens": encoded.supervised_tokens,
            }
        )
    return measured, {
        "by_split": {key: dict(value) for key, value in sorted(by_split.items())},
        "records": len(rows),
    }


def _selected_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, Counter[str]] = {}
    output = bytearray()
    record_ids: list[str] = []
    for item in rows:
        row = item["record"]
        split = str(row["split"])
        bucket = by_split.setdefault(split, Counter())
        bucket["records"] += 1
        bucket["serialized_tokens"] += int(item["serialized_tokens"])
        bucket["supervised_tokens"] += int(item["supervised_tokens"])
        output.extend(item["line"])
        record_ids.append(str(row["record_id"]))
    return {
        "by_split": {key: dict(value) for key, value in sorted(by_split.items())},
        "candidate_bytes": len(output),
        "candidate_sha256": hashlib.sha256(output).hexdigest(),
        "record_ids_sha256": hashlib.sha256(
            json.dumps(record_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "records": len(rows),
    }


def _owner_ids(sample_path: Path) -> set[str]:
    value = json.loads(sample_path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) != 32:
        raise ValueError("owner sample shape changed")
    return {str(row["record_id"]) for row in value}


def _select_nemotron(
    measured: Sequence[dict[str, Any]], owner_ids: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train = [item for item in measured if item["record"]["split"] == "train"]
    validation = [item for item in measured if item["record"]["split"] == "validation"]
    reviewed = [
        item for item in train if str(item["record"]["record_id"]) in owner_ids
    ]
    reviewed.sort(key=lambda item: str(item["record"]["record_id"]))
    selected_ids = {str(item["record"]["record_id"]) for item in reviewed}
    selected_train = list(reviewed)
    train_tokens = sum(int(item["supervised_tokens"]) for item in selected_train)
    for item in sorted(train, key=lambda value: str(value["record"]["record_id"])):
        record_id = str(item["record"]["record_id"])
        if record_id in selected_ids:
            continue
        tokens = int(item["supervised_tokens"])
        if train_tokens + tokens > NEMOTRON_TRAIN_TOKEN_CEILING:
            continue
        selected_train.append(item)
        selected_ids.add(record_id)
        train_tokens += tokens
    selected_ids.update(str(item["record"]["record_id"]) for item in validation)
    selected = [
        item
        for item in measured
        if str(item["record"]["record_id"]) in selected_ids
    ]
    review_selected = sum(
        1 for item in selected if str(item["record"]["record_id"]) in owner_ids
    )
    return selected, {
        "algorithm": "owner_sample_preserving_record_id_order_token_fill_v0.1",
        "train_token_ceiling": NEMOTRON_TRAIN_TOKEN_CEILING,
        "train_tokens_unused_below_ceiling": NEMOTRON_TRAIN_TOKEN_CEILING - train_tokens,
        "prior_owner_review_rows_retained": review_selected,
    }


def _add(base: Mapping[str, int], additions: Mapping[str, int]) -> dict[str, int]:
    value = dict(base)
    for key, amount in additions.items():
        value[key] = value.get(key, 0) + amount
    return value


def _gaps(values: Mapping[str, int], floors: Mapping[str, int]) -> dict[str, int]:
    return {key: max(0, floor - values.get(key, 0)) for key, floor in floors.items()}


def build_report(
    plan: Mapping[str, Any], plan_file_sha256: str, repository_root: str | Path
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resolved = {
        name: (root / binding["path"]).resolve()
        for name, binding in plan["bindings"].items()
    }
    reconciliation = _load_json(resolved["reconciliation_report"])
    verification = _load_json(resolved["reconciliation_verification"])
    if reconciliation.get("status") != "verified_existing_work_reconciled_no_redownload_or_promotion":
        raise ValueError("authoritative reconciliation status changed")
    if verification.get("status") != "PASS_existing_work_reconciled_no_redownload_or_promotion":
        raise ValueError("authoritative reconciliation verification failed")

    tokenizer = PureByteLevelBPETokenizer(resolved["tokenizer"])
    template = load_chat_template(resolved["chat_template"])
    owner_ids = _owner_ids(resolved["owner_review_sample"])

    aya_rows = _load_jsonl(resolved["aya_rights_candidate"])
    aya_mit_rows = [
        item
        for item in aya_rows
        if item[0]["provenance"].get("source_dataset_name_sha256")
        == AYA_MIT_FAMILY_SHA256
    ]
    if len(aya_mit_rows) != 673:
        raise ValueError("Aya MIT family membership changed")
    aya_measured, aya_full_metrics = _measure(aya_mit_rows, tokenizer, template)
    aya_selected = _selected_metrics(aya_measured)
    aya_reviewed_ids = {
        str(item["record"]["record_id"])
        for item in aya_measured
        if str(item["record"]["record_id"]) in owner_ids
    }

    banglabridge_rows = _load_jsonl(resolved["banglabridge_candidate"])
    banglabridge_measured, banglabridge_full_metrics = _measure(
        banglabridge_rows, tokenizer, template
    )
    banglabridge_selected = _selected_metrics(banglabridge_measured)

    banglatlit_rows = _load_jsonl(resolved["banglatlit_candidate"])
    banglatlit_measured, banglatlit_full_metrics = _measure(
        banglatlit_rows, tokenizer, template
    )
    banglatlit_selected = _selected_metrics(banglatlit_measured)

    nemotron_rows = _load_jsonl(resolved["nemotron_candidate"])
    nemotron_measured, nemotron_full_metrics = _measure(
        nemotron_rows, tokenizer, template
    )
    nemotron_subset, nemotron_selection = _select_nemotron(
        nemotron_measured, owner_ids
    )
    nemotron_selected = _selected_metrics(nemotron_subset)

    candidate_metrics = {
        "aya_mit": aya_selected,
        "banglabridge": banglabridge_selected,
        "banglatlit_auxiliary": banglatlit_selected,
        "nemotron_tool_subset": nemotron_selected,
    }
    train_tokens = {
        name: metrics["by_split"]["train"]["supervised_tokens"]
        for name, metrics in candidate_metrics.items()
    }
    added_train_tokens = sum(train_tokens.values())

    task_additions = {
        "multilingual_bengali": train_tokens["aya_mit"]
        + train_tokens["banglabridge"],
        "tool_api": train_tokens["nemotron_tool_subset"],
    }
    accounting_task_additions = _add(
        task_additions, {"multilingual_bengali": train_tokens["banglatlit_auxiliary"]}
    )
    language_additions = Counter()
    language_additions["bengali_script"] += train_tokens["aya_mit"]
    for item in banglabridge_measured:
        if item["record"]["split"] != "train":
            continue
        mode = (
            "bengali_script"
            if item["record"]["language"] == "bn"
            else "natural_code_switching"
        )
        language_additions[mode] += int(item["supervised_tokens"])
    language_additions["english"] += train_tokens["nemotron_tool_subset"]
    accounting_language_additions = _add(
        language_additions, {"banglish": train_tokens["banglatlit_auxiliary"]}
    )

    task_behavior = _add(CURRENT_TASKS, task_additions)
    task_accounting = _add(CURRENT_TASKS, accounting_task_additions)
    language_behavior = _add(CURRENT_LANGUAGES, language_additions)
    language_accounting = _add(CURRENT_LANGUAGES, accounting_language_additions)
    task_gaps = _gaps(task_behavior, TASK_FLOORS)
    language_gaps = _gaps(language_behavior, LANGUAGE_FLOORS)

    families = dict(CURRENT_FAMILIES)
    families["coherelabs_aya_family"] += train_tokens["aya_mit"]
    families["banglabridge_family"] = train_tokens["banglabridge"]
    families["banglatlit_family"] = train_tokens["banglatlit_auxiliary"]
    families["nvidia_family"] += train_tokens["nemotron_tool_subset"]
    projected_total = CURRENT_TRAIN_TOKENS + added_train_tokens
    family_shares = {
        key: round(value / projected_total, 8)
        for key, value in sorted(families.items())
    }
    strict_gap = max(
        max(0, 10_000_000 - projected_total),
        sum(task_gaps.values()),
        sum(language_gaps.values()),
    )

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "verified_local_salvage_subsets_projected_materialization_not_run",
        "observed_at": plan["observed_at"],
        "plan_file_sha256": plan_file_sha256,
        "policy": plan["policy"],
        "bindings": {
            name: {"sha256": binding["sha256"]}
            for name, binding in sorted(plan["bindings"].items())
        },
        "input_row_reads": {
            "aya_rights_candidate": len(aya_rows),
            "aya_mit_selected": len(aya_mit_rows),
            "banglabridge_candidate": len(banglabridge_rows),
            "banglatlit_candidate": len(banglatlit_rows),
            "nemotron_candidate": len(nemotron_rows),
            "total_candidate_rows_read": len(aya_rows)
            + len(banglabridge_rows)
            + len(banglatlit_rows)
            + len(nemotron_rows),
        },
        "tokenizer_consistency": {
            "tokenizer_sha256": plan["bindings"]["tokenizer"]["sha256"],
            "chat_template_sha256": plan["bindings"]["chat_template"]["sha256"],
            "implementation": "dependency_free_pure_byte_level_bpe",
            "aya_mit_full_metrics": aya_full_metrics,
            "banglabridge_full_metrics": banglabridge_full_metrics,
            "banglatlit_full_metrics": banglatlit_full_metrics,
            "nemotron_full_metrics": nemotron_full_metrics,
        },
        "projected_materialization": {
            "aya_mit": {
                **aya_selected,
                "source_family": "coherelabs_aya_family",
                "license_id": "mit",
                "prior_owner_review_rows_retained": len(aya_reviewed_ids),
                "deterministic_owner_review_rows_required": max(
                    0, 8 - len(aya_reviewed_ids)
                ),
            },
            "banglabridge_reuse_byte_exact": {
                **banglabridge_selected,
                "source_family": "banglabridge_family",
                "license_id": "cc-by-4.0",
            },
            "banglatlit_reuse_byte_exact_auxiliary_only": {
                **banglatlit_selected,
                "source_family": "banglatlit_family",
                "license_id": "mit",
                "behavior_floor_credit": 0,
            },
            "nemotron_tool_subset": {
                **nemotron_selected,
                **nemotron_selection,
                "source_family": "nvidia_family",
                "license_id": "cc-by-4.0",
                "new_inference_calls": 0,
            },
        },
        "updated_projection": {
            "current_train_supervised_tokens": CURRENT_TRAIN_TOKENS,
            "added_train_supervised_tokens": added_train_tokens,
            "projected_train_supervised_tokens": projected_total,
            "numeric_gap_to_10m": max(0, 10_000_000 - projected_total),
            "task_train_supervised_tokens_accounting": task_accounting,
            "task_train_supervised_tokens_behavior_safe": task_behavior,
            "task_floor_gaps_behavior_safe": task_gaps,
            "task_floor_gap_total_behavior_safe": sum(task_gaps.values()),
            "language_train_supervised_tokens_accounting": language_accounting,
            "language_train_supervised_tokens_behavior_safe": language_behavior,
            "language_floor_gaps_behavior_safe": language_gaps,
            "language_floor_gap_total_behavior_safe": sum(language_gaps.values()),
            "source_family_train_supervised_tokens": families,
            "source_family_shares": family_shares,
            "source_family_cap_violations": [
                key for key, share in family_shares.items() if share > 0.35
            ],
            "strict_targeted_gap_lower_bound": strict_gap,
        },
        "conflict_controls": {
            "aya_wikisplit_selected": False,
            "aya_wikilingua_selected": False,
            "retired_gpt_oss_candidate_selected": False,
            "new_download_performed": False,
            "new_inference_performed": False,
            "candidate_materialized": False,
            "registry_changed": False,
            "full_cross_source_hygiene_reaudit_complete": False,
        },
        "next_gate": {
            "action": "materialize_exact_aya_mit_and_nemotron_subsets_then_owner_review_delta",
            "reuse_banglabridge_and_banglatlit_byte_exact": True,
            "registry_change_authorized": False,
            "full_cross_source_reaudit_required_before_registry_successor": True,
            "dataset_seal_or_sft_authorized": False,
        },
        "raw_training_text_retained": False,
        "production_eligible": False,
    }
    report["report_sha256"] = sha256_value(report)
    return report


def build_verification(
    report: Mapping[str, Any], report_file_sha256: str
) -> dict[str, Any]:
    unsigned = dict(report)
    internal = unsigned.pop("report_sha256")
    if sha256_value(unsigned) != internal:
        raise ValueError("salvage preflight internal hash does not verify")
    projection = report["updated_projection"]
    if projection["source_family_cap_violations"]:
        raise ValueError("salvage preflight introduced a source-family cap violation")
    if report["projected_materialization"]["nemotron_tool_subset"][
        "prior_owner_review_rows_retained"
    ] != 8:
        raise ValueError("Nemotron owner review carry-forward changed")
    verification: dict[str, Any] = {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "PASS_local_salvage_subsets_projected_materialization_not_run",
        "report_file_sha256": report_file_sha256,
        "report_internal_sha256": internal,
        "checks": {
            "authoritative_no_redownload_reconciliation_bound": True,
            "pure_byte_level_bpe_and_chat_template_replayed": True,
            "aya_mit_family_selected_and_share_alike_excluded": True,
            "banglabridge_and_banglatlit_reused_byte_exact": True,
            "nemotron_subset_preserves_owner_sample_and_train_token_ceiling": True,
            "source_family_cap_has_no_projection_violation": True,
            "no_candidate_registry_seal_or_predecessor_mutation": True,
            "no_download_network_gpu_inference_or_sft": True,
            "report_is_raw_text_free": True,
        },
    }
    verification["verification_sha256"] = sha256_value(verification)
    return verification

