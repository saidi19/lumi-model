"""Adaptive deficit continuation for Lumi's bounded two-pass map phase."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from decimal import ROUND_CEILING, Decimal
from typing import Any
from urllib.parse import urlparse

from .emr_production_dedup import dedup_job_token
from .emr_production_map import (
    map_checkpoint_fingerprint,
    resume_cursor_for_category,
    validate_map_checkpoint,
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def dedup_result_fingerprint(result: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(result)).hexdigest()


def _non_negative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def validate_adaptive_dedup_result(
    result: Mapping[str, Any],
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    map_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a dedup shortfall without accepting it as a final corpus."""

    validate_map_checkpoint(map_checkpoint, plan, authorization)
    if map_checkpoint.get("status") != "ready_for_global_dedup":
        raise ValueError("adaptive dedup requires a complete map checkpoint")
    if result.get("schema_version") != 1:
        raise ValueError("unsupported adaptive dedup-result schema")
    if result.get("gate") != "lumi_emr_production_global_dedup_result_v0.1":
        raise ValueError("unexpected adaptive dedup-result gate")
    if result.get("status") != "needs_more_mapping":
        raise ValueError("adaptive dedup result does not request more mapping")
    if result.get("plan_fingerprint") != plan["plan_fingerprint"]:
        raise ValueError("adaptive dedup result targets another plan")
    if result.get("authorization_run_token") != authorization["run_token"]:
        raise ValueError("adaptive dedup authorization changed")
    if result.get("map_checkpoint_sha256") != map_checkpoint["checkpoint_sha256"]:
        raise ValueError("adaptive dedup targets another map checkpoint")
    if result.get("job_token") != dedup_job_token(authorization, map_checkpoint):
        raise ValueError("adaptive dedup job token changed")
    expected = {str(row["category"]): row for row in plan["categories"]}
    categories = result.get("categories")
    if not isinstance(categories, list) or len(categories) != len(expected):
        raise ValueError("adaptive dedup category set changed")
    seen: set[str] = set()
    total_deficit = 0
    selected_tokens = 0
    selected_documents = 0
    for row in categories:
        if not isinstance(row, Mapping) or row.get("category") not in expected:
            raise ValueError("adaptive dedup category is invalid")
        name = str(row["category"])
        if name in seen:
            raise ValueError("adaptive dedup repeats a category")
        seen.add(name)
        selected = _non_negative_int(row.get("selected_tokens"), "selected tokens")
        documents = _non_negative_int(
            row.get("selected_documents"), "selected documents"
        )
        final = _non_negative_int(row.get("final_tokens"), "final category tokens")
        final_documents = _non_negative_int(
            row.get("final_documents"), "final category documents"
        )
        floor = int(expected[name]["floor_tokens"])
        if row.get("floor_tokens") != floor:
            raise ValueError("adaptive dedup category floor changed")
        if final != int(expected[name]["accepted_tokens"]) + selected:
            raise ValueError("adaptive dedup category token conservation failed")
        if final_documents != int(expected[name]["accepted_documents"]) + documents:
            raise ValueError("adaptive dedup category document conservation failed")
        deficit = max(0, floor - final)
        if row.get("deficit_tokens") != deficit:
            raise ValueError("adaptive dedup category deficit changed")
        total_deficit += deficit
        selected_tokens += selected
        selected_documents += documents
    if seen != set(expected):
        raise ValueError("adaptive dedup category set changed")
    if total_deficit == 0 or result.get("total_deficit_tokens") != total_deficit:
        raise ValueError("adaptive dedup total deficit changed")
    accepted = result.get("accepted_candidates")
    if not isinstance(accepted, Mapping):
        raise ValueError("adaptive dedup accepted output is missing")
    if accepted.get("tokens") != selected_tokens or accepted.get(
        "documents"
    ) != selected_documents:
        raise ValueError("adaptive dedup accepted totals changed")
    parsed = urlparse(str(accepted.get("s3_uri")))
    if (
        parsed.scheme != "s3"
        or parsed.netloc != plan["storage"]["bucket"]
        or not parsed.path.startswith(f"/{plan['storage']['namespace']}/")
    ):
        raise ValueError("adaptive dedup output left the production namespace")
    return dict(result)


def reopen_map_checkpoint(
    checkpoint: Mapping[str, Any],
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    dedup_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Increase only deficient category targets and resume at exact source cursors."""

    current = validate_map_checkpoint(checkpoint, plan, authorization)
    result_sha256 = dedup_result_fingerprint(dedup_result)
    if any(
        row.get("dedup_result_sha256") == result_sha256
        for row in current["adaptive_rounds"]
    ):
        return current
    validate_adaptive_dedup_result(
        dedup_result, plan, authorization, current
    )
    multiplier = Decimal(str(plan["pipeline"]["mapped_category_multiplier"]))
    dedup_categories = {
        str(row["category"]): row for row in dedup_result["categories"]
    }
    categories = [dict(row) for row in current["categories"]]
    round_categories = []
    first_cursor = None
    total_deficit = 0
    total_increase = 0
    for plan_category in plan["categories"]:
        name = str(plan_category["category"])
        deficit = int(dedup_categories[name]["deficit_tokens"])
        if deficit == 0:
            continue
        increase = int(
            (Decimal(deficit) * multiplier).to_integral_value(rounding=ROUND_CEILING)
        )
        row = next(value for value in categories if value["category"] == name)
        prior = int(row["mapped_target_tokens"])
        row["mapped_target_tokens"] = prior + increase
        round_categories.append(
            {
                "category": name,
                "deficit_tokens": deficit,
                "new_target_tokens": prior + increase,
                "prior_target_tokens": prior,
                "target_increase_tokens": increase,
            }
        )
        cursor = resume_cursor_for_category(plan, list(current["outputs"]), name)
        if first_cursor is None:
            first_cursor = cursor
        total_deficit += deficit
        total_increase += increase
    if first_cursor is None:
        raise ValueError("adaptive dedup result has no deficient category")
    raw_cap = int(plan["limits"]["candidate_hard_maximum_tokens"]) - int(
        plan["resume"]["accepted_tokens"]
    )
    worst_case = (
        int(current["total_mapped_tokens"])
        + total_increase
        + len(round_categories)
        * int(plan["limits"]["maximum_category_overshoot_tokens"])
    )
    if worst_case > raw_cap:
        raise RuntimeError("adaptive map continuation would exceed the 150B raw-token cap")
    adaptive_round = {
        "categories": round_categories,
        "dedup_map_checkpoint_sha256": current["checkpoint_sha256"],
        "dedup_result_sha256": result_sha256,
        "prior_checkpoint_sha256": current["checkpoint_sha256"],
        "round_sequence": len(current["adaptive_rounds"]) + 1,
        "total_deficit_tokens": total_deficit,
        "total_target_increase_tokens": total_increase,
    }
    result = {
        **current,
        "adaptive_rounds": [*current["adaptive_rounds"], adaptive_round],
        "categories": categories,
        "checkpoint_sequence": int(current["checkpoint_sequence"]) + 1,
        "previous_checkpoint_sha256": current["checkpoint_sha256"],
        "source_cursor": first_cursor,
        "status": "mapping",
    }
    result.pop("checkpoint_sha256", None)
    result["checkpoint_sha256"] = map_checkpoint_fingerprint(result)
    return validate_map_checkpoint(result, plan, authorization)
