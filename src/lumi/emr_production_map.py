"""Hash-bound first-pass map checkpoint for Lumi's two-pass EMR corpus run."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from decimal import ROUND_CEILING, Decimal
from typing import Any

from .emr_production_staging import validate_published_staged_source
from .emr_serverless_production import (
    _bounded_s3_uri,
    _resource_shape,
    validate_paid_production_authorization,
    validate_production_checkpoint,
)

SCHEMA_VERSION = 1


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _non_negative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def map_checkpoint_fingerprint(checkpoint: Mapping[str, Any]) -> str:
    material = dict(checkpoint)
    material.pop("checkpoint_sha256", None)
    return _sha256_bytes(_canonical_json_bytes(material))


def map_job_token(
    authorization: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> str:
    run_token = authorization.get("run_token")
    checkpoint_sha256 = checkpoint.get("checkpoint_sha256")
    if not isinstance(run_token, str) or len(run_token) != 32:
        raise ValueError("production map authorization token is invalid")
    _sha256(checkpoint_sha256, "production map source checkpoint")
    return _sha256_bytes(f"map:{run_token}:{checkpoint_sha256}".encode())[:32]


def map_reservation_token(
    authorization: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    category: str,
    source_ordinal: int,
    start_after_line: int,
    mapped_tokens: int,
    mapped_target_tokens: int,
) -> str:
    """Derive one category-lane work token independent of other active lanes."""

    run_token = authorization.get("run_token")
    if not isinstance(run_token, str) or len(run_token) != 32:
        raise ValueError("production map authorization token is invalid")
    material = {
        "category": category,
        "mapped_target_tokens": mapped_target_tokens,
        "mapped_tokens": mapped_tokens,
        "plan_fingerprint": plan["plan_fingerprint"],
        "run_token": run_token,
        "source_ordinal": source_ordinal,
        "start_after_line": start_after_line,
    }
    return _sha256_bytes(b"reservation:" + _canonical_json_bytes(material))[:32]


def _category_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    multiplier = Decimal(str(plan["pipeline"]["mapped_category_multiplier"]))
    rows = []
    for category in plan["categories"]:
        remaining = max(0, int(category["floor_tokens"]) - int(category["accepted_tokens"]))
        target = int(
            (Decimal(remaining) * multiplier).to_integral_value(rounding=ROUND_CEILING)
        )
        rows.append(
            {
                "category": category["category"],
                "mapped_documents": 0,
                "mapped_target_tokens": target,
                "mapped_tokens": 0,
            }
        )
    return rows


def _adaptive_category_targets(
    plan: Mapping[str, Any], rounds: Any
) -> dict[str, int]:
    initial = {row["category"]: row["mapped_target_tokens"] for row in _category_rows(plan)}
    if not isinstance(rounds, list):
        raise ValueError("production map adaptive-round ledger is missing")
    targets = dict(initial)
    seen_results: set[str] = set()
    multiplier = Decimal(str(plan["pipeline"]["mapped_category_multiplier"]))
    for expected_sequence, round_row in enumerate(rounds, start=1):
        if not isinstance(round_row, Mapping):
            raise ValueError("production map adaptive round is invalid")
        if round_row.get("round_sequence") != expected_sequence:
            raise ValueError("production map adaptive-round sequence changed")
        prior = _sha256(
            round_row.get("prior_checkpoint_sha256"), "adaptive prior checkpoint"
        )
        if round_row.get("dedup_map_checkpoint_sha256") != prior:
            raise ValueError("adaptive dedup targets another map checkpoint")
        result_sha = _sha256(
            round_row.get("dedup_result_sha256"), "adaptive dedup result"
        )
        if result_sha in seen_results:
            raise ValueError("production map repeats an adaptive dedup result")
        seen_results.add(result_sha)
        categories = round_row.get("categories")
        if not isinstance(categories, list) or not categories:
            raise ValueError("production map adaptive categories are missing")
        seen_categories: set[str] = set()
        total_deficit = 0
        total_increase = 0
        for category in categories:
            if not isinstance(category, Mapping):
                raise ValueError("production map adaptive category is invalid")
            name = category.get("category")
            if name not in targets or name in seen_categories:
                raise ValueError("production map adaptive category changed")
            seen_categories.add(str(name))
            deficit = _non_negative_int(
                category.get("deficit_tokens"), "adaptive category deficit"
            )
            if deficit == 0:
                raise ValueError("production map adaptive category has no deficit")
            expected_increase = int(
                (Decimal(deficit) * multiplier).to_integral_value(
                    rounding=ROUND_CEILING
                )
            )
            increase = _non_negative_int(
                category.get("target_increase_tokens"), "adaptive target increase"
            )
            if increase != expected_increase:
                raise ValueError("production map adaptive multiplier changed")
            if category.get("prior_target_tokens") != targets[str(name)]:
                raise ValueError("production map adaptive prior target changed")
            new_target = targets[str(name)] + increase
            if category.get("new_target_tokens") != new_target:
                raise ValueError("production map adaptive new target changed")
            targets[str(name)] = new_target
            total_deficit += deficit
            total_increase += increase
        if round_row.get("total_deficit_tokens") != total_deficit:
            raise ValueError("production map adaptive deficit conservation failed")
        if round_row.get("total_target_increase_tokens") != total_increase:
            raise ValueError("production map adaptive target conservation failed")
    raw_cap = int(plan["limits"]["candidate_hard_maximum_tokens"]) - int(
        plan["resume"]["accepted_tokens"]
    )
    if sum(targets.values()) > raw_cap:
        raise ValueError("production map targets exceed the 150B raw-token cap")
    return targets


def initial_map_checkpoint(
    plan: Mapping[str, Any],
    accepted_checkpoint: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Start a raw-candidate map ledger at the independently verified prefix."""

    validate_production_checkpoint(accepted_checkpoint, plan)
    validate_paid_production_authorization(authorization, plan, accepted_checkpoint)
    checkpoint = {
        "accepted_root_checkpoint_sha256": accepted_checkpoint["checkpoint_sha256"],
        "active_reservations": [],
        "adaptive_rounds": [],
        "authorization_sha256": _sha256_bytes(_canonical_json_bytes(authorization)),
        "authorization_run_token": authorization["run_token"],
        "categories": _category_rows(plan),
        "checkpoint_sequence": 0,
        "gate": "lumi_emr_production_map_checkpoint_v0.1",
        "outputs": [],
        "output_sequence": 0,
        "plan_fingerprint": plan["plan_fingerprint"],
        "previous_checkpoint_sha256": None,
        "reservation_sequence": 0,
        "schema_version": SCHEMA_VERSION,
        "source_cursor": dict(accepted_checkpoint["source_cursor"]),
        "status": "mapping",
        "total_mapped_documents": 0,
        "total_mapped_tokens": 0,
    }
    checkpoint["checkpoint_sha256"] = map_checkpoint_fingerprint(checkpoint)
    return validate_map_checkpoint(checkpoint, plan, authorization)


def validate_map_checkpoint(
    checkpoint: Mapping[str, Any],
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported production map-checkpoint schema")
    if checkpoint.get("gate") != "lumi_emr_production_map_checkpoint_v0.1":
        raise ValueError("unexpected production map-checkpoint gate")
    if checkpoint.get("checkpoint_sha256") != map_checkpoint_fingerprint(checkpoint):
        raise ValueError("production map checkpoint SHA-256 changed")
    if checkpoint.get("plan_fingerprint") != plan["plan_fingerprint"]:
        raise ValueError("production map checkpoint targets another plan")
    if checkpoint.get("authorization_run_token") != authorization["run_token"]:
        raise ValueError("production map checkpoint authorization changed")
    if checkpoint.get("authorization_sha256") != _sha256_bytes(
        _canonical_json_bytes(authorization)
    ):
        raise ValueError("production map authorization SHA-256 changed")
    if checkpoint.get("accepted_root_checkpoint_sha256") != authorization.get(
        "checkpoint_sha256"
    ):
        raise ValueError("production map accepted root differs from its authorization")
    _sha256(checkpoint.get("accepted_root_checkpoint_sha256"), "accepted root checkpoint")
    sequence = _non_negative_int(checkpoint.get("checkpoint_sequence"), "map sequence")
    output_sequence = _non_negative_int(
        checkpoint.get("output_sequence"), "map output sequence"
    )
    reservation_sequence = _non_negative_int(
        checkpoint.get("reservation_sequence"), "map reservation sequence"
    )
    previous = checkpoint.get("previous_checkpoint_sha256")
    if sequence == 0 and previous is not None:
        raise ValueError("initial production map checkpoint cannot have a parent")
    if sequence > 0:
        _sha256(previous, "previous map checkpoint")
    if checkpoint.get("status") not in {"mapping", "ready_for_global_dedup", "paused"}:
        raise ValueError("production map-checkpoint status is invalid")
    adaptive_rounds = checkpoint.get("adaptive_rounds")
    targets = _adaptive_category_targets(plan, adaptive_rounds)
    if sequence != output_sequence + len(adaptive_rounds) + reservation_sequence:
        raise ValueError("production map checkpoint transition sequence changed")
    expected = {row["category"]: row for row in _category_rows(plan)}
    categories = checkpoint.get("categories")
    if not isinstance(categories, list) or len(categories) != len(expected):
        raise ValueError("production map categories changed")
    mapped_documents = 0
    mapped_tokens = 0
    for row in categories:
        if not isinstance(row, Mapping) or row.get("category") not in expected:
            raise ValueError("production map category is invalid")
        target = _non_negative_int(row.get("mapped_target_tokens"), "map category target")
        if target != targets[str(row["category"])]:
            raise ValueError("production map category target changed")
        tokens = _non_negative_int(row.get("mapped_tokens"), "mapped category tokens")
        documents = _non_negative_int(
            row.get("mapped_documents"), "mapped category documents"
        )
        if tokens > target + int(plan["limits"]["maximum_category_overshoot_tokens"]):
            raise ValueError("production map category exceeds its bounded overshoot")
        mapped_tokens += tokens
        mapped_documents += documents
    if mapped_tokens != checkpoint.get("total_mapped_tokens"):
        raise ValueError("production map token conservation failed")
    if mapped_documents != checkpoint.get("total_mapped_documents"):
        raise ValueError("production map document conservation failed")
    raw_cap = int(plan["limits"]["candidate_hard_maximum_tokens"]) - int(
        plan["resume"]["accepted_tokens"]
    )
    if mapped_tokens > raw_cap:
        raise ValueError("production map checkpoint exceeds the 150B raw-token cap")
    outputs = checkpoint.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != output_sequence:
        raise ValueError("production map output ledger changed")
    seen: set[str] = set()
    seen_uris: set[str] = set()
    output_documents = 0
    output_tokens = 0
    for output in outputs:
        if not isinstance(output, Mapping):
            raise ValueError("production map output is invalid")
        digest = _sha256(output.get("manifest_sha256"), "map-output manifest")
        if digest in seen:
            raise ValueError("production map checkpoint repeats an output")
        seen.add(digest)
        documents = _non_negative_int(
            output.get("candidate_documents"), "map-output documents"
        )
        tokens = _non_negative_int(output.get("candidate_tokens"), "map-output tokens")
        if (tokens == 0) != (documents == 0):
            raise ValueError("production map output token/document emptiness differs")
        uri = output.get("parquet_s3_uri")
        if tokens:
            if (
                not isinstance(uri, str)
                or not uri.startswith("s3://")
                or uri in seen_uris
            ):
                raise ValueError(
                    "production map checkpoint repeats an invalid Parquet URI"
                )
            seen_uris.add(uri)
        elif uri is not None:
            raise ValueError("empty production map output must not publish Parquet")
        output_documents += documents
        output_tokens += tokens
    if output_documents != mapped_documents or output_tokens != mapped_tokens:
        raise ValueError("production map output ledger does not conserve mapped totals")
    reservations = checkpoint.get("active_reservations")
    if not isinstance(reservations, list):
        raise ValueError("production map active reservations are missing")
    if len(reservations) > int(plan["pipeline"]["maximum_concurrent_map_jobs"]):
        raise ValueError("production map exceeds its concurrent-job ceiling")
    category_values = {str(row["category"]): row for row in categories}
    reservation_categories: set[str] = set()
    reservation_tokens: set[str] = set()
    for reservation in reservations:
        if not isinstance(reservation, Mapping):
            raise ValueError("production map reservation is invalid")
        category_name = reservation.get("category")
        if category_name not in category_values or category_name in reservation_categories:
            raise ValueError("production map repeats an invalid reserved category")
        category_name = str(category_name)
        reservation_categories.add(category_name)
        category_row = category_values[category_name]
        if int(category_row["mapped_tokens"]) >= int(
            category_row["mapped_target_tokens"]
        ):
            raise ValueError("production map reserves a completed category")
        expected_cursor = resume_cursor_for_category(plan, outputs, category_name)
        if reservation.get("source_cursor") != expected_cursor:
            raise ValueError("production map reservation cursor changed")
        source_ordinal = int(expected_cursor["source_ordinal"])
        source = plan["source_schedule"][source_ordinal]
        if source["category"] != category_name:
            raise ValueError("production map reservation source category changed")
        if reservation.get("source_sha256") != source["source_sha256"]:
            raise ValueError("production map reservation source SHA-256 changed")
        remaining = int(category_row["mapped_target_tokens"]) - int(
            category_row["mapped_tokens"]
        )
        if reservation.get("remaining_target_tokens") != remaining:
            raise ValueError("production map reservation target changed")
        expected_token = map_reservation_token(
            authorization,
            plan,
            category=category_name,
            source_ordinal=source_ordinal,
            start_after_line=int(expected_cursor["next_line"]),
            mapped_tokens=int(category_row["mapped_tokens"]),
            mapped_target_tokens=int(category_row["mapped_target_tokens"]),
        )
        token = reservation.get("reservation_token")
        if token != expected_token or token in reservation_tokens:
            raise ValueError("production map reservation token changed")
        reservation_tokens.add(str(token))
    cursor = checkpoint.get("source_cursor")
    if not isinstance(cursor, Mapping):
        raise ValueError("production map source cursor is missing")
    ordinal = _non_negative_int(cursor.get("source_ordinal"), "map source ordinal")
    _non_negative_int(cursor.get("next_line"), "map source line")
    if ordinal > len(plan["source_schedule"]):
        raise ValueError("production map cursor exceeds the source inventory")
    complete = not reservations and all(
        int(row["mapped_tokens"]) >= int(row["mapped_target_tokens"])
        for row in categories
    )
    if complete != (checkpoint.get("status") == "ready_for_global_dedup"):
        raise ValueError("production map completion status differs from its category targets")
    return dict(checkpoint)


def reserve_map_categories(
    checkpoint: Mapping[str, Any],
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Reserve at most one source job per category for safe parallel mapping."""

    current = validate_map_checkpoint(checkpoint, plan, authorization)
    if current["status"] != "mapping":
        raise ValueError("production map checkpoint is not accepting reservations")
    active = [dict(row) for row in current["active_reservations"]]
    active_categories = {str(row["category"]) for row in active}
    categories = {str(row["category"]): row for row in current["categories"]}
    ceiling = int(plan["pipeline"]["maximum_concurrent_map_jobs"])
    for plan_category in plan["categories"]:
        if len(active) >= ceiling:
            break
        name = str(plan_category["category"])
        row = categories[name]
        if name in active_categories or int(row["mapped_tokens"]) >= int(
            row["mapped_target_tokens"]
        ):
            continue
        cursor = resume_cursor_for_category(plan, list(current["outputs"]), name)
        source = plan["source_schedule"][int(cursor["source_ordinal"])]
        token = map_reservation_token(
            authorization,
            plan,
            category=name,
            source_ordinal=int(cursor["source_ordinal"]),
            start_after_line=int(cursor["next_line"]),
            mapped_tokens=int(row["mapped_tokens"]),
            mapped_target_tokens=int(row["mapped_target_tokens"]),
        )
        active.append(
            {
                "category": name,
                "remaining_target_tokens": int(row["mapped_target_tokens"])
                - int(row["mapped_tokens"]),
                "reservation_token": token,
                "source_cursor": cursor,
                "source_sha256": source["source_sha256"],
            }
        )
        active_categories.add(name)
    if active == current["active_reservations"]:
        raise ValueError("production map has no category available for reservation")
    result = {
        **current,
        "active_reservations": active,
        "checkpoint_sequence": int(current["checkpoint_sequence"]) + 1,
        "previous_checkpoint_sha256": current["checkpoint_sha256"],
        "reservation_sequence": int(current["reservation_sequence"]) + 1,
        "source_cursor": dict(active[0]["source_cursor"]),
    }
    result.pop("checkpoint_sha256", None)
    result["checkpoint_sha256"] = map_checkpoint_fingerprint(result)
    return validate_map_checkpoint(result, plan, authorization)


def validate_reserved_staged_source(
    staged: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    *,
    reservation_token: str,
) -> dict[str, Any]:
    """Bind an internet-facing CPU stager result to one active map lane."""

    current = validate_map_checkpoint(checkpoint, plan, authorization)
    reservation = next(
        (
            row
            for row in current["active_reservations"]
            if row["reservation_token"] == reservation_token
        ),
        None,
    )
    if reservation is None:
        raise ValueError("staged production map reservation is not active")
    expected = {
        "authorization_run_token": authorization["run_token"],
        "gate": "lumi_emr_reserved_map_staged_source_v0.1",
        "plan_fingerprint": plan["plan_fingerprint"],
        "reservation_token": reservation_token,
        "schema_version": 1,
        "source_ordinal": reservation["source_cursor"]["source_ordinal"],
        "status": "staged",
    }
    for name, value in expected.items():
        if staged.get(name) != value:
            raise ValueError(f"reserved staged-source changed {name}")
    manifest = staged.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("reserved staged-source manifest is missing")
    prefix = (
        f"{plan['storage']['namespace']}/raw/transient/reservations/"
        f"{reservation_token}"
    )
    validate_published_staged_source(
        manifest,
        bucket=str(plan["storage"]["bucket"]),
        key_prefix=prefix,
        hard_transient_bytes=int(plan["limits"]["maximum_transient_source_bytes"]),
    )
    if manifest.get("source_sha256") != reservation["source_sha256"]:
        raise ValueError("reserved staged-source SHA-256 changed")
    if manifest.get("start_after_line") != reservation["source_cursor"]["next_line"]:
        raise ValueError("reserved staged-source resume line changed")
    return dict(staged)


def resume_cursor_for_category(
    plan: Mapping[str, Any], outputs: list[Mapping[str, Any]], category: str
) -> dict[str, int]:
    """Return the first not-yet-mapped source line for one category."""

    ordinals = [
        int(row["ordinal"])
        for row in plan["source_schedule"]
        if row["category"] == category
    ]
    if not ordinals:
        raise ValueError("production map category has no source inventory")
    category_outputs = [row for row in outputs if row.get("category") == category]
    if not category_outputs:
        if category == plan["source_schedule"][int(plan["resume"]["source_ordinal"])][
            "category"
        ]:
            return {
                "next_line": int(plan["resume"]["source_next_line"]),
                "source_ordinal": int(plan["resume"]["source_ordinal"]),
            }
        return {"next_line": 0, "source_ordinal": ordinals[0]}
    last = category_outputs[-1]
    ordinal = int(last["source_ordinal"])
    if ordinal not in ordinals:
        raise ValueError("production map output category/source identity changed")
    if bool(last["source_complete"]):
        next_ordinal = ordinal + 1
        if next_ordinal not in ordinals:
            raise RuntimeError("production map exhausted a category source inventory")
        return {"next_line": 0, "source_ordinal": next_ordinal}
    return {"next_line": int(last["end_line"]), "source_ordinal": ordinal}


def _next_incomplete_cursor(
    plan: Mapping[str, Any],
    categories: list[Mapping[str, Any]],
    outputs: list[Mapping[str, Any]],
    *,
    after_category: str,
) -> dict[str, int]:
    names = [str(row["category"]) for row in plan["categories"]]
    start = names.index(after_category) + 1
    values = {str(row["category"]): row for row in categories}
    for name in names[start:]:
        if int(values[name]["mapped_tokens"]) < int(
            values[name]["mapped_target_tokens"]
        ):
            return resume_cursor_for_category(plan, outputs, name)
    return {"next_line": 0, "source_ordinal": len(plan["source_schedule"])}


def advance_map_checkpoint(
    checkpoint: Mapping[str, Any],
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    output: Mapping[str, Any],
) -> dict[str, Any]:
    """Commit one content-addressed, checkpoint-bound candidate-map output."""

    current = validate_map_checkpoint(checkpoint, plan, authorization)
    if current["status"] != "mapping":
        raise ValueError("production map checkpoint is not accepting outputs")
    if current["active_reservations"]:
        raise ValueError("sequential map advancement cannot bypass active reservations")
    material = dict(output)
    observed = material.pop("manifest_sha256", None)
    if _sha256(observed, "map-output manifest") != _sha256_bytes(
        _canonical_json_bytes(material)
    ):
        raise ValueError("production map-output manifest SHA-256 changed")
    if output.get("plan_fingerprint") != plan["plan_fingerprint"]:
        raise ValueError("production map output targets another plan")
    if output.get("parent_checkpoint_sha256") != current["checkpoint_sha256"]:
        raise ValueError("production map output parent changed")
    cursor = current["source_cursor"]
    ordinal = int(cursor["source_ordinal"])
    if ordinal >= len(plan["source_schedule"]):
        raise ValueError("production map source inventory is exhausted")
    source = plan["source_schedule"][ordinal]
    if output.get("source_ordinal") != ordinal or output.get("category") != source["category"]:
        raise ValueError("production map output source identity changed")
    if output.get("source_sha256") != source["source_sha256"]:
        raise ValueError("production map output source SHA-256 changed")
    if output.get("start_after_line") != int(cursor["next_line"]):
        raise ValueError("production map output start line changed")
    end_line = _non_negative_int(output.get("end_line"), "map-output end line")
    if end_line <= int(cursor["next_line"]):
        raise ValueError("production map output did not advance its source line")
    if not isinstance(output.get("source_complete"), bool):
        raise ValueError("production map output completion state is invalid")
    tokens = _non_negative_int(output.get("candidate_tokens"), "map-output tokens")
    documents = _non_negative_int(output.get("candidate_documents"), "map-output documents")
    if (tokens == 0) != (documents == 0):
        raise ValueError("production map output token/document emptiness differs")
    categories = [dict(row) for row in current["categories"]]
    category = next(row for row in categories if row["category"] == source["category"])
    category["mapped_tokens"] += tokens
    category["mapped_documents"] += documents
    if category["mapped_tokens"] > category["mapped_target_tokens"] + int(
        plan["limits"]["maximum_category_overshoot_tokens"]
    ):
        raise ValueError("production map output exceeds category overshoot")
    outputs = [*current["outputs"], dict(output)]
    if category["mapped_tokens"] >= category["mapped_target_tokens"]:
        next_cursor = _next_incomplete_cursor(
            plan,
            categories,
            outputs,
            after_category=str(source["category"]),
        )
    elif output["source_complete"]:
        next_ordinal = ordinal + 1
        if (
            next_ordinal >= len(plan["source_schedule"])
            or plan["source_schedule"][next_ordinal]["category"] != source["category"]
        ):
            raise RuntimeError("production map exhausted a category before its target")
        next_cursor = {"next_line": 0, "source_ordinal": next_ordinal}
    else:
        next_cursor = {"next_line": end_line, "source_ordinal": ordinal}
    result = {
        **current,
        "categories": categories,
        "checkpoint_sequence": int(current["checkpoint_sequence"]) + 1,
        "output_sequence": int(current["output_sequence"]) + 1,
        "outputs": outputs,
        "previous_checkpoint_sha256": current["checkpoint_sha256"],
        "source_cursor": next_cursor,
        "total_mapped_documents": int(current["total_mapped_documents"]) + documents,
        "total_mapped_tokens": int(current["total_mapped_tokens"]) + tokens,
    }
    result.pop("checkpoint_sha256", None)
    if all(
        int(row["mapped_tokens"]) >= int(row["mapped_target_tokens"])
        for row in categories
    ):
        result["status"] = "ready_for_global_dedup"
    result["checkpoint_sha256"] = map_checkpoint_fingerprint(result)
    return validate_map_checkpoint(result, plan, authorization)


def commit_reserved_map_output(
    checkpoint: Mapping[str, Any],
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    output: Mapping[str, Any],
) -> dict[str, Any]:
    """Commit one completed category lane without blocking unrelated lanes."""

    current = validate_map_checkpoint(checkpoint, plan, authorization)
    token = output.get("reservation_token")
    existing = next(
        (
            row
            for row in current["outputs"]
            if row.get("reservation_token") == token
        ),
        None,
    )
    if existing is not None:
        if existing != output:
            raise ValueError("production map reservation token was reused")
        return current
    reservation = next(
        (
            row
            for row in current["active_reservations"]
            if row["reservation_token"] == token
        ),
        None,
    )
    if reservation is None:
        raise ValueError("production map output has no active reservation")
    material = dict(output)
    observed = material.pop("manifest_sha256", None)
    if _sha256(observed, "map-output manifest") != _sha256_bytes(
        _canonical_json_bytes(material)
    ):
        raise ValueError("production map-output manifest SHA-256 changed")
    if output.get("plan_fingerprint") != plan["plan_fingerprint"]:
        raise ValueError("production map output targets another plan")
    category_name = str(reservation["category"])
    cursor = reservation["source_cursor"]
    ordinal = int(cursor["source_ordinal"])
    source = plan["source_schedule"][ordinal]
    expected = {
        "category": category_name,
        "source_ordinal": ordinal,
        "source_sha256": source["source_sha256"],
        "start_after_line": int(cursor["next_line"]),
    }
    for name, value in expected.items():
        if output.get(name) != value:
            raise ValueError(f"production reserved map output changed {name}")
    end_line = _non_negative_int(output.get("end_line"), "map-output end line")
    if end_line <= int(cursor["next_line"]):
        raise ValueError("production reserved map output did not advance")
    if not isinstance(output.get("source_complete"), bool):
        raise ValueError("production reserved map source completion is invalid")
    tokens = _non_negative_int(output.get("candidate_tokens"), "map-output tokens")
    documents = _non_negative_int(
        output.get("candidate_documents"), "map-output documents"
    )
    if (tokens == 0) != (documents == 0):
        raise ValueError("production map output token/document emptiness differs")
    uri = output.get("parquet_s3_uri")
    if tokens:
        if not isinstance(uri, str) or not uri.startswith(
            f"s3://{plan['storage']['bucket']}/{plan['storage']['namespace']}/"
        ):
            raise ValueError("production map output left the production namespace")
    elif uri is not None:
        raise ValueError("empty production map output must not publish Parquet")
    categories = [dict(row) for row in current["categories"]]
    category = next(row for row in categories if row["category"] == category_name)
    category["mapped_tokens"] += tokens
    category["mapped_documents"] += documents
    if category["mapped_tokens"] > category["mapped_target_tokens"] + int(
        plan["limits"]["maximum_category_overshoot_tokens"]
    ):
        raise ValueError("production reserved map output exceeds category overshoot")
    outputs = [*current["outputs"], dict(output)]
    reservations = [
        dict(row)
        for row in current["active_reservations"]
        if row["reservation_token"] != token
    ]
    incomplete = [
        str(row["category"])
        for row in categories
        if int(row["mapped_tokens"]) < int(row["mapped_target_tokens"])
    ]
    if incomplete:
        first = incomplete[0]
        matching = next(
            (row for row in reservations if row["category"] == first), None
        )
        next_cursor = (
            dict(matching["source_cursor"])
            if matching is not None
            else resume_cursor_for_category(plan, outputs, first)
        )
    else:
        next_cursor = {"next_line": 0, "source_ordinal": len(plan["source_schedule"])}
    result = {
        **current,
        "active_reservations": reservations,
        "categories": categories,
        "checkpoint_sequence": int(current["checkpoint_sequence"]) + 1,
        "output_sequence": int(current["output_sequence"]) + 1,
        "outputs": outputs,
        "previous_checkpoint_sha256": current["checkpoint_sha256"],
        "source_cursor": next_cursor,
        "total_mapped_documents": int(current["total_mapped_documents"]) + documents,
        "total_mapped_tokens": int(current["total_mapped_tokens"]) + tokens,
    }
    result.pop("checkpoint_sha256", None)
    if not reservations and not incomplete:
        result["status"] = "ready_for_global_dedup"
    result["checkpoint_sha256"] = map_checkpoint_fingerprint(result)
    return validate_map_checkpoint(result, plan, authorization)


def build_map_job_request(
    plan: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    authorization: Mapping[str, Any],
    *,
    application_id: str,
    execution_role_arn: str,
    plan_s3_uri: str,
    map_checkpoint_s3_uri: str,
    authorization_s3_uri: str,
    staged_source_s3_uri: str,
    code_archive_s3_uri: str,
    job_script_s3_uri: str,
) -> dict[str, Any]:
    """Render one 16-vCPU stateless map job without calling AWS."""

    validate_map_checkpoint(checkpoint, plan, authorization)
    if re.fullmatch(r"[0-9a-z]+", application_id) is None:
        raise ValueError("production map application ID is invalid")
    if re.fullmatch(
        r"arn:aws:iam::[0-9]{12}:role/[A-Za-z0-9+=,.@_/-]+", execution_role_arn
    ) is None:
        raise ValueError("production map execution role ARN is invalid")
    bucket = str(plan["storage"]["bucket"])
    files = [
        plan_s3_uri,
        map_checkpoint_s3_uri,
        authorization_s3_uri,
        staged_source_s3_uri,
        f"s3://{bucket}/{plan['storage']['artifacts']['benchmark_registry']}",
        f"s3://{bucket}/{plan['storage']['artifacts']['frozen_tokenizer']}",
        f"s3://{bucket}/{plan['storage']['artifacts']['training_exclusion_ledger']}",
    ]
    file_names = [_bounded_s3_uri(uri, bucket=bucket) for uri in files]
    if len(file_names) != len(set(file_names)):
        raise ValueError("production map SparkFiles basenames collide")
    _bounded_s3_uri(code_archive_s3_uri, bucket=bucket)
    _bounded_s3_uri(job_script_s3_uri, bucket=bucket)
    maximum_vcpu = int(plan["pipeline"]["map_job_vcpu"])
    shape = _resource_shape(plan["application"], maximum_vcpu)
    python = "/opt/lumi/python312/bin/python"
    partitions = shape["executor_instances"] * 16
    spark_parameters = [
        "--conf", "spark.driver.cores=4",
        "--conf", "spark.driver.memory=14g",
        "--conf", "spark.executor.cores=4",
        "--conf", "spark.executor.memory=14g",
        "--conf", f"spark.executor.instances={shape['executor_instances']}",
        "--conf", "spark.dynamicAllocation.enabled=false",
        "--conf", "spark.emr-serverless.driver.disk=20G",
        "--conf", "spark.emr-serverless.executor.disk=20G",
        "--conf", f"spark.sql.shuffle.partitions={partitions}",
        "--conf", f"spark.default.parallelism={partitions}",
        "--conf", "spark.network.crypto.enabled=true",
        "--conf", "spark.io.encryption.enabled=true",
        "--conf", f"spark.emr-serverless.driverEnv.PYSPARK_DRIVER_PYTHON={python}",
        "--conf", f"spark.emr-serverless.driverEnv.PYSPARK_PYTHON={python}",
        "--conf", f"spark.executorEnv.PYSPARK_PYTHON={python}",
        "--py-files", code_archive_s3_uri,
        "--files", ",".join(files),
    ]
    job_token = map_job_token(authorization, checkpoint)
    runtime_minutes = int(
        Decimal(str(plan["pipeline"]["map_job_timeout_hours"])) * Decimal(60)
    )
    return {
        "applicationId": application_id,
        "clientToken": f"lumi-production-map-{job_token}",
        "configurationOverrides": {
            "monitoringConfiguration": {
                "s3MonitoringConfiguration": {
                    "logUri": (
                        f"s3://{bucket}/{plan['storage']['namespace']}/control/distributed/"
                        f"production/map-runs/{job_token}/logs"
                    )
                }
            }
        },
        "executionRoleArn": execution_role_arn,
        "executionTimeoutMinutes": runtime_minutes,
        "jobDriver": {
            "sparkSubmit": {
                "entryPoint": job_script_s3_uri,
                "entryPointArguments": [
                    "--lumi-authorization-file", file_names[2],
                    "--lumi-job-token", job_token,
                    "--lumi-map-checkpoint-file", file_names[1],
                    "--lumi-plan-file", file_names[0],
                    "--lumi-run-token", str(authorization["run_token"]),
                    "--lumi-staged-source-file", file_names[3],
                ],
                "sparkSubmitParameters": " ".join(spark_parameters),
            }
        },
        "mode": "BATCH",
        "name": f"lumi-production-map-16vcpu-{job_token[:12]}",
        "retryPolicy": {"maxAttempts": 1},
        "tags": {
            "LumiPlanFingerprint": str(plan["plan_fingerprint"]),
            "LumiScope": "bounded-production-corpus-map-only",
        },
    }


def build_reserved_map_job_request(
    plan: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    authorization: Mapping[str, Any],
    *,
    reservation_token: str,
    application_id: str,
    execution_role_arn: str,
    plan_s3_uri: str,
    map_checkpoint_s3_uri: str,
    authorization_s3_uri: str,
    staged_source_s3_uri: str,
    code_archive_s3_uri: str,
    job_script_s3_uri: str,
) -> dict[str, Any]:
    """Render one private-S3-only 16-vCPU category-lane map job."""

    current = validate_map_checkpoint(checkpoint, plan, authorization)
    reservation = next(
        (
            row
            for row in current["active_reservations"]
            if row["reservation_token"] == reservation_token
        ),
        None,
    )
    if reservation is None:
        raise ValueError("production map reservation is not active")
    if re.fullmatch(r"[0-9a-z]+", application_id) is None:
        raise ValueError("production map application ID is invalid")
    if re.fullmatch(
        r"arn:aws:iam::[0-9]{12}:role/[A-Za-z0-9+=,.@_/-]+", execution_role_arn
    ) is None:
        raise ValueError("production map execution role ARN is invalid")
    bucket = str(plan["storage"]["bucket"])
    files = [
        plan_s3_uri,
        map_checkpoint_s3_uri,
        authorization_s3_uri,
        staged_source_s3_uri,
        f"s3://{bucket}/{plan['storage']['artifacts']['benchmark_registry']}",
        f"s3://{bucket}/{plan['storage']['artifacts']['frozen_tokenizer']}",
        f"s3://{bucket}/{plan['storage']['artifacts']['training_exclusion_ledger']}",
    ]
    file_names = [_bounded_s3_uri(uri, bucket=bucket) for uri in files]
    if len(file_names) != len(set(file_names)):
        raise ValueError("reserved production map SparkFiles basenames collide")
    _bounded_s3_uri(code_archive_s3_uri, bucket=bucket)
    _bounded_s3_uri(job_script_s3_uri, bucket=bucket)
    shape = _resource_shape(plan["application"], int(plan["pipeline"]["map_job_vcpu"]))
    python = "/opt/lumi/python312/bin/python"
    partitions = shape["executor_instances"] * 16
    spark_parameters = [
        "--conf", "spark.driver.cores=4",
        "--conf", "spark.driver.memory=14g",
        "--conf", "spark.executor.cores=4",
        "--conf", "spark.executor.memory=14g",
        "--conf", f"spark.executor.instances={shape['executor_instances']}",
        "--conf", "spark.dynamicAllocation.enabled=false",
        "--conf", "spark.emr-serverless.driver.disk=20G",
        "--conf", "spark.emr-serverless.executor.disk=20G",
        "--conf", f"spark.sql.shuffle.partitions={partitions}",
        "--conf", f"spark.default.parallelism={partitions}",
        "--conf", "spark.network.crypto.enabled=true",
        "--conf", "spark.io.encryption.enabled=true",
        "--conf", f"spark.emr-serverless.driverEnv.PYSPARK_DRIVER_PYTHON={python}",
        "--conf", f"spark.emr-serverless.driverEnv.PYSPARK_PYTHON={python}",
        "--conf", f"spark.executorEnv.PYSPARK_PYTHON={python}",
        "--py-files", code_archive_s3_uri,
        "--files", ",".join(files),
    ]
    runtime_minutes = int(
        Decimal(str(plan["pipeline"]["map_job_timeout_hours"])) * Decimal(60)
    )
    category = str(reservation["category"])
    return {
        "applicationId": application_id,
        "clientToken": f"lumi-production-map-{reservation_token}",
        "configurationOverrides": {
            "monitoringConfiguration": {
                "s3MonitoringConfiguration": {
                    "logUri": (
                        f"s3://{bucket}/{plan['storage']['namespace']}/control/distributed/"
                        f"production/map-runs/{reservation_token}/logs"
                    )
                }
            }
        },
        "executionRoleArn": execution_role_arn,
        "executionTimeoutMinutes": runtime_minutes,
        "jobDriver": {
            "sparkSubmit": {
                "entryPoint": job_script_s3_uri,
                "entryPointArguments": [
                    "--lumi-authorization-file", file_names[2],
                    "--lumi-job-token", reservation_token,
                    "--lumi-map-checkpoint-file", file_names[1],
                    "--lumi-plan-file", file_names[0],
                    "--lumi-run-token", str(authorization["run_token"]),
                    "--lumi-staged-source-file", file_names[3],
                ],
                "sparkSubmitParameters": " ".join(spark_parameters),
            }
        },
        "mode": "BATCH",
        "name": f"lumi-map-{category[:16]}-{reservation_token[:12]}",
        "retryPolicy": {"maxAttempts": 1},
        "tags": {
            "LumiCategory": category,
            "LumiPlanFingerprint": str(plan["plan_fingerprint"]),
            "LumiScope": "bounded-production-corpus-reserved-map-only",
        },
    }
