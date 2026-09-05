#!/usr/bin/env python3
"""Preflight and independently replay Be-DSTC2 curation exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from lumi.posttraining_bedstc2_banglish_conversation_curation_v1 import (
    EXPECTED_LIMITS,
    BuildArtifacts,
    build_artifacts,
    build_verification,
    json_bytes,
    load_plan,
    render_owner_packet,
    validate_report,
    validate_verification,
)
from lumi.posttraining_candidate_mix_preflight import file_sha256
from lumi.posttraining_first_party import canonical_json_bytes
from scripts.curate_sft_bedstc2_banglish_conversation_v1 import (
    RUNNER_MARKER,
    VERIFIER_MARKER,
    _attempt_private_inventory,
    _finish_marker,
    _prospective_total_write_bytes,
    _start_marker,
    arm_runtime_alarm,
    cancel_runtime_alarm,
    output_paths,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _parse_canonical_jsonl(payload: bytes, *, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not payload or not payload.endswith(b"\n"):
        raise ValueError(f"Be-DSTC2 {name} is empty or not newline terminated")
    for line in payload.splitlines():
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Be-DSTC2 {name} is invalid JSONL") from error
        if not isinstance(row, dict) or canonical_json_bytes(row) != line:
            raise ValueError(f"Be-DSTC2 {name} is not canonical JSONL")
        rows.append(row)
    return rows


def _validate_telemetry(payload: bytes, report: Mapping[str, Any]) -> dict[str, str]:
    rows = _parse_canonical_jsonl(payload, name="telemetry")
    candidate = report["candidate"]
    if len(rows) != candidate["records"]:
        raise ValueError("Be-DSTC2 telemetry row count changed")
    ids: dict[str, str] = {}
    train_tokens = 0
    validation_tokens = 0
    for row in rows:
        if (
            set(row) != {"record_id", "serialized_tokens", "split", "supervised_tokens"}
            or not isinstance(row["record_id"], str)
            or row["split"] not in {"train", "validation"}
            or not isinstance(row["serialized_tokens"], int)
            or not 1 <= row["serialized_tokens"] <= 4096
            or not isinstance(row["supervised_tokens"], int)
            or row["supervised_tokens"] < 2
            or row["supervised_tokens"] > row["serialized_tokens"]
            or row["record_id"] in ids
        ):
            raise ValueError("Be-DSTC2 telemetry schema changed")
        ids[row["record_id"]] = row["split"]
        if row["split"] == "train":
            train_tokens += row["supervised_tokens"]
        else:
            validation_tokens += row["supervised_tokens"]
    if (
        train_tokens != candidate["train_supervised_tokens"]
        or validation_tokens != candidate["validation_supervised_tokens"]
    ):
        raise ValueError("Be-DSTC2 telemetry token totals changed")
    return ids


def _validate_rejections(payload: bytes, report: Mapping[str, Any]) -> None:
    rows = _parse_canonical_jsonl(payload, name="rejection ledger")
    if len(rows) != report["curation"]["raw_text_free_rejection_records"]:
        raise ValueError("Be-DSTC2 rejection-ledger count changed")
    for row in rows:
        if set(row) not in (
            {"reason_codes", "record_id"},
            {"matched_record_id_sha256", "reason_codes", "record_id"},
        ):
            raise ValueError("Be-DSTC2 rejection-ledger fields changed")
        if (
            not isinstance(row["record_id"], str)
            or not isinstance(row["reason_codes"], list)
            or not row["reason_codes"]
            or any(not isinstance(reason, str) or not reason for reason in row["reason_codes"])
        ):
            raise ValueError("Be-DSTC2 rejection-ledger value changed")
        matched = row.get("matched_record_id_sha256")
        if matched is not None and (not isinstance(matched, str) or len(matched) != 64):
            raise ValueError("Be-DSTC2 rejection match binding changed")
        if any(key in row for key in ("messages", "content", "user_prompt", "assistant_response")):
            raise ValueError("raw text entered Be-DSTC2 rejection ledger")


def _validate_owner(
    *,
    packet_payload: bytes,
    html_payload: bytes,
    report: Mapping[str, Any],
    telemetry_ids: Mapping[str, str],
) -> None:
    try:
        packet = json.loads(packet_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Be-DSTC2 owner packet is invalid JSON") from error
    if not isinstance(packet, list) or json_bytes(packet) != packet_payload:
        raise ValueError("Be-DSTC2 owner packet bytes are not canonical")
    expected_packet, expected_html, semantic, file_hash = render_owner_packet(packet)
    packet_ids = {str(row["record_id"]) for row in packet}
    if (
        packet_payload != expected_packet
        or html_payload != expected_html
        or not packet_ids.issubset(telemetry_ids)
        or semantic != report["owner_review"]["packet_semantic_sha256"]
        or file_hash != report["owner_review"]["packet_file_sha256"]
    ):
        raise ValueError("Be-DSTC2 owner packet or HTML binding changed")


def _artifact_payloads(artifacts: BuildArtifacts) -> dict[str, bytes]:
    return {
        "candidate": artifacts.candidate_bytes,
        "owner_review_html": artifacts.html_bytes,
        "owner_review_packet": artifacts.owner_packet_bytes,
        "rejection_ledger": artifacts.rejection_bytes,
        "telemetry": artifacts.telemetry_bytes,
    }


def preflight_first_pass_outputs(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    paths: Mapping[str, Path],
    root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any], bytes]:
    artifact_names = {
        "candidate",
        "owner_review_html",
        "owner_review_packet",
        "rejection_ledger",
        "telemetry",
    }
    if set(paths) != artifact_names | {"report", "verification"}:
        raise ValueError("Be-DSTC2 verifier output inventory changed")
    required = {paths[name] for name in artifact_names | {"report"}}
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("Be-DSTC2 first-pass output set is incomplete")
    if paths["verification"].exists():
        raise FileExistsError("Be-DSTC2 verification already exists")
    if root is not None:
        observed_private = _attempt_private_inventory(root)
        expected_private = {paths[name].resolve() for name in artifact_names}
        if observed_private != expected_private:
            raise ValueError("unexpected Be-DSTC2 private output inventory")

    report_payload = paths["report"].read_bytes()
    if report_payload == RUNNER_MARKER:
        raise ValueError("Be-DSTC2 pass-one marker was never committed")
    try:
        report = json.loads(report_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Be-DSTC2 report is invalid JSON") from error
    validate_report(report)
    if json_bytes(report) != report_payload:
        raise ValueError("Be-DSTC2 report is not canonical JSON")
    if (
        report["plan_file_sha256"] != file_sha256(plan_path)
        or report["authorization"] != plan["authorization"]
        or report["bindings"]
        != {name: value["sha256"] for name, value in sorted(plan["bindings"].items())}
    ):
        raise ValueError("Be-DSTC2 report is bound to a different plan")

    payloads = {name: paths[name].read_bytes() for name in artifact_names}
    for name, payload in payloads.items():
        evidence = report["artifacts"][name]
        if len(payload) != evidence["bytes"] or _sha256(payload) != evidence["sha256"]:
            raise ValueError(f"Be-DSTC2 first-pass artifact changed: {name}")
    telemetry_ids = _validate_telemetry(payloads["telemetry"], report)
    _validate_rejections(payloads["rejection_ledger"], report)
    _validate_owner(
        packet_payload=payloads["owner_review_packet"],
        html_payload=payloads["owner_review_html"],
        report=report,
        telemetry_ids=telemetry_ids,
    )
    prospective = build_verification(
        report=report,
        report_file_sha256=_sha256(report_payload),
    )
    validate_verification(prospective)
    prospective_payload = json_bytes(prospective)
    if root is not None:
        total = _prospective_total_write_bytes(
            root=root,
            plan_path=plan_path,
            payloads=payloads,
            report=report_payload,
            verification=prospective_payload,
        )
    else:
        total = (
            sum(len(payload) for payload in payloads.values())
            + len(report_payload)
            + len(prospective_payload)
        )
    if total > EXPECTED_LIMITS["maximum_private_write_bytes"]:
        raise ValueError("Be-DSTC2 complete output set exceeds private-write cap")
    return report, payloads, prospective, prospective_payload


def _preflight_then_replay(
    preflight: Callable[[], Any], replay: Callable[[], Any]
) -> tuple[Any, Any]:
    state = preflight()
    return state, replay()


def main() -> int:
    started = time.monotonic()
    arm_runtime_alarm()
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = load_plan(plan_path, repository_root=root)
    paths = output_paths(root)
    preflight_state = preflight_first_pass_outputs(
        plan=plan,
        plan_path=plan_path,
        paths=paths,
        root=root,
    )
    if time.monotonic() - started > 30 * 60:
        raise TimeoutError("Be-DSTC2 verifier exceeded its 30-minute half-budget")

    # The authorized verification path is the durable pass-two no-retry marker.
    _start_marker(paths["verification"], payload=VERIFIER_MARKER)
    report, observed_payloads, prospective, prospective_payload = preflight_state
    replay = build_artifacts(
        plan=plan,
        plan_file_sha256=file_sha256(plan_path),
        repository_root=root,
        started=started,
    )
    replay_payloads = _artifact_payloads(replay)
    if replay_payloads != observed_payloads or replay.report != report:
        raise ValueError("Be-DSTC2 independent pass did not exactly reproduce pass one")
    reproduced = build_verification(
        report=report,
        report_file_sha256=file_sha256(paths["report"]),
    )
    if reproduced != prospective or json_bytes(reproduced) != prospective_payload:
        raise ValueError("Be-DSTC2 prospective verification changed after replay")
    if time.monotonic() - started > 30 * 60:
        raise TimeoutError("Be-DSTC2 verifier exceeded its 30-minute half-budget")
    _finish_marker(paths["verification"], marker=VERIFIER_MARKER, payload=prospective_payload)
    cancel_runtime_alarm()
    print(
        json.dumps(
            {
                "report_file_sha256": reproduced["report_file_sha256"],
                "status": reproduced["status"],
                "verification_file_sha256": _sha256(prospective_payload),
                "verification_internal_sha256": reproduced["verification_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
