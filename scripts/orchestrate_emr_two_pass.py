#!/usr/bin/env python3
"""Resume Lumi's authorized two-pass EMR run without automatic retries."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lumi.emr_production_adaptive import (
    dedup_result_fingerprint as adaptive_dedup_fingerprint,
)
from lumi.emr_production_adaptive import (
    reopen_map_checkpoint,
    validate_adaptive_dedup_result,
)
from lumi.emr_production_dedup import build_global_dedup_job_request, dedup_job_token
from lumi.emr_production_map import (
    build_reserved_map_job_request,
    commit_reserved_map_output,
    initial_map_checkpoint,
    reserve_map_categories,
    validate_map_checkpoint,
    validate_reserved_staged_source,
)
from lumi.emr_production_seal import build_final_corpus_seal, write_final_corpus_seal
from lumi.emr_production_shard_writer import (
    build_shard_writer_job_request,
    dedup_result_fingerprint,
    shard_writer_job_token,
    validate_global_dedup_result,
)
from lumi.emr_production_staging import publish_staged_source, split_gzip_source
from lumi.emr_serverless_production import (
    validate_paid_production_authorization,
    validate_production_checkpoint,
)
from lumi.emr_two_pass_controller import (
    bind_controller_artifacts,
    initial_two_pass_controller_state,
    record_job_terminal,
    record_jobs_started,
    stop_two_pass_controller,
    transition_controller_phase,
    validate_two_pass_controller_state,
)
from lumi.production_corpus import _download_source

TERMINAL_EMR_STATES = {"CANCELLED", "FAILED", "SUCCESS"}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def _aws_error_code(error: BaseException) -> str | None:
    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return None
    details = response.get("Error")
    if not isinstance(details, Mapping):
        return None
    code = details.get("Code")
    return None if code is None else str(code)


def _put_immutable_json(
    value: Mapping[str, Any], *, bucket: str, key: str, s3_client: Any
) -> str:
    body = _canonical_json_bytes(value)
    digest = _sha256_bytes(body)
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ChecksumSHA256=base64.b64encode(bytes.fromhex(digest)).decode(),
            ContentType="application/json",
            IfNoneMatch="*",
            Metadata={"sha256": digest},
            ServerSideEncryption="AES256",
        )
    except BaseException as error:
        if _aws_error_code(error) not in {"PreconditionFailed", "412"}:
            raise
        head = s3_client.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
        if int(head.get("ContentLength", -1)) != len(body):
            raise ValueError("existing two-pass control object byte size changed") from error
        if head.get("Metadata", {}).get("sha256") != digest:
            raise ValueError("existing two-pass control object SHA-256 changed") from error
    return f"s3://{bucket}/{key}"


def _get_json(*, bucket: str, key: str, s3_client: Any) -> dict[str, Any]:
    response = s3_client.get_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
    body = response["Body"].read()
    digest = _sha256_bytes(body)
    metadata_digest = response.get("Metadata", {}).get("sha256")
    if metadata_digest is not None and metadata_digest != digest:
        raise ValueError("downloaded two-pass object SHA-256 changed")
    value = json.loads(body)
    if not isinstance(value, dict):
        raise ValueError("downloaded two-pass object is not a JSON object")
    return value


def _s3_key(uri_or_key: str) -> str:
    if uri_or_key.startswith("s3://"):
        return uri_or_key.split("/", 3)[3]
    return uri_or_key


def _runtime_seconds(job: Mapping[str, Any]) -> int:
    measured = job.get("totalExecutionDurationSeconds")
    if isinstance(measured, int) and measured > 0:
        return measured
    created = job.get("createdAt")
    updated = job.get("updatedAt")
    if created is not None and updated is not None:
        return max(1, int((updated - created).total_seconds()))
    return 1


def _ensure_application_started(emr_client: Any, application_id: str, poll: int) -> None:
    while True:
        state = emr_client.get_application(applicationId=application_id)["application"][
            "state"
        ]
        if state == "STARTED":
            return
        if state in {"CREATED", "STOPPED"}:
            emr_client.start_application(applicationId=application_id)
        elif state not in {"STARTING", "STOPPING"}:
            raise RuntimeError(f"EMR application cannot start from {state}")
        time.sleep(poll)


def _stop_application(emr_client: Any, application_id: str, poll: int) -> None:
    while True:
        state = emr_client.get_application(applicationId=application_id)["application"][
            "state"
        ]
        if state in {"CREATED", "STOPPED", "TERMINATED"}:
            return
        if state == "STARTED":
            emr_client.stop_application(applicationId=application_id)
        elif state not in {"STOPPING"}:
            return
        time.sleep(poll)


def _persist(
    state: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    state_path: Path,
    checkpoint_path: Path,
    plan: Mapping[str, Any],
    s3_client: Any,
) -> None:
    _write_json(checkpoint_path, checkpoint)
    _write_json(state_path, state)
    key = (
        f"{plan['storage']['namespace']}/control/distributed/production/two-pass-state/"
        f"{state['state_sha256']}.json"
    )
    _put_immutable_json(
        state, bucket=str(plan["storage"]["bucket"]), key=key, s3_client=s3_client
    )


def _publish_inputs(
    plan: Mapping[str, Any],
    accepted: Mapping[str, Any],
    authorization: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    s3_client: Any,
) -> dict[str, str]:
    bucket = str(plan["storage"]["bucket"])
    base = f"{plan['storage']['namespace']}/control/distributed/production/inputs"
    return {
        "accepted": _put_immutable_json(
            accepted,
            bucket=bucket,
            key=f"{base}/accepted-{accepted['checkpoint_sha256']}.json",
            s3_client=s3_client,
        ),
        "authorization": _put_immutable_json(
            authorization,
            bucket=bucket,
            key=f"{base}/authorization-{authorization['run_token']}.json",
            s3_client=s3_client,
        ),
        "checkpoint": _put_immutable_json(
            checkpoint,
            bucket=bucket,
            key=f"{base}/map-{checkpoint['checkpoint_sha256']}.json",
            s3_client=s3_client,
        ),
        "plan": _put_immutable_json(
            plan,
            bucket=bucket,
            key=f"{base}/plan-{plan['plan_fingerprint']}.json",
            s3_client=s3_client,
        ),
    }


def _stage_reservation(
    plan: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    authorization: Mapping[str, Any],
    reservation: Mapping[str, Any],
    *,
    work_directory: Path,
    s3_client: Any,
) -> dict[str, Any]:
    """Download once outside EMR, split deterministically, and publish to private S3."""

    token = str(reservation["reservation_token"])
    durable = work_directory / "staged" / f"{token}.json"
    if durable.is_file():
        staged = _read_json(durable)
        return validate_reserved_staged_source(
            staged,
            checkpoint,
            plan,
            authorization,
            reservation_token=token,
        )
    cursor = reservation["source_cursor"]
    source = plan["source_schedule"][int(cursor["source_ordinal"])]
    work_directory.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f"lumi-stage-{token}-", dir=work_directory))
    try:
        source_directory = scratch / "source"
        source_directory.mkdir()
        maximum_local = int(plan["source_stager"]["maximum_local_staged_bytes"])
        source_path = _download_source(source, source_directory, maximum_local)
        compressed_cap = maximum_local - int(source["source_bytes"]) - 1_000_000_000
        if compressed_cap <= 0:
            raise RuntimeError("production stager local envelope is too small")
        chunks = scratch / "chunks"
        manifest = split_gzip_source(
            source_path,
            chunks,
            source_file=str(source["source_file"]),
            source_sha256=str(source["source_sha256"]),
            start_after_line=int(cursor["next_line"]),
            target_uncompressed_bytes=int(
                plan["source_stager"]["split_chunk_uncompressed_bytes"]
            ),
            hard_transient_bytes=int(plan["limits"]["maximum_transient_source_bytes"]),
            hard_compressed_bytes=compressed_cap,
            maximum_chunks=None,
        )
        prefix = (
            f"{plan['storage']['namespace']}/raw/transient/reservations/{token}"
        )
        remote = publish_staged_source(
            manifest,
            directory=chunks,
            bucket=str(plan["storage"]["bucket"]),
            key_prefix=prefix,
            s3_client=s3_client,
        )
        staged = {
            "authorization_run_token": authorization["run_token"],
            "gate": "lumi_emr_reserved_map_staged_source_v0.1",
            "manifest": remote,
            "plan_fingerprint": plan["plan_fingerprint"],
            "reservation_token": token,
            "schema_version": 1,
            "source_ordinal": cursor["source_ordinal"],
            "status": "staged",
        }
        validate_reserved_staged_source(
            staged,
            checkpoint,
            plan,
            authorization,
            reservation_token=token,
        )
        _write_json(durable, staged)
        return staged
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _publish_staged_input(
    staged: Mapping[str, Any], plan: Mapping[str, Any], *, s3_client: Any
) -> str:
    token = str(staged["reservation_token"])
    key = (
        f"{plan['storage']['namespace']}/control/distributed/production/inputs/"
        f"staged-{token}.json"
    )
    return _put_immutable_json(
        staged,
        bucket=str(plan["storage"]["bucket"]),
        key=key,
        s3_client=s3_client,
    )


def _cleanup_committed_staging(
    checkpoint: Mapping[str, Any],
    *,
    bucket: str,
    work_directory: Path,
    s3_client: Any,
) -> None:
    """Delete only verified raw chunk versions after their map output is committed."""

    staged_directory = work_directory / "staged"
    for output in checkpoint["outputs"]:
        token = output.get("reservation_token")
        if not isinstance(token, str):
            continue
        marker = staged_directory / f"{token}.cleaned"
        if marker.is_file():
            continue
        durable = staged_directory / f"{token}.json"
        if not durable.is_file():
            continue
        staged = _read_json(durable)
        objects = []
        for chunk in staged["manifest"]["chunks"]:
            value = {"Key": str(chunk["s3_key"])}
            if chunk.get("version_id") is not None:
                value["VersionId"] = str(chunk["version_id"])
            objects.append(value)
        for start in range(0, len(objects), 1_000):
            response = s3_client.delete_objects(
                Bucket=bucket,
                Delete={"Objects": objects[start : start + 1_000], "Quiet": True},
            )
            if response.get("Errors"):
                raise RuntimeError("raw staged-source cleanup reported S3 errors")
        marker.write_text(f"{output['manifest_sha256']}\n", encoding="utf-8")
def _launch(
    state: Mapping[str, Any],
    request: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    emr_client: Any,
) -> dict[str, Any]:
    preflight = {**descriptor, "job_run_id": "preflight"}
    record_jobs_started(state, plan, authorization, [preflight])
    response = emr_client.start_job_run(**dict(request))
    actual = {**descriptor, "job_run_id": str(response["jobRunId"])}
    return record_jobs_started(state, plan, authorization, [actual])


def _validate_map_report(
    report: Mapping[str, Any],
    job: Mapping[str, Any],
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> Mapping[str, Any]:
    expected = {
        "authorization_run_token": authorization["run_token"],
        "gate": "lumi_emr_production_reserved_map_result_v0.1",
        "job_token": job["job_token"],
        "map_checkpoint_sha256": job["metadata"]["map_checkpoint_sha256"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "schema_version": 1,
    }
    for name, value in expected.items():
        if report.get(name) != value:
            raise ValueError(f"reserved-map report changed {name}")
    output = report.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("reserved-map report output is missing")
    return output


def _terminal_job(emr_client: Any, job: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = emr_client.get_job_run(
        applicationId=job["application_id"], jobRunId=job["job_run_id"]
    )["jobRun"]
    return value if value["state"] in TERMINAL_EMR_STATES else None


def _record_terminal(
    state: Mapping[str, Any],
    job: Mapping[str, Any],
    remote: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    return record_job_terminal(
        state,
        plan,
        authorization,
        job_token=str(job["job_token"]),
        terminal_state=str(remote["state"]),
        billed=remote.get("billedResourceUtilization", {}),
        runtime_seconds=_runtime_seconds(remote),
    )


def _cancel_after_failure(
    state: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    emr_client: Any,
    poll: int,
) -> dict[str, Any]:
    current = dict(state)
    while current["active_jobs"]:
        changed = False
        for job in list(current["active_jobs"]):
            remote = _terminal_job(emr_client, job)
            if remote is None:
                try:
                    emr_client.cancel_job_run(
                        applicationId=job["application_id"], jobRunId=job["job_run_id"]
                    )
                except BaseException as error:
                    if _aws_error_code(error) not in {"ConflictException", "ValidationException"}:
                        raise
                continue
            current = _record_terminal(
                current,
                job,
                remote,
                plan=plan,
                authorization=authorization,
            )
            changed = True
        if current["active_jobs"] and not changed:
            time.sleep(poll)
    return current


def _verify_shard_objects(
    results: list[Mapping[str, Any]], *, bucket: str, s3_client: Any
) -> None:
    for result in results:
        for shard in result["shards"]:
            for key_name, bytes_name, sha_name in (
                ("object_key", "object_bytes", "object_sha256"),
                ("token_object_key", "token_object_bytes", "token_object_sha256"),
                (
                    "provenance_index_key",
                    "provenance_index_bytes",
                    "provenance_index_sha256",
                ),
            ):
                head = s3_client.head_object(
                    Bucket=bucket, Key=shard[key_name], ChecksumMode="ENABLED"
                )
                if int(head.get("ContentLength", -1)) != int(shard[bytes_name]):
                    raise ValueError("sealed shard object byte size changed")
                if head.get("Metadata", {}).get("sha256") != shard[sha_name]:
                    raise ValueError("sealed shard object SHA-256 changed")
            manifest = s3_client.get_object(
                Bucket=bucket,
                Key=shard["manifest_object_key"],
                ChecksumMode="ENABLED",
            )["Body"].read()
            if _sha256_bytes(manifest) != shard["manifest_sha256"]:
                raise ValueError("sealed shard publication manifest changed")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-checkpoint", type=Path, required=True)
    parser.add_argument("--application-id", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--code-archive-s3-uri", required=True)
    parser.add_argument("--controller-state", type=Path, required=True)
    parser.add_argument("--dedup-script-s3-uri", required=True)
    parser.add_argument("--execution-role-arn", required=True)
    parser.add_argument("--map-checkpoint", type=Path, required=True)
    parser.add_argument("--map-script-s3-uri", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--shard-script-s3-uri", required=True)
    parser.add_argument("--staging-work-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    import boto3

    args = _args()
    if not 5 <= args.poll_seconds <= 60:
        raise ValueError("two-pass poll interval must be between 5 and 60 seconds")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    plan = _read_json(args.plan)
    accepted = _read_json(args.accepted_checkpoint)
    authorization = _read_json(args.authorization)
    validate_production_checkpoint(accepted, plan)
    validate_paid_production_authorization(authorization, plan, accepted)
    if args.map_checkpoint.is_file():
        checkpoint = _read_json(args.map_checkpoint)
        validate_map_checkpoint(checkpoint, plan, authorization)
    else:
        checkpoint = initial_map_checkpoint(plan, accepted, authorization)
        _write_json(args.map_checkpoint, checkpoint)
    if args.controller_state.is_file():
        state = validate_two_pass_controller_state(
            _read_json(args.controller_state), plan, authorization
        )
    else:
        state = initial_two_pass_controller_state(plan, checkpoint, authorization)
        _write_json(args.controller_state, state)
    bucket = str(plan["storage"]["bucket"])
    namespace = str(plan["storage"]["namespace"])
    s3 = boto3.client("s3", region_name=str(plan["region"]))
    emr = boto3.client("emr-serverless", region_name=str(plan["region"]))
    if state["phase"] == "complete":
        _stop_application(emr, args.application_id, args.poll_seconds)
        print(json.dumps(state, indent=2, sort_keys=True))
        return
    if state["status"] != "failed":
        _ensure_application_started(emr, args.application_id, args.poll_seconds)

    while state["phase"] != "complete":
        if state["status"] == "failed":
            state = _cancel_after_failure(
                state,
                plan=plan,
                authorization=authorization,
                emr_client=emr,
                poll=args.poll_seconds,
            )
            _persist(
                state,
                checkpoint,
                state_path=args.controller_state,
                checkpoint_path=args.map_checkpoint,
                plan=plan,
                s3_client=s3,
            )
            _stop_application(emr, args.application_id, args.poll_seconds)
            break

        if state["phase"] == "mapping":
            _cleanup_committed_staging(
                checkpoint,
                bucket=bucket,
                work_directory=args.staging_work_directory,
                s3_client=s3,
            )
            changed = False
            for job in list(state["active_jobs"]):
                remote = _terminal_job(emr, job)
                if remote is None:
                    continue
                if remote["state"] == "SUCCESS":
                    try:
                        report = _get_json(
                            bucket=bucket, key=str(job["result_key"]), s3_client=s3
                        )
                        output = _validate_map_report(
                            report, job, plan, authorization
                        )
                        checkpoint = commit_reserved_map_output(
                            checkpoint, plan, authorization, output
                        )
                    except BaseException as error:
                        state = stop_two_pass_controller(
                            state,
                            plan,
                            authorization,
                            status="failed",
                            reason=f"map result reconciliation failed: {error}",
                        )
                state = _record_terminal(
                    state,
                    job,
                    remote,
                    plan=plan,
                    authorization=authorization,
                )
                inputs = _publish_inputs(
                    plan, accepted, authorization, checkpoint, s3_client=s3
                )
                state = bind_controller_artifacts(
                    state,
                    plan,
                    authorization,
                    map_checkpoint_key=_s3_key(inputs["checkpoint"]),
                    map_checkpoint_sha256=checkpoint["checkpoint_sha256"],
                )
                _persist(
                    state,
                    checkpoint,
                    state_path=args.controller_state,
                    checkpoint_path=args.map_checkpoint,
                    plan=plan,
                    s3_client=s3,
                )
                changed = True
            if state["status"] == "failed":
                continue
            if state["active_jobs"]:
                if not changed:
                    time.sleep(args.poll_seconds)
                continue
            if checkpoint["status"] == "ready_for_global_dedup":
                state = transition_controller_phase(
                    state, plan, authorization, "global_dedup"
                )
                _persist(
                    state,
                    checkpoint,
                    state_path=args.controller_state,
                    checkpoint_path=args.map_checkpoint,
                    plan=plan,
                    s3_client=s3,
                )
                continue
            if not checkpoint["active_reservations"]:
                checkpoint = reserve_map_categories(
                    checkpoint, plan, authorization
                )
                _write_json(args.map_checkpoint, checkpoint)
            inputs = _publish_inputs(
                plan, accepted, authorization, checkpoint, s3_client=s3
            )
            state = bind_controller_artifacts(
                state,
                plan,
                authorization,
                map_checkpoint_key=_s3_key(inputs["checkpoint"]),
                map_checkpoint_sha256=checkpoint["checkpoint_sha256"],
            )
            for reservation in checkpoint["active_reservations"]:
                token = str(reservation["reservation_token"])
                if any(job["job_token"] == token for job in state["active_jobs"]):
                    continue
                staged = _stage_reservation(
                    plan,
                    checkpoint,
                    authorization,
                    reservation,
                    work_directory=args.staging_work_directory,
                    s3_client=s3,
                )
                staged_uri = _publish_staged_input(staged, plan, s3_client=s3)
                request = build_reserved_map_job_request(
                    plan,
                    checkpoint,
                    authorization,
                    reservation_token=token,
                    application_id=args.application_id,
                    execution_role_arn=args.execution_role_arn,
                    plan_s3_uri=inputs["plan"],
                    map_checkpoint_s3_uri=inputs["checkpoint"],
                    authorization_s3_uri=inputs["authorization"],
                    staged_source_s3_uri=staged_uri,
                    code_archive_s3_uri=args.code_archive_s3_uri,
                    job_script_s3_uri=args.map_script_s3_uri,
                )
                result_key = (
                    f"{namespace}/control/distributed/production/map-runs/"
                    f"{token}/result.json"
                )
                descriptor = {
                    "application_id": args.application_id,
                    "job_token": token,
                    "kind": "map",
                    "metadata": {
                        "category": reservation["category"],
                        "map_checkpoint_key": _s3_key(inputs["checkpoint"]),
                        "map_checkpoint_sha256": checkpoint["checkpoint_sha256"],
                        "source_ordinal": reservation["source_cursor"]["source_ordinal"],
                    },
                    "result_key": result_key,
                    "timeout_minutes": request["executionTimeoutMinutes"],
                }
                state = _launch(
                    state,
                    request,
                    descriptor,
                    plan=plan,
                    authorization=authorization,
                    emr_client=emr,
                )
                _persist(
                    state,
                    checkpoint,
                    state_path=args.controller_state,
                    checkpoint_path=args.map_checkpoint,
                    plan=plan,
                    s3_client=s3,
                )
            continue

        if state["phase"] == "global_dedup":
            if state["active_jobs"]:
                job = state["active_jobs"][0]
                remote = _terminal_job(emr, job)
                if remote is None:
                    time.sleep(args.poll_seconds)
                    continue
                if remote["state"] != "SUCCESS":
                    state = _record_terminal(
                        state,
                        job,
                        remote,
                        plan=plan,
                        authorization=authorization,
                    )
                    _persist(
                        state,
                        checkpoint,
                        state_path=args.controller_state,
                        checkpoint_path=args.map_checkpoint,
                        plan=plan,
                        s3_client=s3,
                    )
                    continue
                try:
                    dedup = _get_json(
                        bucket=bucket, key=str(job["result_key"]), s3_client=s3
                    )
                    snapshot = _get_json(
                        bucket=bucket,
                        key=str(job["metadata"]["map_checkpoint_key"]),
                        s3_client=s3,
                    )
                    if dedup.get("status") == "pass_category_floors":
                        validate_global_dedup_result(dedup, plan, authorization)
                        next_phase = "sharding"
                    else:
                        validate_adaptive_dedup_result(
                            dedup, plan, authorization, snapshot
                        )
                        checkpoint = reopen_map_checkpoint(
                            checkpoint, plan, authorization, dedup
                        )
                        next_phase = "mapping"
                except BaseException as error:
                    state = stop_two_pass_controller(
                        state,
                        plan,
                        authorization,
                        status="failed",
                        reason=f"global dedup reconciliation failed: {error}",
                    )
                    state = _record_terminal(
                        state,
                        job,
                        remote,
                        plan=plan,
                        authorization=authorization,
                    )
                    _persist(
                        state,
                        checkpoint,
                        state_path=args.controller_state,
                        checkpoint_path=args.map_checkpoint,
                        plan=plan,
                        s3_client=s3,
                    )
                    continue
                state = _record_terminal(
                    state,
                    job,
                    remote,
                    plan=plan,
                    authorization=authorization,
                )
                state = bind_controller_artifacts(
                    state,
                    plan,
                    authorization,
                    dedup_result_key=str(job["result_key"]),
                    dedup_result_sha256=adaptive_dedup_fingerprint(dedup),
                    map_checkpoint_sha256=checkpoint["checkpoint_sha256"],
                )
                state = transition_controller_phase(
                    state, plan, authorization, next_phase
                )
                _persist(
                    state,
                    checkpoint,
                    state_path=args.controller_state,
                    checkpoint_path=args.map_checkpoint,
                    plan=plan,
                    s3_client=s3,
                )
                continue
            inputs = _publish_inputs(
                plan, accepted, authorization, checkpoint, s3_client=s3
            )
            token = dedup_job_token(authorization, checkpoint)
            request = build_global_dedup_job_request(
                plan,
                checkpoint,
                authorization,
                application_id=args.application_id,
                execution_role_arn=args.execution_role_arn,
                plan_s3_uri=inputs["plan"],
                map_checkpoint_s3_uri=inputs["checkpoint"],
                authorization_s3_uri=inputs["authorization"],
                code_archive_s3_uri=args.code_archive_s3_uri,
                job_script_s3_uri=args.dedup_script_s3_uri,
            )
            result_key = (
                f"{namespace}/control/distributed/production/dedup-runs/"
                f"{token}/result.json"
            )
            descriptor = {
                "application_id": args.application_id,
                "job_token": token,
                "kind": "global_dedup",
                "metadata": {
                    "map_checkpoint_key": _s3_key(inputs["checkpoint"]),
                    "map_checkpoint_sha256": checkpoint["checkpoint_sha256"],
                },
                "result_key": result_key,
                "timeout_minutes": request["executionTimeoutMinutes"],
            }
            state = _launch(
                state,
                request,
                descriptor,
                plan=plan,
                authorization=authorization,
                emr_client=emr,
            )
            _persist(
                state,
                checkpoint,
                state_path=args.controller_state,
                checkpoint_path=args.map_checkpoint,
                plan=plan,
                s3_client=s3,
            )
            continue

        if state["phase"] == "sharding":
            dedup = _get_json(
                bucket=bucket,
                key=str(state["artifacts"]["dedup_result_key"]),
                s3_client=s3,
            )
            validate_global_dedup_result(dedup, plan, authorization)
            changed = False
            for job in list(state["active_jobs"]):
                remote = _terminal_job(emr, job)
                if remote is None:
                    continue
                if remote["state"] == "SUCCESS":
                    try:
                        result = _get_json(
                            bucket=bucket, key=str(job["result_key"]), s3_client=s3
                        )
                        expected = {
                            "authorization_run_token": authorization["run_token"],
                            "category": job["metadata"]["category"],
                            "dedup_result_sha256": dedup_result_fingerprint(dedup),
                            "gate": "lumi_emr_production_shard_writer_result_v0.1",
                            "job_token": job["job_token"],
                            "plan_fingerprint": plan["plan_fingerprint"],
                            "schema_version": 1,
                            "status": "complete",
                        }
                        if any(result.get(name) != value for name, value in expected.items()):
                            raise ValueError("shard-writer result identity changed")
                    except BaseException as error:
                        state = stop_two_pass_controller(
                            state,
                            plan,
                            authorization,
                            status="failed",
                            reason=f"shard result reconciliation failed: {error}",
                        )
                state = _record_terminal(
                    state,
                    job,
                    remote,
                    plan=plan,
                    authorization=authorization,
                )
                if remote["state"] == "SUCCESS" and state["status"] != "failed":
                    keys = dict(state["artifacts"]["shard_result_keys"])
                    keys[str(job["metadata"]["category"])] = str(job["result_key"])
                    state = bind_controller_artifacts(
                        state, plan, authorization, shard_result_keys=keys
                    )
                _persist(
                    state,
                    checkpoint,
                    state_path=args.controller_state,
                    checkpoint_path=args.map_checkpoint,
                    plan=plan,
                    s3_client=s3,
                )
                changed = True
            if state["status"] == "failed":
                continue
            if state["active_jobs"]:
                if not changed:
                    time.sleep(args.poll_seconds)
                continue
            completed_categories = set(state["artifacts"]["shard_result_keys"])
            expected_categories = {str(row["category"]) for row in plan["categories"]}
            if completed_categories == expected_categories:
                state = transition_controller_phase(
                    state, plan, authorization, "sealing"
                )
                _persist(
                    state,
                    checkpoint,
                    state_path=args.controller_state,
                    checkpoint_path=args.map_checkpoint,
                    plan=plan,
                    s3_client=s3,
                )
                continue
            inputs = _publish_inputs(
                plan, accepted, authorization, checkpoint, s3_client=s3
            )
            dedup_uri = f"s3://{bucket}/{state['artifacts']['dedup_result_key']}"
            for category in sorted(expected_categories - completed_categories):
                token = shard_writer_job_token(authorization, dedup, category)
                request = build_shard_writer_job_request(
                    plan,
                    accepted,
                    authorization,
                    dedup,
                    category=category,
                    application_id=args.application_id,
                    execution_role_arn=args.execution_role_arn,
                    plan_s3_uri=inputs["plan"],
                    accepted_checkpoint_s3_uri=inputs["accepted"],
                    authorization_s3_uri=inputs["authorization"],
                    dedup_result_s3_uri=dedup_uri,
                    code_archive_s3_uri=args.code_archive_s3_uri,
                    job_script_s3_uri=args.shard_script_s3_uri,
                )
                result_key = (
                    f"{namespace}/control/distributed/production/shard-runs/"
                    f"{token}/result.json"
                )
                descriptor = {
                    "application_id": args.application_id,
                    "job_token": token,
                    "kind": "shard_writer",
                    "metadata": {
                        "category": category,
                        "dedup_result_sha256": dedup_result_fingerprint(dedup),
                    },
                    "result_key": result_key,
                    "timeout_minutes": request["executionTimeoutMinutes"],
                }
                state = _launch(
                    state,
                    request,
                    descriptor,
                    plan=plan,
                    authorization=authorization,
                    emr_client=emr,
                )
                _persist(
                    state,
                    checkpoint,
                    state_path=args.controller_state,
                    checkpoint_path=args.map_checkpoint,
                    plan=plan,
                    s3_client=s3,
                )
            continue

        if state["phase"] == "sealing":
            dedup = _get_json(
                bucket=bucket,
                key=str(state["artifacts"]["dedup_result_key"]),
                s3_client=s3,
            )
            writer_results = [
                _get_json(bucket=bucket, key=key, s3_client=s3)
                for _, key in sorted(state["artifacts"]["shard_result_keys"].items())
            ]
            _verify_shard_objects(writer_results, bucket=bucket, s3_client=s3)
            seal = build_final_corpus_seal(
                plan, accepted, authorization, dedup, writer_results
            )
            seal_directory = args.output_directory / f"seal-{seal['seal_sha256']}"
            if seal_directory.exists():
                existing = _read_json(seal_directory / "final-candidate-corpus.json")
                if existing != seal:
                    raise ValueError("existing local final corpus seal changed")
            else:
                write_final_corpus_seal(seal, seal_directory)
            seal_key = (
                f"{namespace}/control/distributed/production/final-seals/"
                f"{seal['seal_sha256']}.json"
            )
            _put_immutable_json(seal, bucket=bucket, key=seal_key, s3_client=s3)
            state = bind_controller_artifacts(
                state,
                plan,
                authorization,
                final_seal_key=seal_key,
                final_seal_sha256=seal["seal_sha256"],
            )
            state = transition_controller_phase(
                state, plan, authorization, "complete"
            )
            _persist(
                state,
                checkpoint,
                state_path=args.controller_state,
                checkpoint_path=args.map_checkpoint,
                plan=plan,
                s3_client=s3,
            )

    if state["phase"] == "complete":
        _stop_application(emr, args.application_id, args.poll_seconds)
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
