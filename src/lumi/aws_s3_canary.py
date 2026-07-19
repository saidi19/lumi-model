"""Bounded, resumable AWS S3 I/O canary for the Lumi corpus store."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .aws_s3_provider import build_aws_s3_provider_report

SCHEMA_VERSION = 1
RUN_ID_PATTERN = re.compile(r"[a-z0-9-]{8,40}")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any], *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if private:
        temporary.chmod(0o600)
    temporary.replace(path)


class AwsCli:
    """Small AWS CLI adapter that never logs credentials or environment values."""

    def __init__(self, profile: str | None, region: str) -> None:
        if profile == "" or not region:
            raise ValueError("AWS profile must be None or non-empty, and region is required")
        self.profile = profile
        self.region = region

    def call(
        self,
        service: str,
        operation: str,
        *arguments: str,
        allow_failure: bool = False,
    ) -> tuple[int, dict[str, Any], str]:
        command = ["aws"]
        if self.profile is not None:
            command.extend(("--profile", self.profile))
        command.extend(
            [
                "--region",
                self.region,
                "--no-cli-pager",
                "--output",
                "json",
                service,
                operation,
                *arguments,
            ]
        )
        environment = {**os.environ, "AWS_PAGER": ""}
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=environment,
            text=True,
        )
        if completed.returncode and not allow_failure:
            message = completed.stderr.strip() or "AWS CLI command failed"
            raise RuntimeError(f"{service} {operation}: {message}")
        output: dict[str, Any] = {}
        if completed.stdout.strip():
            parsed = json.loads(completed.stdout)
            if isinstance(parsed, dict):
                output = parsed
        return completed.returncode, output, completed.stderr.strip()


def _make_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(4)


def _validate_run_id(run_id: str) -> str:
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("canary run ID is invalid")
    return run_id


def _bucket_names(account_id: str, run_id: str) -> tuple[str, str]:
    if not account_id.isdigit() or len(account_id) != 12:
        raise ValueError("AWS account ID is invalid")
    suffix = hashlib.sha256(f"{account_id}:{run_id}".encode()).hexdigest()[:20]
    return f"lumi-corpus-canary-{suffix}", f"lumi-corpus-audit-{suffix}"


def _write_payload(path: Path, size: int, seed: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    remaining = size
    counter = 0
    with path.open("wb") as handle:
        while remaining:
            block = hashlib.sha256(f"{seed}:{counter}".encode()).digest()
            chunk = block[:remaining]
            handle.write(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
            counter += 1
        handle.flush()
        os.fsync(handle.fileno())
    return digest.hexdigest()


def _checksum_b64(path: Path) -> str:
    return base64.b64encode(bytes.fromhex(_sha256_file(path))).decode()


def _security_configuration(
    cli: AwsCli,
    bucket: str,
    account_id: str,
    run_id: str,
    *,
    lifecycle: dict[str, Any],
    policy: dict[str, Any],
) -> None:
    cli.call(
        "s3api",
        "put-public-access-block",
        "--bucket",
        bucket,
        "--public-access-block-configuration",
        json.dumps(
            {
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
            separators=(",", ":"),
        ),
        "--expected-bucket-owner",
        account_id,
    )
    cli.call(
        "s3api",
        "put-bucket-ownership-controls",
        "--bucket",
        bucket,
        "--ownership-controls",
        '{"Rules":[{"ObjectOwnership":"BucketOwnerEnforced"}]}',
        "--expected-bucket-owner",
        account_id,
    )
    cli.call(
        "s3api",
        "put-bucket-encryption",
        "--bucket",
        bucket,
        "--server-side-encryption-configuration",
        '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"},"BucketKeyEnabled":false}]}',
        "--expected-bucket-owner",
        account_id,
    )
    cli.call(
        "s3api",
        "put-bucket-versioning",
        "--bucket",
        bucket,
        "--versioning-configuration",
        '{"Status":"Enabled"}',
        "--expected-bucket-owner",
        account_id,
    )
    cli.call(
        "s3api",
        "put-bucket-lifecycle-configuration",
        "--bucket",
        bucket,
        "--lifecycle-configuration",
        json.dumps(lifecycle, separators=(",", ":")),
        "--expected-bucket-owner",
        account_id,
    )
    cli.call(
        "s3api",
        "put-bucket-policy",
        "--bucket",
        bucket,
        "--policy",
        json.dumps(policy, separators=(",", ":")),
        "--expected-bucket-owner",
        account_id,
    )
    cli.call(
        "s3api",
        "put-bucket-tagging",
        "--bucket",
        bucket,
        "--tagging",
        json.dumps(
            {
                "TagSet": [
                    {"Key": "Project", "Value": "Lumi"},
                    {"Key": "Purpose", "Value": "BoundedS3Canary"},
                    {"Key": "RunId", "Value": run_id},
                    {"Key": "DeleteAfterCanary", "Value": "true"},
                ]
            },
            separators=(",", ":"),
        ),
        "--expected-bucket-owner",
        account_id,
    )


def _deny_insecure_policy(bucket: str) -> dict[str, Any]:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "DenyInsecureTransport",
                "Effect": "Deny",
                "Principal": "*",
                "Action": "s3:*",
                "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"],
                "Condition": {"Bool": {"aws:SecureTransport": "false"}},
            }
        ],
    }


def _log_bucket_policy(log_bucket: str, data_bucket: str, account_id: str) -> dict[str, Any]:
    policy = _deny_insecure_policy(log_bucket)
    policy["Statement"].append(
        {
            "Sid": "AllowS3ServerAccessLogs",
            "Effect": "Allow",
            "Principal": {"Service": "logging.s3.amazonaws.com"},
            "Action": "s3:PutObject",
            "Resource": f"arn:aws:s3:::{log_bucket}/access/*",
            "Condition": {
                "ArnLike": {"aws:SourceArn": f"arn:aws:s3:::{data_bucket}"},
                "StringEquals": {"aws:SourceAccount": account_id},
            },
        }
    )
    return policy


def _audit_bucket(
    cli: AwsCli,
    bucket: str,
    account_id: str,
    *,
    expected_log_bucket: str | None,
) -> dict[str, bool]:
    _, public, _ = cli.call(
        "s3api",
        "get-public-access-block",
        "--bucket",
        bucket,
        "--expected-bucket-owner",
        account_id,
    )
    _, ownership, _ = cli.call(
        "s3api",
        "get-bucket-ownership-controls",
        "--bucket",
        bucket,
        "--expected-bucket-owner",
        account_id,
    )
    _, encryption, _ = cli.call(
        "s3api", "get-bucket-encryption", "--bucket", bucket, "--expected-bucket-owner", account_id
    )
    _, versioning, _ = cli.call(
        "s3api", "get-bucket-versioning", "--bucket", bucket, "--expected-bucket-owner", account_id
    )
    _, lifecycle, _ = cli.call(
        "s3api",
        "get-bucket-lifecycle-configuration",
        "--bucket",
        bucket,
        "--expected-bucket-owner",
        account_id,
    )
    _, policy_status, _ = cli.call(
        "s3api",
        "get-bucket-policy-status",
        "--bucket",
        bucket,
        "--expected-bucket-owner",
        account_id,
    )
    block = public.get("PublicAccessBlockConfiguration", {})
    rules = ownership.get("OwnershipControls", {}).get("Rules", [])
    encryption_rules = encryption.get("ServerSideEncryptionConfiguration", {}).get("Rules", [])
    lifecycle_rules = lifecycle.get("Rules", [])
    result = {
        "abort_incomplete_multipart_configured": any(
            rule.get("AbortIncompleteMultipartUpload", {}).get("DaysAfterInitiation") == 1
            for rule in lifecycle_rules
        ),
        "aes256_default_encryption": any(
            rule.get("ApplyServerSideEncryptionByDefault", {}).get("SSEAlgorithm") == "AES256"
            for rule in encryption_rules
        ),
        "bucket_owner_enforced": any(
            rule.get("ObjectOwnership") == "BucketOwnerEnforced" for rule in rules
        ),
        "not_public": policy_status.get("PolicyStatus", {}).get("IsPublic") is False,
        "public_access_fully_blocked": all(
            block.get(name) is True
            for name in (
                "BlockPublicAcls",
                "IgnorePublicAcls",
                "BlockPublicPolicy",
                "RestrictPublicBuckets",
            )
        ),
        "versioning_enabled": versioning.get("Status") == "Enabled",
    }
    if expected_log_bucket is not None:
        _, logging, _ = cli.call(
            "s3api", "get-bucket-logging", "--bucket", bucket, "--expected-bucket-owner", account_id
        )
        enabled = logging.get("LoggingEnabled", {})
        result["server_access_logging"] = (
            enabled.get("TargetBucket") == expected_log_bucket
            and enabled.get("TargetPrefix") == "access/"
        )
    if not all(result.values()):
        raise RuntimeError(f"S3 bucket security audit failed: {result}")
    return result


def _prepare_payloads(output: Path, plan: dict[str, Any], run_id: str) -> dict[str, Any]:
    temp = output / "temporary-payloads"
    conditional_size = int(plan["canary"]["conditional_put_bytes"])
    multipart_size = int(plan["canary"]["multipart_payload_bytes"])
    part_size = int(plan["canary"]["multipart_part_bytes"])
    conditional = temp / "conditional.bin"
    multipart = temp / "multipart.bin"
    conditional_digest = _write_payload(conditional, conditional_size, f"{run_id}:conditional")
    multipart_digest = _write_payload(multipart, multipart_size, f"{run_id}:multipart")
    parts = []
    with multipart.open("rb") as source:
        number = 1
        while True:
            chunk = source.read(part_size)
            if not chunk:
                break
            part = temp / f"multipart-part-{number:03d}.bin"
            part.write_bytes(chunk)
            parts.append(
                {
                    "bytes": len(chunk),
                    "number": number,
                    "path": str(part),
                    "sha256": _sha256_bytes(chunk),
                }
            )
            number += 1
    return {
        "conditional": {
            "bytes": conditional_size,
            "path": str(conditional),
            "sha256": conditional_digest,
        },
        "multipart": {
            "bytes": multipart_size,
            "parts": parts,
            "path": str(multipart),
            "sha256": multipart_digest,
        },
    }


def _verify_download(path: Path, expected_sha256: str, expected_bytes: int) -> None:
    if path.stat().st_size != expected_bytes or _sha256_file(path) != expected_sha256:
        raise RuntimeError("S3 downloaded payload failed size or SHA-256 verification")


def _create_bucket(cli: AwsCli, bucket: str) -> None:
    cli.call("s3api", "create-bucket", "--bucket", bucket)


def _tags_match(cli: AwsCli, bucket: str, account_id: str, run_id: str) -> bool:
    code, output, error = cli.call(
        "s3api",
        "get-bucket-tagging",
        "--bucket",
        bucket,
        "--expected-bucket-owner",
        account_id,
        allow_failure=True,
    )
    if code:
        if "NoSuchBucket" in error:
            return False
        raise RuntimeError(f"unable to verify canary bucket tags: {error}")
    tags = {item["Key"]: item["Value"] for item in output.get("TagSet", [])}
    return tags.get("RunId") == run_id and tags.get("DeleteAfterCanary") == "true"


def _empty_versioned_bucket(cli: AwsCli, bucket: str, account_id: str) -> None:
    _, uploads, _ = cli.call(
        "s3api",
        "list-multipart-uploads",
        "--bucket",
        bucket,
        "--expected-bucket-owner",
        account_id,
    )
    for upload in uploads.get("Uploads", []):
        cli.call(
            "s3api",
            "abort-multipart-upload",
            "--bucket",
            bucket,
            "--key",
            upload["Key"],
            "--upload-id",
            upload["UploadId"],
            "--expected-bucket-owner",
            account_id,
        )
    _, versions, _ = cli.call(
        "s3api",
        "list-object-versions",
        "--bucket",
        bucket,
        "--expected-bucket-owner",
        account_id,
    )
    for item in [*versions.get("Versions", []), *versions.get("DeleteMarkers", [])]:
        cli.call(
            "s3api",
            "delete-object",
            "--bucket",
            bucket,
            "--key",
            item["Key"],
            "--version-id",
            item["VersionId"],
            "--expected-bucket-owner",
            account_id,
        )


def _cleanup(cli: AwsCli, state: dict[str, Any]) -> dict[str, bool]:
    account_id = state["account_id"]
    run_id = state["run_id"]
    data_bucket = state["data_bucket"]
    log_bucket = state["log_bucket"]
    result = {
        "data_bucket_deleted": not state.get("data_bucket_created", False),
        "log_bucket_deleted": not state.get("log_bucket_created", False),
    }
    if state.get("data_bucket_created"):
        if not _tags_match(cli, data_bucket, account_id, run_id):
            raise RuntimeError("refusing to clean an untagged or foreign data bucket")
        cli.call(
            "s3api",
            "put-bucket-logging",
            "--bucket",
            data_bucket,
            "--bucket-logging-status",
            "{}",
            "--expected-bucket-owner",
            account_id,
        )
        _empty_versioned_bucket(cli, data_bucket, account_id)
        cli.call(
            "s3api", "delete-bucket", "--bucket", data_bucket, "--expected-bucket-owner", account_id
        )
        result["data_bucket_deleted"] = True
    if state.get("log_bucket_created"):
        if not _tags_match(cli, log_bucket, account_id, run_id):
            raise RuntimeError("refusing to clean an untagged or foreign log bucket")
        for attempt in range(6):
            _empty_versioned_bucket(cli, log_bucket, account_id)
            code, _, error = cli.call(
                "s3api",
                "delete-bucket",
                "--bucket",
                log_bucket,
                "--expected-bucket-owner",
                account_id,
                allow_failure=True,
            )
            if code == 0:
                result["log_bucket_deleted"] = True
                break
            if "BucketNotEmpty" not in error or attempt == 5:
                raise RuntimeError(f"unable to delete canary log bucket: {error}")
            time.sleep(2)
    return result


def _provider_plan(contract: Path, root: Path) -> dict[str, Any]:
    return build_aws_s3_provider_report(contract, repository_root=root)


def start_canary(
    contract: Path,
    output: Path,
    *,
    profile: str,
    repository_root: Path,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Create secure ephemeral buckets and stop after multipart part one."""

    state_path = output / "state.json"
    if state_path.exists() or (output / "report.json").exists():
        raise ValueError("canary output already has state or a final report")
    plan = _provider_plan(contract, repository_root)
    region = plan["provider"]["region"]
    cli = AwsCli(profile, region)
    _, identity, _ = cli.call("sts", "get-caller-identity")
    account_id = str(identity["Account"])
    run_id = _validate_run_id(run_id or _make_run_id())
    data_bucket, log_bucket = _bucket_names(account_id, run_id)
    state = {
        "account_id": account_id,
        "data_bucket": data_bucket,
        "data_bucket_created": False,
        "log_bucket": log_bucket,
        "log_bucket_created": False,
        "phase": "initializing",
        "plan_fingerprint": plan["plan_fingerprint"],
        "profile": profile,
        "region": region,
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "started_at": _utc_now(),
    }
    _write_json_atomic(state_path, state, private=True)

    _create_bucket(cli, log_bucket)
    state["log_bucket_created"] = True
    _write_json_atomic(state_path, state, private=True)
    log_lifecycle = {
        "Rules": [
            {
                "ID": "expire-canary-audit-after-seven-days",
                "Status": "Enabled",
                "Filter": {"Prefix": "access/"},
                "Expiration": {"Days": 7},
                "NoncurrentVersionExpiration": {"NoncurrentDays": 7},
                "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1},
            }
        ]
    }
    _security_configuration(
        cli,
        log_bucket,
        account_id,
        run_id,
        lifecycle=log_lifecycle,
        policy=_log_bucket_policy(log_bucket, data_bucket, account_id),
    )

    _create_bucket(cli, data_bucket)
    state["data_bucket_created"] = True
    _write_json_atomic(state_path, state, private=True)
    prefix = f"lumi-corpus/v0.1/canary/{run_id}/"
    data_lifecycle = {
        "Rules": [
            {
                "ID": "abort-canary-multipart-after-one-day",
                "Status": "Enabled",
                "Filter": {"Prefix": prefix},
                "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1},
            }
        ]
    }
    _security_configuration(
        cli,
        data_bucket,
        account_id,
        run_id,
        lifecycle=data_lifecycle,
        policy=_deny_insecure_policy(data_bucket),
    )
    cli.call(
        "s3api",
        "put-bucket-logging",
        "--bucket",
        data_bucket,
        "--bucket-logging-status",
        json.dumps(
            {"LoggingEnabled": {"TargetBucket": log_bucket, "TargetPrefix": "access/"}},
            separators=(",", ":"),
        ),
        "--expected-bucket-owner",
        account_id,
    )
    state["initial_security_audit"] = {
        "data": _audit_bucket(cli, data_bucket, account_id, expected_log_bucket=log_bucket),
        "log": _audit_bucket(cli, log_bucket, account_id, expected_log_bucket=None),
    }

    payloads = _prepare_payloads(output, plan, run_id)
    state["payloads"] = payloads
    conditional_key = f"{prefix}conditional/{payloads['conditional']['sha256']}.bin"
    multipart_key = f"{prefix}multipart/{payloads['multipart']['sha256']}.bin"
    state["conditional_key"] = conditional_key
    state["multipart_key"] = multipart_key
    conditional_path = Path(payloads["conditional"]["path"])
    _, conditional_put, _ = cli.call(
        "s3api",
        "put-object",
        "--bucket",
        data_bucket,
        "--key",
        conditional_key,
        "--body",
        str(conditional_path),
        "--if-none-match",
        "*",
        "--metadata",
        f"sha256={payloads['conditional']['sha256']}",
        "--checksum-algorithm",
        "SHA256",
        "--checksum-sha256",
        _checksum_b64(conditional_path),
        "--server-side-encryption",
        "AES256",
        "--storage-class",
        "STANDARD",
        "--expected-bucket-owner",
        account_id,
    )
    code, _, duplicate_error = cli.call(
        "s3api",
        "put-object",
        "--bucket",
        data_bucket,
        "--key",
        conditional_key,
        "--body",
        str(conditional_path),
        "--if-none-match",
        "*",
        "--expected-bucket-owner",
        account_id,
        allow_failure=True,
    )
    if code == 0 or not ("PreconditionFailed" in duplicate_error or "412" in duplicate_error):
        raise RuntimeError("S3 conditional-create conflict was not rejected")
    full_download = output / "temporary-payloads" / "conditional-download.bin"
    cli.call(
        "s3api",
        "get-object",
        "--bucket",
        data_bucket,
        "--key",
        conditional_key,
        "--checksum-mode",
        "ENABLED",
        "--expected-bucket-owner",
        account_id,
        str(full_download),
    )
    _verify_download(
        full_download,
        payloads["conditional"]["sha256"],
        payloads["conditional"]["bytes"],
    )
    state["conditional_test"] = {
        "checksum_roundtrip": True,
        "conditional_conflict_rejected": True,
        "version_id_present": bool(conditional_put.get("VersionId")),
    }

    _, upload, _ = cli.call(
        "s3api",
        "create-multipart-upload",
        "--bucket",
        data_bucket,
        "--key",
        multipart_key,
        "--metadata",
        f"sha256={payloads['multipart']['sha256']}",
        "--server-side-encryption",
        "AES256",
        "--storage-class",
        "STANDARD",
        "--expected-bucket-owner",
        account_id,
    )
    state["upload_id"] = upload["UploadId"]
    first = payloads["multipart"]["parts"][0]
    _, uploaded, _ = cli.call(
        "s3api",
        "upload-part",
        "--bucket",
        data_bucket,
        "--key",
        multipart_key,
        "--part-number",
        str(first["number"]),
        "--upload-id",
        state["upload_id"],
        "--body",
        first["path"],
        "--expected-bucket-owner",
        account_id,
    )
    state["uploaded_parts"] = [{"ETag": uploaded["ETag"], "PartNumber": first["number"]}]
    state["phase"] = "interrupted_after_first_multipart_part"
    state["start_phase_completed_at"] = _utc_now()
    _write_json_atomic(state_path, state, private=True)
    return {
        "data_bucket_sha256": _sha256_bytes(data_bucket.encode()),
        "log_bucket_sha256": _sha256_bytes(log_bucket.encode()),
        "phase": state["phase"],
        "run_id": run_id,
        "uploaded_parts": 1,
    }


def _seal_report(output: Path, report: dict[str, Any]) -> None:
    report_path = output / "report.json"
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    report_path.write_text(encoded, encoding="utf-8")
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    (output / "manifest.sha256").write_text(f"{digest}  report.json\n", encoding="utf-8")


def resume_canary(
    contract: Path,
    output: Path,
    *,
    profile: str,
    repository_root: Path,
) -> dict[str, Any]:
    """Resume multipart upload in a fresh process, verify, and clean all resources."""

    state_path = output / "state.json"
    state = _read_json(state_path)
    if state.get("phase") != "interrupted_after_first_multipart_part":
        raise ValueError("S3 canary state is not at the intentional resume boundary")
    plan = _provider_plan(contract, repository_root)
    if state.get("plan_fingerprint") != plan["plan_fingerprint"]:
        raise ValueError("S3 canary state belongs to another provider plan")
    if profile != state.get("profile"):
        raise ValueError("S3 canary resume profile changed")
    cli = AwsCli(profile, state["region"])
    _, identity, _ = cli.call("sts", "get-caller-identity")
    if str(identity["Account"]) != state["account_id"]:
        raise ValueError("S3 canary resumed in a different AWS account")
    payloads = _prepare_payloads(output, plan, state["run_id"])
    if payloads["conditional"]["sha256"] != state["payloads"]["conditional"]["sha256"]:
        raise RuntimeError("conditional payload changed across the process boundary")
    if payloads["multipart"]["sha256"] != state["payloads"]["multipart"]["sha256"]:
        raise RuntimeError("multipart payload changed across the process boundary")
    data_bucket = state["data_bucket"]
    account_id = state["account_id"]
    _, remote, _ = cli.call(
        "s3api",
        "list-parts",
        "--bucket",
        data_bucket,
        "--key",
        state["multipart_key"],
        "--upload-id",
        state["upload_id"],
        "--expected-bucket-owner",
        account_id,
    )
    remote_parts = remote.get("Parts", [])
    if len(remote_parts) != 1 or remote_parts[0]["ETag"] != state["uploaded_parts"][0]["ETag"]:
        raise RuntimeError("remote multipart state does not match the durable resume checkpoint")
    parts = list(state["uploaded_parts"])
    for part in payloads["multipart"]["parts"][1:]:
        _, uploaded, _ = cli.call(
            "s3api",
            "upload-part",
            "--bucket",
            data_bucket,
            "--key",
            state["multipart_key"],
            "--part-number",
            str(part["number"]),
            "--upload-id",
            state["upload_id"],
            "--body",
            part["path"],
            "--expected-bucket-owner",
            account_id,
        )
        parts.append({"ETag": uploaded["ETag"], "PartNumber": part["number"]})
        state["uploaded_parts"] = parts
        _write_json_atomic(state_path, state, private=True)
    cli.call(
        "s3api",
        "complete-multipart-upload",
        "--bucket",
        data_bucket,
        "--key",
        state["multipart_key"],
        "--upload-id",
        state["upload_id"],
        "--multipart-upload",
        json.dumps({"Parts": parts}, separators=(",", ":")),
        "--expected-bucket-owner",
        account_id,
    )
    _, head, _ = cli.call(
        "s3api",
        "head-object",
        "--bucket",
        data_bucket,
        "--key",
        state["multipart_key"],
        "--expected-bucket-owner",
        account_id,
    )
    if (
        head.get("ContentLength") != payloads["multipart"]["bytes"]
        or head.get("Metadata", {}).get("sha256") != payloads["multipart"]["sha256"]
        or head.get("ServerSideEncryption") != "AES256"
        or not head.get("VersionId")
    ):
        raise RuntimeError("completed multipart object metadata failed verification")
    download = output / "temporary-payloads" / "multipart-download.bin"
    cli.call(
        "s3api",
        "get-object",
        "--bucket",
        data_bucket,
        "--key",
        state["multipart_key"],
        "--expected-bucket-owner",
        account_id,
        str(download),
    )
    _verify_download(download, payloads["multipart"]["sha256"], payloads["multipart"]["bytes"])
    range_start = int(plan["canary"]["range_start_byte"])
    range_end = int(plan["canary"]["range_end_byte"])
    range_path = output / "temporary-payloads" / "multipart-range.bin"
    cli.call(
        "s3api",
        "get-object",
        "--bucket",
        data_bucket,
        "--key",
        state["multipart_key"],
        "--range",
        f"bytes={range_start}-{range_end}",
        "--expected-bucket-owner",
        account_id,
        str(range_path),
    )
    with Path(payloads["multipart"]["path"]).open("rb") as source:
        source.seek(range_start)
        expected_range = source.read(range_end - range_start + 1)
    if range_path.read_bytes() != expected_range:
        raise RuntimeError("S3 range GET did not match the local payload")
    final_security = {
        "data": _audit_bucket(
            cli, data_bucket, account_id, expected_log_bucket=state["log_bucket"]
        ),
        "log": _audit_bucket(cli, state["log_bucket"], account_id, expected_log_bucket=None),
    }
    cleanup = _cleanup(cli, state)
    report = {
        "account_id_sha256": _sha256_bytes(account_id.encode()),
        "authorization": {
            "bulk_corpus_materialization": False,
            "full_4b_training": False,
            "paid_compute": False,
            "transfer_bulk_source_data": False,
        },
        "bucket_name_sha256": {
            "data": _sha256_bytes(data_bucket.encode()),
            "log": _sha256_bytes(state["log_bucket"].encode()),
        },
        "checks": {
            "conditional_checksum_roundtrip": state["conditional_test"]["checksum_roundtrip"],
            "conditional_create_conflict_rejected": state["conditional_test"][
                "conditional_conflict_rejected"
            ],
            "conditional_version_id_present": state["conditional_test"]["version_id_present"],
            "final_security_audit": final_security,
            "fresh_process_resume": True,
            "full_multipart_sha256_roundtrip": True,
            "initial_security_audit": state["initial_security_audit"],
            "multipart_parts": len(parts),
            "range_get_parity": True,
            "remote_parts_before_resume": len(remote_parts),
        },
        "cleanup": cleanup,
        "completed_at": _utc_now(),
        "cost": {
            "estimated_maximum_cost_usd": plan["canary"]["estimated_maximum_cost_usd"],
            "hard_cap_usd": plan["canary"]["maximum_canary_cost_usd"],
        },
        "decision": {
            "bounded_s3_io_canary": "pass",
            "bucket_resources_retained": False,
            "bulk_corpus_materialization": "not_authorized",
            "full_4b_training": "not_authorized",
            "next_action": "seal_complete_source_manifest_and_cccc_owner_decision",
            "paid_compute": "not_authorized",
        },
        "gate": "aws_s3_bounded_io_canary_v0.1",
        "implementation_sha256": _sha256_file(Path(__file__).resolve()),
        "payloads": {
            "conditional": {
                "bytes": payloads["conditional"]["bytes"],
                "sha256": payloads["conditional"]["sha256"],
            },
            "multipart": {
                "bytes": payloads["multipart"]["bytes"],
                "sha256": payloads["multipart"]["sha256"],
            },
            "total_bytes": payloads["conditional"]["bytes"] + payloads["multipart"]["bytes"],
        },
        "plan_fingerprint": plan["plan_fingerprint"],
        "region": state["region"],
        "run_id": state["run_id"],
        "schema_version": SCHEMA_VERSION,
        "started_at": state["started_at"],
    }
    validate_canary_report(report, plan, repository_root=repository_root)
    _seal_report(output, report)
    state_path.unlink()
    temporary = output / "temporary-payloads"
    for path in sorted(temporary.glob("*")):
        path.unlink()
    temporary.rmdir()
    return report


def cleanup_canary(output: Path, *, profile: str) -> dict[str, bool]:
    """Safely remove resources after a failed or deliberately stopped canary."""

    state = _read_json(output / "state.json")
    if profile != state.get("profile"):
        raise ValueError("S3 cleanup profile changed")
    cli = AwsCli(profile, state["region"])
    _, identity, _ = cli.call("sts", "get-caller-identity")
    if str(identity["Account"]) != state["account_id"]:
        raise ValueError("S3 cleanup is using a different AWS account")
    result = _cleanup(cli, state)
    state["cleanup"] = result
    state["phase"] = "cleaned_after_incomplete_canary"
    _write_json_atomic(output / "cleanup.json", state, private=True)
    (output / "state.json").unlink()
    temporary = output / "temporary-payloads"
    if temporary.exists():
        for path in sorted(temporary.glob("*")):
            path.unlink()
        temporary.rmdir()
    return result


def validate_canary_report(
    report: dict[str, Any],
    provider_plan: dict[str, Any],
    *,
    repository_root: Path,
) -> None:
    """Validate final evidence without retaining bucket names or credentials."""

    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported AWS S3 canary report schema")
    if report.get("gate") != "aws_s3_bounded_io_canary_v0.1":
        raise ValueError("unexpected AWS S3 canary report gate")
    if report.get("plan_fingerprint") != provider_plan["plan_fingerprint"]:
        raise ValueError("S3 canary report belongs to another provider plan")
    if report.get("region") != "us-east-1":
        raise ValueError("S3 canary ran outside the approved region")
    payloads = report.get("payloads", {})
    if payloads.get("total_bytes", 0) > provider_plan["canary"]["authorized_payload_bytes"]:
        raise ValueError("S3 canary report exceeds its payload authorization")
    checks = report.get("checks", {})
    required_checks = (
        "conditional_checksum_roundtrip",
        "conditional_create_conflict_rejected",
        "conditional_version_id_present",
        "fresh_process_resume",
        "full_multipart_sha256_roundtrip",
        "range_get_parity",
    )
    if any(checks.get(name) is not True for name in required_checks):
        raise ValueError("S3 canary report has a failed I/O check")
    if checks.get("remote_parts_before_resume") != 1 or checks.get("multipart_parts", 0) < 2:
        raise ValueError("S3 canary did not prove a bounded multipart resume")
    for audit_name in ("initial_security_audit", "final_security_audit"):
        audit = checks.get(audit_name, {})
        if not isinstance(audit, dict) or not audit:
            raise ValueError("S3 canary security audit is missing")
        if any(not all(bucket.values()) for bucket in audit.values()):
            raise ValueError("S3 canary security audit failed")
    cleanup = report.get("cleanup", {})
    if cleanup != {"data_bucket_deleted": True, "log_bucket_deleted": True}:
        raise ValueError("S3 canary left bucket resources behind")
    if report.get("decision") != {
        "bounded_s3_io_canary": "pass",
        "bucket_resources_retained": False,
        "bulk_corpus_materialization": "not_authorized",
        "full_4b_training": "not_authorized",
        "next_action": "seal_complete_source_manifest_and_cccc_owner_decision",
        "paid_compute": "not_authorized",
    }:
        raise ValueError("S3 canary decision changed")
    implementation = repository_root / "src" / "lumi" / "aws_s3_canary.py"
    if report.get("implementation_sha256") != _sha256_file(implementation):
        raise ValueError("S3 canary implementation hash changed")


def verify_canary_evidence(
    contract: Path,
    output: Path,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    """Verify the sealed canary report and manifest after cloud cleanup."""

    report_path = output / "report.json"
    manifest_path = output / "manifest.sha256"
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise ValueError("S3 canary manifest must have one entry")
    parts = lines[0].split()
    if len(parts) != 2 or parts[1] != "report.json":
        raise ValueError("S3 canary manifest entry is invalid")
    if _sha256_file(report_path) != parts[0]:
        raise ValueError("S3 canary report hash does not match its manifest")
    report = _read_json(report_path)
    plan = _provider_plan(contract, repository_root)
    validate_canary_report(report, plan, repository_root=repository_root)
    return report
