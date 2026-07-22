#!/usr/bin/env python3
"""Dry-run or launch exactly one authorized Lumi final-corpus Glue canary."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from lumi.glue_final_corpus import (
    AUTHORIZATION_FILE,
    CODE_ARCHIVE,
    JOB_SCRIPT,
    PLAN_FILE,
    RELEASE_FILE,
    build_glue_final_corpus_canary_job_definition,
    glue_final_corpus_canary_job_token,
    validate_glue_final_corpus_canary_authorization,
    validate_glue_final_corpus_canary_release,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _aws_error_code(error: BaseException) -> str | None:
    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return None
    details = response.get("Error")
    if not isinstance(details, Mapping):
        return None
    code = details.get("Code")
    return None if code is None else str(code)


def _verify_package(package: Path) -> None:
    manifest = package / "manifest.sha256"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    observed: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if name in observed or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Glue final-corpus package entry is invalid: {name}")
        observed.add(name)
        path = package / name
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"Glue final-corpus package manifest mismatch: {name}")
    if observed != {CODE_ARCHIVE, JOB_SCRIPT, PLAN_FILE, RELEASE_FILE}:
        raise ValueError("Glue final-corpus package file set changed")


def _s3_key(uri: str, *, bucket: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or parsed.netloc != bucket:
        raise ValueError("Glue final-corpus URI left the sealed bucket")
    key = parsed.path.lstrip("/")
    if not key or ".." in Path(key).parts:
        raise ValueError("Glue final-corpus S3 key is invalid")
    return key


def _put_exact(s3: Any, *, bucket: str, key: str, path: Path) -> dict[str, Any]:
    digest = _sha256(path)
    size = path.stat().st_size
    try:
        with path.open("rb") as body:
            response = s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ChecksumSHA256=base64.b64encode(bytes.fromhex(digest)).decode(),
                IfNoneMatch="*",
                Metadata={"sha256": digest},
                ServerSideEncryption="AES256",
            )
    except BaseException as error:
        if _aws_error_code(error) not in {"PreconditionFailed", "412"}:
            raise
        head = s3.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
        if int(head.get("ContentLength", -1)) != size:
            raise ValueError(f"existing bootstrap object size changed: {key}") from error
        if head.get("Metadata", {}).get("sha256") != digest:
            raise ValueError(f"existing bootstrap object hash changed: {key}") from error
        version_id = head.get("VersionId")
    else:
        version_id = response.get("VersionId")
    return {
        "bytes": size,
        "key": key,
        "sha256": digest,
        "version_id": None if version_id is None else str(version_id),
    }


def _claim_once(
    s3: Any, *, bucket: str, key: str, claim: Mapping[str, Any]
) -> dict[str, Any]:
    body = json.dumps(claim, separators=(",", ":"), sort_keys=True).encode()
    digest = hashlib.sha256(body).hexdigest()
    try:
        response = s3.put_object(
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
        if _aws_error_code(error) in {"PreconditionFailed", "412"}:
            raise RuntimeError(
                "Glue final-corpus authorization already has a launch claim"
            ) from error
        raise
    return {
        "bytes": len(body),
        "key": key,
        "sha256": digest,
        "version_id": response.get("VersionId"),
    }


def _trust() -> dict[str, Any]:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Action": "sts:AssumeRole",
                "Effect": "Allow",
                "Principal": {"Service": "glue.amazonaws.com"},
            }
        ],
    }


def _policy(
    *,
    account: str,
    region: str,
    bucket: str,
    accepted_prefix: str,
    exact_read_keys: list[str],
    canary_prefix: str,
    result_prefix: str,
    temp_prefix: str,
) -> dict[str, Any]:
    bucket_arn = f"arn:aws:s3:::{bucket}"
    list_prefixes = [
        f"{accepted_prefix}/*",
        f"{canary_prefix}/*",
        f"{result_prefix}/*",
        f"{temp_prefix}/*",
        *exact_read_keys,
    ]
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Action": ["s3:GetBucketLocation", "s3:ListBucketMultipartUploads"],
                "Effect": "Allow",
                "Resource": bucket_arn,
                "Sid": "BucketLocation",
            },
            {
                "Action": "s3:ListBucket",
                "Condition": {"StringLike": {"s3:prefix": list_prefixes}},
                "Effect": "Allow",
                "Resource": bucket_arn,
                "Sid": "ListBoundedPrefixes",
            },
            {
                "Action": ["s3:GetObject", "s3:GetObjectVersion"],
                "Effect": "Allow",
                "Resource": [
                    f"{bucket_arn}/{accepted_prefix}/*",
                    *[f"{bucket_arn}/{key}" for key in exact_read_keys],
                ],
                "Sid": "ReadExactInputs",
            },
            {
                "Action": [
                    "s3:AbortMultipartUpload",
                    "s3:DeleteObject",
                    "s3:GetObject",
                    "s3:GetObjectVersion",
                    "s3:ListMultipartUploadParts",
                    "s3:PutObject",
                ],
                "Effect": "Allow",
                "Resource": [
                    f"{bucket_arn}/{canary_prefix}/*",
                    f"{bucket_arn}/{temp_prefix}/*",
                ],
                "Sid": "ReadWriteCanaryAndTempOnly",
            },
            {
                "Action": [
                    "s3:AbortMultipartUpload",
                    "s3:GetObject",
                    "s3:GetObjectVersion",
                    "s3:ListMultipartUploadParts",
                    "s3:PutObject",
                ],
                "Effect": "Allow",
                "Resource": f"{bucket_arn}/{result_prefix}/*",
                "Sid": "WriteImmutableResultOnly",
            },
            {
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                "Effect": "Allow",
                "Resource": f"arn:aws:logs:{region}:{account}:log-group:/aws-glue/jobs/*",
                "Sid": "GlueLogs",
            },
            {
                "Action": "cloudwatch:PutMetricData",
                "Condition": {"StringEquals": {"cloudwatch:namespace": "Glue"}},
                "Effect": "Allow",
                "Resource": "*",
                "Sid": "GlueMetrics",
            },
        ],
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--evidence-directory", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _args()
    package = args.package.resolve()
    _verify_package(package)
    release_path = package / RELEASE_FILE
    release = _read_json(release_path)
    authorization_path = args.authorization.resolve()
    if authorization_path.name != AUTHORIZATION_FILE:
        raise ValueError("Glue final-corpus authorization filename changed")
    authorization = _read_json(authorization_path)
    validate_glue_final_corpus_canary_release(release)
    validate_glue_final_corpus_canary_authorization(authorization, release)
    bucket = str(release["storage"]["bucket"])
    namespace = str(release["storage"]["namespace"])
    region = str(release["region"])
    token = glue_final_corpus_canary_job_token(authorization)
    release_key = (
        f"{namespace}/control/glue-canary/bootstrap/releases/"
        f"{_sha256(release_path)}.json"
    )
    authorization_key = (
        f"{namespace}/control/glue-canary/bootstrap/authorizations/"
        f"{_sha256(authorization_path)}.json"
    )
    claim_key = (
        f"{namespace}/control/glue-canary/claims/"
        f"{authorization['authorization_sha256']}.json"
    )
    artifact_paths = {
        "code_archive": package / CODE_ARCHIVE,
        "final_plan": package / PLAN_FILE,
        "job_script": package / JOB_SCRIPT,
    }
    artifact_keys: dict[str, str] = {}
    for name, path in artifact_paths.items():
        artifact = release["artifacts"][name]
        if _sha256(path) != artifact["sha256"] or path.stat().st_size != int(
            artifact["bytes"]
        ):
            raise ValueError(f"Glue final-corpus artifact changed: {name}")
        artifact_keys[name] = _s3_key(str(artifact["s3_uri"]), bucket=bucket)

    accepted_prefix = _s3_key(
        str(release["input"]["accepted_candidates"]["s3_uri"]), bucket=bucket
    ).rstrip("/")
    baseline_keys = [
        _s3_key(str(row["s3_uri"]), bucket=bucket)
        for row in release["baseline"]["sealed_shards"]
    ]
    baseline_keys.append(
        _s3_key(str(release["baseline"]["archive"]["s3_uri"]), bucket=bucket)
    )
    exact_read_keys = [
        release_key,
        authorization_key,
        *artifact_keys.values(),
        *baseline_keys,
    ]
    canary_prefix = f"{release['output']['canary_prefix']}/{token}"
    result_prefix = f"{release['output']['result_prefix']}/{token}"
    temp_prefix = f"{namespace}/control/glue-canary/temp/{token}"
    role_name = f"lumi-final-canary-{token[:12]}"
    release_uri = f"s3://{bucket}/{release_key}"
    authorization_uri = f"s3://{bucket}/{authorization_key}"
    plan = {
        "authorization_sha256": authorization["authorization_sha256"],
        "canary_prefix": canary_prefix,
        "claim_key": claim_key,
        "job_name": f"lumi-glue-final-corpus-canary-{token[:12]}",
        "maximum_compute_usd": authorization["bounds"]["maximum_usd"],
        "number_of_workers": release["job"]["number_of_workers"],
        "release_fingerprint": release["release_fingerprint"],
        "result_prefix": result_prefix,
        "role_name": role_name,
        "timeout_minutes": release["job"]["maximum_runtime_minutes"],
    }
    if not args.execute:
        print(
            json.dumps(
                {**plan, "status": "dry_run_not_launched"},
                indent=2,
                sort_keys=True,
            )
        )
        return

    args.evidence_directory.mkdir(parents=True, exist_ok=True)
    output = args.evidence_directory / "launch-record.json"
    if output.exists():
        raise FileExistsError(output)

    import boto3

    session = boto3.Session(region_name=region)
    sts = session.client("sts")
    s3 = session.client("s3")
    iam = session.client("iam")
    glue = session.client("glue")
    account = str(sts.get_caller_identity()["Account"])
    uploaded = {
        "authorization": _put_exact(
            s3, bucket=bucket, key=authorization_key, path=authorization_path
        ),
        "code_archive": _put_exact(
            s3,
            bucket=bucket,
            key=artifact_keys["code_archive"],
            path=artifact_paths["code_archive"],
        ),
        "final_plan": _put_exact(
            s3,
            bucket=bucket,
            key=artifact_keys["final_plan"],
            path=artifact_paths["final_plan"],
        ),
        "job_script": _put_exact(
            s3,
            bucket=bucket,
            key=artifact_keys["job_script"],
            path=artifact_paths["job_script"],
        ),
        "release": _put_exact(
            s3, bucket=bucket, key=release_key, path=release_path
        ),
    }
    role_created = False
    try:
        role = iam.create_role(
            RoleName=role_name,
            Path="/lumi/",
            AssumeRolePolicyDocument=json.dumps(_trust(), separators=(",", ":")),
            Description=(
                "One bounded Lumi final-80B selection/sharding canary; no "
                "production or model training"
            ),
            Tags=[
                {"Key": "Project", "Value": "Lumi"},
                {"Key": "Purpose", "Value": "FinalCorpusCanary"},
                {"Key": "Authorization", "Value": token[:12]},
            ],
        )["Role"]
        role_created = True
    except iam.exceptions.EntityAlreadyExistsException:
        role = iam.get_role(RoleName=role_name)["Role"]
    policy = _policy(
        account=account,
        region=region,
        bucket=bucket,
        accepted_prefix=accepted_prefix,
        exact_read_keys=exact_read_keys,
        canary_prefix=canary_prefix,
        result_prefix=result_prefix,
        temp_prefix=temp_prefix,
    )
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName="LumiFinalCorpusCanaryBoundedS3",
        PolicyDocument=json.dumps(policy, separators=(",", ":")),
    )
    role_arn = str(role["Arn"])
    request = build_glue_final_corpus_canary_job_definition(
        release,
        authorization,
        role_arn=role_arn,
        release_s3_uri=release_uri,
        authorization_s3_uri=authorization_uri,
    )
    try:
        glue.create_job(**request)
    except glue.exceptions.AlreadyExistsException:
        existing = glue.get_job(JobName=request["Name"])["Job"]
        if glue.get_job_runs(JobName=request["Name"], MaxResults=1).get("JobRuns"):
            raise RuntimeError("authorized final-corpus canary already has a run") from None
        for name in (
            "Command",
            "GlueVersion",
            "MaxRetries",
            "NonOverridableArguments",
            "NumberOfWorkers",
            "Role",
            "Timeout",
            "WorkerType",
        ):
            if existing.get(name) != request.get(name):
                raise ValueError(f"existing final-corpus canary job changed: {name}") from None
    time.sleep(60 if role_created else 30)
    claim = _claim_once(
        s3,
        bucket=bucket,
        key=claim_key,
        claim={
            "authorization_sha256": authorization["authorization_sha256"],
            "job_name": request["Name"],
            "release_fingerprint": release["release_fingerprint"],
            "schema_version": 1,
            "status": "authorization_claimed_for_one_start_attempt",
        },
    )
    for attempt in range(7):
        try:
            run_id = glue.start_job_run(JobName=request["Name"])["JobRunId"]
            break
        except glue.exceptions.InvalidInputException:
            if attempt == 6:
                raise
            time.sleep(5)
    else:  # pragma: no cover
        raise RuntimeError("final-corpus Glue canary did not start")
    launch = {
        **plan,
        "account": account,
        "job_run_id": run_id,
        "region": region,
        "role_arn": role_arn,
        "status": "launched_one_authorized_final_corpus_canary",
        "uploaded": {**uploaded, "launch_claim": claim},
    }
    output.write_text(json.dumps(launch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(launch, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
