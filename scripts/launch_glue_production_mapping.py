#!/usr/bin/env python3
"""Upload and launch one exactly authorized 512-DPU Glue mapping run."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lumi.glue_production_mapping import (
    CODE_ARCHIVE,
    JOB_SCRIPT,
    RELEASE_FILE,
    validate_glue_production_mapping_authorization,
    validate_glue_production_mapping_release,
    validate_mapping_checkpoint,
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
            raise ValueError(f"existing mapping bootstrap size changed: {key}") from error
        if head.get("Metadata", {}).get("sha256") != digest:
            raise ValueError(f"existing mapping bootstrap hash changed: {key}") from error
        version_id = head.get("VersionId")
    else:
        version_id = response.get("VersionId")
    return {
        "bytes": size,
        "key": key,
        "sha256": digest,
        "version_id": None if version_id is None else str(version_id),
    }


def _verify_package(package: Path) -> None:
    manifest = package / "manifest.sha256"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = package / name
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"Glue mapping package manifest mismatch: {name}")


def _policy(
    *,
    account: str,
    region: str,
    bucket: str,
    exact_read_keys: list[str],
    staged_prefix: str,
    checkpoint_prefix: str,
    mapped_prefix: str,
    result_prefix: str,
    temp_prefix: str,
) -> dict[str, Any]:
    bucket_arn = f"arn:aws:s3:::{bucket}"
    read_prefixes = [staged_prefix]
    write_prefixes = [checkpoint_prefix, mapped_prefix, result_prefix, temp_prefix]
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "BucketLocation",
                "Effect": "Allow",
                "Action": ["s3:GetBucketLocation", "s3:ListBucketMultipartUploads"],
                "Resource": bucket_arn,
            },
            {
                "Sid": "ListBoundedMappingPrefixes",
                "Effect": "Allow",
                "Action": "s3:ListBucket",
                "Resource": bucket_arn,
                "Condition": {
                    "StringLike": {
                        "s3:prefix": [
                            *exact_read_keys,
                            *[f"{prefix}/*" for prefix in read_prefixes],
                            *[f"{prefix}/*" for prefix in write_prefixes],
                        ]
                    }
                },
            },
            {
                "Sid": "ReadExactMappingBootstrap",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:GetObjectVersion"],
                "Resource": [f"{bucket_arn}/{key}" for key in exact_read_keys],
            },
            {
                "Sid": "ReadSealedProductionStage",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:GetObjectVersion"],
                "Resource": f"{bucket_arn}/{staged_prefix}/*",
            },
            {
                "Sid": "ReadWriteBoundedMapping",
                "Effect": "Allow",
                "Action": [
                    "s3:AbortMultipartUpload",
                    "s3:GetObject",
                    "s3:GetObjectVersion",
                    "s3:ListMultipartUploadParts",
                    "s3:PutObject",
                ],
                "Resource": [
                    f"{bucket_arn}/{checkpoint_prefix}/*",
                    f"{bucket_arn}/{mapped_prefix}/*",
                    f"{bucket_arn}/{result_prefix}/*",
                ],
            },
            {
                "Sid": "ReadWriteDeleteMappingTempOnly",
                "Effect": "Allow",
                "Action": [
                    "s3:AbortMultipartUpload",
                    "s3:DeleteObject",
                    "s3:GetObject",
                    "s3:GetObjectVersion",
                    "s3:ListMultipartUploadParts",
                    "s3:PutObject",
                ],
                "Resource": f"{bucket_arn}/{temp_prefix}/*",
            },
            {
                "Sid": "GlueLogs",
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                "Resource": f"arn:aws:logs:{region}:{account}:log-group:/aws-glue/jobs/*",
            },
            {
                "Sid": "GlueMetrics",
                "Effect": "Allow",
                "Action": "cloudwatch:PutMetricData",
                "Resource": "*",
                "Condition": {"StringEquals": {"cloudwatch:namespace": "Glue"}},
            },
        ],
    }


def _trust() -> dict[str, Any]:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "glue.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--evidence-directory", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _args()
    package = args.package.resolve()
    _verify_package(package)
    release_path = package / RELEASE_FILE
    checkpoint_path = (
        package / "initial-map-checkpoint-v0.1.json"
        if args.checkpoint is None
        else args.checkpoint.resolve()
    )
    release = _read_json(release_path)
    checkpoint = _read_json(checkpoint_path)
    authorization = _read_json(args.authorization)
    validate_glue_production_mapping_release(release)
    validate_mapping_checkpoint(checkpoint, release)
    validate_glue_production_mapping_authorization(
        authorization, release, checkpoint=checkpoint
    )
    if checkpoint["checkpoint_sha256"] != authorization["bounds"][
        "starting_checkpoint_sha256"
    ]:
        raise ValueError("mapping launcher checkpoint differs from authorization")
    region = str(release["region"])
    bucket = str(release["storage"]["bucket"])
    fingerprint = str(release["release_fingerprint"])
    short = fingerprint[:12]
    checkpoint_short = str(checkpoint["checkpoint_sha256"])[:8]
    bootstrap = (
        f"{release['storage']['namespace']}/control/distributed/"
        "glue-production/mapping/bootstrap"
    )
    release_key = f"{bootstrap}/releases/{_sha256(release_path)}.json"
    authorization_key = f"{bootstrap}/authorizations/{_sha256(args.authorization)}.json"
    checkpoint_key = f"{bootstrap}/checkpoints/{_sha256(checkpoint_path)}.json"
    code_key = str(release["artifacts"]["code_archive"]["s3_uri"]).split(
        f"s3://{bucket}/", 1
    )[1]
    script_key = str(release["artifacts"]["job_script"]["s3_uri"]).split(
        f"s3://{bucket}/", 1
    )[1]
    exact_artifact_keys = [
        str(release["artifacts"][name]["s3_uri"]).split(f"s3://{bucket}/", 1)[1]
        for name in (
            "benchmark_registry",
            "frozen_tokenizer",
            "training_exclusion_ledger",
        )
    ]
    staged_prefix = next(iter(release["staged_sources"]))["manifest_s3_key"].split(
        "/manifests/", 1
    )[0]
    checkpoint_prefix = f"{release['output']['checkpoint_prefix']}/{fingerprint}"
    mapped_prefix = f"{release['output']['mapped_prefix']}/{fingerprint}"
    result_prefix = f"{release['output']['result_prefix']}/{fingerprint}"
    temp_prefix = f"{release['output']['temp_prefix']}/{fingerprint}"
    role_name = f"lumi-glue-map-{short}-{checkpoint_short}"
    job_name = f"lumi-glue-map-{short}-{checkpoint_short}"
    plan = {
        "authorization_key": authorization_key,
        "bucket": bucket,
        "checkpoint_key": checkpoint_key,
        "hard_maximum_dpu": release["job"]["hard_maximum_dpu"],
        "job_name": job_name,
        "maximum_compute_usd": release["limits"]["maximum_compute_usd"],
        "number_of_workers": release["job"]["number_of_workers"],
        "release_key": release_key,
        "role_name": role_name,
        "starting_checkpoint_sequence": checkpoint["checkpoint_sequence"],
        "starting_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "target_total_candidate_tokens": release["limits"][
            "target_total_candidate_tokens"
        ],
        "timeout_minutes": release["job"]["timeout_minutes"],
        "worker_type": release["job"]["worker_type"],
    }
    if not args.execute:
        print(json.dumps({**plan, "status": "dry_run_not_launched"}, indent=2, sort_keys=True))
        return

    import boto3

    session = boto3.Session(region_name=region)
    sts = session.client("sts")
    s3 = session.client("s3")
    iam = session.client("iam")
    glue = session.client("glue")
    account = str(sts.get_caller_identity()["Account"])
    uploaded = {
        "authorization": _put_exact(
            s3, bucket=bucket, key=authorization_key, path=args.authorization
        ),
        "checkpoint": _put_exact(
            s3, bucket=bucket, key=checkpoint_key, path=checkpoint_path
        ),
        "code": _put_exact(s3, bucket=bucket, key=code_key, path=package / CODE_ARCHIVE),
        "release": _put_exact(s3, bucket=bucket, key=release_key, path=release_path),
        "script": _put_exact(s3, bucket=bucket, key=script_key, path=package / JOB_SCRIPT),
    }
    exact_read_keys = [
        authorization_key,
        checkpoint_key,
        code_key,
        release_key,
        script_key,
        *exact_artifact_keys,
    ]
    try:
        role = iam.create_role(
            RoleName=role_name,
            Path="/lumi/",
            AssumeRolePolicyDocument=json.dumps(_trust(), separators=(",", ":")),
            Description=(
                "One authorized Lumi 512-DPU candidate mapping run; no dedup, "
                "sharding, GPU, or model training"
            ),
            Tags=[
                {"Key": "Project", "Value": "Lumi"},
                {"Key": "Purpose", "Value": "ProductionCorpusMapping"},
                {"Key": "Release", "Value": short},
            ],
        )["Role"]
    except iam.exceptions.EntityAlreadyExistsException:
        role = iam.get_role(RoleName=role_name)["Role"]
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName="LumiGlueProductionMappingBoundedS3",
        PolicyDocument=json.dumps(
            _policy(
                account=account,
                region=region,
                bucket=bucket,
                exact_read_keys=exact_read_keys,
                staged_prefix=staged_prefix,
                checkpoint_prefix=checkpoint_prefix,
                mapped_prefix=mapped_prefix,
                result_prefix=result_prefix,
                temp_prefix=temp_prefix,
            ),
            separators=(",", ":"),
        ),
    )
    role_arn = str(role["Arn"])
    extra_files = [
        f"s3://{bucket}/{release_key}",
        f"s3://{bucket}/{authorization_key}",
        f"s3://{bucket}/{checkpoint_key}",
        *[f"s3://{bucket}/{key}" for key in exact_artifact_keys],
    ]
    non_overridable = {
        "--LUMI_AUTHORIZATION_FILE": Path(authorization_key).name,
        "--LUMI_CHECKPOINT_FILE": Path(checkpoint_key).name,
        "--LUMI_RELEASE_FILE": Path(release_key).name,
        "--TempDir": f"s3://{bucket}/{temp_prefix}/",
        "--additional-python-modules": release["job"]["additional_python_modules"],
        "--enable-auto-scaling": "false",
        "--extra-files": ",".join(extra_files),
        "--extra-py-files": f"s3://{bucket}/{code_key}",
        "--job-language": "python",
    }
    request = {
        "Name": job_name,
        "Description": (
            "Checkpointed Lumi 150B candidate mapping at 512 DPU; no dedup, "
            "sharding, GPU, or model training"
        ),
        "Role": role_arn,
        "ExecutionProperty": {"MaxConcurrentRuns": 1},
        "Command": {
            "Name": "glueetl",
            "ScriptLocation": f"s3://{bucket}/{script_key}",
            "PythonVersion": "3",
        },
        "DefaultArguments": {
            "--enable-continuous-cloudwatch-log": "true",
            "--enable-metrics": "true",
            "--enable-observability-metrics": "true",
        },
        "NonOverridableArguments": non_overridable,
        "GlueVersion": release["job"]["glue_version"],
        "MaxRetries": 0,
        "Timeout": int(release["job"]["timeout_minutes"]),
        "WorkerType": release["job"]["worker_type"],
        "NumberOfWorkers": int(release["job"]["number_of_workers"]),
        "ExecutionClass": release["job"]["execution_class"],
        "Tags": {
            "Project": "Lumi",
            "Purpose": "ProductionCorpusMapping",
            "Release": short,
        },
    }
    try:
        glue.create_job(**request)
    except glue.exceptions.AlreadyExistsException:
        existing = glue.get_job(JobName=job_name)["Job"]
        if glue.get_job_runs(JobName=job_name, MaxResults=1).get("JobRuns"):
            raise RuntimeError("authorized production mapping job already has a run") from None
        for key in (
            "Command",
            "GlueVersion",
            "MaxRetries",
            "NonOverridableArguments",
            "NumberOfWorkers",
            "Role",
            "Timeout",
            "WorkerType",
        ):
            if existing.get(key) != request.get(key):
                raise ValueError(f"existing production mapping job changed: {key}") from None
    for attempt in range(6):
        try:
            run_id = glue.start_job_run(JobName=job_name)["JobRunId"]
            break
        except glue.exceptions.InvalidInputException:
            if attempt == 5:
                raise
            time.sleep(5)
    else:  # pragma: no cover
        raise RuntimeError("production mapping Glue job did not start")
    launch = {
        **plan,
        "account": account,
        "authorization_sha256": authorization["authorization_sha256"],
        "job_run_id": run_id,
        "region": region,
        "release_fingerprint": fingerprint,
        "role_arn": role_arn,
        "status": "launched_one_authorized_512_dpu_mapping_run",
        "uploaded": uploaded,
    }
    args.evidence_directory.mkdir(parents=True, exist_ok=True)
    output = args.evidence_directory / "launch-record.json"
    output.write_text(json.dumps(launch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(launch, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
