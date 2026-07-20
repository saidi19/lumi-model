"""Immutable S3 source staging for Lumi's distributed corpus pipeline.

This module separates the internet-facing source fetcher from the EMR consumer.
The fetcher verifies each exact pinned source object before a conditional S3
write.  The consumer accepts only a content-addressed S3 binding and has no HTTP
fallback.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .glue_canary import _select_canary_sources
from .production_corpus import (
    _canonical_json_bytes,
    _component_provenance,
    _download_source,
    _read_json,
    _scratch_bytes,
    _sha256_file,
    build_round_robin_source_schedule,
)

SCHEMA_VERSION = 1
PLAN_GATE = "lumi_s3_staged_corpus_plan_v0.1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
BUCKET_PATTERN = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]")
NOT_FOUND_CODES = {"404", "NoSuchKey", "NotFound"}
CONDITIONAL_CONFLICT_CODES = {"409", "412", "ConditionalRequestConflict", "PreconditionFailed"}
EXPECTED_AUTHORIZATION = {
    "cloud_mutation": False,
    "full_4b_training": False,
    "model_training": False,
    "paid_compute": False,
    "production_acquisition": False,
}
EXPECTED_CONSUMER = {
    "direct_external_http_fallback": False,
    "network_path": "same_region_s3_only",
    "service": "aws_emr_serverless",
}
EXPECTED_PRODUCER = {
    "direct_source_network_required": True,
    "future_canary_array_size": 8,
    "future_service": "aws_batch_fargate_array",
    "launch_authorized": False,
}
EXPECTED_INPUTS = {
    "baseline_evidence",
    "component_inventory",
    "data_mix",
    "production_authorization",
    "source_file_manifest",
}
EXPECTED_CANARY = {
    "maximum_records_per_source": 2_000,
    "maximum_source_compressed_bytes": 700_000_000,
    "maximum_total_source_compressed_bytes": 1_500_000_000,
    "source_files": 8,
}


def _valid_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _safe_source_file(value: Any) -> str:
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise ValueError("staged source file is invalid")
    parts = Path(value).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("staged source file is unsafe")
    return value


def source_identity(source: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact upstream identity used to bind one staged object."""

    dataset_id = source.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id or "/" not in dataset_id:
        raise ValueError("staged source dataset ID is invalid")
    revision = source.get("revision", source.get("dataset_revision"))
    if not isinstance(revision, str) or REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError("staged source revision must be an exact Git commit")
    ordinal = source.get("ordinal", source.get("source_ordinal"))
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
        raise ValueError("staged source ordinal is invalid")
    category = source.get("category")
    component = source.get("component")
    if not isinstance(category, str) or not category:
        raise ValueError("staged source category is invalid")
    if not isinstance(component, str) or not component:
        raise ValueError("staged source component is invalid")
    return {
        "category": category,
        "component": component,
        "dataset_id": dataset_id,
        "ordinal": ordinal,
        "revision": revision,
        "source_bytes": _positive_int(source.get("source_bytes"), "staged source bytes"),
        "source_file": _safe_source_file(source.get("source_file")),
        "source_sha256": _valid_sha256(
            source.get("source_sha256"), "staged source SHA-256"
        ),
    }


def _identity_fingerprint(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(dict(identity))).hexdigest()


def _checksum_base64(digest: str) -> str:
    return base64.b64encode(bytes.fromhex(digest)).decode("ascii")


def build_staged_object(
    source: Mapping[str, Any], *, bucket: str, key_prefix: str
) -> dict[str, Any]:
    """Create a content-addressed, immutable S3 binding for one source file."""

    if BUCKET_PATTERN.fullmatch(bucket) is None:
        raise ValueError("staged source bucket is invalid")
    if not isinstance(key_prefix, str) or not key_prefix or key_prefix.startswith("/"):
        raise ValueError("staged source key prefix is invalid")
    if "//" in key_prefix or any(part in {"", ".", ".."} for part in Path(key_prefix).parts):
        raise ValueError("staged source key prefix is unsafe")
    identity = source_identity(source)
    digest = identity["source_sha256"]
    key = f"{key_prefix.rstrip('/')}/{digest[:2]}/{digest}.json.gz"
    return {
        "bucket": bucket,
        "bytes": identity["source_bytes"],
        "checksum_sha256_base64": _checksum_base64(digest),
        "content_type": "application/gzip",
        "identity_fingerprint": _identity_fingerprint(identity),
        "key": key,
        "s3_uri": f"s3://{bucket}/{key}",
        "server_side_encryption": "AES256",
        "sha256": digest,
        "write_condition": "IfNoneMatch:*",
    }


def _validate_object_binding(
    source: Mapping[str, Any], staged_object: Mapping[str, Any]
) -> dict[str, Any]:
    identity = source_identity(source)
    digest = identity["source_sha256"]
    bucket = staged_object.get("bucket")
    key = staged_object.get("key")
    if not isinstance(bucket, str) or BUCKET_PATTERN.fullmatch(bucket) is None:
        raise ValueError("staged S3 bucket binding is invalid")
    if not isinstance(key, str) or not key.endswith(
        f"/sha256/{digest[:2]}/{digest}.json.gz"
    ):
        raise ValueError("staged S3 key is not content-addressed to the source")
    expected = {
        "bucket": bucket,
        "bytes": identity["source_bytes"],
        "checksum_sha256_base64": _checksum_base64(digest),
        "content_type": "application/gzip",
        "identity_fingerprint": _identity_fingerprint(identity),
        "key": key,
        "s3_uri": f"s3://{bucket}/{key}",
        "server_side_encryption": "AES256",
        "sha256": digest,
        "write_condition": "IfNoneMatch:*",
    }
    if dict(staged_object) != expected:
        raise ValueError("staged S3 object binding changed")
    return expected


def validate_staged_consumer_source(source: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one EMR row and return its exact staged-object binding."""

    staged_object = source.get("staged_object")
    if not isinstance(staged_object, Mapping):
        raise ValueError("S3-only source is missing its staged object binding")
    return _validate_object_binding(source, staged_object)


def _object_metadata(source: Mapping[str, Any]) -> dict[str, str]:
    identity = source_identity(source)
    return {
        "lumi-schema-version": str(SCHEMA_VERSION),
        "lumi-source-bytes": str(identity["source_bytes"]),
        "lumi-source-identity": _identity_fingerprint(identity),
        "lumi-source-sha256": identity["source_sha256"],
    }


def _aws_error_code(error: BaseException) -> str | None:
    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return None
    details = response.get("Error")
    if not isinstance(details, Mapping):
        return None
    code = details.get("Code")
    return str(code) if code is not None else None


def _head_or_none(client: Any, staged_object: Mapping[str, Any]) -> Mapping[str, Any] | None:
    try:
        return client.head_object(
            Bucket=staged_object["bucket"],
            Key=staged_object["key"],
            ChecksumMode="ENABLED",
        )
    except Exception as error:
        if _aws_error_code(error) in NOT_FOUND_CODES:
            return None
        raise


def verify_staged_object_head(
    source: Mapping[str, Any], staged_object: Mapping[str, Any], head: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify S3's stored checksum, metadata, encryption, size, and content type."""

    expected = _validate_object_binding(source, staged_object)
    if head.get("ContentLength") != expected["bytes"]:
        raise ValueError("staged S3 object byte size changed")
    if head.get("ChecksumSHA256") != expected["checksum_sha256_base64"]:
        raise ValueError("staged S3 object SHA-256 checksum changed")
    if head.get("ContentType") != expected["content_type"]:
        raise ValueError("staged S3 object content type changed")
    if head.get("ServerSideEncryption") != expected["server_side_encryption"]:
        raise ValueError("staged S3 object encryption changed")
    metadata = head.get("Metadata")
    if not isinstance(metadata, Mapping) or dict(metadata) != _object_metadata(source):
        raise ValueError("staged S3 object identity metadata changed")
    return {
        "bytes": expected["bytes"],
        "etag": head.get("ETag"),
        "sha256": expected["sha256"],
        "version_id": head.get("VersionId"),
    }


def stage_source_file(
    entry: Mapping[str, Any], source_path: Path, *, s3_client: Any
) -> dict[str, Any]:
    """Verify and conditionally stage one source file without allowing overwrite."""

    source = entry.get("source")
    staged_object = entry.get("object")
    if not isinstance(source, Mapping) or not isinstance(staged_object, Mapping):
        raise ValueError("staging entry is incomplete")
    expected = _validate_object_binding(source, staged_object)
    path = source_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing pinned source payload: {path}")
    if path.stat().st_size != expected["bytes"] or _sha256_file(path) != expected["sha256"]:
        raise ValueError("local pinned source payload failed size or SHA-256")
    existing = _head_or_none(s3_client, staged_object)
    if existing is not None:
        verified = verify_staged_object_head(source, staged_object, existing)
        return {**verified, "s3_uri": expected["s3_uri"], "status": "reused_verified"}
    try:
        with path.open("rb") as body:
            s3_client.put_object(
                Bucket=expected["bucket"],
                Key=expected["key"],
                Body=body,
                ChecksumAlgorithm="SHA256",
                ChecksumSHA256=expected["checksum_sha256_base64"],
                ContentLength=expected["bytes"],
                ContentType=expected["content_type"],
                IfNoneMatch="*",
                Metadata=_object_metadata(source),
                ServerSideEncryption=expected["server_side_encryption"],
            )
        status = "staged_new"
    except Exception as error:
        if _aws_error_code(error) not in CONDITIONAL_CONFLICT_CODES:
            raise
        status = "reused_after_conditional_race"
    head = _head_or_none(s3_client, staged_object)
    if head is None:
        raise RuntimeError("staged S3 object is absent after conditional write")
    verified = verify_staged_object_head(source, staged_object, head)
    return {**verified, "s3_uri": expected["s3_uri"], "status": status}


def fetch_and_stage_source(
    entry: Mapping[str, Any],
    scratch: Path,
    *,
    scratch_cap: int,
    s3_client: Any,
) -> dict[str, Any]:
    """Internet-facing producer path; this function is never used by EMR."""

    source = entry.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("staging entry has no source identity")
    source_path = _download_source(source, scratch, scratch_cap)
    return stage_source_file(entry, source_path, s3_client=s3_client)


def _body_chunks(body: Any, chunk_size: int) -> Iterable[bytes]:
    iterator = getattr(body, "iter_chunks", None)
    if callable(iterator):
        for chunk in iterator(chunk_size=chunk_size):
            if chunk:
                yield bytes(chunk)
        return
    while True:
        chunk = body.read(chunk_size)
        if not chunk:
            return
        yield bytes(chunk)


def download_staged_source(
    source: Mapping[str, Any],
    scratch: Path,
    *,
    scratch_cap: int,
    s3_client: Any,
    chunk_size: int = 8 * 1024 * 1024,
) -> Path:
    """Materialize one verified S3 object with byte-range resume and no HTTP fallback."""

    staged_object = source.get("staged_object")
    expected = validate_staged_consumer_source(source)
    assert isinstance(staged_object, Mapping)
    head = _head_or_none(s3_client, staged_object)
    if head is None:
        raise FileNotFoundError(f"staged source object is missing: {expected['s3_uri']}")
    verify_staged_object_head(source, staged_object, head)
    scratch.mkdir(parents=True, exist_ok=True)
    final = scratch / f"source-{expected['sha256']}.json.gz"
    partial = scratch / f"source-{expected['sha256']}.json.gz.part"
    if final.exists():
        if final.stat().st_size != expected["bytes"] or _sha256_file(final) != expected["sha256"]:
            raise ValueError("cached staged source failed size or SHA-256")
        return final
    partial_bytes = partial.stat().st_size if partial.exists() else 0
    if partial_bytes > expected["bytes"]:
        raise ValueError("partial staged source is larger than its pinned object")
    remaining = expected["bytes"] - partial_bytes
    if _scratch_bytes(scratch) + remaining > scratch_cap:
        raise RuntimeError("S3-only consumer scratch cap would be exceeded")
    digest = hashlib.sha256()
    if partial_bytes:
        with partial.open("rb") as existing:
            for block in iter(lambda: existing.read(chunk_size), b""):
                digest.update(block)
    if partial_bytes == expected["bytes"]:
        if digest.hexdigest() != expected["sha256"]:
            partial.unlink()
            raise ValueError("complete staged-source partial failed SHA-256")
        partial.replace(final)
        return final
    request: dict[str, Any] = {
        "Bucket": expected["bucket"],
        "Key": expected["key"],
        "ChecksumMode": "ENABLED",
    }
    if partial_bytes:
        request["Range"] = f"bytes={partial_bytes}-"
    response = s3_client.get_object(**request)
    if partial_bytes:
        content_range = response.get("ContentRange")
        if not isinstance(content_range, str) or not content_range.startswith(
            f"bytes {partial_bytes}-"
        ):
            raise ValueError("S3 byte-range resume response changed")
    body = response.get("Body")
    if body is None:
        raise RuntimeError("S3 get-object response has no body")
    mode = "ab" if partial_bytes else "xb"
    written = partial_bytes
    try:
        with partial.open(mode) as handle:
            for chunk in _body_chunks(body, chunk_size):
                written += len(chunk)
                if written > expected["bytes"]:
                    raise ValueError("S3 object exceeded its pinned byte size")
                handle.write(chunk)
                digest.update(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()
    if written != expected["bytes"]:
        raise RuntimeError("S3 source download ended before the pinned byte size")
    if digest.hexdigest() != expected["sha256"]:
        partial.unlink(missing_ok=True)
        raise ValueError("downloaded staged source failed SHA-256")
    partial.replace(final)
    return final


def bind_staged_sources(
    sources: Iterable[Mapping[str, Any]], plan: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Attach only verified content-addressed S3 bindings to EMR source rows."""

    validate_s3_staging_plan(plan)
    entries = {
        int(entry["source"]["ordinal"]): entry for entry in plan["sources"]
    }
    bound = []
    for source in sources:
        identity = source_identity(source)
        entry = entries.get(identity["ordinal"])
        if entry is None or entry["source"] != identity:
            raise ValueError("S3 staging plan does not match the EMR source identity")
        staged_object = _validate_object_binding(identity, entry["object"])
        row = dict(source)
        row["staged_object"] = staged_object
        bound.append(row)
    if len(bound) != len(entries):
        raise ValueError("S3 staging plan contains unbound source objects")
    return bound


def _load_contract(
    contract_path: Path, repository_root: Path
) -> tuple[dict[str, Any], dict[str, Path], dict[str, dict[str, str]]]:
    root = repository_root.resolve()
    contract = _read_json(contract_path.resolve())
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported S3 staged-corpus contract schema")
    if contract.get("gate") != "s3_staged_corpus_transport_v0.1":
        raise ValueError("unexpected S3 staged-corpus gate")
    if contract.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("S3 staging plan must not authorize cloud mutation or paid work")
    if contract.get("consumer") != EXPECTED_CONSUMER:
        raise ValueError("S3 staging consumer must be same-region S3-only")
    if contract.get("producer") != EXPECTED_PRODUCER:
        raise ValueError("S3 staging producer boundary changed")
    if contract.get("canary") != EXPECTED_CANARY:
        raise ValueError("S3 staging canary boundary changed")
    pins = contract.get("inputs")
    if not isinstance(pins, Mapping) or set(pins) != EXPECTED_INPUTS:
        raise ValueError("S3 staging input set changed")
    paths: dict[str, Path] = {}
    observed: dict[str, dict[str, str]] = {}
    for name in sorted(EXPECTED_INPUTS):
        pin = pins[name]
        if not isinstance(pin, Mapping):
            raise ValueError(f"invalid S3 staging input pin: {name}")
        raw_path = pin.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"invalid S3 staging input path: {name}")
        path = (root / raw_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(f"missing S3 staging input: {name}")
        digest = _sha256_file(path)
        if digest != _valid_sha256(pin.get("sha256"), f"{name} SHA-256"):
            raise ValueError(f"S3 staging input hash changed: {name}")
        paths[name] = path
        observed[name] = {"path": raw_path, "sha256": digest}
    return contract, paths, observed


def build_s3_staging_plan(
    contract_path: Path, *, repository_root: Path
) -> dict[str, Any]:
    """Build the deterministic eight-source S3-staged parity plan locally."""

    contract, paths, inputs = _load_contract(contract_path, repository_root)
    baseline = _read_json(paths["baseline_evidence"])
    source_manifest = _read_json(paths["source_file_manifest"])
    mixture = _read_json(paths["data_mix"])
    production = _read_json(paths["production_authorization"])
    inventory = _read_json(paths["component_inventory"])
    object_contract = contract.get("object_contract")
    expected_object_contract = {
        "checksum_algorithm": "SHA256",
        "content_type": "application/gzip",
        "key_prefix": f"{baseline['namespace']}/raw/source-objects/sha256",
        "maximum_single_put_bytes": 5_000_000_000,
        "maximum_source_bytes": 2_704_144_015,
        "raw_retention": "transient_until_processed_output_is_verified",
        "server_side_encryption": "AES256",
        "write_condition": "IfNoneMatch:*",
    }
    if object_contract != expected_object_contract:
        raise ValueError("S3 staged-object contract changed")
    if production.get("provider", {}).get("region") != "us-east-1":
        raise ValueError("S3 staged corpus must remain in us-east-1")
    if production.get("provider", {}).get("namespace") != baseline.get("namespace"):
        raise ValueError("S3 staged-corpus namespace changed")
    files = [
        file
        for source in source_manifest.get("sources", [])
        for file in source.get("data_files", [])
    ]
    if max(int(file["source_bytes"]) for file in files) != object_contract[
        "maximum_source_bytes"
    ]:
        raise ValueError("maximum pinned source size changed")
    if any(
        int(file["source_bytes"]) > object_contract["maximum_single_put_bytes"]
        for file in files
    ):
        raise ValueError("a pinned source exceeds the immutable single-PUT boundary")
    schedule, _ = build_round_robin_source_schedule(source_manifest, mixture)
    selected = _select_canary_sources(schedule, contract["canary"], baseline["partial_prefix"])
    provenance = _component_provenance(inventory)
    sources = []
    for source in selected:
        identity = source_identity(source)
        start_line = 1
        if identity["ordinal"] == baseline["partial_prefix"]["source_ordinal"]:
            start_line = int(baseline["partial_prefix"]["source_next_line"]) + 1
        sources.append(
            {
                "object": build_staged_object(
                    source,
                    bucket=baseline["bucket"],
                    key_prefix=object_contract["key_prefix"],
                ),
                "source": identity,
                "source_provenance": provenance[identity["component"]],
                "start_line": start_line,
                "maximum_records": int(contract["canary"]["maximum_records_per_source"]),
            }
        )
    material = {
        "authorization": dict(contract["authorization"]),
        "consumer": dict(contract["consumer"]),
        "gate": PLAN_GATE,
        "inputs": inputs,
        "object_contract": dict(object_contract),
        "producer": dict(contract["producer"]),
        "schema_version": SCHEMA_VERSION,
        "sources": sources,
        "summary": {
            "all_categories_covered": len({row["source"]["category"] for row in sources})
            == 7,
            "bucket": baseline["bucket"],
            "direct_external_http_in_consumer": False,
            "maximum_records_per_source": int(
                contract["canary"]["maximum_records_per_source"]
            ),
            "source_bytes": sum(row["source"]["source_bytes"] for row in sources),
            "source_files": len(sources),
        },
    }
    fingerprint = hashlib.sha256(_canonical_json_bytes(material)).hexdigest()
    plan = {
        **material,
        "decision": {
            "cloud_canary": "not_authorized",
            "paid_compute": "not_authorized",
            "production_acquisition": "not_authorized",
            "status": "verified_local_s3_staging_plan_only",
        },
        "plan_fingerprint": fingerprint,
    }
    validate_s3_staging_plan(plan)
    return plan


def validate_s3_staging_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if plan.get("schema_version") != SCHEMA_VERSION or plan.get("gate") != PLAN_GATE:
        raise ValueError("unsupported S3 staging plan")
    material = {
        key: value
        for key, value in plan.items()
        if key not in {"decision", "plan_fingerprint"}
    }
    fingerprint = hashlib.sha256(_canonical_json_bytes(material)).hexdigest()
    if fingerprint != plan.get("plan_fingerprint"):
        raise ValueError("S3 staging plan fingerprint changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("S3 staging plan unexpectedly authorizes paid work")
    if plan.get("consumer") != EXPECTED_CONSUMER:
        raise ValueError("S3 staging plan consumer changed")
    if plan.get("producer") != EXPECTED_PRODUCER:
        raise ValueError("S3 staging plan producer changed")
    if plan.get("decision") != {
        "cloud_canary": "not_authorized",
        "paid_compute": "not_authorized",
        "production_acquisition": "not_authorized",
        "status": "verified_local_s3_staging_plan_only",
    }:
        raise ValueError("S3 staging decision boundary changed")
    sources = plan.get("sources")
    if not isinstance(sources, list) or len(sources) != 8:
        raise ValueError("S3 staging canary must contain exactly eight source files")
    ordinals = set()
    for entry in sources:
        if not isinstance(entry, Mapping):
            raise ValueError("S3 staging source entry is invalid")
        source = entry.get("source")
        staged_object = entry.get("object")
        if not isinstance(source, Mapping) or not isinstance(staged_object, Mapping):
            raise ValueError("S3 staging source entry is incomplete")
        identity = source_identity(source)
        _validate_object_binding(identity, staged_object)
        if identity["ordinal"] in ordinals:
            raise ValueError("S3 staging source ordinal is duplicated")
        ordinals.add(identity["ordinal"])
        if entry.get("maximum_records") != 2_000:
            raise ValueError("S3 staging record cap changed")
        if not isinstance(entry.get("source_provenance"), Mapping):
            raise ValueError("S3 staging source provenance is missing")
        if not isinstance(entry.get("start_line"), int) or entry["start_line"] < 1:
            raise ValueError("S3 staging source start line is invalid")
    summary = plan.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("S3 staging plan summary is missing")
    if summary.get("source_files") != 8 or summary.get("all_categories_covered") is not True:
        raise ValueError("S3 staging source/category coverage changed")
    if summary.get("direct_external_http_in_consumer") is not False:
        raise ValueError("S3 staging consumer reintroduced external HTTP")
    if summary.get("source_bytes") != sum(
        int(entry["source"]["source_bytes"]) for entry in sources
    ):
        raise ValueError("S3 staging source byte accounting changed")
    return {
        "plan_fingerprint": fingerprint,
        "source_bytes": summary["source_bytes"],
        "source_files": 8,
        "status": "verified_local_no_cloud_mutation",
    }


def write_s3_staging_plan(path: Path, plan: Mapping[str, Any]) -> None:
    validate_s3_staging_plan(plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
