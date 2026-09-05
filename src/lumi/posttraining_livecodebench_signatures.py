"""Bounded private LiveCodeBench prompt-signature acquisition for Lumi L1 research."""

from __future__ import annotations

import json
import os
import re
import ssl
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .data_quality import benchmark_signature_entry, sha256_file
from .posttraining_bigcodebench_signatures import (
    records_sha256,
    sha256_value,
)

PLAN_SCHEMA_VERSION = "lumi.livecodebench-research-signature-acquisition.v0.1"
REPORT_SCHEMA_VERSION = "lumi.livecodebench-research-signature-acquisition-report.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.livecodebench-research-signature-acquisition-verification.v0.1"
)

OWNER_AUTHORIZATION_STATEMENT = (
    "I approve the exact CPU-only private research/evaluation-only LiveCodeBench "
    "acquisition described by plan SHA-256 "
    "0bd16693c2a520d8e064bf1956d37c2bca8ad0be43fe3e4199c3bebf795b1dab, "
    "with its stated 160 MiB network / 256 MiB decoded / 1,055 rows / 3,165 "
    "signatures / 90-minute limits and all listed prohibitions."
)
RESEARCH_PLAN_SHA256 = (
    "0bd16693c2a520d8e064bf1956d37c2bca8ad0be43fe3e4199c3bebf795b1dab"
)
SOURCE_DATASET_ID = "livecodebench/code_generation_lite"
SOURCE_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
SOURCE_OBJECT = "test6.jsonl"
SOURCE_BYTES = 134_303_240
SOURCE_SHA256 = "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5"
SOURCE_URL = (
    "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/"
    f"{SOURCE_REVISION}/{SOURCE_OBJECT}?download=true"
)
SOURCE_ROWS = 175
MAXIMUM_SOURCE_ROWS = 1_055
SOURCE_COLUMNS = (
    "question_id",
    "platform",
    "question_title",
    "question_content",
    "starter_code",
)
PROMPT_VARIANTS = ("question_title", "question_content", "full_prompt")
PROHIBITED_FIELDS = (
    "public_test_cases",
    "private_test_cases",
    "metadata",
    "reference_solution",
    "solution",
)
PREDECESSOR_ENTRIES = 50_419
EXPECTED_SIGNATURES = 525
MAXIMUM_SIGNATURES = 3_165
ANCHOR_NGRAM_TOKENS = 8
MAXIMUM_ANCHORS = 2
MINIMUM_ANCHOR_MATCHES = 2

EXPECTED_AUTHORIZATION = {
    "active_registry_mutation_authorized": False,
    "candidate_registry_promotion_authorized": False,
    "cpu_only_private_prompt_signature_acquisition_authorized": True,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "gpu_authorized": False,
    "network_access_authorized": True,
    "rights_clearance_claimed": False,
    "sft_training_authorized": False,
    "training_ingestion_authorized": False,
    "unrelated_data_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_decoded_bytes": 268_435_456,
    "maximum_network_bytes": 167_772_160,
    "maximum_prompt_signatures": MAXIMUM_SIGNATURES,
    "maximum_runtime_minutes": 90,
    "maximum_source_objects": 1,
    "maximum_source_rows": MAXIMUM_SOURCE_ROWS,
}
EXPECTED_BINDINGS = {
    "data_quality_implementation",
    "implementation",
    "metadata_preflight_report",
    "metadata_preflight_verification",
    "predecessor_data_quality",
    "predecessor_registry",
    "research_plan",
    "runner",
    "verifier",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def resolve_binding(
    repository_root: str | Path,
    specification: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    root = Path(repository_root).resolve()
    relative = specification.get("path")
    expected = _require_sha256(specification.get("sha256"), f"bindings.{name}.sha256")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"invalid LiveCodeBench binding: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.parent.resolve()) or not resolved.is_file():
        raise ValueError(f"LiveCodeBench binding missing or out of scope: {name}")
    if sha256_file(resolved) != expected:
        raise ValueError(f"LiveCodeBench binding changed: {name}")
    return resolved


def load_acquisition_authorization(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _read_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported LiveCodeBench acquisition authorization")
    if plan.get("status") != "authorized_private_research_prompt_signature_acquisition":
        raise ValueError("LiveCodeBench acquisition status changed")
    if plan.get("observed_at") != "2026-08-07":
        raise ValueError("LiveCodeBench acquisition date changed")
    if plan.get("owner_authorization_statement") != OWNER_AUTHORIZATION_STATEMENT:
        raise ValueError("LiveCodeBench owner authorization changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("LiveCodeBench authorization boundary changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("LiveCodeBench acquisition limits changed")
    if plan.get("runtime") != {
        "ca_bundle": "certifi==2026.7.22",
        "json_projection": "pyarrow==25.0.0",
        "unexpected_field_behavior": "ignore",
    }:
        raise ValueError("LiveCodeBench projection runtime changed")
    if plan.get("source") != {
        "allowed_columns": list(SOURCE_COLUMNS),
        "dataset_id": SOURCE_DATASET_ID,
        "declared_license_id": "cc_non_spdx_ambiguous",
        "object": {
            "path": SOURCE_OBJECT,
            "sha256": SOURCE_SHA256,
            "size_bytes": SOURCE_BYTES,
            "url": SOURCE_URL,
        },
        "prohibited_value_columns": list(PROHIBITED_FIELDS),
        "revision": SOURCE_REVISION,
        "split": "release_v6",
    }:
        raise ValueError("LiveCodeBench source contract changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("LiveCodeBench binding inventory changed")
    for name, specification in bindings.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"invalid LiveCodeBench binding object: {name}")
        resolve_binding(root, specification, name=name)
    if bindings["research_plan"]["sha256"] != RESEARCH_PLAN_SHA256:
        raise ValueError("LiveCodeBench research plan identity changed")
    research_plan = _read_json(
        resolve_binding(root, bindings["research_plan"], name="research_plan")
    )
    if research_plan.get("decision", {}).get("raw_examples_must_never_enter_sft_training"):
        pass
    else:
        raise ValueError("LiveCodeBench research-only boundary changed")
    if research_plan.get("source", {}).get("object", {}).get("sha256") != SOURCE_SHA256:
        raise ValueError("LiveCodeBench research-plan object changed")
    output = plan.get("output")
    if not isinstance(output, Mapping) or set(output) != {
        "candidate_registry_path",
        "private_root",
        "raw_benchmark_text_in_git",
        "report_path",
        "verification_path",
    }:
        raise ValueError("LiveCodeBench output inventory changed")
    if output.get("raw_benchmark_text_in_git") is not False:
        raise ValueError("LiveCodeBench raw benchmark text cannot enter Git")
    private_root = (root / str(output["private_root"])).resolve()
    if not private_root.is_relative_to(root.parent.resolve()) or private_root.is_relative_to(
        root
    ):
        raise ValueError("LiveCodeBench private root must stay outside the repository")
    candidate = (root / str(output["candidate_registry_path"])).resolve()
    if not candidate.is_relative_to((root / "data").resolve()):
        raise ValueError("LiveCodeBench candidate registry must stay under data/")
    predecessor = resolve_binding(
        root,
        bindings["predecessor_registry"],
        name="predecessor_registry",
    )
    if candidate == predecessor:
        raise ValueError("LiveCodeBench candidate cannot replace its predecessor")
    return plan


def download_exact_object(destination: Path, *, maximum_network_bytes: int) -> int:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite LiveCodeBench source: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    if temporary.exists():
        raise FileExistsError(f"stale LiveCodeBench partial download exists: {temporary}")
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "Lumi-LiveCodeBench-research-signatures-v0.1"},
    )
    import certifi

    tls_context = ssl.create_default_context(cafile=certifi.where())
    written = 0
    try:
        with (
            urllib.request.urlopen(request, timeout=60, context=tls_context) as response,
            temporary.open("xb") as output,
        ):
            while True:
                block = response.read(min(1024 * 1024, maximum_network_bytes + 1 - written))
                if not block:
                    break
                written += len(block)
                if written > maximum_network_bytes:
                    raise ValueError("LiveCodeBench network limit exceeded")
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if written != SOURCE_BYTES or sha256_file(temporary) != SOURCE_SHA256:
            raise ValueError("LiveCodeBench exact source object verification failed")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return written


def verify_source_object(path: Path) -> None:
    if not path.is_file() or path.stat().st_size != SOURCE_BYTES:
        raise ValueError("LiveCodeBench source object size changed")
    if sha256_file(path) != SOURCE_SHA256:
        raise ValueError("LiveCodeBench source object SHA-256 changed")


def load_projected_prompt_rows(
    path: Path,
    *,
    maximum_decoded_bytes: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Project only allowed columns; unexpected JSON fields are ignored by Arrow."""

    verify_source_object(path)
    import pyarrow
    import pyarrow as pa
    import pyarrow.json as pajson

    if pyarrow.__version__ != "25.0.0":
        raise ValueError("LiveCodeBench pyarrow version changed")
    schema = pa.schema([pa.field(field, pa.string()) for field in SOURCE_COLUMNS])
    table = pajson.read_json(
        path,
        read_options=pajson.ReadOptions(use_threads=False, block_size=16 * 1024 * 1024),
        parse_options=pajson.ParseOptions(
            explicit_schema=schema,
            unexpected_field_behavior="ignore",
        ),
    )
    if table.column_names != list(SOURCE_COLUMNS):
        raise ValueError("LiveCodeBench projected column order changed")
    if table.num_rows != SOURCE_ROWS:
        raise ValueError("LiveCodeBench source row count changed")
    rows: list[dict[str, str]] = []
    decoded_utf8_bytes = 0
    question_ids: set[str] = set()
    platform_counts: Counter[str] = Counter()
    for row_number, value in enumerate(table.to_pylist(), start=1):
        if set(value) != set(SOURCE_COLUMNS):
            raise ValueError(f"LiveCodeBench projected row schema changed: {row_number}")
        row: dict[str, str] = {}
        for field in SOURCE_COLUMNS:
            item = value.get(field)
            if item is None and field == "starter_code":
                item = ""
            if not isinstance(item, str):
                raise ValueError(f"LiveCodeBench projected field type changed: {field}")
            if field != "starter_code" and not item.strip():
                raise ValueError(f"LiveCodeBench projected field is empty: {field}")
            row[field] = item
            decoded_utf8_bytes += len(item.encode("utf-8"))
        if row["question_id"] in question_ids:
            raise ValueError("LiveCodeBench question IDs are duplicated")
        question_ids.add(row["question_id"])
        platform_counts[row["platform"]] += 1
        rows.append(row)
    decoded_bound = max(decoded_utf8_bytes, int(table.nbytes))
    if decoded_bound > maximum_decoded_bytes:
        raise ValueError("LiveCodeBench decoded-byte limit exceeded")
    rows.sort(key=lambda row: row["question_id"])
    return rows, {
        "decoded_bound_bytes": decoded_bound,
        "projected_arrow_table_bytes": int(table.nbytes),
        "projected_columns": list(SOURCE_COLUMNS),
        "projected_utf8_bytes": decoded_utf8_bytes,
        "prohibited_columns_materialized": False,
        "unexpected_field_behavior": "ignore",
        "platform_counts": dict(sorted(platform_counts.items())),
    }


def build_prompt_signatures(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    signatures: list[dict[str, Any]] = []
    for row in rows:
        question_id = row["question_id"]
        full_prompt = f"{row['question_title']}\n\n{row['question_content']}"
        if row["starter_code"].strip():
            full_prompt += f"\n\nStarter code:\n{row['starter_code']}"
        values = {
            "full_prompt": full_prompt,
            "question_content": row["question_content"],
            "question_title": row["question_title"],
        }
        for variant in PROMPT_VARIANTS:
            entry = benchmark_signature_entry(
                benchmark="livecodebench",
                example_id=f"release_v6:{question_id}:{variant}",
                split="release_v6",
                text=values[variant],
                anchor_ngram_tokens=ANCHOR_NGRAM_TOKENS,
                max_anchors=MAXIMUM_ANCHORS,
                minimum_anchor_matches=MINIMUM_ANCHOR_MATCHES,
            )
            if {"text", "prompt", *PROHIBITED_FIELDS}.intersection(entry):
                raise ValueError("LiveCodeBench signature retained prohibited data")
            signatures.append(entry)
    signatures.sort(key=lambda item: (item["benchmark"], item["example_id"]))
    identities = [(item["benchmark"], item["example_id"]) for item in signatures]
    if len(signatures) != EXPECTED_SIGNATURES or len(set(identities)) != len(identities):
        raise ValueError("LiveCodeBench prompt-signature inventory changed")
    return signatures


def build_candidate_metadata(
    predecessor_metadata: Mapping[str, Any],
    predecessor_entries: Sequence[Mapping[str, Any]],
    signatures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if predecessor_metadata.get("entries") != PREDECESSOR_ENTRIES:
        raise ValueError("LiveCodeBench predecessor entry count changed")
    if predecessor_metadata.get("raw_text_retained") is not False:
        raise ValueError("LiveCodeBench predecessor raw-text policy changed")
    return {
        "anchor_ngram_tokens": ANCHOR_NGRAM_TOKENS,
        "appended_source": {
            "dataset_id": SOURCE_DATASET_ID,
            "declared_license_id": "cc_non_spdx_ambiguous",
            "entries": len(signatures),
            "object_sha256": SOURCE_SHA256,
            "prompt_variants": list(PROMPT_VARIANTS),
            "research_only": True,
            "revision": SOURCE_REVISION,
            "split": "release_v6",
            "training_ingestion_prohibited": True,
        },
        "candidate_not_active": True,
        "entries": len(predecessor_entries) + len(signatures),
        "max_anchors_per_example": MAXIMUM_ANCHORS,
        "minimum_anchor_matches": MINIMUM_ANCHOR_MATCHES,
        "normalization": "nfkc_casefold_whitespace_v1",
        "predecessor": {
            "entries": len(predecessor_entries),
            "entries_sha256": records_sha256(predecessor_entries),
        },
        "raw_text_retained": False,
        "schema_version": 1,
    }


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported LiveCodeBench acquisition report")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if report.get("report_sha256") != sha256_value(unsigned):
        raise ValueError("LiveCodeBench acquisition report seal is invalid")
    if report.get("production_eligible") is not False:
        raise ValueError("LiveCodeBench acquisition crossed production boundary")
    boundaries = report.get("boundaries")
    if not isinstance(boundaries, Mapping):
        raise ValueError("LiveCodeBench acquisition boundaries are missing")
    if any(
        boundaries.get(field) is not False
        for field in (
            "candidate_promoted",
            "gpu_or_sft_used",
            "prohibited_field_values_materialized",
            "prohibited_field_values_retained_outside_exact_source_object",
            "prohibited_field_values_hashed",
            "raw_benchmark_text_in_git",
            "training_ingestion_performed",
        )
    ):
        raise ValueError("LiveCodeBench acquisition crossed an authorization boundary")
    source = report.get("source")
    if not isinstance(source, Mapping) or source.get("object_sha256") != SOURCE_SHA256:
        raise ValueError("LiveCodeBench acquisition source identity changed")
    signatures = report.get("signatures")
    if not isinstance(signatures, Mapping) or signatures.get("entries") != (
        EXPECTED_SIGNATURES
    ):
        raise ValueError("LiveCodeBench acquisition signature count changed")


__all__ = [
    "MAXIMUM_SIGNATURES",
    "EXPECTED_SIGNATURES",
    "OWNER_AUTHORIZATION_STATEMENT",
    "PREDECESSOR_ENTRIES",
    "PROHIBITED_FIELDS",
    "PROMPT_VARIANTS",
    "REPORT_SCHEMA_VERSION",
    "SOURCE_OBJECT",
    "SOURCE_REVISION",
    "SOURCE_SHA256",
    "VERIFICATION_SCHEMA_VERSION",
    "build_candidate_metadata",
    "build_prompt_signatures",
    "download_exact_object",
    "load_acquisition_authorization",
    "load_projected_prompt_rows",
    "validate_report",
]
