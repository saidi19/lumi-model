"""Bounded prompt-only BigCodeBench signature acquisition."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import ssl
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .data_quality import benchmark_signature_entry, sha256_file

PLAN_SCHEMA_VERSION = "lumi.bigcodebench-prompt-signature-acquisition-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.bigcodebench-prompt-signature-acquisition.v0.1"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.bigcodebench-prompt-signature-acquisition-verification.v0.1"
)

OWNER_AUTHORIZATION_STATEMENT = (
    "I approve one bounded CPU-only BigCodeBench v0.1.4 prompt-signature acquisition "
    "at Hugging Face revision b74c0d0bf70d2c0bc459be537895cca163007f1a, "
    "limited to exact object data/v0.1.4-00000-of-00001.parquet (2,362,110 "
    "bytes; SHA-256 "
    "d9a4965821c9507ebdfb551c288656b2d5fe553234f5183044333ca8a4018267), "
    "maximum 4 MiB network / 32 MiB decoded / 60 minutes. It may read only "
    "task_id, complete_prompt, instruct_prompt, and code_prompt, create at most "
    "3,420 private prompt signatures, and build a raw-text-free candidate successor "
    "preserving all 46,999 existing registry entries. It must not retain or hash "
    "canonical_solution, test, doc_struct, entry_point, or libs. No external API, "
    "GPU, active-registry mutation, candidate-source promotion, dataset seal, SFT "
    "training, LiveCodeBench data, unrelated data, or external distribution."
)

SOURCE_DATASET_ID = "bigcode/bigcodebench"
SOURCE_REVISION = "b74c0d0bf70d2c0bc459be537895cca163007f1a"
SOURCE_OBJECT = "data/v0.1.4-00000-of-00001.parquet"
SOURCE_BYTES = 2_362_110
SOURCE_SHA256 = "d9a4965821c9507ebdfb551c288656b2d5fe553234f5183044333ca8a4018267"
SOURCE_URL = (
    "https://huggingface.co/datasets/bigcode/bigcodebench/resolve/"
    f"{SOURCE_REVISION}/{SOURCE_OBJECT}?download=true"
)
SOURCE_ROWS = 1_140
SOURCE_COLUMNS = ("task_id", "complete_prompt", "instruct_prompt", "code_prompt")
PROMPT_FIELDS = SOURCE_COLUMNS[1:]
PROHIBITED_FIELDS = (
    "canonical_solution",
    "test",
    "doc_struct",
    "entry_point",
    "libs",
)
PREDECESSOR_ENTRIES = 46_999
MAXIMUM_SIGNATURES = 3_420
ANCHOR_NGRAM_TOKENS = 8
MAXIMUM_ANCHORS = 2
MINIMUM_ANCHOR_MATCHES = 2

EXPECTED_AUTHORIZATION = {
    "active_registry_mutation_authorized": False,
    "candidate_registry_promotion_authorized": False,
    "cpu_only_prompt_signature_acquisition_authorized": True,
    "dataset_seal_authorized": False,
    "external_api_authorized": False,
    "external_distribution_authorized": False,
    "gpu_authorized": False,
    "livecodebench_data_authorized": False,
    "network_access_authorized": True,
    "sft_training_authorized": False,
    "unrelated_data_access_authorized": False,
}
EXPECTED_LIMITS = {
    "maximum_decoded_bytes": 33_554_432,
    "maximum_network_bytes": 4_194_304,
    "maximum_prompt_signatures": MAXIMUM_SIGNATURES,
    "maximum_runtime_minutes": 60,
    "maximum_source_objects": 1,
    "maximum_source_rows": SOURCE_ROWS,
}
EXPECTED_BINDINGS = {
    "active_benchmark_registry",
    "data_quality_implementation",
    "implementation",
    "preflight_acquisition_plan",
    "preflight_report",
    "preflight_verification",
    "runner",
    "verifier",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_value(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    workspace = root.parent.resolve()
    relative = specification.get("path")
    expected = _require_sha256(specification.get("sha256"), f"bindings.{name}.sha256")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"invalid BigCodeBench binding: {name}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(workspace) or not resolved.is_file():
        raise ValueError(f"BigCodeBench binding missing or out of scope: {name}")
    if sha256_file(resolved) != expected:
        raise ValueError(f"BigCodeBench binding changed: {name}")
    return resolved


def load_acquisition_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _read_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported BigCodeBench acquisition plan")
    if plan.get("status") != "authorized_cpu_only_prompt_signature_acquisition":
        raise ValueError("BigCodeBench acquisition plan status changed")
    if plan.get("observed_at") != "2026-08-07":
        raise ValueError("BigCodeBench acquisition observation date changed")
    if plan.get("owner_authorization_statement") != OWNER_AUTHORIZATION_STATEMENT:
        raise ValueError("BigCodeBench owner authorization changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("BigCodeBench authorization boundary changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("BigCodeBench acquisition limits changed")
    if plan.get("runtime") != {
        "ca_bundle": "certifi==2026.7.22",
        "parquet_reader": "pyarrow==25.0.0",
    }:
        raise ValueError("BigCodeBench parquet runtime changed")
    if plan.get("source") != {
        "allowed_columns": list(SOURCE_COLUMNS),
        "dataset_id": SOURCE_DATASET_ID,
        "declared_license_id": "apache-2.0",
        "object": {
            "path": SOURCE_OBJECT,
            "sha256": SOURCE_SHA256,
            "size_bytes": SOURCE_BYTES,
            "url": SOURCE_URL,
        },
        "prohibited_value_columns": list(PROHIBITED_FIELDS),
        "revision": SOURCE_REVISION,
        "split": "v0.1.4",
    }:
        raise ValueError("BigCodeBench source contract changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDINGS:
        raise ValueError("BigCodeBench binding inventory changed")
    for name, specification in bindings.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"invalid BigCodeBench binding object: {name}")
        resolve_binding(root, specification, name=name)
    preflight = _read_json(
        resolve_binding(
            root,
            bindings["preflight_acquisition_plan"],
            name="preflight_acquisition_plan",
        )
    )
    if preflight.get("plan_sha256") != (
        "e679a7556838b4ec30f163472dff9714f6851540349388bbf4b149a6037dc734"
    ):
        raise ValueError("BigCodeBench preflight decision binding changed")
    if preflight.get("bigcodebench", {}).get("network_object") != {
        "bytes": SOURCE_BYTES,
        "git_blob_sha1": "ef4634c67d030dd64fd23f5304afdebed11cc6ed",
        "lfs_sha256": SOURCE_SHA256,
        "path": SOURCE_OBJECT,
    }:
        raise ValueError("BigCodeBench preflight object binding changed")
    output = plan.get("output")
    if not isinstance(output, Mapping) or set(output) != {
        "candidate_registry_path",
        "private_root",
        "raw_benchmark_text_in_git",
        "report_path",
        "verification_path",
    }:
        raise ValueError("BigCodeBench output inventory changed")
    if output.get("raw_benchmark_text_in_git") is not False:
        raise ValueError("BigCodeBench raw benchmark text cannot enter Git")
    private_root = (root / str(output["private_root"])).resolve()
    if not private_root.is_relative_to(root.parent.resolve()) or private_root.is_relative_to(root):
        raise ValueError("BigCodeBench private root must stay outside the repository")
    candidate = (root / str(output["candidate_registry_path"])).resolve()
    if not candidate.is_relative_to((root / "data").resolve()):
        raise ValueError("BigCodeBench candidate registry must stay under data/")
    active = resolve_binding(
        root,
        bindings["active_benchmark_registry"],
        name="active_benchmark_registry",
    )
    if candidate == active:
        raise ValueError("BigCodeBench candidate cannot replace the active registry")
    return plan


def download_exact_object(destination: Path, *, maximum_network_bytes: int) -> int:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite BigCodeBench source: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    if temporary.exists():
        raise FileExistsError(f"stale BigCodeBench partial download exists: {temporary}")
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "Lumi-BigCodeBench-signature-acquisition-v0.1"},
    )
    import certifi

    tls_context = ssl.create_default_context(cafile=certifi.where())
    written = 0
    try:
        with (
            urllib.request.urlopen(request, timeout=60, context=tls_context) as response,
            temporary.open("xb") as out,
        ):
            while True:
                block = response.read(min(1024 * 1024, maximum_network_bytes + 1 - written))
                if not block:
                    break
                written += len(block)
                if written > maximum_network_bytes:
                    raise ValueError("BigCodeBench network limit exceeded")
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        if written != SOURCE_BYTES or sha256_file(temporary) != SOURCE_SHA256:
            raise ValueError("BigCodeBench exact source object verification failed")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return written


def verify_source_object(path: Path) -> None:
    if not path.is_file() or path.stat().st_size != SOURCE_BYTES:
        raise ValueError("BigCodeBench source object size changed")
    if sha256_file(path) != SOURCE_SHA256:
        raise ValueError("BigCodeBench source object SHA-256 changed")


def load_prompt_rows(
    path: Path,
    *,
    maximum_decoded_bytes: int,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    verify_source_object(path)
    import pyarrow
    import pyarrow.parquet as parquet

    if pyarrow.__version__ != "25.0.0":
        raise ValueError("BigCodeBench pyarrow version changed")

    table = parquet.read_table(path, columns=list(SOURCE_COLUMNS), use_threads=False)
    if table.column_names != list(SOURCE_COLUMNS):
        raise ValueError("BigCodeBench selected column order changed")
    if table.num_rows != SOURCE_ROWS:
        raise ValueError("BigCodeBench source row count changed")
    rows: list[dict[str, str]] = []
    decoded_utf8_bytes = 0
    task_ids: set[str] = set()
    for row_number, value in enumerate(table.to_pylist(), start=1):
        if set(value) != set(SOURCE_COLUMNS):
            raise ValueError(f"BigCodeBench selected row schema changed: {row_number}")
        row: dict[str, str] = {}
        for field in SOURCE_COLUMNS:
            item = value.get(field)
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"BigCodeBench selected field is empty: {row_number}/{field}")
            row[field] = item
            decoded_utf8_bytes += len(item.encode("utf-8"))
        if row["task_id"] in task_ids:
            raise ValueError("BigCodeBench task IDs are duplicated")
        task_ids.add(row["task_id"])
        rows.append(row)
    decoded_bound = max(decoded_utf8_bytes, int(table.nbytes))
    if decoded_bound > maximum_decoded_bytes:
        raise ValueError("BigCodeBench decoded-byte limit exceeded")
    rows.sort(key=lambda row: row["task_id"])
    return rows, {
        "decoded_bound_bytes": decoded_bound,
        "selected_arrow_table_bytes": int(table.nbytes),
        "selected_utf8_bytes": decoded_utf8_bytes,
    }


def build_prompt_signatures(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    signatures: list[dict[str, Any]] = []
    for row in rows:
        task_id = row["task_id"]
        for field in PROMPT_FIELDS:
            entry = benchmark_signature_entry(
                benchmark="bigcodebench",
                example_id=f"v0.1.4:{task_id}:{field}",
                split="v0.1.4",
                text=row[field],
                anchor_ngram_tokens=ANCHOR_NGRAM_TOKENS,
                max_anchors=MAXIMUM_ANCHORS,
                minimum_anchor_matches=MINIMUM_ANCHOR_MATCHES,
            )
            if {"text", "prompt", *PROHIBITED_FIELDS}.intersection(entry):
                raise ValueError("BigCodeBench signature retained a prohibited field")
            signatures.append(entry)
    signatures.sort(key=lambda item: (item["benchmark"], item["example_id"]))
    identities = [(item["benchmark"], item["example_id"]) for item in signatures]
    if len(signatures) != MAXIMUM_SIGNATURES or len(set(identities)) != len(identities):
        raise ValueError("BigCodeBench prompt-signature inventory changed")
    return signatures


def load_registry(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with gzip.open(path, mode="rt", encoding="utf-8") as handle:
        try:
            metadata_record = json.loads(next(handle))
        except (StopIteration, json.JSONDecodeError) as error:
            raise ValueError("benchmark signature registry is empty or invalid") from error
        metadata = metadata_record.get("_meta")
        if not isinstance(metadata, dict):
            raise ValueError("benchmark signature registry metadata is invalid")
        entries = []
        for line_number, line in enumerate(handle, start=2):
            entry = json.loads(line)
            if not isinstance(entry, dict):
                raise ValueError(f"benchmark registry entry is invalid: {line_number}")
            if {"text", "prompt"}.intersection(entry):
                raise ValueError("benchmark registry retained raw evaluation text")
            entries.append(entry)
    identities = [(entry.get("benchmark"), entry.get("example_id")) for entry in entries]
    if len(identities) != len(set(identities)):
        raise ValueError("benchmark registry identities are duplicated")
    return metadata, entries


def records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def build_candidate_metadata(
    predecessor_metadata: Mapping[str, Any],
    predecessor_entries: Sequence[Mapping[str, Any]],
    signatures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if predecessor_metadata.get("entries") != PREDECESSOR_ENTRIES:
        raise ValueError("active benchmark registry entry count changed")
    if predecessor_metadata.get("raw_text_retained") is not False:
        raise ValueError("active benchmark registry raw-text policy changed")
    return {
        "anchor_ngram_tokens": ANCHOR_NGRAM_TOKENS,
        "appended_source": {
            "dataset_id": SOURCE_DATASET_ID,
            "entries": len(signatures),
            "object_sha256": SOURCE_SHA256,
            "prompt_fields": list(PROMPT_FIELDS),
            "revision": SOURCE_REVISION,
            "split": "v0.1.4",
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


def write_candidate_registry_atomic(
    path: Path,
    *,
    metadata: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite candidate registry: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gzip_handle:
            with io.TextIOWrapper(gzip_handle, encoding="utf-8") as text_handle:
                text_handle.write(
                    json.dumps(
                        {"_meta": dict(metadata)},
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
                for entry in entries:
                    text_handle.write(
                        json.dumps(
                            entry,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                        + "\n"
                    )
        raw_handle.flush()
        os.fsync(raw_handle.fileno())
    temporary.replace(path)


def write_json_atomic(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite BigCodeBench artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite BigCodeBench artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported BigCodeBench acquisition report")
    seal = _require_sha256(report.get("report_sha256"), "report_sha256")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if sha256_value(unsigned) != seal:
        raise ValueError("BigCodeBench acquisition report seal is invalid")
    boundaries = report.get("boundaries", {})
    if boundaries != {
        "active_registry_mutated": False,
        "candidate_promoted": False,
        "gpu_or_sft_used": False,
        "livecodebench_accessed": False,
        "prohibited_field_values_hashed": False,
        "prohibited_field_values_read": False,
        "raw_benchmark_text_in_git": False,
    }:
        raise ValueError("BigCodeBench acquisition boundary changed")
    if report.get("production_eligible") is not False:
        raise ValueError("BigCodeBench candidate crossed production boundary")


__all__ = [
    "MAXIMUM_SIGNATURES",
    "OWNER_AUTHORIZATION_STATEMENT",
    "PLAN_SCHEMA_VERSION",
    "PREDECESSOR_ENTRIES",
    "PROMPT_FIELDS",
    "PROHIBITED_FIELDS",
    "REPORT_SCHEMA_VERSION",
    "SOURCE_BYTES",
    "SOURCE_COLUMNS",
    "SOURCE_OBJECT",
    "SOURCE_REVISION",
    "SOURCE_ROWS",
    "SOURCE_SHA256",
    "SOURCE_URL",
    "VERIFICATION_SCHEMA_VERSION",
    "build_candidate_metadata",
    "build_prompt_signatures",
    "download_exact_object",
    "load_acquisition_plan",
    "load_prompt_rows",
    "load_registry",
    "records_sha256",
    "sha256_value",
    "validate_report",
    "verify_source_object",
    "write_candidate_registry_atomic",
    "write_json_atomic",
    "write_jsonl_atomic",
]
