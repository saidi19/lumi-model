"""Small, reproducible data-sampling primitives for Lumi.

This module intentionally builds a canary corpus, not the final training corpus.
It streams pinned gzip JSONL files, writes only accepted text, and records enough
provenance to reproduce or remove every accepted segment later.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA1_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
PROVENANCE_KEY_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}")

SECRET_PATTERNS = {
    "private_key": re.compile(
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----",
        re.IGNORECASE,
    ),
    "aws_access_key": re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
    "github_token": re.compile(
        r"(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{20,255}|github_pat_[A-Za-z0-9_]{20,255})"
    ),
    "google_api_key": re.compile(r"(?<![A-Za-z0-9])AIza[A-Za-z0-9_-]{35}(?![A-Za-z0-9])"),
    "slack_token": re.compile(r"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{20,}"),
    "assigned_credential": re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|password|passwd|secret)\s*[:=]\s*"
        r"[\"']?[A-Za-z0-9_./+=-]{20,}"
    ),
}

PII_PATTERNS = {
    "email": (
        re.compile(r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])", re.I),
        "[EMAIL]",
    ),
    "ipv4": (
        re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?!\d|\.\d)"),
        "[IP_ADDRESS]",
    ),
    "phone": (
        re.compile(
            r"(?<!\w)(?:\+\d{1,3}[-. ]?)?(?:\(\d{2,4}\)[-. ]?|\d{2,4}[-. ])"
            r"\d{3,4}[-. ]\d{4}(?!\w)"
        ),
        "[PHONE_NUMBER]",
    ),
}


@dataclass(frozen=True)
class FilterResult:
    text: str | None
    rejection_reason: str | None
    redactions: dict[str, int]
    raw_sha256: str


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def detect_secret(text: str) -> str | None:
    for name, pattern in SECRET_PATTERNS.items():
        if pattern.search(text):
            return name
    return None


def detect_pii(text: str) -> str | None:
    for name, (pattern, _) in PII_PATTERNS.items():
        if pattern.search(text):
            return name
    return None


def filter_document(text: str, policy: Mapping[str, Any]) -> FilterResult:
    raw_sha256 = _sha256_bytes(text.encode("utf-8"))
    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.strip()
    if len(normalized) < int(policy["min_document_characters"]):
        return FilterResult(None, "too_short", {}, raw_sha256)

    secret_kind = detect_secret(normalized)
    if secret_kind is not None:
        return FilterResult(None, f"secret:{secret_kind}", {}, raw_sha256)

    disallowed_controls = sum(
        unicodedata.category(character) == "Cc" and character not in "\n\t"
        for character in normalized
    )
    if disallowed_controls / max(len(normalized), 1) > float(
        policy["max_control_character_fraction"]
    ):
        return FilterResult(None, "control_characters", {}, raw_sha256)

    meaningful_lines = [line.strip() for line in normalized.splitlines() if len(line.strip()) >= 20]
    if len(meaningful_lines) >= 10:
        repeated_fraction = 1.0 - (len(set(meaningful_lines)) / len(meaningful_lines))
        if repeated_fraction > float(policy["max_repeated_line_fraction"]):
            return FilterResult(None, "repeated_lines", {}, raw_sha256)

    redactions: dict[str, int] = {}
    for name, (pattern, replacement) in PII_PATTERNS.items():
        normalized, count = pattern.subn(replacement, normalized)
        if count:
            redactions[name] = count
    return FilterResult(normalized, None, redactions, raw_sha256)


def split_utf8(text: str, max_bytes: int, min_characters: int) -> Iterator[str]:
    """Split text at a nearby whitespace boundary without breaking UTF-8."""

    remainder = text.strip()
    while remainder:
        encoded = remainder.encode("utf-8")
        if len(encoded) <= max_bytes:
            if len(remainder) >= min_characters:
                yield remainder
            return

        low, high = 1, len(remainder)
        while low < high:
            middle = (low + high + 1) // 2
            if len(remainder[:middle].encode("utf-8")) <= max_bytes:
                low = middle
            else:
                high = middle - 1
        hard_cut = low
        preferred_floor = max(int(hard_cut * 0.65), 1)
        cut = hard_cut
        for separator in ("\n\n", "\n", " "):
            candidate = remainder.rfind(separator, preferred_floor, hard_cut)
            if candidate >= preferred_floor:
                cut = candidate + len(separator)
                break
        segment = remainder[:cut].strip()
        remainder = remainder[cut:].lstrip()
        if len(segment) >= min_characters:
            yield segment


def validate_sample_config(config: Mapping[str, Any]) -> None:
    dataset = config.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("sample config must contain a dataset object")
    if not isinstance(dataset.get("id"), str) or "/" not in dataset["id"]:
        raise ValueError("dataset.id must be a Hugging Face owner/name identifier")
    revision = dataset.get("revision")
    if not isinstance(revision, str) or REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError("dataset.revision must be a pinned 40-character commit SHA")

    target = config.get("target_uncompressed_text_bytes")
    if not isinstance(target, int) or isinstance(target, bool) or target <= 0:
        raise ValueError("target_uncompressed_text_bytes must be a positive integer")

    filters = config.get("filters")
    if not isinstance(filters, Mapping):
        raise ValueError("sample config must contain a filters object")
    required_filters = {
        "max_control_character_fraction",
        "max_repeated_line_fraction",
        "max_segment_utf8_bytes",
        "min_document_characters",
        "normalization",
        "pii_policy",
        "secret_policy",
    }
    missing_filters = required_filters - set(filters)
    if missing_filters:
        raise ValueError(f"missing filter settings: {sorted(missing_filters)}")
    if filters["normalization"] != "NFC":
        raise ValueError("the v0.1 sampler requires NFC normalization")
    if filters["pii_policy"] != "redact" or filters["secret_policy"] != "drop_document":
        raise ValueError("the v0.1 safety policy must redact PII and drop secret-bearing documents")
    if int(filters["max_segment_utf8_bytes"]) < 1024:
        raise ValueError("max_segment_utf8_bytes must be at least 1024")
    if int(filters["min_document_characters"]) < 1:
        raise ValueError("min_document_characters must be positive")

    mixture = config.get("mixture")
    if not isinstance(mixture, list) or not mixture:
        raise ValueError("sample config mixture must be a non-empty list")
    categories: set[str] = set()
    total_weight = 0.0
    for entry in mixture:
        if not isinstance(entry, Mapping):
            raise ValueError("every mixture entry must be an object")
        category = entry.get("category")
        if not isinstance(category, str) or not category:
            raise ValueError("every mixture entry needs a category")
        if category in categories:
            raise ValueError(f"duplicate mixture category: {category}")
        categories.add(category)
        weight = entry.get("weight")
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight <= 0:
            raise ValueError(f"category {category} has an invalid weight")
        total_weight += float(weight)
        sources = entry.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"category {category} must have at least one source")
        for source in sources:
            component = source.get("component")
            source_file = source.get("file")
            blob_sha1 = source.get("blob_sha1")
            if not isinstance(component, str) or not component:
                raise ValueError(f"category {category} has a source without a component")
            if (
                not isinstance(source_file, str)
                or source_file.startswith("/")
                or ".." in Path(source_file).parts
                or not source_file.endswith(".gz")
            ):
                raise ValueError(f"category {category} has an unsafe or unsupported source file")
            if not isinstance(blob_sha1, str) or SHA1_PATTERN.fullmatch(blob_sha1) is None:
                raise ValueError(f"source {source_file} must pin a 40-character blob SHA")
            compressed_bytes = source.get("compressed_bytes")
            if (
                not isinstance(compressed_bytes, int)
                or isinstance(compressed_bytes, bool)
                or compressed_bytes <= 0
            ):
                raise ValueError(f"source {source_file} must record its compressed size")
            source_dataset_id = source.get("dataset_id")
            source_revision = source.get("revision")
            if (source_dataset_id is None) != (source_revision is None):
                raise ValueError(
                    f"source {source_file} must set both dataset_id and revision or neither"
                )
            if source_dataset_id is not None:
                if not isinstance(source_dataset_id, str) or "/" not in source_dataset_id:
                    raise ValueError(f"source {source_file} has an invalid dataset_id")
                if (
                    not isinstance(source_revision, str)
                    or REVISION_PATTERN.fullmatch(source_revision) is None
                ):
                    raise ValueError(f"source {source_file} must pin its dataset revision")
            retained_fields = source.get("retain_provenance", {})
            static_fields = source.get("static_provenance", {})
            if not isinstance(retained_fields, Mapping) or not isinstance(static_fields, Mapping):
                raise ValueError(f"source {source_file} provenance settings must be objects")
            overlap = set(retained_fields) & set(static_fields)
            if overlap:
                raise ValueError(
                    f"source {source_file} repeats static and retained keys: {sorted(overlap)}"
                )
            for name, path in retained_fields.items():
                if PROVENANCE_KEY_PATTERN.fullmatch(name) is None:
                    raise ValueError(f"source {source_file} has an invalid provenance key: {name}")
                if not isinstance(path, str) or not path or path.startswith("."):
                    raise ValueError(f"source {source_file} has an invalid path for {name}")
            for name, value in static_fields.items():
                if PROVENANCE_KEY_PATTERN.fullmatch(name) is None:
                    raise ValueError(f"source {source_file} has an invalid static key: {name}")
                _validate_provenance_value(value, f"source {source_file} static field {name}")
            required_fields = source.get("required_non_null_provenance", [])
            if not isinstance(required_fields, list) or not all(
                isinstance(name, str) for name in required_fields
            ):
                raise ValueError(f"source {source_file} required provenance must be a list")
            available_fields = set(retained_fields) | set(static_fields)
            if not set(required_fields) <= available_fields:
                raise ValueError(
                    f"source {source_file} requires unavailable provenance fields: "
                    f"{sorted(set(required_fields) - available_fields)}"
                )
    if not math.isclose(total_weight, 1.0, abs_tol=1e-9):
        raise ValueError(f"mixture weights must sum to 1.0, got {total_weight}")


def load_sample_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    validate_sample_config(config)
    return config


def _category_targets(config: Mapping[str, Any], target_bytes: int) -> dict[str, int]:
    mixture = config["mixture"]
    remaining = target_bytes
    targets: dict[str, int] = {}
    for index, entry in enumerate(mixture):
        if index == len(mixture) - 1:
            value = remaining
        else:
            value = int(target_bytes * float(entry["weight"]))
            remaining -= value
        targets[entry["category"]] = value
    return targets


def _source_dataset(
    default_dataset: Mapping[str, Any], source: Mapping[str, Any]
) -> dict[str, str]:
    return {
        "id": str(source.get("dataset_id", default_dataset["id"])),
        "revision": str(source.get("revision", default_dataset["revision"])),
    }


def _source_url(dataset: Mapping[str, Any], source: Mapping[str, Any]) -> str:
    if "url" in source:
        return str(source["url"])
    source_dataset = _source_dataset(dataset, source)
    dataset_id = quote(source_dataset["id"], safe="/")
    revision = quote(source_dataset["revision"], safe="")
    source_file = quote(str(source["file"]), safe="/")
    return f"https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/{source_file}"


def _iter_source_lines(
    dataset: Mapping[str, Any], source: Mapping[str, Any], timeout_seconds: float
) -> Iterator[tuple[int, str]]:
    request = urllib.request.Request(
        _source_url(dataset, source),
        headers={"User-Agent": "lumi-data-canary/0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
        with gzip.GzipFile(fileobj=response) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as text_stream:
                yield from enumerate(text_stream, start=1)


def _atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def _record_id(dataset: Mapping[str, Any], source_file: str, line: int, segment: int) -> str:
    location = f"{dataset['id']}@{dataset['revision']}:{source_file}:{line}:{segment}"
    return f"lumi-{_sha256_bytes(location.encode('utf-8'))}"


def _value_at_path(record: Mapping[str, Any], path: str) -> Any:
    value: Any = record
    for key in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _validate_provenance_value(value: Any, location: str) -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list) and all(
        item is None or isinstance(item, (str, int, float, bool)) for item in value
    ):
        return
    raise ValueError(f"{location} must be a scalar or a list of scalars")


def _extract_provenance(
    upstream_record: Mapping[str, Any], source: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    retained = dict(source.get("static_provenance", {}))
    for name, path in source.get("retain_provenance", {}).items():
        value = _value_at_path(upstream_record, path)
        _validate_provenance_value(value, f"retained provenance field {name}")
        retained[name] = value
    missing = [
        name
        for name in source.get("required_non_null_provenance", [])
        if retained.get(name) is None or retained.get(name) == "" or retained.get(name) == []
    ]
    return retained, missing


def _scan_records(sample_path: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "category_bytes": defaultdict(int),
        "category_documents": defaultdict(int),
        "content_hashes": set(),
        "document_ids": set(),
        "documents": 0,
        "source_files": set(),
        "text_bytes": 0,
    }
    if not sample_path.exists():
        return stats
    with sample_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
                text = record["text"]
                category = record["provenance"]["category"]
                document_id = record["id"]
                content_hash = record["content_sha256"]
                source_file = record["provenance"]["source_file"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(
                    f"invalid existing sample at line {line_number}: {error}"
                ) from error
            if not isinstance(text, str):
                raise ValueError(f"invalid existing sample text at line {line_number}")
            if document_id in stats["document_ids"]:
                raise ValueError(f"duplicate document ID in existing sample: {document_id}")
            if content_hash in stats["content_hashes"]:
                raise ValueError(f"duplicate content hash in existing sample: {content_hash}")
            text_bytes = len(text.encode("utf-8"))
            stats["document_ids"].add(document_id)
            stats["content_hashes"].add(content_hash)
            stats["source_files"].add(source_file)
            stats["documents"] += 1
            stats["text_bytes"] += text_bytes
            stats["category_bytes"][category] += text_bytes
            stats["category_documents"][category] += 1
    return stats


def _manifest_from_run(
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    run_fingerprint: str,
    sample_path: Path,
    targets: Mapping[str, int],
    stats: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    categories = {
        category: {
            "documents": int(stats["category_documents"].get(category, 0)),
            "target_text_utf8_bytes": target,
            "text_utf8_bytes": int(stats["category_bytes"].get(category, 0)),
        }
        for category, target in targets.items()
    }
    source_files = []
    for mixture_entry in config["mixture"]:
        for source in mixture_entry["sources"]:
            source_files.append(
                {
                    "blob_sha1": source["blob_sha1"],
                    "category": mixture_entry["category"],
                    "component": source["component"],
                    "compressed_bytes": source["compressed_bytes"],
                    "dataset_id": _source_dataset(config["dataset"], source)["id"],
                    "file": source["file"],
                    "required_non_null_provenance": source.get(
                        "required_non_null_provenance", []
                    ),
                    "revision": _source_dataset(config["dataset"], source)["revision"],
                    "used": source["file"] in stats["source_files"],
                }
            )
    return {
        "config_sha256": config_sha256,
        "created_at": _utc_now(),
        "dataset": dict(config["dataset"]),
        "filters": dict(config["filters"]),
        "limitations": [
            "Canary only: this sample does not approve a pilot or full-model training run.",
            "Near-duplicate removal, language identification, and benchmark "
            "decontamination are not yet applied.",
            "Secret and PII detection is pattern-based and is not an exhaustive privacy audit.",
        ],
        "output": {
            "file": sample_path.name,
            "file_bytes": sample_path.stat().st_size,
            "sha256": _sha256_file(sample_path),
            "text_utf8_bytes": int(stats["text_bytes"]),
        },
        "run_fingerprint": run_fingerprint,
        "schema_version": 1,
        "source_files": source_files,
        "statistics": {
            "categories": categories,
            "documents": int(stats["documents"]),
            "network_retries": int(state.get("network_retries", 0)),
            "pii_redactions": dict(sorted(state.get("pii_redactions", {}).items())),
            "rejections": dict(sorted(state.get("rejections", {}).items())),
        },
        "status": "complete",
        "target_uncompressed_text_bytes": sum(targets.values()),
        "version": config["version"],
    }


def build_sample(
    config: Mapping[str, Any],
    output_directory: Path,
    *,
    target_bytes: int | None = None,
    resume: bool = False,
    timeout_seconds: float = 60.0,
    retries: int = 2,
) -> dict[str, Any]:
    """Stream a deterministic canary sample into ``output_directory``."""

    validate_sample_config(config)
    if retries < 0:
        raise ValueError("retries cannot be negative")
    effective_target = (
        int(config["target_uncompressed_text_bytes"]) if target_bytes is None else target_bytes
    )
    if effective_target <= 0:
        raise ValueError("target_bytes must be positive")
    targets = _category_targets(config, effective_target)
    config_sha256 = _sha256_bytes(_canonical_json_bytes(config))
    run_fingerprint = _sha256_bytes(
        _canonical_json_bytes({"config_sha256": config_sha256, "target_bytes": effective_target})
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    sample_path = output_directory / "sample.jsonl"
    manifest_path = output_directory / "manifest.json"
    state_path = output_directory / "run-state.json"
    if sample_path.exists() and not resume:
        raise FileExistsError(f"sample already exists; pass resume=True: {sample_path}")

    if state_path.exists():
        with state_path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("run_fingerprint") != run_fingerprint:
            raise ValueError("existing output was created with a different config or target")
        if state.get("status") == "complete" and manifest_path.exists():
            validate_sample(output_directory, expected_config=config)
            with manifest_path.open(encoding="utf-8") as handle:
                return json.load(handle)
    else:
        if sample_path.exists():
            raise ValueError("cannot resume sample without matching run-state.json")
        state = {
            "config_sha256": config_sha256,
            "network_retries": 0,
            "pii_redactions": {},
            "rejections": {},
            "run_fingerprint": run_fingerprint,
            "schema_version": 1,
            "started_at": _utc_now(),
            "status": "building",
            "target_uncompressed_text_bytes": effective_target,
        }
        _atomic_json_write(state_path, state)

    stats = _scan_records(sample_path)
    rejections = Counter(state.get("rejections", {}))
    pii_redactions = Counter(state.get("pii_redactions", {}))
    dataset = config["dataset"]
    policy = config["filters"]
    writes_since_state = 0

    with sample_path.open("a", encoding="utf-8") as output:
        for mixture_entry in config["mixture"]:
            category = mixture_entry["category"]
            target = targets[category]
            if stats["category_bytes"].get(category, 0) >= target:
                continue
            for source in mixture_entry["sources"]:
                if stats["category_bytes"].get(category, 0) >= target:
                    break
                last_error: Exception | None = None
                source_dataset = _source_dataset(dataset, source)
                for attempt in range(retries + 1):
                    try:
                        source_lines = _iter_source_lines(dataset, source, timeout_seconds)
                        for source_line, line in source_lines:
                            if stats["category_bytes"].get(category, 0) >= target:
                                break
                            try:
                                upstream_record = json.loads(line)
                            except json.JSONDecodeError:
                                rejections["invalid_json"] += 1
                                continue
                            raw_text = upstream_record.get("text")
                            if not isinstance(raw_text, str):
                                rejections["missing_text"] += 1
                                continue
                            retained_provenance, missing_provenance = _extract_provenance(
                                upstream_record, source
                            )
                            if missing_provenance:
                                for name in missing_provenance:
                                    rejections[f"missing_provenance:{name}"] += 1
                                continue
                            filtered = filter_document(raw_text, policy)
                            if filtered.rejection_reason is not None:
                                rejections[filtered.rejection_reason] += 1
                                continue
                            assert filtered.text is not None
                            redactions_recorded = False
                            segments = split_utf8(
                                filtered.text,
                                int(policy["max_segment_utf8_bytes"]),
                                int(policy["min_document_characters"]),
                            )
                            for segment_index, segment in enumerate(segments):
                                if stats["category_bytes"].get(category, 0) >= target:
                                    break
                                document_id = _record_id(
                                    source_dataset,
                                    source["file"],
                                    source_line,
                                    segment_index,
                                )
                                if document_id in stats["document_ids"]:
                                    continue
                                segment_bytes = segment.encode("utf-8")
                                content_hash = _sha256_bytes(segment_bytes)
                                if content_hash in stats["content_hashes"]:
                                    rejections["exact_duplicate"] += 1
                                    continue
                                record = {
                                    "content_sha256": content_hash,
                                    "id": document_id,
                                    "provenance": {
                                        "category": category,
                                        "component": source["component"],
                                        "dataset_id": source_dataset["id"],
                                        "dataset_revision": source_dataset["revision"],
                                        "raw_text_sha256": filtered.raw_sha256,
                                        "segment_index": segment_index,
                                        "source_blob_sha1": source["blob_sha1"],
                                        "source_file": source["file"],
                                        "source_line": source_line,
                                    },
                                    "text": segment,
                                    "transformations": {
                                        "normalization": "NFC",
                                        "pii_redactions": (
                                            filtered.redactions if not redactions_recorded else {}
                                        ),
                                    },
                                }
                                if retained_provenance:
                                    record["retained_provenance"] = retained_provenance
                                output.write(
                                    json.dumps(
                                        record,
                                        ensure_ascii=False,
                                        separators=(",", ":"),
                                        sort_keys=True,
                                    )
                                    + "\n"
                                )
                                if not redactions_recorded:
                                    pii_redactions.update(filtered.redactions)
                                    redactions_recorded = True
                                stats["document_ids"].add(document_id)
                                stats["content_hashes"].add(content_hash)
                                stats["source_files"].add(source["file"])
                                stats["documents"] += 1
                                stats["text_bytes"] += len(segment_bytes)
                                stats["category_bytes"][category] += len(segment_bytes)
                                stats["category_documents"][category] += 1
                                writes_since_state += 1
                                if writes_since_state >= 100:
                                    output.flush()
                                    state["rejections"] = dict(rejections)
                                    state["pii_redactions"] = dict(pii_redactions)
                                    state["updated_at"] = _utc_now()
                                    _atomic_json_write(state_path, state)
                                    writes_since_state = 0
                        last_error = None
                        break
                    except (EOFError, OSError, TimeoutError, urllib.error.URLError) as error:
                        last_error = error
                        if attempt >= retries:
                            break
                        state["network_retries"] = int(state.get("network_retries", 0)) + 1
                        time.sleep(min(2**attempt, 4))
                if last_error is not None:
                    raise RuntimeError(
                        f"failed streaming pinned source {source['file']}"
                    ) from last_error
            if stats["category_bytes"].get(category, 0) < target:
                raise RuntimeError(
                    f"pinned sources for {category} ended at "
                    f"{stats['category_bytes'].get(category, 0)} of {target} target bytes"
                )
            output.flush()
            state["rejections"] = dict(rejections)
            state["pii_redactions"] = dict(pii_redactions)
            state["updated_at"] = _utc_now()
            _atomic_json_write(state_path, state)

    stats = _scan_records(sample_path)
    manifest = _manifest_from_run(
        config=config,
        config_sha256=config_sha256,
        run_fingerprint=run_fingerprint,
        sample_path=sample_path,
        targets=targets,
        stats=stats,
        state=state,
    )
    _atomic_json_write(manifest_path, manifest)
    validate_sample(output_directory, expected_config=config)
    state["manifest_sha256"] = _sha256_file(manifest_path)
    state["status"] = "complete"
    state["updated_at"] = _utc_now()
    _atomic_json_write(state_path, state)
    return manifest


def validate_sample(
    output_directory: Path, *, expected_config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    manifest_path = output_directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"sample manifest does not exist: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "complete":
        raise ValueError("sample manifest is not complete")
    if expected_config is not None:
        validate_sample_config(expected_config)
        expected_config_sha256 = _sha256_bytes(_canonical_json_bytes(expected_config))
        if manifest.get("config_sha256") != expected_config_sha256:
            raise ValueError("sample manifest does not match the expected config")
    output_name = manifest.get("output", {}).get("file")
    if not isinstance(output_name, str) or Path(output_name).name != output_name:
        raise ValueError("manifest output path must be a plain file name")
    sample_path = output_directory / output_name
    if not sample_path.is_file():
        raise FileNotFoundError(f"sample output does not exist: {sample_path}")
    expected_output = manifest["output"]
    if sample_path.stat().st_size != expected_output["file_bytes"]:
        raise ValueError("sample output byte size does not match manifest")
    if _sha256_file(sample_path) != expected_output["sha256"]:
        raise ValueError("sample output SHA-256 does not match manifest")

    filters = manifest["filters"]
    default_dataset = manifest["dataset"]
    allowed_sources = {
        (
            source.get("dataset_id", default_dataset["id"]),
            source.get("revision", default_dataset["revision"]),
            source["file"],
            source["blob_sha1"],
            source["category"],
        )
        for source in manifest["source_files"]
    }
    required_provenance = {
        (
            source.get("dataset_id", default_dataset["id"]),
            source.get("revision", default_dataset["revision"]),
            source["file"],
        ): source.get("required_non_null_provenance", [])
        for source in manifest["source_files"]
    }
    content_hashes: set[str] = set()
    document_ids: set[str] = set()
    category_bytes: Counter[str] = Counter()
    category_documents: Counter[str] = Counter()
    total_text_bytes = 0
    with sample_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
                text = record["text"]
                provenance = record["provenance"]
                content_hash = record["content_sha256"]
                document_id = record["id"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"invalid sample record at line {line_number}: {error}") from error
            if not isinstance(text, str) or not text:
                raise ValueError(f"empty or non-text sample record at line {line_number}")
            if unicodedata.normalize("NFC", text) != text:
                raise ValueError(f"sample text is not NFC-normalized at line {line_number}")
            secret_kind = detect_secret(text)
            if secret_kind is not None:
                raise ValueError(f"secret pattern {secret_kind} remains at line {line_number}")
            pii_kind = detect_pii(text)
            if pii_kind is not None:
                raise ValueError(f"PII pattern {pii_kind} remains at line {line_number}")
            text_bytes = text.encode("utf-8")
            if len(text_bytes) > int(filters["max_segment_utf8_bytes"]):
                raise ValueError(
                    f"sample segment exceeds configured byte cap at line {line_number}"
                )
            actual_hash = _sha256_bytes(text_bytes)
            if content_hash != actual_hash or SHA256_PATTERN.fullmatch(content_hash) is None:
                raise ValueError(f"content hash mismatch at line {line_number}")
            if content_hash in content_hashes:
                raise ValueError(f"duplicate content at line {line_number}")
            if document_id in document_ids:
                raise ValueError(f"duplicate document ID at line {line_number}")
            expected_id = _record_id(
                {"id": provenance["dataset_id"], "revision": provenance["dataset_revision"]},
                provenance["source_file"],
                provenance["source_line"],
                provenance["segment_index"],
            )
            if document_id != expected_id:
                raise ValueError(f"provenance-derived ID mismatch at line {line_number}")
            source_key = (
                provenance["dataset_id"],
                provenance["dataset_revision"],
                provenance["source_file"],
                provenance["source_blob_sha1"],
                provenance["category"],
            )
            if source_key not in allowed_sources:
                raise ValueError(f"unlisted source provenance at line {line_number}")
            requirement_key = (
                provenance["dataset_id"],
                provenance["dataset_revision"],
                provenance["source_file"],
            )
            retained = record.get("retained_provenance", {})
            for name in required_provenance[requirement_key]:
                retained_value = retained.get(name)
                if retained_value is None or retained_value == "" or retained_value == []:
                    raise ValueError(
                        f"required retained provenance {name!r} is missing at line {line_number}"
                    )
            content_hashes.add(content_hash)
            document_ids.add(document_id)
            total_text_bytes += len(text_bytes)
            category_bytes[provenance["category"]] += len(text_bytes)
            category_documents[provenance["category"]] += 1

    if total_text_bytes != expected_output["text_utf8_bytes"]:
        raise ValueError("sample text byte count does not match manifest")
    if len(document_ids) != manifest["statistics"]["documents"]:
        raise ValueError("sample document count does not match manifest")
    for category, expected in manifest["statistics"]["categories"].items():
        if category_bytes[category] != expected["text_utf8_bytes"]:
            raise ValueError(f"category byte count mismatch for {category}")
        if category_documents[category] != expected["documents"]:
            raise ValueError(f"category document count mismatch for {category}")
        if category_bytes[category] < expected["target_text_utf8_bytes"]:
            raise ValueError(f"category target was not reached for {category}")
    return {
        "documents": len(document_ids),
        "sha256": expected_output["sha256"],
        "text_utf8_bytes": total_text_bytes,
    }
