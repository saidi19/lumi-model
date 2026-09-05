"""Authorization-bound CPU acquisition and curation for pinned OpenR1 data."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import ssl
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any, Protocol

import certifi

from .data_quality import BenchmarkRegistry
from .posttraining_data import SFTRecordDecision, evaluate_sft_record, load_sft_policy
from .posttraining_format import ChatTemplate, ChatTokenizer, encode_sft_messages
from .posttraining_openr1_full_curation import (
    DATA_OBJECTS,
    DATASET_ID,
    EXACT_OBJECTS,
    EXPECTED_FEATURES,
    SOURCE_ID,
    SOURCE_REVISION,
)
from .posttraining_openr1_pii_secondary import diagnose_line_scoped_phone_patterns
from .posttraining_sources import load_sft_source_registry

RUN_SCHEMA_VERSION = "lumi.sft-openr1-full-curation-run.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-openr1-full-curation-report.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-openr1-full-curation-verification.v0.1"
EXPECTED_STATUS = "authorized_cpu_only_acquisition_and_curation"
EXPECTED_AUTHORIZATION = {
    "bulk_data_download_authorized": True,
    "dataset_promotion_authorized": False,
    "external_api_spend_authorized": False,
    "network_access_authorized": True,
    "paid_gpu_authorized": False,
    "production_dataset_seal_authorized": False,
    "production_sft_authorized": False,
    "purpose": "pinned_openr1_cpu_only_full_source_curation",
    "unrelated_dataset_access_authorized": False,
}
EXPECTED_LIMITS = {
    "context_window": 4_096,
    "expected_data_download_bytes": 2_149_897_914,
    "expected_decoded_data_bytes": 4_964_543_659,
    "maximum_data_objects": 10,
    "maximum_decoded_bytes": 5_368_709_120,
    "maximum_network_bytes": 2_415_919_104,
    "maximum_objects": 11,
    "maximum_retained_supervised_tokens": 6_600_000,
    "maximum_runtime_minutes": 360,
    "maximum_source_rows": 93_733,
    "owner_sample_records": 32,
}
EXPECTED_OWNER_AUTHORIZATION_STATEMENT = (
    "I approve one bounded CPU-only OpenR1 full-source acquisition and curation at "
    "revision e4e141ec9dea9f8326f4d347be56105859b2bd68, maximum 11 exact objects / "
    "93,733 rows / 2,415,919,104 network bytes / 5,368,709,120 decoded bytes / "
    "360 minutes. It may fetch only the 10 exact default-config train parquet shards "
    "and README, write private raw data, a non-production candidate, attribution "
    "evidence, raw-text-free reports, and a deterministic 32-row owner-review packet "
    "outside Git, retaining at most 6,600,000 supervised tokens. Use only "
    "verifier-passed complete generations, never reference solutions; preserve the "
    "global PII filter and quarantine unresolved rights or safety cases. No external "
    "API, GPU, source-registry promotion, production dataset seal, SFT training, "
    "unrelated data access, or external distribution."
)
EXPECTED_POLICY = {
    "assistant_trace_field": "generations",
    "assistant_trace_must_differ_from_reference_solution": True,
    "benchmark_decontamination_required": True,
    "cross_split_exact_and_near_deduplication_required": True,
    "external_image_or_url_dependencies_allowed": False,
    "global_pii_filter_modified": False,
    "near_duplicate": {
        "num_perm": 128,
        "package_version": "2.0.0",
        "scheme": "affine32",
        "seed": 1,
        "shingle_tokens": 5,
        "threshold": 0.9,
    },
    "origin_label": "synthetic",
    "problem_group_split_required": True,
    "record_level_attribution_required": True,
    "reference_solution_field_allowed_as_assistant_target": False,
    "require_aligned_generation_verifier_vectors": True,
    "require_nonempty_upstream_problem_source": True,
    "require_reasoning_complete": True,
    "require_verifier_field": "correctness_math_verify",
    "require_verifier_pass": True,
    "secondary_pii_reconciliation": {
        "allowed_match_role": "assistant",
        "contact_keyword_scope": "same_line_as_phone_match",
        "cross_message_join_is_not_a_source_text_match": True,
        "math_context_required": True,
        "other_sensitive_detector_hits_allowed": False,
        "quarantine_before_reclassification": True,
    },
    "validation_fraction": 0.05,
}

_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_URL = re.compile(r"https?://", re.IGNORECASE)
_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]+\)|<img\b", re.IGNORECASE)


class DownloadResponse(Protocol):
    headers: Mapping[str, str]

    def __enter__(self) -> DownloadResponse: ...

    def __exit__(self, *args: object) -> None: ...

    def geturl(self) -> str: ...

    def read(self, size: int = -1) -> bytes: ...


OpenURL = Callable[[urllib.request.Request], DownloadResponse]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_file_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256")
    return value


def resolve_workspace_path(
    repository_root: Path,
    binding: Mapping[str, Any],
    *,
    require_hash: bool = True,
) -> Path:
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("run binding path must be non-empty")
    resolved = (repository_root / relative).resolve()
    workspace_root = repository_root.parent.resolve()
    if not resolved.is_relative_to(workspace_root):
        raise ValueError(f"run binding escapes workspace: {relative}")
    if not resolved.is_file():
        raise FileNotFoundError(f"run binding does not exist: {relative}")
    if require_hash:
        expected = _require_sha256(binding.get("sha256"), f"binding {relative}")
        if file_sha256(resolved) != expected:
            raise ValueError(f"run binding SHA-256 changed: {relative}")
    return resolved


def load_full_run_plan(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Load the exact owner-authorized OpenR1 run and verify static bindings."""

    root = Path(repository_root).resolve()
    plan = _load_json(path)
    if plan.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError("unsupported OpenR1 full-run plan")
    if plan.get("status") != EXPECTED_STATUS:
        raise ValueError("OpenR1 full-run status changed")
    if plan.get("observed_at") != "2026-08-06":
        raise ValueError("OpenR1 full-run observation date changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("OpenR1 full-run authorization changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("OpenR1 full-run limits changed")
    if plan.get("policy") != EXPECTED_POLICY:
        raise ValueError("OpenR1 full-run policy changed")

    source = plan.get("source")
    expected_source = {
        "config": "default",
        "dataset_id": DATASET_ID,
        "exact_objects": EXACT_OBJECTS,
        "license_id": "apache-2.0",
        "revision": SOURCE_REVISION,
        "source_id": SOURCE_ID,
        "split": "train",
    }
    if source != expected_source:
        raise ValueError("OpenR1 source identity or allowlist changed")

    owner = plan.get("owner_authorization")
    if not isinstance(owner, Mapping):
        raise ValueError("OpenR1 owner authorization is missing")
    statement = owner.get("statement")
    if statement != EXPECTED_OWNER_AUTHORIZATION_STATEMENT:
        raise ValueError("OpenR1 owner authorization statement changed")
    if hashlib.sha256(statement.encode("utf-8")).hexdigest() != _require_sha256(
        owner.get("sha256"), "owner authorization"
    ):
        raise ValueError("OpenR1 owner authorization SHA-256 changed")

    bindings = plan.get("bindings")
    required_bindings = {
        "benchmark_registry",
        "candidate_source_registry",
        "chat_template",
        "data_contract",
        "encoding_implementation",
        "preflight_plan",
        "preflight_report",
        "run_implementation",
        "run_runner",
        "run_verifier",
        "secondary_pii_implementation",
        "source_registry",
        "tokenizer_config",
        "tokenizer_json",
        "verified_trace_implementation",
    }
    if not isinstance(bindings, Mapping) or set(bindings) != required_bindings:
        raise ValueError("OpenR1 full-run binding inventory changed")
    for name in sorted(required_bindings):
        binding = bindings[name]
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid OpenR1 full-run binding: {name}")
        resolve_workspace_path(root, binding)

    output = plan.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("OpenR1 full-run output policy is missing")
    expected_output_keys = {
        "private_artifact_root",
        "production_promotion_permitted",
        "raw_text_free_report_path",
        "raw_text_free_verification_path",
        "raw_training_text_in_git",
    }
    if set(output) != expected_output_keys:
        raise ValueError("OpenR1 full-run output inventory changed")
    if output.get("raw_training_text_in_git") is not False:
        raise ValueError("raw OpenR1 text may not enter Git")
    if output.get("production_promotion_permitted") is not False:
        raise ValueError("OpenR1 production promotion is not authorized")
    private_root = output.get("private_artifact_root")
    report_path = output.get("raw_text_free_report_path")
    verification_path = output.get("raw_text_free_verification_path")
    if not isinstance(private_root, str) or not private_root:
        raise ValueError("private OpenR1 artifact root is missing")
    if not isinstance(report_path, str) or not report_path.startswith("evidence/"):
        raise ValueError("OpenR1 report must stay under evidence/")
    if not isinstance(verification_path, str) or not verification_path.startswith(
        "evidence/"
    ):
        raise ValueError("OpenR1 verification must stay under evidence/")
    private_resolved = (root / private_root).resolve()
    if private_resolved.is_relative_to(root):
        raise ValueError("raw OpenR1 artifacts must stay outside the repository")
    if not private_resolved.is_relative_to(root.parent):
        raise ValueError("raw OpenR1 artifacts must stay inside the workspace")
    return plan


def exact_object_url(object_path: str) -> str:
    if object_path not in {str(item["path"]) for item in EXACT_OBJECTS}:
        raise ValueError("refusing unlisted OpenR1 object")
    quoted = urllib.parse.quote(object_path, safe="/")
    return (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
        f"{SOURCE_REVISION}/{quoted}?download=true"
    )


def _safe_redirect_host(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not hostname:
        raise ValueError("OpenR1 download redirected outside HTTPS")
    if not (
        hostname == "huggingface.co"
        or hostname.endswith(".huggingface.co")
        or hostname.endswith(".hf.co")
    ):
        raise ValueError(
            f"OpenR1 download redirected to an unapproved host: {hostname}"
        )
    return hostname


def secure_urlopen(request: urllib.request.Request) -> DownloadResponse:
    context = ssl.create_default_context(cafile=certifi.where())
    return urllib.request.urlopen(request, context=context)


def download_exact_object(
    *,
    destination: Path,
    maximum_bytes: int,
    object_path: str,
    open_url: OpenURL = secure_urlopen,
) -> dict[str, Any]:
    """Download one allowlisted object atomically under a strict byte ceiling."""

    if maximum_bytes < 1:
        raise ValueError("OpenR1 download byte allowance exhausted")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite OpenR1 object: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale OpenR1 temporary exists: {temporary}")
    requested_url = exact_object_url(object_path)
    request = urllib.request.Request(
        requested_url,
        headers={"User-Agent": "lumi-openr1-curation-v0.1"},
        method="GET",
    )
    digest = hashlib.sha256()
    downloaded = 0
    final_host = ""
    try:
        with open_url(request) as response:
            final_host = _safe_redirect_host(response.geturl())
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared = int(content_length)
                except ValueError as error:
                    raise ValueError("invalid OpenR1 Content-Length") from error
                if declared < 0 or declared > maximum_bytes:
                    raise ValueError("OpenR1 object exceeds remaining network limit")
            with temporary.open("xb") as handle:
                while True:
                    chunk = response.read(
                        min(4 * 1024 * 1024, maximum_bytes - downloaded + 1)
                    )
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > maximum_bytes:
                        raise ValueError("OpenR1 object exceeded network byte limit")
                    digest.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "bytes": downloaded,
        "object_path": object_path,
        "redirect_host": final_host,
        "requested_url": requested_url,
        "sha256": digest.hexdigest(),
    }


def dependency_flags(text: str) -> tuple[bool, bool]:
    return _MARKDOWN_IMAGE.search(text) is not None, _URL.search(text) is not None


def select_verified_generation(
    row: Mapping[str, Any],
    *,
    global_row_index: int,
    object_path: str,
    object_row_index: int,
    object_sha256: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """Select the first complete verifier-passed generation, never the solution."""

    problem = row.get("problem")
    solution = row.get("solution")
    uuid = row.get("uuid")
    upstream_source = row.get("source")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (problem, solution, uuid, upstream_source)
    ):
        return None, None, "missing_row_level_provenance_or_text"
    problem_image, problem_url = dependency_flags(problem)
    if problem_image or problem_url:
        return None, None, "problem_external_dependency"

    generations = row.get("generations")
    completeness = row.get("is_reasoning_complete")
    verifier = row.get("correctness_math_verify")
    if not all(
        isinstance(value, list) for value in (generations, completeness, verifier)
    ):
        return None, None, "missing_generation_verifier_vectors"
    if not generations or not (
        len(generations) == len(completeness) == len(verifier)
    ):
        return None, None, "misaligned_generation_verifier_vectors"

    selected: tuple[int, str] | None = None
    reference_matches = 0
    dependency_matches = 0
    for generation_index, trace in enumerate(generations):
        if (
            completeness[generation_index] is not True
            or verifier[generation_index] is not True
            or not isinstance(trace, str)
            or not trace.strip()
        ):
            continue
        if trace == solution:
            reference_matches += 1
            continue
        trace_image, trace_url = dependency_flags(trace)
        if trace_image or trace_url:
            dependency_matches += 1
            continue
        selected = generation_index, trace
        break
    if selected is None:
        if reference_matches:
            return None, None, "verified_trace_matches_reference_solution"
        if dependency_matches:
            return None, None, "verified_trace_external_dependency"
        return None, None, "no_safe_verified_complete_generation"

    generation_index, trace = selected
    identity = {
        "dataset_id": DATASET_ID,
        "generation_index": generation_index,
        "revision": SOURCE_REVISION,
        "row_index": global_row_index,
        "uuid": uuid,
    }
    record_digest = sha256_value(identity)
    messages = [
        {"content": problem, "role": "user"},
        {"content": trace, "role": "assistant"},
    ]
    transform = {
        **identity,
        "adapter": "openr1_verified_generation_full_v0.1",
        "output_schema": "lumi.sft.v0.1",
        "trace_sha256": hashlib.sha256(trace.encode("utf-8")).hexdigest(),
    }
    record_id = f"openr1_math_220k-{record_digest[:24]}"
    record = {
        "generation_provenance": {
            "generation_index": generation_index,
            "reasoning_complete": True,
            "trace_field": "generations",
            "upstream_source_sha256": hashlib.sha256(
                upstream_source.encode("utf-8")
            ).hexdigest(),
            "verifier_field": "correctness_math_verify",
            "verifier_passed": True,
        },
        "language": "en",
        "messages": messages,
        "provenance": {
            "consent_basis": "source_level_open_license",
            "content_sha256": sha256_value(messages),
            "transform_sha256": sha256_value(transform),
        },
        "quality": {
            "pii_scan": "passed",
            "review_method": "openr1_verified_generation_full_gate_v0.1",
            "review_status": "accepted",
            "score": 0.9,
            "secret_scan": "passed",
        },
        "record_id": record_id,
        "safety": {
            "policy_version": "lumi-sft-basic-pattern-scan-v0.1",
            "risk_labels": [],
        },
        "schema_version": "lumi.sft.v0.1",
        "source": {
            "dataset_id": DATASET_ID,
            "document_id": uuid,
            "license_id": "apache-2.0",
            "revision": SOURCE_REVISION,
            "source_id": SOURCE_ID,
            "uri": (
                f"https://huggingface.co/datasets/{DATASET_ID}/blob/"
                f"{SOURCE_REVISION}/{object_path}#row={object_row_index}"
            ),
        },
        "split": "train",
        "task_category": "reasoning_math_science",
    }
    attribution = {
        "dataset_id": DATASET_ID,
        "generation_index": generation_index,
        "global_row_index": global_row_index,
        "record_id": record_id,
        "source_object_path": object_path,
        "source_object_row_index": object_row_index,
        "source_object_sha256": object_sha256,
        "source_revision": SOURCE_REVISION,
        "upstream_problem_source": upstream_source,
        "upstream_problem_source_sha256": hashlib.sha256(
            upstream_source.encode("utf-8")
        ).hexdigest(),
        "uuid": uuid,
    }
    return record, attribution, None


def reconcile_openr1_pii_decision(
    record: Mapping[str, Any],
    decision: SFTRecordDecision,
) -> tuple[bool, str | None]:
    """Reclassify only proven line-scoped math false positives; never change the filter."""

    if decision.accepted:
        return True, None
    if set(decision.reasons) != {"pii_pattern"}:
        return False, None
    metrics = diagnose_line_scoped_phone_patterns([record])
    if (
        metrics["global_phone_detector_records"] == 1
        and metrics["unresolved_records"] == 0
        and metrics["other_sensitive_detector_records"] == 0
        and (
            metrics["line_scoped_math_numeric_false_positive_records"] == 1
            or metrics["cross_message_join_artifact_records"] == 1
        )
    ):
        if metrics["line_scoped_math_numeric_false_positive_records"] == 1:
            return True, "line_scoped_math_numeric_false_positive"
        return True, "cross_message_join_artifact"
    return False, "unresolved_phone_pattern"


def encode_record(
    record: Mapping[str, Any],
    *,
    template: ChatTemplate,
    tokenizer: ChatTokenizer,
) -> dict[str, int] | None:
    try:
        encoded = encode_sft_messages(
            record["messages"],
            tokenizer=tokenizer,
            template=template,
        )
    except ValueError as error:
        if "maximum is 4096" not in str(error):
            raise
        return None
    return {
        "masked_tokens": encoded.masked_tokens,
        "serialized_tokens": len(encoded.input_ids),
        "supervised_tokens": encoded.supervised_tokens,
    }


def deterministic_split(records: list[dict[str, Any]], fraction: float) -> int:
    """Assign an exact deterministic validation count after global deduplication."""

    if not 0 < fraction < 1:
        raise ValueError("validation fraction must be between zero and one")
    validation_count = max(1, round(len(records) * fraction)) if records else 0
    validation_ids = {
        str(record["record_id"])
        for record in sorted(
            records,
            key=lambda row: hashlib.sha256(
                f"validation:{row['record_id']}".encode()
            ).hexdigest(),
        )[:validation_count]
    }
    for record in records:
        record["split"] = (
            "validation" if record["record_id"] in validation_ids else "train"
        )
    return validation_count


def nearest_rank(values: Sequence[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def build_token_metrics(
    records: Sequence[Mapping[str, Any]],
    token_by_id: Mapping[str, Mapping[str, int]],
    *,
    overlength_rejected: int,
) -> dict[str, Any]:
    lengths: list[int] = []
    supervised: list[int] = []
    masked: list[int] = []
    by_split: defaultdict[str, Counter[str]] = defaultdict(Counter)
    ledger: list[dict[str, Any]] = []
    for record in records:
        record_id = str(record["record_id"])
        metrics = token_by_id[record_id]
        total = int(metrics["serialized_tokens"])
        supervised_tokens = int(metrics["supervised_tokens"])
        masked_tokens = int(metrics["masked_tokens"])
        lengths.append(total)
        supervised.append(supervised_tokens)
        masked.append(masked_tokens)
        bucket = by_split[str(record["split"])]
        bucket["records"] += 1
        bucket["serialized_tokens"] += total
        bucket["supervised_tokens"] += supervised_tokens
        bucket["masked_tokens"] += masked_tokens
        ledger.append(
            {
                "masked_tokens": masked_tokens,
                "record_id": record_id,
                "serialized_tokens": total,
                "supervised_tokens": supervised_tokens,
            }
        )
    return {
        "accepted_records": len(records),
        "assistant_only_loss_verified": True,
        "by_split": {key: dict(value) for key, value in sorted(by_split.items())},
        "context_window": 4_096,
        "masked_tokens": sum(masked),
        "overlength_rejected": overlength_rejected,
        "serialized_length": {
            "maximum": max(lengths, default=0),
            "mean": mean(lengths) if lengths else 0,
            "median": median(lengths) if lengths else 0,
            "minimum": min(lengths, default=0),
            "p95": nearest_rank(lengths, 0.95),
        },
        "serialized_tokens": sum(lengths),
        "supervised_tokens": sum(supervised),
        "token_ledger_sha256": sha256_value(sorted(ledger, key=lambda row: row["record_id"])),
    }


def deterministic_owner_sample(
    records: Sequence[Mapping[str, Any]],
    *,
    maximum_records: int,
) -> list[dict[str, Any]]:
    if maximum_records < 1:
        raise ValueError("owner sample size must be positive")
    buckets: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[str(record["split"])].append(record)
    for rows in buckets.values():
        rows.sort(
            key=lambda row: hashlib.sha256(
                f"owner:{row['record_id']}".encode()
            ).hexdigest()
        )
    selected: list[Mapping[str, Any]] = []
    offset = 0
    keys = sorted(buckets)
    target = min(maximum_records, len(records))
    while len(selected) < target:
        added = False
        for key in keys:
            rows = buckets[key]
            if offset < len(rows):
                selected.append(rows[offset])
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
        offset += 1
    return [
        {
            "language_mode": "english",
            "messages": [dict(message) for message in row["messages"]],
            "record_id": row["record_id"],
            "skill_id": "reasoning_math_science",
            "split": row["split"],
        }
        for row in selected
    ]


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite JSON artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite JSONL artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite text artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def artifact_inventory(
    paths: Mapping[str, Path],
    *,
    raw_text_names: frozenset[str],
) -> dict[str, Any]:
    inventory = []
    for name, path in sorted(paths.items()):
        inventory.append(
            {
                "bytes": path.stat().st_size,
                "contains_private_training_text": name in raw_text_names,
                "name": name,
                "sha256": file_sha256(path),
            }
        )
    return {
        "objects": inventory,
        "total_bytes": sum(item["bytes"] for item in inventory),
    }


def load_runtime_components(
    plan: Mapping[str, Any],
    *,
    repository_root: Path,
) -> tuple[Any, Any, BenchmarkRegistry, ChatTemplate, ChatTokenizer]:
    from transformers import AutoTokenizer

    bindings = plan["bindings"]
    contract_path = resolve_workspace_path(repository_root, bindings["data_contract"])
    source_registry_path = resolve_workspace_path(
        repository_root, bindings["source_registry"]
    )
    benchmark_path = resolve_workspace_path(
        repository_root, bindings["benchmark_registry"]
    )
    template_path = resolve_workspace_path(repository_root, bindings["chat_template"])
    tokenizer_path = resolve_workspace_path(repository_root, bindings["tokenizer_json"])
    tokenizer_config = resolve_workspace_path(
        repository_root, bindings["tokenizer_config"]
    )
    if tokenizer_path.parent != tokenizer_config.parent:
        raise ValueError("frozen Lumi tokenizer files are not colocated")

    policy = load_sft_policy(contract_path)
    source_registry = load_sft_source_registry(
        source_registry_path,
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
        expected_sha256=policy.source_registry_sha256,
    )
    benchmark = BenchmarkRegistry(policy.benchmark_decontamination, repository_root)
    if benchmark.path.resolve() != benchmark_path:
        raise ValueError("benchmark registry resolved to a different path")
    from .posttraining_format import load_chat_template

    template = load_chat_template(template_path)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path.parent, local_files_only=True)
    return policy, source_registry, benchmark, template, tokenizer


def source_registry_presplit_decision(
    record: Mapping[str, Any],
    *,
    policy: Any,
    source_registry: Any,
    benchmark: BenchmarkRegistry,
) -> SFTRecordDecision:
    if record.get("split") != "train":
        raise ValueError("source-registry check must run before derived validation split")
    return evaluate_sft_record(
        record,
        policy=policy,
        benchmark_registry=benchmark,
        source_registry=source_registry,
    )


def report_sha256(report: Mapping[str, Any]) -> str:
    return sha256_value({key: value for key, value in report.items() if key != "report_sha256"})


def validate_report_is_raw_text_free(report: Mapping[str, Any]) -> None:
    forbidden_keys = {
        "answer",
        "generations",
        "messages",
        "problem",
        "raw_text",
        "solution",
        "text",
    }

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            if forbidden_keys.intersection(value):
                raise ValueError("OpenR1 report retained a raw-text field")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(report)


def source_object_allowlist() -> frozenset[str]:
    return frozenset([*DATA_OBJECTS, "README.md"])


def expected_data_columns() -> frozenset[str]:
    return frozenset(EXPECTED_FEATURES)
