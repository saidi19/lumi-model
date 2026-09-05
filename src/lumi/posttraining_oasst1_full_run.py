"""Authorization-bound CPU acquisition and curation for the pinned OASST1 source."""

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
from .posttraining_data import load_sft_policy
from .posttraining_format import ChatTemplate, ChatTokenizer, encode_sft_messages
from .posttraining_oasst1_curation import (
    EXPECTED_OBJECTS,
    SOURCE_DATASET_ID,
    SOURCE_ID,
    SOURCE_LICENSE,
    SOURCE_REVISION,
    OASST1Selection,
)
from .posttraining_sources import load_sft_source_registry

RUN_SCHEMA_VERSION = "lumi.sft-oasst1-full-curation-run.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-oasst1-full-curation-report.v0.1"
EXPECTED_STATUS = "authorized_cpu_only_acquisition_and_curation"
EXPECTED_AUTHORIZATION = {
    "bulk_data_download_authorized": True,
    "dataset_promotion_authorized": False,
    "external_api_spend_authorized": False,
    "network_access_authorized": True,
    "paid_gpu_authorized": False,
    "production_dataset_seal_authorized": False,
    "production_sft_authorized": False,
    "purpose": "pinned_oasst1_cpu_only_full_source_curation",
}
EXPECTED_LIMITS = {
    "maximum_data_objects": 2,
    "maximum_decoded_bytes": 134_217_728,
    "maximum_network_bytes": 67_108_864,
    "maximum_objects": 4,
    "maximum_runtime_minutes": 120,
    "maximum_source_rows": 88_838,
    "maximum_train_rows": 84_437,
    "maximum_validation_rows": 4_401,
    "owner_sample_records": 32,
}
EXPECTED_OWNER_AUTHORIZATION_STATEMENT = (
    "I approve one bounded CPU-only OASST1 full-source acquisition and curation at "
    "revision fdf72ae0827c1cda404aff25b6603abec9e3399b, maximum 4 exact objects / "
    "88,838 rows / 64 MiB network / 128 MiB decoded and 120 minutes. It may fetch "
    "only the pinned train and validation parquet files, LICENSE, and README, then run "
    "the sealed English root-direct rank-0 quality selector, attribution, schema, "
    "PII/secret, deduplication, benchmark decontamination, 4K-fit, token measurement, "
    "and raw-text-free reporting. No external API, paid GPU, unrelated objects, "
    "production promotion/seal, SFT training, or external distribution."
)
EXPECTED_DATA_COLUMNS = frozenset(
    {
        "created_date",
        "deleted",
        "detoxify",
        "emojis",
        "labels",
        "lang",
        "message_id",
        "message_tree_id",
        "model_name",
        "parent_id",
        "rank",
        "review_count",
        "review_result",
        "role",
        "synthetic",
        "text",
        "tree_state",
        "user_id",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
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


def _resolve_workspace_path(
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
    """Load the exact owner-authorized run plan and verify every static binding."""

    root = Path(repository_root).resolve()
    plan = _load_json(path)
    if plan.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError("unsupported OASST1 full-run plan")
    if plan.get("status") != EXPECTED_STATUS:
        raise ValueError("OASST1 full-run status changed")
    if plan.get("observed_at") != "2026-08-06":
        raise ValueError("OASST1 full-run observation date changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise ValueError("OASST1 full-run authorization changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("OASST1 full-run limits changed")

    source = plan.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("OASST1 full-run source is missing")
    if source.get("dataset_id") != SOURCE_DATASET_ID:
        raise ValueError("OASST1 dataset changed")
    if source.get("source_id") != SOURCE_ID or source.get("license_id") != SOURCE_LICENSE:
        raise ValueError("OASST1 source identity changed")
    if source.get("revision") != SOURCE_REVISION:
        raise ValueError("OASST1 revision changed")
    if source.get("exact_objects") != EXPECTED_OBJECTS:
        raise ValueError("OASST1 exact object allowlist changed")

    owner = plan.get("owner_authorization")
    if not isinstance(owner, Mapping):
        raise ValueError("owner authorization evidence is missing")
    statement = owner.get("statement")
    if statement != EXPECTED_OWNER_AUTHORIZATION_STATEMENT:
        raise ValueError("owner authorization statement changed")
    if hashlib.sha256(statement.encode("utf-8")).hexdigest() != _require_sha256(
        owner.get("sha256"), "owner authorization"
    ):
        raise ValueError("owner authorization statement SHA-256 changed")

    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("OASST1 full-run bindings are missing")
    required = {
        "benchmark_registry",
        "chat_template",
        "curation_implementation",
        "data_contract",
        "encoding_implementation",
        "preflight_plan",
        "preflight_report",
        "run_implementation",
        "run_runner",
        "source_registry",
        "tokenizer_config",
        "tokenizer_json",
    }
    if set(bindings) != required:
        raise ValueError("OASST1 full-run binding inventory changed")
    for name in sorted(required):
        binding = bindings[name]
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid OASST1 full-run binding: {name}")
        _resolve_workspace_path(root, binding)

    output = plan.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("OASST1 full-run output policy is missing")
    if output.get("raw_training_text_in_git") is not False:
        raise ValueError("raw OASST1 text may not enter Git")
    if output.get("production_promotion_permitted") is not False:
        raise ValueError("OASST1 production promotion is not authorized")
    private_root = output.get("private_artifact_root")
    report_path = output.get("raw_text_free_report_path")
    if not isinstance(private_root, str) or not private_root:
        raise ValueError("private OASST1 artifact root is missing")
    if not isinstance(report_path, str) or not report_path.startswith("evidence/"):
        raise ValueError("raw-text-free OASST1 report must stay under evidence/")
    private_resolved = (root / private_root).resolve()
    if private_resolved.is_relative_to(root):
        raise ValueError("raw OASST1 artifacts must stay outside the repository")
    if not private_resolved.is_relative_to(root.parent):
        raise ValueError("raw OASST1 artifacts must stay inside the workspace")
    return plan


def exact_object_url(object_path: str) -> str:
    if object_path not in {item["path"] for item in EXPECTED_OBJECTS}:
        raise ValueError("refusing unlisted OASST1 object")
    quoted = urllib.parse.quote(object_path, safe="/")
    return (
        f"https://huggingface.co/datasets/{SOURCE_DATASET_ID}/resolve/"
        f"{SOURCE_REVISION}/{quoted}?download=true"
    )


def _safe_redirect_host(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not hostname:
        raise ValueError("OASST1 download redirected outside HTTPS")
    if not (
        hostname == "huggingface.co"
        or hostname.endswith(".huggingface.co")
        or hostname.endswith(".hf.co")
    ):
        raise ValueError(f"OASST1 download redirected to an unapproved host: {hostname}")
    return hostname


def secure_urlopen(request: urllib.request.Request) -> DownloadResponse:
    """Open HTTPS with the pinned certifi trust store; never disable verification."""

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
        raise ValueError("OASST1 download byte allowance exhausted")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite OASST1 object: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale OASST1 download temporary exists: {temporary}")
    requested_url = exact_object_url(object_path)
    request = urllib.request.Request(
        requested_url,
        headers={"User-Agent": "lumi-oasst1-curation-v0.1"},
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
                    raise ValueError("invalid OASST1 Content-Length") from error
                if declared < 0 or declared > maximum_bytes:
                    raise ValueError("OASST1 object exceeds remaining network byte limit")
            with temporary.open("xb") as handle:
                while True:
                    chunk = response.read(min(1024 * 1024, maximum_bytes - downloaded + 1))
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > maximum_bytes:
                        raise ValueError("OASST1 object exceeded network byte limit")
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


def inspect_parquet_object(
    path: Path,
    *,
    expected_decoded_bytes: int,
    expected_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Decode one exact parquet object and return wrappers plus raw-text-free metrics."""

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(f"OASST1 parquet row count changed: {path.name}")
    columns = frozenset(parquet.schema_arrow.names)
    if columns != EXPECTED_DATA_COLUMNS:
        raise ValueError(f"OASST1 parquet schema changed: {path.name}")
    table = parquet.read()
    if table.num_rows != expected_rows:
        raise ValueError(f"OASST1 decoded row count changed: {path.name}")
    wrappers = [
        {"row": row, "row_idx": index}
        for index, row in enumerate(table.to_pylist())
    ]
    return wrappers, {
        "columns": sorted(columns),
        "metadata_logical_bytes": expected_decoded_bytes,
        "pyarrow_decoded_bytes": table.nbytes,
        "rows": table.num_rows,
    }


def build_sft_records(
    candidates: Iterable[Mapping[str, Any]],
    *,
    object_sha256_by_path: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert strict selections to the Lumi SFT schema and attribution rows."""

    records: list[dict[str, Any]] = []
    attributions: list[dict[str, Any]] = []
    for candidate in candidates:
        attribution = dict(candidate["attribution"])
        object_path = attribution["source_object_path"]
        object_sha256 = _require_sha256(
            object_sha256_by_path.get(object_path),
            f"source object {object_path}",
        )
        attribution["source_object_sha256"] = object_sha256
        answer_message_id = attribution["answer_message_id"]
        record = {
            "language": candidate["language"],
            "messages": [dict(message) for message in candidate["messages"]],
            "provenance": {
                "consent_basis": "source_level_open_license",
                "content_sha256": attribution["content_sha256"],
                "transform_sha256": attribution["transform_sha256"],
            },
            "quality": {
                "pii_scan": "passed",
                "review_method": (
                    "oasst1_upstream_human_rank_and_quality_plus_lumi_filters_v0.1"
                ),
                "review_status": "accepted",
                "score": candidate["answer_quality"],
                "secret_scan": "passed",
            },
            "record_id": candidate["record_id"],
            "safety": {
                "policy_version": "lumi-sft-basic-pattern-scan-v0.1",
                "risk_labels": [],
            },
            "schema_version": "lumi.sft.v0.1",
            "source": {
                "dataset_id": SOURCE_DATASET_ID,
                "document_id": answer_message_id,
                "license_id": SOURCE_LICENSE,
                "revision": SOURCE_REVISION,
                "source_id": SOURCE_ID,
                "uri": (
                    f"https://huggingface.co/datasets/{SOURCE_DATASET_ID}/blob/"
                    f"{SOURCE_REVISION}/{object_path}#row="
                    f"{attribution['answer_row_index']}"
                ),
            },
            "split": candidate["split"],
            "task_category": candidate["task_category"],
        }
        records.append(record)
        attributions.append(attribution)
    return records, attributions


def _nearest_rank(values: Sequence[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def measure_token_fit(
    records: Iterable[Mapping[str, Any]],
    *,
    template: ChatTemplate,
    tokenizer: ChatTokenizer,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Reject overlength records and report assistant-only token measurements."""

    accepted: list[Mapping[str, Any]] = []
    lengths: list[int] = []
    supervised: list[int] = []
    masked: list[int] = []
    by_split: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_task: defaultdict[str, Counter[str]] = defaultdict(Counter)
    overlength = 0
    token_ledger: list[dict[str, Any]] = []
    for record in records:
        try:
            encoded = encode_sft_messages(
                record["messages"],
                tokenizer=tokenizer,
                template=template,
            )
        except ValueError as error:
            if "maximum is 4096" not in str(error):
                raise
            overlength += 1
            continue
        total = len(encoded.input_ids)
        accepted.append(record)
        lengths.append(total)
        supervised.append(encoded.supervised_tokens)
        masked.append(encoded.masked_tokens)
        for bucket in (by_split[str(record["split"])], by_task[str(record["task_category"])]):
            bucket["records"] += 1
            bucket["serialized_tokens"] += total
            bucket["supervised_tokens"] += encoded.supervised_tokens
            bucket["masked_tokens"] += encoded.masked_tokens
        token_ledger.append(
            {
                "masked_tokens": encoded.masked_tokens,
                "record_id": record["record_id"],
                "serialized_tokens": total,
                "supervised_tokens": encoded.supervised_tokens,
            }
        )
    metrics = {
        "accepted_records": len(accepted),
        "assistant_only_loss_verified": True,
        "by_split": {key: dict(value) for key, value in sorted(by_split.items())},
        "by_task": {key: dict(value) for key, value in sorted(by_task.items())},
        "context_window": template.context_window,
        "masked_tokens": sum(masked),
        "overlength_rejected": overlength,
        "serialized_length": {
            "maximum": max(lengths, default=0),
            "mean": mean(lengths) if lengths else 0,
            "median": median(lengths) if lengths else 0,
            "minimum": min(lengths, default=0),
            "p95": _nearest_rank(lengths, 0.95),
        },
        "serialized_tokens": sum(lengths),
        "supervised_tokens": sum(supervised),
        "token_ledger_sha256": sha256_value(sorted(token_ledger, key=lambda row: row["record_id"])),
    }
    return accepted, metrics


def deterministic_owner_sample(
    records: Sequence[Mapping[str, Any]],
    *,
    maximum_records: int,
) -> list[Mapping[str, Any]]:
    """Select a deterministic, task/split-interleaved private owner sample."""

    if maximum_records < 1:
        raise ValueError("owner sample size must be positive")
    buckets: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[(str(record["split"]), str(record["task_category"]))].append(record)
    for rows in buckets.values():
        rows.sort(key=lambda row: hashlib.sha256(str(row["record_id"]).encode()).hexdigest())
    selected: list[Mapping[str, Any]] = []
    keys = sorted(buckets)
    offset = 0
    while len(selected) < min(maximum_records, len(records)):
        added = False
        for key in keys:
            rows = buckets[key]
            if offset < len(rows):
                selected.append(rows[offset])
                added = True
                if len(selected) == min(maximum_records, len(records)):
                    break
        if not added:
            break
        offset += 1
    return selected


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


def artifact_inventory(paths: Mapping[str, Path]) -> dict[str, Any]:
    inventory = []
    for name, path in sorted(paths.items()):
        inventory.append(
            {
                "bytes": path.stat().st_size,
                "contains_raw_training_text": name in {"accepted_records", "owner_sample"},
                "name": name,
                "sha256": file_sha256(path),
            }
        )
    return {
        "objects": inventory,
        "total_bytes": sum(item["bytes"] for item in inventory),
    }


def build_raw_text_free_report(
    *,
    acquisition: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    filter_report: Mapping[str, Any],
    owner_sample: Sequence[Mapping[str, Any]],
    plan_file_sha256: str,
    selections: Mapping[str, OASST1Selection],
    token_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the non-production receipt without copying any source text."""

    selection_metrics = {
        split: selection.metrics for split, selection in sorted(selections.items())
    }
    owner_ids = sorted(str(row["record_id"]) for row in owner_sample)
    report = {
        "acquisition": dict(acquisition),
        "artifacts": dict(artifacts),
        "authorization": dict(EXPECTED_AUTHORIZATION),
        "curation": {
            "filter": {
                "accepted": filter_report["accepted"],
                "accepted_dataset_sha256": filter_report["accepted_dataset_sha256"],
                "raw_text_retained": False,
                "rejected": filter_report["rejected"],
                "rejection_counts": filter_report["rejection_counts"],
                "report_sha256": filter_report["report_sha256"],
                "total": filter_report["total"],
            },
            "selection": selection_metrics,
            "token_measurement": dict(token_metrics),
        },
        "observed_at": "2026-08-06",
        "owner_review": {
            "complete": False,
            "record_count": len(owner_sample),
            "sample_record_ids_sha256": sha256_value(owner_ids),
            "status": "pending_owner_review",
        },
        "plan_file_sha256": _require_sha256(plan_file_sha256, "plan_file_sha256"),
        "production": {
            "dataset_promotion_authorized": False,
            "dataset_seal_authorized": False,
            "production_eligible": False,
            "sft_training_authorized": False,
        },
        "raw_training_text_present": False,
        "remaining_blockers": [
            "deterministic_owner_quality_sample_not_reviewed",
            "source_registry_production_promotion_not_authorized",
            "production_dataset_seal_not_authorized",
            "production_sft_not_authorized",
        ],
        "schema_version": REPORT_SCHEMA_VERSION,
        "source": {
            "dataset_id": SOURCE_DATASET_ID,
            "license_id": SOURCE_LICENSE,
            "revision": SOURCE_REVISION,
            "source_id": SOURCE_ID,
        },
        "status": "curated_nonproduction_owner_review_pending",
    }
    report["report_sha256"] = sha256_value(report)
    return report


def load_runtime_components(
    plan: Mapping[str, Any],
    *,
    repository_root: Path,
) -> tuple[Any, Any, BenchmarkRegistry, ChatTemplate, ChatTokenizer]:
    """Load exact policy, source registry, benchmark registry, template and tokenizer."""

    from transformers import AutoTokenizer

    bindings = plan["bindings"]
    contract_path = _resolve_workspace_path(repository_root, bindings["data_contract"])
    source_registry_path = _resolve_workspace_path(repository_root, bindings["source_registry"])
    benchmark_path = _resolve_workspace_path(repository_root, bindings["benchmark_registry"])
    template_path = _resolve_workspace_path(repository_root, bindings["chat_template"])
    tokenizer_path = _resolve_workspace_path(repository_root, bindings["tokenizer_json"])
    tokenizer_config = _resolve_workspace_path(repository_root, bindings["tokenizer_config"])
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
