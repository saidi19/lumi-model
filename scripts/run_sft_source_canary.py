#!/usr/bin/env python3
"""Run a bounded, CPU-only Lumi SFT source canary without logging raw text."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import certifi
from transformers import AutoTokenizer

from lumi.data_quality import BenchmarkRegistry
from lumi.posttraining_canary import (
    CANARY_REPORT_SCHEMA_VERSION,
    adapt_aya_bengali_rows,
    adapt_oasst1_rows,
    adapt_openr1_rows,
    bengali_script_metrics,
    distribution,
    encode_canary_records,
    file_sha256,
    finalize_canary_report,
    load_canary_plan,
    sha256_bytes,
)
from lumi.posttraining_data import filter_sft_records, load_sft_policy
from lumi.posttraining_format import load_chat_template, verify_chat_tokenizer
from lumi.posttraining_sources import load_sft_source_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/lumi-4b-sft-source-canary-v0.1.json"),
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    return parser.parse_args()


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_json_atomic(path: Path, value: Any) -> None:
    _write_bytes_atomic(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n",
    )


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    payload = b"".join(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for record in records
    )
    _write_bytes_atomic(path, payload)


class BoundedFetcher:
    def __init__(self, *, maximum_single: int, maximum_total: int, timeout: int) -> None:
        self.maximum_single = maximum_single
        self.maximum_total = maximum_total
        self.timeout = timeout
        self.total_bytes = 0
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

    def get_json(self, url: str) -> tuple[dict[str, Any], bytes]:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "lumi-sft-source-canary-v0.1"},
        )
        body: bytes | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(  # noqa: S310
                    request,
                    context=self.ssl_context,
                    timeout=self.timeout,
                ) as response:
                    content_length = response.headers.get("Content-Length")
                    if (
                        content_length is not None
                        and int(content_length) > self.maximum_single
                    ):
                        raise ValueError("source response exceeds the single-response byte cap")
                    body = response.read(self.maximum_single + 1)
                break
            except urllib.error.HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise
                time.sleep(2**attempt)
        if body is None:  # pragma: no cover - closed by success or raised error above
            raise RuntimeError("source retrieval ended without a response")
        return self.admit_json(body)

    def admit_json(self, body: bytes) -> tuple[dict[str, Any], bytes]:
        if len(body) > self.maximum_single:
            raise ValueError("source response exceeds the single-response byte cap")
        if self.total_bytes + len(body) > self.maximum_total:
            raise ValueError("source responses exceed the total byte cap")
        self.total_bytes += len(body)
        value = json.loads(body)
        if not isinstance(value, dict):
            raise ValueError("source endpoint did not return a JSON object")
        return value, body


def _load_or_fetch_json(
    *,
    destination: Path,
    fetcher: BoundedFetcher,
    resume_source: Path | None,
    url: str,
) -> tuple[dict[str, Any], bytes, bool]:
    if resume_source is not None and resume_source.is_file():
        value, body = fetcher.admit_json(resume_source.read_bytes())
        _write_bytes_atomic(destination, body)
        return value, body, True
    value, body = fetcher.get_json(url)
    _write_bytes_atomic(destination, body)
    return value, body, False


def _metadata_url(dataset_id: str) -> str:
    return "https://huggingface.co/api/datasets/" + urllib.parse.quote(
        dataset_id,
        safe="/",
    )


def _rows_url(source: dict[str, Any]) -> str:
    query = source["query"]
    parameters: dict[str, Any] = {
        "config": source["config"],
        "dataset": source["dataset_id"],
        "length": source["maximum_fetched_rows"],
        "offset": query["offset"],
        "split": source["split"],
    }
    kind = query["kind"]
    if kind == "filter":
        parameters["where"] = query["where"]
    elif kind != "rows":
        raise ValueError("unsupported source query kind")
    return f"https://datasets-server.huggingface.co/{kind}?{urllib.parse.urlencode(parameters)}"


def _verify_metadata(metadata: dict[str, Any], source: dict[str, Any]) -> None:
    if metadata.get("id") != source["dataset_id"]:
        raise ValueError("source metadata dataset identity changed")
    if metadata.get("sha") != source["revision"]:
        raise ValueError("source head no longer matches the pinned revision")
    if metadata.get("private") is not False or metadata.get("gated") not in {False, None}:
        raise ValueError("source access boundary changed")
    if metadata.get("disabled") is not False:
        raise ValueError("source is disabled")


def _adapt(source: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    common = {
        "revision": source["revision"],
        "target_records": source["target_records"],
    }
    if source["source_id"] == "openassistant_oasst1":
        return adapt_oasst1_rows(rows, **common)
    if source["source_id"] == "coherelabs_aya_dataset":
        return adapt_aya_bengali_rows(rows, **common)
    if source["source_id"] == "openr1_math_220k":
        return adapt_openr1_rows(rows, **common)
    raise ValueError("unsupported source adapter")


def _ensure_output_is_not_in_git(output: Path, repository_root: Path) -> None:
    repository = repository_root.resolve()
    resolved = output.resolve()
    if resolved == repository or repository in resolved.parents:
        raise ValueError("raw-text canary artifacts may not be stored in the Git repository")


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else repository_root / args.plan
    plan = load_canary_plan(plan_path, repository_root=repository_root)
    plan_sha256 = file_sha256(plan_path)
    plan_root = args.artifact_root.resolve() / plan_sha256
    _ensure_output_is_not_in_git(plan_root, repository_root)
    plan_root.mkdir(parents=True, exist_ok=True)
    output: Path | None = None
    for attempt in range(1, 1000):
        candidate = plan_root / f"attempt-{attempt:03d}"
        if not candidate.exists():
            output = candidate
            break
    if output is None:  # pragma: no cover - bounded above and practically unreachable
        raise RuntimeError("canary attempt namespace is exhausted")
    output.mkdir(parents=True)
    raw_directory = output / "raw"
    raw_directory.mkdir()
    shutil.copyfile(plan_path, output / "canary-plan.json")
    resume_raw: Path | None = None
    if args.resume_from is not None:
        resume_raw = args.resume_from.resolve() / "raw"
        _ensure_output_is_not_in_git(resume_raw, repository_root)
        if not resume_raw.is_dir():
            raise ValueError("resume attempt does not contain a raw response directory")

    tokenizer_path = args.tokenizer_dir / "tokenizer.json"
    if file_sha256(tokenizer_path) != plan["tokenizer"]["required_sha256"]:
        raise ValueError("frozen tokenizer SHA-256 changed")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)

    sft_contract = repository_root / plan["bindings"]["sft_contract"]["path"]
    policy = load_sft_policy(sft_contract)
    source_registry = load_sft_source_registry(
        repository_root / policy.source_registry_path,
        allowed_licenses=policy.allowed_licenses,
        allowed_languages=policy.allowed_languages,
        allowed_splits=policy.allowed_splits,
        allowed_tasks=frozenset(policy.task_mix),
        expected_sha256=policy.source_registry_sha256,
    )
    benchmark_registry = BenchmarkRegistry(policy.benchmark_decontamination, repository_root)
    chat_template_path = repository_root / plan["bindings"]["chat_template"]["path"]
    chat_template = load_chat_template(chat_template_path)
    verify_chat_tokenizer(tokenizer, chat_template)

    limits = plan["limits"]
    fetcher = BoundedFetcher(
        maximum_single=limits["maximum_single_response_bytes"],
        maximum_total=limits["maximum_total_response_bytes"],
        timeout=args.timeout_seconds,
    )
    candidates: list[dict[str, Any]] = []
    retrieval: dict[str, Any] = {}
    initial_heads: dict[str, str] = {}
    for source in plan["sources"]:
        source_id = source["source_id"]
        pre_name = f"{source_id}-metadata-pre.json"
        metadata, metadata_bytes, metadata_resumed = _load_or_fetch_json(
            destination=raw_directory / pre_name,
            fetcher=fetcher,
            resume_source=resume_raw / pre_name if resume_raw is not None else None,
            url=_metadata_url(source["dataset_id"]),
        )
        _verify_metadata(metadata, source)
        initial_heads[source_id] = metadata["sha"]

        rows_name = f"{source_id}-rows.json"
        rows_value, rows_bytes, rows_resumed = _load_or_fetch_json(
            destination=raw_directory / rows_name,
            fetcher=fetcher,
            resume_source=resume_raw / rows_name if resume_raw is not None else None,
            url=_rows_url(source),
        )
        rows = rows_value.get("rows")
        if not isinstance(rows, list):
            raise ValueError("source row endpoint omitted rows")
        if len(rows) > source["maximum_fetched_rows"]:
            raise ValueError("source row endpoint exceeded the row cap")
        if rows_value.get("partial") is not False:
            raise ValueError("source row endpoint returned a partial result")
        converted = _adapt(source, rows)
        candidates.extend(converted)
        retrieval[source_id] = {
            "candidate_records": len(converted),
            "dataset_head": metadata["sha"],
            "metadata_bytes": len(metadata_bytes),
            "metadata_sha256": sha256_bytes(metadata_bytes),
            "metadata_resumed": metadata_resumed,
            "response_bytes": len(rows_bytes),
            "response_resumed": rows_resumed,
            "response_sha256": sha256_bytes(rows_bytes),
            "rows_fetched": len(rows),
            "rows_total": rows_value.get("num_rows_total"),
        }

    for source in plan["sources"]:
        source_id = source["source_id"]
        metadata, metadata_bytes = fetcher.get_json(_metadata_url(source["dataset_id"]))
        _verify_metadata(metadata, source)
        if metadata["sha"] != initial_heads[source_id]:
            raise ValueError("source head changed during the bounded canary")
        _write_bytes_atomic(raw_directory / f"{source_id}-metadata-post.json", metadata_bytes)
        retrieval[source_id]["post_metadata_bytes"] = len(metadata_bytes)
        retrieval[source_id]["post_metadata_sha256"] = sha256_bytes(metadata_bytes)

    filtered, filter_report = filter_sft_records(
        candidates,
        policy=policy,
        benchmark_registry=benchmark_registry,
        source_registry=source_registry,
    )
    encoded, encoding_report = encode_canary_records(
        filtered,
        tokenizer=tokenizer,
        template=chat_template,
    )
    accepted = [dict(record) for record in encoded]
    source_encoding: dict[str, Any] = {}
    for source_id in sorted(distribution(accepted, "source_id")):
        source_records = [
            record
            for record in accepted
            if record.get("source", {}).get("source_id") == source_id
        ]
        _, source_encoding[source_id] = encode_canary_records(
            source_records,
            tokenizer=tokenizer,
            template=chat_template,
        )
    accepted_path = output / "accepted-canary.jsonl"
    _write_jsonl_atomic(accepted_path, accepted)

    implementation_files = [
        repository_root / "scripts" / "run_sft_source_canary.py",
        repository_root / "src" / "lumi" / "posttraining_canary.py",
    ]
    report = finalize_canary_report(
        {
            "accepted_artifact": {
                "bytes": accepted_path.stat().st_size,
                "contains_raw_text": True,
                "relative_path": accepted_path.name,
                "sha256": file_sha256(accepted_path),
            },
            "authorization": plan["authorization"],
            "bindings": {
                "benchmark_registry_sha256": policy.benchmark_decontamination[
                    "registry_sha256"
                ],
                "chat_template_sha256": file_sha256(chat_template_path),
                "final_base_binding_sha256": plan["bindings"]["final_base_binding"][
                    "sha256"
                ],
                "plan_sha256": plan_sha256,
                "sft_contract_sha256": file_sha256(sft_contract),
                "source_registry_sha256": source_registry.registry_sha256,
                "tokenizer_sha256": file_sha256(tokenizer_path),
            },
            "candidate_distributions": {
                "language": distribution(candidates, "language"),
                "source": distribution(candidates, "source_id"),
                "task": distribution(candidates, "task_category"),
            },
            "bengali_script": bengali_script_metrics(accepted),
            "encoding": encoding_report,
            "filter": filter_report,
            "final_distributions": {
                "language": distribution(accepted, "language"),
                "source": distribution(accepted, "source_id"),
                "task": distribution(accepted, "task_category"),
            },
            "implementation": {
                path.relative_to(repository_root).as_posix(): file_sha256(path)
                for path in implementation_files
            },
            "package_versions": {
                "tokenizers": importlib.metadata.version("tokenizers"),
                "transformers": importlib.metadata.version("transformers"),
            },
            "production": {
                "eligible": False,
                "seal_attempted": False,
                "blockers": [
                    "production_sft_not_authorized",
                    "sources_not_production_eligible",
                    "human_quality_and_safety_review_not_complete",
                    "task_mix_not_filled",
                ],
            },
            "raw_source_artifacts": {
                "directory": "raw",
                "git_storage_allowed": False,
                "response_bytes_total": fetcher.total_bytes,
            },
            "raw_text_retained": False,
            "retrieval": retrieval,
            "schema_version": CANARY_REPORT_SCHEMA_VERSION,
            "source_encoding": source_encoding,
        }
    )
    report_path = output / "canary-report.json"
    _write_json_atomic(report_path, report)
    print(
        "PASS: bounded CPU-only SFT source canary; "
        f"accepted={len(accepted)}/{len(candidates)}; "
        f"report={report['report_sha256']}; artifact={output}"
    )


if __name__ == "__main__":
    main()
