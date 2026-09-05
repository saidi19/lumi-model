#!/usr/bin/env python3
"""Fetch and audit only the pinned full Aya Bengali subset within hard caps."""

from __future__ import annotations

import argparse
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import certifi

from lumi.posttraining_source_quality import (
    build_aya_bengali_source_audit_report,
    file_sha256,
    load_aya_bengali_audit_plan,
)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class BoundedJsonFetcher:
    def __init__(
        self,
        *,
        maximum_responses: int,
        maximum_single_bytes: int,
        maximum_total_bytes: int,
        timeout_seconds: int,
    ) -> None:
        self.maximum_responses = maximum_responses
        self.maximum_single_bytes = maximum_single_bytes
        self.maximum_total_bytes = maximum_total_bytes
        self.timeout_seconds = timeout_seconds
        self.response_bytes = 0
        self.response_count = 0
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

    def get_json(self, url: str) -> tuple[dict[str, Any], bytes]:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "lumi-sft-aya-bengali-source-audit-v0.1"},
        )
        for attempt in range(3):
            if self.response_count >= self.maximum_responses:
                raise ValueError("Aya Bengali network response cap reached")
            remaining = self.maximum_total_bytes - self.response_bytes
            if remaining <= 0:
                raise ValueError("Aya Bengali total network byte cap reached")
            hard_read_cap = min(self.maximum_single_bytes, remaining)
            try:
                with urllib.request.urlopen(  # noqa: S310
                    request,
                    context=self.ssl_context,
                    timeout=self.timeout_seconds,
                ) as response:
                    self.response_count += 1
                    content_length = response.headers.get("Content-Length")
                    if content_length is not None and int(content_length) > hard_read_cap:
                        raise ValueError("Aya Bengali response exceeds a byte cap")
                    body = response.read(hard_read_cap + 1)
            except urllib.error.HTTPError as error:
                self.response_count += 1
                error_body = error.read(hard_read_cap + 1)
                if len(error_body) > hard_read_cap:
                    raise ValueError("Aya Bengali error response exceeds a byte cap") from error
                self.response_bytes += len(error_body)
                if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise
                time.sleep(2**attempt)
                continue
            except TimeoutError:
                self.response_count += 1
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
                continue
            if len(body) > hard_read_cap:
                raise ValueError("Aya Bengali response exceeds a byte cap")
            self.response_bytes += len(body)
            value = json.loads(body)
            if not isinstance(value, dict):
                raise ValueError("Aya Bengali endpoint did not return a JSON object")
            return value, body
        raise RuntimeError("Aya Bengali retrieval exhausted bounded retries")


def _metadata_url(dataset_id: str) -> str:
    return "https://huggingface.co/api/datasets/" + urllib.parse.quote(
        dataset_id,
        safe="/",
    )


def _rows_url(source: dict[str, Any], *, length: int, offset: int) -> str:
    parameters = {
        "config": source["config"],
        "dataset": source["dataset_id"],
        "length": length,
        "offset": offset,
        "split": source["split"],
        "where": source["filter"],
    }
    return "https://datasets-server.huggingface.co/filter?" + urllib.parse.urlencode(
        parameters
    )


def _verify_metadata(metadata: dict[str, Any], source: dict[str, Any]) -> None:
    if metadata.get("id") != source["dataset_id"]:
        raise ValueError("Aya Bengali dataset identity changed")
    if metadata.get("sha") != source["revision"]:
        raise ValueError("Aya Bengali dataset head differs from the pinned revision")
    if metadata.get("private") is not False:
        raise ValueError("Aya Bengali dataset privacy boundary changed")
    if metadata.get("gated") not in {False, None}:
        raise ValueError("Aya Bengali dataset became gated")
    if metadata.get("disabled") is not False:
        raise ValueError("Aya Bengali dataset is disabled")


def _ensure_outside_repository(path: Path, repository_root: Path) -> None:
    resolved = path.resolve()
    root = repository_root.resolve()
    if resolved == root or root in resolved.parents:
        raise ValueError("Aya Bengali raw pages must remain outside the Git repository")


def _next_attempt(plan_root: Path) -> Path:
    plan_root.mkdir(parents=True, exist_ok=True)
    for number in range(1, 1000):
        attempt = plan_root / f"attempt-{number:03d}"
        if not attempt.exists():
            attempt.mkdir()
            return attempt
    raise RuntimeError("Aya Bengali audit attempt namespace is exhausted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    args = parser.parse_args()

    root = args.repository_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    evidence_output = (
        args.evidence_output
        if args.evidence_output.is_absolute()
        else root / args.evidence_output
    )
    if evidence_output.exists():
        raise FileExistsError(f"refusing to overwrite Aya evidence: {evidence_output}")
    if not 1 <= args.timeout_seconds <= 60:
        raise ValueError("Aya Bengali timeout must be between 1 and 60 seconds")

    plan = load_aya_bengali_audit_plan(plan_path, repository_root=root)
    plan_sha256 = file_sha256(plan_path)
    artifact_root = args.artifact_root.resolve()
    _ensure_outside_repository(artifact_root, root)
    attempt = _next_attempt(artifact_root / plan_sha256)
    raw_directory = attempt / "raw"
    raw_directory.mkdir()
    _write_bytes_atomic(attempt / "audit-plan.json", plan_path.read_bytes())

    limits = plan["limits"]
    source = plan["source"]
    fetcher = BoundedJsonFetcher(
        maximum_responses=limits["maximum_network_responses"],
        maximum_single_bytes=limits["maximum_single_response_bytes"],
        maximum_total_bytes=limits["maximum_total_response_bytes"],
        timeout_seconds=args.timeout_seconds,
    )

    metadata_pre, metadata_pre_bytes = fetcher.get_json(
        _metadata_url(source["dataset_id"])
    )
    _verify_metadata(metadata_pre, source)
    _write_bytes_atomic(raw_directory / "aya-metadata-pre.json", metadata_pre_bytes)

    expected_rows = limits["expected_filtered_rows"]
    fetched_rows: list[dict[str, Any]] = []
    page_inventory: list[dict[str, Any]] = []
    offset = 0
    if args.resume_from is not None:
        resume_attempt = args.resume_from.resolve()
        _ensure_outside_repository(resume_attempt, root)
        resume_plan_path = resume_attempt / "audit-plan.json"
        resume_raw = resume_attempt / "raw"
        if not resume_plan_path.is_file() or not resume_raw.is_dir():
            raise ValueError("Aya Bengali resume attempt is incomplete")
        resume_plan = json.loads(resume_plan_path.read_text(encoding="utf-8"))
        for key in ("authorization", "criteria", "output", "source"):
            if resume_plan.get(key) != plan.get(key):
                raise ValueError(f"Aya Bengali resume {key} boundary changed")
        resume_limits = resume_plan.get("limits")
        if not isinstance(resume_limits, dict):
            raise ValueError("Aya Bengali resume limits are missing")
        for key in (
            "expected_filtered_rows",
            "maximum_fetched_rows",
            "maximum_single_response_bytes",
            "maximum_total_response_bytes",
        ):
            if resume_limits.get(key) != limits.get(key):
                raise ValueError(f"Aya Bengali resume limit changed: {key}")
        resume_metadata = json.loads(
            (resume_raw / "aya-metadata-pre.json").read_text(encoding="utf-8")
        )
        if not isinstance(resume_metadata, dict):
            raise ValueError("Aya Bengali resume metadata is invalid")
        _verify_metadata(resume_metadata, source)

        resume_pages = sorted(resume_raw.glob("aya-bengali-page-*.json"))
        if not resume_pages:
            raise ValueError("Aya Bengali resume attempt has no complete pages")
        first_resume_rows: list[dict[str, Any]] | None = None
        for page_number, resume_page in enumerate(resume_pages):
            if resume_page.name != f"aya-bengali-page-{page_number:03d}.json":
                raise ValueError("Aya Bengali resume page sequence changed")
            page_bytes = resume_page.read_bytes()
            if len(page_bytes) > limits["maximum_single_response_bytes"]:
                raise ValueError("Aya Bengali resume page exceeds the byte cap")
            page = json.loads(page_bytes)
            if not isinstance(page, dict) or page.get("partial") is not False:
                raise ValueError("Aya Bengali resume page is invalid")
            if page.get("num_rows_total") != expected_rows:
                raise ValueError("Aya Bengali resume filtered row total changed")
            rows = page.get("rows")
            if not isinstance(rows, list) or not 1 <= len(rows) <= 100:
                raise ValueError("Aya Bengali resume page row count is invalid")
            for wrapped in rows:
                row = wrapped.get("row") if isinstance(wrapped, dict) else None
                if not isinstance(row, dict) or row.get("language_code") != "ben":
                    raise ValueError("Aya Bengali resume page contains an unrelated row")
            if page_number == 0:
                first_resume_rows = rows
            filename = f"aya-bengali-page-{page_number:03d}.json"
            destination = raw_directory / filename
            _write_bytes_atomic(destination, page_bytes)
            page_inventory.append(
                {
                    "filename": filename,
                    "offset": offset,
                    "rows": len(rows),
                    "sha256": file_sha256(destination),
                    "size_bytes": len(page_bytes),
                }
            )
            fetched_rows.extend(rows)
            offset += len(rows)

        if first_resume_rows is None:  # pragma: no cover - guarded by resume_pages
            raise RuntimeError("Aya Bengali resume first page is missing")
        probe, _ = fetcher.get_json(
            _rows_url(source, length=len(first_resume_rows), offset=0)
        )
        if probe.get("num_rows_total") != expected_rows or probe.get("rows") != first_resume_rows:
            raise ValueError("Aya Bengali filtered view changed since the resume attempt")
    while offset < expected_rows:
        requested = min(limits["page_size"], expected_rows - offset)
        page, page_bytes = fetcher.get_json(
            _rows_url(source, length=requested, offset=offset)
        )
        if page.get("partial") is not False:
            raise ValueError("Aya Bengali dataset server returned a partial page")
        if page.get("num_rows_total") != expected_rows:
            raise ValueError("Aya Bengali filtered row total changed")
        rows = page.get("rows")
        if not isinstance(rows, list) or len(rows) != requested:
            raise ValueError("Aya Bengali page did not contain the requested rows")
        for wrapped in rows:
            if not isinstance(wrapped, dict):
                raise ValueError("Aya Bengali row wrapper is invalid")
            row = wrapped.get("row")
            if not isinstance(row, dict) or row.get("language_code") != "ben":
                raise ValueError("Aya Bengali filter returned an unrelated row")

        filename = f"aya-bengali-page-{len(page_inventory):03d}.json"
        page_path = raw_directory / filename
        _write_bytes_atomic(page_path, page_bytes)
        page_inventory.append(
            {
                "filename": filename,
                "offset": offset,
                "rows": len(rows),
                "sha256": file_sha256(page_path),
                "size_bytes": len(page_bytes),
            }
        )
        fetched_rows.extend(rows)
        offset += len(rows)

    metadata_post, metadata_post_bytes = fetcher.get_json(
        _metadata_url(source["dataset_id"])
    )
    _verify_metadata(metadata_post, source)
    if metadata_post.get("sha") != metadata_pre.get("sha"):
        raise ValueError("Aya Bengali dataset head changed during retrieval")
    _write_bytes_atomic(raw_directory / "aya-metadata-post.json", metadata_post_bytes)

    report = build_aya_bengali_source_audit_report(
        expected_rows=expected_rows,
        maximum_top_annotator_fraction=plan["criteria"][
            "maximum_top_annotator_fraction"
        ],
        metadata_post_sha256=file_sha256(raw_directory / "aya-metadata-post.json"),
        metadata_pre_sha256=file_sha256(raw_directory / "aya-metadata-pre.json"),
        observed_at=plan["observed_at"],
        page_inventory=page_inventory,
        plan_sha256=plan_sha256,
        response_bytes=fetcher.response_bytes,
        response_count=fetcher.response_count,
        source_revision=source["revision"],
        wrapped_rows=fetched_rows,
    )
    inventory = {
        "metadata_post_sha256": report["bindings"]["metadata_post_sha256"],
        "metadata_pre_sha256": report["bindings"]["metadata_pre_sha256"],
        "page_inventory": page_inventory,
        "plan_sha256": plan_sha256,
        "raw_text_retained": False,
        "response_bytes": fetcher.response_bytes,
        "response_count": fetcher.response_count,
        "rows": len(fetched_rows),
    }
    _write_bytes_atomic(attempt / "inventory.json", _json_bytes(inventory))
    _write_bytes_atomic(attempt / "audit-report.json", _json_bytes(report))
    _write_bytes_atomic(evidence_output, _json_bytes(report))
    print(
        "PASS: Aya Bengali full-subset machine audit completed; "
        f"rows={report['source_rows']} "
        f"bytes={fetcher.response_bytes} "
        f"blockers={len(report['remaining_production_blockers'])} "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
