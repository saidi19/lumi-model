#!/usr/bin/env python3
"""Scan a JSONL sample for residual PII without recording sensitive values."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumi.data_pipeline import PII_DETECTOR_VERSION, detect_pii


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument(
        "--start-line",
        type=int,
        default=0,
        help="number of records before --start-offset",
    )
    parser.add_argument("--progress-every", type=int, default=10_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample_size = args.sample.stat().st_size
    if not 0 <= args.start_offset <= sample_size:
        raise ValueError("start offset is outside the sample")
    if args.start_line < 0:
        raise ValueError("start line cannot be negative")
    if args.progress_every < 1:
        raise ValueError("progress interval must be positive")

    matches: list[dict[str, Any]] = []
    kinds: Counter[str] = Counter()
    scanned_records = 0
    scanned_bytes = 0
    started_at = _utc_now()
    with args.sample.open("rb") as handle:
        if args.start_offset:
            handle.seek(args.start_offset - 1)
            if handle.read(1) != b"\n":
                raise ValueError("start offset is not immediately after a JSONL record")
        handle.seek(args.start_offset)
        while raw_line := handle.readline():
            scanned_records += 1
            scanned_bytes += len(raw_line)
            line_number = args.start_line + scanned_records
            try:
                record = json.loads(raw_line)
                text = record["text"]
                provenance = record["provenance"]
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"invalid sample record at line {line_number}: {error}") from error
            if not isinstance(text, str):
                raise ValueError(f"sample text is not a string at line {line_number}")
            kind = detect_pii(text)
            if kind is not None:
                kinds[kind] += 1
                matches.append(
                    {
                        "category": provenance.get("category"),
                        "component": provenance.get("component"),
                        "document_id": record.get("id"),
                        "kind": kind,
                        "line_number": line_number,
                        "source_file": provenance.get("source_file"),
                        "source_line": provenance.get("source_line"),
                    }
                )
            if scanned_records % args.progress_every == 0:
                percent = 100.0 * (args.start_offset + scanned_bytes) / max(sample_size, 1)
                print(
                    f"audit line={line_number:,} file={percent:.1f}% residual={len(matches):,}",
                    file=sys.stderr,
                    flush=True,
                )

    report = {
        "completed_at": _utc_now(),
        "kinds": dict(sorted(kinds.items())),
        "matches": matches,
        "pii_detector": PII_DETECTOR_VERSION,
        "sample": str(args.sample.resolve()),
        "sample_file_bytes": sample_size,
        "scanned_bytes": scanned_bytes,
        "scanned_records": scanned_records,
        "start_line": args.start_line,
        "start_offset": args.start_offset,
        "started_at": started_at,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json_write(args.report, report)
    print(
        f"audit complete: records={scanned_records:,} residual={len(matches):,} "
        f"report={args.report}",
        flush=True,
    )
    return 1 if matches else 0


if __name__ == "__main__":
    raise SystemExit(main())
