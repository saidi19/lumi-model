#!/usr/bin/env python3
"""Print resumable Lumi data-build and validation progress."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    args = parser.parse_args()
    summary: dict[str, Any] = {"directory": str(args.sample)}

    run_state_path = args.sample / "run-state.json"
    if run_state_path.is_file():
        run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
        summary["build"] = {
            "network_retries": run_state.get("network_retries", 0),
            "progress": run_state.get("progress", {}),
            "status": run_state.get("status"),
            "updated_at": run_state.get("updated_at"),
        }
    sample_path = args.sample / "sample.jsonl"
    if sample_path.is_file():
        summary["sample_file_bytes"] = sample_path.stat().st_size

    validation_path = args.sample / "validation-state.sqlite3"
    if validation_path.is_file():
        with sqlite3.connect(f"file:{validation_path}?mode=ro", uri=True) as connection:
            row = connection.execute(
                "SELECT status, next_offset, line_number, statistics, updated_at "
                "FROM validation_progress WHERE id = 1"
            ).fetchone()
        if row is not None:
            statistics = json.loads(row[3])
            summary["validation"] = {
                "line_number": row[2],
                "next_offset": row[1],
                "status": row[0],
                "text_utf8_bytes": statistics["total_text_bytes"],
                "updated_at": row[4],
            }

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
