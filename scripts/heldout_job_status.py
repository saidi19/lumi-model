#!/usr/bin/env python3
"""Show durable progress for a Lumi held-out evaluation-data build."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def read_metadata(connection: sqlite3.Connection, key: str) -> dict | None:
    row = connection.execute(
        "SELECT value FROM metadata WHERE key = ?", (key,)
    ).fetchone()
    return None if row is None else json.loads(row[0])


def main() -> None:
    args = parse_args()
    manifest_path = args.output / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        status = {
            "candidate_documents": manifest["statistics"]["candidate_documents"],
            "documents_scanned": manifest["statistics"]["documents_scanned"],
            "documents_selected": manifest["statistics"]["documents_selected"],
            "sequences": manifest["statistics"]["sequences"],
            "status": "complete",
            "tokens": manifest["statistics"]["tokens"],
        }
    else:
        state_path = args.output / "heldout-state.sqlite3"
        if not state_path.is_file():
            status = {"status": "not_started"}
        else:
            uri = f"file:{state_path.resolve()}?mode=ro"
            connection = sqlite3.connect(uri, uri=True, timeout=2)
            try:
                state = read_metadata(connection, "state")
                if state is None:
                    raise ValueError("held-out database has no durable state")
                training_rows = connection.execute(
                    "SELECT COUNT(*) FROM training_documents"
                ).fetchone()[0]
                candidate_rows = connection.execute(
                    "SELECT COUNT(*) FROM candidates"
                ).fetchone()[0]
            finally:
                connection.close()
            source_bytes = (
                args.input.stat().st_size
                if args.input is not None and args.input.is_file()
                else None
            )
            status = {
                "candidate_documents": candidate_rows,
                "documents_scanned": state["documents_seen"],
                "eligible_documents": state["eligible_documents"],
                "input_bytes_scanned": state["input_offset"],
                "input_percent": (
                    100 * state["input_offset"] / source_bytes if source_bytes else None
                ),
                "source_bytes": source_bytes,
                "status": state["status"],
                "training_documents": training_rows,
            }

    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
        return
    if status["status"] == "complete":
        print(
            f"complete: {status['tokens']:,} tokens, {status['sequences']:,} sequences, "
            f"{status['documents_selected']:,} documents selected from "
            f"{status['documents_scanned']:,} scanned"
        )
    elif status["status"] == "not_started":
        print("not started")
    else:
        percent = (
            "unknown"
            if status["input_percent"] is None
            else f"{status['input_percent']:.2f}%"
        )
        print(
            f"{status['status']}: {percent}, {status['documents_scanned']:,} documents, "
            f"{status['candidate_documents']:,} candidates, "
            f"{status['training_documents']:,} training exclusions"
        )


if __name__ == "__main__":
    main()
