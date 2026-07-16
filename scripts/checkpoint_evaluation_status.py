#!/usr/bin/env python3
"""Show durable progress for a Lumi checkpoint evaluation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_sequences = json.loads(args.manifest.read_text(encoding="utf-8"))[
        "statistics"
    ]["sequences"]
    report_path = args.output / "evaluation-report.json"
    state_path = args.output / "evaluation-state.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        status = {
            "loss": report["loss"]["mean"],
            "perplexity": report["loss"]["perplexity"],
            "sequences": report["runtime"]["sequences"],
            "status": "complete",
            "tokens_per_second": report["runtime"]["tokens_per_second"],
        }
    elif state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        completed = state["sequence_cursor"]
        seconds_per_sequence = (
            state["forward_seconds"] / completed if completed else None
        )
        status = {
            "eta_seconds": (
                (total_sequences - completed) * seconds_per_sequence
                if seconds_per_sequence is not None
                else None
            ),
            "loss": (
                state["total_negative_log_likelihood"] / state["prediction_tokens"]
                if state["prediction_tokens"]
                else None
            ),
            "percent": 100 * completed / total_sequences,
            "sequences": completed,
            "status": state["status"],
            "tokens_per_second": (
                state["input_tokens"] / state["forward_seconds"]
                if state["forward_seconds"]
                else None
            ),
            "total_sequences": total_sequences,
        }
    else:
        status = {"status": "not_started"}

    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
        return
    if status["status"] == "complete":
        print(
            f"complete: loss={status['loss']:.6f}, "
            f"perplexity={status['perplexity']:.3f}, "
            f"{status['tokens_per_second']:.1f} tokens/s"
        )
    elif status["status"] == "not_started":
        print("not started")
    else:
        eta_minutes = (
            "unknown"
            if status["eta_seconds"] is None
            else f"{status['eta_seconds'] / 60:.1f}m"
        )
        print(
            f"{status['status']}: {status['sequences']:,}/{status['total_sequences']:,} "
            f"({status['percent']:.2f}%), loss={status['loss']:.6f}, "
            f"{status['tokens_per_second']:.1f} tokens/s, ETA {eta_minutes}"
        )


if __name__ == "__main__":
    main()
