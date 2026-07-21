#!/usr/bin/env python3
"""Verify Lingua outputs without persisting any source text."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from collections import Counter
from pathlib import Path

from lingua import LanguageDetectorBuilder

from lumi.lingua_glue_compat import load_lingua_glue_config


def _canonical(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--language-model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_lingua_glue_config(args.config)
    parity = config["parity"]
    if importlib.metadata.version("lingua-language-detector") != "2.2.0":
        raise RuntimeError("Lingua runtime version changed")

    detector = LanguageDetectorBuilder.from_all_spoken_languages().build()
    lumi_digest = hashlib.sha256()
    threshold_digest = hashlib.sha256()
    labels: Counter[str] = Counter()
    paths = sorted(args.language_model_root.glob("*/testdata/*.txt"))
    total = 0
    started = time.monotonic()

    for path in paths:
        relative = path.relative_to(args.language_model_root).as_posix()
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                text = raw_line.rstrip("\n\r")
                language = detector.detect_language_of(text)
                if language is None or language.iso_code_639_1 is None:
                    label = "und"
                    confidence = 0.0
                else:
                    label = language.iso_code_639_1.name.lower()
                    confidence = round(
                        float(detector.compute_language_confidence(text, language)),
                        6,
                    )
                text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
                lumi_row = _canonical(
                    {
                        "confidence_6": confidence,
                        "line": line_number,
                        "label": label,
                        "path": relative,
                        "text_sha256": text_sha256,
                    }
                )
                threshold_row = _canonical(
                    {
                        "line": line_number,
                        "passes_0_7": confidence >= parity["minimum_confidence"],
                        "path": relative,
                        "text_sha256": text_sha256,
                    }
                )
                lumi_digest.update(lumi_row)
                lumi_digest.update(b"\n")
                threshold_digest.update(threshold_row)
                threshold_digest.update(b"\n")
                labels[label] += 1
                total += 1
                if total % 25_000 == 0:
                    print(
                        json.dumps(
                            {
                                "elapsed_seconds": round(time.monotonic() - started, 3),
                                "processed": total,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

    report = {
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "file_count": len(paths),
        "gate": "lumi_lingua_runtime_parity_v0.1",
        "label_counts": dict(sorted(labels.items())),
        "lingua_version": importlib.metadata.version("lingua-language-detector"),
        "lumi_decision_sha256": lumi_digest.hexdigest(),
        "python_version": platform.python_version(),
        "record_count": total,
        "schema_version": 1,
        "threshold_0_7_sha256": threshold_digest.hexdigest(),
    }
    if report["file_count"] != parity["expected_file_count"]:
        raise RuntimeError("Lingua parity file count changed")
    if report["record_count"] != parity["expected_record_count"]:
        raise RuntimeError("Lingua parity record count changed")
    if report["lumi_decision_sha256"] != parity["expected_lumi_decision_sha256"]:
        raise RuntimeError("Lingua Lumi-rounded decisions changed")
    if report["threshold_0_7_sha256"] != parity["expected_threshold_0_7_sha256"]:
        raise RuntimeError("Lingua 0.7 threshold decisions changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
