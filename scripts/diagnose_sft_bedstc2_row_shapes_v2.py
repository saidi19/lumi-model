#!/usr/bin/env python3
"""One-pass, raw-text-free Be-DSTC2 row-shape diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import signal
import sys
from collections import Counter
from pathlib import Path

from scripts.diagnose_sft_bedstc2_parser_contract_v1 import EXPECTED, RAW_ROOT, TEST_OBJECT

REPORT_PATH = "evidence/lumi-4b-sft-bedstc2-row-shape-diagnostic-v0.5.json"
VERIFICATION_PATH = "evidence/lumi-4b-sft-bedstc2-row-shape-diagnostic-verification-v0.5.json"
AUTHORIZATION_TEXT_SHA256 = (
    "26ea212fd276641452ab1297b6881b3cb9d8a19d2ffb0257368bff0c3d7ebd4b"
)
PREDECESSOR = {
    "file_sha256": "d9e9feb3b0b999c3277f44e470deb3091911db9f305b318ea90a48bcdc447f27",
    "internal_sha256": "dba9bd01e8814b41a39e13cb6f58797586107be13e90e4385d72cef05bc48cfa",
    "path": "evidence/lumi-4b-sft-bedstc2-parser-contract-diagnostic-v0.4.json",
}
_NUMBERED = re.compile(r"^([1-9][0-9]*) (.*)$")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"


def _seal(value: dict[str, object], field: str) -> dict[str, object]:
    sealed = dict(value)
    sealed[field] = _sha(
        json.dumps(sealed, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    )
    return sealed


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def _finish(path: Path, marker: bytes, payload: bytes) -> None:
    if path.read_bytes() != marker:
        raise ValueError("diagnostic marker changed")
    with path.open("r+b") as handle:
        handle.seek(0)
        handle.truncate()
        handle.write(payload)


def _relation_positions(tokens: list[str]) -> tuple[int, ...]:
    return tuple(index for index, token in enumerate(tokens) if token.startswith("R_"))


def scan(path: Path, *, expected: dict[str, object]) -> dict[str, object]:
    counts: Counter[str] = Counter()
    histograms: dict[str, Counter[str]] = {
        "non_tab_relation_positions": Counter(),
        "non_tab_token_counts": Counter(),
        "ordinal_delta": Counter(),
        "pair_field_count": Counter(),
    }
    expected_ordinal = 1
    sha = hashlib.sha256()
    git = hashlib.sha1(
        f"blob {expected['bytes']}\0".encode(), usedforsecurity=False
    )
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if line_number > int(expected["lines"]):
                raise ValueError("shape diagnostic line cap exceeded")
            counts["physical_lines"] += 1
            counts["bytes_read"] += len(raw)
            sha.update(raw)
            git.update(raw)
            line = raw.decode("utf-8")
            line = (
                line[:-2]
                if line.endswith("\r\n")
                else line[:-1]
                if line.endswith("\n")
                else line
            )
            if line == "":
                counts["blank"] += 1
                expected_ordinal = 1
                continue
            if line.startswith((" ", "\t")):
                counts["line_leading_whitespace"] += 1
            if line.endswith((" ", "\t")):
                counts["line_trailing_whitespace"] += 1
            match = _NUMBERED.fullmatch(line.strip())
            if match is None:
                counts["numbering_shape_error"] += 1
                continue
            ordinal = int(match.group(1))
            if ordinal != expected_ordinal:
                histograms["ordinal_delta"][str(ordinal - expected_ordinal)] += 1
            expected_ordinal = ordinal + 1
            body = match.group(2)
            fields = body.split("\t")
            histograms["pair_field_count"][str(len(fields))] += 1
            if len(fields) == 2:
                user, assistant = fields
                counts["tab_pair"] += 1
                counts["pair_user_empty"] += user == ""
                counts["pair_assistant_empty"] += assistant == ""
                counts["pair_user_leading_whitespace"] += user.startswith(" ")
                counts["pair_user_trailing_whitespace"] += user.endswith(" ")
                counts["pair_assistant_leading_whitespace"] += assistant.startswith(" ")
                counts["pair_assistant_trailing_whitespace"] += assistant.endswith(" ")
            elif len(fields) == 1:
                counts["zero_tab"] += 1
                tokens = body.split(" ")
                histograms["non_tab_token_counts"][str(len(tokens))] += 1
                positions = _relation_positions(tokens)
                key = ",".join(str(item) for item in positions) if positions else "none"
                histograms["non_tab_relation_positions"][key] += 1
                counts["zero_tab_has_empty_space_token"] += "" in tokens
                counts["zero_tab_relation_count_not_one"] += len(positions) != 1
                if len(positions) == 1:
                    position = positions[0]
                    counts["zero_tab_relation_at_index_one"] += position == 1
                    counts["zero_tab_missing_entity"] += position == 0
                    counts["zero_tab_missing_value"] += position == len(tokens) - 1
            else:
                counts["multiple_tab"] += 1
    return {
        "counts": dict(sorted(counts.items())),
        "git_blob_sha1": git.hexdigest(),
        "histograms": {
            name: dict(sorted(values.items())) for name, values in sorted(histograms.items())
        },
        "sha256": sha.hexdigest(),
    }


def _alarm(_signum: int, _frame: object) -> None:
    raise TimeoutError("shape diagnostic runtime cap exceeded")


def main() -> int:
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, 30 * 60)
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    root = parser.parse_args().repository_root.resolve()
    report_path = root / REPORT_PATH
    verification_path = root / VERIFICATION_PATH
    if report_path.exists() or verification_path.exists():
        raise FileExistsError("shape diagnostic evidence already exists")
    predecessor_path = root / str(PREDECESSOR["path"])
    predecessor = json.loads(predecessor_path.read_bytes())
    if (
        _sha(predecessor_path.read_bytes()) != PREDECESSOR["file_sha256"]
        or predecessor.get("report_sha256") != PREDECESSOR["internal_sha256"]
        or predecessor.get("hypothesis_verdict") != "REFUTED_OTHER_GRAMMAR_MISMATCH"
    ):
        raise ValueError("predecessor diagnostic changed")
    raw_root = root / RAW_ROOT
    if (raw_root / TEST_OBJECT).exists():
        raise ValueError("prohibited test object present")
    marker = b'{"started":"bedstc2_row_shape_diagnostic_v0.5"}\n'
    _write(report_path, marker)
    results = {
        name: scan(raw_root / str(expected["path"]), expected=expected)
        for name, expected in (("train", EXPECTED["train"]), ("dev", EXPECTED["dev"]))
    }
    for name, result in results.items():
        expected = EXPECTED[name]
        if (
            result["sha256"] != expected["sha256"]
            or result["git_blob_sha1"] != expected["git_blob_sha1"]
            or result["counts"]["physical_lines"] != expected["lines"]
            or result["counts"]["bytes_read"] != expected["bytes"]
        ):
            raise ValueError("shape diagnostic binding changed")
    report = _seal(
        {
            "authorization_text_sha256": AUTHORIZATION_TEXT_SHA256,
            "diagnostic": results,
            "limits": {
                "candidate_row_reads": 0,
                "network_bytes": 0,
                "source_line_reads": 113718,
            },
            "predecessor": PREDECESSOR,
            "raw_text_retained": False,
            "schema_version": "lumi.sft-bedstc2-row-shape-diagnostic.v0.5",
            "status": "PASS_row_shape_diagnostic_complete",
        },
        "report_sha256",
    )
    payload = _json_bytes(report)
    _finish(report_path, marker, payload)
    verification = _seal(
        {
            "checks": {
                "aggregate_only": True,
                "bindings_reproduced": True,
                "candidate_reads_zero": True,
                "raw_text_absent": True,
            },
            "report_file_sha256": _sha(payload),
            "report_internal_sha256": report["report_sha256"],
            "schema_version": "lumi.sft-bedstc2-row-shape-diagnostic-verification.v0.5",
            "status": "PASS_verified_row_shape_diagnostic",
        },
        "verification_sha256",
    )
    _write(verification_path, _json_bytes(verification))
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps({"status": report["status"]}, sort_keys=True))
    return 0


def _entrypoint() -> int:
    try:
        return main()
    except Exception as error:
        print(
            json.dumps(
                {"exception_type": type(error).__name__, "status": "FAIL_CLOSED"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
