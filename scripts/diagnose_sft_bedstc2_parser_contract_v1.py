#!/usr/bin/env python3
"""One-pass, raw-text-free Be-DSTC2 framing diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import signal
import sys
from collections import Counter
from pathlib import Path
from typing import BinaryIO

SCHEMA = "lumi.sft-bedstc2-parser-contract-diagnostic.v0.4"
VERIFICATION_SCHEMA = "lumi.sft-bedstc2-parser-contract-diagnostic-verification.v0.4"
AUTHORIZATION_TEXT_SHA256 = (
    "26ea212fd276641452ab1297b6881b3cb9d8a19d2ffb0257368bff0c3d7ebd4b"
)
REPORT_PATH = "evidence/lumi-4b-sft-bedstc2-parser-contract-diagnostic-v0.4.json"
VERIFICATION_PATH = (
    "evidence/lumi-4b-sft-bedstc2-parser-contract-diagnostic-verification-v0.4.json"
)
RAW_ROOT = (
    "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/attempt-001/private/raw"
)
TEST_OBJECT = "data/bengali/dialog-dstc2-tst.txt"
EXPECTED = {
    "dev": {
        "bytes": 1_361_043,
        "dialogues": 500,
        "git_blob_sha1": "543eb0aa02d9ec7f1af65516017a1bfcace1f636",
        "lines": 28_617,
        "path": "data/bengali/dialog-dstc2-dev.txt",
        "sha256": "a46b98e4f3ff3c9575240fe42fee2d735fa61ec41aacd53d73627b33b7147116",
    },
    "train": {
        "bytes": 4_085_040,
        "dialogues": 1_618,
        "git_blob_sha1": "4683420d14bf546c31e7698ce9ecd799ba0f88d3",
        "lines": 85_101,
        "path": "data/bengali/dialog-dstc2-trn.txt",
        "sha256": "79644ed579c831626b1f2a2dfef1f804b835c9e3da19f93fe493c9fbbb3b591c",
    },
}
STATIC_BINDINGS = {
    "v0_3_implementation": "8cac4260d0b82a220996b5c06ef1d76f7dda7d50b2307009e64faae2c2d97618",
    "v0_3_marker": "a09a86416d42c7d11df6f6694f36f4a0a119da4b4fdd11b2f606354b359f75ae",
    "v0_3_plan": "4db4becc887ddfa64fe8983426bf50cc9ac6f24b050152bbdb26f1f5cb0b570c",
    "v0_3_runner": "6714aa594c40f560690c962df421937427adcb76527d8bee1a704d04b654a8e5",
    "v0_3_test": "4970fbc67b72b6c42e58b55c69beab241c39454b575ebcc2944b2f7ddee33731",
    "v0_3_verifier": "2d2e18965e4987729cc9cadc7a0bc160da4724557ed897d98288068d6de80c06",
}
STATIC_PATHS = {
    "v0_3_implementation": "src/lumi/posttraining_bedstc2_banglish_conversation_curation_v2.py",
    "v0_3_marker": "evidence/lumi-4b-sft-bedstc2-banglish-conversation-curation-v0.3.json",
    "v0_3_plan": "configs/lumi-4b-sft-bedstc2-banglish-conversation-curation-recovery-v0.3.json",
    "v0_3_runner": "scripts/curate_sft_bedstc2_banglish_conversation_v2.py",
    "v0_3_test": "tests/test_bedstc2_banglish_conversation_curation_v2.py",
    "v0_3_verifier": "scripts/verify_sft_bedstc2_banglish_conversation_v2.py",
}

_NUMBERED = re.compile(r"^([1-9][0-9]*) (.*)$")
_KB = re.compile(r"^(\S+) (R_\S+) (\S(?:.*\S)?)$")
_COUNT_KEYS = {
    "api_rows",
    "blank_lines",
    "bytes_read",
    "empty_or_repeated_boundaries",
    "eof_dialogue_boundaries",
    "explicit_dialogue_boundaries",
    "implicit_ordinal_one_boundaries",
    "kb_rows",
    "malformed_non_tab_rows",
    "multiple_tab_rows",
    "natural_pair_rows",
    "numbered_framing_errors",
    "ordinal_errors_after_reset_aware_parse",
    "pair_framing_errors",
    "physical_lines",
    "utf8_errors",
    "whitespace_framing_errors",
}
_ERROR_KEYS = {
    "empty_or_repeated_boundaries",
    "malformed_non_tab_rows",
    "multiple_tab_rows",
    "numbered_framing_errors",
    "ordinal_errors_after_reset_aware_parse",
    "pair_framing_errors",
    "utf8_errors",
    "whitespace_framing_errors",
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )


def _seal(value: dict[str, object], field: str) -> dict[str, object]:
    sealed = dict(value)
    sealed[field] = _sha256(
        json.dumps(sealed, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    )
    return sealed


def _exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()


def _finish_marker(path: Path, marker: bytes, payload: bytes) -> None:
    if path.read_bytes() != marker:
        raise ValueError("diagnostic marker changed")
    with path.open("r+b") as handle:
        handle.seek(0)
        handle.truncate()
        handle.write(payload)
        handle.flush()


def scan_stream(handle: BinaryIO, *, maximum_lines: int, size: int) -> dict[str, object]:
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(f"blob {size}\0".encode(), usedforsecurity=False)
    counts: Counter[str] = Counter()
    expected_ordinal = 1
    entries = False
    first_v0_3_failure_code: str | None = None

    for line_number, raw in enumerate(handle, start=1):
        if line_number > maximum_lines:
            raise ValueError("diagnostic source-line cap exceeded")
        counts["physical_lines"] += 1
        counts["bytes_read"] += len(raw)
        sha256.update(raw)
        git_sha1.update(raw)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            counts["utf8_errors"] += 1
            continue
        line = text[:-2] if text.endswith("\r\n") else text[:-1] if text.endswith("\n") else text
        if line == "":
            counts["blank_lines"] += 1
            if entries:
                counts["explicit_dialogue_boundaries"] += 1
                entries = False
                expected_ordinal = 1
            else:
                counts["empty_or_repeated_boundaries"] += 1
            continue
        if not line.strip() or line != line.strip():
            counts["whitespace_framing_errors"] += 1
            continue
        match = _NUMBERED.fullmatch(line)
        if match is None:
            counts["numbered_framing_errors"] += 1
            continue
        ordinal = int(match.group(1))
        if ordinal != expected_ordinal and first_v0_3_failure_code is None:
            first_v0_3_failure_code = (
                "ordinal_one_reset_without_blank"
                if ordinal == 1 and entries
                else "ordinal_mismatch"
            )
        if ordinal == 1 and entries and expected_ordinal != 1:
            counts["implicit_ordinal_one_boundaries"] += 1
            expected_ordinal = 1
            entries = False
        if ordinal != expected_ordinal:
            counts["ordinal_errors_after_reset_aware_parse"] += 1
            continue
        entries = True
        expected_ordinal += 1
        body = match.group(2)
        tabs = body.count("\t")
        if tabs == 1:
            user, assistant = body.split("\t")
            if not user or not assistant or user != user.strip() or assistant != assistant.strip():
                counts["pair_framing_errors"] += 1
            elif assistant.casefold() == "api_call" or assistant.casefold().startswith(
                "api_call "
            ):
                counts["api_rows"] += 1
            else:
                counts["natural_pair_rows"] += 1
        elif tabs == 0:
            if _KB.fullmatch(body) is None:
                counts["malformed_non_tab_rows"] += 1
            else:
                counts["kb_rows"] += 1
        else:
            counts["multiple_tab_rows"] += 1
    if entries:
        counts["eof_dialogue_boundaries"] += 1
    dialogues = (
        counts["explicit_dialogue_boundaries"]
        + counts["implicit_ordinal_one_boundaries"]
        + counts["eof_dialogue_boundaries"]
    )
    return {
        "counts": {key: counts[key] for key in sorted(_COUNT_KEYS)},
        "dialogues_reset_aware": dialogues,
        "first_v0_3_failure_code": first_v0_3_failure_code,
        "git_blob_sha1": git_sha1.hexdigest(),
        "sha256": sha256.hexdigest(),
    }


def _alarm(_signum: int, _frame: object) -> None:
    raise TimeoutError("diagnostic runtime cap exceeded")


def main() -> int:
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, 30 * 60)
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    report_path = root / REPORT_PATH
    verification_path = root / VERIFICATION_PATH
    if report_path.exists() or verification_path.exists():
        raise FileExistsError("diagnostic evidence already exists")
    raw_root = root / RAW_ROOT
    if (raw_root / TEST_OBJECT).exists():
        raise ValueError("prohibited test object is present")
    for name, relative in STATIC_PATHS.items():
        path = root / relative
        if not path.is_file() or _sha256(path.read_bytes()) != STATIC_BINDINGS[name]:
            raise ValueError("static diagnostic binding changed")

    marker = b'{"started":"bedstc2_parser_diagnostic_v0.4"}\n'
    _exclusive(report_path, marker)
    results: dict[str, object] = {}
    for name in ("train", "dev"):
        expected = EXPECTED[name]
        path = raw_root / str(expected["path"])
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError("raw object metadata changed")
        with path.open("rb") as handle:
            result = scan_stream(
                handle,
                maximum_lines=int(expected["lines"]),
                size=int(expected["bytes"]),
            )
        if (
            result["sha256"] != expected["sha256"]
            or result["git_blob_sha1"] != expected["git_blob_sha1"]
            or result["counts"].get("physical_lines") != expected["lines"]
            or result["counts"].get("bytes_read") != expected["bytes"]
            or result["dialogues_reset_aware"] != expected["dialogues"]
        ):
            raise ValueError("diagnostic object binding or inventory changed")
        results[name] = result
    combined_counts = {
        key: sum(int(results[name]["counts"][key]) for name in results)
        for key in sorted(_COUNT_KEYS)
    }
    implicit_resets = combined_counts["implicit_ordinal_one_boundaries"]
    confirmed = (
        implicit_resets > 0
        and all(combined_counts[key] == 0 for key in _ERROR_KEYS)
        and all(
            result["first_v0_3_failure_code"]
            in {None, "ordinal_one_reset_without_blank"}
            for result in results.values()
        )
    )
    verdict = (
        "CONFIRMED_ORDINAL_1_BOUNDARY_ONLY"
        if confirmed
        else "REFUTED_OTHER_GRAMMAR_MISMATCH"
    )
    report = _seal(
        {
            "authorization_text_sha256": AUTHORIZATION_TEXT_SHA256,
            "bindings": STATIC_BINDINGS,
            "diagnostic": {"combined_counts": combined_counts, "objects": results},
            "hypothesis_verdict": verdict,
            "limits": {
                "candidate_row_reads": 0,
                "gpu_use": 0,
                "inference_calls": 0,
                "network_bytes": 0,
                "source_line_reads": sum(int(EXPECTED[name]["lines"]) for name in EXPECTED),
                "test_object_accessed": False,
            },
            "raw_text_retained": False,
            "recommended_repair": (
                "accept_ordinal_one_as_implicit_dialogue_boundary"
                if confirmed
                else "none_fail_closed"
            ),
            "schema_version": SCHEMA,
            "status": "PASS_parser_contract_diagnostic_complete",
        },
        "report_sha256",
    )
    report_payload = _canonical(report)
    _finish_marker(report_path, marker, report_payload)
    verification = _seal(
        {
            "checks": {
                "bindings_verified": True,
                "no_candidate_rows_read": True,
                "raw_text_absent": True,
                "source_inventory_reproduced": True,
            },
            "report_file_sha256": _sha256(report_payload),
            "report_internal_sha256": report["report_sha256"],
            "schema_version": VERIFICATION_SCHEMA,
            "status": "PASS_verified_parser_contract_diagnostic",
        },
        "verification_sha256",
    )
    _exclusive(verification_path, _canonical(verification))
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(
        json.dumps(
            {
                "report_file_sha256": _sha256(report_payload),
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
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
