#!/usr/bin/env python3
"""One-pass, aggregate-only Be-DSTC2 malformed-row intersection diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

SCHEMA_VERSION = "lumi.sft-bedstc2-malformed-intersection-diagnostic.v0.7"
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-bedstc2-malformed-intersection-diagnostic-verification.v0.7"
)
AUTHORIZATION_TEXT_SHA256 = (
    "01ada5f9f04b1b369901661390f821ae5d0ad533164390ed144d42a684b8a05f"
)
PRIOR_AUTHORIZATION_TEXT_SHA256 = (
    "b9adeb30ce93d3cb54c3f7ac237e47e65163ceebc048d7d0ffcf50fa9c00fae7"
)
REPORT_PATH = "evidence/lumi-4b-sft-bedstc2-malformed-intersection-diagnostic-v0.7.json"
VERIFICATION_PATH = (
    "evidence/lumi-4b-sft-bedstc2-malformed-intersection-diagnostic-verification-v0.7.json"
)
SCRIPT_PATH = "scripts/diagnose_sft_bedstc2_malformed_intersections_v4.py"
TEST_PATH = "tests/test_bedstc2_malformed_intersection_diagnostic_v4.py"
FAILED_DIAGNOSTIC_VERIFICATION_PATH = (
    "evidence/lumi-4b-sft-bedstc2-malformed-intersection-diagnostic-verification-v0.6.json"
)
RAW_ROOT = (
    "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/"
    "attempt-001/private/raw"
)
TEST_OBJECT = "data/bengali/dialog-dstc2-tst.txt"
MAXIMUM_SOURCE_LINE_READS = 113_718
MAXIMUM_RUNTIME_SECONDS = 10 * 60
MAXIMUM_DIAGNOSTIC_WRITE_BYTES = 65_536

EXPECTED_OBJECTS: dict[str, dict[str, Any]] = {
    "train": {
        "bytes": 4_085_040,
        "git_blob_sha1": "4683420d14bf546c31e7698ce9ecd799ba0f88d3",
        "lines": 85_101,
        "path": "data/bengali/dialog-dstc2-trn.txt",
        "sha256": "79644ed579c831626b1f2a2dfef1f804b835c9e3da19f93fe493c9fbbb3b591c",
    },
    "dev": {
        "bytes": 1_361_043,
        "git_blob_sha1": "543eb0aa02d9ec7f1af65516017a1bfcace1f636",
        "lines": 28_617,
        "path": "data/bengali/dialog-dstc2-dev.txt",
        "sha256": "a46b98e4f3ff3c9575240fe42fee2d735fa61ec41aacd53d73627b33b7147116",
    },
}
EXPECTED_PREDECESSOR_BINDINGS = {
    "failed_diagnostic_marker": {
        "bytes": 61,
        "path": "evidence/lumi-4b-sft-bedstc2-malformed-intersection-diagnostic-v0.6.json",
        "sha256": "83a3d723116770dca970702ddca7684ca5da2cc5f1400f0cbc3ea66cf86853b1",
    },
    "failed_diagnostic_script": {
        "path": "scripts/diagnose_sft_bedstc2_malformed_intersections_v3.py",
        "sha256": "50355a36492802df5146f26b2faa79692f8a9113b90204f414f8de668cc1b4ba",
    },
    "failed_diagnostic_test": {
        "path": "tests/test_bedstc2_malformed_intersection_diagnostic_v3.py",
        "sha256": "4d1c1e9c30f17e929b72f23f58b54fd478fc6ccb801918bd60d112f38812e155",
    },
    "implementation": {
        "path": "src/lumi/posttraining_bedstc2_banglish_conversation_curation_v3.py",
        "sha256": "b4a411142358dcaf6937a09b50ccbd94c1a0bdf9baf44052aa11210cb1e276dc",
    },
    "plan": {
        "path": "configs/lumi-4b-sft-bedstc2-banglish-conversation-curation-recovery-v0.4.json",
        "sha256": "4015afc5ebba188fdc6b649ec3801e2c0a171ee1be73fcb2a425d611a21d5eab",
    },
    "runner": {
        "path": "scripts/curate_sft_bedstc2_banglish_conversation_v3.py",
        "sha256": "0b60a93d3ee27d71cde8c6a710deb37ae8309da55a1e63a09530acce551fe46b",
    },
    "runner_marker": {
        "bytes": 47,
        "path": "evidence/lumi-4b-sft-bedstc2-banglish-conversation-curation-v0.4.json",
        "sha256": "0f60a1baca9d289ffbb2936394873517d98f42df4437016866d304c4ae69fa49",
    },
    "test": {
        "path": "tests/test_bedstc2_banglish_conversation_curation_v3.py",
        "sha256": "9fc40fe1ca294d8d669c9394932ca934c6640d78df88eb2f21baf9d565de639c",
    },
    "verifier": {
        "path": "scripts/verify_sft_bedstc2_banglish_conversation_v3.py",
        "sha256": "5e11b637c8a6ed106b1b151eb561c6026f26420676bbfcd04586d5033d0aee54",
    },
}
EXPECTED_COUNTS = {
    "non_kb_numbered_rows": 421,
    "outer_trailing_whitespace_rows": 17,
    "pair_user_trailing_whitespace_rows": 37,
    "unique_malformed_rows": 475,
}
EXPECTED_INTERSECTIONS = {
    "non_kb_and_outer": 0,
    "non_kb_and_pair_user": 0,
    "outer_and_pair_user": 0,
    "all_three": 0,
}
_NUMBERED = re.compile(r"^([1-9][0-9]*) (.*)$")
_KB = re.compile(r"^(\S+) (R_\S+) (\S(?:.*\S)?)$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_REPORT_MARKER = b'{"started":"bedstc2_malformed_intersection_diagnostic_v0.7"}\n'
_FORBIDDEN_RAW_KEYS = {
    "assistant",
    "assistant_response",
    "body",
    "content",
    "entity",
    "line",
    "messages",
    "relation",
    "source_text",
    "user",
    "user_prompt",
    "value",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    sealed = dict(value)
    sealed[field] = _sha256_bytes(_canonical_bytes(sealed))
    return sealed


def _raw_key_scan(value: Any) -> None:
    if isinstance(value, Mapping):
        if _FORBIDDEN_RAW_KEYS.intersection(value):
            raise ValueError("raw source text entered diagnostic evidence")
        for nested in value.values():
            _raw_key_scan(nested)
    elif isinstance(value, list):
        for nested in value:
            _raw_key_scan(nested)


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _replace_marker(path: Path, marker: bytes, payload: bytes) -> None:
    with path.open("r+b") as handle:
        observed = handle.read()
        if observed != marker:
            raise ValueError("diagnostic marker changed")
        handle.seek(0)
        handle.truncate()
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _verify_predecessor(root: Path) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for name, expected in EXPECTED_PREDECESSOR_BINDINGS.items():
        path = root / str(expected["path"])
        if not path.is_file():
            raise ValueError("bound predecessor artifact missing")
        size = path.stat().st_size
        digest = _file_sha256(path)
        if digest != expected["sha256"] or (
            "bytes" in expected and size != expected["bytes"]
        ):
            raise ValueError("bound predecessor artifact changed")
        observed[name] = {"bytes": size, "path": expected["path"], "sha256": digest}
    return observed


def _decode_line(raw: bytes) -> str:
    decoded = raw.decode("utf-8")
    if decoded.endswith("\r\n"):
        return decoded[:-2]
    if decoded.endswith("\n"):
        return decoded[:-1]
    return decoded


def _scan_stream(
    handle: BinaryIO,
    *,
    expected: Mapping[str, Any],
    total_budget: list[int],
) -> tuple[dict[str, Any], dict[str, int], dict[str, int]]:
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(  # noqa: S324 - canonical Git blob identity.
        f"blob {expected['bytes']}\0".encode(), usedforsecurity=False
    )
    observed_bytes = 0
    observed_lines = 0
    expected_ordinal = 1
    active_rows = 0
    singles = {
        "non_kb_numbered_rows": 0,
        "outer_trailing_whitespace_rows": 0,
        "pair_user_trailing_whitespace_rows": 0,
        "unique_malformed_rows": 0,
    }
    intersections = {key: 0 for key in EXPECTED_INTERSECTIONS}
    for raw in handle:
        observed_lines += 1
        total_budget[0] += 1
        if (
            observed_lines > int(expected["lines"])
            or total_budget[0] > MAXIMUM_SOURCE_LINE_READS
        ):
            raise ValueError("diagnostic source-line cap exceeded")
        observed_bytes += len(raw)
        sha256.update(raw)
        git_sha1.update(raw)
        line = _decode_line(raw)
        if line == "":
            if active_rows == 0:
                raise ValueError("source dialogue boundary changed")
            expected_ordinal = 1
            active_rows = 0
            continue
        if not line.strip() or line != line.lstrip():
            raise ValueError("source framing changed")
        framing_line = line.rstrip()
        match = _NUMBERED.fullmatch(framing_line)
        if match is None or int(match.group(1)) != expected_ordinal:
            raise ValueError("source ordinal framing changed")
        expected_ordinal += 1
        active_rows += 1
        classified_body = match.group(2)
        outer = line != framing_line
        non_kb = False
        pair_user = False
        tabs = classified_body.count("\t")
        if tabs == 1:
            user, assistant = classified_body.split("\t")
            if not user or not assistant:
                raise ValueError("source pair field framing changed")
            if (
                user == user.lstrip()
                and user != user.rstrip()
                and assistant == assistant.strip()
            ):
                pair_user = True
            elif user != user.strip() or assistant != assistant.strip():
                raise ValueError("source pair field framing changed")
        elif tabs == 0:
            if _KB.fullmatch(classified_body) is None:
                fields = classified_body.split(" ")
                if (
                    len(fields) == 3
                    and all(fields)
                    and not any(field.startswith("R_") for field in fields)
                ):
                    non_kb = True
                else:
                    raise ValueError("malformed non-tab source row")
        else:
            raise ValueError("source row contains multiple tabs")
        singles["non_kb_numbered_rows"] += int(non_kb)
        singles["outer_trailing_whitespace_rows"] += int(outer)
        singles["pair_user_trailing_whitespace_rows"] += int(pair_user)
        singles["unique_malformed_rows"] += int(non_kb or outer or pair_user)
        intersections["non_kb_and_outer"] += int(non_kb and outer)
        intersections["non_kb_and_pair_user"] += int(non_kb and pair_user)
        intersections["outer_and_pair_user"] += int(outer and pair_user)
        intersections["all_three"] += int(non_kb and outer and pair_user)
    if observed_lines != int(expected["lines"]):
        raise ValueError("diagnostic source-line inventory changed")
    binding = {
        "bytes": observed_bytes,
        "git_blob_sha1": git_sha1.hexdigest(),
        "lines": observed_lines,
        "path": expected["path"],
        "sha256": sha256.hexdigest(),
    }
    if (
        binding["bytes"] != expected["bytes"]
        or binding["git_blob_sha1"] != expected["git_blob_sha1"]
        or binding["sha256"] != expected["sha256"]
    ):
        raise ValueError("diagnostic source binding changed")
    return binding, singles, intersections


def scan_bound_objects(root: Path) -> dict[str, Any]:
    raw_root = root / RAW_ROOT
    if (raw_root / TEST_OBJECT).exists():
        raise ValueError("prohibited test object present")
    total_budget = [0]
    object_bindings: dict[str, dict[str, Any]] = {}
    combined_counts = {key: 0 for key in EXPECTED_COUNTS}
    combined_intersections = {key: 0 for key in EXPECTED_INTERSECTIONS}
    for name in ("train", "dev"):
        expected = EXPECTED_OBJECTS[name]
        path = raw_root / str(expected["path"])
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError("diagnostic source metadata changed")
        with path.open("rb") as handle:
            binding, counts, intersections = _scan_stream(
                handle, expected=expected, total_budget=total_budget
            )
        object_bindings[name] = binding
        for key, value in counts.items():
            combined_counts[key] += value
        for key, value in intersections.items():
            combined_intersections[key] += value
    if total_budget[0] != MAXIMUM_SOURCE_LINE_READS:
        raise ValueError("diagnostic source-line accounting changed")
    return {
        "counts": combined_counts,
        "intersections": combined_intersections,
        "object_bindings": object_bindings,
        "source_line_reads": total_budget[0],
    }


def _diagnostic_confirmed(result: Mapping[str, Any]) -> bool:
    return (
        result.get("counts") == EXPECTED_COUNTS
        and result.get("intersections") == EXPECTED_INTERSECTIONS
        and result.get("source_line_reads") == MAXIMUM_SOURCE_LINE_READS
    )


def validate_report(report: Mapping[str, Any]) -> None:
    expected_top = {
        "authorization_text_sha256",
        "bindings",
        "diagnostic",
        "limits",
        "prior_authorization_text_sha256",
        "prior_consumption",
        "prohibited_actions",
        "raw_text_retained",
        "report_sha256",
        "schema_version",
        "status",
    }
    if set(report) != expected_top:
        raise ValueError("diagnostic report inventory changed")
    internal = report.get("report_sha256")
    if not isinstance(internal, str) or _SHA256.fullmatch(internal) is None:
        raise ValueError("diagnostic report seal changed")
    core = dict(report)
    core.pop("report_sha256")
    if _sha256_bytes(_canonical_bytes(core)) != internal:
        raise ValueError("diagnostic report seal mismatch")
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("authorization_text_sha256") != AUTHORIZATION_TEXT_SHA256
        or report.get("prior_authorization_text_sha256")
        != PRIOR_AUTHORIZATION_TEXT_SHA256
        or report.get("raw_text_retained") is not False
    ):
        raise ValueError("diagnostic report contract changed")
    diagnostic = report.get("diagnostic")
    if not isinstance(diagnostic, Mapping) or set(diagnostic) != {
        "counts",
        "intersections",
        "source_line_reads",
    }:
        raise ValueError("diagnostic aggregate inventory changed")
    confirmed = _diagnostic_confirmed(diagnostic)
    expected_status = (
        "PASS_confirmed_421_17_37_unique_475_disjoint"
        if confirmed
        else "STOP_FAIL_CLOSED_malformed_intersection_mismatch"
    )
    if report.get("status") != expected_status:
        raise ValueError("diagnostic result status changed")
    counts = diagnostic.get("counts")
    intersections = diagnostic.get("intersections")
    if (
        not isinstance(counts, Mapping)
        or set(counts) != set(EXPECTED_COUNTS)
        or not all(isinstance(value, int) and value >= 0 for value in counts.values())
        or not isinstance(intersections, Mapping)
        or set(intersections) != set(EXPECTED_INTERSECTIONS)
        or not all(
            isinstance(value, int) and value >= 0 for value in intersections.values()
        )
    ):
        raise ValueError("diagnostic aggregate shape changed")
    inclusion_exclusion = (
        counts["non_kb_numbered_rows"]
        + counts["outer_trailing_whitespace_rows"]
        + counts["pair_user_trailing_whitespace_rows"]
        - intersections["non_kb_and_outer"]
        - intersections["non_kb_and_pair_user"]
        - intersections["outer_and_pair_user"]
        + intersections["all_three"]
    )
    if inclusion_exclusion != counts["unique_malformed_rows"]:
        raise ValueError("diagnostic intersection arithmetic changed")
    limits = report.get("limits")
    if limits != {
        "baseline_row_reads": 0,
        "candidate_row_reads": 0,
        "maximum_runtime_seconds": MAXIMUM_RUNTIME_SECONDS,
        "maximum_write_bytes": MAXIMUM_DIAGNOSTIC_WRITE_BYTES,
        "network_bytes": 0,
        "source_line_reads": MAXIMUM_SOURCE_LINE_READS,
    }:
        raise ValueError("diagnostic limits changed")
    if report.get("prior_consumption") != {
        "baseline_row_reads": 0,
        "candidate_row_reads": 0,
        "diagnostic_passes_completed": 0,
        "diagnostic_passes_started": 1,
        "gpu_use": 0,
        "inference_calls": 0,
        "network_bytes": 0,
        "private_write_bytes": 61,
        "runtime_seconds": 0.07,
        "source_line_reads_upper_bound": 113_718,
        "test_object_accessed": False,
        "verification_invoked": False,
    }:
        raise ValueError("prior diagnostic consumption changed")
    prohibited = report.get("prohibited_actions")
    if not isinstance(prohibited, Mapping) or set(prohibited) != {
        "gpu_used",
        "inference_called",
        "network_used",
        "registry_mutated",
        "source_mutated",
        "test_object_accessed",
    } or any(value is not False for value in prohibited.values()):
        raise ValueError("diagnostic prohibited-action evidence changed")
    bindings = report.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "diagnostic_code",
        "predecessor",
        "source_objects",
    }:
        raise ValueError("diagnostic binding inventory changed")
    code = bindings.get("diagnostic_code")
    if not isinstance(code, Mapping) or set(code) != {"script", "test"}:
        raise ValueError("diagnostic code binding changed")
    for item in code.values():
        if (
            not isinstance(item, Mapping)
            or set(item) != {"bytes", "path", "sha256"}
            or not isinstance(item["bytes"], int)
            or item["bytes"] <= 0
            or not isinstance(item["path"], str)
            or not isinstance(item["sha256"], str)
            or _SHA256.fullmatch(item["sha256"]) is None
        ):
            raise ValueError("diagnostic code binding malformed")
    if code["script"]["path"] != SCRIPT_PATH or code["test"]["path"] != TEST_PATH:
        raise ValueError("diagnostic code path changed")
    predecessor = bindings.get("predecessor")
    if not isinstance(predecessor, Mapping) or set(predecessor) != set(
        EXPECTED_PREDECESSOR_BINDINGS
    ):
        raise ValueError("diagnostic predecessor inventory changed")
    for name, item in predecessor.items():
        expected = EXPECTED_PREDECESSOR_BINDINGS[name]
        if (
            not isinstance(item, Mapping)
            or set(item) != {"bytes", "path", "sha256"}
            or item.get("path") != expected["path"]
            or item.get("sha256") != expected["sha256"]
            or not isinstance(item.get("bytes"), int)
            or item["bytes"] <= 0
            or ("bytes" in expected and item["bytes"] != expected["bytes"])
        ):
            raise ValueError("diagnostic predecessor binding changed")
    source_objects = bindings.get("source_objects")
    if not isinstance(source_objects, Mapping) or set(source_objects) != {"train", "dev"}:
        raise ValueError("diagnostic source binding inventory changed")
    for name, item in source_objects.items():
        expected = EXPECTED_OBJECTS[name]
        if item != {
            "bytes": expected["bytes"],
            "git_blob_sha1": expected["git_blob_sha1"],
            "lines": expected["lines"],
            "path": expected["path"],
            "sha256": expected["sha256"],
        }:
            raise ValueError("diagnostic source binding changed")
        if _SHA1.fullmatch(str(item["git_blob_sha1"])) is None:
            raise ValueError("diagnostic Git binding malformed")
    _raw_key_scan(report)


def build_verification(report: Mapping[str, Any], report_payload: bytes) -> dict[str, Any]:
    validate_report(report)
    confirmed = _diagnostic_confirmed(report["diagnostic"])
    return _seal(
        {
            "authorization_text_sha256": AUTHORIZATION_TEXT_SHA256,
            "checks": {
                "aggregate_only": True,
                "bindings_reproduced": True,
                "candidate_and_baseline_reads_zero": True,
                "intersection_arithmetic_verified": True,
                "prior_evidence_preserved": True,
                "raw_text_absent": True,
                "source_replay_not_performed": True,
            },
            "conditional_recovery_authorized": confirmed,
            "prior_authorization_text_sha256": PRIOR_AUTHORIZATION_TEXT_SHA256,
            "report_file_sha256": _sha256_bytes(report_payload),
            "report_internal_sha256": report["report_sha256"],
            "schema_version": VERIFICATION_SCHEMA_VERSION,
            "status": (
                "PASS_structurally_verified_unique_475"
                if confirmed
                else "STOP_verified_fail_closed_diagnostic_mismatch"
            ),
        },
        "verification_sha256",
    )


def validate_verification(value: Mapping[str, Any]) -> None:
    if set(value) != {
        "authorization_text_sha256",
        "checks",
        "conditional_recovery_authorized",
        "prior_authorization_text_sha256",
        "report_file_sha256",
        "report_internal_sha256",
        "schema_version",
        "status",
        "verification_sha256",
    }:
        raise ValueError("diagnostic verification inventory changed")
    internal = value.get("verification_sha256")
    if not isinstance(internal, str) or _SHA256.fullmatch(internal) is None:
        raise ValueError("diagnostic verification seal changed")
    core = dict(value)
    core.pop("verification_sha256")
    if _sha256_bytes(_canonical_bytes(core)) != internal:
        raise ValueError("diagnostic verification seal mismatch")
    if (
        value.get("schema_version") != VERIFICATION_SCHEMA_VERSION
        or value.get("authorization_text_sha256") != AUTHORIZATION_TEXT_SHA256
        or value.get("prior_authorization_text_sha256")
        != PRIOR_AUTHORIZATION_TEXT_SHA256
        or not isinstance(value.get("conditional_recovery_authorized"), bool)
        or not isinstance(value.get("report_file_sha256"), str)
        or _SHA256.fullmatch(value["report_file_sha256"]) is None
        or not isinstance(value.get("report_internal_sha256"), str)
        or _SHA256.fullmatch(value["report_internal_sha256"]) is None
        or not isinstance(value.get("checks"), Mapping)
        or set(value["checks"])
        != {
            "aggregate_only",
            "bindings_reproduced",
            "candidate_and_baseline_reads_zero",
            "intersection_arithmetic_verified",
            "prior_evidence_preserved",
            "raw_text_absent",
            "source_replay_not_performed",
        }
        or any(item is not True for item in value["checks"].values())
    ):
        raise ValueError("diagnostic verification contract changed")
    expected_status = (
        "PASS_structurally_verified_unique_475"
        if value["conditional_recovery_authorized"]
        else "STOP_verified_fail_closed_diagnostic_mismatch"
    )
    if value.get("status") != expected_status:
        raise ValueError("diagnostic verification status changed")
    _raw_key_scan(value)


def _alarm(_signum: int, _frame: object) -> None:
    raise TimeoutError("diagnostic runtime cap exceeded")


def main() -> int:
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, MAXIMUM_RUNTIME_SECONDS)
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    root = parser.parse_args().repository_root.resolve()
    report_path = root / REPORT_PATH
    verification_path = root / VERIFICATION_PATH
    if report_path.exists() or verification_path.exists():
        raise FileExistsError("diagnostic output already exists")
    if (root / FAILED_DIAGNOSTIC_VERIFICATION_PATH).exists():
        raise ValueError("failed diagnostic verification unexpectedly exists")
    predecessor = _verify_predecessor(root)
    script_path = root / SCRIPT_PATH
    test_path = root / TEST_PATH
    code_bindings = {
        "script": {
            "bytes": script_path.stat().st_size,
            "path": SCRIPT_PATH,
            "sha256": _file_sha256(script_path),
        },
        "test": {
            "bytes": test_path.stat().st_size,
            "path": TEST_PATH,
            "sha256": _file_sha256(test_path),
        },
    }
    _exclusive_write(report_path, _REPORT_MARKER)
    result = scan_bound_objects(root)
    confirmed = _diagnostic_confirmed(result)
    report = _seal(
        {
            "authorization_text_sha256": AUTHORIZATION_TEXT_SHA256,
            "bindings": {
                "diagnostic_code": code_bindings,
                "predecessor": predecessor,
                "source_objects": result["object_bindings"],
            },
            "diagnostic": {
                "counts": result["counts"],
                "intersections": result["intersections"],
                "source_line_reads": result["source_line_reads"],
            },
            "limits": {
                "baseline_row_reads": 0,
                "candidate_row_reads": 0,
                "maximum_runtime_seconds": MAXIMUM_RUNTIME_SECONDS,
                "maximum_write_bytes": MAXIMUM_DIAGNOSTIC_WRITE_BYTES,
                "network_bytes": 0,
                "source_line_reads": MAXIMUM_SOURCE_LINE_READS,
            },
            "prior_authorization_text_sha256": PRIOR_AUTHORIZATION_TEXT_SHA256,
            "prior_consumption": {
                "baseline_row_reads": 0,
                "candidate_row_reads": 0,
                "diagnostic_passes_completed": 0,
                "diagnostic_passes_started": 1,
                "gpu_use": 0,
                "inference_calls": 0,
                "network_bytes": 0,
                "private_write_bytes": 61,
                "runtime_seconds": 0.07,
                "source_line_reads_upper_bound": 113_718,
                "test_object_accessed": False,
                "verification_invoked": False,
            },
            "prohibited_actions": {
                "gpu_used": False,
                "inference_called": False,
                "network_used": False,
                "registry_mutated": False,
                "source_mutated": False,
                "test_object_accessed": False,
            },
            "raw_text_retained": False,
            "schema_version": SCHEMA_VERSION,
            "status": (
                "PASS_confirmed_421_17_37_unique_475_disjoint"
                if confirmed
                else "STOP_FAIL_CLOSED_malformed_intersection_mismatch"
            ),
        },
        "report_sha256",
    )
    validate_report(report)
    report_payload = _json_bytes(report)
    verification = build_verification(report, report_payload)
    validate_verification(verification)
    verification_payload = _json_bytes(verification)
    total_writes = len(_REPORT_MARKER) + len(report_payload) + len(verification_payload)
    if total_writes > MAXIMUM_DIAGNOSTIC_WRITE_BYTES:
        raise ValueError("diagnostic write cap exceeded")
    _replace_marker(report_path, _REPORT_MARKER, report_payload)
    _exclusive_write(verification_path, verification_payload)
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(
        json.dumps(
            {
                "counts": result["counts"],
                "intersections": result["intersections"],
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    return 0 if confirmed else 2


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
