"""Read-only, externally pinned safety URI metadata audit. Never replay or publish.

Run with Python -S -B. Only the two explicitly supplied JSON metadata files are
read; paths embedded in either document are never followed. Report PASS means
metadata contract consistency, not candidate/tokenizer replay or execution rights.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import signal
import stat
import sys
from pathlib import Path

MAXIMUM_FILE_BYTES = 131_072
MAXIMUM_RUNTIME_SECONDS = 30
HASH = re.compile(r"[0-9a-f]{64}\Z")
CODES = frozenset(
    {
        "arguments_rejected",
        "runtime_rejected",
        "pin_rejected",
        "path_rejected",
        "file_rejected",
        "file_changed",
        "file_hash_mismatch",
        "json_rejected",
        "metadata_contract_rejected",
        "deadline_exceeded",
        "audit_failed",
    }
)


class AuditError(ValueError):
    def __init__(self, code: str):
        self.code = code if type(code) is str and code in CODES else "audit_failed"
        super().__init__(self.code)


def require(condition: bool, code: str) -> None:
    if not condition:
        raise AuditError(code) from None


class Parser(argparse.ArgumentParser):
    def error(self, _message):
        raise AuditError("arguments_rejected")


def _pin(value: object) -> None:
    require(type(value) is str and HASH.fullmatch(value) is not None, "pin_rejected")


def _identity(info) -> tuple:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def read_bound(path: Path, expected_sha256: str, accounting: dict) -> bytes:
    """Bounded no-follow descriptor walk; no raw path or OS errors are exposed."""
    _pin(expected_sha256)
    value = os.fspath(path)
    require(
        type(value) is str
        and value.startswith("/")
        and not value.startswith("//")
        and os.path.normpath(value) == value
        and len(value) <= 4096
        and not any(char in value for char in ("\x00", "\n", "\r"))
        and value.endswith(".json"),
        "path_rejected",
    )
    parts = Path(value).parts[1:]
    require(0 < len(parts) <= 64, "path_rejected")
    handles = []
    failure = None
    payload = b""
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        current = os.open("/", flags)
        handles.append(current)
        for part in parts[:-1]:
            current = os.open(part, flags, dir_fd=current)
            handles.append(current)
        fd = os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=current
        )
        handles.append(fd)
        accounting["metadata_files_opened"] += 1
        before = os.fstat(fd)
        require(
            stat.S_ISREG(before.st_mode)
            and before.st_nlink == 1
            and 0 < before.st_size <= MAXIMUM_FILE_BYTES,
            "file_rejected",
        )
        chunks = []
        remaining = before.st_size
        while remaining:
            allowance = 2 * MAXIMUM_FILE_BYTES - accounting["metadata_bytes_read"]
            require(allowance > 0, "file_rejected")
            chunk = os.read(fd, min(65_536, remaining, allowance))
            if not chunk:
                break
            accounting["metadata_bytes_read"] += len(chunk)
            require(accounting["metadata_bytes_read"] <= 2 * MAXIMUM_FILE_BYTES, "file_rejected")
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(fd)
        named = os.stat(parts[-1], dir_fd=current, follow_symlinks=False)
        require(
            _identity(before) == _identity(after) == _identity(named)
            and len(payload) == before.st_size,
            "file_changed",
        )
        require(hashlib.sha256(payload).hexdigest() == expected_sha256, "file_hash_mismatch")
    except AuditError as error:
        failure = error.code
    except OSError:
        failure = "file_rejected"
    finally:
        for fd in reversed(handles):
            try:
                os.close(fd)
            except OSError:
                failure = failure or "file_rejected"
    if failure is not None:
        raise AuditError(failure) from None
    return payload


def strict_object(payload: bytes) -> dict:
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "json_rejected")
            output[key] = value
        return output

    def constant(_value):
        raise AuditError("json_rejected")

    failed = False
    try:
        value = json.loads(payload, object_pairs_hook=pairs, parse_constant=constant)
        require(type(value) is dict, "json_rejected")
    except AuditError:
        raise
    except (ValueError, TypeError, UnicodeError, RecursionError):
        failed = True
    if failed:
        raise AuditError("json_rejected") from None
    return value


def _load_validator():
    source = (
        Path(__file__).resolve().parents[1] / "src/lumi/posttraining_safety_uri_verification_v2.py"
    )
    spec = importlib.util.spec_from_file_location("safety_uri_contract_v2", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _deadline(_number, _frame):
    raise AuditError("deadline_exceeded")


def main(argv=None) -> int:
    accounting = {"metadata_files_opened": 0, "metadata_bytes_read": 0}
    result = None
    output = {
        "schema_version": "lumi.sft-safety-uri-metadata-assurance-report.v0.2",
        "status": "STOP",
        "metadata_contract_passed": False,
        "candidate_rows_replayed": False,
        "prior_execution_actions_verified": False,
        "full_private_integration_passed": False,
        "execution_authorized": False,
        "production_eligible": False,
    }
    try:
        require(
            sys.flags.no_site == 1
            and sys.dont_write_bytecode
            and not sys.flags.optimize
            and os.name == "posix"
            and hasattr(signal, "SIGALRM"),
            "runtime_rejected",
        )
        require(
            not any(
                name.split(".")[0] in {"lumi", "torch", "transformers", "numpy", "tokenizers"}
                for name in sys.modules
            ),
            "runtime_rejected",
        )
        arguments = list(sys.argv[1:] if argv is None else argv)
        require(
            len(arguments) == 8
            and set(arguments[::2])
            == {
                "--plan",
                "--report",
                "--expected-plan-sha256",
                "--expected-report-sha256",
            },
            "arguments_rejected",
        )
        parser = Parser(description=__doc__, add_help=False, allow_abbrev=False)
        parser.add_argument("--plan", type=Path, required=True)
        parser.add_argument("--report", type=Path, required=True)
        parser.add_argument("--expected-plan-sha256", required=True)
        parser.add_argument("--expected-report-sha256", required=True)
        args = parser.parse_args(arguments)
        _pin(args.expected_plan_sha256)
        _pin(args.expected_report_sha256)
        require(args.plan != args.report, "path_rejected")
        signal.signal(signal.SIGALRM, _deadline)
        signal.alarm(MAXIMUM_RUNTIME_SECONDS)
        plan = strict_object(read_bound(args.plan, args.expected_plan_sha256, accounting))
        report = strict_object(read_bound(args.report, args.expected_report_sha256, accounting))
        validator = _load_validator()
        rejected = False
        try:
            result = validator.audit_report(
                plan,
                report,
                plan_file_sha256=args.expected_plan_sha256,
                report_file_sha256=args.expected_report_sha256,
            )
        except AuditError:
            raise
        except (ValueError, TypeError, KeyError, RecursionError):
            rejected = True
        require(not rejected, "metadata_contract_rejected")
    except AuditError as error:
        output["failure_code"] = error.code
    except BaseException:
        output["failure_code"] = "audit_failed"
    finally:
        try:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
        except AuditError as error:
            output["failure_code"] = error.code
        except BaseException:
            output["failure_code"] = "audit_failed"
    # Publish success only after the deadline has been disarmed successfully.
    if result is not None and "failure_code" not in output:
        output = result
    output.update(accounting)
    output.update(
        {
            "input_files_written": 0,
            "candidate_row_reads": 0,
            "network_used": False,
            "gpu_used": False,
            "inference_called": False,
        }
    )
    print(json.dumps(output, sort_keys=True))
    return 0 if output["metadata_contract_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
