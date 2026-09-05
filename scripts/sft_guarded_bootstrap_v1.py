"""Standalone pre-ML byte-pinned SFT bootstrap validation foundation.

Only stdlib imports are permitted. This does not import the lumi package, load
the pinned worker, initialize a GPU, or execute a command. A byte pin supplied
by an independently reviewed caller is an input, not an authentication service.
Supervisor evidence is structurally checked only: neither a flag nor a claimed
receipt proves actual OS isolation. Real supervisor authentication and handoff
remain unimplemented, so validated inputs still cannot dispatch a worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import time

SCHEMA = "lumi.sft-guarded-bootstrap-ticket.v0.1"
SCOPE = "private_research_nonproduction"
MAX_TICKET_BYTES = 65_536
MAX_SOURCE_BYTES = 2_097_152
HASH_FIELDS = (
    "source_sha256", "runtime_lock_sha256", "target_receipt_sha256",
    "import_receipt_sha256", "relocation_receipt_sha256", "cuda_acceptance_sha256",
    "resource_caps_sha256", "supervisor_receipt_sha256",
)
CODES = frozenset((
    "python_flags_required", "execute_disabled", "invalid_arguments", "invalid_hash",
    "input_size", "ticket_hash", "source_hash", "json_invalid", "json_duplicate_key",
    "ticket_schema", "ticket_scope", "bindings_schema", "runtime_schema",
    "runtime_binding", "approval_schema", "approval_scope", "approval_subject",
    "approval_time", "supervisor_schema", "supervisor_binding", "control_file_invalid",
    "control_file_changed", "bootstrap_internal_error",
))


class BootstrapError(ValueError):
    def __init__(self, code):
        super().__init__(code if type(code) is str and code in CODES else "bootstrap_internal_error")


def require(condition, code):
    if not condition:
        raise BootstrapError(code) from None


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError):
        pass
    raise BootstrapError("json_invalid") from None


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "invalid_hash")


def exact_keys(value, fields, code):
    require(type(value) is dict and all(type(key) is str for key in value)
            and set(value) == set(fields), code)


def strict_json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "json_duplicate_key")
            value[key] = item
        return value
    def constant(_):
        raise BootstrapError("json_invalid")
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        pass
    raise BootstrapError("json_invalid") from None


def require_python_flags(no_site, no_bytecode):
    require(type(no_site) is int and no_site == 1
            and type(no_bytecode) is int and no_bytecode == 1, "python_flags_required")


def validate_runtime(runtime):
    exact_keys(runtime, ("python_executable", "runtime_lock_sha256", "world_size", "rank", "argv"),
               "runtime_schema")
    path = runtime["python_executable"]
    require(type(path) is str and path.startswith("/") and not path.startswith("//")
            and len(path) <= 4096 and os.path.normpath(path) == path
            and all(32 <= ord(c) != 127 for c in path), "runtime_schema")
    sha(runtime["runtime_lock_sha256"])
    require(type(runtime["world_size"]) is int and 2 <= runtime["world_size"] <= 12
            and type(runtime["rank"]) is int and 0 <= runtime["rank"] < runtime["world_size"],
            "runtime_schema")
    argv = runtime["argv"]
    require(type(argv) is list and len(argv) <= 128
            and all(type(arg) is str and len(arg) <= 4096
                    and all(32 <= ord(c) != 127 for c in arg) for arg in argv), "runtime_schema")


def validate_bootstrap(ticket_bytes, source_bytes, *, expected_ticket_sha256,
                       expected_subject_sha256, expected_source_sha256,
                       now_epoch_seconds, runtime_binding, supervisor_evidence,
                       no_site, no_bytecode):
    """Pure metadata validation; never imports source_bytes or grants a permit.

    runtime_binding is the caller-observed launch identity, not inferred from the
    ticket. supervisor_evidence is a strict structural receipt object supplied by
    the future attachment adapter, NOT trusted authentication of an OS watchdog.
    """
    require_python_flags(no_site, no_bytecode)
    for value in (expected_ticket_sha256, expected_subject_sha256, expected_source_sha256):
        sha(value)
    require(type(ticket_bytes) is bytes and 0 < len(ticket_bytes) <= MAX_TICKET_BYTES
            and type(source_bytes) is bytes and 0 < len(source_bytes) <= MAX_SOURCE_BYTES, "input_size")
    require(hashlib.sha256(ticket_bytes).hexdigest() == expected_ticket_sha256, "ticket_hash")
    require(hashlib.sha256(source_bytes).hexdigest() == expected_source_sha256, "source_hash")
    ticket = strict_json(ticket_bytes)
    exact_keys(ticket, ("schema_version", "scope", "operation", "production_eligible",
                        "bindings", "runtime", "approval"), "ticket_schema")
    require(ticket["schema_version"] == SCHEMA and ticket["scope"] == SCOPE
            and ticket["operation"] == "sft_gpu_smoke_and_fresh_resume_only"
            and ticket["production_eligible"] is False, "ticket_scope")
    bindings = ticket["bindings"]
    exact_keys(bindings, HASH_FIELDS, "bindings_schema")
    for value in bindings.values():
        sha(value)
    require(bindings["source_sha256"] == expected_source_sha256, "source_hash")
    validate_runtime(ticket["runtime"])
    validate_runtime(runtime_binding)
    require(canonical(ticket["runtime"]) == canonical(runtime_binding)
            and bindings["runtime_lock_sha256"] == runtime_binding["runtime_lock_sha256"], "runtime_binding")
    approval = ticket["approval"]
    exact_keys(approval, ("scope", "kind", "approved", "subject_sha256", "issued_at_epoch_seconds",
                          "expires_at_epoch_seconds"), "approval_schema")
    sha(approval["subject_sha256"])
    require(approval["scope"] == SCOPE and approval["kind"] == "bounded_sft_execution"
            and approval["approved"] is True, "approval_scope")
    subject = digest({key: value for key, value in ticket.items() if key != "approval"})
    require(subject == approval["subject_sha256"] == expected_subject_sha256, "approval_subject")
    issued, expires = approval["issued_at_epoch_seconds"], approval["expires_at_epoch_seconds"]
    require(all(type(value) is int and 0 <= value <= 2**63 - 1
                for value in (issued, expires, now_epoch_seconds))
            and issued <= now_epoch_seconds < expires and issued < expires, "approval_time")
    exact_keys(supervisor_evidence, ("schema_version", "validation_kind", "source_sha256",
                                    "runtime_binding_sha256", "resource_caps_sha256"), "supervisor_schema")
    require(supervisor_evidence["schema_version"] == "lumi.sft-supervisor-attachment.v0.1"
            and supervisor_evidence["validation_kind"] == "structural_only", "supervisor_schema")
    for key in ("source_sha256", "runtime_binding_sha256", "resource_caps_sha256"):
        sha(supervisor_evidence[key])
    require(digest(supervisor_evidence) == bindings["supervisor_receipt_sha256"]
            and supervisor_evidence["source_sha256"] == expected_source_sha256
            and supervisor_evidence["runtime_binding_sha256"] == digest(runtime_binding)
            and supervisor_evidence["resource_caps_sha256"] == bindings["resource_caps_sha256"],
            "supervisor_binding")
    return {
        "schema_version": "lumi.sft-guarded-bootstrap-validation.v0.1",
        "metadata_validation_passed": True,
        "source_byte_pin_verified": True,
        "execution_enabled": False,
        "worker_imported": False,
        "supervisor_attachment_authenticated": False,
        "production_eligible": False,
        "blocker_codes": ["os_supervisor_authentication_and_worker_handoff_not_implemented"],
    }


def read_control_file(path, limit):
    """Bounded regular read; final component no-follow, before/after identity check."""
    fd = None
    failed = False
    try:
        require(type(path) is str and path.startswith("/") and os.path.normpath(path) == path
                and type(limit) is int and 0 < limit <= MAX_SOURCE_BYTES,
                "control_file_invalid")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit, "control_file_invalid")
        parts, remaining = [], before.st_size
        while remaining:
            chunk = os.read(fd, min(65_536, remaining))
            require(bool(chunk), "control_file_changed")
            parts.append(chunk)
            remaining -= len(chunk)
        require(os.read(fd, 1) == b"", "control_file_changed")
        after = os.fstat(fd)
        identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        require(identity(before) == identity(after), "control_file_changed")
        data = b"".join(parts)
    except OSError:
        failed = True
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                failed = True
    require(not failed, "control_file_invalid")
    return data


class _Parser(argparse.ArgumentParser):
    def error(self, _message):
        raise BootstrapError("invalid_arguments")


def main(argv=None):
    """Only a validation CLI. --execute always rejects before any control read."""
    try:
        require_python_flags(sys.flags.no_site, sys.flags.dont_write_bytecode)
        argv = sys.argv[1:] if argv is None else argv
        require("--execute" not in argv, "execute_disabled")
        parser = _Parser(description=__doc__)
        for field in ("ticket", "source", "supervisor-evidence", "runtime-binding",
                      "expected-ticket-sha256", "expected-subject-sha256", "expected-source-sha256"):
            parser.add_argument("--" + field, required=True)
        args = parser.parse_args(argv)
        for value in (args.expected_ticket_sha256, args.expected_subject_sha256, args.expected_source_sha256):
            sha(value)
        ticket_bytes = read_control_file(args.ticket, MAX_TICKET_BYTES)
        require(hashlib.sha256(ticket_bytes).hexdigest() == args.expected_ticket_sha256, "ticket_hash")
        runtime_binding = strict_json(read_control_file(args.runtime_binding, MAX_TICKET_BYTES))
        validate_runtime(runtime_binding)
        require(runtime_binding["python_executable"] == os.path.abspath(sys.executable), "runtime_binding")
        report = validate_bootstrap(
            ticket_bytes,
            read_control_file(args.source, MAX_SOURCE_BYTES),
            expected_ticket_sha256=args.expected_ticket_sha256,
            expected_subject_sha256=args.expected_subject_sha256,
            expected_source_sha256=args.expected_source_sha256,
            now_epoch_seconds=int(time.time()),
            runtime_binding=runtime_binding,
            supervisor_evidence=strict_json(read_control_file(args.supervisor_evidence, MAX_TICKET_BYTES)),
            no_site=sys.flags.no_site, no_bytecode=sys.flags.dont_write_bytecode,
        )
        print(json.dumps(report, sort_keys=True))
        return 0
    except BootstrapError as exc:
        print(json.dumps({"execution_enabled": False, "failure_code": str(exc)}, sort_keys=True))
        return 2
    except BaseException:
        print(json.dumps({"execution_enabled": False, "failure_code": "bootstrap_internal_error"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
