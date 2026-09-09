#!/usr/bin/env python3
"""Remote-only, metadata-only diagnostic of the frozen OCI origin observer.

NEVER import or execute this file on the Mac. Source/AST tests are the only
local test surface. This wrapper does not repair, retry or grant acceptance.
Frozen origin predicates, fixture workload and cleanup are called unchanged.

Counts describe the parent, validated received failure snapshots and completed
frozen invocations, not unseen child work. need_checks_observed is a saturating
telemetry counter, not a new workload or acceptance cap. Fixed path categories
are lexical diagnostics only, not relaxed origin decisions or byte proofs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
from pathlib import Path

SCHEMA = "lumi.sft-cpu-origin-diagnostic.v0.1"
OBSERVER_NAME = "_lumi_cpu_origin_frozen_probe_v1"
OBSERVER_PATH = "/opt/lumi-release/source/scripts/run_sft_oci_worker_probe_v1.py"
OBSERVER_SIZE = 22633
OBSERVER_SHA256 = "bed636e8cc08546a6be1b43c00c2f516fdc92dbf907dca674f7fb5aa7b5cb769"
FIXTURE_SHA256 = "53c50c92965ed1cef0c9624c57367c31ed7ebfb9d7cfb34257188ad179d333d0"
FIXTURE_SIZE = 39948
MAXIMUM_REPORT_BYTES = 16384
MAXIMUM_PACKET_BYTES = 4096
PACKET_FIELD = "cpu_origin_diagnostic"
PACKET_SCHEMA = "lumi.sft-cpu-origin-failure-packet.v0.1"

REPORT_KEYS = (
    "schema",
    "status",
    "underlying_status",
    "controls",
    "diagnostic",
    "counts",
    "frozen_stages",
    "frozen_component_diagnostics",
    "frozen_origin_counts",
    "real_sft_execution_permitted",
    "production_eligible",
    "real_model_or_data_access",
    "cuda_used",
    "complete_dynamic_dependency_closure_verified",
    "all_executable_code_origin_authenticated",
    "large_library_rehash_performed",
    "host_configuration_independently_verified",
    "real_host_admission_verified",
    "acceptance_passed",
    "recovery_performed",
)

CONTROL_KEYS = (
    "source_commit",
    "software_lock_sha256",
    "base_inventory_sha256",
    "source_inventory_sha256",
)

DIAGNOSTIC_ENUMS = {
    "captured_check": (
        "not_reported",
        "canonical_path_check_01",
        "canonical_path_check_02",
        "canonical_path_check_03",
        "pinned_bytes_check_01",
        "pinned_bytes_check_02",
        "pinned_bytes_check_03",
        "pinned_bytes_check_04",
        "pinned_bytes_check_05",
        "pinned_bytes_check_06",
        "unique_pairs_check_01",
        "control_check_01",
        "normalized_name_check_01",
        "inventory_check_01",
        "inventory_check_02",
        "inventory_check_03",
        "inventory_check_04",
        "inventory_check_05",
        "inventory_check_06",
        "inventory_check_07",
        "runtime_guard_check_01",
        "runtime_guard_check_02",
        "runtime_guard_check_03",
        "runtime_guard_check_04",
        "load_controls_check_01",
        "load_controls_check_02",
        "load_controls_check_03",
        "load_controls_check_04",
        "load_controls_check_05",
        "load_controls_check_06",
        "load_controls_check_07",
        "load_controls_check_08",
        "load_controls_check_09",
        "load_controls_check_10",
        "load_controls_check_11",
        "load_controls_check_12",
        "file_origin_check_01",
        "file_origin_check_02",
        "distribution_versions_check_01",
        "distribution_versions_check_02",
        "distribution_versions_check_03",
        "module_origins_check_01",
        "module_origins_check_02",
        "module_origins_check_03",
        "module_origins_check_04",
        "module_origins_check_05",
        "module_origins_check_06",
        "module_origins_check_07",
        "module_origins_check_08",
        "module_origins_check_09",
        "module_origins_check_10",
        "module_origins_check_11",
        "module_origins_check_12",
        "module_origins_check_13",
        "native_origins_check_01",
        "native_origins_check_02",
        "native_origins_check_03",
        "native_origins_check_04",
        "native_origins_check_05",
        "native_origins_check_06",
        "native_origins_check_07",
        "observed_outcome_check_01",
        "observed_outcome_check_02",
        "observed_invocation_check_01",
        "observed_invocation_check_02",
        "observed_invocation_check_03",
        "observed_invocation_check_04",
        "prepare_check_01",
        "prepare_check_02",
        "prepare_check_03",
        "run_check_01",
        "run_check_02",
        "entry_check_01",
        "canonical_path_unexpected",
        "pinned_bytes_unexpected",
        "control_unexpected",
        "inventory_unexpected",
        "normalized_name_unexpected",
        "file_origin_unexpected",
        "distribution_versions_unexpected",
        "module_origins_unexpected",
        "native_origins_unexpected",
        "witness_unexpected",
    ),
    "captured_function": (
        "not_reported",
        "canonical_path",
        "pinned_bytes",
        "unique_pairs",
        "control",
        "normalized_name",
        "inventory",
        "runtime_guard",
        "load_controls",
        "file_origin",
        "distribution_versions",
        "module_origins",
        "native_origins",
        "observed_outcome",
        "observed_invocation",
        "prepare",
        "run",
        "entry",
        "witness",
    ),
    "phase": ("not_reported", "prepare", "inputs", "final", "invocation", "run"),
    "origin_category": (
        "not_available",
        "not_string",
        "relative_or_symbolic",
        "source",
        "dependency",
        "runtime",
        "outside",
    ),
    "module_family": ("none", "lumi", "torch", "transformers", "other"),
    "exception_class": (
        "not_reported",
        "ProbeError",
        "AssertionError",
        "AttributeError",
        "FileNotFoundError",
        "ImportError",
        "KeyError",
        "MemoryError",
        "OSError",
        "PermissionError",
        "RuntimeError",
        "TypeError",
        "UnicodeDecodeError",
        "ValueError",
        "OtherException",
    ),
    "probe_error_code": (
        "not_reported",
        "control_rejected",
        "distribution_rejected",
        "fixture_binding_rejected",
        "fixture_failed",
        "module_origin_rejected",
        "native_origin_rejected",
        "output_failed",
        "probe_failed",
        "runtime_rejected",
        "witness_sequence_rejected",
    ),
}

COUNT_LIMITS = {
    "witnesses_started": 12,
    "witnesses_completed": 12,
    "need_checks_observed": 2000000,
    "failed_outcomes_observed": 6,
    "failure_messages_received": 12,
    "failure_records_retained": 6,
    "rank_workers_completed": 6,
    "invocations_completed": 3,
    "origin_observations_completed": 12,
}

FROZEN_STAGE_ENUMS = {
    "probe_stage": (
        "guard",
        "controls",
        "fixture_binding",
        "worker_run",
        "complete",
        "not_reported",
    ),
    "fixture_stage": (
        "guard",
        "fixture",
        "workers",
        "ledger",
        "cleanup",
        "comparison",
        "accounting",
        "complete",
        "not_reported",
    ),
    "child_stage": (
        "checks",
        "cleanup",
        "dependencies",
        "done",
        "gloo",
        "guard",
        "models",
        "result",
        "run",
        "not_reported",
    ),
    "worker_stage": (
        "admission",
        "checkpoint",
        "complete",
        "idle",
        "inputs",
        "planning",
        "restore_checkpoint",
        "restore_state",
        "runtime",
        "state",
        "step",
        "not_reported",
    ),
    "worker_code": (
        "admission_rejected",
        "already_consumed",
        "checkpoint_rejected",
        "input_rejected",
        "invalid_arguments",
        "owner_mismatch",
        "reentry",
        "reserve_failed",
        "run_failed",
        "runtime_rejected",
        "state_rejected",
        "synchronize_failed",
        "terminal_failure",
        "not_reported",
    ),
}

COMPONENT_ENUMS = {
    "checkpoint_stage": (
        "checkpoint_verify",
        "complete",
        "idle",
        "model_restore",
        "model_save",
        "prepare",
        "publication",
        "reader_setup",
        "runtime",
        "staging_verify",
        "state_save",
        "trainer_restore",
        "trainer_save",
        "writer_setup",
        "not_reported",
    ),
    "checkpoint_code": (
        "close_failed",
        "collective_failed",
        "descriptor_changed",
        "invalid_arguments",
        "invalid_descriptor",
        "operation_failed",
        "owner_mismatch",
        "read_cap_exceeded",
        "reentry",
        "reserve_failed",
        "runtime_rejected",
        "storage_rejected",
        "terminal_failure",
        "not_reported",
    ),
    "step_stage": (
        "batches",
        "complete",
        "completion",
        "forward",
        "gradient",
        "idle",
        "metrics",
        "optimizer",
        "reservation",
        "runtime",
        "settings",
        "not_reported",
    ),
    "step_code": (
        "batch_rejected",
        "invalid_arguments",
        "nonfinite_gradient",
        "nonfinite_loss",
        "owner_mismatch",
        "reentry",
        "runtime_rejected",
        "step_failed",
        "terminal_failure",
        "not_reported",
    ),
    "input_code": (
        "already_consumed",
        "input_rejected",
        "invalid_arguments",
        "load_failed",
        "owner_mismatch",
        "reentry",
        "reserve_failed",
        "runtime_rejected",
        "terminal_failure",
        "not_reported",
    ),
    "gate_stage": (
        "complete",
        "completed",
        "idle",
        "reserve",
        "reserved",
        "updated",
        "work",
        "not_reported",
    ),
    "gate_code": (
        "body_failed",
        "complete_failed",
        "invalid_arguments",
        "owner_mismatch",
        "reentry",
        "reserve_failed",
        "step_mismatch",
        "synchronize_failed",
        "terminal_failure",
        "not_reported",
    ),
}

ORIGIN_COUNT_FIELDS = (
    "file_backed_modules",
    "source_modules",
    "dependency_modules",
    "runtime_modules",
    "distribution_versions",
    "map_lines",
    "executable_file_mappings",
    "executable_files",
    "anonymous_executable_mappings",
    "generated_python_modules",
)

FALSE_FLAGS = (
    "real_sft_execution_permitted",
    "production_eligible",
    "real_model_or_data_access",
    "cuda_used",
    "complete_dynamic_dependency_closure_verified",
    "all_executable_code_origin_authenticated",
    "large_library_rehash_performed",
    "host_configuration_independently_verified",
    "real_host_admission_verified",
    "acceptance_passed",
    "recovery_performed",
)

CHECKPOINTS = (
    ("canonical_path", 100, 100, "canonical_path_check_01"),
    ("canonical_path", 101, 101, "canonical_path_check_02"),
    ("canonical_path", 104, 104, "canonical_path_check_03"),
    ("pinned_bytes", 109, 109, "pinned_bytes_check_01"),
    ("pinned_bytes", 113, 113, "pinned_bytes_check_02"),
    ("pinned_bytes", 117, 117, "pinned_bytes_check_03"),
    ("pinned_bytes", 119, 119, "pinned_bytes_check_04"),
    ("pinned_bytes", 121, 123, "pinned_bytes_check_05"),
    ("pinned_bytes", 124, 124, "pinned_bytes_check_06"),
    ("unique_pairs", 133, 133, "unique_pairs_check_01"),
    ("control", 143, 143, "control_check_01"),
    ("normalized_name", 148, 149, "normalized_name_check_01"),
    ("inventory", 154, 155, "inventory_check_01"),
    ("inventory", 158, 159, "inventory_check_02"),
    ("inventory", 161, 163, "inventory_check_03"),
    ("inventory", 164, 164, "inventory_check_04"),
    ("inventory", 167, 169, "inventory_check_05"),
    ("inventory", 172, 174, "inventory_check_06"),
    ("inventory", 175, 176, "inventory_check_07"),
    ("runtime_guard", 182, 185, "runtime_guard_check_01"),
    ("runtime_guard", 186, 187, "runtime_guard_check_02"),
    ("runtime_guard", 188, 189, "runtime_guard_check_03"),
    ("runtime_guard", 190, 192, "runtime_guard_check_04"),
    ("load_controls", 199, 202, "load_controls_check_01"),
    ("load_controls", 203, 205, "load_controls_check_02"),
    ("load_controls", 207, 209, "load_controls_check_03"),
    ("load_controls", 210, 212, "load_controls_check_04"),
    ("load_controls", 213, 215, "load_controls_check_05"),
    ("load_controls", 216, 217, "load_controls_check_06"),
    ("load_controls", 219, 222, "load_controls_check_07"),
    ("load_controls", 227, 227, "load_controls_check_08"),
    ("load_controls", 230, 230, "load_controls_check_09"),
    ("load_controls", 232, 232, "load_controls_check_10"),
    ("load_controls", 234, 235, "load_controls_check_11"),
    ("load_controls", 237, 237, "load_controls_check_12"),
    ("file_origin", 252, 252, "file_origin_check_01"),
    ("file_origin", 254, 255, "file_origin_check_02"),
    ("distribution_versions", 262, 262, "distribution_versions_check_01"),
    ("distribution_versions", 264, 264, "distribution_versions_check_02"),
    ("distribution_versions", 266, 266, "distribution_versions_check_03"),
    ("module_origins", 271, 271, "module_origins_check_01"),
    ("module_origins", 272, 272, "module_origins_check_02"),
    ("module_origins", 274, 276, "module_origins_check_03"),
    ("module_origins", 285, 285, "module_origins_check_04"),
    ("module_origins", 287, 287, "module_origins_check_05"),
    ("module_origins", 289, 291, "module_origins_check_06"),
    ("module_origins", 297, 301, "module_origins_check_07"),
    ("module_origins", 310, 311, "module_origins_check_08"),
    ("module_origins", 313, 314, "module_origins_check_09"),
    ("module_origins", 316, 317, "module_origins_check_10"),
    ("module_origins", 320, 320, "module_origins_check_11"),
    ("module_origins", 322, 323, "module_origins_check_12"),
    ("module_origins", 324, 325, "module_origins_check_13"),
    ("native_origins", 338, 338, "native_origins_check_01"),
    ("native_origins", 342, 342, "native_origins_check_02"),
    ("native_origins", 346, 346, "native_origins_check_03"),
    ("native_origins", 353, 353, "native_origins_check_04"),
    ("native_origins", 356, 358, "native_origins_check_05"),
    ("native_origins", 359, 360, "native_origins_check_06"),
    ("native_origins", 365, 365, "native_origins_check_07"),
    ("observed_outcome", 378, 378, "observed_outcome_check_01"),
    ("observed_outcome", 381, 382, "observed_outcome_check_02"),
    ("observed_invocation", 394, 394, "observed_invocation_check_01"),
    ("observed_invocation", 397, 400, "observed_invocation_check_02"),
    ("observed_invocation", 403, 404, "observed_invocation_check_03"),
    ("observed_invocation", 408, 408, "observed_invocation_check_04"),
    ("prepare", 419, 421, "prepare_check_01"),
    ("prepare", 423, 423, "prepare_check_02"),
    ("prepare", 425, 425, "prepare_check_03"),
    ("run", 435, 435, "run_check_01"),
    ("run", 438, 440, "run_check_02"),
    ("entry", 494, 495, "entry_check_01"),
)

UNEXPECTED_FUNCTIONS = (
    "canonical_path",
    "pinned_bytes",
    "control",
    "inventory",
    "normalized_name",
    "file_origin",
    "distribution_versions",
    "module_origins",
    "native_origins",
    "witness",
)

_V1 = None
_ORIGINALS = {}
_CODES = {}
_OUTCOME = None
_RECORD_FAILURE = None
_FIRST = None
_PHASE = "not_reported"
_LOCAL_COUNTS = dict.fromkeys(COUNT_LIMITS, 0)
_RECEIVED = []
_SEEN_PACKETS = set()


def increment(field, amount=1):
    # Telemetry saturation must never introduce a new acceptance rejection.
    _LOCAL_COUNTS[field] = min(COUNT_LIMITS[field], _LOCAL_COUNTS[field] + amount)


def empty_diagnostic():
    return {
        "captured_check": "not_reported",
        "captured_function": "not_reported",
        "phase": "not_reported",
        "origin_category": "not_available",
        "module_family": "none",
        "exception_class": "not_reported",
        "probe_error_code": "not_reported",
    }


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def category(value):
    # Do not resolve/stat/read a path for telemetry: the frozen predicate owns that.
    if type(value) is not str:
        return "not_string"
    if not value.startswith("/"):
        return "relative_or_symbolic"
    for prefix, label in (
        ("/opt/lumi-release/source", "source"),
        ("/usr/local/lib/python3.12/site-packages", "dependency"),
        ("/usr/local", "runtime"),
        ("/usr/lib", "runtime"),
        ("/etc/ssl", "runtime"),
        ("/usr/share/zoneinfo", "runtime"),
    ):
        if value == prefix or value.startswith(prefix + "/"):
            return label
    return "outside"


def context_categories(frame):
    origin, family = "not_available", "none"
    try:
        # Only inspect the hash-bound observer's frame dictionaries. Never retain
        # frame references, raw module names, source, paths or exception values.
        for _ in range(8):
            if frame is None:
                break
            if frame.f_code in _CODES:
                values = frame.f_locals
                if origin == "not_available":
                    for key in ("resolved", "resolved_location", "entry_path", "location", "path"):
                        if key in values:
                            origin = category(values[key])
                            break
                if family == "none" and "name" in values:
                    name = values["name"]
                    if type(name) is str:
                        family = "other"
                        for prefix in ("lumi", "torch", "transformers"):
                            if name == prefix or name.startswith(prefix + "."):
                                family = prefix
                                break
            frame = frame.f_back
    finally:
        del frame
    return origin, family


def exception_class(error):
    # Class names themselves are never serialized: exact class identity maps to
    # a closed literal vocabulary, with one fixed fallback.
    for kind, code in (
        (AssertionError, "AssertionError"),
        (AttributeError, "AttributeError"),
        (FileNotFoundError, "FileNotFoundError"),
        (ImportError, "ImportError"),
        (KeyError, "KeyError"),
        (MemoryError, "MemoryError"),
        (OSError, "OSError"),
        (PermissionError, "PermissionError"),
        (RuntimeError, "RuntimeError"),
        (TypeError, "TypeError"),
        (UnicodeDecodeError, "UnicodeDecodeError"),
        (ValueError, "ValueError"),
    ):
        if type(error) is kind:
            return code
    return "OtherException"


def captured_error_code(value):
    return (
        value
        if type(value) is str and value in DIAGNOSTIC_ENUMS["probe_error_code"]
        else "not_reported"
    )


def capture(frame, check, function, error_class, error_code):
    global _FIRST
    if _FIRST is not None:
        return
    origin, family = context_categories(frame)
    value = {
        "captured_check": check,
        "captured_function": function,
        "phase": _PHASE,
        "origin_category": origin,
        "module_family": family,
        "exception_class": error_class,
        "probe_error_code": error_code,
    }
    if valid_diagnostic(value):
        _FIRST = value


def observed_need(condition, code):
    increment("need_checks_observed")
    if not condition and _FIRST is None:
        frame = sys._getframe(1)
        try:
            function = _CODES.get(frame.f_code)
            for expected_function, start, end, check in CHECKPOINTS:
                if function == expected_function and start <= frame.f_lineno <= end:
                    capture(frame, check, function, "ProbeError", captured_error_code(code))
                    break
        finally:
            del frame
    # Preserve the exact original predicate, error type, code and rejection.
    return _ORIGINALS["need"](condition, code)


def capture_unexpected(error, fallback):
    if _FIRST is not None or type(error) is _V1.ProbeError:
        return
    trace, selected = error.__traceback__, None
    try:
        function = fallback
        for _ in range(64):
            if trace is None:
                break
            candidate = _CODES.get(trace.tb_frame.f_code)
            if candidate in UNEXPECTED_FUNCTIONS:
                selected, function = trace.tb_frame, candidate
            trace = trace.tb_next
        if selected is not None:
            capture(
                selected, function + "_unexpected", function, exception_class(error), "not_reported"
            )
    finally:
        del trace, selected


def origin_wrapper(function):
    original = _ORIGINALS[function]

    def observe(*args, **kwargs):
        if function == "witness":
            increment("witnesses_started")
        try:
            result = original(*args, **kwargs)
        except BaseException as error:
            capture_unexpected(error, function)
            raise
        if function == "witness":
            increment("witnesses_completed")
        return result

    return observe


def valid_diagnostic(value):
    return (
        type(value) is dict
        and set(value) == set(DIAGNOSTIC_ENUMS)
        and all(
            type(value[key]) is str and value[key] in codes
            for key, codes in DIAGNOSTIC_ENUMS.items()
        )
    )


def causal_diagnostic(value):
    if not valid_diagnostic(value):
        return False
    check, function = value["captured_check"], value["captured_function"]
    if function in ("not_reported", "run", "entry"):
        return False  # A downstream orchestration failure does not explain origin failure.
    if (
        function in UNEXPECTED_FUNCTIONS
        and check == function + "_unexpected"
        and value["exception_class"] not in ("not_reported", "ProbeError")
    ):
        return True
    return (
        value["exception_class"] == "ProbeError"
        and value["probe_error_code"] != "not_reported"
        and any(function == name and check == code for name, _, _, code in CHECKPOINTS)
    )


def packet():
    return {
        "schema": PACKET_SCHEMA,
        "diagnostic": dict(_FIRST or empty_diagnostic()),
        "counts": {
            key: _LOCAL_COUNTS[key]
            for key in (
                "witnesses_started",
                "witnesses_completed",
                "need_checks_observed",
                "failed_outcomes_observed",
            )
        },
    }


def valid_packet(value):
    if (
        type(value) is not dict
        or set(value) != {"schema", "diagnostic", "counts"}
        or value["schema"] != PACKET_SCHEMA
        or not valid_diagnostic(value["diagnostic"])
        or type(value["counts"]) is not dict
        or set(value["counts"])
        != {
            "witnesses_started",
            "witnesses_completed",
            "need_checks_observed",
            "failed_outcomes_observed",
        }
    ):
        return False
    counts = value["counts"]
    return (
        all(
            type(counts[key]) is int and 0 <= counts[key] <= maximum
            for key, maximum in (
                ("witnesses_started", 2),
                ("witnesses_completed", 2),
                ("need_checks_observed", 2000000),
                ("failed_outcomes_observed", 1),
            )
        )
        and counts["witnesses_completed"] <= counts["witnesses_started"]
        and counts["failed_outcomes_observed"] == 1
        and len(canonical(value)) <= MAXIMUM_PACKET_BYTES
    )


def diagnostic_outcome(pipe, value):
    global _PHASE
    if value.get("status") == "phase" and value.get("phase") == "inputs":
        _PHASE = "inputs"
    elif value.get("status") == "passed":
        _PHASE = "final"
    elif value.get("status") == "failed":
        increment("failed_outcomes_observed")
        value = dict(value, **{PACKET_FIELD: packet()})
    # The first two cases invoke both original witnesses, not stand-in checks.
    # The failure packet is the sole added wire field; no extra pipe messages.
    return _OUTCOME(pipe, value)


def diagnostic_record_failure(observed):
    if type(observed) is dict and observed.get("status") == "failed":
        increment("failure_messages_received")
        value = observed.get(PACKET_FIELD)
        if valid_packet(value):
            # De-duplicate byte-identical packets if frozen cleanup sees a report
            # twice. This bounded digest is telemetry, never a worker identity.
            digest = hashlib.sha256(canonical(value)).digest()
            if digest not in _SEEN_PACKETS and len(_RECEIVED) < 6:
                _SEEN_PACKETS.add(digest)
                _RECEIVED.append(value)
                increment("failure_records_retained")
    # Always preserve frozen child/worker/component first-failure processing.
    return _RECORD_FAILURE(observed)


def observed_invocation(*args, **kwargs):
    global _PHASE
    _PHASE = "invocation"
    return _ORIGINALS["observed_invocation"](*args, **kwargs)


def guard():
    if not (
        sys.platform == "linux"
        and os.uname().machine == "x86_64"
        and sys.implementation.name == "cpython"
        and sys.version_info[:2] == (3, 12)
        and sys.flags.isolated == 1
        and sys.dont_write_bytecode
        and os.geteuid() == os.getegid() == 10001
        and sys.argv[1:] == ["--synthetic-only"]
        and "torch" not in sys.modules
        and "transformers" not in sys.modules
        and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and os.environ.get("LUMI_WORKER_RUN_TORCH_CI_FIXTURE") == "1"
        and os.environ.get("GLOO_SOCKET_IFNAME") == "lo"
    ):
        raise ValueError("diagnostic_guard")
    if (
        str(Path(sys.executable).resolve(strict=True)) != "/usr/local/bin/python3.12"
        or sys.prefix != "/usr/local"
        or sys.base_prefix != "/usr/local"
    ):
        raise ValueError("diagnostic_guard")


def load_observer():
    # This is the only new source read. The frozen prepare still performs its
    # own original control and fixture reads once per parent/spawned interpreter.
    descriptor = os.open(OBSERVER_PATH, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != OBSERVER_SIZE:
            raise ValueError("diagnostic_binding")
        raw = bytearray()
        while len(raw) < OBSERVER_SIZE:
            block = os.read(descriptor, min(65536, OBSERVER_SIZE - len(raw)))
            if not block:
                raise ValueError("diagnostic_binding")
            raw.extend(block)
        if os.read(descriptor, 1) != b"":
            raise ValueError("diagnostic_binding")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or hashlib.sha256(raw).hexdigest() != OBSERVER_SHA256:
            raise ValueError("diagnostic_binding")
    finally:
        os.close(descriptor)
    if OBSERVER_NAME in sys.modules:
        raise ValueError("diagnostic_binding")
    spec = importlib.util.spec_from_file_location(OBSERVER_NAME, OBSERVER_PATH)
    if spec is None:
        raise ValueError("diagnostic_binding")
    observer = importlib.util.module_from_spec(spec)
    sys.modules[OBSERVER_NAME] = observer
    exec(compile(bytes(raw), OBSERVER_PATH, "exec"), observer.__dict__)
    return observer


def prepare():
    global _V1, _OUTCOME, _RECORD_FAILURE, _PHASE
    guard()
    _V1 = load_observer()
    names = {name for name, _, _, _ in CHECKPOINTS} | set(UNEXPECTED_FUNCTIONS) | {"need"}
    for name in names:
        original = getattr(_V1, name)
        _ORIGINALS[name] = original
        _CODES[original.__code__] = name
    _V1.need = observed_need
    for name in UNEXPECTED_FUNCTIONS:
        setattr(_V1, name, origin_wrapper(name))
    _V1.observed_invocation = observed_invocation
    _PHASE = "prepare"
    _V1.prepare()
    # No replacement of rank_worker, model, workload, ledger or cleanup methods.
    _OUTCOME = _V1._FIXTURE.outcome
    _RECORD_FAILURE = _V1._FIXTURE.record_failure
    _V1._FIXTURE.outcome = diagnostic_outcome
    _V1._FIXTURE.record_failure = diagnostic_record_failure


def bounded_enum(value, allowed):
    return value if type(value) is str and value in allowed else "not_reported"


def frozen_stages():
    fixture = getattr(_V1, "_FIXTURE", None)
    values = {
        "probe_stage": getattr(_V1, "_STAGE", None),
        "fixture_stage": getattr(fixture, "STAGE", None),
        "child_stage": getattr(fixture, "CHILD_STAGE", None),
        "worker_stage": getattr(fixture, "WORKER_STAGE", None),
        "worker_code": getattr(fixture, "WORKER_CODE", None),
    }
    return {key: bounded_enum(values[key], codes) for key, codes in FROZEN_STAGE_ENUMS.items()}


def frozen_components():
    values = getattr(getattr(_V1, "_FIXTURE", None), "COMPONENT_DIAGNOSTICS", {})
    if type(values) is not dict:
        values = {}
    return {key: bounded_enum(values.get(key), codes) for key, codes in COMPONENT_ENUMS.items()}


def control_pins():
    state = getattr(_V1, "_STATE", None)
    return {
        key: state.get(key, "not_reported") if type(state) is dict else "not_reported"
        for key in CONTROL_KEYS
    }


def final_counts():
    counts = dict(_LOCAL_COUNTS)
    ranks, invocations = getattr(_V1, "_RANKS", 0), getattr(_V1, "_INVOCATIONS", 0)
    counts["rank_workers_completed"] = min(6, max(0, ranks)) if type(ranks) is int else 0
    counts["invocations_completed"] = min(3, max(0, invocations)) if type(invocations) is int else 0
    counts["origin_observations_completed"] = 2 * counts["rank_workers_completed"]
    for key in ("witnesses_started", "witnesses_completed"):
        counts[key] += counts["origin_observations_completed"]
    for record in _RECEIVED:
        for key, amount in record["counts"].items():
            counts[key] += amount
    return {key: min(COUNT_LIMITS[key], value) for key, value in counts.items()}


def origin_counts():
    values = getattr(_V1, "_COUNTS", {})
    return {
        key: (
            values.get(key, 0)
            if type(values.get(key, 0)) is int and 0 <= values.get(key, 0) <= 196608
            else 0
        )
        for key in ORIGIN_COUNT_FIELDS
    }


def report(passed):
    diagnostic = _FIRST or empty_diagnostic()
    # A child causal observation precedes a later parent orchestration exception.
    for record in _RECEIVED:
        if causal_diagnostic(record["diagnostic"]):
            diagnostic = record["diagnostic"]
            break
    if passed:
        diagnostic = empty_diagnostic()
    completed = passed or causal_diagnostic(diagnostic)
    return {
        "schema": SCHEMA,
        "status": "diagnostic_completed" if completed else "diagnostic_incomplete",
        "underlying_status": "passed" if passed else "failed",
        "controls": control_pins(),
        "diagnostic": dict(diagnostic),
        "counts": final_counts(),
        "frozen_stages": frozen_stages(),
        "frozen_component_diagnostics": frozen_components(),
        "frozen_origin_counts": origin_counts(),
        **dict.fromkeys(FALSE_FLAGS, False),
    }


def valid_report(value):
    if (
        type(value) is not dict
        or set(value) != set(REPORT_KEYS)
        or value["schema"] != SCHEMA
        or value["status"] not in ("diagnostic_completed", "diagnostic_incomplete")
        or value["underlying_status"] not in ("passed", "failed")
        or any(value[key] is not False for key in FALSE_FLAGS)
        or not valid_diagnostic(value["diagnostic"])
    ):
        return False
    for section, specification in (
        ("frozen_stages", FROZEN_STAGE_ENUMS),
        ("frozen_component_diagnostics", COMPONENT_ENUMS),
    ):
        values = value[section]
        if (
            type(values) is not dict
            or set(values) != set(specification)
            or any(
                type(values[key]) is not str or values[key] not in codes
                for key, codes in specification.items()
            )
        ):
            return False
    counts = value["counts"]
    if (
        type(counts) is not dict
        or set(counts) != set(COUNT_LIMITS)
        or any(
            type(counts[key]) is not int or not 0 <= counts[key] <= maximum
            for key, maximum in COUNT_LIMITS.items()
        )
    ):
        return False
    origins = value["frozen_origin_counts"]
    if (
        type(origins) is not dict
        or set(origins) != set(ORIGIN_COUNT_FIELDS)
        or any(type(number) is not int or not 0 <= number <= 196608 for number in origins.values())
    ):
        return False
    controls = value["controls"]
    if type(controls) is not dict or set(controls) != set(CONTROL_KEYS):
        return False
    if value["status"] == "diagnostic_completed":
        if any(
            type(controls[key]) is not str
            or not re.fullmatch(
                r"[0-9a-f]{40}" if key == "source_commit" else r"[0-9a-f]{64}",
                controls[key],
            )
            for key in CONTROL_KEYS
        ):
            return False
        if value["underlying_status"] == "passed":
            if (
                counts["rank_workers_completed"] != 6
                or counts["invocations_completed"] != 3
                or counts["origin_observations_completed"] != 12
                or value["diagnostic"] != empty_diagnostic()
            ):
                return False
        elif not causal_diagnostic(value["diagnostic"]):
            return False
    return True


def entry():
    global _PHASE
    # Repeat preparation on spawn before frozen rank_worker unpickling, exactly
    # as the accepted observer does. Parent never imports an ML package.
    output = os.dup(1) if __name__ == "__main__" else None
    sink = os.open(os.devnull, os.O_WRONLY | os.O_CLOEXEC)
    try:
        os.dup2(sink, 1)
        os.dup2(sink, 2)
    finally:
        os.close(sink)
    passed = False
    try:
        prepare()
        if output is None:
            return
        _PHASE = "run"
        _V1.run()  # Exactly one started run; no retry or acceptance successor.
        passed = True
    except BaseException:
        if output is None:
            os._exit(2)
    if output is not None:
        value = report(passed)
        if not valid_report(value):
            os.close(output)
            raise SystemExit(2) from None
        raw = canonical(value) + b"\n"
        if len(raw) > MAXIMUM_REPORT_BYTES or os.write(output, raw) != len(raw):
            os.close(output)
            raise SystemExit(2) from None
        os.close(output)
        raise SystemExit(0 if value["status"] == "diagnostic_completed" else 2)


if __name__ in ("__main__", "__mp_main__"):
    entry()
