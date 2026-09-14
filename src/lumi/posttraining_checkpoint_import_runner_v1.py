"""One-shot CPU import/package coordination with a tiny-only executable boundary.

The public real-input entry point fails before I/O: no independently admitted
host or hard native-I/O/RAM/write guardian is wired yet. Synthetic execution is
strictly capped and integrates the actual kernel and finalizer, not mocks.
Coordinator reservations are logical bytes, not DCP/HF/mmap byte mediation.
The synchronous deadline cannot interrupt blocked native calls; a parent process
watchdog must independently bound every executable test. No network/GPU/training.

Trusted single-owner, quiescent input trees are required. Hash/identity checks
detect observed mutations; they do not authenticate an origin or atomically lock
a hostile owner's files. Existing attempts are permanently consumed, even after
failed creation or a partially written marker. No erase, resume or retry API.
COMPLETE is a provisional local receipt, not a commit/admission capability: its
final fsync/close or a later deadline can fail, and even FAILED may be unwritable.
Success requires the returned outcome plus a successful independent supervisor.
"""

from __future__ import annotations

import math
import os
import stat
import threading
import time
from pathlib import Path

from .posttraining_checkpoint_import_plan_v1 import build_import_plan
from .posttraining_checkpoint_stream_v1 import verify_inventory
from .posttraining_final_base_metadata_v1 import canonical, json_object, sha

SCHEMA = "lumi.sft-checkpoint-import-runner.v0.1"
STARTED = "import-started-v0.3.json"
COMPLETE = "import-complete-v0.3.json"
FAILED = "import-failed-v0.3.json"
MAXIMUM_SYNTHETIC_PARAMETERS = 1_000_000
MAXIMUM_SYNTHETIC_SOURCE_BYTES = 64 * 1024**2
MAXIMUM_SYNTHETIC_WRITE_BYTES = 128 * 1024**2
MAXIMUM_SYNTHETIC_READ_BYTES = 1024**3
MAXIMUM_CONTROL_BYTES = 131072
LIMIT_FIELDS = frozenset({
    "maximum_seconds", "maximum_coordinator_read_bytes", "maximum_private_write_bytes",
    "maximum_package_read_bytes", "maximum_package_write_bytes", "maximum_control_bytes",
})
STAGES = frozenset({"created", "inputs_before", "convert", "package", "inputs_after", "complete"})
ERROR_CODES = frozenset({
    "host_guard_not_bound", "invalid_arguments", "metadata_rejected", "synthetic_scope_rejected",
    "path_rejected", "attempt_exists_no_retry", "input_output_overlap", "read_budget_exhausted",
    "write_budget_exhausted", "deadline_exhausted", "input_audit_failed", "kernel_failed",
    "package_failed", "output_changed", "write_failed", "runner_failed",
    "single_owner_process_required",
})


class ImportRunnerError(ValueError):
    def __init__(self, code):
        self.code = code if type(code) is str and code in ERROR_CODES else "runner_failed"
        super().__init__(self.code)


def _need(value, code):
    if not value:
        raise ImportRunnerError(code) from None


def run_import(**_arguments):
    """Closed real-input boundary. Approval text/booleans cannot unlock this API."""
    raise ImportRunnerError("host_guard_not_bound") from None


def _path(path, *, directory=True, absent=False):
    _need(type(path) is type(Path()) and path.is_absolute() and ".." not in path.parts
          and len(path.parts) >= 3, "path_rejected")
    for parent in (path, *path.parents):
        _need(not parent.is_symlink(), "path_rejected")
    if absent:
        _need(not path.exists(), "attempt_exists_no_retry")
        _need(path.parent.is_dir(), "path_rejected")
        return
    info = path.lstat()
    _need((stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
          and info.st_uid == os.geteuid(), "path_rejected")
    _need(stat.S_IMODE(info.st_mode) in ({0o700} if directory else {0o400, 0o600}),
          "path_rejected")
    if not directory:
        _need(info.st_nlink == 1, "path_rejected")


def _snapshot(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _physical_overlap(inputs, attempt):
    """Reject case-insensitive/bind aliases before creating anything in an input."""
    roots = {(info.st_dev, info.st_ino) for info in (path.stat() for path in inputs)}
    _need(len(roots) == len(inputs), "input_output_overlap")
    for ancestor in attempt.parents:
        info = ancestor.stat()
        _need((info.st_dev, info.st_ino) not in roots, "input_output_overlap")
    for path in inputs:
        for ancestor in path.parents:
            info = ancestor.stat()
            _need((info.st_dev, info.st_ino) not in roots, "input_output_overlap")


class _Ledger:
    def __init__(self, limits, *, started):
        _need(type(limits) is dict and len(limits) == len(LIMIT_FIELDS)
              and set(limits) == LIMIT_FIELDS, "invalid_arguments")
        seconds = limits["maximum_seconds"]
        _need(type(seconds) in {float, int} and math.isfinite(seconds) and 0 < seconds <= 60,
              "invalid_arguments")
        for key in LIMIT_FIELDS - {"maximum_seconds"}:
            _need(type(limits[key]) is int and 0 < limits[key] <= MAXIMUM_SYNTHETIC_READ_BYTES,
                  "invalid_arguments")
        _need(limits["maximum_private_write_bytes"] <= MAXIMUM_SYNTHETIC_WRITE_BYTES
              and limits["maximum_package_write_bytes"] <= limits["maximum_private_write_bytes"]
              and limits["maximum_control_bytes"] <= MAXIMUM_CONTROL_BYTES,
              "invalid_arguments")
        self.limits, self.deadline = dict(limits), started + seconds
        self.reads = self.writes = 0
        self.failed = False

    def tick(self):
        _need(not self.failed, "runner_failed")
        if time.monotonic() >= self.deadline:
            self.failed = True
            raise ImportRunnerError("deadline_exhausted") from None

    def reserve(self, *, reads=0, writes=0):
        self.tick()
        _need(type(reads) is int and reads >= 0 and type(writes) is int and writes >= 0,
              "invalid_arguments")
        if self.reads + reads > self.limits["maximum_coordinator_read_bytes"]:
            self.failed = True
            raise ImportRunnerError("read_budget_exhausted") from None
        if self.writes + writes > self.limits["maximum_private_write_bytes"]:
            self.failed = True
            raise ImportRunnerError("write_budget_exhausted") from None
        # Reservations are consumed even when subsequent IO fails. No refunds.
        self.reads += reads
        self.writes += writes


def _write_new(root, name, raw, ledger, *, prepaid=False):
    _need(type(raw) is bytes and len(raw) <= ledger.limits["maximum_control_bytes"],
          "write_failed")
    _path(root)
    if not prepaid:
        ledger.reserve(writes=len(raw))
    fd = os.open(root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        # A short/failed write is preserved and never retried.
        _need(os.write(fd, raw) == len(raw), "write_failed")
        os.fsync(fd)
    finally:
        os.close(fd)
    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _audit(root, inventory, ledger):
    size = sum(item["bytes"] for item in inventory)
    ledger.reserve(reads=size)
    try:
        result = verify_inventory(
            root, inventory, maximum_read_bytes=size, maximum_files=len(inventory),
            maximum_seconds=max(.000001, ledger.deadline - time.monotonic()), require_private=True,
        )
    except BaseException:
        raise ImportRunnerError("input_audit_failed") from None
    ledger.tick()
    return result


def _identities(root, inventory):
    """Bounded metadata observations; not a lock or claim about transient changes."""
    result = {".": _snapshot(root.lstat())}
    for item in inventory:
        path = root / item["path"]
        _path(path, directory=False)
        result[item["path"]] = _snapshot(path.lstat())
    return result


def run_synthetic_import(*, metadata: tuple[bytes, ...], expected_base_sha256: str,
                         checkpoint_root: Path, tokenizer_root: Path, attempt: Path,
                         dtype: str, limits: dict) -> dict:
    """Integrate genuine tiny CPU fixtures only; callers cannot opt into real scale.

    `metadata` is the exact base, manifest, state, config, handoff byte tuple.
    The original base-file pin is independent of this tuple. Tokenizer root must
    contain exactly tokenizer/ plus its three bound files, not inferred names.
    The kernel imports ML lazily after the durable marker and first input audit.
    """
    started, stage, created = time.monotonic(), "created", False
    ledger, terminal_reserved = None, False
    try:
        _need(threading.current_thread() is threading.main_thread()
              and threading.active_count() == 1, "single_owner_process_required")
        ledger = _Ledger(limits, started=started)
        limits = ledger.limits  # Never use caller-mutable limits after admission.
        _need(type(metadata) is tuple and len(metadata) == 5
              and all(type(raw) is bytes for raw in metadata), "invalid_arguments")
        try:
            plan = build_import_plan(*metadata, expected_base_sha256=expected_base_sha256,
                                     dtype=dtype)
            base = json_object(metadata[0])
        except BaseException:
            raise ImportRunnerError("metadata_rejected") from None
        bounds = plan["resource_bounds"]
        _need(base.get("synthetic_fixture_only") is True
              and plan["model_parameters"] <= MAXIMUM_SYNTHETIC_PARAMETERS
              and bounds["checkpoint_inventory_bytes"] + bounds["tokenizer_inventory_bytes"]
                  <= MAXIMUM_SYNTHETIC_SOURCE_BYTES
              and bounds["tokenizer_inventory_bytes"] <= 8 * 1024**2
              and bounds["maximum_backend_output_bytes"] <= 32 * 1024**2,
              "synthetic_scope_rejected")
        _need(bounds["checkpoint_audit_read_bytes"] + bounds["tokenizer_audit_read_bytes"]
              + limits["maximum_package_read_bytes"] <= limits["maximum_coordinator_read_bytes"],
              "read_budget_exhausted")
        _need(bounds["maximum_backend_output_bytes"] + limits["maximum_package_write_bytes"]
              + len(metadata[3]) + 3 * limits["maximum_control_bytes"]
              <= limits["maximum_private_write_bytes"], "write_budget_exhausted")
        _path(checkpoint_root)
        _path(tokenizer_root)
        _path(attempt, absent=True)
        paths = (checkpoint_root, tokenizer_root, attempt)
        _need(all(not a.is_relative_to(b) and not b.is_relative_to(a)
                  for i, a in enumerate(paths) for b in paths[i + 1:]), "input_output_overlap")
        _physical_overlap((checkpoint_root, tokenizer_root), attempt)
        ledger.tick()
        parent = attempt.parent
        parent_before = parent.stat()
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            _need(_snapshot(os.fstat(parent_fd)) == _snapshot(parent_before), "path_rejected")
            os.mkdir(attempt.name, 0o700, dir_fd=parent_fd)
            created = True
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        # Prepay both terminal envelopes; a failure can be durably recorded even
        # after the deadline/read budget is exhausted, with no extra authorization.
        ledger.reserve(writes=2 * limits["maximum_control_bytes"])
        terminal_reserved = True
        marker_raw = canonical({"schema_version": SCHEMA, "stage": "created",
                                "plan_sha256": plan["plan_sha256"],
                                "base_file_sha256": expected_base_sha256,
                                "synthetic_only": True, "retry_allowed": False,
                                "training_authorized": False, "production_eligible": False}) + b"\n"
        _write_new(attempt, STARTED, marker_raw, ledger)
        marker_before = (attempt / STARTED).lstat()
        _write_new(attempt, "source-config.json", metadata[3], ledger)
        config_before = (attempt / "source-config.json").lstat()
        stage = "inputs_before"
        before_source = _audit(checkpoint_root, plan["checkpoint_inventory"], ledger)
        before_tokens = _audit(tokenizer_root, plan["tokenizer_inventory"], ledger)
        source_identities = _identities(checkpoint_root, plan["checkpoint_inventory"])
        token_identities = _identities(tokenizer_root, plan["tokenizer_inventory"])
        stage = "convert"
        ledger.reserve(writes=bounds["maximum_backend_output_bytes"])
        from .posttraining_checkpoint_import_torch_v3 import convert_and_verify
        try:
            kernel = convert_and_verify(
                attempt / "source-config.json", checkpoint_root, attempt / "checkpoint", dtype,
                maximum_parameters=plan["model_parameters"],
                maximum_tensor_bytes=bounds["maximum_projected_live_tensor_bytes"],
                maximum_output_bytes=bounds["maximum_backend_output_bytes"],
                check_budget=ledger.tick,
            )
        except BaseException:
            raise ImportRunnerError("kernel_failed") from None
        ledger.tick()
        stage = "package"
        ledger.reserve(reads=limits["maximum_package_read_bytes"],
                       writes=limits["maximum_package_write_bytes"])
        from .posttraining_checkpoint_package_v1 import finalize_package
        try:
            package = finalize_package(
                output=attempt / "checkpoint", tokenizer_root=tokenizer_root,
                plan=plan, expected_plan_sha256=plan["plan_sha256"], kernel_evidence=kernel,
                maximum_read_bytes=limits["maximum_package_read_bytes"],
                maximum_write_bytes=limits["maximum_package_write_bytes"],
                maximum_control_bytes=limits["maximum_control_bytes"], check_budget=ledger.tick,
            )
        except BaseException:
            raise ImportRunnerError("package_failed") from None
        stage = "inputs_after"
        _need(_audit(checkpoint_root, plan["checkpoint_inventory"], ledger) == before_source
              and _audit(tokenizer_root, plan["tokenizer_inventory"], ledger) == before_tokens,
              "input_audit_failed")
        _need(_identities(checkpoint_root, plan["checkpoint_inventory"]) == source_identities
              and _identities(tokenizer_root, plan["tokenizer_inventory"]) == token_identities,
              "input_audit_failed")
        _need(_snapshot((attempt / STARTED).lstat()) == _snapshot(marker_before)
              and _snapshot((attempt / "source-config.json").lstat()) == _snapshot(config_before),
              "output_changed")
        ledger.tick()
        stage = "complete"
        result = {
            "schema_version": SCHEMA, "status": "PASS_synthetic_coordinator_not_real_import",
            "stage": stage, "plan_sha256": plan["plan_sha256"],
            "kernel_evidence_sha256": sha(canonical(kernel)), "package": package,
            "full_input_audits_completed": 2, "observed_input_identities_unchanged": True,
            "atomic_input_immutability_verified": False,
            "coordinator_read_bytes_reserved": ledger.reads,
            "private_write_bytes_reserved": ledger.writes,
            "reservations_refunded": False, "retry_allowed": False,
            "synthetic_only": True, "native_backend_read_bytes_metered": False,
            "durable_completion_is_execution_authority": False,
            "success_requires_returned_outcome_and_parent_supervisor": True,
            "os_memory_write_limits_enforced": False, "hard_timeout_enforced_here": False,
            "real_4b_import": False, "training_authorized": False, "production_eligible": False,
        }
        ledger.tick()
        _write_new(attempt, COMPLETE, canonical(result) + b"\n", ledger, prepaid=True)
        ledger.tick()
        return result
    except BaseException as error:
        code = error.code if type(error) is ImportRunnerError else "runner_failed"
        if created and ledger is not None and terminal_reserved:
            try:
                _write_new(attempt, FAILED, canonical({"schema_version": SCHEMA, "stage": stage,
                    "code": code, "retry_allowed": False, "training_authorized": False,
                    "production_eligible": False}) + b"\n", ledger, prepaid=True)
            except BaseException:
                pass  # Preserve the consumed directory/marker even if failure recording fails.
        raise ImportRunnerError(code) from None
