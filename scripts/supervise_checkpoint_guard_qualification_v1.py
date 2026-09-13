#!/usr/bin/env python3
"""One fixed Linux CPU harness under an independent, bounded PID watchdog.

The trusted host checkout and /usr/bin/python3 are premises, not authenticated
runtime or immutable-source claims. Only the directly owned harness pidfd may
be signalled. Its independent guardian retains responsibility for its cgroup;
this parent only reaps adopted children and never kills a guardian/process group.
No native entry is exercised by local source/double tests.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import re
import select
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
HARNESS = ROOT / "scripts/run_checkpoint_guard_qualification_v1.py"
PYTHON = "/usr/bin/python3"
SCHEMA = "lumi.checkpoint-guard-supervised-qualification.v0.1"
HARNESS_SCHEMA = "lumi.checkpoint-guard-native-qualification.v0.1"
FIXED_ENV = {"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8",
             "LUMI_CHECKPOINT_GUARD_QUALIFICATION_CI": "1"}
MAXIMUM_HARNESS_SECONDS = 900
CLEANUP_SECONDS = 120
TERM_SECONDS = 10
MAXIMUM_SOURCE_BYTES = 131072
MAXIMUM_OUTPUT_BYTES = 32768
MAXIMUM_REAPED_CHILDREN = 2048
FALSE_FLAGS = ("real_4b_import", "training_authorized", "production_eligible",
               "gpu_used", "real_model_or_data_access")
ERROR_CODES = frozenset({"ok", "runtime_rejected", "arguments_rejected", "source_rejected",
                         "already_attempted", "process_failed", "deadline_exhausted",
                         "output_rejected", "cleanup_unconfirmed", "supervision_failed"})
_ATTEMPTED = False


class SupervisorError(ValueError):
    def __init__(self, code="supervision_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "supervision_failed"
        super().__init__(self.code)


def need(value, code):
    if not value:
        raise SupervisorError(code) from None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"


def _state(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _path(path, *, maximum=None):
    need(type(path) is str and len(path) <= 4096, "arguments_rejected")
    selected = Path(path)
    need(selected.is_absolute() and ".." not in selected.parts, "arguments_rejected")
    for ancestor in reversed((selected, *selected.parents)):
        need(not stat.S_ISLNK(ancestor.lstat().st_mode), "source_rejected")
    info = selected.lstat()
    need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
         and not info.st_mode & 0o022 and info.st_size > 0, "source_rejected")
    need(maximum is None or info.st_size <= maximum, "source_rejected")
    return _state(info)


def verify_harness(expected):
    need(type(expected) is str and re.fullmatch(r"[a-f0-9]{64}", expected),
         "arguments_rejected")
    before = _path(str(HARNESS), maximum=MAXIMUM_SOURCE_BYTES)
    descriptor = os.open(HARNESS, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        need(_state(os.fstat(descriptor)) == before, "source_rejected")
        digest, total = hashlib.sha256(), 0
        while total < before[6]:
            raw = os.read(descriptor, min(65536, before[6] - total))
            need(raw, "source_rejected")
            digest.update(raw)
            total += len(raw)
        need(not os.read(descriptor, 1) and digest.hexdigest() == expected
             and _state(os.fstat(descriptor)) == before
             and _path(str(HARNESS), maximum=MAXIMUM_SOURCE_BYTES) == before,
             "source_rejected")
    finally:
        os.close(descriptor)
    return before


def runtime():
    # Ignored SIGCHLD can make Popen.poll() synthesize a zero exit after ECHILD.
    need(sys.platform == "linux"
         and signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL, "runtime_rejected")
    need(platform.machine() == "x86_64"
         and sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12)
         and sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
         and os.getresuid() == os.getresgid() == (0, 0, 0)
         and threading.current_thread() is threading.main_thread()
         and threading.active_count() == 1
         and hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal")
         and dict(os.environ) == FIXED_ENV
         and not any(name.split(".", 1)[0] in {
             "lumi", "torch", "transformers", "numpy", "safetensors", "tokenizers",
             "tensorflow", "jax", "cuda", "cupy",
         } for name in sys.modules), "runtime_rejected")
    task = Path("/proc/self/task")
    need(sorted(item.name for item in task.iterdir()) == [str(os.getpid())], "runtime_rejected")
    with (task / str(os.getpid()) / "children").open("rb") as source:
        need(source.read(4097) == b"", "runtime_rejected")


def subreaper():
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.restype = ctypes.c_int
    observed = ctypes.c_int()
    need(prctl(36, 1, 0, 0, 0) == 0, "runtime_rejected")
    need(prctl(37, ctypes.byref(observed), 0, 0, 0) == 0 and observed.value == 1,
         "runtime_rejected")


def _pairs(rows):
    value = {}
    for key, part in rows:
        need(key not in value, "output_rejected")
        value[key] = part
    return value


def _integer(raw):
    need(len(raw) <= 19, "output_rejected")
    value = int(raw)
    need(0 <= value < 2**63, "output_rejected")
    return value


def _number(_raw):
    raise SupervisorError("output_rejected") from None


def parse_report(raw, returncode):
    need(type(raw) is bytes and 0 < len(raw) <= MAXIMUM_OUTPUT_BYTES
         and raw.isascii(), "output_rejected")
    try:
        value = json.loads(raw, object_pairs_hook=_pairs, parse_int=_integer,
                           parse_float=_number, parse_constant=_number)
        need(type(value) is dict and value.get("schema") == HARNESS_SCHEMA
             and value.get("status") in {"passed", "failed"}
             and ((value["status"] == "passed") == (returncode == 0))
             and all(value.get(key) is False for key in FALSE_FLAGS), "output_rejected")
        need(value["status"] != "passed" or (value.get("code") == "ok"
             and value.get("stage") == "complete"), "output_rejected")
        pending, count = [(value, 0)], 0
        while pending:
            item, depth = pending.pop()
            count += 1
            need(count <= 1024 and depth <= 8, "output_rejected")
            if type(item) is dict:
                need(all(type(key) is str and 0 < len(key) <= 256
                         and all(32 <= ord(char) <= 126 and char not in "/\\"
                                 for char in key) for key in item),
                     "output_rejected")
                pending.extend((part, depth + 1) for part in item.values())
            elif type(item) is list:
                need(len(item) <= 64, "output_rejected")
                pending.extend((part, depth + 1) for part in item)
            else:
                need(type(item) in (str, int, bool, type(None)), "output_rejected")
                need(type(item) is not str or (len(item) <= 256
                     and all(32 <= ord(char) <= 126 and char not in "/\\" for char in item)),
                     "output_rejected")
        need(raw in (canonical(value), canonical(value).rstrip(b"\n")), "output_rejected")
        return value
    except SupervisorError:
        raise
    except BaseException:
        raise SupervisorError("output_rejected") from None


def _reap(deadline):
    count, failed = 0, False
    try:
        if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
            return count, False
    except BaseException:
        return count, False
    while time.monotonic() < deadline:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return count, not failed
        except BaseException:
            return count, False
        if pid:
            count += 1
            failed |= status != 0
            if count >= MAXIMUM_REAPED_CHILDREN:
                return count, False
        else:
            select.select([], [], [], min(0.05, max(0, deadline - time.monotonic())))
    return count, False


def watch(process, pidfd, started):
    """No signal target is accepted except this just-created harness pidfd."""
    deadline = started + MAXIMUM_HARNESS_SECONDS
    final_deadline = deadline + CLEANUP_SECONDS
    stopping = None
    failure, term, kill = (None if pidfd is not None else "cleanup_unconfirmed"), 0, 0
    raw, eof, returncode = bytearray(), False, None
    descriptor = None
    try:
        descriptor = process.stdout.fileno()
        os.set_blocking(descriptor, False)
    except BaseException:
        failure, eof = "supervision_failed", True
    while time.monotonic() < final_deadline:
        try:
            returncode = process.poll()
        except BaseException:
            failure, returncode = "cleanup_unconfirmed", None
        now = time.monotonic()
        if now >= deadline and failure is None:
            failure = "deadline_exhausted"
        if returncode is not None and eof:
            final_deadline = min(final_deadline, now + CLEANUP_SECONDS)
            break
        if returncode is not None and stopping is None:
            stopping = now
            final_deadline = min(final_deadline, now + CLEANUP_SECONDS)
        if failure and stopping is None:
            stopping = now
            final_deadline = min(final_deadline, now + CLEANUP_SECONDS)
        if stopping is not None and returncode is None:
            if not term:
                term = 1
                try:
                    need(pidfd is not None, "cleanup_unconfirmed")
                    signal.pidfd_send_signal(pidfd, signal.SIGTERM, None, 0)
                except BaseException:
                    failure = "cleanup_unconfirmed"
            elif not kill and now >= stopping + TERM_SECONDS:
                kill = 1
                try:
                    need(pidfd is not None, "cleanup_unconfirmed")
                    signal.pidfd_send_signal(pidfd, signal.SIGKILL, None, 0)
                except BaseException:
                    failure = "cleanup_unconfirmed"
        readable = [] if eof else [descriptor]
        try:
            ready = select.select(readable, [], [], min(0.05, final_deadline - now))[0]
            if ready:
                part = os.read(descriptor, min(4096, MAXIMUM_OUTPUT_BYTES + 1 - len(raw)))
                if part:
                    raw.extend(part)
                    if len(raw) > MAXIMUM_OUTPUT_BYTES:
                        failure, eof = "output_rejected", True
                else:
                    eof = True
        except BaseException:
            failure, eof = "supervision_failed", True
    try:
        returncode = process.poll()
    except BaseException:
        returncode = None
    if returncode is None or not eof:
        failure = "cleanup_unconfirmed"
    adopted, reaped = _reap(final_deadline) if returncode is not None else (0, False)
    # A successful harness must reap its own descendants. Even a zero-exit
    # adopted child means its reported complete cleanup was not observed here.
    if adopted or not reaped:
        failure = "cleanup_unconfirmed"
    return {"raw": bytes(raw), "returncode": returncode, "failure": failure,
            "harness_term_attempts": term, "harness_kill_attempts": kill,
            "harness_reaped": returncode is not None, "adopted_children_reaped": adopted,
            "all_children_reaped": reaped}


def supervise(archive, expected):
    global _ATTEMPTED
    runtime()
    need(not _ATTEMPTED, "already_attempted")
    source_state = verify_harness(expected)
    archive_state = _path(archive)
    subreaper()
    need(_path(str(HARNESS), maximum=MAXIMUM_SOURCE_BYTES) == source_state
         and _path(archive) == archive_state, "source_rejected")
    _ATTEMPTED = True
    started = time.monotonic()
    process = subprocess.Popen(
        [PYTHON, "-I", "-S", "-B", str(HARNESS), "--synthetic-only", archive],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        close_fds=True, start_new_session=True, env=dict(FIXED_ENV), cwd=ROOT,
    )
    pidfd, result, close_failed = None, None, False
    try:
        try:
            pidfd = os.pidfd_open(process.pid, 0)
        except BaseException:
            pass  # No numeric-PID fallback, even for this known child.
        result = watch(process, pidfd, started)
    finally:
        try:
            process.stdout.close()
        except BaseException:
            close_failed = True
        if pidfd is not None:
            try:
                os.close(pidfd)
            except BaseException:
                close_failed = True
    need(result is not None, "supervision_failed")
    raw, returncode = result.pop("raw"), result.pop("returncode")
    failure = result.pop("failure")
    if pidfd is None or close_failed:
        failure = "cleanup_unconfirmed"
    report = None
    if not failure:
        try:
            report = parse_report(raw, returncode)
        except SupervisorError:
            failure = "output_rejected"
    if not failure and returncode != 0:
        failure = "process_failed"
    return {"schema": SCHEMA, "status": "failed" if failure else "passed",
            "code": failure or "ok", "harness_sha256": expected,
            "harness_stdout_bytes": len(raw),
            "harness_stdout_sha256": hashlib.sha256(raw).hexdigest(),
            "harness_exit_success": returncode == 0, "harness_pidfd_bound": pidfd is not None,
            **result, "harness_report": report,
            "guardian_signalled_by_supervisor": False, "process_group_signalled": False,
            "guardian_death_cleanup_verified": False, "job_teardown_cleanup_verified": False,
            "source_immutability_verified": False, "real_4b_import": False,
            "training_authorized": False, "production_eligible": False, "gpu_used": False}


def main():
    try:
        need(len(sys.argv) == 4 and sys.argv[1] == "--synthetic-only", "arguments_rejected")
        result = supervise(sys.argv[2], sys.argv[3])
        need(len(canonical(result)) <= MAXIMUM_OUTPUT_BYTES, "output_rejected")
    except BaseException as error:
        code = error.code if type(error) is SupervisorError else "supervision_failed"
        result = {"schema": SCHEMA, "status": "failed", "code": code,
                  "real_4b_import": False, "training_authorized": False,
                  "production_eligible": False, "gpu_used": False,
                  "guardian_death_cleanup_verified": False, "job_teardown_cleanup_verified": False}
    raw = canonical(result)
    need(len(raw) <= MAXIMUM_OUTPUT_BYTES, "output_rejected")
    sys.stdout.buffer.write(raw)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
