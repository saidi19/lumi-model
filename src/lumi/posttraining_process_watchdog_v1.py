"""Actual POSIX child-process watchdog, not an SFT authorization or OS sandbox.

Enforces a monotonic process lifetime and a bounded stdout/stderr read allowance,
discards child text, and kills the owned process group on ALL completion paths.
Does not contain children that deliberately escape with setsid(), enforce disk,
network, RAM or CUDA memory limits, or terminate a paid cloud instance. Linux
cgroup integration, durable rank ledger and a guarded SFT worker remain required.

The caller supplies independently pinned, trusted executable/control source and
an explicit attempt directory. No shell, arbitrary environment inheritance or
retry is provided. Tests execute only small synthetic Python children.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import signal
import stat
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

MAX_SCRIPT_BYTES = 1024 * 1024
MAX_INTEGER = 2**63 - 1
ENV_ALLOWLIST = frozenset({
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
    "TOKENIZERS_PARALLELISM", "NCCL_DEBUG", "NCCL_SOCKET_IFNAME",
})
FIXED_ENV = {
    "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
# Trusted group guardian remains alive until group cleanup, avoiding zombie-only
# PGID ownership ambiguity on Darwin. EOF of the parent-lifetime pipe also kills
# the group if the supervisor is abruptly lost. Worker cannot inherit these FDs.
GUARDIAN_SOURCE = b'''import os,select,signal,sys
try:
 life,status_fd=int(sys.argv[1]),int(sys.argv[2])
 signals=(signal.SIGTERM,signal.SIGHUP,signal.SIGINT)
 for sig in signals: signal.signal(sig,signal.SIG_IGN)
 child=os.fork()
 if child == 0:
  for sig in signals: signal.signal(sig,signal.SIG_DFL)
  os.close(life); os.close(status_fd)
  os.execve(sys.executable,[sys.executable,"-S","-B",sys.argv[3],*sys.argv[4:]],dict(os.environ))
 finished=False
 while True:
  if not finished:
   pid,status=os.waitpid(child,os.WNOHANG)
   if pid:
    result=os.waitstatus_to_exitcode(status)
    os.write(status_fd,result.to_bytes(4,"big",signed=True)); os.close(status_fd)
    finished=True
  if select.select([life],[],[],0.01)[0] and not os.read(life,1):
   os.killpg(os.getpgrp(),signal.SIGKILL)
except BaseException:
 os.killpg(os.getpgrp(),signal.SIGKILL)
'''
ERROR_CODES = frozenset({
    "contract_invalid", "platform_unsupported", "source_path_unsafe", "source_open_failed",
    "source_shape_invalid", "source_digest_mismatch", "source_changed", "attempt_path_unsafe",
    "attempt_already_exists", "attempt_create_failed", "supervisor_internal_error",
})


class WatchdogError(ValueError):
    def __init__(self, code: str):
        super().__init__(code if code in ERROR_CODES else "supervisor_internal_error")


def require(condition: bool, code: str):
    if not condition:
        raise WatchdogError(code) from None


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ProcessContract:
    python_executable: str
    script_path: str
    script_sha256: str
    arguments: tuple[str, ...]
    runtime_milliseconds: int
    maximum_stream_bytes: int
    term_grace_milliseconds: int = 100
    maximum_cleanup_milliseconds: int = 2000
    environment: tuple[tuple[str, str], ...] = ()


def validate_contract(contract: ProcessContract) -> None:
    require(type(contract) is ProcessContract, "contract_invalid")
    require(os.name == "posix", "platform_unsupported")
    require(all(hasattr(os, name) for name in ("waitid", "WNOWAIT", "WEXITED", "P_PID"))
            and threading.current_thread() is threading.main_thread()
            and signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL, "platform_unsupported")
    for name in ("python_executable", "script_path"):
        value = getattr(contract, name)
        require(type(value) is str and Path(value).is_absolute() and len(Path(value).parts) >= 3
                and str(Path(value)) == value and ".." not in Path(value).parts
                and all(ord(c) >= 32 and ord(c) != 127 for c in value), "contract_invalid")
    require(type(contract.script_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", contract.script_sha256), "contract_invalid")
    require(type(contract.arguments) is tuple and len(contract.arguments) <= 128 and all(
        type(a) is str and len(a) <= 4096 and "\x00" not in a for a in contract.arguments), "contract_invalid")
    for name, maximum in (("runtime_milliseconds", 14_400_000), ("maximum_stream_bytes", 1_048_576),
                          ("term_grace_milliseconds", 5000), ("maximum_cleanup_milliseconds", 10000)):
        value = getattr(contract, name)
        require(type(value) is int and 0 < value <= maximum, "contract_invalid")
    require(contract.term_grace_milliseconds < contract.maximum_cleanup_milliseconds, "contract_invalid")
    require(type(contract.environment) is tuple and len(contract.environment) <= len(ENV_ALLOWLIST), "contract_invalid")
    names = set()
    for item in contract.environment:
        require(type(item) is tuple and len(item) == 2, "contract_invalid")
        name, value = item
        require(type(name) is str and name in ENV_ALLOWLIST and name not in names and type(value) is str
                and len(value) <= 256 and all(32 <= ord(c) < 127 for c in value), "contract_invalid")
        names.add(name)


def _safe_regular(path: Path, maximum: int) -> bytes:
    require(path.is_absolute() and path.resolve() == path, "source_path_unsafe")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb", buffering=0) as stream:
            before = os.fstat(stream.fileno())
            require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= maximum, "source_shape_invalid")
            raw = stream.read(maximum + 1)
            after = os.fstat(stream.fileno())
        require(len(raw) == before.st_size and (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                == (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "source_changed")
        return raw
    except OSError:
        raise WatchdogError("source_open_failed") from None


def _write_new(directory_fd: int, name: str, raw: bytes) -> None:
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _kill_group(pid: int, sig: int) -> bool:
    try:
        os.killpg(pid, sig)
        return True
    except ProcessLookupError:
        return True
    except OSError:
        return False


def _cleanup(process: subprocess.Popen, grace_ms: int, cleanup_ms: int) -> dict:
    """Always kill the owned PG, even if its leader has already exited."""
    start = time.monotonic_ns()
    term = _kill_group(process.pid, signal.SIGTERM)
    grace_end = start + grace_ms * 1_000_000
    while time.monotonic_ns() < grace_end:
        time.sleep(min(0.01, max(0, (grace_end - time.monotonic_ns()) / 1e9)))
    kill = _kill_group(process.pid, signal.SIGKILL)
    remaining = max(0.001, (start + cleanup_ms * 1_000_000 - time.monotonic_ns()) / 1e9)
    reaped = False
    try:
        process.wait(timeout=remaining)
        reaped = True
    except (subprocess.TimeoutExpired, OSError):
        pass
    return {"group_term_attempted": term, "group_kill_attempted": kill, "direct_child_reaped": reaped,
            "setsid_escape_containment": False, "linux_cgroup_kill_verified": False}


def _completion_reason(exit_code: int, cancelled: bool, now_ns: int, deadline_ns: int) -> str:
    """A late success frame cannot override an already exhausted/cancelled run."""
    if cancelled:
        return "parent_interrupted"
    if now_ns >= deadline_ns:
        return "deadline_exhausted"
    return "child_completed" if exit_code == 0 else "child_failed"


def run_process(contract: ProcessContract, attempt_directory: str | Path) -> dict:
    """Run one trusted script with real process limits, retaining no child text.

The attempt is consumed before spawn, including spawn failure. Caller ownership
and OS quota/authorization checks are prerequisites outside this generic utility.
    Copied source bytes close the ordinary mutable-script race after hashing.
    Caller must retain default SIGCHLD semantics and must not independently reap
    this function's guardian. The run deadline initiates shutdown; TERM grace,
    cleanup allowance, filesystem calls and OS scheduling add lifecycle time.
"""
    validate_contract(contract)
    source = _safe_regular(Path(contract.script_path), MAX_SCRIPT_BYTES)
    require(digest(source) == contract.script_sha256, "source_digest_mismatch")
    executable = Path(contract.python_executable)
    require(executable.resolve() == executable and executable.is_file() and os.access(executable, os.X_OK), "source_path_unsafe")
    attempt = Path(attempt_directory)
    require(attempt.is_absolute() and len(attempt.parts) >= 4 and attempt.parent.resolve() == attempt.parent
            and not attempt.exists() and not attempt.is_symlink() and ".." not in attempt.parts,
            "attempt_already_exists" if attempt.exists() or attempt.is_symlink() else "attempt_path_unsafe")
    parent_fd = None
    directory_fd = None
    process = None
    life_read = life_write = status_read = status_write = None
    worker_status = bytearray()
    selector = selectors.DefaultSelector()
    observed = 0
    reason = "supervisor_internal_error"
    exit_code = None
    cleanup = {"group_term_attempted": False, "group_kill_attempted": False, "direct_child_reaped": False,
               "setsid_escape_containment": False, "linux_cgroup_kill_verified": False}
    start = time.monotonic_ns()
    deadline = start + contract.runtime_milliseconds * 1_000_000
    handled_signals = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
    previous_handlers = {sig: signal.getsignal(sig) for sig in handled_signals}
    cancelled = [False]
    def request_cancel(*_):
        # Don't raise during Popen: its child handle may not yet be assigned.
        cancelled[0] = True
    for sig in handled_signals:
        signal.signal(sig, request_cancel)
    try:
        parent_fd = os.open(attempt.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        os.mkdir(attempt.name, 0o700, dir_fd=parent_fd)
        directory_fd = os.open(attempt.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        os.fsync(parent_fd)
        _write_new(directory_fd, "started.json", json.dumps({
            "schema_version": "lumi.process-watchdog-started.v0.1", "script_sha256": contract.script_sha256,
            "retry_allowed": False, "gpu_execution_authorized_by_this_utility": False,
        }, sort_keys=True).encode() + b"\n")
        _write_new(directory_fd, "pinned_child.py", source)
        _write_new(directory_fd, "group_guardian.py", GUARDIAN_SOURCE)
        os.fsync(directory_fd)
        if cancelled[0]:
            reason = "parent_interrupted"
        elif time.monotonic_ns() >= deadline:
            reason = "deadline_exhausted_before_spawn"
        else:
            try:
                life_read, life_write = os.pipe()
                status_read, status_write = os.pipe()
                process = subprocess.Popen(
                    [str(executable), "-S", "-B", str(attempt / "group_guardian.py"), str(life_read),
                     str(status_write), str(attempt / "pinned_child.py"), *contract.arguments],
                    cwd=attempt, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    env={**FIXED_ENV, **dict(contract.environment)}, shell=False, start_new_session=True,
                    close_fds=True, pass_fds=(life_read, status_write), bufsize=0,
                )
                os.close(life_read)
                life_read = None
                os.close(status_write)
                status_write = None
            except OSError:
                reason = "spawn_failed"
            if process is not None:
                for stream in (process.stdout, process.stderr):
                    os.set_blocking(stream.fileno(), False)
                    selector.register(stream, selectors.EVENT_READ, "stream")
                os.set_blocking(status_read, False)
                selector.register(status_read, selectors.EVENT_READ, "status")
                while True:
                    now = time.monotonic_ns()
                    if cancelled[0]:
                        reason = "parent_interrupted"
                        break
                    if now >= deadline:
                        reason = "deadline_exhausted"
                        break
                    events = selector.select(min(0.025, (deadline - now) / 1e9))
                    for key, _ in events:
                        if key.data == "status":
                            chunk = os.read(status_read, 4 - len(worker_status))
                            if not chunk:
                                reason = "guardian_failed"
                                break
                            worker_status.extend(chunk)
                            if len(worker_status) == 4:
                                exit_code = int.from_bytes(worker_status, "big", signed=True)
                                reason = _completion_reason(exit_code, cancelled[0], time.monotonic_ns(), deadline)
                                break
                            continue
                        # At most cap+1 observed bytes total; contents discarded immediately.
                        chunk = os.read(key.fileobj.fileno(), min(4096, contract.maximum_stream_bytes + 1 - observed))
                        if not chunk:
                            selector.unregister(key.fileobj)
                        else:
                            observed += len(chunk)
                            if observed > contract.maximum_stream_bytes:
                                reason = "stream_budget_exhausted"
                                break
                    if reason in {"stream_budget_exhausted", "child_completed", "child_failed", "guardian_failed",
                                  "deadline_exhausted", "parent_interrupted"}:
                        break
                    # Retain the leader as our unreaped child until final PGkill.
                    # Reaping it first could allow PID/PGID reuse against another job.
                    status = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                    if status is not None:
                        # Group cleanup is mandatory even when grandchildren keep pipes open.
                        reason = "guardian_failed"
                        break
    except KeyboardInterrupt:
        reason = "parent_interrupted"
    except Exception:
        reason = "supervisor_internal_error"
    finally:
        # Cleanup must survive repeated terminal/hangup interrupts. Original
        # handlers are restored before returning to the caller.
        for sig in handled_signals:
            signal.signal(sig, signal.SIG_IGN)
        if process is not None:
            try:
                cleanup = _cleanup(process, contract.term_grace_milliseconds, contract.maximum_cleanup_milliseconds)
            except BaseException:
                reason = "cleanup_failed"
                _kill_group(process.pid, signal.SIGKILL)
                try:
                    process.wait(timeout=contract.maximum_cleanup_milliseconds / 1000)
                except BaseException:
                    pass
            # A fast-exiting child may have left unread pipe buffers. Audit those
            # after killing its group rather than declaring success too early.
            for stream in (process.stdout, process.stderr):
                try:
                    os.set_blocking(stream.fileno(), False)
                except (OSError, ValueError):
                    continue
                drain_deadline = time.monotonic_ns() + 100_000_000
                while observed <= contract.maximum_stream_bytes and time.monotonic_ns() < drain_deadline:
                    try:
                        chunk = os.read(stream.fileno(), min(4096, contract.maximum_stream_bytes + 1 - observed))
                    except (BlockingIOError, OSError):
                        break
                    if not chunk:
                        break
                    observed += len(chunk)
            if observed > contract.maximum_stream_bytes:
                reason = "stream_budget_exhausted"
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        selector.close()
        for fd in (life_read, life_write, status_read, status_write):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        if parent_fd is not None:
            os.close(parent_fd)
        for sig in handled_signals:
            signal.signal(sig, previous_handlers[sig])
    result = {
        "schema_version": "lumi.process-watchdog-result.v0.1", "reason": reason,
        "status": "PASS_child_process_only" if reason == "child_completed" and cleanup["direct_child_reaped"]
                  and cleanup["group_kill_attempted"] else "STOP_child_process",
        "child_spawned": process is not None, "exit_code": exit_code,
        "worker_status_reported": len(worker_status) == 4,
        "guardian_exit_code": process.returncode if process is not None else None,
        "observed_stream_bytes": observed, "retained_child_text_bytes": 0,
        "elapsed_milliseconds": (time.monotonic_ns() - start + 999999) // 1_000_000,
        "execution_deadline_milliseconds": contract.runtime_milliseconds,
        "cleanup_allowance_milliseconds": contract.maximum_cleanup_milliseconds,
        "retry_allowed": False, "gpu_execution_authorized_by_this_utility": False,
        "network_or_filesystem_isolation_enforced": False,
        "cloud_instance_terminated": False, "cloud_billing_stopped": False, **cleanup,
    }
    if directory_fd is not None:
        try:
            _write_new(directory_fd, "result.json", json.dumps(result, sort_keys=True).encode() + b"\n")
            os.fsync(directory_fd)
        except OSError:
            raise WatchdogError("supervisor_internal_error") from None
        finally:
            os.close(directory_fd)
    return result
