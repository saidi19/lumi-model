"""Fixed independent controller process and inherited-only bounded FD protocol.

The executable path CAN perform real provider I/O with valid supplied authority.
It is not permission to run: no real credential, bootstrap or cloud call is used
by this source checkpoint. A trusted caller supplies already-open private FDs.
No secrets in argv/environment/stdout, no arbitrary worker command, no restart.
Process separation is not controller survival, containment or billing insurance.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import select
import signal
import stat
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

# Fixed file execution under -I -S, without importing the ML package initializer.
if not __package__:
    _package_name = "_lumi_controller_process_runtime"
    _package = types.ModuleType(_package_name)
    _package.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules[_package_name] = _package
    __package__ = _package_name

from . import posttraining_sft_controller_service_v1 as service  # noqa: E402
from . import posttraining_sft_provider_controller_v1 as provider  # noqa: E402

BOOTSTRAP_SCHEMA = "lumi.sft-controller-bootstrap.v0.1"
BOOTSTRAP_KEYS = frozenset({
    "schema_version", "plan_ascii", "plan_sha256", "authority_ascii", "authority_sha256",
    "authority_key_hex", "authority_key_sha256", "launch_ascii", "token_ascii",
})
MAXIMUM_BOOTSTRAP_BYTES = 32768
MAXIMUM_REPORT_BYTES = 4096
MAXIMUM_READS = 128
ERROR_CODES = frozenset({"process_configuration_rejected", "bootstrap_rejected",
                         "process_spawn_rejected", "process_io_rejected", "process_failed",
                         "result_unavailable", "result_rejected", "signal_scope_rejected"})


class ProcessError(ValueError):
    def __init__(self, code="process_configuration_rejected"):
        self.code = code if type(code) is str and code in ERROR_CODES else (
            "process_configuration_rejected")
        super().__init__(self.code)


def need(condition, code="process_configuration_rejected"):
    if not condition:
        raise ProcessError(code) from None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _fd(value):
    need(type(value) is int and 3 <= value < 2**20)
    return os.fstat(value)


def _pipe(value, access):
    info = _fd(value)
    flags = fcntl.fcntl(value, fcntl.F_GETFL)
    need(stat.S_ISFIFO(info.st_mode) and flags & os.O_ACCMODE == access
         and flags & os.O_NONBLOCK)


def _root(value):
    info = _fd(value)
    need(stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o700
         and info.st_uid == os.geteuid())


def _close(fd):
    try:
        os.close(fd)
        return True
    except BaseException:
        return False


def _duration(value, maximum):
    need(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= maximum)


def _decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "result_rejected")
            result[key] = value
        return result

    def constant(_value):
        raise ProcessError("result_rejected")

    value = json.loads(raw.decode("ascii"), object_pairs_hook=pairs, parse_constant=constant)
    need(canonical(value) == raw, "result_rejected")
    return value


def decode_bootstrap(raw):
    """Pure strict framing only; ProviderController verifies MAC/plan/launch bindings."""
    failed = False
    result = None
    try:
        need(type(raw) is bytes and 0 < len(raw) <= MAXIMUM_BOOTSTRAP_BYTES)
        value = _decode(raw)
        need(type(value) is dict and value.keys() == BOOTSTRAP_KEYS
             and value["schema_version"] == BOOTSTRAP_SCHEMA)
        for field, maximum in (("plan_ascii", 16384), ("authority_ascii", 8192),
                               ("launch_ascii", 4096), ("token_ascii", 512)):
            need(type(value[field]) is str and 0 < len(value[field]) <= maximum
                 and value[field].isascii())
        for field in ("plan_sha256", "authority_sha256", "authority_key_sha256",
                      "authority_key_hex"):
            need(provider.http.matches(value[field], provider.http.HASH))
        result = {
            "plan_bytes": value["plan_ascii"].encode("ascii"),
            "plan_sha256": value["plan_sha256"],
            "authority_bytes": value["authority_ascii"].encode("ascii"),
            "authority_sha256": value["authority_sha256"],
            "authority_key": bytes.fromhex(value["authority_key_hex"]),
            "authority_key_sha256": value["authority_key_sha256"],
            "launch_body": value["launch_ascii"].encode("ascii"),
            "token": value["token_ascii"].encode("ascii"),
        }
    except BaseException:
        failed = True
    if failed:
        raise ProcessError("bootstrap_rejected") from None
    return result


def read_bootstrap(fd):
    """Consume one bounded supplied nonblocking pipe through EOF, never a path."""
    raw = bytearray()
    failed = False
    try:
        _pipe(fd, os.O_RDONLY)
        deadline = time.monotonic() + 5.0
        for _ in range(MAXIMUM_READS):
            remaining = deadline - time.monotonic()
            need(remaining > 0, "bootstrap_rejected")
            ready, _, _ = select.select([fd], [], [], remaining)
            need(bool(ready), "bootstrap_rejected")
            block = os.read(fd, min(4096, MAXIMUM_BOOTSTRAP_BYTES + 1 - len(raw)))
            need(time.monotonic() < deadline, "bootstrap_rejected")
            if not block:
                return decode_bootstrap(bytes(raw))
            raw.extend(block)
            need(len(raw) <= MAXIMUM_BOOTSTRAP_BYTES, "bootstrap_rejected")
        failed = True
    except BaseException:
        failed = True
    finally:
        raw.clear()
    if failed:
        raise ProcessError("bootstrap_rejected") from None


class ControlReader:
    """Borrow a readonly nonblocking pipe; own only a CLOEXEC duplicate.

    Stop/EOF/fault is absorbing. There is no start, target, work or authority command.
    The caller must close unused pipe endpoints; endpoint secrecy is a prerequisite.
    """

    def __init__(self, fd):
        _pipe(fd, os.O_RDONLY)
        self._fd = os.dup(fd)
        try:
            os.set_inheritable(self._fd, False)
        except BaseException:
            _close(self._fd)
            raise ProcessError("process_io_rejected") from None
        self._signaled = False
        self._latched = None
        self._closed = False
        self._owner = (os.getpid(), threading.current_thread())

    def poll(self, timeout_seconds):
        try:
            need(self._owner == (os.getpid(), threading.current_thread()))
            _duration(timeout_seconds, 0.25)
        except BaseException:
            self._latched = "fault"
            return self._latched
        if self._latched is not None:
            # EOF/stop stays ready forever. Honor the wait without rereading it,
            # or terminal polling would busy-spin and exhaust the control budget.
            if timeout_seconds:
                try:
                    select.select([], [], [], timeout_seconds)
                except BaseException:
                    self._latched = "fault"
            return self._latched
        if self._signaled:
            self._latched = "signal"
        elif self._closed:
            self._latched = "fault"
        else:
            try:
                ready, _, _ = select.select([self._fd], [], [], timeout_seconds)
                if self._signaled:
                    self._latched = "signal"
                elif ready:
                    value = os.read(self._fd, 2)
                    self._latched = {b"S": "stop", b"": "eof"}.get(value, "fault")
            except BaseException:
                self._latched = "fault"
        return self._latched or "continue"

    def close(self):
        if self._owner != (os.getpid(), threading.current_thread()):
            self._latched = "fault"
            return False
        if self._closed:
            return True
        self._closed = True
        return _close(self._fd)


class SignalStop:
    """Main-thread temporary handlers only latch cancellation; no I/O in a handler."""

    def __init__(self, reader):
        need(type(reader) is ControlReader, "signal_scope_rejected")
        self._reader = reader
        self._previous = []

    def _handler(self, _number, _frame):
        self._reader._signaled = True

    def __enter__(self):
        need(threading.current_thread() is threading.main_thread(), "signal_scope_rejected")
        failed = False
        try:
            for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                previous = signal.signal(number, self._handler)
                self._previous.append((number, previous))
        except BaseException:
            failed = True
        if failed:
            self.__exit__(None, None, None)
            raise ProcessError("signal_scope_rejected") from None
        return self

    def __exit__(self, *_args):
        failed = False
        previous, self._previous = self._previous, []
        for number, handler in reversed(previous):
            try:
                signal.signal(number, handler)
            except BaseException:
                failed = True
        if failed:
            raise ProcessError("signal_scope_rejected") from None


def write_report(fd, report):
    """One prepaid <=4096-byte write; no retry after a partial/failed write."""
    try:
        _pipe(fd, os.O_WRONLY)
        raw = canonical(service.validate_report(report)) + b"\n"
        need(len(raw) <= MAXIMUM_REPORT_BYTES)
        return os.write(fd, raw) == len(raw)
    except BaseException:
        return False


def _failure_report(code):
    return {
        "schema_version": service.SCHEMA, "status": "reconciliation_required",
        "stop_code": code, "modeled_terminal": False, "reconciliation_required": True,
        "local_controller_closed": False, "provider_requests": None, "control_polls": None,
        "service_actions": None, "journal_records": None,
        "execution_authorized": False, "provider_termination_verified": False,
        "cloud_billing_stopped": False, "independent_survival_verified": False,
        "production_eligible": False,
    }


class OwnedControllerProcess:
    """Own child and parent endpoints; unavailable output is never a cleanup receipt.

    Timeout leaves the child running for owned cleanup, not killed/restarted. A
    later collect may finish reading the SAME bounded result. Caller abandonment,
    SIGKILL and machine failure are not recoverable by this library.
    """

    def __init__(self, process, control_writer, result_reader):
        _pipe(control_writer, os.O_WRONLY)
        _pipe(result_reader, os.O_RDONLY)
        need(control_writer != result_reader)
        self._process = process
        self._control = control_writer
        self._result = result_reader
        self._raw = bytearray()
        self._reads = 0
        self._eof = self._failed = False
        self._completed = None
        self._owner = (os.getpid(), threading.current_thread())

    def __repr__(self):
        return "<OwnedControllerProcess: redacted>"

    def _check_owner(self):
        need(self._owner == (os.getpid(), threading.current_thread()), "process_io_rejected")

    def request_stop(self):
        self._check_owner()
        if self._control is None:
            return True
        fd, self._control = self._control, None
        sent = False
        try:
            sent = os.write(fd, b"S") == 1
        except BaseException:
            pass
        closed = _close(fd)
        return sent and closed

    def disconnect(self):
        self._check_owner()
        if self._control is None:
            return True
        fd, self._control = self._control, None
        return _close(fd)

    def collect(self, timeout_seconds=1.0):
        self._check_owner()
        _duration(timeout_seconds, 30)
        need(timeout_seconds > 0, "result_unavailable")
        need(not self._failed, "result_rejected")
        if self._completed is not None:
            return dict(self._completed)
        error = None
        report = None
        try:
            deadline = time.monotonic() + timeout_seconds
            while not self._eof:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                ready, _, _ = select.select([self._result], [], [], remaining)
                if not ready:
                    raise TimeoutError
                need(self._reads < MAXIMUM_READS, "result_rejected")
                self._reads += 1
                block = os.read(self._result, min(4096, MAXIMUM_REPORT_BYTES + 1 - len(self._raw)))
                if not block:
                    self._eof = True
                    fd, self._result = self._result, None
                    need(_close(fd), "process_io_rejected")
                else:
                    self._raw.extend(block)
                    need(len(self._raw) <= MAXIMUM_REPORT_BYTES, "result_rejected")
            need(self._raw.endswith(b"\n"), "result_rejected")
            report = service.validate_report(_decode(bytes(self._raw[:-1])))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            code = self._process.wait(timeout=remaining)
            need(type(code) is int and code == 0, "process_failed")
            need(time.monotonic() < deadline, "result_unavailable")
            need(self.disconnect(), "process_io_rejected")
        except (TimeoutError, subprocess.TimeoutExpired):
            error = "result_unavailable"
        except BaseException:
            self._failed = True
            self._raw.clear()
            if self._result is not None:
                fd, self._result = self._result, None
                _close(fd)
            error = "result_rejected"
        if error is not None:
            raise ProcessError(error) from None
        self._raw.clear()
        self._completed = dict(report)
        return dict(self._completed)


def spawn_controller(root_fd, bootstrap_fd):
    """Fixed fresh interpreter; caller retains its borrowed root/bootstrap descriptors.

    Supply a readonly nonblocking FIFO with bounded canonical bootstrap and EOF.
    The child constructs its controller after exec, never inherits a live instance.
    This spawns a process only: it cannot mint authority or establish host isolation.
    """
    owned = []
    process = None
    failed = False
    try:
        need(os.name == "posix" and threading.current_thread() is threading.main_thread())
        _root(root_fd)
        _pipe(bootstrap_fd, os.O_RDONLY)
        need(root_fd != bootstrap_fd)
        for _ in range(2):
            pair = os.pipe()
            owned.extend(pair)
            for fd in pair:
                os.set_blocking(fd, False)
                os.set_inheritable(fd, False)
        control_r, control_w, result_r, result_w = owned
        passed = (root_fd, bootstrap_fd, control_r, result_w)
        need(len(set(passed)) == 4)
        process = subprocess.Popen(
            [sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()),
             *(str(fd) for fd in passed)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            close_fds=True, pass_fds=passed, start_new_session=True,
            env={"PATH": os.defpath, "LC_ALL": "C"},
        )
        # Remove before close: even a failed close is never retried by fd number.
        for fd in (control_r, result_w):
            owned.remove(fd)
            need(_close(fd), "process_io_rejected")
        result = OwnedControllerProcess(process, control_w, result_r)
        owned.clear()
        return result
    except BaseException:
        failed = True
    finally:
        for fd in owned:
            _close(fd)
        # On post-spawn failure EOF requests cleanup. Do not kill/restart this child.
        # The caller must externally reconcile if the handle could not be returned.
    if failed:
        raise ProcessError("process_spawn_rejected") from None


def main(arguments=None):
    """Fixed inherited-descriptor child entrypoint; never takes raw configuration argv."""
    arguments = sys.argv[1:] if arguments is None else arguments
    fds = []
    controller = reader = payload = signals = None
    report = _failure_report("bootstrap_rejected")
    exit_code = 1
    try:
        need(type(arguments) is list and len(arguments) == 4)
        need(all(type(item) is str and item.isascii() and item.isdecimal()
                 and str(int(item)) == item and len(item) <= 7 for item in arguments))
        supplied = [int(item) for item in arguments]
        need(len(set(supplied)) == 4)
        root_fd, bootstrap_fd, control_fd, result_fd = supplied
        _root(root_fd)
        _pipe(bootstrap_fd, os.O_RDONLY)
        _pipe(control_fd, os.O_RDONLY)
        _pipe(result_fd, os.O_WRONLY)
        # Take ownership only after every descriptor has the expected role.
        fds = supplied
        for fd in fds:
            os.set_inheritable(fd, False)
        reader = ControlReader(control_fd)
        signals = SignalStop(reader)
        signals.__enter__()
        payload = read_bootstrap(bootstrap_fd)
        controller = provider.ProviderController(root_fd, **payload)
        payload = None
        report = service.run_controller(controller, reader)
        exit_code = 0
    except BaseException:
        # Never serialize exception, bootstrap, provider body, ID or interpreter text.
        report = _failure_report("process_failure" if controller is not None else
                                 "bootstrap_rejected")
    finally:
        payload = None
        if controller is not None:
            try:
                controller.close()
            except BaseException:
                exit_code = 1
        if reader is not None and not reader.close():
            exit_code = 1
        if exit_code != 0:
            report = _failure_report("process_failure")
        if len(fds) == 4:
            if not write_report(fds[3], report):
                exit_code = 1
        for fd in dict.fromkeys(fds):
            if not _close(fd):
                exit_code = 1
        if signals is not None:
            try:
                signals.__exit__(None, None, None)
            except BaseException:
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
