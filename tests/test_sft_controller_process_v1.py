"""Owned-process tests with local pipes and explicitly synthetic provider replies.

The child fixture uses the actual provider controller and service loop, but denies
every DNS, socket and TLS constructor before construction and replaces exchange
with pinned synthetic responses. It never reaches a real account or credential.
Process independence and signal tests are local observations, not cloud survival.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import hmac
import importlib
import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_test_sft_controller_process_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
process = importlib.import_module(PACKAGE + ".posttraining_sft_controller_process_v1")
service = importlib.import_module(PACKAGE + ".posttraining_sft_controller_service_v1")
provider = importlib.import_module(PACKAGE + ".posttraining_sft_provider_controller_v1")
net = importlib.import_module(PACKAGE + ".posttraining_sft_lambda_https_v1")
NS = 1_000_000_000
OWNED, PROTECTED, ATTEMPT = "a" * 32, "b" * 32, "c" * 64
TOKEN = b"SYNTHETIC-NOT-A-CREDENTIAL"
KEY = b"SYNTHETIC-KEY-NOT-FOR-REAL-USE!!"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def sha(value):
    return hashlib.sha256(value).hexdigest()


def packet():
    launch = canonical({
        "region_name": "us-east-1", "instance_type_name": "gpu_8x_a100_80gb_sxm4",
        "ssh_key_names": ["Synthetic Key"],
        "image": {"id": "12345678-1234-1234-1234-123456789abc"},
    })
    plan = canonical({
        "schema_version": provider.durable.model.PLAN_SCHEMA,
        "mode": "local_decision_model_only", "attempt_id": ATTEMPT,
        "account_scope_sha256": "d" * 64, "approval_sha256": "e" * 64,
        "launch_request_sha256": sha(launch), "protected_instance_ids": [PROTECTED],
        "region": "us-east-1", "instance_type": "gpu_8x_a100_80gb_sxm4", "gpu_count": 8,
        "total_host_hourly_rate_microusd": 22_320_000,
        "maximum_cost_microusd": 150_000_000, "noncompute_reserve_microusd": 10_000_000,
        "maximum_lifecycle_seconds": 121, "shutdown_reserve_seconds": 60,
        "create_not_before_unix": 999, "create_expires_unix": 2000, "maximum_requests": 8,
    })
    unsigned = {
        "schema_version": provider.authority.SCHEMA,
        "mode": "externally_issued_single_attempt_dispatch",
        "provider_host": "cloud.lambda.ai", "plan_sha256": sha(plan),
        "account_scope_sha256": "d" * 64, "approval_sha256": "e" * 64,
        "launch_request_sha256": sha(launch), "credential_sha256": sha(TOKEN),
        "create_not_before_unix": 999, "create_expires_unix": 2000, "maximum_requests": 8,
        "cleanup_scope": "only_fresh_create_bound_id_within_lifecycle_caps",
    }
    authority = canonical({**unsigned, "mac_sha256": hmac.new(
        KEY, provider.authority.DOMAIN + canonical(unsigned), hashlib.sha256,
    ).hexdigest()})
    return {
        "schema_version": "lumi.sft-controller-bootstrap.v0.1",
        "plan_ascii": plan.decode("ascii"), "plan_sha256": sha(plan),
        "authority_ascii": authority.decode("ascii"), "authority_sha256": sha(authority),
        "authority_key_hex": KEY.hex(), "authority_key_sha256": sha(KEY),
        "launch_ascii": launch.decode("ascii"), "token_ascii": TOKEN.decode("ascii"),
    }


def report(**changes):
    return {
        "schema_version": "lumi.sft-controller-service-report.v0.1",
        "status": "reconciliation_required", "stop_code": "parent_eof",
        "modeled_terminal": False, "reconciliation_required": True,
        "local_controller_closed": True, "provider_requests": 0, "control_polls": 1,
        "service_actions": 0, "journal_records": 2, "execution_authorized": False,
        "provider_termination_verified": False, "cloud_billing_stopped": False,
        "independent_survival_verified": False, "production_eligible": False, **changes,
    }


def nonblocking_pipe():
    reader, writer = os.pipe()
    os.set_blocking(reader, False)
    os.set_blocking(writer, False)
    return reader, writer


def close_if_open(fd):
    try:
        os.close(fd)
    except OSError:
        pass


def close_same_pipe(fd, identity, access):
    """Synthetic fixture cleanup must not close a later reused descriptor number."""
    try:
        info = os.fstat(fd)
        if (info.st_dev, info.st_ino) == identity and (
            fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE
        ) == access:
            os.close(fd)
    except OSError:
        pass


def network_denied(stack):
    for target, name in ((net.socket, "getaddrinfo"), (net.socket, "socket"),
                         (net.ssl, "SSLContext"), (net.ssl, "create_default_context")):
        stack.enter_context(patch.object(target, name, side_effect=AssertionError(
            "synthetic process fixture forbids network and certificate constructors",
        )))


def synthetic_controller_child(root_fd, control_fd, result_fd, ready_fd):
    """Actual child-owned controller/service; only the transport exchange is fake."""
    with contextlib.ExitStack() as stack:
        network_denied(stack)
        values = packet()
        offset = [0]
        ready = [False]
        terminated = [False]
        calls = []

        def clock():
            return time.monotonic_ns() + offset[0]

        def exchange(_transport, request, *, deadline_ns):
            if clock() >= deadline_ns:
                raise AssertionError("synthetic deadline violated")
            raw = request.claim()
            calls.append(request.operation)
            if request.operation == "create":
                if calls != ["create"]:
                    raise AssertionError("synthetic create repeated")
                value = {"data": {"instance_ids": [OWNED]}}
            else:
                if PROTECTED.encode("ascii") in raw:
                    raise AssertionError("protected target used")
                if request.operation == "terminate":
                    body = raw.split(b"\r\n\r\n", 1)[1]
                    if body != canonical({"instance_ids": [OWNED]}) or terminated[0]:
                        raise AssertionError("synthetic termination scope violated")
                    terminated[0] = True
                    value = {"data": {"terminated_instances": [instance("terminating")]}}
                elif request.operation == "observe":
                    if not raw.startswith(b"GET /api/v1/instances/" + OWNED.encode("ascii")):
                        raise AssertionError("synthetic observation scope violated")
                    value = {"data": instance("terminated" if terminated[0] else "active")}
                    if not terminated[0] and not ready[0]:
                        os.write(ready_fd, b"A")
                        ready[0] = True
                else:
                    raise AssertionError("unknown synthetic operation")
            return net.Reply(200, canonical(value))

        stack.enter_context(patch.object(net.LambdaHttps, "exchange", exchange))
        reader = process.ControlReader(control_fd)
        stack.callback(reader.close)
        close_if_open(control_fd)

        class Control:
            def poll(self, timeout):
                result = reader.poll(timeout)
                if result != "continue":
                    offset[0] += 16 * NS if offset[0] == 0 else 0
                return result

        controller = provider.ProviderController(
            root_fd, values["plan_ascii"].encode("ascii"), values["plan_sha256"],
            authority_bytes=values["authority_ascii"].encode("ascii"),
            authority_sha256=values["authority_sha256"], authority_key=KEY,
            authority_key_sha256=sha(KEY), launch_body=values["launch_ascii"].encode("ascii"),
            token=TOKEN, clock_ns=clock, unix_ns=lambda: 1000 * NS,
        )
        with process.SignalStop(reader):
            result = service.run_controller(controller, Control(), clock_ns=clock)
        if ready[0] and calls != ["create", "observe", "terminate", "observe"]:
            return 21
        if not process.write_report(result_fd, result):
            return 22
        return 0


def instance(status):
    return {
        "id": OWNED, "status": status, "region": {"name": "us-east-1"},
        "instance_type": {"name": "gpu_8x_a100_80gb_sxm4", "price_cents_per_hour": 2232,
                          "specs": {"gpus": 8}},
        "jupyter_token": "SYNTHETIC-SECRET-NEVER-REPORT",
    }


def synthetic_report_child(mode, control_fd, result_fd):
    raw = canonical(report()) + b"\n"
    if mode == "truncated":
        raw = raw[:-2]
    elif mode == "unknown":
        raw = canonical(report(unexpected="SYNTHETIC-SECRET")) + b"\n"
    elif mode == "oversize":
        raw = b"X" * 4097
    elif mode == "noncanonical":
        raw = json.dumps(report()).encode("ascii") + b"\n"
    elif mode == "empty":
        raw = b""
    elif mode == "duplicate":
        raw = raw + raw
    elif mode == "wait_for_stop":
        select.select([control_fd], [], [], 2)
    if raw:
        os.write(result_fd, raw)
    os.close(result_fd)
    if mode == "eof_before_exit":
        time.sleep(0.3)
    return 3 if mode == "nonzero" else 0


class PipeFixture(unittest.TestCase):
    def setUp(self):
        self.stack = self.enterContext(contextlib.ExitStack())
        network_denied(self.stack)
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory(
            prefix="lumi-controller-process-synthetic-",
        ))).resolve()
        self.directory.chmod(0o700)
        self.root_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        self.addCleanup(close_if_open, self.root_fd)

    def pipe(self):
        reader, writer = nonblocking_pipe()
        for fd, access in ((reader, os.O_RDONLY), (writer, os.O_WRONLY)):
            info = os.fstat(fd)
            self.addCleanup(close_same_pipe, fd, (info.st_dev, info.st_ino), access)
        return reader, writer

    def cleanup_owned(self, owned):
        def release():
            owned.disconnect()
            if owned._result is not None:
                fd, owned._result = owned._result, None
                os.close(fd)

        self.addCleanup(release)

    def assert_private(self, value):
        raw = canonical(value)
        for text in (TOKEN, KEY, OWNED.encode(), PROTECTED.encode(), ATTEMPT.encode(),
                     b"SYNTHETIC-SECRET", str(self.directory).encode(), b"Traceback"):
            self.assertNotIn(text, raw)
        for key in ("execution_authorized", "provider_termination_verified",
                    "cloud_billing_stopped", "independent_survival_verified",
                    "production_eligible"):
            self.assertIs(value[key], False)

    def assert_rejected(self, function):
        with self.assertRaises(ValueError) as caught:
            function()
        self.assertNotIn("SYNTHETIC", str(caught.exception))
        self.assertNotIn(str(self.directory), str(caught.exception))

    def child(self, mode="valid", *, controller=False):
        control_read, control_write = self.pipe()
        result_read, result_write = self.pipe()
        descriptors = [control_read, result_write]
        if controller:
            ready_read, ready_write = self.pipe()
            descriptors.extend([self.root_fd, ready_write])
            args = ["--synthetic-controller-child", str(self.root_fd), str(control_read),
                    str(result_write), str(ready_write)]
        else:
            args = ["--synthetic-report-child", mode, str(control_read), str(result_write)]
        child = subprocess.Popen(
            [sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()), *args],
            pass_fds=tuple(descriptors), close_fds=True, start_new_session=True,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
        os.close(control_read)
        os.close(result_write)
        if controller:
            os.close(ready_write)
        owned = process.OwnedControllerProcess(child, control_write, result_read)
        self.cleanup_owned(owned)

        def reap():
            owned.disconnect()
            child.wait(timeout=5)

        self.addCleanup(reap)
        if controller:
            self.assertTrue(select.select([ready_read], [], [], 5)[0],
                            "synthetic active child readiness missing")
            self.assertEqual(os.read(ready_read, 1), b"A")
        return owned, child


class ControlReaderTests(PipeFixture):
    def test_latched_stop_honors_wait_without_rereading_ready_eof(self):
        read_fd, write_fd = self.pipe()
        reader = process.ControlReader(read_fd)
        self.addCleanup(reader.close)
        os.close(write_fd)
        self.assertEqual(reader.poll(0), "eof")
        with patch.object(process.select, "select", return_value=([], [], [])) as wait, \
                patch.object(process.os, "read", side_effect=AssertionError("no reread")):
            self.assertEqual(reader.poll(0.25), "eof")
        wait.assert_called_once_with([], [], [], 0.25)

    def test_cloexec_configuration_failure_closes_duplicate_once(self):
        read_fd, _ = self.pipe()
        real_dup = os.dup
        duplicates = []

        def duplicate(fd):
            result = real_dup(fd)
            duplicates.append(result)
            return result

        with patch.object(process.os, "dup", side_effect=duplicate), \
                patch.object(process.os, "set_inheritable", side_effect=OSError("SYNTHETIC")):
            self.assert_rejected(lambda: process.ControlReader(read_fd))
        self.assertEqual(len(duplicates), 1)
        with self.assertRaises(OSError):
            os.fstat(duplicates[0])
        os.fstat(read_fd)

    def test_borrowed_reader_is_duplicated_and_only_duplicate_closed(self):
        read_fd, write_fd = self.pipe()
        reader = process.ControlReader(read_fd)
        self.addCleanup(reader.close)
        self.assertEqual(reader.poll(0), "continue")
        reader.close()
        os.fstat(read_fd)
        os.write(write_fd, b"S")
        self.assertEqual(os.read(read_fd, 1), b"S")

    def test_stop_is_latched(self):
        read_fd, write_fd = self.pipe()
        reader = process.ControlReader(read_fd)
        self.addCleanup(reader.close)
        os.write(write_fd, b"S")
        self.assertEqual(reader.poll(0), "stop")
        self.assertEqual(reader.poll(0), "stop")

    def test_parent_eof_is_latched(self):
        read_fd, write_fd = self.pipe()
        reader = process.ControlReader(read_fd)
        self.addCleanup(reader.close)
        os.close(write_fd)
        self.assertEqual(reader.poll(0), "eof")
        self.assertEqual(reader.poll(0), "eof")

    def test_unknown_control_byte_is_fault(self):
        read_fd, write_fd = self.pipe()
        reader = process.ControlReader(read_fd)
        self.addCleanup(reader.close)
        os.write(write_fd, b"X")
        self.assertEqual(reader.poll(0), "fault")
        self.assertEqual(reader.poll(0), "fault")

    def test_multiple_control_bytes_are_not_a_command_rpc(self):
        for raw in (b"SS", b"S create", b"\n", b"\x00"):
            with self.subTest(raw=raw):
                read_fd, write_fd = self.pipe()
                reader = process.ControlReader(read_fd)
                self.addCleanup(reader.close)
                os.write(write_fd, raw)
                self.assertEqual(reader.poll(0), "fault")

    def test_invalid_reader_types_blocking_writer_and_directory_rejected(self):
        read_fd, write_fd = self.pipe()
        for descriptor in (True, -1, "3", write_fd, self.root_fd):
            with self.subTest(descriptor=descriptor):
                self.assert_rejected(lambda descriptor=descriptor: process.ControlReader(
                    descriptor))
        os.set_blocking(read_fd, True)
        self.assert_rejected(lambda: process.ControlReader(read_fd))

    def test_invalid_poll_timeout_fails_closed(self):
        for timeout in (-1, 0.251, True, "0", float("nan"), float("inf")):
            with self.subTest(timeout=timeout):
                read_fd, _ = self.pipe()
                reader = process.ControlReader(read_fd)
                self.addCleanup(reader.close)
                self.assertEqual(reader.poll(timeout), "fault")

    def test_closed_reader_never_continues(self):
        read_fd, _ = self.pipe()
        reader = process.ControlReader(read_fd)
        reader.close()
        self.assertEqual(reader.poll(0), "fault")

    def test_foreign_thread_cannot_poll_owned_reader(self):
        read_fd, _ = self.pipe()
        reader = process.ControlReader(read_fd)
        self.addCleanup(reader.close)
        values = []
        thread = threading.Thread(target=lambda: values.append(reader.poll(0)))
        thread.start()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(values, ["fault"])


class SignalStopTests(PipeFixture):
    def test_only_term_int_hup_handlers_latch_and_original_handlers_restore(self):
        read_fd, _ = self.pipe()
        reader = process.ControlReader(read_fd)
        self.addCleanup(reader.close)
        signals = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
        before = {number: signal.getsignal(number) for number in signals}
        with process.SignalStop(reader):
            with patch.object(os, "write", side_effect=AssertionError("no handler I/O")):
                for number in signals:
                    signal.getsignal(number)(number, None)
                    signal.getsignal(number)(number, None)
            self.assertEqual(reader.poll(0), "signal")
        self.assertEqual({number: signal.getsignal(number) for number in signals}, before)

    def test_handlers_restore_after_exception(self):
        read_fd, _ = self.pipe()
        reader = process.ControlReader(read_fd)
        self.addCleanup(reader.close)
        before = signal.getsignal(signal.SIGTERM)
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with process.SignalStop(reader):
                raise RuntimeError("synthetic")
        self.assertIs(signal.getsignal(signal.SIGTERM), before)


class SpawnTests(PipeFixture):
    def test_exact_production_entrypoint_invalid_bootstrap_never_starts_controller(self):
        bootstrap_read, bootstrap_write = self.pipe()
        os.write(bootstrap_write, b"{}")
        os.close(bootstrap_write)
        # This malformed packet is rejected before any ProviderController or transport.
        owned = process.spawn_controller(self.root_fd, bootstrap_read)
        self.cleanup_owned(owned)
        self.addCleanup(lambda: owned._process.wait(timeout=5))
        self.assert_rejected(lambda: owned.collect(5))
        self.assertEqual(owned._process.wait(timeout=1), 1)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_spawn_preexec_failure_closes_every_new_endpoint_once(self):
        bootstrap_read, _ = self.pipe()
        real_pipe = os.pipe
        descriptors = []

        def make_pipe():
            result = real_pipe()
            descriptors.extend(result)
            return result

        with patch.object(process.os, "pipe", side_effect=make_pipe), \
                patch.object(process.subprocess, "Popen", side_effect=OSError("SYNTHETIC")), \
                patch.object(process, "_close", wraps=process._close) as close:
            self.assert_rejected(lambda: process.spawn_controller(self.root_fd, bootstrap_read))
        self.assertEqual(len(descriptors), 4)
        self.assertCountEqual([call.args[0] for call in close.call_args_list], descriptors)
        for fd in descriptors:
            with self.assertRaises(OSError):
                os.fstat(fd)

    def test_spawn_has_fixed_argv_sanitized_environment_and_descriptor_allowlist(self):
        bootstrap_read, _ = self.pipe()
        fake = Mock(spec=subprocess.Popen)
        with patch.object(process.subprocess, "Popen", return_value=fake) as popen:
            owned = process.spawn_controller(self.root_fd, bootstrap_read)
        self.cleanup_owned(owned)
        args, kwargs = popen.call_args
        command = args[0]
        self.assertEqual(command[:4], [sys.executable, "-I", "-S", "-B"])
        self.assertEqual(Path(command[4]).resolve(), Path(process.__file__).resolve())
        self.assertEqual(len(command), 9)
        self.assertTrue(all(argument.isdecimal() for argument in command[5:]))
        self.assertEqual(set(kwargs["pass_fds"]), set(map(int, command[5:])))
        self.assertEqual(len(kwargs["pass_fds"]), 4)
        self.assertTrue(kwargs["close_fds"])
        self.assertTrue(kwargs["start_new_session"])
        for stream in ("stdin", "stdout", "stderr"):
            self.assertEqual(kwargs[stream], subprocess.DEVNULL)
        for key in ("AWS_ACCESS_KEY_ID", "LAMBDA_API_KEY", "SSLKEYLOGFILE", "PYTHONPATH",
                    "PYTHONSTARTUP", "HTTP_PROXY", "HTTPS_PROXY", "LD_PRELOAD"):
            self.assertNotIn(key, kwargs["env"])
        self.assertNotIn(TOKEN.decode("ascii"), repr(popen.call_args))
        os.fstat(self.root_fd)
        os.fstat(bootstrap_read)

    def test_spawn_rejects_insecure_directory_before_child_or_pipes(self):
        bootstrap_read, _ = self.pipe()
        self.directory.chmod(0o755)
        with patch.object(process.subprocess, "Popen") as popen:
            self.assert_rejected(lambda: process.spawn_controller(self.root_fd, bootstrap_read))
        popen.assert_not_called()

    def test_spawn_rejects_bootstrap_wrong_descriptor_and_blocking_pipe(self):
        bootstrap_read, bootstrap_write = self.pipe()
        with patch.object(process.subprocess, "Popen") as popen:
            for descriptor in (True, -1, bootstrap_write, self.root_fd):
                self.assert_rejected(lambda descriptor=descriptor: process.spawn_controller(
                    self.root_fd, descriptor))
            os.set_blocking(bootstrap_read, True)
            self.assert_rejected(lambda: process.spawn_controller(self.root_fd, bootstrap_read))
        popen.assert_not_called()

    def test_spawn_error_never_retries_or_exposes_exception(self):
        bootstrap_read, _ = self.pipe()
        with patch.object(process.subprocess, "Popen", side_effect=OSError(
            "SYNTHETIC-PRIVATE-ERROR",
        )) as popen:
            self.assert_rejected(lambda: process.spawn_controller(self.root_fd, bootstrap_read))
        popen.assert_called_once()


class OwnedProcessTests(PipeFixture):
    def test_failed_control_close_is_not_retried_by_descriptor_number(self):
        _, writer = self.pipe()
        result, _ = self.pipe()
        owned = process.OwnedControllerProcess(Mock(), writer, result)
        original = process._close

        def close(fd):
            original(fd)
            return False

        with patch.object(process, "_close", side_effect=close) as closed:
            self.assertFalse(owned.request_stop())
            self.assertTrue(owned.request_stop())
            self.assertTrue(owned.disconnect())
        closed.assert_called_once_with(writer)

    def test_result_capacity_and_read_call_budget_are_not_reset_on_collection(self):
        _, writer = self.pipe()
        result, _ = self.pipe()
        owned = process.OwnedControllerProcess(Mock(), writer, result)
        self.cleanup_owned(owned)
        with patch.object(process.select, "select", return_value=([result], [], [])), \
                patch.object(process.os, "read", return_value=b" ") as read:
            self.assert_rejected(lambda: owned.collect(1))
        self.assertEqual(read.call_count, 128)
        self.assert_rejected(lambda: owned.collect(1))

    def test_result_close_failure_after_eof_withholds_success(self):
        _, writer = self.pipe()
        result, result_writer = self.pipe()
        owned = process.OwnedControllerProcess(Mock(), writer, result)
        self.cleanup_owned(owned)
        os.write(result_writer, canonical(report()) + b"\n")
        os.close(result_writer)
        real_close = process._close

        def close(fd):
            real_close(fd)
            return fd != result

        with patch.object(process, "_close", side_effect=close):
            self.assert_rejected(lambda: owned.collect(1))
        owned._process.wait.assert_not_called()

    def test_stop_writes_exactly_once_and_closes_to_eof(self):
        read_fd, write_fd = self.pipe()
        result_read, _ = self.pipe()
        owned = process.OwnedControllerProcess(Mock(), write_fd, result_read)
        self.addCleanup(owned.disconnect)
        owned.request_stop()
        owned.request_stop()
        owned.disconnect()
        self.assertEqual(os.read(read_fd, 8), b"S")
        self.assertEqual(os.read(read_fd, 8), b"")

    def test_disconnect_is_eof_without_stop_message(self):
        read_fd, write_fd = self.pipe()
        result_read, _ = self.pipe()
        owned = process.OwnedControllerProcess(Mock(), write_fd, result_read)
        owned.disconnect()
        owned.disconnect()
        self.assertEqual(os.read(read_fd, 8), b"")

    def test_stop_short_write_is_never_retried(self):
        _, write_fd = self.pipe()
        result_read, _ = self.pipe()
        owned = process.OwnedControllerProcess(Mock(), write_fd, result_read)
        with patch.object(process.os, "write", return_value=0) as write:
            self.assertFalse(owned.request_stop())
            owned.request_stop()
        write.assert_called_once()

    def test_report_requires_canonical_complete_eof_and_zero_exit(self):
        owned, child = self.child()
        value = owned.collect(3)
        self.assertEqual(value, report())
        self.assertEqual(child.returncode, 0)
        self.assert_private(value)

    def test_invalid_result_reports_and_nonzero_exit_never_pass(self):
        for mode in ("truncated", "unknown", "oversize", "noncanonical", "empty",
                     "duplicate", "nonzero"):
            with self.subTest(mode=mode):
                owned, _ = self.child(mode)
                self.assert_rejected(lambda owned=owned: owned.collect(3))

    def test_pipe_eof_is_not_child_exit_proof(self):
        owned, child = self.child("eof_before_exit")
        self.assert_rejected(lambda: owned.collect(0.05))
        self.assertIsNone(child.poll())
        self.assertEqual(owned.collect(3), report())

    def test_collect_timeout_never_kills_and_can_collect_after_stop(self):
        owned, child = self.child("wait_for_stop")
        self.assert_rejected(lambda: owned.collect(0.01))
        self.assertIsNone(child.poll())
        owned.request_stop()
        self.assertEqual(owned.collect(3), report())

    def test_invalid_collection_timeouts_never_wait_or_read(self):
        _, write_fd = self.pipe()
        result_read, _ = self.pipe()
        child = Mock()
        owned = process.OwnedControllerProcess(child, write_fd, result_read)
        self.addCleanup(owned.disconnect)
        for value in (0, -1, 31, True, "3", float("nan"), float("inf")):
            with self.subTest(value=value):
                self.assert_rejected(lambda value=value: owned.collect(value))
        child.wait.assert_not_called()

    def test_no_kill_terminate_restart_or_provider_operation_api(self):
        _, write_fd = self.pipe()
        result_read, _ = self.pipe()
        owned = process.OwnedControllerProcess(Mock(), write_fd, result_read)
        self.addCleanup(owned.disconnect)
        for name in ("kill", "terminate", "restart", "retry", "create", "observe", "launch"):
            self.assertFalse(hasattr(owned, name), name)


class IndependentChildTests(PipeFixture):
    def assert_terminal(self, owned, child, stop_code):
        value = owned.collect(5)
        self.assertEqual(child.returncode, 0)
        self.assertEqual(value["status"], "terminal_modeled")
        self.assertEqual(value["stop_code"], stop_code)
        self.assertEqual(value["provider_requests"], 4)
        self.assertTrue(value["modeled_terminal"])
        self.assertTrue(value["local_controller_closed"])
        self.assert_private(value)

    def test_child_new_session_and_parent_eof_drive_targeted_cleanup(self):
        owned, child = self.child(controller=True)
        self.assertEqual(os.getsid(child.pid), child.pid)
        self.assertNotEqual(os.getsid(child.pid), os.getsid(0))
        owned.disconnect()
        self.assert_terminal(owned, child, "parent_eof")

    def test_parent_stop_is_single_byte_and_child_reaps_after_cleanup(self):
        owned, child = self.child(controller=True)
        owned.request_stop()
        owned.request_stop()
        self.assert_terminal(owned, child, "stop_requested")

    def test_repeated_sigterm_is_latched_while_child_cleans_up(self):
        owned, child = self.child(controller=True)
        os.kill(child.pid, signal.SIGTERM)
        os.kill(child.pid, signal.SIGTERM)
        self.assert_terminal(owned, child, "signal_requested")

    def test_sighup_is_cleanup_request_not_immediate_death(self):
        owned, child = self.child(controller=True)
        os.kill(child.pid, signal.SIGHUP)
        self.assert_terminal(owned, child, "signal_requested")


class ResultWriterTests(PipeFixture):
    def test_bootstrap_failure_counts_are_unknown_not_false_zero(self):
        value = process._failure_report("bootstrap_rejected")
        self.assertEqual(service.validate_report(value), value)
        for name in service.REPORT_COUNTS:
            self.assertIsNone(value[name])
        self.assertTrue(value["reconciliation_required"])
        bad = {**value, "provider_requests": 0}
        self.assert_rejected(lambda: service.validate_report(bad))

    def test_exact_canonical_newline_report_and_no_secret_fields(self):
        read_fd, write_fd = self.pipe()
        value = report()
        self.assertTrue(process.write_report(write_fd, value))
        self.assertEqual(os.read(read_fd, 4096), canonical(value) + b"\n")
        self.assert_private(value)

    def test_invalid_or_extended_report_does_not_write(self):
        _, write_fd = self.pipe()
        for value in (None, {}, report(extra="SYNTHETIC-SECRET"),
                      report(production_eligible=True), report(provider_requests=True)):
            with self.subTest(value=value):
                with patch.object(process.os, "write") as write:
                    self.assertFalse(process.write_report(write_fd, value))
                write.assert_not_called()

    def test_short_or_failed_write_is_not_retried(self):
        _, write_fd = self.pipe()
        for result in (0, 1):
            with patch.object(process.os, "write", return_value=result) as write:
                self.assertFalse(process.write_report(write_fd, report()))
            write.assert_called_once()
        with patch.object(process.os, "write", side_effect=BlockingIOError(
            "SYNTHETIC-SECRET",
        )) as write:
            self.assertFalse(process.write_report(write_fd, report()))
        write.assert_called_once()


class BootstrapTests(PipeFixture):
    def test_signal_handlers_remain_latched_during_final_report_and_fd_cleanup(self):
        bootstrap_r, bootstrap_w = nonblocking_pipe()
        control_r, control_w = nonblocking_pipe()
        result_r, result_w = nonblocking_pipe()
        root = os.dup(self.root_fd)
        expected = report(stop_code="parent_eof")
        observed = []
        fake_controller = Mock()
        real_write = process.write_report

        def writer(fd, value):
            for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                handler = signal.getsignal(number)
                self.assertTrue(callable(handler))
                handler(number, None)
            observed.append(True)
            return real_write(fd, value)

        try:
            with patch.object(process, "read_bootstrap", return_value={}), \
                    patch.object(provider, "ProviderController", return_value=fake_controller), \
                    patch.object(service, "run_controller", return_value=expected), \
                    patch.object(process, "write_report", side_effect=writer):
                code = process.main([str(fd) for fd in (root, bootstrap_r, control_r, result_w)])
            self.assertEqual(code, 0)
            root = bootstrap_r = control_r = result_w = None
            self.assertEqual(observed, [True])
            self.assertEqual(json.loads(os.read(result_r, 4096)), expected)
        finally:
            for fd in (root, bootstrap_r, bootstrap_w, control_r, control_w, result_r, result_w):
                if fd is not None:
                    close_if_open(fd)

    def test_exact_packet_decodes_only_fixed_constructor_arguments(self):
        value = packet()
        decoded = process.decode_bootstrap(canonical(value))
        self.assertEqual(decoded["token"], TOKEN)
        self.assertEqual(decoded["authority_key"], KEY)
        self.assertEqual(set(decoded), {"plan_bytes", "plan_sha256", "authority_bytes",
                                       "authority_sha256", "authority_key",
                                       "authority_key_sha256", "launch_body", "token"})
        self.assertNotIn("schema_version", decoded)

    def test_noncanonical_duplicate_nonfinite_and_oversized_packet_rejected(self):
        raw = canonical(packet())
        cases = (b"", b"x" * 32769, b"\xff", raw + b"\n", b" " + raw,
                 b'{"token_ascii":"SYNTHETIC",' + raw[1:], b"[" * 4000 + b"]" * 4000,
                 b'{"token_ascii":NaN}', b"null", b"[]", b"{}", bytearray(raw))
        for value in cases:
            with self.subTest(kind=type(value).__name__):
                self.assert_rejected(lambda value=value: process.decode_bootstrap(value))

    def test_unknown_missing_typed_and_raw_secret_field_rejected(self):
        original = packet()
        for field in original:
            self.assert_rejected(lambda field=field: process.decode_bootstrap(canonical({
                key: value for key, value in original.items() if key != field
            })))
        for changes in ({"token_ascii": True}, {"plan_sha256": "A" * 64},
                        {"authority_key_hex": "0" * 63}, {"plan_ascii": "x" * 16385},
                        {"authority_ascii": "x" * 8193}, {"launch_ascii": "x" * 4097},
                        {"token_ascii": "x" * 513}, {"token_ascii": "\u00e9"},
                        {"command": "SYNTHETIC"}):
            self.assert_rejected(lambda changes=changes: process.decode_bootstrap(canonical({
                **original, **changes,
            })))

    def test_read_bootstrap_requires_eof_and_preserves_borrowed_fd(self):
        reader, writer = self.pipe()
        raw = canonical(packet())
        self.assertEqual(os.write(writer, raw), len(raw))
        os.close(writer)
        self.assertEqual(process.read_bootstrap(reader), process.decode_bootstrap(raw))
        os.fstat(reader)

    def test_bootstrap_timeout_or_too_many_fragments_has_fixed_error(self):
        reader, _ = self.pipe()
        with patch.object(process.select, "select", return_value=([], [], [])):
            self.assert_rejected(lambda: process.read_bootstrap(reader))
        with patch.object(process.select, "select", return_value=([reader], [], [])), \
                patch.object(process.os, "read", return_value=b" ") as read:
            self.assert_rejected(lambda: process.read_bootstrap(reader))
        self.assertEqual(read.call_count, 128)

    def test_malformed_argv_does_not_close_unclaimed_fds(self):
        read_fd, write_fd = self.pipe()
        for args in ([], ["3"], ["1", "2", "3", "4"], ["03", "4", "5", "6"],
                     [str(self.root_fd), str(read_fd), str(read_fd), str(write_fd)]):
            self.assertEqual(process.main(args), 1)
            os.fstat(read_fd)
            os.fstat(write_fd)
            os.fstat(self.root_fd)

    def test_child_main_with_valid_synthetic_authority_and_precreate_eof_has_no_network(self):
        # main adopts only dedicated duplicates; no production provider operation occurs.
        bootstrap_r, bootstrap_w = nonblocking_pipe()
        control_r, control_w = nonblocking_pipe()
        result_r, result_w = nonblocking_pipe()
        root = os.dup(self.root_fd)
        try:
            raw = canonical(packet())
            os.write(bootstrap_w, raw)
            os.close(bootstrap_w)
            bootstrap_w = None
            os.close(control_w)
            control_w = None
            with patch.object(provider.time, "time_ns", return_value=1000 * NS):
                code = process.main([str(fd) for fd in (root, bootstrap_r, control_r, result_w)])
            self.assertEqual(code, 0)
            root = bootstrap_r = control_r = result_w = None
            value = json.loads(os.read(result_r, 4096))
            self.assertEqual(value["stop_code"], "parent_eof")
            self.assertEqual(value["provider_requests"], 0)
            self.assertTrue(value["local_controller_closed"])
            self.assert_private(value)
        finally:
            for fd in (bootstrap_r, bootstrap_w, control_r, control_w, result_r, result_w, root):
                if fd is not None:
                    close_if_open(fd)


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "--synthetic-controller-child":
        try:
            code = synthetic_controller_child(*(int(value) for value in sys.argv[2:]))
        except BaseException:
            code = 20
        raise SystemExit(code)
    if len(sys.argv) == 5 and sys.argv[1] == "--synthetic-report-child":
        if sys.argv[2] not in {"valid", "truncated", "unknown", "oversize", "noncanonical",
                               "empty", "duplicate", "nonzero", "wait_for_stop",
                               "eof_before_exit"}:
            raise SystemExit(20)
        raise SystemExit(synthetic_report_child(sys.argv[2], int(sys.argv[3]), int(sys.argv[4])))
    unittest.main()
