"""Synthetic bootstrap handoff tests; no credentials, provider or child execution.

The only real I/O is anonymous local pipes. DNS, sockets, TLS construction and
subprocess/provider entrypoints are denied throughout every test. A successful
handoff proves local framing and descriptor handling, never owner authorization.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import hmac
import importlib
import json
import os
import socket
import ssl
import subprocess
import sys
import threading
import types
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_test_sft_controller_bootstrap_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
bootstrap = importlib.import_module(PACKAGE + ".posttraining_sft_controller_bootstrap_v1")
process = importlib.import_module(PACKAGE + ".posttraining_sft_controller_process_v1")
authority = importlib.import_module(PACKAGE + ".posttraining_sft_dispatch_authority_v1")
http = importlib.import_module(PACKAGE + ".posttraining_sft_lambda_http_v1")
TOKEN = b"SYNTHETIC-BOOTSTRAP-NOT-A-CREDENTIAL"
KEY = b"SYNTHETIC-BOOTSTRAP-KEY".ljust(32, b"!")
ATTEMPT, PROTECTED = "c" * 64, "b" * 32
ERROR = "bootstrap_handoff_rejected"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def sha(value):
    return hashlib.sha256(value).hexdigest()


def synthetic_options(*, plan_changes=None, launch_changes=None, authority_changes=None):
    launch = canonical({
        "region_name": "us-east-1", "instance_type_name": "gpu_8x_a100_80gb_sxm4",
        "ssh_key_names": ["Synthetic Bootstrap Key"],
        "image": {"id": "12345678-1234-1234-1234-123456789abc"},
        **(launch_changes or {}),
    })
    plan = {
        "schema_version": authority.model.PLAN_SCHEMA, "mode": "local_decision_model_only",
        "attempt_id": ATTEMPT, "account_scope_sha256": "d" * 64,
        "approval_sha256": "e" * 64, "launch_request_sha256": sha(launch),
        "protected_instance_ids": [PROTECTED], "region": "us-east-1",
        "instance_type": "gpu_8x_a100_80gb_sxm4", "gpu_count": 8,
        "total_host_hourly_rate_microusd": 22_320_000,
        "maximum_cost_microusd": 150_000_000, "noncompute_reserve_microusd": 10_000_000,
        "maximum_lifecycle_seconds": 121, "shutdown_reserve_seconds": 60,
        "create_not_before_unix": 999, "create_expires_unix": 2000, "maximum_requests": 8,
        **(plan_changes or {}),
    }
    plan_bytes = canonical(plan)
    unsigned = {
        "schema_version": authority.SCHEMA,
        "mode": "externally_issued_single_attempt_dispatch", "provider_host": "cloud.lambda.ai",
        "plan_sha256": sha(plan_bytes), "account_scope_sha256": plan["account_scope_sha256"],
        "approval_sha256": plan["approval_sha256"],
        "launch_request_sha256": plan["launch_request_sha256"], "credential_sha256": sha(TOKEN),
        "create_not_before_unix": plan["create_not_before_unix"],
        "create_expires_unix": plan["create_expires_unix"],
        "maximum_requests": plan["maximum_requests"],
        "cleanup_scope": "only_fresh_create_bound_id_within_lifecycle_caps",
        **(authority_changes or {}),
    }
    authority_bytes = canonical({**unsigned, "mac_sha256": hmac.new(
        KEY, authority.DOMAIN + canonical(unsigned), hashlib.sha256,
    ).hexdigest()})
    return {
        "plan_bytes": plan_bytes, "plan_sha256": sha(plan_bytes),
        "authority_bytes": authority_bytes, "authority_sha256": sha(authority_bytes),
        "authority_key": KEY, "authority_key_sha256": sha(KEY),
        "launch_body": launch, "token": TOKEN,
    }


def identity(fd):
    info = os.fstat(fd)
    return info.st_dev, info.st_ino, fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE


def close_same_fd(fd, expected):
    """Test cleanup never closes a later reused descriptor number."""
    try:
        if identity(fd) == expected:
            os.close(fd)
    except OSError:
        pass


class Fixture(unittest.TestCase):
    def setUp(self):
        targets = [
            (socket, "getaddrinfo"), (socket, "socket"),
            (ssl, "SSLContext"), (ssl, "create_default_context"),
            (subprocess, "Popen"), (subprocess, "run"), (subprocess, "call"),
            (subprocess, "check_call"), (subprocess, "check_output"),
            (process, "spawn_controller"), (process.provider, "ProviderController"),
            (os, "system"),
        ]
        targets.extend((os, name) for name in ("fork", "posix_spawn", "posix_spawnp")
                       if hasattr(os, name))
        for target, name in targets:
            denied = self.enterContext(patch.object(target, name, side_effect=AssertionError(
                "synthetic bootstrap test forbids network, provider and subprocess execution",
            )))
            self.addCleanup(denied.assert_not_called)
        self.options = synthetic_options()

    def rejected(self, operation):
        with self.assertRaises(bootstrap.BootstrapError) as caught:
            operation()
        self.assertEqual(str(caught.exception), ERROR)
        for secret in (TOKEN.decode("ascii"), KEY.decode("ascii"), KEY.hex(), ATTEMPT):
            self.assertNotIn(secret, repr(caught.exception))
        return caught.exception

    def make(self, **changes):
        result = bootstrap.BootstrapPipe(**{**self.options, **changes})
        self.addCleanup(result.close)
        return result

    def tracked_pipe(self):
        allocated = []
        original = os.pipe

        def allocate():
            pair = original()
            allocated.extend(pair)
            for fd in pair:
                self.addCleanup(close_same_fd, fd, identity(fd))
            return pair

        return allocated, allocate

    def assert_closed(self, *fds):
        for fd in fds:
            with self.subTest(fd=fd):
                with self.assertRaises(OSError) as caught:
                    os.fstat(fd)
                self.assertEqual(caught.exception.errno, errno.EBADF)

    def read_packet(self, fd):
        blocks = []
        for _ in range(process.MAXIMUM_READS):
            block = os.read(fd, process.MAXIMUM_BOOTSTRAP_BYTES + 1)
            if not block:
                return b"".join(blocks)
            blocks.append(block)
        self.fail("prefilled pipe did not reach EOF within the existing read bound")


class EncodingTests(Fixture):
    def test_exact_existing_schema_and_canonical_bytes(self):
        raw = bootstrap.encode_bootstrap(**self.options)
        expected = {
            "schema_version": process.BOOTSTRAP_SCHEMA,
            "plan_ascii": self.options["plan_bytes"].decode("ascii"),
            "plan_sha256": self.options["plan_sha256"],
            "authority_ascii": self.options["authority_bytes"].decode("ascii"),
            "authority_sha256": self.options["authority_sha256"],
            "authority_key_hex": KEY.hex(), "authority_key_sha256": sha(KEY),
            "launch_ascii": self.options["launch_body"].decode("ascii"),
            "token_ascii": TOKEN.decode("ascii"),
        }
        self.assertIs(type(raw), bytes)
        self.assertEqual(raw, canonical(expected))
        self.assertEqual(process.decode_bootstrap(raw), self.options)
        self.assertLessEqual(len(raw), process.MAXIMUM_BOOTSTRAP_BYTES)
        self.assertFalse(raw.endswith(b"\n"))

    def test_encoding_is_pure_and_does_not_check_live_clock(self):
        with patch.object(os, "pipe") as pipe, patch.object(os, "write") as write:
            raw = bootstrap.encode_bootstrap(**self.options)
        pipe.assert_not_called()
        write.assert_not_called()
        self.assertEqual(process.decode_bootstrap(raw), self.options)

    def test_repeated_encoding_does_not_mutate_inputs(self):
        original = dict(self.options)
        self.assertEqual(bootstrap.encode_bootstrap(**self.options),
                         bootstrap.encode_bootstrap(**self.options))
        self.assertEqual(self.options, original)

    def test_hash_and_key_bindings_reject_before_allocating(self):
        for name, changed in (
            ("plan_sha256", "f" * 64), ("authority_sha256", "f" * 64),
            ("authority_key_sha256", "f" * 64), ("authority_key", b"F" * 32),
            ("token", b"SYNTHETIC-WRONG-TOKEN"),
        ):
            with self.subTest(field=name), patch.object(os, "pipe") as pipe:
                self.rejected(partial(bootstrap.encode_bootstrap,
                    **{**self.options, name: changed},
                ))
                self.rejected(partial(bootstrap.BootstrapPipe,
                    **{**self.options, name: changed},
                ))
                pipe.assert_not_called()

    def test_wrong_types_reject_before_allocating(self):
        cases = []
        for field in ("plan_bytes", "authority_bytes", "authority_key", "launch_body", "token"):
            cases.extend((field, value) for value in (None, True, "SYNTHETIC", bytearray(b"x")))
        for field in ("plan_sha256", "authority_sha256", "authority_key_sha256"):
            cases.extend((field, value) for value in (None, True, b"f" * 64, 1))
        for name, changed in cases:
            with self.subTest(field=name, value_type=type(changed).__name__):
                with patch.object(os, "pipe") as pipe:
                    self.rejected(partial(bootstrap.BootstrapPipe,
                        **{**self.options, name: changed},
                    ))
                    pipe.assert_not_called()

    def test_empty_oversized_nonascii_and_injection_values_reject(self):
        cases = (
            ("plan_bytes", b""), ("plan_bytes", b"X" * 16385),
            ("authority_bytes", b"X" * 8193), ("launch_body", b"X" * 4097),
            ("authority_key", KEY[:-1]), ("authority_key", KEY + b"!"),
            ("token", b""), ("token", b"X" * 513), ("token", b"X" * 8 + b"\xff"),
            ("token", b"SYNTHETIC\r\nInjected: value"), ("plan_sha256", "F" * 64),
        )
        for name, changed in cases:
            with self.subTest(field=name, size=len(changed)), patch.object(os, "pipe") as pipe:
                self.rejected(partial(bootstrap.BootstrapPipe,
                    **{**self.options, name: changed},
                ))
                pipe.assert_not_called()

    def test_valid_mac_with_mismatched_approval_or_account_rejects(self):
        for field in ("account_scope_sha256", "approval_sha256", "launch_request_sha256"):
            options = synthetic_options(authority_changes={field: "f" * 64})
            with self.subTest(field=field), patch.object(os, "pipe") as pipe:
                self.rejected(partial(bootstrap.BootstrapPipe, **options))
                pipe.assert_not_called()

    def test_valid_mac_with_wrong_window_or_request_bound_rejects(self):
        for field, value in (("create_not_before_unix", 998), ("create_expires_unix", 2001),
                             ("maximum_requests", 9)):
            options = synthetic_options(authority_changes={field: value})
            with self.subTest(field=field), patch.object(os, "pipe") as pipe:
                self.rejected(partial(bootstrap.BootstrapPipe, **options))
                pipe.assert_not_called()

    def test_authority_mac_tampering_rejects_even_with_correct_outer_digest(self):
        value = json.loads(self.options["authority_bytes"])
        value["mac_sha256"] = "0" * 64
        raw = canonical(value)
        with patch.object(os, "pipe") as pipe:
            self.rejected(lambda: bootstrap.BootstrapPipe(**{
                **self.options, "authority_bytes": raw, "authority_sha256": sha(raw),
            }))
            pipe.assert_not_called()

    def test_noncanonical_and_duplicate_authority_reject(self):
        original = self.options["authority_bytes"]
        changed = (
            json.dumps(json.loads(original)).encode("ascii"),
            b'{"schema_version":"duplicate",' + original[1:],
            original[:-1] + b',"unexpected":NaN}',
        )
        for raw in changed:
            with self.subTest(size=len(raw)), patch.object(os, "pipe") as pipe:
                self.rejected(partial(bootstrap.BootstrapPipe, **{
                    **self.options, "authority_bytes": raw, "authority_sha256": sha(raw),
                }))
                pipe.assert_not_called()

    def test_valid_authority_cannot_bypass_launch_validation(self):
        cases = (
            {"region_name": "us-west-1"}, {"instance_type_name": "other_type"},
            {"ssh_key_names": ["One", "Two"]}, {"ssh_key_names": [" Unsafe "]},
            {"image": {"id": "not-an-image"}}, {"user_data": "SYNTHETIC-FORBIDDEN"},
        )
        for changed in cases:
            options = synthetic_options(launch_changes=changed)
            with self.subTest(field=next(iter(changed))), patch.object(os, "pipe") as pipe:
                self.rejected(partial(bootstrap.BootstrapPipe, **options))
                pipe.assert_not_called()

    def test_invalid_bound_plan_rejects_before_pipe_allocation(self):
        for changed in ({"gpu_count": 9}, {"maximum_cost_microusd": 150_000_001},
                        {"protected_instance_ids": []}, {"maximum_requests": 65}):
            options = synthetic_options(plan_changes=changed)
            with self.subTest(field=next(iter(changed))), patch.object(os, "pipe") as pipe:
                self.rejected(partial(bootstrap.BootstrapPipe, **options))
                pipe.assert_not_called()


class PipeTests(Fixture):
    def test_real_pipe_roundtrip_and_immediate_eof(self):
        handoff = self.make()
        fd = handoff.fileno()
        self.assertEqual(process.read_bootstrap(fd), self.options)
        self.assertEqual(os.read(fd, 1), b"")
        self.assertTrue(handoff.close())
        self.assert_closed(fd)

    def test_read_end_nonblocking_readonly_and_cloexec(self):
        handoff = self.make()
        fd = handoff.fileno()
        self.assertFalse(os.get_blocking(fd))
        self.assertFalse(os.get_inheritable(fd))
        self.assertEqual(fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE, os.O_RDONLY)
        with self.assertRaises(OSError):
            os.write(fd, b"X")

    def test_exactly_one_write_with_nonblocking_cloexec_endpoints(self):
        allocated, allocate = self.tracked_pipe()
        original_write = os.write
        seen = []

        def write(fd, raw):
            seen.append((fd, raw))
            self.assertEqual(fd, allocated[1])
            for endpoint in allocated:
                self.assertFalse(os.get_blocking(endpoint))
                self.assertFalse(os.get_inheritable(endpoint))
            return original_write(fd, raw)

        with patch.object(os, "pipe", side_effect=allocate), patch.object(
            os, "write", side_effect=write,
        ) as writer:
            handoff = self.make()
        writer.assert_called_once()
        self.assertEqual(seen[0][1], bootstrap.encode_bootstrap(**self.options))
        self.assert_closed(allocated[1])
        self.assertEqual(self.read_packet(handoff.fileno()), seen[0][1])

    def test_distinct_pipe_endpoint_identities_are_valid_on_every_platform(self):
        descriptor = bootstrap._descriptor
        reader_identity = []

        def different_endpoints(fd, access):
            result = descriptor(fd, access)
            if access == os.O_RDONLY:
                reader_identity[:] = result
                return result
            return reader_identity[0], reader_identity[1] + 1

        with patch.object(bootstrap, "_descriptor", side_effect=different_endpoints):
            handoff = self.make()
        self.assertEqual(process.read_bootstrap(handoff.fileno()), self.options)

    def test_final_reader_validation_failure_closes_both_without_exposure(self):
        allocated, allocate = self.tracked_pipe()
        descriptor = bootstrap._descriptor
        reads = []

        def fail_second_read_check(fd, access):
            if access == os.O_RDONLY:
                reads.append(fd)
                if len(reads) == 2:
                    raise OSError("SYNTHETIC-PRIVATE-TEXT")
            return descriptor(fd, access)

        with patch.object(os, "pipe", side_effect=allocate), patch.object(
            bootstrap, "_descriptor", side_effect=fail_second_read_check,
        ):
            self.rejected(lambda: bootstrap.BootstrapPipe(**self.options))
        self.assertEqual(len(reads), 2)
        self.assert_closed(*allocated)

    def test_short_write_rejects_and_closes_both_ends_without_retry(self):
        allocated, allocate = self.tracked_pipe()
        original_write = os.write

        def short(fd, raw):
            return original_write(fd, raw[:7])

        with patch.object(os, "pipe", side_effect=allocate), patch.object(
            os, "write", side_effect=short,
        ) as writer:
            self.rejected(lambda: bootstrap.BootstrapPipe(**self.options))
        writer.assert_called_once()
        self.assert_closed(*allocated)

    def test_write_eagain_rejects_and_closes_without_retry(self):
        self.check_write_error(BlockingIOError(errno.EAGAIN, "SYNTHETIC-PRIVATE-TEXT"))

    def test_write_eintr_rejects_and_closes_without_retry(self):
        self.check_write_error(InterruptedError(errno.EINTR, "SYNTHETIC-PRIVATE-TEXT"))

    def test_write_epipe_rejects_and_closes_without_retry(self):
        self.check_write_error(BrokenPipeError(errno.EPIPE, "SYNTHETIC-PRIVATE-TEXT"))

    def check_write_error(self, failure):
        allocated, allocate = self.tracked_pipe()
        with patch.object(os, "pipe", side_effect=allocate), patch.object(
            os, "write", side_effect=failure,
        ) as writer:
            self.rejected(lambda: bootstrap.BootstrapPipe(**self.options))
        writer.assert_called_once()
        self.assert_closed(*allocated)

    def test_zero_or_invalid_write_counts_reject(self):
        for result in (0, -1, True, None, process.MAXIMUM_BOOTSTRAP_BYTES + 1):
            allocated, allocate = self.tracked_pipe()
            with self.subTest(result=result), patch.object(os, "pipe", side_effect=allocate):
                with patch.object(os, "write", return_value=result) as writer:
                    self.rejected(lambda: bootstrap.BootstrapPipe(**self.options))
                writer.assert_called_once()
                self.assert_closed(*allocated)

    def test_allocation_failure_is_redacted_without_write_or_close(self):
        with patch.object(os, "pipe", side_effect=OSError("SYNTHETIC-PRIVATE-TEXT")):
            with patch.object(os, "write") as writer, patch.object(os, "close") as closer:
                self.rejected(lambda: bootstrap.BootstrapPipe(**self.options))
                writer.assert_not_called()
                closer.assert_not_called()

    def test_nonblocking_configuration_failure_closes_both_without_write(self):
        allocated, allocate = self.tracked_pipe()
        with patch.object(os, "pipe", side_effect=allocate), patch.object(
            os, "set_blocking", side_effect=OSError("SYNTHETIC-PRIVATE-TEXT"),
        ), patch.object(os, "write") as writer:
            self.rejected(lambda: bootstrap.BootstrapPipe(**self.options))
        writer.assert_not_called()
        self.assert_closed(*allocated)

    def test_cloexec_configuration_failure_closes_both_without_write(self):
        allocated, allocate = self.tracked_pipe()
        with patch.object(os, "pipe", side_effect=allocate), patch.object(
            os, "set_inheritable", side_effect=OSError("SYNTHETIC-PRIVATE-TEXT"),
        ), patch.object(os, "write") as writer:
            self.rejected(lambda: bootstrap.BootstrapPipe(**self.options))
        writer.assert_not_called()
        self.assert_closed(*allocated)

    def test_writer_close_failure_rejects_without_descriptor_number_retry(self):
        allocated, allocate = self.tracked_pipe()
        original_close = os.close
        closed = []

        def close(fd):
            closed.append(fd)
            original_close(fd)
            if fd == allocated[1]:
                raise OSError(errno.EINTR, "SYNTHETIC-PRIVATE-TEXT")

        with patch.object(os, "pipe", side_effect=allocate), patch.object(
            os, "close", side_effect=close,
        ):
            self.rejected(lambda: bootstrap.BootstrapPipe(**self.options))
        self.assertCountEqual(closed, allocated)
        self.assert_closed(*allocated)

    def test_fileno_borrows_same_descriptor_until_closed(self):
        handoff = self.make()
        fd = handoff.fileno()
        self.assertEqual(handoff.fileno(), fd)
        self.assertTrue(handoff.close())
        self.rejected(handoff.fileno)
        self.rejected(handoff.take_fd)
        self.assertTrue(handoff.close())
        self.assert_closed(fd)

    def test_take_fd_transfers_once_and_close_does_not_close_transferred_fd(self):
        handoff = self.make()
        borrowed = handoff.fileno()
        fd = handoff.take_fd()
        self.addCleanup(close_same_fd, fd, identity(fd))
        self.assertEqual(fd, borrowed)
        self.rejected(handoff.take_fd)
        self.rejected(handoff.fileno)
        self.assertTrue(handoff.close())
        self.assertEqual(process.read_bootstrap(fd), self.options)

    def test_close_performs_one_call_and_is_idempotent(self):
        handoff = self.make()
        fd = handoff.fileno()
        original_close = os.close
        with patch.object(os, "close", side_effect=original_close) as closer:
            self.assertTrue(handoff.close())
            self.assertTrue(handoff.close())
        closer.assert_called_once_with(fd)
        self.assert_closed(fd)

    def test_close_failure_is_reported_and_never_retried(self):
        handoff = self.make()
        fd = handoff.fileno()
        original_close = os.close

        def close(endpoint):
            original_close(endpoint)
            raise OSError(errno.EINTR, "SYNTHETIC-PRIVATE-TEXT")

        with patch.object(os, "close", side_effect=close) as closer:
            self.assertFalse(handoff.close())
            self.assertFalse(handoff.close())
        closer.assert_called_once_with(fd)
        self.rejected(handoff.fileno)
        self.assert_closed(fd)

    def test_wrong_pid_cannot_borrow_transfer_or_close(self):
        handoff = self.make()
        fd = handoff.fileno()
        pid = os.getpid()
        with patch.object(os, "getpid", return_value=pid + 1):
            with patch.object(os, "close") as closer:
                self.rejected(handoff.fileno)
                self.rejected(handoff.take_fd)
                try:
                    self.assertFalse(handoff.close())
                except bootstrap.BootstrapError as exc:
                    self.assertEqual(str(exc), ERROR)
                closer.assert_not_called()
        self.assertEqual(handoff.fileno(), fd)
        self.assertEqual(process.read_bootstrap(fd), self.options)

    def test_wrong_thread_cannot_borrow_transfer_or_close(self):
        handoff = self.make()
        fd = handoff.fileno()
        outcomes = []

        def worker():
            for method in (handoff.fileno, handoff.take_fd, handoff.close):
                try:
                    outcomes.append(method())
                except bootstrap.BootstrapError as exc:
                    outcomes.append(str(exc))

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(outcomes[:2], [ERROR, ERROR])
        self.assertIn(outcomes[2], (False, ERROR))
        self.assertEqual(handoff.fileno(), fd)
        self.assertEqual(process.read_bootstrap(fd), self.options)

    def test_reinitializing_open_object_rejects_without_allocating_or_closing(self):
        handoff = self.make()
        fd = handoff.fileno()
        with patch.object(os, "pipe") as pipe, patch.object(os, "close") as closer:
            self.rejected(lambda: handoff.__init__(**self.options))
            pipe.assert_not_called()
            closer.assert_not_called()
        self.assertEqual(handoff.fileno(), fd)
        self.assertEqual(process.read_bootstrap(fd), self.options)

    def test_reentrant_operations_reject_without_closing_or_transferring(self):
        handoff = self.make()
        fd = handoff.fileno()
        original_fstat = os.fstat

        def inspect(endpoint):
            self.rejected(handoff.fileno)
            self.rejected(handoff.take_fd)
            self.assertFalse(handoff.close())
            return original_fstat(endpoint)

        with patch.object(os, "fstat", side_effect=inspect), patch.object(os, "close") as closer:
            self.assertEqual(handoff.fileno(), fd)
            closer.assert_not_called()
        self.assertEqual(process.read_bootstrap(fd), self.options)

    def test_read_end_flag_tampering_detaches_without_closing(self):
        for mutate in (lambda fd: os.set_blocking(fd, True),
                       lambda fd: os.set_inheritable(fd, True)):
            handoff = self.make()
            fd = handoff.fileno()
            expected = identity(fd)
            self.addCleanup(close_same_fd, fd, expected)
            mutate(fd)
            with patch.object(os, "close") as closer:
                self.rejected(handoff.fileno)
                self.assertFalse(handoff.close())
                self.assertFalse(handoff.close())
                closer.assert_not_called()
            self.assertEqual(identity(fd), expected)

    def test_reused_descriptor_number_never_closes_replacement(self):
        original, replacement = self.make(), self.make()
        fd, other = original.fileno(), replacement.fileno()
        expected = identity(other)
        os.dup2(other, fd, inheritable=False)
        self.addCleanup(close_same_fd, fd, expected)
        with patch.object(os, "close") as closer:
            self.rejected(original.take_fd)
            self.assertFalse(original.close())
            closer.assert_not_called()
        self.assertEqual(identity(fd), expected)
        self.assertEqual(process.read_bootstrap(fd), self.options)
        self.assertEqual(replacement.fileno(), other)

    def test_externally_closed_descriptor_is_not_closed_again(self):
        handoff = self.make()
        fd = handoff.fileno()
        os.close(fd)
        with patch.object(os, "close") as closer:
            self.assertFalse(handoff.close())
            self.rejected(handoff.fileno)
            self.assertFalse(handoff.close())
            closer.assert_not_called()

    def test_reinitializing_closed_or_transferred_object_rejects(self):
        for transfer in (False, True):
            handoff = self.make()
            if transfer:
                fd = handoff.take_fd()
                self.addCleanup(close_same_fd, fd, identity(fd))
            else:
                self.assertTrue(handoff.close())
            with self.subTest(transferred=transfer), patch.object(os, "pipe") as pipe:
                self.rejected(partial(handoff.__init__, **self.options))
                pipe.assert_not_called()

    def test_repr_is_redacted_in_open_closed_and_transferred_states(self):
        for state in ("open", "closed", "transferred"):
            handoff = self.make()
            if state == "closed":
                handoff.close()
            elif state == "transferred":
                fd = handoff.take_fd()
                self.addCleanup(close_same_fd, fd, identity(fd))
            rendered = repr(handoff)
            with self.subTest(state=state):
                self.assertIn("redacted", rendered.lower())
                for secret in (TOKEN.decode("ascii"), KEY.decode("ascii"), KEY.hex(), ATTEMPT,
                               PROTECTED, self.options["plan_sha256"], "ssh_key_names"):
                    self.assertNotIn(secret, rendered)


if __name__ == "__main__":
    unittest.main()
