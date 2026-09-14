"""Private generator transport/lifecycle doubles; no native calls or ML imports."""

from __future__ import annotations

import array
import base64
import copy
import importlib
import socket
import stat
import sys
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_synthetic_checkpoint_generator_guard_v2"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
subject = importlib.import_module(PACKAGE + ".posttraining_checkpoint_generator_guard_v2")
PRIVATE = "generated-private-body-never-public"


def environment():
    return dict(subject.child_bootstrap.FIXED_ENV) | {
        "LUMI_GENERATOR_ENTRY_SHA256": "a" * 64,
        "LUMI_IMPORT_SOURCE_MANIFEST_SHA256": "b" * 64,
        "LUMI_GENERATION_NONCE_SHA256": "c" * 64,
    }


class MemoryPath:
    def __init__(self, value):
        self.value = value

    def __truediv__(self, child):
        return MemoryPath(self.value + "/" + child)

    def mkdir(self, **_kwargs):
        pass


def packet():
    """Actual frozen metadata builder on an in-memory path double, never generation."""
    codec = subject.codec
    stored = {"checkpoint/model/.metadata": b"synthetic-meta",
              "checkpoint/model/__0_0.distcp": PRIVATE.encode("ascii")}

    def write(path, raw):
        if path.value in stored:
            raise AssertionError("fixture_rewrite")
        stored[path.value] = raw

    def inventory(path, callback):
        return [{"path": key[len(path.value) + 1:], "bytes": len(raw), "sha256": codec.sha(raw)}
                for key, raw in sorted(stored.items()) if key.startswith(path.value + "/")]

    with patch.object(codec.fixture, "_write", side_effect=write), \
         patch.object(codec.fixture, "_inventory", side_effect=inventory):
        metadata = codec.fixture._metadata(MemoryPath("checkpoint"), MemoryPath("tokenizer"),
                                            lambda: None)
    inputs = {role: {key[len(role) + 1:]: raw for key, raw in stored.items()
                     if key.startswith(role + "/")} for role in ("checkpoint", "tokenizer")}
    plan = codec._original_plan(metadata)
    generated = {"metadata": metadata, "generated_inputs": inputs,
        "receipt": codec._generation_receipt(metadata, inputs, plan, "c" * 64)}
    source = {"source_manifest_sha256": "b" * 64, "source_files_verified": 35,
              "source_bytes_verified": 600000, "source_verification_read_bytes": 610000}
    copied = {"private_source_copy_files": 35, "private_source_copy_bytes_reserved": 600000,
              "private_source_copy_readback_bytes": 600000,
              "private_source_copies_rootfs_immutable": False,
              "hostile_owner_code_immutability_verified": False}
    return codec.encode_generated_packet(generated, entry_sha256="a" * 64,
        source_manifest_sha256="b" * 64, generation_nonce_sha256="c" * 64,
        source_report=source, copy_report=copied)


def final(raw):
    return {"status": "passed", "case": "worker_run", "reason": "workload_empty",
            "empty_and_removed": True, "workload_processes_reaped": 1, "kill_attempts": 1,
            "private_socket_buffers_verified": True,
            "private_packet_b64": base64.b64encode(raw).decode("ascii")}


def channel():
    value = Mock()
    value.family = socket.AF_UNIX
    value.gettimeout.return_value = None
    value.getsockopt.side_effect = lambda _level, key: (
        socket.SOCK_SEQPACKET if key == socket.SO_TYPE else subject.MINIMUM_SOCKET_BUFFER_BYTES)
    value.send.side_effect = len
    return value


class Exited(BaseException):
    def __init__(self, code):
        self.code = code


def exit_native(code):
    raise Exited(code)


class Lease:
    def __init__(self):
        self.populated = self.kills = self.closes = 0
        self.close_error = self.kill_error = False

    def verify_limits(self):
        pass

    def snapshot(self):
        return {"populated": self.populated, "pids_current": self.populated}

    def fileno(self):
        return 61

    def kill(self):
        self.kills += 1
        self.populated = 0
        if self.kill_error:
            raise OSError(PRIVATE)

    def close(self):
        self.closes += 1
        if self.close_error:
            raise OSError(PRIVATE)


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(subject, "_ATTEMPTED", False))
        self.raw = packet()

    def api(self, result=None):
        self.stack.enter_context(patch.object(subject.child_bootstrap, "_runtime"))
        self.stack.enter_context(patch.object(subject.time, "monotonic_ns", return_value=100))
        base = types.SimpleNamespace(subreaper=Mock(), run_case=Mock(
            return_value=final(self.raw) if result is None else result), wait_statuses=Mock())
        loader = self.stack.enter_context(patch.object(subject, "_load_base", return_value=base))
        return base, loader

    def invoke(self, **changes):
        arguments = {"rootfs_path": "/tmp/lumi-generator/rootfs",
            "mountpoint_path": "/tmp/lumi-generator/mountpoint",
            "root_identity": (1, 2, stat.S_IFDIR, 0, 0), "environment": environment(),
            "base_script_path": "/release/scripts/run_sft_cgroup_fixture_v1.py",
            "expected_base_sha256": subject.BASE_SHA256, "deadline_ns": 90000000000}
        return subject.run_generator_guard(**(arguments | changes))

    def native(self, *, ready=None, raw=None, eof=b"", status=0, watch_error=False,
               close_error=False, kill_error=False, duplicate_release=False,
               small_outer_buffer=False, child_branch=False):
        events, lease, values = [], Lease(), []
        lease.close_error, lease.kill_error = close_error, kill_error
        outer, parent, child = channel(), channel(), channel()
        outer.fileno.return_value = 80
        if small_outer_buffer:
            outer.getsockopt.side_effect = lambda _level, key: (
                socket.SOCK_SEQPACKET if key == socket.SO_TYPE else 4096)
        parent.send.side_effect = lambda body: events.append(("release", body)) or len(body)

        def receive(_channel, _deadline):
            lease.populated = 1
            events.append("ready")
            return copy.deepcopy(subject.child_bootstrap.READY if ready is None else ready)

        def watched(passed, *, expected_parent_pid, deadline_ns, cleanup_timeout_ms, release):
            self.assertIs(passed, lease)
            self.assertEqual((expected_parent_pid, cleanup_timeout_ms), (77, 10000))
            events.append("watch_installed")
            release()
            if duplicate_release:
                release()
            lease.kill()
            if watch_error:
                raise OSError(PRIVATE)
            return {"reason": "workload_empty", "kill_attempted": True, "empty_verified": True,
                    "parent_pidfd_bound": True, "release_called": True, "pidfd_closed": True}

        modules = {"posttraining_sft_cgroup_v1": types.SimpleNamespace(
                       CgroupLease=Mock(return_value=lease)),
                   "posttraining_sft_cgroup_guardian_v1": types.SimpleNamespace(
                       watch_parent=watched)}
        base = types.SimpleNamespace(close_except=Mock(), subreaper=Mock(),
            module=modules.__getitem__, wait_statuses=Mock(return_value={123: status}),
            send=lambda _channel, value: values.append(copy.deepcopy(value)))
        self.stack.enter_context(patch.object(subject.os, "open", return_value=51))
        self.stack.enter_context(patch.object(subject.os, "close"))
        def fork_ready():
            self.assertEqual(parent.setsockopt.call_count, 2)
            self.assertEqual(child.setsockopt.call_count, 2)
            events.append("fork_after_socket_readback")
            return 0 if child_branch else 123

        fork = self.stack.enter_context(patch.object(subject.os, "fork", side_effect=fork_ready))
        self.stack.enter_context(patch.object(subject.os, "_exit", side_effect=exit_native))
        self.stack.enter_context(patch.object(subject.socket, "socketpair",
                                              return_value=(parent, child)))
        self.stack.enter_context(patch.object(subject, "_receive", side_effect=receive))
        receiving = self.stack.enter_context(patch.object(subject, "_packet",
            side_effect=[self.raw if raw is None else raw, eof]))
        self.stack.enter_context(patch.object(subject.time, "monotonic_ns", return_value=100))
        bootstrap = self.stack.enter_context(patch.object(subject.child_bootstrap, "bootstrap"))
        context = {"rootfs_path": "/tmp/lumi-generator/rootfs",
                   "mountpoint_path": "/tmp/lumi-generator/mountpoint",
                   "root_identity": (1, 2, stat.S_IFDIR, 0, 0), "environment": environment()}
        with self.assertRaises(Exited) as caught:
            subject._guard(base, context, "worker_run", outer, 77, 10001, 10001,
                           subject.child_bootstrap.RUNTIME, 90000000000)
        fork.assert_called_once_with()
        if not child_branch:
            bootstrap.assert_not_called()
        return values[-1], lease, events, caught.exception.code, receiving, bootstrap, context

    def test_private_api_returns_separate_validated_data_and_public_aggregate(self):
        base, _loader = self.api()
        result = self.invoke()
        self.assertEqual(set(result), {"private_data", "receipt"})
        self.assertEqual(len(result["private_data"]["metadata"]), 5)
        self.assertIn(PRIVATE.encode(),
                      result["private_data"]["generated_inputs"]["checkpoint"].values())
        receipt = result["receipt"]
        self.assertEqual(receipt["schema"], subject.SCHEMA)
        self.assertNotIn(PRIVATE, subject._canonical(receipt).decode())
        self.assertNotIn("private_packet_b64", receipt)
        self.assertTrue(all(receipt[key] is False for key in subject.NONCLAIMS))
        base.run_case.assert_called_once_with("worker_run", 10001, 10001,
                                             subject.child_bootstrap.RUNTIME, 90000000000)
        self.assertIs(base.send, subject._private_send)
        self.assertIs(base.receive, subject._receive)
        self.assertIsNot(base.socket, socket)
        self.assertIs(base.socket.socketpair, subject._configured_socketpair)

    def test_codec_checks_entry_source_nonce_and_private_payload(self):
        decoded = subject._worker(self.raw, environment())
        self.assertEqual(decoded["receipt"]["packet_bytes"], len(self.raw))
        for key in subject.child_bootstrap.HASH_ENV:
            with self.subTest(key=key), self.assertRaisesRegex(
                    subject.GuardCompositionError, "worker_rejected"):
                subject._worker(self.raw, environment() | {key: "d" * 64})
        for raw in (b"{}", self.raw + b"\n", b"x" * 65537):
            with self.assertRaisesRegex(subject.GuardCompositionError, "worker_rejected"):
                subject._worker(raw, environment())

    def test_native_watch_precedes_release_and_empty_reap_precedes_packet_acceptance(self):
        result, lease, events, code, receiving, *_ = self.native()
        self.assertEqual(result, final(self.raw))
        self.assertEqual((lease.kills, lease.closes, code), (1, 1, 0))
        self.assertLess(events.index("fork_after_socket_readback"), events.index("ready"))
        self.assertLess(events.index("watch_installed"), events.index(("release", b"G")))
        self.assertEqual(receiving.call_args_list[0].kwargs, {"maximum": 65536})
        self.assertEqual(receiving.call_count, 2)

    def test_failed_exit_never_reads_private_packet_despite_possible_success_body(self):
        result, lease, _events, code, receiving, *_ = self.native(status=256)
        self.assertEqual((result["status"], lease.kills, lease.closes, code), ("failed", 1, 1, 2))
        receiving.assert_not_called()
        self.assertNotIn("private_packet_b64", result)

    def test_extra_packet_is_terminal_without_private_failure_output(self):
        result, lease, *_ = self.native(eof=b"PRIVATE")
        self.assertEqual((result["status"], lease.kills, lease.closes), ("failed", 1, 1))
        self.assertNotIn("private_packet_b64", result)

    def test_ready_type_or_extra_field_prevents_watch_and_release(self):
        result, _lease, events, *_ = self.native(
            ready=subject.child_bootstrap.READY | {"extra": True})
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("watch_installed", events)

    def test_outer_buffer_readback_failure_prevents_release(self):
        result, lease, events, *_ = self.native(small_outer_buffer=True)
        self.assertEqual((result["status"], lease.kills, lease.closes), ("failed", 1, 1))
        self.assertNotIn(("release", b"G"), events)

    def test_child_bootstrap_receives_only_fixed_root_controls_and_environment(self):
        _result, _lease, _events, _code, _receiving, bootstrap, context = self.native(
            child_branch=True)
        self.assertEqual(bootstrap.call_count, 1)
        self.assertEqual(set(bootstrap.call_args.kwargs),
                         {"rootfs_path", "mountpoint_path", "root_identity", "environment"})
        self.assertEqual(bootstrap.call_args.kwargs, context)

    def test_no_repeat_release_after_callback_failure(self):
        result, lease, events, *_ = self.native(duplicate_release=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(events.count(("release", b"G")), 1)
        self.assertEqual(lease.kills, 1)

    def test_ambiguous_kill_not_repeated(self):
        result, lease, *_ = self.native(kill_error=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(lease.kills, 1)

    def test_ambiguous_lease_close_reports_cleanup_unconfirmed(self):
        result, lease, *_ = self.native(close_error=True)
        self.assertEqual(result["code"], "cleanup_unconfirmed")
        self.assertEqual(lease.closes, 1)

    def test_watch_failure_has_one_kill_and_no_private_result(self):
        result, lease, *_ = self.native(watch_error=True)
        self.assertEqual((result["status"], lease.kills), ("failed", 1))
        self.assertNotIn("private_packet_b64", result)

    def test_complete_outer_packet_reserved_cap_exceeds_largest_worker_wrapper(self):
        raw = subject._canonical(final(b"x" * 65536))
        self.assertGreater(len(raw), 65536)
        self.assertLess(len(raw), subject.MAXIMUM_PACKET_BYTES)
        self.assertEqual(subject._parse(raw), final(b"x" * 65536))

    def test_both_socket_endpoints_configured_then_read_back(self):
        left, right = channel(), channel()
        with patch.object(subject.socket, "socketpair", return_value=(left, right)):
            result = subject._configured_socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.assertEqual(result, (left, right))
        for endpoint in result:
            self.assertEqual([call.args for call in endpoint.setsockopt.call_args_list],
                [(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144),
                 (socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)])
            endpoint.close.assert_not_called()

    def test_small_socket_buffer_closes_both_without_retry(self):
        left, right = channel(), channel()
        right.getsockopt.side_effect = lambda _level, key: (
            socket.SOCK_SEQPACKET if key == socket.SO_TYPE else 262143)
        with patch.object(subject.socket, "socketpair", return_value=(left, right)), \
             self.assertRaisesRegex(subject.GuardCompositionError, "socket_buffer_rejected"):
            subject._configured_socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        left.close.assert_called_once()
        right.close.assert_called_once()

    def test_socket_setup_close_failure_preserves_cleanup_uncertainty(self):
        left, right = channel(), channel()
        left.setsockopt.side_effect = OSError(PRIVATE)
        right.close.side_effect = OSError(PRIVATE)
        with patch.object(subject.socket, "socketpair", return_value=(left, right)), \
             self.assertRaisesRegex(subject.GuardCompositionError, "cleanup_unconfirmed"):
            subject._configured_socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        left.close.assert_called_once()
        right.close.assert_called_once()

    def test_private_send_caps_whole_envelope_and_short_send_not_retried(self):
        endpoint = channel()
        subject._private_send(endpoint, final(self.raw))
        self.assertEqual(endpoint.send.call_count, 1)
        with self.assertRaisesRegex(subject.GuardCompositionError, "packet_rejected"):
            subject._private_send(endpoint, {"private": "x" * subject.MAXIMUM_PACKET_BYTES})
        self.assertEqual(endpoint.send.call_count, 1)
        endpoint.send.side_effect = lambda raw: len(raw) - 1
        with self.assertRaisesRegex(subject.GuardCompositionError, "packet_rejected"):
            subject._private_send(endpoint, final(self.raw))
        self.assertEqual(endpoint.send.call_count, 2)

    def test_environment_copied_before_loader_and_independent_guard_context(self):
        base, loader = self.api()
        supplied = environment()
        observed = []

        def load(*_args):
            supplied["LUMI_GENERATION_NONCE_SHA256"] = "d" * 64
            supplied["PATH"] = PRIVATE
            return base

        def run(*_args):
            base.guardian("worker_run", "synthetic-channel", 77, 10001, 10001,
                          subject.child_bootstrap.RUNTIME, 90000000000)
            return final(self.raw)

        loader.side_effect = load
        base.run_case.side_effect = run
        with patch.object(subject, "_guard", side_effect=lambda _base, context, *_args:
                          observed.append(context)):
            result = self.invoke(environment=supplied)
        self.assertEqual(observed[0]["environment"], environment())
        self.assertIsNot(observed[0]["environment"], supplied)
        self.assertEqual(result["receipt"]["packet_receipt"]["generation_nonce_sha256"], "c" * 64)

    def test_untrusted_outer_count_types_and_worker_only_receipt_rejected(self):
        cases = [subject._worker(self.raw, environment())["receipt"],
                 final(self.raw) | {"private_socket_buffers_verified": 1},
                 final(self.raw) | {"kill_attempts": True},
                 final(self.raw) | {"real_4b_import": True}]
        for result in cases:
            with self.subTest(fields=sorted(result)):
                base, _ = self.api(result)
                with self.assertRaises(subject.GuardCompositionError):
                    self.invoke()
                base.run_case.assert_called_once()
                subject._ATTEMPTED = False

    def test_malformed_private_base64_rejected(self):
        for encoded in ("!invalid!", "eA==\n", "x" * 87385):
            with self.subTest(length=len(encoded)):
                self.api(final(self.raw) | {"private_packet_b64": encoded})
                with self.assertRaisesRegex(subject.GuardCompositionError, "packet_rejected"):
                    self.invoke()
                subject._ATTEMPTED = False

    def test_runtime_denial_before_any_source_io(self):
        native = self.stack.enter_context(patch.object(subject.child_bootstrap, "_runtime",
                                                       side_effect=RuntimeError(PRIVATE)))
        loader = self.stack.enter_context(patch.object(subject, "_load_base"))
        with self.assertRaisesRegex(subject.GuardCompositionError, "runtime_rejected"):
            self.invoke()
        native.assert_called_once()
        loader.assert_not_called()
        self.assertFalse(subject._ATTEMPTED)

    def test_bad_paths_nonce_or_deadline_reject_before_started_load(self):
        _base, loader = self.api()
        for changes in ({"base_script_path":
                         "/release/scripts/../scripts/run_sft_cgroup_fixture_v1.py"},
                        {"base_script_path": "relative/scripts/run_sft_cgroup_fixture_v1.py"},
                        {"base_script_path": "/release/scripts/other.py"},
                        {"rootfs_path": "/tmp/lumi generator/rootfs"},
                        {"deadline_ns": True}, {"deadline_ns": 100},
                        {"environment": environment() | {"LUMI_GENERATION_NONCE_SHA256": "0" * 64}},
                        {"expected_base_sha256": "0" * 64}):
            with self.subTest(fields=sorted(changes)), self.assertRaises(
                    subject.GuardCompositionError):
                self.invoke(**changes)
        loader.assert_not_called()
        self.assertFalse(subject._ATTEMPTED)

    def test_started_failure_has_no_retry_and_redacts_private_details(self):
        base, loader = self.api()
        loader.side_effect = OSError(PRIVATE)
        for expected in ("guard_failed", "already_attempted"):
            with self.assertRaises(subject.GuardCompositionError) as caught:
                self.invoke()
            self.assertEqual(caught.exception.args, (expected,))
            self.assertTrue(caught.exception.__suppress_context__)
        loader.assert_called_once()
        base.run_case.assert_not_called()

    def test_guard_failure_cleanup_code_preserved(self):
        base, _loader = self.api()
        base.run_case.side_effect = subject.GuardCompositionError("cleanup_unconfirmed")
        with self.assertRaisesRegex(subject.GuardCompositionError, "cleanup_unconfirmed"):
            self.invoke()
        base.wait_statuses.assert_called_once()

    def test_late_deadline_rejects_success(self):
        base, _loader = self.api()
        self.stack.enter_context(patch.object(subject.time, "monotonic_ns",
                                              side_effect=[100, 90000000001, 90000000002]))
        with self.assertRaisesRegex(subject.GuardCompositionError, "deadline_exhausted"):
            self.invoke()
        base.wait_statuses.assert_called_once()

    def test_receive_truncation_ancillary_rights_closed_and_not_decoded(self):
        endpoint = channel()
        self.stack.enter_context(patch.object(subject.time, "monotonic_ns", return_value=0))
        self.stack.enter_context(patch.object(subject.select, "select",
                                              return_value=([endpoint], [], [])))
        close = self.stack.enter_context(patch.object(subject.os, "close"))
        rights = array.array("i", [81, 82]).tobytes()
        for ancillary, flags in (([], socket.MSG_TRUNC), ([], socket.MSG_CTRUNC),
                                 ([(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights)], 0)):
            endpoint.recvmsg.return_value = (self.raw, ancillary, flags, None)
            with self.assertRaisesRegex(subject.GuardCompositionError, "packet_rejected"):
                subject._packet(endpoint, 1000000000, maximum=65536)
        self.assertEqual([call.args for call in close.call_args_list], [(81,), (82,)])
        self.assertEqual(endpoint.recvmsg.call_args.args, (65536, 256))

    def test_outer_parser_duplicate_noncanonical_and_deep_values_rejected(self):
        for raw in (b'{"x":1,"x":1}', b'{"x":NaN}', b'{"x":1.0}', b'{"x":[]}',
                    b'{"x":null}', b' {"x":1}', b'{"x":-1}', b"x" * 131073,
                    subject._canonical({"a": {"b": {"c": {"d": {"e": 1}}}}})):
            with self.assertRaisesRegex(subject.GuardCompositionError, "packet_rejected"):
                subject._parse(raw)


if __name__ == "__main__":
    unittest.main()

