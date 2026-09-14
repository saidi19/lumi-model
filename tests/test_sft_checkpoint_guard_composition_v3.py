"""Pure doubles: no native cgroup, namespace, rootfs, fork, ML or real input."""

import array
import copy
import importlib
import socket
import stat
import sys
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_lumi_prepared_guard_v2_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
subject = importlib.import_module(PACKAGE + ".posttraining_checkpoint_guard_composition_v3")
NONCE = "c" * 64


def mount_arguments():
    parent = "/tmp/lumi-prepared-root"
    identities = {}
    for index, role in enumerate(subject.child_bootstrap.mount.input_mount.ROLES, 1):
        uid = 10001 if role in {"checkpoint", "tokenizer", "output"} else 0
        mode = 0o755 if role in {"root", "inputs"} else 0o700
        identities[role] = (1, index, stat.S_IFDIR | mode, uid, uid, 2, 0, 1, 1)
    return {"rootfs_path": parent + "/rootfs", "mountpoint_path": parent + "/mount",
            "checkpoint_path": parent + "/checkpoint", "tokenizer_path": parent + "/tokenizer",
            "output_path": parent + "/output", "expected_identities": identities,
            "expected_mountpoint_identity": (1, 100, stat.S_IFDIR | 0o700, 0, 0, 2, 0, 1, 1),
            "inventories": {"checkpoint": {"model/generated": 10},
                            "tokenizer": {"tokenizer/generated": 10}},
            "scratch_bytes": 536870912, "shm_bytes": 67108864}


def environment():
    return {**subject.child_bootstrap.FIXED_ENV,
            "LUMI_IMPORT_SOURCE_MANIFEST_SHA256": "a" * 64,
            "LUMI_PREPARED_FIXTURE_ENTRY_SHA256": "b" * 64,
            "LUMI_PREPARED_FIXTURE_CONTROL_SHA256": "d" * 64}


def worker():
    return {
        "schema": subject.WORKER_SCHEMA, "status": "passed", "code": "ok",
        "synthetic_only": True, "coordinator_passes": 1, "checkpoint_full_audits": 2,
        "tokenizer_full_audits": 2, "model_parameters": 736,
        "generated_input_bytes": 20, "generated_input_files": 2,
        "coordinator_result_sha256": "f" * 64,
        "control_sha256": "d" * 64, "generation_nonce_sha256": NONCE,
        "coordinator_read_bytes_reserved": 4096, "private_write_bytes_reserved": 4096,
        "real_4b_import": False, "training_authorized": False, "production_eligible": False,
        "retry_allowed": False, "parent_supervisor_verification_required": True,
        "resource_envelope_is_planning_only": True, "host_capacity_verified": False,
        "source_immutability_verified": False, "physical_device_io_measured": False,
        "forward_calls": 0, "optimizer_steps": 0,
        "source_manifest_sha256": "a" * 64, "source_files_verified": 35,
        "source_bytes_verified": 610752, "source_verification_read_bytes": 615000,
        "private_source_copy_bytes_reserved": 610752,
        "private_source_copy_readback_bytes": 610752, "private_source_copy_files": 35,
        "private_source_copies_rootfs_immutable": False,
        "hostile_owner_code_immutability_verified": False, "entry_sha256": "b" * 64,
    }

def final():
    return {"status": "passed", "case": "worker_run", "reason": "workload_empty",
            "empty_and_removed": True, "workload_processes_reaped": 1,
            "kill_attempts": 1, "worker": worker()}


class ExitNative(BaseException):
    def __init__(self, code):
        self.code = code


def exit_native(code):
    raise ExitNative(code)


class Lease:
    def __init__(self):
        self.populated = 0
        self.kills = self.closes = 0
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
            raise OSError("private native error")

    def close(self):
        self.closes += 1
        if self.close_error:
            raise OSError("private close error")


class CompositionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(subject, "_ATTEMPTED", False))

    def native_doubles(self, *, ready=None, report=None, eof=b"", status=0,
                       watch_error=False, close_error=False, kill_error=False,
                       duplicate_release=False, descriptor_close_error=False):
        events, lease = [], Lease()
        lease.close_error, lease.kill_error = close_error, kill_error
        outer, parent, child = Mock(), Mock(), Mock()
        outer.fileno.return_value = 80
        parent.send.side_effect = lambda raw: events.append(("release", raw)) or len(raw)
        outer_result = []

        def send(channel, value):
            self.assertIs(channel, outer)
            events.append(("send", value.get("phase", value.get("status"))))
            outer_result.append(copy.deepcopy(value))

        def receive(_channel, _deadline):
            if "ready" not in events:
                events.append("ready")
                lease.populated = 1
                return copy.deepcopy(subject.child_bootstrap.READY if ready is None else ready)
            return copy.deepcopy(worker() if report is None else report)

        def watch(passed, *, expected_parent_pid, deadline_ns, cleanup_timeout_ms, release):
            self.assertIs(passed, lease)
            self.assertEqual(expected_parent_pid, 77)
            self.assertEqual(cleanup_timeout_ms, 10000)
            events.append("watch_installed")
            release()
            if duplicate_release:
                release()
            lease.kill()
            if watch_error:
                raise OSError("private watch failure")
            return {"reason": "workload_empty", "kill_attempted": True,
                    "empty_verified": True, "parent_pidfd_bound": True,
                    "release_called": True, "pidfd_closed": True}

        factory = Mock(return_value=lease)
        modules = {"posttraining_sft_cgroup_v1": types.SimpleNamespace(CgroupLease=factory),
                   "posttraining_sft_cgroup_guardian_v1": types.SimpleNamespace(watch_parent=watch)}
        base = types.SimpleNamespace(close_except=Mock(), subreaper=Mock(),
                                     module=modules.__getitem__, send=send,
                                     wait_statuses=Mock(return_value={123: status}))
        self.stack.enter_context(patch.object(subject.os, "open", return_value=51))
        closing = self.stack.enter_context(patch.object(subject.os, "close"))
        if descriptor_close_error:
            closing.side_effect = OSError("private descriptor close error")
        fork = self.stack.enter_context(patch.object(subject.os, "fork", return_value=123))
        self.stack.enter_context(patch.object(subject.os, "_exit", side_effect=exit_native))
        self.stack.enter_context(patch.object(subject.socket, "socketpair",
                                              return_value=(parent, child)))
        self.stack.enter_context(patch.object(subject, "_receive", side_effect=receive))
        self.stack.enter_context(patch.object(subject, "_packet", return_value=eof))
        self.stack.enter_context(patch.object(subject.time, "monotonic_ns", return_value=100))
        bootstrap = self.stack.enter_context(patch.object(subject.child_bootstrap, "bootstrap"))
        context = {"mount_arguments": mount_arguments(), "environment": environment()}
        with self.assertRaises(ExitNative) as captured:
            subject._guard(base, context, NONCE, "worker_run", outer, 77, 10001, 10001,
                           subject.child_bootstrap.RUNTIME, 90_000_000_000)
        bootstrap.assert_not_called()
        fork.assert_called_once_with()
        return outer_result[-1], lease, events, captured.exception.code, factory, parent, child

    def api_doubles(self, result=None):
        self.stack.enter_context(patch.object(subject.child_bootstrap, "_runtime"))
        self.stack.enter_context(patch.object(subject.time, "monotonic_ns", return_value=100))
        base = types.SimpleNamespace(subreaper=Mock(), run_case=Mock(return_value=final()
                                     if result is None else result), wait_statuses=Mock())
        loader = self.stack.enter_context(patch.object(subject, "_load_base", return_value=base))
        return base, loader

    def invoke(self, **overrides):
        arguments = dict(mount_arguments=mount_arguments(), environment=environment(),
                         expected_generation_nonce_sha256=NONCE,
                         base_script_path="/release/scripts/run_sft_cgroup_fixture_v1.py",
                         expected_base_sha256=subject.BASE_SHA256, deadline_ns=90_000_000_000)
        arguments.update(overrides)
        return subject.run_prepared_synthetic_guard(**arguments)

    def test_success_orders_watch_before_exact_release_and_reaps(self):
        result, lease, events, code, factory, parent, child = self.native_doubles()
        self.assertEqual(result, final())
        self.assertEqual((lease.kills, lease.closes, code), (1, 1, 0))
        self.assertLess(events.index("watch_installed"), events.index(("release", b"G")))
        self.assertEqual(factory.call_args.kwargs["limits"], subject.LIMITS)
        parent.close.assert_called_once()
        child.close.assert_called_once()

    def test_bool_as_integer_ready_is_rejected_before_release(self):
        ready = {**subject.child_bootstrap.READY, "isolation_installed": 1}
        result, lease, events, code, *_ = self.native_doubles(ready=ready)
        self.assertEqual((result["status"], lease.kills, lease.closes, code), ("failed", 1, 1, 2))
        self.assertNotIn("watch_installed", events)

    def test_extra_ready_field_rejected(self):
        result, _, events, *_ = self.native_doubles(ready={**subject.child_bootstrap.READY, "x": 1})
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("watch_installed", events)

    def test_failed_worker_exit_denies_despite_success_report(self):
        result, lease, *_ = self.native_doubles(status=256)
        self.assertEqual((result["status"], lease.kills), ("failed", 1))

    def test_extra_worker_packet_denies(self):
        result, lease, *_ = self.native_doubles(eof=b"extra")
        self.assertEqual((result["status"], lease.kills, lease.closes), ("failed", 1, 1))

    def test_watch_failure_no_second_kill(self):
        result, lease, *_ = self.native_doubles(watch_error=True)
        self.assertEqual((result["status"], lease.kills, lease.closes), ("failed", 1, 1))

    def test_ambiguous_kill_no_retry(self):
        result, lease, *_ = self.native_doubles(kill_error=True)
        self.assertEqual((result["status"], lease.kills), ("failed", 1))

    def test_ambiguous_lease_close_no_retry(self):
        result, lease, *_ = self.native_doubles(close_error=True)
        self.assertEqual((result["code"], lease.closes), ("cleanup_unconfirmed", 1))

    def test_late_descriptor_close_cannot_report_success(self):
        result, lease, *_ = self.native_doubles(descriptor_close_error=True)
        self.assertEqual((result["code"], lease.kills, lease.closes), ("cleanup_unconfirmed", 1, 1))

    def test_release_cannot_repeat(self):
        result, _, events, *_ = self.native_doubles(duplicate_release=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(events.count(("release", b"G")), 1)

    def test_success_api_is_fixed_and_has_no_authority(self):
        base, _ = self.api_doubles()
        result = self.invoke()
        self.assertTrue(all(result[key] is False for key in subject.NONCLAIMS))
        self.assertEqual(result["worker"], worker())
        base.run_case.assert_called_once_with("worker_run", 10001, 10001,
                                             subject.child_bootstrap.RUNTIME, 90_000_000_000)
        self.assertIs(base.receive, subject._receive)
        self.assertTrue(callable(base.guardian))

    def test_started_attempt_cannot_retry_after_load_failure(self):
        _, loader = self.api_doubles()
        loader.side_effect = OSError("private source path")
        for code in ("guard_failed", "already_attempted"):
            with self.assertRaises(subject.GuardCompositionError) as captured:
                self.invoke()
            self.assertEqual(str(captured.exception), code)
        loader.assert_called_once()

    def test_native_preflight_precedes_all_io(self):
        native = self.stack.enter_context(patch.object(subject.child_bootstrap, "_runtime",
                                                       side_effect=RuntimeError("private")))
        loader = self.stack.enter_context(patch.object(subject, "_load_base"))
        with self.assertRaisesRegex(subject.GuardCompositionError, "runtime_rejected"):
            self.invoke()
        native.assert_called_once()
        loader.assert_not_called()
        self.assertFalse(subject._ATTEMPTED)

    def test_deadline_and_closed_environment_rejected_before_attempt(self):
        _, loader = self.api_doubles()
        for override in ({"deadline_ns": True}, {"deadline_ns": 100},
                         {"deadline_ns": 91_000_000_000},
                         {"environment": {**environment(), "LD_PRELOAD": "/private"}}):
            with self.subTest(override=tuple(override)), self.assertRaises(
                    subject.GuardCompositionError):
                self.invoke(**override)
        loader.assert_not_called()
        self.assertFalse(subject._ATTEMPTED)

    def test_paths_and_identity_rejected_before_attempt(self):
        _, loader = self.api_doubles()
        bad = mount_arguments()
        bad["rootfs_path"] = "/owned/../root"
        for override in ({"mount_arguments": bad}, {"mount_arguments": {}},
                         {"expected_generation_nonce_sha256": True},
                         {"expected_generation_nonce_sha256": "0" * 64},
                         {"expected_generation_nonce_sha256": "unbound"},
                         {"expected_base_sha256": "a" * 64}):
            with self.subTest(override=tuple(override)), self.assertRaises(
                    subject.GuardCompositionError):
                self.invoke(**override)
        loader.assert_not_called()

    def test_failed_outer_case_reaps_but_does_not_claim_removal(self):
        base, _ = self.api_doubles()
        base.run_case.side_effect = OSError("private failure")
        with self.assertRaisesRegex(subject.GuardCompositionError, "guard_failed"):
            self.invoke()
        base.wait_statuses.assert_called_once()

    def test_tiny_input_and_overlay_caps_precede_attempt(self):
        _, loader = self.api_doubles()
        variants = []
        for key, value in (("scratch_bytes", 1024), ("shm_bytes", 1024)):
            selected = mount_arguments()
            selected[key] = value
            variants.append(selected)
        selected = mount_arguments()
        selected["inventories"]["checkpoint"] = {"model/generated": 4 * 1024**2}
        variants.append(selected)
        selected = mount_arguments()
        selected["inventories"]["checkpoint"] = {"model/f" + str(i): 1 for i in range(32)}
        variants.append(selected)
        for mounts in variants:
            with self.assertRaisesRegex(subject.GuardCompositionError, "arguments_rejected"):
                self.invoke(mount_arguments=mounts)
        self.assertFalse(subject._ATTEMPTED)
        loader.assert_not_called()

    def test_guard_and_controller_match_worker_to_mounted_inventory_counts(self):
        for key in ("generated_input_bytes", "generated_input_files"):
            report = {**worker(), key: worker()[key] + 1}
            result, lease, _, code, *_ = self.native_doubles(report=report)
            self.assertEqual((result["status"], lease.closes, code), ("failed", 1, 2))
        with self.assertRaisesRegex(subject.GuardCompositionError, "worker_rejected"):
            subject._match_input_counts({**worker(), "generated_input_bytes": 21},
                                        mount_arguments())

    def test_context_is_detached_and_does_not_send_nonce_as_bootstrap_kwarg(self):
        base, loader = self.api_doubles()
        original_mounts, original_env = mount_arguments(), environment()
        observed = []

        def load(*_args):
            original_mounts["inventories"]["checkpoint"]["model/generated"] = 99
            original_mounts["expected_identities"]["root"] = (0,) * 9
            original_env["LUMI_PREPARED_FIXTURE_CONTROL_SHA256"] = "e" * 64
            return base

        def run(*args):
            base.guardian(*args)
            return final()

        loader.side_effect, base.run_case.side_effect = load, run
        with patch.object(subject, "_guard", side_effect=lambda *args: observed.append(args)):
            result = self.invoke(mount_arguments=original_mounts, environment=original_env)
        context = observed[0][1]
        self.assertEqual(set(context), {"mount_arguments", "environment"})
        self.assertEqual(context["mount_arguments"], mount_arguments())
        self.assertEqual(context["environment"], environment())
        self.assertEqual(observed[0][2], NONCE)
        self.assertEqual(result["worker"], worker())

    def test_outer_cleanup_failure_has_fixed_code(self):
        base, _ = self.api_doubles()
        base.run_case.side_effect = OSError("private failure")
        base.wait_statuses.side_effect = OSError("private cleanup")
        with self.assertRaisesRegex(subject.GuardCompositionError, "cleanup_unconfirmed"):
            self.invoke()

    def test_late_deadline_denies_success(self):
        self.api_doubles()
        self.stack.enter_context(patch.object(subject.time, "monotonic_ns",
                                              side_effect=[100, 90_000_000_001, 90_000_000_002]))
        with self.assertRaisesRegex(subject.GuardCompositionError, "deadline_exhausted"):
            self.invoke()

    def test_worker_exact_schema_and_caps(self):
        self.assertEqual(subject._worker(worker(), environment(), NONCE), worker())
        for key, value in (("coordinator_passes", True), ("model_parameters", 737),
                           ("source_bytes_verified", 1), ("source_files_verified", 1),
                           ("entry_sha256", "c" * 64), ("training_authorized", True),
                           ("unapproved_field", 0), ("checkpoint_full_audits", 3),
                           ("control_sha256", "e" * 64), ("generation_nonce_sha256", "e" * 64),
                           ("resource_envelope_is_planning_only", False),
                           ("host_capacity_verified", True)):
            with self.subTest(key=key), self.assertRaisesRegex(
                    subject.GuardCompositionError, "worker_rejected"):
                subject._worker({**worker(), key: value}, environment(), NONCE)

    def test_packet_round_trip_and_finite_closed_types(self):
        self.assertEqual(subject._parse(subject._canonical(worker()) + b"\n"), worker())
        for raw in (b'{"x":1,"x":1}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1.0}',
                    b'{"x":-1}', b'{"x":9223372036854775808}', b'{"x":null}',
                    b'{"x":[]}', b' {"x":1}', b'[]', b'{}\n\n', b'\xff', b'a' * 4097):
            with self.subTest(raw=raw[:30]), self.assertRaises(subject.GuardCompositionError):
                subject._parse(raw)

    def test_packet_depth_and_string_bounded(self):
        for value in ({"a": {"b": {"c": {"d": {"e": 1}}}}}, {"x": "x" * 257},
                      {"x" * 129: 1}):
            with self.assertRaises(subject.GuardCompositionError):
                subject._parse(subject._canonical(value))

    def test_receive_rejects_truncation_and_ancillary_and_closes_rights(self):
        channel = Mock()
        self.stack.enter_context(patch.object(subject.time, "monotonic_ns", return_value=0))
        self.stack.enter_context(patch.object(subject.select, "select",
                                              return_value=([channel], [], [])))
        close = self.stack.enter_context(patch.object(subject.os, "close"))
        rights = array.array("i", [81, 82]).tobytes()
        for ancillary, flags in (([], socket.MSG_TRUNC), ([], socket.MSG_CTRUNC),
                                  ([(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights)], 0)):
            channel.recvmsg.return_value = (b"{}", ancillary, flags, None)
            with self.assertRaises(subject.GuardCompositionError):
                subject._receive(channel, 10_000_000_000)
        self.assertEqual([call.args for call in close.call_args_list], [(81,), (82,)])

    def test_receive_eof_not_json_and_timeout_no_recv(self):
        channel = Mock()
        self.stack.enter_context(patch.object(subject.time, "monotonic_ns", return_value=0))
        ready = self.stack.enter_context(patch.object(subject.select, "select",
                                                       return_value=([channel], [], [])))
        channel.recvmsg.return_value = (b"", [], 0, None)
        with self.assertRaisesRegex(subject.GuardCompositionError, "packet_rejected"):
            subject._receive(channel, 10)
        channel.recvmsg.reset_mock()
        ready.return_value = ([], [], [])
        with self.assertRaisesRegex(subject.GuardCompositionError, "packet_rejected"):
            subject._receive(channel, 10)
        channel.recvmsg.assert_not_called()

    def source_doubles(self, *, raw=b"x=1\n", close_error=False):
        selected = MagicMock()
        selected.__str__.return_value = "/release/scripts/run_sft_cgroup_fixture_v1.py"
        selected.parts = ("/", "release", "scripts", "run_sft_cgroup_fixture_v1.py")
        selected.parents = []
        selected.is_absolute.return_value = True
        info = types.SimpleNamespace(st_dev=1, st_ino=2, st_mode=stat.S_IFREG | 0o400,
                                     st_uid=0, st_gid=0, st_nlink=1, st_size=4,
                                     st_mtime_ns=3, st_ctime_ns=4)
        selected.lstat.return_value = info
        self.stack.enter_context(patch.object(subject, "Path", return_value=selected))
        opening = self.stack.enter_context(patch.object(subject.os, "open", return_value=80))
        reading = self.stack.enter_context(patch.object(subject.os, "read", return_value=raw))
        self.stack.enter_context(patch.object(subject.os, "fstat", return_value=info))
        closing = self.stack.enter_context(patch.object(subject.os, "close"))
        if close_error:
            closing.side_effect = OSError("private close failure")
        digest = Mock(hexdigest=lambda: subject.BASE_SHA256)
        self.stack.enter_context(patch.object(subject.hashlib, "sha256", return_value=digest))
        return selected, opening, reading, closing

    def test_source_bytes_bound_and_module_is_fresh(self):
        _, _, reading, closing = self.source_doubles()
        loaded = subject._load_base("/release/scripts/run_sft_cgroup_fixture_v1.py",
                                    subject.BASE_SHA256)
        self.assertEqual(loaded.x, 1)
        self.assertEqual(loaded.PACKAGE, loaded.__name__ + "_components")
        reading.assert_called_once_with(80, 4)
        closing.assert_called_once_with(80)

    def test_source_short_read_no_retry(self):
        _, _, reading, closing = self.source_doubles(raw=b"x=")
        with self.assertRaisesRegex(subject.GuardCompositionError, "source_rejected"):
            subject._load_base("/release/scripts/run_sft_cgroup_fixture_v1.py", subject.BASE_SHA256)
        reading.assert_called_once_with(80, 4)
        closing.assert_called_once_with(80)

    def test_source_symlink_and_wrong_hash_before_read(self):
        selected, opening, reading, _ = self.source_doubles()
        selected.lstat.return_value.st_mode = stat.S_IFLNK | 0o777
        with self.assertRaisesRegex(subject.GuardCompositionError, "source_rejected"):
            subject._load_base("/release/scripts/run_sft_cgroup_fixture_v1.py", subject.BASE_SHA256)
        opening.assert_not_called()
        reading.assert_not_called()
        with self.assertRaisesRegex(subject.GuardCompositionError, "source_rejected"):
            subject._load_base("/release/scripts/run_sft_cgroup_fixture_v1.py", "a" * 64)

    def test_source_close_failure_never_executes(self):
        _, _, _, closing = self.source_doubles(close_error=True)
        with patch("builtins.compile") as compiling, self.assertRaises(OSError):
            subject._load_base("/release/scripts/run_sft_cgroup_fixture_v1.py", subject.BASE_SHA256)
        compiling.assert_not_called()
        closing.assert_called_once_with(80)


if __name__ == "__main__":
    unittest.main()

