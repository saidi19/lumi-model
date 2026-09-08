"""Guardian metadata/syscall-double tests only; never create leases or drop IDs.

Every cgroup, pidfd, poll, clock, libc and privilege mutation below is doubled.
These tests neither demonstrate Linux containment nor execute model/training code.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_synthetic_sft_cgroup_guardian_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
before_import = set(sys.modules)
guardian = importlib.import_module(PACKAGE + ".posttraining_sft_cgroup_guardian_v1")
IMPORT_ADDITIONS = set(sys.modules) - before_import


class GuardianWatchTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.now = 1_000_000_000
        self.lease = object.__new__(guardian.CgroupLease)
        self.runtime = self.enterContext(patch.object(guardian, "_runtime_check"))
        self.verify = self.enterContext(patch.object(
            guardian.CgroupLease, "verify_limits", side_effect=lambda: self.events.append("verify"),
        ))
        self.snapshots = [1, 0]
        self.killed = False
        self.snapshot = self.enterContext(patch.object(
            guardian.CgroupLease, "snapshot", side_effect=self.snapshot_value,
        ))
        self.kill = self.enterContext(patch.object(
            guardian.CgroupLease, "kill", side_effect=self.kill_value,
        ))
        self.lease_close = self.enterContext(patch.object(guardian.CgroupLease, "close"))
        self.ppid = self.enterContext(patch.object(guardian.os, "getppid", return_value=1234))
        self.pidfd = self.enterContext(patch.object(
            guardian.os, "pidfd_open", side_effect=self.pidfd_value, create=True,
        ))
        self.fcntl = self.enterContext(patch.object(
            guardian.fcntl, "fcntl", return_value=guardian.fcntl.FD_CLOEXEC,
        ))
        self.fstat = self.enterContext(patch.object(guardian.os, "fstat", return_value=(
            types.SimpleNamespace(st_dev=1, st_ino=2, st_mode=3)
        )))
        self.close = self.enterContext(patch.object(
            guardian.os, "close", side_effect=lambda fd: self.events.append(("close", fd)),
        ))
        self.alive = self.enterContext(patch.object(
            guardian.signal, "pidfd_send_signal", return_value=None, create=True,
        ))
        self.poller = Mock()
        self.poller.register.side_effect = lambda *args: self.events.append(("register", *args))
        self.poller.poll.side_effect = self.poll_value
        self.poll = self.enterContext(patch.object(
            guardian.select, "poll", return_value=self.poller,
        ))
        self.clock = self.enterContext(patch.object(
            guardian.time, "monotonic_ns", side_effect=lambda: self.now,
        ))
        self.sleep = self.enterContext(patch.object(guardian.time, "sleep", side_effect=self.slept))
        self.release = Mock(side_effect=lambda: self.events.append("release"))
        self.open_fd = self.enterContext(patch.object(
            guardian.os, "open", side_effect=AssertionError("SYNTHETIC-PRIVATE-REAL-OPEN"),
        ))
        self.numeric_kill = self.enterContext(patch.object(
            guardian.os, "kill", side_effect=AssertionError("SYNTHETIC-PRIVATE-NUMERIC-KILL"),
        ))

    def tearDown(self):
        self.open_fd.assert_not_called()
        self.numeric_kill.assert_not_called()
        self.lease_close.assert_not_called()
        self.assertFalse(guardian._WATCH_LOCK.locked())

    def pidfd_value(self, pid, flags):
        self.events.append(("pidfd", pid, flags))
        return 71

    def snapshot_value(self):
        value = 0 if self.killed else self.snapshots.pop(0) if self.snapshots else 1
        self.events.append(("snapshot", value))
        return {"populated": value}

    def kill_value(self):
        self.events.append("kill")
        self.killed = True

    def poll_value(self, milliseconds):
        self.events.append(("poll", milliseconds))
        self.now += milliseconds * 1_000_000
        return []

    def slept(self, seconds):
        self.events.append(("sleep", seconds))
        self.now += int(seconds * 1_000_000_000)

    def watch(self, **changes):
        options = {
            "expected_parent_pid": 1234, "deadline_ns": 1_200_000_000,
            "cleanup_timeout_ms": 100, "release": self.release,
        }
        options.update(changes)
        return guardian.watch_parent(self.lease, **options)

    def error(self, code, action=None, **changes):
        with self.assertRaises(guardian.GuardianError) as caught:
            (self.watch if action is None else action)(**changes)
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(caught.exception.args, (code,))
        self.assertTrue(caught.exception.__suppress_context__)
        self.assertNotIn("SYNTHETIC-PRIVATE", repr(caught.exception))

    def assert_report(self, report, reason, *, bound=True, released=True, closed=True):
        expected = {
            "schema_version": guardian.SCHEMA, "reason": reason, "parent_pidfd_bound": bound,
            "release_called": released, "kill_attempted": True, "empty_verified": True,
            "pidfd_closed": closed, **dict.fromkeys(guardian.NONCLAIMS, False),
        }
        self.assertEqual(report, expected)
        self.kill.assert_called_once_with()
        self.assertEqual(self.close.call_args_list, [call(71)] if closed else [])

    def test_import_loads_no_ml(self):
        self.assertFalse(any(name.split(".", 1)[0] in guardian.ML_ROOTS
                             for name in IMPORT_ADDITIONS))

    def test_natural_empty_is_retired_after_binding_and_release(self):
        self.assert_report(self.watch(), "workload_empty")
        self.assertEqual(self.events[:5], [
            "verify", ("snapshot", 1), ("pidfd", 1234, 0),
            ("register", 71, guardian.select.POLLIN | guardian.select.POLLHUP
             | guardian.select.POLLERR), "release",
        ])
        self.assertEqual(self.events[-3:], ["kill", ("snapshot", 0), ("close", 71)])
        self.alive.assert_has_calls([call(71, 0, None, 0), call(71, 0, None, 0)])

    def test_invalid_arguments_do_not_touch_lease_or_parent(self):
        invalid = {
            "expected_parent_pid": (True, 1, -1, 2**31, 1234.0),
            "deadline_ns": (True, 0, -1, 2**63, 1.0),
            "cleanup_timeout_ms": (True, 0, 10_001, 1.0), "release": (None, True),
        }
        for field, values in invalid.items():
            for value in values:
                self.error("invalid_arguments", **{field: value})
        self.verify.assert_not_called()
        self.pidfd.assert_not_called()
        self.kill.assert_not_called()
        self.close.assert_not_called()

    def test_exact_lease_type_required_without_constructing_a_live_lease(self):
        self.lease = object()
        self.error("invalid_arguments")

        class LeaseSubclass(guardian.CgroupLease):
            pass

        self.lease = object.__new__(LeaseSubclass)
        self.error("invalid_arguments")
        self.verify.assert_not_called()
        self.kill.assert_not_called()

    def test_runtime_denial_precedes_all_lease_and_parent_syscalls(self):
        self.runtime.side_effect = lambda: guardian._need(False, "runtime_rejected")
        self.error("runtime_rejected")
        self.verify.assert_not_called()
        self.pidfd.assert_not_called()
        self.kill.assert_not_called()

    def test_no_kill_authority_if_initial_lease_verification_fails(self):
        self.verify.side_effect = RuntimeError("SYNTHETIC-PRIVATE")
        self.error("guardian_failed")
        self.snapshot.assert_not_called()
        self.pidfd.assert_not_called()
        self.release.assert_not_called()
        self.kill.assert_not_called()

    def test_initial_empty_or_invalid_snapshot_denies_release_but_retires_once(self):
        for first in ({"populated": 0}, {"populated": True}, {"populated": 2}, {}):
            self.snapshot.side_effect = [first, {"populated": 0}]
            self.error("lease_rejected")
            self.release.assert_not_called()
            self.pidfd.assert_not_called()
            self.kill.assert_called_once_with()
            self.kill.reset_mock()

    def test_parent_lost_before_binding_never_opens_or_releases(self):
        self.ppid.return_value = 1
        self.assert_report(self.watch(), "parent_lost", bound=False, released=False, closed=False)
        self.pidfd.assert_not_called()
        self.release.assert_not_called()

    def test_parent_exits_during_pidfd_open_never_releases(self):
        self.pidfd.side_effect = ProcessLookupError("SYNTHETIC-PRIVATE")
        self.assert_report(self.watch(), "parent_lost", bound=False, released=False, closed=False)
        self.release.assert_not_called()

    def test_parent_changes_between_pidfd_open_and_second_parent_check(self):
        self.ppid.side_effect = [1234, 1]
        self.assert_report(self.watch(), "parent_lost", bound=False, released=False)
        self.release.assert_not_called()

    def test_dead_parent_pidfd_before_arm_never_releases(self):
        self.alive.side_effect = ProcessLookupError("SYNTHETIC-PRIVATE")
        self.assert_report(self.watch(), "parent_lost", bound=False, released=False)
        self.release.assert_not_called()

    def test_parent_death_takes_priority_over_empty_after_release(self):
        self.ppid.side_effect = [1234, 1234, 1]
        self.assert_report(self.watch(), "parent_lost")
        self.assertEqual(self.snapshot.call_count, 2)  # Initial and post-kill only.

    def test_expired_deadline_never_binds_or_releases(self):
        self.assert_report(self.watch(deadline_ns=self.now), "deadline_exhausted",
                           bound=False, released=False, closed=False)
        self.pidfd.assert_not_called()
        self.release.assert_not_called()

    def test_deadline_expiring_after_arming_precedes_release(self):
        self.clock.side_effect = [self.now, 1_200_000_000, 1_200_000_000]
        self.assert_report(self.watch(), "deadline_exhausted", released=False)
        self.release.assert_not_called()

    def test_deadline_expiring_in_release_precedes_apparently_empty_workload(self):
        def released():
            self.events.append("release")
            self.now = 1_200_000_000

        self.release.side_effect = released
        self.assert_report(self.watch(), "deadline_exhausted")
        self.assertEqual(self.snapshot.call_count, 2)

    def test_all_waits_are_bounded_and_deadline_is_not_reset(self):
        self.snapshots = [1]
        self.assert_report(self.watch(), "deadline_exhausted")
        self.assertEqual(self.poller.poll.call_count, 4)
        self.assertTrue(all(0 <= args.args[0] <= 50 for args in self.poller.poll.call_args_list))
        self.assertEqual(self.now, 1_200_000_000)

    def test_poll_parent_event_kills_once_and_closes_pidfd_once(self):
        self.snapshots = [1, 1]
        self.poller.poll.side_effect = None
        self.poller.poll.return_value = [(71, guardian.select.POLLIN)]
        self.assert_report(self.watch(), "parent_lost")

    def test_wrong_or_invalid_poll_descriptor_fails_closed(self):
        self.snapshots = [1, 1]
        self.poller.poll.side_effect = None
        self.poller.poll.return_value = [(72, guardian.select.POLLIN)]
        self.error("pidfd_rejected")
        self.kill.assert_called_once_with()
        self.close.assert_called_once_with(71)

    def test_release_exception_still_retires_and_closes_once(self):
        self.release.side_effect = RuntimeError("SYNTHETIC-PRIVATE")
        self.error("guardian_failed")
        self.release.assert_called_once_with()
        self.kill.assert_called_once_with()
        self.close.assert_called_once_with(71)

    def test_non_none_release_is_denied_and_cleanup_is_still_mandatory(self):
        self.release.side_effect = None
        self.release.return_value = False
        self.error("release_rejected")
        self.kill.assert_called_once_with()
        self.close.assert_called_once_with(71)

    def test_failed_kill_is_never_retried_or_reported_as_success(self):
        self.kill.side_effect = OSError("SYNTHETIC-PRIVATE")
        self.error("cleanup_failed")
        self.kill.assert_called_once_with()
        self.close.assert_called_once_with(71)

    def test_nonempty_cleanup_exhausts_its_separate_bounded_deadline(self):
        self.snapshot.side_effect = [{"populated": 1}, {"populated": 0}] + [
            {"populated": 1} for _ in range(4)
        ]
        self.error("cleanup_failed", cleanup_timeout_ms=20)
        self.kill.assert_called_once_with()
        self.assertEqual(self.sleep.call_args_list, [call(0.01), call(0.01)])
        self.assertEqual(self.now, 1_020_000_000)
        self.close.assert_called_once_with(71)

    def test_empty_observation_failure_is_not_silent_cleanup_success(self):
        self.snapshot.side_effect = [{"populated": 1}, {"populated": 0},
                                    OSError("SYNTHETIC-PRIVATE")]
        self.error("cleanup_failed")
        self.kill.assert_called_once_with()
        self.close.assert_called_once_with(71)

    def test_ambiguous_close_is_attempted_once_without_retry(self):
        self.close.side_effect = OSError("SYNTHETIC-PRIVATE")
        self.error("cleanup_failed")
        self.close.assert_called_once_with(71)
        self.kill.assert_called_once_with()

    def test_changed_pidfd_identity_is_never_closed_as_owned(self):
        self.fstat.side_effect = [types.SimpleNamespace(st_dev=1, st_ino=2, st_mode=3),
                                 types.SimpleNamespace(st_dev=1, st_ino=99, st_mode=3),
                                 types.SimpleNamespace(st_dev=1, st_ino=99, st_mode=3)]
        self.error("cleanup_failed")
        self.kill.assert_called_once_with()
        self.close.assert_not_called()

    def test_initial_pidfd_cloexec_rejection_still_closes_fresh_owned_fd_once(self):
        self.fcntl.return_value = 0
        self.error("pidfd_rejected")
        self.release.assert_not_called()
        self.kill.assert_called_once_with()
        self.close.assert_called_once_with(71)

    def test_initial_pidfd_fstat_failure_closes_once_without_rebinding(self):
        self.fstat.side_effect = OSError("SYNTHETIC-PRIVATE")
        self.error("guardian_failed")
        self.release.assert_not_called()
        self.pidfd.assert_called_once_with(1234, 0)
        self.kill.assert_called_once_with()
        self.close.assert_called_once_with(71)

    def test_failed_initial_pidfd_cleanup_takes_precedence_without_close_retry(self):
        self.fcntl.return_value = 0
        self.close.side_effect = OSError("SYNTHETIC-PRIVATE")
        self.error("cleanup_failed")
        self.close.assert_called_once_with(71)

    def test_watch_lock_rejects_nested_watch_before_second_lease_or_release(self):
        def release():
            self.error("reentry")

        self.release.side_effect = release
        self.assert_report(self.watch(), "workload_empty")
        self.verify.assert_called_once_with()
        self.release.assert_called_once_with()
        self.pidfd.assert_called_once_with(1234, 0)

    def test_unknown_error_text_is_fixed_and_private_free(self):
        for value in ("SYNTHETIC-PRIVATE", None, True, {}):
            error = guardian.GuardianError(value)
            self.assertEqual(error.args, ("guardian_failed",))
            self.assertNotIn("PRIVATE", repr(error))


class RuntimeGuardTests(unittest.TestCase):
    """Only Python/ABI/root metadata doubles; no privileged syscalls are invoked."""

    def setUp(self):
        self.metadata = types.SimpleNamespace(
            platform="linux", implementation=types.SimpleNamespace(name="cpython"),
            flags=types.SimpleNamespace(no_site=1), dont_write_bytecode=True, modules={},
        )
        self.enterContext(patch.object(guardian, "sys", self.metadata))
        self.machine = self.enterContext(patch.object(
            guardian.platform, "machine", return_value="x86_64",
        ))
        self.uid = self.enterContext(patch.object(guardian.os, "geteuid", return_value=0))
        self.gid = self.enterContext(patch.object(guardian.os, "getegid", return_value=0))
        self.main = object()
        self.enterContext(patch.object(guardian.threading, "main_thread", return_value=self.main))
        self.current = self.enterContext(patch.object(
            guardian.threading, "current_thread", return_value=self.main,
        ))
        self.count = self.enterContext(patch.object(
            guardian.threading, "active_count", return_value=1,
        ))
        self.pidfd = self.enterContext(patch.object(guardian.os, "pidfd_open", create=True))
        self.signal = self.enterContext(patch.object(
            guardian.signal, "pidfd_send_signal", create=True,
        ))
        self.poll = self.enterContext(patch.object(guardian.select, "poll"))

    def tearDown(self):
        self.pidfd.assert_not_called()
        self.signal.assert_not_called()
        self.poll.assert_not_called()

    def rejected(self):
        with self.assertRaises(guardian.GuardianError) as caught:
            guardian._runtime_check()
        self.assertEqual(caught.exception.args, ("runtime_rejected",))

    def test_complete_runtime_metadata_contract_is_inert(self):
        self.assertIsNone(guardian._runtime_check())

    def test_linux_x86_64_cpython_only(self):
        self.metadata.platform = "darwin"
        self.rejected()
        self.metadata.platform = "linux"
        self.machine.return_value = "aarch64"
        self.rejected()
        self.machine.return_value = "x86_64"
        self.metadata.implementation.name = "pypy"
        self.rejected()

    def test_no_site_and_no_bytecode_are_both_required(self):
        self.metadata.flags.no_site = 0
        self.rejected()
        self.metadata.flags.no_site = 1
        self.metadata.dont_write_bytecode = False
        self.rejected()

    def test_root_effective_uid_and_gid_are_both_required(self):
        self.uid.return_value = 1001
        self.rejected()
        self.uid.return_value = 0
        self.gid.return_value = 1002
        self.rejected()

    def test_single_python_main_thread_is_required(self):
        self.current.return_value = object()
        self.rejected()
        self.current.return_value = self.main
        self.count.return_value = 2
        self.rejected()

    def test_any_ml_root_or_descendant_denies_even_when_import_value_is_none(self):
        for root in guardian.ML_ROOTS:
            for name in (root, root + ".synthetic"):
                self.metadata.modules = {name: None}
                self.rejected()


class PrivilegeDropTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.uid, self.gid = (0, 0, 0), (0, 0, 0)
        self.runtime = self.enterContext(patch.object(guardian, "_runtime_check"))
        self.getuid = self.enterContext(patch.object(
            guardian.os, "getresuid", side_effect=lambda: self.uid, create=True,
        ))
        self.getgid = self.enterContext(patch.object(
            guardian.os, "getresgid", side_effect=lambda: self.gid, create=True,
        ))
        self.groups = self.enterContext(patch.object(guardian.os, "getgroups", return_value=[]))
        self.setgroups = self.enterContext(patch.object(
            guardian.os, "setgroups",
            side_effect=lambda value: self.events.append(("groups", value)),
        ))
        self.setgid = self.enterContext(patch.object(
            guardian.os, "setresgid", side_effect=self.gid_set, create=True,
        ))
        self.setuid = self.enterContext(patch.object(
            guardian.os, "setresuid", side_effect=self.uid_set, create=True,
        ))
        self.prctl = Mock(side_effect=self.prctl_value)
        self.libc = self.enterContext(patch.object(
            guardian.ctypes, "CDLL", return_value=types.SimpleNamespace(prctl=self.prctl),
        ))
        self.status = self.enterContext(patch.object(
            guardian, "_verify_dropped_status",
            side_effect=lambda uid, gid: self.events.append(("status", uid, gid)),
        ))

    def gid_set(self, *values):
        self.events.append(("gid", *values))
        self.gid = values

    def uid_set(self, *values):
        self.events.append(("uid", *values))
        self.uid = values

    def prctl_value(self, option, *values):
        self.events.append(("prctl", option, *values))
        return 1 if option == 39 else 0

    def error(self, code="privilege_drop_rejected", uid=1001, gid=1002):
        with self.assertRaises(guardian.GuardianError) as caught:
            guardian.drop_worker_privileges(uid, gid)
        self.assertEqual(caught.exception.args, (code,))
        self.assertTrue(caught.exception.__suppress_context__)
        self.assertNotIn("SYNTHETIC-PRIVATE", repr(caught.exception))

    def test_drop_order_clears_groups_gid_uid_and_verifies_no_new_privileges(self):
        self.assertIsNone(guardian.drop_worker_privileges(1001, 1002))
        self.assertEqual(self.events, [
            ("prctl", 7, 0, 0, 0, 0), ("prctl", 38, 1, 0, 0, 0), ("groups", []),
            ("gid", 1002, 1002, 1002), ("uid", 1001, 1001, 1001),
            ("prctl", 39, 0, 0, 0, 0), ("status", 1001, 1002),
        ])
        self.assertEqual(self.uid, (1001, 1001, 1001))
        self.assertEqual(self.gid, (1002, 1002, 1002))
        self.libc.assert_called_once_with(None, use_errno=True)
        self.assertEqual(self.prctl.argtypes,
                         [guardian.ctypes.c_int] + [guardian.ctypes.c_ulong] * 4)
        self.status.assert_called_once_with(1001, 1002)

    def test_non_root_or_boolean_ids_are_denied_before_any_mutation(self):
        for value in (True, 0, -1, 2**31, 1.0, "1001"):
            self.error("invalid_arguments", uid=value)
            self.error("invalid_arguments", gid=value)
        self.libc.assert_not_called()
        self.setgroups.assert_not_called()
        self.setgid.assert_not_called()
        self.setuid.assert_not_called()

    def test_runtime_rejection_precedes_identity_or_libc_calls(self):
        self.runtime.side_effect = lambda: guardian._need(False, "runtime_rejected")
        self.error("runtime_rejected")
        self.getuid.assert_not_called()
        self.libc.assert_not_called()
        self.setgroups.assert_not_called()

    def test_all_real_effective_and_saved_initial_ids_must_be_root(self):
        self.uid = (0, 0, 1001)
        self.error()
        self.libc.assert_not_called()
        self.setgroups.assert_not_called()

    def test_keepcaps_enabled_denies_before_new_privileges_or_id_mutations(self):
        self.prctl.side_effect = None
        self.prctl.return_value = 1
        self.error()
        self.assertEqual(self.prctl.call_args_list, [call(7, 0, 0, 0, 0)])
        self.setgroups.assert_not_called()
        self.setgid.assert_not_called()
        self.setuid.assert_not_called()

    def test_no_new_privileges_set_failure_denies_before_group_and_id_mutation(self):
        self.prctl.side_effect = [0, -1]
        self.error()
        self.setgroups.assert_not_called()
        self.setgid.assert_not_called()
        self.setuid.assert_not_called()

    def test_failed_setgroups_is_not_retried_and_prevents_later_id_mutation(self):
        self.setgroups.side_effect = OSError("SYNTHETIC-PRIVATE")
        self.error()
        self.setgroups.assert_called_once_with([])
        self.setgid.assert_not_called()
        self.setuid.assert_not_called()

    def test_failed_gid_drop_prevents_uid_drop(self):
        self.setgid.side_effect = OSError("SYNTHETIC-PRIVATE")
        self.error()
        self.setgid.assert_called_once_with(1002, 1002, 1002)
        self.setuid.assert_not_called()

    def test_retained_saved_uid_or_gid_rejects_after_mutation(self):
        self.setuid.side_effect = lambda *_args: None
        self.error()
        self.setuid.assert_called_once_with(1001, 1001, 1001)

    def test_remaining_supplementary_groups_reject_after_mutation(self):
        self.groups.return_value = [1002]
        self.error()
        self.setuid.assert_called_once_with(1001, 1001, 1001)

    def test_no_new_privileges_readback_must_be_one(self):
        self.prctl.side_effect = [0, 0, 0]
        self.error()
        self.assertEqual(self.prctl.call_args_list[-1], call(39, 0, 0, 0, 0))
        self.status.assert_not_called()

    def test_status_verification_failure_is_converted_to_fixed_public_drop_error(self):
        self.status.side_effect = OSError("SYNTHETIC-PRIVATE")
        self.error()
        self.status.assert_called_once_with(1001, 1002)
        self.setuid.assert_called_once_with(1001, 1001, 1001)


class DroppedStatusTests(unittest.TestCase):
    """Bounded synthetic procfs bytes with no actual procfs or libc operations."""

    def setUp(self):
        fields = {
            "Pid": "2468", "Tgid": "2468", "PPid": "1234", "TracerPid": "0", "Threads": "1",
            "Uid": "1001\t1001\t1001\t1001", "Gid": "1002\t1002\t1002\t1002",
            "NSpid": "2468", "NStgid": "2468", "CapInh": "0000000000000000",
            "CapPrm": "0000000000000000", "CapEff": "0000000000000000",
            "CapAmb": "0000000000000000", "CapBnd": "ffffffffffffffff", "NoNewPrivs": "1",
        }
        self.raw = "".join(f"{key}:\t{value}\n" for key, value in fields.items()).encode()
        self.remaining = self.raw
        self.magic, self.statfs_result = guardian.PROC_MAGIC, 0
        self.open = self.enterContext(patch.object(guardian.os, "open", return_value=91))
        self.read = self.enterContext(patch.object(
            guardian.os, "read", side_effect=self.read_value,
        ))
        self.close = self.enterContext(patch.object(guardian.os, "close"))
        self.pid = self.enterContext(patch.object(guardian.os, "getpid", return_value=2468))
        self.ppid = self.enterContext(patch.object(guardian.os, "getppid", return_value=1234))
        self.fstatfs = Mock(side_effect=self.fstatfs_value)
        self.libc = self.enterContext(patch.object(
            guardian.ctypes, "CDLL", return_value=types.SimpleNamespace(fstatfs=self.fstatfs),
        ))
        self.parse = self.enterContext(patch.object(
            guardian, "parse_status", wraps=guardian.parse_status,
        ))

    def read_value(self, fd, size):
        self.assertEqual(fd, 91)
        self.assertGreater(size, 0)
        self.assertLessEqual(size, 4096)
        chunk, self.remaining = self.remaining[:size], self.remaining[size:]
        return chunk

    def fstatfs_value(self, fd, address):
        self.assertEqual(fd, 91)
        metadata = guardian.ctypes.cast(address, guardian.ctypes.POINTER(guardian.StatFS)).contents
        metadata.kind = self.magic
        return self.statfs_result

    def verify(self):
        return guardian._verify_dropped_status(1001, 1002)

    def rejected(self):
        with self.assertRaises(guardian.GuardianError) as caught:
            self.verify()
        self.assertEqual(caught.exception.args, ("privilege_drop_rejected",))
        self.close.assert_called_once_with(91)

    def test_exact_procfs_flags_parse_identity_and_single_close(self):
        self.assertIsNone(self.verify())
        self.open.assert_called_once_with(
            "/proc/self/status",
            guardian.os.O_RDONLY | guardian.os.O_CLOEXEC | guardian.os.O_NOFOLLOW,
        )
        self.parse.assert_called_once_with(
            self.raw, expected_pid=2468, expected_ppid=1234, expected_uid=1001, expected_gid=1002,
        )
        self.assertEqual(self.ppid.call_count, 2)
        self.close.assert_called_once_with(91)
        self.libc.assert_called_once_with(None, use_errno=True)

    def test_wrong_filesystem_denies_before_read_or_parse_and_closes_once(self):
        self.magic = 0
        self.rejected()
        self.read.assert_not_called()
        self.parse.assert_not_called()

    def test_failed_fstatfs_denies_before_read_or_parse_and_closes_once(self):
        self.statfs_result = -1
        self.rejected()
        self.read.assert_not_called()
        self.parse.assert_not_called()

    def test_exact_maximum_status_requests_only_one_overflow_byte_after_limit(self):
        prefix = self.raw + b"Padding:\t"
        self.remaining = prefix + b"x" * (
            guardian.MAXIMUM_STATUS_BYTES - len(prefix) - 1
        ) + b"\n"
        self.assertEqual(len(self.remaining), guardian.MAXIMUM_STATUS_BYTES)
        self.verify()
        self.assertEqual([entry.args[1] for entry in self.read.call_args_list],
                         [4096, 4096, 4096, 4096, 1])
        self.close.assert_called_once_with(91)

    def test_oversized_status_denies_without_parse_after_at_most_limit_plus_one(self):
        self.remaining = b"x" * (guardian.MAXIMUM_STATUS_BYTES + 1)
        self.rejected()
        self.assertEqual(sum(entry.args[1] for entry in self.read.call_args_list),
                         guardian.MAXIMUM_STATUS_BYTES + 1)
        self.parse.assert_not_called()

    def test_short_read_iterations_are_bounded_without_unlimited_read_loop(self):
        self.read.side_effect = None
        self.read.return_value = b"x"
        self.rejected()
        self.assertEqual(self.read.call_count, 128)
        self.parse.assert_not_called()

    def test_parent_identity_change_rejects_after_parse_and_closes_once(self):
        self.ppid.side_effect = [1234, 4321]
        self.rejected()
        self.parse.assert_called_once()

    def test_frozen_parser_rejects_retained_capability_and_closes_once(self):
        self.remaining = self.raw.replace(b"CapEff:\t0000000000000000",
                                          b"CapEff:\t0000000000000001")
        with self.assertRaises(ValueError) as caught:
            self.verify()
        self.assertEqual(caught.exception.args, ("child_status_capability_rejected",))
        self.close.assert_called_once_with(91)

    def test_parse_failure_closes_once_without_retry(self):
        self.parse.side_effect = ValueError("SYNTHETIC-PRIVATE")
        with self.assertRaises(ValueError):
            self.verify()
        self.parse.assert_called_once()
        self.close.assert_called_once_with(91)

    def test_read_failure_closes_once_without_retry(self):
        self.read.side_effect = OSError("SYNTHETIC-PRIVATE")
        with self.assertRaises(OSError):
            self.verify()
        self.read.assert_called_once()
        self.parse.assert_not_called()
        self.close.assert_called_once_with(91)

    def test_open_failure_has_no_descriptor_close_or_read(self):
        self.open.side_effect = OSError("SYNTHETIC-PRIVATE")
        with self.assertRaises(OSError):
            self.verify()
        self.read.assert_not_called()
        self.close.assert_not_called()

    def test_ambiguous_close_is_not_retried(self):
        self.close.side_effect = OSError("SYNTHETIC-PRIVATE")
        with self.assertRaises(OSError):
            self.verify()
        self.close.assert_called_once_with(91)


if __name__ == "__main__":
    unittest.main()
