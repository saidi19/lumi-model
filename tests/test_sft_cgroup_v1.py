"""Metadata/syscall doubles only; never touch live cgroups or require Linux/root."""

from __future__ import annotations

import ctypes
import fcntl
import importlib
import os
import stat
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_lumi_synthetic_cgroup_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
before_import = set(sys.modules)
cgroup = importlib.import_module(f"{PACKAGE}.posttraining_sft_cgroup_v1")
IMPORTED_MODULES = set(sys.modules) - before_import
NAME = "lumi-sft-" + "a" * 32
LIMITS = {"memory_max_bytes": 64 * 1024**2, "pids_max": 64,
          "cpu_quota_us": 100_000, "cpu_period_us": 100_000}


def info(inode, *, directory=False, mode=None, uid=0, device=17):
    default_mode = stat.S_IFDIR | 0o700 if directory else stat.S_IFREG | 0o644
    return types.SimpleNamespace(st_dev=device, st_ino=inode,
                                 st_mode=default_mode if mode is None else mode,
                                 st_uid=uid, st_gid=0, st_size=0)


class KernelMetadata:
    """In-memory kernel-interface double, never a process or filesystem fixture."""

    def __init__(self):
        self.parent_info, self.leaf_info = info(100, directory=True), info(200, directory=True)
        self.file_info = {}
        self.parent_fd, self.next_fd, self.pid, self.uid, self.thread = 10, 20, 1234, 0, 11
        self.fds = {10: {"kind": "parent", "flags": cgroup._DIRECTORY_FLAGS}}
        self.files = {
            "cgroup.type": b"domain\n", "cgroup.events": b"populated 0\nfrozen 0\n",
            "cgroup.procs": b"", "cgroup.kill": b"", "memory.max": b"max\n",
            "memory.swap.max": b"max\n", "memory.oom.group": b"0\n", "pids.max": b"max\n",
            "cpu.max": b"max 100000\n", "cpu.max.burst": b"0\n",
            "memory.events": b"low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n",
            "pids.events": b"max 0\n", "cpu.stat": b"usage_usec 0\nuser_usec 0\nsystem_usec 0\n"
            b"nr_periods 0\nnr_throttled 0\nthrottled_usec 0\n",
            "pids.current": b"0\n", "memory.current": b"0\n",
        }
        self.created, self.reads, self.writes, self.closed, self.removed, self.mkdir_calls = (
            False, [], [], [], [], [],
        )
        self.magic = cgroup.CGROUP_MAGIC
        self.read_chunk = self.short_write = self.fail_close = None
        self.post_attach = None

    def fstatfs(self, fd, pointer):
        if fd not in self.fds:
            raise OSError("SYNTHETIC-PRIVATE-FD")
        pointer._obj.kind = self.magic
        return 0

    def fstat(self, fd):
        opened = self.fds[fd]
        if opened["kind"] == "parent":
            return self.parent_info
        if opened["kind"] == "leaf":
            return self.leaf_info
        return self.stat(opened["name"], dir_fd=opened["parent"], follow_symlinks=False)

    def stat(self, name, *, dir_fd, follow_symlinks):
        if follow_symlinks is not False:
            raise AssertionError("symlinks_must_not_be_followed")
        if self.fds[dir_fd]["kind"] == "parent":
            if name != NAME or not self.created:
                raise FileNotFoundError("SYNTHETIC-PRIVATE")
            return self.leaf_info
        if name not in self.files:
            raise FileNotFoundError("SYNTHETIC-PRIVATE")
        if name not in self.file_info:
            self.file_info[name] = info(300 + len(self.file_info))
        return self.file_info[name]

    def mkdir(self, name, *, mode, dir_fd):
        self.mkdir_calls.append((name, mode, dir_fd))
        if self.created:
            raise FileExistsError("SYNTHETIC-PRIVATE")
        if name != NAME or dir_fd != self.parent_fd or mode != 0o700:
            raise AssertionError("unexpected_directory_target")
        self.created = True

    def open(self, name, flags, *, dir_fd):
        self.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        fd, self.next_fd = self.next_fd, self.next_fd + 1
        self.fds[fd] = {"kind": "leaf" if name == NAME else "file", "name": name,
                        "flags": flags, "parent": dir_fd, "position": 0}
        return fd

    def fcntl(self, fd, command):
        if command == fcntl.F_GETFL:
            return self.fds[fd]["flags"]
        if command == fcntl.F_GETFD:
            return fcntl.FD_CLOEXEC if self.fds[fd]["flags"] & os.O_CLOEXEC else 0
        raise AssertionError("unexpected_fcntl")

    def read(self, fd, amount):
        opened = self.fds[fd]
        self.reads.append((opened["name"], amount))
        amount = min(amount, self.read_chunk) if self.read_chunk is not None else amount
        start = opened["position"]
        data = self.files[opened["name"]][start:start + amount]
        opened["position"] += len(data)
        return data

    def write(self, fd, payload):
        name = self.fds[fd]["name"]
        self.writes.append((name, payload))
        if self.short_write == name:
            return len(payload) - 1
        if name == "cgroup.procs":
            self.files[name] = (self.post_attach if self.post_attach is not None
                                else f"{self.pid}\n".encode())
            self.files["cgroup.events"] = b"populated 1\nfrozen 0\n"
            self.files["pids.current"] = b"1\n"
        elif name == "cgroup.kill":
            self.files["cgroup.procs"] = b""
            self.files["cgroup.events"] = b"populated 0\nfrozen 0\n"
            self.files["pids.current"] = b"0\n"
        else:
            self.files[name] = payload
        return len(payload)

    def close(self, fd):
        self.closed.append(fd)
        del self.fds[fd]
        if self.fail_close == fd:
            raise OSError("SYNTHETIC-PRIVATE-CLOSE")

    def rmdir(self, name, *, dir_fd):
        if (name != NAME or dir_fd != self.parent_fd or not self.created
                or self.files["cgroup.procs"]):
            raise AssertionError("unowned_or_populated_remove")
        self.removed.append((name, dir_fd))
        self.created = False


class CgroupTests(unittest.TestCase):
    def setUp(self):
        self.kernel = KernelMetadata()
        kernel = self.kernel
        calls = {name: getattr(kernel, name) for name in
                 ("fstat", "stat", "mkdir", "open", "read", "write", "close", "rmdir")}
        self.fake_os = types.SimpleNamespace(
            **{name: getattr(os, name) for name in ("O_RDONLY", "O_WRONLY", "O_DIRECTORY",
                                                   "O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK",
                                                   "O_APPEND", "O_ACCMODE")},
            **calls, geteuid=lambda: kernel.uid, getpid=lambda: kernel.pid,
            uname=lambda: types.SimpleNamespace(machine="x86_64"),
        )
        self.enterContext(patch.object(cgroup, "os", self.fake_os))
        self.enterContext(patch.object(cgroup, "sys", types.SimpleNamespace(platform="linux",
                                                                           byteorder="little")))
        self.enterContext(patch.object(cgroup, "threading",
                                       types.SimpleNamespace(get_ident=lambda: kernel.thread)))
        self.enterContext(patch.object(cgroup, "fcntl", types.SimpleNamespace(
            fcntl=kernel.fcntl, F_GETFL=fcntl.F_GETFL, F_GETFD=fcntl.F_GETFD,
            FD_CLOEXEC=fcntl.FD_CLOEXEC)))
        def fstatfs(fd, pointer):
            return kernel.fstatfs(fd, pointer)
        self.enterContext(patch.object(cgroup, "ctypes", types.SimpleNamespace(
            CDLL=lambda *args, **kwargs: types.SimpleNamespace(fstatfs=fstatfs),
            **{name: getattr(ctypes, name) for name in ("c_int", "c_void_p", "POINTER",
                                                       "sizeof", "byref")},
        )))

    def lease(self, **changes):
        arguments = {"name": NAME, "limits": dict(LIMITS)}
        arguments.update(changes)
        return cgroup.CgroupLease(self.kernel.parent_fd, **arguments)

    def rejected(self, code, call, *args, **kwargs):
        with self.assertRaises(cgroup.CgroupError) as caught:
            call(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(caught.exception.args, (code,))
        self.assertNotIn("SYNTHETIC-PRIVATE", repr(caught.exception))

    def test_import_has_no_ml(self):
        forbidden = {"torch", "transformers", "numpy", "safetensors", "tokenizers"}
        self.assertFalse({name.split(".")[0] for name in IMPORTED_MODULES} & forbidden)

    def test_exclusive_six_control_setup_exact_readbacks_borrowed_parent(self):
        lease = self.lease()
        self.assertEqual(self.kernel.writes, list(cgroup._settings(LIMITS).items()))
        self.assertEqual(self.kernel.mkdir_calls, [(NAME, 0o700, 10)])
        self.assertEqual(repr(lease), "<CgroupLease private>")
        self.assertEqual(set(lease.snapshot()), {
            "populated", "pids_current", "pids_max_events", "memory_current_bytes",
            "memory_max_events", "memory_oom_events", "memory_oom_kill_events",
            "memory_oom_group_kill_events", "cpu_usage_usec", "cpu_nr_throttled",
            "cpu_throttled_usec",
        })
        self.assertTrue(all(value == 0 for value in lease.snapshot().values()))
        lease.verify_limits()
        borrowed = lease.fileno()
        self.assertIn(borrowed, self.kernel.fds)
        lease.close()
        lease.close()
        self.assertEqual(self.kernel.removed, [(NAME, 10)])
        self.assertEqual(set(self.kernel.fds), {10})
        self.assertNotIn(10, self.kernel.closed)

    def test_limits_are_copied(self):
        limits = dict(LIMITS)
        lease = self.lease(limits=limits)
        limits["memory_max_bytes"] *= 2
        lease.verify_limits()
        lease.close()

    def test_invalid_names_and_limits_never_create_or_write(self):
        for name in ("..", NAME + "\n", "/" + NAME, "lumi-sft-" + "A" * 32, True):
            self.rejected("invalid_arguments", self.lease, name=name)
        for limits in ({}, {**LIMITS, "extra": 1}, {**LIMITS, "pids_max": True},
                       {**LIMITS, "pids_max": 3}, {**LIMITS, "pids_max": 65_537},
                       {**LIMITS, "memory_max_bytes": 16 * 1024**2 - 1},
                       {**LIMITS, "memory_max_bytes": 1024**4 + 1},
                       {**LIMITS, "cpu_quota_us": 999},
                       {**LIMITS, "cpu_period_us": 1000, "cpu_quota_us": 128_001}):
            self.rejected("invalid_arguments", self.lease, limits=limits)
        self.assertEqual(self.kernel.mkdir_calls, [])
        self.assertEqual(self.kernel.writes, [])

    def test_unsupported_platform_and_nonroot_deny_before_kernel_descriptors(self):
        cgroup.sys.platform = "darwin"
        self.rejected("unsupported", self.lease)
        cgroup.sys.platform = "linux"
        self.kernel.uid = 1000
        self.rejected("unsupported", self.lease)
        self.assertEqual(self.kernel.mkdir_calls, [])
        self.assertEqual(self.kernel.writes, [])

    def test_wrong_filesystem_and_parent_permissions_deny_before_mkdir(self):
        self.kernel.magic = 1
        self.rejected("invalid_descriptor", self.lease)
        self.kernel.magic = cgroup.CGROUP_MAGIC
        self.kernel.parent_info.st_uid = 1000
        self.rejected("invalid_descriptor", self.lease)
        self.kernel.parent_info.st_uid = 0
        self.kernel.parent_info.st_mode |= 0o020
        self.rejected("invalid_descriptor", self.lease)
        self.assertEqual(self.kernel.mkdir_calls, [])

    def test_missing_required_descriptor_flags_denied(self):
        for flag in (os.O_DIRECTORY, os.O_NOFOLLOW, os.O_NONBLOCK, os.O_CLOEXEC):
            self.kernel.fds[10]["flags"] = cgroup._DIRECTORY_FLAGS & ~flag
            self.rejected("invalid_descriptor", self.lease)
        self.assertEqual(self.kernel.mkdir_calls, [])

    def test_existing_name_is_never_touched_or_removed(self):
        self.kernel.created = True
        self.rejected("already_exists", self.lease)
        self.assertEqual(self.kernel.writes, [])
        self.assertEqual(self.kernel.removed, [])
        self.assertEqual(set(self.kernel.fds), {10})

    def test_parent_alias_leaf_denied_without_kill_or_remove(self):
        self.kernel.leaf_info.st_ino = self.kernel.parent_info.st_ino
        self.rejected("descriptor_changed", self.lease)
        self.assertEqual(self.kernel.writes, [])
        self.assertEqual(self.kernel.removed, [])
        self.assertEqual(set(self.kernel.fds), {10})

    def test_control_symlink_or_group_writable_file_denied_before_writes(self):
        self.kernel.file_info["memory.max"] = info(900, mode=stat.S_IFLNK | 0o777)
        self.rejected("invalid_descriptor", self.lease)
        self.assertEqual(self.kernel.writes, [])
        self.assertEqual(self.kernel.removed, [(NAME, 10)])
        self.kernel.file_info["memory.max"] = info(900, mode=stat.S_IFREG | 0o664)
        self.rejected("invalid_descriptor", self.lease)
        self.assertEqual(self.kernel.writes, [])

    def test_missing_burst_denies_before_mutation_and_cleans_only_new_empty_leaf(self):
        del self.kernel.files["cpu.max.burst"]
        self.rejected("io_failed", self.lease)
        self.assertEqual(self.kernel.writes, [])
        self.assertEqual(self.kernel.removed, [(NAME, 10)])
        self.assertEqual(set(self.kernel.fds), {10})

    def test_missing_oom_group_event_denies_before_control_writes(self):
        self.kernel.files["memory.events"] = b"max 0\noom 0\noom_kill 0\n"
        self.rejected("snapshot_rejected", self.lease)
        self.assertEqual(self.kernel.writes, [])
        self.assertEqual(self.kernel.removed, [(NAME, 10)])

    def test_nonempty_new_leaf_retained_without_kill(self):
        self.kernel.files["cgroup.events"] = b"populated 1\nfrozen 0\n"
        self.rejected("not_empty", self.lease)
        self.assertEqual(self.kernel.writes, [])
        self.assertEqual(self.kernel.removed, [])
        self.assertEqual(set(self.kernel.fds), {10})

    def test_threaded_type_rejected(self):
        self.kernel.files["cgroup.type"] = b"threaded\n"
        self.rejected("configuration_rejected", self.lease)
        self.assertEqual(self.kernel.writes, [])

    def test_short_control_write_is_once_terminal_and_emptyleaf_cleanup_only(self):
        self.kernel.short_write = "memory.max"
        self.rejected("io_failed", self.lease)
        self.assertEqual(self.kernel.writes, [("memory.max", b"67108864\n")])
        self.assertEqual(self.kernel.removed, [(NAME, 10)])

    def test_readback_drift_poisoned_without_rewrite_kill_and_snapshot_still_allowed(self):
        lease = self.lease()
        cgroup.enter_current_process(lease.fileno())
        self.kernel.files["cpu.max.burst"] = b"1\n"
        self.rejected("configuration_rejected", lease.verify_limits)
        self.rejected("terminal_failure", lease.verify_limits)
        self.rejected("terminal_failure", lease.fileno)
        self.assertEqual(lease.snapshot()["populated"], 1)
        lease.kill()
        self.assertEqual(lease.snapshot()["populated"], 0)
        self.assertEqual(len(self.kernel.writes), 8)
        lease.close()

    def test_snapshot_exposes_only_fixed_aggregates_tolerates_other_bounded_numeric_keys(self):
        lease = self.lease()
        self.kernel.files["memory.events"] = (
            b"max 3\noom 2\noom_kill 1\noom_group_kill 1\nfuture 7\n")
        self.kernel.files["cpu.stat"] = (
            b"usage_usec 12\nnr_throttled 2\nthrottled_usec 9\nfuture 8\n")
        snapshot = lease.snapshot()
        self.assertEqual(snapshot["memory_max_events"], 3)
        self.assertEqual(snapshot["cpu_usage_usec"], 12)
        self.assertNotIn("future", snapshot)
        lease.close()

    def test_duplicate_noncanonical_missing_and_huge_event_inputs_reject(self):
        invalid = (b"max 0\nmax 1\n", b"max 00\n", b"max -1\n", b"max 1", b"max 1\n\n",
                   b"max 18446744073709551616\n", b"other 0\n", b"x" * 4097)
        for raw in invalid:
            self.rejected("snapshot_rejected", cgroup._events, raw, {"max"})

    def test_zero_size_pseudofile_reads_short_chunks_to_eof(self):
        self.kernel.read_chunk = 3
        lease = self.lease()
        lease.verify_limits()
        lease.close()
        self.assertTrue(all(0 < amount <= 4096 for _, amount in self.kernel.reads))

    def test_overcapacity_snapshot_poisoned_and_no_extra_read(self):
        lease = self.lease()
        self.kernel.files["memory.events"] = b"x" * 4097
        self.kernel.reads.clear()
        self.rejected("snapshot_rejected", lease.snapshot)
        self.assertEqual(self.kernel.reads, [("memory.events", 4096), ("memory.events", 1)])
        lease.close()

    def test_kill_exactly_once_repeated_call_poisoned_empty_close_allowed(self):
        lease = self.lease()
        cgroup.enter_current_process(lease.fileno())
        self.assertEqual(lease.snapshot()["populated"], 1)
        lease.kill()
        self.assertEqual(lease.snapshot()["populated"], 0)
        self.rejected("already_killed", lease.kill)
        self.assertEqual(lease.snapshot()["populated"], 0)
        self.rejected("terminal_failure", lease.verify_limits)
        self.assertEqual(self.kernel.writes[-1], ("cgroup.kill", b"1\n"))
        self.assertEqual(self.kernel.writes.count(("cgroup.kill", b"1\n")), 1)
        lease.close()

    def test_short_kill_is_never_retried(self):
        lease = self.lease()
        self.kernel.short_write = "cgroup.kill"
        self.rejected("io_failed", lease.kill)
        self.rejected("already_killed", lease.kill)
        self.assertEqual(self.kernel.writes.count(("cgroup.kill", b"1\n")), 1)
        lease.close()

    def test_killed_lease_cannot_verify_for_second_guardian_release(self):
        lease = self.lease()
        lease.kill()
        before = (len(self.kernel.reads), len(self.kernel.writes), len(self.kernel.closed))
        self.rejected("already_killed", lease.verify_limits)
        self.rejected("terminal_failure", lease.fileno)
        self.assertEqual((len(self.kernel.reads), len(self.kernel.writes),
                          len(self.kernel.closed)), before)
        self.assertEqual(lease.snapshot()["populated"], 0)
        lease.close()

    def test_killed_lease_cannot_issue_another_borrowed_descriptor(self):
        lease = self.lease()
        lease.kill()
        before = (len(self.kernel.reads), len(self.kernel.writes), len(self.kernel.closed))
        self.rejected("already_killed", lease.fileno)
        self.assertEqual((len(self.kernel.reads), len(self.kernel.writes),
                          len(self.kernel.closed)), before)
        lease.close()

    def test_close_refuses_populated_leaf_without_removing_or_closing_borrowed_fds(self):
        lease = self.lease()
        fd = lease.fileno()
        cgroup.enter_current_process(fd)
        self.rejected("not_empty", lease.close)
        self.assertIn(fd, self.kernel.fds)
        self.assertEqual(self.kernel.removed, [])
        self.assertEqual(self.kernel.writes[-1], ("cgroup.procs", b"0\n"))

    def test_renamed_or_replaced_leaf_denies_kill_and_remove(self):
        lease = self.lease()
        self.kernel.leaf_info.st_ino += 1
        self.rejected("descriptor_changed", lease.kill)
        self.rejected("descriptor_changed", lease.close)
        self.assertEqual(len(self.kernel.writes), 6)
        self.assertEqual(self.kernel.removed, [])

    def test_parent_fd_identity_change_denies_any_leaf_mutation(self):
        lease = self.lease()
        self.kernel.parent_info.st_ino += 1
        self.rejected("descriptor_changed", lease.kill)
        self.assertEqual(len(self.kernel.writes), 6)

    def test_owner_change_is_terminal(self):
        lease = self.lease()
        self.kernel.thread += 1
        self.rejected("owner_mismatch", lease.fileno)
        self.kernel.thread -= 1
        self.rejected("terminal_failure", lease.verify_limits)
        self.assertEqual(lease.snapshot()["populated"], 0)
        lease.close()

    def test_snapshot_error_preserves_identity_checked_kill_authority(self):
        lease = self.lease()
        cgroup.enter_current_process(lease.fileno())
        original = self.kernel.files["cpu.stat"]
        self.kernel.files["cpu.stat"] = b"SYNTHETIC-PRIVATE"
        self.rejected("snapshot_rejected", lease.snapshot)
        lease.kill()
        self.kernel.files["cpu.stat"] = original
        self.assertEqual(lease.snapshot()["populated"], 0)
        lease.close()

    def test_reinitializing_cannot_reset_kill_or_ownership(self):
        lease = self.lease()
        lease.kill()
        self.rejected("invalid_arguments", lease.__init__, 10, name=NAME, limits=LIMITS)
        self.rejected("already_killed", lease.kill)
        self.assertEqual(self.kernel.writes.count(("cgroup.kill", b"1\n")), 1)
        self.assertEqual(len(self.kernel.mkdir_calls), 1)
        lease.close()

    def test_close_failure_not_retried_and_never_closes_parent(self):
        lease = self.lease()
        fd = lease.fileno()
        self.kernel.fail_close = fd
        self.rejected("close_failed", lease.close)
        lease.close()
        self.assertEqual(self.kernel.closed.count(fd), 1)
        self.assertNotIn(10, self.kernel.closed)

    def test_attach_only_current_pid_one_write_and_borrowed_fd_stays_open(self):
        lease = self.lease()
        fd, before = lease.fileno(), len(self.kernel.writes)
        self.assertIsNone(cgroup.enter_current_process(fd))
        self.assertEqual(self.kernel.writes[before:], [("cgroup.procs", b"0\n")])
        self.assertIn(fd, self.kernel.fds)
        lease.kill()
        lease.close()

    def test_attach_missing_own_pid_rejected_without_retry(self):
        lease = self.lease()
        self.kernel.post_attach = b"999\n"
        self.rejected("configuration_rejected", cgroup.enter_current_process, lease.fileno())
        self.assertEqual(self.kernel.writes.count(("cgroup.procs", b"0\n")), 1)
        lease.kill()
        lease.close()

    def test_attach_short_write_rejected_and_fd_borrowed(self):
        lease = self.lease()
        fd = lease.fileno()
        self.kernel.short_write = "cgroup.procs"
        self.rejected("io_failed", cgroup.enter_current_process, fd)
        self.assertIn(fd, self.kernel.fds)
        self.assertEqual(self.kernel.writes.count(("cgroup.procs", b"0\n")), 1)
        lease.close()

    def test_procs_parser_is_canonical_bounded_and_no_pid_aliases(self):
        self.assertEqual(cgroup._pids(b""), set())
        self.assertEqual(cgroup._pids(b"1\n1234\n"), {1, 1234})
        for raw in (b"0\n", b"01\n", b"1\n1\n", b"1", b"-1\n", b"2147483648\n", b"x" * 65_537):
            self.rejected("snapshot_rejected", cgroup._pids, raw)


if __name__ == "__main__":
    unittest.main()
