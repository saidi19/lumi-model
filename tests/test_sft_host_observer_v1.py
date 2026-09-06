"""Pure mocked self-observer boundaries and one isolated read-only CLI observation."""

from __future__ import annotations

import ctypes
import errno
import importlib
import importlib.util
import json
import os
import socket
import stat
import subprocess
import sys
import types
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import Mock, call, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_test_sft_host_observer_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
observer = importlib.import_module(PACKAGE + ".posttraining_sft_host_observer_v1")
CLI_PATH = ROOT / "scripts/observe_sft_host_containment_v1.py"
SPEC = importlib.util.spec_from_file_location("_test_sft_host_observer_cli_v1", CLI_PATH)
assert SPEC is not None and SPEC.loader is not None
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)
MOUNT = b"27 20 0:42 / /sys/fs/cgroup rw,nosuid,nodev - cgroup2 cgroup rw\n"
MEMBERSHIP = b"0::/synthetic-private-label\n"
NETWORK = b"\n".join(observer.snapshot.NET_DEV_HEADERS) + b"\n" + (
    b" lo: " + b" ".join([b"0"] * 16) + b"\n"
    b" synthetic0: " + b" ".join([b"123456"] * 16) + b"\n"
)


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.clock = self.enterContext(patch.object(observer.time, "monotonic", return_value=10))
        self.libc = Mock()
        self.cdll = self.enterContext(patch.object(observer.ctypes, "CDLL", return_value=self.libc))
        self.opened = self.enterContext(patch.object(
            observer.os, "open", side_effect=range(71, 300),
        ))
        self.closed = self.enterContext(patch.object(observer.os, "close"))
        self.read = self.enterContext(patch.object(observer.os, "read", return_value=b""))
        self.info = types.SimpleNamespace(st_mode=stat.S_IFREG | 0o444, st_dev=42,
                                         st_ino=9001, st_size=0)
        self.fstat = self.enterContext(patch.object(observer.os, "fstat", return_value=self.info))
        self.stat = self.enterContext(patch.object(observer.os, "stat", return_value=self.info))
        self.enterContext(patch.object(observer.os, "major", return_value=0))
        self.enterContext(patch.object(observer.os, "minor", return_value=42))
        self.ioctl = self.enterContext(patch.object(observer.fcntl, "ioctl"))
        self.magic = observer.CGROUP_MAGIC

        def filesystem(_fd, pointer):
            pointer._obj.kind = self.magic
            return 0

        self.libc.fstatfs.side_effect = filesystem
        self.reader = observer.Reader(30)
        self.addCleanup(self.reader.finish)

    def assert_code(self, code, operation):
        with self.assertRaises(observer.ObservationError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)

    def file(self, capacity=16, *, optional=False):
        return self.reader.read("memory.max", 51, capacity, observer.CGROUP_MAGIC,
                                device=(0, 42), optional=optional)

    def test_fstatfs_layout_and_signature_are_exact_typed_lp64(self):
        self.assertEqual(ctypes.sizeof(observer.StatFS), 120)
        self.assertEqual(observer.StatFS.fsid.offset, 56)
        self.assertEqual(observer.StatFS.spare.offset, 88)
        self.assertEqual(self.libc.fstatfs.argtypes,
                         [ctypes.c_int, ctypes.POINTER(observer.StatFS)])
        self.assertIs(self.libc.fstatfs.restype, ctypes.c_int)
        self.cdll.assert_called_once_with(None, use_errno=True)
        self.assertEqual(self.reader.filesystem(71, observer.CGROUP_MAGIC, (0, 42)), (42, 9001))

    def test_wrong_statfs_abi_denies_before_libc_or_kernel_io(self):
        self.cdll.reset_mock()
        with patch.object(observer.ctypes, "sizeof", return_value=119):
            self.assert_code("host_filesystem_rejected", lambda: observer.Reader(30))
        self.cdll.assert_not_called()
        self.opened.assert_not_called()

    def test_root_and_component_directory_opening_is_anchored_nofollow(self):
        self.info.st_mode = stat.S_IFDIR | 0o555
        self.magic = observer.PROC_MAGIC
        root = self.reader.directory("/proc", None, observer.PROC_MAGIC)
        self.reader.directory("12345", root, observer.PROC_MAGIC)
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_DIRECTORY
        self.assertEqual(self.opened.call_args_list,
                         [call("/proc", flags, dir_fd=None), call("12345", flags, dir_fd=71)])

    def test_arbitrary_roots_paths_dot_components_and_namespace_aliases_never_open(self):
        invalid = (
            ("/etc", {}), ("/proc/self/status", {}), ("/sys/fs", {}),
            ("..", {"parent": 71}), (".", {"parent": 71}), ("", {"parent": 71}),
            ("a/b", {"parent": 71}), ("x" * 256, {"parent": 71}),
            (b"net", {"parent": 71}), ("net", {"namespace": True}),
            ("user", {"parent": 71, "namespace": True}),
            ("net", {"parent": 71, "namespace": True, "directory": True}),
        )
        for name, options in invalid:
            with self.subTest(name=name):
                self.assert_code(
                    "host_configuration_rejected",
                    lambda name=name, options=options: self.reader.open(name, **options),
                )
        self.opened.assert_not_called()
        self.assertEqual(self.reader.opens, 0)

    def test_only_explicit_namespace_magiclinks_omit_nofollow_and_stay_pinned(self):
        self.magic = observer.NS_MAGIC
        for name, kind in observer.NAMESPACES.items():
            self.ioctl.return_value = kind
            self.assertEqual(self.reader.namespace(51, name), (42, 9001))
        for operation in self.opened.call_args_list:
            self.assertFalse(operation.args[1] & os.O_NOFOLLOW)
            self.assertEqual(operation.kwargs, {"dir_fd": 51})
        self.assertEqual(self.ioctl.call_args_list, [
            call(71, observer.NS_GET_NSTYPE), call(72, observer.NS_GET_NSTYPE),
            call(73, observer.NS_GET_NSTYPE),
        ])
        self.closed.assert_not_called()
        self.assertEqual(self.reader.pending, [71, 72, 73])

    def test_wrong_namespace_type_is_rejected_with_descriptor_owned_for_cleanup(self):
        self.magic = observer.NS_MAGIC
        self.ioctl.return_value = 0
        self.assert_code("host_namespace_rejected", lambda: self.reader.namespace(51, "net"))
        self.assertEqual(self.reader.pending, [71])
        self.assertTrue(self.reader.finish())
        self.closed.assert_called_once_with(71)

    def test_symlink_directory_socket_and_device_cannot_be_read_as_regular_metadata(self):
        for mode in (stat.S_IFLNK, stat.S_IFDIR, stat.S_IFSOCK, stat.S_IFCHR, stat.S_IFBLK):
            self.info.st_mode = mode
            with self.subTest(mode=mode):
                self.assert_code("host_filesystem_rejected",
                                 lambda: self.reader.open("file", parent=51))
                self.reader.finish()
        self.read.assert_not_called()
        self.assertEqual(self.closed.call_count, 5)

    def test_wrong_filesystem_magic_device_or_fstatfs_error_rejects(self):
        self.magic = observer.SYS_MAGIC
        self.assert_code("host_filesystem_rejected", self.file)
        self.magic = observer.CGROUP_MAGIC
        with patch.object(observer.os, "minor", return_value=43):
            self.assert_code("host_filesystem_rejected", self.file)
        self.libc.fstatfs.side_effect = None
        self.libc.fstatfs.return_value = -1
        self.assert_code("host_filesystem_rejected", self.file)
        self.read.assert_not_called()
        self.assertEqual(self.closed.call_count, 3)

    def test_zero_size_pseudofile_reads_all_short_chunks_through_eof(self):
        self.read.side_effect = [b"12", b"3", b"\n", b""]
        self.assertEqual(self.file(capacity=16), b"123\n")
        self.assertEqual(self.info.st_size, 0)
        self.assertEqual(self.read.call_args_list,
                         [call(71, 17), call(71, 15), call(71, 14), call(71, 13)])
        self.assertEqual((self.reader.files, self.reader.capacity, self.reader.read_calls),
                         (1, 17, 4))
        self.assertEqual(self.reader.bytes_read, 4)
        self.closed.assert_called_once_with(71)

    def test_exact_capacity_requires_one_more_read_to_verify_eof(self):
        self.read.side_effect = [b"1234", b""]
        self.assertEqual(self.file(capacity=4), b"1234")
        self.assertEqual(self.read.call_args_list, [call(71, 5), call(71, 1)])

    def test_short_read_cannot_hide_trailing_overcapacity_byte(self):
        self.read.side_effect = [b"1234", b"x"]
        self.assert_code("host_io_budget_exhausted", lambda: self.file(capacity=4))
        self.assertEqual(self.reader.bytes_read, 5)
        self.assertEqual((self.reader.files, self.reader.capacity), (1, 5))
        self.closed.assert_called_once_with(71)

    def test_read_syscalls_are_chunked_to_32768_bytes(self):
        self.read.side_effect = [b"x" * 32768, b"x" * 32768, b"x", b""]
        self.assertEqual(len(self.file(capacity=65537)), 65537)
        self.assertEqual([item.args[1] for item in self.read.call_args_list],
                         [32768, 32768, 2, 1])

    def test_global_128_open_reservations_never_reset(self):
        for _ in range(128):
            self.reader.open("file", parent=51)
        for _ in range(2):
            self.assert_code("host_io_budget_exhausted",
                             lambda: self.reader.open("file", parent=51))
        self.assertEqual(self.reader.opens, 128)
        self.assertEqual(self.opened.call_count, 128)

    def test_global_32_file_reservations_never_reset(self):
        for _ in range(32):
            self.assertEqual(self.file(capacity=1), b"")
        for _ in range(2):
            self.assert_code("host_io_budget_exhausted", lambda: self.file(capacity=1))
        self.assertEqual((self.reader.files, self.reader.capacity), (32, 64))
        self.assertEqual(self.opened.call_count, 32)

    def test_global_128_read_call_budget_is_not_reset_for_next_file(self):
        self.read.return_value = b"x"
        self.assert_code("host_io_budget_exhausted", lambda: self.file(capacity=256))
        self.assert_code("host_io_budget_exhausted", lambda: self.file(capacity=256))
        self.assertEqual(self.reader.read_calls, 128)
        self.assertEqual(self.read.call_count, 128)
        self.assertEqual((self.reader.files, self.reader.capacity), (2, 514))
        self.assertEqual(self.closed.call_count, 2)

    def test_global_capacity_reservation_includes_eof_sentinel_and_never_refunds(self):
        for _ in range(4):
            self.file(capacity=131071)
        self.assertEqual(self.reader.capacity, 524288)
        self.assert_code("host_io_budget_exhausted", lambda: self.file(capacity=1))
        self.assertEqual(self.reader.files, 4)
        self.assertEqual(self.opened.call_count, 4)

    def test_invalid_file_capacities_are_rejected_before_reservation_or_open(self):
        for capacity in (True, False, 0, -1, 1.5, "16", 131073):
            with self.subTest(capacity=capacity):
                self.assert_code("host_configuration_rejected",
                                 lambda capacity=capacity: self.file(capacity=capacity))
        self.assertEqual((self.reader.files, self.reader.capacity), (0, 0))
        self.opened.assert_not_called()

    def test_open_failure_keeps_all_prepaid_file_and_open_capacity(self):
        self.opened.side_effect = PermissionError(errno.EACCES, "SECRET")
        for _ in range(2):
            with self.assertRaises(PermissionError):
                self.file(capacity=4096)
        self.assertEqual((self.reader.opens, self.reader.files, self.reader.capacity), (2, 2, 8194))
        self.closed.assert_not_called()

    def test_only_enoent_makes_optional_file_missing(self):
        self.opened.side_effect = FileNotFoundError(errno.ENOENT, "SECRET")
        self.assertIsNone(self.file(optional=True))
        self.assertEqual(self.reader.missing, 1)
        with self.assertRaises(FileNotFoundError):
            self.file(optional=False)
        for error in (
            FileNotFoundError(errno.EIO, "SECRET"), PermissionError(errno.EACCES, "SECRET"),
            NotADirectoryError(errno.ENOTDIR, "SECRET"),
        ):
            self.opened.side_effect = error
            with self.subTest(errno=error.errno), self.assertRaises(OSError):
                self.file(optional=True)
        self.assertEqual(self.reader.missing, 1)
        self.assertEqual(self.reader.files, 5)

    def test_interface_probe_is_nofollow_and_only_enoent_is_absence(self):
        self.assertTrue(self.reader.interface_exists(51, "cgroup.kill", (0, 42)))
        self.stat.assert_called_once_with("cgroup.kill", dir_fd=51, follow_symlinks=False)
        self.stat.side_effect = FileNotFoundError(errno.ENOENT, "SECRET")
        self.assertFalse(self.reader.interface_exists(51, "cgroup.kill", (0, 42)))
        for error in (
            FileNotFoundError(errno.EIO, "SECRET"), PermissionError(errno.EACCES, "SECRET"),
        ):
            self.stat.side_effect = error
            with self.assertRaises(OSError):
                self.reader.interface_exists(51, "cgroup.kill", (0, 42))
        self.opened.assert_not_called()

    def test_interface_symlink_or_different_device_is_rejected(self):
        self.info.st_mode = stat.S_IFLNK
        self.assert_code("host_filesystem_rejected",
                         lambda: self.reader.interface_exists(51, "cgroup.kill", (0, 42)))
        self.info.st_mode = stat.S_IFREG
        with patch.object(observer.os, "minor", return_value=43):
            self.assert_code("host_filesystem_rejected",
                             lambda: self.reader.interface_exists(51, "cgroup.kill", (0, 42)))

    def test_close_failure_removes_ownership_once_and_is_sticky(self):
        fd = self.reader.open("file", parent=51)
        self.closed.side_effect = OSError(errno.EIO, "SECRET")
        self.assert_code("host_cleanup_failed", lambda: self.reader.close(fd))
        self.assertEqual(self.reader.pending, [])
        self.assertFalse(self.reader.finish())
        self.assertFalse(self.reader.finish())
        self.closed.assert_called_once_with(fd)
        self.assertEqual((self.reader.close_attempts, self.reader.closed), (1, 0))

    def test_cleanup_attempts_every_pending_descriptor_once_even_after_error(self):
        self.reader.open("first", parent=51)
        self.reader.open("second", parent=51)
        self.closed.side_effect = [OSError(errno.EIO, "SECRET"), None]
        self.assertFalse(self.reader.finish())
        self.assertEqual(self.closed.call_args_list, [call(72), call(71)])
        self.assertFalse(self.reader.finish())
        self.assertEqual(self.closed.call_count, 2)
        self.assertEqual(self.reader.pending, [])

    def test_pending_allocation_failure_closes_new_descriptor_immediately(self):
        class NoAppend(list):
            def append(self, _value):
                raise MemoryError("SECRET")

        self.reader.pending = NoAppend()
        self.assert_code("host_io_rejected", lambda: self.reader.open("file", parent=51))
        self.closed.assert_called_once_with(71)
        self.fstat.assert_not_called()
        self.assertEqual((self.reader.close_attempts, self.reader.closed), (1, 1))

    def test_initial_deadline_exhaustion_reserves_nothing(self):
        self.clock.return_value = 30
        self.assert_code("host_deadline_exhausted", self.file)
        self.assertEqual((self.reader.opens, self.reader.files, self.reader.capacity), (0, 0, 0))
        self.opened.assert_not_called()

    def test_deadline_after_read_keeps_capacity_and_closes_without_returning_partial_data(self):
        def late(_fd, _capacity):
            self.clock.return_value = 31
            return b"partial"

        self.read.side_effect = late
        self.assert_code("host_deadline_exhausted", self.file)
        self.assertEqual((self.reader.files, self.reader.capacity, self.reader.bytes_read),
                         (1, 17, 7))
        self.closed.assert_called_once_with(71)


class SyntheticReader:
    """Kernel-shaped metadata and descriptor identities, entirely in memory."""

    def __init__(self, *, optional=True, changed=None):
        self.changed = changed
        self.calls = []
        self.counts = Counter()
        self.finished = False
        self.clean = True
        self.next_fd = 50
        self.opens = self.capacity = self.files = self.bytes_read = self.missing = 0
        self.values = {
            "cgroup.controllers": b"memory pids io\n", "cgroup.subtree_control": b"memory pids\n",
            "cgroup.type": b"domain\n", "cgroup.events": b"populated 1\nfrozen 0\n",
            "memory.max": b"268435456\n", "memory.high": b"max\n", "memory.current": b"4096\n",
            "memory.swap.max": b"0\n", "memory.oom.group": b"1\n",
            "pids.max": b"64\n", "pids.current": b"2\n",
        } if optional else {}
        self.kill_exists = optional

    def directory(self, name, parent, magic, device=None):
        self.calls.append(("directory", name, parent, magic, device))
        self.opens += 1
        self.next_fd += 1
        return self.next_fd

    def walk(self, parent, parts, device):
        self.calls.append(("walk", tuple(parts), device))
        for part in parts:
            parent = self.directory(part, parent, observer.CGROUP_MAGIC, device)
        return parent

    def filesystem(self, fd, magic, device=None):
        self.calls.append(("filesystem", fd, magic, device))
        self.counts["inode"] += 1
        return (42, 9002 if self.changed == "inode" and self.counts["inode"] == 2 else 9001)

    def namespace(self, parent, name):
        self.calls.append(("namespace", parent, name))
        self.opens += 1
        self.counts[("namespace", name)] += 1
        changed = self.changed == name and self.counts[("namespace", name)] == 2
        return (88, observer.NAMESPACES[name] + int(changed))

    def read(self, name, parent, capacity, magic, *, device=None, optional=False):
        self.calls.append(("read", name, parent, capacity, magic, device, optional))
        self.counts[name] += 1
        self.opens += 1
        self.files += 1
        self.capacity += capacity + 1
        if name == "cgroup":
            raw = (b"0::/different-group\n" if self.changed == "membership"
                   and self.counts[name] == 2 else MEMBERSHIP)
        elif name == "mountinfo":
            raw = (MOUNT.replace(b"rw,nosuid", b"ro,nosuid") if self.changed == "mount"
                   and self.counts[name] == 2 else MOUNT)
        elif name == "dev":
            raw = NETWORK
        else:
            assert optional
            raw = self.values.get(name)
        self.bytes_read += len(raw) if raw is not None else 0
        self.missing += int(raw is None)
        return raw

    def interface_exists(self, parent, name, device):
        self.calls.append(("interface", parent, name, device))
        return self.kill_exists

    def check(self):
        return None

    def finish(self):
        self.finished = True
        return self.clean

    def summary(self):
        return {
            "open_attempts": self.opens, "file_capacity_reservations": self.files,
            "read_capacity_bytes": self.capacity, "read_calls": (self.files - self.missing) * 2,
            "metadata_bytes_read": self.bytes_read, "optional_files_missing": self.missing,
            "close_attempts": self.opens - self.missing if self.finished else 0,
            "descriptors_closed": self.opens - self.missing if self.finished and self.clean else 0,
            "pending_descriptors": 0 if self.finished else self.opens - self.missing,
            "cleanup_complete": self.clean,
        }


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(observer.os, "getpid", return_value=12345))

    def test_complete_optional_observation_is_aggregate_only_and_checks_two_snapshots(self):
        reader = SyntheticReader()
        value = observer.collect(reader)
        self.assertTrue(value["cgroup2_filesystem_observed"])
        self.assertTrue(value["membership_nonroot_in_visible_namespace"])
        self.assertTrue(value["memory_controller_available"])
        self.assertTrue(value["pids_controller_enabled_for_children"])
        self.assertFalse(value["mount_read_only_observed"])
        self.assertEqual(value["namespace_snapshots_checked"], 3)
        self.assertEqual(value["resource_snapshots"]["memory_max"],
                         {"present": True, "value": 268435456})
        self.assertEqual(value["resource_snapshots"]["memory_high"],
                         {"present": True, "value": None})
        self.assertEqual(value["network_snapshot"], {
            "interface_count": 2, "non_loopback_interfaces": 1, "loopback_interface_observed": True,
        })
        for name in ("cgroup", "mountinfo", "inode"):
            self.assertEqual(reader.counts[name], 2, name)
        for name in observer.NAMESPACES:
            self.assertEqual(reader.counts[("namespace", name)], 2, name)
        for operation in reader.calls:
            if operation[0] == "read" and operation[1] in reader.values:
                self.assertEqual(operation[3:], (4096, observer.CGROUP_MAGIC, (0, 42), True))
        encoded = json.dumps(value)
        for raw in ("synthetic-private-label", "synthetic0", "123456", "12345", "9001"):
            self.assertNotIn(raw, encoded)

    def test_absent_optional_features_do_not_become_verified_enforcement(self):
        value = observer.collect(SyntheticReader(optional=False))
        self.assertFalse(value["cgroup_kill_interface_observed"])
        self.assertFalse(value["memory_controller_available"])
        self.assertFalse(value["pids_controller_enabled_for_children"])
        self.assertIsNone(value["cgroup_populated"])
        self.assertIsNone(value["cgroup_frozen"])
        self.assertTrue(all(item == {"present": False, "value": None}
                            for item in value["resource_snapshots"].values()))

    def test_empty_leaf_subtree_control_is_present_but_enables_no_child_controller(self):
        reader = SyntheticReader()
        reader.values["cgroup.subtree_control"] = b""
        value = observer.collect(reader)
        self.assertTrue(value["memory_controller_available"])
        self.assertTrue(value["pids_controller_available"])
        self.assertFalse(value["memory_controller_enabled_for_children"])
        self.assertFalse(value["pids_controller_enabled_for_children"])
        self.assertEqual(reader.missing, 0)

    def test_membership_mount_each_namespace_and_group_inode_changes_deny(self):
        for changed in ("membership", "mount", "inode", *observer.NAMESPACES):
            with self.subTest(changed=changed):
                with self.assertRaises(observer.ObservationError) as caught:
                    observer.collect(SyntheticReader(changed=changed))
                self.assertEqual(caught.exception.code, "host_snapshot_changed")

    def test_invalid_optional_group_type_or_oom_boolean_is_not_silently_absent(self):
        for name, raw, code in (
            ("cgroup.type", b"SECRET\n", "host_io_rejected"),
            ("memory.oom.group", b"2\n", "host_scalar_rejected"),
        ):
            reader = SyntheticReader()
            reader.values[name] = raw
            with self.subTest(name=name), self.assertRaises(observer.ObservationError) as caught:
                observer.collect(reader)
            self.assertEqual(caught.exception.code, code)


class ObserveTests(unittest.TestCase):
    def boundaries(self, report):
        for key in observer.UNVERIFIED:
            self.assertIs(report[key], False, key)
        for key in (
            "host_mutations", "child_processes_started", "external_network_bytes",
            "model_or_dataset_reads", "private_artifact_writes", "gpu_use", "inference_calls",
        ):
            self.assertEqual(report[key], 0, key)
        encoded = json.dumps(report)
        for secret in ("SECRET", "Traceback", "synthetic-private-label", "synthetic0", "123456"):
            self.assertNotIn(secret, encoded)
        self.assertNotIn("receipt", report)

    def run_fake(self, reader, *, finish=None, error=None):
        with (
            patch.object(observer.runtime, "require_abi"),
            patch.object(observer.runtime, "install",
                         side_effect=AssertionError("mutation")) as install,
            patch.object(observer.threading, "active_count", return_value=1),
            patch.object(observer.threading, "current_thread", observer.threading.main_thread),
            patch.object(observer, "Reader", return_value=reader),
            patch.object(observer.time, "monotonic", return_value=10) as clock,
            patch.object(observer.os, "open", side_effect=AssertionError("kernel read")) as opened,
            patch.object(observer.os, "write", side_effect=AssertionError("host write")) as written,
            patch.object(subprocess, "Popen", side_effect=AssertionError("spawn")) as popen,
            patch.object(socket, "socket", side_effect=AssertionError("network")) as network,
        ):
            if finish is not None:
                reader.finish = lambda: finish(clock)
            if error is not None:
                reader.check = Mock(side_effect=error)
            report = observer.observe_self()
            for operation in (install, opened, written, popen, network):
                operation.assert_not_called()
        self.boundaries(report)
        return report

    def test_success_is_observed_metadata_never_permission_or_receipt(self):
        reader = SyntheticReader()
        report = self.run_fake(reader)
        self.assertEqual(report["state"], "observed")
        self.assertTrue(report["observation_complete"])
        self.assertTrue(reader.finished)
        self.assertEqual(report["accounting"]["pending_descriptors"], 0)
        self.assertGreater(len(report["required_next_evidence"]), 0)

    def test_final_cleanup_deadline_discards_otherwise_complete_observation(self):
        def late(clock):
            clock.return_value = 16
            return True

        report = self.run_fake(SyntheticReader(), finish=late)
        self.assertEqual(report["failure_code"], "host_deadline_exhausted")
        self.assertEqual(report["observations"], {})
        self.assertFalse(report["observation_complete"])

    def test_cleanup_failure_is_sticky_and_discards_observations(self):
        reader = SyntheticReader()
        reader.clean = False
        report = self.run_fake(reader, error=observer.ObservationError("host_snapshot_changed"))
        self.assertEqual(report["failure_code"], "host_cleanup_failed")
        self.assertEqual(report["observations"], {})
        self.assertFalse(report["accounting"]["cleanup_complete"])

    def test_snapshot_change_or_unknown_exception_discards_partial_observations(self):
        for reader, error, expected in (
            (SyntheticReader(changed="net"), None, "host_snapshot_changed"),
            (SyntheticReader(), RuntimeError("SECRET"), "host_observation_failed"),
        ):
            report = self.run_fake(reader, error=error)
            self.assertEqual(report["failure_code"], expected)
            self.assertEqual(report["observations"], {})
            self.assertTrue(reader.finished)

    def test_unsupported_mac_has_zero_kernel_reads_libc_loads_and_mutations(self):
        with (
            patch.object(observer.runtime.sys, "platform", "darwin"),
            patch.object(observer, "Reader") as reader,
            patch.object(observer, "collect") as collect,
            patch.object(observer.ctypes, "CDLL") as cdll,
            patch.object(observer.os, "open") as opened,
            patch.object(observer.os, "read") as read,
            patch.object(observer.os, "write") as write,
            patch.object(observer.runtime, "install") as install,
        ):
            report = observer.observe_self()
        self.boundaries(report)
        self.assertEqual(report["state"], "unsupported")
        self.assertFalse(report["observation_complete"])
        self.assertEqual(report["observations"], {})
        for operation in (reader, collect, cdll, opened, read, write, install):
            operation.assert_not_called()
        for key, value in report["accounting"].items():
            self.assertEqual(value, key == "cleanup_complete", key)

    def test_multithread_or_nonmain_thread_is_rejected_before_reader_construction(self):
        for count, current in ((2, observer.threading.main_thread()), (1, object())):
            with (
                patch.object(observer.runtime, "require_abi"),
                patch.object(observer.threading, "active_count", return_value=count),
                patch.object(observer.threading, "current_thread", return_value=current),
                patch.object(observer, "Reader") as reader,
            ):
                report = observer.observe_self()
            self.boundaries(report)
            self.assertEqual(report["failure_code"], "host_configuration_rejected")
            reader.assert_not_called()


class CLITests(unittest.TestCase):
    def invoke(self, *, args=None, no_site=1, bytecode=True, optimize=0, report=None, error=None,
               observation_error=None, written=None):
        if report is None:
            report = {"state": "observed", "observation_complete": True,
                      **dict.fromkeys(observer.UNVERIFIED, False)}
        observe = Mock(return_value=report, side_effect=observation_error)
        imported = Mock(return_value=types.SimpleNamespace(observe_self=observe))
        if error is not None:
            imported.side_effect = error
        write = Mock(side_effect=lambda _fd, raw: len(raw) if written is None else written)
        with (
            patch.object(cli, "sys", types.SimpleNamespace(
                argv=[str(CLI_PATH), *(args if args is not None else ["--self-only"])],
                flags=types.SimpleNamespace(no_site=no_site, optimize=optimize),
                dont_write_bytecode=bytecode, modules={},
            )),
            patch.object(cli, "importlib", types.SimpleNamespace(import_module=imported)),
            patch.object(cli, "os", types.SimpleNamespace(write=write)),
        ):
            result = cli.main()
        output = write.call_args.args[1] if write.called else b""
        return result, output, imported, write

    def test_only_explicit_self_only_stdlib_flags_can_import_observer(self):
        invalid = (
            {"args": []}, {"args": ["--help"]}, {"args": ["--self-only", "--pid", "123"]},
            {"args": ["--self-only", "/SECRET"]}, {"args": ["--path", "/proc"]},
            {"no_site": 0}, {"bytecode": False}, {"optimize": 1}, {"optimize": 2},
        )
        for options in invalid:
            with self.subTest(options=options):
                result, raw, imported, write = self.invoke(**options)
                self.assertEqual(result, 2)
                value = json.loads(raw)
                self.assertEqual(value["failure_code"], "require_stdlib_self_only")
                self.assertFalse(value["execution_authorized"])
                self.assertFalse(value["production_eligible"])
                imported.assert_not_called()
                write.assert_called_once()
                self.assertNotIn(b"SECRET", raw)

    def test_complete_observation_returns_zero_with_every_permission_false(self):
        result, raw, imported, write = self.invoke()
        self.assertEqual(result, 0)
        value = json.loads(raw)
        self.assertEqual(value["state"], "observed")
        for key in observer.UNVERIFIED:
            self.assertIs(value[key], False, key)
        imported.assert_called_once_with(
            "_lumi_self_containment_observation.posttraining_sft_host_observer_v1",
        )
        write.assert_called_once()
        self.assertEqual(write.call_args.args[0], 1)
        self.assertLessEqual(len(raw), 8192)
        self.assertTrue(raw.endswith(b"\n"))

    def test_import_error_is_fixed_raw_free_json_and_never_traceback(self):
        result, raw, _, write = self.invoke(error=RuntimeError("SECRET"))
        self.assertEqual(result, 2)
        self.assertEqual(json.loads(raw)["failure_code"], "host_observation_failed")
        self.assertNotIn(b"SECRET", raw)
        self.assertNotIn(b"Traceback", raw)
        write.assert_called_once()

    def test_observer_error_is_fixed_raw_free_json_and_never_traceback(self):
        result, raw, _, write = self.invoke(observation_error=RuntimeError("SECRET"))
        self.assertEqual(result, 2)
        self.assertEqual(json.loads(raw)["failure_code"], "host_observation_failed")
        self.assertNotIn(b"SECRET", raw)
        self.assertNotIn(b"Traceback", raw)
        write.assert_called_once()

    def test_oversized_or_nonserializable_report_is_not_written(self):
        for extra in ("x" * 8192, object()):
            result, raw, _, write = self.invoke(report={
                "observation_complete": True, "execution_authorized": False, "extra": extra,
            })
            self.assertEqual(result, 2)
            self.assertEqual(raw, b"")
            write.assert_not_called()

    def test_short_stdout_write_is_attempted_once_and_never_reports_success(self):
        result, _, _, write = self.invoke(written=1)
        self.assertEqual(result, 2)
        write.assert_called_once()

    def test_stdout_write_exception_is_not_retried_or_exposed(self):
        write = Mock(side_effect=BrokenPipeError("SECRET"))
        with patch.object(cli, "os", types.SimpleNamespace(write=write)):
            self.assertFalse(cli.emit({"execution_authorized": False}))
        write.assert_called_once()

    def test_false_or_nonboolean_completion_does_not_become_a_success_exit(self):
        for complete in (False, 1, "true", None):
            result, _, _, _ = self.invoke(report={
                "state": "observed", "observation_complete": complete,
                "execution_authorized": False,
            })
            self.assertEqual(result, 2)


class LiveSelfObservationTests(unittest.TestCase):
    def test_isolated_linux_observes_self_or_mac_denies_without_kernel_reads(self):
        # This is the only test that spawns: one stdlib-only self-observation CLI.
        # It performs no write test, confinement install, host mutation, or networking.
        result = subprocess.run(
            [sys.executable, "-S", "-B", str(CLI_PATH), "--self-only"],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=15,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
        self.assertEqual(result.stderr, b"")
        self.assertLessEqual(len(result.stdout), 8192)
        value = json.loads(result.stdout)
        for key in observer.UNVERIFIED:
            self.assertIs(value[key], False, key)
        self.assertNotIn("receipt", value)
        self.assertEqual(value["host_mutations"], 0)
        self.assertEqual(value["child_processes_started"], 0)
        self.assertEqual(value["external_network_bytes"], 0)
        if sys.platform == "linux":
            self.assertEqual(result.returncode, 0, value["failure_code"])
            self.assertEqual(value["state"], "observed")
            self.assertTrue(value["observation_complete"])
            self.assertEqual(value["accounting"]["pending_descriptors"], 0)
            self.assertTrue(value["accounting"]["cleanup_complete"])
        else:
            self.assertEqual(result.returncode, 2)
            self.assertEqual(value["state"], "unsupported")
            self.assertEqual(value["accounting"]["open_attempts"], 0)
            self.assertEqual(value["accounting"]["read_calls"], 0)


if __name__ == "__main__":
    unittest.main()
