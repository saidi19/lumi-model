"""Pure syscall doubles only: no live socket, namespace, procfs or libc calls.

These tests do not establish Linux containment or permit a real training run.
All runtime-facing module surfaces are replaced before any entrypoint executes.
"""

from __future__ import annotations

import ctypes
import fcntl
import importlib
import os
import socket
import stat
import struct
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_lumi_synthetic_netns_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
before_import = set(sys.modules)
netns = importlib.import_module(PACKAGE + ".posttraining_sft_netns_v1")
IMPORT_ADDITIONS = set(sys.modules) - before_import
PRIVATE = "SYNTHETIC-PRIVATE-EXCEPTION"


class MemoryLock:
    def __init__(self):
        self.held = False

    def acquire(self, *, blocking):
        if blocking is not False:
            raise AssertionError("unexpected_blocking_lock")
        if self.held:
            return False
        self.held = True
        return True

    def release(self):
        if not self.held:
            raise AssertionError("unowned_lock")
        self.held = False


class MemoryKernel:
    """Each file/socket/native operation has only an in-memory implementation."""

    def __init__(self):
        self.events, self.closed, self.fds = [], [], {}
        self.next_fd, self.changed = 10, False
        self.link_flags = netns.IFF_LOOPBACK
        self.interfaces = [(1, "lo")]
        self.close_failures, self.stat_calls = set(), {}
        self.proc_magic, self.ns_magic = netns.PROC_MAGIC, netns.NSFS_MAGIC
        self.namespace_type = netns.CLONE_NEWNET
        self.cloexec, self.stat_hook = {}, None
        self.unshare_return, self.statfs_return = 0, 0
        self.ioctl_hook, self.open_hook, self.socket_close_failure = None, None, False
        self.socket_fd, self.control = 30, Mock()
        self.control.fileno.return_value = self.socket_fd
        self.control.close.side_effect = self.close_socket

    def open(self, path, flags, *, dir_fd=None):
        self.events.append(("open", path, flags, dir_fd))
        if self.open_hook is not None:
            self.open_hook(path, flags, dir_fd)
        if path == "/proc":
            if dir_fd is not None:
                raise AssertionError("unexpected_proc_parent")
            kind, inode, mode = "proc", 5, stat.S_IFDIR | 0o555
        elif path == "self/ns/net":
            if self.fds[dir_fd]["kind"] != "proc":
                raise AssertionError("unexpected_ns_parent")
            kind, inode, mode = "ns", 202 if self.changed else 101, stat.S_IFREG | 0o444
        else:
            raise AssertionError("unexpected_open_target")
        fd, self.next_fd = self.next_fd, self.next_fd + 1
        self.fds[fd] = {"kind": kind, "flags": flags, "info": types.SimpleNamespace(
            st_dev=17, st_ino=inode, st_mode=mode, st_uid=0, st_gid=0,
        )}
        return fd

    def fstat(self, fd):
        self.events.append(("fstat", fd))
        self.stat_calls[fd] = self.stat_calls.get(fd, 0) + 1
        info = self.fds[fd]["info"]
        if self.stat_hook is not None:
            self.stat_hook(fd, self.stat_calls[fd], info)
        return info

    def fstatfs(self, fd, pointer):
        self.events.append(("fstatfs", fd))
        pointer._obj.kind = self.proc_magic if self.fds[fd]["kind"] == "proc" else self.ns_magic
        return self.statfs_return

    def fcntl(self, fd, command):
        self.events.append(("fcntl", fd, command))
        if command != fcntl.F_GETFD:
            raise AssertionError("unexpected_fcntl_command")
        return self.cloexec.get(fd, fcntl.FD_CLOEXEC)

    def ioctl(self, fd, command, request):
        self.events.append(("ioctl", fd, command, request))
        if self.ioctl_hook is not None:
            result = self.ioctl_hook(fd, command, request)
            if result is not None:
                return result
        if command == netns.NS_GET_NSTYPE:
            if self.fds[fd]["kind"] != "ns" or request != 0:
                raise AssertionError("unexpected_namespace_ioctl")
            return self.namespace_type
        if fd != self.socket_fd or not self.changed:
            raise AssertionError("old_namespace_ioctl")
        if type(request) is not bytes or len(request) != 40 or request[:16] != netns._LO_NAME:
            raise AssertionError("unexpected_ifreq")
        if command == netns.SIOCGIFFLAGS:
            return struct.pack("<16sH22x", netns._LO_NAME, self.link_flags)
        if command == netns.SIOCSIFFLAGS:
            self.link_flags = struct.unpack_from("<H", request, 16)[0]
            return request
        raise AssertionError("unexpected_interface_ioctl")

    def unshare(self, flags):
        self.events.append(("unshare", flags))
        if flags != netns.CLONE_NEWNET or not any(
            value["kind"] == "ns" for value in self.fds.values()
        ):
            raise AssertionError("unbound_unshare")
        self.changed = self.unshare_return == 0
        return self.unshare_return

    def socket(self, family, kind, protocol):
        self.events.append(("socket", family, kind, protocol))
        if not self.changed:
            raise AssertionError("socket_before_unshare")
        self.fds[self.socket_fd] = {"kind": "socket"}
        return self.control

    def if_nameindex(self):
        self.events.append("interfaces")
        if not self.changed:
            raise AssertionError("interface_check_before_unshare")
        return self.interfaces

    def close(self, fd):
        self.events.append(("close", fd))
        self.closed.append(fd)
        if fd not in self.fds:
            raise AssertionError("duplicate_or_unowned_close")
        del self.fds[fd]
        if fd in self.close_failures:
            raise OSError(PRIVATE)

    def close_socket(self):
        self.close(self.socket_fd)
        if self.socket_close_failure:
            raise OSError(PRIVATE)


class NetworkIsolationTests(unittest.TestCase):
    def setUp(self):
        self.kernel_thread = self.enterContext(patch.object(netns, "_kernel_single_thread"))
        self.kernel = MemoryKernel()
        kernel = self.kernel
        self.lock = MemoryLock()
        self.enterContext(patch.object(netns, "_SETUP_LOCK", self.lock))
        self.enterContext(patch.object(netns, "_ATTEMPT_OWNER", None))
        self.os = types.SimpleNamespace(
            **{name: getattr(os, name) for name in (
                "O_RDONLY", "O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK",
            )},
            open=Mock(side_effect=kernel.open), close=Mock(side_effect=kernel.close),
            fstat=Mock(side_effect=kernel.fstat), getpid=Mock(return_value=1001),
            getresuid=Mock(return_value=(0, 0, 0)), getresgid=Mock(return_value=(0, 0, 0)),
        )
        self.enterContext(patch.object(netns, "os", self.os))
        self.sys = types.SimpleNamespace(
            platform="linux", byteorder="little", dont_write_bytecode=True,
            implementation=types.SimpleNamespace(name="cpython"),
            flags=types.SimpleNamespace(no_site=1), modules={},
        )
        self.enterContext(patch.object(netns, "sys", self.sys))
        self.machine = Mock(return_value="x86_64")
        self.enterContext(patch.object(
            netns, "platform", types.SimpleNamespace(machine=self.machine),
        ))
        self.main_thread = object()
        self.threading = types.SimpleNamespace(
            current_thread=Mock(return_value=self.main_thread),
            main_thread=Mock(return_value=self.main_thread), active_count=Mock(return_value=1),
            get_ident=Mock(return_value=55),
        )
        self.enterContext(patch.object(netns, "threading", self.threading))
        self.fcntl = types.SimpleNamespace(
            fcntl=Mock(side_effect=kernel.fcntl), ioctl=Mock(side_effect=kernel.ioctl),
            F_GETFD=fcntl.F_GETFD, FD_CLOEXEC=fcntl.FD_CLOEXEC,
        )
        self.enterContext(patch.object(netns, "fcntl", self.fcntl))
        self.socket = types.SimpleNamespace(
            AF_INET=socket.AF_INET, SOCK_DGRAM=socket.SOCK_DGRAM, SOCK_CLOEXEC=0x80000,
            socket=Mock(side_effect=kernel.socket),
            if_nameindex=Mock(side_effect=kernel.if_nameindex),
        )
        self.enterContext(patch.object(netns, "socket", self.socket))
        self.libc = types.SimpleNamespace(
            unshare=Mock(side_effect=kernel.unshare), fstatfs=Mock(side_effect=kernel.fstatfs),
        )
        self.ctypes = types.SimpleNamespace(
            **{name: getattr(ctypes, name) for name in (
                "c_int", "c_void_p", "c_long", "POINTER", "byref",
            )}, sizeof=Mock(side_effect=ctypes.sizeof), CDLL=Mock(return_value=self.libc),
        )
        self.enterContext(patch.object(netns, "ctypes", self.ctypes))

    def tearDown(self):
        self.assertFalse(self.lock.held)
        self.assertEqual(len(self.kernel.closed), len(set(self.kernel.closed)))

    def error(self, code, action=None):
        with self.assertRaises(netns.NetworkIsolationError) as caught:
            (netns.create_loopback_namespace if action is None else action)()
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(caught.exception.args, (code,))
        self.assertTrue(caught.exception.__suppress_context__)
        self.assertNotIn(PRIVATE, repr(caught.exception))
        return caught.exception

    def no_side_effects(self):
        self.os.open.assert_not_called()
        self.os.close.assert_not_called()
        self.os.fstat.assert_not_called()
        self.ctypes.CDLL.assert_not_called()
        self.socket.socket.assert_not_called()
        self.fcntl.ioctl.assert_not_called()

    def test_import_adds_no_ml(self):
        self.assertFalse(any(name.split(".", 1)[0] in netns.ML_ROOTS
                             for name in IMPORT_ADDITIONS))

    def test_success_receipt_has_exact_fields_and_false_nonclaims(self):
        self.assertEqual(netns.create_loopback_namespace(), {
            "schema": netns.SCHEMA, "namespace_changed": True, "loopback_up": True,
            "only_loopback_interface": True, "namespace_fds_closed": True,
            **dict.fromkeys(netns.NONCLAIMS, False),
        })
        self.assertEqual(self.kernel.closed, [10, 12, 30, 13, 11])
        self.assertEqual(self.kernel.fds, {})
        self.assertEqual(netns._ATTEMPT_OWNER, (1001, 55))
        self.kernel_thread.assert_called_once_with()
        self.libc.unshare.assert_called_once_with(netns.CLONE_NEWNET)

    def test_authentication_and_configuration_order_is_explicit(self):
        netns.create_loopback_namespace()
        events = self.kernel.events
        index = events.index(("unshare", netns.CLONE_NEWNET))
        self.assertIn(("fstatfs", 11), events[:index])
        self.assertNotIn(("close", 11), events[:index])
        self.assertNotIn("interfaces", events[:index])
        self.assertLess(index, events.index(("fstatfs", 13)))
        self.assertLess(events.index(("fstatfs", 13)), events.index((
            "socket", self.socket.AF_INET, self.socket.SOCK_DGRAM | self.socket.SOCK_CLOEXEC, 0,
        )))
        self.assertEqual(events[-2:], [("close", 13), ("close", 11)])
        self.assertEqual(self.kernel.stat_calls[11], 3)
        self.assertEqual(self.kernel.stat_calls[13], 3)

    def test_fixed_descriptor_relative_targets_and_flags(self):
        netns.create_loopback_namespace()
        proc_flags = (os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                      | os.O_CLOEXEC | os.O_NONBLOCK)
        self.assertEqual(self.os.open.call_args_list, [
            call("/proc", proc_flags), call("self/ns/net", os.O_RDONLY | os.O_CLOEXEC, dir_fd=10),
            call("/proc", proc_flags), call("self/ns/net", os.O_RDONLY | os.O_CLOEXEC, dir_fd=12),
        ])

    def test_ifreq_is_exact_40_byte_x64_lo_and_only_up_is_added(self):
        self.kernel.link_flags |= 0x40
        netns.create_loopback_namespace()
        requests = [args.args for args in self.fcntl.ioctl.call_args_list
                    if args.args[1] in (netns.SIOCGIFFLAGS, netns.SIOCSIFFLAGS)]
        self.assertEqual([value[1] for value in requests], [
            netns.SIOCGIFFLAGS, netns.SIOCSIFFLAGS, netns.SIOCGIFFLAGS,
        ])
        for _fd, _command, request in requests:
            self.assertEqual(len(request), 40)
            self.assertEqual(request[:16], b"lo" + b"\0" * 14)
            self.assertEqual(request[18:], b"\0" * 22)
        self.assertEqual(struct.unpack_from("<H", requests[1][2], 16)[0], 0x49)
        self.assertEqual(self.socket.if_nameindex.call_count, 2)

    def test_native_signatures_are_fixed(self):
        netns.create_loopback_namespace()
        self.assertEqual(self.libc.unshare.argtypes, [ctypes.c_int])
        self.assertIs(self.libc.unshare.restype, ctypes.c_int)
        self.assertEqual(self.libc.fstatfs.argtypes, [ctypes.c_int, ctypes.POINTER(netns.StatFS)])
        self.assertIs(self.libc.fstatfs.restype, ctypes.c_int)
        self.ctypes.CDLL.assert_has_calls([call(None, use_errno=True)])

    def test_repeated_success_denied_without_another_open_or_unshare(self):
        netns.create_loopback_namespace()
        before = list(self.kernel.events)
        self.error("already_attempted")
        self.assertEqual(self.kernel.events, before)

    def test_failed_unshare_consumes_attempt_and_closes_old_descriptor(self):
        self.kernel.unshare_return = -1
        self.error("setup_failed")
        self.assertEqual(self.kernel.closed, [10, 11])
        self.socket.socket.assert_not_called()
        self.error("already_attempted")
        self.libc.unshare.assert_called_once_with(netns.CLONE_NEWNET)

    def test_initial_open_failure_also_consumes_attempt(self):
        self.os.open.side_effect = OSError(PRIVATE)
        self.error("setup_failed")
        self.error("already_attempted")
        self.os.open.assert_called_once()
        self.os.close.assert_not_called()

    def test_fork_inherited_attempt_and_thread_reuse_are_denied(self):
        netns._ATTEMPT_OWNER = (2002, 55)
        self.error("owner_mismatch")
        netns._ATTEMPT_OWNER = (1001, 66)
        self.error("owner_mismatch")
        self.no_side_effects()

    def test_nested_call_cannot_open_or_release_outer_lock(self):
        self.lock.held = True
        self.error("reentry")
        self.assertTrue(self.lock.held)
        self.lock.held = False
        self.no_side_effects()

    def test_runtime_platform_implementation_and_flags_reject_before_effects(self):
        for target, field, bad in (
            (self.sys, "platform", "darwin"), (self.sys, "byteorder", "big"),
            (self.sys, "dont_write_bytecode", False),
            (self.sys.implementation, "name", "pypy"), (self.sys.flags, "no_site", 0),
        ):
            with self.subTest(field=field), patch.object(target, field, bad):
                self.error("runtime_rejected")
        self.assertIsNone(netns._ATTEMPT_OWNER)
        self.no_side_effects()

    def test_wrong_machine_rejected_before_native_loading(self):
        self.machine.return_value = "aarch64"
        self.error("runtime_rejected")
        self.no_side_effects()

    def test_pointer_long_and_statfs_layout_sizes_are_mandatory(self):
        for target in (ctypes.c_void_p, ctypes.c_long, netns.StatFS):
            with self.subTest(target=target):
                self.ctypes.sizeof.side_effect = lambda value, target=target: (
                    1 if value is target else ctypes.sizeof(value)
                )
                self.error("runtime_rejected")
        self.no_side_effects()

    def test_every_real_effective_and_saved_root_identity_is_required(self):
        for function in (self.os.getresuid, self.os.getresgid):
            for value in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                function.return_value = value
                self.error("runtime_rejected")
            function.return_value = (0, 0, 0)
        self.no_side_effects()

    def test_missing_resuid_resgid_and_cloexec_facility_fail_closed(self):
        for target, field in ((self.os, "getresuid"), (self.os, "getresgid"),
                              (self.socket, "SOCK_CLOEXEC")):
            value = getattr(target, field)
            delattr(target, field)
            self.error("runtime_rejected")
            setattr(target, field, value)
        self.no_side_effects()

    def test_nonmain_thread_or_multiple_threads_are_denied_without_spawning(self):
        self.threading.current_thread.return_value = object()
        self.error("runtime_rejected")
        self.threading.current_thread.return_value = self.main_thread
        self.threading.active_count.return_value = 2
        self.error("runtime_rejected")
        self.no_side_effects()

    def test_each_ml_root_and_submodule_is_rejected_without_import(self):
        for root in netns.ML_ROOTS:
            for name in (root, root + ".synthetic"):
                self.sys.modules = {name: object()}
                self.error("runtime_rejected")
        self.no_side_effects()

    def test_procfs_magic_and_native_statfs_failure_reject_before_unshare(self):
        self.kernel.proc_magic = 0
        self.error("namespace_rejected")
        self.assertEqual(self.kernel.closed, [10])
        self.libc.unshare.assert_not_called()

    def test_nsfs_magic_rejects_and_closes_both_freshly_opened_fds(self):
        self.kernel.ns_magic = 0
        self.error("namespace_rejected")
        self.assertEqual(self.kernel.closed, [10, 11])
        self.libc.unshare.assert_not_called()

    def test_fstatfs_nonzero_return_denied_even_with_expected_magic(self):
        self.kernel.statfs_return = -1
        self.error("namespace_rejected")
        self.assertEqual(self.kernel.closed, [10])

    def test_non_network_namespace_type_is_rejected(self):
        self.kernel.namespace_type = 0x20000
        self.error("namespace_rejected")
        self.libc.unshare.assert_not_called()
        self.assertEqual(self.kernel.closed, [10, 11])

    def test_boolean_namespace_type_is_not_integer_authority(self):
        self.kernel.namespace_type = True
        self.error("namespace_rejected")
        self.libc.unshare.assert_not_called()

    def test_nsfs_ioctl_failure_is_sanitized_and_all_owned_fds_close(self):
        self.fcntl.ioctl.side_effect = OSError(PRIVATE)
        self.error("setup_failed")
        self.assertEqual(self.kernel.closed, [10, 11])

    def test_proc_metadata_requires_directory_root_ownership_and_cloexec(self):
        def mutate(fd, _count, info):
            if fd == 10:
                info.st_uid = 1
        self.kernel.stat_hook = mutate
        self.error("namespace_rejected")
        self.libc.unshare.assert_not_called()
        self.assertEqual(self.kernel.closed, [10])

    def test_namespace_metadata_requires_root_regular_positive_inode(self):
        def mutate(fd, _count, info):
            if fd == 11:
                info.st_mode = stat.S_IFDIR | 0o444
        self.kernel.stat_hook = mutate
        self.error("namespace_rejected")
        self.assertEqual(self.kernel.closed, [10, 11])

    def test_namespace_cloexec_required_before_unshare(self):
        self.kernel.cloexec[11] = 0
        self.error("namespace_rejected")
        self.libc.unshare.assert_not_called()

    def test_proc_cloexec_required_before_namespace_open(self):
        self.kernel.cloexec[10] = 0
        self.error("namespace_rejected")
        self.os.open.assert_called_once()

    def test_no_changed_namespace_identity_denies_socket_construction(self):
        def unchanged(fd, _count, info):
            if fd == 13:
                info.st_ino = 101
        self.kernel.stat_hook = unchanged
        self.error("namespace_unchanged")
        self.socket.socket.assert_not_called()
        self.assertEqual(self.kernel.closed, [10, 12, 13, 11])

    def test_old_namespace_identity_drift_after_configuration_is_denied(self):
        def drift(fd, count, info):
            if fd == 11 and count == 3:
                info.st_ino = 999
        self.kernel.stat_hook = drift
        self.error("namespace_rejected")
        self.assertEqual(self.kernel.closed, [10, 12, 30, 13, 11])

    def test_new_namespace_identity_drift_after_configuration_is_denied(self):
        def drift(fd, count, info):
            if fd == 13 and count == 3:
                info.st_ino = 999
        self.kernel.stat_hook = drift
        self.error("namespace_rejected")
        self.assertEqual(self.kernel.closed, [10, 12, 30, 13, 11])

    def test_new_namespace_open_failure_closes_old_and_new_proc(self):
        def failure(path, _flags, _parent):
            if path == "self/ns/net" and self.kernel.changed:
                raise OSError(PRIVATE)
        self.kernel.open_hook = failure
        self.error("setup_failed")
        self.assertEqual(self.kernel.closed, [10, 12, 11])
        self.socket.socket.assert_not_called()

    def test_libc_exception_after_old_namespace_binding_does_not_retry(self):
        self.libc.unshare.side_effect = RuntimeError(PRIVATE)
        self.error("setup_failed")
        self.assertEqual(self.kernel.closed, [10, 11])
        self.libc.unshare.assert_called_once()

    def test_socket_construction_failure_closes_both_namespaces(self):
        self.socket.socket.side_effect = OSError(PRIVATE)
        self.error("setup_failed")
        self.assertEqual(self.kernel.closed, [10, 12, 13, 11])
        self.kernel.control.close.assert_not_called()

    def test_socket_cloexec_missing_closes_control_once_and_denies_ioctl(self):
        self.kernel.cloexec[30] = 0
        self.error("interface_rejected")
        self.kernel.control.close.assert_called_once_with()
        self.socket.if_nameindex.assert_not_called()

    def test_socket_fileno_failure_still_closes_object_once(self):
        self.kernel.control.fileno.side_effect = OSError(PRIVATE)
        self.error("setup_failed")
        self.kernel.control.close.assert_called_once_with()

    def test_only_loopback_interface_contract_rejects_extra_link(self):
        self.kernel.interfaces = [(1, "lo"), (2, "eth0")]
        self.error("interface_rejected")
        self.assertFalse(any(args.args[1] == netns.SIOCSIFFLAGS
                             for args in self.fcntl.ioctl.call_args_list))

    def test_only_loopback_helper_rejects_malformed_values(self):
        self.kernel.changed = True
        for value in ((), [], [(1, "eth0")], [[1, "lo"]], [(True, "lo")],
                      [(0, "lo")], [(2**31, "lo")], [(1, b"lo")], [(1, "lo", 3)]):
            self.kernel.interfaces = value
            self.error("interface_rejected", netns._only_loopback)
        self.os.open.assert_not_called()

    def test_post_configuration_extra_link_denies_success(self):
        self.socket.if_nameindex.side_effect = [[(1, "lo")], [(1, "lo"), (2, "eth0")]]
        self.error("interface_rejected")
        self.kernel.control.close.assert_called_once_with()

    def test_initial_link_must_have_loopback_flag(self):
        self.kernel.link_flags = 0
        self.error("interface_rejected")
        self.assertFalse(any(args.args[1] == netns.SIOCSIFFLAGS
                             for args in self.fcntl.ioctl.call_args_list))

    def test_ioctl_response_shape_and_name_are_authenticated(self):
        self.kernel.ioctl_hook = lambda _fd, cmd, _req: b"x" * 40 if (
            cmd == netns.SIOCGIFFLAGS
        ) else None
        self.error("interface_rejected")
        self.kernel.control.close.assert_called_once_with()

    def test_get_flags_helper_rejects_nonbytes_short_and_wrong_name(self):
        for response in (None, True, bytearray(40), b"", b"lo" + b"\0" * 37, b"x" * 40):
            self.fcntl.ioctl.side_effect = None
            self.fcntl.ioctl.return_value = response
            self.error("interface_rejected", lambda: netns._get_flags(30))
        self.os.open.assert_not_called()

    def test_set_flags_response_must_be_exact_and_named(self):
        self.kernel.ioctl_hook = lambda _fd, cmd, _req: b"" if (
            cmd == netns.SIOCSIFFLAGS
        ) else None
        self.error("interface_rejected")
        self.kernel.control.close.assert_called_once_with()

    def test_flag_readback_must_confirm_up_after_successful_set(self):
        def no_effect(_fd, command, request):
            return request if command == netns.SIOCSIFFLAGS else None
        self.kernel.ioctl_hook = no_effect
        self.error("interface_rejected")

    def test_set_flags_exception_is_not_retried(self):
        def failure(_fd, command, _request):
            if command == netns.SIOCSIFFLAGS:
                raise OSError(PRIVATE)
        self.kernel.ioctl_hook = failure
        self.error("setup_failed")
        self.assertEqual(sum(args.args[1] == netns.SIOCSIFFLAGS
                             for args in self.fcntl.ioctl.call_args_list), 1)

    def test_close_failure_of_proc_still_closes_namespace_and_never_unshares(self):
        self.kernel.close_failures = {10}
        self.error("close_failed")
        self.assertEqual(self.kernel.closed, [10, 11])
        self.libc.unshare.assert_not_called()

    def test_close_failure_of_control_still_closes_both_namespaces(self):
        self.kernel.socket_close_failure = True
        self.error("close_failed")
        self.assertEqual(self.kernel.closed, [10, 12, 30, 13, 11])
        self.kernel.control.close.assert_called_once_with()

    def test_close_failure_new_namespace_does_not_prevent_old_close(self):
        self.kernel.close_failures = {13}
        self.error("close_failed")
        self.assertEqual(self.kernel.closed, [10, 12, 30, 13, 11])

    def test_close_failure_old_namespace_denies_otherwise_valid_receipt(self):
        self.kernel.close_failures = {11}
        self.error("close_failed")
        self.assertEqual(self.kernel.closed, [10, 12, 30, 13, 11])

    def test_close_failure_overrides_setup_error_and_is_never_retried(self):
        self.libc.unshare.side_effect = OSError(PRIVATE)
        self.kernel.close_failures = {11}
        self.error("close_failed")
        self.error("already_attempted")
        self.assertEqual(self.kernel.closed, [10, 11])

    def test_invalid_direct_namespace_fd_rejected_before_any_syscall(self):
        for value in (None, True, -1, 1.0, "11"):
            self.error("namespace_rejected", lambda value=value: netns._namespace_identity(value))
        self.no_side_effects()

    def test_error_codes_cannot_retain_arbitrary_values(self):
        for value in (PRIVATE, None, True, {"private": PRIVATE}):
            error = netns.NetworkIsolationError(value)
            self.assertEqual(error.code, "setup_failed")
            self.assertEqual(error.args, ("setup_failed",))
            self.assertNotIn(PRIVATE, repr(error))


class KernelThreadTests(unittest.TestCase):
    """Read and native authentication are wholly in-memory, including failures."""

    def setUp(self):
        self.data = b"Name:\tsynthetic\nThreads:\t1\n"
        self.position, self.read_calls, self.closed = 0, [], []
        self.metadata = {
            10: types.SimpleNamespace(st_dev=17, st_mode=stat.S_IFDIR | 0o555, st_uid=0, st_gid=0),
            11: types.SimpleNamespace(st_dev=17, st_mode=stat.S_IFREG | 0o444, st_uid=0, st_gid=0),
        }
        self.os = types.SimpleNamespace(
            **{name: getattr(os, name) for name in (
                "O_RDONLY", "O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK",
            )}, open=Mock(side_effect=[10, 11]), fstat=Mock(side_effect=self.metadata.__getitem__),
            read=Mock(side_effect=self.read), close=Mock(side_effect=self.closed.append),
        )
        self.enterContext(patch.object(netns, "os", self.os))
        self.fcntl = types.SimpleNamespace(
            fcntl=Mock(return_value=fcntl.FD_CLOEXEC), F_GETFD=fcntl.F_GETFD,
            FD_CLOEXEC=fcntl.FD_CLOEXEC,
        )
        self.enterContext(patch.object(netns, "fcntl", self.fcntl))
        self.magic, self.native_result = netns.PROC_MAGIC, 0
        self.native = Mock(side_effect=self.fstatfs)
        self.ctypes = types.SimpleNamespace(
            **{name: getattr(ctypes, name) for name in ("c_int", "POINTER", "byref")},
            CDLL=Mock(return_value=types.SimpleNamespace(fstatfs=self.native)),
        )
        self.enterContext(patch.object(netns, "ctypes", self.ctypes))
        self.no_socket = self.enterContext(patch.object(netns, "socket", types.SimpleNamespace()))

    def read(self, fd, size):
        self.assertEqual(fd, 11)
        self.assertGreater(size, 0)
        self.assertLessEqual(size, 4096)
        self.read_calls.append(size)
        part = self.data[self.position:self.position + size]
        self.position += len(part)
        return part

    def fstatfs(self, fd, pointer):
        self.assertIn(fd, (10, 11))
        pointer._obj.kind = self.magic
        return self.native_result

    def error(self, code="runtime_rejected"):
        with self.assertRaises(netns.NetworkIsolationError) as caught:
            netns._kernel_single_thread()
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(caught.exception.args, (code,))
        self.assertTrue(caught.exception.__suppress_context__)
        self.assertNotIn(PRIVATE, repr(caught.exception))

    def test_authenticated_single_thread_status_ends_with_eof_and_closes_once(self):
        self.assertIsNone(netns._kernel_single_thread())
        self.assertEqual(self.closed, [11, 10])
        self.assertEqual(self.read_calls, [4096, 4096])
        self.assertEqual(self.native.call_count, 2)
        self.os.open.assert_has_calls([
            call("/proc", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                 | os.O_CLOEXEC | os.O_NONBLOCK),
            call("self/status", os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
                 | os.O_NONBLOCK, dir_fd=10),
        ])

    def test_multiple_native_threads_deny_even_without_python_thread_registry(self):
        self.data = b"Threads:\t2\n"
        self.error()
        self.assertEqual(self.closed, [11, 10])

    def test_missing_threads_field_denied(self):
        self.data = b"Name:\tsynthetic\n"
        self.error()

    def test_duplicate_threads_field_denied(self):
        self.data = b"Threads:\t1\nThreads:\t1\n"
        self.error()

    def test_noncanonical_threads_value_denied(self):
        self.data = b"Threads:\t01\n"
        self.error()

    def test_embedded_nul_denied(self):
        self.data = b"Name:\tx\0\nThreads:\t1\n"
        self.error()

    def test_missing_final_newline_denied(self):
        self.data = b"Threads:\t1"
        self.error()

    def test_empty_status_denied(self):
        self.data = b""
        self.error()

    def test_exact_maximum_status_requires_eof_sentinel_read(self):
        self.data = b"X" * (netns._MAX_STATUS_BYTES - 12) + b"\nThreads:\t1\n"
        self.assertEqual(len(self.data), netns._MAX_STATUS_BYTES)
        self.assertIsNone(netns._kernel_single_thread())
        self.assertEqual(self.read_calls, [4096, 4096, 4096, 4096, 1])

    def test_oversized_status_rejected_after_one_sentinel_byte(self):
        self.data = b"X" * (netns._MAX_STATUS_BYTES + 10)
        self.error()
        self.assertEqual(self.read_calls, [4096, 4096, 4096, 4096, 1])
        self.assertEqual(self.position, netns._MAX_STATUS_BYTES + 1)
        self.assertEqual(self.closed, [11, 10])

    def test_128_read_call_limit_denied_without_unbounded_loop(self):
        self.os.read.side_effect = None
        self.os.read.return_value = b"x"
        self.error()
        self.assertEqual(self.os.read.call_count, 128)

    def test_nonbyte_read_rejected(self):
        self.os.read.side_effect = None
        self.os.read.return_value = bytearray(b"Threads:\t1\n")
        self.error()
        self.assertEqual(self.closed, [11, 10])

    def test_status_different_proc_device_denied_before_read(self):
        self.metadata[11].st_dev = 18
        self.error()
        self.os.read.assert_not_called()

    def test_status_non_root_owner_denied_before_read(self):
        self.metadata[11].st_uid = 1001
        self.error()
        self.os.read.assert_not_called()

    def test_status_nonregular_mode_denied_before_read(self):
        self.metadata[11].st_mode = stat.S_IFLNK | 0o777
        self.error()
        self.os.read.assert_not_called()

    def test_proc_non_root_group_denied_before_status_open(self):
        self.metadata[10].st_gid = 1001
        self.error()
        self.os.open.assert_called_once()
        self.assertEqual(self.closed, [10])

    def test_status_cloexec_missing_denied_before_read(self):
        self.fcntl.fcntl.side_effect = [fcntl.FD_CLOEXEC, 0]
        self.error()
        self.os.read.assert_not_called()

    def test_wrong_procfs_magic_denied_and_proc_closed(self):
        self.magic = 0
        self.error()
        self.assertEqual(self.closed, [10])

    def test_native_fstatfs_failure_denied_even_if_magic_matches(self):
        self.native_result = -1
        self.error()
        self.assertEqual(self.closed, [10])

    def test_read_error_is_sanitized_and_closes_status_then_proc(self):
        self.os.read.side_effect = OSError(PRIVATE)
        self.error()
        self.assertEqual(self.closed, [11, 10])

    def test_status_open_failure_closes_proc(self):
        self.os.open.side_effect = [10, OSError(PRIVATE)]
        self.error()
        self.assertEqual(self.closed, [10])

    def test_initial_open_failure_has_no_cleanup_target(self):
        self.os.open.side_effect = OSError(PRIVATE)
        self.error()
        self.os.close.assert_not_called()

    def test_status_close_failure_does_not_retry_or_skip_proc_close(self):
        self.os.close.side_effect = [OSError(PRIVATE), None]
        self.error("close_failed")
        self.assertEqual(self.os.close.call_args_list, [call(11), call(10)])

    def test_proc_close_failure_denies_otherwise_valid_status(self):
        self.os.close.side_effect = [None, OSError(PRIVATE)]
        self.error("close_failed")
        self.assertEqual(self.os.close.call_args_list, [call(11), call(10)])


if __name__ == "__main__":
    unittest.main()
