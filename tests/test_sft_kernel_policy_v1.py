"""Pure synthetic cBPF and mocked installation tests; never install a filter."""

from __future__ import annotations

import ctypes
import importlib.util
import struct
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_synthetic_sft_kernel_policy_v1", ROOT / "src/lumi/posttraining_sft_kernel_policy_v1.py",
)
policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy)


def interpret(instructions, number, arguments=(), *, arch=policy.AUDIT_ARCH_X86_64):
    """Tiny independent cBPF interpreter over a synthetic 64-byte seccomp_data."""
    args = [*arguments, *([0] * (6 - len(arguments)))]
    data = struct.pack(
        "<IIQ6Q", number & 0xFFFFFFFF, arch & 0xFFFFFFFF, 0,
        *(value & 0xFFFFFFFFFFFFFFFF for value in args),
    )
    position = accumulator = 0
    for _attempt in range(len(instructions) + 1):
        if not 0 <= position < len(instructions):
            raise AssertionError("synthetic filter jumped outside its instruction array")
        code, yes, no, value = instructions[position]
        position += 1
        if code == 0x20:
            if value % 4 or not 0 <= value <= 60:
                raise AssertionError("synthetic filter used an invalid data offset")
            accumulator = struct.unpack_from("<I", data, value)[0]
        elif code == 0x15:
            position += yes if accumulator == value else no
        elif code == 0x45:
            position += yes if accumulator & value else no
        elif code == 0x06:
            return value
        else:
            raise AssertionError("synthetic filter used an unexpected opcode")
    raise AssertionError("synthetic filter did not terminate")


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.fd = 11
        self.instructions = policy.compile_filter(self.fd)

    def result(self, name, *arguments):
        return interpret(self.instructions, policy.SYSCALLS[name], arguments)

    def allowed(self, name, *arguments):
        self.assertEqual(self.result(name, *arguments), policy.SECCOMP_RET_ALLOW)

    def denied(self, name, *arguments):
        self.assertEqual(self.result(name, *arguments), policy.DENY)

    def test_filter_is_bounded_immutable_well_typed_and_forward_only(self):
        self.assertIs(type(self.instructions), tuple)
        self.assertLessEqual(len(self.instructions), 4096)
        for position, instruction in enumerate(self.instructions):
            with self.subTest(position=position):
                self.assertIs(type(instruction), tuple)
                self.assertEqual(len(instruction), 4)
                self.assertTrue(all(type(value) is int for value in instruction))
                code, yes, no, value = instruction
                self.assertIn(code, (0x20, 0x15, 0x45, 0x06))
                self.assertTrue(0 <= yes <= 255 and 0 <= no <= 255)
                self.assertTrue(0 <= value <= 0xFFFFFFFF)
                if code in (0x15, 0x45):
                    self.assertLess(position + 1 + yes, len(self.instructions))
                    self.assertLess(position + 1 + no, len(self.instructions))

    def test_fd_requires_exact_integer_in_fixed_range(self):
        for value in (None, True, False, 0, 1, 2, -1, 1_048_577, 2**64, "11", 11.0):
            with self.subTest(value=value), self.assertRaises(policy.PolicyError):
                policy.compile_filter(value)
        for value in (3, 64, 1_048_576):
            with self.subTest(value=value):
                instructions = policy.compile_filter(value)
                self.assertEqual(interpret(instructions, policy.SYSCALLS["sendmsg"], (value,)),
                                 policy.SECCOMP_RET_ALLOW)

    def test_architecture_rejection_precedes_all_syscall_allowance(self):
        for arch in (0, 0x40000003, 0x4000003E, 0xC00000B7, 0xFFFFFFFF):
            for name in ("exit", "sendmsg", "prctl", "fork"):
                with self.subTest(arch=arch, name=name):
                    self.assertEqual(
                        interpret(self.instructions, policy.SYSCALLS[name], (self.fd,), arch=arch),
                        policy.SECCOMP_RET_KILL_PROCESS,
                    )

    def test_x32_and_negative_syscall_numbers_kill_before_dispatch(self):
        for number in (0x40000000, 0x40000003, 0x4000003C, 0x40000200, -1):
            with self.subTest(number=number):
                self.assertEqual(interpret(self.instructions, number, (self.fd,)),
                                 policy.SECCOMP_RET_KILL_PROCESS)

    def test_all_unconditional_runtime_operations_allow(self):
        for name in policy.UNCONDITIONAL_SYSCALLS:
            with self.subTest(name=name):
                self.allowed(name)

    def test_all_named_dangerous_operations_deny(self):
        guarded = {"mmap", "mprotect", "sendmsg", "recvmsg", "ioctl", "getsockopt", "prctl"}
        expected = set(policy.SYSCALLS) - set(policy.UNCONDITIONAL_SYSCALLS) - guarded
        for name in sorted(expected):
            with self.subTest(name=name):
                self.denied(name)

    def test_unknown_and_historical_compat_syscalls_default_to_eperm(self):
        known = set(policy.SYSCALLS.values())
        for number in range(1024):
            if number not in known:
                with self.subTest(number=number):
                    self.assertEqual(interpret(self.instructions, number), policy.DENY)

    def test_ipc_permits_only_bound_fd_without_high_word_alias(self):
        for name in ("sendmsg", "recvmsg"):
            self.allowed(name, self.fd, 0, 0)
            for value in (0, 1, 2, -1, self.fd - 1, self.fd + 1, self.fd + 2**32):
                with self.subTest(name=name, value=value):
                    self.denied(name, value, 0, 0)

    def test_ioctl_permits_only_bound_fd_and_exact_fionbio(self):
        self.allowed("ioctl", self.fd, 0x5421, 0)
        for fd, command in ((self.fd + 1, 0x5421), (self.fd + 2**32, 0x5421),
                            (self.fd, 0), (self.fd, 0x5421 + 2**32), (self.fd, -1)):
            with self.subTest(fd=fd, command=command):
                self.denied("ioctl", fd, command, 0)

    def test_socket_readback_options_are_closed_and_exact(self):
        for option in (3, 4, 16):
            with self.subTest(option=option):
                self.allowed("getsockopt", self.fd, 1, option, 0, 0)
                self.denied("getsockopt", self.fd, 1, option + 2**32, 0, 0)
        for fd, level, option in ((self.fd + 1, 1, 3), (self.fd + 2**32, 1, 3),
                                  (self.fd, 2, 3), (self.fd, 1 + 2**32, 3),
                                  (self.fd, 1, 2), (self.fd, 1, 17)):
            with self.subTest(fd=fd, level=level, option=option):
                self.denied("getsockopt", fd, level, option, 0, 0)

    def test_anonymous_private_mapping_accepts_only_nonexecutable_profile(self):
        for protection in (0, 1, 2, 3):
            for descriptor in (0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF):
                with self.subTest(protection=protection, descriptor=descriptor):
                    self.allowed("mmap", 0, 4096, protection, 0x22, descriptor, 0)
        for protection in (4, 5, 6, 7, 8, 0x1000000, 3 + 2**32, -1):
            with self.subTest(protection=protection):
                self.denied("mmap", 0, 4096, protection, 0x22, -1, 0)

    def test_mapping_rejects_file_shared_fixed_alias_and_nonzero_offset(self):
        for index, value in ((3, 0x21), (3, 0x12), (3, 0x32), (3, 0x22 + 2**32),
                             (4, 0), (4, self.fd), (4, -2), (4, 0xFFFFFFFE),
                             (4, 0x00000001FFFFFFFF), (4, 0x7FFFFFFFFFFFFFFF),
                             (4, 0xFFFFFFFEFFFFFFFF),
                             (5, 1), (5, 2**32)):
            arguments = [0, 4096, 3, 0x22, -1, 0]
            arguments[index] = value
            with self.subTest(index=index, value=value):
                self.denied("mmap", *arguments)

    def test_mprotect_can_remove_execution_but_never_add_execution(self):
        for protection in (0, 1, 2, 3):
            self.allowed("mprotect", 0, 4096, protection)
        for protection in (4, 5, 7, 8, 3 + 2**32, -1):
            with self.subTest(protection=protection):
                self.denied("mprotect", 0, 4096, protection)

    def test_prctl_only_two_getters_and_zero_other_parameters(self):
        for getter in (policy.PR_GET_SECCOMP, policy.PR_GET_NO_NEW_PRIVS):
            self.allowed("prctl", getter, 0, 0, 0, 0)
            self.denied("prctl", getter + 2**32, 0, 0, 0, 0)
            for position in range(1, 5):
                for bad in (1, 2**32, -1):
                    arguments = [getter, 0, 0, 0, 0]
                    arguments[position] = bad
                    with self.subTest(getter=getter, position=position, bad=bad):
                        self.denied("prctl", *arguments)
        for operation in (0, policy.PR_SET_SECCOMP, policy.PR_SET_NO_NEW_PRIVS, 8, 47):
            with self.subTest(operation=operation):
                self.denied("prctl", operation, 0, 0, 0, 0)

    def test_policy_error_discards_nonallowlisted_text(self):
        for value in (None, "SECRET_EXCEPTION_TEXT", 1, True, [], {}):
            with self.subTest(value=value):
                error = policy.PolicyError(value)
                self.assertEqual(str(error), "kernel_policy_configuration_rejected")
                self.assertEqual(error.code, str(error))


class AbiTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(policy.sys, "platform", "linux"))
        self.enterContext(patch.object(policy.sys, "byteorder", "little"))
        self.enterContext(patch.object(policy.sys, "implementation", types.SimpleNamespace(
            name="cpython",
        )))
        self.enterContext(patch.object(policy.sys, "version_info", (3, 12, 0)))
        self.enterContext(patch.object(policy.sys, "flags", types.SimpleNamespace(
            no_site=1, optimize=0,
        )))
        self.enterContext(patch.object(policy.sys, "dont_write_bytecode", True))
        self.uname = self.enterContext(patch.object(policy.os, "uname", return_value=(
            types.SimpleNamespace(machine="x86_64")
        )))
        self.uid = self.enterContext(patch.object(policy.os, "getuid", return_value=1000))
        self.gid = self.enterContext(patch.object(policy.os, "getgid", return_value=1000))
        self.uids = self.enterContext(patch.object(policy.os, "getresuid",
                                                 return_value=(1000, 1000, 1000), create=True))
        self.gids = self.enterContext(patch.object(policy.os, "getresgid",
                                                 return_value=(1000, 1000, 1000), create=True))

    def rejected(self):
        with self.assertRaises(policy.PolicyError) as caught:
            policy.require_abi()
        self.assertEqual(caught.exception.code, "kernel_policy_unavailable")

    def test_supported_abi_read_only(self):
        with patch.object(policy.resource, "setrlimit") as limits, patch.object(
            policy.ctypes, "CDLL",
        ) as libc:
            self.assertIsNone(policy.require_abi())
        limits.assert_not_called()
        libc.assert_not_called()

    def test_linux_platform_gate_precedes_identity_access(self):
        with patch.object(policy.sys, "platform", "darwin"):
            self.rejected()
        self.uname.assert_not_called()
        self.uid.assert_not_called()

    def test_machine_and_endianness_are_exact(self):
        for machine in ("aarch64", "AMD64", "i686", ""):
            self.uname.return_value.machine = machine
            with self.subTest(machine=machine):
                self.rejected()
        self.uname.return_value.machine = "x86_64"
        with patch.object(policy.sys, "byteorder", "big"):
            self.rejected()

    def test_little_endian_ilp32_is_rejected(self):
        original = policy.struct.calcsize
        with patch.object(policy.struct, "calcsize", side_effect=(
            lambda kind: 4 if kind == "l" else original(kind)
        )):
            self.rejected()

    def test_interpreter_and_isolation_flags_are_exact(self):
        for version in ((3, 11, 0), (3, 14, 0)):
            with self.subTest(version=version), patch.object(policy.sys, "version_info", version):
                self.rejected()
        with patch.object(policy.sys, "implementation", types.SimpleNamespace(name="pypy")):
            self.rejected()
        for flags in (types.SimpleNamespace(no_site=0, optimize=0),
                      types.SimpleNamespace(no_site=1, optimize=1)):
            with patch.object(policy.sys, "flags", flags):
                self.rejected()
        with patch.object(policy.sys, "dont_write_bytecode", False):
            self.rejected()

    def test_nonroot_uniform_uid_and_gid_required(self):
        for attribute, value in ((self.uid, 0), (self.gid, 0),
                                 (self.uids, (1000, 0, 1000)),
                                 (self.gids, (1000, 1000, 0))):
            original = attribute.return_value
            attribute.return_value = value
            with self.subTest(value=value):
                self.rejected()
            attribute.return_value = original

    def test_identity_failure_is_opaque(self):
        self.uids.side_effect = OSError("SECRET_EXCEPTION_TEXT")
        self.rejected()


class InstallationTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(policy, "require_abi"))
        self.events = []
        self.profile = {key: (policy.resource.RLIM_INFINITY,) * 2 for key in policy.LIMITS}
        self.getlimit = self.enterContext(patch.object(policy.resource, "getrlimit",
                                                       side_effect=self.get_limit))
        self.setlimit = self.enterContext(patch.object(policy.resource, "setrlimit",
                                                       side_effect=self.set_limit))
        self.prctl = Mock(side_effect=self.call_prctl)
        self.libc = self.enterContext(patch.object(policy.ctypes, "CDLL", return_value=(
            types.SimpleNamespace(prctl=self.prctl)
        )))
        self.observed_filter = None

    def get_limit(self, name):
        self.events.append(("getlimit", name))
        return self.profile[name]

    def set_limit(self, name, value):
        self.events.append(("setlimit", name))
        self.profile[name] = value

    def call_prctl(self, option, value, address, fourth, fifth):
        self.events.append(("prctl", option))
        self.assertEqual((fourth, fifth), (0, 0))
        if option == policy.PR_SET_SECCOMP:
            self.assertEqual(value, 2)
            self.assertGreater(address, 0)
            program = ctypes.cast(address, ctypes.POINTER(policy._SockFprog)).contents
            self.observed_filter = tuple(
                (item.code, item.jt, item.jf, item.k)
                for item in (program.filter[index] for index in range(program.len))
            )
        else:
            self.assertEqual(address, 0)
        return {policy.PR_SET_NO_NEW_PRIVS: 0, policy.PR_GET_NO_NEW_PRIVS: 1,
                policy.PR_SET_SECCOMP: 0, policy.PR_GET_SECCOMP: 2}[option]

    def failed(self, code):
        with self.assertRaises(policy.PolicyError) as caught:
            policy.install(11)
        self.assertEqual(caught.exception.code, code)
        self.assertNotIn("SECRET", str(caught.exception))

    def test_fixed_limits_match_exact_six_resource_contracts(self):
        self.assertEqual(policy.LIMITS, {
            policy.resource.RLIMIT_AS: (134_217_728, 134_217_728),
            policy.resource.RLIMIT_CPU: (2, 2), policy.resource.RLIMIT_CORE: (0, 0),
            policy.resource.RLIMIT_FSIZE: (0, 0), policy.resource.RLIMIT_NOFILE: (64, 64),
            policy.resource.RLIMIT_NPROC: (1, 1),
        })

    def test_installation_preflights_all_limits_then_sets_reads_nnp_filter_reads(self):
        value = policy.install(11)
        names = list(policy.LIMITS)
        self.assertEqual(self.events, [
            *(("getlimit", name) for name in names),
            *(("setlimit", name) for name in names),
            *(("getlimit", name) for name in names),
            ("prctl", policy.PR_SET_NO_NEW_PRIVS), ("prctl", policy.PR_GET_NO_NEW_PRIVS),
            ("prctl", policy.PR_SET_SECCOMP), ("prctl", policy.PR_GET_SECCOMP),
            ("prctl", policy.PR_GET_NO_NEW_PRIVS),
        ])
        self.assertEqual(self.observed_filter, policy.compile_filter(11))
        self.libc.assert_called_once_with(None, use_errno=True)
        self.assertEqual(self.prctl.argtypes, [ctypes.c_int] + [ctypes.c_ulong] * 4)
        self.assertIs(self.prctl.restype, ctypes.c_int)
        self.assertEqual(value, {
            "schema_version": "lumi.sft-kernel-policy-installation.v0.1",
            "policy_installed": True, "resource_limits_verified": 6,
            "no_new_privileges": True, "seccomp_filter_mode_verified": True,
            "cgroup_memory_enforcement": False, "namespace_containment": False,
            "real_sft_execution_permitted": False,
        })

    def test_existing_exact_limits_are_not_raised(self):
        self.profile = dict(policy.LIMITS)
        self.assertTrue(policy.install(11)["policy_installed"])
        self.assertEqual(self.profile, policy.LIMITS)

    def test_inherited_stricter_soft_limit_rejects_before_any_change(self):
        self.profile[policy.resource.RLIMIT_NOFILE] = (63, 64)
        self.failed("kernel_policy_limits_rejected")
        self.setlimit.assert_not_called()
        self.prctl.assert_not_called()

    def test_inherited_stricter_hard_limit_rejects_before_any_change(self):
        self.profile[policy.resource.RLIMIT_AS] = (64, 64)
        self.failed("kernel_policy_limits_rejected")
        self.setlimit.assert_not_called()
        self.prctl.assert_not_called()

    def test_malformed_inherited_values_reject_without_mutation(self):
        key = next(iter(policy.LIMITS))
        for value in (None, [], [1, 1], (True, True), ("unlimited", "unlimited"), (1,), (-2, -2)):
            self.profile[key] = value
            with self.subTest(value=value):
                self.failed("kernel_policy_limits_rejected")
        self.setlimit.assert_not_called()
        self.prctl.assert_not_called()

    def test_limit_preflight_exception_is_opaque_and_no_mutation(self):
        self.getlimit.side_effect = OSError("SECRET_EXCEPTION_TEXT")
        self.failed("kernel_policy_limits_rejected")
        self.setlimit.assert_not_called()
        self.prctl.assert_not_called()

    def test_partial_limit_failure_has_no_retry_rollback_or_filter(self):
        def fail(name, _value):
            self.events.append(("setlimit", name))
            raise OSError("SECRET_EXCEPTION_TEXT")
        self.setlimit.side_effect = fail
        self.failed("kernel_policy_limits_rejected")
        self.assertEqual(self.setlimit.call_count, 1)
        self.prctl.assert_not_called()

    def test_limit_readback_mismatch_prevents_nnp_and_filter(self):
        self.setlimit.side_effect = lambda *_args: None
        self.failed("kernel_policy_limits_rejected")
        self.assertEqual(self.setlimit.call_count, 6)
        self.prctl.assert_not_called()

    def test_boolean_limit_readback_is_rejected(self):
        original = self.get_limit

        def observe(name):
            if self.setlimit.call_count == 6 and name == policy.resource.RLIMIT_NPROC:
                return (True, True)
            return original(name)
        self.getlimit.side_effect = observe
        self.failed("kernel_policy_limits_rejected")
        self.prctl.assert_not_called()

    def test_no_new_privilege_set_failure_never_installs_filter(self):
        self.prctl.side_effect = [-1]
        self.failed("kernel_policy_no_new_privileges_rejected")
        self.assertEqual(self.prctl.call_count, 1)

    def test_no_new_privilege_readback_failure_never_installs_filter(self):
        self.prctl.side_effect = [0, 0]
        self.failed("kernel_policy_no_new_privileges_rejected")
        self.assertEqual(self.prctl.call_count, 2)

    def test_filter_installation_failure_is_terminal_without_retry(self):
        self.prctl.side_effect = [0, 1, -1]
        self.failed("kernel_policy_installation_rejected")
        self.assertEqual(self.prctl.call_count, 3)

    def test_filter_mode_readback_must_be_exact_filter_mode(self):
        for value in (0, 1, -1):
            self.prctl.reset_mock(side_effect=True)
            self.prctl.side_effect = [0, 1, 0, value]
            with self.subTest(value=value):
                self.failed("kernel_policy_readback_rejected")
                self.assertEqual(self.prctl.call_count, 4)

    def test_final_no_new_privileges_readback_failure_is_terminal(self):
        self.prctl.side_effect = [0, 1, 0, 2, 0]
        self.failed("kernel_policy_readback_rejected")
        self.assertEqual(self.prctl.call_count, 5)

    def test_foreign_prctl_exception_is_opaque(self):
        self.prctl.side_effect = OSError("SECRET_EXCEPTION_TEXT")
        self.failed("kernel_policy_no_new_privileges_rejected")

    def test_invalid_fd_and_libc_binding_failure_do_not_change_limits(self):
        with self.assertRaises(policy.PolicyError):
            policy.install(True)
        self.getlimit.assert_not_called()
        self.libc.side_effect = OSError("SECRET_EXCEPTION_TEXT")
        self.failed("kernel_policy_configuration_rejected")
        self.getlimit.assert_not_called()
        self.setlimit.assert_not_called()

    def test_unsupported_abi_precedes_libc_and_limits(self):
        with patch.object(policy, "require_abi", side_effect=(
            policy.PolicyError("kernel_policy_unavailable")
        )):
            self.failed("kernel_policy_unavailable")
        self.libc.assert_not_called()
        self.getlimit.assert_not_called()
        self.setlimit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
