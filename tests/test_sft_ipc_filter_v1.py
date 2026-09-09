"""Pure bytecode interpretation and syscall doubles; no live seccomp or ML."""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import struct
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_lumi_synthetic_ipc_filter_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
before_import = set(sys.modules)
ipc = importlib.import_module(PACKAGE + ".posttraining_sft_ipc_filter_v1")
IMPORT_ADDITIONS = set(sys.modules) - before_import
PRIVATE = "SYNTHETIC-PRIVATE-EXCEPTION"


def evaluate(number, arguments=(), *, arch=0xC000003E, instructions=None):
    """Independent tiny cBPF VM over real seccomp_data layout, with a step cap."""
    args = tuple(arguments) + (0,) * (6 - len(arguments))
    raw = struct.pack("<IIQ6Q", number & 0xFFFFFFFF, arch, 0, *args)
    program = ipc.FILTER if instructions is None else instructions
    pc, accumulator = 0, 0
    for _ in range(256):
        if not 0 <= pc < len(program):
            raise AssertionError("invalid_program_counter")
        code, yes, no, value = program[pc]
        if code == 0x20:
            if value % 4 or not 0 <= value <= len(raw) - 4:
                raise AssertionError("invalid_load")
            accumulator = struct.unpack_from("<I", raw, value)[0]
        elif code == 0x15:
            pc += yes if accumulator == value else no
        elif code == 0x35:
            pc += yes if accumulator >= value else no
        elif code == 0x05:
            pc += value
        elif code == 0x54:
            accumulator &= value
        elif code == 0x06:
            return value
        else:
            raise AssertionError("unsupported_opcode")
        pc += 1
    raise AssertionError("unterminated_program")


def expected_socket(family, kind, protocol):
    if any(value >> 32 for value in (family, kind, protocol)):
        return ipc.DENY
    base = kind & ~0x80800
    if family in (2, 10):
        accepted = ((base == 1 and protocol in (0, 6))
                    or (base == 2 and protocol in (0, 17)))
    else:
        accepted = family == 16 and base in (2, 3) and protocol == 0
    return ipc.ALLOW if accepted else ipc.DENY


class FilterProgramTests(unittest.TestCase):
    def test_deterministic_immutable_program_and_digest(self):
        self.assertIs(type(ipc.FILTER), tuple)
        self.assertTrue(all(type(row) is tuple for row in ipc.FILTER))
        self.assertEqual(ipc.FILTER, ipc.build_filter())
        encoded = b"".join(struct.pack("<HBBI", *row) for row in ipc.FILTER)
        self.assertEqual(ipc.FILTER_SHA256, hashlib.sha256(encoded).hexdigest())
        self.assertEqual(ipc.RECEIPT["filter_sha256"], ipc.FILTER_SHA256)
        self.assertEqual(ipc.RECEIPT["instruction_count"], len(ipc.FILTER))
        self.assertLessEqual(len(ipc.FILTER), 128)

    def test_all_branches_are_forward_bounded_and_terminate(self):
        for index, (code, yes, no, value) in enumerate(ipc.FILTER):
            with self.subTest(index=index):
                if code in (0x15, 0x35):
                    self.assertTrue(0 <= yes <= 255 and 0 <= no <= 255)
                    self.assertLess(index + 1 + max(yes, no), len(ipc.FILTER))
                elif code == 0x05:
                    self.assertGreaterEqual(value, 0)
                    self.assertLess(index + value + 1, len(ipc.FILTER))
                else:
                    self.assertEqual((yes, no), (0, 0))
                self.assertIn(code, (0x20, 0x15, 0x35, 0x05, 0x54, 0x06))
        self.assertEqual(ipc.FILTER[-1], (0x06, 0, 0, ipc.ALLOW))

    def test_foreign_architecture_is_killed_before_syscall_dispatch(self):
        for architecture in (0, 3, 0x40000003, 0x8000003E, 0xC00000B7, 0xFFFFFFFF):
            for number in (0, 41, 53, 102, 317, 425, 0xFFFFFFFF):
                with self.subTest(architecture=architecture, number=number):
                    self.assertEqual(evaluate(number, arch=architecture), ipc.KILL)

    def test_x32_bit_negative_and_historical_dispatches_are_killed(self):
        for number in (0x40000000, 0x40000029, 0x40000035, 0x7FFFFFFF,
                       0x80000000, 0xFFFFFFFF, -1, *range(512, 548)):
            with self.subTest(number=number):
                self.assertEqual(evaluate(number, (2, 1, 0)), ipc.KILL)
        for number in (511, 548, 0x3FFFFFFF):
            self.assertEqual(evaluate(number), ipc.ALLOW)

    def test_io_uring_ptrace_and_pidfd_getfd_are_unconditionally_denied(self):
        self.assertEqual(ipc.DENIED_SYSCALLS, (101, 425, 426, 427, 438))
        for number in ipc.DENIED_SYSCALLS:
            for args in ((0,) * 6, (0xFFFFFFFFFFFFFFFF,) * 6):
                with self.subTest(number=number, arguments=args):
                    self.assertEqual(evaluate(number, args), ipc.DENY)

    def test_unrelated_syscalls_remain_outside_narrow_policy(self):
        # x86_64 102 is getuid, not the i386 socketcall multiplexor. The latter
        # is rejected by the architecture gate, not by breaking native getuid.
        for number in (0, 1, 2, 3, 9, 16, 39, 42, 43, 44, 45, 46, 47, 49,
                       50, 51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 102, 126,
                       157, 202, 231, 257, 288, 292, 293, 317, 424, 435):
            with self.subTest(number=number):
                self.assertEqual(evaluate(number, (0xFFFFFFFFFFFFFFFF,) * 6), ipc.ALLOW)

    def test_socket_family_type_protocol_matrix_matches_independent_spec(self):
        for family in range(48):
            for kind in range(16):
                for flags in (0, 0x800, 0x80000, 0x80800):
                    for protocol in (0, 6, 17, 255):
                        args = family, kind | flags, protocol
                        self.assertEqual(evaluate(41, args), expected_socket(*args), args)

    def test_new_unix_sockets_always_denied_including_raw_and_flag_aliases(self):
        for kind in (0, 1, 2, 3, 5, 10, 0x801, 0x80001, 0x80805, 0xFFFFFFFF):
            for protocol in (0, 1, 6, 17, 0xFFFFFFFF):
                with self.subTest(kind=kind, protocol=protocol):
                    self.assertEqual(evaluate(41, (1, kind, protocol)), ipc.DENY)

    def test_socketpair_exact_connected_unix_only_matrix(self):
        for family in (0, 1, 2, 10, 16, 38, 40, 0xFFFFFFFF):
            for kind in range(16):
                for flags in (0, 0x800, 0x80000, 0x80800):
                    for protocol in (0, 1, 6, 17, 255):
                        wanted = (ipc.ALLOW if family == 1 and kind in (1, 5)
                                  and protocol == 0 else ipc.DENY)
                        self.assertEqual(evaluate(53, (family, kind | flags, protocol)), wanted)

    def test_datagram_socketpairs_cannot_supply_retargetable_unix_authority(self):
        for flags in (0, 0x800, 0x80000, 0x80800):
            self.assertEqual(evaluate(53, (1, 2 | flags, 0)), ipc.DENY)

    def test_full_width_integer_arguments_cannot_hide_upper_bits(self):
        for number, triple in ((41, (2, 1, 0)), (41, (10, 2, 17)),
                               (41, (16, 3, 0)), (53, (1, 1, 0)), (53, (1, 5, 0))):
            for position in range(3):
                for high in (1, 0x80000000, 0xFFFFFFFF):
                    values = list(triple)
                    values[position] |= high << 32
                    with self.subTest(number=number, values=values):
                        self.assertEqual(evaluate(number, values), ipc.DENY)

    def test_unknown_flags_and_negative_int_patterns_are_denied(self):
        for bad in (0x10, 0x100, 0x400, 0x1000, 0x40000, 0x100000, 0x80000000):
            self.assertEqual(evaluate(41, (2, 1 | bad, 0)), ipc.DENY)
            self.assertEqual(evaluate(53, (1, 1 | bad, 0)), ipc.DENY)
        for number in (41, 53):
            for position in range(3):
                values = [1 if number == 53 else 2, 1, 0]
                values[position] = 0xFFFFFFFF
                self.assertEqual(evaluate(number, values), ipc.DENY)

    def test_socketpair_output_pointer_is_not_mistaken_for_an_integer_policy_argument(self):
        for pointer in (0, 0x80000000, 0x7FFFFFFFF000, 0xFFFFFFFFFFFFFFFF):
            self.assertEqual(evaluate(53, (1, 1, 0, pointer)), ipc.ALLOW)

    def test_receipt_keeps_out_of_scope_claims_false(self):
        self.assertEqual(ipc.RECEIPT["schema"], "lumi.sft-ipc-filter.v0.1")
        for key in ("inherited_authority_closed", "all_ipc_isolated",
                    "complete_os_sandbox_verified", "real_sft_execution_permitted",
                    "production_eligible"):
            self.assertIs(ipc.RECEIPT[key], False)

    def test_import_does_not_load_ml_or_call_kernel(self):
        forbidden = {"torch", "numpy", "transformers", "safetensors", "tokenizers",
                     "tensorflow", "jax", "cupy"}
        self.assertFalse({name.split(".", 1)[0] for name in IMPORT_ADDITIONS} & forbidden)


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.runtime = Mock(side_effect=lambda: self.events.append("runtime"))
        self.result, self.nnp, self.mode = 0, 1, 2
        self.rows = None
        for target, value in (("_runtime", self.runtime), ("_ATTEMPT_OWNER", None),
                              ("_prctl", self.prctl), ("_install", self.install)):
            patcher = patch.object(ipc, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def prctl(self, option, value=0):
        self.events.append(("prctl", option, value))
        return {38: 0, 39: self.nnp, 21: self.mode}[option]

    def install(self, program):
        self.events.append("install")
        self.assertIs(type(program), ipc.SockFprog)
        self.assertEqual(program.length, len(ipc.FILTER))
        self.rows = tuple((program.filter[index].code, program.filter[index].jt,
                           program.filter[index].jf, program.filter[index].k)
                          for index in range(program.length))
        return self.result

    def failure(self, code):
        with self.assertRaises(ipc.IpcFilterError) as caught:
            ipc.install_ipc_filter()
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(str(caught.exception), code)
        self.assertNotIn(PRIVATE, str(caught.exception))

    def test_success_installs_exact_bytes_and_returns_separate_fixed_receipt(self):
        receipt = ipc.install_ipc_filter()
        self.assertEqual(self.rows, ipc.FILTER)
        self.assertEqual(receipt, ipc.RECEIPT)
        self.assertIsNot(receipt, ipc.RECEIPT)
        self.assertEqual(self.events, ["runtime", ("prctl", 38, 1),
                                      ("prctl", 39, 0), "install", ("prctl", 21, 0)])

    def test_success_cannot_be_repeated(self):
        ipc.install_ipc_filter()
        self.failure("already_attempted")
        self.assertEqual(self.events.count("install"), 1)

    def test_install_failure_is_sticky_and_never_read_back_as_success(self):
        self.result = -1
        self.failure("install_failed")
        self.result = 0
        self.failure("already_attempted")
        self.assertNotIn(("prctl", 21, 0), self.events)
        self.assertEqual(self.events.count("install"), 1)

    def test_install_exception_is_fixed_and_sticky(self):
        with patch.object(ipc, "_install", side_effect=OSError(PRIVATE)):
            self.failure("setup_failed")
        self.failure("already_attempted")

    def test_runtime_rejection_has_no_native_side_effect(self):
        self.runtime.side_effect = OSError(PRIVATE)
        self.failure("runtime_rejected")
        self.assertEqual(self.events, [])
        self.assertIsNone(ipc._ATTEMPT_OWNER)

    def test_bad_native_layout_rejected_before_start(self):
        original = ctypes.sizeof
        with patch.object(ipc.ctypes, "sizeof", side_effect=lambda value:
                          12 if value is ipc.SockFprog else original(value)):
            self.failure("runtime_rejected")
        self.assertEqual(self.events, ["runtime"])
        self.assertIsNone(ipc._ATTEMPT_OWNER)

    def test_changed_program_rejected_before_start(self):
        with patch.object(ipc, "FILTER", ((ipc.RET_K, 0, 0, ipc.ALLOW),)):
            self.failure("program_rejected")
        self.assertEqual(self.events, ["runtime"])
        self.assertIsNone(ipc._ATTEMPT_OWNER)

    def test_nnp_set_failure_and_exception_prevent_install(self):
        with patch.object(ipc, "_prctl", return_value=-1):
            self.failure("privilege_rejected")
        self.assertNotIn("install", self.events)
        self.failure("already_attempted")

    def test_nnp_readback_failure_is_sticky(self):
        self.nnp = 0
        self.failure("privilege_rejected")
        self.assertNotIn("install", self.events)
        self.failure("already_attempted")

    def test_seccomp_mode_readback_failure_is_sticky(self):
        self.mode = 0
        self.failure("readback_failed")
        self.failure("already_attempted")
        self.assertEqual(self.events.count("install"), 1)

    def test_owner_change_rejected_before_rechecking_runtime(self):
        ipc.install_ipc_filter()
        with patch.object(ipc.os, "getpid", return_value=ipc._ATTEMPT_OWNER[0] + 1):
            self.failure("owner_mismatch")
        self.assertEqual(self.runtime.call_count, 1)

    def test_thread_owner_change_rejected(self):
        ipc.install_ipc_filter()
        with patch.object(ipc.threading, "get_ident", return_value=ipc._ATTEMPT_OWNER[1] + 1):
            self.failure("owner_mismatch")
        self.assertEqual(self.runtime.call_count, 1)

    def test_reentry_does_not_release_another_callers_lock(self):
        self.assertTrue(ipc._SETUP_LOCK.acquire(blocking=False))
        try:
            self.failure("reentry")
            self.assertTrue(ipc._SETUP_LOCK.locked())
        finally:
            ipc._SETUP_LOCK.release()
        self.assertEqual(self.events, [])

    def test_lock_released_after_native_failure(self):
        self.result = -1
        self.failure("install_failed")
        self.assertFalse(ipc._SETUP_LOCK.locked())

    def test_native_prctl_signature_is_width_safe(self):
        # Call the original through its module code globals with an entirely
        # synthetic libc. No OS syscall is made on this or any test path.
        fake_call = Mock(return_value=1)
        fake_libc = types.SimpleNamespace(prctl=fake_call)
        original = self._get_original("_prctl")
        with patch.object(ipc.ctypes, "CDLL", return_value=fake_libc):
            self.assertEqual(original(39), 1)
        self.assertEqual(fake_call.argtypes,
                         [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                          ctypes.c_ulong, ctypes.c_ulong])
        self.assertIs(fake_call.restype, ctypes.c_int)
        fake_call.assert_called_once_with(39, 0, 0, 0, 0)

    def test_native_seccomp_call_uses_no_listener_or_tsync_and_keeps_pointer_width(self):
        fake_call = Mock(return_value=0)
        fake_libc = types.SimpleNamespace(syscall=fake_call)
        original = self._get_original("_install")
        rows = (ipc.SockFilter * len(ipc.FILTER))(*(ipc.SockFilter(*row) for row in ipc.FILTER))
        program = ipc.SockFprog(len(rows), rows)
        with patch.object(ipc.ctypes, "CDLL", return_value=fake_libc):
            self.assertEqual(original(program), 0)
        self.assertEqual(fake_call.argtypes, [ctypes.c_long])
        self.assertIs(fake_call.restype, ctypes.c_long)
        arguments = fake_call.call_args.args
        self.assertEqual(arguments[0], 317)
        self.assertIs(type(arguments[1]), ctypes.c_uint)
        self.assertIs(type(arguments[2]), ctypes.c_uint)
        self.assertEqual((arguments[1].value, arguments[2].value), (1, 0))
        self.assertIs(arguments[3]._obj, program)

    @staticmethod
    def _get_original(name):
        return ORIGINALS[name]


ORIGINALS = {"_prctl": ipc._prctl, "_install": ipc._install}


if __name__ == "__main__":
    unittest.main()
