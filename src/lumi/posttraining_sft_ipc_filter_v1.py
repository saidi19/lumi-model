"""One root-bootstrap-only, narrow socket/IPC seccomp policy.

Install after the owned network namespace, private mount root and Landlock
policy, before permanent privilege drop and payload release. All inherited
authority, including io_uring mappings and descriptors, must be separately
excluded. Retained UNIX control pairs must have trusted peers that never import
outside descriptors. This filter cannot inspect pointer contents or revoke an
existing descriptor; it is not a whole-syscall or complete IPC sandbox.

New UNIX sockets are denied. Only connected UNIX STREAM/SEQPACKET socketpairs
remain creatable; datagram pairs can retarget a pathname and are forbidden.
INET TCP/UDP and NETLINK_ROUTE sockets remain within the owned network namespace.
io_uring creation/submission/registration, pidfd_getfd and ptrace are denied.
Kernel seccomp inheritance, not Python state, constrains fork and exec.

Setup is one-shot, including native failures. Log only fixed codes/receipt;
suppressed exception context may retain private OS text and must not be exposed.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import struct
import threading

from .posttraining_sft_netns_v1 import _runtime

SCHEMA = "lumi.sft-ipc-filter.v0.1"
ERROR_CODES = frozenset({
    "runtime_rejected", "already_attempted", "owner_mismatch", "reentry",
    "program_rejected", "privilege_rejected", "install_failed", "readback_failed",
    "setup_failed",
})
AUDIT_ARCH_X86_64 = 0xC000003E
SYS_SOCKET, SYS_SOCKETPAIR, SYS_PTRACE, SYS_SECCOMP = 41, 53, 101, 317
DENIED_SYSCALLS = (SYS_PTRACE, 425, 426, 427, 438)
AF_UNIX, AF_INET, AF_INET6, AF_NETLINK = 1, 2, 10, 16
SOCK_STREAM, SOCK_DGRAM, SOCK_RAW, SOCK_SEQPACKET = 1, 2, 3, 5
SOCK_NONBLOCK, SOCK_CLOEXEC = 0x800, 0x80000
SOCKET_FLAGS = SOCK_NONBLOCK | SOCK_CLOEXEC
LD_ABS, JMP_JA, JMP_JEQ, JMP_JGE, ALU_AND, RET_K = 0x20, 0x05, 0x15, 0x35, 0x54, 0x06
ALLOW, DENY, KILL = 0x7FFF0000, 0x00050001, 0x80000000
_SETUP_LOCK = threading.Lock()
_ATTEMPT_OWNER = None


class IpcFilterError(RuntimeError):
    def __init__(self, code="setup_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "setup_failed"
        super().__init__(self.code)


class SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte),
               ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint32)]


class SockFprog(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ushort), ("filter", ctypes.POINTER(SockFilter))]


def _need(value, code):
    if not value:
        raise IpcFilterError(code) from None


def build_filter():
    """Return deterministic immutable classic-BPF instructions; no native calls."""
    rows, labels = [], {}

    def label(name):
        _need(name not in labels, "program_rejected")
        labels[name] = len(rows)

    def emit(code, k=0, yes=0, no=0):
        rows.append((code, yes, no, k))

    def eq(value, yes=0, no=0):
        emit(JMP_JEQ, value, yes, no)

    def argument(index):
        # seccomp_data stores full 64-bit arguments. Kernel int truncation must
        # never make an unchecked upper word look like a permitted family/type.
        emit(LD_ABS, 16 + index * 8 + 4)
        eq(0, no="deny")
        emit(LD_ABS, 16 + index * 8)

    def socket_type():
        argument(1)
        emit(ALU_AND, ~SOCKET_FLAGS & 0xFFFFFFFF)

    emit(LD_ABS, 4)
    eq(AUDIT_ARCH_X86_64, no="kill")
    emit(LD_ABS, 0)
    # x32 shares AUDIT_ARCH_X86_64. Also reject negative syscall numbers and
    # historical non-bit-tagged x32 numbers 512..547 on older x86 kernels.
    emit(JMP_JGE, 0x40000000, yes="kill")
    emit(JMP_JGE, 512, yes="historical_x32", no="dispatch")
    label("historical_x32")
    emit(JMP_JGE, 548, yes="dispatch", no="kill")
    label("dispatch")
    for number in DENIED_SYSCALLS:
        eq(number, yes="deny")
    eq(SYS_SOCKET, yes="socket")
    eq(SYS_SOCKETPAIR, yes="pair")
    emit(JMP_JA, "allow")

    label("socket")
    argument(0)
    eq(AF_INET, yes="inet")
    eq(AF_INET6, yes="inet")
    eq(AF_NETLINK, yes="netlink", no="deny")
    label("inet")
    socket_type()
    eq(SOCK_STREAM, yes="tcp")
    eq(SOCK_DGRAM, yes="udp", no="deny")
    label("tcp")
    argument(2)
    eq(0, yes="allow")
    eq(6, yes="allow", no="deny")
    label("udp")
    argument(2)
    eq(0, yes="allow")
    eq(17, yes="allow", no="deny")
    label("netlink")
    socket_type()
    eq(SOCK_RAW, yes="zero_protocol")
    eq(SOCK_DGRAM, yes="zero_protocol", no="deny")

    label("pair")
    argument(0)
    eq(AF_UNIX, no="deny")
    socket_type()
    eq(SOCK_STREAM, yes="zero_protocol")
    eq(SOCK_SEQPACKET, no="deny")
    label("zero_protocol")
    argument(2)
    eq(0, yes="allow", no="deny")
    label("kill")
    emit(RET_K, KILL)
    label("deny")
    emit(RET_K, DENY)
    label("allow")
    emit(RET_K, ALLOW)

    def offset(target, index):
        result = labels[target] - index - 1 if type(target) is str else target
        _need(type(result) is int and 0 <= result <= 255, "program_rejected")
        return result

    instructions = []
    for index, (code, yes, no, value) in enumerate(rows):
        if code == JMP_JA:
            value = offset(value, index)
        yes, no = offset(yes, index), offset(no, index)
        _need(type(value) is int and 0 <= value <= 0xFFFFFFFF, "program_rejected")
        instructions.append((code, yes, no, value))
    _need(0 < len(instructions) <= 128, "program_rejected")
    return tuple(instructions)


FILTER = build_filter()
FILTER_SHA256 = hashlib.sha256(b"".join(struct.pack("<HBBI", *row) for row in FILTER)).hexdigest()
RECEIPT = {
    "schema": SCHEMA, "filter_sha256": FILTER_SHA256, "instruction_count": len(FILTER),
    "seccomp_filter_installed": True, "native_x86_64_only": True,
    "unix_socket_creation_denied": True, "unix_stream_seqpacket_pairs_only": True,
    "inet_tcp_udp_and_route_netlink_only": True, "io_uring_denied": True,
    "pidfd_getfd_denied": True, "ptrace_denied": True, "inherits_fork_exec": True,
    "inherited_authority_closed": False, "all_ipc_isolated": False,
    "complete_os_sandbox_verified": False, "real_sft_execution_permitted": False,
    "production_eligible": False,
}


def _prctl(option, value=0):
    call = ctypes.CDLL(None, use_errno=True).prctl
    call.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                     ctypes.c_ulong, ctypes.c_ulong]
    call.restype = ctypes.c_int
    return call(option, value, 0, 0, 0)


def _install(program):
    call = ctypes.CDLL(None, use_errno=True).syscall
    call.argtypes, call.restype = [ctypes.c_long], ctypes.c_long
    return call(SYS_SECCOMP, ctypes.c_uint(1), ctypes.c_uint(0), ctypes.byref(program))


def install_ipc_filter():
    """Install the exact immutable policy once. Any started failure requires exit."""
    global _ATTEMPT_OWNER
    _need(_SETUP_LOCK.acquire(blocking=False), "reentry")
    failure = None
    try:
        owner = os.getpid(), threading.get_ident()
        if _ATTEMPT_OWNER is not None:
            _need(_ATTEMPT_OWNER == owner, "owner_mismatch")
            raise IpcFilterError("already_attempted") from None
        try:
            _runtime()
        except BaseException:
            raise IpcFilterError("runtime_rejected") from None
        _need(ctypes.sizeof(SockFilter) == 8 and SockFilter.k.offset == 4
              and ctypes.sizeof(SockFprog) == 16 and SockFprog.filter.offset == 8,
              "runtime_rejected")
        _need(FILTER == build_filter(), "program_rejected")
        rows = (SockFilter * len(FILTER))(*(SockFilter(*row) for row in FILTER))
        program = SockFprog(len(rows), rows)
        _ATTEMPT_OWNER = owner
        # No listeners, trace/user-notification escape hatch, TSYNC fallback,
        # policy negotiation or retry. _runtime authenticated one kernel thread.
        _need(_prctl(38, 1) == 0 and _prctl(39) == 1, "privilege_rejected")
        _need(_install(program) == 0, "install_failed")
        _need(_prctl(21) == 2, "readback_failed")
    except BaseException as error:
        failure = error.code if type(error) is IpcFilterError else "setup_failed"
    finally:
        _SETUP_LOCK.release()
    if failure is not None:
        raise IpcFilterError(failure) from None
    return dict(RECEIPT)
