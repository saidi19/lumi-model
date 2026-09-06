"""Fixed Linux x86-64 kernel policy for the inert IPC fixture, never real SFT.

This standalone stdlib source is authenticated before use in the child. Import
and ``require_abi`` do not install a filter or change resource limits. ``install``
is irreversible and is only for an owned fresh child after its fixed bootstrap.
There is no caller-supplied profile, fallback, allowlist or policy override.

Limits apply to one process, not a cgroup, aggregate RSS, GPU memory or billing.
Seccomp constrains syscalls, not namespaces or arbitrary hostile-code admission.
The inherited IPC peer is trusted; recvmsg still requires descriptor-rights
rejection and cleanup by the unchanged application protocol.
"""

from __future__ import annotations

import ctypes
import errno
import os
import resource
import struct
import sys

LIMITS = {
    resource.RLIMIT_AS: (134_217_728, 134_217_728),
    resource.RLIMIT_CPU: (2, 2),
    resource.RLIMIT_CORE: (0, 0),
    resource.RLIMIT_FSIZE: (0, 0),
    resource.RLIMIT_NOFILE: (64, 64),
    resource.RLIMIT_NPROC: (1, 1),
}

# Native x86-64 numbers, from Linux v6.8 arch/x86/entry/syscalls/syscall_64.tbl.
# Listed denied operations are identifiers for fixed tests, not an allowlist.
SYSCALLS = {
    "read": 0, "write": 1, "open": 2, "close": 3, "poll": 7,
    "mmap": 9, "mprotect": 10, "munmap": 11, "brk": 12,
    "rt_sigaction": 13, "rt_sigprocmask": 14, "rt_sigreturn": 15, "ioctl": 16,
    "mremap": 25, "madvise": 28, "dup": 32, "dup2": 33, "getpid": 39,
    "socket": 41, "connect": 42, "accept": 43, "sendto": 44, "recvfrom": 45,
    "sendmsg": 46, "recvmsg": 47, "bind": 49, "listen": 50, "socketpair": 53,
    "setsockopt": 54, "getsockopt": 55, "clone": 56, "fork": 57, "vfork": 58,
    "execve": 59, "exit": 60, "kill": 62, "fcntl": 72, "ptrace": 101,
    "getuid": 102, "getgid": 104, "geteuid": 107, "getegid": 108,
    "getppid": 110, "sigaltstack": 131, "prctl": 157, "setrlimit": 160,
    "futex": 202, "clock_gettime": 228, "clock_getres": 229, "exit_group": 231,
    "openat": 257, "unshare": 272, "dup3": 292, "prlimit64": 302,
    "setns": 308, "seccomp": 317, "bpf": 321, "execveat": 322,
    "io_uring_setup": 425, "io_uring_enter": 426, "io_uring_register": 427,
    "clone3": 435, "close_range": 436, "openat2": 437, "pidfd_getfd": 438,
}
UNCONDITIONAL_SYSCALLS = (
    "close", "brk", "munmap", "mremap", "madvise", "futex", "poll",
    "clock_gettime", "clock_getres", "getpid", "getppid", "getuid", "geteuid",
    "getgid", "getegid", "rt_sigreturn", "rt_sigaction", "rt_sigprocmask",
    "sigaltstack", "exit", "exit_group",
)
AUDIT_ARCH_X86_64 = 0xC000003E
X32_SYSCALL_BIT = 0x40000000
SECCOMP_RET_KILL_PROCESS = 0x80000000
SECCOMP_RET_ALLOW = 0x7FFF0000
SECCOMP_RET_ERRNO = 0x00050000
DENY = SECCOMP_RET_ERRNO | errno.EPERM
PR_SET_NO_NEW_PRIVS = 38
PR_GET_NO_NEW_PRIVS = 39
PR_SET_SECCOMP = 22
PR_GET_SECCOMP = 21
SECCOMP_MODE_FILTER = 2

ERROR_CODES = frozenset({
    "kernel_policy_unavailable", "kernel_policy_configuration_rejected",
    "kernel_policy_limits_rejected", "kernel_policy_no_new_privileges_rejected",
    "kernel_policy_installation_rejected", "kernel_policy_readback_rejected",
})


class PolicyError(ValueError):
    """Fixed stage code only; never preserve platform, process or exception text."""

    def __init__(self, code):
        self.code = code if type(code) is str and code in ERROR_CODES else (
            "kernel_policy_configuration_rejected"
        )
        super().__init__(self.code)


def _need(condition, code="kernel_policy_configuration_rejected"):
    if not condition:
        raise PolicyError(code) from None


def require_abi():
    """Read-only strict ABI/runtime gate; no resource or filter installation."""
    try:
        _need(
            sys.platform == "linux" and os.uname().machine == "x86_64"
            and sys.byteorder == "little" and struct.calcsize("P") == 8
            and struct.calcsize("l") == 8 and ctypes.sizeof(ctypes.c_void_p) == 8
            and ctypes.sizeof(ctypes.c_long) == 8 and ctypes.sizeof(ctypes.c_uint) == 4
            and ctypes.sizeof(ctypes.c_ushort) == 2
            and sys.implementation.name == "cpython"
            and sys.version_info[:2] in {(3, 12), (3, 13)}
            and sys.flags.no_site == 1 and sys.dont_write_bytecode
            and not sys.flags.optimize
            and os.getuid() > 0 and os.getgid() > 0
            and os.getresuid() == (os.getuid(),) * 3
            and os.getresgid() == (os.getgid(),) * 3,
            "kernel_policy_unavailable",
        )
    except BaseException:
        raise PolicyError("kernel_policy_unavailable") from None


def _load(offset):
    return (0x20, 0, 0, offset)  # BPF_LD | BPF_W | BPF_ABS


def _equal(offset, value):
    return [_load(offset), (0x15, 1, 0, value), (0x06, 0, 0, DENY)]


def _one_of(offset, values):
    instructions = [_load(offset)]
    for position, value in enumerate(values):
        instructions.append((0x15, len(values) - position, 0, value))
    instructions.append((0x06, 0, 0, DENY))
    return instructions


def _arg_equal(index, value):
    _need(type(value) is int and 0 <= value <= 0xFFFFFFFFFFFFFFFF)
    return _equal(20 + 8 * index, value >> 32) + _equal(
        16 + 8 * index, value & 0xFFFFFFFF,
    )


def _arg_one_of(index, values):
    return _equal(20 + 8 * index, 0) + _one_of(16 + 8 * index, values)


def _protection(index):
    # PROT_NONE, READ, WRITE, or READ|WRITE; no EXEC or additional flags.
    return _arg_one_of(index, (0, 1, 2, 3))


def compile_filter(ipc_fd):
    """Return immutable native-ABI cBPF instructions for the one fixed profile.

    Architecture and x32 rejection precede every syscall decision. Every
    restricted integer argument checks its high word too; no arbitrary aliases.
    Pointer arguments are not inspected. Kernel validation remains authoritative.
    """
    _need(type(ipc_fd) is int and 3 <= ipc_fd <= 1_048_576)
    instructions = [
        _load(4), (0x15, 1, 0, AUDIT_ARCH_X86_64),
        (0x06, 0, 0, SECCOMP_RET_KILL_PROCESS),
        _load(0), (0x45, 0, 1, X32_SYSCALL_BIT),
        (0x06, 0, 0, SECCOMP_RET_KILL_PROCESS),
    ]
    bodies = {
        "mmap": (
            _protection(2) + _arg_equal(3, 0x22)
            # Native fd is an int: accept only canonical zero/sign extension of -1.
            + _equal(48, 0xFFFFFFFF) + _one_of(52, (0, 0xFFFFFFFF)) + _arg_equal(5, 0)
        ),
        "mprotect": _protection(2),
        "sendmsg": _arg_equal(0, ipc_fd),
        "recvmsg": _arg_equal(0, ipc_fd),
        "ioctl": _arg_equal(0, ipc_fd) + _arg_equal(1, 0x5421),
        "getsockopt": (
            _arg_equal(0, ipc_fd) + _arg_equal(1, 1) + _arg_one_of(2, (3, 4, 16))
        ),
        "prctl": (
            _arg_one_of(0, (PR_GET_SECCOMP, PR_GET_NO_NEW_PRIVS))
            + sum((_arg_equal(index, 0) for index in range(1, 5)), [])
        ),
    }
    for name, body in bodies.items():
        body = [*body, (0x06, 0, 0, SECCOMP_RET_ALLOW)]
        _need(len(body) <= 255)
        instructions.extend([(0x15, 0, len(body), SYSCALLS[name]), *body])
    for name in UNCONDITIONAL_SYSCALLS:
        instructions.extend([
            (0x15, 0, 1, SYSCALLS[name]), (0x06, 0, 0, SECCOMP_RET_ALLOW),
        ])
    instructions.append((0x06, 0, 0, DENY))
    _need(len(instructions) <= 4096)
    return tuple(instructions)


class _SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte),
                ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint)]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(_SockFilter))]


def _can_lower(current, required):
    return type(current) is int and (
        current == resource.RLIM_INFINITY or current >= required
    )


def install(ipc_fd):
    """Irreversibly install the exact profile in a fresh authenticated child.

    All inherited limits are checked before the first mutation. An inherited
    soft or hard limit below the exact profile is rejected, never raised. Partial
    installation failure is terminal: there is no rollback or alternate profile.
    The caller must exit; no token/READY may be released after any failure.
    """
    require_abi()
    stage = "kernel_policy_configuration_rejected"
    try:
        instructions = compile_filter(ipc_fd)
        storage = (_SockFilter * len(instructions))(*(
            _SockFilter(*instruction) for instruction in instructions
        ))
        program = _SockFprog(len(instructions), storage)
        _need(ctypes.sizeof(_SockFilter) == 8 and ctypes.sizeof(_SockFprog) == 16
              and _SockFprog.filter.offset == 8)
        libc = ctypes.CDLL(None, use_errno=True)
        prctl = libc.prctl
        prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
        prctl.restype = ctypes.c_int

        stage = "kernel_policy_limits_rejected"
        for name, exact in LIMITS.items():
            inherited = resource.getrlimit(name)
            _need(type(inherited) is tuple and len(inherited) == 2
                  and all(_can_lower(value, target)
                          for value, target in zip(inherited, exact, strict=True)), stage)
        for name, exact in LIMITS.items():
            resource.setrlimit(name, exact)
        for name, exact in LIMITS.items():
            observed = resource.getrlimit(name)
            _need(type(observed) is tuple and len(observed) == 2
                  and all(type(value) is int for value in observed) and observed == exact, stage)

        stage = "kernel_policy_no_new_privileges_rejected"
        _need(prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) == 0, stage)
        _need(prctl(PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0) == 1, stage)
        stage = "kernel_policy_installation_rejected"
        _need(prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER,
                    ctypes.addressof(program), 0, 0) == 0, stage)
        stage = "kernel_policy_readback_rejected"
        _need(prctl(PR_GET_SECCOMP, 0, 0, 0, 0) == SECCOMP_MODE_FILTER, stage)
        _need(prctl(PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0) == 1, stage)
    except BaseException:
        raise PolicyError(stage) from None
    return {
        "schema_version": "lumi.sft-kernel-policy-installation.v0.1",
        "policy_installed": True, "resource_limits_verified": 6,
        "no_new_privileges": True, "seccomp_filter_mode_verified": True,
        "cgroup_memory_enforcement": False, "namespace_containment": False,
        "real_sft_execution_permitted": False,
    }
