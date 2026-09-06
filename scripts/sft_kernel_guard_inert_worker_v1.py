"""Fixed kernel-guard fixture; no model, training, source data, or useful work.

The trusted parent supplies a separately hash-authenticated ``kernel_policy``
module. All imports and ctypes symbol resolution precede that policy's install.
Only two credential-checked messages may be received on the inherited endpoint.
The deliberately invalid challenge arguments remain inert if a filter regresses.
"""

import ctypes
import errno
import os
import resource
import socket
import struct
import sys
import time

READY = b"kernel-guard-ready-v1"
PROOF = b"kernel-guard-proof-v1"
GO = b"G"
RELEASE = b"K"
FAULTS = frozenset({"none", "exit_after_ready", "stall"})
MAXIMUM_MESSAGES = 2
MAXIMUM_PAYLOAD_BYTES = 64
MAXIMUM_SECONDS = 6
ADDRESS_SPACE_BYTES = 128 * 1024 * 1024
MEMORY_CHALLENGE_BYTES = 129 * 1024 * 1024
ALLOWED_MEMORY_BYTES = 4096
EXPECTED_DENIAL_CHECKS = 17
EXPECTED_MEMORY_CHECKS = 1
EXPECTED_ALLOWED_MEMORY_CHECKS = 1


def need(condition):
    if not condition:
        raise ValueError from None


def frame(payload):
    need(type(payload) is bytes and 0 < len(payload) <= MAXIMUM_PAYLOAD_BYTES)
    return struct.pack("!I", len(payload)) + payload


def close_rights(ancillary):
    """Close every delivered descriptor before rejecting any packet evidence."""
    clean = True
    seen = set()
    for item in ancillary:
        if type(item) not in (tuple, list) or len(item) != 3:
            clean = False
            continue
        if item[0] != socket.SOL_SOCKET or item[1] != socket.SCM_RIGHTS:
            continue
        data = item[2]
        if type(data) is not bytes:
            clean = False
            continue
        if len(data) % 4:
            clean = False
        for offset in range(0, len(data) - len(data) % 4, 4):
            descriptor = struct.unpack("i", data[offset:offset + 4])[0]
            if descriptor in seen:
                clean = False
                continue
            seen.add(descriptor)
            try:
                os.close(descriptor)
            except BaseException:
                clean = False
    return clean


class Channel:
    def __init__(self, stream, deadline, parent_credentials):
        self.stream = stream
        self.deadline = deadline
        self.parent_credentials = parent_credentials
        self.sent = 0
        self.received = 0
        self.ancillary_capacity = socket.CMSG_SPACE(12) + socket.CMSG_SPACE(32)

    def before_io(self):
        remaining = self.deadline - time.monotonic()
        need(remaining > 0)
        self.stream.settimeout(remaining)

    def send(self, payload):
        packet = frame(payload)
        need(self.sent < MAXIMUM_MESSAGES)
        self.sent += 1
        self.before_io()
        need(self.stream.sendmsg([packet], []) == len(packet))
        need(time.monotonic() < self.deadline)

    def receive(self):
        need(self.received < MAXIMUM_MESSAGES)
        self.received += 1
        self.before_io()
        raw, ancillary, flags, _address = self.stream.recvmsg(
            MAXIMUM_PAYLOAD_BYTES + 5, self.ancillary_capacity, socket.MSG_CMSG_CLOEXEC
        )
        clean = close_rights(ancillary)
        need(clean and time.monotonic() < self.deadline)
        need(type(flags) is int and flags in {0, socket.MSG_CMSG_CLOEXEC})
        need(type(ancillary) is list and len(ancillary) == 1)
        item = ancillary[0]
        need(type(item) is tuple and len(item) == 3)
        need(item[0] == socket.SOL_SOCKET and item[1] == socket.SCM_CREDENTIALS)
        need(type(item[2]) is bytes and len(item[2]) == 12)
        need(struct.unpack("iII", item[2]) == self.parent_credentials)
        need(type(raw) is bytes and 4 < len(raw) <= MAXIMUM_PAYLOAD_BYTES + 4)
        need(int.from_bytes(raw[:4], "big") == len(raw) - 4)
        need(time.monotonic() < self.deadline)
        return raw[4:]


class Challenges:
    """Resolve libc before confinement; perform only fixed, bounded challenges."""

    def __init__(self, policy):
        self.policy = policy
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.syscall = self.libc.syscall
        self.syscall.argtypes = [ctypes.c_long] * 7
        self.syscall.restype = ctypes.c_long
        self.mmap = self.libc.mmap
        self.mmap.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_long,
        ]
        self.mmap.restype = ctypes.c_void_p
        self.munmap = self.libc.munmap
        self.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.munmap.restype = ctypes.c_int
        self.empty_path = ctypes.create_string_buffer(b"")
        empty = ctypes.addressof(self.empty_path)
        self.map_failed = ctypes.c_void_p(-1).value
        specifications = (
            ("socket", (socket.AF_INET, socket.SOCK_STREAM, 0), True),
            ("socket", (socket.AF_INET6, socket.SOCK_STREAM, 0), True),
            ("socket", (socket.AF_UNIX, socket.SOCK_STREAM, 0), True),
            ("connect", (-1, 0, 0), False),
            ("openat", (-100, empty, 0, 0), False),
            ("clone3", (0, 0), False),
            ("execve", (empty, 0, 0), False),
            ("unshare", (0,), False),
            ("setns", (-1, 0), False),
            ("ptrace", (-1, 0, 0, 0), False),
            ("prlimit64", (0, resource.RLIMIT_AS, 0, 0), False),
            ("io_uring_setup", (0, 0), False),
            ("sendmsg", (-1, 0, 0), False),
            ("recvmsg", (-1, 0, 0), False),
            ("read", (-1, 0, 0), False),
            ("write", (-1, 0, 0), False),
            ("dup3", (-1, -1, 0), False),
        )
        need(len(specifications) == EXPECTED_DENIAL_CHECKS)
        self.calls = tuple(
            (policy.SYSCALLS[name], arguments + (0,) * (6 - len(arguments)), may_open)
            for name, arguments, may_open in specifications
        )

    def run(self):
        for number, arguments, may_open in self.calls:
            ctypes.set_errno(0)
            result = self.syscall(number, *arguments)
            error = ctypes.get_errno()
            if may_open and result >= 0:
                # A broken socket filter must not leak its newly created endpoint.
                os.close(result)
            need(result == -1 and error == errno.EPERM)
        # Verify one ordinary private allocation is permitted; do not touch it.
        address = self.mmap(0, ALLOWED_MEMORY_BYTES, 3, 0x22, -1, 0)
        need(address != self.map_failed)
        need(self.munmap(address, ALLOWED_MEMORY_BYTES) == 0)
        ctypes.set_errno(0)
        # No page is touched. AS=128 MiB must reject this one 129 MiB mapping.
        address = self.mmap(0, MEMORY_CHALLENGE_BYTES, 3, 0x22, -1, 0)
        error = ctypes.get_errno()
        if address != self.map_failed:
            need(self.munmap(address, MEMORY_CHALLENGE_BYTES) == 0)
            raise ValueError from None
        need(error == errno.ENOMEM)


def conversation(channel, challenges, fault):
    channel.send(READY)
    if fault == "exit_after_ready":
        return 2
    need(channel.receive() == GO)
    if fault == "stall":
        channel.receive()
        return 2
    challenges.run()
    channel.send(PROOF)
    need(channel.receive() == RELEASE)
    return 0


def main():
    deadline = time.monotonic() + MAXIMUM_SECONDS
    try:
        arguments = sys.argv[1:]
        need(
            sys.platform == "linux" and sys.flags.no_site == 1
            and sys.dont_write_bytecode and not sys.flags.optimize
            and len(arguments) == 2 and arguments[1] in FAULTS
            and 1 <= len(arguments[0]) <= 7 and arguments[0].isascii()
            and arguments[0].isdecimal() and 3 <= int(arguments[0]) <= 1_048_576
            and struct.calcsize("iII") == 12 and struct.calcsize("i") == 4
        )
        policy = globals()["kernel_policy"]
        policy.require_abi()
        need(policy.LIMITS[resource.RLIMIT_AS] == (ADDRESS_SPACE_BYTES, ADDRESS_SPACE_BYTES))
        challenges = Challenges(policy)
        parent_credentials = (os.getppid(), os.getuid(), os.getgid())
        need(parent_credentials[0] > 1)
        with socket.socket(fileno=int(arguments[0])) as stream:
            need(
                stream.family == socket.AF_UNIX
                and stream.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) == socket.SOCK_SEQPACKET
                and stream.getsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED) == 1
            )
            stream.set_inheritable(False)
            channel = Channel(stream, deadline, parent_credentials)
            installed = policy.install(stream.fileno())
            need(
                type(installed) is dict and installed.get("policy_installed") is True
                and installed.get("resource_limits_verified") == 6
                and installed.get("no_new_privileges") is True
                and installed.get("seccomp_filter_mode_verified") is True
            )
            return conversation(channel, challenges, arguments[1])
    except BaseException:
        return 2


if __name__ == "__main__":
    os._exit(main())
