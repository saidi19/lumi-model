"""Fixed Linux privilege/lifecycle fixture; no training, data or extra children."""

from __future__ import annotations

import ctypes
import socket
import struct
import sys

FAULTS = frozenset({"none", "wrong_parent", "exit_after_ready", "stall"})
READY = b"lifecycle-ready-v1"
PAYLOAD = b"lifecycle-inert-message-v1"


def no_new_privileges():
    # Fixed libc symbol and fixed argument types; no library lookup subprocess.
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    prctl.restype = ctypes.c_int
    return prctl(38, 1, 0, 0, 0) == 0 and prctl(39, 0, 0, 0, 0) == 1


def main():
    try:
        arguments = sys.argv[1:]
        if (
            sys.platform != "linux" or sys.flags.no_site != 1
            or not sys.dont_write_bytecode or sys.flags.optimize
            or len(arguments) != 2 or arguments[1] not in FAULTS
            or not 1 <= len(arguments[0]) <= 7 or not arguments[0].isascii()
            or not arguments[0].isdecimal() or not 3 <= int(arguments[0]) <= 1_048_576
            or not no_new_privileges()
        ):
            return 2
        with socket.socket(fileno=int(arguments[0])) as stream:
            if (
                stream.family != socket.AF_UNIX
                or stream.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_SEQPACKET
                or stream.getsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED) != 1
            ):
                return 2
            stream.settimeout(2)
            ready = struct.pack("!I", len(READY)) + READY
            if stream.sendmsg([ready]) != len(ready):
                return 2
            if arguments[1] == "exit_after_ready":
                return 3
            if stream.recv(1) != b"G":
                return 2
            if arguments[1] == "stall":
                stream.recv(1)
                return 2
            packet = struct.pack("!I", len(PAYLOAD)) + PAYLOAD
            if stream.sendmsg([packet]) != len(packet) or stream.recv(1) != b"K":
                return 2
        return 0
    except BaseException:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
