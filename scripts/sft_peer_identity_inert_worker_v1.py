"""Fixed one-packet Linux identity probe child; never a training entry point."""

from __future__ import annotations

import socket
import struct
import sys

FAULTS = frozenset({"none", "unexpected_sender", "rights", "oversized", "stall"})
PAYLOAD = b'{"kind":"inert_identity_probe"}'


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if (
        sys.platform != "linux" or not sys.flags.no_site or not sys.dont_write_bytecode
        or sys.flags.optimize or len(arguments) != 2 or arguments[1] not in FAULTS
        or not arguments[0].isascii() or not arguments[0].isdecimal()
        or not 3 <= int(arguments[0]) <= 1_048_576
    ):
        return 2
    try:
        with socket.socket(fileno=int(arguments[0])) as stream:
            if (
                stream.family != socket.AF_UNIX
                or stream.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_SEQPACKET
                or stream.getsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED) != 1
            ):
                return 2
            stream.settimeout(4)
            if stream.recv(1) != b"G":
                return 2
            if arguments[1] == "stall":
                stream.recv(1)
                return 2
            body = b"x" * 4097 if arguments[1] == "oversized" else PAYLOAD
            frame = struct.pack("!I", len(body)) + body
            ancillary = []
            if arguments[1] == "rights":
                ancillary = [(
                    socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack("i", stream.fileno()),
                )]
            if stream.sendmsg([frame], ancillary) != len(frame) or stream.recv(1) != b"K":
                return 2
        return 0
    except BaseException:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
