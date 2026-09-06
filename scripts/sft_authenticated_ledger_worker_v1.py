"""Fixed Linux epoch-ledger fixture; IPC only, never training or real resource work.

Executed only from authenticated source bytes by the trusted inert probe parent.
The command set cannot open files, import repository code, or launch descendants.
Kernel credentials authenticate each incoming message; epoch tokens stay private.
"""

from __future__ import annotations

import ctypes
import hashlib
import hmac
import json
import os
import socket
import struct
import sys
import time

CONTROL_SCHEMA = "lumi.sft-synthetic-ipc-control.v0.1"
REQUEST_SCHEMA = "lumi.sft-epoch-ledger-request.v0.1"
ACK_SCHEMA = "lumi.sft-epoch-ledger-ack.v0.1"
READY = b"authenticated-ledger-ready-v1"
FAULTS = frozenset({
    "none", "wrong_epoch", "wrong_rank", "rights", "exit_before_confirmation",
    "stall", "bad_exit", "trailing_after_bye",
})
RESOURCE_FIELDS = frozenset({
    "checkpoint_read_bytes", "encoded_binary_read_bytes", "encoded_index_read_bytes",
    "private_write_bytes", "log_bytes", "encoded_index_rows",
})
MAXIMUM_PAYLOAD_BYTES = 4096
MAXIMUM_MESSAGES = 32
MAXIMUM_INTEGER = 2**63 - 1
MAXIMUM_SECONDS = 20
MESSAGE_SECONDS = 3


def need(condition):
    if not condition:
        raise ValueError from None


def no_new_privileges():
    # Fixed libc symbol and argument types; never a library-lookup subprocess.
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    prctl.restype = ctypes.c_int
    return prctl(38, 1, 0, 0, 0) == 0 and prctl(39, 0, 0, 0, 0) == 1


def canonical(value):
    need(type(value) is dict)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def frame(payload):
    need(type(payload) is bytes and 0 < len(payload) <= MAXIMUM_PAYLOAD_BYTES)
    return struct.pack("!I", len(payload)) + payload


def unique_pairs(items):
    output = {}
    for key, value in items:
        need(key not in output)
        output[key] = value
    return output


def reject_constant(_value):
    raise ValueError from None


def decode(raw):
    need(type(raw) is bytes and 4 < len(raw) <= MAXIMUM_PAYLOAD_BYTES + 4)
    need(int.from_bytes(raw[:4], "big") == len(raw) - 4)
    value = json.loads(raw[4:], object_pairs_hook=unique_pairs, parse_constant=reject_constant)
    need(hmac.compare_digest(canonical(value), raw[4:]))
    return value


def control(kind, epoch, rank, **fields):
    return {"schema_version": CONTROL_SCHEMA, "kind": kind, "epoch": epoch, "rank": rank,
            **fields}


def matches(value, expected):
    return hmac.compare_digest(canonical(value), canonical(expected))


def token(value):
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def integer(value, *, zero=False):
    return type(value) is int and (0 if zero else 1) <= value <= MAXIMUM_INTEGER


def valid_arguments(operation, arguments, rank):
    if type(operation) is not str or type(arguments) is not dict:
        return False
    if operation == "reserve":
        return (bool(arguments) and set(arguments) <= RESOURCE_FIELDS
                and all(integer(value, zero=True) for value in arguments.values())
                and any(value > 0 for value in arguments.values()))
    if operation == "reserve_step":
        return (rank == 0 and set(arguments) == {"from_step", "to_step"}
                and integer(arguments["from_step"], zero=True) and integer(arguments["to_step"]))
    if operation == "step_complete":
        return set(arguments) == {"to_step"} and integer(arguments["to_step"])
    return False


def close_rights(ancillary):
    """Close every delivered descriptor once, even when the packet is rejected."""
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
        self.stream.settimeout(min(remaining, MESSAGE_SECONDS))

    def send_payload(self, payload, *, rights=False):
        packet = frame(payload)
        need(self.sent < MAXIMUM_MESSAGES)
        self.sent += 1
        self.before_io()
        ancillary = ([(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                       struct.pack("i", self.stream.fileno()))] if rights else [])
        need(self.stream.sendmsg([packet], ancillary) == len(packet))
        need(time.monotonic() < self.deadline)

    def send(self, value, *, rights=False):
        self.send_payload(canonical(value), rights=rights)

    def receive(self):
        need(self.received < MAXIMUM_MESSAGES)
        self.received += 1
        self.before_io()
        raw, ancillary, flags, _address = self.stream.recvmsg(
            MAXIMUM_PAYLOAD_BYTES + 5, self.ancillary_capacity, socket.MSG_CMSG_CLOEXEC
        )
        # Resource cleanup precedes all frame, credential and deadline rejection.
        clean = close_rights(ancillary)
        need(clean and time.monotonic() < self.deadline)
        need(type(flags) is int and flags in {0, socket.MSG_CMSG_CLOEXEC})
        need(type(ancillary) is list and len(ancillary) == 1)
        item = ancillary[0]
        need(type(item) is tuple and len(item) == 3)
        need(item[0] == socket.SOL_SOCKET and item[1] == socket.SCM_CREDENTIALS)
        need(type(item[2]) is bytes and len(item[2]) == 12)
        need(struct.unpack("iII", item[2]) == self.parent_credentials)
        value = decode(raw)
        need(time.monotonic() < self.deadline)
        return raw, value


def conversation(channel, fault):
    channel.send_payload(READY)
    _raw, initial = channel.receive()
    need(set(initial) == {"schema_version", "kind", "epoch", "rank", "token"})
    epoch, rank, rank_token = initial["epoch"], initial["rank"], initial["token"]
    need(type(rank) is int and rank in {0, 1} and token(epoch) and token(rank_token))
    need(matches(initial, control("initialize", epoch, rank, token=rank_token)))
    channel.send(control("hello", epoch, rank, token=rank_token))
    sequence = 0
    while True:
        _raw, command = channel.receive()
        if matches(command, control("stop", epoch, rank)):
            channel.send(control("bye", epoch, rank))
            _raw, release = channel.receive()
            need(matches(release, control("release", epoch, rank)))
            if fault == "trailing_after_bye":
                channel.send_payload(b"authenticated-ledger-trailing-v1")
            return 2 if fault == "bad_exit" else 0
        need(set(command) == {"schema_version", "kind", "epoch", "rank",
                              "operation", "arguments"})
        operation, arguments = command["operation"], command["arguments"]
        need(matches(command, control("command", epoch, rank,
                                      operation=operation, arguments=arguments)))
        need(valid_arguments(operation, arguments, rank))
        sequence += 1
        if fault == "stall" and sequence == 1:
            channel.receive()
            return 2
        request = {
            "schema_version": REQUEST_SCHEMA, "epoch": epoch, "rank": rank,
            "sequence": sequence, "token": rank_token, "operation": operation,
            "arguments": arguments,
        }
        if fault == "wrong_epoch" and sequence == 1:
            request["epoch"] = "0" * 64 if epoch != "0" * 64 else "1" * 64
        if fault == "wrong_rank" and sequence == 1:
            request["rank"] = 1 - rank
        channel.send(request, rights=fault == "rights" and sequence == 1)
        ack_raw, ack = channel.receive()
        need(matches(ack, {
            "schema_version": ACK_SCHEMA, "epoch": epoch, "rank": rank,
            "sequence": sequence, "operation": operation, "status": "reserved",
        }))
        if fault == "exit_before_confirmation" and sequence == 1:
            return 2
        channel.send({
            **request, "epoch": epoch, "rank": rank, "operation": "confirm_ack",
            "arguments": {"ack_sha256": hashlib.sha256(ack_raw).hexdigest()},
        })
        _raw, continuation = channel.receive()
        need(matches(continuation, control("continue", epoch, rank, sequence=sequence)))


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
            and no_new_privileges()
        )
        parent_credentials = (os.getppid(), os.getuid(), os.getgid())
        need(parent_credentials[0] > 1)
        with socket.socket(fileno=int(arguments[0])) as stream:
            need(
                stream.family == socket.AF_UNIX
                and stream.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) == socket.SOCK_SEQPACKET
                and stream.getsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED) == 1
            )
            stream.set_inheritable(False)
            return conversation(Channel(stream, deadline, parent_credentials), arguments[1])
    except BaseException:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
