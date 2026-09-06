"""Compose frozen kernel confinement and frozen inert ledger IPC, never SFT.

The trusted parent supplies three separately hash-authenticated source modules.
All setup, imports and ctypes symbol resolution precede policy installation.
The installed policy and every fixed challenge must pass before any READY or
epoch-token receive. Existing ledger framing, credentials and faults are reused.
"""

import os
import socket
import struct
import sys
import time

kernel_policy = globals()["kernel_policy"]
kernel_challenges = globals()["kernel_challenges"]
ledger_worker = globals()["ledger_worker"]

READY = b"guarded-ledger-ready-v1"
GUARD_EXIT_FAULT = "guard_exit_before_ready"
FAULTS = ledger_worker.FAULTS | frozenset({GUARD_EXIT_FAULT})
MAXIMUM_SECONDS = 20
MAXIMUM_MESSAGES = 32
EXPECTED_INSTALLATION = {
    "schema_version": "lumi.sft-kernel-policy-installation.v0.1",
    "policy_installed": True,
    "resource_limits_verified": 6,
    "no_new_privileges": True,
    "seccomp_filter_mode_verified": True,
    "cgroup_memory_enforcement": False,
    "namespace_containment": False,
    "real_sft_execution_permitted": False,
}


def need(condition):
    if not condition:
        raise ValueError from None


def installed_exactly(value):
    """Reject missing, extra, mistyped or weakened frozen installation fields."""
    return (
        type(value) is dict and value.keys() == EXPECTED_INSTALLATION.keys()
        and all(type(key) is str for key in value)
        and all(type(value[key]) is type(expected) and value[key] == expected
                for key, expected in EXPECTED_INSTALLATION.items())
    )


class Channel(ledger_worker.Channel):
    """Only the first READY changes; every other frozen transport rule is intact."""

    def __init__(self, stream, deadline, parent_credentials):
        super().__init__(stream, deadline, parent_credentials)
        self._ready_pending = True

    def send_payload(self, payload, *, rights=False):
        if self._ready_pending:
            need(type(payload) is bytes and payload == ledger_worker.READY and rights is False)
            self._ready_pending = False
            payload = READY
        return super().send_payload(payload, rights=rights)


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
            and ledger_worker.MAXIMUM_SECONDS == MAXIMUM_SECONDS
            and ledger_worker.MAXIMUM_MESSAGES == MAXIMUM_MESSAGES
        )
        kernel_policy.require_abi()
        challenges = kernel_challenges.Challenges(kernel_policy)
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
            need(installed_exactly(kernel_policy.install(stream.fileno())))
            challenges.run()
            need(time.monotonic() < deadline)
            if arguments[1] == GUARD_EXIT_FAULT:
                return 2
            return ledger_worker.conversation(channel, arguments[1])
    except BaseException:
        return 2


if __name__ == "__main__":
    os._exit(main())
