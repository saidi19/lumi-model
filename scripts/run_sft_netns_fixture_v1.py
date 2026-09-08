#!/usr/bin/env python3
"""Remote-only generated network-namespace acceptance; NEVER run on the Mac.

The exact immutable cgroup fixture is loaded from hash-checked bytes into a
private importlib module. Fixture-only composition replaces its workload,
controller entry, case-verifier and run entrypoints, not its controller body,
guardian, pidfd/deadline cleanup, cgroup ownership or WorkerRun binding. Only four
fresh owned cgroups are created. No host interface, route or sysctl is changed.

Normal-case probes establish direct-IP isolation, private IPv4/IPv6 loopback,
host-loopback and abstract-UNIX separation. UDP connect-only negative probes
never send packets. Pathname UNIX sockets/shared files remain possible relays:
this is NOT complete network/filesystem isolation or real execution admission.
Only fixed codes/counts/booleans leave the fixture, never addresses or paths.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import importlib.util
import os
import select
import socket
import stat
import struct
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN_PATH = ROOT / "scripts/run_sft_cgroup_fixture_v1.py"
FROZEN_SHA256 = "f91ed0c353a20c309ff3e1bfaab8dbe990b28c36626fdeeb0ac5131e0cf75fad"
FROZEN_SIZE = 27_839
SCHEMA = "lumi.sft-netns-linux-fixture.v0.1"
CASES = ("normal", "deadline", "parent_loss", "worker_run")
NONCLAIMS = (
    "real_model_or_data_access", "cuda_used", "production_eligible",
    "real_sft_execution_permitted", "complete_os_sandbox_verified",
    "full_network_isolation_verified", "filesystem_isolation_verified",
    "real_host_admission_verified", "runtime_origin_authenticated",
    "root_parent_controllers_modified", "host_network_configuration_modified",
    "packages_installed", "pathname_unix_socket_isolation_verified",
)
PROBE_RECEIPT = {
    "phase": "done", "generated_work_completed": True,
    "external_ipv4_no_route": True, "external_ipv6_no_route": True,
    "host_ipv4_loopback_unreachable": True, "host_ipv6_loopback_unreachable": True,
    "host_abstract_unix_unreachable": True,
    "private_ipv4_loopback_roundtrip": True, "private_ipv6_loopback_roundtrip": True,
    "network_admin_denied_after_privilege_drop": True,
}
NAMESPACE_RECEIPT = {
    "schema": "lumi.sft-netns.v0.1", "namespace_changed": True,
    "loopback_up": True, "only_loopback_interface": True, "namespace_fds_closed": True,
    "full_network_isolation_verified": False,
    "filesystem_isolation_verified": False, "complete_os_sandbox_verified": False,
    "real_sft_execution_permitted": False, "production_eligible": False,
}
BASE = None
FROZEN_CONTROLLER = FROZEN_VERIFY_CASE = None
HOST_SOCKET_FD, HOST_SOCKET_IDENTITY = None, None
HOST_IPV4_PORT, HOST_IPV6_PORT, HOST_UNIX_ADDRESS = None, None, None


def load_frozen():
    """Compile exactly the checked bytes; importlib never rereads an unchecked path."""
    for current in reversed((FROZEN_PATH, *FROZEN_PATH.parents)):
        if stat.S_ISLNK(current.lstat().st_mode):
            raise RuntimeError("frozen_binding_failed")
    fd = os.open(str(FROZEN_PATH), os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != FROZEN_SIZE:
            raise RuntimeError("frozen_binding_failed")
        raw = os.read(fd, FROZEN_SIZE + 1)
        after = os.fstat(fd)
        if (len(raw) != FROZEN_SIZE or os.read(fd, 1)
                or hashlib.sha256(raw).hexdigest() != FROZEN_SHA256
                or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                    before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size,
                                           after.st_mtime_ns, after.st_ctime_ns)):
            raise RuntimeError("frozen_binding_failed")
    finally:
        os.close(fd)
    spec = importlib.util.spec_from_file_location("_lumi_netns_frozen_cgroup_v1", FROZEN_PATH)
    if spec is None:
        raise RuntimeError("frozen_binding_failed")
    selected = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(FROZEN_PATH), "exec"), selected.__dict__)
    return selected


def socket_identity(fd):
    value = os.fstat(fd)
    BASE.require(stat.S_ISSOCK(value.st_mode))
    return value.st_dev, value.st_ino


def sanitize_standard_descriptors(group_fd, channel):
    BASE.require(group_fd > 2 and channel.fileno() > 2 and group_fd != channel.fileno())
    BASE.require(socket_identity(0) == socket_identity(2) == HOST_SOCKET_IDENTITY)
    BASE.require(channel.family == socket.AF_UNIX
                 and channel.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) == socket.SOCK_SEQPACKET)
    null_fd = os.open("/dev/null", os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        value = os.fstat(null_fd)
        BASE.require(stat.S_ISCHR(value.st_mode) and value.st_uid == 0
                     and os.major(value.st_rdev) == 1 and os.minor(value.st_rdev) == 3)
        for fd in (0, 1, 2):
            os.dup2(null_fd, fd, inheritable=True)
            observed = os.fstat(fd)
            BASE.require((observed.st_dev, observed.st_ino, observed.st_rdev)
                         == (value.st_dev, value.st_ino, value.st_rdev))
    finally:
        os.close(null_fd)
    BASE.close_except({group_fd, channel.fileno()})
    assert_descriptors({0, 1, 2, group_fd, channel.fileno()})


def assert_descriptors(expected):
    entries = os.listdir("/proc/self/fd")
    BASE.require(len(entries) <= 1024)
    observed = set()
    for entry in entries:
        BASE.require(entry.isdecimal())
        fd = int(entry)
        try:
            os.fstat(fd)
        except OSError as error:
            BASE.require(error.errno == errno.EBADF)  # Closed listdir's transient descriptor.
        else:
            observed.add(fd)
    BASE.require(observed == expected)


def controller(*arguments):
    # Deliberately inject a live host socket into standard descriptors. The frozen
    # hierarchy leaves these slots alone; our workload must remove both copies.
    BASE.require(socket_identity(HOST_SOCKET_FD) == HOST_SOCKET_IDENTITY)
    os.dup2(HOST_SOCKET_FD, 0)
    os.dup2(HOST_SOCKET_FD, 2)
    return FROZEN_CONTROLLER(*arguments)


def expect_connect_error(family, kind, address, expected_errno):
    with socket.socket(family, kind) as client:
        client.settimeout(0.5)
        try:
            client.connect(address)
        except OSError as error:
            BASE.require(error.errno == expected_errno)
        else:
            raise BASE.FixtureError()


def receive_exact(connection, expected):
    deadline = time.monotonic_ns() + 1_000_000_000
    received = bytearray()
    for _ in range(len(expected)):
        remaining = deadline - time.monotonic_ns()
        BASE.require(remaining > 0)
        connection.settimeout(remaining / 1_000_000_000)
        chunk = connection.recv(len(expected) - len(received))
        BASE.require(chunk)
        received.extend(chunk)
        if len(received) == len(expected):
            BASE.require(received == expected)
            return
    raise BASE.FixtureError()


def roundtrip(listener, family, address):
    listener.settimeout(1)
    with socket.socket(family, socket.SOCK_STREAM) as client:
        client.settimeout(1)
        client.connect(address)
        connection, _peer = listener.accept()
        with connection:
            connection.settimeout(1)
            client.sendall(b"generated-ping")
            receive_exact(connection, b"generated-ping")
            connection.sendall(b"generated-pong")
            receive_exact(client, b"generated-pong")


def private_loopback(family, host):
    with socket.socket(family, socket.SOCK_STREAM) as listener:
        if family == socket.AF_INET6:
            listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        listener.bind((host, 0))
        listener.listen(1)
        roundtrip(listener, family, listener.getsockname())


def network_probes():
    BASE.require(socket.if_nameindex() == [(1, "lo")])
    # No send/sendto: a mistakenly available route fails without emitting traffic.
    expect_connect_error(socket.AF_INET, socket.SOCK_DGRAM, ("192.0.2.1", 9), errno.ENETUNREACH)
    expect_connect_error(socket.AF_INET6, socket.SOCK_DGRAM, ("2001:db8::1", 9, 0, 0),
                         errno.ENETUNREACH)
    # Host sentinel negatives precede local ephemeral listeners to avoid collisions.
    expect_connect_error(socket.AF_INET, socket.SOCK_STREAM, ("127.0.0.1", HOST_IPV4_PORT),
                         errno.ECONNREFUSED)
    expect_connect_error(socket.AF_INET6, socket.SOCK_STREAM, ("::1", HOST_IPV6_PORT, 0, 0),
                         errno.ECONNREFUSED)
    expect_connect_error(socket.AF_UNIX, socket.SOCK_STREAM, HOST_UNIX_ADDRESS, errno.ECONNREFUSED)
    private_loopback(socket.AF_INET, "127.0.0.1")
    private_loopback(socket.AF_INET6, "::1")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as control:
        observed = fcntl.ioctl(control.fileno(), 0x8913, struct.pack("16sH22x", b"lo", 0))
        try:
            # Rewriting the same flags still requires CAP_NET_ADMIN.
            fcntl.ioctl(control.fileno(), 0x8914, observed)
        except OSError as error:
            BASE.require(error.errno == errno.EPERM)
        else:
            raise BASE.FixtureError()


def workload(case, group_fd, channel, uid, gid, runtime, deadline_ns):
    try:
        sanitize_standard_descriptors(group_fd, channel)
        BASE.module("posttraining_sft_cgroup_v1").enter_current_process(group_fd)
        network = BASE.module("posttraining_sft_netns_v1")
        BASE.require(network.create_loopback_namespace() == NAMESPACE_RECEIPT)
        BASE.module("posttraining_sft_cgroup_guardian_v1").drop_worker_privileges(uid, gid)
        os.close(group_fd)
        BASE.close_except({channel.fileno()})
        assert_descriptors({0, 1, 2, channel.fileno()})
        BASE.require(os.getresuid() == (uid, uid, uid) and os.getresgid() == (gid, gid, gid))
        BASE.require(os.getgroups() == [])
        BASE.send(channel, {"phase": "ready", "uid_gid_dropped": True,
                            "authority_fds_closed": True})
        BASE.require(select.select([channel], [], [], 10)[0] and channel.recv(1) == b"G")
        if case == "worker_run":
            os.dup2(channel.fileno(), 1)
            BASE.close_except(set())
            os.execve(runtime, [runtime, "-I", "-B", str(BASE.ACCEPTED_SCRIPT), "--synthetic-only"],
                      BASE.worker_environment(runtime))
        elif case == "parent_loss":
            BASE.parent_loss_payload(channel)
        elif case == "deadline":
            time.sleep(30)
            raise BASE.FixtureError()
        else:
            BASE.require(case == "normal" and time.monotonic_ns() < deadline_ns)
            network_probes()
            assert_descriptors({0, 1, 2, channel.fileno()})
            BASE.send(channel, PROBE_RECEIPT)
        channel.close()
        os._exit(0)
    except BaseException as error:
        # Frozen guardian recognizes its original component codes. New network
        # errors deliberately fail as fixture_failed rather than mutating that
        # immutable allowlist or exposing any exception text.
        for name, error_type in (("posttraining_sft_cgroup_v1", "CgroupError"),
                                 ("posttraining_sft_cgroup_guardian_v1", "GuardianError")):
            try:
                implementation = BASE.module(name)
                if (type(error) is getattr(implementation, error_type)
                        and error.code in implementation.ERROR_CODES):
                    BASE.send(channel, {"phase": "bootstrap_failed", "component_code": error.code})
            except BaseException:
                pass
        os._exit(2)


def verify_case(case, watched, snapshot, statuses, worker_pid, messages):
    if case == "normal":
        BASE.require(messages == [PROBE_RECEIPT])
        messages = [{"phase": "done", "generated_work_completed": True}]
    return FROZEN_VERIFY_CASE(case, watched, snapshot, statuses, worker_pid, messages)


def run():
    global HOST_SOCKET_FD, HOST_SOCKET_IDENTITY
    global HOST_IPV4_PORT, HOST_IPV6_PORT, HOST_UNIX_ADDRESS
    uid, gid, value = BASE.bootstrap()
    deadline_ns = time.monotonic_ns() + BASE.MAXIMUM_SECONDS * 1_000_000_000
    os.umask(0o077)
    BASE.subreaper()
    BASE.STAGE = "cases"
    before = os.stat("/proc/self/ns/net")
    host_identity = before.st_dev, before.st_ino
    reports = []
    with (socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ipv4,
          socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as ipv6,
          socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as abstract):
        ipv4.bind(("127.0.0.1", 0))
        ipv6.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        ipv6.bind(("::1", 0))
        HOST_UNIX_ADDRESS = "\0lumi-sft-netns-" + uuid.uuid4().hex
        abstract.bind(HOST_UNIX_ADDRESS)
        for listener, family in ((ipv4, socket.AF_INET), (ipv6, socket.AF_INET6),
                                 (abstract, socket.AF_UNIX)):
            listener.listen(1)
            roundtrip(listener, family, listener.getsockname())
        HOST_IPV4_PORT, HOST_IPV6_PORT = ipv4.getsockname()[1], ipv6.getsockname()[1]
        HOST_SOCKET_FD, HOST_SOCKET_IDENTITY = ipv4.fileno(), socket_identity(ipv4.fileno())
        for case in CASES:
            reports.append(BASE.run_case(case, uid, gid, value["runtime_path"], deadline_ns))
            observed = os.stat("/proc/self/ns/net")
            BASE.require((observed.st_dev, observed.st_ino) == host_identity)
        # The same outer-held sockets are still reachable after all namespace teardown.
        for listener, family in ((ipv4, socket.AF_INET), (ipv6, socket.AF_INET6),
                                 (abstract, socket.AF_UNIX)):
            roundtrip(listener, family, listener.getsockname())
    BASE.require("torch" not in sys.modules)
    BASE.STAGE = "complete"
    return {
        "schema_version": SCHEMA, "status": "passed", **dict.fromkeys(NONCLAIMS, False),
        "cases_verified": 4, "owned_cgroups_empty_and_removed": 4,
        "fresh_loopback_only_namespaces_verified": 4,
        "direct_ip_network_isolation_verified": True,
        **{key: value for key, value in PROBE_RECEIPT.items() if key not in
           {"phase", "generated_work_completed"}},
        "host_namespace_unchanged": True, "host_sentinels_live_before_and_after": True,
        "inherited_standard_socket_descriptors_removed": True,
        "deadline_cleanup_verified": True, "actual_parent_loss_cleanup_verified": True,
        "double_fork_setsid_descendants_reaped": True, "privileges_dropped_before_payload": True,
        "cgroup_pidfd_authority_closed_before_payload": True,
        "guardian_outside_workload_cgroup": True, "no_root_ml_import": True,
        "pinned_worker_run_fixture_completed": True, "worker_run_invocations_completed": 3,
        "worker_run_rank_workers_verified": 6, "worker_run_global_optimizer_steps_completed": 4,
        "worker_run_rank_optimizer_steps_completed": 8,
        "workload_processes_reaped": sum(report["workload_processes_reaped"] for report in reports),
    }


def main():
    global BASE, FROZEN_CONTROLLER, FROZEN_VERIFY_CASE
    try:
        BASE = load_frozen()
    except BaseException:
        # No fixture or privileged subprocess was started if the immutable binding failed.
        os.write(1, b'{"schema_version":"lumi.sft-netns-linux-fixture.v0.1",'
                    b'"status":"failed","code":"frozen_binding_failed"}\n')
        return 1
    FROZEN_CONTROLLER, FROZEN_VERIFY_CASE = BASE.controller, BASE.verify_case
    BASE.controller, BASE.workload, BASE.verify_case = controller, workload, verify_case
    BASE.run, BASE.CASES, BASE.SCHEMA, BASE.NONCLAIMS = run, CASES, SCHEMA, NONCLAIMS
    return BASE.main()


if __name__ == "__main__":
    raise SystemExit(main())
