#!/usr/bin/env python3
"""Remote-only narrow IPC acceptance; NEVER execute or import on the Mac.

The accepted filesystem fixture is composed from exactly checked bytes. Its
workload, four-case ownership, deadlines, guardian and pinned CPU WorkerRun stay
unchanged. New normal probes name the earlier UNIX creation denial honestly;
they do not fabricate the old connect-time errors. SysV IPC and trusted inherited
control peers remain outside this narrow boundary. No complete sandbox claim.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import importlib.util
import os
import select
import socket
import stat
import struct
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "scripts/run_sft_filesystem_fixture_v1.py"
FROZEN_HASH = "31c3f271a15d412419fd0609497f92bc7bb0a4bf08f9a591e7086c956436ad89"
FROZEN_SIZE = 24_179
SCHEMA = "lumi.sft-ipc-linux-fixture.v0.1"
NONCLAIMS = (
    "all_ipc_isolated", "sysv_ipc_isolation_verified", "hostile_control_peer_isolated",
    "runtime_origin_authenticated", "real_host_admission_verified",
    "full_pathname_unix_isolation_verified", "complete_os_sandbox_verified",
    "real_model_or_data_access", "cuda_used", "real_sft_execution_permitted",
    "production_eligible",
)
NORMAL_RECEIPT = {
    "ipc_filter_verified": True, "visible_same_uid_unix_creation_denied": True,
    "connected_unix_pairs_roundtrip": True, "connected_unix_pairs_cannot_retarget": True,
    "datagram_socketpair_denied": True, "io_uring_syscalls_denied": True,
    "pidfd_getfd_syscall_denied": True, "ptrace_syscall_denied": True,
}
STAGES = frozenset({
    "not_reported", "install", "installed", "network", "filesystem", "visible_identity",
    "unix_creation", "datagram_pair", "stream_pair", "seqpacket_pair",
    "stream_retarget", "seqpacket_retarget", "stream_shutdown", "seqpacket_shutdown",
    "stream_peer_closed", "seqpacket_peer_closed", "io_uring_setup", "io_uring_enter",
    "io_uring_register", "pidfd_getfd", "ptrace", "filter_readback", "complete",
})
FS = ORIGINAL_RUN = ORIGINAL_CASE = ORIGINAL_HELPER = ORIGINAL_POSITIVE = None
ORIGINAL_DROP = ORIGINAL_SEND = ORIGINAL_RECEIVE = ORIGINAL_VERIFY = None
VISIBLE = {}
IPC_STAGE, IPC_CODE = "not_reported", "not_reported"
INSTALL_WITNESS = False
INSTALLATIONS_VERIFIED = 0


def base():
    return FS.base()


def need(value):
    base().require(value)


def stage(value):
    global IPC_STAGE, IPC_CODE
    need(value in STAGES)
    IPC_STAGE, IPC_CODE = value, "not_reported"


def implementation():
    return base().module("posttraining_sft_ipc_filter_v1")


def valid_code(value):
    return (type(value) is str and value in
            {"not_reported", "passed", "fixture_failed", *implementation().ERROR_CODES,
             *base().module("posttraining_sft_ipc_diagnostic_v1").ERROR_CODES})


def probe(label, action):
    global IPC_CODE
    stage(label)
    try:
        return action()
    except BaseException as error:
        IPC_CODE = base().module("posttraining_sft_ipc_diagnostic_v1").failure_code(error)
        raise


def load_frozen():
    for path in reversed((FROZEN, *FROZEN.parents)):
        if stat.S_ISLNK(path.lstat().st_mode):
            raise RuntimeError("frozen_binding_failed")
    fd = os.open(FROZEN, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        raw = os.read(fd, FROZEN_SIZE + 1)
        after = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_size != FROZEN_SIZE
                or len(raw) != FROZEN_SIZE or os.read(fd, 1)
                or hashlib.sha256(raw).hexdigest() != FROZEN_HASH
                or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                    before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size,
                                           after.st_mtime_ns, after.st_ctime_ns)):
            raise RuntimeError("frozen_binding_failed")
    finally:
        os.close(fd)
    spec = importlib.util.spec_from_file_location("_lumi_ipc_frozen_filesystem_v1", FROZEN)
    if spec is None:
        raise RuntimeError("frozen_binding_failed")
    selected = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(FROZEN), "exec"), selected.__dict__)
    return selected


def drop(uid, gid):
    global IPC_CODE, INSTALL_WITNESS
    # Helper/preflight copies still have not_reported: only the actual workload
    # reaches this stage, after its private mount/Landlock setup and before drop.
    if FS.FS_STAGE == "drop":
        stage("install")
        module = implementation()
        try:
            need(module.install_ipc_filter() == dict(module.RECEIPT))
        except BaseException as error:
            IPC_CODE = (error.code if type(error) is module.IpcFilterError
                        and error.code in module.ERROR_CODES else "fixture_failed")
            raise
        stage("installed")
        IPC_CODE, INSTALL_WITNESS = "passed", True
    return ORIGINAL_DROP(uid, gid)


def send(channel, value):
    global IPC_CODE
    if value.get("phase") == "filesystem_failed":
        if IPC_CODE in {"not_reported", "passed"}:
            IPC_CODE = "fixture_failed"
    if value.get("phase") in {"ready", "filesystem_failed"} or "empty_and_removed" in value:
        need(IPC_STAGE in STAGES and valid_code(IPC_CODE) and type(INSTALL_WITNESS) is bool)
        value = dict(value, ipc_stage=IPC_STAGE, ipc_code=IPC_CODE,
                     ipc_install_witness=INSTALL_WITNESS)
    if value.get("phase") == "done" and value.get("filesystem_verified") is True:
        need(IPC_STAGE == "complete" and IPC_CODE == "passed" and INSTALL_WITNESS)
        value = dict(value, **NORMAL_RECEIPT)
    return ORIGINAL_SEND(channel, value)


def project_metadata(value):
    global IPC_STAGE, IPC_CODE, INSTALL_WITNESS
    keys = {"ipc_stage", "ipc_code", "ipc_install_witness"}
    if keys & value.keys():
        need(keys <= value.keys() and type(value["ipc_stage"]) is str
             and value["ipc_stage"] in STAGES
             and valid_code(value["ipc_code"]) and type(value["ipc_install_witness"]) is bool)
        IPC_STAGE, IPC_CODE = value["ipc_stage"], value["ipc_code"]
        INSTALL_WITNESS = value["ipc_install_witness"]
        value = {key: item for key, item in value.items() if key not in keys}
    return value


def receive(channel, deadline_ns):
    value = project_metadata(ORIGINAL_RECEIVE(channel, deadline_ns))
    if value.get("phase") == "ready":
        need(IPC_STAGE == "installed" and IPC_CODE == "passed" and INSTALL_WITNESS)
    if "empty_and_removed" in value:
        if IPC_CODE not in {"not_reported", "passed"}:
            base().COMPONENT_CODE = "ipc_" + IPC_STAGE + "_" + IPC_CODE
        if value.get("status") == "passed":
            need(INSTALL_WITNESS)
    return value


def verify_case(case, watched, snapshot, statuses, worker_pid, messages):
    projected = [project_metadata(value) for value in messages]
    if case == "normal":
        expected = dict(FS.NET.PROBE_RECEIPT, filesystem_verified=True, **NORMAL_RECEIPT)
        need(projected == [expected] and INSTALL_WITNESS)
        projected = [{key: value for key, value in projected[0].items()
                      if key not in NORMAL_RECEIPT}]
    return ORIGINAL_VERIFY(case, watched, snapshot, statuses, worker_pid, projected)


def helper(channel, listener, parent_pid, uid, gid, deadline_ns):
    """One helper owns hidden STREAM and exposed STREAM/SEQPACKET listeners."""
    try:
        parent_fd = os.pidfd_open(parent_pid, 0)
        listeners = (listener, *(value["listener"] for value in VISIBLE.values()))
        base().close_except({channel.fileno(), parent_fd, *(item.fileno() for item in listeners)})
        base().module("posttraining_sft_cgroup_guardian_v1").drop_worker_privileges(uid, gid)
        prctl = ctypes.CDLL(None, use_errno=True).prctl
        prctl.restype = ctypes.c_int
        need(prctl(4, 1, 0, 0, 0) == 0 and prctl(3, 0, 0, 0, 0) == 1)
        FS.ORIGINAL_SEND(channel, {"helper_ready": True})
        accepted = [0, 0, 0]
        stopped = False
        while time.monotonic_ns() < deadline_ns:
            ready, _, _ = select.select([channel, parent_fd, *listeners], [], [], 0.1)
            if parent_fd in ready:
                os._exit(0)
            if channel in ready:
                need(channel.recv(1) == b"S" and accepted == [1, 1, 1])
                # Do not overlook an extra connection queued alongside stop.
                need(not select.select(listeners, [], [], 0)[0])
                stopped = True
                break
            for index, item in enumerate(listeners):
                if item in ready:
                    accepted[index] += 1
                    need(accepted[index] == 1)
                    connection, _ = item.accept()
                    with connection:
                        connection.settimeout(1)
                        FS.NET.receive_exact(connection, b"generated-ping")
                        connection.sendall(b"generated-pong")
        need(stopped)
        os._exit(0)
    except BaseException:
        os._exit(2)


def outside_positive(uid, gid, deadline_ns):
    ORIGINAL_POSITIVE(uid, gid, deadline_ns)
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    pid = os.fork()
    if pid == 0:
        try:
            base().close_except({child.fileno()})
            base().module("posttraining_sft_cgroup_guardian_v1").drop_worker_privileges(uid, gid)
            for kind, value in VISIBLE.items():
                need(FS.identity(value["path"]) == value["identity"])
                with socket.socket(socket.AF_UNIX, kind) as connection:
                    connection.settimeout(1)
                    connection.connect(value["path"])
                    connection.sendall(b"generated-ping")
                    FS.NET.receive_exact(connection, b"generated-pong")
            FS.ORIGINAL_SEND(child, {"visible_same_uid_positive": True})
            os._exit(0)
        except BaseException:
            os._exit(2)
    child.close()
    pidfd, completed = os.pidfd_open(pid, 0), False
    try:
        need(FS.ORIGINAL_RECEIVE(parent, deadline_ns) == {"visible_same_uid_positive": True})
        status = FS.wait_owned(pid, pidfd, deadline_ns)
        completed = True
        need(status == 0)
    finally:
        if not completed:
            FS.wait_owned(pid, pidfd, time.monotonic_ns() + 5_000_000_000, terminate=True)
        os.close(pidfd)
        parent.close()


def case_wrapper(case, uid, gid, runtime, deadline_ns):
    global VISIBLE
    global IPC_STAGE, IPC_CODE, INSTALL_WITNESS, INSTALLATIONS_VERIFIED
    IPC_STAGE, IPC_CODE, INSTALL_WITNESS = "not_reported", "not_reported", False
    base().REMOVAL_REPORTED = False
    VISIBLE = {}
    try:
        for kind in (socket.SOCK_STREAM, socket.SOCK_SEQPACKET):
            path = str(ROOT / "src" / (".lumi-ipc-" + uuid.uuid4().hex))
            need(len(path.encode("utf-8")) < 108)
            value = {"path": path, "listener": socket.socket(socket.AF_UNIX, kind),
                     "bound": False, "identity": None}
            VISIBLE[kind] = value
            value["listener"].bind(path)
            value["bound"] = True
            value["identity"] = FS.identity(path)
            os.chown(path, uid, gid)
            os.chmod(path, 0o600)
            value["identity"] = FS.identity(path)
            value["listener"].listen(1)
        result = ORIGINAL_CASE(case, uid, gid, runtime, deadline_ns)
        need(INSTALL_WITNESS and FS.HELPER_PID is None
             and all(FS.identity(value["path"]) == value["identity"]
                     for value in VISIBLE.values()))
        INSTALLATIONS_VERIFIED += 1
        return result
    finally:
        cgroup_removed = base().REMOVAL_REPORTED
        base().REMOVAL_REPORTED = False
        failed = False
        for value in VISIBLE.values():
            try:
                value["listener"].close()
            except BaseException:
                failed = True
            if value["bound"]:
                try:
                    need(value["identity"] is not None
                         and FS.identity(value["path"])[:2] == value["identity"][:2])
                    os.unlink(value["path"])
                except BaseException:
                    failed = True
        VISIBLE = {}
        if failed:
            raise base().FixtureError("cleanup_unconfirmed") from None
        base().REMOVAL_REPORTED = cgroup_removed


def denied(action, errors=(errno.EPERM,)):
    try:
        value = action()
    except OSError as error:
        if error.errno not in errors:
            raise
    else:
        if isinstance(value, socket.socket):
            value.close()
        elif type(value) is tuple:
            for item in value:
                item.close()
        elif type(value) is int:
            os.close(value)
        raise base().FixtureError()


def network_probes():
    stage("network")
    net = FS.NET
    need(socket.if_nameindex() == [(1, "lo")])
    net.expect_connect_error(socket.AF_INET, socket.SOCK_DGRAM, ("192.0.2.1", 9),
                             errno.ENETUNREACH)
    net.expect_connect_error(socket.AF_INET6, socket.SOCK_DGRAM, ("2001:db8::1", 9, 0, 0),
                             errno.ENETUNREACH)
    net.expect_connect_error(socket.AF_INET, socket.SOCK_STREAM, ("127.0.0.1", net.HOST_IPV4_PORT),
                             errno.ECONNREFUSED)
    net.expect_connect_error(socket.AF_INET6, socket.SOCK_STREAM,
                             ("::1", net.HOST_IPV6_PORT, 0, 0), errno.ECONNREFUSED)
    # The previous abstract-UNIX connect is blocked earlier, at creation.
    denied(lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM))
    net.private_loopback(socket.AF_INET, "127.0.0.1")
    net.private_loopback(socket.AF_INET6, "::1")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as control:
        observed = fcntl.ioctl(control.fileno(), 0x8913, struct.pack("16sH22x", b"lo", 0))
        denied(lambda: fcntl.ioctl(control.fileno(), 0x8914, observed))


def filesystem_probes():
    stage("filesystem")
    root_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        need(stat.S_ISDIR(os.fstat(root_fd).st_mode))
    finally:
        os.close(root_fd)
    FS.read_canary(FS.CANARY_PATH)
    FS.denied(lambda: os.open(FS.HOST_FILE, os.O_RDONLY), (errno.ENOENT,))
    FS.denied(lambda: os.open(f"/proc/{FS.HELPER_PID}/root{FS.HOST_FILE}", os.O_RDONLY))
    FS.denied(lambda: os.open("/.oldroot", os.O_RDONLY), (errno.ENOENT,))
    # Hidden-path and /proc/host/root UNIX attempts cannot obtain a new socket.
    denied(lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM))
    FS.denied(lambda: os.open(FS.CANARY_PATH, os.O_WRONLY | os.O_TRUNC))
    FS.denied(lambda: os.chmod(FS.CANARY_PATH, 0o400))
    FS.denied(lambda: os.unlink(FS.CANARY_PATH))
    FS.denied(lambda: os.rename(FS.CANARY_PATH, FS.CANARY_PATH + ".moved"))
    FS.denied(lambda: os.link(FS.CANARY_PATH, FS.CANARY_PATH + ".linked"))
    FS.read_canary(FS.CANARY_PATH)
    fd = os.open("/tmp/generated-roundtrip", os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        need(os.write(fd, FS.CANARY_BYTES) == len(FS.CANARY_BYTES))
        os.lseek(fd, 0, os.SEEK_SET)
        need(os.read(fd, len(FS.CANARY_BYTES) + 1) == FS.CANARY_BYTES)
    finally:
        os.close(fd)
        os.unlink("/tmp/generated-roundtrip")
    fd = os.open("/tmp/generated-capacity", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    exhausted, written = False, 0
    block = b"\0" * (1024 * 1024)
    try:
        for _ in range(10):
            try:
                amount = os.write(fd, block)
            except OSError as error:
                need(error.errno == errno.ENOSPC)
                exhausted = True
                break
            need(0 < amount <= len(block))
            written += amount
            need(written <= 8 * 1024**2)
        need(exhausted and written > 0)
    finally:
        os.close(fd)
        os.unlink("/tmp/generated-capacity")
    fd = os.open("/proc/self/statm", os.O_RDONLY | os.O_CLOEXEC)
    try:
        raw = os.read(fd, 257)
        need(0 < len(raw) <= 256 and raw.split()[1].isdigit())
    finally:
        os.close(fd)
    ipc_probes()


def call_denied(number, *arguments):
    syscall = ctypes.CDLL(None, use_errno=True).syscall
    syscall.restype = ctypes.c_long
    syscall.argtypes = [ctypes.c_long]
    ctypes.set_errno(0)
    result = syscall(number, *(ctypes.c_long(item) for item in arguments))
    need(result == -1 and ctypes.get_errno() == errno.EPERM)


def pair_roundtrip(left, right, payload):
    need(left.send(payload, socket.MSG_NOSIGNAL) == len(payload))
    FS.NET.receive_exact(right, payload)
    need(right.send(payload, socket.MSG_NOSIGNAL) == len(payload))
    FS.NET.receive_exact(left, payload)


def addressed_pair_send(left, right, path, *, message):
    payload = b"generated-pair-addressed"
    try:
        if message:
            amount = left.sendmsg([payload], [], socket.MSG_NOSIGNAL, path)
        else:
            amount = left.sendto(payload, socket.MSG_NOSIGNAL, path)
    except OSError as error:
        if error.errno not in {errno.EISCONN, errno.EINVAL, errno.EOPNOTSUPP}:
            raise
    else:
        # Linux STREAM may ignore a destination on an already connected pair.
        # Success must therefore deliver only to its original, generated peer.
        need(amount == len(payload))
        FS.NET.receive_exact(right, payload)


def cannot_disconnect(left):
    # AF_UNSPEC disconnect works for datagrams, but must not reset permitted pairs.
    address = ctypes.create_string_buffer(16)
    syscall = ctypes.CDLL(None, use_errno=True).syscall
    syscall.argtypes, syscall.restype = [ctypes.c_long], ctypes.c_long
    ctypes.set_errno(0)
    result = syscall(42, ctypes.c_int(left.fileno()), ctypes.byref(address), ctypes.c_int(16))
    need(result == -1)
    if ctypes.get_errno() not in {errno.EINVAL, errno.EAFNOSUPPORT, errno.EISCONN}:
        raise OSError(ctypes.get_errno(), "probe_failed")


def cannot_become_listener(left, name):
    path, bound = "/tmp/generated-pair-bind-" + name, False
    try:
        try:
            left.bind(path)
        except OSError as error:
            if error.errno != errno.EINVAL:
                raise
        else:
            bound = True
            info = os.lstat(path)
            need(stat.S_ISSOCK(info.st_mode) and info.st_uid == os.getuid())
        denied(lambda: left.listen(1), (errno.EINVAL, errno.EOPNOTSUPP))
    finally:
        if bound:
            os.unlink(path)


def connected_pair_probes(kind, name):
    stage(name + "_pair")
    path = VISIBLE[kind]["path"]
    left, right = socket.socketpair(socket.AF_UNIX, kind)
    try:
        left.settimeout(1)
        right.settimeout(1)
        pair_roundtrip(left, right, b"generated-pair-roundtrip")
        stage(name + "_retarget")
        probe(name + "_retarget_disconnect", lambda: cannot_disconnect(left))
        probe(name + "_retarget_connect", lambda: denied(
            lambda: left.connect(path), (errno.EISCONN, errno.EINVAL)))
        probe(name + "_retarget_bind_listen", lambda: cannot_become_listener(
            left, name + "-connected"))
        probe(name + "_retarget_sendto", lambda: addressed_pair_send(
            left, right, path, message=False))
        probe(name + "_retarget_sendmsg", lambda: addressed_pair_send(
            left, right, path, message=True))
        probe(name + "_retarget_roundtrip", lambda: pair_roundtrip(
            left, right, b"generated-original-peer"))
        stage(name + "_shutdown")
        left.shutdown(socket.SHUT_RDWR)
        cannot_disconnect(left)
        denied(lambda: left.connect(path), (errno.EISCONN, errno.EINVAL))
        denied(lambda: left.listen(1), (errno.EINVAL, errno.EOPNOTSUPP))
        errors = (errno.EPIPE, errno.ECONNRESET, errno.ENOTCONN, errno.EISCONN, errno.EINVAL)
        denied(lambda: left.sendto(b"generated-closed", socket.MSG_NOSIGNAL, path), errors)
        denied(lambda: left.sendmsg([b"generated-closed"], [], socket.MSG_NOSIGNAL,
                                     path), errors)
    finally:
        left.close()
        right.close()
    stage(name + "_peer_closed")
    left, right = socket.socketpair(socket.AF_UNIX, kind)
    try:
        left.settimeout(1)
        right.close()
        cannot_disconnect(left)
        denied(lambda: left.connect(path), (errno.EISCONN, errno.EINVAL))
        denied(lambda: left.listen(1), (errno.EINVAL, errno.EOPNOTSUPP))
        errors = (errno.EPIPE, errno.ECONNRESET, errno.ENOTCONN, errno.EISCONN, errno.EINVAL)
        denied(lambda: left.sendto(b"generated-closed", socket.MSG_NOSIGNAL, path), errors)
        denied(lambda: left.sendmsg([b"generated-closed"], [], socket.MSG_NOSIGNAL,
                                     path), errors)
        # Sending after peer death must not reset SEQPACKET to a reconnectable state.
        denied(lambda: left.connect(path), (errno.EISCONN, errno.EINVAL))
        cannot_become_listener(left, name + "-peer-closed")
    finally:
        left.close()
        right.close()


def ipc_probes():
    global IPC_CODE
    stage("visible_identity")
    need(set(VISIBLE) == {socket.SOCK_STREAM, socket.SOCK_SEQPACKET})
    for value in VISIBLE.values():
        observed = FS.identity(value["path"])
        need(observed == value["identity"] and stat.S_ISSOCK(observed[2]))
    stage("unix_creation")
    for kind in (socket.SOCK_STREAM, socket.SOCK_DGRAM, socket.SOCK_SEQPACKET):
        for flags in (0, socket.SOCK_CLOEXEC, socket.SOCK_NONBLOCK,
                      socket.SOCK_CLOEXEC | socket.SOCK_NONBLOCK):
            denied(lambda kind=kind, flags=flags: socket.socket(socket.AF_UNIX, kind | flags))
    stage("datagram_pair")
    denied(lambda: socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM))
    connected_pair_probes(socket.SOCK_STREAM, "stream")
    connected_pair_probes(socket.SOCK_SEQPACKET, "seqpacket")
    for label, number, arguments in (
        ("io_uring_setup", 425, (0, 0)),
        ("io_uring_enter", 426, (-1, 0, 0, 0, 0, 0)),
        ("io_uring_register", 427, (-1, 0, 0, 0)),
        ("pidfd_getfd", 438, (-1, -1, 0)),
        ("ptrace", 101, (-1, -1, 0, 0)),
    ):
        stage(label)
        call_denied(number, *arguments)
    stage("filter_readback")
    prctl = ctypes.CDLL(None, use_errno=True).prctl
    prctl.restype = ctypes.c_int
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    need(prctl(21, 0, 0, 0, 0) == 2 and prctl(39, 0, 0, 0, 0) == 1)
    stage("complete")
    IPC_CODE = "passed"


def run():
    global ORIGINAL_DROP, STAGES
    STAGES = STAGES | base().module("posttraining_sft_ipc_diagnostic_v1").STAGES
    guardian = base().module("posttraining_sft_cgroup_guardian_v1")
    ORIGINAL_DROP, guardian.drop_worker_privileges = guardian.drop_worker_privileges, drop
    FS.NET.network_probes = network_probes
    report = ORIGINAL_RUN()
    need(INSTALLATIONS_VERIFIED == 4)
    report.update(schema_version=SCHEMA, **dict.fromkeys(NONCLAIMS, False))
    report.update(
        seccomp_installations_verified=4, visible_same_uid_sentinel_baselines_verified=8,
        owned_visible_unix_sockets_removed=8, named_unix_creation_boundary_verified=True,
        connected_unix_pair_boundary_verified=True, io_uring_syscall_boundary_verified=True,
        pidfd_getfd_syscall_boundary_verified=True, ptrace_syscall_boundary_verified=True,
        preserved_ip_filesystem_checks_verified=True,
    )
    return report


def main():
    global FS, ORIGINAL_RUN, ORIGINAL_CASE, ORIGINAL_HELPER, ORIGINAL_POSITIVE
    global ORIGINAL_SEND, ORIGINAL_RECEIVE, ORIGINAL_VERIFY
    try:
        FS = load_frozen()
    except BaseException:
        os.write(1, b'{"schema_version":"lumi.sft-ipc-linux-fixture.v0.1",'
                    b'"status":"failed","code":"frozen_binding_failed"}\n')
        return 1
    ORIGINAL_RUN, ORIGINAL_CASE = FS.run, FS.case_wrapper
    ORIGINAL_HELPER, ORIGINAL_POSITIVE = FS.helper, FS.outside_positive
    ORIGINAL_SEND, ORIGINAL_RECEIVE, ORIGINAL_VERIFY = FS.send, FS.receive, FS.verify_case
    FS.run, FS.case_wrapper, FS.helper = run, case_wrapper, helper
    FS.outside_positive, FS.send, FS.receive = outside_positive, send, receive
    FS.verify_case, FS.filesystem_probes = verify_case, filesystem_probes
    FS.SCHEMA = SCHEMA
    FS.NONCLAIMS = tuple(dict.fromkeys((*FS.NONCLAIMS, *NONCLAIMS)))
    return FS.main()


if __name__ == "__main__":
    raise SystemExit(main())
