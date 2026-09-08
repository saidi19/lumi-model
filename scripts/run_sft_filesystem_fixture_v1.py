#!/usr/bin/env python3
"""Remote-only mount/Landlock composition; NEVER execute or import on the Mac.

Only newly owned synthetic canaries may be changed. Old source/runtime bytes,
host mounts and prior fixtures remain untouched. Four cgroup/netns cases retain
the pinned WorkerRun and exact-target guardian cleanup. Exposed software trees
remain trusted: ABI5 is NOT full pathname-UNIX IPC isolation.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import importlib.util
import os
import select
import signal
import socket
import stat
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "scripts/run_sft_netns_fixture_v1.py"
FROZEN_HASH = "70400a1428d0e54834721a5163db7724940a271bbeeeaec50714f28108f20bfc"
FROZEN_SIZE = 16067
SCHEMA = "lumi.sft-filesystem-linux-fixture.v0.1"
CANARY_BYTES = b"generated-filesystem-canary\n"
NONCLAIMS = (
    "real_model_or_data_access", "cuda_used", "production_eligible",
    "real_sft_execution_permitted", "complete_os_sandbox_verified",
    "full_pathname_unix_isolation_verified", "host_metadata_confidentiality_verified",
    "runtime_origin_authenticated", "real_host_admission_verified",
    "cumulative_write_quota_verified", "host_mounts_modified",
)
STAGES = frozenset({"sanitize", "attach", "netns", "mount", "landlock", "drop",
                    "release", "probes", "worker_run", "complete", "not_reported"})
NET = ORIGINAL_RUN = ORIGINAL_CASE = ORIGINAL_RECEIVE = ORIGINAL_SEND = None
ORIGINAL_CONTROLLER = None
OUTER_PID = HELPER_PID = HELPER_PIDFD = HELPER_CHANNEL = None
HOST_DIRECTORY = HOST_FILE = HOST_SOCKET = ROOT_PATH = CANARY_PATH = None
INJECTED_FILE_FD = INJECTED_DIRECTORY_FD = INJECTED_FILE_IDENTITY = None
FS_STAGE = "not_reported"
FS_CODE = "not_reported"


def safe_code(value):
    if type(value) is not str:
        return False
    if value in {"not_reported", "fixture_failed"}:
        return True
    return type(value) is str and any(value in base().module(name).ERROR_CODES for name in (
        "posttraining_sft_cgroup_v1", "posttraining_sft_netns_v1",
        "posttraining_sft_mount_root_v1", "posttraining_sft_landlock_v1",
        "posttraining_sft_cgroup_guardian_v1",
    ))


def base():
    return NET.BASE


def need(value):
    base().require(value)


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
    spec = importlib.util.spec_from_file_location("_lumi_fs_frozen_netns_v1", FROZEN)
    if spec is None:
        raise RuntimeError("frozen_binding_failed")
    module = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(FROZEN), "exec"), module.__dict__)
    return module


def identity(path):
    value = os.lstat(path)
    return value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid


def read_canary(path):
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        need(os.read(fd, len(CANARY_BYTES) + 1) == CANARY_BYTES and os.read(fd, 1) == b"")
    finally:
        os.close(fd)


def create_canary(path, uid, gid):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                 0o600)
    try:
        need(os.write(fd, CANARY_BYTES) == len(CANARY_BYTES))
        os.fchown(fd, uid, gid)
        os.fchmod(fd, 0o600)
    finally:
        os.close(fd)


def wait_owned(pid, pidfd, deadline_ns, *, terminate=False):
    if terminate and not select.select([pidfd], [], [], 0)[0]:
        signal.pidfd_send_signal(pidfd, signal.SIGKILL, None, 0)
    remaining = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
    need(remaining > 0 and select.select([pidfd], [], [], remaining)[0])
    observed, status = os.waitpid(pid, 0)
    need(observed == pid)
    return status


def helper(channel, listener, parent_pid, uid, gid, deadline_ns):
    try:
        parent_fd = os.pidfd_open(parent_pid, 0)
        base().close_except({channel.fileno(), listener.fileno(), parent_fd})
        base().module("posttraining_sft_cgroup_guardian_v1").drop_worker_privileges(uid, gid)
        prctl = ctypes.CDLL(None, use_errno=True).prctl
        prctl.restype = ctypes.c_int
        # Only this generated helper's canary/socket authority is made inspectable.
        need(prctl(4, 1, 0, 0, 0) == 0 and prctl(3, 0, 0, 0, 0) == 1)
        ORIGINAL_SEND(channel, {"helper_ready": True})
        while time.monotonic_ns() < deadline_ns:
            ready, _, _ = select.select([channel, listener, parent_fd], [], [], 0.1)
            if parent_fd in ready or channel in ready:
                break
            if listener in ready:
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(1)
                    NET.receive_exact(connection, b"generated-ping")
                    connection.sendall(b"generated-pong")
        os._exit(0)
    except BaseException:
        os._exit(2)


def outside_positive(uid, gid, deadline_ns):
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    pid = os.fork()
    if pid == 0:
        try:
            base().close_except({child.fileno()})
            base().module("posttraining_sft_cgroup_guardian_v1").drop_worker_privileges(uid, gid)
            read_canary(HOST_FILE)
            read_canary(f"/proc/{HELPER_PID}/root{HOST_FILE}")
            fd = os.open(CANARY_PATH, os.O_WRONLY | os.O_TRUNC | os.O_CLOEXEC)
            try:
                need(os.write(fd, CANARY_BYTES) == len(CANARY_BYTES))
            finally:
                os.close(fd)
            os.chmod(CANARY_PATH, 0o600)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(1)
                connection.connect(HOST_SOCKET)
                connection.sendall(b"generated-ping")
                NET.receive_exact(connection, b"generated-pong")
            ORIGINAL_SEND(child, {"outside_same_uid_positive": True})
            os._exit(0)
        except BaseException:
            os._exit(2)
    child.close()
    pidfd = os.pidfd_open(pid, 0)
    completed = False
    try:
        need(ORIGINAL_RECEIVE(parent, deadline_ns) == {"outside_same_uid_positive": True})
        status = wait_owned(pid, pidfd, deadline_ns)
        completed = True
        need(status == 0)
    finally:
        if not completed:
            wait_owned(pid, pidfd, time.monotonic_ns() + 5_000_000_000, terminate=True)
        os.close(pidfd)
        parent.close()


def stop_helper():
    global HELPER_PID, HELPER_PIDFD, HELPER_CHANNEL
    if HELPER_PID is None:
        return
    pid, pidfd, channel = HELPER_PID, HELPER_PIDFD, HELPER_CHANNEL
    HELPER_PID = HELPER_PIDFD = HELPER_CHANNEL = None
    try:
        try:
            requested = channel.send(b"S") == 1
        except OSError:
            requested = False
        exited = bool(select.select([pidfd], [], [], 2)[0])
        status = wait_owned(pid, pidfd, time.monotonic_ns() + 5_000_000_000,
                            terminate=not exited)
        need(requested and exited and status == 0)
    finally:
        os.close(pidfd)
        channel.close()


def receive(channel, deadline_ns):
    global FS_STAGE, FS_CODE
    value = ORIGINAL_RECEIVE(channel, deadline_ns)
    if value.get("filesystem_stage") in STAGES:
        FS_STAGE = value["filesystem_stage"]
    if safe_code(value.get("filesystem_code")):
        FS_CODE = value["filesystem_code"]
    if os.getpid() == OUTER_PID and "empty_and_removed" in value:
        base().COMPONENT_CODE = "filesystem_" + FS_STAGE + "_" + FS_CODE
        # Frozen wait_statuses(-1) must not encounter our unrelated live helper.
        stop_helper()
    return value


def send(channel, value):
    if "empty_and_removed" in value:
        value = dict(value, filesystem_stage=FS_STAGE, filesystem_code=FS_CODE)
    return ORIGINAL_SEND(channel, value)


def denied(action, errors=(errno.EACCES, errno.EPERM, errno.EROFS)):
    try:
        value = action()
    except OSError as error:
        need(error.errno in errors)
    else:
        if type(value) is int:
            os.close(value)
        raise base().FixtureError()


def controller(*arguments):
    observed = os.fstat(INJECTED_FILE_FD)
    need(stat.S_ISREG(observed.st_mode)
         and (observed.st_dev, observed.st_ino) == INJECTED_FILE_IDENTITY)
    os.dup2(INJECTED_FILE_FD, 1)
    # The old controller closes the deliberately inherited high directory FD;
    # the workload must additionally sanitize the regular file in standard fd1.
    return ORIGINAL_CONTROLLER(*arguments)


def filesystem_probes():
    read_canary(CANARY_PATH)
    denied(lambda: os.open(HOST_FILE, os.O_RDONLY), (errno.ENOENT,))
    denied(lambda: os.open(f"/proc/{HELPER_PID}/root{HOST_FILE}", os.O_RDONLY))
    denied(lambda: os.open("/.oldroot", os.O_RDONLY), (errno.ENOENT,))
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(1)
        denied(lambda: connection.connect(HOST_SOCKET), (errno.ENOENT,))
        denied(lambda: connection.connect(f"/proc/{HELPER_PID}/root{HOST_SOCKET}"))
    # These operations target ONLY this run's generated, worker-owned canary.
    denied(lambda: os.open(CANARY_PATH, os.O_WRONLY | os.O_TRUNC))
    denied(lambda: os.chmod(CANARY_PATH, 0o400))
    denied(lambda: os.unlink(CANARY_PATH))
    denied(lambda: os.rename(CANARY_PATH, CANARY_PATH + ".moved"))
    denied(lambda: os.link(CANARY_PATH, CANARY_PATH + ".linked"))
    read_canary(CANARY_PATH)
    fd = os.open("/tmp/generated-roundtrip", os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        need(os.write(fd, CANARY_BYTES) == len(CANARY_BYTES))
        os.lseek(fd, 0, os.SEEK_SET)
        need(os.read(fd, len(CANARY_BYTES) + 1) == CANARY_BYTES)
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


def workload(case, group_fd, channel, uid, gid, runtime, deadline_ns):
    global FS_STAGE, FS_CODE
    try:
        FS_STAGE = "sanitize"
        inherited = os.fstat(1)
        need(stat.S_ISREG(inherited.st_mode)
             and (inherited.st_dev, inherited.st_ino) == INJECTED_FILE_IDENTITY)
        NET.sanitize_standard_descriptors(group_fd, channel)
        FS_STAGE = "attach"
        base().module("posttraining_sft_cgroup_v1").enter_current_process(group_fd)
        FS_STAGE = "netns"
        need(base().module("posttraining_sft_netns_v1").create_loopback_namespace()
             == NET.NAMESPACE_RECEIPT)
        FS_STAGE = "mount"
        runtime_root = str(Path(runtime).parents[1])
        scratch = (128 if case == "worker_run" else 8) * 1024**2
        mounted = base().module("posttraining_sft_mount_root_v1").enter_mount_root(
            root_path=ROOT_PATH, runtime_root=runtime_root, source_root=str(ROOT / "src"),
            script_path=str(base().ACCEPTED_SCRIPT), uid=uid, gid=gid, scratch_bytes=scratch,
        )
        need(mounted == {"schema": "lumi.sft-mount-root.v0.1", "readonly_root": True,
                         "readonly_binds": True, "old_root_detached": True,
                         "private_propagation": True, "private_tmpfs": True,
                         "scratch_capacity_bytes": scratch, "host_mounts_modified": False,
                         "real_sft_execution_permitted": False, "production_eligible": False})
        FS_STAGE = "landlock"
        policy = base().module("posttraining_sft_landlock_v1").restrict_filesystem(
            read_only_paths=("/usr", runtime_root, str(ROOT / "src"), str(base().ACCEPTED_SCRIPT),
                             "/etc/ld.so.cache", "/etc/hosts", "/etc/nsswitch.conf",
                             "/etc/passwd", "/etc/group", "/proc"),
            writable_paths=("/tmp",),
            device_paths=("/dev/null", "/dev/zero", "/dev/random", "/dev/urandom"),
        )
        need(type(policy.get("abi_version")) is int and policy["abi_version"] >= 5)
        need(policy == {"schema": "lumi.sft-landlock.v0.1", "abi_version": policy["abi_version"],
                        "filesystem_content_policy_enforced": True,
                        "ptrace_domain_restricted": True,
                        "full_pathname_unix_isolation_verified": False,
                        "metadata_policy_alone_sufficient": False,
                        "real_sft_execution_permitted": False, "production_eligible": False})
        FS_STAGE = "drop"
        base().module("posttraining_sft_cgroup_guardian_v1").drop_worker_privileges(uid, gid)
        os.close(group_fd)
        base().close_except({channel.fileno()})
        NET.assert_descriptors({0, 1, 2, channel.fileno()})
        FS_STAGE = "release"
        base().send(channel, {"phase": "ready", "uid_gid_dropped": True,
                              "authority_fds_closed": True})
        need(select.select([channel], [], [], 10)[0] and channel.recv(1) == b"G")
        if case == "worker_run":
            FS_STAGE = "worker_run"
            os.dup2(channel.fileno(), 1)
            base().close_except(set())
            os.execve(runtime, [runtime, "-I", "-B", str(base().ACCEPTED_SCRIPT),
                                "--synthetic-only"], base().worker_environment(runtime))
        elif case == "parent_loss":
            base().parent_loss_payload(channel)
        elif case == "deadline":
            time.sleep(30)
            raise base().FixtureError()
        else:
            need(case == "normal")
            FS_STAGE = "probes"
            NET.network_probes()
            filesystem_probes()
            NET.assert_descriptors({0, 1, 2, channel.fileno()})
            FS_STAGE = "complete"
            base().send(channel, dict(NET.PROBE_RECEIPT, filesystem_verified=True))
        os._exit(0)
    except BaseException as error:
        try:
            FS_CODE = "fixture_failed"
            for module_name, error_type in (
                ("posttraining_sft_cgroup_v1", "CgroupError"),
                ("posttraining_sft_netns_v1", "NetworkIsolationError"),
                ("posttraining_sft_mount_root_v1", "MountRootError"),
                ("posttraining_sft_landlock_v1", "LandlockError"),
                ("posttraining_sft_cgroup_guardian_v1", "GuardianError"),
            ):
                implementation = base().module(module_name)
                if type(error) is getattr(implementation, error_type):
                    need(error.code in implementation.ERROR_CODES)
                    FS_CODE = error.code
                    break
            base().send(channel, {"phase": "filesystem_failed", "filesystem_stage": FS_STAGE,
                                  "filesystem_code": FS_CODE})
        except BaseException:
            pass
        os._exit(2)


def verify_case(case, watched, snapshot, statuses, worker_pid, messages):
    global FS_STAGE, FS_CODE
    if messages and messages[0].get("filesystem_stage") in STAGES:
        FS_STAGE = messages[0]["filesystem_stage"]
        if safe_code(messages[0].get("filesystem_code")):
            FS_CODE = messages[0]["filesystem_code"]
    if case == "normal":
        need(messages == [dict(NET.PROBE_RECEIPT, filesystem_verified=True)])
        messages = [{"phase": "done", "generated_work_completed": True}]
    return NET.FROZEN_VERIFY_CASE(case, watched, snapshot, statuses, worker_pid, messages)


def case_wrapper(case, uid, gid, runtime, deadline_ns):
    global ROOT_PATH, HOST_DIRECTORY, HOST_FILE, HOST_SOCKET, CANARY_PATH
    global HELPER_PID, HELPER_PIDFD, HELPER_CHANNEL, FS_STAGE, FS_CODE
    global INJECTED_FILE_FD, INJECTED_DIRECTORY_FD, INJECTED_FILE_IDENTITY
    FS_STAGE = FS_CODE = "not_reported"
    # A previous case's receipt cannot describe this case's host-side objects.
    base().REMOVAL_REPORTED = False
    base().CURRENT_CASE, base().STAGE = case, "filesystem_host_preflight"
    tag = uuid.uuid4().hex
    ROOT_PATH = "/tmp/lumi-sft-root-" + tag
    HOST_DIRECTORY = "/tmp/lumi-sft-fs-sentinel-" + tag
    HOST_FILE, HOST_SOCKET = HOST_DIRECTORY + "/canary", HOST_DIRECTORY + "/socket"
    CANARY_PATH = str(ROOT / "src" / (".lumi-SYNTHETIC-fs-" + tag))
    os.mkdir(ROOT_PATH, 0o700)
    root_identity = identity(ROOT_PATH)
    os.mkdir(HOST_DIRECTORY, 0o700)
    os.chown(HOST_DIRECTORY, uid, gid)
    directory_identity = identity(HOST_DIRECTORY)
    create_canary(HOST_FILE, uid, gid)
    create_canary(CANARY_PATH, uid, gid)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(HOST_SOCKET)
    os.chown(HOST_SOCKET, uid, gid)
    os.chmod(HOST_SOCKET, 0o600)
    listener.listen(1)
    channels = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    parent_ns = os.stat("/proc/self/ns/mnt")
    parent_root = os.stat("/")
    before = {path: identity(path) for path in (HOST_FILE, HOST_SOCKET, CANARY_PATH)}
    pid = os.fork()
    if pid == 0:
        helper(channels[1], listener, OUTER_PID, uid, gid, deadline_ns)
        os._exit(2)
    channels[1].close()
    HELPER_PID, HELPER_PIDFD, HELPER_CHANNEL = pid, os.pidfd_open(pid, 0), channels[0]
    try:
        need(ORIGINAL_RECEIVE(channels[0], deadline_ns) == {"helper_ready": True})
        outside_positive(uid, gid, min(deadline_ns, time.monotonic_ns() + 5_000_000_000))
        INJECTED_FILE_FD = os.open(HOST_FILE, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        injected = os.fstat(INJECTED_FILE_FD)
        INJECTED_FILE_IDENTITY = injected.st_dev, injected.st_ino
        INJECTED_DIRECTORY_FD = os.open(HOST_DIRECTORY, os.O_RDONLY | os.O_DIRECTORY
                                        | os.O_CLOEXEC | os.O_NOFOLLOW)
        base().STAGE = "cases"
        result = ORIGINAL_CASE(case, uid, gid, runtime, deadline_ns)
        need(HELPER_PID is None)
        after_ns, after_root = os.stat("/proc/self/ns/mnt"), os.stat("/")
        need((parent_ns.st_dev, parent_ns.st_ino) == (after_ns.st_dev, after_ns.st_ino))
        need((parent_root.st_dev, parent_root.st_ino) == (after_root.st_dev, after_root.st_ino))
        read_canary(HOST_FILE)
        read_canary(CANARY_PATH)
        need(all(identity(path) == value for path, value in before.items()))
        need(identity(ROOT_PATH) == root_identity and os.listdir(ROOT_PATH) == [])
        return result
    finally:
        cgroup_removed = base().REMOVAL_REPORTED
        base().REMOVAL_REPORTED = False
        cleanup_failed = False
        try:
            stop_helper()
        except BaseException:
            cleanup_failed = True
        try:
            listener.close()
        except BaseException:
            cleanup_failed = True
        for fd in (INJECTED_FILE_FD, INJECTED_DIRECTORY_FD):
            if fd is not None:
                try:
                    os.close(fd)
                except BaseException:
                    cleanup_failed = True
        INJECTED_FILE_FD = INJECTED_DIRECTORY_FD = INJECTED_FILE_IDENTITY = None
        # Exact owned targets only. No recursive deletion or broad process scan.
        for path, expected in before.items():
            try:
                observed = identity(path)
                need(observed[:2] == expected[:2])
                os.unlink(path)
            except BaseException:
                cleanup_failed = True
        for path, expected in ((HOST_DIRECTORY, directory_identity), (ROOT_PATH, root_identity)):
            try:
                need(identity(path) == expected)
                os.rmdir(path)
            except BaseException:
                cleanup_failed = True
        if cleanup_failed:
            raise base().FixtureError("cleanup_unconfirmed") from None
        base().REMOVAL_REPORTED = cgroup_removed


def run():
    global ORIGINAL_CASE, ORIGINAL_RECEIVE, ORIGINAL_SEND, OUTER_PID
    OUTER_PID = os.getpid()
    ORIGINAL_CASE, ORIGINAL_RECEIVE, ORIGINAL_SEND = (
        base().run_case, base().receive, base().send,
    )
    base().run_case, base().receive, base().send = case_wrapper, receive, send
    report = ORIGINAL_RUN()
    need(report["cases_verified"] == report["owned_cgroups_empty_and_removed"] == 4)
    report.update(schema_version=SCHEMA, **dict.fromkeys(NONCLAIMS, False))
    report.update(filesystem_view_verified=True, landlock_policy_verified=True,
                  same_uid_host_proc_root_boundary_verified=True, readonly_canary_verified=True,
                  bounded_tmpfs_exhaustion_verified=True, owned_mountpoints_removed=4,
                  inherited_file_directory_authority_removed=True,
                  owned_sentinel_processes_reaped=4, host_root_and_mount_namespace_unchanged=True)
    return report


def main():
    global NET, ORIGINAL_RUN, ORIGINAL_CONTROLLER
    try:
        NET = load_frozen()
    except BaseException:
        os.write(1, b'{"schema_version":"lumi.sft-filesystem-linux-fixture.v0.1",'
                    b'"status":"failed","code":"frozen_binding_failed"}\n')
        return 1
    ORIGINAL_RUN = NET.run
    ORIGINAL_CONTROLLER = NET.controller
    NET.run, NET.workload, NET.verify_case = run, workload, verify_case
    NET.controller = controller
    NET.SCHEMA = SCHEMA
    NET.NONCLAIMS = tuple(dict.fromkeys((*NET.NONCLAIMS, *NONCLAIMS)))
    return NET.main()


if __name__ == "__main__":
    raise SystemExit(main())
