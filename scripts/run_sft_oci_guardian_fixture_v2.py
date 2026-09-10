#!/usr/bin/env python3
"""One new Labels-contract-corrected accepted-image guardian composition.

NEVER execute this native fixture on a Mac. Local tests inspect source or call
pure functions with doubles. A fresh trusted Linux CI host and exclusive local
Docker daemon are premises, not independently authenticated here. Docker only
loads, creates, inspects and exports software: it never starts a container.
Actual workloads attach to guardian-owned cgroups before namespaces and exec.

One positive and two deliberately wrong identity-pin cases, sequentially, with
no retry. Rootfs software bytes come from the accepted image; two separately
bound adapter files are added. Kernel-generated tmpfs/proc/device overlays are
not release software. Original accepted files and all historical evidence stay
unchanged. No model/data, GPU, package installation, network workload or real SFT.
Only fixed stages/codes, identities and aggregate counts are emitted.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import resource
import select
import socket
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "lumi.sft-oci-guardian-fixture.v0.2"
ADMISSION_SOURCE = "src/lumi/posttraining_sft_oci_admission_v2.py"
ADMISSION_SHA256 = "d57c87ecdcb31058e779deb72eba7928930d1edd5b99c0752d112edcbefc6136"
FROZEN = "scripts/run_sft_cgroup_fixture_v1.py"
FROZEN_SHA256 = "f91ed0c353a20c309ff3e1bfaab8dbe990b28c36626fdeeb0ac5131e0cf75fad"
ARCHIVE_BYTES = 1_062_113_280
ARCHIVE_SHA256 = "99d21fc60a54a65f28ceb4780f603f7ee48e49ac7b6d00a053dcdabe9ebc9ff4"
ADAPTER = "scripts/run_sft_oci_guarded_worker_v1.py"
WORKER_PATH = "/opt/lumi-release/guardian/worker_entry.py"
CONTROL_PATH = "/opt/lumi-release/guardian/admission.json"
MAXIMUM_SECONDS = 900
MAXIMUM_EXPORT_BYTES = 2 * 1024**3
MAXIMUM_EXTRACT_BYTES = 4 * 1024**3
MAXIMUM_WORKER_BYTES = 32768
CASES = ("wrong_control", "wrong_adapter", "positive")
STAGES = ("preflight", "copy_image", "load_image", "create_container", "inspect_container",
          "export_rootfs", "extract_rootfs", "install_adapter", "cases", "cleanup", "complete")
ERROR_CODES = frozenset({"fixture_failed", "runtime_rejected", "source_rejected",
                         "deadline_exhausted", "command_failed", "output_exhausted",
                         "worker_rejected", "cleanup_unconfirmed", "admission_rejected",
                         "rootfs_rejected"})
NONCLAIMS = ("real_sft_execution_permitted", "production_eligible", "real_model_or_data_access",
             "gpu_used", "registry_published", "private_integration_passed", "full_ci_passed",
             "real_host_admission_verified", "complete_dynamic_dependency_closure_verified",
             "publisher_signature_verified", "guardian_death_containment")
BASE = None
ADMISSION = None
ROOTFS = None
BASE_RECEIVE = None
LAST_BOOTSTRAP_STAGE = "not_reported"
BOOTSTRAP_STAGES = frozenset({"sanitize", "attach", "network", "mount", "landlock", "ipc",
                              "drop", "release"})
CONTEXT = None
STAGE = "preflight"
DESCENDANTS_REAPED = True


class FixtureError(ValueError):
    def __init__(self, code="fixture_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "fixture_failed"
        super().__init__(self.code)


def need(value, code="fixture_failed"):
    if not value:
        raise FixtureError(code) from None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"


def remaining(deadline):
    value = deadline - time.monotonic()
    need(value > 0, "deadline_exhausted")
    return value


def identity(info):
    return info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode), info.st_uid, info.st_gid


def file_state(info):
    # Reading can update atime; it must not create a false content-change failure.
    return (*identity(info), info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def read_source(relative, expected=None):
    path = ROOT / relative
    for ancestor in (path, *path.parents):
        need(not ancestor.is_symlink(), "source_rejected")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= 131072, "source_rejected")
        raw = bytearray()
        while len(raw) < before.st_size:
            chunk = os.read(fd, min(65536, before.st_size - len(raw)))
            need(chunk, "source_rejected")
            raw.extend(chunk)
        after = os.fstat(fd)
        need(file_state(before) == file_state(after) and not os.read(fd, 1), "source_rejected")
        need(expected is None or hashlib.sha256(raw).hexdigest() == expected, "source_rejected")
        return bytes(raw)
    finally:
        os.close(fd)


def load_base():
    raw = read_source(FROZEN, FROZEN_SHA256)
    spec = importlib.util.spec_from_file_location("_lumi_oci_guardian_base_v1", ROOT / FROZEN)
    need(spec is not None)
    selected = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = selected
    exec(compile(raw, str(ROOT / FROZEN), "exec"), selected.__dict__)
    return selected


def load_admission():
    raw = read_source(ADMISSION_SOURCE, ADMISSION_SHA256)
    spec = importlib.util.spec_from_file_location("_lumi_oci_admission_v2", ROOT / ADMISSION_SOURCE)
    need(spec is not None, "source_rejected")
    selected = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = selected
    exec(compile(raw, str(ROOT / ADMISSION_SOURCE), "exec"), selected.__dict__)
    return selected


def failure_fields(error):
    if type(error) is FixtureError:
        return {"code": error.code}
    if ADMISSION is not None and type(error) is ADMISSION.OciAdmissionError:
        if (type(error.code) is str and error.code in ADMISSION.ERROR_CODES
                and type(error.predicate) is str and error.predicate in ADMISSION.PREDICATE_CODES):
            return {"code": "admission_rejected", "admission_code": error.code,
                    "admission_predicate": error.predicate}
    if ROOTFS is not None and type(error) is ROOTFS.RootfsError:
        if type(error.code) is str and error.code in ROOTFS.ERROR_CODES:
            return {"code": "rootfs_rejected", "rootfs_code": error.code}
    return {"code": "fixture_failed"}


def receive_with_stage(channel, deadline):
    global LAST_BOOTSTRAP_STAGE
    value = BASE_RECEIVE(channel, deadline)
    if (type(value) is dict and value.get("status") == "failed"
            and value.get("case") == "worker_run"
            and type(value.get("bootstrap_stage")) is str
            and value["bootstrap_stage"] in BOOTSTRAP_STAGES):
        LAST_BOOTSTRAP_STAGE = value["bootstrap_stage"]
    return value  # Observe one existing bounded packet; never alter or reread it.


def control_bytes(binding):
    return canonical({
        "schema": "lumi.sft-guarded-oci-worker-admission.v0.1",
        "software_binding": {name: getattr(binding, name) for name in binding.__slots__},
        "source_identity_sha256": binding.source_identity_sha256(),
    })


def worker_environment(binding, adapter_sha256, control_sha256, case):
    need(case in CASES)
    values = {
        "PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "HOME": "/tmp", "HF_HOME": "/tmp/huggingface", "XDG_CACHE_HOME": "/tmp/cache",
        "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "AWS_EC2_METADATA_DISABLED": "true",
        "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "GLOO_SOCKET_IFNAME": "lo", "TP_SOCKET_IFNAME": "lo",
        "LUMI_WORKER_RUN_TORCH_CI_FIXTURE": "1",
        "LUMI_RELEASE_LOCK_SHA256": binding.software_lock_sha256,
        "LUMI_RELEASE_BASE_INVENTORY_SHA256": binding.base_inventory_sha256,
        "LUMI_RELEASE_SOURCE_INVENTORY_SHA256": binding.source_inventory_sha256,
        "LUMI_RELEASE_SOURCE_COMMIT": binding.source_commit,
        "LUMI_GUARDIAN_ADAPTER_SHA256": adapter_sha256,
        "LUMI_GUARDIAN_CONTROL_SHA256": control_sha256,
    }
    if case == "wrong_control":
        values["LUMI_GUARDIAN_CONTROL_SHA256"] = "0" * 64
    elif case == "wrong_adapter":
        values["LUMI_GUARDIAN_ADAPTER_SHA256"] = "0" * 64
    return values


def command(config, arguments, *, deadline, maximum=65536, input_file=None, output_file=None,
            accepted_statuses=(0,)):
    """A single bounded local Docker CLI; never log stdout/stderr or retry."""
    remaining(deadline)  # An exhausted budget must not launch a daemon write at all.
    process = subprocess.Popen(
        ["/usr/bin/docker", "--host", "unix:///var/run/docker.sock", "--config", str(config),
         *arguments], stdin=subprocess.DEVNULL if input_file is None else input_file,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, close_fds=True,
        env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
    raw, total, digest = bytearray(), 0, hashlib.sha256()
    try:
        while True:
            need(select.select([process.stdout], [], [], remaining(deadline))[0],
                 "deadline_exhausted")
            chunk = os.read(process.stdout.fileno(), min(65536, maximum + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            need(total <= maximum, "output_exhausted")
            digest.update(chunk)
            if output_file is None:
                raw.extend(chunk)
            else:
                need(output_file.write(chunk) == len(chunk), "output_exhausted")
        status = process.wait(timeout=remaining(deadline))
        need(status in accepted_statuses, "command_failed")
        return bytes(raw), total, digest.hexdigest(), status
    finally:
        try:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)
        finally:
            process.stdout.close()


def copy_image(source, target, deadline, file_ids):
    # The downloaded user-owned archive is never passed directly to root Docker.
    # Copy once into a private root-owned file while checking its fixed bytes.
    need(source.is_absolute() and ".." not in source.parts, "source_rejected")
    for ancestor in (source, *source.parents):
        need(not ancestor.is_symlink(), "source_rejected")
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_size == ARCHIVE_BYTES, "source_rejected")
        size, digest = 0, hashlib.sha256()
        with target.open("xb") as output:
            file_ids[target] = identity(os.fstat(output.fileno()))
            while size < ARCHIVE_BYTES:
                remaining(deadline)
                chunk = os.read(fd, min(1024**2, ARCHIVE_BYTES - size))
                need(chunk, "source_rejected")
                need(output.write(chunk) == len(chunk), "source_rejected")
                size += len(chunk)
                digest.update(chunk)
            need(not os.read(fd, 1) and file_state(before) == file_state(os.fstat(fd)),
                 "source_rejected")
            need(digest.hexdigest() == ARCHIVE_SHA256, "source_rejected")
            os.fchmod(output.fileno(), 0o400)
        return size
    finally:
        os.close(fd)


def create_options(image_id, container_name):
    need(re.fullmatch(r"lumi-oci-guardian-[0-9a-f]{32}", container_name) is not None)
    return ["create", "--pull=never", "--platform=linux/amd64", "--network=none", "--read-only",
            "--user=10001:10001", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--cpus=2", "--memory=4g", "--memory-swap=4g", "--pids-limit=256",
            "--ulimit=core=0", "--ulimit=nofile=1024:1024", "--ipc=private",
            "--restart=no", "--shm-size=67108864",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=536870912,mode=1777",
            "--name=" + container_name, image_id]


def image_inventory(config, deadline):
    raw, _size, _digest, _status = command(
        config, ["image", "ls", "--all", "--no-trunc", "--quiet"], deadline=deadline,
    )
    values = raw.decode("ascii").splitlines()
    need(all(re.fullmatch(r"sha256:[0-9a-f]{64}", value) for value in values))
    return frozenset(values)


def container_inventory(config, deadline):
    raw, _size, _digest, _status = command(
        config, ["container", "ls", "--all", "--format", "{{.Names}}"], deadline=deadline,
    )
    values = raw.decode("ascii").splitlines()
    need(all(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,255}", value) for value in values))
    return frozenset(values)


def sanitize(group_fd, channel):
    need(group_fd > 2 and channel.fileno() > 2 and group_fd != channel.fileno())
    fd = os.open("/dev/null", os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        need(stat.S_ISCHR(info.st_mode) and (os.major(info.st_rdev), os.minor(info.st_rdev))
             == (1, 3))
        for number in (0, 1, 2):
            os.dup2(fd, number)
    finally:
        if fd > 2:
            os.close(fd)
    BASE.close_except({group_fd, channel.fileno()})


def workload(group_fd, channel):
    stage = "sanitize"
    try:
        sanitize(group_fd, channel)
        stage = "attach"
        BASE.module("posttraining_sft_cgroup_v1").enter_current_process(group_fd)
        stage = "network"
        network = BASE.module("posttraining_sft_netns_v1").create_loopback_namespace()
        need(network["namespace_changed"] and network["only_loopback_interface"])
        stage = "mount"
        mounted = BASE.module("posttraining_sft_oci_mount_v1").enter_oci_rootfs(
            str(CONTEXT["rootfs"]), str(CONTEXT["mountpoint"]), uid=10001, gid=10001,
        )
        need(mounted["old_root_detached"] and mounted["readonly_root"])
        stage = "landlock"
        policy = BASE.module("posttraining_sft_oci_landlock_v1").restrict_filesystem(
            read_only_paths=("/usr", "/etc", "/opt/lumi-release", "/proc"),
            writable_paths=("/tmp", "/dev/shm"),
            device_paths=("/dev/null", "/dev/zero", "/dev/random", "/dev/urandom"),
            expected_root_identity=CONTEXT["root_identity"],
        )
        need(policy["filesystem_content_policy_enforced"] and policy["abi_version"] >= 5)
        stage = "ipc"
        ipc = BASE.module("posttraining_sft_ipc_filter_v1")
        need(ipc.install_ipc_filter() == dict(ipc.RECEIPT))
        stage = "drop"
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 1024))
        BASE.module("posttraining_sft_cgroup_guardian_v1").drop_worker_privileges(10001, 10001)
        os.close(group_fd)
        BASE.close_except({channel.fileno()})
        need(os.getresuid() == os.getresgid() == (10001, 10001, 10001) and os.getgroups() == [])
        # listdir's own transient fd is gone when this independently opened
        # descriptor inventory is inspected via fstat.
        opened = set()
        for item in os.listdir("/proc/self/fd"):
            try:
                os.fstat(int(item))
                opened.add(int(item))
            except OSError:
                pass
        need(opened == {0, 1, 2, channel.fileno()})
        BASE.send(channel, {"phase": "ready", "isolation_installed": True,
                            "uid_gid_dropped": True, "authority_fds_closed": True})
        stage = "release"
        need(select.select([channel], [], [], 10)[0] and channel.recv(1) == b"G")
        os.chdir("/opt/lumi-release/source")
        os.dup2(channel.fileno(), 1)
        BASE.close_except(set())
        os.execve("/usr/local/bin/python3.12",
                  ["/usr/local/bin/python3.12", "-I", "-B", WORKER_PATH, "--synthetic-only"],
                  CONTEXT["environment"])
    except BaseException:
        try:
            BASE.send(channel, {"phase": "bootstrap_failed", "stage": stage})
        except BaseException:
            pass
        os._exit(2)


def worker_message(channel):
    need(select.select([channel], [], [], 0)[0], "worker_rejected")
    raw, _ancillary, flags, _address = channel.recvmsg(MAXIMUM_WORKER_BYTES)
    need(raw and not flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC), "worker_rejected")
    value = json.loads(raw)
    need(type(value) is dict, "worker_rejected")
    # Exactly one packet, then closed. Raw worker/exception text is not retained.
    need(select.select([channel], [], [], 0)[0] and channel.recv(1) == b"", "worker_rejected")
    return value


def validate_worker(value, case, status):
    # The versioned worker adapter supplies its own exact report validator.
    # Loaded only from the same locally checked adapter bytes, without entry().
    if case != "wrong_adapter":
        need(value.get("adapter_sha256") == CONTEXT["adapter_sha256"], "worker_rejected")
    if case == "positive":
        need(value.get("control_sha256") == CONTEXT["control_sha256"], "worker_rejected")
    return CONTEXT["adapter_module"].validate_report(
        value, case=case, source_identity_sha256=CONTEXT["binding"].source_identity_sha256(),
        wait_status=status,
    )


def guardian(case, channel, controller_pid, _uid, _gid, _runtime, deadline_ns):
    BASE.close_except({channel.fileno()})
    BASE.subreaper()
    parent_fd = lease = pair = None
    retired, kills, final = False, [0], None
    bootstrap_stage = "not_reported"
    try:
        parent_fd = os.open("/sys/fs/cgroup", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                            | os.O_CLOEXEC | os.O_NONBLOCK)
        lease = BASE.module("posttraining_sft_cgroup_v1").CgroupLease(
            parent_fd, name=f"lumi-sft-{uuid.uuid4().hex}", limits={
                "memory_max_bytes": 4294967296, "pids_max": 256,
                "cpu_quota_us": 200000, "cpu_period_us": 100000,
            },
        )
        kill = lease.kill

        def kill_once():
            need(kills[0] == 0)
            kills[0] += 1
            return kill()

        lease.kill = kill_once
        lease.verify_limits()
        need(lease.snapshot()["populated"] == 0)
        pair, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        group_fd = lease.fileno()
        worker_pid = os.fork()
        if worker_pid == 0:
            try:
                workload(group_fd, child)
            finally:
                os._exit(2)
        child.close()
        startup = BASE.receive(pair, deadline_ns)
        if (type(startup) is dict and set(startup) == {"phase", "stage"}
                and startup["phase"] == "bootstrap_failed" and type(startup["stage"]) is str
                and startup["stage"] in BOOTSTRAP_STAGES):
            bootstrap_stage = startup["stage"]
        need(startup == {
            "phase": "ready", "isolation_installed": True,
            "uid_gid_dropped": True, "authority_fds_closed": True,
        })
        snapshot = lease.snapshot()
        need(snapshot["populated"] == snapshot["pids_current"] == 1)

        def release():
            need(pair.send(b"G") == 1)
            BASE.send(channel, {"phase": "armed", "case": case})

        watched = BASE.module("posttraining_sft_cgroup_guardian_v1").watch_parent(
            lease, expected_parent_pid=controller_pid, deadline_ns=deadline_ns,
            cleanup_timeout_ms=10000, release=release,
        )
        need(watched["reason"] == "workload_empty" and kills[0] == 1
             and all(watched[key] is True for key in
                     ("parent_pidfd_bound", "release_called", "kill_attempted",
                      "empty_verified", "pidfd_closed")))
        statuses = BASE.wait_statuses(time.monotonic_ns() + 10_000_000_000)
        snapshot = lease.snapshot()
        need(snapshot["populated"] == snapshot["pids_current"] == 0
             and worker_pid in statuses and 1 <= len(statuses) <= 4)
        receipt = validate_worker(worker_message(pair), CONTEXT["case"], statuses[worker_pid])
        lease.close()
        retired = True
        final = {"status": "passed", "case": case, "empty_and_removed": True,
                 "workload_processes_reaped": len(statuses), "worker": receipt,
                 "guardian_parent_bound": True, "owned_cgroup_kill_attempts": kills[0]}
    except BaseException:
        final = {"status": "failed", "case": case, "code": "worker_rejected",
                 "empty_and_removed": False}
    finally:
        if lease is not None and not retired:
            try:
                if kills[0] == 0:
                    lease.kill()
                stop = time.monotonic_ns() + 10_000_000_000
                while lease.snapshot()["populated"] and time.monotonic_ns() < stop:
                    time.sleep(0.02)
                need(lease.snapshot()["populated"] == 0)
                BASE.wait_statuses(stop)
                lease.close()
                retired = True
            except BaseException:
                pass
        if pair is not None:
            pair.close()
        if parent_fd is not None:
            os.close(parent_fd)
        if not retired:
            final = {"status": "failed", "case": case, "code": "cleanup_unconfirmed",
                     "empty_and_removed": False}
        elif final["status"] == "failed":
            final["empty_and_removed"] = True
        if final["status"] == "failed" and bootstrap_stage in BOOTSTRAP_STAGES:
            final["bootstrap_stage"] = bootstrap_stage
        BASE.send(channel, final)
        channel.close()
    os._exit(0 if final["status"] == "passed" else 2)


def cases(root_path, mountpoint, binding, adapter_raw, control_raw, deadline):
    global CONTEXT, DESCENDANTS_REAPED, BASE_RECEIVE
    spec = importlib.util.spec_from_file_location(
        "_lumi_guarded_adapter_validator_v1", ROOT / ADAPTER,
    )
    need(spec is not None)
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    exec(compile(adapter_raw, str(ROOT / ADAPTER), "exec"), adapter.__dict__)
    BASE.guardian = guardian
    need(BASE_RECEIVE is None)
    BASE_RECEIVE = BASE.receive
    BASE.receive = receive_with_stage
    BASE.subreaper()
    reports = []
    adapter_sha256 = hashlib.sha256(adapter_raw).hexdigest()
    control_sha256 = hashlib.sha256(control_raw).hexdigest()
    # Load every privileged helper before changing root; none imports ML.
    for name in ("posttraining_sft_cgroup_v1", "posttraining_sft_cgroup_guardian_v1",
                 "posttraining_sft_netns_v1", "posttraining_sft_oci_mount_v1",
                 "posttraining_sft_oci_landlock_v1", "posttraining_sft_ipc_filter_v1"):
        BASE.module(name)
    for case in CASES:
        remaining(deadline)
        environment = worker_environment(binding, adapter_sha256, control_sha256, case)
        CONTEXT = {"case": case, "rootfs": root_path, "mountpoint": mountpoint,
                   "root_identity": identity(root_path.stat()),
                   "binding": binding, "adapter_module": adapter,
                   "adapter_sha256": adapter_sha256, "control_sha256": control_sha256,
                   "environment": environment}
        # Keep the frozen controller's 220-second worker deadline and exact
        # pidfd/subreaper lifecycle. Actual workload case is held privately.
        DESCENDANTS_REAPED = False
        try:
            result = BASE.run_case("worker_run", 10001, 10001, "/usr/local/bin/python3.12",
                                   int(deadline * 1_000_000_000))
        finally:
            # run_case signals only its owned controller on failure. Give the
            # independent guardian time to retire; do not erase its rootfs
            # before every adopted descendant has actually been reaped.
            BASE.wait_statuses(time.monotonic_ns() + 30_000_000_000)
            DESCENDANTS_REAPED = True
        reports.append({"case": case, **{key: value for key, value in result.items()
                                        if key != "case"}})
    return reports


def run(archive_path):
    global BASE, STAGE, ADMISSION, ROOTFS
    need(sys.platform == "linux" and os.uname().machine == "x86_64"
         and sys.version_info[:2] == (3, 12) and sys.flags.no_site and sys.flags.isolated
         and sys.dont_write_bytecode and os.getresuid() == os.getresgid() == (0, 0, 0),
         "runtime_rejected")
    need(os.environ.get("LUMI_OCI_GUARDIAN_V2_CI") == "1", "runtime_rejected")
    os.environ.clear()
    os.environ.update({"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"})
    os.umask(0o077)
    overall_deadline = time.monotonic() + MAXIMUM_SECONDS
    deadline = overall_deadline - 100  # Reserve owned cleanup inside the overall budget.
    BASE = load_base()
    admission = ADMISSION = load_admission()
    rootfs = ROOTFS = BASE.module("posttraining_sft_oci_rootfs_v1")
    adapter_raw = read_source(ADAPTER)
    parent = Path(tempfile.mkdtemp(prefix="lumi-oci-guardian-v2-", dir="/tmp"))
    parent_id = identity(parent.stat())
    config, image, exported, mountpoint = (parent / name for name in
                                         ("docker-config", "image.tar", "rootfs.tar", "mount"))
    directory_ids = {}
    owned, container, result = None, None, None
    load_started, load_completed, create_started, create_completed = False, False, False, False
    retirement_started, retirement_completed = False, False
    container_name = "lumi-oci-guardian-" + uuid.uuid4().hex
    cleanup = False
    cleanup_inventory = {"owned_container_absent": False, "owned_image_absent": False}
    file_ids = {}
    try:
        for path in (config, mountpoint):
            path.mkdir(mode=0o700)
            directory_ids[path] = identity(path.stat())
        space = os.statvfs(parent)
        need(space.f_bavail * space.f_frsize >= 12 * 1024**3)
        STAGE = "copy_image"
        copy_image(archive_path, image, deadline, file_ids)
        STAGE = "load_image"
        # A CLI failure is never interpreted as absence of somebody else's image.
        need(admission.IMAGE_ID not in image_inventory(config, min(deadline, time.monotonic()+10)))
        need(container_name not in container_inventory(config, min(deadline, time.monotonic()+10)))
        with image.open("rb") as source:
            load_started = True
            command(config, ["image", "load"], deadline=min(deadline, time.monotonic() + 180),
                    input_file=source)
        load_completed = True
        STAGE = "create_container"
        create_started = True
        raw, _size, _sha, _status = command(
            config, create_options(admission.IMAGE_ID, container_name),
            deadline=min(deadline, time.monotonic() + 30),
        )
        create_completed = True
        selected = raw.decode("ascii").strip()
        need(re.fullmatch(r"[0-9a-f]{64}", selected) is not None)
        container = selected
        STAGE = "inspect_container"
        binding = admission.verify_created_oci(
            container, deadline=min(deadline, time.monotonic() + 15),
        )
        STAGE = "export_rootfs"
        with exported.open("xb") as output:
            file_ids[exported] = identity(os.fstat(output.fileno()))
            _raw, export_bytes, export_sha, _status = command(
                config, ["export", container], deadline=min(deadline, time.monotonic() + 180),
                maximum=MAXIMUM_EXPORT_BYTES, output_file=output,
            )
        need(export_bytes > 0)
        need(admission.verify_created_oci(container, deadline=min(deadline, time.monotonic()+15))
             == binding)
        retired, container = container_name, None
        create_started = False  # No retry of an ambiguous retirement.
        retirement_started = True
        command(config, ["rm", retired], deadline=min(deadline, time.monotonic() + 15))
        retirement_completed = True
        need(retired not in container_inventory(config, min(deadline, time.monotonic()+10)),
             "cleanup_unconfirmed")
        cleanup_inventory["owned_container_absent"] = True
        STAGE = "extract_rootfs"
        owned = rootfs.extract_rootfs(
            exported, parent, expected_sha256=export_sha,
            maximum_archive_bytes=MAXIMUM_EXPORT_BYTES,
            maximum_file_bytes=1024**3, maximum_total_bytes=MAXIMUM_EXTRACT_BYTES,
            maximum_entries=65536, deadline=deadline - 300,
        )
        STAGE = "install_adapter"
        control_raw = control_bytes(binding)
        for relative, data in ((WORKER_PATH[1:], adapter_raw), (CONTROL_PATH[1:], control_raw)):
            owned.install_file(relative, data, expected_sha256=hashlib.sha256(data).hexdigest(),
                               mode=0o444)
        owned.freeze()
        root_path = owned.path  # Cache before the shorter extraction deadline expires.
        extracted = owned.summary()
        STAGE = "cases"
        reports = cases(root_path, mountpoint, binding, adapter_raw, control_raw, deadline - 30)
        result = {"schema": SCHEMA, "status": "passed", "cases": reports,
                  "accepted_image_id": binding.image_id,
                  "source_identity_sha256": binding.source_identity_sha256(),
                  "archive_bytes": ARCHIVE_BYTES, "export_bytes": export_bytes,
                  "export_sha256": export_sha, "extraction": extracted,
                  "adapter_sha256": hashlib.sha256(adapter_raw).hexdigest(),
                  "control_sha256": hashlib.sha256(control_raw).hexdigest(),
                  "docker_containers_started": 0, "images_built": 0,
                  "actual_workload_cgroup_containment": True,
                  **dict.fromkeys(NONCLAIMS, False)}
    finally:
        previous_stage = STAGE
        STAGE = "cleanup"
        cleanup_deadline = min(overall_deadline, time.monotonic()+95)
        failures = []
        if retirement_started and not retirement_completed:
            failures.append(True)
        if create_started:
            create_started = False
            try:
                if container_name in container_inventory(
                        config, min(cleanup_deadline, time.monotonic()+10)):
                    command(config, ["rm", container_name],
                            deadline=min(cleanup_deadline, time.monotonic()+15))
                # A failed CLI cannot prove the daemon's write has finished.
                need(create_completed, "cleanup_unconfirmed")
                need(container_name not in container_inventory(
                    config, min(cleanup_deadline, time.monotonic()+10)), "cleanup_unconfirmed")
                cleanup_inventory["owned_container_absent"] = True
            except BaseException:
                failures.append(True)
        if load_started:
            load_started = False
            try:
                if admission.IMAGE_ID in image_inventory(
                        config, min(cleanup_deadline, time.monotonic()+10)):
                    command(config, ["image", "rm", "--no-prune", admission.IMAGE_ID],
                            deadline=min(cleanup_deadline, time.monotonic()+30))
                need(load_completed, "cleanup_unconfirmed")
                need(admission.IMAGE_ID not in image_inventory(
                    config, min(cleanup_deadline, time.monotonic()+10)), "cleanup_unconfirmed")
                cleanup_inventory["owned_image_absent"] = True
            except BaseException:
                failures.append(True)
        if not DESCENDANTS_REAPED:
            failures.append(True)
        if owned is not None and DESCENDANTS_REAPED:
            try:
                owned.close()
            except BaseException:
                failures.append(True)
        # Only exact known private files; never recursive delete a directory.
        for path, expected in file_ids.items():
            try:
                need(identity(path.lstat()) == expected)
                path.unlink()
            except BaseException:
                failures.append(True)
        for path in reversed(directory_ids):
            try:
                need(identity(path.lstat()) == directory_ids[path])
                path.rmdir()  # Refuse any unexpected new entry, never clean it broadly.
            except BaseException:
                failures.append(True)
        if not failures:
            need(identity(parent.lstat()) == parent_id)
            parent.rmdir()
            cleanup = True
        need(cleanup, "cleanup_unconfirmed")
        STAGE = previous_stage
    need(result is not None)
    STAGE = "complete"
    need(all(cleanup_inventory.values()), "cleanup_unconfirmed")
    return dict(result, private_owned_cleanup_verified=cleanup, cleanup_inventory=cleanup_inventory)


def main():
    result = None
    try:
        need(len(sys.argv) == 3 and sys.argv[1] == "--synthetic-only")
        result = run(Path(sys.argv[2]))
    except BaseException as error:
        result = {"schema": SCHEMA, "status": "failed", "stage": STAGE, **failure_fields(error),
                  **dict.fromkeys(NONCLAIMS, False)}
        if LAST_BOOTSTRAP_STAGE in BOOTSTRAP_STAGES:
            result["reported_bootstrap_stage"] = LAST_BOOTSTRAP_STAGE
    raw = canonical(result)
    need(len(raw) <= MAXIMUM_WORKER_BYTES)
    sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()
    raise SystemExit(0 if result["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
