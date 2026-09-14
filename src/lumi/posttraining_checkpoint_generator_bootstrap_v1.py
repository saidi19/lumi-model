"""Fixed Linux-only attached bootstrap for one private tiny-generator worker.

Never invoke the native bootstrap on a Mac. Local tests replace every runtime,
descriptor and syscall operation. This module does not create/authorize a host,
cgroup, source tree, worker command or real import. The guardian supplies its
owned cgroup and independently admitted OCI root, then arms parent/deadline
cleanup before sending the single release packet. A ready packet is not by
itself kernel attestation or permission to release a worker.

Only the new child changes namespaces, privileges or descriptors. Every started
failure exits; no retry, rollback or arbitrary executable/argument is accepted.
No ML import occurs here. Public frames contain fixed fields/stages only, never
paths, environment values, process IDs or suppressed exception text.
"""

from __future__ import annotations

import errno
import json
import os
import platform
import re
import resource
import select
import socket
import stat
import sys
import threading
from pathlib import PurePosixPath

from . import posttraining_sft_cgroup_guardian_v1 as guardian
from . import posttraining_sft_cgroup_v1 as cgroup
from . import posttraining_sft_ipc_filter_v1 as ipc
from . import posttraining_sft_netns_v1 as netns
from . import posttraining_sft_oci_landlock_v1 as landlock
from . import posttraining_sft_oci_mount_v1 as mount

RUNTIME = "/usr/local/bin/python3.12"
ENTRY = "/opt/lumi-release/guardian/generator_entry.py"
ARGV = (RUNTIME, "-I", "-B", ENTRY, "--synthetic-only")
FIXED_ENV = {
    "PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "AWS_EC2_METADATA_DISABLED": "true",
    "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
HASH_ENV = frozenset({"LUMI_IMPORT_SOURCE_MANIFEST_SHA256", "LUMI_GENERATOR_ENTRY_SHA256",
                      "LUMI_GENERATION_NONCE_SHA256"})
READ_ONLY_PATHS = ("/usr", "/etc", "/opt/lumi-release", "/proc")
WRITABLE_PATHS = ("/tmp", "/dev/shm")
DEVICE_PATHS = ("/dev/null", "/dev/zero", "/dev/random", "/dev/urandom")
READY = {"phase": "ready", "isolation_installed": True,
         "uid_gid_dropped": True, "authority_fds_closed": True}
STAGES = frozenset({"sanitize", "attach", "network", "mount", "landlock", "ipc",
                    "limits", "drop", "ready", "release", "exec"})
ERROR_CODES = frozenset({"runtime_rejected", "arguments_rejected", "environment_rejected",
                         "already_attempted", "bootstrap_failed"})
ML_ROOTS = frozenset({"torch", "transformers", "numpy", "safetensors", "tokenizers",
                      "tensorflow", "jax", "cupy", "cuda"})
_ATTEMPTED = False


class GeneratorBootstrapError(ValueError):
    def __init__(self, code="bootstrap_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "bootstrap_failed"
        super().__init__(self.code)


def _need(value, code="bootstrap_failed"):
    if not value:
        raise GeneratorBootstrapError(code) from None


def validate_environment(environment):
    """Copy a closed environment; pins bind source bytes, not origin/approval."""
    _need(type(environment) is dict and set(environment) == set(FIXED_ENV) | HASH_ENV
          and all(type(key) is str and type(value) is str
                  for key, value in environment.items()), "environment_rejected")
    _need(all(environment[key] == value for key, value in FIXED_ENV.items())
          and all(re.fullmatch(r"[0-9a-f]{64}", environment[key]) is not None
                  and environment[key] != "0" * 64 for key in HASH_ENV), "environment_rejected")
    return dict(environment)


def _runtime():
    _need(sys.platform == "linux" and platform.machine() == "x86_64"
          and sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12)
          and sys.flags.no_site == 1 and sys.dont_write_bytecode
          and os.getresuid() == os.getresgid() == (0, 0, 0)
          and threading.current_thread() is threading.main_thread()
          and threading.active_count() == 1
          and not any(name.split(".", 1)[0] in ML_ROOTS for name in sys.modules),
          "runtime_rejected")


def _arguments(group_fd, channel, rootfs_path, mountpoint_path, root_identity):
    _need(type(group_fd) is int and 3 <= group_fd < 1024
          and type(channel) is socket.socket and channel.family == socket.AF_UNIX
          and channel.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) == socket.SOCK_SEQPACKET
          and channel.gettimeout() is None and 3 <= channel.fileno() < 1024
          and channel.fileno() != group_fd, "arguments_rejected")
    parts = []
    for path in (rootfs_path, mountpoint_path):
        _need(type(path) is str and 0 < len(path) <= 4096
              and all(32 <= ord(c) < 127 for c in path), "arguments_rejected")
        parsed = PurePosixPath(path)
        _need(parsed.is_absolute() and str(parsed) == path and ".." not in parsed.parts
              and 3 <= len(parsed.parts) <= 64 and not path.startswith("//"),
              "arguments_rejected")
        parts.append(parsed.parts)
    _need(parts[0][:-1] == parts[1][:-1] and parts[0][-1] != parts[1][-1],
          "arguments_rejected")
    _need(type(root_identity) is tuple and len(root_identity) == 5
          and all(type(value) is int for value in root_identity)
          and 0 <= root_identity[0] < 2**64 and 0 < root_identity[1] < 2**64
          and root_identity[2:] == (stat.S_IFDIR, 0, 0), "arguments_rejected")


def _descriptors():
    entries = os.listdir("/proc/self/fd")
    _need(len(entries) <= 1024)
    result = set()
    for item in entries:
        _need(type(item) is str and re.fullmatch(r"(?:0|[1-9][0-9]{0,6})", item) is not None)
        number = int(item)
        _need(number < 2**20 and number not in result)
        try:
            os.fstat(number)
        except OSError as error:
            _need(error.errno == errno.EBADF)  # Only the now-closed listdir transient FD.
        else:
            result.add(number)
    return result


def _close_except(kept):
    for descriptor in _descriptors():
        if descriptor > 2 and descriptor not in kept:
            os.close(descriptor)  # No retry after ambiguous close.


def _sanitize(group_fd, channel_fd):
    descriptor = os.open("/dev/null", os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        _need(stat.S_ISCHR(info.st_mode)
              and (os.major(info.st_rdev), os.minor(info.st_rdev)) == (1, 3))
        for number in (0, 1, 2):
            os.dup2(descriptor, number)
    finally:
        if descriptor > 2:
            os.close(descriptor)
    _close_except({group_fd, channel_fd})


def _send(channel, frame):
    raw = json.dumps(frame, sort_keys=True, separators=(",", ":")).encode("ascii")
    _need(0 < len(raw) <= 256 and channel.send(raw) == len(raw))


def _release(channel):
    _need(select.select([channel], [], [], 10)[0] == [channel])
    raw, ancillary, flags, _address = channel.recvmsg(2, 256)
    _need(type(raw) is bytes and raw == b"G" and not ancillary and flags == 0)
    _need(not select.select([channel], [], [], 0)[0])  # No queued extra packet or closed peer.


def bootstrap(group_fd, channel, *, rootfs_path, mountpoint_path, root_identity, environment):
    """Attach/isolate/drop/wait/exec one fixed child; native success never returns.

    Invalid runtime/arguments reject before mutations. Once started, failure
    emits at most one fixed stage frame then exits 2. Only the guardian decides
    whether an independently checked ready child may receive its release byte.
    """
    global _ATTEMPTED
    _runtime()
    copied_environment = validate_environment(environment)
    _arguments(group_fd, channel, rootfs_path, mountpoint_path, root_identity)
    _need(not _ATTEMPTED, "already_attempted")
    _ATTEMPTED = True
    stage = "sanitize"
    report_channel = channel
    try:
        _sanitize(group_fd, channel.fileno())
        stage = "attach"
        cgroup.enter_current_process(group_fd)
        stage = "network"
        network = netns.create_loopback_namespace()
        _need(network["namespace_changed"] is True and network["only_loopback_interface"] is True)
        stage = "mount"
        mounted = mount.enter_oci_rootfs(rootfs_path, mountpoint_path, uid=10001, gid=10001)
        _need(mounted["old_root_detached"] is True and mounted["readonly_root"] is True)
        stage = "landlock"
        policy = landlock.restrict_filesystem(
            read_only_paths=READ_ONLY_PATHS, writable_paths=WRITABLE_PATHS,
            device_paths=DEVICE_PATHS, expected_root_identity=root_identity,
        )
        _need(policy["filesystem_content_policy_enforced"] is True
              and type(policy["abi_version"]) is int and policy["abi_version"] >= 5)
        stage = "ipc"
        _need(ipc.install_ipc_filter() == dict(ipc.RECEIPT))
        stage = "limits"
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 1024))
        stage = "drop"
        guardian.drop_worker_privileges(10001, 10001)
        os.close(group_fd)
        _close_except({channel.fileno()})
        _need(os.getresuid() == os.getresgid() == (10001, 10001, 10001)
              and os.getgroups() == []
              and _descriptors() == {0, 1, 2, channel.fileno()})
        stage = "ready"
        _send(channel, READY)
        stage = "release"
        _release(channel)
        stage = "exec"
        os.chdir("/opt/lumi-release/source")
        os.dup2(channel.fileno(), 1)
        report_channel = None  # The original socket FD is revoked by the next close sweep.
        _close_except(set())
        _need(_descriptors() == {0, 1, 2})
        os.execve(RUNTIME, list(ARGV), copied_environment)
        raise GeneratorBootstrapError("bootstrap_failed")  # exec returning cannot be success.
    except BaseException:
        try:
            if report_channel is not None:
                _send(report_channel, {"phase": "bootstrap_failed", "stage": stage})
        except BaseException:
            pass
        os._exit(2)

