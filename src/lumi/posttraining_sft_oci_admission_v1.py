"""Pre-ML, one-shot handoff of the exact accepted CPU image identity.

The trusted Linux host and its local Docker daemon are trust premises. The
caller must retain exclusive control of its container throughout the handoff.
Inspect only a never-started container, by full ID, and the independently pinned image.
No pull/create/start/export/mount/training API is provided here. The continuation
receives software metadata, NOT execution permission or a cgroup capability.
It must separately establish the real guardian/rootfs/worker binding. In
particular, putting a Docker CLI process in a cgroup does not contain its daemon.

This is not the flat release-manifest verifier. No export hash, publisher
signature, current rootfs contents, execution immutability or hard supervision
is inferred from an inspect response. Never log suppressed exception contexts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import select
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass

IMAGE_ID = "sha256:321d50ae5207dee05cd58ec5436d97978becde2ee8a767f2e8768229e6a9ad43"
SOURCE_COMMIT = "c4272e36bc98354067dd23a83931a3db58823d76"
LOCK_SHA256 = "26bde6a0163e5db9d8d4564117546482601f4d0beb09f8705be5fc9183594965"
BASE_SHA256 = "e30ec2e16ab8360be394b890e9f06e8819ae9f12c2414b975148beb76d93f8e8"
SOURCE_SHA256 = "bf97197a3015753e5565a4f1105be748b7a2b0eb5baff3264ce24f6d79646d38"
ACCEPTANCE_SHA256 = "24712f08f2522a0319da1179178aa8c73eef5969303faea73de16d89a23b386b"
PYTHON = "/usr/local/bin/python3.12"
WRAPPER = "/opt/lumi-release/source/scripts/run_sft_cpu_lookup_acceptance_v1.py"
TMPFS = {"/tmp": "rw,noexec,nosuid,nodev,size=536870912,mode=1777"}
MAXIMUM_INSPECT_BYTES = 65_536
MAXIMUM_INSPECT_SECONDS = 5
MAXIMUM_HANDOFF_SECONDS = 15
SCHEMA = "lumi.sft-accepted-oci-handoff.v0.1"
ERROR_CODES = frozenset({
    "configuration_rejected", "runtime_rejected", "owner_rejected", "parent_lost",
    "deadline_exhausted", "already_consumed", "inspect_failed", "inspect_budget_exhausted",
    "inspect_shape_rejected", "image_rejected", "container_rejected", "profile_rejected",
    "handoff_failed", "handoff_return_rejected", "internal_failed",
})
NONCLAIMS = (
    "guardian_containment_verified", "worker_identity_bound", "rootfs_contents_verified",
    "real_sft_execution_permitted", "production_eligible",
)


class OciAdmissionError(ValueError):
    def __init__(self, code="internal_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "internal_failed"
        super().__init__(self.code)


def need(value, code):
    if not value:
        raise OciAdmissionError(code) from None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


@dataclass(frozen=True, slots=True)
class SoftwareBinding:
    """An immutable metadata value for a trusted continuation, never a permit."""

    image_id: str = IMAGE_ID
    source_commit: str = SOURCE_COMMIT
    software_lock_sha256: str = LOCK_SHA256
    base_inventory_sha256: str = BASE_SHA256
    source_inventory_sha256: str = SOURCE_SHA256
    acceptance_report_sha256: str = ACCEPTANCE_SHA256

    def source_identity_sha256(self):
        # Domain-separated identity for a future worker/controller handoff.
        # Not an assertion that the existing synthetic worker uses this value.
        return hashlib.sha256(canonical({
            "schema": "lumi.sft-accepted-oci-software-identity.v0.1",
            **{name: getattr(self, name) for name in self.__slots__},
        })).hexdigest()


def _runtime():
    need(sys.platform == "linux" and os.uname().machine == "x86_64"
         and sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12)
         and sys.flags.no_site == 1 and sys.dont_write_bytecode
         and not any(name.split(".", 1)[0] in
                     {"torch", "transformers", "numpy", "cuda", "cupy", "jax", "tensorflow"}
                     for name in tuple(sys.modules)), "runtime_rejected")


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    need(remaining > 0, "deadline_exhausted")
    return remaining


def _inspect(kind, identifier, deadline):
    """Two possible read-only commands, local socket, empty private CLI config.

    Bound stdout while reading; never accumulate an unbounded capture_output.
    Stderr is discarded. On failure, kill/reap only our own inspect CLI process.
    An independent guardian must still enforce a hard deadline on its caller.
    """
    need((kind == "image" and identifier == IMAGE_ID)
         or (kind == "container" and type(identifier) is str
             and re.fullmatch(r"[0-9a-f]{64}", identifier)), "configuration_rejected")
    stop = min(deadline, time.monotonic() + MAXIMUM_INSPECT_SECONDS)
    _remaining(stop)
    with tempfile.TemporaryDirectory(prefix="lumi-oci-inspect-") as config:
        command = ["/usr/bin/docker", "--host", "unix:///var/run/docker.sock",
                   "--config", config, kind, "inspect", identifier]
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, close_fds=True,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
        try:
            raw = bytearray()
            while True:
                readable, _, _ = select.select([process.stdout], [], [], _remaining(stop))
                need(bool(readable), "deadline_exhausted")
                part = os.read(process.stdout.fileno(), min(8192, MAXIMUM_INSPECT_BYTES + 1
                                                            - len(raw)))
                if not part:
                    break
                raw.extend(part)
                need(len(raw) <= MAXIMUM_INSPECT_BYTES, "inspect_budget_exhausted")
            need(process.wait(timeout=_remaining(stop)) == 0, "inspect_failed")
            return bytes(raw)
        finally:
            try:
                if process.poll() is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass  # Already exited: reap our own child, never resend a signal.
                    process.wait(timeout=1)
            finally:
                process.stdout.close()


def _json(raw):
    need(type(raw) is bytes and 0 < len(raw) <= MAXIMUM_INSPECT_BYTES,
         "inspect_budget_exhausted")

    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "inspect_shape_rejected")
            result[key] = value
        return result

    def number(_text):
        raise OciAdmissionError("inspect_shape_rejected")

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=number)
    need(type(value) is list and len(value) == 1 and type(value[0]) is dict,
         "inspect_shape_rejected")
    return value[0]


def _configuration(image, container):
    original, selected = image["Config"], container["Config"]
    need(type(original) is dict and type(selected) is dict, "profile_rejected")
    for key in ("Env", "Entrypoint", "Cmd", "User", "WorkingDir", "Labels", "Volumes",
                "Healthcheck", "StopSignal", "Shell", "ExposedPorts", "OnBuild"):
        need(canonical(original.get(key)) == canonical(selected.get(key)), "profile_rejected")
    need(original["User"] == "10001:10001" and not original.get("Volumes")
         and original["WorkingDir"] == "/opt/lumi-release/source"
         and original["Entrypoint"] == [PYTHON, "-I", "-B"]
         and original["Cmd"] == [WRAPPER, "--synthetic-only"], "profile_rejected")
    env = original["Env"]
    need(type(env) is list and len(env) <= 128
         and all(type(item) is str and "=" in item and len(item) <= 4096 for item in env),
         "profile_rejected")
    values = dict(item.split("=", 1) for item in env)
    need(len(values) == len(env), "profile_rejected")
    required = {
        "LUMI_RELEASE_LOCK_SHA256": LOCK_SHA256,
        "LUMI_RELEASE_BASE_INVENTORY_SHA256": BASE_SHA256,
        "LUMI_RELEASE_SOURCE_INVENTORY_SHA256": SOURCE_SHA256,
        "LUMI_RELEASE_SOURCE_COMMIT": SOURCE_COMMIT,
        "LUMI_WORKER_RUN_TORCH_CI_FIXTURE": "1", "CUDA_VISIBLE_DEVICES": "",
        "GLOO_SOCKET_IFNAME": "lo", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    }
    need(all(values.get(key) == value for key, value in required.items()), "profile_rejected")


def _container(value, identifier):
    need(value["Id"] == identifier and value["Image"] == IMAGE_ID
         and value["Config"]["Image"] == IMAGE_ID, "container_rejected")
    need(value["Path"] == PYTHON and value["Args"] == ["-I", "-B", WRAPPER, "--synthetic-only"],
         "container_rejected")
    state = value["State"]
    need(state["Status"] == "created" and type(value["RestartCount"]) is int
         and value["RestartCount"] == 0, "container_rejected")
    for key in ("Running", "Paused", "Restarting", "OOMKilled", "Dead"):
        need(state[key] is False, "container_rejected")
    for key in ("Pid", "ExitCode"):
        need(type(state[key]) is int and state[key] == 0, "container_rejected")
    for key in ("StartedAt", "FinishedAt"):
        need(state[key] == "0001-01-01T00:00:00Z", "container_rejected")
    host = value["HostConfig"]
    need(host["ReadonlyRootfs"] is True and host["Privileged"] is False
         and host["AutoRemove"] is False and host["NetworkMode"] == "none"
         and host["CapDrop"] == ["ALL"]
         and host["SecurityOpt"] in (["no-new-privileges"], ["no-new-privileges:true"]),
         "profile_rejected")
    for key in ("Binds", "Mounts", "Devices", "DeviceRequests", "DeviceCgroupRules",
                "CapAdd", "Links", "PortBindings"):
        need(not host.get(key), "profile_rejected")
    for key in ("PidMode", "UTSMode", "UsernsMode", "CgroupParent"):
        need(host.get(key, "") == "", "profile_rejected")
    need(host["IpcMode"] == "private" and host["Tmpfs"] == TMPFS
         and canonical(host["RestartPolicy"]) == canonical({"Name": "no", "MaximumRetryCount": 0}),
         "profile_rejected")
    for key, expected in {"Memory": 4294967296, "MemorySwap": 4294967296,
                          "ShmSize": 67108864, "PidsLimit": 256, "NanoCpus": 2000000000}.items():
        need(type(host[key]) is int and host[key] == expected, "profile_rejected")
    need(canonical(host["Ulimits"]) in (
        canonical([{"Name": "core", "Soft": 0, "Hard": 0},
                   {"Name": "nofile", "Soft": 1024, "Hard": 1024}]),
        canonical([{"Name": "nofile", "Soft": 1024, "Hard": 1024},
                   {"Name": "core", "Soft": 0, "Hard": 0}]),
    ), "profile_rejected")
    mounts = value["Mounts"]
    need(type(mounts) is list and len(mounts) <= 1, "profile_rejected")
    for mount in mounts:
        need(mount["Type"] == "tmpfs" and mount["Destination"] == "/tmp"
             and not mount.get("Source"), "profile_rejected")


def verify_created_oci(container_id, *, deadline):
    """Read current daemon metadata, never accept a caller-selected image/receipt."""
    failure, result = None, None
    try:
        _runtime()
        need(type(container_id) is str and re.fullmatch(r"[0-9a-f]{64}", container_id),
             "configuration_rejected")
        need(type(deadline) in (int, float) and 0 < deadline - time.monotonic()
             <= MAXIMUM_HANDOFF_SECONDS, "configuration_rejected")
        image = _json(_inspect("image", IMAGE_ID, deadline))
        need(image["Id"] == IMAGE_ID and image["Os"] == "linux"
             and image["Architecture"] == "amd64", "image_rejected")
        # Image IDs are immutable under the trusted daemon. Observe mutable
        # container state last; exclusivity is still a caller responsibility.
        container = _json(_inspect("container", container_id, deadline))
        _container(container, container_id)
        _configuration(image, container)
        _remaining(deadline)
        result = SoftwareBinding()
    except BaseException as error:
        failure = error.code if type(error) is OciAdmissionError else "inspect_failed"
    if failure is not None:
        raise OciAdmissionError(failure) from None
    return result


class VerifiedOciHandoff:
    """One attempt, one trusted metadata continuation, process/parent/deadline bound.

    Parent checks bracket verification; a separate pidfd guardian is still needed
    for hard lifetime enforcement. Status deliberately grants no execution claim.
    The public call has no supplied receipt, success flag or retry option.
    """

    def __init__(self, *, container_id, expected_parent_pid, deadline, continuation):
        need(not hasattr(self, "_owner"), "configuration_rejected")
        need(type(container_id) is str and re.fullmatch(r"[0-9a-f]{64}", container_id)
             and type(expected_parent_pid) is int and expected_parent_pid > 1
             and type(deadline) in (int, float) and 0 < deadline - time.monotonic()
             <= MAXIMUM_HANDOFF_SECONDS and callable(continuation), "configuration_rejected")
        self._id, self._parent, self._deadline = container_id, expected_parent_pid, deadline
        self._continuation, self._owner = continuation, os.getpid()
        self._consumed = False
        self._state, self._code = "ready", "not_started"
        self._attempts = self._verified = self._calls = self._completed = 0

    def _guard(self):
        _runtime()
        need(os.getpid() == self._owner and threading.current_thread() is threading.main_thread()
             and threading.active_count() == 1, "owner_rejected")
        need(os.getppid() == self._parent, "parent_lost")
        _remaining(self._deadline)

    def __call__(self):
        if self._consumed:
            self._state, self._code = "failed", "already_consumed"
            raise OciAdmissionError("already_consumed") from None
        self._consumed = True
        failure, stage = None, "verify"
        try:
            self._state = "verifying"
            self._guard()
            self._attempts += 1
            binding = verify_created_oci(self._id, deadline=self._deadline)
            self._guard()
            need(self._state == "verifying", "already_consumed")
            need(type(binding) is SoftwareBinding and binding == SoftwareBinding(),
                 "image_rejected")
            self._verified += 1
            self._state, stage = "handing_off", "handoff"
            self._calls += 1
            result = self._continuation(binding)
            need(self._state == "handing_off", "already_consumed")
            need(result is None, "handoff_return_rejected")
            self._guard()
            self._completed += 1
            self._state, self._code = "handed_off", "completed"
        except BaseException as error:
            failure = (error.code if type(error) is OciAdmissionError
                       else "handoff_failed" if stage == "handoff" else "inspect_failed")
        if failure is not None:
            self._state, self._code = "failed", failure
            raise OciAdmissionError(failure) from None

    def status(self):
        return {
            "schema": SCHEMA, "state": self._state, "code": self._code,
            "verification_attempts": self._attempts, "verification_completed": self._verified,
            "handoff_attempts": self._calls, "handoff_completed": self._completed,
            **dict.fromkeys(NONCLAIMS, False),
        }
