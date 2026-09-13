#!/usr/bin/env python3
"""Fresh accepted-software qualification for three fixed tiny importer cases.

NEVER execute this native harness on the Mac. The independently pinned outer
supervisor signals only this owned PID, not its guardian process group. Trusted
Linux host, Docker daemon and stable reviewed checkout remain explicit premises.
The accepted image is loaded/created/inspected/exported, never started or rebuilt.
No real checkpoint/data argument exists. Every started case is one-shot.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import re
import select
import signal
import socket
import stat
import sys
import tempfile
import threading
import time
import types
import uuid
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).absolute().parents[1]
SCHEMA = "lumi.checkpoint-guard-native-qualification.v0.1"
SOURCE_ADMISSION = "configs/lumi-4b-sft-import-qualification-sources-v0.1.json"
SOURCE_ADMISSION_SHA256 = "f761ef5179b2e226347959f13ba7f4f385e70cd977e89c50a8a2498dff5a819f"
SOURCE_SCHEMA = "lumi.checkpoint-qualification-source-admission.v0.1"
LEGACY = "scripts/run_sft_oci_guardian_fixture_v2.py"
LEGACY_SHA256 = "79dcbfec678f43a39b31bb2dd2b4c9535fc162447cc591a781b72f93f079ce58"
FROZEN = "scripts/run_sft_cgroup_fixture_v1.py"
FROZEN_SHA256 = "f91ed0c353a20c309ff3e1bfaab8dbe990b28c36626fdeeb0ac5131e0cf75fad"
IMPORTER_MANIFEST = "configs/lumi-4b-sft-import-guard-source-manifest-v0.1.json"
ENTRY = "scripts/run_checkpoint_guarded_import_v1.py"
MAXIMUM_SECONDS = 900
MAXIMUM_EXPORT_BYTES = 2 * 1024**3
MAXIMUM_EXTRACT_BYTES = 4 * 1024**3
MAXIMUM_REPORT_BYTES = 32768
MAXIMUM_SOURCE_BYTES = 8 * 1024**2
ARCHIVE_BYTES = 1_062_113_280
ARCHIVE_SHA256 = "99d21fc60a54a65f28ceb4780f603f7ee48e49ac7b6d00a053dcdabe9ebc9ff4"
CASES = ("wrong_entry", "wrong_manifest", "positive")
STAGES = frozenset({"preflight", "source_admission", "copy_image", "load_image",
                    "create_container", "inspect_container", "export_rootfs", "extract_rootfs",
                    "install_importer", "cases", "cleanup", "complete"})
ERROR_CODES = frozenset({"runtime_rejected", "source_rejected", "fixture_failed",
                         "deadline_exhausted", "cleanup_unconfirmed", "case_rejected"})
NONCLAIMS = ("real_4b_import", "training_authorized", "production_eligible", "gpu_used",
             "real_model_or_data_access", "guardian_death_containment", "hostile_root_containment",
             "full_dynamic_dependency_closure_verified", "host_admission_verified",
             "publisher_signature_verified", "registry_published")
HELPER = BASE = ADMISSION = ROOTFS = ADDON = QUALIFICATION = None
STAGE = "preflight"
DESCENDANTS_REAPED = True
_ATTEMPTED = False


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


def identity(info):
    return info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode), info.st_uid, info.st_gid


def _state(info):
    return (*identity(info), info.st_mode, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def remaining(deadline):
    value = deadline - time.monotonic()
    need(value > 0, "deadline_exhausted")
    return value


def runtime():
    need(sys.platform == "linux" and os.uname().machine == "x86_64"
         and sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12)
         and sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
         and os.getresuid() == os.getresgid() == (0, 0, 0)
         and signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL
         and os.environ.get("LUMI_CHECKPOINT_GUARD_QUALIFICATION_CI") == "1"
         and threading.current_thread() is threading.main_thread()
         and threading.active_count() == 1
         and not any(name.split(".", 1)[0] in {
             "lumi", "torch", "numpy", "transformers", "safetensors", "tokenizers",
             "tensorflow", "jax", "cuda", "cupy",
         } for name in sys.modules), "runtime_rejected")


def source_bytes(relative, expected=None):
    need(type(relative) is str and len(relative) <= 256
         and re.fullmatch(r"(?:src/lumi|scripts|configs)/[A-Za-z0-9_.-]+", relative)
         and str(PurePosixPath(relative)) == relative, "source_rejected")
    path = ROOT / relative
    for ancestor in (path, *path.parents):
        need(not ancestor.is_symlink(), "source_rejected")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
             and not before.st_mode & 0o022 and 0 < before.st_size <= 262144,
             "source_rejected")
        parts, count = [], 0
        while count < before.st_size:
            size = min(65536, before.st_size - count)
            body = os.read(fd, size)
            need(len(body) == size, "source_rejected")
            parts.append(body)
            count += len(body)
        raw = b"".join(parts)
        need(_state(before) == _state(os.fstat(fd)) == _state(path.lstat()), "source_rejected")
        need(expected is None or hashlib.sha256(raw).hexdigest() == expected, "source_rejected")
        return raw
    finally:
        os.close(fd)


def parse_admission(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "source_rejected")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=pairs)
        need(type(value) is dict and set(value) == {"schema", "files"}
             and value["schema"] == SOURCE_SCHEMA and type(value["files"]) is list
             and 1 <= len(value["files"]) <= 128 and canonical(value) == raw, "source_rejected")
        rows, total = {}, 0
        for row in value["files"]:
            need(type(row) is dict and set(row) == {"path", "bytes", "sha256"}
                 and type(row["path"]) is str
                 and re.fullmatch(r"(?:src/lumi|scripts|configs)/[A-Za-z0-9_.-]+", row["path"])
                 and row["path"] not in rows and type(row["bytes"]) is int
                 and 0 < row["bytes"] <= 262144 and type(row["sha256"]) is str
                 and re.fullmatch(r"[a-f0-9]{64}", row["sha256"]), "source_rejected")
            rows[row["path"]] = row
            total += row["bytes"]
        need(total <= MAXIMUM_SOURCE_BYTES and {LEGACY, FROZEN, ENTRY, IMPORTER_MANIFEST,
             "src/lumi/posttraining_checkpoint_guard_sources_v1.py",
             "src/lumi/posttraining_checkpoint_guard_qualification_v1.py"} <= set(rows),
             "source_rejected")
        return rows
    except BaseException:
        raise FixtureError("source_rejected") from None


class VerifiedHostSources:
    """Compile authenticated bodies for only our private host package namespaces.

    -B disables bytecode writes, not reads. This scoped finder therefore never
    consults FileFinder, a .pyc cache, extension module or unverified source for
    these modules. Unrelated stdlib/runtime packages keep their normal loaders.
    The one-shot native process retains the finder through its child lifetimes.
    """

    def __init__(self, package, bodies):
        self.package = package
        self.bodies = dict(bodies)

    def find_spec(self, fullname, path=None, target=None):
        parent, _, name = fullname.rpartition(".")
        if not (parent == self.package or parent == "_lumi_generated_cgroup_fixture_v1"
                or re.fullmatch(r"_lumi_checkpoint_guard_base_[a-f0-9]{32}_components", parent)):
            return None
        need(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is not None, "source_rejected")
        relative = "src/lumi/" + name + ".py"
        need(relative in self.bodies, "source_rejected")
        return importlib.util.spec_from_loader(fullname, self, origin=str(ROOT / relative))

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        relative = "src/lumi/" + module.__name__.rsplit(".", 1)[1] + ".py"
        filename = str(ROOT / relative)
        module.__file__, module.__cached__ = filename, None
        exec(compile(self.bodies[relative], filename, "exec"), module.__dict__)


def load_sources():
    """Authenticate all listed source bytes before importing host components."""
    need(re.fullmatch(r"[a-f0-9]{64}", SOURCE_ADMISSION_SHA256) is not None, "source_rejected")
    rows = parse_admission(source_bytes(SOURCE_ADMISSION, SOURCE_ADMISSION_SHA256))
    bodies = {name: source_bytes(name, row["sha256"]) for name, row in rows.items()}
    need(all(len(bodies[name]) == row["bytes"] for name, row in rows.items()), "source_rejected")
    need(hashlib.sha256(bodies[LEGACY]).hexdigest() == LEGACY_SHA256
         and hashlib.sha256(bodies[FROZEN]).hexdigest() == FROZEN_SHA256, "source_rejected")
    package = types.ModuleType("_lumi_import_qualification_" + uuid.uuid4().hex)
    package.__path__ = []
    sys.modules[package.__name__] = package
    # Scoped verified source loader also covers the frozen base's private
    # dynamically named helper packages. Never patch the frozen implementations.
    sys.meta_path.insert(0, VerifiedHostSources(package.__name__, bodies))
    addon = importlib.import_module(package.__name__ + ".posttraining_checkpoint_guard_sources_v1")
    qualification = importlib.import_module(
        package.__name__ + ".posttraining_checkpoint_guard_qualification_v1")
    helper = types.ModuleType(package.__name__ + "_oci")
    helper.__file__ = str(ROOT / LEGACY)
    exec(compile(bodies[LEGACY], str(ROOT / LEGACY), "exec"), helper.__dict__)
    manifest, entry = bodies[IMPORTER_MANIFEST], bodies[ENTRY]
    declared = json.loads(manifest)["files"]
    files = {row["path"]: bodies["src/" + row["path"]] for row in declared}
    addon.validate_bundle(files, manifest, entry,
                          expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
                          expected_entry_sha256=hashlib.sha256(entry).hexdigest())
    return helper, addon, qualification, files, manifest, entry


def command(*args, **kwargs):
    return HELPER.command(*args, **kwargs)


def copy_image(*args):
    return HELPER.copy_image(*args)


def create_options(*args):
    return HELPER.create_options(*args)


def image_inventory(*args):
    return HELPER.image_inventory(*args)


def container_inventory(*args):
    return HELPER.container_inventory(*args)


def validate_case(value, case, environment):
    q = QUALIFICATION
    fixed = {"schema": q.SCHEMA, "status": "passed", "synthetic_only": True, "case": case,
             "expects_negative": case != "positive", "worker_exit_code": int(case != "positive"),
             "owned_cgroups_empty_and_removed": 1, "kill_attempts": 1,
             "controller_and_deadline_watch_installed_before_release": True,
             "worker_exit_status_verified": True, "memory_max_bytes": 4294967296,
             "swap_max_bytes": 0, "pids_max": 256, "cpu_quota_us": 200000,
             "cpu_period_us": 100000, "maximum_seconds": 90,
             "tmpfs_capacity_bytes": 536870912, "shm_capacity_bytes": 67108864,
             **dict.fromkeys(q.frozen.NONCLAIMS, False)}
    need(type(value) is dict and set(value) == set(fixed) | {"worker", "workload_processes_reaped"}
         and all(type(value[key]) is type(expected) and value[key] == expected
                 for key, expected in fixed.items())
         and type(value["workload_processes_reaped"]) is int
         and 1 <= value["workload_processes_reaped"] <= 4, "case_rejected")
    q._worker(value["worker"], case, q._environment(case, environment), value["worker_exit_code"])
    return dict(value)


def one_case(case, root_path, mountpoint, environment, deadline):
    """One root child per case; only owned PID signalling and bounded reaping."""
    global DESCENDANTS_REAPED
    remaining(deadline)
    case_deadline = min(int(deadline * 1_000_000_000), time.monotonic_ns() + 90_000_000_000)
    cleanup_deadline = min(int((deadline + 15) * 1_000_000_000), case_deadline + 15_000_000_000)
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    pidfd = None
    reaped = False
    DESCENDANTS_REAPED = False
    try:
        pid = os.fork()
        if pid == 0:
            try:
                BASE.close_except({child.fileno()})
                result = QUALIFICATION.run_prepared_case(
                    case, str(root_path), str(mountpoint), identity(root_path.stat()), environment,
                    str(ROOT / FROZEN), FROZEN_SHA256, case_deadline)
                BASE.send(child, result)
                child.close()
                os._exit(0)
            except BaseException:
                os._exit(2)  # No untrusted exception or partially validated frame.
        child.close()
        child = None
        pidfd = os.pidfd_open(pid, 0)
        while not select.select([parent], [], [], min(
                30, max(0, (cleanup_deadline - time.monotonic_ns()) / 1_000_000_000)))[0]:
            need(time.monotonic_ns() < cleanup_deadline, "deadline_exhausted")
        result = QUALIFICATION.frozen._receive(parent, cleanup_deadline)
        need(QUALIFICATION.frozen._packet(parent, cleanup_deadline) == b"", "case_rejected")
        while time.monotonic_ns() < cleanup_deadline:
            selected, status = os.waitpid(pid, os.WNOHANG)
            if selected:
                reaped = True
                need(selected == pid and status == 0, "case_rejected")
                break
            time.sleep(0.02)
        need(reaped, "cleanup_unconfirmed")
        return validate_case(result, case, environment)
    finally:
        try:
            if pidfd is not None and not reaped:
                if not select.select([pidfd], [], [], 0)[0]:
                    signal.pidfd_send_signal(pidfd, signal.SIGTERM, None, 0)
            retired = BASE.wait_statuses(cleanup_deadline)
            need(not reaped or not retired, "cleanup_unconfirmed")
            DESCENDANTS_REAPED = True
        finally:
            parent.close()
            if child is not None:
                child.close()
            if pidfd is not None:
                os.close(pidfd)


def cases(root_path, mountpoint, manifest, entry, deadline):
    BASE.subreaper()
    environment = {**QUALIFICATION.frozen.child_bootstrap.FIXED_ENV,
                   "LUMI_IMPORT_ENTRY_SHA256": hashlib.sha256(entry).hexdigest(),
                   "LUMI_IMPORT_SOURCE_MANIFEST_SHA256": hashlib.sha256(manifest).hexdigest()}
    return [one_case(case, root_path, mountpoint, environment, deadline) for case in CASES]


def main():
    try:
        need(len(sys.argv) == 3 and sys.argv[1] == "--synthetic-only")
        result = run(Path(sys.argv[2]))
    except BaseException as error:
        code = error.code if type(error) is FixtureError else "fixture_failed"
        result = {"schema": SCHEMA, "status": "failed", "stage": STAGE, "code": code,
                  **dict.fromkeys(NONCLAIMS, False)}
    raw = canonical(result)
    need(len(raw) <= MAXIMUM_REPORT_BYTES)
    need(os.write(1, raw) == len(raw))
    return 0 if result["status"] == "passed" else 2


def run(archive_path):
    global BASE, STAGE, ADMISSION, ROOTFS, HELPER, ADDON, QUALIFICATION, _ATTEMPTED
    runtime()
    need(not _ATTEMPTED, "runtime_rejected")
    _ATTEMPTED = True
    os.environ.clear()
    os.environ.update({"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"})
    os.umask(0o077)
    overall_deadline = time.monotonic() + MAXIMUM_SECONDS
    deadline = overall_deadline - 100  # Reserve owned cleanup inside the overall budget.
    STAGE = "source_admission"
    HELPER, ADDON, QUALIFICATION, files, manifest, entry = load_sources()
    BASE = HELPER.load_base()
    admission = ADMISSION = HELPER.load_admission()
    rootfs = ROOTFS = ADDON.rootfs  # Exact class identity required by the frozen add-on.
    manifest_sha = hashlib.sha256(manifest).hexdigest()
    entry_sha = hashlib.sha256(entry).hexdigest()
    parent = Path(tempfile.mkdtemp(prefix="lumi-checkpoint-guard-qualification-", dir="/tmp"))
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
        STAGE = "install_importer"
        addon_summary = ADDON.install_importer_addon(
            owned, files, manifest, entry,
            expected_manifest_sha256=manifest_sha, expected_entry_sha256=entry_sha)
        root_path = owned.path  # Cache before the shorter extraction deadline expires.
        extracted = owned.summary()
        STAGE = "cases"
        reports = cases(root_path, mountpoint, manifest, entry, deadline - 30)
        result = {"schema": SCHEMA, "status": "passed", "stage": "complete", "code": "ok",
                  "cases": reports,
                  "accepted_image_id": binding.image_id,
                  "source_identity_sha256": binding.source_identity_sha256(),
                  "archive_bytes": ARCHIVE_BYTES, "export_bytes": export_bytes,
                  "export_sha256": export_sha, "extraction": extracted,
                  "entry_sha256": entry_sha, "importer_manifest_sha256": manifest_sha,
                  "source_admission_sha256": SOURCE_ADMISSION_SHA256,
                  "addon": addon_summary,
                  "docker_containers_started": 0, "images_built": 0,
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


if __name__ == "__main__":
    raise SystemExit(main())
