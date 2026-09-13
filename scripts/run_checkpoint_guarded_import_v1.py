#!/usr/bin/env python3
"""Fixed Linux guarded importer entry; never invoke main on the Mac.

The trusted guardian binds this root-owned entry and source manifest, provides
read-only software/mounts, contains the child, closes authority FDs and drops it
to uid/gid10001. Hashes below are not self-granted host or execution authority.
Only newly staged private code copies and generated tiny fixtures are writable.
Neither root-owned sources nor partial attempts are changed, erased or reused.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import stat
import sys
import tempfile
import threading
import time
from pathlib import Path

ENTRY = Path("/opt/lumi-release/guardian/import_entry.py")
SOURCE_ROOT = Path("/opt/lumi-release/importer/src")
MANIFEST = Path("/opt/lumi-release/importer/manifest.json")
SOURCE_SCHEMA = "lumi.checkpoint-import-source-manifest.v0.1"
REPORT_SCHEMA = "lumi.checkpoint-guard-fixture-result.v0.1"
MAXIMUM_SOURCE_FILES = 128
MAXIMUM_SOURCE_FILE_BYTES = 256 * 1024
MAXIMUM_SOURCE_BYTES = 4 * 1024**2
MAXIMUM_MANIFEST_BYTES = 65536
MAXIMUM_REPORT_BYTES = 4096
READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
DIR_FLAGS = READ_FLAGS | os.O_DIRECTORY
SOURCE_NAME = re.compile(r"lumi/[A-Za-z_][A-Za-z0-9_]*\.py\Z", re.ASCII)
ML_ROOTS = frozenset({"lumi", "torch", "numpy", "transformers", "safetensors", "tokenizers",
                      "tensorflow", "jax", "cupy", "cuda"})
FIXED_ENV = {
    "PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "AWS_EC2_METADATA_DISABLED": "true", "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
}
PIN_KEYS = frozenset({"LUMI_IMPORT_ENTRY_SHA256", "LUMI_IMPORT_SOURCE_MANIFEST_SHA256"})
ERROR_CODES = frozenset({"runtime_rejected", "entry_binding_rejected", "source_binding_rejected",
                         "source_staging_failed", "fixture_failed", "deadline_exhausted",
                         "output_failed"})


class EntryError(ValueError):
    def __init__(self, code="source_binding_rejected"):
        self.code = code if type(code) is str and code in ERROR_CODES else "source_binding_rejected"
        super().__init__(self.code)


def need(value, code="source_binding_rejected"):
    if not value:
        raise EntryError(code) from None


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pin(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def runtime_check(argv, environment):
    need(type(argv) is list and argv == [str(ENTRY), "--synthetic-only"], "runtime_rejected")
    need(sys.platform == "linux" and os.uname().machine == "x86_64"
         and sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12)
         and sys.flags.isolated == 1 and sys.flags.ignore_environment == 1
         and sys.flags.no_user_site == 1 and sys.dont_write_bytecode
         and not sys.flags.optimize and sys.byteorder == "little"
         and os.getresuid() == os.getresgid() == (10001, 10001, 10001)
         and os.getgroups() == [] and threading.current_thread() is threading.main_thread()
         and threading.active_count() == 1
         and not any(name.split(".", 1)[0] in ML_ROOTS for name in sys.modules),
         "runtime_rejected")
    need(type(environment) is dict and set(environment) == set(FIXED_ENV) | PIN_KEYS
         and all(environment[key] == value for key, value in FIXED_ENV.items())
         and all(pin(environment[key]) for key in PIN_KEYS), "runtime_rejected")


def snapshot(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_uid,
            info.st_gid, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def require_owned(info, *, directory=False, owner=0):
    need((stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
         and info.st_uid == owner and not stat.S_IMODE(info.st_mode) & 0o022
         and (directory or info.st_nlink == 1))


def open_directory(path):
    need(type(path) is type(Path()) and path.is_absolute() and ".." not in path.parts)
    fd = os.open("/", DIR_FLAGS)
    try:
        for component in path.parts[1:]:
            child = os.open(component, DIR_FLAGS, dir_fd=fd)
            previous, fd = fd, child
            os.close(previous)
        result, fd = fd, None
        return result
    finally:
        if fd is not None:
            os.close(fd)


def read_file(path, *, maximum, owner=0):
    parent = open_directory(path.parent)
    fd = None
    try:
        require_owned(os.fstat(parent), directory=True, owner=owner)
        fd = os.open(path.name, READ_FLAGS, dir_fd=parent)
        before = os.fstat(fd)
        require_owned(before, owner=owner)
        need(0 <= before.st_size <= maximum)
        chunks = []
        remaining = before.st_size
        while remaining:
            raw = os.read(fd, min(65536, remaining))
            need(type(raw) is bytes and 0 < len(raw) <= remaining)
            chunks.append(raw)
            remaining -= len(raw)
        need(snapshot(os.fstat(fd)) == snapshot(before)
             and snapshot(os.stat(path.name, dir_fd=parent, follow_symlinks=False))
             == snapshot(before))
        return b"".join(chunks)
    finally:
        try:
            if fd is not None:
                os.close(fd)
        finally:
            os.close(parent)


def _json(raw):
    def pairs(items):
        value = {}
        for key, child in items:
            need(key not in value)
            value[key] = child
        return value

    def constant(_value):
        raise EntryError()

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def validate_source_manifest(raw, expected_manifest_sha256):
    """Pure pinned manifest validation; return a fresh name-to-row mapping, no I/O."""
    need(pin(expected_manifest_sha256) and type(raw) is bytes
         and 0 < len(raw) <= MAXIMUM_MANIFEST_BYTES
         and digest(raw) == expected_manifest_sha256)
    value = _json(raw)
    need(type(value) is dict and set(value) == {"schema", "files"}
         and value["schema"] == SOURCE_SCHEMA and type(value["files"]) is list
         and 2 <= len(value["files"]) <= MAXIMUM_SOURCE_FILES)
    rows, total = {}, 0
    for row in value["files"]:
        need(type(row) is dict and set(row) == {"path", "bytes", "sha256"}
             and type(row["path"]) is str and SOURCE_NAME.fullmatch(row["path"])
             and row["path"] not in rows and type(row["bytes"]) is int
             and 0 <= row["bytes"] <= MAXIMUM_SOURCE_FILE_BYTES and pin(row["sha256"]))
        total += row["bytes"]
        need(total <= MAXIMUM_SOURCE_BYTES)
        rows[row["path"]] = row
    need({"lumi/__init__.py", "lumi/posttraining_checkpoint_guard_fixture_v1.py"} <= set(rows))
    return rows


def verify_source_tree(source_root, manifest_path, expected_manifest_sha256):
    """Authenticate the exact root-owned flat module tree before Lumi/ML import."""
    need(pin(expected_manifest_sha256))
    raw = read_file(manifest_path, maximum=MAXIMUM_MANIFEST_BYTES)
    rows = validate_source_manifest(raw, expected_manifest_sha256)
    total = sum(row["bytes"] for row in rows.values())
    fd = open_directory(source_root)
    package = None
    try:
        require_owned(os.fstat(fd), directory=True)
        root_identity = snapshot(os.fstat(fd))
        with os.scandir(fd) as entries:
            first = next(entries, None)
            need(first is not None and first.name == "lumi" and next(entries, None) is None)
        package = os.open("lumi", DIR_FLAGS, dir_fd=fd)
        require_owned(os.fstat(package), directory=True)
        package_identity = snapshot(os.fstat(package))
        seen = set()
        with os.scandir(package) as entries:
            for entry in entries:
                need(len(seen) < MAXIMUM_SOURCE_FILES)
                name = "lumi/" + entry.name
                need(name in rows)
                info = entry.stat(follow_symlinks=False)
                require_owned(info)
                need(info.st_size == rows[name]["bytes"])
                seen.add(name)
        need(seen == set(rows))
        files = {}
        for name, row in sorted(rows.items()):
            body = read_file(source_root / name, maximum=MAXIMUM_SOURCE_FILE_BYTES)
            need(len(body) == row["bytes"] and digest(body) == row["sha256"])
            files[name] = body
        need(snapshot(os.fstat(fd)) == root_identity
             and snapshot(os.fstat(package)) == package_identity)
        return files, {"source_manifest_sha256": expected_manifest_sha256,
                       "source_files_verified": len(files), "source_bytes_verified": total,
                       "source_verification_read_bytes": len(raw) + total}
    finally:
        try:
            if package is not None:
                os.close(package)
        finally:
            os.close(fd)


def stage_private_sources(parent, files):
    """New uid-owned read-only copies; never chmod/chown the root-owned originals."""
    need(type(parent) is type(Path()) and parent.is_absolute() and ".." not in parent.parts,
         "source_staging_failed")
    need(type(files) is dict and 2 <= len(files) <= MAXIMUM_SOURCE_FILES
         and all(type(name) is str and SOURCE_NAME.fullmatch(name)
                 and type(raw) is bytes and len(raw) <= MAXIMUM_SOURCE_FILE_BYTES
                 for name, raw in files.items())
         and sum(map(len, files.values())) <= MAXIMUM_SOURCE_BYTES,
         "source_staging_failed")
    stage = Path(tempfile.mkdtemp(prefix="lumi-import-code-", dir=parent))
    stage.chmod(0o700)
    source = stage / "src"
    source.mkdir(mode=0o700)
    (source / "lumi").mkdir(mode=0o700)
    writes = reads = 0
    for name, raw in sorted(files.items()):
        path = source / name
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                     0o600)
        try:
            os.fchmod(fd, 0o600)
            for start in range(0, len(raw), 65536):
                part = raw[start:start + 65536]
                writes += len(part)  # Reserve before each write; never retry or refund.
                need(writes <= MAXIMUM_SOURCE_BYTES, "source_staging_failed")
                need(os.write(fd, part) == len(part), "source_staging_failed")
            os.fsync(fd)
            os.fchmod(fd, 0o400)
            info = os.fstat(fd)
            require_owned(info, owner=os.geteuid())
            need(stat.S_IMODE(info.st_mode) == 0o400 and info.st_size == len(raw),
                 "source_staging_failed")
        finally:
            os.close(fd)
        observed = read_file(path, maximum=MAXIMUM_SOURCE_FILE_BYTES, owner=os.geteuid())
        reads += len(observed)
        need(observed == raw, "source_staging_failed")
    return source, {"private_source_copy_bytes_reserved": writes,
                    "private_source_copy_readback_bytes": reads,
                    "private_source_copy_files": len(files),
                    "private_source_copies_rootfs_immutable": False,
                    "hostile_owner_code_immutability_verified": False}


def main():
    report = {"schema": REPORT_SCHEMA, "status": "failed", "code": "runtime_rejected",
              "synthetic_only": True, "real_4b_import": False,
              "training_authorized": False, "production_eligible": False}
    success = False
    saved_stdout = None
    try:
        environment = dict(os.environ)
        runtime_check(sys.argv, environment)
        deadline = time.monotonic() + 45

        def budget():
            need(time.monotonic() < deadline, "deadline_exhausted")

        entry = read_file(ENTRY, maximum=MAXIMUM_SOURCE_FILE_BYTES)
        need(digest(entry) == environment["LUMI_IMPORT_ENTRY_SHA256"], "entry_binding_rejected")
        files, source_report = verify_source_tree(
            SOURCE_ROOT, MANIFEST, environment["LUMI_IMPORT_SOURCE_MANIFEST_SHA256"])
        budget()
        source, copy_report = stage_private_sources(Path("/tmp"), files)
        del files
        budget()
        need(not any(name.split(".", 1)[0] in ML_ROOTS for name in sys.modules),
             "runtime_rejected")
        sys.path.insert(0, str(source))
        saved_stdout = os.dup(1)
        os.dup2(2, 1)  # The trusted guardian supplies /dev/null on stderr.
        module = importlib.import_module("lumi.posttraining_checkpoint_guard_fixture_v1")
        root = Path(tempfile.mkdtemp(prefix="lumi-import-fixture-", dir="/tmp"))
        root.chmod(0o700)
        outcome = module.run_tiny_fixture(root, check_budget=budget)
        budget()
        report = {**outcome, **source_report, **copy_report,
                  "entry_sha256": environment["LUMI_IMPORT_ENTRY_SHA256"]}
        need(len(canonical(report)) <= MAXIMUM_REPORT_BYTES, "output_failed")
        success = True
    except EntryError as error:
        report = {"schema": REPORT_SCHEMA, "status": "failed", "code": error.code,
                  "synthetic_only": True, "real_4b_import": False,
                  "training_authorized": False, "production_eligible": False}
    except BaseException:
        report = {"schema": REPORT_SCHEMA, "status": "failed", "code": "fixture_failed",
                  "synthetic_only": True, "real_4b_import": False,
                  "training_authorized": False, "production_eligible": False}
    finally:
        if saved_stdout is not None:
            try:
                os.dup2(saved_stdout, 1)
            finally:
                os.close(saved_stdout)
    try:
        raw = canonical(report) + b"\n"
        need(len(raw) <= MAXIMUM_REPORT_BYTES, "output_failed")
        need(os.write(1, raw) == len(raw), "output_failed")
    except BaseException:
        return 2
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
