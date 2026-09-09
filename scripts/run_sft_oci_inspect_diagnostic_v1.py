#!/usr/bin/env python3
"""One remote-only, never-started OCI inspection localization transaction.

The old accepted software, create profile and admission predicates are frozen.
Only fixed diagnostic codes/booleans escape; no inspect values or exception text.
No export, rootfs extraction, workload, cgroup, GPU, inference or SFT entry exists.
Local tests must use source inspection and doubles, never this native entry.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import time
import types
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "lumi.sft-oci-inspect-fixture.v0.1"
HELPER_SOURCE = "scripts/run_sft_oci_guardian_fixture_v1.py"
HELPER_SHA256 = "38d64b4159bcc418ff799c14fae6cde00ec98b35b58cbfeb58fb14b2e83d8f76"
DIAGNOSTIC_SOURCE = "src/lumi/posttraining_sft_oci_inspect_diagnostic_v1.py"
DIAGNOSTIC_SHA256 = "54fb165e864db5a69d41875e014e933dcba443f8ce39ea3d8a0532656848067b"
IMAGE_ID = "sha256:321d50ae5207dee05cd58ec5436d97978becde2ee8a767f2e8768229e6a9ad43"
MAXIMUM_SECONDS = 420
MAXIMUM_REPORT_BYTES = 8192
STAGES = ("preflight", "copy_image", "prove_absence", "load_image", "create_container",
          "diagnose", "cleanup", "complete")
ERROR_CODES = frozenset({"fixture_failed", "runtime_rejected", "source_rejected",
                         "deadline_exhausted", "command_failed", "output_exhausted",
                         "diagnostic_rejected", "cleanup_unconfirmed"})
NONCLAIMS = ("real_sft_execution_permitted", "production_eligible", "real_model_or_data_access",
             "gpu_used", "inference_called", "registry_published", "private_integration_passed",
             "full_ci_passed", "guardian_containment_verified", "worker_identity_bound",
             "rootfs_contents_verified", "software_binding_released")
STAGE = "preflight"
HELPERS = None


class DiagnosticFixtureError(ValueError):
    def __init__(self, code="fixture_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "fixture_failed"
        super().__init__(self.code)


def need(value, code="fixture_failed"):
    if not value:
        raise DiagnosticFixtureError(code) from None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"


def identity(info):
    return info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode), info.st_uid, info.st_gid


def read_source(relative, expected):
    need(type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected), "source_rejected")
    path = ROOT / relative
    for ancestor in (path, *path.parents):
        need(not ancestor.is_symlink(), "source_rejected")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= 131072, "source_rejected")
        raw = bytearray()
        while len(raw) < before.st_size:
            part = os.read(fd, min(65536, before.st_size - len(raw)))
            need(part, "source_rejected")
            raw.extend(part)
        after = os.fstat(fd)
        fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_size",
                  "st_mtime_ns", "st_ctime_ns")
        need(all(getattr(before, name) == getattr(after, name) for name in fields)
             and not os.read(fd, 1) and hashlib.sha256(raw).hexdigest() == expected,
             "source_rejected")
        return bytes(raw)
    finally:
        os.close(fd)


def load_source(relative, expected, name):
    raw = read_source(relative, expected)
    need(name not in sys.modules, "source_rejected")
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / relative)
    sys.modules[name] = module
    try:
        exec(compile(raw, module.__file__, "exec"), module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise DiagnosticFixtureError("source_rejected") from None
    return module


def runtime():
    need(sys.platform == "linux" and os.uname().machine == "x86_64"
         and sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12)
         and sys.flags.no_site and sys.flags.isolated and sys.dont_write_bytecode
         and os.getresuid() == os.getresgid() == (0, 0, 0)
         and os.environ.get("LUMI_OCI_INSPECT_DIAGNOSTIC_CI") == "1", "runtime_rejected")
    need(not any(name.split(".", 1)[0] in
                 {"torch", "transformers", "numpy", "cuda", "cupy", "jax", "tensorflow"}
                 for name in tuple(sys.modules)), "runtime_rejected")
    os.environ.clear()
    os.environ.update({"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"})
    os.umask(0o077)


def run(archive_path):
    global HELPERS, STAGE
    runtime()
    overall_deadline = time.monotonic() + MAXIMUM_SECONDS
    deadline = overall_deadline - 100  # Reserve owned retirement before the outer timeout.
    helpers = HELPERS = load_source(HELPER_SOURCE, HELPER_SHA256, "_lumi_inspect_frozen_helpers")
    diagnostic = load_source(DIAGNOSTIC_SOURCE, DIAGNOSTIC_SHA256, "_lumi_inspect_localization")
    parent = Path(tempfile.mkdtemp(prefix="lumi-oci-inspect-diagnostic-", dir="/tmp"))
    parent_id = identity(parent.stat())
    config, image = parent / "docker-config", parent / "image.tar"
    directory_ids, file_ids = {}, {}
    load_started = load_completed = create_started = create_completed = False
    container_name = "lumi-oci-guardian-" + uuid.uuid4().hex
    cleanup = {"owned_container_absent": False, "owned_image_absent": False,
               "owned_files_retired": False, "owned_directories_retired": False,
               "verified": False}
    result = None
    try:
        config.mkdir(mode=0o700)
        directory_ids[config] = identity(config.stat())
        space = os.statvfs(parent)
        need(space.f_bavail * space.f_frsize >= 12 * 1024**3)
        STAGE = "copy_image"
        helpers.copy_image(archive_path, image, deadline, file_ids)
        STAGE = "prove_absence"
        need(IMAGE_ID not in helpers.image_inventory(config, min(deadline, time.monotonic()+10)))
        need(container_name not in helpers.container_inventory(
            config, min(deadline, time.monotonic()+10)))
        STAGE = "load_image"
        with image.open("rb") as source:
            load_started = True
            helpers.command(config, ["image", "load"],
                            deadline=min(deadline, time.monotonic()+180), input_file=source)
        load_completed = True
        STAGE = "create_container"
        create_started = True
        raw, _size, _sha, _status = helpers.command(
            config, helpers.create_options(IMAGE_ID, container_name),
            deadline=min(deadline, time.monotonic()+30))
        create_completed = True
        container_id = raw.decode("ascii").strip()
        need(re.fullmatch(r"[0-9a-f]{64}", container_id) is not None)
        STAGE = "diagnose"
        localized = diagnostic.diagnose(container_id, deadline=min(deadline, time.monotonic()+15))
        localized = diagnostic.validate_report(localized)
        # Preserve even a validated source/instrumentation failure report; do
        # not recreate the old wrapper's error-code information loss.
        result = {"schema": SCHEMA, "status": localized["status"], "diagnostic": localized,
                  "image_loaded": True, "container_created": True,
                  "container_started": False, "image_exported": False,
                  "rootfs_extracted": False, "workload_created": False,
                  **dict.fromkeys(NONCLAIMS, False)}
    finally:
        previous_stage = STAGE
        STAGE = "cleanup"
        cleanup_deadline = min(overall_deadline, time.monotonic()+95)
        failures = []
        if create_started:
            create_started = False  # Never retry an ambiguous daemon write or retirement.
            try:
                if container_name in helpers.container_inventory(
                        config, min(cleanup_deadline, time.monotonic()+10)):
                    helpers.command(config, ["rm", container_name],
                                    deadline=min(cleanup_deadline, time.monotonic()+15))
                need(create_completed, "cleanup_unconfirmed")
                need(container_name not in helpers.container_inventory(
                    config, min(cleanup_deadline, time.monotonic()+10)), "cleanup_unconfirmed")
                cleanup["owned_container_absent"] = True
            except BaseException:
                failures.append(True)
        else:
            cleanup["owned_container_absent"] = True
        if load_started:
            load_started = False
            try:
                if IMAGE_ID in helpers.image_inventory(
                        config, min(cleanup_deadline, time.monotonic()+10)):
                    helpers.command(config, ["image", "rm", "--no-prune", IMAGE_ID],
                                    deadline=min(cleanup_deadline, time.monotonic()+30))
                need(load_completed, "cleanup_unconfirmed")
                need(IMAGE_ID not in helpers.image_inventory(
                    config, min(cleanup_deadline, time.monotonic()+10)),
                     "cleanup_unconfirmed")
                cleanup["owned_image_absent"] = True
            except BaseException:
                failures.append(True)
        else:
            cleanup["owned_image_absent"] = True
        for path, expected in file_ids.items():
            try:
                need(identity(path.lstat()) == expected)
                path.unlink()
            except BaseException:
                failures.append(True)
        cleanup["owned_files_retired"] = not any(path.exists() for path in file_ids)
        for path in reversed(directory_ids):
            try:
                need(identity(path.lstat()) == directory_ids[path])
                path.rmdir()  # Refuse unexpected entries, never delete recursively.
            except BaseException:
                failures.append(True)
        if not failures:
            need(identity(parent.lstat()) == parent_id)
            parent.rmdir()
            cleanup["owned_directories_retired"] = True
        cleanup["verified"] = not failures and all(cleanup[name] for name in
            ("owned_container_absent", "owned_image_absent", "owned_files_retired",
             "owned_directories_retired"))
        need(cleanup["verified"], "cleanup_unconfirmed")
        STAGE = previous_stage
    need(result is not None, "diagnostic_rejected")
    STAGE = "complete"
    return dict(result, cleanup=cleanup)


def main():
    try:
        need(len(sys.argv) == 3 and sys.argv[1] == "--diagnostic-only")
        result = run(Path(sys.argv[2]))
    except BaseException as error:
        recognized = type(error) is DiagnosticFixtureError or (
            HELPERS is not None and type(error) is HELPERS.FixtureError)
        code = error.code if recognized and error.code in ERROR_CODES else "fixture_failed"
        result = {"schema": SCHEMA, "status": "failed", "stage": STAGE, "code": code,
                  **dict.fromkeys(NONCLAIMS, False)}
    raw = canonical(result)
    need(len(raw) <= MAXIMUM_REPORT_BYTES)
    sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()
    raise SystemExit(0 if result["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
