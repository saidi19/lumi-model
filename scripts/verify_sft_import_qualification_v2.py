#!/usr/bin/env python3
"""Bound source and exact aggregate checks; never a native execution permit."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = "artifacts/lumi-sft-import-qualification-v0.2"
SCHEMA = "lumi.sft-import-qualification-verification.v0.2"
MAXIMUM = 32768
PINS = {
    "scripts/run_checkpoint_guard_qualification_v1.py":
        "918267c9d051991f2b9ae5abd150ca33fdc53c3d7d4e293e7b7ffcf56ea27753",
    "scripts/supervise_checkpoint_guard_qualification_v1.py":
        "3525eb5bf60fd0bac35480dd845a1cb11c6fc57303c6c3cb03b5c56576cfc1d6",
    "scripts/prepare_sft_import_qualification_v2.py":
        "c291e73b8ced404661818af85ae31371bc33056c2840f9c311d1786064b11770",
}
FILES = ("aggregate-acquisition.json", "aggregate-native.json", "aggregate-input-cleanup.json")
FALSE_FLAGS = ("real_import_verified", "execution_permitted", "training_authorized",
               "production_eligible", "gpu_used", "guardian_death_cleanup_verified",
               "job_teardown_cleanup_verified", "source_immutability_verified")


class VerificationError(ValueError):
    pass


def need(value):
    if not value:
        raise VerificationError("qualification_rejected") from None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii") + b"\n"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def snapshot(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid,
            value.st_nlink, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def read(path, maximum):
    for part in (path, *path.parents):
        need(not part.is_symlink())
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
             and not before.st_mode & 0o022 and 0 < before.st_size <= maximum)
        raw = os.read(fd, maximum + 1)
        need(len(raw) == before.st_size
             and snapshot(before) == snapshot(os.fstat(fd)) == snapshot(path.lstat()))
        return raw
    finally:
        os.close(fd)


def decode(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            need(key not in result)
            result[key] = value
        return result

    def reject(_value):
        raise VerificationError("qualification_rejected") from None

    need(type(raw) is bytes and 0 < len(raw) <= MAXIMUM and raw.isascii())
    try:
        result = json.loads(raw, object_pairs_hook=pairs, parse_float=reject,
                            parse_constant=reject)
        need(type(result) is dict and canonical(result) == raw)
        return result
    except BaseException:
        raise VerificationError("qualification_rejected") from None


def exact(value, expected, extra=()):
    need(type(value) is dict and set(value) == set(expected) | set(extra))
    need(all(type(value[key]) is type(part) and value[key] == part
             for key, part in expected.items()))


def load_sources():
    modules = []
    for name, pin in PINS.items():
        raw = read(ROOT / name, 131072)
        need(digest(raw) == pin)
        module = types.ModuleType("_qualification_verify_" + Path(name).stem)
        module.__file__ = str(ROOT / name)
        exec(compile(raw, module.__file__, "exec"), module.__dict__)
        modules.append(module)
    h, s, a = modules
    helper, addon, q, files, manifest, entry = h.load_sources()
    h.QUALIFICATION = q
    binding = helper.load_admission().SoftwareBinding()
    expected_addon = {
        "schema": addon.RESULT_SCHEMA, "status": "installed",
        **addon.validate_bundle(files, manifest, entry,
            expected_manifest_sha256=digest(manifest), expected_entry_sha256=digest(entry)),
        "installation_frozen": True, "base_origin_verified_by_this_adapter": False,
        "filesystem_immutable": False, "native_guardian_verified": False,
        "real_import_authorized": False, "production_eligible": False,
    }
    return h, s, a, addon, expected_addon, binding


def expected_acquisition(a):
    return {"schema": a.SCHEMA, "status": "passed", "https_requests": 4,
            "prior_guardian_run": a.PRIOR_RUN_ID, "prior_guardian_job": a.PRIOR_JOB_ID,
            "prior_guardian_success_verified": True, "prior_input_cleanup_verified": True,
            "artifact_id": a.ARTIFACT_ID, "accepted_run_id": a.ACCEPTED_RUN_ID,
            "source_commit": a.ACCEPTED_COMMIT, "artifact_zip_bytes": a.ZIP_BYTES,
            "artifact_zip_sha256": a.ZIP_SHA256, "software_tar_bytes": a.TAR_BYTES,
            "software_tar_sha256": a.TAR_SHA256, "network_retries": 0,
            "old_diagnostic_artifact_reads": 0, "native_qualification_retried": False,
            **dict.fromkeys(a.FALSE_FLAGS, False)}


def expected_retirement(a):
    return {"schema": a.SCHEMA, "status": "passed", "owned_input_files_retired": 1,
            "owned_input_directories_retired": 1, **dict.fromkeys(a.FALSE_FLAGS, False)}


def extraction(value, h, addon, summary, export_bytes):
    fixed = {"schema": addon.rootfs.SCHEMA, "archive_bytes": export_bytes,
             "installed_files": summary["addon_files"], "installed_bytes": summary["addon_bytes"],
             "installation_frozen": True, "cleanup_attempted": False, "cleanup_completed": False,
             "operation_tainted": False, "metadata_normalized": True,
             "filesystem_immutable": False, "production_eligible": False,
             "real_sft_execution_permitted": False}
    numeric = ("archive_entries", "owned_entries", "archive_headers", "metadata_bytes",
               "content_bytes", "hardlinks", "symlinks")
    exact(value, fixed, numeric)
    need(all(type(value[key]) is int and value[key] >= 0 for key in numeric))
    need(0 < value["archive_entries"] <= value["archive_headers"] <= 65536
         and summary["addon_files"] < value["owned_entries"] <= 65536
         and value["metadata_bytes"] <= export_bytes
         and summary["addon_bytes"] <= value["content_bytes"] <= h.MAXIMUM_EXTRACT_BYTES
         and value["symlinks"] + value["hardlinks"] <= value["archive_entries"])


def validate_harness(value, sources):
    h, s, _a, addon, summary, binding = sources
    s.parse_report(canonical(value), 0)  # Bounded types, strings and recursive shape.
    fixed = {"schema": h.SCHEMA, "status": "passed", "stage": "complete", "code": "ok",
             "accepted_image_id": binding.image_id,
             "source_identity_sha256": binding.source_identity_sha256(),
             "archive_bytes": h.ARCHIVE_BYTES, "entry_sha256": summary["entry_sha256"],
             "importer_manifest_sha256": summary["source_manifest_sha256"],
             "source_admission_sha256": h.SOURCE_ADMISSION_SHA256,
             "docker_containers_started": 0, "images_built": 0,
             "private_owned_cleanup_verified": True, **dict.fromkeys(h.NONCLAIMS, False)}
    exact(value, fixed, ("cases", "export_bytes", "export_sha256", "extraction",
                         "addon", "cleanup_inventory"))
    need(type(value["export_bytes"]) is int
         and 0 < value["export_bytes"] <= h.MAXIMUM_EXPORT_BYTES)
    need(type(value["export_sha256"]) is str
         and h.re.fullmatch(r"[a-f0-9]{64}", value["export_sha256"]) is not None)
    exact(value["addon"], summary)
    exact(value["cleanup_inventory"],
          {"owned_container_absent": True, "owned_image_absent": True})
    extraction(value["extraction"], h, addon, summary, value["export_bytes"])
    need(type(value["cases"]) is list and len(value["cases"]) == 3)
    environment = {**h.QUALIFICATION.frozen.child_bootstrap.FIXED_ENV,
                   "LUMI_IMPORT_ENTRY_SHA256": summary["entry_sha256"],
                   "LUMI_IMPORT_SOURCE_MANIFEST_SHA256": summary["source_manifest_sha256"]}
    for case, part in zip(h.CASES, value["cases"], strict=True):
        h.validate_case(part, case, environment)


def validate(acquisition, native, retirement, sources):
    h, s, a, _addon, _summary, _binding = sources
    exact(acquisition, expected_acquisition(a))
    exact(retirement, expected_retirement(a))
    need(type(native) is dict and type(native.get("harness_report")) is dict)
    raw = canonical(native["harness_report"])
    fixed = {"schema": s.SCHEMA, "status": "passed", "code": "ok",
             "harness_sha256": PINS["scripts/run_checkpoint_guard_qualification_v1.py"],
             "harness_stdout_bytes": len(raw), "harness_stdout_sha256": digest(raw),
             "harness_exit_success": True, "harness_pidfd_bound": True,
             "harness_term_attempts": 0, "harness_kill_attempts": 0,
             "harness_reaped": True, "adopted_children_reaped": 0, "all_children_reaped": True,
             "guardian_signalled_by_supervisor": False, "process_group_signalled": False,
             "guardian_death_cleanup_verified": False, "job_teardown_cleanup_verified": False,
             "source_immutability_verified": False, "real_4b_import": False,
             "training_authorized": False, "production_eligible": False, "gpu_used": False}
    exact(native, fixed, ("harness_report",))
    validate_harness(native["harness_report"], sources)
    return {"schema": SCHEMA, "status": "passed", "code": "ok",
            "tiny_native_qualification_receipts_verified": True,
            "receipt_count": 3, "ordered_cases_verified": len(h.CASES),
            "source_files_verified": 57, "all_owned_cleanup_receipts_verified": True,
            **dict.fromkeys(FALSE_FLAGS, False)}


def verify_receipts():
    sources = load_sources()
    parts = [decode(read(ROOT / OUTPUT / name, MAXIMUM)) for name in FILES]
    return validate(*parts, sources)


def write_verification(result):
    path = ROOT / OUTPUT / "aggregate-verification.json"
    parent = path.parent
    need(parent.is_dir() and not parent.is_symlink()
         and parent.stat().st_uid == os.geteuid() and not parent.stat().st_mode & 0o077)
    for part in parent.parents:
        need(not part.is_symlink())
    raw = canonical(result)
    need(len(raw) <= MAXIMUM)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        need(os.write(fd, raw) == len(raw))
    finally:
        os.close(fd)


def main():
    receipts = sys.argv[1:] == ["--verify-receipts"]
    try:
        need(receipts or sys.argv[1:] == ["--check-source"])
        if receipts:
            result = verify_receipts()
        else:
            load_sources()
            result = {"schema": SCHEMA, "status": "passed", "code": "source_bindings_only",
                      "source_files_verified": 57, **dict.fromkeys(FALSE_FLAGS, False)}
    except BaseException:
        result = {"schema": SCHEMA, "status": "failed", "code": "qualification_rejected",
                  **dict.fromkeys(FALSE_FLAGS, False)}
    if receipts:
        try:
            write_verification(result)
        except BaseException:
            result = {"schema": SCHEMA, "status": "failed", "code": "verification_write_rejected",
                      **dict.fromkeys(FALSE_FLAGS, False)}
    sys.stdout.buffer.write(canonical(result))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
