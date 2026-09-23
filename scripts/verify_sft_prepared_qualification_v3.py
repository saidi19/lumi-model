#!/usr/bin/env python3
"""Source-only and closed receipt checks for the inert local v3 successor.

Source bytes are authenticated before compilation. Native validation is reused
unchanged from v2; fresh acquisition and prefix evidence are additional bindings,
never permission to execute a native worker or real training.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "scripts/verify_sft_prepared_qualification_v2.py"
FROZEN_SHA256 = "d888d17f893e8827e3a051410de5c93a6706996a232f461016b2930caad5ee26"
ADAPTER = ROOT / "scripts/prepare_sft_prepared_qualification_v3.py"
ADAPTER_SHA256 = "b874a73d527c9ad8381f9616726a1d020bccf54d84dabdf332810a76ddc0a256"
PREFIX = ROOT / "scripts/reclaim_sft_sdk_prefix_v1.py"
PREFIX_SHA256 = "d8a0fd9e370aec987724799b10a20c501ccdafb3b71bb502fff86b57d76fb6b1"
PREFIX_REPORT = ROOT / "artifacts/lumi-sft-sdk-prefix-capacity-v0.1/aggregate-prefix-capacity.json"
OUTPUT = "artifacts/lumi-sft-prepared-qualification-v0.3"
SCHEMA = "lumi.sft-prepared-qualification-verification.v0.3"
MAXIMUM = 32768
FILES = ("aggregate-acquisition.json", "aggregate-native.json", "aggregate-input-cleanup.json")
FALSE_FLAGS = ("real_import_verified", "execution_permitted", "training_authorized",
               "production_eligible", "gpu_used", "guardian_death_cleanup_verified",
               "job_teardown_cleanup_verified", "source_immutability_verified",
               "model_or_dataset_access")


class VerificationError(ValueError):
    pass


def need(value):
    if not value:
        raise VerificationError("qualification_rejected") from None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii") + b"\n"


def state(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def read_closed(path, maximum):
    """Bounded stable nofollow read; never return a mutable source pathname."""
    need(type(maximum) is int and 0 < maximum <= MAXIMUM)
    need(not any(part.is_symlink() for part in (path, *path.parents)))
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
             and not before.st_mode & 0o022 and 0 < before.st_size <= maximum)
        parts, count = [], 0
        while count < before.st_size:
            part = os.read(fd, before.st_size - count)
            need(part)
            parts.append(part)
            count += len(part)
        need(not os.read(fd, 1)
             and state(before) == state(os.fstat(fd)) == state(path.lstat()))
        return b"".join(parts)
    finally:
        os.close(fd)


def authenticated(path, expected, maximum):
    body = read_closed(path, maximum)
    need(hashlib.sha256(body).hexdigest() == expected)
    return body


def load_module(path, expected, maximum=MAXIMUM):
    body = authenticated(path, expected, maximum)
    module = types.ModuleType("_lumi_v3_bound_" + path.stem)
    module.__file__ = str(path)
    exec(compile(body, str(path), "exec"), module.__dict__)
    return module


def load_frozen():
    return load_module(FROZEN, FROZEN_SHA256)


def load_sources():
    frozen = load_frozen()
    adapter = load_module(ADAPTER, ADAPTER_SHA256)
    acquisition = adapter.load_frozen()
    sources = frozen.load_sources()  # Unchanged 78-file/native source admission.
    need(sources[2].FALSE_FLAGS == acquisition.FALSE_FLAGS)
    return frozen, (*sources[:2], acquisition, *sources[3:])


def decode(raw):
    def pairs(rows):
        result = {}
        for key, part in rows:
            need(key not in result)
            result[key] = part
        return result

    def integer(raw_number):
        need(len(raw_number) <= 20)
        value = int(raw_number)
        need(-(2**63) <= value < 2**63)
        return value

    def reject(_raw):
        raise VerificationError("qualification_rejected") from None

    need(type(raw) is bytes and 0 < len(raw) <= MAXIMUM and raw.isascii())
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_int=integer,
                           parse_float=reject, parse_constant=reject)
        need(type(value) is dict and canonical(value) == raw)
        return value
    except BaseException:
        raise VerificationError("qualification_rejected") from None


def verify_prefix(check_capacity=False):
    prefix = load_module(PREFIX, PREFIX_SHA256)
    probe = prefix.load_helper().load_probe()
    raw = read_closed(PREFIX_REPORT, prefix.MAXIMUM_REPORT_BYTES)
    report = decode(raw)
    prefix.validate_report(report, probe)
    need(report["status"] == "observed" and report["code"] == "ok"
         and report["ready_for_unchanged_setup"] and report["sdk_root_preserved"]
         and report["selected_absence_verified"] and report["unselected_root_names_unchanged"])
    if check_capacity:
        # Fresh measurement before acquisition, not a reservation.
        need(probe.capacity(os.statvfs("/tmp"))["acquisition_headroom_met"])
    return report, hashlib.sha256(raw).hexdigest()


def verify_receipts():
    frozen, sources = load_sources()
    prefix, prefix_digest = verify_prefix()
    parts = [decode(read_closed(ROOT / OUTPUT / name, MAXIMUM)) for name in FILES]
    result = frozen.validate(*parts, sources)
    return {**result, "schema": SCHEMA, "receipt_count": 4,
            "prefix_capacity_receipt_verified": True,
            "prefix_capacity_receipt_sha256": prefix_digest,
            "prefix_selected_shards": prefix["selected_shards"],
            "prefix_census_nodes": prefix["census_nodes"],
            **dict.fromkeys(FALSE_FLAGS, False)}


def write_verification(result):
    parent = ROOT / OUTPUT
    need(not any(part.is_symlink() for part in (parent, *parent.parents)))
    info = parent.lstat()
    need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
         and not info.st_mode & 0o077)
    fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        need(state(info) == state(os.fstat(fd)))
        raw = canonical(result)
        need(len(raw) <= MAXIMUM)
        target = os.open("aggregate-verification.json",
                         os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                         0o600, dir_fd=fd)
        try:
            need(os.write(target, raw) == len(raw))
        finally:
            os.close(target)
    finally:
        os.close(fd)


def failure(code="qualification_rejected"):
    need(code in {"qualification_rejected", "verification_write_rejected"})
    return {"schema": SCHEMA, "status": "failed", "code": code,
            **dict.fromkeys(FALSE_FLAGS, False)}


def main():
    receipts = sys.argv[1:] == ["--verify-receipts"]
    try:
        if receipts:
            result = verify_receipts()
        elif sys.argv[1:] == ["--check-source"]:
            _frozen, sources = load_sources()
            result = {"schema": SCHEMA, "status": "passed", "code": "source_bindings_only",
                      "source_files_verified": sources[5], **dict.fromkeys(FALSE_FLAGS, False)}
        else:
            need(sys.argv[1:] == ["--check-prefix"])
            _report, digest = verify_prefix(check_capacity=True)
            result = {"schema": SCHEMA, "status": "passed", "code": "prefix_capacity_ready",
                      "prefix_capacity_receipt_sha256": digest,
                      **dict.fromkeys(FALSE_FLAGS, False)}
    except BaseException:
        result = failure()
    if receipts:
        try:
            write_verification(result)
        except BaseException:
            result = failure("verification_write_rejected")
    try:
        sys.stdout.buffer.write(canonical(result))
    except BaseException:
        return 2  # No traceback or retry of an uncertain write.
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
