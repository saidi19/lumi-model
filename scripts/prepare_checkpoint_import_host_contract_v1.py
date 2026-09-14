#!/usr/bin/env python3
"""Read fixed local control/source pins and print an inert preparation result.

No arguments, arbitrary input paths, writes, Git, network, payloads or native
execution. Only the new pure contract module is evaluated, after its exact pin.
The old runner, model kernel, mount helpers and report are only hashed.
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
PLAN = "configs/lumi-4b-sft-checkpoint-import-plan-v0.1.json"
QUALIFICATION = "evidence/lumi-4b-sft-import-qualification-execution-verification-v0.2.json"
SUBJECT = "src/lumi/posttraining_checkpoint_host_contract_v1.py"
PINS = {
    PLAN: (70554, "3138f88ddb871c0a24134a5aae84cb28128581b8d943ca599f24553bd3c3cbd8"),
    QUALIFICATION: (2236, "680f9fd3a26d18908caf906505ab05ade5fa87b45aafb471f01790c5df9b63f8"),
    "evidence/lumi-4b-sft-import-qualification-execution-v0.2.json":
        (27831, "bcac947786eec52131d9a3e25cffbdb2f567969c77a6f8236934093ac943e931"),
    "src/lumi/posttraining_checkpoint_import_runner_v2.py":
        (21718, "d4babc49583691cc065d5048b66100ce33875d05d2bdf61b74c105ac5eb94018"),
    "src/lumi/posttraining_checkpoint_identity_v1.py":
        (6387, "191aa6f256e63ec7fc651e5c3a58870b3724720c8b38f94e1439bbdd342a543f"),
    "src/lumi/posttraining_sft_oci_mount_v1.py":
        (14573, "76850cfa55ccfb94490cd789b164f69d076cb2b01bd0c5f1586ceca9da79434a"),
    "src/lumi/posttraining_checkpoint_guard_bootstrap_v1.py":
        (10282, "b1b02586f45980d2af52e0afc52ff6c02b64067dc122dac08f664a2e80f8f95e"),
    "src/lumi/posttraining_checkpoint_guard_composition_v1.py":
        (19122, "3078287b1dafeadc6572b7365c138c93f893f6cfa1e9e1b7a36aef337b3ca580"),
    "src/lumi/posttraining_checkpoint_package_v2.py":
        (34669, "a3cb6503dd78e5c0e230460c6bff78009fe305f117ca9978c241c21e080a2411"),
    "src/lumi/posttraining_checkpoint_import_torch_v5.py":
        (30718, "1e1b18204198825fd02680d65b402418d167100afb56965d06de44f48309a46f"),
    "src/lumi/posttraining_checkpoint_io_contract_v1.py":
        (7020, "509f19258c43638fe22cc325533b3834c82bbf7ac3b9d948bd88a7b81a920882"),
    SUBJECT: (15754, "90ca307ad0d907ec3fce7f4fc8963ca496a0b31970ba3202b5e28443580fe873"),
}
MAXIMUM_TOTAL_CONTROL_READ_BYTES = 300000


class PreparationError(ValueError):
    def __init__(self):
        super().__init__("control_preparation_rejected")


def need(value):
    if not value:
        raise PreparationError() from None


def identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink, info.st_mtime_ns, info.st_ctime_ns)


def read_bound(root, relative, binding):
    """One exact bounded read; any short read/mutation is terminal, no retry."""
    size, expected = binding
    path = root / relative
    need(type(size) is int and 0 < size <= 131072 and relative in PINS)
    need(not any(parent.is_symlink() for parent in (path, *path.parents)))
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and before.st_size == size and before.st_nlink == 1)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        need(identity(os.fstat(descriptor)) == identity(before))
        raw = os.read(descriptor, size)
        need(len(raw) == size and hashlib.sha256(raw).hexdigest() == expected
             and identity(os.fstat(descriptor)) == identity(before)
             and identity(path.lstat()) == identity(before))
    finally:
        os.close(descriptor)
    return raw


def prepare(root):
    need(sum(size for size, _ in PINS.values()) <= MAXIMUM_TOTAL_CONTROL_READ_BYTES)
    files = {name: read_bound(root, name, pin) for name, pin in PINS.items()}
    subject = types.ModuleType("_lumi_inert_host_contract")
    exec(compile(files[SUBJECT], str(root / SUBJECT), "exec"), subject.__dict__)
    contract = subject.build_contract(
        files[PLAN], files[QUALIFICATION], expected_plan_file_sha256=PINS[PLAN][1],
        expected_qualification_file_sha256=PINS[QUALIFICATION][1])
    return {
        "schema_version": "lumi.sft-checkpoint-host-preparation.v0.1",
        "status": "PASS_control_preparation_not_host_admission", "contract": contract,
        "verified_control_and_source_files": [
            {"path": name, "bytes": size, "sha256": digest}
            for name, (size, digest) in PINS.items()],
        "control_and_source_bytes_read": sum(len(raw) for raw in files.values()),
        "model_tokenizer_dataset_payload_reads": 0, "network_operations": 0,
        "native_mount_or_harness_calls": 0, "files_written_by_preparer": 0,
        "execution_permitted": False, "training_authorized": False,
    }


def main():
    try:
        need(len(sys.argv) == 1)
        result = prepare(ROOT)
        print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False))
        return 0
    except Exception:
        print('{"status":"STOP","code":"control_preparation_rejected"}')
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
