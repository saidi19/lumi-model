#!/usr/bin/env python3
"""Versioned successor adapter for one fresh prepared CPU qualification.

The v2 acquisition implementation is loaded only after a bounded byte/hash
authentication. Its parsing, download, hash, ownership and retirement logic is
unchanged; this adapter changes only the successor workflow/parent/output/schema
bindings. It never runs on the Mac and never reads model or dataset payloads.
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
FROZEN = ROOT / "scripts/prepare_sft_prepared_qualification_v2.py"
FROZEN_SHA256 = "7dc3543a354d3b4bca154b72ca6d2e77ffd2e229559a4f4d4da157dd46213065"
SCHEMA = "lumi.sft-prepared-qualification-acquisition.v0.3"
WORKFLOW = "SFT prepared importer qualification v3"
CI_KEY = "LUMI_PREPARED_QUALIFICATION_ACQUISITION_V3_CI"
APPROVED_PARENT = "bd38d257038632d6424cbc08128219776feeb90a"
OUTPUT = "artifacts/lumi-sft-prepared-qualification-v0.3"
# Frozen retirement regex is v2-bound; a fresh VM creates a unique directory.
INPUT_PREFIX = "lumi-prepared-qualification-v2-input-"
_MODULE = None
FALSE_FLAGS = ("real_model_or_data_access", "gpu_used", "inference_called",
               "real_sft_execution_permitted", "production_eligible", "worker_dispatched")


class AdapterError(ValueError):
    pass


def need(value):
    if not value:
        raise AdapterError("binding_rejected") from None


def _state(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def load_frozen():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    path = FROZEN
    need(not any(part.is_symlink() for part in (path, *path.parents)))
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
             and not before.st_mode & 0o022 and 0 < before.st_size <= 32768)
        parts, count = [], 0
        while count < before.st_size:
            part = os.read(fd, before.st_size - count)
            need(part)
            parts.append(part)
            count += len(part)
        body = b"".join(parts)
        need(len(body) == before.st_size and not os.read(fd, 1)
             and _state(before) == _state(os.fstat(fd)) == _state(path.lstat())
             and hashlib.sha256(body).hexdigest() == FROZEN_SHA256)
    finally:
        os.close(fd)
    module = types.ModuleType("_lumi_prepared_acquisition_v2_frozen")
    module.__file__ = str(path)
    exec(compile(body, str(path), "exec"), module.__dict__)
    need(module.INPUT_PREFIX == INPUT_PREFIX and module.FALSE_FLAGS == FALSE_FLAGS)
    module.SCHEMA = SCHEMA
    module.WORKFLOW = WORKFLOW
    module.CI_KEY = CI_KEY
    module.APPROVED_PARENT = APPROVED_PARENT
    module.OUTPUT = OUTPUT
    _MODULE = module
    return module


def main():
    try:
        # Preserve the frozen main's closed failures, exclusive report writes,
        # input retirement, argument gate and consumed-attempt state machine.
        return load_frozen().main()
    except BaseException:
        result = {"schema": SCHEMA, "status": "failed", "stage": "binding",
                  "code": "binding_rejected", **dict.fromkeys(FALSE_FLAGS, False)}
        raw = (json.dumps(result, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n")
        try:
            sys.stdout.buffer.write(raw)
        except BaseException:
            pass  # Never disclose an exception or retry a failed output write.
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
