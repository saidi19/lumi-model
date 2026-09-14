"""Fixed tiny input preparation for a separately contained, authenticated worker.

The trusted parent must authenticate this source and the runtime and enforce
the process deadline/resources/network boundary before calling this API. It
does not establish those premises, launch a child, import a real checkpoint or
authorize training. Never call the native path on a Mac. Tests replace the
tensor generator with writes of newly generated opaque bytes.

Only a fresh empty private directory is accepted. There is no input-data path,
model configuration, callback, cleanup, resume or retry argument. The frozen
736-parameter generator and metadata writer are reused, but no importer runs.
Final output-size checks are not a claim of cumulative write metering.
"""

from __future__ import annotations

import json
import os
import re
import stat
import sys
import threading
import time
from pathlib import Path, PurePosixPath

from . import posttraining_checkpoint_guard_fixture_v1 as fixture
from .posttraining_checkpoint_import_plan_v1 import build_import_plan
from .posttraining_final_base_metadata_v1 import canonical, sha

SCHEMA = "lumi.prepared-checkpoint-input-generation.v0.1"
MAXIMUM_SECONDS = 30
MAXIMUM_INPUT_BYTES = 4 * 1024**2
MAXIMUM_INPUT_FILES = 32
MAXIMUM_METADATA_BYTES = 524288
MAXIMUM_FILE_BYTES = 1024**2
MARKER = b"lumi-prepared-tiny-generation-attempt-v1\n"
ERROR_CODES = frozenset({"runtime_rejected", "arguments_rejected", "already_attempted",
                         "generation_failed", "inventory_rejected", "deadline_exhausted"})
_ATTEMPTED = False


class PreparedGenerationError(ValueError):
    def __init__(self, code="generation_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "generation_failed"
        super().__init__(self.code)


def _need(value, code="generation_failed"):
    if not value:
        raise PreparedGenerationError(code) from None


def _runtime():
    _need(sys.platform == "linux" and os.uname().machine == "x86_64"
          and sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12)
          and sys.flags.isolated == sys.flags.ignore_environment == sys.flags.no_user_site == 1
          and sys.dont_write_bytecode and not sys.flags.optimize and sys.byteorder == "little"
          and os.getresuid() == os.getresgid() == (10001, 10001, 10001)
          and os.getgroups() == [] and threading.current_thread() is threading.main_thread()
          and threading.active_count() == 1
          and not any(name.split(".", 1)[0] in {"torch", "numpy", "transformers",
              "safetensors", "tokenizers", "tensorflow", "jax", "cuda", "cupy"}
              for name in sys.modules), "runtime_rejected")


def _state(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_generated(root, rows, check_budget):
    """Closed no-follow traversal of only this attempt's newly generated files."""
    expected_files, expected_dirs = {}, {""}
    for row in rows:
        name = row["path"]
        path = PurePosixPath(name)
        _need(type(name) is str and str(path) == name and not path.is_absolute()
              and ".." not in path.parts and 1 <= len(path.parts) <= 3
              and name not in expected_files and type(row["bytes"]) is int
              and 0 < row["bytes"] <= MAXIMUM_FILE_BYTES, "inventory_rejected")
        expected_files[name] = row
        for depth in range(1, len(path.parts)):
            expected_dirs.add("/".join(path.parts[:depth]))
    _need(not set(expected_files) & expected_dirs, "inventory_rejected")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    found, identities = {}, set()

    def walk(fd, prefix):
        check_budget()
        info = os.fstat(fd)
        _need(stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o700
              and info.st_uid == os.geteuid() and info.st_gid == os.getegid(),
              "inventory_rejected")
        names = os.listdir(fd)
        direct = {name[len(prefix) + bool(prefix):].split("/", 1)[0]
                  for name in set(expected_files) | expected_dirs
                  if name and (not prefix or name.startswith(prefix + "/"))}
        _need(len(names) == len(set(names)) and set(names) == direct, "inventory_rejected")
        for leaf in sorted(names):
            check_budget()
            name = prefix + "/" + leaf if prefix else leaf
            is_directory = name in expected_dirs
            child = os.open(leaf, flags | (os.O_DIRECTORY if is_directory else 0), dir_fd=fd)
            try:
                before = os.fstat(child)
                ident = before.st_dev, before.st_ino
                _need(ident not in identities, "inventory_rejected")
                identities.add(ident)
                if is_directory:
                    walk(child, name)
                else:
                    row = expected_files[name]
                    _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                          and stat.S_IMODE(before.st_mode) == 0o600
                          and before.st_uid == os.geteuid() and before.st_gid == os.getegid()
                          and before.st_size == row["bytes"], "inventory_rejected")
                    chunks, remaining = [], before.st_size
                    while remaining:
                        check_budget()
                        raw = os.read(child, min(65536, remaining))
                        _need(0 < len(raw) <= remaining, "inventory_rejected")
                        chunks.append(raw)
                        remaining -= len(raw)
                    raw = b"".join(chunks)
                    _need(sha(raw) == row["sha256"], "inventory_rejected")
                    found[name] = raw
                _need(_state(before) == _state(os.fstat(child))
                      == _state(os.stat(leaf, dir_fd=fd, follow_symlinks=False)),
                      "inventory_rejected")
            finally:
                os.close(child)

    fd = os.open(root, flags | os.O_DIRECTORY)
    try:
        walk(fd, "")
    finally:
        os.close(fd)
    _need(set(found) == set(expected_files), "inventory_rejected")
    return found


def generate_prepared_inputs(root, *, generation_nonce_sha256):
    """One fixed generation; return private byte maps and a raw-text-free receipt.

    Returned metadata and inputs contain only newly generated synthetic data;
    they must stay private and must not be printed in aggregate-only evidence.
    The receipt is the only public/reportable part. Parent provenance and
    authenticated source admission cannot be inferred from caller arguments.
    """
    global _ATTEMPTED
    _need(not _ATTEMPTED, "already_attempted")
    try:
        _runtime()  # Reject the Mac and unsupported callers before filesystem I/O.
    except BaseException:
        raise PreparedGenerationError("runtime_rejected") from None
    _need(type(root) is type(Path()) and root.is_absolute() and ".." not in root.parts
          and type(generation_nonce_sha256) is str
          and re.fullmatch(r"[a-f0-9]{64}", generation_nonce_sha256)
          and generation_nonce_sha256 != "0" * 64, "arguments_rejected")
    try:
        fixture._root(root)
    except BaseException:
        raise PreparedGenerationError("arguments_rejected") from None
    _ATTEMPTED = True
    deadline = time.monotonic() + MAXIMUM_SECONDS
    expired = False

    def budget():
        nonlocal expired
        if time.monotonic() >= deadline:
            expired = True
            raise PreparedGenerationError("deadline_exhausted") from None

    try:
        budget()
        fixture._write(root / "attempt.started", MARKER)
        checkpoint, tokenizer = root / "checkpoint", root / "tokenizer"
        checkpoint.mkdir(mode=0o700)
        tokenizer.mkdir(mode=0o700)
        fixture._generate(checkpoint, budget)
        metadata = tuple(fixture._metadata(checkpoint, tokenizer, budget))
        _need(len(metadata) == 5 and all(type(raw) is bytes and 0 < len(raw) <= 131072
              for raw in metadata) and sum(map(len, metadata)) <= MAXIMUM_METADATA_BYTES)
        base = json.loads(metadata[0])
        _need(base.get("synthetic_fixture_only") is True
              and "prepared_fixture_marker" not in base)
        plan = build_import_plan(*metadata, expected_base_sha256=sha(metadata[0]),
                                 dtype="bfloat16")
        _need(plan["model_parameters"] == 736, "inventory_rejected")
        rows = {"checkpoint": plan["checkpoint_inventory"],
                "tokenizer": plan["tokenizer_inventory"]}
        count = sum(len(value) for value in rows.values())
        size = sum(row["bytes"] for value in rows.values() for row in value)
        _need(0 < count <= MAXIMUM_INPUT_FILES and 0 < size <= MAXIMUM_INPUT_BYTES,
              "inventory_rejected")
        inputs = {role: _read_generated(root / role, value, budget)
                  for role, value in rows.items()}
        budget()
        receipt = {"schema": SCHEMA, "status": "passed", "synthetic_only": True,
            "generation_nonce_sha256": generation_nonce_sha256, "generation_passes": 1,
            "model_parameters": 736, "generated_input_files": count,
            "generated_input_bytes": size, "metadata_bytes": sum(map(len, metadata)),
            "generated_readback_bytes": size,
            "metadata_sha256": sha(canonical([sha(raw) for raw in metadata])),
            "input_inventory_sha256": sha(canonical(rows)),
            "maximum_seconds": MAXIMUM_SECONDS, "parent_containment_required": True,
            "source_origin_authenticated": False, "host_capacity_verified": False,
            "cumulative_generation_writes_metered": False,
            "real_model_or_data_access": False, "real_4b_import": False,
            "import_passes": 0, "forward_calls": 0, "optimizer_steps": 0,
            "training_authorized": False, "production_eligible": False, "retry_allowed": False}
        return {"metadata": metadata, "generated_inputs": inputs, "receipt": receipt}
    except BaseException as error:
        code = ("deadline_exhausted" if expired else
                error.code if type(error) is PreparedGenerationError else "generation_failed")
        raise PreparedGenerationError(code) from None
