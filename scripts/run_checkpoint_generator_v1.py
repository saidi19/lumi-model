#!/usr/bin/env python3
"""Fixed contained tiny generator; stdout is a PRIVATE one-packet parent channel.

Never execute main on the Mac. The independent parent authenticates this entry,
its immutable source manifest, software and fresh nonce, then installs containment,
closes authority descriptors and drops the worker to uid/gid10001. The entry has
no importer, model input, configuration, host output, cleanup or retry argument.
All code copies and freshly generated files live only in its private /tmp.
Never log the success packet: only the codec's aggregate receipt is reportable.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import json
import os
import re
import stat
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

ENTRY = Path("/opt/lumi-release/guardian/generator_entry.py")
HELPER = Path("/opt/lumi-release/guardian/guarded_import_helpers.py")
HELPER_BYTES = 14356
HELPER_SHA256 = "687a2989ccefdf2466dc77cda62d99d5c6428bc60c74e0e32e15991cec120690"
SOURCE_ROOT = Path("/opt/lumi-release/importer/src")
MANIFEST = Path("/opt/lumi-release/importer/manifest.json")
SCHEMA = "lumi.checkpoint-generator-entry.v0.1"
MAXIMUM_PACKET_BYTES = 65536
MAXIMUM_ENTRY_BYTES = 262144
MAXIMUM_SECONDS = 45
READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
DIR_FLAGS = READ_FLAGS | os.O_DIRECTORY
ML_ROOTS = frozenset({"torch", "numpy", "transformers", "safetensors", "tokenizers",
                      "tensorflow", "jax", "cupy", "cuda"})
FIXED_ENV = {
    "PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "AWS_EC2_METADATA_DISABLED": "true", "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
}
PIN_KEYS = frozenset({"LUMI_GENERATOR_ENTRY_SHA256", "LUMI_IMPORT_SOURCE_MANIFEST_SHA256",
                      "LUMI_GENERATION_NONCE_SHA256"})
REQUIRED_SOURCES = frozenset({
    "lumi/__init__.py", "lumi/posttraining_checkpoint_guard_fixture_v1.py",
    "lumi/posttraining_checkpoint_prepared_generation_v1.py",
    "lumi/posttraining_checkpoint_generator_packet_v1.py",
    "lumi/posttraining_checkpoint_import_plan_v1.py",
    "lumi/posttraining_final_base_metadata_v1.py",
    "lumi/configuration_lumi.py", "lumi/modeling_lumi.py", "lumi/parameter_count.py",
})
ERROR_CODES = frozenset({"runtime_rejected", "entry_binding_rejected",
    "helper_binding_rejected", "source_binding_rejected", "source_staging_failed",
    "generation_failed", "packet_rejected", "deadline_exhausted"})


class GeneratorEntryError(ValueError):
    def __init__(self, code="source_binding_rejected"):
        self.code = (code if type(code) is str and code in ERROR_CODES
                     else "source_binding_rejected")
        super().__init__(self.code)


def need(value, code="source_binding_rejected"):
    if not value:
        raise GeneratorEntryError(code) from None


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def _snapshot(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_uid,
            info.st_gid, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _owned(info, *, directory=False):
    need((stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
         and info.st_uid == info.st_gid == 0 and not stat.S_IMODE(info.st_mode) & 0o022
         and (directory or info.st_nlink == 1))


def root_read(path, *, maximum):
    """Authenticate bounded root-owned entry/helper bytes before helper execution."""
    need(type(path) is type(Path()) and path.is_absolute() and ".." not in path.parts)
    parent, descriptor = os.open("/", DIR_FLAGS), None
    try:
        _owned(os.fstat(parent), directory=True)
        for part in path.parts[1:-1]:
            child = os.open(part, DIR_FLAGS, dir_fd=parent)
            previous, parent = parent, child
            os.close(previous)
            _owned(os.fstat(parent), directory=True)
        descriptor = os.open(path.name, READ_FLAGS, dir_fd=parent)
        before = os.fstat(descriptor)
        _owned(before)
        need(0 < before.st_size <= maximum)
        chunks, remaining = [], before.st_size
        while remaining:
            raw = os.read(descriptor, min(65536, remaining))
            need(type(raw) is bytes and 0 < len(raw) <= remaining)
            chunks.append(raw)
            remaining -= len(raw)
        need(_snapshot(before) == _snapshot(os.fstat(descriptor))
             == _snapshot(os.stat(path.name, dir_fd=parent, follow_symlinks=False)))
        return b"".join(chunks)
    finally:
        try:
            if descriptor is not None:
                os.close(descriptor)
        finally:
            os.close(parent)


def runtime_check(argv, environment):
    need(type(argv) is list and argv == [str(ENTRY), "--synthetic-only"], "runtime_rejected")
    need(sys.platform == "linux" and os.uname().machine == "x86_64"
         and sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12)
         and sys.flags.isolated == sys.flags.ignore_environment == sys.flags.no_user_site == 1
         and sys.dont_write_bytecode and not sys.flags.optimize and sys.byteorder == "little"
         and os.getresuid() == os.getresgid() == (10001, 10001, 10001)
         and os.getgroups() == [] and threading.current_thread() is threading.main_thread()
         and threading.active_count() == 1
         and not any(name.split(".", 1)[0] in ML_ROOTS | {"lumi"} for name in sys.modules),
         "runtime_rejected")
    need(type(environment) is dict and set(environment) == set(FIXED_ENV) | PIN_KEYS
         and all(environment[key] == value for key, value in FIXED_ENV.items())
         and all(type(environment[key]) is str
                 and re.fullmatch(r"[0-9a-f]{64}", environment[key]) for key in PIN_KEYS)
         and environment["LUMI_GENERATION_NONCE_SHA256"] != "0" * 64, "runtime_rejected")


def load_helpers():
    raw = root_read(HELPER, maximum=HELPER_BYTES)
    need(len(raw) == HELPER_BYTES and digest(raw) == HELPER_SHA256, "helper_binding_rejected")
    module = types.ModuleType("_lumi_generator_verified_helpers_v1")
    module.__file__ = str(HELPER)
    exec(compile(raw, str(HELPER), "exec"), module.__dict__)
    return module


def install_package(source):
    """Skip __init__ autoregistration until the fixed generator's runtime gate."""
    need(not any(name.split(".", 1)[0] in ML_ROOTS | {"lumi"} for name in sys.modules),
         "runtime_rejected")
    package = types.ModuleType("lumi")
    package.__package__ = "lumi"
    package.__path__ = [str(source / "lumi")]
    package.__spec__ = importlib.machinery.ModuleSpec("lumi", loader=None, is_package=True)
    package.__spec__.submodule_search_locations = package.__path__
    sys.modules["lumi"] = package
    generation = importlib.import_module("lumi.posttraining_checkpoint_prepared_generation_v1")
    codec = importlib.import_module("lumi.posttraining_checkpoint_generator_packet_v1")
    return generation, codec


def _failure(code):
    return {"schema": SCHEMA, "status": "failed", "code": code, "synthetic_only": True,
            "real_4b_import": False, "training_authorized": False,
            "production_eligible": False, "retry_allowed": False}


def main():
    saved_stdout, packet, success = None, None, False
    failure = _failure("runtime_rejected")
    try:
        environment = dict(os.environ)
        runtime_check(sys.argv, environment)
        deadline = time.monotonic() + MAXIMUM_SECONDS

        def budget():
            need(time.monotonic() < deadline, "deadline_exhausted")

        raw_entry = root_read(ENTRY, maximum=MAXIMUM_ENTRY_BYTES)
        need(digest(raw_entry) == environment["LUMI_GENERATOR_ENTRY_SHA256"],
             "entry_binding_rejected")
        helper = load_helpers()
        try:
            files, source_report = helper.verify_source_tree(
                SOURCE_ROOT, MANIFEST, environment["LUMI_IMPORT_SOURCE_MANIFEST_SHA256"])
            need(REQUIRED_SOURCES <= set(files), "source_binding_rejected")
        except BaseException:
            raise GeneratorEntryError("source_binding_rejected") from None
        budget()
        try:
            source, copy_report = helper.stage_private_sources(Path("/tmp"), files)
        except BaseException:
            raise GeneratorEntryError("source_staging_failed") from None
        del files
        budget()
        saved_stdout = os.dup(1)
        os.dup2(2, 1)  # Parent supplies /dev/null stderr; native prints cannot enter private IPC.
        generation, codec = install_package(source)
        root = Path(tempfile.mkdtemp(prefix="lumi-generated-input-", dir="/tmp"))
        root.chmod(0o700)
        budget()
        try:
            generated = generation.generate_prepared_inputs(root,
                generation_nonce_sha256=environment["LUMI_GENERATION_NONCE_SHA256"])
        except BaseException as error:
            code = ("deadline_exhausted" if type(error) is generation.PreparedGenerationError
                    and error.code == "deadline_exhausted" else "generation_failed")
            raise GeneratorEntryError(code) from None
        budget()
        try:
            packet = codec.encode_generated_packet(generated,
                entry_sha256=environment["LUMI_GENERATOR_ENTRY_SHA256"],
                source_manifest_sha256=environment["LUMI_IMPORT_SOURCE_MANIFEST_SHA256"],
                generation_nonce_sha256=environment["LUMI_GENERATION_NONCE_SHA256"],
                source_report=source_report, copy_report=copy_report)
            need(type(packet) is bytes and 0 < len(packet) <= MAXIMUM_PACKET_BYTES,
                 "packet_rejected")
        except BaseException:
            raise GeneratorEntryError("packet_rejected") from None
        budget()
        success = True
    except GeneratorEntryError as error:
        failure = _failure(error.code)
    except BaseException:
        failure = _failure("generation_failed")
    finally:
        if saved_stdout is not None:
            try:
                os.dup2(saved_stdout, 1)
            finally:
                os.close(saved_stdout)
    try:
        outgoing = packet if success else canonical(failure)
        need(type(outgoing) is bytes and 0 < len(outgoing) <= MAXIMUM_PACKET_BYTES,
             "packet_rejected")
        need(os.write(1, outgoing) == len(outgoing), "packet_rejected")  # One send, never retry.
    except BaseException:
        return 2
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
