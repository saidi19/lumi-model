#!/usr/bin/env python3
"""Fixed synthetic-only entry for a future separately supervised Linux worker.

Never invoke main on the Mac. The parent must pin this entry, the immutable
root-owned source manifest and a freshly generated root-owned control bundle,
then supply read-only generated inputs and a private writable output mount.
This entry neither generates tensors nor admits an arbitrary model as a fixture.
Its fixed tiny geometry, fresh base marker, closed controls and hard ceilings
are additional checks, not a cryptographic proof that a caller generated data.
Authenticity of generation remains the trusted parent's responsibility.

The public real-input runner stays closed. No retry, cleanup, networking, host
provisioning, mount creation, real checkpoint import or training API is added.
The only executable operation is one private coordinator call after all gates.
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
import threading
import time
import types
from pathlib import Path

ENTRY = Path("/opt/lumi-release/guardian/prepared_fixture_entry.py")
HELPER = Path("/opt/lumi-release/guardian/guarded_import_helpers.py")
HELPER_BYTES = 14356
HELPER_SHA256 = "687a2989ccefdf2466dc77cda62d99d5c6428bc60c74e0e32e15991cec120690"
SOURCE_ROOT = Path("/opt/lumi-release/importer/src")
SOURCE_MANIFEST = Path("/opt/lumi-release/importer/manifest.json")
CONTROL_ROOT = Path("/opt/lumi-release/prepared-fixture-control")
CONTROL = CONTROL_ROOT / "control.json"
CHECKPOINT = Path("/inputs/checkpoint")
TOKENIZER = Path("/inputs/tokenizer")
OUTPUT = Path("/outputs")
IMPORT_PARENT = OUTPUT / "import"
ATTEMPT = IMPORT_PARENT / "attempt-001"
CONTROL_SCHEMA = "lumi.prepared-checkpoint-fixture-control.v0.1"
BASE_MARKER_SCHEMA = "lumi.prepared-checkpoint-fixture-base.v0.1"
REPORT_SCHEMA = "lumi.prepared-checkpoint-fixture-entry.v0.1"
METADATA_NAMES = ("base.json", "manifest.sha256", "state.json", "config.json", "handoff.json")
CONTROL_NAMES = METADATA_NAMES + ("plan.json", "envelope.json", "kernel-io.json")
MAXIMUM_CONTROL_BYTES = 131072
MAXIMUM_BUNDLE_BYTES = 524288
MAXIMUM_REPORT_BYTES = 8192
MAXIMUM_PARAMETERS = 1_000_000
MAXIMUM_INPUT_BYTES = 4 * 1024**2
MAXIMUM_INPUT_FILES = 32
MAXIMUM_READ_BYTES = 64 * 1024**2
MAXIMUM_WRITE_BYTES = 16 * 1024**2
MAXIMUM_SECONDS = 60
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
PIN_KEYS = frozenset({"LUMI_PREPARED_FIXTURE_ENTRY_SHA256",
                      "LUMI_IMPORT_SOURCE_MANIFEST_SHA256",
                      "LUMI_PREPARED_FIXTURE_CONTROL_SHA256"})
REQUIRED_SOURCES = frozenset({
    "lumi/__init__.py", "lumi/posttraining_checkpoint_guard_fixture_v1.py",
    "lumi/posttraining_checkpoint_import_runner_v3.py",
    "lumi/posttraining_checkpoint_import_runner_v2.py",
    "lumi/posttraining_checkpoint_resource_envelope_v1.py",
    "lumi/posttraining_checkpoint_host_contract_v1.py",
    "lumi/posttraining_checkpoint_import_plan_v1.py",
    "lumi/posttraining_final_base_metadata_v1.py",
    "lumi/posttraining_checkpoint_io_contract_v1.py",
    "lumi/posttraining_checkpoint_import_torch_v5.py",
    "lumi/posttraining_checkpoint_package_v2.py",
    "lumi/configuration_lumi.py", "lumi/modeling_lumi.py",
})
CONFIG = {
    "vocab_size": 17, "hidden_size": 8, "intermediate_size": 16,
    "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
    "head_dim": 4, "max_position_embeddings": 16, "rope_theta": 10000.0,
    "rms_norm_eps": 0.00001, "attention_bias": False, "tie_word_embeddings": True,
    "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2, "doc_token_id": 3,
    "model_type": "lumi",
}
ERROR_CODES = frozenset({"runtime_rejected", "entry_binding_rejected",
    "helper_binding_rejected", "source_binding_rejected", "control_binding_rejected",
    "synthetic_scope_rejected", "resource_contract_rejected", "output_rejected",
    "source_staging_failed", "import_failed", "result_rejected", "deadline_exhausted"})


class PreparedFixtureError(ValueError):
    def __init__(self, code="control_binding_rejected"):
        self.code = (code if type(code) is str and code in ERROR_CODES
                     else "control_binding_rejected")
        super().__init__(self.code)


def need(value, code="control_binding_rejected"):
    if not value:
        raise PreparedFixtureError(code) from None


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pin(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def decode(raw):
    def pairs(items):
        value = {}
        for key, child in items:
            need(key not in value)
            value[key] = child
        return value

    def constant(_value):
        raise PreparedFixtureError()

    try:
        need(type(raw) is bytes and 0 < len(raw) <= MAXIMUM_CONTROL_BYTES)
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
        need(type(value) is dict)
        return value
    except PreparedFixtureError:
        raise
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise PreparedFixtureError() from None


def _snapshot(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_uid,
            info.st_gid, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _root_owned(info, *, directory=False):
    need((stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
         and info.st_uid == info.st_gid == 0 and not stat.S_IMODE(info.st_mode) & 0o022
         and (directory or info.st_nlink == 1))


def root_read(path, *, maximum):
    """Read only bounded root-owned controls, all ancestors opened without symlinks."""
    need(type(path) is type(Path()) and path.is_absolute() and ".." not in path.parts)
    parent, fd = os.open("/", DIR_FLAGS), None
    try:
        _root_owned(os.fstat(parent), directory=True)
        for component in path.parts[1:-1]:
            child = os.open(component, DIR_FLAGS, dir_fd=parent)
            previous, parent = parent, child
            os.close(previous)
            _root_owned(os.fstat(parent), directory=True)
        fd = os.open(path.name, READ_FLAGS, dir_fd=parent)
        before = os.fstat(fd)
        _root_owned(before)
        need(0 < before.st_size <= maximum)
        chunks, remaining = [], before.st_size
        while remaining:
            raw = os.read(fd, min(65536, remaining))
            need(type(raw) is bytes and 0 < len(raw) <= remaining)
            chunks.append(raw)
            remaining -= len(raw)
        need(_snapshot(os.fstat(fd)) == _snapshot(before)
             and _snapshot(os.stat(path.name, dir_fd=parent, follow_symlinks=False))
                 == _snapshot(before))
        return b"".join(chunks)
    finally:
        try:
            if fd is not None:
                os.close(fd)
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
         and all(pin(environment[key]) for key in PIN_KEYS), "runtime_rejected")


def load_helpers():
    raw = root_read(HELPER, maximum=HELPER_BYTES)
    need(len(raw) == HELPER_BYTES and digest(raw) == HELPER_SHA256, "helper_binding_rejected")
    module = types.ModuleType("_lumi_prepared_fixture_verified_helpers_v1")
    module.__file__ = str(HELPER)
    exec(compile(raw, str(HELPER), "exec"), module.__dict__)
    return module


def validate_control(raw, *, expected_sha256, entry_sha256, source_manifest_sha256):
    """Pure root-control binding. A caller-supplied approval flag is not accepted."""
    need(type(raw) is bytes and 0 < len(raw) <= MAXIMUM_CONTROL_BYTES
         and pin(expected_sha256) and digest(raw) == expected_sha256)
    value = decode(raw)
    need(set(value) == {"schema", "scope", "generation_nonce_sha256", "entry_sha256",
                        "source_manifest_sha256", "files", "limits"}
         and value["schema"] == CONTROL_SCHEMA
         and value["scope"] == "generated_tiny_checkpoint_only"
         and pin(value["generation_nonce_sha256"])
         and value["generation_nonce_sha256"] != "0" * 64
         and pin(entry_sha256) and value["entry_sha256"] == entry_sha256
         and pin(source_manifest_sha256)
         and value["source_manifest_sha256"] == source_manifest_sha256
         and canonical(value) == raw)
    need(type(value["limits"]) is dict and canonical(value["limits"]) == canonical({
        "maximum_parameters": MAXIMUM_PARAMETERS, "maximum_input_bytes": MAXIMUM_INPUT_BYTES}))
    need(type(value["files"]) is list and len(value["files"]) == len(CONTROL_NAMES))
    for row, name in zip(value["files"], CONTROL_NAMES, strict=True):
        need(type(row) is dict and set(row) == {"path", "bytes", "sha256"}
             and row["path"] == name and type(row["bytes"]) is int
             and 0 < row["bytes"] <= MAXIMUM_CONTROL_BYTES and pin(row["sha256"]))
    need(sum(row["bytes"] for row in value["files"]) <= MAXIMUM_BUNDLE_BYTES)
    return value


def validate_bundle(control, bodies):
    """Authenticate all controls and tiny bounds before source staging or Lumi import."""
    need(type(bodies) is dict and set(bodies) == set(CONTROL_NAMES))
    for row in control["files"]:
        raw = bodies[row["path"]]
        need(type(raw) is bytes and len(raw) == row["bytes"] and digest(raw) == row["sha256"])
    base, plan, envelope = (decode(bodies[name])
                           for name in ("base.json", "plan.json", "envelope.json"))
    marker = {"schema": BASE_MARKER_SCHEMA, "generator": "fixed-tiny-config-v1",
              "generation_nonce_sha256": control["generation_nonce_sha256"]}
    need(base.get("synthetic_fixture_only") is True
         and type(base.get("prepared_fixture_marker")) is dict
         and canonical(base["prepared_fixture_marker"]) == canonical(marker)
         and bodies["config.json"] == canonical(CONFIG), "synthetic_scope_rejected")
    need(type(plan.get("model_parameters")) is int
         and 0 < plan["model_parameters"] <= MAXIMUM_PARAMETERS, "synthetic_scope_rejected")
    total = file_count = 0
    for name in ("checkpoint_inventory", "tokenizer_inventory"):
        rows = plan.get(name)
        need(type(rows) is list and 0 < len(rows) <= MAXIMUM_INPUT_FILES,
             "synthetic_scope_rejected")
        file_count += len(rows)
        need(file_count <= MAXIMUM_INPUT_FILES, "synthetic_scope_rejected")
        for row in rows:
            need(type(row) is dict and type(row.get("bytes")) is int
                 and 0 < row["bytes"] <= MAXIMUM_INPUT_BYTES, "synthetic_scope_rejected")
            total += row["bytes"]
            need(total <= MAXIMUM_INPUT_BYTES, "synthetic_scope_rejected")
    for key, cap in (("maximum_seconds", MAXIMUM_SECONDS),
                     ("coordinator_read_cap_bytes", MAXIMUM_READ_BYTES),
                     ("private_write_cap_bytes", MAXIMUM_WRITE_BYTES)):
        need(type(envelope.get(key)) is int and 0 < envelope[key] <= cap,
             "synthetic_scope_rejected")
    contract = decode(bodies["kernel-io.json"])
    need(pin(contract.get("contract_sha256")))
    return {"metadata": tuple(bodies[name] for name in METADATA_NAMES),
            "expected_base_sha256": digest(bodies["base.json"]),
            "plan_raw": bodies["plan.json"],
            "expected_plan_file_sha256": digest(bodies["plan.json"]),
            "envelope_raw": bodies["envelope.json"],
            "expected_envelope_file_sha256": digest(bodies["envelope.json"]),
            "kernel_io_contract": contract,
            "expected_kernel_io_contract_sha256": contract["contract_sha256"]}


def install_package(source):
    """Avoid the frozen __init__ ML autoregistration before runner runtime gates.

Frozen kernel absolute imports need the exact `lumi` namespace. Only its pinned
staged tree is searched; __init__ bytes remain authenticated but are not executed.
No user callback or replacement source body is installed.
"""
    need(not any(name.split(".", 1)[0] in ML_ROOTS | {"lumi"} for name in sys.modules),
         "runtime_rejected")
    package = types.ModuleType("lumi")
    package.__package__ = "lumi"
    package.__path__ = [str(source / "lumi")]
    package.__spec__ = importlib.machinery.ModuleSpec("lumi", loader=None, is_package=True)
    package.__spec__.submodule_search_locations = package.__path__
    sys.modules["lumi"] = package
    return importlib.import_module("lumi.posttraining_checkpoint_import_runner_v3")


def validate_runner_controls(runner, arguments, files):
    """Rebuild the plan and resource algebra, still without reading model inputs."""
    try:
        plan, envelope, _limits, _contract = runner._resource_inputs(**arguments)
        for row in envelope["runtime_sources"]:
            raw = files["lumi/" + row["path"]]
            need(len(raw) == row["bytes"] and digest(raw) == row["sha256"],
                 "resource_contract_rejected")
        need(plan["model_parameters"] == 736, "synthetic_scope_rejected")
        return plan
    except PreparedFixtureError:
        raise
    except BaseException:
        raise PreparedFixtureError("resource_contract_rejected") from None


def _output_check(helper):
    fd = helper.open_directory(OUTPUT)
    try:
        info = os.fstat(fd)
        need(stat.S_ISDIR(info.st_mode) and info.st_uid == info.st_gid == 10001
             and stat.S_IMODE(info.st_mode) == 0o700, "output_rejected")
        with os.scandir(fd) as entries:
            need(next(entries, None) is None, "output_rejected")
    finally:
        os.close(fd)


def prepare_import_parent(helper):
    """Create only one new fixed private directory; never reuse, chmod or remove it.

    The frozen coordinator requires both attempt and parent to have at least
    three path components. /outputs itself is therefore not a valid parent.
    Code copies remain separate siblings under the output mount.
    """
    parent = created = None
    try:
        parent = helper.open_directory(OUTPUT)
        info = os.fstat(parent)
        need(stat.S_ISDIR(info.st_mode) and info.st_uid == info.st_gid == 10001
             and stat.S_IMODE(info.st_mode) == 0o700, "output_rejected")
        os.mkdir("import", 0o700, dir_fd=parent)  # Exclusive: existing means consumed.
        created = os.open("import", DIR_FLAGS, dir_fd=parent)
        info = os.fstat(created)
        need(stat.S_ISDIR(info.st_mode) and info.st_uid == info.st_gid == 10001
             and stat.S_IMODE(info.st_mode) == 0o700, "output_rejected")
        need(_snapshot(os.stat("import", dir_fd=parent, follow_symlinks=False))
             == _snapshot(info), "output_rejected")
        os.fsync(created)
        os.fsync(parent)
    except BaseException:
        raise PreparedFixtureError("output_rejected") from None
    finally:
        try:
            if created is not None:
                os.close(created)
        finally:
            if parent is not None:
                os.close(parent)


def _failure(code):
    return {"schema": REPORT_SCHEMA, "status": "failed", "code": code,
            "synthetic_only": True, "real_4b_import": False, "training_authorized": False,
            "production_eligible": False, "retry_allowed": False}


def validate_result(result, plan, arguments, *, runner_schema):
    """No exception, path or arbitrary string can enter a retained count field."""
    need(type(result) is dict and result.get("schema_version") == runner_schema
         and result.get("status") == "PASS_import_coordinator_requires_parent_verification"
         and result.get("stage") == "complete"
         and type(result.get("full_input_audits_completed")) is int
         and result["full_input_audits_completed"] == 2
         and result.get("plan_sha256") == plan["plan_sha256"]
         and result.get("resource_envelope_sha256") == arguments["expected_envelope_file_sha256"]
         and result.get("kernel_io_contract_sha256")
             == arguments["expected_kernel_io_contract_sha256"]
         and all(result.get(key) is False for key in
                 ("real_4b_import", "training_authorized", "production_eligible", "retry_allowed")),
         "result_rejected")
    for key, cap in (("coordinator_read_bytes_reserved", MAXIMUM_READ_BYTES),
                     ("private_write_bytes_reserved", MAXIMUM_WRITE_BYTES)):
        need(type(result.get(key)) is int and 0 < result[key] <= cap, "result_rejected")


def main():
    report, saved_stdout, success = _failure("runtime_rejected"), None, False
    try:
        environment = dict(os.environ)
        runtime_check(sys.argv, environment)
        deadline = time.monotonic() + MAXIMUM_SECONDS

        def budget():
            need(time.monotonic() < deadline, "deadline_exhausted")

        entry = root_read(ENTRY, maximum=262144)
        need(digest(entry) == environment["LUMI_PREPARED_FIXTURE_ENTRY_SHA256"],
             "entry_binding_rejected")
        helper = load_helpers()
        raw = root_read(CONTROL, maximum=MAXIMUM_CONTROL_BYTES)
        control = validate_control(raw,
            expected_sha256=environment["LUMI_PREPARED_FIXTURE_CONTROL_SHA256"],
            entry_sha256=environment["LUMI_PREPARED_FIXTURE_ENTRY_SHA256"],
            source_manifest_sha256=environment["LUMI_IMPORT_SOURCE_MANIFEST_SHA256"])
        bodies = {row["path"]: root_read(CONTROL_ROOT / row["path"], maximum=row["bytes"])
                  for row in control["files"]}
        arguments = validate_bundle(control, bodies)
        budget()
        try:
            files, source_report = helper.verify_source_tree(SOURCE_ROOT, SOURCE_MANIFEST,
                environment["LUMI_IMPORT_SOURCE_MANIFEST_SHA256"])
            need(REQUIRED_SOURCES <= set(files), "source_binding_rejected")
        except BaseException:
            raise PreparedFixtureError("source_binding_rejected") from None
        _output_check(helper)
        prepare_import_parent(helper)
        try:
            source, copy_report = helper.stage_private_sources(OUTPUT, files)
        except BaseException:
            raise PreparedFixtureError("source_staging_failed") from None
        budget()
        saved_stdout = os.dup(1)
        os.dup2(2, 1)  # Trusted parent must provide /dev/null stderr.
        runner = install_package(source)
        plan = validate_runner_controls(runner, arguments, files)
        del files
        budget()
        try:
            result = runner._coordinate_admitted_import(**arguments, checkpoint_root=CHECKPOINT,
                tokenizer_root=TOKENIZER, attempt=ATTEMPT)
        except BaseException:
            raise PreparedFixtureError("import_failed") from None
        budget()
        validate_result(result, plan, arguments, runner_schema=runner.SCHEMA)
        report = {"schema": REPORT_SCHEMA, "status": "passed", "code": "ok",
            "entry_sha256": environment["LUMI_PREPARED_FIXTURE_ENTRY_SHA256"],
            "control_sha256": environment["LUMI_PREPARED_FIXTURE_CONTROL_SHA256"],
            "generation_nonce_sha256": control["generation_nonce_sha256"],
            "model_parameters": plan["model_parameters"], "coordinator_passes": 1,
            "generated_input_bytes": sum(row["bytes"] for name in
                ("checkpoint_inventory", "tokenizer_inventory") for row in plan[name]),
            "generated_input_files": sum(len(plan[name]) for name in
                ("checkpoint_inventory", "tokenizer_inventory")),
            "checkpoint_full_audits": 2, "tokenizer_full_audits": 2,
            "coordinator_read_bytes_reserved": result["coordinator_read_bytes_reserved"],
            "private_write_bytes_reserved": result["private_write_bytes_reserved"],
            "coordinator_result_sha256": digest(canonical(result)),
            "synthetic_only": True, "real_4b_import": False, "training_authorized": False,
            "production_eligible": False, "retry_allowed": False,
            "parent_supervisor_verification_required": True,
            "resource_envelope_is_planning_only": True, "host_capacity_verified": False,
            "source_immutability_verified": False, "physical_device_io_measured": False,
            "forward_calls": 0, "optimizer_steps": 0, **source_report, **copy_report}
        success = True
    except PreparedFixtureError as error:
        report = _failure(error.code)
    except BaseException:
        report = _failure("import_failed")
    finally:
        if saved_stdout is not None:
            try:
                os.dup2(saved_stdout, 1)
            finally:
                os.close(saved_stdout)
    try:
        raw = canonical(report) + b"\n"
        need(len(raw) <= MAXIMUM_REPORT_BYTES, "result_rejected")
        need(os.write(1, raw) == len(raw), "result_rejected")
    except BaseException:
        return 2
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
