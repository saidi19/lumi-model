"""Full-size CPU coordinator core; public execution stays closed pending host binding.

This successor preserves the frozen kernel, package, parser and audit helpers.
Only a future authenticated fixed guardian entry may call the private core,
after independently binding read-only input/source mounts, writable output,
runtime, isolation, resource limits and one-shot authorization. An underscore
is not an authorization boundary: arbitrary Python execution is NOT admitted.
No approval boolean, caller callback or public run_import can bypass the closed
entry. The private core requires the Linux isolated uid/gid10001 worker runtime.
Local tests replace that check and use generated inputs with kernel doubles.

The private core can perform real payload I/O when eventually integrated.
It is not invoked on real inputs in this development step. Logical content
accounting does not cover dependency/module imports or physical device I/O.
The cooperative clock cannot interrupt a blocked native call; the independent
guardian must enforce deadlines, RAM, write policy and cleanup. Completion
requires both the returned outcome and the parent receipt. No native host or
full-size performance claim is made here; old versions are never modified.
"""

from __future__ import annotations

import hashlib
import math
import os
import stat
import sys
import threading
import time
from pathlib import Path

from . import posttraining_checkpoint_import_runner_v2 as frozen
from .posttraining_checkpoint_resource_envelope_v1 import validate_resource_envelope
from .posttraining_final_base_metadata_v1 import canonical, json_object, sha

SCHEMA = "lumi.sft-checkpoint-import-runner.v0.3"
STARTED = "import-started-v0.5.json"
COMPLETE = "import-complete-v0.5.json"
FAILED = "import-failed-v0.5.json"
ERROR_CODES = frozen.ERROR_CODES | frozenset({
    "runtime_rejected", "resource_envelope_rejected", "source_binding_rejected",
})
_path = frozen._path
_snapshot = frozen._snapshot
_physical_overlap = frozen._physical_overlap
_write_new = frozen._write_new
_audit = frozen._audit
_identities = frozen._identities
_source_snapshot = frozen._source_snapshot
_kernel_audit = frozen._kernel_audit


class ImportRunnerError(ValueError):
    def __init__(self, code):
        self.code = code if type(code) is str and code in ERROR_CODES else "runner_failed"
        super().__init__(self.code)


def _need(value, code):
    if not value:
        raise ImportRunnerError(code) from None


def run_import(**_arguments):
    """No public full-size launch until a separately verified guardian is wired."""
    raise ImportRunnerError("host_guard_not_bound") from None


def _runtime():
    _need(sys.platform == "linux" and os.uname().machine == "x86_64"
          and sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12)
          and sys.flags.isolated == 1 and sys.flags.ignore_environment == 1
          and sys.flags.no_user_site == 1 and sys.dont_write_bytecode
          and not sys.flags.optimize and sys.byteorder == "little"
          and os.getresuid() == os.getresgid() == (10001, 10001, 10001)
          and os.getgroups() == [] and threading.current_thread() is threading.main_thread()
          and threading.active_count() == 1
          and not any(name.split(".")[0] in {"torch", "numpy", "transformers", "safetensors",
                                             "tokenizers", "jax", "tensorflow", "cupy"}
                      for name in sys.modules), "runtime_rejected")


class _Ledger(frozen._Ledger):
    def __init__(self, limits, *, started):
        # Only the independently validated full-size envelope constructs these.
        _need(type(limits) is dict and set(limits) == frozen.LIMIT_FIELDS, "invalid_arguments")
        _need(type(limits["maximum_seconds"]) is int
              and 0 < limits["maximum_seconds"] <= 86400
              and type(started) in {float, int} and math.isfinite(started), "invalid_arguments")
        _need(all(type(limits[k]) is int and 0 < limits[k] <= 2**63 - 1
                  for k in frozen.LIMIT_FIELDS - {"maximum_seconds"})
              and limits["maximum_control_bytes"] <= 131072
              and limits["maximum_package_write_bytes"] <= limits["maximum_private_write_bytes"],
              "invalid_arguments")
        self.limits = dict(limits)
        self.deadline = started + limits["maximum_seconds"]
        self.reads = self.writes = 0
        self.failed = False


def _resource_inputs(metadata, expected_base_sha256, plan_raw, expected_plan_file_sha256,
                     envelope_raw, expected_envelope_file_sha256, kernel_io_contract,
                     expected_kernel_io_contract_sha256):
    try:
        _need(type(metadata) is tuple and len(metadata) == 5
              and all(type(raw) is bytes for raw in metadata), "metadata_rejected")
        admission = validate_resource_envelope(
            plan_raw, envelope_raw, expected_plan_file_sha256=expected_plan_file_sha256,
            expected_envelope_file_sha256=expected_envelope_file_sha256)
        plan = frozen.build_import_plan(*metadata, expected_base_sha256=expected_base_sha256,
                                         dtype="bfloat16")
        _need(canonical(plan) == canonical(json_object(plan_raw)), "metadata_rejected")
        envelope = admission["envelope"]
        _need(len(metadata[3]) == envelope["config_bytes"]
              and sha(metadata[3]) == envelope["source_config_sha256"], "metadata_rejected")
        contract = frozen.validate_kernel_io_contract(
            plan, kernel_io_contract, expected_kernel_io_contract_sha256)
        dcp, hf = contract["checkpoint_read_contract"], contract["hf_io_contract"]
        _need(dcp["maximum_read_bytes"] == envelope["dcp_read_cap_bytes"]
              and hf["maximum_read_bytes"] == envelope["hf_read_cap_bytes"]
              and hf["maximum_write_bytes"] == envelope["hf_write_cap_bytes"]
              and hf["maximum_control_bytes"] == min(65536, envelope["control_bytes"]),
              "resource_envelope_rejected")
        return plan, envelope, admission["runner_limits"], contract
    except ImportRunnerError:
        raise
    except Exception:
        raise ImportRunnerError("resource_envelope_rejected") from None


def _runtime_source_preflight(rows, ledger):
    """Hash only the two bounded fixed source modules; these are not model inputs."""
    pins = {}
    for row in rows:
        name = row["path"]
        _need(name in {"configuration_lumi.py", "modeling_lumi.py"}
              and name not in pins, "source_binding_rejected")
        path = Path(__file__).with_name(name)
        _need(path.is_absolute() and ".." not in path.parts
              and not any(p.is_symlink() for p in (path, *path.parents)),
              "source_binding_rejected")
        before = path.lstat()
        # Preserve frozen HF read_control's source-code modes; data remain 0600.
        _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
              and stat.S_IMODE(before.st_mode) in {0o400, 0o600, 0o644}
              and before.st_uid == os.geteuid() and before.st_gid == os.getegid()
              and before.st_size == row["bytes"],
              "source_binding_rejected")
        ledger.reserve(reads=row["bytes"])
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
        try:
            _need(_snapshot(os.fstat(descriptor)) == _snapshot(before), "source_binding_rejected")
            raw = os.read(descriptor, row["bytes"])
            _need(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"]
                  and _snapshot(os.fstat(descriptor)) == _snapshot(before)
                  and _snapshot(path.lstat()) == _snapshot(before), "source_binding_rejected")
        finally:
            os.close(descriptor)
        pins[name] = row["sha256"]
    _need(set(pins) == {"configuration_lumi.py", "modeling_lumi.py"}, "source_binding_rejected")
    ledger.tick()
    return pins, sum(row["bytes"] for row in rows)


def _accounting(plan, contract, kernel, package, ledger, preflight_bytes):
    result = frozen._accounting(plan, contract, kernel, package, ledger)
    result["runtime_source_preflight_read_bytes"] = preflight_bytes
    result["successful_content_read_bytes_requested"] += preflight_bytes
    result["unspent_read_envelope_bytes_not_refunded"] -= preflight_bytes
    _need(result["unspent_read_envelope_bytes_not_refunded"] >= 0, "runner_failed")
    return result


def _coordinate_admitted_import(*, metadata, expected_base_sha256, plan_raw,
                               expected_plan_file_sha256, envelope_raw,
                               expected_envelope_file_sha256, checkpoint_root, tokenizer_root,
                               attempt, kernel_io_contract, expected_kernel_io_contract_sha256):
    """Private future guardian target, NOT authorization or a user-facing entrypoint."""
    started, stage, created = time.monotonic(), "created", False
    ledger, terminal_reserved = None, False
    try:
        _runtime()  # Must precede control parsing, file I/O and any module callbacks.
        plan, envelope, limits, contract = _resource_inputs(
            metadata, expected_base_sha256, plan_raw, expected_plan_file_sha256,
            envelope_raw, expected_envelope_file_sha256, kernel_io_contract,
            expected_kernel_io_contract_sha256)
        ledger = _Ledger(limits, started=started)
        limits = ledger.limits
        bounds, dtype = plan["resource_bounds"], plan["dtype"]
        dcp_contract, hf_contract = (contract["checkpoint_read_contract"],
                                     contract["hf_io_contract"])
        source_pins, source_preflight_bytes = _runtime_source_preflight(
            envelope["runtime_sources"], ledger)
        _path(checkpoint_root)
        _path(tokenizer_root)
        _path(attempt, absent=True)
        paths = (checkpoint_root, tokenizer_root, attempt)
        _need(all(not a.is_relative_to(b) and not b.is_relative_to(a)
                  for i, a in enumerate(paths) for b in paths[i + 1:]), "input_output_overlap")
        _physical_overlap((checkpoint_root, tokenizer_root), attempt)
        ledger.tick()
        parent = attempt.parent
        _path(parent)
        parent_before = parent.stat()
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            _need(_snapshot(os.fstat(parent_fd)) == _snapshot(parent_before), "path_rejected")
            os.mkdir(attempt.name, 0o700, dir_fd=parent_fd)
            created = True
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        # Prepay both terminal envelopes; a failure can be durably recorded even
        # after the deadline/read budget is exhausted, with no extra authorization.
        ledger.reserve(writes=2 * limits["maximum_control_bytes"])
        terminal_reserved = True
        marker_raw = canonical({"schema_version": SCHEMA, "stage": "created",
                                "plan_sha256": plan["plan_sha256"],
                                "base_file_sha256": expected_base_sha256,
                                "kernel_io_contract_sha256": contract["contract_sha256"],
                                "resource_envelope_sha256": expected_envelope_file_sha256,
                                "scope": "private_import_under_external_guard",
                                "retry_allowed": False,
                                "training_authorized": False, "production_eligible": False}) + b"\n"
        _write_new(attempt, STARTED, marker_raw, ledger)
        marker_before = (attempt / STARTED).lstat()
        _write_new(attempt, "source-config.json", metadata[3], ledger)
        config_before = (attempt / "source-config.json").lstat()
        stage = "inputs_before"
        before_tokens = _audit(tokenizer_root, plan["tokenizer_inventory"], ledger)
        token_identities = _identities(tokenizer_root, plan["tokenizer_inventory"])
        source_identities = _source_snapshot(checkpoint_root, plan, ledger)
        stage = "convert"
        ledger.reserve(reads=dcp_contract["maximum_read_bytes"] + hf_contract["maximum_read_bytes"],
                       writes=hf_contract["maximum_write_bytes"])
        from .posttraining_checkpoint_import_torch_v5 import convert_and_verify
        try:
            kernel = convert_and_verify(
                attempt / "source-config.json", checkpoint_root, attempt / "checkpoint", dtype,
                maximum_parameters=plan["model_parameters"],
                maximum_tensor_bytes=bounds["maximum_projected_live_tensor_bytes"],
                maximum_output_bytes=bounds["maximum_backend_output_bytes"],
                check_budget=ledger.tick, checkpoint_read_contract=dcp_contract,
                hf_io_contract=hf_contract,
            )
        except BaseException:
            raise ImportRunnerError("kernel_failed") from None
        _need(_source_snapshot(checkpoint_root, plan, ledger) == source_identities,
              "input_audit_failed")
        _kernel_audit(kernel, plan, contract)
        _need(kernel.get("runtime_code_sha256") == source_pins, "source_binding_rejected")
        ledger.tick()
        stage = "package"
        ledger.reserve(reads=limits["maximum_package_read_bytes"],
                       writes=limits["maximum_package_write_bytes"])
        from .posttraining_checkpoint_package_v2 import finalize_package
        try:
            package = finalize_package(
                output=attempt / "checkpoint", tokenizer_root=tokenizer_root,
                plan=plan, expected_plan_sha256=plan["plan_sha256"], kernel_evidence=kernel,
                kernel_io_contract=contract,
                expected_kernel_io_contract_sha256=contract["contract_sha256"],
                maximum_read_bytes=limits["maximum_package_read_bytes"],
                maximum_write_bytes=limits["maximum_package_write_bytes"],
                maximum_control_bytes=limits["maximum_control_bytes"], check_budget=ledger.tick,
            )
        except BaseException:
            raise ImportRunnerError("package_failed") from None
        stage = "inputs_after"
        _audit(checkpoint_root, plan["checkpoint_inventory"], ledger)
        _need(_audit(tokenizer_root, plan["tokenizer_inventory"], ledger) == before_tokens,
              "input_audit_failed")
        _need(_source_snapshot(checkpoint_root, plan, ledger) == source_identities
              and _identities(tokenizer_root, plan["tokenizer_inventory"]) == token_identities,
              "input_audit_failed")
        _need(_snapshot((attempt / STARTED).lstat()) == _snapshot(marker_before)
              and _snapshot((attempt / "source-config.json").lstat()) == _snapshot(config_before),
              "output_changed")
        ledger.tick()
        stage = "complete"
        result = {
            "schema_version": SCHEMA,
            "status": "PASS_import_coordinator_requires_parent_verification",
            "stage": stage, "plan_sha256": plan["plan_sha256"],
            "kernel_evidence_sha256": sha(canonical(kernel)), "package": package,
            "kernel_io_contract_sha256": contract["contract_sha256"],
            "resource_envelope_sha256": expected_envelope_file_sha256,
            "content_accounting": _accounting(plan, contract, kernel, package, ledger,
                                               source_preflight_bytes),
            "full_input_audits_completed": 2, "observed_input_identities_unchanged": True,
            "atomic_input_immutability_verified": False,
            "coordinator_read_bytes_reserved": ledger.reads,
            "private_write_bytes_reserved": ledger.writes,
            "reservations_refunded": False, "retry_allowed": False,
            "native_backend_read_bytes_metered": False,
            "explicit_checkpoint_and_HF_content_metered": True,
            "dependency_module_reads_metered": False,
            "durable_completion_is_execution_authority": False,
            "success_requires_returned_outcome_and_parent_supervisor": True,
            "os_memory_write_limits_enforced": False, "hard_timeout_enforced_here": False,
            "real_4b_import": False, "training_authorized": False, "production_eligible": False,
        }
        ledger.tick()
        _write_new(attempt, COMPLETE, canonical(result) + b"\n", ledger, prepaid=True)
        ledger.tick()
        return result
    except BaseException as error:
        code = error.code if type(error) in {ImportRunnerError, frozen.ImportRunnerError} \
            else "runner_failed"
        if created and ledger is not None and terminal_reserved:
            try:
                _write_new(attempt, FAILED, canonical({"schema_version": SCHEMA, "stage": stage,
                    "code": code, "retry_allowed": False, "training_authorized": False,
                    "production_eligible": False}) + b"\n", ledger, prepaid=True)
            except BaseException:
                pass  # Preserve the consumed directory/marker even if failure recording fails.
        raise ImportRunnerError(code) from None
