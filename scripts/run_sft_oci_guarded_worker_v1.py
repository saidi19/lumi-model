#!/usr/bin/env python3
"""Native guarded CPU bridge; NEVER import or execute this wrapper on the Mac.

The accepted image sources stay unchanged. This separately pinned adapter and
admission control are the only additional files. The trusted outer guardian is
responsible for the actual rootfs, process, cgroup and lifecycle boundaries.
Only the six existing synthetic workers import ML. No real training is admitted.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

SCHEMA = "lumi.sft-guarded-oci-worker.v0.1"
RECEIPT_SCHEMA = "lumi.sft-guarded-oci-worker-receipt.v0.1"
CONTROL_SCHEMA = "lumi.sft-guarded-oci-worker-admission.v0.1"
IDENTITY_SCHEMA = "lumi.sft-accepted-oci-software-identity.v0.1"
PACKET_SCHEMA = "lumi.sft-guarded-oci-rank-proof.v0.1"
PACKET_FIELD = "guardian_admission"
ADAPTER_PATH = "/opt/lumi-release/guardian/worker_entry.py"
CONTROL_PATH = "/opt/lumi-release/guardian/admission.json"
PROBE_PATH = "/opt/lumi-release/source/scripts/run_sft_oci_worker_probe_v2.py"
PROBE_SHA256 = "b45b04fc9cb93dd2301df0444f2db95666ba4f7a32e8f4d8e28c5b4077314450"
PROBE_SIZE = 22638
PROBE_NAME = "_lumi_guarded_accepted_probe_v2"
RUNTIME = "/usr/local/bin/python3.12"
MAXIMUM_CONTROL_BYTES = 4096
MAXIMUM_ADAPTER_BYTES = 65536
MAXIMUM_REPORT_BYTES = 32768
MAXIMUM_RECEIPT_BYTES = 2048
SOFTWARE_BINDING = {
    "image_id": "sha256:321d50ae5207dee05cd58ec5436d97978becde2ee8a767f2e8768229e6a9ad43",
    "source_commit": "c4272e36bc98354067dd23a83931a3db58823d76",
    "software_lock_sha256": "26bde6a0163e5db9d8d4564117546482601f4d0beb09f8705be5fc9183594965",
    "base_inventory_sha256": "e30ec2e16ab8360be394b890e9f06e8819ae9f12c2414b975148beb76d93f8e8",
    "source_inventory_sha256": "bf97197a3015753e5565a4f1105be748b7a2b0eb5baff3264ce24f6d79646d38",
    "acceptance_report_sha256": "24712f08f2522a0319da1179178aa8c73eef5969303faea73de16d89a23b386b",
}
COUNT_KEYS = (
    "pre_ml_rank_checks", "worker_constructors", "worker_admissions", "controller_agreements",
)
ERROR_CODES = (
    "runtime_rejected", "adapter_binding_rejected", "control_binding_rejected",
    "control_rejected", "source_identity_rejected", "probe_binding_rejected",
    "worker_bridge_rejected", "rank_proof_rejected", "guarded_worker_failed", "output_failed",
    "report_rejected",
)
CONTROL_FIELDS = (
    "source_commit", "software_lock_sha256", "base_inventory_sha256", "source_inventory_sha256",
)
ORIGIN_FIELDS = (
    "file_backed_modules", "source_modules", "dependency_modules", "runtime_modules",
    "distribution_versions", "map_lines", "executable_file_mappings", "executable_files",
    "anonymous_executable_mappings", "generated_python_modules",
)
OBSERVER_TRUE_FLAGS = (
    "inputs_barrier_and_final_origins_verified", "declared_distribution_versions_verified",
    "file_backed_executable_origins_verified", "parent_ml_import_absent",
)
OBSERVER_FALSE_FLAGS = (
    "real_sft_execution_permitted", "production_eligible", "real_model_or_data_access",
    "cuda_used", "complete_dynamic_dependency_closure_verified",
    "all_executable_code_origin_authenticated", "large_library_rehash_performed",
    "host_configuration_independently_verified", "real_host_admission_verified",
)
OBSERVER_COUNTS = {
    "rank_workers_witnessed": 6, "origin_observations": 12, "invocations_witnessed": 3,
}
WORKER_TRUE_FLAGS = (
    "real_cpu_ddp_used", "real_torch_dcp_used", "real_worker_run_used",
    "single_parent_supervisor_verified", "all_three_invocations_completed",
    "fresh_workers_and_epochs_verified", "all_optimizer_steps_in_shared_ledger",
    "six_prefix_loads_in_shared_ledger", "checkpoint_io_in_shared_ledger",
    "validation_metadata_present", "selected_prefix_only_read", "epoch_ack_confirmations_verified",
    "completion_fences_verified", "io_phase_confirmations_drained", "reference_model_resume_equal",
    "reference_adam_resume_equal", "reference_rng_resume_equal",
    "reference_training_state_resume_equal",
    "dropout_rng_consumption_verified", "private_modes_verified", "owned_processes_reaped",
    "rss_snapshot_observed", "loopback_gloo_used",
)
WORKER_FALSE_FLAGS = (
    "real_model_or_data_access", "cuda_used", "external_network_authorized",
    "os_resource_enforcement_verified", "production_eligible", "real_sft_execution_permitted",
    "full_content_hashes_verified", "fixture_generation_in_shared_ledger",
    "private_comparison_work_in_shared_ledger", "total_fixture_cost_metered",
    "safe_unpickling_or_origin_authentication_verified",
)
WORKER_COUNTS = {
    "rank_workers_verified": 6, "controlled_rank_optimizer_steps": 8,
    "global_optimizer_steps_reserved": 4, "global_optimizer_steps_completed": 4,
    "logical_sample_events_reserved": 16, "invocations_started": 3, "invocations_completed": 3,
    "controlled_loads": 6, "records_loaded": 12, "encoded_index_rows": 12,
    "binary_bytes_read": 288, "index_bytes_read": 1896, "checkpoints_published": 4,
    "rank_checkpoint_restores": 2, "commit_fences_sent": 8, "memory_observations": 3,
}
WORKER_COUNT_LIMITS = {
    "metadata_bytes_read": 65536, "ledger_requests": 100663296, "parent_wire_bytes": 100663296,
    "checkpoint_read_bytes_reserved": 100663296, "private_write_bytes_reserved": 33554432,
}
REPORT_KEYS = (
    "schema", "status", "code", "source_identity_sha256", "counts", "counts_semantics",
    "ml_absent", "parent_ml_import_absent", "adapter_separately_bound", "adapter_sha256",
    "control_sha256", "observer", "real_sft_execution_permitted", "production_eligible",
)
ML_ROOTS = ("torch", "transformers", "numpy", "cuda", "cupy", "jax", "tensorflow")
_PROBE = _FIXTURE = _CONTROL = None
_ORIGINAL_RANK = _ORIGINAL_OUTCOME = _ORIGINAL_INVOCATION = _ORIGINAL_FILE_ORIGIN = None
_ORIGINAL_IMPORT = importlib.import_module
_WORKER_MODULE = _WORKER = None
_COUNTS = dict.fromkeys(COUNT_KEYS, 0)
_LOCAL = dict.fromkeys(COUNT_KEYS[:3], 0)
_RANK = None
_INVOCATIONS = 0
_RUN_IDENTITY = None
_ADAPTER_BOUND = False
_ADAPTER_PIN = _CONTROL_PIN = None


class GuardianError(ValueError):
    def __init__(self, code):
        self.code = code if type(code) is str and code in ERROR_CODES else "guarded_worker_failed"
        super().__init__(self.code)


def need(value, code):
    if not value:
        raise GuardianError(code) from None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def sha256(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def source_identity(binding):
    return hashlib.sha256(canonical({"schema": IDENTITY_SCHEMA, **binding})).hexdigest()


def pairs(items):
    result = {}
    for key, value in items:
        need(key not in result, "control_rejected")
        result[key] = value
    return result


def reject_number(_value):
    raise GuardianError("control_rejected")


def validate_control_bytes(raw, expected):
    need(type(raw) is bytes and 0 < len(raw) <= MAXIMUM_CONTROL_BYTES,
         "control_binding_rejected")
    need(sha256(expected) and hashlib.sha256(raw).hexdigest() == expected,
         "control_binding_rejected")
    try:
        value = json.loads(raw.decode("ascii"), object_pairs_hook=pairs,
                           parse_int=reject_number, parse_float=reject_number,
                           parse_constant=reject_number)
        need(type(value) is dict and set(value) == {
            "schema", "software_binding", "source_identity_sha256",
        }, "control_rejected")
        need(value["schema"] == CONTROL_SCHEMA and type(value["software_binding"]) is dict
             and value["software_binding"] == SOFTWARE_BINDING, "control_rejected")
        need(value["source_identity_sha256"] == source_identity(SOFTWARE_BINDING),
             "source_identity_rejected")
        need(canonical(value) + b"\n" == raw, "control_rejected")
    except GuardianError:
        raise
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        raise GuardianError("control_rejected") from None
    return {**value, "software_binding": dict(value["software_binding"])}


def validate_observer(value):
    """Pure exact-schema check of the unchanged accepted v2/WorkerRun result."""
    need(type(value) is dict and set(value) == {
        "schema", "status", "worker_run", "origin_counts", *CONTROL_FIELDS,
        *OBSERVER_TRUE_FLAGS, *OBSERVER_FALSE_FLAGS, *OBSERVER_COUNTS,
    }, "report_rejected")
    need(type(value["schema"]) is type(value["status"]) is str
         and value["schema"] == "lumi.sft-oci-worker-probe.v0.1" and value["status"] == "passed"
         and all(type(value[key]) is str and value[key] == SOFTWARE_BINDING[key]
                 for key in CONTROL_FIELDS), "report_rejected")
    need(all(value[key] is True for key in OBSERVER_TRUE_FLAGS)
         and all(value[key] is False for key in OBSERVER_FALSE_FLAGS), "report_rejected")
    need(all(type(value[key]) is int and value[key] == count
             for key, count in OBSERVER_COUNTS.items()), "report_rejected")
    origins = value["origin_counts"]
    need(type(origins) is dict and set(origins) == set(ORIGIN_FIELDS)
         and all(type(count) is int and 0 <= count <= 196608 for count in origins.values()),
         "report_rejected")
    worker = value["worker_run"]
    need(type(worker) is dict and set(worker) == {
        "schema_version", "status", *WORKER_TRUE_FLAGS, *WORKER_FALSE_FLAGS,
        *WORKER_COUNTS, *WORKER_COUNT_LIMITS,
    }, "report_rejected")
    need(type(worker["schema_version"]) is type(worker["status"]) is str
         and worker["schema_version"] == "lumi.sft-worker-run-fixture.v0.1"
         and worker["status"] == "passed" and all(worker[key] is True for key in WORKER_TRUE_FLAGS)
         and all(worker[key] is False for key in WORKER_FALSE_FLAGS), "report_rejected")
    need(all(type(worker[key]) is int and worker[key] == count
             for key, count in WORKER_COUNTS.items()), "report_rejected")
    need(all(type(worker[key]) is int and 0 < worker[key] <= maximum
             for key, maximum in WORKER_COUNT_LIMITS.items()), "report_rejected")


def validate_report(value, *, case, source_identity_sha256, wait_status):
    """Pure host validator; returns a small receipt, never the private observer."""
    try:
        need(type(case) is str and case in ("positive", "wrong_control", "wrong_adapter")
             and type(wait_status) is int
             and sha256(source_identity_sha256)
             and source_identity_sha256 == source_identity(SOFTWARE_BINDING), "report_rejected")
        need(type(value) is dict and set(value) == set(REPORT_KEYS), "report_rejected")
        need(all(type(value[key]) is str for key in (
            "schema", "status", "code", "source_identity_sha256", "counts_semantics",
        )), "report_rejected")
        need(value["schema"] == SCHEMA and value["source_identity_sha256"] == source_identity_sha256
             and value["counts_semantics"] == "verified_completed_rank_proofs"
             and value["ml_absent"] is True and value["parent_ml_import_absent"] is True
             and value["real_sft_execution_permitted"] is False
             and value["production_eligible"] is False, "report_rejected")
        counts = value["counts"]
        need(type(counts) is dict and set(counts) == set(COUNT_KEYS), "report_rejected")
        if case == "positive":
            need(wait_status == 0 and value["status"] == "passed" and value["code"] == "ok"
                 and value["adapter_separately_bound"] is True
                 and sha256(value["adapter_sha256"]) and sha256(value["control_sha256"]),
                 "report_rejected")
            need(all(type(count) is int and count == 6 for count in counts.values()),
                 "report_rejected")
            validate_observer(value["observer"])
        else:
            expected = ("control_binding_rejected" if case == "wrong_control"
                        else "adapter_binding_rejected")
            need(wait_status == 512 and value["status"] == "failed" and value["code"] == expected
                 and value["observer"] is None and value["control_sha256"] is None,
                 "report_rejected")
            need(all(type(count) is int and count == 0 for count in counts.values()),
                 "report_rejected")
            if case == "wrong_control":
                need(value["adapter_separately_bound"] is True and sha256(value["adapter_sha256"]),
                     "report_rejected")
            else:
                need(value["adapter_separately_bound"] is False and value["adapter_sha256"] is None,
                     "report_rejected")
        receipt = {
            "schema": RECEIPT_SCHEMA, "status": "verified", "case": case,
            "worker_status": value["status"], "worker_code": value["code"],
            "source_identity_sha256": source_identity_sha256, "counts": dict(counts),
            "parent_ml_import_absent": True,
            "adapter_separately_bound": value["adapter_separately_bound"],
            "real_sft_execution_permitted": False, "production_eligible": False,
        }
        need(len(canonical(receipt)) + 1 <= MAXIMUM_RECEIPT_BYTES, "report_rejected")
        return receipt
    except GuardianError:
        raise
    except (KeyError, ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        raise GuardianError("report_rejected") from None


def ml_absent():
    return not any(name.split(".", 1)[0] in ML_ROOTS for name in tuple(sys.modules))


def pinned_file(path, expected, maximum, code, *, added=False):
    need(sha256(expected), code)
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
             and 0 < before.st_size <= maximum, code)
        if added:
            need(before.st_uid == before.st_gid == 0 and stat.S_IMODE(before.st_mode) == 0o444,
                 code)
        raw = bytearray()
        while len(raw) < before.st_size:
            part = os.read(descriptor, min(65536, before.st_size - len(raw)))
            need(bool(part), code)
            raw.extend(part)
        need(os.read(descriptor, 1) == b"", code)
        after = os.fstat(descriptor)
        need((before.st_dev, before.st_ino, before.st_mode, before.st_uid, before.st_gid,
              before.st_nlink, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
             == (after.st_dev, after.st_ino, after.st_mode, after.st_uid, after.st_gid,
                 after.st_nlink, after.st_size, after.st_mtime_ns, after.st_ctime_ns), code)
        need(hashlib.sha256(raw).hexdigest() == expected, code)
        named = os.stat(path, follow_symlinks=False)
        need((named.st_dev, named.st_ino) == (before.st_dev, before.st_ino), code)
        return bytes(raw), before
    finally:
        os.close(descriptor)


def verify_adapter():
    global _ADAPTER_BOUND
    expected = os.environ.get("LUMI_GUARDIAN_ADAPTER_SHA256")
    need(_ADAPTER_PIN is None or expected == _ADAPTER_PIN, "adapter_binding_rejected")
    need(str(Path(__file__).resolve(strict=True)) == ADAPTER_PATH, "adapter_binding_rejected")
    _, info = pinned_file(ADAPTER_PATH, expected, MAXIMUM_ADAPTER_BYTES,
                          "adapter_binding_rejected", added=True)
    _ADAPTER_BOUND = True
    return info


def load_control():
    expected = os.environ.get("LUMI_GUARDIAN_CONTROL_SHA256")
    need(_CONTROL_PIN is None or expected == _CONTROL_PIN, "control_binding_rejected")
    raw, _ = pinned_file(CONTROL_PATH, expected, MAXIMUM_CONTROL_BYTES,
                         "control_binding_rejected", added=True)
    result = validate_control_bytes(raw, expected)
    for environment, field in (
        ("LUMI_RELEASE_LOCK_SHA256", "software_lock_sha256"),
        ("LUMI_RELEASE_BASE_INVENTORY_SHA256", "base_inventory_sha256"),
        ("LUMI_RELEASE_SOURCE_INVENTORY_SHA256", "source_inventory_sha256"),
        ("LUMI_RELEASE_SOURCE_COMMIT", "source_commit"),
    ):
        need(os.environ.get(environment) == SOFTWARE_BINDING[field], "control_rejected")
    need(_CONTROL is None or result == _CONTROL, "control_rejected")
    return result


def runtime_guard():
    need(sys.platform == "linux" and os.uname().machine == "x86_64"
         and sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12)
         and sys.flags.isolated == 1 and sys.flags.no_site == 0 and sys.dont_write_bytecode
         and os.geteuid() == os.getegid() == 10001 and ml_absent(), "runtime_rejected")
    need(sys.argv[1:] == ["--synthetic-only"]
         and str(Path(sys.executable).resolve(strict=True)) == RUNTIME, "runtime_rejected")


def adapter_file_origin(path):
    # This exact extra file is independently byte-bound, never inserted into or
    # presented as a member of the accepted image's frozen source inventory.
    if type(path) is str and path == ADAPTER_PATH:
        return "source", ADAPTER_PATH, verify_adapter()
    return _ORIGINAL_FILE_ORIGIN(path)


def guarded_admit(original):
    need(_LOCAL["pre_ml_rank_checks"] == _LOCAL["worker_constructors"] == 1
         and _LOCAL["worker_admissions"] == 0, "worker_bridge_rejected")
    verify_adapter()
    need(load_control() == _CONTROL, "control_rejected")
    need(original() is None, "worker_bridge_rejected")
    _LOCAL["worker_admissions"] = 1


def construct_worker(*args, **kwargs):
    global _WORKER
    need(not args and _WORKER is None and _WORKER_MODULE is not None
         and _LOCAL["pre_ml_rank_checks"] == 1 and _LOCAL["worker_constructors"] == 0,
         "worker_bridge_rejected")
    need(type(kwargs.get("source_identity_sha256")) is str
         and kwargs.get("source_identity_sha256") == "f" * 64
         and type(kwargs.get("rank")) is int and kwargs.get("rank") == _RANK
         and type(kwargs.get("world_size")) is int and kwargs.get("world_size") == 2
         and kwargs.get("device_type") == "cpu" and callable(kwargs.get("admit")),
         "worker_bridge_rejected")
    need(load_control() == _CONTROL, "control_rejected")
    original_admit = kwargs["admit"]
    selected = dict(kwargs, source_identity_sha256=_CONTROL["source_identity_sha256"],
                    admit=lambda: guarded_admit(original_admit))
    worker = _WORKER_MODULE.WorkerRun(**selected)
    expected_plan = _WORKER_MODULE._digest({
        "schema_version": _WORKER_MODULE.SCHEMA_VERSION,
        "source_identity_sha256": _CONTROL["source_identity_sha256"],
        "loader_bindings": dict(worker._pins), "profile": dict(worker._profile),
        "world_size": 2, "record_limit": selected["record_limit"],
        "device_type": "cpu", "target_steps": 2,
    })
    need(worker._identity == expected_plan, "source_identity_rejected")
    _WORKER = worker
    _LOCAL["worker_constructors"] = 1
    return worker


def fixture_import(name, package=None):
    global _WORKER_MODULE
    original = _ORIGINAL_IMPORT(name, package)
    if name != "lumi.posttraining_sft_worker_run_v1":
        return original
    need(_WORKER_MODULE is None or _WORKER_MODULE is original, "worker_bridge_rejected")
    _WORKER_MODULE = original
    return SimpleNamespace(STAGES=original.STAGES, ERROR_CODES=original.ERROR_CODES,
                           WorkerRun=construct_worker)


def guarded_rank(*args, **kwargs):
    global _RANK
    try:
        need(not kwargs and len(args) == 13 and type(args[0]) is int and args[0] in (0, 1)
             and _RANK is None and all(value == 0 for value in _LOCAL.values()),
             "worker_bridge_rejected")
        need(ml_absent(), "runtime_rejected")
        verify_adapter()
        need(load_control() == _CONTROL, "control_rejected")
        _RANK = args[0]
        _LOCAL["pre_ml_rank_checks"] = 1
        return _ORIGINAL_RANK(*args, **kwargs)
    except BaseException:
        # No worker traceback or untrusted exception text, and no retry. The
        # unchanged parent's bounded EOF/failure handling retires the invocation.
        os._exit(2)


def guarded_outcome(pipe, value):
    if type(value) is dict and value.get("status") == "passed":
        need(PACKET_FIELD not in value and _WORKER is not None
             and all(type(count) is int and count == 1 for count in _LOCAL.values()),
             "rank_proof_rejected")
        identity = value["training_state"]["run_identity_sha256"]
        need(sha256(identity) and identity == _WORKER._identity, "source_identity_rejected")
        value = dict(value)
        value[PACKET_FIELD] = {
            "schema": PACKET_SCHEMA, "source_identity_sha256": _CONTROL["source_identity_sha256"],
            "run_identity_sha256": identity, "adapter_sha256": _ADAPTER_PIN,
            "control_sha256": _CONTROL_PIN, "counts": dict(_LOCAL),
        }
    return _ORIGINAL_OUTCOME(pipe, value)


def guarded_invocation(*args, **kwargs):
    global _INVOCATIONS, _RUN_IDENTITY
    need(ml_absent() and load_control() == _CONTROL and _INVOCATIONS < 3,
         "worker_bridge_rejected")
    reports, totals = _ORIGINAL_INVOCATION(*args, **kwargs)
    need(set(reports) == {0, 1}, "rank_proof_rejected")
    for rank in (0, 1):
        proof = reports[rank].pop(PACKET_FIELD, None)
        need(type(proof) is dict and set(proof) == {
            "schema", "source_identity_sha256", "run_identity_sha256", "adapter_sha256",
            "control_sha256", "counts",
        }, "rank_proof_rejected")
        need(proof["schema"] == PACKET_SCHEMA
             and proof["source_identity_sha256"] == _CONTROL["source_identity_sha256"]
             and proof["adapter_sha256"] == _ADAPTER_PIN
             and proof["control_sha256"] == _CONTROL_PIN,
             "rank_proof_rejected")
        counts = proof["counts"]
        need(type(counts) is dict and set(counts) == set(COUNT_KEYS[:3])
             and all(type(count) is int and count == 1 for count in counts.values()),
             "rank_proof_rejected")
        identity = proof["run_identity_sha256"]
        need(sha256(identity) and identity == reports[rank]["training_state"]["run_identity_sha256"]
             and (_RUN_IDENTITY is None or identity == _RUN_IDENTITY), "source_identity_rejected")
        _RUN_IDENTITY = identity
        for key in COUNT_KEYS[:3]:
            _COUNTS[key] += counts[key]
        _COUNTS["controller_agreements"] += 1
        need(all(value <= 6 for value in _COUNTS.values()), "rank_proof_rejected")
    _INVOCATIONS += 1
    return reports, totals


def prepare():
    global _PROBE, _FIXTURE, _CONTROL, _ADAPTER_PIN, _CONTROL_PIN
    global _ORIGINAL_RANK, _ORIGINAL_OUTCOME, _ORIGINAL_INVOCATION, _ORIGINAL_FILE_ORIGIN
    runtime_guard()
    verify_adapter()
    _ADAPTER_PIN = os.environ["LUMI_GUARDIAN_ADAPTER_SHA256"]
    _CONTROL = load_control()
    _CONTROL_PIN = os.environ["LUMI_GUARDIAN_CONTROL_SHA256"]
    raw, _ = pinned_file(PROBE_PATH, PROBE_SHA256, PROBE_SIZE, "probe_binding_rejected")
    need(len(raw) == PROBE_SIZE and PROBE_NAME not in sys.modules, "probe_binding_rejected")
    spec = importlib.util.spec_from_file_location(PROBE_NAME, PROBE_PATH)
    need(spec is not None, "probe_binding_rejected")
    _PROBE = importlib.util.module_from_spec(spec)
    sys.modules[PROBE_NAME] = _PROBE
    exec(compile(raw, PROBE_PATH, "exec"), _PROBE.__dict__)
    _PROBE.prepare()
    for key in ("source_commit", "software_lock_sha256", "base_inventory_sha256",
                "source_inventory_sha256"):
        need(_PROBE._STATE[key] == SOFTWARE_BINDING[key], "control_rejected")
    _FIXTURE = _PROBE._FIXTURE
    _ORIGINAL_FILE_ORIGIN = _PROBE.file_origin
    _ORIGINAL_RANK = _FIXTURE.rank_worker
    _ORIGINAL_OUTCOME, _ORIGINAL_INVOCATION = _FIXTURE.outcome, _FIXTURE.run_invocation
    _PROBE.file_origin = adapter_file_origin
    _FIXTURE.rank_worker = guarded_rank
    _FIXTURE.outcome, _FIXTURE.run_invocation = guarded_outcome, guarded_invocation
    _FIXTURE.importlib = SimpleNamespace(import_module=fixture_import)


def result(status, code, observer=None):
    return {
        "schema": SCHEMA, "status": status, "code": code,
        "source_identity_sha256": source_identity(SOFTWARE_BINDING),
        "counts": dict(_COUNTS), "counts_semantics": "verified_completed_rank_proofs",
        "ml_absent": ml_absent(), "parent_ml_import_absent": ml_absent(),
        "adapter_separately_bound": _ADAPTER_BOUND,
        "adapter_sha256": _ADAPTER_PIN, "control_sha256": _CONTROL_PIN,
        "observer": observer, "real_sft_execution_permitted": False, "production_eligible": False,
    }


def entry():
    output = os.dup(1) if __name__ == "__main__" else None
    sink = os.open(os.devnull, os.O_WRONLY | os.O_CLOEXEC)
    try:
        os.dup2(sink, 1)
        os.dup2(sink, 2)
    finally:
        os.close(sink)
    try:
        prepare()
        if output is None:
            return
        observer = _PROBE.run()
        need(ml_absent() and _INVOCATIONS == 3 and all(value == 6 for value in _COUNTS.values()),
             "rank_proof_rejected")
        need(observer["status"] == "passed" and observer["rank_workers_witnessed"] == 6
             and observer["origin_observations"] == 12, "rank_proof_rejected")
        report, code = result("passed", "ok", observer), 0
    except BaseException as error:
        if output is None:
            os._exit(2)
        fixed = error.code if type(error) is GuardianError else "guarded_worker_failed"
        report, code = result("failed", fixed), 2
    if output is not None:
        try:
            raw = canonical(report) + b"\n"
            need(len(raw) <= MAXIMUM_REPORT_BYTES and os.write(output, raw) == len(raw),
                 "output_failed")
        finally:
            os.close(output)
        raise SystemExit(code)


if __name__ in ("__main__", "__mp_main__"):
    entry()
