"""Pure real-import preparation, never a host admission or executable adapter.

Inputs are already acquired, independently pinned control JSON bytes. No path is
opened, runtime imported, mount installed, payload read or process launched here.
The qualified tiny runner stays unchanged and its real entry stays closed.
Read-only mount flags and Unix ownership are separate requirements: the frozen
checkpoint reader needs worker-owned 0600 files, not root-owned/0400 substitutes.
No claim is made that a read-only bind prevents writes through another alias.
"""

from __future__ import annotations

import hashlib
import json

SCHEMA = "lumi.sft-checkpoint-host-contract.v0.1"
STATUS = "PASS_control_preparation_not_host_admission"
MAXIMUM_CONTROL_BYTES = 131072
MIB, GIB = 1024**2, 1024**3
ERROR_CODES = frozenset({"input_binding", "control_shape", "plan_rejected",
                         "qualification_rejected", "contract_binding", "envelope_rejected"})
PLAN_FALSE = (
    "checkpoint_payload_verified", "tokenizer_payload_verified", "real_4b_import",
    "host_resources_verified", "execution_permitted", "training_authorized",
    "real_sft_execution_permitted", "production_eligible",
)
FALSE_FLAGS = (
    "host_bound", "runtime_bound", "mounts_verified", "source_immutability_verified",
    "os_limits_enforced", "real_import_adapter_implemented", "execution_permitted",
    "real_4b_import_verified", "training_authorized", "production_eligible", "gate_5_complete",
)
BOUND_FIELDS = (
    "checkpoint_inventory_bytes", "checkpoint_inventory_files", "tokenizer_inventory_bytes",
    "source_fp32_tensor_bytes_including_aliases", "cast_tensor_bytes_including_aliases",
    "maximum_individual_tensor_bytes", "maximum_projected_live_tensor_bytes",
    "minimum_host_ram_planning_bytes", "maximum_backend_output_bytes",
    "maximum_private_output_write_bytes", "minimum_staged_disk_planning_bytes",
    "checkpoint_verification_passes", "checkpoint_audit_read_bytes", "tokenizer_audit_read_bytes",
)
ENVELOPE_FIELDS = (
    "dcp_read_cap_bytes", "hf_read_cap_bytes", "package_read_cap_bytes",
    "hf_write_cap_bytes", "package_write_cap_bytes", "config_bytes", "control_bytes",
    "coordinator_read_cap_bytes", "private_write_cap_bytes", "worker_memory_max_bytes",
    "output_capacity_bytes", "extra_host_disk_reserve_bytes", "host_disk_capacity_bytes",
    "maximum_seconds",
)


class HostContractError(ValueError):
    def __init__(self, code="control_shape"):
        self.code = code if type(code) is str and code in ERROR_CODES else "control_shape"
        super().__init__(self.code)


def _need(value, code):
    if not value:
        raise HostContractError(code) from None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _pin(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _positive(value):
    return type(value) is int and 0 < value <= 2**63 - 1


def _decode(raw, expected):
    _need(type(raw) is bytes and 0 < len(raw) <= MAXIMUM_CONTROL_BYTES
          and _pin(expected) and digest(raw) == expected, "input_binding")

    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "control_shape")
            result[key] = value
        return result

    def constant(_value):
        raise HostContractError("control_shape") from None

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
        _need(type(value) is dict, "control_shape")
        return value
    except HostContractError:
        raise
    except (ValueError, TypeError, RecursionError, UnicodeError):
        raise HostContractError("control_shape") from None


def _inventory(rows):
    _need(type(rows) is list and 0 < len(rows) <= 129, "plan_rejected")
    names, total = set(), 0
    for row in rows:
        _need(type(row) is dict and set(row) == {"path", "bytes", "sha256"}, "plan_rejected")
        name = row["path"]
        _need(type(name) is str and 0 < len(name) <= 256 and name not in names
              and all(part not in {"", ".", ".."} for part in name.split("/"))
              and "\\" not in name and all(32 < ord(c) < 127 for c in name)
              and _positive(row["bytes"]) and _pin(row["sha256"]), "plan_rejected")
        names.add(name)
        total += row["bytes"]
    _need(total <= 64 * GIB, "plan_rejected")
    return total


def _plan(raw, expected):
    plan = _decode(raw, expected)
    try:
        _need(plan["schema_version"] == "lumi.sft-checkpoint-import-plan.v0.1"
              and plan["status"] == "PASS_plan_only_not_execution"
              and plan["dtype"] == "bfloat16"
              and all(plan[key] is False for key in PLAN_FALSE), "plan_rejected")
        _need(_pin(plan["plan_sha256"]) and plan["plan_sha256"] == digest(canonical({
            k: v for k, v in plan.items() if k != "plan_sha256"})), "plan_rejected")
        b = plan["resource_bounds"]
        _need(type(b) is dict and set(b) == set(BOUND_FIELDS) | {
            "backend_payload_read_cap_verified", "os_memory_time_disk_limits_enforced"}
            and all(_positive(b[key]) for key in BOUND_FIELDS)
            and b["backend_payload_read_cap_verified"] is False
            and b["os_memory_time_disk_limits_enforced"] is False, "plan_rejected")
        checkpoint, tokens = (_inventory(plan[key]) for key in
                              ("checkpoint_inventory", "tokenizer_inventory"))
        peak = max(b["source_fp32_tensor_bytes_including_aliases"]
                   + b["cast_tensor_bytes_including_aliases"],
                   2 * b["cast_tensor_bytes_including_aliases"])
        peak += b["maximum_individual_tensor_bytes"] + 16 * MIB
        ram = ((2 * peak + 4 * GIB + 16 * GIB - 1) // (16 * GIB)) * 16 * GIB
        _need(b["checkpoint_inventory_bytes"] == checkpoint
              and b["checkpoint_inventory_files"] == len(plan["checkpoint_inventory"])
              and b["tokenizer_inventory_bytes"] == tokens
              and b["checkpoint_verification_passes"] == 2
              and b["checkpoint_audit_read_bytes"] == 2 * checkpoint
              and b["tokenizer_audit_read_bytes"] == 2 * tokens
              and b["maximum_projected_live_tensor_bytes"] == peak
              and b["minimum_host_ram_planning_bytes"] == ram
              and b["maximum_backend_output_bytes"]
                  == b["cast_tensor_bytes_including_aliases"] + MIB
              and b["maximum_private_output_write_bytes"]
                  == b["maximum_backend_output_bytes"] + tokens + 2 * MIB
              and b["minimum_staged_disk_planning_bytes"]
                  == checkpoint + tokens + b["maximum_private_output_write_bytes"],
              "plan_rejected")
        return plan
    except HostContractError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError):
        raise HostContractError("plan_rejected") from None


def _qualification(raw, expected):
    q = _decode(raw, expected)
    try:
        _need(q["schema_version"] == "lumi.sft-import-qualification-execution-verification.v0.2"
              and q["status"] == "PASS_tiny_native_qualification_and_all_triggered_CI"
              and type(q["native_run_attempts"]) is int and q["native_run_attempts"] == 1
              and type(q["workflow_retries"]) is int and q["workflow_retries"] == 0
              and q["all_triggered_CI_success"] is True
              and all(q[key] is False for key in ("real_4B_import_verified", "GPU_used",
                      "gate_5_complete", "execution_permitted", "production_eligible"))
              and _pin(q["report"]["sha256"]), "qualification_rejected")
        _need(type(q["passes"]) is list and len(q["passes"]) == 2, "qualification_rejected")
        for index, row in enumerate(q["passes"], 1):
            _need(type(row) is dict and type(row["pass"]) is int and row["pass"] == index
                  and row["status"] == "PASS_exact_native_receipts_publication_and_CI"
                  and row["prior_evidence_preserved"] is True
                  and row["all_triggered_CI_success"] is True
                  and row["tests_or_native_harness_executed"] is False
                  and all(type(row[key]) is int and row[key] == count for key, count in
                          (("published_files", 8), ("required_file_bindings", 76),
                           ("aggregate_receipts", 4), ("ordered_cases", 3),
                           ("network_operations", 0), ("real_payload_reads", 0))),
                  "qualification_rejected")
        return q
    except HostContractError:
        raise
    except (KeyError, TypeError):
        raise HostContractError("qualification_rejected") from None


def build_contract(plan_raw, qualification_raw, *, expected_plan_file_sha256,
                   expected_qualification_file_sha256):
    """Return deterministic requirements; unresolved fields cannot grant admission."""
    plan = _plan(plan_raw, expected_plan_file_sha256)
    q = _qualification(qualification_raw, expected_qualification_file_sha256)
    b = plan["resource_bounds"]
    result = {
        "schema_version": SCHEMA, "status": STATUS,
        "bindings": {"plan_file_sha256": expected_plan_file_sha256,
                     "plan_internal_sha256": plan["plan_sha256"],
                     "qualification_verification_file_sha256": expected_qualification_file_sha256,
                     "qualification_report_file_sha256": q["report"]["sha256"]},
        "resource_requirements": {key: b[key] for key in BOUND_FIELDS},
        "source_audit_only_bytes": (b["checkpoint_audit_read_bytes"]
                                    + b["tokenizer_audit_read_bytes"]),
        "source_audit_total_is_coordinator_read_cap": False,
        "logical_reads_are_physical_device_io": False,
        "mount_requirements": {
            "worker_uid": 10001, "worker_gid": 10001,
            "checkpoint": {"namespace_target": "/inputs/checkpoint", "directory_mode": "0700",
                           "file_mode": "0600", "worker_owned": True, "read_only": True},
            "tokenizer": {"namespace_target": "/inputs/tokenizer", "directory_mode": "0700",
                          "file_mode": "0600", "worker_owned": True, "read_only": True,
                          "preserve_tokenizer_subdirectory": True},
            "output": {"namespace_target": "/outputs", "directory_mode": "0700",
                       "worker_owned": True, "fresh_exclusive_attempt": True,
                       "retry_after_start": False},
            "symlinks_hardlinks_and_input_output_aliases_rejected": True,
            "recursive_read_only_nosuid_nodev_noexec_input_readback_required": True,
            "other_writable_aliases_and_writers_excluded": True,
            "chmod_or_chown_of_existing_sources_permitted": False,
            "root_owned_input_is_worker_owned_input": False,
            "read_only_bind_is_source_immutability_proof": False,
        },
        "host_requirements": {
            "dedicated": True, "platform": "linux", "architecture": "x86_64",
            "runtime_family": "cpython3.12", "gpu_required": False,
            "accepted_runtime_and_successor_source_closure_pinned": True,
            "independent_pid_and_cgroup_supervision": True, "swap_max_bytes": 0,
            "network_isolation_and_authority_fd_closure": True,
            "cumulative_write_accounting_not_just_filesystem_capacity": True,
            "no_trainer_deserialization_or_dataset_access": True,
            "no_forward_optimizer_training_or_promotion": True,
        },
        "read_envelope_terms": ["dcp_cap_including_first_checkpoint_audit", "hf_read_cap",
                                "one_additional_checkpoint_audit", "two_tokenizer_audits",
                                "package_read_cap"],
        "write_envelope_terms": ["hf_write_cap", "package_write_cap", "source_config_bytes",
                                 "three_control_envelopes"],
        "required_envelope_fields": list(ENVELOPE_FIELDS),
        "unresolved": dict.fromkeys(("dedicated_host_identity", "runtime_artifact_binding",
                                     "successor_source_closure", "input_backing_store_strategy",
                                     "native_mount_and_full_size_runner_adapter",
                                     "approved_resource_envelope", "host_control_readback",
                                     "real_import_authorization")),
        **dict.fromkeys(FALSE_FLAGS, False),
    }
    result["contract_sha256"] = digest(canonical(result))
    return result


def validate_contract(contract_raw, plan_raw, qualification_raw, *, expected_contract_file_sha256,
                      expected_plan_file_sha256, expected_qualification_file_sha256):
    value = _decode(contract_raw, expected_contract_file_sha256)
    wanted = build_contract(plan_raw, qualification_raw,
                            expected_plan_file_sha256=expected_plan_file_sha256,
                            expected_qualification_file_sha256=expected_qualification_file_sha256)
    _need(canonical(value) == canonical(wanted), "contract_binding")
    return wanted


def check_resource_envelope(plan_raw, envelope_raw, *, expected_plan_file_sha256,
                            expected_envelope_file_sha256):
    """Check proposed logical budget algebra, not host capacity or hard enforcement.

Passing this check does not fill the prepared contract's unresolved fields.
I/O sufficiency for a full-size conversion remains a separately qualified claim.
"""
    b = _plan(plan_raw, expected_plan_file_sha256)["resource_bounds"]
    e = _decode(envelope_raw, expected_envelope_file_sha256)
    _need(set(e) == set(ENVELOPE_FIELDS) and all(_positive(e[k]) for k in ENVELOPE_FIELDS),
          "envelope_rejected")
    reads = (e["dcp_read_cap_bytes"] + e["hf_read_cap_bytes"] + b["checkpoint_inventory_bytes"]
             + b["tokenizer_audit_read_bytes"] + e["package_read_cap_bytes"])
    writes = (e["hf_write_cap_bytes"] + e["package_write_cap_bytes"] + e["config_bytes"]
              + 3 * e["control_bytes"])
    _need(e["dcp_read_cap_bytes"] > b["checkpoint_inventory_bytes"]
          and e["hf_write_cap_bytes"] == b["maximum_backend_output_bytes"]
          and e["hf_read_cap_bytes"] >= e["hf_write_cap_bytes"]
          and e["package_write_cap_bytes"] >= b["tokenizer_inventory_bytes"]
          and e["package_read_cap_bytes"] >= e["hf_write_cap_bytes"]
              + b["tokenizer_inventory_bytes"]
          and e["config_bytes"] <= 65536 and e["control_bytes"] <= 131072
          and reads <= e["coordinator_read_cap_bytes"] <= 2**63 - 1
          and writes <= e["private_write_cap_bytes"] <= b["maximum_private_output_write_bytes"]
          and e["worker_memory_max_bytes"] >= b["minimum_host_ram_planning_bytes"]
          and e["output_capacity_bytes"] >= e["private_write_cap_bytes"]
          and e["host_disk_capacity_bytes"] >= b["checkpoint_inventory_bytes"]
              + b["tokenizer_inventory_bytes"] + e["output_capacity_bytes"]
              + e["extra_host_disk_reserve_bytes"]
          and e["maximum_seconds"] <= 86400, "envelope_rejected")
    return {"status": "PASS_proposed_budget_algebra_only", "required_read_envelope_bytes": reads,
            "required_write_envelope_bytes": writes, "full_size_io_sufficiency_verified": False,
            "host_capacity_verified": False, "os_limits_enforced": False,
            "execution_permitted": False, "training_authorized": False}
