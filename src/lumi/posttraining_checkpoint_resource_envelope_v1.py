"""Pure proposed full-size logical resource algebra, never execution authority.

Only independently pinned, already acquired JSON bytes are accepted. No file,
payload, model runtime, process, network or mount is accessed. A passing result
is a copied proposal: it neither admits a host nor opens the real-import API.

Frozen v5 HF inventory calls are metadata-only. Its content reads are source
config C, initial runtime sources Q, twice the copied runtime Q, and the saved
config/index/weight contents (at most B-Q): total at most B+2Q+C. The package v2
finalizer hashes backend B and copies tokens T, then its verifier hashes B+T,
reads receipt R and manifest M, and the frozen consumer hashes B+T+R and reads M
twice. Thus package reads reserve 3B+3T+2R+3M, writes T+R+M. All component caps,
not merely their predicted successful usage, are prepaid by the coordinator.
The successor's separate runtime-source preflight reserves another Q read.
The package has a single control limit, so R=M=that enforced control limit.

DCP needs its first full checkpoint audit plus a separately declared payload
cap. Two metadata parses and the model storage/tensor sizes give only a minimum
planning floor: native serializer seek/repeated-read overhead is unmeasured.
This validator does not infer full-size I/O sufficiency or performance from a
tiny successful qualification. Dependency and physical-device I/O are excluded.
"""

from __future__ import annotations

import re

from . import posttraining_checkpoint_host_contract_v1 as host

SCHEMA = "lumi.sft-checkpoint-resource-envelope.v0.1"
STATUS = "PASS_proposed_resource_algebra_not_host_admission"
MAXIMUM_BYTES = 2**63 - 1
RUNTIME_NAMES = ("configuration_lumi.py", "modeling_lumi.py")
FIELDS = frozenset(host.ENVELOPE_FIELDS) | {
    "schema_version", "plan_sha256", "dcp_payload_read_cap_bytes",
    "source_config_sha256", "runtime_sources", "package_receipt_cap_bytes",
    "package_manifest_cap_bytes",
}
INTEGER_FIELDS = frozenset(host.ENVELOPE_FIELDS) | {
    "dcp_payload_read_cap_bytes", "package_receipt_cap_bytes", "package_manifest_cap_bytes",
}
ERROR_CODES = frozenset({"input_binding", "control_shape", "plan_rejected",
                         "envelope_rejected", "resource_exhausted"})
FALSE_FLAGS = (
    "host_capacity_verified", "runtime_bound", "os_limits_enforced",
    "native_io_sufficiency_verified", "full_size_io_sufficiency_verified",
    "full_size_performance_measured", "physical_device_io_measured",
    "dependency_module_reads_metered", "source_immutability_verified",
    "execution_permitted", "real_4b_import_verified", "training_authorized",
    "production_eligible", "gate_5_complete",
)


class ResourceEnvelopeError(ValueError):
    def __init__(self, code="envelope_rejected"):
        self.code = code if type(code) is str and code in ERROR_CODES else "envelope_rejected"
        super().__init__(self.code)


def _need(value, code="envelope_rejected"):
    if not value:
        raise ResourceEnvelopeError(code) from None


def _sum(*values):
    result = sum(values)
    _need(0 < result <= MAXIMUM_BYTES, "resource_exhausted")
    return result


def _runtime_sources(value):
    _need(type(value) is list and len(value) == len(RUNTIME_NAMES))
    for row, name in zip(value, RUNTIME_NAMES, strict=True):
        _need(type(row) is dict and set(row) == {"path", "bytes", "sha256"}
              and row["path"] == name and type(row["path"]) is str
              and type(row["bytes"]) is int and 0 < row["bytes"] <= 65536
              and host._pin(row["sha256"]))
    return _sum(*(row["bytes"] for row in value))


def validate_resource_envelope(plan_raw, envelope_raw, *, expected_plan_file_sha256,
                               expected_envelope_file_sha256):
    """Return copied caps and runner limits; never host/execution capabilities.

The future coordinator must separately compare the source-config bytes and
frozen runtime-source closure against these bindings before any native call.
Receipt/manifest ceilings are caller proposals, checked by the unchanged package
writer at runtime; this does not claim their full-size contents have been built.
"""
    try:
        plan = host._plan(plan_raw, expected_plan_file_sha256)
        e = host._decode(envelope_raw, expected_envelope_file_sha256)
    except host.HostContractError as error:
        raise ResourceEnvelopeError(error.code) from None
    try:
        _need(set(e) == FIELDS and e["schema_version"] == SCHEMA
              and e["plan_sha256"] == plan["plan_sha256"]
              and all(host._positive(e[key]) for key in INTEGER_FIELDS))
        b = plan["resource_bounds"]
        metadata = plan["metadata"]
        _need(type(metadata) is dict and host._pin(metadata["source_config_sha256"])
              and host._pin(e["source_config_sha256"])
              and e["source_config_sha256"] == metadata["source_config_sha256"],
              "plan_rejected")
        rows = {row["path"]: row for row in plan["checkpoint_inventory"]}
        _need("model/.metadata" in rows and "manifest.sha256" in rows
              and rows["manifest.sha256"]["sha256"]
                  == metadata["checkpoint_manifest_sha256"], "plan_rejected")
        model_rows = [row for name, row in rows.items()
                      if re.fullmatch(r"model/__[0-9]+_[0-9]+\.distcp", name)]
        _need(bool(model_rows), "plan_rejected")
        q = _runtime_sources(e["runtime_sources"])
        c, t, backend = (e["config_bytes"], b["tokenizer_inventory_bytes"],
                         b["maximum_backend_output_bytes"])
        receipt, manifest = e["package_receipt_cap_bytes"], e["package_manifest_cap_bytes"]
        control = e["control_bytes"]
        _need(c <= 65536 and c <= control <= 131072
              and receipt == control and manifest == control
              and all(row["bytes"] <= min(65536, control) for row in e["runtime_sources"]))
        dcp_floor = _sum(max(b["source_fp32_tensor_bytes_including_aliases"],
                            sum(row["bytes"] for row in model_rows)),
                         2 * rows["model/.metadata"]["bytes"])
        hf_floor = _sum(backend, 2 * q, c)
        package_read_floor = _sum(3 * backend, 3 * t, 2 * receipt, 3 * manifest)
        package_write_floor = _sum(t, receipt, manifest)
        _need(e["dcp_payload_read_cap_bytes"] >= dcp_floor
              and e["dcp_read_cap_bytes"] == _sum(b["checkpoint_inventory_bytes"],
                                                 e["dcp_payload_read_cap_bytes"])
              and e["hf_write_cap_bytes"] == backend
              and e["hf_read_cap_bytes"] >= hf_floor
              and e["package_read_cap_bytes"] >= package_read_floor
              and e["package_write_cap_bytes"] >= package_write_floor)
        reads = _sum(e["dcp_read_cap_bytes"], e["hf_read_cap_bytes"],
                     b["checkpoint_inventory_bytes"], b["tokenizer_audit_read_bytes"],
                     e["package_read_cap_bytes"], q)
        writes = _sum(e["hf_write_cap_bytes"], e["package_write_cap_bytes"], c, 3 * control)
        disk = _sum(b["checkpoint_inventory_bytes"], t, e["output_capacity_bytes"],
                    e["extra_host_disk_reserve_bytes"])
        _need(reads <= e["coordinator_read_cap_bytes"]
              and writes <= e["private_write_cap_bytes"]
                  <= b["maximum_private_output_write_bytes"]
              and e["worker_memory_max_bytes"] >= b["minimum_host_ram_planning_bytes"]
              and e["output_capacity_bytes"] >= e["private_write_cap_bytes"]
              and e["host_disk_capacity_bytes"] >= disk
              and e["maximum_seconds"] <= 86400)
        return {
            "schema_version": SCHEMA, "status": STATUS,
            "plan_file_sha256": expected_plan_file_sha256,
            "plan_internal_sha256": plan["plan_sha256"],
            "envelope_file_sha256": expected_envelope_file_sha256,
            "envelope": e,
            "runner_limits": {
                "maximum_seconds": e["maximum_seconds"],
                "maximum_coordinator_read_bytes": e["coordinator_read_cap_bytes"],
                "maximum_private_write_bytes": e["private_write_cap_bytes"],
                "maximum_package_read_bytes": e["package_read_cap_bytes"],
                "maximum_package_write_bytes": e["package_write_cap_bytes"],
                "maximum_control_bytes": control,
            },
            "minimum_dcp_payload_planning_bytes": dcp_floor,
            "minimum_hf_content_read_bytes": hf_floor,
            "minimum_package_content_read_bytes": package_read_floor,
            "minimum_package_content_write_bytes": package_write_floor,
            "required_coordinator_read_envelope_bytes": reads,
            "required_private_write_envelope_bytes": writes,
            "required_host_disk_capacity_bytes": disk,
            "frozen_private_write_cap_bytes": b["maximum_private_output_write_bytes"],
            "runtime_source_bytes": q,
            "runtime_source_preflight_read_bytes": q,
            "checkpoint_full_audits_reserved": 2,
            "tokenizer_full_audits_reserved": 2,
            "dcp_metadata_reads_in_planning_floor": 2,
            "terminal_control_envelopes_reserved": 3,
            "dcp_native_serializer_overhead_measured": False,
            "package_control_size_sufficiency_verified": False,
            "runtime_source_payloads_verified": False,
            "reservations_refundable": False,
            **dict.fromkeys(FALSE_FLAGS, False),
        }
    except ResourceEnvelopeError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError):
        raise ResourceEnvelopeError("envelope_rejected") from None
