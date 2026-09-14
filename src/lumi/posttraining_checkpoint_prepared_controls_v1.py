"""Pure controls for one newly generated, fixed tiny prepared-import fixture.

Input bytes have already been acquired by the parent. This builder performs no
filesystem, transport, mount, process, model or inference operations. It creates
a NEW base document; all other supplied metadata bytes are retained unchanged.
The original synthetic marker is required, and a prepared marker is forbidden
on input: the caller must not reuse this builder's already prepared base.

Matching hashes and a generation nonce do not prove generation, software origin,
host ownership or freshness. The independently trusted parent must establish
those facts, pin the resulting controls and install them root-owned/read-only.
The frozen entry independently checks source-manifest membership and geometry.
This module cannot authorize execution or open the closed public real importer.
The resource envelope is a planning proposal, not observed host capacity.
"""

from __future__ import annotations

import re

from . import posttraining_checkpoint_import_plan_v1 as planner
from . import posttraining_checkpoint_io_contract_v1 as kernel_io
from . import posttraining_checkpoint_resource_envelope_v1 as resources
from . import posttraining_final_base_metadata_v1 as metadata_contract
from .posttraining_final_base_metadata_v1 import canonical, json_object, sha

SCHEMA = "lumi.prepared-checkpoint-controls-builder.v0.1"
STATUS = "PASS_pure_prepared_controls_not_execution_authority"
VALIDATION_SCHEMA = "lumi.prepared-checkpoint-controls-validation.v0.1"
VALIDATION_STATUS = "PASS_exact_prepared_controls_not_origin_or_execution"
CONTROL_SCHEMA = "lumi.prepared-checkpoint-fixture-control.v0.1"
BASE_MARKER_SCHEMA = "lumi.prepared-checkpoint-fixture-base.v0.1"
METADATA_NAMES = ("base.json", "manifest.sha256", "state.json", "config.json", "handoff.json")
CONTROL_NAMES = METADATA_NAMES + ("plan.json", "envelope.json", "kernel-io.json")
RUNTIME_NAMES = ("configuration_lumi.py", "modeling_lumi.py")
BASE_FIELDS = frozenset({"schema_version", "checkpoint_stage", "model_id", "run_fingerprint",
    "source_commit", "synthetic_fixture_only", "checkpoint", "tokenizer", "handoff"})
MAXIMUM_INPUT_FILES = 32
MAXIMUM_INPUT_BYTES = 4 * 1024**2
MAXIMUM_SOURCE_FILE_BYTES = 65536
MAXIMUM_CONTROL_BYTES = 131072
MAXIMUM_BUNDLE_BYTES = 524288
MAXIMUM_PARAMETERS = 1_000_000
MAXIMUM_READ_BYTES = 64 * 1024**2
MAXIMUM_WRITE_BYTES = 16 * 1024**2
MAXIMUM_SECONDS = 30
DCP_PAYLOAD_READ_BYTES = 8 * 1024**2
EXTRA_HOST_DISK_RESERVE_BYTES = 1024**3
FIXED_CONFIG_RAW = canonical({
    "vocab_size": 17, "hidden_size": 8, "intermediate_size": 16,
    "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
    "head_dim": 4, "max_position_embeddings": 16, "rope_theta": 10000.0,
    "rms_norm_eps": 0.00001, "attention_bias": False, "tie_word_embeddings": True,
    "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2, "doc_token_id": 3,
    "model_type": "lumi",
})
ERROR_CODES = frozenset({"arguments_rejected", "metadata_rejected", "synthetic_scope_rejected",
                         "source_contract_rejected", "resource_contract_rejected",
                         "control_cap_exceeded", "builder_failed"})
FALSE_FLAGS = (
    "generation_proven", "generation_freshness_verified", "runtime_origin_authenticated",
    "runtime_source_manifest_membership_verified", "host_capacity_verified",
    "source_immutability_verified", "model_payload_verified", "native_execution_performed",
    "dcp_native_serializer_overhead_measured", "real_4b_import", "execution_permitted",
    "training_authorized", "production_eligible", "retry_authorized",
)


class PreparedControlsError(ValueError):
    def __init__(self, code="builder_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "builder_failed"
        super().__init__(self.code)


def _need(value, code):
    if not value:
        raise PreparedControlsError(code) from None


def _pin(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None \
        and value != "0" * 64


def _metadata(value):
    maximum = metadata_contract.MAXIMUM_INPUT_BYTES
    caps = (maximum["base"], maximum["manifest"], maximum["state"], maximum["config"],
            planner.MAX_HANDOFF_BYTES)
    _need(type(value) is tuple and len(value) == len(METADATA_NAMES), "arguments_rejected")
    _need(all(type(raw) is bytes and 0 < len(raw) <= cap
              for raw, cap in zip(value, caps, strict=True))
          and sum(map(len, value[:4])) <= metadata_contract.MAXIMUM_COMBINED_BYTES,
          "metadata_rejected")
    _need(value[3] == FIXED_CONFIG_RAW, "synthetic_scope_rejected")
    try:
        base = json_object(value[0])
    except Exception:
        raise PreparedControlsError("metadata_rejected") from None
    _need(set(base) == BASE_FIELDS and base.get("synthetic_fixture_only") is True,
          "synthetic_scope_rejected")
    _need(base.get("run_fingerprint") == "a" * 64 and base.get("source_commit") == "b" * 40,
          "synthetic_scope_rejected")
    checkpoint = base.get("checkpoint")
    fixed = {"step": 4, "sequence_cursor": 8, "training_tokens": 256, "world_size": 2}
    _need(type(checkpoint) is dict and set(checkpoint) == set(fixed) | {"manifest_sha256"}
          and all(type(checkpoint[key]) is int and checkpoint[key] == expected
                  for key, expected in fixed.items()), "synthetic_scope_rejected")
    _need(type(base.get("tokenizer")) is dict
          and set(base["tokenizer"]) == {"files", "manifest_sha256"}
          and type(base.get("handoff")) is dict
          and set(base["handoff"]) == {"sha256", "file_sha256"}, "metadata_rejected")
    # Validate the original binding before changing only the NEW in-memory base.
    try:
        original_plan = planner.build_import_plan(*value,
            expected_base_sha256=sha(value[0]), dtype="bfloat16")
    except Exception:
        raise PreparedControlsError("metadata_rejected") from None
    _tiny_plan(original_plan)
    return base


def _tiny_plan(plan):
    _need(type(plan["model_parameters"]) is int and plan["model_parameters"] == 736,
          "synthetic_scope_rejected")
    rows = plan["checkpoint_inventory"] + plan["tokenizer_inventory"]
    _need(0 < len(rows) <= MAXIMUM_INPUT_FILES
          and 0 < sum(row["bytes"] for row in rows) <= MAXIMUM_INPUT_BYTES,
          "synthetic_scope_rejected")


def _runtime_rows(bodies):
    _need(type(bodies) is dict and set(bodies) == set(RUNTIME_NAMES),
          "source_contract_rejected")
    rows = []
    for name in RUNTIME_NAMES:
        raw = bodies[name]
        _need(type(raw) is bytes and 0 < len(raw) <= MAXIMUM_SOURCE_FILE_BYTES,
              "source_contract_rejected")
        rows.append({"path": name, "bytes": len(raw), "sha256": sha(raw)})
    return rows


def _envelope(plan, config_raw, runtime_rows):
    b = plan["resource_bounds"]
    source_bytes = sum(row["bytes"] for row in runtime_rows)
    tokens, backend, control = (b["tokenizer_inventory_bytes"],
                                b["maximum_backend_output_bytes"], MAXIMUM_CONTROL_BYTES)
    value = {
        "schema_version": resources.SCHEMA, "plan_sha256": plan["plan_sha256"],
        "dcp_payload_read_cap_bytes": DCP_PAYLOAD_READ_BYTES,
        "dcp_read_cap_bytes": b["checkpoint_inventory_bytes"] + DCP_PAYLOAD_READ_BYTES,
        "hf_read_cap_bytes": backend + 2 * source_bytes + len(config_raw),
        "package_read_cap_bytes": 3 * backend + 3 * tokens + 5 * control,
        "hf_write_cap_bytes": backend, "package_write_cap_bytes": tokens + 2 * control,
        "config_bytes": len(config_raw), "source_config_sha256": sha(config_raw),
        "control_bytes": control, "package_receipt_cap_bytes": control,
        "package_manifest_cap_bytes": control, "runtime_sources": runtime_rows,
        "private_write_cap_bytes": b["maximum_private_output_write_bytes"],
        "worker_memory_max_bytes": b["minimum_host_ram_planning_bytes"],
        "output_capacity_bytes": b["maximum_private_output_write_bytes"],
        "extra_host_disk_reserve_bytes": EXTRA_HOST_DISK_RESERVE_BYTES,
        "host_disk_capacity_bytes": b["minimum_staged_disk_planning_bytes"]
            + EXTRA_HOST_DISK_RESERVE_BYTES,
        "maximum_seconds": MAXIMUM_SECONDS,
    }
    value["coordinator_read_cap_bytes"] = (value["dcp_read_cap_bytes"]
        + value["hf_read_cap_bytes"] + b["checkpoint_inventory_bytes"]
        + b["tokenizer_audit_read_bytes"] + value["package_read_cap_bytes"] + source_bytes)
    _need(value["coordinator_read_cap_bytes"] <= MAXIMUM_READ_BYTES
          and value["private_write_cap_bytes"] <= MAXIMUM_WRITE_BYTES,
          "resource_contract_rejected")
    return value


def build_prepared_controls(*, metadata, generation_nonce_sha256, entry_sha256,
                            source_manifest_sha256, runtime_sources):
    """Return detached controls/proposals; never mutate input or perform I/O.

    ``metadata`` is the tuple (base, manifest, state, fixed config, handoff) of
    newly generated fixture bytes. ``runtime_sources`` contains only the two
    named runtime-source byte strings. Their membership/origin is not inferred
    from the separately supplied source-manifest hash. The parent and frozen
    entry must authenticate it independently before any executable use.
    """
    try:
        _need(all(_pin(value) for value in
                  (generation_nonce_sha256, entry_sha256, source_manifest_sha256)),
              "arguments_rejected")
        base = _metadata(metadata)
        runtime_rows = _runtime_rows(runtime_sources)
        base["prepared_fixture_marker"] = {
            "schema": BASE_MARKER_SCHEMA, "generator": "fixed-tiny-config-v1",
            "generation_nonce_sha256": generation_nonce_sha256,
        }
        prepared_metadata = (canonical(base), *metadata[1:])
        _need(len(prepared_metadata[0]) <= metadata_contract.MAXIMUM_INPUT_BYTES["base"]
              and sum(map(len, prepared_metadata[:4]))
                  <= metadata_contract.MAXIMUM_COMBINED_BYTES, "control_cap_exceeded")
        plan = planner.build_import_plan(*prepared_metadata,
            expected_base_sha256=sha(prepared_metadata[0]), dtype="bfloat16")
        _tiny_plan(plan)
        envelope = _envelope(plan, prepared_metadata[3], runtime_rows)
        plan_raw, envelope_raw = canonical(plan), canonical(envelope)
        admission = resources.validate_resource_envelope(plan_raw, envelope_raw,
            expected_plan_file_sha256=sha(plan_raw),
            expected_envelope_file_sha256=sha(envelope_raw))
        _need(admission["runner_limits"]["maximum_seconds"] == MAXIMUM_SECONDS
              and admission["execution_permitted"] is False, "resource_contract_rejected")
        backend = plan["resource_bounds"]["maximum_backend_output_bytes"]
        contract = kernel_io.build_kernel_io_contract(plan, checkpoint_read_contract={
            "expected_manifest_sha256": plan["metadata"]["checkpoint_manifest_sha256"],
            "maximum_file_bytes": max(row["bytes"] for row in plan["checkpoint_inventory"]),
            "maximum_read_bytes": envelope["dcp_read_cap_bytes"],
            "maximum_read_call_bytes": 1048576,
        }, hf_io_contract={
            "maximum_read_bytes": envelope["hf_read_cap_bytes"],
            "maximum_write_bytes": backend, "maximum_file_bytes": backend,
            "maximum_control_bytes": 65536, "target_shard_bytes": 268435456,
        })
        bodies = dict(zip(METADATA_NAMES, prepared_metadata, strict=True))
        bodies.update({"plan.json": plan_raw, "envelope.json": envelope_raw,
                       "kernel-io.json": canonical(contract)})
        control = {
            "schema": CONTROL_SCHEMA, "scope": "generated_tiny_checkpoint_only",
            "generation_nonce_sha256": generation_nonce_sha256, "entry_sha256": entry_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "limits": {"maximum_parameters": MAXIMUM_PARAMETERS,
                       "maximum_input_bytes": MAXIMUM_INPUT_BYTES},
            "files": [{"path": name, "bytes": len(bodies[name]), "sha256": sha(bodies[name])}
                      for name in CONTROL_NAMES],
        }
        control_raw = canonical(control)
        _need(all(0 < len(raw) <= MAXIMUM_CONTROL_BYTES
                  for raw in (*bodies.values(), control_raw))
              and sum(map(len, bodies.values())) + len(control_raw) <= MAXIMUM_BUNDLE_BYTES,
              "control_cap_exceeded")
        return {
            "schema": SCHEMA, "status": STATUS, "control_raw": control_raw, "bodies": bodies,
            # Fresh JSON copies also detach these objects from each other.
            "plan": json_object(plan_raw), "envelope": json_object(envelope_raw),
            "kernel_io_contract": json_object(bodies["kernel-io.json"]),
            "bindings": {
                "original_base_sha256": sha(metadata[0]),
                "prepared_base_sha256": sha(prepared_metadata[0]),
                "plan_file_sha256": sha(plan_raw), "plan_internal_sha256": plan["plan_sha256"],
                "envelope_file_sha256": sha(envelope_raw),
                "kernel_io_file_sha256": sha(bodies["kernel-io.json"]),
                "kernel_io_internal_sha256": contract["contract_sha256"],
                "control_file_sha256": sha(control_raw), "entry_sha256": entry_sha256,
                "source_manifest_sha256": source_manifest_sha256,
                "generation_nonce_sha256": generation_nonce_sha256,
            },
            "counts": {
                "metadata_files": len(metadata), "original_metadata_bytes": sum(map(len, metadata)),
                "prepared_metadata_bytes": sum(map(len, prepared_metadata)),
                "control_files": len(bodies) + 1,
                "control_file_bytes": sum(map(len, bodies.values())) + len(control_raw),
                "runtime_source_files": len(runtime_rows),
                "runtime_source_bytes": sum(row["bytes"] for row in runtime_rows),
                "generated_input_files": sum(len(plan[name]) for name in
                    ("checkpoint_inventory", "tokenizer_inventory")),
                "generated_input_bytes": sum(row["bytes"] for name in
                    ("checkpoint_inventory", "tokenizer_inventory") for row in plan[name]),
                "model_parameters": plan["model_parameters"], "model_payload_bytes_read": 0,
            },
            **dict.fromkeys(FALSE_FLAGS, False),
        }
    except PreparedControlsError:
        raise
    except Exception:
        raise PreparedControlsError("builder_failed") from None


def validate_prepared_controls(control_raw, bodies, *, expected_control_sha256,
                               expected_entry_sha256, expected_source_manifest_sha256,
                               expected_generation_nonce_sha256, runtime_sources):
    """Pure exact-v1 validation for the independently authorized root installer.

    Reconstruct the deterministic proposal, not any payload or original input
    file. Runtime bytes must separately belong to the installer's authenticated
    source manifest. A successful result is structural, never origin authority.
    The original pre-marker base's file bytes/hash are deliberately not claimed.
    """
    try:
        _need(all(_pin(value) for value in (expected_control_sha256, expected_entry_sha256,
            expected_source_manifest_sha256, expected_generation_nonce_sha256)),
            "arguments_rejected")
        _need(type(control_raw) is bytes and 0 < len(control_raw) <= MAXIMUM_CONTROL_BYTES
              and sha(control_raw) == expected_control_sha256, "metadata_rejected")
        _need(type(bodies) is dict and set(bodies) == set(CONTROL_NAMES)
              and all(type(raw) is bytes and 0 < len(raw) <= MAXIMUM_CONTROL_BYTES
                      for raw in bodies.values())
              and sum(map(len, bodies.values())) + len(control_raw) <= MAXIMUM_BUNDLE_BYTES,
              "control_cap_exceeded")
        control = json_object(control_raw)
        _need(type(control) is dict and set(control) == {
            "schema", "scope", "generation_nonce_sha256", "entry_sha256",
            "source_manifest_sha256", "limits", "files"}
            and control.get("schema") == CONTROL_SCHEMA
            and control.get("scope") == "generated_tiny_checkpoint_only"
            and control.get("entry_sha256") == expected_entry_sha256
            and control.get("source_manifest_sha256") == expected_source_manifest_sha256
            and control.get("generation_nonce_sha256") == expected_generation_nonce_sha256
            and canonical(control) == control_raw, "metadata_rejected")
        _need(type(control["files"]) is list and len(control["files"]) == len(CONTROL_NAMES),
              "metadata_rejected")
        for row, name in zip(control["files"], CONTROL_NAMES, strict=True):
            _need(type(row) is dict and set(row) == {"path", "bytes", "sha256"}
                  and row["path"] == name and type(row["bytes"]) is int
                  and row["bytes"] == len(bodies[name]) and row["sha256"] == sha(bodies[name]),
                  "metadata_rejected")
        base = json_object(bodies["base.json"])
        marker = {"schema": BASE_MARKER_SCHEMA, "generator": "fixed-tiny-config-v1",
                  "generation_nonce_sha256": expected_generation_nonce_sha256}
        _need(type(base.get("prepared_fixture_marker")) is dict
              and canonical(base["prepared_fixture_marker"]) == canonical(marker),
              "synthetic_scope_rejected")
        del base["prepared_fixture_marker"]  # Only this detached parsed in-memory copy.
        reconstructed = (canonical(base), *(bodies[name] for name in METADATA_NAMES[1:]))
        rebuilt = build_prepared_controls(metadata=reconstructed,
            generation_nonce_sha256=expected_generation_nonce_sha256,
            entry_sha256=expected_entry_sha256,
            source_manifest_sha256=expected_source_manifest_sha256,
            runtime_sources=runtime_sources)
        _need(rebuilt["control_raw"] == control_raw and rebuilt["bodies"] == bodies,
              "metadata_rejected")
        return {
            "schema": VALIDATION_SCHEMA, "status": VALIDATION_STATUS,
            "bindings": {key: value for key, value in rebuilt["bindings"].items()
                         if key != "original_base_sha256"},
            "counts": {key: value for key, value in rebuilt["counts"].items()
                       if key != "original_metadata_bytes"},
            "plan": rebuilt["plan"], "envelope": rebuilt["envelope"],
            "kernel_io_contract": rebuilt["kernel_io_contract"],
            **dict.fromkeys(FALSE_FLAGS, False),
        }
    except PreparedControlsError:
        raise
    except Exception:
        raise PreparedControlsError("metadata_rejected") from None
