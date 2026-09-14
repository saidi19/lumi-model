"""One bounded PRIVATE generated-input packet; never log encoded/decoded payloads.

The codec reads no files and imports no ML runtime. Metadata describes only the
fixed 736-parameter generator and its private generated maps. It is not source
origin, containment, generation-freshness or real-model authorization. The caller
must separately authenticate and contain the fixed generator before accepting
its output. Only the returned ``receipt`` may enter aggregate-only evidence.
"""

from __future__ import annotations

import base64
import json
import re

from . import posttraining_checkpoint_guard_fixture_v1 as fixture
from . import posttraining_checkpoint_import_plan_v1 as planner
from .posttraining_final_base_metadata_v1 import canonical, json_object, sha

SCHEMA = "lumi.checkpoint-generator-private-packet.v0.1"
RECEIPT_SCHEMA = "lumi.checkpoint-generator-packet-receipt.v0.1"
GENERATION_SCHEMA = "lumi.prepared-checkpoint-input-generation.v0.1"
MAXIMUM_PACKET_BYTES = 65536
MAXIMUM_INPUT_BYTES = 4 * 1024**2
MAXIMUM_INPUT_FILES = 32
MAXIMUM_INPUT_FILE_BYTES = 1024**2
MAXIMUM_SOURCE_BYTES = 4 * 1024**2
MAXIMUM_SOURCE_FILES = 128
METADATA_NAMES = ("base.json", "manifest.sha256", "state.json", "config.json", "handoff.json")
METADATA_CAPS = (16384, 16384, 32768, 16384, 32768)
ROLES = ("checkpoint", "tokenizer")
FIXED_CONFIG_RAW = canonical(fixture.CONFIG)
PACKET_KEYS = frozenset({"schema", "status", "entry_sha256", "source_manifest_sha256",
    "generation_nonce_sha256", "metadata", "generated_inputs", "generation_receipt",
    "source_report", "copy_report"})
SOURCE_KEYS = frozenset({"source_manifest_sha256", "source_files_verified",
                       "source_bytes_verified", "source_verification_read_bytes"})
COPY_KEYS = frozenset({"private_source_copy_files", "private_source_copy_bytes_reserved",
    "private_source_copy_readback_bytes", "private_source_copies_rootfs_immutable",
    "hostile_owner_code_immutability_verified"})
ERROR_CODES = frozenset({"packet_rejected", "binding_rejected", "scope_rejected",
    "inventory_rejected", "receipt_rejected", "source_report_rejected", "wire_cap_exceeded"})
PATH = re.compile(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+){0,2}\Z", re.ASCII)


class GeneratorPacketError(ValueError):
    def __init__(self, code="packet_rejected"):
        self.code = code if type(code) is str and code in ERROR_CODES else "packet_rejected"
        super().__init__(self.code)


def _need(value, code="packet_rejected"):
    if not value:
        raise GeneratorPacketError(code) from None


def _pin(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None \
        and value != "0" * 64


def _bindings(entry_sha256, source_manifest_sha256, generation_nonce_sha256):
    _need(all(_pin(value) for value in
              (entry_sha256, source_manifest_sha256, generation_nonce_sha256)), "binding_rejected")


def _number(value, maximum, minimum=1):
    return type(value) is int and minimum <= value <= maximum


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result)
            result[key] = value
        return result

    def integer(value):
        _need(len(value) <= 19)
        result = int(value)
        _need(0 <= result < 2**63)
        return result

    def invalid(_value):
        raise GeneratorPacketError() from None

    value = json.loads(raw, object_pairs_hook=pairs, parse_int=integer,
                       parse_float=invalid, parse_constant=invalid)
    _need(type(value) is dict and canonical(value) == raw)
    return value


def _source_reports(source, copied, source_manifest_sha256):
    _need(type(source) is dict and set(source) == SOURCE_KEYS
          and type(copied) is dict and set(copied) == COPY_KEYS, "source_report_rejected")
    _need(source["source_manifest_sha256"] == source_manifest_sha256
          and _number(source["source_files_verified"], MAXIMUM_SOURCE_FILES, 2)
          and _number(source["source_bytes_verified"], MAXIMUM_SOURCE_BYTES)
          and _number(source["source_verification_read_bytes"], MAXIMUM_SOURCE_BYTES + 65536)
          and 0 < source["source_verification_read_bytes"] - source["source_bytes_verified"]
              <= 65536, "source_report_rejected")
    for copied_key, source_key in (
        ("private_source_copy_files", "source_files_verified"),
        ("private_source_copy_bytes_reserved", "source_bytes_verified"),
        ("private_source_copy_readback_bytes", "source_bytes_verified"),
    ):
        _need(type(copied[copied_key]) is int and copied[copied_key] == source[source_key],
              "source_report_rejected")
    _need(copied["private_source_copies_rootfs_immutable"] is False
          and copied["hostile_owner_code_immutability_verified"] is False,
          "source_report_rejected")


def _original_plan(metadata):
    _need(type(metadata) is tuple and len(metadata) == 5, "scope_rejected")
    _need(all(type(raw) is bytes and 0 < len(raw) <= cap
              for raw, cap in zip(metadata, METADATA_CAPS, strict=True))
          and sum(map(len, metadata[:4])) <= 65536, "scope_rejected")
    _need(metadata[3] == FIXED_CONFIG_RAW, "scope_rejected")
    base = json_object(metadata[0])
    _need(base.get("synthetic_fixture_only") is True
          and "prepared_fixture_marker" not in base
          and base.get("run_fingerprint") == "a" * 64
          and base.get("source_commit") == "b" * 40, "scope_rejected")
    plan = planner.build_import_plan(*metadata, expected_base_sha256=sha(metadata[0]),
                                     dtype="bfloat16")
    _need(type(plan["model_parameters"]) is int and plan["model_parameters"] == 736,
          "scope_rejected")
    lineage = plan["metadata"]
    _need(all(type(lineage[key]) is int and lineage[key] == value for key, value in
              {"step": 4, "sequence_cursor": 8, "training_tokens": 256, "world_size": 2}.items()),
          "scope_rejected")
    return plan


def _generation_receipt(metadata, inputs, plan, nonce):
    rows = {"checkpoint": plan["checkpoint_inventory"], "tokenizer": plan["tokenizer_inventory"]}
    count = sum(map(len, inputs.values()))
    size = sum(len(raw) for mapping in inputs.values() for raw in mapping.values())
    return {"schema": GENERATION_SCHEMA, "status": "passed", "synthetic_only": True,
        "generation_nonce_sha256": nonce, "generation_passes": 1, "model_parameters": 736,
        "generated_input_files": count, "generated_input_bytes": size,
        "metadata_bytes": sum(map(len, metadata)), "generated_readback_bytes": size,
        "metadata_sha256": sha(canonical([sha(raw) for raw in metadata])),
        "input_inventory_sha256": sha(canonical(rows)), "maximum_seconds": 30,
        "parent_containment_required": True, "source_origin_authenticated": False,
        "host_capacity_verified": False, "cumulative_generation_writes_metered": False,
        "real_model_or_data_access": False, "real_4b_import": False, "import_passes": 0,
        "forward_calls": 0, "optimizer_steps": 0, "training_authorized": False,
        "production_eligible": False, "retry_allowed": False}


def _generated(value, nonce):
    _need(type(value) is dict and set(value) == {"metadata", "generated_inputs", "receipt"})
    metadata, inputs, observed = value["metadata"], value["generated_inputs"], value["receipt"]
    plan = _original_plan(metadata)
    _need(type(inputs) is dict and set(inputs) == set(ROLES), "inventory_rejected")
    total = count = 0
    for role in ROLES:
        mapping = inputs[role]
        inventory = plan[role + "_inventory"]
        _need(type(mapping) is dict and set(mapping) == {row["path"] for row in inventory},
              "inventory_rejected")
        count += len(mapping)
        _need(0 < count <= MAXIMUM_INPUT_FILES, "inventory_rejected")
        for row in inventory:
            raw = mapping[row["path"]]
            _need(type(raw) is bytes and 0 < len(raw) <= MAXIMUM_INPUT_FILE_BYTES
                  and len(raw) == row["bytes"] and sha(raw) == row["sha256"],
                  "inventory_rejected")
            total += len(raw)
            _need(total <= MAXIMUM_INPUT_BYTES, "inventory_rejected")
    _need(inputs["checkpoint"]["manifest.sha256"] == metadata[1]
          and inputs["checkpoint"]["distributed-training-state.json"] == metadata[2]
          and all(raw == b"opaque synthetic trainer" for name, raw in inputs["checkpoint"].items()
                  if name.startswith("trainer-rank-"))
          and all(raw == b'{"synthetic":true}' for raw in inputs["tokenizer"].values()),
          "inventory_rejected")
    wanted = _generation_receipt(metadata, inputs, plan, nonce)
    _need(type(observed) is dict and set(observed) == set(wanted)
          and all(type(observed[key]) is type(expected) and observed[key] == expected
                  for key, expected in wanted.items()), "receipt_rejected")
    return plan, wanted


def _encode_row(path, raw):
    return {"path": path, "bytes": len(raw), "sha256": sha(raw),
            "base64": base64.b64encode(raw).decode("ascii")}


def _decode_row(row, *, maximum):
    _need(type(row) is dict and set(row) == {"path", "bytes", "sha256", "base64"})
    name, count, encoded = row["path"], row["bytes"], row["base64"]
    _need(type(name) is str and len(name) <= 256 and PATH.fullmatch(name)
          and all(part not in {".", ".."} for part in name.split("/"))
          and _number(count, maximum) and _pin(row["sha256"])
          and type(encoded) is str and len(encoded) == 4 * ((count + 2) // 3))
    raw = base64.b64decode(encoded, validate=True)
    _need(base64.b64encode(raw).decode("ascii") == encoded
          and len(raw) == count and sha(raw) == row["sha256"])
    return name, raw


def encode_generated_packet(generated, *, entry_sha256, source_manifest_sha256,
                            generation_nonce_sha256, source_report, copy_report):
    """Return PRIVATE bytes. Do not print them or put them in aggregate evidence."""
    try:
        _bindings(entry_sha256, source_manifest_sha256, generation_nonce_sha256)
        _generated(generated, generation_nonce_sha256)
        _source_reports(source_report, copy_report, source_manifest_sha256)
        metadata, inputs = generated["metadata"], generated["generated_inputs"]
        # Reject an impossible wire size before allocating base64 expansions.
        raw_parts = (*metadata, *(raw for role in ROLES for raw in inputs[role].values()))
        _need(sum(4 * ((len(raw) + 2) // 3) for raw in raw_parts) < MAXIMUM_PACKET_BYTES,
              "wire_cap_exceeded")
        packet = {"schema": SCHEMA, "status": "passed", "entry_sha256": entry_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "generation_nonce_sha256": generation_nonce_sha256,
            "metadata": [_encode_row(name, raw)
                         for name, raw in zip(METADATA_NAMES, metadata, strict=True)],
            "generated_inputs": {role: [_encode_row(name, raw)
                for name, raw in sorted(inputs[role].items())] for role in ROLES},
            "generation_receipt": generated["receipt"],
            "source_report": source_report, "copy_report": copy_report}
        raw = canonical(packet)
        _need(0 < len(raw) <= MAXIMUM_PACKET_BYTES, "wire_cap_exceeded")
        return raw
    except GeneratorPacketError:
        raise
    except Exception:
        raise GeneratorPacketError() from None


def decode_generated_packet(raw, *, expected_entry_sha256, expected_source_manifest_sha256,
                            expected_generation_nonce_sha256):
    """Return PRIVATE maps and metadata plus an independently retainable receipt."""
    try:
        _bindings(expected_entry_sha256, expected_source_manifest_sha256,
                  expected_generation_nonce_sha256)
        _need(type(raw) is bytes and 0 < len(raw) <= MAXIMUM_PACKET_BYTES, "wire_cap_exceeded")
        packet = _json(raw)
        _need(set(packet) == PACKET_KEYS and packet["schema"] == SCHEMA
              and packet["status"] == "passed")
        _need(packet["entry_sha256"] == expected_entry_sha256
              and packet["source_manifest_sha256"] == expected_source_manifest_sha256
              and packet["generation_nonce_sha256"] == expected_generation_nonce_sha256,
              "binding_rejected")
        values = packet["metadata"]
        _need(type(values) is list and len(values) == len(METADATA_NAMES))
        metadata = []
        for row, name, maximum in zip(values, METADATA_NAMES, METADATA_CAPS, strict=True):
            selected, body = _decode_row(row, maximum=maximum)
            _need(selected == name)
            metadata.append(body)
        wire_inputs = packet["generated_inputs"]
        _need(type(wire_inputs) is dict and set(wire_inputs) == set(ROLES))
        inputs, count = {}, 0
        for role in ROLES:
            rows = wire_inputs[role]
            _need(type(rows) is list and 0 < len(rows) <= MAXIMUM_INPUT_FILES)
            count += len(rows)
            _need(count <= MAXIMUM_INPUT_FILES)
            mapping = {}
            for row in rows:
                name, body = _decode_row(row, maximum=MAXIMUM_INPUT_FILE_BYTES)
                _need(name not in mapping)
                mapping[name] = body
            _need(list(mapping) == sorted(mapping))
            inputs[role] = mapping
        generated = {"metadata": tuple(metadata), "generated_inputs": inputs,
                     "receipt": packet["generation_receipt"]}
        _plan, generation = _generated(generated, expected_generation_nonce_sha256)
        _source_reports(packet["source_report"], packet["copy_report"],
                        expected_source_manifest_sha256)
        aggregate = {"schema": RECEIPT_SCHEMA, "status": "passed",
            "packet_sha256": sha(raw), "packet_bytes": len(raw),
            "entry_sha256": expected_entry_sha256,
            "source_manifest_sha256": expected_source_manifest_sha256,
            "generation_nonce_sha256": expected_generation_nonce_sha256,
            "generation_receipt": generation, "source_report": dict(packet["source_report"]),
            "copy_report": dict(packet["copy_report"]), "raw_packet_public": False,
            "parent_containment_verified": False, "real_4b_import": False,
            "training_authorized": False, "production_eligible": False}
        return {"metadata": generated["metadata"], "generated_inputs": inputs,
                "receipt": aggregate}
    except GeneratorPacketError:
        raise
    except Exception:
        raise GeneratorPacketError() from None
