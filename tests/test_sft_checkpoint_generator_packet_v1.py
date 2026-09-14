"""Generated in-memory payloads only. Never print private packets or decoded bodies."""

from __future__ import annotations

import ast
import base64
import copy
import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_lumi_generator_packet_tests_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
before_import = set(sys.modules)
m = importlib.import_module(PACKAGE + ".posttraining_checkpoint_generator_packet_v1")
IMPORT_ADDITIONS = set(sys.modules) - before_import
PINS = {"entry_sha256": "a" * 64, "source_manifest_sha256": "b" * 64,
        "generation_nonce_sha256": "9" * 64}
PRIVATE = b"generated-private-payload-never-public"


def generated_fixture(payload=PRIVATE):
    """All bytes originate here. No model/trainer/tokenizer content is read."""
    config = m.FIXED_CONFIG_RAW
    state = m.canonical({"checkpoint_schema_version": 1, "step": 4,
        "sequence_cursor": 8, "tokens_seen": 256, "run_fingerprint": "a" * 64,
        "run_specification": {"source_commit": "b" * 40, "config_sha256": m.sha(config),
                              "distributed": {"world_size": 2}}})
    checkpoint = {"distributed-training-state.json": state,
        "model/.metadata": b"fresh generated metadata", "model/__0_0.distcp": payload,
        "trainer-rank-00000.pt": b"opaque synthetic trainer",
        "trainer-rank-00001.pt": b"opaque synthetic trainer"}
    manifest = "".join(f"{m.sha(raw)}  {name}\n"
                       for name, raw in sorted(checkpoint.items())).encode("ascii")
    checkpoint["manifest.sha256"] = manifest
    checkpoint_rows = [{"path": name, "bytes": len(raw), "sha256": m.sha(raw)}
                       for name, raw in sorted(checkpoint.items())]
    tokenizer = {"tokenizer/" + name: b'{"synthetic":true}' for name in
                 ("lumi_tokenizer_metadata.json", "tokenizer.json", "tokenizer_config.json")}
    token_rows = [{"relative_path": name, "bytes": len(raw), "sha256": m.sha(raw)}
                  for name, raw in sorted(tokenizer.items())]
    handoff = {"schema_version": 1, "gate": "lumi_4b_checkpoint_handoff_v0.1",
        "status": "sealed", "source_commit": "b" * 40, "run_fingerprint": "a" * 64,
        "world_size": 2, "files": checkpoint_rows,
        "checkpoint": {"directory": "checkpoint", "step": 4,
                       "sequence_cursor": 8, "tokens_seen": 256}}
    handoff["handoff_sha256"] = m.sha(m.canonical(handoff))
    handoff_raw = m.canonical(handoff)
    base = m.canonical({"schema_version": "lumi.posttraining-base-binding.v0.1",
        "checkpoint_stage": "base", "model_id": "lumi-4b-base", "run_fingerprint": "a" * 64,
        "source_commit": "b" * 40, "synthetic_fixture_only": True,
        "checkpoint": {"step": 4, "sequence_cursor": 8, "training_tokens": 256,
                       "world_size": 2, "manifest_sha256": m.sha(manifest)},
        "tokenizer": {"files": token_rows, "manifest_sha256": m.sha(m.canonical(token_rows))},
        "handoff": {"sha256": handoff["handoff_sha256"], "file_sha256": m.sha(handoff_raw)}})
    metadata = (base, manifest, state, config, handoff_raw)
    plan = m.planner.build_import_plan(*metadata, expected_base_sha256=m.sha(base),
                                      dtype="bfloat16")
    inputs = {"checkpoint": checkpoint, "tokenizer": tokenizer}
    count = sum(map(len, inputs.values()))
    size = sum(len(raw) for mapping in inputs.values() for raw in mapping.values())
    receipt = {"schema": "lumi.prepared-checkpoint-input-generation.v0.1", "status": "passed",
        "synthetic_only": True, "generation_nonce_sha256": PINS["generation_nonce_sha256"],
        "generation_passes": 1, "model_parameters": 736, "generated_input_files": count,
        "generated_input_bytes": size, "metadata_bytes": sum(map(len, metadata)),
        "generated_readback_bytes": size,
        "metadata_sha256": m.sha(m.canonical([m.sha(raw) for raw in metadata])),
        "input_inventory_sha256": m.sha(m.canonical({"checkpoint": plan["checkpoint_inventory"],
                                                    "tokenizer": plan["tokenizer_inventory"]})),
        "maximum_seconds": 30, "parent_containment_required": True,
        "source_origin_authenticated": False, "host_capacity_verified": False,
        "cumulative_generation_writes_metered": False, "real_model_or_data_access": False,
        "real_4b_import": False, "import_passes": 0, "forward_calls": 0, "optimizer_steps": 0,
        "training_authorized": False, "production_eligible": False, "retry_allowed": False}
    return {"metadata": metadata, "generated_inputs": inputs, "receipt": receipt}


def source_reports():
    return ({"source_manifest_sha256": PINS["source_manifest_sha256"],
        "source_files_verified": 12, "source_bytes_verified": 10000,
        "source_verification_read_bytes": 12000},
        {"private_source_copy_files": 12, "private_source_copy_bytes_reserved": 10000,
         "private_source_copy_readback_bytes": 10000,
         "private_source_copies_rootfs_immutable": False,
         "hostile_owner_code_immutability_verified": False})


def encode(value=None, **changes):
    source, copied = source_reports()
    return m.encode_generated_packet(generated_fixture() if value is None else value,
        **(PINS | {"source_report": source, "copy_report": copied} | changes))


def decode(raw, **changes):
    return m.decode_generated_packet(raw, **({"expected_" + key: value for key, value in
                                              PINS.items()} | changes))


class PacketTests(unittest.TestCase):
    def test_private_roundtrip_exact_original_bytes_and_public_receipt_only(self):
        original = generated_fixture()
        raw = encode(original)
        observed = decode(raw)
        self.assertTrue(observed["metadata"] == original["metadata"])
        self.assertTrue(observed["generated_inputs"] == original["generated_inputs"])
        self.assertEqual(set(observed), {"metadata", "generated_inputs", "receipt"})
        receipt = observed["receipt"]
        self.assertEqual(receipt["packet_bytes"], len(raw))
        self.assertEqual(receipt["packet_sha256"], m.sha(raw))
        self.assertFalse(receipt["raw_packet_public"])
        self.assertFalse(receipt["parent_containment_verified"])
        for secret in (PRIVATE, base64.b64encode(PRIVATE), b"model/__0_0.distcp"):
            self.assertNotIn(secret, m.canonical(receipt))
        self.assertLess(len(raw), m.MAXIMUM_PACKET_BYTES)

    def test_codec_performs_no_filesystem_operations_or_native_imports(self):
        generated = generated_fixture()
        with patch("builtins.open") as opened, patch.object(os, "open") as os_open:
            decode(encode(generated))
        opened.assert_not_called()
        os_open.assert_not_called()
        ml = {"torch", "numpy", "transformers", "safetensors", "tokenizers"}
        self.assertFalse(any(name.split(".", 1)[0] in ml for name in IMPORT_ADDITIONS))

    def test_roundtrip_feeds_frozen_pure_control_builder(self):
        controls = importlib.import_module(
            PACKAGE + ".posttraining_checkpoint_prepared_controls_v1")
        generated = decode(encode())
        result = controls.build_prepared_controls(metadata=generated["metadata"],
            generation_nonce_sha256=PINS["generation_nonce_sha256"],
            entry_sha256="c" * 64, source_manifest_sha256=PINS["source_manifest_sha256"],
            runtime_sources={"configuration_lumi.py": b"# generated config",
                             "modeling_lumi.py": b"# generated model"})
        self.assertEqual(result["counts"]["model_parameters"], 736)
        self.assertFalse(result["execution_permitted"])

    def test_actual_frozen_generation_receipt_roundtrip_with_only_byte_generator_double(self):
        generation = importlib.import_module(
            PACKAGE + ".posttraining_checkpoint_prepared_generation_v1")

        def generate_bytes(checkpoint, budget):
            budget()
            model = checkpoint / "model"
            model.mkdir(mode=0o700)
            generation.fixture._write(model / ".metadata", b"new generated metadata")
            generation.fixture._write(model / "__0_0.distcp", PRIVATE)

        with tempfile.TemporaryDirectory(prefix="lumi-packet-generated-test-") as directory, \
                patch.object(generation, "_runtime"), \
                patch.object(generation, "_ATTEMPTED", False), \
                patch.object(generation.fixture, "_generate", side_effect=generate_bytes) as tensor:
            original = generation.generate_prepared_inputs(Path(directory).resolve(),
                generation_nonce_sha256=PINS["generation_nonce_sha256"])
            observed = decode(encode(original))
        tensor.assert_called_once()
        self.assertTrue(observed["metadata"] == original["metadata"])
        self.assertTrue(observed["generated_inputs"] == original["generated_inputs"])
        self.assertEqual(observed["receipt"]["generation_receipt"], original["receipt"])

    def test_independent_pin_mismatch_and_zero_pins_rejected(self):
        raw = encode()
        for key in PINS:
            with self.subTest(key=key), self.assertRaises(m.GeneratorPacketError):
                decode(raw, **{"expected_" + key: "c" * 64})
            with self.assertRaises(m.GeneratorPacketError):
                encode(**{key: "0" * 64})

    def test_canonical_json_duplicates_numbers_and_extra_keys_rejected(self):
        raw = encode()
        for bad in (raw + b"\n", b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1.5}',
                    b'{"x":999999999999999999999}', b'[]'):
            with self.subTest(length=len(bad)), self.assertRaises(m.GeneratorPacketError):
                decode(bad)
        value = json.loads(raw)
        value["approved"] = True
        with self.assertRaises(m.GeneratorPacketError):
            decode(m.canonical(value))

    def test_wire_bounds_checked_before_json_or_base64_allocation(self):
        with patch.object(m, "_json") as parsed:
            with self.assertRaisesRegex(m.GeneratorPacketError, "wire_cap_exceeded"):
                decode(b"x" * (m.MAXIMUM_PACKET_BYTES + 1))
        parsed.assert_not_called()
        oversized = generated_fixture(payload=b"x" * 50000)
        with patch.object(m.base64, "b64encode") as encoded:
            with self.assertRaisesRegex(m.GeneratorPacketError, "wire_cap_exceeded"):
                encode(oversized)
        encoded.assert_not_called()

    def test_malformed_base64_count_digest_path_and_extra_row_field_rejected(self):
        value = json.loads(encode())
        variants = ({"base64": "!"}, {"bytes": True}, {"sha256": "c" * 64},
                    {"path": "../base.json"}, {"extra": "raw private text"})
        for changes in variants:
            bad = copy.deepcopy(value)
            bad["metadata"][0].update(changes)
            with self.subTest(keys=list(changes)), self.assertRaises(m.GeneratorPacketError):
                decode(m.canonical(bad))

    def test_metadata_tuple_order_and_input_row_order_are_exact(self):
        for key in ("metadata", "checkpoint"):
            value = json.loads(encode())
            rows = value["metadata"] if key == "metadata" else value["generated_inputs"][key]
            rows.reverse()
            with self.subTest(key=key), self.assertRaises(m.GeneratorPacketError):
                decode(m.canonical(value))

    def test_duplicate_missing_or_extra_input_rejected(self):
        for kind in ("duplicate", "missing", "extra"):
            value = json.loads(encode())
            rows = value["generated_inputs"]["checkpoint"]
            if kind == "duplicate":
                rows.insert(0, copy.deepcopy(rows[0]))
            elif kind == "missing":
                rows.pop()
            else:
                extra = copy.deepcopy(rows[0])
                extra["path"] = "unexpected.bin"
                rows.append(extra)
            with self.subTest(kind=kind), self.assertRaises(m.GeneratorPacketError):
                decode(m.canonical(value))

    def test_original_plan_hashes_not_only_wire_row_hashes_are_required(self):
        value = json.loads(encode())
        row = next(row for row in value["generated_inputs"]["checkpoint"]
                   if row["path"] == "model/__0_0.distcp")
        changed = b"different but wire-self-consistent"
        row.update(bytes=len(changed), sha256=m.sha(changed),
                   base64=base64.b64encode(changed).decode("ascii"))
        with self.assertRaisesRegex(m.GeneratorPacketError, "inventory_rejected"):
            decode(m.canonical(value))

    def test_prepared_marker_or_missing_synthetic_flag_is_not_original_generation(self):
        for kind in ("prepared", "missing"):
            generated = generated_fixture()
            metadata = list(generated["metadata"])
            base = m.json_object(metadata[0])
            if kind == "prepared":
                base["prepared_fixture_marker"] = {"approved": True}
            else:
                del base["synthetic_fixture_only"]
            metadata[0] = m.canonical(base)
            generated["metadata"] = tuple(metadata)
            with self.subTest(kind=kind), self.assertRaisesRegex(m.GeneratorPacketError,
                                                               "scope_rejected"):
                encode(generated)

    def test_different_tiny_config_is_not_accepted(self):
        generated = generated_fixture()
        metadata = list(generated["metadata"])
        config = m.json_object(metadata[3])
        config["vocab_size"] = 18
        metadata[3] = m.canonical(config)
        generated["metadata"] = tuple(metadata)
        with self.assertRaisesRegex(m.GeneratorPacketError, "scope_rejected"):
            encode(generated)

    def test_generation_receipt_counts_flags_and_hashes_are_recomputed(self):
        for changes in ({"generation_passes": True}, {"generated_readback_bytes": 1},
                        {"input_inventory_sha256": "c" * 64}, {"forward_calls": 1},
                        {"source_origin_authenticated": True}, {"approved": True}):
            generated = generated_fixture()
            generated["receipt"].update(changes)
            with self.subTest(keys=list(changes)), self.assertRaisesRegex(
                    m.GeneratorPacketError, "receipt_rejected"):
                encode(generated)

    def test_source_receipt_and_private_copy_relations_are_checked(self):
        variants = (({"source_files_verified": True}, {}),
            ({"source_manifest_sha256": "c" * 64}, {}),
            ({"source_verification_read_bytes": 10000}, {}),
            ({}, {"private_source_copy_bytes_reserved": 10001}),
            ({}, {"private_source_copies_rootfs_immutable": True}),
            ({"private_source": "text"}, {}))
        for source_change, copy_change in variants:
            source, copied = source_reports()
            source.update(source_change)
            copied.update(copy_change)
            with self.subTest(source_keys=list(source_change), copy_keys=list(copy_change)), \
                    self.assertRaisesRegex(m.GeneratorPacketError, "source_report_rejected"):
                encode(source_report=source, copy_report=copied)

    def test_results_and_receipts_are_detached(self):
        original = generated_fixture()
        before = copy.deepcopy(original)
        raw = encode(original)
        observed, other = decode(raw), decode(raw)
        observed["generated_inputs"]["checkpoint"].clear()
        observed["receipt"]["source_report"].clear()
        self.assertTrue(original == before)
        self.assertTrue(other["generated_inputs"] == before["generated_inputs"])
        self.assertTrue(other["receipt"]["source_report"])

    def test_errors_never_include_payload_text(self):
        with self.assertRaises(m.GeneratorPacketError) as caught:
            decode(PRIVATE)
        self.assertNotIn(PRIVATE.decode(), str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_codec_has_no_filesystem_model_or_execution_calls(self):
        tree = ast.parse(Path(m.__file__).read_bytes())
        forbidden = {"open", "read", "write", "read_bytes", "write_bytes", "execve", "fork",
                     "run_import", "generate_prepared_inputs", "_generate", "Popen"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else (
                    node.func.id if isinstance(node.func, ast.Name) else "")
                self.assertNotIn(name, forbidden)


if __name__ == "__main__":
    unittest.main()
