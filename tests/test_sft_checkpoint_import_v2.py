"""Stdlib contract/security tests; not substitutes for genuine tensor import tests."""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from lumi import posttraining_checkpoint_import_v2 as m


class ImportContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.checkpoint = self.root / "dcp"
        self.tokenizer = self.root / "tokenizer"
        self.checkpoint.mkdir(mode=0o700)
        self.tokenizer.mkdir(mode=0o700)
        values = {
            "vocab_size": 32, "hidden_size": 8, "intermediate_size": 16,
            "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 4, "max_position_embeddings": 16, "rope_theta": 10000.0,
            "rms_norm_eps": 0.00001, "attention_bias": False, "tie_word_embeddings": True,
            "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2, "doc_token_id": 3,
            "model_type": "lumi",
        }
        self.config_raw = m.canonical(values)
        self.config = self.root / "config.json"
        self.config.write_bytes(self.config_raw)
        self.state = {"checkpoint_schema_version": 1, "step": 1, "sequence_cursor": 2,
                      "tokens_seen": 32, "run_fingerprint": "a" * 64,
                      "run_specification": {"source_commit": "b" * 40,
                                            "config_sha256": m.sha(self.config_raw),
                                            "distributed": {"world_size": 2}}}
        self.state_raw = m.canonical(self.state)
        m.write_new(self.checkpoint, m.STATE, self.state_raw)
        for name in ("model/.metadata", "trainer-rank-00000.pt", "trainer-rank-00001.pt"):
            m.write_new(self.checkpoint, name, b"synthetic-not-a-tensor")
        self.source_manifest = self.seal(self.checkpoint)
        files = []
        for name in ("tokenizer.json", "tokenizer_config.json", "lumi_tokenizer_metadata.json"):
            raw = b'{"synthetic":true}'
            m.write_new(self.tokenizer, name, raw)
            files.append({"relative_path": "tokenizer/" + name,
                          "bytes": len(raw), "sha256": m.sha(raw)})
        self.tokenizer_manifest = self.seal(self.tokenizer)
        self.base = {
            "schema_version": "lumi.posttraining-base-binding.v0.1", "checkpoint_stage": "base",
            "model_id": "lumi-synthetic-import-fixture", "synthetic_only": True,
            "run_fingerprint": "a" * 64, "source_commit": "b" * 40,
            "checkpoint": {"manifest_sha256": m.sha(self.source_manifest), "step": 1,
                           "sequence_cursor": 2, "training_tokens": 32, "world_size": 2},
            "tokenizer": {"manifest_sha256": m.sha(self.tokenizer_manifest), "files": files},
        }

    def seal(self, root):
        entries = m.tree_inventory(root)
        raw = "".join(f"{x['sha256']}  {x['path']}\n" for x in entries).encode()
        m.write_new(root, m.MANIFEST, raw)
        return raw

    def validate(self, **changes):
        args = {"base": self.base, "source_manifest": self.source_manifest,
                "state_raw": self.state_raw, "config_raw": self.config_raw,
                "tokenizer_manifest": self.tokenizer_manifest, **changes}
        return m.validate_lineage(**args)

    def execute(self, **changes):
        return m.run_synthetic_import(**{
            "base": self.base, "checkpoint": self.checkpoint, "config": self.config,
            "tokenizer": self.tokenizer, "attempt": self.root / "attempt", **changes,
        })

    def test_complete_lineage_contract(self):
        result = self.validate()
        self.assertEqual(result["source_config_sha256"], m.sha(self.config_raw))
        self.assertEqual(result["checkpoint"]["training_tokens"], self.state["tokens_seen"])

    def test_manifest_binding_rejection(self):
        with self.assertRaisesRegex(ValueError, "source_manifest_binding"):
            self.validate(source_manifest=self.source_manifest + b"\n")

    def test_wrong_state_bytes_rejected(self):
        with self.assertRaisesRegex(ValueError, "checkpoint_state_not_bound"):
            self.validate(state_raw=self.state_raw + b" ")

    def test_wrong_config_rejected_even_if_same_shape(self):
        with self.assertRaisesRegex(ValueError, "source_config_binding"):
            self.validate(config_raw=self.config_raw + b" ")

    def test_wrong_tokenizer_manifest_rejected(self):
        with self.assertRaisesRegex(ValueError, "tokenizer_manifest_binding"):
            self.validate(tokenizer_manifest=self.tokenizer_manifest + b"\n")

    def test_checkpoint_cursor_contract(self):
        for key in ("step", "sequence_cursor", "training_tokens", "world_size"):
            for value in (True, 999):
                changed = copy.deepcopy(self.base)
                changed["checkpoint"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.validate(base=changed)

    def test_run_and_commit_contract(self):
        for key in ("run_fingerprint", "source_commit"):
            changed = {**self.base, key: "wrong"}
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(base=changed)

    def test_manifest_rejects_duplicate_self_and_paths(self):
        for name in ("../secret", "/secret", "a//b", "a/./b", "a\\b", "a b", "manifest.sha256"):
            raw = f"{'a' * 64}  {name}\n".encode()
            with self.subTest(name=name), self.assertRaises(ValueError):
                m.parse_manifest(raw)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            m.parse_manifest((f"{'a' * 64}  file\n" * 2).encode())

    def test_manifest_rejects_malformed_lines(self):
        for raw in (b"", b"no-newline", b"bad\n", b"a  b\r\n", b"\n"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                m.parse_manifest(raw)

    def test_duplicate_json_and_nonfinite_rejected(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b"[]"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                m.json_object(raw)

    def test_closed_inventory_rejects_undeclared_file(self):
        m.write_new(self.checkpoint, "extra", b"x")
        with self.assertRaisesRegex(ValueError, "closed_inventory"):
            m.verify_closed_manifest(self.checkpoint, m.sha(self.source_manifest))

    def test_closed_inventory_rejects_mutated_file(self):
        (self.checkpoint / "model/.metadata").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "closed_inventory"):
            m.verify_closed_manifest(self.checkpoint, m.sha(self.source_manifest))

    def test_symlink_rejected_before_read(self):
        (self.root / "link").symlink_to(self.config)
        with self.assertRaisesRegex(ValueError, "symlink"):
            m.read_regular(self.root, "link")

    def test_hardlink_rejected(self):
        os.link(self.config, self.root / "hardlink")
        with self.assertRaisesRegex(ValueError, "unsafe_regular"):
            m.read_regular(self.root, "hardlink")

    def test_size_cap_before_content_read(self):
        with patch.object(m, "MAX_FILE_BYTES", 1), patch.object(
            m.os, "open", side_effect=AssertionError("must not read")
        ), self.assertRaisesRegex(ValueError, "file_size_cap"):
            m.read_regular(self.root, self.config.name)

    def test_private_mode_rejection(self):
        (self.tokenizer / "tokenizer.json").chmod(0o644)
        with self.assertRaisesRegex(ValueError, "unsafe_private_file"):
            m.tree_inventory(self.tokenizer, private=True)

    def test_real_base_not_executable_before_dependency_or_input_read(self):
        real = {**self.base, "model_id": "lumi-4b-base", "synthetic_only": False}
        with patch.object(m, "read_regular", side_effect=AssertionError("no reads")):
            with self.assertRaisesRegex(ValueError, "real_import_not_authorized"):
                self.execute(base=real)

    def test_synthetic_limits_are_not_boolean_or_4b_escape(self):
        cases = (("step", 76294), ("training_tokens", 80_000_000_000), ("world_size", True))
        for key, value in cases:
            changed = copy.deepcopy(self.base)
            changed["checkpoint"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, "synthetic_checkpoint_cap"
            ):
                self.execute(base=changed)

    def test_dtype_rejected_before_backend(self):
        with self.assertRaisesRegex(ValueError, "unsupported_dtype"):
            self.execute(dtype="float16")

    def test_existing_attempt_is_never_retried(self):
        attempt = self.root / "attempt"
        attempt.mkdir()
        (attempt / m.STARTED).write_bytes(b"preserve")
        with self.assertRaisesRegex(ValueError, "attempt_exists_no_retry"):
            self.execute()
        self.assertEqual((attempt / m.STARTED).read_bytes(), b"preserve")

    def test_input_output_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, "input_output_overlap"):
            self.execute(attempt=self.checkpoint / "attempt")

    def test_missing_dependency_precedes_started_marker(self):
        from lumi import posttraining_checkpoint_import_torch_v2 as backend
        with patch.object(backend, "require_dependencies", side_effect=RuntimeError("missing")):
            with patch.object(m, "read_regular", side_effect=AssertionError("no fixture read")):
                with self.assertRaisesRegex(RuntimeError, "missing"):
                    self.execute()
        self.assertFalse((self.root / "attempt").exists())

    def test_started_backend_failure_preserved_and_not_retried(self):
        from lumi import posttraining_checkpoint_import_torch_v2 as backend
        before = m.tree_inventory(self.checkpoint)
        with patch.object(backend, "require_dependencies"), patch.object(
            backend, "convert_and_verify", side_effect=RuntimeError("fixture backend failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "fixture backend failure"):
                self.execute()
        self.assertTrue((self.root / "attempt" / m.STARTED).is_file())
        self.assertFalse((self.root / "attempt" / m.SUCCESS).exists())
        self.assertEqual(before, m.tree_inventory(self.checkpoint))
        with self.assertRaisesRegex(ValueError, "attempt_exists_no_retry"):
            self.execute()

    def test_model_evidence_rejects_false_digest_and_dtype(self):
        evidence = {"dtype": "bfloat16", "parameter_dtype": "bfloat16",
                    "buffer_dtype_policy": "configured_buffer_dtype_preserve_values",
                    "model_parameters": 10, "model_tensor_count": 2,
                    "source_tensor_sha256": "a" * 64, "cast_tensor_sha256": "b" * 64,
                    "reload_tensor_sha256": "b" * 64, "complete_keys_shapes_verified": True,
                    "hf_reload_tensor_equal": True, "no_forward_pass": True, "synthetic_only": True}
        m.validate_model_evidence(evidence, "bfloat16")
        for key, value in (("dtype", "float32"), ("model_parameters", True),
                           ("reload_tensor_sha256", "c" * 64), ("no_forward_pass", False),
                           ("synthetic_only", 1), ("model_parameters", 1_000_001)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.validate_model_evidence({**evidence, key: value}, "bfloat16")

    def test_exclusive_write_preserves_original(self):
        with self.assertRaises(FileExistsError):
            m.write_new(self.tokenizer, "tokenizer.json", b"overwrite")
        self.assertEqual(json.loads((self.tokenizer / "tokenizer.json").read_bytes()),
                         {"synthetic": True})

    def mock_conversion(self, config, checkpoint, output, dtype):
        output.mkdir(mode=0o700)
        m.write_new(output, "config.json", config.read_bytes())
        m.write_new(output, "model.safetensors", b"mock-container-not-real-tensors")
        return {"dtype": dtype, "parameter_dtype": dtype,
                "buffer_dtype_policy": "configured_buffer_dtype_preserve_values",
                "model_parameters": 10, "model_tensor_count": 2,
                "source_tensor_sha256": "a" * 64, "cast_tensor_sha256": "b" * 64,
                "reload_tensor_sha256": "b" * 64, "complete_keys_shapes_verified": True,
                "hf_reload_tensor_equal": True, "no_forward_pass": True, "synthetic_only": True}

    def mocked_package(self, convert=None, consumer_error=None):
        from lumi import posttraining_checkpoint_import_torch_v2 as backend
        def consume(root, *, expected_manifest_sha256, require_private):
            self.assertTrue(require_private)
            if consumer_error:
                raise consumer_error
            m.verify_closed_manifest(root, expected_manifest_sha256, private=True)
        runtime = types.ModuleType("lumi.posttraining_sft_runtime_v4")
        runtime._verify_checksum_manifest = consume
        # This explicitly mocked consumer tests call wiring only, not PyTorch or
        # actual frozen-worker compatibility. Genuine integration lives separately.
        with patch.object(backend, "require_dependencies"), patch.object(
            backend, "convert_and_verify", side_effect=convert or self.mock_conversion
        ), patch.dict(sys.modules, {"lumi.posttraining_sft_runtime_v4": runtime}):
            return self.execute()

    def test_mocked_complete_package_receipt_and_manifest_no_hash_cycle(self):
        before = m.tree_inventory(self.checkpoint)
        result = self.mocked_package()
        package = self.root / "attempt/checkpoint"
        receipt = m.verify_package(package, result["manifest_file_sha256"])
        self.assertFalse(receipt["real_4b_import"])
        self.assertFalse(receipt["production_eligible"])
        self.assertEqual(before, m.tree_inventory(self.checkpoint))
        self.assertTrue((self.root / "attempt" / m.SUCCESS).is_file())
        declared = m.parse_manifest((package / m.MANIFEST).read_bytes())
        self.assertIn(m.RECEIPT, declared)
        self.assertNotIn(m.MANIFEST, declared)
        self.assertEqual(len(receipt["payload_inventory"]), 5)

    def test_package_tampering_and_extra_file_rejected(self):
        result = self.mocked_package()
        package = self.root / "attempt/checkpoint"
        m.write_new(package, "extra", b"unexpected")
        with self.assertRaisesRegex(ValueError, "closed_inventory"):
            m.verify_package(package, result["manifest_file_sha256"])

    def test_receipt_internal_hash_rejected_even_with_new_manifest(self):
        self.mocked_package()
        package = self.root / "attempt/checkpoint"
        receipt = json.loads((package / m.RECEIPT).read_bytes())
        receipt["dtype"] = "float32"
        (package / m.RECEIPT).write_bytes(m.canonical(receipt))
        inventory = [x for x in m.tree_inventory(package) if x["path"] != m.MANIFEST]
        raw = "".join(f"{x['sha256']}  {x['path']}\n" for x in inventory).encode()
        (package / m.MANIFEST).write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "receipt_internal_hash"):
            m.verify_package(package, m.sha(raw))

    def test_source_mutation_after_backend_is_not_published_success(self):
        def mutate(config, checkpoint, output, dtype):
            result = self.mock_conversion(config, checkpoint, output, dtype)
            (checkpoint / "model/.metadata").write_bytes(b"mutated synthetic fixture")
            return result
        with self.assertRaisesRegex(ValueError, "source_mutated"):
            self.mocked_package(convert=mutate)
        self.assertFalse((self.root / "attempt" / m.SUCCESS).exists())

    def test_consumer_failure_cannot_publish_completion(self):
        with self.assertRaisesRegex(ValueError, "consumer fixture rejection"):
            self.mocked_package(consumer_error=ValueError("consumer fixture rejection"))
        self.assertTrue((self.root / "attempt" / m.STARTED).exists())
        self.assertFalse((self.root / "attempt" / m.SUCCESS).exists())
        with self.assertRaisesRegex(ValueError, "attempt_exists_no_retry"):
            self.execute()

    def test_parent_directory_symlink_rejected(self):
        (self.root / "link").symlink_to(self.tokenizer, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            m.read_regular(self.root / "link", "tokenizer.json")


if __name__ == "__main__":
    unittest.main()
