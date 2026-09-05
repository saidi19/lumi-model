"""Genuine tiny CPU DCP/HF tests for the parameter-only dtype successor.

Missing ML dependencies are errors, never skips. Every fixture is synthetic;
none of these tests authorizes or exercises real 4B import, training or GPU use.
"""

from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lumi.posttraining_checkpoint_import_torch_v2 import convert_and_verify, require_dependencies


class CheckpointImportTorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            require_dependencies()
            cls.torch = importlib.import_module("torch")
            cls.dcp = importlib.import_module("torch.distributed.checkpoint")
            cls.safetensors = importlib.import_module("safetensors.torch")
            importlib.import_module("transformers")
            cls.config_class = importlib.import_module("lumi.configuration_lumi").LumiConfig
            cls.model_class = importlib.import_module("lumi.modeling_lumi").LumiForCausalLM
        except (ImportError, RuntimeError) as error:
            raise RuntimeError(
                "PENDING: real tiny DCP/HF tests require torch, transformers and "
                "safetensors; no round-trip success is established"
            ) from error

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="lumi-import-synthetic-v2-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.config_path = self.root / "config.json"
        self.checkpoint_path = self.root / "checkpoint"
        self.output_path = self.root / "hf"
        self.payload = {
            "vocab_size": 17,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 16,
            "tie_word_embeddings": True,
        }
        self.config_path.write_text(json.dumps(self.payload), encoding="utf-8")

    def _checkpoint(self, mutate=None) -> None:
        torch = self.torch
        self.checkpoint_path.mkdir()
        config = self.config_class(**self.payload)
        config._attn_implementation = "eager"
        with torch.device("cpu"):
            model = self.model_class(config).float()
        with torch.no_grad():
            for index, parameter in enumerate(model.parameters()):
                values = torch.arange(parameter.numel(), dtype=torch.float32, device="cpu")
                parameter.copy_(((values.remainder(23) - 11) / 31 + index / 100).reshape(
                    parameter.shape
                ))
        self.buffer_names = set(dict(model.named_buffers()))
        state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        if mutate is not None:
            mutate(state)
        self.fixture_state = {name: tensor.clone() for name, tensor in state.items()}
        self.dcp.save(
            {"model": state},
            storage_writer=self.dcp.FileSystemWriter(self.checkpoint_path / "model"),
        )

    def _run(self, dtype="float32"):
        with mock.patch.object(self.model_class, "forward", side_effect=AssertionError("forward")):
            return convert_and_verify(
                self.config_path, self.checkpoint_path, self.output_path, dtype
            )

    def _assert_dtype_evidence(self, result, dtype) -> None:
        self.assertEqual(result["dtype"], dtype)
        self.assertEqual(result["parameter_dtype"], dtype)
        self.assertEqual(
            result["buffer_dtype_policy"], "configured_buffer_dtype_preserve_values"
        )
        self.assertEqual(result["cast_tensor_sha256"], result["reload_tensor_sha256"])

    def _assert_exported_buffer_policy(self) -> None:
        weights = list(self.output_path.glob("model*.safetensors"))
        self.assertEqual(len(weights), 1)
        exported = self.safetensors.load_file(weights[0], device="cpu")
        self.assertTrue(self.buffer_names)
        for name in self.buffer_names:
            self.assertEqual(exported[name].device.type, "cpu")
            self.assertEqual(exported[name].dtype, self.torch.float32)
            self.assertTrue(self.torch.equal(exported[name], self.fixture_state[name].float()))

    def _assert_roundtrip(self, dtype) -> None:
        self._checkpoint()
        result = self._run(dtype)
        self.assertTrue(result["synthetic_only"])
        self.assertTrue(result["no_forward_pass"])
        self.assertTrue(result["complete_keys_shapes_verified"])
        self.assertTrue(result["hf_reload_tensor_equal"])
        self._assert_dtype_evidence(result, dtype)
        self.assertGreater(result["model_parameters"], 0)
        self.assertLessEqual(result["model_parameters"], 1_000_000)
        if dtype == "float32":
            self.assertEqual(result["source_tensor_sha256"], result["reload_tensor_sha256"])
        else:
            self.assertNotEqual(result["source_tensor_sha256"], result["reload_tensor_sha256"])

    def test_real_float32_roundtrip(self) -> None:
        self._assert_roundtrip("float32")

    def test_real_bfloat16_roundtrip(self) -> None:
        self._assert_roundtrip("bfloat16")

    def test_real_bfloat16_source_roundtrip(self) -> None:
        self._checkpoint(lambda state: state.update({
            name: tensor.bfloat16() for name, tensor in state.items()
        }))
        self.assertTrue(all(tensor.dtype == self.torch.bfloat16
                            for tensor in self.fixture_state.values()))
        result = self._run("bfloat16")
        self._assert_dtype_evidence(result, "bfloat16")
        self._assert_exported_buffer_policy()

    def test_real_mixed_parameter_buffer_source_roundtrip(self) -> None:
        self._checkpoint(lambda state: state.update({
            name: tensor.bfloat16() for name, tensor in state.items()
            if name not in self.buffer_names
        }))
        for name, tensor in self.fixture_state.items():
            expected = self.torch.float32 if name in self.buffer_names else self.torch.bfloat16
            self.assertEqual(tensor.dtype, expected)
        result = self._run("bfloat16")
        self._assert_dtype_evidence(result, "bfloat16")
        self._assert_exported_buffer_policy()

    def test_bfloat16_export_preserves_original_float32_buffer_values(self) -> None:
        self._checkpoint()
        self.assertTrue(any(
            not self.torch.equal(self.fixture_state[name],
                                 self.fixture_state[name].bfloat16().float())
            for name in self.buffer_names
        ), "fixture must expose lossy whole-model BF16 buffer casting")
        result = self._run("bfloat16")
        self._assert_dtype_evidence(result, "bfloat16")
        self._assert_exported_buffer_policy()

    def test_missing_key_fails(self) -> None:
        self._checkpoint(lambda state: state.pop("lm_head.weight"))
        with self.assertRaisesRegex(ValueError, "key inventory"):
            self._run()
        self.assertFalse(self.output_path.exists())

    def test_extra_key_fails(self) -> None:
        self._checkpoint(lambda state: state.update(extra=self.torch.ones(1)))
        with self.assertRaisesRegex(ValueError, "key inventory"):
            self._run()
        self.assertFalse(self.output_path.exists())

    def test_shape_mismatch_fails(self) -> None:
        self._checkpoint(lambda state: state.update({"lm_head.weight": self.torch.ones(1)}))
        with self.assertRaisesRegex(ValueError, "shape inventory"):
            self._run()

    def test_unsupported_checkpoint_dtype_fails(self) -> None:
        self._checkpoint(
            lambda state: state.update({"lm_head.weight": state["lm_head.weight"].double()})
        )
        with self.assertRaisesRegex(ValueError, "dtype is unsupported"):
            self._run()

    def test_tied_alias_disagreement_fails(self) -> None:
        self._checkpoint(
            lambda state: state.update({"lm_head.weight": state["lm_head.weight"] + 1})
        )
        with self.assertRaisesRegex(ValueError, "tied aliases"):
            self._run()

    def test_nonfinite_checkpoint_fails(self) -> None:
        self._checkpoint(lambda state: state["lm_head.weight"].fill_(float("nan")))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self._run()

    def test_invalid_requested_dtype_fails_before_restore(self) -> None:
        with self.assertRaisesRegex(ValueError, "dtype must"):
            self._run("float16")
        self.assertFalse(self.output_path.exists())

    def test_parameter_cap_before_model_allocation(self) -> None:
        self.payload["vocab_size"] = 1_000_000
        self.config_path.write_text(json.dumps(self.payload), encoding="utf-8")
        with mock.patch.object(
            self.model_class, "__init__", side_effect=AssertionError("allocated")
        ):
            with self.assertRaisesRegex(ValueError, "parameter cap"):
                self._run()
        self.assertFalse(self.output_path.exists())

    def test_existing_output_is_never_overwritten(self) -> None:
        self.output_path.mkdir()
        sentinel = self.output_path / "sentinel"
        sentinel.write_bytes(b"unchanged")
        with self.assertRaises(FileExistsError):
            self._run()
        self.assertEqual(sentinel.read_bytes(), b"unchanged")

    def test_distributed_process_group_is_refused(self) -> None:
        with mock.patch.object(self.torch.distributed, "is_initialized", return_value=True):
            with self.assertRaisesRegex(ValueError, "no distributed process group"):
                self._run()

    def _adapter_roundtrip(self, dtype):
        from lumi import posttraining_checkpoint_import_v2 as adapter

        self.payload = self.config_class(**self.payload).to_dict()
        self.config_path.write_bytes(adapter.canonical(self.payload))
        self._checkpoint()
        checkpoint = self.checkpoint_path.resolve()
        config = self.config_path.resolve()
        config_sha = adapter.sha(config.read_bytes())
        state = {"checkpoint_schema_version": 1, "step": 1, "sequence_cursor": 2,
                 "tokens_seen": 32, "run_fingerprint": "a" * 64,
                 "run_specification": {"source_commit": "b" * 40,
                                       "config_sha256": config_sha,
                                       "distributed": {"world_size": 2}}}
        adapter.write_new(checkpoint, adapter.STATE, adapter.canonical(state))
        for rank in range(2):
            adapter.write_new(checkpoint, f"trainer-rank-{rank:05d}.pt", b"fixture-no-optimizer")

        def seal(root):
            inventory = adapter.tree_inventory(root)
            raw = "".join(f"{x['sha256']}  {x['path']}\n" for x in inventory).encode()
            adapter.write_new(root, adapter.MANIFEST, raw)
            return adapter.sha(raw)

        source_manifest = seal(checkpoint)
        tokenizer = self.root / "fixture-tokenizer"
        tokenizer.mkdir(mode=0o700)
        files = []
        for name in ("tokenizer.json", "tokenizer_config.json", "lumi_tokenizer_metadata.json"):
            raw = b'{"synthetic_fixture":true}'
            adapter.write_new(tokenizer, name, raw)
            files.append({"relative_path": f"tokenizer/{name}", "bytes": len(raw),
                          "sha256": adapter.sha(raw)})
        tokenizer_manifest = seal(tokenizer)
        base = {
            "schema_version": "lumi.posttraining-base-binding.v0.1", "checkpoint_stage": "base",
            "synthetic_only": True, "model_id": "lumi-synthetic-import-fixture",
            "run_fingerprint": "a" * 64, "source_commit": "b" * 40,
            "checkpoint": {"manifest_sha256": source_manifest, "step": 1,
                           "sequence_cursor": 2, "training_tokens": 32, "world_size": 2},
            "tokenizer": {"manifest_sha256": tokenizer_manifest, "files": files},
        }
        before = adapter.tree_inventory(checkpoint)
        token_before = adapter.tree_inventory(tokenizer)
        attempt = self.root / "adapter-attempt"
        with mock.patch.object(self.model_class, "forward", side_effect=AssertionError("forward")):
            result = adapter.run_synthetic_import(
                base=base, checkpoint=checkpoint, config=config, tokenizer=tokenizer,
                attempt=attempt, dtype=dtype,
            )
        receipt = adapter.verify_package(attempt / "checkpoint", result["manifest_file_sha256"])
        self.assertTrue(result["real_worker_checksum_verifier_passed"])
        self.assertFalse(result["real_4b_import"])
        self.assertFalse(receipt["production_eligible"])
        self._assert_dtype_evidence(receipt["model_evidence"], dtype)
        self.assertTrue((attempt / adapter.SUCCESS).is_file())
        self.assertEqual(before, adapter.tree_inventory(checkpoint))
        self.assertEqual(token_before, adapter.tree_inventory(tokenizer))

    def test_real_adapter_float32_and_worker_consumer(self):
        self._adapter_roundtrip("float32")

    def test_real_adapter_bfloat16_and_worker_consumer(self):
        self._adapter_roundtrip("bfloat16")


if __name__ == "__main__":
    unittest.main()
