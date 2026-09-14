"""Synthetic metadata and exact geometry arithmetic; no payloads or ML imports."""

from __future__ import annotations

import ast
import copy
import unittest
from pathlib import Path

from lumi import posttraining_checkpoint_import_plan_v1 as m
from lumi.posttraining_final_base_metadata_v1 import FinalBaseMetadataError, canonical, sha


def documents():
    config = {
        "vocab_size": 17, "hidden_size": 8, "intermediate_size": 16,
        "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 4, "max_position_embeddings": 16, "rope_theta": 10000.0,
        "rms_norm_eps": 0.00001, "attention_bias": False, "tie_word_embeddings": True,
        "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2, "doc_token_id": 3,
        "model_type": "lumi",
    }
    config_raw = canonical(config)
    state = {
        "checkpoint_schema_version": 1, "step": 4, "sequence_cursor": 8, "tokens_seen": 256,
        "run_fingerprint": "a" * 64,
        "run_specification": {"source_commit": "b" * 40, "config_sha256": sha(config_raw),
                              "distributed": {"world_size": 2}},
    }
    state_raw = canonical(state)
    files = [
        {"path": "distributed-training-state.json", "bytes": len(state_raw),
         "sha256": sha(state_raw)},
        {"path": "model/.metadata", "bytes": 100, "sha256": "c" * 64},
        {"path": "model/__0_0.distcp", "bytes": 1024, "sha256": "d" * 64},
        {"path": "trainer-rank-00000.pt", "bytes": 64, "sha256": "e" * 64},
        {"path": "trainer-rank-00001.pt", "bytes": 64, "sha256": "f" * 64},
    ]
    manifest_raw = "".join(f"{x['sha256']}  {x['path']}\n" for x in files).encode()
    files.append({"path": "manifest.sha256", "bytes": len(manifest_raw),
                  "sha256": sha(manifest_raw)})
    handoff = {
        "checkpoint": {"directory": "checkpoint", "step": 4, "sequence_cursor": 8,
                       "tokens_seen": 256},
        "files": files, "gate": "lumi_4b_checkpoint_handoff_v0.1",
        "run_fingerprint": "a" * 64, "schema_version": 1, "source_commit": "b" * 40,
        "status": "sealed", "world_size": 2,
    }
    handoff["handoff_sha256"] = sha(canonical(handoff))
    tokenfiles = [
        {"relative_path": f"tokenizer/{name}", "bytes": i + 1, "sha256": str(i + 1) * 64}
        for i, name in enumerate(("lumi_tokenizer_metadata.json", "tokenizer.json",
                                  "tokenizer_config.json"))
    ]
    base = {
        "schema_version": "lumi.posttraining-base-binding.v0.1", "checkpoint_stage": "base",
        "model_id": "lumi-4b-base", "run_fingerprint": "a" * 64, "source_commit": "b" * 40,
        "checkpoint": {"step": 4, "sequence_cursor": 8, "training_tokens": 256,
                       "world_size": 2, "manifest_sha256": sha(manifest_raw)},
        "tokenizer": {"files": tokenfiles, "manifest_sha256": sha(canonical(tokenfiles))},
        "handoff": {"sha256": handoff["handoff_sha256"],
                    "file_sha256": sha(canonical(handoff))},
    }
    return [canonical(base), manifest_raw, state_raw, config_raw, canonical(handoff)]


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.docs = documents()

    def run_plan(self, dtype="bfloat16"):
        return m.build_import_plan(*self.docs, expected_base_sha256=sha(self.docs[0]), dtype=dtype)

    def change_handoff(self, change, internal=True, external=True):
        handoff = m.json_object(self.docs[4])
        change(handoff)
        if internal:
            handoff["handoff_sha256"] = sha(canonical({
                k: v for k, v in handoff.items() if k != "handoff_sha256"
            }))
        self.docs[4] = canonical(handoff)
        if external:
            base = m.json_object(self.docs[0])
            base["handoff"] = {"sha256": handoff["handoff_sha256"],
                               "file_sha256": sha(self.docs[4])}
            self.docs[0] = canonical(base)

    def reject(self, code):
        with self.assertRaises(m.ImportPlanError) as caught:
            self.run_plan()
        self.assertEqual(caught.exception.args, (code,))

    def test_plan_binds_every_checkpoint_and_tokenizer_declaration(self):
        result = self.run_plan()
        self.assertEqual(len(result["checkpoint_inventory"]), 6)
        self.assertEqual(len(result["tokenizer_inventory"]), 3)
        bounds = result["resource_bounds"]
        count = sum(x["bytes"] for x in result["checkpoint_inventory"])
        self.assertEqual(bounds["checkpoint_inventory_bytes"], count)
        self.assertEqual(bounds["checkpoint_audit_read_bytes"], 2 * count)
        self.assertEqual(bounds["tokenizer_audit_read_bytes"], 12)
        self.assertTrue(all(result[flag] is False for flag in m.FALSE_FLAGS))
        self.assertFalse(bounds["backend_payload_read_cap_verified"])
        self.assertFalse(bounds["os_memory_time_disk_limits_enforced"])
        self.assertEqual(result["plan_sha256"], sha(canonical({
            k: v for k, v in result.items() if k != "plan_sha256"
        })))

    def test_output_cap_contains_alias_and_control_reserves(self):
        result = self.run_plan()
        bounds = result["resource_bounds"]
        self.assertEqual(bounds["maximum_backend_output_bytes"],
                         bounds["cast_tensor_bytes_including_aliases"] + m.MIB)
        self.assertEqual(bounds["maximum_private_output_write_bytes"],
                         bounds["maximum_backend_output_bytes"] + 6 + 2 * m.MIB)
        self.assertGreaterEqual(bounds["minimum_host_ram_planning_bytes"],
                                2 * bounds["maximum_projected_live_tensor_bytes"] + 4 * m.GIB)

    def test_fp32_does_not_change_buffer_bytes_or_scope(self):
        bf16, fp32 = self.run_plan(), self.run_plan("float32")
        self.assertEqual(bf16["buffer_elements"], fp32["buffer_elements"])
        b = fp32["resource_bounds"]
        self.assertEqual(b["source_fp32_tensor_bytes_including_aliases"],
                         b["cast_tensor_bytes_including_aliases"])
        self.assertGreater(b["maximum_backend_output_bytes"],
                           bf16["resource_bounds"]["maximum_backend_output_bytes"])

    def test_wrong_base_pin_is_rejected_before_planning(self):
        with self.assertRaisesRegex(FinalBaseMetadataError, "base_binding_mismatch"):
            m.build_import_plan(*self.docs, expected_base_sha256="0" * 64, dtype="bfloat16")

    def test_wrong_tokenizer_checksum_semantics_rejected(self):
        base = m.json_object(self.docs[0])
        base["tokenizer"]["manifest_sha256"] = "0" * 64
        self.docs[0] = canonical(base)
        with self.assertRaisesRegex(FinalBaseMetadataError, "tokenizer_manifest_binding"):
            self.run_plan()

    def test_handoff_bytes_cap(self):
        self.docs[4] = b" " * (m.MAX_HANDOFF_BYTES + 1)
        self.reject("handoff_input_cap")

    def test_handoff_external_pin(self):
        self.change_handoff(lambda h: h.update(status="changed"), external=False)
        self.reject("handoff_binding")

    def test_handoff_internal_pin(self):
        self.change_handoff(lambda h: h.update(world_size=3), internal=False)
        self.reject("handoff_internal_hash")

    def test_handoff_schema(self):
        for key, value in (("schema_version", True), ("gate", "wrong"), ("status", "draft")):
            with self.subTest(key=key):
                self.docs = documents()
                self.change_handoff(lambda h, k=key, v=value: h.update({k: v}))
                self.reject("handoff_schema")

    def test_handoff_lineage(self):
        for key, value in (("source_commit", "c" * 40), ("run_fingerprint", "d" * 64),
                           ("world_size", True), ("world_size", 3)):
            with self.subTest(key=key):
                self.docs = documents()
                self.change_handoff(lambda h, k=key, v=value: h.update({k: v}))
                self.reject("handoff_lineage")

    def test_checkpoint_cursor_bool_and_mismatch(self):
        for key, value in (("step", True), ("sequence_cursor", 10), ("tokens_seen", 999)):
            with self.subTest(key=key):
                self.docs = documents()
                self.change_handoff(lambda h, k=key, v=value: h["checkpoint"].update({k: v}))
                self.reject("handoff_lineage")

    def test_whole_inventory_cannot_omit_trainer(self):
        self.change_handoff(lambda h: h["files"].pop(3))
        self.reject("handoff_closed_manifest")

    def test_extra_file_rejected(self):
        self.change_handoff(lambda h: h["files"].append({
            "path": "extra", "bytes": 1, "sha256": "a" * 64
        }))
        self.reject("handoff_closed_manifest")

    def test_duplicate_inventory_rejected(self):
        self.change_handoff(lambda h: h["files"].append(dict(h["files"][0])))
        self.reject("handoff_inventory")

    def test_inventory_scalar_and_size_types(self):
        for value in (0, -1, True, "1"):
            with self.subTest(value=value):
                self.docs = documents()
                self.change_handoff(lambda h, v=value: h["files"][1].update(bytes=v))
                self.reject("handoff_inventory")

    def test_manifest_state_size_contract(self):
        self.change_handoff(lambda h: h["files"][0].update(bytes=1))
        self.reject("handoff_closed_manifest")

    def test_inventory_total_cap(self):
        self.change_handoff(lambda h: h["files"][1].update(bytes=m.MAX_INVENTORY_BYTES))
        self.reject("handoff_inventory_cap")

    def test_inventory_count_cap(self):
        self.change_handoff(lambda h: h.update(files=[{}] * 130))
        self.reject("handoff_inventory_cap")

    def test_dtype_closed(self):
        for value in ("float16", "auto", None):
            with self.subTest(value=value), self.assertRaisesRegex(m.ImportPlanError,
                                                                   "unsupported_dtype"):
                self.run_plan(value)


class AnalyticalInventoryTests(unittest.TestCase):
    def config(self):
        return {
            "vocab_size": 65536, "hidden_size": 3072, "intermediate_size": 8384,
            "num_hidden_layers": 36, "num_attention_heads": 64, "num_key_value_heads": 8,
            "head_dim": 64, "attention_bias": False, "tie_word_embeddings": True,
        }

    def test_actual_4b_geometry_without_allocating_model(self):
        items = m.tensor_inventory(self.config())
        self.assertEqual(len(items), 363)
        self.assertEqual(sum(x["elements"] for x in items
                             if x["role"] == "parameter" and "alias_of" not in x), 4002376704)
        self.assertEqual(sum(x["elements"] for x in items if x["role"] == "buffer"), 1152)
        table = {x["name"]: x for x in items}
        self.assertEqual(table["model.layers.0.self_attn.q_proj.weight"]["shape"], [4096, 3072])
        self.assertEqual(table["model.layers.0.self_attn.k_proj.weight"]["shape"], [512, 3072])
        self.assertEqual(table["model.layers.35.mlp.down_proj.weight"]["shape"], [3072, 8384])
        self.assertEqual(table["lm_head.weight"]["alias_of"], "model.embed_tokens.weight")

    def test_original_parameter_arithmetic_independent_ast(self):
        source = Path(__file__).resolve().parents[1] / "src/lumi/parameter_count.py"
        tree = ast.parse(source.read_text())
        definitions = [node for node in tree.body
                       if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
        from dataclasses import dataclass
        from types import SimpleNamespace
        namespace = {"dataclass": dataclass, "LumiConfig": object}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
        for bias in (False, True):
            for tied in (False, True):
                config = self.config() | {"attention_bias": bias, "tie_word_embeddings": tied}
                original = namespace["parameter_breakdown"](SimpleNamespace(
                    **config, query_projection_size=4096, key_value_projection_size=512
                )).total
                observed = sum(x["elements"] for x in m.tensor_inventory(config)
                               if x["role"] == "parameter" and "alias_of" not in x)
                self.assertEqual(original, observed)

    def test_untied_and_bias_inventory(self):
        config = self.config() | {"attention_bias": True, "tie_word_embeddings": False}
        items = m.tensor_inventory(config)
        self.assertEqual(len(items), 363 + 36 * 4)
        self.assertFalse(any("alias_of" in x for x in items))
        self.assertEqual(sum(x["role"] == "buffer" for x in items), 36)

    def test_invalid_geometry_rejected(self):
        for key, value in (("head_dim", 3), ("num_hidden_layers", 129),
                           ("hidden_size", True), ("num_key_value_heads", 7),
                           ("attention_bias", 0), ("tie_word_embeddings", "yes")):
            with self.subTest(key=key), self.assertRaisesRegex(m.ImportPlanError,
                                                              "unsupported_geometry"):
                m.tensor_inventory(self.config() | {key: value})

    def test_parameter_bound_before_allocation(self):
        with self.assertRaisesRegex(m.ImportPlanError, "parameter_cap"):
            m.tensor_inventory(self.config() | {"hidden_size": 1000000})

    def test_inputs_preserved(self):
        config = self.config()
        saved = copy.deepcopy(config)
        m.tensor_inventory(config)
        self.assertEqual(config, saved)


if __name__ == "__main__":
    unittest.main()
