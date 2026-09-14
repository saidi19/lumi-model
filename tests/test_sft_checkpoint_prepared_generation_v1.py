"""Fresh tiny byte fixtures only; no torch, native Linux, real input or network."""

from __future__ import annotations

import copy
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_lumi_prepared_generation_v1_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
m = importlib.import_module(PACKAGE + ".posttraining_checkpoint_prepared_generation_v1")
REAL_RUNTIME = m._runtime


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lumi-generated-test-")
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name).resolve()
        self.root = self.parent / "fresh"
        self.root.mkdir(mode=0o700)
        self.enterContext(patch.object(m, "_ATTEMPTED", False))
        self.enterContext(patch.object(m, "_runtime"))
        self.generator = self.enterContext(patch.object(m.fixture, "_generate",
                                                        side_effect=self.generate_bytes))

    def generate_bytes(self, checkpoint, budget):
        budget()
        model = checkpoint / "model"
        model.mkdir(mode=0o700)
        m.fixture._write(model / ".metadata", b"opaque freshly generated metadata")
        m.fixture._write(model / "__0_0.distcp", b"opaque freshly generated tensor")

    def invoke(self, **kwargs):
        return m.generate_prepared_inputs(self.root, generation_nonce_sha256="9" * 64, **kwargs)

    def test_generation_only_uses_frozen_metadata_and_reads_its_own_exact_outputs(self):
        result = self.invoke()
        self.generator.assert_called_once()
        self.assertEqual(set(result), {"metadata", "generated_inputs", "receipt"})
        receipt = result["receipt"]
        self.assertEqual(receipt["model_parameters"], 736)
        self.assertEqual(receipt["generation_passes"], 1)
        self.assertEqual(receipt["import_passes"], 0)
        self.assertEqual(receipt["generated_input_bytes"], sum(len(raw)
                         for rows in result["generated_inputs"].values() for raw in rows.values()))
        self.assertEqual(receipt["generated_readback_bytes"], receipt["generated_input_bytes"])
        self.assertEqual((self.root / "attempt.started").read_bytes(), m.MARKER)
        self.assertTrue(receipt["parent_containment_required"])
        self.assertFalse(receipt["cumulative_generation_writes_metered"])
        self.assertFalse(receipt["training_authorized"])
        self.assertFalse(receipt["production_eligible"])
        self.assertNotIn(b"opaque", m.canonical(receipt))
        self.assertNotIn(str(self.root).encode(), m.canonical(receipt))

    def test_no_retry_after_success(self):
        self.invoke()
        with patch.object(m, "_runtime", REAL_RUNTIME), patch.dict(
                sys.modules, {"torch": types.ModuleType("torch")}):
            with self.assertRaisesRegex(m.PreparedGenerationError, "already_attempted"):
                self.invoke()
        self.generator.assert_called_once()

    def test_failure_preserves_marker_and_no_retry(self):
        self.generator.side_effect = RuntimeError("private exception text")
        with self.assertRaisesRegex(m.PreparedGenerationError, "^generation_failed$"):
            self.invoke()
        self.assertEqual((self.root / "attempt.started").read_bytes(), m.MARKER)
        with self.assertRaisesRegex(m.PreparedGenerationError, "already_attempted"):
            self.invoke()
        self.generator.assert_called_once()

    def test_invalid_nonce_before_mutation(self):
        for value in (True, "0" * 64, "A" * 64, "unbound", None):
            with self.subTest(value=value), self.assertRaisesRegex(
                    m.PreparedGenerationError, "arguments_rejected"):
                m.generate_prepared_inputs(self.root, generation_nonce_sha256=value)
        self.assertFalse(m._ATTEMPTED)
        self.assertEqual(list(self.root.iterdir()), [])
        self.generator.assert_not_called()

    def test_config_input_and_approval_arguments_are_not_exposed(self):
        for key, value in (("config", {}), ("checkpoint", "/real/checkpoint"),
                           ("approved", True), ("check_budget", lambda: None)):
            with self.subTest(key=key), self.assertRaises(TypeError):
                self.invoke(**{key: value})
        self.generator.assert_not_called()

    def test_nonempty_root_rejected_without_consuming_attempt(self):
        m.fixture._write(self.root / "existing", b"unrelated generated sentinel")
        with self.assertRaisesRegex(m.PreparedGenerationError, "arguments_rejected"):
            self.invoke()
        self.assertFalse(m._ATTEMPTED)
        self.generator.assert_not_called()

    def test_symlink_root_rejected_without_reuse(self):
        selected = self.parent / "link"
        selected.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(m.PreparedGenerationError, "arguments_rejected"):
            m.generate_prepared_inputs(selected, generation_nonce_sha256="9" * 64)
        self.generator.assert_not_called()

    def test_unexpected_generated_file_rejected(self):
        original = m.fixture._metadata

        def add_file(checkpoint, tokenizer, budget):
            result = original(checkpoint, tokenizer, budget)
            m.fixture._write(checkpoint / "unlisted", b"unexpected generated file")
            return result

        with patch.object(m.fixture, "_metadata", side_effect=add_file), self.assertRaisesRegex(
                m.PreparedGenerationError, "inventory_rejected"):
            self.invoke()

    def test_changed_generated_bytes_rejected(self):
        original = m.fixture._metadata

        def change(checkpoint, tokenizer, budget):
            result = original(checkpoint, tokenizer, budget)
            with (checkpoint / "model/.metadata").open("r+b") as stream:
                stream.write(b"changed")
            return result

        with patch.object(m.fixture, "_metadata", side_effect=change), self.assertRaisesRegex(
                m.PreparedGenerationError, "inventory_rejected"):
            self.invoke()

    def test_symlink_generated_input_rejected_before_content_read(self):
        original = m.fixture._metadata

        def change(checkpoint, tokenizer, budget):
            result = original(checkpoint, tokenizer, budget)
            path = checkpoint / "model/.metadata"
            path.rename(checkpoint / "moved")
            path.symlink_to(checkpoint / "moved")
            return result

        with patch.object(m.fixture, "_metadata", side_effect=change), self.assertRaises(
                m.PreparedGenerationError):
            self.invoke()

    def test_generated_mode_rejected_without_normalizing_after_metadata(self):
        original = m.fixture._metadata

        def change(checkpoint, tokenizer, budget):
            result = original(checkpoint, tokenizer, budget)
            (checkpoint / "model/.metadata").chmod(0o644)
            return result

        with patch.object(m.fixture, "_metadata", side_effect=change), self.assertRaisesRegex(
                m.PreparedGenerationError, "inventory_rejected"):
            self.invoke()

    def test_short_read_is_terminal_and_not_resumed(self):
        with patch.object(m.os, "read", return_value=b""), self.assertRaises(
                m.PreparedGenerationError):
            self.invoke()
        self.assertTrue(m._ATTEMPTED)

    def test_budget_stops_before_first_write_and_does_not_retry(self):
        with patch.object(m.time, "monotonic", side_effect=(0, 30)), self.assertRaisesRegex(
                m.PreparedGenerationError, "deadline_exhausted"):
            self.invoke()
        self.generator.assert_not_called()
        self.assertTrue(m._ATTEMPTED)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_real_runtime_gate_rejects_unsupported_host_before_path_access(self):
        with patch.object(m, "_runtime", REAL_RUNTIME), patch.object(
                m, "sys", types.SimpleNamespace(platform="unsupported")):
            with patch.object(m.fixture, "_root") as root_check, self.assertRaisesRegex(
                    m.PreparedGenerationError, "runtime_rejected"):
                self.invoke()
        root_check.assert_not_called()

    def test_deadline_inside_frozen_tick_keeps_precise_error_code(self):
        now = [0]

        def expires(_checkpoint, budget):
            now[0] = 31
            m.fixture._tick(budget)

        self.generator.side_effect = expires
        with patch.object(m.time, "monotonic", side_effect=lambda: now[0]):
            with self.assertRaisesRegex(m.PreparedGenerationError, "^deadline_exhausted$"):
                self.invoke()
        self.assertTrue(m._ATTEMPTED)
        self.assertEqual((self.root / "attempt.started").read_bytes(), m.MARKER)

    def test_prepared_inputs_feed_real_pure_control_builder(self):
        # Included once the independently implemented pure control builder exists.
        builder = importlib.import_module(PACKAGE + ".posttraining_checkpoint_prepared_controls_v1")
        result = self.invoke()
        before = copy.deepcopy(result)
        proposal = builder.build_prepared_controls(metadata=result["metadata"],
            generation_nonce_sha256="9" * 64, entry_sha256="a" * 64,
            source_manifest_sha256="b" * 64, runtime_sources={
                "configuration_lumi.py": b"# generated config\n",
                "modeling_lumi.py": b"# generated model\n"})
        self.assertEqual(result, before)
        self.assertEqual(proposal["plan"]["model_parameters"], 736)
        for role, field in (("checkpoint", "checkpoint_inventory"),
                            ("tokenizer", "tokenizer_inventory")):
            self.assertEqual({row["path"]: row["bytes"] for row in proposal["plan"][field]},
                             {name: len(raw)
                              for name, raw in result["generated_inputs"][role].items()})


if __name__ == "__main__":
    unittest.main()
