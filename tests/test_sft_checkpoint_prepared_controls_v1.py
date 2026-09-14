"""Generated JSON controls and real pure validators only, no native ML or payload I/O."""

from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_lumi_prepared_controls_tests_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
before_import = set(sys.modules)
m = importlib.import_module(PACKAGE + ".posttraining_checkpoint_prepared_controls_v1")
IMPORT_ADDITIONS = set(sys.modules) - before_import
runner = importlib.import_module(PACKAGE + ".posttraining_checkpoint_import_runner_v3")
entry_spec = importlib.util.spec_from_file_location("_prepared_controls_frozen_entry_test_v1",
    ROOT / "scripts/run_checkpoint_prepared_fixture_v1.py")
entry = importlib.util.module_from_spec(entry_spec)
entry_spec.loader.exec_module(entry)
PRIVATE = "private arbitrary source or exception text"


def generated_metadata():
    """Reuse just the existing generated-control function, not its test imports."""
    tree = ast.parse((ROOT / "tests/test_sft_checkpoint_import_plan_v1.py").read_bytes())
    function = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "documents")
    namespace = {"canonical": m.canonical, "sha": m.sha}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "generated-metadata", "exec"),
         namespace)
    values = namespace["documents"]()
    base = m.json_object(values[0])
    base["synthetic_fixture_only"] = True
    return (m.canonical(base), *values[1:])


def arguments():
    return {"metadata": generated_metadata(), "generation_nonce_sha256": "9" * 64,
            "entry_sha256": "a" * 64, "source_manifest_sha256": "b" * 64,
            "runtime_sources": {"configuration_lumi.py": b"# generated configuration\n",
                                "modeling_lumi.py": b"# generated modeling\n"}}


def validate(result, options, **changes):
    values = {
        "expected_control_sha256": m.sha(result["control_raw"]),
        "expected_entry_sha256": options["entry_sha256"],
        "expected_source_manifest_sha256": options["source_manifest_sha256"],
        "expected_generation_nonce_sha256": options["generation_nonce_sha256"],
        "runtime_sources": options["runtime_sources"],
    }
    return m.validate_prepared_controls(result["control_raw"], result["bodies"],
                                       **(values | changes))


def rebind_control(result):
    control = m.json_object(result["control_raw"])
    control["files"] = [{"path": name, "bytes": len(result["bodies"][name]),
                         "sha256": m.sha(result["bodies"][name])} for name in m.CONTROL_NAMES]
    result["control_raw"] = m.canonical(control)


class PreparedControlsTests(unittest.TestCase):
    def setUp(self):
        self.options = arguments()

    def build(self, **changes):
        return m.build_prepared_controls(**(self.options | changes))

    def test_build_and_frozen_entry_roundtrip_have_no_filesystem_operations(self):
        with patch("builtins.open") as opened, patch.object(os, "open") as os_open:
            result = self.build()
            admitted = validate(result, self.options)
            control = entry.validate_control(result["control_raw"],
                expected_sha256=result["bindings"]["control_file_sha256"],
                entry_sha256=self.options["entry_sha256"],
                source_manifest_sha256=self.options["source_manifest_sha256"])
            args = entry.validate_bundle(control, result["bodies"])
            files = {"lumi/" + name: raw for name, raw in self.options["runtime_sources"].items()}
            plan = entry.validate_runner_controls(runner, args, files)
        opened.assert_not_called()
        os_open.assert_not_called()
        self.assertEqual(plan, result["plan"])
        self.assertEqual(admitted["plan"], plan)
        self.assertEqual(plan["model_parameters"], 736)
        self.assertEqual(result["counts"]["control_files"], 9)
        self.assertEqual(result["counts"]["generated_input_files"], 9)
        self.assertEqual(result["status"], m.STATUS)

    def test_exact_closed_result_fields_and_every_nonclaim_false(self):
        result = self.build()
        self.assertEqual(set(result), {"schema", "status", "control_raw", "bodies", "plan",
            "envelope", "kernel_io_contract", "bindings", "counts"} | set(m.FALSE_FLAGS))
        self.assertTrue(all(result[key] is False for key in m.FALSE_FLAGS))
        self.assertEqual(result["counts"]["model_payload_bytes_read"], 0)
        self.assertEqual(set(result["bodies"]), set(entry.CONTROL_NAMES))
        control = m.json_object(result["control_raw"])
        self.assertEqual([row["path"] for row in control["files"]], list(entry.CONTROL_NAMES))
        self.assertEqual(m.FIXED_CONFIG_RAW, entry.canonical(entry.CONFIG))

    def test_only_new_base_changes_all_original_inputs_remain_unchanged(self):
        before = copy.deepcopy(self.options)
        result = self.build()
        self.assertEqual(self.options, before)
        new_base = m.json_object(result["bodies"]["base.json"])
        marker = new_base.pop("prepared_fixture_marker")
        self.assertEqual(new_base, m.json_object(before["metadata"][0]))
        self.assertEqual(marker, {"schema": entry.BASE_MARKER_SCHEMA,
            "generator": "fixed-tiny-config-v1", "generation_nonce_sha256": "9" * 64})
        for name, raw in zip(m.METADATA_NAMES[1:], before["metadata"][1:], strict=True):
            self.assertEqual(result["bodies"][name], raw)
        self.assertEqual(result["bindings"]["original_base_sha256"],
                         m.sha(before["metadata"][0]))
        self.assertNotEqual(result["bindings"]["prepared_base_sha256"],
                            result["bindings"]["original_base_sha256"])

    def test_objects_are_detached_between_results_inputs_and_raw_documents(self):
        result, second = self.build(), self.build()
        snapshot = copy.deepcopy(second)
        result["plan"]["metadata"].clear()
        result["envelope"]["runtime_sources"][0]["bytes"] = 1
        result["kernel_io_contract"]["hf_io_contract"].clear()
        result["bodies"]["config.json"] = b"changed"
        self.assertEqual(second, snapshot)
        self.assertEqual(self.options, arguments())
        self.assertEqual(m.json_object(second["bodies"]["plan.json"]), second["plan"])

    def test_nonce_change_rebinds_base_plan_envelope_io_and_control(self):
        first, second = self.build(), self.build(generation_nonce_sha256="8" * 64)
        for name in ("base.json", "plan.json", "envelope.json", "kernel-io.json"):
            self.assertNotEqual(first["bodies"][name], second["bodies"][name])
        for name in m.METADATA_NAMES[1:]:
            self.assertEqual(first["bodies"][name], second["bodies"][name])
        self.assertNotEqual(first["control_raw"], second["control_raw"])

    def test_fixed_dcp_headroom_and_every_frozen_envelope_floor(self):
        result = self.build()
        e, p = result["envelope"], result["plan"]
        b = p["resource_bounds"]
        self.assertEqual(e["dcp_payload_read_cap_bytes"], 8 * 1024**2)
        self.assertEqual(e["dcp_read_cap_bytes"], b["checkpoint_inventory_bytes"] + 8 * 1024**2)
        self.assertEqual(e["private_write_cap_bytes"], b["maximum_private_output_write_bytes"])
        self.assertLessEqual(e["coordinator_read_cap_bytes"], entry.MAXIMUM_READ_BYTES)
        self.assertLessEqual(e["private_write_cap_bytes"], entry.MAXIMUM_WRITE_BYTES)
        self.assertLessEqual(e["maximum_seconds"], entry.MAXIMUM_SECONDS)
        self.assertEqual(e["worker_memory_max_bytes"], b["minimum_host_ram_planning_bytes"])
        self.assertFalse(result["host_capacity_verified"])
        self.assertFalse(result["dcp_native_serializer_overhead_measured"])

    def test_hash_and_count_bindings_match_each_returned_body(self):
        result = self.build()
        for row in m.json_object(result["control_raw"])["files"]:
            raw = result["bodies"][row["path"]]
            self.assertEqual((row["bytes"], row["sha256"]), (len(raw), m.sha(raw)))
        self.assertEqual(result["counts"]["control_file_bytes"],
            len(result["control_raw"]) + sum(map(len, result["bodies"].values())))
        for stem, name in (("plan", "plan.json"), ("envelope", "envelope.json"),
                           ("kernel_io", "kernel-io.json")):
            self.assertEqual(result["bindings"][stem + "_file_sha256"],
                             m.sha(result["bodies"][name]))

    def test_metadata_tuple_exact_bytes_and_caps_are_required(self):
        original = self.options["metadata"]
        for bad in (list(original), original[:4], (*original, b"extra"),
                    (bytearray(original[0]), *original[1:]), (b"", *original[1:]),
                    (b"x" * 16385, *original[1:])):
            with self.subTest(kind=type(bad)), self.assertRaises(m.PreparedControlsError):
                self.build(metadata=bad)

    def test_all_pins_must_be_nonzero_sha256_lowercase_strings(self):
        for field in ("generation_nonce_sha256", "entry_sha256", "source_manifest_sha256"):
            for value in ("0" * 64, "A" * 64, "c" * 63, True, None):
                with self.subTest(field=field, value=value), \
                        self.assertRaisesRegex(m.PreparedControlsError, "arguments_rejected"):
                    self.build(**{field: value})

    def test_old_false_or_missing_synthetic_marker_is_not_upgraded(self):
        for value in (False, 1, None):
            base = m.json_object(self.options["metadata"][0])
            if value is None:
                del base["synthetic_fixture_only"]
            else:
                base["synthetic_fixture_only"] = value
            with self.subTest(value=value), \
                    self.assertRaisesRegex(m.PreparedControlsError, "synthetic_scope_rejected"):
                self.build(metadata=(m.canonical(base), *self.options["metadata"][1:]))

    def test_prepared_base_cannot_be_reused_or_relabelled(self):
        result = self.build()
        prepared = tuple(result["bodies"][name] for name in m.METADATA_NAMES)
        with self.assertRaisesRegex(m.PreparedControlsError, "synthetic_scope_rejected"):
            self.build(metadata=prepared, generation_nonce_sha256="8" * 64)

    def test_approval_base_field_or_extra_runtime_source_is_rejected(self):
        base = m.json_object(self.options["metadata"][0])
        base["execution_permitted"] = True
        with self.assertRaisesRegex(m.PreparedControlsError, "synthetic_scope_rejected"):
            self.build(metadata=(m.canonical(base), *self.options["metadata"][1:]))
        with self.assertRaisesRegex(m.PreparedControlsError, "source_contract_rejected"):
            self.build(runtime_sources=self.options["runtime_sources"] | {"other.py": b"x"})
        with self.assertRaises(TypeError):
            self.build(approved=True)

    def test_different_tiny_geometry_and_lineage_are_rejected(self):
        config = m.json_object(self.options["metadata"][3])
        config["hidden_size"] = 9
        changed = list(self.options["metadata"])
        changed[3] = m.canonical(config)
        with self.assertRaisesRegex(m.PreparedControlsError, "synthetic_scope_rejected"):
            self.build(metadata=tuple(changed))
        base = m.json_object(self.options["metadata"][0])
        base["run_fingerprint"] = "c" * 64
        with self.assertRaisesRegex(m.PreparedControlsError, "synthetic_scope_rejected"):
            self.build(metadata=(m.canonical(base), *self.options["metadata"][1:]))

    def test_generated_inventory_bytes_cannot_exceed_tiny_cap(self):
        base = m.json_object(self.options["metadata"][0])
        base["tokenizer"]["files"][0]["bytes"] = m.MAXIMUM_INPUT_BYTES
        base["tokenizer"]["manifest_sha256"] = m.sha(m.canonical(base["tokenizer"]["files"]))
        with self.assertRaisesRegex(m.PreparedControlsError, "synthetic_scope_rejected"):
            self.build(metadata=(m.canonical(base), *self.options["metadata"][1:]))

    def test_generated_inventory_file_count_cannot_exceed_tiny_cap(self):
        values = list(self.options["metadata"])
        base, handoff = m.json_object(values[0]), m.json_object(values[4])
        rows = [row for row in handoff["files"] if row["path"] != "manifest.sha256"]
        rows.extend({"path": f"model/__{index}_0.distcp", "bytes": 1, "sha256": "c" * 64}
                    for index in range(1, 28))
        values[1] = "".join(f"{row['sha256']}  {row['path']}\n" for row in rows).encode("ascii")
        handoff["files"] = rows + [{"path": "manifest.sha256", "bytes": len(values[1]),
                                     "sha256": m.sha(values[1])}]
        del handoff["handoff_sha256"]
        handoff["handoff_sha256"] = m.sha(m.canonical(handoff))
        values[4] = m.canonical(handoff)
        base["checkpoint"]["manifest_sha256"] = m.sha(values[1])
        base["handoff"] = {"sha256": handoff["handoff_sha256"],
                           "file_sha256": m.sha(values[4])}
        values[0] = m.canonical(base)
        with self.assertRaisesRegex(m.PreparedControlsError, "synthetic_scope_rejected"):
            self.build(metadata=tuple(values))

    def test_runtime_sources_are_exact_bounded_bytes_and_rows_are_derived(self):
        for sources in ({}, {"configuration_lumi.py": b"x"},
                        dict(self.options["runtime_sources"], configuration_lumi=b"x")):
            with self.subTest(sources=set(sources)), \
                    self.assertRaisesRegex(m.PreparedControlsError, "source_contract_rejected"):
                self.build(runtime_sources=sources)
        for value in (b"", b"x" * 65537, "source text", bytearray(b"x")):
            sources = dict(self.options["runtime_sources"], **{"configuration_lumi.py": value})
            with self.subTest(value_type=type(value)), \
                    self.assertRaisesRegex(m.PreparedControlsError, "source_contract_rejected"):
                self.build(runtime_sources=sources)
        result = self.build()
        for row in result["envelope"]["runtime_sources"]:
            raw = self.options["runtime_sources"][row["path"]]
            self.assertEqual((row["bytes"], row["sha256"]), (len(raw), m.sha(raw)))

    def test_bad_metadata_lineage_is_rejected_without_exception_text(self):
        values = list(self.options["metadata"])
        values[2] = m.canonical({"private": PRIVATE})
        with self.assertRaises(m.PreparedControlsError) as caught:
            self.build(metadata=tuple(values))
        self.assertEqual(caught.exception.args, ("metadata_rejected",))
        self.assertNotIn(PRIVATE, str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_duplicate_json_and_nonfinite_values_are_rejected(self):
        for base in (b'{"x":1,"x":2}', b'{"x":NaN}', b'[]'):
            with self.subTest(base=base), self.assertRaises(m.PreparedControlsError):
                self.build(metadata=(base, *self.options["metadata"][1:]))

    def test_installer_validation_requires_all_independent_bindings(self):
        result = self.build()
        for name in ("expected_control_sha256", "expected_entry_sha256",
                     "expected_source_manifest_sha256", "expected_generation_nonce_sha256"):
            with self.subTest(name=name), self.assertRaises(m.PreparedControlsError):
                validate(result, self.options, **{name: "c" * 64})

    def test_installer_validates_exact_body_set_order_and_hashes(self):
        for kind in ("missing", "extra", "tampered", "reordered"):
            result = self.build()
            if kind == "missing":
                del result["bodies"]["state.json"]
            elif kind == "extra":
                result["bodies"]["extra.json"] = b"{}"
            elif kind == "tampered":
                result["bodies"]["state.json"] += b" "
            else:
                control = m.json_object(result["control_raw"])
                control["files"].reverse()
                result["control_raw"] = m.canonical(control)
            with self.subTest(kind=kind), self.assertRaises(m.PreparedControlsError):
                validate(result, self.options)

    def test_installer_rejects_rebound_but_non_builder_resource_envelope(self):
        result = self.build()
        envelope = m.json_object(result["bodies"]["envelope.json"])
        envelope["coordinator_read_cap_bytes"] += 1
        result["bodies"]["envelope.json"] = m.canonical(envelope)
        rebind_control(result)
        with self.assertRaisesRegex(m.PreparedControlsError, "metadata_rejected"):
            validate(result, self.options)

    def test_installer_control_limits_cannot_be_replaced_with_approval_flags(self):
        result = self.build()
        control = m.json_object(result["control_raw"])
        control["limits"]["execution_permitted"] = True
        result["control_raw"] = m.canonical(control)
        with self.assertRaisesRegex(m.PreparedControlsError, "metadata_rejected"):
            validate(result, self.options)

    def test_installer_runtime_body_membership_must_match_control_rows(self):
        result = self.build()
        wrong = dict(self.options["runtime_sources"], **{"modeling_lumi.py": b"wrong"})
        with self.assertRaisesRegex(m.PreparedControlsError, "metadata_rejected"):
            validate(result, self.options, runtime_sources=wrong)

    def test_installer_result_is_detached_and_does_not_claim_original_base_bytes(self):
        result = self.build()
        observed = validate(result, self.options)
        self.assertNotIn("original_base_sha256", observed["bindings"])
        self.assertNotIn("original_metadata_bytes", observed["counts"])
        self.assertTrue(all(observed[key] is False for key in m.FALSE_FLAGS))
        observed["plan"]["metadata"].clear()
        self.assertTrue(result["plan"]["metadata"])
        self.assertEqual(validate(result, self.options)["plan"], result["plan"])

    def test_ast_and_import_closure_have_no_builder_filesystem_or_ml_calls(self):
        tree = ast.parse(Path(m.__file__).read_bytes())
        forbidden = {"open", "read", "read_bytes", "write", "write_bytes", "mkdir", "unlink",
                     "execve", "fork", "system", "Popen", "load", "save", "run_import"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else (
                    node.func.id if isinstance(node.func, ast.Name) else "")
                self.assertNotIn(name, forbidden)
        self.assertFalse(any(name.split(".", 1)[0] in entry.ML_ROOTS for name in IMPORT_ADDITIONS))


if __name__ == "__main__":
    unittest.main()
