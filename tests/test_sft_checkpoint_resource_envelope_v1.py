"""Synthetic metadata algebra only; no ML imports, payloads or target-host IO."""

from __future__ import annotations

import copy
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_lumi_resource_envelope_tests"
namespace = types.ModuleType(PACKAGE)
namespace.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = namespace
before = set(sys.modules)
m = importlib.import_module(PACKAGE + ".posttraining_checkpoint_resource_envelope_v1")
IMPORTED = set(sys.modules) - before
host = m.host


def repin(plan):
    plan["plan_sha256"] = host.digest(host.canonical({
        k: v for k, v in plan.items() if k != "plan_sha256"}))


def fixtures():
    checkpoint = [
        {"path": "manifest.sha256", "bytes": 100, "sha256": "a" * 64},
        {"path": "model/.metadata", "bytes": 100, "sha256": "b" * 64},
        {"path": "model/__0_0.distcp", "bytes": 2000, "sha256": "c" * 64},
    ]
    b = dict(zip(host.BOUND_FIELDS,
                 [2200, 3, 10, 1024, 512, 256, 0, 0, 0, 0, 0, 2, 4400, 20], strict=True))
    b["maximum_projected_live_tensor_bytes"] = 1024 + 512 + 256 + 16 * host.MIB
    b["minimum_host_ram_planning_bytes"] = 16 * host.GIB
    b["maximum_backend_output_bytes"] = 512 + host.MIB
    b["maximum_private_output_write_bytes"] = b["maximum_backend_output_bytes"] + 10 + 2 * host.MIB
    b["minimum_staged_disk_planning_bytes"] = 2210 + b["maximum_private_output_write_bytes"]
    b.update(backend_payload_read_cap_verified=False, os_memory_time_disk_limits_enforced=False)
    plan = {"schema_version": "lumi.sft-checkpoint-import-plan.v0.1",
            "status": "PASS_plan_only_not_execution", "dtype": "bfloat16",
            "metadata": {"source_config_sha256": "d" * 64,
                         "checkpoint_manifest_sha256": "a" * 64},
            "checkpoint_inventory": checkpoint,
            "tokenizer_inventory": [{"path": "tokenizer/generated", "bytes": 10,
                                     "sha256": "e" * 64}],
            "resource_bounds": b, **dict.fromkeys(host.PLAN_FALSE, False)}
    repin(plan)
    e = dict.fromkeys(host.ENVELOPE_FIELDS, 1)
    backend = b["maximum_backend_output_bytes"]
    e.update(schema_version=m.SCHEMA, plan_sha256=plan["plan_sha256"],
             dcp_payload_read_cap_bytes=2200, dcp_read_cap_bytes=4400,
             hf_read_cap_bytes=backend + 2 * 30 + 100,
             package_read_cap_bytes=3 * backend + 3 * 10 + 2 * 1000 + 3 * 1000,
             hf_write_cap_bytes=backend, package_write_cap_bytes=10 + 1000 + 1000,
             config_bytes=100, source_config_sha256="d" * 64, control_bytes=1000,
             package_receipt_cap_bytes=1000, package_manifest_cap_bytes=1000,
             runtime_sources=[{"path": name, "bytes": size, "sha256": "f" * 64}
                              for name, size in zip(m.RUNTIME_NAMES, (10, 20), strict=True)],
             private_write_cap_bytes=b["maximum_private_output_write_bytes"],
             worker_memory_max_bytes=b["minimum_host_ram_planning_bytes"],
             output_capacity_bytes=b["maximum_private_output_write_bytes"],
             extra_host_disk_reserve_bytes=host.GIB,
             host_disk_capacity_bytes=b["minimum_staged_disk_planning_bytes"] + host.GIB,
             maximum_seconds=300)
    e["coordinator_read_cap_bytes"] = (e["dcp_read_cap_bytes"] + e["hf_read_cap_bytes"]
                                       + 2200 + 20 + e["package_read_cap_bytes"] + 30)
    return plan, e


class ResourceEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.e = fixtures()

    def check(self, *, plan_pin=None, envelope_pin=None, raw=None):
        p = host.canonical(self.plan)
        e = host.canonical(self.e) if raw is None else raw
        return m.validate_resource_envelope(
            p, e, expected_plan_file_sha256=host.digest(p) if plan_pin is None else plan_pin,
            expected_envelope_file_sha256=host.digest(e) if envelope_pin is None else envelope_pin)

    def rejected(self, code=None):
        return (self.assertRaises(m.ResourceEnvelopeError) if code is None else
                self.assertRaisesRegex(m.ResourceEnvelopeError, "^" + code + "$"))

    def test_imports_no_ml_runtime(self):
        self.assertFalse({name.split(".")[0] for name in IMPORTED}
                         & {"lumi", "torch", "numpy", "transformers", "safetensors", "tokenizers"})

    def test_validation_has_no_file_or_network_io(self):
        with patch("builtins.open", side_effect=AssertionError), \
                patch("os.open", side_effect=AssertionError), \
                patch("socket.socket", side_effect=AssertionError):
            self.assertEqual(self.check()["status"], m.STATUS)

    def test_result_is_copied_and_never_execution_authority(self):
        saved = copy.deepcopy(self.e)
        result = self.check()
        self.assertTrue(all(result[key] is False for key in m.FALSE_FLAGS))
        result["envelope"]["runtime_sources"][0]["bytes"] += 1
        result["runner_limits"]["maximum_seconds"] += 1
        self.assertEqual(self.e, saved)
        self.assertEqual(self.check()["envelope"], saved)

    def test_runner_limits_have_exact_frozen_six_keys(self):
        self.assertEqual(set(self.check()["runner_limits"]), {
            "maximum_seconds", "maximum_coordinator_read_bytes", "maximum_private_write_bytes",
            "maximum_package_read_bytes", "maximum_package_write_bytes", "maximum_control_bytes"})

    def test_wrong_independent_hashes_rejected(self):
        with self.rejected("input_binding"):
            self.check(plan_pin="0" * 64)
        with self.rejected("input_binding"):
            self.check(envelope_pin="0" * 64)

    def test_plan_internal_hash_is_required(self):
        self.plan["plan_sha256"] = "0" * 64
        with self.rejected("plan_rejected"):
            self.check()

    def test_wrong_plan_binding_in_envelope(self):
        self.e["plan_sha256"] = "0" * 64
        with self.rejected():
            self.check()

    def test_invalid_duplicate_nonfinite_oversize_json(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'[]', b'{',
                    b' ' * (host.MAXIMUM_CONTROL_BYTES + 1)):
            with self.subTest(raw_bytes=len(raw)), self.rejected():
                self.check(raw=raw)

    def test_exact_schema_rejects_missing_extra_and_approval_fields(self):
        for key in (*m.FIELDS, "approval", "execution_permitted"):
            self.plan, self.e = fixtures()
            if key in self.e:
                del self.e[key]
            else:
                self.e[key] = True
            with self.subTest(key=key), self.rejected():
                self.check()

    def test_exact_integer_types_and_bounds(self):
        for key in m.INTEGER_FIELDS:
            for value in (True, 1.0, "1", 0, -1, 2**63):
                self.plan, self.e = fixtures()
                self.e[key] = value
                with self.subTest(key=key, value=value), self.rejected():
                    self.check()

    def test_package_includes_three_backend_and_token_passes(self):
        result = self.check()
        b = self.plan["resource_bounds"]
        self.assertEqual(result["minimum_package_content_read_bytes"],
                         3 * b["maximum_backend_output_bytes"] + 3 * 10 + 2 * 1000 + 3 * 1000)
        self.e["package_read_cap_bytes"] -= 1
        with self.rejected():
            self.check()

    def test_package_write_includes_receipt_and_manifest(self):
        self.assertEqual(self.check()["minimum_package_content_write_bytes"], 2010)
        self.e["package_write_cap_bytes"] -= 1
        with self.rejected():
            self.check()

    def test_hf_read_includes_source_config_and_runtime_rereads(self):
        self.assertEqual(self.check()["minimum_hf_content_read_bytes"],
                         self.e["hf_write_cap_bytes"] + 160)
        self.e["hf_read_cap_bytes"] -= 1
        with self.rejected():
            self.check()

    def test_dcp_payload_must_cover_metadata_and_storage_floor(self):
        self.assertEqual(self.check()["minimum_dcp_payload_planning_bytes"], 2200)
        self.e["dcp_payload_read_cap_bytes"] -= 1
        self.e["dcp_read_cap_bytes"] -= 1
        with self.rejected():
            self.check()

    def test_dcp_total_is_exactly_first_audit_plus_payload_cap(self):
        self.e["dcp_read_cap_bytes"] += 1
        self.e["coordinator_read_cap_bytes"] += 1
        with self.rejected():
            self.check()

    def test_dcp_native_sufficiency_never_inferred(self):
        result = self.check()
        self.assertFalse(result["dcp_native_serializer_overhead_measured"])
        self.assertFalse(result["native_io_sufficiency_verified"])
        self.assertFalse(result["full_size_performance_measured"])

    def test_total_reads_add_only_one_checkpoint_audit_to_dcp(self):
        result = self.check()
        self.assertEqual(result["required_coordinator_read_envelope_bytes"],
                         self.e["coordinator_read_cap_bytes"])
        self.assertEqual(result["checkpoint_full_audits_reserved"], 2)
        self.e["coordinator_read_cap_bytes"] -= 1
        with self.rejected():
            self.check()

    def test_entire_component_caps_are_reserved_not_only_minimums(self):
        self.e["hf_read_cap_bytes"] += 1
        with self.rejected():
            self.check()
        self.e["coordinator_read_cap_bytes"] += 1
        self.assertEqual(self.check()["required_coordinator_read_envelope_bytes"],
                         self.e["coordinator_read_cap_bytes"])

    def test_separate_runtime_source_preflight_is_prepaid_once(self):
        result = self.check()
        self.assertEqual(result["runtime_source_preflight_read_bytes"], 30)
        self.e["coordinator_read_cap_bytes"] -= 30
        with self.rejected():
            self.check()

    def test_package_controls_cannot_claim_unenforced_smaller_limits(self):
        for key in ("package_receipt_cap_bytes", "package_manifest_cap_bytes"):
            self.plan, self.e = fixtures()
            self.e[key] -= 1
            with self.subTest(key=key), self.rejected():
                self.check()

    def test_config_and_three_terminal_controls_are_reserved(self):
        wanted = self.e["hf_write_cap_bytes"] + self.e["package_write_cap_bytes"] + 100 + 3000
        self.e["private_write_cap_bytes"] = wanted
        self.assertEqual(self.check()["required_private_write_envelope_bytes"], wanted)
        self.e["private_write_cap_bytes"] -= 1
        with self.rejected():
            self.check()

    def test_frozen_private_write_cap_cannot_be_raised(self):
        self.e["private_write_cap_bytes"] += 1
        self.e["output_capacity_bytes"] += 1
        self.e["host_disk_capacity_bytes"] += 1
        with self.rejected():
            self.check()

    def test_backend_write_cap_cannot_diverge_from_bound_plan(self):
        for delta in (-1, 1):
            self.plan, self.e = fixtures()
            self.e["hf_write_cap_bytes"] += delta
            with self.rejected():
                self.check()

    def test_control_reserves_cannot_exceed_actual_writer_limit(self):
        for key, value in (("config_bytes", 65537), ("control_bytes", 131073),
                           ("package_receipt_cap_bytes", 1001),
                           ("package_manifest_cap_bytes", 1001), ("control_bytes", 99)):
            self.plan, self.e = fixtures()
            self.e[key] = value
            with self.subTest(key=key), self.rejected():
                self.check()

    def test_host_ram_output_disk_and_time_are_enforced_as_proposals(self):
        for key, value in (("worker_memory_max_bytes", 4 * host.GIB),
                           ("output_capacity_bytes", 1), ("host_disk_capacity_bytes", 1),
                           ("maximum_seconds", 86401)):
            self.plan, self.e = fixtures()
            self.e[key] = value
            with self.subTest(key=key), self.rejected():
                self.check()

    def test_disk_includes_extra_reserve_and_does_not_equal_write_counter(self):
        wanted = (2200 + 10 + self.e["output_capacity_bytes"]
                  + self.e["extra_host_disk_reserve_bytes"])
        self.assertEqual(self.check()["required_host_disk_capacity_bytes"], wanted)
        self.e["host_disk_capacity_bytes"] = wanted - 1
        with self.rejected():
            self.check()

    def test_runtime_source_bindings_exact_names_order_and_types(self):
        mutations = [lambda e: e["runtime_sources"].reverse(),
                     lambda e: e["runtime_sources"].append(e["runtime_sources"][0]),
                     lambda e: e["runtime_sources"][0].update(path="../configuration_lumi.py"),
                     lambda e: e["runtime_sources"][0].update(bytes=True),
                     lambda e: e["runtime_sources"][0].update(bytes=65537),
                     lambda e: e["runtime_sources"][0].update(sha256="f" * 63),
                     lambda e: e["runtime_sources"][0].update(approved=True)]
        for mutate in mutations:
            self.plan, self.e = fixtures()
            mutate(self.e)
            with self.rejected():
                self.check()

    def test_runtime_source_must_fit_bound_hf_control_cap_before_start(self):
        # Config fits, all logical totals are increased correctly, but one known
        # source exceeds the same HF control cap that the coordinator enforces.
        previous = self.e["runtime_sources"][0]["bytes"]
        self.e["runtime_sources"][0]["bytes"] = self.e["control_bytes"] + 1
        extra = self.e["runtime_sources"][0]["bytes"] - previous
        self.e["hf_read_cap_bytes"] += 2 * extra
        self.e["coordinator_read_cap_bytes"] += 3 * extra
        self.assertLess(self.e["config_bytes"], self.e["control_bytes"])
        with self.rejected("envelope_rejected"):
            self.check()

    def test_runtime_source_equal_to_bound_hf_control_cap_is_accepted(self):
        previous = self.e["runtime_sources"][0]["bytes"]
        self.e["runtime_sources"][0]["bytes"] = self.e["control_bytes"]
        extra = self.e["runtime_sources"][0]["bytes"] - previous
        self.e["hf_read_cap_bytes"] += 2 * extra
        self.e["coordinator_read_cap_bytes"] += 3 * extra
        self.assertEqual(self.check()["status"], m.STATUS)

    def test_source_config_sha256_bound_to_plan(self):
        self.e["source_config_sha256"] = "0" * 64
        with self.rejected("plan_rejected"):
            self.check()

    def test_required_dcp_metadata_manifest_and_model_rows(self):
        for index in range(3):
            self.plan, self.e = fixtures()
            self.plan["checkpoint_inventory"][index]["path"] = "other/fixture"
            repin(self.plan)
            self.e["plan_sha256"] = self.plan["plan_sha256"]
            with self.rejected("plan_rejected"):
                self.check()

    def test_manifest_pin_cannot_diverge_from_plan_metadata(self):
        self.plan["metadata"]["checkpoint_manifest_sha256"] = "0" * 64
        repin(self.plan)
        self.e["plan_sha256"] = self.plan["plan_sha256"]
        with self.rejected("plan_rejected"):
            self.check()

    def test_arithmetic_overflow_rejected_before_any_effect(self):
        self.e["dcp_payload_read_cap_bytes"] = m.MAXIMUM_BYTES
        self.e["dcp_read_cap_bytes"] = m.MAXIMUM_BYTES
        with self.rejected("resource_exhausted"):
            self.check()

    def test_errors_never_retain_input_or_exception_text(self):
        raw = b'{"private_text":"never-retain-this"}'
        with self.rejected() as raised:
            self.check(raw=raw)
        self.assertEqual(str(raised.exception), "envelope_rejected")
        self.assertNotIn("never-retain", repr(raised.exception))


if __name__ == "__main__":
    unittest.main()
