"""Control JSON fixtures only: no ML stack, real inputs, mounts or native runner."""

from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_lumi_host_contract_tests", ROOT / "src/lumi/posttraining_checkpoint_host_contract_v1.py")
m = importlib.util.module_from_spec(spec)
before = set(sys.modules)
spec.loader.exec_module(m)
IMPORTED = set(sys.modules) - before


def repin(plan):
    plan["plan_sha256"] = m.digest(m.canonical({k: v for k, v in plan.items()
                                             if k != "plan_sha256"}))
    return m.canonical(plan)


def fixtures():
    b = dict(zip(m.BOUND_FIELDS, [100, 1, 10, 1024, 512, 256, 0, 0, 0, 0, 0, 2, 200, 20],
                 strict=True))
    b["maximum_projected_live_tensor_bytes"] = 1024 + 512 + 256 + 16 * m.MIB
    b["minimum_host_ram_planning_bytes"] = 16 * m.GIB
    b["maximum_backend_output_bytes"] = 512 + m.MIB
    b["maximum_private_output_write_bytes"] = b["maximum_backend_output_bytes"] + 10 + 2 * m.MIB
    b["minimum_staged_disk_planning_bytes"] = 110 + b["maximum_private_output_write_bytes"]
    b.update(backend_payload_read_cap_verified=False, os_memory_time_disk_limits_enforced=False)
    plan = {"schema_version": "lumi.sft-checkpoint-import-plan.v0.1",
            "status": "PASS_plan_only_not_execution", "dtype": "bfloat16",
            "checkpoint_inventory": [{"path": "generated", "bytes": 100, "sha256": "a" * 64}],
            "tokenizer_inventory": [{"path": "tokenizer/generated", "bytes": 10,
                                     "sha256": "b" * 64}],
            "resource_bounds": b, **dict.fromkeys(m.PLAN_FALSE, False)}
    q = {"schema_version": "lumi.sft-import-qualification-execution-verification.v0.2",
         "status": "PASS_tiny_native_qualification_and_all_triggered_CI",
         "native_run_attempts": 1, "workflow_retries": 0, "all_triggered_CI_success": True,
         "report": {"sha256": "c" * 64}, "passes": [],
         **dict.fromkeys(("real_4B_import_verified", "GPU_used", "gate_5_complete",
                          "execution_permitted", "production_eligible"), False)}
    for index in (1, 2):
        q["passes"].append({"pass": index,
                            "status": "PASS_exact_native_receipts_publication_and_CI",
                            "prior_evidence_preserved": True, "all_triggered_CI_success": True,
                            "tests_or_native_harness_executed": False, "published_files": 8,
                            "required_file_bindings": 76, "aggregate_receipts": 4,
                            "ordered_cases": 3, "network_operations": 0, "real_payload_reads": 0})
    repin(plan)
    return plan, q


def envelope(plan):
    b = plan["resource_bounds"]
    value = dict.fromkeys(m.ENVELOPE_FIELDS, 1)
    value.update(dcp_read_cap_bytes=300, hf_read_cap_bytes=4 * b["maximum_backend_output_bytes"],
                 package_read_cap_bytes=4 * b["maximum_backend_output_bytes"] + 10,
                 hf_write_cap_bytes=b["maximum_backend_output_bytes"], package_write_cap_bytes=100,
                 config_bytes=100, control_bytes=100, coordinator_read_cap_bytes=100 * m.MIB,
                 private_write_cap_bytes=b["maximum_private_output_write_bytes"],
                 worker_memory_max_bytes=b["minimum_host_ram_planning_bytes"],
                 output_capacity_bytes=b["maximum_private_output_write_bytes"],
                 extra_host_disk_reserve_bytes=m.GIB,
                 host_disk_capacity_bytes=b["minimum_staged_disk_planning_bytes"] + m.GIB,
                 maximum_seconds=300)
    return value


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.q = fixtures()

    def build(self):
        p, q = m.canonical(self.plan), m.canonical(self.q)
        return m.build_contract(p, q, expected_plan_file_sha256=m.digest(p),
                                expected_qualification_file_sha256=m.digest(q))

    def check(self, e):
        p, raw = m.canonical(self.plan), m.canonical(e)
        return m.check_resource_envelope(p, raw, expected_plan_file_sha256=m.digest(p),
                                         expected_envelope_file_sha256=m.digest(raw))

    def validate(self, value):
        p, q, raw = m.canonical(self.plan), m.canonical(self.q), m.canonical(value)
        return m.validate_contract(raw, p, q, expected_contract_file_sha256=m.digest(raw),
                                   expected_plan_file_sha256=m.digest(p),
                                   expected_qualification_file_sha256=m.digest(q))

    def test_stdlib_imports_only(self):
        self.assertFalse({n.split(".")[0] for n in IMPORTED}
                         & {"lumi", "torch", "numpy", "transformers", "safetensors", "tokenizers"})

    def test_build_is_deterministic_without_file_io(self):
        before_plan = copy.deepcopy(self.plan)
        with patch("builtins.open", side_effect=AssertionError), \
                patch("os.open", side_effect=AssertionError):
            a, b = self.build(), self.build()
            self.assertEqual(self.validate(a), b)
        self.assertEqual(self.plan, before_plan)
        self.assertTrue(all(a[k] is False for k in m.FALSE_FLAGS))
        self.assertTrue(all(v is None for v in a["unresolved"].values()))

    def test_wrong_external_plan_hash(self):
        p, q = m.canonical(self.plan), m.canonical(self.q)
        with self.assertRaisesRegex(m.HostContractError, "input_binding"):
            m.build_contract(p, q, expected_plan_file_sha256="0" * 64,
                             expected_qualification_file_sha256=m.digest(q))

    def test_wrong_internal_plan_hash(self):
        self.plan["plan_sha256"] = "0" * 64
        with self.assertRaisesRegex(m.HostContractError, "plan_rejected"):
            self.build()

    def test_duplicate_invalid_and_oversize_json(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'[]', b'{',
                    b' ' * (m.MAXIMUM_CONTROL_BYTES + 1)):
            with self.subTest(raw_bytes=len(raw)), self.assertRaises(m.HostContractError):
                m._decode(raw, m.digest(raw))

    def test_plan_resource_mutations_and_bool_rejected(self):
        for key in m.BOUND_FIELDS:
            with self.subTest(key=key):
                self.plan, self.q = fixtures()
                self.plan["resource_bounds"][key] = True
                repin(self.plan)
                with self.assertRaisesRegex(m.HostContractError, "plan_rejected"):
                    self.build()

    def test_audit_and_capacity_arithmetic_cannot_be_reduced(self):
        for key in ("checkpoint_audit_read_bytes", "tokenizer_audit_read_bytes",
                    "minimum_host_ram_planning_bytes", "maximum_private_output_write_bytes",
                    "minimum_staged_disk_planning_bytes", "checkpoint_verification_passes"):
            with self.subTest(key=key):
                self.plan, self.q = fixtures()
                self.plan["resource_bounds"][key] -= 1
                repin(self.plan)
                with self.assertRaisesRegex(m.HostContractError, "plan_rejected"):
                    self.build()

    def test_inventory_alias_paths_and_totals_rejected(self):
        for path in ("/root", "../x", "x//y", "x/./y", "x\\y"):
            with self.subTest(path=path):
                self.plan["checkpoint_inventory"][0]["path"] = path
                repin(self.plan)
                with self.assertRaisesRegex(m.HostContractError, "plan_rejected"):
                    self.build()

    def test_qualification_must_be_successful_twice_and_not_real(self):
        mutations = [lambda q: q.update(native_run_attempts=True),
                     lambda q: q.update(real_4B_import_verified=True),
                     lambda q: q.update(all_triggered_CI_success=False),
                     lambda q: q["passes"].pop(),
                     lambda q: q["passes"][1].update(network_operations=1),
                     lambda q: q["passes"][0].update(ordered_cases=2)]
        for mutate in mutations:
            self.plan, self.q = fixtures()
            mutate(self.q)
            with self.assertRaisesRegex(m.HostContractError, "qualification_rejected"):
                self.build()

    def test_checkpoint_mode_owner_readonly_and_alias_rules_immutable(self):
        for key, value in (("file_mode", "0400"), ("directory_mode", "0755"),
                           ("worker_owned", False), ("read_only", False),
                           ("namespace_target", "/outputs")):
            with self.subTest(key=key):
                c = self.build()
                c["mount_requirements"]["checkpoint"][key] = value
                with self.assertRaisesRegex(m.HostContractError, "contract_binding"):
                    self.validate(c)

    def test_approval_boolean_cannot_unlock_contract(self):
        for key in m.FALSE_FLAGS:
            c = self.build()
            c[key] = True
            c["contract_sha256"] = m.digest(m.canonical({k: v for k, v in c.items()
                                                       if k != "contract_sha256"}))
            with self.assertRaisesRegex(m.HostContractError, "contract_binding"):
                self.validate(c)

    def test_unknown_or_resolved_contract_fields_rejected(self):
        for mutation in (lambda c: c.update(approval=True),
                         lambda c: c["unresolved"].update(dedicated_host_identity="pretend")):
            c = self.build()
            mutation(c)
            with self.assertRaisesRegex(m.HostContractError, "contract_binding"):
                self.validate(c)

    def test_source_audit_total_is_not_total_read_cap(self):
        self.assertEqual(self.build()["source_audit_only_bytes"], 220)
        result = self.check(envelope(self.plan))
        self.assertGreater(result["required_read_envelope_bytes"], 220)
        self.assertFalse(result["execution_permitted"])
        self.assertFalse(result["full_size_io_sufficiency_verified"])

    def test_dcp_first_audit_not_counted_three_times(self):
        e = envelope(self.plan)
        wanted = (e["dcp_read_cap_bytes"] + e["hf_read_cap_bytes"]
                  + 100 + 20 + e["package_read_cap_bytes"])
        e["coordinator_read_cap_bytes"] = wanted
        self.assertEqual(self.check(e)["required_read_envelope_bytes"], wanted)
        e["coordinator_read_cap_bytes"] -= 1
        with self.assertRaisesRegex(m.HostContractError, "envelope_rejected"):
            self.check(e)

    def test_config_and_three_terminal_controls_are_reserved(self):
        e = envelope(self.plan)
        wanted = e["hf_write_cap_bytes"] + e["package_write_cap_bytes"] + 100 + 300
        e["private_write_cap_bytes"] = wanted
        self.assertEqual(self.check(e)["required_write_envelope_bytes"], wanted)
        e["private_write_cap_bytes"] -= 1
        with self.assertRaisesRegex(m.HostContractError, "envelope_rejected"):
            self.check(e)

    def test_tiny_guard_resources_cannot_admit_larger_plan(self):
        for key, value in (("worker_memory_max_bytes", 4 * m.GIB),
                           ("output_capacity_bytes", 1024), ("coordinator_read_cap_bytes", 220),
                           ("dcp_read_cap_bytes", 100), ("package_write_cap_bytes", 9),
                           ("maximum_seconds", 86401)):
            e = envelope(self.plan)
            e[key] = value
            with self.subTest(key=key), \
                    self.assertRaisesRegex(m.HostContractError, "envelope_rejected"):
                self.check(e)

    def test_disk_plan_must_include_explicit_software_and_scratch_reserve(self):
        e = envelope(self.plan)
        e["host_disk_capacity_bytes"] = self.plan["resource_bounds"][
            "minimum_staged_disk_planning_bytes"]
        with self.assertRaisesRegex(m.HostContractError, "envelope_rejected"):
            self.check(e)

    def test_proposal_cannot_raise_frozen_write_cap(self):
        e = envelope(self.plan)
        e["private_write_cap_bytes"] += 1
        with self.assertRaisesRegex(m.HostContractError, "envelope_rejected"):
            self.check(e)

    def test_all_envelope_fields_reject_bool_zero_and_negative(self):
        for key in m.ENVELOPE_FIELDS:
            for value in (True, 0, -1, 1.0, None):
                e = envelope(self.plan)
                e[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(m.HostContractError):
                    self.check(e)


if __name__ == "__main__":
    unittest.main()
