"""AST plus isolated stdlib codec/hook doubles; NEVER import the native adapter.

No native entry, prepare, rank target, runtime guard or file reader is executed.
Synthetic hook functions use explicit in-memory doubles only; no ML, runtime,
image, source dataset, Docker, process, network or filesystem fixture operations.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "scripts/run_sft_oci_guarded_worker_v1.py"
PROBE = ROOT / "scripts/run_sft_oci_worker_probe_v2.py"
FIXTURE = ROOT / "scripts/run_sft_worker_run_torch_fixture_v1.py"
ADMISSION = ROOT / "src/lumi/posttraining_sft_oci_admission_v1.py"
ACCEPTED_REPORT = ROOT / "scripts/sft_cpu_lookup_acceptance_report_v1.py"
WORKER = ROOT / "src/lumi/posttraining_sft_worker_run_v1.py"
PURE_FUNCTIONS = (
    "need", "canonical", "sha256", "source_identity", "pairs", "reject_number",
    "validate_control_bytes", "validate_observer", "validate_report",
)
DOUBLE_FUNCTIONS = (
    "guarded_admit", "construct_worker", "fixture_import", "guarded_outcome", "guarded_invocation",
)


def function(tree, name):
    found = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
             and node.name == name]
    if len(found) != 1:
        raise AssertionError("function missing")
    return found[0]


def literal(tree, name):
    found = [node.value for node in tree.body if isinstance(node, ast.Assign)
             and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
             and node.targets[0].id == name]
    if len(found) != 1:
        raise AssertionError("literal missing")
    return ast.literal_eval(found[0])


def calls(tree, name):
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and ast.unparse(node.func) == name]


def equal(node, expression):
    expected = ast.parse(expression, mode="eval").body
    return ast.dump(node, include_attributes=False) == ast.dump(expected, include_attributes=False)


def namespace(tree, *, hooks=False):
    """Compile ONLY audited pure functions, or isolated hook functions with doubles."""
    values = {"hashlib": hashlib, "json": json, "re": re, "SimpleNamespace": SimpleNamespace}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and not target.id.startswith("_"):
                values[target.id] = ast.literal_eval(node.value)
    wanted = set(PURE_FUNCTIONS + (DOUBLE_FUNCTIONS if hooks else ()))
    chosen = [copy.deepcopy(node) for node in tree.body
              if (isinstance(node, ast.FunctionDef) and node.name in wanted)
              or (isinstance(node, ast.ClassDef) and node.name == "GuardianError")]
    module = ast.fix_missing_locations(ast.Module(body=chosen, type_ignores=[]))
    exec(compile(module, "<synthetic-guarded-worker-functions>", "exec"), values)
    return values


def control(ns):
    return {
        "schema": ns["CONTROL_SCHEMA"], "software_binding": dict(ns["SOFTWARE_BINDING"]),
        "source_identity_sha256": ns["source_identity"](ns["SOFTWARE_BINDING"]),
    }


def observer(ns):
    worker = {
        "schema_version": "lumi.sft-worker-run-fixture.v0.1", "status": "passed",
        **dict.fromkeys(ns["WORKER_TRUE_FLAGS"], True),
        **dict.fromkeys(ns["WORKER_FALSE_FLAGS"], False), **ns["WORKER_COUNTS"],
        **dict.fromkeys(ns["WORKER_COUNT_LIMITS"], 1),
    }
    return {
        "schema": "lumi.sft-oci-worker-probe.v0.1", "status": "passed", "worker_run": worker,
        "origin_counts": dict.fromkeys(ns["ORIGIN_FIELDS"], 1),
        **{key: ns["SOFTWARE_BINDING"][key] for key in ns["CONTROL_FIELDS"]},
        **dict.fromkeys(ns["OBSERVER_TRUE_FLAGS"], True),
        **dict.fromkeys(ns["OBSERVER_FALSE_FLAGS"], False), **ns["OBSERVER_COUNTS"],
    }


def report(ns, case="positive"):
    positive = case == "positive"
    adapter_ok = case != "wrong_adapter"
    return {
        "schema": ns["SCHEMA"], "status": "passed" if positive else "failed",
        "code": "ok" if positive else (
            "control_binding_rejected" if case == "wrong_control" else "adapter_binding_rejected"
        ),
        "source_identity_sha256": ns["source_identity"](ns["SOFTWARE_BINDING"]),
        "counts": dict.fromkeys(ns["COUNT_KEYS"], 6 if positive else 0),
        "counts_semantics": "verified_completed_rank_proofs",
        "ml_absent": True, "parent_ml_import_absent": True,
        "adapter_separately_bound": adapter_ok,
        "adapter_sha256": "a" * 64 if adapter_ok else None,
        "control_sha256": "b" * 64 if positive else None,
        "observer": observer(ns) if positive else None,
        "real_sft_execution_permitted": False, "production_eligible": False,
    }


class GuardedWorkerPureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(ADAPTER.read_text(encoding="utf-8"))

    def setUp(self):
        self.ns = namespace(self.tree)

    def validate_control(self, value=None, *, raw=None, pin=None):
        if raw is None:
            raw = self.ns["canonical"](control(self.ns) if value is None else value) + b"\n"
        if pin is None:
            pin = hashlib.sha256(raw).hexdigest()
        return self.ns["validate_control_bytes"](raw, pin)

    def validate_report(self, value=None, *, case="positive", wait_status=None, identity=None):
        if wait_status is None:
            wait_status = 0 if case == "positive" else 512
        if identity is None:
            identity = self.ns["source_identity"](self.ns["SOFTWARE_BINDING"])
        return self.ns["validate_report"](
            report(self.ns, case) if value is None else value,
            case=case, source_identity_sha256=identity, wait_status=wait_status,
        )

    def reject_control(self, expected, **kwargs):
        with self.assertRaises(self.ns["GuardianError"]) as caught:
            self.validate_control(**kwargs)
        self.assertEqual(caught.exception.code, expected)
        self.assertEqual(str(caught.exception), expected)

    def reject_report(self, **kwargs):
        with self.assertRaises(self.ns["GuardianError"]) as caught:
            self.validate_report(**kwargs)
        self.assertEqual(caught.exception.code, "report_rejected")

    def test_control_is_exact_fresh_and_source_identity_domain_separated(self):
        original = control(self.ns)
        first, second = self.validate_control(), self.validate_control()
        self.assertEqual(first, original)
        self.assertIsNot(first, second)
        self.assertIsNot(first["software_binding"], second["software_binding"])
        first["software_binding"]["source_commit"] = "0" * 40
        self.assertEqual(second, original)
        expected = hashlib.sha256(self.ns["canonical"]({
            "schema": "lumi.sft-accepted-oci-software-identity.v0.1",
            **self.ns["SOFTWARE_BINDING"],
        })).hexdigest()
        self.assertEqual(expected, second["source_identity_sha256"])
        self.assertNotEqual(expected, "f" * 64)

    def test_control_raw_type_size_and_pin_rejections(self):
        for raw in (b"", b" " * 4097, bytearray(b"{}"), "{}", None):
            with self.subTest(raw_type=type(raw).__name__):
                with self.assertRaises(self.ns["GuardianError"]) as caught:
                    self.ns["validate_control_bytes"](raw, "0" * 64)
                self.assertEqual(caught.exception.code, "control_binding_rejected")
        for pin in ("0" * 64, "F" * 64, None, True, [], ""):
            with self.subTest(pin_type=type(pin).__name__):
                raw = self.ns["canonical"](control(self.ns)) + b"\n"
                with self.assertRaises(self.ns["GuardianError"]) as caught:
                    self.ns["validate_control_bytes"](raw, pin)
                self.assertEqual(caught.exception.code, "control_binding_rejected")

    def test_control_exact_keys_binding_and_identity(self):
        for key in control(self.ns):
            value = control(self.ns)
            del value[key]
            with self.subTest(missing=key):
                self.reject_control("control_rejected", value=value)
        for key in self.ns["SOFTWARE_BINDING"]:
            value = control(self.ns)
            value["software_binding"][key] = "SYNTHETIC-PRIVATE-TEXT"
            with self.subTest(binding=key):
                self.reject_control("control_rejected", value=value)
        value = control(self.ns)
        value["source_identity_sha256"] = "f" * 64
        self.reject_control("source_identity_rejected", value=value)
        value = control(self.ns)
        value["extra"] = False
        self.reject_control("control_rejected", value=value)

    def test_control_canonical_lf_duplicates_numbers_and_json_are_closed(self):
        raw = self.ns["canonical"](control(self.ns)) + b"\n"
        for bad in (
            raw[:-1], raw + b"\n", b" " + raw, raw[:-1] + b"\r\n", b"\xff", b"{",
            raw.replace(b'{"schema":', b'{"schema":null,"schema":', 1),
            b'{"extra":1.0}', b'{"extra":1}', b'{"extra":NaN}',
        ):
            with self.subTest(size=len(bad)):
                self.reject_control("control_rejected", raw=bad)

    def test_positive_and_two_exact_pre_ml_negative_receipts_are_compact(self):
        for case in ("positive", "wrong_control", "wrong_adapter"):
            with self.subTest(case=case):
                receipt = self.validate_report(case=case)
                self.assertEqual(receipt["schema"], self.ns["RECEIPT_SCHEMA"])
                self.assertEqual(receipt["status"], "verified")
                self.assertEqual(receipt["case"], case)
                self.assertNotIn("observer", receipt)
                self.assertLessEqual(len(self.ns["canonical"](receipt)) + 1, 2048)
                self.assertFalse(receipt["real_sft_execution_permitted"])
                self.assertFalse(receipt["production_eligible"])

    def test_report_configuration_wait_status_and_identity_fail_closed(self):
        for bad in (None, True, "normal", "", [], {}):
            with self.subTest(case_type=type(bad).__name__):
                self.reject_report(value=report(self.ns), case=bad, wait_status=0)
        for status in (True, False, 1, 256, 512, "0"):
            with self.subTest(wait_status=status):
                self.reject_report(wait_status=status)
        self.reject_report(identity="f" * 64)
        for case in ("wrong_control", "wrong_adapter"):
            for status in (0, 256, True):
                with self.subTest(case=case, wait_status=status):
                    self.reject_report(case=case, wait_status=status)

    def test_report_exact_root_keys_types_and_false_execution_claims(self):
        for key in report(self.ns):
            value = report(self.ns)
            del value[key]
            with self.subTest(missing=key):
                self.reject_report(value=value)
        for key, bad in (
            ("schema", "wrong"), ("ml_absent", 1), ("parent_ml_import_absent", False),
            ("adapter_separately_bound", 1), ("production_eligible", 0),
            ("real_sft_execution_permitted", True), ("adapter_sha256", "bad"),
            ("control_sha256", None), ("extra", "SYNTHETIC-PRIVATE-TEXT"),
        ):
            value = report(self.ns)
            value[key] = bad
            with self.subTest(key=key):
                self.reject_report(value=value)

    def test_each_guarded_count_is_exact_integer_not_bool(self):
        for case in ("positive", "wrong_control", "wrong_adapter"):
            for key in self.ns["COUNT_KEYS"]:
                for bad in (True, False, -1, 1, 5, 7, "6"):
                    value = report(self.ns, case)
                    value["counts"][key] = bad
                    with self.subTest(case=case, key=key, bad=bad):
                        self.reject_report(value=value, case=case)

    def test_negative_case_cannot_hide_observer_or_crosswire_error_pins(self):
        for case in ("wrong_control", "wrong_adapter"):
            for key, bad in (
                ("observer", observer(self.ns)), ("control_sha256", "b" * 64),
                ("code", "guarded_worker_failed"), ("status", "passed"),
            ):
                value = report(self.ns, case)
                value[key] = bad
                with self.subTest(case=case, key=key):
                    self.reject_report(value=value, case=case)
        value = report(self.ns, "wrong_adapter")
        value["adapter_separately_bound"] = True
        self.reject_report(value=value, case="wrong_adapter")

    def test_full_observer_flags_controls_and_counts_are_verified(self):
        for key in self.ns["CONTROL_FIELDS"]:
            value = report(self.ns)
            value["observer"][key] = "0" * 64
            with self.subTest(control=key):
                self.reject_report(value=value)
        for key in self.ns["OBSERVER_TRUE_FLAGS"] + self.ns["OBSERVER_FALSE_FLAGS"]:
            value = report(self.ns)
            value["observer"][key] = not value["observer"][key]
            with self.subTest(flag=key):
                self.reject_report(value=value)
        for key in self.ns["OBSERVER_COUNTS"]:
            value = report(self.ns)
            value["observer"][key] = True
            with self.subTest(count=key):
                self.reject_report(value=value)

    def test_full_worker_report_resume_flags_exact_counts_and_caps_are_verified(self):
        for key in self.ns["WORKER_TRUE_FLAGS"] + self.ns["WORKER_FALSE_FLAGS"]:
            value = report(self.ns)
            value["observer"]["worker_run"][key] = not value["observer"]["worker_run"][key]
            with self.subTest(flag=key):
                self.reject_report(value=value)
        for key in self.ns["WORKER_COUNTS"]:
            value = report(self.ns)
            value["observer"]["worker_run"][key] += 1
            with self.subTest(count=key):
                self.reject_report(value=value)
        for key, maximum in self.ns["WORKER_COUNT_LIMITS"].items():
            for bad in (True, 0, maximum + 1):
                value = report(self.ns)
                value["observer"]["worker_run"][key] = bad
                with self.subTest(count=key, bad=bad):
                    self.reject_report(value=value)

    def test_nested_report_extra_keys_and_origin_numbers_rejected(self):
        for path in (("observer",), ("observer", "worker_run"), ("observer", "origin_counts")):
            value = report(self.ns)
            selected = value
            for key in path:
                selected = selected[key]
            selected["extra"] = "SYNTHETIC-PRIVATE-TEXT"
            with self.subTest(path=path):
                self.reject_report(value=value)
        for bad in (True, -1, 196609, "1"):
            value = report(self.ns)
            value["observer"]["origin_counts"]["source_modules"] = bad
            self.reject_report(value=value)

    def test_pure_validation_error_text_never_contains_inputs(self):
        for bad in (None, [], {}, "SYNTHETIC-PRIVATE-TEXT"):
            error = self.ns["GuardianError"](bad)
            self.assertEqual(error.code, "guarded_worker_failed")
            self.assertNotIn("SYNTHETIC-PRIVATE-TEXT", repr(error))


class GuardedWorkerHookDoubleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(ADAPTER.read_text(encoding="utf-8"))

    def setUp(self):
        self.ns = namespace(self.tree, hooks=True)
        self.ns.update({
            "_CONTROL": control(self.ns), "_ADAPTER_PIN": "a" * 64, "_CONTROL_PIN": "b" * 64,
            "_WORKER": None, "_WORKER_MODULE": None, "_RANK": 0, "_INVOCATIONS": 0,
            "_RUN_IDENTITY": None, "_COUNTS": dict.fromkeys(self.ns["COUNT_KEYS"], 0),
            "_LOCAL": dict.fromkeys(self.ns["COUNT_KEYS"][:3], 0),
        })
        self.reads, self.constructed = [], []
        self.ns["load_control"] = lambda: self.ns["_CONTROL"]
        self.ns["verify_adapter"] = lambda: self.reads.append("adapter_verified")
        self.ns["ml_absent"] = lambda: True

    def fake_module(self):
        ns, constructed = self.ns, self.constructed

        class Worker:
            def __init__(self, **kwargs):
                constructed.append(kwargs)
                self._pins = kwargs["loader_bindings"]
                self._profile = kwargs["profile"]
                self._admit = kwargs["admit"]
                self._identity = hashlib.sha256(ns["canonical"]({
                    "schema_version": "lumi.sft-worker-run.v0.1",
                    "source_identity_sha256": kwargs["source_identity_sha256"],
                    "loader_bindings": self._pins, "profile": self._profile,
                    "world_size": 2, "record_limit": kwargs["record_limit"],
                    "device_type": "cpu", "target_steps": 2,
                })).hexdigest()

        return SimpleNamespace(
            WorkerRun=Worker, SCHEMA_VERSION="lumi.sft-worker-run.v0.1",
            STAGES=("idle", "complete"), ERROR_CODES=("synthetic_error",),
            _digest=lambda value: hashlib.sha256(ns["canonical"](value)).hexdigest(),
        )

    def kwargs(self, admit=None):
        return {
            "rank": 0, "world_size": 2, "device_type": "cpu", "record_limit": 2,
            "source_identity_sha256": "f" * 64, "loader_bindings": {"synthetic": "a" * 64},
            "profile": {"synthetic": True}, "admit": (lambda: None) if admit is None else admit,
        }

    def test_constructor_delivers_real_identity_and_preserves_original_admit(self):
        self.ns["_LOCAL"]["pre_ml_rank_checks"] = 1
        self.ns["_WORKER_MODULE"] = self.fake_module()
        admitted = []
        original = self.kwargs(admit=lambda: admitted.append("original_admit"))
        worker = self.ns["construct_worker"](**original)
        self.assertEqual(self.constructed[0]["source_identity_sha256"],
                         self.ns["_CONTROL"]["source_identity_sha256"])
        self.assertEqual(original["source_identity_sha256"], "f" * 64)
        self.assertEqual(self.ns["_LOCAL"]["worker_constructors"], 1)
        worker._admit()
        self.assertEqual(admitted, ["original_admit"])
        self.assertEqual(self.reads, ["adapter_verified"])
        self.assertEqual(self.ns["_LOCAL"]["worker_admissions"], 1)
        with self.assertRaises(self.ns["GuardianError"]):
            worker._admit()
        self.assertEqual(admitted, ["original_admit"])

    def test_constructor_rejects_missing_pre_ml_check_wrong_original_identity_and_repeat(self):
        self.ns["_WORKER_MODULE"] = self.fake_module()
        with self.assertRaises(self.ns["GuardianError"]):
            self.ns["construct_worker"](**self.kwargs())
        self.assertFalse(self.constructed)
        self.ns["_LOCAL"]["pre_ml_rank_checks"] = 1
        with self.assertRaises(self.ns["GuardianError"]):
            self.ns["construct_worker"](**(self.kwargs() | {"source_identity_sha256": "e" * 64}))
        self.assertFalse(self.constructed)
        self.ns["construct_worker"](**self.kwargs())
        with self.assertRaises(self.ns["GuardianError"]):
            self.ns["construct_worker"](**self.kwargs())
        self.assertEqual(len(self.constructed), 1)

    def test_constructor_verifies_actual_worker_plan_consumed_source_identity(self):
        self.ns["_LOCAL"]["pre_ml_rank_checks"] = 1
        module = self.fake_module()
        original = module.WorkerRun

        def wrong(**kwargs):
            worker = original(**(kwargs | {"source_identity_sha256": "f" * 64}))
            return worker

        module.WorkerRun = wrong
        self.ns["_WORKER_MODULE"] = module
        with self.assertRaises(self.ns["GuardianError"]) as caught:
            self.ns["construct_worker"](**self.kwargs())
        self.assertEqual(caught.exception.code, "source_identity_rejected")
        self.assertEqual(self.ns["_LOCAL"]["worker_constructors"], 0)

    def test_import_facade_only_intercepts_one_exact_fixture_import(self):
        module, other = self.fake_module(), object()
        requested = []

        def importer(name, package):
            requested.append((name, package))
            return module if name == "lumi.posttraining_sft_worker_run_v1" else other

        self.ns["_ORIGINAL_IMPORT"] = importer
        for name in ("torch", "lumi.posttraining_sft_worker_run_v2", "x.worker_run_v1"):
            self.assertIs(self.ns["fixture_import"](name), other)
        facade = self.ns["fixture_import"]("lumi.posttraining_sft_worker_run_v1")
        self.assertEqual(vars(facade).keys(), {"WorkerRun", "STAGES", "ERROR_CODES"})
        self.assertIs(facade.WorkerRun, self.ns["construct_worker"])
        self.assertIs(facade.STAGES, module.STAGES)
        self.assertIsNot(facade, module)
        self.assertIs(self.ns["_WORKER_MODULE"], module)
        self.assertEqual(len(requested), 4)

    def test_private_packet_preserves_existing_origin_hook_and_phase_messages(self):
        self.ns["_LOCAL"] = dict.fromkeys(self.ns["COUNT_KEYS"][:3], 1)
        self.ns["_WORKER"] = SimpleNamespace(_identity="c" * 64)
        seen = []
        self.ns["_ORIGINAL_OUTCOME"] = lambda pipe, value: seen.append((pipe, value))
        phase = {"status": "phase", "phase": "inputs", "step": 0}
        self.ns["guarded_outcome"]("synthetic-pipe", phase)
        self.assertIs(seen[0][1], phase)
        passed = {"status": "passed", "training_state": {"run_identity_sha256": "c" * 64}}
        self.ns["guarded_outcome"]("synthetic-pipe", passed)
        self.assertNotIn(self.ns["PACKET_FIELD"], passed)
        proof = seen[1][1][self.ns["PACKET_FIELD"]]
        self.assertEqual(proof["source_identity_sha256"],
                         self.ns["_CONTROL"]["source_identity_sha256"])
        self.assertEqual(proof["run_identity_sha256"], "c" * 64)

    def private_reports(self):
        proof = {
            "schema": self.ns["PACKET_SCHEMA"],
            "source_identity_sha256": self.ns["_CONTROL"]["source_identity_sha256"],
            "run_identity_sha256": "c" * 64, "adapter_sha256": "a" * 64,
            "control_sha256": "b" * 64, "counts": dict.fromkeys(self.ns["COUNT_KEYS"][:3], 1),
        }
        return {rank: {"training_state": {"run_identity_sha256": "c" * 64},
                       self.ns["PACKET_FIELD"]: copy.deepcopy(proof)} for rank in (0, 1)}

    def test_controller_checks_six_real_identity_proofs_and_preserves_ordinary_reports(self):
        self.ns["_ORIGINAL_INVOCATION"] = lambda *args, **kwargs: (
            self.private_reports(), {"wire": 1},
        )
        for _ in range(3):
            reports, totals = self.ns["guarded_invocation"]("synthetic")
            self.assertEqual(set(reports), {0, 1})
            self.assertEqual(totals, {"wire": 1})
            self.assertTrue(all(self.ns["PACKET_FIELD"] not in value for value in reports.values()))
        self.assertEqual(self.ns["_COUNTS"], dict.fromkeys(self.ns["COUNT_KEYS"], 6))
        self.assertEqual(self.ns["_INVOCATIONS"], 3)
        self.assertEqual(self.ns["_RUN_IDENTITY"], "c" * 64)
        with self.assertRaises(self.ns["GuardianError"]):
            self.ns["guarded_invocation"]("no-fourth-invocation")

    def test_controller_rejects_source_control_run_identity_and_boolean_proof_counts(self):
        for field, bad in (
            ("source_identity_sha256", "f" * 64), ("control_sha256", "0" * 64),
            ("adapter_sha256", "0" * 64), ("run_identity_sha256", "d" * 64),
        ):
            reports = self.private_reports()
            reports[0][self.ns["PACKET_FIELD"]][field] = bad
            self.ns["_ORIGINAL_INVOCATION"] = lambda data=reports: (data, {})
            with self.subTest(field=field), self.assertRaises(self.ns["GuardianError"]):
                self.ns["guarded_invocation"]()
        reports = self.private_reports()
        reports[0][self.ns["PACKET_FIELD"]]["counts"]["worker_admissions"] = True
        self.ns["_ORIGINAL_INVOCATION"] = lambda: (reports, {})
        with self.assertRaises(self.ns["GuardianError"]):
            self.ns["guarded_invocation"]()


class GuardedWorkerSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(ADAPTER.read_text(encoding="utf-8"))
        cls.probe_raw, cls.fixture_raw = PROBE.read_bytes(), FIXTURE.read_bytes()
        cls.probe, cls.fixture = ast.parse(cls.probe_raw), ast.parse(cls.fixture_raw)
        cls.admission = ast.parse(ADMISSION.read_text(encoding="utf-8"))

    def test_accepted_probe_and_fixture_bytes_stay_frozen(self):
        self.assertEqual(hashlib.sha256(self.probe_raw).hexdigest(),
                         literal(self.tree, "PROBE_SHA256"))
        self.assertEqual(len(self.probe_raw), literal(self.tree, "PROBE_SIZE"))
        self.assertEqual(hashlib.sha256(self.fixture_raw).hexdigest(),
                         "53c50c92965ed1cef0c9624c57367c31ed7ebfb9d7cfb34257188ad179d333d0")

    def test_control_embeds_exact_six_accepted_software_binding_constants(self):
        binding = literal(self.tree, "SOFTWARE_BINDING")
        expected = dict(zip((
            "image_id", "source_commit", "software_lock_sha256", "base_inventory_sha256",
            "source_inventory_sha256", "acceptance_report_sha256",
        ), (literal(self.admission, key) for key in (
            "IMAGE_ID", "SOURCE_COMMIT", "LOCK_SHA256", "BASE_SHA256", "SOURCE_SHA256",
            "ACCEPTANCE_SHA256",
        )), strict=True))
        self.assertEqual(binding, expected)
        ns = namespace(self.tree)
        selected = copy.deepcopy(function(self.admission, "source_identity_sha256"))
        selected.name = "reference_source_identity"
        module = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
        exec(compile(module, "<synthetic-reference-identity>", "exec"), ns)
        reference = SimpleNamespace(**binding, __slots__=tuple(binding))
        self.assertEqual(ns["source_identity"](binding), ns["reference_source_identity"](reference))

    def test_full_report_contract_is_identical_to_accepted_pure_report_contract(self):
        reference = ast.parse(ACCEPTED_REPORT.read_text(encoding="utf-8"))
        for key in (
            "CONTROL_FIELDS", "ORIGIN_FIELDS", "OBSERVER_TRUE_FLAGS", "OBSERVER_FALSE_FLAGS",
            "OBSERVER_COUNTS", "WORKER_TRUE_FLAGS", "WORKER_FALSE_FLAGS", "WORKER_COUNTS",
            "WORKER_COUNT_LIMITS",
        ):
            with self.subTest(binding=key):
                self.assertEqual(literal(self.tree, key), literal(reference, key))

    def test_adapter_control_paths_modes_pins_and_caps_are_explicit(self):
        self.assertEqual(literal(self.tree, "ADAPTER_PATH"),
                         "/opt/lumi-release/guardian/worker_entry.py")
        self.assertEqual(literal(self.tree, "CONTROL_PATH"),
                         "/opt/lumi-release/guardian/admission.json")
        self.assertEqual(literal(self.tree, "MAXIMUM_CONTROL_BYTES"), 4096)
        self.assertEqual(literal(self.tree, "MAXIMUM_REPORT_BYTES"), 32768)
        self.assertEqual(literal(self.tree, "MAXIMUM_RECEIPT_BYTES"), 2048)
        reader = function(self.tree, "pinned_file")
        opening = calls(reader, "os.open")[0]
        self.assertTrue(equal(opening.args[1],
                              "os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK"))
        self.assertIn("stat.S_IMODE(before.st_mode) == 292", ast.unparse(reader))
        self.assertEqual(len(calls(reader, "os.close")), 1)

    def test_only_exact_separately_verified_adapter_gets_extra_file_origin(self):
        selected = function(self.tree, "adapter_file_origin")
        branch = selected.body[0]
        self.assertIsInstance(branch, ast.If)
        self.assertTrue(equal(branch.test, "type(path) is str and path == ADAPTER_PATH"))
        self.assertTrue(equal(branch.body[0].value,
                              "('source', ADAPTER_PATH, verify_adapter())"))
        self.assertEqual(len(calls(selected, "_ORIGINAL_FILE_ORIGIN")), 1)
        mutations = [target for node in ast.walk(self.tree) if isinstance(node, ast.Assign)
                     for target in node.targets if isinstance(target, ast.Subscript)
                     and "_STATE" in ast.unparse(target)]
        self.assertFalse(mutations)

    def test_native_rank_pre_ml_guard_is_before_original_target(self):
        rank = function(self.tree, "guarded_rank")
        original = calls(rank, "_ORIGINAL_RANK")
        self.assertEqual(len(original), 1)
        self.assertLess(calls(rank, "ml_absent")[0].lineno, original[0].lineno)
        self.assertLess(calls(rank, "load_control")[0].lineno, original[0].lineno)
        self.assertLess(calls(rank, "verify_adapter")[0].lineno, original[0].lineno)
        self.assertFalse(calls(rank, "_ORIGINAL_IMPORT"))
        self.assertEqual(len(calls(rank, "os._exit")), 1)

    def test_hooks_are_fixture_local_and_preserve_original_probe_observers(self):
        prepare = function(self.tree, "prepare")
        changed = {}
        for node in ast.walk(prepare):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        changed[ast.unparse(target)] = node.value
                    elif isinstance(target, ast.Tuple):
                        for item in target.elts:
                            if isinstance(item, ast.Attribute):
                                changed[ast.unparse(item)] = node.value
        self.assertEqual(set(changed), {
            "_PROBE.file_origin", "_FIXTURE.rank_worker", "_FIXTURE.outcome",
            "_FIXTURE.run_invocation", "_FIXTURE.importlib",
        })
        self.assertTrue(equal(changed["_FIXTURE.importlib"],
                              "SimpleNamespace(import_module=fixture_import)"))
        self.assertFalse(any(isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Attribute) and equal(target.value, "importlib")
            for target in node.targets) for node in ast.walk(self.tree)))
        self.assertEqual(len(calls(prepare, "_PROBE.prepare")), 1)

    def test_frozen_identity_literal_has_one_constructor_site_and_real_worker_keeps_agreement(self):
        constructor = calls(function(self.fixture, "rank_worker"), "implementation.WorkerRun")
        self.assertEqual(len(constructor), 1)
        identity = next(key.value for key in constructor[0].keywords
                        if key.arg == "source_identity_sha256")
        self.assertTrue(equal(identity, "'f' * 64"))
        worker = ast.parse(WORKER.read_text(encoding="utf-8"))
        self.assertEqual(len(calls(function(worker, "run"), "self._agree")), 3)
        validation = calls(function(worker, "run"), "validate_state")
        self.assertEqual(len(validation), 1)
        self.assertTrue(any(key.arg == "run_identity_sha256" and equal(key.value, "self._identity")
                            for key in validation[0].keywords))
        bridge = function(self.tree, "construct_worker")
        self.assertEqual(len(calls(bridge, "_WORKER_MODULE.WorkerRun")), 1)
        self.assertEqual(len(calls(bridge, "_WORKER_MODULE._digest")), 1)

    def test_native_entry_is_main_and_spawn_only_and_failure_exit_is_two(self):
        guards = [node for node in self.tree.body if isinstance(node, ast.If)]
        self.assertEqual(len(guards), 1)
        self.assertTrue(equal(guards[0].test, "__name__ in ('__main__', '__mp_main__')"))
        entry = function(self.tree, "entry")
        self.assertEqual(len(calls(entry, "_PROBE.run")), 1)
        self.assertEqual(len(calls(entry, "prepare")), 1)
        self.assertTrue(any(isinstance(node, ast.Assign)
                            and equal(node.value, "(result('failed', fixed), 2)")
                            for node in ast.walk(entry)))
        self.assertFalse(calls(self.tree, "print"))
        self.assertFalse(calls(self.tree, "traceback.print_exc"))

    def test_test_extraction_excludes_all_native_entry_file_reader_and_rank_functions(self):
        forbidden = {"entry", "prepare", "guarded_rank", "runtime_guard", "pinned_file",
                     "load_control", "verify_adapter", "adapter_file_origin"}
        self.assertTrue(set(PURE_FUNCTIONS + DOUBLE_FUNCTIONS).isdisjoint(forbidden))
        for name in PURE_FUNCTIONS:
            node = function(self.tree, name)
            names = {ast.unparse(call.func) for call in ast.walk(node)
                     if isinstance(call, ast.Call)}
            self.assertTrue(names.isdisjoint({"open", "os.open", "os.read", "subprocess.run"}))


if __name__ == "__main__":
    unittest.main()
