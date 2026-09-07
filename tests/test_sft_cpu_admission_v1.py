"""Pure synthetic admission tests; no model, dataset, cloud, or GPU operations."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_test_sft_cpu_admission_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
admission = importlib.import_module(PACKAGE + ".posttraining_sft_cpu_admission_v1")


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.value = admission.fixed_request()
        self.raw = admission.canonical(self.value)
        self.pin = hashlib.sha256(self.raw).hexdigest()

    def validate(self, raw=None, pin=None, **flags):
        raw = self.raw if raw is None else raw
        return admission.validate_request(raw, self.pin if pin is None else pin,
                                          **{"no_site": 1, "no_bytecode": True, **flags})

    def reject(self, callback, code="request_rejected"):
        with self.assertRaises(admission.AdmissionError) as caught:
            callback()
        self.assertEqual(str(caught.exception), code)
        self.assertTrue(caught.exception.__suppress_context__)
        self.assertNotIn("SYNTHETIC-SECRET", repr(caught.exception))

    def reject_value(self, value):
        raw = admission.canonical(value)
        self.reject(lambda: self.validate(raw, hashlib.sha256(raw).hexdigest()))

    def test_fixed_request_roundtrip(self):
        self.assertEqual(self.validate(), self.value)
        self.assertEqual(admission.digest(self.raw), self.pin)
        self.assertLessEqual(len(self.raw), admission.MAXIMUM_REQUEST_BYTES)

    def test_request_is_local_cpu_only(self):
        self.assertEqual(self.value["scope"], "synthetic_cpu_sft_only")
        self.assertEqual(self.value["device"], "cpu")
        self.assertEqual(self.value["backend"], "gloo")
        self.assertEqual(self.value["world_size"], 2)
        for name in ("production_eligible", "real_model_allowed", "external_network_allowed",
                     "inference_allowed"):
            self.assertIs(self.value[name], False)

    def test_fresh_snapshots_cannot_change_contract(self):
        first = admission.fixed_request()
        first["device"] = "cuda"
        self.assertEqual(admission.fixed_request(), self.value)
        validated = self.validate()
        validated["world_size"] = 8
        self.assertEqual(self.validate(), self.value)

    def test_changed_cpu_gpu_or_resource_options_reject_even_with_new_pin(self):
        replacements = {
            "scope": "private_research_nonproduction", "operation": "train_real_model",
            "device": "cuda", "backend": "nccl", "world_size": 8, "precision": "bf16",
            "attention_implementation": "sdpa", "record_limit": 12, "target_steps": 3,
            "per_rank_batch_size": 2, "gradient_accumulation_steps": 2,
            "maximum_runtime_seconds": 181, "maximum_private_write_bytes": 67_108_865,
            "production_eligible": True, "real_model_allowed": True,
            "external_network_allowed": True, "inference_allowed": True,
        }
        for field, changed in replacements.items():
            with self.subTest(field=field):
                self.reject_value({**self.value, field: changed})

    def test_bool_int_float_and_string_substitutions_reject(self):
        for field, original in self.value.items():
            for changed in (None, [], {}, True, False, 0, 1, 2.0, "1"):
                if type(changed) is type(original) and changed == original:
                    continue
                with self.subTest(field=field, changed_type=type(changed).__name__):
                    self.reject_value({**self.value, field: changed})

    def test_every_missing_field_rejects(self):
        for field in self.value:
            with self.subTest(field=field):
                self.reject_value({k: v for k, v in self.value.items() if k != field})

    def test_paths_credentials_commands_and_extra_fields_reject(self):
        for field in ("model_path", "data_path", "output_path", "ssh_host", "pem_path", "token",
                      "command", "argv", "approved", "execution_enabled"):
            with self.subTest(field=field):
                self.reject_value({**self.value, field: "SYNTHETIC-SECRET"})

    def test_wrong_raw_type_size_and_encoding(self):
        for raw in (b"", b"x" * 4097, b"\xff", bytearray(self.raw), self.raw.decode(), True, 1):
            with self.subTest(kind=type(raw).__name__):
                self.reject(lambda raw=raw: self.validate(raw))

    def test_null_input_rejected(self):
        self.reject(lambda: admission.validate_request(None, self.pin,
                                                       no_site=1, no_bytecode=True))

    def test_bad_pins(self):
        for pin in ("f" * 64, "A" * 64, "g" * 64, "a" * 63, "a" * 65,
                    self.pin + "\n", b"a" * 64, None, True):
            with self.subTest(kind=type(pin).__name__):
                self.reject(lambda pin=pin: admission.validate_request(
                    self.raw, pin, no_site=1, no_bytecode=True))

    def test_noncanonical_duplicate_nonfinite_and_nested_json_reject(self):
        raws = [
            json.dumps(self.value).encode(), self.raw + b"\n", b" " + self.raw,
            b'{"device":"cpu",' + self.raw[1:],
            b'{"x":NaN}', b'{"x":Infinity}', b"[" * 1100 + b"0" + b"]" * 1100,
            b"[]", b"null", b'"SYNTHETIC-SECRET"', b"true",
        ]
        for raw in raws:
            with self.subTest(size=len(raw)):
                self.reject(lambda raw=raw: self.validate(raw, hashlib.sha256(raw).hexdigest()))

    def test_schema_is_exact(self):
        self.reject_value({**self.value, "schema_version": admission.SCHEMA + " "})

    def test_wrong_runtime_flags_reject(self):
        for value in (None, False, 0, 2, 1.0, "1"):
            with self.subTest(flag="no_site", value=value):
                self.reject(lambda value=value: self.validate(no_site=value),
                            "pre_ml_runtime_required")
        self.reject(lambda: self.validate(no_site=True), "pre_ml_runtime_required")
        for value in (None, False, 0, 2, 1.0, "1"):
            with self.subTest(flag="no_bytecode", value=value):
                self.reject(lambda value=value: self.validate(no_bytecode=value),
                            "pre_ml_runtime_required")

    def test_boolean_and_integer_no_bytecode_forms_supported(self):
        self.assertEqual(self.validate(no_bytecode=1), self.value)
        self.assertEqual(self.validate(no_bytecode=True), self.value)

    def test_only_exact_cli_is_accepted(self):
        self.assertIsNone(admission.validate_cli(["--run-synthetic-cpu"]))
        for argv in ([], ["--execute"], ["--help"], ["--device", "cpu"],
                     ["--run-synthetic-cpu", "--device", "cuda"],
                     ["--run-synthetic-cpu", "SYNTHETIC-SECRET"],
                     ["--run-synthetic-cpu "], [True], "--run-synthetic-cpu",
                     ("--run-synthetic-cpu",), None):
            with self.subTest(kind=type(argv).__name__):
                self.reject(lambda argv=argv: admission.validate_cli(argv), "offline_cli_rejected")

    def test_error_messages_are_allowlisted(self):
        self.assertEqual(str(admission.AdmissionError("SYNTHETIC-SECRET")), "request_rejected")
        self.assertEqual(str(admission.AdmissionError(True)), "request_rejected")

    def test_nonserializable_and_nonfinite_values_are_redacted(self):
        for value in (object(), {"x": float("nan")}, {"x": float("inf")}, {"x": b"secret"}):
            with self.subTest(kind=type(value).__name__):
                self.reject(lambda value=value: admission.canonical(value))

    def test_digest_rejects_nonbytes_empty_and_oversized(self):
        for value in (b"", b"x" * 4097, self.raw.decode(), None, True, bytearray(self.raw)):
            with self.subTest(kind=type(value).__name__):
                self.reject(lambda value=value: admission.digest(value))

    def test_validation_performs_no_file_reads_or_dynamic_imports(self):
        with patch("builtins.open") as opened, patch.object(importlib, "import_module") as imported:
            self.assertEqual(self.validate(), self.value)
            self.reject_value({**self.value, "device": "cuda"})
        opened.assert_not_called()
        imported.assert_not_called()

    def test_actual_runtime_gate_accepts_clean_observed_runtime(self):
        with patch.object(sys, "flags", types.SimpleNamespace(no_site=1)):
            with patch.object(sys, "dont_write_bytecode", True), patch.object(sys, "modules", {}):
                self.assertIsNone(admission.pre_ml_runtime_check())

    def test_actual_runtime_gate_rejects_every_ml_root_and_submodule(self):
        with patch.object(sys, "flags", types.SimpleNamespace(no_site=1)):
            with patch.object(sys, "dont_write_bytecode", True):
                for name in admission.ML_ROOTS:
                    for key in (name, name + ".submodule"):
                        with self.subTest(root=name), patch.object(sys, "modules", {key: None}):
                            self.reject(admission.pre_ml_runtime_check, "pre_ml_runtime_required")

    def test_actual_runtime_gate_requires_real_startup_flags(self):
        with patch.object(sys, "flags", types.SimpleNamespace(no_site=0)):
            self.reject(admission.pre_ml_runtime_check, "pre_ml_runtime_required")
        with patch.object(sys, "flags", types.SimpleNamespace(no_site=1)):
            with patch.object(sys, "dont_write_bytecode", False):
                self.reject(admission.pre_ml_runtime_check, "pre_ml_runtime_required")


if __name__ == "__main__":
    unittest.main()
