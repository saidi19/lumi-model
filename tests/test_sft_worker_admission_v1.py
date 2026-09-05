"""Stdlib admission tests; read only one hash-pinned control metadata draft.

All targets, receipts, approvals and callbacks below are visibly synthetic.
No checkpoint, encoded input, private row, network, subprocess or ML is used.
Run with Python -S -B; the normal historical package initializer is not safe to
use as a pre-ML bootstrap in environments that expose ML packages.
"""

from __future__ import annotations

import copy
import hashlib
import sys
import threading
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from lumi import posttraining_sft_launch_contract_v1 as launch
from lumi import posttraining_sft_worker_admission_v1 as admission


DRAFT_FILE_SHA256 = "4ff643554ab1f707c8927eeec5c4e07498988f77eced804ed07610bb869c491d"


def synthetic_sha(label: str) -> str:
    return hashlib.sha256(("SYNTHETIC-NO-IO:" + label).encode("ascii")).hexdigest()


def fill_synthetic(reference: dict) -> dict:
    contract = copy.deepcopy(reference)
    target = {
        **admission.SYNTHETIC_PATHS,
        "host_id": admission.SYNTHETIC_HOST,
        "gpu_uuids": [f"GPU-SYNTHETIC-{rank}" for rank in range(8)],
        "runtime_lock_sha256": synthetic_sha("runtime"),
        "sft_source_commit": "b" * 40,
        "local_rendezvous": {
            "address": "127.0.0.1", "port": 29501, "node_count": 1, "maximum_restarts": 0
        },
    }
    contract["target"] = target
    base = reference["source"]["historical_binding"]
    contract["import_receipt"] = {
        "file_sha256": synthetic_sha("import-file"),
        "internal_sha256": synthetic_sha("import-internal"),
        "verification_file_sha256": synthetic_sha("import-verification"),
        "host_id": target["host_id"],
        "checkpoint_directory": target["checkpoint_directory"],
        "real_4b_import": True,  # A synthetic claim; no authentic receipt or permit.
        "original_dcp_manifest_sha256": base["checkpoint"]["manifest_sha256"],
        "hf_checksum_manifest_sha256": synthetic_sha("hf-manifest"),
        "tokenizer_manifest_sha256": base["tokenizer"]["manifest_sha256"],
        "final_base_binding_sha256": launch.BASE_SHA,
    }
    origin = contract["data"]["origin_loader_hashes"]
    contract["relocation_receipt"] = {
        "file_sha256": synthetic_sha("relocation-file"),
        "internal_sha256": synthetic_sha("relocation-internal"),
        "verification_file_sha256": synthetic_sha("relocation-verification"),
        "host_id": target["host_id"],
        "encoded_directory": target["encoded_directory"],
        "content_verified": True,
        "private_modes_verified": True,
        "origin_metadata_inventory_sha256": origin["metadata_inventory_sha256"],
        "loader_hashes": {**origin, "metadata_inventory_sha256": synthetic_sha("new-host")},
    }
    contract["cuda_acceptance"] = {
        "criteria_file_sha256": synthetic_sha("criteria"),
        "host_id": target["host_id"],
        "runtime_lock_sha256": target["runtime_lock_sha256"],
        "criteria": ["masking", "precision", "memory", "import", "fresh_process_resume", "fault_handling"],
    }
    counts = launch.geometry(contract["profile"])
    contract["resource_caps"] = {
        **{field: 10_000_000 for field in launch.CAP_FIELDS},
        "maximum_combined_runtime_seconds": 60,
        "total_host_hourly_rate_microusd": 3_600_000,
        "maximum_cost_microusd": 60_000,
        "maximum_encoded_binary_read_bytes": counts["maximum_selected_binary_bytes"],
        "maximum_encoded_index_read_bytes": counts["maximum_selected_index_bytes"],
        "maximum_invocations": 3,
        "maximum_global_optimizer_steps": 4,
        "maximum_external_network_bytes": 0,
        "maximum_inference_calls": 0,
    }
    resign_synthetic(contract)
    return contract


def resign_synthetic(contract: dict) -> None:
    """Test fixture only; never an owner approval or usable execution credential."""
    contract["execution_authorization"] = {
        "schema_version": launch.AUTH_SCHEMA,
        "status": "owner_approved",
        "run_kind": "sft_gpu_smoke_and_fresh_resume_only",
        "scope": launch.SCOPE,
        "subject_sha256": launch.authorization_subject(contract),
        "owner_approval_text_sha256": admission.SYNTHETIC_APPROVAL_TEXT_SHA256,
        "not_before_unix": 1000,
        "expires_unix": 2000,
        "gpu_execution_approved": True,
        "production_eligible": False,
    }


def pins_for(contract: dict, now: int = 1500) -> admission.ExternalPins:
    return admission.ExternalPins(
        launch.digest(contract), launch.digest(contract["execution_authorization"]), now
    )


class AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = Path(__file__).resolve().parents[1] / "configs/lumi-4b-sft-gpu-smoke-launch-contract-v0.1.json"
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != DRAFT_FILE_SHA256:
            raise AssertionError("frozen_control_metadata_file_changed")
        cls.reference = launch.strict_json(raw)
        if launch.digest(cls.reference) != admission.FROZEN_DRAFT_SHA256:
            raise AssertionError("frozen_control_metadata_canonical_hash_changed")

    def setUp(self) -> None:
        self.contract = fill_synthetic(self.reference)
        self.pins = pins_for(self.contract)
        self.calls: list[int] = []

    def callbacks(self, reject: int | None = None, error: BaseException | None = None):
        def callback(index):
            def invoke():
                self.calls.append(index)
                if index == reject:
                    if error is not None:
                        raise error
                    return admission.SyntheticOutcome.REJECT
                return admission.SyntheticOutcome.ACCEPT
            return invoke
        return admission.SyntheticCallbacks(*(callback(index) for index in range(6)))

    def assert_denied(self, operation, code):
        with self.assertRaises(admission.AdmissionDenied) as caught:
            operation()
        self.assertIs(caught.exception.code, code)
        self.assertEqual(str(caught.exception), code.value)
        return caught.exception

    def test_valid_metadata_never_grants_execution_or_authenticates_receipts(self):
        result = admission.inspect_metadata(self.contract, self.pins)
        self.assertEqual(result.contract_sha256, self.pins.contract_sha256)
        self.assertFalse(result.execution_permitted)
        self.assertFalse(result.receipt_contents_authenticated)
        self.assertFalse(result.operating_system_supervision_attached)
        self.assertEqual(self.calls, [])

    def test_success_does_not_mutate_contract(self):
        before = launch.canonical(self.contract)
        admission.inspect_metadata(self.contract, self.pins)
        self.assertEqual(launch.canonical(self.contract), before)

    def test_inspection_is_frozen_but_not_a_production_permit(self):
        result = admission.inspect_metadata(self.contract, self.pins)
        with self.assertRaises(FrozenInstanceError):
            result.execution_permitted = True
        self.assert_denied(lambda: admission.run_worker(result), admission.Code.EXECUTION_UNAVAILABLE)

    def test_missing_pins_rejected_without_callbacks(self):
        for pins in (None, {}, self.contract["execution_authorization"]):
            with self.subTest(kind=type(pins).__name__):
                self.assert_denied(
                    lambda: admission.inspect_metadata(self.contract, pins), admission.Code.INVALID_PINS
                )
        self.assertEqual(self.calls, [])

    def test_pin_shapes_and_boolean_time_rejected(self):
        for contract_sha, auth_sha, now in (
            (None, self.pins.authorization_sha256, 1500),
            ("A" * 64, self.pins.authorization_sha256, 1500),
            (self.pins.contract_sha256, "a", 1500),
            (self.pins.contract_sha256, self.pins.authorization_sha256, True),
            (self.pins.contract_sha256, self.pins.authorization_sha256, 0),
        ):
            with self.subTest(now=now):
                pins = admission.ExternalPins(contract_sha, auth_sha, now)
                self.assert_denied(
                    lambda: admission.inspect_metadata(self.contract, pins), admission.Code.INVALID_PINS
                )

    def test_contract_change_with_original_pin_rejected_first(self):
        self.contract["target"]["host_id"] = "CHANGED"
        self.assert_denied(
            lambda: admission.inspect_metadata(self.contract, self.pins),
            admission.Code.CONTRACT_PIN_MISMATCH,
        )

    def test_changed_authorization_external_pin_rejected(self):
        pins = admission.ExternalPins(self.pins.contract_sha256, synthetic_sha("wrong"), 1500)
        self.assert_denied(
            lambda: admission.inspect_metadata(self.contract, pins), admission.Code.AUTHORIZATION_REJECTED
        )

    def test_all_null_and_missing_slots_reject_before_any_callbacks(self):
        for slot in launch.SLOTS:
            for missing in (False, True):
                with self.subTest(slot=slot, missing=missing):
                    changed = copy.deepcopy(self.contract)
                    if missing:
                        del changed[slot]
                    else:
                        changed[slot] = None
                    pins = admission.ExternalPins(launch.digest(changed), self.pins.authorization_sha256, 1500)
                    self.assert_denied(
                        lambda: admission.inspect_metadata(changed, pins), admission.Code.REQUIRED_SLOT_MISSING
                    )
        self.assertEqual(self.calls, [])

    def test_every_frozen_top_level_field_rejects_even_with_new_pins(self):
        for field in set(self.reference) - set(launch.SLOTS):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.contract)
                changed[field] = "SYNTHETIC-CHANGED" if changed[field] is None else None
                resign_synthetic(changed)
                self.assert_denied(
                    lambda: admission.inspect_metadata(changed, pins_for(changed)),
                    admission.Code.FROZEN_BINDING_CHANGED,
                )

    def test_each_prohibition_cannot_be_disabled(self):
        for field in launch.PROHIBITIONS:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.contract)
                changed["prohibitions"][field] = False
                resign_synthetic(changed)
                self.assert_denied(
                    lambda: admission.inspect_metadata(changed, pins_for(changed)),
                    admission.Code.FROZEN_BINDING_CHANGED,
                )

    def test_other_profile_geometry_cannot_replace_frozen_eight_rank_design(self):
        for field, value in (("world_size", 2), ("context_window", 2048), ("record_limit", 12)):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.contract)
                changed["profile"][field] = value
                resign_synthetic(changed)
                self.assert_denied(
                    lambda: admission.inspect_metadata(changed, pins_for(changed)),
                    admission.Code.FROZEN_BINDING_CHANGED,
                )

    def test_target_receipt_and_path_mismatches_reject_before_callbacks(self):
        changes = (
            ("target", "host_id", "OTHER-HOST"),
            ("target", "checkpoint_directory", "/other/import/checkpoint"),
            ("import_receipt", "real_4b_import", False),
            ("import_receipt", "original_dcp_manifest_sha256", synthetic_sha("bad-dcp")),
            ("relocation_receipt", "content_verified", False),
            ("relocation_receipt", "private_modes_verified", False),
            ("cuda_acceptance", "runtime_lock_sha256", synthetic_sha("wrong-runtime")),
            ("resource_caps", "maximum_inference_calls", 1),
            ("resource_caps", "maximum_external_network_bytes", 1),
        )
        for slot, field, value in changes:
            with self.subTest(slot=slot, field=field):
                changed = copy.deepcopy(self.contract)
                changed[slot][field] = value
                resign_synthetic(changed)
                self.assert_denied(
                    lambda: admission.inspect_metadata(changed, pins_for(changed)),
                    admission.Code.CONTRACT_REJECTED,
                )
        self.assertEqual(self.calls, [])

    def test_origin_host_metadata_cannot_qualify_relocated_data(self):
        self.contract["relocation_receipt"]["loader_hashes"]["metadata_inventory_sha256"] = (
            self.contract["data"]["origin_loader_hashes"]["metadata_inventory_sha256"]
        )
        resign_synthetic(self.contract)
        self.assert_denied(
            lambda: admission.inspect_metadata(self.contract, pins_for(self.contract)),
            admission.Code.CONTRACT_REJECTED,
        )

    def test_expired_and_not_yet_current_approvals_reject(self):
        for now in (999, 2000, 2001):
            with self.subTest(now=now):
                self.assert_denied(
                    lambda: admission.inspect_metadata(self.contract, pins_for(self.contract, now)),
                    admission.Code.AUTHORIZATION_REJECTED,
                )

    def test_current_approval_window_is_half_open(self):
        for now in (1000, 1999):
            result = admission.inspect_metadata(self.contract, pins_for(self.contract, now))
            self.assertFalse(result.execution_permitted)

    def test_generic_pretraining_approval_is_not_sft_approval(self):
        self.contract["execution_authorization"]["run_kind"] = "pretraining"
        self.assert_denied(
            lambda: admission.inspect_metadata(self.contract, pins_for(self.contract)),
            admission.Code.AUTHORIZATION_REJECTED,
        )

    def test_approval_subject_cannot_be_reused_after_target_change(self):
        self.contract["target"]["sft_source_commit"] = "c" * 40
        self.assert_denied(
            lambda: admission.inspect_metadata(self.contract, pins_for(self.contract)),
            admission.Code.AUTHORIZATION_REJECTED,
        )

    def test_invalid_metadata_has_code_only(self):
        for contract in ([], b"private-text", {"unexpected": object()}, {"unexpected": float("nan")}):
            with self.subTest(kind=type(contract).__name__):
                self.assert_denied(
                    lambda: admission.inspect_metadata(contract, self.pins), admission.Code.INVALID_METADATA
                )

    def test_oversized_control_metadata_is_rejected(self):
        self.assert_denied(
            lambda: admission.inspect_metadata({"private": "x" * 65_536}, self.pins),
            admission.Code.METADATA_TOO_LARGE,
        )

    def test_cyclic_control_metadata_is_rejected(self):
        cyclic = {}
        cyclic["self"] = cyclic
        self.assert_denied(
            lambda: admission.inspect_metadata(cyclic, self.pins), admission.Code.INVALID_METADATA
        )

    def test_existing_ml_module_rejects_inspection(self):
        for name in ("torch", "transformers.utils", "cuda", "cupy"):
            with self.subTest(name=name), patch.dict(sys.modules, {name: object()}):
                self.assert_denied(
                    lambda: admission.inspect_metadata(self.contract, self.pins),
                    admission.Code.STDLIB_RUNTIME_REQUIRED,
                )

    def test_bytecode_enabled_runtime_is_rejected(self):
        with patch.object(sys, "dont_write_bytecode", False):
            self.assert_denied(
                lambda: admission.inspect_metadata(self.contract, self.pins),
                admission.Code.STDLIB_RUNTIME_REQUIRED,
            )

    def test_synthetic_stage_order_and_non_authorizing_trace(self):
        result = admission.SyntheticSession().exercise(self.contract, self.pins, self.callbacks())
        self.assertEqual(self.calls, list(range(6)))
        self.assertEqual(result.stages, tuple(admission.SyntheticStage))
        self.assertTrue(result.synthetic_only)
        self.assertFalse(result.execution_permitted)
        self.assertFalse(result.target_receipt_contents_authenticated)
        self.assertFalse(result.operating_system_supervision_attached)

    def test_bad_contract_rejects_before_first_callback(self):
        self.contract["prohibitions"]["no_inference"] = False
        self.assert_denied(
            lambda: admission.SyntheticSession().exercise(self.contract, self.pins, self.callbacks()),
            admission.Code.CONTRACT_PIN_MISMATCH,
        )
        self.assertEqual(self.calls, [])

    def test_synthetic_harness_rejects_real_host_paths_and_owner_text(self):
        for field, value in (("host_id", "real-host"), ("python_path", "/usr/bin/python"), ("gpu_uuids", ["GPU-real"] * 8)):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.contract)
                changed["target"][field] = value
                self.assert_denied(
                    lambda: admission.SyntheticSession().exercise(changed, pins_for(changed), self.callbacks()),
                    admission.Code.SYNTHETIC_DOMAIN_REQUIRED,
                )
        self.contract["execution_authorization"]["owner_approval_text_sha256"] = synthetic_sha("not-marker")
        self.assert_denied(
            lambda: admission.SyntheticSession().exercise(self.contract, pins_for(self.contract), self.callbacks()),
            admission.Code.SYNTHETIC_DOMAIN_REQUIRED,
        )
        self.assertEqual(self.calls, [])

    def test_each_stage_failure_prevents_later_stages(self):
        failures = (
            admission.Code.TARGET_REJECTED, admission.Code.SUPERVISOR_REJECTED,
            admission.Code.DATA_REJECTED, admission.Code.ML_REJECTED,
            admission.Code.CUDA_REJECTED, admission.Code.WORKER_REJECTED,
        )
        for index, code in enumerate(failures):
            with self.subTest(stage=index):
                self.calls.clear()
                self.assert_denied(
                    lambda: admission.SyntheticSession().exercise(self.contract, self.pins, self.callbacks(reject=index)),
                    code,
                )
                self.assertEqual(self.calls, list(range(index + 1)))

    def test_callback_exception_and_interrupt_text_never_escape(self):
        for error in (RuntimeError("PRIVATE EXCEPTION TEXT"), KeyboardInterrupt("PRIVATE INTERRUPT")):
            with self.subTest(kind=type(error).__name__):
                self.calls.clear()
                caught = self.assert_denied(
                    lambda: admission.SyntheticSession().exercise(self.contract, self.pins, self.callbacks(reject=0, error=error)),
                    admission.Code.TARGET_REJECTED,
                )
                self.assertTrue(caught.__suppress_context__)
                self.assertEqual(self.calls, [0])

    def test_callback_true_is_not_typed_acceptance(self):
        callbacks = admission.SyntheticCallbacks(*(lambda: True for _ in range(6)))
        self.assert_denied(
            lambda: admission.SyntheticSession().exercise(self.contract, self.pins, callbacks),
            admission.Code.TARGET_REJECTED,
        )

    def test_callback_schema_must_be_exact_and_callable(self):
        for callbacks in ({}, admission.SyntheticCallbacks(*(None for _ in range(6)))):
            self.assert_denied(
                lambda: admission.SyntheticSession().exercise(self.contract, self.pins, callbacks),
                admission.Code.INVALID_CALLBACKS,
            )

    def test_session_cannot_retry_after_success(self):
        session = admission.SyntheticSession()
        session.exercise(self.contract, self.pins, self.callbacks())
        self.assert_denied(
            lambda: session.exercise(self.contract, self.pins, self.callbacks()), admission.Code.SESSION_CONSUMED
        )
        self.assertEqual(self.calls, list(range(6)))

    def test_session_cannot_retry_after_metadata_rejection(self):
        session = admission.SyntheticSession()
        self.assert_denied(
            lambda: session.exercise({}, self.pins, self.callbacks()), admission.Code.SYNTHETIC_DOMAIN_REQUIRED
        )
        self.assert_denied(
            lambda: session.exercise(self.contract, self.pins, self.callbacks()), admission.Code.SESSION_CONSUMED
        )
        self.assertEqual(self.calls, [])

    def test_session_cannot_retry_after_started_callback_failure(self):
        session = admission.SyntheticSession()
        self.assert_denied(
            lambda: session.exercise(self.contract, self.pins, self.callbacks(reject=0)), admission.Code.TARGET_REJECTED
        )
        self.assert_denied(
            lambda: session.exercise(self.contract, self.pins, self.callbacks()), admission.Code.SESSION_CONSUMED
        )
        self.assertEqual(self.calls, [0])

    def test_concurrent_session_attempts_allow_only_one_callback_chain(self):
        session = admission.SyntheticSession()
        start = threading.Barrier(3)
        results = []
        def attempt():
            start.wait()
            try:
                results.append(session.exercise(self.contract, self.pins, self.callbacks()))
            except admission.AdmissionDenied as error:
                results.append(error.code)
        threads = [threading.Thread(target=attempt) for _ in range(2)]
        for thread in threads:
            thread.start()
        start.wait()
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(sum(type(value) is admission.SyntheticTrace for value in results), 1)
        self.assertIn(admission.Code.SESSION_CONSUMED, results)
        self.assertEqual(self.calls, list(range(6)))

    def test_direct_worker_entry_always_denies_without_inspection_or_callback(self):
        class Poison:
            def __getattribute__(self, _name):
                raise AssertionError("arguments_must_not_be_inspected")
            def __call__(self):
                raise AssertionError("callback_must_not_be_invoked")
        self.assert_denied(
            lambda: admission.run_worker(contract=Poison(), pins=Poison(), callbacks=Poison(), execute=True),
            admission.Code.EXECUTION_UNAVAILABLE,
        )

    def test_valid_development_sequence_cannot_bypass_public_worker(self):
        trace = admission.SyntheticSession().exercise(self.contract, self.pins, self.callbacks())
        self.calls.clear()
        self.assert_denied(
            lambda: admission.run_worker(self.contract, self.pins, trace, callbacks=self.callbacks()),
            admission.Code.EXECUTION_UNAVAILABLE,
        )
        self.assertEqual(self.calls, [])

    def test_existing_compiler_stays_permanently_closed(self):
        result = launch.analyze(
            self.contract, self.reference,
            expected_authorization_sha256=self.pins.authorization_sha256, now_unix=self.pins.now_unix,
        )
        self.assertFalse(result["execution_permitted"])
        self.assertIsNone(result["executable_argv"])
        with self.assertRaises(PermissionError):
            launch.deny_execution()


if __name__ == "__main__":
    unittest.main()
