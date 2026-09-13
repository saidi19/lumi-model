"""Generated aggregate receipts and source checks; no native Linux execution."""

from __future__ import annotations

import copy
import importlib.util
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_sft_checkpoint_guard_harness_v1 import case_report

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_generated_qualification_verifier_v2", ROOT / "scripts/verify_sft_import_qualification_v2.py")
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


def generated(sources):
    h, s, a, addon, summary, binding = sources
    cases = [case_report(case) for case in h.CASES]
    cases[-1]["worker"].update(entry_sha256=summary["entry_sha256"],
                              source_manifest_sha256=summary["source_manifest_sha256"])
    export = 1058599424
    tree = {"schema": addon.rootfs.SCHEMA, "archive_bytes": export, "archive_entries": 28984,
            "owned_entries": 29030, "archive_headers": 28987, "metadata_bytes": 319,
            "content_bytes": 1037181068 + summary["addon_bytes"], "hardlinks": 2,
            "symlinks": 844, "installed_files": summary["addon_files"],
            "installed_bytes": summary["addon_bytes"], "installation_frozen": True,
            "cleanup_attempted": False, "cleanup_completed": False, "operation_tainted": False,
            "metadata_normalized": True, "filesystem_immutable": False,
            "production_eligible": False, "real_sft_execution_permitted": False}
    harness = {"schema": h.SCHEMA, "status": "passed", "stage": "complete", "code": "ok",
               "cases": cases, "accepted_image_id": binding.image_id,
               "source_identity_sha256": binding.source_identity_sha256(),
               "archive_bytes": h.ARCHIVE_BYTES, "export_bytes": export,
               "export_sha256": "c" * 64, "extraction": tree,
               "entry_sha256": summary["entry_sha256"],
               "importer_manifest_sha256": summary["source_manifest_sha256"],
               "source_admission_sha256": h.SOURCE_ADMISSION_SHA256, "addon": summary,
               "docker_containers_started": 0, "images_built": 0,
               "private_owned_cleanup_verified": True,
               "cleanup_inventory": {"owned_container_absent": True, "owned_image_absent": True},
               **dict.fromkeys(h.NONCLAIMS, False)}
    raw = subject.canonical(harness)
    native = {"schema": s.SCHEMA, "status": "passed", "code": "ok",
              "harness_sha256": subject.PINS["scripts/run_checkpoint_guard_qualification_v1.py"],
              "harness_stdout_bytes": len(raw), "harness_stdout_sha256": subject.digest(raw),
              "harness_exit_success": True, "harness_pidfd_bound": True,
              "harness_term_attempts": 0, "harness_kill_attempts": 0, "harness_reaped": True,
              "adopted_children_reaped": 0, "all_children_reaped": True, "harness_report": harness,
              "guardian_signalled_by_supervisor": False, "process_group_signalled": False,
              "guardian_death_cleanup_verified": False, "job_teardown_cleanup_verified": False,
              "source_immutability_verified": False, "real_4b_import": False,
              "training_authorized": False, "production_eligible": False, "gpu_used": False}
    return subject.expected_acquisition(a), native, subject.expected_retirement(a)


def rehash(native):
    raw = subject.canonical(native["harness_report"])
    native.update(harness_stdout_bytes=len(raw), harness_stdout_sha256=subject.digest(raw))


class VerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous = list(sys.meta_path)
        before = set(sys.modules)
        cls.sources = subject.load_sources()
        cls.added_modules = set(sys.modules) - before

    @classmethod
    def tearDownClass(cls):
        sys.meta_path[:] = cls.previous

    def setUp(self):
        self.acq, self.native, self.retired = generated(self.sources)

    def invoke(self):
        return subject.validate(self.acq, self.native, self.retired, self.sources)

    def test_complete_generated_three_case_receipts_pass(self):
        result = self.invoke()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["ordered_cases_verified"], 3)
        self.assertTrue(result["all_owned_cleanup_receipts_verified"])
        for key in subject.FALSE_FLAGS:
            self.assertIs(result[key], False)

    def test_source_only_load_has_no_ML_or_native_action(self):
        self.assertFalse({"torch", "transformers", "safetensors", "numpy"}
                         & self.added_modules)
        self.assertEqual(self.sources[4]["source_files"], 35)
        self.assertEqual(self.sources[4]["source_bytes"], 610752)
        self.assertEqual(self.sources[4]["addon_files"], 37)

    def test_acquisition_extra_wrong_source_bool_request_or_false_cleanup_denies(self):
        for key, value in (("https_requests", True), ("source_commit", "0" * 40),
                           ("prior_input_cleanup_verified", False), ("unexpected", 1)):
            before = copy.deepcopy(self.acq)
            self.acq[key] = value
            with self.assertRaises(subject.VerificationError):
                self.invoke()
            self.acq = before

    def test_missing_retirement_or_changed_count_denies(self):
        for candidate in ({}, {**self.retired, "owned_input_files_retired": 2},
                          {**self.retired, "production_eligible": True}):
            self.retired = candidate
            with self.assertRaises(subject.VerificationError):
                self.invoke()

    def test_outer_cleanup_signals_and_adoption_denied(self):
        for key, value in (("adopted_children_reaped", 1), ("all_children_reaped", False),
                           ("harness_term_attempts", 1), ("harness_pidfd_bound", False),
                           ("guardian_death_cleanup_verified", True)):
            before = copy.deepcopy(self.native)
            self.native[key] = value
            with self.assertRaises(subject.VerificationError):
                self.invoke()
            self.native = before

    def test_harness_transport_hash_or_byte_count_tampering_denies(self):
        for key, value in (("harness_stdout_sha256", "0" * 64), ("harness_stdout_bytes", 1)):
            before = self.native[key]
            self.native[key] = value
            with self.assertRaises(subject.VerificationError):
                self.invoke()
            self.native[key] = before

    def test_case_reordering_and_missing_case_denied_even_with_correct_hash(self):
        self.native["harness_report"]["cases"].reverse()
        rehash(self.native)
        with self.assertRaises(ValueError):
            self.invoke()
        self.native["harness_report"]["cases"].pop()
        rehash(self.native)
        with self.assertRaises(subject.VerificationError):
            self.invoke()

    def test_positive_worker_binding_and_anchor_cannot_be_forged(self):
        for key, value in (("entry_sha256", "0" * 64), ("status", "failed"),
                           ("training_authorized", True)):
            before = copy.deepcopy(self.native)
            self.native["harness_report"]["cases"][-1]["worker"][key] = value
            rehash(self.native)
            with self.assertRaises(ValueError):
                self.invoke()
            self.native = before

    def test_exact_negative_rejection_and_exit_code_required(self):
        self.native["harness_report"]["cases"][0]["worker_exit_code"] = 0
        rehash(self.native)
        with self.assertRaises(ValueError):
            self.invoke()

    def test_rootfs_cleanup_state_and_addon_counts_denied(self):
        for key, value in (("cleanup_attempted", True), ("operation_tainted", True),
                           ("installed_files", 2), ("content_bytes", 2**33)):
            before = copy.deepcopy(self.native)
            self.native["harness_report"]["extraction"][key] = value
            rehash(self.native)
            with self.assertRaises(subject.VerificationError):
                self.invoke()
            self.native = before

    def test_changed_image_or_export_or_unexpected_field_denied(self):
        for key, value in (("accepted_image_id", "sha256:" + "0" * 64),
                           ("export_bytes", 2**32), ("extra", "value")):
            before = copy.deepcopy(self.native)
            self.native["harness_report"][key] = value
            rehash(self.native)
            with self.assertRaises(subject.VerificationError):
                self.invoke()
            self.native = before

    def test_duplicate_noncanonical_float_path_text_and_oversize_rejected(self):
        for raw in (b'{"x":1,"x":2}\n', b'{"x": 1}\n', b'{"x":1.5}\n',
                    b'{"x":NaN}\n', b"x" * (subject.MAXIMUM + 1)):
            with self.assertRaises(subject.VerificationError):
                subject.decode(raw)
        self.native["harness_report"]["cases"][0]["worker"]["code"] = "/private/secret"
        rehash(self.native)
        with self.assertRaises(ValueError):
            self.invoke()

    def test_generated_files_verify_and_only_new_result_is_written(self):
        with tempfile.TemporaryDirectory(prefix="lumi-generated-verification-") as temporary:
            root = Path(temporary).resolve()
            output = root / subject.OUTPUT
            output.mkdir(parents=True, mode=0o700)
            for name, value in zip(subject.FILES, (self.acq, self.native, self.retired),
                                   strict=True):
                path = output / name
                path.write_bytes(subject.canonical(value))
                path.chmod(0o600)
            with patch.object(subject, "ROOT", root), \
                    patch.object(subject, "load_sources", return_value=self.sources):
                result = subject.verify_receipts()
                subject.write_verification(result)
                before = (output / "aggregate-verification.json").read_bytes()
                with self.assertRaises(FileExistsError):
                    subject.write_verification(result)
                self.assertEqual((output / "aggregate-verification.json").read_bytes(), before)

    def test_symlink_oversize_and_group_writable_read_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lumi-generated-read-") as temporary:
            root = Path(temporary).resolve()
            path = root / "aggregate.json"
            path.write_bytes(b"{}\n")
            link = root / "link"
            link.symlink_to(path)
            with self.assertRaises(subject.VerificationError):
                subject.read(link, subject.MAXIMUM)
            with self.assertRaises(subject.VerificationError):
                subject.read(path, 2)
            path.chmod(0o666)
            with self.assertRaises(subject.VerificationError):
                subject.read(path, subject.MAXIMUM)

    def test_failed_receipt_main_emits_only_fixed_error(self):
        output = types.SimpleNamespace(buffer=io.BytesIO())
        with patch.object(subject.sys, "argv", ["verify", "--verify-receipts"]), \
                patch.object(subject.sys, "stdout", output), \
                patch.object(subject, "verify_receipts", side_effect=ValueError("private value")), \
                patch.object(subject, "write_verification") as write:
            self.assertEqual(subject.main(), 2)
        write.assert_called_once()
        self.assertNotIn(b"private value", output.buffer.getvalue())
        self.assertEqual(subject.decode(output.buffer.getvalue())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
