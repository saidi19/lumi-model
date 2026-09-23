"""Owned synthetic files and frozen definitions only; no acquisition or native work."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load("scripts/prepare_sft_prepared_qualification_v3.py", "_sft_prepare_v3_contract")
verifier = load("scripts/verify_sft_prepared_qualification_v3.py", "_sft_verify_v3_contract")
prefix = load("scripts/reclaim_sft_sdk_prefix_v1.py", "_sft_prefix_v3_contract")
probe = load("scripts/probe_sft_prepared_setup_v1.py", "_sft_probe_v3_contract")
fixtures = load(
    "tests/test_sft_checkpoint_prepared_qualification_supervisor_v2.py",
    "_sft_v3_supervisor_fixtures",
)


def local(module, **overrides):
    return types.SimpleNamespace(**{**vars(module), **overrides})


def changed_info(info, **changes):
    return types.SimpleNamespace(
        **{
            **{key: getattr(info, key) for key in dir(info) if key.startswith("st_")},
            **changes,
        }
    )


def space(enough=True):
    return types.SimpleNamespace(f_bavail=(32 if enough else 1) * 1024**3 // 4096, f_frsize=4096)


def prefix_receipt():
    setup = probe.empty_report()
    setup.update(
        status="observed",
        stage="complete",
        code="ok",
        source_bound_files=78,
        owned_temp_directories_created=2,
        owned_temp_directories_removed=2,
        setup_subreaper_installed=True,
        ready_for_unchanged_setup=True,
    )
    setup.update(probe.capacity(space()))
    value = prefix.empty_report()
    value.update(
        status="observed",
        stage="complete",
        code="ok",
        selected_shards=2,
        audited_shards=2,
        deleted_shards=2,
        census_nodes=4,
        census_name_bytes=32,
        census_allocated_bytes=8192,
        removed_leaves=2,
        removed_directories=2,
        audit_complete=True,
        removal_started=True,
        sdk_root_preserved=True,
        selected_absence_verified=True,
        unselected_root_names_unchanged=True,
        capacity_before=probe.capacity(space(False)),
        capacity_after=probe.capacity(space()),
        setup_report=setup,
        ready_for_unchanged_setup=True,
    )
    return prefix.validate_report(value, probe)


def native_receipts(frozen, sources):
    harness, supervisor, acquisition, _workspaces, _bundles, count, binding = sources
    report = fixtures.report()
    report.update(
        source_bound_files=count,
        source_admission_sha256=harness.SOURCE_ADMISSION_SHA256,
        source_identity_sha256=binding.source_identity_sha256(),
        accepted_image_id=binding.image_id,
    )
    raw = frozen.canonical(report)
    native = {
        "schema": supervisor.SCHEMA,
        "status": "passed",
        "code": "ok",
        "harness_sha256": frozen.PINS["scripts/run_checkpoint_prepared_qualification_v2.py"],
        "harness_stdout_bytes": len(raw),
        "harness_stdout_sha256": frozen.digest(raw),
        "harness_exit_success": True,
        "harness_pidfd_bound": True,
        "harness_term_attempts": 0,
        "harness_kill_attempts": 0,
        "harness_reaped": True,
        "adopted_children_reaped": 0,
        "all_children_reaped": True,
        "harness_report": report,
        "guardian_signalled_by_supervisor": False,
        "process_group_signalled": False,
        "guardian_death_cleanup_verified": False,
        "job_teardown_cleanup_verified": False,
        "source_immutability_verified": False,
        **dict.fromkeys(supervisor.FALSE_FLAGS, False),
    }
    return frozen.expected_acquisition(acquisition), native, frozen.expected_retirement(acquisition)


class TemporaryTests(unittest.TestCase):
    def setUp(self):
        folder = self.enterContext(tempfile.TemporaryDirectory(prefix="sft-v3-synthetic-"))
        self.directory = Path(folder).resolve()

    def file(self, raw=b"{}\n", name="aggregate.json"):
        result = self.directory / name
        result.write_bytes(raw)
        result.chmod(0o600)
        return result


class AdapterAuthenticationTests(TemporaryTests):
    def setUp(self):
        super().setUp()
        self.enterContext(patch.object(adapter, "_MODULE", None))

    def test_frozen_source_authenticated_and_only_successor_bindings_change(self):
        frozen = adapter.load_frozen()
        original = load("scripts/prepare_sft_prepared_qualification_v2.py", "_v3_original")
        self.assertEqual(
            hashlib.sha256(adapter.FROZEN.read_bytes()).hexdigest(), adapter.FROZEN_SHA256
        )
        changed = {"SCHEMA", "WORKFLOW", "CI_KEY", "APPROVED_PARENT", "OUTPUT"}
        for name in (key for key in vars(original) if key.isupper()):
            expected = getattr(adapter if name in changed else original, name)
            self.assertEqual(getattr(frozen, name), expected, name)

    def test_hash_mismatch_rejects_before_executing_source(self):
        path = self.file(b"raise AssertionError('must never execute')\n", "prepare.py")
        with patch.object(adapter, "FROZEN", path), self.assertRaises(adapter.AdapterError):
            adapter.load_frozen()
        self.assertIsNone(adapter._MODULE)

    def test_symlink_parent_world_write_and_hardlink_rejected(self):
        path = self.file(adapter.FROZEN.read_bytes(), "prepare.py")
        parent = self.directory / "alias"
        parent.symlink_to(self.directory, target_is_directory=True)
        link = self.directory / "linked.py"
        link.symlink_to(path)
        for candidate in (link, parent / path.name):
            with (
                patch.object(adapter, "FROZEN", candidate),
                self.assertRaises(adapter.AdapterError),
            ):
                adapter.load_frozen()
        path.chmod(0o666)
        with patch.object(adapter, "FROZEN", path), self.assertRaises(adapter.AdapterError):
            adapter.load_frozen()
        path.chmod(0o600)
        os.link(path, self.directory / "hard.py")
        with patch.object(adapter, "FROZEN", path), self.assertRaises(adapter.AdapterError):
            adapter.load_frozen()

    def test_short_reads_completed_and_identity_drift_rejected(self):
        path = self.file(adapter.FROZEN.read_bytes(), "prepare.py")

        def short(fd, count):
            return os.read(fd, min(count, 17))

        with (
            patch.object(adapter, "FROZEN", path),
            patch.object(adapter, "os", local(os, read=short)),
        ):
            self.assertEqual(adapter.load_frozen().SCHEMA, adapter.SCHEMA)
        adapter._MODULE = None
        before = path.stat()
        states = [before, changed_info(before, st_mtime_ns=before.st_mtime_ns + 1)]
        with (
            patch.object(adapter, "FROZEN", path),
            patch.object(adapter, "os", local(os, fstat=Mock(side_effect=states))),
            self.assertRaises(adapter.AdapterError),
        ):
            adapter.load_frozen()

    def test_cached_module_preserves_consumed_attempt_and_stage(self):
        first = adapter.load_frozen()
        first._ATTEMPTED = True
        first.STAGE = "acquisition"
        self.assertIs(adapter.load_frozen(), first)
        self.assertIs(adapter.load_frozen()._ATTEMPTED, True)
        self.assertEqual(adapter.load_frozen().STAGE, "acquisition")

    def test_dispatch_calls_frozen_main_exactly_once(self):
        frozen = types.SimpleNamespace(main=Mock(return_value=2))
        with patch.object(adapter, "load_frozen", return_value=frozen):
            self.assertEqual(adapter.main(), 2)
        frozen.main.assert_called_once_with()

    def test_binding_exception_is_closed_without_file_writes(self):
        output = io.BytesIO()
        with (
            patch.object(adapter, "load_frozen", side_effect=RuntimeError("private exception")),
            patch.object(adapter, "sys", local(sys, stdout=types.SimpleNamespace(buffer=output))),
        ):
            self.assertEqual(adapter.main(), 2)
        value = json.loads(output.getvalue())
        self.assertEqual(value["code"], "binding_rejected")
        self.assertNotIn(b"private exception", output.getvalue())
        self.assertTrue(all(value[key] is False for key in adapter.FALSE_FLAGS))
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_frozen_runtime_blocks_mac_before_acquisition(self):
        frozen = adapter.load_frozen()
        output = io.BytesIO()
        calls = Mock(side_effect=AssertionError("acquisition forbidden"))
        with (
            patch.object(
                frozen,
                "sys",
                local(
                    sys,
                    platform="darwin",
                    argv=["prepare", "--acquire-accepted-software"],
                    stdout=types.SimpleNamespace(buffer=output),
                ),
            ),
            patch.object(frozen, "acquire", calls),
        ):
            self.assertEqual(adapter.main(), 2)
        calls.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["code"], "acquisition_rejected")


class VerifierReaderTests(TemporaryTests):
    def test_exact_bound_and_short_reads(self):
        path = self.file(b'{"a":1}\n')

        def short(fd, count):
            return os.read(fd, min(count, 2))

        with patch.object(verifier, "os", local(os, read=short)):
            self.assertEqual(verifier.read_closed(path, 8), b'{"a":1}\n')
        with self.assertRaises(verifier.VerificationError):
            verifier.read_closed(path, 7)

    def test_empty_nonregular_symlink_parent_and_hardlink_rejected(self):
        path = self.file()
        alias = self.directory / "alias"
        alias.symlink_to(self.directory, target_is_directory=True)
        empty = self.file(b"", "empty")
        for candidate in (empty, self.directory, alias / path.name):
            with (
                self.subTest(candidate=candidate.name),
                self.assertRaises(verifier.VerificationError),
            ):
                verifier.read_closed(candidate, 32768)
        path.chmod(0o666)
        with self.assertRaises(verifier.VerificationError):
            verifier.read_closed(path, 32768)
        path.chmod(0o600)
        os.link(path, self.directory / "hard")
        with self.assertRaises(verifier.VerificationError):
            verifier.read_closed(path, 32768)

    def test_identity_drift_and_premature_eof_fail_closed(self):
        path = self.file()
        before = path.stat()
        for field in (
            "st_ino",
            "st_uid",
            "st_gid",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        ):
            drift = changed_info(before, **{field: getattr(before, field) + 1})
            with (
                self.subTest(field=field),
                patch.object(verifier, "os", local(os, fstat=Mock(side_effect=[before, drift]))),
                self.assertRaises(verifier.VerificationError),
            ):
                verifier.read_closed(path, 32768)
        with (
            patch.object(verifier, "os", local(os, read=Mock(return_value=b""))),
            self.assertRaises(verifier.VerificationError),
        ):
            verifier.read_closed(path, 32768)

    def test_authentication_before_execution_uses_stable_reader(self):
        path = self.file(b"raise AssertionError('do not execute')\n", "unsafe.py")
        with self.assertRaises(verifier.VerificationError):
            verifier.load_module(path, "0" * 64)
        safe = self.file(b"VALUE = 31\n", "safe.py")
        expected = hashlib.sha256(safe.read_bytes()).hexdigest()
        self.assertEqual(verifier.load_module(safe, expected).VALUE, 31)
        with (
            patch.object(verifier, "read_closed", side_effect=verifier.VerificationError("closed")),
            self.assertRaises(verifier.VerificationError),
        ):
            verifier.authenticated(safe, expected, 32768)

    def test_bad_adapter_pin_cannot_execute_or_enter_native_source_loader(self):
        path = self.file(b"raise AssertionError('must never execute')\n", "adapter.py")
        frozen = types.SimpleNamespace(load_sources=Mock())
        with (
            patch.object(verifier, "load_frozen", return_value=frozen),
            patch.object(verifier, "ADAPTER", path),
            self.assertRaises(verifier.VerificationError),
        ):
            verifier.load_sources()
        frozen.load_sources.assert_not_called()

    def test_decoder_requires_ascii_canonical_integer_object(self):
        self.assertEqual(verifier.decode(b'{"status":"ok"}\n'), {"status": "ok"})
        for raw in (
            b'{"status": "ok"}\n',
            b'{"n":1.0}\n',
            b'{"x":1,"x":2}\n',
            b"[]\n",
            b'{"x":"\xc3\xa9"}\n',
            b'{"x":NaN}\n',
            b"{}",
            b"{}\n\n",
            b'{"n":9223372036854775808}\n',
        ):
            with self.subTest(raw=raw), self.assertRaises(verifier.VerificationError):
                verifier.decode(raw)


class PrefixTests(TemporaryTests):
    def setUp(self):
        super().setUp()
        self.path = self.file(verifier.canonical(prefix_receipt()))
        self.enterContext(patch.object(verifier, "PREFIX_REPORT", self.path))

    def test_success_receipt_returns_bound_digest_without_sdk_action(self):
        report, digest = verifier.verify_prefix()
        self.assertEqual(report, prefix_receipt())
        self.assertEqual(digest, hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_live_capacity_gate_rejects_stale_ready_receipt(self):
        for enough in (False, True):
            with patch.object(verifier, "os", local(os, statvfs=Mock(return_value=space(enough)))):
                if enough:
                    self.assertEqual(verifier.verify_prefix(True)[0], prefix_receipt())
                else:
                    with self.assertRaises(verifier.VerificationError):
                        verifier.verify_prefix(True)

    def test_failed_unknown_and_false_authority_prefix_fields_rejected(self):
        original = prefix_receipt()
        changes = (
            ("status", "failed"),
            ("ready_for_unchanged_setup", False),
            ("sdk_root_preserved", False),
            ("selected_absence_verified", False),
            ("unselected_root_names_unchanged", False),
            ("selected_shards", True),
            ("gpu_used", True),
            ("unexpected", "text must not survive"),
        )
        for key, value in changes:
            self.path.write_bytes(verifier.canonical({**original, key: value}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                verifier.verify_prefix()

    def test_bad_prefix_source_pin_rejected(self):
        with (
            patch.object(verifier, "PREFIX_SHA256", "0" * 64),
            self.assertRaises(verifier.VerificationError),
        ):
            verifier.verify_prefix()


class ReceiptTests(TemporaryTests):
    @classmethod
    def setUpClass(cls):
        previous = list(sys.meta_path)
        cls.addClassCleanup(lambda: sys.meta_path.__setitem__(slice(None), previous))
        cls.frozen, cls.sources = verifier.load_sources()

    def setUp(self):
        super().setUp()
        self.parts = native_receipts(self.frozen, self.sources)
        self.enterContext(
            patch.object(verifier, "load_sources", return_value=(self.frozen, self.sources))
        )
        self.enterContext(
            patch.object(verifier, "verify_prefix", return_value=(prefix_receipt(), "a" * 64))
        )
        self.output = self.directory / verifier.OUTPUT
        self.output.mkdir(parents=True, mode=0o700)
        self.enterContext(patch.object(verifier, "ROOT", self.directory))
        self.write_parts()

    def write_parts(self):
        for name, value in zip(verifier.FILES, self.parts, strict=True):
            path = self.output / name
            path.write_bytes(verifier.canonical(value))
            path.chmod(0o600)

    def test_generated_v3_receipts_use_real_frozen_native_validation(self):
        result = verifier.verify_receipts()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["schema"], verifier.SCHEMA)
        self.assertEqual(result["receipt_count"], 4)
        self.assertEqual(result["source_files_verified"], 78)
        self.assertEqual(result["ordered_guardian_passes_verified"], 2)
        self.assertEqual(result["prefix_capacity_receipt_sha256"], "a" * 64)
        for key in verifier.FALSE_FLAGS:
            self.assertIs(result[key], False)

    def test_old_v2_acquisition_cannot_be_used_as_v3_success(self):
        self.parts[0]["schema"] = "lumi.sft-prepared-qualification-acquisition.v0.2"
        self.write_parts()
        with self.assertRaises(ValueError):
            verifier.verify_receipts()

    def test_failed_native_and_incomplete_cleanup_rejected(self):
        original = copy.deepcopy(self.parts)
        for part, key, value in (
            (1, "status", "failed"),
            (1, "all_children_reaped", False),
            (2, "owned_input_files_retired", 0),
            (0, "network_retries", 1),
        ):
            self.parts = copy.deepcopy(original)
            self.parts[part][key] = value
            self.write_parts()
            with self.subTest(key=key), self.assertRaises(ValueError):
                verifier.verify_receipts()

    def test_failed_prefix_prevents_receipt_success(self):
        with (
            patch.object(verifier, "verify_prefix", side_effect=ValueError("untrusted")),
            self.assertRaises(ValueError),
        ):
            verifier.verify_receipts()

    def test_main_writes_exclusively_and_never_overwrites_existing_verification(self):
        output = io.BytesIO()
        fake = local(
            sys, argv=["verify", "--verify-receipts"], stdout=types.SimpleNamespace(buffer=output)
        )
        with patch.object(verifier, "sys", fake):
            self.assertEqual(verifier.main(), 0)
        destination = self.output / "aggregate-verification.json"
        before = destination.read_bytes()
        self.assertEqual(before, output.getvalue())
        with patch.object(verifier, "sys", fake):
            self.assertEqual(verifier.main(), 2)
        self.assertEqual(destination.read_bytes(), before)

    def test_verification_writer_rejects_symlink_and_nonprivate_parent(self):
        self.output.chmod(0o755)
        with self.assertRaises(verifier.VerificationError):
            verifier.write_verification(verifier.failure())
        self.output.chmod(0o700)
        target = self.output / "aggregate-verification.json"
        sentinel = self.file(b"synthetic sentinel", "sentinel")
        target.symlink_to(sentinel)
        with self.assertRaises(FileExistsError):
            verifier.write_verification(verifier.failure())
        self.assertEqual(sentinel.read_bytes(), b"synthetic sentinel")


class MainAndWorkflowTests(unittest.TestCase):
    def invoke(self, argument):
        output = io.BytesIO()
        fake = local(sys, argv=["verify", argument], stdout=types.SimpleNamespace(buffer=output))
        with patch.object(verifier, "sys", fake):
            result = verifier.main()
        return result, output.getvalue()

    def test_source_check_without_receipt_reads_or_output_writes(self):
        sources = (None, None, None, None, None, 78, None)
        with (
            patch.object(verifier, "load_sources", return_value=(None, sources)),
            patch.object(
                verifier, "verify_prefix", side_effect=AssertionError("receipt forbidden")
            ),
            patch.object(verifier, "read_closed", side_effect=AssertionError("receipt forbidden")),
            patch.object(verifier, "os", local(os, open=Mock(side_effect=AssertionError("write")))),
        ):
            result, _raw = self.invoke("--check-source")
        self.assertEqual(result, 0)

    def test_prefix_check_uses_live_capacity_and_never_writes(self):
        check = Mock(return_value=(prefix_receipt(), "a" * 64))
        with (
            patch.object(verifier, "verify_prefix", check),
            patch.object(verifier, "os", local(os, open=Mock(side_effect=AssertionError("write")))),
        ):
            result, _raw = self.invoke("--check-prefix")
        self.assertEqual(result, 0)
        check.assert_called_once_with(check_capacity=True)

    def test_check_failure_and_unknown_argument_never_disclose_exception(self):
        with patch.object(verifier, "verify_prefix", side_effect=RuntimeError("private text")):
            result, raw = self.invoke("--check-prefix")
        self.assertNotEqual(result, 0)
        self.assertNotIn(b"private text", raw)
        self.assertEqual(json.loads(raw)["status"], "failed")
        result, raw = self.invoke("--unknown")
        self.assertNotEqual(result, 0)
        self.assertEqual(json.loads(raw)["status"], "failed")

    def test_workflow_has_only_disabled_or_exact_bound_guard_and_ordered_steps(self):
        text = (ROOT / ".github/workflows/sft-prepared-qualification-v3.yml").read_text()
        job = text.split("  private-prepared-importer-v3:\n", 1)[1].split("    runs-on:", 1)[0]
        guards = [line for line in job.splitlines() if line.startswith("    if:")]
        approved = (
            "    if: github.event.repository.private == true && "
            "github.repository_id == '1300831104' && github.run_attempt == 1 && "
            "github.event.before == '" + adapter.APPROVED_PARENT + "' && "
            "github.event.after == github.sha"
        )
        self.assertIn(guards, (["    if: false"], [approved]))
        self.assertIn(adapter.APPROVED_PARENT, text)
        self.assertNotIn("workflow_dispatch", text)
        self.assertIn("persist-credentials: false", text)
        self.assertIn("--check-source", text)
        self.assertLess(
            text.index("--fresh-sdk-prefix-capacity-check"), text.index("--check-prefix")
        )
        self.assertLess(text.index("--check-prefix"), text.index("--acquire-accepted-software"))
        self.assertEqual(text.count("--acquire-accepted-software"), 1)
        self.assertEqual(text.count("--synthetic-only"), 1)
        self.assertIn("Retire exactly the newly acquired input", text)
        self.assertIn("--verify-receipts", text)


if __name__ == "__main__":
    unittest.main()
