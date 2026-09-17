"""Synthetic setup diagnostics only: never run native setup, models or downloads."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/probe_sft_prepared_setup_v1.py"
SPEC = importlib.util.spec_from_file_location("_prepared_setup_preflight_tests_v1", SCRIPT)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
PRIVATE = "synthetic-private-detail-/never/read/model"


class RunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lumi-setup-preflight-tests-")
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name).resolve() / "owned"
        self.events = []
        self.rootfs = types.SimpleNamespace()
        self.base = types.SimpleNamespace(subreaper=Mock())
        self.admission = types.SimpleNamespace(
            IMAGE_ID="sha256:" + "a" * 64,
            SoftwareBinding=Mock(return_value=types.SimpleNamespace(
                image_id="sha256:" + "a" * 64)))
        self.helper = types.SimpleNamespace(
            load_base=Mock(return_value=self.base),
            load_admission=Mock(return_value=self.admission))
        self.pipeline = types.SimpleNamespace(
            addon=types.SimpleNamespace(rootfs=self.rootfs),
            layout=types.SimpleNamespace(rootfs=self.rootfs))
        self.harness = types.SimpleNamespace(
            MINIMUM_FREE_DISK_BYTES=m.MINIMUM_FREE_DISK_BYTES, ARCHIVE_BYTES=m.TAR_BYTES,
            runtime=Mock(), load_sources=Mock(return_value=(
                self.helper, self.pipeline, object(),
                {"generator": object(), "importer": object()}, 78)))
        self.enterContext(patch.object(m, "_ATTEMPTED", False))
        self.runtime = self.enterContext(patch.object(m, "runtime"))
        self.arm = self.enterContext(patch.object(m, "arm_deadline"))
        self.disarm = self.enterContext(patch.object(m, "disarm_deadline"))
        self.guard = self.enterContext(patch.object(m, "install_guard"))
        self.loader = self.enterContext(patch.object(m, "load_harness", return_value=self.harness))
        self.mkdir = self.enterContext(patch.object(
            m.tempfile, "mkdtemp", side_effect=self.make_parent))
        self.disk = self.enterContext(patch.object(m.os, "statvfs", return_value=(
            types.SimpleNamespace(f_bavail=32 * 1024**3, f_frsize=1))))

    def make_parent(self, **_kwargs):
        self.parent.mkdir(mode=0o700)
        return str(self.parent)

    def report(self):
        value = m.run()
        m.validate_report(value)
        raw = json.dumps(value, sort_keys=True)
        self.assertNotIn(PRIVATE, raw)
        self.assertNotIn(str(self.parent), raw)
        self.assertLessEqual(len(raw.encode()), 8192)
        return value

    def failed(self, stage):
        value = self.report()
        self.assertEqual(value["status"], "failed")
        self.assertEqual(value["stage"], stage)
        self.assertFalse(self.parent.exists())
        return value

    def test_success_calls_setup_once_and_removes_owned_empty_directories(self):
        value = self.report()
        self.assertEqual((value["status"], value["stage"], value["code"]),
                         ("observed", "complete", "ok"))
        self.runtime.assert_called_once()
        self.loader.assert_called_once()
        self.harness.runtime.assert_called_once()
        self.harness.load_sources.assert_called_once()
        self.helper.load_base.assert_called_once()
        self.base.subreaper.assert_called_once()
        self.helper.load_admission.assert_called_once()
        self.mkdir.assert_called_once()
        self.disk.assert_called_once()
        self.assertFalse(self.parent.exists())

    def test_runtime_failure_stops_before_source_reads(self):
        self.runtime.side_effect = RuntimeError(PRIVATE)
        self.failed("runtime")
        self.loader.assert_not_called()
        self.mkdir.assert_not_called()

    def test_harness_binding_failure(self):
        self.loader.side_effect = OSError(PRIVATE)
        self.failed("harness_binding")
        self.harness.load_sources.assert_not_called()

    def test_frozen_harness_runtime_failure(self):
        self.harness.runtime.side_effect = RuntimeError(PRIVATE)
        self.failed("harness_binding")
        self.harness.load_sources.assert_not_called()

    def test_frozen_floor_or_archive_binding_cannot_drift(self):
        self.harness.MINIMUM_FREE_DISK_BYTES -= 1
        value = self.failed("harness_binding")
        self.assertEqual(value["code"], "binding_rejected")
        self.harness.load_sources.assert_not_called()

    def test_source_closure_failure(self):
        self.harness.load_sources.side_effect = OSError(PRIVATE)
        self.failed("source_closure")
        self.helper.load_base.assert_not_called()

    def test_base_module_failure(self):
        self.helper.load_base.side_effect = OSError(PRIVATE)
        self.failed("base_module")
        self.base.subreaper.assert_not_called()

    def test_source_count_mismatch_rejects_before_base_loader(self):
        self.harness.load_sources.return_value = (
            self.helper, self.pipeline, object(), {}, 77)
        value = self.failed("source_closure")
        self.assertEqual(value["code"], "binding_rejected")
        self.helper.load_base.assert_not_called()

    def test_subreaper_failure(self):
        self.base.subreaper.side_effect = RuntimeError(PRIVATE)
        self.failed("subreaper")
        self.helper.load_admission.assert_not_called()

    def test_admission_module_failure(self):
        self.helper.load_admission.side_effect = OSError(PRIVATE)
        self.failed("admission_module")
        self.mkdir.assert_not_called()

    def test_admission_image_binding_mismatch(self):
        self.admission.SoftwareBinding.return_value.image_id = "sha256:" + "b" * 64
        value = self.failed("admission_module")
        self.assertEqual(value["code"], "binding_rejected")

    def test_rootfs_binding_failure(self):
        self.pipeline.addon = types.SimpleNamespace()
        self.failed("rootfs_binding")
        self.mkdir.assert_not_called()

    def test_rootfs_distinct_objects_rejected(self):
        self.pipeline.layout.rootfs = object()
        value = self.failed("rootfs_binding")
        self.assertEqual(value["code"], "binding_rejected")

    def test_parent_creation_failure(self):
        self.mkdir.side_effect = OSError(PRIVATE)
        self.failed("temp_parent")
        self.disk.assert_not_called()

    def test_parent_created_then_error_never_claims_cleanup_or_deletes_unbound_node(self):
        def create_then_fail(**_kwargs):
            self.parent.mkdir(mode=0o700)
            raise OSError(PRIVATE)

        self.mkdir.side_effect = create_then_fail
        value = self.report()
        self.assertEqual((value["stage"], value["code"]),
                         ("temp_parent", "cleanup_unconfirmed"))
        self.assertTrue(value["directory_creation_ambiguous"])
        self.assertFalse(value["private_owned_cleanup_verified"])
        self.assertEqual(value["owned_temp_directories_created"], 0)
        self.assertTrue(self.parent.exists())
        self.disk.assert_not_called()

    def test_config_creation_failure_retires_parent(self):
        original = Path.mkdir

        def mkdir(path, *args, **kwargs):
            if path.parent == self.parent:
                raise OSError(PRIVATE)
            return original(path, *args, **kwargs)

        with patch.object(Path, "mkdir", mkdir):
            self.failed("temp_config")
        self.disk.assert_not_called()

    def test_config_created_then_error_preserves_unbound_node_and_parent(self):
        original = Path.mkdir

        def create_then_fail(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path.parent == self.parent:
                raise OSError(PRIVATE)
            return result

        with patch.object(Path, "mkdir", create_then_fail):
            value = self.report()
        self.assertEqual((value["stage"], value["code"]),
                         ("temp_config", "cleanup_unconfirmed"))
        self.assertTrue(value["directory_creation_ambiguous"])
        self.assertFalse(value["private_owned_cleanup_verified"])
        self.assertEqual(value["owned_temp_directories_created"], 1)
        self.assertEqual(value["owned_temp_directories_removed"], 0)
        self.assertTrue((self.parent / "docker-config").is_dir())
        self.disk.assert_not_called()

    def test_disk_query_failure_retires_both_directories(self):
        self.disk.side_effect = OSError(PRIVATE)
        self.failed("disk_query")

    def test_deadline_has_fixed_code_and_retires_owned_directories(self):
        self.disk.side_effect = m.PreflightError("deadline_exhausted")
        value = self.failed("disk_query")
        self.assertEqual(value["code"], "deadline_exhausted")
        self.disarm.assert_called_once()

    def test_cleanup_failure_preserves_setup_stage_and_unowned_content(self):
        def disk(_path):
            (self.parent / "docker-config" / "unowned-sentinel").touch()
            raise OSError(PRIVATE)

        self.disk.side_effect = disk
        value = self.report()
        self.assertEqual((value["status"], value["stage"], value["code"]),
                         ("failed", "disk_query", "cleanup_unconfirmed"))
        self.assertEqual(value["owned_temp_directories_created"], 2)
        self.assertEqual(value["owned_temp_directories_removed"], 0)
        self.assertFalse(value["private_owned_cleanup_verified"])
        self.assertTrue((self.parent / "docker-config" / "unowned-sentinel").exists())

    def test_cleanup_failure_after_observation_cannot_be_ready(self):
        with patch.object(m, "cleanup", return_value=(0, False)):
            value = self.report()
        self.assertEqual((value["status"], value["stage"], value["code"]),
                         ("failed", "complete", "cleanup_unconfirmed"))
        self.assertFalse(value["ready_for_unchanged_setup"])

    def test_one_byte_below_old_floor_is_observation_not_setup_failure(self):
        self.disk.return_value = types.SimpleNamespace(
            f_bavail=m.MINIMUM_FREE_DISK_BYTES - 1, f_frsize=1)
        value = self.report()
        self.assertEqual((value["status"], value["code"]),
                         ("observed", "disk_floor_insufficient"))
        self.assertFalse(self.parent.exists())

    def test_exact_old_floor_lacks_acquisition_headroom(self):
        self.disk.return_value = types.SimpleNamespace(
            f_bavail=m.MINIMUM_FREE_DISK_BYTES, f_frsize=1)
        value = self.report()
        self.assertEqual((value["status"], value["code"]),
                         ("observed", "acquisition_headroom_insufficient"))

    def test_exact_unrounded_acquisition_headroom_is_sufficient(self):
        self.disk.return_value = types.SimpleNamespace(
            f_bavail=m.MINIMUM_FREE_DISK_BYTES + m.TAR_BYTES, f_frsize=1)
        value = self.report()
        self.assertEqual((value["status"], value["code"]), ("observed", "ok"))

    def test_second_run_does_not_retry_sources_or_native_setup(self):
        self.report()
        value = self.report()
        self.assertEqual((value["status"], value["code"]),
                         ("failed", "already_attempted"))
        self.loader.assert_called_once()
        self.base.subreaper.assert_called_once()
        self.mkdir.assert_called_once()

    def test_failure_after_start_also_consumes_attempt(self):
        self.harness.load_sources.side_effect = OSError(PRIVATE)
        self.failed("source_closure")
        value = self.report()
        self.assertEqual(value["code"], "already_attempted")
        self.harness.load_sources.assert_called_once()


class CapacityTests(unittest.TestCase):
    def test_bound_sizes_and_unchanged_floor(self):
        self.assertEqual((m.TAR_BYTES, m.ZIP_BYTES), (1_062_113_280, 1_062_113_436))
        self.assertEqual(m.MINIMUM_FREE_DISK_BYTES, 16 * 1024**3)
        value = m.capacity(types.SimpleNamespace(f_bavail=0, f_frsize=1))
        self.assertEqual(value["acquisition_peak_allocation_bytes"], 2_124_226_716)
        self.assertEqual(value["minimum_preacquisition_free_disk_bytes"], 18_241_982_464)

    def test_allocation_rounds_both_inputs_separately(self):
        fragment = 4096
        value = m.capacity(types.SimpleNamespace(f_bavail=8_000_000, f_frsize=fragment))
        tar = (m.TAR_BYTES + fragment - 1) // fragment * fragment
        zipped = (m.ZIP_BYTES + fragment - 1) // fragment * fragment
        self.assertEqual(value["available_disk_bytes"], 8_000_000 * fragment)
        self.assertEqual(value["retained_tar_allocation_bytes"], tar)
        self.assertEqual(value["acquisition_peak_allocation_bytes"], tar + zipped)
        self.assertEqual(value["minimum_preacquisition_free_disk_bytes"],
                         m.MINIMUM_FREE_DISK_BYTES + tar)

    def test_headroom_one_fragment_below_and_exact(self):
        fragment = 4096
        initial = m.capacity(types.SimpleNamespace(f_bavail=0, f_frsize=fragment))
        blocks = initial["minimum_preacquisition_free_disk_bytes"] // fragment
        before = m.capacity(types.SimpleNamespace(f_bavail=blocks - 1, f_frsize=fragment))
        exact = m.capacity(types.SimpleNamespace(f_bavail=blocks, f_frsize=fragment))
        self.assertTrue(before["disk_floor_met"])
        self.assertFalse(before["acquisition_headroom_met"])
        self.assertTrue(exact["acquisition_headroom_met"])

    def test_invalid_statvfs_values_rejected(self):
        for blocks, fragment in ((-1, 1), (True, 1), (1, True), (0, 0), (1, -1),
                                 (1.0, 1), (1, 1.0), (2**63, 1), (2**62, 2),
                                 (1, 1024**2 + 1)):
            with self.subTest(blocks=blocks, fragment=fragment), \
                    self.assertRaises(m.PreflightError):
                m.capacity(types.SimpleNamespace(f_bavail=blocks, f_frsize=fragment))


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lumi-setup-cleanup-tests-")
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name).resolve() / "owned"
        self.parent.mkdir(mode=0o700)

    def own(self, path):
        return (path, m.identity(path.lstat()))

    def test_empty_owned_directories_removed_in_child_first_order(self):
        child = self.parent / "config"
        child.mkdir(mode=0o700)
        self.assertEqual(m.cleanup([self.own(self.parent), self.own(child)]), (2, True))
        self.assertFalse(self.parent.exists())

    def test_no_ownership_identity_means_no_removal(self):
        self.assertEqual(m.cleanup([(self.parent, None)]), (0, False))
        self.assertTrue(self.parent.exists())

    def test_mode_change_means_no_removal(self):
        entry = self.own(self.parent)
        self.parent.chmod(0o755)
        self.assertEqual(m.cleanup([entry]), (0, False))
        self.assertTrue(self.parent.exists())

    def test_wrong_inode_means_no_removal(self):
        path, bound = self.own(self.parent)
        replaced = (bound[0], bound[1] + 1, *bound[2:])
        self.assertEqual(m.cleanup([(path, replaced)]), (0, False))
        self.assertTrue(self.parent.exists())

    def test_symlink_is_never_followed_or_removed(self):
        link = Path(self.temp.name).resolve() / "link"
        link.symlink_to(self.parent, target_is_directory=True)
        self.assertEqual(m.cleanup([self.own(link)]), (0, False))
        self.assertTrue(link.is_symlink())
        self.assertTrue(self.parent.exists())

    def test_nonempty_directory_is_never_recursively_removed(self):
        sentinel = self.parent / "sentinel"
        sentinel.touch()
        self.assertEqual(m.cleanup([self.own(self.parent)]), (0, False))
        self.assertTrue(sentinel.exists())


class ReportTests(unittest.TestCase):
    def observed(self):
        value = m.empty_report()
        value.update(m.capacity(types.SimpleNamespace(f_bavail=32 * 1024**3, f_frsize=1)))
        value.update(status="observed", stage="complete", code="ok", source_bound_files=78,
                     owned_temp_directories_created=2, owned_temp_directories_removed=2,
                     setup_subreaper_installed=True, ready_for_unchanged_setup=True)
        return value

    def test_valid_failure_and_observed_reports_are_closed_and_small(self):
        for value in (m.empty_report(), self.observed()):
            with self.subTest(status=value["status"]):
                self.assertIs(m.validate_report(value), value)
                self.assertLessEqual(len(m.canonical(value)), 8192)

    def test_unknown_fields_and_private_strings_rejected(self):
        for key in ("exception", "path", "source", "candidate", "id"):
            value = self.observed()
            value[key] = PRIVATE
            with self.subTest(key=key), self.assertRaises(m.PreflightError):
                m.validate_report(value)

    def test_missing_keys_and_unknown_fixed_codes_rejected(self):
        values = []
        value = self.observed()
        del value["schema"]
        values.append(value)
        for field in ("status", "stage", "code", "schema"):
            value = self.observed()
            value[field] = PRIVATE
            values.append(value)
        for value in values:
            with self.subTest(value=value), self.assertRaises(m.PreflightError):
                m.validate_report(value)

    def test_every_nonclaim_must_remain_exact_false(self):
        for name in m.NONCLAIMS:
            for invalid in (True, 0, None):
                value = self.observed()
                value[name] = invalid
                with self.subTest(name=name, invalid=invalid), \
                        self.assertRaises(m.PreflightError):
                    m.validate_report(value)

    def test_observed_report_requires_all_completed_setup_evidence(self):
        for name, invalid in (("source_bound_files", 0), ("setup_subreaper_installed", False),
                              ("stage", "disk_query"), ("owned_temp_directories_created", 1),
                              ("owned_temp_directories_removed", 1),
                              ("ready_for_unchanged_setup", False)):
            value = self.observed()
            value[name] = invalid
            with self.subTest(name=name), self.assertRaises(m.PreflightError):
                m.validate_report(value)

    def test_aggregate_counts_are_exact_int_not_bool(self):
        for name in (*m.DISK_INTS, "source_bound_files", "owned_temp_directories_created",
                     "owned_temp_directories_removed", "maximum_seconds"):
            value = self.observed()
            value[name] = True
            with self.subTest(name=name), self.assertRaises(m.PreflightError):
                m.validate_report(value)

    def test_capacity_cannot_be_partially_present_or_inconsistent(self):
        for name in m.DISK_INTS:
            for invalid in (None, -1, 2**63):
                value = self.observed()
                value[name] = invalid
                with self.subTest(name=name, invalid=invalid), \
                        self.assertRaises(m.PreflightError):
                    m.validate_report(value)
        value = self.observed()
        value["acquisition_headroom_met"] = False
        with self.assertRaises(m.PreflightError):
            m.validate_report(value)


class EntrypointTests(unittest.TestCase):
    def test_mac_runtime_rejects_before_any_source_read(self):
        with patch.object(m, "sys", types.SimpleNamespace(platform="darwin")), \
                patch.object(m, "load_harness") as loader, \
                patch.object(m.os, "open") as opening, \
                self.assertRaises(m.PreflightError) as caught:
            m.runtime()
        self.assertEqual(caught.exception.code, "runtime_rejected")
        opening.assert_not_called()
        loader.assert_not_called()

    def test_main_accepts_only_exact_argument_and_fixed_json(self):
        report = ReportTests().observed()
        with patch.object(m, "sys", types.SimpleNamespace(argv=["entry", "--probe-setup"])), \
                patch.object(m, "run", return_value=report) as run, \
                patch.object(m.os, "write", side_effect=lambda _fd, raw: len(raw)) as write:
            self.assertEqual(m.main(), 0)
        run.assert_called_once_with()
        self.assertEqual(write.call_args.args, (1, m.canonical(report)))

    def test_invalid_arguments_never_start_probe_or_echo_arguments(self):
        for args in ([], ["--probe-setup", PRIVATE], ["--synthetic-only"], [PRIVATE]):
            with self.subTest(args=args), \
                    patch.object(m, "sys", types.SimpleNamespace(argv=["entry", *args])), \
                    patch.object(m, "run") as run, \
                    patch.object(m.os, "write", side_effect=lambda _fd, raw: len(raw)) as write:
                self.assertEqual(m.main(), 2)
            run.assert_not_called()
            value = json.loads(write.call_args.args[1])
            self.assertEqual(value["code"], "runtime_rejected")
            self.assertNotIn(PRIVATE.encode(), write.call_args.args[1])

    def test_partial_stdout_write_rejected(self):
        with patch.object(m, "sys", types.SimpleNamespace(argv=["entry"])), \
                patch.object(m.os, "write", return_value=1), self.assertRaises(m.PreflightError):
            m.main()

    def test_expired_and_untrusted_error_codes_never_retain_detail(self):
        with self.assertRaises(m.PreflightError) as caught:
            m.expired(None, None)
        self.assertEqual(caught.exception.code, "deadline_exhausted")
        self.assertEqual(str(m.PreflightError(PRIVATE)), "step_failed")

    def test_audit_callback_rejects_network_process_and_other_dynamic_libraries(self):
        for event, args in (("socket.__new__", ()), ("socket.connect", ()),
                            ("subprocess.Popen", ()), ("os.fork", ()), ("os.forkpty", ()),
                            ("os.exec", ()), ("os.posix_spawn", ()), ("os.spawn", ()),
                            ("os.system", ()), ("ctypes.dlopen", ("/private/library",))):
            with self.subTest(event=event), self.assertRaises(m.PreflightError):
                m.audit_guard(event, args)

    def test_audit_callback_allows_only_needed_inert_and_process_local_operations(self):
        for event, args in (("ctypes.dlopen", (None,)), ("open", ()), ("compile", ()),
                            ("exec", ()), ("os.mkdir", ()), ("os.rmdir", ())):
            with self.subTest(event=event):
                self.assertIsNone(m.audit_guard(event, args))


class FrozenContractTests(unittest.TestCase):
    def test_frozen_harness_digest_and_native_floor_still_bound(self):
        raw = (ROOT / m.HARNESS).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), m.HARNESS_SHA256)
        self.assertEqual(m.HARNESS_SHA256,
                         "253bb5d0b113949baecb5ea229590c1f1e2091d9f2e659abf6fbbbe9c3e1f107")
        self.assertIn(b"MINIMUM_FREE_DISK_BYTES = 16 * 1024**3", raw)

    def test_workflow_only_fresh_private_branch_attempt_and_own_path(self):
        raw = (ROOT / ".github/workflows/sft-prepared-setup-preflight-v1.yml").read_text()
        for required in (
                "- codex/lumi-posttraining-product-v0.1",
                "- .github/workflows/sft-prepared-setup-preflight-v1.yml",
                "github.event.repository.private == true",
                "github.repository_id == '1300831104'", "github.run_attempt == 1",
                "github.event.before == '9ae708c73a327b7cac69c16d29d075450a500e9d'",
                "github.event.after == github.sha", "persist-credentials: false",
                "runs-on: ubuntu-24.04", "timeout-minutes: 8",
                "timeout --signal=TERM --kill-after=5s 90s",
                "contents: read", "set -euC"):
            self.assertIn(required, raw)
        self.assertNotIn("workflow_dispatch", raw)
        self.assertNotIn("pull_request:", raw)

    def test_workflow_script_digest_and_no_acquisition_or_pipeline_execution(self):
        raw = (ROOT / ".github/workflows/sft-prepared-setup-preflight-v1.yml").read_text()
        self.assertIn(hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
                      + "  scripts/probe_sft_prepared_setup_v1.py", raw)
        for forbidden in ("pip install", "apt-get", "curl ", "wget ", "docker ",
                          "--acquire-accepted-software", "run_checkpoint_prepared_qualification",
                          "supervise_checkpoint_prepared_qualification", "workflow run"):
            self.assertNotIn(forbidden, raw)
        self.assertIn("scripts/probe_sft_prepared_setup_v1.py --probe-setup", raw)
        self.assertIn("-p test_sft_prepared_setup_preflight_v1.py", raw)


if __name__ == "__main__":
    unittest.main()
