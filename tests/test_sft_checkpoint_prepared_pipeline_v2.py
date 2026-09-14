"""Real codec/controls/layout/launch composition with synthetic syscall boundaries."""

from __future__ import annotations

import copy
import importlib
import importlib.util
import stat
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_pipeline_layout_test_helpers_v1",
    ROOT / "tests/test_sft_checkpoint_prepared_launch_v2.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
fixture = helpers.fixture
m = importlib.import_module(fixture.PACKAGE + ".posttraining_checkpoint_prepared_pipeline_v2")
PRIVATE = b"generated-data"


class MemoryPath:
    def __init__(self, name):
        self.name = name

    def __truediv__(self, child):
        return MemoryPath(self.name + "/" + child)

    def mkdir(self, **_kwargs):
        pass


def generated_metadata():
    stored = {"checkpoint/model/.metadata": b"generated-meta",
              "checkpoint/model/__0_0.distcp": PRIVATE}

    def write(path, raw):
        if path.name in stored:
            raise AssertionError("fixture_rewrite")
        stored[path.name] = raw

    def inventory(path, _callback):
        return [{"path": key[len(path.name) + 1:], "bytes": len(raw), "sha256": m.sha(raw)}
                for key, raw in sorted(stored.items()) if key.startswith(path.name + "/")]

    with patch.object(m.codec.fixture, "_write", side_effect=write), \
            patch.object(m.codec.fixture, "_inventory", side_effect=inventory):
        metadata = m.codec.fixture._metadata(MemoryPath("checkpoint"), MemoryPath("tokenizer"),
                                              lambda: None)
    inputs = {role: {key[len(role) + 1:]: raw for key, raw in stored.items()
                    if key.startswith(role + "/")} for role in ("checkpoint", "tokenizer")}
    return metadata, inputs


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.harness = fixture.LayoutTests(
            "test_exact_addons_private_modes_frozen_tree_and_mount_controls")
        self.harness.setUp()
        self.addCleanup(self.harness.doCleanups)
        self.imp = self.harness.owned
        self.gen = object.__new__(m.addon.rootfs.OwnedRootfs)
        self.gen._path = Path("/tmp/lumi-generated-generator/root")
        self.gen._parent = self.gen._path.parent
        self.gen._parent_fd = 777
        self.gen._guard, self.gen._verify_tree = Mock(), Mock()
        self.gen._frozen = False
        self.gen._installed_files = self.gen._installed_bytes = 0
        self.gen._nodes = {(): ((7, 777, stat.S_IFDIR, 0, 0), "directory", 0o755)}
        self.parent_infos = {
            fd: types.SimpleNamespace(st_dev=7, st_ino=index, st_mode=stat.S_IFDIR | 0o700,
                                      st_uid=0, st_gid=0)
            for fd, index in ((self.gen._parent_fd, 111), (self.imp._parent_fd, 222))}
        self.enterContext(patch.object(m, "os", types.SimpleNamespace(
            fstat=Mock(side_effect=lambda fd: self.parent_infos[fd]))))
        self.bundle = {key: self.harness.bundle[key] for key in m.BUNDLE_KEYS}
        files = dict.fromkeys(m.addon.REQUIRED_SOURCES, b"# generated inert generator source\n")
        manifest, entry = m.addon.sources.build_manifest(files), b"# generated inert entry\n"
        self.generator_bundle = {"files": files, "manifest": manifest, "entry": entry,
            "helper": self.bundle["helper"], "expected_manifest_sha256": m.sha(manifest),
            "expected_entry_sha256": m.sha(entry)}
        metadata, inputs = generated_metadata()
        nonce = "9" * 64
        plan = m.codec._original_plan(metadata)
        generated = {"metadata": metadata, "generated_inputs": inputs,
            "receipt": m.codec._generation_receipt(metadata, inputs, plan, nonce)}
        size = sum(map(len, files.values()))
        source = {"source_manifest_sha256": m.sha(manifest), "source_files_verified": len(files),
            "source_bytes_verified": size, "source_verification_read_bytes": size + len(manifest)}
        copied = {"private_source_copy_files": len(files),
            "private_source_copy_bytes_reserved": size, "private_source_copy_readback_bytes": size,
            "private_source_copies_rootfs_immutable": False,
            "hostile_owner_code_immutability_verified": False}
        raw = m.codec.encode_generated_packet(generated, entry_sha256=m.sha(entry),
            source_manifest_sha256=m.sha(manifest), generation_nonce_sha256=nonce,
            source_report=source, copy_report=copied)
        decoded = m.codec.decode_generated_packet(raw, expected_entry_sha256=m.sha(entry),
            expected_source_manifest_sha256=m.sha(manifest), expected_generation_nonce_sha256=nonce)
        self.generated = {
            "private_data": {key: decoded[key] for key in ("metadata", "generated_inputs")},
            "receipt": {"schema": m.generator.SCHEMA, "status": "passed", "synthetic_only": True,
                "owned_cgroups_empty_and_removed": 1, "worker_exit_status_verified": True,
                "controller_and_deadline_watch_installed_before_release": True,
                "private_socket_buffers_verified": True, "packet_receipt": decoded["receipt"],
                "kill_attempts": 1, "workload_processes_reaped": 1,
                "worker_packet_maximum_bytes": 65536, "outer_packet_maximum_bytes": 131072,
                "minimum_socket_buffer_bytes": 262144, "memory_max_bytes": 4294967296,
                "swap_max_bytes": 0, "pids_max": 256, "cpu_quota_us": 200000,
                "cpu_period_us": 100000, "maximum_seconds": 90,
                "tmpfs_capacity_bytes": 536870912, "shm_capacity_bytes": 67108864,
                **dict.fromkeys(m.generator.NONCLAIMS, False)}}
        self.enterContext(patch.object(m, "_ATTEMPTED", False))
        self.enterContext(patch.object(m.launcher, "_ATTEMPTED", False))
        self.runtime = self.enterContext(patch.object(m.generator.child_bootstrap, "_runtime"))
        self.enterContext(patch.object(m.launcher.guard.child_bootstrap, "_runtime"))
        self.clock = self.enterContext(patch.object(m.time, "monotonic_ns", return_value=100))
        self.events = []
        self.addon = self.enterContext(patch.object(m.addon, "install_generator_addon",
            side_effect=lambda *_args, **_kwargs: self.events.append("generator_addon")))
        self.generate = self.enterContext(patch.object(m.generator, "run_generator_guard",
            side_effect=self.generator_result))
        original_install = m.layout.install_prepared_layout

        def install(owned, **arguments):
            self.events.append("import_layout")
            self.installed = original_install(owned, **arguments)
            return self.installed

        self.install = self.enterContext(patch.object(m.layout, "install_prepared_layout",
                                                       side_effect=install))
        self.import_guard = self.enterContext(patch.object(m.launcher.guard,
            "run_prepared_synthetic_guard", side_effect=self.import_result))
        self.args = {"generator_root": self.gen, "importer_root": self.imp,
            "generator_mountpoint_path": "/tmp/lumi-generated-generator/mountpoint",
            "generator_bundle": self.generator_bundle, "importer_bundle": self.bundle,
            "generation_nonce_sha256": nonce,
            "base_script_path": "/release/scripts/run_sft_cgroup_fixture_v1.py",
            "expected_base_sha256": m.generator.BASE_SHA256, "deadline_ns": 200_000_000_000}

    def generator_result(self, *_args, **_kwargs):
        self.events.append("generator_guard")
        return self.generated

    def import_result(self, **_arguments):
        self.events.append("import_guard")
        return helpers.guard_result(self.installed)

    def invoke(self, **changes):
        return m.run_prepared_pipeline(**(self.args | changes))

    def test_real_codec_controls_layout_launch_connect_in_order_once(self):
        result = self.invoke()
        self.assertEqual(self.events, ["generator_addon", "generator_guard", "import_layout",
                                       "import_guard"])
        for operation in (self.addon, self.generate, self.install, self.import_guard):
            operation.assert_called_once()
        self.assertEqual(result["status"], "two_guardian_passes_completed_cleanup_pending")
        self.assertEqual(result["owned_cgroups_empty_and_removed"], 2)
        for key in m.NONCLAIMS:
            self.assertIs(result[key], False)
        self.assertTrue(result["cleanup_required_on_success_and_failure"])

    def test_aggregate_result_never_contains_paths_payloads_or_private_packet(self):
        result = self.invoke()
        encoded = m.canonical(result)
        for raw in (PRIVATE, str(self.gen._path).encode(), fixture.PARENT.encode(),
                    b"base64", b"private_data", b"generated_inputs", b"metadata"):
            self.assertNotIn(raw, encoded)
        self.assertLess(len(encoded), 4096)

    def test_runtime_rejected_before_validation_or_io(self):
        self.runtime.side_effect = RuntimeError("private detail")
        with patch.object(m, "_bundle") as validate:
            with self.assertRaisesRegex(m.PreparedPipelineError, "^runtime_rejected$"):
                self.invoke()
            validate.assert_not_called()
        self.addon.assert_not_called()

    def test_consumed_success_is_not_retried_even_after_runtime_changes(self):
        self.invoke()
        self.runtime.side_effect = RuntimeError("private")
        with self.assertRaisesRegex(m.PreparedPipelineError, "^already_attempted$"):
            self.invoke()
        self.generate.assert_called_once()

    def test_bad_source_pins_or_extra_bundle_fields_before_attempt(self):
        for role in ("generator_bundle", "importer_bundle"):
            for changes in ({"expected_entry_sha256": "e" * 64}, {"approval": True},
                            {"helper": b"changed"}):
                with self.assertRaises(m.PreparedPipelineError):
                    self.invoke(**{role: self.args[role] | changes})
        self.assertFalse(m._ATTEMPTED)
        self.addon.assert_not_called()

    def test_same_root_and_same_identity_rejected(self):
        with self.assertRaisesRegex(m.PreparedPipelineError, "^roots_rejected$"):
            self.invoke(importer_root=self.gen)
        self.gen._nodes[()] = (self.imp._nodes[()][0], "directory", 0o755)
        with self.assertRaisesRegex(m.PreparedPipelineError, "^roots_rejected$"):
            self.invoke()
        self.addon.assert_not_called()

    def test_same_parent_or_nested_parent_rejected(self):
        for parent in (self.imp._parent, self.imp._parent / "nested", self.imp._parent.parent):
            self.gen._parent = parent
            self.gen._path = parent / "other-root"
            with self.assertRaisesRegex(m.PreparedPipelineError, "^roots_rejected$"):
                self.invoke(generator_mountpoint_path=str(parent / "mountpoint"))
        self.addon.assert_not_called()

    def test_bind_mount_parent_alias_identity_rejected(self):
        self.parent_infos[self.gen._parent_fd] = self.parent_infos[self.imp._parent_fd]
        with self.assertRaisesRegex(m.PreparedPipelineError, "^roots_rejected$"):
            self.invoke()
        self.addon.assert_not_called()
        self.assertFalse(m._ATTEMPTED)

    def test_generator_mountpoint_may_not_alias_any_importer_sibling(self):
        for path in (str(self.gen._path), *(str(self.imp._parent / name) for name in
                     m.layout.SIBLINGS), "/tmp/other/mountpoint"):
            with self.assertRaisesRegex(m.PreparedPipelineError, "^roots_rejected$"):
                self.invoke(generator_mountpoint_path=path)
        self.addon.assert_not_called()

    def test_frozen_or_previously_installed_root_before_attempt(self):
        for owned in (self.gen, self.imp):
            for field in ("_frozen", "_installed_files", "_installed_bytes"):
                with patch.object(owned, field, 1):
                    with self.assertRaisesRegex(m.PreparedPipelineError, "^roots_rejected$"):
                        self.invoke()
        self.addon.assert_not_called()

    def test_invalid_nonce_controller_or_shared_deadline_before_attempt(self):
        for changes in ({"generation_nonce_sha256": "0" * 64}, {"deadline_ns": True},
                        {"deadline_ns": 100}, {"deadline_ns": 241_000_000_000},
                        {"base_script_path": "/release/scripts/other.py"},
                        {"expected_base_sha256": "e" * 64}):
            with self.assertRaises(m.PreparedPipelineError):
                self.invoke(**changes)
        self.assertFalse(m._ATTEMPTED)
        self.addon.assert_not_called()

    def test_failure_after_started_addon_cannot_retry(self):
        self.addon.side_effect = OSError("private input path")
        with self.assertRaisesRegex(m.PreparedPipelineError, "^pipeline_failed$"):
            self.invoke()
        with self.assertRaisesRegex(m.PreparedPipelineError, "^already_attempted$"):
            self.invoke()
        self.generate.assert_not_called()

    def test_generator_failure_prevents_import_construction(self):
        self.generate.side_effect = m.generator.GuardCompositionError("cleanup_unconfirmed")
        with self.assertRaisesRegex(m.PreparedPipelineError, "^cleanup_unconfirmed$"):
            self.invoke()
        self.install.assert_not_called()
        self.assertTrue(m._ATTEMPTED)

    def test_mutated_generated_bytes_cannot_rebind_packet_receipt(self):
        selected = copy.deepcopy(self.generated)
        checkpoint = selected["private_data"]["generated_inputs"]["checkpoint"]
        checkpoint["model/__0_0.distcp"] = b"changed"
        self.generated = selected
        with self.assertRaisesRegex(m.PreparedPipelineError, "^generation_rejected$"):
            self.invoke()
        self.install.assert_not_called()

    def test_wrong_packet_nonce_or_source_accounting_rejected(self):
        for key in ("generation_nonce_sha256", "source_files_verified"):
            selected = copy.deepcopy(self.generated)
            packet = selected["receipt"]["packet_receipt"]
            if key == "generation_nonce_sha256":
                packet[key] = "e" * 64
            else:
                packet["source_report"][key] += 1
            with self.assertRaisesRegex(m.PreparedPipelineError, "^generation_rejected$"):
                m._generation(selected, self.generator_bundle, self.args["generation_nonce_sha256"])
        self.install.assert_not_called()

    def test_missing_guardian_proof_prevents_import(self):
        self.generated["receipt"]["worker_exit_status_verified"] = False
        with self.assertRaisesRegex(m.PreparedPipelineError, "^generation_rejected$"):
            self.invoke()
        self.install.assert_not_called()

    def test_all_guardian_lifecycle_resource_and_nonclaim_fields_required(self):
        for key in self.generated["receipt"]:
            selected = copy.deepcopy(self.generated)
            del selected["receipt"][key]
            with self.assertRaisesRegex(m.PreparedPipelineError, "^generation_rejected$"):
                m._generation(selected, self.generator_bundle, self.args["generation_nonce_sha256"])
        for key, value in (("kill_attempts", 0), ("workload_processes_reaped", True),
                           ("workload_processes_reaped", 0), ("training_authorized", True)):
            selected = copy.deepcopy(self.generated)
            selected["receipt"][key] = value
            with self.assertRaisesRegex(m.PreparedPipelineError, "^generation_rejected$"):
                m._generation(selected, self.generator_bundle, self.args["generation_nonce_sha256"])

    def test_layout_failure_after_generation_never_starts_import_guard(self):
        self.install.side_effect = OSError("private generated path")
        with self.assertRaisesRegex(m.PreparedPipelineError, "^pipeline_failed$"):
            self.invoke()
        self.import_guard.assert_not_called()
        self.generate.assert_called_once()
        self.assertTrue(m._ATTEMPTED)

    def test_import_guard_failure_no_retry_or_qualification(self):
        self.import_guard.side_effect = m.launcher.guard.GuardCompositionError(
            "cleanup_unconfirmed")
        with self.assertRaisesRegex(m.PreparedPipelineError, "^cleanup_unconfirmed$"):
            self.invoke()
        with self.assertRaisesRegex(m.PreparedPipelineError, "^already_attempted$"):
            self.invoke()
        self.generate.assert_called_once()
        self.import_guard.assert_called_once()

    def test_shared_deadline_exhaustion_after_generator_prevents_import(self):
        def generator_result(*args, **kwargs):
            result = self.generator_result(*args, **kwargs)
            self.clock.return_value = self.args["deadline_ns"]
            return result

        self.generate.side_effect = generator_result
        with self.assertRaisesRegex(m.PreparedPipelineError, "^deadline_exhausted$"):
            self.invoke()
        self.install.assert_not_called()
        self.generate.assert_called_once()

    def test_deadline_exhausted_in_preflight_prevents_first_write(self):
        self.gen._verify_tree.side_effect = lambda: setattr(
            self.clock, "return_value", self.args["deadline_ns"])
        with self.assertRaisesRegex(m.PreparedPipelineError, "^deadline_exhausted$"):
            self.invoke()
        self.addon.assert_not_called()
        self.assertFalse(m._ATTEMPTED)

    def test_deadline_exhausted_building_controls_prevents_layout_writes(self):
        original = m.controls.build_prepared_controls

        def build(**arguments):
            result = original(**arguments)
            self.clock.return_value = self.args["deadline_ns"]
            return result

        with patch.object(m.controls, "build_prepared_controls", side_effect=build):
            with self.assertRaisesRegex(m.PreparedPipelineError, "^deadline_exhausted$"):
                self.invoke()
        self.install.assert_not_called()
        self.generate.assert_called_once()

    def test_shared_deadline_exhaustion_after_import_cannot_be_success(self):
        def imported(**arguments):
            result = self.import_result(**arguments)
            self.clock.return_value = self.args["deadline_ns"]
            return result

        self.import_guard.side_effect = imported
        with self.assertRaisesRegex(m.PreparedPipelineError, "^deadline_exhausted$"):
            self.invoke()
        self.import_guard.assert_called_once()
