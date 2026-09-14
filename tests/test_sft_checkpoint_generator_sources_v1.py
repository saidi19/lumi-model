"""Generated tiny tar trees and inert source bytes; no native runtime/model data."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import stat
import sys
import tarfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_generator_source_tree_helpers_v1",
    ROOT / "tests/test_sft_checkpoint_guard_sources_v1.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
before = set(sys.modules)
subject = importlib.import_module(
    fixtures.PACKAGE + ".posttraining_checkpoint_generator_sources_v1")
ADDED = set(sys.modules) - before


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class GeneratorSourcesTests(unittest.TestCase):
    extract = fixtures.InstallationTests.extract
    cleanup_owned = fixtures.InstallationTests.cleanup_owned

    def setUp(self):
        fixtures.InstallationTests.setUp(self)
        self.files = dict.fromkeys(subject.REQUIRED_SOURCES, b"# generated inert source\n")
        self.manifest = subject.sources.build_manifest(self.files)
        self.helper = (ROOT / "scripts/run_checkpoint_guarded_import_v1.py").read_bytes()

    def install(self, owned, **changes):
        args = {"files": self.files, "manifest": self.manifest, "entry": self.code,
                "helper": self.helper, "expected_manifest_sha256": digest(self.manifest),
                "expected_entry_sha256": digest(self.code)}
        args.update(changes)
        return subject.install_generator_addon(owned, **args)

    def test_stdlib_only(self):
        self.assertFalse({name.split(".", 1)[0] for name in ADDED}
                         & {"lumi", "torch", "numpy", "transformers", "safetensors"})

    def test_installs_exact_fixed_paths_bytes_and_freezes(self):
        owned = self.extract()
        result = self.install(owned)
        additions = {subject.ENTRY_PATH: self.code, subject.HELPER_PATH: self.helper,
                     subject.sources.MANIFEST_PATH: self.manifest,
                     **{subject.sources.SOURCE_PREFIX + key: value
                        for key, value in self.files.items()}}
        self.assertEqual(result["addon_files"], len(additions))
        self.assertEqual(result["addon_bytes"], sum(map(len, additions.values())))
        self.assertEqual(owned.summary()["installed_files"], len(additions))
        self.assertTrue(result["installation_frozen"])
        for name, raw in additions.items():
            path = owned.path / name
            self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o444)
        self.assertEqual((owned.path / "usr/fixture.txt").read_bytes(), b"fixture")
        self.assertNotIn(str(owned.path), json.dumps(result))
        for key in ("base_origin_verified", "dependency_closure_verified", "filesystem_immutable",
                    "native_generator_verified", "real_import_authorized",
                    "training_authorized", "production_eligible"):
            self.assertIs(result[key], False)

    def test_helper_pin_checked_before_any_tree_operation(self):
        owned = self.extract()
        for raw in (b"", self.helper + b"\n", b"x" * len(self.helper), self.helper.decode()):
            with patch.object(owned, "_guard") as guard:
                with self.assertRaises(subject.GeneratorSourceError):
                    self.install(owned, helper=raw)
                guard.assert_not_called()

    def test_all_generator_required_sources_checked(self):
        owned = self.extract()
        for name in self.files:
            files = dict(self.files)
            del files[name]
            if not subject.sources.REQUIRED <= set(files):
                manifest = self.manifest
            else:
                manifest = subject.sources.build_manifest(files)
            with patch.object(owned, "_guard") as guard:
                with self.assertRaises(subject.GeneratorSourceError):
                    self.install(owned, files=files, manifest=manifest,
                                 expected_manifest_sha256=digest(manifest))
                guard.assert_not_called()

    def test_wrong_entry_pin_rejected_before_write(self):
        owned = self.extract()
        with patch.object(owned, "_write") as write:
            with self.assertRaises(subject.GeneratorSourceError):
                self.install(owned, expected_entry_sha256="1" * 64)
            write.assert_not_called()

    def test_frozen_root_and_second_install_rejected(self):
        owned = self.extract()
        self.install(owned)
        before = owned.summary()
        with self.assertRaises(subject.GeneratorSourceError):
            self.install(owned)
        self.assertEqual(before, owned.summary())

    def test_previously_installed_root_rejected(self):
        owned = self.extract()
        owned._installed_files = 1
        with patch.object(owned, "_write") as write:
            with self.assertRaises(subject.GeneratorSourceError):
                self.install(owned)
            write.assert_not_called()

    def test_existing_helper_or_entry_rejected(self):
        for name in (subject.HELPER_PATH, subject.ENTRY_PATH, "opt/lumi-release"):
            owned = self.extract(name)
            with patch.object(owned, "_write") as write:
                with self.assertRaises(subject.GeneratorSourceError):
                    self.install(owned)
                write.assert_not_called()

    def test_existing_importer_tree_rejected(self):
        owned = self.extract("opt/lumi-release/importer", kind=tarfile.DIRTYPE)
        with self.assertRaises(subject.GeneratorSourceError):
            self.install(owned)
        self.assertEqual(owned.summary()["installed_files"], 0)

    def test_capacity_checked_before_first_write(self):
        for limits in ({"maximum_entries": 5}, {"maximum_file_bytes": 1024},
                       {"maximum_file_bytes": 14500, "maximum_total_bytes": 15000}):
            owned = self.extract(limits=limits)
            with patch.object(owned, "_write") as write:
                with self.assertRaises(subject.GeneratorSourceError):
                    self.install(owned)
                write.assert_not_called()
            self.assertFalse(owned.summary()["operation_tainted"])

    def test_started_failure_taints_and_redacts_no_retry(self):
        owned = self.extract()
        with patch.object(owned, "_write", side_effect=OSError("private detail")) as write:
            with self.assertRaises(subject.GeneratorSourceError) as caught:
                self.install(owned)
            self.assertEqual(write.call_count, 1)
        self.assertEqual(caught.exception.args, ("installation_rejected",))
        self.assertIsNone(caught.exception.__context__)
        self.assertTrue(owned.summary()["operation_tainted"])
        with self.assertRaises(subject.GeneratorSourceError):
            self.install(owned)

    def test_failed_freeze_taints(self):
        owned = self.extract()
        with patch.object(owned, "freeze", side_effect=OSError("private detail")):
            with self.assertRaises(subject.GeneratorSourceError):
                self.install(owned)
        self.assertTrue(owned.summary()["operation_tainted"])

    def test_exact_owned_type_and_original_allowlist(self):
        with self.assertRaises(subject.GeneratorSourceError):
            self.install(object())
        owned = self.extract()
        with self.assertRaises(subject.rootfs.RootfsError):
            owned.install_file(subject.ENTRY_PATH, self.code, expected_sha256=digest(self.code))

    def test_runtime_rejection_before_io(self):
        with patch.object(subject.sources, "_runtime", side_effect=RuntimeError("private")), \
                patch.object(subject, "validate_bundle") as validate:
            with self.assertRaises(subject.GeneratorSourceError) as caught:
                self.install(object())
            self.assertEqual(caught.exception.code, "runtime_rejected")
            validate.assert_not_called()
