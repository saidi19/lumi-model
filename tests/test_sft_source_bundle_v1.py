"""Synthetic source bytes/files only; no dataset, weights or transport."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from lumi import posttraining_sft_source_bundle_v1 as bundle


class SourceBundleTests(unittest.TestCase):
    def fixture(self):
        return {"src/lumi/example.py": b"VALUE = 1\n"}, {
            "static_local_import_closure_included": True, "dynamic_call_sites": 0,
            "dynamic_dependency_completeness_verified": False,
            "runtime_dependency_completeness_verified": False,
        }

    def test_roundtrip_without_extracting_or_executing(self):
        files, inspection = self.fixture()
        raw, manifest = bundle.make_bundle(files, inspection=inspection)
        report = bundle.verify_bundle(raw, bundle.digest(raw), files)
        self.assertEqual(report["files_verified"], 1)
        self.assertFalse(report["bundled_code_executed"])
        self.assertFalse(manifest["deployment_ready"])

    def test_deterministic_independent_of_dictionary_order(self):
        files, inspection = self.fixture()
        files["scripts/example.py"] = b"# synthetic\n"
        first = bundle.make_bundle(files, inspection=inspection)
        second = bundle.make_bundle(dict(reversed(list(files.items()))), inspection=inspection)
        self.assertEqual(first, second)

    def test_private_nonexecutable_zip_modes(self):
        files, inspection = self.fixture()
        raw, _ = bundle.make_bundle(files, inspection=inspection)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            self.assertTrue(all((i.external_attr >> 16) & 0o777 == 0o600 for i in archive.infolist()))

    def test_rejects_path_traversal_and_private_data(self):
        for name in ("../secret.py", "/tmp/secret.py", "src/lumi/../secret.py", "src//lumi/x.py",
                     "data/test.jsonl", "artifacts/private.py", "evidence/report.json", ".env",
                     "configs/owner-approval.json", "src/lumi/x.py\n", "src\\lumi\\x.py"):
            with self.subTest(name=name):
                self.assertFalse(bundle.allowed_path(name))

    def test_rejects_unselected_control_path(self):
        with self.assertRaises(bundle.BundleError):
            bundle.collect(Path("/synthetic/repo"), ["scripts/example.py"], ["configs/secret.json"])

    def test_duplicate_seeds_rejected(self):
        with self.assertRaises(bundle.BundleError):
            bundle.collect(Path("/synthetic/repo"), ["scripts/example.py"] * 2, [])

    def test_local_imports_handle_cycles_relative_and_from_package(self):
        imports, count = bundle.local_imports("src/lumi/x.py", b"from .y import thing\nfrom . import z\nfrom lumi import posttraining_example\n")
        self.assertEqual(imports, {"src/lumi/y.py", "src/lumi/z.py", "src/lumi/posttraining_example.py", "src/lumi/__init__.py"})
        self.assertEqual(count, 0)

    def test_external_imports_not_read(self):
        paths, _ = bundle.local_imports("scripts/x.py", b"import torch\nfrom transformers import AutoModel\n")
        self.assertFalse(paths)

    def test_dynamic_imports_counted_not_executed(self):
        paths, count = bundle.local_imports("scripts/x.py", b"importlib.import_module(user_input)\nexec(user_input)\n")
        self.assertFalse(paths)
        self.assertEqual(count, 2)

    def test_unsupported_parent_import_rejected(self):
        with self.assertRaises(bundle.BundleError):
            bundle.local_imports("src/lumi/x.py", b"from ..outside import x\n")

    def test_invalid_syntax_sanitized(self):
        with self.assertRaisesRegex(bundle.BundleError, "^source_syntax_invalid$"):
            bundle.local_imports("scripts/x.py", b"not valid private exception syntax !!!")

    def test_credentials_rejected_without_returning_values(self):
        _, inspection = self.fixture()
        for raw in (b"-----BEGIN " + b"PRIVATE KEY-----", b"AKIA" + b"A" * 16,
                    b"ghp_" + b"a" * 35, b"aws_secret_access_key=" + b"synthetic-value"):
            with self.assertRaisesRegex(bundle.BundleError, "^credential_signature_detected$"):
                bundle.make_bundle({"src/lumi/x.py": raw}, inspection=inspection)

    def test_source_file_size_cap(self):
        files, inspection = self.fixture()
        with patch.object(bundle, "MAX_FILE_BYTES", 1), self.assertRaises(bundle.BundleError):
            bundle.make_bundle(files, inspection=inspection)

    def test_total_source_size_cap(self):
        files, inspection = self.fixture()
        with patch.object(bundle, "MAX_SOURCE_BYTES", 1), self.assertRaises(bundle.BundleError):
            bundle.make_bundle(files, inspection=inspection)

    def test_file_count_cap(self):
        files, inspection = self.fixture()
        with patch.object(bundle, "MAX_FILES", 0), self.assertRaises(bundle.BundleError):
            bundle.make_bundle(files, inspection=inspection)

    def test_archive_size_cap(self):
        files, inspection = self.fixture()
        with patch.object(bundle, "MAX_BUNDLE_BYTES", 100), self.assertRaises(bundle.BundleError):
            bundle.make_bundle(files, inspection=inspection)

    def test_wrong_digest_rejected(self):
        files, inspection = self.fixture()
        raw, _ = bundle.make_bundle(files, inspection=inspection)
        with self.assertRaises(bundle.BundleError):
            bundle.verify_bundle(raw, "0" * 64, files)

    def test_wrong_source_bytes_rejected(self):
        files, inspection = self.fixture()
        raw, _ = bundle.make_bundle(files, inspection=inspection)
        with self.assertRaises(bundle.BundleError):
            bundle.verify_bundle(raw, bundle.digest(raw), {"src/lumi/example.py": b"changed\n"})

    def test_wrong_expected_entry_set_rejected(self):
        files, inspection = self.fixture()
        raw, _ = bundle.make_bundle(files, inspection=inspection)
        with self.assertRaises(bundle.BundleError):
            bundle.verify_bundle(raw, bundle.digest(raw), {**files, "src/lumi/extra.py": b""})

    def test_truncated_archive_rejected(self):
        files, inspection = self.fixture()
        raw, _ = bundle.make_bundle(files, inspection=inspection)
        damaged = raw[:-100]
        with self.assertRaises(bundle.BundleError):
            bundle.verify_bundle(damaged, bundle.digest(damaged), files)

    def test_real_synthetic_regular_file_read(self):
        with tempfile.TemporaryDirectory(prefix="lumi-source-test-") as directory:
            root = Path(directory).resolve()
            (root / "scripts").mkdir()
            (root / "scripts/x.py").write_bytes(b"# synthetic\n")
            self.assertEqual(bundle.read_source(root, "scripts/x.py"), b"# synthetic\n")

    def test_symlink_file_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lumi-source-test-") as directory:
            root = Path(directory).resolve()
            (root / "scripts").mkdir()
            (root / "scripts/x.py").symlink_to(root / "missing.py")
            with self.assertRaisesRegex(bundle.BundleError, "^source_read_failed$"):
                bundle.read_source(root, "scripts/x.py")

    def test_symlink_parent_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lumi-source-test-") as directory:
            root = Path(directory).resolve()
            (root / "actual").mkdir()
            (root / "actual/x.py").write_bytes(b"# synthetic\n")
            (root / "scripts").symlink_to(root / "actual")
            with self.assertRaises(bundle.BundleError):
                bundle.read_source(root, "scripts/x.py")

    def test_hardlink_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lumi-source-test-") as directory:
            root = Path(directory).resolve()
            (root / "scripts").mkdir()
            (root / "scripts/x.py").write_bytes(b"# synthetic\n")
            os.link(root / "scripts/x.py", root / "scripts/y.py")
            with self.assertRaisesRegex(bundle.BundleError, "^source_not_regular_unique$"):
                bundle.read_source(root, "scripts/x.py")

    def test_fifo_rejected_without_blocking(self):
        with tempfile.TemporaryDirectory(prefix="lumi-source-test-") as directory:
            root = Path(directory).resolve()
            (root / "scripts").mkdir()
            os.mkfifo(root / "scripts/x.py")
            with self.assertRaises(bundle.BundleError):
                bundle.read_source(root, "scripts/x.py")

    def test_collect_closure_without_import_execution(self):
        fixture = {"scripts/x.py": b"from lumi.posttraining_fixture import value\n",
                   "src/lumi/posttraining_fixture.py": b"from .other import value\n",
                   "src/lumi/other.py": b"from .posttraining_fixture import value\n",
                   "src/lumi/__init__.py": b"raise RuntimeError('must not execute')\n"}
        with patch.object(bundle, "read_source", side_effect=lambda root, name: fixture[name]):
            files, report = bundle.collect(Path("/synthetic/repo"), ["scripts/x.py"], [])
        self.assertEqual(files, fixture)
        self.assertTrue(report["static_local_import_closure_included"])


if __name__ == "__main__":
    unittest.main()
