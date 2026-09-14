"""Tiny generated trees only; no native guard, actual runtime or model inputs."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import io
import json
import os
import stat
import sys
import tarfile
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_lumi_synthetic_import_sources_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
before_import = set(sys.modules)
subject = importlib.import_module(PACKAGE + ".posttraining_checkpoint_guard_sources_v1")
IMPORT_ADDITIONS = set(sys.modules) - before_import
spec = importlib.util.spec_from_file_location(
    "_lumi_synthetic_import_entry_sources_v1", ROOT / "scripts/run_checkpoint_guarded_import_v1.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)  # Definitions only; never native main.


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def bundle():
    return {"lumi/__init__.py": b'"""Original generated initializer."""\n',
            "lumi/posttraining_checkpoint_guard_fixture_v1.py": b"VALUE = 1\n"}


class ContractTests(unittest.TestCase):
    def test_stdlib_only(self):
        self.assertFalse({name.split(".", 1)[0] for name in IMPORT_ADDITIONS}
                         & {"lumi", "torch", "numpy", "transformers", "safetensors"})

    def test_manifest_matches_fixed_entry_contract_and_is_deterministic(self):
        files = bundle()
        raw = subject.build_manifest(files)
        self.assertEqual(raw, subject.build_manifest(dict(reversed(list(files.items())))))
        rows = entry.validate_source_manifest(raw, digest(raw))
        self.assertEqual(set(rows), set(files))
        self.assertEqual(rows["lumi/__init__.py"]["sha256"], digest(files["lumi/__init__.py"]))

    def test_noncanonical_or_extra_manifest_fields_rejected(self):
        files = bundle()
        value = json.loads(subject.build_manifest(files))
        for raw in [json.dumps(value).encode(), subject.build_manifest(files) + b"\n",
                    b'{"schema":"wrong","files":[]}']:
            with self.subTest(raw=raw), self.assertRaises(subject.SourceError):
                subject.validate_bundle(files, raw, b"entry", expected_manifest_sha256=digest(raw),
                                        expected_entry_sha256=digest(b"entry"))

    def test_wrong_bindings_rejected(self):
        files = bundle()
        raw = subject.build_manifest(files)
        for manifest_pin, entry_pin in [("0" * 64, digest(b"entry")), (digest(raw), "0" * 64)]:
            with self.subTest(manifest_pin=manifest_pin), self.assertRaises(subject.SourceError):
                subject.validate_bundle(files, raw, b"entry", expected_manifest_sha256=manifest_pin,
                                        expected_entry_sha256=entry_pin)

    def test_missing_required_file_rejected(self):
        for name in bundle():
            files = bundle()
            del files[name]
            files["lumi/extra.py"] = b""
            with self.subTest(name=name), self.assertRaises(subject.SourceError):
                subject.build_manifest(files)

    def test_bad_names_and_nonbytes_rejected(self):
        for name in ["lumi/../escape.py", "lumi/nested/file.py", "/lumi/a.py", "lumi/a.pyc",
                     "other/a.py", "lumi/a-b.py", "lumi/a.py\n", 1]:
            files = bundle()
            files[name] = b""
            with self.subTest(name=name), self.assertRaises(subject.SourceError):
                subject.build_manifest(files)
        for body in ["source", bytearray(b"source"), None, 3]:
            files = bundle()
            files["lumi/a.py"] = body
            with self.subTest(body=body), self.assertRaises(subject.SourceError):
                subject.build_manifest(files)

    def test_count_and_file_and_total_caps(self):
        cases = [dict(bundle(), **{f"lumi/f{i}.py": b"" for i in range(127)}),
                 dict(bundle(), **{"lumi/large.py": b"x" * (subject.MAXIMUM_FILE_BYTES + 1)}),
                 dict(bundle(), **{f"lumi/f{i}.py": b"x" * subject.MAXIMUM_FILE_BYTES
                                   for i in range(17)})]
        for files in cases:
            with self.subTest(count=len(files)), self.assertRaises(subject.SourceError):
                subject.build_manifest(files)

    def test_manifest_entry_argument_types_and_caps(self):
        files = bundle()
        manifest = subject.build_manifest(files)
        for raw, code, mp, ep in [(manifest.decode(), b"entry", digest(manifest), digest(b"entry")),
                                  (manifest, b"", digest(manifest), digest(b"")),
                                  (manifest, b"entry", True, digest(b"entry")),
                                  (manifest, b"entry", digest(manifest), "A" * 64),
                                  (manifest, b"x" * (262144 + 1), digest(manifest), "0" * 64),
                                  (b"x" * 65537, b"entry", "0" * 64, digest(b"entry"))]:
            with self.assertRaises(subject.SourceError):
                subject.validate_bundle(files, raw, code, expected_manifest_sha256=mp,
                                        expected_entry_sha256=ep)

    def test_runtime_denial_before_filesystem(self):
        with patch.object(subject.sys, "platform", "darwin"), \
                patch.object(subject.rootfs.OwnedRootfs, "_guard") as guard:
            with self.assertRaises(subject.SourceError) as caught:
                subject.install_importer_addon(object(), {}, b"", b"", expected_manifest_sha256="",
                                               expected_entry_sha256="")
            self.assertEqual(caught.exception.code, "source_runtime_rejected")
            guard.assert_not_called()


class InstallationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lumi-generated-source-addon-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.parent = self.base / "parent"
        self.parent.mkdir(mode=0o700)
        self.files = bundle()
        self.manifest = subject.build_manifest(self.files)
        self.code = b'"""Generated inert entry."""\n'
        self.owned = []
        self.addCleanup(self.cleanup_owned)
        runtime = patch.object(subject, "_runtime")
        runtime.start()
        self.addCleanup(runtime.stop)

    def cleanup_owned(self):
        for owned in self.owned:
            if not owned.summary()["cleanup_attempted"]:
                owned.close()

    def extract(self, name="usr/fixture.txt", *, kind=tarfile.REGTYPE, limits=None):
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as archive:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.mode = 0o755 if kind == tarfile.DIRTYPE else 0o644
            info.size = 0 if kind == tarfile.DIRTYPE else 7
            archive.addfile(info, io.BytesIO(b"fixture") if info.size else None)
        path = self.base / f"generated-{len(self.owned)}.tar"
        path.write_bytes(raw.getvalue())
        options = {"maximum_archive_bytes": 1024**2, "maximum_file_bytes": 32768,
                   "maximum_total_bytes": 65536, "maximum_entries": 64}
        options.update(limits or {})
        owned = subject.rootfs.extract_rootfs(path, self.parent,
                                             expected_sha256=digest(raw.getvalue()),
                                             deadline=time.monotonic() + 10, **options)
        self.owned.append(owned)
        return owned

    def install(self, owned, **kwargs):
        return subject.install_importer_addon(
            owned, self.files, self.manifest, self.code,
            expected_manifest_sha256=kwargs.get("manifest_pin", digest(self.manifest)),
            expected_entry_sha256=kwargs.get("entry_pin", digest(self.code)))

    def test_install_exact_bytes_readonly_files_and_freeze(self):
        owned = self.extract()
        root = owned.path
        prior = (root / "usr/fixture.txt").read_bytes()
        result = self.install(owned)
        self.assertEqual(result["addon_files"], 4)
        self.assertEqual(result["addon_bytes"], sum(map(len, self.files.values()))
                         + len(self.manifest) + len(self.code))
        self.assertTrue(result["installation_frozen"])
        for key in ("filesystem_immutable", "base_origin_verified_by_this_adapter",
                    "native_guardian_verified", "real_import_authorized", "production_eligible"):
            self.assertIs(result[key], False)
        for name, body in self.files.items():
            path = root / subject.SOURCE_PREFIX / name
            self.assertEqual(path.read_bytes(), body)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o444)
            self.assertEqual(path.stat().st_uid, os.geteuid())
        self.assertEqual((root / subject.MANIFEST_PATH).read_bytes(), self.manifest)
        self.assertEqual((root / subject.ENTRY_PATH).read_bytes(), self.code)
        self.assertEqual((root / "usr/fixture.txt").read_bytes(), prior)
        self.assertNotIn(str(root), json.dumps(result))
        self.assertEqual(owned.summary()["installed_files"], 4)

    def test_frozen_root_and_second_install_rejected(self):
        owned = self.extract()
        self.install(owned)
        before = owned.summary()
        with self.assertRaises(subject.SourceError):
            self.install(owned)
        self.assertEqual(owned.summary(), before)

    def test_existing_importer_subtree_rejected_before_write(self):
        owned = self.extract("opt/lumi-release/importer", kind=tarfile.DIRTYPE)
        before = owned.summary()
        with self.assertRaises(subject.SourceError):
            self.install(owned)
        self.assertEqual(owned.summary(), before)

    def test_existing_entry_and_nondirectory_ancestor_rejected(self):
        for name in (subject.ENTRY_PATH, "opt/lumi-release"):
            owned = self.extract(name)
            before = owned.summary()
            with self.assertRaises(subject.SourceError):
                self.install(owned)
            self.assertEqual(owned.summary(), before)

    def test_all_capacity_checks_precede_first_write(self):
        for limits in [{"maximum_entries": 5}, {"maximum_file_bytes": 10},
                       {"maximum_file_bytes": 20, "maximum_total_bytes": 20}]:
            owned = self.extract(limits=limits)
            with patch.object(owned, "_write", wraps=owned._write) as write:
                with self.assertRaises(subject.SourceError):
                    self.install(owned)
                write.assert_not_called()
            self.assertFalse(owned.summary()["operation_tainted"])

    def test_wrong_content_binding_has_no_tree_effect(self):
        owned = self.extract()
        with patch.object(owned, "_guard", wraps=owned._guard) as guard:
            with self.assertRaises(subject.SourceError):
                self.install(owned, entry_pin="0" * 64)
            guard.assert_not_called()

    def test_started_write_failure_taints_and_redacts(self):
        owned = self.extract()
        with patch.object(owned, "_write", side_effect=OSError("private source text")) as write:
            with self.assertRaises(subject.SourceError) as caught:
                self.install(owned)
            self.assertEqual(write.call_count, 1)
        self.assertTrue(owned.summary()["operation_tainted"])
        self.assertEqual(caught.exception.args, ("source_installation_rejected",))
        self.assertIsNone(caught.exception.__context__)
        with self.assertRaises(subject.SourceError):
            self.install(owned)

    def test_failed_freeze_taints_no_retry(self):
        owned = self.extract()
        with patch.object(owned, "freeze", side_effect=OSError("private detail")):
            with self.assertRaises(subject.SourceError):
                self.install(owned)
        self.assertTrue(owned.summary()["operation_tainted"])
        self.assertEqual(owned.summary()["installed_files"], 4)

    def test_original_installer_allowlist_is_not_widened(self):
        owned = self.extract()
        with self.assertRaises(subject.rootfs.RootfsError):
            owned.install_file(subject.ENTRY_PATH, self.code, expected_sha256=digest(self.code))
        self.assertEqual(owned.summary()["installed_files"], 0)

    def test_nonexact_owned_type_rejected(self):
        with self.assertRaises(subject.SourceError):
            self.install(object())
