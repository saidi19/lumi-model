"""Tiny generated tar fixtures only; no real software, Docker, ML or network."""

from __future__ import annotations

import hashlib
import importlib
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
PACKAGE = "_lumi_synthetic_oci_rootfs_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
before_import = set(sys.modules)
subject = importlib.import_module(PACKAGE + ".posttraining_sft_oci_rootfs_v1")
IMPORT_ADDITIONS = set(sys.modules) - before_import


def member(name, kind="file", data=b"synthetic", *, link="", mode=0o644, pax=None):
    info = tarfile.TarInfo(name)
    info.type = {
        "file": tarfile.REGTYPE, "directory": tarfile.DIRTYPE,
        "symlink": tarfile.SYMTYPE, "hardlink": tarfile.LNKTYPE,
        "fifo": tarfile.FIFOTYPE, "character": tarfile.CHRTYPE,
        "block": tarfile.BLKTYPE, "sparse": tarfile.GNUTYPE_SPARSE, "unknown": b"Z",
    }[kind]
    info.size = len(data) if kind == "file" else 0
    info.linkname, info.mode, info.uid, info.gid = link, mode, 12345, 12346
    info.pax_headers = {} if pax is None else pax
    return info, data if kind == "file" else None


class RootfsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lumi-synthetic-rootfs-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.parent = self.base / "owned-parent"
        self.parent.mkdir(mode=0o700)
        self.archive = self.base / "synthetic.tar"
        self.write_archive([member("a")])
        self.owned = []
        self.addCleanup(self.cleanup_owned)

    def cleanup_owned(self):
        for owned in self.owned:
            if not owned.summary()["cleanup_attempted"]:
                owned.close()

    def write_archive(self, entries):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for info, data in entries:
                archive.addfile(info, None if data is None else io.BytesIO(data))
        self.archive.write_bytes(output.getvalue())

    def extract(self, **overrides):
        arguments = {
            "expected_sha256": hashlib.sha256(self.archive.read_bytes()).hexdigest(),
            "maximum_archive_bytes": 1024**2, "maximum_file_bytes": 32768,
            "maximum_total_bytes": 65536, "maximum_entries": 64,
            "deadline": time.monotonic() + 10,
        }
        arguments.update(overrides)
        result = subject.extract_rootfs(self.archive, self.parent, **arguments)
        self.owned.append(result)
        return result

    def rejected(self, entries=None, code=None, **overrides):
        if entries is not None:
            self.write_archive(entries)
        with self.assertRaises(subject.RootfsError) as caught:
            self.extract(**overrides)
        self.assertIn(caught.exception.code, subject.ERROR_CODES)
        if code is not None:
            self.assertEqual(caught.exception.code, code)
        self.assertEqual(caught.exception.args, (caught.exception.code,))
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        self.assertEqual(list(self.parent.iterdir()), [])

    def test_import_additions_are_stdlib_only(self):
        forbidden = {"torch", "transformers", "numpy", "tensorflow", "jax", "docker"}
        self.assertFalse({name.split(".", 1)[0] for name in IMPORT_ADDITIONS} & forbidden)

    def test_extract_regular_bytes_and_aggregate_summary(self):
        self.write_archive([member("./", "directory"), member("d/", "directory"),
                            member("d/a", data=b"tiny synthetic payload")])
        owned = self.extract()
        self.assertEqual((owned.path / "d/a").read_bytes(), b"tiny synthetic payload")
        result = owned.summary()
        self.assertEqual(result["archive_entries"], 3)
        self.assertEqual(result["owned_entries"], 3)
        self.assertEqual(result["content_bytes"], 22)
        self.assertTrue(result["metadata_normalized"])
        self.assertFalse(result["filesystem_immutable"])
        self.assertFalse(result["production_eligible"])
        self.assertFalse(result["real_sft_execution_permitted"])
        serialized = json.dumps(result)
        self.assertNotIn(str(self.base), serialized)
        self.assertNotIn("tiny synthetic payload", serialized)

    def test_root_directory_forms(self):
        for name in (".", "./", "/"):
            with self.subTest(name=name):
                self.write_archive([member(name, "directory"), member("a")])
                owned = self.extract()
                owned.close()

    def test_implicit_directories_can_later_be_declared_once(self):
        self.write_archive([member("d/a"), member("d", "directory", mode=0o755)])
        owned = self.extract()
        self.assertTrue((owned.path / "d/a").is_file())

    def test_pax_long_path_and_regular_pax_metadata(self):
        name = "a" * 150 + "/" + "b" * 150
        self.write_archive([member(name, pax={"comment": "synthetic only"})])
        owned = self.extract()
        self.assertEqual((owned.path / name).read_bytes(), b"synthetic")

    def test_global_pax_metadata_is_supported_and_counted(self):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT,
                          pax_headers={"comment": "synthetic global metadata"}) as archive:
            info, data = member("a")
            archive.addfile(info, io.BytesIO(data))
        self.archive.write_bytes(output.getvalue())
        owned = self.extract()
        self.assertEqual(owned.summary()["archive_headers"], 2)
        self.assertGreater(owned.summary()["metadata_bytes"], 0)

    def test_gnu_long_name_metadata_is_supported(self):
        output = io.BytesIO()
        name = "a" * 150 + "/" + "b" * 150
        with tarfile.open(fileobj=output, mode="w", format=tarfile.GNU_FORMAT) as archive:
            info, data = member(name)
            archive.addfile(info, io.BytesIO(data))
        self.archive.write_bytes(output.getvalue())
        owned = self.extract()
        self.assertEqual((owned.path / name).read_bytes(), b"synthetic")

    def test_absolute_and_parent_relative_symlink_targets_preserved(self):
        self.write_archive([member("d/a", "symlink", link="/outside/synthetic"),
                            member("b", "symlink", link="../outside/synthetic")])
        owned = self.extract()
        self.assertEqual(os.readlink(owned.path / "d/a"), "/outside/synthetic")
        self.assertEqual(os.readlink(owned.path / "b"), "../outside/synthetic")
        self.assertEqual(owned.summary()["symlinks"], 2)

    def test_hardlink_forward_backward_and_chained_order(self):
        self.write_archive([member("first", "hardlink", link="second"),
                            member("second", "hardlink", link="data"), member("data"),
                            member("last", "hardlink", link="./data")])
        owned = self.extract()
        identities = {(owned.path / name).stat().st_ino
                      for name in ("first", "second", "data", "last")}
        self.assertEqual(len(identities), 1)
        self.assertEqual(owned.summary()["hardlinks"], 3)
        self.assertEqual(owned.summary()["content_bytes"], 9)

    def test_normalizes_ownership_modes_and_preserves_directory_sticky_bit(self):
        self.write_archive([member("tmp", "directory", mode=0o3777),
                            member("bin/app", mode=0o6755)])
        owned = self.extract()
        path = owned.path
        self.assertEqual((path / "bin/app").stat().st_uid, os.geteuid())
        self.assertEqual(stat.S_IMODE((path / "bin/app").stat().st_mode), 0o755)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)
        owned.freeze()
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o755)
        self.assertEqual(stat.S_IMODE((path / "tmp").stat().st_mode), 0o1777)
        self.assertEqual(stat.S_IMODE((path / "bin").stat().st_mode), 0o755)
        self.assertEqual(stat.S_IMODE(self.parent.stat().st_mode), 0o700)

    def test_zero_mode_directories_remain_owned_and_cleanable(self):
        self.write_archive([member("d", "directory", mode=0), member("d/a", mode=0)])
        owned = self.extract()
        path = owned.path
        owned.freeze()
        self.assertEqual(stat.S_IMODE((path / "d").stat().st_mode), 0o700)
        owned.close()
        self.assertFalse(path.exists())

    def test_hash_rejected_before_root_creation(self):
        self.rejected(code="rootfs_archive_hash_rejected", expected_sha256="0" * 64)

    def test_archive_size_cap(self):
        self.rejected(code="rootfs_archive_rejected", maximum_archive_bytes=1024)

    def test_regular_file_size_cap(self):
        self.rejected(code="rootfs_bounds_exhausted", maximum_file_bytes=8)

    def test_total_content_cap(self):
        self.rejected([member("a", data=b"12345"), member("b", data=b"12345")],
                      code="rootfs_bounds_exhausted", maximum_file_bytes=6, maximum_total_bytes=9)

    def test_owned_entry_cap_includes_implicit_directories(self):
        self.rejected([member("a/b/c")], code="rootfs_bounds_exhausted", maximum_entries=3)

    def test_archive_entry_cap_includes_declared_root(self):
        self.rejected([member(".", "directory"), member("d", "directory"), member("a")],
                      code="rootfs_bounds_exhausted", maximum_entries=2)

    def test_bounds_and_pin_configuration(self):
        cases = [
            {"maximum_archive_bytes": 2 * 1024**3 + 1},
            {"maximum_total_bytes": 4 * 1024**3 + 1},
            {"maximum_file_bytes": 65537}, {"maximum_entries": 65537},
            {"maximum_entries": True}, {"maximum_entries": 0},
            {"expected_sha256": "A" * 64}, {"expected_sha256": "x"},
            {"deadline": float("nan")}, {"deadline": float("inf")}, {"deadline": True},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                self.rejected(code="rootfs_configuration_rejected", **overrides)

    def test_expired_deadline(self):
        self.rejected(code="rootfs_deadline_exhausted", deadline=time.monotonic() - 1)

    def test_deadline_exhaustion_after_creation_cleans_owned_root(self):
        with patch.object(subject.OwnedRootfs, "_member",
                          side_effect=subject.RootfsError("rootfs_deadline_exhausted")):
            self.rejected(code="rootfs_deadline_exhausted")

    def test_nonprivate_parent_rejected(self):
        self.parent.chmod(0o755)
        self.rejected(code="rootfs_configuration_rejected")

    def test_archive_symlink_rejected_without_following(self):
        target = self.base / "actual.tar"
        self.archive.rename(target)
        self.archive.symlink_to(target)
        self.rejected()

    def test_archive_symlink_ancestor_rejected(self):
        alias = self.base / "alias"
        alias.symlink_to(self.base, target_is_directory=True)
        self.archive = alias / "synthetic.tar"
        self.rejected()

    def test_parent_symlink_rejected(self):
        alias = self.base / "alias"
        alias.symlink_to(self.parent, target_is_directory=True)
        original = self.parent
        self.parent = alias
        self.rejected()
        self.parent = original

    def test_archive_directory_rejected_without_content_reads(self):
        with self.assertRaises(subject.RootfsError):
            subject.extract_rootfs(self.parent, self.parent, expected_sha256="0" * 64,
                                   maximum_archive_bytes=1024**2, maximum_file_bytes=10,
                                   maximum_total_bytes=100, maximum_entries=20,
                                   deadline=time.monotonic() + 10)
        self.assertEqual(list(self.parent.iterdir()), [])

    def test_traversal_absolute_and_ambiguous_paths_rejected(self):
        for name in ("../outside", "a/../../outside", "/outside", "a/../b", "a/./b",
                     "a//b", "a\\b", "a" * 256, "/".join(["a"] * 65)):
            with self.subTest(name=name):
                self.rejected([member(name)], code="rootfs_path_rejected")

    def test_path_nul_rejected_before_filesystem_use(self):
        with self.assertRaises(subject.RootfsError):
            subject._parts("a\x00b")

    def test_duplicate_file_directory_and_root_rejected(self):
        for entries in ([member("a"), member("./a")],
                        [member("d", "directory"), member("d/", "directory")],
                        [member(".", "directory"), member("./", "directory")]):
            with self.subTest(entries=entries):
                self.rejected(entries, code="rootfs_duplicate_rejected")

    def test_file_replacing_implicit_directory_rejected(self):
        self.rejected([member("a/b"), member("a")], code="rootfs_duplicate_rejected")

    def test_special_and_unknown_types_rejected(self):
        for kind in ("fifo", "character", "block", "unknown", "sparse"):
            with self.subTest(kind=kind):
                self.rejected([member("a", kind)], code="rootfs_entry_rejected")

    def test_pax_sparse_metadata_rejected(self):
        for pax in ({"GNU.sparse.size": "9"}, {"GNU.sparse.map": "0,9"},
                    {"GNU.sparse.major": "1", "GNU.sparse.minor": "0"}):
            with self.subTest(pax=pax):
                self.rejected([member("a", pax=pax)], code="rootfs_entry_rejected")

    def test_pax_headers_count_towards_entry_bound(self):
        self.rejected([member("a", pax={"comment": "synthetic"})],
                      code="rootfs_bounds_exhausted", maximum_entries=1)

    def test_oversized_pax_body_rejected(self):
        self.rejected([member("a", pax={"comment": "x" * 65536})],
                      code="rootfs_bounds_exhausted")

    def test_writes_through_symlink_ancestors_rejected(self):
        self.rejected([member("d", "symlink", link=str(self.base)), member("d/outside")],
                      code="rootfs_path_rejected")
        self.assertFalse((self.base / "outside").exists())

    def test_writes_through_file_ancestors_rejected(self):
        self.rejected([member("d"), member("d/outside")], code="rootfs_path_rejected")

    def test_hardlink_missing_cycle_directory_and_symlink_rejected(self):
        cases = [
            [member("a", "hardlink", link="absent")],
            [member("a", "hardlink", link="b"), member("b", "hardlink", link="a")],
            [member("d", "directory"), member("a", "hardlink", link="d")],
            [member("s", "symlink", link="/outside"), member("a", "hardlink", link="s")],
        ]
        for entries in cases:
            with self.subTest(entries=entries):
                self.rejected(entries, code="rootfs_hardlink_rejected")

    def test_hardlink_path_traversal_rejected(self):
        for target in ("../outside", "/outside", "d/../a"):
            with self.subTest(target=target):
                self.rejected([member("a", "hardlink", link=target)], code="rootfs_path_rejected")

    def test_nonzero_trailing_archive_bytes_rejected(self):
        self.archive.write_bytes(self.archive.read_bytes() + b"NOT-TAR")
        self.rejected(code="rootfs_archive_rejected")

    def test_missing_terminator_rejected(self):
        self.archive.write_bytes(self.archive.read_bytes()[:1024])
        self.rejected(code="rootfs_archive_rejected")

    def test_only_one_terminator_block_rejected(self):
        self.archive.write_bytes(self.archive.read_bytes()[:1536])
        self.rejected(code="rootfs_archive_rejected")

    def test_invalid_header_after_valid_member_is_not_silently_ignored(self):
        original = self.archive.read_bytes()
        self.archive.write_bytes(original[:1024] + b"INVALID" + original[1031:])
        self.rejected(code="rootfs_archive_rejected")

    def test_compressed_archive_rejected(self):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w:gz") as archive:
            archive.addfile(*member("a", data=b""))
        self.archive.write_bytes(output.getvalue() + b"\x00" * 1024)
        self.rejected()

    def test_exact_two_pinned_installations_and_freeze(self):
        owned = self.extract()
        path = owned.path
        for name, mode in (("worker_entry.py", 0o444), ("admission.json", 0o644)):
            data = b"synthetic pinned control"
            target = "opt/lumi-release/guardian/" + name
            owned.install_file(target, data, expected_sha256=hashlib.sha256(data).hexdigest(),
                               mode=mode)
            self.assertEqual((path / target).read_bytes(), data)
            self.assertEqual(stat.S_IMODE((path / target).stat().st_mode), mode)
        owned.freeze()
        self.assertTrue(owned.summary()["installation_frozen"])
        self.assertEqual(owned.summary()["installed_files"], 2)
        self.assertEqual(owned.summary()["installed_bytes"], 48)
        self.assertEqual(stat.S_IMODE((path / "opt/lumi-release/guardian").stat().st_mode), 0o755)

    def test_installation_wrong_path_hash_type_and_mode_rejected(self):
        owned = self.extract()
        target, data = "opt/lumi-release/guardian/worker_entry.py", b"synthetic"
        cases = [("elsewhere", data, hashlib.sha256(data).hexdigest(), 0o644),
                 (target, data, "0" * 64, 0o644),
                 (target, "synthetic", hashlib.sha256(data).hexdigest(), 0o644),
                 (target, data, hashlib.sha256(data).hexdigest(), 0o755)]
        for name, content, digest, mode in cases:
            with self.subTest(name=name, mode=mode):
                with self.assertRaisesRegex(subject.RootfsError, "rootfs_installation_rejected"):
                    owned.install_file(name, content, expected_sha256=digest, mode=mode)
        self.assertEqual(owned.summary()["installed_files"], 0)

    def test_installation_cannot_overwrite_original(self):
        target = "opt/lumi-release/guardian/worker_entry.py"
        self.write_archive([member(target)])
        owned = self.extract()
        with self.assertRaisesRegex(subject.RootfsError, "rootfs_duplicate_rejected"):
            owned.install_file(target, b"replacement", expected_sha256=hashlib.sha256(
                b"replacement").hexdigest())
        self.assertEqual((owned.path / target).read_bytes(), b"synthetic")

    def test_installation_counts_towards_total_content_cap(self):
        owned = self.extract(maximum_file_bytes=10, maximum_total_bytes=10)
        with self.assertRaisesRegex(subject.RootfsError, "rootfs_bounds_exhausted"):
            owned.install_file("opt/lumi-release/guardian/worker_entry.py", b"xx",
                               expected_sha256=hashlib.sha256(b"xx").hexdigest())
        self.assertEqual(owned.summary()["installed_files"], 0)

    def test_installation_cannot_follow_existing_image_symlink(self):
        self.write_archive([member("opt", "symlink", link=str(self.base))])
        owned = self.extract()
        with self.assertRaisesRegex(subject.RootfsError, "rootfs_path_rejected"):
            owned.install_file("opt/lumi-release/guardian/worker_entry.py", b"x",
                               expected_sha256=hashlib.sha256(b"x").hexdigest())
        self.assertFalse((self.base / "lumi-release").exists())

    def test_failed_adapter_write_cannot_be_frozen_or_retried(self):
        owned = self.extract()
        with patch.object(subject.os, "write", side_effect=OSError("private synthetic error")):
            with self.assertRaisesRegex(subject.RootfsError, "rootfs_operation_failed"):
                owned.install_file("opt/lumi-release/guardian/worker_entry.py", b"x",
                                   expected_sha256=hashlib.sha256(b"x").hexdigest())
        self.assertTrue(owned.summary()["operation_tainted"])
        with self.assertRaisesRegex(subject.RootfsError, "rootfs_operation_failed"):
            owned.freeze()
        with self.assertRaisesRegex(subject.RootfsError, "rootfs_operation_failed"):
            owned.install_file("opt/lumi-release/guardian/admission.json", b"x",
                               expected_sha256=hashlib.sha256(b"x").hexdigest())

    def test_freeze_is_one_shot_and_stops_installation(self):
        owned = self.extract()
        owned.freeze()
        with self.assertRaisesRegex(subject.RootfsError, "rootfs_frozen"):
            owned.freeze()
        with self.assertRaisesRegex(subject.RootfsError, "rootfs_frozen"):
            owned.install_file("opt/lumi-release/guardian/worker_entry.py", b"x",
                               expected_sha256=hashlib.sha256(b"x").hexdigest())

    def test_cleanup_owned_tree_preserves_symlink_targets_and_archive(self):
        outside = self.base / "outside"
        outside.write_bytes(b"unowned")
        self.write_archive([member("d/a"), member("link", "symlink", link=str(outside))])
        owned = self.extract()
        path = owned.path
        owned.close()
        self.assertFalse(path.exists())
        self.assertEqual(outside.read_bytes(), b"unowned")
        self.assertTrue(self.archive.exists())
        self.assertTrue(owned.summary()["cleanup_completed"])
        with self.assertRaisesRegex(subject.RootfsError, "rootfs_already_closed"):
            owned.close()

    def test_cleanup_runs_after_deadline_expiry(self):
        owned = self.extract()
        path = owned.path
        with patch.object(subject.time, "monotonic", return_value=10**30):
            owned.close()
        self.assertFalse(path.exists())

    def test_cleanup_refuses_untracked_content_before_deleting_anything(self):
        owned = self.extract()
        path = owned.path
        (path / "foreign").write_bytes(b"must remain")
        with self.assertRaisesRegex(subject.RootfsError, "rootfs_identity_changed"):
            owned.close()
        self.assertEqual((path / "foreign").read_bytes(), b"must remain")
        self.assertTrue((path / "a").exists())
        self.assertFalse(owned.summary()["cleanup_completed"])

    def test_cleanup_refuses_replaced_file_identity(self):
        owned = self.extract()
        path = owned.path
        (path / "a").rename(path / "retained-original")
        (path / "a").write_bytes(b"foreign replacement")
        with self.assertRaisesRegex(subject.RootfsError, "rootfs_identity_changed"):
            owned.close()
        self.assertEqual((path / "a").read_bytes(), b"foreign replacement")

    def test_cleanup_refuses_replaced_directory_symlink(self):
        self.write_archive([member("d/a")])
        owned = self.extract()
        path = owned.path
        (path / "d").rename(path / "original")
        (path / "d").symlink_to(self.base, target_is_directory=True)
        with self.assertRaisesRegex(subject.RootfsError, "rootfs_identity_changed"):
            owned.close()
        self.assertTrue((path / "original/a").exists())
        self.assertTrue(self.archive.exists())

    def test_cleanup_refuses_replaced_root_identity(self):
        owned = self.extract()
        path = owned.path
        renamed = self.parent / "renamed-owned-root"
        path.rename(renamed)
        path.mkdir(mode=0o700)
        (path / "foreign").write_bytes(b"remain")
        with self.assertRaisesRegex(subject.RootfsError, "rootfs_identity_changed"):
            owned.close()
        self.assertTrue((renamed / "a").exists())
        self.assertEqual((path / "foreign").read_bytes(), b"remain")

    def test_freeze_rejects_untracked_contents(self):
        owned = self.extract()
        (owned.path / "foreign").write_bytes(b"synthetic")
        with self.assertRaisesRegex(subject.RootfsError, "rootfs_identity_changed"):
            owned.freeze()
        (owned.path / "foreign").unlink()

    def test_archive_mutation_between_hash_and_extraction_rejected(self):
        original = subject.OwnedRootfs._member

        def mutate(owned, *args):
            original(owned, *args)
            current = self.archive.stat()
            os.utime(self.archive, ns=(current.st_atime_ns, current.st_mtime_ns + 1000000))

        with patch.object(subject.OwnedRootfs, "_member", new=mutate):
            self.rejected(code="rootfs_archive_rejected")

    def test_archive_read_failure_after_root_creation_still_cleans(self):
        original = subject.OwnedRootfs._member

        def fail_after_member(owned, *args):
            original(owned, *args)
            raise OSError("synthetic private exception")

        with patch.object(subject.OwnedRootfs, "_member", new=fail_after_member):
            self.rejected(code="rootfs_operation_failed")

    def test_errors_hide_raw_underlying_exception_text(self):
        with patch.object(subject.os, "read", side_effect=OSError("PRIVATE PATH RAW TEXT")):
            self.rejected(code="rootfs_operation_failed")


if __name__ == "__main__":
    unittest.main()
