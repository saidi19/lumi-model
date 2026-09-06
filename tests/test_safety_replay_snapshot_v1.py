"""Synthetic-only tests of the immutable safety replay input view."""

from __future__ import annotations

import builtins
import concurrent.futures
import gzip
import importlib.util
import io
import os
import pathlib
import sys
import unittest
from unittest.mock import patch

SOURCE = pathlib.Path(__file__).parents[1] / "src/lumi/posttraining_safety_replay_snapshot_v1.py"
SPEC = importlib.util.spec_from_file_location("synthetic_safety_replay_snapshot", SOURCE)
snapshot_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = snapshot_module
SPEC.loader.exec_module(snapshot_module)
Snapshot = snapshot_module.Snapshot
ROOT = "/synthetic/workspace/repository"
TEXT_PATH = ROOT + "/data/text.jsonl"
GZIP_PATH = ROOT + "/hashes.jsonl.gz"
ARTIFACT_PATH = "/synthetic/workspace/lumi-artifacts/check.json"


class SnapshotTests(unittest.TestCase):
    def make_view(self, **kwargs):
        return Snapshot(
            {
                TEXT_PATH: b'{"safe":true}\n',
                GZIP_PATH: gzip.compress(b'{"hash":"synthetic"}\n', mtime=0),
                ARTIFACT_PATH: b'{"artifact":"synthetic"}',
            },
            repository_root=ROOT,
            **kwargs,
        )

    def assert_code(self, expected, callback):
        with self.assertRaises(ValueError) as caught:
            callback()
        self.assertEqual(str(caught.exception), expected)
        self.assertIsNone(caught.exception.__context__)

    def test_immutable_input_copy_and_bytes_only(self):
        blobs = {TEXT_PATH: b"original"}
        view = Snapshot(blobs, repository_root=ROOT)
        blobs[TEXT_PATH] = b"changed"
        blobs[ROOT + "/extra"] = b"unbound"
        self.assertEqual(view.Path(TEXT_PATH).read_bytes(), b"original")
        self.assertFalse(view.Path(ROOT + "/extra").exists())
        self.assert_code(
            "snapshot_configuration_rejected",
            lambda: Snapshot({TEXT_PATH: bytearray(b"mutable")}, repository_root=ROOT),
        )

    def test_path_factory_cwd_parent_name_join_and_relative_shape(self):
        view = self.make_view()
        self.assertEqual(str(view.Path.cwd()), ROOT)
        relative = view.Path("data") / "text.jsonl"
        self.assertFalse(relative.is_absolute())
        self.assertEqual(relative.name, "text.jsonl")
        self.assertEqual(str(relative.parent), "data")
        self.assertEqual(str(relative.resolve()), TEXT_PATH)
        self.assertTrue(relative.resolve().is_absolute())
        self.assertEqual(str(view.Path()), ".")
        self.assertEqual(str(ROOT / view.Path("data/text.jsonl")), TEXT_PATH)
        self.assertEqual(str(view.Path.cwd() / view.Path(TEXT_PATH)), TEXT_PATH)
        self.assertEqual(view.Path.cwd().expanduser(), view.Path.cwd())

    def test_parent_relative_bound_artifact_resolves_and_reads(self):
        view = self.make_view()
        path = view.Path.cwd() / ".." / "lumi-artifacts" / "check.json"
        self.assertEqual(str(path.resolve()), ARTIFACT_PATH)
        self.assertEqual(path.read_text(encoding="utf-8"), '{"artifact":"synthetic"}')
        self.assertTrue(path.resolve().is_relative_to(view.Path.cwd().parent))
        self.assertFalse(path.resolve().is_relative_to(view.Path.cwd()))

    def test_unknown_traversal_and_missing_reads_cannot_fall_back_to_filesystem(self):
        view = self.make_view()
        for path in ("unknown", "../../../../etc/passwd", "/etc/passwd", ROOT + "/missing"):
            with self.subTest(path=path):
                item = view.Path(path)
                self.assertFalse(item.is_file())
                self.assertFalse(item.exists())
                self.assert_code("snapshot_lookup_rejected", item.read_bytes)
                self.assert_code("snapshot_lookup_rejected", item.read_text)
                self.assert_code("snapshot_lookup_rejected", lambda item=item: item.open("rb"))
                self.assert_code(
                    "snapshot_lookup_rejected", lambda item=item: view.gzip_open(item, "rt")
                )
        self.assertEqual(view.summary()["virtual_opens"], 0)

    def test_only_bound_files_and_their_parents_exist(self):
        view = self.make_view()
        self.assertTrue(view.Path(TEXT_PATH).is_file())
        self.assertFalse(view.Path(TEXT_PATH).is_dir())
        self.assertTrue(view.Path(ROOT).exists())
        self.assertTrue(view.Path(ROOT).is_dir())
        self.assertFalse(view.Path(ROOT).is_file())
        self.assertTrue(view.Path("/").exists())
        self.assert_code("snapshot_lookup_rejected", view.Path(ROOT).read_bytes)
        self.assert_code("snapshot_lookup_rejected", lambda: view.Path("missing").resolve(True))
        self.assertEqual(view.Path(TEXT_PATH).resolve(True), view.Path(TEXT_PATH))

    def test_tilde_control_characters_and_pathlike_callbacks_rejected(self):
        view = self.make_view()

        class UntrustedPath:
            def __fspath__(self):
                raise AssertionError("must not call pathlike callback")

        for value in ("~/data", "~someone/data", ROOT + "/~hidden", "bad\x00name", "bad\nname",
                      "//synthetic/path", UntrustedPath(), 1, None):
            self.assert_code("snapshot_path_rejected", lambda value=value: view.Path(value))
        self.assertEqual(view.summary()["virtual_opens"], 0)

    def test_keys_and_root_must_be_normalized_absolute_paths(self):
        for key in ("relative", ROOT + "/../x", ROOT + "/./x", ROOT + "/x/", "/"):
            self.assert_code(
                "snapshot_configuration_rejected",
                lambda key=key: Snapshot({key: b"x"}, repository_root=ROOT),
            )
        for root in ("relative", ROOT + "/../other", ROOT + "/"):
            self.assert_code(
                "snapshot_configuration_rejected",
                lambda root=root: Snapshot({}, repository_root=root),
            )
        self.assert_code(
            "snapshot_configuration_rejected",
            lambda: Snapshot({ROOT: b"file", TEXT_PATH: b"child"}, repository_root=ROOT),
        )

    def test_foreign_view_paths_rejected_even_with_identical_names(self):
        view, foreign = self.make_view(), self.make_view()
        item = foreign.Path(TEXT_PATH)
        for callback in (
            lambda: view.Path(item),
            lambda: view.Path.cwd() / item,
            lambda: view.Path(TEXT_PATH).is_relative_to(foreign.Path.cwd()),
            lambda: view.gzip_open(foreign.Path(GZIP_PATH)),
        ):
            self.assert_code("snapshot_foreign_view_rejected", callback)
        self.assertEqual(view.summary()["virtual_opens"], 0)

    def test_os_fspath_escape_is_rejected(self):
        item = self.make_view().Path(TEXT_PATH)
        self.assert_code("snapshot_filesystem_escape_rejected", lambda: os.fspath(item))
        self.assert_code("snapshot_filesystem_escape_rejected", lambda: pathlib.Path(item))

    def test_readonly_binary_streams_predebit_full_blob(self):
        view = self.make_view()
        with view.Path(TEXT_PATH).open("rb") as stream:
            self.assertIsInstance(stream, io.BytesIO)
            self.assertFalse(stream.writable())
            self.assertTrue(stream.getbuffer().readonly)
            self.assertEqual(view.summary()["logical_read_bytes_debited"], 14)
            self.assertEqual(stream.read(1), b"{")
            for callback in (
                lambda: stream.write(b"x"),
                lambda: stream.writelines([b"x"]),
                lambda: stream.truncate(0),
            ):
                self.assert_code("snapshot_write_rejected", callback)
        self.assertEqual(view.summary()["virtual_opens"], 1)
        self.assertEqual(view.summary()["logical_read_bytes_debited"], 14)

    def test_readonly_text_streams_and_newline_translation(self):
        view = Snapshot({TEXT_PATH: b"a\r\nb\rc\n"}, repository_root=ROOT)
        with view.Path(TEXT_PATH).open("rt", encoding="utf-8") as stream:
            self.assertIsInstance(stream, io.StringIO)
            self.assertFalse(stream.writable())
            self.assertEqual(list(stream), ["a\n", "b\n", "c\n"])
            for callback in (lambda: stream.write("x"), lambda: stream.writelines(["x"]),
                             lambda: stream.truncate(0)):
                self.assert_code("snapshot_write_rejected", callback)
        self.assertEqual(view.Path(TEXT_PATH).read_text(newline=""), "a\r\nb\rc\n")

    def test_write_and_update_modes_rejected_without_debit(self):
        view = self.make_view()
        for mode in ("w", "wb", "a", "x", "r+", "r+b", "", "br", True, None):
            self.assert_code(
                "snapshot_mode_rejected", lambda mode=mode: view.Path(TEXT_PATH).open(mode)
            )
            self.assert_code(
                "snapshot_mode_rejected", lambda mode=mode: view.gzip_open(GZIP_PATH, mode)
            )
        self.assertEqual(view.summary()["virtual_opens"], 0)

    def test_gzip_adapter_uses_pinned_bytes_and_accounts_decoded_bytes(self):
        view = self.make_view()
        with view.gzip_open(view.Path(GZIP_PATH), "rt", encoding="utf-8") as stream:
            self.assertEqual(stream.read(), '{"hash":"synthetic"}\n')
            self.assertFalse(stream.writable())
        self.assertEqual(view.summary()["logical_read_bytes_debited"], 21)
        with view.gzip_open(GZIP_PATH, "r") as stream:
            self.assertEqual(stream.read(), b'{"hash":"synthetic"}\n')
        self.assertEqual(view.summary()["virtual_opens"], 2)

    def test_gzip_concatenated_members_are_bounded_together(self):
        raw = gzip.compress(b"abc", mtime=0) + gzip.compress(b"def", mtime=0)
        view = Snapshot({GZIP_PATH: raw}, repository_root=ROOT, maximum_gzip_bytes=6)
        with view.gzip_open(GZIP_PATH) as stream:
            self.assertEqual(stream.read(), b"abcdef")
        limited = Snapshot({GZIP_PATH: raw}, repository_root=ROOT, maximum_gzip_bytes=5)
        self.assert_code("snapshot_gzip_limit_exceeded", lambda: limited.gzip_open(GZIP_PATH))

    def test_gzip_expansion_limit_rejects_before_stream_return(self):
        raw = gzip.compress(b"x" * 1024, mtime=0)
        view = Snapshot({GZIP_PATH: raw}, repository_root=ROOT, maximum_gzip_bytes=1023)
        self.assert_code("snapshot_gzip_limit_exceeded", lambda: view.gzip_open(GZIP_PATH))
        self.assertEqual(view.summary()["logical_read_bytes_debited"], 0)

    def test_gzip_truncation_crc_failure_empty_and_trailing_junk_rejected(self):
        raw = gzip.compress(b"synthetic-data", mtime=0)
        damaged_crc = raw[:-8] + bytes([raw[-8] ^ 1]) + raw[-7:]
        for broken in (b"", raw[:2], raw[:-1], raw[:-8], damaged_crc, raw + b"junk"):
            view = Snapshot({GZIP_PATH: broken}, repository_root=ROOT)
            self.assert_code("snapshot_gzip_rejected", lambda view=view: view.gzip_open(GZIP_PATH))

    def test_failed_gzip_opens_consume_open_budget_without_returning_a_stream(self):
        view = Snapshot({GZIP_PATH: b"invalid"}, repository_root=ROOT, maximum_virtual_opens=1)
        self.assert_code("snapshot_gzip_rejected", lambda: view.gzip_open(GZIP_PATH))
        self.assertEqual(view.summary()["virtual_opens"], 1)
        self.assert_code("snapshot_open_limit_exceeded", lambda: view.gzip_open(GZIP_PATH))

    def test_open_budget_has_hard_inclusive_limit_and_no_close_refunds(self):
        view = Snapshot({TEXT_PATH: b"x"}, repository_root=ROOT, maximum_virtual_opens=2)
        view.Path(TEXT_PATH).read_bytes()
        view.Path(TEXT_PATH).read_bytes()
        self.assert_code("snapshot_open_limit_exceeded", view.Path(TEXT_PATH).read_bytes)
        self.assertEqual(view.summary()["virtual_opens"], 2)
        self.assertEqual(view.summary()["logical_read_bytes_debited"], 2)

    def test_total_logical_byte_budget_includes_full_partial_and_gzip_reads(self):
        view = Snapshot(
            {TEXT_PATH: b"abc", GZIP_PATH: gzip.compress(b"def", mtime=0)},
            repository_root=ROOT,
            maximum_logical_read_bytes=6,
        )
        with view.Path(TEXT_PATH).open("rb") as stream:
            self.assertEqual(stream.read(1), b"a")
        with view.gzip_open(GZIP_PATH) as stream:
            self.assertEqual(stream.read(1), b"d")
        self.assert_code("snapshot_read_limit_exceeded", view.Path(TEXT_PATH).read_bytes)
        self.assertEqual(view.summary()["logical_read_bytes_debited"], 6)

    def test_shared_budget_is_atomic_across_threads(self):
        view = Snapshot({TEXT_PATH: b"abc"}, repository_root=ROOT, maximum_logical_read_bytes=3)

        def read(_index):
            try:
                return view.Path(TEXT_PATH).read_bytes()
            except ValueError as error:
                return str(error)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            values = list(pool.map(read, range(2)))
        self.assertCountEqual(values, [b"abc", "snapshot_read_limit_exceeded"])
        self.assertEqual(view.summary()["logical_read_bytes_debited"], 3)

    def test_hard_caps_cannot_be_increased_or_changed_to_booleans(self):
        for field, hard_cap in (
            ("maximum_virtual_opens", 256),
            ("maximum_logical_read_bytes", 256 * 1024 * 1024),
            ("maximum_gzip_bytes", 32 * 1024 * 1024),
        ):
            for value in (0, True, hard_cap + 1):
                self.assert_code(
                    "snapshot_configuration_rejected",
                    lambda field=field, value=value: self.make_view(**{field: value}),
                )

    def test_decode_errors_are_fixed_and_do_not_retain_input(self):
        view = Snapshot({TEXT_PATH: b"\xffSYNTHETIC-PRIVATE-TEXT"}, repository_root=ROOT)
        self.assert_code("snapshot_encoding_rejected", view.Path(TEXT_PATH).read_text)
        self.assert_code(
            "snapshot_encoding_rejected", lambda: view.Path(TEXT_PATH).read_text("unknown-codec")
        )
        self.assertEqual(view.summary()["virtual_opens"], 2)

    def test_supported_text_codecs_are_memory_only_and_invalid_options_fail_fixed(self):
        view = Snapshot({TEXT_PATH: b"\xef\xbb\xbfsynthetic"}, repository_root=ROOT)
        self.assertEqual(view.Path(TEXT_PATH).read_text("utf-8-sig"), "synthetic")
        self.assertEqual(view.Path(TEXT_PATH).read_text("UTF_8"), "\ufeffsynthetic")
        for kwargs in ({"errors": "ignore"}, {"encoding": []}, {"encoding": "rot_13"}):
            self.assert_code(
                "snapshot_encoding_rejected",
                lambda kwargs=kwargs: view.Path(TEXT_PATH).read_text(**kwargs),
            )
        for kwargs in ({"newline": []}, {"newline": "invalid"}, {"buffering": True}):
            self.assert_code(
                "snapshot_mode_rejected",
                lambda kwargs=kwargs: view.Path(TEXT_PATH).open(**kwargs),
            )

    def test_summary_is_aggregate_and_defensive(self):
        view = self.make_view()
        summary = view.summary()
        self.assertNotIn(ROOT, repr(summary))
        self.assertFalse(summary["filesystem_access_performed"])
        self.assertFalse(summary["execution_permitted"])
        summary["virtual_opens"] = 999
        self.assertEqual(view.summary()["virtual_opens"], 0)

    def test_all_operations_work_while_actual_filesystem_entrypoints_are_forbidden(self):
        raw = gzip.compress(b"safe", mtime=0)
        with (
            patch.object(builtins, "open", side_effect=AssertionError("real open forbidden")),
            patch.object(io, "open", side_effect=AssertionError("real io.open forbidden")),
            patch.object(os, "open", side_effect=AssertionError("real os.open forbidden")),
            patch.object(os, "stat", side_effect=AssertionError("real stat forbidden")),
            patch.object(os, "lstat", side_effect=AssertionError("real lstat forbidden")),
            patch.object(os, "getcwd", side_effect=AssertionError("real cwd forbidden")),
            patch.object(gzip, "open", side_effect=AssertionError("real gzip.open forbidden")),
        ):
            view = Snapshot({TEXT_PATH: b"safe", GZIP_PATH: raw}, repository_root=ROOT)
            self.assertEqual(view.Path(TEXT_PATH).read_bytes(), b"safe")
            self.assertEqual(view.Path(TEXT_PATH).read_text(), "safe")
            self.assertTrue(view.Path(TEXT_PATH).is_file())
            self.assertTrue(view.Path(TEXT_PATH).parent.exists())
            self.assertEqual(str(view.Path.cwd()), ROOT)
            self.assertEqual(str(view.Path("../repository/data/text.jsonl").resolve()), TEXT_PATH)
            with view.gzip_open(GZIP_PATH, "rt", encoding="utf-8") as stream:
                self.assertEqual(stream.read(), "safe")
            self.assert_code("snapshot_lookup_rejected", view.Path("/etc/passwd").read_bytes)


if __name__ == "__main__":
    unittest.main()
