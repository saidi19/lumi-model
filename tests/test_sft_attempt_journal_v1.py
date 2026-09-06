"""Synthetic tests for the create-once journal; no ML, real inputs or workers."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SOURCE = Path(__file__).parents[1] / "src/lumi/posttraining_sft_attempt_journal_v1.py"
SPEC = importlib.util.spec_from_file_location("synthetic_attempt_journal", SOURCE)
journal_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(journal_module)


class AttemptJournalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lumi-journal-synthetic-")
        self.root = Path(self.temporary.name).resolve()
        self.charges = []
        self.handles = []

    def tearDown(self):
        for handle in self.handles:
            handle.close()
        self.temporary.cleanup()

    def create(self, **kwargs):
        handle = journal_module.create_new(
            self.root,
            debit=kwargs.get("debit", self.charges.append),
            maximum_bytes=kwargs.get("maximum_bytes", 65_536),
        )
        self.handles.append(handle)
        return handle

    def assert_code(self, code, operation):
        with self.assertRaises(journal_module.JournalError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(str(caught.exception), code)
        self.assertIsNone(caught.exception.__context__)
        self.assertIsNone(caught.exception.__cause__)

    def read(self):
        return (self.root / journal_module.FILENAME).read_bytes()

    def decode(self, raw=None):
        raw = self.read() if raw is None else raw
        output = []
        while raw:
            size = int.from_bytes(raw[:4], "big")
            output.append(json.loads(raw[4 : 4 + size]))
            raw = raw[4 + size :]
        return output

    def test_create_appends_initial_record_and_file_is_private(self):
        handle = self.create()
        info = (self.root / journal_module.FILENAME).stat()
        self.assertEqual(stat.S_IMODE(info.st_mode), 0o600)
        self.assertEqual(info.st_nlink, 1)
        self.assertEqual(self.decode()[0]["event"], "attempt_created")
        self.assertEqual(self.decode()[0]["counters"], [])
        self.assertEqual(sum(self.charges), len(self.read()))
        self.assertEqual(handle.summary()["records_committed"], 1)

    def test_debit_write_file_fsync_and_directory_fsync_precede_return(self):
        events = []
        real_write, real_fsync = os.write, os.fsync

        def debit(amount):
            events.append(("debit", amount))

        def write(fd, raw):
            self.assertEqual(events[-1], ("debit", len(raw)))
            events.append(("write", len(raw)))
            return real_write(fd, raw)

        def fsync(fd):
            kind = "directory_fsync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file_fsync"
            events.append((kind, 0))
            return real_fsync(fd)

        with mock.patch.object(journal_module.os, "write", side_effect=write):
            with mock.patch.object(journal_module.os, "fsync", side_effect=fsync):
                self.create(debit=debit)
        self.assertEqual(
            [event[0] for event in events], ["debit", "write", "file_fsync", "directory_fsync"]
        )

    def test_append_returns_committed_hash_only_after_fsync(self):
        handle = self.create()
        before = len(self.read())
        result = handle.append("resource_reserved", (0, 4, 19))
        frame = self.read()[before:]
        self.assertEqual(
            result, {"sequence": 2, "record_sha256": hashlib.sha256(frame).hexdigest()}
        )
        self.assertEqual(self.decode()[1]["counters"], [0, 4, 19])
        self.assertEqual(
            journal_module.verify_bytes(self.read())["last_record_sha256"], result["record_sha256"]
        )

    def test_every_allowed_event_and_maximum_counter_round_trip(self):
        handle = self.create()
        for event in sorted(
            journal_module.EVENTS - {"attempt_created"} - journal_module.TERMINAL_EVENTS
        ):
            handle.append(event, (journal_module.MAXIMUM_COUNTER,) * 16)
        handle.append("attempt_completed")
        report = journal_module.verify_bytes(self.read())
        self.assertTrue(report["complete_attempt"])
        self.assertFalse(report["resume_permitted"])
        self.assertFalse(report["execution_permitted"])

    def test_existing_artifact_is_never_opened_modified_or_charged(self):
        handle = self.create()
        handle.close()
        raw = self.read()
        before = list(self.charges)
        self.assert_code("attempt_already_exists", self.create)
        self.assertEqual(self.read(), raw)
        self.assertEqual(self.charges, before)

    def test_empty_existing_artifact_is_consumed(self):
        (self.root / journal_module.FILENAME).touch(mode=0o600)
        self.assert_code("attempt_already_exists", self.create)
        self.assertEqual(self.read(), b"")

    def test_symlink_file_is_not_followed(self):
        target = self.root / "synthetic-target"
        target.write_bytes(b"SYNTHETIC-PRIVATE-TEXT")
        (self.root / journal_module.FILENAME).symlink_to(target)
        self.assert_code("attempt_already_exists", self.create)
        self.assertEqual(target.read_bytes(), b"SYNTHETIC-PRIVATE-TEXT")

    def test_symlink_directory_and_ancestor_are_rejected(self):
        target = self.root / "target"
        target.mkdir(mode=0o700)
        child = target / "child"
        child.mkdir(mode=0o700)
        link = self.root / "link"
        link.symlink_to(target, target_is_directory=True)
        for directory in (link, link / "child"):
            self.assert_code(
                "directory_invalid",
                lambda directory=directory: journal_module.create_new(
                    directory, debit=self.charges.append, maximum_bytes=65_536
                ),
            )
        self.assertFalse((target / journal_module.FILENAME).exists())
        self.assertFalse((child / journal_module.FILENAME).exists())

    def test_permissive_and_nonowned_directories_rejected(self):
        self.root.chmod(0o755)
        self.assert_code("directory_invalid", self.create)
        self.root.chmod(0o700)
        with mock.patch.object(journal_module.os, "geteuid", return_value=os.geteuid() + 1):
            self.assert_code("directory_invalid", self.create)

    def test_bad_directory_paths_have_no_raw_error_context(self):
        for bad in (
            "relative/SYNTHETIC-SECRET",
            str(self.root) + "/../elsewhere",
            "//synthetic",
            str(self.root) + "/SYNTHETIC-SECRET-MISSING",
            "/synthetic\nsecret",
            None,
        ):
            self.assert_code(
                "directory_invalid",
                lambda bad=bad: journal_module.create_new(
                    bad, debit=self.charges.append, maximum_bytes=65_536
                ),
            )

    def test_cap_and_callback_configuration_exact_types(self):
        for bad in (True, 0, -1, 1.5, "1024", journal_module.MAXIMUM_BYTES + 1):
            self.assert_code(
                "configuration_invalid", lambda bad=bad: self.create(maximum_bytes=bad)
            )
        self.assert_code("configuration_invalid", lambda: self.create(debit=None))
        self.assert_code("journal_byte_cap_exceeded", lambda: self.create(maximum_bytes=1))
        self.assertFalse((self.root / journal_module.FILENAME).exists())

    def test_cap_includes_initial_record_and_is_inclusive(self):
        initial_size = len(
            journal_module._frame(1, journal_module.ZERO_DIGEST, "attempt_created", ())
        )
        handle = self.create(maximum_bytes=initial_size)
        self.assertEqual(len(self.read()), initial_size)
        before = list(self.charges)
        self.assert_code("journal_byte_cap_exceeded", lambda: handle.append("epoch_started"))
        self.assertEqual(self.charges, before)
        self.assertEqual(len(self.read()), initial_size)
        self.assert_code("journal_byte_cap_exceeded", lambda: handle.append("attempt_failed"))

    def test_short_writes_precharge_each_attempt_and_complete(self):
        handle = self.create()
        real_write = os.write
        calls = []

        def short_write(fd, raw):
            calls.append(len(raw))
            self.assertEqual(self.charges[-1], len(raw))
            return real_write(fd, raw[: max(1, len(raw) // 2)])

        before_charges = len(self.charges)
        before_size = len(self.read())
        with mock.patch.object(journal_module.os, "write", side_effect=short_write):
            handle.append("epoch_started", (1,))
        self.assertEqual(self.charges[before_charges:], calls)
        self.assertGreater(sum(calls), len(self.read()) - before_size)
        self.assertEqual(journal_module.verify_bytes(self.read())["records_verified"], 2)

    def test_partial_write_then_error_retains_partial_consumed_artifact(self):
        handle = self.create()
        real_write = os.write
        calls = 0

        def partial_error(fd, raw):
            nonlocal calls
            calls += 1
            if calls == 1:
                return real_write(fd, raw[:5])
            raise OSError("SYNTHETIC-SECRET")

        before = len(self.read())
        with mock.patch.object(journal_module.os, "write", side_effect=partial_error):
            self.assert_code(
                "journal_write_failed", lambda: handle.append("resource_reserved", (7,))
            )
        self.assertEqual(len(self.read()), before + 5)
        self.assert_code("journal_truncated", lambda: journal_module.verify_bytes(self.read()))
        self.assert_code("journal_write_failed", lambda: handle.append("attempt_failed"))
        handle.close()
        self.assert_code("attempt_already_exists", self.create)

    def test_zero_write_fails_sticky_without_refund(self):
        handle = self.create()
        before = len(self.read())
        before_charge = sum(self.charges)
        with mock.patch.object(journal_module.os, "write", return_value=0):
            self.assert_code("journal_write_failed", lambda: handle.append("epoch_started"))
        self.assertEqual(len(self.read()), before)
        self.assertGreater(sum(self.charges), before_charge)
        self.assert_code("journal_write_failed", lambda: handle.append("epoch_completed"))

    def test_debit_failure_prevents_write_and_stays_consumed(self):
        handle = self.create()
        before = self.read()

        def failed_debit(_amount):
            raise RuntimeError("SYNTHETIC-SECRET")

        handle._debit = failed_debit
        with mock.patch.object(
            journal_module.os, "write", side_effect=AssertionError("must not write")
        ) as writer:
            self.assert_code("journal_debit_failed", lambda: handle.append("epoch_started"))
        writer.assert_not_called()
        self.assertEqual(self.read(), before)
        self.assert_code("journal_debit_failed", lambda: handle.append("epoch_completed"))

    def test_non_none_debit_result_fails_closed(self):
        self.assert_code("journal_debit_failed", lambda: self.create(debit=lambda _amount: True))
        self.assertTrue((self.root / journal_module.FILENAME).exists())
        self.assert_code("attempt_already_exists", self.create)

    def test_file_fsync_failure_does_not_commit_or_refund_record(self):
        handle = self.create()
        before = sum(self.charges)
        with mock.patch.object(journal_module.os, "fsync", side_effect=OSError("SYNTHETIC-SECRET")):
            self.assert_code("journal_fsync_failed", lambda: handle.append("step_reserved", (1,)))
        self.assertGreater(sum(self.charges), before)
        self.assertEqual(handle.summary()["records_committed"], 1)
        self.assert_code("journal_fsync_failed", lambda: handle.append("attempt_failed"))
        self.assertEqual(journal_module.verify_bytes(self.read())["records_verified"], 2)

    def test_creation_directory_fsync_failure_leaves_existing_marker(self):
        real_fsync = os.fsync

        def fail_directory(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError("SYNTHETIC-SECRET")
            return real_fsync(fd)

        with mock.patch.object(journal_module.os, "fsync", side_effect=fail_directory):
            self.assert_code("journal_fsync_failed", self.create)
        self.assertTrue((self.root / journal_module.FILENAME).exists())
        self.assert_code("attempt_already_exists", self.create)

    def test_event_and_counter_rejection_are_sticky_without_writes(self):
        invalid = (
            ("SYNTHETIC-SECRET", (), "journal_event_invalid"),
            ("attempt_created", (), "journal_event_invalid"),
            (True, (), "journal_event_invalid"),
            ("epoch_started", [], "journal_counters_invalid"),
            ("epoch_started", (True,), "journal_counters_invalid"),
            ("epoch_started", (-1,), "journal_counters_invalid"),
            ("epoch_started", (2**63,), "journal_counters_invalid"),
            ("epoch_started", (1,) * 17, "journal_counters_invalid"),
            ("epoch_started", ("SYNTHETIC-SECRET",), "journal_counters_invalid"),
        )
        for index, (event, counters, code) in enumerate(invalid):
            directory = self.root / str(index)
            directory.mkdir(mode=0o700)
            handle = journal_module.create_new(
                directory, debit=self.charges.append, maximum_bytes=4096
            )
            self.handles.append(handle)
            before = handle.summary()["file_bytes_written"]
            self.assert_code(
                code,
                lambda handle=handle, event=event, counters=counters: handle.append(
                    event, counters
                ),
            )
            self.assert_code(code, lambda handle=handle: handle.append("attempt_failed"))
            self.assertEqual(handle.summary()["file_bytes_written"], before)

    def test_file_replacement_is_rejected_before_next_charge(self):
        handle = self.create()
        path = self.root / journal_module.FILENAME
        path.rename(self.root / "retained-original")
        path.write_bytes(b"SYNTHETIC-REPLACEMENT")
        path.chmod(0o600)
        before = list(self.charges)
        self.assert_code("journal_identity_changed", lambda: handle.append("epoch_started"))
        self.assertEqual(self.charges, before)
        self.assertEqual(path.read_bytes(), b"SYNTHETIC-REPLACEMENT")

    def test_permission_change_or_hardlink_is_rejected(self):
        handle = self.create()
        path = self.root / journal_module.FILENAME
        os.link(path, self.root / "retained-link")
        self.assert_code("journal_identity_changed", lambda: handle.append("epoch_started"))

    def test_directory_permission_change_is_rejected(self):
        handle = self.create()
        self.root.chmod(0o755)
        self.assert_code("journal_identity_changed", lambda: handle.append("epoch_started"))
        self.root.chmod(0o700)

    def test_close_is_idempotent_retains_file_and_denies_append(self):
        handle = self.create()
        raw = self.read()
        handle.close()
        handle.close()
        self.assertEqual(self.read(), raw)
        self.assertTrue(handle.summary()["closed"])
        self.assert_code("journal_closed", lambda: handle.append("epoch_started"))

    def test_terminal_events_do_not_enable_resume_or_later_append(self):
        handle = self.create()
        handle.append("attempt_failed", (1,))
        self.assertEqual(handle.summary()["terminal_event"], "attempt_failed")
        self.assert_code("journal_closed", lambda: handle.append("epoch_started"))
        self.assertFalse(journal_module.verify_bytes(self.read())["resume_permitted"])

    def test_summary_is_metadata_only_and_defensive_copy(self):
        handle = self.create()
        handle.append("epoch_started", (0, 2))
        summary = handle.summary()
        summary["records_committed"] = 999
        self.assertEqual(handle.summary()["records_committed"], 2)
        raw = json.dumps(handle.summary())
        self.assertNotIn(str(self.root), raw)
        self.assertNotIn("SYNTHETIC-SECRET", raw)
        self.assertFalse(handle.summary()["execution_permitted"])
        self.assertFalse(handle.summary()["resume_permitted"])

    def test_verify_rejects_truncation_and_chain_tampering(self):
        handle = self.create()
        handle.append("epoch_started", (1,))
        raw = self.read()
        for cut in (1, 3, len(raw) - 1):
            self.assert_code(
                "journal_truncated", lambda cut=cut: journal_module.verify_bytes(raw[:cut])
            )
        records = self.decode()
        records[1]["previous_sha256"] = "f" * 64
        changed = b"".join(
            len(journal_module._canonical(record)).to_bytes(4, "big")
            + journal_module._canonical(record)
            for record in records
        )
        self.assert_code("journal_chain_invalid", lambda: journal_module.verify_bytes(changed))

    def test_verify_prefix_is_explicitly_not_complete_or_resumable(self):
        self.create()
        report = journal_module.verify_bytes(self.read())
        self.assertFalse(report["complete_attempt"])
        self.assertFalse(report["resume_permitted"])

    def test_verify_rejects_noncanonical_duplicate_extra_and_bad_counter_fields(self):
        self.create()
        record = self.decode()[0]
        invalid_payloads = []
        invalid_payloads.append((json.dumps(record, indent=2).encode(), "journal_noncanonical"))
        canonical = journal_module._canonical(record)
        invalid_payloads.append((b'{"sequence":1,' + canonical[1:], "journal_noncanonical"))
        for change in (
            {"raw_text": "SYNTHETIC-SECRET"},
            {"counters": [True]},
            {"sequence": True},
            {"counters": [float("nan")]},
            {"event": "SYNTHETIC-SECRET"},
        ):
            changed = {**record, **change}
            invalid_payloads.append(
                (
                    json.dumps(changed, sort_keys=True, separators=(",", ":")).encode(),
                    "journal_record_invalid",
                )
            )
        for payload, code in invalid_payloads:
            frame = len(payload).to_bytes(4, "big") + payload
            self.assert_code(code, lambda frame=frame: journal_module.verify_bytes(frame))

    def test_verify_input_limits_and_invalid_frame_sizes(self):
        for bad in (
            b"",
            "SYNTHETIC-SECRET",
            bytearray(b"x"),
            b"x" * (journal_module.MAXIMUM_BYTES + 1),
        ):
            self.assert_code(
                "journal_input_invalid", lambda bad=bad: journal_module.verify_bytes(bad)
            )
        for size in (0, journal_module.MAXIMUM_FRAME_BYTES + 1):
            self.assert_code(
                "journal_record_invalid",
                lambda size=size: journal_module.verify_bytes(size.to_bytes(4, "big")),
            )

    def test_module_has_no_ml_network_process_or_lumi_imports(self):
        for node in ast.walk(ast.parse(SOURCE.read_text())):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            for name in names:
                self.assertNotIn(
                    name.split(".")[0], {"lumi", "torch", "transformers", "socket", "subprocess"}
                )


if __name__ == "__main__":
    unittest.main()
