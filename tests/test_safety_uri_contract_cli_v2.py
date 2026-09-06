"""Synthetic-only checks for the non-replaying safety metadata CLI."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_sft_safety_uri_contract_v2.py"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cli = _module("safety_metadata_cli", SCRIPT)
fixtures = _module("safety_metadata_fixtures", ROOT / "tests/test_safety_uri_verification_v2.py")


def _bytes(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class SafetyUriContractCliV2Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="safety-metadata-synthetic-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.plan, self.report = fixtures.synthetic_inputs()
        self.plan_path = self.directory / "plan.json"
        self.report_path = self.directory / "report.json"
        self._write_inputs()

    def _write_inputs(self):
        self.plan_bytes = _bytes(self.plan)
        self.report["bindings"]["plan_file_sha256"] = _digest(self.plan_bytes)
        fixtures._seal(self.report)
        self.report_bytes = _bytes(self.report)
        self.plan_path.write_bytes(self.plan_bytes)
        self.report_path.write_bytes(self.report_bytes)

    def _arguments(self):
        return [
            "--plan",
            str(self.plan_path),
            "--report",
            str(self.report_path),
            "--expected-plan-sha256",
            _digest(self.plan_bytes),
            "--expected-report-sha256",
            _digest(self.report_bytes),
        ]

    def _run(self, arguments=None, *, isolated=True, cancel_deadline=False):
        before = {path: path.read_bytes() for path in self.directory.glob("*.json")}
        invocation = [str(SCRIPT)]
        if cancel_deadline:
            invocation = [
                "-c",
                "\n".join(
                    [
                        "import importlib.util, sys",
                        "spec = importlib.util.spec_from_file_location('audit_cli', sys.argv[1])",
                        "module = importlib.util.module_from_spec(spec)",
                        "spec.loader.exec_module(module)",
                        "original_alarm = module.signal.alarm",
                        "def injected_alarm(seconds):",
                        "    value = original_alarm(seconds)",
                        "    if seconds == 0:",
                        "        module._deadline(None, None)",
                        "    return value",
                        "module.signal.alarm = injected_alarm",
                        "sys.exit(module.main(sys.argv[2:]))",
                    ]
                ),
                str(SCRIPT),
            ]
        process = subprocess.run(
            [
                sys.executable,
                *(["-S", "-B"] if isolated else []),
                *invocation,
                *(self._arguments() if arguments is None else arguments),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(process.stderr, "")
        result = json.loads(process.stdout)
        self.assertNotIn(str(self.directory), process.stdout)
        self.assertNotIn("SYNTHETIC_RAW_TEXT_SENTINEL", process.stdout)
        self.assertEqual(
            before, {path: path.read_bytes() for path in self.directory.glob("*.json")}
        )
        self.assertEqual(len(list(self.directory.iterdir())), len(before))
        for key in (
            "candidate_rows_replayed",
            "prior_execution_actions_verified",
            "full_private_integration_passed",
            "execution_authorized",
            "production_eligible",
            "network_used",
            "gpu_used",
            "inference_called",
        ):
            self.assertIs(result[key], False)
        self.assertEqual(result["input_files_written"], 0)
        self.assertEqual(result["candidate_row_reads"], 0)
        self.assertEqual(process.returncode, 0 if result["metadata_contract_passed"] else 1)
        return result

    def _read(self, path=None, digest=None):
        accounting = {"metadata_files_opened": 0, "metadata_bytes_read": 0}
        return cli.read_bound(
            path or self.plan_path, digest or _digest(self.plan_bytes), accounting
        )

    def test_success_reads_exactly_two_metadata_files_and_follows_no_embedded_path(self):
        result = self._run()
        self.assertEqual(result["status"], "PASS_METADATA_CONTRACT_ONLY")
        self.assertIs(result["metadata_contract_passed"], True)
        self.assertNotIn("failure_code", result)
        self.assertEqual(result["metadata_files_opened"], 2)
        self.assertEqual(
            result["metadata_bytes_read"], len(self.plan_bytes) + len(self.report_bytes)
        )

    def test_wrong_external_pins_reject_without_internal_seal_fallback(self):
        for position, reads in ((5, 1), (7, 2)):
            with self.subTest(position=position):
                arguments = self._arguments()
                arguments[position] = "0" * 64
                result = self._run(arguments)
                self.assertEqual(result["failure_code"], "file_hash_mismatch")
                self.assertEqual(result["metadata_files_opened"], reads)

    def test_bad_hash_missing_extra_duplicate_and_abbreviated_arguments_read_nothing(self):
        original = self._arguments()
        bad_pin = original.copy()
        bad_pin[5] = "SYNTHETIC_RAW_TEXT_SENTINEL"
        duplicate = original.copy()
        duplicate[2] = "--plan"
        abbreviated = original.copy()
        abbreviated[0] = "--pla"
        for arguments in (
            bad_pin,
            [],
            original[:-2],
            original + ["--execute"],
            duplicate,
            abbreviated,
        ):
            with self.subTest(arguments=arguments):
                result = self._run(arguments)
                self.assertIn(result["failure_code"], {"pin_rejected", "arguments_rejected"})
                self.assertEqual(result["metadata_files_opened"], 0)
                self.assertEqual(result["metadata_bytes_read"], 0)

    def test_nonisolated_interpreter_rejected_before_metadata_open(self):
        result = self._run(isolated=False)
        self.assertEqual(result["failure_code"], "runtime_rejected")
        self.assertEqual(result["metadata_files_opened"], 0)

    def test_same_input_path_rejected_before_metadata_open(self):
        arguments = self._arguments()
        arguments[3] = arguments[1]
        result = self._run(arguments)
        self.assertEqual(result["failure_code"], "path_rejected")
        self.assertEqual(result["metadata_files_opened"], 0)

    def test_resealed_token_and_authorization_drift_rejected_end_to_end(self):
        for mutation in ("tokens", "authorization"):
            with self.subTest(mutation=mutation):
                self.plan, self.report = fixtures.synthetic_inputs()
                if mutation == "tokens":
                    self.report["frozen_token_verification"]["by_split"]["train"][
                        "supervised_tokens"
                    ] = 0
                else:
                    self.report["authorization"]["network_access_authorized"] = True
                self._write_inputs()
                self.assertEqual(self._run()["failure_code"], "metadata_contract_rejected")

    def test_unknown_raw_text_field_never_echoed(self):
        self.report["messages"] = "SYNTHETIC_RAW_TEXT_SENTINEL"
        self._write_inputs()
        self.assertEqual(self._run()["failure_code"], "metadata_contract_rejected")

    def test_strict_json_rejects_duplicates_constants_nonobjects_and_invalid_encoding(self):
        for payload in (
            b'{"a":1,"a":2}',
            b'{"a":{"b":1,"b":2}}',
            b'{"a":NaN}',
            b'{"a":Infinity}',
            b"[]",
            b"null",
            b'"raw"',
            b"{",
            b"\xff",
        ):
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(cli.AuditError, "^json_rejected$"):
                    cli.strict_object(payload)

    def test_bad_json_cli_has_no_decoder_or_input_text(self):
        self.report_bytes = b'{"SYNTHETIC_RAW_TEXT_SENTINEL":NaN}'
        self.report_path.write_bytes(self.report_bytes)
        self.assertEqual(self._run()["failure_code"], "json_rejected")

    def test_relative_parent_and_non_json_paths_rejected(self):
        for path in (
            Path("relative.json"),
            self.directory / ".." / "plan.json",
            self.directory / "rows.jsonl",
        ):
            with self.subTest(path=path):
                with self.assertRaisesRegex(cli.AuditError, "^path_rejected$"):
                    self._read(path)

    def test_symlink_file_and_ancestor_and_hardlink_rejected(self):
        symlink = self.directory / "link.json"
        symlink.symlink_to(self.plan_path)
        ancestor = self.directory / "linked"
        ancestor.symlink_to(self.directory, target_is_directory=True)
        for path in (symlink, ancestor / "plan.json"):
            with self.subTest(path=path):
                with self.assertRaisesRegex(cli.AuditError, "^file_rejected$"):
                    self._read(path)
        hardlink = self.directory / "hard.json"
        os.link(self.plan_path, hardlink)
        with self.assertRaisesRegex(cli.AuditError, "^file_rejected$"):
            self._read(hardlink)

    def test_empty_oversize_directory_and_fifo_rejected_before_reads(self):
        empty = self.directory / "empty.json"
        empty.touch()
        large = self.directory / "large.json"
        large.write_bytes(b"x" * (cli.MAXIMUM_FILE_BYTES + 1))
        directory = self.directory / "directory.json"
        directory.mkdir()
        fifo = self.directory / "fifo.json"
        os.mkfifo(fifo)
        for path in (empty, large, directory, fifo):
            with self.subTest(path=path):
                accounting = {"metadata_files_opened": 0, "metadata_bytes_read": 0}
                with self.assertRaisesRegex(cli.AuditError, "^file_rejected$"):
                    cli.read_bound(path, "0" * 64, accounting)
                self.assertEqual(accounting["metadata_bytes_read"], 0)

    def test_size_drift_rejected_and_reads_never_exceed_original_size(self):
        original_read = os.read
        mutated = False

        def change_during_read(fd, count):
            nonlocal mutated
            if not mutated:
                mutated = True
                with self.plan_path.open("ab") as stream:
                    stream.write(b" ")
            return original_read(fd, count)

        accounting = {"metadata_files_opened": 0, "metadata_bytes_read": 0}
        with patch.object(cli.os, "read", side_effect=change_during_read):
            with self.assertRaisesRegex(cli.AuditError, "^file_changed$"):
                cli.read_bound(self.plan_path, _digest(self.plan_bytes), accounting)
        self.assertLessEqual(accounting["metadata_bytes_read"], len(self.plan_bytes))

    def test_deadline_and_os_errors_are_fixed_codes(self):
        with self.assertRaisesRegex(cli.AuditError, "^deadline_exceeded$"):
            cli._deadline(None, None)
        with patch.object(cli.os, "open", side_effect=OSError("SYNTHETIC_RAW_TEXT_SENTINEL")):
            with self.assertRaisesRegex(cli.AuditError, "^file_rejected$"):
                self._read()
        with patch.object(cli.json, "loads", side_effect=cli.AuditError("deadline_exceeded")):
            with self.assertRaisesRegex(cli.AuditError, "^deadline_exceeded$"):
                cli.strict_object(b"{}")

    def test_deadline_after_valid_result_cannot_publish_success(self):
        result = self._run(cancel_deadline=True)
        self.assertEqual(result["status"], "STOP")
        self.assertIs(result["metadata_contract_passed"], False)
        self.assertEqual(result["failure_code"], "deadline_exceeded")
        self.assertEqual(result["metadata_files_opened"], 2)


if __name__ == "__main__":
    unittest.main()
