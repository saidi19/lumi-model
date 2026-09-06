"""Synthetic file-I/O checks; never open the real private candidate pair."""

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
SCRIPT = ROOT / "scripts/check_safety_uri_rows_local_v1.py"


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


cli = _module("safety_rows_cli", SCRIPT)


class SafetyUriRowsLocalV1Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="safety-row-synthetic-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.path = self.directory / "sample.jsonl"
        self.path.write_bytes(b"synthetic bytes\n")
        self.payload = self.path.read_bytes()
        self.digest = hashlib.sha256(self.payload).hexdigest()
        self.accounting = {"candidate_files_opened": 0, "candidate_bytes_read": 0}

    def _bound(self, path=None, size=None, digest=None):
        return cli.bound_file(
            path or self.path,
            len(self.payload) if size is None else size,
            digest or self.digest,
            self.accounting,
        )

    def test_pins_match_hash_bound_tracked_plans(self):
        for index, name, expected, key in (
            (
                0,
                "configs/lumi-4b-sft-safety-uri-repair-v0.1.json",
                "47c1cfda1fd884372b8c1aaf482c015e52c18d76368b679bd43d3fac6cdf6fb0",
                "predecessor_candidate",
            ),
            (
                1,
                "configs/lumi-4b-sft-candidate-mix-reaudit-v0.2.json",
                "61db271128469393f086b3d41be22962fcc1cc5cf0a895713f984eeb8fcaa047",
                "safety_candidate",
            ),
        ):
            with self.subTest(index=index):
                payload = (ROOT / name).read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), expected)
                binding = json.loads(payload)["bindings"][key]
                self.assertEqual(
                    (cli.BINDINGS[index][0], cli.BINDINGS[index][2]),
                    (binding["path"], binding["sha256"]),
                )
        self.assertEqual(sum(item[1] for item in cli.BINDINGS), cli.MAXIMUM_TOTAL_BYTES)
        self.assertEqual(cli.MAXIMUM_TOTAL_BYTES, 1_306_432)
        self.assertEqual(cli.MAXIMUM_SECONDS, 30)

    def test_reader_authenticates_one_open_and_exact_bytes(self):
        with self._bound() as payload:
            self.assertEqual(payload, self.payload)
        self.assertEqual(
            self.accounting,
            {"candidate_files_opened": 1, "candidate_bytes_read": len(self.payload)},
        )

    def test_bad_hash_and_size_are_fixed_failures(self):
        for kwargs, code in (
            ({"digest": "0" * 64}, "hash_mismatch"),
            ({"size": len(self.payload) + 1}, "file_rejected"),
        ):
            self.accounting = {"candidate_files_opened": 0, "candidate_bytes_read": 0}
            with self.subTest(code=code):
                with self.assertRaisesRegex(cli.Stop, "^" + code + "$"):
                    with self._bound(**kwargs):
                        self.fail("must reject")

    def test_symlink_hardlink_fifo_and_directory_rejected_without_reads(self):
        link = self.directory / "link.jsonl"
        link.symlink_to(self.path)
        fifo = self.directory / "fifo.jsonl"
        os.mkfifo(fifo)
        directory = self.directory / "directory.jsonl"
        directory.mkdir()
        hardlink = self.directory / "hard.jsonl"
        os.link(self.path, hardlink)
        for path in (link, fifo, directory, hardlink):
            with self.subTest(path=path):
                self.accounting = {"candidate_files_opened": 0, "candidate_bytes_read": 0}
                with self.assertRaisesRegex(cli.Stop, "^file_rejected$"):
                    with self._bound(path=path):
                        self.fail("must reject")
                self.assertEqual(self.accounting["candidate_bytes_read"], 0)

    def test_file_mutation_during_comparison_rejected(self):
        with self.assertRaisesRegex(cli.Stop, "^file_changed$"):
            with self._bound():
                self.path.write_bytes(b"changed synthetic data\n")

    def test_ancestor_swap_during_comparison_rejected(self):
        parent = self.directory / "parent"
        parent.mkdir()
        path = parent / "sample.jsonl"
        path.write_bytes(self.payload)
        with self.assertRaisesRegex(cli.Stop, "^file_changed$"):
            with self._bound(path=path):
                parent.rename(self.directory / "moved")
                parent.mkdir()
                path.write_bytes(self.payload)

    def test_file_and_byte_caps_prevent_additional_reads(self):
        self.accounting["candidate_files_opened"] = 2
        with self.assertRaisesRegex(cli.Stop, "^read_budget$"):
            with self._bound():
                self.fail("must reject")
        self.accounting = {
            "candidate_files_opened": 0,
            "candidate_bytes_read": cli.MAXIMUM_TOTAL_BYTES,
        }
        with patch.object(cli.os, "read", side_effect=AssertionError("unexpected read")):
            with self.assertRaisesRegex(cli.Stop, "^read_budget$"):
                with self._bound():
                    self.fail("must reject")

    def _run_synthetic(self, *, bad_hash=False, late_deadline=False, bad_row=False):
        fixtures = _module("safety_delta_fixtures", ROOT / "tests/test_safety_uri_row_delta_v1.py")
        payloads = fixtures.synthetic_pairs()
        if bad_row:
            payloads = (payloads[0], payloads[0])
        paths = (self.directory / "before.jsonl", self.directory / "after.jsonl")
        bindings = []
        for path, payload in zip(paths, payloads, strict=True):
            path.write_bytes(payload)
            bindings.append((str(path), len(payload), hashlib.sha256(payload).hexdigest()))
        if bad_hash:
            bindings[0] = (*bindings[0][:2], "0" * 64)
        bootstrap = "\n".join(
            [
                "import importlib.util, json, sys",
                "spec = importlib.util.spec_from_file_location('safety_rows', sys.argv[1])",
                "module = importlib.util.module_from_spec(spec)",
                "spec.loader.exec_module(module)",
                "module.BINDINGS = json.loads(sys.argv[2])",
                "if sys.argv[3] == 'late':",
                "    original = module.signal.alarm",
                "    def alarm(seconds):",
                "        result = original(seconds)",
                "        if seconds == 0: module.deadline(None, None)",
                "        return result",
                "    module.signal.alarm = alarm",
                "sys.exit(module.main(['--audit-existing-local-pair']))",
            ]
        )
        process = subprocess.run(
            [
                sys.executable,
                "-S",
                "-B",
                "-c",
                bootstrap,
                str(SCRIPT),
                json.dumps(bindings),
                "late" if late_deadline else "normal",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(process.stderr, "")
        self.assertNotIn(str(self.directory), process.stdout)
        self.assertNotIn("SYNTHETIC", process.stdout)
        self.assertEqual(tuple(path.read_bytes() for path in paths), payloads)
        result = json.loads(process.stdout)
        self.assertEqual(process.returncode, 0 if result["row_delta_verified"] else 1)
        for key in (
            "tokenizer_recomputed",
            "generic_filtering_replayed",
            "full_private_integration_passed",
            "execution_authorized",
            "production_eligible",
            "network_used",
            "gpu_used",
            "inference_called",
        ):
            self.assertIs(result[key], False)
        self.assertEqual(result["input_files_written"], 0)
        return result

    def test_complete_cli_with_synthetic_files(self):
        result = self._run_synthetic()
        self.assertTrue(result["row_delta_verified"])
        self.assertTrue(result["input_hashes_authenticated"])
        self.assertTrue(result["input_files_preserved"])
        self.assertEqual(result["candidate_files_opened"], 2)
        self.assertEqual(result["rows_compared"], 320)

    def test_hash_and_row_rejections_never_publish_success(self):
        for kwargs, code in (
            ({"bad_hash": True}, "hash_mismatch"),
            ({"bad_row": True}, "row_delta_rejected"),
        ):
            with self.subTest(code=code):
                result = self._run_synthetic(**kwargs)
                self.assertEqual(result["failure_code"], code)
                self.assertEqual(result["status"], "STOP")

    def test_late_deadline_never_publishes_success(self):
        result = self._run_synthetic(late_deadline=True)
        self.assertFalse(result["row_delta_verified"])
        self.assertEqual(result["failure_code"], "deadline_exceeded")

    def test_missing_flag_reads_no_private_files(self):
        process = subprocess.run(
            [sys.executable, "-S", "-B", str(SCRIPT)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(process.returncode, 1)
        self.assertEqual(process.stderr, "")
        result = json.loads(process.stdout)
        self.assertEqual(result["failure_code"], "arguments_rejected")
        self.assertEqual(result["candidate_files_opened"], 0)
        self.assertEqual(result["candidate_bytes_read"], 0)


if __name__ == "__main__":
    unittest.main()
