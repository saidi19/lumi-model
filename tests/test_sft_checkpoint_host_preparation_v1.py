"""Generated control-file and pure mock checks, never real model/runtime inputs."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_lumi_host_preparation_tests", ROOT / "scripts/prepare_checkpoint_import_host_contract_v1.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lumi-host-control-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.name = "generated-control.json"
        self.raw = b'{"synthetic":true}\n'
        self.file = self.root / self.name
        self.file.write_bytes(self.raw)
        self.binding = len(self.raw), hashlib.sha256(self.raw).hexdigest()
        patcher = patch.object(m, "PINS", {self.name: self.binding})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_exact_generated_control_file(self):
        self.assertEqual(m.read_bound(self.root, self.name, self.binding), self.raw)

    def test_unlisted_relative_rejected(self):
        with self.assertRaises(m.PreparationError):
            m.read_bound(self.root, "unlisted", self.binding)

    def test_wrong_digest_or_length(self):
        for pin in ((len(self.raw), "0" * 64), (len(self.raw) + 1, self.binding[1])):
            with self.subTest(pin=pin), self.assertRaises(m.PreparationError):
                m.read_bound(self.root, self.name, pin)

    def test_symlink_or_hardlink_rejected(self):
        for link_type in ("symlink", "hardlink"):
            with self.subTest(link_type=link_type):
                alias = self.root / "generated-alias"
                if link_type == "symlink":
                    alias.symlink_to(self.file)
                else:
                    os.link(self.file, alias)
                try:
                    with patch.object(m, "PINS", {alias.name: self.binding}), \
                            self.assertRaises(m.PreparationError):
                        m.read_bound(self.root, alias.name, self.binding)
                finally:
                    alias.unlink()

    def test_short_read_is_not_retried_and_descriptor_closed(self):
        with patch.object(m.os, "read", return_value=self.raw[:-1]) as reader, \
                patch.object(m.os, "close", wraps=os.close) as closer, \
                self.assertRaises(m.PreparationError):
            m.read_bound(self.root, self.name, self.binding)
        self.assertEqual(reader.call_count, 1)
        self.assertEqual(closer.call_count, 1)

    def test_pin_failure_precedes_module_execution(self):
        with patch.object(m, "read_bound", side_effect=m.PreparationError), \
                patch("builtins.compile", side_effect=AssertionError) as compiler, \
                self.assertRaises(m.PreparationError):
            m.prepare(self.root)
        compiler.assert_not_called()

    def test_control_read_cap_precedes_reads(self):
        with patch.object(m, "MAXIMUM_TOTAL_CONTROL_READ_BYTES", 1), \
                patch.object(m, "read_bound") as reader, self.assertRaises(m.PreparationError):
            m.prepare(self.root)
        reader.assert_not_called()

    def test_error_output_is_fixed_code_not_exception_text(self):
        output = io.StringIO()
        with patch.object(m.sys, "argv", ["preparer"]), \
                patch.object(m, "prepare", side_effect=ValueError("synthetic-private-detail")), \
                contextlib.redirect_stdout(output):
            self.assertEqual(m.main(), 1)
        self.assertEqual(output.getvalue(),
                         '{"status":"STOP","code":"control_preparation_rejected"}\n')

    def test_arbitrary_argument_is_rejected_before_reads(self):
        with patch.object(m.sys, "argv", ["preparer", "--execute"]), \
                patch.object(m, "prepare") as prepare, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m.main(), 1)
        prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
