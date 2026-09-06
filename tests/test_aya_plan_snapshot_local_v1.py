"""Synthetic runner regressions; no private binding reads or audit-hook installs."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "aya_snapshot_cli_tests", ROOT / "scripts/check_aya_plan_snapshot_local_v1.py"
)
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


class AyaPlanLocalTests(unittest.TestCase):
    def setUp(self):
        self.io = cli.io_primitives()

    def test_exact_source_control_bindings_and_limits_are_preserved(self):
        spec = cli.read_spec()
        plan_binding = spec["bindings"]["plan"]
        self.assertEqual(
            plan_binding["sha256"],
            "01d835d05e38832fa14b97c9817edc1defbb1e0c95549eb777b3f0aea01e5054",
        )
        # The actual source projection checks every original plan binding after
        # authenticating it; this source test only reads the new pinned spec.
        files = [b for b in spec["bindings"].values() if "git_revision" not in b]
        self.assertEqual(len(files), cli.MAX_FILES)
        self.assertEqual(sum(b["bytes"] for b in files), cli.MAX_BYTES)
        self.assertEqual(cli.MAX_SECONDS, spec["limits"]["maximum_runtime_seconds"])
        self.assertEqual(self.io.MAX_FILES, cli.MAX_FILES)
        self.assertEqual(self.io.MAX_BYTES, cli.MAX_BYTES)
        self.assertEqual(spec["nodeid"], cli.NODEID)

    def test_control_read_cap_is_enforced_during_read(self):
        stream = Mock()
        stream.__enter__ = Mock(return_value=stream)
        stream.__exit__ = Mock(return_value=False)
        stream.read.return_value = b"too long"
        path = Mock(open=Mock(return_value=stream))
        with self.assertRaisesRegex(ValueError, "^control_binding_rejected$"):
            cli.control_bytes(path, "0" * 64, 3)
        stream.read.assert_called_once_with(4)

    def test_quiet_output_satisfies_terminal_api_without_retaining_text(self):
        sink = cli.QuietText()
        self.assertFalse(sink.isatty())
        self.assertEqual(sink.encoding, "utf-8")
        self.assertEqual(sink.write("SYNTHETIC_SECRET"), 16)
        self.assertIsNone(sink.flush())
        self.assertEqual(vars(sink), {})

    def test_only_exact_collection_is_accepted(self):
        for items in (
            [],
            [types.SimpleNamespace(nodeid="wrong")],
            [types.SimpleNamespace(nodeid=cli.NODEID)] * 2,
        ):
            with self.subTest(count=len(items)):
                with self.assertRaises(self.io.Stop):
                    cli.OneNode(self.io).pytest_collection_modifyitems(items)
        plugin = cli.OneNode(self.io)
        plugin.pytest_collection_modifyitems([types.SimpleNamespace(nodeid=cli.NODEID)])
        self.assertFalse(plugin.passed())

    def test_skipped_failed_incomplete_or_duplicate_phases_cannot_pass(self):
        for outcome in ("failed", "skipped", "passed"):
            plugin = cli.OneNode(self.io)
            plugin.pytest_collection_modifyitems([types.SimpleNamespace(nodeid=cli.NODEID)])
            for phase in ("setup", "call", "teardown"):
                plugin.pytest_runtest_logreport(
                    types.SimpleNamespace(
                        nodeid=cli.NODEID,
                        when=phase,
                        outcome=outcome,
                        longrepr="SYNTHETIC_SECRET_MUST_NOT_BE_RETAINED",
                    )
                )
            self.assertEqual(plugin.passed(), outcome == "passed")
            self.assertNotIn("SYNTHETIC_SECRET", str(plugin.phases))
            with self.assertRaises(self.io.Stop):
                plugin.pytest_runtest_logreport(
                    types.SimpleNamespace(
                        nodeid=cli.NODEID,
                        when="call",
                        outcome="passed",
                    )
                )

    def test_preexisting_module_is_not_overwritten(self):
        existing = types.ModuleType(cli.MODULE_NAME)
        with patch.dict(sys.modules, {cli.MODULE_NAME: existing}):
            with self.assertRaises(self.io.Stop):
                cli.run_projected_test(self.io, Mock(), cli.synthetic_module())
            self.assertIs(sys.modules[cli.MODULE_NAME], existing)

    def test_projection_is_removed_after_a_pytest_failure(self):
        # Full pytest collection may already have loaded the original module.
        # Exercise the fresh-namespace branch without removing that shared state.
        for preloaded in (False, True):
            with self.subTest(preloaded=preloaded), patch.dict(sys.modules):
                existing = types.ModuleType(cli.MODULE_NAME)
                if preloaded:
                    sys.modules[cli.MODULE_NAME] = existing
                else:
                    sys.modules.pop(cli.MODULE_NAME, None)
                before = sys.modules.get(cli.MODULE_NAME)
                with patch.dict(sys.modules):
                    sys.modules.pop(cli.MODULE_NAME, None)
                    module = cli.synthetic_module()
                    fake_pytest = Mock()
                    fake_pytest.main.return_value = 1
                    with self.assertRaises(self.io.Stop):
                        cli.run_projected_test(self.io, fake_pytest, module)
                    fake_pytest.main.assert_called_once()
                    self.assertNotIn(cli.MODULE_NAME, sys.modules)
                    args = fake_pytest.main.call_args.args[0]
                    for flag in (
                        "--noconftest",
                        "--assert=plain",
                        "no:cacheprovider",
                        "no:faulthandler",
                    ):
                        self.assertIn(flag, args)
                self.assertIs(sys.modules.get(cli.MODULE_NAME), before)

    def test_cleanup_failure_cannot_publish_success_and_deadline_is_cancelled(self):
        for failure in (OSError("SYNTHETIC_SECRET"), self.io.Stop("input_changed"), None):
            inputs = Mock(opens=0, bytes_read=0)
            inputs.close.side_effect = failure
            boundary = Mock(blocked=0, roots=())
            snapshot = Mock(summary=Mock(return_value={"synthetic": True}))
            core = Mock()
            core.build_test_module.return_value = (cli.synthetic_module(), snapshot)
            spec = {"bindings": {"filter_implementation": {}}}
            capture = io.StringIO()
            with (
                patch.object(cli, "io_primitives", return_value=self.io),
                patch.object(cli, "read_spec", return_value=spec),
                patch.object(self.io, "history_bytes", return_value=b"synthetic"),
                patch.object(self.io, "Boundary", return_value=boundary),
                patch.object(self.io, "BoundInputs", return_value=inputs),
                patch.object(cli, "network_denial"),
                patch.object(cli, "warm_runtime", return_value=(Mock(), core)),
                patch.object(cli, "run_projected_test", return_value={"synthetic": True}),
                patch.object(cli.sys, "flags", types.SimpleNamespace(no_site=1, optimize=0)),
                patch.object(cli.sys, "dont_write_bytecode", True),
                patch.object(cli.sys, "addaudithook"),
                patch.object(cli.signal, "signal"),
                patch.object(cli.signal, "alarm") as alarm,
                patch.object(
                    cli.time, "monotonic", side_effect=[0, 1, 31 if failure is None else 2]
                ),
                contextlib.redirect_stdout(capture),
            ):
                result = cli.main(["--verify-original-plan-test"])
            output = json.loads(capture.getvalue())
            self.assertEqual(result, 1)
            self.assertEqual(output["status"], "STOP")
            self.assertFalse(output["original_test_body_executed"])
            self.assertEqual(
                output["failure_code"],
                "deadline_exceeded" if failure is None else "input_cleanup_failed",
            )
            self.assertNotIn("SYNTHETIC_SECRET", capture.getvalue())
            self.assertEqual(alarm.call_args_list[-1].args, (0,))
            inputs.read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
