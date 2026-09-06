"""Synthetic lifecycle failures: no real inputs, subprocesses, or audit hooks."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from lumi import posttraining_safety_output_replay_v1 as core

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "safety_output_lifecycle_cli", ROOT / "scripts/check_safety_output_replay_local_v1.py"
)
cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cli)


class SafetyOutputReplayLifecycleTests(unittest.TestCase):
    def _namespace_collision(self, name: str) -> None:
        existing = types.ModuleType(name)
        marker = object()
        existing.marker = marker
        payload = b"synthetic source snapshot"
        bindings = {
            "synthetic": {
                "path": "synthetic.py",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        }
        with mock.patch.dict(sys.modules, {name: existing}):
            with mock.patch.object(core, "load_historical_modules") as load:
                with self.assertRaisesRegex(ValueError, "^invalid safety output replay$"):
                    core.reproduce({"synthetic": payload}, bindings, "/synthetic/repo")
                self.assertIs(sys.modules.get(name), existing)
                self.assertIs(sys.modules[name].marker, marker)
                load.assert_not_called()

    def test_existing_historical_package_is_preserved_on_collision(self) -> None:
        self._namespace_collision(core.PACKAGE)

    def test_existing_historical_child_namespace_is_preserved_on_collision(self) -> None:
        self._namespace_collision(core.PACKAGE + ".synthetic_existing_child")

    def test_fstat_failure_retains_every_open_descriptor_for_cleanup(self) -> None:
        leaf_open_number = len((cli.ROOT / "synthetic.json").parts)
        for fail_at in (1, 2, leaf_open_number):
            with self.subTest(fail_at=fail_at):
                boundary = types.SimpleNamespace(bound_open=False)
                inputs = cli.BoundInputs(boundary)
                descriptors = list(range(100, 100 + fail_at))
                stats = [object()] * (fail_at - 1) + [OSError("SYNTHETIC_SENTINEL")]
                with (
                    mock.patch.object(cli.os, "open", side_effect=descriptors) as opening,
                    mock.patch.object(cli.os, "fstat", side_effect=stats),
                    mock.patch.object(
                        cli.os, "read", side_effect=AssertionError("unexpected read")
                    ),
                    mock.patch.object(
                        cli.os, "stat", side_effect=AssertionError("unexpected stat")
                    ),
                    mock.patch.object(cli.os, "close") as closing,
                ):
                    with self.assertRaisesRegex(cli.Stop, "^input_rejected$"):
                        inputs.read("synthetic.json", {"bytes": 1, "sha256": "0" * 64})
                    self.assertFalse(boundary.bound_open)
                    inputs.close()
                    self.assertEqual(opening.call_count, fail_at)
                    self.assertEqual(
                        closing.call_args_list, [mock.call(fd) for fd in reversed(descriptors)]
                    )
                    self.assertEqual(inputs.bytes_read, 0)
                    self.assertEqual(inputs.descriptors, [])

    def test_close_failure_stops_success_and_still_cancels_deadline(self) -> None:
        for failure, code in (
            (cli.Stop("input_changed"), "input_changed"),
            (OSError("SYNTHETIC_SENTINEL"), "audit_failed"),
        ):
            with self.subTest(code=code):
                inputs = mock.Mock(opens=0, bytes_read=0)
                inputs.close.side_effect = failure
                inputs.read.side_effect = AssertionError("unexpected input read")
                boundary = mock.Mock(blocked=0)
                runtime = types.SimpleNamespace(
                    STAGES=core.STAGES,
                    reproduce=mock.Mock(
                        return_value={
                            "status": "PASS_SAFETY_OUTPUT_REPRODUCTION_ONLY",
                            "safety_outputs_reproduced": True,
                        }
                    ),
                )
                spec = {"bindings": {"posttraining_data_implementation": {"synthetic": True}}}
                capture = io.StringIO()
                with (
                    mock.patch.object(cli, "read_spec", return_value=spec),
                    mock.patch.object(cli, "history_bytes", return_value=b"synthetic source"),
                    mock.patch.object(cli, "warm_filelock_probe"),
                    mock.patch.object(cli, "runtime", return_value=runtime),
                    mock.patch.object(cli, "Boundary", return_value=boundary),
                    mock.patch.object(cli, "BoundInputs", return_value=inputs),
                    mock.patch.object(
                        cli.sys, "flags", types.SimpleNamespace(no_site=1, optimize=0)
                    ),
                    mock.patch.object(cli.sys, "dont_write_bytecode", True),
                    mock.patch.object(cli.sys, "addaudithook") as audit_hook,
                    mock.patch.object(cli.signal, "signal"),
                    mock.patch.object(cli.signal, "alarm") as alarm,
                    mock.patch.object(cli.time, "monotonic", side_effect=[0, 1]),
                    mock.patch.object(cli.subprocess, "run", side_effect=AssertionError("process")),
                    mock.patch("builtins.open", side_effect=AssertionError("unexpected file read")),
                    mock.patch("io.open", side_effect=AssertionError("unexpected file read")),
                    mock.patch("os.open", side_effect=AssertionError("unexpected file read")),
                    contextlib.redirect_stdout(capture),
                ):
                    status = cli.main(["--reproduce-existing-outputs"])
                result = json.loads(capture.getvalue())
                self.assertEqual(status, 1)
                self.assertEqual(result["status"], "STOP")
                self.assertFalse(result["safety_outputs_reproduced"])
                self.assertEqual(result["failure_code"], code)
                self.assertNotIn("SYNTHETIC_SENTINEL", capture.getvalue())
                self.assertEqual(alarm.call_args_list, [mock.call(cli.MAX_SECONDS), mock.call(0)])
                inputs.close.assert_called_once_with()
                inputs.read.assert_not_called()
                runtime.reproduce.assert_called_once()
                # The attempted registration hit the mock, never sys.addaudithook.
                audit_hook.assert_called_once_with(boundary.guard)


if __name__ == "__main__":
    unittest.main()
