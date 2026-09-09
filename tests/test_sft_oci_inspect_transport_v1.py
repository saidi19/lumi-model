"""Inert CLI/pipe doubles only: no Docker, native fixtures, network or ML."""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_lumi_synthetic_oci_transport_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
before = set(sys.modules)
admission = importlib.import_module(PACKAGE + ".posttraining_sft_oci_admission_v1")
ADDITIONS = set(sys.modules) - before


class InspectTransportTests(unittest.TestCase):
    def setUp(self):
        self.stream = types.SimpleNamespace(fileno=Mock(return_value=42), close=Mock())
        self.process = types.SimpleNamespace(stdout=self.stream, poll=Mock(return_value=0),
                                             wait=Mock(return_value=0), kill=Mock())
        self.spawn = Mock(return_value=self.process)
        self.directory = Mock(return_value=nullcontext("/synthetic/empty-cli-config"))
        self.clock = Mock(return_value=100.0)
        self.read = Mock(side_effect=[b"[]", b""])
        self.select = Mock(return_value=([self.stream], [], []))
        for target, name, value in (
            (admission.subprocess, "Popen", self.spawn),
            (admission.tempfile, "TemporaryDirectory", self.directory),
            (admission.time, "monotonic", self.clock),
            (admission.os, "read", self.read),
            (admission.select, "select", self.select),
        ):
            selected = patch.object(target, name, value)
            selected.start()
            self.addCleanup(selected.stop)

    def inspect(self, kind="image", identifier=admission.IMAGE_ID, deadline=104.0):
        return admission._inspect(kind, identifier, deadline)

    def test_only_local_read_only_command_with_no_inherited_docker_context(self):
        self.assertEqual(self.inspect(), b"[]")
        arguments, keywords = self.spawn.call_args
        self.assertEqual(arguments, ([
            "/usr/bin/docker", "--host", "unix:///var/run/docker.sock", "--config",
            "/synthetic/empty-cli-config", "image", "inspect", admission.IMAGE_ID,
        ],))
        self.assertEqual(keywords, {
            "stdin": admission.subprocess.DEVNULL, "stdout": admission.subprocess.PIPE,
            "stderr": admission.subprocess.DEVNULL, "close_fds": True,
            "env": {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        })
        self.directory.assert_called_once_with(prefix="lumi-oci-inspect-")
        self.process.wait.assert_called_once_with(timeout=4.0)
        self.process.kill.assert_not_called()
        self.stream.close.assert_called_once_with()

    def test_container_requires_full_id_no_tag_or_cli_injection(self):
        for identifier in ("short", "--help", "a" * 63, "A" * 64, "a" * 64 + "\n", None):
            with self.subTest(kind=type(identifier).__name__):
                with self.assertRaises(admission.OciAdmissionError):
                    self.inspect("container", identifier)
        self.spawn.assert_not_called()
        self.directory.assert_not_called()
        self.assertEqual(self.inspect("container", "a" * 64), b"[]")
        self.assertEqual(self.spawn.call_args.args[0][-3:], ["container", "inspect", "a" * 64])

    def test_mutating_commands_and_unpinned_images_are_not_expressible(self):
        for kind, identifier in (("start", "a" * 64), ("pull", admission.IMAGE_ID),
                                 ("export", "a" * 64), ("image", "sha256:" + "0" * 64)):
            with self.subTest(kind=kind):
                with self.assertRaises(admission.OciAdmissionError):
                    self.inspect(kind, identifier)
        self.spawn.assert_not_called()

    def test_expired_deadline_precedes_config_directory_and_child_creation(self):
        with self.assertRaisesRegex(admission.OciAdmissionError, "deadline_exhausted"):
            self.inspect(deadline=99.0)
        self.directory.assert_not_called()
        self.spawn.assert_not_called()

    def test_per_command_deadline_is_clamped_to_five_seconds(self):
        self.inspect(deadline=115.0)
        self.process.wait.assert_called_once_with(timeout=5.0)
        self.assertTrue(all(call.args[3] == 5.0 for call in self.select.call_args_list))

    def test_exact_output_cap_succeeds_without_larger_read_capacity(self):
        self.read.side_effect = [b"x" * 8192] * 8 + [b""]
        self.assertEqual(len(self.inspect()), admission.MAXIMUM_INSPECT_BYTES)
        capacities = [call.args[1] for call in self.read.call_args_list]
        self.assertEqual(capacities, [8192] * 8 + [1])

    def test_overflow_kills_only_owned_cli_once_and_closes_pipe(self):
        self.read.side_effect = [b"x" * 8192] * 8 + [b"x"]
        self.process.poll.return_value = None
        with self.assertRaisesRegex(admission.OciAdmissionError, "inspect_budget_exhausted"):
            self.inspect()
        self.spawn.assert_called_once()
        self.process.kill.assert_called_once_with()
        self.process.wait.assert_called_once_with(timeout=1)
        self.stream.close.assert_called_once_with()

    def test_timeout_kills_and_reaps_owned_cli_without_retry(self):
        self.select.return_value = ([], [], [])
        self.process.poll.return_value = None
        with self.assertRaisesRegex(admission.OciAdmissionError, "deadline_exhausted"):
            self.inspect()
        self.process.kill.assert_called_once_with()
        self.process.wait.assert_called_once_with(timeout=1)
        self.spawn.assert_called_once()
        self.read.assert_not_called()
        self.stream.close.assert_called_once_with()

    def test_absolute_deadline_is_not_reset_by_a_partial_chunk(self):
        self.clock.side_effect = [100.0, 100.0, 100.0, 106.0]
        self.process.poll.return_value = None
        with self.assertRaisesRegex(admission.OciAdmissionError, "deadline_exhausted"):
            self.inspect(deadline=115.0)
        self.read.assert_called_once()
        self.process.kill.assert_called_once()
        self.stream.close.assert_called_once()

    def test_nonzero_cli_exit_is_not_success_or_a_retry(self):
        self.process.wait.return_value = 1
        self.process.poll.return_value = 1
        with self.assertRaisesRegex(admission.OciAdmissionError, "inspect_failed"):
            self.inspect()
        self.spawn.assert_called_once()
        self.process.kill.assert_not_called()
        self.stream.close.assert_called_once()

    def test_exit_racing_with_kill_is_reaped_without_resending_kill(self):
        self.select.return_value = ([], [], [])
        self.process.poll.return_value = None
        self.process.kill.side_effect = ProcessLookupError("synthetic race")
        with self.assertRaisesRegex(admission.OciAdmissionError, "deadline_exhausted"):
            self.inspect()
        self.process.kill.assert_called_once()
        self.process.wait.assert_called_once_with(timeout=1)
        self.stream.close.assert_called_once()

    def test_pipe_closed_even_if_owned_process_cleanup_fails(self):
        self.select.return_value = ([], [], [])
        self.process.poll.return_value = None
        self.process.kill.side_effect = RuntimeError("synthetic cleanup failure")
        with self.assertRaises(RuntimeError):
            self.inspect()
        self.process.kill.assert_called_once()
        self.stream.close.assert_called_once()

    def test_this_import_does_not_import_ml_or_any_native_fixture(self):
        roots = {name.split(".", 1)[0] for name in ADDITIONS}
        self.assertFalse(roots & {"torch", "transformers", "numpy", "cuda", "cupy"})
        self.assertFalse(any("fixture" in name for name in ADDITIONS))


if __name__ == "__main__":
    unittest.main()
