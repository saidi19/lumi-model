"""In-memory socket doubles only: no native sockets, Linux fixture or ML."""

import errno
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

PACKAGE = "_synthetic_ipc_pair_probe_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1] / "src/lumi")]
sys.modules[PACKAGE] = package
BEFORE_IMPORT = set(sys.modules)
probe = importlib.import_module(PACKAGE + ".posttraining_sft_ipc_pair_probe_v1")
IMPORT_ADDITIONS = set(sys.modules) - BEFORE_IMPORT


class PairProbeTests(unittest.TestCase):
    def selected(self, listen_error=None):
        selected = Mock()
        selected.getsockopt.return_value = 0
        selected.listen.return_value = None
        selected.listen.side_effect = listen_error
        selected.gettimeout.return_value = 1.0
        selected.accept.side_effect = OSError(errno.EINVAL, "PRIVATE")
        return selected

    def failure(self, selected, code):
        with self.assertRaises(probe.PairProbeError) as caught:
            probe.verify_no_listener(selected)
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(str(caught.exception), code)
        self.assertNotIn("PRIVATE", str(caught.exception))

    def test_old_kernel_rejection_requires_state_and_accept_checks(self):
        for number in (errno.EINVAL, errno.EOPNOTSUPP):
            selected = self.selected(OSError(number, "PRIVATE"))
            self.assertEqual(probe.verify_no_listener(selected), {
                "not_listening": True, "accept_rejected": True, "listen_call_rejected": True})
            self.assertEqual(selected.getsockopt.call_count, 3)
            selected.accept.assert_called_once_with()
            self.assertEqual([call.args for call in selected.settimeout.call_args_list],
                             [(0,), (1.0,)])

    def test_new_kernel_noop_success_requires_same_state_and_accept_rejection(self):
        selected = self.selected()
        self.assertEqual(probe.verify_no_listener(selected), {
            "not_listening": True, "accept_rejected": True, "listen_call_rejected": False})
        self.assertEqual(selected.getsockopt.call_count, 3)
        selected.listen.assert_called_once_with(1)
        selected.accept.assert_called_once_with()

    def test_actual_listener_rejected_before_attempt(self):
        selected = self.selected()
        selected.getsockopt.return_value = 1
        self.failure(selected, "listener_transition_detected")
        selected.listen.assert_not_called()

    def test_successful_listen_transition_is_never_accepted(self):
        selected = self.selected()
        selected.getsockopt.side_effect = [0, 1]
        self.failure(selected, "listener_transition_detected")
        selected.accept.assert_not_called()

    def test_errno_alone_cannot_mask_listener_transition(self):
        selected = self.selected(OSError(errno.EINVAL, "PRIVATE"))
        selected.getsockopt.side_effect = [0, 1]
        self.failure(selected, "listener_transition_detected")

    def test_accept_phase_state_transition_is_rejected(self):
        selected = self.selected()
        selected.getsockopt.side_effect = [0, 0, 1]
        self.failure(selected, "listener_transition_detected")
        selected.settimeout.assert_called_with(1.0)

    def test_malformed_state_is_not_false_equivalent(self):
        for value in (False, True, None, "0", -1, 2, []):
            selected = self.selected()
            selected.getsockopt.return_value = value
            self.failure(selected, "listener_state_invalid")

    def test_listen_must_have_exact_none_result_on_success(self):
        selected = self.selected()
        selected.listen.return_value = 0
        self.failure(selected, "listen_result_invalid")

    def test_unexpected_listen_errno_remains_failure(self):
        selected = self.selected(OSError(errno.EACCES, "PRIVATE"))
        self.failure(selected, "listen_errno_unexpected")

    def test_nonblocking_accept_again_is_not_a_denial_proof(self):
        for number in (errno.EAGAIN, errno.ETIMEDOUT, errno.EACCES):
            selected = self.selected()
            selected.accept.side_effect = OSError(number, "PRIVATE")
            self.failure(selected, "accept_errno_unexpected")
            selected.settimeout.assert_called_with(1.0)

    def test_accept_success_fails_and_closes_only_returned_connection(self):
        selected, connection = self.selected(), Mock()
        selected.accept.side_effect = None
        selected.accept.return_value = (connection, "PRIVATE")
        self.failure(selected, "accept_succeeded")
        connection.close.assert_called_once_with()
        selected.close.assert_not_called()
        selected.settimeout.assert_called_with(1.0)

    def test_accept_close_failure_is_still_failure_without_close_retry(self):
        selected, connection = self.selected(), Mock()
        selected.accept.side_effect = None
        selected.accept.return_value = (connection, "PRIVATE")
        connection.close.side_effect = OSError("PRIVATE")
        self.failure(selected, "accept_succeeded")
        connection.close.assert_called_once_with()

    def test_readback_error_is_fixed_and_fail_closed(self):
        selected = self.selected()
        selected.getsockopt.side_effect = OSError("PRIVATE")
        self.failure(selected, "socket_call_failed")

    def test_timeout_restore_failure_is_not_success(self):
        selected = self.selected()
        selected.settimeout.side_effect = [None, OSError("PRIVATE")]
        self.failure(selected, "timeout_restore_failed")
        self.assertEqual(selected.settimeout.call_count, 2)

    def test_known_error_vocabulary_and_no_ml_imports(self):
        self.assertEqual(len(probe.ERROR_CODES), 8)
        self.assertEqual(probe.PairProbeError("PRIVATE").code, "socket_call_failed")
        self.assertFalse({"torch", "numpy", "transformers"}
                         & {name.split(".")[0] for name in IMPORT_ADDITIONS})


if __name__ == "__main__":
    unittest.main()
