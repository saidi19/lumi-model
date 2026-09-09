"""State-based connected-pair listener probe; native calls only on explicit use.

Linux v6.17 unix_listen can return zero for an established bound pair without
changing sk_state. Never equate that return with listener authority: require
SO_ACCEPTCONN=0 and a nonblocking accept rejection. No policy installation here.
"""

from __future__ import annotations

import errno
import socket

ERROR_CODES = frozenset({
    "listener_state_invalid", "listener_transition_detected", "listen_result_invalid",
    "listen_errno_unexpected", "accept_errno_unexpected", "accept_succeeded",
    "socket_call_failed", "timeout_restore_failed",
})


class PairProbeError(RuntimeError):
    def __init__(self, code="socket_call_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "socket_call_failed"
        super().__init__(self.code)


def _not_listening(selected):
    value = selected.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN)
    if type(value) is not int or value not in (0, 1):
        raise PairProbeError("listener_state_invalid") from None
    if value != 0:
        raise PairProbeError("listener_transition_detected") from None


def verify_no_listener(selected):
    """One attempt, fixed result/error only; never accept a listener or timeout."""
    failure, rejected, restore, previous = None, False, False, None
    try:
        _not_listening(selected)
        try:
            result = selected.listen(1)
        except OSError as error:
            if error.errno not in (errno.EINVAL, errno.EOPNOTSUPP):
                raise PairProbeError("listen_errno_unexpected") from None
            rejected = True
        else:
            if result is not None:
                raise PairProbeError("listen_result_invalid") from None
        _not_listening(selected)
        previous = selected.gettimeout()
        restore = True
        selected.settimeout(0)
        try:
            connection, _ = selected.accept()
        except OSError as error:
            if error.errno not in (errno.EINVAL, errno.EOPNOTSUPP):
                raise PairProbeError("accept_errno_unexpected") from None
        else:
            # Unexpected owned descriptor is closed once; never use peer data.
            try:
                connection.close()
            finally:
                raise PairProbeError("accept_succeeded") from None
        _not_listening(selected)
    except BaseException as error:
        failure = error.code if type(error) is PairProbeError else "socket_call_failed"
    finally:
        if restore:
            try:
                selected.settimeout(previous)
            except BaseException:
                failure = "timeout_restore_failed"
    if failure is not None:
        raise PairProbeError(failure) from None
    return {"not_listening": True, "accept_rejected": True, "listen_call_rejected": rejected}
