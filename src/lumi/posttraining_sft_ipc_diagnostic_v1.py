"""Pure fixed-code projection; never stringify or retain an exception or path."""

from __future__ import annotations

import errno

ERRNO_CODES = {
    errno.EPERM: "errno_permission", errno.EACCES: "errno_access",
    errno.EROFS: "errno_readonly", errno.ECONNREFUSED: "errno_refused",
    errno.EINVAL: "errno_invalid", errno.EISCONN: "errno_connected",
    errno.EOPNOTSUPP: "errno_unsupported", errno.EAFNOSUPPORT: "errno_family",
    errno.EPROTOTYPE: "errno_protocol", errno.ENOENT: "errno_missing",
    errno.EBADF: "errno_badfd", errno.ETIMEDOUT: "errno_timeout",
    errno.EAGAIN: "errno_again", errno.EPIPE: "errno_pipe",
    errno.ECONNRESET: "errno_reset", errno.ENOTCONN: "errno_notconnected",
}
ERROR_CODES = frozenset({*ERRNO_CODES.values(), "errno_other", "expectation_failed",
                         "python_failed"})
OPERATIONS = ("disconnect", "connect", "bind_listen", "sendto", "sendmsg", "roundtrip")
STAGES = frozenset(f"{kind}_{phase}_{operation}"
                   for kind in ("stream", "seqpacket")
                   for phase in ("retarget",)
                   for operation in OPERATIONS)


def failure_code(error):
    if isinstance(error, OSError):
        # Read the built-in slot, not an arbitrary subclass property or __str__.
        number = OSError.errno.__get__(error)
        if type(number) is int:
            return ERRNO_CODES.get(number, "errno_other")
        return "errno_timeout" if type(error) is TimeoutError else "errno_other"
    if isinstance(error, (AssertionError, RuntimeError)):
        return "expectation_failed"
    return "python_failed"
