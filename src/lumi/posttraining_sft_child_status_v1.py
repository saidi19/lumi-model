"""Pure, bounded process-status snapshot policies for the fixed inert Linux probe.

Child callers must read a trusted procfs view of the owned child while a
pinned, live pidfd brackets the read. Controller callers must bind their own
expected identities through trusted local APIs. Neither parser authenticates
bytes or a process generation. Single NSpid/NStgid entries exclude only nesting
visible from that procfs mount, not every namespace or host relationship.

Both policies require no current inheritable/permitted/effective/ambient
capabilities. The child requires NoNewPrivs=1; the separate controller policy
allows either 0 or 1 without changing the controller. The bounding set need
not be empty. Tracer absence means no tracer visible in the trusted procfs
view; it does not exclude a tracer in an outer PID namespace. This is not
permanent privilege containment: no_new_privs constrains exec transitions,
not every privilege-changing operation. No thread/lifecycle, sandbox, ledger,
worker admission, network or resource enforcement is provided here.

Only aggregate counts/booleans and opaque error codes leave the parser. Raw
bytes exist transiently while parsing; this is not a secure-erasure claim.
Field meanings: https://man7.org/linux/man-pages/man5/proc_pid_status.5.html
NoNewPrivs: https://docs.kernel.org/userspace-api/no_new_privs.html
"""

from __future__ import annotations

import re

MAXIMUM_STATUS_BYTES = 16384
MAXIMUM_STATUS_FIELDS = 256
MAXIMUM_PID = (1 << 31) - 1
MAXIMUM_ID = (1 << 32) - 2
REQUIRED_FIELDS = frozenset(
    (
        b"Pid", b"Tgid", b"PPid", b"TracerPid", b"Threads", b"Uid", b"Gid",
        b"NSpid", b"NStgid", b"CapInh", b"CapPrm", b"CapEff", b"CapAmb",
        b"CapBnd", b"NoNewPrivs",
    )
)
CAPABILITY_FIELDS = (b"CapInh", b"CapPrm", b"CapEff", b"CapAmb", b"CapBnd")
ERROR_CODES = frozenset(
    (
        "child_status_configuration_rejected",
        "child_status_shape_rejected",
        "child_status_identity_rejected",
        "child_status_capability_rejected",
        "child_status_policy_rejected",
    )
)
_KEY = re.compile(rb"[A-Za-z][A-Za-z0-9_]{0,63}")
_DECIMAL = re.compile(rb"(?:0|[1-9][0-9]{0,9})")
_CAPABILITY = re.compile(rb"[0-9a-fA-F]{16}")


class StatusError(ValueError):
    """An allowlisted code without source fields, values or exception details."""

    def __init__(self, code: str):
        if type(code) is not str or code not in ERROR_CODES:
            code = "child_status_policy_rejected"
        self.code = code
        super().__init__(code)


def _need(condition: bool, code: str) -> None:
    if not condition:
        raise StatusError(code) from None


def _numbers(value: bytes, count: int, maximum: int) -> tuple[int, ...]:
    pieces = value.split()
    _need(
        len(pieces) == count and all(_DECIMAL.fullmatch(piece) for piece in pieces),
        "child_status_shape_rejected",
    )
    result = tuple(int(piece) for piece in pieces)
    _need(all(number <= maximum for number in result), "child_status_shape_rejected")
    return result


def _parse_snapshot(
    raw: bytes, *, expected_pid: int, expected_ppid: int, expected_uid: int, expected_gid: int
) -> dict[str, bool | int]:
    """Check the shared identity, capability and snapshot shape requirements.

    All required fields occur exactly once. Additional syntactically valid
    keys are allowed, but every duplicate key and malformed/control-byte line
    is rejected. Numeric fields use canonical unsigned decimal and capability
    fields the kernel's 16-hex-digit representation. Identifiers are not output.
    """
    for identifier, maximum in (
        (expected_pid, MAXIMUM_PID), (expected_ppid, MAXIMUM_PID),
        (expected_uid, MAXIMUM_ID), (expected_gid, MAXIMUM_ID),
    ):
        _need(
            type(identifier) is int and 0 < identifier <= maximum,
            "child_status_configuration_rejected",
        )
    _need(
        type(raw) is bytes and 0 < len(raw) <= MAXIMUM_STATUS_BYTES,
        "child_status_shape_rejected",
    )
    _need(
        raw.endswith(b"\n") and all(byte in (9, 10) or 32 <= byte <= 126 for byte in raw),
        "child_status_shape_rejected",
    )
    lines = raw[:-1].split(b"\n")
    _need(len(lines) <= MAXIMUM_STATUS_FIELDS, "child_status_shape_rejected")
    fields = {}
    seen = set()
    for line in lines:
        key, separator, value = line.partition(b":")
        _need(
            bool(separator) and bool(_KEY.fullmatch(key)) and value[:1] in (b" ", b"\t"),
            "child_status_shape_rejected",
        )
        _need(key not in seen, "child_status_shape_rejected")
        seen.add(key)
        if key in REQUIRED_FIELDS:
            fields[key] = value
    _need(fields.keys() == REQUIRED_FIELDS, "child_status_shape_rejected")

    expected_numbers = {
        b"Pid": (expected_pid,), b"Tgid": (expected_pid,), b"PPid": (expected_ppid,),
        b"NSpid": (expected_pid,), b"NStgid": (expected_pid,),
        b"Uid": (expected_uid,) * 4, b"Gid": (expected_gid,) * 4,
    }
    for key, expected in expected_numbers.items():
        maximum = MAXIMUM_ID if key in (b"Uid", b"Gid") else MAXIMUM_PID
        values = _numbers(fields[key], len(expected), maximum)
        _need(values == expected, "child_status_identity_rejected")
    for key, expected in ((b"TracerPid", 0), (b"Threads", 1)):
        values = _numbers(fields[key], 1, MAXIMUM_PID)
        _need(values == (expected,), "child_status_policy_rejected")
    no_new_privs = _numbers(fields[b"NoNewPrivs"], 1, MAXIMUM_PID)
    _need(no_new_privs in ((0,), (1,)), "child_status_policy_rejected")

    masks = {}
    for key in CAPABILITY_FIELDS:
        value = fields[key].strip(b" \t")
        _need(bool(_CAPABILITY.fullmatch(value)), "child_status_shape_rejected")
        masks[key] = int(value, 16)
    _need(
        all(masks[key] == 0 for key in CAPABILITY_FIELDS if key != b"CapBnd"),
        "child_status_capability_rejected",
    )
    return {
        "status_fields_checked": len(seen),
        "required_fields_checked": len(REQUIRED_FIELDS),
        "identity_values_match_expected": True,
        "single_thread_snapshot": True,
        "tracer_absent_snapshot": True,
        "single_visible_pid_namespace": True,
        "nonzero_uniform_uid_gid": True,
        "current_capability_sets_empty": True,
        "capability_bounding_set_empty": masks[b"CapBnd"] == 0,
        "no_new_privs_set": no_new_privs == (1,),
        "permanent_privilege_containment": False,
        "process_generation_bound": False,
        "execution_authorized": False,
    }


def parse_status(
    raw: bytes, *, expected_pid: int, expected_ppid: int, expected_uid: int, expected_gid: int
) -> dict[str, bool | int]:
    """Require the owned child's full snapshot policy, including NoNewPrivs=1.

    This pure parser performs no OS actions and provides no caller-controlled
    option to weaken the child policy. Tracer absence is relative to the
    caller's trusted procfs view, not a statement about outer PID namespaces.
    """
    report = _parse_snapshot(
        raw, expected_pid=expected_pid, expected_ppid=expected_ppid,
        expected_uid=expected_uid, expected_gid=expected_gid,
    )
    _need(report["no_new_privs_set"] is True, "child_status_policy_rejected")
    return {"child_status_policy_passed": True, **report}


def parse_controller_status(
    raw: bytes, *, expected_pid: int, expected_ppid: int, expected_uid: int, expected_gid: int
) -> dict[str, bool | int]:
    """Require the controller snapshot policy without setting its NoNewPrivs.

    NoNewPrivs may be exactly 0 or 1; its actual boolean is reported. All other
    identity, capability, thread, visible-tracer and namespace rules match the
    child contract. This parser neither reads procfs nor modifies the process.
    """
    report = _parse_snapshot(
        raw, expected_pid=expected_pid, expected_ppid=expected_ppid,
        expected_uid=expected_uid, expected_gid=expected_gid,
    )
    return {"controller_status_policy_passed": True, **report}
