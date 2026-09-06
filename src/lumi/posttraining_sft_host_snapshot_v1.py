"""Pure bounded parsers for self-only Linux host prerequisite observations.

No file, directory, namespace, process, network or device operation occurs here.
The caller must authenticate its procfs/cgroupfs view and discard transient paths,
controller labels and event keys rather than treating these parsers as authority.
Read-only flags describe kernel metadata, never delegation or actual writability.
"""

from __future__ import annotations

import re

MAXIMUM_MEMBERSHIP_BYTES = 4096
MAXIMUM_MOUNTINFO_BYTES = 131072
MAXIMUM_SCALAR_BYTES = 4096
MAXIMUM_CONTROLLERS_BYTES = 4096
MAXIMUM_EVENTS_BYTES = 4096
MAXIMUM_NET_DEV_BYTES = 16384
MAXIMUM_COMPONENTS = 32
MAXIMUM_COMPONENT_BYTES = 255
MAXIMUM_LABEL_BYTES = 64
MAXIMUM_CONTROLLERS = 32
MAXIMUM_EVENTS = 32
MAXIMUM_INTERFACES = 64
MAXIMUM_MOUNT_LINES = 2048
MAXIMUM_MOUNT_FIELDS = 64
MAXIMUM_PATH_BYTES = 4096
MAXIMUM_SCALAR = 2**63 - 1
MAXIMUM_COUNTER = 2**64 - 1
TARGET_MOUNT = b"/sys/fs/cgroup"
NET_DEV_HEADERS = (
    b"Inter-|   Receive                                                |  Transmit",
    b" face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs "
    b"drop fifo colls carrier compressed",
)
ERROR_CODES = frozenset({
    "host_snapshot_configuration_rejected", "host_membership_rejected",
    "host_mountinfo_rejected", "host_cgroup_mapping_unsupported", "host_scalar_rejected",
    "host_controllers_rejected", "host_events_rejected", "host_network_snapshot_rejected",
})
_COMPONENT = re.compile(rb"[A-Za-z0-9_.@:-]{1,255}")
_LABEL = re.compile(rb"[a-z][a-z_]{0,63}")
_FSTYPE = re.compile(rb"[A-Za-z0-9_.-]{1,64}")
_INTERFACE = re.compile(rb"[A-Za-z0-9_.@-]{1,15}")
_DECIMAL = re.compile(rb"(?:0|[1-9][0-9]*)")
_PATH_ESCAPES = {b"040": 32, b"011": 9, b"012": 10, b"134": 92}


class SnapshotError(ValueError):
    """Allowlisted stage code only; never retain input, path or exception text."""

    def __init__(self, code):
        self.code = code if type(code) is str and code in ERROR_CODES else (
            "host_snapshot_configuration_rejected"
        )
        super().__init__(self.code)


def _need(condition, code):
    if not condition:
        raise SnapshotError(code) from None


def _raw(value, maximum, code):
    _need(type(value) is bytes and 0 < len(value) <= maximum and value.endswith(b"\n"), code)


def _decimal(value, maximum, code, *, positive=False):
    _need(type(value) is bytes and 0 < len(value) <= len(str(maximum))
          and _DECIMAL.fullmatch(value) is not None, code)
    result = int(value)
    _need((1 if positive else 0) <= result <= maximum, code)
    return result


def parse_membership(raw: bytes) -> tuple[str, ...]:
    """Accept one canonical unified hierarchy path; returned components are transient."""
    code = "host_membership_rejected"
    _raw(raw, MAXIMUM_MEMBERSHIP_BYTES, code)
    _need(raw.startswith(b"0::/") and raw.count(b"\n") == 1, code)
    path = raw[3:-1]
    if path == b"/":
        return ()
    components = path[1:].split(b"/")
    _need(0 < len(components) <= MAXIMUM_COMPONENTS, code)
    _need(all(_COMPONENT.fullmatch(part) is not None and part not in (b".", b"..")
              for part in components), code)
    return tuple(part.decode("ascii") for part in components)


def _path(token, code):
    """Decode only the documented mountinfo path escapes, never arbitrary octal aliases."""
    _need(0 < len(token) <= MAXIMUM_PATH_BYTES and token.startswith(b"/"), code)
    decoded = bytearray()
    index = 0
    while index < len(token):
        value = token[index]
        if value == 92:
            escape = token[index + 1:index + 4]
            _need(len(escape) == 3 and escape in _PATH_ESCAPES, code)
            decoded.append(_PATH_ESCAPES[escape])
            index += 4
        else:
            _need(33 <= value <= 126, code)
            decoded.append(value)
            index += 1
    result = bytes(decoded)
    _need(len(result) <= MAXIMUM_PATH_BYTES, code)
    _need(result == b"/" or all(part and part not in (b".", b"..")
                                for part in result[1:].split(b"/")), code)
    return result


def _token(token, code):
    _need(0 < len(token) <= MAXIMUM_PATH_BYTES, code)
    index = 0
    while index < len(token):
        value = token[index]
        _need(33 <= value <= 126, code)
        if value == 92:
            escape = token[index + 1:index + 4]
            _need(len(escape) == 3 and all(48 <= byte <= 55 for byte in escape)
                  and 0 < int(escape, 8) <= 255, code)
            index += 4
        else:
            index += 1


def _options(token, code):
    _token(token, code)
    options = token.split(b",")
    _need(all(options) and len(options) <= 64 and len(set(options)) == len(options), code)
    access = [value for value in options if value in (b"ro", b"rw")]
    _need(len(access) == 1, code)
    return access[0] == b"ro"


def _optional(tokens, code):
    seen = set()
    for token in tokens:
        _token(token, code)
        key, separator, value = token.partition(b":")
        _need(_LABEL.fullmatch(key) is not None and key not in seen, code)
        seen.add(key)
        if key in (b"shared", b"master", b"propagate_from"):
            _need(separator == b":" and bool(value), code)
            _decimal(value, MAXIMUM_SCALAR, code, positive=True)
        elif key == b"unbindable":
            _need(not separator, code)
        else:
            # Unknown optional tags are structurally checked but carry no authority.
            _need(not separator or bool(value), code)


def parse_mountinfo(raw: bytes) -> dict[str, tuple[int, int] | bool]:
    """Require one root-mapped cgroup2 target with no stacked/descendant mount.

    Read-only metadata combines per-mount and superblock ro markers. This does
    not establish permissions, ownership, delegation, or ability to create jobs.
    """
    code = "host_mountinfo_rejected"
    mapping = "host_cgroup_mapping_unsupported"
    _raw(raw, MAXIMUM_MOUNTINFO_BYTES, code)
    lines = raw[:-1].split(b"\n")
    _need(0 < len(lines) <= MAXIMUM_MOUNT_LINES, code)
    seen_ids = set()
    targets = []
    child_mount = False
    for line in lines:
        fields = line.split(b" ")
        _need(10 <= len(fields) <= MAXIMUM_MOUNT_FIELDS and all(fields), code)
        separator = len(fields) - 4
        _need(separator >= 6 and fields[separator] == b"-", code)
        mount_id = _decimal(fields[0], MAXIMUM_SCALAR, code, positive=True)
        _need(mount_id not in seen_ids, code)
        seen_ids.add(mount_id)
        _decimal(fields[1], MAXIMUM_SCALAR, code, positive=True)
        device = fields[2].split(b":")
        _need(len(device) == 2, code)
        device = tuple(_decimal(value, 2**32 - 1, code) for value in device)
        root, mountpoint = _path(fields[3], code), _path(fields[4], code)
        mount_ro = _options(fields[5], code)
        _optional(fields[6:separator], code)
        filesystem, source, super_options = fields[-3:]
        _need(_FSTYPE.fullmatch(filesystem) is not None, code)
        _token(source, code)
        super_ro = _options(super_options, code)
        if mountpoint == TARGET_MOUNT:
            targets.append((root, filesystem, device, mount_ro or super_ro))
        elif mountpoint.startswith(TARGET_MOUNT + b"/"):
            child_mount = True
    _need(not child_mount and len(targets) == 1, mapping)
    root, filesystem, device, read_only = targets[0]
    _need(root == b"/" and filesystem == b"cgroup2", mapping)
    return {"device": device, "mount_read_only": read_only}


def parse_scalar(raw: bytes, allow_max: bool = False) -> int | None:
    code = "host_scalar_rejected"
    _need(type(allow_max) is bool, "host_snapshot_configuration_rejected")
    _raw(raw, MAXIMUM_SCALAR_BYTES, code)
    if raw == b"max\n":
        _need(allow_max, code)
        return None
    return _decimal(raw[:-1], MAXIMUM_SCALAR, code)


def parse_controllers(raw: bytes) -> frozenset[str]:
    """Accept the kernel's empty-byte form and an empty newline as no controllers."""
    code = "host_controllers_rejected"
    if type(raw) is bytes and raw in (b"", b"\n"):
        return frozenset()
    _raw(raw, MAXIMUM_CONTROLLERS_BYTES, code)
    labels = raw[:-1].split(b" ")
    _need(0 < len(labels) <= MAXIMUM_CONTROLLERS
          and all(_LABEL.fullmatch(label) is not None for label in labels)
          and len(set(labels)) == len(labels), code)
    return frozenset(label.decode("ascii") for label in labels)


def parse_events(raw: bytes) -> dict[str, int]:
    code = "host_events_rejected"
    _raw(raw, MAXIMUM_EVENTS_BYTES, code)
    lines = raw[:-1].split(b"\n")
    _need(0 < len(lines) <= MAXIMUM_EVENTS, code)
    events = {}
    for line in lines:
        fields = line.split(b" ")
        _need(len(fields) == 2 and _LABEL.fullmatch(fields[0]) is not None, code)
        key = fields[0].decode("ascii")
        _need(key not in events, code)
        events[key] = _decimal(fields[1], MAXIMUM_SCALAR, code)
    _need("populated" in events and "frozen" in events
          and events["populated"] in (0, 1) and events["frozen"] in (0, 1), code)
    return events


def parse_net_dev(raw: bytes) -> dict[str, int | bool]:
    """Validate every interface/counter but retain only three interface aggregates."""
    code = "host_network_snapshot_rejected"
    _raw(raw, MAXIMUM_NET_DEV_BYTES, code)
    lines = raw[:-1].split(b"\n")
    _need(2 <= len(lines) <= MAXIMUM_INTERFACES + 2 and tuple(lines[:2]) == NET_DEV_HEADERS, code)
    labels = set()
    for line in lines[2:]:
        label_field, separator, counters = line.partition(b":")
        label = label_field.lstrip(b" ")
        _need(separator == b":" and _INTERFACE.fullmatch(label) is not None
              and label not in labels and counters.startswith(b" ")
              and not counters.endswith(b" "), code)
        labels.add(label)
        values = [part for part in counters.split(b" ") if part]
        _need(len(values) == 16, code)
        for value in values:
            _decimal(value, MAXIMUM_COUNTER, code)
    loopback = b"lo" in labels
    return {"interface_count": len(labels), "non_loopback_interfaces": len(labels) - int(loopback),
            "loopback_interface_observed": loopback}
