"""Synthetic parser fixtures only; no procfs, cgroupfs, device or host reads."""

from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_synthetic_host_snapshot_v1", ROOT / "src/lumi/posttraining_sft_host_snapshot_v1.py",
)
assert SPEC is not None and SPEC.loader is not None
snapshot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot)


def mount_line(*, identifier=37, parent=29, device=b"0:29", root=b"/",
               target=b"/sys/fs/cgroup", options=b"rw,nosuid,nodev,noexec,relatime",
               optional=b"", filesystem=b"cgroup2", source=b"cgroup", super_options=b"rw"):
    fields = [str(identifier).encode(), str(parent).encode(), device, root, target, options]
    if optional:
        fields.extend(optional.split(b" "))
    fields.extend([b"-", filesystem, source, super_options])
    return b" ".join(fields) + b"\n"


def network(*rows):
    return b"\n".join((*snapshot.NET_DEV_HEADERS, *rows)) + b"\n"


def interface(label=b"lo", value=b"0", count=16):
    return b"    " + label + b": " + b" ".join([value] * count)


def label(index):
    return b"key_" + bytes((97 + index // 26, 97 + index % 26))


class RejectionTests(unittest.TestCase):
    def rejected(self, function, raw, code, **kwargs):
        with self.assertRaises(snapshot.SnapshotError) as caught:
            function(raw, **kwargs)
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(str(caught.exception), code)


class MembershipTests(RejectionTests):
    def deny(self, raw):
        self.rejected(snapshot.parse_membership, raw, "host_membership_rejected")

    def test_root_and_canonical_systemd_components(self):
        self.assertEqual(snapshot.parse_membership(b"0::/\n"), ())
        self.assertEqual(snapshot.parse_membership(b"0::/user.slice/app@one:two.scope/.hidden/x-y_9\n"),
                         ("user.slice", "app@one:two.scope", ".hidden", "x-y_9"))

    def test_component_and_count_boundaries(self):
        self.assertEqual(len(snapshot.parse_membership(b"0::/" + b"a" * 255 + b"\n")[0]), 255)
        self.deny(b"0::/" + b"a" * 256 + b"\n")
        self.assertEqual(len(snapshot.parse_membership(b"0::/" + b"/".join([b"a"] * 32)
                                                       + b"\n")), 32)
        self.deny(b"0::/" + b"/".join([b"a"] * 33) + b"\n")

    def test_exact_byte_capacity_and_overflow(self):
        raw = b"0::/" + b"/".join([b"a" * 255] * 15 + [b"b" * 251]) + b"\n"
        self.assertEqual(len(raw), snapshot.MAXIMUM_MEMBERSHIP_BYTES)
        self.assertEqual(len(snapshot.parse_membership(raw)), 16)
        self.deny(raw[:-1] + b"b\n")

    def test_legacy_multiple_deleted_noncanonical_and_nonascii_rejected(self):
        cases = (
            b"", b"0::/", b"0::/\n\n", b"0::/\n1:cpu:/\n", b"1:cpu:/\n",
            b"00::/\n", b"0:cpu:/\n", b"0::relative\n", b"0:://\n",
            b"0::/a/\n", b"0::/a//b\n", b"0::/a/./b\n", b"0::/a/../b\n",
            b"0::/a (deleted)\n", b"0::/a\\b\n", b"0::/a b\n", b"0::/a\tb\n",
            b"0::/a\r\n", b"0::/\xff\n", "0::/\n", bytearray(b"0::/\n"), None,
        )
        for raw in cases:
            with self.subTest(type_name=type(raw).__name__):
                self.deny(raw)


class MountInfoTests(RejectionTests):
    def deny(self, raw, *, mapping=False):
        self.rejected(snapshot.parse_mountinfo, raw, "host_cgroup_mapping_unsupported"
                      if mapping else "host_mountinfo_rejected")

    def test_exact_target_device_without_path_retention(self):
        unrelated = mount_line(identifier=29, parent=1, device=b"8:1", target=b"/",
                               filesystem=b"ext4", source=b"/dev/sda1", optional=b"shared:1")
        self.assertEqual(snapshot.parse_mountinfo(unrelated + mount_line()),
                         {"device": (0, 29), "mount_read_only": False})

    def test_read_only_combines_mount_and_superblock_metadata(self):
        for mount_ro, super_ro, expected in ((False, False, False), (True, False, True),
                                             (False, True, True), (True, True, True)):
            with self.subTest(mount=mount_ro, superblock=super_ro):
                raw = mount_line(options=b"ro" if mount_ro else b"rw",
                                 super_options=b"ro" if super_ro else b"rw")
                self.assertIs(snapshot.parse_mountinfo(raw)["mount_read_only"], expected)

    def test_missing_stacked_child_nonroot_or_legacy_target_rejected(self):
        cases = (
            mount_line(target=b"/elsewhere"),
            mount_line() + mount_line(identifier=38),
            mount_line() + mount_line(identifier=38, target=b"/sys/fs/cgroup/child"),
            mount_line(root=b"/user.slice"), mount_line(filesystem=b"cgroup"),
            mount_line(target=b"/sys/fs/cgroup2"),
        )
        for raw in cases:
            with self.subTest(length=len(raw)):
                self.deny(raw, mapping=True)

    def test_unrelated_cgroup_mount_is_not_the_target(self):
        raw = mount_line() + mount_line(identifier=38, target=b"/other/cgroup", root=b"/other")
        self.assertEqual(snapshot.parse_mountinfo(raw)["device"], (0, 29))

    def test_known_and_unknown_optional_fields_remain_structural(self):
        raw = mount_line(
            optional=b"shared:7 master:3 propagate_from:4 unbindable idmapped new_tag:x",
        )
        self.assertFalse(snapshot.parse_mountinfo(raw)["mount_read_only"])
        for optional in (b"shared", b"shared:0", b"shared:01", b"master:no", b"unbindable:1",
                         b"shared:1 shared:2", b"new_tag:", b"new_tag:a new_tag:b", b"Bad:1"):
            with self.subTest(optional=optional):
                self.deny(mount_line(optional=optional))

    def test_documented_path_escapes_and_option_escapes_accepted(self):
        unrelated = mount_line(identifier=38, target=b"/else\\040where/tab\\011slash\\134",
                               root=b"/line\\012name", source=b"/dev/a\\040b",
                               super_options=b"rw,key=a\\054b")
        self.assertFalse(snapshot.parse_mountinfo(mount_line() + unrelated)["mount_read_only"])

    def test_octal_target_aliases_bad_escapes_and_noncanonical_paths_rejected(self):
        for target in (b"/sys/fs/\\143group", b"/sys/fs/cgroup/", b"/sys//fs/cgroup",
                       b"/sys/fs/./cgroup", b"relative", b"/a\\", b"/a\\000", b"/a\\999"):
            with self.subTest(target=target):
                self.deny(mount_line(target=target))

    def test_all_lines_are_checked_including_unrelated_records(self):
        bad_lines = (
            b"malformed\n", mount_line(identifier=38, target=b"/other", device=b"x:1"),
            mount_line(identifier=38, target=b"/other", optional=b"shared:bad"),
            mount_line(identifier=38, target=b"/other", options=b"nosuid"),
        )
        for bad in bad_lines:
            with self.subTest(length=len(bad)):
                self.deny(mount_line() + bad)

    def test_numeric_ids_device_and_duplicates_rejected(self):
        for kwargs in ({"identifier": 0}, {"parent": 0}, {"identifier": "01"},
                       {"identifier": 2**63}, {"device": b"-1:2"}, {"device": b"0:01"},
                       {"device": b"1:2:3"}, {"device": b"4294967296:0"}):
            with self.subTest(kwargs=kwargs):
                self.deny(mount_line(**kwargs))
        self.deny(mount_line() + mount_line(target=b"/other"))

    def test_options_require_one_access_marker_each_and_no_duplicate_tokens(self):
        for options in (b"rw,ro", b"rw,rw", b"nosuid", b"rw,", b",rw", b"rw,nosuid,nosuid"):
            with self.subTest(options=options):
                self.deny(mount_line(options=options))
                self.deny(mount_line(super_options=options))

    def test_line_field_byte_and_termination_bounds(self):
        self.deny(mount_line()[:-1])
        self.deny(mount_line() + b"\n")
        self.deny(mount_line().replace(b"37 29", b"37  29"))
        self.deny(b"x" * snapshot.MAXIMUM_MOUNTINFO_BYTES + b"\n")
        records = [mount_line(identifier=index + 100, target=b"/m" + str(index).encode(),
                              options=b"rw", filesystem=b"tmpfs", source=b"none")
                   for index in range(snapshot.MAXIMUM_MOUNT_LINES - 1)]
        raw = mount_line() + b"".join(records)
        self.assertLessEqual(len(raw), snapshot.MAXIMUM_MOUNTINFO_BYTES)
        self.assertFalse(snapshot.parse_mountinfo(raw)["mount_read_only"])
        self.deny(raw + mount_line(identifier=9999, target=b"/other/extra"))

    def test_field_option_and_path_boundaries(self):
        tags = [label(index) for index in range(54)]
        self.assertFalse(snapshot.parse_mountinfo(mount_line(optional=b" ".join(tags)))[
            "mount_read_only"])
        self.deny(mount_line(optional=b" ".join([*tags, b"extra"])))
        options = b",".join([b"rw", *(label(index) for index in range(63))])
        self.assertFalse(snapshot.parse_mountinfo(mount_line(options=options))["mount_read_only"])
        self.deny(mount_line(options=options + b",extra"))
        other = mount_line(identifier=38, target=b"/" + b"x" * 4095)
        self.assertFalse(snapshot.parse_mountinfo(mount_line() + other)["mount_read_only"])
        self.deny(mount_line() + mount_line(identifier=38, target=b"/" + b"x" * 4096))


class ScalarTests(RejectionTests):
    def test_zero_maximum_and_explicit_unlimited(self):
        self.assertEqual(snapshot.parse_scalar(b"0\n"), 0)
        self.assertEqual(snapshot.parse_scalar(b"9223372036854775807\n"), 2**63 - 1)
        self.assertIsNone(snapshot.parse_scalar(b"max\n", allow_max=True))
        self.assertEqual(snapshot.parse_scalar(b"1\n", allow_max=True), 1)

    def test_canonical_scalar_and_explicit_max_gate(self):
        for raw in (b"", b"1", b"01\n", b"+1\n", b"-1\n", b" 1\n", b"1 \n", b"1\n2\n",
                    b"max\n", b"MAX\n", b"9223372036854775808\n", b"9" * 4095 + b"\n",
                    b"9" * 4096 + b"\n", b"1\r\n", "1\n", bytearray(b"1\n")):
            with self.subTest(type_name=type(raw).__name__):
                self.rejected(snapshot.parse_scalar, raw, "host_scalar_rejected")

    def test_allow_max_configuration_rejects_truthy_aliases(self):
        for value in (0, 1, None, "true", [], {}):
            with self.subTest(type_name=type(value).__name__):
                self.rejected(
                    snapshot.parse_scalar, b"max\n", "host_snapshot_configuration_rejected",
                    allow_max=value,
                )


class ControllersTests(RejectionTests):
    def test_empty_and_unique_controller_sets(self):
        self.assertEqual(snapshot.parse_controllers(b""), frozenset())
        self.assertEqual(snapshot.parse_controllers(b"\n"), frozenset())
        self.assertEqual(snapshot.parse_controllers(b"cpu memory io pids\n"),
                         frozenset({"cpu", "memory", "io", "pids"}))

    def test_controller_label_and_count_boundaries(self):
        self.assertEqual(snapshot.parse_controllers(b"a" * 64 + b"\n"), frozenset({"a" * 64}))
        raw = b" ".join(label(index) for index in range(32)) + b"\n"
        self.assertEqual(len(snapshot.parse_controllers(raw)), 32)
        for raw in (b"a" * 65 + b"\n", b" ".join(label(index) for index in range(33)) + b"\n"):
            self.rejected(snapshot.parse_controllers, raw, "host_controllers_rejected")

    def test_duplicate_whitespace_case_and_shape_rejected(self):
        for raw in (b"cpu", b"cpu cpu\n", b"cpu  memory\n", b"cpu\tmemory\n", b" cpu\n",
                    b"cpu \n", b"CPU\n", b"cpu2\n", b"cpu-memory\n", b"cpu\n\n",
                    b"\xff\n", b"x" * 4096 + b"\n", None, bytearray(), ""):
            with self.subTest(type_name=type(raw).__name__):
                self.rejected(snapshot.parse_controllers, raw, "host_controllers_rejected")


class EventsTests(RejectionTests):
    def test_required_booleans_and_unknown_numeric_keys_retained_transiently(self):
        self.assertEqual(snapshot.parse_events(b"frozen 1\npopulated 0\nfuture_counter 7\n"),
                         {"frozen": 1, "populated": 0, "future_counter": 7})
        self.assertEqual(snapshot.parse_events(b"populated 1\nfrozen 0\n")['populated'], 1)

    def test_missing_duplicate_nonboolean_or_noncanonical_values_rejected(self):
        for raw in (b"", b"\n", b"populated 1\n", b"frozen 0\n", b"populated 2\nfrozen 0\n",
                    b"populated 1\nfrozen 2\n", b"populated 0\nfrozen 0\nfrozen 1\n",
                    b"populated 01\nfrozen 0\n", b"populated 0\nfrozen -1\n",
                    b"populated 0\nfrozen 0\nunknown max\n", b"populated 0\nfrozen 0\nUnknown 1\n",
                    b"populated  0\nfrozen 0\n", b"populated 0 \nfrozen 0\n",
                    b"populated 0\nfrozen 0", b"x" * 4096 + b"\n"):
            with self.subTest(length=len(raw)):
                self.rejected(snapshot.parse_events, raw, "host_events_rejected")

    def test_event_count_and_integer_boundaries(self):
        base = b"populated 0\nfrozen 0\n"
        extra = b"".join(label(index) + b" 9223372036854775807\n" for index in range(30))
        self.assertEqual(len(snapshot.parse_events(base + extra)), 32)
        self.rejected(snapshot.parse_events, base + extra + b"another 1\n", "host_events_rejected")
        self.rejected(snapshot.parse_events, base + b"future 9223372036854775808\n",
                      "host_events_rejected")


class NetworkTests(RejectionTests):
    def deny(self, raw):
        self.rejected(snapshot.parse_net_dev, raw, "host_network_snapshot_rejected")

    def test_only_interface_aggregates_leave_parser(self):
        value = snapshot.parse_net_dev(network(interface(), interface(b"eth0")))
        self.assertEqual(value, {"interface_count": 2, "non_loopback_interfaces": 1,
                                 "loopback_interface_observed": True})
        self.assertNotIn("eth0", repr(value))
        self.assertEqual(snapshot.parse_net_dev(network()), {
            "interface_count": 0, "non_loopback_interfaces": 0,
            "loopback_interface_observed": False,
        })

    def test_nonloopback_without_loopback_is_observation_not_inference(self):
        value = snapshot.parse_net_dev(network(interface(b"lo_like"), interface(b"eth0")))
        self.assertEqual(value["non_loopback_interfaces"], 2)
        self.assertFalse(value["loopback_interface_observed"])

    def test_headers_are_exact(self):
        valid = network(interface())
        for raw in (valid.replace(b"Receive", b"receive"), valid.replace(b" face", b"face"),
                    b"\n" + valid, valid.split(b"\n", 1)[1], valid[:-1]):
            with self.subTest(length=len(raw)):
                self.deny(raw)

    def test_counter_count_canonicality_and_uint64_boundary(self):
        self.assertEqual(snapshot.parse_net_dev(network(interface(value=b"18446744073709551615")))[
            "interface_count"], 1)
        for row in (interface(count=15), interface(count=17),
                    interface(value=b"18446744073709551616"),
                    interface(value=b"01"), interface(value=b"-1"), interface(value=b"+1"),
                    interface(value=b"0\t0"), interface() + b" "):
            with self.subTest(length=len(row)):
                self.deny(network(row))

    def test_duplicate_label_shape_and_interface_count_boundaries(self):
        self.deny(network(interface(), interface()))
        for name in (b"", b"a" * 16, b"a/b", b"a:b", b"a b", b"a\tb", b"\xff"):
            with self.subTest(name=name):
                self.deny(network(interface(name)))
        rows = [interface(b"if" + str(index).encode()) for index in range(64)]
        self.assertEqual(snapshot.parse_net_dev(network(*rows))["interface_count"], 64)
        self.deny(network(*rows, interface(b"extra")))
        self.deny(b"x" * snapshot.MAXIMUM_NET_DEV_BYTES + b"\n")


class GlobalParserTests(RejectionTests):
    def test_error_messages_do_not_expose_nonallowlisted_values(self):
        for value in ("SECRET input/path", None, True, [], {}, 1):
            with self.subTest(type_name=type(value).__name__):
                error = snapshot.SnapshotError(value)
                self.assertEqual(error.code, "host_snapshot_configuration_rejected")
                self.assertEqual(str(error), error.code)

    def test_single_byte_fuzz_and_wrong_types_never_raise_unclassified_errors(self):
        parsers = (
            snapshot.parse_membership, snapshot.parse_mountinfo, snapshot.parse_scalar,
            snapshot.parse_controllers, snapshot.parse_events, snapshot.parse_net_dev,
        )
        for parser in parsers:
            for value in (*[bytes((byte,)) + b"\n" for byte in range(256)], b"", None, 0, [], {},
                          bytearray(b"0\n"), "0\n"):
                with self.subTest(parser=parser.__name__, type_name=type(value).__name__):
                    try:
                        result = parser(value)
                    except snapshot.SnapshotError as error:
                        self.assertIn(error.code, snapshot.ERROR_CODES)
                        self.assertEqual(str(error), error.code)
                    else:
                        self.assertIn(type(result), (int, frozenset))

    def test_pure_module_has_no_io_imports_or_calls(self):
        raw = (ROOT / "src/lumi/posttraining_sft_host_snapshot_v1.py").read_bytes()
        tree = ast.parse(raw)
        imports = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                   for alias in node.names]
        self.assertEqual(imports, ["re"])
        blocked = {"open", "read", "write", "socket", "Popen", "run", "system", "exec", "eval"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else (
                    node.func.attr if isinstance(node.func, ast.Attribute) else ""
                )
                self.assertNotIn(name, blocked)


if __name__ == "__main__":
    unittest.main()
