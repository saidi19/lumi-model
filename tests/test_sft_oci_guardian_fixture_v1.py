"""Source and tiny synthetic tests; never invoke the privileged native fixture."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import socket
import stat
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/run_sft_oci_guardian_fixture_v1.py"
TREE = ast.parse(SOURCE.read_text())
spec = importlib.util.spec_from_file_location("_synthetic_oci_guardian_fixture_v1", SOURCE)
subject = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = subject
spec.loader.exec_module(subject)


def function(name):
    return next(node for node in TREE.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


class PureContractTests(unittest.TestCase):
    def binding(self):
        package = types.ModuleType("_synthetic_guardian_admission")
        package.__path__ = [str(ROOT / "src/lumi")]
        sys.modules[package.__name__] = package
        module = __import__(package.__name__ + ".posttraining_sft_oci_admission_v1", fromlist=["_"])
        return module.SoftwareBinding()

    def test_source_import_does_not_load_ml_or_frozen_controller(self):
        self.assertIsNone(subject.BASE)
        self.assertFalse(any(key in sys.modules for key in
                             ("_lumi_oci_guardian_base_v1", "_lumi_guarded_adapter_validator_v1")))
        imports = {node.names[0].name for node in TREE.body if isinstance(node, ast.Import)}
        self.assertFalse(imports & {"torch", "transformers", "numpy", "docker", "requests"})

    def test_frozen_cgroup_body_hash_is_current(self):
        self.assertEqual(hashlib.sha256((ROOT / subject.FROZEN).read_bytes()).hexdigest(),
                         subject.FROZEN_SHA256)

    def test_control_binds_all_six_original_pins_and_domain_identity(self):
        binding = self.binding()
        import json
        raw = subject.control_bytes(binding)
        value = json.loads(raw)
        self.assertLessEqual(len(raw), 4096)
        self.assertEqual(set(value), {"schema", "software_binding", "source_identity_sha256"})
        self.assertEqual(value["software_binding"],
                         {name: getattr(binding, name) for name in binding.__slots__})
        self.assertEqual(value["source_identity_sha256"], binding.source_identity_sha256())
        self.assertEqual(subject.canonical(value), raw)

    def test_environment_has_no_credentials_host_runtime_or_online_policy(self):
        value = subject.worker_environment(self.binding(), "a" * 64, "b" * 64, "positive")
        self.assertNotIn("LD_LIBRARY_PATH", value)
        self.assertNotIn("PYTHONPATH", value)
        self.assertEqual(value["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(value["HF_HUB_OFFLINE"], "1")
        self.assertEqual(value["TRANSFORMERS_OFFLINE"], "1")
        self.assertFalse(any("TOKEN" in key or "SECRET" in key for key in value
                             if key != "TOKENIZERS_PARALLELISM"))
        self.assertEqual(value["LUMI_GUARDIAN_ADAPTER_SHA256"], "a" * 64)
        self.assertEqual(value["LUMI_GUARDIAN_CONTROL_SHA256"], "b" * 64)

    def test_each_negative_case_changes_only_its_intended_pin(self):
        binding = self.binding()
        original = subject.worker_environment(binding, "a" * 64, "b" * 64, "positive")
        for case, key in (("wrong_control", "LUMI_GUARDIAN_CONTROL_SHA256"),
                          ("wrong_adapter", "LUMI_GUARDIAN_ADAPTER_SHA256")):
            value = subject.worker_environment(binding, "a" * 64, "b" * 64, case)
            self.assertEqual({name for name in value if value[name] != original[name]}, {key})
            self.assertEqual(value[key], "0" * 64)

    def test_unknown_case_fails_closed(self):
        with self.assertRaises(subject.FixtureError):
            subject.worker_environment(self.binding(), "a" * 64, "b" * 64, "retry")

    def test_created_profile_is_never_started_or_privileged(self):
        value = subject.create_options(self.binding().image_id, "lumi-oci-guardian-" + "a" * 32)
        self.assertEqual(value[0], "create")
        self.assertEqual(value[-1], self.binding().image_id)
        for option in ("--pull=never", "--user=10001:10001", "--network=none", "--read-only",
                       "--ipc=private", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                       "--cpus=2", "--memory=4g", "--memory-swap=4g", "--pids-limit=256"):
            self.assertIn(option, value)
        self.assertFalse(any(item.startswith(("--volume", "--mount", "--device", "--privileged"))
                             for item in value))

    def test_error_codes_never_carry_arbitrary_text(self):
        value = subject.FixtureError("synthetic private exception text")
        self.assertEqual(value.code, "fixture_failed")
        self.assertEqual(str(value), "fixture_failed")

    def test_atime_is_not_part_of_file_state(self):
        before = types.SimpleNamespace(st_dev=1, st_ino=2, st_mode=stat.S_IFREG | 0o400,
                                       st_uid=0, st_gid=0, st_size=10, st_mtime_ns=11,
                                       st_ctime_ns=12, st_atime_ns=13)
        after = types.SimpleNamespace(**vars(before))
        after.st_atime_ns = 999
        self.assertEqual(subject.file_state(before), subject.file_state(after))
        for key in ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_size", "st_mtime_ns",
                    "st_ctime_ns"):
            changed = types.SimpleNamespace(**vars(before))
            setattr(changed, key, getattr(changed, key) + 1)
            self.assertNotEqual(subject.file_state(before), subject.file_state(changed), key)

    def test_remaining_rejects_expired_deadline(self):
        with self.assertRaises(subject.FixtureError) as caught:
            subject.remaining(time.monotonic() - 1)
        self.assertEqual(caught.exception.code, "deadline_exhausted")

    def test_host_crosschecks_reported_adapter_and_control_pins(self):
        validator = Mock(return_value={"status": "verified"})
        context = {"adapter_module": types.SimpleNamespace(validate_report=validator),
                   "binding": self.binding(), "adapter_sha256": "a" * 64,
                   "control_sha256": "b" * 64}
        report = {"adapter_sha256": "a" * 64, "control_sha256": "b" * 64}
        with patch.object(subject, "CONTEXT", context):
            self.assertEqual(subject.validate_worker(report, "positive", 0),
                             {"status": "verified"})
        validator.assert_called_once()

    def test_wrong_report_pin_rejected_before_success_validator(self):
        validator = Mock()
        context = {"adapter_module": types.SimpleNamespace(validate_report=validator),
                   "binding": self.binding(), "adapter_sha256": "a" * 64,
                   "control_sha256": "b" * 64}
        for field in ("adapter_sha256", "control_sha256"):
            report = {"adapter_sha256": "a" * 64, "control_sha256": "b" * 64,
                      field: "c" * 64}
            with patch.object(subject, "CONTEXT", context):
                with self.assertRaises(subject.FixtureError):
                    subject.validate_worker(report, "positive", 0)
        validator.assert_not_called()


class TinyImageCopyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lumi-synthetic-image-copy-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.source, self.target = self.base / "source.tar", self.base / "private.tar"
        self.raw = b"generated software archive stand-in"
        self.source.write_bytes(self.raw)
        self.ids = {}
        self.size_patch = patch.object(subject, "ARCHIVE_BYTES", len(self.raw))
        self.hash_patch = patch.object(subject, "ARCHIVE_SHA256",
                                       hashlib.sha256(self.raw).hexdigest())
        self.size_patch.start()
        self.hash_patch.start()
        self.addCleanup(self.size_patch.stop)
        self.addCleanup(self.hash_patch.stop)

    def copy(self):
        return subject.copy_image(self.source, self.target, time.monotonic() + 2, self.ids)

    def test_checked_copy_is_new_owned_readonly_file(self):
        self.assertEqual(self.copy(), len(self.raw))
        self.assertEqual(self.target.read_bytes(), self.raw)
        self.assertEqual(stat.S_IMODE(self.target.stat().st_mode), 0o400)
        self.assertEqual(self.ids, {self.target: subject.identity(self.target.stat())})
        self.assertEqual(self.source.read_bytes(), self.raw)

    def test_hash_failure_still_tracks_exact_partial_file(self):
        with patch.object(subject, "ARCHIVE_SHA256", "0" * 64):
            with self.assertRaises(subject.FixtureError):
                self.copy()
        self.assertEqual(self.ids, {self.target: subject.identity(self.target.stat())})

    def test_existing_target_never_overwritten_or_claimed(self):
        self.target.write_bytes(b"prior evidence")
        with self.assertRaises(FileExistsError):
            self.copy()
        self.assertEqual(self.target.read_bytes(), b"prior evidence")
        self.assertEqual(self.ids, {})

    def test_wrong_length_rejected_before_private_write(self):
        with patch.object(subject, "ARCHIVE_BYTES", len(self.raw) + 1):
            with self.assertRaises(subject.FixtureError):
                self.copy()
        self.assertFalse(self.target.exists())
        self.assertEqual(self.ids, {})

    def test_symlink_source_never_followed(self):
        linked = self.base / "link.tar"
        linked.symlink_to(self.source)
        with self.assertRaises(subject.FixtureError):
            subject.copy_image(linked, self.target, time.monotonic() + 2, self.ids)
        self.assertFalse(self.target.exists())

    def test_stream_change_is_detected(self):
        original = subject.os.fstat
        calls = [0]

        def changed(fd):
            value = original(fd)
            if value.st_ino == self.source.stat().st_ino:
                calls[0] += 1
                if calls[0] > 1:
                    return types.SimpleNamespace(**{key: getattr(value, key) for key in
                        ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_size",
                         "st_mtime_ns")}, st_ctime_ns=value.st_ctime_ns + 1)
            return value

        with patch.object(subject.os, "fstat", side_effect=changed):
            with self.assertRaises(subject.FixtureError):
                self.copy()
        self.assertIn(self.target, self.ids)


class PacketTests(unittest.TestCase):
    def test_one_bounded_packet_and_eof(self):
        channel = Mock()
        channel.recvmsg.return_value = (b'{"status":"synthetic"}', [], 0, None)
        channel.recv.return_value = b""
        with patch.object(subject.select, "select", return_value=([channel], [], [])):
            self.assertEqual(subject.worker_message(channel), {"status": "synthetic"})

    def test_mocked_packet_rejects_extra_or_unclosed_sender(self):
        channel = Mock()
        channel.recvmsg.return_value = (b'{"status":"synthetic"}', [], 0, None)
        channel.recv.return_value = b"X"
        with patch.object(subject.select, "select", return_value=([channel], [], [])):
            with self.assertRaises(subject.FixtureError):
                subject.worker_message(channel)

    def test_mocked_truncated_packet_rejected(self):
        channel = Mock()
        channel.recvmsg.return_value = (b"{}", [], socket.MSG_TRUNC, None)
        with patch.object(subject.select, "select", return_value=([channel], [], [])):
            with self.assertRaises(subject.FixtureError):
                subject.worker_message(channel)
        channel.recv.assert_not_called()


class SourceSequenceTests(unittest.TestCase):
    def test_native_guard_before_any_privileged_setup(self):
        source = ast.unparse(function("run"))
        self.assertLess(source.index("runtime_rejected"), source.index("load_base()"))
        self.assertIn("sys.flags.no_site", source)
        self.assertIn("sys.flags.isolated", source)
        self.assertIn("LUMI_OCI_GUARDIAN_CI", source)

    def test_actual_worker_attaches_before_namespace_and_exec(self):
        source = ast.unparse(function("workload"))
        ordered = ["sanitize(group_fd", "enter_current_process", "create_loopback_namespace",
                   "enter_oci_rootfs", "restrict_filesystem", "install_ipc_filter",
                   "drop_worker_privileges", "os.close(group_fd)", "phase': 'ready'",
                   "channel.recv(1)", "os.execve"]
        positions = [source.index(value) for value in ordered]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("posttraining_sft_mount_root_v1", source)
        self.assertNotIn("allow_private_root_listing", source)
        self.assertIn("expected_root_identity=CONTEXT['root_identity']", source)

    def test_old_controller_is_reused_not_rewritten(self):
        source = ast.unparse(function("cases"))
        self.assertIn("BASE.guardian = guardian", source)
        self.assertIn("BASE.run_case('worker_run'", source)
        self.assertNotIn("BASE.controller =", source)
        self.assertIn("BASE.wait_statuses", source)
        self.assertLess(source.index("BASE.wait_statuses"),
                        source.index("DESCENDANTS_REAPED = True"))

    def test_guardian_arms_only_after_single_attached_dropped_worker(self):
        source = ast.unparse(function("guardian"))
        self.assertLess(source.index("isolation_installed"), source.index("def release()"))
        self.assertLess(source.index("snapshot['pids_current'] == 1"),
                        source.index("watch_parent("))
        self.assertIn("expected_parent_pid=controller_pid", source)
        self.assertIn("worker_pid in statuses", source)
        self.assertLess(source.index("validate_worker("), source.index("lease.close()"))

    def test_exact_three_cases_no_rebuild_start_pull_or_package_installs(self):
        self.assertEqual(subject.CASES, ("wrong_control", "wrong_adapter", "positive"))
        source = SOURCE.read_text()
        for text in ('["start"', '["pull"', '["build"', "pip install", "--force", "rmtree("):
            self.assertNotIn(text, source)
        calls = [node for node in ast.walk(function("run")) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and node.func.attr == "verify_created_oci"]
        self.assertEqual(len(calls), 2)

    def test_only_two_separately_bound_files_installed(self):
        source = ast.unparse(function("run"))
        self.assertIn("((WORKER_PATH[1:], adapter_raw), (CONTROL_PATH[1:], control_raw))", source)
        self.assertIn("mode=292", source)  # 0444
        self.assertLess(source.index("owned.freeze()"), source.index("reports = cases("))

    def test_cleanup_requires_reaping_and_owned_identity(self):
        source = ast.unparse(function("run"))
        self.assertIn("owned is not None and DESCENDANTS_REAPED", source)
        self.assertIn("identity(path.lstat()) == expected", source)
        self.assertIn("identity(path.lstat()) == directory_ids[path]", source)
        self.assertIn("identity(parent.lstat()) == parent_id", source)
        self.assertNotIn("shutil", source)

    def test_enforced_caps_are_bounded(self):
        self.assertEqual(subject.MAXIMUM_SECONDS, 900)
        self.assertEqual(subject.MAXIMUM_EXPORT_BYTES, 2 * 1024**3)
        self.assertEqual(subject.MAXIMUM_EXTRACT_BYTES, 4 * 1024**3)
        self.assertEqual(subject.MAXIMUM_WORKER_BYTES, 32768)


if __name__ == "__main__":
    unittest.main()
