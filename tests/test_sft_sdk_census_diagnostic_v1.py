"""Synthetic metadata-only tests for the read-only SDK census diagnostic."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def definitions(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = definitions("scripts/diagnose_sft_sdk_census_v1.py", "_sdk_census_test_definitions")


def helper_for(root_name):
    return types.SimpleNamespace(
        PARTS=(root_name,),
        identity=m.os.fstat,
        check_mounts=Mock(),
    )


class CensusFixtures(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory(prefix="sdk-census-"))
        self.root = Path(self.directory) / "sdk"
        self.root.mkdir()
        self.fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        self.addCleanup(os.close, self.fd)
        self.helper = types.SimpleNamespace(identity=lambda info: (
            info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
        ))

    def file(self, name, data=b"synthetic"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def report(self):
        return m.empty_report()

    def test_nested_tree_counts_metadata_without_payload_reads(self):
        self.file("a")
        self.file("nested/b")
        with patch.object(m.os, "read", side_effect=AssertionError("payload read")):
            result = self.report()
            m.census(self.fd, self.helper, result)
        self.assertEqual(result["visited_nodes"], 3)
        self.assertGreater(result["accepted_relative_name_bytes"], 0)
        self.assertGreaterEqual(result["accepted_allocated_bytes"], 0)
        self.assertFalse(result["sdk_mutation_performed"])

    def test_external_symlink_is_inspected_as_leaf_not_followed(self):
        outside = Path(self.directory) / "outside"
        outside.mkdir()
        (outside / "sentinel").write_bytes(b"untouched")
        (self.root / "link").symlink_to(outside, target_is_directory=True)
        result = self.report()
        m.census(self.fd, self.helper, result)
        self.assertEqual(result["visited_nodes"], 1)
        self.assertEqual((outside / "sentinel").read_bytes(), b"untouched")

    def test_hardlink_and_special_node_reject_before_more_walk(self):
        original = self.file("a")
        os.link(original, self.root / "b")
        with self.assertRaises(m.CensusError) as error:
            m.census(self.fd, self.helper, self.report())
        self.assertEqual(error.exception.code, "tree_rejected")
        os.unlink(self.root / "b")
        os.unlink(original)
        os.mkfifo(self.root / "fifo")
        with self.assertRaises(m.CensusError) as error:
            m.census(self.fd, self.helper, self.report())
        self.assertEqual(error.exception.code, "tree_rejected")

    def test_listing_budget_is_distinct_and_reports_next_entry(self):
        self.file("a")
        self.file("b")
        with patch.object(m, "MAXIMUM_NODES", 1):
            result = self.report()
            with self.assertRaises(m.CensusError) as error:
                m.census(self.fd, self.helper, result)
        self.assertEqual(error.exception.code, "budget_hit")
        self.assertEqual(result["budget_kind"], "listing_entries")
        self.assertEqual(result["budget_limit"], 1)
        self.assertEqual(result["attempted_value"], 2)

    def test_node_budget_is_distinct_after_listing(self):
        result = self.report()
        with self.assertRaises(m.CensusError) as error:
            m.budget(2, 1, "nodes", result)
        self.assertEqual(error.exception.code, "budget_hit")
        self.assertEqual(result["budget_kind"], "nodes")

    def test_relative_name_budget_counts_ancestor_bytes(self):
        self.file("parent/child")
        with patch.object(m, "MAXIMUM_NAME_BYTES", 1):
            result = self.report()
            with self.assertRaises(m.CensusError) as error:
                m.census(self.fd, self.helper, result)
        self.assertEqual(error.exception.code, "budget_hit")
        self.assertEqual(result["budget_kind"], "relative_name_bytes")
        self.assertGreater(result["attempted_value"], result["budget_limit"])

    def test_allocated_budget_uses_blocks_not_logical_payload_size(self):
        self.file("sparse", b"x")
        with patch.object(m, "MAXIMUM_ALLOCATED_BYTES", 0):
            result = self.report()
            with self.assertRaises(m.CensusError) as error:
                m.census(self.fd, self.helper, result)
        self.assertEqual(error.exception.code, "budget_hit")
        self.assertEqual(result["budget_kind"], "allocated_bytes")

    def test_depth_budget_preserves_old_boundary(self):
        current = self.root
        for _index in range(48):
            current /= "d"
            current.mkdir()
        (current / "leaf").write_bytes(b"x")
        with patch.object(m, "MAXIMUM_DEPTH", 48):
            result = self.report()
            with self.assertRaises(m.CensusError) as error:
                m.census(self.fd, self.helper, result)
        self.assertEqual(error.exception.code, "budget_hit")
        self.assertEqual(result["budget_kind"], "directory_depth")


class ReportAndReadOnlyTests(unittest.TestCase):
    def test_budget_report_is_closed_and_valid(self):
        value = m.empty_report()
        value.update(status="observed", stage="census", code="budget_hit",
                     budget_kind="nodes", budget_limit=m.MAXIMUM_NODES,
                     attempted_value=m.MAXIMUM_NODES + 1)
        m.validate_report(value)
        self.assertEqual(set(json.loads(m.canonical(value))), set(value))

    def test_failed_report_discards_budget_detail(self):
        value = m.empty_report()
        value.update(status="failed", stage="binding", code="binding_rejected",
                     budget_kind=None, budget_limit=None, attempted_value=None)
        m.validate_report(value)
        self.assertNotIn("private", m.canonical(value).decode())

    def test_audit_guard_rejects_writes_processes_and_unapproved_opens(self):
        events = [
            ("open", ("/tmp/x", None, os.O_WRONLY)),
            ("os.remove", ("/tmp/x",)),
            ("subprocess.Popen", ()),
            ("socket.__new__", ()),
            ("ctypes.dlopen", ("libc.so",)),
        ]
        for event, args in events:
            with self.subTest(event=event), self.assertRaises(m.CensusError) as error:
                m.audit_guard(event, args)
            self.assertEqual(error.exception.code, "read_only_rejected")

    def test_audit_guard_allows_only_directory_and_mountinfo_reads(self):
        m.audit_guard("open", ("/", None, os.O_RDONLY | os.O_DIRECTORY))
        m.audit_guard("open", ("/proc/self/mountinfo", "rb", os.O_RDONLY))
        with self.assertRaises(m.CensusError):
            m.audit_guard("open", ("/tmp/file", None, os.O_RDONLY))

    def test_runtime_and_cli_are_guarded(self):
        with patch.object(m, "runtime", side_effect=m.CensusError("runtime_rejected")), \
                patch.object(m, "sys", types.SimpleNamespace(argv=["diag", "bad"],
                                                               stdout=sys.stdout)):
            with patch.object(m.os, "write", side_effect=lambda _fd, raw: len(raw)):
                self.assertEqual(m.main(), 2)
        with patch.object(m, "runtime", side_effect=m.CensusError("runtime_rejected")):
            result = m.empty_report()
            self.assertEqual(result["status"], "failed")


class RunOrchestrationTests(unittest.TestCase):
    def setUp(self):
        m._ATTEMPTED = False
        self.directory = self.enterContext(tempfile.TemporaryDirectory(prefix="sdk-census-run-"))
        self.root = Path(self.directory) / "sdk"
        self.root.mkdir()
        (self.root / "payload").write_bytes(b"synthetic")
        self.fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        self.addCleanup(lambda: self._close(self.fd))
        self.parent_fd = os.open(Path(self.directory), os.O_RDONLY | os.O_DIRECTORY)
        self.addCleanup(lambda: self._close(self.parent_fd))
        self.helper = types.SimpleNamespace(
            PARTS=("sdk",),
            identity=lambda info: (
                info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            ),
            open_target=Mock(return_value=([self.parent_fd, self.fd],
                                            (os.fstat(self.fd).st_dev, os.fstat(self.fd).st_ino,
                                             os.fstat(self.fd).st_mode, os.fstat(self.fd).st_uid,
                                             os.fstat(self.fd).st_gid))),
            check_mounts=Mock(),
            ReclaimError=ValueError,
        )

    def test_run_is_observed_without_mutation_and_no_probe(self):
        with patch.object(m, "runtime"), patch.object(m, "load_helper", return_value=self.helper), \
                patch.object(m, "install_guard"), patch.object(m, "arm_deadline"), \
                patch.object(m, "disarm_deadline"):
            result = m.run()
        self.assertEqual(result["code"], "census_complete")
        self.assertTrue(result["census_complete"])
        self.assertTrue((self.root / "payload").exists())
        self.helper.check_mounts.assert_called_once_with()

    def test_budget_failure_is_observed_diagnostic_and_second_run_rejected(self):
        with patch.object(m, "MAXIMUM_NODES", 0), patch.object(m, "runtime"), \
                patch.object(m, "load_helper", return_value=self.helper), \
                patch.object(m, "install_guard"), patch.object(m, "arm_deadline"), \
                patch.object(m, "disarm_deadline"):
            result = m.run()
            self.assertEqual(result["code"], "budget_hit")
            self.assertEqual(result["status"], "observed")
            self.assertFalse(result["census_complete"])
            second = m.run()
        self.assertEqual(second["code"], "already_attempted")
        self.assertTrue((self.root / "payload").exists())

    @staticmethod
    def _close(fd):
        try:
            os.close(fd)
        except OSError:
            pass


class WorkflowContractTests(unittest.TestCase):
    def setUp(self):
        self.workflow_path = ROOT / ".github/workflows/sft-sdk-census-diagnostic-v1.yml"
        self.workflow = self.workflow_path.read_text()

    def test_exact_push_parent_and_single_job(self):
        trigger = self.workflow.split("\non:\n", 1)[1].split("\npermissions:\n", 1)[0]
        self.assertEqual(trigger, "  push:\n    branches:\n"
                         "      - codex/lumi-posttraining-product-v0.1\n    paths:\n"
                         "      - .github/workflows/sft-sdk-census-diagnostic-v1.yml\n")
        self.assertIn("github.repository_id == '1300831104'", self.workflow)
        self.assertIn("github.run_attempt == 1", self.workflow)
        self.assertIn("github.event.before == '7d6007defa8868c4276372411b7ae6ca1189ec75'",
                      self.workflow)
        self.assertEqual(self.workflow.count("    runs-on: ubuntu-24.04"), 1)
        self.assertIn("    timeout-minutes: 8\n", self.workflow)

    def test_exact_pins_and_read_only_entrypoint(self):
        pins = {
            "scripts/diagnose_sft_sdk_census_v1.py":
                "cd661fc6ef24fa6c91bc8465d175d8a8ffdbb42a480c84a1588513dfc5b33d21",
            "scripts/reclaim_sft_android_sdk_v1.py":
                "724cb2d2a6ff01dff790d77bf88434f2be15b0f43e2def32b9def25af5c74ba5",
        }
        for relative, digest in pins.items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest)
            self.assertIn(f"{digest}  {relative}", self.workflow)
        command = "/usr/bin/python3 -I -S -B scripts/diagnose_sft_sdk_census_v1.py"
        self.assertEqual(self.workflow.count(command), 1)
        self.assertIn(command + " --read-only-sdk-census", self.workflow)
        self.assertEqual(self.workflow.count("sudo -n /usr/bin/env -i"), 1)

    def test_synthetic_scope_and_no_mutation_commands(self):
        selected = __import__("re").findall(
            r"unittest discover -q -s tests -p (\S+)", self.workflow,
        )
        self.assertEqual(selected, ["test_sft_sdk_census_diagnostic_v1.py",
                                    "test_sft_sdk_capacity_preflight_v1.py",
                                    "test_sft_prepared_setup_preflight_v1.py"])
        for setting in ('CUDA_VISIBLE_DEVICES: ""', 'HF_HUB_OFFLINE: "1"',
                        'TRANSFORMERS_OFFLINE: "1"', 'AWS_EC2_METADATA_DISABLED: "true"'):
            self.assertIn(setting, self.workflow)
        for pattern in (r"\brm\s", r"\bcurl\s", r"\bwget\s", r"\bdocker\s",
                        r"\baws\s", r"\bpip\s", r"\bgit push\b", r"\bgit clean\b",
                        r"\bsetup-python", r"\bprobe-setup", r"\bfresh-sdk-capacity-check"):
            with self.subTest(pattern=pattern):
                self.assertNotRegex(self.workflow, pattern)

    def test_bounded_closed_aggregate_upload(self):
        target = "artifacts/lumi-sft-sdk-census-diagnostic-v0.1/aggregate-census.json"
        self.assertIn("> " + target + " 2>/dev/null", self.workflow)
        self.assertEqual(self.workflow.count("path: " + target), 1)
        self.assertIn("timeout --signal=TERM --kill-after=5s 300s", self.workflow)
        self.assertIn("if-no-files-found: error\n          retention-days: 14\n", self.workflow)
        self.assertEqual(self.workflow.count("actions/checkout@v6"), 1)
        self.assertEqual(self.workflow.count("actions/upload-artifact@v4"), 1)


if __name__ == "__main__":
    unittest.main()
