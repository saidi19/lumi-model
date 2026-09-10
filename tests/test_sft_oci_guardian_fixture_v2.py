"""Frozen-path equivalence and AST-only cleanup/report tests; no native execution."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import stat
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/run_sft_oci_guardian_fixture_v2.py"
OLD_SOURCE = ROOT / "scripts/run_sft_oci_guardian_fixture_v1.py"
TREE = ast.parse(SOURCE.read_text())
OLD_TREE = ast.parse(OLD_SOURCE.read_text())


def function(tree, name):
    return next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def constant(name):
    node = next(node for node in TREE.body if isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name)
    if isinstance(node.value, ast.Call):
        return frozenset(ast.literal_eval(node.value.args[0]))
    return ast.literal_eval(node.value)


def namespace(*names):
    definitions = [node for node in TREE.body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    selected = {"ERROR_CODES": constant("ERROR_CODES"), "ADMISSION": None, "ROOTFS": None,
                "json": json, "stat": stat}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "synthetic-v2", "exec"), selected)
    return selected


class SourceTests(unittest.TestCase):
    def test_frozen_original_bytes_unchanged(self):
        self.assertEqual(hashlib.sha256(OLD_SOURCE.read_bytes()).hexdigest(),
                         "38d64b4159bcc418ff799c14fae6cde00ec98b35b58cbfeb58fb14b2e83d8f76")

    def test_all_unchanged_functions_keep_exact_ast(self):
        allowed = {"run", "main", "guardian", "cases"}
        functions = [node.name for node in OLD_TREE.body if isinstance(node, ast.FunctionDef)]
        for name in functions:
            if name not in allowed:
                with self.subTest(function=name):
                    self.assertEqual(ast.dump(function(TREE, name), include_attributes=False),
                                     ast.dump(function(OLD_TREE, name), include_attributes=False))
        added = ({node.name for node in TREE.body if isinstance(node, ast.FunctionDef)}
                 - set(functions))
        self.assertEqual(added, {"load_admission", "failure_fields", "receive_with_stage"})

    def test_new_admission_is_hash_bound_before_native_work(self):
        digest = constant("ADMISSION_SHA256")
        self.assertRegex(digest, r"\A[0-9a-f]{64}\Z")
        raw = (ROOT / constant("ADMISSION_SOURCE")).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
        text = ast.unparse(function(TREE, "load_admission"))
        self.assertLess(text.index("read_source(ADMISSION_SOURCE, ADMISSION_SHA256)"),
                        text.index("exec("))
        run = ast.unparse(function(TREE, "run"))
        self.assertLess(run.index("load_admission()"), run.index("tempfile.mkdtemp"))
        self.assertNotIn("posttraining_sft_oci_admission_v1", run)

    def test_case_order_and_all_limits_remain_bounded(self):
        self.assertEqual(constant("CASES"), ("wrong_control", "wrong_adapter", "positive"))
        self.assertEqual(constant("MAXIMUM_SECONDS"), 900)
        for name in ("MAXIMUM_EXPORT_BYTES", "MAXIMUM_EXTRACT_BYTES", "ARCHIVE_BYTES",
                     "ARCHIVE_SHA256", "ADAPTER", "FROZEN", "FROZEN_SHA256",
                     "WORKER_PATH", "CONTROL_PATH", "MAXIMUM_WORKER_BYTES"):
            new = next(node for node in TREE.body if isinstance(node, ast.Assign)
                       and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name)
            old = next(node for node in OLD_TREE.body if isinstance(node, ast.Assign)
                       and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name)
            self.assertEqual(ast.dump(new.value), ast.dump(old.value))

    def test_new_environment_and_unique_owned_parent(self):
        run = ast.unparse(function(TREE, "run"))
        self.assertIn("LUMI_OCI_GUARDIAN_V2_CI", run)
        self.assertIn("prefix='lumi-oci-guardian-v2-'", run)
        self.assertIn("deadline = overall_deadline - 100", run)
        self.assertIn("min(overall_deadline, time.monotonic() + 95)", run)

    def test_created_container_never_starts_and_is_checked_twice(self):
        run = ast.unparse(function(TREE, "run"))
        self.assertEqual(run.count("admission.verify_created_oci("), 2)
        self.assertEqual(run.count("create_options("), 1)
        self.assertEqual(run.count("rootfs.extract_rootfs("), 1)
        self.assertEqual(run.count("cases("), 1)
        self.assertNotIn("'start'", run)
        self.assertNotIn("'run'", run)
        self.assertLess(run.index("retirement_completed = True"),
                        run.index("rootfs.extract_rootfs"))
        self.assertLess(run.index("cleanup_inventory['owned_container_absent'] = True"),
                        run.index("rootfs.extract_rootfs"))


class AdmissionError(ValueError):
    def __init__(self, code="profile_rejected", predicate="configuration_labels_equal"):
        self.code, self.predicate = code, predicate


class RootfsError(ValueError):
    def __init__(self, code="rootfs_entry_rejected"):
        self.code = code


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.ns = namespace("FixtureError", "need", "failure_fields", "canonical", "main")
        self.ns.update(LAST_BOOTSTRAP_STAGE="not_reported",
                       BOOTSTRAP_STAGES=constant("BOOTSTRAP_STAGES"))
        self.ns["ADMISSION"] = types.SimpleNamespace(
            OciAdmissionError=AdmissionError, ERROR_CODES={"profile_rejected"},
            PREDICATE_CODES={"configuration_labels_equal", "not_localized"})
        self.ns["ROOTFS"] = types.SimpleNamespace(
            RootfsError=RootfsError, ERROR_CODES={"rootfs_entry_rejected"})

    def test_exact_typed_admission_details_retained(self):
        self.assertEqual(self.ns["failure_fields"](AdmissionError()), {
            "code": "admission_rejected", "admission_code": "profile_rejected",
            "admission_predicate": "configuration_labels_equal"})

    def test_exact_typed_rootfs_code_retained(self):
        self.assertEqual(self.ns["failure_fields"](RootfsError()), {
            "code": "rootfs_rejected", "rootfs_code": "rootfs_entry_rejected"})

    def test_arbitrary_exception_attributes_or_subclasses_not_accepted(self):
        class Pretender(AdmissionError):
            pass

        for error in (Pretender(), ValueError("PRIVATE RAW TEXT"),
                      AdmissionError("PRIVATE CODE"), AdmissionError(predicate="PRIVATE FIELD"),
                      RootfsError("PRIVATE PATH")):
            self.assertEqual(self.ns["failure_fields"](error), {"code": "fixture_failed"})

    def test_fixture_code_survives_without_arbitrary_text(self):
        self.assertEqual(self.ns["failure_fields"](self.ns["FixtureError"]("command_failed")),
                         {"code": "command_failed"})
        self.assertEqual(self.ns["failure_fields"](self.ns["FixtureError"]("PRIVATE")),
                         {"code": "fixture_failed"})

    def test_main_preserves_codes_suppresses_raw_context_and_exits_two(self):
        output = io.BytesIO()
        self.ns.update({"SCHEMA": constant("SCHEMA"), "STAGE": "inspect_container",
                        "NONCLAIMS": constant("NONCLAIMS"), "MAXIMUM_WORKER_BYTES": 32768,
                        "Path": Path, "run": Mock(side_effect=AdmissionError()),
                        "sys": types.SimpleNamespace(argv=["fixture", "--synthetic-only", "/fake"],
                            stdout=types.SimpleNamespace(buffer=output))})
        with self.assertRaises(SystemExit) as caught:
            self.ns["main"]()
        self.assertEqual(caught.exception.code, 2)
        result = json.loads(output.getvalue())
        self.assertEqual(result["admission_predicate"], "configuration_labels_equal")
        self.assertTrue(all(result[key] is False for key in constant("NONCLAIMS")))
        self.assertNotIn(b"/fake", output.getvalue())

    def test_main_only_reports_allowlisted_bootstrap_stage(self):
        stages = (*sorted(constant("BOOTSTRAP_STAGES")), "PRIVATE STAGE", "not_reported", 7)
        for stage in stages:
            with self.subTest(stage=stage):
                output = io.BytesIO()
                self.ns.update({
                    "SCHEMA": constant("SCHEMA"), "STAGE": "cases",
                    "LAST_BOOTSTRAP_STAGE": stage, "NONCLAIMS": constant("NONCLAIMS"),
                    "MAXIMUM_WORKER_BYTES": 32768, "Path": Path,
                    "run": Mock(side_effect=ValueError("PRIVATE EXCEPTION")),
                    "sys": types.SimpleNamespace(
                        argv=["fixture", "--synthetic-only", "/fake"],
                        stdout=types.SimpleNamespace(buffer=output)),
                })
                with self.assertRaises(SystemExit) as caught:
                    self.ns["main"]()
                self.assertEqual(caught.exception.code, 2)
                result = json.loads(output.getvalue())
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["code"], "fixture_failed")
                if stage in constant("BOOTSTRAP_STAGES"):
                    self.assertEqual(result["reported_bootstrap_stage"], stage)
                else:
                    self.assertNotIn("reported_bootstrap_stage", result)
                self.assertNotIn(b"PRIVATE", output.getvalue())
                self.ns["run"].assert_called_once()


class BootstrapObservationTests(unittest.TestCase):
    def setUp(self):
        self.ns = namespace("receive_with_stage")
        self.ns.update(BOOTSTRAP_STAGES=constant("BOOTSTRAP_STAGES"),
                       LAST_BOOTSTRAP_STAGE="not_reported")

    def test_single_packet_observed_without_modification_or_reread(self):
        for stage in self.ns["BOOTSTRAP_STAGES"]:
            value = {"status": "failed", "case": "worker_run", "bootstrap_stage": stage}
            receive = self.ns["BASE_RECEIVE"] = Mock(return_value=value)
            self.assertIs(self.ns["receive_with_stage"]("channel", 123), value)
            self.assertEqual(self.ns["LAST_BOOTSTRAP_STAGE"], stage)
            receive.assert_called_once_with("channel", 123)

    def test_arbitrary_stage_wrong_case_or_passed_packet_not_retained(self):
        for changes in ({"bootstrap_stage": "PRIVATE"}, {"case": "other"},
                        {"status": "passed"}, {"bootstrap_stage": 1}):
            value = {"status": "failed", "case": "worker_run", "bootstrap_stage": "mount",
                     **changes}
            self.ns["BASE_RECEIVE"] = Mock(return_value=value)
            self.assertIs(self.ns["receive_with_stage"]("channel", 123), value)
            self.assertEqual(self.ns["LAST_BOOTSTRAP_STAGE"], "not_reported")

    def test_ready_policy_and_workload_body_stay_unchanged(self):
        old = function(OLD_TREE, "guardian")
        new = function(TREE, "guardian")
        old_need = next(node for node in ast.walk(old) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name) and node.func.id == "need"
                        and isinstance(node.args[0], ast.Compare)
                        and isinstance(node.args[0].comparators[0], ast.Dict)
                        and "ready" in ast.unparse(node.args[0].comparators[0]))
        new_need = next(node for node in ast.walk(new) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name) and node.func.id == "need"
                        and isinstance(node.args[0], ast.Compare)
                        and isinstance(node.args[0].left, ast.Name)
                        and node.args[0].left.id == "startup")
        self.assertEqual(ast.dump(old_need.args[0].comparators[0]),
                         ast.dump(new_need.args[0].comparators[0]))
        self.assertEqual(ast.dump(function(TREE, "workload")),
                         ast.dump(function(OLD_TREE, "workload")))


class SyntheticExit(BaseException):
    def __init__(self, code):
        self.code = code


class GuardianLocalizationTests(unittest.TestCase):
    """Execute only guardian AST with every OS/process/cgroup operation replaced."""

    def setUp(self):
        self.ns = namespace("FixtureError", "need", "guardian")
        self.ready = {"phase": "ready", "isolation_installed": True,
                      "uid_gid_dropped": True, "authority_fds_closed": True}
        self.channel = types.SimpleNamespace(fileno=Mock(return_value=11), close=Mock())
        self.pair = types.SimpleNamespace(send=Mock(return_value=1), close=Mock())
        self.child = types.SimpleNamespace(close=Mock())
        self.kill = Mock()
        self.lease = types.SimpleNamespace(
            kill=self.kill, verify_limits=Mock(), fileno=Mock(return_value=42), close=Mock(),
            snapshot=Mock(side_effect=[{"populated": 0}, {"populated": 1, "pids_current": 1},
                                       {"populated": 0, "pids_current": 0}]))

        def watch(lease, *, release, **_kwargs):
            self.assertIs(lease, self.lease)
            release()
            lease.kill()
            return {"reason": "workload_empty", "parent_pidfd_bound": True,
                    "release_called": True, "kill_attempted": True,
                    "empty_verified": True, "pidfd_closed": True}

        def exit_process(code):
            raise SyntheticExit(code)

        self.watch = Mock(side_effect=watch)
        self.create_lease = Mock(return_value=self.lease)
        modules = {
            "posttraining_sft_cgroup_v1": types.SimpleNamespace(CgroupLease=self.create_lease),
            "posttraining_sft_cgroup_guardian_v1": types.SimpleNamespace(watch_parent=self.watch),
        }
        self.base = types.SimpleNamespace(
            close_except=Mock(), subreaper=Mock(), module=Mock(side_effect=modules.__getitem__),
            receive=Mock(return_value=self.ready), send=Mock(),
            wait_statuses=Mock(return_value={1234: 0}))
        self.os = types.SimpleNamespace(
            O_RDONLY=0, O_DIRECTORY=1, O_NOFOLLOW=2, O_CLOEXEC=4, O_NONBLOCK=8,
            open=Mock(return_value=41), close=Mock(), fork=Mock(return_value=1234),
            _exit=Mock(side_effect=exit_process))
        self.ns.update({
            "BASE": self.base, "BOOTSTRAP_STAGES": constant("BOOTSTRAP_STAGES"),
            "CONTEXT": {"case": "positive"}, "os": self.os,
            "socket": types.SimpleNamespace(AF_UNIX=1, SOCK_SEQPACKET=5,
                socketpair=Mock(return_value=(self.pair, self.child))),
            "uuid": types.SimpleNamespace(uuid4=Mock(return_value=types.SimpleNamespace(hex="a"))),
            "time": types.SimpleNamespace(monotonic_ns=lambda: 1000, sleep=Mock()),
            "workload": Mock(side_effect=AssertionError("native workload forbidden")),
            "worker_message": Mock(return_value={"generated": True}),
            "validate_worker": Mock(return_value={"status": "verified"}),
        })

    def invoke(self, expected_exit):
        with self.assertRaises(SyntheticExit) as caught:
            self.ns["guardian"]("worker_run", self.channel, 777, 10001, 10001,
                                "/synthetic/runtime", 1_000_000)
        self.assertEqual(caught.exception.code, expected_exit)
        self.os.fork.assert_called_once_with()
        self.ns["workload"].assert_not_called()
        self.base.receive.assert_called_once_with(self.pair, 1_000_000)
        self.kill.assert_called_once_with()
        self.lease.close.assert_called_once_with()
        self.base.wait_statuses.assert_called_once()
        self.pair.close.assert_called_once_with()
        self.child.close.assert_called_once_with()
        self.channel.close.assert_called_once_with()
        self.os.close.assert_called_once_with(41)
        return self.base.send.call_args.args[1]

    def test_failed_bootstrap_preserves_stage_without_release_and_cleans_once(self):
        for stage in sorted(constant("BOOTSTRAP_STAGES")):
            with self.subTest(stage=stage):
                self.setUp()
                self.base.receive.return_value = {"phase": "bootstrap_failed", "stage": stage}
                self.lease.snapshot.side_effect = None
                self.lease.snapshot.return_value = {"populated": 0, "pids_current": 0}
                final = self.invoke(2)
                self.assertEqual(final, {
                    "status": "failed", "case": "worker_run", "code": "worker_rejected",
                    "empty_and_removed": True, "bootstrap_stage": stage})
                self.base.send.assert_called_once_with(self.channel, final)
                self.watch.assert_not_called()
                self.pair.send.assert_not_called()
                self.ns["worker_message"].assert_not_called()
                self.ns["validate_worker"].assert_not_called()

    def test_ready_success_preserves_receipt_release_and_cleanup(self):
        final = self.invoke(0)
        self.assertEqual(final, {
            "status": "passed", "case": "worker_run", "empty_and_removed": True,
            "workload_processes_reaped": 1, "worker": {"status": "verified"},
            "guardian_parent_bound": True, "owned_cgroup_kill_attempts": 1})
        self.assertEqual(self.base.send.call_count, 2)
        self.assertEqual(self.base.send.call_args_list[0].args,
                         (self.channel, {"phase": "armed", "case": "worker_run"}))
        self.pair.send.assert_called_once_with(b"G")
        self.watch.assert_called_once()
        self.ns["worker_message"].assert_called_once_with(self.pair)
        self.ns["validate_worker"].assert_called_once_with({"generated": True}, "positive", 0)


class OwnedPath:
    def __init__(self, inode):
        self.inode = inode
        self.unlink, self.rmdir = Mock(), Mock()

    def lstat(self):
        return types.SimpleNamespace(st_dev=1, st_ino=self.inode, st_mode=stat.S_IFREG | 0o400,
                                     st_uid=0, st_gid=0)


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.ns = namespace("FixtureError", "need", "identity")
        run = function(TREE, "run")
        cleanup = next(node for node in run.body if isinstance(node, ast.Try)).finalbody
        self.code = compile(ast.Module(body=cleanup, type_ignores=[]),
                            "synthetic-cleanup-v2", "exec")
        self.file, self.directory, self.parent = OwnedPath(1), OwnedPath(2), OwnedPath(3)
        ident = self.ns["identity"]
        self.ns.update({
            "STAGE": "inspect_container", "overall_deadline": 1090.0,
            "time": types.SimpleNamespace(monotonic=lambda: 1000.0),
            "retirement_started": False, "retirement_completed": False,
            "create_started": True, "create_completed": True, "load_started": True,
            "load_completed": True, "container_name": "owned-container", "config": "private",
            "admission": types.SimpleNamespace(IMAGE_ID="owned-image"),
            "container_inventory": Mock(side_effect=[{"owned-container"}, set()]),
            "image_inventory": Mock(side_effect=[{"owned-image"}, set()]), "command": Mock(),
            "DESCENDANTS_REAPED": True, "owned": types.SimpleNamespace(close=Mock()),
            "file_ids": {self.file: ident(self.file.lstat())},
            "directory_ids": {self.directory: ident(self.directory.lstat())},
            "parent": self.parent, "parent_id": ident(self.parent.lstat()), "cleanup": False,
            "cleanup_inventory": {"owned_container_absent": False, "owned_image_absent": False},
        })

    def execute(self):
        exec(self.code, self.ns)

    def reject(self):
        with self.assertRaises(self.ns["FixtureError"]) as caught:
            self.execute()
        self.assertEqual(caught.exception.code, "cleanup_unconfirmed")
        self.assertFalse(self.ns["cleanup"])

    def test_success_requires_post_removal_inventory_and_reaped_descendants(self):
        self.execute()
        self.assertTrue(self.ns["cleanup"])
        self.assertTrue(all(self.ns["cleanup_inventory"].values()))
        self.assertEqual(self.ns["container_inventory"].call_count, 2)
        self.assertEqual(self.ns["image_inventory"].call_count, 2)
        self.assertEqual(self.ns["command"].call_count, 2)
        self.ns["owned"].close.assert_called_once()
        self.file.unlink.assert_called_once()
        self.parent.rmdir.assert_called_once()
        self.assertEqual(self.ns["STAGE"], "inspect_container")

    def test_shared_cleanup_deadline_never_exceeds_overall(self):
        self.execute()
        for call in self.ns["command"].call_args_list:
            self.assertLessEqual(call.kwargs["deadline"], 1090.0)
        for key in ("container_inventory", "image_inventory"):
            for call in self.ns[key].call_args_list:
                self.assertLessEqual(call.args[1], 1090.0)

    def test_ambiguous_daemon_writes_never_become_confirmed_cleanup(self):
        for key in ("create_completed", "load_completed"):
            with self.subTest(key=key):
                self.setUp()
                self.ns[key] = False
                self.reject()
                self.assertEqual(self.ns["command"].call_count, 2)
                self.parent.rmdir.assert_not_called()

    def test_successful_removal_without_absence_is_not_retried(self):
        for key, item in (("container_inventory", "owned-container"),
                          ("image_inventory", "owned-image")):
            with self.subTest(key=key):
                self.setUp()
                self.ns[key].side_effect = [{item}, {item}]
                self.reject()
                self.assertEqual(self.ns["command"].call_count, 2)

    def test_post_removal_inventory_failure_does_not_retry_removal(self):
        for key, item in (("container_inventory", "owned-container"),
                          ("image_inventory", "owned-image")):
            with self.subTest(key=key):
                self.setUp()
                self.ns[key].side_effect = [{item}, OSError("PRIVATE INVENTORY ERROR")]
                self.reject()
                self.assertEqual(self.ns[key].call_count, 2)
                self.assertEqual(self.ns["command"].call_count, 2)
                self.assertEqual([call.args[1] for call in self.ns["command"].call_args_list],
                                 [["rm", "owned-container"],
                                  ["image", "rm", "--no-prune", "owned-image"]])
                failed_proof = ("owned_container_absent" if key == "container_inventory"
                                else "owned_image_absent")
                self.assertFalse(self.ns["cleanup_inventory"][failed_proof])
                self.parent.rmdir.assert_not_called()

    def test_failed_removal_is_not_retried(self):
        self.ns["command"].side_effect = [OSError("PRIVATE"), None]
        self.reject()
        self.assertEqual(self.ns["command"].call_count, 2)

    def test_unreaped_descendants_prevent_rootfs_retirement(self):
        self.ns["DESCENDANTS_REAPED"] = False
        self.reject()
        self.ns["owned"].close.assert_not_called()
        self.parent.rmdir.assert_not_called()

    def test_changed_file_identity_preserves_replacement(self):
        self.file.inode = 900
        self.reject()
        self.file.unlink.assert_not_called()

    def test_changed_directory_identity_preserves_replacement(self):
        self.directory.inode = 901
        self.reject()
        self.directory.rmdir.assert_not_called()

    def test_rootfs_cleanup_failure_remains_closed(self):
        self.ns["owned"].close.side_effect = OSError("PRIVATE")
        self.reject()

    def test_ambiguous_prior_retirement_does_not_trigger_another_rm(self):
        self.ns.update(retirement_started=True, retirement_completed=False, create_started=False)
        self.reject()
        self.ns["container_inventory"].assert_not_called()
        self.assertEqual(self.ns["command"].call_count, 1)


if __name__ == "__main__":
    unittest.main()
