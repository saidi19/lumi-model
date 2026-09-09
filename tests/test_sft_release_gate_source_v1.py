"""Source/literal inspection only: never import or execute a native fixture.

These checks bind the generated ordering experiment, not a real runtime tree.
Only repository source is read; no subprocess, runtime, data or model is opened.
"""

from __future__ import annotations

import ast
import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "scripts/run_sft_release_gate_fixture_v1.py"
WRAPPER = ROOT / "tests/test_sft_release_gate_linux_v1.py"
FROZEN = {
    "scripts/run_sft_ipc_fixture_v1.py": (
        25095, "88a5e703621fa796a7f110887ea360a0954a3880038f1620f4c7130634677466"),
    "scripts/run_sft_filesystem_fixture_v1.py": (
        24179, "31c3f271a15d412419fd0609497f92bc7bb0a4bf08f9a591e7086c956436ad89"),
    "scripts/run_sft_netns_fixture_v1.py": (
        16067, "70400a1428d0e54834721a5163db7724940a271bbeeeaec50714f28108f20bfc"),
    "scripts/run_sft_cgroup_fixture_v1.py": (
        27839, "f91ed0c353a20c309ff3e1bfaab8dbe990b28c36626fdeeb0ac5131e0cf75fad"),
    "scripts/run_sft_worker_run_torch_fixture_v1.py": (
        39948, "53c50c92965ed1cef0c9624c57367c31ed7ebfb9d7cfb34257188ad179d333d0"),
}


def source(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def literal(tree, name):
    matches = [node.value for node in tree.body if isinstance(node, ast.Assign)
               and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
               and node.targets[0].id == name]
    if len(matches) != 1:
        raise AssertionError("expected one top-level literal binding: " + name)
    return ast.literal_eval(matches[0])


def function(tree, name):
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
               and node.name == name]
    if len(matches) != 1:
        raise AssertionError("expected one function definition: " + name)
    return matches[0]


def calls(tree, name):
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and ast.unparse(node.func) == name]


def expression(text):
    return ast.dump(ast.parse(text, mode="eval").body, include_attributes=False)


def equal_expression(node, text):
    return ast.dump(node, include_attributes=False) == expression(text)


class ReleaseGateSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = source(FIXTURE)

    def assert_need(self, node, text):
        self.assertTrue(any(len(call.args) == 1 and equal_expression(call.args[0], text)
                            for call in calls(node, "need")), text)

    def test_all_accepted_fixture_bytes_stay_frozen(self):
        for relative, (size, digest) in FROZEN.items():
            with self.subTest(source=relative):
                raw = (ROOT / relative).read_bytes()
                self.assertEqual(len(raw), size)
                self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)

    def test_frozen_loader_binds_accepted_ipc_and_checks_before_compiling(self):
        self.assertEqual(literal(self.fixture, "FROZEN_SIZE"), 25095)
        self.assertEqual(literal(self.fixture, "FROZEN_SHA256"),
                         FROZEN["scripts/run_sft_ipc_fixture_v1.py"][1])
        loader = function(self.fixture, "load_frozen")
        self.assertEqual(len(calls(loader, "os.open")), 1)
        self.assertTrue(equal_expression(calls(loader, "os.open")[0].args[1],
                                        "os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW"))
        checks = [node for node in ast.walk(loader) if isinstance(node, ast.If)
                  and "hashlib.sha256(raw).hexdigest()" in ast.unparse(node.test)]
        self.assertEqual(len(checks), 1)
        self.assertEqual(len(calls(loader, "compile")), 1)
        compiled = calls(loader, "compile")[0]
        self.assertTrue(equal_expression(compiled.args[0], "raw"))
        self.assertLess(checks[0].lineno, compiled.lineno)
        self.assertTrue(calls(loader, "os.close"))

    def test_manifest_is_independently_fixed_not_hashed_from_scanned_bytes(self):
        declared = literal(self.fixture, "DECLARED")
        raw = json.dumps(declared, ensure_ascii=True, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("ascii")
        self.assertEqual(len(raw), 600)
        self.assertEqual(literal(self.fixture, "PIN"),
                         "2c83d57f86fe861979fcc74b342e0981281ab4465437989e3eae3302fa23e112")
        self.assertEqual(hashlib.sha256(raw).hexdigest(), literal(self.fixture, "PIN"))
        self.assertEqual(set(declared),
                         {"schema", "platform", "python_abi", "interpreter", "entrypoint", "files"})
        self.assertEqual(declared["schema"], "lumi.sft-release.v0.1")
        self.assertEqual((declared["platform"], declared["python_abi"]), ("linux-x86_64", "cp312"))

    def test_three_stand_ins_match_exact_sizes_hashes_roles_and_modes(self):
        declared = literal(self.fixture, "DECLARED")
        payloads = literal(self.fixture, "PAYLOADS")
        expected = {
            "dependencies/declared.bin": ("dependency", 30, 0o444),
            "runtime/python3.12": ("runtime", 37, 0o555),
            "source/worker.py": ("source", 39, 0o444),
        }
        self.assertEqual(set(payloads), set(expected))
        self.assertEqual([row["path"] for row in declared["files"]], sorted(expected))
        self.assertEqual(sum(map(len, payloads.values())), 106)
        for row in declared["files"]:
            with self.subTest(path=row["path"]):
                self.assertEqual(set(row), {"path", "role", "bytes", "mode", "sha256"})
                self.assertEqual((row["role"], row["bytes"], row["mode"]), expected[row["path"]])
                self.assertEqual(len(payloads[row["path"]]), row["bytes"])
                self.assertEqual(hashlib.sha256(payloads[row["path"]]).hexdigest(), row["sha256"])
        self.assertEqual(declared["interpreter"], "runtime/python3.12")
        self.assertEqual(declared["entrypoint"], "source/worker.py")
        self.assertTrue(payloads[declared["interpreter"]].startswith(b"\x7fELF"))
        self.assertIn(b"NONEXECUTABLE-STAND-IN", payloads[declared["interpreter"]])
        self.assertIn(b"NEVER-IMPORT", payloads[declared["entrypoint"]])

    def test_manifest_and_tree_read_limits_are_exact_and_small(self):
        self.assertEqual(literal(self.fixture, "LIMITS"), {
            "maximum_manifest_bytes": 600, "maximum_entries": 3, "maximum_file_bytes": 39,
            "maximum_total_bytes": 106, "maximum_tree_entries": 6,
            "maximum_content_read_bytes": 109, "maximum_opens": 64,
            "maximum_runtime_ms": 1000,
        })

    def test_tamper_changes_exactly_thirty_dependency_bytes_not_declared_manifest(self):
        build = function(self.fixture, "build")
        branches = [node for node in ast.walk(build) if isinstance(node, ast.If)
                    and equal_expression(node.test, "MODE == 'tamper'")]
        self.assertEqual(len(branches), 1)
        branch = branches[0]
        self.assert_need(branch, "os.write(descriptor, b'X' * 30) == 30")
        self.assertTrue(any(isinstance(node, ast.Assign) and
                            equal_expression(node.value,
                                             "self.path + '/dependencies/declared.bin'")
                            for node in branch.body))
        assignments = [ast.unparse(target) for node in ast.walk(branch)
                       if isinstance(node, ast.Assign) for target in node.targets]
        self.assertEqual(set(assignments), {"path", "descriptor"})
        self.assertFalse(calls(branch, "os.chmod"))
        self.assertFalse(calls(branch, "os.fchmod"))

    def test_native_verifier_preflight_precedes_fixture_bootstrap(self):
        run = function(self.fixture, "run")
        native = calls(run, "base().module('posttraining_sft_release_verifier_v1')._runtime")
        self.assertEqual(len(native), 1)
        self.assertEqual(len(calls(run, "ORIGINAL_RUN")), 1)
        self.assertLess(native[0].lineno, calls(run, "ORIGINAL_RUN")[0].lineno)
        self.assert_need(run, "sys.flags.isolated == 1 and os.getuid() == os.getgid() == 0")
        main = function(self.fixture, "main")
        rejects = [node for node in ast.walk(main) if isinstance(node, ast.If)
                   and "sys.platform" in ast.unparse(node.test)]
        self.assertEqual(len(rejects), 1)
        self.assertTrue(equal_expression(rejects[0].test,
            "sys.platform != 'linux' or sys.version_info[:2] != (3, 12) "
            "or not sys.flags.isolated or not sys.flags.no_site "
            "or not sys.dont_write_bytecode or os.getuid() != 0"))
        self.assertLess(rejects[0].lineno, calls(main, "load_frozen")[0].lineno)

    def test_native_runtime_checks_and_verifier_are_never_patched(self):
        forbidden = {"_runtime", "verify_release", "sys", "threading", "platform"}
        for node in ast.walk(self.fixture):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    for child in ast.walk(target):
                        if isinstance(child, ast.Attribute):
                            self.assertNotIn(child.attr, forbidden)
            if isinstance(node, ast.Call):
                self.assertNotIn(ast.unparse(node.func), {"setattr", "patch", "mock.patch"})

    def test_verified_release_wraps_the_actual_guardian_callback(self):
        watch = function(self.fixture, "watch_parent")
        constructed = calls(watch, "gate.VerifiedRelease")
        self.assertEqual(len(constructed), 1)
        keywords = {item.arg: ast.unparse(item.value) for item in constructed[0].keywords}
        self.assertEqual(keywords, {
            "root": "TREE.path", "raw_manifest": "base().canonical(DECLARED)",
            "expected_sha256": "'0' * 64 if MODE == 'wrong_pin' else PIN", "limits": "LIMITS",
            "expected_uid": "0", "expected_gid": "0",
            "expected_parent_pid": "expected_parent_pid", "deadline_ns": "deadline_ns",
            "release": "release",
        })
        delegated = calls(watch, "ORIGINAL_WATCH")
        self.assertEqual(len(delegated), 1)
        self.assertEqual(ast.unparse(delegated[0].args[0]), "lease")
        self.assertEqual({item.arg: ast.unparse(item.value) for item in delegated[0].keywords}, {
            "expected_parent_pid": "expected_parent_pid", "deadline_ns": "deadline_ns",
            "cleanup_timeout_ms": "cleanup_timeout_ms", "release": "ACTIVE_GATE",
        })
        self.assertLess(constructed[0].lineno, delegated[0].lineno)
        self.assertFalse(calls(watch, "release"))
        self.assertFalse(calls(watch, "ACTIVE_GATE"))

    def test_report_comes_from_gate_status_not_caller_supplied_success_receipt(self):
        send = function(self.fixture, "send")
        self.assertEqual(len(calls(send, "ACTIVE_GATE.status")), 1)
        self.assertTrue(any(isinstance(node, ast.If) and equal_expression(node.test,
                            "'empty_and_removed' in value and ACTIVE_GATE is not None")
                            for node in send.body))
        self.assertFalse(calls(send, "gate.VerifiedRelease"))
        self.assertFalse(calls(send, "verify_release"))
        self.assertEqual(len(calls(send, "ORIGINAL_SEND")), 1)
        self.assert_need(function(self.fixture, "receive"),
                         "type(value.get('release_gate')) is dict "
                         "or value.get('status') == 'failed'")

    def test_four_original_positive_cases_precede_two_fresh_worker_negatives(self):
        run = function(self.fixture, "run")
        original = calls(run, "ORIGINAL_RUN")[0]
        negative_loops = [node for node in ast.walk(run) if isinstance(node, ast.For)
                          and equal_expression(node.iter, "('wrong_pin', 'tamper')")]
        self.assertEqual(len(negative_loops), 1)
        self.assertLess(original.lineno, negative_loops[0].lineno)
        self.assert_need(run, "len(POSITIVE_GATES) == IPC.INSTALLATIONS_VERIFIED "
                             "== TREES_REMOVED == 4")
        self.assert_need(run, "len(NEGATIVE_GATES) == 2 "
                             "and IPC.INSTALLATIONS_VERIFIED == TREES_REMOVED == 6")
        negative_calls = calls(negative_loops[0], "base().run_case")
        self.assertEqual(len(negative_calls), 1)
        self.assertEqual([ast.unparse(arg) for arg in negative_calls[0].args],
                         ["'worker_run'", "uid", "gid", "binding['runtime_path']", "RUN_DEADLINE"])

    def test_rejection_requires_zero_completed_verification_or_release_counts(self):
        validate = function(self.fixture, "validate_gate")
        expected = [node.value for node in ast.walk(validate) if isinstance(node, ast.Assign)
                    and any(ast.unparse(target) == "expected" for target in node.targets)]
        self.assertEqual(len(expected), 1)
        self.assertTrue(equal_expression(expected[0],
                                        "(1, 0, 0, 0) if negative else (1, 1, 1, 1)"))
        self.assert_need(validate, "type(summary[field]) is int and summary[field] == count")
        self.assert_need(validate, "summary['state'] == ('failed' if negative else 'released')")
        self.assert_need(validate, "summary['code'] == "
                         "({'wrong_pin': 'manifest_hash_mismatch', 'tamper': 'digest_mismatch'}"
                         "[MODE] if negative else 'completed')")
        self.assert_need(validate, "value['retirements'] == 1 "
                         "and value['descriptor_balance'] is True")

    def test_rejected_blocked_worker_must_be_sigkill_reaped_once(self):
        waited = function(self.fixture, "wait_statuses")
        self.assert_need(waited, "REAPED == 0 and len(result) == 1")
        self.assert_need(waited, "os.WIFSIGNALED(status) "
                         "and os.WTERMSIG(status) == signal.SIGKILL")
        self.assert_need(function(self.fixture, "retire_once"), "RETIREMENTS == 1")
        validate = function(self.fixture, "validate_gate")
        self.assert_need(validate, "value['blocked_worker_reaped'] == (1 if negative else 0) "
                         "and value['blocked_worker_sigkill'] is negative")

    def test_negative_controller_failure_is_checked_then_reaped_with_exact_status(self):
        negative = function(self.fixture, "negative_case")
        self.assert_need(negative, "case == 'worker_run' and MODE in {'wrong_pin', 'tamper'}")
        self.assert_need(negative, "final.get('status') == 'failed' "
                         "and final.get('case') == 'worker_run' "
                         "and final.get('code') == 'fixture_failed' "
                         "and final.get('component_code') == 'guardian_failed' "
                         "and base().REMOVAL_REPORTED")
        self.assert_need(negative, "set(statuses) == {controller_pid} "
                         "and os.WIFEXITED(statuses[controller_pid]) "
                         "and os.WEXITSTATUS(statuses[controller_pid]) == 2")
        self.assertEqual(len(calls(negative, "os.pidfd_open")), 1)
        self.assertEqual(len(calls(negative, "signal.pidfd_send_signal")), 1)
        self.assertEqual(len(calls(negative, "os.close")), 1)

    def test_generated_tree_cleanup_checks_identity_before_nonrecursive_deletion(self):
        close = function(self.fixture, "close")
        self.assert_need(close, "not self.closed")
        self.assert_need(close, "identity(path) == expected")
        self.assertEqual(len(calls(close, "os.rmdir")), 1)
        self.assertEqual(len(calls(close, "os.unlink")), 1)
        loops = [node for node in ast.walk(close) if isinstance(node, ast.For)]
        self.assertEqual(len(loops), 1)
        self.assertTrue(equal_expression(loops[0].iter, "reversed(self.owned)"))
        consumed = [node for node in close.body if isinstance(node, ast.Assign)
                    and any(ast.unparse(target) == "self.closed" for target in node.targets)]
        self.assertEqual(len(consumed), 1)
        self.assertEqual(ast.literal_eval(consumed[0].value), True)
        self.assertLess(consumed[0].lineno, loops[0].lineno)
        case = function(self.fixture, "run_case")
        self.assertTrue(any(calls(node, "TREE.close") for node in ast.walk(case)
                            if isinstance(node, ast.Try) and node.finalbody))

    def test_success_claims_are_narrow_and_real_execution_stays_false(self):
        expected_nonclaims = {
            "generated_release_is_running_runtime", "runtime_origin_authenticated",
            "dynamic_dependency_completeness_verified", "point_in_time_snapshot_verified",
            "immutable_during_execution", "real_host_admission_verified",
            "real_sft_execution_permitted", "production_eligible",
        }
        self.assertEqual(set(literal(self.fixture, "NONCLAIMS")), expected_nonclaims)
        run = function(self.fixture, "run")
        self.assert_need(run, "'torch' not in sys.modules")
        updates = calls(run, "report.update")
        self.assertEqual(len(updates), 1)
        projections = [node.value for node in run.body if isinstance(node, ast.Assign)
                       and isinstance(node.value, ast.BinOp)]
        self.assertEqual(len(projections), 2)
        self.assertTrue(any(equal_expression(node,
                            "{key: report[key] for key in selected_fields} | "
                            "{'schema_version': SCHEMA, 'status': 'passed', "
                            "**dict.fromkeys(IPC.NONCLAIMS, False)}") for node in projections))
        self.assertEqual({item.arg: ast.literal_eval(item.value)
                          for item in updates[0].keywords}, {
            "native_release_gate_context_verified": True,
            "release_before_worker_order_verified": True,
            "positive_release_gates_verified": 4, "rejected_release_gates_verified": 2,
            "rejected_release_calls": 0, "rejected_worker_run_invocations": 0,
            "wrong_pin_rejected": True, "same_size_tamper_rejected": True,
            "release_gate_total_cgroups_removed": 6, "release_gate_total_trees_removed": 6,
            "rejected_blocked_workers_reaped": 2, "rejected_controllers_reaped": 2,
            "rejected_cases_seccomp_installations": 2,
            "rejected_cases_visible_socket_removals": 4,
            "release_gate_descriptor_balance_verified": True,
        })
        self.assert_need(run, "len(base().canonical(report)) + 1 <= base().MAXIMUM_MESSAGE_BYTES")

    def test_identity_registration_precedes_chmod_and_payload_write(self):
        build = function(self.fixture, "build")
        files = [node for node in build.body if isinstance(node, ast.For)
                 and equal_expression(node.iter, "DECLARED['files']")]
        self.assertEqual(len(files), 1)
        register = calls(files[0], "self.owned.append")
        self.assertEqual(len(register), 1)
        self.assertLess(register[0].lineno, calls(files[0], "os.fchmod")[0].lineno)
        self.assertLess(register[0].lineno, calls(files[0], "os.write")[0].lineno)
        identity = function(self.fixture, "identity")
        returned = [node for node in identity.body if isinstance(node, ast.Return)]
        self.assertEqual(len(returned), 1)
        self.assertTrue(equal_expression(returned[0].value,
            "(info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode), info.st_uid, info.st_gid)"))

    def test_no_recursive_removal_global_cgroup_mutation_or_new_worker_exec(self):
        prohibited = {"rmtree", "remove_tree", "system", "popen", "execve", "execv", "spawnv",
                      "chown", "chmod", "Popen", "run", "check_call", "check_output"}
        for node in ast.walk(self.fixture):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, prohibited)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                self.assertNotIn("cgroup.subtree_control", node.value)
                self.assertNotIn("/sys/fs/cgroup", node.value)
                self.assertNotIn("s3://", node.value)
        allowed_imports = {"__future__", "errno", "hashlib", "importlib.util", "os", "select",
                           "signal", "socket", "stat", "sys", "tempfile", "time", "pathlib"}
        modules = set()
        for node in ast.walk(self.fixture):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.add(node.module)
        self.assertEqual(modules, allowed_imports)


class ReleaseGateWrapperSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrapper = source(WRAPPER)

    def test_only_nonlinux_is_skipped_not_failed_linux_prerequisites(self):
        skips = calls(self.wrapper, "unittest.skipUnless")
        self.assertEqual(len(skips), 1)
        self.assertTrue(equal_expression(skips[0].args[0], "sys.platform == 'linux'"))
        self.assertFalse(calls(self.wrapper, "unittest.skipIf"))
        self.assertFalse(calls(self.wrapper, "self.skipTest"))

    def test_root_invocation_uses_system_python_isolated_no_site_no_bytecode(self):
        popens = calls(self.wrapper, "subprocess.Popen")
        self.assertEqual(len(popens), 1)
        command = popens[0].args[0]
        self.assertIsInstance(command, ast.List)
        self.assertEqual([ast.literal_eval(value) for value in command.elts[:6]],
                         ["/usr/bin/sudo", "-n", "/usr/bin/python3", "-I", "-S", "-B"])
        self.assertTrue(equal_expression(command.elts[6],
                         "str(ROOT / 'scripts/run_sft_release_gate_fixture_v1.py')"))
        self.assertEqual(ast.literal_eval(command.elts[7]), "--synthetic-only")
        keywords = {item.arg: item.value for item in popens[0].keywords}
        self.assertEqual(ast.literal_eval(keywords["env"]),
                         {"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"})
        self.assertTrue(equal_expression(keywords["stderr"], "subprocess.DEVNULL"))
        self.assertEqual(ast.literal_eval(keywords["start_new_session"]), True)

    def test_wrapper_has_one_bounded_run_and_no_retry_or_cleanup_authority(self):
        communicate = calls(self.wrapper, "process.communicate")
        self.assertEqual(len(communicate), 1)
        self.assertEqual({item.arg: ast.unparse(item.value) for item in communicate[0].keywords},
                         {"input": "payload", "timeout": "300"})
        self.assertFalse(calls(self.wrapper, "process.kill"))
        self.assertFalse(calls(self.wrapper, "os.kill"))
        self.assertFalse(calls(self.wrapper, "shutil.rmtree"))
        self.assertTrue(any(len(call.args) >= 2
                            and equal_expression(call.args[0], "len(stdout)")
                            and equal_expression(call.args[1], "4096")
                            for call in calls(self.wrapper, "self.assertLessEqual")))

    def test_wrapper_checks_selected_accepted_fields_and_full_ipc_nonclaims(self):
        old = source(ROOT / "tests/test_sft_ipc_linux_v1.py")
        def checked_fields(tree):
            fields = {True: set(), False: set()}
            for node in ast.walk(tree):
                if not isinstance(node, ast.For) or not isinstance(node.iter, ast.Tuple):
                    continue
                for selected in calls(node, "self.assertIs"):
                    if (len(selected.args) >= 2 and isinstance(selected.args[1], ast.Constant)
                            and type(selected.args[1].value) is bool):
                        fields[selected.args[1].value].update(ast.literal_eval(node.iter))
            return fields
        accepted, updated = checked_fields(old), checked_fields(self.wrapper)
        selected = set(literal(function(source(FIXTURE), "run"), "selected_fields"))
        additions = {"native_release_gate_context_verified", "release_before_worker_order_verified",
                     "wrong_pin_rejected", "same_size_tamper_rejected",
                     "release_gate_descriptor_balance_verified"}
        self.assertEqual(updated[True], (accepted[True] & selected) | additions)
        self.assertEqual(updated[False],
                         set(literal(source(ROOT / "scripts/run_sft_ipc_fixture_v1.py"),
                                     "NONCLAIMS")) | set(literal(source(FIXTURE), "NONCLAIMS")))

    def test_wrapper_checks_old_optimizer_counts_and_new_rejection_cleanup_counts(self):
        tables = [ast.literal_eval(node.iter.func.value) for node in ast.walk(self.wrapper)
                  if isinstance(node, ast.For) and isinstance(node.iter, ast.Call)
                  and isinstance(node.iter.func, ast.Attribute) and node.iter.func.attr == "items"
                  and isinstance(node.iter.func.value, ast.Dict)]
        self.assertEqual(len(tables), 1)
        counts = tables[0]
        self.assertEqual({key: counts[key] for key in (
            "worker_run_invocations_completed", "worker_run_rank_workers_verified",
            "worker_run_global_optimizer_steps_completed",
            "worker_run_rank_optimizer_steps_completed",
        )}, {"worker_run_invocations_completed": 3, "worker_run_rank_workers_verified": 6,
             "worker_run_global_optimizer_steps_completed": 4,
             "worker_run_rank_optimizer_steps_completed": 8})
        self.assertEqual({key: counts[key] for key in (
            "positive_release_gates_verified", "rejected_release_gates_verified",
            "rejected_release_calls", "rejected_worker_run_invocations",
            "release_gate_total_cgroups_removed", "release_gate_total_trees_removed",
            "rejected_blocked_workers_reaped", "rejected_controllers_reaped",
            "rejected_cases_seccomp_installations", "rejected_cases_visible_socket_removals",
        )}, {"positive_release_gates_verified": 4, "rejected_release_gates_verified": 2,
             "rejected_release_calls": 0, "rejected_worker_run_invocations": 0,
             "release_gate_total_cgroups_removed": 6, "release_gate_total_trees_removed": 6,
             "rejected_blocked_workers_reaped": 2, "rejected_controllers_reaped": 2,
             "rejected_cases_seccomp_installations": 2,
             "rejected_cases_visible_socket_removals": 4})
        self.assertTrue(any(len(call.args) >= 2
                            and equal_expression(call.args[0], "type(report[field])")
                            and equal_expression(call.args[1], "int")
                            for call in calls(self.wrapper, "self.assertIs")))


if __name__ == "__main__":
    unittest.main()
