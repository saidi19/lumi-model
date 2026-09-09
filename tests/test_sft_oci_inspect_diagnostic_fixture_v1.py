"""AST-selected transaction tests: no native entry imports or real daemon calls.

Every runtime, path, Docker and diagnostic operation in run() is doubled.
Source-loader checks execute only tiny generated Python text, never the frozen
guardian. Real file reads/writes are restricted to generated temporary fixtures.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import re
import stat
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/run_sft_oci_inspect_diagnostic_v1.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
DEFINITIONS = [node for node in TREE.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
CODE = compile(ast.Module(body=DEFINITIONS, type_ignores=[]), "synthetic-inspect-fixture", "exec")
CONSTANTS = {}
for node in TREE.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id != "ROOT":
            value = node.value
            CONSTANTS[target.id] = (frozenset(ast.literal_eval(value.args[0]))
                                    if isinstance(value, ast.Call) else ast.literal_eval(value))
OLD_SOURCE = ROOT / CONSTANTS["HELPER_SOURCE"]
OLD_TREE = ast.parse(OLD_SOURCE.read_text(encoding="utf-8"), filename=str(OLD_SOURCE))
CREATE_NODE = next(node for node in OLD_TREE.body
                   if isinstance(node, ast.FunctionDef) and node.name == "create_options")
CREATE_CODE = compile(ast.Module(body=[CREATE_NODE], type_ignores=[]),
                      "synthetic-frozen-create-options", "exec")
CONTAINER_ID = "c" * 64
CONTAINER_NAME = "lumi-oci-guardian-" + "d" * 32


def namespace():
    selected = {**CONSTANTS, "hashlib": hashlib, "json": json, "re": re, "stat": stat,
                "types": types, "ROOT": ROOT, "Path": Path}
    exec(CODE, selected)
    return selected


def info(inode, *, directory=False):
    return types.SimpleNamespace(st_dev=1, st_ino=inode, st_uid=0, st_gid=0,
                                 st_mode=(stat.S_IFDIR | 0o700 if directory
                                          else stat.S_IFREG | 0o400))


class FakePath:
    def __init__(self, name, inode, events, *, directory=False, exists=False):
        self.name, self.info, self.events = name, info(inode, directory=directory), events
        self.present, self.children = exists, {}
        self.mkdir_error = self.unlink_error = self.rmdir_error = None

    def __truediv__(self, name):
        return self.children[name]

    def stat(self):
        if not self.present:
            raise FileNotFoundError("synthetic path absent")
        return self.info

    lstat = stat

    def mkdir(self, *, mode):
        self.events.append(("mkdir", self.name, mode))
        if self.mkdir_error is not None:
            raise self.mkdir_error
        self.present = True

    def open(self, mode):
        if mode != "rb" or not self.present:
            raise AssertionError("unexpected synthetic open")
        self.events.append(("open", self.name))
        return io.BytesIO(b"generated owned archive stand-in")

    def unlink(self):
        self.events.append(("unlink", self.name))
        if self.unlink_error is not None:
            raise self.unlink_error
        self.present = False

    def rmdir(self):
        self.events.append(("rmdir", self.name))
        if self.rmdir_error is not None:
            raise self.rmdir_error
        if any(child.present for child in self.children.values()):
            raise OSError("synthetic directory not empty")
        self.present = False

    def exists(self):
        return self.present


class HelperError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class TransactionTests(unittest.TestCase):
    def setUp(self):
        self.ns = namespace()
        self.events, self.current_time = [], 100.0
        self.parent = FakePath("parent", 1, self.events, directory=True, exists=True)
        self.config = FakePath("config", 2, self.events, directory=True)
        self.image = FakePath("image", 3, self.events)
        self.source = FakePath("source", 4, self.events, exists=True)
        self.parent.children = {"docker-config": self.config, "image.tar": self.image}
        self.images, self.containers = set(), set()
        self.load_error = self.create_error = self.remove_error = self.copy_error = None
        self.create_output = (CONTAINER_ID + "\n").encode("ascii")
        self.leave_container = self.leave_image = False
        self.localized = {"schema": "synthetic.inspect", "status": "passed",
                          "admission_passed": False, "fixed_codes": ["synthetic_mismatch"]}
        self.diagnostic = types.SimpleNamespace(
            diagnose=Mock(side_effect=self.diagnose),
            validate_report=Mock(side_effect=lambda value: value),
        )
        profile = {"need": self.ns["need"], "re": re}
        exec(CREATE_CODE, profile)
        self.helpers = types.SimpleNamespace(
            FixtureError=HelperError, copy_image=Mock(side_effect=self.copy),
            image_inventory=Mock(side_effect=self.image_inventory),
            container_inventory=Mock(side_effect=self.container_inventory),
            command=Mock(side_effect=self.command),
            create_options=Mock(side_effect=profile["create_options"]),
        )
        self.runtime = Mock()
        self.ns.update({
            "runtime": self.runtime,
            "load_source": Mock(side_effect=[self.helpers, self.diagnostic]),
            "Path": lambda _path: self.parent,
            "tempfile": types.SimpleNamespace(mkdtemp=Mock(return_value="/synthetic/private")),
            "time": types.SimpleNamespace(monotonic=lambda: self.current_time),
            "uuid": types.SimpleNamespace(uuid4=lambda: types.SimpleNamespace(hex="d" * 32)),
            "os": types.SimpleNamespace(statvfs=Mock(return_value=types.SimpleNamespace(
                f_bavail=12 * 1024**3, f_frsize=1))),
        })

    def check_deadline(self, deadline):
        if self.current_time >= deadline:
            raise HelperError("deadline_exhausted")

    def copy(self, source, target, deadline, file_ids):
        self.check_deadline(deadline)
        self.assertIs(source, self.source)
        self.assertIs(target, self.image)
        self.events.append(("copy", deadline))
        self.image.present = True
        file_ids[self.image] = self.ns["identity"](self.image.info)
        if self.copy_error is not None:
            raise self.copy_error

    def image_inventory(self, config, deadline):
        self.check_deadline(deadline)
        self.assertIs(config, self.config)
        self.events.append(("image_inventory", deadline))
        return frozenset(self.images)

    def container_inventory(self, config, deadline):
        self.check_deadline(deadline)
        self.assertIs(config, self.config)
        self.events.append(("container_inventory", deadline))
        return frozenset(self.containers)

    def command(self, config, arguments, *, deadline, input_file=None):
        self.check_deadline(deadline)
        self.assertIs(config, self.config)
        if arguments == ["image", "load"]:
            self.events.append(("load", deadline))
            self.assertIsNotNone(input_file)
            self.images.add(self.ns["IMAGE_ID"])
            if self.load_error is not None:
                raise self.load_error
            raw = b"synthetic load result"
        elif arguments[0] == "create":
            self.events.append(("create", deadline))
            self.containers.add(CONTAINER_NAME)
            if self.create_error is not None:
                raise self.create_error
            raw = self.create_output
        elif arguments == ["rm", CONTAINER_NAME]:
            self.events.append(("remove_container", deadline))
            if self.remove_error is not None:
                raise self.remove_error
            if not self.leave_container:
                self.containers.discard(CONTAINER_NAME)
            raw = b"synthetic removed container"
        elif arguments == ["image", "rm", "--no-prune", self.ns["IMAGE_ID"]]:
            self.events.append(("remove_image", deadline))
            if not self.leave_image:
                self.images.discard(self.ns["IMAGE_ID"])
            raw = b"synthetic removed image"
        else:
            raise AssertionError("unexpected daemon action")
        return raw, len(raw), hashlib.sha256(raw).hexdigest(), 0

    def diagnose(self, container_id, *, deadline):
        self.check_deadline(deadline)
        self.assertEqual(container_id, CONTAINER_ID)
        self.events.append(("diagnose", deadline))
        return self.localized

    def execute(self):
        return self.ns["run"](self.source)

    def reject(self, code=None):
        with self.assertRaises((self.ns["DiagnosticFixtureError"], HelperError, OSError)) as caught:
            self.execute()
        if code is not None:
            self.assertEqual(caught.exception.code, code)
        self.assertTrue(self.source.present)
        return caught.exception

    def count(self, action):
        return sum(event[0] == action for event in self.events)

    def test_completed_localization_is_success_even_when_admission_rejected(self):
        result = self.execute()
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["diagnostic"]["admission_passed"])
        self.assertTrue(all(result["cleanup"].values()))
        self.assertEqual(self.count("load"), 1)
        self.assertEqual(self.count("create"), 1)
        self.assertEqual(self.count("diagnose"), 1)
        self.assertEqual(self.count("remove_container"), 1)
        self.assertEqual(self.count("remove_image"), 1)
        self.assertEqual(self.helpers.image_inventory.call_count, 3)
        self.assertEqual(self.helpers.container_inventory.call_count, 3)
        self.assertFalse(self.parent.present)
        self.assertTrue(self.source.present)
        self.assertEqual(self.ns["STAGE"], "complete")

    def test_old_helpers_and_diagnostic_loaded_once_with_exact_source_pins(self):
        self.execute()
        self.assertEqual(self.ns["load_source"].call_args_list[0].args,
                         (self.ns["HELPER_SOURCE"], self.ns["HELPER_SHA256"],
                          "_lumi_inspect_frozen_helpers"))
        self.assertEqual(self.ns["load_source"].call_args_list[1].args,
                         (self.ns["DIAGNOSTIC_SOURCE"], self.ns["DIAGNOSTIC_SHA256"],
                          "_lumi_inspect_localization"))
        self.assertEqual(self.ns["load_source"].call_count, 2)
        self.runtime.assert_called_once()

    def test_create_uses_exact_frozen_profile_without_new_labels_or_entrypoint(self):
        self.execute()
        self.helpers.create_options.assert_called_once_with(self.ns["IMAGE_ID"], CONTAINER_NAME)
        commands = [call.args[1] for call in self.helpers.command.call_args_list]
        selected = next(command for command in commands if command[0] == "create")
        expected = self.helpers.create_options.side_effect(self.ns["IMAGE_ID"], CONTAINER_NAME)
        self.assertEqual(selected, expected)
        self.assertIn("--pull=never", selected)
        self.assertIn("--network=none", selected)
        self.assertFalse(any(option.startswith(("--label", "--entrypoint", "--volume"))
                             for option in selected))

    def test_report_contains_only_declared_nonclaims_and_fixed_diagnostic(self):
        result = self.execute()
        for key in self.ns["NONCLAIMS"]:
            self.assertIs(result[key], False)
        for key in ("container_started", "image_exported", "rootfs_extracted", "workload_created"):
            self.assertIs(result[key], False)
        raw = self.ns["canonical"](result)
        self.assertLessEqual(len(raw), self.ns["MAXIMUM_REPORT_BYTES"])
        for forbidden in (CONTAINER_ID, CONTAINER_NAME, "synthetic load result", "/synthetic/"):
            self.assertNotIn(forbidden.encode(), raw)
        self.diagnostic.validate_report.assert_called_once_with(self.localized)

    def test_existing_image_stops_before_load_without_deleting_it(self):
        self.images.add(self.ns["IMAGE_ID"])
        self.reject()
        self.assertEqual(self.count("load"), 0)
        self.assertEqual(self.count("remove_image"), 0)
        self.diagnostic.diagnose.assert_not_called()
        self.assertIn(self.ns["IMAGE_ID"], self.images)

    def test_existing_reserved_container_stops_without_deleting_it(self):
        self.containers.add(CONTAINER_NAME)
        self.reject()
        self.assertEqual(self.count("load"), 0)
        self.assertEqual(self.count("remove_container"), 0)
        self.assertIn(CONTAINER_NAME, self.containers)

    def test_initial_inventory_failure_is_not_proof_of_absence(self):
        self.helpers.image_inventory.side_effect = HelperError("command_failed")
        self.reject("command_failed")
        self.helpers.image_inventory.assert_called_once()
        self.helpers.command.assert_not_called()
        self.diagnostic.diagnose.assert_not_called()

    def test_container_inventory_failure_stops_before_any_daemon_write(self):
        self.helpers.container_inventory.side_effect = HelperError("command_failed")
        self.reject("command_failed")
        self.helpers.command.assert_not_called()
        self.diagnostic.diagnose.assert_not_called()

    def test_partial_copy_failure_cleans_only_registered_partial_file(self):
        self.copy_error = HelperError("source_rejected")
        self.reject("source_rejected")
        self.assertEqual(self.count("copy"), 1)
        self.assertEqual(self.count("unlink"), 1)
        self.assertFalse(self.image.present)
        self.helpers.command.assert_not_called()
        self.diagnostic.diagnose.assert_not_called()

    def test_ambiguous_load_is_not_retried_or_claimed_complete(self):
        self.load_error = HelperError("command_failed")
        self.reject("cleanup_unconfirmed")
        self.assertEqual(self.count("load"), 1)
        self.assertEqual(self.count("remove_image"), 1)
        self.assertEqual(self.count("create"), 0)
        self.diagnostic.diagnose.assert_not_called()
        self.assertTrue(self.parent.present)

    def test_ambiguous_create_is_not_retried_and_never_diagnosed(self):
        self.create_error = HelperError("command_failed")
        self.reject("cleanup_unconfirmed")
        self.assertEqual(self.count("create"), 1)
        self.assertEqual(self.count("remove_container"), 1)
        self.assertEqual(self.count("remove_image"), 1)
        self.diagnostic.diagnose.assert_not_called()

    def test_invalid_create_output_never_reaches_diagnosis(self):
        self.create_output = b"PRIVATE INVALID CONTAINER ID\n"
        self.reject()
        self.assertEqual(self.count("create"), 1)
        self.assertEqual(self.count("remove_container"), 1)
        self.diagnostic.diagnose.assert_not_called()

    def test_diagnosis_error_cleans_once_without_rediagnosis(self):
        self.diagnostic.diagnose.side_effect = OSError("PRIVATE DIAGNOSTIC EXCEPTION")
        self.reject()
        self.diagnostic.diagnose.assert_called_once()
        self.diagnostic.validate_report.assert_not_called()
        self.assertEqual(self.count("remove_container"), 1)
        self.assertEqual(self.count("remove_image"), 1)

    def test_invalid_diagnostic_report_never_becomes_success(self):
        self.diagnostic.validate_report.side_effect = ValueError("PRIVATE REPORT BODY")
        with self.assertRaises(ValueError):
            self.execute()
        self.assertFalse(self.parent.present)
        self.assertEqual(self.count("create"), 1)
        self.diagnostic.validate_report.assert_called_once()

    def test_incomplete_diagnostic_report_is_fail_closed(self):
        self.localized["status"] = "failed"
        result = self.execute()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["diagnostic"], self.localized)
        self.assertTrue(result["cleanup"]["verified"])
        self.assertEqual(self.count("load"), 1)
        self.assertEqual(self.count("create"), 1)
        self.diagnostic.diagnose.assert_called_once()
        self.assertFalse(self.parent.present)
        output = io.BytesIO()
        self.ns["run"] = Mock(return_value=result)
        self.ns["sys"] = types.SimpleNamespace(
            argv=["synthetic", "--diagnostic-only", "/synthetic/archive"],
            stdout=types.SimpleNamespace(buffer=output),
        )
        with self.assertRaises(SystemExit) as caught:
            self.ns["main"]()
        self.assertEqual(caught.exception.code, 2)
        emitted = json.loads(output.getvalue())
        self.assertEqual(emitted["diagnostic"], self.localized)
        self.assertTrue(emitted["cleanup"]["verified"])

    def test_container_removal_error_is_not_retried(self):
        self.remove_error = HelperError("command_failed")
        self.reject("cleanup_unconfirmed")
        self.assertEqual(self.count("remove_container"), 1)
        self.assertEqual(self.count("remove_image"), 1)

    def test_successful_rm_without_confirmed_container_absence_fails(self):
        self.leave_container = True
        self.reject("cleanup_unconfirmed")
        self.assertEqual(self.count("remove_container"), 1)
        self.assertIn(CONTAINER_NAME, self.containers)

    def test_successful_rm_without_confirmed_image_absence_fails(self):
        self.leave_image = True
        self.reject("cleanup_unconfirmed")
        self.assertEqual(self.count("remove_image"), 1)
        self.assertIn(self.ns["IMAGE_ID"], self.images)

    def test_cleanup_inventory_failure_is_not_silently_accepted(self):
        original = self.helpers.container_inventory.side_effect

        def fail_last(config, deadline):
            if self.helpers.container_inventory.call_count == 3:
                raise HelperError("command_failed")
            return original(config, deadline)

        self.helpers.container_inventory.side_effect = fail_last
        self.reject("cleanup_unconfirmed")
        self.assertEqual(self.count("remove_container"), 1)

    def test_changed_file_identity_preserves_replacement(self):
        def replace(*_args, **_kwargs):
            self.image.info = info(999)
            return self.localized

        self.diagnostic.diagnose.side_effect = replace
        self.reject("cleanup_unconfirmed")
        self.assertTrue(self.image.present)
        self.assertEqual(self.count("unlink"), 0)

    def test_changed_directory_identity_prevents_removal(self):
        def replace(*_args, **_kwargs):
            self.config.info = info(999, directory=True)
            return self.localized

        self.diagnostic.diagnose.side_effect = replace
        self.reject("cleanup_unconfirmed")
        self.assertTrue(self.config.present)
        self.assertNotIn(("rmdir", "config"), self.events)

    def test_file_unlink_failure_is_not_retried(self):
        self.image.unlink_error = OSError("synthetic unlink failure")
        self.reject("cleanup_unconfirmed")
        self.assertEqual(self.events.count(("unlink", "image")), 1)

    def test_directory_creation_failure_does_not_remove_untracked_directory(self):
        self.config.mkdir_error = OSError("synthetic mkdir failure")
        self.reject()
        self.assertNotIn(("rmdir", "config"), self.events)
        self.helpers.copy_image.assert_not_called()
        self.helpers.command.assert_not_called()

    def test_runtime_rejection_prevents_source_loading_and_temp_creation(self):
        self.runtime.side_effect = self.ns["DiagnosticFixtureError"]("runtime_rejected")
        self.reject("runtime_rejected")
        self.ns["load_source"].assert_not_called()
        self.ns["tempfile"].mkdtemp.assert_not_called()

    def test_work_and_cleanup_share_bounded_absolute_budget(self):
        self.execute()
        self.assertEqual(self.ns["MAXIMUM_SECONDS"], 420)
        for action, *values in self.events:
            if action in {"copy", "load", "create", "diagnose"}:
                self.assertLessEqual(values[0], 420.0)  # start100 +320 work seconds.
            if action in {"remove_container", "remove_image"}:
                self.assertLessEqual(values[0], 195.0)  # Shared cleanup start100 +95.

    def test_work_deadline_exhaustion_never_reaches_load_or_diagnosis(self):
        original = self.helpers.copy_image.side_effect

        def exhaust(*args):
            original(*args)
            self.current_time = 421.0

        self.helpers.copy_image.side_effect = exhaust
        self.reject("deadline_exhausted")
        self.helpers.command.assert_not_called()
        self.diagnostic.diagnose.assert_not_called()

    def test_late_cleanup_commands_never_exceed_overall_deadline(self):
        def late(*_args, **_kwargs):
            self.current_time = 500.0
            return self.localized

        self.diagnostic.diagnose.side_effect = late
        self.execute()
        for call in self.helpers.command.call_args_list:
            self.assertLessEqual(call.kwargs["deadline"], 520.0)
        for helper in (self.helpers.image_inventory, self.helpers.container_inventory):
            for call in helper.call_args_list:
                self.assertLessEqual(call.args[1], 520.0)

    def test_main_suppresses_raw_exception_text_and_container_values(self):
        self.ns["run"] = Mock(side_effect=OSError("PRIVATE SOURCE TEXT /secret " + CONTAINER_ID))
        output = io.BytesIO()
        self.ns["sys"] = types.SimpleNamespace(
            argv=["synthetic", "--diagnostic-only", "/synthetic/archive"],
            stdout=types.SimpleNamespace(buffer=output),
        )
        with self.assertRaises(SystemExit) as caught:
            self.ns["main"]()
        self.assertEqual(caught.exception.code, 2)
        result = json.loads(output.getvalue())
        self.assertEqual(result["code"], "fixture_failed")
        self.assertEqual(result["status"], "failed")
        self.assertNotIn(b"PRIVATE", output.getvalue())
        self.assertNotIn(CONTAINER_ID.encode(), output.getvalue())
        self.assertTrue(all(result[key] is False for key in self.ns["NONCLAIMS"]))

    def test_main_retains_only_allowlisted_helper_code(self):
        self.ns["HELPERS"] = self.helpers
        self.ns["run"] = Mock(side_effect=HelperError("command_failed"))
        output = io.BytesIO()
        self.ns["sys"] = types.SimpleNamespace(
            argv=["synthetic", "--diagnostic-only", "/synthetic/archive"],
            stdout=types.SimpleNamespace(buffer=output),
        )
        with self.assertRaises(SystemExit):
            self.ns["main"]()
        self.assertEqual(json.loads(output.getvalue())["code"], "command_failed")


class SourceAndRuntimeTests(unittest.TestCase):
    def test_fixture_imports_only_explicit_stdlib_roots(self):
        imports = set()
        for node in ast.walk(TREE):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0)
                imports.add(node.module.split(".")[0])
        self.assertLessEqual(imports, {"__future__", "hashlib", "json", "os", "re", "stat",
                                       "sys", "tempfile", "time", "types", "uuid", "pathlib"})

    def test_frozen_helper_pin_matches_exact_local_source(self):
        self.assertEqual(hashlib.sha256(OLD_SOURCE.read_bytes()).hexdigest(),
                         CONSTANTS["HELPER_SHA256"])

    def test_diagnostic_pin_matches_exact_local_source(self):
        expected = CONSTANTS["DIAGNOSTIC_SHA256"]
        self.assertRegex(expected, r"\A[0-9a-f]{64}\Z")
        raw = (ROOT / CONSTANTS["DIAGNOSTIC_SOURCE"]).read_bytes()
        self.assertGreater(len(raw), 0)
        self.assertLessEqual(len(raw), 131072)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), expected)

    def test_no_native_workload_or_export_entry_is_called(self):
        forbidden = {"export", "extract_rootfs", "load_base", "run_case", "workload", "guardian",
                     "fork", "execve", "Popen", "mount", "start", "spawn"}
        called = {node.func.attr for node in ast.walk(TREE)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        called |= {node.func.id for node in ast.walk(TREE)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertFalse(called & forbidden)

    def runtime_namespace(self):
        ns = namespace()
        ns["sys"] = types.SimpleNamespace(
            platform="linux", implementation=types.SimpleNamespace(name="cpython"),
            version_info=(3, 12, 14), flags=types.SimpleNamespace(no_site=1, isolated=1),
            dont_write_bytecode=True, modules={},
        )
        ns["os"] = types.SimpleNamespace(
            uname=lambda: types.SimpleNamespace(machine="x86_64"),
            getresuid=lambda: (0, 0, 0), getresgid=lambda: (0, 0, 0),
            environ={"LUMI_OCI_INSPECT_DIAGNOSTIC_CI": "1", "PRIVATE_KEY": "synthetic"},
            umask=Mock(),
        )
        return ns

    def test_runtime_guard_sanitizes_only_doubled_environment(self):
        ns = self.runtime_namespace()
        ns["runtime"]()
        self.assertEqual(ns["os"].environ, {"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"})
        ns["os"].umask.assert_called_once_with(0o077)

    def test_runtime_guard_rejects_all_existing_ml_module_roots(self):
        for root in ("torch", "transformers", "numpy", "cuda", "cupy", "jax", "tensorflow"):
            with self.subTest(root=root):
                ns = self.runtime_namespace()
                ns["sys"].modules[root + ".synthetic"] = object()
                with self.assertRaisesRegex(ns["DiagnosticFixtureError"], "runtime_rejected"):
                    ns["runtime"]()
                ns["os"].umask.assert_not_called()

    def test_runtime_guard_rejects_mac_or_wrong_python(self):
        for field, value in (("platform", "darwin"), ("version_info", (3, 13, 7)),
                             ("dont_write_bytecode", False)):
            with self.subTest(field=field):
                ns = self.runtime_namespace()
                setattr(ns["sys"], field, value)
                with self.assertRaisesRegex(ns["DiagnosticFixtureError"], "runtime_rejected"):
                    ns["runtime"]()
                ns["os"].umask.assert_not_called()


class GeneratedSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lumi-synthetic-inspect-source-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.path = self.base / "generated.py"
        self.raw = b"GENERATED_SYNTHETIC_VALUE = 7\n"
        self.path.write_bytes(self.raw)
        self.pin = hashlib.sha256(self.raw).hexdigest()
        self.ns = namespace()
        self.ns.update(ROOT=self.base, os=os, sys=types.SimpleNamespace(modules={}))

    def test_exact_tiny_source_is_read_and_loaded_without_native_import(self):
        result = self.ns["load_source"]("generated.py", self.pin, "_synthetic_generated")
        self.assertEqual(result.GENERATED_SYNTHETIC_VALUE, 7)
        self.assertEqual(result.__file__, str(self.path))
        self.assertIs(self.ns["sys"].modules["_synthetic_generated"], result)

    def test_source_pin_failure_never_executes_generated_code(self):
        with self.assertRaisesRegex(self.ns["DiagnosticFixtureError"], "source_rejected"):
            self.ns["load_source"]("generated.py", "0" * 64, "_synthetic_generated")
        self.assertEqual(self.ns["sys"].modules, {})

    def test_loader_never_replaces_an_existing_module(self):
        sentinel = object()
        self.ns["sys"].modules["_synthetic_generated"] = sentinel
        with self.assertRaisesRegex(self.ns["DiagnosticFixtureError"], "source_rejected"):
            self.ns["load_source"]("generated.py", self.pin, "_synthetic_generated")
        self.assertIs(self.ns["sys"].modules["_synthetic_generated"], sentinel)

    def test_generated_source_exception_is_fixed_code_and_module_removed(self):
        raw = b"raise ValueError('PRIVATE GENERATED EXCEPTION')\n"
        self.path.write_bytes(raw)
        with self.assertRaisesRegex(self.ns["DiagnosticFixtureError"], "source_rejected") as caught:
            self.ns["load_source"]("generated.py", hashlib.sha256(raw).hexdigest(), "_generated")
        self.assertEqual(str(caught.exception), "source_rejected")
        self.assertEqual(self.ns["sys"].modules, {})

    def test_symlink_source_is_rejected(self):
        alias = self.base / "alias.py"
        alias.symlink_to(self.path)
        with self.assertRaisesRegex(self.ns["DiagnosticFixtureError"], "source_rejected"):
            self.ns["read_source"]("alias.py", self.pin)

    def test_invalid_pin_shape_and_empty_source_are_rejected(self):
        for value in (None, "A" * 64, "short"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(self.ns["DiagnosticFixtureError"], "source_rejected"):
                    self.ns["read_source"]("generated.py", value)
        self.path.write_bytes(b"")
        with self.assertRaisesRegex(self.ns["DiagnosticFixtureError"], "source_rejected"):
            self.ns["read_source"]("generated.py", hashlib.sha256(b"").hexdigest())


if __name__ == "__main__":
    unittest.main()
