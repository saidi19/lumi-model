"""Source AST and inert stdlib metadata doubles only; no native/Docker/ML execution.

The diagnostic module is imported under a fake package, never lumi.__init__.
Frozen helpers are compiled in isolated dictionaries whose complete runtime,
subprocess, clock, selector and byte-reader dependencies are in-memory doubles.
Neither the native _read_frozen reader nor a real original runtime is executed.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "src/lumi/posttraining_sft_oci_admission_v1.py"
SOURCE = ROOT / "src/lumi/posttraining_sft_oci_inspect_diagnostic_v1.py"
RAW = FROZEN.read_bytes()
PACKAGE = "_lumi_synthetic_oci_inspect_diagnostic_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
before = set(sys.modules)
diagnostic = importlib.import_module(PACKAGE + ".posttraining_sft_oci_inspect_diagnostic_v1")
IMPORT_ADDITIONS = set(sys.modules) - before
CONTAINER_ID = "a" * 64
PRIVATE_TEXT = "SYNTHETIC-PRIVATE-TOKEN-MUST-NOT-ESCAPE"


def metadata(ns):
    environment = {
        "LUMI_RELEASE_LOCK_SHA256": ns["LOCK_SHA256"],
        "LUMI_RELEASE_BASE_INVENTORY_SHA256": ns["BASE_SHA256"],
        "LUMI_RELEASE_SOURCE_INVENTORY_SHA256": ns["SOURCE_SHA256"],
        "LUMI_RELEASE_SOURCE_COMMIT": ns["SOURCE_COMMIT"],
        "LUMI_WORKER_RUN_TORCH_CI_FIXTURE": "1", "CUDA_VISIBLE_DEVICES": "",
        "GLOO_SOCKET_IFNAME": "lo", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    }
    config = {
        "User": "10001:10001", "WorkingDir": "/opt/lumi-release/source",
        "Entrypoint": [ns["PYTHON"], "-I", "-B"],
        "Cmd": [ns["WRAPPER"], "--synthetic-only"],
        "Env": [key + "=" + value for key, value in environment.items()],
        "Volumes": None, "Labels": None,
    }
    image = {"Id": ns["IMAGE_ID"], "Os": "linux", "Architecture": "amd64", "Config": config}
    container = {
        "Id": CONTAINER_ID, "Image": ns["IMAGE_ID"],
        "Config": {**copy.deepcopy(config), "Image": ns["IMAGE_ID"]},
        "Path": ns["PYTHON"], "Args": ["-I", "-B", ns["WRAPPER"], "--synthetic-only"],
        "RestartCount": 0,
        "State": {"Status": "created", "Running": False, "Paused": False, "Restarting": False,
                  "OOMKilled": False, "Dead": False, "Pid": 0, "ExitCode": 0,
                  "StartedAt": "0001-01-01T00:00:00Z", "FinishedAt": "0001-01-01T00:00:00Z"},
        "HostConfig": {
            "ReadonlyRootfs": True, "Privileged": False, "AutoRemove": False,
            "NetworkMode": "none", "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges"],
            "IpcMode": "private", "Tmpfs": dict(ns["TMPFS"]),
            "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
            "Memory": 4294967296, "MemorySwap": 4294967296, "ShmSize": 67108864,
            "PidsLimit": 256, "NanoCpus": 2000000000,
            "Ulimits": [{"Name": "core", "Soft": 0, "Hard": 0},
                        {"Name": "nofile", "Soft": 1024, "Hard": 1024}],
        },
        "Mounts": [],
    }
    return image, container


class InertTemporaryDirectory:
    def __init__(self, **_kwargs):
        pass

    def __enter__(self):
        return "synthetic-private-config"

    def __exit__(self, *_args):
        return False


class InertDaemon:
    """Every production IO dependency is replaced locally, not globally patched."""

    def __init__(self, ns, image, container):
        self.responses = {
            "image": json.dumps([image]).encode("ascii"),
            "container": json.dumps([container]).encode("ascii"),
        }
        self.calls, self.reads, self.kills = [], [], 0
        self.now, self.status, self.readable = 100.0, 0, True
        self.remaining = b""
        self.failure = None
        self.platform = "linux"
        self.uname_calls = 0
        ns.update(
            sys=types.SimpleNamespace(
                platform=self.platform, implementation=types.SimpleNamespace(name="cpython"),
                version_info=(3, 12), flags=types.SimpleNamespace(no_site=1),
                dont_write_bytecode=True, modules={},
            ),
            os=types.SimpleNamespace(uname=self.uname, read=self.read),
            subprocess=types.SimpleNamespace(Popen=self.popen, DEVNULL=-3, PIPE=-1),
            select=types.SimpleNamespace(select=self.select),
            tempfile=types.SimpleNamespace(TemporaryDirectory=InertTemporaryDirectory),
            time=types.SimpleNamespace(monotonic=lambda: self.now),
        )

    def uname(self):
        self.uname_calls += 1
        return types.SimpleNamespace(machine="x86_64")

    def popen(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if self.failure is not None:
            raise self.failure
        self.remaining = self.responses[command[-3]]
        output = types.SimpleNamespace(fileno=lambda: 42, close=lambda: None)
        return types.SimpleNamespace(stdout=output, wait=lambda **_kwargs: self.status,
                                     poll=lambda: self.status, kill=self.kill)

    def select(self, inputs, _outputs, _errors, _timeout):
        return (inputs if self.readable else []), [], []

    def read(self, fd, size):
        if fd != 42 or not 0 < size <= 8192:
            raise AssertionError("invalid synthetic read")
        self.reads.append(size)
        selected, self.remaining = self.remaining[:size], self.remaining[size:]
        return selected

    def kill(self):
        self.kills += 1


def setup(mutate=None):
    recorder = diagnostic._Recorder()
    ns = diagnostic._build_namespace(RAW, recorder)
    image, container = metadata(ns)
    if mutate is not None:
        mutate(image, container)
    daemon = InertDaemon(ns, image, container)
    return recorder, ns, daemon


def execute(mutate=None, configure=None, *, container_id=CONTAINER_ID, deadline=110.0):
    recorder, ns, daemon = setup(mutate)
    if configure is not None:
        configure(ns, daemon)
    try:
        result = ns["verify_created_oci"](container_id, deadline=deadline)
    except ns["OciAdmissionError"] as error:
        report = recorder.report("passed", False, error.code)
    else:
        if result is not True:
            raise AssertionError("software binding was released")
        recorder.enter("complete")
        report = recorder.report("passed", True, "ok")
    return diagnostic.validate_report(report), daemon


class RemoveOnlyMarkers(ast.NodeTransformer):
    def visit_Subscript(self, node):
        if (isinstance(node.value, ast.Tuple) and len(node.value.elts) == 2
                and isinstance(node.value.elts[0], ast.Call)
                and isinstance(node.value.elts[0].func, ast.Name)
                and node.value.elts[0].func.id in ("_diagnostic_mark", "_diagnostic_field")
                and isinstance(node.slice, ast.Constant) and node.slice.value == 1):
            return self.visit(node.value.elts[1])
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id in ("_diagnostic_mark", "_diagnostic_enter")):
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if node.lineno == 260:
            if not isinstance(node.value, ast.Constant) or node.value.value is not True:
                raise AssertionError("unexpected binding replacement")
            node.value = ast.Call(func=ast.Name(id="SoftwareBinding", ctx=ast.Load()),
                                  args=[], keywords=[])
        return self.generic_visit(node)


class SourceEquivalenceTests(unittest.TestCase):
    def test_exact_frozen_bytes_and_hash(self):
        self.assertEqual(len(RAW), diagnostic.FROZEN_BYTES)
        self.assertEqual(hashlib.sha256(RAW).hexdigest(), diagnostic.FROZEN_SHA256)

    def test_all_original_function_asts_preserved_after_only_marker_removal(self):
        original = {node.name: node for node in ast.parse(RAW).body
                    if isinstance(node, ast.FunctionDef) and node.name in diagnostic.FUNCTIONS}
        transformed = diagnostic._instrumented_tree(RAW)
        restored = RemoveOnlyMarkers().visit(copy.deepcopy(transformed))
        selected = {node.name: node for node in restored.body if isinstance(node, ast.FunctionDef)}
        self.assertEqual(set(original), set(selected))
        for name in original:
            with self.subTest(function=name):
                self.assertEqual(ast.dump(original[name], include_attributes=False),
                                 ast.dump(selected[name], include_attributes=False))

    def test_every_need_site_and_boolean_leaf_mapping_is_enforced(self):
        with patch.dict(diagnostic.SPECS, {252: ("image_id",)}):
            with self.assertRaises(diagnostic.DiagnosticError):
                diagnostic._instrumented_tree(RAW)
        with patch.dict(diagnostic.SPECS, {999: ("image_id",)}):
            with self.assertRaises(diagnostic.DiagnosticError):
                diagnostic._instrumented_tree(RAW)

    def test_source_changes_rejected_before_compilation(self):
        for raw in (RAW[:-1], RAW + b"\n", b"x" + RAW[1:], "not bytes"):
            with self.subTest(raw_type=type(raw).__name__):
                with self.assertRaises(diagnostic.DiagnosticError) as caught:
                    diagnostic._instrumented_tree(raw)
                self.assertEqual(caught.exception.code, "source_rejected")

    def test_no_software_binding_handoff_or_import_execution_in_copy(self):
        tree = diagnostic._instrumented_tree(RAW)
        self.assertFalse(any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body))
        self.assertEqual([node.name for node in tree.body if isinstance(node, ast.ClassDef)],
                         ["OciAdmissionError"])
        calls = [node.func.id for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertNotIn("SoftwareBinding", calls)
        self.assertNotIn("VerifiedOciHandoff", calls)

    def test_import_is_stdlib_only_and_frozen_module_not_imported(self):
        forbidden = {"torch", "transformers", "numpy", "cuda", "cupy", "jax", "tensorflow", "lumi"}
        self.assertFalse(any(name.split(".")[0] in forbidden for name in IMPORT_ADDITIONS))
        self.assertNotIn(PACKAGE + ".posttraining_sft_oci_admission_v1", sys.modules)

    def test_public_diagnose_has_no_caller_injection_or_continuation(self):
        tree = ast.parse(SOURCE.read_bytes())
        node = next(item for item in tree.body
                    if isinstance(item, ast.FunctionDef) and item.name == "diagnose")
        self.assertEqual([item.arg for item in node.args.args], ["container_id"])
        self.assertEqual([item.arg for item in node.args.kwonlyargs], ["deadline"])
        self.assertIsNone(node.args.kwarg)


class LocalizationTests(unittest.TestCase):
    def assert_rejection(self, report, predicate, code="profile_rejected", phase=None):
        self.assertEqual(report["status"], "passed")
        self.assertIs(report["admission_passed"], False)
        self.assertEqual(report["predicate"], predicate)
        self.assertEqual(report["code"], code)
        if phase is not None:
            self.assertEqual(report["phase"], phase)
        self.assertNotIn(PRIVATE_TEXT, json.dumps(report))

    def test_positive_means_diagnosed_not_execution_permission(self):
        report, daemon = execute()
        self.assertIs(report["admission_passed"], True)
        self.assertEqual(report["phase"], "complete")
        self.assertEqual(report["code"], "ok")
        self.assertEqual([call[0][-3] for call in daemon.calls], ["image", "container"])
        self.assertEqual(daemon.kills, 0)
        self.assertTrue(all(report[key] is False for key in diagnostic.NONCLAIMS))
        self.assertLess(len(json.dumps(report).encode("ascii")), 4096)

    def test_exact_local_inspect_commands_and_no_raw_stderr(self):
        _report, daemon = execute()
        for command, options in daemon.calls:
            self.assertEqual(command[:5], ["/usr/bin/docker", "--host",
                                          "unix:///var/run/docker.sock", "--config",
                                          "synthetic-private-config"])
            self.assertEqual(command[-2], "inspect")
            self.assertEqual(options["stderr"], -3)
            self.assertEqual(options["stdin"], -3)
            self.assertIs(options["close_fds"], True)
            self.assertEqual(options["env"], {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"})

    def test_image_failure_stops_before_container_inspect(self):
        report, daemon = execute(lambda image, _container: image.update(Id=PRIVATE_TEXT))
        self.assert_rejection(report, "image_id", "image_rejected", "image_identity")
        self.assertEqual(len(daemon.calls), 1)
        self.assertIs(report["container_inspect_attempted"], False)

    def test_original_runtime_short_circuit_is_preserved(self):
        report, daemon = execute(
            configure=lambda ns, _daemon: setattr(ns["sys"], "platform", "darwin"))
        self.assert_rejection(report, "runtime_platform", "runtime_rejected", "preflight")
        self.assertEqual(daemon.calls, [])
        self.assertEqual(daemon.uname_calls, 0)

    def test_invalid_identifier_and_deadline_start_no_inspect(self):
        for options, predicate in (
            ({"container_id": PRIVATE_TEXT}, "input_container_id_format"),
            ({"container_id": None}, "input_container_id_type"),
            ({"deadline": 116}, "input_deadline_budget"),
            ({"deadline": True}, "input_deadline_type"),
            ({"deadline": 99}, "input_deadline_budget"),
        ):
            with self.subTest(predicate=predicate):
                report, daemon = execute(**options)
                self.assert_rejection(report, predicate, "configuration_rejected", "preflight")
                self.assertEqual(daemon.calls, [])

    def test_configuration_equality_localizes_every_frozen_key(self):
        for key in diagnostic.FIELD_GROUPS["configuration"]:
            with self.subTest(field=key):
                report, _daemon = execute(
                    lambda _image, container, key=key:
                    container["Config"].update({key: PRIVATE_TEXT}))
                self.assert_rejection(report, "configuration_" + key.lower() + "_equal",
                                      phase="configuration")
                self.assertIs(report["configuration_shape"]["available"], True)
                self.assertIs(report["configuration_shape"]["container_is_string"], True)

    def test_absent_and_null_canonical_equality_stays_accepted(self):
        report, _daemon = execute(
            lambda _image, container: container["Config"].update(StopSignal=None))
        self.assertTrue(report["admission_passed"])
        self.assertFalse(report["configuration_shape"]["available"])

    def test_absent_null_empty_string_map_and_list_are_only_shape_booleans(self):
        cases = [("absent", None), ("null", None), ("empty_string", ""),
                 ("empty_map", {}), ("empty_list", [])]
        for name, value in cases:
            with self.subTest(shape=name):
                def mutate(image, container, name=name, value=value):
                    if name != "absent":
                        image["Config"]["StopSignal"] = value
                    container["Config"]["StopSignal"] = PRIVATE_TEXT

                report, _daemon = execute(mutate)
                self.assert_rejection(report, "configuration_stopsignal_equal")
                shape = report["configuration_shape"]
                self.assertTrue(shape["image_" + name])
                self.assertTrue(all(type(item) is bool for item in shape.values()))
                self.assertEqual(shape["image_absent"], name == "absent")
                self.assertEqual(shape["image_null"], name == "null")

    def test_state_loop_fields_localized_with_no_value_retention(self):
        for group, suffix, replacement in (("state_boolean", "false", True),
                                           ("state_integer", "integer", True),
                                           ("state_time", "unset", PRIVATE_TEXT)):
            for key in diagnostic.FIELD_GROUPS[group]:
                with self.subTest(group=group, field=key):
                    report, _daemon = execute(
                        lambda _image, container, key=key, replacement=replacement:
                        container["State"].update({key: replacement}))
                    self.assert_rejection(report, diagnostic._field_code(group, key, suffix),
                                          "container_rejected", "container_metadata")

    def test_missing_key_exception_preserves_current_predicate(self):
        report, _daemon = execute(lambda _image, container: container["State"].pop("Running"))
        self.assert_rejection(report, "state_boolean_running_false", "inspect_failed",
                              "container_metadata")

    def test_compound_condition_preserves_missing_key_short_circuit(self):
        def mutate(_image, container):
            container["HostConfig"]["ReadonlyRootfs"] = False
            container["HostConfig"].pop("Privileged")

        report, _daemon = execute(mutate)
        self.assert_rejection(report, "host_readonly", phase="container_metadata")

    def test_compound_boolean_fields_are_individually_localized(self):
        cases = (("ReadonlyRootfs", False, "host_readonly"),
                 ("Privileged", True, "host_unprivileged"),
                 ("AutoRemove", True, "host_no_autoremove"),
                 ("NetworkMode", PRIVATE_TEXT, "host_network_none"),
                 ("CapDrop", [], "host_cap_drop"),
                 ("SecurityOpt", [], "host_no_new_privileges"),
                 ("IpcMode", PRIVATE_TEXT, "host_ipc_private"),
                 ("Tmpfs", {}, "host_tmpfs"),
                 ("RestartPolicy", {}, "host_restart_policy"))
        for key, value, predicate in cases:
            with self.subTest(field=key):
                report, _daemon = execute(
                    lambda _image, container, key=key, value=value:
                    container["HostConfig"].update({key: value}))
                self.assert_rejection(report, predicate, phase="container_metadata")

    def test_host_loop_fields_localized(self):
        for group, suffix, replacement in (("host_absent", "absent", [PRIVATE_TEXT]),
                                           ("host_namespace", "empty", PRIVATE_TEXT),
                                           ("host_integer", "expected", -1)):
            for key in diagnostic.FIELD_GROUPS[group]:
                with self.subTest(group=group, field=key):
                    report, _daemon = execute(
                        lambda _image, container, key=key, replacement=replacement:
                        container["HostConfig"].update({key: replacement}))
                    self.assert_rejection(report, diagnostic._field_code(group, key, suffix),
                                          phase="container_metadata")

    def test_required_environment_generator_localizes_each_fixed_key(self):
        for key in diagnostic.FIELD_GROUPS["environment"]:
            with self.subTest(field=key):
                def mutate(image, container, key=key):
                    env = [item if not item.startswith(key + "=") else key + "=" + PRIVATE_TEXT
                           for item in image["Config"]["Env"]]
                    image["Config"]["Env"] = env
                    container["Config"]["Env"] = list(env)

                report, _daemon = execute(mutate)
                self.assert_rejection(report,
                                      diagnostic._field_code("environment", key, "expected"),
                                      phase="configuration")

    def test_environment_item_short_circuit_localizes_type(self):
        def mutate(image, container):
            image["Config"]["Env"] = [None]
            container["Config"]["Env"] = [None]

        report, _daemon = execute(mutate)
        self.assert_rejection(report, "environment_item_type", phase="configuration")

    def test_json_failures_are_fixed_and_stop_after_one_inspect(self):
        for raw, predicate, code in (
            (b"[]", "json_single_object", "inspect_shape_rejected"),
            (b"[1]", "json_object_type", "inspect_shape_rejected"),
            (b"[{\"a\":1,\"a\":2}]", "json_unique_keys", "inspect_shape_rejected"),
            (b"[NaN]", "json_constant", "inspect_shape_rejected"),
            (PRIVATE_TEXT.encode(), "json_parse", "inspect_failed"),
        ):
            with self.subTest(predicate=predicate):
                report, daemon = execute(
                    configure=lambda _ns, daemon, raw=raw: daemon.responses.update(image=raw))
                self.assert_rejection(report, predicate, code, "image_json")
                self.assertEqual(len(daemon.calls), 1)

    def test_stream_budget_65536_and_no_second_inspect(self):
        report, daemon = execute(
            configure=lambda _ns, daemon: daemon.responses.update(image=b"x" * 65537))
        self.assert_rejection(report, "inspect_stdout_budget", "inspect_budget_exhausted",
                              "image_inspect")
        self.assertEqual(len(daemon.calls), 1)
        self.assertLessEqual(sum(daemon.reads), 65537)

    def test_inspect_exit_and_launch_errors_are_not_exception_text(self):
        report, _daemon = execute(configure=lambda _ns, daemon: setattr(daemon, "status", 1))
        self.assert_rejection(report, "inspect_exit", "inspect_failed", "image_inspect")
        report, _daemon = execute(configure=lambda _ns, daemon: setattr(
            daemon, "failure", RuntimeError(PRIVATE_TEXT)))
        self.assert_rejection(report, "inspect_launch", "inspect_failed", "image_inspect")

    def test_inspect_readiness_timeout_is_fixed(self):
        report, daemon = execute(configure=lambda _ns, daemon: setattr(daemon, "readable", False))
        self.assert_rejection(report, "inspect_readable", "deadline_exhausted", "image_inspect")
        self.assertEqual(len(daemon.calls), 1)

    def test_final_deadline_phase_is_not_stale_configuration(self):
        def configure(ns, daemon):
            original = ns["_configuration"]

            def finish(image, container):
                original(image, container)
                daemon.now = 120.0

            ns["_configuration"] = finish

        report, _daemon = execute(configure=configure)
        self.assert_rejection(report, "deadline_remaining", "deadline_exhausted", "final_deadline")

    def test_security_opt_spellings_remain_equivalent(self):
        report, _daemon = execute(lambda _image, container: container["HostConfig"].update(
            SecurityOpt=["no-new-privileges:true"]))
        self.assertTrue(report["admission_passed"])


class PublicAndReportTests(unittest.TestCase):
    def test_public_diagnose_consumes_only_one_attempt_with_inert_dependencies(self):
        made = []
        original = diagnostic._build_namespace

        def build(raw, recorder):
            ns = original(raw, recorder)
            made.append(InertDaemon(ns, *metadata(ns)))
            return ns

        with patch.object(diagnostic, "_ATTEMPTED", False), \
                patch.object(diagnostic, "_read_frozen", return_value=RAW) as reader, \
                patch.object(diagnostic, "_build_namespace", side_effect=build):
            first = diagnostic.diagnose(CONTAINER_ID, deadline=110.0)
            second = diagnostic.diagnose(CONTAINER_ID, deadline=110.0)
        self.assertTrue(first["admission_passed"])
        self.assertEqual(second["status"], "failed")
        self.assertEqual(second["code"], "already_consumed")
        self.assertFalse(second["image_inspect_attempted"])
        self.assertEqual(reader.call_count, 1)
        self.assertEqual(len(made), 1)
        self.assertEqual(len(made[0].calls), 2)

    def test_source_binding_failure_never_inspects(self):
        with patch.object(diagnostic, "_ATTEMPTED", False), \
                patch.object(diagnostic, "_read_frozen", return_value=b"not frozen"):
            report = diagnostic.diagnose(CONTAINER_ID, deadline=110.0)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["code"], "source_rejected")
        self.assertFalse(report["image_inspect_attempted"])

    def test_instrumentation_mapping_failure_cannot_be_valid_admission_rejection(self):
        original = diagnostic._build_namespace

        def build(raw, recorder):
            ns = original(raw, recorder)
            InertDaemon(ns, *metadata(ns))
            ns["_diagnostic_mark"] = lambda _predicate: recorder.mark(PRIVATE_TEXT)
            return ns

        with patch.object(diagnostic, "_ATTEMPTED", False), \
                patch.object(diagnostic, "_read_frozen", return_value=RAW), \
                patch.object(diagnostic, "_build_namespace", side_effect=build):
            report = diagnostic.diagnose(CONTAINER_ID, deadline=110.0)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["code"], "diagnostic_internal_failed")
        self.assertNotIn(PRIVATE_TEXT, json.dumps(report))

    def test_marker_or_repeat_guard_sets_invalid_mapping(self):
        recorder = diagnostic._Recorder()
        for method, args in ((recorder.mark, (PRIVATE_TEXT,)),
                             (recorder.enter, (PRIVATE_TEXT,)),
                             (recorder.field, ("configuration", PRIVATE_TEXT, "equal")),
                             (recorder.check, (False,))):
            with self.subTest(method=method.__name__):
                recorder.mapping_valid = True
                with self.assertRaises(diagnostic.DiagnosticError):
                    method(*args)
                self.assertFalse(recorder.mapping_valid)

    def test_report_validator_returns_fresh_nested_dictionary(self):
        report, _daemon = execute()
        fresh = diagnostic.validate_report(report)
        self.assertEqual(fresh, report)
        self.assertIsNot(fresh, report)
        self.assertIsNot(fresh["configuration_shape"], report["configuration_shape"])

    def test_arbitrary_keys_strings_types_claims_and_impossible_attempts_rejected(self):
        original, _daemon = execute()
        mutations = (
            lambda value: value.update(extra=PRIVATE_TEXT),
            lambda value: value.update(code=PRIVATE_TEXT),
            lambda value: value.update(predicate=PRIVATE_TEXT),
            lambda value: value.update(phase=PRIVATE_TEXT),
            lambda value: value.update(status=PRIVATE_TEXT),
            lambda value: value.update(schema=PRIVATE_TEXT),
            lambda value: value.update(admission_passed=1),
            lambda value: value.update(production_eligible=True),
            lambda value: value.update(software_binding_released=True),
            lambda value: value.update(image_inspect_attempted=False),
            lambda value: value["configuration_shape"].update(image_null=1),
            lambda value: value["configuration_shape"].update(extra=False),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                value = copy.deepcopy(original)
                mutate(value)
                with self.assertRaises(diagnostic.DiagnosticError) as caught:
                    diagnostic.validate_report(value)
                self.assertEqual(caught.exception.code, "report_rejected")

    def test_shape_contradictions_rejected(self):
        original, _daemon = execute(
            lambda _image, container: container["Config"].update(StopSignal=""))
        for update in ({"image_null": True}, {"container_is_string": False},
                       {"container_is_map": True}, {"available": False}):
            with self.subTest(update=update):
                value = copy.deepcopy(original)
                value["configuration_shape"].update(update)
                with self.assertRaises(diagnostic.DiagnosticError):
                    diagnostic.validate_report(value)


if __name__ == "__main__":
    unittest.main()
