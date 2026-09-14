"""Pure generated control and mocked-boundary tests; no native ML or model I/O."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import importlib.util
import json
import os
import stat
import sys
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_checkpoint_prepared_fixture_v1.py"
spec = importlib.util.spec_from_file_location("_prepared_entry_test_v1", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
PACKAGE = "_lumi_prepared_entry_pure_tests_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
runner = importlib.import_module(PACKAGE + ".posttraining_checkpoint_import_runner_v3")


def generated_documents():
    tree = ast.parse((ROOT / "tests/test_sft_checkpoint_import_plan_v1.py").read_bytes())
    function = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "documents")
    namespace = {"canonical": m.canonical, "sha": m.digest}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "generated-controls", "exec"),
         namespace)
    return namespace["documents"]()


def fixture():
    metadata = generated_documents()
    base = m.decode(metadata[0])
    base.update(synthetic_fixture_only=True, prepared_fixture_marker={
        "schema": m.BASE_MARKER_SCHEMA, "generator": "fixed-tiny-config-v1",
        "generation_nonce_sha256": "9" * 64})
    metadata[0] = m.canonical(base)
    plan = runner.frozen.build_import_plan(*metadata,
        expected_base_sha256=m.digest(metadata[0]), dtype="bfloat16")
    source_rows = [{"path": name, "bytes": len(raw), "sha256": m.digest(raw)}
                  for name, raw in (("configuration_lumi.py", b"# synthetic config\n"),
                                    ("modeling_lumi.py", b"# synthetic model\n"))]
    source_bytes = sum(row["bytes"] for row in source_rows)
    b, control = plan["resource_bounds"], 131072
    rows = {row["path"]: row for row in plan["checkpoint_inventory"]}
    dcp_payload = max(b["source_fp32_tensor_bytes_including_aliases"], sum(
        row["bytes"] for name, row in rows.items() if name.endswith(".distcp")))
    dcp_payload += 2 * rows["model/.metadata"]["bytes"]
    tokens, backend = b["tokenizer_inventory_bytes"], b["maximum_backend_output_bytes"]
    envelope = {"schema_version": "lumi.sft-checkpoint-resource-envelope.v0.1",
        "plan_sha256": plan["plan_sha256"], "dcp_payload_read_cap_bytes": dcp_payload,
        "dcp_read_cap_bytes": b["checkpoint_inventory_bytes"] + dcp_payload,
        "hf_read_cap_bytes": backend + 2 * source_bytes + len(metadata[3]),
        "package_read_cap_bytes": 3 * backend + 3 * tokens + 5 * control,
        "hf_write_cap_bytes": backend, "package_write_cap_bytes": tokens + 2 * control,
        "config_bytes": len(metadata[3]), "source_config_sha256": m.digest(metadata[3]),
        "control_bytes": control, "package_receipt_cap_bytes": control,
        "package_manifest_cap_bytes": control, "runtime_sources": source_rows,
        "private_write_cap_bytes": b["maximum_private_output_write_bytes"],
        "worker_memory_max_bytes": b["minimum_host_ram_planning_bytes"],
        "output_capacity_bytes": b["maximum_private_output_write_bytes"],
        "extra_host_disk_reserve_bytes": 1024**3,
        "host_disk_capacity_bytes": b["minimum_staged_disk_planning_bytes"] + 1024**3,
        "maximum_seconds": 30}
    envelope["coordinator_read_cap_bytes"] = (envelope["dcp_read_cap_bytes"]
        + envelope["hf_read_cap_bytes"] + b["checkpoint_inventory_bytes"]
        + b["tokenizer_audit_read_bytes"] + envelope["package_read_cap_bytes"] + source_bytes)
    io = importlib.import_module(PACKAGE + ".posttraining_checkpoint_io_contract_v1")
    contract = io.build_kernel_io_contract(plan, checkpoint_read_contract={
        "expected_manifest_sha256": plan["metadata"]["checkpoint_manifest_sha256"],
        "maximum_file_bytes": max(row["bytes"] for row in rows.values()),
        "maximum_read_bytes": envelope["dcp_read_cap_bytes"], "maximum_read_call_bytes": 1048576},
        hf_io_contract={"maximum_read_bytes": envelope["hf_read_cap_bytes"],
                        "maximum_write_bytes": backend, "maximum_file_bytes": backend,
                        "maximum_control_bytes": 65536, "target_shard_bytes": 268435456})
    bodies = dict(zip(m.METADATA_NAMES, metadata, strict=True))
    bodies.update({"plan.json": m.canonical(plan), "envelope.json": m.canonical(envelope),
                   "kernel-io.json": m.canonical(contract)})
    value = {"schema": m.CONTROL_SCHEMA, "scope": "generated_tiny_checkpoint_only",
             "generation_nonce_sha256": "9" * 64, "entry_sha256": "a" * 64,
             "source_manifest_sha256": "b" * 64,
             "limits": {"maximum_parameters": m.MAXIMUM_PARAMETERS,
                        "maximum_input_bytes": m.MAXIMUM_INPUT_BYTES}}
    rebind(value, bodies)
    return value, bodies


def rebind(value, bodies):
    value["files"] = [{"path": name, "bytes": len(bodies[name]), "sha256": m.digest(bodies[name])}
                      for name in m.CONTROL_NAMES]


def validate(value):
    raw = m.canonical(value)
    return m.validate_control(raw, expected_sha256=m.digest(raw), entry_sha256="a" * 64,
                              source_manifest_sha256="b" * 64)


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.control, self.bodies = fixture()

    def test_bound_generated_bundle_returns_exact_arguments_without_io(self):
        with patch.object(m.os, "open") as opened:
            arguments = m.validate_bundle(validate(self.control), self.bodies)
        opened.assert_not_called()
        self.assertEqual(len(arguments["metadata"]), 5)
        self.assertEqual(arguments["metadata"][3], m.canonical(m.CONFIG))
        self.assertNotIn("checkpoint_root", arguments)

    def test_real_resource_algebra_rebuilds_without_any_input_io(self):
        arguments = m.validate_bundle(validate(self.control), self.bodies)
        files = {"lumi/configuration_lumi.py": b"# synthetic config\n",
                 "lumi/modeling_lumi.py": b"# synthetic model\n"}
        with patch.object(m.os, "open") as opened:
            plan = m.validate_runner_controls(runner, arguments, files)
        opened.assert_not_called()
        self.assertEqual(plan["model_parameters"], 736)
        self.assertFalse(plan["execution_permitted"])

    def test_runtime_source_hash_must_match_bound_envelope(self):
        arguments = m.validate_bundle(validate(self.control), self.bodies)
        with self.assertRaisesRegex(m.PreparedFixtureError, "resource_contract_rejected"):
            m.validate_runner_controls(runner, arguments, {
                "lumi/configuration_lumi.py": b"wrong", "lumi/modeling_lumi.py": b"wrong"})

    def test_bad_internal_plan_binding_rejects_before_input_io(self):
        plan = m.decode(self.bodies["plan.json"])
        plan["model_parameters"] = 900
        self.bodies["plan.json"] = m.canonical(plan)
        rebind(self.control, self.bodies)
        arguments = m.validate_bundle(validate(self.control), self.bodies)
        with patch.object(m.os, "open") as opened, \
                self.assertRaisesRegex(m.PreparedFixtureError, "resource_contract_rejected"):
            m.validate_runner_controls(runner, arguments, {})
        opened.assert_not_called()

    def test_independent_control_pin_is_required(self):
        with self.assertRaises(m.PreparedFixtureError):
            m.validate_control(m.canonical(self.control), expected_sha256="0" * 64,
                               entry_sha256="a" * 64, source_manifest_sha256="b" * 64)

    def test_control_schema_is_closed_against_approval_flags(self):
        for name in ("approved", "execution_permitted", "real_input_allowed", "checkpoint_path"):
            with self.subTest(name=name):
                value = copy.deepcopy(self.control)
                value[name] = True
                with self.assertRaises(m.PreparedFixtureError):
                    validate(value)

    def test_file_order_names_and_exact_lengths_are_bound(self):
        changes = (("path", "../base.json"), ("bytes", True), ("bytes", 0),
                   ("bytes", m.MAXIMUM_CONTROL_BYTES + 1), ("sha256", "A" * 64))
        for key, value in changes:
            with self.subTest(key=key, value=value):
                control = copy.deepcopy(self.control)
                control["files"][0][key] = value
                with self.assertRaises(m.PreparedFixtureError):
                    validate(control)
        self.control["files"].reverse()
        with self.assertRaises(m.PreparedFixtureError):
            validate(self.control)

    def test_file_body_corruption_or_extra_file_is_rejected(self):
        self.bodies["base.json"] += b" "
        with self.assertRaises(m.PreparedFixtureError):
            m.validate_bundle(validate(self.control), self.bodies)
        self.bodies["unbound.json"] = b"{}"
        with self.assertRaises(m.PreparedFixtureError):
            m.validate_bundle(validate(self.control), self.bodies)

    def test_plain_old_synthetic_boolean_is_insufficient(self):
        base = m.decode(self.bodies["base.json"])
        del base["prepared_fixture_marker"]
        self.bodies["base.json"] = m.canonical(base)
        rebind(self.control, self.bodies)
        with self.assertRaisesRegex(m.PreparedFixtureError, "synthetic_scope_rejected"):
            m.validate_bundle(validate(self.control), self.bodies)

    def test_nonce_must_match_independently_bound_root_control(self):
        base = m.decode(self.bodies["base.json"])
        base["prepared_fixture_marker"]["generation_nonce_sha256"] = "8" * 64
        self.bodies["base.json"] = m.canonical(base)
        rebind(self.control, self.bodies)
        with self.assertRaisesRegex(m.PreparedFixtureError, "synthetic_scope_rejected"):
            m.validate_bundle(validate(self.control), self.bodies)
        self.control["generation_nonce_sha256"] = "0" * 64
        with self.assertRaises(m.PreparedFixtureError):
            validate(self.control)

    def test_different_tiny_geometry_is_not_admitted(self):
        config = dict(m.CONFIG, hidden_size=9)
        self.bodies["config.json"] = m.canonical(config)
        rebind(self.control, self.bodies)
        with self.assertRaisesRegex(m.PreparedFixtureError, "synthetic_scope_rejected"):
            m.validate_bundle(validate(self.control), self.bodies)

    def test_parameter_and_input_caps_are_checked_before_runtime_imports(self):
        for kind in ("parameters", "bytes", "files", "bool"):
            with self.subTest(kind=kind):
                control, bodies = fixture()
                plan = m.decode(bodies["plan.json"])
                if kind == "parameters":
                    plan["model_parameters"] = m.MAXIMUM_PARAMETERS + 1
                elif kind == "bytes":
                    plan["checkpoint_inventory"][0]["bytes"] = m.MAXIMUM_INPUT_BYTES
                elif kind == "files":
                    plan["checkpoint_inventory"] *= 6
                else:
                    plan["model_parameters"] = True
                bodies["plan.json"] = m.canonical(plan)
                rebind(control, bodies)
                with self.assertRaisesRegex(m.PreparedFixtureError, "synthetic_scope_rejected"):
                    m.validate_bundle(validate(control), bodies)

    def test_time_read_and_write_caps_cannot_be_raised(self):
        for key, cap in (("maximum_seconds", m.MAXIMUM_SECONDS),
                         ("coordinator_read_cap_bytes", m.MAXIMUM_READ_BYTES),
                         ("private_write_cap_bytes", m.MAXIMUM_WRITE_BYTES)):
            with self.subTest(key=key):
                control, bodies = fixture()
                envelope = m.decode(bodies["envelope.json"])
                envelope[key] = cap + 1
                bodies["envelope.json"] = m.canonical(envelope)
                rebind(control, bodies)
                with self.assertRaisesRegex(m.PreparedFixtureError, "synthetic_scope_rejected"):
                    m.validate_bundle(validate(control), bodies)

    def test_strict_json_and_canonical_root_control(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'[]', b'not-json'):
            with self.subTest(raw=raw), self.assertRaises(m.PreparedFixtureError):
                m.decode(raw)
        raw = m.canonical(self.control) + b"\n"
        with self.assertRaises(m.PreparedFixtureError):
            m.validate_control(raw, expected_sha256=m.digest(raw), entry_sha256="a" * 64,
                               source_manifest_sha256="b" * 64)


class EntryBoundaryTests(unittest.TestCase):
    def test_frozen_helper_pin_matches_unchanged_file(self):
        raw = (ROOT / "scripts/run_checkpoint_guarded_import_v1.py").read_bytes()
        self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()),
                         (m.HELPER_BYTES, m.HELPER_SHA256))

    def test_wrong_helper_pin_never_executes(self):
        with patch.object(m, "root_read", return_value=b"raise RuntimeError('raw text')"), \
                patch("builtins.exec") as execute:
            with self.assertRaisesRegex(m.PreparedFixtureError, "helper_binding_rejected"):
                m.load_helpers()
        execute.assert_not_called()

    def test_runtime_rejects_local_platform_before_reads(self):
        with patch.object(m.sys, "platform", "darwin"), patch.object(m, "root_read") as read:
            with self.assertRaisesRegex(m.PreparedFixtureError, "runtime_rejected"):
                m.runtime_check([str(m.ENTRY), "--synthetic-only"], {})
        read.assert_not_called()

    def test_runtime_exact_argv_has_no_user_paths_or_approval_options(self):
        for argv in ([str(m.ENTRY)], [str(m.ENTRY), "--approved"],
                     [str(m.ENTRY), "--synthetic-only", "/real/checkpoint"]):
            with self.subTest(argv=argv), self.assertRaises(m.PreparedFixtureError):
                m.runtime_check(argv, {})

    def test_root_owner_checks_reject_worker_control_files(self):
        info = types.SimpleNamespace(st_mode=stat.S_IFREG | 0o400, st_uid=10001,
                                     st_gid=10001, st_nlink=1)
        with self.assertRaises(m.PreparedFixtureError):
            m._root_owned(info)
        for mode, links in ((0o666, 1), (0o400, 2)):
            info = types.SimpleNamespace(st_mode=stat.S_IFREG | mode, st_uid=0,
                                         st_gid=0, st_nlink=links)
            with self.assertRaises(m.PreparedFixtureError):
                m._root_owned(info)

    def test_public_real_runner_is_still_closed(self):
        with patch.object(runner.os, "open") as opened, \
                self.assertRaisesRegex(runner.ImportRunnerError, "host_guard_not_bound"):
            runner.run_import(approved=True, synthetic_only=True)
        opened.assert_not_called()

    def test_package_avoids_init_autoregistration_and_uses_exact_lumi_namespace(self):
        modules = {key: value for key, value in sys.modules.items()
                   if key.split(".", 1)[0] not in m.ML_ROOTS | {"lumi"}}
        with patch.dict(sys.modules, modules, clear=True), \
                patch.object(m.importlib, "import_module") as imported:
            m.install_package(Path("/outputs/generated-code/src"))
            package = sys.modules["lumi"]
            self.assertEqual(package.__path__, ["/outputs/generated-code/src/lumi"])
            self.assertIsNone(package.__spec__.loader)
        imported.assert_called_once_with("lumi.posttraining_checkpoint_import_runner_v3")

    def test_main_runtime_failure_is_aggregate_only_and_does_no_control_reads(self):
        writes = []
        with patch.object(m, "runtime_check",
                          side_effect=m.PreparedFixtureError("runtime_rejected")), \
                patch.object(m, "root_read") as read, \
                patch.object(m.os, "write",
                             side_effect=lambda _fd, raw: writes.append(raw) or len(raw)):
            result = m.main()
        self.assertEqual(result, 1)
        read.assert_not_called()
        report = json.loads(writes[0])
        self.assertEqual(report, m._failure("runtime_rejected"))
        self.assertFalse(report["real_4b_import"])

    def test_main_control_failure_precedes_source_stage_and_runner(self):
        helper = MagicMock()
        writes = []
        environment = {**m.FIXED_ENV, **dict.fromkeys(m.PIN_KEYS, "a" * 64)}
        with ExitStack() as stack:
            stack.enter_context(patch.dict(m.os.environ, environment, clear=True))
            stack.enter_context(patch.object(m, "runtime_check"))
            stack.enter_context(patch.object(m, "root_read", side_effect=[b"entry", b"{}"] ))
            stack.enter_context(patch.object(m, "digest", return_value="a" * 64))
            stack.enter_context(patch.object(m, "load_helpers", return_value=helper))
            package = stack.enter_context(patch.object(m, "install_package"))
            stack.enter_context(patch.object(m.os, "write",
                side_effect=lambda _fd, raw: writes.append(raw) or len(raw)))
            self.assertEqual(m.main(), 1)
        helper.verify_source_tree.assert_not_called()
        helper.stage_private_sources.assert_not_called()
        package.assert_not_called()
        self.assertEqual(json.loads(writes[0])["code"], "control_binding_rejected")

    def test_main_positive_flow_calls_private_coordinator_once_and_pins_fixed_paths(self):
        control, bodies = fixture()
        arguments = m.validate_bundle(validate(control), bodies)
        plan = m.decode(bodies["plan.json"])
        outcome = {"schema_version": runner.SCHEMA,
            "status": "PASS_import_coordinator_requires_parent_verification", "stage": "complete",
            "full_input_audits_completed": 2, "plan_sha256": plan["plan_sha256"],
            "resource_envelope_sha256": arguments["expected_envelope_file_sha256"],
            "kernel_io_contract_sha256": arguments["expected_kernel_io_contract_sha256"],
            "real_4b_import": False, "training_authorized": False,
            "production_eligible": False, "retry_allowed": False,
            "coordinator_read_bytes_reserved": 5000000, "private_write_bytes_reserved": 1000000}
        mock_runner = MagicMock(SCHEMA=runner.SCHEMA)
        mock_runner._coordinate_admitted_import.return_value = outcome
        helper = MagicMock()
        files = dict.fromkeys(m.REQUIRED_SOURCES, b"# generated source\n")
        source_bytes = sum(map(len, files.values()))
        helper.verify_source_tree.return_value = (files, {
            "source_manifest_sha256": "b" * 64,
            "source_files_verified": len(files), "source_bytes_verified": source_bytes,
            "source_verification_read_bytes": source_bytes + 1000})
        helper.stage_private_sources.return_value = (Path("/outputs/staged/src"),
            {"private_source_copy_files": len(files),
             "private_source_copy_bytes_reserved": source_bytes,
             "private_source_copy_readback_bytes": source_bytes,
             "private_source_copies_rootfs_immutable": False,
             "hostile_owner_code_immutability_verified": False})
        control_raw = m.canonical(control)
        environment = {**m.FIXED_ENV, "LUMI_PREPARED_FIXTURE_ENTRY_SHA256": "a" * 64,
            "LUMI_IMPORT_SOURCE_MANIFEST_SHA256": "b" * 64,
            "LUMI_PREPARED_FIXTURE_CONTROL_SHA256": m.digest(control_raw)}
        original_digest, writes = m.digest, []

        def read(path, *, maximum):
            if path == m.ENTRY:
                return b"entry"
            if path == m.CONTROL:
                return control_raw
            return bodies[path.name]

        with ExitStack() as stack:
            stack.enter_context(patch.dict(m.os.environ, environment, clear=True))
            stack.enter_context(patch.object(m, "runtime_check"))
            stack.enter_context(patch.object(m, "root_read", side_effect=read))
            stack.enter_context(patch.object(m, "digest", side_effect=lambda raw:
                "a" * 64 if raw == b"entry" else original_digest(raw)))
            stack.enter_context(patch.object(m, "load_helpers", return_value=helper))
            stack.enter_context(patch.object(m, "_output_check"))
            prepare = stack.enter_context(patch.object(m, "prepare_import_parent"))
            stack.enter_context(patch.object(m, "install_package", return_value=mock_runner))
            stack.enter_context(patch.object(m, "validate_runner_controls", return_value=plan))
            stack.enter_context(patch.object(m.os, "dup", return_value=99999))
            stack.enter_context(patch.object(m.os, "dup2"))
            stack.enter_context(patch.object(m.os, "close"))
            stack.enter_context(patch.object(m.os, "write",
                side_effect=lambda _fd, raw: writes.append(raw) or len(raw)))
            self.assertEqual(m.main(), 0)
        mock_runner._coordinate_admitted_import.assert_called_once_with(**arguments,
            checkpoint_root=m.CHECKPOINT, tokenizer_root=m.TOKENIZER, attempt=m.ATTEMPT)
        prepare.assert_called_once_with(helper)
        report = json.loads(writes[0])
        self.assertEqual((report["model_parameters"], report["coordinator_passes"]), (736, 1))
        self.assertEqual(report["generated_input_files"], 9)
        self.assertFalse(report["host_capacity_verified"])
        self.assertTrue(report["resource_envelope_is_planning_only"])
        self.assertNotIn("metadata", report)
        self.assertNotIn("checkpoint_root", report)
        guard = importlib.import_module(PACKAGE + ".posttraining_checkpoint_guard_composition_v3")
        observed = guard._parse(writes[0])
        self.assertEqual(guard._worker(observed, environment,
                                       control["generation_nonce_sha256"]), report)
        with self.assertRaisesRegex(guard.GuardCompositionError, "worker_rejected"):
            guard._worker(observed, environment, "8" * 64)

    def test_result_aggregate_fields_reject_strings_and_boolean_counts(self):
        control, bodies = fixture()
        args = m.validate_bundle(validate(control), bodies)
        plan = m.decode(bodies["plan.json"])
        outcome = {"schema_version": runner.SCHEMA,
            "status": "PASS_import_coordinator_requires_parent_verification", "stage": "complete",
            "full_input_audits_completed": 2, "plan_sha256": plan["plan_sha256"],
            "resource_envelope_sha256": args["expected_envelope_file_sha256"],
            "kernel_io_contract_sha256": args["expected_kernel_io_contract_sha256"],
            "real_4b_import": False, "training_authorized": False,
            "production_eligible": False, "retry_allowed": False,
            "coordinator_read_bytes_reserved": 5000000, "private_write_bytes_reserved": 1000000}
        for value in ("private arbitrary exception text", True, m.MAXIMUM_READ_BYTES + 1):
            with self.subTest(value=value):
                bad = {**outcome, "coordinator_read_bytes_reserved": value}
                with self.assertRaisesRegex(m.PreparedFixtureError, "result_rejected"):
                    m.validate_result(bad, plan, args, runner_schema=runner.SCHEMA)

    def test_ast_has_one_private_call_no_public_import_and_no_cli_path(self):
        tree = ast.parse(SCRIPT.read_bytes())
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)]
        self.assertEqual(calls.count("_coordinate_admitted_import"), 1)
        self.assertNotIn("run_import", calls)
        self.assertNotIn("run_tiny_fixture", calls)
        self.assertNotIn("_generate", calls)
        self.assertEqual((m.CHECKPOINT, m.TOKENIZER, m.ATTEMPT),
                         (Path("/inputs/checkpoint"), Path("/inputs/tokenizer"),
                          Path("/outputs/import/attempt-001")))

    def test_frozen_path_guard_checks_exact_attempt_and_parent_depth(self):
        # This is the actual frozen validator, not a coordinator/path mock.
        # Only metadata operations are doubled: /outputs is not a local path.
        with self.assertRaisesRegex(runner.frozen.ImportRunnerError, "path_rejected"):
            runner.frozen._path(Path("/outputs"))
        info = types.SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_uid=os.geteuid())
        with patch.object(Path, "is_symlink", return_value=False), \
                patch.object(Path, "exists", return_value=False), \
                patch.object(Path, "is_dir", return_value=True), \
                patch.object(Path, "lstat", return_value=info):
            runner.frozen._path(m.ATTEMPT, absent=True)
            runner.frozen._path(m.ATTEMPT.parent)
        self.assertEqual(m.ATTEMPT.parent, m.IMPORT_PARENT)

    def test_new_import_parent_is_exclusive_nofollow_private_and_fixed(self):
        helper = MagicMock()
        helper.open_directory.return_value = 80
        info = types.SimpleNamespace(st_dev=1, st_ino=2, st_mode=stat.S_IFDIR | 0o700,
            st_uid=10001, st_gid=10001, st_nlink=2, st_size=0, st_mtime_ns=1, st_ctime_ns=1)
        with patch.object(m.os, "fstat", return_value=info), \
                patch.object(m.os, "mkdir") as mkdir, \
                patch.object(m.os, "open", return_value=81) as opened, \
                patch.object(m.os, "stat", return_value=info) as observed, \
                patch.object(m.os, "fsync") as synced, \
                patch.object(m.os, "close") as closed:
            m.prepare_import_parent(helper)
        helper.open_directory.assert_called_once_with(m.OUTPUT)
        mkdir.assert_called_once_with("import", 0o700, dir_fd=80)
        opened.assert_called_once_with("import", m.DIR_FLAGS, dir_fd=80)
        observed.assert_called_once_with("import", dir_fd=80, follow_symlinks=False)
        self.assertEqual([call.args for call in synced.call_args_list], [(81,), (80,)])
        self.assertEqual([call.args for call in closed.call_args_list], [(81,), (80,)])

    def test_existing_import_parent_fails_without_reuse_or_cleanup(self):
        helper = MagicMock()
        helper.open_directory.return_value = 80
        info = types.SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_uid=10001, st_gid=10001)
        with patch.object(m.os, "fstat", return_value=info), \
                patch.object(m.os, "mkdir", side_effect=FileExistsError("private path")) as mkdir, \
                patch.object(m.os, "open") as opened, patch.object(m.os, "rmdir") as remove, \
                patch.object(m.os, "chmod") as chmod, patch.object(m.os, "close") as closed:
            with self.assertRaisesRegex(m.PreparedFixtureError, "^output_rejected$"):
                m.prepare_import_parent(helper)
        mkdir.assert_called_once_with("import", 0o700, dir_fd=80)
        opened.assert_not_called()
        remove.assert_not_called()
        chmod.assert_not_called()
        closed.assert_called_once_with(80)

    def test_new_import_parent_wrong_mode_or_owner_is_rejected(self):
        helper = MagicMock()
        helper.open_directory.return_value = 80
        good = types.SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_uid=10001, st_gid=10001)
        for wrong in (types.SimpleNamespace(st_mode=stat.S_IFDIR | 0o755,
                                            st_uid=10001, st_gid=10001),
                      types.SimpleNamespace(st_mode=stat.S_IFDIR | 0o700, st_uid=0, st_gid=0)):
            with self.subTest(info=wrong), patch.object(m.os, "fstat", side_effect=[good, wrong]), \
                    patch.object(m.os, "mkdir"), patch.object(m.os, "open", return_value=81), \
                    patch.object(m.os, "close"):
                with self.assertRaisesRegex(m.PreparedFixtureError, "output_rejected"):
                    m.prepare_import_parent(helper)


if __name__ == "__main__":
    unittest.main()
