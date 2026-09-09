"""AST/literal checks only; no diagnostic, builder or native fixture imports.

Reads small repository source/configuration text to bind one new diagnostic.
Never imports or runs Docker, an OCI observer, a fixture, ML, or a report helper.
"""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_sft_cpu_origin_diagnostic_v1.py"
WORKFLOW = ROOT / ".github/workflows/sft-cpu-origin-diagnostic.yml"
OLD_WORKFLOW = ROOT / ".github/workflows/sft-oci-release-candidate.yml"
APPROVED_PARENT = "edaa86e238223cf3a1ecfb6292640d8bc43e72de"
NEW_FILES = frozenset(
    (
        ".github/workflows/sft-cpu-origin-diagnostic.yml",
        "scripts/build_sft_cpu_origin_diagnostic_v1.py",
        "scripts/run_sft_cpu_origin_diagnostic_v1.py",
        "scripts/sft_cpu_origin_diagnostic_report_v1.py",
        "tests/test_sft_cpu_origin_diagnostic_source_v1.py",
        "tests/test_sft_cpu_diagnostic_builder_source_v1.py",
        "tests/test_sft_cpu_origin_diagnostic_report_v1.py",
        "docs/sft-execution-status-v0.51.md",
    )
)
FROZEN = {
    "FROZEN_BUILDER": (
        "scripts/build_sft_oci_release_candidate_v1.py",
        "458faf97c62689c9d7cc71a6eb89a4f84fef8c68bbcf9279b949dee450565044",
    ),
    "FROZEN_PROBE": (
        "scripts/run_sft_oci_worker_probe_v1.py",
        "bed636e8cc08546a6be1b43c00c2f516fdc92dbf907dca674f7fb5aa7b5cb769",
    ),
    "FROZEN_FIXTURE": (
        "scripts/run_sft_worker_run_torch_fixture_v1.py",
        "53c50c92965ed1cef0c9624c57367c31ed7ebfb9d7cfb34257188ad179d333d0",
    ),
    "FROZEN_BASE": (
        "scripts/build_sft_oci_base_inventory_v1.py",
        "34dc6a70b6d074808b864f53b4d511456423786aa62068ea771253fd4f8f29d1",
    ),
    "LOCK_MODULE": (
        "src/lumi/posttraining_sft_oci_lock_v1.py",
        "9f70737db527c038d2ddd54992ce6d1ff999e9c7f204f4828a3f64fa08543ed3",
    ),
}
FROZEN_WORKFLOWS = {
    ".github/workflows/sft-oci-release-candidate.yml": (
        "85c64d8ae86993a7d99bd7abc03ebf2da33afa17b252b6bab9ad4b26618a35ef"
    ),
    ".github/workflows/source-checks.yml": (
        "ed75ad40b262c47864e080fff9ff3551106201b9fe4002e8a665ab9a688bbb8e"
    ),
    ".github/workflows/spark-source-checks.yml": (
        "1171cef435580f7eb1736c5250f980f91890a781669e1aca1853bfaaee37caa8"
    ),
    ".github/workflows/tests.yml": (
        "09c8f4fa326953d5f3df270bf5e18c4a2d95ddebfbed7089ec4122429bd27ab2"
    ),
}


def literal(tree, name):
    values = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    if len(values) != 1:
        raise AssertionError("literal binding missing")
    return ast.literal_eval(values[0])


def function(tree, name):
    values = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(values) != 1:
        raise AssertionError("function binding missing")
    return values[0]


def calls(tree, name):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == name
    ]


def equal(node, value):
    expected = ast.dump(ast.parse(value, mode="eval").body, include_attributes=False)
    return ast.dump(node, include_attributes=False) == expected


def conjuncts(node):
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        return [part for value in node.values for part in conjuncts(value)]
    return [ast.dump(node, include_attributes=False)]


def strings(tree):
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and type(node.value) is str
    ]


class CpuDiagnosticBuilderSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(BUILDER.read_text(encoding="utf-8"))
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")
        cls.old_workflow = OLD_WORKFLOW.read_text(encoding="utf-8")

    def assert_need(self, tree, expected):
        observed = [
            part
            for call in calls(tree, "need")
            if len(call.args) == 1
            for part in conjuncts(call.args[0])
        ]
        wanted = conjuncts(ast.parse(expected, mode="eval").body)
        self.assertTrue(all(part in observed for part in wanted), expected)

    def test_contract_schema_parent_and_resource_caps(self):
        for key, value in {
            "SCHEMA": "lumi.sft-cpu-origin-diagnostic-build.v0.1",
            "APPROVED_PARENT": APPROVED_PARENT,
            "MAXIMUM_WORK_SECONDS": 840,
            "MAXIMUM_RETAINED_BYTES": 33554432,
            "MAXIMUM_SOURCE_SCRIPT_BYTES": 131072,
            "MAXIMUM_DIAGNOSTIC_BYTES": 16384,
        }.items():
            with self.subTest(binding=key):
                self.assertEqual(literal(self.tree, key), value)

    def test_frozen_native_sources_and_pure_lock_are_exact_unchanged_bytes(self):
        for binding, (relative, expected) in FROZEN.items():
            with self.subTest(binding=binding):
                self.assertEqual(literal(self.tree, binding), relative)
                self.assertEqual(literal(self.tree, binding + "_SHA256"), expected)
                raw = (ROOT / relative).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), expected)
                self.assertLessEqual(len(raw), 131072)
        raw = (ROOT / "configs/lumi-sft-linux-cpu-release-v0.1.lock.json").read_bytes()
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            "26bde6a0163e5db9d8d4564117546482601f4d0beb09f8705be5fc9183594965",
        )

    def test_old_oci_source_spark_and_test_workflows_remain_byte_identical(self):
        for relative, expected in FROZEN_WORKFLOWS.items():
            with self.subTest(workflow=relative):
                raw = (ROOT / relative).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), expected)

    def test_new_observer_and_pure_report_helper_have_final_exact_source_pins(self):
        for binding, relative in (
            ("DIAGNOSTIC", "scripts/run_sft_cpu_origin_diagnostic_v1.py"),
            ("VALIDATOR", "scripts/sft_cpu_origin_diagnostic_report_v1.py"),
        ):
            with self.subTest(binding=binding):
                self.assertEqual(literal(self.tree, binding), relative)
                expected = literal(self.tree, binding + "_SHA256")
                self.assertRegex(expected, r"\A[0-9a-f]{64}\Z")
                raw = (ROOT / relative).read_bytes()
                self.assertLessEqual(len(raw), 131072)
                self.assertEqual(hashlib.sha256(raw).hexdigest(), expected)

    def test_new_file_allowlist_is_exact_and_does_not_match_old_oci_triggers(self):
        observed = literal(self.tree, "NEW_FILES")
        self.assertEqual(len(observed), len(NEW_FILES))
        self.assertEqual(set(observed), NEW_FILES)
        old_patterns = [
            line.strip()[2:]
            for line in self.old_workflow.split("permissions:", 1)[0].splitlines()
            if line.startswith("      - ")
        ]
        self.assertEqual(len(old_patterns), 7)
        for relative in NEW_FILES:
            with self.subTest(path=relative):
                self.assertFalse(
                    any(fnmatch.fnmatchcase(relative, pattern) for pattern in old_patterns)
                )

    def test_first_attempt_remote_guard_precedes_loading_native_source(self):
        main = function(self.tree, "main")
        guard_values = (
            "sys.argv[1:] == ['--cpu-origin-diagnostic']",
            "sys.platform == 'linux' and platform.machine() == 'x86_64'",
            "sys.version_info[:2] == (3, 12) and sys.flags.dont_write_bytecode",
            "os.environ.get('GITHUB_ACTIONS') == 'true'",
            "os.environ.get('LUMI_CPU_ORIGIN_DIAGNOSTIC_CI') == '1'",
            "os.environ.get('GITHUB_REPOSITORY') == 'saidi19/lumi-model'",
            "os.environ.get('GITHUB_RUN_ATTEMPT') == '1'",
            "re.fullmatch('[0-9a-f]{40}', commit)",
        )
        first_load = min(call.lineno for call in calls(main, "load_pinned"))
        for expected in guard_values:
            with self.subTest(guard=expected):
                self.assert_need(main, expected)
        self.assertEqual(len([call for call in calls(main, "need") if call.lineno < first_load]), 8)
        self.assertLess(first_load, calls(main, "output.mkdir")[0].lineno)

    def test_pinned_source_is_bounded_and_exact_hash_before_compile(self):
        source = function(self.tree, "pinned_source")
        self.assert_need(source, "re.fullmatch('[0-9a-f]{64}', expected)")
        self.assert_need(source, "not path.is_symlink()")
        self.assert_need(
            source, "0 < len(raw) <= MAXIMUM_SOURCE_SCRIPT_BYTES and digest(raw) == expected"
        )
        reads = calls(source, "source.read")
        self.assertEqual(len(reads), 1)
        self.assertTrue(equal(reads[0].args[0], "MAXIMUM_SOURCE_SCRIPT_BYTES + 1"))
        loading = function(self.tree, "load_pinned")
        pins = calls(loading, "pinned_source")
        compiled = calls(loading, "compile")
        self.assertEqual(len(pins), 1)
        self.assertEqual(len(compiled), 1)
        self.assertLess(pins[0].lineno, compiled[0].lineno)
        self.assertTrue(equal(compiled[0].args[0], "raw"))

    def test_only_probe_and_source_scripts_are_frozen_builder_global_overrides(self):
        changed = {}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and equal(target.value, "frozen"):
                        self.assertNotIn(target.attr, changed)
                        changed[target.attr] = node.value
        self.assertEqual(set(changed), {"PROBE", "SOURCE_SCRIPTS"})
        self.assertTrue(equal(changed["PROBE"], "IMAGE_DIAGNOSTIC"))
        self.assertTrue(
            equal(changed["SOURCE_SCRIPTS"], "(FROZEN_FIXTURE, FROZEN_PROBE, DIAGNOSTIC)")
        )

    def test_approved_parent_precedes_snapshot_and_image_construction(self):
        main = function(self.tree, "main")
        self.assert_need(
            main, "builder.run(['git', 'rev-parse', 'HEAD^']).decode().strip() == APPROVED_PARENT"
        )
        check = next(call for call in calls(main, "need") if "APPROVED_PARENT" in ast.unparse(call))
        self.assertLess(check.lineno, calls(main, "builder.snapshot")[0].lineno)
        self.assertLess(check.lineno, calls(main, "builder.image")[0].lineno)

    def test_commit_scope_allows_only_exact_new_additions_before_source_snapshot(self):
        main = function(self.tree, "main")
        comparisons = [
            call
            for call in calls(main, "builder.run")
            if call.args
            and equal(call.args[0], "['git', 'diff', '--name-status', APPROVED_PARENT, commit]")
        ]
        self.assertEqual(len(comparisons), 1)
        self.assertTrue(
            any(key.arg == "cap" and equal(key.value, "16384") for key in comparisons[0].keywords)
        )
        self.assert_need(main, "changes == ['A\\t' + path for path in NEW_FILES]")
        self.assertLess(comparisons[0].lineno, calls(main, "builder.snapshot")[0].lineno)
        self.assertEqual(list(literal(self.tree, "NEW_FILES")), sorted(NEW_FILES))

    def test_software_lock_and_diagnostic_schema_are_verified_before_image(self):
        main = function(self.tree, "main")
        spec = calls(main, "validator.spec_from_source")
        self.assertEqual(len(spec), 1)
        self.assertTrue(equal(spec[0].args[0], "diagnostic_source"))
        self.assertTrue(equal(spec[0].args[1], "DIAGNOSTIC_SHA256"))
        selected = calls(main, "lock_module.validate_lock")
        self.assertEqual(len(selected), 1)
        self.assertTrue(equal(selected[0].args[0], "lock_raw"))
        self.assertTrue(equal(selected[0].args[1], "frozen.LOCK_SHA256"))
        self.assertLess(spec[0].lineno, calls(main, "builder.image")[0].lineno)
        self.assertLess(selected[0].lineno, calls(main, "builder.image")[0].lineno)

    def test_exact_software_totals_and_binding_receipt_precede_diagnostic(self):
        main = function(self.tree, "main")
        self.assert_need(
            main, "builder.download_responses == 34 and builder.download_bytes == 248715637"
        )
        receipt = [
            call
            for call in calls(main, "record")
            if len(call.args) == 3
            and equal(call.args[1], "'software-binding-before-diagnostic.json'")
        ]
        self.assertEqual(len(receipt), 1)
        invocation = calls(main, "run_diagnostic")
        self.assertEqual(len(invocation), 1)
        self.assertLess(receipt[0].lineno, invocation[0].lineno)
        previous = [
            call for call in calls(main, "summary.update") if call.lineno < receipt[0].lineno
        ]
        self.assertEqual(len(previous), 1)
        self.assertIn("image_id", strings(previous[0]))
        self.assertIn("controls", strings(previous[0]))
        self.assertIn("diagnostic_source_sha256", strings(previous[0]))

    def test_same_image_command_uses_only_the_new_probe_with_inherited_synthetic_cli(self):
        run = function(self.tree, "run_diagnostic")
        self.assert_need(run, "info['Image'] == image_id and info['Path'] == frozen.PYTHON")
        self.assert_need(run, "info['Args'] == ['-I', '-B', IMAGE_DIAGNOSTIC, '--synthetic-only']")
        appends = calls(run, "command.append")
        self.assertEqual(len(appends), 1)
        self.assertTrue(equal(appends[0].args[0], "image_id"))

    def test_inspection_requires_readonly_nonroot_no_network_no_binds_and_no_gpu(self):
        run = function(self.tree, "run_diagnostic")
        for condition in (
            "config['User'] == '10001:10001' and not config.get('Volumes')",
            "host['ReadonlyRootfs'] is True and host['Privileged'] is False",
            "host['NetworkMode'] == 'none' and host['CapDrop'] == ['ALL']",
            "set(host['SecurityOpt']) & {'no-new-privileges', 'no-new-privileges:true'}",
            "not host.get('Binds') and not host.get('Devices') and not host.get('DeviceRequests')",
            "host['Tmpfs'] == frozen.TMPFS and host['Memory'] == 4294967296",
            "host['MemorySwap'] == 4294967296 and host['ShmSize'] == 67108864",
            "host['PidsLimit'] == 256 and host['NanoCpus'] == 2000000000",
            "all(mount['Type'] == 'tmpfs' and mount['Destination'] in frozen.TMPFS "
            "and not mount.get('Source') for mount in info['Mounts'])",
        ):
            with self.subTest(condition=condition):
                self.assert_need(run, condition)

    def test_exclusive_start_marker_is_after_inspection_before_the_only_diagnostic_start(self):
        run = function(self.tree, "run_diagnostic")
        starts = [
            call
            for call in calls(self.tree, "builder.run")
            if call.args and equal(call.args[0], "['docker', 'start', '--attach', container]")
        ]
        self.assertEqual(len(starts), 1)
        marker = [
            call
            for call in calls(run, "record")
            if len(call.args) == 3 and equal(call.args[1], "'diagnostic-start-marker.json'")
        ]
        self.assertEqual(len(marker), 1)
        self.assertLess(marker[0].lineno, starts[0].lineno)
        inspections = [
            call
            for call in calls(run, "builder.run")
            if call.args and equal(call.args[0], "['docker', 'inspect', container]")
        ]
        self.assertEqual(len(inspections), 2)
        self.assertLess(min(call.lineno for call in inspections), marker[0].lineno)
        for loop in (node for node in ast.walk(run) if isinstance(node, ast.For | ast.While)):
            self.assertNotIn(starts[0], list(ast.walk(loop)))
        record = function(self.tree, "record")
        self.assertTrue(
            any(
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "open"
                and call.args
                and equal(call.args[0], "'xb'")
                for call in ast.walk(record)
                if isinstance(call, ast.Call)
            )
        )

    def test_raw_closed_report_validator_is_used_before_retaining_diagnostic_bytes(self):
        run = function(self.tree, "run_diagnostic")
        validation = calls(run, "validator.validate_report")
        self.assertEqual(len(validation), 1)
        self.assertTrue(
            equal(validation[0], "validator.validate_report(raw, controls, report_spec)")
        )
        writes = calls(run, "target.write")
        self.assertEqual(len(writes), 1)
        self.assertTrue(equal(writes[0].args[0], "raw"))
        self.assertLess(validation[0].lineno, writes[0].lineno)
        self.assert_need(run, "len(raw) <= MAXIMUM_DIAGNOSTIC_BYTES")
        self.assert_need(
            run,
            "state['Running'] is False and state['ExitCode'] == 0 and state['OOMKilled'] is False",
        )
        self.assert_need(run, "not builder.run(['docker', 'diff', container]).strip()")
        returned = [node for node in ast.walk(run) if isinstance(node, ast.Return)]
        self.assertEqual(len(returned), 1)
        self.assertTrue(equal(returned[0].value, "(report, digest(raw))"))

    def test_never_calls_frozen_acceptance_main_or_archiving(self):
        names = {
            ast.unparse(call.func) for call in ast.walk(self.tree) if isinstance(call, ast.Call)
        }
        self.assertTrue(
            names.isdisjoint(
                {
                    "frozen.main",
                    "frozen.run",
                    "frozen.Builder.accept",
                    "builder.accept",
                    "frozen.hashed_file",
                    "builder.archive",
                }
            )
        )
        forbidden = {"save", "push", "login", "--gpus", "--privileged", "cpu-release-image.tar"}
        self.assertTrue(set(strings(self.tree)).isdisjoint(forbidden))
        self.assertNotIn("cpu-release-image.tar", self.workflow)
        self.assertNotIn("docker push", self.workflow)

    def test_success_means_diagnostic_completed_not_acceptance_or_recovery(self):
        flags = literal(self.tree, "FALSE_FLAGS")
        self.assertEqual(
            set(flags),
            {
                "acceptance_passed",
                "recovery_performed",
                "production_eligible",
                "real_sft_execution_permitted",
                "real_model_or_data_access",
                "gpu_used",
                "registry_published",
                "image_archived",
                "real_host_admission_verified",
                "complete_dynamic_dependency_closure_verified",
                "full_ci_passed",
                "private_integration_passed",
            },
        )
        main = function(self.tree, "main")
        for update in calls(main, "summary.update"):
            self.assertTrue(set(strings(update.args[0])).isdisjoint(flags))
        returned = [node for node in ast.walk(main) if isinstance(node, ast.Return)]
        self.assertEqual(len(returned), 1)
        self.assertTrue(
            equal(returned[0].value, "0 if summary['status'] == 'diagnostic_completed' else 2")
        )
        self.assertIn("underlying_status", strings(main))

    def test_cleanup_is_bounded_exact_owned_retirement_not_an_execution_retry(self):
        main = function(self.tree, "main")
        retirements = [
            call
            for call in calls(main, "builder.run")
            if any(
                keyword.arg == "retirement" and equal(keyword.value, "True")
                for keyword in call.keywords
            )
        ]
        self.assertEqual(len(retirements), 1)
        self.assertTrue(
            equal(retirements[0].args[0], "['docker', 'rm', '--force', builder.container]")
        )
        self.assertTrue(
            any(
                keyword.arg == "timeout" and equal(keyword.value, "15")
                for keyword in retirements[0].keywords
            )
        )
        self.assertTrue(
            any(
                equal(call, "signal.signal(signal.SIGTERM, terminate)")
                for call in calls(main, "signal.signal")
            )
        )
        self.assert_need(main, "retained <= MAXIMUM_RETAINED_BYTES")

    def test_workflow_is_one_private_first_attempt_with_parent_available(self):
        text = self.workflow
        self.assertIn("github.event.repository.private == true && github.run_attempt == 1", text)
        self.assertIn("      - codex/lumi-posttraining-product-v0.1", text)
        self.assertIn("persist-credentials: false", text)
        self.assertIn("fetch-depth: 2", text)
        self.assertIn("permissions:\n  contents: read\n", text)
        self.assertIn("cancel-in-progress: false", text)
        self.assertIn("runs-on: ubuntu-24.04", text)
        self.assertIn("timeout-minutes: 20", text)
        self.assertIn("timeout --signal=TERM --kill-after=15s 15m", text)
        self.assertEqual(text.count("scripts/build_sft_cpu_origin_diagnostic_v1.py "), 1)
        self.assertNotIn("workflow_dispatch:", text)
        self.assertNotIn("schedule:", text)
        self.assertNotIn("packages: write", text)
        self.assertNotIn("id-token: write", text)

    def test_workflow_retains_only_small_private_diagnostic_files(self):
        text = self.workflow
        self.assertEqual(text.count("uses: actions/upload-artifact@v4"), 1)
        self.assertIn("if: always()", text)
        self.assertIn("artifacts/lumi-sft-cpu-origin-diagnostic-v0.1/*.json", text)
        self.assertIn("artifacts/lumi-sft-cpu-origin-diagnostic-v0.1/*.log", text)
        self.assertIn("retention-days: 30", text)
        self.assertNotIn("*.tar", text)
        self.assertNotIn("compression-level: 0", text)
        self.assertNotIn("docker/login-action", text)

    def test_native_scripts_are_never_imported_by_this_source_suite(self):
        own = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        imports = [
            alias.name
            for node in ast.walk(own)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        self.assertEqual(set(imports), {"ast", "fnmatch", "hashlib", "unittest"})
        from_imports = {node.module for node in ast.walk(own) if isinstance(node, ast.ImportFrom)}
        self.assertEqual(from_imports, {"__future__", "pathlib"})
        self.assertFalse(calls(own, "exec"))
        self.assertFalse(calls(own, "compile"))
        self.assertFalse(calls(own, "subprocess.run"))


if __name__ == "__main__":
    unittest.main()
