"""Read-only AST/literal tests; NEVER import, compile or run either native script."""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts/build_sft_cpu_lookup_acceptance_v1.py"
WRAPPER_PATH = ROOT / "scripts/run_sft_cpu_lookup_acceptance_v1.py"
PUBLISH_FILES = (
    ".github/workflows/sft-cpu-lookup-acceptance.yml",
    "docs/sft-execution-status-v0.55.md",
    "scripts/build_sft_cpu_lookup_acceptance_v1.py",
    "scripts/run_sft_cpu_lookup_acceptance_v1.py",
    "scripts/run_sft_oci_worker_probe_v2.py",
    "scripts/sft_cpu_lookup_acceptance_report_v1.py",
    "tests/test_sft_cpu_lookup_acceptance_report_v1.py",
    "tests/test_sft_cpu_lookup_acceptance_source_v1.py",
    "tests/test_sft_cpu_version_lookup_v2.py",
)
FROZEN_WORKFLOWS = {
    "sft-oci-release-candidate.yml": (
        "85c64d8ae86993a7d99bd7abc03ebf2da33afa17b252b6bab9ad4b26618a35ef"
    ),
    "sft-cpu-origin-diagnostic.yml": (
        "9104b408d8e80426fd841d4399f8073c529676c061c5ac8b29bc5109e3aa88f1"
    ),
    "source-checks.yml": "ed75ad40b262c47864e080fff9ff3551106201b9fe4002e8a665ab9a688bbb8e",
    "spark-source-checks.yml": "1171cef435580f7eb1736c5250f980f91890a781669e1aca1853bfaaee37caa8",
    "tests.yml": "09c8f4fa326953d5f3df270bf5e18c4a2d95ddebfbed7089ec4122429bd27ab2",
}


def literal(tree, name):
    matches = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    if len(matches) != 1:
        raise AssertionError("exactly one literal binding required")
    return ast.literal_eval(matches[0])


def function(tree, name):
    return next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def calls(tree, name):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == name
    ]


def text(tree, name):
    return ast.unparse(function(tree, name))


class CpuLookupAcceptanceSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder_raw = BUILDER_PATH.read_bytes()
        cls.wrapper_raw = WRAPPER_PATH.read_bytes()
        cls.builder = ast.parse(cls.builder_raw)
        cls.wrapper = ast.parse(cls.wrapper_raw)
        cls.corrected_raw = (ROOT / "scripts/run_sft_oci_worker_probe_v2.py").read_bytes()
        cls.corrected = ast.parse(cls.corrected_raw)
        cls.frozen = ast.parse(
            (ROOT / "scripts/build_sft_oci_release_candidate_v1.py").read_bytes()
        )
        cls.workflow = (ROOT / ".github/workflows/sft-cpu-lookup-acceptance.yml").read_text()

    def test_native_scripts_are_not_imported_or_run_by_this_suite(self):
        for module in (
            "scripts.build_sft_cpu_lookup_acceptance_v1",
            "scripts.run_sft_cpu_lookup_acceptance_v1",
            "scripts.run_sft_oci_worker_probe_v2",
            literal(self.wrapper, "OBSERVER_NAME"),
        ):
            self.assertNotIn(module, sys.modules)
        imports = [
            node.module
            for node in ast.walk(ast.parse(Path(__file__).read_bytes()))
            if isinstance(node, ast.ImportFrom)
        ]
        self.assertEqual(imports, ["__future__", "pathlib"])

    def test_all_loaded_source_bindings_are_final_bounded_exact_hashes(self):
        for name in (
            "FROZEN_BUILDER",
            "OLD_PROBE",
            "PROBE",
            "FIXTURE",
            "BASE",
            "LOCK_MODULE",
            "WRAPPER",
            "DIAGNOSTIC_VALIDATOR",
            "ACCEPTANCE_VALIDATOR",
        ):
            with self.subTest(binding=name):
                raw = (ROOT / literal(self.builder, name)).read_bytes()
                expected = literal(self.builder, name + "_SHA256")
                self.assertRegex(expected, r"\A[0-9a-f]{64}\Z")
                self.assertEqual(hashlib.sha256(raw).hexdigest(), expected)
                self.assertLessEqual(len(raw), 131072)
        self.assertEqual(len(self.corrected_raw), literal(self.wrapper, "OBSERVER_SIZE"))
        self.assertEqual(
            hashlib.sha256(self.corrected_raw).hexdigest(),
            literal(self.wrapper, "OBSERVER_SHA256"),
        )

    def test_v2_is_only_the_approved_single_lookup_expression_change(self):
        old = (ROOT / "scripts/run_sft_oci_worker_probe_v1.py").read_bytes()
        before = b'str(vars(sys.modules[name]).get("__version__"))'
        after = b'str(getattr(sys.modules[name], "__version__", None))'
        self.assertEqual(old.count(before), 1)
        self.assertEqual(self.corrected_raw, old.replace(before, after, 1))
        self.assertIn(
            "corrected == old.replace(old_lookup, new_lookup, 1)", text(self.builder, "main")
        )

    def test_all_seventy_two_corrected_need_sites_keep_exact_line_mapping(self):
        expected = []
        for node in self.corrected.body:
            if isinstance(node, ast.FunctionDef):
                for index, call in enumerate(
                    sorted(calls(node, "need"), key=lambda c: c.lineno), 1
                ):
                    expected.append(
                        (node.name, call.lineno, call.end_lineno, f"{node.name}_check_{index:02d}")
                    )
        actual = literal(self.wrapper, "CHECKPOINTS")
        self.assertEqual(len(actual), 72)
        self.assertEqual(set(actual), set(expected))

    def test_original_need_workload_and_failure_hooks_remain_authoritative(self):
        self.assertEqual(
            ast.unparse(function(self.wrapper, "observed_need").body[-1]),
            "return _ORIGINALS['need'](condition, code)",
        )
        self.assertEqual(
            ast.unparse(function(self.wrapper, "diagnostic_record_failure").body[-1]),
            "return _RECORD_FAILURE(observed)",
        )
        writes = {
            ast.unparse(target)
            for node in ast.walk(self.wrapper)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Attribute) and "_V1" in ast.unparse(target)
        }
        self.assertEqual(
            writes,
            {
                "_V1.need",
                "_V1.observed_invocation",
                "_V1._FIXTURE.outcome",
                "_V1._FIXTURE.record_failure",
            },
        )
        raises = [
            node
            for node in ast.walk(function(self.wrapper, "origin_wrapper"))
            if isinstance(node, ast.Raise)
        ]
        self.assertEqual(len(raises), 1)
        self.assertIsNone(raises[0].exc)

    def test_wrapper_has_one_run_and_retains_its_full_returned_observer_report(self):
        self.assertEqual(len(calls(self.wrapper, "_V1.run")), 1)
        entry = text(self.wrapper, "entry")
        self.assertIn("observer_report = _V1.run()", entry)
        self.assertIn("'observer_report': observer_report", entry)
        self.assertIn("'diagnostic': value", entry)
        self.assertIn("__mp_main__", ast.unparse(self.wrapper.body[-1]))
        self.assertEqual(literal(self.wrapper, "MAXIMUM_ENVELOPE_BYTES"), 32768)
        self.assertIn("len(raw) > MAXIMUM_ENVELOPE_BYTES", entry)

    def test_raw_output_suppressed_and_diagnostic_does_not_grant_acceptance(self):
        entry = text(self.wrapper, "entry")
        for required in ("os.dup2(sink, 1)", "os.dup2(sink, 2)", "os.close(output)"):
            self.assertIn(required, entry)
        self.assertEqual(len(calls(self.wrapper, "os.write")), 1)
        self.assertFalse(calls(self.wrapper, "print"))
        for field in ("acceptance_passed", "recovery_performed", "real_sft_execution_permitted"):
            self.assertIn(field, literal(self.wrapper, "FALSE_FLAGS"))
        self.assertIn("dict.fromkeys(FALSE_FLAGS, False)", text(self.wrapper, "report"))
        self.assertNotIn("str(error)", self.wrapper_raw.decode())
        self.assertNotIn("repr(error)", self.wrapper_raw.decode())

    def test_wrapper_guard_precedes_hashbound_readonly_observer_load(self):
        prepare = function(self.wrapper, "prepare")
        self.assertLess(
            calls(prepare, "guard")[0].lineno, calls(prepare, "load_observer")[0].lineno
        )
        guard = text(self.wrapper, "guard")
        for required in (
            "sys.platform == 'linux'",
            "sys.flags.isolated == 1",
            "os.geteuid() == os.getegid() == 10001",
            "'--synthetic-only'",
            "'torch' not in sys.modules",
            "'transformers' not in sys.modules",
        ):
            self.assertIn(required, guard)
        loader = text(self.wrapper, "load_observer")
        for required in (
            "os.O_RDONLY",
            "os.O_NOFOLLOW",
            "os.O_NONBLOCK",
            "before.st_size != OBSERVER_SIZE",
            "OBSERVER_SHA256",
        ):
            self.assertIn(required, loader)

    def test_builder_first_attempt_and_platform_guards_precede_native_load(self):
        main = function(self.builder, "main")
        first_load = min(call.lineno for call in calls(main, "load_pinned"))
        guards = [
            ast.unparse(call.args[0]) for call in calls(main, "need") if call.lineno < first_load
        ]
        self.assertEqual(len(guards), 8)
        for expected in (
            "sys.argv[1:] == ['--cpu-lookup-acceptance']",
            "sys.platform == 'linux' and platform.machine() == 'x86_64'",
            "os.environ.get('GITHUB_RUN_ATTEMPT') == '1'",
            "os.environ.get('GITHUB_REPOSITORY') == 'saidi19/lumi-model'",
            "os.environ.get('LUMI_CPU_LOOKUP_ACCEPTANCE_CI') == '1'",
        ):
            self.assertIn(expected, guards)
        loader = function(self.builder, "load_pinned")
        self.assertLess(
            calls(loader, "pinned_source")[0].lineno, calls(loader, "compile")[0].lineno
        )

    def test_exact_parent_and_nine_additions_verified_before_source_snapshot(self):
        self.assertEqual(
            literal(self.builder, "APPROVED_PARENT"), "15127b62e702244923a6f56c1a936fb667ea4486"
        )
        self.assertEqual(literal(self.builder, "NEW_FILES"), PUBLISH_FILES)
        main = function(self.builder, "main")
        checks = calls(main, "need")
        for required in ("APPROVED_PARENT", "changes == ['A\\t' + path for path in NEW_FILES]"):
            check = next(node for node in checks if required in ast.unparse(node))
            self.assertLess(check.lineno, calls(main, "builder.snapshot")[0].lineno)

    def test_old_workflows_byte_identical_and_new_files_do_not_trigger_old_runs(self):
        for name, expected in FROZEN_WORKFLOWS.items():
            raw = (ROOT / ".github/workflows" / name).read_bytes()
            with self.subTest(workflow=name):
                self.assertEqual(hashlib.sha256(raw).hexdigest(), expected)
            if name not in ("sft-oci-release-candidate.yml", "sft-cpu-origin-diagnostic.yml"):
                continue
            patterns = [
                line.strip()[2:]
                for line in raw.decode().split("permissions:", 1)[0].splitlines()
                if line.startswith("      - ")
            ]
            for relative in PUBLISH_FILES:
                self.assertFalse(
                    any(fnmatch.fnmatchcase(relative, pat) for pat in patterns), (name, relative)
                )

    def test_safe_version_test_alias_avoids_old_wildcard_without_dropping_tests(self):
        alias = ROOT / "tests/test_sft_cpu_version_lookup_v2.py"
        self.assertIn(str(alias.relative_to(ROOT)), PUBLISH_FILES)
        self.assertNotIn("tests/test_sft_oci_version_lookup_v2.py", PUBLISH_FILES)
        self.assertFalse(
            fnmatch.fnmatchcase(str(alias.relative_to(ROOT)), "tests/test_sft_oci_*.py")
        )
        suite = ast.parse(alias.read_bytes())
        names = [
            node.name
            for node in ast.walk(suite)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        ]
        self.assertEqual(len(names), 20)

    def test_frozen_builder_overrides_are_only_probe_and_source_script_bindings(self):
        changed = {
            target.attr: ast.unparse(node.value)
            for node in ast.walk(self.builder)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Attribute) and ast.unparse(target.value) == "frozen"
        }
        self.assertEqual(
            changed, {"PROBE": "IMAGE_WRAPPER", "SOURCE_SCRIPTS": "(FIXTURE, PROBE, WRAPPER)"}
        )
        self.assertFalse(calls(self.builder, "frozen.main"))
        self.assertFalse(calls(self.builder, "builder.accept"))

    def test_one_cpu_start_has_exclusive_image_binding_and_marker_before_it(self):
        accept = function(self.builder, "run_acceptance")
        starts = [
            call
            for call in calls(self.builder, "builder.run")
            if ast.unparse(call.args[0]).startswith("['docker', 'start', '--attach'")
        ]
        self.assertEqual(len(starts), 1)
        marker = next(
            call
            for call in calls(accept, "record")
            if ast.literal_eval(call.args[1]) == "acceptance-start-marker.json"
        )
        self.assertLess(marker.lineno, starts[0].lineno)
        main = function(self.builder, "main")
        binding = next(
            call
            for call in calls(main, "record")
            if ast.literal_eval(call.args[1]) == "software-binding-before-acceptance.json"
        )
        self.assertLess(binding.lineno, calls(main, "run_acceptance")[0].lineno)
        self.assertIn(".open('xb')", text(self.builder, "record"))
        self.assertIn("not output.exists()", text(self.builder, "main"))

    def test_container_resource_and_isolation_inspection_precedes_start(self):
        accept = text(self.builder, "run_acceptance")
        for expected in (
            "config['User'] == '10001:10001'",
            "host['ReadonlyRootfs'] is True",
            "host['Privileged'] is False",
            "host['NetworkMode'] == 'none'",
            "host['CapDrop'] == ['ALL']",
            "'no-new-privileges'",
            "not host.get('Binds')",
            "not host.get('Devices')",
            "not host.get('DeviceRequests')",
            "host['Memory'] == 4294967296",
            "host['MemorySwap'] == 4294967296",
            "host['ShmSize'] == 67108864",
            "host['PidsLimit'] == 256",
            "host['NanoCpus'] == 2000000000",
            "mount['Type'] == 'tmpfs'",
        ):
            self.assertIn(expected, accept)
            self.assertLess(accept.index(expected), accept.index("'acceptance-start-marker.json'"))

    def test_envelope_validation_and_clean_exit_required_before_acceptance(self):
        accept = function(self.builder, "run_acceptance")
        self.assertEqual(len(calls(accept, "validator.validate_report")), 1)
        rendered = ast.unparse(accept)
        for expected in (
            "state['ExitCode'] == 0",
            "state['OOMKilled'] is False",
            "['docker', 'diff', container]",
            "'cpu-acceptance-envelope.json'",
            "value['observer_report']",
            "'cpu-worker-probe.json'",
        ):
            self.assertIn(expected, rendered)
        main = text(self.builder, "main")
        required = "result['diagnostic']['underlying_status'] == 'passed'"
        accepted = "summary['same_image_cpu_acceptance_passed'] = True"
        self.assertIn("need(result['observer_report'] is not None)", main)
        self.assertLess(main.index(required), main.index(accepted))

    def test_archive_is_separate_bounded_and_after_the_acceptance_receipt(self):
        main = text(self.builder, "main")
        self.assertLess(
            main.index("'cpu-acceptance-before-archive.json'"),
            main.index("['docker', 'image', 'save', image_id]"),
        )
        self.assertLess(
            main.index("frozen.hashed_file(archive, frozen.MAXIMUM_IMAGE_BYTES)"),
            main.index("'image_archived': True"),
        )
        self.assertEqual(literal(self.frozen, "MAXIMUM_IMAGE_BYTES"), 3221225472)
        self.assertEqual(literal(self.frozen, "MAXIMUM_RETAINED_BYTES"), 4294967296)
        self.assertIn("retained + MAXIMUM_REPORT_BYTES <= frozen.MAXIMUM_RETAINED_BYTES", main)
        self.assertIn("archive_bytes <= MAXIMUM_IMAGE_BYTES", ast.unparse(self.frozen))

    def test_runtime_report_caps_and_independent_owned_container_retirement(self):
        self.assertEqual(literal(self.builder, "MAXIMUM_WORK_SECONDS"), 1140)
        self.assertEqual(literal(self.builder, "MAXIMUM_REPORT_BYTES"), 32768)
        self.assertEqual(literal(self.builder, "MAXIMUM_SOURCE_SCRIPT_BYTES"), 131072)
        self.assertIn("timeout=240, cap=MAXIMUM_REPORT_BYTES", text(self.builder, "run_acceptance"))
        main = text(self.builder, "main")
        self.assertIn("timeout=15, retirement=True", main)
        self.assertIn("['docker', 'rm', '--force', builder.container]", main)
        self.assertIn("remaining = 15", ast.unparse(self.frozen))
        self.assertIn("started + MAXIMUM_WORK_SECONDS", main)

    def test_workflow_is_first_attempt_private_cpu_no_dispatch_and_no_early_image_upload(self):
        for expected in (
            "github.event.repository.private == true && github.run_attempt == 1",
            "fetch-depth: 2",
            "persist-credentials: false",
            "timeout-minutes: 25",
            "timeout-minutes: 21",
            "timeout --signal=TERM --kill-after=15s 20m",
            'CUDA_VISIBLE_DEVICES: ""',
            'HF_HUB_OFFLINE: "1"',
            'TRANSFORMERS_OFFLINE: "1"',
            "cancel-in-progress: false",
            "compression-level: 0",
            "retention-days: 30",
        ):
            self.assertIn(expected, self.workflow)
        self.assertNotIn("workflow_dispatch", self.workflow)
        self.assertNotIn("continue-on-error", self.workflow)
        image_step = self.workflow.split("- name: Retain only a fully accepted software image", 1)[
            1
        ]
        self.assertNotIn("if: always()", image_step)
        self.assertEqual(self.workflow.count("python -S -B scripts/"), 1)

    def test_new_builder_has_no_gpu_real_model_registry_or_remote_admission_claims(self):
        expected = {
            "production_eligible",
            "real_sft_execution_permitted",
            "real_model_or_data_access",
            "gpu_used",
            "registry_published",
            "real_host_admission_verified",
            "complete_dynamic_dependency_closure_verified",
            "publisher_signature_verified",
            "reproducible_build_verified",
            "private_integration_passed",
            "full_ci_passed",
        }
        self.assertEqual(set(literal(self.builder, "FALSE_CLAIMS")), expected)
        main = text(self.builder, "main")
        self.assertIn("dict.fromkeys(FALSE_CLAIMS, False)", main)
        for forbidden in ("docker push", "aws ", "ssh ", "from_pretrained", "--gpus", "curl "):
            self.assertNotIn(forbidden, self.builder_raw.decode())
        self.assertIn("builder.download_responses == 34", main)
        self.assertIn("builder.download_bytes == 248715637", main)


if __name__ == "__main__":
    unittest.main()
