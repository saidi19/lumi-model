"""AST/literal checks only; never import or run the OCI/native worker probe.

Only three repository Python source files are read. No runtime/library binary,
model, dataset, fixture process, Docker client or remote service is accessed.
"""

from __future__ import annotations

import ast
import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts/run_sft_oci_worker_probe_v1.py"
FROZEN = ROOT / "scripts/run_sft_worker_run_torch_fixture_v1.py"
WORKER = ROOT / "src/lumi/posttraining_sft_worker_run_v1.py"


def literal(tree, name):
    values = [node.value for node in tree.body if isinstance(node, ast.Assign)
              and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id == name]
    if len(values) != 1:
        raise AssertionError("literal binding missing")
    return ast.literal_eval(values[0])


def function(tree, name):
    values = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
              and node.name == name]
    if len(values) != 1:
        raise AssertionError("function binding missing")
    return values[0]


def calls(tree, name):
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and ast.unparse(node.func) == name]


def expression(value):
    return ast.dump(ast.parse(value, mode="eval").body, include_attributes=False)


def equal(node, value):
    return ast.dump(node, include_attributes=False) == expression(value)


class OciWorkerProbeSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(PROBE.read_text(encoding="utf-8"))
        cls.frozen_raw = FROZEN.read_bytes()
        cls.frozen = ast.parse(cls.frozen_raw)
        cls.worker = ast.parse(WORKER.read_text(encoding="utf-8"))

    def assert_need(self, node, text, code):
        self.assertTrue(any(len(item.args) == 2 and equal(item.args[0], text)
                            and equal(item.args[1], repr(code)) for item in calls(node, "need")),
                        text)

    def test_fixed_layout_and_schema_match_build_contract(self):
        expected = {
            "SCHEMA": "lumi.sft-oci-worker-probe.v0.1",
            "LOCK_SCHEMA": "lumi.sft-linux-cpu-release-lock.v0.1",
            "BASE_SCHEMA": "lumi.sft-oci-base-inventory.v0.1",
            "SOURCE_SCHEMA": "lumi.sft-oci-source-inventory.v0.1",
            "SOURCE_ROOT": "/opt/lumi-release/source",
            "CONTROL_ROOT": "/opt/lumi-release/control",
            "RUNTIME": "/usr/local/bin/python3.12", "RUNTIME_PREFIX": "/usr/local",
            "STDLIB_ROOT": "/usr/local/lib/python3.12",
            "DEPENDENCY_ROOT": "/usr/local/lib/python3.12/site-packages",
        }
        for key, value in expected.items():
            with self.subTest(binding=key):
                self.assertEqual(literal(self.tree, key), value)

    def test_frozen_worker_source_is_exact_and_unchanged(self):
        self.assertEqual(len(self.frozen_raw), 39948)
        self.assertEqual(hashlib.sha256(self.frozen_raw).hexdigest(),
                         "53c50c92965ed1cef0c9624c57367c31ed7ebfb9d7cfb34257188ad179d333d0")
        self.assertEqual(literal(self.tree, "FROZEN_SIZE"), len(self.frozen_raw))
        self.assertEqual(literal(self.tree, "FROZEN_SHA256"),
                         hashlib.sha256(self.frozen_raw).hexdigest())

    def test_native_script_is_not_imported_or_compiled_by_tests(self):
        imported = [alias.name for node in self.tree.body if isinstance(node, ast.Import)
                    for alias in node.names]
        self.assertEqual(set(imported), {
            "hashlib", "importlib.metadata", "importlib.util", "json", "os", "re", "stat", "sys",
        })
        self.assertFalse(any(name.split(".")[0] in {"torch", "numpy", "transformers", "lumi"}
                             for name in imported))

    def test_guard_is_real_linux_amd64_cpython312_isolated_nonroot(self):
        guard = function(self.tree, "runtime_guard")
        self.assert_need(guard,
                         "sys.platform == 'linux' and os.uname().machine == 'x86_64' "
                         "and sys.implementation.name == 'cpython' "
                         "and sys.version_info[:2] == (3, 12) and sys.flags.isolated == 1 "
                         "and sys.dont_write_bytecode and os.geteuid() == os.getegid() == 10001",
                         "runtime_rejected")
        self.assert_need(guard,
                         "str(Path(sys.executable).resolve(strict=True)) == RUNTIME "
                         "and sys.prefix == sys.base_prefix == RUNTIME_PREFIX", "runtime_rejected")

    def test_no_parent_ml_before_or_after_fixture(self):
        self.assert_need(function(self.tree, "runtime_guard"),
                         "'torch' not in sys.modules and 'transformers' not in sys.modules",
                         "runtime_rejected")
        run = function(self.tree, "run")
        self.assert_need(run,
                         "report.get('status') == 'passed' and _RANKS == 6 and _INVOCATIONS == 3 "
                         "and 'torch' not in sys.modules and 'transformers' not in sys.modules",
                         "fixture_failed")

    def test_control_files_use_three_external_raw_hash_pins(self):
        values = [(ast.literal_eval(call.args[0]), ast.literal_eval(call.args[1]))
                  for call in calls(function(self.tree, "load_controls"), "control")]
        self.assertEqual(values, [
            ("software-lock.json", "LUMI_RELEASE_LOCK_SHA256"),
            ("base-system-inventory.json", "LUMI_RELEASE_BASE_INVENTORY_SHA256"),
            ("source-inventory.json", "LUMI_RELEASE_SOURCE_INVENTORY_SHA256"),
        ])

    def test_control_reads_are_nofollow_nonblocking_size_and_hash_bound(self):
        selected = function(self.tree, "pinned_bytes")
        opened = calls(selected, "os.open")
        self.assertEqual(len(opened), 1)
        self.assertTrue(equal(opened[0].args[1],
                              "os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK"))
        self.assertTrue(any(equal(call.args[0],
                                  "stat.S_ISREG(before.st_mode) and 0 < before.st_size <= maximum")
                            and equal(call.args[1], "code") for call in calls(selected, "need")))
        self.assertTrue(any(equal(call.args[0], "hashlib.sha256(raw).hexdigest() == expected")
                            for call in calls(selected, "need")))
        self.assertEqual(len(calls(selected, "os.close")), 1)

    def test_control_inventory_and_map_caps_are_fixed(self):
        for name, expected in {
            "MAXIMUM_CONTROL_BYTES": 16777216, "MAXIMUM_INVENTORY_FILES": 65536,
            "MAXIMUM_MODULES": 16384, "MAXIMUM_DISTRIBUTIONS": 256,
            "MAXIMUM_MAP_BYTES": 1048576, "MAXIMUM_MAP_LINES": 8192,
            "MAXIMUM_REPORT_BYTES": 8192,
        }.items():
            with self.subTest(cap=name):
                self.assertEqual(literal(self.tree, name), expected)

    def test_loader_binds_frozen_file_to_source_inventory_before_exec(self):
        prepare = function(self.tree, "prepare")
        self.assert_need(prepare,
                         "path in _STATE['source'] "
                         "and _STATE['source'][path]['bytes'] == FROZEN_SIZE "
                         "and _STATE['source'][path]['sha256'] == FROZEN_SHA256",
                         "fixture_binding_rejected")
        pin = calls(prepare, "pinned_bytes")[0]
        compiled = calls(prepare, "compile")[0]
        self.assertTrue(equal(pin.args[1], "FROZEN_SHA256"))
        self.assertTrue(equal(compiled.args[0], "raw"))
        self.assertLess(pin.lineno, compiled.lineno)

    def test_only_two_frozen_fixture_attributes_are_rebound(self):
        changes = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    for part in ast.walk(target):
                        if isinstance(part, ast.Attribute) and equal(part.value, "_FIXTURE"):
                            changes.append(part.attr)
        self.assertEqual(sorted(changes), ["outcome", "run_invocation"])

    def test_rank_worker_target_and_original_api_are_unchanged(self):
        run_invocation = function(self.frozen, "run_invocation")
        constructor = calls(run_invocation, "context.Process")[0]
        target = next(keyword.value for keyword in constructor.keywords if keyword.arg == "target")
        self.assertTrue(equal(target, "rank_worker"))
        self.assertEqual([arg.arg for arg in function(self.frozen, "outcome").args.args],
                         ["pipe", "value"])

    def test_inputs_witness_is_before_original_phase_send(self):
        selected = function(self.tree, "observed_outcome")
        inputs = selected.body[1]
        self.assertIsInstance(inputs, ast.If)
        self.assertTrue(equal(inputs.test,
                              "value.get('status') == 'phase' and value.get('phase') == 'inputs'"))
        self.assertEqual(len(calls(inputs, "witness")), 2)
        self.assert_need(inputs, "_INPUT_WITNESS is None", "witness_sequence_rejected")
        witness_call = min(call.lineno for call in calls(selected, "witness"))
        send_call = calls(selected, "_ORIGINAL_OUTCOME")[0]
        self.assertLess(witness_call, send_call.lineno)

    def test_inputs_barrier_precedes_optimizer_steps_in_accepted_fixture(self):
        phases = [node for node in self.frozen.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "CONTROL_PHASES"
                          for target in node.targets)]
        self.assertEqual(len(phases), 1)
        self.assertEqual(ast.literal_eval(phases[0].value.args[0]),
                         ("inputs", "restored", "checkpoint"))
        run = function(self.worker, "run")
        barrier = next(call for call in calls(run, "self._sync")
                       if equal(call.args[0], "'inputs'"))
        steps = calls(run, "self._step.perform")
        self.assertEqual(len(steps), 1)
        self.assertLess(barrier.lineno, steps[0].lineno)

    def test_two_witnesses_are_transported_only_in_existing_private_result(self):
        selected = function(self.tree, "observed_outcome")
        self.assertEqual(len(calls(selected, "witness")), 2)
        self.assertEqual(len(calls(selected, "_ORIGINAL_OUTCOME")), 1)
        self.assertFalse(calls(selected, "os.write"))
        self.assertFalse(calls(selected, "print"))
        self.assert_need(selected,
                         "_INPUT_WITNESS is not None and 'oci_origin_witness' not in value",
                         "witness_sequence_rejected")

    def test_parent_consumes_exact_two_ranks_and_returns_original_pair(self):
        selected = function(self.tree, "observed_invocation")
        self.assertEqual(len(calls(selected, "_ORIGINAL_INVOCATION")), 1)
        self.assert_need(selected, "set(reports) == {0, 1}", "witness_sequence_rejected")
        removed = calls(selected, "reports[rank].pop")
        self.assertEqual(len(removed), 1)
        self.assertTrue(equal(removed[0].args[0], "'oci_origin_witness'"))
        returned = [node for node in ast.walk(selected) if isinstance(node, ast.Return)]
        self.assertEqual(len(returned), 1)
        self.assertTrue(equal(returned[0].value, "(reports, totals)"))

    def test_observer_does_not_start_or_retry_workers(self):
        names = {ast.unparse(node.func) for node in ast.walk(self.tree)
                 if isinstance(node, ast.Call)}
        for forbidden in ("multiprocessing.Process", "subprocess.run", "subprocess.Popen",
                          "os.fork", "os.execve", "os.system", "context.Process"):
            self.assertNotIn(forbidden, names)
        self.assertEqual(len(calls(function(self.tree, "run"), "_FIXTURE.run")), 1)

    def test_core_package_origins_have_strict_release_roots(self):
        selected = function(self.tree, "module_origins")
        self.assert_need(selected,
                         "category == 'source' and below(resolved, SOURCE_ROOT + '/src/lumi')",
                         "module_origin_rejected")
        for name in ("torch", "transformers"):
            self.assert_need(selected,
                             "category == 'dependency' and below(resolved, DEPENDENCY_ROOT + '/"
                             + name + "')", "module_origin_rejected")
        self.assert_need(selected, "core == {'lumi', 'torch', 'transformers', 'torch._C'}",
                         "module_origin_rejected")

    def test_source_and_base_module_origins_require_inventory_membership(self):
        selected = function(self.tree, "file_origin")
        self.assert_need(selected, "row is not None and row['bytes'] == info.st_size",
                         "module_origin_rejected")
        self.assert_need(selected,
                         "resolved in _STATE['base'] "
                         "and _STATE['base'][resolved]['bytes'] == info.st_size",
                         "module_origin_rejected")

    def test_versions_compare_complete_installed_distribution_map(self):
        selected = function(self.tree, "distribution_versions")
        lookup = calls(selected, "importlib.metadata.distributions")[0]
        self.assertEqual(len(lookup.keywords), 1)
        self.assertEqual(lookup.keywords[0].arg, "path")
        self.assertTrue(equal(lookup.keywords[0].value, "[DEPENDENCY_ROOT]"))
        self.assert_need(selected, "observed == _STATE['versions']", "distribution_rejected")

    def test_generated_helper_is_exact_and_not_a_tmp_import_allowance(self):
        selected = function(self.tree, "module_origins")
        self.assert_need(selected,
                         "spec is not None and spec.origin == 'torch-git' "
                         "and type(spec.loader).__name__ == '_StringLoader' "
                         "and type(spec.loader).__module__ "
                         "== 'torch.distributed.nn.jit.instantiator' "
                         "and path in (None, '<_remote_module_non_scriptable>.py')",
                         "module_origin_rejected")
        self.assertNotIn("/tmp", ast.unparse(selected))

    def test_mapped_code_is_bounded_current_process_only(self):
        selected = function(self.tree, "native_origins")
        opened = calls(selected, "os.open")
        self.assertEqual(len(opened), 1)
        self.assertTrue(equal(opened[0].args[0], "'/proc/self/maps'"))
        self.assert_need(selected, "len(raw) <= MAXIMUM_MAP_BYTES", "native_origin_rejected")
        self.assert_need(selected, "1 <= len(lines) <= MAXIMUM_MAP_LINES", "native_origin_rejected")

    def test_mapped_native_files_need_identity_base_or_dependency_and_torch_c(self):
        selected = function(self.tree, "native_origins")
        self.assert_need(selected, "not path.endswith(' (deleted)')", "native_origin_rejected")
        self.assert_need(selected,
                         "below(resolved, DEPENDENCY_ROOT) or resolved in _STATE['base']",
                         "native_origin_rejected")
        self.assert_need(selected, "torch_native in mapped and RUNTIME in mapped",
                         "native_origin_rejected")
        self.assertTrue(any("info.st_ino" in ast.unparse(call.args[0])
                            and "os.major(info.st_dev)" in ast.unparse(call.args[0])
                            for call in calls(selected, "need")))

    def test_mapping_scope_is_file_backed_executable_not_all_dynamic_code(self):
        selected = function(self.tree, "native_origins")
        conditions = [node.test for node in ast.walk(selected) if isinstance(node, ast.If)]
        self.assertTrue(any(equal(node, "'x' not in fields[1]") for node in conditions))
        self.assertTrue(any(equal(node, "len(fields) == 5 or fields[5].startswith('[')")
                            for node in conditions))
        nonclaims = set(literal(self.tree, "NONCLAIMS"))
        self.assertTrue({"all_executable_code_origin_authenticated",
                         "complete_dynamic_dependency_closure_verified",
                         "large_library_rehash_performed"} <= nonclaims)

    def test_witness_never_rehashes_large_library_files(self):
        for name in ("witness", "module_origins", "native_origins", "file_origin"):
            with self.subTest(function=name):
                selected = function(self.tree, name)
                self.assertFalse(calls(selected, "pinned_bytes"))
                self.assertFalse(calls(selected, "hashlib.sha256"))
                self.assertFalse(calls(selected, "Path.read_bytes"))

    def test_spawn_reimports_same_prepare_before_frozen_target_resolution(self):
        guards = [node for node in self.tree.body if isinstance(node, ast.If)]
        self.assertEqual(len(guards), 1)
        self.assertTrue(equal(guards[0].test, "__name__ in ('__main__', '__mp_main__')"))
        selected = function(self.tree, "entry")
        self.assertEqual(len(calls(selected, "prepare")), 1)
        self.assertEqual(len(calls(selected, "run")), 1)
        self.assertLess(calls(selected, "prepare")[0].lineno, calls(selected, "run")[0].lineno)

    def test_native_output_is_suppressed_before_preparation(self):
        selected = function(self.tree, "entry")
        prepared = calls(selected, "prepare")[0]
        redirections = calls(selected, "os.dup2")
        self.assertEqual(len(redirections), 2)
        self.assertTrue(all(call.lineno < prepared.lineno for call in redirections))
        self.assertEqual({ast.literal_eval(call.args[1]) for call in redirections}, {1, 2})

    def test_public_report_has_only_counts_digests_fixed_flags_and_frozen_report(self):
        selected = function(self.tree, "run")
        returned = [node.value for node in ast.walk(selected) if isinstance(node, ast.Return)]
        self.assertEqual(len(returned), 1)
        self.assertIsInstance(returned[0], ast.Dict)
        keys = {ast.literal_eval(key) for key in returned[0].keys if key is not None}
        self.assertFalse(keys & {"path", "paths", "pid", "module", "exception", "traceback"})
        self.assertTrue({"origin_counts", "source_commit", "software_lock_sha256",
                         "origin_observations", "worker_run"} <= keys)
        self.assertTrue(any(equal(call.args[0], "NONCLAIMS") and equal(call.args[1], "False")
                            for call in calls(selected, "dict.fromkeys")))

    def test_private_training_and_real_host_admission_stay_false(self):
        self.assertTrue({
            "real_sft_execution_permitted", "production_eligible", "real_model_or_data_access",
            "cuda_used", "real_host_admission_verified",
            "host_configuration_independently_verified",
        } <= set(literal(self.tree, "NONCLAIMS")))


if __name__ == "__main__":
    unittest.main()
