"""AST/literal/JSON checks only; never import or execute native OCI builders.

Reads the candidate builder source, workflow text, and small software lock only.
The sole imported project module is the pure byte-only lock validator. No image,
wheel, runtime binary, dataset, native fixture, Docker process or network access.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_sft_oci_release_candidate_v1.py"
WORKFLOW = ROOT / ".github/workflows/sft-oci-release-candidate.yml"
LOCK = ROOT / "configs/lumi-sft-linux-cpu-release-v0.1.lock.json"
LOCK_SHA256 = "26bde6a0163e5db9d8d4564117546482601f4d0beb09f8705be5fc9183594965"
PACKAGE = "_lumi_synthetic_oci_candidate_source_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
before_import = set(sys.modules)
lock_validator = importlib.import_module(PACKAGE + ".posttraining_sft_oci_lock_v1")
IMPORT_ADDITIONS = set(sys.modules) - before_import


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


def equal(node, value):
    expected = ast.dump(ast.parse(value, mode="eval").body, include_attributes=False)
    return ast.dump(node, include_attributes=False) == expected


def strings(tree):
    return [node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and type(node.value) is str]


def assigned(tree, name):
    return [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]


def conjuncts(node):
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        return [part for value in node.values for part in conjuncts(value)]
    return [ast.dump(node, include_attributes=False)]


class OciReleaseCandidateSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_source = BUILDER.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.raw_source)
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")
        cls.raw_lock = LOCK.read_bytes()
        cls.lock = lock_validator.validate_lock(cls.raw_lock, LOCK_SHA256)

    def assert_need(self, tree, expected):
        observed = [part for call in calls(tree, "need") if len(call.args) == 1
                    for part in conjuncts(call.args[0])]
        wanted = conjuncts(ast.parse(expected, mode="eval").body)
        self.assertTrue(all(part in observed for part in wanted), expected)

    def assert_command(self, tree, expected):
        matches = [call for call in calls(tree, "self.run")
                   if call.args and equal(call.args[0], expected)]
        self.assertEqual(len(matches), 1, expected)
        return matches[0]

    def test_builder_schema_lock_path_and_external_pin(self):
        self.assertEqual(literal(self.tree, "SCHEMA"), "lumi.sft-oci-release-candidate.v0.1")
        self.assertEqual(literal(self.tree, "LOCK_PATH"), str(LOCK.relative_to(ROOT)))
        self.assertEqual(literal(self.tree, "LOCK_SHA256"), LOCK_SHA256)
        self.assertEqual(literal(self.tree, "PYTHON"), "/usr/local/bin/python3.12")
        self.assertEqual(literal(self.tree, "PROBE"),
                         "/opt/lumi-release/source/scripts/run_sft_oci_worker_probe_v1.py")

    def test_real_lock_full_file_hash_and_exact_lf_encoding(self):
        self.assertEqual(len(self.raw_lock), 12187)
        self.assertEqual(hashlib.sha256(self.raw_lock).hexdigest(), LOCK_SHA256)
        self.assertTrue(self.raw_lock.endswith(b"\n"))
        self.assertFalse(self.raw_lock.endswith(b"\n\n"))
        canonical = json.dumps(
            self.lock, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode("ascii") + b"\n"
        self.assertEqual(canonical, self.raw_lock)
        self.assertEqual(self.lock, json.loads(self.raw_lock))

    def test_real_lock_pins_exact_base_manifest_config_and_python(self):
        self.assertEqual(self.lock["base"], {
            "repository": "docker.io/library/python",
            "digest": "sha256:9c47360a2a0355e2da18516d0b1c2126ec22c195d2185e97347c9d98398c5bef",
            "config_digest":
                "sha256:59bf1d95c965f12dfc14afaf5af778fc1dbe5b372bd5c281645fc34d4c75d4e7",
            "python_version": "3.12.14", "tag_reference": "3.12-slim-bookworm",
        })
        self.assertEqual(self.lock["platform"], "linux/amd64")
        self.assertEqual(self.lock["python_abi"], "cp312")

    def test_real_cpu_lock_has_exact_count_bytes_and_torch_artifact(self):
        rows = self.lock["requirements"]
        self.assertEqual(len(rows), 34)
        self.assertEqual(sum(row["bytes"] for row in rows), 248715637)
        torch = next(row for row in rows if row["name"] == "torch")
        self.assertEqual(torch["version"], "2.14.0+cpu")
        self.assertEqual(torch["bytes"], 196253793)
        self.assertEqual(torch["filename"],
                         "torch-2.14.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl")
        self.assertEqual(torch["url"], "https://download.pytorch.org/whl/cpu/"
                         "torch-2.14.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl")
        self.assertEqual(torch["sha256"],
                         "a09987c95ec4cffdb6df798d3d641558110a334cfccce22f2f046d83142bc260")
        self.assertEqual(self.lock["policy"], {
            "device": "cpu", "strategy": "ddp", "max_context_window": 4095,
            "production_eligible": False, "real_sft_execution_permitted": False,
        })

    def test_runtime_guard_precedes_writes_imports_or_native_work(self):
        main = function(self.tree, "main")
        for guard in (
            "sys.argv[1:] == ['--cpu-release-candidate']",
            "sys.platform == 'linux' and platform.machine() == 'x86_64'",
            "sys.version_info[:2] == (3, 12) and sys.flags.dont_write_bytecode",
            "os.environ.get('GITHUB_ACTIONS') == 'true'",
            "os.environ.get('LUMI_OCI_RELEASE_CI') == '1'",
            "os.environ.get('GITHUB_REPOSITORY') == 'saidi19/lumi-model'",
            "re.fullmatch('[0-9a-f]{40}', commit)",
        ):
            with self.subTest(guard=guard):
                self.assert_need(main, guard)
        first_write = calls(main, "output.mkdir")[0].lineno
        early_needs = [call for call in calls(main, "need") if call.lineno < first_write]
        self.assertEqual(len(early_needs), 8)
        self.assertLess(first_write, calls(main, "spec.loader.exec_module")[0].lineno)
        self.assertLess(first_write, calls(main, "builder.snapshot")[0].lineno)

    def test_real_lock_validation_precedes_snapshot_and_download(self):
        main = function(self.tree, "main")
        validation = calls(main, "lock_module.validate_lock")
        self.assertEqual(len(validation), 1)
        self.assertTrue(equal(validation[0].args[0], "lock_raw"))
        self.assertTrue(equal(validation[0].args[1], "LOCK_SHA256"))
        self.assertLess(validation[0].lineno, calls(main, "builder.snapshot")[0].lineno)
        self.assertLess(validation[0].lineno, calls(main, "builder.image")[0].lineno)

    def test_source_snapshot_is_exact_commit_tracked_blob_objects_only(self):
        snapshot = function(self.tree, "snapshot")
        self.assert_need(snapshot,
                         "self.run(['git', 'rev-parse', 'HEAD']).decode().strip() == commit")
        self.assert_need(snapshot,
                         "not self.run(['git', 'status', '--porcelain', "
                         "'--untracked-files=no']).strip()")
        self.assert_command(snapshot,
                            "['git', 'ls-tree', '-r', '-z', commit, '--', "
                            "'src/lumi', *SOURCE_SCRIPTS]")
        blob = self.assert_command(snapshot, "['git', 'cat-file', 'blob', blob]")
        self.assertTrue(any(key.arg == "cap" and equal(key.value, "2097152")
                            for key in blob.keywords))
        self.assert_need(snapshot, "mode in ('100644', '100755') and kind == 'blob'")
        self.assert_need(snapshot,
                         "relative.startswith('src/lumi/') or relative in SOURCE_SCRIPTS")
        self.assert_need(snapshot, "'..' not in relative.split('/') and '\\\\' not in relative")
        self.assertFalse(any(isinstance(call.func, ast.Attribute)
                             and call.func.attr in {"read_bytes", "read_text", "open", "copytree"}
                             for call in ast.walk(snapshot) if isinstance(call, ast.Call)))
        self.assertFalse(calls(snapshot, "shutil.copytree"))
        payloads = assigned(snapshot, "payload")
        self.assertEqual(len(payloads), 1)
        self.assertIs(payloads[0].value, blob)
        writes = calls(snapshot, "target.write_bytes")
        self.assertEqual(len(writes), 1)
        self.assertTrue(equal(writes[0].args[0], "payload"))

    def test_snapshot_source_allowlist_and_bounds_are_explicit(self):
        self.assertEqual(literal(self.tree, "SOURCE_SCRIPTS"), (
            "scripts/run_sft_worker_run_torch_fixture_v1.py",
            "scripts/run_sft_oci_worker_probe_v1.py",
        ))
        snapshot = function(self.tree, "snapshot")
        self.assert_need(snapshot,
                         "total <= MAXIMUM_SOURCE_BYTES and len(entries) < MAXIMUM_SOURCE_FILES")
        self.assert_need(snapshot,
                         "len(entries) > 10 and "
                         "set(SOURCE_SCRIPTS) <= {x['path'] for x in entries}")
        self.assertIn("lumi.sft-oci-source-inventory.v0.1", strings(snapshot))
        self.assertIn("source_commit", strings(snapshot))
        self.assertIn("sha256", strings(snapshot))

    def test_base_execution_reference_is_digest_not_informational_tag(self):
        image = function(self.tree, "image")
        reference = assigned(image, "reference")
        self.assertEqual(len(reference), 1)
        self.assertTrue(equal(reference[0].value, "base['repository'] + '@' + base['digest']"))
        self.assertNotIn("tag_reference", strings(image))
        self.assert_command(image, "['docker', 'pull', '--platform=linux/amd64', reference]")
        self.assert_need(image,
                         "value['Id'] == base['config_digest'] "
                         "and value['Architecture'] == 'amd64'")
        self.assert_need(image,
                         "value['Os'] == 'linux' and 'PYTHON_VERSION=' + base['python_version'] "
                         "in value['Config']['Env']")

    def test_base_inventory_is_guarded_readonly_native_work_not_local_test(self):
        image = function(self.tree, "image")
        self.assert_command(image,
                            "['git', 'show', "
                            "commit + ':scripts/build_sft_oci_base_inventory_v1.py']")
        self.assert_command(image,
                            "['docker', 'create', '-i', *CONTAINER_OPTIONS, '-e', "
                            "'LUMI_OCI_BASE_INVENTORY_CI=1', '--entrypoint', PYTHON, reference, "
                            "'-I', '-B', '-c', inventory_source, '--base-inventory-only']")
        self.assert_command(image,
                            "['docker', 'start', '--attach', '--interactive', base_container]")
        self.assert_need(image, "base_state['Running'] is False and base_state['ExitCode'] == 0")
        self.assert_command(image, "['docker', 'rm', base_container]")
        self.assert_need(image,
                         "base_doc['base_digest'] == base['digest'] "
                         "and base_doc['config_digest'] == base['config_digest']")

    def test_wheels_have_no_redirect_retry_fallback_or_source_distribution(self):
        redirect = function(self.tree, "redirect_request")
        self.assertEqual(len(redirect.body), 1)
        self.assertIsInstance(redirect.body[0], ast.Raise)
        self.assertTrue(equal(redirect.body[0].exc, "BuildError('redirect_rejected')"))
        wheels = function(self.tree, "wheels")
        self.assertEqual(len(calls(wheels, "opener.open")), 1)
        self.assertEqual(len([node for node in ast.walk(wheels) if isinstance(node, ast.For)]), 1)
        self.assertFalse(any(isinstance(node, ast.Try) for node in ast.walk(wheels)))
        request = calls(wheels, "urllib.request.Request")[0]
        self.assertTrue(equal(request.args[0], "row['url']"))
        self.assert_need(wheels, "self.download_responses == len(lock['requirements'])")

    def test_every_download_requires_exact_size_and_sha_before_build(self):
        wheels = function(self.tree, "wheels")
        self.assert_need(wheels,
                         "response.status == 200 and response.headers.get('Content-Encoding') "
                         "in (None, 'identity')")
        self.assert_need(wheels, "size is None or int(size) == row['bytes']")
        self.assert_need(wheels, "total <= row['bytes'] and self.download_bytes <= 1073741824")
        self.assert_need(wheels, "total == row['bytes'] and hasher.hexdigest() == row['sha256']")
        self.assert_need(wheels, "time.monotonic() < self.deadline")
        image = function(self.tree, "image")
        self.assertLess(calls(image, "self.wheels")[0].lineno,
                        self.assert_command(image,
                                            "['docker', 'build', '--network=none', '--pull=false', "
                                            "'--no-cache', '--platform=linux/amd64', '--iidfile', "
                                            "str(work / 'image.id'), str(work)]").lineno)

    def test_install_is_offline_binary_hash_pinned_no_dependency_resolution(self):
        image = function(self.tree, "image")
        dockerfile = assigned(image, "dockerfile")[0].value
        text = "".join(strings(dockerfile))
        for option in (
            "--mount=type=bind,source=wheels,target=/wheels,readonly", "--no-cache-dir",
            "--no-compile", "--no-index", "--find-links=/wheels", "--only-binary=:all:",
            "--require-hashes", "--no-deps", "-m pip check",
        ):
            with self.subTest(option=option):
                self.assertIn(option, text)
        self.assertNotIn("--extra-index-url", text)
        self.assertNotIn("--trusted-host", text)
        self.assertNotIn("--upgrade", text)
        self.assertNotIn("curl ", text)
        self.assertNotIn("wget ", text)

    def test_cpu_offline_no_credentials_environment_is_baked(self):
        text = "".join(strings(assigned(function(self.tree, "image"), "dockerfile")[0].value))
        for environment in (
            'CUDA_VISIBLE_DEVICES=""', "GLOO_SOCKET_IFNAME=lo", "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1", "HF_HUB_DISABLE_TELEMETRY=1",
            "AWS_EC2_METADATA_DISABLED=true", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
            "USER 10001:10001", "LUMI_RELEASE_LOCK_SHA256=", "LUMI_RELEASE_SOURCE_COMMIT=",
        ):
            with self.subTest(environment=environment):
                self.assertIn(environment, text)

    def test_container_options_are_fixed_readonly_nonroot_no_network(self):
        self.assertEqual(literal(self.tree, "CONTAINER_OPTIONS"), (
            "--platform=linux/amd64", "--network=none", "--read-only", "--user=10001:10001",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--cpus=2", "--memory=4g",
            "--memory-swap=4g", "--pids-limit=256", "--ulimit=core=0", "--ulimit=nofile=1024:1024",
        ))
        self.assertEqual(literal(self.tree, "TMPFS"), {
            "/tmp": "rw,noexec,nosuid,nodev,size=536870912,mode=1777",
        })

    def test_acceptance_creates_and_inspects_same_image_without_host_binds(self):
        accept = function(self.tree, "accept")
        creation = assigned(accept, "command")[0]
        self.assertTrue(equal(creation.value,
                              "['docker', 'create', *CONTAINER_OPTIONS, '--shm-size=67108864']"))
        appended = calls(accept, "command.append")
        self.assertEqual(len(appended), 1)
        self.assertTrue(equal(appended[0].args[0], "image_id"))
        for condition in (
            "info['Image'] == image_id and info['Path'] == PYTHON",
            "info['Args'] == ['-I', '-B', PROBE, '--synthetic-only']",
            "config['User'] == '10001:10001' and not config.get('Volumes')",
            "host['ReadonlyRootfs'] is True and host['Privileged'] is False",
            "host['NetworkMode'] == 'none' and host['CapDrop'] == ['ALL']",
            "set(host['SecurityOpt']) & {'no-new-privileges', 'no-new-privileges:true'}",
            "not host.get('Binds') and not host.get('Devices') and not host.get('DeviceRequests')",
            "host['Tmpfs'] == TMPFS and host['Memory'] == 4294967296",
            "host['PidsLimit'] == 256 and host['NanoCpus'] == 2000000000",
            "all(mount['Type'] == 'tmpfs' and mount['Destination'] in TMPFS "
            "and not mount.get('Source') for mount in info['Mounts'])",
        ):
            with self.subTest(condition=condition):
                self.assert_need(accept, condition)

    def test_one_acceptance_start_after_inspection_and_no_retry(self):
        accept = function(self.tree, "accept")
        start = self.assert_command(accept, "['docker', 'start', '--attach', container]")
        starts = [call for call in calls(accept, "self.run")
                  if call.args and isinstance(call.args[0], ast.List)
                  and len(call.args[0].elts) >= 2
                  and equal(call.args[0].elts[0], "'docker'")
                  and equal(call.args[0].elts[1], "'start'")]
        self.assertEqual(starts, [start])
        # The separate pinned-base inventory starts once before image build;
        # it is not another worker acceptance or an acceptance retry.
        all_starts = [call for call in calls(self.tree, "self.run")
                      if call.args and isinstance(call.args[0], ast.List)
                      and len(call.args[0].elts) >= 2
                      and equal(call.args[0].elts[0], "'docker'")
                      and equal(call.args[0].elts[1], "'start'")]
        self.assertEqual(len(all_starts), 2)
        inspections = [call for call in calls(accept, "self.run")
                       if call.args and equal(call.args[0], "['docker', 'inspect', container]")]
        self.assertEqual(len(inspections), 2)
        self.assertLess(min(call.lineno for call in inspections), start.lineno)
        for loop in [node for node in ast.walk(accept)
                     if isinstance(node, ast.For | ast.While)]:
            self.assertNotIn(start, list(ast.walk(loop)))
        self.assertEqual(len(calls(function(self.tree, "main"), "builder.accept")), 1)

    def test_acceptance_requires_probe_success_zero_exit_no_oom_and_no_diff(self):
        accept = function(self.tree, "accept")
        self.assert_need(accept, "len(raw) <= 8192")
        self.assert_need(accept,
                         "probe['schema'] == 'lumi.sft-oci-worker-probe.v0.1' "
                         "and probe['status'] == 'passed'")
        self.assert_need(accept,
                         "state['Running'] is False and state['ExitCode'] == 0 "
                         "and state['OOMKilled'] is False")
        self.assert_need(accept, "not self.run(['docker', 'diff', container]).strip()")
        self.assert_command(accept, "['docker', 'rm', container]")

    def test_probe_binds_controls_and_exact_worker_observation_totals(self):
        image = function(self.tree, "image")
        returned = [node for node in ast.walk(image) if isinstance(node, ast.Return)]
        self.assertEqual(len(returned), 1)
        self.assertTrue(equal(returned[0].value,
                              "(image_id, {'software_lock_sha256': LOCK_SHA256, "
                              "'source_commit': commit, 'base_inventory_sha256': "
                              "digest(base_inventory), 'source_inventory_sha256': "
                              "digest(source_inventory)})"))
        accept = function(self.tree, "accept")
        self.assert_need(accept,
                         "all(probe[key] == value for key, value in expected_controls.items())")
        self.assert_need(accept,
                         "probe['rank_workers_witnessed'] == 6 "
                         "and probe['origin_observations'] == 12 "
                         "and probe['invocations_witnessed'] == 3")
        self.assert_need(accept, "probe['worker_run']['status'] == 'passed'")
        self.assert_need(accept,
                         "probe['worker_run']['controlled_rank_optimizer_steps'] == 8 "
                         "and probe['worker_run']['global_optimizer_steps_completed'] == 4")

    def test_probe_requires_narrow_origin_evidence_without_overclaim(self):
        accept = function(self.tree, "accept")
        self.assert_need(accept,
                         "probe['inputs_barrier_and_final_origins_verified'] is True "
                         "and probe['declared_distribution_versions_verified'] is True "
                         "and probe['file_backed_executable_origins_verified'] is True "
                         "and probe['parent_ml_import_absent'] is True")
        self.assert_need(accept, "all(probe[key] is False for key in ("
                         "'production_eligible', 'real_sft_execution_permitted', "
                         "'real_model_or_data_access', 'cuda_used', "
                         "'complete_dynamic_dependency_closure_verified', "
                         "'all_executable_code_origin_authenticated', "
                         "'real_host_admission_verified'))")
        main = function(self.tree, "main")
        self.assertTrue(any(equal(call.args[0],
                                  "(output / 'cpu-worker-probe.json').read_bytes()")
                            for call in calls(main, "digest")))

    def test_archive_uses_accepted_image_id_not_registry(self):
        main = function(self.tree, "main")
        archives = [call for call in calls(main, "builder.run")
                    if call.args and equal(call.args[0],
                                          "['docker', 'image', 'save', image_id]")]
        self.assertEqual(len(archives), 1)
        self.assertTrue(any(key.arg == "stdout_path" and equal(key.value, "archive")
                            for key in archives[0].keywords))
        self.assertLess(calls(main, "builder.accept")[0].lineno, archives[0].lineno)
        hash_call = calls(main, "hashed_file")[0]
        self.assertTrue(equal(hash_call.args[0], "archive"))
        self.assertTrue(equal(hash_call.args[1], "MAXIMUM_IMAGE_BYTES"))
        self.assertFalse(any(value in {"push", "login", "--gpus", "--privileged"}
                             for value in strings(self.tree)))

    def test_runtime_content_output_and_image_observation_caps_are_declared(self):
        for key, value in {
            "MAXIMUM_SECONDS": 1140, "MAXIMUM_COMMAND_OUTPUT": 16777216,
            "MAXIMUM_SOURCE_BYTES": 33554432, "MAXIMUM_SOURCE_FILES": 4096,
            "MAXIMUM_IMAGE_BYTES": 3221225472, "MAXIMUM_RETAINED_BYTES": 4294967296,
        }.items():
            with self.subTest(cap=key):
                self.assertEqual(literal(self.tree, key), value)
        self.assert_need(function(self.tree, "image"),
                         "image_info['Id'] == image_id "
                         "and image_info['Size'] <= MAXIMUM_IMAGE_BYTES")
        self.assert_need(function(self.tree, "main"), "retained_bytes <= MAXIMUM_RETAINED_BYTES")

    def test_archive_stream_cap_precedes_every_write(self):
        run = function(self.tree, "run")
        self.assert_need(run, "archive_bytes <= MAXIMUM_IMAGE_BYTES")
        cap = next(call for call in calls(run, "need")
                   if equal(call.args[0], "archive_bytes <= MAXIMUM_IMAGE_BYTES"))
        writes = calls(run, "archive.write")
        self.assertEqual(len(writes), 1)
        self.assertLess(cap.lineno, writes[0].lineno)
        self.assertTrue(equal(writes[0].args[0], "chunk"))
        opening = calls(run, "stdout_path.open")
        self.assertEqual(len(opening), 1)
        self.assertTrue(equal(opening[0].args[0], "'xb'"))

    def test_archive_open_failure_is_inside_child_cleanup_handler(self):
        run = function(self.tree, "run")
        opening = calls(run, "stdout_path.open")[0]
        guarded = [node for node in run.body if isinstance(node, ast.Try)]
        self.assertEqual(len(guarded), 1)
        self.assertIn(opening, [item for node in guarded[0].body for item in ast.walk(node)])
        self.assertTrue(any(calls(handler, "os.killpg") for handler in guarded[0].handlers))
        final_calls = [item for node in guarded[0].finalbody for item in ast.walk(node)]
        self.assertTrue(any(isinstance(item, ast.Call)
                            and equal(item.func, "archive.close") for item in final_calls))

    def test_retirement_has_separate_15_second_exact_owned_cleanup_budget(self):
        run = function(self.tree, "run")
        branch = next(node for node in run.body if isinstance(node, ast.If)
                      and equal(node.test, "retirement"))
        self.assert_need(branch, "self.container is not None")
        self.assert_need(branch, "command == ['docker', 'rm', '--force', self.container]")
        assignments = assigned(branch, "remaining")
        self.assertEqual(len(assignments), 2)
        self.assertTrue(equal(assignments[0].value, "15"))
        self.assertTrue(equal(assignments[1].value, "self.deadline - time.monotonic()"))
        special = [call for call in ast.walk(self.tree) if isinstance(call, ast.Call)
                   and any(keyword.arg == "retirement" and equal(keyword.value, "True")
                           for keyword in call.keywords)]
        self.assertEqual(len(special), 1)
        self.assertTrue(equal(special[0].args[0],
                              "['docker', 'rm', '--force', builder.container]"))
        self.assertTrue(any(keyword.arg == "timeout" and equal(keyword.value, "15")
                            for keyword in special[0].keywords))

    def test_term_handler_and_unique_logs_preserve_failure_cleanup_evidence(self):
        main = function(self.tree, "main")
        handler = calls(main, "signal.signal")
        self.assertEqual(len(handler), 1)
        self.assertTrue(equal(handler[0], "signal.signal(signal.SIGTERM, terminate)"))
        termination = function(self.tree, "terminate")
        self.assertEqual(len(termination.body), 1)
        self.assertIsInstance(termination.body[0], ast.Raise)
        self.assertTrue(equal(termination.body[0].exc, "BuildError('terminated')"))
        run = function(self.tree, "run")
        log_path = assigned(run, "log_path")[0].value
        self.assertTrue(equal(log_path,
                              "self.output / "
                              "f'software-command-{self.command_number:04d}-failure.log'"))
        opening = calls(run, "log_path.open")
        self.assertEqual(len(opening), 1)
        self.assertTrue(equal(opening[0].args[0], "'xb'"))
        self.assertTrue(equal(assigned(run, "log")[0].value, "bytes(stdout + stderr)[-65536:]"))

    def test_native_command_launcher_has_deadline_output_cap_no_shell_or_retry(self):
        run = function(self.tree, "run")
        self.assertEqual(len(calls(run, "subprocess.Popen")), 1)
        spawn = calls(run, "subprocess.Popen")[0]
        self.assertNotIn("shell", {keyword.arg for keyword in spawn.keywords})
        self.assertTrue(any(keyword.arg == "start_new_session" and equal(keyword.value, "True")
                            for keyword in spawn.keywords))
        self.assert_need(run, "remaining > 0")
        self.assert_need(run, "time.monotonic() < limit")
        self.assert_need(run, "len(stdout) + len(stderr) <= cap")
        self.assertEqual(len(calls(run, "os.killpg")), 1)
        self.assertTrue(equal(calls(run, "os.killpg")[0].args[0], "proc.pid"))

    def test_unproven_closure_origin_and_real_training_claims_remain_false(self):
        claims = literal(self.tree, "FALSE_CLAIMS")
        self.assertEqual(set(claims), {
            "production_eligible", "real_sft_execution_permitted", "real_model_or_data_access",
            "gpu_used", "registry_published", "real_host_admission_verified",
            "complete_dynamic_dependency_closure_verified", "publisher_signature_verified",
            "reproducible_build_verified", "private_integration_passed", "full_ci_passed",
        })
        main = function(self.tree, "main")
        self.assertTrue(any(equal(call, "dict.fromkeys(FALSE_CLAIMS, False)")
                            for call in calls(main, "dict.fromkeys")))
        updates = calls(main, "report.update")
        for update in updates:
            self.assertTrue(set(strings(update.args[0])).isdisjoint(claims))
        self.assertIn("not a reproducible-build claim", ast.get_docstring(self.tree))

    def test_workflow_is_private_read_only_permissions_and_one_bounded_run(self):
        text = self.workflow
        self.assertIn("permissions:\n  contents: read\n", text)
        self.assertIn("if: github.event.repository.private == true", text)
        self.assertIn("runs-on: ubuntu-24.04", text)
        self.assertIn("timeout-minutes: 25", text)
        self.assertIn("timeout --signal=TERM --kill-after=15s 20m", text)
        self.assertEqual(text.count("scripts/build_sft_oci_release_candidate_v1.py "), 1)
        self.assertIn("persist-credentials: false", text)
        self.assertIn("cancel-in-progress: false", text)
        self.assertNotIn("packages: write", text)
        self.assertNotIn("id-token: write", text)
        self.assertNotIn("docker/login-action", text)

    def test_workflow_retains_private_evidence_and_image_without_registry_publication(self):
        text = self.workflow
        self.assertEqual(text.count("uses: actions/upload-artifact@v4"), 2)
        self.assertEqual(text.count("retention-days: 30"), 2)
        self.assertIn("artifacts/lumi-sft-oci-release-candidate-v0.1/*.json", text)
        self.assertIn("artifacts/lumi-sft-oci-release-candidate-v0.1/*.log", text)
        self.assertIn("artifacts/lumi-sft-oci-release-candidate-v0.1/cpu-release-image.tar", text)
        self.assertIn("compression-level: 0", text)
        self.assertNotIn("docker push", text)
        self.assertNotIn("docker run", text)

    def test_test_imports_only_pure_project_validator_not_native_scripts(self):
        forbidden = {"torch", "numpy", "transformers", "tensorflow", "jax", "lumi"}
        self.assertFalse({name.split(".")[0] for name in IMPORT_ADDITIONS} & forbidden)
        self.assertFalse(any("build_sft_oci" in name or "run_sft_oci" in name
                             for name in IMPORT_ADDITIONS))


if __name__ == "__main__":
    unittest.main()
