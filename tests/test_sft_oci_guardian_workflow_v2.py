"""Source/AST contracts for the exact one-shot v2 workflow; no native actions."""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/sft-oci-guardian-acceptance-v2.yml"
TEXT = WORKFLOW.read_text()
PREPARE = ast.parse((ROOT / "scripts/prepare_sft_oci_guardian_v2.py").read_text())
OLD_PREPARE = ROOT / "scripts/prepare_sft_oci_inspect_diagnostic_v1.py"
OLD_TREE = ast.parse(OLD_PREPARE.read_text())
PARENT = "f3d8554d8999e32bb750fb8aa9020fca6818bf72"
FROZEN_PREPARER_SHA256 = "b53378d70bb8f7c7b826077f5512fb4cf10f27522d0acedf43439ad2a039f53f"
FROZEN_HISTORICAL = {
    ".github/workflows/sft-oci-guardian-acceptance.yml": (
        "e64dffedb7c7c55b9fcc787921ff157efce50b654a28d824ba215d26c42a458d"
    ),
    ".github/workflows/sft-oci-inspect-diagnostic.yml": (
        "c2b8c142a4bb7b04a3a98df6769e07874e195d8cab684fceecc1e3b36f21975a"
    ),
    "scripts/prepare_sft_oci_guardian_v1.py": (
        "9b3c7ad74838029d9cd0e324ebbdf15658d22ca1ed59416ad776496ccea7e9ff"
    ),
    "scripts/run_sft_oci_guardian_fixture_v1.py": (
        "38d64b4159bcc418ff799c14fae6cde00ec98b35b58cbfeb58fb14b2e83d8f76"
    ),
    "src/lumi/posttraining_sft_oci_admission_v1.py": (
        "a2f26771b22fedfe8a003bc3104c40f46477465b7f0ab2c50bee7ad3abb945f0"
    ),
    "scripts/run_sft_oci_inspect_diagnostic_v1.py": (
        "23df871fe7fb57719e44ebe4ac2185dfdbd178bd7ca43927cedae0adc62eb954"
    ),
    "src/lumi/posttraining_sft_oci_inspect_diagnostic_v1.py": (
        "54fb165e864db5a69d41875e014e933dcba443f8ce39ea3d8a0532656848067b"
    ),
}
EXPECTED_TESTS = (
    "tests/test_sft_oci_rootfs_v1.py",
    "tests/test_sft_oci_mount_v1.py",
    "tests/test_sft_oci_landlock_v1.py",
    "tests/test_sft_oci_guarded_worker_v1.py",
    "tests/test_sft_oci_admission_v2.py",
    "tests/test_sft_oci_guardian_fixture_v2.py",
    "tests/test_sft_oci_guardian_acquisition_v2.py",
    "tests/test_sft_oci_guardian_workflow_v2.py",
)


def function(tree, name):
    return next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def source(name):
    return ast.unparse(function(PREPARE, name))


def literal(name, tree=PREPARE):
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    )
    return ast.literal_eval(node.value)


class WorkflowTests(unittest.TestCase):
    def test_one_private_direct_successor_first_attempt_only(self):
        guard = next(line.strip() for line in TEXT.splitlines() if line.strip().startswith("if:"))
        self.assertEqual(
            guard,
            "if: github.event.repository.private == true"
            " && github.run_attempt == 1"
            f" && github.event.before == '{PARENT}'"
            " && github.event.after == github.sha",
        )
        self.assertIn("branches:\n      - codex/lumi-posttraining-product-v0.1", TEXT)
        for forbidden in ("workflow_dispatch", "pull_request", "schedule:", "workflow_run:"):
            self.assertNotIn(forbidden, TEXT)

    def test_same_parent_inside_runtime_preflight(self):
        self.assertEqual(literal("APPROVED_PARENT"), PARENT)
        guarded = source("preflight")
        for value in (
            "HEAD^",
            "APPROVED_PARENT",
            "event['before']",
            "event['after']",
            "event['repository']['private']",
            "GITHUB_RUN_ATTEMPT",
            "LUMI_OCI_GUARDIAN_V2_CI",
        ):
            self.assertIn(value, guarded)

    def test_readonly_permissions_no_persisted_credentials_no_cancel(self):
        self.assertIn("permissions:\n  contents: read\n  actions: read", TEXT)
        for value in ("persist-credentials: false", "fetch-depth: 2", "cancel-in-progress: false"):
            self.assertIn(value, TEXT)
        self.assertNotIn("write-all", TEXT)

    def test_exact_eight_source_suites_precede_acquisition(self):
        self.assertLess(TEXT.index("python -S -B -m unittest"), TEXT.index("id: software"))
        suites = tuple(
            re.findall(r"tests/test_sft_oci_[a-z0-9_]*\.py", TEXT.split("run: >-", 1)[1])
        )
        self.assertEqual(suites, EXPECTED_TESTS)

    def test_no_old_entry_remote_model_gpu_rebuild_or_retry_command(self):
        for forbidden in (
            "run_sft_oci_guardian_fixture_v1",
            "run_sft_oci_inspect_diagnostic_v1",
            "run_sft_oci_guarded_worker_v1.py",
            "run_sft_worker_run",
            "docker start",
            "docker run",
            "docker build",
            "docker pull",
            "pip install",
            "aws ",
            "s3://",
            "ssh ",
            "lambda",
            "--rerun",
            "--diagnostic-only",
        ):
            self.assertNotIn(forbidden, TEXT)
        self.assertIn('CUDA_VISIBLE_DEVICES: ""', TEXT)

    def test_one_clean_root_v2_entry_with_distro_python(self):
        expected = "/usr/bin/python3 -I -S -B scripts/run_sft_oci_guardian_fixture_v2.py"
        self.assertEqual(TEXT.count(expected), 1)
        self.assertIn("sudo -n /usr/bin/env -i LUMI_OCI_GUARDIAN_V2_CI=1", TEXT)
        native = TEXT.split("- name: One v2 composition", 1)[1].split("- name:", 1)[0]
        for sensitive in ("GH_TOKEN", "github.token"):
            self.assertNotIn(sensitive, native)
        self.assertIn(' --synthetic-only "$LUMI_IMAGE_ARCHIVE"', native)
        self.assertIn("aggregate-native.json 2>/dev/null", native)

    def test_twenty_eight_minute_job_and_bounded_phases(self):
        self.assertIn("timeout-minutes: 28", TEXT)
        self.assertIn("--kill-after=15s 6m", TEXT)
        self.assertIn("--kill-after=30s 17m", TEXT)
        self.assertEqual(literal("MAXIMUM_SECONDS"), 300)
        self.assertEqual(literal("MAXIMUM_METADATA_BYTES"), 32768)
        self.assertEqual(literal("MAXIMUM_DIAGNOSTIC_ZIP_BYTES"), 65536)

    def test_exact_input_retirement_even_after_native_failure(self):
        self.assertIn("if: always() && steps.software.outcome == 'success'", TEXT)
        self.assertIn('"$LUMI_IMAGE_IDENTITY" "$LUMI_PARENT_IDENTITY"', TEXT)
        self.assertIn("prepare_sft_oci_guardian_v2.py --retire-input", TEXT)
        self.assertNotIn("rm -", TEXT)
        self.assertIn("/tmp/lumi-oci-guardian-v2-input-", source("retire_input"))

    def test_only_aggregate_json_uploaded(self):
        self.assertEqual(TEXT.count("uses: actions/upload-artifact@v4"), 1)
        self.assertIn("path: artifacts/lumi-sft-oci-guardian-acceptance-v0.2/aggregate*.json", TEXT)
        self.assertIn("retention-days: 30", TEXT)
        upload = TEXT.split("uses: actions/upload-artifact@v4", 1)[1]
        for suffix in (".tar", ".zip", ".log", "inspect.json"):
            self.assertNotIn(suffix, upload)

    def test_fresh_input_and_output_no_old_artifact_reuse(self):
        self.assertEqual(literal("OUTPUT"), "artifacts/lumi-sft-oci-guardian-acceptance-v0.2")
        self.assertIn("prefix='lumi-oci-guardian-v2-input-'", source("acquire"))
        self.assertIn("not output.exists()", source("preflight"))

    def test_exact_seven_gets_split_four_and_three_without_redirect_replay(self):
        acquired, diagnostic = source("acquire"), source("verify_prior_diagnostic")
        self.assertEqual(acquired.count("opener.open("), 4)
        self.assertEqual(diagnostic.count("opener.open("), 3)
        for value in ("ProxyHandler({})", "NoRedirect()", "headers.clear()", "token = None"):
            self.assertIn(value, acquired)
        for code in (acquired, diagnostic):
            last = code.split("urllib.request.Request(location", 1)[1].split("as response")[0]
            self.assertNotIn("Authorization", last)
            self.assertNotIn("headers=headers", last)

    def test_prior_success_and_exact_evidence_before_large_software_request(self):
        self.assertEqual(literal("PRIOR_DIAGNOSTIC_RUN"), 34363848352)
        self.assertEqual(literal("PRIOR_DIAGNOSTIC_JOB"), 102507402716)
        acquired = source("acquire")
        self.assertLess(
            acquired.index("prior_diagnostic_success(json.loads(raw))"),
            acquired.index("verify_prior_diagnostic(opener, headers, deadline)"),
        )
        self.assertLess(
            acquired.index("verify_prior_diagnostic(opener, headers, deadline)"),
            acquired.index("metadata(json.loads(raw))"),
        )
        self.assertIn("/attempts/1/jobs?per_page=2", acquired)

    def test_frozen_transport_functions_and_preparer_unchanged(self):
        self.assertEqual(
            hashlib.sha256(OLD_PREPARE.read_bytes()).hexdigest(), FROZEN_PREPARER_SHA256
        )
        self.assertEqual(literal("FROZEN_PREPARER_SHA256"), FROZEN_PREPARER_SHA256)
        for name in (
            "need",
            "canonical",
            "identity",
            "request_timeout",
            "metadata",
            "signed_url",
            "stream",
            "git",
        ):
            with self.subTest(function=name):
                self.assertEqual(
                    ast.dump(function(PREPARE, name), include_attributes=False),
                    ast.dump(function(OLD_TREE, name), include_attributes=False),
                )

    def test_accepted_software_pins_unchanged(self):
        for name in (
            "ARTIFACT_ID",
            "ACCEPTED_RUN_ID",
            "ACCEPTED_COMMIT",
            "ZIP_BYTES",
            "ZIP_SHA256",
            "TAR_BYTES",
            "TAR_SHA256",
        ):
            self.assertEqual(literal(name), literal(name, OLD_TREE))

    def test_exact_prior_diagnostic_artifact_and_all_three_member_pins(self):
        self.assertEqual(literal("DIAGNOSTIC_ARTIFACT_ID"), 10109060039)
        self.assertEqual(literal("DIAGNOSTIC_ZIP_BYTES"), 1531)
        self.assertEqual(
            literal("DIAGNOSTIC_ZIP_SHA256"),
            "64b9f11faa76f33b059f5e08a644664c3c1e6d1ea8f2732cbaff70fd592ebe42",
        )
        self.assertEqual(
            literal("DIAGNOSTIC_MEMBERS"),
            {
                "aggregate-acquisition.json": (
                    766,
                    "9163602b770b26ab09f332c6149efd3edf118ae9356a7c3a684426870e37af39",
                ),
                "aggregate-diagnostic.json": (
                    1540,
                    "e828ba303f517d217e9afba8cea389eec591523704dc9919e9a8e4104400c732",
                ),
                "aggregate-input-cleanup.json": (
                    211,
                    "c035fedb481f6eb23e2b69741f8b54df9d4fe5f980a13c21e3edcfff7356e52f",
                ),
            },
        )

    def test_diagnostic_hash_before_zip_and_member_hash_before_semantics(self):
        code = source("diagnostic_archive")
        self.assertLess(code.index("hashlib.sha256(raw)"), code.index("zipfile.ZipFile"))
        self.assertLess(
            code.index("stream(source, target, size, digest, deadline)"),
            code.index("diagnostic_report(body)"),
        )
        self.assertNotIn("extract", code)
        report = source("diagnostic_report")
        self.assertLess(report.index("hashlib.sha256(raw)"), report.index("json.loads(raw)"))

    def test_old_code_workflow_bytes_unchanged(self):
        for relative, digest in FROZEN_HISTORICAL.items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest)

    def test_old_guardian_wildcard_may_trigger_but_job_parent_guard_is_false(self):
        old = (ROOT / ".github/workflows/sft-oci-guardian-acceptance.yml").read_text()
        paths = re.findall(r"^      - (.+)$", old.split("    paths:\n", 1)[1], re.M)
        self.assertTrue(any(fnmatch.fnmatchcase(EXPECTED_TESTS[-1], pattern) for pattern in paths))
        prior = re.search(r"github.event.before == '([a-f0-9]{40})'", old)
        self.assertIsNotNone(prior)
        self.assertNotEqual(prior[1], PARENT)
        self.assertIn("github.run_attempt == 1", old)

    def test_inspect_workflow_does_not_match_v2_paths(self):
        old = (ROOT / ".github/workflows/sft-oci-inspect-diagnostic.yml").read_text()
        paths = re.findall(r"^      - (.+)$", old.split("    paths:\n", 1)[1], re.M)
        changed = (
            ".github/workflows/sft-oci-guardian-acceptance-v2.yml",
            "scripts/prepare_sft_oci_guardian_v2.py",
            "scripts/run_sft_oci_guardian_fixture_v2.py",
            "src/lumi/posttraining_sft_oci_admission_v2.py",
            *EXPECTED_TESTS[4:],
        )
        self.assertFalse(
            any(fnmatch.fnmatchcase(path, pattern) for path in changed for pattern in paths)
        )

    def test_only_exact_v2_paths_trigger_new_workflow(self):
        paths = re.findall(
            r"^      - (.+)$", TEXT.split("    paths:\n", 1)[1].split("\npermissions:", 1)[0], re.M
        )
        self.assertEqual(len(paths), 8)
        self.assertTrue(all("*" not in path for path in paths))
        self.assertTrue(all(path.endswith("v2.py") or path.endswith("-v2.yml") for path in paths))


if __name__ == "__main__":
    unittest.main()
