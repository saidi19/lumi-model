"""Stdlib source/AST workflow contracts only; never launch native or network work."""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/sft-oci-inspect-diagnostic.yml"
TEXT = WORKFLOW.read_text()
PREPARE = ast.parse((ROOT / "scripts/prepare_sft_oci_inspect_diagnostic_v1.py").read_text())
OLD_PREPARE = ROOT / "scripts/prepare_sft_oci_guardian_v1.py"
OLD_TREE = ast.parse(OLD_PREPARE.read_text())
FROZEN_PREPARER_SHA256 = "9b3c7ad74838029d9cd0e324ebbdf15658d22ca1ed59416ad776496ccea7e9ff"
FROZEN_HISTORICAL = {
    ".github/workflows/sft-oci-guardian-acceptance.yml":
        "e64dffedb7c7c55b9fcc787921ff157efce50b654a28d824ba215d26c42a458d",
    "scripts/run_sft_oci_guardian_fixture_v1.py":
        "38d64b4159bcc418ff799c14fae6cde00ec98b35b58cbfeb58fb14b2e83d8f76",
    "src/lumi/posttraining_sft_oci_admission_v1.py":
        "a2f26771b22fedfe8a003bc3104c40f46477465b7f0ab2c50bee7ad3abb945f0",
}
EXPECTED_TESTS = (
    "tests/test_sft_oci_inspect_diagnostic_v1.py",
    "tests/test_sft_oci_inspect_diagnostic_fixture_v1.py",
    "tests/test_sft_oci_inspect_diagnostic_acquisition_v1.py",
    "tests/test_sft_oci_inspect_diagnostic_workflow_v1.py",
)


def function(tree, name):
    return next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def source(name):
    return ast.unparse(function(PREPARE, name))


def literal(name):
    node = next(node for node in PREPARE.body if isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name)
    return ast.literal_eval(node.value)


class WorkflowTests(unittest.TestCase):
    def test_one_private_direct_successor_first_attempt_only(self):
        guard = next(line.strip() for line in TEXT.splitlines() if line.strip().startswith("if:"))
        self.assertEqual(guard, "if: github.event.repository.private == true"
                         " && github.run_attempt == 1"
                         " && github.event.before == '11486aa570657e8c71589ced46054459972d1493'"
                         " && github.event.after == github.sha")
        self.assertIn("branches:\n      - codex/lumi-posttraining-product-v0.1", TEXT)
        for forbidden in ("workflow_dispatch", "pull_request", "schedule:", "workflow_run:"):
            self.assertNotIn(forbidden, TEXT)

    def test_same_parent_inside_runtime_preflight(self):
        self.assertEqual(literal("APPROVED_PARENT"), "11486aa570657e8c71589ced46054459972d1493")
        guarded = source("preflight")
        for value in ("HEAD^", "APPROVED_PARENT", "event['before']", "event['after']",
                      "event['repository']['private']", "GITHUB_RUN_ATTEMPT",
                      "LUMI_OCI_INSPECT_DIAGNOSTIC_CI"):
            self.assertIn(value, guarded)

    def test_readonly_permissions_credential_free_checkout_and_no_cancel(self):
        self.assertIn("permissions:\n  contents: read\n  actions: read", TEXT)
        self.assertIn("persist-credentials: false", TEXT)
        self.assertIn("fetch-depth: 2", TEXT)
        self.assertIn("cancel-in-progress: false", TEXT)
        self.assertNotIn("write-all", TEXT)

    def test_exact_four_source_suites_precede_any_acquisition(self):
        self.assertLess(TEXT.index("python -S -B -m unittest"), TEXT.index("id: software"))
        suites = tuple(re.findall(r"tests/test_sft_oci_inspect_diagnostic[a-z0-9_]*\.py", TEXT))
        self.assertEqual(suites, EXPECTED_TESTS)
        self.assertEqual(len(set(suites)), 4)

    def test_no_old_guardian_payload_worker_export_or_gpu_command(self):
        for forbidden in ("run_sft_oci_guardian_fixture_v1", "run_sft_oci_guarded_worker_v1",
                          "run_sft_worker_run", "--synthetic-only", "docker export",
                          "docker start", "docker run", "docker build", "docker pull",
                          "pip install", "aws ", "s3://", "ssh ", "lambda", "--rerun"):
            self.assertNotIn(forbidden, TEXT)
        self.assertIn("--diagnostic-only", TEXT)
        self.assertIn('CUDA_VISIBLE_DEVICES: ""', TEXT)

    def test_one_clean_root_inspect_entry_with_distro_python(self):
        expected = "/usr/bin/python3 -I -S -B scripts/run_sft_oci_inspect_diagnostic_v1.py"
        self.assertEqual(TEXT.count(expected), 1)
        self.assertIn("sudo -n /usr/bin/env -i LUMI_OCI_INSPECT_DIAGNOSTIC_CI=1", TEXT)
        native = (TEXT.split("- name: One bounded OCI inspect diagnostic", 1)[1]
                  .split("- name:", 1)[0])
        self.assertNotIn("GH_TOKEN", native)
        self.assertNotIn("github.token", native)
        self.assertIn("2>/dev/null", native)
        self.assertIn("aggregate-diagnostic.json", native)

    def test_twenty_minute_job_and_bounded_acquisition_and_native(self):
        self.assertIn("timeout-minutes: 20", TEXT)
        self.assertIn("--kill-after=15s 6m", TEXT)
        self.assertIn("--kill-after=30s 8m", TEXT)
        self.assertEqual(literal("MAXIMUM_SECONDS"), 300)

    def test_exact_input_retirement_runs_even_after_diagnostic_failure(self):
        self.assertIn("if: always() && steps.software.outcome == 'success'", TEXT)
        self.assertIn('"$LUMI_IMAGE_IDENTITY" "$LUMI_PARENT_IDENTITY"', TEXT)
        self.assertIn("prepare_sft_oci_inspect_diagnostic_v1.py --retire-input", TEXT)
        self.assertNotIn("rm -", TEXT)
        retirement = source("retire_input")
        self.assertIn("/tmp/lumi-oci-inspect-input-", retirement)
        self.assertNotIn("lumi-oci-guardian-input-", retirement)

    def test_upload_only_aggregate_json_never_software_or_raw_inspect(self):
        self.assertEqual(TEXT.count("uses: actions/upload-artifact@v4"), 1)
        self.assertIn("path: artifacts/lumi-sft-oci-inspect-diagnostic-v0.1/aggregate*.json", TEXT)
        self.assertIn("retention-days: 30", TEXT)
        upload = TEXT.split("uses: actions/upload-artifact@v4", 1)[1]
        for suffix in (".tar", ".zip", ".log", "inspect.json"):
            self.assertNotIn(suffix, upload)

    def test_fresh_output_and_input_prefix_do_not_reuse_old_artifacts(self):
        self.assertEqual(literal("OUTPUT"), "artifacts/lumi-sft-oci-inspect-diagnostic-v0.1")
        acquired = source("acquire")
        self.assertIn("prefix='lumi-oci-inspect-input-'", acquired)
        self.assertIn("'aggregate-acquisition.json'", acquired)
        self.assertIn("not output.exists()", source("preflight"))

    def test_four_gets_no_redirect_replay_or_signed_request_credentials(self):
        acquired = source("acquire")
        self.assertEqual(acquired.count("opener.open("), 4)
        self.assertIn("ProxyHandler({})", acquired)
        self.assertIn("NoRedirect()", acquired)
        self.assertIn("headers.clear()", acquired)
        self.assertIn("token = None", acquired)
        last = acquired.split("urllib.request.Request(location", 1)[1].split("as response")[0]
        self.assertNotIn("Authorization", last)
        self.assertNotIn("headers=headers", last)

    def test_prior_native_failure_and_successful_cleanup_before_artifact_metadata(self):
        self.assertEqual(literal("PRIOR_NATIVE_RUN"), 34359447892)
        self.assertEqual(literal("PRIOR_NATIVE_JOB"), 102492361796)
        acquired = source("acquire")
        self.assertLess(acquired.index("prior_native_stop(json.loads(raw))"),
                        acquired.index("metadata(json.loads(raw))"))
        self.assertIn("/attempts/1/jobs?per_page=2", acquired)
        self.assertIn("'guardian_composition_retried': False", acquired)
        self.assertIn("'worker_dispatched': False", acquired)

    def test_frozen_preparer_preserved_and_transport_functions_identical(self):
        self.assertEqual(hashlib.sha256(OLD_PREPARE.read_bytes()).hexdigest(),
                         FROZEN_PREPARER_SHA256)
        self.assertEqual(literal("FROZEN_PREPARER_SHA256"), FROZEN_PREPARER_SHA256)
        for name in ("need", "canonical", "identity", "request_timeout", "metadata",
                     "signed_url", "stream", "git"):
            with self.subTest(function=name):
                self.assertEqual(ast.dump(function(PREPARE, name), include_attributes=False),
                                 ast.dump(function(OLD_TREE, name), include_attributes=False))

    def test_accepted_artifact_and_software_pins_unchanged(self):
        for name in ("ARTIFACT_ID", "ACCEPTED_RUN_ID", "ACCEPTED_COMMIT", "ZIP_BYTES",
                     "ZIP_SHA256", "TAR_BYTES", "TAR_SHA256"):
            old = next(node for node in OLD_TREE.body if isinstance(node, ast.Assign)
                       and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name)
            self.assertEqual(literal(name), ast.literal_eval(old.value))

    def test_historical_guardian_and_admission_bytes_unchanged(self):
        for relative, digest in FROZEN_HISTORICAL.items():
            with self.subTest(path=relative):
                self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest)

    def test_new_diagnostic_files_do_not_trigger_old_guardian_paths(self):
        old_text = (ROOT / ".github/workflows/sft-oci-guardian-acceptance.yml").read_text()
        old_paths = re.findall(r"^      - (.+)$", old_text.split("    paths:\n", 1)[1], re.M)
        changed = (
            ".github/workflows/sft-oci-inspect-diagnostic.yml",
            "scripts/prepare_sft_oci_inspect_diagnostic_v1.py",
            "scripts/run_sft_oci_inspect_diagnostic_v1.py",
            "src/lumi/posttraining_sft_oci_inspect_diagnostic_v1.py", *EXPECTED_TESTS,
        )
        for path in changed:
            self.assertFalse(any(fnmatch.fnmatchcase(path, pattern) for pattern in old_paths), path)

    def test_zip_and_tar_are_independently_pinned_without_archive_extractall(self):
        acquired = source("acquire")
        self.assertLess(acquired.index("stream(response, target, ZIP_BYTES, ZIP_SHA256"),
                        acquired.index("zipfile.ZipFile"))
        self.assertIn("len(entries) == 1", acquired)
        self.assertIn("entry.filename == 'cpu-release-image.tar'", acquired)
        self.assertIn("stream(source, target, TAR_BYTES, TAR_SHA256", acquired)
        self.assertNotIn("extractall", acquired)


if __name__ == "__main__":
    unittest.main()
