"""Structural one-shot CI/source boundary checks; no workflow/native execution."""

from __future__ import annotations

import ast
import hashlib
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/sft-oci-guardian-acceptance.yml"
TEXT = WORKFLOW.read_text()
PREPARE = ast.parse((ROOT / "scripts/prepare_sft_oci_guardian_v1.py").read_text())
FROZEN = {
    "scripts/run_sft_cgroup_fixture_v1.py":
        "f91ed0c353a20c309ff3e1bfaab8dbe990b28c36626fdeeb0ac5131e0cf75fad",
    "scripts/run_sft_worker_run_torch_fixture_v1.py":
        "53c50c92965ed1cef0c9624c57367c31ed7ebfb9d7cfb34257188ad179d333d0",
    "scripts/run_sft_oci_worker_probe_v2.py":
        "b45b04fc9cb93dd2301df0444f2db95666ba4f7a32e8f4d8e28c5b4077314450",
    "src/lumi/posttraining_sft_oci_admission_v1.py":
        "a2f26771b22fedfe8a003bc3104c40f46477465b7f0ab2c50bee7ad3abb945f0",
    ".github/workflows/sft-cpu-lookup-acceptance.yml":
        "5e6797dc94817220fc872c11295ac82ae2ec40668ed24d547a0adf4f993c32fa",
    ".github/workflows/sft-oci-release-candidate.yml":
        "48af59107d4e17c898bb3c2508fad0c1a37cb2bdf152ce564252c4af93767b00",
    ".github/workflows/sft-cpu-origin-diagnostic.yml":
        "b8a2f6d838138bc16f3666cac4f4749da49d39ae63662b28ce9f23fbc7bbf76f",
}


def source(name):
    node = next(node for node in PREPARE.body
                if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.unparse(node)


class WorkflowTests(unittest.TestCase):
    def test_unique_direct_successor_only_and_no_retry(self):
        guard = next(line.strip() for line in TEXT.splitlines() if line.strip().startswith("if:"))
        self.assertEqual(guard, "if: github.event.repository.private == true"
                         " && github.run_attempt == 1"
                         " && github.event.before == 'a0296d4dd80463023a154b5eb51bd4384158f513'"
                         " && github.event.after == github.sha")
        self.assertNotIn("workflow_dispatch", TEXT)
        self.assertNotIn("pull_request", TEXT)
        self.assertNotIn("schedule:", TEXT)

    def test_same_parent_and_commit_guard_inside_acquisition(self):
        literals = {node.targets[0].id: ast.literal_eval(node.value)
                    for node in PREPARE.body if isinstance(node, ast.Assign)
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Constant)}
        self.assertIn(literals["APPROVED_PARENT"], TEXT)
        guarded = source("preflight")
        for required in ("HEAD^", "APPROVED_PARENT", "event['before']", "event['after']",
                         "event['repository']['private']", "GITHUB_RUN_ATTEMPT"):
            self.assertIn(required, guarded)

    def test_read_only_permissions_and_no_cancel_existing_job(self):
        self.assertIn("permissions:\n  contents: read\n  actions: read", TEXT)
        self.assertIn("cancel-in-progress: false", TEXT)
        self.assertIn("persist-credentials: false", TEXT)
        self.assertNotIn("write-all", TEXT)

    def test_source_tests_precede_artifact_acquisition(self):
        self.assertLess(TEXT.index("python -S -B -m unittest"), TEXT.index("id: software"))
        selected = re.findall(r"tests/test_sft_oci_[a-z0-9_]+\.py", TEXT)
        self.assertEqual(len(set(selected)), 8)
        self.assertTrue(all((ROOT / path).is_file() for path in selected))

    def test_no_image_rebuild_container_start_or_remote_model_access(self):
        for forbidden in ("docker build", "docker pull", "docker start", "pip install",
                          "aws ", "s3://", "ssh ", "lambda", "workflow_dispatch", "--rerun"):
            self.assertNotIn(forbidden, TEXT)
        self.assertIn("--acquire-accepted-software", TEXT)
        self.assertIn("--synthetic-only", TEXT)

    def test_native_has_clean_environment_and_distro_stdlib_root_controller(self):
        self.assertIn("sudo -n /usr/bin/env -i LUMI_OCI_GUARDIAN_CI=1", TEXT)
        self.assertIn("/usr/bin/python3 -I -S -B scripts/run_sft_oci_guardian_fixture_v1.py", TEXT)
        native = TEXT.split("- name: One native composition", 1)[1].split("- name:", 1)[0]
        self.assertNotIn("GH_TOKEN", native)
        self.assertNotIn("github.token", native)
        self.assertIn("2>/dev/null", native)

    def test_bounded_jobs_and_exact_consumed_input_retirement(self):
        self.assertIn("timeout-minutes: 28", TEXT)
        self.assertIn("--kill-after=15s 6m", TEXT)
        self.assertIn("--kill-after=30s 17m", TEXT)
        self.assertIn("if: always() && steps.software.outcome == 'success'", TEXT)
        self.assertIn('"$LUMI_IMAGE_IDENTITY" "$LUMI_PARENT_IDENTITY"', TEXT)
        self.assertNotIn("rm -", TEXT)

    def test_small_private_evidence_only_never_reupload_large_image(self):
        self.assertIn("path: artifacts/lumi-sft-oci-guardian-acceptance-v0.1/*.json", TEXT)
        self.assertEqual(TEXT.count("uses: actions/upload-artifact@v4"), 1)
        self.assertIn("retention-days: 30", TEXT)
        upload = TEXT.split("uses: actions/upload-artifact@v4", 1)[1]
        self.assertNotIn(".tar", upload)
        self.assertNotIn(".zip", upload)
        self.assertNotIn(".log", upload)

    def test_frozen_accepted_sources_and_historical_guards_unchanged(self):
        for relative, expected in FROZEN.items():
            with self.subTest(relative=relative):
                self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
                                 expected)

    def test_download_no_redirect_replay_with_authorization(self):
        acquire = source("acquire")
        self.assertIn("ProxyHandler({})", acquire)
        self.assertIn("NoRedirect()", acquire)
        self.assertEqual(acquire.count("opener.open("), 3)
        self.assertIn("headers.clear()", acquire)
        self.assertIn("token = None", acquire)
        final_request = (acquire.split("urllib.request.Request(location", 1)[1]
                         .split("as response")[0])
        self.assertNotIn("Authorization", final_request)
        self.assertNotIn("headers=headers", final_request)

    def test_signed_url_never_written_to_evidence_or_stdout(self):
        acquire = source("acquire")
        result = acquire.split("result =", 1)[1]
        self.assertNotIn("'location'", result)
        self.assertNotIn("'token'", result)
        self.assertNotIn("'headers'", result)
        self.assertNotIn("print(", acquire)

    def test_zip_verified_before_parsing_and_exact_single_software_member(self):
        acquire = source("acquire")
        self.assertLess(acquire.index("stream(response, target, ZIP_BYTES, ZIP_SHA256"),
                        acquire.index("zipfile.ZipFile"))
        self.assertIn("len(entries) == 1", acquire)
        self.assertIn("entry.filename == 'cpu-release-image.tar'", acquire)
        self.assertIn("stream(source, target, TAR_BYTES, TAR_SHA256", acquire)
        self.assertNotIn("extractall", acquire)


if __name__ == "__main__":
    unittest.main()
