"""Exact fresh workflow contracts; source reads only, no native execution."""

from __future__ import annotations

import ast
import hashlib
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/sft-import-qualification-v1.yml"
PARENT = "ecc545fc423650bf1089311915290523c10b3172"
HARNESS_SHA = "918267c9d051991f2b9ae5abd150ca33fdc53c3d7d4e293e7b7ffcf56ea27753"
SUPERVISOR_SHA = "3525eb5bf60fd0bac35480dd845a1cb11c6fc57303c6c3cb03b5c56576cfc1d6"
FROZEN = {
    "scripts/run_checkpoint_guard_qualification_v1.py": HARNESS_SHA,
    "scripts/supervise_checkpoint_guard_qualification_v1.py": SUPERVISOR_SHA,
    "configs/lumi-4b-sft-import-qualification-sources-v0.1.json":
        "f761ef5179b2e226347959f13ba7f4f385e70cd977e89c50a8a2498dff5a819f",
    "scripts/run_sft_oci_guardian_fixture_v2.py":
        "79dcbfec678f43a39b31bb2dd2b4c9535fc162447cc591a781b72f93f079ce58",
}
SUITES = (
    "tests/test_sft_checkpoint_guard_qualification_v1.py",
    "tests/test_sft_checkpoint_guard_harness_v1.py",
    "tests/test_sft_checkpoint_guard_supervisor_v1.py",
    "tests/test_sft_import_qualification_acquisition_v1.py",
    "tests/test_sft_import_qualification_verification_v1.py",
    "tests/test_sft_import_qualification_workflow_v1.py",
)


def section(text, label):
    return text.split("- name: " + label, 1)[1].split("\n      - ", 1)[0]


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding="utf-8")

    def test_fresh_name_and_only_exact_branch_push_path(self):
        self.assertTrue(self.text.startswith("name: SFT tiny importer qualification v1\n"))
        self.assertIn("branches:\n      - codex/lumi-posttraining-product-v0.1", self.text)
        trigger = self.text.split("\npermissions:", 1)[0]
        self.assertEqual(re.findall(r"      - (\.github/[^\n]+)", trigger),
                         [".github/workflows/sft-import-qualification-v1.yml"])
        for value in ("workflow_dispatch", "pull_request:", "schedule:", "workflow_run:"):
            self.assertNotIn(value, trigger)

    def test_private_exact_parent_first_attempt_guard(self):
        self.assertIn("if: github.event.repository.private == true && github.run_attempt == 1"
                      f" && github.event.before == '{PARENT}'"
                      " && github.event.after == github.sha", self.text)

    def test_permissions_checkout_and_concurrency(self):
        for value in ("permissions:\n  contents: read\n  actions: read",
                      "persist-credentials: false", "fetch-depth: 2",
                      "cancel-in-progress: false"):
            self.assertIn(value, self.text)
        self.assertNotIn("write-all", self.text)

    def test_distro_python_no_setup_install_or_gpu(self):
        self.assertIn("runs-on: ubuntu-24.04", self.text)
        for value in ("setup-python", "pip install", "aws ", "s3://", "ssh ",
                      "docker build", "docker pull", "docker run", "docker start"):
            self.assertNotIn(value, self.text)
        self.assertIn('CUDA_VISIBLE_DEVICES: ""', self.text)

    def test_separate_phase_budgets_and_job_reserve(self):
        job = re.search(r"^    timeout-minutes: (\d+)$", self.text, re.M)
        steps = re.findall(r"^        timeout-minutes: (\d+)$", self.text, re.M)
        self.assertEqual(int(job[1]), 36)
        self.assertEqual(list(map(int, steps)), [2, 2, 7, 19, 1, 1, 2])
        self.assertEqual(sum(map(int, steps)), 34)

    def test_four_executable_pins_are_actual_sources(self):
        block = self.text.split("<<'LUMI_PINS'\n", 1)[1].split("          LUMI_PINS", 1)[0]
        lines = re.findall(r"^          ([a-f0-9]{64})  (scripts/[a-z0-9_]+\.py)$",
                           block, re.M)
        self.assertEqual(len(lines), 4)
        self.assertEqual({name for _sha, name in lines}, {
            "scripts/run_checkpoint_guard_qualification_v1.py",
            "scripts/supervise_checkpoint_guard_qualification_v1.py",
            "scripts/prepare_sft_import_qualification_v1.py",
            "scripts/verify_sft_import_qualification_v1.py",
        })
        for sha, name in lines:
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), sha)
        self.assertNotIn("_PENDING", self.text)

    def test_source_authentication_and_exact_pure_suites_precede_token(self):
        block = section(self.text,
                        "Bound sources and pure contracts before any software acquisition")
        self.assertLess(block.index("sha256sum --check --strict"), block.index("--check-source"))
        self.assertLess(block.index("--check-source"), block.index("-m unittest"))
        self.assertEqual(tuple(re.findall(r"tests/test_[a-z0-9_]+\.py", block)), SUITES)
        self.assertNotIn("GH_TOKEN", block)

    def test_new_acquisition_only_with_bounded_outer_timeout(self):
        block = section(self.text, "Fresh four-response accepted-software acquisition")
        self.assertIn("GH_TOKEN: ${{ github.token }}", block)
        self.assertIn("timeout --signal=TERM --kill-after=15s 6m", block)
        self.assertIn("prepare_sft_import_qualification_v1.py", block)
        self.assertIn("--acquire-accepted-software", block)
        self.assertEqual(self.text.count("--acquire-accepted-software"), 1)

    def test_native_supervisor_not_process_group_timeout(self):
        block = section(self.text, "One independent PID-supervised tiny importer qualification")
        self.assertNotIn("timeout --", block)
        self.assertNotIn("kill-after", block)
        self.assertIn("sudo -n /usr/bin/env -i", block)
        self.assertIn("PATH=/usr/bin:/bin LC_ALL=C.UTF-8 "
                      "LUMI_CHECKPOINT_GUARD_QUALIFICATION_CI=1", block)
        self.assertIn("/usr/bin/python3 -I -S -B "
                      "scripts/supervise_checkpoint_guard_qualification_v1.py", block)
        self.assertIn('--synthetic-only "$LUMI_IMAGE_ARCHIVE"', block)
        self.assertIn(HARNESS_SHA, block)
        for value in ("GH_TOKEN", "github.token", "PYTHONPATH=", "HF_HUB_OFFLINE="):
            self.assertNotIn(value, block)
        self.assertIn("umask 077", block)
        self.assertIn("aggregate-native.json 2>/dev/null", block)

    def test_exact_native_invocation_occurs_once(self):
        command = "/usr/bin/python3 -I -S -B " \
                  "scripts/supervise_checkpoint_guard_qualification_v1.py"
        self.assertEqual(self.text.count(command), 1)
        self.assertNotIn("run_checkpoint_guard_qualification_v1.py --synthetic-only", self.text)

    def test_retirement_always_after_successful_acquisition_even_native_failure(self):
        block = section(self.text, "Retire exactly the newly acquired input")
        self.assertIn("if: always() && steps.software.outcome == 'success'", block)
        self.assertIn("prepare_sft_import_qualification_v1.py --retire-input", block)
        self.assertIn('"$LUMI_IMAGE_ARCHIVE" "$LUMI_IMAGE_IDENTITY" "$LUMI_PARENT_IDENTITY"', block)
        self.assertNotIn("rm -", block)
        self.assertNotIn("steps.native.outcome", block)

    def test_structural_receipt_check_after_retirement(self):
        self.assertLess(self.text.index("- name: Retire exactly"),
                        self.text.index("- name: Exact aggregate qualification"))
        block = section(self.text, "Exact aggregate qualification and cleanup verification")
        self.assertIn("if: always() && steps.software.outcome == 'success'", block)
        self.assertIn("verify_sft_import_qualification_v1.py", block)
        self.assertIn("--verify-receipts", block)

    def test_only_four_named_aggregates_upload(self):
        block = section(self.text, "Retain only the four named bounded aggregates")
        names = re.findall(r"artifacts/lumi-sft-import-qualification-v0.1/([^\n]+)", block)
        self.assertEqual(names, ["aggregate-acquisition.json", "aggregate-native.json",
                                 "aggregate-input-cleanup.json", "aggregate-verification.json"])
        self.assertNotIn("*", block)
        self.assertNotIn(".tar", block)
        self.assertNotIn(".zip", block)
        self.assertNotIn(".log", block)
        self.assertIn("retention-days: 30", block)
        self.assertIn("if: always()", block)

    def test_new_attempt_does_not_execute_old_preparer_or_worker(self):
        for value in ("prepare_sft_oci_guardian_v2.py",
                      "run_sft_oci_guardian_fixture_v2.py",
                      "run_sft_oci_inspect_diagnostic_v1.py", "--rerun"):
            self.assertNotIn(value, self.text)

    def test_progress_bar_contract_is_set_before_tests(self):
        self.assertIn('HF_HUB_DISABLE_PROGRESS_BARS: "1"', self.text)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            self.assertIn(key + ': "1"', self.text)

    def test_frozen_runtime_sources_unchanged(self):
        for name, expected in FROZEN.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)

    def test_retained_provider_and_teardown_trust_not_fake_proof(self):
        self.assertIn("Trusted disposable hosted VM and provider actions are explicit premises",
                      self.text)
        self.assertIn("cancellation/guardian-death teardown is not independently proved",
                      self.text)
        self.assertEqual(re.findall(r"uses: ([^\n]+)", self.text),
                         ["actions/checkout@v6", "actions/upload-artifact@v4"])

    def test_new_script_entrypoints_are_main_guarded(self):
        for name in ("prepare", "verify"):
            tree = ast.parse((ROOT / f"scripts/{name}_sft_import_qualification_v1.py").read_bytes())
            guards = [node for node in tree.body if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "__name__ == '__main__'"]
            self.assertEqual(len(guards), 1)


if __name__ == "__main__":
    unittest.main()
