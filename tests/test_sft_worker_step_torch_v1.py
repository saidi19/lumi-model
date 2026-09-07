"""Linux-only real two-rank CPU fixture; collection never imports Torch."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_sft_worker_step_torch_fixture_v1.py"


@unittest.skipUnless(sys.platform == "linux", "real tiny DDP fixture is Linux CI only")
class WorkerStepTorchTests(unittest.TestCase):
    def test_two_rank_weighted_steps_match_independent_global_batch_reference(self):
        environment = {key: os.environ[key]
                       for key in ("PATH", "LD_LIBRARY_PATH", "SYSTEMROOT")
                       if key in os.environ}
        environment.update({
            "LC_ALL": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "AWS_EC2_METADATA_DISABLED": "true", "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "GLOO_SOCKET_IFNAME": "lo", "TP_SOCKET_IFNAME": "lo",
            "LUMI_WORKER_STEP_TORCH_CI_FIXTURE": "1",
        })
        process = subprocess.Popen(
            [sys.executable, "-I", "-B", str(SCRIPT), "--synthetic-only"],
            cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
        )
        try:
            stdout, _ = process.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            # Kill only this newly owned process group, including its DDP peers.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate(timeout=5)
            self.fail("worker_step_fixture_deadline")
        self.assertLessEqual(len(stdout), 4096)
        try:
            report = json.loads(stdout)
        except (ValueError, UnicodeError):
            self.fail("worker_step_fixture_missing_structural_report")
        self.assertEqual(report.get("status"), "passed", report)
        self.assertEqual(process.returncode, 0)
        self.assertEqual(report["schema_version"], "lumi.sft-worker-step-fixture.v0.1")
        self.assertEqual(report["rank_workers_verified"], 2)
        self.assertEqual(report["global_optimizer_steps_reserved"], 2)
        self.assertEqual(report["global_optimizer_steps_completed"], 2)
        self.assertEqual(report["logical_sample_events_reserved"], 8)
        self.assertEqual(report["invocations_completed"], 1)
        self.assertEqual(report["commit_fences_sent"], 4)
        self.assertEqual(report["controlled_rank_optimizer_steps"], 4)
        self.assertEqual(report["reference_optimizer_steps"], 4)
        for key in (
            "real_cpu_ddp_used", "global_batch_model_equal_at_each_step",
            "global_batch_adam_equal_at_each_step", "weighted_loss_equal",
            "gradient_norm_equal", "aggregate_token_counts_equal",
            "imbalanced_supervision_verified", "both_gradient_sync_modes_verified",
            "forward_follows_global_reservation", "single_parent_supervisor_verified",
            "epoch_ack_confirmations_verified", "completion_fences_verified",
            "owned_processes_reaped", "rss_snapshot_observed", "loopback_gloo_used",
        ):
            self.assertIs(report[key], True, key)
        for key in (
            "real_model_or_data_access", "cuda_used", "external_network_authorized",
            "os_resource_enforcement_verified", "all_three_invocations_completed",
            "production_eligible", "real_sft_execution_permitted",
            "generated_reference_work_in_shared_ledger", "total_fixture_cost_metered",
        ):
            self.assertIs(report[key], False, key)


if __name__ == "__main__":
    unittest.main()
