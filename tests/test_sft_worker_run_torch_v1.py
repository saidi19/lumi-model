"""Remote Linux CI only: three fresh CPU/DDP WorkerRun invocations, no eager ML."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_sft_worker_run_torch_fixture_v1.py"


@unittest.skipUnless(sys.platform == "linux", "full tiny WorkerRun fixture is remote Linux CI only")
class WorkerRunTorchTests(unittest.TestCase):
    def test_three_fresh_two_rank_invocations_share_ledger_and_resume_exact_state(self):
        environment = {key: os.environ[key]
                       for key in ("PATH", "LD_LIBRARY_PATH", "SYSTEMROOT")
                       if key in os.environ}
        environment.update({
            "LC_ALL": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
            "AWS_EC2_METADATA_DISABLED": "true", "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "GLOO_SOCKET_IFNAME": "lo", "TP_SOCKET_IFNAME": "lo",
            "LUMI_WORKER_RUN_TORCH_CI_FIXTURE": "1",
        })
        process = subprocess.Popen(
            [sys.executable, "-I", "-B", str(SCRIPT), "--synthetic-only"],
            cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        try:
            stdout, _ = process.communicate(timeout=210)
        except subprocess.TimeoutExpired:
            # Exactly the newly owned fixture group, including all live Gloo peers.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate(timeout=5)
            self.fail("worker_run_fixture_deadline")
        self.assertLessEqual(len(stdout), 4096)
        try:
            report = json.loads(stdout)
        except (ValueError, UnicodeError):
            self.fail("worker_run_fixture_missing_structural_report")
        self.assertEqual(report.get("status"), "passed", report)
        self.assertEqual(process.returncode, 0)
        self.assertEqual(report["schema_version"], "lumi.sft-worker-run-fixture.v0.1")
        for key in (
            "real_cpu_ddp_used", "real_torch_dcp_used", "real_worker_run_used",
            "single_parent_supervisor_verified", "all_three_invocations_completed",
            "fresh_workers_and_epochs_verified", "all_optimizer_steps_in_shared_ledger",
            "six_prefix_loads_in_shared_ledger", "checkpoint_io_in_shared_ledger",
            "validation_metadata_present", "selected_prefix_only_read",
            "epoch_ack_confirmations_verified", "completion_fences_verified",
            "io_phase_confirmations_drained", "reference_model_resume_equal",
            "reference_adam_resume_equal", "reference_rng_resume_equal",
            "reference_training_state_resume_equal", "dropout_rng_consumption_verified",
            "private_modes_verified", "owned_processes_reaped", "rss_snapshot_observed",
            "loopback_gloo_used",
        ):
            self.assertIs(report[key], True, key)
        for key in (
            "real_model_or_data_access", "cuda_used", "external_network_authorized",
            "os_resource_enforcement_verified", "production_eligible",
            "real_sft_execution_permitted",
            "full_content_hashes_verified", "fixture_generation_in_shared_ledger",
            "private_comparison_work_in_shared_ledger", "total_fixture_cost_metered",
            "safe_unpickling_or_origin_authentication_verified",
        ):
            self.assertIs(report[key], False, key)
        for key, expected in {
            "rank_workers_verified": 6, "controlled_rank_optimizer_steps": 8,
            "global_optimizer_steps_reserved": 4, "global_optimizer_steps_completed": 4,
            "logical_sample_events_reserved": 16, "invocations_started": 3,
            "invocations_completed": 3, "controlled_loads": 6, "records_loaded": 12,
            "encoded_index_rows": 12, "binary_bytes_read": 288, "index_bytes_read": 1896,
            "checkpoints_published": 4, "rank_checkpoint_restores": 2,
            "commit_fences_sent": 8, "memory_observations": 3,
        }.items():
            self.assertEqual(report[key], expected, key)
        for key, maximum in {
            "metadata_bytes_read": 65_536, "parent_wire_bytes": 96 * 1024 * 1024,
            "checkpoint_read_bytes_reserved": 96 * 1024 * 1024,
            "private_write_bytes_reserved": 32 * 1024 * 1024,
        }.items():
            self.assertGreater(report[key], 0, key)
            self.assertLessEqual(report[key], maximum, key)
        self.assertGreater(report["ledger_requests"], 1900)


if __name__ == "__main__":
    unittest.main()
