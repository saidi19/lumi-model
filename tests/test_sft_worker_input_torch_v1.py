"""Linux-only real-loader fixture; collection imports no ML libraries."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_sft_worker_input_torch_fixture_v1.py"


@unittest.skipUnless(sys.platform == "linux", "real tiny input fixture is Linux CI only")
class WorkerInputTorchTests(unittest.TestCase):
    def test_two_rank_prefix_loads_share_parent_metadata_binary_index_accounting(self):
        environment = {key: os.environ[key]
                       for key in ("PATH", "LD_LIBRARY_PATH", "SYSTEMROOT")
                       if key in os.environ}
        environment.update({
            "LC_ALL": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
            "AWS_EC2_METADATA_DISABLED": "true",
            "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "LUMI_WORKER_INPUT_TORCH_CI_FIXTURE": "1",
        })
        process = subprocess.Popen(
            [sys.executable, "-I", "-B", str(SCRIPT), "--synthetic-only"],
            cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        try:
            stdout, _ = process.communicate(timeout=150)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate(timeout=5)
            self.fail("worker_input_fixture_deadline")
        self.assertLessEqual(len(stdout), 4096)
        try:
            report = json.loads(stdout)
        except (ValueError, UnicodeError):
            self.fail("worker_input_fixture_missing_structural_report")
        self.assertEqual(report.get("status"), "passed", report)
        self.assertEqual(process.returncode, 0)
        self.assertEqual(report["schema_version"], "lumi.sft-worker-input-fixture.v0.1")
        for key in (
            "two_process_shared_controller_verified", "genuine_encoded_dataset_verified",
            "frozen_loader_result_equal", "frozen_loader_globals_preserved",
            "every_pread_precharged", "manifest_metadata_category_verified",
            "selected_prefix_only_read", "no_index_readahead", "getitem_requires_no_disk_read",
            "all_epoch_acks_confirmed", "planned_interrupt_verified", "owned_processes_reaped",
            "exact_cpu_tensors_verified",
        ):
            self.assertIs(report[key], True, key)
        for key in (
            "real_model_or_data_access", "cuda_used", "external_network_authorized",
            "full_content_hashes_verified", "production_eligible", "real_input_execution_permitted",
            "os_resource_enforcement_verified", "reference_io_in_shared_ledger",
            "fixture_generation_in_shared_ledger", "total_fixture_cost_metered",
        ):
            self.assertIs(report[key], False, key)
        self.assertEqual(report["controlled_loads"], 2)
        self.assertEqual(report["direct_reference_loads"], 2)
        self.assertEqual(report["records_loaded"], 2)
        self.assertEqual(report["encoded_index_rows"], 2)
        self.assertEqual(report["binary_bytes_read"], 48)
        self.assertEqual(report["file_opens"], 8)
        self.assertGreater(report["metadata_bytes_read"], 0)
        self.assertEqual(report["index_bytes_read"], 316)
        self.assertEqual(report["ledger_requests"], 324)
        self.assertGreater(report["parent_wire_bytes"], 0)
        self.assertLessEqual(report["parent_wire_bytes"], 1_048_576)
        self.assertEqual(report["global_optimizer_steps_reserved"], 0)
        self.assertEqual(report["global_optimizer_steps_completed"], 0)
        self.assertEqual(report["invocations_completed"], 0)
        self.assertEqual(report["invocations_started"], 1)


if __name__ == "__main__":
    unittest.main()
