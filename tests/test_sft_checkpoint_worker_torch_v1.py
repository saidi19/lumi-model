"""Mandatory Linux CI-only generated Torch/DCP checkpoint integration.

Collection imports only the standard library. Never execute this test on the
user's Mac: the subprocess is Linux-only and owns only freshly generated tiny
fixtures. This does not exercise real model/data access or authorize training.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_sft_checkpoint_worker_torch_fixture_v1.py"


@unittest.skipUnless(sys.platform == "linux", "real tiny Torch fixture is Linux CI only")
class CheckpointWorkerTorchTests(unittest.TestCase):
    def test_real_tiny_checkpoint_resume_and_fail_closed_paths(self):
        # No inherited cloud credentials, HOME, proxy settings or Python startup
        # overrides. The executable already identifies CI's installed environment.
        environment = {
            key: os.environ[key]
            for key in ("PATH", "LD_LIBRARY_PATH", "SYSTEMROOT")
            if key in os.environ
        }
        environment.update({
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "AWS_EC2_METADATA_DISABLED": "true",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "LUMI_CHECKPOINT_TORCH_CI_FIXTURE": "1",
        })
        result = subprocess.run(
            [sys.executable, "-I", "-B", str(SCRIPT), "--synthetic-only"],
            cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
            capture_output=True, timeout=90, check=False,
        )
        # Do not echo third-party warnings, tracebacks, environment or paths.
        self.assertLessEqual(len(result.stdout), 8192)
        try:
            report = json.loads(result.stdout)
        except (ValueError, UnicodeError):
            self.fail("checkpoint_worker_fixture_missing_structural_report")
        self.assertEqual(report.get("status"), "passed", report)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(report["schema_version"], "lumi.sft-checkpoint-worker-fixture.v0.1")
        self.assertEqual(report["scenarios_passed"], 5)
        for key in (
            "real_torch_dcp_used", "model_resume_equal", "optimizer_resume_equal",
            "rng_resume_equal", "repeated_restore_charged", "wrong_hash_rejected",
            "read_cap_rejected", "write_cap_rejected", "existing_checkpoint_preserved",
            "private_modes_verified", "borrowed_parent_fd_preserved",
            "explicit_dcp_storage_verified", "serializer_filelikes_verified",
            "stock_filesystem_io_blocked",
        ):
            self.assertIs(report[key], True, key)
        for key in (
            "real_model_or_data_access", "cuda_used", "network_authorized",
            "multiprocess_shared_controller_verified", "os_resource_enforcement_verified",
            "production_eligible", "real_sft_execution_permitted",
        ):
            self.assertIs(report[key], False, key)
        self.assertGreater(report["checkpoint_read_bytes"], 0)
        self.assertGreater(report["private_write_bytes"], 0)
        self.assertGreater(report["torch_save_filelike_calls"], 0)
        self.assertGreater(report["torch_load_filelike_calls"], 0)
        self.assertGreater(report["dcp_explicit_save_calls"], 0)
        self.assertGreater(report["dcp_explicit_load_calls"], 0)


if __name__ == "__main__":
    unittest.main()
