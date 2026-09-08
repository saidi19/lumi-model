"""Remote Linux cgroup acceptance only; never run even the skip wrapper locally."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_sft_cgroup_fixture_v1.py"
RUNTIME_PATTERN = r"/opt/hostedtoolcache/Python/3\.12\.[0-9]{1,3}/x64/bin/python3\.12"


@unittest.skipUnless(sys.platform == "linux", "privileged cgroup fixture is remote Linux only")
class CgroupLinuxTests(unittest.TestCase):
    def test_resource_limits_parent_death_and_pinned_worker_run_inside_owned_cgroups(self):
        self.assertEqual(os.uname().machine, "x86_64")
        self.assertGreater(os.getuid(), 0)
        self.assertGreater(os.getgid(), 0)
        runtime = Path(sys.executable).resolve(strict=True)
        self.assertIsNotNone(re.fullmatch(RUNTIME_PATTERN, str(runtime)))
        info = runtime.lstat()
        self.assertTrue(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode))
        self.assertGreater(info.st_size, 0)
        self.assertLessEqual(info.st_size, 64 * 1024 * 1024)
        hasher = hashlib.sha256()
        with runtime.open("rb") as source:
            first = source.read(65_536)
            self.assertTrue(first.startswith(b"\x7fELF"))
            hasher.update(first)
            for _ in range(1024):
                chunk = source.read(65_536)
                if not chunk:
                    break
                hasher.update(chunk)
            else:
                self.fail("runtime_binding_oversized")
        bootstrap = json.dumps({"runtime_path": str(runtime), "runtime_sha256": hasher.hexdigest()},
                               sort_keys=True, separators=(",", ":")).encode("ascii")
        process = subprocess.Popen(
            ["/usr/bin/sudo", "-n", "/usr/bin/python3", "-I", "-S", "-B", str(SCRIPT),
             "--synthetic-only"], cwd=ROOT,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"},
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            stdout, _ = process.communicate(input=bootstrap, timeout=300)
        except subprocess.TimeoutExpired:
            # The wrapper cannot kill/remove root-owned cgroups. Independent root
            # guardians carry the actual deadline/parent-death cleanup authority.
            # Do not invent a privileged killpg/rmdir success from this timeout.
            self.fail("cgroup_fixture_deadline_cleanup_unconfirmed")
        self.assertLessEqual(len(stdout), 4096)
        try:
            report = json.loads(stdout)
        except (ValueError, UnicodeError):
            self.fail("cgroup_fixture_missing_structural_report")
        self.assertEqual(process.returncode, 0, report)
        self.assertEqual(report.get("status"), "passed", report)
        self.assertEqual(report["schema_version"], "lumi.sft-cgroup-linux-fixture.v0.1")
        for key in (
            "normal_completion_verified", "memory_oom_enforced", "pids_limit_enforced",
            "cpu_quota_throttling_verified", "deadline_cleanup_verified",
            "actual_parent_loss_cleanup_verified", "double_fork_setsid_descendants_reaped",
            "privileges_dropped_before_payload", "cgroup_pidfd_authority_closed_before_payload",
            "guardian_outside_workload_cgroup", "no_root_ml_import",
            "pinned_worker_run_fixture_completed",
        ):
            self.assertIs(report[key], True, key)
        for key in (
            "real_model_or_data_access", "cuda_used", "production_eligible",
            "real_sft_execution_permitted", "complete_os_sandbox_verified",
            "network_isolation_verified", "filesystem_isolation_verified",
            "real_host_admission_verified", "runtime_origin_authenticated",
            "root_parent_controllers_modified", "packages_installed",
        ):
            self.assertIs(report[key], False, key)
        for key, expected in {
            "cases_verified": 7, "owned_cgroups_empty_and_removed": 7,
            "worker_run_invocations_completed": 3, "worker_run_rank_workers_verified": 6,
            "worker_run_global_optimizer_steps_completed": 4,
            "worker_run_rank_optimizer_steps_completed": 8,
        }.items():
            self.assertEqual(report[key], expected, key)
        self.assertGreaterEqual(report["workload_processes_reaped"], 8)


if __name__ == "__main__":
    unittest.main()
