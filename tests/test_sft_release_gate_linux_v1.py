"""Remote Linux acceptance only; never import or run this wrapper on the Mac."""

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


@unittest.skipUnless(sys.platform == "linux", "release-gate acceptance is remote Linux only")
class ReleaseGateLinuxTests(unittest.TestCase):
    def test_verified_release_ordering_and_two_blocked_worker_rejections(self):
        self.assertEqual(os.uname().machine, "x86_64")
        self.assertGreater(os.getuid(), 0)
        self.assertGreater(os.getgid(), 0)
        runtime = Path(sys.executable).resolve(strict=True)
        self.assertIsNotNone(re.fullmatch(
            r"/opt/hostedtoolcache/Python/3\.12\.[0-9]{1,3}/x64/bin/python3\.12", str(runtime),
        ))
        info = runtime.lstat()
        self.assertTrue(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= 64 * 1024**2)
        digest = hashlib.sha256()
        with runtime.open("rb") as source:
            first = source.read(65536)
            self.assertTrue(first.startswith(b"\x7fELF"))
            digest.update(first)
            for _ in range(1024):
                raw = source.read(65536)
                if not raw:
                    break
                digest.update(raw)
            else:
                self.fail("runtime_binding_oversized")
        payload = json.dumps({"runtime_path": str(runtime), "runtime_sha256": digest.hexdigest()},
                             sort_keys=True, separators=(",", ":")).encode("ascii")
        process = subprocess.Popen(
            ["/usr/bin/sudo", "-n", "/usr/bin/python3", "-I", "-S", "-B",
             str(ROOT / "scripts/run_sft_release_gate_fixture_v1.py"), "--synthetic-only"],
            cwd=ROOT, env={"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"},
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            stdout, _ = process.communicate(input=payload, timeout=300)
        except subprocess.TimeoutExpired:
            self.fail("release_gate_fixture_deadline_cleanup_unconfirmed")
        self.assertLessEqual(len(stdout), 4096)
        try:
            report = json.loads(stdout)
        except (ValueError, UnicodeError):
            self.fail("release_gate_fixture_missing_structural_report")
        self.assertEqual(process.returncode, 0, report)
        self.assertEqual(report.get("status"), "passed", report)
        self.assertEqual(report["schema_version"], "lumi.sft-release-gate-linux-fixture.v0.1")
        for field in (
            "pinned_worker_run_fixture_completed", "preserved_ip_filesystem_checks_verified",
            "deadline_cleanup_verified", "actual_parent_loss_cleanup_verified",
            "double_fork_setsid_descendants_reaped", "privileges_dropped_before_payload",
            "cgroup_pidfd_authority_closed_before_payload", "no_root_ml_import",
            "native_release_gate_context_verified", "release_before_worker_order_verified",
            "wrong_pin_rejected", "same_size_tamper_rejected",
            "release_gate_descriptor_balance_verified",
        ):
            self.assertIs(report[field], True, field)
        for field in (
            "real_model_or_data_access", "cuda_used", "production_eligible",
            "real_sft_execution_permitted", "complete_os_sandbox_verified",
            "full_pathname_unix_isolation_verified", "runtime_origin_authenticated",
            "real_host_admission_verified", "all_ipc_isolated", "sysv_ipc_isolation_verified",
            "hostile_control_peer_isolated", "generated_release_is_running_runtime",
            "dynamic_dependency_completeness_verified", "point_in_time_snapshot_verified",
            "immutable_during_execution",
        ):
            self.assertIs(report[field], False, field)
        for field, expected in {
            "cases_verified": 4, "owned_cgroups_empty_and_removed": 4,
            "owned_mountpoints_removed": 4, "worker_run_invocations_completed": 3,
            "worker_run_rank_workers_verified": 6, "worker_run_global_optimizer_steps_completed": 4,
            "worker_run_rank_optimizer_steps_completed": 8, "seccomp_installations_verified": 4,
            "visible_same_uid_sentinel_baselines_verified": 8,
            "owned_visible_unix_sockets_removed": 8,
            "positive_release_gates_verified": 4, "rejected_release_gates_verified": 2,
            "rejected_release_calls": 0, "rejected_worker_run_invocations": 0,
            "release_gate_total_cgroups_removed": 6, "release_gate_total_trees_removed": 6,
            "rejected_blocked_workers_reaped": 2, "rejected_controllers_reaped": 2,
            "rejected_cases_seccomp_installations": 2, "rejected_cases_visible_socket_removals": 4,
        }.items():
            self.assertIs(type(report[field]), int, field)
            self.assertEqual(report[field], expected, field)


if __name__ == "__main__":
    unittest.main()
