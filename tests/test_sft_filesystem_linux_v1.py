"""Remote Linux acceptance only; never execute even this skip wrapper on the Mac."""

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


@unittest.skipUnless(sys.platform == "linux", "filesystem acceptance is remote Linux only")
class FilesystemLinuxTests(unittest.TestCase):
    def test_filesystem_policy_escaping_cleanup_and_pinned_worker_run(self):
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
             str(ROOT / "scripts/run_sft_filesystem_fixture_v1.py"), "--synthetic-only"],
            cwd=ROOT, env={"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"},
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            stdout, _ = process.communicate(input=payload, timeout=300)
        except subprocess.TimeoutExpired:
            # The unprivileged wrapper has no root cgroup/mount cleanup authority.
            self.fail("filesystem_fixture_deadline_cleanup_unconfirmed")
        self.assertLessEqual(len(stdout), 4096)
        try:
            report = json.loads(stdout)
        except (ValueError, UnicodeError):
            self.fail("filesystem_fixture_missing_structural_report")
        self.assertEqual(process.returncode, 0, report)
        self.assertEqual(report.get("status"), "passed", report)
        self.assertEqual(report["schema_version"], "lumi.sft-filesystem-linux-fixture.v0.1")
        for field in (
            "filesystem_view_verified", "landlock_policy_verified",
            "same_uid_host_proc_root_boundary_verified", "readonly_canary_verified",
            "bounded_tmpfs_exhaustion_verified", "host_root_and_mount_namespace_unchanged",
            "direct_ip_network_isolation_verified", "actual_parent_loss_cleanup_verified",
            "deadline_cleanup_verified", "double_fork_setsid_descendants_reaped",
            "pinned_worker_run_fixture_completed", "privileges_dropped_before_payload",
            "cgroup_pidfd_authority_closed_before_payload", "no_root_ml_import",
            "inherited_file_directory_authority_removed",
        ):
            self.assertIs(report[field], True, field)
        for field in (
            "real_model_or_data_access", "cuda_used", "production_eligible",
            "real_sft_execution_permitted", "complete_os_sandbox_verified",
            "full_pathname_unix_isolation_verified", "host_metadata_confidentiality_verified",
            "runtime_origin_authenticated", "real_host_admission_verified",
            "cumulative_write_quota_verified", "host_mounts_modified",
        ):
            self.assertIs(report[field], False, field)
        for field, expected in {
            "cases_verified": 4, "owned_cgroups_empty_and_removed": 4,
            "owned_mountpoints_removed": 4, "owned_sentinel_processes_reaped": 4,
            "fresh_loopback_only_namespaces_verified": 4,
            "worker_run_invocations_completed": 3, "worker_run_rank_workers_verified": 6,
            "worker_run_global_optimizer_steps_completed": 4,
            "worker_run_rank_optimizer_steps_completed": 8,
        }.items():
            self.assertEqual(report[field], expected, field)


if __name__ == "__main__":
    unittest.main()
