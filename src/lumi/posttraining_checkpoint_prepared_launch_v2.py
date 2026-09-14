"""One fixed prepared-fixture launch for an independently authenticated parent.

This adapter binds the private layout returned by the installer and dispatches
the existing guardian once. A caller-supplied hash is not origin, permission or
host admission. The trusted Linux parent must first contain and authenticate
the generator, admit the runtime/rootfs/source closure, exclude external writers
and pin the entire private layout. None of those premises is manufactured here.

No acquisition, generator launch, data-path override, retry, cleanup, real
checkpoint or training interface is provided. Tests replace the guardian; no
native process is started on the Mac. Never log the private layout object.
"""

from __future__ import annotations

import re
import time
from pathlib import PurePosixPath

from . import posttraining_checkpoint_guard_composition_v3 as guard
from . import posttraining_checkpoint_prepared_layout_v2 as layout_contract
from .posttraining_final_base_metadata_v1 import canonical, sha

SCHEMA = "lumi.prepared-checkpoint-launch.v0.1"
MAXIMUM_LAYOUT_BYTES = 65536
ERROR_CODES = frozenset({"runtime_rejected", "layout_rejected", "binding_rejected",
                         "arguments_rejected", "already_attempted", "guard_failed",
                         "result_rejected", "deadline_exhausted"}) | guard.ERROR_CODES
_ATTEMPTED = False


class PreparedLaunchError(ValueError):
    def __init__(self, code="guard_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "guard_failed"
        super().__init__(self.code)


def _need(value, code="layout_rejected"):
    if not value:
        raise PreparedLaunchError(code) from None


def _pin(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) and value != "0" * 64


def validate_layout(layout, *, expected_layout_sha256, expected_manifest_sha256,
                    expected_entry_sha256, expected_control_sha256,
                    expected_generation_nonce_sha256):
    """Pure closed structural binding, detached from all caller-owned mappings."""
    try:
        _need(all(_pin(value) for value in (expected_layout_sha256, expected_manifest_sha256,
            expected_entry_sha256, expected_control_sha256, expected_generation_nonce_sha256)),
            "binding_rejected")
        _need(type(layout) is dict and set(layout) == {"receipt", "mount_arguments", "environment"})
        receipt = layout["receipt"]
        fixed = {"schema": layout_contract.SCHEMA, "status": "installed_private_prepared_fixture",
            "source_payload_files_opened": 0, "new_input_payload_readback_bytes": 0,
            "installation_frozen": True, "source_mounts_installed": False,
            "external_sibling_cleanup_provided": False,
            "source_manifest_sha256": expected_manifest_sha256,
            "entry_sha256": expected_entry_sha256, "helper_sha256": layout_contract.HELPER_SHA256,
            "control_sha256": expected_control_sha256,
            "generation_nonce_sha256": expected_generation_nonce_sha256,
            **dict.fromkeys(layout_contract.FALSE_FLAGS, False)}
        caps = {"rootfs_addon_files": 140, "rootfs_addon_bytes": 6 * 1024**2,
                "generated_input_files": 32, "generated_input_bytes": 4 * 1024**2,
                "private_content_write_bytes": layout_contract.MAXIMUM_INSTALL_BYTES,
                "new_input_bytes_checked_against_bound_inventory": 4 * 1024**2}
        _need(type(receipt) is dict and set(receipt) == set(fixed) | set(caps)
              and all(type(receipt[key]) is type(value) and receipt[key] == value
                      for key, value in fixed.items())
              and all(type(receipt[key]) is int and 0 < receipt[key] <= maximum
                      for key, maximum in caps.items()))
        mounts = guard.child_bootstrap.validate_mount_arguments(layout["mount_arguments"])
        environment = guard.child_bootstrap.validate_environment(layout["environment"])
        _need(environment == dict(guard.child_bootstrap.FIXED_ENV) | {
            "LUMI_IMPORT_SOURCE_MANIFEST_SHA256": expected_manifest_sha256,
            "LUMI_PREPARED_FIXTURE_ENTRY_SHA256": expected_entry_sha256,
            "LUMI_PREPARED_FIXTURE_CONTROL_SHA256": expected_control_sha256}, "binding_rejected")
        _need(mounts["scratch_bytes"] == 536870912 and mounts["shm_bytes"] == 67108864)
        guard._match_input_counts(receipt, mounts)
        _need(receipt["private_content_write_bytes"] == receipt["rootfs_addon_bytes"]
              + receipt["generated_input_bytes"]
              and receipt["new_input_bytes_checked_against_bound_inventory"]
              == receipt["generated_input_bytes"])
        copied = {"receipt": dict(receipt), "mount_arguments": mounts, "environment": environment}
        raw = canonical(copied)
        _need(len(raw) <= MAXIMUM_LAYOUT_BYTES and sha(raw) == expected_layout_sha256,
              "binding_rejected")
        return copied
    except PreparedLaunchError:
        raise
    except BaseException:
        raise PreparedLaunchError("layout_rejected") from None


def validate_guard_result(value, layout, expected_nonce):
    """Require independent guardian lifecycle evidence, not just worker success."""
    try:
        fixed = {"schema": guard.SCHEMA, "status": "passed", "synthetic_only": True,
            "owned_cgroups_empty_and_removed": 1, "kill_attempts": 1,
            "controller_and_deadline_watch_installed_before_release": True,
            "worker_exit_status_verified": True,
            "memory_max_bytes": guard.LIMITS["memory_max_bytes"], "swap_max_bytes": 0,
            "pids_max": guard.LIMITS["pids_max"],
            "cpu_quota_us": guard.LIMITS["cpu_quota_us"],
            "cpu_period_us": guard.LIMITS["cpu_period_us"], "maximum_seconds": 90,
            "tmpfs_capacity_bytes": 536870912, "shm_capacity_bytes": 67108864,
            **dict.fromkeys(guard.NONCLAIMS, False)}
        _need(type(value) is dict and set(value) == set(fixed) | {
            "worker", "workload_processes_reaped"}
            and all(type(value[key]) is type(expected) and value[key] == expected
                    for key, expected in fixed.items())
            and type(value["workload_processes_reaped"]) is int
            and 1 <= value["workload_processes_reaped"] <= 4, "result_rejected")
        worker = guard._worker(value["worker"], layout["environment"], expected_nonce)
        guard._match_input_counts(worker, layout["mount_arguments"])
        return {**value, "worker": worker}
    except PreparedLaunchError:
        raise
    except BaseException:
        raise PreparedLaunchError("result_rejected") from None


def launch_prepared_fixture(layout, *, expected_layout_sha256, expected_manifest_sha256,
                            expected_entry_sha256, expected_control_sha256,
                            expected_generation_nonce_sha256, base_script_path,
                            expected_base_sha256, deadline_ns):
    """One native tiny guardian dispatch; no full-size or production authority."""
    global _ATTEMPTED
    _need(not _ATTEMPTED, "already_attempted")
    try:
        guard.child_bootstrap._runtime()
    except BaseException:
        raise PreparedLaunchError("runtime_rejected") from None
    copied = validate_layout(layout, expected_layout_sha256=expected_layout_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_entry_sha256=expected_entry_sha256,
        expected_control_sha256=expected_control_sha256,
        expected_generation_nonce_sha256=expected_generation_nonce_sha256)
    _need(type(base_script_path) is str and 0 < len(base_script_path) <= 4096
          and all(32 <= ord(char) < 127 for char in base_script_path)
          and type(expected_base_sha256) is str and expected_base_sha256 == guard.BASE_SHA256,
          "arguments_rejected")
    selected = PurePosixPath(base_script_path)
    _need(selected.is_absolute() and str(selected) == base_script_path
          and ".." not in selected.parts
          and selected.parts[-2:] == ("scripts", "run_sft_cgroup_fixture_v1.py"),
          "arguments_rejected")
    now = time.monotonic_ns()
    _need(type(deadline_ns) is int and now < deadline_ns <= now + 90_000_000_000,
          "arguments_rejected")
    _ATTEMPTED = True
    try:
        result = guard.run_prepared_synthetic_guard(mount_arguments=copied["mount_arguments"],
            environment=copied["environment"],
            expected_generation_nonce_sha256=expected_generation_nonce_sha256,
            base_script_path=base_script_path, expected_base_sha256=expected_base_sha256,
            deadline_ns=deadline_ns)
        verified = validate_guard_result(result, copied, expected_generation_nonce_sha256)
        _need(time.monotonic_ns() < deadline_ns, "deadline_exhausted")
        return {"schema": SCHEMA, "status": "PASS_generated_fixture_guardian_verified",
            "private_layout_sha256": expected_layout_sha256,
            "generation_nonce_sha256": expected_generation_nonce_sha256,
            "guardian_result_sha256": sha(canonical(verified)),
            "synthetic_only": True, "guard_launches": 1, "coordinator_passes": 1,
            "generated_input_files": verified["worker"]["generated_input_files"],
            "generated_input_bytes": verified["worker"]["generated_input_bytes"],
            "owned_cgroups_empty_and_removed": 1, "kill_attempts": 1,
            "runtime_origin_authenticated": False, "generator_containment_verified": False,
            "host_admission_verified": False, "full_size_import_authorized": False,
            "real_4b_import": False, "training_authorized": False, "production_eligible": False,
            "retry_allowed": False}
    except PreparedLaunchError:
        raise
    except BaseException as error:
        code = error.code if type(error) is guard.GuardCompositionError else "guard_failed"
        raise PreparedLaunchError(code) from None

