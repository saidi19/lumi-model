"""Pure supplied-byte validation for one CPU lookup acceptance envelope.

The caller supplies trusted controls and the pinned diagnostic validator/spec.
This parser does not authenticate a run/image or grant execution permission.
No filesystem, native, package-discovery, process, or network operations occur.
"""

from __future__ import annotations

import json

SCHEMA = "lumi.sft-cpu-lookup-acceptance-envelope.v0.1"
MAXIMUM_REPORT_BYTES = 32_768
OBSERVER_SCHEMA = "lumi.sft-oci-worker-probe.v0.1"
WORKER_SCHEMA = "lumi.sft-worker-run-fixture.v0.1"
CONTROL_FIELDS = (
    "source_commit",
    "software_lock_sha256",
    "base_inventory_sha256",
    "source_inventory_sha256",
)
ORIGIN_FIELDS = (
    "file_backed_modules",
    "source_modules",
    "dependency_modules",
    "runtime_modules",
    "distribution_versions",
    "map_lines",
    "executable_file_mappings",
    "executable_files",
    "anonymous_executable_mappings",
    "generated_python_modules",
)
OBSERVER_TRUE_FLAGS = (
    "inputs_barrier_and_final_origins_verified",
    "declared_distribution_versions_verified",
    "file_backed_executable_origins_verified",
    "parent_ml_import_absent",
)
OBSERVER_FALSE_FLAGS = (
    "real_sft_execution_permitted",
    "production_eligible",
    "real_model_or_data_access",
    "cuda_used",
    "complete_dynamic_dependency_closure_verified",
    "all_executable_code_origin_authenticated",
    "large_library_rehash_performed",
    "host_configuration_independently_verified",
    "real_host_admission_verified",
)
OBSERVER_COUNTS = {
    "rank_workers_witnessed": 6,
    "origin_observations": 12,
    "invocations_witnessed": 3,
}
WORKER_TRUE_FLAGS = (
    "real_cpu_ddp_used",
    "real_torch_dcp_used",
    "real_worker_run_used",
    "single_parent_supervisor_verified",
    "all_three_invocations_completed",
    "fresh_workers_and_epochs_verified",
    "all_optimizer_steps_in_shared_ledger",
    "six_prefix_loads_in_shared_ledger",
    "checkpoint_io_in_shared_ledger",
    "validation_metadata_present",
    "selected_prefix_only_read",
    "epoch_ack_confirmations_verified",
    "completion_fences_verified",
    "io_phase_confirmations_drained",
    "reference_model_resume_equal",
    "reference_adam_resume_equal",
    "reference_rng_resume_equal",
    "reference_training_state_resume_equal",
    "dropout_rng_consumption_verified",
    "private_modes_verified",
    "owned_processes_reaped",
    "rss_snapshot_observed",
    "loopback_gloo_used",
)
WORKER_FALSE_FLAGS = (
    "real_model_or_data_access",
    "cuda_used",
    "external_network_authorized",
    "os_resource_enforcement_verified",
    "production_eligible",
    "real_sft_execution_permitted",
    "full_content_hashes_verified",
    "fixture_generation_in_shared_ledger",
    "private_comparison_work_in_shared_ledger",
    "total_fixture_cost_metered",
    "safe_unpickling_or_origin_authentication_verified",
)
WORKER_COUNTS = {
    "rank_workers_verified": 6,
    "controlled_rank_optimizer_steps": 8,
    "global_optimizer_steps_reserved": 4,
    "global_optimizer_steps_completed": 4,
    "logical_sample_events_reserved": 16,
    "invocations_started": 3,
    "invocations_completed": 3,
    "controlled_loads": 6,
    "records_loaded": 12,
    "encoded_index_rows": 12,
    "binary_bytes_read": 288,
    "index_bytes_read": 1896,
    "checkpoints_published": 4,
    "rank_checkpoint_restores": 2,
    "commit_fences_sent": 8,
    "memory_observations": 3,
}
# Existing fixture resource ceilings, not new runtime-policy allowances.
# Requests cannot outnumber bytes in the same three bounded parent wire streams.
WORKER_COUNT_LIMITS = {
    "metadata_bytes_read": 65_536,
    "ledger_requests": 100_663_296,
    "parent_wire_bytes": 100_663_296,
    "checkpoint_read_bytes_reserved": 100_663_296,
    "private_write_bytes_reserved": 33_554_432,
}
OBSERVER_KEYS = (
    "schema",
    "status",
    "worker_run",
    "origin_counts",
    *CONTROL_FIELDS,
    *OBSERVER_TRUE_FLAGS,
    *OBSERVER_FALSE_FLAGS,
    *OBSERVER_COUNTS,
)
WORKER_KEYS = (
    "schema_version",
    "status",
    *WORKER_TRUE_FLAGS,
    *WORKER_FALSE_FLAGS,
    *WORKER_COUNTS,
    *WORKER_COUNT_LIMITS,
)
ERROR_CODES = frozenset(
    (
        "invalid_report_type",
        "report_budget_exceeded",
        "report_json_rejected",
        "report_duplicate_key",
        "report_number_rejected",
        "report_noncanonical",
        "envelope_rejected",
        "diagnostic_rejected",
        "observer_rejected",
        "worker_rejected",
        "count_rejected",
        "validation_failed",
    )
)


class ValidationError(ValueError):
    def __init__(self, code):
        self.code = code if code in ERROR_CODES else "validation_failed"
        super().__init__(self.code)


def need(condition, code):
    if not condition:
        raise ValidationError(code)


def canonical(value):
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii")
        + b"\n"
    )


def pairs(items):
    value = {}
    for key, item in items:
        need(key not in value, "report_duplicate_key")
        value[key] = item
    return value


def integer(text):
    need(len(text) <= 12, "report_number_rejected")
    return int(text)


def reject_number(_text):
    raise ValidationError("report_number_rejected")


def counts(value, expected):
    need(
        all(type(value[key]) is int and value[key] == count for key, count in expected.items()),
        "count_rejected",
    )


def worker_report(value):
    need(type(value) is dict and set(value) == set(WORKER_KEYS), "worker_rejected")
    need(
        value["schema_version"] == WORKER_SCHEMA and value["status"] == "passed", "worker_rejected"
    )
    need(
        all(value[key] is True for key in WORKER_TRUE_FLAGS)
        and all(value[key] is False for key in WORKER_FALSE_FLAGS),
        "worker_rejected",
    )
    counts(value, WORKER_COUNTS)
    need(
        all(
            type(value[key]) is int and 0 < value[key] <= maximum
            for key, maximum in WORKER_COUNT_LIMITS.items()
        ),
        "count_rejected",
    )


def observer_report(value, controls, diagnostic):
    need(type(value) is dict and set(value) == set(OBSERVER_KEYS), "observer_rejected")
    need(value["schema"] == OBSERVER_SCHEMA and value["status"] == "passed", "observer_rejected")
    need(
        all(type(value[key]) is str and value[key] == controls[key] for key in CONTROL_FIELDS),
        "observer_rejected",
    )
    need(
        all(value[key] is True for key in OBSERVER_TRUE_FLAGS)
        and all(value[key] is False for key in OBSERVER_FALSE_FLAGS),
        "observer_rejected",
    )
    counts(value, OBSERVER_COUNTS)
    origins = value["origin_counts"]
    need(type(origins) is dict and set(origins) == set(ORIGIN_FIELDS), "observer_rejected")
    need(
        all(type(number) is int and 0 <= number <= 196_608 for number in origins.values()),
        "count_rejected",
    )
    need(origins == diagnostic["frozen_origin_counts"], "observer_rejected")
    worker_report(value["worker_run"])


def validate_report(raw, expected_controls, diagnostic_validator, diagnostic_spec):
    """Return the exact envelope; a valid failed envelope is not acceptance."""
    try:
        need(type(raw) is bytes, "invalid_report_type")
        need(0 < len(raw) <= MAXIMUM_REPORT_BYTES, "report_budget_exceeded")
        try:
            value = json.loads(
                raw.decode("ascii"),
                object_pairs_hook=pairs,
                parse_int=integer,
                parse_float=reject_number,
                parse_constant=reject_number,
            )
        except ValidationError:
            raise
        except (ValueError, UnicodeError, RecursionError, OverflowError):
            raise ValidationError("report_json_rejected") from None
        need(canonical(value) == raw, "report_noncanonical")
        need(
            type(value) is dict
            and set(value) == {"schema", "diagnostic", "observer_report"}
            and value["schema"] == SCHEMA,
            "envelope_rejected",
        )
        try:
            diagnostic = diagnostic_validator.validate_report(
                canonical(value["diagnostic"]), expected_controls, diagnostic_spec
            )
        except Exception:
            raise ValidationError("diagnostic_rejected") from None
        if diagnostic["underlying_status"] == "passed":
            observer_report(value["observer_report"], expected_controls, diagnostic)
        else:
            need(value["observer_report"] is None, "observer_rejected")
        return value
    except ValidationError:
        raise
    except Exception:
        raise ValidationError("validation_failed") from None
