"""Versioned metadata-budget SFT lifecycle and aggregate resource ledger.

This is NOT an executable launcher, process-shared service, OS sandbox, memory
limiter, network meter, or authorization validator. A future guarded worker must
route every rank's operation through one controller BEFORE performing it and
provide trustworthy clocks and memory observations. Calling this module cannot
grant permission to initialize CUDA, read inputs, or dispatch a worker.

Reservations are irrevocable, even when the underlying operation fails. A log
reservation charges both the log and private-write budgets; private_write_bytes
means *additional non-log writes*. All ranks share one atomic ledger. Its lock
protects threads in this process only; separate process copies are NOT shared.
Only fixed codes and aggregate counters are retained, never paths or text.
The required positive encoded-metadata read cap is separate from binary/index
caps. Missing old-style caps are rejected; no implicit allowance or fallback is
introduced. Existing lifecycle, geometry and other resource behavior are frozen.
"""

from __future__ import annotations

import threading
import time

CAP_FIELDS = (
    "maximum_combined_runtime_seconds",
    "maximum_cost_microusd",
    "total_host_hourly_rate_microusd",
    "maximum_host_ram_bytes",
    "maximum_per_gpu_memory_bytes",
    "maximum_private_write_bytes",
    "maximum_checkpoint_read_bytes",
    "maximum_encoded_binary_read_bytes",
    "maximum_encoded_index_read_bytes",
    "maximum_encoded_metadata_read_bytes",
    "maximum_log_bytes",
)
CAP_EXTRA_FIELDS = (
    "maximum_invocations",
    "maximum_global_optimizer_steps",
    "maximum_external_network_bytes",
    "maximum_inference_calls",
)
COUNT_FIELDS = (
    "sequential_invocations",
    "global_optimizer_steps",
    "rank_optimizer_steps",
    "logical_sample_events",
    "maximum_unique_train_records",
    "rank_local_input_loads",
    "maximum_encoded_index_rows",
    "maximum_selected_binary_bytes",
    "maximum_selected_index_bytes",
    "includes_checkpoint_hashing_or_final_digest",
    "is_total_io_or_cost_budget",
)
SEQUENCE = (
    ("reference", 0, 2, False),
    ("resumed", 0, 1, False),
    ("resumed", 1, 2, True),
)
RESOURCE_FIELDS = (
    "checkpoint_read_bytes",
    "encoded_binary_read_bytes",
    "encoded_index_read_bytes",
    "encoded_metadata_read_bytes",
    "private_write_bytes",
    "log_bytes",
    "encoded_index_rows",
)
ERROR_CODES = frozenset(
    (
        "caps_schema",
        "counts_schema",
        "integer_required",
        "counts_mismatch",
        "operation_cap_changed",
        "cost_cap_too_small",
        "io_cap_too_small",
        "log_cap_too_large",
        "clock_failed",
        "clock_invalid",
        "clock_regressed",
        "deadline_exhausted",
        "cost_exhausted",
        "invocation_boundary",
        "invocation_overlap",
        "no_active_invocation",
        "already_completed",
        "resource_schema",
        "resource_amount",
        "rank_invalid",
        "resource_exhausted",
        "optimizer_boundary",
        "optimizer_pending",
        "optimizer_not_pending",
        "optimizer_exhausted",
        "memory_schema",
        "host_memory_exhausted",
        "gpu_memory_exhausted",
        "memory_observation_missing",
        "interrupted",
    )
)
MAX_INTEGER = 2**63 - 1
NANOSECONDS_PER_SECOND = 1_000_000_000
NANOSECONDS_PER_HOUR = 3600 * NANOSECONDS_PER_SECOND


class SupervisorError(ValueError):
    """An allowlisted fixed code; arbitrary exception messages are never copied."""

    def __init__(self, code: str):
        if type(code) is not str or code not in ERROR_CODES:
            code = "resource_schema"
        super().__init__(code)


def _integer(value: object, *, positive: bool = True) -> bool:
    return type(value) is int and (1 if positive else 0) <= value <= MAX_INTEGER


def _keys(value: object, expected: tuple[str, ...]) -> bool:
    return (
        type(value) is dict
        and all(type(key) is str for key in value)
        and set(value) == set(expected)
    )


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise SupervisorError(code) from None


def validate_inputs(caps: dict, counts: dict, world_size: int) -> None:
    """Validate exact v2 compiler schemas, types, geometry and combined caps."""
    _require(_keys(caps, CAP_FIELDS + CAP_EXTRA_FIELDS), "caps_schema")
    _require(_keys(counts, COUNT_FIELDS), "counts_schema")
    _require(_integer(world_size) and 2 <= world_size <= 12, "integer_required")
    for field in CAP_FIELDS:
        _require(_integer(caps[field]), "integer_required")
    for field in COUNT_FIELDS[:-2]:
        _require(_integer(counts[field]), "integer_required")
    _require(
        counts["includes_checkpoint_hashing_or_final_digest"] is False
        and counts["is_total_io_or_cost_budget"] is False,
        "counts_mismatch",
    )
    rows = counts["maximum_encoded_index_rows"]
    records = counts["maximum_unique_train_records"]
    _require(
        counts["sequential_invocations"] == 3
        and counts["global_optimizer_steps"] == 4
        and counts["rank_optimizer_steps"] == 4 * world_size
        and counts["rank_local_input_loads"] == 3 * world_size
        and world_size <= records <= 12
        and rows == 3 * world_size * records
        and counts["maximum_selected_index_bytes"] == rows * 176
        and counts["maximum_selected_binary_bytes"] % (rows * 8) == 0
        and counts["logical_sample_events"] >= 4 * world_size
        and counts["logical_sample_events"] % (4 * world_size) == 0,
        "counts_mismatch",
    )
    for field, expected in (
        ("maximum_invocations", 3),
        ("maximum_global_optimizer_steps", 4),
        ("maximum_external_network_bytes", 0),
        ("maximum_inference_calls", 0),
    ):
        _require(type(caps[field]) is int and caps[field] == expected, "operation_cap_changed")
    cost = (
        caps["total_host_hourly_rate_microusd"] * caps["maximum_combined_runtime_seconds"] + 3599
    ) // 3600
    _require(cost <= caps["maximum_cost_microusd"], "cost_cap_too_small")
    _require(
        caps["maximum_encoded_binary_read_bytes"] >= counts["maximum_selected_binary_bytes"]
        and caps["maximum_encoded_index_read_bytes"] >= counts["maximum_selected_index_bytes"],
        "io_cap_too_small",
    )
    _require(caps["maximum_log_bytes"] <= caps["maximum_private_write_bytes"], "log_cap_too_large")


class Supervisor:
    """One non-retryable three-invocation lifecycle, using monotonic nanoseconds.

    A logical optimizer step is reserved by the coordinator once, before any
    rank performs it, then acknowledged after all ranks succeed. Rank-local
    reads/writes reserve separately on the same object. An interrupted invocation
    remains consumed forever. There is no reset, refund, or resume-after-failure
    API. The approved resumed run is a separate scheduled invocation, not a retry.

    The combined clock starts at construction and includes gaps between runs.
    Deadline checks are cooperative at each method call, not an asynchronous
    process killer. Memory checks reject observed overages only; observations
    cannot enforce or certify continuous memory use. At least one complete memory
    observation per invocation is required before it can finish.
    """

    def __init__(self, caps: dict, counts: dict, *, world_size: int, clock_ns=None):
        validate_inputs(caps, counts, world_size)
        self._caps = dict(caps)
        self._counts = dict(counts)
        self._world_size = world_size
        self._clock = time.monotonic_ns if clock_ns is None else clock_ns
        self._lock = threading.Lock()
        self._state = "ready"
        self._failure = None
        self._started = 0
        self._completed = 0
        self._steps_reserved = 0
        self._steps_completed = 0
        self._samples_reserved = 0
        self._active_step = None
        self._pending_step = None
        self._memory_observed = False
        self._memory_observations = 0
        self._host_peak = 0
        self._gpu_peak = 0
        self._used = dict.fromkeys(RESOURCE_FIELDS, 0)
        self._reservations = 0
        self._elapsed_ns = 0
        self._cost = 0
        self._start_ns = self._read_clock()
        self._last_ns = self._start_ns

    def _fail(self, code: str) -> None:
        if self._failure is None:
            self._failure = code
            self._state = "failed"
        raise SupervisorError(self._failure) from None

    def _need(self, condition: bool, code: str) -> None:
        if not condition:
            self._fail(code)

    def _read_clock(self) -> int:
        try:
            now = self._clock()
        except BaseException:
            self._fail("clock_failed")
        self._need(_integer(now, positive=False), "clock_invalid")
        return now

    def _tick(self) -> None:
        if self._failure is not None:
            self._fail(self._failure)
        self._need(self._state != "completed", "already_completed")
        now = self._read_clock()
        self._need(now >= self._last_ns, "clock_regressed")
        self._last_ns = now
        self._elapsed_ns = now - self._start_ns
        self._cost = (
            self._elapsed_ns * self._caps["total_host_hourly_rate_microusd"]
            + NANOSECONDS_PER_HOUR - 1
        ) // NANOSECONDS_PER_HOUR
        self._need(self._cost <= self._caps["maximum_cost_microusd"], "cost_exhausted")
        self._need(
            self._elapsed_ns
            < self._caps["maximum_combined_runtime_seconds"] * NANOSECONDS_PER_SECOND,
            "deadline_exhausted",
        )

    def _active(self) -> None:
        self._tick()
        self._need(self._state == "active", "no_active_invocation")

    def begin_invocation(self, run: str, from_step: int, stop_step: int, resume: bool) -> None:
        with self._lock:
            self._tick()
            self._need(self._state == "ready", "invocation_overlap")
            self._need(
                type(run) is str
                and type(from_step) is int
                and type(stop_step) is int
                and type(resume) is bool
                and (run, from_step, stop_step, resume) == SEQUENCE[self._started],
                "invocation_boundary",
            )
            self._started += 1
            self._state = "active"
            self._active_step = from_step
            self._memory_observed = False

    def reserve(self, rank: int, **amounts: int) -> None:
        """Atomically reserve nonrefundable bytes/rows before their operations.

        Missing categories mean zero. log_bytes additionally debit private writes.
        If any bound fails no category is committed and the lifecycle fails closed.
        A real operation failure after reservation must call interrupt().
        """
        with self._lock:
            self._active()
            self._need(type(rank) is int and 0 <= rank < self._world_size, "rank_invalid")
            self._need(bool(amounts) and set(amounts) <= set(RESOURCE_FIELDS), "resource_schema")
            self._need(
                all(_integer(value, positive=False) for value in amounts.values())
                and any(value > 0 for value in amounts.values()),
                "resource_amount",
            )
            proposed = {field: self._used[field] + amounts.get(field, 0)
                        for field in RESOURCE_FIELDS}
            proposed["private_write_bytes"] += amounts.get("log_bytes", 0)
            for field, amount in proposed.items():
                limit = (
                    self._counts["maximum_encoded_index_rows"]
                    if field == "encoded_index_rows"
                    else self._caps["maximum_" + field]
                )
                self._need(amount <= limit, "resource_exhausted")
            self._used = proposed
            self._reservations += 1

    def reserve_optimizer_step(self, from_step: int, to_step: int) -> None:
        """Reserve one global step before work; not one step per rank."""
        with self._lock:
            self._active()
            self._need(self._pending_step is None, "optimizer_pending")
            self._need(
                type(from_step) is int
                and type(to_step) is int
                and from_step == self._active_step
                and to_step == from_step + 1
                and to_step <= SEQUENCE[self._started - 1][2],
                "optimizer_boundary",
            )
            self._need(
                self._steps_reserved < self._caps["maximum_global_optimizer_steps"],
                "optimizer_exhausted",
            )
            self._steps_reserved += 1
            self._samples_reserved += self._counts["logical_sample_events"] // 4
            self._pending_step = to_step

    def finish_optimizer_step(self, to_step: int) -> None:
        with self._lock:
            self._active()
            self._need(self._pending_step is not None, "optimizer_not_pending")
            self._need(type(to_step) is int and to_step == self._pending_step, "optimizer_boundary")
            self._steps_completed += 1
            self._active_step = to_step
            self._pending_step = None

    def observe_memory(self, host_ram_bytes: int, per_gpu_memory_bytes: list[int]) -> None:
        """Check one trusted simultaneous host/all-GPU observation; no OS quota."""
        with self._lock:
            self._active()
            self._need(
                _integer(host_ram_bytes, positive=False)
                and type(per_gpu_memory_bytes) is list
                and len(per_gpu_memory_bytes) == self._world_size
                and all(_integer(value, positive=False) for value in per_gpu_memory_bytes),
                "memory_schema",
            )
            self._host_peak = max(self._host_peak, host_ram_bytes)
            self._gpu_peak = max(self._gpu_peak, max(per_gpu_memory_bytes))
            self._memory_observations += 1
            self._need(host_ram_bytes <= self._caps["maximum_host_ram_bytes"],
                       "host_memory_exhausted")
            self._need(
                self._gpu_peak <= self._caps["maximum_per_gpu_memory_bytes"], "gpu_memory_exhausted"
            )
            self._memory_observed = True

    def finish_invocation(self, stop_step: int) -> None:
        with self._lock:
            self._active()
            self._need(self._pending_step is None, "optimizer_pending")
            self._need(
                type(stop_step) is int
                and stop_step == self._active_step == SEQUENCE[self._started - 1][2],
                "invocation_boundary",
            )
            self._need(self._memory_observed, "memory_observation_missing")
            self._completed += 1
            self._active_step = None
            self._state = "completed" if self._completed == len(SEQUENCE) else "ready"

    def interrupt(self) -> None:
        """Permanently fail closed; no exception detail or worker text is accepted."""
        with self._lock:
            if self._failure is not None:
                self._fail(self._failure)
            self._fail("interrupted")

    def checkpoint(self) -> None:
        """Cooperatively poll the total clock/cost budget, including between runs."""
        with self._lock:
            self._tick()

    def summary(self) -> dict:
        """Snapshot only: no clock call, external reads, raw text or rank identifiers."""
        with self._lock:
            return {
                "schema_version": "lumi.sft-supervisor-summary.v0.2",
                "state": self._state,
                "failure_code": self._failure,
                "invocations_started": self._started,
                "invocations_completed": self._completed,
                "global_optimizer_steps_reserved": self._steps_reserved,
                "global_optimizer_steps_completed": self._steps_completed,
                "logical_sample_events_reserved": self._samples_reserved,
                "resource_reservations": self._reservations,
                "resources_reserved": dict(self._used),
                "elapsed_nanoseconds_last_observed": self._elapsed_ns,
                "cost_microusd_last_observed": self._cost,
                "memory_observations": self._memory_observations,
                "maximum_host_ram_bytes_observed": self._host_peak,
                "maximum_per_gpu_memory_bytes_observed": self._gpu_peak,
                "executable_dispatch_enabled": False,
                "os_memory_enforcement": False,
                "os_network_enforcement": False,
                "multiprocess_shared_state": False,
                "production_eligible": False,
            }
