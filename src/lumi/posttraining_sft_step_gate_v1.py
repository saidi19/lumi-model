"""Cooperative shared-step gate for the frozen zero-to-two-step SFT schedule.

No ML, file, socket, model, or optimizer access occurs here. Rank zero reserves
the exact next global step before any caller work. Every rank must then receive
the trusted synchronize('reserved', to_step) service's confirmation before the
body runs. After the body, synchronize('updated', to_step) must establish that
ALL ranks successfully updated. Each rank sends complete_step(to_step), then
synchronize('completed', to_step) must obtain the parent's confirmed all-rank
completion before this gate advances its local cursor.

Synchronize is a trusted collective AND parent-confirmation service, not merely
an ordinary barrier or local socket-send acknowledgement. The parent must bind
the run, ranks, exact step, reservation, and completion acknowledgements. A
callback returning None asserts that whole contract. This gate cannot prove
transport authentication, peer honesty, process liveness, or OS enforcement.

Callbacks must return exactly None. Any failure is terminal, interrupts once,
does not advance the local cursor, and never retries, refunds, or rolls back an
optimizer update. The admitted caller must retire all ranks on failure. Bounds
permit only 0->1, 1->2, or 0->2; authorization for the three frozen invocations
and their combined costs remains with the shared controller, not this object.
Only fixed error args and first-failure codes are safe to report; never emit
suppressed exception contexts or tracebacks.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager

ERROR_CODES = frozenset({
    "invalid_arguments", "owner_mismatch", "reentry", "terminal_failure", "step_mismatch",
    "reserve_failed", "synchronize_failed", "complete_failed", "body_failed",
})
STAGES = frozenset({"idle", "reserve", "reserved", "work", "updated", "complete", "completed"})


class StepGateError(RuntimeError):
    def __init__(self, code="invalid_arguments"):
        self.code = code if type(code) is str and code in ERROR_CODES else "invalid_arguments"
        super().__init__(self.code)


class StepGate:
    """One owner's sequential step contexts; local success is not run admission."""

    def __init__(self, *, rank, world_size, reserve_step, complete_step, synchronize,
                 interrupt, from_step, stop_step):
        if hasattr(self, "_owner"):
            self._fail("invalid_arguments")
        if (type(rank) is not int or type(world_size) is not int
                or not 1 <= world_size <= 12 or not 0 <= rank < world_size
                or type(from_step) is not int or type(stop_step) is not int
                or not 0 <= from_step < stop_step <= 2
                or not all(callable(callback) for callback in (
                    reserve_step, complete_step, synchronize, interrupt))):
            raise StepGateError("invalid_arguments") from None
        self._rank, self._world_size = rank, world_size
        self._current, self._stop = from_step, stop_step
        self._reserve_step, self._complete_step = reserve_step, complete_step
        self._synchronize, self._interrupt = synchronize, interrupt
        self._owner = os.getpid(), threading.get_ident()
        self._busy, self._failed, self._interrupted = False, False, False
        self._stage, self._failure_stage, self._failure_code = "idle", None, None

    def __repr__(self):
        return "<StepGate private>"

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def current_step(self):
        return self._current

    @property
    def failed(self):
        return self._failed

    @property
    def failure_stage(self):
        return self._failure_stage

    @property
    def failure_code(self):
        return self._failure_code

    def _fail(self, code):
        self._failed = True
        if self._failure_stage is None:
            self._failure_stage = self._stage if self._stage in STAGES else "idle"
            self._failure_code = code if code in ERROR_CODES else "invalid_arguments"
        if not self._interrupted:
            self._interrupted = True
            try:
                self._interrupt()
            except BaseException:
                pass
        raise StepGateError(code) from None

    def _check_live(self):
        if self._owner != (os.getpid(), threading.get_ident()):
            self._fail("owner_mismatch")
        if self._failed:
            self._fail("terminal_failure")

    def _call(self, stage, code, callback, *arguments):
        self._check_live()
        self._stage = stage
        try:
            result = callback(*arguments)
        except BaseException:
            self._fail(code)
        self._check_live()  # A callback cannot swallow a nested terminal failure.
        if result is not None:
            self._fail(code)

    @contextmanager
    def step(self, to_step):
        self._check_live()
        if self._busy:
            self._fail("reentry")
        if type(to_step) is not int or to_step != self._current + 1 or to_step > self._stop:
            self._fail("step_mismatch")
        self._busy = True
        try:
            if self._rank == 0:
                self._call("reserve", "reserve_failed", self._reserve_step, self._current, to_step)
            self._call("reserved", "synchronize_failed", self._synchronize, "reserved", to_step)
            self._stage = "work"
            try:
                yield
            except BaseException:
                self._fail("body_failed")
            self._check_live()
            self._call("updated", "synchronize_failed", self._synchronize, "updated", to_step)
            self._call("complete", "complete_failed", self._complete_step, to_step)
            self._call("completed", "synchronize_failed", self._synchronize, "completed", to_step)
            self._check_live()
            self._current, self._stage = to_step, "idle"
        finally:
            self._busy = False
