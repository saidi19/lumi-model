"""Pure bounded SFT cursor/counter contract over already admitted aggregates.

The trusted caller computes the opaque run identity from the fresh execution and
resume plan, all pinned input/profile identities, and the selected data identity.
Comparing that hash is not an admission proof. These helpers do not read files,
load models, select examples, schedule learning rates, or publish checkpoints.
They accept only completed WorkerStep aggregates; a caller must not advance state
for an uncommitted optimizer step or report this private state as public metrics.

Cursor and counter semantics follow frozen distributed SFT v2: each completed
step consumes world_size * per_rank_batch_size * accumulation global records,
including across epoch boundaries. Tokens and shifted supervision are already
globally summed by WorkerStep; loss is already supervision-weighted. This module
does not multiply or reweight them. The next LR/planner invocation remains the
caller's responsibility, using step + 1 and sequence_cursor respectively.
"""

from __future__ import annotations

import math

SCHEMA_VERSION = "lumi.sft-training-state.v0.1"
MAX_STEP = 2
ERROR_CODES = frozenset({
    "invalid_arguments", "state_rejected", "identity_mismatch", "step_mismatch",
    "metrics_rejected",
})
_STATE_FIELDS = (
    "schema_version", "run_identity_sha256", "step", "sequence_cursor", "records_seen",
    "tokens_seen", "supervised_tokens_seen", "last_loss", "best_loss",
)
_CHECKPOINT_FIELDS = frozenset({"latest_checkpoint", "status"})
_METRIC_FIELDS = frozenset({"to_step", "loss", "gradient_norm", "tokens", "supervised_tokens"})


class TrainingStateError(ValueError):
    """Only a fixed code, never caller-controlled text or identities."""

    def __init__(self, code="state_rejected"):
        self.code = code if type(code) is str and code in ERROR_CODES else "state_rejected"
        super().__init__(self.code)


def _hash(value):
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _integer(value, *, minimum=0, maximum=None):
    return (type(value) is int and value >= minimum
            and (maximum is None or value <= maximum))


def _number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _keys(value):
    # Reject subclasses and non-string keys before set/equality invokes hooks.
    if type(value) is not dict or any(type(key) is not str for key in value):
        return None
    return frozenset(value)


def make_state(run_identity_sha256):
    """Return a new zero-step private state for the caller's pinned identity."""
    if not _hash(run_identity_sha256):
        raise TrainingStateError("invalid_arguments") from None
    return {
        "schema_version": SCHEMA_VERSION,
        "run_identity_sha256": run_identity_sha256,
        "step": 0,
        "sequence_cursor": 0,
        "records_seen": 0,
        "tokens_seen": 0,
        "supervised_tokens_seen": 0,
        "last_loss": None,
        "best_loss": None,
    }


def validate_state(state, *, run_identity_sha256, from_step, global_records_per_step):
    """Validate an exact resume point and return a detached canonical base state.

    CheckpointWorker adds only latest_checkpoint and status to this schema. The
    pair is accepted only for the exact nonzero step and removed from the copy;
    arbitrary metadata or partial checkpoint adornments are rejected.
    """
    if (not _hash(run_identity_sha256)
            or not _integer(from_step, maximum=MAX_STEP)
            or not _integer(global_records_per_step, minimum=1)):
        raise TrainingStateError("invalid_arguments") from None
    keys = _keys(state)
    base_fields = frozenset(_STATE_FIELDS)
    if keys not in (base_fields, base_fields | _CHECKPOINT_FIELDS):
        raise TrainingStateError("state_rejected") from None
    if (type(state["schema_version"]) is not str
            or state["schema_version"] != SCHEMA_VERSION
            or not _hash(state["run_identity_sha256"])):
        raise TrainingStateError("state_rejected") from None
    if state["run_identity_sha256"] != run_identity_sha256:
        raise TrainingStateError("identity_mismatch") from None
    step = state["step"]
    if not _integer(step, maximum=MAX_STEP):
        raise TrainingStateError("state_rejected") from None
    if step != from_step:
        raise TrainingStateError("step_mismatch") from None
    if keys != base_fields:
        if (step == 0 or type(state["latest_checkpoint"]) is not str
                or state["latest_checkpoint"] != f"checkpoint-{step:08d}"
                or type(state["status"]) is not str or state["status"] != "checkpoint"):
            raise TrainingStateError("state_rejected") from None
    if any(not _integer(state[field]) for field in (
        "sequence_cursor", "records_seen", "tokens_seen", "supervised_tokens_seen",
    )):
        raise TrainingStateError("state_rejected") from None
    if (state["sequence_cursor"] != step * global_records_per_step
            or state["records_seen"] != state["sequence_cursor"]):
        raise TrainingStateError("state_rejected") from None
    tokens, supervised = state["tokens_seen"], state["supervised_tokens_seen"]
    last_loss, best_loss = state["last_loss"], state["best_loss"]
    if step == 0:
        if tokens != 0 or supervised != 0 or last_loss is not None or best_loss is not None:
            raise TrainingStateError("state_rejected") from None
    elif (not step <= supervised <= tokens
            or not _number(last_loss) or not _number(best_loss)
            or best_loss > last_loss or step == 1 and best_loss != last_loss):
        raise TrainingStateError("state_rejected") from None
    result = {field: state[field] for field in _STATE_FIELDS}
    if step:
        result["last_loss"], result["best_loss"] = float(last_loss), float(best_loss)
    return result


def advance_state(state, *, metrics, global_records_per_step):
    """Return the next state after one exact completed WorkerStep, through step 2.

    The original state and aggregate mapping are never changed. This is not a
    transactional optimizer fence: the caller supplies an already completed step.
    """
    if _keys(state) is None:
        raise TrainingStateError("state_rejected") from None
    current = validate_state(
        state, run_identity_sha256=state.get("run_identity_sha256"),
        from_step=state.get("step"), global_records_per_step=global_records_per_step,
    )
    if _keys(metrics) != _METRIC_FIELDS:
        raise TrainingStateError("metrics_rejected") from None
    next_step = current["step"] + 1
    if (not _integer(metrics["to_step"], minimum=1, maximum=MAX_STEP)
            or metrics["to_step"] != next_step):
        raise TrainingStateError("step_mismatch") from None
    if (not _number(metrics["loss"]) or not _number(metrics["gradient_norm"])
            or not _integer(metrics["tokens"], minimum=1)
            or not _integer(metrics["supervised_tokens"], minimum=1)
            or metrics["supervised_tokens"] > metrics["tokens"]):
        raise TrainingStateError("metrics_rejected") from None
    loss = float(metrics["loss"])
    cursor = current["sequence_cursor"] + global_records_per_step
    return {
        **current,
        "step": next_step,
        "sequence_cursor": cursor,
        "records_seen": cursor,
        "tokens_seen": current["tokens_seen"] + metrics["tokens"],
        "supervised_tokens_seen": current["supervised_tokens_seen"] + metrics["supervised_tokens"],
        "last_loss": loss,
        "best_loss": loss if current["best_loss"] is None else min(current["best_loss"], loss),
    }
