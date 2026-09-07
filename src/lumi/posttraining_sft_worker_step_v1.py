"""Explicit admitted-object SFT step consumer; not a model/data/worker launcher.

No eager ML import. The caller must have admitted and initialized the model,
optimizer, distributed group, bounded memory and input batch provenance already.
This does not read datasets, admit real inputs, create processes, restore models,
publish checkpoints, write metrics or dispatch the frozen training entrypoint.

All tensor/optimizer work performed here is inside one version-pinned StepGate
transaction; constructing caller-supplied batches and independent test references
is outside this module's accounting scope.
Its synchronization callback MUST include the trusted parent's completed-step
fence, not just a rank barrier or a client confirmation-send receipt. A failed
step is terminal; partially updated parameters/optimizer state must be retired.
Process death and collective hangs still require independent bounded teardown.

The shifted-supervision weighting, accumulation/no-sync, clipping and optimizer
order follow frozen distributed SFT v2. Caller-scheduled LR is applied unchanged.
Batch tensors are already collated CPU inputs: their memory use is not another
disk read, and this module must not falsely debit encoded file bytes per batch.
Only returned aggregate values and fixed error codes are safe to report. Never
log suppressed exceptions, third-party tracebacks, model values or batch text.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import threading
from contextlib import nullcontext

from .posttraining_sft_step_gate_v1 import StepGate

ERROR_CODES = frozenset({"invalid_arguments", "owner_mismatch", "reentry", "terminal_failure",
                         "runtime_rejected", "batch_rejected", "nonfinite_loss",
                         "nonfinite_gradient", "step_failed"})
STAGES = frozenset({"idle", "reservation", "runtime", "settings", "batches", "forward", "gradient",
                    "optimizer", "metrics", "completion", "complete"})


class WorkerStepError(RuntimeError):
    def __init__(self, code="step_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "step_failed"
        super().__init__(self.code)


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


class WorkerStep:
    """One rank's step executor over explicitly supplied, exclusively owned inputs.

    Independent rank-local gates must be attached to ONE real parent controller.
    The gate callbacks are trusted capabilities, not verified admission proofs.
    No batch/planner/RNG/cursor or LR-schedule change is performed here.
    """

    def __init__(self, *, gate, world_size, device_type="cpu"):
        if hasattr(self, "_owner"):
            self._fail("invalid_arguments")
        if (type(gate) is not StepGate or type(world_size) is not int
                or not 1 <= world_size <= 12 or gate.world_size != world_size
                or type(device_type) is not str or device_type not in {"cpu", "cuda"}):
            raise WorkerStepError("invalid_arguments") from None
        self._gate, self._world_size, self._device_type = gate, world_size, device_type
        self._owner = os.getpid(), threading.get_ident()
        self._busy, self._failed = False, False
        self._stage, self.failure_stage, self.failure_code = "idle", None, None

    def __repr__(self):
        return "<WorkerStep admitted objects only>"

    def _fail(self, code):
        self._failed = True
        if self.failure_code is None:
            self.failure_code = code if code in ERROR_CODES else "step_failed"
            self.failure_stage = self._stage if self._stage in STAGES else "idle"
        if not self._gate.failed:
            try:
                self._gate._fail("body_failed")
            except BaseException:
                pass
        raise WorkerStepError(self.failure_code) from None

    def _check(self):
        if self._owner != (os.getpid(), threading.get_ident()):
            self._fail("owner_mismatch")
        if self._failed or self._gate.failed:
            self._fail("terminal_failure")

    def _optimizer_binding(self, torch, model, optimizer):
        if not isinstance(optimizer, torch.optim.Optimizer):
            self._fail("runtime_rejected")
        parameters = list(model.parameters())
        available = {id(parameter) for parameter in parameters}
        required = {id(parameter) for parameter in parameters if parameter.requires_grad}
        bound = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
        # Legacy includes frozen parameters; grouped AdamW may omit them.
        if (not required or len(bound) != len(set(bound))
                or not set(bound) <= available or not required <= set(bound)):
            self._fail("runtime_rejected")

    def _runtime(self, model, optimizer, strategy):
        torch = sys.modules.get("torch")
        if torch is None or not re.match(r"2\.(10|14)\.", getattr(torch, "__version__", "")):
            self._fail("runtime_rejected")
        if self._world_size > 1:
            if (not torch.distributed.is_initialized()
                    or torch.distributed.get_world_size() != self._world_size
                    or torch.distributed.get_rank() != self._gate.rank):
                self._fail("runtime_rejected")
        elif torch.distributed.is_initialized():
            if torch.distributed.get_world_size() != 1 or torch.distributed.get_rank() != 0:
                self._fail("runtime_rejected")
        if getattr(model, "training", None) is not True:
            self._fail("runtime_rejected")
        if strategy == "ddp" and self._world_size > 1:
            if (not isinstance(model, torch.nn.parallel.DistributedDataParallel)
                    or model.process_group is not torch.distributed.group.WORLD):
                self._fail("runtime_rejected")
        if strategy == "fsdp_full_shard":
            fsdp = sys.modules.get("torch.distributed.fsdp")
            wrapper = getattr(fsdp, "FullyShardedDataParallel", None)
            if not isinstance(wrapper, type) or not isinstance(model, wrapper):
                self._fail("runtime_rejected")
            if (model.process_group is not torch.distributed.group.WORLD
                    or model.sharding_strategy is not fsdp.ShardingStrategy.FULL_SHARD
                    or getattr(model, "_use_orig_params", None) is not True):
                self._fail("runtime_rejected")
        self._optimizer_binding(torch, model, optimizer)
        device = next(model.parameters()).device
        if device.type != self._device_type:
            self._fail("runtime_rejected")
        if self._device_type == "cuda" and not torch.cuda.is_initialized():
            self._fail("runtime_rejected")
        return torch, device

    def _settings(self, torch, device, settings):
        payload = json.dumps(settings, sort_keys=True, allow_nan=False,
                             separators=(",", ":")).encode("ascii")
        identity = torch.tensor(list(hashlib.sha256(payload).digest()),
                                dtype=torch.uint8, device=device)
        if self._world_size > 1:
            identities = [torch.empty_like(identity) for _ in range(self._world_size)]
            torch.distributed.all_gather(identities, identity)
            if not all(torch.equal(value, identity) for value in identities):
                self._fail("runtime_rejected")

    def _reduce(self, torch, device, value, *, maximum=False):
        dtype = torch.int64 if type(value) is int else torch.float64
        aggregate = torch.tensor(value, dtype=dtype, device=device)
        if self._world_size > 1:
            operation = (torch.distributed.ReduceOp.MAX if maximum
                         else torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(aggregate, op=operation)
        return aggregate.item()

    def _all_healthy(self, torch, device, healthy, code):
        if self._reduce(torch, device, int(not healthy), maximum=True):
            self._fail(code)

    def _batches(self, torch, device, microbatches):
        values, healthy = [], True
        try:
            if type(microbatches) is not list or not microbatches:
                raise ValueError
            for batch in microbatches:
                if (type(batch) is not dict
                        or set(batch) != {"input_ids", "labels", "attention_mask"}):
                    raise ValueError
                tokens, labels, attention = (batch[key] for key in
                                             ("input_ids", "labels", "attention_mask"))
                if any(not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                       or value.dtype != torch.long or value.ndim != 2
                       for value in (tokens, labels, attention)):
                    raise ValueError
                if (tokens.shape != labels.shape or tokens.shape != attention.shape
                        or tokens.shape[0] < 1 or tokens.shape[1] < 2):
                    raise ValueError
                selected = labels.ne(-100)
                if (not bool(tokens.ge(0).all())
                        or not bool(((attention == 0) | (attention == 1)).all())
                        or not torch.equal(labels[selected], tokens[selected])
                        or bool((selected & attention.eq(0)).any())):
                    raise ValueError
                supervised = int(labels[:, 1:].ne(-100).sum())
                count = int(attention.sum())
                if supervised < 1 or count < 1:
                    raise ValueError
                values.append((batch, supervised, count))
        except BaseException:
            healthy = False
        self._all_healthy(torch, device, healthy, "batch_rejected")
        # Unequal numbers of DDP forwards/no-sync boundaries must never be used.
        largest = self._reduce(torch, device, len(values), maximum=True)
        total = self._reduce(torch, device, len(values))
        if total != largest * self._world_size:
            self._fail("batch_rejected")
        return values

    def perform(self, model, optimizer, microbatches, *, to_step, learning_rate,
                max_grad_norm, strategy="ddp", gradient_sync_mode="last_microbatch",
                precision="fp32"):
        self._check()
        if self._busy:
            self._fail("reentry")
        self._busy, self._stage = True, "reservation"
        try:
            with self._gate.step(to_step):
                self._check()
                if (not _positive(learning_rate) or not _positive(max_grad_norm)
                        or strategy not in {"ddp", "fsdp_full_shard"}
                        or gradient_sync_mode not in {"last_microbatch", "every_microbatch"}
                        or precision not in {"fp32", "bf16"}
                        or self._device_type == "cuda" and precision != "bf16"):
                    self._fail("invalid_arguments")
                self._stage = "runtime"
                torch, device = self._runtime(model, optimizer, strategy)
                self._stage = "settings"
                self._settings(torch, device, {
                    "to_step": to_step, "learning_rate": learning_rate,
                    "max_grad_norm": max_grad_norm, "strategy": strategy,
                    "gradient_sync_mode": gradient_sync_mode, "precision": precision,
                    "world_size": self._world_size, "device_type": self._device_type,
                })
                self._stage = "batches"
                batches = self._batches(torch, device, microbatches)
                global_supervised = self._reduce(torch, device, sum(item[1] for item in batches))
                global_tokens = self._reduce(torch, device, sum(item[2] for item in batches))
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                optimizer.zero_grad(set_to_none=True)
                local_numerator = 0.0
                self._stage = "forward"
                for number, (cpu_batch, supervised, _) in enumerate(batches):
                    batch = {key: value.to(device, non_blocking=device.type == "cuda")
                             for key, value in cpu_batch.items()}
                    defer = (gradient_sync_mode == "last_microbatch" and number < len(batches) - 1)
                    synchronization = (model.no_sync() if defer and self._world_size > 1
                                       else nullcontext())
                    with synchronization:
                        autocast = (torch.autocast(device.type, dtype=torch.bfloat16)
                                    if precision == "bf16" else nullcontext())
                        with autocast:
                            loss = model(**batch).loss
                        healthy = (isinstance(loss, torch.Tensor) and loss.numel() == 1
                                   and bool(torch.isfinite(loss)))
                        self._all_healthy(torch, device, healthy, "nonfinite_loss")
                        weight = self._world_size * supervised / global_supervised
                        (loss * weight).backward()
                    local_numerator += float(loss.detach()) * supervised
                    self._check()
                self._stage = "gradient"
                norm = (model.clip_grad_norm_(max_grad_norm) if strategy == "fsdp_full_shard"
                        else torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm))
                self._all_healthy(torch, device, bool(torch.isfinite(norm)), "nonfinite_gradient")
                self._stage = "optimizer"
                optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                self._check()
                self._stage = "metrics"
                result = {
                    "to_step": to_step,
                    "loss": self._reduce(torch, device, local_numerator) / global_supervised,
                    "gradient_norm": self._reduce(torch, device, float(norm), maximum=True),
                    "tokens": global_tokens,
                    "supervised_tokens": global_supervised,
                }
                self._stage = "completion"
            self._stage = "complete"
            return result
        except BaseException:
            self._fail("step_failed")
        finally:
            self._busy = False
