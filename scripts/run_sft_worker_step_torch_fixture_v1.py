#!/usr/bin/env python3
"""Generated Linux CPU DDP fixture, never a model/data/training launcher.

Only two spawned children import Torch, after explicit Linux/environment guards.
One parent owns the real Supervisor and EpochLedgerSession; no ledger is forked
or copied into workers. Private AF_UNIX channels carry bounded ledger frames.
Gloo uses the explicitly selected loopback interface, not external networking.
All third-party output is discarded. Only fixed stages/counts/booleans escape.
RSS observations are snapshots, not cgroup enforcement or peak-memory evidence.
"""

from __future__ import annotations

import copy
import datetime
import importlib
import json
import math
import multiprocessing
import os
import selectors
import socket
import sys
import tempfile
import time
import types
from collections import namedtuple
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_lumi_synthetic_worker_step_fixture_v1"
SCHEMA = "lumi.sft-worker-step-fixture.v0.1"
STAGE = "guard"
CHILD_STAGE = "not_reported"
WORKER_STAGE = "not_reported"
WORKER_CODE = "not_reported"
MAXIMUM_SECONDS = 100
MAXIMUM_WIRE_BYTES = 65_536
MAXIMUM_RECEIVES = 128
CHILD_STAGES = frozenset(("guard", "dependencies", "gloo", "models", "step", "checks", "done"))
FLAGS = (
    "real_cpu_ddp_used", "global_batch_model_equal_at_each_step",
    "global_batch_adam_equal_at_each_step", "weighted_loss_equal", "gradient_norm_equal",
    "aggregate_token_counts_equal", "imbalanced_supervision_verified",
    "both_gradient_sync_modes_verified", "forward_follows_global_reservation",
    "single_parent_supervisor_verified", "epoch_ack_confirmations_verified",
    "completion_fences_verified", "owned_processes_reaped", "rss_snapshot_observed",
    "loopback_gloo_used",
)
NONCLAIMS = (
    "real_model_or_data_access", "cuda_used", "external_network_authorized",
    "os_resource_enforcement_verified", "all_three_invocations_completed",
    "production_eligible", "real_sft_execution_permitted",
    "generated_reference_work_in_shared_ledger", "total_fixture_cost_metered",
)


def require(condition):
    if not condition:
        raise AssertionError("fixture_assertion")


def guards():
    require(sys.platform == "linux")
    require(os.environ.get("LUMI_WORKER_STEP_TORCH_CI_FIXTURE") == "1")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "")
    require(os.environ.get("GLOO_SOCKET_IFNAME") == "lo")


def module(name):
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(ROOT / "src/lumi")]
        sys.modules[PACKAGE] = package
    return importlib.import_module(f"{PACKAGE}.{name}")


def controller():
    counts = {
        "sequential_invocations": 3, "global_optimizer_steps": 4,
        "rank_optimizer_steps": 8, "logical_sample_events": 16,
        "maximum_unique_train_records": 8, "rank_local_input_loads": 6,
        "maximum_encoded_index_rows": 48, "maximum_selected_binary_bytes": 48 * 32 * 8,
        "maximum_selected_index_bytes": 48 * 176,
        "includes_checkpoint_hashing_or_final_digest": False,
        "is_total_io_or_cost_budget": False,
    }
    caps = {
        "maximum_combined_runtime_seconds": MAXIMUM_SECONDS, "maximum_cost_microusd": 10_000,
        "total_host_hourly_rate_microusd": 60_000,
        "maximum_host_ram_bytes": 2 * 1024**3, "maximum_per_gpu_memory_bytes": 1,
        "maximum_private_write_bytes": 4096, "maximum_checkpoint_read_bytes": 4096,
        "maximum_encoded_binary_read_bytes": counts["maximum_selected_binary_bytes"],
        "maximum_encoded_index_read_bytes": counts["maximum_selected_index_bytes"],
        "maximum_log_bytes": 1, "maximum_invocations": 3,
        "maximum_global_optimizer_steps": 4, "maximum_external_network_bytes": 0,
        "maximum_inference_calls": 0,
    }
    return module("posttraining_sft_supervisor_v1").Supervisor(caps, counts, world_size=2)


def batches(torch, rank, step):
    # Four records per global step; deliberately uneven rank/microbatch labels.
    geometry = (
        (((4, (2,)), (6, (1, 2, 4, 5))), ((5, (1, 4)), (6, (1, 2, 3, 4, 5)))),
        (((5, (1, 3)), (6, (1, 3, 4))), ((4, (2,)), (6, (1, 2, 3, 5)))),
    )[step - 1][rank]
    result = []
    for index, (valid, supervised) in enumerate(geometry):
        values = [(rank * 3 + index * 4 + position + step) % 11 for position in range(6)]
        values[valid:] = [0] * (6 - valid)
        labels = [-100] * 6
        for position in supervised:
            labels[position] = values[position]
        result.append({
            "input_ids": torch.tensor([values], dtype=torch.long),
            "labels": torch.tensor([labels], dtype=torch.long),
            "attention_mask": torch.tensor([[int(index < valid) for index in range(6)]],
                                           dtype=torch.long),
        })
    return result


def model_type(torch):
    output = namedtuple("SyntheticCausalOutput", ("loss", "logits"))

    class TinyCausalModel(torch.nn.Module):
        def __init__(self, forward_allowed):
            super().__init__()
            self.embedding = torch.nn.Embedding(11, 5, dtype=torch.float64)
            self.projection = torch.nn.Linear(5, 11, dtype=torch.float64)
            self.forward_allowed = forward_allowed
            self.forwards = 0

        def forward(self, input_ids, labels, attention_mask):
            require(self.forward_allowed())
            self.forwards += 1
            logits = self.projection(self.embedding(input_ids))
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, 11), labels[:, 1:].reshape(-1), ignore_index=-100,
            )
            return output(loss, logits)

    return TinyCausalModel


def close_tree(torch, left, right):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape
                and bool(torch.allclose(left, right, rtol=1e-9, atol=1e-11)))
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(close_tree(torch, left[key], right[key]) for key in left))
    if type(left) in (list, tuple):
        return (type(left) is type(right) and len(left) == len(right)
                and all(close_tree(torch, a, b) for a, b in zip(left, right, strict=True)))
    return type(left) is type(right) and left == right


def reference_step(torch, model, optimizer, step, learning_rate, max_grad_norm):
    """Independent single concatenated batch: never WorkerStep or microbatch weighting."""
    pieces = batches(torch, 0, step) + batches(torch, 1, step)
    combined = {name: torch.cat([piece[name] for piece in pieces]) for name in pieces[0]}
    supervised = int(combined["labels"][:, 1:].ne(-100).sum())
    tokens = int(combined["attention_mask"].sum())
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    optimizer.zero_grad(set_to_none=True)
    logits = model.projection(model.embedding(combined["input_ids"]))
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 11), combined["labels"][:, 1:].reshape(-1),
        ignore_index=-100, reduction="sum",
    ) / supervised
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return {"loss": float(loss.detach()), "gradient_norm": float(norm),
            "tokens": tokens, "supervised_tokens": supervised}


def send_child_result(result_pipe, result):
    raw = json.dumps(result, sort_keys=True, separators=(",", ":")).encode("ascii")
    require(len(raw) <= 1024)
    result_pipe.send_bytes(raw)


def rank_worker(rank, endpoint, result_pipe, epoch, token, rendezvous, deadline_ns):
    # Spawn reconstructs only this rank's channel; no supervisor object is passed.
    stage, client, torch, distributed, worker = "guard", None, None, None, None
    result = {"status": "failed", "stage": stage}
    try:
        guards()
        os.set_inheritable(endpoint.fileno(), False)
        stage = "dependencies"
        torch = importlib.import_module("torch")
        distributed = importlib.import_module("torch.distributed")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(not torch.cuda.is_initialized())
        client_type = module("posttraining_sft_worker_client_v1").WorkerLedgerClient

        def interrupt():
            try:
                endpoint.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        client = client_type(
            endpoint, epoch=epoch, rank=rank, token=token, interrupt=interrupt,
            maximum_wire_bytes=MAXIMUM_WIRE_BYTES, deadline_ns=deadline_ns,
            maximum_exchange_ns=20_000_000_000,
        )
        stage = "gloo"
        distributed.init_process_group(
            backend="gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2,
            timeout=datetime.timedelta(seconds=30),
        )
        allowed, counts = {"forward": False}, {"no_sync": 0, "fences": 0}

        def synchronize(phase, to_step):
            require(phase in {"reserved", "updated", "completed"})
            if phase == "completed":
                client.wait_step_commit(to_step)
                counts["fences"] += 1
            distributed.barrier()
            allowed["forward"] = phase == "reserved"

        gate = module("posttraining_sft_step_gate_v1").StepGate(
            rank=rank, world_size=2, reserve_step=client.reserve_step,
            complete_step=client.step_complete, synchronize=synchronize,
            interrupt=interrupt, from_step=0, stop_step=2,
        )
        worker = module("posttraining_sft_worker_step_v1").WorkerStep(
            gate=gate, world_size=2, device_type="cpu",
        )
        stage = "models"
        torch.manual_seed(161803)
        tiny = model_type(torch)
        actual = tiny(lambda: allowed["forward"])
        reference = tiny(lambda: True)
        reference.load_state_dict(copy.deepcopy(actual.state_dict()))
        model = torch.nn.parallel.DistributedDataParallel(actual)
        original_no_sync = model.no_sync

        def counted_no_sync():
            counts["no_sync"] += 1
            return original_no_sync()

        model.no_sync = counted_no_sync
        options = {"lr": 0.02, "betas": (0.8, 0.9), "weight_decay": 0.01}
        optimizer = torch.optim.Adam(model.parameters(), **options)
        reference_optimizer = torch.optim.Adam(reference.parameters(), **options)
        for step, learning_rate, sync_mode in (
            (1, 0.02, "last_microbatch"), (2, 0.007, "every_microbatch"),
        ):
            stage = "step"
            expected = reference_step(torch, reference, reference_optimizer,
                                      step, learning_rate, 0.2)
            observed = worker.perform(
                model, optimizer, batches(torch, rank, step), to_step=step,
                learning_rate=learning_rate, max_grad_norm=0.2, strategy="ddp",
                gradient_sync_mode=sync_mode, precision="fp32",
            )
            stage = "checks"
            require(set(observed) == {"loss", "gradient_norm", "tokens",
                                      "supervised_tokens", "to_step"})
            require(observed["to_step"] == step)
            require(close_tree(torch, actual.state_dict(), reference.state_dict()))
            require(close_tree(torch, optimizer.state_dict(), reference_optimizer.state_dict()))
            for name in ("loss", "gradient_norm"):
                require(math.isclose(observed[name], expected[name], rel_tol=1e-9, abs_tol=1e-11))
            for name in ("tokens", "supervised_tokens"):
                require(observed[name] == expected[name])
        require(actual.forwards == 4 and counts == {"no_sync": 1, "fences": 2})
        require(not torch.cuda.is_initialized())
        result = {"status": "passed", "stage": "done", "steps_checked": 2,
                  "reference_equal": True, "fences_checked": 2, "forwards_checked": 4}
    except BaseException:
        result = {"status": "failed", "stage": stage if stage in CHILD_STAGES else "guard"}
        if worker is not None:
            implementation = module("posttraining_sft_worker_step_v1")
            if worker.failure_stage in implementation.STAGES:
                result["worker_stage"] = worker.failure_stage
            if worker.failure_code in implementation.ERROR_CODES:
                result["worker_code"] = worker.failure_code
    finally:
        # Deliver only the fixed failure stage before cleanup can block on Gloo.
        sent = result["status"] == "failed"
        if sent:
            send_child_result(result_pipe, result)
        try:
            if client is not None:
                client.close()
            else:
                endpoint.close()
            if distributed is not None and distributed.is_initialized():
                distributed.destroy_process_group()
        except BaseException:
            result = {"status": "failed", "stage": "done"}
        try:
            if not sent:
                send_child_result(result_pipe, result)
        finally:
            result_pipe.close()


def rss_snapshot(pids):
    total = 0
    for pid in pids:
        require(type(pid) is int and pid > 0)
        with open(f"/proc/{pid}/statm", "rb") as source:
            data = source.read(257)
        require(0 < len(data) <= 256)
        fields = data.split()
        require(len(fields) >= 2 and fields[1].isdigit())
        total += int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    require(total > 0)
    return total


def serve(session, ledger, channels, processes, deadline_ns, epoch):
    global STAGE
    protocol = module("posttraining_sft_epoch_protocol_v1")
    fence_schema = module("posttraining_sft_worker_client_v1").STEP_FENCE_SCHEMA
    receives, wire, fence_step, fence_count = 0, 0, 0, 0
    with selectors.DefaultSelector() as selector:
        for rank, channel in enumerate(channels):
            channel.settimeout(1)
            selector.register(channel, selectors.EVENT_READ, rank)
        while selector.get_map():
            require(time.monotonic_ns() < deadline_ns)
            for key, _ in selector.select(timeout=0.1):
                rank, channel = key.data, key.fileobj
                require(receives < MAXIMUM_RECEIVES)
                receives += 1
                require(wire < MAXIMUM_WIRE_BYTES)
                chunk = channel.recv(min(protocol.MAXIMUM_FRAME_BYTES + 4,
                                         MAXIMUM_WIRE_BYTES - wire))
                wire += len(chunk)
                require(wire <= MAXIMUM_WIRE_BYTES)
                if not chunk:
                    session.disconnect(rank)
                    selector.unregister(channel)
                    continue
                for ack in session.feed(rank, chunk):
                    wire += len(ack.frame)
                    require(wire <= MAXIMUM_WIRE_BYTES)
                    channel.sendall(ack.frame)
                    session.mark_ack_sent(ack)
                observed = session.summary()
                completed = observed["global_optimizer_steps_completed"]
                if completed > fence_step:
                    require(completed == fence_step + 1)
                    require(observed["pending_acknowledgements"] == 0)
                    require(observed["rank_step_confirmations_pending"] == 0)
                    require(ledger.summary()["global_optimizer_steps_completed"] == completed)
                    if completed == 2:
                        STAGE = "memory"
                        rss = rss_snapshot([os.getpid()] + [p.pid for p in processes])
                        session.observe_memory(rss, [0, 0])
                        session.finish_invocation(2)
                    for peer_rank, peer in enumerate(channels):
                        frame = protocol.encode_frame({
                            "schema_version": fence_schema, "epoch": epoch,
                            "rank": peer_rank, "to_step": completed, "status": "committed",
                        })
                        wire += len(frame)
                        require(wire <= MAXIMUM_WIRE_BYTES)
                        peer.sendall(frame)
                        fence_count += 1
                    fence_step = completed
                    STAGE = "ledger"
            require(all(process.exitcode in (None, 0) for process in processes))
    require(fence_step == 2 and fence_count == 4)
    return fence_count


def run():
    global STAGE, CHILD_STAGE, WORKER_STAGE, WORKER_CODE
    guards()
    require(sys.argv[1:] == ["--synthetic-only"])
    os.umask(0o077)
    deadline_ns = time.monotonic_ns() + MAXIMUM_SECONDS * 1_000_000_000
    STAGE = "setup"
    ledger = controller()
    protocol = module("posttraining_sft_epoch_protocol_v1")
    epoch, tokens = protocol.create_epoch(), protocol.create_rank_tokens(2)
    session = protocol.EpochLedgerSession(ledger, world_size=2, epoch=epoch, rank_tokens=tokens)
    session.begin_invocation("reference", 0, 2, False)
    channels, children, processes, result_pipes = [], [], [], []
    context = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="lumi-SYNTHETIC-worker-step-v1-") as temporary:
        parent = Path(temporary)
        parent.chmod(0o700)
        try:
            STAGE = "children"
            for rank in range(2):
                left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
                channels.append(left)
                children.append(right)
                receiver, sender = context.Pipe(duplex=False)
                result_pipes.append(receiver)
                process = context.Process(target=rank_worker, args=(
                    rank, right, sender, epoch, tokens[rank],
                    str(parent / "gloo-store"), deadline_ns,
                ))
                process.start()
                processes.append(process)
                right.close()
                sender.close()
            STAGE = "ledger"
            fences = serve(session, ledger, channels, processes, deadline_ns, epoch)
            STAGE = "child_results"
            for process, receiver in zip(processes, result_pipes, strict=True):
                remaining = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
                require(remaining > 0 and receiver.poll(min(remaining, 10)))
                report = json.loads(receiver.recv_bytes(maxlength=1024))
                CHILD_STAGE = report.get("stage", "not_reported")
                require(CHILD_STAGE in CHILD_STAGES)
                require(report == {"status": "passed", "stage": "done", "steps_checked": 2,
                                   "reference_equal": True, "fences_checked": 2,
                                   "forwards_checked": 4})
                process.join(timeout=5)
                require(not process.is_alive() and process.exitcode == 0)
            STAGE = "verify"
            summary, epoch_summary = ledger.summary(), session.summary()
            require(epoch_summary["state"] == "completed")
            require(epoch_summary["requests_predebited"] == 6)
            require(epoch_summary["acknowledgements_sent"] == 6)
            require(epoch_summary["acknowledgements_confirmed"] == 6)
            require(epoch_summary["pending_acknowledgements"] == 0)
            require(summary["global_optimizer_steps_reserved"] == 2)
            require(summary["global_optimizer_steps_completed"] == 2)
            require(summary["logical_sample_events_reserved"] == 8)
            require(summary["invocations_completed"] == 1)
            require(summary["memory_observations"] == 1)
            require(summary["state"] == "ready" and summary["failure_code"] is None)
            require(not any(summary["resources_reserved"].values()))
            STAGE = "complete"
            return {"schema_version": SCHEMA, "status": "passed",
                    **dict.fromkeys(FLAGS, True), **dict.fromkeys(NONCLAIMS, False),
                    "rank_workers_verified": 2, "global_optimizer_steps_reserved": 2,
                    "global_optimizer_steps_completed": 2, "logical_sample_events_reserved": 8,
                    "invocations_completed": 1, "commit_fences_sent": fences,
                    "controlled_rank_optimizer_steps": 4, "reference_optimizer_steps": 4}
        except BaseException:
            # No exception text, tracebacks, tokens, identities or tensor values.
            implementation = module("posttraining_sft_worker_step_v1")
            for receiver in result_pipes:
                try:
                    if not receiver.poll(0.5):
                        continue
                    observed = json.loads(receiver.recv_bytes(maxlength=1024))
                    if observed.get("status") == "failed":
                        if observed.get("stage") in CHILD_STAGES:
                            CHILD_STAGE = observed["stage"]
                        if observed.get("worker_stage") in implementation.STAGES:
                            WORKER_STAGE = observed["worker_stage"]
                        if observed.get("worker_code") in implementation.ERROR_CODES:
                            WORKER_CODE = observed["worker_code"]
                except BaseException:
                    pass
            raise
        finally:
            for channel in channels + children:
                channel.close()
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
                require(not process.is_alive())
                process.close()
            for receiver in result_pipes:
                receiver.close()


def main():
    # Preserve a single bounded report channel; suppress all native/Torch output.
    output_fd = os.dup(1)
    with open(os.devnull, "wb") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    try:
        report = run()
        code = 0
    except BaseException:
        report = {"schema_version": SCHEMA, "status": "failed", "stage": STAGE,
                  "child_stage": CHILD_STAGE if CHILD_STAGE in CHILD_STAGES else "not_reported",
                  "worker_stage": WORKER_STAGE, "worker_code": WORKER_CODE,
                  **dict.fromkeys(NONCLAIMS, False)}
        code = 1
    raw = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
    require(len(raw) <= 4096)
    require(os.write(output_fd, raw) == len(raw))
    os.close(output_fd)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
