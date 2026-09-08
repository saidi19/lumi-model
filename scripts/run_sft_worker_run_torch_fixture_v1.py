#!/usr/bin/env python3
"""Generated three-invocation, two-rank WorkerRun acceptance; remote Linux CI only.

The six fresh spawned workers are the only processes importing Torch. A single
parent-owned Supervisor v2 meters all three training invocations through fresh
Epoch v2 sessions, including input and checkpoint I/O. The first invocation is
the controlled reference, not an extra unaccounted optimizer oracle. Generated
fixture creation, private comparison IPC and comparison CPU work are outside the
ledger; this is not a total fixture cost budget. No real model/data is admitted.

All result payloads containing generated tensors/state stay in inherited private
pipes and memory. Model/AdamW/state comparisons use rtol 1e-9, atol 1e-11; RNG
comparisons are exact. Public output is a bounded fixed-code/count/boolean report.
The parent owns deadline/worker cleanup and RSS snapshots; neither is OS resource
enforcement. Native Gloo uses loopback/FileStore only for this synthetic harness.
Never execute this file, import Torch, or run its skipped wrapper on the Mac.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib
import json
import math
import multiprocessing
import os
import selectors
import socket
import stat
import struct
import sys
import tempfile
import time
import types
from collections import Counter, namedtuple
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_lumi_synthetic_worker_run_fixture_v1"
SCHEMA = "lumi.sft-worker-run-fixture.v0.1"
STAGE, CHILD_STAGE = "guard", "not_reported"
WORKER_STAGE, WORKER_CODE = "not_reported", "not_reported"
COMPONENT_FIELDS = {
    "checkpoint_stage": ("posttraining_sft_checkpoint_worker_v1", "STAGES"),
    "checkpoint_code": ("posttraining_sft_checkpoint_worker_v1", "ERROR_CODES"),
    "step_stage": ("posttraining_sft_worker_step_v1", "STAGES"),
    "step_code": ("posttraining_sft_worker_step_v1", "ERROR_CODES"),
    "input_code": ("posttraining_sft_worker_input_v1", "ERROR_CODES"),
    "gate_stage": ("posttraining_sft_step_gate_v1", "STAGES"),
    "gate_code": ("posttraining_sft_step_gate_v1", "ERROR_CODES"),
}
COMPONENT_DIAGNOSTICS = dict.fromkeys(COMPONENT_FIELDS, "not_reported")
MAXIMUM_SECONDS = 180
MAXIMUM_WIRE_BYTES = 16 * 1024 * 1024  # Per rank, per invocation; never reset in a client.
MAXIMUM_PARENT_WIRE_BYTES = 2 * MAXIMUM_WIRE_BYTES
MAXIMUM_RESULT_BYTES = 512 * 1024  # Each private rank snapshot, never public.
MAXIMUM_RECEIVES = 100_000
MAXIMUM_PHASES = 8
INDEX_ROW_BYTES = 158
RELEASE = b"synthetic-worker-run-retired"
CHILD_STAGES = frozenset(("guard", "dependencies", "gloo", "models", "run", "checks",
                          "result", "cleanup", "done"))
CONTROL_PHASES = frozenset(("inputs", "restored", "checkpoint"))
PROFILE = {
    "per_rank_batch_size": 1, "gradient_accumulation_steps": 2,
    "context_window": 32, "pad_to_multiple_of": 1, "pad_token_id": 0,
    "seed": 17, "learning_rate": 0.01, "lr_schedule": "constant", "warmup_steps": 0,
    "min_learning_rate_ratio": 0.1, "max_grad_norm": 1.0, "strategy": "ddp",
    "gradient_sync_mode": "last_microbatch", "precision": "fp32",
}
CHECKPOINT_LIMITS = {
    "maximum_file_bytes": 4 * 1024 * 1024, "maximum_read_bytes": 64 * 1024 * 1024,
    "maximum_read_call_bytes": 4 * 1024 * 1024,
    "maximum_write_call_bytes": 4 * 1024 * 1024, "maximum_sync_operations": 128,
}
FLAGS = (
    "real_cpu_ddp_used", "real_torch_dcp_used", "real_worker_run_used",
    "single_parent_supervisor_verified", "all_three_invocations_completed",
    "fresh_workers_and_epochs_verified", "all_optimizer_steps_in_shared_ledger",
    "six_prefix_loads_in_shared_ledger", "checkpoint_io_in_shared_ledger",
    "validation_metadata_present", "selected_prefix_only_read",
    "epoch_ack_confirmations_verified", "completion_fences_verified",
    "io_phase_confirmations_drained", "reference_model_resume_equal",
    "reference_adam_resume_equal", "reference_rng_resume_equal",
    "reference_training_state_resume_equal", "dropout_rng_consumption_verified",
    "private_modes_verified", "owned_processes_reaped", "rss_snapshot_observed",
    "loopback_gloo_used",
)
NONCLAIMS = (
    "real_model_or_data_access", "cuda_used", "external_network_authorized",
    "os_resource_enforcement_verified", "production_eligible", "real_sft_execution_permitted",
    "full_content_hashes_verified", "fixture_generation_in_shared_ledger",
    "private_comparison_work_in_shared_ledger", "total_fixture_cost_metered",
    "safe_unpickling_or_origin_authentication_verified",
)


def require(condition):
    if not condition:
        raise AssertionError("fixture_assertion")


def guards():
    require(sys.platform == "linux" and sys.dont_write_bytecode and sys.byteorder == "little")
    require(os.environ.get("LUMI_WORKER_RUN_TORCH_CI_FIXTURE") == "1")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "")
    require(os.environ.get("GLOO_SOCKET_IFNAME") == "lo")


def remaining_seconds(deadline_ns, maximum):
    remaining = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
    require(remaining > 0)
    return min(remaining, maximum)


def module(name):
    # Parent-side control modules have no ML package initialization side effects.
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(ROOT / "src/lumi")]
        sys.modules[PACKAGE] = package
    return importlib.import_module(f"{PACKAGE}.{name}")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def private_write(path, raw):
    with path.open("xb") as handle:
        require(handle.write(raw) == len(raw))
    path.chmod(0o600)


def synthetic_root(parent):
    """Three train rows in ONE shard: selecting two cannot fully hash that shard."""
    root = parent / "encoded"
    root.mkdir(mode=0o700)
    specifications = (("train", ((10, "a" * 64), (20, "b" * 64), (30, "c" * 64))),
                      ("validation", ((40, "d" * 64),)))
    shards, totals = [], {name: Counter() for name in ("train", "validation")}
    encoded_digest, sequence_digest = hashlib.sha256(), hashlib.sha256()
    for split, records in specifications:
        prefix = f"{split}-00000"
        tokens, labels, rows = [], [], bytearray()
        for ordinal, (base, record_hash) in enumerate(records):
            record_tokens, record_labels = [base, base + 1, base + 2], [-100, base + 1, base + 2]
            row = canonical({"offset_tokens": len(tokens), "record_id_sha256": record_hash,
                             "record_ordinal": ordinal, "supervised_tokens": 2,
                             "tokens": 3}) + b"\n"
            require(len(row) == INDEX_ROW_BYTES)
            rows.extend(row)
            encoded = canonical({"input_ids": record_tokens, "labels": record_labels,
                                 "record_id_sha256": record_hash})
            encoded_digest.update(len(encoded).to_bytes(8, "big"))
            encoded_digest.update(encoded)
            sequence_digest.update(f"{split}\0{record_hash}\n".encode())
            tokens.extend(record_tokens)
            labels.extend(record_labels)
        payloads = {f"{prefix}.tokens.u32le": struct.pack(f"<{len(tokens)}I", *tokens),
                    f"{prefix}.labels.i32le": struct.pack(f"<{len(labels)}i", *labels),
                    f"{prefix}.index.jsonl": bytes(rows)}
        files = []
        for name, raw in payloads.items():
            private_write(root / name, raw)
            files.append({"bytes": len(raw), "path": name,
                          "sha256": hashlib.sha256(raw).hexdigest()})
        counts = {"records": len(records), "tokens": len(tokens),
                  "supervised_tokens": 2 * len(records)}
        totals[split].update(counts)
        shards.append({"files": files, "index": 0, "split": split, **counts})
    manifest = {
        "authorization_boundary": {
            "candidate_or_source_files_opened": False, "encoded_shards_created": True,
            "gpu_inference_or_training_used": False, "input_or_predecessor_mutated": False,
            "network_or_remote_access_used": False, "production_dataset_seal": False,
            "production_eligible": False, "registry_or_contract_changed": False,
        },
        "context_window": 4096, "encoded_dataset_sha256": encoded_digest.hexdigest(),
        "format": {"index": "canonical-jsonl-record-hash-only-v0.1",
                   "labels": "little-endian-int32-ignore-index-minus-100",
                   "raw_text_retained": False, "tokens": "little-endian-uint32"},
        "maximum_tokens_per_shard": 4_194_304, "production_eligible": False,
        "record_hash_sequence_sha256": sequence_digest.hexdigest(),
        "schema_version": "lumi.sft-encoded-shards.v0.1", "seal_sha256": "e" * 64,
        "shards": shards, "splits": {name: dict(values) for name, values in totals.items()},
        **{key: sum(values[key] for values in totals.values())
           for key in ("records", "tokens", "supervised_tokens")},
    }
    manifest["manifest_sha256"] = digest(manifest)
    payload = canonical(manifest) + b"\n"
    private_write(root / "manifest.json", payload)
    inventory = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        info = path.lstat()
        inventory.append({
            "mode": f"{stat.S_IMODE(info.st_mode):04o}", "relative_path": path.name,
            "st_ctime_ns": info.st_ctime_ns, "st_dev": info.st_dev, "st_ino": info.st_ino,
            "st_mtime_ns": info.st_mtime_ns, "st_nlink": info.st_nlink,
            "st_size": info.st_size, "st_uid": info.st_uid,
        })
    bindings = {"expected_manifest_sha256": manifest["manifest_sha256"],
                "expected_manifest_file_sha256": hashlib.sha256(payload).hexdigest(),
                "expected_inventory_sha256": digest(shards),
                "expected_metadata_inventory_sha256": digest(inventory),
                "expected_seal_sha256": manifest["seal_sha256"]}
    return root, bindings, len(payload)


def controller():
    counts = {
        "sequential_invocations": 3, "global_optimizer_steps": 4,
        "rank_optimizer_steps": 8, "logical_sample_events": 16,
        "maximum_unique_train_records": 2, "rank_local_input_loads": 6,
        "maximum_encoded_index_rows": 12, "maximum_selected_binary_bytes": 12 * 32 * 8,
        "maximum_selected_index_bytes": 12 * 176,
        "includes_checkpoint_hashing_or_final_digest": False, "is_total_io_or_cost_budget": False,
    }
    caps = {
        "maximum_combined_runtime_seconds": MAXIMUM_SECONDS, "maximum_cost_microusd": 10_000,
        "total_host_hourly_rate_microusd": 60_000,
        "maximum_host_ram_bytes": 4 * 1024**3, "maximum_per_gpu_memory_bytes": 1,
        "maximum_private_write_bytes": 32 * 1024 * 1024,
        "maximum_checkpoint_read_bytes": 96 * 1024 * 1024,
        "maximum_encoded_binary_read_bytes": counts["maximum_selected_binary_bytes"],
        "maximum_encoded_index_read_bytes": counts["maximum_selected_index_bytes"],
        "maximum_encoded_metadata_read_bytes": 65_536,
        "maximum_log_bytes": 1, "maximum_invocations": 3,
        "maximum_global_optimizer_steps": 4, "maximum_external_network_bytes": 0,
        "maximum_inference_calls": 0,
    }
    return module("posttraining_sft_supervisor_v2").Supervisor(caps, counts, world_size=2)


def model_type(torch):
    output = namedtuple("GeneratedCausalOutput", ("loss", "logits"))

    class TinyCausalModel(torch.nn.Module):
        def __init__(self, allowed):
            super().__init__()
            self.embedding = torch.nn.Embedding(64, 5, dtype=torch.float64)
            self.dropout = torch.nn.Dropout(0.25)
            self.projection = torch.nn.Linear(5, 64, dtype=torch.float64)
            self.allowed, self.forwards, self.first_rng = allowed, 0, None

        def forward(self, input_ids, labels, attention_mask):
            require(self.allowed["forward"] and self.training)
            if self.first_rng is None:
                self.first_rng = torch.get_rng_state().clone()
            self.forwards += 1
            logits = self.projection(self.dropout(self.embedding(input_ids)))
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, 64), labels[:, 1:].reshape(-1), ignore_index=-100,
            )
            return output(loss, logits)

    return TinyCausalModel


def private_tree(torch, value):
    """Bounded generated-only structural snapshot, sent only to the owning parent."""
    if isinstance(value, torch.Tensor):
        require(value.device.type == "cpu" and value.numel() <= 8192)
        return {"kind": "tensor", "dtype": str(value.dtype), "shape": list(value.shape),
                "values": value.detach().reshape(-1).tolist()}
    if isinstance(value, dict):
        return {"kind": "mapping", "items": [[private_tree(torch, key), private_tree(torch, item)]
                                              for key, item in value.items()]}
    if type(value) in (list, tuple):
        return {"kind": type(value).__name__,
                "items": [private_tree(torch, item) for item in value]}
    require(value is None or type(value) in (str, int, float, bool))
    return value


def outcome(pipe, value):
    raw = canonical(value)
    require(len(raw) <= MAXIMUM_RESULT_BYTES)
    pipe.send_bytes(raw)


def component_failures(worker):
    """Only allowlisted first-failure codes/stages; never objects or exception text."""
    values = {}
    for prefix, attribute in (("checkpoint", "_checkpoint"), ("step", "_step"),
                               ("gate", "_gate")):
        component = getattr(worker, attribute, None)
        for suffix in ("stage", "code"):
            field = f"{prefix}_{suffix}"
            module_name, vocabulary = COMPONENT_FIELDS[field]
            allowed = getattr(importlib.import_module(f"lumi.{module_name}"), vocabulary)
            value = getattr(component, f"failure_{suffix}", None)
            if type(value) is str and value in allowed:
                values[field] = value
    value = getattr(getattr(worker, "_input", None), "_failure", None)
    input_module = importlib.import_module("lumi.posttraining_sft_worker_input_v1")
    if type(value) is str and value in input_module.ERROR_CODES:
        values["input_code"] = value
    return values


def record_failure(observed):
    global CHILD_STAGE, WORKER_STAGE, WORKER_CODE
    if type(observed) is not dict or observed.get("status") != "failed":
        return
    if observed.get("stage") in CHILD_STAGES:
        CHILD_STAGE = observed["stage"]
    implementation = module("posttraining_sft_worker_run_v1")
    if observed.get("failure_stage") in implementation.STAGES and WORKER_STAGE == "not_reported":
        WORKER_STAGE = observed["failure_stage"]
    if observed.get("failure_code") in implementation.ERROR_CODES and WORKER_CODE == "not_reported":
        WORKER_CODE = observed["failure_code"]
    for field, (module_name, vocabulary) in COMPONENT_FIELDS.items():
        value = observed.get(field)
        if (type(value) is str and value in getattr(module(module_name), vocabulary)
                and COMPONENT_DIAGNOSTICS[field] == "not_reported"):
            COMPONENT_DIAGNOSTICS[field] = value


def rank_worker(rank, endpoint, result_pipe, epoch, token, rendezvous, deadline_ns,
                encoded_root, bindings, output_root, from_step, stop_step, resume_receipt):
    stage, client, distributed, worker, parent_fd = "guard", None, None, None, None
    try:
        guards()
        os.set_inheritable(endpoint.fileno(), False)
        stage = "dependencies"
        sys.path.insert(0, str(ROOT / "src"))
        torch = importlib.import_module("torch")
        distributed = importlib.import_module("torch.distributed")
        importlib.import_module("lumi.posttraining_sft_runtime_v4")
        importlib.import_module("lumi.distributed_training")
        importlib.import_module("lumi.pilot_training")
        implementation = importlib.import_module("lumi.posttraining_sft_worker_run_v1")
        client_type = importlib.import_module("lumi.posttraining_sft_worker_client_v2")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(not torch.cuda.is_initialized())

        def interrupt():
            try:
                endpoint.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        client = client_type.WorkerLedgerClient(
            endpoint, epoch=epoch, rank=rank, token=token, interrupt=interrupt,
            maximum_wire_bytes=MAXIMUM_WIRE_BYTES, deadline_ns=deadline_ns,
            maximum_exchange_ns=30_000_000_000,
        )
        stage = "gloo"
        distributed.init_process_group(
            backend="gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2,
            timeout=datetime.timedelta(seconds=30),
        )
        allowed, observed = {"forward": False}, {"fences": 0, "no_sync": 0, "phases": 0}

        def synchronize(phase, to_step):
            require(phase in CONTROL_PHASES | {"reserved", "updated", "completed"})
            require(type(to_step) is int and from_step <= to_step <= stop_step)
            if phase in CONTROL_PHASES:
                observed["phases"] += 1
                require(observed["phases"] <= MAXIMUM_PHASES)
                outcome(result_pipe, {"status": "phase", "phase": phase, "step": to_step})
                require(result_pipe.poll(remaining_seconds(deadline_ns, 30)))
                released = json.loads(result_pipe.recv_bytes(maxlength=1024))
                require(released == {"status": "phase_committed", "phase": phase, "step": to_step})
            if phase == "completed":
                # WorkerRun already consumed the exact parent commit fence.
                observed["fences"] += 1
            distributed.barrier()
            allowed["forward"] = phase == "reserved"

        def admit():
            # Generated fixture authority only; old real-data/GPU admission stays closed.
            guards()
            require(from_step == 0 or resume_receipt is not None)

        stage = "models"
        torch.manual_seed(PROFILE["seed"])
        actual = model_type(torch)(allowed)
        model = torch.nn.parallel.DistributedDataParallel(actual)
        original_no_sync = model.no_sync

        def counted_no_sync():
            observed["no_sync"] += 1
            return original_no_sync()

        model.no_sync = counted_no_sync
        optimizer = torch.optim.AdamW(model.parameters(), lr=PROFILE["learning_rate"],
                                      betas=(0.8, 0.9), weight_decay=0.01)
        require(type(optimizer) is torch.optim.AdamW and not optimizer.state)
        parent_fd = os.open(output_root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        worker = implementation.WorkerRun(
            client=client, rank=rank, world_size=2, from_step=from_step, stop_step=stop_step,
            profile=dict(PROFILE), record_limit=2, loader_bindings=bindings,
            source_identity_sha256="f" * 64, checkpoint_limits=dict(CHECKPOINT_LIMITS),
            admit=admit, synchronize=synchronize, interrupt=interrupt, device_type="cpu",
        )
        stage = "run"
        result = worker.run(
            model, optimizer, encoded_root=encoded_root, output_parent_fd=parent_fd,
            resume_checkpoint_name=(None if resume_receipt is None
                                    else resume_receipt["checkpoint_name"]),
            resume_manifest_sha256=(None if resume_receipt is None
                                    else resume_receipt["manifest_sha256"]),
        )
        stage = "checks"
        require(set(result) == {"training_state", "checkpoint"})
        require(result["checkpoint"]["checkpoint_name"] == f"checkpoint-{stop_step:08d}")
        require(result["checkpoint"]["published"] is True)
        require(result["checkpoint"]["production_eligible"] is False)
        require(actual.forwards == 2 * (stop_step - from_step))
        require(observed["no_sync"] == observed["fences"] == stop_step - from_step)
        require(actual.first_rng is not None)
        require(not torch.equal(actual.first_rng, torch.get_rng_state()))
        require(not torch.cuda.is_initialized())
        state = result["training_state"]
        require(state["step"] == stop_step and state["sequence_cursor"] == 4 * stop_step)
        require(state["records_seen"] == 4 * stop_step)
        require(state["tokens_seen"] == 12 * stop_step)
        require(state["supervised_tokens_seen"] == 8 * stop_step)
        worker_summary = worker.summary()
        require(worker_summary["state"] == "complete")
        require(worker_summary["completed_steps"] == worker_summary["saved_checkpoints"]
                == stop_step - from_step)
        require(worker_summary["restored_checkpoints"] == int(from_step == 1))
        for name in ("os_admission_verified", "os_resource_enforcement",
                     "real_sft_execution_permitted", "production_eligible"):
            require(worker_summary[name] is False)
        client_summary = client.summary()
        require(client_summary["requests_sent"] == client_summary["acknowledgements_validated"]
                == client_summary["confirmations_sent"])
        stage = "result"
        outcome(result_pipe, {
            "status": "passed", "stage": "done", "steps": stop_step - from_step,
            "fences": observed["fences"], "phases": observed["phases"],
            "requests": client_summary["requests_sent"], "training_state": state,
            "checkpoint": result["checkpoint"],
            "model": private_tree(torch, actual.state_dict()),
            "optimizer": private_tree(torch, optimizer.state_dict()),
            "rng": private_tree(torch, torch.get_rng_state()),
        })
        # The last result can race its final ledger ACK confirmation. Keep every
        # channel and worker alive until the parent has drained and sealed it.
        stage = "cleanup"
        require(result_pipe.poll(remaining_seconds(deadline_ns, 30)))
        require(result_pipe.recv_bytes(maxlength=128) == RELEASE)
        client.close()
        distributed.destroy_process_group()
        outcome(result_pipe, {"status": "closed", "stage": "done"})
    except BaseException:
        report = {"status": "failed", "stage": stage if stage in CHILD_STAGES else "guard"}
        if worker is not None:
            worker_summary = worker.summary()
            for field, allowed_values in (("failure_stage", implementation.STAGES),
                                           ("failure_code", implementation.ERROR_CODES)):
                value = worker_summary.get(field)
                if value in allowed_values:
                    report[field] = value
            report.update(component_failures(worker))
        try:
            outcome(result_pipe, report)
        except BaseException:
            pass
    finally:
        if parent_fd is not None:
            os.close(parent_fd)
        try:
            client.close() if client is not None else endpoint.close()
        except BaseException:
            pass
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


def serve(session, ledger, channels, results, processes, deadline_ns, epoch, from_step, stop_step):
    global CHILD_STAGE
    protocol = module("posttraining_sft_epoch_protocol_v2")
    fence_schema = module("posttraining_sft_worker_client_v2").STEP_FENCE_SCHEMA
    phases, reports = {}, {}
    receives, wire, fenced, phase_count = 0, 0, 0, 0
    result_wire, result_messages = dict.fromkeys((0, 1), 0), dict.fromkeys((0, 1), 0)
    before_steps = ledger.summary()["global_optimizer_steps_completed"]
    with selectors.DefaultSelector() as selector:
        for rank, channel in enumerate(channels):
            channel.settimeout(1)
            selector.register(channel, selectors.EVENT_READ, ("ledger", rank))
            selector.register(results[rank], selectors.EVENT_READ, ("result", rank))
        while True:
            require(time.monotonic_ns() < deadline_ns)
            ledger.checkpoint()
            for key, _ in selector.select(timeout=0.1):
                kind, rank = key.data
                if kind == "result":
                    raw = key.fileobj.recv_bytes(maxlength=MAXIMUM_RESULT_BYTES)
                    result_wire[rank] += len(raw)
                    result_messages[rank] += 1
                    require(result_wire[rank] <= MAXIMUM_RESULT_BYTES + MAXIMUM_PHASES * 1024)
                    require(result_messages[rank] <= MAXIMUM_PHASES + 1)
                    observed = json.loads(raw)
                    require(type(observed) is dict)
                    if observed.get("status") == "phase":
                        require(set(observed) == {"status", "phase", "step"})
                        require(observed["phase"] in CONTROL_PHASES and rank not in phases)
                        require(type(observed["step"]) is int
                                and from_step <= observed["step"] <= stop_step)
                        phases[rank] = observed
                    else:
                        CHILD_STAGE = observed.get("stage", "not_reported")
                        record_failure(observed)
                        require(CHILD_STAGE == "done" and observed.get("status") == "passed")
                        require(rank not in reports)
                        reports[rank] = observed
                        selector.unregister(key.fileobj)
                    continue
                require(receives < MAXIMUM_RECEIVES and wire < MAXIMUM_PARENT_WIRE_BYTES)
                receives += 1
                chunk = key.fileobj.recv(min(protocol.MAXIMUM_FRAME_BYTES + 4,
                                             MAXIMUM_PARENT_WIRE_BYTES - wire))
                wire += len(chunk)
                if not chunk:
                    session.disconnect(rank)
                    raise AssertionError("fixture_early_eof")
                for ack in session.feed(rank, chunk):
                    wire += len(ack.frame)
                    require(wire <= MAXIMUM_PARENT_WIRE_BYTES)
                    key.fileobj.settimeout(remaining_seconds(deadline_ns, 1))
                    key.fileobj.sendall(ack.frame)
                    session.mark_ack_sent(ack)
            snapshot = session.summary()
            completed = snapshot["global_optimizer_steps_completed"]
            if completed > fenced:
                require(completed == fenced + 1 and not snapshot["pending_acknowledgements"])
                require(not snapshot["rank_step_confirmations_pending"])
                require(ledger.summary()["global_optimizer_steps_completed"]
                        == before_steps + completed)
                for rank, channel in enumerate(channels):
                    frame = protocol.encode_frame({
                        "schema_version": fence_schema, "epoch": epoch, "rank": rank,
                        "to_step": from_step + completed, "status": "committed",
                    })
                    wire += len(frame)
                    require(wire <= MAXIMUM_PARENT_WIRE_BYTES)
                    channel.sendall(frame)
                fenced = completed
            if len(phases) == 2 and not snapshot["pending_acknowledgements"]:
                require(phases[0] == phases[1] and not snapshot["rank_step_confirmations_pending"])
                phase_count += 1
                require(phase_count <= MAXIMUM_PHASES)
                message = {**phases[0], "status": "phase_committed"}
                for pipe in results:
                    outcome(pipe, message)
                phases.clear()
            if len(reports) == 2 and not snapshot["pending_acknowledgements"]:
                require(not phases and fenced == stop_step - from_step)
                require(not snapshot["rank_step_confirmations_pending"])
                requests = sum(report["requests"] for report in reports.values())
                require(snapshot["requests_predebited"] == snapshot["acknowledgements_sent"]
                        == snapshot["acknowledgements_confirmed"] == requests)
                require(all(process.is_alive() for process in processes))
                session.observe_memory(rss_snapshot([os.getpid()] + [p.pid for p in processes]),
                                       [0, 0])
                # Checkpoint reservations are complete only here, NOT at the last step fence.
                session.finish_invocation(stop_step)
                require(session.summary()["state"] == "completed")
                return reports, {"wire": wire, "fences": 2 * fenced,
                                 "requests": requests, "phases": phase_count}
            require(all(process.exitcode in (None, 0) for process in processes))


def run_invocation(context, ledger, parent, encoded_root, bindings, deadline_ns,
                   sequence_number, invocation, resume_receipt, used_epochs, used_tokens):
    global STAGE
    run_name, from_step, stop_step, resume = invocation
    protocol = module("posttraining_sft_epoch_protocol_v2")
    epoch, tokens = protocol.create_epoch(), protocol.create_rank_tokens(2)
    require(epoch not in used_epochs and epoch not in used_tokens)
    require(not set(tokens.values()) & (used_epochs | used_tokens | {epoch}))
    used_epochs.add(epoch)
    used_tokens.update(tokens.values())
    session = protocol.EpochLedgerSession(ledger, world_size=2, epoch=epoch, rank_tokens=tokens)
    session.begin_invocation(run_name, from_step, stop_step, resume)
    channels, child_channels, results, child_results, processes = [], [], [], [], []
    try:
        STAGE = "workers"
        for rank in range(2):
            left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
            channels.append(left)
            child_channels.append(right)
            receiver, sender = context.Pipe(duplex=True)
            results.append(receiver)
            child_results.append(sender)
            process = context.Process(target=rank_worker, args=(
                rank, right, sender, epoch, tokens[rank],
                str(parent / f"gloo-store-{sequence_number}"), deadline_ns,
                str(encoded_root), bindings, str(parent / run_name), from_step, stop_step,
                resume_receipt,
            ))
            processes.append(process)
            process.start()
            right.close()
            sender.close()
        STAGE = "ledger"
        reports, totals = serve(session, ledger, channels, results, processes, deadline_ns,
                                epoch, from_step, stop_step)
        STAGE = "cleanup"
        for pipe in results:
            pipe.send_bytes(RELEASE)
        for pipe, process in zip(results, processes, strict=True):
            require(pipe.poll(remaining_seconds(deadline_ns, 10)))
            require(json.loads(pipe.recv_bytes(maxlength=1024))
                    == {"status": "closed", "stage": "done"})
            process.join(timeout=remaining_seconds(deadline_ns, 5))
            require(not process.is_alive() and process.exitcode == 0)
        for rank, channel in enumerate(channels):
            require(channel.recv(1) == b"")
            session.disconnect(rank)
        return reports, totals
    except BaseException:
        # A failed worker's socket EOF can outrun its separate failure-report pipe.
        # One bounded collection pass preserves fixed diagnostics, never retries work.
        for pipe in results:
            try:
                if pipe.poll(0.2):
                    record_failure(json.loads(pipe.recv_bytes(maxlength=MAXIMUM_RESULT_BYTES)))
            except BaseException:
                pass
        raise
    finally:
        for channel in channels + child_channels:
            channel.close()
        # Terminate all siblings before joining any; escalation targets owned processes only.
        for process in processes:
            if process.pid is not None and process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join(timeout=2)
        for process in processes:
            if process.pid is not None and process.is_alive():
                process.kill()
        cleanup_failed = False
        for process in processes:
            if process.pid is not None:
                process.join(timeout=2)
            if process.pid is not None and process.is_alive():
                cleanup_failed = True
            else:
                process.close()
        for pipe in results + child_results:
            pipe.close()
        require(not cleanup_failed)


def close_tree(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is float:
        return math.isfinite(left) and math.isfinite(right) and math.isclose(
            left, right, rel_tol=1e-9, abs_tol=1e-11)
    if type(left) is dict:
        return (left.keys() == right.keys()
                and all(close_tree(left[key], right[key]) for key in left))
    if type(left) is list:
        return (len(left) == len(right)
                and all(close_tree(a, b) for a, b in zip(left, right, strict=True)))
    return left == right


def verify_private_modes(parent):
    for path in parent.rglob("*"):
        info = path.lstat()
        require(info.st_uid == os.geteuid() and not stat.S_ISLNK(info.st_mode))
        if stat.S_ISDIR(info.st_mode):
            require(stat.S_IMODE(info.st_mode) == 0o700)
        else:
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1)
            require(stat.S_IMODE(info.st_mode) == 0o600)


def run():
    global STAGE
    guards()
    require(sys.argv[1:] == ["--synthetic-only"])
    os.umask(0o077)
    deadline_ns = time.monotonic_ns() + MAXIMUM_SECONDS * 1_000_000_000
    context = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="lumi-SYNTHETIC-worker-run-v1-") as temporary:
        STAGE = "fixture"
        parent = Path(temporary)
        parent.chmod(0o700)
        encoded_root, bindings, manifest_bytes = synthetic_root(parent)
        for name in ("reference", "resumed"):
            (parent / name).mkdir(mode=0o700)
        ledger, used_epochs, used_tokens = controller(), set(), set()
        reports, totals, receipt = [], [], None
        sequence = module("posttraining_sft_supervisor_v2").SEQUENCE
        for number, invocation in enumerate(sequence):
            current, counters = run_invocation(
                context, ledger, parent, encoded_root, bindings, deadline_ns,
                number, invocation, receipt if number == 2 else None, used_epochs, used_tokens,
            )
            reports.append(current)
            totals.append(counters)
            if number == 1:
                require(current[0]["checkpoint"] == current[1]["checkpoint"])
                receipt = current[0]["checkpoint"]
            summary = ledger.summary()
            require(summary["invocations_completed"] == number + 1)
            require(summary["global_optimizer_steps_completed"] == (2, 3, 4)[number])
        STAGE = "comparison"
        for rank in range(2):
            reference, resumed = reports[0][rank], reports[2][rank]
            for name in ("model", "optimizer", "training_state"):
                require(close_tree(reference[name], resumed[name]))
            require(reference["rng"] == resumed["rng"])
        for current in reports:
            for name in ("model", "optimizer", "training_state", "checkpoint"):
                require(close_tree(current[0][name], current[1][name]))
            require(current[0]["rng"] != current[1]["rng"])
        STAGE = "accounting"
        summary = ledger.summary()
        resources = summary["resources_reserved"]
        require(summary["state"] == "completed" and summary["failure_code"] is None)
        require(summary["invocations_started"] == summary["invocations_completed"] == 3)
        require(summary["global_optimizer_steps_reserved"]
                == summary["global_optimizer_steps_completed"] == 4)
        require(summary["logical_sample_events_reserved"] == 16)
        require(summary["memory_observations"] == 3)
        require(resources["encoded_metadata_read_bytes"] == 6 * manifest_bytes)
        require(resources["encoded_binary_read_bytes"] == 288)
        require(resources["encoded_index_read_bytes"] == 12 * INDEX_ROW_BYTES)
        require(resources["encoded_index_rows"] == 12 and resources["log_bytes"] == 0)
        require(0 < resources["checkpoint_read_bytes"] <= 96 * 1024 * 1024)
        require(0 < resources["private_write_bytes"] <= 32 * 1024 * 1024)
        require(len(used_epochs) == 3 and len(used_tokens) == 6)
        require(sum(item["fences"] for item in totals) == 8)
        require(sum(report["steps"] for current in reports for report in current.values()) == 8)
        require({path.name for path in (parent / "reference").iterdir()}
                == {"checkpoint-00000001", "checkpoint-00000002"})
        require({path.name for path in (parent / "resumed").iterdir()}
                == {"checkpoint-00000001", "checkpoint-00000002"})
        verify_private_modes(parent)
        require("torch" not in sys.modules)
        STAGE = "complete"
        return {
            "schema_version": SCHEMA, "status": "passed",
            **dict.fromkeys(FLAGS, True), **dict.fromkeys(NONCLAIMS, False),
            "rank_workers_verified": 6, "controlled_rank_optimizer_steps": 8,
            "global_optimizer_steps_reserved": 4, "global_optimizer_steps_completed": 4,
            "logical_sample_events_reserved": 16, "invocations_started": 3,
            "invocations_completed": 3, "controlled_loads": 6, "records_loaded": 12,
            "encoded_index_rows": 12, "binary_bytes_read": 288,
            "index_bytes_read": 12 * INDEX_ROW_BYTES, "metadata_bytes_read": 6 * manifest_bytes,
            "checkpoints_published": 4, "rank_checkpoint_restores": 2,
            "commit_fences_sent": 8, "memory_observations": 3,
            "ledger_requests": sum(item["requests"] for item in totals),
            "parent_wire_bytes": sum(item["wire"] for item in totals),
            "checkpoint_read_bytes_reserved": resources["checkpoint_read_bytes"],
            "private_write_bytes_reserved": resources["private_write_bytes"],
        }


def main():
    # Suppress all native/Torch output and exceptions before importing any ML library.
    output_fd = os.dup(1)
    with open(os.devnull, "wb") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    try:
        report, code = run(), 0
    except BaseException:
        report = {"schema_version": SCHEMA, "status": "failed", "stage": STAGE,
                  "child_stage": CHILD_STAGE if CHILD_STAGE in CHILD_STAGES else "not_reported",
                  "worker_stage": WORKER_STAGE, "worker_code": WORKER_CODE,
                  **COMPONENT_DIAGNOSTICS,
                  **dict.fromkeys(NONCLAIMS, False)}
        code = 1
    raw = canonical(report) + b"\n"
    require(len(raw) <= 4096 and os.write(output_fd, raw) == len(raw))
    os.close(output_fd)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
