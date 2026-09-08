#!/usr/bin/env python3
"""Two generated prefix loads over one parent ledger; Linux CI only.

No model, optimizer, Gloo, GPU, real dataset or external networking is used.
Only spawned children import the frozen Torch-backed runtime after explicit
Linux/environment guards. Fixture generation and two direct frozen-reference
loads are independent test-oracle work, outside the shared input ledger.
The parent consumes one invocation but deliberately interrupts after both
loads; no optimizer step or completed training invocation is claimed.
"""

from __future__ import annotations

import builtins
import hashlib
import importlib
import io
import json
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
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_lumi_synthetic_worker_input_fixture_v1"
SCHEMA = "lumi.sft-worker-input-fixture.v0.1"
STAGE = "guard"
CHILD_STAGE = "not_reported"
MAXIMUM_SECONDS = 120
MAXIMUM_WIRE_BYTES = 524_288
MAXIMUM_PARENT_WIRE_BYTES = MAXIMUM_WIRE_BYTES * 2
MAXIMUM_RECEIVES = 2048
INDEX_ROW_BYTES = 158
CHILD_STAGES = frozenset(("guard", "dependencies", "reference", "load", "checks", "cleanup"))
RESOURCES = ("encoded_metadata_read_bytes", "encoded_binary_read_bytes",
             "encoded_index_read_bytes", "encoded_index_rows")
FLAGS = (
    "two_process_shared_controller_verified", "genuine_encoded_dataset_verified",
    "frozen_loader_result_equal", "frozen_loader_globals_preserved", "every_pread_precharged",
    "manifest_metadata_category_verified", "selected_prefix_only_read", "no_index_readahead",
    "getitem_requires_no_disk_read", "all_epoch_acks_confirmed", "planned_interrupt_verified",
    "owned_processes_reaped", "exact_cpu_tensors_verified",
)
NONCLAIMS = (
    "real_model_or_data_access", "cuda_used", "external_network_authorized",
    "full_content_hashes_verified", "production_eligible", "real_input_execution_permitted",
    "os_resource_enforcement_verified", "reference_io_in_shared_ledger",
    "fixture_generation_in_shared_ledger", "total_fixture_cost_metered",
)
RELEASE = b"synthetic-input-fixture-cleanup"
NETWORK_EVENTS = frozenset((
    "socket.connect", "socket.bind", "socket.getaddrinfo", "socket.gethostbyaddr",
    "socket.gethostbyname", "socket.sendto", "socket.sendmsg",
))


def require(condition):
    if not condition:
        raise AssertionError("fixture_assertion")


def remaining_seconds(deadline_ns, maximum):
    remaining = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
    require(remaining > 0)
    return min(remaining, maximum)


def guards():
    require(sys.platform == "linux")
    require(sys.dont_write_bytecode and sys.byteorder == "little")
    require(os.environ.get("LUMI_WORKER_INPUT_TORCH_CI_FIXTURE") == "1")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "")


def deny_network(event, arguments):
    if event in NETWORK_EVENTS:
        raise AssertionError("fixture_network_forbidden")


def module(name):
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(ROOT / "src/lumi")]
        sys.modules[PACKAGE] = package
    return importlib.import_module(f"{PACKAGE}.{name}")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def private_write(path, raw):
    # These are only new fixture files inside the run's TemporaryDirectory.
    with path.open("xb") as handle:
        require(handle.write(raw) == len(raw))
    path.chmod(0o600)


def synthetic_root(parent):
    root = parent / "encoded"
    root.mkdir(mode=0o700)
    specifications = (("train", 0, ((10, "a" * 64), (20, "b" * 64))),
                      ("train", 1, ((30, "c" * 64),)),
                      ("validation", 0, ((40, "d" * 64),)))
    shards, totals = [], {name: Counter() for name in ("train", "validation")}
    encoded_digest, sequence_digest = hashlib.sha256(), hashlib.sha256()
    first_index_bytes = None
    for split, number, records in specifications:
        prefix = f"{split}-{number:05d}"
        tokens, labels, index_bytes = [], [], bytearray()
        for ordinal, (base, record_hash) in enumerate(records):
            record_tokens, record_labels = [base, base + 1, base + 2], [-100, base + 1, base + 2]
            index = canonical({"offset_tokens": len(tokens), "record_id_sha256": record_hash,
                               "record_ordinal": ordinal, "supervised_tokens": 2,
                               "tokens": 3}) + b"\n"
            index_bytes.extend(index)
            if split == "train" and number == 0 and ordinal == 0:
                first_index_bytes = len(index)
            encoded = canonical({"input_ids": record_tokens, "labels": record_labels,
                                 "record_id_sha256": record_hash})
            encoded_digest.update(len(encoded).to_bytes(8, "big"))
            encoded_digest.update(encoded)
            sequence_digest.update(f"{split}\0{record_hash}\n".encode())
            tokens.extend(record_tokens)
            labels.extend(record_labels)
        payloads = {f"{prefix}.tokens.u32le": struct.pack(f"<{len(tokens)}I", *tokens),
                    f"{prefix}.labels.i32le": struct.pack(f"<{len(labels)}i", *labels),
                    f"{prefix}.index.jsonl": bytes(index_bytes)}
        files = []
        for name, data in payloads.items():
            private_write(root / name, data)
            files.append({"bytes": len(data), "path": name,
                          "sha256": hashlib.sha256(data).hexdigest()})
        counts = {"records": len(records), "tokens": len(tokens),
                  "supervised_tokens": 2 * len(records)}
        totals[split].update(counts)
        shards.append({"files": files, "index": number, "split": split, **counts})
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
    require(first_index_bytes == INDEX_ROW_BYTES)
    return root, bindings, len(payload), first_index_bytes


def controller(metadata_bytes):
    counts = {
        "sequential_invocations": 3, "global_optimizer_steps": 4,
        "rank_optimizer_steps": 8, "logical_sample_events": 8,
        "maximum_unique_train_records": 2, "rank_local_input_loads": 6,
        "maximum_encoded_index_rows": 12, "maximum_selected_binary_bytes": 12 * 3 * 8,
        "maximum_selected_index_bytes": 12 * 176,
        "includes_checkpoint_hashing_or_final_digest": False, "is_total_io_or_cost_budget": False,
    }
    caps = {
        "maximum_combined_runtime_seconds": MAXIMUM_SECONDS, "maximum_cost_microusd": 10_000,
        "total_host_hourly_rate_microusd": 60_000,
        "maximum_host_ram_bytes": 2 * 1024**3, "maximum_per_gpu_memory_bytes": 1,
        "maximum_private_write_bytes": 4096, "maximum_checkpoint_read_bytes": 4096,
        "maximum_encoded_binary_read_bytes": counts["maximum_selected_binary_bytes"],
        "maximum_encoded_index_read_bytes": counts["maximum_selected_index_bytes"],
        "maximum_encoded_metadata_read_bytes": metadata_bytes,
        "maximum_log_bytes": 1, "maximum_invocations": 3,
        "maximum_global_optimizer_steps": 4, "maximum_external_network_bytes": 0,
        "maximum_inference_calls": 0,
    }
    return module("posttraining_sft_supervisor_v2").Supervisor(caps, counts, world_size=2)


def outcome(channel, value):
    raw = canonical(value)
    require(len(raw) <= 2048)
    channel.send_bytes(raw)


def loader_worker(rank, endpoint, result_pipe, epoch, token, root_name, bindings,
                  manifest_bytes, index_bytes, deadline_ns):
    stage, client, worker = "guard", None, None
    try:
        guards()
        # Spawn has already reconstructed the two inherited private endpoints.
        # No new connections, binding, DNS or datagrams are needed by this worker.
        sys.addaudithook(deny_network)
        os.set_inheritable(endpoint.fileno(), False)
        stage = "dependencies"
        sys.path.insert(0, str(ROOT / "src"))
        torch = importlib.import_module("torch")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(not torch.cuda.is_initialized())
        runtime = importlib.import_module("lumi.posttraining_sft_runtime_v4")
        implementation = importlib.import_module("lumi.posttraining_sft_worker_input_v1")
        client_type = importlib.import_module("lumi.posttraining_sft_worker_client_v2")
        root = Path(root_name)
        stage = "reference"
        reference = runtime.EncodedSFTDataset(root, split="train", record_limit=1, **bindings)
        expected = reference[0]
        original_initializer = runtime.EncodedSFTDataset.__init__
        original_getitem = runtime.EncodedSFTDataset.__getitem__
        original_loader = runtime._load_verified_sft_split
        original_open = runtime._open_secure_dataset_file
        original_finish = runtime._finish_secure_dataset_file

        def interrupt():
            try:
                endpoint.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        client = client_type.WorkerLedgerClient(
            endpoint, epoch=epoch, rank=rank, token=token, interrupt=interrupt,
            maximum_wire_bytes=MAXIMUM_WIRE_BYTES, deadline_ns=deadline_ns,
            maximum_exchange_ns=20_000_000_000,
        )
        prepaid, consumed = dict.fromkeys(RESOURCES, 0), dict.fromkeys(RESOURCES, 0)
        calls, content_opens = {"pread": 0}, Counter()
        approved = {
            "manifest.json": ("encoded_metadata_read_bytes", manifest_bytes),
            "train-00000.tokens.u32le": ("encoded_binary_read_bytes", 12),
            "train-00000.labels.i32le": ("encoded_binary_read_bytes", 12),
            "train-00000.index.jsonl": ("encoded_index_read_bytes", index_bytes),
        }
        identities = {}
        positions = dict.fromkeys(approved, 0)
        root_info = root.stat()
        root_identity = root_info.st_dev, root_info.st_ino
        for name, value in approved.items():
            info = (root / name).stat()
            identities[(info.st_dev, info.st_ino)] = name, value

        def reserve(channel_rank, **amounts):
            require(channel_rank == rank and set(amounts) <= set(RESOURCES))
            result = client.reserve(channel_rank, **amounts)
            require(result is None)
            for name, count in amounts.items():
                prepaid[name] += count

        original_pread, original_os_open = os.pread, os.open

        def no_disk(*args, **kwargs):
            raise AssertionError("fixture_unexpected_disk_access")

        def checked_open(path, flags, mode=0o777, *, dir_fd=None):
            require(type(flags) is int and flags & os.O_ACCMODE == os.O_RDONLY)
            if not flags & os.O_DIRECTORY:
                require(type(path) is str and path in approved and dir_fd is not None)
                directory = os.fstat(dir_fd)
                require((directory.st_dev, directory.st_ino) == root_identity)
                require(content_opens[path] == 0)
                content_opens[path] += 1
            return original_os_open(path, flags, mode, dir_fd=dir_fd)

        def checked_pread(fd, size, offset):
            info = os.fstat(fd)
            require((info.st_dev, info.st_ino) in identities)
            name, (resource, extent) = identities[(info.st_dev, info.st_ino)]
            require(type(size) is int and size > 0 and offset == positions[name])
            require(0 <= offset < offset + size <= extent)
            require(prepaid[resource] >= consumed[resource] + size)
            if resource == "encoded_index_read_bytes":
                require(prepaid["encoded_index_rows"] == 1 and size == 1)
            consumed[resource] += size
            positions[name] += size
            calls["pread"] += 1
            return original_pread(fd, size, offset)

        worker = implementation.WorkerInput(rank=rank, reserve=reserve, interrupt=interrupt)
        stage = "load"
        with ExitStack() as stack:
            stack.enter_context(patch.object(os, "pread", checked_pread))
            stack.enter_context(patch.object(os, "open", checked_open))
            for target, names in ((builtins, ("open",)), (io, ("open",)),
                                  (os, ("read", "readv", "preadv", "fdopen")),
                                  (torch, ("from_file",))):
                for name in names:
                    stack.enter_context(patch.object(target, name, no_disk))
            dataset = worker.load(root, record_limit=1, **bindings)
            stage = "checks"
            require(type(dataset) is runtime.EncodedSFTDataset and len(dataset) == 1)
            require(dataset.split == "train" and dataset.maximum_record_tokens == 3)
            require(content_opens == dict.fromkeys(approved, 1))
            before_calls, before_prepaid = calls["pread"], dict(prepaid)
            for name in ("open", "pread", "stat", "lstat", "fstat", "listdir", "scandir"):
                stack.enter_context(patch.object(os, name, no_disk))
            first, second = dataset[0], dataset[0]
            require(calls["pread"] == before_calls and prepaid == before_prepaid)
        for item in (expected, first, second):
            require(type(item) is runtime.SFTTensorExample and item.supervised_tokens == 2)
            for tensor, values in ((item.input_ids, [10, 11, 12]),
                                   (item.labels, [-100, 11, 12])):
                require(type(tensor) is torch.Tensor and tensor.dtype is torch.int64)
                require(tensor.device.type == "cpu" and tuple(tensor.shape) == (3,))
                require(not tensor.requires_grad and tensor.tolist() == values)
        require(torch.equal(first.input_ids, expected.input_ids))
        require(torch.equal(first.labels, expected.labels))
        require(torch.equal(second.input_ids, first.input_ids))
        require(first.supervised_tokens == expected.supervised_tokens == 2)
        require(dataset.verification == reference.verification)
        require(runtime._load_verified_sft_split is original_loader)
        require(runtime._open_secure_dataset_file is original_open)
        require(runtime._finish_secure_dataset_file is original_finish)
        require(runtime.EncodedSFTDataset.__init__ is original_initializer)
        require(runtime.EncodedSFTDataset.__getitem__ is original_getitem)
        expected_counts = {"encoded_metadata_read_bytes": manifest_bytes,
                           "encoded_binary_read_bytes": 24,
                           "encoded_index_read_bytes": index_bytes, "encoded_index_rows": 1}
        require(prepaid == expected_counts)
        require(consumed == {**expected_counts, "encoded_index_rows": 0})
        require(positions == {name: extent for name, (_, extent) in approved.items()})
        summary = worker.summary()
        require(summary["state"] == "complete")
        for name, expected_value in (("metadata_bytes_read", manifest_bytes),
                                     ("binary_bytes_read", 24), ("index_bytes_read", index_bytes),
                                     ("encoded_index_rows", 1), ("file_opens", 4),
                                     ("records_loaded", 1)):
            require(summary[name] == expected_value)
        for name in ("full_content_hashes_verified", "production_eligible",
                     "real_input_execution_permitted"):
            require(summary[name] is False)
        require(dataset.verification["fully_hashed_shards"] == 0)
        require(not torch.cuda.is_initialized())
        client_summary = client.summary()
        requests = index_bytes + 4  # Metadata, one row, each index byte, two binaries.
        require(client_summary["requests_sent"] == client_summary["confirmations_sent"] == requests)
        require(client_summary["acknowledgements_validated"] == requests)
        require(client_summary["wire_bytes_reserved"] <= MAXIMUM_WIRE_BYTES)
        outcome(result_pipe, {"status": "passed", "stage": "checks", "records_loaded": 1,
                              "metadata_bytes_read": manifest_bytes, "binary_bytes_read": 24,
                              "index_bytes_read": index_bytes, "encoded_index_rows": 1,
                              "file_opens": 4, "requests": client_summary["requests_sent"]})
        stage = "cleanup"
        require(result_pipe.poll(remaining_seconds(deadline_ns, MAXIMUM_SECONDS)))
        require(result_pipe.recv_bytes(maxlength=128) == RELEASE)
        client.close()
        outcome(result_pipe, {"status": "closed", "stage": "cleanup"})
    except BaseException:
        report = {"status": "failed", "stage": stage if stage in CHILD_STAGES else "guard"}
        if worker is not None:
            summary = worker.summary()
            if summary.get("state") == "failed":
                report["worker_failed"] = True
        try:
            outcome(result_pipe, report)
        except BaseException:
            pass
    finally:
        try:
            if client is not None:
                client.close()
            else:
                endpoint.close()
        finally:
            result_pipe.close()


def serve(session, ledger, channels, results, processes, deadline_ns):
    global CHILD_STAGE
    protocol = module("posttraining_sft_epoch_protocol_v2")
    reports, receives, wire = {}, 0, 0
    with selectors.DefaultSelector() as selector:
        for rank, channel in enumerate(channels):
            channel.settimeout(1)
            selector.register(channel, selectors.EVENT_READ, ("ledger", rank))
            selector.register(results[rank], selectors.EVENT_READ, ("result", rank))
        while True:
            require(time.monotonic_ns() < deadline_ns)
            for key, _ in selector.select(timeout=0.1):
                kind, rank = key.data
                if kind == "result":
                    observed = json.loads(key.fileobj.recv_bytes(maxlength=2048))
                    require(type(observed) is dict)
                    CHILD_STAGE = observed.get("stage", "not_reported")
                    require(CHILD_STAGE in CHILD_STAGES and observed.get("status") == "passed")
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
            if len(reports) == 2 and snapshot["pending_acknowledgements"] == 0:
                requests = sum(report["requests"] for report in reports.values())
                require(snapshot["requests_predebited"] == requests)
                require(snapshot["acknowledgements_sent"] == requests)
                require(snapshot["acknowledgements_confirmed"] == requests)
                require(snapshot["global_optimizer_steps_completed"] == 0)
                require(ledger.summary()["state"] == "active")
                # Workers cannot close their ledger sockets until the cleanup
                # release, even if their result pipe outran the final confirm.
                require(all(process.is_alive() for process in processes))
                return reports, wire
            require(all(process.exitcode in (None, 0) for process in processes))


def run():
    global STAGE
    guards()
    require(sys.argv[1:] == ["--synthetic-only"])
    os.umask(0o077)
    deadline_ns = time.monotonic_ns() + MAXIMUM_SECONDS * 1_000_000_000
    context = multiprocessing.get_context("spawn")
    channels, child_channels, results, child_results, processes = [], [], [], [], []
    with tempfile.TemporaryDirectory(prefix="lumi-SYNTHETIC-worker-input-v1-") as temporary:
        try:
            STAGE = "fixture"
            parent = Path(temporary)
            parent.chmod(0o700)
            root, bindings, manifest_bytes, index_bytes = synthetic_root(parent)
            ledger = controller(2 * manifest_bytes)
            protocol = module("posttraining_sft_epoch_protocol_v2")
            epoch, tokens = protocol.create_epoch(), protocol.create_rank_tokens(2)
            session = protocol.EpochLedgerSession(ledger, world_size=2, rank_tokens=tokens,
                                                   epoch=epoch)
            session.begin_invocation("reference", 0, 2, False)
            STAGE = "workers"
            for rank in range(2):
                left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
                channels.append(left)
                child_channels.append(right)
                receiver, sender = context.Pipe(duplex=True)
                results.append(receiver)
                child_results.append(sender)
                process = context.Process(target=loader_worker, args=(
                    rank, right, sender, epoch, tokens[rank], str(root), bindings,
                    manifest_bytes, index_bytes, deadline_ns,
                ))
                processes.append(process)
                process.start()
                right.close()
                sender.close()
            STAGE = "ledger"
            reports, wire_bytes = serve(session, ledger, channels, results, processes, deadline_ns)
            STAGE = "accounting"
            used = ledger.summary()["resources_reserved"]
            expected = {"encoded_metadata_read_bytes": 2 * manifest_bytes,
                        "encoded_binary_read_bytes": 48,
                        "encoded_index_read_bytes": 2 * index_bytes, "encoded_index_rows": 2}
            require(all(used[name] == expected.get(name, 0) for name in used))
            for report in reports.values():
                require(report["metadata_bytes_read"] == manifest_bytes)
                require(report["binary_bytes_read"] == 24)
                require(report["index_bytes_read"] == index_bytes)
                require(report["encoded_index_rows"] == report["records_loaded"] == 1)
                require(report["file_opens"] == 4)
                require(report["requests"] == INDEX_ROW_BYTES + 4)
            STAGE = "planned_interrupt"
            try:
                session.disconnect(0)
            except protocol.LedgerProtocolError:
                pass
            else:
                raise AssertionError("fixture_interruption_missing")
            require(session.summary()["state"] == "failed")
            summary = ledger.summary()
            require(summary["state"] == "failed" and summary["failure_code"] == "interrupted")
            require(summary["global_optimizer_steps_reserved"] == 0)
            require(summary["global_optimizer_steps_completed"] == 0)
            require(summary["invocations_completed"] == 0)
            require(summary["invocations_started"] == 1)
            STAGE = "cleanup"
            for result in results:
                result.send_bytes(RELEASE)
            for result, process in zip(results, processes, strict=True):
                require(result.poll(remaining_seconds(deadline_ns, 10)))
                require(json.loads(result.recv_bytes(maxlength=2048))
                        == {"status": "closed", "stage": "cleanup"})
                process.join(timeout=remaining_seconds(deadline_ns, 5))
                require(not process.is_alive() and process.exitcode == 0)
            STAGE = "complete"
            return {"schema_version": SCHEMA, "status": "passed",
                    **dict.fromkeys(FLAGS, True), **dict.fromkeys(NONCLAIMS, False),
                    "controlled_loads": 2, "direct_reference_loads": 2, "records_loaded": 2,
                    "metadata_bytes_read": 2 * manifest_bytes, "binary_bytes_read": 48,
                    "index_bytes_read": 2 * index_bytes, "encoded_index_rows": 2, "file_opens": 8,
                    "ledger_requests": 2 * (INDEX_ROW_BYTES + 4),
                    "parent_wire_bytes": wire_bytes,
                    "global_optimizer_steps_reserved": 0, "global_optimizer_steps_completed": 0,
                    "invocations_started": 1, "invocations_completed": 0}
        finally:
            for channel in channels + child_channels:
                channel.close()
            # Signal all owned workers first; one slow child must never prevent
            # cleanup of its sibling. The outer test adds a process-group bound.
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                if process.pid is not None:
                    process.join(timeout=2)
            for process in processes:
                if process.is_alive():
                    process.kill()
            cleanup_failed = False
            for process in processes:
                if process.pid is not None:
                    process.join(timeout=2)
                if process.is_alive():
                    cleanup_failed = True
                else:
                    process.close()
            for result in results + child_results:
                result.close()
            require(not cleanup_failed)


def main():
    output_fd = os.dup(1)
    with open(os.devnull, "wb") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    try:
        report, code = run(), 0
    except BaseException:
        report = {"schema_version": SCHEMA, "status": "failed", "stage": STAGE,
                  "child_stage": CHILD_STAGE if CHILD_STAGE in CHILD_STAGES else "not_reported",
                  **dict.fromkeys(NONCLAIMS, False)}
        code = 1
    raw = canonical(report) + b"\n"
    require(len(raw) <= 4096 and os.write(output_fd, raw) == len(raw))
    os.close(output_fd)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
