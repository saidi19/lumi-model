"""One generated tiny CPU import for a separately supervised Linux composition.

There is no real-input argument or authorization switch. The caller provides one
new empty private directory, a cooperative budget callback, and hard containment.
Fixture generation is separately bounded but is not the frozen import ledger.
No forward pass, optimizer, network, GPU, cleanup, reuse or retry is provided.
"""

from __future__ import annotations

import os
import stat
import threading
from pathlib import Path

from .posttraining_final_base_metadata_v1 import canonical, sha

SCHEMA = "lumi.checkpoint-guard-fixture-result.v0.1"
MAXIMUM_GENERATED_BYTES = 4 * 1024**2
MAXIMUM_GENERATED_FILES = 32
MAXIMUM_FILE_BYTES = 1024**2
MAXIMUM_REPORT_BYTES = 4096
ERROR_CODES = frozenset({"fixture_root_rejected", "fixture_scope_rejected",
                         "fixture_generation_failed", "fixture_import_failed",
                         "fixture_budget_failed", "fixture_result_rejected"})
CONFIG = {
    "vocab_size": 17, "hidden_size": 8, "intermediate_size": 16,
    "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
    "head_dim": 4, "max_position_embeddings": 16, "rope_theta": 10000.0,
    "rms_norm_eps": 0.00001, "attention_bias": False, "tie_word_embeddings": True,
    "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2, "doc_token_id": 3,
    "model_type": "lumi",
}
LIMITS = {
    "maximum_seconds": 30, "maximum_coordinator_read_bytes": 64 * 1024**2,
    "maximum_package_read_bytes": 32 * 1024**2,
    "maximum_private_write_bytes": 16 * 1024**2,
    "maximum_package_write_bytes": 4 * 1024**2, "maximum_control_bytes": 131072,
}


class GuardFixtureError(ValueError):
    def __init__(self, code="fixture_generation_failed"):
        self.code = (code if type(code) is str and code in ERROR_CODES
                     else "fixture_generation_failed")
        super().__init__(self.code)


def _need(value, code):
    if not value:
        raise GuardFixtureError(code) from None


def _tick(callback):
    try:
        _need(callback() is None, "fixture_budget_failed")
    except BaseException:
        raise GuardFixtureError("fixture_budget_failed") from None


def _root(root):
    _need(type(root) is type(Path()) and root.is_absolute() and ".." not in root.parts
          and len(root.parts) >= 3, "fixture_root_rejected")
    for parent in (root, *root.parents):
        _need(not parent.is_symlink(), "fixture_root_rejected")
    info = root.lstat()
    _need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
          and stat.S_IMODE(info.st_mode) == 0o700, "fixture_root_rejected")
    with os.scandir(root) as entries:
        _need(next(entries, None) is None, "fixture_root_rejected")


def _write(path, raw):
    _need(type(raw) is bytes and 0 < len(raw) <= MAXIMUM_FILE_BYTES, "fixture_generation_failed")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                 0o600)
    try:
        os.fchmod(fd, 0o600)
        _need(os.write(fd, raw) == len(raw), "fixture_generation_failed")
        os.fsync(fd)
    finally:
        os.close(fd)


def _inventory(root, callback):
    """Bound metadata and bytes of own freshly generated tiny files, before hashing."""
    files, total = [], 0
    pending = [(root, "")]
    while pending:
        directory, prefix = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                _tick(callback)
                _need(len(files) + len(pending) < MAXIMUM_GENERATED_FILES,
                      "fixture_generation_failed")
                path = directory / entry.name
                info = entry.stat(follow_symlinks=False)
                _need(info.st_uid == os.geteuid() and not stat.S_ISLNK(info.st_mode),
                      "fixture_generation_failed")
                if stat.S_ISDIR(info.st_mode):
                    _need(prefix == "" and entry.name == "model", "fixture_generation_failed")
                    path.chmod(0o700)
                    pending.append((path, "model/"))
                else:
                    _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                          and 0 < info.st_size <= MAXIMUM_FILE_BYTES,
                          "fixture_generation_failed")
                    total += info.st_size
                    _need(total <= MAXIMUM_GENERATED_BYTES, "fixture_generation_failed")
                    path.chmod(0o600)  # Only fixture-created files, never imported sources.
                    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
                    try:
                        before = os.fstat(fd)
                        _need(before.st_ino == info.st_ino and before.st_dev == info.st_dev
                              and before.st_size == info.st_size, "fixture_generation_failed")
                        chunks, remaining = [], info.st_size
                        while remaining:
                            _tick(callback)
                            part = os.read(fd, min(65536, remaining))
                            _need(part and len(part) <= remaining, "fixture_generation_failed")
                            chunks.append(part)
                            remaining -= len(part)
                        after = os.fstat(fd)
                        fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_uid",
                                  "st_gid", "st_size", "st_mtime_ns", "st_ctime_ns")
                        _need(all(getattr(before, key) == getattr(after, key) for key in fields),
                              "fixture_generation_failed")
                        raw = b"".join(chunks)
                    finally:
                        os.close(fd)
                    files.append({"path": prefix + entry.name, "bytes": len(raw),
                                  "sha256": sha(raw)})
    return sorted(files, key=lambda value: value["path"])


def _metadata(checkpoint, tokens, callback):
    config_raw = canonical(CONFIG)
    state = {
        "checkpoint_schema_version": 1, "step": 4, "sequence_cursor": 8, "tokens_seen": 256,
        "run_fingerprint": "a" * 64,
        "run_specification": {"source_commit": "b" * 40, "config_sha256": sha(config_raw),
                              "distributed": {"world_size": 2}},
    }
    state_raw = canonical(state)
    _write(checkpoint / "distributed-training-state.json", state_raw)
    for rank in range(2):
        _write(checkpoint / f"trainer-rank-{rank:05d}.pt", b"opaque synthetic trainer")
    files = _inventory(checkpoint, callback)
    manifest_raw = "".join(f"{row['sha256']}  {row['path']}\n" for row in files).encode("ascii")
    _write(checkpoint / "manifest.sha256", manifest_raw)
    files.append({"path": "manifest.sha256", "bytes": len(manifest_raw),
                  "sha256": sha(manifest_raw)})
    tokenfiles = []
    (tokens / "tokenizer").mkdir(mode=0o700)
    for name in ("lumi_tokenizer_metadata.json", "tokenizer.json", "tokenizer_config.json"):
        raw = b'{"synthetic":true}'
        relative = "tokenizer/" + name
        _write(tokens / relative, raw)
        tokenfiles.append({"relative_path": relative, "bytes": len(raw), "sha256": sha(raw)})
    handoff = {
        "checkpoint": {"directory": "checkpoint", "step": 4, "sequence_cursor": 8,
                       "tokens_seen": 256},
        "files": files, "gate": "lumi_4b_checkpoint_handoff_v0.1", "run_fingerprint": "a" * 64,
        "schema_version": 1, "source_commit": "b" * 40, "status": "sealed", "world_size": 2,
    }
    handoff["handoff_sha256"] = sha(canonical(handoff))
    handoff_raw = canonical(handoff)
    base = {
        "schema_version": "lumi.posttraining-base-binding.v0.1", "checkpoint_stage": "base",
        "model_id": "lumi-4b-base", "run_fingerprint": "a" * 64, "source_commit": "b" * 40,
        "synthetic_fixture_only": True,
        "checkpoint": {"step": 4, "sequence_cursor": 8, "training_tokens": 256,
                       "world_size": 2, "manifest_sha256": sha(manifest_raw)},
        "tokenizer": {"files": tokenfiles, "manifest_sha256": sha(canonical(tokenfiles))},
        "handoff": {"sha256": handoff["handoff_sha256"], "file_sha256": sha(handoff_raw)},
    }
    return canonical(base), manifest_raw, state_raw, config_raw, handoff_raw


def _generate(checkpoint, callback):
    import torch
    import torch.distributed.checkpoint as dcp

    from .configuration_lumi import LumiConfig
    from .modeling_lumi import LumiForCausalLM

    _need(not torch.distributed.is_initialized() and torch.get_default_dtype() == torch.float32,
          "fixture_scope_rejected")
    torch.set_num_threads(1)
    config = LumiConfig(**CONFIG)
    config._attn_implementation = "eager"
    with torch.device("cpu"):
        model = LumiForCausalLM(config).float()
    _need(0 < sum(parameter.numel() for parameter in model.parameters()) <= 4096,
          "fixture_scope_rejected")
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            _tick(callback)
            values = torch.arange(parameter.numel(), dtype=torch.float32, device="cpu")
            parameter.copy_(((values.remainder(23) - 11) / 31 + index / 100).reshape(
                parameter.shape))
    state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    _need(sum(tensor.numel() * tensor.element_size() for tensor in state.values()) <= 65536
          and all(tensor.device.type == "cpu" for tensor in state.values()),
          "fixture_scope_rejected")
    _tick(callback)
    dcp.save({"model": state}, storage_writer=dcp.FileSystemWriter(checkpoint / "model"))
    _tick(callback)


def run_tiny_fixture(root: Path, *, check_budget):
    """Generate one fixed tiny source and execute exactly one frozen BF16 import."""
    _need(callable(check_budget) and threading.current_thread() is threading.main_thread()
          and threading.active_count() == 1, "fixture_scope_rejected")
    _tick(check_budget)
    try:
        _root(root)
        checkpoint, tokens = root / "input", root / "tokens"
        checkpoint.mkdir(mode=0o700)
        tokens.mkdir(mode=0o700)
        _generate(checkpoint, check_budget)
        metadata = _metadata(checkpoint, tokens, check_budget)
        from .posttraining_checkpoint_import_plan_v1 import build_import_plan
        from .posttraining_checkpoint_import_runner_v2 import run_synthetic_import
        from .posttraining_checkpoint_io_contract_v1 import build_kernel_io_contract

        plan = build_import_plan(*metadata, expected_base_sha256=sha(metadata[0]), dtype="bfloat16")
        bounds = plan["resource_bounds"]
        contract = build_kernel_io_contract(plan, checkpoint_read_contract={
            "expected_manifest_sha256": plan["metadata"]["checkpoint_manifest_sha256"],
            "maximum_file_bytes": max(row["bytes"] for row in plan["checkpoint_inventory"]),
            "maximum_read_bytes": 8 * 1024**2, "maximum_read_call_bytes": 1024**2,
        }, hf_io_contract={
            "maximum_read_bytes": 8 * 1024**2,
            "maximum_write_bytes": bounds["maximum_backend_output_bytes"],
            "maximum_file_bytes": bounds["maximum_backend_output_bytes"],
            "maximum_control_bytes": 65536, "target_shard_bytes": 268435456,
        })
        _tick(check_budget)
        try:
            outcome = run_synthetic_import(
                metadata=metadata, expected_base_sha256=sha(metadata[0]),
                checkpoint_root=checkpoint, tokenizer_root=tokens, attempt=root / "attempt",
                dtype="bfloat16", limits=dict(LIMITS),
                kernel_io_contract=contract,
                expected_kernel_io_contract_sha256=contract["contract_sha256"],
            )
        except BaseException:
            raise GuardFixtureError("fixture_import_failed") from None
        _tick(check_budget)
        _need(outcome["status"] == "PASS_synthetic_coordinator_not_real_import"
              and outcome["full_input_audits_completed"] == 2
              and outcome["package"]["real_worker_checksum_verifier_passed"] is True
              and outcome["explicit_checkpoint_and_HF_content_metered"] is True
              and all(outcome[key] is False for key in (
                  "real_4b_import", "training_authorized", "production_eligible", "retry_allowed")),
              "fixture_result_rejected")
        result = {
            "schema": SCHEMA, "status": "passed", "code": "ok", "synthetic_only": True,
            "import_passes": 1, "checkpoint_full_audits": 2, "tokenizer_full_audits": 2,
            "model_parameters": plan["model_parameters"],
            "generated_input_bytes": bounds["checkpoint_inventory_bytes"]
            + bounds["tokenizer_inventory_bytes"],
            "generated_input_files": bounds["checkpoint_inventory_files"] + 3,
            "plan_sha256": plan["plan_sha256"],
            "kernel_io_contract_sha256": contract["contract_sha256"],
            "kernel_evidence_sha256": outcome["kernel_evidence_sha256"],
            "package_manifest_sha256": outcome["package"]["manifest_file_sha256"],
            "logical_content_read_bytes": outcome["content_accounting"][
                "successful_content_read_bytes_requested"],
            "private_write_bytes_reserved": outcome["private_write_bytes_reserved"],
            "frozen_worker_checksum_verifier_passed": True,
            "fixture_generation_in_import_ledger": False, "real_4b_import": False,
            "training_authorized": False, "production_eligible": False,
            "gpu_used": False, "forward_calls": 0, "optimizer_steps": 0,
            "host_containment_self_attested": False,
        }
        _need(result["generated_input_bytes"] <= MAXIMUM_GENERATED_BYTES
              and result["generated_input_files"] <= MAXIMUM_GENERATED_FILES
              and len(canonical(result)) <= MAXIMUM_REPORT_BYTES, "fixture_result_rejected")
        _tick(check_budget)
        return result
    except GuardFixtureError:
        raise
    except BaseException:
        raise GuardFixtureError("fixture_generation_failed") from None
