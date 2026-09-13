"""Bounded private packaging; no execution, origin-authentication or OS authority.

The caller supplies its unchanged pinned v1 plan, independently pinned logical
I/O contract and v5 CPU evidence, and keeps the input/output trees quiescent.
This component does not
authenticate who produced those objects. Even a successful package is research
only, never evidence that the real 4B checkpoint was imported. Failed attempts
are left in place; the coordinator owns the exclusive no-retry attempt marker.

Content I/O is charged before each syscall with no refund. The callback is
cooperative; a supervising process must enforce blocked-syscall/RAM limits.
The frozen worker consumer is called unchanged through a constrained path
adapter: its hashes still stream and its otherwise unbounded manifest read is
restricted to a preverified small control file. No ML import occurs until that
actual consumer is requested. Neither sources nor frozen modules are modified.
"""

from __future__ import annotations

import os
import re
import stat
import time
from pathlib import Path

from . import posttraining_checkpoint_import_plan_v1 as planner
from . import posttraining_checkpoint_import_torch_v5 as kernel
from . import posttraining_checkpoint_stream_v1 as stream
from .posttraining_checkpoint_io_contract_v1 import (
    KernelIOContractError,
    validate_kernel_io_contract,
)
from .posttraining_final_base_metadata_v1 import canonical, json_object, sha

SCHEMA = "lumi.sft-checkpoint-import-receipt.v0.4"
STATUS = "PASS_private_research_package_not_execution_authority"
MANIFEST = "manifest.sha256"
RECEIPT = "import-receipt.json"
CHUNK_BYTES = 1048576
MAXIMUM_CONTROL_BYTES = 2 * CHUNK_BYTES
MAXIMUM_FILES = kernel.MAXIMUM_OUTPUT_FILES + 5
TOKEN_NAMES = frozenset({
    "tokenizer/lumi_tokenizer_metadata.json", "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
})
CODE_NAMES = frozenset({"configuration_lumi.py", "modeling_lumi.py"})
FALSE_FLAGS = (
    "real_4b_import", "execution_permitted", "training_authorized",
    "real_sft_execution_permitted", "production_eligible", "raw_training_text_retained",
    "os_resource_limits_enforced", "source_authenticity_verified",
)
ERROR_CODES = frozenset({
    "invalid_arguments", "plan_binding", "kernel_evidence", "path_rejected",
    "inventory_mismatch", "identity_changed", "read_limit", "write_limit",
    "control_limit", "budget_failed", "content_changed", "already_finalized",
    "receipt_invalid", "consumer_failed", "package_failed", "kernel_io_contract",
})
DCP_EVIDENCE_KEYS = frozenset({
    "dcp_read_bytes_reserved", "dcp_input_audit_bytes", "model_payload_read_bytes_reserved",
    "hash_verified_files", "dcp_reads_metered", "physical_disk_bytes_measured",
    "os_resource_limits_enforced", "real_4b_import", "training_authorized", "production_eligible",
})
HF_EVIDENCE_KEYS = frozenset({
    "hf_content_read_bytes_reserved", "hf_content_write_bytes_reserved",
    "hf_content_byte_accounting", "physical_io_bytes_measured", "os_resource_limits_enforced",
    "memory_limits_enforced", "hard_timeout_enforced", "serialization_scratch_bytes",
    "maximum_serialized_shard_bytes", "actual_HF_from_pretrained_with_in_memory_state",
    "native_path_checkpoint_read_or_write_used",
})


class CheckpointPackageError(ValueError):
    def __init__(self, code):
        self.code = code if type(code) is str and code in ERROR_CODES else "package_failed"
        super().__init__(self.code)


def _need(value, code):
    if not value:
        raise CheckpointPackageError(code) from None


def _digest(value):
    return (type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _integer(value, maximum=2**63 - 1):
    return type(value) is int and 0 <= value <= maximum


class _Ledger:
    def __init__(self, reads, writes, control, callback):
        _need(_integer(reads) and _integer(writes)
              and _integer(control, MAXIMUM_CONTROL_BYTES) and control > 0
              and callable(callback), "invalid_arguments")
        self.read_cap, self.write_cap, self.control = reads, writes, control
        self.callback = callback
        self.reads = self.writes = self.checks = 0
        self.tick()

    def tick(self):
        self.checks += 1
        try:
            self.callback()
        except BaseException:
            raise CheckpointPackageError("budget_failed") from None

    def admit(self, reads=0, writes=0):
        self.tick()
        _need(self.reads + reads <= self.read_cap, "read_limit")
        _need(self.writes + writes <= self.write_cap, "write_limit")

    def read(self, count):
        self.admit(reads=count)
        self.reads += count

    def write(self, count):
        self.admit(writes=count)
        self.writes += count


class _Guard(stream._Verifier):
    """Reuse the existing pinned-root/no-symlink primitives, adding parent ticks."""

    def __init__(self, ledger):
        self.ledger = ledger
        super().__init__(time.monotonic() + 86400, 2**63 - 1, CHUNK_BYTES, True)

    def check_time(self):
        self.ledger.tick()
        super().check_time()


class _Tree:
    def __init__(self, root, ledger, *, output=True):
        _need(type(root) is type(Path()) and root.is_absolute() and str(root) != "/",
              "path_rejected")
        parts = stream._parts(str(root)[1:])
        self.root, self.ledger, self.guard = root, ledger, _Guard(ledger)
        self.output = output
        self.rows, self.dirs = {}, {}
        try:
            self.fd = self.guard.root(parts)
        except BaseException:
            self.close()
            raise

    def close(self):
        failure = None
        while self.guard.owned:
            try:
                self.guard.close(self.guard.owned[-1])
            except BaseException as error:
                failure = error
        if failure is not None:
            raise CheckpointPackageError("package_failed") from None

    def scan(self):
        """Bounded metadata-only walk, allowing only the normalized tokenizer dir."""
        rows, dirs = {}, {}

        def walk(fd, prefix):
            self.ledger.tick()
            before = os.fstat(fd)
            self.guard.metadata(before, directory=True)
            dirs[prefix] = stream._snapshot(before)
            with os.scandir(fd) as entries:
                for entry in entries:
                    self.ledger.tick()
                    _need(len(rows) + len(dirs) <= MAXIMUM_FILES + 2, "inventory_mismatch")
                    name = prefix + entry.name
                    stream._parts(name)
                    info = os.stat(entry.name, dir_fd=fd, follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode):
                        _need(name == "tokenizer", "inventory_mismatch")
                        child = self.guard.open(entry.name, directory=True, parent=fd)
                        try:
                            _need(stream._snapshot(info) == stream._snapshot(os.fstat(child)),
                                  "identity_changed")
                            walk(child, name + "/")
                            _need(stream._snapshot(info) == stream._snapshot(os.stat(
                                entry.name, dir_fd=fd, follow_symlinks=False)), "identity_changed")
                        finally:
                            self.guard.close(child)
                    else:
                        self.guard.metadata(info, directory=False)
                        # The frozen private consumer requires exactly 0600.
                        _need(not self.output or stat.S_IMODE(info.st_mode) == 0o600,
                              "path_rejected")
                        _need(name not in rows and len(rows) < MAXIMUM_FILES,
                              "inventory_mismatch")
                        rows[name] = stream._snapshot(info)
            _need(stream._snapshot(os.fstat(fd)) == stream._snapshot(before), "identity_changed")

        self.guard.check_ancestry()
        walk(self.fd, "")
        self.guard.check_ancestry()
        return rows, dirs

    def refresh(self):
        self.rows, self.dirs = self.scan()

    def unchanged(self):
        _need(self.scan() == (self.rows, self.dirs), "identity_changed")

    def parent(self, name):
        parts = stream._parts(name)
        _need(len(parts) == 1 or (len(parts) == 2 and parts[0] == "tokenizer"),
              "path_rejected")
        self.guard.check_ancestry()
        if len(parts) == 1:
            return self.fd, parts[-1], False
        before = os.stat("tokenizer", dir_fd=self.fd, follow_symlinks=False)
        self.guard.metadata(before, directory=True)
        fd = self.guard.open("tokenizer", directory=True, parent=self.fd)
        _need(stream._snapshot(before) == stream._snapshot(os.fstat(fd)), "identity_changed")
        return fd, parts[-1], True

    def reader(self, name):
        return _Reader(self, name)

    def create(self, name):
        self.ledger.tick()
        parent, leaf, close = self.parent(name)
        fd = None
        try:
            fd = os.open(leaf, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                         | os.O_CLOEXEC, 0o600, dir_fd=parent)
            os.fchmod(fd, 0o600)
            if close:
                close = False  # An ambiguous parent close must not be retried.
                self.guard.close(parent)
            return fd
        except BaseException:
            if fd is not None:
                os.close(fd)
            raise
        finally:
            if close:
                self.guard.close(parent)

    def write(self, name, raw):
        _need(type(raw) is bytes and 0 < len(raw) <= self.ledger.control, "control_limit")
        self.ledger.admit(writes=len(raw))
        fd = self.create(name)
        try:
            _write(fd, raw, self.ledger)
            self.ledger.tick()
            os.fsync(fd)
        finally:
            os.close(fd)


class _Reader:
    def __init__(self, tree, name):
        self.tree, self.name, self.offset = tree, name, 0
        _need(name in tree.rows, "inventory_mismatch")
        self.expected = tree.rows[name]
        parent, leaf, close = tree.parent(name)
        try:
            info = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            tree.guard.metadata(info, directory=False)
            _need(stream._snapshot(info) == self.expected, "identity_changed")
            self.fd = tree.guard.open(leaf, directory=False, parent=parent)
            _need(stream._snapshot(os.fstat(self.fd)) == self.expected, "identity_changed")
            self.size = info.st_size
        finally:
            if close:
                tree.guard.close(parent)

    def __enter__(self):
        return self

    def read(self, size=-1):
        _need(_integer(size, CHUNK_BYTES), "read_limit")
        request = min(size, self.size - self.offset)
        self.tree.ledger.read(request)
        if not request:
            return b""
        raw = os.read(self.fd, request)
        _need(len(raw) == request, "content_changed")
        self.offset += len(raw)
        return raw

    def __exit__(self, *_args):
        try:
            self.tree.ledger.tick()
            _need(stream._snapshot(os.fstat(self.fd)) == self.expected, "identity_changed")
            parent, leaf, close = self.tree.parent(self.name)
            try:
                _need(stream._snapshot(os.stat(leaf, dir_fd=parent, follow_symlinks=False))
                      == self.expected, "identity_changed")
            finally:
                if close:
                    self.tree.guard.close(parent)
        finally:
            self.tree.guard.close(self.fd)


def _write(fd, raw, ledger):
    for offset in range(0, len(raw), CHUNK_BYTES):
        chunk = raw[offset:offset + CHUNK_BYTES]
        ledger.write(len(chunk))
        _need(os.write(fd, chunk) == len(chunk), "package_failed")


def _hash(tree, name, *, copy_to=None):
    import hashlib

    digest = hashlib.sha256()
    with tree.reader(name) as reader:
        while reader.offset < reader.size:
            raw = reader.read(min(CHUNK_BYTES, reader.size - reader.offset))
            digest.update(raw)
            if copy_to is not None:
                _write(copy_to, raw, tree.ledger)
        return {"path": name, "bytes": reader.size, "sha256": digest.hexdigest()}


def _small(tree, name):
    with tree.reader(name) as reader:
        _need(reader.size <= tree.ledger.control, "control_limit")
        tree.ledger.admit(reads=reader.size)
        chunks = []
        while reader.offset < reader.size:
            chunks.append(reader.read(min(CHUNK_BYTES, reader.size - reader.offset)))
        return b"".join(chunks)


def _bounded_json(value, maximum):
    """Reject oversized/deep objects before canonical serialization or copying."""
    remaining = maximum

    def visit(item, depth=0):
        nonlocal remaining
        _need(depth <= 24 and remaining >= 0, "control_limit")
        remaining -= 8
        if type(item) is dict:
            _need(len(item) <= maximum // 8, "control_limit")
            for key, child in item.items():
                _need(type(key) is str, "control_limit")
                visit(key, depth + 1)
                visit(child, depth + 1)
        elif type(item) is list:
            _need(len(item) <= maximum // 8, "control_limit")
            for child in item:
                visit(child, depth + 1)
        elif type(item) is str:
            _need(len(item) <= remaining, "control_limit")
            remaining -= len(item.encode("utf-8"))
        else:
            _need(item is None or type(item) in {bool, int, float}, "control_limit")
            if type(item) is int:
                _need(abs(item) <= 2**63 - 1, "control_limit")
        _need(remaining >= 0, "control_limit")

    visit(value)
    raw = canonical(value)
    _need(0 < len(raw) <= maximum, "control_limit")
    return json_object(raw)


def _io_evidence(plan, evidence, contract):
    """Bind reported logical reservations; not origin proof or measured OS I/O."""
    dcp = evidence["dcp_read_verification"]
    _need(type(dcp) is dict and set(dcp) == DCP_EVIDENCE_KEYS, "kernel_evidence")
    for key in ("dcp_read_bytes_reserved", "dcp_input_audit_bytes",
                "model_payload_read_bytes_reserved", "hash_verified_files"):
        _need(_integer(dcp[key]) and dcp[key] > 0, "kernel_evidence")
    _need(dcp["dcp_reads_metered"] is True and all(dcp[key] is False for key in (
        "physical_disk_bytes_measured", "os_resource_limits_enforced", "real_4b_import",
        "training_authorized", "production_eligible")), "kernel_evidence")
    bounds = plan["resource_bounds"]
    _need(dcp["dcp_input_audit_bytes"] == bounds["checkpoint_inventory_bytes"]
          and dcp["hash_verified_files"] == bounds["checkpoint_inventory_files"] - 1
          and dcp["dcp_read_bytes_reserved"] == dcp["dcp_input_audit_bytes"]
          + dcp["model_payload_read_bytes_reserved"]
          and dcp["dcp_read_bytes_reserved"]
          <= contract["checkpoint_read_contract"]["maximum_read_bytes"], "kernel_evidence")
    hf = evidence["hf_content_verification"]
    _need(type(hf) is dict and set(hf) == HF_EVIDENCE_KEYS, "kernel_evidence")
    for key in ("hf_content_read_bytes_reserved", "hf_content_write_bytes_reserved",
                "serialization_scratch_bytes", "maximum_serialized_shard_bytes"):
        _need(_integer(hf[key]) and hf[key] > 0, "kernel_evidence")
    _need(hf["hf_content_byte_accounting"] == "cooperative_predebit_no_refund"
          and hf["actual_HF_from_pretrained_with_in_memory_state"] is True
          and all(hf[key] is False for key in (
              "physical_io_bytes_measured", "os_resource_limits_enforced",
              "memory_limits_enforced", "hard_timeout_enforced",
              "native_path_checkpoint_read_or_write_used")), "kernel_evidence")
    hf_cap = contract["hf_io_contract"]
    _need(evidence["output_bytes"] <= hf["hf_content_read_bytes_reserved"]
          <= hf_cap["maximum_read_bytes"]
          and hf["hf_content_write_bytes_reserved"] == evidence["output_bytes"]
          <= hf_cap["maximum_write_bytes"]
          and hf["maximum_serialized_shard_bytes"] <= hf_cap["maximum_file_bytes"]
          and hf["serialization_scratch_bytes"]
          == 4 * hf["maximum_serialized_shard_bytes"] + 2 * CHUNK_BYTES, "kernel_evidence")
    cast = evidence["cast_state_bytes"]
    projected = max(
        max(evidence["source_state_bytes"] + cast, 2 * cast)
        + evidence["largest_tensor_bytes"] + 2 * CHUNK_BYTES,
        2 * cast + hf["serialization_scratch_bytes"],
    )
    _need(evidence["projected_live_tensor_bytes"] == projected, "kernel_evidence")


def _inputs(plan, expected, evidence, contract, expected_contract, ledger):
    plan = _bounded_json(plan, planner.MAX_PLAN_BYTES)
    evidence = _bounded_json(evidence, ledger.control)
    _need(_digest(expected) and plan.get("plan_sha256") == expected
          and sha(canonical({k: v for k, v in plan.items() if k != "plan_sha256"})) == expected
          and plan.get("schema_version") == planner.SCHEMA
          and plan.get("status") == "PASS_plan_only_not_execution"
          and all(plan.get(k) is False for k in planner.FALSE_FLAGS), "plan_binding")
    tokens = plan["tokenizer_inventory"]
    _need(type(tokens) is list and len(tokens) == 3, "plan_binding")
    normalized = []
    for row in tokens:
        _need(type(row) is dict and set(row) == {"path", "bytes", "sha256"}
              and row["path"] in TOKEN_NAMES and _integer(row["bytes"]) and row["bytes"] > 0
              and _digest(row["sha256"]), "plan_binding")
        normalized.append({"relative_path": row["path"], "bytes": row["bytes"],
                           "sha256": row["sha256"]})
    _need({x["relative_path"] for x in normalized} == TOKEN_NAMES
          and sha(canonical(sorted(normalized, key=lambda x: x["relative_path"])))
          == plan["metadata"]["tokenizer_inventory_sha256"], "plan_binding")
    try:
        contract = validate_kernel_io_contract(plan, contract, expected_contract)
    except KernelIOContractError:
        raise CheckpointPackageError("kernel_io_contract") from None
    _need(set(evidence) == set(kernel.EVIDENCE_KEYS)
          and evidence["schema_version"] == "lumi.sft-checkpoint-import-kernel.v0.5"
          and evidence["status"] == "PASS_cpu_kernel_roundtrip_not_execution_authority",
          "kernel_evidence")
    for key in ("model_parameters", "model_tensor_count"):
        _need(_integer(evidence[key]) and evidence[key] > 0, "kernel_evidence")
    for key in ("dtype", "model_parameters", "model_tensor_count", "buffer_dtype_policy"):
        _need(evidence[key] == plan[key], "kernel_evidence")
    _need(evidence["dtype"] in {"float32", "bfloat16"}
          and evidence["parameter_dtype"] == evidence["dtype"], "kernel_evidence")
    for key in ("source_tensor_sha256", "cast_tensor_sha256", "reload_tensor_sha256"):
        _need(_digest(evidence[key]), "kernel_evidence")
    _need(evidence["cast_tensor_sha256"] == evidence["reload_tensor_sha256"], "kernel_evidence")
    for key in ("complete_keys_shapes_verified", "hf_reload_tensor_equal", "strict_assign_verified",
                "source_dicts_released_before_reload", "configuration_fields_preserved",
                "cpu_only", "no_forward_pass"):
        _need(evidence[key] is True, "kernel_evidence")
    for key in ("real_4b_import", "training_authorized", "production_eligible",
                "os_resource_limits_enforced", "complete_source_read_budget_verified"):
        _need(evidence[key] is False, "kernel_evidence")
    _need(type(evidence["tied_aliases_verified"]) is bool
          and evidence["tied_aliases_verified"]
          is plan["metadata"]["model_geometry"]["tie_word_embeddings"], "kernel_evidence")
    for key in ("source_state_bytes", "cast_state_bytes", "largest_tensor_bytes",
                "projected_live_tensor_bytes", "projected_output_bytes", "output_bytes",
                "output_file_count", "safetensor_file_count", "chunk_bytes", "budget_checks"):
        _need(_integer(evidence[key]) and evidence[key] > 0, "kernel_evidence")
    bounds = plan["resource_bounds"]
    _need(evidence["cast_state_bytes"] == bounds["cast_tensor_bytes_including_aliases"],
          "kernel_evidence")
    for left, right in (("source_state_bytes", "source_fp32_tensor_bytes_including_aliases"),
                        ("largest_tensor_bytes", "maximum_individual_tensor_bytes"),
                        ("projected_live_tensor_bytes", "maximum_projected_live_tensor_bytes")):
        _need(evidence[left] <= bounds[right], "kernel_evidence")
    _need(evidence["output_bytes"] <= evidence["projected_output_bytes"]
          <= bounds["maximum_backend_output_bytes"]
          and evidence["output_file_count"] <= kernel.MAXIMUM_OUTPUT_FILES
          and evidence["chunk_bytes"] == CHUNK_BYTES, "kernel_evidence")
    code = evidence["runtime_code_sha256"]
    _need(type(code) is dict and set(code) == CODE_NAMES
          and all(_digest(value) for value in code.values()), "kernel_evidence")
    _io_evidence(plan, evidence, contract)
    return plan, evidence, contract


def _hf_inventory(tree, evidence):
    names = set(tree.rows) - TOKEN_NAMES - {RECEIPT, MANIFEST}
    _need({"config.json"} | CODE_NAMES <= names, "inventory_mismatch")
    allowed = {"config.json", "generation_config.json", "model.safetensors.index.json"}
    _need(all("/" not in name and (re.fullmatch(
        r"model(?:-[0-9]{5}-of-[0-9]{5})?\.safetensors", name)
              or name in allowed or name in CODE_NAMES) for name in names), "inventory_mismatch")
    _need(len(names) == evidence["output_file_count"]
          and all(tree.rows[name][6] > 0 for name in names)
          and sum(tree.rows[name][6] for name in names) == evidence["output_bytes"],
          "inventory_mismatch")
    _need(all(tree.rows[name][6]
              <= evidence["hf_content_verification"]["maximum_serialized_shard_bytes"]
              for name in names if name.endswith(".safetensors")), "inventory_mismatch")
    tree.ledger.admit(reads=evidence["output_bytes"])
    inventory = [_hash(tree, name) for name in sorted(names)]
    _need(len(inventory) == evidence["output_file_count"]
          and sum(x["bytes"] for x in inventory) == evidence["output_bytes"]
          and sum(x["path"].endswith(".safetensors") for x in inventory)
          == evidence["safetensor_file_count"] > 0, "inventory_mismatch")
    _need(all(next(x["sha256"] for x in inventory if x["path"] == name) == digest
              for name, digest in evidence["runtime_code_sha256"].items()), "content_changed")
    return inventory


class _ConsumerPath:
    """Minimal path protocol actually used by the unmodified frozen consumer."""

    def __init__(self, tree, name=""):
        self.tree, self.name = tree, name

    def __truediv__(self, relative):
        name = (self.name + "/" if self.name else "") + str(relative)
        _need(name in self.tree.rows, "inventory_mismatch")
        return _ConsumerPath(self.tree, name)

    def is_dir(self):
        self.tree.guard.check_ancestry()
        return self.name == ""

    def is_file(self):
        return self.name in self.tree.rows and stat.S_ISREG(self.lstat().st_mode)

    def lstat(self):
        self.tree.ledger.tick()
        if not self.name:
            return os.fstat(self.tree.fd)
        parent, leaf, close = self.tree.parent(self.name)
        try:
            info = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            _need(stream._snapshot(info) == self.tree.rows[self.name], "identity_changed")
            return info
        finally:
            if close:
                self.tree.guard.close(parent)

    def resolve(self):
        self.tree.guard.check_ancestry()
        # Absolute normalized paths; every traversed component is pinned/no-symlink.
        return self.tree.root / self.name

    def open(self, mode):
        _need(mode == "rb", "consumer_failed")
        return self.tree.reader(self.name)

    def read_text(self, encoding):
        _need(self.name == MANIFEST and encoding == "utf-8", "consumer_failed")
        return _small(self.tree, self.name).decode("utf-8")

    def rglob(self, pattern):
        _need(self.name == "" and pattern == "*", "consumer_failed")
        self.tree.unchanged()
        return (_ConsumerPath(self.tree, name) for name in sorted(self.tree.rows))

    def relative_to(self, root):
        _need(type(root) is _ConsumerPath and root.tree is self.tree and not root.name,
              "consumer_failed")
        return Path(self.name)


def _frozen_consumer():
    from .posttraining_sft_runtime_v4 import _verify_checksum_manifest

    return _verify_checksum_manifest


def _manifest(raw):
    _need(raw and raw.endswith(b"\n") and b"\r" not in raw, "receipt_invalid")
    lines = raw.decode("utf-8").splitlines()
    _need(0 < len(lines) <= MAXIMUM_FILES, "receipt_invalid")
    rows = {}
    for line in lines:
        fields = line.split("  ")
        _need(len(fields) == 2 and _digest(fields[0]), "receipt_invalid")
        digest, name = fields
        stream._parts(name)
        _need(name != MANIFEST and name not in rows, "receipt_invalid")
        rows[name] = digest
    return rows


def _verify(tree, plan, evidence, contract, expected_manifest):
    _need(_digest(expected_manifest), "receipt_invalid")
    tree.refresh()
    _need(set(tree.rows) >= TOKEN_NAMES | {MANIFEST, RECEIPT}, "inventory_mismatch")
    manifest = _small(tree, MANIFEST)
    _need(sha(manifest) == expected_manifest, "content_changed")
    # This parser is deliberately not the frozen consumer: the actual one runs below.
    declared = _manifest(manifest)
    _need(set(declared) == set(tree.rows) - {MANIFEST}, "inventory_mismatch")
    payload = _hf_inventory(tree, evidence)
    for token in sorted(plan["tokenizer_inventory"], key=lambda x: x["path"]):
        observed = _hash(tree, token["path"])
        _need(observed == token, "content_changed")
        payload.append(observed)
    payload.sort(key=lambda x: x["path"])
    raw = _small(tree, RECEIPT)
    receipt = json_object(raw)
    unsigned = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    _need(receipt.get("receipt_sha256") == sha(canonical(unsigned))
          and canonical(receipt) + b"\n" == raw
          and receipt == _receipt(plan, evidence, payload, contract), "receipt_invalid")
    final_rows = [*payload, {"path": RECEIPT, "bytes": len(raw), "sha256": sha(raw)}]
    _need({row["path"]: row["sha256"] for row in final_rows} == declared, "content_changed")
    tree.unchanged()
    # Pre-admit the entire frozen pass: each payload once, manifest twice.
    tree.ledger.admit(reads=sum(row["bytes"] for row in final_rows) + 2 * len(manifest))
    try:
        result = _frozen_consumer()(
            _ConsumerPath(tree), expected_manifest_sha256=expected_manifest,
            require_exact_inventory=True, require_private=True,
        )
    except CheckpointPackageError:
        raise
    except BaseException:
        raise CheckpointPackageError("consumer_failed") from None
    _need(result == expected_manifest, "consumer_failed")
    tree.unchanged()
    tree.ledger.tick()
    return receipt


def _receipt(plan, evidence, payload, contract):
    receipt = {
        "schema_version": SCHEMA, "status": STATUS,
        "scope": "private_research_only_no_execution_authority",
        "plan_sha256": plan["plan_sha256"],
        "kernel_io_contract": contract,
        "kernel_io_contract_sha256": contract["contract_sha256"],
        "source_lineage": plan["metadata"],
        "handoff_file_sha256": plan["handoff_file_sha256"],
        "tokenizer_inventory": plan["tokenizer_inventory"],
        "tokenizer_inventory_semantics": "sorted_normalized_json_inventory_no_newline",
        "dtype": plan["dtype"], "model_evidence": evidence,
        "payload_inventory": payload,
        "export_config_sha256": next(x["sha256"] for x in payload if x["path"] == "config.json"),
        "legacy_export_inventory_reused": False,
        "worker_manifest_covers_payload_and_receipt": True,
        **dict.fromkeys(FALSE_FLAGS, False),
    }
    receipt["receipt_sha256"] = sha(canonical(receipt))
    return receipt


def _result(receipt, manifest, ledger):
    return {
        "schema_version": "lumi.sft-checkpoint-package-verification.v0.2",
        "status": STATUS, "manifest_file_sha256": manifest,
        "receipt_internal_sha256": receipt["receipt_sha256"],
        "kernel_io_contract_sha256": receipt["kernel_io_contract_sha256"],
        "real_worker_checksum_verifier_passed": True,
        "package_content_read_bytes": ledger.reads,
        "package_content_write_bytes": ledger.writes,
        "package_read_bytes_reserved": ledger.reads,
        "package_write_bytes_reserved": ledger.writes,
        "budget_checks": ledger.checks, **dict.fromkeys(FALSE_FLAGS, False),
    }


def _physical_disjoint(left, right):
    left_ids = {item[3][:2] for item in left.guard.ancestry}
    right_ids = {item[3][:2] for item in right.guard.ancestry}
    _need(left.guard.ancestry[-1][3][:2] not in right_ids
          and right.guard.ancestry[-1][3][:2] not in left_ids, "path_rejected")


def _close_all(trees):
    failure = False
    for tree in reversed(trees):
        try:
            tree.close()
        except BaseException:
            failure = True
    _need(not failure, "package_failed")


def finalize_package(*, output: Path, tokenizer_root: Path, plan: dict,
                     expected_plan_sha256: str, kernel_evidence: dict,
                     kernel_io_contract: dict, expected_kernel_io_contract_sha256: str,
                     maximum_read_bytes: int, maximum_write_bytes: int,
                     maximum_control_bytes: int, check_budget) -> dict:
    """Exclusively finalize the kernel-created new attempt output; never retry.

    tokenizer_root is the preverified private root containing exactly the three
    normalized tokenizer/... paths. No bare-path inference or tokenizer checksum
    manifest is admitted. Caps here count package content I/O only, including all
    verifier/consumer reads, not prior kernel output or interpreter imports.
    """
    trees = []
    result = None
    try:
        ledger = _Ledger(maximum_read_bytes, maximum_write_bytes,
                         maximum_control_bytes, check_budget)
        plan, evidence, contract = _inputs(
            plan, expected_plan_sha256, kernel_evidence, kernel_io_contract,
            expected_kernel_io_contract_sha256, ledger)
        _need(type(output) is type(Path()) and type(tokenizer_root) is type(Path())
              and not output.is_relative_to(tokenizer_root)
              and not tokenizer_root.is_relative_to(output), "path_rejected")
        tree = _Tree(output, ledger)
        trees.append(tree)
        source = _Tree(tokenizer_root, ledger, output=False)
        trees.append(source)
        _physical_disjoint(tree, source)
        tree.refresh()
        source.refresh()
        _need(not ({RECEIPT, MANIFEST} | TOKEN_NAMES) & set(tree.rows)
              and set(tree.dirs) == {""}, "already_finalized")
        _need(set(source.rows) == TOKEN_NAMES and set(source.dirs) == {"", "tokenizer/"},
              "inventory_mismatch")
        tokens = sorted(plan["tokenizer_inventory"], key=lambda x: x["path"])
        token_bytes = sum(x["bytes"] for x in tokens)
        ledger.admit(reads=evidence["output_bytes"] + token_bytes, writes=token_bytes)
        payload = _hf_inventory(tree, evidence)
        tree.unchanged()
        ledger.tick()
        os.mkdir("tokenizer", mode=0o700, dir_fd=tree.fd)
        token_fd = tree.guard.open("tokenizer", directory=True, parent=tree.fd)
        try:
            os.fchmod(token_fd, 0o700)
        finally:
            tree.guard.close(token_fd)
        for token in tokens:
            ledger.admit(reads=token["bytes"], writes=token["bytes"])
            # st_size is checked before opening/writing any token content.
            _need(source.rows[token["path"]][6] == token["bytes"], "content_changed")
            fd = tree.create(token["path"])
            try:
                observed = _hash(source, token["path"], copy_to=fd)
                _need(observed == token, "content_changed")
                ledger.tick()
                os.fsync(fd)
            finally:
                os.close(fd)
            payload.append(observed)
        source.unchanged()
        payload.sort(key=lambda x: x["path"])
        receipt = _receipt(plan, evidence, payload, contract)
        _bounded_json(receipt, ledger.control)
        raw = canonical(receipt) + b"\n"
        tree.write(RECEIPT, raw)
        rows = sorted([*payload, {"path": RECEIPT, "bytes": len(raw), "sha256": sha(raw)}],
                      key=lambda x: x["path"])
        manifest = "".join(f"{x['sha256']}  {x['path']}\n" for x in rows).encode("utf-8")
        tree.write(MANIFEST, manifest)
        verified = _verify(tree, plan, evidence, contract, sha(manifest))
        source.unchanged()
        result = _result(verified, sha(manifest), ledger)
    except CheckpointPackageError:
        raise
    except BaseException:
        raise CheckpointPackageError("package_failed") from None
    finally:
        _close_all(trees)
    ledger.tick()
    return result


def verify_package(*, output: Path, plan: dict, expected_plan_sha256: str,
                   kernel_evidence: dict, expected_manifest_sha256: str,
                   kernel_io_contract: dict, expected_kernel_io_contract_sha256: str,
                   maximum_read_bytes: int, maximum_control_bytes: int, check_budget) -> dict:
    """Read-only streamed closed-package and actual frozen-consumer verification."""
    tree = None
    try:
        ledger = _Ledger(maximum_read_bytes, 0, maximum_control_bytes, check_budget)
        plan, evidence, contract = _inputs(
            plan, expected_plan_sha256, kernel_evidence, kernel_io_contract,
            expected_kernel_io_contract_sha256, ledger)
        tree = _Tree(output, ledger)
        receipt = _verify(tree, plan, evidence, contract, expected_manifest_sha256)
        result = _result(receipt, expected_manifest_sha256, ledger)
    except CheckpointPackageError:
        raise
    except BaseException:
        raise CheckpointPackageError("package_failed") from None
    finally:
        if tree is not None:
            tree.close()
    ledger.tick()
    return result
