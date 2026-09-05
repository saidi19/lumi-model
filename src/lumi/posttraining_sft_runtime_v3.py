"""Hardened assistant-only Lumi SFT runtime and resumable CPU correctness canary."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import struct
import sys
import time
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import torch

from .modeling_lumi import LumiForCausalLM
from .pilot_training import _create_optimizer, scheduled_learning_rate
from .posttraining_shards import SFT_SHARDS_SCHEMA_VERSION

SFT_CANARY_STATE_SCHEMA_VERSION = "lumi.sft-cpu-canary-state.v0.3"
SFT_CANARY_CHECKPOINT_SCHEMA_VERSION = "lumi.sft-cpu-canary-checkpoint.v0.3"
SFT_CANARY_STATE_FILENAME = "sft-canary-state.json"
SFT_CANARY_METRICS_FILENAME = "sft-canary-metrics.jsonl"
SFT_CANARY_STARTED_FILENAME = "sft-canary-started.json"
SFT_CANARY_RESUME_STARTED_FILENAME = "sft-canary-resume-started.json"

_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_private_regular(path: Path) -> os.stat_result:
    info = path.lstat()
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != _PRIVATE_FILE_MODE
        or info.st_uid != os.geteuid()
        or info.st_nlink != 1
    ):
        raise ValueError("SFT canary private file metadata is unsafe")
    return info


def _ensure_private_directory(path: Path) -> None:
    """Create missing components without accepting a symlink in the path."""

    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            info = current.lstat()
        except FileNotFoundError:
            current.mkdir(mode=_PRIVATE_DIRECTORY_MODE)
            info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("SFT canary directory path is unsafe")
    final = absolute.lstat()
    if (
        stat.S_IMODE(final.st_mode) != _PRIVATE_DIRECTORY_MODE
        or final.st_uid != os.geteuid()
    ):
        raise ValueError("SFT canary output directory is not private")


def _harden_private_tree(root: Path) -> None:
    """Normalize and verify a newly-created checkpoint before publication."""

    root.chmod(_PRIVATE_DIRECTORY_MODE)
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("SFT canary checkpoint contains a symlink")
        if stat.S_ISDIR(info.st_mode):
            path.chmod(_PRIVATE_DIRECTORY_MODE)
        elif stat.S_ISREG(info.st_mode):
            path.chmod(_PRIVATE_FILE_MODE)
            _require_private_regular(path)
        else:
            raise ValueError("SFT canary checkpoint contains a special file")


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, _PRIVATE_FILE_MODE)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), _PRIVATE_FILE_MODE)
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _require_private_regular(temporary)
    if os.path.lexists(path):
        _require_private_regular(path)
    os.replace(temporary, path)
    _require_private_regular(path)


def _write_json_exclusive(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, _PRIVATE_FILE_MODE)
    try:
        os.fchmod(descriptor, _PRIVATE_FILE_MODE)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("short SFT canary marker write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _require_private_regular(path)


def _append_jsonl(path: Path, value: Any) -> None:
    flags = (
        os.O_WRONLY
        | os.O_APPEND
        | os.O_CREAT
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, _PRIVATE_FILE_MODE)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        info = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_nlink != 1
        ):
            raise ValueError("SFT canary metrics metadata is unsafe")
        os.fchmod(handle.fileno(), _PRIVATE_FILE_MODE)
        handle.write(_canonical_json(value).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_checksum_manifest(root: Path) -> str:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.sha256":
            entries.append((_sha256_file(path), path.relative_to(root).as_posix()))
    if not entries:
        raise RuntimeError("refusing to publish an empty SFT checkpoint")
    path = root / "manifest.sha256"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, _PRIVATE_FILE_MODE)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), _PRIVATE_FILE_MODE)
        for digest, relative in entries:
            handle.write(f"{digest}  {relative}\n")
        handle.flush()
        os.fsync(handle.fileno())
    _require_private_regular(path)
    return _sha256_file(path)


def _verify_checksum_manifest(
    root: Path,
    *,
    expected_manifest_sha256: str | None = None,
    require_exact_inventory: bool = True,
    require_private: bool = False,
) -> str:
    manifest = root / "manifest.sha256"
    if not root.is_dir() or not manifest.is_file():
        raise ValueError(f"checkpoint checksum manifest is missing: {root}")
    if require_private:
        root_info = root.lstat()
        if (
            stat.S_ISLNK(root_info.st_mode)
            or not stat.S_ISDIR(root_info.st_mode)
            or stat.S_IMODE(root_info.st_mode) != _PRIVATE_DIRECTORY_MODE
            or root_info.st_uid != os.geteuid()
        ):
            raise ValueError("SFT canary checkpoint directory is not private")
        _require_private_regular(manifest)
    observed_manifest_sha256 = _sha256_file(manifest)
    if (
        expected_manifest_sha256 is not None
        and observed_manifest_sha256 != expected_manifest_sha256
    ):
        raise ValueError("checkpoint manifest SHA-256 changed")
    declared = {"manifest.sha256"}
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise ValueError(f"invalid checkpoint manifest line {line_number}") from error
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"invalid checkpoint digest at line {line_number}")
        relative_path = Path(relative)
        path = root / relative_path
        if (
            relative_path.is_absolute()
            or not path.resolve().is_relative_to(root.resolve())
            or not path.is_file()
        ):
            raise ValueError(f"checkpoint manifest path is invalid: {relative}")
        if require_private:
            _require_private_regular(path)
        if _sha256_file(path) != digest:
            raise ValueError(f"checkpoint file hash changed: {relative}")
        declared.add(relative_path.as_posix())
    if len(declared) < 3:
        raise ValueError("checkpoint manifest is unexpectedly small")
    if require_exact_inventory:
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        if actual != declared:
            raise ValueError("checkpoint contains undeclared files")
    return observed_manifest_sha256


@dataclass(frozen=True)
class SFTTensorExample:
    input_ids: torch.LongTensor
    labels: torch.LongTensor
    supervised_tokens: int


@dataclass(frozen=True)
class _RecordLocation:
    labels_path: Path
    offset_tokens: int
    supervised_tokens: int
    token_count: int
    tokens_path: Path


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_INDEX_FIELDS = frozenset(
    {
        "offset_tokens",
        "record_id_sha256",
        "record_ordinal",
        "supervised_tokens",
        "tokens",
    }
)
_LEGACY_MANIFEST_FIELDS = frozenset(
    {
        "context_window",
        "encoded_dataset_sha256",
        "format",
        "manifest_sha256",
        "maximum_tokens_per_shard",
        "records",
        "schema_version",
        "seal_sha256",
        "shards",
        "splits",
        "supervised_tokens",
        "tokens",
    }
)
_TOKEN_CONSISTENT_MANIFEST_FIELDS = _LEGACY_MANIFEST_FIELDS | frozenset(
    {
        "authorization_boundary",
        "production_eligible",
        "record_hash_sequence_sha256",
    }
)
_LEGACY_SHARD_FORMAT = {
    "index": "canonical-jsonl-v0.1",
    "labels": "little-endian-int32-ignore-index-minus-100",
    "raw_text_retained": False,
    "tokens": "little-endian-uint32",
}
_TOKEN_CONSISTENT_SHARD_FORMAT = {
    **_LEGACY_SHARD_FORMAT,
    "index": "canonical-jsonl-record-hash-only-v0.1",
}
_TOKEN_CONSISTENT_AUTHORIZATION_BOUNDARY = {
    "candidate_or_source_files_opened": False,
    "encoded_shards_created": True,
    "gpu_inference_or_training_used": False,
    "input_or_predecessor_mutated": False,
    "network_or_remote_access_used": False,
    "production_dataset_seal": False,
    "production_eligible": False,
    "registry_or_contract_changed": False,
}
_MAXIMUM_MANIFEST_BYTES = 16 * 1024 * 1024
_MAXIMUM_LEGACY_INDEX_ROW_BYTES = 1024 * 1024
_MAXIMUM_TOKEN_CONSISTENT_INDEX_ROW_BYTES = 175
_MAXIMUM_TOKEN_CONSISTENT_RECORD_TOKENS = 4095
_MAXIMUM_BOUNDED_PREFIX_RECORDS = 12


@dataclass(frozen=True)
class _SecureFileIdentity:
    changed_ns: int
    device: int
    inode: int
    mode: int
    modified_ns: int
    links: int
    owner: int
    size: int


def _secure_file_identity(info: os.stat_result) -> _SecureFileIdentity:
    return _SecureFileIdentity(
        changed_ns=info.st_ctime_ns,
        device=info.st_dev,
        inode=info.st_ino,
        mode=stat.S_IMODE(info.st_mode),
        modified_ns=info.st_mtime_ns,
        links=info.st_nlink,
        owner=info.st_uid,
        size=info.st_size,
    )


def _metadata_inventory_record(
    relative_path: str,
    identity: _SecureFileIdentity,
) -> dict[str, Any]:
    """Return the exact lstat/fstat projection bound by the v0.2 preflight."""

    return {
        "mode": f"{identity.mode:04o}",
        "relative_path": relative_path,
        "st_ctime_ns": identity.changed_ns,
        "st_dev": identity.device,
        "st_ino": identity.inode,
        "st_mtime_ns": identity.modified_ns,
        "st_nlink": identity.links,
        "st_size": identity.size,
        "st_uid": identity.owner,
    }


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _secure_open_flags(*, directory: bool = False) -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("secure SFT shard loading requires O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    if directory:
        flags |= getattr(os, "O_DIRECTORY", 0)
    return flags


def _open_secure_root(path: Path) -> tuple[Path, int, _SecureFileIdentity]:
    root = _absolute_lexical(path)
    descriptor = os.open(root.anchor, _secure_open_flags(directory=True))
    try:
        for component in root.parts[1:]:
            next_descriptor = os.open(
                component,
                _secure_open_flags(directory=True),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or opened.st_uid != os.geteuid()
        ):
            raise ValueError("encoded SFT shard directory metadata is unsafe")
        return root, descriptor, _secure_file_identity(opened)
    except OSError as error:
        os.close(descriptor)
        raise ValueError(
            "encoded SFT shard directory path is not a real directory chain"
        ) from error
    except BaseException:
        os.close(descriptor)
        raise


def _finish_secure_root(
    root: Path,
    descriptor: int,
    initial: _SecureFileIdentity,
) -> None:
    after = os.fstat(descriptor)
    final = root.lstat()
    if (
        not stat.S_ISDIR(after.st_mode)
        or stat.S_ISLNK(final.st_mode)
        or not stat.S_ISDIR(final.st_mode)
        or _secure_file_identity(after) != initial
        or _secure_file_identity(final) != initial
    ):
        raise ValueError("encoded SFT shard directory changed during loading")


def _open_secure_dataset_file(
    directory_descriptor: int,
    name: str,
    *,
    require_private_mode: bool,
) -> tuple[BinaryIO, _SecureFileIdentity]:
    if Path(name).name != name or not name:
        raise ValueError("encoded SFT shard filename is invalid")
    descriptor = os.open(
        name,
        _secure_open_flags(),
        dir_fd=directory_descriptor,
    )
    try:
        opened = os.fstat(descriptor)
        identity = _secure_file_identity(opened)
        path_info = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or identity.links != 1
            or identity.owner != os.geteuid()
            or (require_private_mode and identity.mode != 0o600)
            or _secure_file_identity(path_info) != identity
        ):
            raise ValueError("encoded SFT shard file metadata is unsafe")
        return os.fdopen(descriptor, "rb", buffering=0), identity
    except BaseException:
        os.close(descriptor)
        raise


def _finish_secure_dataset_file(
    handle: BinaryIO,
    directory_descriptor: int,
    name: str,
    initial: _SecureFileIdentity,
) -> None:
    after = _secure_file_identity(os.fstat(handle.fileno()))
    try:
        final = _secure_file_identity(
            os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        )
    finally:
        handle.close()
    if after != initial or final != initial:
        raise ValueError("encoded SFT shard file changed during loading")


def _reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("encoded SFT shard JSON contains a duplicate key")
        result[key] = value
    return result


def _strict_json_bytes(payload: bytes) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except UnicodeDecodeError as error:
        raise ValueError("encoded SFT shard JSON is not UTF-8") from error


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _read_exact_and_hash(
    handle: BinaryIO,
    byte_count: int,
    digest: Any,
    retained: bytearray | None,
) -> bytes:
    chunks: list[bytes] = []
    remaining = byte_count
    while remaining:
        chunk = handle.read(remaining)
        if not chunk:
            raise ValueError("encoded SFT binary shard ended before its index")
        chunks.append(chunk)
        digest.update(chunk)
        if retained is not None:
            retained.extend(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _encoded_hash_payload(
    record_id_sha256: str,
    tokens: Sequence[int],
    labels: Sequence[int],
) -> bytes:
    return _canonical_json(
        {
            "input_ids": list(tokens),
            "labels": list(labels),
            "record_id_sha256": record_id_sha256,
        }
    )


def _manifest_profile(manifest: Any) -> str:
    if not isinstance(manifest, Mapping):
        raise ValueError("encoded SFT shard manifest must be an object")
    fields = frozenset(manifest)
    if fields == _TOKEN_CONSISTENT_MANIFEST_FIELDS:
        profile = "token_consistent_v0.2"
        if (
            manifest.get("format") != _TOKEN_CONSISTENT_SHARD_FORMAT
            or manifest.get("authorization_boundary")
            != _TOKEN_CONSISTENT_AUTHORIZATION_BOUNDARY
            or manifest.get("production_eligible") is not False
            or manifest.get("context_window") != 4096
            or manifest.get("maximum_tokens_per_shard") != 4_194_304
        ):
            raise ValueError("token-consistent SFT shard manifest contract changed")
        _require_sha256(
            manifest.get("record_hash_sequence_sha256"),
            label="encoded SFT record-hash sequence",
        )
    elif fields == _LEGACY_MANIFEST_FIELDS:
        profile = "legacy_v0.1"
        if manifest.get("format") != _LEGACY_SHARD_FORMAT:
            raise ValueError("legacy SFT shard manifest format changed")
    else:
        raise ValueError("encoded SFT shard manifest field inventory changed")
    if manifest.get("schema_version") != SFT_SHARDS_SCHEMA_VERSION:
        raise ValueError("unsupported encoded SFT shard schema")
    _require_sha256(
        manifest.get("encoded_dataset_sha256"),
        label="encoded SFT dataset hash",
    )
    _require_sha256(manifest.get("seal_sha256"), label="encoded SFT seal hash")
    for field in (
        "context_window",
        "maximum_tokens_per_shard",
        "records",
        "supervised_tokens",
        "tokens",
    ):
        if type(manifest.get(field)) is not int or manifest[field] < 1:
            raise ValueError(f"encoded SFT manifest {field} is invalid")
    if manifest["supervised_tokens"] > manifest["tokens"]:
        raise ValueError("encoded SFT supervised-token total is invalid")
    return profile


def _load_verified_sft_split(
    root: Path,
    *,
    split: str,
    record_limit: int | None,
    expected_manifest_sha256: str | None,
    expected_manifest_file_sha256: str | None,
    expected_inventory_sha256: str | None,
    expected_metadata_inventory_sha256: str | None,
    expected_seal_sha256: str | None,
) -> tuple[
    dict[str, Any],
    list[_RecordLocation],
    dict[Path, torch.Tensor],
    list[bytearray],
    dict[str, Any],
]:
    if sys.byteorder != "little":
        raise RuntimeError("encoded SFT shards require a little-endian runtime")
    for value, label in (
        (expected_manifest_sha256, "expected encoded SFT manifest hash"),
        (expected_manifest_file_sha256, "expected encoded SFT manifest file hash"),
        (expected_inventory_sha256, "expected encoded SFT inventory hash"),
        (
            expected_metadata_inventory_sha256,
            "expected encoded SFT metadata inventory hash",
        ),
        (expected_seal_sha256, "expected encoded SFT seal hash"),
    ):
        if value is not None:
            _require_sha256(value, label=label)
    if record_limit is not None and (
        type(record_limit) is not int
        or not 1 <= record_limit <= _MAXIMUM_BOUNDED_PREFIX_RECORDS
    ):
        raise ValueError("encoded SFT record limit must be an integer from 1 through 12")

    root, directory_descriptor, root_identity = _open_secure_root(root)
    locations: list[_RecordLocation] = []
    buffers: dict[Path, torch.Tensor] = {}
    retained_buffers: list[bytearray] = []
    open_handles: list[BinaryIO] = []
    root_finished = False
    try:
        manifest_handle, manifest_identity = _open_secure_dataset_file(
            directory_descriptor,
            "manifest.json",
            require_private_mode=False,
        )
        open_handles.append(manifest_handle)
        manifest_payload = manifest_handle.read(_MAXIMUM_MANIFEST_BYTES + 1)
        if len(manifest_payload) > _MAXIMUM_MANIFEST_BYTES:
            raise ValueError("encoded SFT shard manifest exceeds its byte limit")
        manifest_file_sha256 = hashlib.sha256(manifest_payload).hexdigest()
        manifest = _strict_json_bytes(manifest_payload)
        if not isinstance(manifest, dict) or _canonical_json(manifest) + b"\n" != manifest_payload:
            raise ValueError("encoded SFT shard manifest is not canonical JSON")
        profile = _manifest_profile(manifest)
        strict_private = profile == "token_consistent_v0.2"
        bounded_prefix = record_limit is not None
        if strict_private and any(
            value is None
            for value in (
                expected_manifest_sha256,
                expected_manifest_file_sha256,
                expected_inventory_sha256,
                expected_metadata_inventory_sha256,
                expected_seal_sha256,
            )
        ):
            raise ValueError(
                "token-consistent SFT loading requires five exact external bindings"
            )
        if strict_private and (
            root_identity.mode != 0o700 or manifest_identity.mode != 0o600
        ):
            raise ValueError("token-consistent SFT shards are not private mode 0700/0600")
        unsigned_manifest = dict(manifest)
        manifest_sha256 = unsigned_manifest.pop("manifest_sha256", None)
        _require_sha256(manifest_sha256, label="encoded SFT manifest hash")
        if manifest_sha256 != _sha256(unsigned_manifest):
            raise ValueError("encoded SFT shard manifest hash changed")
        if (
            expected_manifest_sha256 is not None
            and manifest_sha256 != expected_manifest_sha256
        ):
            raise ValueError("encoded SFT manifest does not match its expected hash")
        if (
            expected_manifest_file_sha256 is not None
            and manifest_file_sha256 != expected_manifest_file_sha256
        ):
            raise ValueError("encoded SFT manifest file does not match its expected hash")
        if expected_seal_sha256 is not None and manifest.get("seal_sha256") != expected_seal_sha256:
            raise ValueError("encoded SFT manifest does not match its expected seal")
        inventory_sha256 = _sha256(manifest.get("shards"))
        if (
            expected_inventory_sha256 is not None
            and inventory_sha256 != expected_inventory_sha256
        ):
            raise ValueError("encoded SFT inventory does not match its expected hash")
        _finish_secure_dataset_file(
            manifest_handle,
            directory_descriptor,
            "manifest.json",
            manifest_identity,
        )
        open_handles.remove(manifest_handle)
        metadata_inventory = [
            _metadata_inventory_record("manifest.json", manifest_identity)
        ]

        shards = manifest.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError("encoded SFT shard inventory is empty")

        # Validate the complete metadata inventory before opening or reading any
        # selected shard content. The manifest itself is the only content needed
        # to enumerate this closed inventory, and its exact file/content hashes
        # have already been checked above.
        declared_files = {"manifest.json"}
        prechecked_identities: dict[str, _SecureFileIdentity] = {}
        precheck_seen_shards: set[tuple[str, int]] = set()
        for shard in shards:
            if not isinstance(shard, Mapping) or set(shard) != {
                "files",
                "index",
                "records",
                "split",
                "supervised_tokens",
                "tokens",
            }:
                raise ValueError("encoded SFT shard inventory entry changed")
            shard_split = shard.get("split")
            shard_index = shard.get("index")
            if (
                shard_split not in {"train", "validation"}
                or type(shard_index) is not int
                or shard_index < 0
                or (shard_split, shard_index) in precheck_seen_shards
            ):
                raise ValueError("encoded SFT shard identity is invalid or duplicated")
            precheck_seen_shards.add((shard_split, shard_index))
            for field in ("records", "supervised_tokens", "tokens"):
                if type(shard.get(field)) is not int or shard[field] < 1:
                    raise ValueError(f"encoded SFT shard {field} is invalid")
            if shard["supervised_tokens"] > shard["tokens"]:
                raise ValueError("encoded SFT shard supervision total is invalid")

            prefix = f"{shard_split}-{shard_index:05d}"
            expected_names = {
                f"{prefix}.tokens.u32le",
                f"{prefix}.labels.i32le",
                f"{prefix}.index.jsonl",
            }
            file_records = shard.get("files")
            if not isinstance(file_records, list) or len(file_records) != 3:
                raise ValueError("encoded SFT shard must declare exactly three files")
            by_name: dict[str, Mapping[str, Any]] = {}
            for record in file_records:
                if (
                    not isinstance(record, Mapping)
                    or set(record) != {"bytes", "path", "sha256"}
                    or not isinstance(record.get("path"), str)
                    or record["path"] in by_name
                    or type(record.get("bytes")) is not int
                    or record["bytes"] < 1
                ):
                    raise ValueError("encoded SFT shard file inventory changed")
                _require_sha256(record.get("sha256"), label="encoded SFT shard file hash")
                by_name[record["path"]] = record
            if set(by_name) != expected_names:
                raise ValueError("encoded SFT shard filenames changed")
            if declared_files.intersection(expected_names):
                raise ValueError("encoded SFT shard filename is duplicated")
            declared_files.update(expected_names)
            for name in sorted(expected_names):
                info = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                identity = _secure_file_identity(info)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or identity.links != 1
                    or identity.owner != os.geteuid()
                    or (strict_private and identity.mode != 0o600)
                    or identity.size != by_name[name]["bytes"]
                ):
                    raise ValueError("encoded SFT shard file metadata is unsafe")
                prechecked_identities[name] = identity
                metadata_inventory.append(_metadata_inventory_record(name, identity))

        if set(os.listdir(directory_descriptor)) != declared_files:
            raise ValueError("encoded SFT shard directory contains undeclared entries")
        metadata_inventory.sort(key=lambda item: item["relative_path"])
        metadata_inventory_sha256 = _sha256(metadata_inventory)
        if (
            expected_metadata_inventory_sha256 is not None
            and metadata_inventory_sha256 != expected_metadata_inventory_sha256
        ):
            raise ValueError("encoded SFT metadata inventory changed")

        seen_shards: set[tuple[str, int]] = set()
        total_records = 0
        total_tokens = 0
        total_supervised = 0
        total_masked = 0
        split_records: Counter[str] = Counter()
        split_tokens: Counter[str] = Counter()
        split_supervised: Counter[str] = Counter()
        encoded_hasher = hashlib.sha256()
        record_hash_sequence_hasher = hashlib.sha256()
        selected_prefix_hasher = hashlib.sha256()
        file_opens = 1
        index_rows = 0
        loaded_records = 0
        opened_shards = 0
        binary_bytes_read = 0
        index_bytes_read = 0
        fully_hashed_shards = 0

        for shard in shards:
            if not isinstance(shard, Mapping) or set(shard) != {
                "files",
                "index",
                "records",
                "split",
                "supervised_tokens",
                "tokens",
            }:
                raise ValueError("encoded SFT shard inventory entry changed")
            shard_split = shard.get("split")
            shard_index = shard.get("index")
            if (
                shard_split not in {"train", "validation"}
                or type(shard_index) is not int
                or shard_index < 0
                or (shard_split, shard_index) in seen_shards
            ):
                raise ValueError("encoded SFT shard identity is invalid or duplicated")
            seen_shards.add((shard_split, shard_index))
            for field in ("records", "supervised_tokens", "tokens"):
                if type(shard.get(field)) is not int or shard[field] < 1:
                    raise ValueError(f"encoded SFT shard {field} is invalid")
            if shard["supervised_tokens"] > shard["tokens"]:
                raise ValueError("encoded SFT shard supervision total is invalid")

            prefix = f"{shard_split}-{shard_index:05d}"
            names = {
                "tokens": f"{prefix}.tokens.u32le",
                "labels": f"{prefix}.labels.i32le",
                "index": f"{prefix}.index.jsonl",
            }
            file_records = shard.get("files")
            if not isinstance(file_records, list) or len(file_records) != 3:
                raise ValueError("encoded SFT shard must declare exactly three files")
            by_name: dict[str, Mapping[str, Any]] = {}
            for record in file_records:
                if (
                    not isinstance(record, Mapping)
                    or set(record) != {"bytes", "path", "sha256"}
                    or not isinstance(record.get("path"), str)
                    or record["path"] in by_name
                    or type(record.get("bytes")) is not int
                    or record["bytes"] < 1
                ):
                    raise ValueError("encoded SFT shard file inventory changed")
                _require_sha256(record.get("sha256"), label="encoded SFT shard file hash")
                by_name[record["path"]] = record
            if set(by_name) != set(names.values()):
                raise ValueError("encoded SFT shard filenames changed")
            declared_files.update(names.values())

            rows_to_read = shard["records"]
            if bounded_prefix:
                if shard_split != split or loaded_records >= record_limit:
                    rows_to_read = 0
                else:
                    rows_to_read = min(shard["records"], record_limit - loaded_records)
            if rows_to_read == 0:
                for name in names.values():
                    info = os.stat(
                        name,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                    identity = _secure_file_identity(info)
                    if (
                        not stat.S_ISREG(info.st_mode)
                        or identity.links != 1
                        or identity.owner != os.geteuid()
                        or (strict_private and identity.mode != 0o600)
                        or identity.size != by_name[name]["bytes"]
                        or identity != prechecked_identities[name]
                    ):
                        raise ValueError("encoded SFT shard file metadata is unsafe")
                total_records += shard["records"]
                total_tokens += shard["tokens"]
                total_supervised += shard["supervised_tokens"]
                split_records[str(shard_split)] += shard["records"]
                split_tokens[str(shard_split)] += shard["tokens"]
                split_supervised[str(shard_split)] += shard["supervised_tokens"]
                continue

            opened: dict[str, tuple[BinaryIO, _SecureFileIdentity]] = {}
            try:
                for role in ("tokens", "labels", "index"):
                    handle, identity = _open_secure_dataset_file(
                        directory_descriptor,
                        names[role],
                        require_private_mode=strict_private,
                    )
                    opened[role] = (handle, identity)
                    open_handles.append(handle)
                    if (
                        identity.size != by_name[names[role]]["bytes"]
                        or identity != prechecked_identities[names[role]]
                    ):
                        raise ValueError("encoded SFT shard file size changed")
                file_opens += 3
                opened_shards += 1
                token_handle, token_identity = opened["tokens"]
                label_handle, label_identity = opened["labels"]
                index_handle, index_identity = opened["index"]
                token_digest = hashlib.sha256()
                label_digest = hashlib.sha256()
                index_digest = hashlib.sha256()
                retain = shard_split == split
                token_buffer = bytearray() if retain else None
                label_buffer = bytearray() if retain else None
                shard_records = 0
                shard_tokens = 0
                shard_supervised = 0
                shard_masked = 0
                maximum_row_bytes = (
                    _MAXIMUM_TOKEN_CONSISTENT_INDEX_ROW_BYTES
                    if strict_private
                    else _MAXIMUM_LEGACY_INDEX_ROW_BYTES
                )

                while shard_records < rows_to_read:
                    line = index_handle.readline(maximum_row_bytes + 1)
                    if not line:
                        raise ValueError("encoded SFT index ended before its declared rows")
                    index_rows += 1
                    index_bytes_read += len(line)
                    index_digest.update(line)
                    if not line.endswith(b"\n") or len(line) > maximum_row_bytes:
                        raise ValueError("encoded SFT index row framing changed")
                    entry = _strict_json_bytes(line)
                    if (
                        not isinstance(entry, Mapping)
                        or set(entry) != _INDEX_FIELDS
                        or _canonical_json(entry) + b"\n" != line
                    ):
                        raise ValueError("encoded SFT index row contract changed")
                    if (
                        type(entry.get("record_ordinal")) is not int
                        or entry["record_ordinal"] != shard_records
                        or type(entry.get("offset_tokens")) is not int
                        or entry["offset_tokens"] != shard_tokens
                    ):
                        raise ValueError("encoded SFT shard index is not contiguous")
                    length = entry.get("tokens")
                    supervised = entry.get("supervised_tokens")
                    record_hash = _require_sha256(
                        entry.get("record_id_sha256"),
                        label="encoded SFT record identity hash",
                    )
                    if (
                        type(length) is not int
                        or length < 1
                        or length
                        > (
                            _MAXIMUM_TOKEN_CONSISTENT_RECORD_TOKENS
                            if strict_private
                            else manifest["context_window"]
                        )
                        or type(supervised) is not int
                        or supervised < 1
                        or supervised > length
                    ):
                        raise ValueError("encoded SFT record counts are invalid")
                    if strict_private and (supervised < 2 or supervised >= length):
                        raise ValueError("token-consistent SFT record mask count changed")

                    token_payload = _read_exact_and_hash(
                        token_handle,
                        length * 4,
                        token_digest,
                        token_buffer,
                    )
                    label_payload = _read_exact_and_hash(
                        label_handle,
                        length * 4,
                        label_digest,
                        label_buffer,
                    )
                    binary_bytes_read += len(token_payload) + len(label_payload)
                    tokens = struct.unpack(f"<{length}I", token_payload)
                    labels = struct.unpack(f"<{length}i", label_payload)
                    observed_supervised = 0
                    observed_masked = 0
                    for token, label in zip(tokens, labels, strict=True):
                        if strict_private and token > 65_535:
                            raise ValueError("token-consistent SFT token exceeds frozen vocabulary")
                        if label == -100:
                            observed_masked += 1
                        elif label == token:
                            observed_supervised += 1
                        else:
                            raise ValueError("encoded SFT unmasked label differs from input token")
                    if observed_supervised != supervised or observed_masked < 1:
                        raise ValueError("encoded SFT record mask count changed")
                    encoded_payload = _encoded_hash_payload(record_hash, tokens, labels)
                    if strict_private:
                        encoded_hasher.update(len(encoded_payload).to_bytes(8, "big"))
                        encoded_hasher.update(encoded_payload)
                        record_hash_sequence_hasher.update(
                            f"{shard_split}\0{record_hash}\n".encode()
                        )
                    if retain:
                        selected_prefix_hasher.update(
                            len(encoded_payload).to_bytes(8, "big")
                        )
                        selected_prefix_hasher.update(encoded_payload)
                        tokens_path = root / names["tokens"]
                        labels_path = root / names["labels"]
                        locations.append(
                            _RecordLocation(
                                labels_path=labels_path,
                                offset_tokens=entry["offset_tokens"],
                                supervised_tokens=supervised,
                                token_count=length,
                                tokens_path=tokens_path,
                            )
                        )
                    shard_records += 1
                    shard_tokens += length
                    shard_supervised += supervised
                    shard_masked += observed_masked
                    loaded_records += 1

                complete_shard = rows_to_read == shard["records"]
                if complete_shard and (
                    token_handle.tell() != token_identity.size
                    or label_handle.tell() != label_identity.size
                    or index_handle.tell() != index_identity.size
                ):
                    raise ValueError("encoded SFT shard contains bytes beyond its index")
                observed_files = {
                    names["tokens"]: token_digest.hexdigest(),
                    names["labels"]: label_digest.hexdigest(),
                    names["index"]: index_digest.hexdigest(),
                }
                if complete_shard and any(
                    observed_hash != by_name[name]["sha256"]
                    for name, observed_hash in observed_files.items()
                ):
                    raise ValueError("encoded SFT shard file hash changed")
                if complete_shard:
                    fully_hashed_shards += 1
                if complete_shard and (
                    token_identity.size != shard_tokens * 4
                    or label_identity.size != shard_tokens * 4
                    or (shard_records, shard_tokens, shard_supervised)
                    != (shard["records"], shard["tokens"], shard["supervised_tokens"])
                ):
                    raise ValueError("encoded SFT shard totals changed")

                for role in ("tokens", "labels", "index"):
                    handle, identity = opened[role]
                    _finish_secure_dataset_file(
                        handle,
                        directory_descriptor,
                        names[role],
                        identity,
                    )
                    open_handles.remove(handle)
                if retain:
                    assert token_buffer is not None and label_buffer is not None
                    tokens_path = root / names["tokens"]
                    labels_path = root / names["labels"]
                    retained_buffers.extend((token_buffer, label_buffer))
                    buffers[tokens_path] = torch.frombuffer(
                        token_buffer,
                        dtype=torch.int32,
                    )
                    buffers[labels_path] = torch.frombuffer(
                        label_buffer,
                        dtype=torch.int32,
                    )
            except BaseException:
                for handle, _identity in opened.values():
                    if not handle.closed:
                        handle.close()
                    if handle in open_handles:
                        open_handles.remove(handle)
                raise

            total_records += shard["records"] if bounded_prefix else shard_records
            total_tokens += shard["tokens"] if bounded_prefix else shard_tokens
            total_supervised += (
                shard["supervised_tokens"] if bounded_prefix else shard_supervised
            )
            total_masked += shard_masked
            split_records[str(shard_split)] += (
                shard["records"] if bounded_prefix else shard_records
            )
            split_tokens[str(shard_split)] += (
                shard["tokens"] if bounded_prefix else shard_tokens
            )
            split_supervised[str(shard_split)] += (
                shard["supervised_tokens"] if bounded_prefix else shard_supervised
            )

        actual_files = set(os.listdir(directory_descriptor))
        if actual_files != declared_files:
            raise ValueError("encoded SFT shard directory contains undeclared entries")
        metadata_inventory.sort(key=lambda item: item["relative_path"])
        metadata_inventory_sha256 = _sha256(metadata_inventory)
        if (
            expected_metadata_inventory_sha256 is not None
            and metadata_inventory_sha256 != expected_metadata_inventory_sha256
        ):
            raise ValueError("encoded SFT metadata inventory changed")
        observed_splits = {
            item: {
                "records": split_records[item],
                "supervised_tokens": split_supervised[item],
                "tokens": split_tokens[item],
            }
            for item in sorted(split_records)
        }
        if (
            (total_records, total_tokens, total_supervised)
            != (manifest["records"], manifest["tokens"], manifest["supervised_tokens"])
            or observed_splits != manifest.get("splits")
        ):
            raise ValueError("encoded SFT shard aggregate totals changed")
        if strict_private and not bounded_prefix and (
            encoded_hasher.hexdigest() != manifest["encoded_dataset_sha256"]
            or record_hash_sequence_hasher.hexdigest()
            != manifest["record_hash_sequence_sha256"]
        ):
            raise ValueError("token-consistent SFT aggregate hash changed")
        if split_records[split] < 1 or (bounded_prefix and loaded_records != record_limit):
            raise ValueError(f"SFT split is empty: {split}")

        _finish_secure_root(root, directory_descriptor, root_identity)
        root_finished = True
        audit = {
            "all_inventory_file_hashes_verified": not bounded_prefix,
            "binary_bytes_read": binary_bytes_read,
            "binary_files_read_once": opened_shards * 2,
            "encoded_index_row_reads": index_rows,
            "file_opens": file_opens,
            "fully_hashed_shards": fully_hashed_shards,
            "full_content_hashes_verified": not bounded_prefix,
            "inventory_sha256": inventory_sha256,
            "index_bytes_read": index_bytes_read,
            "manifest_file_sha256": manifest_file_sha256,
            "manifest_sha256": manifest_sha256,
            "metadata_inventory_sha256": metadata_inventory_sha256,
            "metadata_inventory_verified_before_selected_content_read": True,
            "masked_tokens": total_masked,
            "opened_shards": opened_shards,
            "profile": profile,
            "records": total_records,
            "records_loaded": len(locations),
            "selected_prefix_sha256": selected_prefix_hasher.hexdigest(),
            "supervised_tokens": total_supervised,
            "tokens": total_tokens,
            "unbuffered_selected_content_reads": True,
        }
        return manifest, locations, buffers, retained_buffers, audit
    finally:
        for handle in open_handles:
            if not handle.closed:
                handle.close()
        if not root_finished:
            # The descriptor is still closed below; never mask the primary failure.
            pass
        os.close(directory_descriptor)


class EncodedSFTDataset(torch.utils.data.Dataset[SFTTensorExample]):
    """Load one split from descriptor-bound, verified raw-text-free SFT shards.

    The v0.2 token-consistent profile is consumed in one fused pass: manifest,
    indices, and binaries are each opened once with ``O_NOFOLLOW``; the bytes
    hashed on those descriptors are retained as the tensors used by callers.
    Historical v0.1 manifests retain their public loading semantics while also
    receiving single-pass content and path-integrity validation.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        expected_inventory_sha256: str | None = None,
        expected_metadata_inventory_sha256: str | None = None,
        expected_manifest_file_sha256: str | None = None,
        expected_manifest_sha256: str | None = None,
        expected_seal_sha256: str | None = None,
        record_limit: int | None = None,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("SFT dataset split must be train or validation")
        self.root = _absolute_lexical(Path(root))
        self.split = split
        (
            self.manifest,
            self._locations,
            self._buffers,
            self._retained_buffers,
            self.verification,
        ) = _load_verified_sft_split(
            self.root,
            split=split,
            record_limit=record_limit,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_manifest_file_sha256=expected_manifest_file_sha256,
            expected_inventory_sha256=expected_inventory_sha256,
            expected_metadata_inventory_sha256=expected_metadata_inventory_sha256,
            expected_seal_sha256=expected_seal_sha256,
        )
        expected_records = (
            record_limit
            if record_limit is not None
            else self.manifest.get("splits", {}).get(split, {}).get("records", 0)
        )
        if len(self._locations) != expected_records:
            raise ValueError("SFT split record inventory changed")
        if not self._locations:
            raise ValueError(f"SFT split is empty: {split}")

    def __len__(self) -> int:
        return len(self._locations)

    def __getitem__(self, index: int) -> SFTTensorExample:
        location = self._locations[index]
        start = location.offset_tokens
        end = start + location.token_count
        input_ids = self._buffers[location.tokens_path][start:end].to(torch.long)
        labels = self._buffers[location.labels_path][start:end].to(torch.long)
        if input_ids.shape != labels.shape or input_ids.ndim != 1:
            raise ValueError("SFT encoded input and label shapes changed")
        supervised = labels.ne(-100)
        if int(supervised.sum()) != location.supervised_tokens:
            raise ValueError("SFT encoded supervision count changed")
        if not torch.equal(labels[supervised], input_ids[supervised]):
            raise ValueError("SFT unmasked labels no longer equal input tokens")
        if not bool(labels.eq(-100).any()):
            raise ValueError("SFT prompt labels are not masked")
        return SFTTensorExample(
            input_ids=input_ids,
            labels=labels,
            supervised_tokens=location.supervised_tokens,
        )

    @property
    def maximum_record_tokens(self) -> int:
        return max(location.token_count for location in self._locations)


def collate_sft_tensors(
    examples: Sequence[SFTTensorExample],
    *,
    context_window: int,
    ignore_index: int = -100,
    pad_to_multiple_of: int = 8,
    pad_token_id: int = 0,
) -> dict[str, torch.Tensor]:
    """Right-pad isolated examples while preserving assistant-only supervision."""

    if not examples:
        raise ValueError("SFT batch must contain at least one example")
    if pad_to_multiple_of < 1 or context_window < 1:
        raise ValueError("SFT padding and context limits must be positive")
    maximum = max(example.input_ids.numel() for example in examples)
    padded = ((maximum + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
    if padded > context_window:
        raise ValueError("SFT batch exceeds the configured context window")
    input_ids = torch.full((len(examples), padded), pad_token_id, dtype=torch.long)
    labels = torch.full((len(examples), padded), ignore_index, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), padded), dtype=torch.long)
    for row, example in enumerate(examples):
        length = example.input_ids.numel()
        input_ids[row, :length] = example.input_ids
        labels[row, :length] = example.labels
        attention_mask[row, :length] = 1
    if not bool(labels.ne(ignore_index).any()):
        raise ValueError("SFT batch contains no supervised tokens")
    return {
        "attention_mask": attention_mask,
        "input_ids": input_ids,
        "labels": labels,
    }


class SFTCursorPlanner:
    """Deterministic epoch shuffle whose global cursor is checkpoint sufficient."""

    def __init__(self, *, dataset_length: int, seed: int) -> None:
        if dataset_length < 1:
            raise ValueError("SFT dataset length must be positive")
        self.dataset_length = dataset_length
        self.seed = seed
        self._epoch: int | None = None
        self._permutation: torch.Tensor | None = None

    def _for_epoch(self, epoch: int) -> torch.Tensor:
        if self._epoch != epoch:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + epoch)
            self._permutation = torch.randperm(
                self.dataset_length,
                generator=generator,
                dtype=torch.int64,
            )
            self._epoch = epoch
        assert self._permutation is not None
        return self._permutation

    def indices(self, *, cursor: int, count: int) -> tuple[list[int], int]:
        if cursor < 0 or count < 1:
            raise ValueError("SFT cursor must be non-negative and count positive")
        indices: list[int] = []
        next_cursor = cursor
        while len(indices) < count:
            epoch, offset = divmod(next_cursor, self.dataset_length)
            permutation = self._for_epoch(epoch)
            take = min(count - len(indices), self.dataset_length - offset)
            indices.extend(permutation[offset : offset + take].tolist())
            next_cursor += take
        return indices, next_cursor


def verify_sft_canary_checkpoint(checkpoint: str | Path) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    _verify_checksum_manifest(checkpoint, require_private=True)
    state_path = checkpoint / SFT_CANARY_STATE_FILENAME
    if not state_path.is_file():
        raise ValueError("SFT canary checkpoint state is missing")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("checkpoint_schema_version") != SFT_CANARY_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported SFT canary checkpoint schema")
    return state


def _discover_latest_checkpoint(
    output: Path,
    *,
    run_fingerprint: str,
) -> tuple[Path, dict[str, Any]] | None:
    candidates = []
    for path in output.glob("checkpoint-*"):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            continue
        state = verify_sft_canary_checkpoint(path)
        if state.get("run_fingerprint") == run_fingerprint:
            candidates.append((int(state["step"]), path, state))
    if not candidates:
        return None
    _, path, state = max(candidates, key=lambda item: item[0])
    return path, state


def _save_checkpoint(
    *,
    output: Path,
    model: LumiForCausalLM,
    optimizer: torch.optim.Optimizer,
    state: Mapping[str, Any],
) -> Path:
    name = f"checkpoint-{state['step']:08d}"
    checkpoint = output / name
    if os.path.lexists(checkpoint):
        raise FileExistsError(f"refusing to overwrite SFT checkpoint: {checkpoint}")
    temporary = output / f".{name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(mode=_PRIVATE_DIRECTORY_MODE)
    try:
        model.save_pretrained(temporary, safe_serialization=True)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
            },
            temporary / "trainer.pt",
        )
        checkpoint_state = {
            **state,
            "checkpoint_schema_version": SFT_CANARY_CHECKPOINT_SCHEMA_VERSION,
            "latest_checkpoint": name,
            "status": "checkpoint",
        }
        _write_json_atomic(temporary / SFT_CANARY_STATE_FILENAME, checkpoint_state)
        _harden_private_tree(temporary)
        _write_checksum_manifest(temporary)
        _harden_private_tree(temporary)
        _fsync_tree(temporary)
        temporary.replace(checkpoint)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    verify_sft_canary_checkpoint(checkpoint)
    return checkpoint


def _restore_checkpoint(
    checkpoint: Path,
    *,
    learning_rate: float,
    optimizer_profile: str,
    weight_decay: float,
) -> tuple[LumiForCausalLM, torch.optim.Optimizer]:
    verify_sft_canary_checkpoint(checkpoint)
    model = LumiForCausalLM.from_pretrained(checkpoint, local_files_only=True)
    model.config._attn_implementation = "eager"
    model.to(torch.device("cpu"))
    optimizer = _create_optimizer(
        model,
        device=torch.device("cpu"),
        learning_rate=learning_rate,
        optimizer_profile=optimizer_profile,
        weight_decay=weight_decay,
    )
    payload = torch.load(checkpoint / "trainer.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["rng_state"])
    return model, optimizer


def _trim_metrics(path: Path, *, committed_step: int) -> None:
    records = []
    if path.exists():
        _require_private_regular(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if int(record["step"]) <= committed_step:
                records.append(record)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, _PRIVATE_FILE_MODE)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), _PRIVATE_FILE_MODE)
        for record in records:
            handle.write(_canonical_json(record).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _require_private_regular(temporary)
    if os.path.lexists(path):
        _require_private_regular(path)
    os.replace(temporary, path)
    _require_private_regular(path)


def run_sft_cpu_canary(
    *,
    encoded_shards: str | Path,
    source_checkpoint: str | Path,
    source_checkpoint_manifest_sha256: str,
    encoded_shards_inventory_sha256: str | None = None,
    encoded_shards_manifest_file_sha256: str | None = None,
    encoded_shards_manifest_sha256: str | None = None,
    encoded_shards_metadata_inventory_sha256: str | None = None,
    encoded_shards_record_limit: int | None = None,
    encoded_shards_seal_sha256: str | None = None,
    output: str | Path,
    target_steps: int,
    stop_after_step: int,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    maximum_gradient_norm: float = 1.0,
    warmup_steps: int = 0,
    minimum_learning_rate_ratio: float = 0.1,
    optimizer_profile: str = "grouped_adamw_beta2_0.95_v0.1",
    pad_to_multiple_of: int = 8,
    checkpoint_steps: int = 1,
    seed: int = 2026,
    resume: bool = False,
) -> dict[str, Any]:
    """Run a tiny CPU-only masked-loss canary; this cannot authorize paid SFT."""

    if target_steps < 2 or not 1 <= stop_after_step <= target_steps:
        raise ValueError("SFT CPU canary requires at least two target steps")
    if batch_size < 1 or gradient_accumulation_steps < 1 or checkpoint_steps < 1:
        raise ValueError("SFT CPU canary batch and checkpoint settings must be positive")
    if learning_rate <= 0 or weight_decay < 0 or maximum_gradient_norm <= 0:
        raise ValueError("SFT CPU canary optimizer settings are invalid")
    if warmup_steps < 0 or warmup_steps >= target_steps:
        raise ValueError("SFT CPU canary warmup does not fit its schedule")
    if len(source_checkpoint_manifest_sha256) != 64:
        raise ValueError("SFT source checkpoint manifest SHA-256 is invalid")

    encoded_shards = Path(encoded_shards)
    source_checkpoint = Path(source_checkpoint)
    output = Path(output)
    _ensure_private_directory(output)
    started_path = output / SFT_CANARY_STARTED_FILENAME
    resume_started_path = output / SFT_CANARY_RESUME_STARTED_FILENAME
    started_marker = {
        "schema_version": "lumi.sft-cpu-canary-started.v0.3",
        "status": "started_no_automatic_retry",
    }
    resume_started_marker = {
        "schema_version": "lumi.sft-cpu-canary-resume-started.v0.1",
        "status": "started_no_automatic_retry",
        "stop_after_step": stop_after_step,
        "target_steps": target_steps,
    }
    if resume:
        _require_private_regular(started_path)
        if json.loads(started_path.read_text(encoding="utf-8")) != started_marker:
            raise ValueError("SFT canary started marker changed")
        _write_json_exclusive(resume_started_path, resume_started_marker)
    else:
        if any(output.iterdir()):
            raise FileExistsError(f"SFT canary output is not empty: {output}")
        _write_json_exclusive(started_path, started_marker)
    source_manifest_sha256 = _verify_checksum_manifest(
        source_checkpoint,
        expected_manifest_sha256=source_checkpoint_manifest_sha256,
        require_private=True,
    )
    dataset = EncodedSFTDataset(
        encoded_shards,
        split="train",
        expected_inventory_sha256=encoded_shards_inventory_sha256,
        expected_manifest_file_sha256=encoded_shards_manifest_file_sha256,
        expected_manifest_sha256=encoded_shards_manifest_sha256,
        expected_metadata_inventory_sha256=(
            encoded_shards_metadata_inventory_sha256
        ),
        expected_seal_sha256=encoded_shards_seal_sha256,
        record_limit=encoded_shards_record_limit,
    )
    shard_manifest = dataset.manifest
    run_identity = {
        "data": {
            "inventory_sha256": dataset.verification["inventory_sha256"],
            "manifest_file_sha256": dataset.verification["manifest_file_sha256"],
            "manifest_sha256": shard_manifest["manifest_sha256"],
            "metadata_inventory_sha256": dataset.verification[
                "metadata_inventory_sha256"
            ],
            "read_audit": {
                field: dataset.verification[field]
                for field in (
                    "binary_bytes_read",
                    "encoded_index_row_reads",
                    "file_opens",
                    "fully_hashed_shards",
                    "index_bytes_read",
                    "metadata_inventory_verified_before_selected_content_read",
                    "opened_shards",
                    "records_loaded",
                    "unbuffered_selected_content_reads",
                )
            },
            "record_limit": encoded_shards_record_limit,
            "seal_sha256": shard_manifest["seal_sha256"],
            "selection_scope": (
                "contiguous_prefix_v0.1"
                if encoded_shards_record_limit is not None
                else "full_split_v0.1"
            ),
            "sequence_order": "seeded_epoch_permutation_v0.1",
            "selected_prefix_sha256": dataset.verification[
                "selected_prefix_sha256"
            ],
            "split": "train",
            "verification_profile": dataset.verification["profile"],
        },
        "device": "cpu",
        "objective": "assistant_and_tool_only_cross_entropy_v0.1",
        "optimizer": {
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "maximum_gradient_norm": maximum_gradient_norm,
            "minimum_learning_rate_ratio": minimum_learning_rate_ratio,
            "optimizer_profile": optimizer_profile,
            "checkpoint_steps": checkpoint_steps,
            "pad_to_multiple_of": pad_to_multiple_of,
            "schedule": "cosine",
            "target_steps": target_steps,
            "warmup_steps": warmup_steps,
            "weight_decay": weight_decay,
        },
        "schema_version": SFT_CANARY_STATE_SCHEMA_VERSION,
        "seed": seed,
        "source_checkpoint_manifest_sha256": source_manifest_sha256,
        "torch_version": torch.__version__,
    }
    run_fingerprint = _sha256(run_identity)
    state_path = output / SFT_CANARY_STATE_FILENAME
    metrics_path = output / SFT_CANARY_METRICS_FILENAME

    if state_path.exists():
        _require_private_regular(state_path)
        if not resume:
            raise FileExistsError(f"SFT canary state exists; pass resume=True: {output}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("run_fingerprint") != run_fingerprint:
            raise ValueError("SFT canary state does not match the requested run")
    else:
        if resume:
            raise ValueError("cannot resume an SFT canary without state")
        if {path.name for path in output.iterdir()} != {SFT_CANARY_STARTED_FILENAME}:
            raise FileExistsError(f"SFT canary output is not empty: {output}")
        state = {
            "best_loss": None,
            "last_loss": None,
            "latest_checkpoint": None,
            "records_seen": 0,
            "run_fingerprint": run_fingerprint,
            "run_specification": run_identity,
            "schema_version": SFT_CANARY_STATE_SCHEMA_VERSION,
            "status": "active",
            "step": 0,
            "supervised_tokens_seen": 0,
            "tokens_seen": 0,
        }
        _write_json_atomic(state_path, state)

    recovered = _discover_latest_checkpoint(output, run_fingerprint=run_fingerprint)
    if recovered is not None:
        checkpoint, checkpoint_state = recovered
        if checkpoint_state["step"] >= state["step"]:
            state = {
                **state,
                **checkpoint_state,
                "latest_checkpoint": checkpoint.name,
                "status": "active",
            }
    _trim_metrics(metrics_path, committed_step=int(state["step"]))

    if state["latest_checkpoint"] is None:
        torch.manual_seed(seed)
        model = LumiForCausalLM.from_pretrained(source_checkpoint, local_files_only=True)
        model.config._attn_implementation = "eager"
        model.to(torch.device("cpu"))
        optimizer = _create_optimizer(
            model,
            device=torch.device("cpu"),
            learning_rate=learning_rate,
            optimizer_profile=optimizer_profile,
            weight_decay=weight_decay,
        )
    else:
        model, optimizer = _restore_checkpoint(
            output / state["latest_checkpoint"],
            learning_rate=learning_rate,
            optimizer_profile=optimizer_profile,
            weight_decay=weight_decay,
        )
    if dataset.maximum_record_tokens > model.config.max_position_embeddings:
        raise ValueError("SFT canary record exceeds the source model context window")

    planner = SFTCursorPlanner(dataset_length=len(dataset), seed=seed)
    model.train()
    while state["step"] < stop_after_step:
        learning_rate_at_step = scheduled_learning_rate(
            base_learning_rate=learning_rate,
            step=state["step"] + 1,
            schedule="cosine",
            warmup_steps=warmup_steps,
            schedule_steps=target_steps,
            min_learning_rate_ratio=minimum_learning_rate_ratio,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at_step
        optimizer.zero_grad(set_to_none=True)
        cursor = int(state["records_seen"])
        microbatches = []
        step_tokens = 0
        step_supervised_tokens = 0
        for _ in range(gradient_accumulation_steps):
            indices, cursor = planner.indices(cursor=cursor, count=batch_size)
            batch = collate_sft_tensors(
                [dataset[index] for index in indices],
                context_window=model.config.max_position_embeddings,
                pad_to_multiple_of=pad_to_multiple_of,
                pad_token_id=model.config.pad_token_id,
            )
            supervised = int(batch["labels"][:, 1:].ne(-100).sum())
            if supervised < 1:
                raise ValueError("SFT canary microbatch has no shifted supervised labels")
            microbatches.append((batch, supervised))
            step_tokens += int(batch["attention_mask"].sum())
            step_supervised_tokens += supervised

        started = time.perf_counter()
        weighted_loss = 0.0
        for batch, supervised in microbatches:
            output_value = model(**batch)
            loss = output_value.loss
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("SFT CPU canary produced a non-finite loss")
            weight = supervised / step_supervised_tokens
            (loss * weight).backward()
            weighted_loss += float(loss.detach()) * weight
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            maximum_gradient_norm,
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("SFT CPU canary produced a non-finite gradient")
        optimizer.step()
        elapsed = max(time.perf_counter() - started, 1e-9)

        state["step"] += 1
        state["records_seen"] = cursor
        state["tokens_seen"] += step_tokens
        state["supervised_tokens_seen"] += step_supervised_tokens
        state["last_loss"] = weighted_loss
        state["best_loss"] = (
            weighted_loss if state["best_loss"] is None else min(state["best_loss"], weighted_loss)
        )
        metric = {
            "elapsed_seconds": elapsed,
            "gradient_norm": float(gradient_norm),
            "learning_rate": learning_rate_at_step,
            "loss": weighted_loss,
            "step": state["step"],
            "supervised_tokens": step_supervised_tokens,
            "tokens": step_tokens,
            "tokens_per_second": step_tokens / elapsed,
        }
        should_checkpoint = (
            state["step"] % checkpoint_steps == 0 or state["step"] == stop_after_step
        )
        if should_checkpoint:
            checkpoint = _save_checkpoint(
                output=output,
                model=model,
                optimizer=optimizer,
                state=state,
            )
            state["latest_checkpoint"] = checkpoint.name
        _append_jsonl(metrics_path, metric)
        _write_json_atomic(state_path, state)

    state["status"] = "complete" if state["step"] == target_steps else "paused"
    _write_json_atomic(state_path, state)
    return state
