"""Hardened registry-v0.24 private nonproduction SFT shard materialization.

This module deliberately does not reuse the historical v0.17 shard writer or
verifier.  Pass one reads each already-sealed private JSONL stream exactly once
and writes raw-text-free integer shards.  Pass two reads each encoded index and
its paired binary streams once, together, without reopening either input JSONL.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
import struct
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from . import posttraining_private_sft_encoding_preflight_v1 as encoding_preflight_v1
from .posttraining_format import ChatTemplate, encode_sft_messages, verify_chat_tokenizer
from .posttraining_gpt_oss_teacher_source_promotion_preflight_v16 import (
    PureByteLevelBPETokenizer,
    _byte_encoder,
)
from .posttraining_shards import SFT_SHARDS_SCHEMA_VERSION

PLAN_SCHEMA_VERSION = "lumi.sft-token-consistent-shards-plan.v0.2"
SEAL_SCHEMA_VERSION = "lumi.sft-learning-candidate-token-consistent-seal.v0.2"
RECEIPT_SCHEMA_VERSION = "lumi.sft-token-consistent-shards-receipt.v0.2"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-token-consistent-shards-verification.v0.2"

PLAN_STATUS = "authorized_registry_v0_24_private_nonproduction_token_consistent_shards"
SEAL_STATUS = "PASS_registry_v0_24_private_nonproduction_token_consistent_seal"
RECEIPT_STATUS = "PASS_registry_v0_24_private_nonproduction_token_consistent_shards"
VERIFICATION_STATUS = (
    "PASS_verified_registry_v0_24_private_nonproduction_token_consistent_shards"
)

EXPECTED_PLAN_PATH = Path("configs/lumi-4b-sft-token-consistent-shards-v0.2.json")
EXPECTED_OBSERVED_AT = "2026-09-04"
EXPECTED_AUTHORIZATION_TEXT = "next step shuru kor"
EXPECTED_AUTHORIZATION_BYTES = 19
EXPECTED_AUTHORIZATION_SHA256 = (
    "c570f4f34dbbb415d6b34d6df1adedf536d58044ffd37cfc4b5f089a0edd0cd7"
)
EXPECTED_AUTHORIZATION = {
    "candidate_mutation_authorized": False,
    "candidate_row_access_authorized": False,
    "encoded_output_audit_authorized": True,
    "encoded_shard_materialization_authorized": True,
    "evidence_writes_authorized": True,
    "gpu_authorized": False,
    "inference_authorized": False,
    "network_authorized": False,
    "private_input_or_predecessor_mutation_authorized": False,
    "private_jsonl_row_access_authorized": True,
    "production_dataset_seal_authorized": False,
    "production_promotion_authorized": False,
    "registry_or_contract_mutation_authorized": False,
    "remote_access_authorized": False,
    "sft_training_authorized": False,
    "source_mutation_authorized": False,
    "source_row_access_authorized": False,
    "successor_nonproduction_seal_authorized": True,
    "test_object_access_authorized": False,
    "unrelated_access_authorized": False,
}

EXPECTED_INPUT = copy.deepcopy(encoding_preflight_v1.EXPECTED_INPUT)
EXPECTED_ENCODING_CONTRACT = copy.deepcopy(
    encoding_preflight_v1.EXPECTED_ENCODING_CONTRACT
)
EXPECTED_LIMITS = copy.deepcopy(encoding_preflight_v1.EXPECTED_SUCCESSOR_LIMITS)
EXPECTED_OUTPUT = copy.deepcopy(encoding_preflight_v1.EXPECTED_SUCCESSOR_OUTPUT)
EXPECTED_PRODUCTION_BLOCKERS = copy.deepcopy(
    encoding_preflight_v1.EXPECTED_PRODUCTION_BLOCKERS
)
EXPECTED_SHARD_IDENTITIES = tuple(
    [("train", index) for index in range(9)] + [("validation", 0)]
)

EXPECTED_EXTERNAL_BINDINGS = {
    **copy.deepcopy(encoding_preflight_v1.EXPECTED_EXTERNAL_BINDINGS),
    "runtime_consumer": {
        "path": "src/lumi/posttraining_sft_runtime.py",
        "sha256": "6940238db74f7c79c06b2b4d8098314e10879601242b3987932dfda90ba0cced",
    },
    "encoding_preflight_plan": {
        "path": "configs/lumi-4b-sft-private-nonproduction-encoding-preflight-v0.1.json",
        "sha256": "b00592d849260a752f98dc2935bbca073abff1f25fcbb33c164f9387c5e2bea0",
    },
    "encoding_preflight_report": {
        "path": "evidence/lumi-4b-sft-private-nonproduction-encoding-preflight-v0.1.json",
        "sha256": "a44c827e36dbe656a3b4fd4b5fd41e722d36e3688d988b033fdb6bfeac4de169",
    },
    "encoding_preflight_verification": {
        "path": (
            "evidence/lumi-4b-sft-private-nonproduction-encoding-preflight-"
            "verification-v0.1.json"
        ),
        "sha256": "965291c9c487bbb6a36e57a6376b701ad524b08c2d1c3c9f421edc8525b096c2",
    },
    "encoding_preflight_implementation": {
        "path": "src/lumi/posttraining_private_sft_encoding_preflight_v1.py",
        "sha256": "79b4b4310583289673525bdeb41265ecea881bc6de372e2418ebfa6039d02eac",
    },
    "encoding_preflight_runner": {
        "path": "scripts/preflight_private_sft_encoding_v1.py",
        "sha256": "8716b2ea380a37fd14349d86f750ac494e3501289ec03295a6bcabedc22eb52d",
    },
    "encoding_preflight_test": {
        "path": "tests/test_private_sft_encoding_preflight_v1.py",
        "sha256": "69629217c8f3e546fcbc257b76b0f4369f03e47bb52bc4ec74bee8aa79ab07af",
    },
    "encoding_preflight_verifier": {
        "path": "scripts/verify_private_sft_encoding_preflight_v1.py",
        "sha256": "4942784b65af16ff45b55afdca6f1acd551e6dd3388426f133be87df453430d3",
    },
}
EXPECTED_CODE_PATHS = {
    "implementation": "src/lumi/posttraining_token_consistent_shards_v2.py",
    "runner": "scripts/materialize_sft_token_consistent_shards_v2.py",
    "test": "tests/test_token_consistent_shards_v2.py",
    "verifier": "scripts/verify_sft_token_consistent_shards_v2.py",
}
EXPECTED_BINDING_NAMES = frozenset(EXPECTED_EXTERNAL_BINDINGS) | frozenset(
    EXPECTED_CODE_PATHS
)
EXPECTED_OUTPUT_AUTHORIZATION_BOUNDARY = {
    "candidate_or_source_files_opened": False,
    "encoded_shards_created": True,
    "gpu_inference_or_training_used": False,
    "input_or_predecessor_mutated": False,
    "network_or_remote_access_used": False,
    "production_dataset_seal": False,
    "production_eligible": False,
    "registry_or_contract_changed": False,
}

PASS_ONE_MARKER = b'{"started":"lumi_sft_token_consistent_shards_v0.2_pass_1"}\n'
PASS_TWO_MARKER = b'{"started":"lumi_sft_token_consistent_shards_v0.2_pass_2"}\n'
EXPECTED_PASS_ONE_MARKER = PASS_ONE_MARKER
EXPECTED_PASS_TWO_MARKER = PASS_TWO_MARKER

INDEX_KEYS = frozenset(
    {
        "offset_tokens",
        "record_id_sha256",
        "record_ordinal",
        "supervised_tokens",
        "tokens",
    }
)
_PLAN_KEYS = frozenset(
    {
        "authorization",
        "authorization_bytes",
        "authorization_sha256",
        "authorization_text",
        "bindings",
        "encoding_contract",
        "input",
        "limits",
        "observed_at",
        "output",
        "production_blockers_preserved",
        "production_eligible",
        "schema_version",
        "status",
    }
)
_RAW_TEXT_KEYS = frozenset(
    {
        "assistant_response",
        "content",
        "messages",
        "prompt",
        "raw_text",
        "record_id",
        "response",
        "text",
        "user_prompt",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAXIMUM_INDEX_BYTES_PER_ROW = 175
_EVIDENCE_RESERVE_BYTES = 4_194_304
_STAGING_DIRECTORY_NAME = ".encoded.pass-1-incomplete"


class BoundTokenConsistentShardsPlan(dict[str, Any]):
    """Plan mapping carrying the SHA of the exact bytes parsed by the loader."""

    file_sha256: str

    def __init__(self, value: Mapping[str, Any], *, file_sha256: str) -> None:
        super().__init__(value)
        self.file_sha256 = file_sha256


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key in token-consistent shard input")
        result[key] = value
    return result


def strict_json_bytes(payload: bytes) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant: {value}")
            ),
        )
    except UnicodeDecodeError as exc:
        raise ValueError("token-consistent shard JSON is not UTF-8") from exc


def _strict_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_strict_equal(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _strict_equal(left, right)
                for left, right in zip(actual, expected, strict=True)
            )
        )
    if type(expected) is bool:
        return actual is expected
    if type(expected) is int:
        return type(actual) is int and actual == expected
    return type(actual) is type(expected) and actual == expected


def assert_raw_text_free(value: Any, path: str = "root") -> None:
    """Reject raw-training-text fields from manifests and evidence objects."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _RAW_TEXT_KEYS:
                raise ValueError(f"raw training text key forbidden at {path}.{key}")
            assert_raw_text_free(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_raw_text_free(item, f"{path}[{index}]")


_assert_raw_text_free = assert_raw_text_free


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _require_absolute_parent_chain_without_symlinks(path: Path) -> None:
    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:-1]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("token-consistent path parent is unsafe")


def _require_under(root: Path, path: Path, *, label: str) -> Path:
    root = _lexical_absolute(root)
    path = _lexical_absolute(path)
    if not path.is_relative_to(root):
        raise ValueError(f"{label} escaped its authorized root")
    _require_absolute_parent_chain_without_symlinks(path)
    return path


def _open_flags(base: int, *, directory: bool = False) -> int:
    flags = base
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if directory and hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    return flags


def _read_secure_file(
    path: Path,
    *,
    expected_sha256: str | None = None,
    require_private: bool = False,
) -> bytes:
    _require_absolute_parent_chain_without_symlinks(path)
    descriptor = os.open(path, _open_flags(os.O_RDONLY))
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (
                require_private
                and (
                    stat.S_IMODE(before.st_mode) != 0o600
                    or before.st_nlink != 1
                    or before.st_uid != os.geteuid()
                )
            )
        ):
            raise ValueError("token-consistent binding must be a regular file")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = path.lstat()
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if (
        stat.S_ISLNK(final.st_mode)
        or not stat.S_ISREG(final.st_mode)
        or identity
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns) != identity
        or (
            require_private
            and (
                stat.S_IMODE(final.st_mode) != 0o600
                or final.st_nlink != 1
                or final.st_uid != os.geteuid()
            )
        )
        or (expected_sha256 is not None and digest.hexdigest() != expected_sha256)
    ):
        raise ValueError("token-consistent binding hash or inode changed")
    return b"".join(chunks)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(_read_secure_file(_lexical_absolute(Path(path)))).hexdigest()


def _binding_path(root: Path, specification: Mapping[str, Any], name: str) -> Path:
    expected = EXPECTED_EXTERNAL_BINDINGS.get(name)
    expected_path = expected["path"] if expected else EXPECTED_CODE_PATHS.get(name)
    if (
        expected_path is None
        or not isinstance(specification, Mapping)
        or set(specification) != {"path", "sha256"}
        or specification.get("path") != expected_path
        or not isinstance(specification.get("sha256"), str)
        or _SHA256.fullmatch(specification["sha256"]) is None
        or (expected is not None and not _strict_equal(specification, expected))
    ):
        raise ValueError(f"token-consistent shard binding changed: {name}")
    declared = _lexical_absolute(root / expected_path)
    _read_secure_file(declared, expected_sha256=specification["sha256"])
    return declared


def _load_object(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    value = strict_json_bytes(_read_secure_file(path, expected_sha256=expected_sha256))
    if not isinstance(value, dict):
        raise ValueError("token-consistent shard JSON binding must contain an object")
    return value


def load_token_consistent_shards_plan(
    path: str | Path, *, repository_root: str | Path
) -> dict[str, Any]:
    """Load and bind the exact v0.2 plan without opening either private JSONL."""

    root = _lexical_absolute(Path(repository_root))
    plan_path = _lexical_absolute(Path(path))
    expected_path = _lexical_absolute(root / EXPECTED_PLAN_PATH)
    if plan_path != expected_path:
        raise ValueError("token-consistent shard plan path changed")
    plan_payload = _read_secure_file(plan_path)
    plan_value = strict_json_bytes(plan_payload)
    if not isinstance(plan_value, dict):
        raise ValueError("token-consistent shard plan must contain an object")
    plan = BoundTokenConsistentShardsPlan(
        plan_value,
        file_sha256=hashlib.sha256(plan_payload).hexdigest(),
    )
    if set(plan) != _PLAN_KEYS:
        raise ValueError("token-consistent shard plan fields changed")
    expected_values = {
        "authorization": EXPECTED_AUTHORIZATION,
        "authorization_bytes": EXPECTED_AUTHORIZATION_BYTES,
        "authorization_sha256": EXPECTED_AUTHORIZATION_SHA256,
        "authorization_text": EXPECTED_AUTHORIZATION_TEXT,
        "encoding_contract": EXPECTED_ENCODING_CONTRACT,
        "input": EXPECTED_INPUT,
        "limits": EXPECTED_LIMITS,
        "observed_at": EXPECTED_OBSERVED_AT,
        "output": EXPECTED_OUTPUT,
        "production_blockers_preserved": EXPECTED_PRODUCTION_BLOCKERS,
        "production_eligible": False,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": PLAN_STATUS,
    }
    if any(not _strict_equal(plan.get(key), value) for key, value in expected_values.items()):
        raise ValueError("token-consistent shard plan boundary changed")
    if (
        len(EXPECTED_AUTHORIZATION_TEXT.encode("utf-8")) != EXPECTED_AUTHORIZATION_BYTES
        or hashlib.sha256(EXPECTED_AUTHORIZATION_TEXT.encode("utf-8")).hexdigest()
        != EXPECTED_AUTHORIZATION_SHA256
    ):
        raise ValueError("token-consistent authorization binding is internally inconsistent")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != EXPECTED_BINDING_NAMES:
        raise ValueError("token-consistent shard binding set changed")
    for name, specification in bindings.items():
        _binding_path(root, specification, str(name))

    preflight = _load_object(
        _lexical_absolute(root / EXPECTED_EXTERNAL_BINDINGS["encoding_preflight_report"]["path"]),
        expected_sha256=EXPECTED_EXTERNAL_BINDINGS["encoding_preflight_report"]["sha256"],
    )
    preflight_verification = _load_object(
        _lexical_absolute(
            root / EXPECTED_EXTERNAL_BINDINGS["encoding_preflight_verification"]["path"]
        ),
        expected_sha256=EXPECTED_EXTERNAL_BINDINGS["encoding_preflight_verification"][
            "sha256"
        ],
    )
    predecessor = _load_object(
        _lexical_absolute(root / EXPECTED_EXTERNAL_BINDINGS["learning_seal"]["path"]),
        expected_sha256=EXPECTED_EXTERNAL_BINDINGS["learning_seal"]["sha256"],
    )
    if (
        preflight.get("status") != encoding_preflight_v1.REPORT_STATUS
        or preflight.get("report_sha256")
        != "36ddd84c1a8760afa4df0597a14803d9c75de717e4ab5c98d5e49e471b38080a"
        or preflight_verification.get("status")
        != encoding_preflight_v1.VERIFICATION_STATUS
        or preflight_verification.get("verification_sha256")
        != "effa7d2aed22447364211656c0f083d6e3334eb3b284a5d1c9bad5b378d151cd"
        or predecessor.get("seal_sha256") != EXPECTED_INPUT["learning_seal_internal_sha256"]
        or predecessor.get("production_blockers_preserved") != EXPECTED_PRODUCTION_BLOCKERS
    ):
        raise ValueError("token-consistent predecessor or preflight chain changed")
    assert_raw_text_free(preflight)
    assert_raw_text_free(preflight_verification)
    assert_raw_text_free(predecessor)
    return plan


def output_paths(plan: Mapping[str, Any], root: Path) -> dict[str, Path]:
    root = _lexical_absolute(root)
    if not _strict_equal(plan.get("output"), EXPECTED_OUTPUT):
        raise ValueError("token-consistent shard output paths changed")
    paths = {name: _lexical_absolute(root / value) for name, value in EXPECTED_OUTPUT.items()}
    encoded_root = _lexical_absolute(root / "artifacts/lumi-sft-token-consistent-shards-v0.2")
    evidence_root = _lexical_absolute(root / "evidence")
    _require_under(encoded_root, paths["encoded_shards_path"], label="encoded output")
    for name in ("receipt_path", "successor_seal_path", "verification_path"):
        _require_under(evidence_root, paths[name], label="evidence output")
    return paths


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, _open_flags(os.O_RDONLY, directory=True))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def start_marker(path: Path, marker: bytes) -> tuple[int, int]:
    """Durably create the no-retry marker before a pass can read any row."""

    path = _lexical_absolute(path)
    _require_absolute_parent_chain_without_symlinks(path)
    flags = _open_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    descriptor = os.open(path, flags, 0o600)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise ValueError("token-consistent marker inode changed")
        _write_all(descriptor, marker)
        os.fsync(descriptor)
        identity = (info.st_dev, info.st_ino)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return identity


def finish_marker(
    path: Path,
    marker: bytes,
    payload: bytes,
    identity: tuple[int, int],
) -> None:
    """Replace a marker in-place while requiring the exact original inode."""

    path = _lexical_absolute(path)
    flags = _open_flags(os.O_RDWR)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or (info.st_dev, info.st_ino) != identity
            or os.read(descriptor, len(marker) + 1) != marker
        ):
            raise ValueError("token-consistent marker changed")
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def secure_write_json_exclusive(path: Path, value: Mapping[str, Any]) -> int:
    """Write a raw-text-free JSON evidence object as a private exclusive file."""

    assert_raw_text_free(value)
    payload = json_file_bytes(value)
    path = _lexical_absolute(path)
    _require_absolute_parent_chain_without_symlinks(path)
    descriptor = os.open(
        path,
        _open_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL),
        0o600,
    )
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise ValueError("token-consistent evidence inode changed")
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    final = path.lstat()
    if (
        not stat.S_ISREG(final.st_mode)
        or stat.S_ISLNK(final.st_mode)
        or final.st_nlink != 1
        or (final.st_dev, final.st_ino) != (info.st_dev, info.st_ino)
    ):
        raise ValueError("token-consistent evidence path changed")
    _fsync_directory(path.parent)
    return len(payload)


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("short token-consistent shard write")
        offset += written


def _runtime_check(started: float) -> None:
    if time.monotonic() - started > EXPECTED_LIMITS["maximum_runtime_minutes"] * 60:
        raise TimeoutError("token-consistent shard runtime cap reached")


def iter_exact_bounded_lines(
    handle: BinaryIO,
    *,
    expected_rows: int,
    expected_bytes: int,
    maximum_rows: int,
    label: str,
) -> Iterator[bytes]:
    """Yield exactly the declared lines and prove no byte exists past that bound."""

    if (
        type(expected_rows) is not int
        or type(expected_bytes) is not int
        or type(maximum_rows) is not int
        or expected_rows < 0
        or expected_bytes < 0
        or maximum_rows < 0
        or expected_rows > maximum_rows
    ):
        raise ValueError(f"{label} row or byte bound changed")
    for _ in range(expected_rows):
        line = handle.readline()
        if not line:
            raise ValueError(f"{label} row is missing")
        yield line
    if handle.tell() != expected_bytes:
        raise ValueError(f"{label} has bytes beyond its row bound")


def validate_receipt_plan_hash(
    receipt: Mapping[str, Any],
    plan: Mapping[str, Any],
    expected_plan_file_sha256: str,
) -> None:
    """Bind pass two to the exact plan bytes parsed for this process."""

    if (
        not isinstance(plan, BoundTokenConsistentShardsPlan)
        or not isinstance(expected_plan_file_sha256, str)
        or _SHA256.fullmatch(expected_plan_file_sha256) is None
        or plan.file_sha256 != expected_plan_file_sha256
        or receipt.get("plan_file_sha256") != expected_plan_file_sha256
    ):
        raise ValueError("token-consistent receipt plan hash changed")


def _open_dir_child(
    parent_descriptor: int,
    name: str,
    *,
    create: bool,
    require_private: bool,
) -> int:
    if not name or name in {".", ".."} or "/" in name:
        raise ValueError("invalid token-consistent directory component")
    if create:
        try:
            os.mkdir(name, 0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            pass
    descriptor = os.open(
        name,
        _open_flags(os.O_RDONLY, directory=True),
        dir_fd=parent_descriptor,
    )
    info = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or (require_private and stat.S_IMODE(info.st_mode) != 0o700)
    ):
        os.close(descriptor)
        raise ValueError("token-consistent private directory metadata changed")
    return descriptor


def _ensure_private_output_parent(root: Path) -> tuple[int, Path]:
    root = _lexical_absolute(root)
    root_descriptor = os.open(root, _open_flags(os.O_RDONLY, directory=True))
    descriptors = [root_descriptor]
    try:
        artifacts = _open_dir_child(
            descriptors[-1], "artifacts", create=False, require_private=False
        )
        descriptors.append(artifacts)
        for part in (
            "lumi-sft-token-consistent-shards-v0.2",
            "attempt-001",
            "private",
        ):
            descriptors.append(
                _open_dir_child(
                    descriptors[-1], part, create=True, require_private=True
                )
            )
        private_descriptor = os.dup(descriptors[-1])
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    return private_descriptor, root / (
        "artifacts/lumi-sft-token-consistent-shards-v0.2/attempt-001/private"
    )


@dataclass
class _SecureFileWriter:
    parent_descriptor: int
    name: str
    descriptor: int
    identity: tuple[int, int]
    digest: Any
    bytes_written: int = 0
    closed: bool = False

    @classmethod
    def create(cls, parent_descriptor: int, name: str) -> _SecureFileWriter:
        descriptor = os.open(
            name,
            _open_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL),
            0o600,
            dir_fd=parent_descriptor,
        )
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            os.close(descriptor)
            raise ValueError("token-consistent shard output inode changed")
        return cls(
            parent_descriptor=parent_descriptor,
            name=name,
            descriptor=descriptor,
            identity=(info.st_dev, info.st_ino),
            digest=hashlib.sha256(),
        )

    def write(self, payload: bytes) -> None:
        if self.closed:
            raise ValueError("token-consistent shard writer is closed")
        _write_all(self.descriptor, payload)
        self.digest.update(payload)
        self.bytes_written += len(payload)

    def close(self) -> dict[str, Any]:
        if self.closed:
            raise ValueError("token-consistent shard writer closed twice")
        os.fsync(self.descriptor)
        info = os.fstat(self.descriptor)
        os.close(self.descriptor)
        self.closed = True
        final = os.stat(self.name, dir_fd=self.parent_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(final.st_mode)
            or final.st_nlink != 1
            or final.st_uid != os.geteuid()
            or stat.S_IMODE(final.st_mode) != 0o600
            or (final.st_dev, final.st_ino) != self.identity
            or final.st_size != self.bytes_written
            or (info.st_dev, info.st_ino, info.st_size)
            != (final.st_dev, final.st_ino, final.st_size)
        ):
            raise ValueError("token-consistent shard output path changed")
        return {
            "bytes": self.bytes_written,
            "path": self.name,
            "sha256": self.digest.hexdigest(),
        }

    def abort_close(self) -> None:
        if not self.closed:
            os.close(self.descriptor)
            self.closed = True


@dataclass
class _HardenedShardWriter:
    directory_descriptor: int
    split: str
    index: int
    tokens_file: _SecureFileWriter
    labels_file: _SecureFileWriter
    index_file: _SecureFileWriter
    records: int = 0
    tokens: int = 0
    supervised_tokens: int = 0

    @classmethod
    def create(
        cls, directory_descriptor: int, *, split: str, index: int
    ) -> _HardenedShardWriter:
        prefix = f"{split}-{index:05d}"
        opened: list[_SecureFileWriter] = []
        try:
            for name in (
                f"{prefix}.tokens.u32le",
                f"{prefix}.labels.i32le",
                f"{prefix}.index.jsonl",
            ):
                opened.append(_SecureFileWriter.create(directory_descriptor, name))
        except BaseException:
            for writer in opened:
                writer.abort_close()
            raise
        return cls(
            directory_descriptor=directory_descriptor,
            split=split,
            index=index,
            tokens_file=opened[0],
            labels_file=opened[1],
            index_file=opened[2],
        )

    @property
    def prefix(self) -> str:
        return f"{self.split}-{self.index:05d}"

    def add(self, *, record_id: str, input_ids: Sequence[int], labels: Sequence[int]) -> None:
        if not input_ids or len(input_ids) != len(labels):
            raise ValueError("token-consistent token and label lengths changed")
        if any(type(token) is not int or not 0 <= token <= 65_535 for token in input_ids):
            raise ValueError("token-consistent input token is outside the frozen vocabulary")
        if any(
            type(label) is not int
            or label < -100
            or label > 65_535
            or (label != -100 and label != token)
            for token, label in zip(input_ids, labels, strict=True)
        ):
            raise ValueError("token-consistent unmasked label differs from input token")
        supervised = sum(label != -100 for label in labels)
        if supervised < 2 or supervised >= len(labels):
            raise ValueError("token-consistent record mask count changed")
        entry = {
            "offset_tokens": self.tokens,
            "record_id_sha256": hashlib.sha256(record_id.encode("utf-8")).hexdigest(),
            "record_ordinal": self.records,
            "supervised_tokens": supervised,
            "tokens": len(input_ids),
        }
        index_payload = canonical_json(entry) + b"\n"
        if len(index_payload) > _MAXIMUM_INDEX_BYTES_PER_ROW:
            raise ValueError("token-consistent index row exceeded its byte cap")
        self.tokens_file.write(struct.pack(f"<{len(input_ids)}I", *input_ids))
        self.labels_file.write(struct.pack(f"<{len(labels)}i", *labels))
        self.index_file.write(index_payload)
        self.records += 1
        self.tokens += len(input_ids)
        self.supervised_tokens += supervised

    def close(self) -> dict[str, Any]:
        files = [
            self.tokens_file.close(),
            self.labels_file.close(),
            self.index_file.close(),
        ]
        return {
            "files": files,
            "index": self.index,
            "records": self.records,
            "split": self.split,
            "supervised_tokens": self.supervised_tokens,
            "tokens": self.tokens,
        }

    def abort_close(self) -> None:
        self.tokens_file.abort_close()
        self.labels_file.abort_close()
        self.index_file.abort_close()


def _build_tokenizer_from_bytes(payload: bytes) -> PureByteLevelBPETokenizer:
    value = strict_json_bytes(payload)
    if not isinstance(value, Mapping):
        raise ValueError("frozen tokenizer must be an object")
    model = value.get("model")
    if not isinstance(model, Mapping) or model.get("type") != "BPE":
        raise ValueError("frozen tokenizer is not ByteLevel BPE")
    vocab = model.get("vocab")
    merges = model.get("merges")
    added = value.get("added_tokens")
    if not isinstance(vocab, Mapping) or not isinstance(merges, list):
        raise ValueError("frozen tokenizer vocabulary or merges changed")
    if not isinstance(added, list):
        raise ValueError("frozen tokenizer added tokens changed")
    normalized_vocab: dict[str, int] = {}
    for token, token_id in vocab.items():
        if not isinstance(token, str) or type(token_id) is not int:
            raise ValueError("frozen tokenizer vocabulary changed")
        normalized_vocab[token] = token_id
    if (
        len(normalized_vocab) != EXPECTED_ENCODING_CONTRACT["tokenizer_vocab_size"]
        or set(normalized_vocab.values())
        != set(
            range(
                EXPECTED_ENCODING_CONTRACT["tokenizer_id_minimum"],
                EXPECTED_ENCODING_CONTRACT["tokenizer_id_maximum"] + 1,
            )
        )
    ):
        raise ValueError("frozen tokenizer ID range changed")
    normalized_merges: dict[tuple[str, str], int] = {}
    for rank, pair in enumerate(merges):
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(item, str) for item in pair)
        ):
            raise ValueError("frozen tokenizer merge changed")
        normalized_merges[(pair[0], pair[1])] = rank
    special_ids: list[int] = []
    for item in added:
        if not isinstance(item, Mapping):
            raise ValueError("frozen tokenizer added token changed")
        if item.get("special") is True:
            token_id = item.get("id")
            if type(token_id) is not int:
                raise ValueError("frozen tokenizer special ID changed")
            special_ids.append(token_id)
    if len(added) != EXPECTED_ENCODING_CONTRACT["tokenizer_added_token_count"]:
        raise ValueError("frozen tokenizer added-token count changed")
    tokenizer = PureByteLevelBPETokenizer.__new__(PureByteLevelBPETokenizer)
    tokenizer._vocab = normalized_vocab
    tokenizer._merges = normalized_merges
    tokenizer._bytes = _byte_encoder()
    tokenizer._cache = {}
    tokenizer.all_special_ids = sorted(special_ids)
    return tokenizer


def _load_encoding_dependencies(
    plan: Mapping[str, Any], root: Path
) -> tuple[ChatTemplate, PureByteLevelBPETokenizer]:
    bindings = plan["bindings"]
    template_path = _lexical_absolute(root / bindings["chat_template"]["path"])
    tokenizer_path = _lexical_absolute(root / bindings["tokenizer"]["path"])
    template_value = strict_json_bytes(
        _read_secure_file(
            template_path,
            expected_sha256=bindings["chat_template"]["sha256"],
        )
    )
    if not isinstance(template_value, Mapping):
        raise ValueError("chat template must be an object")
    template = ChatTemplate.from_mapping(template_value)
    tokenizer = _build_tokenizer_from_bytes(
        _read_secure_file(
            tokenizer_path,
            expected_sha256=bindings["tokenizer"]["sha256"],
        )
    )
    verify_chat_tokenizer(tokenizer, template)
    if (
        template.context_window != EXPECTED_ENCODING_CONTRACT["context_window"]
        or template.ignore_index != EXPECTED_ENCODING_CONTRACT["label_ignore_index"]
        or sorted(template.loss_roles) != EXPECTED_ENCODING_CONTRACT["loss_roles"]
    ):
        raise ValueError("token-consistent encoding contract changed")
    return template, tokenizer


@dataclass(frozen=True)
class _InputIdentity:
    device: int
    inode: int
    mode: int
    nlink: int
    owner: int
    size: int
    modified_ns: int


def _input_identity(info: os.stat_result) -> _InputIdentity:
    return _InputIdentity(
        device=info.st_dev,
        inode=info.st_ino,
        mode=stat.S_IMODE(info.st_mode),
        nlink=info.st_nlink,
        owner=info.st_uid,
        size=info.st_size,
        modified_ns=info.st_mtime_ns,
    )


def _open_private_input_once(
    root: Path, specification: Mapping[str, Any]
) -> tuple[BinaryIO, int, Path, _InputIdentity]:
    relative = specification.get("path")
    if not isinstance(relative, str):
        raise ValueError("private input path changed")
    path = _require_under(root, root / relative, label="private input")
    descriptor = os.open(path, _open_flags(os.O_RDONLY))
    try:
        info = os.fstat(descriptor)
        identity = _input_identity(info)
        if (
            not stat.S_ISREG(info.st_mode)
            or identity.mode != 0o600
            or identity.nlink != 1
            or identity.owner != os.geteuid()
            or identity.size != specification.get("bytes")
        ):
            raise ValueError("sealed private input metadata changed")
        final = path.lstat()
        if (
            stat.S_ISLNK(final.st_mode)
            or _input_identity(final) != identity
        ):
            raise ValueError("sealed private input path changed")
        return os.fdopen(descriptor, "rb", buffering=1024 * 1024), descriptor, path, identity
    except BaseException:
        os.close(descriptor)
        raise


def _finish_private_input(
    handle: BinaryIO,
    descriptor: int,
    path: Path,
    identity: _InputIdentity,
) -> None:
    after = _input_identity(os.fstat(descriptor))
    final = path.lstat()
    if (
        after != identity
        or stat.S_ISLNK(final.st_mode)
        or _input_identity(final) != identity
    ):
        raise ValueError("sealed private input changed during encoding")
    handle.close()


def _encoded_payload_bytes(
    record_id_sha256: str, input_ids: Sequence[int], labels: Sequence[int]
) -> bytes:
    return canonical_json(
        {
            "input_ids": list(input_ids),
            "labels": list(labels),
            "record_id_sha256": record_id_sha256,
        }
    )


def _expected_observed_totals() -> dict[str, Any]:
    return {
        "maximum_record_tokens": EXPECTED_INPUT["maximum_record_tokens"],
        "records": EXPECTED_INPUT["records"],
        "train": {
            "records": EXPECTED_INPUT["splits"]["train"]["records"],
            "serialized_tokens": EXPECTED_INPUT["splits"]["train"][
                "serialized_tokens"
            ],
            "supervised_tokens": EXPECTED_INPUT["splits"]["train"][
                "supervised_tokens"
            ],
        },
        "validation": {
            "records": EXPECTED_INPUT["splits"]["validation"]["records"],
            "serialized_tokens": EXPECTED_INPUT["splits"]["validation"][
                "serialized_tokens"
            ],
            "supervised_tokens": EXPECTED_INPUT["splits"]["validation"][
                "supervised_tokens"
            ],
        },
    }


def _validate_output_absence(paths: Mapping[str, Path]) -> None:
    for path in paths.values():
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to overwrite token-consistent output: {path}")


def _validate_private_input_root(root: Path) -> None:
    private_root = _lexical_absolute(
        root / "artifacts/lumi-sft-learning-candidate-seal-v0.3/attempt-001/private"
    )
    _require_absolute_parent_chain_without_symlinks(private_root / "sentinel")
    info = private_root.lstat()
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ValueError("sealed private input root metadata changed")


def validate_private_input_metadata_without_opening_rows(root: Path) -> None:
    """Perform the complete lstat-only input gate before creating pass-one marker."""

    _validate_private_input_root(root)
    for split in ("train", "validation"):
        specification = EXPECTED_INPUT["splits"][split]
        path = _require_under(root, root / specification["path"], label="private input")
        info = path.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
            or info.st_size != specification["bytes"]
        ):
            raise ValueError(f"sealed private input metadata changed: {split}")


def _validate_encoded_private_chain(root: Path, encoded_root: Path) -> None:
    current = _lexical_absolute(root / "artifacts/lumi-sft-token-consistent-shards-v0.2")
    expected = [current, current / "attempt-001", current / "attempt-001/private", encoded_root]
    for path in expected:
        info = path.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise ValueError("token-consistent private directory metadata changed")


def materialize_token_consistent_shards(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
    started: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run pass one, including its durable marker and final evidence writes."""

    root = _lexical_absolute(Path(repository_root))
    paths = output_paths(plan, root)
    _validate_output_absence(paths)
    if (
        not isinstance(plan_file_sha256, str)
        or _SHA256.fullmatch(plan_file_sha256) is None
        or not isinstance(plan, BoundTokenConsistentShardsPlan)
        or plan.file_sha256 != plan_file_sha256
    ):
        raise ValueError("token-consistent plan file hash changed")
    validate_private_input_metadata_without_opening_rows(root)
    marker_identity = start_marker(paths["receipt_path"], PASS_ONE_MARKER)
    started = time.monotonic() if started is None else started
    private_descriptor: int | None = None
    stage_descriptor: int | None = None
    active_writer: _HardenedShardWriter | None = None
    try:
        template, tokenizer = _load_encoding_dependencies(plan, root)
        private_descriptor, private_path = _ensure_private_output_parent(root)
        if os.path.lexists(private_path / "encoded"):
            raise FileExistsError("refusing to overwrite encoded SFT output")
        os.mkdir(_STAGING_DIRECTORY_NAME, 0o700, dir_fd=private_descriptor)
        stage_descriptor = _open_dir_child(
            private_descriptor,
            _STAGING_DIRECTORY_NAME,
            create=False,
            require_private=True,
        )

        inventories: list[dict[str, Any]] = []
        split_records: Counter[str] = Counter()
        split_tokens: Counter[str] = Counter()
        split_supervised: Counter[str] = Counter()
        split_sequence_hashers = {
            "train": hashlib.sha256(),
            "validation": hashlib.sha256(),
        }
        record_hash_sequence_hasher = hashlib.sha256()
        encoded_hasher = hashlib.sha256()
        seen_record_ids: set[str] = set()
        maximum_record_tokens = 0
        input_row_reads = 0
        maximum_tokens_per_shard = EXPECTED_LIMITS["maximum_tokens_per_shard"]

        for split in ("train", "validation"):
            _runtime_check(started)
            expected_split = EXPECTED_INPUT["splits"][split]
            handle, descriptor, input_path, input_identity = _open_private_input_once(
                root, expected_split
            )
            input_hasher = hashlib.sha256()
            input_bytes = 0
            next_shard_index = 0
            try:
                expected_rows = expected_split["records"]
                if input_row_reads + expected_rows > EXPECTED_LIMITS[
                    "maximum_input_jsonl_row_reads"
                ]:
                    raise ValueError("token-consistent input-row cap would be exceeded")
                for line in iter_exact_bounded_lines(
                    handle,
                    expected_rows=expected_rows,
                    expected_bytes=expected_split["bytes"],
                    maximum_rows=expected_rows,
                    label=f"sealed private {split} input",
                ):
                    input_row_reads += 1
                    input_hasher.update(line)
                    input_bytes += len(line)
                    if not line.endswith(b"\n") or not line.strip():
                        raise ValueError("sealed private input contains a malformed row")
                    row = strict_json_bytes(line)
                    if not isinstance(row, dict) or canonical_json(row) + b"\n" != line:
                        raise ValueError("sealed private input is not canonical JSONL")
                    record_id = row.get("record_id")
                    messages = row.get("messages")
                    source = row.get("source")
                    source_id = source.get("source_id") if isinstance(source, Mapping) else None
                    if (
                        row.get("split") != split
                        or not isinstance(record_id, str)
                        or not record_id
                        or not isinstance(messages, list)
                        or not isinstance(source_id, str)
                        or not source_id
                    ):
                        raise ValueError("sealed private input row binding changed")
                    if record_id in seen_record_ids:
                        raise ValueError("duplicate sealed private record ID")
                    seen_record_ids.add(record_id)
                    record_id_sha256 = hashlib.sha256(record_id.encode()).hexdigest()
                    encoded = encode_sft_messages(
                        messages,
                        tokenizer=tokenizer,
                        template=template,
                    )
                    record_tokens = len(encoded.input_ids)
                    if record_tokens > EXPECTED_INPUT["maximum_record_tokens"]:
                        raise ValueError("token-consistent record exceeded the frozen maximum")
                    if active_writer is not None and (
                        active_writer.split != split
                        or active_writer.tokens + record_tokens > maximum_tokens_per_shard
                    ):
                        inventories.append(active_writer.close())
                        active_writer = None
                    if active_writer is None:
                        maximum_split_shards = 9 if split == "train" else 1
                        if next_shard_index >= maximum_split_shards:
                            raise ValueError(
                                "token-consistent output-file cap would be exceeded"
                            )
                        active_writer = _HardenedShardWriter.create(
                            stage_descriptor,
                            split=split,
                            index=next_shard_index,
                        )
                        next_shard_index += 1
                    active_writer.add(
                        record_id=record_id,
                        input_ids=encoded.input_ids,
                        labels=encoded.labels,
                    )

                    identity_bytes = f"{source_id}\0{record_id}\n".encode()
                    split_sequence_hashers[split].update(identity_bytes)
                    record_hash_sequence_hasher.update(
                        f"{split}\0{record_id_sha256}\n".encode()
                    )
                    encoded_bytes = _encoded_payload_bytes(
                        record_id_sha256, encoded.input_ids, encoded.labels
                    )
                    encoded_hasher.update(len(encoded_bytes).to_bytes(8, "big"))
                    encoded_hasher.update(encoded_bytes)
                    split_records[split] += 1
                    split_tokens[split] += record_tokens
                    split_supervised[split] += encoded.supervised_tokens
                    maximum_record_tokens = max(maximum_record_tokens, record_tokens)
                    if input_row_reads % 10_000 == 0:
                        _runtime_check(started)
                if active_writer is not None:
                    inventories.append(active_writer.close())
                    active_writer = None
                _finish_private_input(handle, descriptor, input_path, input_identity)
            except BaseException:
                if not handle.closed:
                    handle.close()
                raise
            if (
                input_bytes != expected_split["bytes"]
                or input_hasher.hexdigest() != expected_split["file_sha256"]
                or split_records[split] != expected_split["records"]
                or split_sequence_hashers[split].hexdigest()
                != expected_split["record_id_sequence_sha256"]
            ):
                raise ValueError(f"sealed private input integrity changed: {split}")

        observed = {
            "maximum_record_tokens": maximum_record_tokens,
            "records": sum(split_records.values()),
            "train": {
                "records": split_records["train"],
                "serialized_tokens": split_tokens["train"],
                "supervised_tokens": split_supervised["train"],
            },
            "validation": {
                "records": split_records["validation"],
                "serialized_tokens": split_tokens["validation"],
                "supervised_tokens": split_supervised["validation"],
            },
        }
        if (
            observed != _expected_observed_totals()
            or input_row_reads != EXPECTED_LIMITS["maximum_input_jsonl_row_reads"]
            or len(seen_record_ids) != EXPECTED_INPUT["records"]
        ):
            raise ValueError("token-consistent encoded totals changed")
        inventories.sort(key=lambda item: (item["split"], item["index"]))
        if tuple((item["split"], item["index"]) for item in inventories) != (
            EXPECTED_SHARD_IDENTITIES
        ):
            raise ValueError("token-consistent shard count or identity changed")

        successor: dict[str, Any] = {
            "authorization_boundary": EXPECTED_OUTPUT_AUTHORIZATION_BOUNDARY,
            "binding_chain": _expected_successor_binding_chain(plan),
            "encoded_dataset_sha256": encoded_hasher.hexdigest(),
            "maximum_record_tokens": maximum_record_tokens,
            "production_blockers_preserved": EXPECTED_PRODUCTION_BLOCKERS,
            "production_eligible": False,
            "raw_text_retained_in_manifest": False,
            "record_hash_sequence_sha256": record_hash_sequence_hasher.hexdigest(),
            "records": observed["records"],
            "schema_version": SEAL_SCHEMA_VERSION,
            "splits": _expected_successor_splits(),
            "status": SEAL_STATUS,
            "supervised_tokens": sum(split_supervised.values()),
            "tokens": sum(split_tokens.values()),
        }
        successor["seal_sha256"] = sha256_value(successor)

        manifest: dict[str, Any] = {
            "authorization_boundary": successor["authorization_boundary"],
            "context_window": template.context_window,
            "encoded_dataset_sha256": successor["encoded_dataset_sha256"],
            "format": {
                "index": EXPECTED_ENCODING_CONTRACT["index_format"],
                "labels": EXPECTED_ENCODING_CONTRACT["label_format"],
                "raw_text_retained": False,
                "tokens": EXPECTED_ENCODING_CONTRACT["token_format"],
            },
            "maximum_tokens_per_shard": maximum_tokens_per_shard,
            "production_eligible": False,
            "record_hash_sequence_sha256": successor[
                "record_hash_sequence_sha256"
            ],
            "records": observed["records"],
            "schema_version": SFT_SHARDS_SCHEMA_VERSION,
            "seal_sha256": successor["seal_sha256"],
            "shards": inventories,
            "splits": {
                split: {
                    "records": split_records[split],
                    "supervised_tokens": split_supervised[split],
                    "tokens": split_tokens[split],
                }
                for split in ("train", "validation")
            },
            "supervised_tokens": sum(split_supervised.values()),
            "tokens": sum(split_tokens.values()),
        }
        manifest["manifest_sha256"] = sha256_value(manifest)
        assert_raw_text_free(successor)
        assert_raw_text_free(manifest)
        manifest_payload = canonical_json(manifest) + b"\n"
        manifest_writer = _SecureFileWriter.create(stage_descriptor, "manifest.json")
        manifest_writer.write(manifest_payload)
        manifest_inventory = manifest_writer.close()

        encoded_write_bytes = sum(
            file_record["bytes"]
            for shard in inventories
            for file_record in shard["files"]
        ) + manifest_inventory["bytes"]
        if (
            encoded_write_bytes + _EVIDENCE_RESERVE_BYTES
            > EXPECTED_LIMITS["maximum_private_write_bytes"]
            or len(inventories) * 3 + 4 > EXPECTED_LIMITS["maximum_output_files"]
        ):
            raise ValueError("token-consistent private-write or output-file cap exceeded")
        receipt: dict[str, Any] = {
            "authorization_boundary": successor["authorization_boundary"],
            "encoded_shards": {
                "directory": EXPECTED_OUTPUT["encoded_shards_path"],
                "manifest_file_sha256": hashlib.sha256(manifest_payload).hexdigest(),
                "manifest_internal_sha256": manifest["manifest_sha256"],
                "private_write_bytes": encoded_write_bytes,
                "shards": len(inventories),
            },
            "plan_file_sha256": plan_file_sha256,
            "production_eligible": False,
            "raw_training_text_retained": False,
            "resource_accounting": {
                "candidate_row_reads": 0,
                "encoded_index_row_reads": 0,
                "full_passes": 1,
                "gpu_use": 0,
                "inference_calls": 0,
                "input_jsonl_row_reads": input_row_reads,
                "network_bytes": 0,
                "output_files": 33,
                "private_write_bytes_upper_bound": (
                    encoded_write_bytes + _EVIDENCE_RESERVE_BYTES
                ),
                "runtime_milliseconds": int((time.monotonic() - started) * 1_000),
                "source_row_reads": 0,
                "test_object_reads": 0,
                "total_row_reads": input_row_reads,
                "training_actions": 0,
            },
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": RECEIPT_STATUS,
            "successor_seal_internal_sha256": successor["seal_sha256"],
            "totals": observed,
        }
        receipt["receipt_sha256"] = sha256_value(receipt)
        assert_raw_text_free(receipt)
        evidence_bytes = len(json_file_bytes(successor)) + len(json_file_bytes(receipt))
        if evidence_bytes > _EVIDENCE_RESERVE_BYTES:
            raise ValueError("token-consistent evidence reserve exceeded")

        os.fsync(stage_descriptor)
        _runtime_check(started)
        # Every semantic check above completes before any final shard or seal path is
        # published.  The durable receipt marker remains if an I/O commit fails.
        os.rename(
            _STAGING_DIRECTORY_NAME,
            "encoded",
            src_dir_fd=private_descriptor,
            dst_dir_fd=private_descriptor,
        )
        os.fsync(private_descriptor)
        secure_write_json_exclusive(paths["successor_seal_path"], successor)
        _runtime_check(started)
        receipt["resource_accounting"]["runtime_milliseconds"] = int(
            (time.monotonic() - started) * 1_000
        )
        receipt.pop("receipt_sha256", None)
        receipt["receipt_sha256"] = sha256_value(receipt)
        assert_raw_text_free(receipt)
        if len(json_file_bytes(receipt)) > _EVIDENCE_RESERVE_BYTES:
            raise ValueError("token-consistent receipt evidence reserve exceeded")
        finish_marker(
            paths["receipt_path"],
            PASS_ONE_MARKER,
            json_file_bytes(receipt),
            marker_identity,
        )
        return successor, receipt
    except BaseException:
        if active_writer is not None:
            active_writer.abort_close()
        raise
    finally:
        if stage_descriptor is not None:
            os.close(stage_descriptor)
        if private_descriptor is not None:
            os.close(private_descriptor)


run_materialization_pass = materialize_token_consistent_shards


def validate_index_entry(
    entry: Any, *, expected_ordinal: int, expected_offset: int
) -> tuple[int, int]:
    """Validate the exact five-field hash-only index contract."""

    if not isinstance(entry, Mapping) or set(entry) != INDEX_KEYS:
        raise ValueError("token-consistent index must have exactly five hash-only fields")
    if (
        type(entry.get("record_ordinal")) is not int
        or entry["record_ordinal"] != expected_ordinal
        or type(entry.get("offset_tokens")) is not int
        or entry["offset_tokens"] != expected_offset
    ):
        raise ValueError("token-consistent shard index is not contiguous")
    tokens = entry.get("tokens")
    supervised = entry.get("supervised_tokens")
    record_hash = entry.get("record_id_sha256")
    if (
        type(tokens) is not int
        or tokens < 1
        or tokens > EXPECTED_ENCODING_CONTRACT["maximum_record_tokens"]
    ):
        raise ValueError("token-consistent record token count is invalid")
    if type(supervised) is not int or not 2 <= supervised < tokens:
        raise ValueError("token-consistent record supervision count is invalid")
    if not isinstance(record_hash, str) or _SHA256.fullmatch(record_hash) is None:
        raise ValueError("token-consistent record identity hash is invalid")
    return tokens, supervised


def _open_audit_file(directory_descriptor: int, name: str) -> tuple[BinaryIO, os.stat_result]:
    descriptor = os.open(name, _open_flags(os.O_RDONLY), dir_fd=directory_descriptor)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise ValueError("token-consistent encoded file metadata changed")
        final = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(final.st_mode)
            or (final.st_dev, final.st_ino, final.st_size)
            != (info.st_dev, info.st_ino, info.st_size)
        ):
            raise ValueError("token-consistent encoded file path changed")
        return os.fdopen(descriptor, "rb", buffering=1024 * 1024), info
    except BaseException:
        os.close(descriptor)
        raise


def _finish_audit_file(
    handle: BinaryIO,
    directory_descriptor: int,
    name: str,
    initial: os.stat_result,
) -> None:
    after = os.fstat(handle.fileno())
    handle.close()
    final = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    if (
        not stat.S_ISREG(final.st_mode)
        or final.st_nlink != 1
        or final.st_uid != os.geteuid()
        or stat.S_IMODE(final.st_mode) != 0o600
        or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        != (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns)
        or (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ValueError("token-consistent encoded file changed during audit")


def _read_exact_and_hash(handle: BinaryIO, length: int, digest: Any) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = handle.read(remaining)
        if not chunk:
            raise ValueError("token-consistent binary shard ended early")
        chunks.append(chunk)
        digest.update(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _manifest_expected_totals(expected_totals: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "records": expected_totals["records"],
        "tokens": expected_totals["train"]["serialized_tokens"]
        + expected_totals["validation"]["serialized_tokens"],
        "supervised_tokens": expected_totals["train"]["supervised_tokens"]
        + expected_totals["validation"]["supervised_tokens"],
        "splits": {
            split: {
                "records": expected_totals[split]["records"],
                "supervised_tokens": expected_totals[split]["supervised_tokens"],
                "tokens": expected_totals[split]["serialized_tokens"],
            }
            for split in ("train", "validation")
        },
    }


def fused_audit_encoded_shards(
    encoded_root: str | Path,
    manifest: Mapping[str, Any] | None = None,
    *,
    expected_totals: Mapping[str, Any] | None = None,
    expected_shard_identities: Sequence[tuple[str, int]] | None = None,
    maximum_index_rows: int | None = None,
    started: float | None = None,
) -> dict[str, Any]:
    """Fused one-pass audit of each index and its paired binary streams."""

    root = _lexical_absolute(Path(encoded_root))
    _require_absolute_parent_chain_without_symlinks(root / "sentinel")
    root_info = root.lstat()
    if (
        stat.S_ISLNK(root_info.st_mode)
        or not stat.S_ISDIR(root_info.st_mode)
        or root_info.st_uid != os.geteuid()
        or stat.S_IMODE(root_info.st_mode) != 0o700
    ):
        raise ValueError("token-consistent encoded directory metadata changed")
    directory_descriptor = os.open(root, _open_flags(os.O_RDONLY, directory=True))
    started = time.monotonic() if started is None else started
    try:
        manifest_handle, manifest_info = _open_audit_file(
            directory_descriptor, "manifest.json"
        )
        try:
            manifest_payload = manifest_handle.read()
        except BaseException:
            manifest_handle.close()
            raise
        _finish_audit_file(
            manifest_handle, directory_descriptor, "manifest.json", manifest_info
        )
        disk_manifest = strict_json_bytes(manifest_payload)
        if (
            not isinstance(disk_manifest, dict)
            or canonical_json(disk_manifest) + b"\n" != manifest_payload
        ):
            raise ValueError("token-consistent manifest is not canonical JSON")
        if manifest is not None and not _strict_equal(disk_manifest, manifest):
            raise ValueError("token-consistent manifest argument differs from disk")
        manifest = disk_manifest
        assert_raw_text_free(manifest)
        expected_manifest_hash = manifest.get("manifest_sha256")
        unsigned_manifest = dict(manifest)
        unsigned_manifest.pop("manifest_sha256", None)
        if (
            set(manifest)
            != {
                "authorization_boundary",
                "context_window",
                "encoded_dataset_sha256",
                "format",
                "manifest_sha256",
                "maximum_tokens_per_shard",
                "production_eligible",
                "record_hash_sequence_sha256",
                "records",
                "schema_version",
                "seal_sha256",
                "shards",
                "splits",
                "supervised_tokens",
                "tokens",
            }
            or manifest.get("schema_version") != SFT_SHARDS_SCHEMA_VERSION
            or not isinstance(expected_manifest_hash, str)
            or expected_manifest_hash != sha256_value(unsigned_manifest)
            or not isinstance(manifest.get("encoded_dataset_sha256"), str)
            or _SHA256.fullmatch(manifest["encoded_dataset_sha256"]) is None
            or not isinstance(manifest.get("seal_sha256"), str)
            or _SHA256.fullmatch(manifest["seal_sha256"]) is None
            or manifest.get("authorization_boundary")
            != EXPECTED_OUTPUT_AUTHORIZATION_BOUNDARY
            or manifest.get("context_window")
            != EXPECTED_ENCODING_CONTRACT["context_window"]
            or manifest.get("maximum_tokens_per_shard")
            != EXPECTED_LIMITS["maximum_tokens_per_shard"]
            or not isinstance(manifest.get("record_hash_sequence_sha256"), str)
            or _SHA256.fullmatch(manifest["record_hash_sequence_sha256"]) is None
            or manifest.get("format")
            != {
                "index": EXPECTED_ENCODING_CONTRACT["index_format"],
                "labels": EXPECTED_ENCODING_CONTRACT["label_format"],
                "raw_text_retained": False,
                "tokens": EXPECTED_ENCODING_CONTRACT["token_format"],
            }
            or manifest.get("production_eligible") is not False
        ):
            raise ValueError("token-consistent encoded manifest changed")
        if manifest_info.st_size != len(manifest_payload):
            raise ValueError("token-consistent manifest size changed")

        expected_totals = (
            _expected_observed_totals() if expected_totals is None else expected_totals
        )
        expected_shard_identities = tuple(
            EXPECTED_SHARD_IDENTITIES
            if expected_shard_identities is None
            else expected_shard_identities
        )
        maximum_index_rows = (
            EXPECTED_LIMITS["maximum_encoded_index_row_reads"]
            if maximum_index_rows is None
            else maximum_index_rows
        )
        if type(maximum_index_rows) is not int or maximum_index_rows < 1:
            raise ValueError("token-consistent encoded-index row cap changed")
        shards = manifest.get("shards")
        if (
            not isinstance(shards, list)
            or not all(isinstance(item, Mapping) for item in shards)
            or tuple((item.get("split"), item.get("index")) for item in shards)
            != tuple(expected_shard_identities)
        ):
            raise ValueError("token-consistent shard identity inventory changed")
        if any(
            set(item)
            != {"files", "index", "records", "split", "supervised_tokens", "tokens"}
            or type(item.get("records")) is not int
            or item["records"] < 1
            or type(item.get("tokens")) is not int
            or not 0 < item["tokens"] <= EXPECTED_LIMITS["maximum_tokens_per_shard"]
            or type(item.get("supervised_tokens")) is not int
            or not 0 < item["supervised_tokens"] <= item["tokens"]
            for item in shards
        ):
            raise ValueError("token-consistent declared shard record count changed")
        declared_index_rows = sum(item["records"] for item in shards)
        if (
            declared_index_rows != expected_totals["records"]
            or declared_index_rows > maximum_index_rows
        ):
            raise ValueError("token-consistent encoded-index row cap would be exceeded")

        declared_files = {"manifest.json"}
        total_records = 0
        total_tokens = 0
        total_supervised = 0
        total_masked = 0
        split_records: Counter[str] = Counter()
        split_tokens: Counter[str] = Counter()
        split_supervised: Counter[str] = Counter()
        index_rows = 0
        encoded_hasher = hashlib.sha256()
        record_hash_sequence_hasher = hashlib.sha256()

        for shard in shards:
            split = shard.get("split")
            index = shard.get("index")
            if split not in {"train", "validation"} or type(index) is not int:
                raise ValueError("token-consistent shard identity changed")
            prefix = f"{split}-{index:05d}"
            names = {
                "tokens": f"{prefix}.tokens.u32le",
                "labels": f"{prefix}.labels.i32le",
                "index": f"{prefix}.index.jsonl",
            }
            files = shard.get("files")
            if not isinstance(files, list) or len(files) != 3:
                raise ValueError("token-consistent shard file inventory changed")
            file_records = {
                item.get("path"): item for item in files if isinstance(item, Mapping)
            }
            if set(file_records) != set(names.values()):
                raise ValueError("token-consistent shard filenames changed")
            declared_files.update(names.values())

            token_handle, token_info = _open_audit_file(
                directory_descriptor, names["tokens"]
            )
            label_handle, label_info = _open_audit_file(
                directory_descriptor, names["labels"]
            )
            index_handle, index_info = _open_audit_file(
                directory_descriptor, names["index"]
            )
            token_digest = hashlib.sha256()
            label_digest = hashlib.sha256()
            index_digest = hashlib.sha256()
            shard_records = 0
            shard_tokens = 0
            shard_supervised = 0
            shard_masked = 0
            try:
                for line in iter_exact_bounded_lines(
                    index_handle,
                    expected_rows=shard["records"],
                    expected_bytes=index_info.st_size,
                    maximum_rows=shard["records"],
                    label=f"token-consistent {prefix} index",
                ):
                    index_rows += 1
                    index_digest.update(line)
                    if (
                        not line.endswith(b"\n")
                        or len(line) > _MAXIMUM_INDEX_BYTES_PER_ROW
                    ):
                        raise ValueError("token-consistent index row framing changed")
                    entry = strict_json_bytes(line)
                    if canonical_json(entry) + b"\n" != line:
                        raise ValueError("token-consistent index row is not canonical JSON")
                    length, supervised = validate_index_entry(
                        entry,
                        expected_ordinal=shard_records,
                        expected_offset=shard_tokens,
                    )
                    token_payload = _read_exact_and_hash(
                        token_handle, length * 4, token_digest
                    )
                    label_payload = _read_exact_and_hash(
                        label_handle, length * 4, label_digest
                    )
                    tokens = struct.unpack(f"<{length}I", token_payload)
                    labels = struct.unpack(f"<{length}i", label_payload)
                    encoded_bytes = _encoded_payload_bytes(
                        entry["record_id_sha256"], tokens, labels
                    )
                    encoded_hasher.update(len(encoded_bytes).to_bytes(8, "big"))
                    encoded_hasher.update(encoded_bytes)
                    record_hash_sequence_hasher.update(
                        f"{split}\0{entry['record_id_sha256']}\n".encode()
                    )
                    observed_supervised = 0
                    observed_masked = 0
                    for token, label in zip(tokens, labels, strict=True):
                        if token > 65_535:
                            raise ValueError("token-consistent token exceeds frozen vocabulary")
                        if label == -100:
                            observed_masked += 1
                        elif label == token:
                            observed_supervised += 1
                        else:
                            raise ValueError(
                                "token-consistent unmasked label differs from input token"
                            )
                    if observed_supervised != supervised or observed_masked < 1:
                        raise ValueError("token-consistent record mask count changed")
                    shard_records += 1
                    shard_tokens += length
                    shard_supervised += supervised
                    shard_masked += observed_masked
                    if index_rows % 10_000 == 0:
                        _runtime_check(started)
                if (
                    token_handle.tell() != token_info.st_size
                    or label_handle.tell() != label_info.st_size
                ):
                    raise ValueError("token-consistent shard has bytes beyond its row bound")
            except BaseException:
                token_handle.close()
                label_handle.close()
                index_handle.close()
                raise
            _finish_audit_file(
                token_handle, directory_descriptor, names["tokens"], token_info
            )
            _finish_audit_file(
                label_handle, directory_descriptor, names["labels"], label_info
            )
            _finish_audit_file(
                index_handle, directory_descriptor, names["index"], index_info
            )

            observed_files = {
                names["tokens"]: (token_info.st_size, token_digest.hexdigest()),
                names["labels"]: (label_info.st_size, label_digest.hexdigest()),
                names["index"]: (index_info.st_size, index_digest.hexdigest()),
            }
            for name, (observed_bytes, observed_hash) in observed_files.items():
                file_record = file_records[name]
                if (
                    set(file_record) != {"bytes", "path", "sha256"}
                    or type(file_record.get("bytes")) is not int
                    or file_record["bytes"] != observed_bytes
                    or not isinstance(file_record.get("sha256"), str)
                    or _SHA256.fullmatch(file_record["sha256"]) is None
                    or file_record.get("sha256") != observed_hash
                ):
                    raise ValueError("token-consistent shard file hash or size changed")
            if (
                token_info.st_size != shard_tokens * 4
                or label_info.st_size != shard_tokens * 4
                or type(shard.get("records")) is not int
                or shard["records"] != shard_records
                or type(shard.get("tokens")) is not int
                or shard["tokens"] != shard_tokens
                or type(shard.get("supervised_tokens")) is not int
                or shard["supervised_tokens"] != shard_supervised
            ):
                raise ValueError("token-consistent shard totals changed")
            total_records += shard_records
            total_tokens += shard_tokens
            total_supervised += shard_supervised
            total_masked += shard_masked
            split_records[str(split)] += shard_records
            split_tokens[str(split)] += shard_tokens
            split_supervised[str(split)] += shard_supervised

        actual_files = set(os.listdir(directory_descriptor))
        if actual_files != declared_files:
            raise ValueError("token-consistent encoded directory has undeclared entries")
        expected_manifest_totals = _manifest_expected_totals(expected_totals)
        observed_manifest_totals = {
            "records": total_records,
            "tokens": total_tokens,
            "supervised_tokens": total_supervised,
            "splits": {
                split: {
                    "records": split_records[split],
                    "supervised_tokens": split_supervised[split],
                    "tokens": split_tokens[split],
                }
                for split in ("train", "validation")
            },
        }
        if (
            observed_manifest_totals != expected_manifest_totals
            or any(
                manifest.get(key) != value
                for key, value in observed_manifest_totals.items()
            )
            or index_rows != expected_totals["records"]
            or encoded_hasher.hexdigest() != manifest.get("encoded_dataset_sha256")
            or record_hash_sequence_hasher.hexdigest()
            != manifest.get("record_hash_sequence_sha256")
        ):
            raise ValueError("token-consistent encoded aggregate totals changed")
        _runtime_check(started)
        return {
            "binary_files_read_once": len(shards) * 2,
            "encoded_dataset_sha256": encoded_hasher.hexdigest(),
            "encoded_index_row_reads": index_rows,
            "index_files_read_once": len(shards),
            "manifest_file_sha256": hashlib.sha256(manifest_payload).hexdigest(),
            "manifest_internal_sha256": manifest["manifest_sha256"],
            "manifest_seal_sha256": manifest["seal_sha256"],
            "masked_tokens": total_masked,
            "record_hash_sequence_sha256": record_hash_sequence_hasher.hexdigest(),
            "records": total_records,
            "supervised_tokens": total_supervised,
            "tokens": total_tokens,
        }
    finally:
        os.close(directory_descriptor)


verify_binary_masks = fused_audit_encoded_shards


def _expected_successor_binding_chain(plan: Mapping[str, Any]) -> dict[str, str]:
    return {
        "chat_template_sha256": plan["bindings"]["chat_template"]["sha256"],
        "data_contract_sha256": plan["bindings"]["data_contract"]["sha256"],
        "encoding_preflight_report_sha256": plan["bindings"][
            "encoding_preflight_report"
        ]["sha256"],
        "final_base_binding_sha256": plan["bindings"]["final_base_binding"]["sha256"],
        "learning_seal_file_sha256": EXPECTED_INPUT["learning_seal_file_sha256"],
        "learning_seal_internal_sha256": EXPECTED_INPUT["learning_seal_internal_sha256"],
        "source_registry_sha256": plan["bindings"]["source_registry"]["sha256"],
        "tokenizer_sha256": plan["bindings"]["tokenizer"]["sha256"],
    }


def _expected_successor_splits() -> dict[str, Any]:
    return {
        split: {
            "record_sequence_sha256": EXPECTED_INPUT["splits"][split][
                "record_id_sequence_sha256"
            ],
            "records": EXPECTED_INPUT["splits"][split]["records"],
            "serialized_tokens": EXPECTED_INPUT["splits"][split]["serialized_tokens"],
            "source_jsonl_sha256": EXPECTED_INPUT["splits"][split]["file_sha256"],
            "supervised_tokens": EXPECTED_INPUT["splits"][split]["supervised_tokens"],
        }
        for split in ("train", "validation")
    }


def build_token_consistent_shards_verification(
    *,
    plan: Mapping[str, Any],
    repository_root: str | Path,
    expected_plan_file_sha256: str | None = None,
    started: float | None = None,
) -> dict[str, Any]:
    """Build pass-two evidence without opening either private input JSONL."""

    pass_two_started = time.monotonic() if started is None else started
    root = _lexical_absolute(Path(repository_root))
    paths = output_paths(plan, root)
    _validate_encoded_private_chain(root, paths["encoded_shards_path"])
    successor_payload = _read_secure_file(
        paths["successor_seal_path"], require_private=True
    )
    receipt_payload = _read_secure_file(paths["receipt_path"], require_private=True)
    successor = strict_json_bytes(successor_payload)
    receipt = strict_json_bytes(receipt_payload)
    if (
        not isinstance(successor, dict)
        or not isinstance(receipt, dict)
        or json_file_bytes(successor) != successor_payload
        or json_file_bytes(receipt) != receipt_payload
    ):
        raise ValueError("token-consistent pass-one evidence is not canonical JSON")
    unsigned_successor = dict(successor)
    successor_hash = unsigned_successor.pop("seal_sha256", None)
    unsigned_receipt = dict(receipt)
    receipt_hash = unsigned_receipt.pop("receipt_sha256", None)
    bound_plan_sha256 = (
        plan.file_sha256
        if expected_plan_file_sha256 is None
        and isinstance(plan, BoundTokenConsistentShardsPlan)
        else expected_plan_file_sha256
    )
    validate_receipt_plan_hash(receipt, plan, str(bound_plan_sha256))
    expected_totals = _expected_observed_totals()
    expected_total_tokens = (
        expected_totals["train"]["serialized_tokens"]
        + expected_totals["validation"]["serialized_tokens"]
    )
    expected_supervised = (
        expected_totals["train"]["supervised_tokens"]
        + expected_totals["validation"]["supervised_tokens"]
    )
    successor_keys = {
        "authorization_boundary",
        "binding_chain",
        "encoded_dataset_sha256",
        "maximum_record_tokens",
        "production_blockers_preserved",
        "production_eligible",
        "raw_text_retained_in_manifest",
        "record_hash_sequence_sha256",
        "records",
        "schema_version",
        "seal_sha256",
        "splits",
        "status",
        "supervised_tokens",
        "tokens",
    }
    receipt_keys = {
        "authorization_boundary",
        "encoded_shards",
        "plan_file_sha256",
        "production_eligible",
        "raw_training_text_retained",
        "receipt_sha256",
        "resource_accounting",
        "schema_version",
        "status",
        "successor_seal_internal_sha256",
        "totals",
    }
    receipt_resources = receipt.get("resource_accounting")
    encoded_shards = receipt.get("encoded_shards")
    pass_one_runtime_milliseconds = (
        receipt_resources.get("runtime_milliseconds")
        if isinstance(receipt_resources, Mapping)
        else None
    )
    if (
        set(successor) != successor_keys
        or successor.get("schema_version") != SEAL_SCHEMA_VERSION
        or successor.get("status") != SEAL_STATUS
        or successor_hash != sha256_value(unsigned_successor)
        or successor.get("authorization_boundary")
        != EXPECTED_OUTPUT_AUTHORIZATION_BOUNDARY
        or successor.get("binding_chain") != _expected_successor_binding_chain(plan)
        or successor.get("maximum_record_tokens")
        != EXPECTED_INPUT["maximum_record_tokens"]
        or successor.get("production_blockers_preserved")
        != EXPECTED_PRODUCTION_BLOCKERS
        or successor.get("production_eligible") is not False
        or successor.get("raw_text_retained_in_manifest") is not False
        or not isinstance(successor.get("encoded_dataset_sha256"), str)
        or _SHA256.fullmatch(successor["encoded_dataset_sha256"]) is None
        or not isinstance(successor.get("record_hash_sequence_sha256"), str)
        or _SHA256.fullmatch(successor["record_hash_sequence_sha256"]) is None
        or successor.get("records") != EXPECTED_INPUT["records"]
        or successor.get("tokens") != expected_total_tokens
        or successor.get("supervised_tokens") != expected_supervised
        or successor.get("splits") != _expected_successor_splits()
        or set(receipt) != receipt_keys
        or receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != RECEIPT_STATUS
        or receipt_hash != sha256_value(unsigned_receipt)
        or receipt.get("authorization_boundary")
        != EXPECTED_OUTPUT_AUTHORIZATION_BOUNDARY
        or receipt.get("successor_seal_internal_sha256") != successor_hash
        or receipt.get("production_eligible") is not False
        or receipt.get("raw_training_text_retained") is not False
        or receipt.get("totals") != expected_totals
        or not isinstance(encoded_shards, Mapping)
        or set(encoded_shards)
        != {
            "directory",
            "manifest_file_sha256",
            "manifest_internal_sha256",
            "private_write_bytes",
            "shards",
        }
        or encoded_shards.get("directory") != EXPECTED_OUTPUT["encoded_shards_path"]
        or encoded_shards.get("shards") != len(EXPECTED_SHARD_IDENTITIES)
        or type(encoded_shards.get("private_write_bytes")) is not int
        or not 0 < encoded_shards["private_write_bytes"]
        <= EXPECTED_LIMITS["maximum_private_write_bytes"] - _EVIDENCE_RESERVE_BYTES
        or not isinstance(receipt_resources, Mapping)
        or set(receipt_resources)
        != {
            "candidate_row_reads",
            "encoded_index_row_reads",
            "full_passes",
            "gpu_use",
            "inference_calls",
            "input_jsonl_row_reads",
            "network_bytes",
            "output_files",
            "private_write_bytes_upper_bound",
            "runtime_milliseconds",
            "source_row_reads",
            "test_object_reads",
            "total_row_reads",
            "training_actions",
        }
        or any(
            receipt_resources.get(key) != value
            for key, value in {
                "candidate_row_reads": 0,
                "encoded_index_row_reads": 0,
                "full_passes": 1,
                "gpu_use": 0,
                "inference_calls": 0,
                "input_jsonl_row_reads": EXPECTED_INPUT["records"],
                "network_bytes": 0,
                "output_files": 33,
                "private_write_bytes_upper_bound": (
                    encoded_shards["private_write_bytes"] + _EVIDENCE_RESERVE_BYTES
                ),
                "source_row_reads": 0,
                "test_object_reads": 0,
                "total_row_reads": EXPECTED_INPUT["records"],
                "training_actions": 0,
            }.items()
        )
        or type(pass_one_runtime_milliseconds) is not int
        or not 0 <= pass_one_runtime_milliseconds
        <= EXPECTED_LIMITS["maximum_runtime_minutes"] * 60_000
    ):
        raise ValueError("token-consistent seal or receipt changed")
    assert_raw_text_free(successor)
    assert_raw_text_free(receipt)
    combined_started = pass_two_started - (pass_one_runtime_milliseconds / 1_000)
    _runtime_check(combined_started)
    audit = fused_audit_encoded_shards(
        paths["encoded_shards_path"],
        expected_totals=_expected_observed_totals(),
        expected_shard_identities=EXPECTED_SHARD_IDENTITIES,
        maximum_index_rows=EXPECTED_LIMITS["maximum_encoded_index_row_reads"],
        started=combined_started,
    )
    if (
        audit["manifest_seal_sha256"] != successor_hash
        or audit["encoded_dataset_sha256"] != successor["encoded_dataset_sha256"]
        or audit["record_hash_sequence_sha256"]
        != successor["record_hash_sequence_sha256"]
        or encoded_shards["manifest_file_sha256"] != audit["manifest_file_sha256"]
        or encoded_shards["manifest_internal_sha256"]
        != audit["manifest_internal_sha256"]
    ):
        raise ValueError("token-consistent manifest evidence chain changed")

    pass_two_runtime_milliseconds = int(
        (time.monotonic() - pass_two_started) * 1_000
    )
    combined_runtime_milliseconds = (
        pass_one_runtime_milliseconds + pass_two_runtime_milliseconds
    )
    if combined_runtime_milliseconds > EXPECTED_LIMITS["maximum_runtime_minutes"] * 60_000:
        raise TimeoutError("token-consistent combined runtime cap reached")

    verification: dict[str, Any] = {
        "checks": {
            "all_329200_encoded_indices_and_binary_slices_audited_once": True,
            "all_binary_hashes_sizes_modes_and_link_counts_verified": True,
            "all_unmasked_labels_equal_input_tokens": True,
            "candidate_source_and_test_rows_unread": True,
            "exact_nine_train_and_one_validation_shards": True,
            "exact_serialized_and_supervised_token_totals": True,
            "five_field_hash_only_canonical_indices_verified": True,
            "input_jsonl_not_reopened_in_pass_two": True,
            "network_gpu_inference_and_training_unused": True,
            "nonproduction_boundary_and_production_blockers_preserved": True,
            "private_modes_ownership_and_single_links_verified": True,
            "raw_training_text_absent_from_outputs_and_evidence": True,
        },
        "encoded_manifest_file_sha256": audit["manifest_file_sha256"],
        "encoded_manifest_internal_sha256": audit["manifest_internal_sha256"],
        "fused_audit": audit,
        "production_blockers_preserved": EXPECTED_PRODUCTION_BLOCKERS,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "receipt_file_sha256": hashlib.sha256(receipt_payload).hexdigest(),
        "receipt_internal_sha256": receipt_hash,
        "records": audit["records"],
        "resource_accounting": {
            "candidate_row_reads": 0,
            "encoded_index_row_reads": audit["encoded_index_row_reads"],
            "full_passes": 2,
            "gpu_use": 0,
            "inference_calls": 0,
            "input_jsonl_row_reads": EXPECTED_LIMITS["maximum_input_jsonl_row_reads"],
            "network_bytes": 0,
            "output_files": EXPECTED_LIMITS["maximum_output_files"],
            "private_write_bytes_upper_bound": receipt_resources[
                "private_write_bytes_upper_bound"
            ],
            "runtime_milliseconds": {
                "combined": combined_runtime_milliseconds,
                "pass_one": pass_one_runtime_milliseconds,
                "pass_two": pass_two_runtime_milliseconds,
            },
            "source_row_reads": 0,
            "test_object_reads": 0,
            "total_row_reads": EXPECTED_LIMITS["maximum_total_row_reads"],
            "training_actions": 0,
        },
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": VERIFICATION_STATUS,
        "successor_seal_file_sha256": hashlib.sha256(successor_payload).hexdigest(),
        "successor_seal_internal_sha256": successor_hash,
    }
    verification["verification_sha256"] = sha256_value(verification)
    assert_raw_text_free(verification)
    if len(json_file_bytes(verification)) > _EVIDENCE_RESERVE_BYTES:
        raise ValueError("token-consistent verification evidence reserve exceeded")
    return verification


def run_verification_pass(
    *,
    plan: Mapping[str, Any],
    plan_file_sha256: str,
    repository_root: str | Path,
    started: float | None = None,
) -> dict[str, Any]:
    """Run pass two with a durable no-retry marker and no private-input reopen."""

    root = _lexical_absolute(Path(repository_root))
    paths = output_paths(plan, root)
    if (
        not isinstance(plan_file_sha256, str)
        or _SHA256.fullmatch(plan_file_sha256) is None
        or not isinstance(plan, BoundTokenConsistentShardsPlan)
        or plan.file_sha256 != plan_file_sha256
    ):
        raise ValueError("token-consistent plan file hash changed")
    if os.path.lexists(paths["verification_path"]):
        raise FileExistsError("refusing to overwrite token-consistent verification")
    pass_two_started = time.monotonic() if started is None else started
    marker_identity = start_marker(paths["verification_path"], PASS_TWO_MARKER)
    verification = build_token_consistent_shards_verification(
        plan=plan,
        repository_root=root,
        expected_plan_file_sha256=plan_file_sha256,
        started=pass_two_started,
    )
    pass_two_runtime_milliseconds = int(
        (time.monotonic() - pass_two_started) * 1_000
    )
    runtime = verification["resource_accounting"]["runtime_milliseconds"]
    combined_runtime_milliseconds = runtime["pass_one"] + pass_two_runtime_milliseconds
    if combined_runtime_milliseconds > EXPECTED_LIMITS["maximum_runtime_minutes"] * 60_000:
        raise TimeoutError("token-consistent combined runtime cap reached")
    runtime["pass_two"] = pass_two_runtime_milliseconds
    runtime["combined"] = combined_runtime_milliseconds
    verification.pop("verification_sha256", None)
    verification["verification_sha256"] = sha256_value(verification)
    _runtime_check(pass_two_started)
    finish_marker(
        paths["verification_path"],
        PASS_TWO_MARKER,
        json_file_bytes(verification),
        marker_identity,
    )
    return verification


__all__ = [
    "EXPECTED_AUTHORIZATION",
    "EXPECTED_AUTHORIZATION_BYTES",
    "EXPECTED_AUTHORIZATION_SHA256",
    "EXPECTED_AUTHORIZATION_TEXT",
    "EXPECTED_BINDING_NAMES",
    "EXPECTED_CODE_PATHS",
    "EXPECTED_ENCODING_CONTRACT",
    "EXPECTED_EXTERNAL_BINDINGS",
    "EXPECTED_INPUT",
    "EXPECTED_LIMITS",
    "EXPECTED_OBSERVED_AT",
    "EXPECTED_OUTPUT",
    "EXPECTED_PASS_ONE_MARKER",
    "EXPECTED_PASS_TWO_MARKER",
    "EXPECTED_PLAN_PATH",
    "EXPECTED_PRODUCTION_BLOCKERS",
    "EXPECTED_SHARD_IDENTITIES",
    "INDEX_KEYS",
    "PASS_ONE_MARKER",
    "PASS_TWO_MARKER",
    "PLAN_SCHEMA_VERSION",
    "PLAN_STATUS",
    "RECEIPT_SCHEMA_VERSION",
    "RECEIPT_STATUS",
    "SEAL_SCHEMA_VERSION",
    "SEAL_STATUS",
    "VERIFICATION_SCHEMA_VERSION",
    "VERIFICATION_STATUS",
    "assert_raw_text_free",
    "build_token_consistent_shards_verification",
    "canonical_json",
    "file_sha256",
    "finish_marker",
    "fused_audit_encoded_shards",
    "json_file_bytes",
    "load_token_consistent_shards_plan",
    "materialize_token_consistent_shards",
    "output_paths",
    "run_materialization_pass",
    "run_verification_pass",
    "secure_write_json_exclusive",
    "sha256_value",
    "start_marker",
    "strict_json_bytes",
    "validate_index_entry",
    "verify_binary_masks",
]
