"""Pure, bounded validation of an externally pinned Linux CPU release lock.

Only supplied bytes are inspected. Matching a digest is not publisher-signature
verification, an independently complete dependency closure, a successful image
build, or permission to run SFT. The mutable base tag is descriptive only; a
consumer must use the separately pinned image manifest and config digests.
"""

from __future__ import annotations

import hashlib
import json
import re
from urllib.parse import urlsplit

SCHEMA = "lumi.sft-linux-cpu-release-lock.v0.1"
PLATFORM = "linux/amd64"
PYTHON_ABI = "cp312"
MAXIMUM_LOCK_BYTES = 256 * 1024
MINIMUM_REQUIREMENTS = 6
MAXIMUM_REQUIREMENTS = 64
MAXIMUM_WHEEL_BYTES = 536870912
MAXIMUM_TOTAL_BYTES = 1073741824
LOCK_KEYS = frozenset(("schema", "platform", "python_abi", "base", "requirements", "policy"))
BASE_KEYS = frozenset((
    "repository", "digest", "config_digest", "python_version", "tag_reference",
))
REQUIREMENT_KEYS = frozenset(("name", "version", "filename", "url", "sha256", "bytes"))
POLICY = {
    "device": "cpu", "strategy": "ddp", "max_context_window": 4095,
    "production_eligible": False, "real_sft_execution_permitted": False,
}
REQUIRED_NAMES = frozenset(("certifi", "safetensors", "tokenizers", "torch", "transformers"))
ERROR_CODES = frozenset((
    "invalid_raw_type", "lock_budget_exceeded", "invalid_expected_sha256",
    "lock_hash_mismatch", "invalid_json", "duplicate_json_key", "invalid_json_number",
    "noncanonical_json", "invalid_lock_shape", "unsupported_release_contract",
    "invalid_base", "invalid_requirements", "requirement_count_exceeded",
    "invalid_requirement_shape", "invalid_name", "forbidden_dependency", "invalid_version",
    "invalid_filename", "incompatible_wheel", "invalid_url", "invalid_wheel_sha256",
    "invalid_wheel_size", "wheel_budget_exceeded", "total_budget_exceeded",
    "requirements_not_sorted_unique", "missing_required_dependency", "invalid_torch_contract",
    "invalid_policy",
))
_SHA256 = re.compile(r"[0-9a-f]{64}")
_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_VERSION = re.compile(r"[A-Za-z0-9.+_-]{1,64}")
_FILENAME = re.compile(r"[A-Za-z0-9_+.-]{1,236}\.whl")
_URL_PATH = re.compile(r"/[A-Za-z0-9_+./%~-]+")
_PYTHON_VERSION = re.compile(r"3\.12\.[0-9]{1,3}")
_BUILD = re.compile(r"[0-9][A-Za-z0-9_]*")
_GPU_COMPONENT = re.compile(
    r"(?:^|[_.+-])(?:cuda[0-9]*x?|nvidia|cu[0-9]{2,3}x?|cudnn|cublas|cufft|cupti|"
    r"curand|cusolver|cusparse|cudart|nccl|nvjitlink)(?:$|[_.+-])"
)
_FORBIDDEN_NAMES = frozenset(("triton", "pyspark", "mlx", "flash-attn"))


class LockError(ValueError):
    """A fixed allowlisted code, without supplied values or exception text."""

    def __init__(self, code: str):
        if type(code) is not str or code not in ERROR_CODES:
            code = "invalid_lock_shape"
        self.code = code
        super().__init__(code)


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise LockError(code)


def _sha256(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _pairs(items: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in items:
        _require(key not in value, "duplicate_json_key")
        value[key] = item
    return value


def _reject_number(_value: str) -> None:
    raise LockError("invalid_json_number")


def _integer(value: str) -> int:
    _require(len(value) <= 20, "invalid_json_number")
    return int(value)


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
        ).encode("ascii") + b"\n"
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        raise LockError("noncanonical_json") from None


def _parse(raw: bytes) -> dict:
    try:
        value = json.loads(
            raw.decode("ascii"), object_pairs_hook=_pairs, parse_int=_integer,
            parse_float=_reject_number, parse_constant=_reject_number,
        )
    except LockError:
        raise
    except (ValueError, UnicodeError, RecursionError, OverflowError):
        raise LockError("invalid_json") from None
    _require(type(value) is dict and value.keys() == LOCK_KEYS, "invalid_lock_shape")
    return value


def _base(value: object) -> None:
    _require(type(value) is dict and value.keys() == BASE_KEYS, "invalid_base")
    _require(value["repository"] == "docker.io/library/python", "invalid_base")
    for key in ("digest", "config_digest"):
        digest = value[key]
        _require(
            type(digest) is str and digest.startswith("sha256:") and _sha256(digest[7:]),
            "invalid_base",
        )
    _require(
        type(value["python_version"]) is str
        and _PYTHON_VERSION.fullmatch(value["python_version"]) is not None,
        "invalid_base",
    )
    _require(value["tag_reference"] == "3.12-slim-bookworm", "invalid_base")


def _manylinux(tag: str) -> bool:
    if tag in ("manylinux1_x86_64", "manylinux2010_x86_64", "manylinux2014_x86_64"):
        return True
    match = re.fullmatch(r"manylinux_2_([0-9]{1,2})_x86_64", tag)
    # Bookworm provides glibc 2.36; accept declared glibc requirements only up
    # to that version. An actual build/import test still has to prove operation.
    return match is not None and 5 <= int(match[1]) <= 36 and str(int(match[1])) == match[1]


def _wheel(row: dict) -> None:
    filename = row["filename"]
    _require(
        type(filename) is str and len(filename) <= 240 and _FILENAME.fullmatch(filename)
        and ".." not in filename,
        "invalid_filename",
    )
    parts = filename[:-4].split("-")
    _require(len(parts) in (5, 6), "invalid_filename")
    _require(
        parts[0] == row["name"].replace("-", "_")
        and parts[1] == row["version"].replace("-", "_"),
        "invalid_filename",
    )
    if len(parts) == 6:
        _require(_BUILD.fullmatch(parts[2]) is not None, "invalid_filename")
    python_tag, abi_tag, platform_tag = parts[-3:]
    if platform_tag == "any":
        compatible = python_tag in ("py3", "py2.py3") and abi_tag == "none"
    else:
        platforms = platform_tag.split(".")
        valid_platforms = len(platforms) <= 4 and len(platforms) == len(set(platforms))
        valid_platforms = valid_platforms and all(_manylinux(tag) for tag in platforms)
        stable_python = python_tag in tuple("cp3" + str(minor) for minor in range(2, 13))
        valid_python = (
            python_tag == "cp312" and abi_tag in ("cp312", "abi3", "none")
        ) or (stable_python and abi_tag == "abi3")
        compatible = valid_platforms and valid_python
    _require(compatible, "incompatible_wheel")
    _require(_GPU_COMPONENT.search(filename.lower()) is None, "forbidden_dependency")
    if row["name"] == "torch":
        _require(
            row["version"] == "2.14.0+cpu" and len(parts) == 5
            and python_tag == "cp312" and abi_tag == "cp312" and platform_tag != "any",
            "invalid_torch_contract",
        )


def _url(row: dict) -> None:
    url = row["url"]
    _require(type(url) is str and 1 <= len(url) <= 2048, "invalid_url")
    _require(all(33 <= ord(char) <= 126 for char in url), "invalid_url")
    try:
        parsed = urlsplit(url)
    except ValueError:
        raise LockError("invalid_url") from None
    hosts = (
        ("download.pytorch.org", "download-r2.pytorch.org")
        if row["name"] == "torch" else ("files.pythonhosted.org",)
    )
    _require(
        url.startswith("https://") and parsed.scheme == "https"
        and parsed.netloc in hosts and not parsed.query
        and not parsed.fragment and "?" not in url and "#" not in url,
        "invalid_url",
    )
    path = parsed.path
    _require(_URL_PATH.fullmatch(path) is not None, "invalid_url")
    # Official CPU wheel URLs can encode the local-version '+'. No encoded
    # separators, unreserved aliases, nested escapes, or lowercase escapes are
    # accepted. Literal '+' is also safe and has no query-string semantics here.
    decoded = path.replace("%2B", "+")
    _require("%" not in decoded, "invalid_url")
    components = decoded.split("/")[1:]
    _require(all(part and part not in (".", "..") for part in components), "invalid_url")
    _require(components[-1] == row["filename"], "invalid_url")
    if row["name"] == "torch":
        _require(path.startswith("/whl/cpu/"), "invalid_torch_contract")


def _requirements(value: object) -> None:
    _require(type(value) is list, "invalid_requirements")
    _require(
        MINIMUM_REQUIREMENTS <= len(value) <= MAXIMUM_REQUIREMENTS,
        "requirement_count_exceeded",
    )
    previous, total, names = None, 0, set()
    for row in value:
        _require(type(row) is dict and row.keys() == REQUIREMENT_KEYS, "invalid_requirement_shape")
        name = row["name"]
        _require(
            type(name) is str and 1 <= len(name) <= 128 and _NAME.fullmatch(name) is not None,
            "invalid_name",
        )
        _require(
            name not in _FORBIDDEN_NAMES and not name.startswith(("nvidia-", "cupy"))
            and _GPU_COMPONENT.search(name) is None,
            "forbidden_dependency",
        )
        _require(previous is None or previous < name, "requirements_not_sorted_unique")
        _require(
            type(row["version"]) is str and _VERSION.fullmatch(row["version"]) is not None,
            "invalid_version",
        )
        _wheel(row)
        _url(row)
        _require(_sha256(row["sha256"]), "invalid_wheel_sha256")
        size = row["bytes"]
        _require(type(size) is int and size >= 1, "invalid_wheel_size")
        _require(size <= MAXIMUM_WHEEL_BYTES, "wheel_budget_exceeded")
        total += size
        _require(total <= MAXIMUM_TOTAL_BYTES, "total_budget_exceeded")
        previous = name
        names.add(name)
    _require(REQUIRED_NAMES <= names, "missing_required_dependency")


def validate_lock(raw: bytes, expected_sha256: str) -> dict:
    """Validate bounded canonical ASCII bytes and return a fresh exact snapshot.

    Canonical bytes are sorted compact ASCII JSON followed by exactly one LF.
    The caller supplies the separately trusted SHA-256 of those complete bytes,
    including the LF. This API does not discover that pin, access the network,
    read wheels, or authorize a run.
    """
    _require(type(raw) is bytes, "invalid_raw_type")
    _require(1 <= len(raw) <= MAXIMUM_LOCK_BYTES, "lock_budget_exceeded")
    _require(_sha256(expected_sha256), "invalid_expected_sha256")
    _require(hashlib.sha256(raw).hexdigest() == expected_sha256, "lock_hash_mismatch")
    value = _parse(raw)
    _require(
        value["schema"] == SCHEMA and value["platform"] == PLATFORM
        and value["python_abi"] == PYTHON_ABI,
        "unsupported_release_contract",
    )
    _base(value["base"])
    _requirements(value["requirements"])
    policy = value["policy"]
    _require(type(policy) is dict and policy.keys() == POLICY.keys(), "invalid_policy")
    _require(
        all(type(policy[key]) is type(wanted) and policy[key] == wanted
            for key, wanted in POLICY.items()),
        "invalid_policy",
    )
    _require(_canonical(value) == raw, "noncanonical_json")
    return {
        **value, "base": dict(value["base"]), "policy": dict(policy),
        "requirements": [dict(row) for row in value["requirements"]],
    }


def requirements_text(validated: dict) -> bytes:
    """Render deterministic hashed direct URLs, rechecking the supplied snapshot.

    This helper's self-computed digest is a structural revalidation, not an
    externally trusted release decision. Call validate_lock with that external
    pin before passing its result here. No pip command or installation is run.
    """
    raw = _canonical(validated)
    snapshot = validate_lock(raw, hashlib.sha256(raw).hexdigest())
    return "".join(
        f"{row['name']} @ {row['url']} --hash=sha256:{row['sha256']}\n"
        for row in snapshot["requirements"]
    ).encode("ascii")
