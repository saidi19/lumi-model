"""Bounded MASSIVE bn-BD train acquisition and raw-text-free aggregate audit.

The executable contract is deliberately narrow: five exact public HTTPS objects,
one exact train Parquet object, one streaming aggregate pass over six allowlisted
columns, and no candidate construction.  Source strings are observed only inside
the aggregate loop and are never returned, logged, or serialized.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import socket
import ssl
import stat
import sys
import unicodedata
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PLAN_SCHEMA = "lumi.sft-massive-bn-bd-train-acquisition-plan.v0.1"
PLAN_STATUS = "authorized_bounded_massive_bn_bd_train_acquisition_and_aggregate_audit"
REPORT_SCHEMA = "lumi.sft-massive-bn-bd-train-acquisition-report.v0.1"
PASS_STATUS = "PASS_massive_bn_bd_train_aggregate_gate_ready_nonproduction"
STOP_STATUS = "STOP_massive_bn_bd_train_aggregate_gate_failed_nonproduction"
VERIFICATION_SCHEMA = "lumi.sft-massive-bn-bd-train-acquisition-verification.v0.1"
VERIFICATION_STATUS = "PASS_two_pass_zero_network_massive_acquisition_verification"

AUTHORIZATION_TEXT = "ok next step shuru kor"
AUTHORIZATION_SHA256 = "1046cda26a21ff4994e4a452870466e609fe93d825cad3622e9fd069fb18b35e"

EXPECTED_SOURCE = {
    "dataset_id": "AmazonScience/massive",
    "revision": "d1792d0e4f32a74d43725cc1b901f600dad990dd",
    "path": "bn-BD/train/0000.parquet",
    "bytes": 1_071_668,
    "sha256": "947f65c28cce0d9a57c16bc1a99df08f84e9b7599dff25f47fd5b9760256c17d",
    "rows": 11_514,
    "url": (
        "https://huggingface.co/datasets/AmazonScience/massive/resolve/"
        "d1792d0e4f32a74d43725cc1b901f600dad990dd/bn-BD/train/0000.parquet"
        "?download=true"
    ),
    "local_name": "AmazonScience_massive/bn-BD/train/0000.parquet",
    "media_class": "parquet",
}

EXPECTED_RIGHTS_DOCUMENTS = [
    {
        "id": "massive_notice",
        "url": (
            "https://raw.githubusercontent.com/alexa/massive/"
            "f966f21846043aabef9b0f974fa7970027f43738/NOTICE.md"
        ),
        "local_name": "MASSIVE-NOTICE.md",
        "bytes": 500,
        "sha256": "db1466b051afd1d9ad34b7bd6346a1857f568a0c5f6534168813b4ad3c26013e",
        "media_class": "text",
    },
    {
        "id": "massive_license",
        "url": (
            "https://raw.githubusercontent.com/alexa/massive/"
            "f966f21846043aabef9b0f974fa7970027f43738/LICENSE.txt"
        ),
        "local_name": "MASSIVE-LICENSE.txt",
        "bytes": 11_542,
        "sha256": "a601496ab5359ce7ea3dacbb0652f07d7f5700081cb47e2da3fc3d1bff6393d7",
        "media_class": "text",
    },
    {
        "id": "massive_third_party",
        "url": (
            "https://raw.githubusercontent.com/alexa/massive/"
            "f966f21846043aabef9b0f974fa7970027f43738/THIRD-PARTY.md"
        ),
        "local_name": "MASSIVE-THIRD-PARTY.md",
        "bytes": 11_646,
        "sha256": "36583837e567f370c4b477bd1b6bdcac50cf26bc30d9b56162e4ec9e590b75eb",
        "media_class": "text",
    },
    {
        "id": "slurp_text_license",
        "url": (
            "https://raw.githubusercontent.com/pswietojanski/slurp/"
            "8eb16545762be97ace75334109d73824217311f1/LICENSE.txt"
        ),
        "local_name": "SLURP-LICENSE.txt",
        "bytes": 583,
        "sha256": "97547dd6236684935697b26629fb6e70b3be4ed535c2d85fdc053a6389192849",
        "media_class": "text",
    },
]

EXPECTED_LIMITS = {
    "maximum_https_responses": 12,
    "maximum_inbound_bytes": 2_097_152,
    "maximum_private_write_bytes": 8_388_608,
    "maximum_source_row_reads": 11_514,
    "maximum_test_dev_validation_row_reads": 0,
    "maximum_candidate_reads": 0,
    "maximum_gpu_actions": 0,
    "maximum_inference_calls": 0,
    "maximum_training_actions": 0,
    "maximum_runtime_seconds": 900,
    "network_retries": 0,
    "logical_gets": 5,
}

EXPECTED_TOP_LEVEL_FIELDS = (
    "id",
    "locale",
    "partition",
    "scenario",
    "intent",
    "utt",
    "annot_utt",
    "worker_id",
    "slot_method",
    "judgments",
)
AUDIT_COLUMNS = ("locale", "partition", "scenario", "intent", "utt", "annot_utt")
_INITIAL_URLS = {EXPECTED_SOURCE["url"]} | {item["url"] for item in EXPECTED_RIGHTS_DOCUMENTS}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ANNOTATION_RE = re.compile(r"\[\s*([^:\[\]]+?)\s*:\s*([^\[\]]+?)\s*\]")
_FORBIDDEN_REPORT_KEYS = (
    "source_text",
    "raw_text",
    "candidate_text",
    "utterance_text",
    "row_ids",
    "id_values",
    "intent_values",
    "scenario_values",
    "slot_labels",
    "exception_text",
    "samples",
    "examples",
)
_FORBIDDEN_EXACT_REPORT_KEYS = {"utt", "annot_utt", "worker_id", "judgments", "id"}

ACQUISITION_MARKER = b'{"started":"massive_bn_bd_train_acquisition_v0.1"}\n'


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _seal(value: dict[str, Any], key: str) -> dict[str, Any]:
    value[key] = sha256_value({name: item for name, item in value.items() if name != key})
    return value


def _verify_seal(value: Mapping[str, Any], key: str) -> None:
    expected = value.get(key)
    if not isinstance(expected, str) or _SHA256_RE.fullmatch(expected) is None:
        raise ValueError(f"invalid {key}")
    unsigned = {name: item for name, item in value.items() if name != key}
    if sha256_value(unsigned) != expected:
        raise ValueError(f"{key} changed")


def write_json_fresh(path: str | Path, value: Mapping[str, Any]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("refusing to overwrite JSON artifact")
    payload = json_file_bytes(value)
    temporary = destination.with_name(f".{destination.name}.part")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            count = os.write(descriptor, payload[offset:])
            if count <= 0:
                raise OSError("short JSON write")
            offset += count
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    os.close(descriptor)
    os.link(temporary, destination, follow_symlinks=False)
    temporary.unlink()
    return len(payload)


def _resolve_repo_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("repository path escaped")
    return path


def _resolve_private_path(root: Path, value: str) -> Path:
    workspace = root.parent.resolve()
    allowed = (workspace / "lumi-artifacts").resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(allowed):
        raise ValueError("private path escaped lumi-artifacts")
    return path


def _validate_code_and_predecessor_bindings(plan: Mapping[str, Any], root: Path) -> None:
    code = plan.get("code_bindings")
    if not isinstance(code, dict) or set(code) != {"implementation", "runner", "verifier", "test"}:
        raise ValueError("code bindings changed")
    for label, binding in code.items():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError(f"invalid code binding {label}")
        path = _resolve_repo_path(root, str(binding["path"]))
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise ValueError(f"code binding changed: {label}")
    predecessors = plan.get("predecessor_bindings")
    if not isinstance(predecessors, dict) or not predecessors:
        raise ValueError("predecessor bindings missing")
    for label, binding in predecessors.items():
        if not isinstance(binding, dict) or set(binding) - {"path", "sha256", "internal_sha256"}:
            raise ValueError(f"invalid predecessor binding {label}")
        path = _resolve_repo_path(root, str(binding.get("path", "")))
        if not path.is_file() or sha256_file(path) != binding.get("sha256"):
            raise ValueError(f"predecessor binding changed: {label}")
        internal = binding.get("internal_sha256")
        if internal is not None:
            value = _read_object(path)
            seals = [
                item
                for key, item in value.items()
                if key.endswith("_sha256") and isinstance(item, str) and item == internal
            ]
            if not seals:
                raise ValueError(f"predecessor internal seal changed: {label}")


def verify_runtime_binding(plan: Mapping[str, Any]) -> None:
    runtime = plan.get("runtime_binding")
    if not isinstance(runtime, dict):
        raise ValueError("runtime binding missing")
    import pyarrow  # type: ignore[import-not-found]
    import pyarrow._parquet  # type: ignore[import-not-found]

    expected_python = Path(str(runtime["python_executable"])).resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise ValueError("bound Python executable changed")
    if pyarrow.__version__ != runtime.get("pyarrow_version"):
        raise ValueError("PyArrow version changed")
    checks = {
        "python_executable": Path(str(runtime["python_executable"])).resolve(),
        "pyarrow_init": Path(pyarrow.__file__).resolve(),
        "pyarrow_parquet_extension": Path(pyarrow._parquet.__file__).resolve(),
    }
    for label, path in checks.items():
        if not path.is_file() or sha256_file(path) != runtime.get(f"{label}_sha256"):
            raise ValueError(f"runtime binding changed: {label}")


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _read_object(path)
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != PLAN_STATUS:
        raise ValueError("unexpected MASSIVE acquisition plan identity")
    _verify_seal(plan, "plan_sha256")
    authorization = plan.get("authorization")
    if not isinstance(authorization, dict) or authorization != {
        "text": AUTHORIZATION_TEXT,
        "text_sha256": AUTHORIZATION_SHA256,
        "massive_bn_bd_train_get_authorized": True,
        "four_pinned_rights_document_gets_authorized": True,
        "aggregate_only_audit_authorized": True,
        "candidate_construction_authorized": False,
        "test_dev_validation_audio_authorized": False,
        "registry_promotion_seal_or_training_authorized": False,
        "gpu_or_inference_authorized": False,
    }:
        raise ValueError("authorization contract changed")
    if hashlib.sha256(AUTHORIZATION_TEXT.encode()).hexdigest() != AUTHORIZATION_SHA256:
        raise ValueError("authorization text hash changed")
    if plan.get("source") != EXPECTED_SOURCE:
        raise ValueError("MASSIVE source binding changed")
    if plan.get("rights_documents") != EXPECTED_RIGHTS_DOCUMENTS:
        raise ValueError("rights-document bindings changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("execution limits changed")
    if plan.get("audit_contract") != {
        "allowlisted_columns": list(AUDIT_COLUMNS),
        "expected_top_level_fields": list(EXPECTED_TOP_LEVEL_FIELDS),
        "retain_only_aggregate_counts_cardinalities_and_fingerprints": True,
        "read_id_worker_id_slot_method_or_judgments_values": False,
        "raw_source_candidate_id_or_exception_text_retained": False,
        "single_streaming_source_pass": True,
        "two_zero_network_verification_passes": True,
    }:
        raise ValueError("aggregate audit contract changed")
    prohibitions = plan.get("prohibitions")
    if (
        not isinstance(prohibitions, dict)
        or not prohibitions
        or not all(value is True for value in prohibitions.values())
    ):
        raise ValueError("prohibitions changed")
    _validate_code_and_predecessor_bindings(plan, root)
    output = plan.get("output")
    if not isinstance(output, dict):
        raise ValueError("output contract missing")
    _resolve_private_path(root, str(output.get("private_root", "")))
    for key in ("report_path", "verification_path"):
        path_value = _resolve_repo_path(root, str(output.get(key, "")))
        if path_value.parent != (root / "evidence").resolve():
            raise ValueError("evidence path changed")
    verify_runtime_binding(plan)
    return plan


@dataclass
class NetworkBudget:
    maximum_responses: int = EXPECTED_LIMITS["maximum_https_responses"]
    maximum_bytes: int = EXPECTED_LIMITS["maximum_inbound_bytes"]
    requests_started: int = 0
    responses_observed: int = 0
    inbound_bytes: int = 0

    def start_request(self) -> None:
        self.requests_started += 1
        if self.requests_started > self.maximum_responses:
            raise ValueError("HTTPS request cap exceeded")

    def observe_response(self) -> None:
        self.responses_observed += 1
        if self.responses_observed > self.maximum_responses:
            raise ValueError("HTTPS response cap exceeded")

    def add_bytes(self, amount: int) -> None:
        if amount < 0:
            raise ValueError("negative inbound byte count")
        self.inbound_bytes += amount
        if self.inbound_bytes > self.maximum_bytes:
            raise ValueError("inbound byte cap exceeded")


def _validate_https_shape(url: str) -> urllib.parse.SplitResult:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or parsed.fragment
    ):
        raise ValueError("invalid HTTPS URL")
    try:
        ipaddress.ip_address(parsed.hostname)
    except ValueError:
        pass
    else:
        raise ValueError("IP-literal URL prohibited")
    return parsed


def _validate_initial_url(url: str) -> None:
    _validate_https_shape(url)
    if url not in _INITIAL_URLS:
        raise ValueError("URL is not in exact acquisition allowlist")


def _validate_redirect_url(source_url: str, redirect_url: str) -> None:
    source = _validate_https_shape(source_url)
    target = _validate_https_shape(redirect_url)
    host = target.hostname or ""
    if source.hostname == "huggingface.co":
        if host != "huggingface.co" and not host.endswith(".hf.co"):
            raise ValueError("Hugging Face redirect host changed")
    elif source.hostname == "raw.githubusercontent.com":
        if host != "raw.githubusercontent.com":
            raise ValueError("rights-document redirect host changed")
    elif source.hostname and source.hostname.endswith(".hf.co"):
        if host != "huggingface.co" and not host.endswith(".hf.co"):
            raise ValueError("Hugging Face redirect chain changed")
    else:
        raise ValueError("redirect source host is not authorized")


def _validate_public_dns(url: str) -> None:
    hostname = _validate_https_shape(url).hostname
    assert hostname is not None
    addresses = {item[4][0] for item in socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)}
    if not addresses or any(not ipaddress.ip_address(item).is_global for item in addresses):
        raise ValueError("HTTPS host did not resolve exclusively public")


class _BoundedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, budget: NetworkBudget) -> None:
        super().__init__(context=ssl.create_default_context())
        self._budget = budget

    def https_open(self, request: urllib.request.Request) -> Any:
        _validate_public_dns(request.full_url)
        self._budget.start_request()
        return super().https_open(request)


class _BoundedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, budget: NetworkBudget, maximum_redirects: int = 5) -> None:
        super().__init__()
        self._budget = budget
        self._maximum_redirects = maximum_redirects
        self.redirects = 0

    def redirect_request(
        self,
        request: urllib.request.Request,
        response: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> urllib.request.Request | None:
        self._budget.observe_response()
        while True:
            block = response.read(65_536)
            if not block:
                break
            self._budget.add_bytes(len(block))
        self.redirects += 1
        if self.redirects > self._maximum_redirects:
            raise ValueError("redirect cap exceeded")
        absolute = urllib.parse.urljoin(request.full_url, new_url)
        _validate_redirect_url(request.full_url, absolute)
        redirected = super().redirect_request(request, response, code, message, headers, absolute)
        if redirected is None:
            raise ValueError("redirect was not a GET")
        redirected.headers.pop("Cookie", None)
        redirected.headers.pop("Authorization", None)
        return redirected


def build_network_opener(
    budget: NetworkBudget,
) -> tuple[urllib.request.OpenerDirector, _BoundedRedirectHandler]:
    redirects = _BoundedRedirectHandler(budget)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), redirects, _BoundedHTTPSHandler(budget)
    )
    return opener, redirects


def _mkdir_private(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("private directory metadata changed")


def start_marker(path: Path) -> None:
    _mkdir_private(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        os.write(descriptor, ACQUISITION_MARKER)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exact_response(
    response: Any,
    destination: Path,
    specification: Mapping[str, Any],
    budget: NetworkBudget,
) -> tuple[int, str]:
    _mkdir_private(destination.parent)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("refusing to overwrite acquired object")
    part = destination.with_name(f".{destination.name}.part")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(part, flags, 0o600)
    digest = hashlib.sha256()
    written = 0
    prefix = b""
    try:
        while True:
            block = response.read(65_536)
            if not block:
                break
            if not isinstance(block, bytes):
                raise ValueError("HTTP body was not bytes")
            budget.add_bytes(len(block))
            written += len(block)
            if written > int(specification["bytes"]):
                raise ValueError("acquired object exceeded exact size")
            if len(prefix) < 4:
                prefix += block[: 4 - len(prefix)]
            digest.update(block)
            offset = 0
            while offset < len(block):
                count = os.write(descriptor, block[offset:])
                if count <= 0:
                    raise OSError("short private write")
                offset += count
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        part.unlink(missing_ok=True)
        raise
    os.close(descriptor)
    actual_sha256 = digest.hexdigest()
    if written != specification["bytes"] or actual_sha256 != specification["sha256"]:
        part.unlink(missing_ok=True)
        raise ValueError("acquired object size or SHA-256 changed")
    if specification["media_class"] == "parquet" and prefix != b"PAR1":
        part.unlink(missing_ok=True)
        raise ValueError("Parquet envelope changed")
    if specification["media_class"] == "text" and b"\x00" in prefix:
        part.unlink(missing_ok=True)
        raise ValueError("rights-document text envelope changed")
    os.link(part, destination, follow_symlinks=False)
    part.unlink()
    return written, actual_sha256


def acquire_one(
    specification: Mapping[str, Any],
    destination: Path,
    budget: NetworkBudget,
    opener: urllib.request.OpenerDirector,
) -> dict[str, Any]:
    url = str(specification["url"])
    _validate_initial_url(url)
    request = urllib.request.Request(
        url,
        method="GET",
        headers={
            "Accept": "application/octet-stream"
            if specification["media_class"] == "parquet"
            else "text/plain",
            "Accept-Encoding": "identity",
            "User-Agent": "Lumi-MASSIVE-bn-BD-train-audit-v0.1",
        },
    )
    with opener.open(request, timeout=60) as response:
        budget.observe_response()
        if getattr(response, "status", None) != 200:
            raise ValueError("unexpected HTTPS status")
        final_url = response.geturl()
        if final_url != url:
            _validate_redirect_url(url, final_url)
        encoding = response.headers.get("Content-Encoding", "identity").lower()
        if encoding not in ("", "identity"):
            raise ValueError("compressed HTTP transfer prohibited")
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) != int(specification["bytes"]):
            raise ValueError("Content-Length changed")
        size, digest = _write_exact_response(response, destination, specification, budget)
    return {
        "id": specification.get("id", "massive_bn_bd_train"),
        "bytes": size,
        "sha256": digest,
        "logical_path": specification["local_name"],
        "media_class": specification["media_class"],
    }


def classify_script(value: str) -> str:
    bengali = any("\u0980" <= character <= "\u09ff" for character in value)
    latin = any(unicodedata.name(character, "").startswith("LATIN ") for character in value)
    if bengali and latin:
        return "bengali_latin_mixed"
    if bengali:
        return "bengali_only"
    if latin:
        return "latin_only"
    return "other"


def parse_annotated_utterance(value: str) -> dict[str, Any]:
    digests: set[str] = set()
    annotations = 0
    cursor = 0
    malformed = False
    for match in _ANNOTATION_RE.finditer(value):
        between = value[cursor : match.start()]
        if "[" in between or "]" in between:
            malformed = True
        label = match.group(1).strip()
        slot_value = match.group(2).strip()
        if not label or not slot_value:
            malformed = True
        else:
            digests.add(hashlib.sha256(label.casefold().encode("utf-8")).hexdigest())
            annotations += 1
        cursor = match.end()
    if "[" in value[cursor:] or "]" in value[cursor:]:
        malformed = True
    return {
        "annotation_count": annotations,
        "malformed": malformed,
        "slot_label_digests": digests,
    }


def _scalar_digest(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("scenario or intent scalar type changed")
    return hashlib.sha256(
        canonical_json_bytes({"type": type(value).__name__, "value": value})
    ).hexdigest()


def _string_observation(value: Any, aggregate: dict[str, Any]) -> str | None:
    if not isinstance(value, str):
        aggregate["non_string"] += 1
        return None
    aggregate["strings"] += 1
    if not value:
        aggregate["empty"] += 1
    if value != value.strip():
        aggregate["outer_whitespace"] += 1
    if value != unicodedata.normalize("NFC", value):
        aggregate["non_nfc"] += 1
    if any(unicodedata.category(character) == "Cc" for character in value):
        aggregate["control_character_rows"] += 1
    aggregate["utf8_bytes"] += len(value.encode("utf-8"))
    aggregate["maximum_characters"] = max(aggregate["maximum_characters"], len(value))
    aggregate["script_buckets"][classify_script(value)] += 1
    return value


def _empty_string_aggregate() -> dict[str, Any]:
    return {
        "strings": 0,
        "non_string": 0,
        "empty": 0,
        "outer_whitespace": 0,
        "non_nfc": 0,
        "control_character_rows": 0,
        "utf8_bytes": 0,
        "maximum_characters": 0,
        "script_buckets": {
            "bengali_only": 0,
            "latin_only": 0,
            "bengali_latin_mixed": 0,
            "other": 0,
        },
    }


def aggregate_parquet(path: str | Path) -> dict[str, Any]:
    import pyarrow.parquet as pq  # type: ignore[import-not-found]

    source = Path(path)
    parquet = pq.ParquetFile(source, pre_buffer=False, memory_map=False)
    metadata = parquet.metadata
    schema = parquet.schema_arrow
    descriptor = [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in schema
    ]
    schema_names = tuple(field.name for field in schema)
    schema_exact = schema_names == EXPECTED_TOP_LEVEL_FIELDS
    missing_audit_columns = sorted(set(AUDIT_COLUMNS) - set(schema_names))
    if missing_audit_columns:
        raise ValueError("required aggregate-audit columns missing")
    if metadata.num_rows != EXPECTED_SOURCE["rows"]:
        raise ValueError("Parquet row count changed")

    utterance = _empty_string_aggregate()
    annotated = _empty_string_aggregate()
    scenario_digests: set[str] = set()
    intent_digests: set[str] = set()
    slot_digests: set[str] = set()
    rows = 0
    locale_expected = 0
    locale_unexpected = 0
    partition_expected = 0
    partition_unexpected = 0
    scenario_null = 0
    intent_null = 0
    malformed_annotation_rows = 0
    rows_with_annotations = 0
    total_annotations = 0
    maximum_annotations = 0
    tool_ready_rows = 0

    for batch in parquet.iter_batches(
        batch_size=1024, columns=list(AUDIT_COLUMNS), use_threads=False
    ):
        if rows + batch.num_rows > EXPECTED_LIMITS["maximum_source_row_reads"]:
            raise ValueError("source-row read cap exceeded")
        values = batch.to_pydict()
        for locale, partition, scenario, intent, utt, annot_utt in zip(
            values["locale"],
            values["partition"],
            values["scenario"],
            values["intent"],
            values["utt"],
            values["annot_utt"],
            strict=True,
        ):
            rows += 1
            locale_expected += int(locale == "bn-BD")
            locale_unexpected += int(locale != "bn-BD")
            partition_expected += int(partition == "train")
            partition_unexpected += int(partition != "train")
            scenario_ok = scenario is not None
            intent_ok = intent is not None
            if scenario_ok:
                scenario_digests.add(_scalar_digest(scenario))
            else:
                scenario_null += 1
            if intent_ok:
                intent_digests.add(_scalar_digest(intent))
            else:
                intent_null += 1
            observed_utt = _string_observation(utt, utterance)
            observed_annot = _string_observation(annot_utt, annotated)
            annotation_ok = False
            if observed_annot is not None:
                parsed = parse_annotated_utterance(observed_annot)
                malformed_annotation_rows += int(parsed["malformed"])
                count = int(parsed["annotation_count"])
                total_annotations += count
                rows_with_annotations += int(count > 0)
                maximum_annotations = max(maximum_annotations, count)
                slot_digests.update(parsed["slot_label_digests"])
                annotation_ok = not parsed["malformed"]
            if (
                scenario_ok
                and intent_ok
                and isinstance(observed_utt, str)
                and bool(observed_utt)
                and annotation_ok
            ):
                tool_ready_rows += 1

    if rows != EXPECTED_SOURCE["rows"]:
        raise ValueError("streamed source-row count changed")
    checks = {
        "exact_object_rows": rows == EXPECTED_SOURCE["rows"],
        "exact_top_level_schema": schema_exact,
        "all_rows_bn_bd": locale_expected == rows and locale_unexpected == 0,
        "all_rows_train": partition_expected == rows and partition_unexpected == 0,
        "utterances_are_nonempty_strings": utterance["strings"] == rows
        and utterance["empty"] == 0
        and utterance["non_string"] == 0,
        "annotated_utterances_are_nonempty_strings": annotated["strings"] == rows
        and annotated["empty"] == 0
        and annotated["non_string"] == 0,
        "string_outer_whitespace_clear": utterance["outer_whitespace"] == 0
        and annotated["outer_whitespace"] == 0,
        "unicode_nfc_and_control_checks_clear": utterance["non_nfc"] == 0
        and annotated["non_nfc"] == 0
        and utterance["control_character_rows"] == 0
        and annotated["control_character_rows"] == 0,
        "scenario_and_intent_nonnull": scenario_null == 0 and intent_null == 0,
        "annotation_structure_parseable": malformed_annotation_rows == 0,
        "tool_projection_inputs_complete": tool_ready_rows == rows,
        "sensitive_columns_present_but_not_read": {"id", "worker_id", "judgments"}.issubset(
            schema_names
        )
        and not ({"id", "worker_id", "judgments"} & set(AUDIT_COLUMNS)),
        "single_streaming_pass_within_cap": rows <= EXPECTED_LIMITS["maximum_source_row_reads"],
    }
    return {
        "parquet": {
            "rows": metadata.num_rows,
            "row_groups": metadata.num_row_groups,
            "top_level_field_count": len(schema_names),
            "top_level_schema": descriptor,
            "schema_sha256": sha256_value(descriptor),
            "created_by_sha256": hashlib.sha256(
                str(metadata.created_by or "").encode("utf-8")
            ).hexdigest(),
        },
        "aggregate_audit": {
            "source_rows_read": rows,
            "locale": {"bn_bd": locale_expected, "unexpected": locale_unexpected},
            "partition": {"train": partition_expected, "unexpected": partition_unexpected},
            "scenario": {"cardinality": len(scenario_digests), "null_rows": scenario_null},
            "intent": {"cardinality": len(intent_digests), "null_rows": intent_null},
            "utterance": utterance,
            "annotated_utterance": annotated,
            "annotations": {
                "rows_with_annotations": rows_with_annotations,
                "total_annotations": total_annotations,
                "maximum_annotations_per_row": maximum_annotations,
                "malformed_rows": malformed_annotation_rows,
                "slot_label_cardinality": len(slot_digests),
            },
            "tool_adapter_feasibility": {
                "structurally_ready_rows": tool_ready_rows,
                "candidate_rows_constructed": 0,
            },
        },
        "checks": checks,
        "gate_pass": all(checks.values()),
    }


def validate_raw_text_free_report(report: Mapping[str, Any]) -> None:
    def walk(value: Any, key_path: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                lowered = str(key).casefold()
                substring_forbidden = (
                    any(forbidden in lowered for forbidden in _FORBIDDEN_REPORT_KEYS)
                    and lowered != "raw_text_retained"
                )
                if lowered in _FORBIDDEN_EXACT_REPORT_KEYS or substring_forbidden:
                    raise ValueError("raw-text-bearing report key prohibited")
                walk(item, key_path + (str(key),))
        elif isinstance(value, list):
            for item in value:
                walk(item, key_path)

    walk(report)
    if report.get("raw_text_retained") is not False:
        raise ValueError("report must retain zero raw text")
    if report.get("production_eligible") is not False:
        raise ValueError("MASSIVE acquisition cannot be production eligible")
    decision = report.get("decision")
    if isinstance(decision, Mapping) and decision.get("candidate_rows_constructed") != 0:
        raise ValueError("candidate construction is unauthorized")
    _verify_seal(report, "report_sha256")


def _inventory_entry(path: Path, logical_path: str, media_class: str) -> dict[str, Any]:
    return {
        "logical_path": logical_path,
        "media_class": media_class,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def run_acquisition(plan: Mapping[str, Any], *, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = plan["output"]
    private_root = _resolve_private_path(root, output["private_root"])
    raw_root = private_root / "raw"
    rights_root = private_root / "rights"
    manifest_path = private_root / "artifact-manifest.json"
    report_path = _resolve_repo_path(root, output["report_path"])
    budget = NetworkBudget()
    opener, redirects = build_network_opener(budget)

    source_path = raw_root / EXPECTED_SOURCE["local_name"]
    receipts = [acquire_one(EXPECTED_SOURCE, source_path, budget, opener)]
    for specification in EXPECTED_RIGHTS_DOCUMENTS:
        receipts.append(
            acquire_one(
                specification,
                rights_root / specification["local_name"],
                budget,
                opener,
            )
        )
    if len(receipts) != EXPECTED_LIMITS["logical_gets"]:
        raise ValueError("logical GET count changed")
    if budget.requests_started != budget.responses_observed:
        raise ValueError("HTTPS request/response accounting diverged")

    audit = aggregate_parquet(source_path)
    inventory = [_inventory_entry(source_path, EXPECTED_SOURCE["local_name"], "parquet")]
    inventory.extend(
        _inventory_entry(rights_root / item["local_name"], item["local_name"], "rights_document")
        for item in EXPECTED_RIGHTS_DOCUMENTS
    )
    manifest = _seal(
        {
            "schema_version": "lumi.sft-massive-bn-bd-private-inventory.v0.1",
            "status": "private_exact_train_and_rights_inventory",
            "objects": inventory,
            "object_count": len(inventory),
            "object_bytes": sum(item["bytes"] for item in inventory),
            "raw_text_in_git": False,
            "production_eligible": False,
            "manifest_sha256": "",
        },
        "manifest_sha256",
    )
    manifest_bytes = write_json_fresh(manifest_path, manifest)
    private_write_bytes = budget.inbound_bytes + len(ACQUISITION_MARKER) + manifest_bytes
    if private_write_bytes > EXPECTED_LIMITS["maximum_private_write_bytes"]:
        raise ValueError("private-write cap exceeded")

    gate_pass = bool(audit["gate_pass"])
    report = _seal(
        {
            "schema_version": REPORT_SCHEMA,
            "status": PASS_STATUS if gate_pass else STOP_STATUS,
            "authorization_text_sha256": AUTHORIZATION_SHA256,
            "plan_file_sha256": sha256_file(
                _resolve_repo_path(root, "configs/lumi-4b-sft-massive-bn-bd-acquisition-v0.1.json")
            ),
            "source": {
                key: EXPECTED_SOURCE[key]
                for key in ("dataset_id", "revision", "path", "bytes", "sha256", "rows")
            },
            "rights_documents": [
                {
                    "document_code": item["id"],
                    "bytes": item["bytes"],
                    "sha256": item["sha256"],
                }
                for item in EXPECTED_RIGHTS_DOCUMENTS
            ],
            "network_accounting": {
                "logical_gets": len(receipts),
                "https_requests": budget.requests_started,
                "https_responses": budget.responses_observed,
                "redirects": redirects.redirects,
                "inbound_bytes": budget.inbound_bytes,
                "retries": 0,
            },
            "private_accounting": {
                "object_count": len(inventory),
                "object_bytes": sum(item["bytes"] for item in inventory),
                "private_write_bytes": private_write_bytes,
                "manifest_sha256": manifest["manifest_sha256"],
                "manifest_file_sha256": sha256_file(manifest_path),
            },
            "parquet": audit["parquet"],
            "aggregate_audit": audit["aggregate_audit"],
            "checks": audit["checks"],
            "decision": {
                "candidate_construction_eligible_for_fresh_successor": gate_pass,
                "candidate_rows_constructed": 0,
                "active_tool_api_credit_rows": 0,
                "owner_review_completed": False,
                "registry_promotion_completed": False,
                "next_action": (
                    "prepare_separately_authorized_deterministic_adapter_candidate_gate"
                    if gate_pass
                    else "stop_fail_closed_before_candidate_construction"
                ),
            },
            "boundaries": {
                "test_dev_validation_or_audio_read": False,
                "id_worker_judgment_values_read": False,
                "candidate_or_registry_read_or_write": False,
                "gpu_inference_or_training_used": False,
                "network_after_acquisition": False,
                "qualified_legal_approval_claimed": False,
            },
            "raw_text_retained": False,
            "production_eligible": False,
            "report_sha256": "",
        },
        "report_sha256",
    )
    validate_raw_text_free_report(report)
    payload_bytes = len(json_file_bytes(report))
    if private_write_bytes + payload_bytes > EXPECTED_LIMITS["maximum_private_write_bytes"]:
        raise ValueError("combined bounded-write cap exceeded")
    write_json_fresh(report_path, report)
    return report


def preflight_paths(
    plan: Mapping[str, Any], *, repository_root: str | Path
) -> tuple[Path, Path, Path]:
    root = Path(repository_root).resolve()
    output = plan["output"]
    private_root = _resolve_private_path(root, output["private_root"])
    report_path = _resolve_repo_path(root, output["report_path"])
    verification_path = _resolve_repo_path(root, output["verification_path"])
    if any(
        path.exists() or path.is_symlink()
        for path in (private_root, report_path, verification_path)
    ):
        raise FileExistsError("MASSIVE acquisition successor already started")
    return private_root, report_path, verification_path


def _verify_inventory_pass(
    plan: Mapping[str, Any], report: Mapping[str, Any], root: Path
) -> dict[str, bool]:
    output = plan["output"]
    private_root = _resolve_private_path(root, output["private_root"])
    manifest_path = private_root / "artifact-manifest.json"
    manifest = _read_object(manifest_path)
    _verify_seal(manifest, "manifest_sha256")
    expected: list[tuple[Mapping[str, Any], Path]] = [
        (EXPECTED_SOURCE, private_root / "raw" / EXPECTED_SOURCE["local_name"])
    ]
    expected.extend(
        (item, private_root / "rights" / item["local_name"]) for item in EXPECTED_RIGHTS_DOCUMENTS
    )
    hashes_match = True
    modes_private = True
    for specification, path in expected:
        info = path.lstat()
        modes_private &= stat.S_ISREG(info.st_mode) and not bool(info.st_mode & 0o077)
        hashes_match &= (
            info.st_size == specification["bytes"] and sha256_file(path) == specification["sha256"]
        )
    return {
        "plan_and_predecessor_bindings_revalidated": True,
        "exact_private_inventory_rehashed": hashes_match,
        "private_files_are_not_group_or_world_accessible": modes_private,
        "private_manifest_internal_seal_valid": report["private_accounting"]["manifest_sha256"]
        == manifest["manifest_sha256"],
        "verification_pass_used_zero_network": True,
        "verification_pass_read_zero_source_rows": True,
    }


def _verify_aggregate_pass(report: Mapping[str, Any]) -> dict[str, bool]:
    validate_raw_text_free_report(report)
    rows = report["aggregate_audit"]["source_rows_read"]
    utterance = report["aggregate_audit"]["utterance"]
    annotated = report["aggregate_audit"]["annotated_utterance"]
    script_totals_match = (
        sum(utterance["script_buckets"].values()) == rows
        and sum(annotated["script_buckets"].values()) == rows
    )
    status_matches = (report["status"] == PASS_STATUS) == all(report["checks"].values())
    decision_matches = report["decision"][
        "candidate_construction_eligible_for_fresh_successor"
    ] == all(report["checks"].values())
    return {
        "raw_text_free_structure_revalidated": True,
        "aggregate_row_arithmetic_reconciled": report["parquet"]["rows"]
        == rows
        == EXPECTED_SOURCE["rows"],
        "aggregate_script_bucket_arithmetic_reconciled": script_totals_match,
        "status_matches_all_content_checks": status_matches,
        "candidate_successor_gate_matches_checks": decision_matches,
        "candidate_credit_and_construction_remain_zero": report["decision"][
            "candidate_rows_constructed"
        ]
        == 0
        and report["decision"]["active_tool_api_credit_rows"] == 0,
        "production_eligible_remains_false": report["production_eligible"] is False,
        "second_verification_pass_used_zero_network": True,
        "second_verification_pass_read_zero_source_rows": True,
    }


def build_verification(
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    repository_root: str | Path,
    report_file_sha256: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    pass_one = _verify_inventory_pass(plan, report, root)
    pass_two = _verify_aggregate_pass(report)
    if not all(pass_one.values()) or not all(pass_two.values()):
        raise ValueError("MASSIVE acquisition verification failed")
    return _seal(
        {
            "schema_version": VERIFICATION_SCHEMA,
            "status": VERIFICATION_STATUS,
            "report_file_sha256": report_file_sha256,
            "report_internal_sha256": report["report_sha256"],
            "passes": [
                {"pass": 1, "kind": "zero_network_inventory_rehash", "checks": pass_one},
                {"pass": 2, "kind": "zero_network_aggregate_reconciliation", "checks": pass_two},
            ],
            "network_responses": 0,
            "source_row_reads": 0,
            "candidate_reads": 0,
            "raw_text_retained": False,
            "production_eligible": False,
            "verification_sha256": "",
        },
        "verification_sha256",
    )


def validate_verification(value: Mapping[str, Any]) -> None:
    if (
        value.get("schema_version") != VERIFICATION_SCHEMA
        or value.get("status") != VERIFICATION_STATUS
        or value.get("network_responses") != 0
        or value.get("source_row_reads") != 0
        or value.get("candidate_reads") != 0
        or value.get("raw_text_retained") is not False
        or value.get("production_eligible") is not False
    ):
        raise ValueError("MASSIVE verification contract changed")
    passes = value.get("passes")
    if not isinstance(passes, list) or [item.get("pass") for item in passes] != [1, 2]:
        raise ValueError("exactly two verification passes required")
    if not all(all(item["checks"].values()) for item in passes):
        raise ValueError("verification check failed")
    _verify_seal(value, "verification_sha256")
