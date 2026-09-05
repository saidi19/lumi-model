"""Bounded HTTPS acquisition of public Be-DSTC2 rights documents.

This module downloads only an exact allowlist of public documentation.  It
cannot fetch datasets, archives, audio, or arbitrary URLs, and it never treats
the acquired material as legal clearance.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import socket
import ssl
import stat
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .posttraining_bedstc2_upstream_rights_review_v1 import (
    file_sha256,
    finish_marker,
    resolve_repo_path,
    sha256_bytes,
    sha256_value,
    start_marker,
    strict_json_bytes,
)

PLAN_SCHEMA_VERSION = (
    "lumi.sft-bedstc2-authoritative-rights-document-acquisition-plan.v0.1"
)
RECEIPT_SCHEMA_VERSION = (
    "lumi.sft-bedstc2-authoritative-rights-document-acquisition-receipt.v0.1"
)
MANIFEST_SCHEMA_VERSION = (
    "lumi.sft-bedstc2-authoritative-rights-private-manifest.v0.1"
)
VERIFICATION_SCHEMA_VERSION = (
    "lumi.sft-bedstc2-authoritative-rights-document-acquisition-verification.v0.1"
)
PLAN_STATUS = "authorized_bounded_public_https_rights_document_acquisition"
RECEIPT_STATUS = "PASS_public_rights_documents_acquired_nonproduction"
VERIFICATION_STATUS = "PASS_verified_public_rights_documents_nonproduction"

EXPECTED_PLAN_PATH = Path(
    "configs/lumi-4b-sft-bedstc2-authoritative-rights-document-acquisition-v0.1.json"
)
EXPECTED_AUTHORIZATION_TEXT = (
    "amra ekhon road map er kothay achi and ar koto step lagbe shes hote and I "
    "approve the bounded public-HTTPS-only Be-DSTC2 authoritative-rights-document "
    "acquisition described above: maximum 12 documents, 24 responses, 16 MiB inbound "
    "and 20 MiB writes; no datasets, archives, audio, login, POST, outreach, promotion, "
    "seal, or training; followed by two zero-network factual verification passes. Keep "
    "production\\_eligible=false pending qualified rights-owner or legal approval."
)
EXPECTED_AUTHORIZATION_SHA256 = (
    "cae31f30cfbfd24a9d81d4d982676e7194f3b3c12bd2bc4078a4b162de0fb163"
)
EXPECTED_LIMITS = {
    "maximum_documents": 12,
    "planned_documents": 8,
    "maximum_http_responses": 24,
    "maximum_inbound_bytes": 16_777_216,
    "maximum_combined_write_bytes": 20_971_520,
    "maximum_seconds": 900,
    "network_retries": 0,
    "gpu_use": 0,
    "inference_calls": 0,
}
EXPECTED_AUTHORIZATION = {
    "public_https_document_get_authorized": True,
    "dataset_archive_or_audio_download_authorized": False,
    "login_cookie_form_or_post_authorized": False,
    "outreach_authorized": False,
    "candidate_source_or_registry_mutation_authorized": False,
    "promotion_authorized": False,
    "dataset_seal_authorized": False,
    "sft_training_authorized": False,
    "qualified_legal_approval_claimed": False,
}
EXPECTED_PROHIBITIONS = {
    "no_dataset_release_archive_or_audio": True,
    "no_login_cookie_form_post_or_outreach": True,
    "no_source_candidate_baseline_test_or_registry_access": True,
    "no_promotion_seal_training_gpu_or_inference": True,
    "no_retry_after_durable_start_marker": True,
}
EXPECTED_CODE_PATHS = {
    "implementation": (
        "src/lumi/posttraining_bedstc2_authoritative_rights_acquisition_v1.py"
    ),
    "runner": "scripts/acquire_sft_bedstc2_authoritative_rights_documents_v1.py",
    "verifier": (
        "scripts/verify_sft_bedstc2_authoritative_rights_acquisition_v1.py"
    ),
    "test": "tests/test_bedstc2_authoritative_rights_acquisition_v1.py",
}
EXPECTED_OUTPUT = {
    "private_root": (
        "data/private/lumi-sft-bedstc2-authoritative-rights-documents-v0.1/"
        "attempt-001"
    ),
    "private_manifest_path": (
        "data/private/lumi-sft-bedstc2-authoritative-rights-documents-v0.1/"
        "attempt-001/artifact-manifest.json"
    ),
    "receipt_path": (
        "evidence/lumi-4b-sft-bedstc2-authoritative-rights-document-"
        "acquisition-v0.1.json"
    ),
    "verification_path": (
        "evidence/lumi-4b-sft-bedstc2-authoritative-rights-document-"
        "acquisition-verification-v0.1.json"
    ),
}
ACQUISITION_MARKER = b'{"started":"bedstc2_authoritative_rights_acquisition_v0.1"}\n'
MANIFEST_MARKER = b'{"started":"bedstc2_authoritative_rights_manifest_v0.1"}\n'
RECEIPT_MARKER = b'{"started":"bedstc2_authoritative_rights_receipt_v0.1"}\n'
VERIFICATION_MARKER = (
    b'{"started":"bedstc2_authoritative_rights_acquisition_verification_v0.1"}\n'
)


def _doc(
    identity: str,
    filename: str,
    url: str,
    media_type: str,
    maximum_bytes: int,
    source_class: str,
) -> dict[str, Any]:
    return {
        "id": identity,
        "filename": filename,
        "url": url,
        "media_type": media_type,
        "maximum_bytes": maximum_bytes,
        "source_class": source_class,
    }


EXPECTED_DOCUMENTS = [
    _doc(
        "microsoft_dstc_event",
        "01-microsoft-dstc-event.html",
        "https://www.microsoft.com/en-us/research/event/dialog-state-tracking-challenge/",
        "text/html",
        2_097_152,
        "official_event_page",
    ),
    _doc(
        "dstc2_acl_anthology_landing",
        "02-dstc2-acl-anthology-landing.html",
        "https://aclanthology.org/W14-4337/",
        "text/html",
        1_048_576,
        "primary_paper_landing_page",
    ),
    _doc(
        "matthen_dstc_readme_exact_revision",
        "03-matthen-dstc-readme.md",
        "https://raw.githubusercontent.com/matthen/dstc/7897c9beb5fe21fdfb57dcbe1e8f8294c2bcdead/README.md",
        "text/plain",
        1_048_576,
        "organizer_associated_repository_readme",
    ),
    _doc(
        "matthen_dstc_license_exact_revision",
        "04-matthen-dstc-license.txt",
        "https://raw.githubusercontent.com/matthen/dstc/7897c9beb5fe21fdfb57dcbe1e8f8294c2bcdead/LICENSE",
        "text/plain",
        262_144,
        "organizer_associated_repository_license",
    ),
    _doc(
        "dstc2_primary_paper",
        "05-dstc2-primary-paper.pdf",
        "https://aclanthology.org/W14-4337.pdf",
        "application/pdf",
        4_194_304,
        "primary_research_paper",
    ),
    _doc(
        "code_mixed_dialog_paper",
        "06-code-mixed-dialog-paper.pdf",
        "https://aclanthology.org/C18-1319.pdf",
        "application/pdf",
        4_194_304,
        "primary_research_paper",
    ),
    _doc(
        "code_mixed_dialog_readme_exact_revision",
        "07-code-mixed-dialog-readme.md",
        "https://raw.githubusercontent.com/sumanbanerjee1/Code-Mixed-Dialog/9df1d4dc800548a883f8bc1a9ce4116c77aebc02/README.md",
        "text/plain",
        1_048_576,
        "downstream_repository_readme",
    ),
    _doc(
        "code_mixed_dialog_license_exact_revision",
        "08-code-mixed-dialog-license.txt",
        "https://raw.githubusercontent.com/sumanbanerjee1/Code-Mixed-Dialog/9df1d4dc800548a883f8bc1a9ce4116c77aebc02/LICENSE",
        "text/plain",
        262_144,
        "downstream_repository_license",
    ),
]
_DOCUMENT_BY_URL = {item["url"]: item for item in EXPECTED_DOCUMENTS}
_SHA256_RE = __import__("re").compile(r"^[0-9a-f]{64}$")
_ARCHIVE_OR_AUDIO_MAGIC = (
    b"PK\x03\x04",
    b"\x1f\x8b",
    b"7z\xbc\xaf\x27\x1c",
    b"Rar!",
    b"ID3",
    b"OggS",
)


def canonical_json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _validate_url(url: str) -> None:
    if url not in _DOCUMENT_BY_URL:
        raise ValueError("URL is not in the exact rights-document allowlist")
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or parsed.fragment
    ):
        raise ValueError("invalid HTTPS rights-document URL")
    try:
        ipaddress.ip_address(parsed.hostname)
    except ValueError:
        pass
    else:
        raise ValueError("IP-literal rights-document URLs are prohibited")


def _validate_public_dns(url: str) -> None:
    hostname = urllib.parse.urlsplit(url).hostname
    if hostname is None:
        raise ValueError("rights-document hostname missing")
    addresses = {
        item[4][0]
        for item in socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
    }
    if not addresses or any(not ipaddress.ip_address(item).is_global for item in addresses):
        raise ValueError("rights-document hostname did not resolve exclusively public")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        raise ValueError("HTTP redirects are not authorized")


@dataclass
class NetworkBudget:
    maximum_responses: int = EXPECTED_LIMITS["maximum_http_responses"]
    maximum_bytes: int = EXPECTED_LIMITS["maximum_inbound_bytes"]
    requests_started: int = 0
    responses_observed: int = 0
    inbound_bytes: int = 0

    def start_request(self) -> None:
        self.requests_started += 1
        if self.requests_started > self.maximum_responses:
            raise ValueError("HTTP response cap would be exceeded")

    def observe_response(self) -> None:
        self.responses_observed += 1
        if self.responses_observed > self.maximum_responses:
            raise ValueError("HTTP response cap exceeded")

    def add_bytes(self, amount: int) -> None:
        self.inbound_bytes += amount
        if self.inbound_bytes > self.maximum_bytes:
            raise ValueError("inbound byte cap exceeded")


class _BoundedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, budget: NetworkBudget, context: ssl.SSLContext) -> None:
        super().__init__(context=context)
        self._budget = budget

    def https_open(self, req: urllib.request.Request) -> Any:
        _validate_url(req.full_url)
        _validate_public_dns(req.full_url)
        return super().https_open(req)


def build_network_opener(budget: NetworkBudget) -> urllib.request.OpenerDirector:
    context = ssl.create_default_context()
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoRedirect(),
        _BoundedHTTPSHandler(budget, context),
    )


def _mkdir_private(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("private output directory metadata changed")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_fresh_stream(path: Path, response: Any, spec: Mapping[str, Any], budget: NetworkBudget) -> tuple[int, str]:
    _mkdir_private(path.parent)
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite a rights document")
    part = path.with_name(f".{path.name}.part")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(part, flags, 0o600)
    digest = hashlib.sha256()
    written = 0
    prefix = b""
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("private document inode changed")
        while True:
            block = response.read(65_536)
            if not block:
                break
            if not isinstance(block, bytes):
                raise ValueError("HTTP body was not bytes")
            budget.add_bytes(len(block))
            written += len(block)
            if written > int(spec["maximum_bytes"]):
                raise ValueError("per-document byte cap exceeded")
            if len(prefix) < 16:
                prefix += block[: 16 - len(prefix)]
            digest.update(block)
            offset = 0
            while offset < len(block):
                count = os.write(descriptor, block[offset:])
                if count <= 0:
                    raise OSError("short rights-document write")
                offset += count
        if written <= 0:
            raise ValueError("empty rights document")
        if any(prefix.startswith(magic) for magic in _ARCHIVE_OR_AUDIO_MAGIC):
            raise ValueError("archive or audio payload prohibited")
        if spec["media_type"] == "application/pdf" and not prefix.startswith(b"%PDF-"):
            raise ValueError("PDF document envelope changed")
        if spec["media_type"].startswith("text/") and b"\x00" in prefix:
            raise ValueError("text document has binary envelope")
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        part.unlink(missing_ok=True)
        raise
    os.close(descriptor)
    os.link(part, path, follow_symlinks=False)
    part.unlink()
    _fsync_directory(path.parent)
    final = os.lstat(path)
    if not stat.S_ISREG(final.st_mode) or final.st_nlink != 1 or final.st_size != written:
        raise ValueError("final rights-document metadata changed")
    return written, digest.hexdigest()


def _header_media_type(response: Any) -> str:
    value = response.headers.get("Content-Type", "")
    return value.split(";", 1)[0].strip().lower()


def acquire_one_document(
    spec: Mapping[str, Any],
    destination: Path,
    budget: NetworkBudget,
    opener: Any,
) -> dict[str, Any]:
    _validate_url(str(spec["url"]))
    request = urllib.request.Request(
        str(spec["url"]),
        method="GET",
        headers={
            "Accept": str(spec["media_type"]),
            "Accept-Encoding": "identity",
            "User-Agent": "Lumi-Be-DSTC2-rights-documents-v0.1",
        },
    )
    budget.start_request()
    with opener.open(request, timeout=45) as response:
        budget.observe_response()
        if getattr(response, "status", None) != 200:
            raise ValueError("rights-document HTTP status changed")
        if response.geturl() != spec["url"]:
            raise ValueError("rights-document final URL changed")
        if response.headers.get("Content-Encoding", "identity").lower() not in ("", "identity"):
            raise ValueError("compressed HTTP body prohibited")
        media_type = _header_media_type(response)
        if media_type != spec["media_type"]:
            raise ValueError("rights-document media type changed")
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError as error:
                raise ValueError("invalid Content-Length") from error
            if declared < 1 or declared > int(spec["maximum_bytes"]):
                raise ValueError("declared document length outside cap")
            if budget.inbound_bytes + declared > budget.maximum_bytes:
                raise ValueError("declared total inbound bytes outside cap")
        size, digest = _write_fresh_stream(destination, response, spec, budget)
    return {
        "id": spec["id"],
        "filename": spec["filename"],
        "url": spec["url"],
        "source_class": spec["source_class"],
        "media_type": spec["media_type"],
        "bytes": size,
        "sha256": digest,
    }


def _json_seal(value: dict[str, Any], key: str) -> dict[str, Any]:
    value[key] = sha256_value({name: item for name, item in value.items() if name != key})
    return value


def _write_marker_payload(path: Path, marker: bytes, value: Mapping[str, Any]) -> bytes:
    payload = canonical_json_file_bytes(value)
    identity = start_marker(path, marker)
    finish_marker(path, marker, payload, identity)
    return payload


def load_plan(path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resolved = resolve_repo_path(root, path)
    if resolved != resolve_repo_path(root, EXPECTED_PLAN_PATH):
        raise ValueError("authoritative-rights acquisition plan path changed")
    info = resolved.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("authoritative-rights acquisition plan metadata changed")
    value = strict_json_bytes(resolved.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("authoritative-rights acquisition plan root changed")
    expected_keys = {
        "schema_version", "status", "authorization_text", "authorization_sha256",
        "authorization", "limits", "documents", "network_contract", "prohibitions",
        "output", "implementation", "runner", "verifier", "test",
    }
    if (
        set(value) != expected_keys
        or value.get("schema_version") != PLAN_SCHEMA_VERSION
        or value.get("status") != PLAN_STATUS
        or value.get("authorization_text") != EXPECTED_AUTHORIZATION_TEXT
        or value.get("authorization_sha256") != EXPECTED_AUTHORIZATION_SHA256
        or sha256_bytes(EXPECTED_AUTHORIZATION_TEXT.encode()) != EXPECTED_AUTHORIZATION_SHA256
        or value.get("authorization") != EXPECTED_AUTHORIZATION
        or value.get("limits") != EXPECTED_LIMITS
        or value.get("documents") != EXPECTED_DOCUMENTS
        or value.get("prohibitions") != EXPECTED_PROHIBITIONS
        or value.get("output") != EXPECTED_OUTPUT
        or value.get("network_contract")
        != {
            "method": "GET", "https_only": True, "redirects_allowed": 0,
            "cookies_allowed": False, "proxy_environment_allowed": False,
            "content_encoding": "identity", "exact_url_allowlist": True,
            "public_dns_only": True,
        }
    ):
        raise ValueError("authoritative-rights acquisition plan contract changed")
    if len(EXPECTED_DOCUMENTS) > EXPECTED_LIMITS["maximum_documents"]:
        raise ValueError("planned rights documents exceed cap")
    for name, expected_path in EXPECTED_CODE_PATHS.items():
        binding = value.get(name)
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256"}
            or binding.get("path") != expected_path
            or not isinstance(binding.get("sha256"), str)
            or _SHA256_RE.fullmatch(binding["sha256"]) is None
            or file_sha256(resolve_repo_path(root, expected_path)) != binding["sha256"]
        ):
            raise ValueError("authoritative-rights acquisition code binding changed")
    private_root = resolve_repo_path(root, EXPECTED_OUTPUT["private_root"])
    if not private_root.is_relative_to(resolve_repo_path(root, "data/private")):
        raise ValueError("private rights-document root escaped")
    for key in ("receipt_path", "verification_path"):
        if resolve_repo_path(root, EXPECTED_OUTPUT[key]).parent != resolve_repo_path(root, "evidence"):
            raise ValueError("rights-document evidence path changed")
    return value


def preflight_acquisition(plan: Mapping[str, Any], root: Path) -> tuple[Path, Path, Path, Path]:
    private_root = resolve_repo_path(root, plan["output"]["private_root"])
    manifest = resolve_repo_path(root, plan["output"]["private_manifest_path"])
    receipt = resolve_repo_path(root, plan["output"]["receipt_path"])
    verification = resolve_repo_path(root, plan["output"]["verification_path"])
    marker = private_root / ".acquisition-started-v0.1"
    if any(path.exists() or path.is_symlink() for path in (private_root, receipt, verification)):
        raise FileExistsError("authoritative-rights acquisition already started")
    return private_root, manifest, receipt, marker


def run_acquisition(plan: Mapping[str, Any], root: Path, opener: Any | None = None) -> dict[str, Any]:
    private_root = resolve_repo_path(root, plan["output"]["private_root"])
    manifest_path = resolve_repo_path(root, plan["output"]["private_manifest_path"])
    receipt_path = resolve_repo_path(root, plan["output"]["receipt_path"])
    raw_root = private_root / "raw"
    budget = NetworkBudget()
    client = opener if opener is not None else build_network_opener(budget)
    documents: list[dict[str, Any]] = []
    for spec in EXPECTED_DOCUMENTS:
        documents.append(
            acquire_one_document(spec, raw_root / str(spec["filename"]), budget, client)
        )
    if budget.responses_observed != len(EXPECTED_DOCUMENTS):
        raise ValueError("HTTP response accounting changed")
    manifest = _json_seal(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": "private_exact_public_document_inventory",
            "documents": documents,
            "document_count": len(documents),
            "document_bytes": sum(item["bytes"] for item in documents),
            "raw_text_in_git": False,
            "production_eligible": False,
            "manifest_sha256": "",
        },
        "manifest_sha256",
    )
    manifest_payload = _write_marker_payload(manifest_path, MANIFEST_MARKER, manifest)
    receipt = _json_seal(
        {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": RECEIPT_STATUS,
            "authorization_sha256": EXPECTED_AUTHORIZATION_SHA256,
            "documents": documents,
            "accounting": {
                "documents": len(documents),
                "http_requests_started": budget.requests_started,
                "http_responses_observed": budget.responses_observed,
                "inbound_bytes": budget.inbound_bytes,
                "network_retries": 0,
            },
            "private_manifest": {
                "bytes": len(manifest_payload),
                "sha256": sha256_bytes(manifest_payload),
                "internal_sha256": manifest["manifest_sha256"],
            },
            "boundaries": {
                "datasets_archives_or_audio_acquired": False,
                "login_cookie_form_post_or_outreach_used": False,
                "source_candidate_baseline_test_or_registry_accessed": False,
                "promotion_seal_or_training_performed": False,
                "gpu_or_inference_used": False,
            },
            "raw_document_text_in_receipt": False,
            "production_eligible": False,
            "qualified_review_pending": True,
            "receipt_sha256": "",
        },
        "receipt_sha256",
    )
    receipt_payload = canonical_json_file_bytes(receipt)
    prospective = (
        budget.inbound_bytes + len(ACQUISITION_MARKER) + len(MANIFEST_MARKER)
        + len(manifest_payload) + len(RECEIPT_MARKER) + len(receipt_payload)
    )
    if prospective > EXPECTED_LIMITS["maximum_combined_write_bytes"]:
        raise ValueError("combined acquisition write cap exceeded")
    _write_marker_payload(receipt_path, RECEIPT_MARKER, receipt)
    return receipt


def read_exact_file(path: Path, *, expected_bytes: int, expected_sha256: str) -> bytes:
    flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != expected_bytes:
            raise ValueError("rights-document metadata changed")
        chunks: list[bytes] = []
        remaining = expected_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if len(payload) != expected_bytes or sha256_bytes(payload) != expected_sha256:
        raise ValueError("rights-document bytes changed")
    return payload


def read_json_file(path: Path, *, maximum_bytes: int = 1_048_576) -> tuple[dict[str, Any], bytes]:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("JSON artifact metadata changed")
    if not 1 <= info.st_size <= maximum_bytes:
        raise ValueError("JSON artifact size changed")
    payload = read_exact_file(path, expected_bytes=info.st_size, expected_sha256=file_sha256(path))
    value = strict_json_bytes(payload)
    if not isinstance(value, dict) or canonical_json_file_bytes(value) != payload:
        raise ValueError("JSON artifact canonical encoding changed")
    return value, payload


def validate_seal(value: Mapping[str, Any], key: str) -> None:
    seal = value.get(key)
    if not isinstance(seal, str) or seal != sha256_value({k: v for k, v in value.items() if k != key}):
        raise ValueError("JSON artifact internal seal changed")


def replay_acquired_inventory(plan: Mapping[str, Any], root: Path) -> tuple[dict[str, Any], dict[str, Any], list[bytes]]:
    private_root = resolve_repo_path(root, plan["output"]["private_root"])
    manifest_path = resolve_repo_path(root, plan["output"]["private_manifest_path"])
    receipt_path = resolve_repo_path(root, plan["output"]["receipt_path"])
    manifest, manifest_payload = read_json_file(manifest_path)
    receipt, receipt_payload = read_json_file(receipt_path)
    validate_seal(manifest, "manifest_sha256")
    validate_seal(receipt, "receipt_sha256")
    documents = manifest.get("documents")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("status") != "private_exact_public_document_inventory"
        or not isinstance(documents, list)
        or len(documents) != len(EXPECTED_DOCUMENTS)
        or receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != RECEIPT_STATUS
        or receipt.get("documents") != documents
        or receipt.get("private_manifest")
        != {
            "bytes": len(manifest_payload), "sha256": sha256_bytes(manifest_payload),
            "internal_sha256": manifest["manifest_sha256"],
        }
        or receipt.get("production_eligible") is not False
    ):
        raise ValueError("rights-document receipt or manifest contract changed")
    expected_names = {item["filename"] for item in EXPECTED_DOCUMENTS}
    raw_root = private_root / "raw"
    actual_names = {item.name for item in raw_root.iterdir() if item.is_file()}
    if actual_names != expected_names:
        raise ValueError("private rights-document inventory changed")
    payloads = []
    for spec, observed in zip(EXPECTED_DOCUMENTS, documents, strict=True):
        if any(observed.get(key) != spec[key] for key in ("id", "filename", "url", "source_class", "media_type")):
            raise ValueError("rights-document identity changed")
        payloads.append(
            read_exact_file(
                raw_root / spec["filename"],
                expected_bytes=int(observed["bytes"]),
                expected_sha256=str(observed["sha256"]),
            )
        )
    return manifest, receipt, payloads


def assemble_acquisition_verification(
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    payloads: list[bytes],
) -> dict[str, Any]:
    value = {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": VERIFICATION_STATUS,
        "checks": {
            "exact_document_inventory_replayed": True,
            "all_document_sizes_and_hashes_replayed": True,
            "private_manifest_and_receipt_seals_verified": True,
            "network_used_during_verification": False,
            "production_boundary_preserved": True,
        },
        "documents": len(payloads),
        "document_bytes": sum(len(item) for item in payloads),
        "manifest_internal_sha256": manifest["manifest_sha256"],
        "receipt_internal_sha256": receipt["receipt_sha256"],
        "production_eligible": False,
        "qualified_review_pending": True,
        "verification_sha256": "",
    }
    return _json_seal(value, "verification_sha256")


def build_acquisition_verification(plan: Mapping[str, Any], root: Path) -> dict[str, Any]:
    manifest, receipt, payloads = replay_acquired_inventory(plan, root)
    return assemble_acquisition_verification(manifest, receipt, payloads)


__all__ = [name for name in globals() if name.isupper()] + [
    "NetworkBudget", "acquire_one_document", "assemble_acquisition_verification",
    "build_acquisition_verification",
    "build_network_opener", "canonical_json_file_bytes", "load_plan",
    "preflight_acquisition", "read_exact_file", "read_json_file",
    "replay_acquired_inventory", "run_acquisition", "validate_seal",
]
