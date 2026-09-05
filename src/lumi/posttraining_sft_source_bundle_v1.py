"""Deterministic private SOURCE-REVIEW bundle, never a deployable training release.

Only explicit code seeds, their statically found local imports, and explicitly
selected control documents are read. No project module is imported/executed.
JSON references, data, evidence and credentials are not traversed. Static closure
does not establish dynamic dependency or Linux/CUDA runtime completeness.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath

SCHEMA = "lumi.sft-source-review-bundle.v0.1"
MAX_FILES = 256
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_BUNDLE_BYTES = 20 * 1024 * 1024
FIXED_DATE = (1980, 1, 1, 0, 0, 0)
MANIFEST = "SOURCE_REVIEW_MANIFEST.json"
NOTICE = "SOURCE_REVIEW_ONLY.txt"
NOTICE_BYTES = (
    b"PRIVATE SOURCE REVIEW ONLY. NOT A DEPLOYABLE OR AUTHORIZED GPU RELEASE.\n"
    b"Includes historical worker code for review, not permission to run it.\n"
    b"No weights, data, owner approvals, real receipts, dependencies or credentials.\n"
    b"Do not run historical entrypoints. No install, transfer or launch is provided.\n"
)
ALLOWED_CONTROL = frozenset({
    "configs/lumi-4b-final-base-binding-v0.1.json",
    "configs/lumi-4b-sft-gpu-smoke-launch-contract-v0.1.json",
    "docs/sft-execution-status-v0.6.md",
})
_CREDENTIAL_PATTERNS = (
    rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b",
    rb"\bgithub_pat_[A-Za-z0-9_]{40,}\b",
    rb"(?im)^\s*(?:aws_secret_access_key|aws_session_token)\s*=\s*\S+",
)


class BundleError(ValueError):
    """Only fixed error codes; never a source line/path or exception text."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise BundleError(code)


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def allowed_path(name: str) -> bool:
    if type(name) is not str or not name or len(name) > 512:
        return False
    path = PurePosixPath(name)
    if str(path) != name or path.is_absolute() or ".." in path.parts:
        return False
    if any(ord(c) < 32 or ord(c) == 127 for c in name) or "\\" in name:
        return False
    if name in ALLOWED_CONTROL:
        return True
    return bool(re.fullmatch(r"(?:src/lumi/[a-zA-Z0-9_]+|scripts/[a-zA-Z0-9_]+|tests/test_[a-zA-Z0-9_]+)\.py", name))


def read_source(root: Path, name: str) -> bytes:
    """No-follow regular-file read; detect concurrent change/hardlinks/oversize."""
    require(allowed_path(name), "source_path_not_allowlisted")
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root, "unsafe_source_root")
    directory_fd = None
    try:
        directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        for part in PurePosixPath(name).parts[:-1]:
            child_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child_fd
        fd = os.open(PurePosixPath(name).name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
        with os.fdopen(fd, "rb", buffering=0) as stream:
            before = os.fstat(stream.fileno())
            require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "source_not_regular_unique")
            require(before.st_size <= MAX_FILE_BYTES, "source_file_too_large")
            raw = stream.read(MAX_FILE_BYTES + 1)
            after = os.fstat(stream.fileno())
        require(len(raw) == before.st_size, "source_read_size_changed")
        signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        require(signature(before) == signature(after), "source_changed_during_read")
        require(len(raw) <= MAX_FILE_BYTES, "source_file_too_large")
        require(not any(re.search(pattern, raw) for pattern in _CREDENTIAL_PATTERNS), "credential_signature_detected")
        return raw
    except OSError:
        raise BundleError("source_read_failed") from None
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def local_imports(name: str, raw: bytes) -> tuple[set[str], int]:
    """Conservative static imports only; count dynamic call sites explicitly."""
    try:
        tree = ast.parse(raw, filename="review_source")
    except (SyntaxError, ValueError, UnicodeError):
        raise BundleError("source_syntax_invalid") from None
    modules = set()
    dynamic = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("lumi."):
                    modules.add(alias.name[5:])
                elif alias.name == "lumi":
                    modules.add("__init__")
                elif alias.name.startswith("test_") and name.startswith("tests/"):
                    modules.add("TEST:" + alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                require(name.startswith("src/lumi/") and node.level == 1, "unsupported_relative_import")
                if node.module:
                    modules.add(node.module)
                else:
                    modules.update(alias.name for alias in node.names)
            elif node.module == "lumi":
                modules.add("__init__")
                # These source seeds use `from lumi import module`, not public classes.
                modules.update(alias.name for alias in node.names if alias.name.startswith("posttraining_"))
            elif node.module and node.module.startswith("lumi."):
                modules.add(node.module[5:])
            elif node.module and node.module.startswith("test_") and name.startswith("tests/"):
                modules.add("TEST:" + node.module)
        elif isinstance(node, ast.Call):
            function = node.func
            if (isinstance(function, ast.Name) and function.id in {"__import__", "exec", "eval"}) or (
                isinstance(function, ast.Attribute) and function.attr in {"import_module", "spec_from_file_location", "exec_module"}
            ):
                dynamic += 1
    paths = set()
    for module in modules:
        if module.startswith("TEST:"):
            path = "tests/" + module[5:].replace(".", "/") + ".py"
        else:
            path = "src/lumi/" + module.replace(".", "/") + ".py"
            paths.add("src/lumi/__init__.py")
        require(allowed_path(path), "unsupported_local_import")
        paths.add(path)
    return paths, dynamic


def collect(root: Path, seeds: list[str], controls: list[str]) -> tuple[dict[str, bytes], dict]:
    require(type(seeds) is list and type(controls) is list, "source_selection_type")
    require(bool(seeds) and len(seeds) == len(set(seeds)), "duplicate_or_empty_seeds")
    require(len(controls) == len(set(controls)) and set(controls) <= ALLOWED_CONTROL, "control_selection_not_allowlisted")
    require(all(allowed_path(p) and p.endswith(".py") for p in seeds), "seed_not_source_code")
    pending = set(seeds) | set(controls)
    files: dict[str, bytes] = {}
    byte_count = dynamic = 0
    while pending:
        name = min(pending)
        pending.remove(name)
        if name in files:
            continue
        require(len(files) < MAX_FILES, "source_file_count_cap")
        raw = read_source(root, name)
        byte_count += len(raw)
        require(byte_count <= MAX_SOURCE_BYTES, "source_total_bytes_cap")
        files[name] = raw
        if name.endswith(".py"):
            dependencies, sites = local_imports(name, raw)
            dynamic += sites
            pending.update(dependencies - files.keys())
    return files, {"static_local_import_closure_included": True, "dynamic_call_sites": dynamic,
                   "dynamic_dependency_completeness_verified": False, "runtime_dependency_completeness_verified": False}


def make_bundle(files: dict[str, bytes], *, inspection: dict) -> tuple[bytes, dict]:
    require(type(files) is dict and 0 < len(files) <= MAX_FILES, "bundle_file_count")
    for name, raw in files.items():
        require(allowed_path(name) and type(raw) is bytes, "bundle_entry_not_allowlisted")
        require(len(raw) <= MAX_FILE_BYTES, "source_file_too_large")
        require(not any(re.search(pattern, raw) for pattern in _CREDENTIAL_PATTERNS), "credential_signature_detected")
    require(sum(map(len, files.values())) <= MAX_SOURCE_BYTES, "source_total_bytes_cap")
    require(type(inspection) is dict and set(inspection) == {
        "static_local_import_closure_included", "dynamic_call_sites",
        "dynamic_dependency_completeness_verified", "runtime_dependency_completeness_verified"
    } and inspection["static_local_import_closure_included"] is True
        and type(inspection["dynamic_call_sites"]) is int and inspection["dynamic_call_sites"] >= 0
        and inspection["dynamic_dependency_completeness_verified"] is False
        and inspection["runtime_dependency_completeness_verified"] is False, "inspection_claim_invalid")
    manifest = {
        "schema_version": SCHEMA, "scope": "private_source_review_only",
        "deployment_ready": False, "gpu_execution_authorized": False,
        "production_eligible": False, "historical_worker_must_not_be_dispatched": True,
        "secret_scan_is_not_exhaustive": True, "inspection": inspection,
        "file_count": len(files), "source_bytes": sum(map(len, files.values())),
        "files": [{"path": p, "bytes": len(files[p]), "sha256": digest(files[p])} for p in sorted(files)],
    }
    contents = {**files, MANIFEST: canonical(manifest) + b"\n", NOTICE: NOTICE_BYTES}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        for name in sorted(contents):
            entry = zipfile.ZipInfo(name, FIXED_DATE)
            entry.create_system = 3
            entry.external_attr = (stat.S_IFREG | 0o600) << 16
            archive.writestr(entry, contents[name])
    raw = buffer.getvalue()
    require(len(raw) <= MAX_BUNDLE_BYTES, "bundle_bytes_cap")
    return raw, manifest


def verify_bundle(raw: bytes, expected_sha256: str, expected_files: dict[str, bytes]) -> dict:
    """Independently reopen ZIP in memory; never extract or import bundled code."""
    require(type(raw) is bytes and len(raw) <= MAX_BUNDLE_BYTES, "bundle_bytes_cap")
    require(digest(raw) == expected_sha256, "bundle_digest_mismatch")
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            require(len(names) == len(set(names)), "bundle_duplicate_entry")
            require(names == sorted(set(expected_files) | {MANIFEST, NOTICE}), "bundle_entry_set_mismatch")
            total = 0
            for entry in infos:
                require(entry.compress_type == zipfile.ZIP_STORED and entry.date_time == FIXED_DATE,
                        "bundle_encoding_changed")
                require(entry.create_system == 3 and entry.external_attr == (stat.S_IFREG | 0o600) << 16,
                        "bundle_private_mode_changed")
                require(not entry.flag_bits & 1 and entry.file_size <= MAX_FILE_BYTES, "bundle_entry_size_or_encryption")
                total += entry.file_size
                require(total <= MAX_BUNDLE_BYTES, "bundle_expanded_bytes_cap")
            require(archive.read(NOTICE) == NOTICE_BYTES, "bundle_notice_changed")
            manifest_raw = archive.read(MANIFEST)
            manifest = json.loads(manifest_raw)
            require(canonical(manifest) + b"\n" == manifest_raw, "bundle_manifest_noncanonical")
            require(manifest["schema_version"] == SCHEMA and manifest["scope"] == "private_source_review_only"
                    and manifest["historical_worker_must_not_be_dispatched"] is True
                    and manifest["secret_scan_is_not_exhaustive"] is True and manifest["deployment_ready"] is False
                    and manifest["gpu_execution_authorized"] is False and manifest["production_eligible"] is False,
                    "bundle_scope_changed")
            require(manifest["inspection"]["static_local_import_closure_included"] is True
                    and manifest["inspection"]["dynamic_dependency_completeness_verified"] is False
                    and manifest["inspection"]["runtime_dependency_completeness_verified"] is False,
                    "bundle_dependency_claim_changed")
            require(manifest["file_count"] == len(expected_files)
                    and manifest["source_bytes"] == sum(map(len, expected_files.values())), "bundle_count_mismatch")
            require(manifest["files"] == [{"path": p, "bytes": len(expected_files[p]), "sha256": digest(expected_files[p])}
                                          for p in sorted(expected_files)], "bundle_manifest_binding_mismatch")
            for name in sorted(expected_files):
                require(allowed_path(name), "bundle_entry_not_allowlisted")
                require(archive.read(name) == expected_files[name], "bundle_source_bytes_mismatch")
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile):
        raise BundleError("bundle_structural_verification_failed") from None
    return {"status": "PASS_source_bundle_integrity_only", "files_verified": len(expected_files),
            "source_bytes_verified": manifest["source_bytes"], "bundle_sha256": digest(raw),
            "files_extracted": 0, "bundled_code_executed": False, "deployment_ready": False}
