"""Closed importer add-on for an independently admitted, owned software root.

No archive acquisition, software-origin approval, native execution or real input
access occurs here. The caller first admits the base runtime independently.
This versioned adapter adds only a fixed entry, manifest and exact flat Python
source tree; it never widens the frozen OwnedRootfs.install_file allowlist.
Native installation requires Linux/root. Local tests replace only that check
and operate on tiny generated tar trees, not an actual software root.

The source manifest is content binding, not arbitrary-code safety, dependency
closure or hostile-owner immutability. Rootfs freeze is an adapter state; the
separate guardian must install the read-only mount and child containment.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
import threading
from io import BytesIO

from . import posttraining_sft_oci_rootfs_v1 as rootfs

SCHEMA = "lumi.checkpoint-import-source-manifest.v0.1"
RESULT_SCHEMA = "lumi.checkpoint-import-source-installation.v0.1"
ENTRY_PATH = "opt/lumi-release/guardian/import_entry.py"
MANIFEST_PATH = "opt/lumi-release/importer/manifest.json"
SOURCE_PREFIX = "opt/lumi-release/importer/src/"
MAXIMUM_FILES = 128
MAXIMUM_FILE_BYTES = 256 * 1024
MAXIMUM_SOURCE_BYTES = 4 * 1024**2
MAXIMUM_MANIFEST_BYTES = 65536
MAXIMUM_ADDON_BYTES = MAXIMUM_SOURCE_BYTES + MAXIMUM_MANIFEST_BYTES + MAXIMUM_FILE_BYTES
SOURCE_NAME = re.compile(r"lumi/[A-Za-z_][A-Za-z0-9_]*\.py\Z", re.ASCII)
REQUIRED = frozenset({"lumi/__init__.py", "lumi/posttraining_checkpoint_guard_fixture_v1.py"})
ERROR_CODES = frozenset({"source_contract_rejected", "source_binding_rejected",
                         "source_runtime_rejected", "source_installation_rejected"})


class SourceError(ValueError):
    def __init__(self, code="source_contract_rejected"):
        self.code = (code if type(code) is str and code in ERROR_CODES
                     else "source_contract_rejected")
        super().__init__(self.code)


def _need(value, code="source_contract_rejected"):
    if not value:
        raise SourceError(code) from None


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _pin(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def build_manifest(files):
    """Pure content contract, with no filesystem reads or imports of payloads."""
    _need(type(files) is dict and 2 <= len(files) <= MAXIMUM_FILES
          and REQUIRED <= set(files))
    total = 0
    rows = []
    for name, body in files.items():
        _need(type(name) is str and SOURCE_NAME.fullmatch(name)
              and type(body) is bytes and len(body) <= MAXIMUM_FILE_BYTES)
        total += len(body)
        _need(total <= MAXIMUM_SOURCE_BYTES)
        rows.append({"path": name, "bytes": len(body), "sha256": _digest(body)})
    result = json.dumps({"schema": SCHEMA, "files": sorted(rows, key=lambda row: row["path"])},
                        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                        allow_nan=False).encode("ascii") + b"\n"
    _need(len(result) <= MAXIMUM_MANIFEST_BYTES)
    return result


def validate_bundle(files, manifest, entry, *, expected_manifest_sha256, expected_entry_sha256):
    """Require the one canonical exact manifest; no duplicates or ignored fields."""
    _need(type(manifest) is bytes and 0 < len(manifest) <= MAXIMUM_MANIFEST_BYTES
          and type(entry) is bytes and 0 < len(entry) <= MAXIMUM_FILE_BYTES
          and _pin(expected_manifest_sha256) and _pin(expected_entry_sha256))
    _need(_digest(manifest) == expected_manifest_sha256
          and _digest(entry) == expected_entry_sha256, "source_binding_rejected")
    _need(build_manifest(files) == manifest, "source_binding_rejected")
    return {
        "source_manifest_sha256": expected_manifest_sha256,
        "entry_sha256": expected_entry_sha256,
        "source_files": len(files), "source_bytes": sum(map(len, files.values())),
        "addon_files": len(files) + 2,
        "addon_bytes": sum(map(len, files.values())) + len(manifest) + len(entry),
    }


def _runtime():
    _need(sys.platform == "linux" and platform.machine() == "x86_64"
          and sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12)
          and sys.flags.no_site == 1 and sys.dont_write_bytecode
          and os.getresuid() == os.getresgid() == (0, 0, 0)
          and threading.current_thread() is threading.main_thread()
          and threading.active_count() == 1
          and not any(name.split(".", 1)[0] in {
              "torch", "transformers", "numpy", "safetensors", "tokenizers", "tensorflow",
              "jax", "cupy", "cuda",
          } for name in sys.modules), "source_runtime_rejected")


def install_importer_addon(owned, files, manifest, entry, *, expected_manifest_sha256,
                           expected_entry_sha256):
    """One closed add-on installation then freeze; never overwrite or retry.

    The original exact OwnedRootfs implementation performs all no-follow writes
    and identity-tracked cleanup. A started failure taints it, including a
    failed freeze; no successful prefix may be resumed. The caller retains the
    tree and its partial evidence until separately authorized cleanup.
    """
    _runtime()  # Reject unsupported hosts before object/filesystem operations.
    summary = validate_bundle(files, manifest, entry,
                              expected_manifest_sha256=expected_manifest_sha256,
                              expected_entry_sha256=expected_entry_sha256)
    _need(type(owned) is rootfs.OwnedRootfs, "source_installation_rejected")
    started, failure = False, False
    try:
        owned._guard()
        _need(not owned._frozen, "source_installation_rejected")
        owned._verify_tree()
        additions = {ENTRY_PATH: entry, MANIFEST_PATH: manifest,
                     **{SOURCE_PREFIX + name: body for name, body in files.items()}}
        # The entire importer subtree must be fresh, even if an existing tree
        # would happen to contain matching bytes. Old guardian files may remain.
        prefix = ("opt", "lumi-release", "importer")
        _need(not any(parts[:3] == prefix for parts in owned._nodes),
              "source_installation_rejected")
        planned = set(owned._nodes)
        total = owned._content_bytes
        for name, body in additions.items():
            parts = tuple(name.split("/"))
            _need(parts not in owned._nodes and parts not in owned._seen,
                  "source_installation_rejected")
            for depth in range(1, len(parts)):
                parent = parts[:depth]
                _need(parent not in owned._nodes or owned._nodes[parent][1] == "directory",
                      "source_installation_rejected")
                planned.add(parent)
            planned.add(parts)
            total += len(body)
            _need(len(body) <= owned._limits["file"], "source_installation_rejected")
        _need(len(planned) <= owned._limits["entries"] and total <= owned._limits["total"]
              and summary["addon_bytes"] <= MAXIMUM_ADDON_BYTES,
              "source_installation_rejected")
        started = True
        for name, body in sorted(additions.items()):
            owned._guard()
            parts = tuple(name.split("/"))
            owned._write(parts, BytesIO(body), len(body), 0o444)
            owned._seen.add(parts)
            owned._installed_files += 1
            owned._installed_bytes += len(body)
        owned.freeze()
    except BaseException:
        failure = True
    if failure:
        if started:
            owned._tainted = True
        raise SourceError("source_installation_rejected") from None
    return {"schema": RESULT_SCHEMA, "status": "installed", **summary,
            "installation_frozen": True, "base_origin_verified_by_this_adapter": False,
            "filesystem_immutable": False, "native_guardian_verified": False,
            "real_import_authorized": False, "production_eligible": False}
