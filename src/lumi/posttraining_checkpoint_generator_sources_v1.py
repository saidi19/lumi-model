"""Fixed generator source add-on for a separately admitted fresh software root.

No acquisition, native execution, model access or host approval is performed.
Only a new unfrozen exact OwnedRootfs may receive these absent destinations.
The fixed helper is authenticated before any write. Source hashes bind bytes,
not origin, dependency closure or permission. Started failures taint the tree;
there is no retry or cleanup here. The existing owner retains cleanup authority.
"""

from __future__ import annotations

from io import BytesIO

from . import posttraining_checkpoint_guard_sources_v1 as sources
from . import posttraining_sft_oci_rootfs_v1 as rootfs

SCHEMA = "lumi.checkpoint-generator-source-installation.v0.1"
ENTRY_PATH = "opt/lumi-release/guardian/generator_entry.py"
HELPER_PATH = "opt/lumi-release/guardian/guarded_import_helpers.py"
HELPER_BYTES = 14356
HELPER_SHA256 = "687a2989ccefdf2466dc77cda62d99d5c6428bc60c74e0e32e15991cec120690"
REQUIRED_SOURCES = frozenset({
    "lumi/__init__.py", "lumi/posttraining_checkpoint_guard_fixture_v1.py",
    "lumi/posttraining_checkpoint_prepared_generation_v1.py",
    "lumi/posttraining_checkpoint_generator_packet_v1.py",
    "lumi/posttraining_checkpoint_import_plan_v1.py",
    "lumi/posttraining_final_base_metadata_v1.py", "lumi/parameter_count.py",
    "lumi/configuration_lumi.py", "lumi/modeling_lumi.py",
})
MAXIMUM_ADDON_BYTES = sources.MAXIMUM_ADDON_BYTES + HELPER_BYTES
ERROR_CODES = frozenset({"binding_rejected", "runtime_rejected", "installation_rejected"})


class GeneratorSourceError(ValueError):
    def __init__(self, code="installation_rejected"):
        self.code = code if type(code) is str and code in ERROR_CODES else "installation_rejected"
        super().__init__(self.code)


def _need(value, code="installation_rejected"):
    if not value:
        raise GeneratorSourceError(code) from None


def validate_bundle(files, manifest, entry, helper, *, expected_manifest_sha256,
                    expected_entry_sha256):
    """Pure bounded exact-byte check; no code evaluation or filesystem reads."""
    try:
        summary = sources.validate_bundle(files, manifest, entry,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_entry_sha256=expected_entry_sha256)
        _need(REQUIRED_SOURCES <= set(files) and type(helper) is bytes
              and len(helper) == HELPER_BYTES and sources._digest(helper) == HELPER_SHA256,
              "binding_rejected")
        _need(expected_manifest_sha256 != "0" * 64 and expected_entry_sha256 != "0" * 64,
              "binding_rejected")
        return {**summary, "addon_files": summary["addon_files"] + 1,
                "addon_bytes": summary["addon_bytes"] + HELPER_BYTES,
                "helper_sha256": HELPER_SHA256}
    except GeneratorSourceError:
        raise
    except BaseException:
        raise GeneratorSourceError("binding_rejected") from None


def install_generator_addon(owned, files, manifest, entry, helper, *,
                            expected_manifest_sha256, expected_entry_sha256):
    """Install once, using only identity-tracked owned writes, then freeze."""
    try:
        sources._runtime()
    except BaseException:
        raise GeneratorSourceError("runtime_rejected") from None
    summary = validate_bundle(files, manifest, entry, helper,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_entry_sha256=expected_entry_sha256)
    _need(type(owned) is rootfs.OwnedRootfs)
    started, failure = False, False
    try:
        owned._guard()
        _need(not owned._frozen and owned._installed_files == 0)
        owned._verify_tree()
        additions = {ENTRY_PATH: entry, HELPER_PATH: helper, sources.MANIFEST_PATH: manifest,
                     **{sources.SOURCE_PREFIX + name: body for name, body in files.items()}}
        prefix = ("opt", "lumi-release", "importer")
        _need(not any(parts[:3] == prefix for parts in owned._nodes))
        planned, total = set(owned._nodes), owned._content_bytes
        for name, body in additions.items():
            parts = tuple(name.split("/"))
            _need(parts not in owned._nodes and parts not in owned._seen)
            for depth in range(1, len(parts)):
                parent = parts[:depth]
                _need(parent not in owned._nodes or owned._nodes[parent][1] == "directory")
                planned.add(parent)
            planned.add(parts)
            total += len(body)
            _need(len(body) <= owned._limits["file"])
        _need(len(planned) <= owned._limits["entries"] and total <= owned._limits["total"]
              and summary["addon_bytes"] <= MAXIMUM_ADDON_BYTES)
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
        raise GeneratorSourceError("installation_rejected") from None
    return {"schema": SCHEMA, "status": "installed", **summary,
            "installation_frozen": True, "base_origin_verified": False,
            "dependency_closure_verified": False, "filesystem_immutable": False,
            "native_generator_verified": False, "real_import_authorized": False,
            "training_authorized": False, "production_eligible": False}
