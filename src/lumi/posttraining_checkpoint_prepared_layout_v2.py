"""Install one fresh private generated-fixture layout, without admitting a host.

The caller independently authenticates an UNFROZEN newly extracted OwnedRootfs,
the source/entry controls, and the origin/freshness of generated input byte maps.
Every destination added here must be absent. Frozen extraction helpers track the
new software additions; freeze normalizes only this fresh private extracted copy,
not the original archive, repository, prior evidence or pre-existing user paths.

Checkpoint/tokenizer/output/mountpoint are four fixed fresh siblings. Only nodes
created by this invocation have ownership changed. No recursive cleanup, retry,
mount, worker launch or ML operation is provided. Started failures taint the
owned root and leave private partial evidence for separately authorized cleanup.
Native use requires Linux/root; local tests use filesystem/syscall doubles only.
The returned paths and byte-free native identities are PRIVATE controls and must
not be logged. Only the separate aggregate receipt is suitable for retention.
"""

from __future__ import annotations

import hashlib
import os
import stat
from io import BytesIO

from . import posttraining_checkpoint_guard_bootstrap_v3 as bootstrap
from . import posttraining_checkpoint_guard_sources_v1 as sources
from . import posttraining_checkpoint_import_rootfs_v2 as import_rootfs
from . import posttraining_checkpoint_input_mount_v2 as input_mount
from . import posttraining_checkpoint_prepared_controls_v1 as controls
from . import posttraining_sft_oci_rootfs_v1 as rootfs

SCHEMA = "lumi.prepared-checkpoint-layout.v0.1"
ENTRY_PATH = "opt/lumi-release/guardian/prepared_fixture_entry.py"
HELPER_PATH = "opt/lumi-release/guardian/guarded_import_helpers.py"
HELPER_BYTES = 14356
HELPER_SHA256 = "687a2989ccefdf2466dc77cda62d99d5c6428bc60c74e0e32e15991cec120690"
CONTROL_PREFIX = "opt/lumi-release/prepared-fixture-control/"
SIBLINGS = ("checkpoint", "tokenizer", "output", "mountpoint")
TARGET_MODES = {("inputs",): 0o755, ("inputs", "checkpoint"): 0o700,
                ("inputs", "tokenizer"): 0o700, ("outputs",): 0o700}
REQUIRED_SOURCES = frozenset({
    "lumi/__init__.py", "lumi/posttraining_checkpoint_guard_fixture_v1.py",
    "lumi/posttraining_checkpoint_import_runner_v3.py",
    "lumi/posttraining_checkpoint_import_runner_v2.py",
    "lumi/posttraining_checkpoint_resource_envelope_v1.py",
    "lumi/posttraining_checkpoint_host_contract_v1.py",
    "lumi/posttraining_checkpoint_import_plan_v1.py",
    "lumi/posttraining_final_base_metadata_v1.py",
    "lumi/posttraining_checkpoint_io_contract_v1.py",
    "lumi/posttraining_checkpoint_import_torch_v5.py",
    "lumi/posttraining_checkpoint_package_v2.py",
    "lumi/configuration_lumi.py", "lumi/modeling_lumi.py",
})
MAXIMUM_INSTALL_BYTES = 10 * 1024**2
MAXIMUM_EXTERNAL_NODES = 300
FALSE_FLAGS = ("base_origin_verified", "generation_origin_verified",
               "generation_freshness_verified",
               "filesystem_immutable", "external_writer_exclusion_verified",
               "native_guardian_verified", "host_admission_verified", "real_import_authorized",
               "training_authorized", "production_eligible", "retry_authorized")
ERROR_CODES = frozenset({"arguments_rejected", "binding_rejected", "layout_rejected",
                         "runtime_rejected", "installation_failed", "close_failed"})


class PreparedLayoutError(ValueError):
    def __init__(self, code="installation_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "installation_failed"
        super().__init__(self.code)


def _need(value, code="layout_rejected"):
    if not value:
        raise PreparedLayoutError(code) from None


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_bundle(*, files, manifest, entry, helper, control_raw, control_bodies,
                    generated_inputs, expected_manifest_sha256, expected_entry_sha256,
                    expected_control_sha256, expected_generation_nonce_sha256):
    """Pure pin/closed-control/byte-inventory validation before any filesystem mutation."""
    try:
        summary = sources.validate_bundle(files, manifest, entry,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_entry_sha256=expected_entry_sha256)
        _need(REQUIRED_SOURCES <= set(files), "binding_rejected")
        _need(type(helper) is bytes and len(helper) == HELPER_BYTES
              and _sha(helper) == HELPER_SHA256, "binding_rejected")
        verified = controls.validate_prepared_controls(control_raw, control_bodies,
            expected_control_sha256=expected_control_sha256,
            expected_entry_sha256=expected_entry_sha256,
            expected_source_manifest_sha256=expected_manifest_sha256,
            expected_generation_nonce_sha256=expected_generation_nonce_sha256,
            runtime_sources={name: files["lumi/" + name] for name in controls.RUNTIME_NAMES})
        _need(type(generated_inputs) is dict
              and set(generated_inputs) == {"checkpoint", "tokenizer"}, "arguments_rejected")
        copied, inventories, directories = {}, {}, {}
        for role in ("checkpoint", "tokenizer"):
            bodies = generated_inputs[role]
            _need(type(bodies) is dict and 0 < len(bodies) <= controls.MAXIMUM_INPUT_FILES
                  and all(type(raw) is bytes and 0 < len(raw) <= controls.MAXIMUM_INPUT_BYTES
                          for raw in bodies.values()), "arguments_rejected")
            inventory, parents = input_mount._manifest({name: len(raw)
                                                       for name, raw in bodies.items()})
            rows = verified["plan"][role + "_inventory"]
            _need({row["path"]: (row["bytes"], row["sha256"]) for row in rows}
                  == {name: (len(raw), _sha(raw)) for name, raw in bodies.items()},
                  "binding_rejected")
            copied[role], inventories[role], directories[role] = dict(bodies), inventory, parents
        input_bytes = sum(sum(map(len, rows.values())) for rows in copied.values())
        input_files = sum(map(len, copied.values()))
        _need(input_bytes <= controls.MAXIMUM_INPUT_BYTES
              and input_files <= controls.MAXIMUM_INPUT_FILES, "arguments_rejected")
        additions = {ENTRY_PATH: entry, HELPER_PATH: helper, sources.MANIFEST_PATH: manifest,
                     CONTROL_PREFIX + "control.json": control_raw,
                     **{CONTROL_PREFIX + name: raw for name, raw in control_bodies.items()},
                     **{sources.SOURCE_PREFIX + name: raw for name, raw in files.items()}}
        writes = sum(map(len, additions.values())) + input_bytes
        _need(writes <= MAXIMUM_INSTALL_BYTES, "arguments_rejected")
        return {"additions": additions, "generated_inputs": copied, "inventories": inventories,
                "directories": directories, "summary": summary,
                "control_sha256": expected_control_sha256,
                "generation_nonce_sha256": expected_generation_nonce_sha256,
                "write_bytes": writes, "input_bytes": input_bytes, "input_files": input_files}
    except PreparedLayoutError:
        raise
    except Exception:
        raise PreparedLayoutError("binding_rejected") from None


def _close(descriptors, descriptor):
    descriptors.remove(descriptor)
    try:
        os.close(descriptor)
    except BaseException:
        raise PreparedLayoutError("close_failed") from None


def _directory(fd, uid, mode):
    info = os.fstat(fd)
    _need(stat.S_ISDIR(info.st_mode) and info.st_uid == info.st_gid == uid
          and stat.S_IMODE(info.st_mode) == mode and info.st_ino > 0)
    return info


def _absent(parent_fd, name):
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise PreparedLayoutError("layout_rejected") from None


def _new_directory(parent_fd, name, uid, descriptors):
    os.mkdir(name, 0o700, dir_fd=parent_fd)
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                 dir_fd=parent_fd)
    descriptors.append(fd)
    _directory(fd, 0, 0o700)
    # This fd names only the new exclusive node created above.
    os.fchown(fd, uid, uid)
    os.fchmod(fd, 0o700)
    _directory(fd, uid, 0o700)
    return fd


def _new_input_file(parent_fd, name, raw, owned, descriptors):
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                 0o600, dir_fd=parent_fd)
    descriptors.append(fd)
    before = os.fstat(fd)
    _need(stat.S_ISREG(before.st_mode) and before.st_uid == before.st_gid == 0
          and before.st_nlink == 1 and before.st_size == 0)
    for offset in range(0, len(raw), 65536):
        owned._guard()
        part = raw[offset:offset + 65536]
        # A short write consumes this attempt; no repeated write/append is issued.
        _need(os.write(fd, part) == len(part))
    os.fchown(fd, 10001, 10001)
    os.fchmod(fd, 0o600)
    info = os.fstat(fd)
    _need(stat.S_ISREG(info.st_mode) and info.st_uid == info.st_gid == 10001
          and stat.S_IMODE(info.st_mode) == 0o600 and info.st_nlink == 1
          and info.st_size == len(raw)
          and (info.st_dev, info.st_ino) == (before.st_dev, before.st_ino))
    _close(descriptors, fd)


def _preflight_owned(owned, additions, directories):
    _need(type(owned) is rootfs.OwnedRootfs)
    owned._guard()
    _need(not owned._frozen and not owned._installed_files and not owned._installed_bytes)
    owned._verify_tree()
    _directory(owned._parent_fd, 0, 0o700)
    # The extracted root is still 0700 until the intended freeze normalization.
    _directory(owned._root_fd, 0, 0o700)
    import_rootfs.mount_root._path(str(owned._path))
    _need(owned._root_name not in SIBLINGS)
    for name in SIBLINGS:
        _absent(owned._parent_fd, name)
    forbidden = (tuple("opt/lumi-release/importer".split("/")),
                 tuple(CONTROL_PREFIX.rstrip("/").split("/")), ("inputs",), ("outputs",))
    _need(not any(parts[:len(prefix)] == prefix for prefix in forbidden for parts in owned._nodes))
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
    planned.update(TARGET_MODES)
    _need(len(planned) <= owned._limits["entries"] and total <= owned._limits["total"])
    external_nodes = (len(SIBLINGS) + sum(len(values) - 1 for values in directories.values())
                      + controls.MAXIMUM_INPUT_FILES)
    _need(external_nodes <= MAXIMUM_EXTERNAL_NODES)


def _mount_arguments(owned, inventories, descriptors):
    parent, root_path = str(owned._parent), str(owned._path)
    paths = {"parent": parent, "root": root_path, "inputs": root_path + "/inputs",
             "target_checkpoint": root_path + "/inputs/checkpoint",
             "target_tokenizer": root_path + "/inputs/tokenizer",
             "target_output": root_path + "/outputs",
             **{role: parent + "/" + role for role in ("checkpoint", "tokenizer", "output")}}
    identities = {}
    for role in input_mount.ROLES:
        fd = rootfs._open_absolute(paths[role], directory=True)
        descriptors.append(fd)
        _directory(fd, 10001 if role in {"checkpoint", "tokenizer", "output"} else 0,
                   0o755 if role in {"root", "inputs"} else 0o700)
        identities[role] = input_mount._identity(fd)
        _close(descriptors, fd)
    fd = rootfs._open_absolute(parent + "/mountpoint", directory=True)
    descriptors.append(fd)
    _directory(fd, 0, 0o700)
    mountpoint_identity = input_mount._identity(fd)
    _close(descriptors, fd)
    return import_rootfs.validate_arguments(rootfs_path=root_path,
        mountpoint_path=parent + "/mountpoint", checkpoint_path=paths["checkpoint"],
        tokenizer_path=paths["tokenizer"], output_path=paths["output"],
        expected_identities=identities, expected_mountpoint_identity=mountpoint_identity,
        inventories=inventories)


def install_prepared_layout(owned, **bundle_arguments):
    """Return private mount/environment controls and a separate aggregate receipt.

    No success grants authorization. Failure retains partial newly owned paths;
    neither this function nor retrying it can remove or resume those paths.
    """
    try:
        sources._runtime()
    except BaseException:
        raise PreparedLayoutError("runtime_rejected") from None
    try:
        bundle = validate_bundle(**bundle_arguments)
        _preflight_owned(owned, bundle["additions"], bundle["directories"])
    except PreparedLayoutError:
        raise
    except BaseException:
        raise PreparedLayoutError("layout_rejected") from None
    descriptors, started, failure = [], False, None
    try:
        started = True
        for name, body in sorted(bundle["additions"].items()):
            owned._guard()
            parts = tuple(name.split("/"))
            owned._write(parts, BytesIO(body), len(body), 0o444)
            owned._seen.add(parts)
            owned._installed_files += 1
            owned._installed_bytes += len(body)
        for parts, mode in TARGET_MODES.items():
            owned._parents(parts)
            identity, kind, _old_mode = owned._nodes[parts]
            owned._nodes[parts] = identity, kind, mode
            owned._seen.add(parts)
        for role in SIBLINGS:
            owned._guard()
            top = _new_directory(owned._parent_fd, role, 0 if role == "mountpoint" else 10001,
                                 descriptors)
            if role in bundle["generated_inputs"]:
                directories = {"": top}
                for relative in sorted(bundle["directories"][role], key=lambda value:
                                       (value.count("/"), value)):
                    if relative:
                        parent, _, name = relative.rpartition("/")
                        directories[relative] = _new_directory(directories[parent], name,
                                                               10001, descriptors)
                for relative, raw in sorted(bundle["generated_inputs"][role].items()):
                    parent, _, name = relative.rpartition("/")
                    _new_input_file(directories[parent], name, raw, owned, descriptors)
                for fd in reversed(list(directories.values())):
                    _close(descriptors, fd)
            else:
                _close(descriptors, top)
        owned.freeze()
        mount_arguments = _mount_arguments(owned, bundle["inventories"], descriptors)
        environment = bootstrap.validate_environment(dict(bootstrap.FIXED_ENV) | {
            "LUMI_IMPORT_SOURCE_MANIFEST_SHA256": bundle["summary"]["source_manifest_sha256"],
            "LUMI_PREPARED_FIXTURE_ENTRY_SHA256": bundle["summary"]["entry_sha256"],
            "LUMI_PREPARED_FIXTURE_CONTROL_SHA256": bundle["control_sha256"],
        })
    except BaseException as error:
        failure = error.code if type(error) is PreparedLayoutError else "installation_failed"
    finally:
        while descriptors:
            try:
                _close(descriptors, descriptors[-1])
            except BaseException:
                failure = "close_failed"
    if failure is not None:
        if started:
            owned._tainted = True
        raise PreparedLayoutError(failure) from None
    receipt = {"schema": SCHEMA, "status": "installed_private_prepared_fixture",
        "rootfs_addon_files": len(bundle["additions"]),
        "rootfs_addon_bytes": sum(map(len, bundle["additions"].values())),
        "generated_input_files": bundle["input_files"],
        "generated_input_bytes": bundle["input_bytes"], "private_content_write_bytes":
            bundle["write_bytes"], "source_manifest_sha256":
            bundle["summary"]["source_manifest_sha256"],
        "entry_sha256": bundle["summary"]["entry_sha256"], "helper_sha256": HELPER_SHA256,
        "control_sha256": bundle["control_sha256"],
        "generation_nonce_sha256": bundle["generation_nonce_sha256"],
        "source_payload_files_opened": 0, "new_input_payload_readback_bytes": 0,
        "new_input_bytes_checked_against_bound_inventory": bundle["input_bytes"],
        "installation_frozen": True, "source_mounts_installed": False,
        "external_sibling_cleanup_provided": False, **dict.fromkeys(FALSE_FLAGS, False)}
    return {"receipt": receipt, "mount_arguments": mount_arguments, "environment": environment}

