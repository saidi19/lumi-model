"""Compose frozen input views with an OCI root without discarding submounts.

The independently authenticated, prepared rootfs, empty mountpoint and three
worker-data roots must be distinct siblings under one root-owned 0700 parent.
The rootfs is root-owned 0755; its prepared mount targets are root-owned 0700.
Data ownership/modes remain the frozen input mount contract (10001, 0700/0600).
This adapter does not create those paths, read payloads or change their metadata.

Bind the software root first, then install input views in a second private mount
namespace. Re-pin the bound root in that final namespace before pivoting. There
is deliberately no later root bind: a nonrecursive bind would lose the inputs
and output. Existing OCI tmp/dev/proc overlays and readback are reused verbatim.

This is a one-shot trusted-parent component, not an authenticated host or real
import entrypoint. The parent still excludes writers/aliases, authenticates all
control inputs and software, provides resource limits, isolates networking,
closes ambient authority, drops privilege and separately admits a worker. Never
invoke native setup on a Mac; local tests replace every syscall with a double.
Every started failure consumes the attempt and requires the owned child to exit.
"""

from __future__ import annotations

import os
import sys
import threading

from . import posttraining_checkpoint_input_mount_v2 as input_mount
from . import posttraining_sft_mount_root_v1 as mount_root
from . import posttraining_sft_netns_v1 as netns
from . import posttraining_sft_oci_mount_v1 as oci

SCHEMA = "lumi.sft-checkpoint-import-rootfs.v0.1"
ERROR_CODES = input_mount.ERROR_CODES | oci.ERROR_CODES
_SETUP_LOCK = threading.Lock()
_ATTEMPT_OWNER = None


class ImportRootfsError(RuntimeError):
    def __init__(self, code="setup_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "setup_failed"
        super().__init__(self.code)


def _need(value, code):
    if not value:
        raise ImportRootfsError(code) from None


def _inputs(rootfs_path, mountpoint_path, checkpoint_path, tokenizer_path, output_path,
            expected_identities, expected_mountpoint_identity, inventories,
            scratch_bytes, shm_bytes):
    parent = oci._inputs(rootfs_path, mountpoint_path, input_mount.UID, input_mount.GID,
                         scratch_bytes, shm_bytes)
    paths, expected, manifests = input_mount._inputs(
        checkpoint_path, tokenizer_path, output_path, mountpoint_path,
        expected_identities, inventories,
    )
    _need(parent == paths["parent"] and rootfs_path not in paths.values(), "input_rejected")
    identity = expected_mountpoint_identity
    _need(type(identity) is tuple and len(identity) == 9
          and all(type(item) is int and item >= 0 for item in identity)
          and identity[1] > 0, "input_rejected")
    _need(identity[:2] not in {value[:2] for value in expected.values()}, "alias_rejected")
    return paths, expected, manifests


def _pin_bound(path, identity, uid, mode, owned):
    descriptor = oci._pin(path, owned)
    _need(input_mount._identity(descriptor) == identity, "identity_rejected")
    input_mount._directory(descriptor, uid, mode, "layout_rejected")
    return descriptor


def validate_arguments(*, rootfs_path, mountpoint_path, checkpoint_path, tokenizer_path,
                       output_path, expected_identities, expected_mountpoint_identity,
                       inventories, scratch_bytes=536870912, shm_bytes=67108864):
    """Pure pre-mutation validation and detached copies of caller-owned mappings.

    The trusted bootstrap may run this before consuming its own attempt. It
    validates control structure only: no identity, owner, content, runtime or
    host-authentication claim follows from the returned keyword arguments.
    """
    try:
        _paths, expected, manifests = _inputs(
            rootfs_path, mountpoint_path, checkpoint_path, tokenizer_path, output_path,
            expected_identities, expected_mountpoint_identity, inventories,
            scratch_bytes, shm_bytes,
        )
    except BaseException as error:
        code = (error.code if type(error) in (
            ImportRootfsError, input_mount.InputMountError, oci.OCIMountError,
            mount_root.MountRootError,
        ) and error.code in ERROR_CODES else "input_rejected")
        raise ImportRootfsError(code) from None
    return {
        "rootfs_path": rootfs_path, "mountpoint_path": mountpoint_path,
        "checkpoint_path": checkpoint_path, "tokenizer_path": tokenizer_path,
        "output_path": output_path, "expected_identities": expected,
        "expected_mountpoint_identity": expected_mountpoint_identity,
        "inventories": {role: value[0] for role, value in manifests.items()},
        "scratch_bytes": scratch_bytes, "shm_bytes": shm_bytes,
    }


def _data_readback(expected, manifests, owned):
    """After pivot, recheck final identities and every declared input node's flags."""
    descriptor = _pin_bound("/", expected["root"], 0, 0o755, owned)
    input_mount._close(owned, descriptor)
    descriptor = _pin_bound("/inputs", expected["inputs"], 0, 0o755, owned)
    mount_root._readback("/inputs", oci.READONLY_ROOT, mount_root.MS_NOEXEC)
    input_mount._close(owned, descriptor)
    seen = {expected[role][:2] for role in input_mount.ROLES
            if role not in {"checkpoint", "tokenizer"}}
    for role, path in (("checkpoint", "/inputs/checkpoint"),
                       ("tokenizer", "/inputs/tokenizer"), ("output", "/outputs")):
        descriptor = _pin_bound(path, expected[role], input_mount.UID, 0o700, owned)
        flags = input_mount.OUTPUT_FLAGS if role == "output" else input_mount.INPUT_FLAGS
        mount_root._readback(path, flags, mount_root.MS_RDONLY if role == "output" else 0)
        if role == "output":
            _need(input_mount._listing(descriptor, owned) == [], "layout_rejected")
        else:
            input_mount._tree(descriptor, manifests[role], owned, readonly=True, seen=seen)
        input_mount._close(owned, descriptor)


def enter_import_rootfs(*, rootfs_path, mountpoint_path, checkpoint_path, tokenizer_path,
                        output_path, expected_identities, expected_mountpoint_identity,
                        inventories, scratch_bytes=536870912, shm_bytes=67108864):
    """Consume one composition attempt and return only a partial mount receipt.

    ``expected_identities`` is the frozen input mount's nine-role native fstat
    mapping, with ``root`` and target identities taken from the prepared rootfs.
    ``expected_mountpoint_identity`` separately binds its fresh empty sibling.
    Neither native identities nor manifest names are returned in the receipt.
    """
    global _ATTEMPT_OWNER
    _need(_SETUP_LOCK.acquire(blocking=False), "reentry")
    owned, failure, mounted = [], None, None
    try:
        owner = os.getpid(), threading.get_ident()
        if _ATTEMPT_OWNER is not None:
            _need(_ATTEMPT_OWNER == owner, "owner_mismatch")
            raise ImportRootfsError("already_attempted") from None
        paths, expected, manifests = _inputs(
            rootfs_path, mountpoint_path, checkpoint_path, tokenizer_path, output_path,
            expected_identities, expected_mountpoint_identity, inventories,
            scratch_bytes, shm_bytes,
        )
        try:
            netns._runtime()
            _need(sys.version_info[:2] == (3, 12), "runtime_rejected")
        except BaseException:
            raise ImportRootfsError("runtime_rejected") from None
        _ATTEMPT_OWNER = owner
        preflight = (
            (paths["parent"], expected["parent"], 0o700),
            (rootfs_path, expected["root"], 0o755),
            (mountpoint_path, expected_mountpoint_identity, 0o700),
        )
        originals = [_pin_bound(path, identity, 0, mode, owned)
                     for path, identity, mode in preflight]
        oci._empty_mountpoint(originals[-1], owned)
        oci._layout(rootfs_path, owned)
        old_namespace = mount_root._open_namespace(owned)
        old_identity = mount_root._namespace_identity(old_namespace)
        mount_root._unshare()
        root_namespace = mount_root._open_namespace(owned)
        root_namespace_identity = mount_root._namespace_identity(root_namespace)
        _need(root_namespace_identity != old_identity, "namespace_unchanged")
        mount_root._mount(None, "/", None, mount_root.MS_REC | mount_root.MS_PRIVATE)
        fresh = [_pin_bound(path, identity, 0, mode, owned)
                 for path, identity, mode in preflight]
        oci._empty_mountpoint(fresh[-1], owned)
        for descriptor in originals:
            input_mount._close(owned, descriptor)
        mount_root._mount(f"/proc/self/fd/{fresh[1]}", f"/proc/self/fd/{fresh[2]}",
                          None, mount_root.MS_BIND, failure_code="bind_mount_failed")
        bound = _pin_bound(mountpoint_path, expected["root"], 0, 0o755, owned)
        mount_root._seal(mountpoint_path, oci.READONLY_ROOT, recursive=False)
        mount_root._readback(mountpoint_path, oci.READONLY_ROOT, mount_root.MS_NOEXEC)
        for descriptor in fresh + [bound]:
            input_mount._close(owned, descriptor)
        # This frozen helper unshares again, cloning the sealed root mount.
        # No root rebind is permitted after its input/output binds are installed.
        mounted = input_mount.mount_input_views(
            checkpoint_path=checkpoint_path, tokenizer_path=tokenizer_path,
            output_path=output_path, root_path=mountpoint_path,
            expected_identities=expected,
            inventories={role: value[0] for role, value in manifests.items()},
        )
        final_namespace = mount_root._open_namespace(owned)
        final_identity = mount_root._namespace_identity(final_namespace)
        _need(final_identity not in (old_identity, root_namespace_identity),
              "namespace_unchanged")
        # Fresh final-namespace FD: never fchdir using an older vfsmount reference.
        bound = _pin_bound(mountpoint_path, expected["root"], 0, 0o755, owned)
        oci._overlays(mountpoint_path, input_mount.UID, input_mount.GID,
                      scratch_bytes, shm_bytes)
        _need(mount_root._namespace_identity(old_namespace) == old_identity
              and mount_root._namespace_identity(root_namespace) == root_namespace_identity
              and mount_root._namespace_identity(final_namespace) == final_identity,
              "namespace_rejected")
        os.fchdir(bound)
        oci._pivot()
        oci._readback(expected["root"][:5], input_mount.UID, input_mount.GID,
                      scratch_bytes, shm_bytes, owned)
        _data_readback(expected, manifests, owned)
    except BaseException as error:
        failure = (error.code if type(error) in (
            ImportRootfsError, input_mount.InputMountError, oci.OCIMountError,
            mount_root.MountRootError,
        ) and error.code in ERROR_CODES else "setup_failed")
    finally:
        while owned:
            try:
                input_mount._close(owned, owned[-1])
            except BaseException:
                failure = "close_failed"
        _SETUP_LOCK.release()
    if failure is not None:
        raise ImportRootfsError(failure) from None
    return {
        "schema": SCHEMA, "readonly_software_root": True,
        "input_output_mounts_survive_pivot": True, "private_propagation": True,
        "old_root_detached": True, "private_tmpfs": True, "private_shm_tmpfs": True,
        "fresh_procfs": True, "scratch_capacity_bytes": scratch_bytes,
        "shm_capacity_bytes": shm_bytes, "input_mount_receipt": mounted,
        "source_metadata_mutated": False, "source_payload_bytes_read": 0,
        "host_software_binds": False, "host_mounts_modified": False,
        "rootfs_content_authenticated": False, "input_content_authenticated": False,
        "other_writable_aliases_excluded": False, "source_immutability_verified": False,
        "ambient_descriptors_closed": False, "resource_limits_verified": False,
        "complete_os_sandbox_verified": False, "worker_dispatched": False,
        "real_import_execution_permitted": False, "training_authorized": False,
        "production_eligible": False,
    }

