"""Bounded, owned extraction of one independently pinned Docker export tar.

This is not tarfile.extractall: every destination walk uses owned dirfds and
O_NOFOLLOW. Symlink contents are preserved for a subsequent pivot_root, never
followed here. Ownership is normalized to the extractor, set-id bits are
removed, and directory owner access is ensured for safe cleanup. Thus file
content is preserved, but byte-identical filesystem metadata is not claimed.

Use a trusted single-process caller and an existing private owned parent. The
cooperative absolute monotonic deadline cannot interrupt a blocked syscall; a
separate supervisor must impose a hard timeout. freeze() closes adapter writes,
not filesystem mutation by the owner. The caller must enforce mount isolation.
No Docker, native libraries, model inputs, subprocesses or network are used.
"""

from __future__ import annotations

import hashlib
import math
import os
import stat
import tarfile
import tempfile
import time
from collections import deque
from io import BytesIO
from pathlib import Path

SCHEMA = "lumi.sft-oci-rootfs.v0.1"
MAXIMUM_ARCHIVE_BYTES = 2 * 1024**3
MAXIMUM_TOTAL_BYTES = 4 * 1024**3
MAXIMUM_ENTRIES = 65536
_CHUNK = 65536
_INSTALL_PATHS = frozenset({
    "opt/lumi-release/guardian/worker_entry.py", "opt/lumi-release/guardian/admission.json",
})
ERROR_CODES = frozenset({
    "rootfs_configuration_rejected", "rootfs_deadline_exhausted", "rootfs_archive_rejected",
    "rootfs_archive_hash_rejected", "rootfs_path_rejected", "rootfs_entry_rejected",
    "rootfs_duplicate_rejected", "rootfs_bounds_exhausted", "rootfs_hardlink_rejected",
    "rootfs_identity_changed", "rootfs_installation_rejected", "rootfs_frozen",
    "rootfs_already_closed", "rootfs_owner_rejected", "rootfs_cleanup_failed",
    "rootfs_operation_failed",
})


class RootfsError(ValueError):
    """Only allowlisted aggregate codes, never paths or underlying error text."""

    def __init__(self, code="rootfs_operation_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "rootfs_operation_failed"
        super().__init__(self.code)


def _need(condition, code):
    if not condition:
        raise RootfsError(code) from None


def _pin(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _identity(value):
    return value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode), value.st_uid, value.st_gid


def _archive_identity(value):
    return (*_identity(value), value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _check_deadline(deadline):
    _need(time.monotonic() < deadline, "rootfs_deadline_exhausted")


def _parts(name, *, directory=False):
    # tarfile normalizes a directory header named '/' to the empty name.
    if directory and type(name) is str and name in {"", ".", "./", "/"}:
        return ()
    _need(type(name) is str and bool(name) and "\x00" not in name and "\\" not in name,
          "rootfs_path_rejected")
    _need(len(name.encode("utf-8")) <= 4096, "rootfs_path_rejected")
    while name.startswith("./"):
        name = name[2:]
    if directory:
        name = name.rstrip("/")
    result = tuple(name.split("/"))
    _need(len(result) <= 64 and all(p not in {"", ".", ".."}
                                     and len(p.encode("utf-8")) <= 255 for p in result),
          "rootfs_path_rejected")
    return result


def _absolute(value):
    value = os.fspath(value)
    _need(type(value) is str and value.startswith("/") and value != "/",
          "rootfs_configuration_rejected")
    parts = _parts(value[1:])
    return value, parts


def _open_absolute(value, *, directory):
    """No symlink ancestors, including the archive's parent directories."""
    _, parts = _absolute(value)
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for index, part in enumerate(parts):
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if index < len(parts) - 1 or directory:
                flags |= os.O_DIRECTORY
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        result, descriptor = descriptor, -1
        return result
    finally:
        if descriptor >= 0:
            os.close(descriptor)


class _BoundedTarInfo(tarfile.TarInfo):
    """Bound parser metadata allocations before tarfile reads their bodies."""

    @classmethod
    def fromtarfile(cls, archive):
        return cls._fromtarfile(archive)

    @classmethod
    def _fromtarfile(cls, archive, *, dircheck=True):
        # CPython 3.12.14+ routes PAX/GNU follow-up headers through this
        # private bridge, bypassing public fromtarfile(). Both entry points
        # must share the physical-header budget, including nested metadata.
        _check_deadline(archive._rootfs_deadline)
        raw = archive.fileobj.read(tarfile.BLOCKSIZE)
        _need(len(raw) == tarfile.BLOCKSIZE, "rootfs_archive_rejected")
        if raw == b"\x00" * tarfile.BLOCKSIZE:
            raise tarfile.EOFHeaderError("end of archive")
        archive._rootfs_headers += 1
        _need(archive._rootfs_headers <= archive._rootfs_maximum_headers,
              "rootfs_bounds_exhausted")
        failure, member = False, None
        try:
            parser = getattr(cls, "_frombuf", None)
            if parser is None:
                member = cls.frombuf(raw, archive.encoding, archive.errors)
            else:
                # Preserve the updated stdlib's security-sensitive V7
                # directory distinction for metadata follow-up headers.
                member = parser(raw, archive.encoding, archive.errors, dircheck=dircheck)
        except tarfile.HeaderError:
            failure = True
        _need(not failure, "rootfs_archive_rejected")
        member.offset = archive.fileobj.tell() - tarfile.BLOCKSIZE
        return member._proc_member(archive)

    def _metadata(self, archive, maximum):
        _need(0 <= self.size <= maximum, "rootfs_bounds_exhausted")
        archive._rootfs_metadata_bytes += self.size
        _need(archive._rootfs_metadata_bytes <= 64 * 1024**2, "rootfs_bounds_exhausted")

    def _proc_pax(self, archive):
        self._metadata(archive, _CHUNK)
        result = super()._proc_pax(archive)
        _need(not any("sparse" in key.lower() for key in result.pax_headers),
              "rootfs_entry_rejected")
        return result

    def _proc_gnulong(self, archive):
        self._metadata(archive, 8192)
        return super()._proc_gnulong(archive)

    def _proc_sparse(self, archive):
        raise RootfsError("rootfs_entry_rejected") from None


    def _proc_gnusparse_00(self, *args):
        raise RootfsError("rootfs_entry_rejected") from None

    def _proc_gnusparse_01(self, *args):
        raise RootfsError("rootfs_entry_rejected") from None

    def _proc_gnusparse_10(self, *args):
        raise RootfsError("rootfs_entry_rejected") from None


class _BoundedTarFile(tarfile.TarFile):
    def __init__(self, *args, rootfs_deadline, rootfs_maximum_headers, **kwargs):
        self._rootfs_deadline = rootfs_deadline
        self._rootfs_maximum_headers = rootfs_maximum_headers
        self._rootfs_headers = self._rootfs_metadata_bytes = 0
        super().__init__(*args, **kwargs)


class OwnedRootfs:
    """Fresh owned tree with exact identity-tracked, one-shot cleanup.

    Construct through extract_rootfs only. A failure during close never follows
    replacements or removes untracked content. Cleanup is bounded by admitted
    entries, and is still attempted when the extraction deadline has expired.
    """

    def __init__(self, parent, parent_fd, root_name, root_fd, *, limits, deadline):
        self._parent = parent
        self._parent_fd, self._root_fd = parent_fd, root_fd
        self._root_name = root_name
        self._path = Path(parent) / root_name
        self._owner_pid = os.getpid()
        self._nodes = {(): (_identity(os.fstat(root_fd)), "directory", 0o755)}
        self._seen = set()
        self._closed = self._frozen = False
        self._tainted = False
        self._cleanup_completed = False
        self._limits, self._deadline = limits, deadline
        self._archive_bytes = self._archive_entries = self._content_bytes = 0
        self._archive_headers = self._metadata_bytes = 0
        self._installed_files = self._installed_bytes = self._hardlinks = self._symlinks = 0

    @property
    def path(self):
        self._guard()
        return self._path

    def summary(self):
        return {
            "schema": SCHEMA, "archive_bytes": self._archive_bytes,
            "archive_entries": self._archive_entries, "owned_entries": len(self._nodes),
            "archive_headers": self._archive_headers, "metadata_bytes": self._metadata_bytes,
            "content_bytes": self._content_bytes, "hardlinks": self._hardlinks,
            "symlinks": self._symlinks, "installed_files": self._installed_files,
            "installed_bytes": self._installed_bytes, "installation_frozen": self._frozen,
            "cleanup_attempted": self._closed, "cleanup_completed": self._cleanup_completed,
            "operation_tainted": self._tainted,
            "metadata_normalized": True,
            "filesystem_immutable": False, "production_eligible": False,
            "real_sft_execution_permitted": False,
        }

    def _guard(self, *, clock=True):
        _need(not self._closed, "rootfs_already_closed")
        _need(not self._tainted, "rootfs_operation_failed")
        _need(os.getpid() == self._owner_pid, "rootfs_owner_rejected")
        if clock:
            _check_deadline(self._deadline)
        self._assert_root()

    def _assert_root(self):
        _need(_identity(os.stat(self._root_name, dir_fd=self._parent_fd,
                                follow_symlinks=False)) == self._nodes[()][0],
              "rootfs_identity_changed")
        parent_check = _open_absolute(self._parent, directory=True)
        try:
            _need(_identity(os.fstat(parent_check)) == _identity(os.fstat(self._parent_fd)),
                  "rootfs_identity_changed")
        finally:
            os.close(parent_check)

    def _directory(self, parts):
        descriptor = os.dup(self._root_fd)
        try:
            for count, name in enumerate(parts, 1):
                key = parts[:count]
                _need(key in self._nodes and self._nodes[key][1] == "directory",
                      "rootfs_path_rejected")
                next_descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                          dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
                _need(_identity(os.fstat(descriptor)) == self._nodes[key][0],
                      "rootfs_identity_changed")
            result, descriptor = descriptor, -1
            return result
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _capacity(self):
        _need(len(self._nodes) < self._limits["entries"], "rootfs_bounds_exhausted")

    def _parents(self, parts):
        for count in range(1, len(parts) + 1):
            key = parts[:count]
            if key in self._nodes:
                _need(self._nodes[key][1] == "directory", "rootfs_path_rejected")
                continue
            self._capacity()
            descriptor = self._directory(key[:-1])
            try:
                os.mkdir(key[-1], 0o700, dir_fd=descriptor)
                value = os.stat(key[-1], dir_fd=descriptor, follow_symlinks=False)
                self._nodes[key] = (_identity(value), "directory", 0o755)
            finally:
                os.close(descriptor)

    def _write(self, parts, stream, size, mode):
        _need(size <= self._limits["file"] and self._content_bytes + size
              <= self._limits["total"], "rootfs_bounds_exhausted")
        self._parents(parts[:-1])
        self._capacity()
        descriptor, output = self._directory(parts[:-1]), -1
        try:
            output = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=descriptor)
            self._nodes[parts] = (_identity(os.fstat(output)), "file", mode & 0o777)
            remaining = size
            while remaining:
                _check_deadline(self._deadline)
                data = stream.read(min(remaining, _CHUNK))
                _need(type(data) is bytes and 0 < len(data) <= remaining,
                      "rootfs_archive_rejected")
                remaining -= len(data)
                view = memoryview(data)
                while view:
                    written = os.write(output, view)
                    _need(written > 0, "rootfs_operation_failed")
                    self._content_bytes += written
                    view = view[written:]
            os.fchmod(output, mode & 0o777)
        finally:
            if output >= 0:
                os.close(output)
            os.close(descriptor)

    def _member(self, archive, member, deferred):
        _check_deadline(self._deadline)
        self._archive_entries += 1
        _need(self._archive_entries <= self._limits["entries"], "rootfs_bounds_exhausted")
        _need(member.type in {tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE,
                              tarfile.SYMTYPE, tarfile.LNKTYPE}
              and member.sparse is None and not any("sparse" in k.lower()
                                                    for k in member.pax_headers),
              "rootfs_entry_rejected")
        _need(type(member.size) is int and member.size >= 0
              and (member.isreg() or member.size == 0), "rootfs_entry_rejected")
        parts = _parts(member.name, directory=member.isdir())
        _need(parts not in self._seen, "rootfs_duplicate_rejected")
        self._seen.add(parts)
        if member.isdir():
            self._parents(parts)
            identity, kind, _ = self._nodes[parts]
            self._nodes[parts] = (identity, kind, (member.mode & 0o1777) | 0o700)
            return
        _need(parts not in self._nodes, "rootfs_duplicate_rejected")
        if member.isreg():
            stream = archive.extractfile(member)
            _need(stream is not None, "rootfs_archive_rejected")
            with stream:
                self._write(parts, stream, member.size, member.mode)
            return
        self._parents(parts[:-1])
        if member.islnk():
            deferred.append((parts, _parts(member.linkname)))
            return
        self._capacity()
        _need(type(member.linkname) is str and bool(member.linkname)
              and "\x00" not in member.linkname
              and len(member.linkname.encode("utf-8")) <= 4096, "rootfs_entry_rejected")
        descriptor = self._directory(parts[:-1])
        try:
            os.symlink(member.linkname, parts[-1], dir_fd=descriptor)
            value = os.stat(parts[-1], dir_fd=descriptor, follow_symlinks=False)
            self._nodes[parts] = (_identity(value), "symlink", 0)
            self._symlinks += 1
        finally:
            os.close(descriptor)

    def _link(self, parts, target):
        _need(target in self._nodes and self._nodes[target][1] == "file",
              "rootfs_hardlink_rejected")
        self._capacity()
        source_fd, target_fd = self._directory(target[:-1]), -1
        try:
            target_fd = self._directory(parts[:-1])
            value = os.stat(target[-1], dir_fd=source_fd, follow_symlinks=False)
            _need(_identity(value) == self._nodes[target][0] and stat.S_ISREG(value.st_mode),
                  "rootfs_identity_changed")
            os.link(target[-1], parts[-1], src_dir_fd=source_fd, dst_dir_fd=target_fd,
                    follow_symlinks=False)
            value = os.stat(parts[-1], dir_fd=target_fd, follow_symlinks=False)
            self._nodes[parts] = (_identity(value), "file", value.st_mode & 0o777)
            _need(_identity(value) == self._nodes[target][0], "rootfs_identity_changed")
            self._hardlinks += 1
        finally:
            os.close(source_fd)
            if target_fd >= 0:
                os.close(target_fd)

    def install_file(self, relative_path, data, *, expected_sha256, mode=0o644):
        """Install either of two new, pinned guardian controls; never overwrite."""
        failure, write_started = None, False
        try:
            self._guard()
            _need(not self._frozen, "rootfs_frozen")
            _need(type(relative_path) is str and relative_path in _INSTALL_PATHS
                  and type(data) is bytes and len(data) <= 1024**2
                  and type(mode) is int and mode in {0o444, 0o644}
                  and _pin(expected_sha256)
                  and hashlib.sha256(data).hexdigest() == expected_sha256,
                  "rootfs_installation_rejected")
            parts = _parts(relative_path)
            _need(parts not in self._seen and parts not in self._nodes,
                  "rootfs_duplicate_rejected")
            write_started = True
            self._write(parts, BytesIO(data), len(data), mode)
            self._seen.add(parts)
            self._installed_files += 1
            self._installed_bytes += len(data)
        except RootfsError as error:
            failure = error.code
        except BaseException:
            failure = "rootfs_operation_failed"
        if failure is not None:
            if write_started:
                self._tainted = True
            raise RootfsError(failure) from None

    def freeze(self):
        """Make directories traversable as admitted, and close adapter writes."""
        failure = None
        try:
            self._guard()
            _need(not self._frozen, "rootfs_frozen")
            self._verify_tree()
            for parts, (_, kind, mode) in self._nodes.items():
                _check_deadline(self._deadline)
                if kind == "directory":
                    descriptor = self._directory(parts)
                    try:
                        os.fchmod(descriptor, 0o755 if not parts else mode)
                    finally:
                        os.close(descriptor)
            self._frozen = True
        except RootfsError as error:
            failure = error.code
        except BaseException:
            failure = "rootfs_operation_failed"
        if failure is not None:
            raise RootfsError(failure) from None

    def _verify_tree(self, *, clock=True):
        children = {}
        for parts in self._nodes:
            if parts:
                children.setdefault(parts[:-1], set()).add(parts[-1])
        for parts, (_, kind, _) in self._nodes.items():
            if clock:
                _check_deadline(self._deadline)
            if kind != "directory":
                continue
            descriptor = self._directory(parts)
            try:
                found = set()
                with os.scandir(descriptor) as entries:
                    for entry in entries:
                        if clock:
                            _check_deadline(self._deadline)
                        _need(entry.name in children.get(parts, set()), "rootfs_identity_changed")
                        key = (*parts, entry.name)
                        _need(_identity(entry.stat(follow_symlinks=False)) == self._nodes[key][0],
                              "rootfs_identity_changed")
                        found.add(entry.name)
                _need(found == children.get(parts, set()), "rootfs_identity_changed")
            finally:
                os.close(descriptor)

    def close(self):
        """Attempt cleanup once, refusing all foreign or identity-changed entries."""
        _need(not self._closed, "rootfs_already_closed")
        self._closed = True
        failure = None
        try:
            _need(os.getpid() == self._owner_pid, "rootfs_owner_rejected")
            self._assert_root()
            self._verify_tree(clock=False)
            for parts in sorted(self._nodes, key=len, reverse=True):
                if not parts:
                    continue
                descriptor = self._directory(parts[:-1])
                try:
                    _need(_identity(os.stat(parts[-1], dir_fd=descriptor, follow_symlinks=False))
                          == self._nodes[parts][0], "rootfs_identity_changed")
                    if self._nodes[parts][1] == "directory":
                        os.rmdir(parts[-1], dir_fd=descriptor)
                    else:
                        os.unlink(parts[-1], dir_fd=descriptor)
                finally:
                    os.close(descriptor)
            self._assert_root()
            os.rmdir(self._root_name, dir_fd=self._parent_fd)
            self._cleanup_completed = True
        except RootfsError as error:
            failure = error.code
        except BaseException:
            failure = "rootfs_cleanup_failed"
        finally:
            os.close(self._root_fd)
            os.close(self._parent_fd)
        if failure is not None:
            raise RootfsError(failure) from None


def extract_rootfs(archive_path, parent_directory, *, expected_sha256, maximum_archive_bytes,
                   maximum_file_bytes, maximum_total_bytes, maximum_entries, deadline):
    """Hash then stream one uncompressed archive into a private owned directory."""
    result, failure, archive_fd, parent_fd, root_fd = None, None, -1, -1, -1
    try:
        _need(_pin(expected_sha256) and type(deadline) in {int, float}
              and math.isfinite(deadline) and deadline > 0, "rootfs_configuration_rejected")
        for value, ceiling in ((maximum_archive_bytes, MAXIMUM_ARCHIVE_BYTES),
                               (maximum_file_bytes, MAXIMUM_TOTAL_BYTES),
                               (maximum_total_bytes, MAXIMUM_TOTAL_BYTES),
                               (maximum_entries, MAXIMUM_ENTRIES)):
            _need(type(value) is int and 0 < value <= ceiling, "rootfs_configuration_rejected")
        _need(maximum_file_bytes <= maximum_total_bytes, "rootfs_configuration_rejected")
        _check_deadline(deadline)
        parent, _ = _absolute(parent_directory)
        parent_fd = _open_absolute(parent, directory=True)
        parent_stat = os.fstat(parent_fd)
        _need(parent_stat.st_uid == os.geteuid() and parent_stat.st_gid == os.getegid()
              and parent_stat.st_mode & 0o077 == 0,
              "rootfs_configuration_rejected")
        archive_fd = _open_absolute(archive_path, directory=False)
        archive_stat = os.fstat(archive_fd)
        _need(stat.S_ISREG(archive_stat.st_mode)
              and 1024 <= archive_stat.st_size <= maximum_archive_bytes,
              "rootfs_archive_rejected")
        digest, count = hashlib.sha256(), 0
        while count < archive_stat.st_size:
            _check_deadline(deadline)
            data = os.read(archive_fd, min(_CHUNK, archive_stat.st_size - count))
            _need(bool(data), "rootfs_archive_rejected")
            count += len(data)
            digest.update(data)
        _need(not os.read(archive_fd, 1)
              and _archive_identity(os.fstat(archive_fd)) == _archive_identity(archive_stat),
              "rootfs_archive_rejected")
        _need(digest.hexdigest() == expected_sha256, "rootfs_archive_hash_rejected")
        os.lseek(archive_fd, 0, os.SEEK_SET)
        root_path = tempfile.mkdtemp(prefix="lumi-owned-rootfs-", dir=parent)
        root_name = Path(root_path).name
        root_fd = os.open(root_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                          dir_fd=parent_fd)
        os.fchmod(root_fd, 0o700)
        result = OwnedRootfs(parent, parent_fd, root_name, root_fd,
                            limits={"file": maximum_file_bytes, "total": maximum_total_bytes,
                                    "entries": maximum_entries}, deadline=deadline)
        parent_fd = root_fd = -1
        result._guard()
        result._archive_bytes = count
        deferred = []
        with os.fdopen(os.dup(archive_fd), "rb", buffering=0) as stream:
            with _BoundedTarFile.open(fileobj=stream, mode="r|", tarinfo=_BoundedTarInfo,
                                     rootfs_deadline=deadline,
                                     rootfs_maximum_headers=maximum_entries) as archive:
                while True:
                    member = archive.next()
                    if member is None:
                        break
                    result._member(archive, member, deferred)
                    archive.members.clear()  # Do not retain lexical metadata for prior members.
                tail_bytes = 0
                while True:
                    _check_deadline(deadline)
                    tail = archive.fileobj.read(_CHUNK)
                    if not tail:
                        break
                    tail_bytes += len(tail)
                    _need(not any(tail), "rootfs_archive_rejected")
                _need(tail_bytes >= tarfile.BLOCKSIZE and tail_bytes % tarfile.BLOCKSIZE == 0,
                      "rootfs_archive_rejected")
                result._archive_headers = archive._rootfs_headers
                result._metadata_bytes = archive._rootfs_metadata_bytes
        _need(_archive_identity(os.fstat(archive_fd)) == _archive_identity(archive_stat),
              "rootfs_archive_rejected")
        waiting, ready = {}, deque()
        for parts, target in deferred:
            if target in result._nodes:
                ready.append((parts, target))
            else:
                waiting.setdefault(target, []).append((parts, target))
        linked = 0
        while ready:
            _check_deadline(deadline)
            parts, target = ready.popleft()
            result._link(parts, target)
            linked += 1
            ready.extend(waiting.pop(parts, []))
        _need(linked == len(deferred), "rootfs_hardlink_rejected")
        result._guard()
    except RootfsError as error:
        failure = error.code
    except BaseException:
        failure = "rootfs_operation_failed"
    finally:
        for descriptor in (archive_fd, parent_fd, root_fd):
            if descriptor >= 0:
                os.close(descriptor)
    if failure is not None:
        if result is not None:
            try:
                result.close()
            except BaseException:
                failure = "rootfs_cleanup_failed"
        raise RootfsError(failure) from None
    return result
