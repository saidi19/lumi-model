"""Read-only, bounded in-memory input view for unchanged safety validators.

The caller authenticates and loads every blob before constructing this view.
This module does not authenticate inputs, consult the filesystem, load private
files, or grant execution permission. Its Path replacement intentionally cannot
escape into real filesystem APIs through ``os.fspath``. All opens predebit the
complete logical input before returning a read-only memory stream.
"""

from __future__ import annotations

import io
import threading
import zlib
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType

MAXIMUM_VIRTUAL_OPENS = 256
MAXIMUM_LOGICAL_READ_BYTES = 256 * 1024 * 1024
MAXIMUM_GZIP_BYTES = 32 * 1024 * 1024
DEFAULT_REPOSITORY_ROOT = str(PurePosixPath(__file__).parents[2])


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code) from None


def _path_text(value: object) -> str:
    _require(type(value) is str, "snapshot_path_rejected")
    _require(
        len(value) <= 4096
        and not value.startswith("//")
        and all(ord(char) >= 32 and ord(char) != 127 for char in value)
        and not any(part.startswith("~") for part in value.split("/")),
        "snapshot_path_rejected",
    )
    return value


def _absolute(value: str, root: str) -> str:
    """Normalize POSIX components lexically; never resolve a real symlink."""
    raw = value if value.startswith("/") else root + "/" + value
    parts = []
    for part in raw.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if parts:
                parts.pop()
        else:
            parts.append(part)
    return "/" + "/".join(parts)


def _decode(raw: bytes, encoding: str | None, errors: str | None) -> str:
    _require(encoding is None or type(encoding) is str, "snapshot_encoding_rejected")
    _require(errors is None or errors == "strict", "snapshot_encoding_rejected")
    # Avoid arbitrary codec discovery/imports or registered search callbacks.
    name = (encoding or "utf-8").lower().replace("_", "-")
    aliases = {"utf8": "utf-8", "utf-8": "utf-8", "ascii": "ascii",
               "latin1": "latin-1", "latin-1": "latin-1", "utf-8-sig": "utf-8"}
    _require(name in aliases, "snapshot_encoding_rejected")
    if name == "utf-8-sig" and raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    failed = False
    try:
        value = raw.decode(aliases[name], errors="strict")
    except (LookupError, UnicodeError, ValueError):
        failed = True
    _require(not failed, "snapshot_encoding_rejected")
    return value


class _ReadOnlyBytes(io.BytesIO):
    def writable(self) -> bool:
        return False

    def write(self, *_args, **_kwargs):
        raise ValueError("snapshot_write_rejected") from None

    writelines = write
    truncate = write

    def getbuffer(self):
        return memoryview(self.getvalue())


class _ReadOnlyText(io.StringIO):
    def writable(self) -> bool:
        return False

    def write(self, *_args, **_kwargs):
        raise ValueError("snapshot_write_rejected") from None

    writelines = write
    truncate = write


@dataclass(frozen=True, slots=True, repr=False)
class _SnapshotPath:
    _view: Snapshot
    _path: PurePosixPath

    def __str__(self) -> str:
        return str(self._path)

    def __repr__(self) -> str:
        return "<SnapshotPath>"

    def __fspath__(self):
        raise ValueError("snapshot_filesystem_escape_rejected") from None

    def __truediv__(self, value):
        return self._view.Path(self, value)

    def __rtruediv__(self, value):
        return self._view.Path(value, self)

    @property
    def parent(self):
        return self._view.Path(str(self._path.parent))

    @property
    def name(self) -> str:
        return self._path.name

    def is_absolute(self) -> bool:
        return self._path.is_absolute()

    def resolve(self, strict: bool = False):
        _require(type(strict) is bool, "snapshot_path_rejected")
        value = self._view.Path(_absolute(str(self), self._view._root))
        _require(not strict or value.exists(), "snapshot_lookup_rejected")
        return value

    def expanduser(self):
        return self

    def is_relative_to(self, other) -> bool:
        parent = self._view.Path(other).resolve()._path
        return self.resolve()._path.is_relative_to(parent)

    def is_file(self) -> bool:
        return str(self.resolve()) in self._view._blobs

    def is_dir(self) -> bool:
        return str(self.resolve()) in self._view._directories

    def exists(self) -> bool:
        return self.is_file() or self.is_dir()

    def read_bytes(self) -> bytes:
        with self.open("rb") as stream:
            return stream.read()

    def read_text(self, encoding=None, errors=None, newline=None) -> str:
        with self.open("r", encoding=encoding, errors=errors, newline=newline) as stream:
            return stream.read()

    def open(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        _require(type(buffering) is int and buffering == -1, "snapshot_mode_rejected")
        return self._view._open(self, mode, encoding, errors, newline, compressed=False)


class _PathFactory:
    def __init__(self, view: Snapshot):
        self._view = view

    def __call__(self, *parts):
        values = []
        for part in parts or (".",):
            if type(part) is _SnapshotPath:
                _require(part._view is self._view, "snapshot_foreign_view_rejected")
                values.append(str(part))
            else:
                values.append(_path_text(part))
        return _SnapshotPath(self._view, PurePosixPath(*values))

    def cwd(self):
        return self(self._view._root)


def _inflate(raw: bytes, maximum: int) -> bytes:
    """Decode complete gzip members with CRC checks and bounded output allocation."""
    _require(bool(raw), "snapshot_gzip_rejected")
    parts = []
    total = 0
    remaining = raw
    while remaining:
        failed = False
        try:
            decoder = zlib.decompressobj(31)
            decoded = decoder.decompress(remaining, maximum - total + 1)
        except zlib.error:
            failed = True
        _require(not failed, "snapshot_gzip_rejected")
        total += len(decoded)
        _require(total <= maximum, "snapshot_gzip_limit_exceeded")
        _require(decoder.eof, "snapshot_gzip_rejected")
        parts.append(decoded)
        remaining = decoder.unused_data
    return b"".join(parts)


class Snapshot:
    """Immutable bound blobs, lexical Path factory and memory-only gzip adapter.

    ``Path`` is callable and has ``cwd()``. ``gzip_open`` replaces gzip.open for
    reads only. ``summary()`` exposes aggregate counters without paths or data.
    Lower test-specific limits are allowed, but the public hard caps cannot grow.
    Missing lookups supply no stream. Bound open attempts consume an open even
    when decoding fails; successful byte reservations are never refunded when
    streams are closed or only partially consumed. Text codecs are a fixed
    in-memory allowlist, never arbitrary codec discovery/import callbacks.
    """

    def __init__(
        self,
        blobs: dict[str, bytes],
        *,
        repository_root: str = DEFAULT_REPOSITORY_ROOT,
        maximum_virtual_opens: int = MAXIMUM_VIRTUAL_OPENS,
        maximum_logical_read_bytes: int = MAXIMUM_LOGICAL_READ_BYTES,
        maximum_gzip_bytes: int = MAXIMUM_GZIP_BYTES,
    ):
        root = _path_text(repository_root)
        _require(root.startswith("/") and _absolute(root, "/") == root,
                 "snapshot_configuration_rejected")
        _require(type(blobs) is dict, "snapshot_configuration_rejected")
        copied = {}
        directories = set()
        for path, raw in blobs.items():
            path = _path_text(path)
            _require(
                path.startswith("/") and _absolute(path, "/") == path and path != "/"
                and type(raw) is bytes,
                "snapshot_configuration_rejected",
            )
            copied[path] = raw
            directories.update(str(parent) for parent in PurePosixPath(path).parents)
        _require(not directories.intersection(copied), "snapshot_configuration_rejected")
        for value, maximum in (
            (maximum_virtual_opens, MAXIMUM_VIRTUAL_OPENS),
            (maximum_logical_read_bytes, MAXIMUM_LOGICAL_READ_BYTES),
            (maximum_gzip_bytes, MAXIMUM_GZIP_BYTES),
        ):
            _require(type(value) is int and 1 <= value <= maximum,
                     "snapshot_configuration_rejected")
        self._root = root
        self._blobs = MappingProxyType(copied)
        self._directories = frozenset(directories)
        self._maximum_opens = maximum_virtual_opens
        self._maximum_bytes = maximum_logical_read_bytes
        self._maximum_gzip = maximum_gzip_bytes
        self._opens = 0
        self._bytes = 0
        self._lock = threading.Lock()
        self.Path = _PathFactory(self)

    def _open(self, path, mode, encoding, errors, newline, *, compressed):
        _require(type(mode) is str and mode in {"r", "rt", "rb"}, "snapshot_mode_rejected")
        binary = mode == "rb" or (compressed and mode == "r")
        _require(not binary or all(value is None for value in (encoding, errors, newline)),
                 "snapshot_mode_rejected")
        _require(newline is None or (type(newline) is str and newline in {"", "\n", "\r", "\r\n"}),
                 "snapshot_mode_rejected")
        key = str(self.Path(path).resolve())
        _require(key in self._blobs, "snapshot_lookup_rejected")
        with self._lock:
            _require(self._opens < self._maximum_opens, "snapshot_open_limit_exceeded")
            self._opens += 1
            raw = self._blobs[key]
            if compressed:
                raw = _inflate(raw, self._maximum_gzip)
            _require(len(raw) <= self._maximum_bytes - self._bytes,
                     "snapshot_read_limit_exceeded")
            self._bytes += len(raw)
        if binary:
            return _ReadOnlyBytes(raw)
        text = _decode(raw, encoding, errors)
        return _ReadOnlyText(text, newline=newline)

    def gzip_open(self, path, mode="rb", compresslevel=9, encoding=None, errors=None, newline=None):
        _require(type(compresslevel) is int and compresslevel == 9, "snapshot_mode_rejected")
        return self._open(path, mode, encoding, errors, newline, compressed=True)

    def summary(self) -> dict:
        with self._lock:
            return {
                "bound_blob_count": len(self._blobs),
                "virtual_opens": self._opens,
                "logical_read_bytes_debited": self._bytes,
                "maximum_virtual_opens": self._maximum_opens,
                "maximum_logical_read_bytes": self._maximum_bytes,
                "maximum_gzip_bytes": self._maximum_gzip,
                "filesystem_access_performed": False,
                "execution_permitted": False,
            }
