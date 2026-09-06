"""Read exactly the two pinned local safety candidates; emit aggregates only.

Use Python -S -B with --audit-existing-local-pair. No historical loader, runner,
tokenizer, registry, benchmark, owner receipt or publisher is imported or run.
The two candidate files remain open for identity checks through comparison.
This code is read-only, not an OS sandbox or a full integration replay.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import re
import signal
import stat
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAXIMUM_FILE_BYTES = 671_624
MAXIMUM_TOTAL_BYTES = 1_306_432
MAXIMUM_SECONDS = 30
BINDINGS = (
    (
        "../lumi-artifacts/4b-sft-candidate-hygiene-repair-v0.1/4892b05c49dc/"
        "attempt-001/safety-tool-candidate.jsonl",
        634_808,
        "ffc4a91613dd228282d2bc8ef2b93a6d508e31fa90e2925d91de837c1b4bd0e1",
    ),
    (
        "../lumi-artifacts/4b-sft-safety-uri-repair-v0.1/8cd8cc0b20cc/"
        "attempt-001/safety-tool-candidate.jsonl",
        671_624,
        "0f327fb0959a512a698c49992ad516467b3ef8058eb1653416df45157d7f070e",
    ),
)
CODES = frozenset(
    {
        "runtime_rejected",
        "arguments_rejected",
        "binding_rejected",
        "file_rejected",
        "file_changed",
        "hash_mismatch",
        "read_budget",
        "row_delta_rejected",
        "deadline_exceeded",
        "audit_failed",
    }
)


class Stop(BaseException):
    def __init__(self, code):
        self.code = code if type(code) is str and code in CODES else "audit_failed"
        super().__init__(self.code)


def require(condition, code):
    if not condition:
        raise Stop(code) from None


def identity(info):
    return tuple(
        getattr(info, name)
        for name in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    )


@contextlib.contextmanager
def bound_file(path: Path, size: int, digest: str, accounting: dict):
    """No-follow traversal; exact-size read and hash; retain descriptor until exit."""
    require(type(size) is int and 0 < size <= MAXIMUM_FILE_BYTES, "binding_rejected")
    require(type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest), "binding_rejected")
    require(
        path.is_absolute()
        and path.suffix == ".jsonl"
        and ".." not in path.parts
        and len(path.parts) <= 64,
        "binding_rejected",
    )
    handles, directories = [], []
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        parent = os.open("/", flags)
        handles.append(parent)
        for part in path.parts[1:-1]:
            child = os.open(part, flags, dir_fd=parent)
            handles.append(child)
            info = os.fstat(child)
            directories.append((parent, part, child, (info.st_dev, info.st_ino, info.st_mode)))
            parent = child
        require(accounting["candidate_files_opened"] < 2, "read_budget")
        descriptor = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=parent
        )
        handles.append(descriptor)
        accounting["candidate_files_opened"] += 1
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size == size,
            "file_rejected",
        )

        def unchanged():
            for directory, name, child, expected in directories:
                for info in (
                    os.fstat(child),
                    os.stat(name, dir_fd=directory, follow_symlinks=False),
                ):
                    require((info.st_dev, info.st_ino, info.st_mode) == expected, "file_changed")
            after = os.fstat(descriptor)
            named = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            require(identity(before) == identity(after) == identity(named), "file_changed")

        chunks, remaining = [], size
        while remaining:
            budget = MAXIMUM_TOTAL_BYTES - accounting["candidate_bytes_read"]
            require(budget > 0, "read_budget")
            chunk = os.read(descriptor, min(65_536, remaining, budget))
            require(bool(chunk), "file_changed")
            accounting["candidate_bytes_read"] += len(chunk)
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        unchanged()
        require(hashlib.sha256(payload).hexdigest() == digest, "hash_mismatch")
        yield payload
        unchanged()
    except OSError:
        raise Stop("file_rejected") from None
    finally:
        close_failed = False
        for descriptor in reversed(handles):
            try:
                os.close(descriptor)
            except OSError:
                close_failed = True
        require(not close_failed, "file_rejected")


class DiscardText:
    def write(self, value):
        return len(value)

    def flush(self):
        pass


def load_checker():
    path = ROOT / "src/lumi/posttraining_safety_uri_row_delta_v1.py"
    spec = importlib.util.spec_from_file_location("safety_row_delta", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit_pairs


def deadline(_number, _frame):
    raise Stop("deadline_exceeded")


def main(argv=None):
    output = {
        "schema_version": "lumi.sft-safety-uri-row-delta-audit-report.v0.1",
        "status": "STOP",
        "row_delta_verified": False,
        "rows_compared": 0,
        "input_hashes_authenticated": False,
        "input_files_preserved": False,
        "tokenizer_recomputed": False,
        "generic_filtering_replayed": False,
        "full_private_integration_passed": False,
        "execution_authorized": False,
        "production_eligible": False,
    }
    accounting = {"candidate_files_opened": 0, "candidate_bytes_read": 0}
    result = None
    try:
        require(
            sys.flags.no_site == 1
            and sys.dont_write_bytecode
            and not sys.flags.optimize
            and os.name == "posix"
            and hasattr(signal, "SIGALRM"),
            "runtime_rejected",
        )
        require(
            not any(
                name.split(".")[0]
                in {
                    "lumi",
                    "torch",
                    "transformers",
                    "numpy",
                    "tokenizers",
                    "tensorflow",
                    "jax",
                }
                for name in sys.modules
            ),
            "runtime_rejected",
        )
        require(
            list(sys.argv[1:] if argv is None else argv) == ["--audit-existing-local-pair"],
            "arguments_rejected",
        )
        started = time.monotonic()
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(MAXIMUM_SECONDS)
        with contextlib.redirect_stdout(DiscardText()), contextlib.redirect_stderr(DiscardText()):
            checker = load_checker()
            with contextlib.ExitStack() as stack:
                payloads = []
                for name, size, digest in BINDINGS:
                    # Lexical normalization, not resolve(): symlinks must remain visible.
                    path = Path(os.path.abspath(ROOT / name))
                    payloads.append(stack.enter_context(bound_file(path, size, digest, accounting)))
                try:
                    result = checker(*payloads)
                except ValueError:
                    raise Stop("row_delta_rejected") from None
            require(time.monotonic() - started < MAXIMUM_SECONDS, "deadline_exceeded")
    except Stop as error:
        output["failure_code"] = error.code
    except BaseException:
        output["failure_code"] = "audit_failed"
    finally:
        try:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
        except Stop as error:
            output["failure_code"] = error.code
        except BaseException:
            output["failure_code"] = "audit_failed"
    if result is not None and "failure_code" not in output:
        output = {**result, "input_hashes_authenticated": True, "input_files_preserved": True}
    output.update(accounting)
    output.update(
        {
            "input_files_written": 0,
            "network_used": False,
            "gpu_used": False,
            "inference_called": False,
        }
    )
    print(json.dumps(output, sort_keys=True))
    return 0 if output["row_delta_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
