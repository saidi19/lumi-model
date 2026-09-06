"""Bounded read-only historical safety reproduction; aggregate stdout only.

Use the trusted local venv with -S -B. --check-runtime reads no private inputs;
--reproduce-existing-outputs runs one bounded replay without an old runner,
publisher, artifact write, model construction, inference or GPU operation.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import select
import signal
import sqlite3  # noqa: F401 -- warm trusted stdlib before private data
import stat
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "configs/lumi-safety-output-replay-inputs-v0.1.json"
SPEC_SHA256 = "42aaeba21e68d249f41671587c74a92c7adfb6f298e0e19a2efbe390245ac33c"
SITE = ROOT / "artifacts/lumi-sft-import-runtime-v0.1/venv/lib/python3.13/site-packages"
MAX_FILES, MAX_BYTES, MAX_SECONDS = 24, 12_000_000, 120
CODES = frozenset(
    {
        "arguments_rejected",
        "runtime_rejected",
        "spec_rejected",
        "history_rejected",
        "input_rejected",
        "input_changed",
        "input_hash_mismatch",
        "read_budget",
        "boundary_violation",
        "deadline_exceeded",
        "reproduction_rejected",
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
        getattr(info, key)
        for key in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    )


class Boundary:
    """Python audit defense in depth; not an OS sandbox or native syscall tracer."""

    def __init__(self):
        self.bound_open = False
        self.blocked = 0
        self.roots = (str(Path(sys.base_prefix).resolve()), str(SITE), str(ROOT / "src"))

    def guard(self, event, args):
        blocked = False
        if event == "open":
            path, _mode, flags = args
            blocked = bool(
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            )
            if not blocked and not self.bound_open:
                if not isinstance(path, (str, bytes)):
                    blocked = True
                else:
                    absolute = os.path.abspath(os.fsdecode(path))
                    blocked = not any(
                        absolute == root or absolute.startswith(root + "/") for root in self.roots
                    )
        elif event.startswith("socket.") or event in {
            "subprocess.Popen",
            "os.system",
            "os.fork",
            "os.forkpty",
            "os.posix_spawn",
            "os.exec",
            "os.remove",
            "os.rename",
            "os.mkdir",
            "os.rmdir",
            "os.link",
            "os.symlink",
            "os.chmod",
            "os.chown",
            "os.utime",
            "os.truncate",
            "shutil.copyfile",
            "sqlite3.connect",
        }:
            blocked = True
        if blocked:
            self.blocked += 1
            raise Stop("boundary_violation")


class BoundInputs:
    """Shared no-follow directory descriptors; each pinned input is read once."""

    def __init__(self, boundary):
        self.boundary = boundary
        self.directories, self.leaves = {}, []
        self.descriptors = []
        self.names = set()
        self.opens = self.bytes_read = 0

    def read(self, name, binding):
        path = Path(os.path.abspath(ROOT / name))
        require(str(path) not in self.names, "input_rejected")
        self.names.add(str(path))
        require(path.is_relative_to(ROOT.parent) and len(path.parts) <= 64, "input_rejected")
        require(self.opens < MAX_FILES and 0 < binding["bytes"] <= 6_000_000, "read_budget")
        self.boundary.bound_open = True
        try:
            directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
            if "/" not in self.directories:
                fd = os.open("/", directory_flags)
                self.descriptors.append(fd)
                self.directories["/"] = (fd, None, None, os.fstat(fd))
            parent = self.directories["/"][0]
            prefix = ""
            for part in path.parts[1:-1]:
                prefix += "/" + part
                if prefix not in self.directories:
                    fd = os.open(part, directory_flags, dir_fd=parent)
                    self.descriptors.append(fd)
                    self.directories[prefix] = (fd, parent, part, os.fstat(fd))
                parent = self.directories[prefix][0]
            fd = os.open(
                path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=parent
            )
            self.descriptors.append(fd)
            self.opens += 1
            before = os.fstat(fd)
            self.leaves.append((fd, parent, path.name, before))
            require(
                stat.S_ISREG(before.st_mode)
                and before.st_nlink == 1
                and before.st_size == binding["bytes"],
                "input_rejected",
            )
            chunks, remaining = [], before.st_size
            while remaining:
                require(self.bytes_read < MAX_BYTES, "read_budget")
                block = os.read(fd, min(remaining, 65_536, MAX_BYTES - self.bytes_read))
                require(bool(block), "input_changed")
                chunks.append(block)
                remaining -= len(block)
                self.bytes_read += len(block)
            data = b"".join(chunks)
            require(hashlib.sha256(data).hexdigest() == binding["sha256"], "input_hash_mismatch")
            self.verify()
            return data
        except OSError:
            raise Stop("input_rejected") from None
        finally:
            self.boundary.bound_open = False

    def verify(self):
        for fd, parent, name, before in self.directories.values():
            values = [os.fstat(fd)]
            if parent is not None:
                values.append(os.stat(name, dir_fd=parent, follow_symlinks=False))
            require(
                all(
                    (info.st_dev, info.st_ino, info.st_mode)
                    == (before.st_dev, before.st_ino, before.st_mode)
                    for info in values
                ),
                "input_changed",
            )
        for fd, parent, name, before in self.leaves:
            require(
                identity(before)
                == identity(os.fstat(fd))
                == identity(os.stat(name, dir_fd=parent, follow_symlinks=False)),
                "input_changed",
            )

    def close(self):
        failed = False
        for fd in reversed(self.descriptors):
            try:
                os.close(fd)
            except OSError:
                failed = True
        self.descriptors.clear()
        require(not failed, "input_changed")


class DiscardText:
    def write(self, value):
        return len(value)

    def flush(self):
        pass


def history_bytes(binding):
    environment = {
        **os.environ,
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }
    process = subprocess.Popen(
        [
            "/usr/bin/git",
            "--no-pager",
            "cat-file",
            "blob",
            binding["git_revision"] + ":" + binding["path"],
        ],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )
    try:
        expires = time.monotonic() + 10
        data = bytearray()
        while True:
            remaining = expires - time.monotonic()
            require(remaining > 0, "history_rejected")
            ready, _, _ = select.select([process.stdout], [], [], remaining)
            require(bool(ready), "history_rejected")
            block = os.read(process.stdout.fileno(), binding["bytes"] + 1 - len(data))
            if not block:
                break
            data.extend(block)
            require(len(data) <= binding["bytes"], "history_rejected")
        require(
            process.wait(timeout=max(0.001, expires - time.monotonic())) == 0
            and len(data) == binding["bytes"]
            and hashlib.sha256(data).hexdigest() == binding["sha256"],
            "history_rejected",
        )
        return bytes(data)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=1)
        process.stdout.close()


def runtime():
    # Configure the standard-library cache without its optional writable-dir
    # probing. No temporary file is needed by this read-only tokenizer task.
    tempfile.tempdir = "/private/tmp" if sys.platform == "darwin" else "/tmp"
    sys.path.insert(0, str(SITE))
    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }.items():
        os.environ[key] = value
    import tokenizers
    import transformers
    from transformers.tokenization_utils_tokenizers import TokenizersBackend

    require(
        tokenizers.__version__ == "0.22.2"
        and transformers.__version__ == "5.2.0"
        and transformers.TokenizersBackend is TokenizersBackend,
        "runtime_rejected",
    )
    package_name = "_lumi_safety_replay_entry"
    require(package_name not in sys.modules, "runtime_rejected")
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "src/lumi")]
    sys.modules[package_name] = package
    return importlib.import_module(package_name + ".posttraining_safety_output_replay_v1")


def warm_filelock_probe():
    """Allow the reviewed empty-file runtime capability probe before private IO.

    filelock/_strict.py checks hard-link support in a TemporaryDirectory at
    import, cleans it itself, and reads no dataset. This is runtime setup, not
    a private replay write. Network/process creation remains forbidden.
    """

    def network_guard(event, _args):
        if event.startswith("socket.") or event in {
            "subprocess.Popen",
            "os.system",
            "os.fork",
            "os.forkpty",
            "os.posix_spawn",
            "os.exec",
        }:
            raise Stop("boundary_violation")

    sys.addaudithook(network_guard)
    sys.path.insert(0, str(SITE))
    tempfile.tempdir = "/private/tmp" if sys.platform == "darwin" else "/tmp"
    importlib.import_module("filelock")


def deadline(_number, _frame):
    raise Stop("deadline_exceeded")


def read_spec():
    with SPEC_PATH.open("rb") as handle:
        payload = handle.read(32_768)
    require(
        len(payload) < 32_768 and hashlib.sha256(payload).hexdigest() == SPEC_SHA256,
        "spec_rejected",
    )
    return json.loads(payload)


def main(argv=None):
    output = {
        "schema_version": "lumi.safety-output-replay-local.v0.1",
        "status": "STOP",
        "safety_outputs_reproduced": False,
        "full_private_integration_passed": False,
        "execution_authorized": False,
        "production_eligible": False,
    }
    inputs = boundary = result = historical = None
    stage = "argument_validation"
    try:
        started = time.monotonic()
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(MAX_SECONDS)
        require(
            sys.flags.no_site and sys.dont_write_bytecode and not sys.flags.optimize,
            "runtime_rejected",
        )
        arguments = list(sys.argv[1:] if argv is None else argv)
        require(
            arguments in (["--check-runtime"], ["--reproduce-existing-outputs"]),
            "arguments_rejected",
        )
        stage = "spec_validation"
        spec = read_spec()
        bindings = spec["bindings"]
        if arguments == ["--reproduce-existing-outputs"]:
            stage = "history_acquisition"
            historical = history_bytes(bindings["posttraining_data_implementation"])
        stage = "runtime_startup"
        warm_filelock_probe()
        with contextlib.redirect_stdout(DiscardText()), contextlib.redirect_stderr(DiscardText()):
            # Trusted runtime startup probes happen with network/process denial
            # and before private inputs. Strict IO denial begins after warming.
            core = runtime()
            boundary = Boundary()
            sys.addaudithook(boundary.guard)
            if arguments == ["--check-runtime"]:
                result = {**output, "status": "PASS_RUNTIME_ONLY"}
            else:
                stage = "bound_input_acquisition"
                inputs = BoundInputs(boundary)
                contents = {"posttraining_data_implementation": historical}
                for role, binding in bindings.items():
                    if role not in contents:
                        contents[role] = inputs.read(binding["path"], binding)
                try:

                    def record_stage(code):
                        nonlocal stage
                        require(code in core.STAGES, "reproduction_rejected")
                        stage = code

                    result = core.reproduce(contents, bindings, str(ROOT), on_stage=record_stage)
                except Exception:
                    raise Stop("reproduction_rejected") from None
                stage = "input_preservation"
                inputs.verify()
            require(
                time.monotonic() - started < MAX_SECONDS and boundary.blocked == 0,
                "deadline_exceeded",
            )
    except Stop as error:
        output["failure_code"] = error.code
    except BaseException:
        output["failure_code"] = "audit_failed"
    finally:
        try:
            if inputs is not None:
                inputs.close()
        except Stop as error:
            output["failure_code"] = error.code
        except BaseException:
            output["failure_code"] = "audit_failed"
        try:
            signal.alarm(0)
        except Stop as error:
            output["failure_code"] = error.code
        except BaseException:
            output["failure_code"] = "audit_failed"
    if result is not None and "failure_code" not in output:
        output.update(result)
        if inputs is not None:
            output["input_hashes_authenticated"] = True
            output["input_files_preserved"] = True
    output.update(
        {
            "last_stage": stage,
            "bound_input_opens": inputs.opens if inputs else 0,
            "bound_input_bytes_read": inputs.bytes_read if inputs else 0,
            "historical_source_bytes_read": len(historical) if historical is not None else 0,
            "blocked_operation_attempts": boundary.blocked if boundary else 0,
            "input_files_written": 0,
            "network_used": False,
            "gpu_used": False,
            "inference_called": False,
        }
    )
    print(json.dumps(output, sort_keys=True))
    return 0 if output["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
