"""Run four unchanged vocab-report unit tests with local metadata, not private replay.

Use the existing installed Python runtime with -B. This process installs a
permanent Python audit guard, so call it as a standalone command, not in-process.
The guard is defense in depth, not an OS sandbox. It permits code/runtime reads
but no private file contents, writes, network, subprocesses or ML imports.
Private plan/marker contents are NOT authenticated by these unit tests. Nothing
here waives hosted CI failures or authorizes the old diagnostic/curation runner.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import signal
import stat
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "lumi.vocab-report-local-unit-ci.v0.1"
SOURCE_BINDINGS = {
    "scripts/diagnose_sft_bedstc2_vocab_contract_v5.py":
        "a042a4ee9607e1a8fdd06a37604c7fc84b60e917f4c8be610746d493ad663375",
    "tests/test_bedstc2_vocab_contract_diagnostic_v5.py":
        "1d3b4ae45c714fe4b13529306e790adc0009bb73a29b675d8c5782f943bb92c2",
}
TEST_PATH = "tests/test_bedstc2_vocab_contract_diagnostic_v5.py"
METADATA_PATHS = (
    "configs/lumi-4b-sft-bedstc2-banglish-conversation-curation-recovery-v0.5.json",
    "evidence/lumi-4b-sft-bedstc2-banglish-conversation-curation-v0.5.json",
)
TEST_NAMES = (
    "test_report_rejects_resealed_predecessor_binding_tamper",
    "test_report_recomputes_mapping_and_conditional_gate",
    "test_report_rejects_resealed_logical_mapping_tamper",
    "test_verification_is_linked_to_exact_report_payload_and_gate",
)
WRITE_EVENTS = frozenset({
    "os.remove", "os.rename", "os.mkdir", "os.rmdir", "os.link", "os.symlink",
    "os.chmod", "os.chown", "os.utime", "os.truncate", "shutil.copyfile",
})
PROCESS_EVENTS = frozenset({
    "subprocess.Popen", "os.system", "os.fork", "os.forkpty", "os.posix_spawn", "os.exec",
})
ML_MODULES = frozenset({
    "torch", "transformers", "tensorflow", "jax", "huggingface_hub", "tokenizers",
    "safetensors", "numpy", "accelerate", "datasets", "sentencepiece",
})


class LocalUnitError(ValueError):
    """A fixed failure code, without private or exception text."""


class DeadlineExceeded(BaseException):
    """Cannot satisfy an original test's expected ValueError assertion."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise LocalUnitError(code)


def identity(path: Path) -> tuple:
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode) and value.st_nlink == 1, "file_shape")
    return (
        value.st_dev, value.st_ino, value.st_mode, value.st_nlink, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )


def no_link_path(root: Path, relative: str) -> Path:
    current = root
    for part in Path(relative).parts:
        current = current / part
        require(not current.is_symlink(), "file_symlink")
    return current


def verify_sources(root: Path) -> None:
    for relative, digest in SOURCE_BINDINGS.items():
        path = no_link_path(root, relative)
        before = identity(path)
        require(0 < before[4] <= 131_072, "source_size")
        with path.open("rb") as handle:
            payload = handle.read(131_073)
        require(identity(path) == before, "source_changed")
        require(hashlib.sha256(payload).hexdigest() == digest, "source_hash")


def private_metadata(root: Path) -> tuple:
    values = tuple(identity(no_link_path(root, name)) for name in METADATA_PATHS)
    require(0 < values[0][4] <= 131_072 and values[1][4] == 47, "private_metadata_shape")
    return values


def forbidden_event(event: str, arguments: tuple, allowed_roots: tuple[Path, ...]) -> str | None:
    if event == "open":
        target, _mode, flags = arguments
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            return "writes"
        if isinstance(target, int):
            return "file_content"
        path = Path(os.fsdecode(target)).resolve()
        if not any(path.is_relative_to(base) for base in allowed_roots):
            return "file_content"
    elif event.startswith("socket."):
        return "network"
    elif event in PROCESS_EVENTS:
        return "process"
    elif event in WRITE_EVENTS:
        return "writes"
    elif event == "import" and arguments[0].split(".")[0] in ML_MODULES:
        return "ml_import"
    return None


class DiscardText:
    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        pass


def run_fixed_tests(module) -> tuple[int, int]:
    passed = failed = 0
    for name in TEST_NAMES:
        try:
            getattr(module, name)()
        except (DeadlineExceeded, LocalUnitError):
            raise
        except BaseException:
            failed += 1
        else:
            passed += 1
    return passed, failed


def deadline(_number, _frame) -> None:
    raise DeadlineExceeded()


def main() -> int:
    blocked = dict.fromkeys(("file_content", "writes", "network", "process", "ml_import"), 0)
    passed = failed = 0
    preserved = False
    status = "STOP"
    started = time.monotonic()
    try:
        require(sys.dont_write_bytecode and not sys.argv[1:], "startup")
        require(not any(name in sys.modules for name in ML_MODULES), "fresh_process")
        require(hasattr(signal, "SIGALRM"), "posix_required")
        # A runtime prefix must not accidentally authorize the workspace as code.
        prefixes = tuple(Path(value).resolve() for value in (sys.prefix, sys.base_prefix))
        require(all(not ROOT.is_relative_to(p) for p in prefixes), "runtime_scope")
        allowed_roots = (ROOT / "scripts", ROOT / "src", ROOT / "tests", *prefixes)

        def guard(event: str, arguments: tuple) -> None:
            kind = forbidden_event(event, arguments, allowed_roots)
            if kind is not None:
                blocked[kind] += 1
                raise LocalUnitError("boundary")

        sys.addaudithook(guard)
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(30)
        verify_sources(ROOT)
        before = private_metadata(ROOT)
        sys.path.insert(0, str(ROOT))
        with (
            contextlib.redirect_stdout(DiscardText()),
            contextlib.redirect_stderr(DiscardText()),
        ):
            spec = importlib.util.spec_from_file_location(
                "fixed_vocab_report_tests", ROOT / TEST_PATH
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            passed, failed = run_fixed_tests(module)
        verify_sources(ROOT)
        preserved = private_metadata(ROOT) == before
        require(preserved and passed == 4 and failed == 0 and not any(blocked.values()), "tests")
        require(time.monotonic() - started < 30, "deadline")
        status = "PASS_LOCAL_UNIT_ONLY"
    except BaseException:
        pass
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
    print(json.dumps({
        "schema_version": SCHEMA,
        "status": status,
        "selected_tests": 4,
        "passed": passed,
        "failed": failed,
        "blocked_operations": blocked,
        "source_hashes_and_private_metadata_preserved": preserved,
        "historical_private_contents_verified": False,
        "hosted_ci_fixed": False,
        "operational_diagnostic_executed": False,
        "raw_test_output_retained": False,
    }, sort_keys=True))
    return 0 if status == "PASS_LOCAL_UNIT_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
