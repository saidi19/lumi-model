"""Run 12 unchanged plan-dependent seal unit cases, never the seal operation.

Run the existing local venv Python with -S -B as a standalone process. The
original module's plan fixture reads one exact bound plan; its manifest/audit
fixtures remain synthetic. The operational load_plan, materializer, tokenizer,
candidate readers, and marker writers are never called. This is not a private
integration replay, approval, seal, or hosted-CI waiver.

The permanent Python audit guard permits code/runtime reads plus exactly three
plan opens (hash-before, original fixture, hash-after), and rejects other reads,
writes, network, subprocesses, and ML imports. It is not an OS sandbox. Original
test output is discarded at the Python stream level; only aggregates are emitted.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib
import importlib.util
import json
import os
import signal
import stat
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "lumi.learning-seal-local-unit-ci.v0.1"
PLAN_PATH = "configs/lumi-4b-sft-learning-candidate-seal-v0.3.json"
PLAN_SHA256 = "c1d478db8b8e334e00c899857a92c289b4c418606266f91dac122c381e0041dc"
PLAN_BYTES = 18_454
TEST_PATH = "tests/test_learning_candidate_seal_v3.py"
RUNTIME_PYTHON = "artifacts/lumi-sft-import-runtime-v0.1/venv/bin/python"
RUNTIME_SITE = "artifacts/lumi-sft-import-runtime-v0.1/venv/lib/python3.13/site-packages"
SOURCE_BINDINGS = {
    "src/lumi/__init__.py":
        "66854248a044f5e9c70055df06f6d39806d0a1121ce53dfd9194575879c9f1ce",
    "src/lumi/posttraining_learning_candidate_seal_v3.py":
        "61369ba5d1a0602b30f200a56d79a21ff81ca3ec57110010cb05a1b882b6068b",
    TEST_PATH: "771d02e1031f4ca2f9538a9e1552c5b66bfc3345a3f0da8b692829d825a1118b",
}
CLOSURE_ROOT = "lumi.posttraining_learning_candidate_seal_v3"
CLOSURE_COUNT = 84
CLOSURE_SHA256 = "433153f24a0ece1d03138ca754a2d689cbf3a863305ccd3f0dd9d2173e3f8977"
MAX_SOURCE_BYTES = 131_072
PLAN_TEST = "test_plan_binds_exact_twenty_source_paths_without_row_reads"
MANIFEST_TEST = "test_manifest_rejects_resigned_boundary_drift"
VERIFICATION_TEST = "test_verification_rejects_resigned_audit_drift"
TEST_NAMES = (PLAN_TEST, MANIFEST_TEST, VERIFICATION_TEST)
PHASES = ("hash_before", "fixture", "hash_after")
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
    """Fixed failure code; never an external exception or private value."""


class BoundaryViolation(BaseException):
    """Cannot satisfy an original test's expected ValueError rejection."""


class DeadlineExceeded(BaseException):
    """Cannot satisfy an original test's expected ValueError rejection."""


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
    parts = Path(relative).parts
    require(parts and not Path(relative).is_absolute() and ".." not in parts, "path")
    current = root
    for part in parts:
        current = current / part
        require(not current.is_symlink(), "file_symlink")
    return current


def bounded_bytes(root: Path, relative: str, maximum: int) -> tuple[bytes, tuple]:
    path = no_link_path(root, relative)
    before = identity(path)
    require(0 < before[4] <= maximum, "file_size")
    with path.open("rb") as handle:
        payload = handle.read(maximum + 1)
    require(len(payload) == before[4] and identity(path) == before, "file_changed")
    return payload, before


def local_imports(payload: bytes) -> set[str]:
    """Only direct module-level imports form this reviewed import closure."""
    names = set()
    for node in ast.parse(payload).body:
        if isinstance(node, ast.ImportFrom):
            if node.level == 1:
                names.update(
                    ["lumi." + node.module] if node.module
                    else ["lumi." + item.name for item in node.names]
                )
            elif node.module and node.module.startswith("lumi."):
                names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(item.name for item in node.names if item.name.startswith("lumi."))
    return names


def verify_sources(root: Path) -> set[Path]:
    for relative, digest in SOURCE_BINDINGS.items():
        payload, _ = bounded_bytes(root, relative, MAX_SOURCE_BYTES)
        require(hashlib.sha256(payload).hexdigest() == digest, "source_hash")
    pending = {CLOSURE_ROOT}
    hashes = {}
    while pending:
        name = pending.pop()
        require(
            name.startswith("lumi.") and all(part.isidentifier() for part in name.split(".")),
            "import_name",
        )
        relative = "src/" + name.replace(".", "/") + ".py"
        if relative in hashes:
            continue
        require(len(hashes) < CLOSURE_COUNT, "closure_count")
        payload, _ = bounded_bytes(root, relative, MAX_SOURCE_BYTES)
        hashes[relative] = hashlib.sha256(payload).hexdigest()
        pending.update(local_imports(payload))
    canonical = "".join(name + "\0" + hashes[name] + "\n" for name in sorted(hashes))
    require(len(hashes) == CLOSURE_COUNT, "closure_count")
    require(hashlib.sha256(canonical.encode("ascii")).hexdigest() == CLOSURE_SHA256,
            "closure_hash")
    return {root / relative for relative in (*hashes, *SOURCE_BINDINGS)}


def verify_plan(root: Path) -> tuple:
    payload, observed = bounded_bytes(root, PLAN_PATH, PLAN_BYTES)
    require(len(payload) == PLAN_BYTES, "plan_size")
    require(hashlib.sha256(payload).hexdigest() == PLAN_SHA256, "plan_hash")
    return observed


class ReadBoundary:
    def __init__(self, plan: Path, allowed_roots: tuple[Path, ...]):
        self.plan = plan
        self.allowed_roots = allowed_roots
        self.phase = None
        self.plan_opens = dict.fromkeys(PHASES, 0)
        self.blocked = dict.fromkeys(
            ("file_content", "plan_open_cap", "writes", "network", "process", "ml_import"), 0
        )

    def classify(self, event: str, arguments: tuple) -> str | None:
        if event == "open":
            target, _mode, flags = arguments
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                return "writes"
            if isinstance(target, int):
                return "file_content"
            path = Path(os.fsdecode(target)).resolve()
            if path == self.plan:
                if self.phase not in PHASES or self.plan_opens[self.phase] != 0:
                    return "plan_open_cap"
                self.plan_opens[self.phase] += 1
            elif not any(path.is_relative_to(base) for base in self.allowed_roots):
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

    def guard(self, event: str, arguments: tuple) -> None:
        kind = self.classify(event, arguments)
        if kind is not None:
            self.blocked[kind] += 1
            raise BoundaryViolation()

    @contextlib.contextmanager
    def allow_plan(self, phase: str):
        require(phase in PHASES and self.phase is None, "plan_phase")
        self.phase = phase
        try:
            yield
        finally:
            self.phase = None


class DiscardText:
    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        pass


def original_parameters(module) -> tuple:
    marks = getattr(getattr(module, MANIFEST_TEST), "pytestmark", ())
    require(len(marks) == 1 and marks[0].name == "parametrize", "parameters")
    mark = marks[0]
    require(len(mark.args) == 2 and mark.args[0] == ("path", "value") and not mark.kwargs,
            "parameters")
    values = mark.args[1]
    require(isinstance(values, list) and len(values) == 10, "parameters")
    require(all(isinstance(item, tuple) and len(item) == 2 for item in values), "parameters")
    return tuple(values)


def run_fixed_tests(module, plan: dict) -> tuple[int, int]:
    parameters = original_parameters(module)
    manifest = module.manifest.__wrapped__(plan)
    calls = [(getattr(module, PLAN_TEST), (plan,))]
    calls.extend((getattr(module, MANIFEST_TEST), (plan, manifest, *item)) for item in parameters)
    calls.append((getattr(module, VERIFICATION_TEST), (plan, manifest)))
    passed = failed = 0
    for function, arguments in calls:
        try:
            function(*arguments)
        except (BoundaryViolation, DeadlineExceeded, LocalUnitError):
            raise
        except BaseException:
            failed += 1
        else:
            passed += 1
    return passed, failed


def preload_pytest(site: Path) -> None:
    original = list(sys.path)
    require(str(site) not in original, "site_preloaded")
    try:
        sys.path.insert(0, str(site))
        importlib.import_module("pytest")
    finally:
        sys.path[:] = original


def verify_loaded_lumi(allowed: set[Path]) -> None:
    loaded = [(name, module) for name, module in sys.modules.items()
              if name == "lumi" or name.startswith("lumi.")]
    require(loaded, "lumi_missing")
    for _name, module in loaded:
        filename = getattr(module, "__file__", None)
        require(isinstance(filename, str) and Path(filename).resolve() in allowed,
                "unbound_lumi_import")


def deadline(_number, _frame) -> None:
    raise DeadlineExceeded()


def main() -> int:
    boundary = ReadBoundary(ROOT / PLAN_PATH, ())
    passed = failed = 0
    preserved = False
    status = "STOP"
    started = time.monotonic()
    try:
        require(sys.dont_write_bytecode and sys.flags.no_site and not sys.argv[1:], "startup")
        require(sys.version_info[:2] == (3, 13), "runtime_version")
        require(Path(sys.executable).absolute() == ROOT / RUNTIME_PYTHON, "runtime_path")
        require(not any(name.split(".")[0] in {*ML_MODULES, "lumi", "pytest"}
                        for name in sys.modules), "fresh_process")
        require(hasattr(signal, "SIGALRM"), "posix_required")
        prefix = Path(sys.base_prefix).resolve()
        require(not ROOT.is_relative_to(prefix) and prefix != Path("/"), "runtime_scope")
        site = no_link_path(ROOT, RUNTIME_SITE)
        require(site.is_dir(), "runtime_site")
        boundary.allowed_roots = (ROOT / "src", ROOT / "tests", prefix, site)
        sys.addaudithook(boundary.guard)
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(30)
        sys.path[:] = [value for value in sys.path
                       if value and Path(value).resolve().is_relative_to(prefix)]
        sources = verify_sources(ROOT)
        with boundary.allow_plan("hash_before"):
            before = verify_plan(ROOT)
        with (
            contextlib.redirect_stdout(DiscardText()),
            contextlib.redirect_stderr(DiscardText()),
        ):
            preload_pytest(site)
            sys.path.insert(0, str(ROOT / "src"))
            spec = importlib.util.spec_from_file_location(
                "fixed_learning_seal_tests", ROOT / TEST_PATH
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            verify_loaded_lumi(sources)
            require(module.PLAN == ROOT / PLAN_PATH, "fixture_path")
            with boundary.allow_plan("fixture"):
                plan = module.plan.__wrapped__()
            passed, failed = run_fixed_tests(module, plan)
        require(verify_sources(ROOT) == sources, "source_closure")
        verify_loaded_lumi(sources)
        with boundary.allow_plan("hash_after"):
            preserved = verify_plan(ROOT) == before
        require(boundary.plan_opens == dict.fromkeys(PHASES, 1), "plan_open_count")
        require(preserved and passed == 12 and failed == 0 and not any(boundary.blocked.values()),
                "tests")
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
        "selected_tests": 12,
        "passed": passed,
        "failed": failed,
        "plan_opens": boundary.plan_opens,
        "blocked_operations": boundary.blocked,
        "source_hashes_and_plan_preserved": preserved,
        "private_integration_reproduced": False,
        "hosted_ci_fixed": False,
        "operational_seal_executed": False,
        "candidate_or_tokenizer_contents_read": False,
        "raw_test_output_retained": False,
    }, sort_keys=True))
    return 0 if status == "PASS_LOCAL_UNIT_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
