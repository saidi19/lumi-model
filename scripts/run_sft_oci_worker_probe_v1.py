#!/usr/bin/env python3
"""Remote OCI-only WorkerRun origin witness; NEVER import or execute on the Mac.

Only the six accepted generated rank workers import ML. Their existing inputs
barrier and final result each receive a bounded origin observation. The original
fixture bytes, training/accounting logic and cleanup remain unchanged. This is
not a real-model/data, CUDA, hostile-host, or complete dynamic-code admission.

The outer trusted builder authenticates the final immutable image ID, its base
digest, control-file pins and no-host-software-bind/read-only launch settings.
These point-in-time file-origin checks do not rehash large wheel libraries or
authenticate anonymous executable memory. No raw paths, IDs or errors are
included in the public report.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import stat
import sys
from pathlib import Path

SCHEMA = "lumi.sft-oci-worker-probe.v0.1"
LOCK_SCHEMA = "lumi.sft-linux-cpu-release-lock.v0.1"
BASE_SCHEMA = "lumi.sft-oci-base-inventory.v0.1"
SOURCE_SCHEMA = "lumi.sft-oci-source-inventory.v0.1"
SOURCE_ROOT = "/opt/lumi-release/source"
CONTROL_ROOT = "/opt/lumi-release/control"
RUNTIME = "/usr/local/bin/python3.12"
RUNTIME_PREFIX = "/usr/local"
STDLIB_ROOT = "/usr/local/lib/python3.12"
DEPENDENCY_ROOT = "/usr/local/lib/python3.12/site-packages"
FIXTURE_NAME = "_lumi_oci_frozen_worker_v1"
FROZEN_RELATIVE = "scripts/run_sft_worker_run_torch_fixture_v1.py"
FROZEN_SIZE = 39948
FROZEN_SHA256 = "53c50c92965ed1cef0c9624c57367c31ed7ebfb9d7cfb34257188ad179d333d0"
MAXIMUM_CONTROL_BYTES = 16777216
MAXIMUM_INVENTORY_FILES = 65536
MAXIMUM_MODULES = 16384
MAXIMUM_DISTRIBUTIONS = 256
MAXIMUM_MAP_BYTES = 1048576
MAXIMUM_MAP_LINES = 8192
MAXIMUM_REPORT_BYTES = 8192
PROBE_STAGES = ("guard", "controls", "fixture_binding", "worker_run", "complete")
FIXTURE_STAGES = (
    "guard", "fixture", "workers", "ledger", "cleanup", "comparison", "accounting", "complete",
)
WITNESS_FIELDS = (
    "file_backed_modules", "source_modules", "dependency_modules", "runtime_modules",
    "distribution_versions", "map_lines", "executable_file_mappings",
    "executable_files", "anonymous_executable_mappings", "generated_python_modules",
)
NONCLAIMS = (
    "real_sft_execution_permitted", "production_eligible", "real_model_or_data_access",
    "cuda_used", "complete_dynamic_dependency_closure_verified",
    "all_executable_code_origin_authenticated", "large_library_rehash_performed",
    "host_configuration_independently_verified", "real_host_admission_verified",
)
ERROR_CODES = frozenset((
    "runtime_rejected", "control_rejected", "fixture_binding_rejected",
    "module_origin_rejected", "distribution_rejected", "native_origin_rejected",
    "witness_sequence_rejected", "fixture_failed", "output_failed", "probe_failed",
))
_STATE = None
_FIXTURE = None
_ORIGINAL_OUTCOME = None
_ORIGINAL_INVOCATION = None
_INPUT_WITNESS = None
_RANKS = 0
_INVOCATIONS = 0
_COUNTS = dict.fromkeys(WITNESS_FIELDS, 0)
_STAGE = "guard"


class ProbeError(ValueError):
    def __init__(self, code):
        self.code = code if type(code) is str and code in ERROR_CODES else "probe_failed"
        super().__init__(self.code)


def need(condition, code):
    if not condition:
        raise ProbeError(code) from None


def sha256(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def below(path, root):
    return path == root or path.startswith(root + "/")


def canonical_path(path, code):
    need(type(path) is str and path.startswith("/") and len(path) <= 4096, code)
    need(not any(ord(char) < 32 or ord(char) == 127 for char in path), code)
    resolved = str(Path(path).resolve(strict=True))
    info = os.stat(resolved, follow_symlinks=False)
    need(stat.S_ISREG(info.st_mode), code)
    return resolved, info


def pinned_bytes(path, expected, maximum, code):
    need(sha256(expected), code)
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= maximum, code)
        raw = bytearray()
        while len(raw) < before.st_size:
            part = os.read(descriptor, min(65536, before.st_size - len(raw)))
            need(bool(part), code)
            raw.extend(part)
        need(os.read(descriptor, 1) == b"", code)
        after = os.fstat(descriptor)
        need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
              before.st_ctime_ns) == (after.st_dev, after.st_ino, after.st_size,
                                      after.st_mtime_ns, after.st_ctime_ns), code)
        need(hashlib.sha256(raw).hexdigest() == expected, code)
        return bytes(raw)
    finally:
        os.close(descriptor)


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, "control_rejected")
        result[key] = value
    return result


def control(filename, environment):
    expected = os.environ.get(environment)
    raw = pinned_bytes(CONTROL_ROOT + "/" + filename, expected,
                       MAXIMUM_CONTROL_BYTES, "control_rejected")
    value = json.loads(raw, object_pairs_hook=unique_pairs)
    need(type(value) is dict, "control_rejected")
    return value, expected


def normalized_name(name):
    need(type(name) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name),
         "distribution_rejected")
    return re.sub(r"[-_.]+", "-", name).lower()


def inventory(rows, *, source):
    need(type(rows) is list and 1 <= len(rows) <= MAXIMUM_INVENTORY_FILES,
         "control_rejected")
    result, previous = {}, None
    for row in rows:
        need(type(row) is dict and set(row) == {"path", "bytes", "sha256"},
             "control_rejected")
        path = row["path"]
        need(type(path) is str and 1 <= len(path) <= 4096
             and type(row["bytes"]) is int and 0 <= row["bytes"] <= 2147483648
             and sha256(row["sha256"]), "control_rejected")
        need(previous is None or previous < path, "control_rejected")
        previous = path
        if source:
            need(not path.startswith("/") and "\\" not in path
                 and all(part not in ("", ".", "..") for part in path.split("/")),
                 "control_rejected")
            path = SOURCE_ROOT + "/" + path
        else:
            need(path == RUNTIME or below(path, "/usr/lib") or below(path, "/usr/local/lib")
                 or below(path, "/etc/ssl") or below(path, "/usr/share/zoneinfo"),
                 "control_rejected")
            need(str(Path(path)) == path and ".." not in Path(path).parts,
                 "control_rejected")
        result[path] = dict(row)
    return result


def runtime_guard():
    need(sys.platform == "linux" and os.uname().machine == "x86_64"
         and sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12)
         and sys.flags.isolated == 1 and sys.dont_write_bytecode
         and os.geteuid() == os.getegid() == 10001, "runtime_rejected")
    need(str(Path(sys.executable).resolve(strict=True)) == RUNTIME
         and sys.prefix == sys.base_prefix == RUNTIME_PREFIX, "runtime_rejected")
    need("torch" not in sys.modules and "transformers" not in sys.modules,
         "runtime_rejected")
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
         and os.environ.get("LUMI_WORKER_RUN_TORCH_CI_FIXTURE") == "1"
         and os.environ.get("GLOO_SOCKET_IFNAME") == "lo", "runtime_rejected")


def load_controls():
    lock, lock_pin = control("software-lock.json", "LUMI_RELEASE_LOCK_SHA256")
    base, base_pin = control("base-system-inventory.json", "LUMI_RELEASE_BASE_INVENTORY_SHA256")
    source, source_pin = control("source-inventory.json", "LUMI_RELEASE_SOURCE_INVENTORY_SHA256")
    need(set(lock) == {"schema", "platform", "python_abi", "base", "requirements", "policy"}
         and lock["schema"] == LOCK_SCHEMA and lock["platform"] == "linux/amd64"
         and lock["python_abi"] == "cp312" and type(lock["policy"]) is dict,
         "control_rejected")
    need(lock["policy"] == {"device": "cpu", "strategy": "ddp", "max_context_window": 4095,
                            "production_eligible": False, "real_sft_execution_permitted": False},
         "control_rejected")
    selected = lock["base"]
    need(type(selected) is dict and set(selected) == {
        "repository", "digest", "config_digest", "python_version", "tag_reference",
    }, "control_rejected")
    need(set(base) == {"schema", "files", "base_distributions", "base_digest", "config_digest"}
         and base["schema"] == BASE_SCHEMA and base["base_digest"] == selected["digest"]
         and base["config_digest"] == selected["config_digest"], "control_rejected")
    need(all(type(selected[field]) is str
             and re.fullmatch(r"sha256:[0-9a-f]{64}", selected[field])
             for field in ("digest", "config_digest")), "control_rejected")
    need(selected["python_version"] == ".".join(map(str, sys.version_info[:3])),
         "runtime_rejected")
    commit = os.environ.get("LUMI_RELEASE_SOURCE_COMMIT")
    need(type(commit) is str and re.fullmatch(r"[0-9a-f]{40}", commit)
         and set(source) == {"schema", "source_commit", "files"}
         and source["schema"] == SOURCE_SCHEMA and source["source_commit"] == commit,
         "control_rejected")
    versions = {}
    for rows, exact in ((base["base_distributions"], {"name", "version"}),
                        (lock["requirements"], {"name", "version", "filename", "url",
                                                "sha256", "bytes"})):
        need(type(rows) is list and len(rows) <= MAXIMUM_DISTRIBUTIONS, "control_rejected")
        previous = None
        for row in rows:
            need(type(row) is dict and set(row) == exact, "control_rejected")
            name = normalized_name(row["name"])
            need(previous is None or previous < name, "control_rejected")
            previous = name
            need(type(row["version"]) is str and 1 <= len(row["version"]) <= 128,
                 "control_rejected")
            versions[name] = row["version"]
    need("torch" in versions and "transformers" in versions, "control_rejected")
    return {
        "base": inventory(base["files"], source=False),
        "source": inventory(source["files"], source=True), "versions": versions,
        "software_lock_sha256": lock_pin, "base_inventory_sha256": base_pin,
        "source_inventory_sha256": source_pin, "source_commit": commit,
    }


def file_origin(path):
    resolved, info = canonical_path(path, "module_origin_rejected")
    if below(resolved, DEPENDENCY_ROOT):
        return "dependency", resolved, info
    if below(resolved, SOURCE_ROOT):
        row = _STATE["source"].get(resolved)
        need(row is not None and row["bytes"] == info.st_size, "module_origin_rejected")
        return "source", resolved, info
    need(resolved in _STATE["base"] and _STATE["base"][resolved]["bytes"] == info.st_size,
         "module_origin_rejected")
    return "runtime", resolved, info


def distribution_versions():
    observed = {}
    for distribution in importlib.metadata.distributions(path=[DEPENDENCY_ROOT]):
        need(len(observed) < MAXIMUM_DISTRIBUTIONS, "distribution_rejected")
        name = normalized_name(distribution.metadata["Name"])
        need(name not in observed, "distribution_rejected")
        observed[name] = distribution.version
    need(observed == _STATE["versions"], "distribution_rejected")
    return len(observed)


def module_origins():
    need(len(sys.modules) <= MAXIMUM_MODULES, "module_origin_rejected")
    need(len(sys.path) <= 32, "module_origin_rejected")
    for entry_path in sys.path:
        need(type(entry_path) is str and entry_path.startswith("/")
             and (below(entry_path, "/usr/local/lib")
                  or below(entry_path, SOURCE_ROOT + "/src")), "module_origin_rejected")
    counts = dict.fromkeys(("file_backed_modules", "source_modules", "dependency_modules",
                            "runtime_modules", "generated_python_modules"), 0)
    core = set()
    for name, module in tuple(sys.modules.items()):
        attributes = getattr(module, "__dict__", {})
        path = attributes.get("__file__")
        locations = attributes.get("__path__")
        if locations is not None:
            need(not isinstance(locations, (str, bytes)), "module_origin_rejected")
            for index, location in enumerate(locations):
                need(index < 16 and type(location) is str, "module_origin_rejected")
                resolved_location = str(Path(location).resolve(strict=True))
                need(below(resolved_location, STDLIB_ROOT)
                     or below(resolved_location, SOURCE_ROOT + "/src"),
                     "module_origin_rejected")
        if name == "_remote_module_non_scriptable":
            # Torch 2.14's reviewed instantiator generates this one helper in
            # memory through _StringLoader. It is explicitly not a file-origin
            # claim; no temporary-directory import allowance is introduced.
            spec = attributes.get("__spec__")
            need(spec is not None and spec.origin == "torch-git"
                 and type(spec.loader).__name__ == "_StringLoader"
                 and type(spec.loader).__module__ == "torch.distributed.nn.jit.instantiator"
                 and path in (None, "<_remote_module_non_scriptable>.py"),
                 "module_origin_rejected")
            counts["generated_python_modules"] += 1
            continue
        if path is None:
            continue  # Built-in, frozen and generated fileless modules are not origin claims.
        category, resolved, _ = file_origin(path)
        counts["file_backed_modules"] += 1
        counts[category + "_modules"] += 1
        if name == "lumi" or name.startswith("lumi."):
            need(category == "source" and below(resolved, SOURCE_ROOT + "/src/lumi"),
                 "module_origin_rejected")
        if name == "torch" or name.startswith("torch."):
            need(category == "dependency" and below(resolved, DEPENDENCY_ROOT + "/torch"),
                 "module_origin_rejected")
        if name == "transformers" or name.startswith("transformers."):
            need(category == "dependency" and below(resolved, DEPENDENCY_ROOT + "/transformers"),
                 "module_origin_rejected")
        if name in ("lumi", "torch", "transformers", "torch._C"):
            core.add(name)
    need(core == {"lumi", "torch", "transformers", "torch._C"}, "module_origin_rejected")
    for name in ("torch", "transformers"):
        need(str(vars(sys.modules[name]).get("__version__")) == _STATE["versions"][name],
             "distribution_rejected")
    need(counts["source_modules"] > 0 and counts["runtime_modules"] > 0,
         "module_origin_rejected")
    return counts


def native_origins():
    descriptor = os.open("/proc/self/maps", os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        raw = bytearray()
        while len(raw) <= MAXIMUM_MAP_BYTES:
            block = os.read(descriptor, min(65536, MAXIMUM_MAP_BYTES + 1 - len(raw)))
            if not block:
                break
            raw.extend(block)
        need(len(raw) <= MAXIMUM_MAP_BYTES, "native_origin_rejected")
    finally:
        os.close(descriptor)
    lines = bytes(raw).decode("ascii").splitlines()
    need(1 <= len(lines) <= MAXIMUM_MAP_LINES, "native_origin_rejected")
    mapped, mappings, anonymous = set(), 0, 0
    for line in lines:
        fields = line.split(None, 5)
        need(len(fields) >= 5 and len(fields[1]) == 4, "native_origin_rejected")
        if "x" not in fields[1]:
            continue
        if len(fields) == 5 or fields[5].startswith("["):
            anonymous += 1
            continue
        path = fields[5]
        need(not path.endswith(" (deleted)"), "native_origin_rejected")
        resolved, info = canonical_path(path, "native_origin_rejected")
        device = fields[3].split(":")
        need(len(device) == 2 and info.st_ino == int(fields[4])
             and (os.major(info.st_dev), os.minor(info.st_dev))
             == (int(device[0], 16), int(device[1], 16)), "native_origin_rejected")
        need(below(resolved, DEPENDENCY_ROOT) or resolved in _STATE["base"],
             "native_origin_rejected")
        mapped.add(resolved)
        mappings += 1
    torch_native, _ = canonical_path(vars(sys.modules["torch._C"])["__file__"],
                                     "native_origin_rejected")
    need(torch_native in mapped and RUNTIME in mapped, "native_origin_rejected")
    return {"map_lines": len(lines), "executable_file_mappings": mappings,
            "executable_files": len(mapped), "anonymous_executable_mappings": anonymous}


def witness():
    return {**module_origins(), "distribution_versions": distribution_versions(),
            **native_origins()}


def observed_outcome(pipe, value):
    global _INPUT_WITNESS
    if value.get("status") == "phase" and value.get("phase") == "inputs":
        need(_INPUT_WITNESS is None, "witness_sequence_rejected")
        _INPUT_WITNESS = witness()
    elif value.get("status") == "passed":
        need(_INPUT_WITNESS is not None and "oci_origin_witness" not in value,
             "witness_sequence_rejected")
        final = witness()
        value = dict(value, oci_origin_witness={
            "observations": 2,
            "counts": {key: _INPUT_WITNESS[key] + final[key] for key in WITNESS_FIELDS},
        })
    return _ORIGINAL_OUTCOME(pipe, value)


def observed_invocation(*args, **kwargs):
    global _RANKS, _INVOCATIONS
    reports, totals = _ORIGINAL_INVOCATION(*args, **kwargs)
    need(set(reports) == {0, 1}, "witness_sequence_rejected")
    for rank in (0, 1):
        proof = reports[rank].pop("oci_origin_witness", None)
        need(type(proof) is dict and set(proof) == {"observations", "counts"}
             and type(proof["observations"]) is int and proof["observations"] == 2
             and type(proof["counts"]) is dict and set(proof["counts"]) == set(WITNESS_FIELDS),
             "witness_sequence_rejected")
        for key in WITNESS_FIELDS:
            value = proof["counts"][key]
            need(type(value) is int and 0 <= value <= 2 * MAXIMUM_MODULES,
                 "witness_sequence_rejected")
            _COUNTS[key] += value
        _RANKS += 1
    _INVOCATIONS += 1
    need(_RANKS <= 6 and _INVOCATIONS <= 3, "witness_sequence_rejected")
    return reports, totals


def prepare():
    global _STATE, _FIXTURE, _ORIGINAL_OUTCOME, _ORIGINAL_INVOCATION, _STAGE
    runtime_guard()
    _STAGE = "controls"
    _STATE = load_controls()
    _STAGE = "fixture_binding"
    path = SOURCE_ROOT + "/" + FROZEN_RELATIVE
    need(path in _STATE["source"]
         and _STATE["source"][path]["bytes"] == FROZEN_SIZE
         and _STATE["source"][path]["sha256"] == FROZEN_SHA256, "fixture_binding_rejected")
    raw = pinned_bytes(path, FROZEN_SHA256, FROZEN_SIZE, "fixture_binding_rejected")
    need(len(raw) == FROZEN_SIZE and FIXTURE_NAME not in sys.modules, "fixture_binding_rejected")
    spec = importlib.util.spec_from_file_location(FIXTURE_NAME, path)
    need(spec is not None, "fixture_binding_rejected")
    _FIXTURE = importlib.util.module_from_spec(spec)
    sys.modules[FIXTURE_NAME] = _FIXTURE
    exec(compile(raw, path, "exec"), _FIXTURE.__dict__)
    _ORIGINAL_OUTCOME, _ORIGINAL_INVOCATION = _FIXTURE.outcome, _FIXTURE.run_invocation
    _FIXTURE.outcome, _FIXTURE.run_invocation = observed_outcome, observed_invocation


def run():
    global _STAGE
    need(sys.argv[1:] == ["--synthetic-only"], "runtime_rejected")
    _STAGE = "worker_run"
    report = _FIXTURE.run()
    need(report.get("status") == "passed" and _RANKS == 6 and _INVOCATIONS == 3
         and "torch" not in sys.modules and "transformers" not in sys.modules,
         "fixture_failed")
    _STAGE = "complete"
    return {
        "schema": SCHEMA, "status": "passed", "worker_run": report,
        "rank_workers_witnessed": _RANKS, "origin_observations": 2 * _RANKS,
        "invocations_witnessed": _INVOCATIONS, "origin_counts": dict(_COUNTS),
        "software_lock_sha256": _STATE["software_lock_sha256"],
        "base_inventory_sha256": _STATE["base_inventory_sha256"],
        "source_inventory_sha256": _STATE["source_inventory_sha256"],
        "source_commit": _STATE["source_commit"],
        "inputs_barrier_and_final_origins_verified": True,
        "declared_distribution_versions_verified": True,
        "file_backed_executable_origins_verified": True,
        "parent_ml_import_absent": True, **dict.fromkeys(NONCLAIMS, False),
    }


def failure_stages():
    fixture_stage = getattr(_FIXTURE, "STAGE", None)
    child_stage = getattr(_FIXTURE, "CHILD_STAGE", None)
    child_codes = getattr(_FIXTURE, "CHILD_STAGES", ())
    return {
        "probe_stage": _STAGE if _STAGE in PROBE_STAGES else "not_reported",
        "fixture_stage": fixture_stage if fixture_stage in FIXTURE_STAGES else "not_reported",
        "child_stage": child_stage if child_stage in child_codes else "not_reported",
    }


def entry():
    # Install the same hooks on spawn re-import, before the target is unpickled.
    # All native output and underlying exceptions remain suppressed, including setup.
    output = os.dup(1) if __name__ == "__main__" else None
    sink = os.open(os.devnull, os.O_WRONLY | os.O_CLOEXEC)
    try:
        os.dup2(sink, 1)
        os.dup2(sink, 2)
    finally:
        os.close(sink)
    code = 0
    try:
        prepare()
        if output is None:
            return  # Spawn now resolves the original frozen rank_worker function.
        report = run()
    except BaseException as error:
        code = 1
        report = {"schema": SCHEMA, "status": "failed",
                  "code": error.code if type(error) is ProbeError else "probe_failed",
                  **failure_stages(),
                  **dict.fromkeys(NONCLAIMS, False)}
        if output is None:
            os._exit(2)  # No retry or further work after failed spawned initialization.
    if output is not None:
        raw = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
        need(len(raw) <= MAXIMUM_REPORT_BYTES and os.write(output, raw) == len(raw),
             "output_failed")
        os.close(output)
        raise SystemExit(code)


if __name__ in ("__main__", "__mp_main__"):
    entry()
