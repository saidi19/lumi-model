#!/usr/bin/env python3
"""One approved remote CPU origin diagnostic; NEVER import/run on the Mac.

The frozen software builder is reused only for snapshot/image construction.
Neither its acceptance method nor archive path is called. Diagnostic completion
does not accept the underlying release and never authorizes recovery or SFT.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import re
import signal
import sys
import tempfile
import time
from pathlib import Path

SCHEMA = "lumi.sft-cpu-origin-diagnostic-build.v0.1"
APPROVED_PARENT = "edaa86e238223cf3a1ecfb6292640d8bc43e72de"
FROZEN_BUILDER = "scripts/build_sft_oci_release_candidate_v1.py"
FROZEN_BUILDER_SHA256 = "458faf97c62689c9d7cc71a6eb89a4f84fef8c68bbcf9279b949dee450565044"
FROZEN_PROBE = "scripts/run_sft_oci_worker_probe_v1.py"
FROZEN_PROBE_SHA256 = "bed636e8cc08546a6be1b43c00c2f516fdc92dbf907dca674f7fb5aa7b5cb769"
FROZEN_FIXTURE = "scripts/run_sft_worker_run_torch_fixture_v1.py"
FROZEN_FIXTURE_SHA256 = "53c50c92965ed1cef0c9624c57367c31ed7ebfb9d7cfb34257188ad179d333d0"
FROZEN_BASE = "scripts/build_sft_oci_base_inventory_v1.py"
FROZEN_BASE_SHA256 = "34dc6a70b6d074808b864f53b4d511456423786aa62068ea771253fd4f8f29d1"
LOCK_MODULE = "src/lumi/posttraining_sft_oci_lock_v1.py"
LOCK_MODULE_SHA256 = "9f70737db527c038d2ddd54992ce6d1ff999e9c7f204f4828a3f64fa08543ed3"
DIAGNOSTIC = "scripts/run_sft_cpu_origin_diagnostic_v1.py"
DIAGNOSTIC_SHA256 = "492c663e8c5cd0ae01d0fa44919ec46472dfb63b5f5aac74a6b1e3692d8909d0"
VALIDATOR = "scripts/sft_cpu_origin_diagnostic_report_v1.py"
VALIDATOR_SHA256 = "d1ec96cd14636eb87e75c4156c7db40891f26d8764de183247434ccacfb0facf"
IMAGE_DIAGNOSTIC = "/opt/lumi-release/source/" + DIAGNOSTIC
MAXIMUM_WORK_SECONDS = 840
MAXIMUM_RETAINED_BYTES = 33554432
MAXIMUM_SOURCE_SCRIPT_BYTES = 131072
MAXIMUM_DIAGNOSTIC_BYTES = 16384
NEW_FILES = (
    ".github/workflows/sft-cpu-origin-diagnostic.yml",
    "docs/sft-execution-status-v0.51.md",
    "scripts/build_sft_cpu_origin_diagnostic_v1.py",
    "scripts/run_sft_cpu_origin_diagnostic_v1.py",
    "scripts/sft_cpu_origin_diagnostic_report_v1.py",
    "tests/test_sft_cpu_diagnostic_builder_source_v1.py",
    "tests/test_sft_cpu_origin_diagnostic_report_v1.py",
    "tests/test_sft_cpu_origin_diagnostic_source_v1.py",
)
FALSE_FLAGS = (
    "acceptance_passed",
    "recovery_performed",
    "production_eligible",
    "real_sft_execution_permitted",
    "real_model_or_data_access",
    "gpu_used",
    "registry_published",
    "image_archived",
    "real_host_admission_verified",
    "complete_dynamic_dependency_closure_verified",
    "full_ci_passed",
    "private_integration_passed",
)


class DiagnosticBuildError(ValueError):
    pass


def need(condition):
    if not condition:
        raise DiagnosticBuildError("diagnostic_contract_rejected")


def canonical(value):
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii")
        + b"\n"
    )


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pinned_source(root, relative, expected):
    need(re.fullmatch(r"[0-9a-f]{64}", expected))
    path = root / relative
    need(not path.is_symlink())
    with path.open("rb") as source:
        raw = source.read(MAXIMUM_SOURCE_SCRIPT_BYTES + 1)
    need(0 < len(raw) <= MAXIMUM_SOURCE_SCRIPT_BYTES and digest(raw) == expected)
    return raw


def load_pinned(root, relative, expected, name):
    raw = pinned_source(root, relative, expected)
    path = root / relative
    need(name not in sys.modules)
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def record(output, name, value):
    raw = canonical(value)
    need(len(raw) <= MAXIMUM_DIAGNOSTIC_BYTES)
    with (output / name).open("xb") as target:
        target.write(raw)


def run_diagnostic(builder, frozen, image_id, controls, validator, report_spec):
    builder.stage = "diagnostic_container_inspection"
    command = ["docker", "create", *frozen.CONTAINER_OPTIONS, "--shm-size=67108864"]
    for path, options in frozen.TMPFS.items():
        command.extend(("--tmpfs", path + ":" + options))
    command.append(image_id)
    container = builder.run(command).decode("ascii").strip()
    need(re.fullmatch(r"[0-9a-f]{64}", container))
    builder.container = container
    info = json.loads(builder.run(["docker", "inspect", container]))[0]
    host, config = info["HostConfig"], info["Config"]
    need(info["Image"] == image_id and info["Path"] == frozen.PYTHON)
    need(info["Args"] == ["-I", "-B", IMAGE_DIAGNOSTIC, "--synthetic-only"])
    need(config["User"] == "10001:10001" and not config.get("Volumes"))
    need(host["ReadonlyRootfs"] is True and host["Privileged"] is False)
    need(host["NetworkMode"] == "none" and host["CapDrop"] == ["ALL"])
    need(set(host["SecurityOpt"]) & {"no-new-privileges", "no-new-privileges:true"})
    need(not host.get("Binds") and not host.get("Devices") and not host.get("DeviceRequests"))
    need(host["Tmpfs"] == frozen.TMPFS and host["Memory"] == 4294967296)
    need(host["MemorySwap"] == 4294967296 and host["ShmSize"] == 67108864)
    need(host["PidsLimit"] == 256 and host["NanoCpus"] == 2000000000)
    need(
        all(
            mount["Type"] == "tmpfs"
            and mount["Destination"] in frozen.TMPFS
            and not mount.get("Source")
            for mount in info["Mounts"]
        )
    )
    builder.stage = "diagnostic_started"
    # Exclusive local marker precedes the sole diagnostic-container start.
    record(
        builder.output,
        "diagnostic-start-marker.json",
        {
            "schema": SCHEMA,
            "stage": "diagnostic_started",
            "image_id": image_id,
            "controls": controls,
            "diagnostic_starts": 1,
            **dict.fromkeys(FALSE_FLAGS, False),
        },
    )
    raw = builder.run(
        ["docker", "start", "--attach", container], timeout=240, cap=MAXIMUM_DIAGNOSTIC_BYTES
    )
    need(len(raw) <= MAXIMUM_DIAGNOSTIC_BYTES)
    report = validator.validate_report(raw, controls, report_spec)
    state = json.loads(builder.run(["docker", "inspect", container]))[0]["State"]
    need(state["Running"] is False and state["ExitCode"] == 0 and state["OOMKilled"] is False)
    need(not builder.run(["docker", "diff", container]).strip())
    with (builder.output / "origin-diagnostic.json").open("xb") as target:
        target.write(raw)
    builder.run(["docker", "rm", container])
    builder.container = None
    return report, digest(raw)


def terminate(_signum, _frame):
    raise DiagnosticBuildError("terminated")


def main():
    # No model/runtime/native work or output writes before these guards.
    need(sys.argv[1:] == ["--cpu-origin-diagnostic"])
    need(sys.platform == "linux" and platform.machine() == "x86_64")
    need(sys.version_info[:2] == (3, 12) and sys.flags.dont_write_bytecode)
    need(os.environ.get("GITHUB_ACTIONS") == "true")
    need(os.environ.get("LUMI_CPU_ORIGIN_DIAGNOSTIC_CI") == "1")
    need(os.environ.get("GITHUB_REPOSITORY") == "saidi19/lumi-model")
    need(os.environ.get("GITHUB_RUN_ATTEMPT") == "1")
    commit = os.environ.get("GITHUB_SHA", "")
    need(re.fullmatch(r"[0-9a-f]{40}", commit))
    root = Path(__file__).resolve().parents[1]
    frozen = load_pinned(
        root, FROZEN_BUILDER, FROZEN_BUILDER_SHA256, "_lumi_frozen_cpu_diagnostic_builder"
    )
    validator = load_pinned(root, VALIDATOR, VALIDATOR_SHA256, "_lumi_cpu_diagnostic_validator")
    lock_module = load_pinned(root, LOCK_MODULE, LOCK_MODULE_SHA256, "_lumi_cpu_diagnostic_lock")
    for relative, expected in (
        (FROZEN_PROBE, FROZEN_PROBE_SHA256),
        (FROZEN_FIXTURE, FROZEN_FIXTURE_SHA256),
        (FROZEN_BASE, FROZEN_BASE_SHA256),
    ):
        pinned_source(root, relative, expected)
    diagnostic_source = pinned_source(root, DIAGNOSTIC, DIAGNOSTIC_SHA256)
    report_spec = validator.spec_from_source(diagnostic_source, DIAGNOSTIC_SHA256)
    # These two versioned bindings are the only frozen-builder global changes.
    frozen.PROBE = IMAGE_DIAGNOSTIC
    frozen.SOURCE_SCRIPTS = (FROZEN_FIXTURE, FROZEN_PROBE, DIAGNOSTIC)
    output = root / "artifacts/lumi-sft-cpu-origin-diagnostic-v0.1"
    need(not output.exists())
    output.mkdir(parents=True)
    started = time.monotonic()
    builder = frozen.Builder(root, output, started + MAXIMUM_WORK_SECONDS)
    signal.signal(signal.SIGTERM, terminate)
    summary = {
        "schema": SCHEMA,
        "status": "diagnostic_incomplete",
        "source_commit": commit,
        "underlying_status": "not_reported",
        **dict.fromkeys(FALSE_FLAGS, False),
    }
    try:
        need(builder.run(["git", "rev-parse", "HEAD^"]).decode().strip() == APPROVED_PARENT)
        changes = (
            builder.run(["git", "diff", "--name-status", APPROVED_PARENT, commit], cap=16384)
            .decode("ascii")
            .splitlines()
        )
        need(changes == ["A\t" + path for path in NEW_FILES])
        lock_raw = (root / frozen.LOCK_PATH).read_bytes()
        lock = lock_module.validate_lock(lock_raw, frozen.LOCK_SHA256)
        with tempfile.TemporaryDirectory(
            prefix="lumi-cpu-origin-diagnostic-",
            dir=os.environ["RUNNER_TEMP"],
        ) as temporary:
            work = Path(temporary)
            source_inventory = builder.snapshot(work, commit)
            image_id, controls = builder.image(work, lock, commit, lock_raw, source_inventory)
            need(builder.download_responses == 34 and builder.download_bytes == 248715637)
            summary.update(
                {
                    "image_id": image_id,
                    "controls": controls,
                    "wheel_count": builder.download_responses,
                    "wheel_download_bytes": builder.download_bytes,
                    "diagnostic_source_sha256": DIAGNOSTIC_SHA256,
                    "frozen_builder_sha256": FROZEN_BUILDER_SHA256,
                }
            )
            # Preserve successful software binding before any diagnostic execution.
            record(output, "software-binding-before-diagnostic.json", summary)
            report, report_hash = run_diagnostic(
                builder,
                frozen,
                image_id,
                controls,
                validator,
                report_spec,
            )
            retained = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
            need(retained <= MAXIMUM_RETAINED_BYTES)
            summary.update(
                {
                    "status": "diagnostic_completed",
                    "underlying_status": report["underlying_status"],
                    "diagnostic_report_sha256": report_hash,
                    "diagnostic_starts": 1,
                    "diagnostic": report["diagnostic"],
                    "frozen_stages": report["frozen_stages"],
                    "owned_container_retired": True,
                }
            )
        builder.stage = "diagnostic_complete"
    except Exception as error:
        summary["status"] = "diagnostic_incomplete"
        summary["failure_class"] = (
            type(error).__name__
            if type(error)
            in (
                DiagnosticBuildError,
                ValueError,
                KeyError,
                OSError,
                TimeoutError,
            )
            else "operation_failed"
        )
        summary["validation_code"] = (
            error.code
            if type(error) is validator.ValidationError and error.code in validator.ERROR_CODES
            else "not_reported"
        )
    finally:
        if builder.container is not None:
            try:
                builder.run(
                    ["docker", "rm", "--force", builder.container], timeout=15, retirement=True
                )
                summary["owned_container_retired"] = True
            except Exception:
                summary["owned_container_retired"] = False
        summary["stage"] = builder.stage
        summary["runtime_seconds"] = int(time.monotonic() - started)
        record(output, "diagnostic-build-report.json", summary)
        print(canonical(summary).decode("ascii"), end="")
    return 0 if summary["status"] == "diagnostic_completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
