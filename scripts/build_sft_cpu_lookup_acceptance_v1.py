#!/usr/bin/env python3
"""One corrected CPU acceptance on private GitHub Linux; NEVER run/import on Mac.

Preserve all prior sources/evidence. Reuse only frozen software construction.
One inspected CPU container emits both fixed diagnostics and its full observer
proof. Archive only after full acceptance; no retry, GPU or real model/data.
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

SCHEMA = "lumi.sft-cpu-lookup-acceptance-build.v0.1"
APPROVED_PARENT = "15127b62e702244923a6f56c1a936fb667ea4486"
FROZEN_BUILDER = "scripts/build_sft_oci_release_candidate_v1.py"
FROZEN_BUILDER_SHA256 = "458faf97c62689c9d7cc71a6eb89a4f84fef8c68bbcf9279b949dee450565044"
OLD_PROBE = "scripts/run_sft_oci_worker_probe_v1.py"
OLD_PROBE_SHA256 = "bed636e8cc08546a6be1b43c00c2f516fdc92dbf907dca674f7fb5aa7b5cb769"
PROBE = "scripts/run_sft_oci_worker_probe_v2.py"
PROBE_SHA256 = "b45b04fc9cb93dd2301df0444f2db95666ba4f7a32e8f4d8e28c5b4077314450"
FIXTURE = "scripts/run_sft_worker_run_torch_fixture_v1.py"
FIXTURE_SHA256 = "53c50c92965ed1cef0c9624c57367c31ed7ebfb9d7cfb34257188ad179d333d0"
BASE = "scripts/build_sft_oci_base_inventory_v1.py"
BASE_SHA256 = "34dc6a70b6d074808b864f53b4d511456423786aa62068ea771253fd4f8f29d1"
LOCK_MODULE = "src/lumi/posttraining_sft_oci_lock_v1.py"
LOCK_MODULE_SHA256 = "9f70737db527c038d2ddd54992ce6d1ff999e9c7f204f4828a3f64fa08543ed3"
WRAPPER = "scripts/run_sft_cpu_lookup_acceptance_v1.py"
WRAPPER_SHA256 = "6e453f0a8ab5dc84f126db1761a7c089e0a239cf588a20cdc56b6d5236af0cf4"
DIAGNOSTIC_VALIDATOR = "scripts/sft_cpu_origin_diagnostic_report_v1.py"
DIAGNOSTIC_VALIDATOR_SHA256 = "d1ec96cd14636eb87e75c4156c7db40891f26d8764de183247434ccacfb0facf"
ACCEPTANCE_VALIDATOR = "scripts/sft_cpu_lookup_acceptance_report_v1.py"
ACCEPTANCE_VALIDATOR_SHA256 = "8c0cc1727e3d6b320586a6cf664bfc0648ed3e84930b2b9d4983683c14a36260"
IMAGE_WRAPPER = "/opt/lumi-release/source/" + WRAPPER
MAXIMUM_WORK_SECONDS = 1140
MAXIMUM_REPORT_BYTES = 32768
MAXIMUM_SOURCE_SCRIPT_BYTES = 131072
NEW_FILES = (
    ".github/workflows/sft-cpu-lookup-acceptance.yml",
    "docs/sft-execution-status-v0.55.md",
    "scripts/build_sft_cpu_lookup_acceptance_v1.py",
    "scripts/run_sft_cpu_lookup_acceptance_v1.py",
    "scripts/run_sft_oci_worker_probe_v2.py",
    "scripts/sft_cpu_lookup_acceptance_report_v1.py",
    "tests/test_sft_cpu_lookup_acceptance_report_v1.py",
    "tests/test_sft_cpu_lookup_acceptance_source_v1.py",
    "tests/test_sft_cpu_version_lookup_v2.py",
)
FALSE_CLAIMS = (
    "production_eligible",
    "real_sft_execution_permitted",
    "real_model_or_data_access",
    "gpu_used",
    "registry_published",
    "real_host_admission_verified",
    "complete_dynamic_dependency_closure_verified",
    "publisher_signature_verified",
    "reproducible_build_verified",
    "private_integration_passed",
    "full_ci_passed",
)


class AcceptanceBuildError(ValueError):
    pass


def need(condition):
    if not condition:
        raise AcceptanceBuildError("acceptance_contract_rejected")


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
    need(name not in sys.modules)
    spec = importlib.util.spec_from_file_location(name, root / relative)
    need(spec is not None and spec.loader is not None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(raw, str(root / relative), "exec"), module.__dict__)
    return module


def record(output, name, value):
    raw = canonical(value)
    need(len(raw) <= MAXIMUM_REPORT_BYTES)
    with (output / name).open("xb") as target:
        target.write(raw)
    return digest(raw)


def run_acceptance(builder, frozen, image_id, controls, validator, diagnostic, spec, summary):
    builder.stage = "acceptance_container_inspection"
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
    need(info["Args"] == ["-I", "-B", IMAGE_WRAPPER, "--synthetic-only"])
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
    builder.stage = "cpu_acceptance_started"
    record(
        builder.output,
        "acceptance-start-marker.json",
        {
            "schema": SCHEMA,
            "image_id": image_id,
            "controls": controls,
            "cpu_starts": 1,
            "same_image_cpu_acceptance_passed": False,
            **dict.fromkeys(FALSE_CLAIMS, False),
        },
    )
    summary["cpu_starts"] = 1
    raw = builder.run(
        ["docker", "start", "--attach", container], timeout=240, cap=MAXIMUM_REPORT_BYTES
    )
    value = validator.validate_report(raw, controls, diagnostic, spec)
    state = json.loads(builder.run(["docker", "inspect", container]))[0]["State"]
    need(state["Running"] is False and state["ExitCode"] == 0 and state["OOMKilled"] is False)
    need(not builder.run(["docker", "diff", container]).strip())
    with (builder.output / "cpu-acceptance-envelope.json").open("xb") as target:
        target.write(raw)
    summary["envelope_sha256"] = digest(raw)
    summary["diagnostic_report_sha256"] = record(
        builder.output,
        "origin-diagnostic.json",
        value["diagnostic"],
    )
    if value["observer_report"] is not None:
        summary["probe_report_sha256"] = record(
            builder.output,
            "cpu-worker-probe.json",
            value["observer_report"],
        )
    builder.run(["docker", "rm", container])
    builder.container = None
    summary["owned_container_retired"] = True
    return value


def terminate(_signum, _frame):
    raise AcceptanceBuildError("terminated")


def main():
    # Reject before native tooling, temporary writes or package/source execution.
    need(sys.argv[1:] == ["--cpu-lookup-acceptance"])
    need(sys.platform == "linux" and platform.machine() == "x86_64")
    need(sys.version_info[:2] == (3, 12) and sys.flags.dont_write_bytecode)
    need(os.environ.get("GITHUB_ACTIONS") == "true")
    need(os.environ.get("LUMI_CPU_LOOKUP_ACCEPTANCE_CI") == "1")
    need(os.environ.get("GITHUB_REPOSITORY") == "saidi19/lumi-model")
    need(os.environ.get("GITHUB_RUN_ATTEMPT") == "1")
    commit = os.environ.get("GITHUB_SHA", "")
    need(re.fullmatch(r"[0-9a-f]{40}", commit))
    root = Path(__file__).resolve().parents[1]
    frozen = load_pinned(root, FROZEN_BUILDER, FROZEN_BUILDER_SHA256, "_lumi_lookup_frozen_builder")
    diagnostic = load_pinned(
        root, DIAGNOSTIC_VALIDATOR, DIAGNOSTIC_VALIDATOR_SHA256, "_lumi_lookup_diagnostic_validator"
    )
    validator = load_pinned(
        root, ACCEPTANCE_VALIDATOR, ACCEPTANCE_VALIDATOR_SHA256, "_lumi_lookup_acceptance_validator"
    )
    lock_module = load_pinned(root, LOCK_MODULE, LOCK_MODULE_SHA256, "_lumi_lookup_lock")
    old = pinned_source(root, OLD_PROBE, OLD_PROBE_SHA256)
    corrected = pinned_source(root, PROBE, PROBE_SHA256)
    old_lookup = b'str(vars(sys.modules[name]).get("__version__"))'
    new_lookup = b'str(getattr(sys.modules[name], "__version__", None))'
    need(old.count(old_lookup) == 1 and corrected == old.replace(old_lookup, new_lookup, 1))
    for relative, expected in ((FIXTURE, FIXTURE_SHA256), (BASE, BASE_SHA256)):
        pinned_source(root, relative, expected)
    wrapper = pinned_source(root, WRAPPER, WRAPPER_SHA256)
    spec = diagnostic.spec_from_source(wrapper, WRAPPER_SHA256)
    frozen.PROBE = IMAGE_WRAPPER
    frozen.SOURCE_SCRIPTS = (FIXTURE, PROBE, WRAPPER)
    output = root / "artifacts/lumi-sft-cpu-lookup-acceptance-v0.1"
    need(not output.exists())
    output.mkdir(parents=True)
    started = time.monotonic()
    builder = frozen.Builder(root, output, started + MAXIMUM_WORK_SECONDS)
    signal.signal(signal.SIGTERM, terminate)
    summary = {
        "schema": SCHEMA,
        "status": "failed",
        "source_commit": commit,
        "cpu_starts": 0,
        "same_image_cpu_acceptance_passed": False,
        "image_archived": False,
        **dict.fromkeys(FALSE_CLAIMS, False),
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
            prefix="lumi-cpu-lookup-acceptance-", dir=os.environ["RUNNER_TEMP"]
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
                    "corrected_probe_sha256": PROBE_SHA256,
                    "wrapper_sha256": WRAPPER_SHA256,
                }
            )
            record(output, "software-binding-before-acceptance.json", summary)
            result = run_acceptance(
                builder, frozen, image_id, controls, validator, diagnostic, spec, summary
            )
            summary["underlying_status"] = result["diagnostic"]["underlying_status"]
            summary["diagnostic"] = result["diagnostic"]["diagnostic"]
            summary["frozen_stages"] = result["diagnostic"]["frozen_stages"]
            need(result["diagnostic"]["underlying_status"] == "passed")
            need(result["observer_report"] is not None)
            summary["same_image_cpu_acceptance_passed"] = True
            summary["read_only_nonroot_no_network_verified"] = True
            summary["no_host_software_or_source_binds_verified"] = True
            record(output, "cpu-acceptance-before-archive.json", summary)
            builder.stage = "archive"
            archive = output / "cpu-release-image.tar"
            builder.run(["docker", "image", "save", image_id], timeout=180, stdout_path=archive)
            image_bytes, image_sha256 = frozen.hashed_file(archive, frozen.MAXIMUM_IMAGE_BYTES)
            retained = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
            need(retained + MAXIMUM_REPORT_BYTES <= frozen.MAXIMUM_RETAINED_BYTES)
            summary.update(
                {
                    "status": "passed",
                    "image_archived": True,
                    "archive_sha256": image_sha256,
                    "archive_bytes": image_bytes,
                }
            )
        builder.stage = "complete"
    except Exception as error:
        summary["status"] = "failed"
        summary["failure_class"] = (
            type(error).__name__
            if type(error)
            in (
                AcceptanceBuildError,
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
        record(output, "acceptance-build-report.json", summary)
        print(canonical(summary).decode("ascii"), end="")
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
