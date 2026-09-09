#!/usr/bin/env python3
"""Build one private CPU software image on GitHub Linux; NEVER run on the Mac.

No source distributions, datasets, real checkpoints, GPU, registry publication,
or host admission. Downloads are software wheels only. Native acceptance runs
once from the inspected image ID with no network or host software/source binds.
The image archive is an acceptance candidate, not a reproducible-build claim.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

SCHEMA = "lumi.sft-oci-release-candidate.v0.1"
LOCK_PATH = "configs/lumi-sft-linux-cpu-release-v0.1.lock.json"
LOCK_SHA256 = "26bde6a0163e5db9d8d4564117546482601f4d0beb09f8705be5fc9183594965"
MAXIMUM_SECONDS = 1140
MAXIMUM_COMMAND_OUTPUT = 16777216
MAXIMUM_SOURCE_BYTES = 33554432
MAXIMUM_SOURCE_FILES = 4096
MAXIMUM_IMAGE_BYTES = 3221225472
MAXIMUM_RETAINED_BYTES = 4294967296
PYTHON = "/usr/local/bin/python3.12"
PROBE = "/opt/lumi-release/source/scripts/run_sft_oci_worker_probe_v1.py"
SOURCE_SCRIPTS = (
    "scripts/run_sft_worker_run_torch_fixture_v1.py",
    "scripts/run_sft_oci_worker_probe_v1.py",
)
CONTAINER_OPTIONS = (
    "--platform=linux/amd64", "--network=none", "--read-only", "--user=10001:10001",
    "--cap-drop=ALL", "--security-opt=no-new-privileges", "--cpus=2", "--memory=4g",
    "--memory-swap=4g", "--pids-limit=256", "--ulimit=core=0", "--ulimit=nofile=1024:1024",
)
TMPFS = {
    "/tmp": "rw,noexec,nosuid,nodev,size=536870912,mode=1777",
}
STAGES = (
    "preflight", "source_snapshot", "base_pull", "base_inventory", "wheel_acquisition",
    "offline_build", "container_inspection", "cpu_acceptance", "archive", "complete",
)
FALSE_CLAIMS = (
    "production_eligible", "real_sft_execution_permitted", "real_model_or_data_access",
    "gpu_used", "registry_published", "real_host_admission_verified",
    "complete_dynamic_dependency_closure_verified", "publisher_signature_verified",
    "reproducible_build_verified", "private_integration_passed", "full_ci_passed",
)


class BuildError(ValueError):
    pass


def need(condition):
    if not condition:
        raise BuildError("contract_rejected")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def hashed_file(path, maximum):
    total, hasher = 0, hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1048576):
            total += len(chunk)
            need(total <= maximum)
            hasher.update(chunk)
    return total, hasher.hexdigest()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise BuildError("redirect_rejected")


class Builder:
    def __init__(self, root, output, deadline):
        self.root, self.output, self.deadline = root, output, deadline
        self.stage = "preflight"
        self.container = None
        self.command_number = 0
        self.download_bytes = 0
        self.download_responses = 0

    def run(self, command, *, timeout=120, stdin=None, cap=MAXIMUM_COMMAND_OUTPUT,
            stdout_path=None, retirement=False):
        """Bound time/output, no shell, no retries; retain software-only diagnostics."""
        if retirement:
            need(self.container is not None)
            need(command == ["docker", "rm", "--force", self.container])
            remaining = 15
        else:
            remaining = self.deadline - time.monotonic()
        need(remaining > 0)
        limit = time.monotonic() + min(timeout, remaining)
        self.command_number += 1
        proc = subprocess.Popen(
            command, cwd=self.root,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
            env=dict(os.environ, DOCKER_BUILDKIT="1"),
        )
        stdout, stderr = bytearray(), bytearray()
        archive = None
        archive_bytes = 0
        try:
            if stdout_path is not None:
                archive = stdout_path.open("xb")
            if stdin is not None:
                proc.stdin.write(stdin)
                proc.stdin.close()
            with selectors.DefaultSelector() as selector:
                selector.register(proc.stdout, selectors.EVENT_READ, stdout)
                selector.register(proc.stderr, selectors.EVENT_READ, stderr)
                while selector.get_map():
                    need(time.monotonic() < limit)
                    for key, _ in selector.select(min(0.2, max(0, limit - time.monotonic()))):
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                        else:
                            if archive is not None and key.data is stdout:
                                archive_bytes += len(chunk)
                                need(archive_bytes <= MAXIMUM_IMAGE_BYTES)
                                archive.write(chunk)
                            else:
                                key.data.extend(chunk)
                                need(len(stdout) + len(stderr) <= cap)
            need(proc.wait(timeout=max(0.1, limit - time.monotonic())) == 0)
            return bytes(stdout)
        except BaseException:
            # Exact child process group, created above; no broad process matching.
            if proc.poll() is None:
                os.killpg(proc.pid, 9)
                proc.wait(timeout=10)
            log = bytes(stdout + stderr)[-65536:]
            log_path = self.output / f"software-command-{self.command_number:04d}-failure.log"
            with log_path.open("xb") as diagnostic:
                diagnostic.write(log)
            raise
        finally:
            proc.stdout.close()
            proc.stderr.close()
            if archive is not None:
                archive.close()

    def snapshot(self, work, commit):
        self.stage = "source_snapshot"
        need(self.run(["git", "rev-parse", "HEAD"]).decode().strip() == commit)
        need(not self.run(["git", "status", "--porcelain", "--untracked-files=no"]).strip())
        source = work / "source"
        source.mkdir()
        raw = self.run(["git", "ls-tree", "-r", "-z", commit, "--", "src/lumi", *SOURCE_SCRIPTS])
        entries, total = [], 0
        for item in raw.split(b"\0"):
            if not item:
                continue
            header, encoded = item.split(b"\t", 1)
            mode, kind, blob = header.decode("ascii").split()
            relative = encoded.decode("ascii")
            need(mode in ("100644", "100755") and kind == "blob")
            need(relative.startswith("src/lumi/") or relative in SOURCE_SCRIPTS)
            need(".." not in relative.split("/") and "\\" not in relative)
            payload = self.run(["git", "cat-file", "blob", blob], cap=2097152)
            total += len(payload)
            need(total <= MAXIMUM_SOURCE_BYTES and len(entries) < MAXIMUM_SOURCE_FILES)
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            entries.append({"path": relative, "bytes": len(payload), "sha256": digest(payload)})
        need(len(entries) > 10 and set(SOURCE_SCRIPTS) <= {x["path"] for x in entries})
        return canonical({
            "schema": "lumi.sft-oci-source-inventory.v0.1", "source_commit": commit,
            "files": sorted(entries, key=lambda x: x["path"]),
        })

    def wheels(self, work, lock):
        self.stage = "wheel_acquisition"
        directory = work / "wheels"
        directory.mkdir()
        opener = urllib.request.build_opener(NoRedirect())
        for row in lock["requirements"]:
            need(time.monotonic() < self.deadline)
            request = urllib.request.Request(row["url"], headers={"Accept-Encoding": "identity"})
            self.download_responses += 1
            with opener.open(request, timeout=30) as response:
                need(response.status == 200)
                need(response.headers.get("Content-Encoding") in (None, "identity"))
                size = response.headers.get("Content-Length")
                need(size is None or int(size) == row["bytes"])
                hasher, total = hashlib.sha256(), 0
                with (directory / row["filename"]).open("xb") as target:
                    while chunk := response.read(min(1048576, row["bytes"] - total + 1)):
                        total += len(chunk)
                        self.download_bytes += len(chunk)
                        need(total <= row["bytes"] and self.download_bytes <= 1073741824)
                        need(time.monotonic() < self.deadline)
                        target.write(chunk)
                        hasher.update(chunk)
                need(total == row["bytes"] and hasher.hexdigest() == row["sha256"])
        need(self.download_responses == len(lock["requirements"]))

    def image(self, work, lock, commit, lock_raw, source_inventory):
        base = lock["base"]
        reference = base["repository"] + "@" + base["digest"]
        self.stage = "base_pull"
        self.run(["docker", "pull", "--platform=linux/amd64", reference], timeout=180)
        inspected = json.loads(self.run(["docker", "image", "inspect", reference]))
        need(len(inspected) == 1)
        value = inspected[0]
        need(value["Id"] == base["config_digest"] and value["Architecture"] == "amd64")
        need(value["Os"] == "linux")
        need("PYTHON_VERSION=" + base["python_version"] in value["Config"]["Env"])
        self.stage = "base_inventory"
        inventory_source = self.run([
            "git", "show", commit + ":scripts/build_sft_oci_base_inventory_v1.py",
        ], cap=65536).decode("ascii")
        base_container = self.run([
            "docker", "create", "-i", *CONTAINER_OPTIONS,
            "-e", "LUMI_OCI_BASE_INVENTORY_CI=1", "--entrypoint", PYTHON,
            reference, "-I", "-B", "-c", inventory_source, "--base-inventory-only",
        ]).decode("ascii").strip()
        need(re.fullmatch(r"[0-9a-f]{64}", base_container))
        self.container = base_container
        base_inventory = self.run([
            "docker", "start", "--attach", "--interactive", base_container,
        ], stdin=canonical({"digest": base["digest"], "config_digest": base["config_digest"]}),
            timeout=120)
        base_state = json.loads(self.run(["docker", "inspect", base_container]))[0]["State"]
        need(base_state["Running"] is False and base_state["ExitCode"] == 0)
        self.run(["docker", "rm", base_container])
        self.container = None
        base_doc = json.loads(base_inventory)
        need(base_doc["schema"] == "lumi.sft-oci-base-inventory.v0.1")
        need(base_doc["base_digest"] == base["digest"])
        need(base_doc["config_digest"] == base["config_digest"])
        control = work / "control"
        control.mkdir()
        for name, raw in (
            ("software-lock.json", lock_raw), ("source-inventory.json", source_inventory),
            ("base-system-inventory.json", base_inventory),
        ):
            (control / name).write_bytes(raw)
            (self.output / name).write_bytes(raw)
        requirements = "".join(
            row["name"] + "==" + row["version"] + " --hash=sha256:" + row["sha256"] + "\n"
            for row in lock["requirements"]
        )
        (control / "offline-requirements.txt").write_text(requirements, encoding="ascii")
        self.wheels(work, lock)
        self.stage = "offline_build"
        dockerfile = "\n".join((
            "FROM " + reference,
            "ENV PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1",
            "COPY control /opt/lumi-release/control",
            "RUN --mount=type=bind,source=wheels,target=/wheels,readonly "
            + PYTHON + " -I -B -m pip install --no-cache-dir --no-compile --no-index "
            "--find-links=/wheels --only-binary=:all: --require-hashes --no-deps "
            "-r /opt/lumi-release/control/offline-requirements.txt && "
            + PYTHON + " -I -B -m pip check",
            "COPY source /opt/lumi-release/source",
            "ENV LUMI_RELEASE_LOCK_SHA256=" + LOCK_SHA256,
            "ENV LUMI_RELEASE_SOURCE_COMMIT=" + commit,
            "ENV LUMI_RELEASE_BASE_INVENTORY_SHA256=" + digest(base_inventory),
            "ENV LUMI_RELEASE_SOURCE_INVENTORY_SHA256=" + digest(source_inventory),
            "ENV LUMI_WORKER_RUN_TORCH_CI_FIXTURE=1 CUDA_VISIBLE_DEVICES=\"\" "
            "GLOO_SOCKET_IFNAME=lo",
            "ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1",
            "ENV AWS_EC2_METADATA_DISABLED=true OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
            "ENV HOME=/tmp HF_HOME=/tmp/huggingface XDG_CACHE_HOME=/tmp/cache",
            "WORKDIR /opt/lumi-release/source",
            "USER 10001:10001",
            "ENTRYPOINT [\"" + PYTHON + "\",\"-I\",\"-B\"]",
            "CMD [\"" + PROBE + "\",\"--synthetic-only\"]",
            "",
        ))
        (work / "Dockerfile").write_text(dockerfile, encoding="ascii")
        self.run([
            "docker", "build", "--network=none", "--pull=false", "--no-cache",
            "--platform=linux/amd64", "--iidfile", str(work / "image.id"), str(work),
        ], timeout=360)
        image_id = (work / "image.id").read_text(encoding="ascii").strip()
        need(re.fullmatch(r"sha256:[0-9a-f]{64}", image_id))
        image_info = json.loads(self.run(["docker", "image", "inspect", image_id]))[0]
        need(image_info["Id"] == image_id and image_info["Size"] <= MAXIMUM_IMAGE_BYTES)
        return image_id, {
            "software_lock_sha256": LOCK_SHA256, "source_commit": commit,
            "base_inventory_sha256": digest(base_inventory),
            "source_inventory_sha256": digest(source_inventory),
        }

    def accept(self, image_id, expected_controls):
        self.stage = "container_inspection"
        command = ["docker", "create", *CONTAINER_OPTIONS, "--shm-size=67108864"]
        for path, options in TMPFS.items():
            command.extend(("--tmpfs", path + ":" + options))
        command.append(image_id)
        container = self.run(command).decode("ascii").strip()
        need(re.fullmatch(r"[0-9a-f]{64}", container))
        self.container = container
        info = json.loads(self.run(["docker", "inspect", container]))[0]
        host, config = info["HostConfig"], info["Config"]
        need(info["Image"] == image_id and info["Path"] == PYTHON)
        need(info["Args"] == ["-I", "-B", PROBE, "--synthetic-only"])
        need(config["User"] == "10001:10001" and not config.get("Volumes"))
        need(host["ReadonlyRootfs"] is True and host["Privileged"] is False)
        need(host["NetworkMode"] == "none" and host["CapDrop"] == ["ALL"])
        need(set(host["SecurityOpt"]) & {"no-new-privileges", "no-new-privileges:true"})
        need(not host.get("Binds") and not host.get("Devices") and not host.get("DeviceRequests"))
        need(host["Tmpfs"] == TMPFS and host["Memory"] == 4294967296)
        need(host["PidsLimit"] == 256 and host["NanoCpus"] == 2000000000)
        need(all(mount["Type"] == "tmpfs" and mount["Destination"] in TMPFS
                 and not mount.get("Source") for mount in info["Mounts"]))
        self.stage = "cpu_acceptance"
        raw = self.run(["docker", "start", "--attach", container], timeout=240, cap=65536)
        need(len(raw) <= 8192)
        probe = json.loads(raw)
        need(probe["schema"] == "lumi.sft-oci-worker-probe.v0.1" and probe["status"] == "passed")
        need(all(probe[key] == value for key, value in expected_controls.items()))
        need(probe["rank_workers_witnessed"] == 6 and probe["origin_observations"] == 12
             and probe["invocations_witnessed"] == 3)
        need(probe["inputs_barrier_and_final_origins_verified"] is True
             and probe["declared_distribution_versions_verified"] is True
             and probe["file_backed_executable_origins_verified"] is True
             and probe["parent_ml_import_absent"] is True)
        need(all(probe[key] is False for key in (
            "production_eligible", "real_sft_execution_permitted", "real_model_or_data_access",
            "cuda_used", "complete_dynamic_dependency_closure_verified",
            "all_executable_code_origin_authenticated", "real_host_admission_verified",
        )))
        need(probe["worker_run"]["status"] == "passed")
        need(probe["worker_run"]["controlled_rank_optimizer_steps"] == 8
             and probe["worker_run"]["global_optimizer_steps_completed"] == 4)
        state = json.loads(self.run(["docker", "inspect", container]))[0]["State"]
        need(state["Running"] is False and state["ExitCode"] == 0 and state["OOMKilled"] is False)
        need(not self.run(["docker", "diff", container]).strip())
        (self.output / "cpu-worker-probe.json").write_bytes(raw)
        self.run(["docker", "rm", container])
        self.container = None
        return probe


def terminate(_signum, _frame):
    raise BuildError("terminated")


def main():
    # Reject before temporary writes, native tooling, package imports or reads.
    need(sys.argv[1:] == ["--cpu-release-candidate"])
    need(sys.platform == "linux" and platform.machine() == "x86_64")
    need(sys.version_info[:2] == (3, 12) and sys.flags.dont_write_bytecode)
    need(os.environ.get("GITHUB_ACTIONS") == "true")
    need(os.environ.get("LUMI_OCI_RELEASE_CI") == "1")
    need(os.environ.get("GITHUB_REPOSITORY") == "saidi19/lumi-model")
    commit = os.environ.get("GITHUB_SHA", "")
    need(re.fullmatch(r"[0-9a-f]{40}", commit))
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/lumi-sft-oci-release-candidate-v0.1"
    need(not output.exists())
    output.mkdir(parents=True)
    started = time.monotonic()
    builder = Builder(root, output, started + MAXIMUM_SECONDS)
    signal.signal(signal.SIGTERM, terminate)
    report = {
        "schema": SCHEMA, "status": "failed", "source_commit": commit,
        **dict.fromkeys(FALSE_CLAIMS, False),
    }
    try:
        spec = importlib.util.spec_from_file_location(
            "_lumi_oci_lock", root / "src/lumi/posttraining_sft_oci_lock_v1.py",
        )
        lock_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lock_module)
        lock_raw = (root / LOCK_PATH).read_bytes()
        lock = lock_module.validate_lock(lock_raw, LOCK_SHA256)
        with tempfile.TemporaryDirectory(
            prefix="lumi-oci-build-", dir=os.environ["RUNNER_TEMP"],
        ) as temporary:
            work = Path(temporary)
            source_inventory = builder.snapshot(work, commit)
            image_id, expected_controls = builder.image(
                work, lock, commit, lock_raw, source_inventory,
            )
            builder.accept(image_id, expected_controls)
            builder.stage = "archive"
            archive = output / "cpu-release-image.tar"
            builder.run(["docker", "image", "save", image_id], timeout=180, stdout_path=archive)
            image_bytes, image_sha256 = hashed_file(archive, MAXIMUM_IMAGE_BYTES)
            retained_bytes = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
            need(retained_bytes <= MAXIMUM_RETAINED_BYTES)
            report.update({
                "status": "passed", "image_id": image_id, "archive_sha256": image_sha256,
                "archive_bytes": image_bytes, "software_lock_sha256": LOCK_SHA256,
                "base_digest": lock["base"]["digest"],
                "base_config_digest": lock["base"]["config_digest"],
                "python_version": lock["base"]["python_version"],
                "wheel_count": builder.download_responses,
                "wheel_download_bytes": builder.download_bytes,
                "same_image_cpu_acceptance_passed": True,
                "read_only_nonroot_no_network_verified": True,
                "no_host_software_or_source_binds_verified": True,
                "probe_report_sha256": digest((output / "cpu-worker-probe.json").read_bytes()),
            })
        builder.stage = "complete"
    except Exception as error:
        report["failure_class"] = type(error).__name__ if type(error) in (
            BuildError, ValueError, KeyError, OSError, TimeoutError,
        ) else "operation_failed"
    finally:
        if builder.container is not None:
            # One exact owned container; retirement never retries acceptance.
            try:
                builder.run(["docker", "rm", "--force", builder.container],
                            timeout=15, retirement=True)
            except Exception:
                report["owned_container_cleanup_failed"] = True
        report["stage"] = builder.stage
        report["runtime_seconds"] = int(time.monotonic() - started)
        (output / "release-report.json").write_bytes(canonical(report) + b"\n")
        print(canonical(report).decode("ascii"))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
