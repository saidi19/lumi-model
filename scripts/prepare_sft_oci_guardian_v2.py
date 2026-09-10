#!/usr/bin/env python3
"""Bounded CI-only retrieval for one fresh Labels-corrected guardian acceptance.

Seven GETs: prior diagnostic job; diagnostic artifact metadata/redirect/signed
ZIP; accepted software metadata/redirect/signed ZIP. The tiny prior evidence is
hash-bound and checked before the large software request. Signed GETs never
carry credentials. No retry, cookies, proxies, POST or ML import. Software ZIP
and tar pins and transport/retirement are preserved from the frozen preparer.
Only fixed aggregate evidence is written; no raw response, token or signed URL.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

SCHEMA = "lumi.sft-oci-guardian-acquisition.v0.2"
FROZEN_PREPARER_SHA256 = "b53378d70bb8f7c7b826077f5512fb4cf10f27522d0acedf43439ad2a039f53f"
ROOT = Path(__file__).resolve().parents[1]
APPROVED_PARENT = "f3d8554d8999e32bb750fb8aa9020fca6818bf72"
PRIOR_DIAGNOSTIC_RUN = 34363848352
PRIOR_DIAGNOSTIC_JOB = 102507402716
REPOSITORY = "saidi19/lumi-model"
ARTIFACT_ID = 10101633554
ACCEPTED_RUN_ID = 34345663337
ACCEPTED_COMMIT = "c4272e36bc98354067dd23a83931a3db58823d76"
ARTIFACT_NAME = "sft-cpu-lookup-image-" + ACCEPTED_COMMIT
ZIP_BYTES = 1062113436
ZIP_SHA256 = "98d395f35af2b23874b6d5d7a56a13c969fd72fa237065fec74d81634927ac1c"
TAR_BYTES = 1062113280
TAR_SHA256 = "99d21fc60a54a65f28ceb4780f603f7ee48e49ac7b6d00a053dcdabe9ebc9ff4"
OUTPUT = "artifacts/lumi-sft-oci-guardian-acceptance-v0.2"
DIAGNOSTIC_ARTIFACT_ID = 10109060039
DIAGNOSTIC_ARTIFACT_NAME = "sft-oci-inspect-diagnostic-" + APPROVED_PARENT
DIAGNOSTIC_ZIP_BYTES = 1531
DIAGNOSTIC_ZIP_SHA256 = "64b9f11faa76f33b059f5e08a644664c3c1e6d1ea8f2732cbaff70fd592ebe42"
DIAGNOSTIC_MEMBERS = {
    "aggregate-acquisition.json": (
        766,
        "9163602b770b26ab09f332c6149efd3edf118ae9356a7c3a684426870e37af39",
    ),
    "aggregate-diagnostic.json": (
        1540,
        "e828ba303f517d217e9afba8cea389eec591523704dc9919e9a8e4104400c732",
    ),
    "aggregate-input-cleanup.json": (
        211,
        "c035fedb481f6eb23e2b69741f8b54df9d4fe5f980a13c21e3edcfff7356e52f",
    ),
}
MAXIMUM_DIAGNOSTIC_ZIP_BYTES = 65536
MAXIMUM_SECONDS = 300
MAXIMUM_METADATA_BYTES = 32768
STAGES = (
    "preflight",
    "prior_diagnostic_success",
    "diagnostic_metadata",
    "diagnostic_archive_url",
    "diagnostic_download",
    "diagnostic_evidence",
    "metadata",
    "archive_url",
    "download",
    "verify_zip",
    "extract_tar",
    "complete",
)
STAGE = "preflight"


class AcquisitionError(ValueError):
    pass


def need(value):
    if not value:
        raise AcquisitionError("acquisition_rejected") from None


def canonical(value):
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii")
        + b"\n"
    )


def identity(info):
    return info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode), info.st_uid, info.st_gid


def request_timeout(deadline, maximum):
    remaining = deadline - time.monotonic()
    need(remaining > 0)
    return min(remaining, maximum)


def metadata(value):
    need(
        type(value) is dict
        and value.get("id") == ARTIFACT_ID
        and value.get("name") == ARTIFACT_NAME
        and value.get("size_in_bytes") == ZIP_BYTES
        and value.get("digest") == "sha256:" + ZIP_SHA256
        and value.get("expired") is False
    )
    need(type(value["id"]) is int and type(value["size_in_bytes"]) is int)
    run = value.get("workflow_run")
    need(
        type(run) is dict
        and run.get("id") == ACCEPTED_RUN_ID
        and run.get("head_sha") == ACCEPTED_COMMIT
        and run.get("head_branch") == "codex/lumi-posttraining-product-v0.1"
        and run.get("repository_id") == run.get("head_repository_id") == 1300831104
    )
    need(all(type(run[key]) is int for key in ("id", "repository_id", "head_repository_id")))


def diagnostic_metadata(value):
    need(
        type(value) is dict
        and value.get("id") == DIAGNOSTIC_ARTIFACT_ID
        and value.get("name") == DIAGNOSTIC_ARTIFACT_NAME
        and value.get("size_in_bytes") == DIAGNOSTIC_ZIP_BYTES
        and value.get("digest") == "sha256:" + DIAGNOSTIC_ZIP_SHA256
        and value.get("expired") is False
    )
    need(type(value["id"]) is int and type(value["size_in_bytes"]) is int)
    run = value.get("workflow_run")
    need(
        type(run) is dict
        and run.get("id") == PRIOR_DIAGNOSTIC_RUN
        and run.get("head_sha") == APPROVED_PARENT
        and run.get("head_branch") == "codex/lumi-posttraining-product-v0.1"
        and run.get("repository_id") == run.get("head_repository_id") == 1300831104
    )
    need(all(type(run[key]) is int for key in ("id", "repository_id", "head_repository_id")))


def prior_diagnostic_success(value):
    need(
        type(value) is dict
        and type(value.get("total_count")) is int
        and value["total_count"] == 1
        and type(value.get("jobs")) is list
        and len(value["jobs"]) == 1
    )
    job = value["jobs"][0]
    need(
        type(job) is dict
        and job.get("id") == PRIOR_DIAGNOSTIC_JOB
        and job.get("run_id") == PRIOR_DIAGNOSTIC_RUN
        and job.get("run_attempt") == 1
        and job.get("head_sha") == APPROVED_PARENT
        and job.get("status") == "completed"
        and job.get("conclusion") == "success"
        and job.get("name") == "private-inspect-diagnostic"
        and all(type(job[key]) is int for key in ("id", "run_id", "run_attempt"))
    )
    steps = job.get("steps")
    need(
        type(steps) is list
        and 4 <= len(steps) <= 32
        and all(type(row) is dict and type(row.get("name")) is str for row in steps)
    )
    selected = {row["name"]: row for row in steps}
    need(len(selected) == len(steps))
    expected = {
        "Inspect diagnostic source and synthetic contracts before acquisition": "success",
        "Acquire exact accepted software for one inspect diagnostic": "success",
        "One bounded OCI inspect diagnostic with no payload execution": "success",
        "Retire only this diagnostic acquisition's exact temporary software copy": "success",
    }
    for name, conclusion in expected.items():
        need(
            name in selected
            and selected[name].get("status") == "completed"
            and selected[name].get("conclusion") == conclusion
        )


def signed_url(value):
    need(type(value) is str and 0 < len(value) <= 16384)
    parsed = urllib.parse.urlsplit(value)
    need(
        parsed.scheme == "https"
        and parsed.port in (None, 443)
        and parsed.username is None
        and parsed.password is None
        and not parsed.fragment
        and parsed.hostname
        and any(
            parsed.hostname.endswith(suffix)
            for suffix in (".blob.core.windows.net", ".actions.githubusercontent.com")
        )
    )
    return value


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, _request, _fp, _code, _message, _headers, _newurl):
        return None


def stream(source, target, maximum, expected, deadline):
    digest, total = hashlib.sha256(), 0
    while True:
        need(time.monotonic() < deadline)
        chunk = source.read(min(65536, maximum + 1 - total))
        need(time.monotonic() < deadline)
        if not chunk:
            break
        total += len(chunk)
        need(total <= maximum and target.write(chunk) == len(chunk))
        digest.update(chunk)
    need(time.monotonic() < deadline and total == maximum and digest.hexdigest() == expected)
    return total


def diagnostic_report(raw):
    expected_bytes, expected_sha = DIAGNOSTIC_MEMBERS["aggregate-diagnostic.json"]
    need(
        type(raw) is bytes
        and len(raw) == expected_bytes
        and hashlib.sha256(raw).hexdigest() == expected_sha
    )
    report = json.loads(raw)
    need(
        type(report) is dict
        and canonical(report) == raw
        and report.get("schema") == "lumi.sft-oci-inspect-fixture.v0.1"
        and report.get("status") == "passed"
    )
    need(
        report.get("cleanup")
        == {
            "owned_container_absent": True,
            "owned_directories_retired": True,
            "owned_files_retired": True,
            "owned_image_absent": True,
            "verified": True,
        }
    )
    diagnostic = report.get("diagnostic")
    need(
        type(diagnostic) is dict
        and diagnostic.get("schema") == "lumi.sft-oci-inspect-diagnostic.v0.1"
        and diagnostic.get("status") == "passed"
        and diagnostic.get("admission_passed") is False
        and diagnostic.get("code") == "profile_rejected"
        and diagnostic.get("phase") == "configuration"
        and diagnostic.get("predicate") == "configuration_labels_equal"
        and diagnostic.get("image_inspect_attempted") is True
        and diagnostic.get("container_inspect_attempted") is True
    )
    shape = diagnostic.get("configuration_shape")
    positive = {"available", "image_null", "container_is_map", "container_empty_map"}
    keys = {"available"} | {
        side + "_" + field
        for side in ("image", "container")
        for field in (
            "absent",
            "null",
            "is_string",
            "is_map",
            "is_list",
            "empty_string",
            "empty_map",
            "empty_list",
        )
    }
    need(
        type(shape) is dict
        and shape.keys() == keys
        and all(shape[key] is (key in positive) for key in keys)
    )
    for key in (
        "container_started",
        "image_exported",
        "rootfs_extracted",
        "workload_created",
        "gpu_used",
        "inference_called",
        "real_model_or_data_access",
        "real_sft_execution_permitted",
        "production_eligible",
        "software_binding_released",
    ):
        need(report.get(key) is False)
    for key in (
        "continuation_called",
        "software_binding_released",
        "real_sft_execution_permitted",
        "production_eligible",
    ):
        need(diagnostic.get(key) is False)


def diagnostic_archive(raw, deadline):
    need(
        type(raw) is bytes
        and len(raw) == DIAGNOSTIC_ZIP_BYTES
        and len(raw) <= MAXIMUM_DIAGNOSTIC_ZIP_BYTES
        and hashlib.sha256(raw).hexdigest() == DIAGNOSTIC_ZIP_SHA256
    )
    # Hash before ZIP parsing; no path extraction and no retained raw values.
    with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
        entries = archive.infolist()
        need(
            len(entries) == len(DIAGNOSTIC_MEMBERS)
            and {entry.filename for entry in entries} == DIAGNOSTIC_MEMBERS.keys()
        )
        for entry in entries:
            size, digest = DIAGNOSTIC_MEMBERS[entry.filename]
            need(
                entry.file_size == size
                and 0 < entry.compress_size <= DIAGNOSTIC_ZIP_BYTES
                and not entry.is_dir()
                and not entry.flag_bits & 1
                and entry.compress_type in (0, 8)
                and stat.S_IFMT(entry.external_attr >> 16) in (0, stat.S_IFREG)
            )
            with archive.open(entry) as source, io.BytesIO() as target:
                stream(source, target, size, digest, deadline)
                body = target.getvalue()
            need(canonical(json.loads(body)) == body)
            if entry.filename == "aggregate-diagnostic.json":
                diagnostic_report(body)


def verify_prior_diagnostic(opener, headers, deadline):
    global STAGE
    endpoint = (
        f"https://api.github.com/repos/{REPOSITORY}/actions/artifacts/{DIAGNOSTIC_ARTIFACT_ID}"
    )
    STAGE = "diagnostic_metadata"
    with opener.open(
        urllib.request.Request(endpoint, headers=headers), timeout=request_timeout(deadline, 20)
    ) as response:
        need(response.status == 200)
        raw = response.read(MAXIMUM_METADATA_BYTES + 1)
        need(time.monotonic() < deadline and len(raw) <= MAXIMUM_METADATA_BYTES)
        diagnostic_metadata(json.loads(raw))
    STAGE = "diagnostic_archive_url"
    location = None
    try:
        response = opener.open(
            urllib.request.Request(endpoint + "/zip", headers=headers),
            timeout=request_timeout(deadline, 20),
        )
    except urllib.error.HTTPError as error:
        try:
            need(error.code == 302)
            location = signed_url(error.headers.get("Location"))
        finally:
            error.close()
    else:
        response.close()
        raise AcquisitionError("acquisition_rejected")
    need(location is not None)
    STAGE = "diagnostic_download"
    with (
        opener.open(
            urllib.request.Request(
                location, headers={"User-Agent": "lumi-oci-guardian-acquisition-v2"}
            ),
            timeout=request_timeout(deadline, 30),
        ) as response,
        io.BytesIO() as target,
    ):
        need(
            response.status == 200
            and 0 < DIAGNOSTIC_ZIP_BYTES <= MAXIMUM_DIAGNOSTIC_ZIP_BYTES
            and response.headers.get("Content-Length") == str(DIAGNOSTIC_ZIP_BYTES)
        )
        stream(response, target, DIAGNOSTIC_ZIP_BYTES, DIAGNOSTIC_ZIP_SHA256, deadline)
        raw = target.getvalue()
    location = None
    STAGE = "diagnostic_evidence"
    diagnostic_archive(raw, deadline)


def git(*arguments):
    # Fixed, tiny metadata only. No user diff/file contents are returned.
    result = subprocess.run(
        ["/usr/bin/git", *arguments],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=10,
        check=False,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
    )
    need(result.returncode == 0 and len(result.stdout) <= 32768)
    return result.stdout.decode("ascii").strip()


def preflight():
    need(
        sys.platform == "linux"
        and sys.version_info[:2] == (3, 12)
        and sys.flags.no_site
        and sys.flags.isolated
        and sys.dont_write_bytecode
        and os.geteuid() != 0
        and os.environ.get("GITHUB_ACTIONS") == "true"
        and os.environ.get("GITHUB_REPOSITORY") == REPOSITORY
        and os.environ.get("GITHUB_RUN_ATTEMPT") == "1"
        and os.environ.get("GITHUB_EVENT_NAME") == "push"
        and os.environ.get("GITHUB_REF") == "refs/heads/codex/lumi-posttraining-product-v0.1"
        and os.environ.get("LUMI_OCI_GUARDIAN_V2_CI") == "1"
    )
    commit = os.environ.get("GITHUB_SHA", "")
    need(
        re.fullmatch(r"[0-9a-f]{40}", commit)
        and git("rev-parse", "HEAD") == commit
        and git("rev-parse", "HEAD^") == APPROVED_PARENT
    )
    need(not git("status", "--porcelain", "--untracked-files=no"))
    event_path = Path(os.environ["GITHUB_EVENT_PATH"])
    with event_path.open("rb") as source:
        raw = source.read(1024**2 + 1)
    need(len(raw) <= 1024**2)
    event = json.loads(raw)
    need(
        event["repository"]["private"] is True
        and event["before"] == APPROVED_PARENT
        and event["after"] == commit
    )
    output = ROOT / OUTPUT
    if not output.parent.exists():
        output.parent.mkdir(mode=0o700)
    need(not output.exists() and output.parent.is_dir() and not output.parent.is_symlink())
    output.mkdir(mode=0o700)
    return output


def acquire(output):
    global STAGE
    token = os.environ.pop("GH_TOKEN", "")
    need(type(token) is str and 0 < len(token) <= 4096 and not any(c.isspace() for c in token))
    deadline = time.monotonic() + MAXIMUM_SECONDS
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    headers = {
        "Authorization": "Bearer " + token,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "lumi-oci-guardian-acquisition-v2",
    }
    endpoint = f"https://api.github.com/repos/{REPOSITORY}/actions/artifacts/{ARTIFACT_ID}"
    STAGE = "prior_diagnostic_success"
    predecessor = (
        f"https://api.github.com/repos/{REPOSITORY}/actions/runs/"
        f"{PRIOR_DIAGNOSTIC_RUN}/attempts/1/jobs?per_page=2"
    )
    with opener.open(
        urllib.request.Request(predecessor, headers=headers), timeout=request_timeout(deadline, 20)
    ) as response:
        need(response.status == 200)
        raw = response.read(MAXIMUM_METADATA_BYTES + 1)
        need(len(raw) <= MAXIMUM_METADATA_BYTES)
        prior_diagnostic_success(json.loads(raw))
    verify_prior_diagnostic(opener, headers, deadline)
    STAGE = "metadata"
    with opener.open(
        urllib.request.Request(endpoint, headers=headers), timeout=request_timeout(deadline, 20)
    ) as response:
        need(response.status == 200)
        raw = response.read(MAXIMUM_METADATA_BYTES + 1)
        need(len(raw) <= MAXIMUM_METADATA_BYTES)
        metadata(json.loads(raw))
    STAGE = "archive_url"
    location = None
    try:
        response = opener.open(
            urllib.request.Request(endpoint + "/zip", headers=headers),
            timeout=request_timeout(deadline, 20),
        )
    except urllib.error.HTTPError as error:
        try:
            need(error.code == 302)
            location = signed_url(error.headers.get("Location"))
        finally:
            error.close()
    else:
        response.close()
        raise AcquisitionError("acquisition_rejected")
    headers.clear()
    token = None
    need(location is not None)
    parent = Path(tempfile.mkdtemp(prefix="lumi-oci-guardian-v2-input-", dir="/tmp"))
    parent_identity = identity(parent.lstat())
    archive, software = parent / "artifact.zip", parent / "cpu-release-image.tar"
    identities = {}
    try:
        STAGE = "download"
        with opener.open(
            urllib.request.Request(
                location, headers={"User-Agent": "lumi-oci-guardian-acquisition-v2"}
            ),
            timeout=request_timeout(deadline, 30),
        ) as response:
            need(
                response.status == 200 and response.headers.get("Content-Length") == str(ZIP_BYTES)
            )
            with archive.open("xb") as target:
                identities[archive] = identity(os.fstat(target.fileno()))
                stream(response, target, ZIP_BYTES, ZIP_SHA256, deadline)
        location = None
        STAGE = "verify_zip"
        # Parse only after the exact previously accepted ZIP digest has matched.
        with zipfile.ZipFile(archive, "r") as zipped:
            entries = zipped.infolist()
            need(len(entries) == 1)
            entry = entries[0]
            need(
                entry.filename == "cpu-release-image.tar"
                and entry.file_size == TAR_BYTES
                and 0 < entry.compress_size <= ZIP_BYTES
                and not entry.is_dir()
                and not entry.flag_bits & 1
                and entry.compress_type in (0, 8)
                and stat.S_IFMT(entry.external_attr >> 16) in (0, stat.S_IFREG)
            )
            STAGE = "extract_tar"
            with zipped.open(entry) as source, software.open("xb") as target:
                identities[software] = identity(os.fstat(target.fileno()))
                stream(source, target, TAR_BYTES, TAR_SHA256, deadline)
                os.fchmod(target.fileno(), 0o400)
        need(identity(archive.lstat()) == identities[archive])
        archive.unlink()
        del identities[archive]
        # Github's output file is a trusted runner-provided per-step channel.
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="ascii") as channel:
            channel.write("image_archive=" + str(software) + "\n")
            channel.write("image_identity=" + ":".join(map(str, identities[software][:2])) + "\n")
            channel.write("parent_identity=" + ":".join(map(str, parent_identity[:2])) + "\n")
        STAGE = "complete"
        result = {
            "schema": SCHEMA,
            "status": "passed",
            "artifact_id": ARTIFACT_ID,
            "accepted_run_id": ACCEPTED_RUN_ID,
            "source_commit": ACCEPTED_COMMIT,
            "https_requests": 7,
            "prior_diagnostic_run": PRIOR_DIAGNOSTIC_RUN,
            "prior_diagnostic_success_verified": True,
            "prior_input_cleanup_verified": True,
            "diagnostic_artifact_id": DIAGNOSTIC_ARTIFACT_ID,
            "diagnostic_zip_bytes": DIAGNOSTIC_ZIP_BYTES,
            "diagnostic_zip_sha256": DIAGNOSTIC_ZIP_SHA256,
            "diagnostic_report_sha256": DIAGNOSTIC_MEMBERS["aggregate-diagnostic.json"][1],
            "exact_labels_mismatch_verified": True,
            "guardian_composition_retried": False,
            "worker_dispatched": False,
            "artifact_zip_bytes": ZIP_BYTES,
            "artifact_zip_sha256": ZIP_SHA256,
            "software_tar_bytes": TAR_BYTES,
            "software_tar_sha256": TAR_SHA256,
            "network_retries": 0,
            "real_model_or_data_access": False,
            "gpu_used": False,
            "real_sft_execution_permitted": False,
            "production_eligible": False,
        }
        with (output / "aggregate-acquisition.json").open("xb") as target:
            target.write(canonical(result))
        return result
    except BaseException:
        # Only exact private files created here; ambiguous identity stops cleanup.
        for path, expected in identities.items():
            need(identity(path.lstat()) == expected)
            path.unlink()
        need(identity(parent.lstat()) == parent_identity)
        parent.rmdir()
        raise AcquisitionError("acquisition_rejected") from None


def retire_input(path, expected_file, expected_parent):
    # Only a successful acquisition's exact private object pair can be retired.
    # No glob, recursive walk, downloaded archive reread or source deletion.
    need(
        sys.platform == "linux"
        and os.geteuid() != 0
        and os.environ.get("GITHUB_ACTIONS") == "true"
        and os.environ.get("GITHUB_REPOSITORY") == REPOSITORY
        and os.environ.get("LUMI_OCI_GUARDIAN_V2_CI") == "1"
    )
    need(
        type(path) is str
        and re.fullmatch(
            r"/tmp/lumi-oci-guardian-v2-input-[a-z0-9_]{8}/cpu-release-image\.tar", path
        )
    )
    for value in (expected_file, expected_parent):
        need(type(value) is str and re.fullmatch(r"[0-9]{1,20}:[1-9][0-9]{0,19}", value))
    selected = Path(path)
    parent = selected.parent
    before_parent, before_file = parent.lstat(), selected.lstat()
    need(
        stat.S_ISDIR(before_parent.st_mode)
        and stat.S_IMODE(before_parent.st_mode) == 0o700
        and before_parent.st_uid == os.geteuid()
        and before_parent.st_gid == os.getegid()
        and stat.S_ISREG(before_file.st_mode)
        and stat.S_IMODE(before_file.st_mode) == 0o400
        and before_file.st_uid == os.geteuid()
        and before_file.st_gid == os.getegid()
        and before_file.st_nlink == 1
        and before_file.st_size == TAR_BYTES
    )
    need(
        ":".join(map(str, identity(before_parent)[:2])) == expected_parent
        and ":".join(map(str, identity(before_file)[:2])) == expected_file
        and list(parent.iterdir()) == [selected]
    )
    selected.unlink()
    parent.rmdir()
    return {
        "schema": SCHEMA,
        "status": "passed",
        "owned_input_files_retired": 1,
        "owned_input_directories_retired": 1,
        "real_sft_execution_permitted": False,
        "production_eligible": False,
    }


def main():
    output = None
    try:
        if len(sys.argv) == 5 and sys.argv[1] == "--retire-input":
            result = retire_input(*sys.argv[2:])
        else:
            need(sys.argv[1:] == ["--acquire-accepted-software"])
            output = preflight()
            result = acquire(output)
    except BaseException:
        result = {
            "schema": SCHEMA,
            "status": "failed",
            "stage": STAGE,
            "code": "acquisition_rejected",
            "real_sft_execution_permitted": False,
            "production_eligible": False,
        }
        if output is not None:
            with (output / "aggregate-acquisition-failed.json").open("xb") as target:
                target.write(canonical(result))
    sys.stdout.buffer.write(canonical(result))
    raise SystemExit(0 if result["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
