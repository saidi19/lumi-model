#!/usr/bin/env python3
"""One fresh four-response software acquisition for tiny Linux CPU qualification.

This is not a replay of an old acquisition or native attempt. The prior successful
guardian job is provenance, not importer qualification. No model/data objects,
retries, proxies, cookies, POST, native execution or raw response retention.
"""

from __future__ import annotations

import hashlib
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

SCHEMA = "lumi.sft-import-qualification-acquisition.v0.1"
ROOT = Path(__file__).resolve().parents[1]
APPROVED_PARENT = "ecc545fc423650bf1089311915290523c10b3172"
WORKFLOW = "SFT tiny importer qualification v1"
CI_KEY = "LUMI_IMPORT_QUALIFICATION_ACQUISITION_CI"
REPOSITORY = "saidi19/lumi-model"
REPOSITORY_ID = 1300831104
BRANCH = "codex/lumi-posttraining-product-v0.1"
PRIOR_RUN_ID = 34439230964
PRIOR_JOB_ID = 102750582069
ARTIFACT_ID = 10101633554
ACCEPTED_RUN_ID = 34345663337
ACCEPTED_COMMIT = "c4272e36bc98354067dd23a83931a3db58823d76"
ARTIFACT_NAME = "sft-cpu-lookup-image-" + ACCEPTED_COMMIT
ZIP_BYTES = 1062113436
ZIP_SHA256 = "98d395f35af2b23874b6d5d7a56a13c969fd72fa237065fec74d81634927ac1c"
TAR_BYTES = 1062113280
TAR_SHA256 = "99d21fc60a54a65f28ceb4780f603f7ee48e49ac7b6d00a053dcdabe9ebc9ff4"
OUTPUT = "artifacts/lumi-sft-import-qualification-v0.1"
INPUT_PREFIX = "lumi-import-qualification-input-"
MAXIMUM_SECONDS = 300
MAXIMUM_METADATA_BYTES = 32768
MAXIMUM_RESPONSES = 4
MAXIMUM_REPORT_BYTES = 32768
STAGE = "preflight"
_ATTEMPTED = False
_ACQUIRED = False
PRIOR_STEPS = (
    "Labels successor source and synthetic contracts before acquisition",
    "Verify exact Labels diagnostic, then retrieve accepted software",
    "One v2 composition, two negative pins then one positive WorkerRun",
    "Retire only this acquisition's exact temporary software copy",
)
FALSE_FLAGS = ("real_model_or_data_access", "gpu_used", "inference_called",
               "real_sft_execution_permitted", "production_eligible", "worker_dispatched")


class AcquisitionError(ValueError):
    pass


def need(value):
    if not value:
        raise AcquisitionError("acquisition_rejected") from None


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                       allow_nan=False).encode("ascii") + b"\n")


def pairs(rows):
    result = {}
    for key, value in rows:
        need(key not in result)
        result[key] = value
    return result


def integer(raw):
    need(len(raw) <= 20)
    value = int(raw)
    need(-(2**63) <= value < 2**63)
    return value


def reject_number(_raw):
    raise AcquisitionError("acquisition_rejected") from None


def strict_json(raw):
    need(type(raw) is bytes and len(raw) <= 1024**2)
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_int=integer,
                          parse_float=reject_number, parse_constant=reject_number)
    except BaseException:
        raise AcquisitionError("acquisition_rejected") from None


def identity(info):
    return info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode), info.st_uid, info.st_gid


def metadata(value):
    need(type(value) is dict)
    expected = {"id": ARTIFACT_ID, "name": ARTIFACT_NAME, "size_in_bytes": ZIP_BYTES,
                "digest": "sha256:" + ZIP_SHA256, "expired": False}
    need(all(type(value.get(key)) is type(part) and value[key] == part
             for key, part in expected.items()))
    run = value.get("workflow_run")
    expected_run = {"id": ACCEPTED_RUN_ID, "head_sha": ACCEPTED_COMMIT,
                    "head_branch": BRANCH, "repository_id": REPOSITORY_ID,
                    "head_repository_id": REPOSITORY_ID}
    need(type(run) is dict and all(type(run.get(key)) is type(part) and run[key] == part
                                  for key, part in expected_run.items()))


def prior_success(value):
    need(type(value) is dict and type(value.get("total_count")) is int
         and value["total_count"] == 1 and type(value.get("jobs")) is list
         and len(value["jobs"]) == 1)
    job = value["jobs"][0]
    expected = {"id": PRIOR_JOB_ID, "run_id": PRIOR_RUN_ID, "run_attempt": 1,
                "head_sha": APPROVED_PARENT, "status": "completed", "conclusion": "success",
                "name": "private-native-composition-v2"}
    need(type(job) is dict and all(type(job.get(key)) is type(part) and job[key] == part
                                  for key, part in expected.items()))
    steps = job.get("steps")
    need(type(steps) is list and 4 <= len(steps) <= 32
         and all(type(row) is dict and type(row.get("name")) is str for row in steps))
    selected = {row["name"]: row for row in steps}
    need(len(selected) == len(steps))
    for name in PRIOR_STEPS:
        need(name in selected and selected[name].get("status") == "completed"
             and selected[name].get("conclusion") == "success")


def signed_url(value):
    need(type(value) is str and 0 < len(value) <= 16384
         and not any(ord(char) <= 32 or ord(char) >= 127 for char in value))
    try:
        parsed = urllib.parse.urlsplit(value)
        need(parsed.scheme == "https" and parsed.port in (None, 443)
             and parsed.username is None and parsed.password is None and not parsed.fragment
             and parsed.hostname and any(parsed.hostname.endswith(suffix) for suffix in
                 (".blob.core.windows.net", ".actions.githubusercontent.com")))
    except BaseException:
        raise AcquisitionError("acquisition_rejected") from None
    return value


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, _request, _fp, _code, _message, _headers, _newurl):
        return None


class Transport:
    def __init__(self, deadline):
        self.deadline, self.requests = deadline, 0
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def open(self, request, maximum=20):
        need(self.requests < MAXIMUM_RESPONSES and request.get_method() == "GET")
        remaining = self.deadline - time.monotonic()
        need(remaining > 0)
        self.requests += 1  # An uncertain started request is consumed, never retried.
        response = self.opener.open(request, timeout=min(remaining, maximum))
        if time.monotonic() >= self.deadline:
            response.close()
            raise AcquisitionError("acquisition_rejected") from None
        return response

    def json(self, endpoint, headers):
        with self.open(urllib.request.Request(endpoint, headers=headers)) as response:
            need(response.status == 200)
            raw = response.read(MAXIMUM_METADATA_BYTES + 1)
            length = response.headers.get("Content-Length")
            need(time.monotonic() < self.deadline and len(raw) <= MAXIMUM_METADATA_BYTES
                 and (length is None or length == str(len(raw))))
            return strict_json(raw)


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


def git(*arguments):
    result = subprocess.run(["/usr/bin/git", *arguments], cwd=ROOT,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, timeout=10, check=False,
                            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"})
    need(result.returncode == 0 and len(result.stdout) <= 32768)
    return result.stdout.decode("ascii").strip()


def runtime():
    need(sys.platform == "linux" and sys.version_info[:2] == (3, 12)
         and sys.flags.no_site and sys.flags.isolated and sys.dont_write_bytecode
         and os.geteuid() != 0 and os.environ.get("GITHUB_ACTIONS") == "true"
         and os.environ.get("GITHUB_REPOSITORY") == REPOSITORY
         and os.environ.get("GITHUB_WORKFLOW") == WORKFLOW
         and os.environ.get("GITHUB_RUN_ATTEMPT") == "1"
         and os.environ.get("GITHUB_EVENT_NAME") == "push"
         and os.environ.get("GITHUB_REF") == "refs/heads/" + BRANCH
         and os.environ.get(CI_KEY) == "1")


def preflight():
    global _ATTEMPTED
    runtime()  # Reject Mac before writes, subprocesses or requests.
    need(not _ATTEMPTED)
    _ATTEMPTED = True
    commit = os.environ.get("GITHUB_SHA", "")
    need(re.fullmatch(r"[0-9a-f]{40}", commit)
         and git("rev-parse", "HEAD") == commit
         and git("rev-list", "--parents", "-n", "1", "HEAD").split()
         == [commit, APPROVED_PARENT]
         and not git("status", "--porcelain", "--untracked-files=no"))
    with Path(os.environ["GITHUB_EVENT_PATH"]).open("rb") as source:
        raw = source.read(1024**2 + 1)
    event = strict_json(raw)
    repository = event.get("repository") if type(event) is dict else None
    need(type(repository) is dict and repository.get("private") is True
         and type(repository.get("id")) is int and repository["id"] == REPOSITORY_ID
         and repository.get("full_name") == REPOSITORY
         and event.get("before") == APPROVED_PARENT and event.get("after") == commit)
    output = ROOT / OUTPUT
    if not output.parent.exists():
        output.parent.mkdir(mode=0o700)
    need(not output.exists() and output.parent.is_dir() and not output.parent.is_symlink())
    output.mkdir(mode=0o700)
    return output


def write_report(output, name, result):
    raw = canonical(result)
    need(len(raw) <= MAXIMUM_REPORT_BYTES)
    with (output / name).open("xb") as target:
        os.fchmod(target.fileno(), 0o600)
        need(target.write(raw) == len(raw))


def acquire(output):
    global STAGE, _ACQUIRED
    need(not _ACQUIRED)
    _ACQUIRED = True
    token = os.environ.pop("GH_TOKEN", "")
    need(type(token) is str and 0 < len(token) <= 4096
         and all(33 <= ord(char) <= 126 for char in token))
    deadline = time.monotonic() + MAXIMUM_SECONDS
    transport = Transport(deadline)
    headers = {"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
               "Accept-Encoding": "identity", "X-GitHub-Api-Version": "2022-11-28",
               "User-Agent": "lumi-import-qualification-acquisition-v1"}
    prefix = f"https://api.github.com/repos/{REPOSITORY}/actions/"
    STAGE = "prior_success"
    prior_success(transport.json(prefix + f"runs/{PRIOR_RUN_ID}/attempts/1/jobs?per_page=2",
                                 headers))
    STAGE = "metadata"
    endpoint = prefix + f"artifacts/{ARTIFACT_ID}"
    metadata(transport.json(endpoint, headers))
    STAGE = "archive_url"
    location = None
    try:
        response = transport.open(urllib.request.Request(endpoint + "/zip", headers=headers))
    except urllib.error.HTTPError as error:
        try:
            need(error.code == 302)
            location = signed_url(error.headers.get("Location"))
        finally:
            error.close()
    else:
        response.close()
        raise AcquisitionError("acquisition_rejected") from None
    headers.clear()
    token = None
    need(location is not None and time.monotonic() < deadline)
    parent = Path(tempfile.mkdtemp(prefix=INPUT_PREFIX, dir="/tmp"))
    parent_identity = identity(parent.lstat())
    archive, software = parent / "artifact.zip", parent / "cpu-release-image.tar"
    identities = {}
    try:
        STAGE = "download"
        with transport.open(urllib.request.Request(location, headers={
                "User-Agent": "lumi-import-qualification-acquisition-v1",
                "Accept-Encoding": "identity"}), 30) as response:
            need(response.status == 200
                 and response.headers.get("Content-Length") == str(ZIP_BYTES))
            with archive.open("xb") as target:
                identities[archive] = identity(os.fstat(target.fileno()))
                os.fchmod(target.fileno(), 0o600)
                stream(response, target, ZIP_BYTES, ZIP_SHA256, deadline)
        location = None
        STAGE = "verify_zip"
        with zipfile.ZipFile(archive, "r") as zipped:
            entries = zipped.infolist()
            need(len(entries) == 1)
            entry = entries[0]
            need(entry.filename == "cpu-release-image.tar" and entry.file_size == TAR_BYTES
                 and 0 < entry.compress_size <= ZIP_BYTES and not entry.is_dir()
                 and not entry.flag_bits & 1 and entry.compress_type in (0, 8)
                 and stat.S_IFMT(entry.external_attr >> 16) in (0, stat.S_IFREG))
            STAGE = "extract_tar"
            with zipped.open(entry) as source, software.open("xb") as target:
                identities[software] = identity(os.fstat(target.fileno()))
                os.fchmod(target.fileno(), 0o600)
                stream(source, target, TAR_BYTES, TAR_SHA256, deadline)
                os.fchmod(target.fileno(), 0o400)
        need(identity(archive.lstat()) == identities[archive])
        archive.unlink()
        del identities[archive]
        need(transport.requests == MAXIMUM_RESPONSES and time.monotonic() < deadline)
        result = {"schema": SCHEMA, "status": "passed", "https_requests": transport.requests,
                  "prior_guardian_run": PRIOR_RUN_ID, "prior_guardian_job": PRIOR_JOB_ID,
                  "prior_guardian_success_verified": True, "prior_input_cleanup_verified": True,
                  "artifact_id": ARTIFACT_ID, "accepted_run_id": ACCEPTED_RUN_ID,
                  "source_commit": ACCEPTED_COMMIT, "artifact_zip_bytes": ZIP_BYTES,
                  "artifact_zip_sha256": ZIP_SHA256, "software_tar_bytes": TAR_BYTES,
                  "software_tar_sha256": TAR_SHA256, "network_retries": 0,
                  "old_diagnostic_artifact_reads": 0, "native_qualification_retried": False,
                  **dict.fromkeys(FALSE_FLAGS, False)}
        # Trusted runner-provided channel; no token or signed URL enters it.
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="ascii") as channel:
            raw = ("image_archive=" + str(software) + "\nimage_identity="
                   + ":".join(map(str, identities[software][:2])) + "\nparent_identity="
                   + ":".join(map(str, parent_identity[:2])) + "\n")
            need(channel.write(raw) == len(raw))
        write_report(output, "aggregate-acquisition.json", result)
        STAGE = "complete"
        return result
    except BaseException:
        for path, expected in identities.items():
            need(identity(path.lstat()) == expected)
            path.unlink()
        need(identity(parent.lstat()) == parent_identity)
        parent.rmdir()
        raise AcquisitionError("acquisition_rejected") from None


def retire_input(path, expected_file, expected_parent):
    runtime()
    need(type(path) is str and re.fullmatch(
        r"/tmp/lumi-import-qualification-input-[a-z0-9_]{8}/cpu-release-image\.tar", path))
    for value in (expected_file, expected_parent):
        need(type(value) is str and re.fullmatch(r"[0-9]{1,20}:[1-9][0-9]{0,19}", value))
    selected = Path(path)
    parent = selected.parent
    before_parent, before_file = parent.lstat(), selected.lstat()
    need(stat.S_ISDIR(before_parent.st_mode) and stat.S_IMODE(before_parent.st_mode) == 0o700
         and before_parent.st_uid == os.geteuid() and before_parent.st_gid == os.getegid()
         and stat.S_ISREG(before_file.st_mode) and stat.S_IMODE(before_file.st_mode) == 0o400
         and before_file.st_uid == os.geteuid() and before_file.st_gid == os.getegid()
         and before_file.st_nlink == 1 and before_file.st_size == TAR_BYTES)
    need(":".join(map(str, identity(before_parent)[:2])) == expected_parent
         and ":".join(map(str, identity(before_file)[:2])) == expected_file
         and list(parent.iterdir()) == [selected])
    selected.unlink()
    parent.rmdir()
    return {"schema": SCHEMA, "status": "passed", "owned_input_files_retired": 1,
            "owned_input_directories_retired": 1, **dict.fromkeys(FALSE_FLAGS, False)}


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
        result = {"schema": SCHEMA, "status": "failed", "stage": STAGE,
                  "code": "acquisition_rejected", **dict.fromkeys(FALSE_FLAGS, False)}
        if output is not None:
            try:
                if not (output / "aggregate-acquisition.json").exists():
                    write_report(output, "aggregate-acquisition.json", result)
            except BaseException:
                pass  # Never expose a secondary file exception or retry a started write.
    sys.stdout.buffer.write(canonical(result))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
