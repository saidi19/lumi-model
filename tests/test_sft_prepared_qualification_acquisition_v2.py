"""Tiny generated ZIP/HTTP doubles only; never network or native execution."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tempfile
import types
import unittest
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_generated_prepared_qualification_acquirer_v1",
    ROOT / "scripts/prepare_sft_prepared_qualification_v2.py")
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


def prior():
    return {"total_count": 1, "jobs": [{
        "id": subject.PRIOR_JOB_ID, "run_id": subject.PRIOR_RUN_ID, "run_attempt": 1,
        "head_sha": subject.PRIOR_GUARDIAN_HEAD, "status": "completed", "conclusion": "success",
        "name": "private-native-composition-v2", "steps": [
            {"name": name, "status": "completed", "conclusion": "success"}
            for name in subject.PRIOR_STEPS]}]}


def metadata():
    return {"id": subject.ARTIFACT_ID, "name": subject.ARTIFACT_NAME,
            "size_in_bytes": subject.ZIP_BYTES, "digest": "sha256:" + subject.ZIP_SHA256,
            "expired": False, "workflow_run": {
                "id": subject.ACCEPTED_RUN_ID, "head_sha": subject.ACCEPTED_COMMIT,
                "head_branch": subject.BRANCH, "repository_id": subject.REPOSITORY_ID,
                "head_repository_id": subject.REPOSITORY_ID}}


class Response(io.BytesIO):
    def __init__(self, raw, status=200, headers=None):
        super().__init__(raw)
        self.status = status
        self.headers = headers if headers is not None else {"Content-Length": str(len(raw))}


class ContractTests(unittest.TestCase):
    def test_prepared_successor_identity_and_frozen_acquisition_bounds(self):
        self.assertEqual(subject.SCHEMA, "lumi.sft-prepared-qualification-acquisition.v0.2")
        self.assertEqual(subject.WORKFLOW, "SFT prepared importer qualification v2")
        self.assertEqual(subject.CI_KEY, "LUMI_PREPARED_QUALIFICATION_ACQUISITION_CI")
        self.assertEqual(subject.OUTPUT, "artifacts/lumi-sft-prepared-qualification-v0.2")
        self.assertEqual(subject.INPUT_PREFIX, "lumi-prepared-qualification-v2-input-")
        self.assertEqual((subject.MAXIMUM_RESPONSES, subject.MAXIMUM_SECONDS), (4, 300))
        self.assertEqual((subject.PRIOR_RUN_ID, subject.PRIOR_JOB_ID),
                         (34439230964, 102750582069))
        self.assertEqual((subject.ARTIFACT_ID, subject.ACCEPTED_RUN_ID),
                         (10101633554, 34345663337))
        self.assertEqual(subject.ACCEPTED_COMMIT,
                         "c4272e36bc98354067dd23a83931a3db58823d76")
        self.assertEqual((subject.ZIP_BYTES, subject.TAR_BYTES), (1062113436, 1062113280))
        self.assertEqual(subject.ZIP_SHA256,
                         "98d395f35af2b23874b6d5d7a56a13c969fd72fa237065fec74d81634927ac1c")
        self.assertEqual(subject.TAR_SHA256,
                         "99d21fc60a54a65f28ceb4780f603f7ee48e49ac7b6d00a053dcdabe9ebc9ff4")

    def test_exact_metadata_and_prior_job(self):
        subject.metadata(metadata())
        subject.prior_success(prior())

    def test_historical_guardian_head_is_not_current_publication_parent(self):
        self.assertEqual(subject.APPROVED_PARENT,
                         "781d4c9de9d6acf9f2161855a6089914b980a294")
        self.assertEqual(subject.PRIOR_GUARDIAN_HEAD,
                         "ecc545fc423650bf1089311915290523c10b3172")
        self.assertNotEqual(subject.APPROVED_PARENT, subject.PRIOR_GUARDIAN_HEAD)
        subject.prior_success(prior())
        for head in (subject.APPROVED_PARENT, "0" * 40):
            row = prior()
            row["jobs"][0]["head_sha"] = head
            with self.subTest(head=head), self.assertRaises(subject.AcquisitionError):
                subject.prior_success(row)

    def test_metadata_wrong_binding_or_bool_integer_rejected(self):
        for key, value in (("id", True), ("expired", True), ("digest", "wrong"),
                           ("size_in_bytes", 1), ("name", "other")):
            row = metadata()
            row[key] = value
            with self.subTest(key=key), self.assertRaises(subject.AcquisitionError):
                subject.metadata(row)
        row = metadata()
        row["workflow_run"]["repository_id"] = True
        with self.assertRaises(subject.AcquisitionError):
            subject.metadata(row)

    def test_prior_failure_missing_step_duplicate_or_second_attempt_rejected(self):
        values = []
        for key, value in (("run_attempt", 2), ("conclusion", "failure"), ("id", True)):
            row = prior()
            row["jobs"][0][key] = value
            values.append(row)
        row = prior()
        row["jobs"][0]["steps"][0]["conclusion"] = "skipped"
        values.append(row)
        row = prior()
        row["jobs"][0]["steps"].append(row["jobs"][0]["steps"][0])
        values.append(row)
        for row in values:
            with self.assertRaises(subject.AcquisitionError):
                subject.prior_success(row)

    def test_duplicate_float_nonfinite_and_oversized_JSON_rejected(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":1.0}', b'{"x":NaN}',
                    b'{"x":123456789012345678901}', b"x" * (1024**2 + 1)):
            with self.assertRaises(subject.AcquisitionError):
                subject.strict_json(raw)

    def test_signed_URL_closed_contract(self):
        good = "https://owned.blob.core.windows.net/generated?signature=fixture"
        self.assertEqual(subject.signed_url(good), good)
        for value in ("http://owned.blob.core.windows.net/a",
                      "https://owned.blob.core.windows.net.evil.example/a",
                      "https://u:p@owned.blob.core.windows.net/a",
                      good + "#fragment", "https://owned.blob.core.windows.net:444/a",
                      good + "\n", "https://evil.example/a"):
            with self.subTest(value=value), self.assertRaises(subject.AcquisitionError):
                subject.signed_url(value)

    def test_mac_preflight_denies_before_git_write_or_transport(self):
        with patch.object(subject.sys, "platform", "darwin"), \
                patch.object(subject, "git") as git, \
                patch.object(subject, "Transport") as transport, \
                self.assertRaises(subject.AcquisitionError):
            subject.preflight()
        git.assert_not_called()
        transport.assert_not_called()

    def test_linux_preflight_accepts_new_parent_and_rejects_historical_parent(self):
        commit = "a" * 40
        for parent, accepted in ((subject.APPROVED_PARENT, True),
                                 (subject.PRIOR_GUARDIAN_HEAD, False)):
            with self.subTest(accepted=accepted), tempfile.TemporaryDirectory(
                    prefix="lumi-generated-prepared-v1-preflight-") as name:
                root = Path(name).resolve()
                event = root / "push.json"
                event.write_bytes(subject.canonical({
                    "repository": {"private": True, "id": subject.REPOSITORY_ID,
                                   "full_name": subject.REPOSITORY},
                    "before": parent, "after": commit}))
                environment = {
                    "GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": subject.REPOSITORY,
                    "GITHUB_WORKFLOW": subject.WORKFLOW, "GITHUB_RUN_ATTEMPT": "1",
                    "GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/" + subject.BRANCH,
                    "GITHUB_SHA": commit, "GITHUB_EVENT_PATH": str(event), subject.CI_KEY: "1"}
                replies = {("rev-parse", "HEAD"): commit,
                           ("rev-list", "--parents", "-n", "1", "HEAD"): commit + " " + parent,
                           ("status", "--porcelain", "--untracked-files=no"): ""}
                with patch.object(subject, "ROOT", root), \
                        patch.object(subject, "_ATTEMPTED", False), \
                        patch.object(subject.sys, "platform", "linux"), \
                        patch.object(subject.sys, "version_info", (3, 12, 0)), \
                        patch.object(subject.sys, "flags",
                                     types.SimpleNamespace(no_site=True, isolated=True)), \
                        patch.object(subject.sys, "dont_write_bytecode", True), \
                        patch.object(subject.os, "geteuid", return_value=1000), \
                        patch.dict(subject.os.environ, environment, clear=True), \
                        patch.object(subject, "git",
                                     side_effect=lambda *args, replies=replies: replies[args]), \
                        patch.object(subject.subprocess, "run") as subprocess_run, \
                        patch.object(subject.urllib.request, "build_opener") as build_opener:
                    if accepted:
                        self.assertEqual(subject.preflight(), root / subject.OUTPUT)
                        self.assertTrue((root / subject.OUTPUT).is_dir())
                    else:
                        with self.assertRaises(subject.AcquisitionError):
                            subject.preflight()
                        self.assertFalse((root / subject.OUTPUT).exists())
                    self.assertTrue(subject._ATTEMPTED)
                    subprocess_run.assert_not_called()
                    build_opener.assert_not_called()

    def test_transport_response_limit_and_POST_reject_before_opener(self):
        opener = Mock()
        with patch.object(subject.urllib.request, "build_opener", return_value=opener):
            transport = subject.Transport(subject.time.monotonic() + 30)
        transport.requests = 4
        with self.assertRaises(subject.AcquisitionError):
            transport.open(urllib.request.Request("https://api.github.com/generated"))
        transport.requests = 0
        with self.assertRaises(subject.AcquisitionError):
            transport.open(urllib.request.Request("https://api.github.com/generated", data=b"x"))
        opener.open.assert_not_called()

    def test_deadline_and_metadata_cap_consume_no_extra_request(self):
        opener = Mock()
        with patch.object(subject.urllib.request, "build_opener", return_value=opener):
            transport = subject.Transport(-1)
        with self.assertRaises(subject.AcquisitionError):
            transport.open(urllib.request.Request("https://api.github.com/generated"))
        opener.open.assert_not_called()
        transport.deadline = subject.time.monotonic() + 30
        opener.open.return_value = Response(b"x" * (subject.MAXIMUM_METADATA_BYTES + 1))
        with self.assertRaises(subject.AcquisitionError):
            transport.json("https://api.github.com/generated", {})
        self.assertEqual(transport.requests, 1)

    def test_stream_short_overflow_wrong_hash_and_deadline_reject(self):
        for raw, maximum, pin, deadline in [
            (b"x", 2, "0" * 64, subject.time.monotonic() + 30),
            (b"xx", 1, "0" * 64, subject.time.monotonic() + 30),
            (b"x", 1, "0" * 64, subject.time.monotonic() + 30),
            (b"x", 1, hashlib.sha256(b"x").hexdigest(), -1),
        ]:
            with self.assertRaises(subject.AcquisitionError):
                subject.stream(io.BytesIO(raw), io.BytesIO(), maximum, pin, deadline)


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory(
            prefix="lumi-generated-acquisition-"))
        self.root = Path(self.directory).resolve()
        self.output = self.root / "reports"
        self.output.mkdir(mode=0o700)
        self.input = self.root / "owned"
        self.input.mkdir(mode=0o700)
        self.channel = self.root / "output-channel"
        self.tar = b"tiny generated software, not model or dataset"
        target = io.BytesIO()
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("cpu-release-image.tar", self.tar)
        self.zip = target.getvalue()
        for key, value in {"ZIP_BYTES": len(self.zip),
                           "ZIP_SHA256": hashlib.sha256(self.zip).hexdigest(),
                           "TAR_BYTES": len(self.tar),
                           "TAR_SHA256": hashlib.sha256(self.tar).hexdigest(),
                           "_ACQUIRED": False}.items():
            self.enterContext(patch.object(subject, key, value))
        self.enterContext(patch.dict(subject.os.environ,
                                    GH_TOKEN="synthetic-token", GITHUB_OUTPUT=str(self.channel)))
        self.enterContext(patch.object(subject, "tempfile", types.SimpleNamespace(
            mkdtemp=Mock(return_value=str(self.input)))))
        self.responses = [
            Response(subject.canonical(prior())), Response(subject.canonical(metadata())),
            urllib.error.HTTPError("https://api.github.com/generated", 302, "found",
                {"Location": "https://fixture.blob.core.windows.net/one?synthetic"}, io.BytesIO()),
            Response(self.zip),
        ]
        self.requests = []
        self.opener = types.SimpleNamespace(open=self.open)
        self.enterContext(patch.object(subject.urllib.request, "build_opener",
                                       return_value=self.opener))

    def open(self, request, **_kwargs):
        self.requests.append(request)
        value = self.responses[len(self.requests) - 1]
        if isinstance(value, Exception):
            raise value
        return value

    def test_four_response_success_no_signed_credentials_or_raw_retention(self):
        result = subject.acquire(self.output)
        self.assertEqual(len(self.requests), 4)
        self.assertEqual(result["https_requests"], 4)
        self.assertTrue(all(request.get_method() == "GET" for request in self.requests))
        self.assertEqual(self.requests[-1].get_header("Authorization"), None)
        self.assertTrue(all(request.get_header("Authorization") == "Bearer synthetic-token"
                            for request in self.requests[:3]))
        software = self.input / "cpu-release-image.tar"
        self.assertEqual(software.read_bytes(), self.tar)
        self.assertEqual(software.stat().st_mode & 0o777, 0o400)
        self.assertEqual(list(self.input.iterdir()), [software])
        raw = (self.output / "aggregate-acquisition.json").read_bytes()
        self.assertEqual(raw, subject.canonical(result))
        for forbidden in (b"synthetic-token", b"blob.core", str(self.input).encode()):
            self.assertNotIn(forbidden, raw)

    def test_expired_metadata_stops_before_redirect_or_download(self):
        row = metadata()
        row["expired"] = True
        self.responses[1] = Response(subject.canonical(row))
        with self.assertRaises(subject.AcquisitionError):
            subject.acquire(self.output)
        self.assertEqual(len(self.requests), 2)
        self.assertFalse(list(self.output.iterdir()))

    def test_wrong_zip_digest_cleans_own_files_and_consumes_attempt(self):
        self.responses[3] = Response(b"x" * len(self.zip))
        with self.assertRaises(subject.AcquisitionError):
            subject.acquire(self.output)
        self.assertFalse(self.input.exists())
        count = len(self.requests)
        with self.assertRaises(subject.AcquisitionError):
            subject.acquire(self.output)
        self.assertEqual(len(self.requests), count)

    def test_extra_signed_redirect_is_not_followed(self):
        self.responses[3] = urllib.error.HTTPError("https://fixture.blob.core.windows.net/a",
            302, "found", {"Location": "https://other.blob.core.windows.net/a"}, io.BytesIO())
        with self.assertRaises(subject.AcquisitionError):
            subject.acquire(self.output)
        self.assertEqual(len(self.requests), 4)
        self.assertFalse(self.input.exists())

    def test_wrong_Content_Length_denies_and_cleans_input(self):
        self.responses[3] = Response(self.zip, headers={"Content-Length": "1"})
        with self.assertRaises(subject.AcquisitionError):
            subject.acquire(self.output)
        self.assertFalse(self.input.exists())

    def test_channel_failure_has_no_success_receipt_and_retires_own_input(self):
        self.channel.mkdir()
        with self.assertRaises(subject.AcquisitionError):
            subject.acquire(self.output)
        self.assertFalse(self.input.exists())
        self.assertFalse((self.output / "aggregate-acquisition.json").exists())

    def test_failure_main_keeps_only_named_failure_no_exception(self):
        stdout = types.SimpleNamespace(buffer=io.BytesIO())
        with patch.object(subject, "preflight", return_value=self.output), \
                patch.object(subject, "acquire", side_effect=RuntimeError("sensitive exception")), \
                patch.object(subject.sys, "argv", ["prepare", "--acquire-accepted-software"]), \
                patch.object(subject.sys, "stdout", stdout):
            self.assertEqual(subject.main(), 2)
        self.assertEqual([path.name for path in self.output.iterdir()],
                         ["aggregate-acquisition.json"])
        raw = stdout.buffer.getvalue()
        self.assertNotIn(b"sensitive", raw)
        self.assertEqual(json.loads(raw)["status"], "failed")


class FixturePortabilityTests(unittest.TestCase):
    def test_all_acquisition_cases_work_without_macos_specific_parent(self):
        create_directory = tempfile.TemporaryDirectory
        generated = []

        def portable_directory(*args, **kwargs):
            if kwargs.get("dir") == "/private/tmp":
                raise FileNotFoundError("generated missing macOS-specific parent")
            self.assertNotIn("dir", kwargs)
            directory = create_directory(*args, **kwargs)
            generated.append(Path(directory.name).resolve())
            return directory

        result = unittest.TestResult()
        with patch.object(tempfile, "TemporaryDirectory", side_effect=portable_directory):
            unittest.defaultTestLoader.loadTestsFromTestCase(AcquisitionTests).run(result)
        self.assertEqual((result.testsRun, len(result.failures), len(result.errors)), (7, 0, 0))
        self.assertEqual(len(generated), 7)
        self.assertTrue(all(not directory.exists() for directory in generated))

    def test_fixture_uses_configured_default_parent_and_cleans_up(self):
        with tempfile.TemporaryDirectory(prefix="lumi-generated-portable-parent-") as name:
            parent = Path(name).resolve()
            case = AcquisitionTests(
                "test_four_response_success_no_signed_credentials_or_raw_retention")
            result = unittest.TestResult()
            with patch.object(tempfile, "tempdir", str(parent)):
                case.run(result)
            self.assertEqual((result.testsRun, len(result.failures), len(result.errors)),
                             (1, 0, 0))
            self.assertEqual(Path(case.directory).parent, parent)
            self.assertFalse(Path(case.directory).exists())
            self.assertEqual(list(parent.iterdir()), [])


class RetirementTests(unittest.TestCase):
    def test_only_owned_generated_input_is_retired(self):
        name = tempfile.mkdtemp(prefix=subject.INPUT_PREFIX, dir="/tmp")
        parent = Path(name)
        file = parent / "cpu-release-image.tar"
        try:
            file.write_bytes(b"synthetic")
            file.chmod(0o400)
            ids = [":".join(map(str, subject.identity(path.lstat())[:2]))
                   for path in (file, parent)]
            # macOS inherits the temporary parent group; model the matching
            # Linux effective group without changing the production guard.
            with patch.object(subject, "runtime"), patch.object(subject, "TAR_BYTES", 9), \
                    patch.object(subject.os, "getegid", return_value=parent.stat().st_gid):
                result = subject.retire_input(str(file), *ids)
            self.assertFalse(parent.exists())
            self.assertEqual(result["owned_input_files_retired"], 1)
        finally:
            if file.exists():
                file.unlink()
            if parent.exists():
                parent.rmdir()

    def test_out_of_namespace_path_rejected_before_file_lookup(self):
        with patch.object(subject, "runtime"), self.assertRaises(subject.AcquisitionError):
            subject.retire_input("/tmp/unrelated/source.tar", "1:2", "1:3")

    def test_existing_report_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(prefix="lumi-generated-report-") as name:
            path = Path(name)
            subject.write_report(path, "aggregate-acquisition.json", {"status": "first"})
            before = (path / "aggregate-acquisition.json").read_bytes()
            with self.assertRaises(FileExistsError):
                subject.write_report(path, "aggregate-acquisition.json", {"status": "second"})
            self.assertEqual((path / "aggregate-acquisition.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

