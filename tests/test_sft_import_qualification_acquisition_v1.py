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
    "_generated_qualification_acquirer", ROOT / "scripts/prepare_sft_import_qualification_v1.py")
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


def prior():
    return {"total_count": 1, "jobs": [{
        "id": subject.PRIOR_JOB_ID, "run_id": subject.PRIOR_RUN_ID, "run_attempt": 1,
        "head_sha": subject.APPROVED_PARENT, "status": "completed", "conclusion": "success",
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
    def test_exact_metadata_and_prior_job(self):
        subject.metadata(metadata())
        subject.prior_success(prior())

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
            prefix="lumi-generated-acquisition-", dir="/private/tmp"))
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
