"""Pure contracts, AST and tiny synthetic I/O; no entrypoint or network calls."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import io
import json
import stat
import sys
import tempfile
import types
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/prepare_sft_oci_guardian_v1.py"
spec = importlib.util.spec_from_file_location("_synthetic_oci_guardian_acquisition_v1", SOURCE)
assert spec is not None and spec.loader is not None
acquisition = importlib.util.module_from_spec(spec)
before_import = set(sys.modules)
spec.loader.exec_module(acquisition)
IMPORT_ADDITIONS = set(sys.modules) - before_import
SIGNED = "https://synthetic.blob.core.windows.net/software/accepted.zip?sig=SYNTHETIC"


def good_metadata():
    return {
        "id": acquisition.ARTIFACT_ID,
        "name": acquisition.ARTIFACT_NAME,
        "size_in_bytes": acquisition.ZIP_BYTES,
        "digest": "sha256:" + acquisition.ZIP_SHA256,
        "expired": False,
        "workflow_run": {
            "id": acquisition.ACCEPTED_RUN_ID,
            "head_sha": acquisition.ACCEPTED_COMMIT,
            "head_branch": "codex/lumi-posttraining-product-v0.1",
            "repository_id": 1300831104,
            "head_repository_id": 1300831104,
        },
    }


def good_prior_source_stop():
    return {
        "total_count": 1,
        "jobs": [
            {
                "id": 102487798348,
                "run_id": 34358096760,
                "run_attempt": 1,
                "head_sha": "01b89f9477dc28a6f2b771f164f347dafb53d6dc",
                "status": "completed",
                "conclusion": "failure",
                "name": "private-native-composition",
                "steps": [
                    {
                        "name": "New source and synthetic contracts before any software download",
                        "status": "completed",
                        "conclusion": "failure",
                    },
                    {
                        "name": "Retrieve exact already accepted software, no rebuild or retry",
                        "status": "completed",
                        "conclusion": "skipped",
                    },
                    {
                        "name": ("One native composition, two negative pins "
                                 "then one positive WorkerRun"),
                        "status": "completed",
                        "conclusion": "skipped",
                    },
                    {
                        "name": "Retire only this acquisition's exact temporary software copy",
                        "status": "completed",
                        "conclusion": "skipped",
                    },
                ],
            }
        ],
    }


class PriorSourceStopTests(unittest.TestCase):
    def reject(self, value):
        with self.assertRaises(acquisition.AcquisitionError) as caught:
            acquisition.prior_source_stop(value)
        self.assertEqual(caught.exception.args, ("acquisition_rejected",))

    def test_exact_first_attempt_unit_stop_and_skipped_downstream_accepted(self):
        value = good_prior_source_stop()
        original = copy.deepcopy(value)
        self.assertIsNone(acquisition.prior_source_stop(value))
        self.assertEqual(value, original)

    def test_any_acquisition_native_or_retirement_completion_rejected(self):
        for index in (1, 2, 3):
            for conclusion in ("success", "failure", "cancelled", "timed_out", None):
                with self.subTest(stage=index, conclusion=conclusion):
                    value = good_prior_source_stop()
                    value["jobs"][0]["steps"][index]["conclusion"] = conclusion
                    self.reject(value)

    def test_any_downstream_running_or_queued_stage_rejected(self):
        for index in (1, 2, 3):
            for status in ("in_progress", "queued", "pending", None):
                value = good_prior_source_stop()
                value["jobs"][0]["steps"][index]["status"] = status
                self.reject(value)

    def test_unit_step_must_have_failed_and_completed(self):
        for field, items in (
            ("conclusion", ("success", "skipped", "cancelled", None)),
            ("status", ("in_progress", "queued", None)),
        ):
            for item in items:
                value = good_prior_source_stop()
                value["jobs"][0]["steps"][0][field] = item
                self.reject(value)

    def test_wrong_job_run_attempt_sha_or_name_rejected(self):
        for field, items in {
            "id": (102487798349, "102487798348", 102487798348.0, True),
            "run_id": (34358096761, "34358096760", 34358096760.0, True),
            "run_attempt": (0, 2, "1", 1.0, True),
            "head_sha": ("0" * 40, acquisition.ACCEPTED_COMMIT, None),
            "name": ("other-job", None),
        }.items():
            for item in items:
                value = good_prior_source_stop()
                value["jobs"][0][field] = item
                self.reject(value)

    def test_job_must_be_completed_failure_not_success_or_unfinished(self):
        for field, items in (
            ("status", ("in_progress", "queued", None)),
            ("conclusion", ("success", "skipped", "cancelled", None)),
        ):
            for item in items:
                value = good_prior_source_stop()
                value["jobs"][0][field] = item
                self.reject(value)

    def test_missing_required_job_fields_rejected(self):
        for field in good_prior_source_stop()["jobs"][0]:
            value = good_prior_source_stop()
            del value["jobs"][0][field]
            self.reject(value)

    def test_nonint_or_nonexact_job_count_rejected(self):
        for count in (None, 0, 2, True, 1.0, "1"):
            value = good_prior_source_stop()
            value["total_count"] = count
            self.reject(value)

    def test_wrong_top_level_or_job_shapes_rejected(self):
        for value in (
            None,
            [],
            "jobs",
            {},
            {"total_count": 1},
            {"total_count": 1, "jobs": []},
            {"total_count": 1, "jobs": [None]},
        ):
            self.reject(value)
        value = good_prior_source_stop()
        value["jobs"].append(copy.deepcopy(value["jobs"][0]))
        self.reject(value)

    def test_duplicate_stage_names_rejected(self):
        value = good_prior_source_stop()
        value["jobs"][0]["steps"].append(copy.deepcopy(value["jobs"][0]["steps"][1]))
        self.reject(value)

    def test_each_missing_required_stage_rejected_even_with_four_rows(self):
        for index in range(4):
            value = good_prior_source_stop()
            del value["jobs"][0]["steps"][index]
            value["jobs"][0]["steps"].append(
                {
                    "name": "Other unrelated step",
                    "status": "completed",
                    "conclusion": "success",
                }
            )
            self.reject(value)

    def test_malformed_or_excessive_step_rows_rejected(self):
        for steps in (None, {}, [], [None] * 4, [{"name": 1}] * 4):
            value = good_prior_source_stop()
            value["jobs"][0]["steps"] = steps
            self.reject(value)
        value = good_prior_source_stop()
        value["jobs"][0]["steps"].extend({"name": f"extra-{i}"} for i in range(29))
        self.reject(value)


class MetadataTests(unittest.TestCase):
    def reject(self, value):
        with self.assertRaises(acquisition.AcquisitionError) as caught:
            acquisition.metadata(value)
        self.assertEqual(caught.exception.args, ("acquisition_rejected",))

    def test_exact_bound_metadata_accepted(self):
        self.assertIsNone(acquisition.metadata(good_metadata()))

    def test_required_top_level_fields_cannot_be_missing(self):
        for field in good_metadata():
            value = good_metadata()
            del value[field]
            self.reject(value)

    def test_wrong_artifact_id_name_size_and_digest_rejected(self):
        for field, rejected in {
            "id": (True, str(acquisition.ARTIFACT_ID), acquisition.ARTIFACT_ID + 1),
            "name": (None, "different-artifact"),
            "size_in_bytes": (True, acquisition.ZIP_BYTES - 1, acquisition.TAR_BYTES),
            "digest": (None, acquisition.ZIP_SHA256, "sha256:" + "0" * 64),
        }.items():
            for item in rejected:
                value = good_metadata()
                value[field] = item
                self.reject(value)

    def test_expired_requires_exact_false_not_false_like_values(self):
        for item in (True, None, 0, "false", [], {}):
            value = good_metadata()
            value["expired"] = item
            self.reject(value)

    def test_wrong_workflow_run_identity_or_source_rejected(self):
        for field, rejected in {
            "id": (True, acquisition.ACCEPTED_RUN_ID + 1, str(acquisition.ACCEPTED_RUN_ID)),
            "head_sha": ("0" * 40, None),
            "head_branch": ("main", "refs/heads/codex/lumi-posttraining-product-v0.1"),
            "repository_id": (True, 1300831105, "1300831104"),
            "head_repository_id": (True, 1300831105, "1300831104"),
        }.items():
            for item in rejected:
                value = good_metadata()
                value["workflow_run"][field] = item
                self.reject(value)

    def test_required_workflow_fields_cannot_be_missing(self):
        for field in good_metadata()["workflow_run"]:
            value = good_metadata()
            del value["workflow_run"][field]
            self.reject(value)

    def test_integral_floats_do_not_satisfy_numeric_identity_contract(self):
        for field in ("id", "size_in_bytes"):
            value = good_metadata()
            value[field] = float(value[field])
            self.reject(value)
        for field in ("id", "repository_id", "head_repository_id"):
            value = good_metadata()
            value["workflow_run"][field] = float(value["workflow_run"][field])
            self.reject(value)

    def test_non_mapping_shapes_rejected(self):
        for value in (None, [], "metadata", True):
            self.reject(value)
        for item in (None, [], "run", True):
            value = good_metadata()
            value["workflow_run"] = item
            self.reject(value)

    def test_unrelated_api_fields_do_not_replace_bound_fields(self):
        value = good_metadata()
        value["created_at"] = "synthetic-unused"
        value["workflow_run"]["extra"] = {"ignored": True}
        before = copy.deepcopy(value)
        acquisition.metadata(value)
        self.assertEqual(value, before)


class SignedUrlTests(unittest.TestCase):
    def test_only_approved_https_suffixes_and_optional_443(self):
        for value in (
            SIGNED,
            "https://a.actions.githubusercontent.com/path?sig=synthetic",
            "https://a.blob.core.windows.net:443/path",
        ):
            self.assertEqual(acquisition.signed_url(value), value)

    def test_http_file_ftp_and_relative_urls_rejected(self):
        for value in (
            "http://a.blob.core.windows.net/path",
            "file:///tmp/accepted.zip",
            "ftp://a.actions.githubusercontent.com/path",
            "//a.blob.core.windows.net/x",
        ):
            with self.assertRaises(acquisition.AcquisitionError):
                acquisition.signed_url(value)

    def test_suffix_confusion_unapproved_domain_and_ip_rejected(self):
        for host in (
            "blob.core.windows.net",
            "actions.githubusercontent.com",
            "api.github.com",
            "evilblob.core.windows.net",
            "a.blob.core.windows.net.evil.example",
            "a.actions.githubusercontent.com.evil.example",
            "127.0.0.1",
            "[::1]",
        ):
            with self.assertRaises(acquisition.AcquisitionError):
                acquisition.signed_url("https://" + host + "/artifact")

    def test_userinfo_password_non443_port_and_fragment_rejected(self):
        for value in (
            "https://user@a.blob.core.windows.net/p",
            "https://user:password@a.blob.core.windows.net/p",
            "https://a.blob.core.windows.net:444/p",
            "https://a.actions.githubusercontent.com/p#fragment",
        ):
            with self.assertRaises(acquisition.AcquisitionError):
                acquisition.signed_url(value)

    def test_malformed_url_port_or_brackets_rejected(self):
        for value in ("https://a.blob.core.windows.net:invalid/p", "https://[bad/p"):
            with self.assertRaises(ValueError):
                acquisition.signed_url(value)

    def test_type_and_length_cap_rejected(self):
        for value in (None, b"https://a.blob.core.windows.net/p", "", SIGNED + "a" * 16384):
            with self.assertRaises(acquisition.AcquisitionError):
                acquisition.signed_url(value)

    def test_redirect_handler_never_constructs_followup_request(self):
        handler = acquisition.NoRedirect()
        self.assertIsNone(handler.redirect_request(None, None, 302, None, {}, SIGNED))
        self.assertIsNone(handler.redirect_request(None, None, 307, None, {}, SIGNED))


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.clock = self.enterContext(patch.object(acquisition.time, "monotonic", return_value=1))

    def test_exact_count_and_digest_with_bounded_read_sizes(self):
        data = b"synthetic" * 10000
        source, target = io.BytesIO(data), io.BytesIO()
        read = Mock(wraps=source.read)
        source.read = read
        self.assertEqual(
            acquisition.stream(source, target, len(data), hashlib.sha256(data).hexdigest(), 10),
            len(data),
        )
        self.assertEqual(target.getvalue(), data)
        self.assertTrue(all(0 < item.args[0] <= 65536 for item in read.call_args_list))
        self.assertEqual(read.call_args_list[-1].args, (1,))

    def test_oversized_input_is_rejected_before_overflow_write(self):
        target = io.BytesIO()
        with self.assertRaises(acquisition.AcquisitionError):
            acquisition.stream(
                io.BytesIO(b"abcd"), target, 3, hashlib.sha256(b"abc").hexdigest(), 10
            )
        self.assertEqual(target.getvalue(), b"")

    def test_short_input_and_bad_digest_rejected(self):
        for data, maximum, digest in (
            (b"ab", 3, hashlib.sha256(b"abc").hexdigest()),
            (b"abc", 3, "0" * 64),
        ):
            with self.assertRaises(acquisition.AcquisitionError):
                acquisition.stream(io.BytesIO(data), io.BytesIO(), maximum, digest, 10)

    def test_short_write_does_not_retry(self):
        target = Mock()
        target.write.return_value = 2
        with self.assertRaises(acquisition.AcquisitionError):
            acquisition.stream(
                io.BytesIO(b"abc"), target, 3, hashlib.sha256(b"abc").hexdigest(), 10
            )
        target.write.assert_called_once_with(b"abc")

    def test_deadline_prevents_initial_io(self):
        self.clock.return_value = 10
        source, target = Mock(), Mock()
        with self.assertRaises(acquisition.AcquisitionError):
            acquisition.stream(source, target, 3, "0" * 64, 10)
        source.read.assert_not_called()
        target.write.assert_not_called()

    def test_final_eof_read_cannot_cross_deadline_and_still_succeed(self):
        source = io.BytesIO(b"abc")
        original = source.read

        def read(size):
            data = original(size)
            if not data:
                self.clock.return_value = 11
            return data

        source.read = read
        with self.assertRaises(acquisition.AcquisitionError):
            acquisition.stream(source, io.BytesIO(), 3, hashlib.sha256(b"abc").hexdigest(), 10)

    def test_request_timeout_is_clamped_to_remaining_budget(self):
        self.assertEqual(acquisition.request_timeout(10, 20), 9)
        self.assertEqual(acquisition.request_timeout(100, 20), 20)
        self.clock.return_value = 10
        with self.assertRaises(acquisition.AcquisitionError):
            acquisition.request_timeout(10, 20)


class Response(io.BytesIO):
    def __init__(self, data, *, status=200, headers=None):
        super().__init__(data)
        self.status, self.headers = status, dict(headers or {})


class AcquisitionWithSyntheticIOTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(
            self.enterContext(tempfile.TemporaryDirectory(prefix="oci-acq-test-"))
        )
        self.output = self.directory / "reports"
        self.output.mkdir()
        self.input_parent = self.directory / "synthetic-input"
        self.input_parent.mkdir(mode=0o700)
        self.github_output = self.directory / "github-output"
        self.tar = b"generated-software-only" * 4
        self.zip_buffer = io.BytesIO()
        with zipfile.ZipFile(self.zip_buffer, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("cpu-release-image.tar", self.tar)
        self.zip_data = self.zip_buffer.getvalue()
        for name, value in {
            "ZIP_BYTES": len(self.zip_data),
            "ZIP_SHA256": hashlib.sha256(self.zip_data).hexdigest(),
            "TAR_BYTES": len(self.tar),
            "TAR_SHA256": hashlib.sha256(self.tar).hexdigest(),
        }.items():
            self.enterContext(patch.object(acquisition, name, value))
        self.enterContext(patch.object(acquisition, "STAGE", "preflight"))
        self.environment = {
            "GH_TOKEN": "synthetic-api-token",
            "GITHUB_OUTPUT": str(self.github_output),
        }
        self.enterContext(patch.object(acquisition.os, "environ", self.environment))
        self.enterContext(
            patch.object(acquisition.tempfile, "mkdtemp", return_value=str(self.input_parent))
        )
        self.opener = Mock()
        self.builder = self.enterContext(
            patch.object(acquisition.urllib.request, "build_opener", return_value=self.opener)
        )
        self.native_network = self.enterContext(
            patch.object(
                acquisition.urllib.request, "urlopen", side_effect=AssertionError("no_network")
            )
        )
        self.process = self.enterContext(
            patch.object(acquisition.subprocess, "run", side_effect=AssertionError("no_process"))
        )
        self.preflight = self.enterContext(
            patch.object(acquisition, "preflight", side_effect=AssertionError("no_preflight"))
        )
        self.main = self.enterContext(
            patch.object(acquisition, "main", side_effect=AssertionError("no_entrypoint"))
        )
        self.responses = []
        self.configure()

    def tearDown(self):
        for response in self.responses:
            response.close()
        self.native_network.assert_not_called()
        self.process.assert_not_called()
        self.preflight.assert_not_called()
        self.main.assert_not_called()

    def configure(
        self,
        *,
        prior_value=None,
        metadata_value=None,
        download_status=200,
        length=None,
        download_data=None,
        redirect=SIGNED,
    ):
        predecessor = Response(
            json.dumps(good_prior_source_stop() if prior_value is None else prior_value).encode()
        )
        first = Response(
            json.dumps(good_metadata() if metadata_value is None else metadata_value).encode()
        )
        redirect_error = urllib.error.HTTPError(
            "https://api.github.com/synthetic",
            302,
            "synthetic",
            {"Location": redirect},
            io.BytesIO(),
        )
        last = Response(
            self.zip_data if download_data is None else download_data,
            status=download_status,
            headers={"Content-Length": str(len(self.zip_data)) if length is None else length},
        )
        self.responses.extend((predecessor, first, redirect_error, last))
        self.opener.open.side_effect = (predecessor, first, redirect_error, last)

    def test_four_gets_prior_stop_first_and_no_auth_on_signed_request(self):
        result = acquisition.acquire(self.output)
        self.assertEqual(result["https_requests"], 4)
        self.assertEqual(result["prior_source_stop_run"], 34358096760)
        self.assertIs(result["prior_artifact_acquisition_started"], False)
        self.assertIs(result["prior_native_pass_started"], False)
        self.assertEqual(self.opener.open.call_count, 4)
        requests = [item.args[0] for item in self.opener.open.call_args_list]
        self.assertTrue(all(request.get_method() == "GET" for request in requests))
        self.assertTrue(all(request.data is None for request in requests))
        self.assertEqual(
            requests[0].full_url,
            (
                "https://api.github.com/repos/saidi19/lumi-model/actions/runs/"
                "34358096760/attempts/1/jobs?per_page=2"
            ),
        )
        self.assertEqual(
            requests[1].full_url,
            ("https://api.github.com/repos/saidi19/lumi-model/actions/artifacts/10101633554"),
        )
        self.assertEqual(requests[2].full_url, requests[1].full_url + "/zip")
        for request in requests[:3]:
            self.assertEqual(request.get_header("Authorization"), "Bearer synthetic-api-token")
        self.assertIsNone(requests[3].get_header("Authorization"))
        self.assertIsNone(requests[3].get_header("Cookie"))
        self.assertEqual(requests[3].full_url, SIGNED)
        self.assertEqual(self.builder.call_args.args[0].proxies, {})
        self.assertIsInstance(self.builder.call_args.args[1], acquisition.NoRedirect)
        self.assertNotIn("GH_TOKEN", self.environment)

    def test_verified_tar_handoff_and_private_mode_no_sensitive_report_values(self):
        result = acquisition.acquire(self.output)
        software = self.input_parent / "cpu-release-image.tar"
        self.assertEqual(software.read_bytes(), self.tar)
        self.assertEqual(stat.S_IMODE(software.stat().st_mode), 0o400)
        self.assertFalse((self.input_parent / "artifact.zip").exists())
        file_info, parent_info = software.stat(), self.input_parent.stat()
        self.assertEqual(
            self.github_output.read_text(),
            (
                "image_archive="
                + str(software)
                + "\n"
                + f"image_identity={file_info.st_dev}:{file_info.st_ino}\n"
                + f"parent_identity={parent_info.st_dev}:{parent_info.st_ino}\n"
            ),
        )
        raw = (self.output / "acquisition.json").read_bytes()
        self.assertEqual(raw, acquisition.canonical(result))
        for private in (SIGNED, "synthetic-api-token", str(self.input_parent)):
            self.assertNotIn(private.encode(), raw)
        for flag in (
            "real_model_or_data_access",
            "gpu_used",
            "real_sft_execution_permitted",
            "production_eligible",
            "prior_artifact_acquisition_started",
            "prior_native_pass_started",
        ):
            self.assertIs(result[flag], False)

    def test_expired_metadata_stops_before_archive_request(self):
        value = good_metadata()
        value["expired"] = True
        self.configure(metadata_value=value)
        with self.assertRaises(acquisition.AcquisitionError):
            acquisition.acquire(self.output)
        self.assertEqual(self.opener.open.call_count, 2)

    def test_prior_native_started_stops_before_artifact_metadata_request(self):
        value = good_prior_source_stop()
        value["jobs"][0]["steps"][2]["conclusion"] = "failure"
        self.configure(prior_value=value)
        with self.assertRaises(acquisition.AcquisitionError):
            acquisition.acquire(self.output)
        self.assertEqual(self.opener.open.call_count, 1)
        self.assertFalse(self.github_output.exists())
        self.assertEqual(acquisition.STAGE, "prior_source_stop")

    def test_prior_acquisition_started_stops_before_artifact_metadata_request(self):
        value = good_prior_source_stop()
        value["jobs"][0]["steps"][1]["status"] = "in_progress"
        self.configure(prior_value=value)
        with self.assertRaises(acquisition.AcquisitionError):
            acquisition.acquire(self.output)
        self.assertEqual(self.opener.open.call_count, 1)
        self.assertFalse(self.github_output.exists())

    def test_unapproved_redirect_stops_before_signed_request(self):
        self.configure(redirect="https://unapproved.example/archive")
        with self.assertRaises(acquisition.AcquisitionError):
            acquisition.acquire(self.output)
        self.assertEqual(self.opener.open.call_count, 3)

    def test_wrong_content_length_rejected_without_writing_archive(self):
        self.configure(length=str(len(self.zip_data) + 1))
        with self.assertRaises(acquisition.AcquisitionError):
            acquisition.acquire(self.output)
        self.assertFalse(self.input_parent.exists())
        self.assertFalse(self.github_output.exists())

    def test_corrupt_zip_bytes_cleanup_exact_private_file_without_handoff(self):
        altered = bytes([self.zip_data[0] ^ 1]) + self.zip_data[1:]
        self.configure(download_data=altered)
        with self.assertRaises(acquisition.AcquisitionError):
            acquisition.acquire(self.output)
        self.assertFalse(self.input_parent.exists())
        self.assertFalse(self.github_output.exists())

    def test_second_tar_hash_pin_independently_required(self):
        with patch.object(acquisition, "TAR_SHA256", "0" * 64):
            with self.assertRaises(acquisition.AcquisitionError):
                acquisition.acquire(self.output)
        self.assertFalse(self.input_parent.exists())
        self.assertFalse(self.github_output.exists())


class RetirementWithMetadataDoublesTests(unittest.TestCase):
    def setUp(self):
        self.path = "/tmp/lumi-oci-guardian-input-abcdefgh/cpu-release-image.tar"
        self.file_info = types.SimpleNamespace(
            st_dev=7,
            st_ino=19,
            st_mode=stat.S_IFREG | 0o400,
            st_uid=1001,
            st_gid=1001,
            st_size=acquisition.TAR_BYTES,
            st_nlink=1,
        )
        self.parent_info = types.SimpleNamespace(
            st_dev=7,
            st_ino=18,
            st_mode=stat.S_IFDIR | 0o700,
            st_uid=1001,
            st_gid=1001,
        )
        self.parent = types.SimpleNamespace(
            lstat=Mock(return_value=self.parent_info), rmdir=Mock(), iterdir=Mock()
        )
        self.selected = types.SimpleNamespace(
            parent=self.parent, lstat=Mock(return_value=self.file_info), unlink=Mock()
        )
        self.parent.iterdir.return_value = [self.selected]
        self.paths = self.enterContext(
            patch.object(acquisition, "Path", return_value=self.selected)
        )
        self.enterContext(patch.object(acquisition, "sys", types.SimpleNamespace(platform="linux")))
        self.uid = self.enterContext(patch.object(acquisition.os, "geteuid", return_value=1001))
        self.enterContext(patch.object(acquisition.os, "getegid", return_value=1001))
        self.environment = {
            "GITHUB_ACTIONS": "true",
            "GITHUB_REPOSITORY": acquisition.REPOSITORY,
            "LUMI_OCI_GUARDIAN_CI": "1",
        }
        self.enterContext(patch.object(acquisition.os, "environ", self.environment))

    def run_retirement(self, path=None, expected_file="7:19", expected_parent="7:18"):
        return acquisition.retire_input(
            self.path if path is None else path, expected_file, expected_parent
        )

    def reject(self, **changes):
        with self.assertRaises(acquisition.AcquisitionError):
            self.run_retirement(**changes)
        self.selected.unlink.assert_not_called()
        self.parent.rmdir.assert_not_called()

    def test_exact_private_pair_retired_once_with_fixed_receipt(self):
        self.assertEqual(
            self.run_retirement(),
            {
                "schema": acquisition.SCHEMA,
                "status": "passed",
                "owned_input_files_retired": 1,
                "owned_input_directories_retired": 1,
                "real_sft_execution_permitted": False,
                "production_eligible": False,
            },
        )
        self.paths.assert_called_once_with(self.path)
        self.selected.unlink.assert_called_once_with()
        self.parent.rmdir.assert_called_once_with()

    def test_non_linux_runtime_or_root_uid_rejected_before_path_access(self):
        acquisition.sys.platform = "darwin"
        self.reject()
        acquisition.sys.platform = "linux"
        self.uid.return_value = 0
        self.reject()
        self.paths.assert_not_called()

    def test_missing_ci_context_denied_before_path_access(self):
        self.environment["LUMI_OCI_GUARDIAN_CI"] = "0"
        self.reject()
        self.paths.assert_not_called()

    def test_broad_nested_foreign_and_symlink_lexical_paths_denied(self):
        for path in (
            "/tmp",
            "/",
            "/tmp/foreign/cpu-release-image.tar",
            "/tmp/lumi-oci-guardian-input-abcdefgh/../cpu-release-image.tar",
            "/tmp/lumi-oci-guardian-input-abcdefgh/nested/cpu-release-image.tar",
            "/tmp/lumi-oci-guardian-input-abcdefgh/other.tar",
            "/tmp/lumi-oci-guardian-input-abcdefgh/cpu-release-image.tar\n",
        ):
            self.reject(path=path)
        self.paths.assert_not_called()

    def test_identity_string_shapes_and_mismatch_denied(self):
        for value in ("", "7:0", "-7:19", "7:19:1", "7:19\n", "7:20"):
            self.reject(expected_file=value)
        self.reject(expected_parent="7:20")

    def test_file_mode_size_owner_type_and_links_must_match(self):
        for field, value in (
            ("st_mode", stat.S_IFREG | 0o600),
            ("st_mode", stat.S_IFLNK | 0o400),
            ("st_size", acquisition.TAR_BYTES - 1),
            ("st_uid", 0),
            ("st_gid", 0),
            ("st_nlink", 2),
        ):
            original = getattr(self.file_info, field)
            setattr(self.file_info, field, value)
            self.reject()
            setattr(self.file_info, field, original)

    def test_parent_must_be_private_owned_real_directory(self):
        for field, value in (
            ("st_mode", stat.S_IFDIR | 0o755),
            ("st_mode", stat.S_IFLNK | 0o700),
            ("st_uid", 0),
            ("st_gid", 0),
        ):
            original = getattr(self.parent_info, field)
            setattr(self.parent_info, field, value)
            self.reject()
            setattr(self.parent_info, field, original)

    def test_foreign_extra_parent_entry_blocks_all_deletion(self):
        self.parent.iterdir.return_value = [self.selected, object()]
        self.reject()


class SourceContractTests(unittest.TestCase):
    def test_module_import_has_no_ml(self):
        blocked = {"torch", "transformers", "numpy", "tensorflow", "jax", "safetensors"}
        self.assertFalse(any(name.split(".", 1)[0] in blocked for name in IMPORT_ADDITIONS))

    def test_zip_entry_checks_and_hash_before_zip_parser_are_present(self):
        tree = ast.parse(SOURCE.read_text())
        acquire = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "acquire"
        )
        source = ast.unparse(acquire)
        for expression in (
            "len(entries) == 1",
            "entry.filename == 'cpu-release-image.tar'",
            "entry.file_size == TAR_BYTES",
            "entry.compress_size <= ZIP_BYTES",
            "entry.flag_bits & 1",
            "entry.compress_type in (0, 8)",
            "entry.external_attr >> 16",
        ):
            self.assertIn(expression, source)
        self.assertLess(
            source.index("stream(response, target, ZIP_BYTES, ZIP_SHA256, deadline)"),
            source.index("zipfile.ZipFile(archive, 'r')"),
        )
        self.assertIn("stream(source, target, TAR_BYTES, TAR_SHA256, deadline)", source)

    def test_no_broad_archive_extract_or_recursive_delete(self):
        tree = ast.parse(SOURCE.read_text())
        forbidden = {"extractall", "rmtree", "system", "Popen"}
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in forbidden
                for node in ast.walk(tree)
            )
        )

    def test_entrypoint_is_guarded_and_preflight_requires_fresh_ci_attempt(self):
        tree = ast.parse(SOURCE.read_text())
        entry = tree.body[-1]
        self.assertIsInstance(entry, ast.If)
        self.assertEqual(ast.unparse(entry.test), "__name__ == '__main__'")
        preflight = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "preflight"
        )
        source = ast.unparse(preflight)
        for expression in (
            "sys.platform == 'linux'",
            "sys.version_info[:2] == (3, 12)",
            "sys.flags.no_site",
            "sys.flags.isolated",
            "sys.dont_write_bytecode",
            "os.geteuid() != 0",
            "os.environ.get('GITHUB_RUN_ATTEMPT') == '1'",
            "git('rev-parse', 'HEAD^') == APPROVED_PARENT",
        ):
            self.assertIn(expression, source)


if __name__ == "__main__":
    unittest.main()
