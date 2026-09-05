from __future__ import annotations

import ast
import hashlib
import tempfile
import unittest
from email.message import Message
from pathlib import Path

import lumi.posttraining_bedstc2_authoritative_rights_acquisition_v1 as contract


class FakeResponse:
    def __init__(
        self,
        url: str,
        payload: bytes,
        media_type: str,
        *,
        status: int = 200,
        encoding: str = "identity",
    ) -> None:
        self.status = status
        self._url = url
        self._payload = payload
        self._offset = 0
        self.headers = Message()
        self.headers["Content-Type"] = media_type
        self.headers["Content-Length"] = str(len(payload))
        self.headers["Content-Encoding"] = encoding

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def geturl(self) -> str:
        return self._url

    def read(self, size: int) -> bytes:
        value = self._payload[self._offset : self._offset + size]
        self._offset += len(value)
        return value


class FakeOpener:
    def __init__(self, response: FakeResponse, budget: contract.NetworkBudget) -> None:
        self.response = response
        self.budget = budget
        self.calls = 0

    def open(self, request: object, timeout: int) -> FakeResponse:
        self.calls += 1
        if getattr(request, "method") != "GET" or timeout != 45:
            raise AssertionError("unexpected synthetic request")
        return self.response


class MultiOpener:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.calls = 0

    def open(self, request: object, timeout: int) -> FakeResponse:
        self.calls += 1
        url = getattr(request, "full_url")
        spec = next(item for item in contract.EXPECTED_DOCUMENTS if item["url"] == url)
        return FakeResponse(url, self.payloads[spec["id"]], spec["media_type"])


class AcquisitionContractTests(unittest.TestCase):
    def test_exact_authorization_binding(self) -> None:
        payload = contract.EXPECTED_AUTHORIZATION_TEXT.encode("utf-8")
        self.assertEqual(len(payload), 472)
        self.assertEqual(hashlib.sha256(payload).hexdigest(), contract.EXPECTED_AUTHORIZATION_SHA256)

    def test_exact_caps_and_document_allowlist(self) -> None:
        self.assertEqual(contract.EXPECTED_LIMITS["maximum_documents"], 12)
        self.assertEqual(contract.EXPECTED_LIMITS["maximum_http_responses"], 24)
        self.assertEqual(contract.EXPECTED_LIMITS["maximum_inbound_bytes"], 16 * 1024 * 1024)
        self.assertEqual(contract.EXPECTED_LIMITS["maximum_combined_write_bytes"], 20 * 1024 * 1024)
        self.assertEqual(len(contract.EXPECTED_DOCUMENTS), 8)
        self.assertEqual(len({item["url"] for item in contract.EXPECTED_DOCUMENTS}), 8)
        for item in contract.EXPECTED_DOCUMENTS:
            self.assertTrue(item["url"].startswith("https://"))
            self.assertNotIn("?", item["url"])
            self.assertFalse(item["filename"].endswith((".zip", ".tar", ".gz", ".wav", ".mp3")))

    def test_revision_pinned_repository_urls(self) -> None:
        urls = "\n".join(item["url"] for item in contract.EXPECTED_DOCUMENTS)
        self.assertIn("7897c9beb5fe21fdfb57dcbe1e8f8294c2bcdead", urls)
        self.assertIn("9df1d4dc800548a883f8bc1a9ce4116c77aebc02", urls)
        self.assertNotIn("/master/", urls)
        self.assertNotIn("/main/", urls)

    def test_url_gate_rejects_non_allowlisted_and_ip_literal(self) -> None:
        for url in (
            "http://aclanthology.org/W14-4337/",
            "https://127.0.0.1/W14-4337/",
            "https://user@aclanthology.org/W14-4337/",
            "https://aclanthology.org:444/W14-4337/",
            "https://aclanthology.org/W14-4337/#fragment",
        ):
            with self.assertRaises(ValueError):
                contract._validate_url(url)

    def test_valid_synthetic_text_download(self) -> None:
        spec = contract.EXPECTED_DOCUMENTS[0]
        payload = b"<html>Dialog State Tracking Challenge</html>"
        budget = contract.NetworkBudget()
        response = FakeResponse(spec["url"], payload, spec["media_type"])
        opener = FakeOpener(response, budget)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / spec["filename"]
            receipt = contract.acquire_one_document(spec, path, budget, opener)
            self.assertEqual(receipt["bytes"], len(payload))
            self.assertEqual(receipt["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(path.stat().st_nlink, 1)
        self.assertEqual(budget.requests_started, 1)
        self.assertEqual(budget.responses_observed, 1)
        self.assertEqual(budget.inbound_bytes, len(payload))

    def test_rejects_wrong_mime_compression_final_url_and_archive(self) -> None:
        spec = contract.EXPECTED_DOCUMENTS[0]
        cases = [
            FakeResponse(spec["url"], b"hello", "application/octet-stream"),
            FakeResponse(spec["url"], b"hello", spec["media_type"], encoding="gzip"),
            FakeResponse(spec["url"] + "x", b"hello", spec["media_type"]),
            FakeResponse(spec["url"], b"PK\x03\x04archive", spec["media_type"]),
        ]
        for response in cases:
            with self.subTest(response=response):
                budget = contract.NetworkBudget()
                with tempfile.TemporaryDirectory() as directory:
                    with self.assertRaises(ValueError):
                        contract.acquire_one_document(
                            spec,
                            Path(directory) / spec["filename"],
                            budget,
                            FakeOpener(response, budget),
                        )

    def test_pdf_magic_required(self) -> None:
        spec = next(item for item in contract.EXPECTED_DOCUMENTS if item["media_type"] == "application/pdf")
        budget = contract.NetworkBudget()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                contract.acquire_one_document(
                    spec,
                    Path(directory) / spec["filename"],
                    budget,
                    FakeOpener(FakeResponse(spec["url"], b"not-pdf", "application/pdf"), budget),
                )

    def test_budget_rejects_twenty_fifth_response_and_byte_overrun(self) -> None:
        budget = contract.NetworkBudget(maximum_responses=24, maximum_bytes=10)
        for _ in range(24):
            budget.start_request()
            budget.observe_response()
        with self.assertRaises(ValueError):
            budget.start_request()
        budget.add_bytes(10)
        with self.assertRaises(ValueError):
            budget.add_bytes(1)

    def test_refuses_existing_or_symlink_destination(self) -> None:
        spec = contract.EXPECTED_DOCUMENTS[0]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target"
            target.write_text("existing")
            link = Path(directory) / spec["filename"]
            link.symlink_to(target)
            budget = contract.NetworkBudget()
            with self.assertRaises(FileExistsError):
                contract.acquire_one_document(
                    spec,
                    link,
                    budget,
                    FakeOpener(FakeResponse(spec["url"], b"text", spec["media_type"]), budget),
                )

    def test_runner_marker_precedes_network_function(self) -> None:
        source = Path("scripts/acquire_sft_bedstc2_authoritative_rights_documents_v1.py").read_text()
        tree = ast.parse(source)
        calls = [
            ast.unparse(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        ]
        self.assertLess(calls.index("contract.start_marker"), calls.index("contract.run_acquisition"))
        self.assertIn('"network_retries": 0', Path(contract.__file__).read_text())

    def test_full_synthetic_acquisition_and_inventory_replay(self) -> None:
        payload_by_id = {
            item["id"]: (
                b"%PDF-1.7 synthetic"
                if item["media_type"] == "application/pdf"
                else f"synthetic rights document {item['id']}".encode()
            )
            for item in contract.EXPECTED_DOCUMENTS
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plan = {
                "output": {
                    "private_root": "data/private/attempt-001",
                    "private_manifest_path": "data/private/attempt-001/artifact-manifest.json",
                    "receipt_path": "evidence/receipt.json",
                    "verification_path": "evidence/verification.json",
                }
            }
            opener = MultiOpener(payload_by_id)
            receipt = contract.run_acquisition(plan, root, opener=opener)
            self.assertEqual(opener.calls, 8)
            self.assertEqual(receipt["accounting"]["http_requests_started"], 8)
            manifest, replayed_receipt, payloads = contract.replay_acquired_inventory(plan, root)
            self.assertEqual(replayed_receipt, receipt)
            self.assertEqual(manifest["document_count"], 8)
            self.assertEqual(payloads, [payload_by_id[item["id"]] for item in contract.EXPECTED_DOCUMENTS])


if __name__ == "__main__":
    unittest.main()
