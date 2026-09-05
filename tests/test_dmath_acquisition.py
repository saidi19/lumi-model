from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from lumi.posttraining_dmath_acquisition import (
    EXACT_OBJECTS,
    download_exact_object,
    git_blob_sha1,
    load_dmath_acquisition_plan,
    summarize_train_payload,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/lumi-4b-sft-dmath-acquisition-v0.1.json"


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, url: str) -> None:
        super().__init__(payload)
        self.headers = {}
        self._url = url

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def geturl(self) -> str:
        return self._url


def test_plan_binds_only_three_train_side_objects() -> None:
    plan = load_dmath_acquisition_plan(PLAN, repository_root=ROOT)
    assert plan["authorization"]["test_split_access_authorized"] is False
    assert plan["authorization"]["paid_gpu_authorized"] is False
    assert [item["path"] for item in EXACT_OBJECTS] == [
        "LICENSE",
        "README.md",
        "dataset/train.json",
    ]


def test_download_rejects_redirect_and_blob_mismatch() -> None:
    payload = b"example"
    specification = {
        "git_blob_sha1": git_blob_sha1(payload),
        "kind": "data",
        "path": "dataset/train.json",
        "size_bytes": len(payload),
        "url": "https://example.invalid/pinned/train.json",
    }

    def redirected(_request: object) -> _Response:
        return _Response(payload, "https://example.invalid/other/train.json")

    with pytest.raises(ValueError, match="redirected"):
        download_exact_object(specification, maximum_network_bytes=64, open_url=redirected)

    changed = dict(specification, git_blob_sha1="0" * 40)

    def exact(_request: object) -> _Response:
        return _Response(payload, specification["url"])

    with pytest.raises(ValueError, match="Git blob"):
        download_exact_object(changed, maximum_network_bytes=64, open_url=exact)


def test_train_summary_rejects_wrong_row_count() -> None:
    with pytest.raises(ValueError, match="row count"):
        summarize_train_payload(json.dumps([{"field": "value"}]).encode())


def test_train_summary_rejects_noncanonical_mapping_keys() -> None:
    with pytest.raises(ValueError, match="record-key sequence"):
        summarize_train_payload(json.dumps({"0": {"field": "value"}}).encode())
