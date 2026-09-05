from __future__ import annotations

from lumi.posttraining_prosocial_dialog_safety_owner_review_v1 import (
    EXPECTED_OWNER_REQUEST,
    EXPECTED_OWNER_REQUEST_SHA256,
    EXPECTED_SAMPLE_SHA256,
)
from lumi.posttraining_prosocial_dialog_safety_curation_v1 import sha256_value
import hashlib


def test_owner_request_is_exactly_bound() -> None:
    assert hashlib.sha256(EXPECTED_OWNER_REQUEST.encode()).hexdigest() == (
        EXPECTED_OWNER_REQUEST_SHA256
    )


def test_sample_hash_is_a_sha256() -> None:
    assert len(EXPECTED_SAMPLE_SHA256) == 64
    assert sha256_value([]) != EXPECTED_SAMPLE_SHA256
