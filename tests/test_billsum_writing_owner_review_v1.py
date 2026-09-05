from __future__ import annotations

import hashlib

import pytest

from lumi.posttraining_billsum_writing_curation_v1 import sha256_value
from lumi.posttraining_billsum_writing_owner_review_v1 import (
    EXPECTED_AUTHORIZATION,
    EXPECTED_OWNER_REQUEST,
    EXPECTED_OWNER_REQUEST_SHA256,
    EXPECTED_SAMPLE_SHA256,
    build_verification,
)


def test_owner_request_is_exactly_bound() -> None:
    assert hashlib.sha256(EXPECTED_OWNER_REQUEST.encode()).hexdigest() == (
        EXPECTED_OWNER_REQUEST_SHA256
    )


def test_sample_hash_is_bound_and_not_empty() -> None:
    assert len(EXPECTED_SAMPLE_SHA256) == 64
    assert sha256_value([]) != EXPECTED_SAMPLE_SHA256


def test_authorization_only_allows_owner_decision_materialization() -> None:
    enabled = {name for name, value in EXPECTED_AUTHORIZATION.items() if value}
    assert enabled == {"human_owner_decision_materialization_authorized"}


def test_verifier_rejects_unsealed_receipt() -> None:
    with pytest.raises(ValueError, match="receipt hash changed"):
        build_verification(
            {"accepted_records": 32, "rejected_records": 0},
            receipt_file_sha256="0" * 64,
        )
