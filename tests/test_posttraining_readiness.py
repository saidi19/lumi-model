from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lumi.posttraining_readiness import (
    FINAL_CHECKPOINT_MANIFEST_SHA256,
    FINAL_HANDOFF_SHA256,
    FINAL_TOKENS,
    build_posttraining_readiness_report,
    load_final_base_binding,
    tokenizer_manifest_sha256,
    validate_final_base_binding,
)

ROOT = Path(__file__).parents[1]
BINDING_PATH = ROOT / "configs" / "lumi-4b-final-base-binding-v0.1.json"


def _load(name: str) -> dict:
    return json.loads((ROOT / "configs" / name).read_text())


def test_final_base_binding_is_exact_and_tokenizer_inventory_is_reproducible() -> None:
    binding = load_final_base_binding(BINDING_PATH)
    assert binding["checkpoint"]["training_tokens"] == FINAL_TOKENS
    assert binding["checkpoint"]["manifest_sha256"] == FINAL_CHECKPOINT_MANIFEST_SHA256
    assert binding["handoff"]["sha256"] == FINAL_HANDOFF_SHA256
    assert tokenizer_manifest_sha256(binding["tokenizer"]["files"]) == binding["tokenizer"][
        "manifest_sha256"
    ]
    assert tokenizer_manifest_sha256(list(reversed(binding["tokenizer"]["files"]))) == binding[
        "tokenizer"
    ]["manifest_sha256"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["checkpoint"].update(training_tokens=79_999_999_999),
        lambda value: value["checkpoint"].update(manifest_sha256="0" * 64),
        lambda value: value["handoff"].update(sha256="0" * 64),
        lambda value: value["s3_backup"].update(status="pending"),
        lambda value: value["tokenizer"]["files"][0].update(bytes=1),
    ],
)
def test_final_base_binding_rejects_drift(mutation) -> None:
    binding = json.loads(BINDING_PATH.read_text())
    mutation(binding)
    with pytest.raises(ValueError):
        validate_final_base_binding(binding)


def test_readiness_report_passes_only_contract_preparation() -> None:
    report = build_posttraining_readiness_report(
        base_binding=_load("lumi-4b-final-base-binding-v0.1.json"),
        chat_template=_load("lumi-4b-chat-template-v0.1.json"),
        evaluation_plan=_load("lumi-4b-posttraining-eval-v0.1.json"),
        sft_policy=_load("lumi-4b-sft-data-contract-v0.1.json"),
    )
    assert report["base_checkpoint_ready"] is True
    assert report["chat_template_ready"] is True
    assert report["sft_data_contract_ready"] is True
    assert len(report["chat_template_sha256"]) == 64
    assert len(report["evaluation_plan_sha256"]) == 64
    assert len(report["sft_data_contract_sha256"]) == 64
    assert report["paid_gpu_authorized"] is False
    assert report["next_gate"] == "curate_filter_deduplicate_and_seal_sft_dataset"
    assert "sft_dataset:not_curated_or_sealed" in report["blockers"]

    changed = copy.deepcopy(_load("lumi-4b-chat-template-v0.1.json"))
    changed["context_window"] = 8192
    with pytest.raises(ValueError, match="4096"):
        build_posttraining_readiness_report(
            base_binding=_load("lumi-4b-final-base-binding-v0.1.json"),
            chat_template=changed,
            evaluation_plan=_load("lumi-4b-posttraining-eval-v0.1.json"),
            sft_policy=_load("lumi-4b-sft-data-contract-v0.1.json"),
        )
