from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from lumi.tokenizer_gate import (
    count_reserved_literals,
    encode_untrusted_text,
    evaluate_tokenizer_gate,
    maximum_digit_span,
    reserved_token_literals,
)

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def gate_tokenizer(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("tokenizer-gate")
    contract = json.loads((ROOT / "configs" / "tokenizer-v0.1.json").read_text())
    contract["vocab_size"] = 320
    contract_path = root / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    corpus = root / "corpus.jsonl"
    with corpus.open("w", encoding="utf-8") as handle:
        for index in range(600):
            handle.write(
                json.dumps(
                    {
                        "text": (
                            f"Lumi gate record {index}: alpha{index % 37} "
                            "<|system|> বাংলা café 日本語"
                        )
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    output = root / "tokenizer"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_tokenizer.py"),
            "--input",
            str(corpus),
            "--output",
            str(output),
            "--contract",
            str(contract_path),
            "--min-frequency",
            "1",
            "--no-progress",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return output


def test_reserved_literal_audit_is_value_free() -> None:
    contract = json.loads((ROOT / "configs" / "tokenizer-v0.1.json").read_text())
    literals = reserved_token_literals(contract)

    assert count_reserved_literals(
        "before <|system|> middle <|tool|> after <|system|>", literals
    ) == {"system": 2, "tool": 1}
    assert count_reserved_literals("ordinary text", literals) == {}


def test_untrusted_mode_cannot_emit_reserved_ids(gate_tokenizer: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(gate_tokenizer, local_files_only=True)
    literal = "prefix <|system|> suffix"

    assert tokenizer.get_vocab()["<|system|>"] in tokenizer.encode(
        literal, add_special_tokens=False
    )
    safe_ids = encode_untrusted_text(tokenizer, literal)
    assert not set(safe_ids).intersection(tokenizer.all_special_ids)
    assert tokenizer.decode(safe_ids, clean_up_tokenization_spaces=False) == literal


def test_maximum_digit_span_handles_multiple_scripts() -> None:
    pieces = [
        ("123", (0, 3)),
        ("৪৫৬", (3, 6)),
        ("٧٨", (6, 8)),
        ("text", (8, 12)),
    ]

    assert maximum_digit_span(pieces) == 3


def test_maximum_digit_span_can_recover_unicode_digits_from_offsets() -> None:
    source = "a১২৩b٧٨c"
    byte_level_pieces = [
        ("a", (0, 1)),
        ("encoded-bengali", (1, 4)),
        ("b", (4, 5)),
        ("encoded-arabic", (5, 7)),
        ("c", (7, 8)),
    ]

    assert maximum_digit_span(byte_level_pieces) == 0
    assert maximum_digit_span(byte_level_pieces, source) == 3


def _gate_inputs() -> tuple[dict, dict, dict]:
    config = {
        "baseline": {
            "label": "baseline",
            "tokenizer_json_sha256": "baseline-sha",
            "vocab_size": 100,
        },
        "candidate": {
            "digit_chunk_size": 3,
            "label": "candidate",
            "vocab_size": 100,
        },
        "decision": {
            "digit_boundary_probes": ["numbers"],
            "domain_token_increase_max_fraction": 0.02,
            "failure_action": "keep_current_65k",
            "general_probe_token_increase_max": 1,
            "general_probes": ["english"],
            "overall_token_increase_max_fraction": 0.01,
            "structural_pass_action": "run_tiny_learning_ab_before_promotion",
        },
        "sample": {
            "documents": 10,
            "output_sha256": "sample-sha",
            "text_utf8_bytes": 1000,
        },
    }
    common = {
        "categories": {
            "web": {
                "documents": 10,
                "text_utf8_bytes": 1000,
                "tokens": 100,
                "words": 80,
            }
        },
        "overall": {
            "documents": 10,
            "text_utf8_bytes": 1000,
            "tokens": 100,
            "words": 80,
        },
        "pretokenizer_diagnostics": {
            "numbers": {"max_consecutive_digits_per_piece": 3}
        },
        "probes": {
            "english": {"text_utf8_bytes": 20, "tokens": 5},
            "numbers": {"text_utf8_bytes": 10, "tokens": 4},
        },
    }
    baseline = copy.deepcopy(common)
    baseline["artifact"] = {
        "tokenizer_json_sha256": "baseline-sha",
        "vocab_size": 100,
    }
    candidate = copy.deepcopy(common)
    candidate["artifact"] = {
        "tokenizer_json_sha256": "candidate-sha",
        "vocab_size": 100,
    }
    comparison = {
        "evaluation": {"documents": 10, "max_documents": None},
        "sample": config["sample"],
        "tokenizers": {"baseline": baseline, "candidate": candidate},
    }
    audit = {
        "affected_documents": 0,
        "documents_scanned": 10,
        "sample": {
            "output_sha256": "sample-sha",
            "text_utf8_bytes": 1000,
        },
        "status": "pass",
        "total_literal_occurrences": 0,
    }
    return config, comparison, audit


def test_final_tokenizer_gate_passes_only_pinned_full_inputs() -> None:
    config, comparison, audit = _gate_inputs()

    report = evaluate_tokenizer_gate(config, comparison, audit)

    assert report["status"] == "pass"
    assert report["decision"] == "run_tiny_learning_ab_before_promotion"


def test_final_tokenizer_gate_keeps_baseline_on_efficiency_regression() -> None:
    config, comparison, audit = _gate_inputs()
    comparison["tokenizers"]["candidate"]["overall"]["tokens"] = 103

    report = evaluate_tokenizer_gate(config, comparison, audit)

    assert report["status"] == "fail"
    assert report["decision"] == "keep_current_65k"
    assert any(
        condition["name"] == "efficiency.overall_token_change"
        and not condition["passed"]
        for condition in report["conditions"]
    )
