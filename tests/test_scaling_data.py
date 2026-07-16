from __future__ import annotations

import hashlib
import json
import sqlite3
from array import array
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from lumi import (
    PackedTokenDataset,
    build_scaling_packed_dataset,
    build_scaling_source_index,
    create_scaling_data_contract,
    validate_scaling_packed_manifest,
    validate_scaling_source_index,
)
from lumi.scaling_data import INDEX_DATABASE_FILENAME, PACK_STATE_FILENAME


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_scaling_canary_contract_is_sequence_and_shard_aligned() -> None:
    root = Path(__file__).parents[1]
    config = json.loads(
        (root / "configs" / "scaling-data-1b-v0.1.json").read_text(
            encoding="utf-8"
        )
    )
    target = config["canary"]["target_tokens"]
    sequence_length = config["packing"]["sequence_length"]
    shard_tokens = sequence_length * config["canary"]["sequences_per_shard"]

    assert config["schema_version"] == 1
    assert target == 16_777_216
    assert target % sequence_length == 0
    assert target % shard_tokens == 0
    final = config["final"]
    final_sequences = final["target_tokens_effective"] // sequence_length
    assert final["target_tokens_requested"] // sequence_length * sequence_length == final[
        "target_tokens_effective"
    ]
    assert (
        final_sequences + final["sequences_per_shard"] - 1
    ) // final["sequences_per_shard"] == final["expected_shards"]


def test_packed_dataset_reads_full_uint16_token_range(tmp_path: Path) -> None:
    shard = tmp_path / "shard-00000.bin"
    shard.write_bytes(array("H", [0, 65_535]).tobytes())
    manifest = {
        "build_fingerprint": "a" * 64,
        "format": {
            "dtype": "uint16",
            "endianness": "little",
            "layout": "flat_fixed_sequences",
            "token_bytes": 2,
        },
        "packing": {"sequence_length": 2},
        "schema_version": 1,
        "shards": [
            {
                "bytes": 4,
                "name": shard.name,
                "sequences": 1,
                "sha256": _sha256(shard),
                "tokens": 2,
            }
        ],
        "statistics": {"sequences": 1, "tokens": 2},
        "status": "complete",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    dataset = PackedTokenDataset(manifest_path)

    assert dataset.token_dtype == "uint16"
    assert dataset[0].tolist() == [0, 65_535]


@pytest.fixture
def scaling_fixture(tmp_path: Path) -> dict:
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    vocabulary = {
        "<|pad|>": 0,
        "<|bos|>": 1,
        "<|eos|>": 2,
        "<|doc|>": 3,
        "<|unk|>": 4,
        "alpha": 5,
        "beta": 6,
        "source": 7,
        "document": 8,
        "value": 9,
    }
    for index in range(10, 64):
        vocabulary[f"token{index}"] = index
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<|unk|>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        pad_token="<|pad|>",
        additional_special_tokens=["<|doc|>"],
    )
    tokenizer.save_pretrained(tokenizer_path)

    input_path = tmp_path / "source.jsonl"
    rows = []
    heldout_rows = []
    category_counts: dict[str, dict[str, int]] = {}
    near_a = hashlib.sha256(b"heldout-family-a").hexdigest()
    with input_path.open("w", encoding="utf-8") as handle:
        for category, component, marker in (
            ("category_a", "component_a", "alpha"),
            ("category_b", "component_b", "beta"),
        ):
            for index in range(24):
                text = f"{marker} source document token{10 + index} value token{40 + index}"
                document_id = f"{category}/{component}/{index}"
                content_sha256 = hashlib.sha256(text.encode()).hexdigest()
                near = (
                    near_a
                    if category == "category_a" and index in {2, 3}
                    else hashlib.sha256(f"near/{document_id}".encode()).hexdigest()
                )
                record = {
                    "content_sha256": content_sha256,
                    "id": document_id,
                    "provenance": {"category": category, "component": component},
                    "quality": {
                        "near_duplicate": {"fingerprint_sha256": near}
                    },
                    "text": text,
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                rows.append(record)
                text_tokens = len(
                    tokenizer.backend_tokenizer.encode(
                        text, add_special_tokens=False
                    ).ids
                )
                totals = category_counts.setdefault(category, {"documents": 0, "tokens": 0})
                totals["documents"] += 1
                totals["tokens"] += text_tokens
                if index == 2:
                    heldout_rows.append(
                        {
                            "category": category,
                            "component": component,
                            "content_sha256": content_sha256,
                            "document_id": document_id,
                            "near_duplicate_sha256": near,
                            "rank": "0" * 64,
                            "stratum_id": json.dumps(
                                [category, component], separators=(",", ":")
                            ),
                            "tokens_used": 1 + text_tokens,
                            "truncated": False,
                        }
                    )

    source_manifest_path = tmp_path / "source-manifest.json"
    source_manifest_path.write_text(
        json.dumps(
            {
                "output": {
                    "file_bytes": input_path.stat().st_size,
                    "sha256": _sha256(input_path),
                },
                "source_files": [
                    {
                        "category": category,
                        "component": component,
                        "statistics": {
                            "documents": 24,
                            "target_text_utf8_bytes": 1_000,
                        },
                        "used": True,
                    }
                    for category, component in (
                        ("category_a", "component_a"),
                        ("category_b", "component_b"),
                    )
                ],
                "statistics": {"documents": len(rows)},
                "status": "complete",
            }
        ),
        encoding="utf-8",
    )

    ledger_path = tmp_path / "heldout-selection.jsonl"
    with ledger_path.open("w", encoding="utf-8") as handle:
        for row in heldout_rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    heldout_manifest_path = tmp_path / "heldout-manifest.json"
    heldout_manifest_path.write_text(
        json.dumps(
            {
                "selection_ledger": {
                    "bytes": ledger_path.stat().st_size,
                    "records": len(heldout_rows),
                    "sha256": _sha256(ledger_path),
                },
                "tokenizer": {
                    "doc_token_id": 3,
                    "files": [
                        {
                            "bytes": candidate.stat().st_size,
                            "name": name,
                            "sha256": _sha256(candidate),
                        }
                        for name in (
                            "tokenizer.json",
                            "tokenizer_config.json",
                        )
                        if (candidate := tokenizer_path / name).is_file()
                    ],
                    "vocab_size": len(tokenizer),
                },
                "status": "complete",
            }
        ),
        encoding="utf-8",
    )

    token_count_path = tmp_path / "token-count.json"
    token_count_path.write_text(
        json.dumps(
            {
                "tokenizers": {
                    "65k": {
                        "artifact": {
                            "tokenizer_json_sha256": _sha256(
                                tokenizer_path / "tokenizer.json"
                            ),
                            "vocab_size": len(tokenizer),
                        },
                        "categories": category_counts,
                        "overall": {
                            "tokens": sum(item["tokens"] for item in category_counts.values())
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    contract = create_scaling_data_contract(
        input_path=input_path,
        tokenizer_path=tokenizer_path,
        source_manifest_path=source_manifest_path,
        heldout_manifest_path=heldout_manifest_path,
        heldout_selection_ledger_path=ledger_path,
        token_count_report_path=token_count_path,
        seed=29,
        final_target_tokens=128,
        sequence_length=8,
        expected_vocab_size=len(tokenizer),
    )
    return {
        "contract": contract,
        "heldout_ids": {
            "category_a/component_a/2",
            "category_a/component_a/3",
            "category_b/component_b/2",
        },
        "input": input_path,
        "tokenizer": tokenizer_path,
    }


def _build_index(fixture: dict, output: Path, **kwargs) -> dict:
    return build_scaling_source_index(
        contract=fixture["contract"],
        output=output,
        checkpoint_documents=5,
        checkpoint_text_bytes=1_024,
        progress_documents=1_000,
        **kwargs,
    )


def _build_pack(fixture: dict, index: Path, output: Path, **kwargs) -> dict:
    return build_scaling_packed_dataset(
        index_manifest_path=index / "manifest.json",
        input_path=fixture["input"],
        tokenizer_path=fixture["tokenizer"],
        output=output,
        target_tokens=128,
        sequence_length=8,
        sequences_per_shard=4,
        checkpoint_documents=4,
        checkpoint_text_bytes=512,
        progress_documents=1_000,
        **kwargs,
    )


def test_scaling_index_and_pack_exclude_exact_and_family_heldout(
    scaling_fixture: dict, tmp_path: Path
) -> None:
    index_output = tmp_path / "index"
    index = _build_index(scaling_fixture, index_output)

    assert index["statistics"]["documents"] == 48
    assert index["statistics"]["heldout_exact_documents"] == 2
    assert index["statistics"]["heldout_documents"] == 3
    assert validate_scaling_source_index(index_output / "manifest.json") == index

    packed_output = tmp_path / "packed"
    packed = _build_pack(scaling_fixture, index_output, packed_output)
    assert packed["statistics"]["tokens"] == 128
    assert packed["statistics"]["sequences"] == 16
    assert packed["format"]["dtype"] == "uint16"
    assert sum(item["bytes"] for item in packed["shards"]) == 256
    assert len(packed["selection"]["strata"]) == 2
    assert validate_scaling_packed_manifest(
        packed_output / "manifest.json",
        index_manifest_path=index_output / "manifest.json",
    ) == packed

    connection = sqlite3.connect(packed_output / PACK_STATE_FILENAME)
    connection.execute(
        "ATTACH DATABASE ? AS source_index",
        (str(index_output / INDEX_DATABASE_FILENAME),),
    )
    try:
        selected_ids = {
            row[0]
            for row in connection.execute(
                "SELECT d.document_id FROM selected_documents s "
                "JOIN source_index.documents d ON d.input_offset = s.input_offset"
            )
        }
    finally:
        connection.close()
    assert selected_ids.isdisjoint(scaling_fixture["heldout_ids"])

    dataset = PackedTokenDataset(packed_output / "manifest.json")
    assert len(dataset) == 16
    assert torch.stack([dataset[index] for index in range(len(dataset))]).numel() == 128
    assert _build_pack(scaling_fixture, index_output, packed_output, resume=True) == packed


def test_scaling_index_and_pack_resume_without_changing_selected_bytes(
    scaling_fixture: dict, tmp_path: Path
) -> None:
    interrupted_index = tmp_path / "interrupted-index"
    with pytest.raises(RuntimeError, match="source-index interruption"):
        _build_index(
            scaling_fixture,
            interrupted_index,
            _interrupt_after_documents=7,
        )
    resumed_index = _build_index(scaling_fixture, interrupted_index, resume=True)
    fresh_index_output = tmp_path / "fresh-index"
    fresh_index = _build_index(scaling_fixture, fresh_index_output)
    assert resumed_index["build_fingerprint"] == fresh_index["build_fingerprint"]
    assert resumed_index["statistics"] == fresh_index["statistics"]

    interrupted_pack = tmp_path / "interrupted-pack"
    with pytest.raises(RuntimeError, match="packing interruption"):
        _build_pack(
            scaling_fixture,
            interrupted_index,
            interrupted_pack,
            _interrupt_after_documents=5,
        )
    resumed_pack = _build_pack(
        scaling_fixture,
        interrupted_index,
        interrupted_pack,
        resume=True,
    )
    fresh_pack_output = tmp_path / "fresh-pack"
    fresh_pack = _build_pack(
        scaling_fixture,
        interrupted_index,
        fresh_pack_output,
    )
    assert resumed_pack["build_fingerprint"] == fresh_pack["build_fingerprint"]
    assert resumed_pack["statistics"] == fresh_pack["statistics"]
    assert [item["sha256"] for item in resumed_pack["shards"]] == [
        item["sha256"] for item in fresh_pack["shards"]
    ]
    assert resumed_pack["selection"] == fresh_pack["selection"]


def test_scaling_index_fails_closed_when_final_capacity_is_impossible(
    scaling_fixture: dict, tmp_path: Path
) -> None:
    contract = json.loads(json.dumps(scaling_fixture["contract"]))
    contract["selection"]["final_target_tokens"] = 100_000
    with pytest.raises(RuntimeError, match="cannot cover"):
        build_scaling_source_index(
            contract=contract,
            output=tmp_path / "impossible-index",
            checkpoint_documents=8,
            progress_documents=1_000,
        )
