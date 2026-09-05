from __future__ import annotations

import json
import sys
from array import array
from pathlib import Path

import pytest

from lumi.posttraining_token_consistent_shards_v1 import verify_binary_masks


def _write_int32(path: Path, values: list[int], *, signed: bool) -> None:
    encoded = array("i" if signed else "I", values)
    if sys.byteorder != "little":
        encoded.byteswap()
    path.write_bytes(encoded.tobytes())


def _tiny_manifest(tmp_path: Path, labels: list[int]) -> dict:
    prefix = "train-00000"
    tokens = [1, 6, 20, 10, 7, 30, 10, 2]
    _write_int32(tmp_path / f"{prefix}.tokens.u32le", tokens, signed=False)
    _write_int32(tmp_path / f"{prefix}.labels.i32le", labels, signed=True)
    entry = {
        "offset_tokens": 0,
        "record_id_sha256": "0" * 64,
        "record_ordinal": 0,
        "supervised_tokens": 3,
        "tokens": len(tokens),
    }
    (tmp_path / f"{prefix}.index.jsonl").write_text(
        json.dumps(entry) + "\n", encoding="utf-8"
    )
    return {
        "shards": [{"index": 0, "split": "train"}],
    }


def test_binary_mask_verifier_accepts_assistant_only_labels(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path, [-100, -100, -100, -100, -100, 30, 10, 2])
    audit = verify_binary_masks(tmp_path, manifest)
    assert audit == {"index_rows": 1, "masked_tokens": 5, "supervised_tokens": 3}


def test_binary_mask_verifier_rejects_label_mismatch(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path, [-100, -100, -100, -100, -100, 31, 10, 2])
    with pytest.raises(ValueError, match="differs from input"):
        verify_binary_masks(tmp_path, manifest)
