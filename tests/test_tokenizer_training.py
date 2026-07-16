from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer


def test_tiny_tokenizer_training_preserves_special_ids(tmp_path: Path) -> None:
    repository_root = Path(__file__).parents[1]
    source_contract = repository_root / "configs" / "tokenizer-v0.1.json"
    with source_contract.open(encoding="utf-8") as handle:
        contract = json.load(handle)
    contract["vocab_size"] = 320
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    corpus = tmp_path / "sample.jsonl"
    with corpus.open("w", encoding="utf-8") as handle:
        for index in range(600):
            record = {
                "text": (
                    f"Lumi document {index}: alpha{index % 37} beta{index % 53} "
                    f"code_{index % 71} বাংলা café 日本語"
                )
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    output = tmp_path / "tokenizer"
    script = repository_root / "scripts" / "train_tokenizer.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
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
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"

    tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
    assert len(tokenizer) == 320
    for specification in contract["special_tokens"].values():
        assert tokenizer.get_vocab()[specification["token"]] == specification["id"]
    encoded = tokenizer.encode("বাংলা café 日本語", add_special_tokens=False)
    assert tokenizer.unk_token_id not in encoded
    tokenizer_data = json.loads((output / "tokenizer.json").read_text(encoding="utf-8"))
    assert max(map(len, tokenizer_data["model"]["vocab"])) <= 32

    metadata = json.loads((output / "lumi_tokenizer_metadata.json").read_text(encoding="utf-8"))
    assert metadata["documents_seen"] == 600
    assert metadata["input_files"] == [
        {
            "file_bytes": corpus.stat().st_size,
            "path": str(corpus.resolve()),
            "sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        }
    ]
    assert metadata["training"] == {
        "max_documents": None,
        "max_token_length": 32,
        "min_frequency": 1,
        "model_max_length": 4096,
        "text_key": "text",
    }
    assert metadata["versions"]["tokenizers"]
    assert metadata["versions"]["transformers"]
