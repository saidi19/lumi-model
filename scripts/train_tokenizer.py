#!/usr/bin/env python3
"""Train a Lumi byte-level BPE tokenizer from local text or JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

REPOSITORY_ROOT = Path(__file__).parents[1]
DEFAULT_CONTRACT = REPOSITORY_ROOT / "configs" / "tokenizer-v0.1.json"


class DocumentStream:
    def __init__(self, paths: list[Path], text_key: str, max_documents: int | None) -> None:
        self.paths = paths
        self.text_key = text_key
        self.max_documents = max_documents
        self.documents_yielded = 0

    def __iter__(self) -> Iterator[str]:
        for path in self.paths:
            if not path.is_file():
                raise FileNotFoundError(f"tokenizer input does not exist: {path}")
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if (
                        self.max_documents is not None
                        and self.documents_yielded >= self.max_documents
                    ):
                        return
                    if path.suffix.lower() == ".jsonl":
                        try:
                            record = json.loads(line)
                            text = record[self.text_key]
                        except (json.JSONDecodeError, KeyError) as error:
                            raise ValueError(
                                f"invalid JSONL at {path}:{line_number}: {error}"
                            ) from error
                        if not isinstance(text, str):
                            raise ValueError(
                                f"{path}:{line_number} field {self.text_key!r} must be text"
                            )
                    else:
                        text = line.rstrip("\r\n")
                    if text.strip():
                        self.documents_yielded += 1
                        yield text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--model-max-length", type=int, default=4096)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _validate_special_tokens(tokenizer, contract: dict) -> None:
    vocabulary = tokenizer.get_vocab()
    for name, specification in contract["special_tokens"].items():
        actual_id = vocabulary.get(specification["token"])
        if actual_id != specification["id"]:
            raise RuntimeError(
                f"{name} ID changed during training: expected {specification['id']}, "
                f"got {actual_id}"
            )


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite tokenizer output: {args.output}")
    if args.min_frequency < 1:
        raise ValueError("min-frequency must be positive")

    contract_bytes = args.contract.read_bytes()
    contract = json.loads(contract_bytes)
    vocab_size = args.vocab_size or contract["vocab_size"]
    special_tokens = contract["special_tokens"]
    special_token_strings = [item["token"] for item in special_tokens.values()]
    minimum_vocabulary = len(special_token_strings) + len(pre_tokenizers.ByteLevel.alphabet())
    if vocab_size < minimum_vocabulary:
        raise ValueError(
            f"vocab-size must be at least {minimum_vocabulary} for byte coverage and specials"
        )

    tokenizer = Tokenizer(models.BPE(unk_token=special_tokens["unk"]["token"]))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=args.min_frequency,
        show_progress=not args.no_progress,
        special_tokens=special_token_strings,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    documents = DocumentStream(args.input, args.text_key, args.max_documents)
    tokenizer.train_from_iterator(documents, trainer=trainer)
    if documents.documents_yielded == 0:
        raise ValueError("tokenizer input did not contain any non-empty documents")

    additional_special_tokens = [
        specification["token"]
        for name, specification in special_tokens.items()
        if name not in {"pad", "bos", "eos", "unk"}
    ]
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=args.model_max_length,
        pad_token=special_tokens["pad"]["token"],
        bos_token=special_tokens["bos"]["token"],
        eos_token=special_tokens["eos"]["token"],
        unk_token=special_tokens["unk"]["token"],
        additional_special_tokens=additional_special_tokens,
    )
    _validate_special_tokens(fast_tokenizer, contract)
    if len(fast_tokenizer) != vocab_size:
        raise RuntimeError(
            f"trainer produced {len(fast_tokenizer)} tokens, expected {vocab_size}; "
            "use a larger or more varied sample"
        )

    args.output.mkdir(parents=True)
    fast_tokenizer.save_pretrained(args.output)
    metadata = {
        "algorithm": contract["algorithm"],
        "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "documents_seen": documents.documents_yielded,
        "normalization": contract["normalization"],
        "version": contract["version"],
        "vocab_size": len(fast_tokenizer),
    }
    with (args.output / "lumi_tokenizer_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"PASS: trained {len(fast_tokenizer)}-token Lumi tokenizer "
        f"from {documents.documents_yielded} documents"
    )


if __name__ == "__main__":
    main()
