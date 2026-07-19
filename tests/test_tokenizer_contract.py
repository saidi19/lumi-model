from __future__ import annotations

import json
from pathlib import Path

from lumi import LumiConfig

REPOSITORY_ROOT = Path(__file__).parents[1]


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_tokenizer_special_ids_are_contiguous_and_unique() -> None:
    contract = _read_json(REPOSITORY_ROOT / "configs" / "tokenizer-v0.1.json")
    special_tokens = contract["special_tokens"]
    token_ids = [specification["id"] for specification in special_tokens.values()]
    token_strings = [specification["token"] for specification in special_tokens.values()]
    assert token_ids == list(range(len(token_ids)))
    assert len(token_strings) == len(set(token_strings))


def test_model_configs_match_target_tokenizer_contract() -> None:
    contract = _read_json(REPOSITORY_ROOT / "configs" / "tokenizer-v0.1.json")
    expected_ids = {
        name: contract["special_tokens"][name]["id"] for name in ("pad", "bos", "eos", "doc")
    }
    for config_name in ("lumi-150m.json", "lumi-4b.json", "lumi-4b-v0.2.json"):
        config = LumiConfig(**_read_json(REPOSITORY_ROOT / "configs" / config_name))
        assert config.vocab_size == contract["vocab_size"]
        assert config.pad_token_id == expected_ids["pad"]
        assert config.bos_token_id == expected_ids["bos"]
        assert config.eos_token_id == expected_ids["eos"]
        assert config.doc_token_id == expected_ids["doc"]
