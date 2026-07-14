from __future__ import annotations

import pytest

from lumi import LumiConfig


@pytest.fixture
def tiny_config() -> LumiConfig:
    return LumiConfig(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=32,
        rope_theta=10_000.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        doc_token_id=3,
        tie_word_embeddings=True,
        use_cache=False,
    )
