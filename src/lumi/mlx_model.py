"""Native MLX implementation of the Lumi decoder-only architecture.

This module is intentionally not imported from :mod:`lumi`. MLX is an optional
Apple-Silicon runtime, and converted artifacts copy this file beside their
weights so MLX-LM can load Lumi through ``config.json``'s ``model_file`` field.
"""

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache


@dataclass
class ModelArgs(BaseModelArgs):
    """Fields that define one exact Lumi MLX architecture."""

    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    attention_bias: bool = False
    tie_word_embeddings: bool = True
    rope_traditional: bool = False

    def __post_init__(self) -> None:
        positive_integers = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
        }
        for name, value in positive_integers.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.model_type != "lumi":
            raise ValueError(f"expected model_type='lumi', got {self.model_type!r}")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")


class LumiAttention(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = args.head_dim**-0.5
        query_size = args.num_attention_heads * args.head_dim
        key_value_size = args.num_key_value_heads * args.head_dim

        self.q_proj = nn.Linear(args.hidden_size, query_size, bias=args.attention_bias)
        self.k_proj = nn.Linear(args.hidden_size, key_value_size, bias=args.attention_bias)
        self.v_proj = nn.Linear(args.hidden_size, key_value_size, bias=args.attention_bias)
        self.o_proj = nn.Linear(query_size, args.hidden_size, bias=args.attention_bias)
        # Lumi's PyTorch rotate-half layout pairs dimensions at stride D / 2,
        # which is MLX RoPE's non-traditional layout.
        self.rotary = nn.RoPE(
            args.head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        mask: str | mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        batch, sequence, _ = hidden_states.shape
        query = self.q_proj(hidden_states).reshape(
            batch, sequence, self.num_heads, self.head_dim
        )
        key = self.k_proj(hidden_states).reshape(
            batch, sequence, self.num_key_value_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).reshape(
            batch, sequence, self.num_key_value_heads, self.head_dim
        )
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        query = self.rotary(query, offset=offset)
        key = self.rotary(key, offset=offset)
        if cache is not None:
            key, value = cache.update_and_fetch(key, value)

        output = scaled_dot_product_attention(
            query,
            key,
            value,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, sequence, -1)
        return self.o_proj(output)


class LumiMLP(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class LumiDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.self_attn = LumiAttention(args)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.mlp = LumiMLP(args)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: str | mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), mask=mask, cache=cache
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class LumiModel(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.max_position_embeddings = args.max_position_embeddings
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [LumiDecoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeddings is None else input_embeddings
        )
        if cache is None:
            cache = [None] * len(self.layers)
        if len(cache) != len(self.layers):
            raise ValueError("cache length must equal the number of Lumi decoder layers")
        cache_offset = cache[0].offset if cache and cache[0] is not None else 0
        if cache_offset + hidden_states.shape[1] > self.max_position_embeddings:
            raise ValueError("input and cache exceed Lumi's configured context window")

        mask = create_attention_mask(hidden_states, cache[0])
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
        return self.norm(hidden_states)


class Model(nn.Module):
    """MLX-LM entry point for a converted Lumi checkpoint."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LumiModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        hidden_states = self.model(
            input_ids,
            cache=cache,
            input_embeddings=input_embeddings,
        )
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden_states)
        return self.lm_head(hidden_states)

    @property
    def layers(self) -> list[LumiDecoderLayer]:
        return self.model.layers

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Drop buffers that MLX recomputes rather than storing as weights."""

        sanitized = {
            name: value
            for name, value in weights.items()
            if not (
                name.endswith(".self_attn.rotary.inv_freq")
                or name.endswith(".self_attn.rotary_emb.inv_freq")
            )
        }
        if self.args.tie_word_embeddings:
            sanitized.pop("lm_head.weight", None)
        return sanitized
