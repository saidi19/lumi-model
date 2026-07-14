"""Configuration for Lumi decoder-only language models."""

from __future__ import annotations

from transformers import PretrainedConfig


class LumiConfig(PretrainedConfig):
    """Configuration container for the Lumi architecture.

    Query projection width is deliberately independent from ``hidden_size``:
    ``num_attention_heads * head_dim`` may be wider than the residual stream.
    """

    model_type = "lumi"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 65_536,
        hidden_size: int = 3_072,
        intermediate_size: int = 8_384,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        head_dim: int = 64,
        max_position_embeddings: int = 4_096,
        rope_theta: float = 1_000_000.0,
        rms_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        attention_bias: bool = False,
        use_cache: bool = False,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.attention_bias = attention_bias
        self.use_cache = use_cache

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self._validate()

    @property
    def query_projection_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def key_value_projection_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    def _validate(self) -> None:
        positive_fields = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.max_position_embeddings,
        }
        for name, value in positive_fields.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads "
                f"({self.num_attention_heads} vs {self.num_key_value_heads})"
            )
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")

        special_ids = [self.pad_token_id, self.bos_token_id, self.eos_token_id]
        if any(token_id is None for token_id in special_ids):
            raise ValueError("PAD, BOS, and EOS token IDs must all be configured")
        if len(set(special_ids)) != len(special_ids):
            raise ValueError("PAD, BOS, and EOS token IDs must be distinct")
        if not all(0 <= token_id < self.vocab_size for token_id in special_ids):
            raise ValueError("special token IDs must be inside the vocabulary")
