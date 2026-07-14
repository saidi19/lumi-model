"""Analytical parameter counting that does not allocate a 4B model."""

from __future__ import annotations

from dataclasses import dataclass

from .configuration_lumi import LumiConfig


@dataclass(frozen=True)
class ParameterBreakdown:
    embeddings: int
    attention: int
    mlp: int
    norms: int
    output_head: int

    @property
    def total(self) -> int:
        return self.embeddings + self.attention + self.mlp + self.norms + self.output_head


def parameter_breakdown(config: LumiConfig) -> ParameterBreakdown:
    """Return the exact parameter count for the v0.1 module definitions."""

    hidden = config.hidden_size
    query = config.query_projection_size
    key_value = config.key_value_projection_size
    layers = config.num_hidden_layers

    embeddings = config.vocab_size * hidden
    attention_per_layer = hidden * query + 2 * hidden * key_value + query * hidden
    if config.attention_bias:
        attention_per_layer += query + 2 * key_value + hidden

    mlp_per_layer = 3 * hidden * config.intermediate_size
    norms = (2 * layers + 1) * hidden
    output_head = 0 if config.tie_word_embeddings else config.vocab_size * hidden

    return ParameterBreakdown(
        embeddings=embeddings,
        attention=layers * attention_per_layer,
        mlp=layers * mlp_per_layer,
        norms=norms,
        output_head=output_head,
    )
