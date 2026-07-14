"""A small, correctness-first PyTorch implementation of Lumi v0.1."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .configuration_lumi import LumiConfig


class LumiRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.square().mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class LumiRotaryEmbedding(nn.Module):
    def __init__(self, config: LumiConfig) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )
        # Keep this tiny deterministic buffer in checkpoints. Some loading paths
        # initialize missing non-persistent buffers to zero before custom code can
        # rebuild them, which would silently change logits after a round trip.
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.LongTensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device=hidden_states.device)
        frequencies = torch.einsum("bt,d->btd", position_ids.float(), inv_freq)
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return embeddings.cos().to(hidden_states.dtype), embeddings.sin().to(hidden_states.dtype)


def _apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    return (
        query * cosine + _rotate_half(query) * sine,
        key * cosine + _rotate_half(key) * sine,
    )


def _repeat_key_value(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch, key_value_heads, sequence, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, key_value_heads, repeats, sequence, head_dim
    )
    return hidden_states.reshape(batch, key_value_heads * repeats, sequence, head_dim)


def _lumi_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference attention used when no optimized backend is selected."""

    key = _repeat_key_value(key, module.num_key_value_groups)
    value = _repeat_key_value(value, module.num_key_value_groups)
    weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is None:
        query_length, key_length = query.shape[-2], key.shape[-2]
        causal = torch.ones((query_length, key_length), device=query.device, dtype=torch.bool).tril(
            diagonal=key_length - query_length
        )
        weights = weights.masked_fill(~causal, torch.finfo(weights.dtype).min)
    elif attention_mask.dtype == torch.bool:
        weights = weights.masked_fill(~attention_mask, torch.finfo(weights.dtype).min)
    else:
        weights = weights + attention_mask

    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = F.dropout(weights, p=dropout, training=module.training)
    output = torch.matmul(weights, value).transpose(1, 2).contiguous()
    return output, weights


class LumiAttention(nn.Module):
    def __init__(self, config: LumiConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.is_causal = True
        self.query_size = config.query_projection_size
        self.key_value_size = config.key_value_projection_size

        self.q_proj = nn.Linear(config.hidden_size, self.query_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.key_value_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.key_value_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.query_size, config.hidden_size, bias=config.attention_bias)
        self.rotary = LumiRotaryEmbedding(config)

    def _shape(self, states: torch.Tensor, heads: int) -> torch.Tensor:
        batch, sequence, _ = states.shape
        return states.view(batch, sequence, heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        batch, sequence, _ = hidden_states.shape
        query = self._shape(self.q_proj(hidden_states), self.num_heads)
        key = self._shape(self.k_proj(hidden_states), self.num_key_value_heads)
        value = self._shape(self.v_proj(hidden_states), self.num_key_value_heads)

        cosine, sine = self.rotary(value, position_ids)
        query, key = _apply_rotary(query, key, cosine, sine)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, _lumi_eager_attention_forward
        )
        output, _ = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            **kwargs,
        )
        output = output.reshape(batch, sequence, self.query_size).contiguous()
        return self.o_proj(output)


class LumiMLP(nn.Module):
    def __init__(self, config: LumiConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class LumiDecoderLayer(nn.Module):
    def __init__(self, config: LumiConfig, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = LumiRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = LumiAttention(config, layer_idx)
        self.post_attention_layernorm = LumiRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = LumiMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class LumiPreTrainedModel(PreTrainedModel):
    config_class = LumiConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _supports_sdpa = True
    _supports_attention_backend = True
    _supports_cache_class = False
    _no_split_modules = ["LumiDecoderLayer"]

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class LumiModel(LumiPreTrainedModel):
    def __init__(self, config: LumiConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [LumiDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LumiRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> BaseModelOutput | tuple[torch.Tensor]:
        return_dict = self.config.return_dict if return_dict is None else return_dict
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")

        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        batch, sequence, _ = hidden_states.shape
        if sequence > self.config.max_position_embeddings:
            raise ValueError(
                f"sequence length {sequence} exceeds configured maximum "
                f"{self.config.max_position_embeddings}"
            )
        if position_ids is None:
            position_ids = torch.arange(sequence, device=hidden_states.device).unsqueeze(0)
            position_ids = position_ids.expand(batch, -1)

        if attention_mask is not None and attention_mask.ndim == 2:
            if attention_mask.shape != (batch, sequence):
                raise ValueError(
                    "attention_mask must have shape "
                    f"{(batch, sequence)}, got {tuple(attention_mask.shape)}"
                )
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)

        if not return_dict:
            return (hidden_states,)
        return BaseModelOutput(last_hidden_state=hidden_states)


class LumiForCausalLM(LumiPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: LumiConfig) -> None:
        super().__init__(config)
        self.model = LumiModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> CausalLMOutput | tuple[torch.Tensor, ...]:
        return_dict = self.config.return_dict if return_dict is None else return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state).float()

        loss = None
        if labels is not None:
            if labels.shape != logits.shape[:2]:
                raise ValueError(
                    f"labels must have shape {tuple(logits.shape[:2])}, got {tuple(labels.shape)}"
                )
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous().to(shift_logits.device)
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            result = (logits,)
            return ((loss,) + result) if loss is not None else result
        return CausalLMOutput(loss=loss, logits=logits)
