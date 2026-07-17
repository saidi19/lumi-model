from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from lumi import (
    LumiConfig,
    LumiForCausalLM,
    LumiModel,
    build_document_causal_mask,
    build_packed_document_metadata,
)
from lumi.modeling_lumi import _lumi_eager_attention_forward


def test_forward_shape_and_finite_loss(tiny_config: LumiConfig) -> None:
    torch.manual_seed(1)
    model = LumiForCausalLM(tiny_config)
    input_ids = torch.randint(3, tiny_config.vocab_size, (2, 12))
    output = model(input_ids=input_ids, labels=input_ids)
    assert output.logits.shape == (2, 12, tiny_config.vocab_size)
    assert output.loss is not None
    assert torch.isfinite(output.loss)


def test_future_tokens_cannot_change_prefix_logits(tiny_config: LumiConfig) -> None:
    torch.manual_seed(2)
    model = LumiForCausalLM(tiny_config).eval()
    first = torch.tensor([[1, 8, 9, 10, 11, 12]])
    second = torch.tensor([[1, 8, 9, 10, 55, 56]])
    with torch.no_grad():
        first_logits = model(input_ids=first).logits[:, :4]
        second_logits = model(input_ids=second).logits[:, :4]
    torch.testing.assert_close(first_logits, second_logits, rtol=0.0, atol=1e-6)


def test_eager_attention_path_remains_causal(tiny_config: LumiConfig) -> None:
    tiny_config._attn_implementation = "eager"
    torch.manual_seed(21)
    model = LumiForCausalLM(tiny_config).eval()
    first = torch.tensor([[1, 8, 9, 10, 11]])
    second = torch.tensor([[1, 8, 9, 55, 56]])
    with torch.no_grad():
        first_logits = model(input_ids=first).logits[:, :3]
        second_logits = model(input_ids=second).logits[:, :3]
    torch.testing.assert_close(first_logits, second_logits, rtol=0.0, atol=1e-6)


def test_packed_document_metadata_resets_positions_and_flattens_boundaries() -> None:
    input_ids = torch.tensor(
        [
            [3, 10, 11, 3, 12, 13],
            [20, 21, 3, 30, 3, 40],
        ]
    )

    metadata = build_packed_document_metadata(input_ids, doc_token_id=3)

    assert metadata.position_ids.tolist() == [
        [0, 1, 2, 0, 1, 2],
        [0, 1, 0, 1, 0, 1],
    ]
    assert metadata.cu_seq_lens.dtype == torch.int32
    assert metadata.cu_seq_lens.tolist() == [0, 3, 6, 8, 10, 12]
    assert metadata.max_length == 3


def test_document_causal_mask_blocks_other_segments() -> None:
    position_ids = torch.tensor([[0, 1, 2, 0, 1]])

    mask = build_document_causal_mask(position_ids)

    expected = torch.tensor(
        [
            [True, False, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [False, False, False, True, False],
            [False, False, False, True, True],
        ]
    )
    assert torch.equal(mask[0, 0], expected)


def test_document_isolation_prevents_prefix_leakage(tiny_config: LumiConfig) -> None:
    tiny_config._attn_implementation = "eager"
    torch.manual_seed(24)
    model = LumiForCausalLM(tiny_config).eval()
    first = torch.tensor([[3, 8, 9, 3, 20, 21]])
    second = torch.tensor([[3, 55, 56, 3, 20, 21]])

    with torch.no_grad():
        isolated_first = model(input_ids=first, document_isolation=True).logits[:, 3:]
        isolated_second = model(input_ids=second, document_isolation=True).logits[:, 3:]
        packed_first = model(input_ids=first).logits[:, 3:]
        packed_second = model(input_ids=second).logits[:, 3:]

    torch.testing.assert_close(isolated_first, isolated_second, rtol=0.0, atol=1e-6)
    assert not torch.allclose(packed_first, packed_second, rtol=0.0, atol=1e-6)


def test_flash_varlen_contract_matches_dense_reference(
    tiny_config: LumiConfig, monkeypatch
) -> None:
    torch.manual_seed(25)
    model = LumiForCausalLM(tiny_config).eval()
    input_ids = torch.tensor(
        [
            [3, 8, 9, 3, 10, 11],
            [12, 13, 3, 14, 15, 16],
        ]
    )
    metadata = build_packed_document_metadata(input_ids, doc_token_id=3)
    common = {
        "cu_seq_lens_k": metadata.cu_seq_lens,
        "cu_seq_lens_q": metadata.cu_seq_lens,
        "document_isolation": True,
        "max_length_k": metadata.max_length,
        "max_length_q": metadata.max_length,
        "position_ids": metadata.position_ids,
    }

    model.config._attn_implementation = "eager"
    with torch.no_grad():
        expected = model(input_ids=input_ids, **common).logits

    calls = []

    def fake_flash_varlen(
        module,
        query,
        key,
        value,
        attention_mask,
        *,
        cu_seq_lens_q,
        cu_seq_lens_k,
        max_length_q,
        max_length_k,
        scaling,
        **kwargs,
    ):
        del kwargs
        assert attention_mask is None
        assert torch.equal(cu_seq_lens_q, cu_seq_lens_k)
        assert max_length_q == max_length_k == metadata.max_length
        calls.append(cu_seq_lens_q.tolist())

        batch, _, sequence, _ = query.shape
        flat_query = query.transpose(1, 2).reshape(-1, query.shape[1], query.shape[-1])
        flat_key = key.transpose(1, 2).reshape(-1, key.shape[1], key.shape[-1])
        flat_value = value.transpose(1, 2).reshape(-1, value.shape[1], value.shape[-1])
        outputs = []
        boundaries = cu_seq_lens_q.tolist()
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            segment_output, _ = _lumi_eager_attention_forward(
                module,
                flat_query[start:end].transpose(0, 1).unsqueeze(0),
                flat_key[start:end].transpose(0, 1).unsqueeze(0),
                flat_value[start:end].transpose(0, 1).unsqueeze(0),
                None,
                scaling=scaling,
            )
            outputs.append(segment_output.squeeze(0))
        return torch.cat(outputs, dim=0).view(batch, sequence, query.shape[1], -1), None

    monkeypatch.setitem(
        ALL_ATTENTION_FUNCTIONS._global_mapping,
        "flash_attention_2",
        fake_flash_varlen,
    )
    model.config._attn_implementation = "flash_attention_2"
    with torch.no_grad():
        actual = model(input_ids=input_ids, **common).logits

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert calls == [metadata.cu_seq_lens.tolist()] * tiny_config.num_hidden_layers


def test_document_isolation_rejects_invalid_position_resets(tiny_config: LumiConfig) -> None:
    model = LumiForCausalLM(tiny_config).eval()
    with pytest.raises(ValueError, match="increment by one or reset"):
        model(
            input_ids=torch.tensor([[3, 8, 9, 3, 10]]),
            position_ids=torch.tensor([[0, 1, 3, 0, 1]]),
            document_isolation=True,
        )


def test_left_padding_does_not_change_content_logits(tiny_config: LumiConfig) -> None:
    torch.manual_seed(22)
    model = LumiForCausalLM(tiny_config).eval()
    content = torch.tensor([[1, 8, 9]])
    padded = torch.tensor([[0, 0, 1, 8, 9]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1]])
    with torch.no_grad():
        expected = model(input_ids=content).logits
        actual = model(input_ids=padded, attention_mask=attention_mask).logits[:, -3:]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)


def test_loss_is_shifted_exactly_once(tiny_config: LumiConfig) -> None:
    torch.manual_seed(3)
    model = LumiForCausalLM(tiny_config).eval()
    input_ids = torch.tensor([[1, 5, 7, 9, 2]])
    with torch.no_grad():
        output = model(input_ids=input_ids, labels=input_ids)
        logits = model(input_ids=input_ids).logits
    expected = F.cross_entropy(
        logits[:, :-1].reshape(-1, tiny_config.vocab_size),
        input_ids[:, 1:].reshape(-1),
    )
    torch.testing.assert_close(output.loss, expected)


def test_input_and_output_embeddings_are_tied(tiny_config: LumiConfig) -> None:
    model = LumiForCausalLM(tiny_config)
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight


def test_gradient_checkpointing_preserves_loss_and_gradients(tiny_config: LumiConfig) -> None:
    torch.manual_seed(23)
    model = LumiForCausalLM(tiny_config).train()
    input_ids = torch.tensor([[1, 8, 9, 10, 11, 2]])

    baseline = model(input_ids=input_ids, labels=input_ids)
    baseline.loss.backward()
    expected_loss = baseline.loss.detach().clone()
    expected_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }

    model.zero_grad(set_to_none=True)
    model.gradient_checkpointing_enable()
    checkpointed = model(input_ids=input_ids, labels=input_ids)
    checkpointed.loss.backward()

    assert model.is_gradient_checkpointing
    torch.testing.assert_close(checkpointed.loss, expected_loss, rtol=0.0, atol=1e-7)
    for name, parameter in model.named_parameters():
        if name in expected_gradients:
            torch.testing.assert_close(
                parameter.grad,
                expected_gradients[name],
                rtol=1e-5,
                atol=1e-7,
                msg=name,
            )


def test_save_load_round_trip_preserves_logits(tiny_config: LumiConfig, tmp_path) -> None:
    torch.manual_seed(4)
    model = LumiForCausalLM(tiny_config).eval()
    input_ids = torch.tensor([[1, 3, 5, 7]])
    with torch.no_grad():
        expected = model(input_ids=input_ids).logits

    model.save_pretrained(tmp_path)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        actual = loaded(input_ids=input_ids).logits
    torch.testing.assert_close(actual, expected)


def test_save_load_round_trip_preserves_document_isolated_logits(
    tiny_config: LumiConfig, tmp_path
) -> None:
    tiny_config._attn_implementation = "eager"
    torch.manual_seed(26)
    model = LumiForCausalLM(tiny_config).eval()
    input_ids = torch.tensor([[3, 8, 9, 3, 10, 11]])
    with torch.no_grad():
        expected = model(input_ids=input_ids, document_isolation=True).logits

    model.save_pretrained(tmp_path)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    loaded.config._attn_implementation = "eager"
    with torch.no_grad():
        actual = loaded(input_ids=input_ids, document_isolation=True).logits
    torch.testing.assert_close(actual, expected)


def test_short_generation_runs_without_cache(tiny_config: LumiConfig) -> None:
    torch.manual_seed(5)
    model = LumiForCausalLM(tiny_config).eval()
    input_ids = torch.tensor([[1, 3, 5]])
    with torch.no_grad():
        generated = model.generate(input_ids=input_ids, max_new_tokens=2, use_cache=False)
    assert generated.shape == (1, 5)


def test_attention_backend_receives_forward_kwargs(tiny_config: LumiConfig, monkeypatch) -> None:
    backend_name = "lumi_test_backend"
    markers = []

    def test_backend(
        module,
        query,
        key,
        value,
        attention_mask,
        backend_marker=None,
        **kwargs,
    ):
        del module, key, value, attention_mask, kwargs
        markers.append(backend_marker)
        return query.transpose(1, 2).contiguous(), None

    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS._global_mapping, backend_name, test_backend)
    tiny_config._attn_implementation = backend_name
    model = LumiForCausalLM(tiny_config).eval()
    input_ids = torch.tensor([[1, 5, 7]])
    with torch.no_grad():
        model(input_ids=input_ids, backend_marker="received")

    assert LumiModel._supports_attention_backend is True
    assert markers == ["received"] * tiny_config.num_hidden_layers
