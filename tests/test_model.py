from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from lumi import LumiConfig, LumiForCausalLM


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


def test_save_load_round_trip_preserves_logits(
    tiny_config: LumiConfig, tmp_path
) -> None:
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


def test_short_generation_runs_without_cache(tiny_config: LumiConfig) -> None:
    torch.manual_seed(5)
    model = LumiForCausalLM(tiny_config).eval()
    input_ids = torch.tensor([[1, 3, 5]])
    with torch.no_grad():
        generated = model.generate(input_ids=input_ids, max_new_tokens=2, use_cache=False)
    assert generated.shape == (1, 5)
