"""Construct the genuine frozen-version tokenizer from caller-authenticated bytes.

No directory lookup, from_pretrained call, model construction, weights, device
operation, or inference occurs here. The caller should warm trusted runtime
imports before private reads: Transformers 5.2 can import Torch as a dependency
even when USE_TORCH=0. Constructor inputs themselves remain wholly in memory.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any

MAXIMUM_TOKENIZER_BYTES = 32 * 1024 * 1024
MAXIMUM_CONFIG_BYTES = 4096
_ERROR = "invalid safety tokenizer memory contract"
_CONFIG_KEYS = {
    "backend",
    "bos_token",
    "eos_token",
    "extra_special_tokens",
    "model_max_length",
    "pad_token",
    "tokenizer_class",
    "unk_token",
}


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError(_ERROR)


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in items:
        _require(key not in result)
        result[key] = value
    return result


def _nonfinite(_value: str) -> None:
    raise ValueError(_ERROR)


class _DiscardText:
    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        pass


def _configuration(payload: bytes) -> dict[str, Any]:
    _require(type(payload) is bytes and 0 < len(payload) <= MAXIMUM_CONFIG_BYTES)
    config = json.loads(
        payload.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_nonfinite
    )
    _require(type(config) is dict and set(config) == _CONFIG_KEYS)
    for key in _CONFIG_KEYS - {"extra_special_tokens", "model_max_length"}:
        _require(type(config[key]) is str and bool(config[key]))
    _require(config["backend"] == "tokenizers")
    _require(config["tokenizer_class"] == "TokenizersBackend")
    _require(type(config["model_max_length"]) is int and config["model_max_length"] > 0)
    extras = config["extra_special_tokens"]
    _require(type(extras) is list and all(type(value) is str and value for value in extras))
    special_tokens = [config[key] for key in ("bos_token", "eos_token", "pad_token", "unk_token")]
    special_tokens.extend(extras)
    _require(len(special_tokens) == len(set(special_tokens)))
    return config


def _construct_tokenizer(tokenizer_bytes: bytes, config_bytes: bytes) -> Any:
    """Lower-level constructor unit; not authorization for historical replay.

    Only the genuine generic TokenizersBackend export is accepted. The complete
    backend special-token set and vocabulary must agree with the configuration
    and survive construction unchanged; missing tokens cannot be silently added.
    All ordinary failure messages are fixed and contain no supplied token text.
    """
    try:
        config = _configuration(config_bytes)
        _require(
            type(tokenizer_bytes) is bytes and 0 < len(tokenizer_bytes) <= MAXIMUM_TOKENIZER_BYTES
        )
        with contextlib.redirect_stdout(_DiscardText()), contextlib.redirect_stderr(_DiscardText()):
            import tokenizers
            import transformers
            from transformers.tokenization_utils_tokenizers import TokenizersBackend

            _require(transformers.TokenizersBackend is TokenizersBackend)
            _require(TokenizersBackend.__module__ == "transformers.tokenization_utils_tokenizers")
            _require(TokenizersBackend.__name__ == "TokenizersBackend")
            backend = tokenizers.Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
            vocabulary = backend.get_vocab(with_added_tokens=True)
            original_specials = {
                token_id: token.content
                for token_id, token in backend.get_added_tokens_decoder().items()
                if token.special
            }
            declared_specials = {
                config[key] for key in ("bos_token", "eos_token", "pad_token", "unk_token")
            } | set(config["extra_special_tokens"])
            _require(set(original_specials.values()) == declared_specials)
            result = TokenizersBackend(
                tokenizer_object=backend,
                **{key: value for key, value in config.items() if key != "tokenizer_class"},
            )
            _require(result.get_vocab() == vocabulary)
            _require(set(result.all_special_ids) == set(original_specials))
            _require(set(result.all_special_tokens) == declared_specials)
            _require(result.model_max_length == config["model_max_length"])
            return result
    except Exception:
        raise ValueError(_ERROR) from None


def load_tokenizer(tokenizer_bytes: bytes, config_bytes: bytes) -> Any:
    """Version-bound replay entry; external SHA authentication is the caller's job.

    The historical replay always uses this gate. Source tests may exercise the
    lower-level constructor on their installed dependency without claiming that
    dependency is authorized to reproduce the historical private outputs.
    """
    try:
        with contextlib.redirect_stdout(_DiscardText()), contextlib.redirect_stderr(_DiscardText()):
            import tokenizers
            import transformers

            _require(tokenizers.__version__ == "0.22.2" and transformers.__version__ == "5.2.0")
            return _construct_tokenizer(tokenizer_bytes, config_bytes)
    except Exception:
        raise ValueError(_ERROR) from None
