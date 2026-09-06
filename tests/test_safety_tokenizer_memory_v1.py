"""Generated constructor equivalence, separate from historical runtime approval.

Constructor unit tests use the real installed dependency API without granting
version authorization. The public loader's frozen-version gate is tested
separately against the actual installed versions. No private artifacts are read.
"""

from __future__ import annotations

import copy
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tokenizers
import transformers
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
from transformers import AutoTokenizer, TokenizersBackend

from lumi import posttraining_safety_tokenizer_memory_v1 as memory


def _json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def synthetic_inputs() -> tuple[bytes, bytes]:
    specials = [
        "<|pad|>",
        "<|bos|>",
        "<|eos|>",
        "<|doc|>",
        "<|unk|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|tool|>",
        "<|tool_result|>",
        "<|message_end|>",
    ]
    vocabulary = {value: index for index, value in enumerate(specials)}
    for value in sorted(pre_tokenizers.ByteLevel.alphabet()):
        vocabulary[value] = len(vocabulary)
    backend = Tokenizer(models.BPE(vocab=vocabulary, merges=[], unk_token="<|unk|>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens(
        [AddedToken(value, normalized=False, special=True) for value in specials]
    )
    config = {
        "backend": "tokenizers",
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "extra_special_tokens": [
            value for value in specials if value not in {"<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"}
        ],
        "model_max_length": 4096,
        "pad_token": "<|pad|>",
        "tokenizer_class": "TokenizersBackend",
        "unk_token": "<|unk|>",
    }
    return backend.to_str().encode(), _json(config)


class SafetyTokenizerMemoryTests(unittest.TestCase):
    def _reject(self, tokenizer_bytes: bytes, config_bytes: bytes) -> None:
        with self.assertRaisesRegex(ValueError, "^invalid safety tokenizer memory contract$"):
            memory._construct_tokenizer(tokenizer_bytes, config_bytes)

    def test_real_in_memory_backend_equals_two_file_auto_tokenizer(self) -> None:
        tokenizer_bytes, config_bytes = synthetic_inputs()
        actual = memory._construct_tokenizer(tokenizer_bytes, config_bytes)
        with tempfile.TemporaryDirectory(prefix="lumi-synthetic-tokenizer-") as name:
            directory = Path(name)
            (directory / "tokenizer.json").write_bytes(tokenizer_bytes)
            (directory / "tokenizer_config.json").write_bytes(config_bytes)
            expected = AutoTokenizer.from_pretrained(
                directory, local_files_only=True, trust_remote_code=False
            )
            self.assertEqual(
                {path.name for path in directory.iterdir()},
                {"tokenizer.json", "tokenizer_config.json"},
            )
            self.assertEqual(actual.get_vocab(), expected.get_vocab())
            self.assertEqual(set(actual.all_special_ids), set(expected.all_special_ids))
            self.assertEqual(set(actual.all_special_tokens), set(expected.all_special_tokens))
            self.assertEqual(actual.model_max_length, expected.model_max_length)
            for text in (
                "",
                "plain English",
                "বাংলা উত্তর",
                "১২৩ café 日本語 🙂",
                " space\nline ",
                *actual.all_special_tokens,
                "prefix <|system|> suffix",
            ):
                for split in (False, True):
                    with self.subTest(text=text, split=split):
                        actual_ids = actual.encode(
                            text, add_special_tokens=False, split_special_tokens=split
                        )
                        expected_ids = expected.encode(
                            text, add_special_tokens=False, split_special_tokens=split
                        )
                        self.assertEqual(actual_ids, expected_ids)
                        self.assertEqual(actual.decode(actual_ids), expected.decode(expected_ids))

    def test_reserved_literals_split_without_any_special_id(self) -> None:
        tokenizer = memory._construct_tokenizer(*synthetic_inputs())
        self.assertIsInstance(tokenizer, TokenizersBackend)
        self.assertEqual(set(tokenizer.all_special_ids), set(range(11)))
        for literal in tokenizer.all_special_tokens:
            text = f"prefix {literal} suffix"
            ordinary = tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True)
            self.assertFalse(set(ordinary) & set(tokenizer.all_special_ids))
            self.assertEqual(tokenizer.decode(ordinary, clean_up_tokenization_spaces=False), text)
            special = tokenizer.encode(text, add_special_tokens=False, split_special_tokens=False)
            self.assertTrue(set(special) & set(tokenizer.all_special_ids))

    def test_constructor_never_opens_files_discovers_paths_or_uses_network(self) -> None:
        tokenizer_bytes, config_bytes = synthetic_inputs()
        with (
            patch("builtins.open", side_effect=AssertionError("file access")),
            patch("io.open", side_effect=AssertionError("file access")),
            patch("os.open", side_effect=AssertionError("file access")),
            patch("os.stat", side_effect=AssertionError("path probe")),
            patch("os.listdir", side_effect=AssertionError("path discovery")),
            patch("socket.socket", side_effect=AssertionError("network")),
            patch("socket.create_connection", side_effect=AssertionError("network")),
            patch.object(AutoTokenizer, "from_pretrained", side_effect=AssertionError("discovery")),
            patch.object(
                TokenizersBackend, "from_pretrained", side_effect=AssertionError("discovery")
            ),
        ):
            result = memory._construct_tokenizer(tokenizer_bytes, config_bytes)
            self.assertIsInstance(result, TokenizersBackend)

    def test_versions_and_genuine_export_are_required(self) -> None:
        tokenizer_bytes, config_bytes = synthetic_inputs()
        for module in (tokenizers, transformers):
            with patch.object(module, "__version__", "unsupported"):
                with self.assertRaisesRegex(
                    ValueError, "^invalid safety tokenizer memory contract$"
                ):
                    memory.load_tokenizer(tokenizer_bytes, config_bytes)
        with patch.object(transformers, "TokenizersBackend", object()):
            with self.assertRaisesRegex(ValueError, "^invalid safety tokenizer memory contract$"):
                memory.load_tokenizer(tokenizer_bytes, config_bytes)
            # Exercise the export check even when the real runtime is unsupported.
            self._reject(tokenizer_bytes, config_bytes)

    def test_public_gate_uses_actual_runtime_without_version_spoofing(self) -> None:
        payloads = synthetic_inputs()
        if tokenizers.__version__ == "0.22.2" and transformers.__version__ == "5.2.0":
            self.assertIsInstance(memory.load_tokenizer(*payloads), TokenizersBackend)
        else:
            with self.assertRaisesRegex(ValueError, "^invalid safety tokenizer memory contract$"):
                memory.load_tokenizer(*payloads)

    def test_config_shape_and_strict_types_rejected(self) -> None:
        tokenizer_bytes, config_bytes = synthetic_inputs()
        original = json.loads(config_bytes)
        mutations = [
            {**original, "unknown": "SYNTHETIC_SENTINEL"},
            {key: value for key, value in original.items() if key != "backend"},
            {**original, "backend": "other"},
            {**original, "tokenizer_class": "PreTrainedTokenizerFast"},
            {**original, "tokenizer_class": "SYNTHETIC_SENTINEL"},
            {**original, "model_max_length": True},
            {**original, "model_max_length": 4096.0},
            {**original, "model_max_length": 0},
            {**original, "bos_token": None},
            {**original, "bos_token": ""},
            {**original, "extra_special_tokens": {}},
            {**original, "extra_special_tokens": [False]},
        ]
        for config in mutations:
            with self.subTest(config=config):
                self._reject(tokenizer_bytes, _json(config))

    def test_incomplete_added_or_duplicate_special_sets_rejected(self) -> None:
        tokenizer_bytes, config_bytes = synthetic_inputs()
        original = json.loads(config_bytes)
        for extras in (
            original["extra_special_tokens"][:-1],
            [*original["extra_special_tokens"], "<|new|>"],
            [*original["extra_special_tokens"], original["extra_special_tokens"][0]],
        ):
            changed = {**original, "extra_special_tokens": extras}
            self._reject(tokenizer_bytes, _json(changed))
        changed = {**original, "bos_token": "<|new|>"}
        self._reject(tokenizer_bytes, _json(changed))

    def test_malformed_and_duplicate_config_json_has_fixed_errors(self) -> None:
        tokenizer_bytes, config_bytes = synthetic_inputs()
        for payload in (
            b"",
            b"\xff",
            b"null",
            b"[]",
            b"{",
            b'{"a":NaN}',
            b'{"a":1,"a":2}',
            config_bytes[:-1] + b',"backend":"tokenizers"}',
        ):
            self._reject(tokenizer_bytes, payload)
        for payload in (b"", b"\xff", b"{}", b"SYNTHETIC_SENTINEL"):
            self._reject(payload, config_bytes)

    def test_bytes_and_size_limits_are_enforced(self) -> None:
        tokenizer_bytes, config_bytes = synthetic_inputs()
        for value in (None, True, tokenizer_bytes.decode(), bytearray(tokenizer_bytes)):
            self._reject(value, config_bytes)
        for value in (None, True, config_bytes.decode(), bytearray(config_bytes)):
            self._reject(tokenizer_bytes, value)
        with patch.object(memory, "MAXIMUM_TOKENIZER_BYTES", len(tokenizer_bytes) - 1):
            self._reject(tokenizer_bytes, config_bytes)
        with patch.object(memory, "MAXIMUM_CONFIG_BYTES", len(config_bytes) - 1):
            self._reject(tokenizer_bytes, config_bytes)

    def test_constructor_does_not_silently_change_vocabulary(self) -> None:
        tokenizer_bytes, config_bytes = synthetic_inputs()
        namespace = importlib.import_module("transformers.tokenization_utils_tokenizers")
        original = namespace.TokenizersBackend.get_vocab

        def changed(instance):
            return {**original(instance), "SYNTHETIC_SENTINEL": 99999}

        with patch.object(TokenizersBackend, "get_vocab", changed):
            self._reject(tokenizer_bytes, config_bytes)

    def test_configuration_bytes_are_unchanged(self) -> None:
        payloads = synthetic_inputs()
        original = copy.deepcopy(payloads)
        memory._construct_tokenizer(*payloads)
        self.assertEqual(payloads, original)


if __name__ == "__main__":
    unittest.main()
