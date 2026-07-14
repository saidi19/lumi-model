# Lumi tokenizer decision record v0.1

Status: **candidate; training corpus and vocabulary size are not frozen**

## Candidate

- Byte-level BPE with NFC Unicode normalization.
- Candidate vocabulary size: 65,536 tokens.
- English-first sampling, with code, mathematics, structured data, and a
  multilingual slice so later language support is not blocked by unknown
  characters.
- The 150M pilot and 4B candidate use the same target vocabulary. This makes
  pilot loss, memory, and throughput measurements relevant to the final model.

Byte-level coverage prevents out-of-vocabulary text; it does not guarantee
efficient tokenization for every language. Before freezing, train at least a
32K and 65K prototype on the same representative sample and compare token
fertility, downstream validation, embedding cost, and Mac inference memory.

## Frozen special-token IDs

| ID | Token | Purpose |
|---:|---|---|
| 0 | `<|pad|>` | Batch padding only; labels must be `-100` |
| 1 | `<|bos|>` | Packed-sequence or generation start |
| 2 | `<|eos|>` | End of generated sequence |
| 3 | `<|doc|>` | Boundary between pretraining documents |
| 4 | `<|unk|>` | Compatibility fallback; byte-level text should not emit it |
| 5 | `<|system|>` | Reserved for instruct/chat post-training |
| 6 | `<|user|>` | Reserved for instruct/chat post-training |
| 7 | `<|assistant|>` | Reserved for instruct/chat post-training |
| 8 | `<|tool|>` | Reserved for tool calls |
| 9 | `<|tool_result|>` | Reserved for tool results |
| 10 | `<|message_end|>` | End of one chat message |

The first four IDs are part of the model configuration and must remain distinct.
The chat tokens are reserved before base pretraining so `Lumi-4B-Instruct`
does not need embedding-table surgery later.

## Freeze gates

1. Build a provenance-tracked representative tokenizer sample; do not use the
   entire 80B-100B-token pretraining corpus for tokenizer training.
2. Train 32K and 65K prototypes with identical normalization and special IDs.
3. Measure tokens per byte and tokens per word by domain and language.
4. Verify exact encode/decode round trips, no accidental unknown tokens, and
   stable handling of code, whitespace, numbers, Unicode, and malformed text.
5. Inspect the most frequent tokens and a random vocabulary sample for data
   leakage, junk, and source-specific artifacts.
6. Freeze one tokenizer before the real 150M-data pilot. A tokenizer change
   after 4B pretraining means a new model version, not a safe in-place update.

The machine-readable contract is
[`configs/tokenizer-v0.1.json`](../configs/tokenizer-v0.1.json). Once a prototype
exists, run:

```bash
python scripts/train_tokenizer.py \
  --input /path/to/sample.jsonl \
  --output /path/to/lumi-tokenizer-65k
python scripts/check_tokenizer_contract.py /path/to/tokenizer
```

Use `--vocab-size 32768` for the comparison prototype and pass
`--expected-vocab-size 32768` to the checker. The actual tokenizer sample plan
is in [`data-plan-v0.1.md`](data-plan-v0.1.md).
