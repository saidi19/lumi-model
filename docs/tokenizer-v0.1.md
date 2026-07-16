# Lumi tokenizer decision record v0.1

Status: **frozen for Lumi tokenizer v0.1 at 65,536 tokens; full model training remains unapproved**

## Frozen design

- Byte-level BPE with NFC Unicode normalization and a 32-character maximum
  learned-token length to prevent repeated whitespace or punctuation from
  consuming pathological vocabulary entries.
- Frozen vocabulary size: 65,536 tokens.
- English-first sampling, with code, mathematics, structured data, and a
  multilingual slice so later language support is not blocked by unknown
  characters.
- The 150M pilot and 4B candidate use the same frozen vocabulary. This makes
  pilot loss, memory, and throughput measurements relevant to the final model.

Byte-level coverage prevents out-of-vocabulary text; it does not guarantee
efficient tokenization for every language.

## Full-size bakeoff

On 2026-07-16, corrected 32K and 65K candidates were trained from the exact
same sealed sample with identical normalization, special-token IDs,
`min_frequency=2`, and a 32-character learned-token cap. The evaluation covered
all 1,128,089 documents and 6,000,636,724 accepted text bytes in sample
SHA-256 `80bea9f9634a8daca542b2ffd0aad1c8c994a7cbeef909a5e529b6272671102a`.

| Candidate | `tokenizer.json` SHA-256 | Tokens | Bytes/token | Tokens/word |
|---|---|---:|---:|---:|
| 32K | `1b0729f951acbca41dde755daf4a3aa7c531beee5c7af6cb489ef6d1f60057cf` | 1,544,062,326 | 3.886266 | 1.709418 |
| **65K (selected)** | `1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee` | **1,476,228,500** | **4.064843** | **1.634320** |

The 65K candidate emits 67,833,826 fewer tokens, a 4.393205% reduction.
The gain is not confined to one source family:

| Domain | 65K token reduction vs. 32K |
|---|---:|
| Books | 4.87% |
| Code | 4.14% |
| Government, legal, and patents | 5.03% |
| Multilingual reference | 4.08% |
| Open web | 5.27% |
| Questions and discussion | 3.16% |
| Reference | 5.42% |
| Science and education | 4.25% |

The fixed probes were never worse with 65K: Bangla was 40 to 39 tokens, code
28 to 26, Japanese 43 to 32, and English, mathematics, and structured JSON were
ties. Probe strings are sanity checks, not substitutes for the full-domain
measurements.

The full comparison report is stored outside Git at
`tokenizer-candidates-v0.2/comparison-v0.2.json`; its SHA-256 is
`dd4403b8bbeb862cbbad3596c820586d02be6ddd576d8bc804a3ad61c805483f`.
The frozen 65K directory also pins:

- `tokenizer_config.json`: `d31c91cf95389f72e3b5edcb71b67129e259e4f36149fb435e0a3cc15437d0a5`
- `lumi_tokenizer_metadata.json`: `16964e683e2ea264d54ca6129cb2c6eec51ec07c3481495d6f16727ff3d71047`

## Decision

Lumi v0.1 freezes the 65K candidate. All eight measured domains improve, and
the 4.39% sequence reduction is valuable for the 4B target's training and
inference context. Relative to 32K, the tied embedding table adds 100,663,296
parameters to the 3,072-wide 4B model: about 192 MiB in BF16 or 48 MiB as raw
4-bit weights. The 768-wide 150M plumbing model adds 25,165,824 parameters;
that pilot exists to validate the final tokenizer and pipeline, not to optimize
a standalone 150M product.

This freeze covers normalization, training settings, vocabulary, and all
special-token IDs. Any change requires a new tokenizer version and model
pretraining from a fresh initialization; it is not an in-place model update.

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

## Completed freeze gates

1. The provenance-tracked 6GB representative sample is sealed and fully
   validated; the entire 80B pretraining corpus was not used for tokenizer
   training.
2. The 32K and 65K prototypes used identical input, normalization, token cap,
   package versions, and special IDs.
3. Token fertility was measured overall and for all eight source domains.
4. Both candidates pass exact encode/decode round trips, byte coverage,
   special-ID checks, and multilingual/code probes.
5. Both vocabularies have a longest raw learned token of 31 characters and
   zero tokens over the configured limit. A credential-, private-key-, and
   email-shaped full-token scan found no matches. This scan reduces risk but
   does not prove that no source text can ever be reconstructed.
6. The 65K tokenizer is frozen before the real 150M-data pilot. The 4B run is
   still blocked on the real-data plumbing, checkpoint, conversion, and
   measured single-GPU throughput gates.

The frozen machine-readable contract is
[`configs/tokenizer-v0.1.json`](../configs/tokenizer-v0.1.json). For a future
reproduction, run:

```bash
python scripts/train_tokenizer.py \
  --input /path/to/sample.jsonl \
  --output /path/to/lumi-tokenizer-65k
python scripts/check_tokenizer_contract.py /path/to/tokenizer
```

Use `--vocab-size 32768` for a comparison prototype and pass
`--expected-vocab-size 32768` to the checker. The actual tokenizer sample plan
is in [`data-plan-v0.1.md`](data-plan-v0.1.md).
