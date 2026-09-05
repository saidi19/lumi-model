# Lumi L1 post-training readiness v0.1

## Current boundary

Lumi L1 base pretraining is complete at exact step `76,294` and exactly
`80,000,000,000` tokens. The final checkpoint manifest is
`73c852095441a4246cea7bb97f7831c364137869e00e29c815b7e22a5972f8de` and
the sealed handoff is
`8258d67fa0cd53a9b1075c4f5f0c17e14e40fb928ef2a53f36a523f55763de4b`.
The 52-object archive and its verification receipt are preserved in private,
encrypted, versioned S3; the independent exact-prefix verification reports 53
versions, zero delete markers, zero incomplete multipart uploads, and an
unchanged source checkpoint.

`configs/lumi-4b-final-base-binding-v0.1.json` is the only accepted base input
for post-training v0.1. A different checkpoint, run, tokenizer, or handoff must
fail closed rather than silently creating another model lineage.

## Training-ready chat contract

`configs/lumi-4b-chat-template-v0.1.json` uses the chat tokens reserved before
base pretraining:

- `<|system|>`, `<|user|>`, `<|assistant|>`;
- `<|tool|>` and `<|tool_result|>`;
- `<|message_end|>`, BOS, EOS, and PAD.

Raw message content is encoded in reserved-token-safe mode, so text cannot
inject control-token IDs. Role tokens, system/developer text, user text, tool
results, and padding receive label `-100`. Only assistant responses, tool calls,
their message-end tokens, and the final EOS contribute to SFT loss. Records
longer than the exact 4,096-token context are rejected; they are never silently
truncated.

Initial SFT materialization keeps one conversation per sequence. Cross-example
packing remains blocked until a separate block-diagonal attention/leakage gate
passes. Batch padding uses token ID 0, attention-mask 0, and label `-100`.

The supported tool trajectory is:

`user -> tool -> tool_result -> assistant`

with additional tool/tool-result cycles allowed before the final assistant
answer. Ordinary `user -> assistant` multi-turn conversations remain valid.

## What is ready

- final base/checkpoint/tokenizer/S3 evidence binding;
- licensed/provenance-aware `lumi.sft.v0.1` record validation;
- exact-revision source binding and metadata-only source admissibility report;
- PII, secret, duplicate, and benchmark-contamination rejection;
- exact chat serialization and assistant-only loss masking;
- a 72-record, three-source CPU-only canary with exact revision rechecks, 100%
  4K fit, and raw-text-free evidence;
- a raw-text-free representative source-quality/provenance audit that completed
  fail-closed with zero promotions and 16 blocker findings;
- deterministic raw-text-free sealing that refuses canary-only sources;
- frozen base-evaluation binding and post-training safety plan;
- fail-closed serving contract that now marks only base pretraining as passed;
- a local-only safe Markdown/code/KaTeX renderer with no runtime CDN, inert raw
  HTML, trust-disabled math extensions, malformed-math fallback, and packaged
  fonts.

## What is not yet claimed

- no production-eligible SFT dataset has been curated or sealed;
- no candidate source has passed its production promotion blockers;
- no SFT token target has been selected;
- no SFT training, preference optimization, or safety tuning has run;
- no post-training benchmark has passed;
- no paid GPU work or public deployment is authorized.

## Next execution gate

The first bounded canary normalized 24 records from each of OpenAssistant
OASST1, Aya Bengali, and OpenR1-Math. All 72 passed the current schema,
source-binding, sensitive-pattern, duplicate, benchmark-overlap, exact chat
encoding, assistant-only label, and 4K context gates. This is adapter and sample
measurement evidence, not a production quality claim. The report SHA-256 is
`0270e336b3eb66b3f2564a5418e45c15b1581454f2afa3a52ee254f6f4cf941d`.

The follow-up representative audit completed fail-closed with report SHA-256
`0c8644ae02d6f661e3a0b43d4de07f4ad5029b0b76b2a27bb576743d2c7a37bd`
and zero production promotions. It found dropped OASST1 ancestor history and
unranked replies, Aya annotator concentration and formatting artifacts, and an
OpenR1 adapter that selected reference solutions rather than verified generated
traces. Across the 72 external rows plus 320 AI-assisted rows, safety coverage
is zero and coding, factual-QA, and writing coverage is sparse.

OASST1's full pinned source is now curated to 1,745 strict records / 477,456
supervised tokens, has passed independent verification and 32/32 owner review,
and is registered candidate-only in successor registry `v0.3`. Aya's reviewed
1,478 records / 577,453 supervised tokens are registered candidate-only in
successor `v0.4`, and its 1,330-train/148-validation split is privately
materialized.

OpenR1's pinned 93,733-row source is also fully curated. Its visible-final lane
retains 2,656 records / 1,170,745 supervised tokens after 32 malformed-style
rows were quarantined, and the owner accepted all 32 deterministic sample rows.
However, the exact raw-text-free rights-map audit reproduced seven upstream
problem-source labels and cleared zero records: all 2,656 remain quarantined
because the current evidence does not document per-source downstream rights.

The immediate gate is therefore rights-clear replacement or documented rights
for the blocked math/reasoning lane, followed by the remaining coding,
factual-QA, writing, conversation, multilingual, and safety/tool coverage needed
to measure the accepted 10M-30M target mix. OpenR1 must not enter the production
seal merely because its dataset card declares Apache-2.0.
Production sealing remains fail-closed until every source promotion blocker
closes. Any bulk data access or paid GPU pilot needs a separate exact
authorization.
