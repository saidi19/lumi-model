# Lumi L1 missing SFT mix proposal v0.1

Status: **planning complete; data acquisition, production source promotion, production
dataset sealing, paid GPU use, and SFT training remain unauthorized**

Observed: 2026-08-06

## Decision

Build the smallest rights-clear and evaluation-complete SFT mix first. The
first learning candidate targets **10M-30M accepted train supervised tokens**,
measured only after the frozen chat template, assistant/tool loss masking,
filtering, context-fit checks, and deduplication are applied. Validation is
frozen separately and is not counted as training tokens.

The existing 320-row AI-assisted pilot proves the behavior schema, language
routing examples, arithmetic/tool checks, filtering, and owner spot-check
workflow. It is not treated as a complete SFT dataset or as millions of tokens.

A separate 320-row Lumi-owned safety/tool candidate now covers five safety and
five tool-use skills across English, Bengali script, Banglish, and natural
code-switching. Machine gates and exact 4K tokenization passed: its train split
contains 18,636 supervised tokens and its validation split contains 6,388. The
largest serialized row is 256 tokens. The human owner accepted all 40 sampled
rows with zero rejections, and the fail-closed raw-text-free receipt passed.
This clears only the sample-review gate: the lane remains non-production with
zero production promotions. A separate append-only decision has now registered
it as `candidate_eligible` in successor registry `v0.2`; the active production
contract remains unchanged. The lane fills only about 1.16% of the
safety-plus-tool minimum (1.6M supervised tokens) and 0.19% of the complete 10M
learning-candidate floor.

## First learning-candidate task bands

| Task | Share | 10M floor | 30M ceiling |
| --- | ---: | ---: | ---: |
| Coding | 20% | 2.0M | 6.0M |
| Conversation | 5% | 0.5M | 1.5M |
| Factual QA | 15% | 1.5M | 4.5M |
| Bengali/multilingual behavior | 10% | 1.0M | 3.0M |
| Reasoning, math, and science | 22% | 2.2M | 6.6M |
| Safety and helpfulness | 8% | 0.8M | 2.4M |
| Tool and API behavior | 8% | 0.8M | 2.4M |
| Writing and summarization | 12% | 1.2M | 3.6M |

The matching language plan is 78% English, 10% Bengali script, 8% Banglish,
and 4% natural code-switching. Task and language are independent dimensions:
for example, Bengali safety and Banglish tool-use records count in both their
task category and language mode.

## Source lanes

- Lumi-owned behavior data supplies identity, greeting, correction,
  Bengali/Banglish routing, safety, and honest tool behavior while preserving
  explicit AI-assisted provenance.
- OASST1 may supply reviewed human dialogue only after its rank/thread policy,
  row quality, language, and attribution gates are sealed.
- Aya may supply Bengali records only after annotator concentration, formatting,
  duplicate identity, instruction-compliance, and attribution defects are
  repaired.
- OpenR1 may supply verifier-passed generated reasoning traces only after
  full-set 4K fit, upstream problem rights, attribution, quality, and remaining
  PII-pattern review are complete.
- Coding must be backed by row-level rights and executable tests.
- Factual QA must be grounded in revision-pinned, rights-clear documents with
  answer traces and benchmark decontamination.
- Writing and summarization must use rights-clear source documents and pass
  faithfulness, diversity, and split-isolation checks.

No single source may exceed 35% of accepted train supervised tokens without a
new owner decision. Synthetic output remains labelled synthetic and cannot be
relabeled as human.

## OpenR1 PII diagnostic outcome

The six generic `pii_pattern` rejections were reviewed without retaining raw
text. The initial conservative pass left two records unresolved; the subsequent
line/message-scoped diagnostic resolves the final classification as five
assistant-side math-number collisions and one cross-message join artifact.
Unresolved is zero for the bounded slice. The global PII filter was not changed,
no record was promoted, and the original verified-trace audit remains immutable.

The local full-curation preflight now pins OpenR1's default config at exactly
93,733 rows in 10 parquet shards plus README. It records 2,149,897,914 expected
data download bytes, 4,964,543,659 decoded bytes, a 2,415,919,104-byte network
ceiling, generated-trace-only selection, and fail-closed upstream rights and
attribution requirements. It authorizes no download or promotion.

## Exact next execution order

1. **Completed for one small lane only:** build and machine-validate the bounded
   Lumi-owned safety/tool candidate without an external API, network fetch, GPU,
   production promotion, or training.
2. **Completed:** the human owner reviewed the deterministic 40-row validation
   sample and accepted 40/40 with zero rejections; the receipt is sealed.
3. **Completed as candidate-only:** register exactly this lane in immutable
   successor registry `v0.2`; no remaining lane or production dataset was
   promoted.
4. **Completed for OpenR1 planning only:** seal the exact full-source object,
   row, byte, trace-selection, PII, rights, attribution, and review boundaries
   without a fetch.
5. **Current:** for every remaining acquisition, including OpenR1, issue a
   separate bounded authorization; this proposal itself does not authorize a
   download.
6. Pin source revision, license, attribution requirements, row identity, and
   maximum acquisition size before reading records.
7. Normalize with explicit human/synthetic/first-party provenance and run
   schema, PII/secret, quality, context-fit, deduplication, and benchmark
   contamination checks.
8. Review deterministic task/language samples and repair or reject defective
   sources; never pad a weak category with noisy records.
9. Apply the frozen chat template and measure accepted serialized tokens and
   assistant/tool supervised tokens by source, task, language, and split.
10. Stop acquisition when the complete 10M-30M learning mix is satisfied, freeze
   validation, and seal the exact production dataset.
11. Run the production-seal-bound CPU loss/resume canary. Only after it passes is
   a separately authorized two-step GPU SFT smoke appropriate.

The machine-readable proposal is
`configs/lumi-4b-sft-missing-mix-proposal-v0.1.json`. It deliberately authorizes
no data acquisition, API generation, GPU work, production promotion, or SFT
training.
