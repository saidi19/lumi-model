# Lumi L1 SFT source curation v0.1

Status: **representative source audit complete and failed closed; no source is production-approved**

Observed: 2026-08-06

## Why this gate is needed

The final frozen base evaluation has mean held-out loss `2.0280788413`
(perplexity `7.59947`) over `4,996,727` prediction tokens. English factual,
science, and code continuations are recognizable, but the fixed Bengali
continuation is malformed. This is evidence to prioritize high-quality Bengali
instruction records and Bengali-specific evaluation; it is not a benchmark
claim or permission to train.

Every SFT record must bind an exact source ID, dataset revision, document ID,
license, URI, content hash, transform hash, quality review, PII/secret scan,
task, language, and split. Dataset-card metadata alone cannot promote a source
to production.

## Current source decisions

| Source | Revision | Declared license | Current decision | Main remaining work |
| --- | --- | --- | --- | --- |
| OpenAssistant OASST1 | `fdf72ae0827c1cda404aff25b6603abec9e3399b` | Apache-2.0 | Exact 1,745 records / 477,456 supervised tokens passed curation verification and 32/32 owner review; reviewed metadata is registered in immutable candidate registry `v0.3`, while the public source remains non-production `canary_eligible` | retain exact bindings while completing the remaining 10M-30M learning mix, cross-source deduplication, seal, and SFT authorization |
| CohereLabs Aya | `f9ea04583f02a8f86404ff6c58bf75fe637df8a2` | Apache-2.0 | Exact private candidate and attribution manifest materialized at 1,478 records / 577,453 supervised tokens; deterministic owner sample passed 32/32; reviewed metadata is registered in immutable candidate registry `v0.4`; exact 1,330-train/148-validation split is privately materialized, non-production | retain exact candidate, attribution, owner, and split bindings while completing the remaining 10M-30M mix |
| OpenR1-Math-220k | `e4e141ec9dea9f8326f4d347be56105859b2bd68` | Apache-2.0 | Bounded selector/PII diagnostics passed and an exact 93,733-row/10-shard full-curation preflight is sealed; no full download or promotion | separately authorize the exact acquisition; then measure verified traces, 4K fit, source-specific PII, upstream rights/attribution, split isolation, and quality |
| UltraChat 200k | `8049631c405ae6576f93f445c6b8166f76f5505a` | MIT | Blocked | generated-output terms and upstream prompt provenance |
| BigCode self-OSS instruct 50k | `356bb069eee815daa6e23e9a282eeefe1490ad44` | ODC-BY-1.0 | Blocked | row-level repository/code license mapping and execution checks |
| Salesforce xLAM 60k | not pinned | CC-BY-4.0 | Blocked | gated terms, exact revision, tool execution revalidation |
| Lumi AI-assisted SFT pilot | `0456f47ceaa1b462ee1a5f48fb7635e006e7e09295d01ad1f569185a5e04b10e` | Lumi-owned | Candidate eligible; owner sample passed | broader SFT mix and measured production token target |
| Lumi-owned SFT | not created | Lumi-owned | Planned | author declarations, records, independent review |

`configs/lumi-4b-sft-source-registry-v0.1.json` remains the registry bound by
the active production data contract. Append-only candidate successors now run
through `v0.4`; that latest successor registers reviewed OASST1 and Aya metadata
without changing the production contract. Registry status is deliberately
stricter than a dataset card: canary eligibility permits only bounded
inspection and adapter testing; it does not permit a production seal.

## Enforced implementation

- The SFT policy binds the source registry by exact SHA-256.
- Every accepted record must carry an exact 40-hex external Git revision or a
  64-hex first-party content revision and known source ID.
- Blocked/planned sources cannot pass the record filter. A hash-pinned
  `candidate_eligible` first-party AI-assisted source may pass local candidate
  filtering, but cannot pass the production seal while blockers remain.
- Source/task/language/split/license/revision drift fails closed.
- Existing PII, secret, exact duplicate, normalized duplicate, and benchmark
  contamination gates remain active.
- The sealing implementation revalidates and encodes every record, binds the
  final 80B-token base checkpoint, records raw-text-free token/distribution
  evidence, and refuses any source with unresolved production blockers.
- No source is currently production-eligible, so a real SFT seal must fail.

## Bounded source canary

`configs/lumi-4b-sft-source-canary-v0.1.json` defines the first CPU-only source
canary. It is deliberately limited to at most 100 OASST1 rows, 24 Bengali Aya
rows, and 24 OpenR1 rows, with at most 24 candidate records per source and a
16 MiB aggregate response cap. Before and after row retrieval, the runner
requires the public Hub head to equal the registry's exact source revision. Raw
responses and accepted candidate text must remain outside Git; the report
contains only counts, hashes, distributions, rejection reasons, and
token-length measurements.

The canary runs the normal `lumi.sft.v0.1` schema, source binding, sensitive
pattern scan, exact/normalized deduplication, held-out benchmark contamination
check, frozen-tokenizer chat serialization, assistant-only loss mask, and the
4,096-token reject gate. Passing this canary is measurement evidence only. It
does not make any source production-eligible and does not authorize SFT.

The completed attempt accepted 24 records from each source (72 total), with no
schema/filter/contamination rejection and 100% exact 4K fit. The longest encoded
conversation was 1,473 tokens. All 24 Aya records contained Bengali script.
This small deterministic slice cannot establish full-dataset quality or safety;
the sealed raw-text-free report SHA-256 is
`0270e336b3eb66b3f2564a5418e45c15b1581454f2afa3a52ee254f6f4cf941d`.

## Representative quality and provenance audit

The existing 72 accepted canary rows and 320-row AI-assisted candidate were
audited without bulk download, paid GPU, or production promotion. The audit is
complete but deliberately failed closed with `0` production promotions and `16`
recorded blocker findings. Its raw-text-free report SHA-256 is
`0c8644ae02d6f661e3a0b43d4de07f4ad5029b0b76b2a27bb576743d2c7a37bd`.

- OASST1: `15/24` accepted follow-up pairs omit required ancestor turns, only
  `9/24` are root-direct pairs, and `4/24` admit an unranked reply. The adapter
  must preserve a complete selected path or explicitly restrict itself to
  root-direct, ranked examples before another audit.
- Aya Bengali: one annotator supplies `15/24` rows (`62.5%`), and `5/24` rows
  retain a trailing formatting artifact. A bounded full-Bengali-subset audit
  must measure concentration and instruction quality before promotion.
- OpenR1: all `24/24` accepted assistant outputs match the reference `solution`
  field rather than a verified generated reasoning trace. The slice also has
  `4` mixed-completeness rows, `2` external-image dependencies, `3` external
  URLs, and `3` unusually short reference solutions. Row-level upstream
  provenance and full-set 4K fit remain unresolved.

The repaired OASST1 adapter was then re-audited against the exact existing
100-row pinned slice without another download. Strict root-direct/rank-0
selection admitted 9 records; the accepted set contains 0 context-free
follow-ups, 0 unranked replies, and 0 missing raw bindings. This closes the two
measured adapter defects but does not promote OASST1: full selection/quality
policy and attribution evidence remain open. The raw-text-free receipt seal is
`22472021420ee8fba53b01e24126e6169956d3987e901d8c7acb700e0555f4b3`.

Those two remaining OASST1 design blockers first received a local preflight. The
sealed plan pins both official splits (`88,838` rows), exactly two parquet
objects plus `LICENSE` and `README.md`, a `64 MiB` network cap, English-only
human root-direct/rank-0 selection, minimum review counts, prompt quality
`>=0.5`, answer quality `>=0.75`, a 35% maximum final train-token share, and a
per-record attribution schema that excludes raw text and `user_id`. Reusing the
same pinned 100-row sample, it selected 4 high-quality candidates from 10
structural pairs. That preflight fetched no full dataset and promoted no source.
Its internal seal is
`686d39d0024226bb3cdc60de870c74326b5d6556c4e6da7df10703fa2b9296d9`.

The later owner-authorized CPU run fetched only those four pinned objects:
`41,617,955` network bytes, exactly `88,838` rows, and `106,039,954` PyArrow
decoded bytes. The strict selector admitted `1,763` pairs. PII/secret filtering
rejected 17 records (one matched both classes), the 4K gate rejected one more,
and `1,745` records remained with `477,456` supervised tokens. All source
objects, attributions, filters, benchmark decontamination, deduplication,
assistant-only labels, and token counts passed an independent raw-text-free
verification. The deterministic 32-row owner sample passed 32/32 and reviewed
metadata is now registered in immutable candidate successor `v0.3`; production
eligibility, the active data contract, dataset seal, and SFT remain unchanged.

The pinned Aya Bengali subset was then audited under a 1,534-row/64 MiB CPU-only
boundary. Official dataset statistics and the bounded fetch both report 1,534
rows. The top annotator contributes 789 rows (`51.43%`), exceeding the 50%
limit by 22 rows; 223 rows retain a trailing `|`; and 4 exact prompt/target
duplicates require deduplication. Language labels and empty-field checks pass.
The dataset-viewer wrapper index was unstable across offset pages, so production
materialization must use content-addressed identity. That three-row index
uncertainty cannot reverse either the concentration or format-artifact verdict.
The authoritative raw-text-free reconciliation receipt seal is
`7b928859f668d6263b7a2331b4d3c7976f2c93992289ccad0ab364a69f99002e`;
production promotions remain zero.

The exact existing Aya inventory was then replayed locally without network. The
deterministic repair preflight removed all 223 single terminal-pipe artifacts,
four normalized duplicate pairs, three rows without Bengali script, one over-4K
row, and 48 additional dominant-annotator rows. Under a later bounded owner
authorization, the exact 1,478-record / 577,453-supervised-token private
candidate, raw-text-free attribution manifest, and deterministic 32-row owner
packet were materialized outside Git. The top annotator remains capped at
exactly 50%. A pre-review inline-JavaScript newline-escaping defect was repaired
and syntax-checked without changing candidate or attribution data. The
raw-text-free materialization report internal seal is
`a2f08b1594a2b6fd9def1f43a48e072cbe0f107f4bc7ea43b391d6066ddc6ade`;
the exact owner sample then passed 32/32 with raw-text-free receipt seal
`e23851af5718d10acdd120ca1e0c200672fb5c794ad503ed4d1f7f1650cc685f`.
Source-registry promotion, dataset sealing, and SFT remain separate and
incomplete.

The combined `392`-record diagnostic pool is not a trainable target mix. It has
zero `safety_helpfulness` records and materially sparse `coding`, `factual_qa`,
and `writing_summarization` coverage. These gaps must be filled with reviewed,
rights-clear records rather than by relabeling unrelated examples.

## Lumi identity and introduction

`configs/lumi-4b-identity-sft-spec-v0.1.json` records the intended first-party
identity slice. It now contains short, warm first-run, collaborative,
capability-aware, and technical introductions in English, Bengali script, and
Banglish. The canonical fact remains “I'm Lumi, an AI assistant developed by
Afnan Technologies Inc.”, but that sentence is not a mandatory prefix. Lumi
should identify itself when asked or during explicit product onboarding, not
append a slogan to unrelated answers, repeat the company name on every greeting,
or memorize one rigid response.

`configs/lumi-4b-assistant-behavior-spec-v0.1.json` defines the broader assistant
character. The target is direct, warm, candid, context-aware conversation with
natural language switching, light situational humor, honest uncertainty, error
correction, and respectful disagreement. Natural does not mean pretending to be
human: Lumi must not claim feelings, lived experience, sentience, tool results,
or current knowledge that it does not have. It must also resist sycophancy rather
than changing a factual answer simply to please the user.

The planned behavior slice contains 6,000 train and 920 validation
records across multi-turn conversation, Bengali/Banglish, uncertainty and error
correction, anti-sycophancy, tool honesty, identity, and helpful safety behavior.
Those counts are a review plan, not existing data and not the complete SFT mix.
After SFT, a separately authorized preference stage may rank natural/direct
responses over robotic, flattering, repetitive, or literal-translation variants.
No proprietary assistant transcript or persona text is approved as training
data; public behavior documents are design references only. The active bounded
pilot uses explicit AI provenance and a single-owner dataset-level spot-check;
it never claims that model-generated records were written by humans.

The identity slice must also teach honest boundaries: Lumi may claim connected
search or RAG only when those tools are actually available, must distinguish
model knowledge from retrieved information, and must not claim to be ChatGPT,
made by OpenAI, human, sentient, or capable of something unverified. The file is
a specification, not training data. Generated identity candidates still require
the owner gate and later post-SFT identity evaluation before promotion or release.

## First-party authoring pilot

`configs/lumi-4b-first-party-authoring-pilot-v0.1.json` defines a bounded pilot
of 240 train and 80 validation slots. It explicitly covers natural greetings
and small talk, English/Bengali/Banglish typo recovery, exact simple arithmetic,
calculator-tool routing, multi-turn correction, and Lumi identity boundaries.
The generated template starts with blank text fields and requires original human
writing, a rights declaration, an independent reviewer, a safety pass, and a
minimum quality score. Model-generated training text, raw text in Git, paid GPU
work, and production SFT remain unauthorized.

`configs/lumi-4b-first-party-authoring-workflow-v0.1.json` and
`docs/first-party-sft-authoring-runbook-v0.1.md` operationalize the pilot as 16
private packets of 20 rows, each with 15 train and 5 validation slots. The
packet builder generates no text. The merge gate rejects missing or duplicate
record IDs and emits only raw-text-free progress counts. Real author/reviewer
identity assignment is still incomplete, and slot names do not prove review
independence.

The current low-headcount path is separately defined by
`configs/lumi-4b-ai-assisted-sft-pilot-v0.1.json` and
`docs/ai-assisted-sft-pilot-runbook-v0.1.md`. It produced 320 exact candidates,
all of which passed deterministic and normal SFT filtering with explicit
`ai_generated` provenance. The deterministic 32-row owner sample passed, and
the candidate remains intentionally not production-eligible because it is not
the complete SFT mix. The human-only packet
workflow remains available as an optional higher-assurance route rather than an
immediate staffing blocker.

`configs/lumi-4b-basic-capability-gate-v0.1.json` separately defines 256
deterministic integer arithmetic cases—64 each for addition, subtraction,
multiplication, and exact division—plus human-scored greeting and typo suites.
Model-only arithmetic is diagnostic because a language model can still make a
calculation error. Calculator-backed arithmetic is required to pass every case
before the product may present it as reliable calculation.

`configs/lumi-4b-language-routing-gate-v0.1.json` freezes response-language
precedence without storing prompt rows or claiming model inference. Explicit
response-language requests override detection; otherwise Lumi should mirror the
latest clear user language, fall back to established conversation language for
short ambiguous turns, and use English only when no signal exists. English,
Bengali script, Banglish, and code-switching are core targets. Other languages
must remain best-effort until language-specific SFT data and frozen evaluations
pass. Technical English terms, names, code, commands, and numbers alone must not
force an otherwise Bengali or Banglish conversation into English.

## Next bounded work

1. **Completed:** deterministic 32-row owner spot-check for the AI-assisted
   pilot, with exact source revision and explicit limits preserved.
2. **Completed:** restrict OASST1 to ranked root-direct pairs and repeat the
   bounded audit; the structural findings are closed.
3. **Completed locally, no fetch:** seal OASST1's exact full-source bounds,
   source-level quality policy, split handling, attribution fields, and
   downstream filtering requirements. Acquisition and production promotion
   remain separate.
4. **Completed fail-closed:** measure the full-count Aya Bengali subset and
   record its concentration, formatting, deduplication, and pagination findings.
5. **Completed bounded and fail-closed:** the repaired OpenR1 selector reused the
   exact pinned 24-row slice without network access, selected 18 verifier-passed
   complete `generations` traces instead of reference `solution` text, admitted
   12 through normal filtering, and admitted 6 through strict 4K encoding. The
   source remains non-production because the audit found external-dependency,
   safe-trace availability, PII-pattern, context-length, full-set measurement,
   quality-review, rights/attribution, and attribution-manifest blockers.
6. **Completed, non-production:** run the separately authorized OASST1
   acquisition/curation gate and independently verify the resulting 1,745
   records / 477,456 supervised tokens.
7. **Completed human quality sample:** the owner reviewed OASST1's exact
   deterministic 32-row sample, accepted 32, rejected 0, and the raw-text-free
   receipt passed. This does not promote the source or claim first-party
   authorship.
8. **Completed candidate-registry promotion:** immutable registry `v0.3`
   changes only OASST1's reviewed metadata entry. The active data contract,
   production coverage, dataset seal, and SFT authorization remain unchanged.
9. **Completed bounded materialization:** reproduce the exact 1,478-record /
   577,453-supervised-token Aya repair candidate, write its private candidate and
   attribution manifest outside Git, and prepare the deterministic 32-row owner
   packet. The raw-text-free receipt passed and every production gate remains
   closed.
10. **Completed owner quality sample:** the owner accepted all 32 exact Aya
   rows; the fail-closed raw-text-free receipt passed and retained zero
   production promotions.
11. **Completed candidate-registry promotion:** immutable registry `v0.4`
   changes only Aya's reviewed metadata entry, binds the exact 1,478 records /
   577,453 supervised tokens and 32/32 owner sample, and retains all production
   blockers. The predecessor and active data contract remain unchanged.
12. **Completed split preflight:** cluster Aya
   prompts and responses at MinHash threshold 0.90, keep every connected cluster
   wholly within one split, and project 1,330 train / 148 validation records.
   Exact and near-duplicate cross-split overlap are zero, each split retains a
   50% annotator cap, and the exact tokenizer measures 520,269 train plus 57,184
   validation supervised tokens.
13. **Completed exact split materialization:** write the approved private split
   candidate, raw-text-free manifest, attribution binding, and receipt outside
   Git. All 1,478 records and 577,453 supervised tokens reconcile; private
   writes total 5,480,313 bytes and no production promotion occurred.
14. **Completed bounded OpenR1 secondary PII diagnostic:** line/message-scoped
   analysis resolves the prior two conservative cases. Across the six global
   phone-regex records, five are math-number collisions and one is a
   cross-message join artifact; unresolved is zero, with no global filter
   change and no source promotion.
15. **Completed locally, no fetch:** seal OpenR1's exact default-config boundary
   at 93,733 rows, 10 parquet shards plus README, 2,149,897,914 expected data
   download bytes, 4,964,543,659 decoded bytes, and a 2,415,919,104-byte network
   ceiling. The plan preserves the global PII filter, requires generated
   verifier-passed traces, and quarantines records until upstream source rights
   and attribution are mapped. No acquisition or promotion occurred.
16. **Completed fail-closed for OpenR1; DMath candidate promotion complete and
   DeepMind blocked safely before network:** OpenR1's
   full-set visible-final candidate and 32/32 owner sample completed, but the
   exact seven-label rights audit cleared zero rows and quarantined all 2,656.
   A separate no-download replacement preflight now pins DMath's MIT-licensed
   revision `149a7403619ac1bd6ed10f50694ad82699d5cc03` as the next bounded source:
   `LICENSE`, `README.md`, and `dataset/train.json`, totaling 12,206,887 bytes
   and 7,943 expected train rows; the test split stays reserved. DeepMind's
   generator is a later fallback behind a separate deterministic adapter gate.
   DMath acquisition, curation, 32/32 owner review, and append-only candidate
   promotion are now complete: immutable registry `v0.5` adds the exact 6,943
   records / 51,026 supervised tokens as `canary_eligible`, without modifying
   the ten predecessor sources or active production data contract. The bounded
   DeepMind acquisition/canary then stopped at its mandatory dependency gate
   before any network access because `absl` and `six` are absent. Independent
   evidence records zero fetched objects and zero generated records. A bounded
   no-install follow-up then verified the pinned `setup.py` and official PyPI
   metadata, selecting exact hashed universal wheels for `absl-py==2.5.0` and
   `six==1.17.0`; NumPy 2.5.1 and SymPy 1.14.0 remain preinstalled. No package
   was installed by that resolution step. A separate bounded install then
   placed exactly those two wheels in `/private/tmp/lumi-deepmind-runtime-v0.1`;
   its 40-file / 520,097-byte inventory and fresh-process import paths
   independently verify. No canary row was generated. **Next:** separately retry
   the exact 30-object / 2,000-row canary and continue one rights-clear source lane
   at a time for coding, factual QA,
   writing, identity, natural conversation, anti-sycophancy, and
   Bengali/Banglish. Never label synthetic text as human or use an unlicensed
   mix.
17. Promote only sources whose evidence closes every registry blocker, then seal
   immutable train/validation manifests.

Aya's 32-row sample supports its reviewed Bengali naturalness decision, but it
does not prove punctuation capability. The separate punctuation gate freezes 72
raw-text-free coverage lanes across English, Bengali script, Banglish, six error
types, and four prose/protected-span contexts. Before release, actual post-SFT
inference must cover at least 288 private cases; meaning and protected code,
command, URL, path, decimal, version, and math spans have zero-tolerance
preservation requirements.

No paid GPU, SFT run, preference optimization, public deployment, bulk download,
or legal conclusion is authorized or implied by this metadata audit.
