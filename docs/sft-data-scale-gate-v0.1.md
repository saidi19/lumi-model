# Lumi L1 SFT data scale gate v0.1

Status: **started locally; no bulk acquisition, external API generation, paid
GPU, production dataset seal, or SFT run is authorized**

Observed: 2026-08-06

## What is complete

The 320-row Lumi AI-assisted pilot passed deterministic filtering with 320
accepted and 0 rejected records. Its deterministic 32-row owner sample also
passed. This proves that the schema, language modes, basic arithmetic and tool
format, split isolation, and owner-review mechanism work. It does not make 320
rows a complete SFT dataset.

The earlier combined diagnostic pool has zero safety records and negligible
coding, factual-QA, and writing coverage. The public-source audit also failed
closed with 0 source promotions and 16 provenance or quality findings.

The missing-mix planning gate is now complete. It allocates the first learning
candidate across all eight required tasks and four release language modes while
keeping the target at 10M-30M accepted train supervised tokens. This is a plan,
not acquired data and not permission to train.

The first Lumi-owned safety/tool lane candidate is also machine-complete. It
contains 320 separate records: 240 train and 80 validation, balanced across
five safety skills, five tool-use skills, and four language modes. The exact
frozen Lumi tokenizer accepted all 320 within the 4,096-token context. The
largest record is 256 serialized tokens; the train split contains 29,503
serialized tokens and only **18,636 supervised assistant/tool tokens**. The
validation split contains 6,388 supervised tokens. This is evidence that one
small lane is structurally sound, not evidence that the 10M-30M mix exists.
Its deterministic 40-row owner spot-check is complete: the human owner accepted
40/40 rows with zero rejections, and the raw-text-free receipt independently
binds the exact sample and decision file. This clears the sample-review blocker
only. A separate fail-closed gate then added this exact lane as
`candidate_eligible` in immutable successor registry `v0.2`. The predecessor
registry remains unchanged, and the active production data contract still
points to `v0.1`. The full mix, production seal, and SFT remain separate;
production dataset promotions remain zero.

The next OASST1 gate is now sealed locally. It pins revision
`fdf72ae0827c1cda404aff25b6603abec9e3399b`, exactly two parquet objects plus
`LICENSE` and `README.md`, all `88,838` source rows, a `64 MiB` network cap, and
a `128 MiB` decoded cap. The first production-oriented lane is deliberately
English-only, human/non-synthetic, reviewed, root-direct, rank-0, and bounded by
prompt/answer quality floors. On the already pinned 100-row sample, 10
structural pairs became 4 high-quality candidates spanning coding,
conversation, factual-QA, and writing. Attribution structure also passed
without retaining raw text or `user_id`. This is a plan/preflight only: no full
dataset was downloaded, no source was promoted, and acquisition still requires
separate bounded authorization.

## Why 500M is not a selected target

SFT scale must be measured after filtering and chat serialization. The project
records both total serialized tokens and assistant/tool supervised tokens; row
count alone is not a training-size measurement. `500M` is therefore an optional
upper planning band, not the selected Lumi L1 SFT target.

The staged decision is:

1. Build and seal a rights-clear `10M-30M` supervised-token learning candidate.
2. Continue to `50M-150M` only if frozen quality, safety, Bengali/Banglish, and
   base-retention evaluations improve.
3. Consider `200M-500M` only if marginal gains remain positive and a separate
   owner authorization is issued.
4. Stop at the smallest dataset that passes the release gates.

These are engineering planning bands, not a claim that more tokens always
improve the model.

## Why 1T is the wrong default SFT objective

One trillion tokens is 12.5 times Lumi L1's complete 80B-token base pretraining
run. At the previous optimized training rate of about 140K total tokens/s, one
raw pass would take about 82.7 days and about $63.3K at $31.92/hour; at 100K
tokens/s it would take about 115.7 days and about $88.7K. Variable-length SFT,
loss masking, validation, checkpointing, and additional epochs can make the real
run slower or more expensive.

A 1T-token objective would need a separate continued-pretraining-scale plan,
not a silent expansion of SFT. The dominant problem would be acquiring,
licensing, deduplicating, decontaminating, reviewing, and balancing 1T useful
tokens without synthetic repetition or style collapse.

## Where the data can come from

Use four explicitly separated lanes:

- Lumi-owned behavior data: identity, natural conversation, Bengali/Banglish,
  tool honesty, safety, corrections, and product-specific behavior.
- Revision-pinned public human data: only after row-level license, provenance,
  quality, attribution, PII/secret, and benchmark-contamination checks.
- Verifiable records: executable code/tests, deterministic arithmetic, grounded
  factual answers, and tool calls whose outputs can be recomputed.
- Synthetic teacher output: only when the exact provider/model contract permits
  training Lumi, with teacher identity and generation parameters retained per
  row. Synthetic output must never be relabeled as human.

The existing source repair order is OASST1 context/rank safety, Aya Bengali
concentration and formatting, OpenR1 verified-trace provenance, then the missing
safety/coding/factual/writing mix.

## OpenAI and Azure OpenAI boundary

API privacy and output-training rights are different questions. OpenAI's API
data-control documentation says API data is not used to train OpenAI models by
default, but the standard OpenAI Services Agreement restricts using output to
develop models that compete with OpenAI except for narrow permitted exceptions.

Current Microsoft Product Terms permit synthetic output for only enumerated
fine-tuning destinations and connected use cases. A standalone general Lumi
assistant is not assumed to fit those exceptions. The 2025 Microsoft Q&A answer
suggesting broader permission is not treated as stronger than the current
Product Terms.

Therefore both direct OpenAI API and Azure OpenAI generation are fail-closed for
Lumi training until the company receives written, contract-specific clearance.
No API key, quota, or credit changes this gate.

## Exact next execution order

1. **Completed:** repair the OASST1 adapter locally to reject unranked replies
   and context-free follow-ups.
2. **Completed:** run the targeted local tests and static checks.
3. **Completed without a new fetch:** bind and reuse the existing exact pinned
   100-row OASST1 slice.
4. **Completed:** re-audit the repaired adapter with zero automatic promotions.
   Strict root-direct/rank-0 selection admitted 9 candidates; missing ancestor
   history and unranked-reply admissions are both zero in the accepted set.
5. **Completed fail-closed for Aya:** the official statistic and bounded fetch
   both report 1,534 Bengali rows. The full-count machine audit measured a
   51.43% top-annotator share, 223 trailing-pipe artifacts, 4 exact duplicate
   prompt/target pairs, and unstable dataset-viewer wrapper indices. No data was
   promoted.
6. **Completed bounded and fail-closed for OpenR1:** reuse the exact pinned
   24-row canary without network access and select only verifier-passed complete
   traces from `generations`, never the reference `solution` field. The repaired
   selector admitted 18 traces; normal filtering admitted 12, and strict 4K
   encoding admitted 6. One row had an external dependency, 5 had no safe
   verified complete generation, 6 hit the generic PII-pattern gate, and 6
   overflowed 4K. Production promotions remain zero pending full-set fit,
   upstream problem rights/attribution, record-level quality review, and an
   attribution manifest.
7. **Completed primary fail-closed diagnostic:** diagnose the six OpenR1
   PII-pattern rejections without retaining raw text or weakening the global
   filter. This conservative first pass classified four line-local math-number
   collisions and left two records for the later secondary diagnostic. No
   record was promoted.
8. **Completed as planning only:** define eight source lanes and exact task and
   language bands for a 10M-30M accepted train supervised-token learning
   candidate. The proposal authorizes no acquisition, API generation, GPU work,
   dataset promotion, or training.
9. **Completed as a machine-validated candidate only:** generate the bounded
   Lumi-owned safety/tool lane with explicit AI provenance, exact semantic-family
   split isolation, independently recomputed tool results, and no external API.
   All 320 records passed; the frozen tokenizer measured 18,636 train supervised
   tokens, 6,388 validation supervised tokens, and a 256-token maximum length.
10. **Completed:** the human owner reviewed all 40 rows, accepted 40 and
    rejected 0; the fail-closed raw-text-free receipt passed.
11. **Completed as candidate-only:** add exactly this lane to immutable
    successor registry `v0.2`. The append-only delta is one source, the old
    registry is unchanged, candidate/canary coverage now spans all eight task
    categories, and production coverage remains empty.
12. **Completed locally, no fetch:** seal OASST1's exact 88,838-row/64 MiB
    acquisition envelope, English root-direct/rank-0 quality policy, 35% source
    share cap, and raw-text-free attribution contract. The pinned 100-row
    preflight selected 4/10 structural pairs and authorized no acquisition or
    promotion.
13. **Completed, non-production:** the separately authorized four-object OASST1
    CPU run processed exactly 88,838 rows and retained 1,745 records / 477,456
    supervised tokens after schema, PII/secret, deduplication, benchmark, and 4K
    gates. Independent verification passed.
14. **Completed human quality sample:** the owner reviewed the exact balanced
    32-row OASST1 sample, accepted 32, rejected 0, and the fail-closed
    raw-text-free receipt passed. This is upstream human-authored plus human
    owner-quality-reviewed data, not first-party authored data. Source-registry
    promotion, dataset sealing, and SFT remain separate and unauthorized.
15. **Completed candidate-registry promotion:** immutable successor `v0.3`
    changes only the existing OASST1 metadata entry and binds the reviewed
    1,745 records / 477,456 supervised tokens. The public source remains
    `canary_eligible` and non-production; the active data contract is unchanged.
16. **Completed locally:** the exact existing 1,534-row Aya Bengali inventory
    passed a deterministic repair preflight, projecting 1,478 records / 577,453
    supervised tokens after repairing all 223 terminal-pipe artifacts,
    normalized deduplication, Bengali-script and 4K gates, and a final 50%
    annotator cap.
17. **Completed bounded materialization:** the exact projected Aya candidate,
    raw-text-free attribution manifest, and deterministic 32-row owner packet
    were written privately outside Git. The receipt reproduces all record/token
    ledgers. A pre-review inline-JavaScript escaping defect in the packet was
    corrected and syntax-checked without changing candidate or attribution
    data. Promotion, dataset seal, SFT, network, API, and GPU gates remain
    closed.
18. **Completed owner quality sample:** the exact 32-row Aya decision file
    matched every candidate, attribution, preflight, and sample binding. The
    owner accepted 32 and rejected 0; the raw-text-free receipt passed with zero
    production promotions.
19. **Completed candidate-registry promotion:** immutable successor `v0.4`
    changes only the existing Aya metadata entry and binds the exact reviewed
    candidate, attribution, and 32/32 owner gate. Aya remains `canary_eligible`,
    all production blockers remain, and the active data contract is unchanged.
20. **Completed as a metadata contract, not model evidence:** freeze a
    72-lane English/Bengali/Banglish punctuation gate covering six correction
    classes and prose, code/command, URL/path, and decimal/version/math contexts.
    Actual post-SFT release evidence requires at least 288 private model cases.
21. **Completed as a non-materializing preflight:** the exact connected-component
    projection yields 1,330 train records / 520,269 supervised tokens and 148
    validation records / 57,184 supervised tokens. Prompt/response exact and
    MinHash-0.90 near-duplicate cross-split overlap are all zero; both splits
    preserve the 50% annotator cap. No candidate row or split manifest was
    written.
22. **Completed bounded materialization:** write Aya's exact 1,330-train / 148-
    validation candidate, raw-text-free split manifest, attribution binding,
    and verification receipt outside Git. The exact totals are 520,269 / 57,184
    supervised tokens, private writes are 5,480,313 bytes, and production
    promotions remain zero.
23. **Completed local secondary OpenR1 PII diagnostic:** all six phone-regex
    records in the bounded 18-trace slice reconcile as five line-scoped math
    numeric false positives and one cross-message join artifact. No unresolved
    record remains in this slice; the global PII filter was not modified.
24. **Completed locally, no fetch:** seal the exact OpenR1 full-curation
    preflight for 93,733 rows, 10 parquet shards plus README, 2,149,897,914
    expected data download bytes, 4,964,543,659 decoded bytes, and a
    2,415,919,104-byte network ceiling. The plan permits no acquisition,
    candidate materialization, promotion, GPU, or SFT.
25. **Current:** separately authorize the exact full OpenR1 acquisition/curation
    envelope, complete full-set 4K fit, source rights and attribution, secondary
    PII reconciliation, and quality review, then continue the remaining
    rights-clear lanes and stop at the smallest complete 10M-30M mix.

The machine-readable boundary is
`configs/lumi-4b-sft-data-scale-gate-v0.1.json`.
The raw-text-free OASST1 repair receipt is
`evidence/lumi-4b-sft-oasst1-repair-reaudit-v0.1.json`, SHA-256 seal
`22472021420ee8fba53b01e24126e6169956d3987e901d8c7acb700e0555f4b3`.
The OASST1 full-curation plan is
`configs/lumi-4b-sft-oasst1-full-curation-plan-v0.1.json`, file SHA-256
`573738381f4c2c03163cb1e92962e56488aa52367f6d7c7f5c8b5a9381cf3b07`.
The OASST1 private-review ready receipt is
`evidence/lumi-4b-sft-oasst1-owner-review-ready-v0.1.json`, internal seal
`ef7ad859965bc88234676a6544abd10e7303d5d4a2d3a1c1a777780c38a14a42`.
The completed raw-text-free owner-review receipt is
`evidence/lumi-4b-sft-oasst1-owner-review-v0.1.json`, file SHA-256
`3dc8195582fb495e278f46fb9008abd7f91d7bcb80283b7c84eb08cd7e544bf0`
and internal seal
`6f3f0232983695403f110d898ed59ab5d1bff503f587aa5360db6ec9e46439a4`.
The raw-text-free candidate-registry promotion receipt is
`evidence/lumi-4b-sft-oasst1-source-promotion-v0.1.json`, file SHA-256
`7d2cf27305f9600306ed6ff96029adab94db2e08f4ead90c06107bbd8f31cece`
and internal seal
`98341fbec17668d16c4a783a534238a36724c0756c2a718322d3e508f3c2c4a0`.
Its immutable successor registry is
`configs/lumi-4b-sft-source-registry-v0.3.json`, SHA-256
`e85f591f2a93b1384ed9ce4ee3da0e067bcfc07a1ade2e0b170eb875c7677914`.
Its raw-text-free local preflight is
`evidence/lumi-4b-sft-oasst1-full-curation-preflight-v0.1.json`, internal seal
`686d39d0024226bb3cdc60de870c74326b5d6556c4e6da7df10703fa2b9296d9`.
The completed raw-text-free curation report is
`evidence/lumi-4b-sft-oasst1-full-curation-v0.1.json`, internal SHA-256
`0df672f2b34a6e580fc9a577749fb03359c9628ca92a4cfe0a6ba801e25ff580`.
Its independent verification receipt is
`evidence/lumi-4b-sft-oasst1-full-curation-verification-v0.1.json`, internal
SHA-256 `bdd099dae98812117db028f98a4d4aa11b23ab1e08b59ae67667b1a0484b2cc3`.
The authoritative raw-text-free Aya reconciliation receipt is
`evidence/lumi-4b-sft-aya-bengali-source-audit-v0.2.json`, SHA-256 seal
`7b928859f668d6263b7a2331b4d3c7976f2c93992289ccad0ab364a69f99002e`.
The non-authorizing Aya repair plan is
`configs/lumi-4b-sft-aya-bengali-repair-preflight-v0.1.json`, file SHA-256
`01d835d05e38832fa14b97c9817edc1defbb1e0c95549eb777b3f0aea01e5054`.
Its raw-text-free preflight receipt is
`evidence/lumi-4b-sft-aya-bengali-repair-preflight-v0.1.json`, file SHA-256
`9fa37c0462e20be5849cf68dbed668dd956984cb252b028226b68d0d15cdfddb`
and internal seal
`6d7c20b19f6b8154c58f51c59521d1e0045dbcbe3e06baa10ac1766c0f6800b2`.
The bounded materialization plan is
`configs/lumi-4b-sft-aya-bengali-materialization-v0.1.json`, file SHA-256
`48b035ca77f478afa5a9de04f5d2fdb715c0887fb7ad2dc3c347455b9fe6184b`.
Its raw-text-free receipt is
`evidence/lumi-4b-sft-aya-bengali-materialization-v0.1.json`, file SHA-256
`9658cb37fe659a65d7dd68d8f7d5529b4240ff96c6c747cf61f40db29274c05c`
and internal seal
`a2f08b1594a2b6fd9def1f43a48e072cbe0f107f4bc7ea43b391d6066ddc6ade`.
The Aya owner-review plan is
`configs/lumi-4b-sft-aya-bengali-owner-review-v0.1.json`, file SHA-256
`7a2fa64be0c71943e2995b11b76f256d8118c31b55e9fa7fa4eeee6c0e2ee86a`.
Its completed raw-text-free receipt is
`evidence/lumi-4b-sft-aya-bengali-owner-review-v0.1.json`, file SHA-256
`66c28d06816e6fc69cbde022fa0ebb3820f44ad0e1daf611f1857c6ca9e243be`
and internal seal
`e23851af5718d10acdd120ca1e0c200672fb5c794ad503ed4d1f7f1650cc685f`.
The reviewed Aya candidate-registry promotion receipt is
`evidence/lumi-4b-sft-aya-source-promotion-v0.1.json`, file SHA-256
`243958ed536244df586a6db9423172920888e4089ace8bb165271ed687ef6ac2`
and internal seal
`66c03b268ab5544e7273d81e034be4792be78aa0956f2d73b1496f4950071446`.
Its immutable successor registry is
`configs/lumi-4b-sft-source-registry-v0.4.json`, SHA-256
`4c81d9968ff8ecdef760ab7e78f01d4d4ddb8e75aae7d29e17dc8a349c69161f`.
The metadata-only punctuation contract is
`configs/lumi-4b-punctuation-quality-gate-v0.1.json`, SHA-256
`16b40294c86823cf2069e331cceba76bb7485b3ca7c3b64709814a2656005757`.
The raw-text-free Aya validation preflight is
`evidence/lumi-4b-sft-aya-bengali-validation-preflight-v0.1.json`, file SHA-256
`3f021e617592cee83d83addcf6bf30cbf0eac31690fa3af3a3ea8044f1cbf2ef`
and internal seal
`48b21e9f5b1ce07290beab67c3f77a33572a0ab887b871e19fcebc5b99fc8628`.
Its exact private split verification receipt is
`../lumi-artifacts/4b-sft-aya-bengali-validation-materialization-v0.1/48b21e9f5b1c/attempt-001/verification-receipt.json`,
file SHA-256
`a7dd4926d110587cbe55aec79be80b99cd4f1af9300df7a93407186ecc421374`
and internal seal
`a4507bd0dbe7b1d335f4602205f4d62b1368efe9c0f8ce8f6f31606e667f8932`.
The raw-text-free OpenR1 verified-trace receipt is
`evidence/lumi-4b-sft-openr1-verified-trace-audit-v0.1.json`, internal report
seal `4d5cdab5179154138650059da9044d19826f6510a6fb749a58bd195637431a8d`.
The raw-text-free OpenR1 PII diagnostic is
`evidence/lumi-4b-sft-openr1-pii-pattern-diagnostic-v0.1.json`, internal report
seal `42d3d88d1d5cf9436d8911571515e310fd7d4a2876a6f460698e3c185da13b54`.
The bounded raw-text-free secondary PII receipt is
`evidence/lumi-4b-sft-openr1-pii-secondary-v0.1.json`, file SHA-256
`12e08322c0d52a291860f580f226f7b6e616e0596b46a88c7fefbc774b43864b`
and internal seal
`7761ab5800608658c0ea99d6b2a1d03e19c2c46c85b1641e647789ef1069fb6a`.
The exact local full-curation plan is
`configs/lumi-4b-sft-openr1-full-curation-plan-v0.1.json`, file SHA-256
`0ae5f0d5f389a0ce4f8cfb87103b2dfa9d3f0688fbac04f9486bac0eeb68fad4`.
Its raw-text-free preflight is
`evidence/lumi-4b-sft-openr1-full-curation-preflight-v0.1.json`, file SHA-256
`0cada20dbb2e66f9ec883a23fb922eb7afcb8da3a16c76aa88b158c66363f6e3`
and internal seal
`520cadd35bd2ece1d971986f790b9cfaf0c9ceafc9c330f404e376fab13fbc78`.
The raw-text-free Lumi safety/tool candidate receipt is
`evidence/lumi-4b-safety-tool-sft-pilot-v0.1.json`, internal evidence seal
`62605cc07466ffc9c583655e3a85a3ad64c2bd4ef95a0c95deeb68d72823b57b`.
Its frozen-tokenizer measurement is
`evidence/lumi-4b-safety-tool-sft-token-audit-v0.1.json`, internal report seal
`95496eaced0c6cc1e8ea53a57b7adbf6a8e40df924bfb46731651a1ba63ce658`.
The private raw-text-free owner-review receipt is
`artifacts/lumi-safety-tool-sft-pilot-v0.1/f7dacd1ceb30/attempt-004/owner-review-receipt-safety-tool-v0.1.json`,
internal receipt seal
`369a3110a46de6189a7311eee946daa7092cf04cb9676dfb8ea67cfb41e838fb`.
The raw-text-free candidate source-promotion receipt is
`evidence/lumi-4b-safety-tool-source-promotion-v0.1.json`, internal receipt seal
`53c11f1bba6583ff4a638b37b9418bc73b8f3abe36dbf1522bba2e754b496c2b`.
It binds successor registry SHA-256
`f686668fbb05f0d7930b5047560543ce113fcfb97c423fbe840f65d8a1058cdd`.
The machine-readable missing-mix plan is
`configs/lumi-4b-sft-missing-mix-proposal-v0.1.json`.
