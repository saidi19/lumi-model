# Safety URI verifier correction (metadata only)

## Defect and correction

The frozen v1 report validator accepts a resealed report whose train token count
has changed, or whose authorization allows network access. Its verification
builder then claims unchanged token counts and no network/API/GPU/SFT actions.
These are genuine false-accepts, reproduced using the actual hash-pinned v1
validator and builder extracted from tracked source, without operational imports.

The v2 successor enforces exact field sets, strict JSON types, authorization,
frozen split counts, context window, internal seal, digest syntax and crosslinks.
It rejects both false-accepts. It returns **metadata consistency only**, never
candidate replay, historical execution verification or new execution permission.

New entry point: `scripts/verify_sft_safety_uri_contract_v2.py`, using
`src/lumi/posttraining_safety_uri_verification_v2.py`. Use this entry point for
future metadata assurance of this historical report. The old operational
`verify_sft_safety_uri_repair.py` is preserved as frozen historical code; it is
not a metadata-only checker and must not be invoked as part of this correction.

## Bounded invocation

Run with Python `-S -B` and exactly four explicit arguments:

```text
--plan <absolute-plan.json>
--report <absolute-report.json>
--expected-plan-sha256 <externally-trusted-file-digest>
--expected-report-sha256 <externally-trusted-file-digest>
```

Pins must come from trusted predecessor metadata, not from the inputs being
audited. The CLI checks actual input bytes against both supplied pins; the pure
validator alone cannot authenticate file bytes. Neither follows embedded paths.

The CLI opens at most two metadata inputs, at most 131,072 bytes each / 262,144
bytes combined, with a 30-second deadline and no input/output file writes.
No-follow descriptor walks reject symlinks, hardlinks, nonregular files and
observed identity/size changes. Invalid arguments, malformed JSON, bad seals,
hash mismatches and contract failures emit fixed error codes without raw input,
paths or exception text. Success is published only after deadline teardown;
a deadline after validation still produces STOP and exit code 1.

This is a bounded, read-only implementation, not a claim of an OS security
sandbox or a new private replay authorization.

## Verification on 2026-09-06

- 34 synthetic tests pass: 19 pure-contract tests plus 15 bounded-CLI tests.
  Tests cover both legacy false-accepts, schema/strict-type/seal/crosslink
  mutations, isolated interpreter requirements, exact pins, no-follow paths,
  size drift, fixed-code failures and the post-validation deadline regression.
- Ruff passes for the four new Python files.
- One real aggregate-only audit passed using external pins from tracked
  `configs/lumi-4b-sft-candidate-mix-reaudit-v0.2.json`:
  plan `47c1cfda1fd884372b8c1aaf482c015e52c18d76368b679bd43d3fac6cdf6fb0`;
  report `2a893c3ec5751be5a552b21e53bfca6e403297b0b3c439eea64015d3da1460fc`.
  Result: `PASS_METADATA_CONTRACT_ONLY`, 2 metadata files / 8,012 bytes,
  0 candidate reads, 0 input writes, no network/GPU/inference. The final deadline
  correction was then checked synthetically; the real audit was not repeated.

All frozen source, tests, plans, historical evidence and private objects remain
unchanged. No CI routing, binding inventory, parser/filter, rights, owner-review,
data registry, promotion or production policy was relaxed. New tests are ordinary
source-suite tests, not new skips, deselections or xfails.

## What remains open

This fixes the safety metadata assurance boundary. It does **not** close any of
the 327 historical/private integration nodes: their exact original inputs and
separately bounded replay evidence are still required. The v2 output keeps
`full_private_integration_passed=false`, `candidate_rows_replayed=false`,
`prior_execution_actions_verified=false`, `execution_authorized=false` and
`production_eligible=false`. Actual SFT/GPU execution has not started.

The next private-readiness action is to identify the exact required artifacts
and replay limits for one original safety test slice, without executing old
one-shot runners or treating metadata assurance as data-level completion.
