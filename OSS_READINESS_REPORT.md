# OSS readiness review — 2026-10-15

**Status: READY WITH CONDITIONS for maintainer review of the source-readiness PR.**
This is not an unconditional model release or completed application submission.
Gate 5 is on hold; no cloud execution, GPU run or training was performed for this
review. Existing development evidence is preserved in its original checkout.

## Scope and observed state

- Repository: `saidi19/lumi-model`, already public, default branch `main`.
- Audited base: `1f162717f1995644bce9fe75c66cab1408e895c4`.
- Isolated review branch: `chore/oss-maintainer-readiness`.
- GitHub snapshot: 0 stars, 0 forks, 0 open issues; one contributor account
  returned with 153 contributions. No releases returned by the release listing.
  These are point-in-time observations, not evidence of broad adoption.
- Main has 390 tracked files. No root license was found. The package and README
  identify Afnan Technologies Inc. as project branding/author metadata. On
  2026-10-15 the maintainer confirmed personal ownership of the original source
  under the name **Md. Delower Hossain Saidi**; the license proposal now uses
  that holder instead of the company. Third-party rights review remains open.
- The development branch contains later pretraining-completion metadata. It has
  not been merged wholesale or treated as independently verified weight evidence.

## Reproduced defect and correction

The declared Transformers range includes 5.2.0. Its `create_causal_mask` requires
`cache_position`, which the model did not supply. Ten model tests and the small
CPU smoke failed with a missing-argument TypeError. The correction supplies
sequence offsets when the installed mask API declares that argument, preserving
the call form for APIs without it. Architecture parameters are unchanged.

Before adding the focused compatibility regression, the corrected model/config
suite passed 28 tests and the three-step synthetic smoke passed (loss 8.515859
to 8.416506). These tiny synthetic results do not validate a real 4B checkpoint,
SFT readiness, a complete dependency matrix or production quality.

After adding regressions for both mask API call forms (including padded batch
inputs), all 30 model/config tests passed. Ruff passed for the two changed Python
files, and `git diff --check` passed.

The first broader run stopped because the existing runtime lacked declared data
extras (`lingua-language-detector`, `pyarrow`). After installing the declared
data extras into an isolated temporary directory, sandboxed tests reached local
HTTP fixtures but could not bind loopback sockets. With permission for the local
fixture server, the full suite completed: **366 passed, 3 skipped in 14.47s**.
Whole-repository Ruff passed. These earlier environment failures are retained
here rather than reported as successful runs. Remote CI and a completely clean
installation remain separate checks, not established by the local result.

Subsequent remote verification: both GitHub Actions runs for PR #60 at commit
`40cd303cdcd98917ee1f5780a081ac14f1d05cd6` completed successfully on 2026-10-15:
[PR run](https://github.com/saidi19/lumi-model/actions/runs/38698863043) and
[push run](https://github.com/saidi19/lumi-model/actions/runs/38698812056).
This verifies that commit's configured Ubuntu/Python 3.12 workflow, not every
platform or dependency combination. PR #60 remains open with no submitted reviews.

Application responses measure 371, 356 and 364 characters, each under 500.
Local links in the changed documentation resolve. CI now has read-only contents
permissions, non-persisted checkout credentials, concurrency cancellation and a
30-minute timeout. No paid service workflow was added.

## Security review boundary

A limited current-tree pattern scan found no matching AWS access-key IDs,
GitHub token patterns or private-key headers. A filename-only history check
returned no tracked `.pem`, `.key`, `.env`, `.safetensors`, `.pt` or `.pth` paths.
An additional available-ref history diff scan completed over 40,788,565 bytes
using AWS-ID, GitHub-token, private-key-header and OpenAI-token patterns. Seven
file/commit/category matches came from the development source backup: one was a
schema-name substring; the others were credential-shaped strings in mock/test
contexts. No values are reproduced here and no credentials were used to test
validity. This is not an exhaustive entropy scan or assurance that secrets never
existed. Some historical configs/docs contain cloud identifiers and private
artifact paths. No such values are reproduced here; changing those records
without reviewing their hash bindings would be unsafe.

The tracked compressed benchmark-signature artifact also requires a specific
redistribution review and is excluded from the source-license proposal. No
checkpoints or datasets were downloaded for this audit. The GitHub API reported
private vulnerability reporting **disabled**; SECURITY.md discloses this and
does not invent a private email address. No repository setting was changed.

## Maintainer, contribution and application evidence

Md. Delower Hossain Saidi confirmed ownership of the original source. The GitHub
account has administrator/write access. Maintenance includes architecture and
pipeline decisions, reviewing Codex-assisted implementation, tests, experiment
evidence, failure investigation and release decisions. This is a small project,
not an established widely adopted ecosystem dependency.

Useful contributions include synthetic regression tests, dependency compatibility,
documentation corrections, evaluation reproducibility and provenance review.
The PR supplies contribution/security/conduct guides, issue and PR templates,
CPU and artifact-reproduction guides, a model card and an honest application draft.
Existing pipelines and experiment records are preserved.

Public evidence: [4B pilot PR #58](https://github.com/saidi19/lumi-model/pull/58),
[80B plan PR #59](https://github.com/saidi19/lumi-model/pull/59), and
[audited main commit](https://github.com/saidi19/lumi-model/commit/1f162717f1995644bce9fe75c66cab1408e895c4).
Later development metadata is not silently substituted for independently verified
production artifacts. See [application draft](docs/codex-oss-application.md).

## Remaining release work

1. Maintainer review/merge of the Apache-2.0 source proposal. Confirm coverage and
   exclusions; optional/transitive binary redistribution needs additional review.
   Weight/data/tokenizer and benchmark-signature rights stay separate.
2. Establish a private reporting channel and review any historical infrastructure
   metadata before broader distribution. No history rewrite is authorized.
3. Remote CI passed at the commit above. Validate additional clean-install/platform
   combinations before a release; those passes do not cover every supported dependency.
4. Confirm application account/contact information and applicant acceptance of
   the program terms; disclose the PR's actual merge/license status. Submission
   does not guarantee acceptance or benefits.
5. No automatic merge, asset release or Gate 5 resumption as part of this work.

See [MODEL_CARD.md](MODEL_CARD.md), [licensing review](docs/licensing-review.md),
[CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md). This report is a
bounded progress record, not a completed legal or security certification.
