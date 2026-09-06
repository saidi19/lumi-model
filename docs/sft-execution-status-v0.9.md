# SFT execution status v0.9

Checked on 2026-09-06. The reviewed source backup is now on the private GitHub
branch, but **full CI is still failing and real GPU SFT has not started**.
This status corrects the pre-push snapshot in v0.8 without rewriting that
historical document or any previous plan, receipt, source binding or marker.

## Source backup and current CI

The reviewed source backup is commit
`1c24ef039d25dde0172fb2751d7afdba403c7975` on
`codex/lumi-posttraining-product-v0.1`. The subsequent CI separation commit is
`dcca2f9008deeb412d4373b823c4c58c59af712e`. Main was not merged or rewritten.
The source backup added the reviewed source/runtime work; it did not upload
private datasets, model weights, credentials or populated owner packets.
See [the backup scope](github-source-backup-v0.1.md).

GitHub [run 33971169909](https://github.com/saidi19/lumi-model/actions/runs/33971169909)
at `dcca2f9` completed with these independent results:

| Check | Observed result | Meaning |
| --- | --- | --- |
| Frozen lint integrity | Pass | Exact historical debt retained; zero new findings, not all-source lint cleanliness |
| Isolated SFT controls | Pass | 273 control tests and 43 CI guard tests passed |
| General test suite | Fail | 1,827 passed, 308 failed, 21 errors, 58 skipped |

The original source-backup run stopped at lint before running tests. The new
separation exposes the general-suite failures rather than declaring them fixed.
The historical lint manifest covers 815 findings in 163 byte-pinned files;
[the preservation policy](ci-preservation-policy-v0.1.md) explains this debt.

Read-only review of the 329 Linux failure/error outcomes found 322 outcomes
dependent on absent inputs, five actual existing-file SHA mismatches and two
old BF16 errors. This is a diagnostic classification, not a 322-test skip list.
The absent-input group includes 140 FileNotFoundError outcomes, 142 explicit
missing-input ValueErrors, 37 checks whose wording conflates absence with a
changed binding, one missing-prefix count failure and two expected-error regex
assertions masked by earlier absent inputs. No private payload was needed for
this classification; the ambiguous paths were checked against tracked plan
metadata and the complete source clone.

The Linux failures require distinct treatment:

- Some tests read intentionally untracked private plans, receipts or artifacts.
  A source clone cannot replay those integrations. Missing tracked inputs,
  malformed fixtures and changed source bindings must remain failures.
- Some historical plans bind an older source hash. A hash mismatch is not proof
  that a private fixture is absent and must not be converted into a generic skip.
  The five genuine mismatches comprise one Aya and four safety-URI cases binding
  the existing `src/lumi/posttraining_data.py`. They remain unresolved.
- Two old v1 checkpoint-import BF16 cases fail with `HF reload tensor mismatch`.
  The v1 implementation casts configured buffers along with parameters. The
  existing v2 successor preserves the buffer policy; its 17 tiny CPU tests pass
  in this Linux run. The old files remain historical evidence, not the current
  successful importer. An explicit regression must preserve that distinction.

Local macOS tests use a different historical ML runtime from GitHub's Linux
environment. Local missing-package or API errors are not automatically Linux
regressions. No private fixtures are being uploaded to make CI green, and no
blanket exception-to-skip or failure-to-pass policy is approved by this status.

## Roadmap position

The original roadmap still has eight top-level gates. **Four are fully closed**:
pretraining Segments 1, 2 and 3, then final-base evaluation/seal/backup verification.
This is a gate count, not a percentage of elapsed time or remaining work.

Gate 5's private research data/encoding and CPU preparation have progressed;
production eligibility remains open. Gate 6 is actual bounded SFT and checkpoint
comparison; gate 7 is post-SFT quality/safety evaluation; gate 8 is serving and
deployment. None of gates 6-8 is closed by a CI pass.

The six previously documented pre-GPU conditions remain partially open:

1. Authentic restore/import of the exact S3-backed final base.
2. Integrated worker, parent IPC, durable accounting and OS resource controls.
3. Exact new target identity, runtime and enforceable lifecycle limits.
4. Verified private encoded-data relocation and target metadata.
5. Exact CUDA acceptance followed by authorized smoke/resume execution.
6. Complete target-bound execution ticket and approval checks.

The selected first restore/smoke preparation envelope remains USD 150 and four
hours total host lifecycle; it is not a full-SFT budget or a provider billing
guarantee. The existing busy Lambda instance remains out of scope. No new
instance, model restore, real training, inference or data relocation was performed
in this CI recovery. The pretrained base's S3 backup is separate from GitHub.
`production_eligible=false` remains unchanged.

## Resume boundary

The following local changes have been independently rechecked, but are not yet
committed, pushed or integrated into the GitHub workflow:

| Local change | Verified result |
| --- | --- |
| Four exact-node private-fixture routing plugin | 52 guard tests passed; complete source clone ran 7 ordinary siblings and reported exactly 4 unavailable tests, not passes |
| Explicit checkpoint-import history runner | 28 guard tests passed; actual tiny CPU run executed all 31 cases, with 29 passes and exactly 2 required historical errors; all 17 v2 cases passed |
| Synthetic production-source-policy validator tests | 14 tests passed with in-memory labelled fixtures, no private input reads or operational evidence publication |

The combined new stdlib suite passes 94 tests. The history runner deliberately
reports `all_tests_passed=false`; it pins six historical files and requires the
exact two error IDs, type, message and original raise location. It does not make
old v1 BF16 work. Independent review fixed a draft manifest/schema disagreement,
preserved the virtual-environment interpreter path and excluded read-only atime
changes from source-mutation detection before accepting these local checks.

The older non-publishing readiness checks also passed again: 167 source-readiness
tests plus 106 runtime-control tests, with 54 historical bindings preserved and
zero evidence publication. No tracked file was modified in this local batch.

Recheck the new stdlib tests from the repository root:

```sh
PYTHONPATH=src artifacts/lumi-sft-import-runtime-v0.1/venv/bin/python -S -B \
  -m unittest -q tests/test_ci_private_fixture_plugin_v1.py \
  tests/test_checkpoint_import_history_ci_v1.py \
  tests/test_production_source_policy_preflight_synthetic_v1.py
```

Recheck the synthetic CPU history contract without any private model or data:

```sh
artifacts/lumi-sft-import-runtime-v0.1/venv/bin/python -B \
  scripts/run_checkpoint_import_history_ci_v1.py
```

Next, add source-only synthetic invariant coverage for the learning-candidate
seal and teacher-canary promotion tests, then review the remaining exact
missing-input and source-binding cases. Those two additional synthetic modules
were not created in this batch. A subtask service reported exhausted workspace
credits; no retry through another agent was made. The current account usage
window still showed allowance remaining, so the service error must not be
reported as proof that the ordinary account quota is exhausted.

Do not claim full CI success until workflow integration and a complete Linux run
justify it. Local guards are not private integration replay, model acceptance,
or permission to skip GPU prerequisites. Do not rerun consumed curation attempts
or republish earlier evidence.
