# Local vocab-report unit validation v0.1

Checked on 2026-09-06. This is a local unit-test environment check, not a repair
or waiver of hosted CI, a private historical replay, or a new Be-DSTC2 diagnostic
or curation pass. No old file, test, report, plan or consumed marker is changed.

## Result and scope

All four unchanged report/verification unit tests in
`tests/test_bedstc2_vocab_contract_diagnostic_v5.py` passed using the genuine
existing local predecessor plan and marker metadata. Hosted source clones omit
these two private files, so the same tests cannot construct their report fixture
there. Their presence and sizes, not their contents, are used by these tests.
This confirms a local missing-input environment difference, not a new approval.

The new standalone checker pins the original diagnostic and test source hashes,
runs exactly those four functions without replacing validators or assertions,
and verifies source hashes and private-file metadata again afterward. The check
does not execute the diagnostic's `main`, read vocab/train/dev/candidate content,
authenticate private historical content, write an operational report, or access
a GPU. It prints aggregate results only. It makes no skip/xfail/deselect change.

The process uses a 30-second POSIX alarm, a timeout type that cannot satisfy an
expected ValueError, and a monotonic completion check. Its Python audit guard
rejects file contents outside code/runtime roots, writes, network activity,
subprocesses and the enumerated ML-stack imports. This is defense in depth, not
an OS security sandbox. Execute only in a fresh process using the existing
trusted local runtime, with `-B` to avoid bytecode writes.

```sh
artifacts/lumi-sft-import-runtime-v0.1/venv/bin/python -B \
  scripts/check_vocab_report_local_ci_v1.py
```

Observed: `PASS_LOCAL_UNIT_ONLY`, four passed, zero failed, zero blocked-operation
attempts, and source hashes/private metadata preserved. A separate negative
check against the existing source-only clone returned `STOP`, zero tests passed,
and exit code 1 because the private prerequisites were absent. It did not create
replacement files or convert missing inputs to success.

The checker has 26 synthetic guard tests. Combined with the existing 130-test
source-only collection, 156 tests passed. The existing frozen-lint checker
reported zero new findings and all 815 historical findings retained unchanged.
An initial combined invocation omitted `PYTHONPATH=src` and had three import
errors; the correctly configured invocation below passed without code changes
to the affected modules.

```sh
env PYTHONPATH=src artifacts/lumi-sft-import-runtime-v0.1/venv/bin/python -S -B \
  -m unittest -q tests/test_vocab_report_local_ci_v1.py \
  tests/test_ci_private_fixture_plugin_v1.py \
  tests/test_checkpoint_import_history_ci_v1.py \
  tests/test_checkpoint_history_workflow_v1.py \
  tests/test_production_source_policy_preflight_synthetic_v1.py \
  tests/test_learning_candidate_seal_synthetic_v3.py \
  tests/test_teacher_canary_source_promotion_preflight_synthetic_v1.py
```

## What remains open

No hosted workflow or selection changed. The last verified GitHub run,
[34011375472](https://github.com/saidi19/lumi-model/actions/runs/34011375472), remains
at 1,965 passed, 306 failed and 21 errors in its main suite. Four local passes
must not be subtracted from those hosted failures. They also do not close the
five historical source-hash rejections: correcting the source environment for
those tests would expose further private input requirements.

Learning-seal tests need genuine authorization/evidence and candidate metadata;
derived synthetic plans cannot replace that chain. Hosted CI can only replay
those cases after a separately reviewed private-input execution arrangement.
Do not upload private packets, fabricate consumed markers, refresh old source
hashes, waive failures or execute a consumed curation attempt to obtain green CI.
GPU/SFT work is unchanged and `production_eligible=false` remains in force.
