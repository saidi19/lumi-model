# Authentic local safety URI row-delta audit

## Completed on 2026-09-06

One bounded, read-only audit compared the genuine predecessor and repaired
safety candidates: **all 320 ordered row pairs passed**. Each repaired row has
the exact unique first-party URI and the independently recomputed transform
digest. Every other parsed field, including messages, content, provenance and
JSON primitive types, is unchanged. No raw rows, IDs, URIs or exception text were
retained in the result or this document.

This is actual private-row verification, extending the earlier metadata-only
assurance. It is not tokenizer/filtering replay, a complete original test run,
qualified rights approval, a dataset seal or permission to start SFT.

## Exact inputs and limits

The standalone entry point is `scripts/check_safety_uri_rows_local_v1.py`, using
the pure `src/lumi/posttraining_safety_uri_row_delta_v1.py` checker. It does not
import the historical loader, runner, publisher, tokenizer, data filter or owner
review code. The original loader would hash 17 bindings before any test body;
this narrower comparison needs only the two candidate objects.

| Input | Bytes | SHA-256 |
| --- | ---: | --- |
| Predecessor safety candidate | 634,808 | `ffc4a91613dd228282d2bc8ef2b93a6d508e31fa90e2925d91de837c1b4bd0e1` |
| Repaired safety candidate | 671,624 | `0f327fb0959a512a698c49992ad516467b3ef8058eb1653416df45157d7f070e` |

Expected digests come from the tracked safety repair v0.1 and candidate-mix
reaudit v0.2 plans, respectively—not from hashing arbitrary supplied candidates.
Synthetic tests authenticate those plan bytes before comparing their bindings
to the fixed CLI inputs.

The one real invocation opened each candidate once and read exactly 1,306,432
bytes combined, with 320 LF-terminated rows per file / 640 total. Limits were
two candidate opens, those exact file sizes and hashes, and 30 seconds. No
source-dataset, tokenizer, benchmark, registry or owner-receipt content was read;
there were no input/output file writes, network, GPU or inference calls.

The no-follow reader rejects symlinks, hardlinks, wrong file types/sizes and
observed path/inode/content-metadata changes. Descriptors stay open through
comparison and final identity checks. Ancestor replacement, read-budget
exhaustion and late deadlines fail closed. Success is released only after
descriptor cleanup and deadline teardown. This is not an OS sandbox claim.

Observed aggregate result:

```json
{
  "status": "PASS_ROW_DELTA_ONLY",
  "row_delta_verified": true,
  "rows_compared": 320,
  "unique_record_ids": 320,
  "unique_source_uris": 320,
  "candidate_files_opened": 2,
  "candidate_bytes_read": 1306432,
  "input_hashes_authenticated": true,
  "input_files_preserved": true,
  "input_files_written": 0,
  "tokenizer_recomputed": false,
  "generic_filtering_replayed": false,
  "full_private_integration_passed": false,
  "execution_authorized": false,
  "production_eligible": false
}
```

## Regression verification and remaining boundary

Twenty-seven new synthetic tests pass: 16 pure row-contract tests and 11 bounded
I/O/CLI tests. Combined with the preceding metadata fix, all 61 targeted tests
pass under pytest; the 50 existing source-routing/workflow guards also pass.
The four new Python files pass the exact CI Ruff rules. Independent review
found no additional actionable issue before the real invocation.

The original URI-only test masks the new transform hash while comparing rows;
the new checker additionally recomputes it and rejects transform drift. Neither
the original test assertions nor any frozen implementation/binding was edited.
Historical evidence, private objects and the 327-node OPEN inventory remain
unchanged. No old one-shot run was retried and no test was newly skipped.

The URI-only data invariant now has authentic local evidence. The original
test node is **not** marked executed: its broad loader and historical-source
bindings were not replayed. Full safety output reproduction still needs the
exact historical data-filter implementation, its private prerequisites and
separately bounded tokenizer/filtering verification. The earlier metadata
check verifies saved counts, not recomputation. Do not subtract this result from
the frozen historical CI inventory or promote it to full integration readiness.

## Next replay prerequisite located without private reads

The exact historical `posttraining_data.py` required by the safety plan is
available in local Git at `e08bddc`. Reading that tracked blob and hashing it
produced `df8297e753e97028999c6bcd203edc2a00a02147f7d9ecb2255569c480363970`,
the plan's expected digest. The current source is a different legitimate version;
it was not replaced or its frozen binding refreshed. No historical overlay was
executed and no further private input was read for this source lookup.

A future full reproduction must use an explicitly scoped immutable historical
source view, authenticate the other dependencies and bound the additional reads.
Locating the blob is a resolved prerequisite, not proof that full replay passed.
