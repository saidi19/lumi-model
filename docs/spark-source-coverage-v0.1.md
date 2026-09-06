# Close the three skipped synthetic Spark checks

## Scope and ordering

The 2026-09-06 request was to finish skipped work before the next Aya receipt
check. Source/log review of commit
`acb780f73d68547579fbace225dc74d793548ce7` identified three source-only tests
that call `pytest.importorskip("pyspark")`. The existing CI environment installs
the declared dev/data extras, neither of which includes PySpark. These are real
coverage gaps, not private-data tests that should be claimed as passed.

The exact tests are:

1. `tests/test_distributed_corpus.py::test_local_spark_dedup_matches_independent_sequential_reference`
2. `tests/test_final_corpus.py::test_local_spark_hash_selection_matches_python_rank`
3. `tests/test_glue_production_dedup.py::test_memory_safe_projection_streams_long_text_only_after_selection`

They use synthetic rows and temporary local Spark storage. Their original
bodies include sequential-reference parity, determinism, anchors and bounded
LSH policy checks, hash-ranked token-prefix selection, and wide-payload
projection/rejoin preservation. No original assertion, test, production source
or threshold is modified. The repeated operations inside the original dedup
test are its determinism assertions, not retries of a failed invocation.

## Additive execution lane

`.github/workflows/spark-source-checks.yml` adds a blocking Spark source check
on push and pull request with read-only repository permissions. It does not
modify either existing workflow, the exact BF16 route, the seven isolated
control suites, frozen lint, the 327-node inventory, or source partition.
The two existing workflow guards continue to validate their original contracts.

The new job uses Python 3.12, Temurin Java 17, the existing dev/data extras and
PySpark 3.5.6. Spark 3.5.6 is the repository's documented previous local corpus
test runtime. Its official documentation supports Java 8/11/17 and Python 3.8+;
the detected local Java 25 installation was not changed or used for a speculative
Spark run. See [Spark 3.5.6](https://spark.apache.org/docs/3.5.6/) and
[PySpark installation](https://spark.apache.org/docs/3.5.6/api/python/getting_started/install.html).

`scripts/run_spark_source_ci_v1.py` requires the fixed runtime and exact three
test nodes. Collection and setup/call/teardown must all complete successfully;
skips, missing/duplicate reports, deselection and nonzero pytest exit cannot
produce a PASS. Normal synthetic-test diagnostics remain available on failure.
The workflow adds a 12-minute process-group TERM deadline, KILL after a further
30 seconds, a 15-minute test-step limit and a 20-minute whole-job limit.
It does not rerun failed tests.

Spark uses local workers and loopback communication. Hugging Face offline flags
and disabled AWS metadata lookup are defensive settings, not an OS sandbox or a
proof of zero network bytes. Package/JDK installation uses public dependency
downloads. No private dataset, candidate, owner packet, AWS credentials, remote
Spark cluster, GPU job, model inference or training is part of this test lane.

## Interpreting results

Before publication, the new runtime/lifecycle/workflow guards plus existing
workflow and source-partition guards passed 160 tests and 120 subtests locally.
Java/PySpark execution is mocked in those guard tests; they are not evidence
that the three Spark tests passed. The changed Python files pass the exact CI
Ruff rules. No local Java installation, old CI workflow or private file changed.

The existing source-only run may continue to display its 58 skips because its
runtime and exact command are intentionally unchanged. The separate Spark run
must provide positive execution evidence for these three nodes; a green source
run alone does not close them. If it fails, retain that failure and diagnose it
before claiming coverage. Hosted execution results belong to the exact commit's
`Spark source checks` run, not to this pre-publication plan.

The remaining 55 reconstructed private/evidence-gated skips are separately
documented in `docs/ci-skip-accounting-v0.1.md`. They are not waived or declared
safe. The 327-node historical inventory is also still OPEN and unchanged; those
are test-node counts, not a count of remaining roadmap stages. Full private
integration and GPU/SFT readiness remain unproven. Aya receipt validation is
deferred while this earlier coverage gap is addressed.
