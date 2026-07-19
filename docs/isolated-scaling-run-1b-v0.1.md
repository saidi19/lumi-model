# Lumi-1B Isolated 1.4B-Token Run v0.1

Status: **the authorized 1.4B-token H100 run passed, its archive is independently
verified, the Lambda instance is terminated, and no paid GPU is running**

## Verified H100 outcome

The canonical isolated run began from random initialization, exported an exact
step-100 resumable state, and then continued on one Lambda H100 from cumulative
step 101 through the hard stop at step 42,725. It processed exactly
`1,400,012,800` tokens with the `1,050,981,888`-parameter model. The owner
authorization remained capped at step 42,725 and recorded
`full_4b_training=false`; no 4B command was launched.

| Frozen held-out evaluation | Loss | Perplexity |
| --- | ---: | ---: |
| Isolated, step 100 | 7.794926 | 2,428.251 |
| Isolated, step 7,630 | 3.266177 | 26.211 |
| Isolated, step 42,725 | 2.596964 | 13.423 |
| Legacy separator-only semantics, step 42,725 | 2.643961 | 14.069 |

The decreasing isolated held-out curve establishes learning and stable long-run
execution. It does not make this deliberately undertrained 1B checkpoint a
production chat model. The final isolated evaluation is also better than the
explicitly labelled legacy evaluation on this fixed artifact, but that is an
evaluation-policy comparison, not a frontier-model benchmark.

Steps 101--42,725 sustained a measured median of `19,900.242` tokens/second.
Their summed step time was `70,593.663` seconds (`19.609` hours), giving a
recorded step-compute estimate of USD `84.1241` at the authorized H100 rate of
USD `4.29`/hour. This excludes setup, evaluation, archive-transfer, and any
provider billing granularity, so it is not an invoice total.

The downloaded archive binds source commit
`5c5c20162369d56cd6e467fdb7e9b633a64cca2e`. All 77 current server artifacts
pass the independently downloaded post-seal manifest whose SHA-256 is
`36379874d55d71d36c06f53650254f474d7500cc96d14071d669bd4eaba5c615`.
The original in-workspace manifest passed 76 of 77 entries; its only mismatch
was `run-command.log`. Its expected digest matches the exact log prefix present
when sealing occurred, while the suffix contains only the seal command's final
JSON and PASS line appended by the outer `tee`. Model, optimizer, checkpoint,
evaluation, and authorization artifacts all matched. The sealer now excludes
that necessarily live outer command log from future in-process manifests.

## Outcome of the packed-attention experiment

The Claude-raised packing question produced a useful engineering change, but
the decision came from Lumi's own controlled test rather than from accepting an
external claim. The old separator-only policy lets later documents in a packed
row read earlier documents. The new policy resets RoPE positions at every
document and supplies `cu_seqlens` to FlashAttention so each document is an
independent causal sequence without allocating a dense quadratic mask.

The real A100 comparison held initialization, data order, optimizer, schedule,
hardware, and FlashAttention fixed. The document-isolated policy:

- changed later-document logits by exactly `0.0` when an earlier document was
  mutated;
- matched the dense correctness reference at `0.99998194` cosine similarity;
- reached final step-100 loss `7.576857` instead of `7.690148`;
- sustained `9,809.5` instead of `9,239.5` tokens/second, a `6.17%` gain;
- used the same peak reserved HBM.

This removes both identified risks: accidental cross-document context and the
assumption that correctness must cost throughput. It did not prove better final
reasoning from only 100 steps. The existing separator-only 250M checkpoint stays
valid and immutable, but it was not resumed into the new policy; the canonical
isolated run started from random initialization at step zero.

## Executed run contract

`configs/isolated-scaling-run-1b-v0.1.json` cross-pins the existing cloud,
scaling, evaluation, model, training-data, and held-out contracts. It fixes:

- 1,050,981,888 parameters and 4,096-token training sequences;
- 32,768 processed tokens per optimizer step;
- BF16 CUDA autocast with FP32 parameters and gradient checkpointing;
- peak learning rate `2e-4`, 500-step warm-up, and the existing 42,725-step
  cosine horizon;
- FlashAttention2 plus `position_reset_varlen_v0.1` document isolation;
- two retained resumable trainer checkpoints;
- a fresh step-1 process followed by a separate step-2 resume;
- isolated held-out evaluations at steps 100, 7,630, and 42,725;
- a final eager separator-only held-out evaluation as the legacy comparison;
- a cached `--resume` verification after every evaluation;
- an absolute stop at the approved 1B milestone and no 4B command.

| Cumulative milestone | Step | Processed tokens | Held-out evaluation |
| --- | ---: | ---: | --- |
| Fresh smoke | 1 | 32,768 | no |
| Fresh-process resume | 2 | 65,536 | no |
| Calibration | 100 | 3,276,800 | isolated |
| Learning gate | 7,630 | 250,019,840 | isolated |
| Unique-corpus gate | 42,725 | 1,400,012,800 | isolated + legacy final |

The first 250M tokens are not another separate experiment. They are a
milestone inside the fresh 1.4B-token run, because switching the old checkpoint
from separator-only to isolated attention would invalidate resume semantics.

## Evaluation correction

The checkpoint evaluator now accepts the same attention implementation and
document-isolation policy as training. On CUDA, isolated evaluation fails
closed unless FlashAttention2 and BF16 are available. Its durable fingerprint
includes the attention policy, so an interrupted isolated evaluation cannot be
silently resumed as a cross-document evaluation. CPU/MPS tests use the dense
reference mask, and existing default evaluation fingerprints remain compatible.

This matters because an isolated model evaluated with cross-document leakage
could receive an artificially different held-out score. Lumi now records both
the primary isolated score and one explicitly labelled legacy score.

## Time and cost projection

The original preparation report used the measured isolated A100 median of
`9,809.508788719386` tokens/second. At the previously observed Lambda rate of
USD `1.99` per hour:

| Boundary | Pure training time | Pure training cost |
| --- | ---: | ---: |
| Step 100 | 0.09 h | USD 0.18 |
| Step 7,630 | 7.08 h | USD 14.09 |
| Step 42,725 | 39.64 h | USD 78.89 |

The planning contract adds a conservative two-hour allowance for setup, exact
hash checks, four held-out passes, checkpoint snapshots, evidence sealing, and
teardown. That makes the current lifecycle estimate about `41.64` hours and
USD `82.87`. It is an estimate, not a spending cap or an invoice; current
provider rate, retries, storage, and actual evaluation speed can change it.

Reproduce the planning-only report on the Mac:

```bash
python scripts/plan_isolated_scaling_run.py
```

The output must say `planning_only` and `paid_gpu_launch=false`.

The completed preparation gate passed the full repository test suite, Ruff,
both shell syntax checks, canonical contract/hash validation, projection arithmetic,
authorization-tamper tests, atomic model-only snapshot tests, and isolated
evaluation resume/fingerprint tests.

The pinned FlashAttention setup detects the active GPU compute capability from
PyTorch and builds only for that architecture. It currently permits the
validated A100 (`8.0`) and H100 (`9.0`) paths and fails closed on other GPU
architectures instead of silently producing an incompatible extension.

## Paid-run authorization boundary

The repository plan deliberately contains:

```json
{
  "full_1b_stage": false,
  "full_4b_training": false,
  "paid_gpu_launch": false
}
```

The server setup and runner both refuse to proceed without a separate
owner-approved authorization JSON outside Git. That artifact must bind the
exact merged source commit, transferred source-archive SHA-256, run-plan
SHA-256, provider, instance type, current hourly rate, and one declared maximum
step. It can authorize only steps 1, 2, 100, 7,630, or 42,725 and must keep 4B
training false. The repository plan itself never grants launch authority.

The completed run followed this safe order:

1. merge this preparation branch and bind the resulting source commit;
2. create the out-of-repository authorization for the chosen milestone;
3. create a `git archive` with the `lumi-model/` prefix, bind its SHA-256 in the
   authorization, then launch one validated A100/H100 server and transfer that source
   archive, the 1.4B-token artifact, held-out artifact, tokenizer, and
   authorization;
4. run `scripts/setup_isolated_scaling_run.sh` inside `tmux`;
5. when the authorization binds an isolated step-100 resume, run
   `scripts/import_isolated_scaling_resume.py` to verify and import it;
6. run `scripts/run_isolated_scaling_1b.sh` inside `tmux`;
7. independently download and verify the sealed manifest;
8. terminate the server before considering any later stage.

Rerunning the runner after a disconnect resumes only the matching isolated
fingerprint. It preserves model-only snapshots at the three evaluation
milestones before trainer checkpoint pruning. For a fresh step-zero run, the
final seal checks all 42,725 metric rows, the exact data cursor, CUDA runtime
gates, varlen parity, every evaluation policy, model parameter count, and all
server artifact hashes.

The archived isolated step-100 checkpoint contains the exact model, optimizer,
RNG, cursor, checkpoint state, model snapshot manifest, held-out evaluation,
and a 34-file SHA-256 manifest. Its export did not include the first 100 rows of
`metrics.jsonl`. The resume importer therefore never substitutes rows from a
different run or fabricates missing values: it binds the export and checkpoint
hashes in the owner authorization, records that the root state is derived from
the checkpoint state, and starts new measured metrics at step 101. For this
explicit resume path, the final seal validates 42,625 contiguous metric rows
from step 101 through 42,725 plus the hash-bound step-100 checkpoint, snapshot,
and evaluation evidence.

## What this run can establish

The successful 1.4B-token gate establishes that the 1B architecture, optimizer,
isolated attention, resume path, held-out curve, and long single-GPU runtime are
stable. At only 1.33 training tokens per parameter, it is still intentionally
undertrained. It should not be expected to become ChatGPT-quality, and it does
not authorize the 4B/80B run.

After this gate, the remaining path is a provenance-tracked high-quality 80B
token corpus, distributed 4B training design and short multi-GPU calibration,
an explicitly approved 4B run, then instruction tuning/alignment, safety and
quality evaluation, quantization, API/UI integration, and Mac deployment.
