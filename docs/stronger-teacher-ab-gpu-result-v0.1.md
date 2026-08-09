# Lumi stronger-teacher paired A/B result v0.1

Date: 2026-08-09

## Decision

Do not promote Qwen3-14B-Q4_K_M as Lumi's stronger synthetic-data teacher. The
candidate completed the frozen 100-prompt paired schedule, but its machine
acceptance rate was 38%, compared with 41% for the Qwen3-8B-Q4_K_M baseline.
The measured candidate advantage was therefore -3 percentage points, versus
the required +15 percentage points.

This is a fail-closed quality decision, not a GPU failure. Moving the same
inference workload from the A10 to an H100 would reduce latency but would not
change the generated outputs or make the candidate pass the frozen quality
gate.

## Frozen experiment

- Models: Qwen3-8B-Q4_K_M baseline and Qwen3-14B-Q4_K_M candidate.
- Schedule: 100 exact paired prompts across five tasks, four language modes,
  and five prompts per task-language cell.
- Candidate execution: sequential, self-hosted Ollama 0.32.5 inference over a
  pinned localhost-only SSH tunnel to one NVIDIA A10.
- External inference APIs: none.
- Training, dataset sealing, registry promotion, and production promotion:
  none.
- Owner-review packet: 40 blind pairs were materialized, but the candidate is
  already ineligible because it failed the machine gate.

## Verified results

| Metric | 8B baseline | 14B candidate |
|---|---:|---:|
| Final outputs | 100 | 100 |
| Accepted outputs | 41 | 38 |
| Acceptance rate | 41% | 38% |
| Critical factual/safety/tool failures | 2 | 3 |
| Median call latency | 7.222 s | 6.463 s |
| Retained supervised tokens | 2,303 | 3,187 |

Latency is recorded for operational evidence only. The two lanes were not
completed on identical hardware, so latency is not used as a model-quality
comparison.

The full run ended at 261 cumulative inference attempts, 61 cumulative runtime
retries, and 45,376 generated output tokens. All were inside the corrected
ceilings of 282 attempts, 81 retries, and 56,320 tokens.

## Truncation correction

The first A10 continuation stopped fail-closed when one Bengali record reached
the fixed 256-token response ceiling on every retry. The sealed failure
boundary preserved the baseline journal and all completed candidate outputs.
The corrected continuation reopened only that incomplete record and changed
only candidate `num_predict` from 256 to 512. It then completed all 79 remaining
valid outputs in 79 calls, without another runtime retry.

## Integrity

- Baseline journal SHA-256:
  `09ab1f3a14237ba4923a482f9a18908f09119739da7921110246ddb56f9dc587`
- Final candidate journal SHA-256:
  `9b07b77b2601fe873c875ed7669b4e23849ca9cfd314dcc52139d8cbdbc1c8a5`
- Completion report file SHA-256:
  `574aa66e03a52a16c6c33135ea6eea6b3c0b0906199458fa6cb79894e52eb8dd`
- Completion report internal SHA-256:
  `92a517758bd355645f901e79055aff580466d9f4ce40dc9cf16d7291f83aafd9`
- Independent verification internal SHA-256:
  `a2ab520abe31c57079fb511432a960fa60818ee62620876414bd51414500dbb9`

Independent replay reproduced the journal boundary, comparative audit, plan,
authorization, runtime receipt, tooling hashes, and fail decision. GPU memory
was released after inference, and the A10 reported zero uncorrectable ECC
errors.
