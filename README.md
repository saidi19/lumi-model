# Lumi

Lumi is a from-scratch language-model research project by Afnan Technologies Inc.

The first target is a dense, decoder-only ~4B-parameter base model that can be
post-trained into `Lumi-4B-Instruct`, served with an OpenAI-compatible API, and
quantized for Apple Silicon. This repository contains model code, configuration,
tests, and small smoke-training utilities. It intentionally does **not** contain
training data, secrets, or model checkpoints.

## Current phase

Phase 1 architecture correctness is complete. Phase 2 validates checkpoint,
tokenizer, data, and deployment portability before any expensive GPU run:

1. self-contained Hugging Face export and isolated-process loading;
2. vLLM Transformers-backend compatibility;
3. one frozen 65,536-token contract shared by the 150M pilot and 4B candidate;
4. a pinned 25MB tokenizer canary, a metadata-preserving 10MB provenance
   canary, and corrected 32K/65K tokenizer prototypes pass their local gates;
5. a contextual-PII-v4 1GB development sample passes all 23 component quotas,
   language/secret/provenance gates, near-deduplication, benchmark
   decontamination, full post-write validation, and manual PII snippet review;
6. disk-backed MinHash and byte-offset validation checkpoints pass regression
   coverage and a real 25MB, 23-source disconnect/resume canary;
7. a SHA-256-pinned Bengali Wikipedia Parquet source, category-scoped language
   policy, and 1MB real-network canary pass with retained article URLs;
8. the v0.2 full-size attempt exposed a single-anchor assumption in a multi-file
   component without spending GPU credit; v0.3 adds generic component-level
   continuation, per-file provenance, resumable exhaustion state, and mandatory
   full-quota capacity gates;
9. the resulting 6GB accepted-text sample is sealed at 1,128,089 documents and
   SHA-256 `80bea9f9634a8daca542b2ffd0aad1c8c994a7cbeef909a5e529b6272671102a`;
   resumed full validation and the cached validation check pass;
10. the full 1,128,089-document bakeoff selects 65K: it emits 4.39% fewer
    tokens overall and wins in all eight measured domains;
11. deterministic fixed-sequence packing, sealed-manifest source stratification,
    and checkpointed training pass interruption/resume tests; an 8,192-token
    real-corpus Apple-MPS smoke run saved at step 1 and resumed in a fresh
    process through step 2;
12. the 24-source 50M target is materialized at 49,999,872 tokens in 48
    hash-verified shards;
13. the content-pinned Lambda preflight and paid H100 gate pass: a fresh-process
    smoke resume advances through step 2, and the 150,625,536-parameter model
    completes 763 steps over 50,003,968 tokens at a measured mean of 117,888
    tokens/second; the final resumable checkpoint is hash-verified on the Mac;
14. a hash-ranked, 24-source model-weight-held-out set is sealed at 4,999,168
    tokens with zero ID/content/near-duplicate overlap; a forced-stop MPS run
    resumes and completes at loss 4.680812, perplexity 107.858, and 4,686.9
    tokens/second on the 16GB M5 Mac;
15. fixed generations remain repetitive and factually wrong, as expected from
    a 150M model trained on only 50M tokens;
16. a self-identifying Lumi MLX artifact passes exact config/tensor/RoPE gates,
    prompt-logit parity, and 128/128 cached greedy tokens; its 4-bit affine form
    passes a 98,256-prediction-token stratified quality gate, is 83.70% smaller,
    and reaches a measured 522.3 cached decode tokens/second on the 16GB M5 Mac;
17. the sealed 4-bit artifact serves through a loopback-first OpenAI-compatible
    chat-completions API and responsive local UI; a real M5 black-box gate passes
    packaged HTML/CSS/JavaScript, model discovery, JSON generation, SSE
    streaming, usage accounting, and terminal `[DONE]` handling;
18. an exact 1,050,981,888-parameter scaling candidate preserves the 4B model's
    36 layers and 4K context at half width; its fail-closed plan pins permanent
    held-out exclusion, four cumulative stages through 1.4B tokens, gradient
    checkpointing, cosine scheduling, bounded checkpoints, and measured-input
    cost projection without approving a paid GPU launch;
19. a reusable full-corpus source index reproduces 1,476,228,500 tokenizer
    tokens and leaves 1,472,219,479 held-out-safe eligible tokens; a 16,777,216-
    token canary and final 1,399,996,416-token uint16 artifact pass fresh-process
    resume, cached-resume, rank-prefix, shard-integrity, and zero-leakage gates.
    The final payload is 2,799,992,832 bytes in 42 shards; no GPU was launched;
20. a backward-compatible schema-v2 cloud preflight pins that exact manifest,
    selection proof/state, tokenizer, 1B config, and scaling contracts. A full
    local rehash passes; the Mac correctly fails CUDA and 32GiB host-memory
    admission floors;
21. the same no-skip preflight passes on a live Lambda 1x A100 40GB SXM4 in
    India before training. One optimizer step and a separate fresh-process
    resume advance the 1,050,981,888-parameter model through step 2 and 65,536
    tokens with finite loss/gradient norm, exact data-position recovery, and a
    23.744GiB measured peak CUDA reservation. The latest checkpoint is archived
    on the Mac with all 14 server-generated hashes verified, and the Lambda
    instance is terminated;
22. a separately authorized A100 run restores that exact checkpoint and reaches
    cumulative step 100 and 3,276,800 tokens. Loss falls from 11.335843 to
    7.690175, steps 10--99 sustain a 9,379.6 tokens/second median, and peak CUDA
    reservation remains 23.744GiB. Checkpoint 100 and its reports occupy about
    12GB on the Mac, all 21 server-generated hashes pass, and the second Lambda
    instance is also terminated;
23. a third authorized A100 run resumes that state and hard-stops at cumulative
    step 7,630 and 250,019,840 tokens. Frozen held-out loss improves from the
    150M baseline of 4.680812 to 3.265504 (perplexity 26.193), so the learning
    gate passes, while fixed greedy generations remain repetitive and wrong.
    Median steady-state throughput is 9,382.6 tokens/second and the training
    segment costs USD 14.723236 at USD 1.99/hour. The final 12,612,139,807-byte
    checkpoint and all run evidence are archived on the Mac; all 35 final
    manifest hashes pass independently, and the third Lambda instance is
    terminated.

The 4B configuration is a candidate until all deployment gates pass. The 1B
design, hash-ranked held-out-excluding artifact, two-step A100 smoke gate, and
100-step calibration and 250M-token learning gate are complete. The 1.4B-token
stage and every 4B stage remain unapproved. The completed 150M and 1B pilot
checkpoints are not production-quality models.

## Quick start

Use Python 3.12. The quality-gated data pipeline pins Lingua 2.2.0, whose wheel
requires Python 3.12 or newer; using one interpreter and detector version keeps
language decisions reproducible.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,data]'
pytest
python scripts/count_parameters.py configs/lumi-4b.json
python scripts/smoke_train.py --steps 20
python scripts/check_isolated_hf.py
```

The real-data packer and resumable pilot runner are documented in
[`docs/real-data-pilot-v0.1.md`](docs/real-data-pilot-v0.1.md). Keep their
generated shards and checkpoints outside Git.
The paid CUDA gate and exact transfer/training contract are documented in
[`docs/lambda-pilot-v0.1.md`](docs/lambda-pilot-v0.1.md).
The held-out construction, MPS evaluation, and quality decision are documented
in [`docs/evaluation-pilot-v0.1.md`](docs/evaluation-pilot-v0.1.md).
The native MLX conversion, FP32 parity, 4-bit quality, and M5 runtime results are
documented in
[`docs/local-inference-gate-v0.1.md`](docs/local-inference-gate-v0.1.md).
The loopback OpenAI-compatible API, local chat UI, security boundary, and real
server gate are documented in
[`docs/local-serving-v0.1.md`](docs/local-serving-v0.1.md).
The exact 1B architecture, staged token contract, trainer changes, input audit,
measured-throughput cost rule, and verified 250M-token learning gate are
documented in
[`docs/scaling-pilot-1b-v0.1.md`](docs/scaling-pilot-1b-v0.1.md).
The deterministic source index, 16M canary, final aligned 1.4B pack, resume
boundary, and verified hashes are documented in
[`docs/scaling-data-1b-v0.1.md`](docs/scaling-data-1b-v0.1.md).
The exact 1B transfer bundle, runtime/storage floors, local preflight result,
remote CUDA gate, smoke/resume boundary, and measured 100-step calibration are
documented in
[`docs/cloud-pilot-1b-v0.1.md`](docs/cloud-pilot-1b-v0.1.md).

## Repository policy

- Never commit `.env` files, API keys, datasets, checkpoints, or generated
  weights.
- Keep source provenance and dataset manifests outside raw data storage.
- A green tiny-model test suite is required before scaling.
- A measured single-GPU throughput pilot is required before approving the 4B
  run.

See [`docs/architecture-v0.1.md`](docs/architecture-v0.1.md),
[`docs/portability-v0.1.md`](docs/portability-v0.1.md), and
[`docs/tokenizer-v0.1.md`](docs/tokenizer-v0.1.md) for the current decision
records. The candidate source mix and acquisition gates are in
[`docs/data-plan-v0.1.md`](docs/data-plan-v0.1.md), with the first measured run
in [`docs/data-canary-v0.1.md`](docs/data-canary-v0.1.md).
The metadata-preserving follow-up is in
[`docs/provenance-canary-v0.2.md`](docs/provenance-canary-v0.2.md).
The current audited all-component result is in
[`docs/development-sample-v0.4.md`](docs/development-sample-v0.4.md).
The current scale-up gates are in
[`docs/tokenizer-sample-v0.3.md`](docs/tokenizer-sample-v0.3.md), and the pinned
Bengali source decision is in
[`docs/multilingual-source-v0.1.md`](docs/multilingual-source-v0.1.md).
