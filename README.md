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
3. one tokenizer contract shared by the 150M pilot and 4B candidate;
4. a pinned 25MB tokenizer canary, a metadata-preserving 10MB provenance
   canary, and 32K/65K tokenizer prototypes pass their local gates;
5. a contextual-PII-v4 1GB development sample passes all 23 component quotas,
   language/secret/provenance gates, near-deduplication, benchmark
   decontamination, full post-write validation, and manual PII snippet review;
6. next: an explicitly licensed Bangla/multilingual tokenizer slice,
   representative tokenizer data, then MLX and GGUF conversion;
7. only then: a small real-corpus pilot and measured GPU throughput.

The 4B configuration is a candidate until all deployment gates pass. Do not
start a full H100 training run from this repository yet.

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
