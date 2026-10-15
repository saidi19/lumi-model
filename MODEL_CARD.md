# Lumi research model card

Status: research source; no instruction-tuned model release is established here.
This card distinguishes configuration, repository reports and independently
checked results. It does not license weights, datasets or tokenizer assets.

## Architecture

`configs/lumi-4b-v0.2.json` specifies a dense decoder-only transformer with 36
layers, hidden width 3072, SwiGLU intermediate width 8384, 64 query heads,
8 key/value heads, head dimension 64, 65,536 vocabulary entries, tied embeddings
and a 4096-token context. Analytical counting gives **4,002,376,704 parameters**.
The implementation includes RMS normalization, rotary positions and an optional
document-isolated packed-attention path. This is not a claim to reproduce any
undisclosed proprietary architecture.

## Training evidence and limits

| Claim | Evidence | What this establishes |
| --- | --- | --- |
| Exact parameter count | `scripts/count_parameters.py configs/lumi-4b-v0.2.json` | Locally reproduced analytical count, not checkpoint loading |
| 4B learning pilot: 80 steps, 83,886,080 tokens | [pilot report](docs/4b-learning-pilot-v0.1.md) | Repository-reported pilot on 8 H100 GPUs; archived artifacts were not downloaded or rehashed in this audit |
| 80B production schedule | [production plan](docs/4b-80b-production-run-v0.1.md) | A plan, not proof that training completed |
| Completed 80B base pretraining | `configs/lumi-4b-final-base-binding-v0.1.json` on development commit `284a1f55eba368ef77d84b7d3342712ea89e8585` | Development-branch metadata reports step 76,294 and 80B tokens; not independently verified here |

The pilot report records held-out loss improving from 9.605866 at step 10 to
8.046854 at step 80. These are pilot measurements, not final-80B metrics or a
comparison with other models. Reported pilot generations remained repetitive.
No downstream benchmark score, chat quality, safety guarantee or production
eligibility is established by the OSS audit.

## Intended use and risks

Use the source for architecture research and small synthetic correctness tests.
Any future model release needs a specific checkpoint identity, reproducible
evaluation, limitations, distribution rights and a separate license decision.
Do not rely on an unevaluated model for consequential decisions. Generated text
may be inaccurate, biased, unsafe or expose memorized material; the current
checks do not quantify these risks.

## Data and distribution

Source provenance and filtering plans live in repository documentation. Their
presence is not a blanket redistribution grant. Private datasets, checkpoints,
owner-review packets, credentials and cloud artifacts are not part of a source
release. Dataset, tokenizer, benchmark-signature and weight rights require
separate review. No downloadable weight release is promised by this card.
