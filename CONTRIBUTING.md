# Contributing to Lumi

Start with the [CPU quickstart](docs/cpu-quickstart.md) and the
[model card](MODEL_CARD.md). Source licensing is still pending; see the
[licensing review](docs/licensing-review.md) before contributing code intended
for redistribution. Do not assume a contribution transfers ownership.

Open a focused issue describing the problem, expected behavior, environment and
a minimal synthetic reproduction. Avoid real training records, prompts with
personal data, credentials, private storage URLs and proprietary checkpoints.

For a pull request:

1. Keep changes small and explain the intended behavior and compatibility impact.
2. Add a synthetic regression test for a bug fix; report exact commands and results.
3. Run the relevant tests and `ruff check .`; disclose unrun checks and failures.
4. Preserve historical evidence and frozen bindings. Do not rewrite an old failed
   attempt to make it look successful.
5. Separate CPU fixtures from real checkpoint, GPU and evaluation claims.
6. Disclose copied code and its origin/license; retain required notices.

No contribution authorizes cloud provisioning, downloading private artifacts,
training, destructive cleanup, changing shared policies or publishing weights.
These need separate project-owner approval. Never put secrets in a PR or issue;
follow [SECURITY.md](SECURITY.md) instead.
