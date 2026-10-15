# CPU-only contributor quickstart

Use Python 3.12 or 3.13 in a virtual environment. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
PYTHONPATH=src OMP_NUM_THREADS=2 python -m pytest tests/test_config.py tests/test_model.py
PYTHONPATH=src OMP_NUM_THREADS=2 python scripts/smoke_train.py --steps 3 --batch-size 1 --sequence-length 16
PYTHONPATH=src python scripts/count_parameters.py configs/lumi-4b-v0.2.json
```

Installation needs package-network access. The commands after installation use
synthetic tiny-model inputs or analytical counting; they do not require a
checkpoint, dataset, AWS credentials, or GPU. Counting the 4B configuration does
not instantiate 4B weights. The synthetic smoke run is not pretrained-model
inference, an SFT run, or evidence of useful language quality.

The smoke and model/config tests were exercised locally with Python 3.13,
PyTorch 2.10.0 and Transformers 5.2.0. A clean installation and the full supported
dependency matrix have not yet been verified by this audit. Dependency ranges
are not a reproducibility lockfile.

For broader development, install `.[dev,data]` and run `ruff check .` and
`pytest`. Those broader checks are separate from this quickstart. Do not run
cloud or production scripts just to verify a contribution.
