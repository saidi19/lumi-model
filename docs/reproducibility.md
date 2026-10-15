# Reproducing research without overstating evidence

## Safe local entry point

Follow [CPU quickstart](cpu-quickstart.md). It exercises the real architecture
implementation with tiny synthetic inputs. The analytical 4B count should be
4,002,376,704. Smoke loss values depend on the environment; a tiny loss change is
not a model-quality metric. Python/PyTorch/Transformers versions must accompany
results. Dependency ranges are currently not a fully locked environment.

## Artifact-dependent entry points

These are existing entry points, not permission to execute them or instructions
to substitute arbitrary data for frozen artifacts:

- `scripts/train_pilot.py`: local pilot training.
- `scripts/train_distributed.py`: distributed training; real production runs need
  approved hardware, pinned data/tokenizer/configuration and checkpoint lineage.
- `scripts/evaluate_checkpoint.py`: requires plan, checkpoint, manifest, tokenizer
  and output arguments; supports explicit device and resume options.
- `scripts/export_distributed_checkpoint_hf.py`: distributed checkpoint export.
- `scripts/checkpoint_handoff.py`: checkpoint handoff tooling.
- `scripts/check_tokenizer_contract.py`: tokenizer-contract validation.

Use `--help` and the versioned experiment instructions rather than inventing a
universal production command. These artifact-dependent runs were not executed
in the OSS audit. The audit did not load a real 4B checkpoint or retrieve private
datasets, tokenizers or S3 objects.

## Evidence lineage

The [4B pilot report](4b-learning-pilot-v0.1.md) describes the 8-H100 experiment,
held-out evaluation and archive checks. The [80B plan](4b-80b-production-run-v0.1.md)
describes the later production schedule, not its completion. Later development
metadata reports exact step 76,294 / 80B completion; see [MODEL_CARD.md](../MODEL_CARD.md).

For a real reproduction, obtain authorized artifacts privately, check their
recorded hashes against the intended manifest and preserve failed verification
results. Confirm tokenizer identity, sequence boundaries, data-position state,
precision, world size and evaluation split. Do not call a matching filename or
a plan's expected token count proof of completed training.

Data preparation policies and restrictions remain in [data policy](../data/README.md)
and the versioned corpus documentation. No blanket dataset redistribution right
is inferred from source-code licensing. Private artifact unavailability limits
independent reproduction; this limitation must accompany reported results.
