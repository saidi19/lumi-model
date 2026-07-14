# Data boundary

No raw or processed training data belongs in Git.

Future data work must track, at minimum:

- source and retrieval date;
- license/terms and intended use;
- raw, filtered, and deduplicated token counts;
- language/domain labels;
- quality and safety filters;
- benchmark decontamination results;
- stable document IDs and removal records.

The consolidated Comma dataset is allowed only for the v0.1 plumbing canary and
mix comparison. Development and final samples must use the pinned individual
filtered repositories in `configs/component-sources-v0.1.json`, retain selected
upstream metadata, and reject records without an accepted document license or
documented source-level public-domain/open-license basis. Schema/card mismatches
must remain visible in the manifest rather than being silently treated as a
document-level license.

Actual shards will live in private object storage. This directory is ignored
except for this policy file.

The v0.1 pipeline writes a local canary outside this repository. Build and
validate it with:

```bash
python scripts/build_tokenizer_sample.py --output /path/to/tokenizer-canary-v0.1
python scripts/validate_data_sample.py \
  --sample /path/to/tokenizer-canary-v0.1 \
  --config configs/tokenizer-canary-v0.1.json
```

The metadata-preserving v0.2 canary uses:

```bash
python scripts/build_tokenizer_sample.py \
  --config configs/provenance-canary-v0.2.json \
  --output /path/to/provenance-canary-v0.2
```

Use `--resume` only with the same config and target; the run fingerprint rejects
accidental continuation with changed inputs.

The first data artifact is not the 80B-100B-token corpus. It is a small,
representative tokenizer sample with source-level provenance. It must cover the
intended English-first mix (web, books/reference, code, mathematics/science,
and structured text) plus a multilingual slice. Tokenizer samples and resulting
vocabularies must be scanned for secrets, benchmark leakage, junk, and
source-specific artifacts before they are used.
