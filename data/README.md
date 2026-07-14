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

Actual shards will live in private object storage. This directory is ignored
except for this policy file.

The first data artifact is not the 80B-100B-token corpus. It is a small,
representative tokenizer sample with source-level provenance. It must cover the
intended English-first mix (web, books/reference, code, mathematics/science,
and structured text) plus a multilingual slice. Tokenizer samples and resulting
vocabularies must be scanned for secrets, benchmark leakage, junk, and
source-specific artifacts before they are used.
