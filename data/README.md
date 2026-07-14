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
