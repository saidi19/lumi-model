# Authentic local safety-output reproduction

## Completed on 2026-09-06

One bounded local replay reconstructed all 320 safety records, reran the original
generic filtering and supervised-token measurement, and matched **all four saved
outputs byte-for-byte**: candidate, manifest, report and verification. There were
320 accepted records and zero rejections. Train: 240 records / 18,636 supervised
tokens. Validation: 80 records / 6,388 supervised tokens.

This extends the earlier metadata assurance and authentic row-delta audit. It is
not a full private integration pass, execution of the original pytest nodes,
qualified rights approval, dataset promotion/sealing or permission to start SFT.
The pretrained base and S3 backup were not accessed or changed.

## Input and computation boundary

The exact input specification is
`configs/lumi-safety-output-replay-inputs-v0.1.json`, SHA-256
`42aaeba21e68d249f41671587c74a92c7adfb6f298e0e19a2efbe390245ac33c`.
It preserves every original safety-plan binding and explicitly pins the saved
outputs and supporting source. The entry point is
`scripts/check_safety_output_replay_local_v1.py`.

The real invocation read each of 24 bound files exactly once: **11,860,091 bytes**
under a 12,000,000-byte cap. A separate 21,605-byte historical source blob was
read from local Git at `e08bddc8fbd8afe416a0d09aed6a115f06870f36`, authenticated
against the original data-implementation digest
`df8297e753e97028999c6bcd203edc2a00a02147f7d9ecb2255569c480363970`.
The current working implementation was not replaced. The deadline was 120
seconds including specification acquisition, Git lookup and runtime startup.

Private prerequisites included the two safety candidates, predecessor manifest,
owner receipt and the two frozen tokenizer files. The benchmark input was the
already-local hash-signature registry, not raw held-out benchmark examples.
No additional dataset or model-weight files were read.

The no-follow reader rejects symlinks, hardlinks, wrong types/sizes/hashes,
duplicate reads and path/inode/content-metadata changes. It retains descriptors
through validation, verifies preservation, closes them and tears down the timer
before releasing success. Specification and Git stdout reads are bounded during
acquisition, not just checked after unbounded reads.

Authenticated source is compiled into an isolated in-memory package. Original
plan, predecessor, owner-receipt, policy, source-registry, row-repair, filtering,
token-count and serializer bodies remain unchanged. Their filesystem access is
redirected to an immutable byte snapshot. The frozen serialization tail is
extracted without modifying its statements; the old runner and publisher are
never called. Snapshot logical reads are separately bounded: 46 virtual opens
debited 44,601,746 bytes, below 256 opens / 256 MiB, with at most 32 MiB per gzip
expansion. Those are in-memory reads, not repeated disk input reads.

The sole computation-entry substitution is tokenizer construction from the two
authenticated byte strings instead of directory discovery. The public loader
requires Transformers 5.2.0 and tokenizers 0.22.2 and constructs the genuine
TokenizersBackend. Complete vocabulary and special-token sets must remain
unchanged. The original tokenizer-file co-location, chat-token, context-window,
split-special-token and supervised-loss-role checks remain in place.

## Runtime and verification facts

Runtime-only preflights occurred before the real replay and read zero bound
inputs. They identified dependency-startup tempfile/hardlink probes and runtime
library reads. Trusted dependency warmup now precedes the strict private-input
phase and runs with network/process creation denied. Filelock performs its own
empty temporary-directory capability probe and cleanup during warmup; this is
not a claim of zero startup scratch writes. No input/private-output file was
written. Importing a runtime dependency such as Torch is not model construction,
device use or inference; none of those operations was performed.

After warmup, a Python audit guard denies writes, network/process creation,
database opens and unbound content reads outside trusted runtime/source roots.
This is defense in depth around reviewed code, not an OS sandbox or native
system-call tracing claim. The CLI retains only fixed stage/status codes and
aggregate counts; no candidate text, IDs, token strings or exception text.

There was exactly **one actual private replay**, which passed; it was not retried.
Observed result:

```json
{
  "status": "PASS_SAFETY_OUTPUT_REPRODUCTION_ONLY",
  "safety_outputs_reproduced": true,
  "saved_outputs_byte_matched": 4,
  "candidate_rows_reconstructed": 320,
  "generic_filter_accepted": 320,
  "generic_filter_rejected": 0,
  "tokenized_rows": 320,
  "by_split": {
    "train": {"records": 240, "supervised_tokens": 18636},
    "validation": {"records": 80, "supervised_tokens": 6388}
  },
  "bound_input_opens": 24,
  "bound_input_bytes_read": 11860091,
  "historical_source_bytes_read": 21605,
  "input_hashes_authenticated": true,
  "input_files_preserved": true,
  "blocked_operation_attempts": 0,
  "input_files_written": 0,
  "network_used": false,
  "gpu_used": false,
  "inference_called": false,
  "original_test_nodes_executed": 0,
  "full_private_integration_passed": false,
  "execution_authorized": false,
  "production_eligible": false
}
```

Fifty new synthetic tests pass: 25 snapshot tests, 11 tokenizer tests, 10
core/input tests and four lifecycle tests. Independent review led to bounded
acquisition, descriptor registration before fstat, independent timer cleanup,
namespace-collision preservation and tokenizer co-location regression checks.
Together with adjacent metadata, row-delta and CI-routing tests, the final local
suite passed 171 tests and 328 subtests. New Python files pass the exact CI Ruff
rules.

The last pre-push GitHub run used Transformers 5.15.1. Constructor unit tests
there exercise genuine synthetic-tokenizer behavior separately from historical
runtime authorization. The public gate explicitly rejects unsupported installed
versions; tests do not impersonate a supported version or skip these checks.
The real private replay above used the required local 5.2.0 runtime.

## Remaining readiness boundary

The safety-output reproduction requirement now has authentic local evidence.
The original four private-dependent pytest nodes were not run by pytest and are
not relabelled as executed. The immutable 327-node historical inventory remains
OPEN and unchanged; it is not 327 newly observed code bugs. This result must be
accounted as separately verified safety-family evidence, not subtracted from
that frozen inventory or promoted to a whole-project integration pass.

Other private integration dependencies and downstream SFT readiness still need
their own evidence. Existing source-suite routing, full-readiness gates, frozen
source/tests, previous reports/markers, raw-text restrictions and production
eligibility were not weakened. No GPU/SFT run was started.
