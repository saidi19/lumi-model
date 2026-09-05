# SFT execution status v0.7

Verified locally on 2026-09-05. The new private source-review package and
cooperative lifecycle/admission development tests are complete. **The integrated
GPU worker and operating-system resource supervisor are still incomplete.**
This is additional local preparation, not completed GPU SFT or a deployable
release. All earlier source, configuration, status and evidence files were left
unchanged. Production eligibility remains false.

## What is complete in this successor

1. A deterministic, private source-review ZIP containing 34 selected source and
   control files, including their statically found local imports. Original bytes
   are preserved. Its internal notice and manifest explicitly prohibit treating
   historical worker code as an executable release. It contains no model weights,
   training rows, encoded payloads, owner approvals or authentic target receipts.
   It is not a backup of every uncommitted project file and was not pushed to GitHub.
2. A stdlib cooperative supervisor core: one fixed three-invocation lifecycle,
   aggregate rank reservations before reads/writes, log/private-write accounting,
   exact optimizer boundaries, integer cost arithmetic, combined monotonic
   deadline, strict memory observations, and sticky failure/no refunds/no retry.
3. A separate admission-development boundary: exact frozen base/data/profile
   identity, external contract/approval pins, expiry and target/receipt schema
   checks before synthetic callbacks. Its public real-worker entry always denies.
   Successful synthetic traces cannot issue an execution permit.

The three new suites pass 26 source-package, 51 supervisor, and 36 admission
tests. The existing 54-test launch-contract suite also passes unchanged:
**167 tests, zero failures/errors/skips**, both on publication and on a separate
fresh-process replay. The replay uses the same test implementation, not an
independently written test suite. One development test previously replaced a
null value with null; its mutation fixture was corrected before publication.
No published attempt was retried.

Both passes verified 28 previously bound source/control/evidence files before
and after work. The 34 packaged files were reread and compared after tests.
Fresh verification reopened the saved ZIP without extracting or executing it,
checked its exact member set, private file modes, sizes and hashes, and reproduced
the archive byte-for-byte. Directory mode is 0700; ZIP/reports/markers are 0600.

The two published passes wrote 692,333 bytes total, including the ZIP, reports,
and separate consumed started markers. Synthetic tests also created and removed
small temporary fixture files; this number is not an OS-wide disk-write meter.
There were no model/data-content reads, cloud/network operations, GPU/inference,
child processes, Git writes, remote operations, or instance actions in this
successor. The fixed runner's audit hook is defense in depth, not an OS sandbox.

## Artifact locations

- [Development plan](../configs/lumi-4b-sft-local-readiness-v0.1.json)
- [Private source-review ZIP](../artifacts/lumi-sft-local-readiness-v0.1/attempt-001/private/source-review-only.zip)
- [Publication report](../evidence/lumi-4b-sft-local-readiness-v0.1.json)
- [Fresh-process verification](../evidence/lumi-4b-sft-local-readiness-verification-v0.1.json)
- [Source bundle implementation](../src/lumi/posttraining_sft_source_bundle_v1.py)
- [Cooperative supervisor](../src/lumi/posttraining_sft_supervisor_v1.py)
- [Admission development boundary](../src/lumi/posttraining_sft_worker_admission_v1.py)
- [Local check/publication CLI](../scripts/check_sft_local_readiness_v1.py)

Plan SHA-256:
`5a7ab18722864814a67a76fab60f8221d89057b333d2afcf45fc2ddb332ae268`

ZIP SHA-256 (687,376 bytes):
`6d464483f91af9a91175ea75b83660a01e1893ead86ec080d85654a93ab3812a`

Report file SHA-256:
`40ed1021847c6049ddb7396b8b25d23bf24038618bb8f15b9f6ce1b2793da36f`

Verification file SHA-256:
`8bacfb5819d737abdd715143baf26bc68ec1b194c29ed42a7b760f6c6ed5fee2`

Do not republish/reverify into these consumed paths. The non-publishing recheck
only runs the named source/control checks and synthetic temporary fixtures:

```sh
PYTHONPATH=src artifacts/lumi-sft-import-runtime-v0.1/venv/bin/python -S -B \
  scripts/check_sft_local_readiness_v1.py --mode check \
  --expected-plan-sha256 5a7ab18722864814a67a76fab60f8221d89057b333d2afcf45fc2ddb332ae268
```

## Our existing model, data and code

The starting model remains our own `lumi-4b-base`, final pretraining step 76,294
at exactly 80,000,000,000 tokens, not a third-party model or CPU fixture.
[The immutable base binding](../configs/lumi-4b-final-base-binding-v0.1.json)
records its full S3 bucket/prefix, DCP manifest, tokenizer and handoff hashes.
The preceding read-only inventory turn found 53 objects / 48,844,545,022 bytes
at that exact backup prefix, matching the binding's count and byte total. It
did not restore weights or rehash all remote object contents. A small metadata
GET encountered endpoint connectivity failure; no full restore receipt exists.

Private sealed rows remain under
`artifacts/lumi-sft-learning-candidate-seal-v0.3/attempt-001/private/`.
The encoded directory remains
`artifacts/lumi-sft-token-consistent-shards-v0.2/attempt-001/private/encoded`.
Their existing records specify 309,643 train and 19,557 validation examples,
11,855,217 train supervised tokens and 939,293 validation supervised tokens,
with 9 train and 1 validation shard. No re-encoding or content audit happened here.

The preceding live GitHub inspection found the private
`saidi19/lumi-model` posttraining branch at commit `38499dc` (2026-08-09).
Latest local SFT work was not included there. This successor has not committed
or pushed anything. The ZIP is only a selected review snapshot, not a substitute
for a reviewed, dependency-complete Linux/CUDA release and separate private-data
transfer.

## Remaining prelaunch conditions: unchanged six, not new roadmap stages

| Condition | Current state and remaining work |
|---|---|
| 1. Real base restore/import | S3 inventory located; bounded real DCP restore/import, tensor/tokenizer verification and authentic receipt remain. The existing import adapter's synthetic parameter cap remains unchanged. |
| 2. Guarded SFT worker/supervisor | Contract compiler, admission simulation and cooperative ledger are tested; standalone pre-ML bootstrap, actual worker integration, cross-process accounting, persistent no-retry state and OS-enforced limits remain. |
| 3. Target/runtime/budget | Exact new instance identity, GPU UUIDs, Linux/CUDA lock, total spending/runtime/storage/I/O limits are unbound. Historical instance and pretraining caps cannot substitute. |
| 4. Private data relocation | Original encoded content is retained; exact target transfer, full content verification and new host-specific metadata binding remain. |
| 5. CUDA acceptance | Masking, precision, memory, import, fresh resume and fault criteria still need an exact target-bound specification and real execution evidence. |
| 6. Bounded execution approval | Requires complete inputs, target and resource caps. Current local approval does not mint a GPU execution credential. |

The in-memory supervisor only shares state across threads of one process. It
does not supervise separate torchrun ranks, asynchronously stop a process, enforce
memory/network isolation, terminate an instance, or stop provider billing. Memory
observations do not prove continuous memory compliance. Its clock excludes host
lifecycle time before construction and after completion. The six dynamic import
call sites and JSON payload references in the source snapshot are not resolved;
Linux/CUDA dependencies are not packaged or verified.

The next owner decision is the maximum total budget for the **new** GPU run.
Then bind a suitable target/runtime and price, prepare the real bounded
restore/relocation and worker integration successor, review the execution package,
and obtain its exact execution approval. Do not launch a billable idle instance
while those prerequisites remain unresolved; leave the existing busy instance
untouched. Do not copy root AWS credentials into a GPU image or source bundle.

After prelaunch closure: real GPU smoke/resume, equal-token learning-rate screen,
bounded full SFT/checkpoint verification, frozen capability/safety evaluation,
then authenticated private serving/product readiness. Preference optimization
is conditional. Public/production use still requires rights/owner/policy and
release clearance. `production_eligible=false` is unchanged.
