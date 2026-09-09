# SFT execution status v0.60

Bounded source implementation, 2026-09-09, based on accepted repair commit
`535ffc2c9be4f9611fd051a8e632b3696f30250d`. The user requested the next Gate 5
step and allowed heavier Mac CPU work if needed. Heavy local work was not needed.
This implements the **created-OCI identity handoff**, not completed native
guardian/worker integration, a new acceptance run or permission for real SFT.

## Concrete source change

`src/lumi/posttraining_sft_oci_admission_v1.py` independently pins accepted CPU
image `sha256:321d50ae5207dee05cd58ec5436d97978becde2ee8a767f2e8768229e6a9ad43`,
its c4272e3 source commit, lock/base/source controls and prior acceptance-report
hash. Its public verifier accepts only a full container ID and bounded deadline;
there is no caller-selected image, supplied success receipt or bypass flag.

On the intended trusted Linux host, it reads the image then the never-started
container using only local-socket `docker image/container inspect`. The CLI has
an empty private config directory, a fixed local socket and a minimal environment.
Each response is bounded while streaming to 65,536 bytes, each command to five
seconds, and the enclosing handoff to at most 15 seconds of cooperative deadline.
Failed commands kill/reap only their own CLI child and close its stdout pipe.
Stderr and arbitrary exception text are not retained. The independent guardian
is still required for hard lifetime enforcement of this caller.

Image/config/control pins, never-started state, fixed entrypoint/environment,
nonroot user, read-only filesystem, network/IPC/privilege/mount/device restrictions
and the existing resource profile are checked. A one-shot continuation receives
an immutable software binding with a domain-separated source identity digest.
Reentry, reinitialization, parent loss, expired deadline and callback failures
cannot revive a consumed handoff. This metadata is **not** an execution permit.

The host, Docker daemon and exclusive control of the container are trust
premises. A point-in-time inspect response does not prove current exported file
contents or freeze a container against another daemon client. The code supplies
no pull/create/start/export/mount/training operation. Status keeps rootfs,
guardian containment, worker identity and real-SFT/production claims false.

## Duplicate-workflow risk fixed before publication

The historical OCI workflow still matched every `src/lumi/**` change with only
a private-repository guard. Publishing this implementation without correcting
that condition would have started its old failed build again. The necessary
binding-test update would also have triggered the old diagnostic.

Only these two live job conditions and explanatory comments change:

- OCI candidate: private repository, first attempt, exact original
  `edaa86e238223cf3a1ecfb6292640d8bc43e72de` SHA.
- Origin diagnostic: private repository, first attempt, exact original
  `15127b62e702244923a6f56c1a936fb667ea4486` SHA.

The already-accepted lookup workflow remains bound to c4272e3. Its condition
does not change. All three historical native jobs must skip this successor.
Two directly dependent source-test hashes are updated. New byte tests reverse
exactly the added comment/condition and recover each original workflow SHA,
proving no other workflow bytes changed. Old builder/probe/fixture/gate sources
and prior Git history/evidence remain untouched. No dispatch, schedule or retry
mechanism is added. Source and Spark are the only intended executing CI workflows.

## Verification before publication

321 selected stdlib/synthetic/AST tests passed in 0.317 seconds; exact-file
Ruff E/F/I/B/UP (Python 3.11 target, line length 100) passed. The 70 new tests
comprise 51 metadata/handoff tests, 13 inert CLI transport tests and six AST/byte
binding tests. Other selected tests cover accepted release manifest/verifier/gate,
OCI packaging, lookup and diagnostic contracts. Tests use no real Docker process,
native fixture, model/data, ML library or network. Independent source review found
no blocker for this scoped source implementation and Source-CI publication.

Key source bindings:

- Admission: `a2f26771b22fedfe8a003bc3104c40f46477465b7f0ab2c50bee7ad3abb945f0`.
- Metadata tests: `b0f29fc320b3bfbfa56d6ececfd61461fe37bc638041347bd36e43c6f55f378a`.
- Transport tests: `a1c04b6128666c82edffc4b895f9feef32ec9c2fb0f6f3a8ea8a9a34e1da74f0`.
- Live OCI workflow: `48af59107d4e17c898bb3c2508fad0c1a37cb2bdf152ce564252c4af93767b00`.
- Live diagnostic workflow: `b8a2f6d838138bc16f3666cac4f4749da49d39ae63662b28ce9f23fbc7bbf76f`.

## Remaining actual composition

Gate 5 is still incomplete. No real daemon metadata handoff or native acceptance
has been executed by this step. `WorkerRun` still receives its old synthetic
admission/source identity in the old fixture; this module does not claim otherwise.

The next composition must attach the actual workload to the host guardian's
owned cgroup before release. Merely placing `docker start` CLI in that group
does not contain daemon-created processes. A reviewed alternative is to export
the never-started identity-verified image and run its own interpreter under the
existing host guardian, using a bounded safe extraction and a new read-only
rootfs mount bootstrap. The old mount helper binds host runtime/system files;
it cannot be reused unchanged as proof of the accepted OCI runtime. Export
provenance, explicit bootstrap source binding, admission/source-identity delivery
and positive/negative native verification remain implementation work, not passed
gates. Any new composition must be labelled separately from prior CPU acceptance.

No GPU/instance, S3/SSH, model/data read, real training, registry promotion or seal
occurred. Existing data, rights and training-profile constraints remain unchanged.
