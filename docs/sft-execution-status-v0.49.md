# SFT execution status v0.49

Implementation handoff, 2026-09-09. The prior accepted release-ordering commit is
`3fcb22e01228052bd3abf3d69862ee8dc1f1b525`. This successor adds a real Linux CPU
software candidate; it does not revise any accepted older fixture, workflow,
parser, policy or evidence. Native acceptance is pending when this document is
committed. A subsequent local receipt must name the actual accepted commit,
workflow runs, image ID and retained artifacts, or record the precise stop.

## Actual software, not generated release placeholders

The new lock `configs/lumi-sft-linux-cpu-release-v0.1.lock.json` binds canonical
ASCII JSON with exactly one terminal LF. Its full-file SHA-256 is
`26bde6a0163e5db9d8d4564117546482601f4d0beb09f8705be5fc9183594965`.
It targets Linux amd64 / CPython 3.12 and contains 34 exact runtime wheels,
**248,715,637 bytes** in total. Versions were selected from the previously
successful CPU CI installation, not silently upgraded. Optional Spark/data/dev
packages are not part of this runtime. Actual dependency compatibility still
requires the offline installer check and generated CPU execution.

The official Python `3.12-slim-bookworm` tag was used only to discover a platform
manifest. Build and execution use the immutable selected digest, not the tag:

- Linux amd64 manifest:
  `sha256:9c47360a2a0355e2da18516d0b1c2126ec22c195d2185e97347c9d98398c5bef`
- Config:
  `sha256:59bf1d95c965f12dfc14afaf5af778fc1dbe5b372bd5c281645fc34d4c75d4e7`
- Actual declared Python: **3.12.14**; compressed base layers: **45,438,285 bytes**.
- CPU Torch: **2.14.0+cpu**, wheel **196,253,793 bytes**, SHA-256
  `a09987c95ec4cffdb6df798d3d641558110a334cfccce22f2f046d83142bc260`.
- Transformers **5.15.1**, Tokenizers **0.22.2**, Safetensors **0.8.0**.

Small official PyPI JSON metadata resolved the active Linux 3.12 dependency
edges against the already-observed versions. The official PyTorch CPU index
supplied wheel and metadata hashes. Its R2 host returned HTTP 403 for local
metadata/header requests; the primary official CDN returned the same exact
metadata hash and expected wheel-size header. The primary CDN URL is pinned;
CI must validate the complete wheel hash, with no redirect fallback or retry.
Docker manifest and config JSON hashes were verified before recording them.
No image layers or wheel binaries were downloaded on the Mac.

Hash-locked, binary-only installation follows the controls described by
[pip secure installs](https://pip.pypa.io/en/stable/topics/secure-installs/).
The tag/digest distinction follows
[Docker image digests](https://docs.docker.com/dhi/explore/security-concepts/digests/).
These public content hashes are not independent publisher signatures, a security
vulnerability audit, or proof of a reproducible build.

## One remote-only software candidate workflow

The dedicated private-repository workflow performs one build and one generated
CPU acceptance sequence. It does not push an image registry or touch S3/SSH.

1. Verify `GITHUB_SHA` equals clean tracked HEAD. Snapshot only tracked Git blobs
   under `src/lumi` plus the two exact WorkerRun/observer scripts. Never copy
   untracked artifacts, model/data files, owner packets or the entire checkout.
2. Pull and inspect the exact Python platform image. Inventory its actual
   interpreter/library regular files before installing wheels, plus bounded
   distribution metadata. Explicit SSL/zoneinfo symlink leaves are allowed;
   auxiliary directories are not recursively inventoried. This inventory is
   not a complete filesystem census; the immutable base image binds the rest.
3. Acquire each exact wheel once over validated public HTTPS, enforcing size
   and SHA-256 before use. No source distributions, package-source builds,
   mutable dependency resolver, redirects or automatic retries are used.
4. Build with network disabled, a read-only wheel build mount, `--no-index`,
   `--require-hashes`, `--only-binary=:all:` and `--no-deps`. Run metadata-only
   `pip check`. Include the locked controls and same-commit source in the image.
5. Create from the returned exact image ID; independently inspect that same
   container before starting it. Require nonroot 10001:10001, read-only root,
   network=none, all capabilities dropped, no-new-privileges, no device/GPU
   requests, no host software/source binds, and bounded writable temporary
   storage. Docker-generated hosts/hostname/resolv.conf files mean this is
   deliberately not described as literally mount-free.
6. Execute the accepted generated WorkerRun from this image, once. Its source
   remains byte-identical: 39,948 bytes, SHA-256
   `53c50c92965ed1cef0c9624c57367c31ed7ebfb9d7cfb34257188ad179d333d0`.
   Preserve three invocations, six rank workers, four global/eight rank optimizer
   updates and the existing ledger/resume/cleanup comparisons.
7. Require two file-origin observations for every rank: at the inputs barrier
   before optimizer release and before its final result. Verify declared
   distribution versions, Lumi/Torch/Transformers file origins, executable
   file mappings including `torch._C`, and mapping inode/device agreement.
   Verify successful exit, no OOM and no root-layer diff; retire that exact
   container. Never import ML in the observer's parent process.
8. Stream the accepted image archive with a pre-write size cap; retain it and
   small evidence privately as GitHub artifacts for 30 days. The artifact is
   software only: generated training files lived in container tmpfs, outside
   the immutable image. Do not download the large archive onto the Mac.

The real-source observer composes only `outcome()` and `run_invocation()`;
the underlying rank implementation is not replaced. The reviewed
[Torch 2.14 instantiator source](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/distributed/nn/jit/instantiator.py)
generates its one non-scriptable remote helper in memory. This exact helper and
anonymous executable mappings are counted, not falsely represented as
authenticated file content. Fileless generated/frozen/builtin modules are not
file-origin claims. `complete_dynamic_dependency_closure_verified=false` and
`all_executable_code_origin_authenticated=false` remain explicit.

## Bounds and unchanged limitations

- Builder work deadline: 19 minutes, with an independent 15-second exact-owned
  retirement budget; outer 20-minute timeout plus 15-second kill grace and a
  TERM handler; workflow timeout 25 minutes. No started-pass retry.
- Wheel closure: at most 64 wheels / 1 GiB; actual lock is 34 / 248,715,637 bytes.
- Source snapshot: at most 4096 files / 32 MiB, each Git blob at most 2 MiB.
- Base inventory: at most 40,000 files / 80,000 entry observations, prepaid
  file/metadata/EOF reads at most 1 GiB, one file at most 256 MiB, stdout 16 MiB.
- Worker container: 2 CPUs, 4 GiB RAM with no additional swap, 256 PIDs,
  512 MiB `/tmp`, 64 MiB shared memory, core dumps disabled.
- Saved image archive: at most 3 GiB with streaming pre-write enforcement.
  Retained output total checked at most 4 GiB. These are artifact bounds,
  **not** a claimed Docker daemon-wide disk-write quota; Docker layer/cache
  storage exists only on the ephemeral CI host.
- Runtime software-origin report: at most 8 KiB; proc maps at most 1 MiB;
  software-only failed-command diagnostics at most 64 KiB. No actual model,
  candidate, dataset, owner-packet or credential text is processed.

This builds and accepts the real **CPU software candidate**, not a CUDA runtime
or independent hostile-host attestation. The Docker daemon/host, reviewed source
and cryptographic hash assumptions remain trusted. This observer is not yet
integrated into the prior guardian's immutable-release admission path. The old
macOS lock and historical FSDP/context-4096 GPU draft remain unchanged and are
not used. Current supported WorkerRun still requires DDP and context 2–4095;
this generated acceptance uses context 32 / fp32.

Gate 5 remains partial until this actual candidate is accepted and integrated
worker/supervisor admission is completed. Dedicated real-host binding, original
pretrained S3 checkpoint restoration, exact data relocation and explicitly
approved H100/CUDA checks follow; real SFT, evaluation and rights/quality/release
gates remain downstream. `production_eligible=false`, real SFT permission false,
private integration false and full CI false are not waived by source CI.

Pre-publication local checks pass all 120 new pure/AST tests; exact-file Ruff is
clean. Accepted manifest/verifier/gate pure and AST regressions are also checked
without importing their Linux fixture or wrapper. These checks do not substitute
for the new native image acceptance or remote source/Spark regression jobs.

All Mac work is source editing, small stdlib/synthetic or AST/literal-only tests
and exact-file lint. Never import or execute the new native builder, base
producer, observer, Linux fixture or Linux wrapper on the Mac, even a skip path.
