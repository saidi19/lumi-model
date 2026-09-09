# SFT execution status v0.55

Pre-publication plan, 2026-09-09. After the scoped local lookup fix, the user
asked to start the next step and then resume it. This successor runs one bounded
private GitHub CPU acceptance of that fix; no GPU or real SFT. It is not a retry
of either frozen failed run. Parent must be
`15127b62e702244923a6f56c1a936fb667ea4486`.

## Exact publication boundary

Add exactly nine files, with every previous tracked file byte-identical:

- `.github/workflows/sft-cpu-lookup-acceptance.yml`
- `docs/sft-execution-status-v0.55.md`
- `scripts/build_sft_cpu_lookup_acceptance_v1.py`
- `scripts/run_sft_cpu_lookup_acceptance_v1.py`
- `scripts/run_sft_oci_worker_probe_v2.py`
- `scripts/sft_cpu_lookup_acceptance_report_v1.py`
- `tests/test_sft_cpu_lookup_acceptance_report_v1.py`
- `tests/test_sft_cpu_lookup_acceptance_source_v1.py`
- `tests/test_sft_cpu_version_lookup_v2.py`

The safe-name version test is byte-identical to the existing local-only
`tests/test_sft_oci_version_lookup_v2.py`, which remains untouched and must not
be staged. Publishing its old name would match the old OCI workflow wildcard
and incorrectly launch another old acceptance. New builder/test paths avoid
both old OCI and old diagnostic workflow filters. Only this new CPU acceptance
and the existing Source/Spark regression workflows should start on publication.
Old local result receipts v0.50/v0.52/v0.53/v0.54 also remain local and unchanged.

Checkout fetches two commits; before native construction the new builder checks
the direct parent and exact nine `A` paths. Branch and run-attempt-1 workflow
guards prevent an automatic rerun. There is no manual-dispatch or retry path.

## Correction and immutable software bindings

The corrected probe differs from v1 by exactly one expression: public attribute
lookup with the original missing-value default replaces direct dictionary
lookup for `__version__`. The exact locked equality, string conversion, schemas,
all module/native origin checks, fixture, workload and nonclaims are unchanged.
The source-contract proof and 20 synthetic tests were completed in v0.54; no
native pass was claimed there. The new builder rechecks this exact byte relation.

SHA-256 bindings:

- Frozen software builder:
  `458faf97c62689c9d7cc71a6eb89a4f84fef8c68bbcf9279b949dee450565044`.
- Frozen v1 observer (22,633 bytes):
  `bed636e8cc08546a6be1b43c00c2f516fdc92dbf907dca674f7fb5aa7b5cb769`.
- Corrected v2 observer (22,638 bytes):
  `b45b04fc9cb93dd2301df0444f2db95666ba4f7a32e8f4d8e28c5b4077314450`.
- Frozen WorkerRun fixture:
  `53c50c92965ed1cef0c9624c57367c31ed7ebfb9d7cfb34257188ad179d333d0`.
- Frozen base inventory script:
  `34dc6a70b6d074808b864f53b4d511456423786aa62068ea771253fd4f8f29d1`.
- Frozen lock validator:
  `9f70737db527c038d2ddd54992ce6d1ff999e9c7f204f4828a3f64fa08543ed3`.
- Frozen software lock:
  `26bde6a0163e5db9d8d4564117546482601f4d0beb09f8705be5fc9183594965`.
- Frozen diagnostic supplied-byte validator:
  `d1ec96cd14636eb87e75c4156c7db40891f26d8764de183247434ccacfb0facf`.
- New acceptance wrapper:
  `6e453f0a8ab5dc84f126db1761a7c089e0a239cf588a20cdc56b6d5236af0cf4`.
- New pure envelope validator:
  `8c0cc1727e3d6b320586a6cf664bfc0648ed3e84930b2b9d4983683c14a36260`.
- Safe-name version test:
  `855a8712a084fda9fa48eaa319e03d628a79f949a0feab18ff0ec2990000581a`.

The Python 3.12.14 amd64 base and exact 34 CPU wheels / 248,715,637 bytes are
unchanged. Frozen snapshot/image methods perform the same tracked-source
snapshot, base inventory, hash-locked software acquisition, offline binary-only
install and dependency consistency check. Only image entry/source-script
bindings change; no lock update, library substitution or package fallback.

## One run, with full proof and failure localization

The new wrapper binds v2 and preserves the existing 72 fixed checkpoint mappings,
original failure hooks and workload. It calls `_V1.run()` exactly once and keeps
the complete returned observer/WorkerRun report. The output is a canonical
32-KiB envelope containing the unchanged diagnostic object and either the full
passed observer report or null on failure. No extra worker run is introduced.

The pure host validator checks exact keys/types, fixed flags and bounded counts,
calls the frozen diagnostic validator, and independently enforces every old
acceptance report condition. These include six ranks, three invocations, twelve
origin observations, all four observer verification flags, the full WorkerRun
flags and eight rank/four global optimizer steps. Diagnostic and observer
control pins and origin counts must agree. Extra source/exception text is rejected.
A valid failed diagnostic is retained as failure, never accepted as a release.

Before the sole CPU-container start, the builder records the successfully built
immutable image ID/control pins and an exclusive start marker. The exact created
container is inspected for nonroot UID/GID10001, read-only root, network none,
no host software/source binds or device requests, cap-drop ALL and
no-new-privileges. Clean exit 0, no OOM and empty Docker root-layer diff are
required; the exact owned container is retired before acceptance succeeds.

Only after full acceptance is an independent pre-archive receipt written and
that same image ID archived privately. An archive failure leaves release status
failed while preserving the separate actual CPU-acceptance fact. No archive is
uploaded after a failed build/acceptance step. Diagnostic `acceptance_passed`
remains false because the diagnostic object is not the release verdict; the
outer builder's `same_image_cpu_acceptance_passed` is the separate verdict.

## Bounds and unchanged nonclaims

- CPU workflow work deadline: 19 minutes; separate exact-owned cleanup: 15s;
  outer timeout: 20 minutes plus 15s kill grace; workflow cap: 25 minutes.
- One fresh acceptance-container start. The frozen builder separately starts
  one base-inventory container; this is not a second acceptance pass.
- Original generated workload: at most three invocations/six rank workers,
  four global/eight rank optimizer steps, fixture deadline 180s; outer CPU
  command at most 240s. No retry after any started pass.
- Container: 2 CPUs, 4 GiB RAM/no extra swap, 256 PIDs, 512 MiB tmpfs,
  64 MiB shared memory; same original worker resource caps.
- Frozen source caps: 32 MiB/4,096 files. Software wheel cap: 64/1 GiB;
  actual pinned set 34/248,715,637 bytes. No redirects or alternate acquisition.
- Image/archive at most 3 GiB; retained acceptance output at most 4 GiB,
  including reserved final report space. These are not daemon-wide disk quotas.
- Envelope at most 32 KiB, nested diagnostic at most 16 KiB, failure packet
  at most 4 KiB; private software-command failure logs at most 64 KiB each.
- GitHub artifacts private for 30 days. The accepted image stays in GitHub;
  never download image archives, wheels, checkpoints or datasets onto the Mac.
- Existing Source/Spark regression jobs run separately on publication; their
  outcomes are not substitutes for CPU acceptance or full/private integration.

All prior evidence remains byte-identical. No real model/data, GPU/H100,
instance action, S3/SSH, real inference/SFT, registry mutation, seal, promotion
or unrelated action. Production eligibility, real SFT permission, real-host
admission, complete dynamic closure, publisher signature, reproducibility,
private integration and full-CI claims remain false. Mac work is exclusively
small source edits, pure tests, AST/literal checks, lint and bounded evidence
inspection—never native imports/runs, even through a platform-skip path.

Native outcome remains unclaimed until the one remote run returns evidence.
Gate 5 remains partial pending that result and remaining integrated admission.

## Completed pre-publication verification

All **414** selected pure/synthetic/AST tests passed in 0.494s under Python
`-S -B`; this includes the 353 frozen regressions, 20 lookup tests, 21 envelope
tests and 20 new source/launch tests. Exact seven-file Ruff checks passed with
`E,F,I,B,UP`, line length 100, Python 3.11 target. Independent read-only review
found no blocker. No native script was imported, compiled or executed locally.

Final new builder SHA-256:
`83d95ee73c8cdd9e5d8efc1e5cc8ab7879689c83f68cb64946ef70881731096c`.
Envelope-test SHA-256:
`2883aab6ebc4fc96d23f6de4a286f6e16dd7fe70dc2cfe7a2e224f61906b5ed1`.
Source/launch-test SHA-256:
`8e6d471fb9c2f43ee886587aedf2aebf5eed7476e33829e84b8c3a289a0b9154`.
Both prior failure/diagnostic reports were rehashed unchanged; tracked worktree
and index were clean. GitHub branch readback matched the required parent.
