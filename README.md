# Lumi

Lumi is a from-scratch language-model research project by Afnan Technologies Inc.

The first target is a dense, decoder-only ~4B-parameter base model that can be
post-trained into `Lumi-4B-Instruct`, served with an OpenAI-compatible API, and
quantized for Apple Silicon. This repository contains model code, configuration,
tests, and small smoke-training utilities. It intentionally does **not** contain
training data, secrets, or model checkpoints.

## Current phase

Phase 1 architecture correctness is complete. Phase 2 validates checkpoint,
tokenizer, data, and deployment portability before any expensive GPU run:

1. self-contained Hugging Face export and isolated-process loading;
2. vLLM Transformers-backend compatibility;
3. one frozen 65,536-token contract shared by the 150M pilot and 4B candidate;
4. a pinned 25MB tokenizer canary, a metadata-preserving 10MB provenance
   canary, and corrected 32K/65K tokenizer prototypes pass their local gates;
5. a contextual-PII-v4 1GB development sample passes all 23 component quotas,
   language/secret/provenance gates, near-deduplication, benchmark
   decontamination, full post-write validation, and manual PII snippet review;
6. disk-backed MinHash and byte-offset validation checkpoints pass regression
   coverage and a real 25MB, 23-source disconnect/resume canary;
7. a SHA-256-pinned Bengali Wikipedia Parquet source, category-scoped language
   policy, and 1MB real-network canary pass with retained article URLs;
8. the v0.2 full-size attempt exposed a single-anchor assumption in a multi-file
   component without spending GPU credit; v0.3 adds generic component-level
   continuation, per-file provenance, resumable exhaustion state, and mandatory
   full-quota capacity gates;
9. the resulting 6GB accepted-text sample is sealed at 1,128,089 documents and
   SHA-256 `80bea9f9634a8daca542b2ffd0aad1c8c994a7cbeef909a5e529b6272671102a`;
   resumed full validation and the cached validation check pass;
10. the full 1,128,089-document bakeoff selects 65K: it emits 4.39% fewer
    tokens overall than 32K and wins in all eight measured domains; a later
    Unicode 1–3 digit-segmentation candidate is also rejected because it emits
    23,750,193 extra tokens (+1.6088%), so the original 65K vocabulary remains
    frozen;
11. deterministic fixed-sequence packing, sealed-manifest source stratification,
    and checkpointed training pass interruption/resume tests; an 8,192-token
    real-corpus Apple-MPS smoke run saved at step 1 and resumed in a fresh
    process through step 2;
12. the 24-source 50M target is materialized at 49,999,872 tokens in 48
    hash-verified shards;
13. the content-pinned Lambda preflight and paid H100 gate pass: a fresh-process
    smoke resume advances through step 2, and the 150,625,536-parameter model
    completes 763 steps over 50,003,968 tokens at a measured mean of 117,888
    tokens/second; the final resumable checkpoint is hash-verified on the Mac;
14. a hash-ranked, 24-source model-weight-held-out set is sealed at 4,999,168
    tokens with zero ID/content/near-duplicate overlap; a forced-stop MPS run
    resumes and completes at loss 4.680812, perplexity 107.858, and 4,686.9
    tokens/second on the 16GB M5 Mac;
15. fixed generations remain repetitive and factually wrong, as expected from
    a 150M model trained on only 50M tokens;
16. a self-identifying Lumi MLX artifact passes exact config/tensor/RoPE gates,
    prompt-logit parity, and 128/128 cached greedy tokens; its 4-bit affine form
    passes a 98,256-prediction-token stratified quality gate, is 83.70% smaller,
    and reaches a measured 522.3 cached decode tokens/second on the 16GB M5 Mac;
17. the sealed 4-bit artifact serves through a loopback-first OpenAI-compatible
    chat-completions API and responsive local UI; a real M5 black-box gate passes
    packaged HTML/CSS/JavaScript, model discovery, JSON generation, SSE
    streaming, usage accounting, and terminal `[DONE]` handling;
18. an exact 1,050,981,888-parameter scaling candidate preserves the 4B model's
    36 layers and 4K context at half width; its fail-closed plan pins permanent
    held-out exclusion, four cumulative stages through 1.4B tokens, gradient
    checkpointing, cosine scheduling, bounded checkpoints, and measured-input
    cost projection without approving a paid GPU launch;
19. a reusable full-corpus source index reproduces 1,476,228,500 tokenizer
    tokens and leaves 1,472,219,479 held-out-safe eligible tokens; a 16,777,216-
    token canary and final 1,399,996,416-token uint16 artifact pass fresh-process
    resume, cached-resume, rank-prefix, shard-integrity, and zero-leakage gates.
    The final payload is 2,799,992,832 bytes in 42 shards; no GPU was launched;
20. a backward-compatible schema-v2 cloud preflight pins that exact manifest,
    selection proof/state, tokenizer, 1B config, and scaling contracts. A full
    local rehash passes; the Mac correctly fails CUDA and 32GiB host-memory
    admission floors;
21. the same no-skip preflight passes on a live Lambda 1x A100 40GB SXM4 in
    India before training. One optimizer step and a separate fresh-process
    resume advance the 1,050,981,888-parameter model through step 2 and 65,536
    tokens with finite loss/gradient norm, exact data-position recovery, and a
    23.744GiB measured peak CUDA reservation. The latest checkpoint is archived
    on the Mac with all 14 server-generated hashes verified, and the Lambda
    instance is terminated;
22. a separately authorized A100 run restores that exact checkpoint and reaches
    cumulative step 100 and 3,276,800 tokens. Loss falls from 11.335843 to
    7.690175, steps 10--99 sustain a 9,379.6 tokens/second median, and peak CUDA
    reservation remains 23.744GiB. Checkpoint 100 and its reports occupy about
    12GB on the Mac, all 21 server-generated hashes pass, and the second Lambda
    instance is also terminated;
23. a third authorized A100 run resumes that state and hard-stops at cumulative
    step 7,630 and 250,019,840 tokens. Frozen held-out loss improves from the
    150M baseline of 4.680812 to 3.265504 (perplexity 26.193), so the learning
    gate passes, while fixed greedy generations remain repetitive and wrong.
    Median steady-state throughput is 9,382.6 tokens/second and the training
    segment costs USD 14.723236 at USD 1.99/hour. The final 12,612,139,807-byte
    checkpoint and all run evidence are archived on the Mac; all 35 final
    manifest hashes pass independently, and the third Lambda instance is
    terminated;
24. the optional document-isolated packed-attention path passes 125 local and
    cloud tests, Ruff, and a real A100 FlashAttention 2.8.3 varlen gate. CUDA
    logits match the dense reference at 0.999982 cosine similarity and later-
    document leakage is exactly zero. In two fresh paired 100-step runs, the
    isolated path reaches 9,809.5 versus 9,239.5 steady tokens/second (+6.17%),
    uses identical peak reserved HBM, and ends at loss 7.576857 versus 7.690148.
    All 32 downloaded manifest entries pass independent hashes, evidence is
    archived on the Mac, and the Lambda instance is terminated. The result
    selects isolation for the next fresh scaling candidate but does not alter
    the existing separator-only 250M checkpoint or authorize another paid run.
25. the next fresh 1B run now has a fail-closed, planning-only orchestration
    contract: the old checkpoint cannot cross the packing boundary, isolated
    held-out evaluation is fingerprinted separately, step 1/2 fresh-process
    recovery and cumulative 100/7,630/42,725 boundaries are fixed, model-only
    evaluation snapshots survive trainer pruning, and the final server seal
    validates all metrics and hashes. Measured isolated throughput projects
    39.64 pure-training hours and USD 78.89 at the previously observed A100
    rate, or about USD 82.87 with a two-hour lifecycle allowance. The full test
    suite, Ruff, and shell syntax gates pass. The repository itself never
    authorizes a paid launch; every run requires a separate owner artifact.
26. the separately owner-authorized isolated run resumes its hash-bound step-100
    state and hard-stops exactly at cumulative step 42,725 and 1,400,012,800
    tokens on one Lambda H100. Isolated held-out loss falls from 7.794926 at
    step 100 to 3.266177 at step 7,630 and 2.596964 at the final checkpoint
    (perplexity 13.423); the final legacy-policy evaluation is 2.643961.
    Steps 101--42,725 sustain a 19,900.2 tokens/second median and USD 84.1241
    measured step-compute estimate at USD 4.29/hour. All 77 current server
    artifacts pass an independently downloaded post-seal manifest, the
    35.3GiB archive is retained on the Mac, and the H100 is terminated. The
    authorization still records `full_4b_training=false`.
27. the final local initialization/optimizer gate runs a 9,865,344-parameter,
    eight-layer proxy across four factorial variants and three paired seeds on
    Apple M5 MPS. Residual-scaled output projections with legacy AdamW improve
    median held-out loss from 9.079972 to 8.981004 (1.08997%) at a 0.37374%
    median-throughput cost and improve all three seeds. Both beta2=0.95 grouped-
    decay variants fail the quality conditions. A fresh 4B v0.2 candidate
    therefore selects residual-scaled initialization and retains legacy AdamW;
    all 136 sealed evidence hashes pass independently, while historical 150M/1B
    configs and checkpoints remain unchanged.
28. the production distributed-trainer plumbing now passes a real two-process
    M5/CPU gate for both DDP and FSDP `FULL_SHARD`. Rank-disjoint cached-epoch
    data order, global cursor accounting, DCP model shards, rank-local
    optimizer/RNG state, atomic manifests, pruning, and post-group full-model
    reconstruction are implemented. Corpus hashes are checked once per node,
    while every rank confirms the same manifest identity. After simulated
    partial checkpoints plus a crash between checkpoint publication and root
    state/metric commit, uninterrupted and resumed four-step runs have exact
    per-step quality metrics and exact final model hashes in both strategies.
    A one-step two-process CPU `torch.compile` compatibility canary also
    completes with a verified checkpoint. All 83 archived evidence hashes pass
    independently. CUDA/NCCL, BF16, FlashAttention, compiled CUDA throughput,
    and real 4B memory/throughput remain a separate owner-authorized smoke gate;
    no paid GPU or 4B run was started.
29. the separately owner-authorized real Lumi-4B CUDA smoke passes on Lambda
    Cloud `2x H100 (80 GB SXM5)`. Exact commit `208c7ed` runs the
    4,002,376,704-parameter model with FSDP `FULL_SHARD`, NCCL, BF16, and
    document-isolated FlashAttention through fresh step 1 and a new-process
    resume to step 2, hard-stopping at 16,384 tokens. The two optimizer steps
    reach 7,824.7 and 10,861.3 tokens/second; peak observed HBM is 47,531 and
    47,959 MiB. The final six-entry distributed checkpoint is fully rehashed,
    its manifest SHA-256 is `68c2857239eefcb75aac7ffff96b55ca32719c963666c7fb290475e9be1fcbaa`,
    and the canonical model digest is
    `707a54af49721e21fa35c6de81d33bdc7bc0e14f359ce443e662e0e462d169f9`.
    All 24 downloaded evidence-manifest entries pass independently on the Mac,
    and the instance is terminated. This proves the distributed CUDA and
    checkpoint/resume plumbing, not model quality, and still records
    `full_4b_training=false`.
30. the planning-only 80B corpus source/capacity gate pins official metadata for
    all 23 selected filtered Common Pile repositories. Their exact revisions
    expose 188,143,662 published documents, 1.735TB of UTF-8 text, and 499.82GB
    of compressed payload. Category-specific fertility from the frozen Lumi
    tokenizer estimates 405.37B published tokens; every category clears its
    1.5-times candidate floor, although open web has only 11.62% headroom. This
    initially permits bounded source canaries only: post-filter unique capacity,
    complete license-value audit, object storage, bulk materialization, paid
    compute, and 4B training remain blocked.
31. the bounded, text-free retention canary then streams 192,528,033 raw UTF-8
    bytes from the three tightest categories through the frozen tokenizer and
    production quality gates. It accepts 190,915,171 bytes and 45,981,609
    tokens without persisting corpus text. Measured projections clear every
    floor: open web 13.210B versus 12B, questions/discussion 21.863B versus 18B,
    and reference 25.153B versus 18B. The 194,260,992-byte SQLite evidence,
    report, source/config/code fingerprints, dedup indexes, and authorization
    boundaries pass independent verification. This approves object-storage and
    adaptive-acquisition design only; bulk data, paid compute, and training are
    still not authorized.
32. the provider-neutral object-storage and adaptive-acquisition design passes
    deterministic local planning and independent reproduction. Acquisition now
    stops when all seven category floors total 120B accepted candidate tokens;
    150B is only a hard safety maximum, not a fixed download. Content-addressed
    immutable manifests, atomic resumable shard publication, 20GB worker
    scratch caps, a 4,096-partition removal ledger, and fail-closed authorization
    boundaries are specified. The forecast peak is 1.0375TB of future cloud
    logical storage under a 2TB hard cap, while the local evidence is only
    16,750 bytes. No provider, price, bucket, bulk transfer, paid compute, or
    training is authorized.
33. AWS S3 in `us-east-1` is then selected with exact public Price List hashes,
    a conservative $25.8025 monthly projection, and a $100 S3-only cap. A live
    17,825,915-byte synthetic canary proves fully private encrypted/versioned
    buckets, access logging, conditional-create rejection, a three-part
    fresh-process multipart resume, full SHA-256 and range-GET parity, and
    complete version-aware deletion. Both temporary buckets and the unusable
    temporary IAM role are confirmed absent afterward. The estimated canary
    charge is at most $0.003586. This selects and validates storage only; bulk
    corpus materialization, paid compute, and training remain unapproved.
34. the metadata-only source-file gate seals all 758 data objects across all 23
    pinned Common Pile repositories: exact paths and revisions, 499,817,285,384
    compressed bytes, Git blob IDs, Git LFS SHA-256 identities, optional Xet
    hashes, and 46 repository control files. Every count and byte aggregate
    matches the prior capacity snapshot, all 758 LFS digests are unique, and no
    corpus payload is downloaded. The resumable SQLite state, flattened JSONL,
    repository snapshot, report, and SHA manifest pass independent
    verification. CCCC remains blocked pending the owner's license decision;
    bulk acquisition, cloud resources, paid compute, and training remain
    unapproved.
35. the CCCC owner-decision gate pins the exact 4,060-byte upstream card, its
    Git blob and SHA-256, the source-level domain-audit claims and caveats, the
    0-of-10,000 local license-field mismatch, 53 source objects, and the
    measured open-web capacity consequence. It removes the automatic
    `approved_source_level_basis` shortcut: CCCC now fails closed until the
    owner explicitly chooses `include_with_caveat` or `exclude_and_replace`.
    The owner selected `include_with_caveat` for the stated personal/research
    scope with traceability and removal controls. This resolves only the CCCC
    license-basis blocker; no data transfer or training is authorized.
36. the first owner-authorized production acquisition is stopped safely after
    preserving 237,769,640 accepted tokens and 38,349 documents. Its two sealed
    shards, partial current shard, eight-file manual-stop archive, SQLite state,
    38,349 immutable dedup anchors, hashes, and source-line resume boundary pass
    independent verification; the EC2 instance and both volumes are deleted.
    A replacement AWS Glue/Spark backend now matches the original sequential
    source-order exact/near-dedup result in real local Spark tests, repeats
    deterministically, and fails closed on unsafe conflict sizes or checkpoint
    locations. The release package is non-authorizing. The next possible paid
    action is a separately approved 8-DPU, 30-minute parity canary capped at
    $1.76. That exact owner-authorized canary then fails safe during dependency
    bootstrap because Glue 5.1's fixed Python 3.11 cannot install Lumi's pinned
    Lingua 2.2.0 wheel. It reads no source records, writes no canary or
    production objects, starts no model training, costs an estimated $0.0374,
    and its Glue job and IAM role are deleted.
37. the bounded parity gate is therefore adapted locally—not launched—to EMR
    Serverless 7.13.0 / Spark 3.5.6 with a digest-pinned x86 image and Python
    3.12. The fully hash-pinned image context, 16-vCPU/64GB/30-minute application
    ceiling, eight-source/all-category canary, immutable-prefix anchors,
    source-order Spark/sequential parity harness, request renderer, and separate
    release/authorization checks pass local verification. The maximum EMR plus
    CodeBuild compute envelope is $0.755904. The old Glue approval cannot
    authorize it. Its first exact CodeBuild authorization is consumed by one
    attempt that fails in `PRE_BUILD` before Docker because the S3 source used
    the wrong built-in version variable. No image or EMR job is created, the
    conservative compute upper bound is $0.03, and the project, log group,
    temporary roles, empty ECR repository, and live authorization are deleted.
    The corrected buildspec uses `CODEBUILD_SOURCE_VERSION` and is now included
    in the release-fingerprinted image context; a retry needs new approval.
38. a fresh exact authorization then produces one successful corrected
    CodeBuild image and one bounded EMR job. The 2,631,167,478-byte image passes
    its runtime smoke, resolves by immutable ECR digest, and starts Spark plus
    all three executors. Initial anchor/shuffle stages pass, but source tasks
    cannot reach `huggingface.co:443` because EMR Serverless blocks outbound
    internet by default. The run is cancelled safely after 273 seconds instead
    of exhausting its curl retries; it uses 1.181 vCPU-hours and 4.724
    memory-GB-hours, with combined build/EMR compute estimated at $0.139446578.
    No parity report, production shard, or model training is produced. All
    temporary resources are deleted or terminated. The next revision must stage
    pinned inputs in S3 and keep EMR on same-Region S3 I/O.
39. that S3-only revision is now implemented and verified locally. Its exact
    eight-source plan covers all seven categories and 1,456,514,440 compressed
    bytes. Content-addressed conditional writes, stored SHA-256 checksums,
    wrong-object refusal, interrupted range resume, and a no-HTTP-fallback EMR
    source path pass 79 corpus-focused tests including real Spark/sequential
    parity; the complete repository suite passes 239/239. The eight downloads
    are now folded into the same bounded 30-minute CodeBuild image build, with
    two workers and a verified 64GB-disk/4GiB-memory envelope. Its first exact
    source-staging authorization is consumed by one CodeBuild attempt that
    fails safe before source download: S3 returns `403` for an absent
    `HeadObject` because the temporary role lacks `s3:ListBucket`. No raw
    source, image, EMR job, production shard, or model training is created;
    compute is bounded above by $0.03 and every temporary resource is deleted.
    The role renderer now includes the required read-only bucket permission and
    a regression test. A retry remains unapproved; see
    [`docs/s3-staged-corpus-v0.1.md`](docs/s3-staged-corpus-v0.1.md).
40. the exact eight-source transport and corpus path now also passes a bounded
    real-source Mac canary. All 1,456,514,440 compressed bytes were downloaded
    once and independently hash-verified on resume. The production tokenizer,
    filters, benchmark/held-out exclusions, and secret/PII handling reduced
    14,061 raw records to 13,027 candidates carrying 39,196,478 tokens. Against
    38,349 immutable resume anchors, localhost Spark and the independent
    sequential reference produced the same 12,999 accepted IDs and 28
    near-duplicate rejections. The sealed parity digest is
    `13edb46411c3b5eeaaf35237f70e379d72eb22b8a223c0d70586ece649c96837`.
    No AWS resource, GPU, production shard, or model training was started.
41. one fresh bounded cloud authorization then closes the S3/EMR parity gate.
    CodeBuild stages and independently verifies all eight objects, builds the
    2,631,167,482-byte immutable image, and succeeds without retry. The single
    EMR job processes 13,027 candidates / 39,196,478 tokens and exactly matches
    the Mac reference: 12,999 accepted IDs, 28 near-duplicate rejections, and
    parity SHA-256
    `13edb46411c3b5eeaaf35237f70e379d72eb22b8a223c0d70586ece649c96837`.
    It runs for 361 seconds using 1.57 vCPU-hours and 6.28 memory-GB-hours;
    combined build/EMR compute is estimated at $0.173908660. No production
    shard or model training is created, and all temporary compute, IAM, ECR,
    raw, checkpoint, result, and control versions are removed after the
    72-file evidence archive passes independent verification.
42. the production resume control plane is now implemented locally without
    launching paid resources. It resumes the verified 237,769,640-token prefix,
    uses one internet-facing CPU stager and private S3-only EMR input, maps seven
    category lanes concurrently at 112 vCPU, runs one 128-vCPU global dedup,
    writes seven categories in parallel, and seals a conservation-checked 120B
    candidate manifest. Adaptive shortfalls resume at exact category cursors and
    fail before the 150B/2TB ceilings. Actual billed usage plus active worst-case
    reservations are held inside a $2,000 total budget. The complete repository
    suite passes 278/278 and lint is clean. AWS quota and a new exact paid
    authorization are still required; no production job, GPU, or training is
    running. See
    [`docs/production-emr-resume-v0.1.md`](docs/production-emr-resume-v0.1.md).
43. the final-width native MLX portability gate passes locally on the 16GB M5
    Mac without cloud compute. A 306,914,304-parameter fixture preserves the
    final 4B model's 3,072 hidden width, 8,384 MLP width, 64 query heads, eight
    key/value heads, 64-wide head dimension, 65,536-token vocabulary, and 4,096
    context contract while reducing only depth from 36 layers to one. Config
    and saved tensors are exact, RoPE error is 4.768e-7, prompt top-1 agreement
    is 60/60, and cached greedy generation is 16/16 exact. The self-contained
    manifest and copied implementation provenance pass independent
    verification. This proves final projection-layout portability, not trained
    4B quality, full-model memory, quantization, or speed; vLLM CUDA and GGUF
    remain pending. See
    [`docs/mlx-final-topology-gate-v0.1.md`](docs/mlx-final-topology-gate-v0.1.md).
44. one bounded Glue compatibility follow-up builds Lumi's exact Lingua 2.2.0
    CPython 3.11 wheel on Linux. Upstream tests pass 74/74, and all 222,790
    language-parity records reproduce both sealed decision hashes. The
    170,336,000-byte manylinux wheel is independently verified and pinned by
    SHA-256 plus S3 VersionId. The single follow-up 8-DPU Glue run completes
    custom-module setup, then fails safe before Spark or data processing because
    the script redeclares Glue's reserved `JOB_RUN_ID`; it uses 493 DPU-seconds,
    costs an estimated $0.0603, and writes zero release-specific corpus objects.
    Combined conservative CodeBuild/Glue compute is no more than $0.2203. All
    temporary projects, jobs, roles, and exact log streams are deleted. The
    reserved-argument fix passes the 14-test focused Glue/Spark suite locally,
    but the consumed authorization permits no retry or production run. See
    [`docs/distributed-corpus-glue-v0.1.md`](docs/distributed-corpus-glue-v0.1.md).
45. one separately authorized corrected Glue run uses that exact pinned wheel
    and source commit `acef08b` on 8 x `G.1X`. It succeeds end to end over
    eight real source files: 13,027 candidates and 39,196,478 candidate tokens
    are processed in 399.355 in-script seconds, 12,999 documents are accepted,
    28 near duplicates are rejected, and Spark accepted IDs plus rejection
    reasons exactly match the sequential reference. Total execution is 476
    seconds and 3,808 DPU-seconds, an estimated $0.465422. The report hash and
    13-entry local evidence manifest pass independently; the temporary
    checkpoint, job, IAM role, and nine exact log streams are deleted.
    Production shards remain false and no production acquisition is authorized.
    See
    [`docs/distributed-corpus-glue-v0.1.md`](docs/distributed-corpus-glue-v0.1.md).
46. the bounded production staging phase uses the live 100-DPU Glue ETL ceiling
    to stage all 758 pinned source files in 474 seconds. It publishes 14,658
    immutable zstd chunks containing 168,682,770 rows and 500,163,871,471
    compressed bytes. The successful run uses 47,419 DPU-seconds, while all
    production staging attempts together cost an estimated $7.093656. A
    separate read-only audit verifies every manifest and chunk VersionId,
    length, SHA-256, encryption, source digest, and line interval. The driver
    result and audit report hashes pass; model training remains false. See
    [`docs/distributed-corpus-glue-v0.1.md`](docs/distributed-corpus-glue-v0.1.md).
47. the next production mapping gate is fixed at 512 DPU and a hard
    150,000,000,000 total candidate-token ceiling. The unlaunched harness uses
    256 x `G.2X`, immutable 2,000-chunk waves, exact checkpoint-bound resume,
    a 660-minute soft stop, a 720-minute timeout, and a $2,703.36 maximum Glue
    compute envelope. Local package and launcher dry-runs pass without creating
    a cloud job. AWS Service Quotas reports 1,000 DPU, but Glue's live executor
    still rejects requests over 100 DPU, so no 100-DPU production fallback or
    paid mapping run has been started. See
    [`docs/distributed-corpus-glue-v0.1.md`](docs/distributed-corpus-glue-v0.1.md).
48. after AWS continued enforcing a live 100-DPU ceiling, the owner explicitly
    replaced the unlaunched 512-DPU proposal with one 100-DPU production mapping
    run. The current envelope is 100 x `G.1X`, an unchanged 150B total-token hard
    stop, 500-chunk immutable checkpoints, a 3,960-minute soft stop, a
    4,320-minute hard timeout, and a $3,168 maximum. Deduplication, sharding,
    GPU compute, and model training remain separately unauthorized.
49. the first authorized 100-DPU start failed closed after zero execution
    seconds with no reported DPU-seconds because the newly created IAM role had
    not yet propagated to the Glue executor. The live trust policy is correct,
    no mapping or model training started, and the exact failed-start evidence is
    sealed locally. The launcher now waits 60 seconds after creating a role and
    gives each separately authorized retry a distinct job identity; a
    replacement run then passed role assumption.
50. that separately authorized replacement failed closed during dependency
    bootstrap, before mapping, because the launcher omitted the pinned Lingua
    CPython 3.11 wheel from the role's exact S3-read list. It used 3,676
    DPU-seconds—approximately $0.449289 at $0.44/DPU-hour—and produced no mapped
    candidates or model-training work. The failure is sealed locally. The
    launcher now grants the wheel exact read access without duplicating it in
    `--extra-files`, with a regression test; no further paid run is authorized.
51. the next separately authorized attempt passed IAM propagation, installed
    the exact pinned Lingua wheel, and initialized Spark, proving both previous
    startup fixes. It then failed closed before mapping because the runner did
    not reattach the already verified staged-manifest digest to the validator
    wrapper. It used 7,521 DPU-seconds—approximately $0.919233—and wrote zero
    checkpoint, candidate, or run-manifest objects. The exact attempt is sealed
    locally. The digest handoff now has a behavioral regression test; another
    paid run requires another exact owner approval.
52. the owner-authorized 100-DPU mapping execution now passes after a
    checkpoint-preserving correction. Waves 1--7 were recovered exactly once
    from the first data-producing run, and the corrected continuation completed
    Waves 8--12 with Glue state `SUCCEEDED`. The sealed pool contains
    149,999,976,054 total candidate tokens across 89,481,434 mapped documents
    and 629,842,275,871 Parquet bytes. All seven categories are closed; the
    23,946-token difference from 150B is the bounded document-boundary
    shortfall. The final checkpoint
    `90ccdacf5b8bff23c87061ec68bbce0d0994f811cdfdccaf3abd09d0d01be59b`
    independently replays from sequence 11, and the final report and Wave-12
    object inventory match. No model training started. The consumed job and
    dedicated IAM role are deleted; global deduplication remains the next
    separately authorized production phase.
53. the local-only Glue global-dedup adapter now binds that exact sequence-12
    checkpoint to all 12 immutable wave manifests and 56 bounded Parquet input
    locations. It conserves 89,481,434 documents, 149,762,206,414 newly mapped
    tokens, 629,842,275,871 Parquet bytes, and the 237,769,640-token accepted
    prefix; the sealed input-index SHA-256 is
    `2cd4537a86d6e48e2fa079c1088fef167d856229e208a45284fab216f207d266`.
    Historical Waves 1--7 and corrected Waves 8--12 are migration-aware and
    cannot be omitted, reordered, duplicated, or moved outside their exact S3
    namespaces. Canary and production authorizations are separate: the next
    possible paid action is an 8-DPU/30-minute structural canary capped at
    $1.76; a later 100-DPU/24-hour production dedup would require a new exact
    approval and is capped at $1,056. Full verification passes 312 tests and
    Ruff. No paid dedup, sharding, GPU, or model training was started.
54. the first exact 8-DPU global-dedup canary was consumed and failed closed
    before reading anchors or candidates. Glue 5.1 reported its Spark build as
    `3.5.6-amzn-1`, while the runtime contract required the literal `3.5.6`.
    The run used 671 DPU-seconds (about $0.082011 at $0.44/DPU-hour); independent
    S3 checks found zero checkpoint, canary, result, or production objects. Its
    job and dedicated IAM role are deleted. The local correction accepts only
    the release-pinned Spark base version with an optional numeric AWS
    `-amzn-N` suffix and rejects base-version or vendor drift. A corrected paid
    retry requires a new exact owner authorization. The complete suite passes
    313/313 tests and Ruff lint.

The 4B configuration is a candidate until all deployment gates pass. The 1B
design, hash-ranked held-out-excluding artifact, A100 gates, document-isolation
A/B, isolated 1.4B-token H100 run, final tokenizer gate, and local
initialization/optimizer gate, and bounded two-step 4B CUDA smoke are complete.
The bounded corpus-retention canary, object-storage/acquisition design, AWS S3
provider pricing, live bounded I/O canary, preserved production prefix, and
local distributed-acquisition parity gate, and local S3-only source transport
gate—including the exact real-source Mac canary—are complete. The complete
source file manifest, fail-closed CCCC decision, and bounded S3-staged EMR
parity canary are also complete. The distributed production-resume harness,
corrected Glue runtime/parity canary, and complete 758-source production staging
phase now pass. Adaptive candidate mapping also passes with a sealed
149,999,976,054-token pool, and its checkpoint-bound global-dedup package is
locally verified. The first paid dedup canary failed safe on an over-strict AWS
Spark build suffix check; a corrected canary, production global dedup, final
80B selection and shard sealing, a 4B learning pilot, and the full 4B/80B
pretraining run remain. No model training is authorized by this preparation.
The completed 150M and 1B pilot checkpoints are not production-quality models.

## Quick start

Use Python 3.12. The quality-gated data pipeline pins Lingua 2.2.0, whose wheel
requires Python 3.12 or newer; using one interpreter and detector version keeps
language decisions reproducible.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,data]'
pytest
python scripts/count_parameters.py configs/lumi-4b.json
python scripts/smoke_train.py --steps 20
python scripts/check_isolated_hf.py
```

The real-data packer and resumable pilot runner are documented in
[`docs/real-data-pilot-v0.1.md`](docs/real-data-pilot-v0.1.md). Keep their
generated shards and checkpoints outside Git.
The paid CUDA gate and exact transfer/training contract are documented in
[`docs/lambda-pilot-v0.1.md`](docs/lambda-pilot-v0.1.md).
The held-out construction, MPS evaluation, and quality decision are documented
in [`docs/evaluation-pilot-v0.1.md`](docs/evaluation-pilot-v0.1.md).
The native MLX conversion, FP32 parity, 4-bit quality, and M5 runtime results are
documented in
[`docs/local-inference-gate-v0.1.md`](docs/local-inference-gate-v0.1.md).
The reduced-depth final-width 4B projection-shape conversion, strict FP32
parity, copied implementation provenance, and M5 result are documented in
[`docs/mlx-final-topology-gate-v0.1.md`](docs/mlx-final-topology-gate-v0.1.md).
The loopback OpenAI-compatible API, local chat UI, security boundary, and real
server gate are documented in
[`docs/local-serving-v0.1.md`](docs/local-serving-v0.1.md).
The exact 1B architecture, staged token contract, trainer changes, input audit,
measured-throughput cost rule, and verified 250M-token learning gate are
documented in
[`docs/scaling-pilot-1b-v0.1.md`](docs/scaling-pilot-1b-v0.1.md).
The position-reset/varlen document-isolation policy, resume boundary, completed
A100 CUDA A/B result, and evidence contract are documented in
[`docs/packed-attention-ab-v0.1.md`](docs/packed-attention-ab-v0.1.md).
The fresh step-zero 1B/1.4B isolated-run contract, corrected evaluation
semantics, measured cost projection, authorization boundary, disconnect-safe
runner, and final evidence seal are documented in
[`docs/isolated-scaling-run-1b-v0.1.md`](docs/isolated-scaling-run-1b-v0.1.md).
The deterministic source index, 16M canary, final aligned 1.4B pack, resume
boundary, and verified hashes are documented in
[`docs/scaling-data-1b-v0.1.md`](docs/scaling-data-1b-v0.1.md).
The exact 1B transfer bundle, runtime/storage floors, local preflight result,
remote CUDA gate, smoke/resume boundary, and measured 100-step calibration are
documented in
[`docs/cloud-pilot-1b-v0.1.md`](docs/cloud-pilot-1b-v0.1.md).
The paired-seed residual-initialization and AdamW-profile experiment, resume
canary, conservative decision rule, and sealed M5 evidence are documented in
[`docs/init-optimizer-ab-v0.1.md`](docs/init-optimizer-ab-v0.1.md).
The DDP/FSDP topology, cached global data cursor, distributed checkpoint/resume
format, local interruption gate, safety caps, and remaining CUDA boundary are
documented in
[`docs/distributed-trainer-v0.1.md`](docs/distributed-trainer-v0.1.md).
The exact two-GPU Lumi-4B smoke fixture, source-commit-bound planning flow,
two-step resume boundary, server seal, and verified Lambda H100 result are
documented in
[`docs/distributed-cuda-smoke-v0.1.md`](docs/distributed-cuda-smoke-v0.1.md).
The pinned upstream storage inventory, frozen-tokenizer capacity estimate,
per-category candidate floors, and fail-closed acquisition boundary are
documented in
[`docs/corpus-capacity-gate-v0.1.md`](docs/corpus-capacity-gate-v0.1.md).
The bounded text-free retention run, measured tight-category projections,
sealed local evidence, and next storage boundary are documented in
[`docs/corpus-retention-canary-v0.1.md`](docs/corpus-retention-canary-v0.1.md).
The adaptive 120B-floor acquisition rule, provider-neutral object contract,
storage forecast, removal ledger, resumable state, and unchanged authorization
boundary are documented in
[`docs/corpus-object-storage-v0.1.md`](docs/corpus-object-storage-v0.1.md).
The pinned AWS S3 rates, conservative cost projection, security contract,
fresh-process multipart canary, cleanup evidence, and unchanged no-bulk/no-GPU
boundary are documented in
[`docs/aws-s3-provider-canary-v0.1.md`](docs/aws-s3-provider-canary-v0.1.md).
The exact 23-repository file inventory, upstream object hashes, resumable
metadata state, independent seal, and unchanged no-payload/no-GPU boundary are
documented in
[`docs/source-file-manifest-v0.1.md`](docs/source-file-manifest-v0.1.md).
The pinned CCCC evidence, include/exclude consequences, explicit owner record,
and fail-closed source-manifest policy are documented in
[`docs/cccc-owner-decision-v0.1.md`](docs/cccc-owner-decision-v0.1.md).
The preserved 237.77M-token prefix, immutable resume anchors, ordered parallel
deduplication proof, real local Spark parity gate, fail-closed limits, the
two historical failed Glue gates, verified CPython 3.11 wheel, and successful
corrected 8-DPU parity result are documented in
[`docs/distributed-corpus-glue-v0.1.md`](docs/distributed-corpus-glue-v0.1.md).
The digest-pinned Python 3.12 replacement, exact EMR Serverless capacity/cost
envelope, successful image build, external-egress diagnosis, cleanup, and
S3-staged next boundary are documented in
[`docs/distributed-corpus-emr-serverless-v0.1.md`](docs/distributed-corpus-emr-serverless-v0.1.md).
The immutable S3 source binding, bounded same-build producer, resumable S3-only
consumer, 79-test focused result, 239-test full-suite result, and unchanged
no-cloud/no-training boundary are documented in
[`docs/s3-staged-corpus-v0.1.md`](docs/s3-staged-corpus-v0.1.md).

## Repository policy

- Never commit `.env` files, API keys, datasets, checkpoints, or generated
  weights.
- Keep source provenance and dataset manifests outside raw data storage.
- A green tiny-model test suite is required before scaling.
- A measured single-GPU throughput pilot is required before approving the 4B
  run.

See [`docs/architecture-v0.1.md`](docs/architecture-v0.1.md),
[`docs/portability-v0.1.md`](docs/portability-v0.1.md), and
[`docs/tokenizer-v0.1.md`](docs/tokenizer-v0.1.md) for the current decision
records. The candidate source mix and acquisition gates are in
[`docs/data-plan-v0.1.md`](docs/data-plan-v0.1.md), with the first measured run
in [`docs/data-canary-v0.1.md`](docs/data-canary-v0.1.md).
The metadata-preserving follow-up is in
[`docs/provenance-canary-v0.2.md`](docs/provenance-canary-v0.2.md).
The current audited all-component result is in
[`docs/development-sample-v0.4.md`](docs/development-sample-v0.4.md).
The current scale-up gates are in
[`docs/tokenizer-sample-v0.3.md`](docs/tokenizer-sample-v0.3.md), and the pinned
Bengali source decision is in
[`docs/multilingual-source-v0.1.md`](docs/multilingual-source-v0.1.md).
