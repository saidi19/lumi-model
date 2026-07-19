# Lumi CCCC owner-decision gate v0.1

Status: **evidence verified; owner choice pending; CCCC acquisition remains
blocked; no corpus payload, cloud resource, paid compute, or training is
authorized**

## Why this gate exists

CCCC is the only source currently assigned to Lumi's open-web category. It is
large enough to satisfy that category's planned token floor, but its filtered
rows do not contain the per-document `metadata.license` field claimed by the
dataset card. Treating the phrase `source_level_basis` as automatic approval
would therefore turn an unresolved owner choice into an accidental license
decision.

The contract in
[`cccc-owner-decision-v0.1.json`](../configs/cccc-owner-decision-v0.1.json)
separates three things:

1. reproducible facts from the pinned source and Lumi's local audits;
2. the operational consequences of include versus exclude; and
3. the owner's explicit decision and acknowledgements.

It is an engineering record, not legal advice.

## Verified evidence

| Evidence | Verified result |
|---|---|
| Pinned filtered repository | `common-pile/cccc_filtered` at `03a3de5713a0bb23267d26724346508af0f25327` |
| Dataset card | 4,060 bytes; SHA-256 `13b727dd5bc1045a502705fa70ada1254586fa19b7ddc2fd987dec89763422ad` |
| Dataset-card Git blob | `5c632ee7dcdd38e6834a731968bddcc6dbea9b62` |
| Exact source objects | 53 files; 21,497,337,171 compressed bytes |
| Corpus payload downloaded by this gate | 0 bytes |
| Local license-field audit | 0 of 10,000 audited raw/filtered rows contained the claimed field |
| Bounded provenance audit | 2,000 CCCC rows had ID, source-level basis, provenance, and URL |

The pinned [CCCC filtered card](https://huggingface.co/datasets/common-pile/cccc_filtered/blob/03a3de5713a0bb23267d26724346508af0f25327/README.md)
says that the authors reviewed the 1,000 largest domains and retained 537 where
the Creative Commons designation applied to all text. It also claims that every
example has a metadata license field, warns that license laundering or
inaccurate metadata can still cause errors, and contains an unresolved `todo`
placeholder in the sentence reporting the second annotation pass.

The [Common Pile paper](https://arxiv.org/abs/2506.05209) explains that a
collection-level license does not license the underlying documents, that
license laundering cannot be eliminated exhaustively, and that a site's
license can change after collection. The paper separately documents the
domain-level CCCC review. These facts support a real source-level basis, but
they do not create missing per-document evidence or remove the residual risk.

## Exact consequences

| Choice | Immediate result | Open-web consequence |
|---|---|---|
| `include_with_caveat` | Resolve the CCCC blocker as `owner_accepted_source_level_domain_audit`; production acquisition still needs separate approval | Measured projection is 13,210,015,719 candidate tokens against the 12B floor: 1,210,015,719 tokens or 10.083% headroom |
| `exclude_and_replace` | Keep all 53 CCCC files blocked and exclude them from acquisition | The current category has no source; a pinned, audited, measured replacement must supply the full 12B candidate floor |

Including with a caveat requires retaining the exact revision and object hashes,
source ID, source label, provenance, URL and WARC URL where present, the
source-level basis, and a permanent removal ledger. The pipeline must not
invent a per-document license value.

Excluding does not permit silently repeating another category or changing the
80B mixture. Replacement sources must pass the same provenance, retention,
deduplication, contamination, PII, and source-file gates first.

## Engineering recommendation

For the currently stated personal/research Lumi v0.1 scope, the engineering
recommendation is **`include_with_caveat`**, because the source is pinned, the
upstream domain audit is documented, traceability/removal controls are already
part of the design, and exclusion creates a measured 12B-token hole. This is
still an owner risk decision, not a claim that every document has been legally
adjudicated. Any later commercial release should trigger a fresh licensing and
attribution review.

No option is preselected in Git. The decision remains
`pending_owner_decision` until the owner explicitly chooses one option and all
required acknowledgements are recorded.

The independently reproduced pending packet is stored outside Git at
`lumi-data/cccc-owner-decision-v0.1/`:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `report.json` | 5,075 | `f738452331c90da3faf8e5b58182064cf3fd100b94dd7c3e467e7344a5b27c46` |
| `manifest.sha256` | 78 | `1361d78e56bf7d0e87851ad6ffa5cb988217c1799385ffeae2fb03c0319d35df` |

The pending decision fingerprint is
`9df98b984feafa79b6f3892b33d3f426242aae2ff5f05ec52c173c424f449a40`.

## Fail-closed behavior

The corpus-storage manifest validator now rejects CCCC while the decision is
pending. It also rejects CCCC if the owner selects `exclude_and_replace`.
Only a complete `include_with_caveat` record permits the distinct
`owner_accepted_source_level_domain_audit` policy; the older automatic
`approved_source_level_basis` shortcut is no longer accepted.

Changing the dataset ID, revision, card hash/blob, file count, compressed byte
total, local audit warning, provenance row, mixture, retention result,
acknowledgement set, or authorization boundary fails the gate.

```bash
PYTHONPATH=src python scripts/plan_cccc_owner_decision.py \
  --output /tmp/lumi-cccc-decision/report.json \
  --manifest /tmp/lumi-cccc-decision/manifest.sha256

PYTHONPATH=src python scripts/verify_cccc_owner_decision.py \
  --report /tmp/lumi-cccc-decision/report.json \
  --manifest /tmp/lumi-cccc-decision/manifest.sha256
```

## Next boundary

After the owner choice is recorded and independently verified:

- include: request separate approval for the production S3 namespace and
  adaptive acquisition; or
- exclude: pin, audit, and measure replacement open-web source(s).

Neither path starts corpus transfer or 4B training automatically.
