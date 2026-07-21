"""Deterministic distributed corpus primitives for Lumi.

The production corpus policy is unchanged: the same pinned sources, filters,
tokenizer, exclusion ledger, category floors, and storage caps apply.  This
module only separates expensive stateless document work from global duplicate
selection so source files can be mapped in parallel without changing the
single-writer acceptance order.

Near-duplicate selection uses a fixed-priority parallel greedy independent-set
algorithm.  Edges are oriented by the canonical source order.  In each round,
all active vertices without an earlier active neighbour are accepted and their
later neighbours are rejected.  This is equivalent to the sequential greedy
decision used by the v0.1 single-writer index, including non-transitive chains.
"""

from __future__ import annotations

import hashlib
import json
import sys
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tokenizers import Tokenizer

from .data_pipeline import (
    PII_DETECTOR_VERSION,
    _extract_provenance,
    _record_id,
    filter_document,
    split_utf8,
)
from .data_quality import DataQualityGate
from .production_corpus import _canonical_json_bytes, _read_json, _resolve, _sha256_file

DISTRIBUTED_CORPUS_SCHEMA_VERSION = 1
CANDIDATE_SCHEMA_VERSION = 1
DEDUP_ANCHOR_SCHEMA_VERSION = 1
GLUE_DPU_VCPUS = 4
GLUE_DPU_MEMORY_GIB = 16


def _valid_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _valid_source_commit(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("distributed corpus source commit must be a full Git SHA-1")
    return value


def _order_tuple(candidate: Mapping[str, Any]) -> tuple[int, int, int, str]:
    return (
        int(candidate["source_ordinal"]),
        int(candidate["source_line"]),
        int(candidate["segment_index"]),
        str(candidate["document_id"]),
    )


def _token_bytes(token_ids: Sequence[int]) -> bytes:
    if any(token < 0 or token > 65_535 for token in token_ids):
        raise ValueError("distributed candidate contains a non-uint16 token")
    values = array("H", token_ids)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


@dataclass(frozen=True)
class MinHashCodec:
    """Canonical MinHash serialization and the pinned datasketch LSH layout."""

    num_perm: int
    scheme: str
    threshold: float
    hashranges: tuple[tuple[int, int], ...]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> MinHashCodec:
        from datasketch import MinHashLSH

        num_perm = int(config["num_perm"])
        scheme = str(config["scheme"])
        threshold = float(config["threshold"])
        if scheme not in {"affine32", "affine64"}:
            raise ValueError("unsupported distributed MinHash scheme")
        if num_perm < 32:
            raise ValueError("distributed MinHash requires at least 32 permutations")
        layout = MinHashLSH(threshold=threshold, num_perm=num_perm)
        return cls(
            num_perm=num_perm,
            scheme=scheme,
            threshold=threshold,
            hashranges=tuple(tuple(item) for item in layout.hashranges),
        )

    @property
    def width(self) -> int:
        return 4 if self.scheme == "affine32" else 8

    def signature_bytes(self, signature: Any) -> bytes:
        values = [int(value) for value in signature.hashvalues]
        if len(values) != self.num_perm:
            raise ValueError("distributed MinHash signature length changed")
        return b"".join(value.to_bytes(self.width, "big") for value in values)

    def fingerprint(self, signature_bytes: bytes) -> str:
        self.validate_signature(signature_bytes)
        return hashlib.sha256(signature_bytes).hexdigest()

    def buckets(self, signature: Any) -> tuple[dict[str, Any], ...]:
        values = [int(value) for value in signature.hashvalues]
        if len(values) != self.num_perm:
            raise ValueError("distributed MinHash signature length changed")
        return tuple(
            {
                "band": band,
                "bucket": b"".join(
                    value.to_bytes(self.width, "big") for value in values[start:end]
                ),
            }
            for band, (start, end) in enumerate(self.hashranges)
        )

    def validate_signature(self, value: bytes) -> None:
        if len(value) != self.num_perm * self.width:
            raise ValueError("distributed MinHash signature has an invalid byte length")

    def estimated_jaccard(self, left: bytes, right: bytes) -> float:
        self.validate_signature(left)
        self.validate_signature(right)
        matches = sum(
            left[offset : offset + self.width] == right[offset : offset + self.width]
            for offset in range(0, len(left), self.width)
        )
        return matches / self.num_perm


@dataclass(frozen=True)
class MappedBatch:
    candidates: tuple[dict[str, Any], ...]
    rejections: dict[str, int]
    pii_redactions: dict[str, int]
    language_labels: dict[str, int]
    raw_documents: int
    raw_text_bytes: int


class DistributedCandidateMapper:
    """Run every expensive stateless production gate for one ordered batch.

    The embedded :class:`DataQualityGate` must remain empty: signatures are
    computed but never accepted into its local index.  Global exact and near
    duplicate decisions happen only after all map outputs are ordered.
    """

    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
        quality_gate: DataQualityGate,
        minhash_codec: MinHashCodec,
        filters: Mapping[str, Any],
        excluded_document_ids: set[str],
        excluded_content_sha256: set[str],
        excluded_near_sha256: set[str],
    ) -> None:
        if quality_gate.near_duplicate.count() != 0:
            raise ValueError("distributed stateless mapper received a populated dedup index")
        self.tokenizer = tokenizer
        self.quality_gate = quality_gate
        self.minhash_codec = minhash_codec
        self.filters = dict(filters)
        self.excluded_document_ids = excluded_document_ids
        self.excluded_content_sha256 = excluded_content_sha256
        self.excluded_near_sha256 = excluded_near_sha256

    def map_records(
        self,
        *,
        source: Mapping[str, Any],
        source_provenance: Mapping[str, Any],
        records: Sequence[tuple[int, Mapping[str, Any] | None]],
    ) -> MappedBatch:
        rejections: Counter[str] = Counter()
        redactions: Counter[str] = Counter()
        languages: Counter[str] = Counter()
        raw_documents = 0
        raw_text_bytes = 0
        pending: list[dict[str, Any]] = []
        dataset = {"id": source["dataset_id"], "revision": source["revision"]}

        for source_line, upstream in records:
            raw_documents += 1
            if upstream is None:
                rejections["invalid_json"] += 1
                continue
            raw_text = upstream.get("text")
            if not isinstance(raw_text, str):
                rejections["missing_text"] += 1
                continue
            raw_text_bytes += len(raw_text.encode("utf-8"))
            provenance, missing = _extract_provenance(
                upstream,
                {**source_provenance, "file": source["source_file"]},
            )
            if missing:
                for name in missing:
                    rejections[f"missing_provenance:{name}"] += 1
                continue
            filtered = filter_document(raw_text, self.filters)
            redactions.update(filtered.redactions)
            if filtered.rejection_reason is not None:
                rejections[filtered.rejection_reason] += 1
                continue
            assert filtered.text is not None
            for segment_index, text in enumerate(
                split_utf8(
                    filtered.text,
                    int(self.filters["max_segment_utf8_bytes"]),
                    int(self.filters["min_document_characters"]),
                )
            ):
                text_bytes = text.encode("utf-8")
                content_sha256 = hashlib.sha256(text_bytes).hexdigest()
                document_id = _record_id(
                    dataset,
                    str(source["source_file"]),
                    source_line,
                    segment_index,
                )
                if (
                    document_id in self.excluded_document_ids
                    or content_sha256 in self.excluded_content_sha256
                ):
                    rejections["heldout_exact"] += 1
                    continue
                evaluation = self.quality_gate.evaluate(text, str(source["category"]))
                if evaluation.rejection_reason is not None:
                    rejections[evaluation.rejection_reason] += 1
                    continue
                assert evaluation.metadata is not None
                assert evaluation.near_duplicate_signature is not None
                signature = self.minhash_codec.signature_bytes(
                    evaluation.near_duplicate_signature
                )
                near_sha256 = self.minhash_codec.fingerprint(signature)
                metadata_fingerprint = str(
                    evaluation.metadata["near_duplicate"]["fingerprint_sha256"]
                )
                if near_sha256 != metadata_fingerprint:
                    raise ValueError("distributed MinHash fingerprint differs from quality gate")
                if near_sha256 in self.excluded_near_sha256:
                    rejections["heldout_near_duplicate"] += 1
                    continue
                languages[str(evaluation.metadata["language"]["label"])] += 1
                pending.append(
                    {
                        "category": str(source["category"]),
                        "component": str(source["component"]),
                        "content_sha256": content_sha256,
                        "dataset_id": str(source["dataset_id"]),
                        "dataset_revision": str(source["revision"]),
                        "document_id": document_id,
                        "lsh_buckets": self.minhash_codec.buckets(
                            evaluation.near_duplicate_signature
                        ),
                        "minhash_signature": signature,
                        "pii_redactions": dict(filtered.redactions),
                        "provenance": provenance,
                        "quality": evaluation.metadata,
                        "raw_sha256": filtered.raw_sha256,
                        "segment_index": segment_index,
                        "source_file": str(source["source_file"]),
                        "source_line": source_line,
                        "source_ordinal": int(source["ordinal"]),
                        "source_sha256": str(source["source_sha256"]),
                        "text": text,
                        "text_utf8_bytes": len(text_bytes),
                    }
                )

        encodings = self.tokenizer.encode_batch(
            [row["text"] for row in pending],
            add_special_tokens=False,
        )
        if len(encodings) != len(pending):
            raise RuntimeError("distributed tokenizer returned the wrong batch size")
        candidates = []
        for row, encoding in zip(pending, encodings, strict=True):
            pii_redactions = row.pop("pii_redactions")
            raw_sha256 = row.pop("raw_sha256")
            candidate = {
                **row,
                "schema_version": CANDIDATE_SCHEMA_VERSION,
                "token_bytes": _token_bytes(encoding.ids),
                "token_count": len(encoding.ids),
                "transformations": {
                    "normalization": "NFC",
                    "pii_detector": PII_DETECTOR_VERSION,
                    "pii_redactions": pii_redactions,
                    "raw_sha256": raw_sha256,
                },
            }
            validate_candidate(candidate, self.minhash_codec)
            candidates.append(candidate)
        return MappedBatch(
            candidates=tuple(candidates),
            rejections=dict(sorted(rejections.items())),
            pii_redactions=dict(sorted(redactions.items())),
            language_labels=dict(sorted(languages.items())),
            raw_documents=raw_documents,
            raw_text_bytes=raw_text_bytes,
        )


def load_training_exclusions(path: Path) -> tuple[set[str], set[str], set[str]]:
    documents: set[str] = set()
    contents: set[str] = set()
    near: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
                document_id = str(row["document_id"])
                content_sha256 = _valid_sha256(row["content_sha256"], "excluded content")
                near_sha256 = _valid_sha256(
                    row["near_duplicate_sha256"], "excluded near duplicate"
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid distributed training exclusion line {line_number}"
                ) from error
            if document_id in documents or content_sha256 in contents:
                raise ValueError("distributed training exclusion ledger contains duplicates")
            documents.add(document_id)
            contents.add(content_sha256)
            near.add(near_sha256)
    if not documents:
        raise ValueError("distributed training exclusion ledger is empty")
    return documents, contents, near


def _validate_dedup_fields(
    row: Mapping[str, Any], codec: MinHashCodec, *, label: str
) -> None:
    if not isinstance(row.get("document_id"), str) or not row["document_id"]:
        raise ValueError(f"distributed {label} document ID is missing")
    for name in ("source_ordinal", "source_line", "segment_index"):
        value = row.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"distributed {label} {name} is invalid")
    if row["source_line"] < 1:
        raise ValueError(f"distributed {label} source line must be positive")
    _valid_sha256(row.get("content_sha256"), f"{label} content")
    signature = row.get("minhash_signature")
    if not isinstance(signature, bytes):
        raise ValueError(f"distributed {label} MinHash signature is missing")
    codec.validate_signature(signature)
    buckets = row.get("lsh_buckets")
    if not isinstance(buckets, Sequence) or len(buckets) != len(codec.hashranges):
        raise ValueError(f"distributed {label} LSH bucket count changed")
    bands = []
    for bucket in buckets:
        if not isinstance(bucket, Mapping):
            raise ValueError(f"distributed {label} LSH bucket is invalid")
        bands.append(bucket.get("band"))
        if not isinstance(bucket.get("bucket"), bytes) or not bucket["bucket"]:
            raise ValueError(f"distributed {label} LSH bucket payload is invalid")
    if bands != list(range(len(codec.hashranges))):
        raise ValueError(f"distributed {label} LSH band order changed")
    for bucket, (start, end) in zip(buckets, codec.hashranges, strict=True):
        if len(bucket["bucket"]) != (end - start) * codec.width:
            raise ValueError(f"distributed {label} LSH bucket byte length changed")


def validate_anchor(anchor: Mapping[str, Any], codec: MinHashCodec) -> None:
    if anchor.get("schema_version") != DEDUP_ANCHOR_SCHEMA_VERSION:
        raise ValueError("unsupported distributed dedup anchor schema")
    _validate_dedup_fields(anchor, codec, label="dedup anchor")


def validate_candidate(candidate: Mapping[str, Any], codec: MinHashCodec) -> None:
    if candidate.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        raise ValueError("unsupported distributed candidate schema")
    _validate_dedup_fields(candidate, codec, label="candidate")
    for name in ("text_utf8_bytes", "token_count"):
        value = candidate.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"distributed candidate {name} is invalid")
    text = candidate.get("text")
    if not isinstance(text, str) or not text:
        raise ValueError("distributed candidate text is empty")
    if len(text.encode("utf-8")) != candidate["text_utf8_bytes"]:
        raise ValueError("distributed candidate UTF-8 byte count changed")
    for name in (
        "category",
        "component",
        "dataset_id",
        "dataset_revision",
        "source_file",
    ):
        if not isinstance(candidate.get(name), str) or not candidate[name]:
            raise ValueError(f"distributed candidate {name} is missing")
    _valid_sha256(candidate.get("source_sha256"), "candidate source")
    content_sha256 = str(candidate["content_sha256"])
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != content_sha256:
        raise ValueError("distributed candidate content SHA-256 changed")
    token_bytes = candidate.get("token_bytes")
    if not isinstance(token_bytes, bytes) or len(token_bytes) != candidate["token_count"] * 2:
        raise ValueError("distributed candidate token payload has an invalid size")
    signature = candidate["minhash_signature"]
    fingerprint = codec.fingerprint(signature)
    try:
        recorded = candidate["quality"]["near_duplicate"]["fingerprint_sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError("distributed candidate quality metadata is incomplete") from error
    if fingerprint != recorded:
        raise ValueError("distributed candidate MinHash fingerprint changed")
    for name in ("provenance", "quality", "transformations"):
        if not isinstance(candidate.get(name), Mapping):
            raise ValueError(f"distributed candidate {name} metadata is missing")


@dataclass(frozen=True)
class DeduplicationResult:
    accepted_document_ids: tuple[str, ...]
    rejection_reasons: dict[str, str]
    exact_duplicate_edges: tuple[tuple[str, str], ...]
    near_duplicate_edges: tuple[tuple[str, str], ...]
    rounds: int


def _exact_candidate_edges(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, str], ...]:
    by_id = {str(row["document_id"]): row for row in candidates}
    by_content: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_content[str(row["content_sha256"])].append(row)
    edges = []
    for rows in by_content.values():
        ordered = sorted(rows, key=_order_tuple)
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                edges.append((str(left["document_id"]), str(right["document_id"])))
    return tuple(
        sorted(
            edges,
            key=lambda pair: (
                _order_tuple(by_id[pair[0]]),
                _order_tuple(by_id[pair[1]]),
            ),
        )
    )


def _candidate_edges(
    candidates: Sequence[Mapping[str, Any]], codec: MinHashCodec
) -> tuple[tuple[str, str], ...]:
    by_id = {str(row["document_id"]): row for row in candidates}
    buckets: dict[tuple[int, bytes], list[str]] = defaultdict(list)
    for row in candidates:
        _validate_dedup_fields(row, codec, label="dedup row")
        document_id = str(row["document_id"])
        for bucket in row["lsh_buckets"]:
            buckets[(int(bucket["band"]), bytes(bucket["bucket"]))].append(document_id)
    possible: set[tuple[str, str]] = set()
    for document_ids in buckets.values():
        ordered = sorted(set(document_ids), key=lambda item: _order_tuple(by_id[item]))
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                possible.add((left, right))
    edges = []
    for left, right in sorted(
        possible,
        key=lambda pair: (_order_tuple(by_id[pair[0]]), _order_tuple(by_id[pair[1]])),
    ):
        similarity = codec.estimated_jaccard(
            bytes(by_id[left]["minhash_signature"]),
            bytes(by_id[right]["minhash_signature"]),
        )
        if similarity >= codec.threshold:
            edges.append((left, right))
    return tuple(edges)


def parallel_ordered_greedy(
    order_by_id: Mapping[str, tuple[int, int, int, str]],
    edges: Iterable[tuple[str, str]],
) -> tuple[tuple[str, ...], set[str], int]:
    """Return the fixed-order greedy independent set in parallel rounds."""

    active = set(order_by_id)
    incoming: dict[str, set[str]] = defaultdict(set)
    outgoing: dict[str, set[str]] = defaultdict(set)
    for raw_left, raw_right in edges:
        if raw_left not in active or raw_right not in active or raw_left == raw_right:
            raise ValueError("near-duplicate graph references an invalid document")
        left, right = raw_left, raw_right
        if order_by_id[right] < order_by_id[left]:
            left, right = right, left
        if order_by_id[left] == order_by_id[right]:
            raise ValueError("near-duplicate graph contains equal priorities")
        incoming[right].add(left)
        outgoing[left].add(right)

    accepted: set[str] = set()
    rejected: set[str] = set()
    rounds = 0
    while active:
        roots = {
            document_id
            for document_id in active
            if not (incoming.get(document_id, set()) & active)
        }
        if not roots:
            raise RuntimeError("ordered near-duplicate graph did not make progress")
        newly_rejected = {
            neighbour
            for root in roots
            for neighbour in outgoing.get(root, set())
            if neighbour in active
        }
        accepted.update(roots)
        rejected.update(newly_rejected)
        active.difference_update(roots | newly_rejected)
        rounds += 1
    ordered = tuple(sorted(accepted, key=order_by_id.__getitem__))
    return ordered, rejected, rounds


def sequential_ordered_greedy(
    order_by_id: Mapping[str, tuple[int, int, int, str]],
    edges: Iterable[tuple[str, str]],
) -> tuple[str, ...]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    accepted: list[str] = []
    accepted_set: set[str] = set()
    for document_id in sorted(order_by_id, key=order_by_id.__getitem__):
        if adjacency.get(document_id, set()) & accepted_set:
            continue
        accepted.append(document_id)
        accepted_set.add(document_id)
    return tuple(accepted)


def deduplicate_candidates(
    candidates: Sequence[Mapping[str, Any]],
    codec: MinHashCodec,
    *,
    anchors: Sequence[Mapping[str, Any]] = (),
) -> DeduplicationResult:
    """Apply source-order dedup with an optional immutable accepted prefix.

    ``anchors`` are previously accepted documents from a resumable prefix.  A
    candidate may conflict with an anchor, but anchors are never reconsidered.
    This exactly models continuing the original single-writer traversal.
    """

    ordered_anchors = sorted(anchors, key=_order_tuple)
    ordered = sorted(candidates, key=_order_tuple)
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in ordered_anchors:
        validate_anchor(row, codec)
    for row in ordered:
        validate_candidate(row, codec)
    for row in (*ordered_anchors, *ordered):
        document_id = str(row["document_id"])
        if document_id in by_id:
            raise ValueError("distributed candidates contain a duplicate document ID")
        by_id[document_id] = row
    if (
        ordered_anchors
        and ordered
        and _order_tuple(ordered_anchors[-1]) >= _order_tuple(ordered[0])
    ):
        raise ValueError("distributed resume anchors are not a strict source-order prefix")

    all_rows = [*ordered_anchors, *ordered]
    exact_edges = _exact_candidate_edges(all_rows)
    near_edges = _candidate_edges(all_rows, codec)
    conflict_edges = tuple(sorted(set(exact_edges) | set(near_edges)))
    anchor_ids = {str(row["document_id"]) for row in ordered_anchors}
    candidate_ids = {str(row["document_id"]) for row in ordered}
    if any(left in anchor_ids and right in anchor_ids for left, right in conflict_edges):
        raise ValueError("distributed resume anchors conflict with each other")

    prefix_rejected = {
        right
        for left, right in conflict_edges
        if left in anchor_ids and right in candidate_ids
    }
    active_candidate_ids = candidate_ids - prefix_rejected
    active_edges = tuple(
        (left, right)
        for left, right in conflict_edges
        if left in active_candidate_ids and right in active_candidate_ids
    )
    order_by_id = {
        document_id: _order_tuple(by_id[document_id])
        for document_id in active_candidate_ids
    }
    accepted, graph_rejected, rounds = parallel_ordered_greedy(order_by_id, active_edges)
    sequential = sequential_ordered_greedy(order_by_id, active_edges)
    if accepted != sequential:
        raise RuntimeError("parallel near-dedup differs from sequential source-order semantics")
    rejected = prefix_rejected | graph_rejected
    accepted_with_anchors = anchor_ids | set(accepted)
    exact_rejected = {
        right
        for left, right in exact_edges
        if left in accepted_with_anchors and right in rejected
    }
    rejections = {
        document_id: (
            "exact_duplicate" if document_id in exact_rejected else "near_duplicate"
        )
        for document_id in rejected
    }
    if set(accepted) | rejected != candidate_ids or set(accepted) & rejected:
        raise RuntimeError("distributed dedup document conservation invariant failed")
    return DeduplicationResult(
        accepted_document_ids=accepted,
        rejection_reasons=dict(sorted(rejections.items())),
        exact_duplicate_edges=exact_edges,
        near_duplicate_edges=near_edges,
        rounds=rounds,
    )


def candidate_to_spark_row(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one validated candidate into the Glue Parquet row contract."""

    return {
        "schema_version": int(candidate["schema_version"]),
        "document_id": str(candidate["document_id"]),
        "source_ordinal": int(candidate["source_ordinal"]),
        "source_line": int(candidate["source_line"]),
        "segment_index": int(candidate["segment_index"]),
        "category": str(candidate["category"]),
        "component": str(candidate["component"]),
        "dataset_id": str(candidate["dataset_id"]),
        "dataset_revision": str(candidate["dataset_revision"]),
        "source_file": str(candidate["source_file"]),
        "source_sha256": str(candidate["source_sha256"]),
        "content_sha256": str(candidate["content_sha256"]),
        "minhash_signature": bytes(candidate["minhash_signature"]),
        "lsh_buckets": [
            {"band": int(row["band"]), "bucket": bytes(row["bucket"])}
            for row in candidate["lsh_buckets"]
        ],
        "text": str(candidate["text"]),
        "text_utf8_bytes": int(candidate["text_utf8_bytes"]),
        "token_bytes": bytes(candidate["token_bytes"]),
        "token_count": int(candidate["token_count"]),
        "provenance_json": _canonical_json_bytes(candidate["provenance"]).decode(),
        "quality_json": _canonical_json_bytes(candidate["quality"]).decode(),
        "transformations_json": _canonical_json_bytes(candidate["transformations"]).decode(),
    }


def candidate_to_anchor(candidate: Mapping[str, Any], codec: MinHashCodec) -> dict[str, Any]:
    """Strip a mapped candidate to the immutable resume-anchor contract."""

    validate_candidate(candidate, codec)
    return {
        "schema_version": DEDUP_ANCHOR_SCHEMA_VERSION,
        "document_id": str(candidate["document_id"]),
        "source_ordinal": int(candidate["source_ordinal"]),
        "source_line": int(candidate["source_line"]),
        "segment_index": int(candidate["segment_index"]),
        "content_sha256": str(candidate["content_sha256"]),
        "minhash_signature": bytes(candidate["minhash_signature"]),
        "lsh_buckets": [
            {"band": int(row["band"]), "bucket": bytes(row["bucket"])}
            for row in candidate["lsh_buckets"]
        ],
    }


def build_distributed_corpus_plan(
    contract_path: Path,
    *,
    repository_root: Path,
    source_commit: str,
) -> dict[str, Any]:
    """Validate and fingerprint the roadmap-bounded Glue refactor plan."""

    root = repository_root.resolve()
    contract = _read_json(contract_path)
    if contract.get("schema_version") != DISTRIBUTED_CORPUS_SCHEMA_VERSION:
        raise ValueError("unsupported distributed corpus contract schema")
    if contract.get("gate") != "distributed_production_corpus_glue_v0.1":
        raise ValueError("unexpected distributed corpus gate")
    commit = _valid_source_commit(source_commit)
    authorization = contract.get("authorization", {})
    if authorization.get("production_adaptive_acquisition") is not True:
        raise ValueError("distributed corpus refactor lacks acquisition authorization")
    if any(
        authorization.get(name) is not False
        for name in (
            "full_4b_training",
            "model_training",
            "paid_gpu_compute",
            "paid_glue_canary",
            "paid_glue_production",
        )
    ):
        raise ValueError("distributed refactor must not authorize paid compute or training")

    inputs = contract.get("inputs")
    expected_inputs = {
        "baseline_evidence",
        "component_inventory",
        "data_mix",
        "data_quality",
        "filter_policy",
        "glue_pricing",
        "production_authorization",
        "production_worker",
        "source_file_manifest",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != expected_inputs:
        raise ValueError("distributed corpus input set changed")
    pinned_inputs = {}
    for name, pin in sorted(inputs.items()):
        path = _resolve(root, pin.get("path"), name)
        digest = _sha256_file(path)
        if digest != _valid_sha256(pin.get("sha256"), f"{name} SHA-256"):
            raise ValueError(f"distributed corpus input changed: {name}")
        pinned_inputs[name] = {"path": str(path.relative_to(root)), "sha256": digest}

    limits = contract.get("limits", {})
    expected_limits = {
        "candidate_hard_maximum_tokens": 150_000_000_000,
        "candidate_total_floor_tokens": 120_000_000_000,
        "final_training_tokens": 80_000_000_000,
        "hard_logical_storage_bytes": 2_000_000_000_000,
        "maximum_category_overshoot_tokens": 100_000_000,
    }
    if limits != expected_limits:
        raise ValueError("distributed corpus limits differ from the sealed roadmap")

    backend = contract.get("backend", {})
    if backend.get("service") != "aws_glue" or backend.get("engine") != "spark":
        raise ValueError("distributed corpus backend must be AWS Glue Spark")
    if backend.get("glue_version") != "5.1" or backend.get("python_version") != "3.11":
        raise ValueError("distributed corpus Glue runtime changed")
    if backend.get("worker_type") != "G.1X" or backend.get("dpu_per_worker") != 1:
        raise ValueError("distributed corpus worker shape changed")
    if backend.get("deduplication") != "ordered_parallel_greedy_mis_v0.1":
        raise ValueError("distributed corpus deduplication semantics changed")
    if backend.get("source_partitioning") != "one_pinned_gzip_file_per_task_v0.1":
        raise ValueError("distributed corpus source partitioning changed")
    expected_safety_caps = {
        "maximum_candidate_pairs": 50_000_000,
        "maximum_lsh_bucket_documents": 10_000,
        "maximum_near_dedup_rounds": 256,
    }
    for name, expected in expected_safety_caps.items():
        if backend.get(name) != expected:
            raise ValueError(f"distributed corpus safety cap changed: {name}")

    external = contract.get("external_artifacts", {})
    expected_external = {
        "frozen_tokenizer": {
            "bytes": 4_647_195,
            "s3_key": (
                "lumi-corpus/v0.1/control/bootstrap/artifacts/frozen_tokenizer/"
                "1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee.json"
            ),
            "sha256": "1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee",
        },
        "training_exclusion_ledger": {
            "bytes": 12_476_739,
            "entries": 35_889,
            "s3_key": (
                "lumi-corpus/v0.1/control/bootstrap/artifacts/"
                "training_exclusion_ledger/"
                "709027bb95f3e0f956b22a79d4b29f21490ef0b360ba10f2faca7d5db3d06c62.jsonl"
            ),
            "sha256": "709027bb95f3e0f956b22a79d4b29f21490ef0b360ba10f2faca7d5db3d06c62",
        },
    }
    if external != expected_external:
        raise ValueError("distributed corpus external artifact pins changed")

    pricing_path = root / pinned_inputs["glue_pricing"]["path"]
    pricing = _read_json(pricing_path)
    dpu_rate = Decimal(str(pricing.get("dpu_hour_usd")))
    if dpu_rate != Decimal("0.44"):
        raise ValueError("distributed corpus Glue DPU price changed")
    if pricing.get("dpu_vcpus") != GLUE_DPU_VCPUS:
        raise ValueError("distributed corpus Glue DPU vCPU definition changed")
    if pricing.get("dpu_memory_gib") != GLUE_DPU_MEMORY_GIB:
        raise ValueError("distributed corpus Glue DPU memory definition changed")

    canary = contract.get("canary", {})
    if (
        canary.get("maximum_dpu") != 8
        or canary.get("maximum_records_per_source") != 2_000
        or canary.get("maximum_runtime_minutes") != 30
        or canary.get("maximum_source_compressed_bytes") != 700_000_000
        or canary.get("maximum_total_source_compressed_bytes") != 1_500_000_000
        or canary.get("source_files") != 8
        or canary.get("requires_new_owner_approval") is not True
    ):
        raise ValueError("distributed corpus canary envelope changed")
    canary_cost = (
        Decimal(canary["maximum_dpu"])
        * Decimal(canary["maximum_runtime_minutes"])
        * dpu_rate
        / Decimal(60)
    )
    if canary_cost != Decimal("1.76"):
        raise ValueError("distributed corpus canary cost is not exactly bounded")
    production = contract.get("production", {})
    if production.get("initial_maximum_dpu") != 64:
        raise ValueError("distributed corpus initial production DPU bound changed")
    if production.get("hard_maximum_dpu") != 128:
        raise ValueError("distributed corpus hard DPU bound changed")
    if production.get("requires_new_owner_approval") is not True:
        raise ValueError("distributed production must require new owner approval")
    if production.get("adaptive_scaling") is not True:
        raise ValueError("distributed production must retain adaptive scaling")
    if production.get("resume_from_verified_prefix") is not True:
        raise ValueError("distributed production must resume the verified prefix")

    baseline = _read_json(root / pinned_inputs["baseline_evidence"]["path"])
    if baseline.get("gate") != "verified_single_writer_resume_baseline_v0.1":
        raise ValueError("distributed corpus resume baseline is invalid")
    if baseline.get("verification", {}).get("status") != "independently_verified":
        raise ValueError("distributed corpus resume baseline is not independently verified")
    if baseline.get("instance", {}).get("state") != "terminated":
        raise ValueError("distributed corpus old worker is not terminated")
    baseline_totals = baseline.get("totals", {})
    if baseline_totals != {
        "accepted_documents": 38_349,
        "accepted_tokens": 237_769_640,
        "sealed_documents": 32_393,
        "sealed_tokens": 200_614_708,
    }:
        raise ValueError("distributed corpus resume totals changed")
    if baseline.get("derived_resume_anchor") != {
        "bytes": 24_785_510,
        "canonical_rows_sha256": (
            "41d14457dd9f01973dd17952616451282f570e141b18fdc8b0d68cfebe7b2c51"
        ),
        "rows": 38_349,
        "sha256": "530ac0e1b6773cc2182aba6bd2dd923e4f0362bae3289b312989ab06cebc1ee1",
    }:
        raise ValueError("distributed corpus derived resume anchor changed")
    partial = baseline.get("partial_prefix", {})
    if (
        partial.get("accepted_documents") != 5_956
        or partial.get("accepted_tokens") != 37_154_932
        or partial.get("source_ordinal") != 0
        or partial.get("source_next_line") != 38_848
    ):
        raise ValueError("distributed corpus partial prefix changed")

    source_manifest = _read_json(root / pinned_inputs["source_file_manifest"]["path"])
    source_inventory = {
        "compressed_source_bytes": sum(
            int(source["compressed_source_bytes"])
            for source in source_manifest.get("sources", [])
        ),
        "data_files": sum(
            int(source["data_file_count"]) for source in source_manifest.get("sources", [])
        ),
        "repositories": len(source_manifest.get("sources", [])),
    }
    if source_inventory != {
        "compressed_source_bytes": 499_817_285_384,
        "data_files": 758,
        "repositories": 23,
    }:
        raise ValueError("distributed corpus source inventory changed")

    material = {
        "authorization": authorization,
        "backend": backend,
        "canary": canary,
        "external_artifacts": external,
        "inputs": pinned_inputs,
        "limits": limits,
        "production": production,
        "resume_baseline": baseline_totals,
        "schema_version": DISTRIBUTED_CORPUS_SCHEMA_VERSION,
        "source_inventory": source_inventory,
        "source_commit": commit,
    }
    return {
        **material,
        "cost": {
            "canary_hard_maximum_usd": str(canary_cost),
            "dpu_hour_usd": str(dpu_rate),
            "initial_production_usd_per_hour": str(
                (Decimal(production["initial_maximum_dpu"]) * dpu_rate).quantize(
                    Decimal("0.01")
                )
            ),
            "hard_production_usd_per_hour": str(
                (Decimal(production["hard_maximum_dpu"]) * dpu_rate).quantize(
                    Decimal("0.01")
                )
            ),
        },
        "decision": {
            "full_4b_training": "not_authorized",
            "paid_glue_canary": "not_authorized",
            "paid_glue_production": "not_authorized",
            "status": "local_distributed_implementation_and_parity_only",
        },
        "gate": contract["gate"],
        "plan_fingerprint": hashlib.sha256(_canonical_json_bytes(material)).hexdigest(),
    }
