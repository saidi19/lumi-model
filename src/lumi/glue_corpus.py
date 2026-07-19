"""Lazy PySpark adapter for Lumi's distributed corpus deduplication.

Nothing in this module launches AWS resources.  The functions operate on an
already-created Spark session and fail closed on oversized LSH buckets,
non-convergent duplicate graphs, or schema drift.
"""

from __future__ import annotations

from typing import Any

from .distributed_corpus import (
    CANDIDATE_SCHEMA_VERSION,
    DEDUP_ANCHOR_SCHEMA_VERSION,
    MinHashCodec,
)


def _lsh_bucket_type() -> Any:
    from pyspark.sql import types as T

    return T.ArrayType(
        T.StructType(
            [
                T.StructField("band", T.IntegerType(), nullable=False),
                T.StructField("bucket", T.BinaryType(), nullable=False),
            ]
        ),
        containsNull=False,
    )


def candidate_spark_schema() -> Any:
    """Return the binary-safe v1 map-output schema without importing Spark eagerly."""

    from pyspark.sql import types as T

    return T.StructType(
        [
            T.StructField("schema_version", T.IntegerType(), nullable=False),
            T.StructField("document_id", T.StringType(), nullable=False),
            T.StructField("source_ordinal", T.IntegerType(), nullable=False),
            T.StructField("source_line", T.LongType(), nullable=False),
            T.StructField("segment_index", T.IntegerType(), nullable=False),
            T.StructField("category", T.StringType(), nullable=False),
            T.StructField("component", T.StringType(), nullable=False),
            T.StructField("dataset_id", T.StringType(), nullable=False),
            T.StructField("dataset_revision", T.StringType(), nullable=False),
            T.StructField("source_file", T.StringType(), nullable=False),
            T.StructField("source_sha256", T.StringType(), nullable=False),
            T.StructField("content_sha256", T.StringType(), nullable=False),
            T.StructField("minhash_signature", T.BinaryType(), nullable=False),
            T.StructField("lsh_buckets", _lsh_bucket_type(), nullable=False),
            T.StructField("text", T.StringType(), nullable=False),
            T.StructField("text_utf8_bytes", T.IntegerType(), nullable=False),
            T.StructField("token_bytes", T.BinaryType(), nullable=False),
            T.StructField("token_count", T.IntegerType(), nullable=False),
            T.StructField("provenance_json", T.StringType(), nullable=False),
            T.StructField("quality_json", T.StringType(), nullable=False),
            T.StructField("transformations_json", T.StringType(), nullable=False),
        ]
    )


def dedup_anchor_spark_schema() -> Any:
    """Return the minimal immutable accepted-prefix schema."""

    from pyspark.sql import types as T

    return T.StructType(
        [
            T.StructField("schema_version", T.IntegerType(), nullable=False),
            T.StructField("document_id", T.StringType(), nullable=False),
            T.StructField("source_ordinal", T.IntegerType(), nullable=False),
            T.StructField("source_line", T.LongType(), nullable=False),
            T.StructField("segment_index", T.IntegerType(), nullable=False),
            T.StructField("content_sha256", T.StringType(), nullable=False),
            T.StructField("minhash_signature", T.BinaryType(), nullable=False),
            T.StructField("lsh_buckets", _lsh_bucket_type(), nullable=False),
        ]
    )


def _order_before(left: str, right: str) -> Any:
    from pyspark.sql import functions as F

    return (
        (F.col(f"{left}.source_ordinal") < F.col(f"{right}.source_ordinal"))
        | (
            (F.col(f"{left}.source_ordinal") == F.col(f"{right}.source_ordinal"))
            & (F.col(f"{left}.source_line") < F.col(f"{right}.source_line"))
        )
        | (
            (F.col(f"{left}.source_ordinal") == F.col(f"{right}.source_ordinal"))
            & (F.col(f"{left}.source_line") == F.col(f"{right}.source_line"))
            & (F.col(f"{left}.segment_index") < F.col(f"{right}.segment_index"))
        )
        | (
            (F.col(f"{left}.source_ordinal") == F.col(f"{right}.source_ordinal"))
            & (F.col(f"{left}.source_line") == F.col(f"{right}.source_line"))
            & (F.col(f"{left}.segment_index") == F.col(f"{right}.segment_index"))
            & (F.col(f"{left}.document_id") < F.col(f"{right}.document_id"))
        )
    )


def _estimated_jaccard_udf(codec: MinHashCodec) -> Any:
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    width = codec.width
    expected_bytes = codec.num_perm * width

    @F.udf(T.DoubleType())
    def estimated_jaccard(left: bytearray, right: bytearray) -> float:
        left_bytes = bytes(left)
        right_bytes = bytes(right)
        if len(left_bytes) != expected_bytes or len(right_bytes) != expected_bytes:
            raise ValueError("Spark candidate contains an invalid MinHash signature")
        matches = sum(
            left_bytes[offset : offset + width] == right_bytes[offset : offset + width]
            for offset in range(0, expected_bytes, width)
        )
        return matches / codec.num_perm

    return estimated_jaccard


def spark_deduplicate_candidates(
    candidates: Any,
    *,
    codec: MinHashCodec,
    checkpoint_uri: str,
    anchors: Any | None = None,
    allow_local_checkpoint: bool = False,
    maximum_rounds: int = 256,
    maximum_lsh_bucket_documents: int = 10_000,
    maximum_candidate_pairs: int = 50_000_000,
) -> dict[str, Any]:
    """Apply exact and source-order-equivalent dedup to a Spark DataFrame.

    The returned mapping contains ``accepted``, ``exact_rejected``, and
    ``near_rejected`` DataFrames plus scalar metrics. ``anchors`` is an
    immutable accepted prefix exported from a resumable single-writer state.
    Callers remain responsible for content-addressed shard sealing and S3
    publication.
    """

    from pyspark.sql import functions as F

    if not checkpoint_uri.startswith("s3://") and not (
        allow_local_checkpoint and checkpoint_uri.startswith("file://")
    ):
        raise ValueError("distributed Spark checkpoint URI must be private S3")
    if maximum_rounds < 1:
        raise ValueError("distributed near-dedup maximum rounds must be positive")
    if maximum_lsh_bucket_documents < 2:
        raise ValueError("distributed LSH bucket cap must be at least two")
    if maximum_candidate_pairs < 1:
        raise ValueError("distributed candidate-pair cap must be positive")

    spark = candidates.sparkSession
    spark.sparkContext.setCheckpointDir(checkpoint_uri)
    candidate_required = {field.name for field in candidate_spark_schema().fields}
    if set(candidates.columns) != candidate_required:
        missing = sorted(candidate_required - set(candidates.columns))
        extra = sorted(set(candidates.columns) - candidate_required)
        raise ValueError(
            f"distributed Spark candidate schema changed: missing={missing} extra={extra}"
        )
    if anchors is None:
        anchors = spark.createDataFrame([], dedup_anchor_spark_schema())
    anchor_required = {field.name for field in dedup_anchor_spark_schema().fields}
    if set(anchors.columns) != anchor_required:
        missing = sorted(anchor_required - set(anchors.columns))
        extra = sorted(set(anchors.columns) - anchor_required)
        raise ValueError(
            f"distributed Spark anchor schema changed: missing={missing} extra={extra}"
        )
    for frame, expected, label in (
        (candidates, CANDIDATE_SCHEMA_VERSION, "candidate"),
        (anchors, DEDUP_ANCHOR_SCHEMA_VERSION, "anchor"),
    ):
        versions = [row[0] for row in frame.select("schema_version").distinct().limit(2).collect()]
        if any(version != expected for version in versions):
            raise ValueError(f"distributed Spark {label} schema version changed")

    for frame, label in ((candidates, "candidates"), (anchors, "anchors")):
        duplicate_ids = (
            frame.groupBy("document_id")
            .count()
            .where(F.col("count") != 1)
            .limit(1)
            .collect()
        )
        if duplicate_ids:
            raise ValueError(f"distributed Spark {label} contain duplicate document IDs")
    overlap = candidates.select("document_id").join(
        anchors.select("document_id"), "document_id", "inner"
    )
    if overlap.limit(1).collect():
        raise ValueError("distributed Spark candidates overlap resume anchors")

    order_columns = ["source_ordinal", "source_line", "segment_index", "document_id"]
    last_anchor = anchors.orderBy(*[F.col(name).desc() for name in order_columns]).first()
    first_candidate = candidates.orderBy(*[F.col(name).asc() for name in order_columns]).first()
    if last_anchor is not None and first_candidate is not None:
        anchor_order = tuple(last_anchor[name] for name in order_columns)
        candidate_order = tuple(first_candidate[name] for name in order_columns)
        if anchor_order >= candidate_order:
            raise ValueError("distributed Spark anchors are not a strict source-order prefix")

    dedup_columns = [
        "document_id",
        "source_ordinal",
        "source_line",
        "segment_index",
        "content_sha256",
        "minhash_signature",
        "lsh_buckets",
    ]
    candidate_nodes = candidates.select(*dedup_columns).withColumn(
        "node_kind", F.lit("candidate")
    )
    anchor_nodes = anchors.select(*dedup_columns).withColumn("node_kind", F.lit("anchor"))
    nodes = anchor_nodes.unionByName(candidate_nodes).persist()
    left_nodes = nodes.alias("left")
    right_nodes = nodes.alias("right")
    exact_edges = (
        left_nodes.join(
            right_nodes,
            (F.col("left.content_sha256") == F.col("right.content_sha256"))
            & (F.col("right.node_kind") == F.lit("candidate"))
            & _order_before("left", "right"),
            "inner",
        )
        .select(
            F.col("left.document_id").alias("earlier_id"),
            F.col("left.node_kind").alias("earlier_kind"),
            F.col("right.document_id").alias("later_id"),
        )
        .checkpoint(eager=True)
        .persist()
    )

    exploded = (
        nodes.select(
            "document_id",
            "source_ordinal",
            "source_line",
            "segment_index",
            "minhash_signature",
            "node_kind",
            F.explode("lsh_buckets").alias("lsh"),
        )
        .select(
            "document_id",
            "source_ordinal",
            "source_line",
            "segment_index",
            "minhash_signature",
            "node_kind",
            F.col("lsh.band").alias("band"),
            F.col("lsh.bucket").alias("bucket"),
        )
        .persist()
    )
    oversized = (
        exploded.groupBy("band", "bucket")
        .count()
        .where(F.col("count") > maximum_lsh_bucket_documents)
        .orderBy(F.col("count").desc())
        .limit(1)
        .collect()
    )
    if oversized:
        raise RuntimeError(
            "distributed LSH bucket exceeds the fail-closed candidate-pair cap: "
            f"{oversized[0]['count']}"
        )

    left = exploded.alias("left")
    right = exploded.alias("right")
    possible_edges = (
        left.join(
            right,
            (F.col("left.band") == F.col("right.band"))
            & (F.col("left.bucket") == F.col("right.bucket"))
            & (F.col("right.node_kind") == F.lit("candidate"))
            & _order_before("left", "right"),
            "inner",
        )
        .select(
            F.col("left.document_id").alias("earlier_id"),
            F.col("left.node_kind").alias("earlier_kind"),
            F.col("right.document_id").alias("later_id"),
            F.col("left.minhash_signature").alias("earlier_signature"),
            F.col("right.minhash_signature").alias("later_signature"),
        )
        .dropDuplicates(["earlier_id", "later_id"])
        .persist()
    )
    possible_pair_count = possible_edges.count()
    if possible_pair_count > maximum_candidate_pairs:
        raise RuntimeError(
            "distributed LSH candidate pairs exceed the fail-closed cap: "
            f"{possible_pair_count}"
        )
    similarity = _estimated_jaccard_udf(codec)
    near_edges = (
        possible_edges.withColumn(
            "estimated_jaccard",
            similarity("earlier_signature", "later_signature"),
        )
        .where(F.col("estimated_jaccard") >= F.lit(codec.threshold))
        .select("earlier_id", "earlier_kind", "later_id")
        .checkpoint(eager=True)
        .persist()
    )
    conflicts = (
        exact_edges.select("earlier_id", "earlier_kind", "later_id")
        .unionByName(near_edges.select("earlier_id", "earlier_kind", "later_id"))
        .dropDuplicates(["earlier_id", "later_id"])
        .checkpoint(eager=True)
        .persist()
    )
    prefix_rejected = (
        conflicts.where(F.col("earlier_kind") == F.lit("anchor"))
        .select(F.col("later_id").alias("document_id"))
        .distinct()
        .checkpoint(eager=True)
        .persist()
    )
    active = (
        candidates.select(F.col("document_id").alias("active_id"))
        .join(
            prefix_rejected.select(F.col("document_id").alias("active_id")),
            "active_id",
            "left_anti",
        )
        .checkpoint(eager=True)
    )
    candidate_conflicts = conflicts.where(F.col("earlier_kind") == F.lit("candidate")).select(
        "earlier_id", "later_id"
    )
    accepted_rounds = []
    rejected_rounds = []
    rounds = 0
    while True:
        active = active.persist()
        active_count = active.count()
        if active_count == 0:
            break
        if rounds >= maximum_rounds:
            raise RuntimeError("distributed near-dedup exceeded its maximum round count")
        active_edges = (
            candidate_conflicts.join(
                active.alias("active_earlier"),
                F.col("earlier_id") == F.col("active_earlier.active_id"),
                "inner",
            )
            .join(
                active.alias("active_later"),
                F.col("later_id") == F.col("active_later.active_id"),
                "inner",
            )
            .select("earlier_id", "later_id")
            .persist()
        )
        incoming = active_edges.select(F.col("later_id").alias("active_id")).distinct()
        roots = active.join(incoming, "active_id", "left_anti").persist()
        root_count = roots.count()
        if root_count == 0:
            raise RuntimeError("distributed ordered near-dedup graph made no progress")
        rejected = (
            active_edges.join(
                roots,
                F.col("earlier_id") == F.col("active_id"),
                "inner",
            )
            .select(F.col("later_id").alias("active_id"))
            .distinct()
            .persist()
        )
        accepted_rounds.append(
            roots.select(
                F.col("active_id").alias("document_id"),
                F.lit(rounds).alias("dedup_round"),
            ).checkpoint(eager=True)
        )
        rejected_rounds.append(
            rejected.select(
                F.col("active_id").alias("document_id"),
                F.lit(rounds).alias("dedup_round"),
                F.lit("near_duplicate").alias("rejection_reason"),
            ).checkpoint(eager=True)
        )
        survivors = (
            active.join(roots, "active_id", "left_anti")
            .join(rejected, "active_id", "left_anti")
            .checkpoint(eager=True)
        )
        active.unpersist()
        active_edges.unpersist()
        roots.unpersist()
        rejected.unpersist()
        active = survivors
        rounds += 1

    if not accepted_rounds:
        accepted_ids = spark.createDataFrame([], "document_id string, dedup_round int")
    else:
        accepted_ids = accepted_rounds[0]
        for frame in accepted_rounds[1:]:
            accepted_ids = accepted_ids.unionByName(frame)
        accepted_ids = accepted_ids.checkpoint(eager=True)
    if not rejected_rounds:
        graph_rejected_ids = spark.createDataFrame(
            [], "document_id string, dedup_round int, rejection_reason string"
        )
    else:
        graph_rejected_ids = rejected_rounds[0]
        for frame in rejected_rounds[1:]:
            graph_rejected_ids = graph_rejected_ids.unionByName(frame)
        graph_rejected_ids = graph_rejected_ids.checkpoint(eager=True)

    all_rejected_ids = (
        prefix_rejected.select(
            "document_id",
            F.lit(-1).alias("dedup_round"),
            F.lit("conflict_with_resume_anchor").alias("rejection_reason"),
        )
        .unionByName(graph_rejected_ids)
        .dropDuplicates(["document_id"])
        .checkpoint(eager=True)
        .persist()
    )
    accepted_with_anchors = anchors.select(
        F.col("document_id").alias("accepted_id")
    ).unionByName(accepted_ids.select(F.col("document_id").alias("accepted_id")))
    exact_rejected_ids = (
        exact_edges.join(
            accepted_with_anchors,
            F.col("earlier_id") == F.col("accepted_id"),
            "inner",
        )
        .select(F.col("later_id").alias("document_id"))
        .join(all_rejected_ids.select("document_id"), "document_id", "inner")
        .distinct()
        .checkpoint(eager=True)
    )
    near_rejected_ids = (
        all_rejected_ids.select("document_id")
        .join(exact_rejected_ids, "document_id", "left_anti")
        .checkpoint(eager=True)
    )
    accepted = candidates.join(accepted_ids, "document_id", "inner")
    exact_rejected = (
        candidates.join(exact_rejected_ids, "document_id", "inner")
        .withColumn("rejection_reason", F.lit("exact_duplicate"))
    )
    near_rejected = (
        candidates.join(near_rejected_ids, "document_id", "inner")
        .withColumn("rejection_reason", F.lit("near_duplicate"))
    )
    metrics = {
        "anchor_documents": anchors.count(),
        "accepted_documents": accepted_ids.count(),
        "candidate_documents": candidates.count(),
        "exact_rejected_documents": exact_rejected.count(),
        "exact_duplicate_edges": exact_edges.count(),
        "lsh_candidate_pairs": possible_pair_count,
        "near_duplicate_edges": near_edges.count(),
        "near_rejected_documents": near_rejected_ids.count(),
        "prefix_rejected_documents": prefix_rejected.count(),
        "rounds": rounds,
    }
    if (
        metrics["accepted_documents"]
        + metrics["exact_rejected_documents"]
        + metrics["near_rejected_documents"]
        != metrics["candidate_documents"]
    ):
        raise RuntimeError("distributed dedup document conservation invariant failed")
    exploded.unpersist()
    exact_edges.unpersist()
    near_edges.unpersist()
    conflicts.unpersist()
    prefix_rejected.unpersist()
    all_rejected_ids.unpersist()
    possible_edges.unpersist()
    nodes.unpersist()
    return {
        "accepted": accepted,
        "accepted_ids": accepted_ids,
        "exact_rejected": exact_rejected,
        "near_rejected": near_rejected,
        "rejected_ids": all_rejected_ids,
        "metrics": metrics,
    }
