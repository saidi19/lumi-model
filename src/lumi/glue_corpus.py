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

DEDUP_CANDIDATE_COLUMNS = (
    "schema_version",
    "document_id",
    "source_ordinal",
    "source_line",
    "segment_index",
    "content_sha256",
    "minhash_signature",
    "lsh_buckets",
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


def dedup_candidate_spark_schema() -> Any:
    """Return the narrow candidate schema used by the global dedup graph."""

    from pyspark.sql import types as T

    fields = {
        field.name: field
        for field in candidate_spark_schema().fields
        if field.name in DEDUP_CANDIDATE_COLUMNS
    }
    return T.StructType([fields[name] for name in DEDUP_CANDIDATE_COLUMNS])


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


def select_ordered_token_prefix(
    candidates: Any,
    *,
    target_tokens: int,
    maximum_overshoot_tokens: int,
    partitions: int | None = None,
) -> dict[str, Any]:
    """Select a canonical token prefix without a single-partition window.

    Range partitions are individually sorted by the complete source-order key.
    Only tiny per-partition token totals are collected by the driver.  The
    cutoff partition then streams until the first document that reaches the
    requested token target, preserving the sequential document-boundary rule.
    """

    from pyspark.sql import functions as F

    if not isinstance(target_tokens, int) or isinstance(target_tokens, bool):
        raise ValueError("ordered token-prefix target must be an integer")
    if target_tokens <= 0:
        raise ValueError("ordered token-prefix target must be positive")
    if not isinstance(maximum_overshoot_tokens, int) or isinstance(
        maximum_overshoot_tokens, bool
    ):
        raise ValueError("ordered token-prefix overshoot cap must be an integer")
    if maximum_overshoot_tokens < 0:
        raise ValueError("ordered token-prefix overshoot cap must be non-negative")
    required = {
        "document_id",
        "segment_index",
        "source_line",
        "source_ordinal",
        "token_count",
    }
    if not required.issubset(candidates.columns):
        raise ValueError("ordered token-prefix input schema is missing required columns")
    if partitions is None:
        partitions = int(candidates.sparkSession.conf.get("spark.sql.shuffle.partitions"))
    if not isinstance(partitions, int) or isinstance(partitions, bool) or partitions < 1:
        raise ValueError("ordered token-prefix partition count must be positive")
    order = ["source_ordinal", "source_line", "segment_index", "document_id"]
    ordered = (
        candidates.repartitionByRange(partitions, *[F.col(name) for name in order])
        .sortWithinPartitions(*[F.col(name).asc() for name in order])
        .persist()
    )

    def summarize(index: int, rows: Any) -> Any:
        documents = 0
        tokens = 0
        for row in rows:
            documents += 1
            token_count = int(row["token_count"])
            if token_count <= 0:
                raise ValueError("ordered token-prefix row has no tokens")
            tokens += token_count
        yield index, documents, tokens

    summaries = sorted(ordered.rdd.mapPartitionsWithIndex(summarize).collect())
    total_documents = sum(int(row[1]) for row in summaries)
    total_tokens = sum(int(row[2]) for row in summaries)
    if total_documents == 0 or total_tokens == 0:
        ordered.unpersist()
        raise ValueError("ordered token-prefix input is empty")
    before_tokens = 0
    cutoff_partition = summaries[-1][0]
    reached_target = total_tokens >= target_tokens
    if reached_target:
        for partition, _, tokens in summaries:
            if before_tokens + int(tokens) >= target_tokens:
                cutoff_partition = int(partition)
                break
            before_tokens += int(tokens)
    else:
        cutoff_partition = summaries[-1][0]
        before_tokens = total_tokens - int(summaries[-1][2])
    cutoff_target = target_tokens - before_tokens

    def select(index: int, rows: Any) -> Any:
        if index < cutoff_partition:
            yield from rows
            return
        if index > cutoff_partition:
            return
        selected_tokens = 0
        for row in rows:
            if reached_target and selected_tokens >= cutoff_target:
                break
            token_count = int(row["token_count"])
            if token_count <= 0:
                raise ValueError("ordered token-prefix row has no tokens")
            yield row
            selected_tokens += token_count

    selected = candidates.sparkSession.createDataFrame(
        ordered.rdd.mapPartitionsWithIndex(select), schema=candidates.schema
    ).persist()
    totals = selected.agg(
        F.count(F.lit(1)).alias("documents"), F.sum("token_count").alias("tokens")
    ).first()
    selected_documents = int(totals["documents"] or 0)
    selected_tokens = int(totals["tokens"] or 0)
    if selected_documents == 0 or selected_tokens == 0:
        selected.unpersist()
        ordered.unpersist()
        raise RuntimeError("ordered token-prefix selected no documents")
    overshoot = max(0, selected_tokens - target_tokens)
    if reached_target and selected_tokens < target_tokens:
        selected.unpersist()
        ordered.unpersist()
        raise RuntimeError("ordered token-prefix stopped before its target")
    if overshoot > maximum_overshoot_tokens:
        selected.unpersist()
        ordered.unpersist()
        raise RuntimeError("ordered token-prefix exceeded its overshoot cap")
    last = selected.orderBy(*[F.col(name).desc() for name in order]).first()
    assert last is not None
    ordered.unpersist()
    return {
        "available_documents": total_documents,
        "available_tokens": total_tokens,
        "cutoff": {name: last[name] for name in order},
        "overshoot_tokens": overshoot,
        "reached_target": reached_target,
        "selected": selected,
        "selected_documents": selected_documents,
        "selected_tokens": selected_tokens,
    }


def select_ordered_token_prefix_at_most(
    candidates: Any,
    *,
    maximum_tokens: int,
    partitions: int | None = None,
) -> dict[str, Any]:
    """Select the largest canonical document prefix that never exceeds a cap.

    Production mapping uses this variant for the final wave in each category.
    Unlike :func:`select_ordered_token_prefix`, it drops the first document
    that would cross the cap, so the aggregate 150B ceiling remains hard.
    """

    from pyspark.sql import functions as F

    if not isinstance(maximum_tokens, int) or isinstance(maximum_tokens, bool):
        raise ValueError("ordered at-most token cap must be an integer")
    if maximum_tokens <= 0:
        raise ValueError("ordered at-most token cap must be positive")
    required = {
        "document_id",
        "segment_index",
        "source_line",
        "source_ordinal",
        "token_count",
    }
    if not required.issubset(candidates.columns):
        raise ValueError("ordered at-most input schema is missing required columns")
    if partitions is None:
        partitions = int(candidates.sparkSession.conf.get("spark.sql.shuffle.partitions"))
    if not isinstance(partitions, int) or isinstance(partitions, bool) or partitions < 1:
        raise ValueError("ordered at-most partition count must be positive")
    order = ["source_ordinal", "source_line", "segment_index", "document_id"]
    ordered = (
        candidates.repartitionByRange(partitions, *[F.col(name) for name in order])
        .sortWithinPartitions(*[F.col(name).asc() for name in order])
        .persist()
    )

    def summarize(index: int, rows: Any) -> Any:
        documents = 0
        tokens = 0
        for row in rows:
            token_count = int(row["token_count"])
            if token_count <= 0:
                raise ValueError("ordered at-most row has no tokens")
            documents += 1
            tokens += token_count
        yield index, documents, tokens

    summaries = sorted(ordered.rdd.mapPartitionsWithIndex(summarize).collect())
    available_documents = sum(int(row[1]) for row in summaries)
    available_tokens = sum(int(row[2]) for row in summaries)
    if available_documents == 0 or available_tokens == 0:
        ordered.unpersist()
        raise ValueError("ordered at-most input is empty")
    if available_tokens < maximum_tokens:
        ordered.unpersist()
        raise ValueError("ordered at-most input does not reach its closing cap")

    before_tokens = 0
    cutoff_partition = summaries[-1][0]
    for partition, _, tokens in summaries:
        if before_tokens + int(tokens) >= maximum_tokens:
            cutoff_partition = int(partition)
            break
        before_tokens += int(tokens)
    cutoff_budget = maximum_tokens - before_tokens

    def select(index: int, rows: Any) -> Any:
        if index < cutoff_partition:
            yield from rows
            return
        if index > cutoff_partition:
            return
        selected_tokens = 0
        for row in rows:
            token_count = int(row["token_count"])
            if token_count <= 0:
                raise ValueError("ordered at-most row has no tokens")
            if selected_tokens + token_count > cutoff_budget:
                break
            yield row
            selected_tokens += token_count

    selected = candidates.sparkSession.createDataFrame(
        ordered.rdd.mapPartitionsWithIndex(select), schema=candidates.schema
    ).persist()
    totals = selected.agg(
        F.count(F.lit(1)).alias("documents"), F.sum("token_count").alias("tokens")
    ).first()
    selected_documents = int(totals["documents"] or 0)
    selected_tokens = int(totals["tokens"] or 0)
    if selected_tokens > maximum_tokens:
        selected.unpersist()
        ordered.unpersist()
        raise RuntimeError("ordered at-most prefix exceeded its hard cap")
    last = selected.orderBy(*[F.col(name).desc() for name in order]).first()
    cutoff = None if last is None else {name: last[name] for name in order}
    ordered.unpersist()
    return {
        "available_documents": available_documents,
        "available_tokens": available_tokens,
        "cutoff": cutoff,
        "selected": selected,
        "selected_documents": selected_documents,
        "selected_tokens": selected_tokens,
        "shortfall_tokens": maximum_tokens - selected_tokens,
    }


def assign_ordered_token_shards(
    candidates: Any,
    *,
    target_tokens: int,
    maximum_overshoot_tokens: int,
    start_sequence: int = 0,
    initial_tokens: int = 0,
    partitions: int | None = None,
) -> dict[str, Any]:
    """Assign deterministic shard IDs without a global single-partition window."""

    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    for value, label, allow_zero in (
        (target_tokens, "target", False),
        (maximum_overshoot_tokens, "overshoot", True),
        (start_sequence, "start sequence", True),
        (initial_tokens, "initial tokens", True),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"ordered shard {label} must be an integer")
        if value < 0 or (value == 0 and not allow_zero):
            raise ValueError(f"ordered shard {label} is outside its allowed range")
    required = {
        "document_id",
        "segment_index",
        "source_line",
        "source_ordinal",
        "token_count",
    }
    if not required.issubset(candidates.columns):
        raise ValueError("ordered shard input schema is missing required columns")
    if partitions is None:
        partitions = int(candidates.sparkSession.conf.get("spark.sql.shuffle.partitions"))
    if not isinstance(partitions, int) or isinstance(partitions, bool) or partitions < 1:
        raise ValueError("ordered shard partition count must be positive")
    order = ["source_ordinal", "source_line", "segment_index", "document_id"]
    ordered = (
        candidates.repartitionByRange(partitions, *[F.col(name) for name in order])
        .sortWithinPartitions(*[F.col(name).asc() for name in order])
        .persist()
    )

    def summarize(index: int, rows: Any) -> Any:
        documents = 0
        tokens = 0
        for row in rows:
            token_count = int(row["token_count"])
            if token_count <= 0:
                raise ValueError("ordered shard row has no tokens")
            documents += 1
            tokens += token_count
        yield index, documents, tokens

    summaries = sorted(ordered.rdd.mapPartitionsWithIndex(summarize).collect())
    total_documents = sum(int(row[1]) for row in summaries)
    total_tokens = sum(int(row[2]) for row in summaries)
    if total_documents == 0 or total_tokens == 0:
        ordered.unpersist()
        raise ValueError("ordered shard input is empty")
    offsets = {}
    running = initial_tokens
    for index, _, tokens in summaries:
        offsets[int(index)] = running
        running += int(tokens)
    def assign(index: int, rows: Any) -> Any:
        cumulative = int(offsets[index])
        for row in rows:
            token_count = int(row["token_count"])
            sequence = start_sequence + cumulative // target_tokens
            yield (*tuple(row), sequence)
            cumulative += token_count

    schema = T.StructType(
        [*candidates.schema.fields, T.StructField("shard_sequence", T.LongType(), False)]
    )
    assigned = candidates.sparkSession.createDataFrame(
        ordered.rdd.mapPartitionsWithIndex(assign), schema=schema
    ).persist()
    totals = assigned.agg(
        F.count(F.lit(1)).alias("documents"), F.sum("token_count").alias("tokens")
    ).first()
    if int(totals["documents"] or 0) != total_documents or int(
        totals["tokens"] or 0
    ) != total_tokens:
        assigned.unpersist()
        ordered.unpersist()
        raise RuntimeError("ordered shard assignment does not conserve its input")
    shard_totals = assigned.groupBy("shard_sequence").agg(
        F.count(F.lit(1)).alias("documents"), F.sum("token_count").alias("tokens")
    )
    maximum = shard_totals.agg(F.max("tokens").alias("tokens")).first()
    if int(maximum["tokens"] or 0) > target_tokens + maximum_overshoot_tokens:
        assigned.unpersist()
        ordered.unpersist()
        raise RuntimeError("ordered shard assignment exceeded its overshoot cap")
    minimum_sequence = int(shard_totals.agg(F.min("shard_sequence")).first()[0])
    maximum_sequence = int(shard_totals.agg(F.max("shard_sequence")).first()[0])
    shard_count = int(shard_totals.count())
    if shard_count != maximum_sequence - minimum_sequence + 1:
        assigned.unpersist()
        ordered.unpersist()
        raise RuntimeError("ordered shard assignment sequence is not contiguous")
    ordered.unpersist()
    return {
        "assigned": assigned,
        "documents": total_documents,
        "first_sequence": minimum_sequence,
        "last_sequence": maximum_sequence,
        "shard_count": shard_count,
        "tokens": total_tokens,
    }


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
    candidate_columns = set(candidates.columns)
    full_candidate_columns = {field.name for field in candidate_spark_schema().fields}
    dedup_candidate_columns = set(DEDUP_CANDIDATE_COLUMNS)
    if candidate_columns not in (full_candidate_columns, dedup_candidate_columns):
        missing = sorted(dedup_candidate_columns - candidate_columns)
        extra = sorted(candidate_columns - full_candidate_columns)
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
