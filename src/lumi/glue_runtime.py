"""Shared fail-closed Spark startup configuration for AWS Glue jobs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

FINAL_CORPUS_SPARK_CONFIG: tuple[tuple[str, str], ...] = (
    ("spark.sql.parquet.enableVectorizedReader", "false"),
    ("spark.sql.parquet.columnarReaderBatchSize", "64"),
    ("spark.sql.files.maxPartitionBytes", str(64 * 1024 * 1024)),
    ("spark.sql.shuffle.partitions", "1600"),
    ("spark.default.parallelism", "1600"),
    ("spark.speculation", "false"),
)


def glue_conf_argument(config: Sequence[tuple[str, str]]) -> str:
    """Render immutable Spark startup settings for Glue's single ``--conf`` key."""

    if not config:
        raise ValueError("Glue Spark startup configuration is empty")
    fragments = []
    seen: set[str] = set()
    for name, value in config:
        if not name.startswith("spark.") or not name or not value or name in seen:
            raise ValueError("Glue Spark startup configuration is invalid")
        if any(character.isspace() for character in name + value):
            raise ValueError("Glue Spark startup configuration contains whitespace")
        seen.add(name)
        fragments.append(f"{name}={value}")
    return " --conf ".join(fragments)


def verify_spark_configuration(
    spark: Any, config: Sequence[tuple[str, str]]
) -> dict[str, str]:
    """Read and verify startup configuration without mutating a live Spark context."""

    expected = dict(config)
    actual = {name: str(spark.conf.get(name)) for name in expected}
    if actual != expected:
        raise RuntimeError("Glue Spark startup configuration changed")
    return actual
