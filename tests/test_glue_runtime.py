from __future__ import annotations

from pathlib import Path

import pytest

from lumi.glue_runtime import (
    FINAL_CORPUS_SPARK_CONFIG,
    glue_conf_argument,
    verify_spark_configuration,
)

ROOT = Path(__file__).resolve().parents[1]


class _ReadOnlyConf:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values

    def get(self, name: str) -> str:
        return self.values[name]

    def set(self, name: str, value: str) -> None:
        raise AssertionError(f"live Spark configuration was mutated: {name}={value}")


class _Spark:
    def __init__(self, values: dict[str, str]) -> None:
        self.conf = _ReadOnlyConf(values)


def test_final_corpus_spark_config_is_startup_only_and_read_only() -> None:
    expected = dict(FINAL_CORPUS_SPARK_CONFIG)
    assert verify_spark_configuration(_Spark(expected), FINAL_CORPUS_SPARK_CONFIG) == (
        expected
    )
    rendered = glue_conf_argument(FINAL_CORPUS_SPARK_CONFIG)
    assert rendered.startswith("spark.sql.parquet.enableVectorizedReader=false")
    assert " --conf spark.speculation=false" in rendered


def test_final_corpus_spark_config_fails_closed_on_runtime_drift() -> None:
    changed = dict(FINAL_CORPUS_SPARK_CONFIG)
    changed["spark.speculation"] = "true"
    with pytest.raises(RuntimeError, match="startup configuration changed"):
        verify_spark_configuration(_Spark(changed), FINAL_CORPUS_SPARK_CONFIG)


def test_final_corpus_glue_scripts_never_mutate_live_spark_configuration() -> None:
    for script in (
        "run_glue_final_corpus_canary.py",
        "run_glue_final_corpus_production.py",
    ):
        source = (ROOT / "scripts" / script).read_text(encoding="utf-8")
        assert "spark.conf.set" not in source
        assert "verify_spark_configuration" in source


@pytest.mark.parametrize(
    "config",
    [(), (("not-spark.key", "value"),), (("spark.key", "bad value"),)],
)
def test_glue_conf_argument_rejects_invalid_startup_config(
    config: tuple[tuple[str, str], ...],
) -> None:
    with pytest.raises(ValueError, match="configuration"):
        glue_conf_argument(config)
