from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lumi.corpus_retention import (
    MAX_ARTIFACT_BYTES,
    MAX_CATEGORY_RAW_TEXT_BYTES,
    MAX_TOTAL_RAW_TEXT_BYTES,
    retention_progress,
    run_retention_canary,
    verify_retention_canary,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "configs" / "corpus-retention-canary-v0.1.json"


def _load_contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _write_test_contract(path: Path) -> Path:
    contract = _load_contract()
    limits = {
        "cccc": 1_200,
        "stackexchange": 600,
        "ubuntu_irc": 600,
        "wikimedia": 400,
        "wikiteam": 400,
        "library_of_congress": 400,
    }
    for canary in contract["canaries"]:
        canary["raw_text_utf8_bytes_limit"] = limits[canary["component"]]
    contract["category_raw_text_utf8_bytes"] = {
        "open_web": 1_200,
        "questions_and_discussion": 1_200,
        "reference": 1_200,
    }
    contract["disk_policy"]["maximum_artifact_bytes"] = 100_000_000
    contract["disk_policy"]["minimum_free_bytes_before_start"] = 1
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


TOPICS = {
    "cccc-0": (
        "Astronomers compare the spectra of distant stars to understand temperature, motion, "
        "and chemical composition. A careful observation separates measurement from theory. "
    ),
    "cccc-1": (
        "A patient baker weighs flour, water, salt, and yeast before mixing bread dough. "
        "Fermentation develops flavor while temperature controls the pace of the process. "
    ),
    "stackexchange-0": (
        "A software engineer investigates a concurrency bug by reducing the program to a small "
        "example. Logs, invariants, and repeatable tests make the final explanation trustworthy. "
    ),
    "ubuntu_irc-0": (
        "Community volunteers help a new computer user diagnose an audio configuration problem. "
        "They ask precise questions, document commands, and explain how to reverse each change. "
    ),
    "wikimedia-0": (
        "Geologists reconstruct an ancient landscape by studying layers of sedimentary rock. "
        "Fossils and mineral grains provide independent evidence about the surrounding climate. "
    ),
    "wikiteam-0": (
        "Architectural historians preserve drawings that show how public buildings changed over "
        "time. Catalog records connect each plan with its creator, location, and historical era. "
    ),
    "library_of_congress-0": (
        "Historians examine letters, newspapers, and public records before describing a political "
        "debate. Comparing independent archives helps expose omissions and conflicting accounts. "
    ),
}


def _text(component: str, index: int) -> str:
    key = f"{component}-{index}"
    paragraph = TOPICS[key]
    suffix = (
        f" This is fixture {key}. The passage uses complete English sentences and records "
        "its assumptions so another reader can reproduce the analysis."
    )
    text = paragraph
    while len(text.encode("utf-8")) < 800:
        text += suffix
    assert 800 <= len(text.encode("utf-8")) < 1_200
    return text


def _record(component: str, index: int) -> dict[str, Any]:
    return {
        "id": f"fixture-{component}-{index}",
        "metadata": {
            "all_licenses": ["CC-BY-SA-4.0"],
            "item_url": f"https://example.invalid/items/{component}/{index}",
            "license": "CC-BY-4.0",
            "provenance": f"fixture:{component}:{index}",
            "url": f"https://example.invalid/{component}/{index}",
        },
        "source": f"fixture-{component}",
        "text": _text(component, index),
    }


def _records() -> dict[str, list[dict[str, Any]]]:
    return {
        "cccc": [_record("cccc", 0), _record("cccc", 1)],
        "stackexchange": [_record("stackexchange", 0)],
        "ubuntu_irc": [_record("ubuntu_irc", 0)],
        "wikimedia": [_record("wikimedia", 0)],
        "wikiteam": [_record("wikiteam", 0)],
        "library_of_congress": [_record("library_of_congress", 0)],
    }


def _token_counts(texts: list[str]) -> list[int]:
    return [max(1, len(text.encode("utf-8")) // 4) for text in texts]


def _iterator(records: dict[str, list[dict[str, Any]]]):
    def factory(dataset: dict, source: dict, timeout_seconds: float):
        del dataset, timeout_seconds
        yield from enumerate(records[source["component"]], start=1)

    return factory


def test_production_contract_is_bounded_and_cannot_authorize_training() -> None:
    contract = _load_contract()
    assert contract["authorization"] == {
        "bulk_corpus_materialization": False,
        "full_4b_training": False,
        "paid_compute": False,
        "stream_bounded_canaries": True,
    }
    category_limits = contract["category_raw_text_utf8_bytes"]
    assert set(category_limits.values()) == {MAX_CATEGORY_RAW_TEXT_BYTES}
    assert sum(category_limits.values()) == MAX_TOTAL_RAW_TEXT_BYTES
    assert contract["disk_policy"] == {
        "maximum_artifact_bytes": MAX_ARTIFACT_BYTES,
        "minimum_free_bytes_before_start": 20_000_000_000,
        "persist_accepted_text": False,
        "persist_raw_text": False,
    }


def test_compiled_raw_byte_ceiling_fails_closed(tmp_path: Path) -> None:
    contract = _load_contract()
    contract["canaries"][0]["raw_text_utf8_bytes_limit"] += 1
    contract["category_raw_text_utf8_bytes"]["open_web"] += 1
    changed = tmp_path / "unsafe.json"
    changed.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="per-source canary safety ceiling"):
        run_retention_canary(
            changed,
            None,
            tmp_path / "output",
            repository_root=ROOT,
            record_iterator_factory=_iterator(_records()),
            token_counter_override=_token_counts,
            allow_unverified_test_tokenizer=True,
        )


def test_interrupted_run_resumes_atomically_without_persisting_text(tmp_path: Path) -> None:
    contract = _write_test_contract(tmp_path / "contract.json")
    records = _records()

    def interrupted_factory(dataset: dict, source: dict, timeout_seconds: float):
        del dataset, timeout_seconds
        for line, record in enumerate(records[source["component"]], start=1):
            yield line, record
            if source["component"] == "cccc" and line == 1:
                raise OSError("simulated Wi-Fi interruption")

    resumed_output = tmp_path / "resumed"
    with pytest.raises(RuntimeError, match="failed streaming retention source cccc"):
        run_retention_canary(
            contract,
            None,
            resumed_output,
            repository_root=ROOT,
            retries=0,
            record_iterator_factory=interrupted_factory,
            token_counter_override=_token_counts,
            allow_unverified_test_tokenizer=True,
        )
    failed_progress = retention_progress(resumed_output)
    assert failed_progress["status"] == "failed"
    assert failed_progress["sources"][0]["next_line"] == 1

    resumed = run_retention_canary(
        contract,
        None,
        resumed_output,
        repository_root=ROOT,
        resume=True,
        retries=0,
        record_iterator_factory=_iterator(records),
        token_counter_override=_token_counts,
        allow_unverified_test_tokenizer=True,
    )
    clean_output = tmp_path / "clean"
    clean = run_retention_canary(
        contract,
        None,
        clean_output,
        repository_root=ROOT,
        retries=0,
        record_iterator_factory=_iterator(records),
        token_counter_override=_token_counts,
        allow_unverified_test_tokenizer=True,
    )

    assert resumed == clean
    assert resumed["status"] == "complete"
    assert resumed["decision"]["retention_canary"] == "fail"
    assert resumed["summary"]["tokenizer_verified"] is False
    assert verify_retention_canary(resumed_output, repository_root=ROOT) == resumed
    assert retention_progress(resumed_output)["status"] == "complete"

    distinctive_text = records["cccc"][0]["text"].encode("utf-8")
    for path in resumed_output.iterdir():
        if path.is_file():
            assert distinctive_text not in path.read_bytes()

    report_path = resumed_output / "report.json"
    report_path.write_text(report_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        verify_retention_canary(resumed_output, repository_root=ROOT)
