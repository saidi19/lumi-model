from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from lumi.source_file_manifest import (
    run_source_file_manifest,
    validate_source_tree,
    verify_source_file_manifest,
)

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "configs" / "source-file-manifest-v0.1.json"


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _source_specs() -> dict[str, dict]:
    inventory = {
        row["dataset_id"]: row for row in _load("configs/component-sources-v0.1.json")["sources"]
    }
    capacity = {
        row["dataset_id"]: row
        for row in _load("configs/common-pile-capacity-snapshot-v0.1.json")["sources"]
    }
    return {
        dataset_id: {
            **row,
            "expected_bytes": capacity[dataset_id]["tree_compressed_bytes"],
            "expected_files": capacity[dataset_id]["tree_data_file_count"],
        }
        for dataset_id, row in inventory.items()
    }


def _hashes(dataset_id: str, path: str) -> tuple[str, str]:
    value = f"{dataset_id}:{path}".encode()
    return hashlib.sha1(value).hexdigest(), hashlib.sha256(value).hexdigest()  # noqa: S324


def _fake_tree(dataset_id: str, revision: str) -> list[dict]:
    source = _source_specs()[dataset_id]
    assert revision == source["revision"]
    count = source["expected_files"]
    paths = [source["schema_anchor"]]
    paths.extend(f"fixture-{index:04d}.json.gz" for index in range(1, count))
    sizes = [1] * (count - 1) + [source["expected_bytes"] - (count - 1)]
    entries = []
    for path, size in zip(paths, sizes, strict=True):
        blob_id, lfs_sha256 = _hashes(dataset_id, path)
        entries.append(
            {
                "blob_id": blob_id,
                "lfs": {"sha256": lfs_sha256, "size": size},
                "path": path,
                "size": size,
                "type": "file",
                "xet_hash": lfs_sha256,
            }
        )
    for path in (".gitattributes", "README.md"):
        blob_id, _ = _hashes(dataset_id, path)
        entries.append(
            {
                "blob_id": blob_id,
                "lfs": None,
                "path": path,
                "size": 100,
                "type": "file",
                "xet_hash": None,
            }
        )
    return entries


def _small_source() -> dict:
    return {
        "component": "fixture",
        "expected_bytes": 30,
        "expected_files": 2,
        "schema_anchor": "fixture-0000.json.gz",
    }


def _rules() -> dict:
    return _load("configs/source-file-manifest-v0.1.json")["tree_rules"]


def _small_tree() -> list[dict]:
    return [
        {
            "blob_id": "a" * 40,
            "lfs": {"sha256": "b" * 64, "size": 10},
            "path": "fixture-0000.json.gz",
            "size": 10,
            "type": "file",
            "xet_hash": "c" * 64,
        },
        {
            "blob_id": "d" * 40,
            "lfs": {"sha256": "e" * 64, "size": 20},
            "path": "fixture-0001.json.gz",
            "size": 20,
            "type": "file",
            "xet_hash": None,
        },
        {
            "blob_id": "f" * 40,
            "lfs": None,
            "path": "README.md",
            "size": 100,
            "type": "file",
            "xet_hash": None,
        },
    ]


def test_pinned_contract_keeps_metadata_only_boundary_and_exact_totals() -> None:
    contract = _load("configs/source-file-manifest-v0.1.json")
    assert contract["authorization"] == {
        "bulk_corpus_materialization": False,
        "create_cloud_resources": False,
        "fetch_public_repository_metadata": True,
        "full_4b_training": False,
        "paid_compute": False,
        "transfer_source_payload": False,
    }
    assert contract["expected_totals"] == {
        "compressed_source_bytes": 499_817_285_384,
        "data_files": 758,
        "sources": 23,
    }
    assert contract["metadata_api"]["token"] is False
    assert contract["metadata_api"]["expand"] is False


def test_repository_snapshot_contains_every_exact_source_object_once() -> None:
    snapshot = _load("configs/common-pile-source-files-v0.1.json")
    assert snapshot["summary"] == {
        "compressed_source_bytes": 499_817_285_384,
        "corpus_payload_bytes_downloaded": 0,
        "data_files": 758,
        "repository_control_files": 46,
        "sources": 23,
    }
    assert len(snapshot["sources"]) == 23
    files = [row for source in snapshot["sources"] for row in source["data_files"]]
    assert len(files) == 758
    assert sum(row["source_bytes"] for row in files) == 499_817_285_384
    digests = [row["source_digest"]["value"] for row in files]
    assert len(set(digests)) == 758
    assert all(row["source_digest"]["basis"] == "git_lfs_object" for row in files)
    cccc = next(row for row in snapshot["sources"] if row["component"] == "cccc")
    assert cccc["acquisition_status"].startswith("blocked_pending_owner_cccc")


def test_source_tree_requires_complete_lfs_identity_and_known_files() -> None:
    rows = validate_source_tree(_small_source(), _small_tree(), _rules())
    assert [row["path"] for row in rows if row["kind"] == "data"] == [
        "fixture-0000.json.gz",
        "fixture-0001.json.gz",
    ]

    missing_hash = copy.deepcopy(_small_tree())
    missing_hash[0]["lfs"]["sha256"] = None
    with pytest.raises(ValueError, match="no LFS SHA-256"):
        validate_source_tree(_small_source(), missing_hash, _rules())

    mismatched_size = copy.deepcopy(_small_tree())
    mismatched_size[0]["lfs"]["size"] = 9
    with pytest.raises(ValueError, match="LFS size differs"):
        validate_source_tree(_small_source(), mismatched_size, _rules())

    unknown = copy.deepcopy(_small_tree())
    unknown.append(
        {
            "blob_id": "1" * 40,
            "lfs": None,
            "path": "unexpected.txt",
            "size": 1,
            "type": "file",
            "xet_hash": None,
        }
    )
    with pytest.raises(ValueError, match="unknown non-data"):
        validate_source_tree(_small_source(), unknown, _rules())


def test_source_tree_rejects_count_bytes_anchor_and_unsafe_paths() -> None:
    missing = _small_tree()[:-2] + _small_tree()[-1:]
    with pytest.raises(ValueError, match="data file count differs"):
        validate_source_tree(_small_source(), missing, _rules())

    wrong_bytes = copy.deepcopy(_small_tree())
    wrong_bytes[1]["size"] = 21
    wrong_bytes[1]["lfs"]["size"] = 21
    with pytest.raises(ValueError, match="compressed byte total differs"):
        validate_source_tree(_small_source(), wrong_bytes, _rules())

    missing_anchor = copy.deepcopy(_small_tree())
    missing_anchor[0]["path"] = "fixture-0002.json.gz"
    with pytest.raises(ValueError, match="schema anchor is absent"):
        validate_source_tree(_small_source(), missing_anchor, _rules())

    unsafe = copy.deepcopy(_small_tree())
    unsafe[0]["path"] = "../fixture-0000.json.gz"
    with pytest.raises(ValueError, match="unsafe"):
        validate_source_tree(_small_source(), unsafe, _rules())


def test_full_metadata_run_is_sealed_resumable_and_cccc_stays_blocked(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    repository_snapshot = tmp_path / "repository-snapshot.json"
    calls: list[str] = []

    def fetch(dataset_id: str, revision: str) -> list[dict]:
        calls.append(dataset_id)
        return _fake_tree(dataset_id, revision)

    report = run_source_file_manifest(
        CONTRACT,
        output,
        repository_root=ROOT,
        repository_snapshot=repository_snapshot,
        fetch_tree=fetch,
        retries=0,
        retry_delay_seconds=0,
        progress=lambda _: None,
    )
    assert len(calls) == 23
    assert report["summary"]["data_files"] == 758
    assert report["summary"]["compressed_source_bytes"] == 499_817_285_384
    assert report["summary"]["corpus_payload_bytes_downloaded"] == 0
    assert report["decision"]["bulk_corpus_materialization"] == "not_authorized"
    verified = verify_source_file_manifest(
        output,
        contract_path=CONTRACT,
        repository_root=ROOT,
        repository_snapshot=repository_snapshot,
    )
    assert verified == report

    snapshot = json.loads(repository_snapshot.read_text(encoding="utf-8"))
    cccc = next(row for row in snapshot["sources"] if row["component"] == "cccc")
    assert cccc["acquisition_status"] == (
        "blocked_pending_owner_cccc_license_decision_and_production_authorization"
    )
    source_lines = (output / "source-files.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in source_lines]
    assert len(records) == 758
    assert all(record["acquisition_authorized"] is False for record in records)
    assert not any("text" in record for record in records)

    calls.clear()

    def must_not_fetch(dataset_id: str, revision: str) -> list[dict]:
        raise AssertionError(f"completed source was fetched again: {dataset_id}@{revision}")

    resumed = run_source_file_manifest(
        CONTRACT,
        output,
        repository_root=ROOT,
        repository_snapshot=repository_snapshot,
        fetch_tree=must_not_fetch,
        resume=True,
        retries=0,
        retry_delay_seconds=0,
        progress=lambda _: None,
    )
    assert resumed["summary"] == report["summary"]


def test_partial_metadata_state_resumes_without_refetching_completed_sources(
    tmp_path: Path,
) -> None:
    output = tmp_path / "partial"
    repository_snapshot = tmp_path / "repository-snapshot.json"
    initial_calls: list[str] = []

    def fail_third(dataset_id: str, revision: str) -> list[dict]:
        initial_calls.append(dataset_id)
        if len(initial_calls) == 3:
            raise OSError("fixture interruption")
        return _fake_tree(dataset_id, revision)

    with pytest.raises(RuntimeError, match="metadata fetch failed"):
        run_source_file_manifest(
            CONTRACT,
            output,
            repository_root=ROOT,
            repository_snapshot=repository_snapshot,
            fetch_tree=fail_third,
            retries=0,
            retry_delay_seconds=0,
            progress=lambda _: None,
        )
    assert len(initial_calls) == 3

    resumed_calls: list[str] = []

    def resume_fetch(dataset_id: str, revision: str) -> list[dict]:
        resumed_calls.append(dataset_id)
        return _fake_tree(dataset_id, revision)

    run_source_file_manifest(
        CONTRACT,
        output,
        repository_root=ROOT,
        repository_snapshot=repository_snapshot,
        fetch_tree=resume_fetch,
        resume=True,
        retries=0,
        retry_delay_seconds=0,
        progress=lambda _: None,
    )
    assert len(resumed_calls) == 21
    assert not set(initial_calls[:2]) & set(resumed_calls)


def test_hash_or_snapshot_tampering_fails_independent_verification(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    repository_snapshot = tmp_path / "repository-snapshot.json"
    run_source_file_manifest(
        CONTRACT,
        output,
        repository_root=ROOT,
        repository_snapshot=repository_snapshot,
        fetch_tree=_fake_tree,
        retries=0,
        retry_delay_seconds=0,
        progress=lambda _: None,
    )
    source_files = output / "source-files.jsonl"
    source_files.write_text(source_files.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash does not match"):
        verify_source_file_manifest(
            output,
            contract_path=CONTRACT,
            repository_root=ROOT,
            repository_snapshot=repository_snapshot,
        )
