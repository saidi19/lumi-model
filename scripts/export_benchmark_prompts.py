#!/usr/bin/env python3
"""Export pinned benchmark prompts into a temporary canonical JSONL file.

The output contains raw held-out evaluation text and is therefore private and
temporary.  Feed it to ``build_benchmark_signatures.py`` and delete it after the
hash-only registry has been verified.  Never add the raw export to Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
REPOSITORY_ROOT = Path(__file__).parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "benchmark-sources-v0.1.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _format_choices(question: str, choices: list[str], labels: list[str] | None = None) -> str:
    labels = labels or [chr(ord("A") + index) for index in range(len(choices))]
    rendered = (
        f"{label}. {choice}" for label, choice in zip(labels, choices, strict=True)
    )
    return "\n".join([question, *rendered])


def _records_for_row(
    adapter: str,
    row: Mapping[str, Any],
    row_index: int,
) -> Iterator[tuple[str, str]]:
    if adapter == "mmlu":
        base = f"{row['subject']}:{row_index}"
        yield base, _format_choices(str(row["question"]), list(row["choices"]))
    elif adapter in {"arc", "openbookqa"}:
        question_key = "question" if adapter == "arc" else "question_stem"
        choices = row["choices"]
        yield str(row["id"]), _format_choices(
            str(row[question_key]), list(choices["text"]), list(choices["label"])
        )
    elif adapter == "hellaswag":
        yield f"{row['source_id']}:{row['ind']}", _format_choices(
            str(row["ctx"]), list(row["endings"])
        )
    elif adapter == "humaneval":
        base = str(row["task_id"])
        prompt = str(row["prompt"])
        solution = prompt + str(row["canonical_solution"])
        yield f"{base}:prompt", prompt
        yield f"{base}:complete", solution
        yield f"{base}:tests", str(row["test"])
    elif adapter == "gsm8k":
        base = str(row_index)
        question = str(row["question"])
        yield f"{base}:question", question
        yield f"{base}:complete", question + "\n" + str(row["answer"])
    elif adapter == "truthfulqa":
        base = str(row_index)
        question = str(row["question"])
        answers = [
            str(row["best_answer"]),
            *map(str, row["correct_answers"]),
            *map(str, row["incorrect_answers"]),
        ]
        yield f"{base}:question", question
        yield f"{base}:complete", "\n".join([question, *answers])
    elif adapter == "mbpp":
        base = str(row["task_id"])
        prompt = str(row["prompt"])
        code = str(row["code"])
        tests = "\n".join(map(str, row["test_list"]))
        yield f"{base}:prompt", prompt
        yield f"{base}:code", code
        yield f"{base}:tests", tests
        yield f"{base}:complete", "\n".join((prompt, code, tests))
    else:
        raise ValueError(f"unsupported benchmark adapter: {adapter}")


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("version") != "0.1":
        raise ValueError("benchmark source config version must be 0.1")
    if config.get("policy", {}).get("commit_raw_evaluation_text") is not False:
        raise ValueError("benchmark policy must prohibit committing raw evaluation text")
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("benchmark source config needs a non-empty sources list")
    identities: set[tuple[str, str]] = set()
    for source in sources:
        dataset_id = source.get("dataset_id")
        revision = source.get("revision")
        if not isinstance(dataset_id, str) or "/" not in dataset_id:
            raise ValueError("benchmark dataset_id must be owner/name")
        if not isinstance(revision, str) or REVISION_PATTERN.fullmatch(revision) is None:
            raise ValueError(f"benchmark {dataset_id} must pin a 40-character revision")
        for source_file in source.get("files", []):
            path = source_file.get("path")
            if (
                not isinstance(path, str)
                or path.startswith("/")
                or ".." in Path(path).parts
                or not path.endswith(".parquet")
            ):
                raise ValueError(f"benchmark {dataset_id} has an unsafe Parquet path")
            sha256 = source_file.get("sha256")
            if not isinstance(sha256, str) or SHA256_PATTERN.fullmatch(sha256) is None:
                raise ValueError(f"benchmark {dataset_id}/{path} must pin SHA-256")
            if int(source_file.get("bytes", 0)) <= 0:
                raise ValueError(f"benchmark {dataset_id}/{path} must pin byte size")
            identity = (dataset_id, path)
            if identity in identities:
                raise ValueError(f"duplicate benchmark file: {dataset_id}/{path}")
            identities.add(identity)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite raw benchmark export: {args.output}")
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    _validate_config(config)

    try:
        import pyarrow.parquet as parquet
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "benchmark export requires `pip install -e '.[benchmark-build]'`"
        ) from error

    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with args.output.open("x", encoding="utf-8") as output:
        for source in config["sources"]:
            local_directory = args.cache / source["dataset_id"].replace("/", "--")
            for source_file in source["files"]:
                local_path = Path(
                    hf_hub_download(
                        repo_id=source["dataset_id"],
                        repo_type="dataset",
                        revision=source["revision"],
                        filename=source_file["path"],
                        local_dir=local_directory,
                    )
                )
                if local_path.stat().st_size != source_file["bytes"]:
                    raise ValueError(f"benchmark byte-size mismatch: {source_file['path']}")
                if _sha256_file(local_path) != source_file["sha256"]:
                    raise ValueError(f"benchmark SHA-256 mismatch: {source_file['path']}")

                row_offset = 0
                parquet_file = parquet.ParquetFile(local_path)
                for batch in parquet_file.iter_batches(batch_size=1024):
                    for batch_index, row in enumerate(batch.to_pylist()):
                        row_index = row_offset + batch_index
                        benchmark = source_file.get("benchmark", source["benchmark"])
                        for example_id, text in _records_for_row(
                            source["adapter"], row, row_index
                        ):
                            record = {
                                "benchmark": benchmark,
                                "example_id": f"{source_file['split']}:{example_id}",
                                "split": source_file["split"],
                                "text": text,
                            }
                            output.write(
                                json.dumps(
                                    record,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                            rows_written += 1
                    row_offset += batch.num_rows

    print(f"PRIVATE RAW EXPORT: {args.output}")
    print(f"PASS: exported {rows_written} held-out benchmark records")
    print(f"SHA-256: {_sha256_file(args.output)}")


if __name__ == "__main__":
    main()
