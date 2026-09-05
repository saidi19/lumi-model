#!/usr/bin/env python3
"""Materialize only the exact approved private Aya Bengali train/validation split."""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

from lumi.posttraining_aya_validation_materialization import (
    MAXIMUM_PRIVATE_WRITE_BYTES,
    MAXIMUM_RUNTIME_MINUTES,
    OUTPUT_FILENAMES,
    build_artifact_entry,
    build_materialization_receipt,
    build_materialized_split,
    json_file_bytes,
    jsonl_file_bytes,
    load_exact_materialization_inputs,
    resolve_private_root,
    validate_materialization_receipt,
    verify_materialized_output,
)
from lumi.posttraining_first_party import file_sha256


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    started = time.monotonic()
    root = args.repository_root.resolve()
    implementation_path = root / "src/lumi/posttraining_aya_validation_materialization.py"
    runner_path = root / "scripts/materialize_aya_validation_split.py"
    private_root = resolve_private_root(root)
    if private_root.exists():
        raise FileExistsError(f"refusing to overwrite Aya private split: {private_root}")

    plan, _report, rows, attribution, template, tokenizer = load_exact_materialization_inputs(
        root
    )
    candidate, manifest, attribution_binding, projection = build_materialized_split(
        rows,
        attribution,
        plan=plan,
        template=template,
        tokenizer=tokenizer,
    )
    payloads = {
        "attribution_binding": json_file_bytes(attribution_binding),
        "candidate": jsonl_file_bytes(candidate),
        "manifest": jsonl_file_bytes(manifest),
    }
    logical_paths = {
        name: private_root / OUTPUT_FILENAMES[name] for name in payloads
    }
    artifacts = {
        name: build_artifact_entry(
            logical_path=logical_paths[name],
            payload=payload,
            repository_root=root,
            contains_raw_training_text=name == "candidate",
        )
        for name, payload in payloads.items()
    }
    receipt = build_materialization_receipt(
        plan=plan,
        projection=projection,
        artifacts=artifacts,
        implementation_file_sha256=file_sha256(implementation_path),
        runner_file_sha256=file_sha256(runner_path),
    )
    validate_materialization_receipt(receipt)
    payloads["receipt"] = json_file_bytes(receipt)
    if sum(map(len, payloads.values())) > MAXIMUM_PRIVATE_WRITE_BYTES:
        raise ValueError("Aya split payload exceeds the approved 16 MiB write bound")
    if time.monotonic() - started > MAXIMUM_RUNTIME_MINUTES * 60:
        raise TimeoutError("Aya split preparation exceeded the approved runtime")

    private_root.parent.mkdir(parents=True, exist_ok=True)
    staging = private_root.parent / f".{private_root.name}.staging-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"Aya split staging path already exists: {staging}")
    staging.mkdir()
    try:
        for name, payload in payloads.items():
            _write_bytes(staging / OUTPUT_FILENAMES[name], payload)
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        staging.replace(private_root)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    verification = verify_materialized_output(
        private_root,
        repository_root=root,
        implementation_path=implementation_path,
        runner_path=runner_path,
    )
    if time.monotonic() - started > MAXIMUM_RUNTIME_MINUTES * 60:
        raise TimeoutError("Aya split verification exceeded the approved runtime")
    print(
        "PASS: exact private Aya Bengali split materialized; "
        "train=1330/520269 validation=148/57184 production_promotions=0 "
        f"bytes={verification['total_private_bytes']} "
        f"receipt={verification['receipt_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
