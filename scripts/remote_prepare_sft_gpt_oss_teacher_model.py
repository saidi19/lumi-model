#!/usr/bin/env python3
"""Download the frozen gpt-oss model object set once and verify every byte."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

MODEL_ID = "openai/gpt-oss-120b"
MODEL_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
MODEL_OBJECT_COUNT = 25
MODEL_BYTES = 65_276_857_840
CHUNK_BYTES = 8 * 1024 * 1024


def load_manifest(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    model = value.get("model", {})
    objects = value.get("model_objects")
    if model.get("model_id") != MODEL_ID:
        raise ValueError("model id binding changed")
    if model.get("revision") != MODEL_REVISION:
        raise ValueError("model revision binding changed")
    if model.get("runtime_object_count") != MODEL_OBJECT_COUNT:
        raise ValueError("model object-count binding changed")
    if model.get("runtime_object_bytes") != MODEL_BYTES:
        raise ValueError("model byte binding changed")
    if not isinstance(objects, list) or len(objects) != MODEL_OBJECT_COUNT:
        raise ValueError("model object manifest changed")
    if sum(int(item["bytes"]) for item in objects) != MODEL_BYTES:
        raise ValueError("model object byte sum changed")
    paths = [str(item["path"]) for item in objects]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate model object path")
    for item in objects:
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe model object path")
        digest = str(item["sha256"])
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("invalid model object digest")
    return objects


def sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def download_object(
    model_dir: Path, item: dict[str, Any], stop_event: threading.Event
) -> dict[str, Any]:
    relative = Path(str(item["path"]))
    destination = model_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial")
    expected_size = int(item["bytes"])
    expected_sha256 = str(item["sha256"])
    if destination.exists():
        size, digest = sha256_file(destination)
        if size != expected_size or digest != expected_sha256:
            raise RuntimeError(f"existing object binding differs: {relative}")
        print(f"VERIFIED_EXISTING {relative} {size}", flush=True)
        return {"bytes": size, "path": str(relative), "sha256": digest}

    encoded_path = urllib.parse.quote(str(relative), safe="/")
    url = (
        f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/"
        f"{encoded_path}?download=true"
    )
    existing_size = partial.stat().st_size if partial.exists() else 0
    if existing_size > expected_size:
        raise RuntimeError(f"partial object exceeded byte binding: {relative}")
    digest = hashlib.sha256()
    if existing_size == expected_size:
        size, observed_sha256 = sha256_file(partial)
        if size != expected_size or observed_sha256 != expected_sha256:
            raise RuntimeError(f"complete partial object binding differs: {relative}")
        partial.replace(destination)
        print(f"DOWNLOAD_VERIFIED {relative} {size}", flush=True)
        return {"bytes": size, "path": str(relative), "sha256": observed_sha256}
    if existing_size:
        partial_size, partial_sha256 = sha256_file(partial)
        if partial_size != existing_size:
            raise RuntimeError(f"partial object changed while hashing: {relative}")
        with partial.open("rb") as handle:
            while chunk := handle.read(CHUNK_BYTES):
                digest.update(chunk)
        headers = {
            "Range": f"bytes={existing_size}-",
            "User-Agent": "Lumi-gpt-oss-teacher-canary-v0.1",
        }
        mode = "ab"
        expected_status = 206
        print(
            f"DOWNLOAD_RESUME {relative} {existing_size} {partial_sha256}",
            flush=True,
        )
    else:
        headers = {"User-Agent": "Lumi-gpt-oss-teacher-canary-v0.1"}
        mode = "xb"
        expected_status = 200
        print(f"DOWNLOAD_START {relative} {expected_size}", flush=True)
    request = urllib.request.Request(url, headers=headers, method="GET")
    size = existing_size
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open(mode) as handle:
            if response.status != expected_status:
                raise RuntimeError(
                    f"object HTTP status differs: {relative}: {response.status}"
                )
            if existing_size:
                expected_content_range = (
                    f"bytes {existing_size}-{expected_size - 1}/{expected_size}"
                )
                if response.headers.get("Content-Range") != expected_content_range:
                    raise RuntimeError(f"object range binding differs: {relative}")
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != expected_size - existing_size:
                raise RuntimeError(f"object content-length differs: {relative}")
            while chunk := response.read(CHUNK_BYTES):
                if stop_event.is_set():
                    raise RuntimeError(f"peer object failed; stopping: {relative}")
                size += len(chunk)
                if size > expected_size:
                    raise RuntimeError(f"object exceeded byte binding: {relative}")
                digest.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        stop_event.set()
        print(f"DOWNLOAD_STOPPED {relative} {size}", file=sys.stderr, flush=True)
        raise
    observed_sha256 = digest.hexdigest()
    if size != expected_size:
        raise RuntimeError(f"object size differs: {relative}: {size} != {expected_size}")
    if observed_sha256 != expected_sha256:
        raise RuntimeError(f"object digest differs: {relative}")
    partial.replace(destination)
    print(f"DOWNLOAD_VERIFIED {relative} {size}", flush=True)
    return {"bytes": size, "path": str(relative), "sha256": observed_sha256}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=15)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 15:
        raise ValueError("workers must remain between 1 and the 15 bound weight shards")
    objects = load_manifest(args.manifest.resolve())
    model_dir = args.model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    stop_event = threading.Event()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        completed = list(
            executor.map(
                lambda item: download_object(model_dir, item, stop_event), objects
            )
        )
    if sum(item["bytes"] for item in completed) != MODEL_BYTES:
        raise RuntimeError("verified model byte sum changed")
    receipt = {
        "download_attempts_per_object": 1,
        "model": {"bytes": MODEL_BYTES, "id": MODEL_ID, "object_count": MODEL_OBJECT_COUNT, "revision": MODEL_REVISION},
        "objects": completed,
        "schema_version": "lumi.sft-gpt-oss-teacher-model-download-receipt.v0.1",
        "status": "PASS_exact_25_objects_verified",
    }
    atomic_json(args.receipt.resolve(), receipt)
    print("MODEL_READY 25 65276857840", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
