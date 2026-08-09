#!/usr/bin/env python3
"""Download only the three exact authorized Qwen3-14B paired-A/B objects."""

from __future__ import annotations

import argparse
import hashlib
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path

from lumi.posttraining_stronger_teacher_ab import (
    artifact_root,
    load_plan,
    sha256_value,
    verify_candidate_objects,
)
from lumi.posttraining_stronger_teacher_ab_preflight import REMOTE_OBJECT_BYTES
from lumi.posttraining_first_party_scale_canary import _write_json, file_sha256  # noqa: PLC2701


def _download_one(item: dict, destination: Path) -> int:
    if destination.is_file():
        if destination.stat().st_size != item["bytes"]:
            raise ValueError(f"existing object size changed: {destination.name}")
        if file_sha256(destination) != item["sha256"]:
            raise ValueError(f"existing object hash changed: {destination.name}")
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial")
    offset = partial.stat().st_size if partial.is_file() else 0
    if offset > item["bytes"]:
        raise ValueError(f"partial object exceeds bound: {destination.name}")
    digest = hashlib.sha256()
    if offset:
        with partial.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
                digest.update(chunk)
    headers = {"User-Agent": "Lumi-Private-Teacher-AB/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(item["url"], headers=headers, method="GET")
    try:
        context = ssl.create_default_context(cafile="/etc/ssl/cert.pem")
        response = urllib.request.urlopen(  # noqa: S310
            request, timeout=180, context=context
        )
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"authorized object download failed: {error}") from error
    status = getattr(response, "status", response.getcode())
    if offset and status != 206:
        response.close()
        raise RuntimeError("server did not honor the bounded resume range")
    if not offset and status != 200:
        response.close()
        raise RuntimeError(f"unexpected object response status: {status}")
    content_length = response.headers.get("Content-Length")
    expected_remaining = item["bytes"] - offset
    if content_length is None or int(content_length) != expected_remaining:
        response.close()
        raise RuntimeError("candidate object Content-Length changed")
    received = 0
    mode = "ab" if offset else "xb"
    try:
        with partial.open(mode) as handle, response:
            while True:
                chunk = response.read(8 * 1024**2)
                if not chunk:
                    break
                received += len(chunk)
                if received > expected_remaining:
                    raise RuntimeError("candidate object exceeded its byte bound")
                digest.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        raise
    if received != expected_remaining or partial.stat().st_size != item["bytes"]:
        raise RuntimeError("candidate object download ended short")
    if digest.hexdigest() != item["sha256"]:
        raise RuntimeError("candidate object checksum mismatch")
    os.replace(partial, destination)
    return received


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan_path = args.plan.resolve()
    plan = load_plan(plan_path, repository_root=root)
    artifact = artifact_root(plan, root)
    received_this_invocation = 0
    for item in plan["candidate_objects"]:
        destination = (artifact / item["path"]).resolve()
        received_this_invocation += _download_one(dict(item), destination)
        print(
            f"verified {destination.name} bytes={item['bytes']} sha256={item['sha256']}",
            flush=True,
        )
    objects = verify_candidate_objects(plan, repository_root=root)
    if sum(item["bytes"] for item in objects) != REMOTE_OBJECT_BYTES:
        raise ValueError("downloaded object byte total changed")
    receipt = {
        "network_payload_bytes_bound": REMOTE_OBJECT_BYTES,
        "objects": objects,
        "plan_file_sha256": file_sha256(plan_path),
        "received_this_invocation": received_this_invocation,
        "schema_version": "lumi.sft-stronger-teacher-ab-download.v0.1",
        "status": "exact_three_candidate_objects_downloaded_and_verified",
    }
    receipt["receipt_sha256"] = sha256_value(receipt)
    _write_json(artifact / "evidence" / "download-receipt.json", receipt)
    print(
        f"PASS objects=3 payload_bytes={REMOTE_OBJECT_BYTES} receipt={receipt['receipt_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
