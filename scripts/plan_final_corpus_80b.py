#!/usr/bin/env python3
"""Build Lumi's non-authorizing, checkpoint-bound final 80B corpus plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lumi.final_corpus import build_final_corpus_plan, final_corpus_plan_fingerprint


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", type=Path, default=Path("configs/final-corpus-80b-v0.1.json")
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _args()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("final-corpus plan directory is not empty")
    plan = build_final_corpus_plan(
        args.contract,
        repository_root=args.repository_root,
        source_commit=args.source_commit,
    )
    if final_corpus_plan_fingerprint(plan) != plan["plan_fingerprint"]:
        raise RuntimeError("final-corpus plan fingerprint changed before publication")
    plan_path = output / "plan.json"
    plan_path.write_bytes(_canonical_json_bytes(plan))
    plan_sha = _sha256(plan_path)
    manifest_path = output / "manifest.sha256"
    manifest_path.write_text(f"{plan_sha}  plan.json\n")
    print(
        json.dumps(
            {
                "manifest_sha256": _sha256(manifest_path),
                "plan_fingerprint": plan["plan_fingerprint"],
                "plan_sha256": plan_sha,
                "status": plan["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
