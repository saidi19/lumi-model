#!/usr/bin/env python3
"""Build or inspect Lumi-1B source-index and held-out-safe packed artifacts."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from lumi.configuration_lumi import LumiConfig
from lumi.scaling_data import (
    INDEX_DATABASE_FILENAME,
    INDEX_MANIFEST_FILENAME,
    PACK_STATE_FILENAME,
    build_scaling_packed_dataset,
    build_scaling_source_index,
    create_scaling_data_contract,
    validate_scaling_packed_manifest,
    validate_scaling_source_index,
)
from lumi.scaling_pilot import load_scaling_pilot_plan


def _common_evidence(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", type=Path, default=Path("configs/scaling-pilot-1b-v0.1.json"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--heldout-selection-ledger", type=Path, required=True)
    parser.add_argument("--token-count-report", type=Path, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index")
    _common_evidence(index)
    index.add_argument("--output", type=Path, required=True)
    index.add_argument("--checkpoint-documents", type=int, default=256)
    index.add_argument("--checkpoint-text-bytes", type=int, default=16 * 1024 * 1024)
    index.add_argument("--progress-documents", type=int, default=10_000)
    index.add_argument("--resume", action="store_true")

    preflight = commands.add_parser("preflight")
    _common_evidence(preflight)

    pack = commands.add_parser("pack")
    pack.add_argument("--index-manifest", type=Path, required=True)
    pack.add_argument("--input", type=Path, required=True)
    pack.add_argument("--tokenizer", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--target-tokens", type=int, required=True)
    pack.add_argument("--sequence-length", type=int, default=4096)
    pack.add_argument("--sequences-per-shard", type=int, default=256)
    pack.add_argument("--checkpoint-documents", type=int, default=256)
    pack.add_argument("--checkpoint-text-bytes", type=int, default=16 * 1024 * 1024)
    pack.add_argument("--progress-documents", type=int, default=10_000)
    pack.add_argument("--purpose", choices=("canary", "final"), default="canary")
    pack.add_argument("--resume", action="store_true")

    validate = commands.add_parser("validate")
    validate.add_argument("--index-manifest", type=Path, required=True)
    validate.add_argument("--packed-manifest", type=Path)

    status = commands.add_parser("status")
    status.add_argument("--output", type=Path, required=True)
    status.add_argument("--json", action="store_true")
    return parser.parse_args()


def _contract(args: argparse.Namespace) -> dict:
    plan = load_scaling_pilot_plan(args.plan)
    model_path = args.plan.parent / plan["model"]["config"]
    config = LumiConfig.from_json_file(model_path)
    return create_scaling_data_contract(
        input_path=args.input,
        tokenizer_path=args.tokenizer,
        source_manifest_path=args.source_manifest,
        heldout_manifest_path=args.heldout_manifest,
        heldout_selection_ledger_path=args.heldout_selection_ledger,
        token_count_report_path=args.token_count_report,
        seed=plan["data"]["selection"]["seed"],
        final_target_tokens=plan["data"]["selection"]["target_tokens_requested"],
        sequence_length=plan["data"]["sequence_length"],
        expected_vocab_size=config.vocab_size,
        expected_doc_token_id=config.doc_token_id,
        plan_path=args.plan,
    )


def _read_state(path: Path) -> dict | None:
    if not path.is_file():
        return None
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        row = connection.execute("SELECT value FROM metadata WHERE key = 'state'").fetchone()
        return None if row is None else json.loads(row[0])
    finally:
        connection.close()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        contract = _contract(args)
        print(
            f"PASS: source={contract['source']['documents']:,} documents; "
            f"observed={contract['token_count_report']['observed_text_tokens']:,} tokens; "
            f"heldout={contract['heldout']['documents']:,}; "
            f"strata={len(contract['selection']['strata'])}"
        )
    elif args.command == "index":
        manifest = build_scaling_source_index(
            contract=_contract(args),
            output=args.output,
            checkpoint_documents=args.checkpoint_documents,
            checkpoint_text_bytes=args.checkpoint_text_bytes,
            progress_documents=args.progress_documents,
            resume=args.resume,
        )
        statistics = manifest["statistics"]
        print(
            f"PASS: indexed {statistics['documents']:,} documents; "
            f"eligible={statistics['eligible_tokens']:,} tokens; "
            f"heldout_exact={statistics['heldout_exact_documents']:,}"
        )
    elif args.command == "pack":
        manifest = build_scaling_packed_dataset(
            index_manifest_path=args.index_manifest,
            input_path=args.input,
            tokenizer_path=args.tokenizer,
            output=args.output,
            target_tokens=args.target_tokens,
            sequence_length=args.sequence_length,
            sequences_per_shard=args.sequences_per_shard,
            checkpoint_documents=args.checkpoint_documents,
            checkpoint_text_bytes=args.checkpoint_text_bytes,
            progress_documents=args.progress_documents,
            purpose=args.purpose,
            resume=args.resume,
        )
        print(
            f"PASS: packed {manifest['statistics']['tokens']:,} tokens in "
            f"{manifest['statistics']['sequences']:,} sequences"
        )
    elif args.command == "validate":
        if args.packed_manifest is None:
            manifest = validate_scaling_source_index(args.index_manifest)
            print(
                f"PASS: source index has {manifest['statistics']['eligible_tokens']:,} "
                "eligible tokens"
            )
        else:
            manifest = validate_scaling_packed_manifest(
                args.packed_manifest,
                index_manifest_path=args.index_manifest,
            )
            print(f"PASS: packed artifact has {manifest['statistics']['tokens']:,} tokens")
    else:
        manifest_path = args.output / INDEX_MANIFEST_FILENAME
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if args.json:
                print(json.dumps({"status": "complete", "statistics": manifest["statistics"]}))
            else:
                statistics = manifest["statistics"]
                documents = statistics.get(
                    "documents", statistics.get("documents_packed")
                )
                tokens = statistics.get("eligible_tokens", statistics.get("tokens"))
                print(
                    f"complete: documents={documents:,} "
                    f"tokens={tokens:,}"
                )
            return
        state = _read_state(args.output / INDEX_DATABASE_FILENAME)
        kind = "index"
        if state is None:
            state = _read_state(args.output / PACK_STATE_FILENAME)
            kind = "pack"
        if state is None:
            raise FileNotFoundError(f"no scaling-data state found in {args.output}")
        if args.json:
            print(json.dumps({"kind": kind, "state": state}, ensure_ascii=False))
        elif kind == "index":
            eligible = sum(item["eligible_tokens"] for item in state["strata"].values())
            heldout = sum(item["heldout_exact_documents"] for item in state["strata"].values())
            print(
                f"{state['status']}: documents={state['documents_seen']:,} "
                f"text_tokens={state['source_text_tokens']:,} "
                f"eligible_tokens={eligible:,} heldout_exact={heldout:,}"
            )
        else:
            print(
                f"{state['status']}: documents={state['documents_packed']:,} "
                f"tokens={state['tokens_written']:,}"
            )


if __name__ == "__main__":
    main()
