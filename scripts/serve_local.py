#!/usr/bin/env python3
"""Serve a local Lumi MLX model through an OpenAI-compatible API and chat UI."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.local_api import DEFAULT_MODEL_ID, serve_local


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--no-preload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    serve_local(
        model_path=args.model,
        model_id=args.model_id,
        host=args.host,
        port=args.port,
        allow_network=args.allow_network,
        preload=not args.no_preload,
    )


if __name__ == "__main__":
    main()
