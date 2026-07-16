#!/usr/bin/env python3
"""Run the black-box Lumi local API and UI gate against a live server."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumi.local_server_gate import validate_local_server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_local_server(
        base_url=args.base_url,
        output_path=args.output,
        max_tokens=args.max_tokens,
    )
    print(
        f"PASS: model={report['model']['id']} UI/API/SSE green; "
        f"nonstream={report['nonstream']['elapsed_seconds']:.3f}s "
        f"stream={report['stream']['elapsed_seconds']:.3f}s"
    )


if __name__ == "__main__":
    main()
