#!/usr/bin/env python3
"""Run Lumi-owned safety cases against an OpenAI-compatible model endpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from lumi.posttraining_evaluation import (
    EvaluationCase,
    build_evaluation_report,
    validate_evaluation_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model-binding", type=Path, required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key-env", default="LUMI_API_KEY")
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_cases(path: Path) -> list[EvaluationCase]:
    cases = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                cases.append(EvaluationCase.from_mapping(value))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"invalid evaluation case at line {line_number}: {error}"
                ) from error
    if not cases:
        raise ValueError("evaluation case file is empty")
    return cases


def _completion(
    *,
    base_url: str,
    model: str,
    messages: tuple[dict[str, str], ...],
    api_key: str | None,
    timeout: float,
) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base URL must be absolute HTTP(S)")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "max_completion_tokens": 512,
        "messages": list(messages),
        "model": model,
        "stream": False,
        "temperature": 0,
    }
    request = Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            result = json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"evaluation request failed with HTTP {error.code}") from error
    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("endpoint returned an invalid chat completion") from error


def _write_report(path: Path, report: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    cases = _load_cases(args.cases)
    plan = validate_evaluation_plan(_load_json(args.plan))
    binding = _load_json(args.model_binding)
    if args.dry_run:
        print(f"PASS: {len(cases)} cases and {len(plan['suites'])} suites validated")
        return
    if not args.base_url or not args.model or args.output is None:
        raise ValueError("--base-url, --model, and --output are required outside --dry-run")
    api_key = os.environ.get(args.api_key_env)
    responses = {
        case.case_id: _completion(
            base_url=args.base_url,
            model=args.model,
            messages=case.messages,
            api_key=api_key,
            timeout=args.timeout_seconds,
        )
        for case in cases
    }
    report = build_evaluation_report(
        cases=cases,
        responses=responses,
        plan=plan,
        model_binding=binding,
    )
    _write_report(args.output, report)
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {report['report_sha256']}")


if __name__ == "__main__":
    main()
