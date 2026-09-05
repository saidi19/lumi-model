#!/usr/bin/env python3
"""Seal a local-only verification receipt for Lumi's safe Markdown/math renderer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "lumi.local-ui-renderer-verification.v0.1"
KATEX_VERSION = "0.18.1"
EXPECTED_KATEX_SHA256 = {
    "LICENSE": "766ccc1f306c885aa45542a9846bbd0a505b27a0374f146778171c2254ce18e3",
    "katex.min.css": "0fb711c9c74cb1718661933948b653fbc09a627da5dde8926b4d10585370993e",
    "katex.min.js": "68b9115510b8cedb9909a10de7799c94c0707481296f755c0a8888cb8fcde216",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite renderer receipt: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/lumi-4b-local-ui-renderer-verification-v0.1.json"),
    )
    args = parser.parse_args()

    root = args.repository_root.resolve()
    ui = root / "src" / "lumi" / "local_ui"
    vendor = ui / "vendor" / "katex"
    files = {
        "app": ui / "app.js",
        "index": ui / "index.html",
        "local_api": root / "src" / "lumi" / "local_api.py",
        "local_server_gate": root / "src" / "lumi" / "local_server_gate.py",
        "package_config": root / "pyproject.toml",
        "renderer": ui / "renderer.js",
        "styles": ui / "styles.css",
    }
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing renderer binding: {name}")

    for name, expected in EXPECTED_KATEX_SHA256.items():
        if file_sha256(vendor / name) != expected:
            raise ValueError(f"vendored KaTeX asset changed: {name}")

    html = files["index"].read_text(encoding="utf-8")
    renderer = files["renderer"].read_text(encoding="utf-8")
    app = files["app"].read_text(encoding="utf-8")
    styles = files["styles"].read_text(encoding="utf-8")
    local_api = files["local_api"].read_text(encoding="utf-8")
    package_config = files["package_config"].read_text(encoding="utf-8")
    checks = {
        "app_uses_renderer_with_text_fallback": (
            "window.LumiRenderer.renderMessage" in app
            and "content.textContent = message.content" in app
        ),
        "csp_keeps_scripts_self_only": (
            'script-src \'self\'' in local_api
            and "style-src-attr 'unsafe-inline'" in local_api
        ),
        "katex_is_local_and_ordered": (
            "https://" not in html
            and html.index("/vendor/katex/katex.min.js") < html.index("/renderer.js")
            and html.index("/renderer.js") < html.index("/app.js")
        ),
        "katex_trust_disabled": "trust: false" in renderer,
        "malformed_math_has_text_fallback": (
            "throwOnError: true" in renderer
            and "element.textContent = displayMode" in renderer
        ),
        "message_html_is_never_injected": (
            ".innerHTML" not in renderer and "createTextNode" in renderer
        ),
        "nested_assets_are_packaged": (
            '"local_ui/vendor/katex/*"' in package_config
            and '"local_ui/vendor/katex/fonts/*"' in package_config
        ),
        "renderer_styles_present": all(
            selector in styles
            for selector in (
                ".message-content pre",
                ".message-content .math-inline",
                ".message-content .math-block",
                ".message-content .math-fallback",
            )
        ),
        "unsafe_math_extensions_disabled": "trust: false" in renderer,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"renderer verification failed: {failed}")

    fonts = sorted((vendor / "fonts").iterdir())
    if len(fonts) != 60 or not all(path.is_file() for path in fonts):
        raise ValueError("vendored KaTeX font inventory changed")
    font_inventory = [
        {"bytes": path.stat().st_size, "name": path.name, "sha256": file_sha256(path)}
        for path in fonts
    ]
    report: dict[str, Any] = {
        "assets": {
            name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for name, path in sorted(files.items())
        },
        "browser_probe": {
            "required": True,
            "sealed_by_this_cpu_receipt": False,
        },
        "checks": checks,
        "font_inventory": {
            "files": len(font_inventory),
            "inventory_sha256": value_sha256(font_inventory),
            "total_bytes": sum(item["bytes"] for item in font_inventory),
        },
        "katex": {
            "asset_sha256": EXPECTED_KATEX_SHA256,
            "license": "MIT",
            "runtime_uses_cdn": False,
            "version": KATEX_VERSION,
        },
        "observed_at": "2026-08-06",
        "production_model_or_dataset_changed": False,
        "schema_version": SCHEMA_VERSION,
        "status": "verified_local_safe_markdown_math_renderer",
        "verifier_file_sha256": file_sha256(Path(__file__).resolve()),
    }
    report["report_sha256"] = value_sha256(report)
    output = args.output if args.output.is_absolute() else root / args.output
    write_json_atomic(output.resolve(), report)
    print(
        "PASS: Lumi local renderer verified "
        f"katex={KATEX_VERSION} fonts={len(fonts)} report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
