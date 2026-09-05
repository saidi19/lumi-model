from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "src" / "lumi" / "local_ui" / "renderer.js"
INDEX = ROOT / "src" / "lumi" / "local_ui" / "index.html"
APP = ROOT / "src" / "lumi" / "local_ui" / "app.js"
KATEX = ROOT / "src" / "lumi" / "local_ui" / "vendor" / "katex"
NODE = shutil.which("node")


def _node_json(expression: str):
    if NODE is None:
        pytest.skip("Node.js is required for the pure renderer tokenizer gate")
    script = (
        f"const renderer = require({json.dumps(str(RENDERER))});"
        f"console.log(JSON.stringify({expression}));"
    )
    completed = subprocess.run(  # noqa: S603
        [NODE, "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_block_tokenizer_covers_chat_markdown_code_and_math() -> None:
    source = "# Result\n\n- one\n- two\n\n```python\nprint(2 + 2)\n```\n\n$$x^2 + y^2$$"
    tokens = _node_json(f"renderer.tokenizeBlocks({json.dumps(source)})")

    assert [token["type"] for token in tokens] == [
        "heading",
        "list",
        "code_block",
        "math_block",
    ]
    assert tokens[1]["items"] == ["one", "two"]
    assert tokens[2]["language"] == "python"
    assert tokens[2]["value"] == "print(2 + 2)"
    assert tokens[3]["value"] == "x^2 + y^2"


def test_inline_tokenizer_keeps_html_inert_and_rejects_script_links() -> None:
    source = (
        '<img src=x onerror="alert(1)"> '
        "[bad](javascript:alert(1)) [good](https://example.com) $x+1$ `code`"
    )
    tokens = _node_json(f"renderer.tokenizeInline({json.dumps(source)})")

    assert not any(
        token["type"] == "link"
        and token.get("href", "").startswith("javascript:")
        for token in tokens
    )
    assert any(
        token["type"] == "link" and token["href"] == "https://example.com"
        for token in tokens
    )
    assert any(token["type"] == "math" and token["value"] == "x+1" for token in tokens)
    assert any(token["type"] == "code" and token["value"] == "code" for token in tokens)
    assert "<img src=x" in "".join(token.get("value", "") for token in tokens)


def test_renderer_is_dom_only_fail_closed_and_uses_trust_false_math() -> None:
    source = RENDERER.read_text(encoding="utf-8")
    assert ".innerHTML" not in source
    assert "createTextNode" in source
    assert 'trust: false' in source
    assert 'throwOnError: true' in source
    assert 'element.textContent = displayMode' in source


def test_ui_loads_only_local_renderer_assets_in_dependency_order() -> None:
    html = INDEX.read_text(encoding="utf-8")
    assert "https://" not in html
    assert html.index("/vendor/katex/katex.min.js") < html.index("/renderer.js")
    assert html.index("/renderer.js") < html.index("/app.js")
    assert "window.LumiRenderer.renderMessage" in APP.read_text(encoding="utf-8")
    assert (KATEX / "LICENSE").is_file()
    assert (KATEX / "katex.min.js").stat().st_size > 250_000
    assert (KATEX / "katex.min.css").stat().st_size > 20_000
    assert len(list((KATEX / "fonts").iterdir())) == 60
