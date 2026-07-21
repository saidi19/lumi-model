from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from lumi.lingua_glue_compat import (
    load_lingua_glue_config,
    patch_lingua_pyproject,
    validate_lingua_wheel,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "lingua-glue-cp311-v0.1.json"


def test_lingua_glue_config_is_pinned_and_non_production() -> None:
    config = load_lingua_glue_config(CONFIG_PATH)
    assert config["upstream"]["commit"] == "77e94d1a2a8e49bed873a7c502bcb51ba803ef0f"
    assert config["wheel"]["expected_tag"] == "cp311-cp311-manylinux_2_34_x86_64"
    assert config["build"]["maximum_build_minutes"] == 30
    assert config["build"]["maximum_compute_usd"] == "0.60"


def test_lingua_pyproject_patch_changes_only_the_python_floor(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text())
    original = b'header\nrequires-python = ">=3.12"\nfooter\n'
    config["patch"]["expected_original_pyproject_sha256"] = hashlib.sha256(
        original
    ).hexdigest()
    modified = original.replace(b'">=3.12"', b'">=3.11"')
    config["patch"]["expected_patched_pyproject_sha256"] = hashlib.sha256(
        modified
    ).hexdigest()
    path = tmp_path / "pyproject.toml"
    path.write_bytes(original)
    report = patch_lingua_pyproject(path, config)
    assert path.read_bytes() == modified
    assert report == {
        "original_sha256": hashlib.sha256(original).hexdigest(),
        "patched_sha256": hashlib.sha256(modified).hexdigest(),
    }
    with pytest.raises(ValueError, match="upstream pyproject"):
        patch_lingua_pyproject(path, config)


def test_lingua_wheel_metadata_is_strictly_validated(tmp_path: Path) -> None:
    config = load_lingua_glue_config(CONFIG_PATH)
    filename = "lingua_language_detector-2.2.0-cp311-cp311-manylinux_2_34_x86_64.whl"
    path = tmp_path / filename
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "lingua_language_detector-2.2.0.dist-info/METADATA",
            "Metadata-Version: 2.4\n"
            "Name: lingua-language-detector\n"
            "Version: 2.2.0\n"
            "Requires-Python: >=3.11\n",
        )
        archive.writestr(
            "lingua_language_detector-2.2.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\n"
            "Root-Is-Purelib: false\n"
            "Tag: cp311-cp311-manylinux_2_34_x86_64\n",
        )
    report = validate_lingua_wheel(path, config)
    assert report["filename"] == filename
    assert report["requires_python"] == ">=3.11"
    assert report["tags"] == ["cp311-cp311-manylinux_2_34_x86_64"]
