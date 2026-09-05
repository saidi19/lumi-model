from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi.posttraining_deepmind_dependency_resolution import (
    PYPI_METADATA,
    SETUP_OBJECT,
    audit_runtime,
    git_blob_sha1,
    load_resolution_plan,
    parse_literal_install_requires,
    validate_pypi_metadata,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/lumi-4b-sft-deepmind-dependency-resolution-v0.1.json"


def test_git_blob_sha1_uses_git_object_framing() -> None:
    assert git_blob_sha1(b"test content\n") == "d670460b4b4aece5915caf5c68d12f560a9fe3e4"


def test_setup_parser_rejects_wrong_object() -> None:
    with pytest.raises(ValueError, match="byte count"):
        parse_literal_install_requires(b"setup(install_requires=[])")


def test_runtime_audit_can_record_exact_missing_pair() -> None:
    audit = audit_runtime(
        find_spec=lambda module: None if module in {"absl", "six"} else object(),
        distribution_version=lambda name: {"numpy": "2.5.1", "sympy": "1.14.0"}[
            name
        ],
        executable="/isolated/python",
        python_version="3.13.7",
    )
    assert [item["distribution"] for item in audit["packages"] if not item["present"]] == [
        "absl-py",
        "six",
    ]


def test_pypi_contract_pins_universal_wheels() -> None:
    assert PYPI_METADATA["absl-py"]["wheel"]["filename"].endswith("none-any.whl")
    assert PYPI_METADATA["six"]["wheel"]["filename"].endswith("none-any.whl")
    assert SETUP_OBJECT["bytes"] == 2458


def test_exact_plan_loads_when_materialized() -> None:
    if not PLAN_PATH.exists():
        pytest.skip("DeepMind dependency-resolution plan is not materialized")
    plan = load_resolution_plan(PLAN_PATH, repository_root=ROOT)
    assert plan["authorization"]["package_installation_authorized"] is False
    assert plan["pypi_metadata"]["six"]["version"] == "1.17.0"


def test_live_private_inputs_match_contract_when_available() -> None:
    setup = Path("/private/tmp/lumi-deepmind-mathematics-setup.py")
    absl = Path("/private/tmp/lumi-absl-py-2.5.0.json")
    six = Path("/private/tmp/lumi-six-1.17.0.json")
    if not all(path.exists() for path in (setup, absl, six)):
        pytest.skip("private dependency-resolution inputs are not present")
    assert parse_literal_install_requires(setup.read_bytes()) == [
        "absl-py>=0.1.0",
        "numpy>=1.10",
        "six",
        "sympy>=1.2",
    ]
    assert validate_pypi_metadata("absl-py", absl.read_bytes())["version"] == "2.5.0"
    assert validate_pypi_metadata("six", six.read_bytes())["version"] == "1.17.0"


def test_json_metadata_is_not_embedded_as_raw_source() -> None:
    for package in PYPI_METADATA.values():
        assert "description" not in json.dumps(package)
