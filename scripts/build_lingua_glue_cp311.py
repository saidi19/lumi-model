#!/usr/bin/env python3
"""Build and verify Lumi's exact Lingua 2.2.0 wheel for AWS Glue 5.1."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from lumi.lingua_glue_compat import (
    load_lingua_glue_config,
    patch_lingua_pyproject,
    sha256_file,
    validate_lingua_wheel,
)


def _run(*command: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    return completed.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    config = load_lingua_glue_config(args.config)
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    build_root = Path(tempfile.mkdtemp(prefix="lumi-lingua-glue-"))
    source = build_root / "lingua-rs"
    build_venv = build_root / "build-venv"
    test_venv = build_root / "test-venv"
    upstream = config["upstream"]

    _run(
        "git",
        "clone",
        "--branch",
        str(upstream["tag"]),
        "--depth",
        "1",
        str(upstream["repository"]),
        str(source),
    )
    if _run("git", "rev-parse", "HEAD", cwd=source).splitlines()[-1] != upstream["commit"]:
        raise RuntimeError("Lingua source commit changed")
    if sha256_file(source / "Cargo.toml") != upstream["cargo_toml_sha256"]:
        raise RuntimeError("Lingua Cargo.toml changed")
    if sha256_file(source / "Cargo.lock") != upstream["cargo_lock_sha256"]:
        raise RuntimeError("Lingua Cargo.lock changed")
    patch_report = patch_lingua_pyproject(source / "pyproject.toml", config)

    _run(sys.executable, "-m", "venv", str(build_venv))
    build_python = build_venv / "bin" / "python"
    _run(
        str(build_python),
        "-m",
        "pip",
        "install",
        f"maturin=={config['build']['maturin_version']}",
    )
    maturin = build_venv / "bin" / "maturin"
    wheel_directory = build_root / "wheels"
    _run(
        str(maturin),
        "build",
        "--locked",
        "--release",
        "--compatibility",
        str(config["wheel"]["compatibility"]),
        "--auditwheel",
        "check",
        "--interpreter",
        sys.executable,
        "--out",
        str(wheel_directory),
        cwd=source,
    )
    wheels = list(wheel_directory.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Lingua build did not produce exactly one wheel")
    wheel_report = validate_lingua_wheel(wheels[0], config)
    packaged_wheel = output / wheels[0].name
    shutil.copyfile(wheels[0], packaged_wheel)
    if sha256_file(packaged_wheel) != wheel_report["sha256"]:
        raise RuntimeError("Lingua wheel changed while packaging")

    _run(sys.executable, "-m", "venv", str(test_venv))
    test_python = test_venv / "bin" / "python"
    _run(
        str(test_python),
        "-m",
        "pip",
        "install",
        f"pytest=={config['build']['pytest_version']}",
        str(packaged_wheel),
    )
    _run(str(test_python), "-m", "pytest", "-q", "tests/python", cwd=source)
    parity_report_path = output / "lingua-glue-cp311-parity-v0.1.json"
    parity_script = Path(__file__).with_name("verify_lingua_runtime_parity.py")
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment["PYTHONPATH"] = str(source_root)
    completed = subprocess.run(
        (
            str(test_python),
            str(parity_script),
            "--config",
            str(args.config.resolve()),
            "--language-model-root",
            str(source / "language-models"),
            "--output",
            str(parity_report_path),
        ),
        check=True,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="", flush=True)
    parity_report = json.loads(parity_report_path.read_text())

    report = {
        "build": {
            "maturin": _run(str(maturin), "--version").splitlines()[-1],
            "python": _run(sys.executable, "--version").splitlines()[-1],
            "rustc": _run("rustc", "--version").splitlines()[-1],
        },
        "built_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "gate": "lumi_lingua_glue_cp311_build_v0.1",
        "parity": parity_report,
        "patch": patch_report,
        "schema_version": 1,
        "source": {
            "commit": upstream["commit"],
            "repository": upstream["repository"],
            "tag": upstream["tag"],
        },
        "status": "verified",
        "wheel": wheel_report,
    }
    report_path = output / "lingua-glue-cp311-build-v0.1.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
