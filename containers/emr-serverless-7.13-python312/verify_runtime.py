"""Fail the image build unless Lumi's Python 3.12 runtime is exact and importable."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform

EXPECTED = {
    "boto3": "1.43.51",
    "datasketch": "2.0.0",
    "lingua-language-detector": "2.2.0",
    "numpy": "2.2.6",
    "scipy": "1.16.3",
    "tokenizers": "0.22.2",
}


def main() -> None:
    if platform.python_version_tuple()[:2] != ("3", "12"):
        raise RuntimeError(f"Lumi EMR image requires Python 3.12, got {platform.python_version()}")
    versions = {name: importlib.metadata.version(name) for name in EXPECTED}
    if versions != EXPECTED:
        raise RuntimeError(f"Lumi EMR dependency versions changed: {versions}")
    for module in ("boto3", "datasketch", "lingua", "numpy", "scipy", "tokenizers"):
        importlib.import_module(module)

    from lingua import Language, LanguageDetectorBuilder

    detector = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.BENGALI).build()
    detected = detector.detect_language_of("This sentence verifies the English detector runtime.")
    if detected != Language.ENGLISH:
        raise RuntimeError("Lumi EMR Lingua smoke check failed")
    print(
        json.dumps(
            {"python": platform.python_version(), "status": "pass", "versions": versions},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
