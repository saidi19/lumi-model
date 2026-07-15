"""Deterministic, CPU-only quality gates for Lumi corpus construction.

The gates in this module are intentionally independent from model training.  A
sample can be labelled, decontaminated, and deduplicated on a laptop before any
GPU is started.  Benchmark registries contain hashes only; raw evaluation text
must never be copied into this repository.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
MATCH_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_for_matching(text: str) -> str:
    """Normalize text for deduplication and contamination matching."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def matching_tokens(text: str) -> list[str]:
    return MATCH_TOKEN_PATTERN.findall(normalize_for_matching(text))


def _window_hashes(tokens: list[str], width: int) -> set[str]:
    if len(tokens) < width:
        return set()
    return {
        sha256_bytes(" ".join(tokens[index : index + width]).encode("utf-8"))
        for index in range(len(tokens) - width + 1)
    }


def benchmark_signature_entry(
    *,
    benchmark: str,
    example_id: str,
    split: str,
    text: str,
    anchor_ngram_tokens: int,
    max_anchors: int,
    minimum_anchor_matches: int,
) -> dict[str, Any]:
    """Create one raw-text-free benchmark signature entry."""

    normalized = normalize_for_matching(text)
    if not normalized:
        raise ValueError("benchmark example text cannot be empty")
    anchors = sorted(_window_hashes(matching_tokens(normalized), anchor_ngram_tokens))[
        :max_anchors
    ]
    return {
        "anchor_sha256": anchors,
        "benchmark": benchmark,
        "example_id": example_id,
        "exact_sha256": sha256_bytes(normalized.encode("utf-8")),
        "minimum_anchor_matches": min(minimum_anchor_matches, len(anchors)),
        "split": split,
    }


def validate_quality_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("quality.schema_version must be 1")

    language = config.get("language")
    if not isinstance(language, Mapping):
        raise ValueError("quality.language must be an object")
    if language.get("backend") != "lingua":
        raise ValueError("quality.language.backend must be lingua")
    if not isinstance(language.get("package_version"), str):
        raise ValueError("quality.language.package_version must be pinned")
    allowed = language.get("allowed_languages")
    if not isinstance(allowed, list) or not allowed:
        raise ValueError("quality.language.allowed_languages must be a non-empty list")
    if not all(isinstance(code, str) and re.fullmatch(r"[a-z]{2}", code) for code in allowed):
        raise ValueError("allowed language labels must be lowercase ISO 639-1 codes")
    confidence = language.get("minimum_confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise ValueError("quality.language.minimum_confidence must be numeric")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("quality.language.minimum_confidence must be between zero and one")
    if int(language.get("max_characters", 0)) < 256:
        raise ValueError("quality.language.max_characters must be at least 256")
    bypass = language.get("bypass_categories", [])
    if not isinstance(bypass, list) or not all(isinstance(item, str) for item in bypass):
        raise ValueError("quality.language.bypass_categories must be a list of strings")

    near_duplicate = config.get("near_duplicate")
    if not isinstance(near_duplicate, Mapping):
        raise ValueError("quality.near_duplicate must be an object")
    if near_duplicate.get("algorithm") != "minhash_lsh":
        raise ValueError("quality.near_duplicate.algorithm must be minhash_lsh")
    if near_duplicate.get("package_version") != "2.0.0":
        raise ValueError("quality near-dedup currently requires datasketch 2.0.0")
    if near_duplicate.get("scheme") not in {"affine32", "affine64"}:
        raise ValueError("quality.near_duplicate.scheme must be affine32 or affine64")
    threshold = near_duplicate.get("threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ValueError("quality.near_duplicate.threshold must be numeric")
    if not 0.5 <= float(threshold) < 1.0:
        raise ValueError("quality.near_duplicate.threshold must be in [0.5, 1.0)")
    if int(near_duplicate.get("num_perm", 0)) < 32:
        raise ValueError("quality.near_duplicate.num_perm must be at least 32")
    if not 2 <= int(near_duplicate.get("shingle_tokens", 0)) <= 16:
        raise ValueError("quality.near_duplicate.shingle_tokens must be between 2 and 16")
    if int(near_duplicate.get("seed", 0)) < 1:
        raise ValueError("quality.near_duplicate.seed must be positive")

    benchmark = config.get("benchmark_decontamination")
    if not isinstance(benchmark, Mapping):
        raise ValueError("quality.benchmark_decontamination must be an object")
    registry = benchmark.get("registry")
    if not isinstance(registry, str) or not registry.endswith((".jsonl", ".jsonl.gz")):
        raise ValueError("benchmark registry must be a JSONL or JSONL.GZ path")
    registry_sha256 = benchmark.get("registry_sha256")
    if not isinstance(registry_sha256, str) or SHA256_PATTERN.fullmatch(registry_sha256) is None:
        raise ValueError("benchmark registry must pin its SHA-256")
    if not 5 <= int(benchmark.get("anchor_ngram_tokens", 0)) <= 64:
        raise ValueError("benchmark anchor_ngram_tokens must be between 5 and 64")
    if int(benchmark.get("minimum_anchor_matches", 0)) < 1:
        raise ValueError("benchmark minimum_anchor_matches must be positive")


@dataclass(frozen=True)
class LanguageResult:
    label: str
    confidence: float
    decision: str


class LinguaLanguageIdentifier:
    """Pinned Lingua wrapper with bounded, deterministic text sampling."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        expected_version = str(config["package_version"])
        try:
            actual_version = importlib.metadata.version("lingua-language-detector")
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                "language identification requires `pip install -e '.[data]'`"
            ) from error
        if actual_version != expected_version:
            raise RuntimeError(
                "Lingua version mismatch: "
                f"quality config pins {expected_version}, installed {actual_version}"
            )

        from lingua import LanguageDetectorBuilder

        self.detector = LanguageDetectorBuilder.from_all_spoken_languages().build()
        self.allowed = set(config["allowed_languages"])
        self.minimum_confidence = float(config["minimum_confidence"])
        self.max_characters = int(config["max_characters"])
        self.bypass_categories = set(config.get("bypass_categories", []))
        self.package_version = actual_version

    def _bounded_text(self, text: str) -> str:
        if len(text) <= self.max_characters:
            return text
        third = max(self.max_characters // 3, 1)
        middle = len(text) // 2
        return "\n".join(
            (
                text[:third],
                text[middle - third // 2 : middle + third // 2],
                text[-third:],
            )
        )[: self.max_characters]

    def identify(self, text: str, category: str) -> LanguageResult:
        if category in self.bypass_categories:
            return LanguageResult("zxx", 1.0, "category_bypass")
        bounded = self._bounded_text(text)
        language = self.detector.detect_language_of(bounded)
        if language is None or language.iso_code_639_1 is None:
            return LanguageResult("und", 0.0, "undetermined")
        label = language.iso_code_639_1.name.lower()
        confidence = round(
            float(self.detector.compute_language_confidence(bounded, language)), 6
        )
        if confidence < self.minimum_confidence:
            return LanguageResult(label, confidence, "low_confidence")
        if label not in self.allowed:
            return LanguageResult(label, confidence, "disallowed")
        return LanguageResult(label, confidence, "accepted")


class BenchmarkRegistry:
    """Hashed exact and n-gram signatures for held-out evaluation examples."""

    def __init__(self, config: Mapping[str, Any], repository_root: Path) -> None:
        raw_path = Path(str(config["registry"])).expanduser()
        candidates = [raw_path] if raw_path.is_absolute() else [Path.cwd() / raw_path]
        if not raw_path.is_absolute():
            candidates.append(repository_root / raw_path)
        self.path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if self.path is None:
            raise FileNotFoundError(f"benchmark signature registry does not exist: {raw_path}")
        actual_sha256 = sha256_file(self.path)
        if actual_sha256 != config["registry_sha256"]:
            raise ValueError("benchmark signature registry SHA-256 does not match config")
        self.sha256 = actual_sha256
        self.anchor_ngram_tokens = int(config["anchor_ngram_tokens"])
        self.minimum_anchor_matches = int(config["minimum_anchor_matches"])
        self.exact: dict[str, tuple[str, str]] = {}
        self.anchor_examples: dict[str, set[tuple[str, str]]] = defaultdict(set)
        self.required_matches: dict[tuple[str, str], int] = {}
        self.benchmarks: set[str] = set()

        if self.path.name.endswith(".jsonl.gz"):
            registry_handle = gzip.open(self.path, "rt", encoding="utf-8")
        else:
            registry_handle = self.path.open(encoding="utf-8")
        with registry_handle as handle:
            try:
                metadata_record = json.loads(next(handle))
            except (StopIteration, json.JSONDecodeError) as error:
                raise ValueError("benchmark registry is empty or invalid") from error
            metadata = metadata_record.get("_meta")
            if not isinstance(metadata, Mapping) or metadata.get("schema_version") != 1:
                raise ValueError("benchmark registry must start with a v1 metadata record")
            if metadata.get("normalization") != "nfkc_casefold_whitespace_v1":
                raise ValueError("benchmark registry normalization is unsupported")
            if metadata.get("anchor_ngram_tokens") != self.anchor_ngram_tokens:
                raise ValueError("benchmark registry n-gram width does not match config")

            entries = 0
            for line_number, line in enumerate(handle, start=2):
                try:
                    entry = json.loads(line)
                    benchmark = entry["benchmark"]
                    example_id = entry["example_id"]
                    exact_hash = entry["exact_sha256"]
                    anchors = entry["anchor_sha256"]
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    raise ValueError(
                        f"invalid benchmark registry entry at line {line_number}: {error}"
                    ) from error
                if "text" in entry or "prompt" in entry:
                    raise ValueError("benchmark registry must not retain raw evaluation text")
                if not isinstance(benchmark, str) or not isinstance(example_id, str):
                    raise ValueError(f"invalid benchmark identity at line {line_number}")
                if not isinstance(exact_hash, str) or SHA256_PATTERN.fullmatch(exact_hash) is None:
                    raise ValueError(f"invalid exact benchmark hash at line {line_number}")
                if not isinstance(anchors, list) or not all(
                    isinstance(anchor, str) and SHA256_PATTERN.fullmatch(anchor)
                    for anchor in anchors
                ):
                    raise ValueError(f"invalid benchmark anchors at line {line_number}")
                key = (benchmark, example_id)
                self.exact[exact_hash] = key
                for anchor in anchors:
                    self.anchor_examples[anchor].add(key)
                required = int(entry.get("minimum_anchor_matches", 0))
                self.required_matches[key] = min(
                    max(required, self.minimum_anchor_matches), len(anchors)
                )
                self.benchmarks.add(benchmark)
                entries += 1
            if entries != metadata.get("entries"):
                raise ValueError("benchmark registry entry count does not match metadata")

    def match(self, text: str) -> tuple[str, str] | None:
        normalized = normalize_for_matching(text)
        exact_match = self.exact.get(sha256_bytes(normalized.encode("utf-8")))
        if exact_match is not None:
            return exact_match
        counts: Counter[tuple[str, str]] = Counter()
        for anchor in _window_hashes(matching_tokens(normalized), self.anchor_ngram_tokens):
            counts.update(self.anchor_examples.get(anchor, ()))
        for key, matches in counts.most_common():
            required = self.required_matches[key]
            if required > 0 and matches >= required:
                return key
        return None


class MinHashNearDuplicateIndex:
    """In-memory MinHash-LSH index suitable for bounded development samples."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        expected_version = str(config["package_version"])
        try:
            actual_version = importlib.metadata.version("datasketch")
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError("near-dedup requires `pip install -e '.[data]'`") from error
        if actual_version != expected_version:
            raise RuntimeError(
                "datasketch version mismatch: "
                f"quality config pins {expected_version}, installed {actual_version}"
            )
        from datasketch import MinHashLSH

        self.threshold = float(config["threshold"])
        self.num_perm = int(config["num_perm"])
        self.shingle_tokens = int(config["shingle_tokens"])
        self.seed = int(config["seed"])
        self.scheme = str(config["scheme"])
        self.index = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self.signatures: dict[str, Any] = {}

    def signature(self, text: str) -> Any:
        from datasketch import MinHash

        tokens = matching_tokens(text)
        if len(tokens) < self.shingle_tokens:
            shingles = [" ".join(tokens).encode("utf-8")]
        else:
            shingles = [
                " ".join(tokens[index : index + self.shingle_tokens]).encode("utf-8")
                for index in range(len(tokens) - self.shingle_tokens + 1)
            ]
        signature = MinHash(
            num_perm=self.num_perm,
            seed=self.seed,
            scheme=self.scheme,
        )
        signature.update_batch(shingles)
        return signature

    def fingerprint(self, signature: Any) -> str:
        width = 4 if self.scheme == "affine32" else 8
        values = b"".join(int(value).to_bytes(width, "big") for value in signature.hashvalues)
        return sha256_bytes(values)

    def duplicate_of(self, signature: Any) -> str | None:
        for candidate in sorted(self.index.query(signature)):
            if signature.jaccard(self.signatures[candidate]) >= self.threshold:
                return candidate
        return None

    def add(self, document_id: str, signature: Any) -> None:
        self.index.insert(document_id, signature)
        self.signatures[document_id] = signature


@dataclass(frozen=True)
class QualityEvaluation:
    rejection_reason: str | None
    metadata: dict[str, Any] | None
    near_duplicate_signature: Any | None


class DataQualityGate:
    """Compose language, contamination, and near-duplicate gates."""

    def __init__(self, config: Mapping[str, Any], repository_root: Path) -> None:
        validate_quality_config(config)
        self.config = config
        self.language = LinguaLanguageIdentifier(config["language"])
        self.benchmark = BenchmarkRegistry(
            config["benchmark_decontamination"], repository_root
        )
        self.near_duplicate = MinHashNearDuplicateIndex(config["near_duplicate"])

    def evaluate(self, text: str, category: str) -> QualityEvaluation:
        benchmark_match = self.benchmark.match(text)
        if benchmark_match is not None:
            return QualityEvaluation(
                f"benchmark_contamination:{benchmark_match[0]}", None, None
            )

        language = self.language.identify(text, category)
        if language.decision == "undetermined":
            return QualityEvaluation("language:undetermined", None, None)
        if language.decision == "low_confidence":
            return QualityEvaluation(f"language:low_confidence:{language.label}", None, None)
        if language.decision == "disallowed":
            return QualityEvaluation(f"language:disallowed:{language.label}", None, None)

        signature = self.near_duplicate.signature(text)
        duplicate_of = self.near_duplicate.duplicate_of(signature)
        if duplicate_of is not None:
            return QualityEvaluation("near_duplicate", None, None)

        metadata = {
            "benchmark_registry_sha256": self.benchmark.sha256,
            "language": {
                "backend": "lingua",
                "confidence": language.confidence,
                "decision": language.decision,
                "label": language.label,
                "package_version": self.language.package_version,
            },
            "near_duplicate": {
                "algorithm": "minhash_lsh",
                "fingerprint_sha256": self.near_duplicate.fingerprint(signature),
                "num_perm": self.near_duplicate.num_perm,
                "scheme": self.near_duplicate.scheme,
                "seed": self.near_duplicate.seed,
                "shingle_tokens": self.near_duplicate.shingle_tokens,
                "threshold": self.near_duplicate.threshold,
            },
        }
        return QualityEvaluation(None, metadata, signature)

    def accept(self, document_id: str, evaluation: QualityEvaluation) -> None:
        if evaluation.rejection_reason is not None or evaluation.near_duplicate_signature is None:
            raise ValueError("cannot accept a rejected or incomplete quality evaluation")
        self.near_duplicate.add(document_id, evaluation.near_duplicate_signature)

    def verify_and_accept_record(self, record: Mapping[str, Any], line_number: int) -> None:
        evaluation = self.evaluate(record["text"], record["provenance"]["category"])
        if evaluation.rejection_reason is not None:
            raise ValueError(
                f"quality gate {evaluation.rejection_reason} failed at line {line_number}"
            )
        if record.get("quality") != evaluation.metadata:
            raise ValueError(f"quality metadata mismatch at line {line_number}")
        self.accept(str(record["id"]), evaluation)
