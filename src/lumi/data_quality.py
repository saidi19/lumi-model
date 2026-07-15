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
import sqlite3
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
    allowed_by_category = language.get("allowed_languages_by_category", {})
    if not isinstance(allowed_by_category, Mapping):
        raise ValueError("quality.language.allowed_languages_by_category must be an object")
    for category, category_allowed in allowed_by_category.items():
        if not isinstance(category, str) or not category:
            raise ValueError("category-specific language rules need non-empty category names")
        if not isinstance(category_allowed, list) or not category_allowed:
            raise ValueError(
                f"category-specific allowed languages for {category} must be a non-empty list"
            )
        if not all(
            isinstance(code, str) and re.fullmatch(r"[a-z]{2}", code)
            for code in category_allowed
        ):
            raise ValueError(
                f"category-specific language labels for {category} must be "
                "lowercase ISO 639-1 codes"
            )
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
        self.allowed_by_category = {
            str(category): set(codes)
            for category, codes in config.get("allowed_languages_by_category", {}).items()
        }
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
        allowed = self.allowed_by_category.get(category, self.allowed)
        if label not in allowed:
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

    def duplicate_of(
        self, signature: Any, *, exclude_document_id: str | None = None
    ) -> str | None:
        for candidate in sorted(self.index.query(signature)):
            if candidate == exclude_document_id:
                continue
            if signature.jaccard(self.signatures[candidate]) >= self.threshold:
                return candidate
        return None

    def add(self, document_id: str, signature: Any) -> None:
        existing = self.signatures.get(document_id)
        if existing is not None:
            if self.fingerprint(existing) != self.fingerprint(signature):
                raise ValueError(f"near-duplicate signature changed for {document_id}")
            return
        self.index.insert(document_id, signature)
        self.signatures[document_id] = signature

    def replace(self, document_id: str, signature: Any) -> None:
        """Atomically replace one accepted document's MinHash signature."""

        existing = self.signatures.get(document_id)
        if existing is None:
            raise KeyError(f"near-duplicate signature does not exist for {document_id}")
        if self.fingerprint(existing) == self.fingerprint(signature):
            return
        self.index.remove(document_id)
        self.index.insert(document_id, signature)
        self.signatures[document_id] = signature

    def count(self) -> int:
        return len(self.signatures)

    def document_ids(self) -> set[str]:
        return set(self.signatures)

    def clear(self) -> None:
        from datasketch import MinHashLSH

        self.index = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self.signatures.clear()

    def flush(self) -> None:
        """Match the durable index interface; the in-memory backend has nothing to flush."""

    def close(self) -> None:
        """Match the durable index interface; the in-memory backend has nothing to close."""


class SQLiteMinHashNearDuplicateIndex(MinHashNearDuplicateIndex):
    """Disk-backed MinHash-LSH index for multi-gigabyte corpus construction.

    Only compact MinHash signatures and band buckets are stored. Raw document
    text never enters the database. The band layout is obtained from the pinned
    ``datasketch`` implementation, and candidates are rechecked using the full
    MinHash signature before a duplicate decision is returned.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        path: Path | None = None,
        connection: sqlite3.Connection | None = None,
        table_prefix: str = "near_duplicate",
    ) -> None:
        if (path is None) == (connection is None):
            raise ValueError("provide exactly one SQLite path or connection")
        if re.fullmatch(r"[a-z][a-z0-9_]{0,31}", table_prefix) is None:
            raise ValueError("invalid SQLite table prefix")

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
        layout = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self.hashranges = tuple(tuple(item) for item in layout.hashranges)
        self.signature_width = 4 if self.scheme == "affine32" else 8
        self.table_prefix = table_prefix
        self._owns_connection = connection is None
        if connection is None:
            assert path is not None
            path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(path)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA temp_store=MEMORY")
        self.connection = connection
        self._create_schema(config)

    @property
    def _metadata_table(self) -> str:
        return f"{self.table_prefix}_metadata"

    @property
    def _signatures_table(self) -> str:
        return f"{self.table_prefix}_signatures"

    @property
    def _buckets_table(self) -> str:
        return f"{self.table_prefix}_buckets"

    def _create_schema(self, config: Mapping[str, Any]) -> None:
        self.connection.execute(
            f"CREATE TABLE IF NOT EXISTS {self._metadata_table} "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.connection.execute(
            f"CREATE TABLE IF NOT EXISTS {self._signatures_table} "
            "(document_id TEXT PRIMARY KEY, signature BLOB NOT NULL)"
        )
        self.connection.execute(
            f"CREATE TABLE IF NOT EXISTS {self._buckets_table} "
            "(band INTEGER NOT NULL, bucket BLOB NOT NULL, document_id TEXT NOT NULL, "
            "PRIMARY KEY (band, bucket, document_id))"
        )
        self.connection.execute(
            f"CREATE INDEX IF NOT EXISTS {self.table_prefix}_bucket_lookup "
            f"ON {self._buckets_table} (band, bucket)"
        )
        fingerprint = sha256_bytes(
            json.dumps(
                {
                    "hashranges": self.hashranges,
                    "num_perm": self.num_perm,
                    "package_version": config["package_version"],
                    "scheme": self.scheme,
                    "seed": self.seed,
                    "shingle_tokens": self.shingle_tokens,
                    "threshold": self.threshold,
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        existing = self.connection.execute(
            f"SELECT value FROM {self._metadata_table} WHERE key = 'fingerprint'"
        ).fetchone()
        if existing is not None and existing[0] != fingerprint and self.count() > 0:
            raise ValueError("SQLite near-duplicate index configuration changed")
        self.connection.execute(
            f"INSERT OR REPLACE INTO {self._metadata_table} (key, value) VALUES (?, ?)",
            ("fingerprint", fingerprint),
        )
        self.connection.execute(
            f"INSERT OR REPLACE INTO {self._metadata_table} (key, value) VALUES (?, ?)",
            ("schema_version", "1"),
        )

    def _signature_bytes(self, signature: Any) -> bytes:
        if len(signature.hashvalues) != self.num_perm:
            raise ValueError("MinHash signature length does not match the configured num_perm")
        return bytes(signature.hashvalues.byteswap().data)

    def _bucket_bytes(self, signature: Any, start: int, end: int) -> bytes:
        return bytes(signature.hashvalues[start:end].byteswap().data)

    def _estimated_jaccard(self, left: bytes, right: bytes) -> float:
        if len(left) != len(right) or len(left) != self.num_perm * self.signature_width:
            raise ValueError("stored MinHash signature has an invalid size")
        matches = sum(
            left[offset : offset + self.signature_width]
            == right[offset : offset + self.signature_width]
            for offset in range(0, len(left), self.signature_width)
        )
        return matches / self.num_perm

    def duplicate_of(
        self, signature: Any, *, exclude_document_id: str | None = None
    ) -> str | None:
        candidates: set[str] = set()
        for band, (start, end) in enumerate(self.hashranges):
            bucket = self._bucket_bytes(signature, start, end)
            candidates.update(
                row[0]
                for row in self.connection.execute(
                    f"SELECT document_id FROM {self._buckets_table} "
                    "WHERE band = ? AND bucket = ?",
                    (band, bucket),
                )
            )
        query_bytes = self._signature_bytes(signature)
        for candidate in sorted(candidates):
            if candidate == exclude_document_id:
                continue
            row = self.connection.execute(
                f"SELECT signature FROM {self._signatures_table} WHERE document_id = ?",
                (candidate,),
            ).fetchone()
            if row is None:
                raise ValueError(f"missing stored MinHash signature for {candidate}")
            if self._estimated_jaccard(query_bytes, bytes(row[0])) >= self.threshold:
                return candidate
        return None

    def add(self, document_id: str, signature: Any) -> None:
        signature_bytes = self._signature_bytes(signature)
        existing = self.connection.execute(
            f"SELECT signature FROM {self._signatures_table} WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        if existing is not None:
            if bytes(existing[0]) != signature_bytes:
                raise ValueError(f"near-duplicate signature changed for {document_id}")
            return
        self.connection.execute(
            f"INSERT INTO {self._signatures_table} (document_id, signature) VALUES (?, ?)",
            (document_id, signature_bytes),
        )
        self.connection.executemany(
            f"INSERT INTO {self._buckets_table} (band, bucket, document_id) "
            "VALUES (?, ?, ?)",
            (
                (band, self._bucket_bytes(signature, start, end), document_id)
                for band, (start, end) in enumerate(self.hashranges)
            ),
        )

    def replace(self, document_id: str, signature: Any) -> None:
        """Replace one durable signature and all of its LSH buckets."""

        signature_bytes = self._signature_bytes(signature)
        existing = self.connection.execute(
            f"SELECT signature FROM {self._signatures_table} WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        if existing is None:
            raise KeyError(f"near-duplicate signature does not exist for {document_id}")
        if bytes(existing[0]) == signature_bytes:
            return
        self.connection.execute(
            f"DELETE FROM {self._buckets_table} WHERE document_id = ?", (document_id,)
        )
        self.connection.execute(
            f"UPDATE {self._signatures_table} SET signature = ? WHERE document_id = ?",
            (signature_bytes, document_id),
        )
        self.connection.executemany(
            f"INSERT INTO {self._buckets_table} (band, bucket, document_id) "
            "VALUES (?, ?, ?)",
            (
                (band, self._bucket_bytes(signature, start, end), document_id)
                for band, (start, end) in enumerate(self.hashranges)
            ),
        )

    def count(self) -> int:
        row = self.connection.execute(
            f"SELECT COUNT(*) FROM {self._signatures_table}"
        ).fetchone()
        assert row is not None
        return int(row[0])

    def document_ids(self) -> set[str]:
        return {
            str(row[0])
            for row in self.connection.execute(
                f"SELECT document_id FROM {self._signatures_table}"
            )
        }

    def clear(self) -> None:
        self.connection.execute(f"DELETE FROM {self._buckets_table}")
        self.connection.execute(f"DELETE FROM {self._signatures_table}")

    def flush(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        if self._owns_connection:
            self.connection.commit()
            self.connection.close()


@dataclass(frozen=True)
class QualityEvaluation:
    rejection_reason: str | None
    metadata: dict[str, Any] | None
    near_duplicate_signature: Any | None


class DataQualityGate:
    """Compose language, contamination, and near-duplicate gates."""

    def __init__(
        self,
        config: Mapping[str, Any],
        repository_root: Path,
        *,
        near_duplicate_path: Path | None = None,
        near_duplicate_connection: sqlite3.Connection | None = None,
        near_duplicate_table_prefix: str = "near_duplicate",
    ) -> None:
        validate_quality_config(config)
        self.config = config
        self.language = LinguaLanguageIdentifier(config["language"])
        self.benchmark = BenchmarkRegistry(
            config["benchmark_decontamination"], repository_root
        )
        if near_duplicate_path is not None or near_duplicate_connection is not None:
            self.near_duplicate = SQLiteMinHashNearDuplicateIndex(
                config["near_duplicate"],
                path=near_duplicate_path,
                connection=near_duplicate_connection,
                table_prefix=near_duplicate_table_prefix,
            )
        else:
            self.near_duplicate = MinHashNearDuplicateIndex(config["near_duplicate"])

    def evaluate(
        self,
        text: str,
        category: str,
        *,
        exclude_document_id: str | None = None,
    ) -> QualityEvaluation:
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
        duplicate_of = self.near_duplicate.duplicate_of(
            signature, exclude_document_id=exclude_document_id
        )
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

    def replace(self, document_id: str, evaluation: QualityEvaluation) -> None:
        """Replace metadata-backed index state after a deterministic text repair."""

        if evaluation.rejection_reason is not None or evaluation.near_duplicate_signature is None:
            raise ValueError("cannot replace with a rejected or incomplete quality evaluation")
        self.near_duplicate.replace(document_id, evaluation.near_duplicate_signature)

    def verify_and_accept_record(self, record: Mapping[str, Any], line_number: int) -> None:
        document_id = str(record["id"])
        evaluation = self.evaluate(
            record["text"],
            record["provenance"]["category"],
            exclude_document_id=document_id,
        )
        if evaluation.rejection_reason is not None:
            raise ValueError(
                f"quality gate {evaluation.rejection_reason} failed at line {line_number}"
            )
        if record.get("quality") != evaluation.metadata:
            raise ValueError(f"quality metadata mismatch at line {line_number}")
        self.accept(document_id, evaluation)

    def flush(self) -> None:
        self.near_duplicate.flush()

    def close(self) -> None:
        self.near_duplicate.close()
