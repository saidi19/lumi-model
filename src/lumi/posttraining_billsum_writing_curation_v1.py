"""Bounded local-only BillSum writing-closure acquisition and curation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import ssl
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from statistics import mean, median
from typing import Any, BinaryIO, Protocol

import certifi

from .data_quality import BenchmarkRegistry, MinHashNearDuplicateIndex, normalize_for_matching
from .posttraining_data import SFTPolicy, evaluate_sft_record, load_sft_policy
from .posttraining_format import (
    ChatTemplate,
    encode_sft_messages,
    load_chat_template,
    verify_chat_tokenizer,
)
from .posttraining_gpt_oss_teacher_source_promotion_preflight_v16 import (
    PureByteLevelBPETokenizer,
)
from .posttraining_owner_review import build_owner_review_html

PLAN_SCHEMA_VERSION = "lumi.sft-billsum-writing-curation-plan.v0.1"
REPORT_SCHEMA_VERSION = "lumi.sft-billsum-writing-curation-report.v0.1"
VERIFICATION_SCHEMA_VERSION = "lumi.sft-billsum-writing-curation-verification.v0.1"

SOURCE_ID = "fiscalnote_billsum_writing_candidate"
DATASET_ID = "FiscalNote/billsum"
REVISION = "3d8510441c06a3d9dfb32eb0d7f80151730bcc4f"
LICENSE_ID = "cc0-1.0"
OBJECT_PATH = "data/train-00000-of-00001.parquet"
EXPECTED_OBJECT = {
    "bytes": 91_811_112,
    "rows": 18_949,
    "sha256": "7ca4e515ad7502ef579a055c2d64e6dfa0d64200017c15bca00f323a34cd2ae3",
}
EXPECTED_COLUMNS = frozenset({"summary", "text", "title"})
EXPECTED_BINDINGS = {
    "source_registry_v0_19": "d88ef71863c38c277260172dd6b3b5ce5e2330726370329bc438b7750f43ec9c",
    "candidate_mix_v0_13_report": (
        "9f77596c80c555c3a44c7a4993db3862f2a5503ed834c57a22d4b8d2f23f8361"
    ),
    "candidate_mix_v0_13_verification": (
        "bbacce4844c201c70e3b3b48bc3871c65c7ad4162d7ece973e6e27dc4c7cf0d4"
    ),
    "data_contract": "9a283d9333d29deb7b516fa45eb4eec50232be9925f4ee4aa3a8f624213b2f06",
    "quality_config": "23fba21dbc822c84a5d16e63147fc1a7716576b8f1258cda8c035391f04f52c0",
    "benchmark_registry": "1ffb5e938ab28c2fc876b9efd6c6ed59059f1e85aed2c396c024ae5925a4745e",
    "chat_template": "bc7853a9277feddae2fd3db786c6a12f4fa3602a135a0408c38d4390f33bec12",
    "tokenizer_json": "1022c9538d9ec795b07c2e2bda3c5c6f21c321de933cfe5ad197628a94caa2ee",
}
EXPECTED_LIMITS = {
    "maximum_input_rows": 18_949,
    "maximum_network_bytes": 100_663_296,
    "maximum_private_write_bytes": 536_870_912,
    "maximum_runtime_minutes": 120,
    "maximum_train_supervised_tokens": 850_000,
    "maximum_validation_supervised_tokens": 50_000,
    "minimum_train_supervised_tokens": 820_000,
    "owner_review_rows": 32,
}
NEAR_DUPLICATE_CONFIG = {
    "algorithm": "minhash_lsh",
    "num_perm": 128,
    "package_version": "2.0.0",
    "scheme": "affine32",
    "seed": 1,
    "shingle_tokens": 5,
    "threshold": 0.9,
}
USER_PREFIX = "Summarize the following bill accurately and concisely.\n\nTitle: "
TEXT_SEPARATOR = "\n\nBill text:\n"
ORIGIN_LABEL = "source_provided_legislative_summary_authorship_not_established"
ADAPTER_ID = "billsum_complete_source_pair_v0.1"
_NUMBER = re.compile(r"(?<![A-Za-z])(?:\$?\d[\d,.]*(?:%|st|nd|rd|th)?)(?![A-Za-z])")
_TERMINAL = re.compile(r"[.!?][\"')\]]*$")


class DownloadResponse(Protocol):
    headers: Mapping[str, str]

    def __enter__(self) -> DownloadResponse: ...

    def __exit__(self, *args: Any) -> None: ...

    def geturl(self) -> str: ...

    def read(self, size: int = -1) -> bytes: ...


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _resolve(root: Path, relative: str, expected_sha256: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.parent) or not path.is_file():
        raise ValueError(f"BillSum binding missing: {relative}")
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"BillSum binding changed: {relative}")
    return path


def load_plan(path: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    plan = _load_json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported BillSum curation plan")
    if plan.get("status") != "authorized_local_only_acquisition_and_curation":
        raise ValueError("BillSum curation authorization changed")
    if plan.get("limits") != EXPECTED_LIMITS:
        raise ValueError("BillSum limits changed")
    if plan.get("near_duplicate") != NEAR_DUPLICATE_CONFIG:
        raise ValueError("BillSum near-duplicate policy changed")
    if plan.get("authorization") != {
        "dataset_seal_authorized": False,
        "external_api_authorized": False,
        "gpu_use_authorized": False,
        "inference_authorized": False,
        "local_acquisition_and_curation_authorized": True,
        "production_promotion_authorized": False,
        "registry_change_authorized": False,
        "sft_training_authorized": False,
    }:
        raise ValueError("BillSum authorization boundary changed")
    source = plan.get("source")
    if not isinstance(source, Mapping) or (
        source.get("source_id") != SOURCE_ID
        or source.get("dataset_id") != DATASET_ID
        or source.get("revision") != REVISION
        or source.get("license_id") != LICENSE_ID
        or source.get("object_path") != OBJECT_PATH
        or source.get("object") != EXPECTED_OBJECT
    ):
        raise ValueError("BillSum source binding changed")
    policy = plan.get("policy")
    if not isinstance(policy, Mapping) or (
        policy.get("identity_split") != "sha256_normalized_title_nul_text_modulo_20_v0.1"
        or policy.get("source_split_used") != "train_only"
        or policy.get("test_and_ca_test_access") != "permanently_prohibited"
        or policy.get("overlength") != "reject_without_truncation"
        or policy.get("supervised_token_range") != [64, 512]
    ):
        raise ValueError("BillSum curation policy changed")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("BillSum bindings missing")
    for name, expected in EXPECTED_BINDINGS.items():
        binding = bindings.get(name)
        if not isinstance(binding, Mapping) or binding.get("sha256") != expected:
            raise ValueError(f"BillSum plan binding changed: {name}")
        _resolve(root, str(binding.get("path")), expected)
    for name in ("implementation", "runner", "verifier", "test"):
        binding = bindings.get(name)
        if not isinstance(binding, Mapping):
            raise ValueError(f"BillSum implementation binding missing: {name}")
        _resolve(root, str(binding.get("path")), str(binding.get("sha256")))
    return plan


def exact_object_url() -> str:
    quoted = urllib.parse.quote(OBJECT_PATH, safe="/")
    return (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{REVISION}/"
        f"{quoted}?download=true"
    )


def _safe_redirect_host(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    allowed = (
        host == "huggingface.co"
        or host.endswith(".huggingface.co")
        or host.endswith(".hf.co")
        or host.endswith(".xethub.hf.co")
    )
    if parsed.scheme != "https" or not allowed:
        raise ValueError(f"BillSum download redirected to an unapproved host: {host}")
    return host


def secure_urlopen(request: urllib.request.Request) -> DownloadResponse:
    context = ssl.create_default_context(cafile=certifi.where())
    return urllib.request.urlopen(request, context=context)


def download_exact_object(
    *, destination: Path, maximum_bytes: int, open_url: Any = secure_urlopen
) -> dict[str, Any]:
    if maximum_bytes < EXPECTED_OBJECT["bytes"]:
        raise ValueError("BillSum network-byte allowance is too small")
    if destination.exists():
        if (
            destination.stat().st_size == EXPECTED_OBJECT["bytes"]
            and file_sha256(destination) == EXPECTED_OBJECT["sha256"]
        ):
            return {
                "bytes": destination.stat().st_size,
                "object_path": OBJECT_PATH,
                "redirect_host": "local_verified_reuse",
                "requested_url": exact_object_url(),
                "sha256": EXPECTED_OBJECT["sha256"],
                "reused_existing_exact_object": True,
            }
        raise FileExistsError(f"refusing to overwrite changed BillSum object: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale BillSum download temporary exists: {temporary}")
    request = urllib.request.Request(
        exact_object_url(), headers={"User-Agent": "lumi-billsum-curation-v0.1"}, method="GET"
    )
    digest = hashlib.sha256()
    downloaded = 0
    final_host = ""
    try:
        with open_url(request) as response:
            final_host = _safe_redirect_host(response.geturl())
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) > maximum_bytes:
                raise ValueError("BillSum object exceeds the network-byte cap")
            with temporary.open("xb") as handle:
                while True:
                    chunk = response.read(min(1024 * 1024, maximum_bytes - downloaded + 1))
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > maximum_bytes:
                        raise ValueError("BillSum object exceeded the network-byte cap")
                    digest.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        if (
            downloaded != EXPECTED_OBJECT["bytes"]
            or digest.hexdigest() != EXPECTED_OBJECT["sha256"]
        ):
            raise ValueError("BillSum downloaded object identity changed")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "bytes": downloaded,
        "object_path": OBJECT_PATH,
        "redirect_host": final_host,
        "requested_url": exact_object_url(),
        "sha256": digest.hexdigest(),
        "reused_existing_exact_object": False,
    }


def bill_identity(title: str, text: str) -> str:
    normalized_title = normalize_for_matching(title)
    normalized_text = normalize_for_matching(text)
    if not normalized_title or not normalized_text:
        raise ValueError("BillSum title/text is empty")
    return hashlib.sha256(
        (normalized_title + "\0" + normalized_text).encode("utf-8")
    ).hexdigest()


def identity_split(identity_sha256: str) -> str:
    digest = hashlib.sha256(
        f"billsum-identity-split-v0.1\0{identity_sha256}".encode()
    ).digest()
    return "validation" if int.from_bytes(digest[:8], "big") % 20 == 0 else "train"


def candidate_from_source(
    row: Mapping[str, Any], *, row_index: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    if set(row) != EXPECTED_COLUMNS:
        raise ValueError("BillSum row schema changed")
    text, summary, title = row.get("text"), row.get("summary"), row.get("title")
    if not all(isinstance(value, str) and value.strip() for value in (text, summary, title)):
        raise ValueError("BillSum row contains an empty source field")
    text, summary, title = text.strip(), summary.strip(), title.strip()
    identity_sha256 = bill_identity(title, text)
    split = identity_split(identity_sha256)
    messages = [
        {"content": f"{USER_PREFIX}{title}{TEXT_SEPARATOR}{text}", "role": "user"},
        {"content": summary, "role": "assistant"},
    ]
    content_sha256 = sha256_value(messages)
    transform_sha256 = sha256_value(
        {
            "adapter": ADAPTER_ID,
            "content_sha256": content_sha256,
            "identity_sha256": identity_sha256,
            "split": split,
        }
    )
    record_id = f"billsum-{identity_sha256[:24]}"
    record = {
        "language": "en",
        "messages": messages,
        "provenance": {
            "assistant_response_origin": ORIGIN_LABEL,
            "consent_basis": "public_domain",
            "content_sha256": content_sha256,
            "identity_sha256": identity_sha256,
            "owner_review_status": "pending_deterministic_spot_check",
            "source_pair_integrity": "exact_untruncated_source_text_and_summary",
            "transform_sha256": transform_sha256,
        },
        "quality": {
            "pii_scan": "passed",
            "review_method": (
                "source_pair_integrity_plus_lumi_frozen_gates_"
                "and_owner_spot_check_v0.1"
            ),
            "review_status": "accepted",
            "score": 0.9,
            "secret_scan": "passed",
        },
        "record_id": record_id,
        "safety": {"policy_version": "lumi-sft-basic-pattern-scan-v0.1", "risk_labels": []},
        "schema_version": "lumi.sft.v0.1",
        "source": {
            "dataset_id": DATASET_ID,
            "document_id": f"billsum-train:{identity_sha256}",
            "license_id": LICENSE_ID,
            "revision": REVISION,
            "source_id": SOURCE_ID,
            "uri": (
                f"https://huggingface.co/datasets/{DATASET_ID}/blob/{REVISION}/"
                f"{OBJECT_PATH}#row-{row_index}"
            ),
        },
        "split": split,
        "task_category": "writing_summarization",
    }
    attribution = {
        "bill_identity_sha256": identity_sha256,
        "object_path": OBJECT_PATH,
        "record_id": record_id,
        "row_index": row_index,
        "schema_version": "lumi.sft-billsum-attribution.v0.1",
        "source_pair_sha256": content_sha256,
        "split": split,
    }
    return record, attribution


def custom_quality_reasons(record: Mapping[str, Any]) -> tuple[str, ...]:
    prompt = str(record["messages"][0]["content"])
    response = str(record["messages"][1]["content"])
    reasons: list[str] = []
    if not prompt.startswith(USER_PREFIX) or TEXT_SEPARATOR not in prompt:
        reasons.append("source_pair_wrapper")
    if _TERMINAL.search(response.rstrip()) is None:
        reasons.append("assistant_terminal_punctuation")
    prompt_numbers = set(_NUMBER.findall(prompt.casefold()))
    response_numbers = set(_NUMBER.findall(response.casefold()))
    if not response_numbers.issubset(prompt_numbers):
        reasons.append("unsupported_numeric_literal")
    if normalize_for_matching(prompt) == normalize_for_matching(response):
        reasons.append("prompt_response_exact_echo")
    return tuple(dict.fromkeys(reasons))


def load_runtime_components(
    plan: Mapping[str, Any], *, repository_root: Path
) -> tuple[SFTPolicy, BenchmarkRegistry, ChatTemplate, PureByteLevelBPETokenizer]:
    bindings = plan["bindings"]
    contract = _resolve(
        repository_root,
        str(bindings["data_contract"]["path"]),
        str(bindings["data_contract"]["sha256"]),
    )
    quality_path = _resolve(
        repository_root,
        str(bindings["quality_config"]["path"]),
        str(bindings["quality_config"]["sha256"]),
    )
    benchmark_path = _resolve(
        repository_root,
        str(bindings["benchmark_registry"]["path"]),
        str(bindings["benchmark_registry"]["sha256"]),
    )
    template_path = _resolve(
        repository_root,
        str(bindings["chat_template"]["path"]),
        str(bindings["chat_template"]["sha256"]),
    )
    tokenizer_path = _resolve(
        repository_root,
        str(bindings["tokenizer_json"]["path"]),
        str(bindings["tokenizer_json"]["sha256"]),
    )
    benchmark_config = _load_json(quality_path).get("benchmark_decontamination")
    if not isinstance(benchmark_config, Mapping):
        raise ValueError("BillSum benchmark policy missing")
    policy = replace(load_sft_policy(contract), benchmark_decontamination=dict(benchmark_config))
    benchmark = BenchmarkRegistry(policy.benchmark_decontamination, repository_root)
    if benchmark.path.resolve() != benchmark_path:
        raise ValueError("BillSum benchmark registry resolved elsewhere")
    template = load_chat_template(template_path)
    tokenizer = PureByteLevelBPETokenizer(tokenizer_path)
    verify_chat_tokenizer(tokenizer, template)
    return policy, benchmark, template, tokenizer


def _role_text(record: Mapping[str, Any]) -> str:
    return normalize_for_matching(
        "\n".join(f"{m['role']} {m['content']}" for m in record["messages"])
    )


def apply_near_dedup(
    measured: Sequence[tuple[dict[str, Any], dict[str, Any], int, int]],
) -> tuple[
    list[tuple[dict[str, Any], dict[str, Any], int, int]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    index = MinHashNearDuplicateIndex(NEAR_DUPLICATE_CONFIG)
    ordered = sorted(
        measured,
        key=lambda item: (
            0 if item[0]["split"] == "train" else 1,
            hashlib.sha256(f"billsum-near-dedup-v0.1\0{item[0]['record_id']}".encode()).hexdigest(),
        ),
    )
    accepted: list[tuple[dict[str, Any], dict[str, Any], int, int]] = []
    ledger: list[dict[str, Any]] = []
    cross_split = 0
    split_by_id: dict[str, str] = {}
    for item in ordered:
        record = item[0]
        signature = index.signature(_role_text(record))
        duplicate_of = index.duplicate_of(signature)
        keep = duplicate_of is None
        if keep:
            index.add(str(record["record_id"]), signature)
            split_by_id[str(record["record_id"])] = str(record["split"])
            accepted.append(item)
        elif split_by_id.get(str(duplicate_of)) != record["split"]:
            cross_split += 1
        ledger.append(
            {
                "accepted": keep,
                "duplicate_of": duplicate_of,
                "record_id": record["record_id"],
                "signature_sha256": index.fingerprint(signature),
                "split": record["split"],
            }
        )
    return accepted, ledger, {
        "accepted": len(accepted),
        "comparison_unit": "role_tagged_full_conversation",
        "cross_split_rejections": cross_split,
        "rejected": len(measured) - len(accepted),
        "threshold": 0.9,
    }


def _write_jsonl_row(handle: BinaryIO, value: Mapping[str, Any]) -> tuple[int, int]:
    payload = canonical_json_bytes(value) + b"\n"
    offset = handle.tell()
    handle.write(payload)
    return offset, len(payload)


def _read_spool_row(handle: BinaryIO, *, offset: int, length: int) -> dict[str, Any]:
    handle.seek(offset)
    value = json.loads(handle.read(length))
    if not isinstance(value, dict):
        raise ValueError("BillSum spool row changed")
    return value


def select_to_token_caps(
    measured: Sequence[tuple[dict[str, Any], dict[str, Any], int, int]],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], int, int]], dict[str, Any]]:
    selected: list[tuple[dict[str, Any], dict[str, Any], int, int]] = []
    tokens: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    for split, cap in (
        ("train", EXPECTED_LIMITS["maximum_train_supervised_tokens"]),
        ("validation", EXPECTED_LIMITS["maximum_validation_supervised_tokens"]),
    ):
        rows = [item for item in measured if item[0]["split"] == split]
        rows.sort(
            key=lambda item: hashlib.sha256(
                f"billsum-token-selection-v0.1\0{item[0]['record_id']}".encode()
            ).hexdigest()
        )
        for item in rows:
            if tokens[split] + item[3] > cap:
                skipped[split] += 1
                continue
            selected.append(item)
            tokens[split] += item[3]
    selected.sort(key=lambda item: (str(item[0]["split"]), str(item[0]["record_id"])))
    return selected, {
        "skipped_for_token_cap": dict(sorted(skipped.items())),
        "train_supervised_tokens": tokens["train"],
        "validation_supervised_tokens": tokens["validation"],
    }


def deterministic_owner_sample(
    records: Sequence[Mapping[str, Any]], *, maximum_records: int
) -> list[dict[str, Any]]:
    buckets: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[str(record["split"])].append(record)
    for rows in buckets.values():
        rows.sort(
            key=lambda row: hashlib.sha256(
                f"billsum-owner-review-v0.1\0{row['record_id']}".encode()
            ).hexdigest()
        )
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < min(maximum_records, len(records)):
        added = False
        for split in ("train", "validation"):
            if offset >= len(buckets[split]):
                continue
            row = buckets[split][offset]
            selected.append(
                {
                    "language_mode": "english",
                    "messages": row["messages"],
                    "record_id": row["record_id"],
                    "skill_id": "writing_summarization",
                    "split": split,
                }
            )
            added = True
            if len(selected) == min(maximum_records, len(records)):
                break
        if not added:
            break
        offset += 1
    return selected


def build_billsum_owner_review_html(sample: list[Mapping[str, Any]]) -> tuple[str, str]:
    sample_hash = sha256_value(sample)
    page = build_owner_review_html(sample, sample_sha256=sample_hash)
    replacements = {
        "Lumi owner review": "Lumi BillSum owner review",
        "Lumi owner spot-check": "Lumi BillSum writing-summary spot-check",
        (
            "These rows are AI-generated and owner-reviewed. Approving them does not "
            "make them human-written."
        ): (
            "These are source-provided legislative summaries with authorship not established by "
            "the dataset card. Check semantic grounding, accuracy, completeness, naturalness, "
            "and punctuation. Approval does not authorize registry change, sealing, or SFT."
        ),
        "lumi.ai-assisted-owner-decisions.v0.3": "lumi.billsum-owner-decisions.v0.1",
        "lumi-owner-review-decisions-v0.3.json": "lumi-billsum-owner-decisions-v0.1.json",
    }
    for old, new in replacements.items():
        if old not in page:
            raise ValueError(f"BillSum owner-review marker missing: {old[:48]}")
        page = page.replace(old, new)
    return page, sample_hash


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite BillSum JSON: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite BillSum JSONL: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite BillSum text: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _artifact(path: Path, *, name: str, raw: bool) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "contains_raw_training_text": raw,
        "name": name,
        "path": str(path),
        "sha256": file_sha256(path),
    }


def _validate_record_binding(record: Mapping[str, Any]) -> None:
    source = record.get("source")
    provenance = record.get("provenance")
    if not isinstance(source, Mapping) or (
        source.get("source_id") != SOURCE_ID
        or source.get("dataset_id") != DATASET_ID
        or source.get("revision") != REVISION
        or source.get("license_id") != LICENSE_ID
    ):
        raise ValueError("BillSum candidate source binding changed")
    if not isinstance(provenance, Mapping) or (
        provenance.get("assistant_response_origin") != ORIGIN_LABEL
        or provenance.get("consent_basis") != "public_domain"
        or provenance.get("source_pair_integrity")
        != "exact_untruncated_source_text_and_summary"
    ):
        raise ValueError("BillSum candidate provenance changed")


def run_curation(
    *, plan_path: str | Path, repository_root: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    import pyarrow.parquet as pq

    started = time.monotonic()
    root = Path(repository_root).resolve()
    plan = load_plan(plan_path, repository_root=root)
    private_root = (root / str(plan["output"]["private_root"])).resolve()
    if not private_root.is_relative_to(root):
        raise ValueError("BillSum private output escapes repository")
    raw_path = private_root / "private" / "raw" / Path(OBJECT_PATH).name
    download = download_exact_object(
        destination=raw_path, maximum_bytes=EXPECTED_LIMITS["maximum_network_bytes"]
    )
    if (
        raw_path.stat().st_size != EXPECTED_OBJECT["bytes"]
        or file_sha256(raw_path) != EXPECTED_OBJECT["sha256"]
    ):
        raise ValueError("BillSum exact object changed before decode")

    policy, benchmark, template, tokenizer = load_runtime_components(plan, repository_root=root)
    parquet = pq.ParquetFile(raw_path)
    if parquet.metadata.num_rows != EXPECTED_OBJECT["rows"]:
        raise ValueError("BillSum parquet row count changed")
    if frozenset(parquet.schema_arrow.names) != EXPECTED_COLUMNS:
        raise ValueError("BillSum parquet schema changed")

    spool_path = private_root / "private" / "work" / "eligible-records.jsonl"
    spool_path.parent.mkdir(parents=True, exist_ok=True)
    if spool_path.exists():
        raise FileExistsError(f"refusing stale BillSum spool: {spool_path}")
    spool_index: list[dict[str, Any]] = []
    input_rows = 0
    decoded_bytes = 0
    selection_rejections: Counter[str] = Counter()
    generic_rejections: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_normalized: set[str] = set()
    identity_splits: defaultdict[str, set[str]] = defaultdict(set)
    decision_digest = hashlib.sha256()
    serialized_lengths: list[int] = []

    with spool_path.open("xb") as spool:
        for batch in parquet.iter_batches(batch_size=64, use_threads=False):
            decoded_bytes += int(batch.nbytes)
            for row in batch.to_pylist():
                row_index = input_rows
                input_rows += 1
                if input_rows > EXPECTED_LIMITS["maximum_input_rows"]:
                    raise ValueError("BillSum input-row cap exceeded")
                try:
                    record, attribution = candidate_from_source(row, row_index=row_index)
                except ValueError as error:
                    selection_rejections[str(error)] += 1
                    continue
                custom = list(custom_quality_reasons(record))
                decision = evaluate_sft_record(
                    record, policy=policy, benchmark_registry=benchmark
                )
                reasons = custom + list(decision.reasons)
                record_id = str(record["record_id"])
                if record_id in seen_ids:
                    reasons.append("duplicate_record_id")
                if decision.normalized_sha256 in seen_normalized:
                    reasons.append("duplicate_normalized_conversation")
                seen_ids.add(record_id)
                seen_normalized.add(decision.normalized_sha256)
                reasons = list(dict.fromkeys(reasons))
                if reasons:
                    generic_rejections.update(reasons)
                    decision_digest.update(
                        canonical_json_bytes(
                            {"accepted": False, "reasons": reasons, "record_id": record_id}
                        )
                        + b"\n"
                    )
                    continue
                _validate_record_binding(record)
                try:
                    encoded = encode_sft_messages(
                        record["messages"], tokenizer=tokenizer, template=template
                    )
                except ValueError as error:
                    if "maximum is 4096" not in str(error):
                        raise
                    selection_rejections["overlength_4k_no_truncation"] += 1
                    continue
                supervised = encoded.supervised_tokens
                if not 64 <= supervised <= 512:
                    selection_rejections["supervised_token_range_64_512"] += 1
                    continue
                serialized = len(encoded.input_ids)
                serialized_lengths.append(serialized)
                identity_splits[str(attribution["bill_identity_sha256"])].add(
                    str(record["split"])
                )
                payload = {
                    "attribution": attribution,
                    "normalized_sha256": decision.normalized_sha256,
                    "record": record,
                    "serialized_tokens": serialized,
                    "supervised_tokens": supervised,
                }
                offset, length = _write_jsonl_row(spool, payload)
                spool_index.append({"length": length, "offset": offset})
                decision_digest.update(
                    canonical_json_bytes(
                        {"accepted": True, "reasons": [], "record_id": record_id}
                    )
                    + b"\n"
                )
        spool.flush()
        os.fsync(spool.fileno())
    if input_rows != EXPECTED_OBJECT["rows"]:
        raise ValueError("BillSum decoded row count changed")
    if any(len(splits) != 1 for splits in identity_splits.values()):
        raise ValueError("BillSum content identity crossed splits")

    measured: list[tuple[dict[str, Any], dict[str, Any], int, int]] = []
    with spool_path.open("rb") as spool:
        for item in spool_index:
            payload = _read_spool_row(
                spool, offset=int(item["offset"]), length=int(item["length"])
            )
            measured.append(
                (
                    payload["record"],
                    payload["attribution"],
                    int(payload["serialized_tokens"]),
                    int(payload["supervised_tokens"]),
                )
            )
    near_accepted, near_ledger, near_metrics = apply_near_dedup(measured)
    selected, cap_metrics = select_to_token_caps(near_accepted)
    train_tokens = int(cap_metrics["train_supervised_tokens"])
    validation_tokens = int(cap_metrics["validation_supervised_tokens"])
    if not EXPECTED_LIMITS["minimum_train_supervised_tokens"] <= train_tokens <= EXPECTED_LIMITS[
        "maximum_train_supervised_tokens"
    ]:
        raise ValueError("BillSum retained train token target not met")
    if validation_tokens > EXPECTED_LIMITS["maximum_validation_supervised_tokens"]:
        raise ValueError("BillSum validation token cap exceeded")

    records = [item[0] for item in selected]
    attributions = [item[1] for item in selected]
    token_ledger = [
        {
            "record_id": item[0]["record_id"],
            "serialized_tokens": item[2],
            "split": item[0]["split"],
            "supervised_tokens": item[3],
        }
        for item in selected
    ]
    owner_sample = deterministic_owner_sample(
        records, maximum_records=EXPECTED_LIMITS["owner_review_rows"]
    )
    if len(owner_sample) != EXPECTED_LIMITS["owner_review_rows"]:
        raise ValueError("BillSum owner-review packet is incomplete")
    owner_html, owner_hash = build_billsum_owner_review_html(owner_sample)

    candidate_path = private_root / "private" / "billsum-writing-candidate.jsonl"
    attribution_path = private_root / "private" / "billsum-writing-attribution.jsonl"
    token_path = private_root / "private" / "billsum-writing-token-ledger.jsonl"
    near_path = private_root / "private" / "billsum-writing-near-dedup-ledger.jsonl"
    rejection_path = private_root / "private" / "billsum-writing-rejection-ledger.json"
    owner_json_path = private_root / "private" / "owner-review-32.json"
    owner_html_path = private_root / "private" / "owner-review-32.html"
    report_path = root / str(plan["output"]["report_path"])
    verification_path = root / str(plan["output"]["verification_path"])

    _write_jsonl_atomic(candidate_path, records)
    _write_jsonl_atomic(attribution_path, attributions)
    _write_jsonl_atomic(token_path, token_ledger)
    _write_jsonl_atomic(near_path, near_ledger)
    _write_json_atomic(
        rejection_path,
        {
            "decision_ledger_sha256": decision_digest.hexdigest(),
            "generic_rejection_counts": dict(sorted(generic_rejections.items())),
            "raw_text_retained": False,
            "selection_rejection_counts": dict(sorted(selection_rejections.items())),
        },
    )
    _write_json_atomic(owner_json_path, owner_sample)
    _write_text_atomic(owner_html_path, owner_html)
    spool_path.unlink()
    spool_path.parent.rmdir()

    raw_artifact = _artifact(raw_path, name="raw_train_parquet", raw=True)
    private_objects = [
        _artifact(candidate_path, name="candidate", raw=True),
        _artifact(attribution_path, name="attribution", raw=False),
        _artifact(token_path, name="token_ledger", raw=False),
        _artifact(near_path, name="near_dedup_ledger", raw=False),
        _artifact(rejection_path, name="rejection_ledger", raw=False),
        _artifact(owner_json_path, name="owner_review_json", raw=True),
        _artifact(owner_html_path, name="owner_review_html", raw=True),
    ]
    private_bytes = raw_artifact["bytes"] + sum(item["bytes"] for item in private_objects)
    if private_bytes > EXPECTED_LIMITS["maximum_private_write_bytes"]:
        raise ValueError("BillSum private-write cap exceeded")

    by_split: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for item in selected:
        bucket = by_split[str(item[0]["split"])]
        bucket["records"] += 1
        bucket["serialized_tokens"] += item[2]
        bucket["supervised_tokens"] += item[3]
    lengths = [item[2] for item in selected]
    report: dict[str, Any] = {
        "acquisition": {
            "download": download,
            "network_bytes": 0 if download["reused_existing_exact_object"] else download["bytes"],
            "only_authorized_train_object_accessed": True,
            "test_and_ca_test_accessed": False,
        },
        "artifacts": {"private_objects": private_objects, "private_write_bytes": private_bytes},
        "authorization_boundary": {
            "dataset_seal_created": False,
            "external_api_or_inference_used": False,
            "gpu_used": False,
            "production_promoted": False,
            "registry_changed": False,
            "sft_training_run": False,
        },
        "curation": {
            "by_split": {key: dict(value) for key, value in sorted(by_split.items())},
            "content_identity_disjoint": True,
            "decoded_bytes": decoded_bytes,
            "generic_rejection_counts": dict(sorted(generic_rejections.items())),
            "input_rows": input_rows,
            "near_duplicate": near_metrics,
            "selection_rejection_counts": dict(sorted(selection_rejections.items())),
            "serialized_length": {
                "maximum": max(lengths, default=0),
                "mean": mean(lengths) if lengths else 0,
                "median": median(lengths) if lengths else 0,
                "minimum": min(lengths, default=0),
            },
            "token_cap": cap_metrics,
        },
        "evaluation_prohibition": {
            "billsum_ca_test_claimed_evaluation": "permanently_prohibited",
            "billsum_test_claimed_evaluation": "permanently_prohibited",
            "reason": "BillSum train split used for SFT candidate construction",
        },
        "gates": {
            "assistant_only_masking": "PASS",
            "benchmark_decontamination": "PASS",
            "content_identity_split_overlap": "PASS_zero",
            "exact_duplicate": "PASS_zero_retained",
            "four_k_fit": "PASS_untruncated",
            "minhash_0_9": "PASS_zero_retained",
            "numeric_literal_support": "PASS",
            "pii_and_secret": "PASS",
            "provenance": "PASS_neutral_source_summary_origin",
            "punctuation": "PASS",
            "rights": "PASS_cc0_1_0_revision_pinned",
            "schema": "PASS",
            "source_pair_integrity": "PASS_exact_untruncated",
        },
        "observed_at": "2026-08-13",
        "owner_review": {
            "factual_semantic_review": "PENDING_OWNER_REVIEW",
            "minimum_acceptance_rate": 0.9,
            "packet_records": len(owner_sample),
            "packet_sha256": owner_hash,
        },
        "plan_file_sha256": file_sha256(plan_path),
        "production_eligible": False,
        "raw_training_text_retained_in_report": False,
        "runtime_seconds": time.monotonic() - started,
        "schema_version": REPORT_SCHEMA_VERSION,
        "source": {
            "assistant_response_origin": ORIGIN_LABEL,
            "dataset_id": DATASET_ID,
            "license_id": LICENSE_ID,
            "revision": REVISION,
            "source_id": SOURCE_ID,
        },
        "status": "PASS_nonproduction_candidate_owner_review_pending",
        "tokens": {
            "accepted_records": len(records),
            "token_ledger_sha256": sha256_value(token_ledger),
            "train_supervised_tokens": train_tokens,
            "validation_supervised_tokens": validation_tokens,
        },
    }
    report["report_sha256"] = sha256_value(report)
    _write_json_atomic(report_path, report)
    verification = verify_curation(
        plan_path=plan_path, report_path=report_path, repository_root=root, write_output=False
    )
    _write_json_atomic(verification_path, verification)
    return report, verification


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"BillSum JSONL row is not an object: {path}")
            rows.append(value)
    return rows


def verify_curation(
    *,
    plan_path: str | Path,
    report_path: str | Path,
    repository_root: str | Path,
    write_output: bool = True,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    started = time.monotonic()
    root = Path(repository_root).resolve()
    plan = load_plan(plan_path, repository_root=root)
    report = _load_json(report_path)
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported BillSum curation report")
    if report.get("raw_training_text_retained_in_report") is not False:
        raise ValueError("BillSum report is not raw-text-free")
    body = dict(report)
    claimed = body.pop("report_sha256", None)
    if claimed != sha256_value(body):
        raise ValueError("BillSum report internal hash changed")
    if report.get("plan_file_sha256") != file_sha256(plan_path):
        raise ValueError("BillSum report plan binding changed")
    if report.get("acquisition", {}).get("test_and_ca_test_accessed") is not False:
        raise ValueError("BillSum test split boundary changed")

    private_objects = report.get("artifacts", {}).get("private_objects")
    if not isinstance(private_objects, list):
        raise ValueError("BillSum private artifact inventory missing")
    by_name = {str(item["name"]): item for item in private_objects}
    for item in private_objects:
        path = Path(str(item["path"]))
        if not path.is_absolute():
            path = root / path
        if not path.is_file() or file_sha256(path) != item.get("sha256"):
            raise ValueError(f"BillSum artifact changed: {item.get('name')}")
        if path.stat().st_size != item.get("bytes"):
            raise ValueError(f"BillSum artifact size changed: {item.get('name')}")
    raw_path = (
        root / str(plan["output"]["private_root"]) / "private" / "raw" / Path(OBJECT_PATH).name
    )
    if (
        raw_path.stat().st_size != EXPECTED_OBJECT["bytes"]
        or file_sha256(raw_path) != EXPECTED_OBJECT["sha256"]
    ):
        raise ValueError("BillSum raw object changed during verification")
    parquet = pq.ParquetFile(raw_path)
    if parquet.metadata.num_rows != EXPECTED_OBJECT["rows"] or frozenset(
        parquet.schema_arrow.names
    ) != EXPECTED_COLUMNS:
        raise ValueError("BillSum verified parquet inventory changed")

    def artifact_path(name: str) -> Path:
        path = Path(str(by_name[name]["path"]))
        return path if path.is_absolute() else root / path

    records = _read_jsonl(artifact_path("candidate"))
    attributions = _read_jsonl(artifact_path("attribution"))
    token_ledger = _read_jsonl(artifact_path("token_ledger"))
    owner_sample = json.loads(artifact_path("owner_review_json").read_text(encoding="utf-8"))
    if not isinstance(owner_sample, list):
        raise ValueError("BillSum owner packet must be an array")
    policy, benchmark, template, tokenizer = load_runtime_components(plan, repository_root=root)
    seen_ids: set[str] = set()
    seen_normalized: set[str] = set()
    replay_ledger: list[dict[str, Any]] = []
    train_tokens = 0
    validation_tokens = 0
    for record in records:
        _validate_record_binding(record)
        if custom_quality_reasons(record):
            raise ValueError("BillSum retained record fails custom gates")
        decision = evaluate_sft_record(record, policy=policy, benchmark_registry=benchmark)
        if not decision.accepted:
            raise ValueError("BillSum retained record fails generic gates")
        if record["record_id"] in seen_ids or decision.normalized_sha256 in seen_normalized:
            raise ValueError("BillSum retained exact duplicate")
        seen_ids.add(str(record["record_id"]))
        seen_normalized.add(decision.normalized_sha256)
        encoded = encode_sft_messages(record["messages"], tokenizer=tokenizer, template=template)
        if not 64 <= encoded.supervised_tokens <= 512:
            raise ValueError("BillSum retained supervised-token range changed")
        item = {
            "record_id": record["record_id"],
            "serialized_tokens": len(encoded.input_ids),
            "split": record["split"],
            "supervised_tokens": encoded.supervised_tokens,
        }
        replay_ledger.append(item)
        if record["split"] == "train":
            train_tokens += encoded.supervised_tokens
        else:
            validation_tokens += encoded.supervised_tokens
    if replay_ledger != token_ledger:
        raise ValueError("BillSum token ledger changed")
    if not EXPECTED_LIMITS["minimum_train_supervised_tokens"] <= train_tokens <= EXPECTED_LIMITS[
        "maximum_train_supervised_tokens"
    ]:
        raise ValueError("BillSum verified train token target changed")
    if validation_tokens > EXPECTED_LIMITS["maximum_validation_supervised_tokens"]:
        raise ValueError("BillSum verified validation token cap changed")
    candidate_ids = {str(row["record_id"]) for row in records}
    if {str(row["record_id"]) for row in attributions} != candidate_ids:
        raise ValueError("BillSum attribution coverage changed")
    identity_splits: defaultdict[str, set[str]] = defaultdict(set)
    for row in attributions:
        identity_splits[str(row["bill_identity_sha256"])].add(str(row["split"]))
    if any(len(splits) != 1 for splits in identity_splits.values()):
        raise ValueError("BillSum verified identity split overlap")
    measured = [
        (record, attribution, item["serialized_tokens"], item["supervised_tokens"])
        for record, attribution, item in zip(records, attributions, token_ledger, strict=True)
    ]
    near_accepted, near_ledger, _ = apply_near_dedup(measured)
    if len(near_accepted) != len(records) or any(not item["accepted"] for item in near_ledger):
        raise ValueError("BillSum retained near duplicate")
    expected_owner = deterministic_owner_sample(
        records, maximum_records=EXPECTED_LIMITS["owner_review_rows"]
    )
    if owner_sample != expected_owner:
        raise ValueError("BillSum owner-review packet changed")
    owner_hash = sha256_value(owner_sample)
    if report.get("owner_review", {}).get("packet_sha256") != owner_hash:
        raise ValueError("BillSum owner-review hash changed")

    verification: dict[str, Any] = {
        "candidate_file_sha256": file_sha256(artifact_path("candidate")),
        "candidate_records": len(records),
        "evaluation_prohibition_replayed": True,
        "gates_replayed": {
            "assistant_only_masking_and_4k_fit": True,
            "benchmark_decontamination": True,
            "content_identity_disjoint_split": True,
            "exact_duplicate": True,
            "minhash_0_9": True,
            "numeric_literal_support": True,
            "pii_and_secret": True,
            "provenance_and_rights": True,
            "punctuation": True,
            "schema": True,
            "source_pair_integrity": True,
        },
        "owner_review_packet_sha256": owner_hash,
        "raw_object_sha256": file_sha256(raw_path),
        "raw_training_text_retained": False,
        "report_file_sha256": file_sha256(report_path),
        "report_internal_sha256": report["report_sha256"],
        "runtime_seconds": time.monotonic() - started,
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "PASS_independent_nonproduction_candidate_verification",
        "train_supervised_tokens": train_tokens,
        "validation_supervised_tokens": validation_tokens,
    }
    verification["verification_sha256"] = sha256_value(verification)
    if write_output:
        _write_json_atomic(root / str(plan["output"]["verification_path"]), verification)
    return verification


__all__ = [
    "EXPECTED_COLUMNS",
    "EXPECTED_OBJECT",
    "bill_identity",
    "candidate_from_source",
    "custom_quality_reasons",
    "deterministic_owner_sample",
    "download_exact_object",
    "exact_object_url",
    "file_sha256",
    "identity_split",
    "load_plan",
    "run_curation",
    "sha256_value",
    "verify_curation",
]
