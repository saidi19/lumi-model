#!/usr/bin/env python3
"""Generate one deterministic DeepMind Mathematics train-only shard."""

from __future__ import annotations

import argparse
import builtins
import hashlib
import importlib.metadata
import json
import random
import socket
import sys
import urllib.request
import warnings
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

SOURCE_REVISION = "427f45075f84b8b9774950196ad63867ca20ffb3"
TOP_MODULES = (
    "algebra",
    "arithmetic",
    "calculus",
    "comparison",
    "measurement",
    "numbers",
    "polynomials",
    "probability",
)
TOP_MODULE_FILES = {
    name: f"mathematics_dataset/modules/{name}.py" for name in TOP_MODULES
}
ALLOWED_NAMESPACES = {
    "lumi-deepmind-math-train-v0.1",
    "lumi-deepmind-math-validation-v0.1",
}
MAX_QUESTION_CHARACTERS = 160
MAX_ANSWER_CHARACTERS = 30
MAX_ATTEMPTS_PER_RECORD = 128
EXPECTED_FLAT_MODULES = 56
EXPECTED_FLAT_MODULE_INVENTORY_SHA256 = (
    "58870a10ba068a94aff4f32ea2b298199a8788aa1d245e4fdd901d3bf93dd618"
)
STATS_PREFIX = "LUMI_DEEPMIND_STATS="


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _derive_seed(namespace: str, module_file: str, index: int) -> int:
    if namespace not in ALLOWED_NAMESPACES:
        raise ValueError("DeepMind worker namespace is not allowed")
    if module_file not in TOP_MODULE_FILES.values():
        raise ValueError("DeepMind worker module file is not allowed")
    if index < 0:
        raise ValueError("DeepMind worker index must be non-negative")
    payload = f"{namespace}\0{module_file}\0{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _seed_all_generators(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    from sympy.core import random as sympy_random

    sympy_random.seed(seed)


def _flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, item in sorted(value.items()):
        name = f"{prefix}__{key}" if prefix else key
        if isinstance(item, Mapping):
            flat.update(_flatten(item, name))
        else:
            if not callable(item):
                raise ValueError("DeepMind train tree gained a non-callable leaf")
            flat[name] = item
    return flat


def _deny_network(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("network disabled during DeepMind canary generation")


def _install_network_guard() -> None:
    socket.socket = _deny_network  # type: ignore[assignment]
    socket.create_connection = _deny_network  # type: ignore[assignment]
    urllib.request.urlopen = _deny_network  # type: ignore[assignment]


def _install_write_guard() -> None:
    original_open = builtins.open

    def guarded_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if any(marker in mode for marker in ("w", "a", "x", "+")):
            raise RuntimeError("file writes disabled during DeepMind canary generation")
        return original_open(file, mode, *args, **kwargs)

    builtins.open = guarded_open


def _forbid_official_split(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("official DeepMind test/evaluation generation is forbidden")


def _install_determinism_adapter() -> None:
    """Remove two upstream set-order dependencies without changing source files."""

    from mathematics_dataset.util import composition

    def stable_context_pop(self: Any) -> str:
        allowed = sorted(
            composition._ALLOWED_SYMBOLS  # noqa: SLF001 - pinned upstream adapter
            .difference(self._relation_symbols)  # noqa: SLF001
            .difference(self._self_symbols)  # noqa: SLF001
            .difference(self._child_symbols)  # noqa: SLF001
        )
        if not allowed:
            raise ValueError("Ran out of symbols")
        symbol = random.choice(allowed)
        self._self_symbols.add(symbol)  # noqa: SLF001
        return symbol

    def stable_expand_entities(context: Any, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        kwargs = kwargs.copy()
        entities: list[Any] = []
        seen_identity: set[int] = set()

        def append_once(entity: Any) -> None:
            identity = id(entity)
            if identity not in seen_identity:
                seen_identity.add(identity)
                entities.append(entity)

        for entity in context.child_entities:
            append_once(entity)
        for key, maybe_entity in kwargs.items():
            if isinstance(maybe_entity, composition.Entity):
                append_once(maybe_entity)
                kwargs[key] = maybe_entity.handle
        random.shuffle(entities)
        child_descriptions: list[str] = []
        for entity in entities:
            child_descriptions.append(entity.child_description)
            if not entity.expression_used:
                child_descriptions.append(entity.description)
        child_description = " ".join(value for value in child_descriptions if value)
        return child_description, kwargs

    composition.Context.pop = stable_context_pop
    composition.expand_entities = stable_expand_entities


def _install_numpy2_compatibility_adapter() -> None:
    """Replace the two removed ndarray.itemset calls in the pinned source."""

    from mathematics_dataset.sample import polynomials

    def sample_coefficients(
        degrees: Any,
        entropy: float,
        min_non_zero: int = 0,
        max_non_zero: int | None = None,
    ) -> Any:
        if isinstance(degrees, int):
            degrees = [degrees]
        degrees = np.asarray(degrees)

        def random_index() -> list[int]:
            return [random.randint(0, degrees[i]) for i in range(len(degrees))]

        indices: set[tuple[int, ...]] = set()
        for axis, degree in enumerate(degrees):
            if degree > 0:
                index = random_index()
                index[axis] = degree
                indices.add(tuple(index))
        absolute_maximum = np.prod(degrees + 1)
        min_non_zero = max(min_non_zero, 1, len(indices))
        if max_non_zero is None:
            max_non_zero = min_non_zero + int(entropy / 2)
        min_non_zero = min(min_non_zero, absolute_maximum)
        max_non_zero = min(max_non_zero, absolute_maximum)
        max_non_zero = max(min_non_zero, max_non_zero)
        non_zero = random.randint(min_non_zero, max_non_zero)
        while len(indices) < non_zero:
            indices.add(tuple(random_index()))
        coefficients = np.zeros(degrees + 1, dtype=np.int64)
        entropies = entropy * np.random.dirichlet(np.ones(non_zero))
        for index, entry_entropy in zip(sorted(indices), entropies, strict=True):
            value = polynomials.number.integer(entry_entropy, signed=True, min_abs=1)
            coefficients[index] = value
        return coefficients

    def expand_coefficients(
        coefficients: Any,
        entropy: float,
        length: int | None = None,
    ) -> Any:
        coefficients = np.asarray(coefficients)
        shape = coefficients.shape
        expanded = np.empty(shape, dtype=object)
        minimum_length = np.count_nonzero(coefficients) + 2
        if length is None:
            maximum_length = minimum_length + int(polynomials.math.ceil(entropy) / 2)
            length = random.randint(minimum_length, maximum_length)
        if length < minimum_length:
            length = minimum_length
        is_zero_flat = np.reshape(coefficients, [-1]) == 0
        counts = polynomials.expanded_coefficient_counts(length, is_zero=is_zero_flat)
        coefficient_entropies = entropy * np.random.dirichlet(
            np.maximum(1e-9, counts - 1)
        )
        counts = np.reshape(counts, shape)
        coefficient_entropies = np.reshape(coefficient_entropies, shape)
        indices = list(zip(*np.indices(shape).reshape([len(shape), -1]), strict=True))
        for power in indices:
            values = polynomials.integers_with_sum(
                value=coefficients.item(power),
                count=counts.item(power),
                entropy=coefficient_entropies.item(power),
            )
            expanded[power] = values
        return expanded

    polynomials.sample_coefficients = sample_coefficients
    polynomials.expand_coefficients = expand_coefficients


def _install_python313_compatibility_adapter() -> None:
    """Keep one pinned Python-2-era randint call valid on Python 3.13."""

    from mathematics_dataset.modules import numbers

    original_round_number = numbers.round_number

    def round_number_with_integer_randint_bounds(*args: Any, **kwargs: Any) -> Any:
        original_randint = random.randint

        def integer_bounds_randint(lower: Any, upper: Any) -> int:
            def normalize(value: Any) -> Any:
                if isinstance(value, float):
                    if not value.is_integer():
                        raise TypeError("randint bound is not integer-valued")
                    return int(value)
                return value

            return original_randint(normalize(lower), normalize(upper))

        random.randint = integer_bounds_randint
        try:
            return original_round_number(*args, **kwargs)
        finally:
            random.randint = original_randint

    numbers.round_number = round_number_with_integer_randint_bounds


def _load_train_modules(
    source_root: Path,
    top_module: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_source = source_root / "mathematics_dataset/modules/modules.py"
    if not expected_source.is_file():
        raise FileNotFoundError("DeepMind worker source root is invalid")
    if str(source_root) not in sys.path:
        raise ValueError("DeepMind source root is not first-party worker input")

    sys.dont_write_bytecode = True
    _seed_all_generators(0)
    _install_network_guard()
    warnings.filterwarnings("ignore")

    from mathematics_dataset.modules import modules

    _install_determinism_adapter()
    _install_numpy2_compatibility_adapter()
    _install_python313_compatibility_adapter()
    for upstream_module in modules.all_.values():
        upstream_module.test = _forbid_official_split
        upstream_module.test_extra = _forbid_official_split
    modules.test = _forbid_official_split
    modules.test_extra = _forbid_official_split
    train_tree = modules.train(lambda value: value)
    flat = _flatten(train_tree)
    names = sorted(flat)
    if len(names) != EXPECTED_FLAT_MODULES:
        raise ValueError("DeepMind flat train-module count changed")
    if _sha256_value(names) != EXPECTED_FLAT_MODULE_INVENTORY_SHA256:
        raise ValueError("DeepMind flat train-module inventory changed")
    selected = {name: fn for name, fn in flat.items() if name.startswith(f"{top_module}__")}
    if not selected:
        raise ValueError("DeepMind requested top module has no train leaves")

    loaded_source_modules = sorted(
        name
        for name in sys.modules
        if name == "mathematics_dataset" or name.startswith("mathematics_dataset.")
    )
    forbidden_loaded = {
        "mathematics_dataset.generate",
        "mathematics_dataset.generate_settings",
        "mathematics_dataset.generate_to_file",
    }.intersection(loaded_source_modules)
    if forbidden_loaded:
        raise RuntimeError(
            f"DeepMind worker imported excluded generator modules: {sorted(forbidden_loaded)}"
        )
    _install_write_guard()
    audit = {
        "dependency_versions": {
            "absl-py": importlib.metadata.version("absl-py"),
            "numpy": importlib.metadata.version("numpy"),
            "six": importlib.metadata.version("six"),
            "sympy": importlib.metadata.version("sympy"),
        },
        "determinism_adapter": [
            "numpy2_itemset_compatibility_with_stable_coefficient_index_order",
            "python313_integer_valued_randint_bounds_for_round_number",
            "stable_context_symbol_order_before_seeded_choice",
            "stable_entity_identity_dedup_before_seeded_shuffle",
            "sympy_internal_rng_seeded_from_record_seed",
        ],
        "flat_module_count": len(names),
        "flat_module_inventory_sha256": _sha256_value(names),
        "loaded_source_modules": loaded_source_modules,
        "network_guard_active": True,
        "official_test_calls": 0,
        "official_test_guards_installed": True,
        "source_revision": SOURCE_REVISION,
        "upstream_generate_cli_imported": False,
        "write_guard_active": True,
    }
    return selected, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--top-module", choices=TOP_MODULES, required=True)
    parser.add_argument("--namespace", choices=sorted(ALLOWED_NAMESPACES), required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    if not 1 <= args.count <= 238:
        raise ValueError("DeepMind worker shard count is outside the bounded schedule")
    module_file = TOP_MODULE_FILES[args.top_module]
    selected, audit = _load_train_modules(source_root, args.top_module)
    names = sorted(selected)

    dropped_length = 0
    exception_counts: Counter[str] = Counter()
    module_sample_counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for index in range(args.count):
        seed = _derive_seed(args.namespace, module_file, index)
        _seed_all_generators(seed)
        module_name = names[index % len(names)]
        module_fn = selected[module_name]
        accepted = None
        for _attempt in range(MAX_ATTEMPTS_PER_RECORD):
            try:
                problem = module_fn()
                question = str(problem.question)
                answer = str(problem.answer)
            except Exception as error:  # noqa: BLE001 - fail-closed bounded upstream canary
                exception_counts[type(error).__name__] += 1
                continue
            if len(question) > MAX_QUESTION_CHARACTERS or len(answer) > MAX_ANSWER_CHARACTERS:
                dropped_length += 1
                continue
            if not question.strip() or not answer.strip():
                exception_counts["EmptyProblem"] += 1
                continue
            accepted = {
                "answer": answer,
                "module": module_name,
                "question": question,
                "seed": seed,
                "split_namespace": args.namespace,
            }
            break
        if accepted is None:
            raise RuntimeError(
                "DeepMind worker exhausted bounded attempts without exposing source text: "
                f"top_module={args.top_module} index={index}"
            )
        records.append(accepted)
        module_sample_counts[module_name] += 1

    for record in records:
        sys.stdout.buffer.write(_canonical_json_bytes(record) + b"\n")
    sys.stdout.buffer.flush()
    stats = {
        **audit,
        "dropped_length": dropped_length,
        "exception_counts": dict(sorted(exception_counts.items())),
        "module_sample_counts": dict(sorted(module_sample_counts.items())),
        "namespace": args.namespace,
        "records": len(records),
        "top_module": args.top_module,
    }
    print(STATS_PREFIX + json.dumps(stats, sort_keys=True, separators=(",", ":")), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
