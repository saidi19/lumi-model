"""Reproduce saved safety outputs using authenticated historical computations.

All historical file IO is redirected to an explicit immutable byte snapshot,
not the real filesystem. Validation, repair, filtering, counting and serializers
retain their frozen source bodies. The only substituted computation entry is
tokenizer construction from the same serialized bytes, tested for equivalence.
No original one-shot runner, publisher or original pytest node is executed.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import posixpath
import sys
import types
from collections import Counter

from .posttraining_safety_replay_snapshot_v1 import Snapshot
from .posttraining_safety_tokenizer_memory_v1 import load_tokenizer

PACKAGE = "_lumi_authenticated_safety_history"
SOURCE_MODULES = (
    ("data_quality", "data_quality_source"),
    ("posttraining_data", "posttraining_data_implementation"),
    ("posttraining_sources", "source_registry_implementation"),
    ("tokenizer_gate", "tokenizer_gate_source"),
    ("posttraining_format", "format_source"),
    ("posttraining_safety_uri_repair", "implementation"),
)
STAGES = frozenset(
    {
        "payload_authentication",
        "source_loading",
        "plan_validation",
        "chain_validation",
        "policy_validation",
        "row_repair",
        "filter_validation",
        "tokenizer_construction",
        "token_measurement",
        "output_serialization",
        "output_comparison",
        "verification_reproduction",
    }
)


def require(condition):
    if not condition:
        raise ValueError("invalid safety output replay")


def authenticate_payloads(payloads, bindings):
    require(type(payloads) is dict and set(payloads) == set(bindings))
    for role, binding in bindings.items():
        value = payloads[role]
        require(type(value) is bytes and len(value) == binding["bytes"])
        require(hashlib.sha256(value).hexdigest() == binding["sha256"])


def load_historical_modules(payloads, snapshot):
    require(not any(name == PACKAGE or name.startswith(PACKAGE + ".") for name in sys.modules))
    package = types.ModuleType(PACKAGE)
    package.__path__ = []
    sys.modules[PACKAGE] = package
    modules = {}
    for name, role in SOURCE_MODULES:
        qualified = PACKAGE + "." + name
        module = types.ModuleType(qualified)
        module.__package__ = PACKAGE
        sys.modules[qualified] = module
        # Full source bytes were authenticated before this point. No function
        # validator body is patched; Path/gzip are the explicit input IO seam.
        exec(compile(payloads[role], "<authenticated-" + name + ">", "exec"), module.__dict__)
        module.Path = snapshot.Path
        if name == "data_quality":
            module.gzip = types.SimpleNamespace(open=snapshot.gzip_open)
        modules[name] = module
    return modules


def frozen_serializer(source, namespace):
    """Extract the unchanged pure serialization tail, excluding loader calls."""
    body = next(
        node.body
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "rebuild_safety_uri_repair_outputs"
    )
    starts = [
        index
        for index, node in enumerate(body)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "manifest"
    ]
    require(len(starts) == 1)
    names = (
        "plan",
        "plan_file_sha256",
        "benchmark_registry",
        "source_registry",
        "repair_metrics",
        "generic_report",
        "by_split",
        "candidate_bytes",
        "template",
    )
    function = ast.FunctionDef(
        name="_frozen_serialization_tail",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=name) for name in names],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=copy.deepcopy(body[starts[0] :]),
        decorator_list=[],
    )
    tree = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), "<frozen-serialization-tail>", "exec"), namespace)
    return namespace["_frozen_serialization_tail"]


def reproduce(payloads, bindings, repository_root, *, on_stage=None):
    """Return aggregates only after all four saved output byte strings match."""

    def stage(code):
        if on_stage is not None:
            on_stage(code)

    stage("payload_authentication")
    authenticate_payloads(payloads, bindings)
    blobs = {
        posixpath.normpath(repository_root + "/" + binding["path"]): payloads[role]
        for role, binding in bindings.items()
    }
    require(len(blobs) == len(bindings))
    snapshot = Snapshot(blobs, repository_root=repository_root)
    require(not any(name == PACKAGE or name.startswith(PACKAGE + ".") for name in sys.modules))
    try:
        stage("source_loading")
        modules = load_historical_modules(payloads, snapshot)
        repair = modules["posttraining_safety_uri_repair"]
        root = snapshot.Path(repository_root)
        stage("plan_validation")
        plan = repair.load_safety_uri_repair_plan(
            root / bindings["plan"]["path"], repository_root=root
        )
        paths = {
            name: repair.resolve_binding(root, binding, name=name)
            for name, binding in plan["bindings"].items()
        }
        stage("chain_validation")
        repair._validate_predecessor_chain(
            manifest=repair._load_json(paths["predecessor_hygiene_manifest"]),
            report=repair._load_json(paths["predecessor_hygiene_report"]),
            verification=repair._load_json(paths["predecessor_hygiene_verification"]),
        )
        repair._validate_owner_receipt(repair._load_json(paths["safety_owner_review_receipt"]))
        stage("policy_validation")
        policy = repair.load_sft_policy(paths["data_contract"])
        benchmark = repair.BenchmarkRegistry(policy.benchmark_decontamination, root)
        require(benchmark.sha256 == plan["bindings"]["benchmark_registry"]["sha256"])
        registry = repair.load_sft_source_registry(
            paths["source_registry"],
            allowed_licenses=policy.allowed_licenses,
            allowed_languages=policy.allowed_languages,
            allowed_splits=policy.allowed_splits,
            allowed_tasks=frozenset(policy.task_mix),
            expected_sha256=repair.EXPECTED_REGISTRY_SHA256,
        )
        # Exact bytes are already externally pinned. Require the explicit line
        # budget before calling the unchanged historical row parser.
        require(payloads["predecessor_candidate"].count(b"\n") == 320)
        require(payloads["output_candidate"].count(b"\n") == 320)
        stage("row_repair")
        rows, candidate_bytes, metrics = repair._repair_rows(
            repair._load_jsonl(paths["predecessor_candidate"])
        )
        stage("filter_validation")
        accepted, generic = repair.filter_sft_records(
            rows,
            policy=policy,
            benchmark_registry=benchmark,
            source_registry=registry,
        )
        require(
            len(accepted) == 320 and generic["rejected"] == 0 and not generic["rejection_counts"]
        )
        stage("tokenizer_construction")
        template = repair.load_chat_template(paths["chat_template"])
        require(paths["tokenizer_json"].parent == paths["tokenizer_config"].parent)
        tokenizer = load_tokenizer(payloads["tokenizer_json"], payloads["tokenizer_config"])
        repair.verify_chat_tokenizer(tokenizer, template)
        stage("token_measurement")
        counts, tokens = Counter(), Counter()
        for row in rows:
            split = row["split"]
            counts[split] += 1
            tokens[split] += repair._measure_conversation(
                row["messages"], tokenizer=tokenizer, template=template
            )
        by_split = {
            split: {"records": counts[split], "supervised_tokens": tokens[split]}
            for split in ("train", "validation")
        }
        require(by_split == repair.EXPECTED_SAFETY_TOKENS)
        stage("output_serialization")
        serializer = frozen_serializer(payloads["implementation"], repair.__dict__)
        outputs = serializer(
            plan,
            bindings["plan"]["sha256"],
            benchmark,
            registry,
            metrics,
            generic,
            by_split,
            candidate_bytes,
            template,
        )
        stage("output_comparison")
        require(outputs["private_safety_candidate_bytes"] == payloads["output_candidate"])
        require(outputs["artifact_manifest_bytes"] == payloads["output_manifest"])
        require(repair.json_file_bytes(outputs["report"]) == payloads["output_report"])
        stage("verification_reproduction")
        verification = repair.build_safety_uri_repair_verification(
            plan=plan,
            report=outputs["report"],
            report_file_sha256=bindings["output_report"]["sha256"],
        )
        require(repair.json_file_bytes(verification) == payloads["output_verification"])
        return {
            "status": "PASS_SAFETY_OUTPUT_REPRODUCTION_ONLY",
            "safety_outputs_reproduced": True,
            "saved_outputs_byte_matched": 4,
            "candidate_rows_reconstructed": 320,
            "generic_filter_accepted": len(accepted),
            "generic_filter_rejected": 0,
            "tokenized_rows": 320,
            "by_split": by_split,
            "original_test_nodes_executed": 0,
            "full_private_integration_passed": False,
            "execution_authorized": False,
            "production_eligible": False,
            "snapshot_accounting": snapshot.summary(),
        }
    finally:
        for name in tuple(sys.modules):
            if name == PACKAGE or name.startswith(PACKAGE + "."):
                del sys.modules[name]
