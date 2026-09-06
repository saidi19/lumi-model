"""Invented metadata tests for the two-node Aya promotion projection.

No existing registry, attribution or receipt body is read. Positive tests use
two explicit exceptions only: the test namespace's predecessor-hash assertion
and the promotion namespace's two historical private seals. The original file
hash still runs before that assertion double returns its fixture value. Every
other digest, including the generated synthetic promotion receipt, is real.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import sys
import types
import unittest
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from lumi import posttraining_aya_promotion_snapshot_v1 as projection

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_ROOT = "/synthetic/workspace/repo"
ERROR = "invalid Aya promotion source projection"
FIXTURE = ROOT / "tests/fixtures/aya_source_registry_9f20652.py.txt"
SOURCE_ROLES = (
    "promotion_implementation", "hash_helper_source", "test_source", "policy_source",
)
APPROVED_PREDECESSOR = "e85f591f2a93b1384ed9ce4ee3da0e067bcfc07a1ade2e0b170eb875c7677914"
PRIVATE_SEALS = {
    "lumi.sft-aya-bengali-materialization-report.v0.1": (
        "a2f08b1594a2b6fd9def1f43a48e072cbe0f107f4bc7ea43b391d6066ddc6ade"
    ),
    "lumi.sft-aya-bengali-owner-review-receipt.v0.1": (
        "e23851af5718d10acdd120ca1e0c200672fb5c794ad503ed4d1f7f1650cc685f"
    ),
}


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def pin(payloads, bindings, role):
    bindings[role] = {
        "path": projection.SOURCE_PATHS[role], "bytes": len(payloads[role]),
        "sha256": hashlib.sha256(payloads[role]).hexdigest(),
    }


def repin_plan(payloads, bindings):
    plan = json.loads(payloads["plan"])
    plan["bindings"] = {
        role: {field: bindings[role][field] for field in ("path", "sha256")}
        for role in projection.PLAN_BINDING_ROLES
    }
    payloads["plan"] = encoded(plan)
    pin(payloads, bindings, "plan")


@contextmanager
def explicit_synthetic_doubles(module):
    """Patch only the two declared synthetic accommodations, never production code."""
    promotion = module.build_source_promotion_receipt.__globals__
    original_hash, original_seal = module.file_sha256, promotion["sha256_value"]

    def test_predecessor_assertion_only(path):
        observed = original_hash(path)  # Preserve real hashing and snapshot accounting.
        return APPROVED_PREDECESSOR if str(path) == str(module.PREDECESSOR) else observed

    def two_private_schema_seals_only(value):
        return PRIVATE_SEALS.get(value.get("schema_version"), original_seal(value))

    with (
        patch.object(module, "file_sha256", side_effect=test_predecessor_assertion_only),
        patch.dict(promotion, {"sha256_value": two_private_schema_seals_only}),
    ):
        assert promotion["file_sha256"] is original_hash
        yield


def synthetic_inputs():
    """Read only four original public sources and the approved public source fixture."""
    payloads = {role: ("synthetic " + role).encode() for role in projection.INPUT_ROLES}
    for role in SOURCE_ROLES:
        payloads[role] = (ROOT / projection.SOURCE_PATHS[role]).read_bytes()
    payloads["source_registry_implementation"] = FIXTURE.read_bytes()
    constants = {"re": re}
    projection._compile(
        projection._select(payloads["promotion_implementation"], projection.PROMOTION_CONSTANTS,
                           frozenset()), "<synthetic-constants>", constants,
    )
    tasks = list(constants["EXPECTED_REQUIRED_TASKS"])
    policy = {
        "schema_version": "lumi.sft.v0.1", "allowed_languages": ["bn", "en"],
        "allowed_licenses": ["apache-2.0", "lumi-owned"],
        "allowed_splits": ["train", "validation"],
        "filters": {"minimum_quality_score": 0.75, "maximum_messages": 64,
                    "maximum_message_characters": 65536},
        "task_mix": dict.fromkeys(tasks, 0.125),
        "benchmark_decontamination": {"registry": "synthetic/not-opened.json",
                                      "registry_sha256": "b" * 64},
        "chat_training": {
            "assistant_only_loss": True,
            "chat_template": "configs/lumi-4b-chat-template-v0.1.json",
            "context_window": 4096, "overlength_policy": "reject",
            "packing_policy": "single_conversation_until_block_diagonal_verified",
            "supervised_roles": ["assistant", "tool"],
        },
        "source_registry": {
            "schema_version": "lumi.sft.sources.v0.1", "required_for_filtering_and_seal": True,
            "path": "configs/lumi-4b-sft-source-registry-v0.1.json",
            "sha256": "f73676bbfab84f9c421ff3eb611d8e395c0bdfb50a9fa87f5559d98ac74107c9",
        },
    }
    common = {
        "adapter": "synthetic", "blockers": [], "languages": ["bn"],
        "production_blockers": ["synthetic_nonproduction"], "production_eligible": False,
        "splits": ["train"],
    }
    sources = [{
        **common, "access": "first_party", "origin": "synthetic", "status": "candidate_eligible",
        "source_id": "synthetic_task_cover", "dataset_id": "synthetic/task-cover",
        "license_id": "lumi-owned", "revision": "a" * 64, "repository_url": None,
        "evidence_urls": [], "task_categories": tasks,
    }, copy.deepcopy(constants["EXPECTED_SOURCE_BEFORE"])]
    for index in range(2, 10):
        repository = f"https://huggingface.co/datasets/synthetic/source-{index}"
        sources.append({
            **common, "access": "public", "origin": "human", "status": "canary_eligible",
            "source_id": f"synthetic_source_{index}", "dataset_id": f"synthetic/source-{index}",
            "license_id": "apache-2.0", "revision": "c" * 40, "repository_url": repository,
            "evidence_urls": [repository], "task_categories": ["coding"],
        })
    predecessor = {
        "schema_version": "lumi.sft.sources.v0.1", "observed_at": "2026-08-06",
        "authorization": {
            "bulk_data_download_authorized": False, "paid_gpu_authorized": False,
            "production_sft_authorized": False,
            "purpose": "metadata_and_source_admissibility_audit_only",
        },
        "sources": sources,
    }
    successor = copy.deepcopy(predecessor)
    successor["sources"][1] = copy.deepcopy(constants["EXPECTED_SOURCE_AFTER"])
    payloads["data_contract"] = encoded(policy)
    payloads["predecessor_registry"] = encoded(predecessor)
    payloads["successor_registry"] = encoded(successor)
    payloads["promotion_runner"] = b"\xff synthetic runner must never execute"
    materialization_schema, owner_schema = tuple(PRIVATE_SEALS)
    payloads["materialization_report"] = encoded({
        "schema_version": materialization_schema,
        "candidate": {"materialized": True, "records": 1478, "supervised_tokens": 577453},
        "raw_training_text_in_report": False, "production": {"production_eligible": False},
        "report_sha256": PRIVATE_SEALS[materialization_schema],
    })
    owner = {
        "schema_version": owner_schema,
        "status": "owner_quality_sample_passed_source_promotion_still_separate",
        "sample_records": 32, "accepted_records": 32, "rejected_records": 0,
        "all_sampled_rows_accepted": True, "production_eligible": False,
        "production_promotions": 0, "raw_training_text_retained": False,
        "receipt_sha256": PRIVATE_SEALS[owner_schema],
    }
    payloads["owner_review_receipt"] = encoded(owner)
    payloads["attribution_manifest"] = b"\n".join(canonical({
        "record_id": f"synthetic-{index:04d}",
        "schema_version": "lumi.sft-aya-bengali-attribution.v0.1",
        "source_id": "coherelabs_aya_dataset", "dataset_id": "CohereLabs/aya_dataset",
        "revision": constants["EXPECTED_SOURCE_AFTER"]["revision"], "license_id": "apache-2.0",
    }) for index in range(1478)) + b"\n"
    payloads["promotion_receipt"] = b"{}"
    payloads["plan"] = encoded({
        "schema_version": constants["PLAN_SCHEMA_VERSION"],
        "status": "authorized_candidate_registry_successor_promotion", "observed_at": "2026-08-06",
        **{field: copy.deepcopy(constants["EXPECTED_" + field.upper()]) for field in (
            "authorization", "limits", "output", "curated_metrics", "required_tasks",
            "source_before", "source_after",
        )},
    })
    bindings = {}
    for role in projection.INPUT_ROLES:
        pin(payloads, bindings, role)
    repin_plan(payloads, bindings)
    # A separate synthetic construction creates an expected receipt in memory.
    # Its authentic builder uses real digest functions; no private-seal double is needed here.
    module, _ = projection.build_test_module(payloads, bindings, SYNTHETIC_ROOT)
    plan = json.loads(payloads["plan"])
    parsed_policy, parsed_predecessor, predecessor_value = module.load_policy_and_predecessor(
        plan=plan, repository_root=module.ROOT,
    )
    parsed_successor = module.validate_successor_registry(
        policy=parsed_policy, predecessor_value=predecessor_value, successor_value=successor,
        successor_file_sha256=bindings["successor_registry"]["sha256"],
    )
    receipt = module.build_source_promotion_receipt(
        owner_review_receipt=owner, plan=plan, plan_file_sha256=bindings["plan"]["sha256"],
        predecessor=parsed_predecessor, successor=parsed_successor,
    )
    payloads["promotion_receipt"] = encoded(receipt)
    pin(payloads, bindings, "promotion_receipt")
    return payloads, bindings


class AyaPromotionSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.payloads, self.bindings = synthetic_inputs()

    def build(self):
        return projection.build_test_module(self.payloads, self.bindings, SYNTHETIC_ROOT)

    def reject(self):
        with self.assertRaisesRegex(ValueError, f"^{ERROR}$"):
            self.build()

    def repin(self, role):
        pin(self.payloads, self.bindings, role)
        if role in projection.PLAN_BINDING_ROLES:
            repin_plan(self.payloads, self.bindings)

    def test_both_original_bodies_pass_with_explicit_doubles_and_35_opens(self):
        module, snapshot = self.build()
        self.assertEqual(snapshot.summary()["virtual_opens"], 0)
        with explicit_synthetic_doubles(module):
            self.assertIsNone(getattr(module, projection.TEST_NAMES[0])())
            self.assertEqual(snapshot.summary()["virtual_opens"], 16)
            self.assertIsNone(getattr(module, projection.TEST_NAMES[1])())
        counts = {role: 2 for role in projection.PLAN_BINDING_ROLES}
        counts.update({"plan": 3, "data_contract": 4, "predecessor_registry": 9,
                       "successor_registry": 3, "promotion_receipt": 1,
                       "attribution_manifest": 3, "materialization_report": 3,
                       "owner_review_receipt": 3})
        expected = sum(self.bindings[role]["bytes"] * count for role, count in counts.items())
        self.assertEqual(snapshot.summary(), {
            "bound_blob_count": 11, "virtual_opens": 35,
            "logical_read_bytes_debited": expected, "maximum_virtual_opens": 35,
            "maximum_logical_read_bytes": expected, "maximum_gzip_bytes": 1,
            "filesystem_access_performed": False, "execution_permitted": False,
        })

    def test_original_hashes_remain_real_and_unmodified_namespace_seals_fail(self):
        module, snapshot = self.build()
        promotion = module.build_source_promotion_receipt.__globals__
        self.assertIs(module.file_sha256, promotion["file_sha256"])
        value = {"synthetic": "বাংলা", "integer": 3}
        self.assertEqual(promotion["sha256_value"](value),
                         hashlib.sha256(canonical(value)).hexdigest())
        self.assertEqual(module.file_sha256(module.PREDECESSOR),
                         self.bindings["predecessor_registry"]["sha256"])
        with self.assertRaises(AssertionError):
            getattr(module, projection.TEST_NAMES[0])()
        module, _ = self.build()
        with self.assertRaisesRegex(ValueError, "^report_sha256 changed$"):
            getattr(module, projection.TEST_NAMES[1])()
        self.assertGreater(snapshot.summary()["virtual_opens"], 0)

    def test_ast_locations_classes_and_namespaces_preserve_original_sources(self):
        module, _ = self.build()
        promotion = module.build_source_promotion_receipt.__globals__
        policy = promotion["load_sft_policy"].__globals__
        registry = promotion["load_sft_source_registry"].__globals__
        helpers = module.file_sha256.__globals__
        namespaces = (module.__dict__, promotion, policy, registry, helpers)
        self.assertEqual(len({id(namespace) for namespace in namespaces}), 5)
        selections = {
            "policy_source": (projection.POLICY_CONSTANTS, projection.POLICY_FUNCTIONS,
                              frozenset({"SFTPolicy"}), policy),
            "source_registry_implementation": (
                projection.REGISTRY_CONSTANTS, projection.REGISTRY_FUNCTIONS,
                frozenset({"SFTSource", "SFTSourceRegistry"}), registry,
            ),
            "promotion_implementation": (
                projection.PROMOTION_CONSTANTS, projection.PROMOTION_FUNCTIONS,
                frozenset(), promotion,
            ),
            "hash_helper_source": (
                frozenset(), frozenset({"canonical_json_bytes", "file_sha256"}),
                frozenset(), helpers,
            ),
            "test_source": (
                frozenset(), frozenset({"_load", "_prerequisites", *projection.TEST_NAMES}),
                frozenset(), module.__dict__,
            ),
        }
        for role, (constants, functions, classes, namespace) in selections.items():
            original = ast.parse(self.payloads[role])
            selected = projection._select(self.payloads[role], constants, functions, classes)
            for node in selected:
                name = node.targets[0].id if isinstance(node, ast.Assign) else node.name
                match = next(item for item in original.body if (
                    isinstance(item, ast.Assign) and isinstance(item.targets[0], ast.Name)
                    and item.targets[0].id == name
                    or isinstance(item, (ast.FunctionDef, ast.ClassDef)) and item.name == name
                ))
                with self.subTest(role=role, name=name):
                    self.assertEqual(ast.dump(node, include_attributes=True),
                                     ast.dump(match, include_attributes=True))
                    if name in functions:
                        code = namespace[name].__code__
                        self.assertEqual(code.co_firstlineno, match.lineno)
                        self.assertEqual(code.co_filename,
                                         SYNTHETIC_ROOT + "/" + projection.SOURCE_PATHS[role])
                    if name in classes:
                        self.assertTrue(namespace[name].__dataclass_params__.frozen)
                        methods = (item for item in node.body if isinstance(item, ast.FunctionDef))
                        for member in methods:
                            function = getattr(namespace[name], member.name)
                            if isinstance(function, property):
                                function = function.fget
                            first_line = min(member.lineno, *(
                                decorator.lineno for decorator in member.decorator_list
                            )) if member.decorator_list else member.lineno
                            self.assertEqual(function.__code__.co_firstlineno, first_line)
                            self.assertEqual(function.__code__.co_filename,
                                             SYNTHETIC_ROOT + "/" + projection.SOURCE_PATHS[role])
        plan = json.loads(self.payloads["plan"])
        parsed, _, _ = module.load_policy_and_predecessor(plan=plan, repository_root=module.ROOT)
        with self.assertRaises(FrozenInstanceError):
            parsed.context_window = 1

    def test_no_real_file_operations_and_no_embedded_path_traversal(self):
        with (
            patch("builtins.open", side_effect=AssertionError("real open")),
            patch("io.open", side_effect=AssertionError("real open")),
            patch("os.open", side_effect=AssertionError("real open")),
            patch("os.stat", side_effect=AssertionError("real stat")),
            patch("os.lstat", side_effect=AssertionError("real stat")),
        ):
            module, _ = self.build()
            with explicit_synthetic_doubles(module):
                for name in projection.TEST_NAMES:
                    getattr(module, name)()

    def test_all_inputs_authenticate_before_json_or_ast(self):
        for role in projection.INPUT_ROLES:
            with self.subTest(role=role):
                original = self.payloads[role]
                self.payloads[role] = original + b" "
                with (patch.object(ast, "parse") as parse,
                      patch.object(projection.json, "loads") as loads):
                    self.reject()
                    parse.assert_not_called()
                    loads.assert_not_called()
                self.payloads[role] = original

    def test_roles_paths_binding_types_and_hashes_are_exact(self):
        original = copy.deepcopy(self.bindings)
        for role in projection.INPUT_ROLES:
            for field, value in (("bytes", True), ("bytes", 0), ("sha256", "0" * 64),
                                 ("path", "unapproved/path"), ("physical_read_path", "copy")):
                with self.subTest(role=role, field=field):
                    self.bindings = copy.deepcopy(original)
                    self.bindings[role][field] = value
                    self.reject()
        self.bindings = original
        for field in ("payloads", "bindings"):
            for extra in (False, True):
                with self.subTest(field=field, extra=extra):
                    mapping = getattr(self, field)
                    original = dict(mapping)
                    if extra:
                        mapping["extra"] = mapping["plan"]
                    else:
                        del mapping["plan"]
                    self.reject()
                    setattr(self, field, original)

    def test_source_and_metadata_and_total_budgets_are_enforced(self):
        for role, maximum in (("policy_source", projection.MAXIMUM_SOURCE_BYTES),
                              ("successor_registry", projection.MAXIMUM_METADATA_BYTES)):
            with self.subTest(role=role):
                old_payload, old_binding = self.payloads[role], self.bindings[role]
                self.payloads[role] = b"x" * (maximum + 1)
                self.repin(role)
                self.reject()
                self.payloads[role], self.bindings[role] = old_payload, old_binding
        with patch.object(projection, "MAXIMUM_TOTAL_BYTES", 1):
            self.reject()

    def test_plan_duplicate_keys_nonfinite_values_and_unmatched_binding_rejected(self):
        original = self.payloads["plan"]
        value = json.loads(original)
        wrong = copy.deepcopy(value)
        wrong["bindings"]["source_registry_implementation"]["path"] = "tests/fixtures/copy.txt"
        cases = (
            b'{"bindings":{},' + original[1:],
            encoded(value).rstrip()[:-1] + b',"nonfinite":NaN}', encoded(wrong), b'[]',
        )
        for case in cases:
            with self.subTest(length=len(case)):
                self.payloads["plan"] = case
                self.repin("plan")
                self.reject()

    def test_private_inputs_remain_unparsed_during_construction(self):
        for role in ("attribution_manifest", "materialization_report", "owner_review_receipt",
                     "successor_registry", "promotion_receipt"):
            self.payloads[role] = b"synthetic unparsable \xff"
            self.repin(role)
        _, snapshot = self.build()
        self.assertEqual(snapshot.summary()["virtual_opens"], 0)

    def test_class_decorators_bases_members_and_defaults_fail_closed(self):
        original = self.payloads["policy_source"]
        mutations = (
            original.replace(b"@dataclass(frozen=True)\nclass SFTPolicy",
                             b"@evil()\nclass SFTPolicy"),
            original.replace(b"class SFTPolicy:", b"class SFTPolicy(object):"),
            original.replace(b"class SFTPolicy:", b"class SFTPolicy(metaclass=type):"),
            original.replace(b"class SFTPolicy:\n", b"class SFTPolicy:\n    danger = evil()\n"),
            original.replace(b"    @classmethod\n    def from_mapping",
                             b"    @evil\n    def from_mapping"),
            original.replace(b"learning_candidate: Mapping[str, Any] | None = None",
                             b"learning_candidate: Mapping[str, Any] | None = evil()"),
            original + b"\n@dataclass(frozen=True)\nclass SFTPolicy:\n    value: str\n",
        )
        for source in mutations:
            with self.subTest(length=len(source)):
                self.payloads["policy_source"] = source
                self.repin("policy_source")
                self.reject()

    def test_only_exact_literal_function_defaults_are_allowed(self):
        original = self.payloads["source_registry_implementation"]
        replacements = (
            (b"allow_empty: bool = False", b"allow_empty: bool = 0"),
            (b"allow_empty: bool = False", b"allow_empty: bool = evil()"),
            (b"expected_sha256: str | None = None", b"expected_sha256: str | None = False"),
        )
        for old, new in replacements:
            with self.subTest(replacement=new):
                self.payloads["source_registry_implementation"] = original.replace(old, new)
                self.repin("source_registry_implementation")
                self.reject()

    def test_constant_references_are_allowlisted_without_evaluating_arbitrary_code(self):
        original = self.payloads["policy_source"]
        for replacement in (b"unknown_name", b"evil()"):
            self.payloads["policy_source"] = original.replace(
                b'"benchmark_decontamination": _PRIVATE_RESEARCH_BENCHMARK_CONTRACT',
                b'"benchmark_decontamination": ' + replacement,
            )
            self.repin("policy_source")
            self.reject()

    def test_class_namespace_collisions_are_preserved_and_cleaned_up(self):
        for base in projection.CLASS_MODULE_NAMES:
            for name in (base, base + ".child"):
                sentinel = types.ModuleType(name)
                with self.subTest(name=name), patch.dict(sys.modules, {name: sentinel}):
                    self.reject()
                    self.assertIs(sys.modules[name], sentinel)
        self.build()
        self.assertTrue(all(name not in sys.modules for name in projection.CLASS_MODULE_NAMES))

    def test_class_namespace_cleanup_on_compile_failure_and_foreign_replacement(self):
        original_compile = projection._compile

        def fail_registry(nodes, filename, namespace):
            if namespace.get("__name__") == projection.CLASS_MODULE_NAMES[1]:
                raise RuntimeError("synthetic private diagnostic must be suppressed")
            return original_compile(nodes, filename, namespace)

        with patch.object(projection, "_compile", side_effect=fail_registry):
            self.reject()
        self.assertTrue(all(name not in sys.modules for name in projection.CLASS_MODULE_NAMES))
        sentinel = types.ModuleType(projection.CLASS_MODULE_NAMES[0])

        def replace_policy(nodes, filename, namespace):
            original_compile(nodes, filename, namespace)
            if namespace.get("__name__") == projection.CLASS_MODULE_NAMES[0]:
                sys.modules[projection.CLASS_MODULE_NAMES[0]] = sentinel

        with (
            patch.dict(sys.modules),
            patch.object(projection, "_compile", side_effect=replace_policy),
        ):
            self.reject()
            self.assertIs(sys.modules[projection.CLASS_MODULE_NAMES[0]], sentinel)
            self.assertNotIn(projection.CLASS_MODULE_NAMES[1], sys.modules)

    def test_original_test_namespace_is_unregistered_and_preexisting_entries_untouched(self):
        sentinel = types.ModuleType(projection.MODULE_NAME)
        with patch.dict(sys.modules, {projection.MODULE_NAME: sentinel}):
            module, _ = self.build()
            self.assertIs(sys.modules[projection.MODULE_NAME], sentinel)
            self.assertIsNot(module, sentinel)
        self.assertEqual({name for name in vars(module) if name.startswith("test_")},
                         set(projection.TEST_NAMES))

    def test_registry_difference_and_metadata_changes_are_rejected(self):
        original = json.loads(self.payloads["successor_registry"])
        cases = []
        value = copy.deepcopy(original)
        value["sources"][0]["adapter"] = "synthetic unauthorized difference"
        cases.append(value)
        value = copy.deepcopy(original)
        value["observed_at"] = "changed"
        cases.append(value)
        value = copy.deepcopy(original)
        value["sources"][1]["production_eligible"] = True
        cases.append(value)
        for value in cases:
            with self.subTest(source_count=len(value["sources"])):
                self.payloads["successor_registry"] = encoded(value)
                self.repin("successor_registry")
                module, _ = self.build()
                with explicit_synthetic_doubles(module), self.assertRaises(AssertionError):
                    getattr(module, projection.TEST_NAMES[0])()

    def test_historical_registry_parser_keeps_original_provenance_and_url_rules(self):
        module, _ = self.build()
        promotion = module.build_source_promotion_receipt.__globals__
        registry = promotion["load_sft_source_registry"].__globals__
        policy = promotion["load_sft_policy"].__globals__
        self.assertNotIn("_sha256", policy)
        value = {"synthetic": [1, False, "বাংলা"]}
        self.assertEqual(registry["_sha256"](value), hashlib.sha256(canonical(value)).hexdigest())
        original = json.loads(self.payloads["predecessor_registry"])["sources"][2]
        for field, changed, message in (
            ("repository_url", "https://github.com/synthetic/source", "source repository URL"),
            ("revision", "unbound", "source revision"),
            ("license_id", "unapproved", "source license"),
            ("source_id", "bad id", "invalid source_id"),
        ):
            with self.subTest(field=field):
                source = copy.deepcopy(original)
                source[field] = changed
                with self.assertRaisesRegex(ValueError, message):
                    registry["SFTSource"].from_mapping(
                        source, allowed_licenses=frozenset({"apache-2.0", "lumi-owned"}),
                        allowed_languages=frozenset({"bn", "en"}),
                        allowed_splits=frozenset({"train", "validation"}),
                        allowed_tasks=frozenset({"coding"}),
                    )

    def test_promotion_receipt_changes_and_seal_tampering_are_rejected(self):
        original = json.loads(self.payloads["promotion_receipt"])
        for field, value in (("production_eligible", True), ("production_dataset_promotions", 1),
                             ("candidate_source_registry_promotions", 2),
                             ("active_production_data_contract_updated", True),
                             ("raw_training_text_retained", True), ("receipt_sha256", "0" * 64)):
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                changed[field] = value
                self.payloads["promotion_receipt"] = encoded(changed)
                self.repin("promotion_receipt")
                module, _ = self.build()
                with explicit_synthetic_doubles(module), self.assertRaises(AssertionError):
                    getattr(module, projection.TEST_NAMES[1])()
                with self.assertRaises(ValueError):
                    module.verify_source_promotion_receipt(changed)

    def test_validly_resealed_promotion_receipt_still_rejects_boundary_changes(self):
        module, _ = self.build()
        promotion = module.build_source_promotion_receipt.__globals__
        original = json.loads(self.payloads["promotion_receipt"])
        for field, value in (
            ("production_eligible", True), ("production_dataset_promotions", 1),
            ("candidate_source_registry_promotions", 2),
            ("active_production_data_contract_updated", True), ("raw_training_text_retained", True),
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                changed[field] = value
                changed["receipt_sha256"] = promotion["sha256_value"]({
                    key: item for key, item in changed.items() if key != "receipt_sha256"
                })
                with self.assertRaisesRegex(
                    ValueError, "^Aya source-promotion receipt boundary changed$",
                ):
                    module.verify_source_promotion_receipt(changed)

    def test_policy_drift_is_rejected_without_loading_extra_policy_paths(self):
        policy = json.loads(self.payloads["data_contract"])
        policy["chat_training"]["context_window"] = 8192
        self.payloads["data_contract"] = encoded(policy)
        self.repin("data_contract")
        module, snapshot = self.build()
        with self.assertRaisesRegex(ValueError, "^SFT context window must remain 4096$"):
            getattr(module, projection.TEST_NAMES[0])()
        self.assertEqual(snapshot.summary()["virtual_opens"], 10)

    def test_unused_imports_and_publishers_are_excluded(self):
        poison = (b"\nimport nonexistent_synthetic_projection_dependency\n"
                  b"raise RuntimeError('unselected source poison')\n"
                  b"def unused_poison(value=evil()):\n    raise RuntimeError('poison')\n")
        for role in (*SOURCE_ROLES, "source_registry_implementation"):
            self.payloads[role] += poison
            self.repin(role)
        module, _ = self.build()
        with explicit_synthetic_doubles(module):
            # Rebinding source code changes the plan/expected receipt; the first original node
            # is sufficient to prove that no appended top-level statement or import executes.
            getattr(module, projection.TEST_NAMES[0])()
        promotion = module.build_source_promotion_receipt.__globals__
        for namespace in (module.__dict__, promotion, promotion["load_sft_policy"].__globals__,
                          promotion["load_sft_source_registry"].__globals__):
            for name in ("unused_poison", "main", "normalize_for_matching", "filter_sft_records",
                         "test_exact_reviewed_aya_prerequisites_pass"):
                self.assertNotIn(name, namespace)

    def test_snapshot_excludes_helper_sources_and_write_or_real_filesystem_escape(self):
        module, snapshot = self.build()
        for role in ("policy_source", "hash_helper_source", "test_source"):
            path = module.ROOT / projection.SOURCE_PATHS[role]
            self.assertFalse(path.exists())
            with self.assertRaisesRegex(ValueError, "^snapshot_lookup_rejected$"):
                path.read_bytes()
        with self.assertRaisesRegex(ValueError, "^snapshot_mode_rejected$"):
            module.PLAN.open("w")
        with self.assertRaisesRegex(ValueError, "^snapshot_filesystem_escape_rejected$"):
            module.PLAN.__fspath__()
        self.assertEqual(snapshot.summary()["virtual_opens"], 0)

    def test_invalid_roots_and_source_errors_have_fixed_raw_free_construction_errors(self):
        for root in ("/", "//synthetic", "relative", "/synthetic/..", "/bad\\path", "/~bad"):
            with self.subTest(root=root), self.assertRaisesRegex(ValueError, f"^{ERROR}$"):
                projection.build_test_module(self.payloads, self.bindings, root)
        self.payloads["policy_source"] += b"\nthis is invalid synthetic private diagnostic !\n"
        self.repin("policy_source")
        with self.assertRaises(ValueError) as raised:
            self.build()
        self.assertEqual(raised.exception.args, (ERROR,))
        self.assertTrue(raised.exception.__suppress_context__)


if __name__ == "__main__":
    unittest.main()
