"""Synthetic-only prerequisite projection tests; no private fixture access.

The unchanged verifier hardcodes two approved historical seals. Invented JSON
cannot reproduce those cryptographic preimages. Positive synthetic control-flow
tests therefore patch only module.sha256_value after projection construction,
with an explicit schema-based test double. Original ASTs are never rewritten;
separate tests exercise authentic hash helpers and fail-closed seal rejection.
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
from pathlib import Path
from unittest.mock import patch

from lumi import posttraining_aya_prerequisite_snapshot_v1 as projection

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_ROOT = "/synthetic/workspace/repo"
ERROR = "invalid Aya prerequisite source projection"
SOURCE_ROLES = ("promotion_implementation", "hash_helper_source", "test_source")
APPROVED_SEALS = {
    "lumi.sft-aya-bengali-materialization-report.v0.1": (
        "a2f08b1594a2b6fd9def1f43a48e072cbe0f107f4bc7ea43b391d6066ddc6ade"
    ),
    "lumi.sft-aya-bengali-owner-review-receipt.v0.1": (
        "e23851af5718d10acdd120ca1e0c200672fb5c794ad503ed4d1f7f1650cc685f"
    ),
}


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
    payloads["plan"] = canonical(plan)
    pin(payloads, bindings, "plan")


def synthetic_inputs():
    """Read three source files only; invent all plan, registry and evidence bytes."""
    payloads = {role: ("synthetic " + role).encode() for role in projection.INPUT_ROLES}
    for role in SOURCE_ROLES:
        payloads[role] = (ROOT / projection.SOURCE_PATHS[role]).read_bytes()
    namespace = {"re": re}
    projection._compile(
        projection._select(
            payloads["promotion_implementation"], projection.AYA_CONSTANTS, frozenset(),
        ),
        "<synthetic-source-constants>", namespace,
    )
    rows = [
        {
            "record_id": f"synthetic-aya-{index:04d}",
            "schema_version": "lumi.sft-aya-bengali-attribution.v0.1",
            "source_id": "coherelabs_aya_dataset", "dataset_id": "CohereLabs/aya_dataset",
            "revision": namespace["EXPECTED_SOURCE_AFTER"]["revision"],
            "license_id": "apache-2.0",
        }
        for index in range(1478)
    ]
    payloads["attribution_manifest"] = b"\n".join(canonical(row) for row in rows) + b"\n"
    materialization_schema, owner_schema = tuple(APPROVED_SEALS)
    payloads["materialization_report"] = canonical({
        "schema_version": materialization_schema,
        "candidate": {"materialized": True, "records": 1478, "supervised_tokens": 577453},
        "raw_training_text_in_report": False, "production": {"production_eligible": False},
        "report_sha256": APPROVED_SEALS[materialization_schema],
    })
    payloads["owner_review_receipt"] = canonical({
        "schema_version": owner_schema,
        "status": "owner_quality_sample_passed_source_promotion_still_separate",
        "sample_records": 32, "accepted_records": 32, "rejected_records": 0,
        "all_sampled_rows_accepted": True, "production_eligible": False,
        "production_promotions": 0, "raw_training_text_retained": False,
        "receipt_sha256": APPROVED_SEALS[owner_schema],
    })
    # These inputs must be hashed, never parsed or executed, by this exact node.
    for role in (
        "source_registry_implementation", "promotion_runner", "data_contract",
        "predecessor_registry",
    ):
        payloads[role] = b"\xff synthetic hash-only poison: " + role.encode()
    bindings = {}
    for role in projection.INPUT_ROLES - {"plan"}:
        pin(payloads, bindings, role)
    payloads["plan"] = canonical({
        "schema_version": namespace["PLAN_SCHEMA_VERSION"],
        "status": "authorized_candidate_registry_successor_promotion",
        "observed_at": "2026-08-06",
        **{
            field: copy.deepcopy(namespace["EXPECTED_" + field.upper()])
            for field in (
                "authorization", "limits", "output", "curated_metrics", "required_tasks",
                "source_before", "source_after",
            )
        },
    })
    repin_plan(payloads, bindings)
    return payloads, bindings


def synthetic_seal_test_double(value):
    """Test-only stand-in for unavailable historical private JSON preimages."""
    return APPROVED_SEALS[value["schema_version"]]


class AyaPrerequisiteSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.payloads, self.bindings = synthetic_inputs()

    def build(self):
        return projection.build_test_module(self.payloads, self.bindings, SYNTHETIC_ROOT)

    def reject(self):
        with self.assertRaisesRegex(ValueError, f"^{ERROR}$"):
            self.build()

    def replace_private_json(self, role, value):
        self.payloads[role] = canonical(value)
        pin(self.payloads, self.bindings, role)
        repin_plan(self.payloads, self.bindings)

    def replace_attribution(self, rows):
        self.payloads["attribution_manifest"] = b"\n".join(canonical(row) for row in rows) + b"\n"
        pin(self.payloads, self.bindings, "attribution_manifest")
        repin_plan(self.payloads, self.bindings)

    def run_with_explicit_seal_test_double(self):
        module, snapshot = self.build()
        with patch.object(module, "sha256_value", side_effect=synthetic_seal_test_double) as seal:
            self.assertIsNone(getattr(module, projection.TEST_NAME)())
            self.assertEqual(seal.call_count, 2)
        return module, snapshot

    def test_synthetic_control_flow_has_twelve_opens_with_explicit_seal_double(self):
        module, snapshot = self.build()
        self.assertEqual(snapshot.summary()["virtual_opens"], 0)
        original_sha256 = module.sha256_value
        with patch.object(module, "sha256_value", side_effect=synthetic_seal_test_double) as seal:
            self.assertIsNone(getattr(module, projection.TEST_NAME)())
            self.assertEqual(seal.call_count, 2)
        self.assertIs(module.sha256_value, original_sha256)
        expected_bytes = (
            sum(self.bindings[role]["bytes"] for role in projection.PLAN_BINDING_ROLES | {"plan"})
            + sum(self.bindings[role]["bytes"] for role in (
                "attribution_manifest", "materialization_report", "owner_review_receipt",
            ))
        )
        self.assertEqual(snapshot.summary(), {
            "bound_blob_count": 9, "virtual_opens": 12,
            "logical_read_bytes_debited": expected_bytes, "maximum_virtual_opens": 12,
            "maximum_logical_read_bytes": expected_bytes, "maximum_gzip_bytes": 1,
            "filesystem_access_performed": False, "execution_permitted": False,
        })

    def test_original_hash_helpers_work_without_any_seal_test_double(self):
        module, snapshot = self.build()
        value = {"synthetic": "বাংলা", "number": 17, "flag": False}
        expected_bytes = canonical(value)
        expected_hash = hashlib.sha256(expected_bytes).hexdigest()
        self.assertEqual(module.canonical_json_bytes(value), expected_bytes)
        self.assertEqual(module.sha256_value(value), expected_hash)
        self.assertEqual(
            module._verify_internal_seal({**value, "seal": expected_hash}, "seal"), expected_hash,
        )
        self.assertEqual(module.file_sha256(module.PLAN), self.bindings["plan"]["sha256"])
        self.assertEqual(snapshot.summary()["virtual_opens"], 1)

    def test_original_verifier_rejects_synthetic_seals_without_test_double(self):
        module, snapshot = self.build()
        with self.assertRaisesRegex(ValueError, "^report_sha256 changed$"):
            getattr(module, projection.TEST_NAME)()
        self.assertEqual(snapshot.summary()["virtual_opens"], 12)

    def test_all_original_constants_function_asts_and_locations_are_preserved(self):
        module, _ = self.build()
        selections = {
            "promotion_implementation": (projection.AYA_CONSTANTS, projection.AYA_FUNCTIONS),
            "hash_helper_source": (frozenset(), {"canonical_json_bytes", "file_sha256"}),
            "test_source": (frozenset(), {"_load", "_prerequisites", projection.TEST_NAME}),
        }
        for role, (constants, functions) in selections.items():
            source = self.payloads[role]
            original = ast.parse(source)
            selected = projection._select(source, frozenset(constants), frozenset(functions))
            originals = {
                node.name if isinstance(node, ast.FunctionDef) else node.targets[0].id: node
                for node in original.body
                if (
                    isinstance(node, ast.FunctionDef) and node.name in functions
                    or isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id in constants
                )
            }
            for node in selected:
                name = node.name if isinstance(node, ast.FunctionDef) else node.targets[0].id
                with self.subTest(role=role, name=name):
                    self.assertEqual(
                        ast.dump(node, include_attributes=True),
                        ast.dump(originals[name], include_attributes=True),
                    )
                    if name in functions:
                        function = getattr(module, name)
                        self.assertEqual(function.__code__.co_firstlineno, originals[name].lineno)
                        self.assertEqual(
                            function.__code__.co_filename,
                            SYNTHETIC_ROOT + "/" + projection.SOURCE_PATHS[role],
                        )

    def test_construction_and_execution_have_no_real_file_operations(self):
        with (
            patch("builtins.open", side_effect=AssertionError("real open")),
            patch("io.open", side_effect=AssertionError("real open")),
            patch("os.open", side_effect=AssertionError("real open")),
            patch("os.stat", side_effect=AssertionError("real stat")),
            patch("os.lstat", side_effect=AssertionError("real stat")),
        ):
            self.run_with_explicit_seal_test_double()

    def test_all_inputs_authenticate_before_ast_or_json_parsing(self):
        for role in projection.INPUT_ROLES:
            with self.subTest(role=role):
                original = self.payloads[role]
                self.payloads[role] = original + b" "
                with (
                    patch.object(ast, "parse", side_effect=AssertionError("early AST")) as parse,
                    patch.object(projection.json, "loads",
                                 side_effect=AssertionError("early JSON")) as loads,
                ):
                    self.reject()
                    parse.assert_not_called()
                    loads.assert_not_called()
                self.payloads[role] = original

    def test_roles_paths_types_digests_and_sizes_are_exact(self):
        original = copy.deepcopy(self.bindings)
        for role in projection.INPUT_ROLES:
            for field, value in (
                ("path", "unexpected/path"), ("bytes", True), ("bytes", 0),
                ("bytes", original[role]["bytes"] + 1), ("sha256", "0" * 64),
                ("sha256", "A" * 64), ("physical_read_path", "build/lib/synthetic.py"),
            ):
                with self.subTest(role=role, field=field, value=value):
                    self.bindings = copy.deepcopy(original)
                    self.bindings[role][field] = value
                    self.reject()
        self.bindings = original
        for target in ("payloads", "bindings"):
            for extra in (False, True):
                with self.subTest(target=target, extra=extra):
                    value = getattr(self, target)
                    before = dict(value)
                    if extra:
                        value["unexpected"] = value["plan"]
                    else:
                        del value["plan"]
                    self.reject()
                    setattr(self, target, before)

    def test_logical_binding_never_changes_to_a_physical_copy_path(self):
        self.bindings["source_registry_implementation"]["path"] = (
            "build/lib/lumi/posttraining_sources.py"
        )
        repin_plan(self.payloads, self.bindings)
        self.reject()

    def test_per_role_and_total_limits_fail_closed(self):
        for role, maximum in (
            ("attribution_manifest", projection.MAXIMUM_METADATA_BYTES),
            ("source_registry_implementation", projection.MAXIMUM_SOURCE_BYTES),
        ):
            with self.subTest(role=role):
                original_payload, original_binding = self.payloads[role], self.bindings[role]
                self.payloads[role] = b"x" * (maximum + 1)
                pin(self.payloads, self.bindings, role)
                self.reject()
                self.payloads[role], self.bindings[role] = original_payload, original_binding
        with patch.object(projection, "MAXIMUM_TOTAL_BYTES", 1):
            self.reject()

    def test_source_tampering_missing_and_unsafe_declarations_are_rejected(self):
        original = self.payloads["hash_helper_source"]
        self.payloads["hash_helper_source"] = b"X" + original[1:]
        self.reject()
        for source in (
            original.replace(b"def file_sha256(", b"def renamed_hash("),
            original + b"\ndef file_sha256(path):\n    return 'duplicate'\n",
            original.replace(b"def file_sha256(path: str | Path)",
                             b"def file_sha256(path: str | Path = 1 / 0)"),
        ):
            with self.subTest(source_length=len(source)):
                self.payloads["hash_helper_source"] = source
                pin(self.payloads, self.bindings, "hash_helper_source")
                self.reject()
        self.payloads["hash_helper_source"] = original
        pin(self.payloads, self.bindings, "hash_helper_source")
        self.payloads["promotion_implementation"] += b"\nEXPECTED_LIMITS = external_state()\n"
        pin(self.payloads, self.bindings, "promotion_implementation")
        repin_plan(self.payloads, self.bindings)
        self.reject()

    def test_plan_duplicates_nonfinite_numbers_and_binding_mismatches_rejected(self):
        original = self.payloads["plan"]
        plan = json.loads(original)
        cases = [
            b'{"bindings":{},' + original[1:],
            original[:-1] + b',"synthetic_nonfinite":NaN}',
            original.replace(b'"accepted_records":1478',
                             b'"accepted_records":1478,"accepted_records":1478'),
            b'[]',
        ]
        for role in projection.PLAN_BINDING_ROLES:
            for field, value in (("path", "unexpected/path"), ("sha256", "0" * 64)):
                changed = copy.deepcopy(plan)
                changed["bindings"][role][field] = value
                cases.append(canonical(changed))
        for payload in cases:
            with self.subTest(payload_length=len(payload)):
                self.payloads["plan"] = payload
                pin(self.payloads, self.bindings, "plan")
                self.reject()

    def test_private_prerequisites_are_not_parsed_during_construction(self):
        for role in ("attribution_manifest", "materialization_report", "owner_review_receipt"):
            self.payloads[role] = b"unparsable synthetic private bytes\xff"
            pin(self.payloads, self.bindings, role)
        repin_plan(self.payloads, self.bindings)
        _, snapshot = self.build()
        self.assertEqual(snapshot.summary()["virtual_opens"], 0)

    def test_original_loader_rejects_authorization_drift_without_granting_authority(self):
        module, _ = self.build()
        self.assertIs(module.EXPECTED_AUTHORIZATION["candidate_source_registry_promotion_authorized"],
                      True)
        plan = json.loads(self.payloads["plan"])
        plan["authorization"]["production_sft_authorized"] = True
        self.payloads["plan"] = canonical(plan)
        pin(self.payloads, self.bindings, "plan")
        module, snapshot = self.build()
        with self.assertRaisesRegex(ValueError, "^Aya source-promotion authorization changed$"):
            getattr(module, projection.TEST_NAME)()
        self.assertEqual(snapshot.summary()["virtual_opens"], 1)
        self.assertIs(snapshot.summary()["execution_permitted"], False)

    def test_original_verifier_rejects_wrong_attribution_count_identity_and_provenance(self):
        original_rows = [
            json.loads(line) for line in self.payloads["attribution_manifest"].splitlines()
        ]
        cases = [(original_rows[:-1], "Aya attribution manifest count changed")]
        duplicate = copy.deepcopy(original_rows)
        duplicate[1]["record_id"] = duplicate[0]["record_id"]
        cases.append((duplicate, "Aya attribution identity changed"))
        for field in ("record_id", "schema_version", "source_id", "dataset_id", "revision",
                      "license_id"):
            rows = copy.deepcopy(original_rows)
            rows[0][field] = "" if field == "record_id" else "synthetic mismatch"
            message = "Aya attribution identity changed" if field == "record_id" else (
                "Aya attribution provenance changed"
            )
            cases.append((rows, message))
        for rows, message in cases:
            with self.subTest(message=message, first_row=rows[0]["record_id"]):
                self.replace_attribution(rows)
                module, snapshot = self.build()
                with (
                    patch.object(module, "sha256_value", side_effect=synthetic_seal_test_double),
                    self.assertRaisesRegex(ValueError, "^" + message + "$"),
                ):
                    getattr(module, projection.TEST_NAME)()
                self.assertEqual(snapshot.summary()["virtual_opens"], 12)

    def test_original_verifier_rejects_materialization_metrics_and_production_drift(self):
        original = json.loads(self.payloads["materialization_report"])
        cases = []
        for key, value in (("materialized", False), ("records", 1477), ("supervised_tokens", 1)):
            report = copy.deepcopy(original)
            report["candidate"][key] = value
            cases.append((report, "Aya materialized metrics changed"))
        report = copy.deepcopy(original)
        report["production"]["production_eligible"] = True
        cases.append((report, "Aya materialization production boundary changed"))
        report = copy.deepcopy(original)
        report["raw_training_text_in_report"] = True
        cases.append((report, "Aya materialization production boundary changed"))
        for report, message in cases:
            with self.subTest(message=message):
                self.replace_private_json("materialization_report", report)
                module, _ = self.build()
                with (
                    patch.object(module, "sha256_value", side_effect=synthetic_seal_test_double),
                    self.assertRaisesRegex(ValueError, "^" + message + "$"),
                ):
                    getattr(module, projection.TEST_NAME)()

    def test_original_verifier_rejects_owner_gate_drift(self):
        original = json.loads(self.payloads["owner_review_receipt"])
        for field, value in (
            ("status", "synthetic rejection"), ("sample_records", 31), ("accepted_records", 31),
            ("rejected_records", 1), ("all_sampled_rows_accepted", False),
            ("production_eligible", True), ("production_promotions", 1),
            ("raw_training_text_retained", True),
        ):
            with self.subTest(field=field):
                owner = copy.deepcopy(original)
                owner[field] = value
                self.replace_private_json("owner_review_receipt", owner)
                module, _ = self.build()
                with (
                    patch.object(module, "sha256_value", side_effect=synthetic_seal_test_double),
                    self.assertRaisesRegex(ValueError, "^Aya owner-review gate did not pass$"),
                ):
                    getattr(module, projection.TEST_NAME)()

    def test_original_verifier_rejects_bad_seal_fields_even_with_explicit_double(self):
        original_payloads = dict(self.payloads)
        original_bindings = copy.deepcopy(self.bindings)
        for role, field in (("materialization_report", "report_sha256"),
                            ("owner_review_receipt", "receipt_sha256")):
            with self.subTest(role=role):
                self.payloads = dict(original_payloads)
                self.bindings = copy.deepcopy(original_bindings)
                value = json.loads(self.payloads[role])
                value[field] = "0" * 64
                self.replace_private_json(role, value)
                module, _ = self.build()
                with (
                    patch.object(module, "sha256_value", side_effect=synthetic_seal_test_double),
                    self.assertRaisesRegex(ValueError, "^" + field + " changed$"),
                ):
                    getattr(module, projection.TEST_NAME)()

    def test_unused_imports_and_hash_only_registry_runner_never_execute(self):
        poison = (
            b"\nimport nonexistent_synthetic_prerequisite_dependency\n"
            b"raise RuntimeError('synthetic top-level poison')\n"
            b"def unused_poison(value=1 / 0):\n    raise RuntimeError('poison')\n"
        )
        for role in SOURCE_ROLES:
            self.payloads[role] += poison
            pin(self.payloads, self.bindings, role)
        repin_plan(self.payloads, self.bindings)
        module, _ = self.run_with_explicit_seal_test_double()
        for name in (
            "unused_poison", "build_successor_registry", "build_source_promotion_receipt",
            "verify_source_promotion_receipt", "load_policy_and_predecessor",
            "load_sft_source_registry", "SFTSource", "SFTPolicy", "main", "PREDECESSOR",
            "SUCCESSOR", "RECEIPT", "json_file_bytes", "copy",
        ):
            self.assertNotIn(name, vars(module))
        self.assertEqual(
            {name for name in vars(module) if name.startswith("test_")}, {projection.TEST_NAME},
        )

    def test_existing_module_namespace_is_untouched(self):
        sentinel = types.ModuleType(projection.MODULE_NAME)
        child = types.ModuleType(projection.MODULE_NAME + ".sentinel")
        entries = {projection.MODULE_NAME: sentinel, child.__name__: child}
        with patch.dict(sys.modules, entries):
            module, _ = self.run_with_explicit_seal_test_double()
            self.assertIsNot(module, sentinel)
            for name, expected in entries.items():
                self.assertIs(sys.modules[name], expected)

    def test_snapshot_excludes_extra_sources_writes_and_filesystem_escape(self):
        module, snapshot = self.build()
        for role in projection.INPUT_ROLES - (projection.PLAN_BINDING_ROLES | {"plan"}):
            with self.subTest(role=role):
                path = module.ROOT / projection.SOURCE_PATHS[role]
                self.assertFalse(path.exists())
                with self.assertRaisesRegex(ValueError, "^snapshot_lookup_rejected$"):
                    path.read_bytes()
        with self.assertRaisesRegex(ValueError, "^snapshot_mode_rejected$"):
            module.PLAN.open("w")
        with self.assertRaisesRegex(ValueError, "^snapshot_filesystem_escape_rejected$"):
            module.PLAN.__fspath__()
        self.assertEqual(snapshot.summary()["virtual_opens"], 0)

    def test_invalid_roots_fail_with_fixed_construction_error(self):
        for root in ("/", "//synthetic", "relative", "/synthetic/..", "/bad\\path", "/~bad"):
            with self.subTest(root=root), self.assertRaisesRegex(ValueError, f"^{ERROR}$"):
                projection.build_test_module(self.payloads, self.bindings, root)


if __name__ == "__main__":
    unittest.main()
