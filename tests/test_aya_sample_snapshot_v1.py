"""Synthetic-only Aya sample projection checks; no real private input access."""

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

from lumi import posttraining_aya_sample_snapshot_v1 as projection

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_ROOT = "/synthetic/workspace/repo"
ERROR = "invalid Aya sample source projection"
SOURCE_ROLES = (
    "review_implementation", "file_hash_source", "sample_helper_source", "test_source",
)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def seal(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def pin(payloads, bindings, role):
    bindings[role] = {
        "path": projection.SOURCE_PATHS[role], "bytes": len(payloads[role]),
        "sha256": hashlib.sha256(payloads[role]).hexdigest(),
    }


def replace_fixture_constant(source, name, value):
    """Change only one synthetic constant, retaining all original function text."""
    node = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name
    )
    lines = source.decode().splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = [
        f"{name} = {value!r}\n", *("\n" for _ in range(node.end_lineno - node.lineno)),
    ]
    return "".join(lines).encode()


def repin_plan(payloads, bindings):
    plan = json.loads(payloads["plan"])
    plan["bindings"] = {
        role: {field: bindings[role][field] for field in ("path", "sha256")}
        for role in projection.PLAN_BINDING_ROLES
    }
    payloads["plan"] = canonical(plan)
    pin(payloads, bindings, "plan")


def synthetic_inputs():
    """Read four tracked source files; all plan/private bytes are invented here.

    Only EXPECTED_CANDIDATE is adapted in the in-memory source fixture. The
    original functions and exact test body, including its 32-row assertion,
    remain unchanged. This fixture is not a verification of real Aya data.
    """
    payloads = {role: ("synthetic " + role).encode() for role in projection.INPUT_ROLES}
    for role in SOURCE_ROLES:
        payloads[role] = (ROOT / projection.SOURCE_PATHS[role]).read_bytes()
    originals = {role: payloads[role] for role in SOURCE_ROLES}
    namespace = {"re": re}
    projection._compile(
        projection._select(
            payloads["review_implementation"], projection.AYA_CONSTANTS, frozenset(),
        ),
        "<synthetic-source-constants>", namespace,
    )
    rows = [
        {
            "record_id": f"synthetic-{index:03d}", "split": "synthetic",
            "task_category": "synthetic", "messages": [
                {"role": "user", "content": "synthetic question"},
                {"role": "assistant", "content": "synthetic answer"},
            ],
        }
        for index in range(40)
    ]
    payloads["accepted_records"] = b"\n".join(canonical(row) for row in rows) + b"\n"
    selected_ids = sorted(
        sorted(
            (row["record_id"] for row in rows),
            key=lambda record_id: hashlib.sha256(record_id.encode()).hexdigest(),
        )[:32]
    )
    report = {
        "candidate": {"materialized": True, "records": len(rows), "supervised_tokens": 80},
        "production": {"production_eligible": False},
    }
    report["report_sha256"] = seal(report)
    payloads["materialization_report"] = canonical(report)
    # These are intentionally unparsable and must be hash-only inputs.
    payloads["owner_review_packet"] = b"\xff<script>raise synthetic packet poison</script>"
    payloads["attribution_manifest"] = b"not synthetic JSON\xff"
    payloads["receipt_finalizer"] = b"raise RuntimeError('synthetic finalizer poison')\n"
    expected = copy.deepcopy(namespace["EXPECTED_CANDIDATE"])
    expected.update({
        "accepted_records": len(rows), "supervised_tokens": 80,
        "sample_record_ids_sha256": seal(selected_ids),
        "materialization_report_internal_sha256": report["report_sha256"],
    })
    for field, role in (
        ("candidate_file_sha256", "accepted_records"),
        ("attribution_file_sha256", "attribution_manifest"),
        ("owner_review_packet_file_sha256", "owner_review_packet"),
        ("materialization_report_file_sha256", "materialization_report"),
    ):
        expected[field] = hashlib.sha256(payloads[role]).hexdigest()
    payloads["review_implementation"] = replace_fixture_constant(
        payloads["review_implementation"], "EXPECTED_CANDIDATE", expected,
    )
    bindings = {}
    for role in projection.INPUT_ROLES - {"plan"}:
        pin(payloads, bindings, role)
    payloads["plan"] = canonical({
        "schema_version": namespace["PLAN_SCHEMA_VERSION"],
        "status": "human_owner_review_ready_no_promotion", "observed_at": "2026-08-06",
        "authorization": namespace["EXPECTED_AUTHORIZATION"], "candidate": expected,
        "limits": namespace["EXPECTED_LIMITS"], "output": namespace["EXPECTED_OUTPUT"],
    })
    repin_plan(payloads, bindings)
    return payloads, bindings, originals


class AyaSampleSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.payloads, self.bindings, self.originals = synthetic_inputs()

    def build(self):
        return projection.build_test_module(self.payloads, self.bindings, SYNTHETIC_ROOT)

    def reject(self):
        with self.assertRaisesRegex(ValueError, f"^{ERROR}$"):
            self.build()

    def run_original(self):
        module, snapshot = self.build()
        self.assertIsNone(getattr(module, projection.TEST_NAME)())
        return module, snapshot

    def test_original_body_passes_with_nine_virtual_opens_and_seven_blobs(self):
        module, snapshot = self.build()
        self.assertEqual(snapshot.summary()["virtual_opens"], 0)
        self.assertIsNone(getattr(module, projection.TEST_NAME)())
        expected_bytes = (
            sum(self.bindings[role]["bytes"] for role in projection.PLAN_BINDING_ROLES | {"plan"})
            + self.bindings["accepted_records"]["bytes"]
            + self.bindings["materialization_report"]["bytes"]
        )
        self.assertEqual(snapshot.summary(), {
            "bound_blob_count": 7, "virtual_opens": 9,
            "logical_read_bytes_debited": expected_bytes, "maximum_virtual_opens": 9,
            "maximum_logical_read_bytes": expected_bytes, "maximum_gzip_bytes": 1,
            "filesystem_access_performed": False, "execution_permitted": False,
        })

    def test_original_function_asts_and_locations_are_preserved(self):
        module, _ = self.build()
        selections = {
            "review_implementation": projection.AYA_FUNCTIONS,
            "file_hash_source": {"file_sha256"},
            "sample_helper_source": {
                "canonical_json_bytes", "sha256_value", "deterministic_owner_sample",
            },
            "test_source": {projection.TEST_NAME},
        }
        for role, names in selections.items():
            for name in names:
                with self.subTest(role=role, name=name):
                    original = next(
                        node for node in ast.parse(self.originals[role]).body
                        if isinstance(node, ast.FunctionDef) and node.name == name
                    )
                    selected = projection._select(
                        self.payloads[role], frozenset(), frozenset({name}),
                    )[0]
                    self.assertEqual(
                        ast.dump(original, include_attributes=True),
                        ast.dump(selected, include_attributes=True),
                    )
                    function = getattr(module, name)
                    self.assertEqual(function.__code__.co_firstlineno, original.lineno)
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
            self.run_original()

    def test_all_inputs_authenticate_before_ast_or_json_parsing(self):
        for role in projection.INPUT_ROLES:
            with self.subTest(role=role):
                original = self.payloads[role]
                self.payloads[role] = original + b" "
                with (
                    patch.object(ast, "parse",
                                 side_effect=AssertionError("early AST parse")) as parse,
                    patch.object(projection.json, "loads",
                                 side_effect=AssertionError("early JSON parse")) as loads,
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
                ("sha256", "A" * 64), ("unexpected", "field"),
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

    def test_per_role_and_total_limits_fail_before_parsing(self):
        for role, maximum in (
            ("accepted_records", projection.MAXIMUM_CANDIDATE_BYTES),
            ("owner_review_packet", projection.MAXIMUM_PRIVATE_BYTES),
            ("test_source", projection.MAXIMUM_SOURCE_BYTES),
        ):
            with self.subTest(role=role):
                original_payload, original_binding = self.payloads[role], self.bindings[role]
                self.payloads[role] = b"x" * (maximum + 1)
                pin(self.payloads, self.bindings, role)
                self.reject()
                self.payloads[role], self.bindings[role] = original_payload, original_binding
        with patch.object(projection, "MAXIMUM_TOTAL_BYTES", 1):
            self.reject()

    def test_same_length_source_tampering_and_unsafe_declarations_are_rejected(self):
        original = self.payloads["file_hash_source"]
        self.payloads["file_hash_source"] = b"X" + original[1:]
        self.reject()
        self.payloads["file_hash_source"] = original
        original_bindings = copy.deepcopy(self.bindings)
        for source in (
            original.replace(b"def file_sha256(", b"def renamed_hash("),
            original + b"\ndef file_sha256(path):\n    return 'duplicate'\n",
            original.replace(b"def file_sha256(path: str | Path)",
                             b"def file_sha256(path: str | Path = 1 / 0)"),
        ):
            with self.subTest(source_length=len(source)):
                self.payloads["file_hash_source"] = source
                pin(self.payloads, self.bindings, "file_hash_source")
                self.reject()
        self.payloads["file_hash_source"] = original
        self.bindings = original_bindings
        self.payloads["review_implementation"] += b"\nEXPECTED_LIMITS = external_state()\n"
        pin(self.payloads, self.bindings, "review_implementation")
        repin_plan(self.payloads, self.bindings)
        self.reject()

    def test_plan_duplicates_nonfinite_numbers_and_binding_mismatches_rejected(self):
        original = self.payloads["plan"]
        plan = json.loads(original)
        cases = [
            b'{"bindings":{},' + original[1:],
            original[:-1] + b',"synthetic_nonfinite":NaN}',
            original.replace(b'"records":32', b'"records":32,"records":32'),
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

    def test_report_and_candidate_are_not_parsed_during_construction(self):
        for role in ("materialization_report", "accepted_records"):
            with self.subTest(role=role):
                self.payloads[role] = b"unparsable synthetic private bytes\xff"
                pin(self.payloads, self.bindings, role)
        repin_plan(self.payloads, self.bindings)
        _, snapshot = self.build()
        self.assertEqual(snapshot.summary()["virtual_opens"], 0)

    def test_wrong_sample_hash_fails_in_unchanged_original_loader(self):
        plan = json.loads(self.payloads["plan"])
        expected = plan["candidate"]
        expected["sample_record_ids_sha256"] = "0" * 64
        self.payloads["review_implementation"] = replace_fixture_constant(
            self.payloads["review_implementation"], "EXPECTED_CANDIDATE", expected,
        )
        pin(self.payloads, self.bindings, "review_implementation")
        self.payloads["plan"] = canonical(plan)
        repin_plan(self.payloads, self.bindings)
        module, snapshot = self.build()
        with self.assertRaisesRegex(ValueError, "^Aya owner-review sample hash changed$"):
            getattr(module, projection.TEST_NAME)()
        self.assertEqual(snapshot.summary()["virtual_opens"], 9)

    def test_unused_imports_functions_and_hash_only_inputs_never_execute(self):
        poison = (
            b"\nimport nonexistent_synthetic_aya_dependency\n"
            b"raise RuntimeError('synthetic top-level poison')\n"
            b"def unused_poison(value=1 / 0):\n    raise RuntimeError('poison')\n"
        )
        for role in SOURCE_ROLES:
            self.payloads[role] += poison
            pin(self.payloads, self.bindings, role)
        repin_plan(self.payloads, self.bindings)
        module, _ = self.run_original()
        for name in (
            "unused_poison", "validate_owner_decisions", "validate_owner_review_receipt",
            "_write_json_atomic", "main", "load_first_party_authoring_plan",
        ):
            self.assertNotIn(name, vars(module))
        self.assertEqual(
            {name for name in vars(module) if name.startswith("test_")}, {projection.TEST_NAME},
        )

    def test_file_hash_helper_comes_only_from_first_party_source(self):
        self.payloads["sample_helper_source"] += (
            b"\ndef file_sha256(path):\n    raise RuntimeError('wrong hash helper origin')\n"
        )
        pin(self.payloads, self.bindings, "sample_helper_source")
        module, _ = self.run_original()
        self.assertEqual(
            module.file_sha256.__code__.co_filename,
            SYNTHETIC_ROOT + "/" + projection.SOURCE_PATHS["file_hash_source"],
        )

    def test_existing_module_namespace_is_untouched(self):
        sentinel = types.ModuleType(projection.MODULE_NAME)
        child = types.ModuleType(projection.MODULE_NAME + ".sentinel")
        entries = {projection.MODULE_NAME: sentinel, child.__name__: child}
        with patch.dict(sys.modules, entries):
            module, _ = self.run_original()
            self.assertIsNot(module, sentinel)
            for name, expected in entries.items():
                self.assertIs(sys.modules[name], expected)

    def test_snapshot_has_no_extra_inputs_or_write_and_filesystem_escape_api(self):
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
