"""Synthetic-only metadata assurance; never historical/private data replay."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import sys
import unittest
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lumi import posttraining_safety_uri_verification_v2 as verifier  # noqa: E402

FROZEN_SOURCE_SHA256 = "be071057c2d3de55069d8dd80fa7c8d802422d0c4eefd8816fa98d50641d63ad"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _seal(report: dict) -> None:
    report["report_sha256"] = _digest(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )


def synthetic_inputs() -> tuple[dict, dict]:
    """Return generated metadata only; callers own external file-byte pinning."""
    bindings = {
        name: {"path": f"synthetic/{name}.json", "sha256": _digest(name)}
        for name in sorted(verifier.EXPECTED_BINDING_NAMES)
    }
    bindings["predecessor_candidate"]["sha256"] = verifier.EXPECTED_PREDECESSOR_SHA256
    bindings["source_registry"]["sha256"] = verifier.EXPECTED_REGISTRY_SHA256
    plan = {
        "authorization": copy.deepcopy(verifier.EXPECTED_AUTHORIZATION),
        "bindings": bindings,
        "limits": copy.deepcopy(verifier.EXPECTED_LIMITS),
        "observed_at": "2026-08-06",
        "output": copy.deepcopy(verifier.EXPECTED_OUTPUT),
        "schema_version": "lumi.sft-safety-uri-repair-plan.v0.1",
        "status": "authorized_bounded_cpu_only_safety_uri_repair",
    }
    candidate_sha = _digest("synthetic candidate bytes not present")
    generic_sha = _digest("synthetic generic validation")
    report = {
        "authorization": copy.deepcopy(verifier.EXPECTED_AUTHORIZATION),
        "bindings": {
            "artifact_manifest_file_sha256": _digest("synthetic manifest"),
            "benchmark_registry_sha256": bindings["benchmark_registry"]["sha256"],
            "generic_validation_report_sha256": generic_sha,
            "output_candidate_file_sha256": candidate_sha,
            "plan_file_sha256": _digest(plan),
            "predecessor_candidate_file_sha256": bindings["predecessor_candidate"]["sha256"],
            "source_registry_sha256": bindings["source_registry"]["sha256"],
        },
        "frozen_token_verification": {
            "by_split": copy.deepcopy(verifier.EXPECTED_SAFETY_TOKENS),
            "chat_template_sha256": bindings["chat_template"]["sha256"],
            "context_window": 4096,
            "tokenizer_json_sha256": bindings["tokenizer_json"]["sha256"],
        },
        "generic_sft_validation": {
            "accepted": 320,
            "benchmark_contamination_rejections": 0,
            "exact_normalized_duplicates": 0,
            "raw_text_retained": False,
            "rejected": 0,
            "rejection_counts": {},
            "report_sha256": generic_sha,
            "source_registry_rejections": 0,
            "total": 320,
        },
        "metadata_repair": {
            "candidate_file_sha256": candidate_sha,
            "content_and_messages_binding_sha256": _digest("synthetic messages binding"),
            "records": 320,
            "records_with_deterministic_source_uri": 320,
            "records_with_only_authorized_metadata_delta": 320,
            "transform_chain_binding_sha256": _digest("synthetic transform binding"),
            "unique_source_uris": 320,
            "uri_set_sha256": _digest("synthetic URI set"),
        },
        "observed_at": "2026-08-06",
        "production_dataset_promotions": 0,
        "production_eligible": False,
        "raw_training_text_retained": False,
        "remaining_blockers": copy.deepcopy(verifier.EXPECTED_REMAINING_BLOCKERS),
        "schema_version": "lumi.sft-safety-uri-repair-report.v0.1",
        "status": "verified_safety_source_uri_repair_nonproduction",
    }
    _seal(report)
    return plan, report


def _legacy_namespace() -> dict:
    """Read only pinned tracked source, never its imports or operational inputs."""
    source = (ROOT / "src/lumi/posttraining_safety_uri_repair.py").read_bytes()
    if hashlib.sha256(source).hexdigest() != FROZEN_SOURCE_SHA256:
        raise AssertionError("frozen safety URI source changed")
    names = {
        "REPORT_SCHEMA_VERSION",
        "VERIFICATION_SCHEMA_VERSION",
        "_SHA256",
        "EXPECTED_AUTHORIZATION",
        "EXPECTED_LIMITS",
        "EXPECTED_OUTPUT",
        "EXPECTED_SAFETY_TOKENS",
        "EXPECTED_REGISTRY_SHA256",
        "EXPECTED_PREDECESSOR_SHA256",
        "canonical_json_bytes",
        "sha256_value",
        "_require_sha256",
        "_verify_internal_sha",
        "validate_safety_uri_repair_report",
        "build_safety_uri_repair_verification",
    }
    nodes = []
    selected = set()
    for node in ast.parse(source).body:
        name = None
        if isinstance(node, ast.FunctionDef):
            name = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
        if name in names:
            selected.add(name)
            nodes.append(node)
    if selected != names:
        raise AssertionError("frozen helper inventory changed")
    module = ast.Module(
        body=[ast.parse("from __future__ import annotations").body[0], *nodes], type_ignores=[]
    )
    namespace = {"hashlib": hashlib, "json": json, "re": re, "Mapping": Mapping}
    exec(
        compile(ast.fix_missing_locations(module), "<frozen-safety-uri-helpers>", "exec"), namespace
    )
    return namespace


class SafetyUriVerificationV2Tests(unittest.TestCase):
    def _audit(self, plan: dict, report: dict) -> dict:
        return verifier.audit_report(
            plan, report, plan_file_sha256=_digest(plan), report_file_sha256=_digest(report)
        )

    def _reject(self, plan: dict, report: dict, *, reseal: bool = True) -> None:
        if reseal:
            _seal(report)
        with self.assertRaisesRegex(ValueError, "^invalid safety URI metadata contract$"):
            self._audit(plan, report)

    def test_normal_pass_is_metadata_only_and_does_not_mutate_inputs(self) -> None:
        plan, report = synthetic_inputs()
        before = copy.deepcopy((plan, report))
        result = self._audit(plan, report)
        self.assertEqual(
            result,
            {
                "schema_version": "lumi.sft-safety-uri-metadata-assurance-report.v0.2",
                "status": "PASS_METADATA_CONTRACT_ONLY",
                "metadata_contract_passed": True,
                "candidate_rows_replayed": False,
                "prior_execution_actions_verified": False,
                "full_private_integration_passed": False,
                "execution_authorized": False,
                "production_eligible": False,
            },
        )
        self.assertEqual((plan, report), before)
        self.assertNotIn("synthetic/", json.dumps(result))
        self.assertNotIn("private", json.dumps(result).replace("full_private_integration", ""))

    def test_metadata_audit_performs_no_file_access(self) -> None:
        plan, report = synthetic_inputs()
        with (
            patch("builtins.open", side_effect=AssertionError("file access")),
            patch("io.open", side_effect=AssertionError("file access")),
            patch("os.open", side_effect=AssertionError("file access")),
        ):
            self.assertTrue(self._audit(plan, report)["metadata_contract_passed"])

    def test_constant_pins_match_hash_pinned_original_source(self) -> None:
        original = _legacy_namespace()
        for name in (
            "EXPECTED_AUTHORIZATION",
            "EXPECTED_LIMITS",
            "EXPECTED_OUTPUT",
            "EXPECTED_SAFETY_TOKENS",
            "EXPECTED_REGISTRY_SHA256",
            "EXPECTED_PREDECESSOR_SHA256",
        ):
            with self.subTest(name=name):
                self.assertEqual(getattr(verifier, name), original[name])

    def test_actual_legacy_verifier_false_accepts_resealed_token_drift(self) -> None:
        plan, report = synthetic_inputs()
        report["frozen_token_verification"]["by_split"]["train"]["supervised_tokens"] = 0
        _seal(report)
        legacy = _legacy_namespace()["build_safety_uri_repair_verification"](
            plan=plan, report=report, report_file_sha256=_digest(report)
        )
        self.assertTrue(legacy["checks"]["frozen_4k_token_counts_unchanged"])
        self._reject(plan, report)

    def test_actual_legacy_verifier_false_accepts_resealed_authorization_drift(self) -> None:
        plan, report = synthetic_inputs()
        report["authorization"]["network_access_authorized"] = True
        _seal(report)
        legacy = _legacy_namespace()["build_safety_uri_repair_verification"](
            plan=plan, report=report, report_file_sha256=_digest(report)
        )
        self.assertTrue(legacy["checks"]["no_network_api_gpu_seal_sft_or_production_promotion"])
        self._reject(plan, report)

    def test_every_authorization_flag_is_exact_on_plan_and_report(self) -> None:
        for target in ("plan", "report"):
            for key, value in verifier.EXPECTED_AUTHORIZATION.items():
                if type(value) is bool:
                    with self.subTest(target=target, key=key):
                        plan, report = synthetic_inputs()
                        (plan if target == "plan" else report)["authorization"][key] = not value
                        report["bindings"]["plan_file_sha256"] = _digest(plan)
                        self._reject(plan, report)

    def test_exact_token_counts_and_context_window(self) -> None:
        for split, field in (
            ("train", "records"),
            ("train", "supervised_tokens"),
            ("validation", "records"),
            ("validation", "supervised_tokens"),
        ):
            with self.subTest(split=split, field=field):
                plan, report = synthetic_inputs()
                report["frozen_token_verification"]["by_split"][split][field] += 1
                self._reject(plan, report)
        plan, report = synthetic_inputs()
        report["frozen_token_verification"]["context_window"] = 8192
        self._reject(plan, report)

    def test_report_unknown_or_missing_fields_rejected_at_every_level(self) -> None:
        for path in (
            (),
            ("authorization",),
            ("bindings",),
            ("frozen_token_verification",),
            ("frozen_token_verification", "by_split"),
            ("frozen_token_verification", "by_split", "train"),
            ("generic_sft_validation",),
            ("metadata_repair",),
        ):
            for extra in (False, True):
                with self.subTest(path=path, extra=extra):
                    plan, report = synthetic_inputs()
                    section = report
                    for key in path:
                        section = section[key]
                    if extra:
                        section["messages"] = "SYNTHETIC_RAW_TEXT_SENTINEL"
                    else:
                        section.pop(next(iter(section)))
                    self._reject(plan, report)

    def test_plan_unknown_or_missing_fields_rejected_at_every_level(self) -> None:
        for path in (
            (),
            ("authorization",),
            ("limits",),
            ("output",),
            ("bindings",),
            ("bindings", "runner"),
        ):
            for extra in (False, True):
                with self.subTest(path=path, extra=extra):
                    plan, report = synthetic_inputs()
                    section = plan
                    for key in path:
                        section = section[key]
                    if extra:
                        section["content"] = "SYNTHETIC_RAW_TEXT_SENTINEL"
                    else:
                        section.pop(next(iter(section)))
                    report["bindings"]["plan_file_sha256"] = _digest(plan)
                    self._reject(plan, report)

    def test_boolean_numeric_aliases_are_rejected(self) -> None:
        cases = (
            (("production_dataset_promotions",), False),
            (("production_eligible",), 0),
            (("raw_training_text_retained",), 0),
            (("authorization", "source_uri_repair_authorized"), 1),
            (("authorization", "network_access_authorized"), 0),
            (("generic_sft_validation", "rejected"), False),
            (("generic_sft_validation", "accepted"), 320.0),
            (("generic_sft_validation", "raw_text_retained"), 0),
            (("metadata_repair", "records"), 320.0),
            (("frozen_token_verification", "context_window"), 4096.0),
            (("frozen_token_verification", "by_split", "train", "records"), 240.0),
        )
        for path, value in cases:
            with self.subTest(path=path):
                plan, report = synthetic_inputs()
                section = report
                for key in path[:-1]:
                    section = section[key]
                section[path[-1]] = value
                self._reject(plan, report)
        plan, report = synthetic_inputs()
        plan["limits"]["maximum_network_bytes"] = False
        report["bindings"]["plan_file_sha256"] = _digest(plan)
        self._reject(plan, report)

    def test_all_crosslinked_hashes_reject_divergence(self) -> None:
        paths = [
            ("bindings", name)
            for name in (
                "benchmark_registry_sha256",
                "generic_validation_report_sha256",
                "output_candidate_file_sha256",
                "plan_file_sha256",
                "predecessor_candidate_file_sha256",
                "source_registry_sha256",
            )
        ] + [
            ("frozen_token_verification", name)
            for name in ("chat_template_sha256", "tokenizer_json_sha256")
        ]
        for section, key in paths:
            with self.subTest(section=section, key=key):
                plan, report = synthetic_inputs()
                report[section][key] = "0" * 64
                self._reject(plan, report)

    def test_hash_syntax_and_external_hash_types(self) -> None:
        for bad in (None, True, 123, "a" * 63, "A" * 64, "g" * 64, "a" * 64 + "\n"):
            with self.subTest(value=bad):
                plan, report = synthetic_inputs()
                report["metadata_repair"]["uri_set_sha256"] = bad
                self._reject(plan, report)
                for argument in ("plan_file_sha256", "report_file_sha256"):
                    arguments = {
                        "plan_file_sha256": _digest(plan),
                        "report_file_sha256": _digest(report),
                        argument: bad,
                    }
                    with self.assertRaises(ValueError):
                        verifier.audit_report(plan, report, **arguments)

    def test_internal_report_seal_must_match_canonical_metadata(self) -> None:
        plan, report = synthetic_inputs()
        report["report_sha256"] = "0" * 64
        self._reject(plan, report, reseal=False)

    def test_static_report_pins_and_nonproduction_limits(self) -> None:
        for key, value in (
            ("schema_version", "other"),
            ("status", "other"),
            ("observed_at", "2026-08-07"),
            ("production_dataset_promotions", 1),
            ("production_eligible", True),
            ("raw_training_text_retained", True),
            ("remaining_blockers", []),
        ):
            with self.subTest(key=key):
                plan, report = synthetic_inputs()
                report[key] = value
                self._reject(plan, report)

    def test_all_generic_counts_and_empty_rejection_inventory(self) -> None:
        for key in (
            "accepted",
            "benchmark_contamination_rejections",
            "exact_normalized_duplicates",
            "rejected",
            "source_registry_rejections",
            "total",
        ):
            with self.subTest(key=key):
                plan, report = synthetic_inputs()
                report["generic_sft_validation"][key] += 1
                self._reject(plan, report)
        plan, report = synthetic_inputs()
        report["generic_sft_validation"]["rejection_counts"] = {"synthetic_rejection": 0}
        self._reject(plan, report)

    def test_all_four_metadata_counts_are_exact(self) -> None:
        for key in (
            "records",
            "records_with_deterministic_source_uri",
            "records_with_only_authorized_metadata_delta",
            "unique_source_uris",
        ):
            with self.subTest(key=key):
                plan, report = synthetic_inputs()
                report["metadata_repair"][key] = 319
                self._reject(plan, report)

    def test_plan_binding_pins_and_path_metadata(self) -> None:
        for name in ("predecessor_candidate", "source_registry"):
            plan, report = synthetic_inputs()
            plan["bindings"][name]["sha256"] = "0" * 64
            report["bindings"]["plan_file_sha256"] = _digest(plan)
            self._reject(plan, report)
        for value in (None, True, "", " ", "synthetic\x00path", "synthetic\npath"):
            with self.subTest(value=value):
                plan, report = synthetic_inputs()
                plan["bindings"]["runner"]["path"] = value
                report["bindings"]["plan_file_sha256"] = _digest(plan)
                self._reject(plan, report)

    def test_wrong_container_types_are_rejected(self) -> None:
        for value in (None, [], "synthetic", True):
            with self.subTest(value=value):
                plan, report = synthetic_inputs()
                with self.assertRaises(ValueError):
                    self._audit(value, report)
                with self.assertRaises(ValueError):
                    self._audit(plan, value)

    def test_external_report_byte_authenticity_remains_callers_responsibility(self) -> None:
        plan, report = synthetic_inputs()
        result = verifier.audit_report(
            plan, report, plan_file_sha256=_digest(plan), report_file_sha256="0" * 64
        )
        self.assertFalse(result["candidate_rows_replayed"])
        self.assertFalse(result["prior_execution_actions_verified"])
        self.assertFalse(result["execution_authorized"])


if __name__ == "__main__":
    unittest.main()
