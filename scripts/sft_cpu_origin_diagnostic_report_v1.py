"""Pure supplied-byte contract for one raw-text-free CPU-origin diagnostic.

Native source is hash checked and parsed as AST only, never imported or executed.
The source pin and expected controls must come from the caller's trusted control
plane. This parser authenticates neither that caller nor an image/run. Successful
validation is diagnostic evidence, never acceptance, recovery or SFT permission.
No filesystem, process, package-discovery, network, ML or runtime imports occur.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re

SCHEMA = "lumi.sft-cpu-origin-diagnostic.v0.1"
MAXIMUM_SOURCE_BYTES = 131_072
MAXIMUM_AST_NODES = 40_000
MAXIMUM_REPORT_BYTES = 16_384
MAXIMUM_ENUM_VALUES = 256
MAXIMUM_LABEL_BYTES = 128
MAXIMUM_COUNT = 2_000_000
MAXIMUM_ORIGIN_COUNT = 196_608
SPEC_KEYS = (
    "REPORT_KEYS",
    "CONTROL_KEYS",
    "DIAGNOSTIC_ENUMS",
    "COUNT_LIMITS",
    "FROZEN_STAGE_ENUMS",
    "COMPONENT_ENUMS",
    "ORIGIN_COUNT_FIELDS",
    "FALSE_FLAGS",
)
CONTROL_FIELDS = (
    "source_commit",
    "software_lock_sha256",
    "base_inventory_sha256",
    "source_inventory_sha256",
)
DIAGNOSTIC_FIELDS = (
    "captured_check",
    "captured_function",
    "phase",
    "origin_category",
    "module_family",
    "exception_class",
    "probe_error_code",
)
COUNT_FIELDS = (
    "witnesses_started",
    "witnesses_completed",
    "need_checks_observed",
    "failed_outcomes_observed",
    "failure_messages_received",
    "failure_records_retained",
    "rank_workers_completed",
    "invocations_completed",
    "origin_observations_completed",
)
HARD_COUNT_LIMITS = {
    "witnesses_started": 12,
    "witnesses_completed": 12,
    "need_checks_observed": 2_000_000,
    "failed_outcomes_observed": 6,
    "failure_messages_received": 12,
    "failure_records_retained": 6,
    "rank_workers_completed": 6,
    "invocations_completed": 3,
    "origin_observations_completed": 12,
}
STAGE_FIELDS = ("probe_stage", "fixture_stage", "child_stage", "worker_stage", "worker_code")
COMPONENT_FIELDS = (
    "checkpoint_stage",
    "checkpoint_code",
    "step_stage",
    "step_code",
    "input_code",
    "gate_stage",
    "gate_code",
)
DIAGNOSTIC_DEFAULTS = {
    "captured_check": "not_reported",
    "captured_function": "not_reported",
    "phase": "not_reported",
    "origin_category": "not_available",
    "module_family": "none",
    "exception_class": "not_reported",
    "probe_error_code": "not_reported",
}
ORIGIN_FIELDS = (
    "file_backed_modules",
    "source_modules",
    "dependency_modules",
    "runtime_modules",
    "distribution_versions",
    "map_lines",
    "executable_file_mappings",
    "executable_files",
    "anonymous_executable_mappings",
    "generated_python_modules",
)
REPORT_BASE_FIELDS = (
    "schema",
    "status",
    "underlying_status",
    "controls",
    "diagnostic",
    "counts",
    "frozen_stages",
    "frozen_component_diagnostics",
    "frozen_origin_counts",
)
BASE_FALSE_FLAGS = (
    "real_sft_execution_permitted",
    "production_eligible",
    "real_model_or_data_access",
    "cuda_used",
    "complete_dynamic_dependency_closure_verified",
    "all_executable_code_origin_authenticated",
    "large_library_rehash_performed",
    "host_configuration_independently_verified",
    "real_host_admission_verified",
)
ADDITIONAL_FALSE_FLAGS = ("acceptance_passed", "recovery_performed")
ERROR_CODES = frozenset(
    (
        "invalid_source_type",
        "source_budget_exceeded",
        "invalid_source_sha256",
        "source_hash_mismatch",
        "source_syntax_rejected",
        "spec_missing_constants",
        "spec_duplicate_constants",
        "spec_literal_rejected",
        "spec_shape_rejected",
        "spec_enum_rejected",
        "spec_count_rejected",
        "spec_flags_rejected",
        "invalid_report_type",
        "report_budget_exceeded",
        "report_json_rejected",
        "report_duplicate_key",
        "report_number_rejected",
        "report_noncanonical",
        "report_shape_rejected",
        "control_rejected",
        "diagnostic_rejected",
        "count_rejected",
        "stage_rejected",
        "component_rejected",
        "nonclaim_rejected",
        "failed_evidence_missing",
        "passed_evidence_rejected",
        "validation_failed",
    )
)


class ValidationError(ValueError):
    def __init__(self, code="validation_failed"):
        self.code = code if type(code) is str and code in ERROR_CODES else "validation_failed"
        super().__init__(self.code)


def _need(value, code):
    if not value:
        raise ValidationError(code) from None


def _label(value):
    return (
        type(value) is str
        and 1 <= len(value) <= MAXIMUM_LABEL_BYTES
        and re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.:-]*", value) is not None
    )


def _sequence(value, code):
    _need(type(value) in (tuple, list) and 1 <= len(value) <= MAXIMUM_ENUM_VALUES, code)
    _need(all(_label(item) for item in value) and len(value) == len(set(value)), code)
    return tuple(value)


def _enums(value, *, fields=None, defaults=None):
    _need(
        type(value) is dict
        and 1 <= len(value) <= MAXIMUM_ENUM_VALUES
        and all(_label(key) for key in value),
        "spec_enum_rejected",
    )
    if fields is not None:
        _need(set(value) == set(fields), "spec_shape_rejected")
    result = {}
    for key, items in value.items():
        result[key] = _sequence(items, "spec_enum_rejected")
        default = "not_reported" if defaults is None else defaults[key]
        _need(default in result[key], "spec_enum_rejected")
    return result


def _spec(value):
    """Revalidate and copy mutable supplied contracts before every use."""
    _need(type(value) is dict and set(value) == set(SPEC_KEYS), "spec_shape_rejected")
    result = {
        name: _sequence(value[name], "spec_shape_rejected")
        for name in ("REPORT_KEYS", "CONTROL_KEYS", "ORIGIN_COUNT_FIELDS", "FALSE_FLAGS")
    }
    _need(
        set(result["CONTROL_KEYS"]) == set(CONTROL_FIELDS)
        and set(result["ORIGIN_COUNT_FIELDS"]) == set(ORIGIN_FIELDS),
        "spec_shape_rejected",
    )
    flags = set(result["FALSE_FLAGS"])
    _need(flags == set(BASE_FALSE_FLAGS) | set(ADDITIONAL_FALSE_FLAGS), "spec_flags_rejected")
    _need(set(result["REPORT_KEYS"]) == set(REPORT_BASE_FIELDS) | flags, "spec_shape_rejected")
    result["DIAGNOSTIC_ENUMS"] = _enums(
        value["DIAGNOSTIC_ENUMS"], fields=DIAGNOSTIC_FIELDS, defaults=DIAGNOSTIC_DEFAULTS
    )
    result["FROZEN_STAGE_ENUMS"] = _enums(value["FROZEN_STAGE_ENUMS"], fields=STAGE_FIELDS)
    result["COMPONENT_ENUMS"] = _enums(value["COMPONENT_ENUMS"], fields=COMPONENT_FIELDS)
    counts = value["COUNT_LIMITS"]
    _need(type(counts) is dict and set(counts) == set(COUNT_FIELDS), "spec_shape_rejected")
    _need(
        all(
            type(limit) is int and 0 < limit <= min(MAXIMUM_COUNT, HARD_COUNT_LIMITS[key])
            for key, limit in counts.items()
        ),
        "spec_count_rejected",
    )
    result["COUNT_LIMITS"] = dict(counts)
    return result


def _assigned_names(node):
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
        targets = [node.target]
    else:
        return set()
    return {
        part.id for target in targets for part in ast.walk(target) if isinstance(part, ast.Name)
    }


def spec_from_source(source: bytes, expected_sha256: str) -> dict:
    """Extract exactly eight bounded literal constants from hash-pinned source."""
    try:
        _need(type(source) is bytes, "invalid_source_type")
        _need(0 < len(source) <= MAXIMUM_SOURCE_BYTES, "source_budget_exceeded")
        _need(
            type(expected_sha256) is str
            and re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None,
            "invalid_source_sha256",
        )
        _need(hashlib.sha256(source).hexdigest() == expected_sha256, "source_hash_mismatch")
        try:
            tree = ast.parse(source, filename="pinned_diagnostic_source", mode="exec")
        except (SyntaxError, ValueError, UnicodeError, RecursionError):
            raise ValidationError("source_syntax_rejected") from None
        result, seen = {}, set()
        top_level = {id(node) for node in tree.body}
        for index, node in enumerate(ast.walk(tree)):
            _need(index < MAXIMUM_AST_NODES, "source_budget_exceeded")
            selected = _assigned_names(node) & set(SPEC_KEYS)
            if not selected:
                continue
            _need(not selected & seen, "spec_duplicate_constants")
            seen.update(selected)
            _need(
                id(node) in top_level
                and isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and len(selected) == 1,
                "spec_literal_rejected",
            )
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError, RecursionError):
                raise ValidationError("spec_literal_rejected") from None
        _need(set(result) == set(SPEC_KEYS), "spec_missing_constants")
        return _spec(result)
    except ValidationError:
        raise
    except Exception:
        raise ValidationError("validation_failed") from None


def _pairs(items):
    result = {}
    for key, value in items:
        _need(key not in result, "report_duplicate_key")
        result[key] = value
    return result


def _integer(text):
    _need(len(text) <= 12, "report_number_rejected")
    return int(text)


def _reject_number(_text):
    raise ValidationError("report_number_rejected") from None


def _canonical(value):
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii")
        + b"\n"
    )


def _parse_report(raw):
    _need(type(raw) is bytes, "invalid_report_type")
    _need(0 < len(raw) <= MAXIMUM_REPORT_BYTES, "report_budget_exceeded")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_pairs,
            parse_int=_integer,
            parse_float=_reject_number,
            parse_constant=_reject_number,
        )
    except ValidationError:
        raise
    except (ValueError, UnicodeError, RecursionError, OverflowError):
        raise ValidationError("report_json_rejected") from None
    _need(type(value) is dict, "report_shape_rejected")
    _need(_canonical(value) == raw, "report_noncanonical")
    return value


def _controls(value):
    _need(type(value) is dict and set(value) == set(CONTROL_FIELDS), "control_rejected")
    for key, item in value.items():
        size = 40 if key == "source_commit" else 64
        _need(
            type(item) is str and re.fullmatch(r"[0-9a-f]{" + str(size) + "}", item),
            "control_rejected",
        )
    return dict(value)


def _enum_values(value, enums, code):
    _need(type(value) is dict and set(value) == set(enums), code)
    _need(all(type(item) is str and item in enums[key] for key, item in value.items()), code)


def _counts(value, limits):
    _need(type(value) is dict and set(value) == set(limits), "count_rejected")
    _need(
        all(type(item) is int and 0 <= item <= limits[key] for key, item in value.items()),
        "count_rejected",
    )


def validate_report(raw: bytes, expected_controls: dict, spec: dict) -> dict:
    """Return a fresh exact diagnostic report, never an execution permit."""
    try:
        selected = _spec(spec)
        expected = _controls(expected_controls)
        value = _parse_report(raw)
        _need(set(value) == set(selected["REPORT_KEYS"]), "report_shape_rejected")
        _need(
            value["schema"] == SCHEMA
            and value["status"] == "diagnostic_completed"
            and type(value["underlying_status"]) is str
            and value["underlying_status"] in ("passed", "failed"),
            "report_shape_rejected",
        )
        _need(_controls(value["controls"]) == expected, "control_rejected")
        _enum_values(value["diagnostic"], selected["DIAGNOSTIC_ENUMS"], "diagnostic_rejected")
        _enum_values(value["frozen_stages"], selected["FROZEN_STAGE_ENUMS"], "stage_rejected")
        _enum_values(
            value["frozen_component_diagnostics"], selected["COMPONENT_ENUMS"], "component_rejected"
        )
        _counts(value["counts"], selected["COUNT_LIMITS"])
        _counts(
            value["frozen_origin_counts"],
            dict.fromkeys(selected["ORIGIN_COUNT_FIELDS"], MAXIMUM_ORIGIN_COUNT),
        )
        _need(all(value[flag] is False for flag in selected["FALSE_FLAGS"]), "nonclaim_rejected")
        diagnostic, counts = value["diagnostic"], value["counts"]
        if value["underlying_status"] == "failed":
            _need(
                diagnostic["captured_check"] != "not_reported"
                and diagnostic["captured_function"] not in ("not_reported", "run", "entry")
                and re.fullmatch(
                    re.escape(diagnostic["captured_function"]) + r"_(?:check_[0-9]{2}|unexpected)",
                    diagnostic["captured_check"],
                )
                is not None
                and (
                    (
                        diagnostic["captured_check"].endswith("_unexpected")
                        and diagnostic["exception_class"] not in ("not_reported", "ProbeError")
                        and diagnostic["probe_error_code"] == "not_reported"
                    )
                    or (
                        not diagnostic["captured_check"].endswith("_unexpected")
                        and diagnostic["exception_class"] == "ProbeError"
                        and diagnostic["probe_error_code"] != "not_reported"
                    )
                ),
                "failed_evidence_missing",
            )
        else:
            _need(
                counts["rank_workers_completed"] == 6
                and counts["invocations_completed"] == 3
                and counts["origin_observations_completed"] == 12
                and diagnostic == DIAGNOSTIC_DEFAULTS,
                "passed_evidence_rejected",
            )
            _need(
                all(
                    counts[key] == 0
                    for key in (
                        "failed_outcomes_observed",
                        "failure_messages_received",
                        "failure_records_retained",
                    )
                ),
                "passed_evidence_rejected",
            )
        return value
    except ValidationError:
        raise
    except Exception:
        raise ValidationError("validation_failed") from None
