"""One-shot, raw-value-free localization of the unchanged accepted OCI policy.

Only a SHA-bound isolated copy of the frozen verification helpers is compiled.
Markers run before original predicate expressions, preserving their order and
short-circuit behavior. No production module or global import machinery is
patched. Successful SoftwareBinding construction alone becomes a boolean;
neither a binding nor a continuation is exposed. Native diagnose is Linux-only
through the unchanged original runtime predicate. Local tests use inert doubles.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import select
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCHEMA = "lumi.sft-oci-inspect-diagnostic.v0.1"
FROZEN_SHA256 = "a2f26771b22fedfe8a003bc3104c40f46477465b7f0ab2c50bee7ad3abb945f0"
FROZEN_BYTES = 15603
MAXIMUM_REPORT_BYTES = 4096
FUNCTIONS = (
    "need", "canonical", "_runtime", "_remaining", "_inspect", "_json",
    "_configuration", "_container", "verify_created_oci",
)
CONSTANTS = (
    "IMAGE_ID", "SOURCE_COMMIT", "LOCK_SHA256", "BASE_SHA256", "SOURCE_SHA256",
    "ACCEPTANCE_SHA256", "PYTHON", "WRAPPER", "TMPFS", "MAXIMUM_INSPECT_BYTES",
    "MAXIMUM_INSPECT_SECONDS", "MAXIMUM_HANDOFF_SECONDS", "ERROR_CODES",
)
ADMISSION_CODES = (
    "configuration_rejected", "runtime_rejected", "owner_rejected", "parent_lost",
    "deadline_exhausted", "already_consumed", "inspect_failed", "inspect_budget_exhausted",
    "inspect_shape_rejected", "image_rejected", "container_rejected", "profile_rejected",
    "handoff_failed", "handoff_return_rejected", "internal_failed",
)
PHASES = (
    "source_binding", "preflight", "image_inspect", "image_json", "image_identity",
    "container_inspect", "container_json", "container_metadata", "configuration",
    "final_deadline", "complete",
)
NONCLAIMS = (
    "software_binding_released", "continuation_called", "real_sft_execution_permitted",
    "production_eligible",
)
SHAPE_NAMES = (
    "absent", "null", "is_string", "is_map", "is_list", "empty_string", "empty_map",
    "empty_list",
)
# Line numbers refer exclusively to FROZEN_SHA256. Every need() site and every
# top-level boolean leaf must match; missing/new sites reject instrumentation.
SPECS = {
    92: ("runtime_platform", "runtime_architecture", "runtime_implementation",
         "runtime_python", "runtime_no_site", "runtime_no_bytecode", "runtime_ml_absent"),
    102: ("deadline_remaining",),
    113: ("inspect_image_kind", "inspect_image_identity", "inspect_container_kind",
          "inspect_container_id_type", "inspect_container_id_format"),
    130: ("inspect_readable",),
    136: ("inspect_stdout_budget",),
    137: ("inspect_exit",),
    152: ("json_bytes_type", "json_byte_budget"),
    158: ("json_unique_keys",),
    166: ("json_list_type", "json_single_object", "json_object_type"),
    173: ("configuration_image_type", "configuration_container_type"),
    176: ("equal",),
    177: ("configuration_user", "configuration_no_volumes", "configuration_workdir",
          "configuration_entrypoint", "configuration_command"),
    182: ("environment_list_type", "environment_count", "environment_items"),
    186: ("environment_unique_keys",),
    195: ("environment_required",),
    199: ("container_id", "container_image", "container_config_image"),
    201: ("container_path", "container_arguments"),
    204: ("container_status", "container_restart_type", "container_restart_zero"),
    207: ("false",),
    209: ("integer", "zero"),
    211: ("unset",),
    213: ("host_readonly", "host_unprivileged", "host_no_autoremove", "host_network_none",
          "host_cap_drop", "host_no_new_privileges"),
    220: ("absent",),
    222: ("empty",),
    223: ("host_ipc_private", "host_tmpfs", "host_restart_policy"),
    228: ("integer", "expected"),
    229: ("host_ulimits",),
    236: ("mounts_list_type", "mounts_count"),
    238: ("mount_type", "mount_destination", "mount_no_source"),
    247: ("input_container_id_type", "input_container_id_format"),
    249: ("input_deadline_type", "input_deadline_budget"),
    252: ("image_id", "image_os", "image_architecture"),
}
FIELD_GROUPS = {
    "configuration": ("Env", "Entrypoint", "Cmd", "User", "WorkingDir", "Labels", "Volumes",
                      "Healthcheck", "StopSignal", "Shell", "ExposedPorts", "OnBuild"),
    "state_boolean": ("Running", "Paused", "Restarting", "OOMKilled", "Dead"),
    "state_integer": ("Pid", "ExitCode"),
    "state_time": ("StartedAt", "FinishedAt"),
    "host_absent": ("Binds", "Mounts", "Devices", "DeviceRequests", "DeviceCgroupRules",
                    "CapAdd", "Links", "PortBindings"),
    "host_namespace": ("PidMode", "UTSMode", "UsernsMode", "CgroupParent"),
    "host_integer": ("Memory", "MemorySwap", "ShmSize", "PidsLimit", "NanoCpus"),
    "environment": ("LUMI_RELEASE_LOCK_SHA256", "LUMI_RELEASE_BASE_INVENTORY_SHA256",
                    "LUMI_RELEASE_SOURCE_INVENTORY_SHA256", "LUMI_RELEASE_SOURCE_COMMIT",
                    "LUMI_WORKER_RUN_TORCH_CI_FIXTURE", "CUDA_VISIBLE_DEVICES",
                    "GLOO_SOCKET_IFNAME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"),
}
LOOP_GROUPS = {176: "configuration", 207: "state_boolean", 209: "state_integer",
               211: "state_time", 220: "host_absent", 222: "host_namespace", 228: "host_integer"}
OPERATIONS = {
    118: "inspect_private_config", 121: "inspect_launch", 129: "inspect_select",
    131: "inspect_read", 163: "json_constant", 165: "json_parse",
    172: "configuration_objects", 181: "environment_object", 185: "environment_mapping",
    203: "state_object", 212: "host_object", 235: "mounts_object",
}
ENTRIES = {252: "image_identity", 259: "final_deadline"}
FIXED_PREDICATES = (
    "source_binding", "already_consumed", "diagnostic_internal", "complete",
    "preflight", "image_inspect", "container_inspect", "image_json", "container_json",
    "container_metadata", "configuration", "image_identity", "final_deadline",
    "environment_item_type", "environment_item_equals", "environment_item_size",
)
_ATTEMPTED = False


class DiagnosticError(ValueError):
    def __init__(self, code="diagnostic_internal_failed"):
        allowed = ("source_rejected", "diagnostic_internal_failed", "report_rejected")
        self.code = code if type(code) is str and code in allowed else "diagnostic_internal_failed"
        super().__init__(self.code)


def _need(value, code="diagnostic_internal_failed"):
    if not value:
        raise DiagnosticError(code) from None


def _field_code(group, key, suffix):
    _need(type(group) is str and group in FIELD_GROUPS and type(key) is str
          and key in FIELD_GROUPS[group] and type(suffix) is str)
    return group + "_" + key.lower() + "_" + suffix


def _predicate_codes():
    values = set(FIXED_PREDICATES) | set(OPERATIONS.values())
    for line, names in SPECS.items():
        if line in LOOP_GROUPS:
            group = LOOP_GROUPS[line]
            values.update(_field_code(group, key, suffix)
                          for key in FIELD_GROUPS[group] for suffix in names)
        else:
            values.update(names)
    values.update(_field_code("environment", key, "expected")
                  for key in FIELD_GROUPS["environment"])
    return frozenset(values)


PREDICATE_CODES = _predicate_codes()


def _empty_shape():
    return {"available": False,
            **{side + "_" + name: False for side in ("image", "container") for name in SHAPE_NAMES}}


def _shape(mapping, key):
    present = key in mapping
    value = mapping.get(key)
    return {"absent": not present, "null": present and value is None,
            "is_string": type(value) is str, "is_map": type(value) is dict,
            "is_list": type(value) is list,
            "empty_string": type(value) is str and len(value) == 0,
            "empty_map": type(value) is dict and len(value) == 0,
            "empty_list": type(value) is list and len(value) == 0}


class _Recorder:
    def __init__(self):
        self.phase, self.predicate = "source_binding", "source_binding"
        self.image_attempted = self.container_attempted = False
        self.shape = _empty_shape()
        self.mapping_valid = True

    def mark(self, predicate):
        if type(predicate) is not str or predicate not in PREDICATE_CODES:
            self.mapping_valid = False
            raise DiagnosticError() from None
        self.predicate = predicate
        self.shape = _empty_shape()

    def check(self, value):
        if not value:
            self.mapping_valid = False
            raise DiagnosticError() from None

    def enter(self, phase):
        self.check(type(phase) is str and phase in PHASES)
        self.phase = phase
        self.mark(phase)

    def field(self, group, key, suffix, *configs):
        try:
            self.mark(_field_code(group, key, suffix))
            if configs:
                _need(group == "configuration" and len(configs) == 2
                      and all(type(value) is dict for value in configs))
                self.shape = {"available": True,
                              **{"image_" + name: value for name, value in
                                 _shape(configs[0], key).items()},
                              **{"container_" + name: value for name, value in
                                 _shape(configs[1], key).items()}}
        except DiagnosticError:
            self.mapping_valid = False
            raise

    def report(self, status, admitted, code):
        return {"schema": SCHEMA, "status": status, "admission_passed": admitted,
                "phase": self.phase, "predicate": self.predicate, "code": code,
                "image_inspect_attempted": self.image_attempted,
                "container_inspect_attempted": self.container_attempted,
                "configuration_shape": dict(self.shape), **dict.fromkeys(NONCLAIMS, False)}


def _marker(name, args):
    return ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=args, keywords=[])


def _marked(expression, call):
    # The original expression remains one intact value and is evaluated once.
    return ast.copy_location(ast.Subscript(
        value=ast.Tuple(elts=[call, expression], ctx=ast.Load()),
        slice=ast.Constant(value=1), ctx=ast.Load()), expression)


class _Instrument(ast.NodeTransformer):
    def __init__(self):
        self.sites, self.operations, self.generators, self.entries = set(), set(), set(), set()
        self.replacements = 0

    def _condition(self, value, names, group=None):
        if isinstance(value, ast.BoolOp):
            value.values = [self._condition(part, names, group) for part in value.values]
            return value
        _need(bool(names))
        name = names.pop(0)
        if group is None:
            call = _marker("_diagnostic_mark", [ast.Constant(value=name)])
        else:
            args = [ast.Constant(value=group), ast.Name(id="key", ctx=ast.Load()),
                    ast.Constant(value=name)]
            if group == "configuration":
                args.extend([ast.Name(id="original", ctx=ast.Load()),
                             ast.Name(id="selected", ctx=ast.Load())])
            call = _marker("_diagnostic_field", args)
        return _marked(value, call)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "need":
            _need(node.lineno in SPECS and node.lineno not in self.sites
                  and len(node.args) == 2 and not node.keywords)
            self.sites.add(node.lineno)
            names = list(SPECS[node.lineno])
            node.args[0] = self._condition(node.args[0], names, LOOP_GROUPS.get(node.lineno))
            _need(not names)
        if isinstance(node.func, ast.Name) and node.func.id == "SoftwareBinding":
            _need(node.lineno == 260 and not node.args and not node.keywords)
            self.replacements += 1
            return ast.copy_location(ast.Constant(value=True), node)
        return node

    def visit_GeneratorExp(self, node):
        node = self.generic_visit(node)
        if node.lineno == 183:
            names = ["environment_item_type", "environment_item_equals", "environment_item_size"]
            node.elt = self._condition(node.elt, names)
            _need(not names)
            self.generators.add(183)
        elif node.lineno == 195:
            node.elt = self._condition(node.elt, ["expected"], "environment")
            self.generators.add(195)
        return node

    def visit(self, node):
        line = getattr(node, "lineno", None)
        operation = OPERATIONS.get(line) if isinstance(node, ast.stmt) else None
        phase = ENTRIES.get(line) if isinstance(node, ast.stmt) else None
        result = super().visit(node)
        if phase is not None:
            _need(line not in self.entries and operation is None)
            self.entries.add(line)
            marker = ast.Expr(value=_marker("_diagnostic_enter", [ast.Constant(value=phase)]))
            return [ast.copy_location(marker, node), result]
        if operation is not None:
            _need(line not in self.operations)
            self.operations.add(line)
            marker = ast.Expr(value=_marker("_diagnostic_mark", [ast.Constant(value=operation)]))
            return [ast.copy_location(marker, node), result]
        return result


def _instrumented_tree(raw):
    _need(type(raw) is bytes and len(raw) == FROZEN_BYTES
          and hashlib.sha256(raw).hexdigest() == FROZEN_SHA256, "source_rejected")
    tree = ast.parse(raw)
    chosen = []
    for node in tree.body:
        if (isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS
                or isinstance(node, ast.ClassDef) and node.name == "OciAdmissionError"
                or isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in CONSTANTS):
            chosen.append(node)
    _need({node.name for node in chosen if isinstance(node, ast.FunctionDef)} == set(FUNCTIONS))
    transform = _Instrument()
    transformed = transform.visit(ast.Module(body=chosen, type_ignores=[]))
    _need(transform.sites == set(SPECS) and transform.operations == set(OPERATIONS)
          and transform.entries == set(ENTRIES) and transform.generators == {183, 195}
          and transform.replacements == 1)
    return ast.fix_missing_locations(transformed)


def _build_namespace(raw, recorder):
    values = {"json": json, "os": os, "re": re, "select": select, "subprocess": subprocess,
              "sys": sys, "tempfile": tempfile, "time": time,
              "_diagnostic_mark": recorder.mark, "_diagnostic_field": recorder.field,
              "_diagnostic_enter": recorder.enter}
    exec(compile(_instrumented_tree(raw), "<bound-oci-inspect-diagnostic>", "exec"), values)
    _need(values["ERROR_CODES"] == frozenset(ADMISSION_CODES)
          and values["MAXIMUM_INSPECT_BYTES"] == 65536
          and values["MAXIMUM_INSPECT_SECONDS"] == 5
          and values["MAXIMUM_HANDOFF_SECONDS"] == 15)
    original_inspect = values["_inspect"]
    original_json = values["_json"]
    original_remaining = values["_remaining"]
    original_container = values["_container"]
    original_configuration = values["_configuration"]
    original_runtime = values["_runtime"]

    def inspect_once(kind, identifier, deadline):
        if kind == "image":
            recorder.check(not recorder.image_attempted and not recorder.container_attempted)
            recorder.image_attempted = True
        else:
            recorder.check(kind == "container" and recorder.image_attempted
                           and not recorder.container_attempted)
            recorder.container_attempted = True
        recorder.enter(kind + "_inspect")
        return original_inspect(kind, identifier, deadline)

    def parse(raw):
        recorder.enter("container_json" if recorder.container_attempted else "image_json")
        try:
            return original_json(raw)
        except values["OciAdmissionError"]:
            raise
        except BaseException:
            recorder.mark("json_parse")
            raise

    def remaining(deadline):
        previous = recorder.phase, recorder.predicate, recorder.shape
        result = original_remaining(deadline)
        recorder.phase, recorder.predicate, recorder.shape = previous
        return result

    def container(value, identifier):
        recorder.enter("container_metadata")
        return original_container(value, identifier)

    def configuration(image, selected):
        recorder.enter("configuration")
        return original_configuration(image, selected)

    def runtime():
        recorder.enter("preflight")
        return original_runtime()

    values.update(_inspect=inspect_once, _json=parse, _remaining=remaining,
                  _container=container, _configuration=configuration, _runtime=runtime)
    return values


def _read_frozen():
    path = Path(__file__).with_name("posttraining_sft_oci_admission_v1.py")
    for ancestor in (path, *path.parents):
        _need(not ancestor.is_symlink(), "source_rejected")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        _need(stat.S_ISREG(before.st_mode) and before.st_size == FROZEN_BYTES, "source_rejected")
        raw = bytearray()
        while len(raw) < FROZEN_BYTES:
            chunk = os.read(fd, FROZEN_BYTES - len(raw))
            _need(bool(chunk), "source_rejected")
            raw.extend(chunk)
        after = os.fstat(fd)
        fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
        _need(not os.read(fd, 1)
              and all(getattr(before, key) == getattr(after, key) for key in fields),
              "source_rejected")
        return bytes(raw)
    finally:
        os.close(fd)


def validate_report(value):
    """Validate the closed raw-free schema and return a fresh aggregate-only value."""
    keys = {"schema", "status", "admission_passed", "phase", "predicate", "code",
            "image_inspect_attempted", "container_inspect_attempted", "configuration_shape",
            *NONCLAIMS}
    _need(type(value) is dict and set(value) == keys, "report_rejected")
    _need(type(value["schema"]) is str and value["schema"] == SCHEMA
          and type(value["status"]) is str and value["status"] in ("passed", "failed")
          and type(value["phase"]) is str and value["phase"] in PHASES
          and type(value["predicate"]) is str and value["predicate"] in PREDICATE_CODES
          and type(value["code"]) is str and value["code"] in
          (*ADMISSION_CODES, "ok", "source_rejected", "diagnostic_internal_failed"),
          "report_rejected")
    flags = ("admission_passed", "image_inspect_attempted", "container_inspect_attempted",
             *NONCLAIMS)
    _need(all(type(value[key]) is bool for key in flags)
          and all(value[key] is False for key in NONCLAIMS), "report_rejected")
    image, container = value["image_inspect_attempted"], value["container_inspect_attempted"]
    _need(not container or image, "report_rejected")
    shape = value["configuration_shape"]
    _need(type(shape) is dict and set(shape) == set(_empty_shape())
          and all(type(item) is bool for item in shape.values()), "report_rejected")
    if shape["available"]:
        _need(value["phase"] == "configuration" and value["predicate"] in
              {_field_code("configuration", key, "equal") for key in FIELD_GROUPS["configuration"]}
              and not value["admission_passed"] and image and container, "report_rejected")
        for side in ("image", "container"):
            selected = {key: shape[side + "_" + key] for key in SHAPE_NAMES}
            _need(sum(selected[key] for key in ("absent", "null", "is_string", "is_map", "is_list"))
                  <= 1, "report_rejected")
            _need(all(not selected["empty_" + kind] or selected["is_" + kind]
                      for kind in ("string", "map", "list")), "report_rejected")
    else:
        _need(not any(shape.values()), "report_rejected")
    if value["admission_passed"]:
        _need(value["status"] == "passed" and value["phase"] == value["predicate"] == "complete"
              and value["code"] == "ok" and image and container, "report_rejected")
    else:
        _need(value["code"] != "ok" and value["phase"] != "complete", "report_rejected")
    if value["status"] == "passed":
        _need(value["code"] in (*ADMISSION_CODES, "ok")
              and value["phase"] != "source_binding", "report_rejected")
    else:
        _need(value["code"] in ("source_rejected", "diagnostic_internal_failed", "already_consumed")
              and not value["admission_passed"] and value["phase"] == "source_binding"
              and value["predicate"] == {"source_rejected": "source_binding",
                                         "diagnostic_internal_failed": "diagnostic_internal",
                                         "already_consumed": "already_consumed"}[value["code"]],
              "report_rejected")
        _need(value["code"] == "diagnostic_internal_failed" or not image, "report_rejected")
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                     allow_nan=False).encode("ascii")
    _need(len(raw) <= MAXIMUM_REPORT_BYTES, "report_rejected")
    return {**value, "configuration_shape": dict(shape)}


def diagnose(container_id, *, deadline):
    """Consume one attempt; passed means localized, never accepted for execution."""
    global _ATTEMPTED
    recorder = _Recorder()
    if _ATTEMPTED:
        recorder.mark("already_consumed")
        return validate_report(recorder.report("failed", False, "already_consumed"))
    _ATTEMPTED = True
    try:
        values = _build_namespace(_read_frozen(), recorder)
        try:
            admitted = values["verify_created_oci"](container_id, deadline=deadline)
        except values["OciAdmissionError"] as error:
            _need(type(error) is values["OciAdmissionError"] and recorder.mapping_valid)
            return validate_report(recorder.report("passed", False, error.code))
        _need(admitted is True and recorder.mapping_valid)
        recorder.enter("complete")
        return validate_report(recorder.report("passed", True, "ok"))
    except BaseException as error:
        code = error.code if type(error) is DiagnosticError else "diagnostic_internal_failed"
        if code not in ("source_rejected", "diagnostic_internal_failed"):
            code = "diagnostic_internal_failed"
        recorder.phase = "source_binding"
        recorder.mark("source_binding" if code == "source_rejected" else "diagnostic_internal")
        return validate_report(recorder.report("failed", False, code))
