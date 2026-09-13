"""Pure independently pinned DCP/HF logical-I/O contract; never execution authority.

The original plan and all of its resource limits remain unchanged. These caps
partition checkpoint and HF/control content accounting; they are not measured
physical I/O, OS enforcement, or a new live-memory allowance. No file, network,
ML, payload, or execution APIs are invoked by this module.
"""

from __future__ import annotations

from . import posttraining_checkpoint_import_plan_v1 as planner
from . import posttraining_checkpoint_import_torch_v5 as kernel
from .posttraining_final_base_metadata_v1 import canonical, json_object, sha

SCHEMA = "lumi.sft-checkpoint-kernel-io-contract.v0.1"
FIELDS = frozenset({"schema_version", "plan_sha256", "checkpoint_read_contract",
                    "hf_io_contract", "contract_sha256"})
ERROR_CODES = frozenset({"plan_binding", "contract_shape", "contract_binding",
                         "checkpoint_contract", "hf_contract"})


class KernelIOContractError(ValueError):
    def __init__(self, code="contract_shape"):
        self.code = code if type(code) is str and code in ERROR_CODES else "contract_shape"
        super().__init__(self.code)


def _need(value, code):
    if not value:
        raise KernelIOContractError(code) from None


def _positive(value):
    return type(value) is int and 0 < value <= 2**63 - 1


def _digest(value):
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _copy(value, code):
    remaining = planner.MAX_PLAN_BYTES

    def visit(item, depth=0):
        nonlocal remaining
        _need(depth <= 24 and remaining > 0, code)
        remaining -= 8
        if type(item) is dict:
            _need(len(item) <= remaining // 8, code)
            for key, child in item.items():
                _need(type(key) is str, code)
                visit(key, depth + 1)
                visit(child, depth + 1)
        elif type(item) is list:
            _need(len(item) <= remaining // 8, code)
            for child in item:
                visit(child, depth + 1)
        elif type(item) is str:
            _need(len(item) <= remaining, code)
            remaining -= len(item.encode("utf-8"))
        else:
            _need(item is None or type(item) in {bool, int, float}, code)
            if type(item) is int:
                _need(abs(item) <= 2**63 - 1, code)
        _need(remaining >= 0, code)

    try:
        visit(value)
        raw = canonical(value)
        _need(0 < len(raw) <= planner.MAX_PLAN_BYTES, code)
        return json_object(raw)
    except KernelIOContractError:
        raise
    except BaseException:
        raise KernelIOContractError(code) from None


def _plan(value):
    plan = _copy(value, "plan_binding")
    try:
        pin = plan["plan_sha256"]
        _need(_digest(pin) and pin == sha(canonical({
            key: child for key, child in plan.items() if key != "plan_sha256"})), "plan_binding")
        _need(plan["schema_version"] == planner.SCHEMA
              and plan["status"] == "PASS_plan_only_not_execution"
              and all(plan.get(key) is False for key in planner.FALSE_FLAGS), "plan_binding")
        bounds = plan["resource_bounds"]
        for key in ("checkpoint_inventory_bytes", "checkpoint_inventory_files",
                    "maximum_backend_output_bytes", "maximum_projected_live_tensor_bytes"):
            _need(_positive(bounds[key]), "plan_binding")
        inventory = plan["checkpoint_inventory"]
        _need(type(inventory) is list and len(inventory) == bounds["checkpoint_inventory_files"]
              and 0 < len(inventory) <= 129, "plan_binding")
        names = set()
        for row in inventory:
            _need(type(row) is dict and set(row) == {"path", "bytes", "sha256"}
                  and type(row["path"]) is str and row["path"] not in names
                  and _positive(row["bytes"]) and _digest(row["sha256"]), "plan_binding")
            names.add(row["path"])
        _need(sum(row["bytes"] for row in inventory) == bounds["checkpoint_inventory_bytes"],
              "plan_binding")
        manifest = [row for row in inventory if row["path"] == "manifest.sha256"]
        _need(len(manifest) == 1 and manifest[0]["sha256"]
              == plan["metadata"]["checkpoint_manifest_sha256"], "plan_binding")
        return plan
    except KernelIOContractError:
        raise
    except BaseException:
        raise KernelIOContractError("plan_binding") from None


def _normalized(plan, contract):
    _need(type(contract) is dict and set(contract) == FIELDS
          and contract["schema_version"] == SCHEMA
          and contract["plan_sha256"] == plan["plan_sha256"], "contract_shape")
    try:
        checkpoint = kernel._read_contract(contract["checkpoint_read_contract"])
    except BaseException:
        raise KernelIOContractError("checkpoint_contract") from None
    try:
        hf = kernel._hf_contract(contract["hf_io_contract"])
    except BaseException:
        raise KernelIOContractError("hf_contract") from None
    bounds = plan["resource_bounds"]
    _need(checkpoint["expected_manifest_sha256"]
          == plan["metadata"]["checkpoint_manifest_sha256"]
          and max(row["bytes"] for row in plan["checkpoint_inventory"])
          <= checkpoint["maximum_file_bytes"] <= bounds["checkpoint_inventory_bytes"]
          and checkpoint["maximum_read_bytes"] >= bounds["checkpoint_inventory_bytes"],
          "checkpoint_contract")
    _need(hf["maximum_write_bytes"] <= bounds["maximum_backend_output_bytes"]
          and hf["maximum_file_bytes"] <= bounds["maximum_backend_output_bytes"], "hf_contract")
    return {"schema_version": SCHEMA, "plan_sha256": plan["plan_sha256"],
            "checkpoint_read_contract": checkpoint, "hf_io_contract": hf,
            "contract_sha256": contract["contract_sha256"]}


def build_kernel_io_contract(plan, *, checkpoint_read_contract, hf_io_contract):
    """Build a pure copied contract; its hash must be independently pinned by the caller."""
    plan = _plan(plan)
    unsigned = {"schema_version": SCHEMA, "plan_sha256": plan["plan_sha256"],
                "checkpoint_read_contract": checkpoint_read_contract,
                "hf_io_contract": hf_io_contract}
    copied = _copy(unsigned, "contract_shape")
    copied["contract_sha256"] = sha(canonical(copied))
    return validate_kernel_io_contract(plan, copied, copied["contract_sha256"])


def validate_kernel_io_contract(plan, contract, expected_sha256):
    """Verify external and internal binding without raising any original plan limit."""
    plan = _plan(plan)
    contract = _copy(contract, "contract_shape")
    _need(_digest(expected_sha256) and contract.get("contract_sha256") == expected_sha256
          and sha(canonical({key: value for key, value in contract.items()
                             if key != "contract_sha256"})) == expected_sha256,
          "contract_binding")
    return _normalized(plan, contract)
