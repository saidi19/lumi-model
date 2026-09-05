"""Stdlib admission development boundary; there is NO executable GPU worker.

Import with Python ``-S -B`` in a controlled stdlib-only process. The historical
``lumi`` package can auto-import ML before importing submodules in a normal ML
environment; this file does not retroactively protect such an import. A future
standalone bootstrap, authenticated target receipt reader and OS supervisor are
still required. Nothing here dispatches the historical worker.

Metadata pins are supplied by the caller's trusted control plane, not extracted
from an untrusted contract. Comparing hashes does not authenticate an approver
or the target receipt contents. Successful inspection NEVER grants a permit.
The public ``run_worker`` entry is unconditionally closed, including for valid
metadata and all development fixtures.

The separate synthetic callback harness tests sequencing only. Its callbacks
must be inert test doubles, not data/ML/CUDA implementations. It is not a
sandbox for arbitrary callback code, an OS quota, or a real admission adapter.
"""

from __future__ import annotations

import hashlib
import sys
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable, NoReturn

from . import posttraining_sft_launch_contract_v1 as launch

# Exact canonical metadata fingerprint of the immutable v0.1 draft. No caller-
# supplied reference can substitute a different base, data lineage or profile.
FROZEN_DRAFT_SHA256 = "9e3a509ee964c90839135bc457efdd75c189c04cf37266f12bb665ed040cde0c"
MAXIMUM_CONTROL_METADATA_BYTES = 65_536
SYNTHETIC_HOST = "SFT-ADMISSION-SYNTHETIC-ONLY"
SYNTHETIC_APPROVAL_TEXT_SHA256 = hashlib.sha256(
    b"SFT-ADMISSION-SYNTHETIC-ONLY:NOT-AN-EXECUTION-APPROVAL"
).hexdigest()
SYNTHETIC_PATHS = {
    "python_path": "/synthetic/no-io/runtime/python",
    "checkpoint_directory": "/synthetic/no-io/import/checkpoint",
    "encoded_directory": "/synthetic/no-io/data/encoded",
    "private_output_root": "/synthetic/no-io/output/private",
    "authorization_path": "/synthetic/no-io/approval/authorization.json",
}


class Code(str, Enum):
    EXECUTION_UNAVAILABLE = "execution_unavailable_in_admission_development_v1"
    STDLIB_RUNTIME_REQUIRED = "stdlib_no_site_no_bytecode_runtime_required"
    INVALID_PINS = "invalid_external_admission_pins"
    INVALID_METADATA = "invalid_control_metadata"
    METADATA_TOO_LARGE = "control_metadata_size_limit"
    CONTRACT_PIN_MISMATCH = "contract_external_pin_mismatch"
    FROZEN_BINDING_CHANGED = "frozen_base_data_profile_or_prohibition_changed"
    REQUIRED_SLOT_MISSING = "required_admission_slot_missing"
    AUTHORIZATION_REJECTED = "sft_authorization_rejected"
    CONTRACT_REJECTED = "launch_contract_rejected"
    SYNTHETIC_DOMAIN_REQUIRED = "synthetic_only_domain_required"
    INVALID_CALLBACKS = "synthetic_callback_schema_rejected"
    SESSION_CONSUMED = "synthetic_session_already_consumed"
    TARGET_REJECTED = "synthetic_target_receipt_check_rejected"
    SUPERVISOR_REJECTED = "synthetic_supervisor_attachment_rejected"
    DATA_REJECTED = "synthetic_data_stage_rejected"
    ML_REJECTED = "synthetic_ml_stage_rejected"
    CUDA_REJECTED = "synthetic_cuda_stage_rejected"
    WORKER_REJECTED = "synthetic_worker_stage_rejected"


class AdmissionDenied(PermissionError):
    """Only a fixed code; never include source, candidate, identity or error text."""

    def __init__(self, code: Code) -> None:
        self.code = code
        super().__init__(code.value)


def _deny(code: Code) -> NoReturn:
    raise AdmissionDenied(code) from None


@dataclass(frozen=True, slots=True)
class ExternalPins:
    """Canonical JSON digests and a current timestamp from a trusted caller.

    These are mandatory expectations, not values to copy from the document
    being checked. This type does not authenticate their out-of-band provenance.
    """

    contract_sha256: str
    authorization_sha256: str
    now_unix: int


@dataclass(frozen=True, slots=True)
class MetadataInspection:
    contract_sha256: str
    authorization_sha256: str
    checked_at_unix: int
    execution_permitted: bool = False
    receipt_contents_authenticated: bool = False
    operating_system_supervision_attached: bool = False


def _runtime_check() -> None:
    if (
        sys.flags.no_site != 1
        or not sys.dont_write_bytecode
        or any(
            name == root or name.startswith(root + ".")
            for name in sys.modules
            for root in ("torch", "transformers", "cuda", "cupy")
        )
    ):
        _deny(Code.STDLIB_RUNTIME_REQUIRED)


def _snapshot(contract: dict) -> dict:
    if type(contract) is not dict:
        _deny(Code.INVALID_METADATA)
    try:
        raw = launch.canonical(contract)
        if len(raw) > MAXIMUM_CONTROL_METADATA_BYTES:
            _deny(Code.METADATA_TOO_LARGE)
        return launch.strict_json(raw)
    except AdmissionDenied:
        raise
    except (ValueError, TypeError, RecursionError, OverflowError):
        _deny(Code.INVALID_METADATA)


def inspect_metadata(contract: dict, pins: ExternalPins) -> MetadataInspection:
    """Validate fixed SFT metadata before any target/data/ML callbacks.

    Takes an isolated canonical snapshot to avoid mutable caller state changing
    within validation. There are no file reads, receipt callbacks, worker
    capabilities, argument dispatch or ML imports. Receipts remain claims.
    """
    _runtime_check()
    if type(pins) is not ExternalPins:
        _deny(Code.INVALID_PINS)
    try:
        launch.sha(pins.contract_sha256)
        launch.sha(pins.authorization_sha256)
        launch.positive_int(pins.now_unix)
    except (ValueError, TypeError):
        _deny(Code.INVALID_PINS)
    snapshot = _snapshot(contract)
    if launch.digest(snapshot) != pins.contract_sha256:
        _deny(Code.CONTRACT_PIN_MISMATCH)
    reference = {**snapshot, **{slot: None for slot in launch.SLOTS}}
    if launch.digest(reference) != FROZEN_DRAFT_SHA256:
        _deny(Code.FROZEN_BINDING_CHANGED)
    if any(snapshot.get(slot) is None for slot in launch.SLOTS):
        _deny(Code.REQUIRED_SLOT_MISSING)
    try:
        accepted = launch.validate_authorization(
            snapshot["execution_authorization"],
            snapshot,
            pins.authorization_sha256,
            pins.now_unix,
        )
        if accepted is not True:
            _deny(Code.AUTHORIZATION_REJECTED)
    except (ValueError, TypeError, KeyError):
        _deny(Code.AUTHORIZATION_REJECTED)
    try:
        result = launch.analyze(
            snapshot,
            reference,
            expected_authorization_sha256=pins.authorization_sha256,
            now_unix=pins.now_unix,
        )
    except (ValueError, TypeError, KeyError, IndexError, OverflowError):
        _deny(Code.CONTRACT_REJECTED)
    if result["execution_permitted"] is not False:
        _deny(Code.CONTRACT_REJECTED)
    return MetadataInspection(
        contract_sha256=pins.contract_sha256,
        authorization_sha256=pins.authorization_sha256,
        checked_at_unix=pins.now_unix,
    )


def run_worker(*args: object, **kwargs: object) -> NoReturn:
    """Public real-worker boundary: deny BEFORE examining arguments or callbacks.

    It deliberately accepts no execution permit, development adapter or bypass
    flag. Valid metadata and synthetic successes cannot make this executable.
    """
    _deny(Code.EXECUTION_UNAVAILABLE)


class SyntheticOutcome(Enum):
    ACCEPT = "synthetic_accept"
    REJECT = "synthetic_reject"


class SyntheticStage(str, Enum):
    METADATA = "synthetic_metadata_checked"
    TARGET_RECEIPTS = "synthetic_target_receipts_checked"
    SUPERVISOR = "synthetic_supervisor_attached"
    DATA = "synthetic_inputs_loaded"
    ML = "synthetic_ml_imported"
    CUDA = "synthetic_cuda_initialized"
    WORKER = "synthetic_worker_called"


SyntheticCallback = Callable[[], SyntheticOutcome]


@dataclass(frozen=True, slots=True)
class SyntheticCallbacks:
    """Inert test doubles only; callbacks receive no paths, records or handles."""

    simulate_target_receipt_verification: SyntheticCallback
    simulate_supervisor_attachment: SyntheticCallback
    simulate_input_loading: SyntheticCallback
    simulate_ml_import: SyntheticCallback
    simulate_cuda_initialization: SyntheticCallback
    simulate_worker: SyntheticCallback


_STAGES = (
    ("simulate_target_receipt_verification", SyntheticStage.TARGET_RECEIPTS, Code.TARGET_REJECTED),
    ("simulate_supervisor_attachment", SyntheticStage.SUPERVISOR, Code.SUPERVISOR_REJECTED),
    ("simulate_input_loading", SyntheticStage.DATA, Code.DATA_REJECTED),
    ("simulate_ml_import", SyntheticStage.ML, Code.ML_REJECTED),
    ("simulate_cuda_initialization", SyntheticStage.CUDA, Code.CUDA_REJECTED),
    ("simulate_worker", SyntheticStage.WORKER, Code.WORKER_REJECTED),
)


@dataclass(frozen=True, slots=True)
class SyntheticTrace:
    stages: tuple[SyntheticStage, ...]
    synthetic_only: bool = True
    execution_permitted: bool = False
    target_receipt_contents_authenticated: bool = False
    operating_system_supervision_attached: bool = False


def _require_synthetic_domain(contract: dict) -> None:
    target = contract.get("target")
    auth = contract.get("execution_authorization")
    if (
        type(target) is not dict
        or type(auth) is not dict
        or target.get("host_id") != SYNTHETIC_HOST
        or target.get("gpu_uuids") != [f"GPU-SYNTHETIC-{rank}" for rank in range(8)]
        or any(target.get(name) != path for name, path in SYNTHETIC_PATHS.items())
        or auth.get("owner_approval_text_sha256") != SYNTHETIC_APPROVAL_TEXT_SHA256
    ):
        _deny(Code.SYNTHETIC_DOMAIN_REQUIRED)


class SyntheticSession:
    """One-use in-memory ordering simulation, not a durable invocation marker.

    A real launch contract cannot select this adapter: the frozen v0.1 contract
    requires execution_adapter=None and execution_enabled=False. A caller must
    invoke this development API explicitly with synthetic-only metadata and
    inert callbacks. A successful trace is not a worker admission credential.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._consumed = False

    def exercise(
        self, contract: dict, pins: ExternalPins, callbacks: SyntheticCallbacks
    ) -> SyntheticTrace:
        # Consume even a rejected attempt. Only one thread may begin this session.
        with self._lock:
            if self._consumed:
                _deny(Code.SESSION_CONSUMED)
            self._consumed = True
        snapshot = _snapshot(contract)
        _require_synthetic_domain(snapshot)
        inspect_metadata(snapshot, pins)
        if type(callbacks) is not SyntheticCallbacks or not all(
            callable(getattr(callbacks, name)) for name, _, _ in _STAGES
        ):
            _deny(Code.INVALID_CALLBACKS)
        stages = [SyntheticStage.METADATA]
        for name, stage, failure in _STAGES:
            try:
                outcome = getattr(callbacks, name)()
            except BaseException:
                # Callback errors, including interruption, retain no arbitrary
                # exception message and never progress to a later stage.
                _deny(failure)
            if outcome is not SyntheticOutcome.ACCEPT:
                _deny(failure)
            stages.append(stage)
        return SyntheticTrace(stages=tuple(stages))
