"""Actual tiny synthetic CPU SFT after a separate, stdlib-first admission.

Only the newly generated fixture is in scope. The frozen v2 trainer is called
without changing its objective, checkpoint implementation or resume semantics.
This is a trusted local integration, NOT production admission, an adversarial
sandbox, hard aggregate I/O enforcement, memory isolation or a network meter.
An outer stdlib process owns the runner's session, wall deadline and cleanup.
The timeout requires that controller to remain alive, plus bounded cleanup;
SIGKILL/controller death is not guarded. A valid report and EOF are followed by
deliberate runner-session termination, not an assertion of natural runner exit.
The fresh stdlib controller requires default SIGCHLD and no external child reaper.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import os
import platform
import re
import select
import signal
import stat
import struct
import subprocess
import sys
import sysconfig
import tempfile
import time
import types
from collections import Counter
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

SCHEMA = "lumi.sft-offline-cpu-integration.v0.1"
PACKAGE = "_lumi_sft_offline"
MARKER = "SYNTHETIC-CPU-ONLY.json"
MAX_CONTROL_BYTES = 16_384
MAX_REPORT_BYTES = 4_096
SCHEDULE = (("reference", 2, False), ("resumed", 1, False), ("resumed", 2, True))
CHILD_BOOTSTRAP = "import runpy,sys; runpy.run_path(sys.argv[1])['_child_main'](int(sys.argv[2]))"
STATE_TIMING = frozenset(("created_at", "updated_at", "last_checkpoint_seconds"))
METRIC_TIMING = frozenset(("elapsed_seconds", "tokens_per_second", "checkpoint_seconds"))


class IntegrationError(RuntimeError):
    """Deliberately fixed and text-free failure surface."""

    def __init__(self):
        super().__init__("synthetic_cpu_integration_failed")


def _need(condition):
    if not condition:
        raise IntegrationError()


def _repository():
    return Path(__file__).resolve().parents[2]


def _admission():
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(_repository() / "src" / "lumi")]
        sys.modules[PACKAGE] = package
    return importlib.import_module(PACKAGE + ".posttraining_sft_cpu_admission_v1")


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result)
            result[key] = value
        return result

    def constant(_value):
        raise IntegrationError()

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _admit(raw, pin):
    admission = _admission()
    admission.pre_ml_runtime_check()
    return admission.validate_request(
        raw,
        pin,
        no_site=sys.flags.no_site,
        no_bytecode=sys.dont_write_bytecode,
    )


def _runtime_paths():
    """After admission only: add a fixed venv directory, never site/.pth hooks."""
    repository = _repository()
    executable = Path(os.path.abspath(sys.executable))
    _need(executable.is_file())
    prefix = executable.parent.parent
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    if (prefix / "pyvenv.cfg").is_file():
        candidates = [prefix / "lib" / version / "site-packages"]
        allowed = prefix.resolve()
    else:
        candidates = [Path(sysconfig.get_path(name)) for name in ("purelib", "platlib")]
        allowed = Path(sys.base_prefix).resolve()
    candidates = list(dict.fromkeys(candidates))
    for packages in candidates:
        _need(packages.is_absolute() and packages.is_dir() and not packages.is_symlink())
        _need(packages.resolve().is_relative_to(allowed))
    for name in ("torch", "transformers"):
        _need(any((packages / name / "__init__.py").is_file() for packages in candidates))
    sys.path[:0] = [str(repository / "src"), *(str(path) for path in candidates)]


def _environment(root):
    # Never inherit user credentials, SSH agents, provider configuration or HOME.
    return {
        "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": "C.UTF-8",
        "TMPDIR": str(root.parent),
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HOME": str(root / "cache" / "hf"),
        "TORCH_HOME": str(root / "cache" / "torch"),
        "XDG_CACHE_HOME": str(root / "cache"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "GLOO_SOCKET_IFNAME": "lo0" if platform.system() == "Darwin" else "lo",
    }


def _deny_python_external_network(event, arguments):
    """Defense in depth only; native Gloo is constrained by fixed loopback IF."""
    if event in {"socket.connect", "socket.bind"}:
        address = arguments[1]
        _need(isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"})
    elif event == "socket.getaddrinfo":
        _need(arguments[0] in {"localhost", "127.0.0.1", "::1"})


def _initialize_ml(root):
    temporary = root / "temporary"
    temporary.mkdir(mode=0o700, exist_ok=True)
    _need(not temporary.is_symlink())
    os.environ["TMPDIR"] = str(temporary)
    tempfile.tempdir = str(temporary)
    _runtime_paths()
    sys.addaudithook(_deny_python_external_network)
    import torch
    import transformers

    torch_version = tuple(_version_numbers(torch.__version__)[:2])
    transformers_version = tuple(_version_numbers(transformers.__version__)[:2])
    _need((2, 5) <= torch_version < (3, 0))
    _need((5, 2) <= transformers_version < (6, 0))
    torch.set_num_threads(1)
    _need(not torch.cuda.is_initialized())
    return torch


def _version_numbers(value):
    matched = re.match(r"^([0-9]+)\.([0-9]+)\.([0-9]+)(?:[.+-].*|[a-z].*)?$", value)
    _need(matched is not None)
    return [int(part) for part in matched.groups()]


def _expected_causal_padding_mask(inputs_embeds, attention_mask, torch):
    batch, sequence, _ = inputs_embeds.shape
    expected = torch.full(
        (batch, 1, sequence, sequence),
        torch.finfo(inputs_embeds.dtype).min,
        dtype=inputs_embeds.dtype,
        device="cpu",
    )
    # Independent scalar oracle: padding suppresses KEYS, not whole query rows.
    for batch_index in range(batch):
        for query in range(sequence):
            for key in range(sequence):
                if key <= query and (attention_mask is None or attention_mask[batch_index, key]):
                    expected[batch_index, 0, query, key] = 0
    return expected


def _make_cpu_mask_adapter(original, torch):
    parameters = inspect.signature(original).parameters
    common = {
        "config",
        "inputs_embeds",
        "attention_mask",
        "past_key_values",
        "position_ids",
        "or_mask_function",
        "and_mask_function",
    }
    newer = {"block_sequence_ids", "layer_idx", "allow_is_causal_skip"}
    _need(set(parameters) in (common, common | {"cache_position"}, common | newer))
    if set(parameters) == common | newer:
        for name, expected in (
            ("block_sequence_ids", None),
            ("layer_idx", None),
            ("allow_is_causal_skip", True),
        ):
            default = parameters[name].default
            _need(type(default) is type(expected) and default == expected)
    _need(
        all(
            parameter.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            for parameter in parameters.values()
        )
    )

    def adapter(*, config, inputs_embeds, attention_mask, past_key_values, position_ids):
        _need(getattr(config, "_attn_implementation", None) == "eager")
        _need(getattr(config, "is_causal", True) is True and past_key_values is None)
        _need(isinstance(inputs_embeds, torch.Tensor) and inputs_embeds.device.type == "cpu")
        _need(inputs_embeds.dtype == torch.float32 and inputs_embeds.ndim == 3)
        batch, sequence, hidden = inputs_embeds.shape
        _need(1 <= batch <= 2 and 1 <= sequence <= 8 and hidden == 16)
        positions = torch.arange(sequence, dtype=torch.long, device="cpu")
        _need(isinstance(position_ids, torch.Tensor) and position_ids.device.type == "cpu")
        _need(position_ids.dtype == torch.long and position_ids.shape == (batch, sequence))
        _need(torch.equal(position_ids, positions.unsqueeze(0).expand(batch, -1)))
        if attention_mask is not None:
            _need(isinstance(attention_mask, torch.Tensor) and attention_mask.device.type == "cpu")
            _need(
                attention_mask.shape == (batch, sequence)
                and attention_mask.dtype in (torch.long, torch.bool)
            )
            _need(torch.all((attention_mask == 0) | (attention_mask == 1)).item())
            _need(torch.all(attention_mask.sum(dim=1) > 0).item())
        arguments = {
            "config": config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        if "cache_position" in parameters:
            arguments["cache_position"] = positions
        if "allow_is_causal_skip" in parameters:
            arguments["allow_is_causal_skip"] = False
        result = original(**arguments)
        expected = _expected_causal_padding_mask(inputs_embeds, attention_mask, torch)
        _need(isinstance(result, torch.Tensor) and result.device.type == "cpu")
        _need(result.dtype == expected.dtype and result.shape == expected.shape)
        _need(torch.equal(result, expected))
        return result

    return adapter


@contextmanager
def _cpu_causal_mask_adapter(torch):
    from transformers.masking_utils import create_causal_mask

    from lumi import modeling_lumi

    original = modeling_lumi.create_causal_mask
    _need(original is create_causal_mask)
    adapter = _make_cpu_mask_adapter(original, torch)
    modeling_lumi.create_causal_mask = adapter
    try:
        yield adapter
    finally:
        _need(modeling_lumi.create_causal_mask is adapter)
        modeling_lumi.create_causal_mask = original


def _mask_selfcheck(torch):
    from lumi.configuration_lumi import LumiConfig

    config = LumiConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=4096,
    )
    config._attn_implementation = "eager"
    inputs = torch.ones(2, 4, 16)
    positions = torch.arange(4, dtype=torch.long).unsqueeze(0).expand(2, -1)
    arguments = {
        "config": config,
        "inputs_embeds": inputs,
        "attention_mask": None,
        "past_key_values": None,
        "position_ids": positions,
    }
    observed = []

    def with_cache_position(
        config,
        inputs_embeds,
        attention_mask,
        cache_position,
        past_key_values,
        position_ids=None,
        or_mask_function=None,
        and_mask_function=None,
    ):
        _need(torch.equal(cache_position, torch.arange(4, dtype=torch.long)))
        observed.append("with_cache_position")
        return _expected_causal_padding_mask(inputs_embeds, attention_mask, torch)

    def without_cache_position(
        config,
        inputs_embeds,
        attention_mask,
        past_key_values,
        position_ids=None,
        or_mask_function=None,
        and_mask_function=None,
    ):
        observed.append("without_cache_position")
        return _expected_causal_padding_mask(inputs_embeds, attention_mask, torch)

    def with_explicit_mask_options(
        config,
        inputs_embeds,
        attention_mask,
        past_key_values,
        position_ids=None,
        or_mask_function=None,
        and_mask_function=None,
        block_sequence_ids=None,
        layer_idx=None,
        allow_is_causal_skip=True,
    ):
        _need(block_sequence_ids is None and layer_idx is None and allow_is_causal_skip is False)
        observed.append("with_explicit_mask_options")
        return _expected_causal_padding_mask(inputs_embeds, attention_mask, torch)

    for function in (with_cache_position, without_cache_position, with_explicit_mask_options):
        _make_cpu_mask_adapter(function, torch)(**arguments)
    _need(
        observed == ["with_cache_position", "without_cache_position", "with_explicit_mask_options"]
    )
    with _cpu_causal_mask_adapter(torch) as adapter:
        for padding in (
            None,
            torch.ones(2, 4, dtype=torch.long),
            torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]], dtype=torch.long),
        ):
            result = adapter(**{**arguments, "attention_mask": padding})
            _need(torch.equal(result, _expected_causal_padding_mask(inputs, padding, torch)))
        for changed in (
            {"past_key_values": object()},
            {"position_ids": positions + 1},
            {"position_ids": positions[:, :3]},
            {"inputs_embeds": torch.ones(2, 4)},
            {"inputs_embeds": torch.empty(2, 4, 16, device="meta")},
            {"attention_mask": torch.zeros(2, 4, dtype=torch.long)},
            {"attention_mask": torch.ones(2, 3, dtype=torch.long)},
            {"attention_mask": torch.full((2, 4), 0.5)},
            {"config": types.SimpleNamespace(_attn_implementation="sdpa", is_causal=True)},
        ):
            try:
                adapter(**{**arguments, **changed})
            except IntegrationError:
                continue
            raise IntegrationError()


def _private_write(path, payload):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            _need(written > 0)
            view = view[written:]
    finally:
        os.close(descriptor)


def _tree_bytes(root):
    total = 0
    for directory, names, files in os.walk(root, followlinks=False):
        for name in names:
            info = (Path(directory) / name).lstat()
            _need(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode))
        for name in files:
            info = (Path(directory) / name).lstat()
            _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1)
            total += info.st_size
    return total


def _postcheck_size(root, request):
    count = _tree_bytes(root)
    _need(count <= request["maximum_private_write_bytes"])
    return count


def _ticket(request, root, **fields):
    raw = _canonical(request)
    return {
        "request": request,
        "request_sha256": hashlib.sha256(raw).hexdigest(),
        "synthetic_root": str(root),
        **fields,
    }


def _spawn(ticket, *, new_session, stdout):
    raw = _canonical(ticket)
    _need(len(raw) <= MAX_CONTROL_BYTES)
    reader, writer = os.pipe()
    process = None
    try:
        # One nonblocking prefill. Capacity/short-write failures reject, no retry.
        os.set_blocking(writer, False)
        _need(os.write(writer, raw) == len(raw))
        closing, writer = writer, None
        os.close(closing)
        command = [
            sys.executable,
            "-S",
            "-B",
            "-c",
            CHILD_BOOTSTRAP,
            str(Path(__file__).resolve()),
            str(reader),
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=subprocess.DEVNULL,
            env=_environment(Path(ticket["synthetic_root"])),
            cwd=_repository(),
            pass_fds=(reader,),
            start_new_session=new_session,
        )
    finally:
        failed = False
        for closing in (reader, writer):
            if closing is not None:
                try:
                    os.close(closing)
                except OSError:
                    failed = True
        if failed:
            if process is not None:
                if new_session:
                    _stop_owned_session(process)
                else:
                    process.kill()
                    process.wait()
            raise IntegrationError() from None
    return process


def _stop_owned_session(process):
    """Only the session/process group created by our runner spawn is targeted."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    # Keep the leader unreaped until both signals: its PID still owns the group
    # identity even if it has already exited. No communicate/poll/wait beforehand.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def _read_runner_report(process, deadline, *, cancellation=None):
    """Bounded pipe drain without polling or reaping the session leader."""
    chunks = []
    count = 0
    while True:
        _need(cancellation is None or not cancellation["cancelled"])
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("synthetic_cpu_deadline_exhausted")
        readable, _, _ = select.select([process.stdout], [], [], min(remaining, 0.1))
        if not readable:
            continue
        chunk = os.read(process.stdout.fileno(), min(4096, MAX_REPORT_BYTES + 1 - count))
        if not chunk:
            break
        chunks.append(chunk)
        count += len(chunk)
        _need(count <= MAX_REPORT_BYTES)
    return b"".join(chunks)


@contextmanager
def _catchable_signal_cleanup():
    previous = {}
    cancellation = {"cancelled": False}

    def interrupted(_signal, _frame):
        # Never raise from a handler: even a first late or repeated signal must
        # not interrupt owned-group cleanup. The controller observes this latch.
        cancellation["cancelled"] = True

    try:
        for number in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            previous[number] = signal.signal(number, interrupted)
        yield cancellation
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _validate_report(report):
    fixed = {
        "schema_version": SCHEMA,
        "status": "PASS_synthetic_cpu_ddp_fresh_resume",
        "synthetic_only": True,
        "production_eligible": False,
        "private_or_gpu_execution_enabled": False,
        "cuda_initialized": False,
        "inference_calls": 0,
        "world_size": 2,
        "sequential_invocations": 3,
        "rank_processes": 6,
        "global_optimizer_steps": 4,
        "rank_optimizer_steps": 8,
        "unique_selected_records": 4,
        "admission_before_ml": True,
        "maximum_private_write_bytes": 67_108_864,
        "hard_aggregate_write_enforcement": False,
        "memory_enforcement": False,
        "external_network_metering": False,
        "parent_death_guardian": False,
        "model_changed": True,
        "exact_model": True,
        "exact_optimizer": True,
        "exact_rng": True,
        "exact_stable_state": True,
        "exact_stable_metrics": True,
        "checkpoints_verified": 4,
        "optimizer_rng_pairs_verified": 4,
        "finite_loss_and_gradient": True,
        "supervised_token_accounting_verified": True,
        "causal_padding_mask_oracle_verified": True,
        "causal_mask_adapter": "synthetic_cpu_eager_no_cache_v0.1",
    }
    _need(
        type(report) is dict
        and set(report)
        == set(fixed)
        | {
            "source_parameters",
            "checked_tree_bytes_high_water",
            "runtime_torch_version",
            "runtime_transformers_version",
        }
    )
    for key, value in fixed.items():
        _need(type(report[key]) is type(value) and report[key] == value)
    _need(type(report["source_parameters"]) is int and 0 < report["source_parameters"] < 20_000)
    _need(
        type(report["checked_tree_bytes_high_water"]) is int
        and 0 < report["checked_tree_bytes_high_water"] <= 67_108_864
    )
    versions = {}
    for key, lower, upper in (
        ("runtime_torch_version", (2, 5), (3, 0)),
        ("runtime_transformers_version", (5, 2), (6, 0)),
    ):
        version = report[key]
        _need(
            type(version) is list
            and len(version) == 3
            and all(type(part) is int and 0 <= part <= 999 for part in version)
        )
        _need(lower <= tuple(version[:2]) < upper)
        versions[key] = list(version)
    return {
        **fixed,
        **versions,
        "source_parameters": report["source_parameters"],
        "checked_tree_bytes_high_water": report["checked_tree_bytes_high_water"],
    }


def run_integration(request_raw, request_sha256, *, repository_root):
    """Admit, supervise while alive, and accept only an audited completion receipt.

    Rank exit codes are checked inside the coordinator. The coordinator's own
    natural exit is not required: after its final report/EOF we terminate its
    still-owned process group before reaping its leader.
    """
    request = _admit(request_raw, request_sha256)
    _need(signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL)
    _need(Path(repository_root).resolve() == _repository())
    started = time.monotonic()
    process = None
    with (
        _catchable_signal_cleanup() as cancellation,
        tempfile.TemporaryDirectory(
            prefix="lumi-SYNTHETIC-CPU-v1-",
        ) as temporary,
    ):
        root = Path(temporary).resolve()
        root.chmod(0o700)
        _private_write(root / MARKER, _canonical(request))
        try:
            _need(not cancellation["cancelled"])
            process = _spawn(
                _ticket(request, root, role="coordinator"),
                new_session=True,
                stdout=subprocess.PIPE,
            )
            raw = _read_runner_report(
                process,
                started + request["maximum_runtime_seconds"],
                cancellation=cancellation,
            )
            _need(0 < len(raw) <= MAX_REPORT_BYTES)
            report = _validate_report(_strict_json(raw))
            _postcheck_size(root, request)
        finally:
            if process is not None:
                _stop_owned_session(process)
                process.stdout.close()
    _need(not cancellation["cancelled"])
    report["temporary_cleanup_completed"] = True
    report["wall_watchdog_seconds"] = request["maximum_runtime_seconds"]
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return report


def _read_control(descriptor):
    _need(type(descriptor) is int and descriptor >= 3)
    try:
        info = os.fstat(descriptor)
        _need(stat.S_ISFIFO(info.st_mode))
        chunks = []
        count = 0
        while True:
            chunk = os.read(descriptor, min(4096, MAX_CONTROL_BYTES + 1 - count))
            if not chunk:
                break
            chunks.append(chunk)
            count += len(chunk)
            _need(count <= MAX_CONTROL_BYTES)
        return _strict_json(b"".join(chunks))
    finally:
        os.close(descriptor)


def _validated_control(control):
    _need(type(control) is dict)
    request = control.get("request")
    _admit(_canonical(request), control.get("request_sha256"))
    common = {"request", "request_sha256", "synthetic_root", "role"}
    role = control.get("role")
    expected = common if role == "coordinator" else common | {"rank", "stage", "bindings"}
    _need(role in {"coordinator", "worker"} and set(control) == expected)
    root = Path(control["synthetic_root"])
    _need(root.is_absolute() and root == root.resolve())
    _need(root.name.startswith("lumi-SYNTHETIC-CPU-v1-"))
    _need(root.parent == Path(tempfile.gettempdir()).resolve())
    info = root.lstat()
    _need(stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o700)
    _need(info.st_uid == os.geteuid())
    marker = root / MARKER
    info = marker.lstat()
    _need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600)
    _need(info.st_nlink == 1 and info.st_size < MAX_CONTROL_BYTES)
    _need(marker.read_bytes() == _canonical(request))
    if role == "worker":
        _need(type(control["rank"]) is int and control["rank"] in {0, 1})
        _need(type(control["stage"]) is int and control["stage"] in {0, 1, 2})
        bindings = control["bindings"]
        _need(
            type(bindings) is dict
            and set(bindings)
            == {
                "inventory",
                "manifest_file",
                "manifest_internal",
                "metadata",
                "seal",
                "source",
            }
        )
        _need(
            all(
                type(value) is str and len(value) == 64 and set(value) <= set("0123456789abcdef")
                for value in bindings.values()
            )
        )
    return request, root


def _child_main(descriptor):
    report_descriptor = os.dup(1)
    null_descriptor = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null_descriptor, 1)
    os.dup2(null_descriptor, 2)
    os.close(null_descriptor)
    try:
        control = _read_control(descriptor)
        request, root = _validated_control(control)
        os.umask(0o077)
        torch = _initialize_ml(root)
        if control["role"] == "coordinator":
            report = _coordinator(request, root, torch)
            raw = _canonical(report)
            _need(len(raw) <= MAX_REPORT_BYTES)
            _need(os.write(report_descriptor, raw) == len(raw))
        else:
            _worker(control, root, torch)
    except BaseException:
        raise SystemExit(1) from None
    finally:
        os.close(report_descriptor)


def _build_fixture(root, torch):
    """Fixed token integers and a random tiny Lumi, never imported private data."""
    from lumi import posttraining_sft_runtime_v4 as runtime
    from lumi.configuration_lumi import LumiConfig
    from lumi.modeling_lumi import LumiForCausalLM

    encoded = root / "encoded"
    encoded.mkdir(mode=0o700)
    specs = (
        ("train", tuple((10 + ordinal * 3, f"{ordinal + 1:064x}") for ordinal in range(5))),
        ("validation", ((40, "f" * 64),)),
    )
    shards = []
    split_records, split_tokens, split_supervised = Counter(), Counter(), Counter()
    encoded_hasher, sequence_hasher = hashlib.sha256(), hashlib.sha256()
    for split, records in specs:
        prefix = f"{split}-00000"
        token_values, label_values = [], []
        index_payload = bytearray()
        for ordinal, (base, record_hash) in enumerate(records):
            tokens, labels = (base, base + 1, base + 2), (-100, base + 1, base + 2)
            entry = {
                "offset_tokens": len(token_values),
                "record_id_sha256": record_hash,
                "record_ordinal": ordinal,
                "supervised_tokens": 2,
                "tokens": 3,
            }
            index_payload.extend(_canonical(entry) + b"\n")
            payload = runtime._encoded_hash_payload(record_hash, tokens, labels)
            encoded_hasher.update(len(payload).to_bytes(8, "big"))
            encoded_hasher.update(payload)
            sequence_hasher.update(f"{split}\0{record_hash}\n".encode())
            token_values.extend(tokens)
            label_values.extend(labels)
        payloads = {
            f"{prefix}.tokens.u32le": struct.pack(f"<{len(token_values)}I", *token_values),
            f"{prefix}.labels.i32le": struct.pack(f"<{len(label_values)}i", *label_values),
            f"{prefix}.index.jsonl": bytes(index_payload),
        }
        files = []
        for name, payload in payloads.items():
            _private_write(encoded / name, payload)
            files.append(
                {"bytes": len(payload), "path": name, "sha256": hashlib.sha256(payload).hexdigest()}
            )
        shards.append(
            {
                "files": files,
                "index": 0,
                "records": len(records),
                "split": split,
                "supervised_tokens": len(records) * 2,
                "tokens": len(token_values),
            }
        )
        split_records[split] = len(records)
        split_tokens[split] = len(token_values)
        split_supervised[split] = len(records) * 2
    manifest = {
        "authorization_boundary": runtime._TOKEN_CONSISTENT_AUTHORIZATION_BOUNDARY,
        "context_window": 4096,
        "encoded_dataset_sha256": encoded_hasher.hexdigest(),
        "format": runtime._TOKEN_CONSISTENT_SHARD_FORMAT,
        "maximum_tokens_per_shard": 4_194_304,
        "production_eligible": False,
        "record_hash_sequence_sha256": sequence_hasher.hexdigest(),
        "records": sum(split_records.values()),
        "schema_version": "lumi.sft-encoded-shards.v0.1",
        "seal_sha256": "c" * 64,
        "shards": shards,
        "splits": {
            split: {
                "records": split_records[split],
                "tokens": split_tokens[split],
                "supervised_tokens": split_supervised[split],
            }
            for split in sorted(split_records)
        },
        "supervised_tokens": sum(split_supervised.values()),
        "tokens": sum(split_tokens.values()),
    }
    manifest["manifest_sha256"] = _digest(manifest)
    manifest_payload = _canonical(manifest) + b"\n"
    _private_write(encoded / "manifest.json", manifest_payload)
    metadata = []
    for path in sorted(encoded.iterdir()):
        info = path.lstat()
        metadata.append(
            {
                "mode": f"{stat.S_IMODE(info.st_mode):04o}",
                "relative_path": path.name,
                "st_ctime_ns": info.st_ctime_ns,
                "st_dev": info.st_dev,
                "st_ino": info.st_ino,
                "st_mtime_ns": info.st_mtime_ns,
                "st_nlink": info.st_nlink,
                "st_size": info.st_size,
                "st_uid": info.st_uid,
            }
        )
    source = root / "source"
    source.mkdir(mode=0o700)
    torch.manual_seed(17)
    model = LumiForCausalLM(
        LumiConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=4096,
        )
    )
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    parameters = sum(parameter.numel() for parameter in model.parameters())
    _need(parameters < 20_000)
    model.save_pretrained(source, safe_serialization=True)
    entries = []
    for path in sorted(source.iterdir()):
        _need(path.is_file() and not path.is_symlink())
        path.chmod(0o600)
        entries.append((hashlib.sha256(path.read_bytes()).hexdigest(), path.name))
    source_manifest = "".join(f"{digest}  {name}\n" for digest, name in entries).encode()
    _private_write(source / "manifest.sha256", source_manifest)
    return (
        {
            "inventory": _digest(shards),
            "manifest_file": hashlib.sha256(manifest_payload).hexdigest(),
            "manifest_internal": manifest["manifest_sha256"],
            "metadata": _digest(metadata),
            "seal": manifest["seal_sha256"],
            "source": hashlib.sha256(source_manifest).hexdigest(),
        },
        original,
        parameters,
    )


def _training_arguments(root, bindings, stage):
    name, stop_step, resume = SCHEDULE[stage]
    return {
        "encoded_shards": root / "encoded",
        "source_checkpoint": root / "source",
        "source_checkpoint_manifest_sha256": bindings["source"],
        "encoded_shards_inventory_sha256": bindings["inventory"],
        "encoded_shards_manifest_file_sha256": bindings["manifest_file"],
        "encoded_shards_manifest_sha256": bindings["manifest_internal"],
        "encoded_shards_metadata_inventory_sha256": bindings["metadata"],
        "encoded_shards_seal_sha256": bindings["seal"],
        "encoded_shards_record_limit": 4,
        "output": root / name,
        "target_steps": 2,
        "stop_after_step": stop_step,
        "strategy": "ddp",
        "per_rank_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "checkpoint_steps": 1,
        "keep_checkpoints": None,
        "seed": 2027,
        "resume": resume,
        "lr_schedule": "cosine",
        "warmup_steps": 0,
        "min_learning_rate_ratio": 0.1,
        "attention_implementation": "eager",
        "optimizer_profile": "grouped_adamw_beta2_0.95_v0.1",
        "precision": "fp32",
        "gradient_sync_mode": "last_microbatch",
        "pad_to_multiple_of": 8,
        "source_commit": None,
        "authorization_path": None,
    }


def _worker(control, root, torch):
    from lumi.distributed_training import DistributedContext, destroy_distributed
    from lumi.posttraining_distributed_sft_v2 import train_distributed_sft

    rank, stage = control["rank"], control["stage"]
    arguments = _training_arguments(root, control["bindings"], stage)
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{root / ('rendezvous-' + str(stage))}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        with _cpu_causal_mask_adapter(torch):
            state = train_distributed_sft(
                context=DistributedContext(
                    rank=rank,
                    local_rank=rank,
                    world_size=2,
                    backend="gloo",
                    device=torch.device("cpu"),
                ),
                **arguments,
            )
        _need(state["step"] == SCHEDULE[stage][1])
        _need(not torch.cuda.is_initialized())
    finally:
        destroy_distributed()


def _launch_stage(request, root, bindings, stage):
    processes = []
    try:
        for rank in range(2):
            processes.append(
                _spawn(
                    _ticket(
                        request, root, role="worker", rank=rank, stage=stage, bindings=bindings
                    ),
                    new_session=False,
                    stdout=subprocess.DEVNULL,
                )
            )
        # The surviving outer controller owns the timeout and bounded teardown.
        for process in processes:
            _need(process.wait() == 0)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
        for process in processes:
            process.wait()


def _nested_exact(left, right, torch):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and set(left) == set(right)
            and all(_nested_exact(left[key], right[key], torch) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_nested_exact(a, b, torch) for a, b in zip(left, right, strict=True))
        )
    return type(left) is type(right) and left == right


def _read_state(root):
    return _strict_json((root / "distributed-training-state.json").read_bytes())


def _stable(value, ignored):
    return {key: child for key, child in value.items() if key not in ignored}


def _metrics(root):
    return [
        _strict_json(line)
        for line in (root / "distributed-metrics.jsonl").read_bytes().splitlines()
    ]


def _checkpoint_model(checkpoint, root, torch):
    import torch.distributed.checkpoint as dcp

    from lumi.configuration_lumi import LumiConfig
    from lumi.modeling_lumi import LumiForCausalLM

    config = LumiConfig(**_strict_json((root / "source" / "config.json").read_bytes()))
    model = LumiForCausalLM(config)
    state = model.state_dict()
    dcp.load({"model": state}, checkpoint_id=checkpoint / "model", no_dist=True)
    incompatible = model.load_state_dict(state)
    _need(not incompatible.missing_keys and not incompatible.unexpected_keys)
    _need(all(torch.isfinite(value).all().item() for value in state.values()))
    return state


def _audit(root, original, torch):
    from lumi.posttraining_distributed_sft_v2 import verify_distributed_sft_checkpoint

    reference, resumed = root / "reference", root / "resumed"
    left_state, right_state = _read_state(reference), _read_state(resumed)
    _need(
        _nested_exact(_stable(left_state, STATE_TIMING), _stable(right_state, STATE_TIMING), torch)
    )
    _need(left_state["status"] == "complete" and left_state["step"] == 2)
    _need(left_state["records_seen"] == 4 and left_state["tokens_seen"] == 12)
    _need(left_state["supervised_tokens_seen"] == 8)
    audit = left_state["run_specification"]["data"]["read_audit"]
    _need(audit["records_loaded"] == 4 and audit["fully_hashed_shards"] == 0)
    _need(audit["encoded_index_row_reads"] == 4 and audit["binary_bytes_read"] == 96)
    left_metrics, right_metrics = _metrics(reference), _metrics(resumed)
    _need(len(left_metrics) == len(right_metrics) == 2)
    _need(
        [_stable(row, METRIC_TIMING) for row in left_metrics]
        == [_stable(row, METRIC_TIMING) for row in right_metrics]
    )
    for step, row in enumerate(left_metrics, 1):
        _need(row["step"] == step and row["supervised_tokens"] == 4 and row["tokens"] == 6)
        _need(row["device"] == "cpu" and row["backend"] == "gloo" and row["strategy"] == "ddp")
        _need(math.isfinite(row["loss"]) and row["loss"] > 0)
        _need(math.isfinite(row["gradient_norm"]) and row["gradient_norm"] > 0)
    final_model = None
    for step in (1, 2):
        left_checkpoint = reference / f"checkpoint-{step:08d}"
        right_checkpoint = resumed / f"checkpoint-{step:08d}"
        verify_distributed_sft_checkpoint(left_checkpoint)
        verify_distributed_sft_checkpoint(right_checkpoint)
        left_model = _checkpoint_model(left_checkpoint, root, torch)
        right_model = _checkpoint_model(right_checkpoint, root, torch)
        _need(_nested_exact(left_model, right_model, torch))
        final_model = left_model
        for rank in (0, 1):
            filename = f"trainer-rank-{rank:05d}.pt"
            left = torch.load(left_checkpoint / filename, map_location="cpu", weights_only=True)
            right = torch.load(right_checkpoint / filename, map_location="cpu", weights_only=True)
            _need(set(left) == set(right) == {"optimizer", "rng_state"})
            _need(_nested_exact(left["optimizer"], right["optimizer"], torch))
            _need(_nested_exact(left["rng_state"], right["rng_state"], torch))
    _need(not _nested_exact(original, final_model, torch))
    _need(
        not torch.equal(
            original["model.embed_tokens.weight"], final_model["model.embed_tokens.weight"]
        )
    )
    _need(not torch.cuda.is_initialized())
    return {
        "model_changed": True,
        "exact_model": True,
        "exact_optimizer": True,
        "exact_rng": True,
        "exact_stable_state": True,
        "exact_stable_metrics": True,
        "checkpoints_verified": 4,
        "optimizer_rng_pairs_verified": 4,
        "finite_loss_and_gradient": True,
        "supervised_token_accounting_verified": True,
    }


def _coordinator(request, root, torch):
    import transformers

    _mask_selfcheck(torch)
    bindings, original, parameters = _build_fixture(root, torch)
    high_water = _postcheck_size(root, request)
    for stage in range(3):
        _launch_stage(request, root, bindings, stage)
        high_water = max(high_water, _postcheck_size(root, request))
        state = _read_state(root / SCHEDULE[stage][0])
        _need(state["step"] == SCHEDULE[stage][1])
        _need(state["status"] == ("paused" if stage == 1 else "complete"))
    audit = _audit(root, original, torch)
    high_water = max(high_water, _postcheck_size(root, request))
    return {
        "schema_version": SCHEMA,
        "status": "PASS_synthetic_cpu_ddp_fresh_resume",
        "synthetic_only": True,
        "production_eligible": False,
        "private_or_gpu_execution_enabled": False,
        "cuda_initialized": False,
        "inference_calls": 0,
        "world_size": 2,
        "sequential_invocations": 3,
        "rank_processes": 6,
        "global_optimizer_steps": 4,
        "rank_optimizer_steps": 8,
        "unique_selected_records": 4,
        "source_parameters": parameters,
        "admission_before_ml": True,
        "checked_tree_bytes_high_water": high_water,
        "maximum_private_write_bytes": request["maximum_private_write_bytes"],
        "hard_aggregate_write_enforcement": False,
        "memory_enforcement": False,
        "external_network_metering": False,
        "parent_death_guardian": False,
        "causal_padding_mask_oracle_verified": True,
        "causal_mask_adapter": "synthetic_cpu_eager_no_cache_v0.1",
        "runtime_torch_version": _version_numbers(torch.__version__),
        "runtime_transformers_version": _version_numbers(transformers.__version__),
        **audit,
    }
