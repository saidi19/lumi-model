"""Tiny stdlib guard/callback tests; no ML, data, or successful training emulation.

Clients use owned local socketpairs without external network access. Private
running/retired states below isolate callback guards, not successful run paths.
Only the isolated synthetic package's client methods are patched where needed.
"""

from __future__ import annotations

import importlib
import json
import socket
import sys
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_synthetic_sft_worker_run_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
before_import = set(sys.modules)
runs = importlib.import_module(PACKAGE + ".posttraining_sft_worker_run_v1")
IMPORT_ADDITIONS = set(sys.modules) - before_import


def profile(**changes):
    return {
        "per_rank_batch_size": 1, "gradient_accumulation_steps": 2, "context_window": 8,
        "pad_to_multiple_of": 2, "pad_token_id": 0, "seed": 17, "learning_rate": 0.01,
        "lr_schedule": "cosine", "warmup_steps": 0, "min_learning_rate_ratio": 0.1,
        "max_grad_norm": 1.0, "strategy": "ddp", "gradient_sync_mode": "last_microbatch",
        "precision": "fp32", **changes,
    }


class WorkerRunGuardTests(unittest.TestCase):
    def setUp(self):
        self.clients, self.sockets, self.events = [], [], []

    def tearDown(self):
        for client in self.clients:
            client.close()
        for sock in self.sockets:
            sock.close()

    def client(self, rank=0):
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sockets.extend((left, right))
        client = runs.WorkerLedgerClient(
            left, epoch="a" * 64, rank=rank, token="b" * 64,
            interrupt=lambda: self.events.append("client_interrupt"), maximum_wire_bytes=65_536,
            deadline_ns=10_000, maximum_exchange_ns=1000, clock_ns=lambda: 1,
        )
        self.clients.append(client)
        return client

    def options(self, **changes):
        options = {
            "client": self.client(), "rank": 0, "world_size": 2, "from_step": 0,
            "stop_step": 2, "profile": profile(), "record_limit": 4,
            "loader_bindings": dict.fromkeys(runs.PIN_FIELDS, "c" * 64),
            "source_identity_sha256": "d" * 64,
            "checkpoint_limits": dict.fromkeys(runs.LIMIT_KEYS, 1024),
            "admit": lambda: self.events.append("admit"),
            "synchronize": lambda phase, step: self.events.append((phase, step)),
            "interrupt": lambda: self.events.append("interrupt"), "device_type": "cpu",
        }
        options.update(changes)
        return options

    def worker(self, **changes):
        return runs.WorkerRun(**self.options(**changes))

    def error(self, code, operation, *args, **kwargs):
        with self.assertRaises(runs.WorkerRunError) as caught:
            operation(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(caught.exception.args, (code,))
        self.assertTrue(caught.exception.__suppress_context__)
        self.assertNotIn("SYNTHETIC-PRIVATE", repr(caught.exception))

    def run_sentinel(self, worker, **changes):
        return worker.run(object(), object(), encoded_root=object(), output_parent_fd=object(),
                          **changes)

    def block_after_admission(self):
        """Fail loudly if a denied guard reaches FD, helper, runtime or input work."""
        stack = ExitStack()
        targets = (
            (runs, "_directory_info"), (runs.WorkerRun, "_helpers"),
            (runs.WorkerRun, "_runtime"), (runs.WorkerInput, "load"),
            (runs.CheckpointWorker, "restore_checkpoint"),
            (runs.CheckpointWorker, "save_checkpoint"), (runs, "read_training_state"),
        )
        mocks = [stack.enter_context(patch.object(
            owner, name, side_effect=AssertionError("SYNTHETIC-PRIVATE-UNREACHABLE"),
        )) for owner, name in targets]
        return stack, mocks

    def test_imports_are_stdlib_only(self):
        forbidden = {"torch", "numpy", "transformers", "tokenizers", "safetensors"}
        self.assertFalse(any(name.split(".")[0] in forbidden for name in IMPORT_ADDITIONS))

    def test_constructor_is_inert_and_safe_summary_omits_private_values(self):
        worker = self.worker()
        self.assertEqual(self.events, [])
        self.assertEqual(worker._client.summary()["requests_sent"], 0)
        self.assertEqual(worker._client.summary()["wire_bytes_sent"], 0)
        summary = worker.summary()
        self.assertEqual(summary["state"], "idle")
        self.assertEqual(summary["completed_steps"], 0)
        self.assertIs(summary["real_sft_execution_permitted"], False)
        self.assertIs(summary["production_eligible"], False)
        self.assertNotIn(worker._identity, json.dumps(summary))
        self.assertNotIn("c" * 64, json.dumps(summary))
        self.assertEqual(repr(worker), "<WorkerRun admitted objects only>")

    def test_client_requires_exact_v2_type_and_matching_rank(self):
        self.error("invalid_arguments", runs.WorkerRun, **self.options(client=object()))
        self.error("invalid_arguments", runs.WorkerRun, **self.options(rank=1))

        class ClientSubclass(runs.WorkerLedgerClient):
            pass

        sentinel = object.__new__(ClientSubclass)
        self.error("invalid_arguments", runs.WorkerRun, **self.options(client=sentinel))

    def test_integer_constructor_bounds_are_exact(self):
        invalid = {
            "rank": (True, -1, 2, 0.0), "world_size": (True, 1, 13, 2.0),
            "from_step": (True, -1, 2, 0.0), "stop_step": (True, 0, 3, 2.0),
            "record_limit": (True, 1, 13, 4.0),
        }
        base = self.options()
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, kind=type(value).__name__):
                    self.error("invalid_arguments", runs.WorkerRun, **{**base, field: value})
        self.error("invalid_arguments", runs.WorkerRun, **{**base, "from_step": 1, "stop_step": 1})
        self.assertEqual(self.events, [])

    def test_constructor_requires_callable_capabilities_and_device_enum(self):
        base = self.options()
        for field in ("admit", "synchronize", "interrupt"):
            self.error("invalid_arguments", runs.WorkerRun, **{**base, field: None})
        for value in (None, True, "mps", "CPU"):
            self.error("invalid_arguments", runs.WorkerRun, **{**base, "device_type": value})

    def test_profile_has_exact_fields_and_builtin_mapping_type(self):
        base = self.options()
        for field in profile():
            changed = profile()
            del changed[field]
            self.error("invalid_arguments", runs.WorkerRun, **{**base, "profile": changed})
        self.error("invalid_arguments", runs.WorkerRun,
                   **{**base, "profile": profile(private_text="SYNTHETIC-PRIVATE")})

        class Hostile(dict):
            def __iter__(self):
                raise AssertionError("SYNTHETIC-PRIVATE")

        self.error("invalid_arguments", runs.WorkerRun, **{**base, "profile": Hostile(profile())})
        self.error("invalid_arguments", runs.WorkerRun, **{**base, "profile": {**profile(), 1: 2}})

    def test_profile_integer_boundaries_and_geometry(self):
        base = self.options()
        invalid = {
            "per_rank_batch_size": (0, 13, True, 1.0, 3),
            "gradient_accumulation_steps": (0, 13, False, 1.0),
            "context_window": (1, 4096, True, 8.0),
            "pad_to_multiple_of": (0, 4096, 3, 16, True),
            "pad_token_id": (-1, 2**32, True), "seed": (-1, 2**31, True),
            "warmup_steps": (-1, 2, True),
        }
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    self.error("invalid_arguments", runs.WorkerRun,
                               **{**base, "profile": profile(**{field: value})})

    def test_profile_rejects_nonfinite_negative_boolean_and_huge_numbers(self):
        base = self.options()
        for field in ("learning_rate", "max_grad_norm", "min_learning_rate_ratio"):
            for value in (float("nan"), float("inf"), -1, True, "1", 10 ** 1000):
                self.error("invalid_arguments", runs.WorkerRun,
                           **{**base, "profile": profile(**{field: value})})
        for field in ("learning_rate", "max_grad_norm"):
            self.error("invalid_arguments", runs.WorkerRun,
                       **{**base, "profile": profile(**{field: 0})})
        self.error("invalid_arguments", runs.WorkerRun,
                   **{**base, "profile": profile(min_learning_rate_ratio=0)})

    def test_profile_enums_and_no_implicit_device_strategy_fallback(self):
        base = self.options()
        for field in ("lr_schedule", "strategy", "gradient_sync_mode", "precision"):
            for value in (None, True, "SYNTHETIC-PRIVATE"):
                self.error("invalid_arguments", runs.WorkerRun,
                           **{**base, "profile": profile(**{field: value})})
        self.error("invalid_arguments", runs.WorkerRun,
                   **{**base, "profile": profile(strategy="fsdp_full_shard")})
        self.error("invalid_arguments", runs.WorkerRun, **{
            **base, "device_type": "cuda",
            "profile": profile(precision="bf16", strategy="fsdp_full_shard"),
        })
        self.error("invalid_arguments", runs.WorkerRun, **{**base, "device_type": "cuda"})

    def test_hash_pins_are_exact_and_all_five_are_required(self):
        base = self.options()
        for field in runs.PIN_FIELDS:
            pins = dict(base["loader_bindings"])
            del pins[field]
            self.error("invalid_arguments", runs.WorkerRun, **{**base, "loader_bindings": pins})
            for value in ("A" * 64, "g" * 64, "c" * 63, True, None):
                self.error("invalid_arguments", runs.WorkerRun,
                           **{**base, "loader_bindings": {**base["loader_bindings"], field: value}})
        for value in (None, True, "D" * 64, "d" * 63):
            self.error("invalid_arguments", runs.WorkerRun,
                       **{**base, "source_identity_sha256": value})

    def test_checkpoint_limits_are_exact_positive_bounded_integers(self):
        base = self.options()
        for field in runs.LIMIT_KEYS:
            limits = dict(base["checkpoint_limits"])
            del limits[field]
            self.error("invalid_arguments", runs.WorkerRun,
                       **{**base, "checkpoint_limits": limits})
            for value in (0, -1, True, 1.0, runs.MAX_FILE_BYTES + 1):
                self.error("invalid_arguments", runs.WorkerRun, **{
                    **base, "checkpoint_limits": {**base["checkpoint_limits"], field: value},
                })

    def test_identity_is_shared_across_rank_and_fresh_invocation_boundaries(self):
        base = self.options()
        uninterrupted = runs.WorkerRun(**base)
        rank_one = runs.WorkerRun(**{**base, "client": self.client(1), "rank": 1})
        initial = runs.WorkerRun(**{**base, "stop_step": 1})
        resume = runs.WorkerRun(**{**base, "from_step": 1, "stop_step": 2})
        self.assertEqual({worker._identity for worker in
                          (uninterrupted, rank_one, initial, resume)}, {uninterrupted._identity})

    def test_identity_binds_source_every_profile_field_and_every_pin(self):
        base = self.options()
        original = runs.WorkerRun(**base)._identity
        changes = {
            "per_rank_batch_size": 2, "gradient_accumulation_steps": 1, "context_window": 16,
            "pad_to_multiple_of": 1, "pad_token_id": 1, "seed": 18, "learning_rate": 0.02,
            "lr_schedule": "constant", "warmup_steps": 1, "min_learning_rate_ratio": 0.2,
            "max_grad_norm": 2.0, "gradient_sync_mode": "every_microbatch", "precision": "bf16",
        }
        for field, value in changes.items():
            worker = runs.WorkerRun(**{**base, "profile": profile(**{field: value})})
            self.assertNotEqual(worker._identity, original)
        for field in runs.PIN_FIELDS:
            worker = runs.WorkerRun(**{
                **base, "loader_bindings": {**base["loader_bindings"], field: "e" * 64},
            })
            self.assertNotEqual(worker._identity, original)
        changed_source = runs.WorkerRun(**{**base, "source_identity_sha256": "e" * 64})
        self.assertNotEqual(changed_source._identity, original)
        cuda = {**base, "device_type": "cuda", "profile": profile(precision="bf16")}
        self.assertNotEqual(runs.WorkerRun(**cuda)._identity, original)

    def test_identity_binds_global_geometry_but_not_resource_caps(self):
        base = self.options()
        original = runs.WorkerRun(**base)._identity
        self.assertNotEqual(runs.WorkerRun(**{**base, "world_size": 3})._identity, original)
        self.assertNotEqual(runs.WorkerRun(**{**base, "record_limit": 6})._identity, original)
        limited = {**base, "checkpoint_limits": dict.fromkeys(runs.LIMIT_KEYS, 2048)}
        self.assertEqual(runs.WorkerRun(**limited)._identity, original)

    def test_profile_numbers_canonicalize_and_inputs_are_defensively_copied(self):
        base = self.options(profile=profile(learning_rate=1, max_grad_norm=2))
        worker = runs.WorkerRun(**base)
        identity = worker._identity
        for field in ("learning_rate", "max_grad_norm", "min_learning_rate_ratio"):
            self.assertIs(type(worker._profile[field]), float)
        base["profile"]["seed"] = 999
        base["loader_bindings"][runs.PIN_FIELDS[0]] = "f" * 64
        base["checkpoint_limits"]["maximum_file_bytes"] = 5
        self.assertEqual(worker._profile["seed"], 17)
        self.assertEqual(worker._pins[runs.PIN_FIELDS[0]], "c" * 64)
        self.assertEqual(worker._limits["maximum_file_bytes"], 1024)
        self.assertEqual(worker._identity, identity)
        for mapping in (worker._profile, worker._pins, worker._limits):
            with self.assertRaises(TypeError):
                mapping[next(iter(mapping))] = None

    def test_admission_denial_precedes_fd_helpers_runtime_and_inputs(self):
        def deny():
            self.events.append("deny")
            raise RuntimeError("SYNTHETIC-PRIVATE")

        worker = self.worker(admit=deny)
        stack, mocks = self.block_after_admission()
        with stack:
            self.error("admission_rejected", self.run_sentinel, worker)
        for operation in mocks:
            operation.assert_not_called()
        self.assertEqual(self.events, ["deny", "interrupt"])
        self.assertEqual(worker._failure_stage, "admission")
        self.assertFalse(worker._client.closed)

    def test_non_none_admission_return_is_a_denial(self):
        for value in (True, False, 0, "SYNTHETIC-PRIVATE"):
            worker = self.worker(admit=lambda result=value: result)
            stack, mocks = self.block_after_admission()
            with stack:
                self.error("admission_rejected", self.run_sentinel, worker)
            for operation in mocks:
                operation.assert_not_called()

    def test_resume_arguments_are_validated_before_admission(self):
        cases = (
            ({}, {"resume_checkpoint_name": "checkpoint-00000001"}),
            ({}, {"resume_manifest_sha256": "e" * 64}),
            ({"from_step": 1}, {}),
            ({"from_step": 1}, {"resume_checkpoint_name": "checkpoint-00000002",
                                "resume_manifest_sha256": "e" * 64}),
            ({"from_step": 1}, {"resume_checkpoint_name": "../checkpoint-00000001",
                                "resume_manifest_sha256": "e" * 64}),
            ({"from_step": 1}, {"resume_checkpoint_name": "checkpoint-00000001",
                                "resume_manifest_sha256": "E" * 64}),
        )
        for options, resume in cases:
            worker = self.worker(**options)
            stack, mocks = self.block_after_admission()
            with stack:
                self.error("invalid_arguments", self.run_sentinel, worker, **resume)
            for operation in mocks:
                operation.assert_not_called()
        self.assertNotIn("admit", self.events)

    def test_exact_resume_pin_reaches_admission_but_no_later_work_when_denied(self):
        worker = self.worker(from_step=1, admit=lambda: False)
        stack, mocks = self.block_after_admission()
        with stack:
            self.error("admission_rejected", self.run_sentinel, worker,
                       resume_checkpoint_name="checkpoint-00000001",
                       resume_manifest_sha256="e" * 64)
        for operation in mocks:
            operation.assert_not_called()

    def test_denial_is_sticky_and_interrupt_is_once(self):
        worker = self.worker(admit=lambda: False)
        for _ in range(2):
            self.error("admission_rejected", self.run_sentinel, worker)
        self.assertEqual(self.events, ["interrupt"])
        self.assertEqual(worker._state, "failed")

    def test_failed_summary_preserves_fixed_first_failure_without_reinterrupting(self):
        worker = self.worker(admit=lambda: False)
        self.error("admission_rejected", self.run_sentinel, worker)
        summary = worker.summary()
        self.assertEqual(summary["state"], "failed")
        self.assertEqual(summary["failure_code"], "admission_rejected")
        self.assertEqual(summary["failure_stage"], "admission")
        self.assertEqual(summary["completed_steps"], 0)
        self.assertEqual(self.events, ["interrupt"])
        self.assertNotIn(worker._identity, json.dumps(summary))

    def test_reentrant_run_cannot_be_swallowed_by_admission(self):
        worker = self.worker()

        def reenter():
            self.error("reentry", self.run_sentinel, worker)

        worker._admit = reenter
        stack, mocks = self.block_after_admission()
        with stack:
            self.error("reentry", self.run_sentinel, worker)
        for operation in mocks:
            operation.assert_not_called()
        self.assertEqual(self.events, ["interrupt"])

    def test_owner_mismatch_precedes_admission(self):
        for owner, name in ((runs.os, "getpid"), (runs.threading, "get_ident")):
            worker = self.worker()
            with patch.object(owner, name, return_value=-1):
                self.error("owner_mismatch", self.run_sentinel, worker)
        self.assertNotIn("admit", self.events)

    def test_reinitialization_is_terminal_without_transport_or_data_work(self):
        options = self.options()
        worker = runs.WorkerRun(**options)
        self.error("already_consumed", worker.__init__, **options)
        self.error("already_consumed", self.run_sentinel, worker)
        self.assertEqual(self.events, ["interrupt"])
        self.assertEqual(worker._client.summary()["requests_sent"], 0)

    def test_completed_callback_waits_for_parent_fence_before_synchronize(self):
        worker = self.worker()
        worker._state = "running"  # Callback-only guard setup; no run/model execution.
        with patch.object(runs.WorkerLedgerClient, "wait_step_commit",
                          lambda client, step: self.events.append(("fence", step))):
            worker._sync("completed", 1)
        self.assertEqual(self.events, [("fence", 1), ("completed", 1)])

    def test_failed_fence_never_releases_synchronize(self):
        worker = self.worker()
        worker._state = "running"
        with patch.object(runs.WorkerLedgerClient, "wait_step_commit",
                          side_effect=RuntimeError("SYNTHETIC-PRIVATE")):
            self.error("synchronize_failed", worker._sync, "completed", 1)
        self.assertEqual(self.events, ["interrupt"])

    def test_non_completion_phases_do_not_claim_parent_step_fence(self):
        worker = self.worker()
        worker._state = "running"
        with patch.object(runs.WorkerLedgerClient, "wait_step_commit") as fence:
            for phase in sorted(runs.PHASES - {"completed"}):
                worker._sync(phase, 1)
        fence.assert_not_called()
        self.assertEqual(len(self.events), len(runs.PHASES) - 1)

    def test_invalid_sync_and_rank_guards_precede_callbacks(self):
        for phase, step in (("private", 1), (True, 1), ("inputs", True), ("inputs", 3)):
            worker = self.worker()
            worker._state = "running"
            self.error("synchronize_failed", worker._sync, phase, step)
        for rank in (1, True, -1):
            worker = self.worker()
            worker._state = "running"
            with patch.object(runs.WorkerLedgerClient, "reserve") as reserve:
                self.error("reserve_failed", worker._reserve, rank, private_write_bytes=1)
            reserve.assert_not_called()
        self.assertTrue(all(event == "interrupt" for event in self.events))

    def test_swallowed_callback_reentry_remains_terminal(self):
        worker = self.worker()
        worker._state = "running"

        def callback():
            self.error("reentry", worker._call, lambda: None, "synchronize_failed")

        self.error("reentry", worker._call, callback, "synchronize_failed")
        self.assertEqual(self.events, ["interrupt"])

    def test_retired_callbacks_cannot_interrupt_a_later_shared_epoch(self):
        worker = self.worker()
        worker._state, worker._stage = "complete", "complete"  # Retired-guard unit setup only.
        worker._interrupt_once()
        self.error("already_consumed", self.run_sentinel, worker)
        self.error("already_consumed", worker._sync, "completed", 1)
        self.error("already_consumed", worker._reserve, 0, private_write_bytes=1)
        self.error("already_consumed", worker._fail, "run_failed")
        self.assertEqual(self.events, [])
        self.assertEqual(worker._state, "complete")
        self.assertIsNone(worker._failure)
        self.assertFalse(worker._client.closed)

    def test_unknown_error_text_is_not_retained(self):
        for value in ("SYNTHETIC-PRIVATE", None, True, {}):
            error = runs.WorkerRunError(value)
            self.assertEqual(error.args, ("run_failed",))
            self.assertNotIn("PRIVATE", repr(error))


if __name__ == "__main__":
    unittest.main()
