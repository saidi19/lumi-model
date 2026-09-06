"""Synthetic stdlib-only epoch tests; no real data, network, workers, or GPUs."""

from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "src" / "lumi"
PACKAGE = "_synthetic_sft_epoch_lumi"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(SOURCE)]
sys.modules[PACKAGE] = package


def _load(filename):
    name = PACKAGE + "." + filename.removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, SOURCE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load("posttraining_sft_epoch_protocol_v1.py")
supervision = _load("posttraining_sft_supervisor_v1.py")


def synthetic_inputs():
    counts = {
        "sequential_invocations": 3,
        "global_optimizer_steps": 4,
        "rank_optimizer_steps": 8,
        "logical_sample_events": 8,
        "maximum_unique_train_records": 8,
        "rank_local_input_loads": 6,
        "maximum_encoded_index_rows": 48,
        "maximum_selected_binary_bytes": 48 * 32 * 8,
        "maximum_selected_index_bytes": 48 * 176,
        "includes_checkpoint_hashing_or_final_digest": False,
        "is_total_io_or_cost_budget": False,
    }
    caps = {
        "maximum_combined_runtime_seconds": 60,
        "maximum_cost_microusd": 1000,
        "total_host_hourly_rate_microusd": 60_000,
        "maximum_host_ram_bytes": 4096,
        "maximum_per_gpu_memory_bytes": 2048,
        "maximum_private_write_bytes": 1024,
        "maximum_checkpoint_read_bytes": 4096,
        "maximum_encoded_binary_read_bytes": counts["maximum_selected_binary_bytes"],
        "maximum_encoded_index_read_bytes": counts["maximum_selected_index_bytes"],
        "maximum_log_bytes": 256,
        "maximum_invocations": 3,
        "maximum_global_optimizer_steps": 4,
        "maximum_external_network_bytes": 0,
        "maximum_inference_calls": 0,
    }
    return caps, counts


def synthetic_secret(label):
    return hashlib.sha256(label.encode("ascii")).hexdigest()


class EpochProtocolTests(unittest.TestCase):
    def setUp(self):
        caps, counts = synthetic_inputs()
        self.clock = [0]
        self.supervisor = supervision.Supervisor(
            caps,
            counts,
            world_size=2,
            clock_ns=lambda: self.clock[0],
        )
        self.new_session(0)
        self.session.begin_invocation(*supervision.SEQUENCE[0])

    def new_session(self, index):
        self.epoch = synthetic_secret(f"SYNTHETIC-EPOCH-{index}")
        self.tokens = {
            rank: synthetic_secret(f"SYNTHETIC-TOKEN-{index}-{rank}") for rank in range(2)
        }
        self.session = protocol.EpochLedgerSession(
            self.supervisor,
            world_size=2,
            rank_tokens=self.tokens,
            epoch=self.epoch,
        )
        self.sequences = {0: 1, 1: 1}

    def message(self, rank=0, operation="reserve", arguments=None, sequence=None):
        return {
            "schema_version": protocol.REQUEST_SCHEMA,
            "epoch": self.epoch,
            "rank": rank,
            "sequence": self.sequences[rank] if sequence is None else sequence,
            "token": self.tokens[rank],
            "operation": operation,
            "arguments": {"private_write_bytes": 10} if arguments is None else arguments,
        }

    def send(self, rank=0, operation="reserve", arguments=None):
        frame = protocol.encode_frame(self.message(rank, operation, arguments))
        (ack,) = self.session.feed(rank, frame)
        self.sequences[rank] += 1
        return ack

    def confirmation(self, ack):
        return self.message(
            ack.rank, "confirm_ack", {"ack_sha256": ack.sha256}, sequence=ack.sequence
        )

    def confirm(self, ack):
        self.session.mark_ack_sent(ack)
        self.assertEqual(
            self.session.feed(ack.rank, protocol.encode_frame(self.confirmation(ack))), ()
        )

    def step(self, first):
        self.confirm(self.send(0, "reserve_step", {"from_step": first, "to_step": first + 1}))
        for rank in range(2):
            self.confirm(self.send(rank, "step_complete", {"to_step": first + 1}))

    def finish(self, index):
        _run, first, last, _resume = supervision.SEQUENCE[index]
        self.session.observe_memory(256, [64, 64])
        for first_step in range(first, last):
            self.step(first_step)
        self.session.finish_invocation(last)

    def assert_code(self, code, action):
        with self.assertRaises(protocol.LedgerProtocolError) as caught:
            action()
        self.assertEqual(caught.exception.args, (code,))
        self.assertIn(caught.exception.code, protocol.ERROR_CODES)
        self.assertTrue(caught.exception.__suppress_context__)
        return caught.exception

    def test_import_shim_bypasses_lumi_init_and_ml_dependencies(self):
        import subprocess

        program = "\n".join(
            (
                "import importlib, sys, types",
                "assert sys.flags.no_site == 1 and sys.dont_write_bytecode",
                "assert all(name not in sys.modules for name in ('lumi', 'torch', 'numpy'))",
                "package = types.ModuleType('_isolated_sft_epoch_import')",
                "package.__path__ = [sys.argv[1]]",
                "sys.modules[package.__name__] = package",
                "importlib.import_module(package.__name__ + '.posttraining_sft_epoch_protocol_v1')",
                "assert all(name not in sys.modules for name in ('lumi', 'torch', 'numpy'))",
            )
        )
        completed = subprocess.run(
            [sys.executable, "-S", "-B", "-c", program, str(SOURCE)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
        self.assertEqual(completed.returncode, 0, "isolated import proof failed")

    def test_ack_exact_schema_binds_epoch_rank_sequence_and_digest(self):
        ack = self.send()
        (message,) = protocol.FrameDecoder().feed(ack.frame)
        self.assertEqual(set(message), protocol.ACK_FIELDS)
        self.assertEqual(
            message,
            {
                "schema_version": protocol.ACK_SCHEMA,
                "epoch": self.epoch,
                "rank": 0,
                "sequence": 1,
                "operation": "reserve",
                "status": "reserved",
            },
        )
        self.assertEqual(ack.sha256, hashlib.sha256(ack.frame).hexdigest())

    def test_reservation_is_irrevocably_charged_before_ack(self):
        ack = self.send()
        self.assertEqual(self.supervisor.summary()["resources_reserved"]["private_write_bytes"], 10)
        self.assertEqual(self.session.summary()["acknowledgements_sent"], 0)
        self.assertEqual(self.session.summary()["pending_acknowledgements"], 1)
        self.confirm(ack)
        self.assertEqual(self.session.summary()["pending_acknowledgements"], 0)

    def test_partial_header_and_body_are_incremental(self):
        frame = protocol.encode_frame(self.message())
        self.assertEqual(self.session.feed(0, frame[:2]), ())
        self.assertEqual(self.session.feed(0, frame[2:8]), ())
        (ack,) = self.session.feed(0, frame[8:])
        self.confirm(ack)

    def test_length_bomb_rejected_from_header(self):
        self.assert_code(
            "frame_size_rejected",
            lambda: self.session.feed(
                0,
                (2**32 - 1).to_bytes(4, "big"),
            ),
        )
        self.assertEqual(self.supervisor.summary()["state"], "failed")

    def test_overlarge_chunk_rejected(self):
        self.assert_code("frame_size_rejected", lambda: self.session.feed(0, bytes(4101)))

    def test_zero_frame_rejected(self):
        self.assert_code("frame_size_rejected", lambda: self.session.feed(0, bytes(4)))

    def test_nonbytes_chunk_rejected(self):
        self.assert_code("frame_type_rejected", lambda: self.session.feed(0, "UNTRUSTED-TEXT"))

    def test_noncanonical_whitespace_rejected(self):
        raw = json.dumps(self.message()).encode()
        self.assert_code(
            "frame_noncanonical", lambda: self.session.feed(0, len(raw).to_bytes(4, "big") + raw)
        )

    def test_duplicate_keys_rejected(self):
        raw = b'{"epoch":"a","epoch":"b"}'
        self.assert_code(
            "frame_noncanonical", lambda: self.session.feed(0, len(raw).to_bytes(4, "big") + raw)
        )

    def test_nonfinite_json_rejected(self):
        raw = b'{"epoch":NaN}'
        self.assert_code(
            "frame_noncanonical", lambda: self.session.feed(0, len(raw).to_bytes(4, "big") + raw)
        )

    def test_unknown_request_field_rejected_and_not_retained(self):
        message = self.message()
        message["raw_text"] = "DO-NOT-RETAIN"
        self.assert_code(
            "frame_schema_rejected", lambda: self.session.feed(0, protocol.encode_frame(message))
        )
        self.assertNotIn("DO-NOT-RETAIN", json.dumps(self.session.summary()))

    def test_missing_epoch_rejected(self):
        message = self.message()
        del message["epoch"]
        self.assert_code(
            "frame_schema_rejected", lambda: self.session.feed(0, protocol.encode_frame(message))
        )

    def test_legacy_schema_rejected(self):
        message = self.message()
        message["schema_version"] = "lumi.sft-ledger-request.v0.1"
        self.assert_code(
            "frame_schema_rejected", lambda: self.session.feed(0, protocol.encode_frame(message))
        )

    def test_wrong_epoch_rejected_before_any_debit(self):
        message = self.message()
        message["epoch"] = synthetic_secret("OLD-EPOCH")
        self.assert_code(
            "authentication_rejected", lambda: self.session.feed(0, protocol.encode_frame(message))
        )
        self.assertEqual(self.supervisor.summary()["resource_reservations"], 0)

    def test_malformed_epoch_rejected(self):
        message = self.message()
        message["epoch"] = "G" * 64
        self.assert_code(
            "authentication_rejected", lambda: self.session.feed(0, protocol.encode_frame(message))
        )

    def test_rank_channel_mismatch_rejected(self):
        self.assert_code(
            "authentication_rejected",
            lambda: self.session.feed(1, protocol.encode_frame(self.message(0))),
        )

    def test_wrong_rank_token_rejected(self):
        message = self.message()
        message["token"] = self.tokens[1]
        self.assert_code(
            "authentication_rejected", lambda: self.session.feed(0, protocol.encode_frame(message))
        )

    def test_boolean_channel_rank_rejected(self):
        self.assert_code(
            "authentication_rejected",
            lambda: self.session.feed(False, protocol.encode_frame(self.message())),
        )

    def test_boolean_sequence_rejected(self):
        message = self.message(sequence=True)
        self.assert_code(
            "sequence_rejected", lambda: self.session.feed(0, protocol.encode_frame(message))
        )

    def test_sequence_starts_at_one_and_cannot_skip(self):
        message = self.message(sequence=2)
        self.assert_code(
            "sequence_rejected", lambda: self.session.feed(0, protocol.encode_frame(message))
        )

    def test_replay_after_confirmation_rejected(self):
        original = protocol.encode_frame(self.message())
        self.confirm(self.send())
        self.assert_code("sequence_rejected", lambda: self.session.feed(0, original))

    def test_second_request_before_confirmation_rejected(self):
        self.send()
        self.assert_code("request_pending", self.send)

    def test_confirmation_requires_successful_send(self):
        ack = self.send()
        self.assert_code(
            "ack_not_sent",
            lambda: self.session.feed(0, protocol.encode_frame(self.confirmation(ack))),
        )

    def test_confirmation_requires_current_epoch(self):
        ack = self.send()
        self.session.mark_ack_sent(ack)
        message = self.confirmation(ack)
        message["epoch"] = synthetic_secret("WRONG-ACK-EPOCH")
        self.assert_code(
            "authentication_rejected", lambda: self.session.feed(0, protocol.encode_frame(message))
        )

    def test_confirmation_requires_exact_sequence(self):
        ack = self.send()
        self.session.mark_ack_sent(ack)
        message = self.confirmation(ack)
        message["sequence"] += 1
        self.assert_code(
            "sequence_rejected", lambda: self.session.feed(0, protocol.encode_frame(message))
        )

    def test_confirmation_requires_exact_digest(self):
        ack = self.send()
        self.session.mark_ack_sent(ack)
        message = self.confirmation(ack)
        message["arguments"]["ack_sha256"] = "0" * 64
        self.assert_code(
            "ack_rejected", lambda: self.session.feed(0, protocol.encode_frame(message))
        )

    def test_duplicate_ack_send_callback_rejected(self):
        ack = self.send()
        self.session.mark_ack_sent(ack)
        self.assert_code("ack_rejected", lambda: self.session.mark_ack_sent(ack))

    def test_duplicate_worker_confirmation_rejected(self):
        ack = self.send()
        self.confirm(ack)
        self.assert_code(
            "ack_rejected",
            lambda: self.session.feed(0, protocol.encode_frame(self.confirmation(ack))),
        )

    def test_failed_ack_send_is_sticky_and_does_not_refund(self):
        self.send()
        self.assert_code("ack_send_failed", lambda: self.session.ack_send_failed(0))
        self.assert_code("ack_send_failed", lambda: self.send(1))
        self.assertEqual(self.supervisor.summary()["resources_reserved"]["private_write_bytes"], 10)
        self.assertEqual(self.session._tokens, {})

    def test_ack_deadline_failure_is_sticky(self):
        ack = self.send()
        self.session.mark_ack_sent(ack)
        self.assert_code("ack_timeout", self.session.check_pending)
        self.assert_code("ack_timeout", self.session.check_pending)

    def test_empty_pending_checkpoint_succeeds_while_active(self):
        self.confirm(self.send())
        self.session.check_pending()

    def test_incomplete_frame_disconnect_fails(self):
        frame = protocol.encode_frame(self.message())
        self.session.feed(0, frame[:-1])
        self.assert_code("frame_incomplete", lambda: self.session.disconnect(0))

    def test_early_eof_even_without_pending_requests_fails(self):
        self.confirm(self.send())
        self.assert_code("rank_disconnected", lambda: self.session.disconnect(0))

    def test_noncoordinator_cannot_reserve_step(self):
        self.assert_code(
            "step_coordinator_required",
            lambda: self.send(
                1,
                "reserve_step",
                {"from_step": 0, "to_step": 1},
            ),
        )

    def test_all_ranks_must_confirm_before_step_completion(self):
        self.confirm(self.send(0, "reserve_step", {"from_step": 0, "to_step": 1}))
        self.confirm(self.send(0, "step_complete", {"to_step": 1}))
        self.assertEqual(self.supervisor.summary()["global_optimizer_steps_completed"], 0)
        last = self.send(1, "step_complete", {"to_step": 1})
        self.session.mark_ack_sent(last)
        self.assertEqual(self.supervisor.summary()["global_optimizer_steps_completed"], 0)
        self.session.feed(1, protocol.encode_frame(self.confirmation(last)))
        self.assertEqual(self.supervisor.summary()["global_optimizer_steps_completed"], 1)

    def test_duplicate_rank_completion_rejected(self):
        self.confirm(self.send(0, "reserve_step", {"from_step": 0, "to_step": 1}))
        self.confirm(self.send(0, "step_complete", {"to_step": 1}))
        self.assert_code(
            "step_ack_duplicate", lambda: self.send(0, "step_complete", {"to_step": 1})
        )

    def test_completion_before_reservation_confirmation_rejected(self):
        self.send(0, "reserve_step", {"from_step": 0, "to_step": 1})
        self.assert_code(
            "step_boundary_rejected", lambda: self.send(1, "step_complete", {"to_step": 1})
        )

    def test_worker_cannot_invoke_parent_lifecycle(self):
        self.assert_code(
            "operation_rejected", lambda: self.send(0, "finish_invocation", {"stop_step": 2})
        )

    def test_boolean_resource_amount_rejected(self):
        self.assert_code(
            "argument_rejected", lambda: self.send(0, "reserve", {"private_write_bytes": True})
        )

    def test_zero_resource_reservation_rejected(self):
        self.assert_code(
            "argument_rejected", lambda: self.send(0, "reserve", {"private_write_bytes": 0})
        )

    def test_unknown_resource_rejected(self):
        self.assert_code("argument_rejected", lambda: self.send(0, "reserve", {"network_bytes": 1}))

    def test_simultaneous_ranks_share_atomic_cumulative_budget(self):
        messages = [
            protocol.encode_frame(self.message(rank, arguments={"private_write_bytes": 600}))
            for rank in range(2)
        ]

        def debit(rank):
            try:
                return self.session.feed(rank, messages[rank])
            except protocol.LedgerProtocolError as error:
                return error.code

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(debit, range(2)))
        self.assertEqual(sum(type(result) is tuple for result in results), 1)
        self.assertIn("supervisor_rejected", results)
        self.assertEqual(
            self.supervisor.summary()["resources_reserved"]["private_write_bytes"], 600
        )
        self.assertEqual(self.supervisor.summary()["state"], "failed")

    def test_supervisor_exception_text_scrubbed(self):
        def poisoned(*_args, **_kwargs):
            raise RuntimeError("SECRET-RAW-TEXT")

        self.supervisor.reserve = poisoned
        self.assert_code("supervisor_rejected", self.send)
        self.assertNotIn("SECRET", json.dumps(self.session.summary()))

    def test_tokens_not_in_ack_or_summary_and_epoch_not_in_summary(self):
        ack = self.send()
        text = json.dumps(self.session.summary()) + ack.frame.decode("latin1") + repr(ack)
        for token in self.tokens.values():
            self.assertNotIn(token, text)
        self.assertNotIn(self.epoch, json.dumps(self.session.summary()))
        self.assertFalse(self.session.summary()["execution_permitted"])
        self.assertFalse(self.session.summary()["os_pid_authenticated"])

    def test_invalid_constructor_never_mutates_cumulative_supervisor(self):
        before = self.supervisor.summary()
        for epoch in (None, True, "", "a" * 63, "A" * 64, self.tokens[0]):
            with self.subTest(epoch_shape=type(epoch).__name__):
                self.assert_code(
                    "configuration_rejected",
                    lambda epoch=epoch: protocol.EpochLedgerSession(
                        self.supervisor,
                        world_size=2,
                        rank_tokens=self.tokens,
                        epoch=epoch,
                    ),
                )
        self.assertEqual(self.supervisor.summary(), before)

    def test_duplicate_rank_tokens_rejected_without_mutation(self):
        before = self.supervisor.summary()
        self.assert_code(
            "configuration_rejected",
            lambda: protocol.EpochLedgerSession(
                self.supervisor,
                world_size=2,
                rank_tokens={0: self.tokens[0], 1: self.tokens[0]},
                epoch=self.epoch,
            ),
        )
        self.assertEqual(self.supervisor.summary(), before)

    def test_generated_capabilities_have_256_bit_hex_shape(self):
        epochs = [protocol.create_epoch(), protocol.create_epoch()]
        tokens = protocol.create_rank_tokens(2)
        values = epochs + list(tokens.values())
        self.assertEqual(len(set(values)), 4)
        self.assertTrue(
            all(
                len(value) == 64 and all(c in "0123456789abcdef" for c in value) for value in values
            )
        )

    def test_finish_with_pending_transaction_rejected(self):
        self.send()
        self.assert_code("lifecycle_rejected", lambda: self.session.finish_invocation(2))

    def test_finish_with_partial_frame_rejected_before_advance(self):
        self.session.observe_memory(256, [64, 64])
        self.step(0)
        self.step(1)
        self.session.feed(1, protocol.encode_frame(self.message(1))[:5])
        self.assert_code("frame_incomplete", lambda: self.session.finish_invocation(2))
        self.assertEqual(self.supervisor.summary()["invocations_completed"], 0)

    def test_invocation_begin_cannot_repeat_within_one_session(self):
        self.assert_code(
            "lifecycle_rejected", lambda: self.session.begin_invocation(*supervision.SEQUENCE[0])
        )
        self.assertEqual(self.supervisor.summary()["invocations_started"], 1)

    def test_completed_session_is_sealed_and_eof_is_expected(self):
        self.finish(0)
        before = self.supervisor.summary()
        for rank in range(2):
            self.session.disconnect(rank)
            self.session.disconnect(rank)
        self.assertEqual(self.session._tokens, {})
        self.assertEqual(self.session.summary()["invocations_completed"], 1)
        self.assertEqual(self.session.summary()["state"], "completed")
        self.assertEqual(self.supervisor.summary(), before)

    def test_closed_callbacks_cannot_interrupt_new_active_epoch(self):
        stale_frame = protocol.encode_frame(self.message())
        stale_ack = self.send()
        self.confirm(stale_ack)
        self.finish(0)
        old = self.session
        self.new_session(1)
        self.session.begin_invocation(*supervision.SEQUENCE[1])
        before = self.supervisor.summary()
        for action in (
            lambda: old.feed(0, stale_frame),
            lambda: old.begin_invocation(*supervision.SEQUENCE[1]),
            lambda: old.finish_invocation(2),
            lambda: old.observe_memory(0, [0, 0]),
            lambda: old.mark_ack_sent(stale_ack),
            lambda: old.ack_send_failed(0),
            old.check_pending,
        ):
            self.assert_code("lifecycle_rejected", action)
            self.assertEqual(self.supervisor.summary(), before)
        old.disconnect(0)
        self.assert_code("authentication_rejected", lambda: old.disconnect(True))
        self.assertEqual(self.supervisor.summary(), before)
        self.confirm(self.send())
        self.assertEqual(self.supervisor.summary()["state"], "active")

    def test_new_session_construction_does_not_reset_any_counter(self):
        self.confirm(self.send(arguments={"private_write_bytes": 400, "log_bytes": 10}))
        self.clock[0] = 1_000_000_000
        self.finish(0)
        before = self.supervisor.summary()
        self.new_session(1)
        self.assertEqual(self.supervisor.summary(), before)
        self.session.begin_invocation(*supervision.SEQUENCE[1])
        self.confirm(self.send(arguments={"private_write_bytes": 500}))
        after = self.supervisor.summary()
        self.assertEqual(after["invocations_started"], 2)
        self.assertEqual(after["global_optimizer_steps_reserved"], 2)
        self.assertEqual(after["resources_reserved"]["private_write_bytes"], 910)
        self.assertEqual(after["elapsed_nanoseconds_last_observed"], 1_000_000_000)
        self.assertEqual(
            after["cost_microusd_last_observed"], before["cost_microusd_last_observed"]
        )

    def test_second_epoch_cannot_reset_budget(self):
        self.confirm(self.send(arguments={"private_write_bytes": 600}))
        self.finish(0)
        self.new_session(1)
        self.session.begin_invocation(*supervision.SEQUENCE[1])
        self.assert_code(
            "supervisor_rejected", lambda: self.send(arguments={"private_write_bytes": 600})
        )
        self.assertEqual(
            self.supervisor.summary()["resources_reserved"]["private_write_bytes"], 600
        )

    def test_second_epoch_cannot_replay_reference_invocation(self):
        self.finish(0)
        self.new_session(1)
        self.assert_code(
            "supervisor_rejected", lambda: self.session.begin_invocation(*supervision.SEQUENCE[0])
        )
        self.assertEqual(self.supervisor.summary()["invocations_started"], 1)

    def test_stale_epoch_on_new_channel_rejected(self):
        stale_frame = protocol.encode_frame(self.message())
        self.finish(0)
        self.new_session(1)
        self.session.begin_invocation(*supervision.SEQUENCE[1])
        self.assert_code("authentication_rejected", lambda: self.session.feed(0, stale_frame))

    def test_old_token_cannot_authenticate_new_epoch(self):
        stale_token = self.tokens[0]
        self.finish(0)
        self.new_session(1)
        self.session.begin_invocation(*supervision.SEQUENCE[1])
        message = self.message()
        message["token"] = stale_token
        self.assert_code(
            "authentication_rejected", lambda: self.session.feed(0, protocol.encode_frame(message))
        )

    def test_stale_ack_handle_cannot_match_new_epoch_transaction(self):
        stale_ack = self.send()
        self.confirm(stale_ack)
        self.finish(0)
        self.new_session(1)
        self.session.begin_invocation(*supervision.SEQUENCE[1])
        current_ack = self.send()
        self.assertEqual(current_ack.sequence, stale_ack.sequence)
        self.assertNotEqual(current_ack.sha256, stale_ack.sha256)
        self.assert_code("ack_rejected", lambda: self.session.mark_ack_sent(stale_ack))

    def test_failed_epoch_cannot_advance_or_reenter_cumulative_supervisor(self):
        self.send()
        self.assert_code("ack_send_failed", lambda: self.session.ack_send_failed(0))
        old = self.session
        before = self.supervisor.summary()
        self.new_session(1)
        self.assertEqual(self.supervisor.summary(), before)
        calls = []

        def unexpected_interrupt():
            calls.append("interrupt")

        self.supervisor.interrupt = unexpected_interrupt
        self.assert_code("ack_send_failed", lambda: old.feed(0, b"stale"))
        self.assert_code("ack_send_failed", lambda: old.disconnect(0))
        self.assert_code("ack_send_failed", old.check_pending)
        self.assertEqual(calls, [])
        self.assertEqual(self.supervisor.summary(), before)
        self.assert_code(
            "supervisor_rejected", lambda: self.session.begin_invocation(*supervision.SEQUENCE[1])
        )
        self.assertEqual(self.supervisor.summary()["invocations_started"], 1)
        self.assertEqual(self.supervisor.summary()["invocations_completed"], 0)

    def test_exactly_three_fresh_epochs_complete_four_cumulative_steps(self):
        completed = []
        for index, invocation in enumerate(supervision.SEQUENCE):
            if index:
                self.new_session(index)
                self.session.begin_invocation(*invocation)
            for rank in range(2):
                ack = self.send(rank)
                self.assertEqual(ack.sequence, 1)
                self.confirm(ack)
            self.finish(index)
            for rank in range(2):
                self.session.disconnect(rank)
            completed.append(self.session.summary())
        summary = self.supervisor.summary()
        self.assertEqual(summary["state"], "completed")
        self.assertEqual(summary["invocations_started"], 3)
        self.assertEqual(summary["invocations_completed"], 3)
        self.assertEqual(summary["global_optimizer_steps_reserved"], 4)
        self.assertEqual(summary["global_optimizer_steps_completed"], 4)
        self.assertEqual(summary["resources_reserved"]["private_write_bytes"], 60)
        self.assertEqual(summary["memory_observations"], 3)
        self.assertEqual(
            [item["global_optimizer_steps_completed"] for item in completed], [2, 1, 1]
        )
        self.assertTrue(all(item["invocations_completed"] == 1 for item in completed))
        self.assertTrue(
            all(
                not item["execution_permitted"] and not item["os_pid_authenticated"]
                for item in completed
            )
        )
        self.new_session(3)
        self.assert_code(
            "supervisor_rejected", lambda: self.session.begin_invocation(*supervision.SEQUENCE[0])
        )
        self.assertEqual(self.supervisor.summary()["invocations_started"], 3)


if __name__ == "__main__":
    unittest.main()
