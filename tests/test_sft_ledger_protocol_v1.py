"""Synthetic in-memory protocol tests: no socket, file data, ML or GPU activity."""

from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "src/lumi" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load("synthetic_ledger_protocol", "posttraining_sft_ledger_protocol_v1.py")
supervision = _load("synthetic_protocol_supervisor", "posttraining_sft_supervisor_v1.py")


def synthetic_inputs():
    counts = {
        "sequential_invocations": 3, "global_optimizer_steps": 4,
        "rank_optimizer_steps": 8, "logical_sample_events": 8,
        "maximum_unique_train_records": 8, "rank_local_input_loads": 6,
        "maximum_encoded_index_rows": 48, "maximum_selected_binary_bytes": 48 * 32 * 8,
        "maximum_selected_index_bytes": 48 * 176,
        "includes_checkpoint_hashing_or_final_digest": False, "is_total_io_or_cost_budget": False,
    }
    caps = {
        "maximum_combined_runtime_seconds": 60, "maximum_cost_microusd": 1000,
        "total_host_hourly_rate_microusd": 60_000, "maximum_host_ram_bytes": 4096,
        "maximum_per_gpu_memory_bytes": 2048, "maximum_private_write_bytes": 1024,
        "maximum_checkpoint_read_bytes": 4096,
        "maximum_encoded_binary_read_bytes": counts["maximum_selected_binary_bytes"],
        "maximum_encoded_index_read_bytes": counts["maximum_selected_index_bytes"],
        "maximum_log_bytes": 256, "maximum_invocations": 3,
        "maximum_global_optimizer_steps": 4, "maximum_external_network_bytes": 0,
        "maximum_inference_calls": 0,
    }
    return caps, counts


class LedgerProtocolTests(unittest.TestCase):
    def setUp(self):
        caps, counts = synthetic_inputs()
        self.supervisor = supervision.Supervisor(caps, counts, world_size=2, clock_ns=lambda: 0)
        self.tokens = {rank: hashlib.sha256(f"SYNTHETIC-TOKEN-{rank}".encode()).hexdigest() for rank in range(2)}
        self.session = protocol.ParentLedgerSession(self.supervisor, world_size=2, rank_tokens=self.tokens)
        self.sequences = {0: 1, 1: 1}
        self.session.begin_invocation("reference", 0, 2, False)

    def message(self, rank=0, operation="reserve", arguments=None, sequence=None):
        return {
            "schema_version": protocol.REQUEST_SCHEMA, "rank": rank,
            "sequence": self.sequences[rank] if sequence is None else sequence,
            "token": self.tokens[rank], "operation": operation,
            "arguments": {"private_write_bytes": 10} if arguments is None else arguments,
        }

    def send(self, rank=0, operation="reserve", arguments=None):
        frame = protocol.encode_frame(self.message(rank, operation, arguments))
        ack, = self.session.feed(rank, frame)
        self.sequences[rank] += 1
        return ack

    def confirm(self, ack):
        self.session.mark_ack_sent(ack)
        request = self.message(ack.rank, "confirm_ack", {"ack_sha256": ack.sha256}, sequence=ack.sequence)
        self.assertEqual(self.session.feed(ack.rank, protocol.encode_frame(request)), ())

    def assert_code(self, code, operation):
        with self.assertRaises(protocol.LedgerProtocolError) as caught:
            operation()
        self.assertEqual(caught.exception.args, (code,))
        self.assertIn(caught.exception.code, protocol.ERROR_CODES)
        return caught.exception

    def test_reservation_is_predebited_before_ack_send_and_confirmation(self):
        ack = self.send()
        self.assertEqual(self.supervisor.summary()["resources_reserved"]["private_write_bytes"], 10)
        self.assertEqual(self.session.summary()["acknowledgements_sent"], 0)
        self.assertEqual(self.session.summary()["pending_acknowledgements"], 1)
        self.confirm(ack)
        self.assertEqual(self.session.summary()["pending_acknowledgements"], 0)
        self.assertFalse(self.session.summary()["execution_permitted"])

    def test_partial_header_and_body_are_incremental_and_bounded(self):
        frame = protocol.encode_frame(self.message())
        self.assertEqual(self.session.feed(0, frame[:2]), ())
        self.assertEqual(self.session.feed(0, frame[2:8]), ())
        ack, = self.session.feed(0, frame[8:])
        self.confirm(ack)

    def test_length_bomb_fails_from_header_only_without_payload(self):
        self.assert_code("frame_size_rejected", lambda: self.session.feed(0, (2**32 - 1).to_bytes(4, "big")))
        self.assertEqual(self.supervisor.summary()["state"], "failed")

    def test_zero_length_header_is_invalid(self):
        self.assert_code("frame_size_rejected", lambda: self.session.feed(0, bytes(4)))

    def test_overlarge_input_chunk_is_rejected(self):
        self.assert_code("frame_size_rejected", lambda: self.session.feed(0, bytes(4101)))

    def test_wrong_chunk_type_is_rejected(self):
        self.assert_code("frame_type_rejected", lambda: self.session.feed(0, "secret"))

    def test_noncanonical_whitespace_rejects(self):
        raw = json.dumps(self.message()).encode()
        self.assert_code("frame_noncanonical", lambda: self.session.feed(0, len(raw).to_bytes(4, "big") + raw))

    def test_duplicate_json_key_rejects(self):
        raw = b'{"rank":0,"rank":0}'
        self.assert_code("frame_noncanonical", lambda: self.session.feed(0, len(raw).to_bytes(4, "big") + raw))

    def test_invalid_utf8_rejects(self):
        raw = b'\xff'
        self.assert_code("frame_noncanonical", lambda: self.session.feed(0, len(raw).to_bytes(4, "big") + raw))

    def test_nonfinite_json_rejects(self):
        raw = b'{"x":NaN}'
        self.assert_code("frame_noncanonical", lambda: self.session.feed(0, len(raw).to_bytes(4, "big") + raw))

    def test_nonobject_json_rejects(self):
        raw = b'[]'
        self.assert_code("frame_schema_rejected", lambda: self.session.feed(0, len(raw).to_bytes(4, "big") + raw))

    def test_unknown_schema_field_rejects(self):
        message = self.message()
        message["raw_text"] = "DO-NOT-RETAIN"
        self.assert_code("frame_schema_rejected", lambda: self.session.feed(0, protocol.encode_frame(message)))
        self.assertNotIn("DO-NOT-RETAIN", json.dumps(self.session.summary()))

    def test_rank_channel_mismatch_rejects(self):
        self.assert_code("authentication_rejected", lambda: self.session.feed(1, protocol.encode_frame(self.message(0))))

    def test_wrong_rank_token_rejects(self):
        message = self.message()
        message["token"] = self.tokens[1]
        self.assert_code("authentication_rejected", lambda: self.session.feed(0, protocol.encode_frame(message)))

    def test_boolean_rank_rejects(self):
        message = self.message()
        message["rank"] = False
        self.assert_code("authentication_rejected", lambda: self.session.feed(0, protocol.encode_frame(message)))

    def test_boolean_sequence_rejects(self):
        message = self.message()
        message["sequence"] = True
        self.assert_code("sequence_rejected", lambda: self.session.feed(0, protocol.encode_frame(message)))

    def test_sequence_must_start_at_one_and_cannot_skip(self):
        message = self.message(sequence=2)
        self.assert_code("sequence_rejected", lambda: self.session.feed(0, protocol.encode_frame(message)))

    def test_replay_after_confirmation_rejects(self):
        original = self.message()
        self.confirm(self.send())
        self.assert_code("sequence_rejected", lambda: self.session.feed(0, protocol.encode_frame(original)))

    def test_next_request_before_ack_confirmation_rejects(self):
        self.send()
        self.assert_code("request_pending", lambda: self.send())

    def test_confirm_before_send_success_rejects(self):
        ack = self.send()
        message = self.message(0, "confirm_ack", {"ack_sha256": ack.sha256}, sequence=ack.sequence)
        self.assert_code("ack_not_sent", lambda: self.session.feed(0, protocol.encode_frame(message)))

    def test_duplicate_ack_sent_callback_rejects(self):
        ack = self.send()
        self.session.mark_ack_sent(ack)
        self.assert_code("ack_rejected", lambda: self.session.mark_ack_sent(ack))

    def test_duplicate_worker_ack_confirmation_rejects(self):
        ack = self.send()
        self.confirm(ack)
        message = self.message(0, "confirm_ack", {"ack_sha256": ack.sha256}, sequence=ack.sequence)
        self.assert_code("ack_rejected", lambda: self.session.feed(0, protocol.encode_frame(message)))

    def test_wrong_ack_digest_rejects(self):
        ack = self.send()
        self.session.mark_ack_sent(ack)
        message = self.message(0, "confirm_ack", {"ack_sha256": "0" * 64}, sequence=ack.sequence)
        self.assert_code("ack_rejected", lambda: self.session.feed(0, protocol.encode_frame(message)))

    def test_forged_ack_handle_types_are_scrubbed(self):
        self.send()
        forged = protocol.PendingAck(0, True, object(), "x")
        self.assert_code("ack_rejected", lambda: self.session.mark_ack_sent(forged))

    def test_failed_ack_send_is_sticky_and_does_not_refund(self):
        self.send()
        self.assert_code("ack_send_failed", lambda: self.session.ack_send_failed(0))
        self.assertEqual(self.supervisor.summary()["resources_reserved"]["private_write_bytes"], 10)
        self.assert_code("ack_send_failed", lambda: self.send(1))

    def test_no_worker_ack_at_deadline_is_sticky(self):
        ack = self.send()
        self.session.mark_ack_sent(ack)
        self.assert_code("ack_timeout", self.session.check_pending)

    def test_incomplete_frame_disconnect_is_sticky(self):
        frame = protocol.encode_frame(self.message())
        self.session.feed(0, frame[:-1])
        self.assert_code("frame_incomplete", lambda: self.session.disconnect(0))

    def test_complete_frame_disconnect_is_sticky(self):
        self.confirm(self.send())
        self.assert_code("rank_disconnected", lambda: self.session.disconnect(0))

    def test_noncoordinator_cannot_reserve_global_step(self):
        self.assert_code("step_coordinator_required", lambda: self.send(1, "reserve_step", {"from_step": 0, "to_step": 1}))

    def test_all_rank_confirmations_required_before_optimizer_completion(self):
        self.confirm(self.send(0, "reserve_step", {"from_step": 0, "to_step": 1}))
        self.confirm(self.send(0, "step_complete", {"to_step": 1}))
        self.assertEqual(self.supervisor.summary()["global_optimizer_steps_completed"], 0)
        rank1 = self.send(1, "step_complete", {"to_step": 1})
        self.assertEqual(self.supervisor.summary()["global_optimizer_steps_completed"], 0)
        self.confirm(rank1)
        self.assertEqual(self.supervisor.summary()["global_optimizer_steps_completed"], 1)

    def test_duplicate_rank_step_completion_rejects(self):
        self.confirm(self.send(0, "reserve_step", {"from_step": 0, "to_step": 1}))
        self.confirm(self.send(0, "step_complete", {"to_step": 1}))
        self.assert_code("step_ack_duplicate", lambda: self.send(0, "step_complete", {"to_step": 1}))

    def test_step_complete_before_reservation_confirmation_rejects(self):
        self.send(0, "reserve_step", {"from_step": 0, "to_step": 1})
        self.assert_code("step_boundary_rejected", lambda: self.send(1, "step_complete", {"to_step": 1}))

    def test_worker_cannot_call_lifecycle_or_memory_operations(self):
        self.assert_code("operation_rejected", lambda: self.send(0, "finish_invocation", {"stop_step": 2}))

    def test_negative_resource_or_boolean_amount_rejects(self):
        self.assert_code("argument_rejected", lambda: self.send(0, "reserve", {"private_write_bytes": True}))

    def test_zero_resource_reservation_rejects(self):
        self.assert_code("argument_rejected", lambda: self.send(0, "reserve", {"private_write_bytes": 0}))

    def test_unknown_resource_rejects(self):
        self.assert_code("argument_rejected", lambda: self.send(0, "reserve", {"network_bytes": 0}))

    def test_simultaneous_rank_budget_debits_share_one_atomic_supervisor(self):
        messages = [protocol.encode_frame(self.message(rank, arguments={"private_write_bytes": 600})) for rank in range(2)]
        def debit(rank):
            try:
                return self.session.feed(rank, messages[rank])
            except protocol.LedgerProtocolError as error:
                return error.code
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(debit, range(2)))
        self.assertEqual(sum(type(result) is tuple for result in results), 1)
        self.assertIn("supervisor_rejected", results)
        self.assertEqual(self.supervisor.summary()["resources_reserved"]["private_write_bytes"], 600)
        self.assertEqual(self.supervisor.summary()["state"], "failed")

    def test_supervisor_exception_text_is_scrubbed(self):
        def poisoned(*_args, **_kwargs):
            raise RuntimeError("SECRET-TOKEN-OR-RAW-TEXT")
        self.supervisor.reserve = poisoned
        error = self.assert_code("supervisor_rejected", self.send)
        self.assertTrue(error.__suppress_context__)
        self.assertNotIn("SECRET", json.dumps(self.session.summary()))

    def test_summary_and_ack_frames_never_contain_tokens(self):
        ack = self.send()
        text = json.dumps(self.session.summary()) + ack.frame.decode("latin1") + repr(ack)
        for token in self.tokens.values():
            self.assertNotIn(token, text)

    def test_rank_tokens_must_be_unique_and_match_world_size(self):
        self.assert_code("configuration_rejected", lambda: protocol.ParentLedgerSession(
            self.supervisor, world_size=2, rank_tokens={0: self.tokens[0], 1: self.tokens[0]}
        ))
        self.assert_code("configuration_rejected", lambda: protocol.ParentLedgerSession(
            self.supervisor, world_size=3, rank_tokens={0: self.tokens[0], 1: self.tokens[1]}
        ))

    def test_generated_rank_tokens_have_independent_256_bit_shapes(self):
        tokens = protocol.create_rank_tokens(2)
        self.assertEqual(set(tokens), {0, 1})
        self.assertEqual(len(set(tokens.values())), 2)
        self.assertTrue(all(len(token) == 64 for token in tokens.values()))

    def test_parent_lifecycle_waits_for_no_pending_transactions(self):
        self.send()
        self.assert_code("lifecycle_rejected", lambda: self.session.finish_invocation(2))

    def test_complete_three_invocation_sequence_never_grants_execution(self):
        for index, (run, first, last, resume) in enumerate(supervision.SEQUENCE):
            if index:
                self.session.begin_invocation(run, first, last, resume)
            self.session.observe_memory(256, [64, 64])
            for step in range(first, last):
                self.confirm(self.send(0, "reserve_step", {"from_step": step, "to_step": step + 1}))
                for rank in range(2):
                    self.confirm(self.send(rank, "step_complete", {"to_step": step + 1}))
            self.session.finish_invocation(last)
        self.assertEqual(self.supervisor.summary()["state"], "completed")
        self.assertEqual(self.session.summary()["global_optimizer_steps_completed"], 4)
        self.assertEqual(self.session.summary()["state"], "completed")
        self.assertFalse(self.session.summary()["execution_permitted"])


if __name__ == "__main__":
    unittest.main()
