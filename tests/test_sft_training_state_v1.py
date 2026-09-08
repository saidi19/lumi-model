"""Synthetic pure-stdlib state tests; no model, data, network, or training work."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/lumi/posttraining_sft_training_state_v1.py"
spec = importlib.util.spec_from_file_location("_synthetic_sft_training_state_v1", MODULE_PATH)
states = importlib.util.module_from_spec(spec)
spec.loader.exec_module(states)

IDENTITY = "a" * 64
GLOBAL_RECORDS = 2 * 3 * 2


def metrics(step=1, **changes):
    return {"to_step": step, "loss": 2.5, "gradient_norm": 0.5,
            "tokens": 72, "supervised_tokens": 24, **changes}


class TrainingStateTests(unittest.TestCase):
    def zero(self):
        return states.make_state(IDENTITY)

    def one(self):
        return states.advance_state(self.zero(), metrics=metrics(),
                                    global_records_per_step=GLOBAL_RECORDS)

    def validate(self, state, **changes):
        options = {"run_identity_sha256": IDENTITY, "from_step": state.get("step", 0),
                   "global_records_per_step": GLOBAL_RECORDS, **changes}
        return states.validate_state(state, **options)

    def advance(self, state, aggregate):
        return states.advance_state(state, metrics=aggregate,
                                    global_records_per_step=GLOBAL_RECORDS)

    def rejected(self, function, *args, **kwargs):
        with self.assertRaises(states.TrainingStateError) as caught:
            function(*args, **kwargs)
        self.assertIn(caught.exception.code, states.ERROR_CODES)
        self.assertEqual(caught.exception.args, (caught.exception.code,))

    def test_zero_state_exact_schema_and_independent_copies(self):
        zero = self.zero()
        self.assertEqual(zero, {
            "schema_version": states.SCHEMA_VERSION, "run_identity_sha256": IDENTITY,
            "step": 0, "sequence_cursor": 0, "records_seen": 0, "tokens_seen": 0,
            "supervised_tokens_seen": 0, "last_loss": None, "best_loss": None,
        })
        copied = self.validate(zero)
        self.assertEqual(copied, zero)
        self.assertIsNot(copied, zero)
        copied["step"] = 1
        self.assertEqual(zero["step"], 0)
        self.assertEqual(self.zero(), zero)

    def test_uninterrupted_and_resumed_step_two_match_exactly(self):
        first = self.one()
        checkpoint = {**first, "latest_checkpoint": "checkpoint-00000001", "status": "checkpoint"}
        loaded = json.loads(json.dumps(checkpoint, allow_nan=False))
        resumed = self.validate(loaded, from_step=1)
        uninterrupted = self.advance(first, metrics(2, loss=1.25, tokens=84, supervised_tokens=36))
        continued = self.advance(resumed, metrics(2, loss=1.25, tokens=84, supervised_tokens=36))
        self.assertEqual(continued, uninterrupted)
        self.assertEqual(continued["step"], 2)
        self.assertEqual(continued["sequence_cursor"], 24)
        self.assertEqual(continued["records_seen"], 24)
        self.assertEqual(continued["tokens_seen"], 156)
        self.assertEqual(continued["supervised_tokens_seen"], 60)
        self.assertEqual(continued["last_loss"], 1.25)
        self.assertEqual(continued["best_loss"], 1.25)
        self.assertEqual(self.validate(continued), continued)
        self.assertIn("latest_checkpoint", loaded)

    def test_global_aggregates_are_not_multiplied_or_reweighted(self):
        state = self.one()
        self.assertEqual(state["tokens_seen"], 72)
        self.assertEqual(state["supervised_tokens_seen"], 24)
        self.assertEqual(state["last_loss"], 2.5)
        self.assertEqual(state["records_seen"], GLOBAL_RECORDS)

    def test_best_loss_retained_when_next_loss_increases(self):
        result = self.advance(self.one(), metrics(2, loss=3.5))
        self.assertEqual(result["last_loss"], 3.5)
        self.assertEqual(result["best_loss"], 2.5)

    def test_advance_does_not_mutate_inputs(self):
        original, aggregate = self.zero(), metrics()
        before = json.dumps([original, aggregate], sort_keys=True)
        self.advance(original, aggregate)
        self.assertEqual(json.dumps([original, aggregate], sort_keys=True), before)

    def test_integer_losses_are_canonical_floats_and_zero_is_valid(self):
        result = self.advance(self.zero(), metrics(loss=0, gradient_norm=0))
        self.assertIs(type(result["last_loss"]), float)
        self.assertIs(type(result["best_loss"]), float)
        self.assertEqual(result["last_loss"], 0.0)

    def test_identity_requires_lowercase_exact_sha256(self):
        for identity in (None, True, b"a" * 64, "a" * 63, "a" * 65, "A" * 64, "g" * 64,
                         "a" * 63 + "\n"):
            with self.subTest(kind=type(identity).__name__):
                self.rejected(states.make_state, identity)

    def test_changed_fresh_plan_identity_rejects_resume(self):
        self.rejected(self.validate, self.one(), run_identity_sha256="b" * 64)

    def test_resume_step_and_record_geometry_are_exact(self):
        self.rejected(self.validate, self.one(), from_step=0)
        self.rejected(self.validate, self.one(), from_step=2)
        self.rejected(self.validate, self.one(), global_records_per_step=GLOBAL_RECORDS + 1)

    def test_invalid_validation_arguments(self):
        for field, values in (("from_step", (True, -1, 3, 1.0, "1")),
                              ("global_records_per_step", (False, 0, -1, 1.0, "12")),
                              ("run_identity_sha256", (None, "private-path"))):
            for value in values:
                with self.subTest(field=field, kind=type(value).__name__):
                    self.rejected(self.validate, self.zero(), **{field: value})

    def test_state_schema_does_not_accept_missing_or_extra_keys(self):
        for field in self.zero():
            state = self.zero()
            del state[field]
            self.rejected(self.validate, state)
        self.rejected(self.validate, {**self.zero(), "private_text": "SYNTHETIC-PRIVATE"})
        self.rejected(self.validate, {**self.zero(), "schema_version": "old"})

    def test_only_exact_checkpoint_pair_is_stripped(self):
        state = self.one()
        for additions in ({"status": "checkpoint"}, {"latest_checkpoint": "checkpoint-00000001"},
                          {"status": "active", "latest_checkpoint": "checkpoint-00000001"},
                          {"status": "checkpoint", "latest_checkpoint": "checkpoint-00000002"},
                          {"status": "checkpoint", "latest_checkpoint": "../checkpoint-00000001"},
                          {"status": True, "latest_checkpoint": "checkpoint-00000001"}):
            self.rejected(self.validate, {**state, **additions})
        self.rejected(self.validate, {**self.zero(), "status": "checkpoint",
                                      "latest_checkpoint": "checkpoint-00000000"})

    def test_counter_types_negatives_and_inconsistent_cursor_are_rejected(self):
        for field in ("sequence_cursor", "records_seen", "tokens_seen", "supervised_tokens_seen"):
            for value in (True, -1, 1.0, "1", None):
                self.rejected(self.validate, {**self.one(), field: value})
        self.rejected(self.validate, {**self.one(), "sequence_cursor": 13})
        self.rejected(self.validate, {**self.one(), "records_seen": 13})
        self.rejected(self.validate, {**self.one(), "supervised_tokens_seen": 73})
        self.rejected(self.validate, {**self.one(), "supervised_tokens_seen": 0})

    def test_zero_state_requires_zero_aggregates_and_absent_losses(self):
        for field in ("tokens_seen", "supervised_tokens_seen", "last_loss", "best_loss"):
            self.rejected(self.validate, {**self.zero(), field: 1})

    def test_nonfinite_negative_boolean_or_absent_losses_are_rejected(self):
        for field in ("last_loss", "best_loss"):
            for value in (float("nan"), float("inf"), -float("inf"), -1, True, None, "1",
                          10 ** 1000):
                self.rejected(self.validate, {**self.one(), field: value})
        self.rejected(self.validate, {**self.one(), "best_loss": 2.0})
        two = self.advance(self.one(), metrics(2))
        self.rejected(self.validate, {**two, "best_loss": 3.0})
        self.rejected(self.validate, {**two, "supervised_tokens_seen": 1})

    def test_metrics_exact_schema(self):
        for field in metrics():
            aggregate = metrics()
            del aggregate[field]
            self.rejected(self.advance, self.zero(), aggregate)
        self.rejected(self.advance, self.zero(), metrics(raw_ids=[1, 2]))
        self.rejected(self.advance, self.zero(), None)

    def test_only_sequential_steps_one_and_two_are_allowed(self):
        for step in (True, 0, 2, 3, -1, 1.0, "1"):
            self.rejected(self.advance, self.zero(), metrics(step))
        self.rejected(self.advance, self.one(), metrics(1))
        two = self.advance(self.one(), metrics(2))
        self.rejected(self.advance, two, metrics(3))

    def test_metrics_numeric_contract(self):
        for field in ("loss", "gradient_norm"):
            for value in (float("nan"), float("inf"), -1, True, None, "1", 10 ** 1000):
                self.rejected(self.advance, self.zero(), metrics(**{field: value}))
        for field in ("tokens", "supervised_tokens"):
            for value in (0, -1, True, 1.0, "1", None):
                self.rejected(self.advance, self.zero(), metrics(**{field: value}))
        self.rejected(self.advance, self.zero(), metrics(supervised_tokens=73))

    def test_dict_subclasses_and_non_string_keys_reject_without_hooks(self):
        class Hostile(dict):
            def __iter__(self):
                raise AssertionError("SYNTHETIC-PRIVATE")

        self.rejected(states.validate_state, Hostile(self.zero()), run_identity_sha256=IDENTITY,
                      from_step=0, global_records_per_step=GLOBAL_RECORDS)
        self.rejected(self.advance, Hostile(self.zero()), metrics())
        self.rejected(self.advance, self.zero(), Hostile(metrics()))
        self.rejected(self.advance, self.zero(), {**metrics(), 1: None})

    def test_error_text_is_fixed_and_contains_no_private_input(self):
        for value in ("SYNTHETIC-PRIVATE", True, None, {"private": "value"}):
            error = states.TrainingStateError(value)
            self.assertEqual(str(error), "state_rejected")
            self.assertEqual(error.args, ("state_rejected",))
            self.assertNotIn("PRIVATE", repr(error))

    def test_import_loads_no_ml_modules(self):
        before = set(sys.modules)
        fresh = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fresh)
        forbidden = {"torch", "numpy", "transformers", "safetensors", "tokenizers"}
        self.assertFalse(any(name.split(".")[0] in forbidden
                             for name in set(sys.modules) - before))


if __name__ == "__main__":
    unittest.main()
