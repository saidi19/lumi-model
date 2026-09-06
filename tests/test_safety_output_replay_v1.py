"""Source/synthetic regression checks, never real private-output reproduction."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from lumi import posttraining_safety_output_replay_v1 as core
from lumi.posttraining_safety_replay_snapshot_v1 import Snapshot

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "configs/lumi-safety-output-replay-inputs-v0.1.json"
SPEC_SHA = "42aaeba21e68d249f41671587c74a92c7adfb6f298e0e19a2efbe390245ac33c"


def _cli():
    spec = importlib.util.spec_from_file_location(
        "safety_replay_cli_tests", ROOT / "scripts/check_safety_output_replay_local_v1.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SafetyOutputReplayTests(unittest.TestCase):
    def test_fixed_input_inventory_preserves_original_pins_and_caps(self):
        raw = SPEC_PATH.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), SPEC_SHA)
        spec = json.loads(raw)
        bindings = spec["bindings"]
        plan = json.loads((ROOT / bindings["plan"]["path"]).read_bytes())
        for role, binding in plan["bindings"].items():
            self.assertEqual(bindings[role]["path"], binding["path"])
            self.assertEqual(bindings[role]["sha256"], binding["sha256"])
        files = [v for v in bindings.values() if "git_revision" not in v]
        self.assertEqual(len(bindings), 25)
        self.assertEqual(len(files), 24)
        self.assertEqual(len({v["path"] for v in bindings.values()}), 25)
        self.assertEqual(sum(v["bytes"] for v in files), 11_860_091)
        cli = _cli()
        self.assertEqual(cli.SPEC_SHA256, SPEC_SHA)
        self.assertEqual(cli.MAX_FILES, spec["limits"]["maximum_bound_input_opens"])
        self.assertEqual(cli.MAX_BYTES, spec["limits"]["maximum_bound_input_bytes"])
        self.assertEqual(cli.MAX_SECONDS, spec["limits"]["maximum_runtime_seconds"])

    def test_payload_authentication_rejects_size_hash_missing_and_extra(self):
        raw = b"synthetic"
        pins = {"a": {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}}
        core.authenticate_payloads({"a": raw}, pins)
        for values in (
            {},
            {"a": raw, "b": raw},
            {"a": b"SYNTHETIC_RAW_SENTINEL"},
            {"a": b"different"},
        ):
            with self.subTest(keys=list(values)):
                with self.assertRaisesRegex(ValueError, "^invalid safety output replay$"):
                    core.authenticate_payloads(values, pins)

    def _modules(self):
        # Current tracked code is used only to exercise the adapter API with
        # synthetic data. Authentic replay separately binds the old Git blob.
        roles = {
            role: (ROOT / "src/lumi" / (name + ".py")).read_bytes()
            for name, role in core.SOURCE_MODULES
        }
        snapshot = Snapshot({"/synthetic/value.json": b'{"value":1}'}, repository_root="/synthetic")
        self.addCleanup(self._clean_modules)
        return roles, snapshot, core.load_historical_modules(roles, snapshot)

    @staticmethod
    def _clean_modules():
        for name in tuple(sys.modules):
            if name == core.PACKAGE or name.startswith(core.PACKAGE + "."):
                del sys.modules[name]

    def test_original_loader_reads_only_snapshot(self):
        _, snapshot, modules = self._modules()
        with patch("builtins.open", side_effect=AssertionError("real read")):
            value = modules["posttraining_safety_uri_repair"]._load_json("/synthetic/value.json")
        self.assertEqual(value, {"value": 1})
        self.assertEqual(snapshot.summary()["virtual_opens"], 1)
        self.assertEqual(modules["posttraining_safety_uri_repair"].Path, snapshot.Path)
        self.assertIs(
            modules["posttraining_safety_uri_repair"].filter_sft_records,
            modules["posttraining_data"].filter_sft_records,
        )

    def test_frozen_counting_function_keeps_assistant_tool_and_context_rules(self):
        _, _, modules = self._modules()
        repair = modules["posttraining_safety_uri_repair"]
        calls = []

        class SyntheticTokenizer:
            all_special_ids = [0, 1, 2]

            def encode(self, text, **kwargs):
                calls.append(kwargs)
                return [20] * len(text)

        template = types.SimpleNamespace(context_window=4096, loss_roles={"assistant", "tool"})
        messages = [{"role": "user", "content": "ab"}, {"role": "assistant", "content": "cde"}]
        self.assertEqual(
            repair._measure_conversation(
                messages, tokenizer=SyntheticTokenizer(), template=template
            ),
            5,
        )
        self.assertTrue(
            all(v == {"add_special_tokens": False, "split_special_tokens": True} for v in calls)
        )
        template.context_window = 5
        with self.assertRaises(ValueError):
            repair._measure_conversation(
                messages, tokenizer=SyntheticTokenizer(), template=template
            )

    def test_original_serialization_tail_accepts_metrics_without_file_io(self):
        roles, _, modules = self._modules()
        repair = modules["posttraining_safety_uri_repair"]
        serializer = core.frozen_serializer(roles["implementation"], repair.__dict__)
        metrics = {
            key: "0" * 64
            for key in (
                "content_and_messages_binding_sha256",
                "candidate_file_sha256",
                "transform_chain_binding_sha256",
                "uri_set_sha256",
            )
        }
        metrics.update(
            dict.fromkeys(
                (
                    "records",
                    "records_with_deterministic_source_uri",
                    "records_with_only_authorized_metadata_delta",
                    "unique_source_uris",
                ),
                320,
            )
        )
        generic = {
            "accepted": 320,
            "total": 320,
            "rejected": 0,
            "rejection_counts": {},
            "raw_text_retained": False,
            "report_sha256": "1" * 64,
        }
        plan = {
            "authorization": repair.EXPECTED_AUTHORIZATION,
            "observed_at": "2026-08-06",
            "bindings": {
                "chat_template": {"sha256": "2" * 64},
                "tokenizer_json": {"sha256": "3" * 64},
            },
        }
        with patch("builtins.open", side_effect=AssertionError("real read")):
            outputs = serializer(
                plan,
                "4" * 64,
                types.SimpleNamespace(sha256="5" * 64),
                types.SimpleNamespace(registry_sha256="6" * 64),
                metrics,
                generic,
                repair.EXPECTED_SAFETY_TOKENS,
                b"synthetic candidate\n",
                types.SimpleNamespace(context_window=4096),
            )
        self.assertEqual(outputs["private_safety_candidate_bytes"], b"synthetic candidate\n")
        self.assertEqual(
            outputs["report"]["frozen_token_verification"]["by_split"],
            repair.EXPECTED_SAFETY_TOKENS,
        )
        self.assertFalse(outputs["report"]["production_eligible"])


class SafetyReplayInputTests(unittest.TestCase):
    def setUp(self):
        self.cli = _cli()
        self.temp = tempfile.TemporaryDirectory(prefix="safety-replay-synthetic-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        patcher = patch.object(self.cli, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.path = self.root / "sample.json"
        self.raw = b'{"synthetic":1}'
        self.path.write_bytes(self.raw)
        self.binding = {"bytes": len(self.raw), "sha256": hashlib.sha256(self.raw).hexdigest()}
        self.boundary = self.cli.Boundary()
        self.reader = self.cli.BoundInputs(self.boundary)
        self.addCleanup(self.reader.close)

    def test_one_open_exact_bytes_and_duplicate_read_rejected(self):
        self.assertEqual(self.reader.read("sample.json", self.binding), self.raw)
        self.assertEqual((self.reader.opens, self.reader.bytes_read), (1, len(self.raw)))
        with self.assertRaisesRegex(self.cli.Stop, "^input_rejected$"):
            self.reader.read("sample.json", self.binding)

    def test_symlink_and_hardlink_rejected(self):
        link = self.root / "link.json"
        link.symlink_to(self.path)
        with self.assertRaisesRegex(self.cli.Stop, "^input_rejected$"):
            self.reader.read("link.json", self.binding)
        hard = self.root / "hard.json"
        os.link(self.path, hard)
        with self.assertRaisesRegex(self.cli.Stop, "^input_rejected$"):
            self.reader.read("hard.json", self.binding)

    def test_mutated_input_fails_final_preservation(self):
        self.reader.read("sample.json", self.binding)
        self.path.write_bytes(b"changed")
        with self.assertRaisesRegex(self.cli.Stop, "^input_changed$"):
            self.reader.verify()

    def test_wrong_hash_and_caps_fail_before_extra_reads(self):
        changed = {**self.binding, "sha256": "0" * 64}
        with self.assertRaisesRegex(self.cli.Stop, "^input_hash_mismatch$"):
            self.reader.read("sample.json", changed)
        self.reader.opens = self.cli.MAX_FILES
        with self.assertRaisesRegex(self.cli.Stop, "^read_budget$"):
            self.reader.read("other.json", self.binding)

    def test_private_phase_blocks_writes_network_process_and_unknown_reads(self):
        events = [
            ("open", (str(self.path), "r", os.O_RDONLY)),
            ("open", (str(self.root / "src/module.py"), "w", os.O_WRONLY)),
            ("socket.connect", (None,)),
            ("subprocess.Popen", (None,)),
            ("os.mkdir", (str(self.root / "x"),)),
        ]
        for event, args in events:
            with self.subTest(event=event):
                with self.assertRaisesRegex(self.cli.Stop, "^boundary_violation$"):
                    self.boundary.guard(event, args)


if __name__ == "__main__":
    unittest.main()
