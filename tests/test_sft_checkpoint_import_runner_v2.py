"""Tiny synthetic runner control tests; no native ML or private payloads."""

from __future__ import annotations

import ast
import copy
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lumi import posttraining_checkpoint_import_runner_v2 as m
from lumi.posttraining_checkpoint_io_contract_v1 import build_kernel_io_contract
from lumi.posttraining_final_base_metadata_v1 import canonical, json_object, sha
from tests.test_sft_checkpoint_import_plan_v1 import documents


def seal_fixture(checkpoint, tokenizer_root, *, config_raw=None):
    """Bind own-created tiny files, optionally a genuine native DCP fixture tree."""
    initial = documents()
    config_raw = initial[3] if config_raw is None else config_raw
    state = json_object(initial[2])
    state["run_specification"]["config_sha256"] = sha(config_raw)
    state_raw = canonical(state)
    (checkpoint / "distributed-training-state.json").write_bytes(state_raw)
    for rank in range(2):
        (checkpoint / f"trainer-rank-{rank:05d}.pt").write_bytes(b"opaque synthetic trainer")
    files = []
    for path in sorted(checkpoint.rglob("*")):
        path.chmod(0o700 if path.is_dir() else 0o600)
        if path.is_file():
            raw = path.read_bytes()
            files.append({"path": path.relative_to(checkpoint).as_posix(), "bytes": len(raw),
                          "sha256": sha(raw)})
    manifest_raw = "".join(f"{row['sha256']}  {row['path']}\n" for row in files).encode()
    (checkpoint / "manifest.sha256").write_bytes(manifest_raw)
    (checkpoint / "manifest.sha256").chmod(0o600)
    files.append({"path": "manifest.sha256", "bytes": len(manifest_raw),
                  "sha256": sha(manifest_raw)})
    tokenfiles = []
    (tokenizer_root / "tokenizer").mkdir(mode=0o700)
    for name in ("lumi_tokenizer_metadata.json", "tokenizer.json", "tokenizer_config.json"):
        relative = "tokenizer/" + name
        raw = b'{"synthetic":true}'
        (tokenizer_root / relative).write_bytes(raw)
        (tokenizer_root / relative).chmod(0o600)
        tokenfiles.append({"relative_path": relative, "bytes": len(raw), "sha256": sha(raw)})
    handoff = json_object(initial[4])
    handoff["files"] = files
    del handoff["handoff_sha256"]
    handoff["handoff_sha256"] = sha(canonical(handoff))
    handoff_raw = canonical(handoff)
    base = json_object(initial[0])
    base["synthetic_fixture_only"] = True
    base["checkpoint"]["manifest_sha256"] = sha(manifest_raw)
    base["handoff"] = {"file_sha256": sha(handoff_raw), "sha256": handoff["handoff_sha256"]}
    base["tokenizer"] = {"files": tokenfiles, "manifest_sha256": sha(canonical(tokenfiles))}
    return canonical(base), manifest_raw, state_raw, config_raw, handoff_raw


def fixture_limits():
    return {"maximum_seconds": 30, "maximum_coordinator_read_bytes": 64 * 1024**2,
            "maximum_package_read_bytes": 32 * 1024**2,
            "maximum_private_write_bytes": 16 * 1024**2,
            "maximum_package_write_bytes": 4 * 1024**2, "maximum_control_bytes": 131072}


def fixture_kernel_contract(metadata, dtype="bfloat16"):
    plan = m.build_import_plan(*metadata, expected_base_sha256=sha(metadata[0]), dtype=dtype)
    bounds = plan["resource_bounds"]
    return build_kernel_io_contract(
        plan,
        checkpoint_read_contract={
            "expected_manifest_sha256": plan["metadata"]["checkpoint_manifest_sha256"],
            "maximum_file_bytes": max(row["bytes"] for row in plan["checkpoint_inventory"]),
            "maximum_read_bytes": 8 * 1024**2, "maximum_read_call_bytes": 1024**2,
        },
        hf_io_contract={
            "maximum_read_bytes": 8 * 1024**2,
            "maximum_write_bytes": bounds["maximum_backend_output_bytes"],
            "maximum_file_bytes": bounds["maximum_backend_output_bytes"],
            "maximum_control_bytes": 65536, "target_shard_bytes": 268435456,
        },
    )


class RunnerControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lumi-runner-control-fixture-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.checkpoint = self.root / "input"
        self.tokens = self.root / "tokens"
        self.attempt = self.root / "attempt"
        self.checkpoint.mkdir(mode=0o700)
        self.tokens.mkdir(mode=0o700)
        (self.checkpoint / "model").mkdir(mode=0o700)
        (self.checkpoint / "model/.metadata").write_bytes(b"synthetic opaque metadata")
        (self.checkpoint / "model/__0_0.distcp").write_bytes(b"synthetic opaque weights")
        self.metadata = seal_fixture(self.checkpoint, self.tokens)
        self.limits = fixture_limits()

    def run_fixture(self, **changes):
        kwargs = {"metadata": self.metadata, "expected_base_sha256": sha(self.metadata[0]),
                  "checkpoint_root": self.checkpoint, "tokenizer_root": self.tokens,
                  "attempt": self.attempt, "dtype": "bfloat16", "limits": self.limits}
        kwargs.update(changes)
        contract = kwargs.get("kernel_io_contract") or fixture_kernel_contract(
            kwargs["metadata"], kwargs["dtype"])
        kwargs.setdefault("kernel_io_contract", contract)
        kwargs.setdefault("expected_kernel_io_contract_sha256", contract["contract_sha256"])
        return m.run_synthetic_import(**kwargs)

    def test_real_entrypoint_refuses_before_any_io_or_callbacks(self):
        with patch.object(m.os, "open") as opened, patch.object(m, "build_import_plan") as plan:
            with self.assertRaisesRegex(m.ImportRunnerError, "^host_guard_not_bound$"):
                m.run_import(approval=True, metadata=object(), callback=lambda: self.fail())
            opened.assert_not_called()
            plan.assert_not_called()

    def test_unsigned_synthetic_scope_refuses_before_attempt(self):
        docs = list(self.metadata)
        base = json_object(docs[0])
        base.pop("synthetic_fixture_only")
        docs[0] = canonical(base)
        with self.assertRaisesRegex(m.ImportRunnerError, "synthetic_scope_rejected"):
            self.run_fixture(metadata=tuple(docs), expected_base_sha256=sha(docs[0]))
        self.assertFalse(self.attempt.exists())

    def test_wrong_metadata_pin_refuses_before_attempt(self):
        with self.assertRaisesRegex(m.ImportRunnerError, "metadata_rejected"):
            self.run_fixture(expected_base_sha256="0" * 64)
        self.assertFalse(self.attempt.exists())

    def test_scope_caps_cannot_be_raised_with_limits(self):
        limits = self.limits | {"maximum_private_write_bytes": 1024**3}
        with self.assertRaisesRegex(m.ImportRunnerError, "invalid_arguments"):
            self.run_fixture(limits=limits)
        self.assertFalse(self.attempt.exists())

    def test_bool_nonfinite_and_extra_limits_fail(self):
        for delta in ({"maximum_seconds": float("inf")}, {"maximum_seconds": True},
                      {"maximum_control_bytes": True}, {"extra": 1}):
            with self.subTest(delta=delta), self.assertRaises(m.ImportRunnerError):
                self.run_fixture(limits=self.limits | delta)
        self.assertFalse(self.attempt.exists())

    def test_preflight_read_and_write_budgets_refuse_before_attempt(self):
        for key, value, code in (("maximum_coordinator_read_bytes", 1, "read_budget_exhausted"),
                                 ("maximum_private_write_bytes", 4 * 1024**2,
                                  "write_budget_exhausted")):
            with self.subTest(key=key), self.assertRaisesRegex(m.ImportRunnerError, code):
                self.run_fixture(limits=self.limits | {key: value})
        self.assertFalse(self.attempt.exists())

    def test_overlapping_output_and_inputs_rejected(self):
        with self.assertRaisesRegex(m.ImportRunnerError, "input_output_overlap"):
            self.run_fixture(attempt=self.checkpoint / "attempt")
        self.assertFalse((self.checkpoint / "attempt").exists())

    def test_symlink_root_and_dotdot_paths_refused(self):
        link = self.root / "link"
        link.symlink_to(self.tokens, target_is_directory=True)
        for path in (link, self.root / "missing" / ".." / "tokens"):
            with self.subTest(path=path), \
                    self.assertRaisesRegex(m.ImportRunnerError, "path_rejected"):
                self.run_fixture(tokenizer_root=path)
        self.assertFalse(self.attempt.exists())

    def test_existing_attempt_is_preserved(self):
        self.attempt.mkdir(mode=0o700)
        sentinel = self.attempt / "sentinel"
        sentinel.write_bytes(b"preserve")
        with self.assertRaisesRegex(m.ImportRunnerError, "attempt_exists_no_retry"):
            self.run_fixture()
        self.assertEqual(sentinel.read_bytes(), b"preserve")

    def test_marker_precedes_input_audit_and_failure_is_terminal(self):
        observed = []

        def fail_audit(*args, **kwargs):
            observed.append(True)
            self.assertTrue((self.attempt / m.STARTED).is_file())
            self.assertEqual((self.attempt / "source-config.json").read_bytes(), self.metadata[3])
            raise ValueError("PRIVATE INPUT TEXT")

        with patch.object(m, "verify_inventory", side_effect=fail_audit):
            with self.assertRaisesRegex(m.ImportRunnerError, "^input_audit_failed$"):
                self.run_fixture()
        self.assertEqual(observed, [True])
        saved = json_object((self.attempt / m.FAILED).read_bytes())
        self.assertEqual(saved["code"], "input_audit_failed")
        self.assertNotIn(b"PRIVATE", (self.attempt / m.FAILED).read_bytes())
        self.assertFalse((self.attempt / m.COMPLETE).exists())
        with self.assertRaisesRegex(m.ImportRunnerError, "attempt_exists_no_retry"):
            self.run_fixture()

    def test_first_token_audit_failure_does_not_reach_checkpoint_kernel(self):
        with patch.object(m, "verify_inventory", side_effect=ValueError) as audit:
            with self.assertRaises(m.ImportRunnerError):
                self.run_fixture()
        audit.assert_called_once()

    def test_partial_marker_write_never_retried(self):
        original = os.write
        first = [True]

        def short(fd, raw):
            if first[0]:
                first[0] = False
                return original(fd, raw[:1])
            return original(fd, raw)

        with patch.object(m.os, "write", side_effect=short), patch.object(m, "_audit") as audit:
            with self.assertRaisesRegex(m.ImportRunnerError, "write_failed"):
                self.run_fixture()
            audit.assert_not_called()
        self.assertEqual((self.attempt / m.STARTED).stat().st_size, 1)
        with self.assertRaisesRegex(m.ImportRunnerError, "attempt_exists_no_retry"):
            self.run_fixture()

    def test_ledger_copies_limits_and_never_refunds(self):
        bounds = copy.deepcopy(self.limits)
        ledger = m._Ledger(bounds, started=m.time.monotonic())
        bounds["maximum_coordinator_read_bytes"] = 1
        ledger.reserve(reads=100, writes=100)
        self.assertEqual((ledger.reads, ledger.writes), (100, 100))
        with self.assertRaisesRegex(m.ImportRunnerError, "read_budget_exhausted"):
            ledger.reserve(reads=ledger.limits["maximum_coordinator_read_bytes"])
        self.assertEqual((ledger.reads, ledger.writes), (100, 100))
        with self.assertRaises(m.ImportRunnerError):
            ledger.tick()

    def test_expired_ledger_cannot_write_or_reserve(self):
        ledger = m._Ledger(self.limits, started=0)
        with patch.object(m.time, "monotonic", return_value=31):
            with self.assertRaisesRegex(m.ImportRunnerError, "deadline_exhausted"):
                ledger.reserve(writes=1)
        self.assertEqual(ledger.writes, 0)

    def test_physical_alias_rejected_even_without_lexical_overlap(self):
        alias = self.root / "alias"
        original = Path.stat

        def stat_path(path, *args, **kwargs):
            if path == alias:
                return original(self.checkpoint)
            return original(path, *args, **kwargs)

        with patch.object(Path, "stat", stat_path):
            with self.assertRaisesRegex(m.ImportRunnerError, "input_output_overlap"):
                m._physical_overlap((self.checkpoint, self.tokens), alias / "attempt")
        self.assertFalse(alias.exists())

    def test_duplicate_source_inode_rejected(self):
        with self.assertRaisesRegex(m.ImportRunnerError, "input_output_overlap"):
            m._physical_overlap((self.checkpoint, self.checkpoint), self.attempt)

    def test_single_owner_process_required_before_io(self):
        with patch.object(m.threading, "active_count", return_value=2), \
                patch.object(m.os, "open") as opened:
            with self.assertRaisesRegex(m.ImportRunnerError, "single_owner_process_required"):
                self.run_fixture()
            opened.assert_not_called()
        self.assertFalse(self.attempt.exists())


    def test_wrong_kernel_contract_pin_rejected_before_attempt(self):
        with patch.object(m.os, "open") as opened:
            with self.assertRaisesRegex(m.ImportRunnerError, "kernel_io_contract_rejected"):
                self.run_fixture(expected_kernel_io_contract_sha256="0" * 64)
            opened.assert_not_called()
        self.assertFalse(self.attempt.exists())

    def test_extra_contract_field_even_rehashed_is_rejected(self):
        contract = fixture_kernel_contract(self.metadata)
        contract["unapproved"] = True
        contract["contract_sha256"] = sha(canonical(
            {key: value for key, value in contract.items() if key != "contract_sha256"}))
        with self.assertRaisesRegex(m.ImportRunnerError, "kernel_io_contract_rejected"):
            self.run_fixture(kernel_io_contract=contract)
        self.assertFalse(self.attempt.exists())

    def test_aggregate_kernel_caps_are_preadmitted_before_attempt(self):
        contract = fixture_kernel_contract(self.metadata)
        contract["hf_io_contract"]["maximum_read_bytes"] = 64 * 1024**2
        contract["contract_sha256"] = sha(canonical(
            {key: value for key, value in contract.items() if key != "contract_sha256"}))
        with self.assertRaisesRegex(m.ImportRunnerError, "read_budget_exhausted"):
            self.run_fixture(kernel_io_contract=contract)
        self.assertFalse(self.attempt.exists())

    def test_metadata_only_snapshot_is_not_a_first_audit_receipt(self):
        from lumi import posttraining_checkpoint_import_torch_v5 as kernel

        with (patch.object(kernel, "convert_and_verify", side_effect=ValueError("private")) as call,
              patch.object(m, "_audit", wraps=m._audit) as audit):
            with self.assertRaisesRegex(m.ImportRunnerError, "kernel_failed"):
                self.run_fixture()
            call.assert_called_once()
            self.assertEqual([item.args[0] for item in audit.call_args_list], [self.tokens])
        saved = json_object((self.attempt / m.FAILED).read_bytes())
        self.assertEqual(saved["stage"], "convert")
        self.assertFalse((self.attempt / m.COMPLETE).exists())

    def test_started_marker_binds_independent_kernel_contract(self):
        contract = fixture_kernel_contract(self.metadata)
        with patch.object(m, "_audit", side_effect=ValueError):
            with self.assertRaises(m.ImportRunnerError):
                self.run_fixture(kernel_io_contract=contract)
        saved = json_object((self.attempt / m.STARTED).read_bytes())
        self.assertEqual(saved["kernel_io_contract_sha256"], contract["contract_sha256"])

    def test_frozen_runner_and_saved_real_plan_are_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        for path, digest in (
            ("src/lumi/posttraining_checkpoint_import_runner_v1.py",
             "e3aaaaf54ec1487b9c1dc33ef27d41f841f97ee83388abfc39029ead91ff4ff6"),
            ("configs/lumi-4b-sft-checkpoint-import-plan-v0.1.json",
             "3138f88ddb871c0a24134a5aae84cb28128581b8d943ca599f24553bd3c3cbd8"),
        ):
            self.assertEqual(hashlib.sha256((root / path).read_bytes()).hexdigest(), digest)

    def test_short_token_audit_read_stops_without_a_second_request_or_kernel(self):
        from lumi import posttraining_checkpoint_import_torch_v5 as kernel

        with (patch.object(os, "read", return_value=b"x") as read,
              patch.object(kernel, "convert_and_verify") as convert):
            with self.assertRaisesRegex(m.ImportRunnerError, "input_audit_failed"):
                self.run_fixture()
            read.assert_called_once()
            convert.assert_not_called()
        self.assertEqual(json_object((self.attempt / m.FAILED).read_bytes())["stage"],
                         "inputs_before")
        self.assertFalse((self.attempt / m.COMPLETE).exists())
        with self.assertRaisesRegex(m.ImportRunnerError, "attempt_exists_no_retry"):
            self.run_fixture()

    def test_existing_admission_marker_and_ledger_helpers_are_unchanged(self):
        root = Path(__file__).resolve().parents[1]

        def definitions(version):
            path = root / f"src/lumi/posttraining_checkpoint_import_runner_{version}.py"
            return {node.name: ast.dump(node) for node in ast.parse(path.read_bytes()).body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))}

        old, new = definitions("v1"), definitions("v2")
        for name in ("ImportRunnerError", "_need", "run_import", "_path", "_snapshot",
                     "_physical_overlap", "_Ledger", "_write_new", "_identities"):
            with self.subTest(helper=name):
                self.assertEqual(old[name], new[name])


if __name__ == "__main__":
    unittest.main()
