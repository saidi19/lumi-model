"""Independent synthetic ledger tests and isolated mandatory Linux child probes."""

from __future__ import annotations

import ast
import builtins
import functools
import hashlib
import importlib
import io
import json
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_test_sft_guarded_ledger_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
probe = importlib.import_module(PACKAGE + ".posttraining_sft_guarded_ledger_v1")
EPOCH = "a" * 64
TOKEN = "b" * 64
NONCLAIMS = (
    "continuous_capability_exclusion", "privileged_sender_capabilities_excluded",
    "real_memory_measurement", "os_memory_enforcement", "cgroup_memory_enforcement",
    "aggregate_descendant_enforcement", "os_network_enforcement", "namespace_containment",
    "ptrace_containment", "real_worker_sandbox_verified", "target_gpu_host_verified",
    "guarded_real_worker_ready", "execution_authorized", "real_sft_execution_permitted",
    "production_eligible", "full_private_integration_passed",
)


def isolated(method):
    """Any live confinement happens only inside a fresh -S/-B test subprocess."""
    @functools.wraps(method)
    def run(self):
        if sys.flags.no_site == 1 and sys.dont_write_bytecode:
            return method(self)
        result = subprocess.run(
            [sys.executable, "-S", "-B", str(Path(__file__).resolve()),
             type(self).__name__ + "." + method.__name__],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=35,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
        self.assertEqual(result.returncode, 0, (result.stdout + result.stderr)[-4000:].decode())
    return run


def frame(payload):
    return len(payload).to_bytes(4, "big") + payload


class Transcript:
    """Synthetic worker packets; no worker, socket, guard install, or challenge runs."""

    def __init__(self, sock, **_kwargs):
        self._sock = sock
        self.queue = [frame(probe.READY)]
        self.verified = 0
        self.closed = self.close_attempted = False
        self.sequence = 0
        self.request = None
        sock.sendmsg.side_effect = self.sent
        sock.recvmsg.return_value = (b"", [], 0, None)

    def sent(self, buffers):
        raw = buffers[0]
        value = json.loads(raw[4:])
        kind = value.get("kind")
        if kind == "initialize":
            self.initial = value
            self.queue.append(probe.protocol.encode_frame({**value, "kind": "hello"}))
        elif kind == "command":
            self.sequence += 1
            self.request = {
                "schema_version": probe.protocol.REQUEST_SCHEMA,
                "epoch": value["epoch"], "rank": value["rank"], "sequence": self.sequence,
                "token": self.initial["token"], "operation": value["operation"],
                "arguments": value["arguments"],
            }
            self.queue.append(probe.protocol.encode_frame(self.request))
        elif value["schema_version"] == probe.protocol.ACK_SCHEMA:
            self.queue.append(probe.protocol.encode_frame({
                **self.request, "operation": "confirm_ack",
                "arguments": {"ack_sha256": hashlib.sha256(raw).hexdigest()},
            }))
        elif kind == "stop":
            self.queue.append(probe.protocol.encode_frame({**value, "kind": "bye"}))
        return len(raw)

    def receive(self):
        self.verified += 1
        return self.queue.pop(0)

    def close(self):
        if self.close_attempted:
            if not self.closed:
                raise probe.credentials.PeerCredentialError("credential_cleanup_failed")
            return
        self.close_attempted = True
        self._sock.close()
        self.closed = True

    def summary(self):
        return {"verified_messages": self.verified, "received_descriptors_closed": 0}


class BudgetAndReadinessTests(unittest.TestCase):
    def setUp(self):
        self.clock = self.enterContext(patch.object(probe.time, "monotonic", return_value=10))

    def reader(self, snapshot=None):
        reader = Mock(attempts=1, closed=1, cleanup_failed=False)
        reader.child.return_value = snapshot if snapshot is not None else {
            "filter_mode_snapshot": True, "capability_bounding_set_empty": False,
        }
        return reader

    def assert_code(self, code, operation):
        with self.assertRaises(probe.IntegrationError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)

    def peer(self):
        owner = Mock(assert_live=Mock())
        channel = Mock()
        channel._sock.sendmsg.side_effect = lambda parts: len(parts[0])
        return probe.Peer(owner, channel, Mock(), 0, EPOCH, probe.Budget(), 30)

    def test_fixed_budget_and_readiness_contract_keeps_frozen_base(self):
        self.assertTrue(issubclass(probe.Budget, probe.authenticated.Budget))
        self.assertTrue(issubclass(probe.Peer, probe.authenticated.Peer))
        self.assertIs(probe.previous, probe.authenticated.previous)
        self.assertEqual(probe.READY, b"guarded-ledger-ready-v1")
        self.assertEqual(probe.FAULTS, probe.authenticated.FAULTS | {"guard_exit_before_ready"})
        self.assertEqual((probe.MAXIMUM_RECEIVES, probe.MAXIMUM_STATUS_READS), (64, 64))

    def test_receive_budget_is_cumulative_across_epochs_and_never_refunds(self):
        budget = probe.Budget()
        for _epoch in range(4):
            for _rank in range(2):
                for _message in range(8):
                    budget.receive()
        for _attempt in range(2):
            # The inherited operation retains its frozen exception type.
            with self.assertRaises(probe.authenticated.IntegrationError) as caught:
                budget.receive()
            self.assertEqual(caught.exception.code, "budget_exhausted")
        self.assertEqual(budget.receives, 64)
        self.assertEqual(budget.summary()["receive_capacity_bytes"],
                         64 * probe.credentials.RECEIVE_CAPACITY_BYTES)

    def test_64_status_capacities_are_precharged_across_fresh_guard_readers(self):
        budget = probe.Budget()
        readers = []
        owner = Mock()

        def factory():
            self.assertEqual(budget.status_reservations, len(readers) + 1)
            reader = self.reader({
                "filter_mode_snapshot": True,
                "capability_bounding_set_empty": len(readers) % 2 == 0,
            })
            readers.append(reader)
            return reader

        with patch.object(probe.kernel_guard, "GuardReads", side_effect=factory) as create:
            budget.status(deadline=30)
            for _epoch in range(3):
                for _ in range(21):
                    budget.status(owner, deadline=30)
            for _attempt in range(2):
                self.assert_code("budget_exhausted", lambda: budget.status(owner, deadline=30))
        self.assertEqual(create.call_count, 64)
        readers[0].controller.assert_called_once_with()
        readers[0].child.assert_not_called()
        for reader in readers[1:]:
            reader.child.assert_called_once_with(owner, 30)
            reader.controller.assert_not_called()
        self.assertEqual((budget.status_reservations, budget.status_attempts, budget.status_closed),
                         (64, 64, 64))
        self.assertEqual((budget.snapshots, budget.filter_snapshots), (63, 63))
        self.assertEqual(budget.empty_bounding_snapshots, 31)
        self.assertEqual(budget.summary()["filter_snapshots_verified"], 63)

    def test_failed_snapshot_keeps_precharge_and_cleanup_stays_sticky(self):
        budget = probe.Budget()
        bad = self.reader()
        bad.closed, bad.cleanup_failed = 0, True
        bad.child.side_effect = probe.kernel_guard.GuardError("guard_cleanup_failed")
        with patch.object(probe.kernel_guard, "GuardReads", side_effect=[bad, self.reader()]):
            with self.assertRaises(probe.kernel_guard.GuardError):
                budget.status(Mock(), deadline=30)
            budget.status(Mock(), deadline=30)
        self.assertEqual((budget.status_reservations, budget.status_attempts, budget.status_closed),
                         (2, 2, 1))
        self.assertTrue(budget.status_cleanup_failed)
        self.assertEqual((budget.snapshots, budget.filter_snapshots), (1, 1))

    def test_expired_status_has_no_new_reader_or_capacity_reservation(self):
        budget = probe.Budget()
        with patch.object(probe.kernel_guard, "GuardReads") as create:
            with self.assertRaises(probe.authenticated.IntegrationError):
                budget.status(deadline=10)
        create.assert_not_called()
        self.assertEqual(budget.status_reservations, 0)

    def test_false_or_missing_filter_snapshot_releases_no_token(self):
        for snapshot in (
            {"filter_mode_snapshot": False, "capability_bounding_set_empty": False},
            {"capability_bounding_set_empty": False},
            {"filter_mode_snapshot": 1, "capability_bounding_set_empty": False},
        ):
            with self.subTest(snapshot=snapshot):
                peer = self.peer()
                peer.channel.receive.return_value = frame(probe.READY)
                with (
                    patch.object(probe.kernel_guard, "GuardReads",
                                 return_value=self.reader(snapshot)),
                    self.assertRaises((probe.IntegrationError, KeyError)),
                ):
                    peer.initialize(TOKEN)
                self.assertFalse(peer.guard_verified)
                peer.channel._sock.sendmsg.assert_not_called()
                peer.session.feed.assert_not_called()
                self.assertEqual(peer.budget.filter_snapshots, 0)
                self.assertEqual(peer.budget.status_reservations, 1)

    def test_wrong_ready_releases_no_token_even_with_valid_filter_snapshot(self):
        for payload in (b"", probe.authenticated.READY, probe.kernel_guard.READY, b"wrong"):
            with self.subTest(payload=payload):
                peer = self.peer()
                peer.channel.receive.return_value = frame(payload)
                with patch.object(probe.kernel_guard, "GuardReads", return_value=self.reader()):
                    self.assert_code("frame_rejected", lambda peer=peer: peer.initialize(TOKEN))
                self.assertFalse(peer.guard_verified)
                peer.channel._sock.sendmsg.assert_not_called()
                self.assertEqual(peer.budget.filter_snapshots, 1)

    def test_guard_readiness_and_snapshot_precede_initialize_token_send(self):
        peer = self.peer()
        events = []
        reader = self.reader()
        reader.child.side_effect = lambda *_args: events.append("snapshot") or {
            "filter_mode_snapshot": True, "capability_bounding_set_empty": False,
        }
        peer.channel.receive.side_effect = [
            frame(probe.READY),
            probe.protocol.encode_frame(probe.previous._control("hello", EPOCH, 0, token=TOKEN)),
        ]

        def send(parts):
            self.assertTrue(peer.guard_verified)
            self.assertEqual(peer.budget.filter_snapshots, 1)
            value = json.loads(parts[0][4:])
            self.assertEqual((value["kind"], value["token"]), ("initialize", TOKEN))
            events.append("initialize")
            return len(parts[0])

        peer.channel._sock.sendmsg.side_effect = send
        with patch.object(probe.kernel_guard, "GuardReads", return_value=reader):
            peer.initialize(TOKEN)
        self.assertEqual(events, ["snapshot", "initialize", "snapshot"])
        self.assertTrue(peer.guard_verified)


class SourceTests(unittest.TestCase):
    def sources(self):
        return tuple(
            (ROOT / relative).read_bytes() for relative, _cap, _sha in probe.SOURCE_BINDINGS
        )

    def test_exchange_body_preserves_frozen_ast_with_only_wider_module_error_codes(self):
        def exchange_body(module):
            tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
            peer = next(node for node in tree.body
                        if isinstance(node, ast.ClassDef) and node.name == "Peer")
            exchange = next(node for node in peer.body
                            if isinstance(node, ast.FunctionDef) and node.name == "exchange")
            return ast.dump(ast.Module(body=exchange.body, type_ignores=[]),
                            include_attributes=False)

        self.assertEqual(exchange_body(probe), exchange_body(probe.authenticated))
        successor_codes = probe.Peer.exchange.__globals__["ERROR_CODES"]
        frozen_codes = probe.authenticated.Peer.exchange.__globals__["ERROR_CODES"]
        self.assertIs(successor_codes, probe.ERROR_CODES)
        self.assertIs(frozen_codes, probe.authenticated.ERROR_CODES)
        self.assertIn("guard_snapshot_rejected", successor_codes)
        self.assertNotIn("guard_snapshot_rejected", frozen_codes)

    def test_fixed_source_bundle_authenticates_all_four_public_sources(self):
        self.assertEqual(tuple(item[0] for item in probe.SOURCE_BINDINGS), (
            "src/lumi/posttraining_sft_kernel_policy_v1.py",
            "scripts/sft_kernel_guard_inert_worker_v1.py",
            "scripts/sft_authenticated_ledger_worker_v1.py",
            "scripts/sft_guarded_ledger_worker_v1.py",
        ))
        self.assertEqual(probe.SOURCE_BINDINGS[:2], probe.kernel_guard.SOURCE_BINDINGS)
        self.assertEqual(probe.SOURCE_BINDINGS[2][2], probe.authenticated.WORKER_SHA256)
        sources = probe.authenticate_sources()
        self.assertIs(type(sources), tuple)
        self.assertEqual(len(sources), 4)
        for raw, (_relative, capacity, expected) in zip(
            sources, probe.SOURCE_BINDINGS, strict=True,
        ):
            self.assertIs(type(raw), bytes)
            self.assertLessEqual(len(raw), capacity)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), expected)

    def test_each_source_mutation_is_rejected_before_popen(self):
        sources = self.sources()
        endpoint = Mock()
        for index in range(4):
            mutated = list(sources)
            mutated[index] += b"\n# synthetic mutation\n"
            with self.subTest(index=index), patch.object(probe.subprocess, "Popen") as popen:
                with self.assertRaises(probe.IntegrationError) as caught:
                    probe.spawn(endpoint, "none", tuple(mutated))
                self.assertEqual(caught.exception.code, "source_rejected")
                popen.assert_not_called()
                endpoint.fileno.assert_not_called()

    def test_spawn_revalidates_bundle_type_size_and_fixed_fault(self):
        sources = self.sources()
        invalid = (
            ("none", list(sources)), ("none", sources[:3]), ("none", (*sources, b"extra")),
            ("ack_send_failure", sources), ("SECRET", sources), (True, sources),
        )
        with patch.object(probe.subprocess, "Popen") as popen:
            for fault, bundle in invalid:
                with self.subTest(fault=fault, count=len(bundle)):
                    with self.assertRaises(probe.IntegrationError):
                        probe.spawn(Mock(), fault, bundle)
            for index in range(4):
                invalid_bytes = (
                    bytearray(sources[index]), b"x" * (probe.SOURCE_BINDINGS[index][1] + 1),
                )
                for bad in invalid_bytes:
                    mutated = list(sources)
                    mutated[index] = bad
                    with self.assertRaises(probe.IntegrationError):
                        probe.spawn(Mock(), "none", tuple(mutated))
        popen.assert_not_called()

    def test_bootstrap_compiles_exact_bytes_in_four_separate_namespaces_without_execution(self):
        sources = self.sources()
        endpoint = Mock()
        endpoint.fileno.return_value = 71
        with (
            patch.object(probe.subprocess, "Popen", return_value=Mock()) as popen,
            patch.object(probe.Path, "open", side_effect=AssertionError("source reopened")),
        ):
            probe.spawn(endpoint, "none", sources)
        argv = popen.call_args.args[0]
        self.assertEqual(argv[:4], [sys.executable, "-S", "-B", "-c"])
        self.assertEqual(argv[5:], ["71", "none"])
        options = popen.call_args.kwargs
        self.assertEqual(options["pass_fds"], (71,))
        self.assertTrue(options["close_fds"])
        self.assertTrue(options["start_new_session"])
        self.assertEqual(options["env"], {"PATH": "/usr/bin:/bin", "LC_ALL": "C"})
        compiled, scopes = [], []

        def fake_compile(raw, filename, mode):
            self.assertEqual(mode, "exec")
            compiled.append(raw)
            return (raw, filename)

        def fake_exec(code, namespace):
            self.assertEqual(code[0], sources[len(scopes)])
            scopes.append(namespace)

        safe_builtins = {**vars(builtins), "compile": fake_compile, "exec": fake_exec}
        exec(argv[4], {"__builtins__": safe_builtins})
        self.assertEqual(tuple(compiled), sources)
        self.assertEqual(len({id(scope) for scope in scopes}), 4)
        self.assertTrue(all(scope["__name__"] != "__main__" for scope in scopes[:3]))
        self.assertEqual(scopes[3]["__name__"], "__main__")
        for name, scope in zip(("kernel_policy", "kernel_challenges", "ledger_worker"),
                               scopes[:3], strict=True):
            self.assertIs(scopes[3][name].__dict__, scope)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="lumi-guarded-ledger-synthetic-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.directory.chmod(0o700)

    def boundaries(self, value):
        for key in NONCLAIMS:
            self.assertIs(value[key], False, key)
        for key in ("training_source_or_data_reads", "model_or_checkpoint_reads",
                    "external_network_bytes", "gpu_use", "inference_calls"):
            self.assertEqual(value[key], 0, key)
        self.assertLessEqual(value["accounting"]["receive_capacity_reservations"], 64)
        self.assertLessEqual(value["accounting"]["status_capacity_reservations"], 64)
        self.assertLessEqual(value["accounting"]["send_bytes_precharged"], 131072)
        self.assertLessEqual(value["journal_write_bytes_charged"], 65536)
        readiness = value["guarded_worker_readiness_verified"]
        self.assertEqual(value["denied_syscall_challenges_passed"], 17 * readiness)
        self.assertEqual(value["positive_anonymous_mapping_challenges_passed"], readiness)
        self.assertEqual(value["address_space_rejection_challenges_passed"], readiness)
        self.assertIs(value["inert_child_policy_and_challenges_verified"],
                      value["state"] == "passed" and readiness == 6)
        encoded = json.dumps(value)
        for secret in (str(self.directory), "SECRET", "Traceback", "NoNewPrivs", '"token"'):
            self.assertNotIn(secret, encoded)

    def fake_environment(self, *, authenticate=True):
        self.owners, self.transcripts, self.endpoints = [], [], []
        self.enterContext(patch.object(probe.lifecycle, "require_runtime"))
        self.enterContext(patch.object(probe.kernel_policy, "require_abi"))
        self.install = self.enterContext(patch.object(probe.kernel_policy, "install"))
        self.enterContext(patch.object(probe.lifecycle, "normalize_child_signal"))
        self.enterContext(patch.object(probe.socket, "MSG_CMSG_CLOEXEC", 1 << 30, create=True))
        self.clock = self.enterContext(patch.object(probe.time, "monotonic", return_value=100))
        if authenticate:
            self.enterContext(patch.object(probe, "authenticate_sources", return_value=(
                b"synthetic policy", b"synthetic challenges",
                b"synthetic ledger", b"synthetic guard",
            )))
        self.reader = Mock(attempts=1, closed=1, cleanup_failed=False)
        self.reader.child.return_value = {
            "filter_mode_snapshot": True, "capability_bounding_set_empty": False,
        }
        self.readers = self.enterContext(patch.object(probe.kernel_guard, "GuardReads",
                                                     return_value=self.reader))

        def pair():
            parent, endpoint = Mock(), Mock()
            self.endpoints.append((parent, endpoint))
            return parent, endpoint

        def owner(child):
            item = Mock(child=child, fd=71, owned=True, bound_live=True, reaped=False,
                        closed=False, close_attempted=False, signal_attempts=0, assert_live=Mock())
            item.reap.side_effect = lambda *_a, **_kw: setattr(item, "reaped", True)
            item.terminate_and_reap.side_effect = lambda *_a: setattr(item, "reaped", True)

            def close():
                item.close_attempted = True
                item.closed = True
            item.close.side_effect = close
            self.owners.append(item)
            return item

        def channel(sock, **kwargs):
            item = Transcript(sock, **kwargs)
            self.transcripts.append(item)
            return item

        self.spawn = self.enterContext(patch.object(probe, "spawn", return_value=Mock(pid=12345)))
        self.owner_factory = self.enterContext(patch.object(probe.lifecycle, "OwnedChild",
                                                            side_effect=owner))
        self.pair = self.enterContext(patch.object(probe.credentials, "credential_pair",
                                                 side_effect=pair))
        self.channel_factory = self.enterContext(patch.object(
            probe.credentials, "CredentialChannel", side_effect=channel,
        ))

    def run_fake(self, **kwargs):
        value = probe.run_synthetic_probe(self.directory, **kwargs)
        self.boundaries(value)
        self.install.assert_not_called()
        return value

    def assert_success(self, value):
        self.assertEqual(value["state"], "passed", value["failure_code"])
        for key in (
            "worker_cleanup_complete", "authenticated_inert_ledger_probe_passed",
            "guarded_inert_ledger_probe_passed", "inert_child_policy_and_challenges_verified",
        ):
            self.assertTrue(value[key], key)
        expected = {
            "worker_processes_started": 6, "worker_processes_reaped": 6, "pidfds_bound_live": 6,
            "pidfds_closed": 6, "epochs_completed": 3, "global_optimizer_steps_completed": 4,
            "requests_predebited": 18, "acknowledgements_confirmed": 18,
            "credential_messages_verified": 54, "terminal_eof_verified": 6,
            "checkpoint_bytes_synthetically_reserved": 6, "journal_capacity_predebited": 65536,
            "journal_records_committed": 22, "received_descriptors_closed": 0,
            "guarded_worker_readiness_verified": 6, "denied_syscall_challenges_passed": 102,
        }
        for key, count in expected.items():
            self.assertEqual(value[key], count, key)
        self.assertEqual(value["journal_terminal_event"], "attempt_completed")
        self.assertTrue(value["journal_closed"])
        accounting = value["accounting"]
        for key, count in {
            "receive_capacity_reservations": 60, "status_read_attempts": 55,
            "status_descriptors_closed": 55, "capability_snapshots_verified": 54,
            "filter_snapshots_verified": 54, "status_capacity_reservations": 55,
            "status_capacity_bytes": 55 * probe.lifecycle.STATUS_CAPACITY,
        }.items():
            self.assertEqual(accounting[key], count, key)
        journal = self.directory / "attempt.journal"
        raw = journal.read_bytes()
        self.assertEqual(len(raw), value["journal_write_bytes_charged"])
        self.assertIsNotNone(probe.previous.journal_module.verify_bytes(raw))
        self.assertEqual(journal.stat().st_mode & 0o777, 0o600)
        for secret in (b"token", b"rank", b"pid", b"path", b"exception", b"NoNewPrivs"):
            self.assertNotIn(secret, raw)

    @isolated
    def test_full_synthetic_transcript_keeps_all_frozen_ledger_totals(self):
        self.fake_environment()
        self.assert_success(self.run_fake())
        self.assertEqual(self.spawn.call_count, 6)
        for owner in self.owners:
            owner.bind.assert_called_once()
            owner.reap.assert_called_once()
            owner.close.assert_called_once()
            owner.child.wait.assert_not_called()
        for parent, endpoint in self.endpoints:
            parent.close.assert_called_once()
            endpoint.close.assert_called_once()

    @isolated
    def test_missing_filter_fails_before_first_token_and_never_retries_worker(self):
        self.fake_environment()
        self.reader.child.return_value = {"capability_bounding_set_empty": False}
        value = self.run_fake()
        self.assertEqual(value["state"], "failed")
        self.assertEqual(value["failure_code"], "integration_failed")
        self.assertEqual(value["guarded_worker_readiness_verified"], 0)
        self.assertEqual(value["worker_processes_started"], 1)
        self.assertEqual(value["requests_predebited"], 0)
        self.spawn.assert_called_once()
        self.transcripts[0]._sock.sendmsg.assert_not_called()

    @isolated
    def test_old_ready_is_rejected_without_token_or_guard_challenge_credit(self):
        self.fake_environment()
        original = self.channel_factory.side_effect

        def wrong_ready(sock, **kwargs):
            channel = original(sock, **kwargs)
            channel.queue = [frame(probe.authenticated.READY)]
            return channel

        self.channel_factory.side_effect = wrong_ready
        value = self.run_fake()
        self.assertEqual(value["failure_code"], "frame_rejected")
        self.assertEqual(value["guarded_worker_readiness_verified"], 0)
        self.assertEqual(value["denied_syscall_challenges_passed"], 0)
        self.spawn.assert_called_once()
        self.transcripts[0]._sock.sendmsg.assert_not_called()

    @isolated
    def test_epoch_commit_waits_for_all_reaps_eof_and_handle_closes(self):
        self.fake_environment()
        original = probe.previous.DurableSupervisor.commit_retired_epoch

        def commit(controller):
            self.assertTrue(all(owner.reaped and owner.closed for owner in self.owners))
            self.assertTrue(all(item.closed for item in self.transcripts))
            self.assertTrue(all(parent.recvmsg.call_count == 1 for parent, _ in self.endpoints))
            original(controller)

        with patch.object(probe.previous.DurableSupervisor, "commit_retired_epoch", commit):
            self.assert_success(self.run_fake())

    @isolated
    def test_exclusive_attempt_marker_is_not_reused_or_overwritten(self):
        self.fake_environment()
        marker = self.directory / "attempt.journal"
        marker.write_bytes(b"synthetic consumed attempt")
        for _attempt in range(2):
            value = self.run_fake()
            self.assertEqual(value["state"], "failed")
        self.spawn.assert_not_called()
        self.assertEqual(marker.read_bytes(), b"synthetic consumed attempt")

    @isolated
    def test_ack_failure_keeps_debit_and_stops_without_replacement_or_next_epoch(self):
        self.fake_environment()
        value = self.run_fake(fault="ack_send_failure")
        self.assertEqual(value["state"], "failed")
        self.assertEqual(value["checkpoint_bytes_synthetically_reserved"], 1)
        self.assertEqual(value["requests_predebited"], 1)
        self.assertEqual(value["acknowledgements_confirmed"], 0)
        self.assertEqual(value["epochs_completed"], 0)
        self.assertEqual(self.spawn.call_count, 2)
        self.assertEqual(value["journal_terminal_event"], "attempt_failed")

    @isolated
    def test_reservation_fsync_failure_prevents_ack_send(self):
        self.fake_environment()
        original = probe.previous.journal_module.os.fsync
        calls = []

        def reject(descriptor):
            calls.append(1)
            if len(calls) == 4:
                raise OSError("SECRET")
            return original(descriptor)

        with patch.object(probe.previous.journal_module.os, "fsync", reject):
            value = self.run_fake()
        self.assertEqual(len(calls), 4)
        self.assertEqual(value["state"], "failed")
        self.assertEqual(value["checkpoint_bytes_synthetically_reserved"], 1)
        self.assertEqual(value["acknowledgements_confirmed"], 0)
        self.assertEqual(self.transcripts[0]._sock.sendmsg.call_count, 2)

    @isolated
    def test_cleanup_precedes_failure_journal_append(self):
        self.fake_environment()
        original = probe.previous.journal_module.AttemptJournal.append

        def append(journal, event, counters=()):
            if event == "attempt_failed":
                self.assertTrue(all(owner.reaped and owner.closed for owner in self.owners))
                self.assertTrue(all(item.closed for item in self.transcripts))
            return original(journal, event, counters)

        with patch.object(probe.previous.journal_module.AttemptJournal, "append", append):
            self.assertEqual(self.run_fake(fault="ack_send_failure")["state"], "failed")

    @isolated
    def test_late_journal_close_cannot_convert_terminal_completion_into_guarded_pass(self):
        self.fake_environment()
        original = probe.previous.journal_module.AttemptJournal.close

        def late(journal):
            original(journal)
            self.clock.return_value = 121

        with patch.object(probe.previous.journal_module.AttemptJournal, "close", late):
            value = self.run_fake()
        self.assertEqual(value["journal_terminal_event"], "attempt_completed")
        self.assertEqual(value["epochs_completed"], 3)
        self.assertEqual(value["state"], "failed")
        self.assertEqual(value["failure_code"], "deadline_exhausted")
        self.assertFalse(value["guarded_inert_ledger_probe_passed"])
        self.assertFalse(value["authenticated_inert_ledger_probe_passed"])

    @isolated
    def test_journal_close_failure_denies_guarded_pass_after_all_three_epochs(self):
        self.fake_environment()
        original = probe.previous.journal_module.AttemptJournal.close

        def failed(journal):
            # Close the real temporary fixture descriptor before injecting failure.
            original(journal)
            raise OSError("SECRET")

        with patch.object(probe.previous.journal_module.AttemptJournal, "close", failed):
            value = self.run_fake()
        self.assertEqual(value["epochs_completed"], 3)
        self.assertEqual(value["state"], "failed")
        self.assertEqual(value["failure_code"], "journal_rejected")
        self.assertFalse(value["guarded_inert_ledger_probe_passed"])
        self.assertFalse(value["authenticated_inert_ledger_probe_passed"])
        self.assertIn(b'"event":"attempt_completed"',
                      (self.directory / "attempt.journal").read_bytes())

    @isolated
    def test_pidfd_close_failure_is_not_retried_and_prevents_epoch_commit(self):
        self.fake_environment()
        original = self.owner_factory.side_effect

        def owner(child):
            item = original(child)

            def close():
                item.close_attempted = True
                raise OSError("SECRET")

            item.close.side_effect = close
            return item

        self.owner_factory.side_effect = owner
        with patch.object(probe.previous.DurableSupervisor, "commit_retired_epoch") as commit:
            value = self.run_fake()
        self.assertEqual(value["failure_code"], "cleanup_failed")
        self.assertEqual(value["epochs_completed"], 0)
        self.assertEqual(value["pidfds_closed"], 0)
        self.assertFalse(value["worker_cleanup_complete"])
        commit.assert_not_called()
        for item in self.owners:
            self.assertTrue(item.close_attempted)
            item.close.assert_called_once()

    @isolated
    def test_endpoint_close_failure_is_sticky_without_retry(self):
        self.fake_environment()
        original = self.pair.side_effect

        def pair():
            parent, endpoint = original()
            endpoint.close.side_effect = OSError("SECRET")
            return parent, endpoint

        self.pair.side_effect = pair
        value = self.run_fake()
        self.assertEqual(value["failure_code"], "cleanup_failed")
        self.assertFalse(value["worker_cleanup_complete"])
        self.assertEqual(value["epochs_completed"], 0)
        for parent, endpoint in self.endpoints:
            parent.close.assert_called_once()
            endpoint.close.assert_called_once()

    @isolated
    def test_failed_confirmation_descriptor_cleanup_cannot_be_hidden(self):
        self.fake_environment()
        original = Transcript.receive

        def receive(channel):
            if channel.verified == 3:
                raise probe.credentials.PeerCredentialError("credential_cleanup_failed")
            return original(channel)

        with patch.object(Transcript, "receive", receive):
            value = self.run_fake()
        self.assertEqual(value["failure_code"], "cleanup_failed")
        self.assertFalse(value["worker_cleanup_complete"])
        self.assertEqual(value["checkpoint_bytes_synthetically_reserved"], 1)
        self.assertEqual(value["acknowledgements_confirmed"], 0)

    @isolated
    def test_guard_confirmation_failure_preserves_specific_code_and_predebit(self):
        self.fake_environment()

        def snapshot(*_args):
            if self.transcripts and self.transcripts[0].verified == 4:
                raise probe.kernel_guard.GuardError("guard_snapshot_rejected")
            return {"filter_mode_snapshot": True, "capability_bounding_set_empty": False}

        self.reader.child.side_effect = snapshot
        value = self.run_fake()
        self.assertEqual(value["failure_code"], "guard_snapshot_rejected")
        self.assertEqual(value["checkpoint_bytes_synthetically_reserved"], 1)
        self.assertEqual(value["requests_predebited"], 1)
        self.assertEqual(value["acknowledgements_confirmed"], 0)
        self.assertEqual(value["epochs_completed"], 0)
        self.assertEqual(value["journal_terminal_event"], "attempt_failed")
        self.assertEqual(self.spawn.call_count, 2)

    @isolated
    def test_source_mutation_denies_before_journal_or_spawn(self):
        self.fake_environment(authenticate=False)
        original = probe.Path.open
        for index, (relative, _capacity, _expected) in enumerate(probe.SOURCE_BINDINGS):
            target = ROOT / relative

            def opened(path, *args, target=target, **kwargs):
                if path == target:
                    return io.BytesIO(b"synthetic source mutation")
                return original(path, *args, **kwargs)

            with self.subTest(index=index), patch.object(probe.Path, "open", autospec=True,
                                                        side_effect=opened):
                value = self.run_fake()
            self.assertEqual(value["failure_code"], "source_rejected")
            self.spawn.assert_not_called()
            self.assertEqual(list(self.directory.iterdir()), [])

    @isolated
    def test_unsupported_mac_denies_before_reads_sources_spawn_or_journal(self):
        with (
            patch.object(probe.lifecycle.sys, "platform", "darwin"),
            patch.object(probe.kernel_policy, "require_abi") as abi,
            patch.object(probe.kernel_policy, "install") as install,
            patch.object(probe.Budget, "status") as status,
            patch.object(probe, "authenticate_sources") as source,
            patch.object(probe, "spawn") as spawn,
            patch.object(probe.previous, "DurableSupervisor") as controller,
        ):
            value = probe.run_synthetic_probe(self.directory)
        self.boundaries(value)
        self.assertEqual(value["state"], "unsupported")
        for operation in (abi, install, status, source, spawn, controller):
            operation.assert_not_called()
        self.assertEqual(list(self.directory.iterdir()), [])
        self.assertEqual(value["accounting"]["status_read_attempts"], 0)

    @isolated
    def test_invalid_options_are_rejected_before_runtime(self):
        with patch.object(probe.lifecycle, "require_runtime") as runtime:
            for options in ({"timeout_seconds": True}, {"timeout_seconds": 0},
                            {"timeout_seconds": 21}, {"fault": "SECRET"}):
                value = probe.run_synthetic_probe(self.directory, **options)
                self.boundaries(value)
                self.assertEqual(value["failure_code"], "configuration_rejected")
            runtime.assert_not_called()
        self.assertEqual(list(self.directory.iterdir()), [])

    def live(self, fault="none"):
        value = probe.run_synthetic_probe(self.directory, fault=fault,
                                          timeout_seconds=1 if fault == "stall" else 20)
        self.boundaries(value)
        if sys.platform != "linux":
            self.assertEqual(value["state"], "unsupported")
            self.assertEqual(value["accounting"]["status_read_attempts"], 0)
            self.assertEqual(value["worker_processes_started"], 0)
            self.assertEqual(list(self.directory.iterdir()), [])
            return None
        self.assertTrue(value["worker_cleanup_complete"], value["failure_code"])
        self.assertEqual(value["worker_processes_started"], value["worker_processes_reaped"])
        self.assertEqual(value["worker_processes_started"], value["pidfds_closed"])
        return value

    def failed_live(self, fault, debit, workers=2):
        value = self.live(fault)
        if value is not None:
            self.assertEqual(value["state"], "failed")
            self.assertEqual(value["epochs_completed"], 0)
            self.assertEqual(value["worker_processes_started"], workers)
            self.assertEqual(value["checkpoint_bytes_synthetically_reserved"], debit)
            self.assertEqual(value["journal_terminal_event"], "attempt_failed")
            self.assertFalse(value["guarded_inert_ledger_probe_passed"])
            raw = (self.directory / "attempt.journal").read_bytes()
            self.assertNotIn(b'"event":"epoch_completed"', raw)
            self.assertNotIn(b'"event":"attempt_completed"', raw)
        return value

    @isolated
    def test_live_linux_three_guarded_epochs_or_nonlinux_denial(self):
        value = self.live()
        if value is not None:
            self.assert_success(value)

    @isolated
    def test_live_guard_exit_before_ready_releases_no_token_and_does_not_retry(self):
        value = self.failed_live("guard_exit_before_ready", 0, workers=1)
        if value is not None:
            self.assertEqual(value["guarded_worker_readiness_verified"], 0)
            self.assertEqual(value["accounting"]["send_bytes_precharged"], 0)
            self.assertEqual(value["requests_predebited"], 0)

    @isolated
    def test_live_wrong_epoch_rejected_before_debit(self):
        self.failed_live("wrong_epoch", 0)

    @isolated
    def test_live_wrong_rank_rejected_before_debit(self):
        self.failed_live("wrong_rank", 0)

    @isolated
    def test_live_rights_closed_and_rejected_before_debit(self):
        value = self.failed_live("rights", 0)
        if value is not None:
            self.assertEqual(value["received_descriptors_closed"], 1)

    @isolated
    def test_live_ack_send_failure_never_refunds(self):
        self.failed_live("ack_send_failure", 1)

    @isolated
    def test_live_exit_before_confirmation_never_refunds(self):
        value = self.failed_live("exit_before_confirmation", 1)
        if value is not None:
            self.assertEqual(value["acknowledgements_confirmed"], 0)

    @isolated
    def test_live_stall_stops_without_replacement(self):
        start = time.monotonic()
        self.failed_live("stall", 0)
        self.assertLess(time.monotonic() - start, 5)

    @isolated
    def test_live_nonzero_exit_blocks_epoch_commit(self):
        self.failed_live("bad_exit", 2)

    @isolated
    def test_live_trailing_packet_after_bye_blocks_epoch_commit(self):
        self.failed_live("trailing_after_bye", 2)


if __name__ == "__main__":
    unittest.main()
