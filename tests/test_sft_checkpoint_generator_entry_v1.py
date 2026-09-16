"""Pure/mocked fixed-entry tests; success packet stays only in private test memory."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import stat
import sys
import threading
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

from tests.test_sft_checkpoint_generator_packet_v1 import generated_fixture
from tests.test_sft_checkpoint_generator_packet_v1 import m as codec

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_checkpoint_generator_v1.py"
spec = importlib.util.spec_from_file_location("_lumi_generator_entry_tests_v1", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
generation_module = importlib.import_module(
    codec.__package__ + ".posttraining_checkpoint_prepared_generation_v1")


def runtime_context():
    stack = ExitStack()
    system = types.SimpleNamespace(platform="linux", version_info=(3, 12, 0),
        implementation=types.SimpleNamespace(name="cpython"),
        flags=types.SimpleNamespace(isolated=1, ignore_environment=1, no_user_site=1, optimize=0),
        dont_write_bytecode=True, byteorder="little", modules={})
    operating = types.SimpleNamespace(uname=lambda: types.SimpleNamespace(machine="x86_64"),
        getresuid=lambda: (10001, 10001, 10001), getresgid=lambda: (10001, 10001, 10001),
        getgroups=lambda: [])
    # Model this synthetic runtime's thread count, not the surrounding test process.
    # Replacing the entry's module reference never patches global threading functions.
    threads = types.SimpleNamespace(current_thread=threading.current_thread,
        main_thread=threading.main_thread, active_count=lambda: 1)
    stack.enter_context(patch.object(m, "sys", system))
    stack.enter_context(patch.object(m, "os", operating))
    stack.enter_context(patch.object(m, "threading", threads))
    return stack, system, operating


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.environment = m.FIXED_ENV | {
            "LUMI_GENERATOR_ENTRY_SHA256": m.digest(b"entry"),
            "LUMI_IMPORT_SOURCE_MANIFEST_SHA256": "b" * 64,
            "LUMI_GENERATION_NONCE_SHA256": "9" * 64}
        self.generated = generated_fixture()
        self.generation = MagicMock(
            PreparedGenerationError=generation_module.PreparedGenerationError)
        self.generation.generate_prepared_inputs.return_value = self.generated
        self.helper = MagicMock()
        files = dict.fromkeys(m.REQUIRED_SOURCES, b"# synthetic source\n")
        size = sum(map(len, files.values()))
        source_report = {"source_manifest_sha256": "b" * 64,
            "source_files_verified": len(files), "source_bytes_verified": size,
            "source_verification_read_bytes": size + 1000}
        copied = {"private_source_copy_files": len(files),
            "private_source_copy_bytes_reserved": size, "private_source_copy_readback_bytes": size,
            "private_source_copies_rootfs_immutable": False,
            "hostile_owner_code_immutability_verified": False}
        self.helper.verify_source_tree.return_value = (files, source_report)
        self.helper.stage_private_sources.return_value = (Path("/tmp/generated-code/src"), copied)
        self.writes = []

    def patches(self):
        stack = ExitStack()
        stack.enter_context(patch.dict(m.os.environ, self.environment, clear=True))
        self.runtime = stack.enter_context(patch.object(m, "runtime_check"))
        self.read = stack.enter_context(patch.object(m, "root_read", return_value=b"entry"))
        self.loader = stack.enter_context(patch.object(m, "load_helpers", return_value=self.helper))
        self.install = stack.enter_context(patch.object(m, "install_package",
                                                       return_value=(self.generation, codec)))
        self.root = stack.enter_context(patch.object(m.tempfile, "mkdtemp",
                                             return_value="/tmp/lumi-generated-input-owned"))
        self.chmod = stack.enter_context(patch.object(Path, "chmod"))
        stack.enter_context(patch.object(m.os, "dup", return_value=99))
        self.redirect = stack.enter_context(patch.object(m.os, "dup2"))
        self.closed = stack.enter_context(patch.object(m.os, "close"))
        self.write = stack.enter_context(patch.object(m.os, "write",
            side_effect=lambda fd, raw: self.writes.append((fd, raw)) or len(raw)))
        return stack

    def test_one_generation_private_packet_accepted_by_actual_codec(self):
        with self.patches():
            self.assertEqual(m.main(), 0)
        self.generation.generate_prepared_inputs.assert_called_once_with(
            Path("/tmp/lumi-generated-input-owned"), generation_nonce_sha256="9" * 64)
        self.helper.stage_private_sources.assert_called_once()
        self.assertEqual(self.helper.stage_private_sources.call_args.args[0], Path("/tmp"))
        self.root.assert_called_once_with(prefix="lumi-generated-input-", dir="/tmp")
        self.chmod.assert_called_once_with(0o700)
        self.assertEqual(len(self.writes), 1)
        descriptor, raw = self.writes[0]
        self.assertEqual(descriptor, 1)
        private = codec.decode_generated_packet(raw,
            expected_entry_sha256=self.environment["LUMI_GENERATOR_ENTRY_SHA256"],
            expected_source_manifest_sha256="b" * 64, expected_generation_nonce_sha256="9" * 64)
        self.assertTrue(private["metadata"] == self.generated["metadata"])
        self.assertTrue(private["generated_inputs"] == self.generated["generated_inputs"])
        self.assertFalse(private["receipt"]["raw_packet_public"])
        self.assertEqual([call.args for call in self.redirect.call_args_list], [(2, 1), (99, 1)])
        self.closed.assert_called_once_with(99)

    def test_native_print_stream_redirected_before_generation(self):
        def generate(*_args, **_kwargs):
            self.assertEqual(self.redirect.call_args.args, (2, 1))
            return self.generated
        self.generation.generate_prepared_inputs.side_effect = generate
        with self.patches():
            self.assertEqual(m.main(), 0)

    def test_runtime_refusal_precedes_every_read_and_source_stage(self):
        with self.patches():
            self.runtime.side_effect = m.GeneratorEntryError("runtime_rejected")
            self.assertEqual(m.main(), 1)
        self.read.assert_not_called()
        self.loader.assert_not_called()
        self.generation.generate_prepared_inputs.assert_not_called()
        self.assertEqual(json.loads(self.writes[0][1]), m._failure("runtime_rejected"))

    def test_wrong_entry_pin_rejected_before_helper_source_or_generation(self):
        self.environment["LUMI_GENERATOR_ENTRY_SHA256"] = "0" * 64
        with self.patches():
            self.assertEqual(m.main(), 1)
        self.loader.assert_not_called()
        self.helper.verify_source_tree.assert_not_called()
        self.generation.generate_prepared_inputs.assert_not_called()
        self.assertEqual(json.loads(self.writes[0][1])["code"], "entry_binding_rejected")

    def test_bad_source_or_missing_codec_rejected_before_writable_stage(self):
        self.helper.verify_source_tree.return_value = ({"lumi/__init__.py": b""}, {})
        with self.patches():
            self.assertEqual(m.main(), 1)
        self.helper.stage_private_sources.assert_not_called()
        self.install.assert_not_called()
        self.assertEqual(json.loads(self.writes[0][1])["code"], "source_binding_rejected")

    def test_source_staging_failure_never_enters_generation(self):
        self.helper.stage_private_sources.side_effect = OSError("private source path")
        with self.patches():
            self.assertEqual(m.main(), 1)
        self.install.assert_not_called()
        self.root.assert_not_called()
        self.assertEqual(json.loads(self.writes[0][1])["code"], "source_staging_failed")

    def test_generation_failure_is_redacted_and_never_retried(self):
        self.generation.generate_prepared_inputs.side_effect = ValueError("private generated text")
        with self.patches():
            self.assertEqual(m.main(), 1)
        self.generation.generate_prepared_inputs.assert_called_once()
        self.assertEqual(json.loads(self.writes[0][1])["code"], "generation_failed")
        self.assertNotIn(b"private generated text", self.writes[0][1])

    def test_frozen_generator_deadline_keeps_precise_allowlisted_code(self):
        self.generation.generate_prepared_inputs.side_effect = \
            generation_module.PreparedGenerationError("deadline_exhausted")
        with self.patches():
            self.assertEqual(m.main(), 1)
        self.assertEqual(json.loads(self.writes[0][1])["code"], "deadline_exhausted")

    def test_oversized_packet_never_emitted_and_no_second_generation(self):
        with self.patches(), patch.object(codec, "encode_generated_packet",
                                         return_value=b"x" * 65537):
            self.assertEqual(m.main(), 1)
        self.generation.generate_prepared_inputs.assert_called_once()
        self.assertEqual(len(self.writes), 1)
        self.assertEqual(json.loads(self.writes[0][1])["code"], "packet_rejected")

    def test_short_private_write_returns_two_without_retry(self):
        with self.patches():
            self.write.side_effect = lambda fd, raw: self.writes.append((fd, raw)) or len(raw) - 1
            self.assertEqual(m.main(), 2)
        self.write.assert_called_once()
        self.generation.generate_prepared_inputs.assert_called_once()

    def test_fixed_runtime_environment_and_no_argv_input_configuration(self):
        stack, system, operating = runtime_context()
        with stack:
            m.runtime_check([str(m.ENTRY), "--synthetic-only"], self.environment)
            for extra in ("HOME", "PYTHONPATH", "CUDA_DEVICE", "LUMI_APPROVED"):
                with self.subTest(extra=extra), self.assertRaises(m.GeneratorEntryError):
                    m.runtime_check([str(m.ENTRY), "--synthetic-only"],
                                    self.environment | {extra: "value"})
            system.platform = "darwin"
            with self.assertRaises(m.GeneratorEntryError):
                m.runtime_check([str(m.ENTRY), "--synthetic-only"], self.environment)
            system.platform = "linux"
            operating.getgroups = lambda: [10001]
            with self.assertRaises(m.GeneratorEntryError):
                m.runtime_check([str(m.ENTRY), "--synthetic-only"], self.environment)
        for argv in ([str(m.ENTRY)], [str(m.ENTRY), "--config", "/inputs/model"],
                     [str(m.ENTRY), "--synthetic-only", "--approved"]):
            with self.subTest(argv=argv), self.assertRaises(m.GeneratorEntryError):
                m.runtime_check(argv, self.environment)

    def test_zero_nonce_rejected_before_io(self):
        stack, _system, _operating = runtime_context()
        with stack, self.assertRaises(m.GeneratorEntryError):
            m.runtime_check([str(m.ENTRY), "--synthetic-only"],
                self.environment | {"LUMI_GENERATION_NONCE_SHA256": "0" * 64})

    def test_runtime_thread_count_model_preserves_extra_thread_rejection(self):
        real_count = threading.active_count
        stack, _system, _operating = runtime_context()
        with stack:
            self.assertIsNot(m.threading, threading)
            self.assertIs(m.threading.current_thread, threading.current_thread)
            self.assertIs(m.threading.main_thread, threading.main_thread)
            self.assertIs(threading.active_count, real_count)
            with patch.object(m.threading, "active_count", return_value=2):
                with self.assertRaisesRegex(m.GeneratorEntryError, "^runtime_rejected$"):
                    m.runtime_check([str(m.ENTRY), "--synthetic-only"], self.environment)
                self.assertIs(threading.active_count, real_count)
        self.assertIs(m.threading, threading)
        self.assertIs(threading.active_count, real_count)

    def test_runtime_thread_model_preserves_off_main_rejection(self):
        real_current = threading.current_thread
        stack, _system, _operating = runtime_context()
        with stack, patch.object(m.threading, "current_thread", return_value=object()):
            with self.assertRaisesRegex(m.GeneratorEntryError, "^runtime_rejected$"):
                m.runtime_check([str(m.ENTRY), "--synthetic-only"], self.environment)
            self.assertIs(threading.current_thread, real_current)
        self.assertIs(m.threading, threading)

    def test_runtime_context_isolated_from_owned_background_test_thread(self):
        release = threading.Event()
        worker = threading.Thread(target=lambda: release.wait(5),
                                  name="lumi-entry-synthetic-test-thread")
        worker.start()
        try:
            self.assertTrue(worker.is_alive())
            self.assertGreaterEqual(threading.active_count(), 2)
            stack, _system, _operating = runtime_context()
            with stack:
                m.runtime_check([str(m.ENTRY), "--synthetic-only"], self.environment)
        finally:
            release.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertIs(m.threading, threading)

    def test_helper_pin_matches_prior_bytes_and_wrong_bytes_never_execute(self):
        raw = (ROOT / "scripts/run_checkpoint_guarded_import_v1.py").read_bytes()
        self.assertEqual((len(raw), m.digest(raw)), (m.HELPER_BYTES, m.HELPER_SHA256))
        with patch.object(m, "root_read", return_value=b"raise ValueError('private')"), \
                patch("builtins.exec") as executed, self.assertRaisesRegex(
                    m.GeneratorEntryError, "helper_binding_rejected"):
            m.load_helpers()
        executed.assert_not_called()

    def test_root_control_owner_mode_and_hardlink_checks(self):
        for uid, mode, links in ((10001, 0o444, 1), (0, 0o666, 1), (0, 0o444, 2)):
            info = types.SimpleNamespace(st_mode=stat.S_IFREG | mode, st_uid=uid,
                                         st_gid=uid, st_nlink=links)
            with self.subTest(uid=uid, mode=mode, links=links), \
                    self.assertRaises(m.GeneratorEntryError):
                m._owned(info)

    def test_inert_lumi_namespace_precedes_generator_runtime_no_init_execution(self):
        modules = {key: value for key, value in sys.modules.items()
                   if key.split(".", 1)[0] not in m.ML_ROOTS | {"lumi"}}
        with patch.dict(sys.modules, modules, clear=True), \
                patch.object(m.importlib, "import_module", side_effect=[self.generation, codec]) \
                    as imported:
            m.install_package(Path("/tmp/generated-code/src"))
            self.assertEqual(sys.modules["lumi"].__path__, ["/tmp/generated-code/src/lumi"])
            self.assertIsNone(sys.modules["lumi"].__spec__.loader)
        self.assertEqual([call.args[0] for call in imported.call_args_list], [
            "lumi.posttraining_checkpoint_prepared_generation_v1",
            "lumi.posttraining_checkpoint_generator_packet_v1"])

    def test_one_fixed_generator_call_no_importer_or_external_actions(self):
        tree = ast.parse(SCRIPT.read_bytes())
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)]
        self.assertEqual(calls.count("generate_prepared_inputs"), 1)
        self.assertEqual(calls.count("write"), 1)
        for name in ("run_import", "run_tiny_fixture", "run_synthetic_import",
                     "_coordinate_admitted_import", "Popen", "fork", "execve"):
            self.assertNotIn(name, calls)
        self.assertIn("lumi/posttraining_checkpoint_generator_packet_v1.py", m.REQUIRED_SOURCES)
        self.assertIn("lumi/posttraining_checkpoint_prepared_generation_v1.py", m.REQUIRED_SOURCES)


if __name__ == "__main__":
    unittest.main()
