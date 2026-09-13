"""Bound source, fake daemon and tiny generated tar tests; never native Linux."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tests import test_sft_checkpoint_guard_qualification_v1 as qualification_doubles
from tests.test_sft_checkpoint_guard_composition_v1 import environment

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/run_checkpoint_guard_qualification_v1.py"
spec = importlib.util.spec_from_file_location("_synthetic_import_qualification_harness", PATH)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def case_report(case):
    # Build the genuine case API's shape entirely through existing syscall doubles.
    test = qualification_doubles.QualificationTests("test_exact_case_names")
    test.setUp()
    try:
        test.api(case)
        return test.invoke(case)
    finally:
        test.doCleanups()


class SourcesTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(subject.sys, "meta_path", list(sys.meta_path)))

    def test_every_admitted_source_pin_matches_and_total_is_bounded(self):
        raw = (ROOT / subject.SOURCE_ADMISSION).read_bytes()
        self.assertEqual(digest(raw), subject.SOURCE_ADMISSION_SHA256)
        rows = subject.parse_admission(raw)
        self.assertEqual(len(rows), 57)
        self.assertEqual(sum(row["bytes"] for row in rows.values()), 928035)
        for name, row in rows.items():
            body = subject.source_bytes(name, row["sha256"])
            self.assertEqual(len(body), row["bytes"])

    def test_source_loading_uses_addon_exact_rootfs_module(self):
        before = set(sys.modules)
        helper, addon, qualification, files, manifest, entry = subject.load_sources()
        self.assertEqual(qualification.CASES, subject.CASES)
        self.assertIs(addon.rootfs.OwnedRootfs, sys.modules[addon.rootfs.__name__].OwnedRootfs)
        self.assertEqual(len(files), 35)
        self.assertEqual(sum(map(len, files.values())), 610752)
        self.assertEqual(addon.build_manifest(files), manifest)
        self.assertEqual(entry, (ROOT / subject.ENTRY).read_bytes())
        self.assertEqual(helper.ARCHIVE_SHA256, subject.ARCHIVE_SHA256)
        self.assertFalse({name.split(".", 1)[0] for name in set(sys.modules) - before}
                         & {"torch", "numpy", "transformers", "safetensors", "tokenizers"})

    def test_bad_manifest_contracts_reject(self):
        original = json.loads((ROOT / subject.SOURCE_ADMISSION).read_bytes())
        cases = []
        for field, value in [("schema", "wrong"), ("extra", True)]:
            candidate = copy.deepcopy(original)
            candidate[field] = value
            cases.append(candidate)
        for key, value in [("path", "../private"), ("bytes", True), ("bytes", 262145),
                           ("sha256", "A" * 64)]:
            candidate = copy.deepcopy(original)
            candidate["files"][0][key] = value
            cases.append(candidate)
        candidate = copy.deepcopy(original)
        candidate["files"].append(candidate["files"][0])
        cases.append(candidate)
        candidate = copy.deepcopy(original)
        candidate["files"] = candidate["files"][1:]
        cases.append(candidate)
        for candidate in cases:
            with self.assertRaises(subject.FixtureError):
                subject.parse_admission(subject.canonical(candidate))

    def test_noncanonical_duplicate_and_nonobject_json_reject(self):
        for raw in (b'{}', b'[]\n', b'{"schema":1,"schema":2}\n', b'{"schema":NaN}\n'):
            with self.assertRaises(subject.FixtureError):
                subject.parse_admission(raw)

    def test_source_bad_path_wrong_hash_and_short_read(self):
        for path in ("/private", "src/lumi/../a", "scripts/nested/a.py", "data/source.json"):
            with self.assertRaises(subject.FixtureError):
                subject.source_bytes(path)
        with self.assertRaises(subject.FixtureError):
            subject.source_bytes(subject.ENTRY, "0" * 64)
        with patch.object(subject.os, "read", return_value=b"x") as read, \
                self.assertRaises(subject.FixtureError):
            subject.source_bytes(subject.ENTRY)
        read.assert_called_once()

    def test_binding_failure_precedes_component_imports(self):
        with patch.object(subject, "SOURCE_ADMISSION_SHA256", "0" * 64), \
                patch.object(subject.importlib, "import_module") as imports, \
                self.assertRaises(subject.FixtureError):
            subject.load_sources()
        imports.assert_not_called()

    def test_mac_denies_before_any_source_or_daemon_action(self):
        with patch.object(subject.sys, "platform", "darwin"), \
                patch.object(subject, "load_sources") as load, \
                patch.object(subject, "command") as command, \
                self.assertRaises(subject.FixtureError):
            subject.run(Path("/private/no-payload-read"))
        load.assert_not_called()
        command.assert_not_called()

    def test_frozen_legacy_run_and_cases_are_never_called(self):
        tree = ast.parse(PATH.read_bytes())
        calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertNotIn("HELPER.run", calls)
        self.assertNotIn("HELPER.cases", calls)
        self.assertNotIn("subprocess.Popen", calls)
        self.assertNotIn("os.kill", calls)
        self.assertNotIn("os.killpg", calls)
        self.assertIn("ADDON.install_importer_addon", calls)
        self.assertIn("signal.pidfd_send_signal", calls)
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "run")
        self.assertIn("rootfs = ROOTFS = ADDON.rootfs", ast.unparse(run))
        self.assertNotIn("BASE.module('posttraining_sft_oci_rootfs_v1')", ast.unparse(run))

    def test_verified_loader_ignores_unverified_disk_and_cache_bytes(self):
        with tempfile.TemporaryDirectory(prefix="lumi-generated-host-loader-") as temporary:
            root = Path(temporary).resolve()
            source = root / "src/lumi"
            source.mkdir(parents=True)
            (source / "fixture.py").write_bytes(b"raise AssertionError('unverified disk')\n")
            (source / "__pycache__").mkdir()
            (source / "__pycache__/fixture.cpython-312.pyc").write_bytes(b"unverified bytecode")
            package_name = "_synthetic_authenticated_host_source"
            package = types.ModuleType(package_name)
            package.__path__ = [str(source)]
            with patch.dict(sys.modules, {package_name: package}), \
                    patch.object(subject, "ROOT", root):
                loader = subject.VerifiedHostSources(package_name,
                    {"src/lumi/fixture.py": b"VALUE = 'verified'\n"})
                sys.meta_path.insert(0, loader)
                selected = subject.importlib.import_module(package_name + ".fixture")
                self.assertEqual(selected.VALUE, "verified")
                self.assertEqual(selected.__file__, str(source / "fixture.py"))
                self.assertIsNone(selected.__cached__)

    def test_private_dynamic_helper_names_cannot_fall_back_to_unbound_files(self):
        loader = subject.VerifiedHostSources("_specific_host", {"src/lumi/allowed.py": b"PASS = 1"})
        for parent in ("_specific_host", "_lumi_generated_cgroup_fixture_v1",
                       "_lumi_checkpoint_guard_base_" + "a" * 32 + "_components"):
            self.assertIsNotNone(loader.find_spec(parent + ".allowed"))
            with self.assertRaises(subject.FixtureError):
                loader.find_spec(parent + ".unbound")
        self.assertIsNone(loader.find_spec("json.decoder"))
        self.assertIsNone(loader.find_spec("lumi.configuration_lumi"))


class CaseTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(subject.sys, "meta_path", list(sys.meta_path)))
        self.loaded = subject.load_sources()
        self.q = self.loaded[2]
        self.enterContext(patch.object(subject, "QUALIFICATION", self.q))

    def test_case_validator_positive_and_exact_negative_shapes(self):
        for case in subject.CASES:
            report = case_report(case)
            self.assertEqual(subject.validate_case(report, case, environment()), report)

    def test_wrong_worker_exit_code_or_false_removal_denies(self):
        for key, value in [("worker_exit_code", 0), ("expects_negative", 1),
                           ("owned_cgroups_empty_and_removed", False),
                           ("workload_processes_reaped", True), ("unexpected", "value")]:
            report = case_report("wrong_entry")
            report[key] = value
            with self.assertRaises((subject.FixtureError, self.q.QualificationError)):
                subject.validate_case(report, "wrong_entry", environment())

    def test_fixed_order_stops_on_first_failure(self):
        base = types.SimpleNamespace(subreaper=Mock())
        with patch.object(subject, "BASE", base), \
                patch.object(subject, "one_case", side_effect=subject.FixtureError()) as one:
            with self.assertRaises(subject.FixtureError):
                subject.cases(Path("/owned/root"), Path("/owned/mount"), b"manifest", b"entry", 99)
        self.assertEqual(one.call_count, 1)
        self.assertEqual(one.call_args.args[0], "wrong_entry")

    def test_fixed_order_and_original_environment_pins(self):
        base = types.SimpleNamespace(subreaper=Mock())
        with patch.object(subject, "BASE", base), \
                patch.object(subject, "one_case", side_effect=lambda case, *args: case) as one:
            self.assertEqual(subject.cases(Path("/owned/root"), Path("/owned/mount"),
                                           b"manifest", b"entry", 99), list(subject.CASES))
        self.assertEqual(one.call_count, 3)
        self.assertEqual([call.args[0] for call in one.call_args_list], list(subject.CASES))
        env = one.call_args.args[3]
        self.assertEqual(env["LUMI_IMPORT_ENTRY_SHA256"], digest(b"entry"))
        self.assertEqual(env["LUMI_IMPORT_SOURCE_MANIFEST_SHA256"], digest(b"manifest"))

    def run_one_double(self, *, packet=b"", child_status=0, extra_reaped=None):
        report = case_report("wrong_entry")
        parent, child = Mock(), Mock()
        base = types.SimpleNamespace(wait_statuses=Mock(return_value=extra_reaped or {}))
        self.enterContext(patch.object(subject, "BASE", base))
        self.enterContext(patch.object(subject, "DESCENDANTS_REAPED", True))
        self.enterContext(patch.object(subject.socket, "socketpair", return_value=(parent, child)))
        self.enterContext(patch.object(subject.os, "fork", return_value=123))
        self.enterContext(patch.object(subject.os, "pidfd_open", return_value=44, create=True))
        self.enterContext(patch.object(subject.os, "waitpid", return_value=(123, child_status)))
        self.enterContext(patch.object(subject.os, "close"))
        self.enterContext(patch.object(subject.time, "monotonic", return_value=0))
        self.enterContext(patch.object(subject.time, "monotonic_ns", return_value=100))
        self.enterContext(patch.object(subject.select, "select",
            side_effect=lambda read, *_args: (read if read == [parent] else [], [], [])))
        self.enterContext(patch.object(self.q.frozen, "_receive", return_value=report))
        self.enterContext(patch.object(self.q.frozen, "_packet", return_value=packet))
        signal = self.enterContext(patch.object(subject.signal, "pidfd_send_signal", create=True))
        return base, parent, child, signal

    def test_one_case_uses_fresh_child_and_validated_packet_then_eof(self):
        base, parent, child, signal = self.run_one_double()
        report = subject.one_case("wrong_entry", Path("/owned/root"), Path("/owned/mount"),
                                  environment(), 120)
        self.assertEqual(report["worker_exit_code"], 1)
        self.assertTrue(subject.DESCENDANTS_REAPED)
        signal.assert_not_called()
        base.wait_statuses.assert_called_once()
        parent.close.assert_called_once()
        child.close.assert_called_once()

    def test_extra_packet_signals_only_owned_case_pidfd_then_denies(self):
        _base, _parent, _child, signal = self.run_one_double(packet=b"extra")
        with self.assertRaises(subject.FixtureError):
            subject.one_case("wrong_entry", Path("/owned/root"), Path("/owned/mount"),
                             environment(), 120)
        signal.assert_called_once_with(44, subject.signal.SIGTERM, None, 0)

    def test_bad_case_exit_or_unexpected_adopted_child_never_passes(self):
        self.run_one_double(child_status=256)
        with self.assertRaises(subject.FixtureError):
            subject.one_case("wrong_entry", Path("/owned/root"), Path("/owned/mount"),
                             environment(), 120)

    def test_unexpected_adopted_child_denies_even_after_case_success(self):
        self.run_one_double(extra_reaped={456: 0})
        with self.assertRaises(subject.FixtureError):
            subject.one_case("wrong_entry", Path("/owned/root"), Path("/owned/mount"),
                             environment(), 120)
        self.assertFalse(subject.DESCENDANTS_REAPED)


class TinyTransactionTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(subject.sys, "meta_path", list(sys.meta_path)))
        self.temp = tempfile.TemporaryDirectory(prefix="lumi-generated-qualification-transaction-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.owned_parent = self.base / "owned"
        self.owned_parent.mkdir(mode=0o700)
        self.events, self.images, self.containers = [], set(), set()
        helper, self.addon, self.q, _files, _manifest, _entry = subject.load_sources()
        self.files = {"lumi/__init__.py": b'"""Original generated initializer."""\n',
                      "lumi/posttraining_checkpoint_guard_fixture_v1.py": b"VALUE = 1\n"}
        self.manifest = self.addon.build_manifest(self.files)
        self.entry = b'"""Generated inert entry."""\n'
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as archive:
            info = tarfile.TarInfo("usr/fixture.txt")
            info.mode, info.size = 0o644, 7
            archive.addfile(info, io.BytesIO(b"fixture"))
        self.tar = raw.getvalue()
        self.image_id = "sha256:" + "1" * 64
        self.container_id = "2" * 64
        self.binding = types.SimpleNamespace(image_id=self.image_id,
                                             source_identity_sha256=lambda: "3" * 64)
        self.admission = types.SimpleNamespace(IMAGE_ID=self.image_id,
            verify_created_oci=Mock(return_value=self.binding))
        self.native_base = types.SimpleNamespace(subreaper=Mock())
        self.helper = types.SimpleNamespace(load_base=lambda: self.native_base,
            load_admission=lambda: self.admission, copy_image=self.copy_image,
            command=self.command, create_options=helper.create_options,
            image_inventory=lambda *args: set(self.images),
            container_inventory=lambda *args: set(self.containers))
        self.enterContext(patch.object(subject, "runtime"))
        self.enterContext(patch.object(subject, "_ATTEMPTED", False))
        self.enterContext(patch.object(subject, "DESCENDANTS_REAPED", True))
        self.enterContext(patch.object(subject.os, "umask"))
        self.enterContext(patch.dict(subject.os.environ))
        self.enterContext(patch.object(subject.os, "statvfs", return_value=types.SimpleNamespace(
            f_bavail=16 * 1024**3, f_frsize=1)))
        self.enterContext(patch.object(subject, "tempfile", types.SimpleNamespace(
            mkdtemp=Mock(return_value=str(self.owned_parent)))))
        self.enterContext(patch.object(subject, "load_sources", return_value=(
            self.helper, self.addon, self.q, self.files, self.manifest, self.entry)))
        self.enterContext(patch.object(self.addon, "_runtime"))
        self.cases = self.enterContext(patch.object(subject, "cases", side_effect=self.fake_cases))

    def copy_image(self, _source, target, _deadline, identities):
        self.events.append("copy")
        target.write_bytes(b"generated-image")
        identities[target] = subject.identity(target.stat())

    def command(self, _config, args, **options):
        action = tuple(args[:2])
        self.events.append(action)
        if action == ("image", "load"):
            self.images.add(self.image_id)
        elif args[0] == "create":
            self.containers.add(next(arg[7:] for arg in args if arg.startswith("--name=")))
            return self.container_id.encode(), 64, "0" * 64, 0
        elif args[0] == "export":
            options["output_file"].write(self.tar)
            return b"", len(self.tar), digest(self.tar), 0
        elif args[0] == "rm":
            self.containers.remove(args[1])
        elif action == ("image", "rm"):
            self.images.remove(self.image_id)
        return b"", 0, digest(b""), 0

    def fake_cases(self, root_path, mountpoint, manifest, entry, deadline):
        self.events.append("cases")
        self.assertEqual((root_path / self.addon.MANIFEST_PATH).read_bytes(), manifest)
        self.assertEqual((root_path / self.addon.ENTRY_PATH).read_bytes(), entry)
        self.assertEqual((root_path / self.addon.SOURCE_PREFIX / "lumi/__init__.py").read_bytes(),
                         self.files["lumi/__init__.py"])
        self.assertEqual(list(mountpoint.iterdir()), [])
        return [case_report(case) for case in subject.CASES]

    def test_fake_daemon_real_tiny_rootfs_addon_transaction_and_exact_cleanup(self):
        report = subject.run(Path("/generated/unread-software.tar"))
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["addon"]["source_files"], 2)
        self.assertTrue(report["extraction"]["installation_frozen"])
        self.assertTrue(report["private_owned_cleanup_verified"])
        self.assertTrue(all(report["cleanup_inventory"].values()))
        self.assertFalse(self.images or self.containers or self.owned_parent.exists())
        self.assertEqual(self.admission.verify_created_oci.call_count, 2)
        self.assertLess(self.events.index(("image", "load")), self.events.index("cases"))
        self.assertFalse(any(action in {("start",), ("pull",), ("build",)}
                             for action in self.events))
        for key in subject.NONCLAIMS:
            self.assertIs(report[key], False)
        # Exercise the real nested receipt shape against the independent
        # supervisor's bounded transport contract, still with no native run.
        supervisor_path = ROOT / "scripts/supervise_checkpoint_guard_qualification_v1.py"
        supervisor_spec = importlib.util.spec_from_file_location(
            "_synthetic_harness_transport_supervisor", supervisor_path)
        supervisor = importlib.util.module_from_spec(supervisor_spec)
        supervisor_spec.loader.exec_module(supervisor)
        self.assertEqual(supervisor.parse_report(subject.canonical(report), 0), report)

    def test_case_failure_stops_and_retires_only_own_artifacts(self):
        self.cases.side_effect = subject.FixtureError("case_rejected")
        with self.assertRaises(subject.FixtureError):
            subject.run(Path("/generated/unread-software.tar"))
        self.assertFalse(self.images or self.containers or self.owned_parent.exists())

    def test_existing_image_is_never_removed_or_loaded(self):
        self.images.add(self.image_id)
        with self.assertRaises(subject.FixtureError):
            subject.run(Path("/generated/unread-software.tar"))
        self.assertEqual(self.images, {self.image_id})
        self.assertNotIn(("image", "load"), self.events)
        self.assertNotIn(("image", "rm"), self.events)

    def test_started_run_is_not_retried(self):
        subject.run(Path("/generated/unread-software.tar"))
        count = len(self.events)
        with self.assertRaises(subject.FixtureError):
            subject.run(Path("/generated/unread-software.tar"))
        self.assertEqual(len(self.events), count)

    def test_wrong_ownedrootfs_namespace_rejected_by_original_addon(self):
        # The full positive test above establishes the new same-module construction;
        # an object with only the expected-looking methods is still not accepted.
        with self.assertRaises(self.addon.SourceError):
            self.addon.install_importer_addon(types.SimpleNamespace(), self.files, self.manifest,
                self.entry, expected_manifest_sha256=digest(self.manifest),
                expected_entry_sha256=digest(self.entry))
