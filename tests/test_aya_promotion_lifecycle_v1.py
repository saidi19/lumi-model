"""Isolated synthetic lifecycle tests: no original test imports, inputs or real pytest runs."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "aya_promotion_lifecycle_tests",
    ROOT / "src/lumi/posttraining_aya_promotion_lifecycle_v1.py",
)
lifecycle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lifecycle)
PHASES = ("setup", "call", "teardown")
EVENTS = tuple((nodeid, phase) for nodeid in lifecycle.NODEIDS for phase in PHASES)


class Stop(BaseException):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def require(condition, code):
    if not condition:
        raise Stop(code)


class AyaPromotionLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.io = types.SimpleNamespace(require=require)
        self.module = lifecycle.synthetic_module(str(ROOT))
        self.pytest = Mock()

    def collect(self, plugin):
        plugin.pytest_collection_modifyitems([
            types.SimpleNamespace(nodeid=nodeid) for nodeid in lifecycle.NODEIDS
        ])

    def report(self, plugin, event, **changes):
        values = {
            "nodeid": event[0], "when": event[1], "outcome": "passed",
            "longrepr": "SYNTHETIC_SECRET", "capstdout": "SYNTHETIC_SECRET",
        }
        values.update(changes)
        plugin.pytest_runtest_logreport(types.SimpleNamespace(**values))

    def complete(self, plugin):
        self.collect(plugin)
        for event in EVENTS:
            self.report(plugin, event)

    def assert_rejected(self, plugin):
        self.assertFalse(plugin.passed())
        self.assertNotIn("SYNTHETIC_SECRET", str(vars(plugin)))
        with self.assertRaises(Stop) as caught:
            plugin.summary()
        self.assertEqual(caught.exception.code, "reproduction_rejected")

    def test_exact_identities_and_synthetic_functions_without_registration(self):
        self.assertEqual(lifecycle.MODULE_NAME, "test_aya_source_promotion")
        self.assertEqual(lifecycle.TEST_NAMES, (
            "test_successor_changes_only_the_existing_aya_entry",
            "test_promotion_receipt_is_reproducible_raw_free_and_nonproduction",
        ))
        self.assertEqual(lifecycle.NODEIDS, tuple(
            "tests/test_aya_source_promotion.py::" + name for name in lifecycle.TEST_NAMES
        ))
        before = sys.modules.get(lifecycle.MODULE_NAME)
        module = lifecycle.synthetic_module("/synthetic/repository")
        self.assertEqual(module.__name__, lifecycle.MODULE_NAME)
        self.assertEqual(
            module.__file__, "/synthetic/repository/tests/test_aya_source_promotion.py"
        )
        self.assertEqual(
            {name for name in vars(module) if name.startswith("test_")}, set(lifecycle.TEST_NAMES)
        )
        for name in lifecycle.TEST_NAMES:
            self.assertIsNone(getattr(module, name)())
        self.assertIs(sys.modules.get(lifecycle.MODULE_NAME), before)

    def test_exact_complete_lifecycle_has_only_fixed_summary_fields(self):
        plugin = lifecycle.TwoNodes(self.io)
        self.assertFalse(plugin.passed())
        self.complete(plugin)
        self.assertTrue(plugin.passed())
        expected = {
            "collected": 2,
            "nodeids": list(lifecycle.NODEIDS),
            "phases": {nodeid: dict.fromkeys(PHASES, "passed") for nodeid in lifecycle.NODEIDS},
        }
        self.assertEqual(plugin.summary(), expected)
        self.assertNotIn("SYNTHETIC_SECRET", str(vars(plugin)))
        summary = plugin.summary()
        summary["phases"][lifecycle.NODEIDS[0]]["setup"] = "changed"
        summary["nodeids"].reverse()
        self.assertEqual(plugin.summary(), expected)

    def test_collection_rejects_empty_missing_extra_duplicate_reversed_and_foreign_nodes(self):
        first, second = lifecycle.NODEIDS
        for nodeids in (
            (), (first,), (second,), (first, second, first), (first, first),
            (second, first), (first, "SYNTHETIC_SECRET"), (None, second),
        ):
            with self.subTest(nodeids=nodeids):
                plugin = lifecycle.TwoNodes(self.io)
                with self.assertRaises(Stop):
                    plugin.pytest_collection_modifyitems([
                        types.SimpleNamespace(nodeid=nodeid) for nodeid in nodeids
                    ])
                self.assert_rejected(plugin)
                with self.assertRaises(Stop):
                    self.collect(plugin)

    def test_collection_cannot_repeat_before_or_after_success(self):
        for completed in (False, True):
            with self.subTest(completed=completed):
                plugin = lifecycle.TwoNodes(self.io)
                if completed:
                    self.complete(plugin)
                else:
                    self.collect(plugin)
                with self.assertRaises(Stop):
                    self.collect(plugin)
                self.assert_rejected(plugin)

    def test_report_before_collection_permanently_rejects_plugin(self):
        plugin = lifecycle.TwoNodes(self.io)
        with self.assertRaises(Stop):
            self.report(plugin, EVENTS[0])
        self.assert_rejected(plugin)
        with self.assertRaises(Stop):
            self.collect(plugin)

    def test_failed_skipped_or_xfail_reports_reject_every_event_position(self):
        for position in range(len(EVENTS)):
            for change in (
                {"outcome": "failed"}, {"outcome": "skipped"},
                {"outcome": "rerun"}, {"outcome": "SYNTHETIC_SECRET"},
                {"wasxfail": "SYNTHETIC_SECRET"}, {"wasxfail": False},
            ):
                with self.subTest(position=position, change=change):
                    plugin = lifecycle.TwoNodes(self.io)
                    self.collect(plugin)
                    for event in EVENTS[:position]:
                        self.report(plugin, event)
                    with self.assertRaises(Stop):
                        self.report(plugin, EVENTS[position], **change)
                    self.assert_rejected(plugin)
                    with self.assertRaises(Stop):
                        self.report(plugin, EVENTS[position])

    def test_wrong_node_phase_or_order_is_rejected_at_every_position(self):
        for position, expected in enumerate(EVENTS):
            for event in (
                ("SYNTHETIC_SECRET", expected[1]),
                (expected[0], "unknown"),
                EVENTS[(position + 1) % len(EVENTS)],
            ):
                with self.subTest(position=position, event=event):
                    plugin = lifecycle.TwoNodes(self.io)
                    self.collect(plugin)
                    for previous in EVENTS[:position]:
                        self.report(plugin, previous)
                    with self.assertRaises(Stop):
                        self.report(plugin, event)
                    self.assert_rejected(plugin)

    def test_missing_trailing_reports_never_pass(self):
        for length in range(len(EVENTS)):
            with self.subTest(length=length):
                plugin = lifecycle.TwoNodes(self.io)
                self.collect(plugin)
                for event in EVENTS[:length]:
                    self.report(plugin, event)
                self.assert_rejected(plugin)

    def test_duplicate_report_before_or_after_complete_lifecycle_rejects_permanently(self):
        for position in range(len(EVENTS)):
            with self.subTest(position=position):
                plugin = lifecycle.TwoNodes(self.io)
                self.collect(plugin)
                for event in EVENTS[:position + 1]:
                    self.report(plugin, event)
                with self.assertRaises(Stop):
                    self.report(plugin, EVENTS[position])
                self.assert_rejected(plugin)

    def test_module_and_descendant_collisions_are_preserved_without_pytest_call(self):
        for suffix in ("", ".foreign", ".nested.descendant"):
            with self.subTest(suffix=suffix), patch.dict(sys.modules):
                sys.modules.pop(lifecycle.MODULE_NAME, None)
                name = lifecycle.MODULE_NAME + suffix
                existing = types.ModuleType(name)
                sys.modules[name] = existing
                with self.assertRaises(Stop) as caught:
                    lifecycle.run_projected_tests(self.io, self.pytest, self.module, str(ROOT))
                self.assertEqual(caught.exception.code, "runtime_rejected")
                self.assertIs(sys.modules[name], existing)
                self.pytest.main.assert_not_called()

    def test_invalid_root_or_module_identity_never_starts_pytest(self):
        invalid_roots = (
            None, 1, "", "/", "relative", "//synthetic/repo", "/synthetic/../repo",
            "/synthetic/./repo", "/synthetic//repo", "/synthetic/repo/", "/synthetic/~repo",
            "/synthetic\\repo", "/synthetic/\x00repo", "/synthetic/\x7frepo", "/" + "x" * 4096,
        )
        for root in invalid_roots:
            with self.subTest(root=root):
                with self.assertRaises(ValueError):
                    lifecycle.synthetic_module(root)
                with self.assertRaises(Stop):
                    lifecycle.run_projected_tests(self.io, self.pytest, self.module, root)
                self.pytest.main.assert_not_called()
        for alteration in ("name", "file", "missing_function", "noncallable", "not_module"):
            with self.subTest(alteration=alteration), patch.dict(sys.modules):
                sys.modules.pop(lifecycle.MODULE_NAME, None)
                module = lifecycle.synthetic_module(str(ROOT))
                if alteration == "name":
                    module.__name__ = "wrong"
                elif alteration == "file":
                    module.__file__ = "/wrong/test_aya_source_promotion.py"
                elif alteration == "missing_function":
                    delattr(module, lifecycle.TEST_NAMES[0])
                elif alteration == "noncallable":
                    setattr(module, lifecycle.TEST_NAMES[1], 0)
                else:
                    module = object()
                with self.assertRaises(Stop):
                    lifecycle.run_projected_tests(self.io, self.pytest, module, str(ROOT))
                self.assertNotIn(lifecycle.MODULE_NAME, sys.modules)
                self.pytest.main.assert_not_called()

    def test_one_pytest_invocation_uses_exact_ordered_targets_and_restrictive_flags(self):
        def run(arguments, *, plugins):
            self.assertEqual(arguments, [
                "-c", "/dev/null", "--rootdir", str(ROOT), "--noconftest", "--assert=plain",
                "-p", "no:cacheprovider", "-p", "no:logging", "-p", "no:faulthandler",
                "-s", "-q", "-x", "-o", "addopts=",
                *(str(ROOT) + "/" + nodeid for nodeid in lifecycle.NODEIDS),
            ])
            self.assertEqual(len(plugins), 1)
            self.assertIs(sys.modules[lifecycle.MODULE_NAME], self.module)
            self.complete(plugins[0])
            return 0

        existing = types.ModuleType(lifecycle.MODULE_NAME)
        with patch.dict(sys.modules, {lifecycle.MODULE_NAME: existing}):
            with patch.dict(sys.modules):
                del sys.modules[lifecycle.MODULE_NAME]
                self.pytest.main.side_effect = run
                summary = lifecycle.run_projected_tests(
                    self.io, self.pytest, self.module, str(ROOT)
                )
                self.assertEqual(summary["collected"], 2)
                self.assertEqual(summary["nodeids"], list(lifecycle.NODEIDS))
                self.assertNotIn(lifecycle.MODULE_NAME, sys.modules)
            self.assertIs(sys.modules[lifecycle.MODULE_NAME], existing)
        self.pytest.main.assert_called_once()

    def test_exit_without_complete_reports_or_nonzero_exit_always_cleans_up_without_retry(self):
        for completed, exit_code in ((False, 0), (False, 1), (True, 1), (True, 2)):
            with self.subTest(completed=completed, exit_code=exit_code), patch.dict(sys.modules):
                sys.modules.pop(lifecycle.MODULE_NAME, None)
                fake_pytest = Mock()

                def run(_arguments, *, plugins, completed=completed, exit_code=exit_code):
                    if completed:
                        self.complete(plugins[0])
                    return exit_code

                fake_pytest.main.side_effect = run
                with self.assertRaises(Stop) as caught:
                    lifecycle.run_projected_tests(self.io, fake_pytest, self.module, str(ROOT))
                self.assertEqual(caught.exception.code, "reproduction_rejected")
                self.assertNotIn(lifecycle.MODULE_NAME, sys.modules)
                fake_pytest.main.assert_called_once()

    def test_caught_report_rejection_cannot_be_hidden_by_zero_pytest_exit(self):
        def run(_arguments, *, plugins):
            self.complete(plugins[0])
            with self.assertRaises(Stop):
                self.report(plugins[0], EVENTS[-1])
            return 0

        with patch.dict(sys.modules):
            sys.modules.pop(lifecycle.MODULE_NAME, None)
            self.pytest.main.side_effect = run
            with self.assertRaises(Stop):
                lifecycle.run_projected_tests(self.io, self.pytest, self.module, str(ROOT))
            self.assertNotIn(lifecycle.MODULE_NAME, sys.modules)
        self.pytest.main.assert_called_once()

    def test_pytest_exceptions_remove_only_owned_projection_and_restore_outer_namespace(self):
        for failure in (RuntimeError("SYNTHETIC_SECRET"), Stop("deadline_exceeded")):
            with self.subTest(failure=type(failure).__name__):
                existing = types.ModuleType(lifecycle.MODULE_NAME)
                fake_pytest = Mock()
                fake_pytest.main.side_effect = failure
                with patch.dict(sys.modules, {lifecycle.MODULE_NAME: existing}):
                    with patch.dict(sys.modules):
                        del sys.modules[lifecycle.MODULE_NAME]
                        with self.assertRaises(type(failure)) as caught:
                            lifecycle.run_projected_tests(
                                self.io, fake_pytest, self.module, str(ROOT)
                            )
                        self.assertIs(caught.exception, failure)
                        self.assertNotIn(lifecycle.MODULE_NAME, sys.modules)
                    self.assertIs(sys.modules[lifecycle.MODULE_NAME], existing)
                fake_pytest.main.assert_called_once()

    def test_foreign_replacement_is_preserved_and_prevents_success(self):
        replacement = types.ModuleType(lifecycle.MODULE_NAME)

        def run(_arguments, *, plugins):
            self.complete(plugins[0])
            sys.modules[lifecycle.MODULE_NAME] = replacement
            return 0

        with patch.dict(sys.modules):
            sys.modules.pop(lifecycle.MODULE_NAME, None)
            self.pytest.main.side_effect = run
            with self.assertRaises(Stop):
                lifecycle.run_projected_tests(self.io, self.pytest, self.module, str(ROOT))
            self.assertIs(sys.modules[lifecycle.MODULE_NAME], replacement)
        self.pytest.main.assert_called_once()


if __name__ == "__main__":
    unittest.main()
