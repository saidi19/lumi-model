"""Synthetic resolver regressions, not historical replay or operational evidence.

Only tiny temporary fixture bytes are used, never historical/current source as a
fixture. No plans, private artifacts, Git history, approvals or datasets are read.
The real resolver code and its import dependencies are loaded normally. Run with
installed dependencies and -B, not -S: the existing Aya chain requires certifi.
The historical hash bindings and their five failing tests stay unchanged.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

from lumi.posttraining_aya_repair import resolve_workspace_binding
from lumi.posttraining_safety_uri_repair import resolve_binding

_ORIGINAL = b"synthetic fixture revision one\n"
_CHANGED = b"synthetic fixture revision two\n"
_Resolver = Callable[[Path, dict[str, str]], Path]


def _aya_resolver(root: Path, binding: dict[str, str]) -> Path:
    return resolve_workspace_binding(root, binding)


def _safety_resolver(root: Path, binding: dict[str, str]) -> Path:
    return resolve_binding(root, binding, name="synthetic_filter")


class HistoricalFilterBindingSyntheticTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="lumi-synthetic-filter-binding-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.root = self.base / "workspace" / "repo"
        self.root.mkdir(parents=True)
        self.fixture = self.root / "synthetic-filter.txt"
        self.fixture.write_bytes(_ORIGINAL)
        self.binding = {
            "path": "synthetic-filter.txt",
            "sha256": hashlib.sha256(_ORIGINAL).hexdigest(),
        }

    def _assert_exact_bytes_accepted(self, resolver: _Resolver) -> None:
        before = dict(self.binding)
        self.assertEqual(resolver(self.root, self.binding), self.fixture)
        self.assertEqual(self.binding, before)
        self.assertEqual(self.fixture.read_bytes(), _ORIGINAL)

    def _assert_changed_bytes_rejected(self, resolver: _Resolver) -> None:
        before = dict(self.binding)
        self.assertEqual(len(_ORIGINAL), len(_CHANGED))
        self.fixture.write_bytes(_CHANGED)
        with self.assertRaisesRegex(ValueError, "binding changed"):
            resolver(self.root, self.binding)
        self.assertTrue(self.fixture.is_file())
        self.assertEqual(self.fixture.read_bytes(), _CHANGED)
        self.assertEqual(self.binding, before)
        self.assertNotEqual(self.binding["sha256"], hashlib.sha256(_CHANGED).hexdigest())

    def _assert_missing_distinct_from_changed(
        self, resolver: _Resolver, missing_error: type[Exception]
    ) -> None:
        before = dict(self.binding)
        self.fixture.write_bytes(_CHANGED)
        with self.assertRaisesRegex(ValueError, "binding changed") as changed:
            resolver(self.root, self.binding)
        self.fixture.unlink()
        with self.assertRaisesRegex(missing_error, "binding is missing") as missing:
            resolver(self.root, self.binding)
        self.assertNotIn("missing", str(changed.exception))
        self.assertNotIn("changed", str(missing.exception))
        self.assertEqual(self.binding, before)

    def _assert_workspace_escape_rejected(self, resolver: _Resolver) -> None:
        outside = self.base / "outside" / "synthetic-filter.txt"
        outside.parent.mkdir()
        outside.write_bytes(_ORIGINAL)
        self.assertFalse(outside.is_relative_to(self.root.parent))
        binding = {
            "path": "../../outside/synthetic-filter.txt",
            "sha256": hashlib.sha256(_ORIGINAL).hexdigest(),
        }
        before = dict(binding)
        self.assertEqual((self.root / binding["path"]).resolve(), outside)
        with self.assertRaisesRegex(ValueError, "binding escapes (the )?workspace"):
            resolver(self.root, binding)
        self.assertEqual(binding, before)
        self.assertEqual(outside.read_bytes(), _ORIGINAL)

    def test_aya_accepts_exact_bound_synthetic_bytes(self) -> None:
        self._assert_exact_bytes_accepted(_aya_resolver)

    def test_safety_accepts_exact_bound_synthetic_bytes(self) -> None:
        self._assert_exact_bytes_accepted(_safety_resolver)

    def test_aya_rejects_changed_bytes_without_refreshing_binding(self) -> None:
        self._assert_changed_bytes_rejected(_aya_resolver)

    def test_safety_rejects_changed_bytes_without_refreshing_binding(self) -> None:
        self._assert_changed_bytes_rejected(_safety_resolver)

    def test_aya_distinguishes_missing_file_from_changed_bytes(self) -> None:
        self._assert_missing_distinct_from_changed(_aya_resolver, FileNotFoundError)

    def test_safety_distinguishes_missing_file_from_changed_bytes(self) -> None:
        self._assert_missing_distinct_from_changed(_safety_resolver, ValueError)

    def test_aya_rejects_existing_file_outside_workspace(self) -> None:
        self._assert_workspace_escape_rejected(_aya_resolver)

    def test_safety_rejects_existing_file_outside_workspace(self) -> None:
        self._assert_workspace_escape_rejected(_safety_resolver)


if __name__ == "__main__":
    unittest.main()
