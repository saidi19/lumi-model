"""Strict two-node lifecycle for an authenticated Aya promotion source projection.

No filesystem access, artifact action or prior checker main occurs here. The
caller authenticates the projection, suppresses pytest output and owns the one
cumulative input/snapshot/deadline budget, including cleanup before publication.
"""

from __future__ import annotations

import sys
import types

MODULE_NAME = "test_aya_source_promotion"
TEST_NAMES = (
    "test_successor_changes_only_the_existing_aya_entry",
    "test_promotion_receipt_is_reproducible_raw_free_and_nonproduction",
)
NODEIDS = tuple("tests/test_aya_source_promotion.py::" + name for name in TEST_NAMES)
_PHASES = ("setup", "call", "teardown")
_EXPECTED_EVENTS = tuple((nodeid, phase) for nodeid in NODEIDS for phase in _PHASES)


def _valid_root(repository_root: object) -> bool:
    return (
        type(repository_root) is str
        and 1 < len(repository_root) <= 4096
        and repository_root.startswith("/")
        and "\\" not in repository_root
        and all(ord(char) >= 32 and ord(char) != 127 for char in repository_root)
        and all(
            part not in {"", ".", ".."} and not part.startswith("~")
            for part in repository_root[1:].split("/")
        )
    )


class TwoNodes:
    """Accept one exact collection and six ordered passing reports; rejection sticks."""

    def __init__(self, io):
        self._io = io
        self._collected = False
        self._next_report = 0
        self._rejected = False

    def _require(self, condition):
        self._rejected = self._rejected or not condition
        self._io.require(not self._rejected, "reproduction_rejected")

    def pytest_collection_modifyitems(self, items):
        self._require(not self._collected and len(items) == len(NODEIDS))
        observed = tuple(getattr(item, "nodeid", None) for item in items)
        self._require(all(type(nodeid) is str for nodeid in observed) and observed == NODEIDS)
        self._collected = True

    def pytest_runtest_logreport(self, report):
        self._require(self._collected and self._next_report < len(_EXPECTED_EVENTS))
        nodeid = getattr(report, "nodeid", None)
        phase = getattr(report, "when", None)
        outcome = getattr(report, "outcome", None)
        self._require(
            type(nodeid) is str
            and type(phase) is str
            and (nodeid, phase) == _EXPECTED_EVENTS[self._next_report]
            and type(outcome) is str
            and outcome == "passed"
            and not hasattr(report, "wasxfail")
        )
        self._next_report += 1

    def passed(self):
        return self._collected and self._next_report == len(_EXPECTED_EVENTS) and not self._rejected

    def summary(self):
        self._require(self.passed())
        return {
            "collected": len(NODEIDS),
            "nodeids": list(NODEIDS),
            "phases": {nodeid: dict.fromkeys(_PHASES, "passed") for nodeid in NODEIDS},
        }


def run_projected_tests(io, pytest, module, repository_root: str):
    """Run exactly one pytest invocation; never displace another installed module."""
    io.require(_valid_root(repository_root), "runtime_rejected")
    filename = repository_root + "/tests/test_aya_source_promotion.py"
    io.require(
        not any(name == MODULE_NAME or name.startswith(MODULE_NAME + ".") for name in sys.modules),
        "runtime_rejected",
    )
    io.require(
        isinstance(module, types.ModuleType)
        and module.__name__ == MODULE_NAME
        and getattr(module, "__file__", None) == filename
        and all(callable(getattr(module, name, None)) for name in TEST_NAMES),
        "runtime_rejected",
    )
    plugin = TwoNodes(io)
    sys.modules[MODULE_NAME] = module
    try:
        result = pytest.main(
            [
                "-c", "/dev/null", "--rootdir", repository_root,
                "--noconftest", "--assert=plain",
                "-p", "no:cacheprovider", "-p", "no:logging", "-p", "no:faulthandler",
                "-s", "-q", "-x", "-o", "addopts=",
                *(filename + "::" + name for name in TEST_NAMES),
            ],
            plugins=[plugin],
        )
        io.require(
            result == 0 and plugin.passed() and sys.modules.get(MODULE_NAME) is module,
            "reproduction_rejected",
        )
        return plugin.summary()
    finally:
        if sys.modules.get(MODULE_NAME) is module:
            del sys.modules[MODULE_NAME]


def synthetic_module(repository_root: str):
    """Create two inert assertions with the exact identities, without importing tests."""
    if not _valid_root(repository_root):
        raise ValueError("invalid Aya promotion test root")
    module = types.ModuleType(MODULE_NAME)
    module.__file__ = repository_root + "/tests/test_aya_source_promotion.py"
    source = "\n".join("def " + name + "():\n    assert 2 + 2 == 4\n" for name in TEST_NAMES)
    exec(compile(source, "<synthetic-aya-promotion-runtime>", "exec"), module.__dict__)
    return module
