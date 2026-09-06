"""Project only the unchanged Aya owner-tamper rejection test onto bound bytes.

Reuse prerequisite authentication without running its positive test. The caller
authenticates the manifest, authorizes physical input reads, supplies real pytest
and owns output suppression and lifecycle accounting. No historical seal is
replaced here; registry and runner inputs remain hash-only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import posixpath
import re
import types
from collections.abc import Mapping

from .posttraining_aya_plan_snapshot_v1 import _compile, _select
from .posttraining_aya_prerequisite_snapshot_v1 import (
    AYA_CONSTANTS,
    AYA_FUNCTIONS,
    MAXIMUM_VIRTUAL_OPENS,
    PLAN_BINDING_ROLES,
    SOURCE_PATHS,
    _authenticate,
    _check_plan,
)
from .posttraining_safety_replay_snapshot_v1 import Snapshot

MODULE_NAME = "test_aya_source_promotion"
TEST_NAME = "test_rejected_or_tampered_owner_gate_cannot_be_promoted"
_ERROR = "invalid Aya owner-gate source projection"


def build_test_module(payloads: dict, bindings: dict, repository_root: str):
    """Return an unregistered negative-test module and its nine-blob snapshot.

    Eleven inputs authenticate before any parsing. Construction opens no virtual
    inputs; the unchanged test opens twelve: plan, eight binding hashes, then
    attribution, materialization and owner metadata. Attribution is parsed, but
    its provenance loop is not reached when the owner seal correctly rejects.
    """
    try:
        blobs, checked = _authenticate(payloads, bindings, repository_root)
        _check_plan(blobs["plan"], checked)
        selections = (
            ("promotion_implementation", _select(
                blobs["promotion_implementation"], AYA_CONSTANTS, AYA_FUNCTIONS,
            )),
            ("hash_helper_source", _select(
                blobs["hash_helper_source"], frozenset(),
                frozenset({"canonical_json_bytes", "file_sha256"}),
            )),
            ("test_source", _select(
                blobs["test_source"], frozenset(),
                frozenset({"_load", "_prerequisites", TEST_NAME}),
            )),
        )
        view_roles = PLAN_BINDING_ROLES | {"plan"}
        logical_bytes = sum(checked[role]["bytes"] for role in view_roles) + sum(
            checked[role]["bytes"]
            for role in ("attribution_manifest", "materialization_report", "owner_review_receipt")
        )
        snapshot = Snapshot(
            {
                posixpath.normpath(repository_root + "/" + SOURCE_PATHS[role]): blobs[role]
                for role in view_roles
            },
            repository_root=repository_root,
            maximum_virtual_opens=MAXIMUM_VIRTUAL_OPENS,
            maximum_logical_read_bytes=logical_bytes,
            maximum_gzip_bytes=1,
        )
        module = types.ModuleType(MODULE_NAME)
        module.__file__ = repository_root + "/" + SOURCE_PATHS["test_source"]
        module.__package__ = ""
        module.__dict__.update({
            "copy": copy, "json": json, "re": re, "hashlib": hashlib, "Mapping": Mapping,
            "Path": snapshot.Path, "ROOT": snapshot.Path(repository_root),
            "PLAN": snapshot.Path(repository_root) / SOURCE_PATHS["plan"],
        })
        for role, nodes in selections:
            _compile(nodes, repository_root + "/" + SOURCE_PATHS[role], module.__dict__)
        if (
            module.EXPECTED_BINDINGS != PLAN_BINDING_ROLES
            or {name for name in vars(module) if name.startswith("test_")} != {TEST_NAME}
        ):
            raise ValueError(_ERROR)
        return module, snapshot
    except Exception:
        raise ValueError(_ERROR) from None
