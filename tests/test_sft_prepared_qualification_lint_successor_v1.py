"""Exact lint-only AST and source-binding regression; no runtime or private inputs."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/lumi-4b-sft-prepared-qualification-lint-source-parity-v0.1.json"


def normalize(raw, replacements):
    changes = dict(replacements)
    pattern = "|".join(re.escape(key) for key in sorted(changes, key=len, reverse=True))
    text = re.sub(pattern, lambda match: changes[match.group()], raw.decode("utf-8"))
    tree = ast.parse(text)
    # import-order changes, one unused import and explicit default zip strict=False
    # are the only non-binding normalizations; every other AST node stays exact.
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Import) and len(node.names) == 1 and node.names[0].name == "re"
        and not any(isinstance(item, ast.Name) and item.id == "re"
                    for item in ast.walk(tree)))]
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "zip":
            node.keywords = [part for part in node.keywords if not (
                part.arg == "strict" and isinstance(part.value, ast.Constant)
                and part.value.value is False)]
    start = 0
    while start < len(tree.body):
        if not isinstance(tree.body[start], (ast.Import, ast.ImportFrom)):
            start += 1
            continue
        stop = start
        while stop < len(tree.body) and isinstance(tree.body[stop], (ast.Import, ast.ImportFrom)):
            stop += 1
        tree.body[start:stop] = sorted(tree.body[start:stop], key=ast.dump)
        start = stop
    def stable(node):
        if isinstance(node, ast.AST):
            fields = {key: stable(value) for key, value in ast.iter_fields(node)
                      if not (key == "type_params" and value == [])}
            return {"node": type(node).__name__, "fields": fields}
        if isinstance(node, list):
            return [stable(item) for item in node]
        if isinstance(node, bytes):
            return {"bytes_hex": node.hex()}
        if node is Ellipsis:
            return {"ellipsis": True}
        return node

    # ast.dump changed empty-field formatting in 3.13; serialize fields directly.
    encoded = json.dumps(stable(tree), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class LintSuccessorTests(unittest.TestCase):
    def test_every_successor_matches_reviewed_predecessor_AST_and_new_hash(self):
        plan = json.loads(MANIFEST.read_bytes())
        self.assertEqual(len(plan["files"]), 12)
        for row in plan["files"]:
            with self.subTest(path=row["path"]):
                raw = (ROOT / row["path"]).read_bytes()
                self.assertEqual(len(raw), row["bytes"])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), row["sha256"])
                self.assertEqual(normalize(raw, plan["normalize"]),
                                 row["predecessor_normalized_AST_sha256"])

    def test_zip_keeps_the_previous_non_strict_default(self):
        tree = ast.parse((ROOT / "src/lumi/posttraining_checkpoint_input_mount_v2.py").read_bytes())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "_inputs")
        calls = [node for node in ast.walk(function)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == "zip"]
        self.assertEqual(len(calls), 1)
        self.assertEqual([(part.arg, ast.literal_eval(part.value)) for part in calls[0].keywords],
                         [("strict", False)])

    def test_unused_re_import_removed_without_a_lint_suppression(self):
        raw = (ROOT / "src/lumi/posttraining_checkpoint_generator_guard_v2.py").read_bytes()
        tree = ast.parse(raw)
        self.assertFalse(any(isinstance(node, ast.Import)
                             and any(part.name == "re" for part in node.names)
                             for node in tree.body))
        self.assertNotIn(b"noqa", raw)

    def test_worker_entry_and_both_worker_manifests_unchanged(self):
        plan = json.loads(MANIFEST.read_bytes())
        self.assertEqual(len(plan["unchanged_worker_bindings"]), 3)
        for row in plan["unchanged_worker_bindings"]:
            raw = (ROOT / row["path"]).read_bytes()
            self.assertEqual(len(raw), row["bytes"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row["sha256"])


if __name__ == "__main__":
    unittest.main()
