"""AST and byte bindings only; never import a native builder, probe or fixture."""

from __future__ import annotations

import ast
import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRANSITIONS = {
    "sft-oci-release-candidate.yml": (
        "edaa86e238223cf3a1ecfb6292640d8bc43e72de",
        "85c64d8ae86993a7d99bd7abc03ebf2da33afa17b252b6bab9ad4b26618a35ef",
        "48af59107d4e17c898bb3c2508fad0c1a37cb2bdf152ce564252c4af93767b00",
        "    if: github.event.repository.private == true\n",
        "    # Historical one-shot attempt; source changes must not repeat its failed build.\n",
    ),
    "sft-cpu-origin-diagnostic.yml": (
        "15127b62e702244923a6f56c1a936fb667ea4486",
        "9104b408d8e80426fd841d4399f8073c529676c061c5ac8b29bc5109e3aa88f1",
        "b8a2f6d838138bc16f3666cac4f4749da49d39ae63662b28ce9f23fbc7bbf76f",
        "    if: github.event.repository.private == true && github.run_attempt == 1\n",
        "    # Historical one-shot diagnostic; binding-test edits must not start it again.\n",
    ),
}
FROZEN_SOURCE = {
    "scripts/build_sft_oci_release_candidate_v1.py":
        "458faf97c62689c9d7cc71a6eb89a4f84fef8c68bbcf9279b949dee450565044",
    "scripts/run_sft_oci_worker_probe_v2.py":
        "b45b04fc9cb93dd2301df0444f2db95666ba4f7a32e8f4d8e28c5b4077314450",
    "scripts/run_sft_worker_run_torch_fixture_v1.py":
        "53c50c92965ed1cef0c9624c57367c31ed7ebfb9d7cfb34257188ad179d333d0",
    "scripts/run_sft_release_gate_fixture_v1.py":
        "58cdef4d15055339ea8c8b7e8026371b75a36bb4671f543bb7e2f1df3a0d0106",
    "src/lumi/posttraining_sft_release_gate_v1.py":
        "95d526d88a65d8c57ee667a3e2c7f1b83f802058f8fe969f6b920a145f66fc75",
}


def condition(commit):
    return ("github.event.repository.private == true && github.run_attempt == 1 "
            "&& github.sha == '" + commit + "'")


class OciAdmissionSourceTests(unittest.TestCase):
    def test_only_two_historical_job_guards_and_comments_changed(self):
        for name, (commit, original, successor, old_line, comment) in TRANSITIONS.items():
            with self.subTest(workflow=name):
                raw = (ROOT / ".github/workflows" / name).read_bytes()
                replacement = (comment + "    if: " + condition(commit) + "\n").encode()
                self.assertEqual(raw.count(replacement), 1)
                self.assertEqual(hashlib.sha256(raw).hexdigest(), successor)
                restored = raw.replace(replacement, old_line.encode(), 1)
                self.assertEqual(hashlib.sha256(restored).hexdigest(), original)

    def test_all_three_native_workflows_have_exact_not_relaxable_one_shot_guards(self):
        commits = {name: row[0] for name, row in TRANSITIONS.items()}
        commits["sft-cpu-lookup-acceptance.yml"] = "c4272e36bc98354067dd23a83931a3db58823d76"
        for name, commit in commits.items():
            with self.subTest(workflow=name):
                text = (ROOT / ".github/workflows" / name).read_text()
                job_conditions = [line.strip()[4:] for line in text.splitlines()
                                  if line.startswith("    if: ")]
                self.assertEqual(job_conditions, [condition(commit)])
                self.assertNotIn("workflow_dispatch:", text)
                self.assertNotIn("schedule:", text)

    def test_direct_workflow_hash_consumers_match_live_successor_bindings(self):
        for name in ("test_sft_cpu_lookup_acceptance_source_v1.py",
                     "test_sft_cpu_diagnostic_builder_source_v1.py"):
            tree = ast.parse((ROOT / "tests" / name).read_bytes())
            rows = [ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "FROZEN_WORKFLOWS"
                            for target in node.targets)]
            self.assertEqual(len(rows), 1)
            for relative, expected in rows[0].items():
                path = ROOT / relative if relative.startswith(".github/") else (
                    ROOT / ".github/workflows" / relative)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_old_native_sources_and_guardian_release_gate_remain_byte_identical(self):
        for relative, expected in FROZEN_SOURCE.items():
            with self.subTest(source=relative):
                self.assertEqual(
                    hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected
                )

    def test_new_module_has_only_stdlib_imports_and_no_native_source_loader(self):
        tree = ast.parse((ROOT / "src/lumi/posttraining_sft_oci_admission_v1.py").read_bytes())
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module.split(".")[0])
        self.assertEqual(imports, {"__future__", "hashlib", "json", "os", "re", "select",
                                   "subprocess", "sys", "tempfile", "threading", "time",
                                   "dataclasses"})
        forbidden = {"exec", "eval", "compile", "__import__", "importlib.import_module",
                     "runpy.run_path", "os.system"}
        self.assertFalse(any(isinstance(node, ast.Call) and ast.unparse(node.func) in forbidden
                             for node in ast.walk(tree)))

    def test_runtime_guard_precedes_inspection_in_public_verifier(self):
        tree = ast.parse((ROOT / "src/lumi/posttraining_sft_oci_admission_v1.py").read_bytes())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "verify_created_oci")
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
        runtime = [node.lineno for node in calls if ast.unparse(node.func) == "_runtime"]
        inspections = [node.lineno for node in calls if ast.unparse(node.func) == "_inspect"]
        self.assertEqual(len(runtime), 1)
        self.assertEqual(len(inspections), 2)
        self.assertLess(runtime[0], min(inspections))


if __name__ == "__main__":
    unittest.main()
