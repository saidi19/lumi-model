"""AST/literal checks only: never import or execute the OCI inventory producer.

These assertions inspect repository source, not a container, Python runtime,
installed package tree or library. Native acceptance belongs on remote Linux.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/build_sft_oci_base_inventory_v1.py"


def literal(tree, name):
    matches = [node.value for node in tree.body if isinstance(node, ast.Assign)
               and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
               and node.targets[0].id == name]
    if len(matches) != 1:
        raise AssertionError("one literal binding required: " + name)
    return ast.literal_eval(matches[0])


def function(tree, name):
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
               and node.name == name]
    if len(matches) != 1:
        raise AssertionError("one function required: " + name)
    return matches[0]


def calls(tree, name):
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and ast.unparse(node.func) == name]


def same(node, text):
    return ast.dump(node, include_attributes=False) == ast.dump(
        ast.parse(text, mode="eval").body, include_attributes=False,
    )


class OciBaseInventorySourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))

    def assert_guard(self, node, text, code):
        self.assertTrue(any(len(call.args) == 2 and same(call.args[0], text)
                            and same(call.args[1], repr(code)) for call in calls(node, "need")),
                        text)

    def test_module_is_remote_only_and_has_no_top_level_runtime_dispatch(self):
        docstring = ast.get_docstring(self.tree)
        self.assertIn("NEVER import or execute on the local Mac", docstring)
        self.assertIn("not a whole-filesystem inventory", docstring)
        main = [node for node in self.tree.body if isinstance(node, ast.If)]
        self.assertEqual(len(main), 1)
        self.assertTrue(same(main[0].test, '__name__ == "__main__"'))
        self.assertEqual(ast.unparse(main[0].body[0]), "raise SystemExit(main())")
        self.assertFalse(any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                             for node in self.tree.body))

    def test_only_declared_stdlib_imports(self):
        imports = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0)
                imports.add(node.module)
        self.assertEqual(imports, {
            "__future__", "hashlib", "importlib.metadata", "json", "os", "re", "select",
            "stat", "sys", "time", "pathlib",
        })
        for forbidden in ("exec", "eval", "compile", "__import__", "importlib.import_module",
                          "subprocess.run", "os.system", "os.execve", "os.fork"):
            self.assertFalse(calls(self.tree, forbidden), forbidden)

    def test_schema_roots_and_runtime_are_fixed(self):
        self.assertEqual(literal(self.tree, "SCHEMA"), "lumi.sft-oci-base-inventory.v0.1")
        self.assertEqual(literal(self.tree, "ROOTS"), ("/usr/lib", "/usr/local/lib"))
        self.assertEqual(literal(self.tree, "AUXILIARY_LEAF_ROOTS"),
                         ("/etc/ssl", "/usr/share/zoneinfo"))
        self.assertEqual(literal(self.tree, "RUNTIME"), "/usr/local/bin/python3.12")
        self.assertEqual(literal(self.tree, "SITE_PACKAGES"),
                         "/usr/local/lib/python3.12/site-packages")

    def test_caps_are_exact_and_bounded(self):
        expected = {
            "MAXIMUM_FILES": 40_000, "MAXIMUM_TREE_ENTRIES": 80_000,
            "MAXIMUM_FILE_BYTES": 268_435_456, "MAXIMUM_TOTAL_READ_BYTES": 1_073_741_824,
            "MAXIMUM_OUTPUT_BYTES": 16_777_216, "MAXIMUM_INPUT_BYTES": 512,
            "MAXIMUM_METADATA_BYTES": 2_097_152, "MAXIMUM_DISTRIBUTIONS": 256,
            "MAXIMUM_RUNTIME_SECONDS": 120, "MAXIMUM_PATH_BYTES": 4096,
        }
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertIs(type(literal(self.tree, name)), int)
                self.assertEqual(literal(self.tree, name), value)

    def test_actual_runtime_and_explicit_ci_cli_guard(self):
        guard = function(self.tree, "runtime_guard")
        self.assert_guard(guard,
                          'sys.platform == "linux" and os.uname().machine == "x86_64" '
                          'and sys.implementation.name == "cpython" '
                          'and sys.version_info[:2] == (3, 12) and sys.flags.isolated == 1 '
                          'and sys.dont_write_bytecode', "runtime_rejected")
        self.assert_guard(guard,
                          'sys.argv[1:] == ["--base-inventory-only"] '
                          'and os.environ.get("LUMI_OCI_BASE_INVENTORY_CI") == "1"',
                          "runtime_rejected")
        self.assert_guard(guard,
                          'str(Path(sys.executable).resolve(strict=True)) == RUNTIME '
                          'and sys.prefix == sys.base_prefix == "/usr/local"',
                          "runtime_rejected")
        self.assert_guard(guard,
                          'not any(name.split(".", 1)[0] in ML_ROOTS for name in sys.modules)',
                          "runtime_rejected")

    def test_runtime_check_precedes_input_and_tree_reads_and_repeats(self):
        producer = function(self.tree, "produce")
        runtime = calls(producer, "runtime_guard")
        self.assertEqual(len(runtime), 2)
        for name in ("read_input", "files", "base_distributions"):
            selected = calls(producer, name)
            self.assertEqual(len(selected), 1)
            self.assertLess(runtime[0].lineno, selected[0].lineno)
            self.assertLess(selected[0].lineno, runtime[1].lineno)

    def test_bounded_stdin_and_exact_digest_fields(self):
        reader = function(self.tree, "read_input")
        read = calls(reader, "os.read")
        self.assertEqual(len(read), 1)
        self.assertTrue(same(read[0].args[0], "0"))
        self.assertTrue(same(read[0].args[1], "MAXIMUM_INPUT_BYTES + 1 - len(raw)"))
        self.assert_guard(reader, "select.select([0], [], [], 5)[0]", "input_rejected")
        self.assert_guard(reader, "len(raw) <= MAXIMUM_INPUT_BYTES", "input_rejected")
        self.assert_guard(reader,
                          'type(value) is dict and set(value) == {"digest", "config_digest"}',
                          "input_rejected")
        self.assert_guard(reader,
                          'all(type(value[key]) is str '
                          'and re.fullmatch(r"sha256:[0-9a-f]{64}", value[key]) for key in value)',
                          "input_rejected")
        self.assert_guard(function(self.tree, "unique_pairs"), "key not in value", "input_rejected")

    def test_json_is_ascii_sorted_compact_and_finite(self):
        formatter = function(self.tree, "canonical")
        dumps = calls(formatter, "json.dumps")
        self.assertEqual(len(dumps), 1)
        self.assertEqual({item.arg: ast.literal_eval(item.value) for item in dumps[0].keywords},
                         {"ensure_ascii": True, "sort_keys": True,
                          "separators": (",", ":"), "allow_nan": False})
        encoding = [node for node in ast.walk(formatter) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == "encode"]
        self.assertEqual(len(encoding), 1)
        self.assertTrue(same(encoding[0].args[0], '"ascii"'))

    def test_resolved_targets_are_checked_before_return(self):
        resolver = function(self.tree, "resolved")
        self.assert_guard(resolver, "permitted(result)", "path_rejected")
        self.assertEqual(len(calls(resolver, "ordinary_path")), 2)
        self.assertIn("resolve(strict=True)", ast.unparse(resolver))
        permitted = function(self.tree, "permitted")
        self.assertTrue(same(permitted.body[0].value,
                             "path == RUNTIME or any(below(path, root) "
                             "for root in (*ROOTS, *AUXILIARY_LEAF_ROOTS))"))

    def test_auxiliary_directory_aliases_are_not_recursed(self):
        walker = function(self.tree, "files")
        selections = [node for node in ast.walk(walker) if isinstance(node, ast.If)
                      and same(node.test, "any(below(path, root) "
                               "for root in AUXILIARY_LEAF_ROOTS)")]
        self.assertEqual(len(selections), 1)
        self.assertEqual(len(selections[0].body), 1)
        self.assertIsInstance(selections[0].body[0], ast.Continue)
        self.assert_guard(walker, "any(below(path, root) for root in ROOTS)", "tree_rejected")

    def test_special_nodes_reject_and_canonical_paths_are_deduplicated(self):
        walker = function(self.tree, "files")
        self.assert_guard(walker, "stat.S_ISREG(info.st_mode)", "tree_rejected")
        tests = [ast.unparse(node.test) for node in ast.walk(walker) if isinstance(node, ast.If)]
        self.assertIn("path in result", tests)
        self.assertIn("path in directories", tests)
        self.assert_guard(walker, "len(result) < MAXIMUM_FILES", "entry_budget_exhausted")
        self.assert_guard(walker, "RUNTIME in result", "runtime_rejected")
        self.assertEqual(len(calls(walker, "read_file")), 1)

    def test_only_readonly_regular_file_open_and_guaranteed_close(self):
        opens = calls(self.tree, "os.open")
        self.assertEqual(len(opens), 1)
        self.assertTrue(same(opens[0].args[1],
                             "os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK"))
        reader = function(self.tree, "read_file")
        self.assert_guard(reader, "permitted(path)", "path_rejected")
        self.assert_guard(reader,
                          "stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= maximum",
                          "file_rejected")
        finalizers = [node.finalbody for node in ast.walk(reader) if isinstance(node, ast.Try)]
        self.assertTrue(any(any(calls(statement, "os.close") for statement in finalizer)
                            for finalizer in finalizers))

    def test_hash_stream_prepaid_eof_stability_and_metadata_rereads_count(self):
        reader = function(self.tree, "read_file")
        reservation = calls(reader, "budget.reserve_read")
        reads = calls(reader, "os.read")
        self.assertEqual(len(reservation), 1)
        self.assertTrue(same(reservation[0].args[0], "before.st_size + 1"))
        self.assertTrue(all(reservation[0].lineno < node.lineno for node in reads))
        self.assertTrue(same(reads[0].args[1], "min(65536, remaining)"))
        self.assert_guard(reader, 'os.read(descriptor, 1) == b""', "file_changed")
        self.assert_guard(reader,
                          "signature(os.fstat(descriptor)) == signature(before) "
                          "== signature(os.stat(path, follow_symlinks=False))", "file_changed")
        metadata_read = calls(function(self.tree, "base_distributions"), "read_file")
        self.assertEqual(len(metadata_read), 1)
        self.assertTrue(any(item.arg == "keep" and same(item.value, "True")
                            for item in metadata_read[0].keywords))

    def test_metadata_discovery_is_explicit_site_packages_only(self):
        checker = function(self.tree, "base_distributions")
        discovery = calls(checker, "importlib.metadata.distributions")
        self.assertEqual(len(discovery), 1)
        self.assertEqual(len(discovery[0].keywords), 1)
        self.assertEqual(discovery[0].keywords[0].arg, "path")
        self.assertTrue(same(discovery[0].keywords[0].value, "[SITE_PACKAGES]"))
        self.assert_guard(checker,
                          "len(observed) < MAXIMUM_DISTRIBUTIONS "
                          "and type(distribution) is importlib.metadata.PathDistribution",
                          "distribution_rejected")
        self.assert_guard(checker, "path in rows", "distribution_rejected")
        self.assert_guard(checker, "row == rows[path]", "file_changed")

    def test_supplied_metadata_parser_cannot_read_package_code(self):
        classes = [node for node in self.tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "SuppliedMetadata"]
        self.assertEqual(len(classes), 1)
        self.assertTrue(same(classes[0].bases[0], "importlib.metadata.Distribution"))
        reader = function(classes[0], "read_text")
        self.assertTrue(same(reader.body[0].value,
                             'self.text if filename == "METADATA" else None'))
        self.assertIsInstance(function(classes[0], "locate_file").body[0], ast.Raise)
        self.assertFalse(calls(classes[0], "open"))
        self.assertFalse(calls(classes[0], "os.open"))

    def test_distribution_names_normalize_sort_and_reject_duplicates(self):
        checker = function(self.tree, "base_distributions")
        normalizers = calls(checker, "re.sub")
        self.assertEqual(len(normalizers), 1)
        self.assertTrue(same(normalizers[0].args[0], 'r"[-_.]+"'))
        self.assertTrue(same(normalizers[0].args[1], '"-"'))
        self.assert_guard(checker, "name not in observed", "distribution_rejected")
        self.assertEqual(len(calls(checker, "sorted")), 1)
        self.assertTrue(same(calls(checker, "sorted")[0].args[0], "observed"))

    def test_success_schema_is_exact_and_files_are_sorted(self):
        producer = function(self.tree, "produce")
        assignments = [node.value for node in ast.walk(producer) if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "report"
                               for target in node.targets)]
        self.assertEqual(len(assignments), 1)
        self.assertEqual({ast.literal_eval(key) for key in assignments[0].keys},
                         {"schema", "base_digest", "config_digest", "files", "base_distributions"})
        self.assertEqual(len(calls(producer, "sorted")), 1)
        self.assertTrue(same(calls(producer, "sorted")[0].args[0], "rows"))
        self.assert_guard(producer, "len(raw) <= MAXIMUM_OUTPUT_BYTES", "output_budget_exhausted")

    def test_only_stdout_writes_no_filesystem_or_process_mutation(self):
        writes = calls(self.tree, "os.write")
        self.assertEqual(len(writes), 1)
        self.assertTrue(same(writes[0].args[0], "1"))
        for name in ("open", "os.mkdir", "os.unlink", "os.rmdir", "os.chmod", "os.chown",
                     "os.setuid", "os.kill", "os.fork", "os.execve", "os.system"):
            self.assertFalse(calls(self.tree, name), name)
        strings = [node.value for node in ast.walk(self.tree)
                   if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        self.assertFalse(any(value.startswith(("/proc", "/dev")) for value in strings))

    def test_errors_are_fixed_and_partial_output_is_not_retried(self):
        classes = [node for node in self.tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "InventoryError"]
        self.assertEqual(len(classes), 1)
        constructor = function(classes[0], "__init__")
        self.assertTrue(any(same(
            node.value,
            'code if type(code) is str and code in ERROR_CODES else "inventory_failed"',
        ) for node in ast.walk(constructor) if isinstance(node, ast.Assign)))
        main = function(self.tree, "main")
        tries = [node for node in main.body if isinstance(node, ast.Try)]
        self.assertEqual(len(tries), 2)
        self.assertTrue(any(same(
            node.value, 'error.code if type(error) is InventoryError else "inventory_failed"',
        ) for node in ast.walk(tries[0]) if isinstance(node, ast.Assign)))
        self.assertFalse(calls(tries[1].handlers[0], "write_stdout"))
        self.assertTrue(all(same(node.args[0], "raw") or isinstance(node.args[0], ast.Call)
                            for node in calls(main, "write_stdout")))
        self.assertFalse(calls(self.tree, "print"))
        self.assertFalse(calls(self.tree, "repr"))


if __name__ == "__main__":
    unittest.main()
