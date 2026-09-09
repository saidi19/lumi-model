"""Synthetic supplied bytes only; no runtime, model, dataset or native fixture."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_lumi_synthetic_release_manifest_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
before_import = set(sys.modules)
manifest = importlib.import_module(PACKAGE + ".posttraining_sft_release_manifest_v1")
IMPORT_ADDITIONS = set(sys.modules) - before_import
PRIVATE = "SYNTHETIC-PRIVATE-INPUT"
BUDGETS = {
    "maximum_manifest_bytes": 65536,
    "maximum_entries": 128,
    "maximum_file_bytes": 1024,
    "maximum_total_bytes": 4096,
}


def row(path, role, *, size=1, mode=0o444):
    return {"path": path, "role": role, "bytes": size, "mode": mode, "sha256": "0" * 64}


def example():
    return {
        "schema": "lumi.sft-release.v0.1", "platform": "linux-x86_64", "python_abi": "cp312",
        "interpreter": "runtime/bin/python3.12", "entrypoint": "source/worker.py",
        "files": [
            row("dependencies/pkg/__init__.py", "dependency"),
            row("runtime/bin/python3.12", "runtime", mode=0o555),
            row("source/worker.py", "source"),
        ],
    }


def encode(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    ).encode("ascii")


def validate(raw=None, **budgets):
    if raw is None:
        raw = encode(example())
    return manifest.validate_manifest(raw, hashlib.sha256(raw).hexdigest(), **(BUDGETS | budgets))


class ManifestTests(unittest.TestCase):
    def reject(self, code, *, value=None, raw=None, **budgets):
        if raw is None:
            raw = encode(example() if value is None else value)
        with self.assertRaises(manifest.ManifestError) as caught:
            validate(raw, **budgets)
        self.assertEqual(caught.exception.code, code)
        self.assertIn(code, manifest.ERROR_CODES)
        self.assertEqual(str(caught.exception), code)
        self.assertEqual(caught.exception.args, (code,))
        self.assertNotIn(PRIVATE, repr(caught.exception))
        return caught.exception

    def test_valid_manifest_is_exact_schema_and_fresh_on_each_call(self):
        wanted = example()
        raw = encode(wanted)
        first, second = validate(raw), validate(raw)
        self.assertEqual(first, wanted)
        self.assertEqual(first.keys(), manifest.MANIFEST_KEYS)
        self.assertIsNot(first, second)
        self.assertIsNot(first["files"], second["files"])
        for left, right in zip(first["files"], second["files"], strict=True):
            self.assertIsNot(left, right)
        first["schema"] = PRIVATE
        first["files"][0]["path"] = PRIVATE
        first["files"].clear()
        self.assertEqual(second, wanted)
        self.assertEqual(validate(raw), wanted)

    def test_constant_contract_and_absolute_caps(self):
        self.assertEqual(manifest.SCHEMA, "lumi.sft-release.v0.1")
        self.assertEqual(manifest.PLATFORM, "linux-x86_64")
        self.assertEqual(manifest.PYTHON_ABI, "cp312")
        self.assertEqual(manifest.MAXIMUM_MANIFEST_BYTES, 8 * 1024**2)
        self.assertEqual(manifest.MAXIMUM_ENTRIES, 32768)
        self.assertEqual(manifest.MAXIMUM_FILE_BYTES, 2 * 1024**3)
        self.assertEqual(manifest.MAXIMUM_TOTAL_BYTES, 16 * 1024**3)
        self.assertEqual(manifest.ROLE_PREFIXES, {
            "runtime": "runtime/", "dependency": "dependencies/", "source": "source/",
        })
        self.assertEqual(manifest.ALLOWED_MODES, {
            0o400, 0o444, 0o500, 0o555, 0o600, 0o644, 0o700, 0o755,
        })

    def test_all_budgets_are_explicitly_required(self):
        raw = encode(example())
        for key in BUDGETS:
            with self.subTest(key=key), self.assertRaises(TypeError):
                manifest.validate_manifest(
                    raw, hashlib.sha256(raw).hexdigest(),
                    **{name: value for name, value in BUDGETS.items() if name != key},
                )

    def test_budget_values_are_exact_positive_bounded_integers(self):
        caps = dict(zip(BUDGETS, (
            manifest.MAXIMUM_MANIFEST_BYTES, manifest.MAXIMUM_ENTRIES,
            manifest.MAXIMUM_FILE_BYTES, manifest.MAXIMUM_TOTAL_BYTES,
        ), strict=True))
        for name, cap in caps.items():
            for bad in (False, True, 0, -1, 1.0, "1", None, [], {}, cap + 1, 2**100):
                with self.subTest(name=name, bad=bad):
                    self.reject("invalid_budget", **{name: bad})
        self.assertEqual(validate(**caps), example())

    def test_integer_subclasses_cannot_supply_budgets(self):
        class Integer(int):
            pass

        for name in BUDGETS:
            with self.subTest(name=name):
                self.reject("invalid_budget", **{name: Integer(1)})

    def test_raw_is_exact_immutable_bytes(self):
        class Bytes(bytes):
            pass

        for raw in ("{}", bytearray(b"{}"), memoryview(b"{}"), Bytes(b"{}"), {}, None, 1):
            with self.subTest(raw_type=type(raw).__name__):
                with self.assertRaises(manifest.ManifestError) as caught:
                    manifest.validate_manifest(raw, "0" * 64, **BUDGETS)
                self.assertEqual(caught.exception.code, "invalid_raw_type")

    def test_raw_size_is_checked_before_hashing_or_parsing(self):
        for raw, limit in ((b"", 1), (b"{}", 1)):
            with self.subTest(raw=raw):
                with patch.object(manifest.hashlib, "sha256") as digest:
                    with patch.object(manifest.json, "loads") as loads:
                        with self.assertRaises(manifest.ManifestError) as caught:
                            manifest.validate_manifest(
                                raw, "0" * 64, **(BUDGETS | {"maximum_manifest_bytes": limit}),
                            )
                self.assertEqual(caught.exception.code, "manifest_budget_exceeded")
                digest.assert_not_called()
                loads.assert_not_called()

    def test_manifest_exact_raw_budget_boundary(self):
        raw = encode(example())
        self.assertEqual(validate(raw, maximum_manifest_bytes=len(raw)), example())
        self.reject("manifest_budget_exceeded", raw=raw, maximum_manifest_bytes=len(raw) - 1)

    def test_external_pin_is_mandatory_exact_lowercase_sha256(self):
        raw = encode(example())
        good = hashlib.sha256(raw).hexdigest()
        for pin in (None, 0, [], {}, True, "", good[:-1], good + "0", good.upper(), "g" * 64):
            with self.subTest(pin=pin), patch.object(manifest.json, "loads") as loads:
                with self.assertRaises(manifest.ManifestError) as caught:
                    manifest.validate_manifest(raw, pin, **BUDGETS)
                self.assertEqual(caught.exception.code, "invalid_expected_sha256")
                loads.assert_not_called()
        with self.assertRaises(TypeError):
            manifest.validate_manifest(raw, **BUDGETS)

    def test_hash_mismatch_stops_before_json_decode(self):
        with patch.object(manifest.json, "loads") as loads:
            with self.assertRaises(manifest.ManifestError) as caught:
                manifest.validate_manifest(encode(example()), "0" * 64, **BUDGETS)
            self.assertEqual(caught.exception.code, "manifest_hash_mismatch")
            loads.assert_not_called()

    def test_digest_covers_exact_raw_bytes_not_recanonicalized_value(self):
        raw = encode(example())
        with self.assertRaises(manifest.ManifestError) as caught:
            manifest.validate_manifest(raw + b"\n", hashlib.sha256(raw).hexdigest(), **BUDGETS)
        self.assertEqual(caught.exception.code, "manifest_hash_mismatch")

    def test_invalid_json_has_only_fixed_error(self):
        for raw in (b"{", b"{}junk", b"\xff", b"\xef\xbb\xbf{}", b"\x00", b'{"' +
                    PRIVATE.encode() + b'":'):
            with self.subTest(raw_size=len(raw)):
                error = self.reject("invalid_json", raw=raw)
                self.assertTrue(error.__suppress_context__)

    def test_nested_non_schema_json_fails_closed_across_parser_recursion_limits(self):
        raw = b"[" * 2000 + b"]" * 2000
        with self.assertRaises(manifest.ManifestError) as caught:
            validate(raw)
        self.assertIn(caught.exception.code, ("invalid_json", "invalid_manifest_shape"))
        with patch.object(manifest.json, "loads", side_effect=RecursionError(PRIVATE)):
            self.reject("invalid_json")

    def test_duplicate_keys_are_rejected_at_root_and_in_file_rows(self):
        raw = encode(example())
        self.reject("duplicate_json_key", raw=b'{"schema":"duplicate",' + raw[1:])
        self.reject("duplicate_json_key", raw=raw.replace(b'"bytes":1', b'"bytes":1,"bytes":1', 1))

    def test_floats_and_nonfinite_numbers_are_rejected_during_parse(self):
        raw = encode(example())
        for number in (b"1.0", b"1e0", b"-0.0", b"NaN", b"Infinity", b"-Infinity", b"1e999"):
            with self.subTest(number=number):
                self.reject(
                    "invalid_json_number", raw=raw.replace(b'"bytes":1', b'"bytes":' + number),
                )

    def test_extremely_long_decimal_is_bounded_before_integer_conversion(self):
        raw = encode(example()).replace(b'"bytes":1', b'"bytes":' + b"9" * 5000, 1)
        self.reject("invalid_json_number", raw=raw)

    def test_canonical_json_rejects_whitespace_key_order_and_escape_aliases(self):
        value = example()
        raw = encode(value)
        variants = (
            raw + b"\n", b" " + raw, raw.replace(b",", b", ", 1),
            json.dumps(value, separators=(",", ":")).encode("ascii"),
            raw.replace(b"source/worker.py", b"source\\/worker.py"),
            raw.replace(b'"schema"', b'"\\u0073chema"'),
        )
        for alternative in variants:
            with self.subTest(raw_size=len(alternative)):
                self.reject("noncanonical_json", raw=alternative)

    def test_negative_zero_is_noncanonical_even_when_size_is_allowed_zero(self):
        value = example()
        value["files"][0]["bytes"] = 0
        raw = encode(value).replace(b'"bytes":0', b'"bytes":-0', 1)
        self.reject("noncanonical_json", raw=raw)

    def test_root_requires_exact_object_schema(self):
        for value in ([], None, True, 1, "manifest", {}, {"schema": manifest.SCHEMA}):
            with self.subTest(value=value):
                self.reject("invalid_manifest_shape", raw=encode(value))
        for key in example():
            value = example()
            del value[key]
            with self.subTest(missing=key):
                self.reject("invalid_manifest_shape", value=value)
        for key in ("production_eligible", "runtime_origin_authenticated", "signature", PRIVATE):
            value = example() | {key: True}
            with self.subTest(extra=key):
                self.reject("invalid_manifest_shape", value=value)

    def test_release_schema_platform_and_python_abi_are_frozen(self):
        for key in ("schema", "platform", "python_abi"):
            for bad in (PRIVATE, "", True, 1, [], {}, None):
                with self.subTest(key=key, bad=bad):
                    self.reject("unsupported_release_contract", value=example() | {key: bad})

    def test_files_must_be_a_list_and_respect_entry_budget(self):
        for bad in ({}, None, False, "files", 1):
            with self.subTest(bad=bad):
                self.reject("invalid_files", value=example() | {"files": bad})
        self.reject("entry_budget_exceeded", value=example() | {"files": []})
        self.reject("entry_budget_exceeded", maximum_entries=2)
        self.assertEqual(validate(maximum_entries=3), example())

    def test_file_row_requires_exact_object_keys(self):
        for bad in (None, [], False, 1, "row", {}):
            value = example()
            value["files"][0] = bad
            with self.subTest(bad=bad):
                self.reject("invalid_file_shape", value=value)
        for key in manifest.FILE_KEYS:
            value = example()
            del value["files"][0][key]
            with self.subTest(missing=key):
                self.reject("invalid_file_shape", value=value)
        for key in ("type", "target", "symlink", "owner", "executable", PRIVATE):
            value = example()
            value["files"][0][key] = "symlink"
            with self.subTest(extra=key):
                self.reject("invalid_file_shape", value=value)

    def test_path_rejects_aliases_escape_and_non_ascii_components(self):
        bad_paths = (
            "", "/dependencies/a", "//dependencies/a", "dependencies", "dependencies/",
            "dependencies//a", "dependencies/./a", "dependencies/../a", "dependencies/a/..",
            "dependencies/a/.", "dependencies/..", "dependencies/.", "dependencies/a\\b",
            "dependencies/a b", "dependencies/a\tb", "dependencies/a\nb", "dependencies/a\x00b",
            "dependencies/a\x7fb", "dependencies/é", "dependencies/\ud800", "Dependencies/a",
            "tmp/a", "./dependencies/a", "dependencies/a:b", "dependencies/a%b",
            "dependencies/a@b", "dependencies/a?b", "dependencies/a#b",
        )
        for bad in (*bad_paths, None, True, 1, [], {}):
            value = example()
            value["files"][0]["path"] = bad
            with self.subTest(path=bad):
                self.reject("invalid_path", value=value)

    def test_path_component_length_depth_and_total_length_bounds(self):
        for path in (
            "dependencies/" + "a" * 129,
            "dependencies/" + "/".join(["a"] * 16),
            "dependencies/" + "/".join(["a" * 128] * 8),
        ):
            value = example()
            value["files"][0]["path"] = path
            with self.subTest(size=len(path)):
                self.reject("invalid_path", value=value)
        for path in (
            "dependencies/" + "a" * 128,
            "dependencies/" + "/".join(["a"] * 15),
            "dependencies/" + "/".join(["a" * 128] * 7) + "/" + "a" * 108,
        ):
            value = example()
            value["files"][0]["path"] = path
            with self.subTest(size=len(path)):
                self.assertLessEqual(len(path), 1024)
                self.assertEqual(validate(encode(value)), value)

    def test_canonical_component_alphabet_allows_expected_release_filenames(self):
        value = example()
        value["files"][0]["path"] = "dependencies/Pkg_1.0+cpu/-hidden...so"
        self.assertEqual(validate(encode(value)), value)

    def test_file_hash_is_exact_lowercase_hex(self):
        for bad in (None, True, 1, [], {}, "", "0" * 63, "0" * 65, "aB" * 32, "g" * 64):
            value = example()
            value["files"][0]["sha256"] = bad
            with self.subTest(bad=bad):
                self.reject("invalid_file_sha256", value=value)

    def test_file_sizes_are_exact_nonnegative_integers(self):
        for bad in (-1, True, False, None, "1", [], {}):
            value = example()
            value["files"][0]["bytes"] = bad
            with self.subTest(bad=bad):
                self.reject("invalid_file_size", value=value)
        value = example()
        for item in value["files"]:
            item["bytes"] = 0
        self.assertEqual(
            validate(encode(value), maximum_file_bytes=1, maximum_total_bytes=1), value,
        )

    def test_exact_file_and_total_budget_limits(self):
        value = example()
        value["files"][0]["bytes"] = 10
        raw = encode(value)
        self.assertEqual(validate(raw, maximum_file_bytes=10, maximum_total_bytes=12), value)
        self.reject("file_budget_exceeded", raw=raw, maximum_file_bytes=9)
        self.reject("total_budget_exceeded", raw=raw, maximum_total_bytes=11)

    def test_absolute_two_gib_file_size_is_metadata_only(self):
        value = example()
        value["files"][0]["bytes"] = manifest.MAXIMUM_FILE_BYTES
        self.assertEqual(validate(
            encode(value), maximum_file_bytes=manifest.MAXIMUM_FILE_BYTES,
            maximum_total_bytes=manifest.MAXIMUM_TOTAL_BYTES,
        ), value)
        value["files"][0]["bytes"] += 1
        self.reject(
            "file_budget_exceeded", value=value, maximum_file_bytes=manifest.MAXIMUM_FILE_BYTES,
            maximum_total_bytes=manifest.MAXIMUM_TOTAL_BYTES,
        )

    def test_sixteen_gib_aggregate_boundary_without_allocating_file_data(self):
        value = example()
        for item in value["files"]:
            item["bytes"] = 0
        value["files"] += [
            row(f"source/z{index}", "source", size=manifest.MAXIMUM_FILE_BYTES)
            for index in range(8)
        ]
        caps = {
            "maximum_file_bytes": manifest.MAXIMUM_FILE_BYTES,
            "maximum_total_bytes": manifest.MAXIMUM_TOTAL_BYTES,
        }
        self.assertEqual(validate(encode(value), **caps), value)
        value["files"][0]["bytes"] = 1
        self.reject("total_budget_exceeded", value=value, **caps)

    def test_modes_reject_special_bits_world_or_group_write_and_type_aliases(self):
        for bad in (0, 0o111, 0o440, 0o666, 0o770, 0o777, 0o4755, 0o2755, 0o1755,
                    0o100444, -1, True, False, None, "444", [], {}):
            value = example()
            value["files"][0]["mode"] = bad
            with self.subTest(bad=bad):
                self.reject("invalid_mode", value=value)
        for mode in manifest.ALLOWED_MODES:
            value = example()
            value["files"][0]["mode"] = mode
            with self.subTest(mode=mode):
                self.assertEqual(validate(encode(value)), value)

    def test_roles_are_exact_and_bound_to_distinct_path_prefixes(self):
        for bad in ("dependencies", "Runtime", "symlink", "", PRIVATE, None, True, [], {}):
            value = example()
            value["files"][0]["role"] = bad
            with self.subTest(bad=bad):
                self.reject("invalid_role", value=value)
        for role in ("runtime", "source"):
            value = example()
            value["files"][0]["role"] = role
            with self.subTest(role=role):
                self.reject("role_path_mismatch", value=value)

    def test_each_of_three_roles_is_required(self):
        for role in manifest.ROLE_PREFIXES:
            value = example()
            value["files"] = [item for item in value["files"] if item["role"] != role]
            with self.subTest(role=role):
                self.reject("missing_required_role", value=value)

    def test_files_must_be_sorted_and_unique(self):
        value = example()
        value["files"] = list(reversed(value["files"]))
        self.reject("files_not_sorted_unique", value=value)
        for mutate_hash in (False, True):
            value = example()
            duplicate = dict(value["files"][0])
            if mutate_hash:
                duplicate["sha256"] = "f" * 64
            value["files"].insert(1, duplicate)
            with self.subTest(mutate_hash=mutate_hash):
                self.reject("files_not_sorted_unique", value=value)

    def test_file_directory_prefix_conflicts_are_rejected(self):
        for parent, child in (("dependencies/pkg", "dependencies/pkg/__init__.py"),
                              ("source/a", "source/a/b/c.py")):
            value = example()
            role = "dependency" if parent.startswith("dependencies/") else "source"
            value["files"] += [row(parent, role), row(child, role)]
            value["files"].sort(key=lambda item: item["path"])
            with self.subTest(parent=parent):
                self.reject("file_directory_conflict", value=value)

    def test_prefix_conflict_need_not_be_adjacent_in_lexical_order(self):
        value = example()
        value["files"] += [row(path, "source") for path in ("source/a", "source/a-z", "source/a/b")]
        value["files"].sort(key=lambda item: item["path"])
        self.reject("file_directory_conflict", value=value)

    def test_shared_directory_and_filename_prefix_without_slash_are_valid(self):
        value = example()
        value["files"] += [row(path, "source") for path in (
            "source/a", "source/a-z", "source/ab", "source/package/a", "source/package/b",
        )]
        value["files"].sort(key=lambda item: item["path"])
        self.assertEqual(validate(encode(value)), value)

    def test_interpreter_must_name_a_listed_runtime_file(self):
        for bad in ("runtime/missing", "dependencies/pkg/__init__.py", "source/worker.py",
                    "/runtime/bin/python3.12", None, True, 1, [], {}):
            with self.subTest(bad=bad):
                self.reject("invalid_interpreter", value=example() | {"interpreter": bad})

    def test_interpreter_requires_execute_bit(self):
        for mode in manifest.ALLOWED_MODES:
            value = example()
            value["files"][1]["mode"] = mode
            with self.subTest(mode=mode):
                if mode & 0o111:
                    self.assertEqual(validate(encode(value)), value)
                else:
                    self.reject("invalid_interpreter", value=value)

    def test_entrypoint_must_name_a_listed_source_file(self):
        for bad in ("source/missing.py", "dependencies/pkg/__init__.py", "runtime/bin/python3.12",
                    "/source/worker.py", None, True, 1, [], {}):
            with self.subTest(bad=bad):
                self.reject("invalid_entrypoint", value=example() | {"entrypoint": bad})
        for mode in manifest.ALLOWED_MODES:
            value = example()
            value["files"][2]["mode"] = mode
            with self.subTest(mode=mode):
                self.assertEqual(validate(encode(value)), value)

    def test_error_constructor_cannot_retain_unapproved_or_private_codes(self):
        for code in (PRIVATE, [], {}, None, 1, True):
            error = manifest.ManifestError(code)
            self.assertEqual(error.code, "invalid_manifest_shape")
            self.assertEqual(error.args, ("invalid_manifest_shape",))
        for code in manifest.ERROR_CODES:
            self.assertEqual(manifest.ManifestError(code).code, code)

    def test_manifest_validation_performs_no_filesystem_or_runtime_action(self):
        raw = encode(example())
        with patch("builtins.open", side_effect=AssertionError("unexpected_open")):
            with patch.object(Path, "open", side_effect=AssertionError("unexpected_path_open")):
                self.assertEqual(validate(raw), example())
        self.assertFalse(any(name.split(".")[0] in {"torch", "numpy", "transformers"}
                             for name in IMPORT_ADDITIONS))
        self.assertNotIn("lumi", IMPORT_ADDITIONS)

    def test_no_validation_pass_can_add_origin_or_execution_authority(self):
        result = validate()
        for key in ("runtime_origin_authenticated", "production_eligible", "execution_enabled",
                    "real_sft_execution_permitted", "real_host_admission_verified"):
            self.assertNotIn(key, result)
        self.assertEqual(result, copy.deepcopy(example()))


if __name__ == "__main__":
    unittest.main()
