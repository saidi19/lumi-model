"""Tiny synthetic lock bytes only: no wheel, image, runtime or dataset reads."""

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
PACKAGE = "_lumi_synthetic_oci_lock_v1"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "src/lumi")]
sys.modules[PACKAGE] = package
before_import = set(sys.modules)
lock = importlib.import_module(PACKAGE + ".posttraining_sft_oci_lock_v1")
IMPORT_ADDITIONS = set(sys.modules) - before_import
PRIVATE = "SYNTHETIC-PRIVATE-INPUT"
_UNSET = object()


def requirement(name, version="1.0", tag="py3-none-any"):
    filename = f"{name.replace('-', '_')}-{version.replace('-', '_')}-{tag}.whl"
    host = "https://files.pythonhosted.org/packages/00/11/"
    if name == "torch":
        host = "https://download.pytorch.org/whl/cpu/"
    return {
        "name": name, "version": version, "filename": filename,
        "url": host + filename.replace("+", "%2B"), "sha256": "0" * 64, "bytes": 1,
    }


def example():
    return {
        "schema": "lumi.sft-linux-cpu-release-lock.v0.1", "platform": "linux/amd64",
        "python_abi": "cp312",
        "base": {
            "repository": "docker.io/library/python", "digest": "sha256:" + "1" * 64,
            "config_digest": "sha256:" + "2" * 64, "python_version": "3.12.15",
            "tag_reference": "3.12-slim-bookworm",
        },
        "requirements": [
            requirement("certifi"), requirement("filelock"),
            requirement("safetensors", tag="cp38-abi3-manylinux_2_17_x86_64"),
            requirement("tokenizers", tag="cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64"),
            requirement("torch", "2.14.0+cpu", "cp312-cp312-manylinux_2_28_x86_64"),
            requirement("transformers"),
        ],
        "policy": {
            "device": "cpu", "strategy": "ddp", "max_context_window": 4095,
            "production_eligible": False, "real_sft_execution_permitted": False,
        },
    }


def encode(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    ).encode("ascii") + b"\n"


def validate(raw=None):
    if raw is None:
        raw = encode(example())
    return lock.validate_lock(raw, hashlib.sha256(raw).hexdigest())


class OCILockTests(unittest.TestCase):
    def reject(self, code, *, value=_UNSET, raw=None):
        if raw is None:
            raw = encode(example() if value is _UNSET else value)
        with self.assertRaises(lock.LockError) as caught:
            validate(raw)
        self.assertEqual(caught.exception.code, code)
        self.assertIn(code, lock.ERROR_CODES)
        self.assertEqual(str(caught.exception), code)
        self.assertEqual(caught.exception.args, (code,))
        self.assertNotIn(PRIVATE, repr(caught.exception))

    def test_valid_exact_snapshot_is_fresh(self):
        wanted = example()
        raw = encode(wanted)
        first, second = validate(raw), validate(raw)
        self.assertEqual(first, wanted)
        self.assertIsNot(first, second)
        for key in ("base", "policy", "requirements"):
            self.assertIsNot(first[key], second[key])
        for left, right in zip(first["requirements"], second["requirements"], strict=True):
            self.assertIsNot(left, right)
        first["base"]["digest"] = PRIVATE
        first["policy"]["production_eligible"] = True
        first["requirements"][0]["url"] = PRIVATE
        self.assertEqual(second, wanted)
        self.assertEqual(validate(raw), wanted)

    def test_fixed_contract_and_caps(self):
        self.assertEqual(lock.SCHEMA, "lumi.sft-linux-cpu-release-lock.v0.1")
        self.assertEqual(lock.PLATFORM, "linux/amd64")
        self.assertEqual(lock.PYTHON_ABI, "cp312")
        self.assertEqual(lock.MAXIMUM_LOCK_BYTES, 262144)
        self.assertEqual(lock.MINIMUM_REQUIREMENTS, 6)
        self.assertEqual(lock.MAXIMUM_REQUIREMENTS, 64)
        self.assertEqual(lock.MAXIMUM_WHEEL_BYTES, 536870912)
        self.assertEqual(lock.MAXIMUM_TOTAL_BYTES, 1073741824)

    def test_raw_requires_exact_immutable_bytes(self):
        class Bytes(bytes):
            pass

        for raw in (None, "{}", bytearray(b"{}"), memoryview(b"{}"), Bytes(b"{}"), {}):
            with self.subTest(raw_type=type(raw).__name__):
                with self.assertRaises(lock.LockError) as caught:
                    lock.validate_lock(raw, "0" * 64)
                self.assertEqual(caught.exception.code, "invalid_raw_type")

    def test_raw_budget_precedes_hash_and_json(self):
        for raw in (b"", b" " * (lock.MAXIMUM_LOCK_BYTES + 1)):
            with self.subTest(size=len(raw)), patch.object(lock.hashlib, "sha256") as digest:
                with patch.object(lock.json, "loads") as loads:
                    with self.assertRaises(lock.LockError) as caught:
                        lock.validate_lock(raw, "0" * 64)
                self.assertEqual(caught.exception.code, "lock_budget_exceeded")
                digest.assert_not_called()
                loads.assert_not_called()

    def test_external_pin_is_mandatory_lowercase_sha256(self):
        raw = encode(example())
        for pin in (None, True, [], {}, 1, "", "0" * 63, "0" * 65, "F" * 64, PRIVATE):
            with self.subTest(pin_type=type(pin).__name__):
                with self.assertRaises(lock.LockError) as caught:
                    lock.validate_lock(raw, pin)
                self.assertEqual(caught.exception.code, "invalid_expected_sha256")
        with self.assertRaises(TypeError):
            lock.validate_lock(raw)

    def test_pin_mismatch_precedes_parsing(self):
        with patch.object(lock.json, "loads") as loads:
            with self.assertRaises(lock.LockError) as caught:
                lock.validate_lock(b"SYNTHETIC-PRIVATE-INPUT", "0" * 64)
        self.assertEqual(caught.exception.code, "lock_hash_mismatch")
        loads.assert_not_called()

    def test_duplicate_keys_at_root_and_nested_levels(self):
        raw = encode(example())
        variants = (
            raw.replace(b'{"base":', b'{"base":null,"base":', 1),
            raw.replace(b'"bytes":1', b'"bytes":1,"bytes":1', 1),
            raw.replace(b'"device":"cpu"', b'"device":"cpu","device":"cpu"', 1),
        )
        for bad in variants:
            with self.subTest(prefix=bad[:10]):
                self.reject("duplicate_json_key", raw=bad)

    def test_floats_constants_and_huge_integers_are_rejected(self):
        raw = encode(example())
        for number in (b"1.0", b"1e0", b"NaN", b"Infinity", b"-Infinity", b"9" * 21):
            with self.subTest(number=number):
                bad = raw.replace(b'"bytes":1', b'"bytes":' + number)
                self.reject("invalid_json_number", raw=bad)

    def test_invalid_encoding_json_and_deep_nesting_do_not_leak(self):
        for raw in (b"{", b"\xff", b"\xef\xbb\xbf{}"):
            with self.subTest(size=len(raw)):
                self.reject("invalid_json", raw=raw)
        with patch.object(lock.json, "loads", side_effect=RecursionError(PRIVATE)):
            self.reject("invalid_json")
        with self.assertRaises(lock.LockError) as caught:
            validate(b"[" * 1200 + b"0" + b"]" * 1200)
        # CPython versions differ in their JSON recursion limit, but both
        # bounded outcomes reject this value without including input text.
        self.assertIn(caught.exception.code, {"invalid_json", "invalid_lock_shape"})

    def test_only_canonical_ascii_json_with_exactly_one_terminal_lf(self):
        wanted = example()
        raw = encode(wanted)
        variants = (
            raw[:-1], raw + b"\n", raw[:-1] + b"\r\n", b" " + raw,
            json.dumps(wanted).encode("ascii") + b"\n",
            raw.replace(b'"cpu"', br'"\u0063pu"'),
            raw.replace(b"https://", b"https:\\/\\/"),
        )
        for bad in variants:
            with self.subTest(size=len(bad)):
                self.reject("noncanonical_json", raw=bad)

    def test_external_pin_covers_the_terminal_lf(self):
        raw = encode(example())
        self.assertTrue(raw.endswith(b"\n"))
        self.assertFalse(raw.endswith(b"\n\n"))
        with self.assertRaises(lock.LockError) as caught:
            lock.validate_lock(raw, hashlib.sha256(raw[:-1]).hexdigest())
        self.assertEqual(caught.exception.code, "lock_hash_mismatch")
        self.assertEqual(validate(raw), example())

    def test_root_exact_keys_and_type(self):
        for value in ([], {}, None, 1, True, PRIVATE):
            with self.subTest(value_type=type(value).__name__):
                self.reject("invalid_lock_shape", value=value)
        for key in example():
            value = example()
            del value[key]
            with self.subTest(missing=key):
                self.reject("invalid_lock_shape", value=value)
        value = example()
        value[PRIVATE] = None
        self.reject("invalid_lock_shape", value=value)

    def test_release_contract_is_exact(self):
        for key in ("schema", "platform", "python_abi"):
            for bad in (None, [], {}, True, PRIVATE):
                value = example()
                value[key] = bad
                with self.subTest(key=key, bad_type=type(bad).__name__):
                    self.reject("unsupported_release_contract", value=value)

    def test_base_exact_keys_types_repository_and_tag(self):
        for bad in (None, [], True, PRIVATE, {}):
            value = example()
            value["base"] = bad
            with self.subTest(bad_type=type(bad).__name__):
                self.reject("invalid_base", value=value)
        for key in example()["base"]:
            value = example()
            del value["base"][key]
            with self.subTest(missing=key):
                self.reject("invalid_base", value=value)
        for key, bad in (
            ("repository", "docker.io/library/python:latest"), ("repository", PRIVATE),
            ("tag_reference", "3.13-slim-bookworm"), ("extra", True),
        ):
            value = example()
            value["base"][key] = bad
            with self.subTest(key=key):
                self.reject("invalid_base", value=value)

    def test_image_and_config_digests_are_both_sha256_pinned(self):
        for key in ("digest", "config_digest"):
            for bad in (None, True, {}, "1" * 64, "sha256:" + "F" * 64, "sha512:" + "0" * 64):
                value = example()
                value["base"][key] = bad
                with self.subTest(key=key, bad_type=type(bad).__name__):
                    self.reject("invalid_base", value=value)

    def test_python_version_is_bounded_cp312_patch(self):
        for good in ("3.12.0", "3.12.15", "3.12.999"):
            value = example()
            value["base"]["python_version"] = good
            self.assertEqual(validate(encode(value)), value)
        for bad in (None, 312, [], "3.11.9", "3.12", "3.12.1a1", "3.12.1000", " 3.12.1"):
            value = example()
            value["base"]["python_version"] = bad
            with self.subTest(bad_type=type(bad).__name__):
                self.reject("invalid_base", value=value)

    def test_requirement_collection_bounds(self):
        for bad in (None, {}, True, PRIVATE):
            value = example()
            value["requirements"] = bad
            with self.subTest(bad_type=type(bad).__name__):
                self.reject("invalid_requirements", value=value)
        for count in (0, 5, 65):
            value = example()
            value["requirements"] = [requirement("extra") for _ in range(count)]
            with self.subTest(count=count):
                self.reject("requirement_count_exceeded", value=value)
        value = example()
        value["requirements"] += [requirement(f"zzextra-{index:02d}") for index in range(58)]
        self.assertEqual(len(validate(encode(value))["requirements"]), 64)

    def test_requirement_rows_exact_keys_and_type(self):
        for bad in (None, {}, [], True, PRIVATE):
            value = example()
            value["requirements"][0] = bad
            with self.subTest(bad_type=type(bad).__name__):
                self.reject("invalid_requirement_shape", value=value)
        for key in example()["requirements"][0]:
            value = example()
            del value["requirements"][0][key]
            with self.subTest(missing=key):
                self.reject("invalid_requirement_shape", value=value)

    def test_names_are_bounded_canonical_pep_names(self):
        for bad in (None, True, [], {}, "Certifi", "certifi_", "-certifi", "cert--ifi", "a" * 129):
            value = example()
            value["requirements"][0]["name"] = bad
            with self.subTest(bad_type=type(bad).__name__):
                self.reject("invalid_name", value=value)

    def test_requirements_are_sorted_and_unique(self):
        value = example()
        value["requirements"][0:2] = reversed(value["requirements"][0:2])
        self.reject("requirements_not_sorted_unique", value=value)
        value = example()
        value["requirements"][1] = copy.deepcopy(value["requirements"][0])
        self.reject("requirements_not_sorted_unique", value=value)

    def test_each_required_core_dependency_is_mandatory_but_not_pip(self):
        self.assertNotIn("pip", {row["name"] for row in validate()["requirements"]})
        for name in lock.REQUIRED_NAMES:
            value = example()
            value["requirements"] = [row for row in value["requirements"] if row["name"] != name]
            value["requirements"].append(requirement("zz-extra"))
            with self.subTest(name=name):
                self.reject("missing_required_dependency", value=value)

    def test_gpu_and_excluded_dependencies_rejected(self):
        for name in (
            "triton", "pyspark", "mlx", "flash-attn", "nvidia-cublas-cu12", "cuda-python",
            "cupy", "cupy-cuda12x", "cuda12x", "cudnn", "lib-nccl", "nvjitlink-cu12",
        ):
            value = example()
            value["requirements"].append(requirement(name))
            value["requirements"].sort(key=lambda row: row["name"])
            with self.subTest(name=name):
                self.reject("forbidden_dependency", value=value)
        value = example()
        value["requirements"][0] = requirement("certifi", "1.0+cu128")
        self.reject("forbidden_dependency", value=value)

    def test_versions_are_bounded_safe_strings(self):
        for bad in (None, True, [], {}, "", "x" * 65, "1.0;evil", "1.0\n", "1.0/evil", "1!2"):
            value = example()
            value["requirements"][0]["version"] = bad
            with self.subTest(bad_type=type(bad).__name__):
                self.reject("invalid_version", value=value)

    def test_filenames_match_distribution_and_version(self):
        for bad in (
            None, True, [], {}, "", "../certifi-1.0-py3-none-any.whl", "x/y.whl",
            "x\\y.whl", "certifi-1.0-py3-none-any.whl\n", "x" * 240 + ".whl",
            "other-1.0-py3-none-any.whl", "certifi-2.0-py3-none-any.whl",
            "certifi-1.0-py3-none-any.tar.gz", "certifi-1.0-.whl",
            "certifi-1.0-bad-py3-none-any.whl",
        ):
            value = example()
            value["requirements"][0]["filename"] = bad
            with self.subTest(bad_type=type(bad).__name__):
                self.reject("invalid_filename", value=value)

    def test_compatible_abi3_cp312_and_pure_tags(self):
        tags = (
            "py3-none-any", "py2.py3-none-any", "cp312-cp312-manylinux_2_28_x86_64",
            "cp312-none-manylinux_2_36_x86_64", "cp38-abi3-manylinux2014_x86_64",
            "cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64",
            "cp310-abi3-manylinux2010_x86_64", "cp311-abi3-manylinux1_x86_64",
            "cp312-abi3-manylinux_2_5_x86_64",
        )
        for tag in tags:
            value = example()
            value["requirements"][0] = requirement("certifi", tag=tag)
            with self.subTest(tag=tag):
                self.assertEqual(validate(encode(value)), value)

    def test_incompatible_python_arch_os_and_future_glibc_tags(self):
        tags = (
            "cp311-cp311-manylinux_2_28_x86_64", "cp313-abi3-manylinux_2_28_x86_64",
            "cp312-cp312-win_amd64", "cp312-cp312-macosx_11_0_x86_64",
            "cp312-cp312-manylinux_2_28_aarch64", "cp312-cp312-manylinux_2_37_x86_64",
            "cp312-cp312-manylinux_2_017_x86_64", "cp312-cp312-manylinux_2_4_x86_64",
            "cp312-cp312-musllinux_1_2_x86_64", "cp312-cp312-linux_x86_64",
            "cp312-cp312-any", "py2-none-any", "py3-abi3-any", "cp312d-cp312d-any",
            "cp312-cp312-manylinux2014_x86_64.manylinux2014_x86_64",
            "cp312-cp312-manylinux_2_17_x86_64.win_amd64",
        )
        for tag in tags:
            value = example()
            value["requirements"][0] = requirement("certifi", tag=tag)
            with self.subTest(tag=tag):
                self.reject("incompatible_wheel", value=value)

    def test_build_tag_is_safe_and_optional_except_torch(self):
        value = example()
        row = value["requirements"][0]
        row["filename"] = "certifi-1.0-1build_py3-py3-none-any.whl"
        row["url"] = "https://files.pythonhosted.org/packages/" + row["filename"]
        self.assertEqual(validate(encode(value)), value)
        value = example()
        row = value["requirements"][4]
        row["filename"] = row["filename"].replace("-cp312-cp312-", "-1-cp312-cp312-")
        row["url"] = "https://download.pytorch.org/whl/cpu/" + row["filename"]
        self.reject("invalid_torch_contract", value=value)

    def test_torch_version_abi_platform_and_cpu_channel_are_fixed(self):
        for row in (
            requirement("torch", "2.13.0+cpu", "cp312-cp312-manylinux_2_28_x86_64"),
            requirement("torch", "2.14.0", "cp312-cp312-manylinux_2_28_x86_64"),
            requirement("torch", "2.14.0+cpu", "cp312-abi3-manylinux_2_28_x86_64"),
            requirement("torch", "2.14.0+cpu", "py3-none-any"),
        ):
            value = example()
            value["requirements"][4] = row
            with self.subTest(version=row["version"], filename=row["filename"]):
                self.reject("invalid_torch_contract", value=value)
        value = example()
        row = value["requirements"][4]
        row["url"] = row["url"].replace("/cpu/", "/cu128/")
        self.reject("invalid_torch_contract", value=value)

    def test_url_hosts_and_https_are_role_specific(self):
        for index, prefix in (
            (0, "http://files.pythonhosted.org/"), (0, "https://download.pytorch.org/whl/cpu/"),
            (0, "https://download-r2.pytorch.org/whl/cpu/"),
            (4, "https://files.pythonhosted.org/packages/"), (0, "https://example.org/"),
            (4, "https://download-r2.pytorch.org.evil.org/whl/cpu/"),
            (4, "https://user@download-r2.pytorch.org/whl/cpu/"),
            (4, "https://download-r2.pytorch.org:443/whl/cpu/"),
            (4, "https://download-r3.pytorch.org/whl/cpu/"),
            (0, "https://files.pythonhosted.org.evil.org/"),
            (0, "https://user@files.pythonhosted.org/"),
            (0, "https://files.pythonhosted.org:443/"),
            (0, "https://FILES.PYTHONHOSTED.ORG/"),
            (0, "HTTPS://files.pythonhosted.org/"),
        ):
            value = example()
            value["requirements"][index]["url"] = prefix + value["requirements"][index]["filename"]
            with self.subTest(prefix=prefix):
                self.reject("invalid_url", value=value)

    def test_torch_official_primary_and_r2_hosts_are_preserved_exactly(self):
        for host in ("download.pytorch.org", "download-r2.pytorch.org"):
            value = example()
            row = value["requirements"][4]
            row["url"] = row["url"].replace("download.pytorch.org", host)
            with self.subTest(host=host):
                snapshot = validate(encode(value))
                self.assertEqual(snapshot, value)
                rendered = lock.requirements_text(snapshot)
                self.assertIn(row["url"].encode("ascii"), rendered)

    def test_url_type_size_controls_query_and_fragment(self):
        original = example()["requirements"][0]["url"]
        for bad in (
            None, True, [], {}, "", "x" * 2049, original + "?", original + "#",
            original + "?x=1", original + "#a", "\n" + original,
            original.replace("https", "htt\tps"), original + " ", original + "\x7f",
            "https://[invalid/abc.whl", original.replace("packages/", "packages\\"),
        ):
            value = example()
            value["requirements"][0]["url"] = bad
            with self.subTest(bad_type=type(bad).__name__):
                self.reject("invalid_url", value=value)

    def test_url_exact_decoded_basename_and_safe_components(self):
        original = example()["requirements"][0]["url"]
        variants = (
            original.replace("certifi", "other"), original.replace("/00/", "/../"),
            original.replace("/00/", "/./"), original.replace("/00/", "//"),
            original.replace("/00/", "/%2e%2e/"), original.replace("/00/", "/%2F/"),
            original.replace("/00/", "/%252F/"), original.replace("certifi", "%63ertifi"),
            original + "/", original.replace("certifi", "certif%C3%AD"),
        )
        for bad in variants:
            value = example()
            value["requirements"][0]["url"] = bad
            with self.subTest(bad=bad):
                self.reject("invalid_url", value=value)

    def test_url_plus_is_literal_or_canonically_percent_encoded(self):
        value = example()
        row = value["requirements"][4]
        self.assertIn("%2B", row["url"])
        self.assertEqual(validate(encode(value)), value)
        row["url"] = row["url"].replace("%2B", "+")
        self.assertEqual(validate(encode(value)), value)
        row["url"] = row["url"].replace("+", "%2b")
        self.reject("invalid_url", value=value)

    def test_wheel_sha256_is_lowercase_exact_hex(self):
        for bad in (None, True, [], {}, "", "F" * 64, "0" * 63, "0" * 65, "g" * 64):
            value = example()
            value["requirements"][0]["sha256"] = bad
            with self.subTest(bad_type=type(bad).__name__):
                self.reject("invalid_wheel_sha256", value=value)

    def test_sizes_exact_positive_integers_and_caps(self):
        for bad in (None, True, False, [], {}, "1", 0, -1):
            value = example()
            value["requirements"][0]["bytes"] = bad
            with self.subTest(bad_type=type(bad).__name__):
                self.reject("invalid_wheel_size", value=value)
        value = example()
        value["requirements"][0]["bytes"] = lock.MAXIMUM_WHEEL_BYTES
        self.assertEqual(validate(encode(value)), value)
        value["requirements"][0]["bytes"] += 1
        self.reject("wheel_budget_exceeded", value=value)

    def test_total_size_cap_includes_every_row_exactly(self):
        value = example()
        value["requirements"][0]["bytes"] = lock.MAXIMUM_WHEEL_BYTES
        value["requirements"][1]["bytes"] = lock.MAXIMUM_WHEEL_BYTES - 4
        self.assertEqual(validate(encode(value)), value)
        value["requirements"][1]["bytes"] += 1
        self.reject("total_budget_exceeded", value=value)

    def test_policy_exact_keys_values_and_bool_integer_types(self):
        for bad in (None, [], {}, True, PRIVATE):
            value = example()
            value["policy"] = bad
            with self.subTest(bad_type=type(bad).__name__):
                self.reject("invalid_policy", value=value)
        for key in example()["policy"]:
            value = example()
            del value["policy"][key]
            with self.subTest(missing=key):
                self.reject("invalid_policy", value=value)
        for key, bad in (
            ("device", "cuda"), ("strategy", "fsdp"), ("max_context_window", 4096),
            ("max_context_window", True), ("production_eligible", 0),
            ("production_eligible", True), ("real_sft_execution_permitted", 0),
            ("real_sft_execution_permitted", True), ("extra", False),
        ):
            value = example()
            value["policy"][key] = bad
            with self.subTest(key=key, bad=bad):
                self.reject("invalid_policy", value=value)

    def test_requirements_text_exact_sorted_hashed_direct_urls(self):
        value = validate()
        expected = "".join(
            f"{row['name']} @ {row['url']} --hash=sha256:{row['sha256']}\n"
            for row in value["requirements"]
        ).encode("ascii")
        self.assertEqual(lock.requirements_text(value), expected)
        self.assertTrue(expected.endswith(b"\n"))
        self.assertNotIn(b"--index-url", expected)
        self.assertNotIn(b"--extra-index-url", expected)
        self.assertEqual(len(expected.splitlines()), 6)
        self.assertEqual(value, example())

    def test_requirements_renderer_rejects_mutation_and_option_injection(self):
        for key, bad, code in (
            ("name", "--index-url", "invalid_name"),
            ("url", "https://files.pythonhosted.org/a.whl\n--trusted-host evil", "invalid_url"),
            ("sha256", "0" * 64 + " --no-deps", "invalid_wheel_sha256"),
        ):
            value = validate()
            value["requirements"][0][key] = bad
            with self.subTest(key=key), self.assertRaises(lock.LockError) as caught:
                lock.requirements_text(value)
            self.assertEqual(caught.exception.code, code)

    def test_errors_are_always_fixed_allowlisted_codes(self):
        for code in (None, {}, 1, PRIVATE):
            error = lock.LockError(code)
            self.assertEqual(error.code, "invalid_lock_shape")
            self.assertEqual(str(error), "invalid_lock_shape")
            self.assertNotIn(PRIVATE, repr(error))

    def test_validation_and_rendering_do_not_perform_io(self):
        raw = encode(example())
        digest = hashlib.sha256(raw).hexdigest()
        with patch("builtins.open", side_effect=AssertionError("file access forbidden")):
            value = lock.validate_lock(raw, digest)
            rendered = lock.requirements_text(value)
        self.assertEqual(len(rendered.splitlines()), 6)
        self.assertFalse(value["policy"]["production_eligible"])
        self.assertFalse(value["policy"]["real_sft_execution_permitted"])

    def test_import_additions_exclude_ml_and_real_lumi_package(self):
        forbidden = {"torch", "numpy", "transformers", "tensorflow", "jax", "lumi"}
        self.assertFalse({name.split(".")[0] for name in IMPORT_ADDITIONS} & forbidden)


if __name__ == "__main__":
    unittest.main()
