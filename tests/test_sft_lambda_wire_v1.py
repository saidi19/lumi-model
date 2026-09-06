"""Pure synthetic response fixtures: no API, credentials, instance or host calls."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_synthetic_lambda_wire_v1", ROOT / "src/lumi/posttraining_sft_lambda_wire_v1.py",
)
assert SPEC is not None and SPEC.loader is not None
wire = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wire)
INSTANCE_ID = "0123456789abcdef0123456789abcdef"
OTHER_ID = "fedcba9876543210fedcba9876543210"


def encode(value):
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def instance(*, status="active"):
    return {
        "id": INSTANCE_ID, "status": status,
        "region": {"name": "us-east-1", "description": "synthetic region"},
        "instance_type": {
            "name": "gpu_1x_h100_sxm5", "price_cents_per_hour": 329,
            "specs": {"gpus": 1, "vcpus": 26, "memory_gib": 225},
        },
        "jupyter_token": "SECRET_NOTEBOOK_TOKEN",
        "jupyter_url": "https://SECRET_HOST.invalid/?token=SECRET_TOKEN",
        "ssh_key_names": ["SECRET_KEY_LABEL"],
        "unknown_valid_json": {"a": [None, True, False, -3, 1.5, "synthetic"]},
    }


def launch(identifier=INSTANCE_ID):
    return encode({"data": {"instance_ids": [identifier]}})


class WireTests(unittest.TestCase):
    def rejected(self, function, body, *arguments):
        with self.assertRaises(wire.WireError) as caught:
            function(body, *arguments)
        self.assertEqual(caught.exception.code, "response_rejected")
        self.assertEqual(str(caught.exception), "response_rejected")
        self.assertNotIn("SECRET", repr(caught.exception))


class LaunchTests(WireTests):
    def test_one_canonical_identifier_is_returned(self):
        self.assertEqual(wire.parse_launch(launch()), INSTANCE_ID)
        self.assertEqual(wire.parse_launch(launch("0" * 32)), "0" * 32)
        self.assertEqual(wire.parse_launch(launch("f" * 32)), "f" * 32)

    def test_identifier_profile_rejects_types_case_length_hyphens_and_whitespace(self):
        for identifier in (None, True, 1, [], {}, "", "0" * 31, "0" * 33, "A" * 32,
                           "g" * 32, "0" * 31 + " ", "0" * 31 + "\n", "0" * 31 + "é",
                           "01234567-89ab-cdef-0123-456789abcdef"):
            with self.subTest(type_name=type(identifier).__name__):
                self.rejected(wire.parse_launch, launch(identifier))

    def test_exact_outer_and_operation_shape_and_count(self):
        for value in (None, [], {}, {"error": "SECRET"}, {"data": None},
                      {"data": {"instance_ids": []}},
                      {"data": {"instance_ids": [INSTANCE_ID, OTHER_ID]}},
                      {"data": {"instance_ids": INSTANCE_ID}},
                      {"data": {"instance_ids": [INSTANCE_ID], "extra": 1}},
                      {"data": {"instance_ids": [INSTANCE_ID]}, "extra": "SECRET"}):
            with self.subTest(type_name=type(value).__name__):
                self.rejected(wire.parse_launch, encode(value))


class InstanceTests(WireTests):
    def test_only_six_whitelisted_fields_are_projected(self):
        projected = wire.parse_instance(encode({"data": instance()}))
        self.assertEqual(projected, {
            "instance_id": INSTANCE_ID, "status": "active", "region": "us-east-1",
            "instance_type": "gpu_1x_h100_sxm5", "price_cents_per_hour": 329, "gpu_count": 1,
        })
        self.assertNotIn("SECRET", repr(projected))
        self.assertNotIn("jupyter", repr(projected))
        self.assertNotIn("ssh", repr(projected))

    def test_every_official_status_is_preserved(self):
        self.assertEqual(wire.STATUSES, {
            "booting", "active", "unhealthy", "terminated", "terminating", "preempted",
        })
        for status in wire.STATUSES:
            with self.subTest(status=status):
                self.assertEqual(wire.parse_instance(encode({"data": instance(status=status)}))[
                    "status"], status)

    def test_unknown_status_and_wrong_status_types_rejected(self):
        for status in ("running", "ACTIVE", "active ", "", None, 1, True, [], {}):
            with self.subTest(type_name=type(status).__name__):
                self.rejected(wire.parse_instance, encode({"data": instance(status=status)}))

    def test_every_required_security_field_must_exist(self):
        paths = (("id",), ("status",), ("region",), ("region", "name"), ("instance_type",),
                 ("instance_type", "name"), ("instance_type", "price_cents_per_hour"),
                 ("instance_type", "specs"), ("instance_type", "specs", "gpus"))
        for path in paths:
            value = instance()
            target = value
            for key in path[:-1]:
                target = target[key]
            del target[path[-1]]
            with self.subTest(path=path):
                self.rejected(wire.parse_instance, encode({"data": value}))

    def test_nested_container_types_and_no_top_level_price_substitution(self):
        for path in (("region",), ("instance_type",), ("instance_type", "specs")):
            for wrong in (None, [], True, 1, "SECRET"):
                value = instance()
                target = value
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = wrong
                with self.subTest(path=path, type_name=type(wrong).__name__):
                    self.rejected(wire.parse_instance, encode({"data": value}))
        value = instance()
        value["price_cents_per_hour"] = value["instance_type"].pop("price_cents_per_hour")
        self.rejected(wire.parse_instance, encode({"data": value}))

    def test_price_requires_bounded_positive_integer_without_boolean_or_float_aliases(self):
        for price in (1, wire.MAXIMUM_PRICE_CENTS_PER_HOUR):
            value = instance()
            value["instance_type"]["price_cents_per_hour"] = price
            self.assertEqual(wire.parse_instance(encode({"data": value}))["price_cents_per_hour"],
                             price)
        for price in (0, -1, 1_500_001, True, False, 1.0, "1", None, [], {}, 2**100):
            value = instance()
            value["instance_type"]["price_cents_per_hour"] = price
            with self.subTest(type_name=type(price).__name__):
                self.rejected(wire.parse_instance, encode({"data": value}))

    def test_gpu_count_requires_one_to_eight_strict_integers(self):
        for count in range(1, 9):
            value = instance()
            value["instance_type"]["specs"]["gpus"] = count
            self.assertEqual(wire.parse_instance(encode({"data": value}))["gpu_count"], count)
        for count in (0, -1, 9, True, False, 1.0, "1", None, [], {}):
            value = instance()
            value["instance_type"]["specs"]["gpus"] = count
            with self.subTest(type_name=type(count).__name__):
                self.rejected(wire.parse_instance, encode({"data": value}))

    def test_region_and_type_names_are_bounded_ascii_labels(self):
        for path in ("region", "instance_type"):
            for name in ("a", "a" * 64, "us-east-1", "gpu_8x_h100"):
                value = instance()
                value[path]["name"] = name
                self.assertIn(name, wire.parse_instance(encode({"data": value})).values())
            for name in ("", "a" * 65, "US", "a.b", "a/b", "a b", "a\n", "é", True, [], None):
                value = instance()
                value[path]["name"] = name
                with self.subTest(path=path, type_name=type(name).__name__):
                    self.rejected(wire.parse_instance, encode({"data": value}))

    def test_instance_identifier_validation_not_only_launch_validation(self):
        value = instance()
        value["id"] = "wrong"
        self.rejected(wire.parse_instance, encode({"data": value}))


class TerminationTests(WireTests):
    def test_terminating_is_preserved_and_not_upgraded_to_final_proof(self):
        for status in ("terminating", "terminated"):
            with self.subTest(status=status):
                body = encode({"data": {"terminated_instances": [instance(status=status)]}})
                self.assertEqual(wire.parse_termination(body, INSTANCE_ID), status)

    def test_expected_id_must_be_canonical_and_exact(self):
        body = encode({"data": {"terminated_instances": [instance(status="terminating")]}})
        for expected in (OTHER_ID, "bad", "A" * 32, True, None, 1, [], {}):
            with self.subTest(type_name=type(expected).__name__):
                self.rejected(wire.parse_termination, body, expected)

    def test_other_valid_instance_statuses_are_not_termination_acceptance(self):
        for status in wire.STATUSES - wire.TERMINATION_STATUSES:
            with self.subTest(status=status):
                body = encode({"data": {"terminated_instances": [instance(status=status)]}})
                self.rejected(wire.parse_termination, body, INSTANCE_ID)

    def test_exact_operation_shape_and_one_returned_instance(self):
        item = instance(status="terminating")
        for data in (None, [], {}, {"terminated_instances": []},
                     {"terminated_instances": [item, item]},
                     {"terminated_instances": item}, {"terminated_instances": [item], "extra": 1},
                     {"instance_ids": [INSTANCE_ID]}):
            with self.subTest(type_name=type(data).__name__):
                self.rejected(wire.parse_termination, encode({"data": data}), INSTANCE_ID)

    def test_termination_requires_full_minimal_projection_not_id_status_only(self):
        for item in ({"id": INSTANCE_ID, "status": "terminated"}, None, [], True):
            with self.subTest(type_name=type(item).__name__):
                self.rejected(wire.parse_termination,
                              encode({"data": {"terminated_instances": [item]}}), INSTANCE_ID)
        item = instance(status="terminated")
        item["instance_type"]["specs"]["gpus"] = True
        self.rejected(wire.parse_termination,
                      encode({"data": {"terminated_instances": [item]}}), INSTANCE_ID)


class DecodingTests(WireTests):
    def test_duplicate_keys_rejected_at_all_depths_even_discarded_extras(self):
        valid = encode({"data": instance()})
        cases = (
            b'{"data":{},"data":{}}',
            b'{"data":{},"\\u0064ata":{}}',
            valid.replace(b'"status":"active"', b'"status":"active","status":"active"'),
            valid.replace(b'"name":"us-east-1"', b'"name":"us-east-1","name":"us-east-1"'),
            valid.replace(b'"gpus":1', b'"gpus":1,"gpus":1'),
            valid.replace(b'"unknown_valid_json":{', b'"unknown_valid_json":{"x":1,"x":2,'),
        )
        for body in cases:
            with self.subTest(length=len(body)):
                self.rejected(wire.parse_instance, body)
        self.rejected(wire.parse_launch,
                      b'{"data":{"instance_ids":[],"instance_ids":["' + INSTANCE_ID.encode()
                      + b'"]}}')

    def test_nonfinite_literals_and_float_overflow_rejected_even_in_dropped_extras(self):
        valid = encode({"data": instance()})
        for number in (b"NaN", b"Infinity", b"-Infinity", b"1e999", b"-1e999"):
            with self.subTest(number=number):
                body = valid.replace(b'"unknown_valid_json":{', b'"unknown_valid_json":{"bad":'
                                     + number + b",")
                self.rejected(wire.parse_instance, body)

    def test_only_strict_utf8_without_bom_is_accepted(self):
        text = launch().decode("utf-8")
        for body in (b"\xef\xbb\xbf" + launch(), text.encode("utf-16"), text.encode("utf-16-le"),
                     text.encode("utf-32"), text.encode("utf-32-le"), b"\xff" + launch(),
                     b' {"data":"\xc0\xaf"}', b' {"data":"\xed\xa0\x80"}'):
            with self.subTest(length=len(body)):
                self.rejected(wire.parse_launch, body)

    def test_invalid_escaped_unicode_rejected_but_valid_unicode_extras_dropped(self):
        value = instance()
        value["extra_unicode"] = "বাংলা 😀"
        self.assertEqual(wire.parse_instance(encode({"data": value}))["instance_id"], INSTANCE_ID)
        for surrogate in ("\ud800", "\udfff"):
            value["extra_unicode"] = surrogate
            self.rejected(wire.parse_instance, encode({"data": value}))

    def test_wrong_types_empty_truncated_and_trailing_documents_rejected(self):
        valid = launch()
        for body in (None, [], {}, 1, True, valid.decode(), bytearray(valid), memoryview(valid),
                     b"", b" ", valid[:-1], valid + b"{}", b'{"data":', b'{"data": /*x*/ 1}',
                     b'{"data":1,}', b'{"data": "SECRET\n"}'):
            with self.subTest(type_name=type(body).__name__):
                self.rejected(wire.parse_launch, body)

    def test_exact_body_capacity_and_overflow(self):
        base = launch()
        exact = base + b" " * (wire.MAXIMUM_BODY_BYTES - len(base))
        self.assertEqual(wire.parse_launch(exact), INSTANCE_ID)
        self.rejected(wire.parse_launch, exact + b" ")

    def test_depth_and_total_container_budgets(self):
        value = instance()
        value["deep"] = 0
        for _index in range(wire.MAXIMUM_DEPTH - 2):
            value["deep"] = [value["deep"]]
        self.assertEqual(wire.parse_instance(encode({"data": value}))["instance_id"], INSTANCE_ID)
        value["deep"] = [value["deep"]]
        self.rejected(wire.parse_instance, encode({"data": value}))
        value = instance()
        value["many"] = [{} for _index in range(wire.MAXIMUM_CONTAINERS)]
        self.rejected(wire.parse_instance, encode({"data": value}))

    def test_item_and_separator_budgets_reject_bounded_but_large_trees(self):
        value = instance()
        value["many"] = [0] * wire.MAXIMUM_ITEMS
        self.rejected(wire.parse_instance, encode({"data": value}))
        value["many"] = {f"key{index}": index for index in range(wire.MAXIMUM_ITEMS)}
        self.rejected(wire.parse_instance, encode({"data": value}))

    def test_string_brackets_and_escapes_do_not_consume_container_budget(self):
        value = instance()
        value["large_string"] = '{[,:]}"\\' * 500
        self.assertEqual(wire.parse_instance(encode({"data": value}))["instance_id"], INSTANCE_ID)

    def test_single_byte_and_prefix_fuzz_always_fail_with_fixed_code(self):
        for body in (*[bytes((number,)) for number in range(256)],
                     *[launch()[:index] for index in range(len(launch()))]):
            with self.subTest(length=len(body)):
                self.rejected(wire.parse_launch, body)

    def test_input_object_not_mutated_and_error_text_is_fixed(self):
        value = instance()
        previous = copy.deepcopy(value)
        wire.parse_instance(encode({"data": value}))
        self.assertEqual(value, previous)
        for supplied in (None, "SECRET", [], {}, True, 123):
            with self.subTest(type_name=type(supplied).__name__):
                error = wire.WireError(supplied)
                self.assertEqual(str(error), "response_rejected")
                self.assertEqual(error.args, ("response_rejected",))

    def test_source_has_no_io_or_provider_operation_imports(self):
        tree = ast.parse((ROOT / "src/lumi/posttraining_sft_lambda_wire_v1.py").read_bytes())
        imports = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                   for alias in node.names]
        self.assertEqual(imports, ["json", "math", "re"])
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else (
                    node.func.attr if isinstance(node.func, ast.Attribute) else ""
                )
                self.assertNotIn(name, {"open", "read", "write", "socket", "Popen", "system",
                                        "urlopen", "request", "post", "print"})


if __name__ == "__main__":
    unittest.main()
