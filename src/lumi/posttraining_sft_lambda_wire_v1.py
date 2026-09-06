"""Pure, bounded Lambda API response projection for an internal controller.

No request, provider action, file, credential or host operation occurs here.
Unknown Instance fields may contain secrets: they are bounded, validated, and
dropped from the projection without logging. Transient parsing is not secure
erasure. The 32-lowercase-hex identifier is this conservative local profile,
not a claim that the provider's general string schema requires that format.

A successful terminate response with status ``terminating`` is not final
termination proof. Its status remains unchanged for the caller to handle.
"""

from __future__ import annotations

import json
import math
import re

MAXIMUM_BODY_BYTES = 65536
MAXIMUM_DEPTH = 16
MAXIMUM_CONTAINERS = 128
MAXIMUM_ITEMS = 1024
MAXIMUM_LABEL_BYTES = 64
MAXIMUM_PRICE_CENTS_PER_HOUR = 1_500_000
MAXIMUM_GPU_COUNT = 8
STATUSES = frozenset({"booting", "active", "unhealthy", "terminated", "terminating", "preempted"})
TERMINATION_STATUSES = frozenset({"terminating", "terminated"})
ERROR_CODES = frozenset({"response_rejected"})
_IDENTIFIER = re.compile(r"[0-9a-f]{32}")
_LABEL = re.compile(r"[a-z0-9_-]{1,64}")


class WireError(ValueError):
    """One fixed code; supplied data and underlying exception text are discarded."""

    def __init__(self, _code=None):
        self.code = "response_rejected"
        super().__init__(self.code)


def _need(condition):
    if not condition:
        raise WireError from None


def _identifier(value):
    return type(value) is str and _IDENTIFIER.fullmatch(value) is not None


def _label(value):
    return type(value) is str and _LABEL.fullmatch(value) is not None


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        _need(key not in result)
        result[key] = value
    return result


def _reject_constant(_value):
    raise WireError from None


def _finite_float(value):
    result = float(value)
    _need(math.isfinite(result))
    return result


def _preflight(text):
    """Bound nesting/containers/separators before the JSON parser allocates a tree."""
    stack = []
    containers = separators = 0
    in_string = escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "{[":
            containers += 1
            stack.append(character)
            _need(containers <= MAXIMUM_CONTAINERS and len(stack) <= MAXIMUM_DEPTH)
        elif character in "}]":
            _need(bool(stack) and stack.pop() == ("{" if character == "}" else "["))
        elif character in ",:":
            separators += 1
            _need(separators <= MAXIMUM_ITEMS)
    _need(not in_string and not stack)


def _validate_tree(value):
    """Count keys and values as items; reject invalid Unicode even in dropped extras."""
    pending = [value]
    items = 0
    while pending:
        current = pending.pop()
        items += 1
        _need(items <= MAXIMUM_ITEMS)
        if type(current) is dict:
            items += len(current)
            _need(items <= MAXIMUM_ITEMS and all(type(key) is str for key in current))
            for key in current:
                key.encode("utf-8", errors="strict")
            pending.extend(current.values())
        elif type(current) is list:
            pending.extend(current)
        elif type(current) is str:
            current.encode("utf-8", errors="strict")
        elif type(current) is float:
            _need(math.isfinite(current))
        else:
            _need(current is None or type(current) in (bool, int))


def _decode(body):
    try:
        _need(type(body) is bytes and 0 < len(body) <= MAXIMUM_BODY_BYTES)
        text = body.decode("utf-8", errors="strict")
        _need(not text.startswith("\ufeff"))
        _preflight(text)
        value = json.loads(text, object_pairs_hook=_unique_pairs,
                           parse_constant=_reject_constant, parse_float=_finite_float)
        _validate_tree(value)
        _need(type(value) is dict and value.keys() == {"data"})
        return value["data"]
    except Exception:
        raise WireError from None


def _project(instance):
    _need(type(instance) is dict)
    identifier, status = instance.get("id"), instance.get("status")
    region, kind = instance.get("region"), instance.get("instance_type")
    _need(_identifier(identifier) and type(status) is str and status in STATUSES)
    _need(type(region) is dict and type(kind) is dict)
    region_name, type_name = region.get("name"), kind.get("name")
    _need(_label(region_name) and _label(type_name))
    price, specifications = kind.get("price_cents_per_hour"), kind.get("specs")
    _need(type(price) is int and 0 < price <= MAXIMUM_PRICE_CENTS_PER_HOUR)
    _need(type(specifications) is dict)
    gpu_count = specifications.get("gpus")
    _need(type(gpu_count) is int and 1 <= gpu_count <= MAXIMUM_GPU_COUNT)
    return {
        "instance_id": identifier, "status": status, "region": region_name,
        "instance_type": type_name, "price_cents_per_hour": price, "gpu_count": gpu_count,
    }


def parse_launch(body: bytes) -> str:
    """Require one returned identifier; no launch is executed or inferred here."""
    data = _decode(body)
    _need(type(data) is dict and data.keys() == {"instance_ids"})
    identifiers = data["instance_ids"]
    _need(type(identifiers) is list and len(identifiers) == 1 and _identifier(identifiers[0]))
    return identifiers[0]


def parse_instance(body: bytes) -> dict[str, str | int]:
    """Return only the six-field security projection, never unknown Instance data."""
    return _project(_decode(body))


def parse_termination(body: bytes, expected_instance_id: str) -> str:
    """Preserve terminating versus terminated; only the exact requested object matches."""
    _need(_identifier(expected_instance_id))
    data = _decode(body)
    _need(type(data) is dict and data.keys() == {"terminated_instances"})
    instances = data["terminated_instances"]
    _need(type(instances) is list and len(instances) == 1)
    projected = _project(instances[0])
    _need(projected["instance_id"] == expected_instance_id
          and projected["status"] in TERMINATION_STATUSES)
    return projected["status"]
