from __future__ import annotations

from lumi.posttraining_aya_capacity_scan import (
    CapacityItem,
    _exact_digest,
    _reservoir_add,
    _rights_project,
    _sorted_reservoir,
)


def _record(record_id: str, content: str = "একটি উত্তর") -> dict:
    return {
        "record_id": record_id,
        "messages": [
            {"role": "user", "content": "একটি প্রশ্ন"},
            {"role": "assistant", "content": content},
        ],
        "source": {"license_id": "apache-2.0"},
        "provenance": {},
    }


def test_rights_projection_is_family_specific() -> None:
    wiki = _rights_project(_record("wiki"), "wiki_split_inst_translated")
    joke = _rights_project(_record("joke"), "joke_explaination_inst_translated")
    assert wiki["source"]["license_id"] == "cc-by-sa-4.0"
    assert joke["source"]["license_id"] == "mit"
    assert wiki["provenance"]["constituent_rights_projection_only"] is True


def test_exact_digest_normalizes_whitespace() -> None:
    left = _record("left", "একটি   উত্তর")
    right = _record("right", "একটি উত্তর")
    assert _exact_digest(left) == _exact_digest(right)


def test_reservoir_keeps_lowest_priorities() -> None:
    heap = []
    for priority in (7, 1, 9, 2):
        record = _record(str(priority))
        _reservoir_add(
            heap,
            CapacityItem(
                family_id="wiki_split_inst_translated",
                priority=priority,
                record=record,
                serialized_tokens=10,
                supervised_tokens=5,
            ),
            limit=2,
        )
    assert [item.priority for item in _sorted_reservoir(heap)] == [1, 2]
