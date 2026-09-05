from __future__ import annotations

from lumi.posttraining_existing_candidate_cross_source_reaudit_v18 import _mode_from_row


def test_new_source_language_modes_are_explicit() -> None:
    assert _mode_from_row({"language": "bn"}, "language_field") == "bengali_script"
    assert (
        _mode_from_row({"language": "multilingual"}, "banglabridge_language_field")
        == "natural_code_switching"
    )
    assert (
        _mode_from_row(
            {"language": "multilingual"}, "banglatlit_auxiliary_as_banglish"
        )
        == "banglish"
    )

