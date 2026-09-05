from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path

import pytest

import lumi.posttraining_bedstc2_upstream_rights_review_v1 as rights

ROOT = Path(__file__).resolve().parents[1]


def _assessment() -> dict[str, object]:
    return {
        "document_observations": rights.EXPECTED_DOCUMENT_OBSERVATIONS,
        "rights_criteria": rights.EXPECTED_RIGHTS_CRITERIA,
        "criteria_satisfied": 0,
        "criteria_unresolved": 5,
    }


def _report() -> dict[str, object]:
    return rights.assemble_report("a" * 64, _assessment())


def _reseal_report(report: dict[str, object]) -> None:
    report["report_sha256"] = rights.sha256_value(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )


def _reseal_verification(verification: dict[str, object]) -> None:
    verification["verification_sha256"] = rights.sha256_value(
        {
            key: value
            for key, value in verification.items()
            if key != "verification_sha256"
        }
    )


def test_strict_json_rejects_duplicate_nonfinite_and_surrogate() -> None:
    with pytest.raises(ValueError):
        rights.strict_json_bytes(b'{"a":1,"a":2}')
    with pytest.raises(ValueError):
        rights.strict_json_bytes(b'{"a":NaN}')
    with pytest.raises(ValueError):
        rights.strict_json_bytes(b'{"a":"\\ud800"}')


def test_synthetic_rights_document_markers_classify_only_booleans() -> None:
    readme = "\n".join(rights.README_MARKERS)
    license_text = "\n".join(rights.LICENSE_MARKERS)
    assert rights.inspect_rights_documents(readme, license_text) == (
        rights.EXPECTED_DOCUMENT_OBSERVATIONS
    )


def test_synthetic_rights_document_missing_marker_fails_closed() -> None:
    readme = "\n".join(rights.README_MARKERS[:-1])
    license_text = "\n".join(rights.LICENSE_MARKERS)
    with pytest.raises(ValueError):
        rights.inspect_rights_documents(readme, license_text)


def test_bound_reader_rejects_hash_size_symlink_and_hardlink(tmp_path: Path) -> None:
    payload = b"synthetic rights document"
    source = tmp_path / "source"
    source.write_bytes(payload)
    assert rights.read_bound_bytes(
        source,
        expected_bytes=len(payload),
        expected_sha256=rights.sha256_bytes(payload),
    ) == payload
    with pytest.raises(ValueError):
        rights.read_bound_bytes(
            source,
            expected_bytes=len(payload) + 1,
            expected_sha256=rights.sha256_bytes(payload),
        )
    symlink = tmp_path / "symlink"
    symlink.symlink_to(source)
    with pytest.raises(OSError):
        rights.read_bound_bytes(
            symlink,
            expected_bytes=len(payload),
            expected_sha256=rights.sha256_bytes(payload),
        )
    hardlink = tmp_path / "hardlink"
    os.link(source, hardlink)
    with pytest.raises(ValueError):
        rights.read_bound_bytes(
            source,
            expected_bytes=len(payload),
            expected_sha256=rights.sha256_bytes(payload),
        )


def test_production_path_resolver_rejects_leaf_and_parent_symlinks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    evidence = root / "evidence"
    evidence.mkdir(parents=True)
    real = evidence / "real.json"
    real.write_bytes(b"{}\n")
    leaf = evidence / "lumi-4b-sft-bedstc2-upstream-rights-review-v0.1.json"
    leaf.symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        rights.resolve_repo_path(root, leaf)

    leaf.unlink()
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = root / "artifacts"
    linked_parent.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        rights.resolve_repo_path(
            root,
            "artifacts/lumi-sft-bedstc2-banglish-conversation-v0.1/"
            "attempt-001/private/raw/README.md",
        )


def test_production_path_resolver_is_lexical_and_allows_only_fresh_leaf(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    (root / "evidence").mkdir(parents=True)
    fresh = rights.resolve_repo_path(
        root, "evidence/lumi-4b-sft-bedstc2-upstream-rights-review-v0.1.json"
    )
    assert fresh == (
        root / "evidence/lumi-4b-sft-bedstc2-upstream-rights-review-v0.1.json"
    )
    assert not fresh.exists()
    with pytest.raises(ValueError, match="escaped"):
        rights.resolve_repo_path(root, "../outside.json")
    with pytest.raises(ValueError, match="escaped"):
        rights.resolve_repo_path(root, tmp_path / "outside.json")


def test_plan_symlink_is_rejected_before_json_payload_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    configs = root / "configs"
    configs.mkdir(parents=True)
    real = configs / "real-plan.json"
    real.write_text("not-json", encoding="utf-8")
    expected = configs / "rights-plan.json"
    expected.symlink_to(real)
    monkeypatch.setattr(rights, "EXPECTED_PLAN_PATH", Path("configs/rights-plan.json"))
    with pytest.raises(ValueError, match="symlink"):
        rights.load_plan(expected, root)


def test_internal_seal_uses_unsigned_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    unsigned = {"schema_version": "synthetic", "value": 1}
    internal = rights.sha256_value(unsigned)
    sealed = {**unsigned, "receipt_sha256": internal}
    monkeypatch.setitem(
        rights.EXPECTED_INPUTS,
        "acquisition_receipt",
        {
            **rights.EXPECTED_INPUTS["acquisition_receipt"],
            "internal_sha256": internal,
        },
    )
    rights.validate_internal_seal("acquisition_receipt", sealed)
    sealed["value"] = 2
    with pytest.raises(ValueError):
        rights.validate_internal_seal("acquisition_receipt", sealed)


def test_report_and_verification_validate() -> None:
    report = _report()
    rights.validate_report(report)
    verification = rights.build_verification(
        report,
        report_file_sha256="b" * 64,
        report_file_bytes=len(rights.json_bytes(report)),
        plan_file_sha256="a" * 64,
    )
    rights.validate_verification(verification)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("production_eligible",), True),
        (("rights", "upstream_dstc2_rights_chain_independently_verified"), True),
        (("rights_assessment", "criteria_unresolved"), 4),
        (("remaining_blockers",), []),
        (("reason_codes",), []),
        (("document_text_retained",), True),
        (("next_gate", "separate_authorization_required"), False),
    ],
)
def test_resealed_report_tampering_is_rejected(
    path: tuple[str, ...], value: object
) -> None:
    report = copy.deepcopy(_report())
    cursor: object = report
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = value  # type: ignore[index]
    _reseal_report(report)
    with pytest.raises(ValueError):
        rights.validate_report(report)


def test_report_rejects_raw_text_key_even_when_resealed() -> None:
    report = _report()
    report["snippet"] = "synthetic"
    _reseal_report(report)
    with pytest.raises(ValueError):
        rights.validate_report(report)


def test_resealed_verification_tampering_is_rejected() -> None:
    report = _report()
    verification = rights.build_verification(
        report,
        report_file_sha256="b" * 64,
        report_file_bytes=len(rights.json_bytes(report)),
        plan_file_sha256="a" * 64,
    )
    verification["independent_replay"]["candidate_row_reads"] = 1  # type: ignore[index]
    _reseal_verification(verification)
    with pytest.raises(ValueError):
        rights.validate_verification(verification)


def test_report_is_stop_raw_text_free_and_preserves_every_blocker() -> None:
    report = _report()
    serialized = rights.json_bytes(report).decode("utf-8")
    assert report["status"] == rights.REPORT_STATUS
    assert report["production_eligible"] is False
    assert report["remaining_blockers"] == rights.EXPECTED_REMAINING_BLOCKERS
    assert report["evaluation_prohibition"] == rights.EXPECTED_EVALUATION_PROHIBITION
    assert report["provenance"] == rights.EXPECTED_PROVENANCE
    assert all(marker not in serialized for marker in rights.README_MARKERS)
    assert all(marker not in serialized for marker in rights.LICENSE_MARKERS)


def test_marker_exclusive_and_finalized_in_place(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    identity = rights.start_marker(path, rights.EXPECTED_REPORT_MARKER)
    with pytest.raises(FileExistsError):
        rights.start_marker(path, rights.EXPECTED_REPORT_MARKER)
    payload = rights.json_bytes(_report())
    rights.finish_marker(path, rights.EXPECTED_REPORT_MARKER, payload, identity)
    assert path.read_bytes() == payload


def test_finish_marker_rejects_wrong_marker_and_inode_swap(tmp_path: Path) -> None:
    wrong = tmp_path / "wrong.json"
    wrong.write_bytes(b"wrong")
    with pytest.raises(ValueError):
        rights.finish_marker(
            wrong,
            rights.EXPECTED_REPORT_MARKER,
            b"payload",
            (wrong.stat().st_dev, wrong.stat().st_ino),
        )
    path = tmp_path / "report.json"
    identity = rights.start_marker(path, rights.EXPECTED_REPORT_MARKER)
    path.rename(tmp_path / "original.json")
    path.write_bytes(rights.EXPECTED_REPORT_MARKER)
    with pytest.raises(ValueError):
        rights.finish_marker(
            path, rights.EXPECTED_REPORT_MARKER, b"payload", identity
        )


def test_plan_has_exact_seven_input_scope_and_byte_total() -> None:
    plan = json.loads((ROOT / rights.EXPECTED_PLAN_PATH).read_text(encoding="utf-8"))
    assert set(plan["inputs"]) == {
        "acquisition_receipt",
        "repository_readme",
        "repository_license",
        "curation_report",
        "curation_verification",
        "owner_receipt",
        "owner_verification",
    }
    assert sum(item["bytes"] for item in plan["inputs"].values()) == 54_784
    assert plan["output"] == rights.EXPECTED_OUTPUT
    serialized = json.dumps(plan, sort_keys=True)
    for forbidden in (
        "dialog-dstc2-trn",
        "dialog-dstc2-dev",
        "candidate.jsonl",
        "benchmark-signatures",
    ):
        assert forbidden not in serialized


def test_live_plan_load_does_not_read_bound_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched: list[str] = []

    def metadata_only(_root: Path, name: str) -> None:
        touched.append(name)

    def forbidden(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("bound payload read before marker")

    monkeypatch.setattr(rights, "_assert_input_metadata", metadata_only)
    monkeypatch.setattr(rights, "read_bound_bytes", forbidden)
    plan = rights.load_plan(ROOT / rights.EXPECTED_PLAN_PATH, ROOT)
    assert set(touched) == set(rights.EXPECTED_INPUTS)
    assert plan["limits"]["source_line_reads"] == 0


def test_entrypoints_start_markers_before_build_and_verifier_is_independent() -> None:
    runner = (ROOT / rights.EXPECTED_CODE_PATHS["runner"]).read_text(encoding="utf-8")
    verifier = (ROOT / rights.EXPECTED_CODE_PATHS["verifier"]).read_text(
        encoding="utf-8"
    )
    ast.parse(runner)
    ast.parse(verifier)
    assert runner.count("build_report(") == 1
    assert runner.index("start_marker(") < runner.index("build_report(")
    assert "contract.build_report" not in verifier
    assert "contract.assemble_report" not in verifier
    assert '(root / str(binding["path"])).resolve()' not in verifier
    assert verifier.count("contract.resolve_repo_path(") >= 3
    assert verifier.index("contract.start_marker(") < verifier.rindex(
        "_independent_expected_report("
    )


def test_authorization_limits_and_marker_hashes_are_frozen() -> None:
    assert rights.sha256_bytes(rights.EXPECTED_AUTHORIZATION_TEXT.encode()) == (
        rights.EXPECTED_AUTHORIZATION_SHA256
    )
    assert rights.EXPECTED_LIMITS["passes"] == 2
    assert rights.EXPECTED_LIMITS["bound_input_bytes_per_pass"] == 54_784
    assert rights.EXPECTED_LIMITS["bound_input_file_reads_total"] == 14
    assert rights.EXPECTED_LIMITS["runtime_seconds_total"] == 300
    assert rights.EXPECTED_LIMITS["private_write_bytes"] == 1_048_576
    assert rights.EXPECTED_LIMITS["network_bytes"] == 0
    assert rights.EXPECTED_LIMITS["gpu_use"] == 0
    assert rights.EXPECTED_LIMITS["inference_calls"] == 0
    assert rights.sha256_bytes(rights.EXPECTED_REPORT_MARKER) == (
        "0ad9a8e0e252809a36558efce8c4ea1d8471b09d54fc8bc3d100610e76a162ec"
    )
    assert rights.sha256_bytes(rights.EXPECTED_VERIFICATION_MARKER) == (
        "674b20c1f60388817b2e0f3c87d22cd0fca5dfbedee3885d4e8078f806040e43"
    )


def test_next_gate_requires_separate_authorization() -> None:
    assert rights.EXPECTED_NEXT_GATE == {
        "action": (
            "separately_authorized_authoritative_upstream_dstc2_rights_document_"
            "acquisition_and_legal_review"
        ),
        "authoritative_document_acquisition_authorized": False,
        "legal_review_authorized": False,
        "network_authorized": False,
        "separate_authorization_required": True,
    }
