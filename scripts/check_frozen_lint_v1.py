"""Check exact historical lint debt plus zero new findings, without rewriting files.

This is not a claim that historical source is lint-clean. The baseline is tied
to one reviewed source backup and one Ruff version. It cannot accept changes to
any grandfathered file; use a reviewed successor for future changes.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "configs/lumi-ci-frozen-lint-v0.1.json"
BASELINE_SHA256 = "d043f535ef79239c4048067ce2c285df72660d553ef224c3014599dc36268939"
RUFF_VERSION = "0.15.1"
SOURCE_COMMIT = "1c24ef039d25dde0172fb2751d7afdba403c7975"
REVIEWED_RULES = frozenset(
    {"E731", "F401", "E501", "I001", "UP012", "UP035", "F841", "UP042", "UP037", "B009", "B023"}
)
HASH = re.compile(r"[0-9a-f]{64}\Z")


class LintPolicyError(ValueError):
    """Fixed policy failure; source text is not included."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise LintPolicyError(code)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def valid_path(value: object) -> bool:
    if not isinstance(value, str) or "\\" in value or "\n" in value:
        return False
    path = PurePosixPath(value)
    return (
        str(path) == value
        and not path.is_absolute()
        and bool(path.parts)
        and ".." not in path.parts
        and path.parts[0] in {"src", "scripts", "tests"}
        and path.suffix == ".py"
    )


def read_source(root: Path, name: str) -> bytes:
    require(valid_path(name), "source_path_invalid")
    path = root / name
    require(not any(p.is_symlink() for p in [path, *path.parents]), "source_symlink")
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_size <= 2**20, "source_shape")
    data = path.read_bytes()
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    require(all(getattr(before, f) == getattr(after, f) for f in fields), "source_changed")
    return data


def validate_baseline(value: object) -> dict:
    require(isinstance(value, dict), "baseline_schema")
    require(
        set(value)
        == {"schema_version", "source_commit", "ruff_version", "total_findings", "files"},
        "baseline_schema",
    )
    require(value["schema_version"] == "lumi.ci-frozen-lint.v0.1", "baseline_schema")
    require(value["source_commit"] == SOURCE_COMMIT, "baseline_source_commit")
    require(value["ruff_version"] == RUFF_VERSION, "baseline_ruff_version")
    require(type(value["total_findings"]) is int and value["total_findings"] > 0, "baseline_count")
    files = value["files"]
    require(isinstance(files, dict) and bool(files), "baseline_files")
    for name, entry in files.items():
        require(valid_path(name), "baseline_path")
        require(
            isinstance(entry, dict)
            and set(entry) == {"sha256", "diagnostic_sha256", "count", "rules"},
            "baseline_entry",
        )
        for field in ("sha256", "diagnostic_sha256"):
            require(
                isinstance(entry[field], str) and bool(HASH.fullmatch(entry[field])),
                "baseline_hash",
            )
        require(type(entry["count"]) is int and entry["count"] > 0, "baseline_count")
        rules = entry["rules"]
        require(isinstance(rules, dict) and bool(rules), "baseline_rules")
        require(set(rules) <= REVIEWED_RULES, "unreviewed_rule")
        require(all(type(n) is int and n > 0 for n in rules.values()), "baseline_count")
        require(sum(rules.values()) == entry["count"], "baseline_count")
    require(sum(e["count"] for e in files.values()) == value["total_findings"], "baseline_count")
    return value


def group_findings(report: object, root: Path) -> dict[str, list[dict]]:
    require(isinstance(report, list), "ruff_report_shape")
    result: dict[str, list[dict]] = {}
    for item in report:
        require(isinstance(item, dict), "ruff_finding_shape")
        try:
            name = Path(item["filename"]).relative_to(root).as_posix()
        except (KeyError, TypeError, ValueError):
            raise LintPolicyError("ruff_path") from None
        require(valid_path(name), "ruff_path")
        require(
            isinstance(item.get("code"), str) and isinstance(item.get("message"), str),
            "ruff_finding_shape",
        )
        for field in ("location", "end_location"):
            location = item.get(field)
            require(
                isinstance(location, dict) and set(location) == {"row", "column"}, "ruff_location"
            )
            require(all(type(n) is int and n > 0 for n in location.values()), "ruff_location")
        finding = {k: item[k] for k in ("code", "location", "end_location", "message")}
        result.setdefault(name, []).append(finding)
    return result


def evaluate(baseline: dict, report: object, root: Path, reader=read_source) -> dict:
    baseline = validate_baseline(baseline)
    groups = group_findings(report, root)
    historical = baseline["files"]
    unexpected = set(groups) - set(historical)
    require(not unexpected, "new_lint_findings")
    for name, expected in historical.items():
        require(
            hashlib.sha256(reader(root, name)).hexdigest() == expected["sha256"],
            "historical_source_changed",
        )
        findings = groups.get(name, [])
        require(len(findings) == expected["count"], "historical_finding_count_changed")
        require(
            dict(collections.Counter(f["code"] for f in findings)) == expected["rules"],
            "historical_rules_changed",
        )
        require(
            digest(sorted(findings, key=canonical)) == expected["diagnostic_sha256"],
            "historical_diagnostics_changed",
        )
    return {
        "status": "PASS_exact_historical_debt_and_zero_new_findings",
        "historical_source_files": len(historical),
        "historical_findings_retained": baseline["total_findings"],
        "new_findings": 0,
        "all_source_lint_clean": False,
        "source_files_modified": 0,
    }


def lint_command(root: Path) -> list[str]:
    selected = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.py"],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=30,
    )
    paths = sorted(set(selected.stdout.decode("utf-8").split("\0")) - {""})
    require(bool(paths), "python_file_selection_empty")
    for name in paths:
        path = PurePosixPath(name)
        require(
            not path.is_absolute()
            and ".." not in path.parts
            and str(path) == name
            and "\n" not in name
            and "\\" not in name,
            "python_file_selection_invalid",
        )
    return [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--isolated",
        "--no-cache",
        "--no-respect-gitignore",
        "--select",
        "E,F,I,B,UP",
        "--line-length",
        "100",
        "--target-version",
        "py311",
        "--output-format",
        "json",
        "--",
        *paths,
    ]


def main() -> int:
    require(len(sys.argv) == 1, "arguments_not_allowed")
    version = subprocess.run(
        [sys.executable, "-m", "ruff", "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    require(version.stdout.strip() == "ruff " + RUFF_VERSION, "ruff_version")
    baseline_bytes = (ROOT / BASELINE).read_bytes()
    require(hashlib.sha256(baseline_bytes).hexdigest() == BASELINE_SHA256, "baseline_file_hash")
    baseline = json.loads(baseline_bytes)
    result = subprocess.run(
        lint_command(ROOT),
        cwd=ROOT,
        capture_output=True,
        timeout=60,
    )
    require(result.returncode in (0, 1), "ruff_execution_failed")
    require(len(result.stdout) <= 32 * 2**20, "ruff_report_too_large")
    report = json.loads(result.stdout)
    summary = evaluate(baseline, report, ROOT)
    print(json.dumps(summary, sort_keys=True))
    print(
        "::warning::815 historical lint findings retained unchanged; not an all-source lint pass."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LintPolicyError, OSError, ValueError, subprocess.SubprocessError) as error:
        code = str(error) if isinstance(error, LintPolicyError) else "lint_policy_execution_failed"
        print(json.dumps({"status": "FAIL", "code": code}, sort_keys=True))
        raise SystemExit(1) from None
