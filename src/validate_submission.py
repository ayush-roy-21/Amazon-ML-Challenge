"""Validate the two competition-scored output files against the rules in the problem statement.

We were never given the organisers' own ``utils/validate_submission.py`` (the problem statement only
describes what it checks and how it is invoked), so this is our own re-implementation of exactly those
rules, kept dependency-free (standard library only) so it can be run even in an environment that has none
of ``src/``'s own dependencies installed. If the official script is available, prefer it - this exists so
the checks in the spec can still be run without it.

    python3 validate_submission.py --matching output/matching_results.tsv \\
        --candidate output/candidate_pairs.tsv --test-dir dataset/test

Prints "PASS" and exits 0 if every check passes; otherwise prints one numbered line per issue found and
exits 1. Every check in the problem statement's "Submission Format & Constraints" section is covered:
every test Source-1 id appears exactly once in each file, ids referenced are real Source-2/3 test ids (no
self-matches to Source-1, nothing invented), no id repeated within one row's list, no duplicate
source1_entity_id rows, and every id in matching_results.tsv's list is also in that same row's
candidate_pairs.tsv list.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple


def _read_two_col_tsv(path: str) -> Tuple[List[str], Dict[str, List[str]], List[str]]:
    """Returns (ordered ids as they appear, id -> list-of-ids parsed from column 2, issues found while
    parsing). Parsing is intentionally lenient (never raises) so a malformed file is reported as issues
    rather than a crash."""
    issues: List[str] = []
    order: List[str] = []
    parsed: Dict[str, List[str]] = {}
    if not os.path.exists(path):
        return order, parsed, [f"file not found: {path}"]
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        lines = [ln.rstrip("\r\n") for ln in fh]
    lines = [ln for ln in lines if ln.strip() != ""]
    if not lines:
        return order, parsed, [f"{path}: file is empty (expected a header row plus one row per entity)"]
    start = 1 if lines[0].split("\t")[0].strip().lower().startswith("source1_entity_id") else 0
    for lineno, ln in enumerate(lines[start:], start=start + 1):
        cols = ln.split("\t")
        if len(cols) != 2:
            issues.append(f"{path}:{lineno}: expected exactly 2 tab-separated columns, found {len(cols)}")
            continue
        sid, rest = cols[0].strip(), cols[1].strip()
        if not sid:
            issues.append(f"{path}:{lineno}: empty source1_entity_id")
            continue
        ids = [x.strip() for x in rest.split(",")] if rest else []
        ids = [x for x in ids if x != ""]
        if sid in parsed:
            issues.append(f"{path}:{lineno}: duplicate source1_entity_id row '{sid}' "
                           f"(first seen at an earlier row)")
            continue
        order.append(sid)
        parsed[sid] = ids
    return order, parsed, issues


def _check_id_list_rules(path: str, sid: str, ids: List[str], valid_ids: set, issues: List[str]) -> None:
    if len(ids) != len(set(ids)):
        seen, dups = set(), set()
        for x in ids:
            (dups if x in seen else seen).add(x)
        issues.append(f"{path}: {sid}: duplicate id(s) within the row: {sorted(dups)}")
    for x in ids:
        if x.startswith("S1-"):
            issues.append(f"{path}: {sid}: '{x}' is a Source-1 id - self-matches to Source-1 are not allowed")
        elif not (x.startswith("S2-") or x.startswith("S3-")):
            issues.append(f"{path}: {sid}: '{x}' is not a Source-2/3 id")
        elif x not in valid_ids:
            issues.append(f"{path}: {sid}: '{x}' does not exist in the test set")


def load_test_ids(test_dir: str) -> Tuple[set, set]:
    """(source1 ids, source2+3 ids) straight from the raw TSV text - no pandas, so this has no dependency
    on the rest of the pipeline being importable."""

    def _ids(path: str) -> List[str]:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as fh:
            lines = [ln.rstrip("\r\n") for ln in fh if ln.strip()]
        if not lines:
            return []
        start = 1 if lines[0].split("\t")[0].strip().lower() == "entity_id" else 0
        return [ln.split("\t")[0].strip() for ln in lines[start:] if ln.split("\t")[0].strip()]

    s1 = set(_ids(os.path.join(test_dir, "test_source1.tsv")))
    s23 = set(_ids(os.path.join(test_dir, "test_source2.tsv"))) | set(_ids(os.path.join(test_dir, "test_source3.tsv")))
    return s1, s23


def validate(matching_path: str, candidate_path: str, test_dir: str) -> List[str]:
    issues: List[str] = []
    s1_ids, s23_ids = load_test_ids(test_dir)
    if not s1_ids:
        issues.append(f"no Source-1 ids found under {test_dir} (looked for test_source1.tsv) - "
                       f"cannot validate coverage without it")

    m_order, m_rows, m_issues = _read_two_col_tsv(matching_path)
    c_order, c_rows, c_issues = _read_two_col_tsv(candidate_path)
    issues += m_issues + c_issues

    for path, rows in ((matching_path, m_rows), (candidate_path, c_rows)):
        missing = sorted(s1_ids - set(rows))
        extra = sorted(set(rows) - s1_ids)
        if missing:
            issues.append(f"{path}: {len(missing)} test Source-1 id(s) missing a row, "
                           f"e.g. {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        if extra:
            issues.append(f"{path}: {len(extra)} row(s) whose source1_entity_id is not in the test set, "
                           f"e.g. {extra[:5]}{' ...' if len(extra) > 5 else ''}")
        for sid, ids in rows.items():
            _check_id_list_rules(path, sid, ids, s23_ids, issues)

    for sid, matched in m_rows.items():
        cands = set(c_rows.get(sid, []))
        extra = [x for x in matched if x not in cands]
        if extra and sid in c_rows:
            issues.append(f"matched-but-not-candidate for {sid}: {extra} appear in "
                           f"{os.path.basename(matching_path)} but not in that row's "
                           f"{os.path.basename(candidate_path)} list")
    return issues


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matching", required=True, help="path to matching_results.tsv")
    ap.add_argument("--candidate", required=True, help="path to candidate_pairs.tsv")
    ap.add_argument("--test-dir", required=True, help="directory with test_source1/2/3.tsv")
    args = ap.parse_args()

    issues = validate(args.matching, args.candidate, args.test_dir)
    if not issues:
        print("PASS")
        return 0
    for i, msg in enumerate(issues, 1):
        print(f"{i}. {msg}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
