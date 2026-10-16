"""
tests/test_no_rigged_checks.py
==============================
Meta-test: scans every Python file under tests/ and the acceptance/ directory
for rigged-check patterns that would make a test always pass regardless of the
real data.

Patterns detected:
  - `or True`                  — short-circuits any boolean expression to True
  - `assert True`              — unconditional pass
  - record(..., True)          — acceptance-recording call with a literal True
    (matches: record(  something  ,   True  ))

The scan deliberately excludes this very file to avoid false positives on the
pattern strings embedded in the docstring/comments.

A negative test is included: inject a rigged pattern into a temporary string
and verify the detection regex matches it.
"""

import re
import os
import sys
import glob
import pytest

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(ROOT, "tests")
ACCEPTANCE_DIR = os.path.join(TESTS_DIR, "acceptance")
THIS_FILE = os.path.abspath(__file__)

# ── rigged-check detection patterns ──────────────────────────────────────────
# Each tuple: (pattern_name, compiled_regex)
RIGGED_PATTERNS = [
    # `or True` anywhere on a line (not inside a string literal — best-effort)
    ("or_true",       re.compile(r"\bor\s+True\b")),
    # bare `assert True` (with optional trailing whitespace/comment)
    ("assert_true",   re.compile(r"\bassert\s+True\b")),
    # record(anything, True) — literal True as the pass/fail argument
    ("record_literal_true", re.compile(r"\brecord\s*\([^)]*,\s*True\s*\)")),
]


def _collect_python_files() -> list:
    """Collect all .py files in tests/ and acceptance/."""
    patterns = [
        os.path.join(TESTS_DIR, "*.py"),
        os.path.join(ACCEPTANCE_DIR, "*.py"),
        os.path.join(ACCEPTANCE_DIR, "**", "*.py"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    # Exclude this very file (it contains the patterns as strings in comments)
    return [f for f in files if os.path.abspath(f) != THIS_FILE]


def _scan_file(filepath: str) -> list:
    """
    Return list of (line_number, pattern_name, line_content) for every match.
    Skips comment-only lines and lines inside triple-quoted docstrings (best-effort).
    """
    hits = []
    try:
        with open(filepath, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return hits

    in_docstring = False
    docstring_marker = None

    for lineno, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()

        # Track triple-quote docstrings (best-effort, ignores edge cases)
        for marker in ('"""', "'''"):
            count = stripped.count(marker)
            if not in_docstring and count >= 2:
                # Opened and closed on the same line — not a block docstring
                break
            if not in_docstring and count == 1:
                in_docstring = True
                docstring_marker = marker
                break
            if in_docstring and marker == docstring_marker and count >= 1:
                in_docstring = False
                docstring_marker = None
                break

        if in_docstring:
            continue

        # Skip pure comment lines
        if stripped.startswith("#"):
            continue

        # Strip inline comment before scanning
        code_part = raw_line.split("#")[0]

        for pattern_name, regex in RIGGED_PATTERNS:
            if regex.search(code_part):
                hits.append((lineno, pattern_name, raw_line.rstrip()))

    return hits


# ── main test ─────────────────────────────────────────────────────────────────

def test_no_rigged_checks_in_test_files():
    """
    Scan all test and acceptance files for rigged-check patterns.
    Fails if any match is found, reporting file + line + pattern.
    """
    files = _collect_python_files()
    assert files, "No Python test files found — check TESTS_DIR path"

    violations = []
    for filepath in files:
        hits = _scan_file(filepath)
        for lineno, pattern_name, line_content in hits:
            rel = os.path.relpath(filepath, ROOT)
            violations.append(f"  [{pattern_name}] {rel}:{lineno}  →  {line_content}")

    if violations:
        report = "\n".join(violations)
        pytest.fail(
            f"Rigged-check patterns found in {len(violations)} location(s):\n{report}\n\n"
            "Fix: replace `or True` / `assert True` / `record(..., True)` with "
            "real conditions that can genuinely fail."
        )


# ── negative test ─────────────────────────────────────────────────────────────

def test_negative__rigged_pattern_detector_catches_or_true():
    """Proves the detector catches `or True` in a synthetic code string."""
    synthetic_code = "result = some_check() or True  # bypass"
    _, or_true_regex = RIGGED_PATTERNS[0]
    assert or_true_regex.search(synthetic_code), (
        "Negative test failed: `or True` detector did not match the injected pattern — "
        "the scanner is broken"
    )


def test_negative__rigged_pattern_detector_catches_assert_true():
    """Proves the detector catches `assert True` in a synthetic code string."""
    synthetic_code = "assert True"
    _, assert_true_regex = RIGGED_PATTERNS[1]
    assert assert_true_regex.search(synthetic_code), (
        "Negative test failed: `assert True` detector did not match — scanner is broken"
    )


def test_negative__rigged_pattern_detector_catches_record_literal_true():
    """Proves the detector catches record(..., True) in a synthetic code string."""
    synthetic_code = "    record(result, True)"
    _, record_regex = RIGGED_PATTERNS[2]
    assert record_regex.search(synthetic_code), (
        "Negative test failed: `record(..., True)` detector did not match — scanner is broken"
    )
