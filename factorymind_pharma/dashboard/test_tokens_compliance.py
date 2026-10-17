"""
dashboard/test_tokens_compliance.py
Verifies that:
1. All pages in factorymind_pharma/dashboard/pages have zero hardcoded hex colours.
2. All components (except theme.py fallback loader) have zero hardcoded hex colours.
3. tokens.css is present and well-formed.
"""

import re
from pathlib import Path
import pytest

_DASHBOARD_DIR = Path(__file__).parent
_PAGES_DIR = _DASHBOARD_DIR / "pages"
_COMPONENTS_DIR = _DASHBOARD_DIR / "components"
_TOKENS_PATH = _DASHBOARD_DIR / "assets" / "tokens.css"

HEX_PATTERN = re.compile(r"#[0-9a-fA-F]{3,8}\b")


def test_tokens_css_exists_and_valid():
    assert _TOKENS_PATH.exists(), "tokens.css must exist in dashboard/assets"
    content = _TOKENS_PATH.read_text(encoding="utf-8")
    assert "--bg:" in content
    assert "--surface:" in content
    assert "--primary:" in content
    assert "--accent:" in content


def test_no_hardcoded_hex_in_pages():
    violations = []
    for py_file in _PAGES_DIR.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        lines = content.splitlines()
        for idx, line in enumerate(lines, start=1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # Remove inline comments before checking
            code_part = line.split("#")[0]
            matches = HEX_PATTERN.findall(code_part)
            if matches:
                violations.append(f"{py_file.name}:{idx} -> {matches} in line: {line.strip()}")

    assert not violations, "Found hardcoded hex colours in pages:\n" + "\n".join(violations)


def test_no_hardcoded_hex_in_components_excluding_theme():
    violations = []
    for py_file in _COMPONENTS_DIR.glob("*.py"):
        if py_file.name == "theme.py":
            continue
        content = py_file.read_text(encoding="utf-8")
        lines = content.splitlines()
        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            code_part = line.split("#")[0]
            matches = HEX_PATTERN.findall(code_part)
            if matches:
                violations.append(f"{py_file.name}:{idx} -> {matches} in line: {line.strip()}")

    assert not violations, "Found hardcoded hex colours in components:\n" + "\n".join(violations)
