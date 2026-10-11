"""
FactoryMind AI (Pharma) - Numeric Grounding Checker (§8.4)
Enforces zero ungrounded numbers.
Post-processes every draft LLM response and ensures every single number is present
within rounding tolerance in the tool results returned during the turn.
"""
from __future__ import annotations
import re
from typing import List, Set, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field, ConfigDict
from factorymind_pharma.services.tower_svc.copilot.tools import ToolResult

FALLBACK_MESSAGE = (
    "I could not verify that figure from computed data. "
    "As a safety control, unverified figures are withheld."
)


class GroundingReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    is_grounded: bool
    draft_numbers: List[float]
    ungrounded_numbers: List[float]
    allowed_numbers_count: int
    tolerance_pct: float


class NumericGroundingChecker:
    def __init__(self, tolerance_pct: float = 0.01):
        """
        tolerance_pct: Relative tolerance for rounding (default 1%, e.g., 20.2% vs 20.20%).
        """
        self.tolerance_pct = tolerance_pct

    def extract_numbers_from_text(self, text: str) -> List[float]:
        """
        Extracts candidate factual numbers from conversational LLM response.
        Filters out structural formatting like list enumeration (1., 2.), markdown headers (#),
        and ISO dates (YYYY-MM-DD).
        """
        # Remove ISO dates like 2022-12-31 or 2025-01-03 so date components aren't treated as standalone claims
        cleaned_text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', ' ', text)

        # Remove markdown numbered lists (e.g. "\n1. ", "\n2. ")
        cleaned_text = re.sub(r'(?:^|\n)\s*\d+\.\s+', '\n', cleaned_text)

        # Regex for numbers: floats, ints, currency, percentages, signed, comma-separated
        # e.g. 1,234.56, -0.42, 95.8%, 1500
        pattern = r'(?<![A-Za-z0-9_])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?![A-Za-z0-9_])'
        raw_matches = re.findall(pattern, cleaned_text)

        numbers: List[float] = []
        for m in raw_matches:
            # Strip percentage sign, commas, and currency
            clean_m = m.replace(',', '').replace('%', '').strip()
            # If empty or just a sign, continue
            if clean_m in ['', '+', '-']:
                continue
            try:
                val = float(clean_m)
                numbers.append(val)
            except ValueError:
                continue

        return numbers

    def extract_allowed_numbers_from_tool_results(self, tool_results: List[ToolResult]) -> List[float]:
        """
        Recursively extracts all numbers from tool_results.data, summary, disclaimer, and provenance.
        Expands with common transformations (percentage scaling x100, /100, rounding).
        """
        extracted: List[float] = []

        def _traverse(obj: Any):
            if isinstance(obj, (int, float)) and not isinstance(obj, bool):
                val = float(obj)
                extracted.append(val)
                extracted.append(round(val, 0))
                extracted.append(round(val, 1))
                extracted.append(round(val, 2))
                # Also expand percentages
                if 0.0 <= val <= 1.0:
                    extracted.append(val * 100.0)
                    extracted.append(round(val * 100.0, 1))
                    extracted.append(round(val * 100.0, 2))
                if 0.0 <= val <= 100.0:
                    extracted.append(val / 100.0)
            elif isinstance(obj, dict):
                for v in obj.values():
                    _traverse(v)
            elif isinstance(obj, (list, tuple, set)):
                for item in obj:
                    _traverse(item)
            elif isinstance(obj, str):
                # Search for numbers in summary text, strings, disclaimers
                found = re.findall(r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?', obj)
                for f in found:
                    try:
                        clean_f = f.replace(',', '')
                        val = float(clean_f)
                        extracted.append(val)
                        extracted.append(round(val, 0))
                        extracted.append(round(val, 1))
                        extracted.append(round(val, 2))
                    except ValueError:
                        pass

        for res in tool_results:
            _traverse(res.data)
            _traverse(res.summary)
            if res.disclaimer:
                _traverse(res.disclaimer)
            if res.provenance:
                _traverse(res.provenance.row_count)
                _traverse(res.provenance.as_of)

        return extracted

    def check(self, draft_text: str, tool_results: List[ToolResult]) -> GroundingReport:
        """
        Validates draft_text against tool_results.
        Returns a GroundingReport indicating if every number is grounded.
        """
        draft_numbers = self.extract_numbers_from_text(draft_text)
        allowed_numbers = self.extract_allowed_numbers_from_tool_results(tool_results)

        ungrounded: List[float] = []

        for d in draft_numbers:
            # Check if d matches any allowed number within tolerance
            matched = False
            for a in allowed_numbers:
                # Absolute tolerance for near-zero
                diff = abs(d - a)
                if diff < 1e-4:
                    matched = True
                    break
                # Relative tolerance
                denom = max(abs(a), 1.0)
                if (diff / denom) <= self.tolerance_pct:
                    matched = True
                    break

            if not matched:
                ungrounded.append(d)

        is_grounded = (len(ungrounded) == 0)
        return GroundingReport(
            is_grounded=is_grounded,
            draft_numbers=draft_numbers,
            ungrounded_numbers=ungrounded,
            allowed_numbers_count=len(allowed_numbers),
            tolerance_pct=self.tolerance_pct,
        )


def verify_numeric_grounding(
    draft_text: str,
    tool_results: List[ToolResult],
    tolerance_pct: float = 0.01,
) -> GroundingReport:
    """Convenience helper to verify grounding."""
    checker = NumericGroundingChecker(tolerance_pct=tolerance_pct)
    return checker.check(draft_text, tool_results)
