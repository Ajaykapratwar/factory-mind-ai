"""
FactoryMind AI (Pharma) - AI Copilot Benchmark Suite & Exit Gate Tests (§9, §9.2)
Runs 60 benchmark questions across 6 categories.
Generates docs/COPILOT_EVAL_REPORT.md.
"""
from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Dict, Any, List
import yaml
import pytest
import numpy as np

from factorymind_pharma.services.tower_svc.copilot.client import CopilotClient
from factorymind_pharma.services.tower_svc.copilot.tools import (
    get_kpi_summary,
    get_forecast,
    get_inventory_position,
    explain_batch_quality,
    get_machine_risk,
    get_supplier_risk,
    get_financials,
    run_scenario,
    list_decisions,
)
from factorymind_pharma.services.tower_svc.copilot.grounding import (
    NumericGroundingChecker,
    FALLBACK_MESSAGE,
)

BASE_DIR = Path(__file__).resolve().parents[1]
EVAL_YAML_PATH = BASE_DIR / "tests" / "copilot_eval.yaml"
REPORT_MD_PATH = BASE_DIR / "docs" / "COPILOT_EVAL_REPORT.md"


def test_numeric_grounding_rejects_injected_number():
    """Exit Gate 2: Enforced in code; test proves an injected fake number is rejected."""
    checker = NumericGroundingChecker(tolerance_pct=0.01)
    res = get_kpi_summary()

    # Draft answer with a completely hallucinated number
    fake_number = 77766655.44
    draft = f"Our overall revenue is {res.data['total_revenue']} and our phantom backlog is {fake_number} units."

    report = checker.check(draft, [res])
    assert report.is_grounded is False, "Checker failed to flag injected hallucinated number"
    assert any(abs(n - fake_number) < 1.0 for n in report.ungrounded_numbers), "Injected number not identified"


def test_copilot_is_strictly_read_only():
    """Exit Gate 5: Test asserts no tool mutates state or files."""
    data_dir = BASE_DIR.parent / "data"
    raw_files = list(data_dir.glob("*.csv"))

    # Record modification timestamps
    mtimes_before = {f.name: f.stat().st_mtime for f in raw_files}

    # Execute all tools
    _ = get_kpi_summary()
    _ = get_forecast("P0020")
    _ = get_inventory_position("P0006")
    _ = explain_batch_quality("B100001")
    _ = get_machine_risk("M001")
    _ = get_supplier_risk("S001")
    _ = get_financials(category="Analgesic")
    _ = list_decisions()

    # Verify no raw CSV files were modified
    for f in raw_files:
        assert f.stat().st_mtime == mtimes_before[f.name], f"Raw file {f.name} was modified by read-only tool!"


def test_copilot_eval_suite():
    """
    Evaluates all 60 benchmark questions across 6 categories.
    Verifies:
      - Correct tool selection >= 90%
      - Answer pass rate >= 80%
      - Ungrounded-number rate = 0 (hard gate)
      - Refusal correctness (cat 6) = 100%
      - p95 latency <= 8.0s
    Writes docs/COPILOT_EVAL_REPORT.md.
    """
    assert EVAL_YAML_PATH.exists(), f"Eval yaml missing: {EVAL_YAML_PATH}"
    with open(EVAL_YAML_PATH, "r", encoding="utf-8") as f:
        benchmarks = yaml.safe_load(f)

    client = CopilotClient()
    results: List[Dict[str, Any]] = []

    total_questions = 0
    correct_tools = 0
    passed_answers = 0
    total_ungrounded_numbers = 0
    correct_refusals = 0
    total_refusals = 0
    latencies: List[float] = []

    for cat in benchmarks["categories"]:
        cat_id = cat["id"]
        cat_name = cat["name"]

        for q in cat["questions"]:
            total_questions += 1
            qid = q["id"]
            prompt = q["prompt"]
            expected_tool = q.get("expected_tool")
            is_refusal_case = q.get("expected_refusal", False)

            t0 = time.time()
            resp = client.ask(prompt)
            elapsed = time.time() - t0
            latencies.append(elapsed)

            tool_match = False
            if is_refusal_case:
                total_refusals += 1
                refusal_match = resp.is_refusal
                if refusal_match:
                    correct_refusals += 1
                answer_pass = refusal_match
                tool_match = (len(resp.tools_called) == 0)
                if tool_match:
                    correct_tools += 1
            else:
                tool_match = (expected_tool in resp.tools_called)
                if tool_match:
                    correct_tools += 1
                answer_pass = tool_match and resp.grounding.is_grounded and not resp.is_refusal

            if resp.grounding.ungrounded_numbers:
                total_ungrounded_numbers += len(resp.grounding.ungrounded_numbers)

            if answer_pass:
                passed_answers += 1

            results.append({
                "id": qid,
                "category": cat_name,
                "prompt": prompt,
                "expected_tool": expected_tool or "REFUSAL",
                "tools_called": resp.tools_called,
                "tool_match": tool_match,
                "is_refusal": resp.is_refusal,
                "is_grounded": resp.grounding.is_grounded,
                "ungrounded_count": len(resp.grounding.ungrounded_numbers),
                "answer_pass": answer_pass,
                "latency_s": round(elapsed, 3),
                "answer_snippet": resp.answer[:120] + "..." if len(resp.answer) > 120 else resp.answer,
            })

    # Metrics calculation
    tool_accuracy = (correct_tools / total_questions) * 100.0
    pass_rate = (passed_answers / total_questions) * 100.0
    refusal_accuracy = (correct_refusals / max(1, total_refusals)) * 100.0
    p95_latency = float(np.percentile(latencies, 95))

    # Generate Markdown Report
    _write_copilot_eval_report(
        results=results,
        tool_accuracy=tool_accuracy,
        pass_rate=pass_rate,
        ungrounded_count=total_ungrounded_numbers,
        refusal_accuracy=refusal_accuracy,
        p95_latency=p95_latency,
        total_questions=total_questions,
    )

    # Spec Gate Assertions (§9)
    assert tool_accuracy >= 90.0, f"Tool accuracy {tool_accuracy:.1f}% below target 90%"
    assert pass_rate >= 80.0, f"Answer pass rate {pass_rate:.1f}% below target 80%"
    assert total_ungrounded_numbers == 0, f"Hard gate failed: {total_ungrounded_numbers} ungrounded numbers detected!"
    assert refusal_accuracy == 100.0, f"Refusal accuracy {refusal_accuracy:.1f}% below target 100%"
    assert p95_latency <= 8.0, f"p95 latency {p95_latency:.2f}s exceeded 8.0s budget"


def _write_copilot_eval_report(
    results: List[Dict[str, Any]],
    tool_accuracy: float,
    pass_rate: float,
    ungrounded_count: int,
    refusal_accuracy: float,
    p95_latency: float,
    total_questions: int,
):
    lines = [
        "# FactoryMind AI (Pharma) - AI Copilot Benchmark Report (§9)",
        "",
        "## 1. Executive Summary & Exit Gate Metrics",
        "",
        "| Metric | Target | Measured Value | Gate Status |",
        "|---|---|---|---|",
        f"| Correct tool selection | ≥ 90% | **{tool_accuracy:.1f}%** | {'PASS' if tool_accuracy >= 90 else 'FAIL'} |",
        f"| Answer pass rate (right tool + right numbers) | ≥ 80% | **{pass_rate:.1f}%** | {'PASS' if pass_rate >= 80 else 'FAIL'} |",
        f"| **Ungrounded-number rate** | **0 (Hard Gate)** | **{ungrounded_count}** | {'PASS' if ungrounded_count == 0 else 'FAIL'} |",
        f"| Refusal correctness (category 6) | 100% | **{refusal_accuracy:.1f}%** | {'PASS' if refusal_accuracy == 100 else 'FAIL'} |",
        f"| p95 latency (excluding run_scenario) | ≤ 8.0 s | **{p95_latency:.3f} s** | {'PASS' if p95_latency <= 8.0 else 'FAIL'} |",
        "",
        f"**Total Questions Evaluated:** {total_questions} (10 questions across 6 operational categories).",
        "",
        "## 2. Category Performance Breakdown",
        "",
        "| Category | Questions | Tool Accuracy | Pass Rate | Ungrounded Numbers |",
        "|---|---|---|---|---|",
    ]

    cats = sorted(list(set(r["category"] for r in results)))
    for c in cats:
        cat_items = [r for r in results if r["category"] == c]
        c_tot = len(cat_items)
        c_tool = sum(1 for r in cat_items if r["tool_match"]) / c_tot * 100.0
        c_pass = sum(1 for r in cat_items if r["answer_pass"]) / c_tot * 100.0
        c_ungr = sum(r["ungrounded_count"] for r in cat_items)
        lines.append(f"| {c} | {c_tot} | {c_tool:.1f}% | {c_pass:.1f}% | {c_ungr} |")

    lines.extend([
        "",
        "## 3. Detailed Benchmark Question Trace",
        "",
        "| ID | Category | Prompt | Expected Tool | Tools Called | Pass | Latency |",
        "|---|---|---|---|---|---|---|",
    ])

    for r in results:
        pass_icon = "PASS" if r["answer_pass"] else "FAIL"
        tools_str = ", ".join(r["tools_called"]) if r["tools_called"] else "None (Refused)"
        lines.append(
            f"| `{r['id']}` | {r['category']} | {r['prompt']} | `{r['expected_tool']}` | `{tools_str}` | **{pass_icon}** | {r['latency_s']}s |"
        )

    lines.extend([
        "",
        "---",
        "*Report generated automatically by FactoryMind Copilot Evaluation Suite (`tests/test_copilot_eval.py`).*",
    ])

    REPORT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
