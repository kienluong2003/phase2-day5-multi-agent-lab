"""Benchmark skeleton for single-agent vs multi-agent."""

from pathlib import Path
from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.report import save_markdown_report


Runner = Callable[[str], ResearchState]


def _estimate_quality(state: ResearchState) -> float | None:
    if state.errors:
        return max(0.0, 10.0 - len(state.errors) * 2.0)
    if not state.final_answer:
        return 2.5
    return 8.0


def _build_notes(state: ResearchState) -> str:
    if state.errors:
        return "; ".join(state.errors)
    if state.final_answer:
        return "Completed successfully."
    return "Completed without final answer."


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and return benchmark metrics.

    This helper records runtime and derives simple quality metrics from state.
    """

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=None,
        quality_score=_estimate_quality(state),
        notes=_build_notes(state),
    )
    return state, metrics


def run_benchmark_and_save_report(
    run_name: str,
    query: str,
    runner: Runner,
    report_path: str | Path = "reports/benchmark_report.md",
) -> tuple[ResearchState, BenchmarkMetrics, Path]:
    """Run a benchmark, save the markdown report, and return the results."""
    state, metrics = run_benchmark(run_name, query, runner)
    output_path = save_markdown_report([metrics], report_path)
    return state, metrics, output_path
