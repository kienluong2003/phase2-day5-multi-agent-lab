"""Benchmark report rendering."""

from pathlib import Path

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown."""

    lines = ["# Benchmark Report", "", "| Run | Latency (s) | Cost (USD) | Quality | Notes |", "|---|---:|---:|---:|---|"]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        lines.append(f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | {item.notes} |")
    return "\n".join(lines) + "\n"


def save_markdown_report(metrics: list[BenchmarkMetrics], path: str | Path = "reports/benchmark_report.md") -> Path:
    """Save the rendered benchmark report to disk."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    report = render_markdown_report(metrics)
    path_obj.write_text(report, encoding="utf-8")
    return path_obj
