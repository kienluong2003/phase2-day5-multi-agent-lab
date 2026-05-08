"""Tracing hooks.

This file intentionally avoids binding to one provider. Students can plug in LangSmith,
Langfuse, OpenTelemetry, or simple JSON traces.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import get_settings


_TRACE_CONTEXT: dict[str, Any] = {
    "client": None,
    "run": None,
    "run_url": None,
}


@dataclass
class TraceSpan:
    name: str
    attributes: dict[str, Any]
    duration_seconds: float | None = None
    status: str = "ok"
    error: str | None = None


def _resolve_langsmith_client() -> Any | None:
    if _TRACE_CONTEXT["client"] is not None:
        return _TRACE_CONTEXT["client"]

    settings = get_settings()
    if not settings.langsmith_api_key:
        return None

    try:
        import langsmith
    except ImportError:
        return None

    client_cls = getattr(langsmith, "Client", None) or getattr(langsmith, "LangSmithClient", None)
    if client_cls is None:
        return None

    try:
        client = client_cls(api_key=settings.langsmith_api_key, project=settings.langsmith_project)
    except TypeError:
        client = client_cls(api_key=settings.langsmith_api_key)

    _TRACE_CONTEXT["client"] = client
    return client


def _resolve_langsmith_run() -> Any | None:
    if _TRACE_CONTEXT["run"] is not None:
        return _TRACE_CONTEXT["run"]

    client = _resolve_langsmith_client()
    if client is None:
        return None

    create_run = getattr(client, "create_run", None) or getattr(client, "run", None)
    if create_run is None:
        return None

    settings = get_settings()
    run_name = "multi-agent-research-lab"

    try:
        run = create_run(name=run_name, project=settings.langsmith_project)
    except TypeError:
        try:
            run = create_run(name=run_name)
        except Exception:
            return None

    _TRACE_CONTEXT["run"] = run
    _TRACE_CONTEXT["run_url"] = (
        getattr(run, "url", None)
        or getattr(run, "web_url", None)
        or getattr(run, "link", None)
    )
    return run


def _send_langsmith_span(span: TraceSpan, start_time: float) -> None:
    run = _resolve_langsmith_run()
    client = _TRACE_CONTEXT.get("client")
    if run is None or client is None:
        return

    create_span = getattr(run, "create_span", None) or getattr(client, "create_span", None)
    if create_span is None:
        return

    span_kwargs = {
        "name": span.name,
        "attributes": span.attributes,
        "status": span.status,
        "start_time": start_time,
        "end_time": start_time + (span.duration_seconds or 0.0),
    }

    run_id = getattr(run, "id", None) or getattr(run, "run_id", None)
    if run_id is not None:
        span_kwargs["run_id"] = run_id

    try:
        create_span(**span_kwargs)
    except TypeError:
        span_kwargs.pop("run_id", None)
        try:
            create_span(**span_kwargs)
        except Exception:
            pass
    except Exception:
        pass


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context used by the skeleton.

    Yields a span dictionary with duration and lets callers record attributes.
    If LangSmith is configured, this will attempt to publish spans there.
    """

    started = perf_counter()
    span_obj = TraceSpan(name=name, attributes=attributes or {})
    span = {
        "name": span_obj.name,
        "attributes": span_obj.attributes,
        "duration_seconds": None,
        "status": span_obj.status,
        "error": None,
        "run_url": _TRACE_CONTEXT.get("run_url"),
    }
    try:
        yield span
    except Exception as exc:
        span_obj.status = "error"
        span_obj.error = str(exc)
        span["status"] = "error"
        span["error"] = str(exc)
        raise
    finally:
        span_obj.duration_seconds = perf_counter() - started
        span["duration_seconds"] = span_obj.duration_seconds
        _send_langsmith_span(span_obj, started)


def record_span(name: str, attributes: dict[str, Any] | None = None, duration_seconds: float | None = None) -> dict[str, Any]:
    """Create a finished span without using a context manager."""
    return {
        "name": name,
        "attributes": attributes or {},
        "duration_seconds": duration_seconds,
        "status": "ok" if duration_seconds is not None else "unknown",
    }
