"""LangGraph workflow implementation."""

from typing import Mapping

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def build(self) -> Mapping[str, object]:
        """Return the collection of available agents."""
        return {
            "supervisor": SupervisorAgent(),
            "researcher": ResearcherAgent(),
            "analyst": AnalystAgent(),
            "writer": WriterAgent(),
            "critic": CriticAgent(),
        }

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the workflow until completion or an unrecoverable stop condition."""
        agents = self.build()
        settings = get_settings()

        with trace_span("multi_agent_workflow", {"query": state.request.query}) as workflow_span:
            while True:
                with trace_span("supervisor", {"iteration": state.iteration}) as supervisor_span:
                    state = agents["supervisor"].run(state)
                    supervisor_span["attributes"]["route"] = state.last_route()

                route = state.last_route()
                workflow_span["attributes"]["last_route"] = route

                if route in (None, "done"):
                    break

                if state.iteration > settings.max_iterations:
                    state.errors.append("Max workflow iterations reached.")
                    break

                agent = agents.get(route)
                if not agent:
                    state.errors.append(f"Unknown route requested: {route}")
                    break

                with trace_span(route, {"route": route, "iteration": state.iteration}) as agent_span:
                    state = agent.run(state)
                    agent_span["attributes"]["output_length"] = len(str(state.agent_results[-1].content)) if state.agent_results else 0

                state.add_trace_event("agent_run", {"agent": route, "iteration": state.iteration})

        return state
