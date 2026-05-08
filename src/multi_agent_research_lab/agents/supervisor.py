"""Supervisor / router implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route."""
        settings = get_settings()

        if state.iteration >= settings.max_iterations:
            state.errors.append("Maximum workflow iterations reached.")
            state.record_route("done")
            state.add_trace_event("supervisor", {"next_route": "done"})
            return state

        if not state.research_notes:
            next_route = "researcher"
        elif not state.analysis_notes:
            next_route = "analyst"
        elif not state.final_answer:
            next_route = "writer"
        elif "critic" not in state.route_history:
            next_route = "critic"
        else:
            next_route = "done"

        state.record_route(next_route)
        state.add_trace_event("supervisor", {"next_route": next_route})
        return state
