"""Analyst agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        if state.analysis_notes:
            return state

        if not state.research_notes:
            raise StudentTodoError("Research notes are missing; cannot run analyst agent.")

        prompt = (
            "You are an analyst. Review the research notes and extract the most important claims, "
            "highlight any contrasting viewpoints, and note where evidence is stronger or weaker. "
            "Keep the response concise and technical.\n\n"
            f"Research notes:\n{state.research_notes}"
        )

        response = LLMClient().complete(
            system_prompt="Analyst assistant: synthesize structured insights from research notes.",
            user_prompt=prompt,
        )

        state.analysis_notes = response.content.strip()
        state.add_agent_result(self.name, state.analysis_notes)
        state.add_trace_event("analyst", {})

        return state
