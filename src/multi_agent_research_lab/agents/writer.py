"""Writer agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        if state.final_answer:
            return state

        if not state.research_notes or not state.analysis_notes:
            raise StudentTodoError(
                "Research notes and analysis notes are required before writing the final answer."
            )

        prompt = (
            "You are a technical writer. Use the research notes, analysis notes, and source list to produce "
            "a clear, well-structured summary that is useful for technical learners. Include the main findings, "
            "tradeoffs, and any caveats. Aim for roughly 500 words.\n\n"
            f"Research notes:\n{state.research_notes}\n\n"
            f"Analysis notes:\n{state.analysis_notes}\n\n"
            "Sources:\n"
            + "\n".join(
                f"- {source.title} ({source.url or 'no URL'})" for source in state.sources
            )
        )

        response = LLMClient().complete(
            system_prompt="Writer assistant: synthesize a final answer from research and analysis.",
            user_prompt=prompt,
        )

        state.final_answer = response.content.strip()
        state.add_agent_result(self.name, state.final_answer)
        state.add_trace_event("writer", {"answer_length": len(state.final_answer)})

        return state
