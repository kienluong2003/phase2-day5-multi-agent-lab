"""Researcher agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        if state.research_notes:
            return state

        prompt = (
            "You are an expert researcher. For the following research query, produce a concise "
            "summary of the most important findings and three credible-sounding source titles. "
            "Do not invent claims that cannot be supported by general knowledge.\n\n"
            f"Research query:\n{state.request.query}\n\n"
            "Format the response as plain text. Start with a short research summary and then "
            "provide a list of three source titles with optional URLs."
        )

        response = LLMClient().complete(
            system_prompt="Research assistant: gather information and summarize the key concepts.",
            user_prompt=prompt,
        )

        state.research_notes = response.content.strip()
        state.sources = self._default_sources(state.request.query)
        state.add_agent_result(self.name, state.research_notes, {"source_count": len(state.sources)})
        state.add_trace_event("researcher", {"source_count": len(state.sources)})

        return state

    def _default_sources(self, query: str) -> list[SourceDocument]:
        slug = query.lower().replace(" ", "-").replace("/", "-")[:50]
        return [
            SourceDocument(
                title=f"Introduction to {query}",
                url=f"https://example.com/{slug}-overview",
                snippet=f"A high-level introduction to the core concepts behind {query}.",
            ),
            SourceDocument(
                title=f"State of the art in {query}",
                url=f"https://example.com/{slug}-state-of-the-art",
                snippet=f"A review of current techniques, trends, and challenges in {query}.",
            ),
            SourceDocument(
                title=f"Practical applications of {query}",
                url=f"https://example.com/{slug}-applications",
                snippet=f"Examples of how {query} is applied in real-world systems.",
            ),
        ]
