"""Optional critic agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings."""
        if not state.final_answer:
            return state

        prompt = (
            "You are a critic. Review the final answer for hallucinations, unsupported claims, "
            "and overly confident statements. If the answer is solid, say so. If there are issues, "
            "summarize them clearly and suggest how to improve the answer.\n\n"
            f"Final answer:\n{state.final_answer}\n\n"
            f"Research notes:\n{state.research_notes or 'N/A'}\n\n"
            f"Analysis notes:\n{state.analysis_notes or 'N/A'}"
        )

        response = LLMClient().complete(
            system_prompt="Critic assistant: fact-check and quality review the final answer.",
            user_prompt=prompt,
        )

        review = response.content.strip()
        metadata = {"contains_review": True}

        state.add_agent_result(self.name, review, metadata)
        state.add_trace_event("critic", {"review_length": len(review)})

        lower = review.lower()
        if any(token in lower for token in ["hallucination", "unsupported", "inaccurate", "incorrect"]):
            state.errors.append("Critic identified potential issues in the final answer.")
            state.final_answer = f"{state.final_answer}\n\nCritic review:\n{review}"

        return state
