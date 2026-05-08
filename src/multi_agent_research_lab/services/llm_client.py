"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client skeleton."""

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion.

        Use the configured OpenAI API key and model settings.
        """
        settings = get_settings()

        if not settings.openai_api_key:
            raise StudentTodoError(
                "OPENAI_API_KEY is not configured. Set the environment variable or install "
                "the optional dependency `multi-agent-research-lab[llm]`."
            )

        try:
            import openai
        except ModuleNotFoundError as exc:
            raise StudentTodoError(
                "The OpenAI SDK is not installed. Install it with `pip install multi-agent-research-lab[llm]`."
            ) from exc

        openai.api_key = settings.openai_api_key

        version = getattr(openai, "__version__", "0")
        major_version = 0
        try:
            major_version = int(str(version).split(".")[0])
        except (ValueError, IndexError):
            pass

        input_tokens: int | None = None
        output_tokens: int | None = None

        try:
            if major_version >= 1 or not hasattr(openai, "ChatCompletion"):
                response = openai.responses.create(
                    model=settings.openai_model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    timeout=settings.timeout_seconds,
                )
                usage = getattr(response, "usage", None)
                content = self._extract_response_text(response)
            else:
                response = openai.ChatCompletion.create(
                    model=settings.openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    request_timeout=settings.timeout_seconds,
                )
                usage = response.get("usage", {}) or {}
                choice = response["choices"][0]
                message = choice.get("message", {})
                content = message.get("content", "").strip()

            if usage is not None:
                if isinstance(usage, dict):
                    input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
                    output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
                else:
                    input_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
                    output_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
        except Exception as exc:
            raise StudentTodoError(f"LLM request failed: {exc}") from exc

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=None,
        )

    def _extract_response_text(self, response: object) -> str:
        """Extract the assistant text from a modern OpenAI response."""
        if isinstance(response, dict):
            output_text = response.get("output_text")
            if output_text:
                return output_text.strip()
            output = response.get("output")
        else:
            output_text = getattr(response, "output_text", None)
            if output_text:
                return output_text.strip()
            output = getattr(response, "output", None)

        if not output:
            return str(response).strip()

        if isinstance(output, list) and len(output) > 0:
            first = output[0]
            if isinstance(first, dict):
                if "text" in first:
                    return str(first.get("text", "")).strip()
                content = first.get("content")
                if isinstance(content, list):
                    return "".join(
                        str(item.get("text", ""))
                        for item in content
                        if isinstance(item, dict)
                    ).strip()
            else:
                text = getattr(first, "text", None)
                if text:
                    return str(text).strip()
                content = getattr(first, "content", None)
                if isinstance(content, list):
                    return "".join(
                        str(getattr(item, "text", ""))
                        for item in content
                    ).strip()

        return str(response).strip()
