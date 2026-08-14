"""Deterministic fakes for testing without a paid API.

No test in this repository may require OpenAI or any other paid provider, so the
whole suite runs against these.

``FakeToolCallingModel`` exists because no stock LangChain fake supports tool
calling: ``BaseChatModel.bind_tools`` is a bare ``raise NotImplementedError``,
and ``GenericFakeChatModel`` does not override it (verified against
langchain-core 1.5.4). Without this class, tool-lifecycle tests are impossible
without a real provider — this is risk R1 from the architecture proposal.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

__all__ = ["FakeToolCallingModel", "make_tool_call"]


def make_tool_call(name: str, args: dict[str, Any], id: str) -> dict[str, Any]:  # noqa: A002
    """Build a ``ToolCall``-shaped dict for scripting a fake response.

    ``type`` is required for LangChain to recognize the dict as a tool call.
    """
    return {"name": name, "args": args, "id": id, "type": "tool_call"}


class FakeToolCallingModel(BaseChatModel):
    """A chat model that replays a scripted sequence of responses.

    Each entry in ``responses`` is returned on successive invocations; the last
    entry repeats once exhausted, so an agent loop cannot hang waiting for a
    response that was never scripted.

    Scripting a tool call then a final answer exercises the full agent loop:

        model = FakeToolCallingModel(
            responses=[
                AIMessage(content="", tool_calls=[make_tool_call("search", {"q": "x"}, "call_1")]),
                AIMessage(content="Here is the answer."),
            ]
        )

    ``bind_tools`` records the tools and returns ``self``, so the scripted
    responses stay in control. That is the point: the fake decides what the
    "model" does, and assertions stay deterministic.
    """

    responses: list[AIMessage] = []
    bound_tools: list[Any] = []
    # Every invocation is appended here so tests can assert on what the agent sent.
    invocations: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling-model"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        """Record the bound tools and return self, keeping the script in charge."""
        self.bound_tools = list(tools)
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.responses:
            msg = "FakeToolCallingModel was constructed with no responses to replay."
            raise ValueError(msg)

        self.invocations.append(list(messages))
        index = min(len(self.invocations) - 1, len(self.responses) - 1)
        # Copy so a test mutating the result can't corrupt the script for later calls.
        message = self.responses[index].model_copy(deep=True)
        return ChatResult(generations=[ChatGeneration(message=message)])
