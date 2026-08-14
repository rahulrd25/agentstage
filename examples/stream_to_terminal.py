"""Print a live agent event stream to the terminal.

This is the event flow the browser UI will render once the transport and SPA land
in Milestone 4 — a message_started opens a bubble, each message_delta appends a
token, a tool_call_started draws a card that tool_call_completed fills in.

Run it:

    uv run python examples/stream_to_terminal.py

No API key is required: it uses the deterministic fakes from the test suite.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

# The fakes live in tests/, which is not an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentstage.adapters import LangGraphAdapter
from agentstage.events import AgentEvent
from tests.fakes import FakeToolCallingModel, make_tool_call


@tool
def search_documents(query: str) -> str:
    """Search the document store."""
    return f"3 documents matching {query!r}: intro.md, api.md, faq.md"


def build_tool_agent() -> object:
    """An agent that calls a tool, then answers."""
    model = FakeToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[make_tool_call("search_documents", {"query": "agentstage"}, "call_1")],
            ),
            AIMessage(content="I found three relevant documents."),
        ]
    )
    from langchain.agents import create_agent

    return create_agent(model=model, tools=[search_documents])


def build_streaming_agent() -> object:
    """An agent whose model streams token by token."""
    model = GenericFakeChatModel(
        messages=iter([AIMessage(content="Streaming one token at a time.")])
    )

    async def call_model(state: MessagesState) -> dict:
        return {"messages": [await model.ainvoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    return graph.compile()


def build_failing_agent() -> object:
    """An agent whose tool raises, to show the failure path."""

    @tool
    def broken_search(query: str) -> str:
        """Always fails."""
        msg = f"upstream index unavailable for {query!r}"
        raise RuntimeError(msg)

    model = FakeToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[make_tool_call("broken_search", {"query": "x"}, "call_1")],
            ),
        ]
    )

    async def call_model(state: MessagesState) -> dict:
        return {"messages": [await model.ainvoke(state["messages"])]}

    def route(state: MessagesState) -> str:
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode([broken_search]))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")
    return graph.compile()


# Rendering ------------------------------------------------------------------

GLYPHS = {
    "run_started": "▶",
    "run_completed": "■",
    "run_failed": "✖",
    "message_started": "┌",
    "message_delta": "│",
    "message_completed": "└",
    "tool_call_started": "⚙",
    "tool_call_completed": "✔",
    "tool_call_failed": "✖",
    "interrupt_created": "⏸",
}


def render(event: AgentEvent) -> str:
    glyph = GLYPHS.get(event.type, "·")
    return f"  {glyph} {event.describe()}"


async def run_scenario(name: str, agent: object, prompt: str) -> None:
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    print(f"  prompt: {prompt!r}\n")

    adapter = LangGraphAdapter(agent)  # type: ignore[arg-type]
    transcript: list[str] = []

    async for event in adapter.stream(prompt, thread_id="demo-thread"):
        print(render(event))
        if event.type == "message_delta" and event.data:
            transcript.append(event.data["text"])

    if transcript:
        print(f"\n  reassembled from deltas: {''.join(transcript)!r}")


async def main() -> None:
    await run_scenario("1. Tool call — start, execute, complete", build_tool_agent(), "find docs")
    await run_scenario(
        "2. Token streaming — one message, many deltas", build_streaming_agent(), "hi"
    )
    await run_scenario(
        "3. Tool failure — the card resolves, the run fails", build_failing_agent(), "go"
    )

    print(f"\n{'=' * 78}")
    print("Every line above is one AgentEvent. Milestone 4 renders these in a browser")
    print("instead of a terminal; the event contract does not change.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
