"""Verify the R1 mitigation: tool-calling tests without a paid API.

These drive a real `create_agent` graph, not a mock of one. If the fake could not
carry a real agent loop through a tool call, every later milestone's
tool-lifecycle test would be blocked.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from tests.fakes import FakeToolCallingModel, make_tool_call


@tool
def search_documents(query: str) -> str:
    """Search the document store."""
    return f"RESULT for {query}"


def test_stock_fake_still_cannot_bind_tools():
    """Pins risk R1. If langchain-core ever implements this, we can reconsider
    shipping our own fake — until then this is why the fake exists."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    model = GenericFakeChatModel(messages=iter([AIMessage(content="hi")]))
    with pytest.raises(NotImplementedError):
        model.bind_tools([search_documents])


def test_bind_tools_records_tools_and_returns_the_model():
    model = FakeToolCallingModel(responses=[AIMessage(content="done")])

    bound = model.bind_tools([search_documents])

    assert bound is model
    assert model.bound_tools == [search_documents]


def test_responses_replay_in_order_then_the_last_one_repeats():
    """The last response repeating is what stops an agent loop from hanging on an
    under-scripted fake."""
    model = FakeToolCallingModel(
        responses=[AIMessage(content="first"), AIMessage(content="second")]
    )

    assert model.invoke([HumanMessage(content="a")]).content == "first"
    assert model.invoke([HumanMessage(content="b")]).content == "second"
    assert model.invoke([HumanMessage(content="c")]).content == "second"


def test_empty_response_script_fails_loudly():
    """Requirement 13: no silent swallowing. A misconfigured fake must say so."""
    model = FakeToolCallingModel(responses=[])

    with pytest.raises(ValueError, match="no responses"):
        model.invoke([HumanMessage(content="a")])


def test_invocations_are_recorded_for_assertions():
    model = FakeToolCallingModel(responses=[AIMessage(content="ok")])

    model.invoke([HumanMessage(content="what is up")])

    assert len(model.invocations) == 1
    assert model.invocations[0][0].content == "what is up"


def test_the_fake_is_a_real_chat_model():
    """It must satisfy LangChain's own contract, or `create_agent` will reject it."""
    assert isinstance(FakeToolCallingModel(responses=[AIMessage(content="x")]), BaseChatModel)


async def test_drives_a_real_agent_through_a_full_tool_call():
    """The load-bearing test: a real compiled agent, a real tool, no API key.

    Asserts the correlation the adapter will depend on in Milestone 3 — the
    ToolMessage's `tool_call_id` matches the id on the AIMessage's tool call.
    """
    from langchain.agents import create_agent

    model = FakeToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[make_tool_call("search_documents", {"query": "agentstage"}, "call_1")],
            ),
            AIMessage(content="I found the answer."),
        ]
    )
    agent = create_agent(model=model, tools=[search_documents])

    result = await agent.ainvoke({"messages": [HumanMessage(content="find agentstage")]})

    messages = result["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1, "the tool should have executed exactly once"
    assert tool_messages[0].content == "RESULT for agentstage"
    assert tool_messages[0].tool_call_id == "call_1"
    assert messages[-1].content == "I found the answer."


async def test_tool_lifecycle_arrives_on_the_updates_channel():
    """Pins the Milestone 0 finding that shapes the normalizer: tool calls come
    from `updates` (AIMessage.tool_calls), and completion correlates through
    ToolMessage.tool_call_id. Token streaming is a separate channel.
    """
    from langchain.agents import create_agent

    model = FakeToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[make_tool_call("search_documents", {"query": "x"}, "call_1")],
            ),
            AIMessage(content="done"),
        ]
    )
    agent = create_agent(model=model, tools=[search_documents])

    # With a single `stream_mode` (not a list) and no `subgraphs`, each chunk is a
    # bare {node_name: state} dict — the (namespace, chunk) tuple only appears when
    # multiple modes or subgraphs are requested. Verified against langgraph 1.2.11.
    # `ToolCall["id"]` is typed `str | None`, so the normalizer will have to cope with a
    # call that has no id. Keeping the None here rather than asserting it away means an
    # id that ever goes missing shows up as a mismatch instead of a silent skip.
    started_ids: list[str | None] = []
    completed_ids: list[str | None] = []
    nodes_seen: list[str] = []
    async for update in agent.astream(
        {"messages": [HumanMessage(content="go")]}, stream_mode="updates"
    ):
        for node_name, node_state in update.items():
            nodes_seen.append(node_name)
            for message in node_state.get("messages", []):
                if isinstance(message, AIMessage) and message.tool_calls:
                    started_ids.extend(call["id"] for call in message.tool_calls)
                elif isinstance(message, ToolMessage):
                    completed_ids.append(message.tool_call_id)

    assert nodes_seen == ["model", "tools", "model"]
    assert started_ids == ["call_1"]
    assert completed_ids == started_ids, "every started call must correlate to a completion"
