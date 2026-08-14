"""Adapter tests driving real compiled LangGraph agents. No API key required."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from agentstage.adapters import LangGraphAdapter
from agentstage.errors import AdapterError, ConfigError
from agentstage.events import AgentEvent
from tests.fakes import FakeToolCallingModel, make_tool_call


@tool
def search_documents(query: str) -> str:
    """Search the document store."""
    return f"RESULT for {query}"


@tool
def boom_tool(query: str) -> str:
    """Always raises."""
    msg = f"tool exploded on {query}"
    raise ValueError(msg)


def tool_calling_agent(*, tools: list[Any], checkpointer: Any = None):
    from langchain.agents import create_agent

    model = FakeToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[make_tool_call(tools[0].name, {"query": "x"}, "call_1")],
            ),
            AIMessage(content="All done."),
        ]
    )
    return create_agent(model=model, tools=tools, checkpointer=checkpointer)


def streaming_agent(text: str):
    """A graph whose model streams token by token."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content=text)]))

    async def call_model(state: MessagesState) -> dict[str, Any]:
        return {"messages": [await model.ainvoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    return graph.compile()


async def collect(adapter: LangGraphAdapter, message: str, **kwargs) -> list[AgentEvent]:
    return [event async for event in adapter.stream(message, **kwargs)]


def types_of(events: list[AgentEvent]) -> list[str]:
    return [e.type for e in events]


# ---- Construction --------------------------------------------------------


def test_an_uncompiled_graph_is_rejected_with_the_fix():
    """Requirement 12. Caught at construction, not mid-stream with an
    AttributeError while a user waits."""
    graph = StateGraph(MessagesState)

    with pytest.raises(AdapterError, match=r"\.compile\(\)"):
        # Passing an uncompiled graph is the whole point; the type error here is
        # exactly what a user would hit, and the runtime check is the safety net.
        LangGraphAdapter(graph)  # type: ignore[arg-type]


def test_a_compiled_agent_is_accepted():
    assert LangGraphAdapter(streaming_agent("hi")) is not None


# ---- Checkpointer guard (risk R3) ---------------------------------------


def test_missing_checkpointer_is_detected_before_a_run():
    """Risk R3: LangGraph cannot interrupt without a checkpointer and the symptom
    is a run that silently never pauses."""
    adapter = LangGraphAdapter(tool_calling_agent(tools=[search_documents]))

    with pytest.raises(ConfigError, match="requires a checkpointer"):
        adapter.require_checkpointer()


def test_the_checkpointer_error_shows_how_to_fix_it():
    adapter = LangGraphAdapter(tool_calling_agent(tools=[search_documents]))

    with pytest.raises(ConfigError, match="InMemorySaver"):
        adapter.require_checkpointer()


def test_a_checkpointed_agent_passes_the_guard():
    adapter = LangGraphAdapter(
        tool_calling_agent(tools=[search_documents], checkpointer=InMemorySaver())
    )

    adapter.require_checkpointer()


# ---- Run lifecycle -------------------------------------------------------


async def test_a_run_starts_and_completes():
    adapter = LangGraphAdapter(streaming_agent("Hello world"))

    events = await collect(adapter, "hi")

    assert types_of(events)[0] == "run_started"
    assert types_of(events)[-1] == "run_completed"


async def test_every_event_shares_the_run_id():
    adapter = LangGraphAdapter(streaming_agent("hi"))

    events = await collect(adapter, "hi", run_id="run-fixed")

    assert {e.run_id for e in events} == {"run-fixed"}


async def test_a_run_id_is_generated_when_omitted():
    adapter = LangGraphAdapter(streaming_agent("hi"))

    events = await collect(adapter, "hi")

    assert events[0].run_id.startswith("run-")


async def test_sequence_numbers_are_gapless():
    """A client detects dropped SSE frames by gaps, so there must be none here."""
    adapter = LangGraphAdapter(streaming_agent("Hello world from agent"))

    events = await collect(adapter, "hi")

    assert [e.sequence for e in events] == list(range(len(events)))


async def test_the_thread_id_is_propagated_to_events():
    adapter = LangGraphAdapter(
        tool_calling_agent(tools=[search_documents], checkpointer=InMemorySaver())
    )

    events = await collect(adapter, "go", thread_id="t-42")

    assert events[0].thread_id == "t-42"


# ---- Streaming -----------------------------------------------------------


async def test_text_streams_as_incremental_deltas():
    adapter = LangGraphAdapter(streaming_agent("Hello world from agent"))

    events = await collect(adapter, "hi")
    deltas = [e for e in events if e.type == "message_delta"]

    assert len(deltas) > 1, "a streaming model should produce multiple deltas"
    joined = "".join(e.data["text"] for e in deltas if e.data)
    assert joined == "Hello world from agent"


# ---- Attachments ---------------------------------------------------------


def echoing_agent() -> tuple[Any, list[Any]]:
    """An agent that reports back exactly what content it received, so the test
    can assert on the HumanMessage shape without a real model in the way."""
    from langchain_core.messages import AIMessage as _AIMessage

    received: list[Any] = []

    async def call_model(state: MessagesState) -> dict[str, Any]:
        received.append(state["messages"][-1].content)
        return {"messages": [_AIMessage(content="received")]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    return graph.compile(), received


async def test_content_blocks_ride_alongside_the_text_message():
    agent, received = echoing_agent()
    adapter = LangGraphAdapter(agent)
    block = {"type": "file", "base64": "aGk=", "mime_type": "application/pdf"}

    await collect(adapter, "check this file", content_blocks=[block])

    assert received[0] == [{"type": "text", "text": "check this file"}, block]


async def test_multiple_content_blocks_all_ride_along():
    agent, received = echoing_agent()
    adapter = LangGraphAdapter(agent)
    blocks = [
        {"type": "image", "base64": "aGk=", "mime_type": "image/png"},
        {"type": "file", "base64": "eWk=", "mime_type": "application/pdf"},
    ]

    await collect(adapter, "two files", content_blocks=blocks)

    assert received[0] == [{"type": "text", "text": "two files"}, *blocks]


async def test_no_content_blocks_sends_a_plain_string_as_before():
    """Attachments must not change the shape of an ordinary message — a plain
    string, not a single-element block list, so existing agents built for a bare
    string prompt keep working unchanged."""
    agent, received = echoing_agent()
    adapter = LangGraphAdapter(agent)

    await collect(adapter, "just text")

    assert received[0] == "just text"


async def test_a_streamed_message_is_opened_once_and_closed_once():
    adapter = LangGraphAdapter(streaming_agent("Hello world"))

    events = await collect(adapter, "hi")

    assert types_of(events).count("message_started") == 1
    assert types_of(events).count("message_completed") == 1


async def test_deltas_share_one_message_id():
    adapter = LangGraphAdapter(streaming_agent("Hello world"))

    events = await collect(adapter, "hi")
    ids = {e.message_id for e in events if e.type == "message_delta"}

    assert len(ids) == 1, "deltas must append to a single message, not fan out"


# ---- Tool lifecycle ------------------------------------------------------


async def test_a_tool_call_starts_and_completes_in_order():
    adapter = LangGraphAdapter(tool_calling_agent(tools=[search_documents]))

    events = await collect(adapter, "find x")
    kinds = types_of(events)

    assert "tool_call_started" in kinds
    assert "tool_call_completed" in kinds
    assert kinds.index("tool_call_started") < kinds.index("tool_call_completed")


async def test_the_tool_result_correlates_to_the_call():
    adapter = LangGraphAdapter(tool_calling_agent(tools=[search_documents]))

    events = await collect(adapter, "find x")
    started = next(e for e in events if e.type == "tool_call_started")
    completed = next(e for e in events if e.type == "tool_call_completed")

    assert started.tool_call_id == completed.tool_call_id == "call_1"
    assert started.data == {"name": "search_documents", "args": {"query": "x"}}
    assert completed.data is not None
    assert completed.data["result"] == "RESULT for x"


async def test_a_failing_tool_fails_the_call_and_the_run():
    """LangGraph re-raises non-ToolInvocationError tool failures (verified). Both
    the card and the run must resolve, or the UI spins forever."""
    adapter = LangGraphAdapter(tool_calling_agent(tools=[boom_tool]))

    events = await collect(adapter, "go")
    kinds = types_of(events)

    assert "tool_call_failed" in kinds
    assert kinds[-1] == "run_failed"


async def test_a_failing_tool_reports_the_message_not_a_traceback():
    adapter = LangGraphAdapter(tool_calling_agent(tools=[boom_tool]))

    events = await collect(adapter, "go")
    failed = next(e for e in events if e.type == "tool_call_failed")

    assert failed.data is not None
    assert "tool exploded" in failed.data["error"]
    assert "Traceback" not in failed.data["error"]


async def test_a_failed_run_still_terminates_exactly_once():
    """Requirement 13: a client must always learn the run is over."""
    adapter = LangGraphAdapter(tool_calling_agent(tools=[boom_tool]))

    kinds = types_of(await collect(adapter, "go"))

    assert kinds.count("run_failed") == 1
    assert kinds.count("run_completed") == 0


# ---- Human-in-the-loop --------------------------------------------------


async def test_an_interrupt_is_surfaced_as_an_event():
    from typing import TypedDict

    from langgraph.types import interrupt

    class State(TypedDict):
        steps: list[str]

    def approve(state: State) -> State:
        decision = interrupt({"question": "Approve sending email?"})
        return {"steps": [*state["steps"], f"decision={decision}"]}

    graph = StateGraph(State)
    graph.add_node("approve", approve)
    graph.add_edge(START, "approve")
    graph.add_edge("approve", END)
    agent = graph.compile(checkpointer=InMemorySaver())

    adapter = LangGraphAdapter(agent)
    events = [
        event
        async for event in adapter.stream("go", thread_id="t-hitl")
        if event.type == "interrupt_created"
    ]

    assert len(events) == 1
    assert events[0].data is not None
    assert events[0].data["value"] == {"question": "Approve sending email?"}
    assert events[0].thread_id == "t-hitl", "resuming requires the thread_id"


def approval_agent() -> Any:
    """A graph whose single node pauses for approval, then records the decision."""
    from typing import TypedDict

    from langgraph.types import interrupt

    class State(TypedDict):
        steps: list[str]

    def approve(state: State) -> State:
        decision = interrupt({"question": "Approve?"})
        return {"steps": [*state.get("steps", []), f"decision={decision}"]}

    graph = StateGraph(State)
    graph.add_node("approve", approve)
    graph.add_edge(START, "approve")
    graph.add_edge("approve", END)
    return graph.compile(checkpointer=InMemorySaver())


async def test_has_pending_interrupt_is_false_before_any_run():
    adapter = LangGraphAdapter(approval_agent())

    assert await adapter.has_pending_interrupt("never-run") is False


async def test_has_pending_interrupt_is_true_after_an_interrupt():
    adapter = LangGraphAdapter(approval_agent())
    await collect(adapter, "go", thread_id="t1")

    assert await adapter.has_pending_interrupt("t1") is True


async def test_has_pending_interrupt_is_false_after_resuming():
    adapter = LangGraphAdapter(approval_agent())
    await collect(adapter, "go", thread_id="t1")

    events = [e async for e in adapter.resume(thread_id="t1", resume_value=True)]

    assert types_of(events)[-1] == "run_completed"
    assert await adapter.has_pending_interrupt("t1") is False


async def test_resuming_a_thread_with_no_interrupt_is_rejected():
    """The load-bearing guard: LangGraph does not error on this (verified) — it
    silently starts a fresh run. agentstage must catch it here instead."""
    adapter = LangGraphAdapter(approval_agent())

    with pytest.raises(ConfigError, match="no pending interrupt"):
        async for _ in adapter.resume(thread_id="never-run", resume_value=True):
            pass


async def test_resuming_an_already_resolved_thread_is_rejected():
    adapter = LangGraphAdapter(approval_agent())
    await collect(adapter, "go", thread_id="t1")
    async for _ in adapter.resume(thread_id="t1", resume_value=True):
        pass

    with pytest.raises(ConfigError, match="no pending interrupt"):
        async for _ in adapter.resume(thread_id="t1", resume_value=True):
            pass


async def test_the_resume_value_reaches_the_interrupted_node():
    adapter = LangGraphAdapter(approval_agent())
    await collect(adapter, "go", thread_id="t1")

    async for _ in adapter.resume(thread_id="t1", resume_value="a specific answer"):
        pass

    snapshot = await adapter.agent.aget_state({"configurable": {"thread_id": "t1"}})
    assert snapshot.values["steps"] == ["decision=a specific answer"]


async def test_rejecting_is_just_a_different_resume_value():
    """approve/reject are not separate LangGraph calls — verified — Command(resume=...)
    carries whichever value the caller chose, and the node decides what it means."""
    adapter = LangGraphAdapter(approval_agent())
    await collect(adapter, "go", thread_id="t1")

    async for _ in adapter.resume(thread_id="t1", resume_value=False):
        pass

    snapshot = await adapter.agent.aget_state({"configurable": {"thread_id": "t1"}})
    assert snapshot.values["steps"] == ["decision=False"]


# ---- Config merging ------------------------------------------------------


def test_thread_id_is_merged_into_configurable():
    merged = LangGraphAdapter._build_config(thread_id="t1", config=None)

    assert merged == {"configurable": {"thread_id": "t1"}}


def test_a_caller_config_is_preserved_alongside_the_thread_id():
    merged = LangGraphAdapter._build_config(
        thread_id="t1", config={"configurable": {"user_tag": "x"}, "recursion_limit": 5}
    )

    assert merged["recursion_limit"] == 5
    assert merged["configurable"] == {"user_tag": "x", "thread_id": "t1"}


def test_the_explicit_thread_id_wins_over_one_in_config():
    """The UI resumes on the thread_id it passed; a stale config value would
    silently write to the wrong conversation."""
    merged = LangGraphAdapter._build_config(
        thread_id="authoritative", config={"configurable": {"thread_id": "stale"}}
    )

    assert merged["configurable"]["thread_id"] == "authoritative"


def test_the_caller_config_is_not_mutated():
    original = {"configurable": {"thread_id": "stale"}}

    LangGraphAdapter._build_config(thread_id="new", config=original)

    assert original == {"configurable": {"thread_id": "stale"}}


# ---- Contract guards -----------------------------------------------------


def test_a_non_tuple_chunk_is_reported():
    with pytest.raises(AdapterError, match="mode, payload"):
        LangGraphAdapter._split_chunk({"not": "a tuple"})


async def test_no_event_carries_chain_of_thought():
    """Security requirement, checked end to end on a real run."""
    adapter = LangGraphAdapter(tool_calling_agent(tools=[search_documents]))

    events = await collect(adapter, "go")

    for event in events:
        payload = event.to_json()
        assert "reasoning" not in payload
        assert "additional_kwargs" not in payload


# ---- History ---------------------------------------------------------------


async def test_history_is_empty_for_a_thread_the_checkpointer_never_saw():
    """Not an error — a client listing a never-opened thread is normal, not a
    mistake."""
    agent = tool_calling_agent(tools=[search_documents], checkpointer=InMemorySaver())
    adapter = LangGraphAdapter(agent)

    assert await adapter.get_history("never-run") == []


async def test_history_reconstructs_user_and_assistant_turns():
    checkpointer = InMemorySaver()
    adapter = LangGraphAdapter(
        tool_calling_agent(tools=[search_documents], checkpointer=checkpointer)
    )
    await collect(adapter, "find x", thread_id="t1")

    history = await adapter.get_history("t1")

    assert [h["role"] for h in history] == ["user", "assistant", "tool", "assistant"]
    assert history[0]["text"] == "find x"
    assert history[-1]["text"] == "All done."


async def test_history_includes_the_tool_call_that_produced_a_tool_message():
    checkpointer = InMemorySaver()
    adapter = LangGraphAdapter(
        tool_calling_agent(tools=[search_documents], checkpointer=checkpointer)
    )
    await collect(adapter, "find x", thread_id="t1")

    history = await adapter.get_history("t1")
    ai_with_call = next(h for h in history if h.get("tool_calls"))

    assert ai_with_call["tool_calls"][0]["name"] == "search_documents"
    assert ai_with_call["tool_calls"][0]["id"] == "call_1"


async def test_history_matches_the_tool_call_id_on_the_tool_turn():
    """The same correlation the live event stream uses, so history and a live
    run render tool cards identically."""
    checkpointer = InMemorySaver()
    adapter = LangGraphAdapter(
        tool_calling_agent(tools=[search_documents], checkpointer=checkpointer)
    )
    await collect(adapter, "find x", thread_id="t1")

    history = await adapter.get_history("t1")
    tool_turn = next(h for h in history if h["role"] == "tool")

    assert tool_turn["tool_call_id"] == "call_1"
    assert tool_turn["text"] == "RESULT for x"


async def test_history_reflects_a_second_turn_on_the_same_thread():
    checkpointer = InMemorySaver()
    adapter = LangGraphAdapter(streaming_agent_with_checkpointer(checkpointer))
    await collect(adapter, "first", thread_id="t1")
    await collect(adapter, "second", thread_id="t1")

    history = await adapter.get_history("t1")

    user_turns = [h["text"] for h in history if h["role"] == "user"]
    assert user_turns == ["first", "second"]


async def test_history_includes_citations_from_a_completed_message():
    from langchain_core.messages.content import create_citation

    checkpointer = InMemorySaver()

    def call_model(state: MessagesState) -> dict[str, Any]:
        citation = create_citation(url="https://example.com", title="Source")
        return {
            "messages": [
                AIMessage(
                    content=[{"type": "text", "text": "cited answer", "annotations": [citation]}]
                )
            ]
        }

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    adapter = LangGraphAdapter(graph.compile(checkpointer=checkpointer))
    await collect(adapter, "cite something", thread_id="t1")

    history = await adapter.get_history("t1")
    assistant_turn = next(h for h in history if h["role"] == "assistant")

    assert assistant_turn["citations"] == [{"url": "https://example.com", "title": "Source"}]


def streaming_agent_with_checkpointer(checkpointer: Any):
    model = GenericFakeChatModel(messages=iter([AIMessage(content="a"), AIMessage(content="b")]))

    async def call_model(state: MessagesState) -> dict[str, Any]:
        return {"messages": [await model.ainvoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    return graph.compile(checkpointer=checkpointer)
