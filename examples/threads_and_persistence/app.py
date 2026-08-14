"""A complete agentstage application with a conversation sidebar.

    uv run python examples/threads_and_persistence/app.py

Then open http://127.0.0.1:8000/. Send a few messages, click "+ New chat" to
start another conversation, and switch between them in the sidebar — each
conversation's history reloads from the agent's checkpointer.

Conversations here persist to a SQLite file next to this script (threads.db) —
restart the server and the sidebar still shows your past chats. Message content
persists via the checkpointer for the lifetime of the process; swap in a
persistent checkpointer (see LangGraph's checkpoint savers) for that to survive
a restart too.

No API key is required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agentstage import AgentApp
from agentstage.storage import SQLiteThreadStore

DB_PATH = Path(__file__).parent / "threads.db"


def build_agent() -> Any:
    """A simple echo-style agent — the point of this example is the sidebar,
    not the agent logic. Any real agent works the same way."""

    def call_model(state: MessagesState) -> dict[str, Any]:
        last = state["messages"][-1]
        return {"messages": [AIMessage(content=f"You said: {last.content}")]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    return graph.compile(checkpointer=InMemorySaver())


def main() -> None:
    app = AgentApp(
        build_agent(), title="Chat History Demo", thread_store=SQLiteThreadStore(DB_PATH)
    )
    app.chat(streaming=True)
    app.thread_history(enabled=True)
    app.run()


if __name__ == "__main__":
    main()
