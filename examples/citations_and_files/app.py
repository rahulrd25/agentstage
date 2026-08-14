"""A complete agentstage application with citations and file uploads.

    uv run python examples/citations_and_files/app.py

Then open http://127.0.0.1:8000/. Ask a question to see a "Sources" list under
the answer, or click the paperclip to attach a file and ask about it.

No API key is required.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.messages.content import create_citation
from langgraph.graph import END, START, MessagesState, StateGraph

from agentstage import AgentApp

CITED_ANSWER = """agentstage turns a LangChain/LangGraph agent into a web app using
only Python — no React, no manual frontend code."""


def build_agent() -> Any:
    """A single-node graph so the example stays focused on citations and files,
    not agent architecture. Look at what the model node returns, not how it's
    wired: any real model that emits the same content shapes works the same way.
    """

    def call_model(state: MessagesState) -> dict[str, Any]:
        last = state["messages"][-1]
        has_attachment = isinstance(last.content, list) and any(
            isinstance(b, dict) and b.get("type") in ("file", "image") for b in last.content
        )

        if has_attachment:
            text = (
                "I received your file. In a real agent, a vision- or "
                "document-capable model would read its contents here."
            )
            return {"messages": [AIMessage(content=text)]}

        citation = create_citation(
            url="https://github.com/anthropics",
            title="agentstage README",
            cited_text="a Python-first UI layer",
        )
        content = [{"type": "text", "text": CITED_ANSWER, "annotations": [citation]}]
        return {"messages": [AIMessage(content=content)]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    return graph.compile()


def main() -> None:
    app = AgentApp(build_agent(), title="Research Assistant")
    app.chat(streaming=True, citations=True, attachments=True)
    app.run()


if __name__ == "__main__":
    main()
