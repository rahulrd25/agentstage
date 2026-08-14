"""A complete agentstage application with human-in-the-loop approval.

    uv run python examples/human_approval/app.py

Then open http://127.0.0.1:8000/ and ask the agent to send an email. It will
pause and show an approval card before actually "sending" it.

No API key is required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agentstage import AgentApp


def build_agent() -> Any:
    """A graph with one node that pauses before "sending" anything.

    Real HITL policy — deciding *which* tool calls need approval — belongs to the
    agent's graph, not to agentstage. This shows the mechanism: interrupt(),
    Command(resume=...), and everything else is handled by the adapter.
    """
    from langchain_core.messages import AIMessage

    def send_email(state: MessagesState) -> dict[str, Any]:
        approved = interrupt(
            {
                "question": "Send this email to the customer?",
                "to": "customer@example.com",
                "subject": "Your order has shipped",
            }
        )
        if approved:
            text = "Sent. The customer will get a shipping notification shortly."
        else:
            text = "Understood — I did not send the email."
        return {"messages": [AIMessage(content=text)]}

    graph = StateGraph(MessagesState)
    graph.add_node("send_email", send_email)
    graph.add_edge(START, "send_email")
    graph.add_edge("send_email", END)
    return graph.compile(checkpointer=InMemorySaver())


def main() -> None:
    app = AgentApp(build_agent(), title="Support Assistant")
    app.chat(streaming=True)
    app.human_approval(enabled=True)
    app.run()


if __name__ == "__main__":
    main()
