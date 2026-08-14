"""A complete agentstage application.

    uv run python examples/basic_chat/app.py

Then open http://127.0.0.1:8000/ — chat input, streaming output, markdown, and
visible tool calls, with no frontend code in this file.

No API key is required: the model is a deterministic fake, so the app is fully
explorable offline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.tools import tool

# The deterministic fake model lives in tests/, which is not an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agentstage import AgentApp
from tests.fakes import FakeToolCallingModel, make_tool_call


@tool
def search_documents(query: str) -> str:
    """Search the internal document store."""
    return (
        f"3 documents match {query!r}:\n"
        "  - intro.md      Getting started\n"
        "  - api.md        Public API reference\n"
        "  - faq.md        Troubleshooting"
    )


ANSWER = """Here's what I found in the document store.

Three documents matched:

- **intro.md** — getting started
- **api.md** — the public API reference
- **faq.md** — troubleshooting

The API reference is probably what you want:

```python
from agentstage import AgentApp

app = AgentApp(agent=agent, title="Document Assistant")

app.chat(streaming=True)
app.tool_calls()
app.run()
```

Ask a follow-up if you'd like detail on any of them."""


class ReplayingModel(FakeToolCallingModel):
    """A fake that restarts its script for every user turn.

    ``FakeToolCallingModel`` is built for tests: it replays a fixed list and repeats
    the last entry once exhausted. In a long-lived server that means the *first*
    request spends the tool-call response and every later one gets only the plain
    answer — so the tool card silently stops appearing.

    Resetting when a turn begins keeps the demo showing the full tool lifecycle on
    every message. A real model needs none of this.
    """

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # A new user turn is one where the agent has not yet produced a tool result;
        # LangGraph re-sends the whole history, so the last message tells us.
        if messages and messages[-1].type == "human":
            self.invocations = []
        return super()._generate(messages, stop, run_manager, **kwargs)


def build_agent() -> Any:
    from langchain.agents import create_agent

    model = ReplayingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[make_tool_call("search_documents", {"query": "api"}, "call_1")],
            ),
            AIMessage(content=ANSWER),
        ]
    )
    return create_agent(model=model, tools=[search_documents])


def main() -> None:
    app = AgentApp(build_agent(), title="Document Assistant")
    app.chat(streaming=True)
    app.tool_calls(visible=True, collapsible=True)
    app.run()


if __name__ == "__main__":
    main()
