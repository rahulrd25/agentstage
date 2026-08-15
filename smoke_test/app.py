"""
Smoke test for the published `agentstage-ui` PyPI package.

Verifies the package works end-to-end against a *real* LangChain agent
(OpenAI model + real tools, human-in-the-loop approval, thread history), not
the deterministic fake used in the repo's own examples. Everything here is
installed from PyPI in a clean venv, nothing imports from the agentstage
source repo.

    source .venv/bin/activate
    python app.py

Then open http://127.0.0.1:8000/
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from agentstage import AgentApp

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

if not os.environ.get("OPENAI_API_KEY"):
    raise SystemExit(
        "OPENAI_API_KEY is not set. Add it to a .env file next to this script's "
        "parent directory (OPENAI_API_KEY=sk-...) or export it in your shell "
        "before running."
    )


@tool
def get_weather(city: str) -> str:
    """Look up the current weather for a city from live Open-Meteo data."""
    try:
        geo = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        ).json()
        results = geo.get("results") or []
        if not results:
            return f"No location found for {city!r}."
        place = results[0]

        forecast = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,weather_code",
            },
            timeout=10,
        ).json()
    except httpx.HTTPError as exc:
        return f"Weather lookup failed: {exc}"

    current = forecast["current"]
    where = ", ".join(filter(None, [place["name"], place.get("country")]))
    return f"{where}: {current['temperature_2m']}°C (WMO weather code {current['weather_code']})"


@tool
def send_alert(recipient: str, message: str) -> str:
    """Send an alert message to a person."""
    # Mocked send for this smoke test, the docstring above is the tool's
    # description as seen by the LLM, so it must not hint that the action
    # isn't real or the model talks itself out of calling the tool.
    return f"Alert sent to {recipient}: {message!r}"


def build_agent():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return create_agent(
        model=model,
        tools=[get_weather, send_alert],
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"send_alert": True})],
        checkpointer=InMemorySaver(),
    )


def main() -> None:
    app = AgentApp(build_agent(), title="agentstage PyPI Smoke Test")
    app.chat(streaming=True)
    app.tool_calls(visible=True, collapsible=True)
    app.human_approval(enabled=True)
    app.thread_history(enabled=True)
    app.run(
        host=os.environ.get("SMOKE_TEST_HOST", "127.0.0.1"),
        port=int(os.environ.get("SMOKE_TEST_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
