# Comparisons

Honest comparisons with the closest adjacent tools, as of the versions and features described in this repo. These are different tools solving overlapping but distinct problems — the right choice depends on what you're building.

## Streamlit

Streamlit is a general Python data-app framework: any Python variable can become a widget, and the whole script reruns top-to-bottom on interaction. It's broader than agentstage — you can build dashboards, forms, data explorers, anything — but that generality means chat-specific behavior (streaming tokens, tool-call cards, resumable conversations) isn't built in; you assemble it yourself from `st.chat_message`, `st.session_state`, and manual plumbing.

agentstage is narrower and chat-agent-specific by design: it doesn't do dashboards or arbitrary widgets. In exchange, tool-call visibility, human-in-the-loop approval, citations, and file uploads are asking for a feature, not building one.

**Choose Streamlit** if your app is a general data tool that happens to include a chat panel, or if you want the wider Streamlit component ecosystem. **Choose agentstage** if the app *is* the agent conversation and you want that experience handled for you.

## Reflex

Reflex compiles Python into a full React frontend plus a backend, giving you general-purpose, arbitrarily customizable UI while writing only Python — a broader promise than agentstage makes. It was evaluated directly during this project's architecture phase and not used, for two concrete reasons: it's a whole additional UI runtime layered on top of what's already a two-sided problem (agent state, UI state), and its default transport model (a persistent WebSocket connection per user) doesn't fit the one-way, resumable streaming SSE gives for free.

agentstage makes a narrower bet: a normalized event contract over SSE, and a purpose-built (not general-purpose) chat interface. You can't build an arbitrary app in it the way you can in Reflex — you can build an agent chat interface, well.

**Choose Reflex** if you want one Python-only toolchain for a broader app, agent chat included as one part of it. **Choose agentstage** if you specifically want the agent-chat piece to be as good as possible with minimal Python, and are fine with (or already have) a separate stack for everything else.

## LangChain's Agent Chat UI

The [Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui) is a React/Next.js frontend maintained by the LangChain team, talking to a LangGraph server over its own protocol. It's the closest tool in spirit to agentstage's UI — chat, streaming, tool calls, interrupts — and if you're comfortable running a Node/Next.js frontend and a separate LangGraph server process, it's a mature, actively maintained option.

agentstage's difference is where the line is drawn: everything — server, transport, and UI — ships inside one Python package, with the browser assets prebuilt into the wheel. You write Python; you never touch a `package.json`, a frontend build step, or run a second server process. The trade-off is that agentstage's UI is currently a smaller, more constrained surface (see [Project status](../README.md#project-status) for exactly what's built) than a dedicated, longer-lived React frontend can offer.

**Choose Agent Chat UI** if your team already writes React/Next.js, wants full control over the frontend, or needs UI customization beyond what a configuration-driven API offers. **Choose agentstage** if you want zero frontend code and zero separate frontend deployment, and the built-in feature set already covers what you need.

## Summary

| | Streamlit | Reflex | Agent Chat UI | agentstage |
|---|---|---|---|---|
| Scope | General data apps | General Python-only apps | Agent chat, dedicated | Agent chat, dedicated |
| Frontend code required | None | None (compiles from Python) | React/Next.js | None |
| Separate frontend process | No | Yes (compiled + served) | Yes (Next.js) | No |
| Transport | Reruns + WebSocket | WebSocket per user | Its own protocol | SSE |
| Tool-call / approval UI | Build it yourself | Build it yourself | Built in | Built in |
| Ships inside one Python package | Yes | Yes | No | Yes |

None of these numbers are a ranking — they're a map of trade-offs. If your team is already deep in one of these ecosystems, that context usually matters more than any single row above.
