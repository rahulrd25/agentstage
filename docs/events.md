# The event contract

`AgentEvent` is the normalized, stable contract between the LangGraph adapter and any UI — the one built in, or your own. This doc is for anyone consuming the SSE stream directly rather than through `AgentApp`'s built-in UI.

Source: [`src/agentstage/events/models.py`](../src/agentstage/events/models.py).

## Shape

```python
@dataclass(frozen=True, slots=True)
class AgentEvent:
    type: EventType  # required
    run_id: str  # required
    thread_id: str | None = None
    node_name: str | None = None
    message_id: str | None = None
    tool_call_id: str | None = None
    sequence: int | None = None
    data: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
```

Every field except `type` and `run_id` is optional. Real LangGraph events don't all carry a node name, a message id, or a thread id — inventing values to fill a rigid schema would be a lie the UI then renders as fact.

Events are frozen and immutable: an SSE stream may hand the same event to more than one consumer.

`to_dict()` (and therefore the JSON sent over the wire) omits any field that is `None`, keeping frames small — the payload travels per token, so absent fields shouldn't cost bytes.

## Event types

| Type | Requires | `data` shape |
|---|---|---|
| `run_started` | — | — |
| `run_completed` | — | — |
| `run_failed` | — | `{error, error_type?}` |
| `message_started` | `message_id` | — |
| `message_delta` | `message_id` | `{text}` |
| `message_completed` | `message_id` | `{text?, citations?}` |
| `tool_call_started` | `tool_call_id` | `{name, args}` |
| `tool_call_delta` | `tool_call_id` | `{args_delta}` |
| `tool_call_completed` | `tool_call_id` | `{result, name?}` |
| `tool_call_failed` | `tool_call_id` | `{error, name?}` |
| `interrupt_created` | — | `{interrupt_id, value}` |
| `progress_updated` | — | provider-specific |
| `state_updated` | — | provider-specific |

`tool_call_*` events require `tool_call_id` — without it a result can never be matched back to the card that started it. `message_*` events require `message_id` — without it every streamed token would create a new message bubble instead of appending to one. `AgentEvent.__post_init__` enforces both and raises `EventNormalizationError` if violated.

## A worked example

A real SSE transcript for one message that triggers a tool call (captured by running the app; message ids shortened to `m1`/`m2` here for readability — the real ones are full UUIDs):

```
event: run_started
id: 0
data: {"type":"run_started","run_id":"run-e63052a3f70c","thread_id":"thread-3286114814ae","sequence":0}

event: message_started
id: 1
data: {"type":"message_started","run_id":"run-e63052a3f70c","thread_id":"thread-3286114814ae","node_name":"model","message_id":"m1","sequence":1}

event: tool_call_started
id: 2
data: {"type":"tool_call_started","run_id":"run-e63052a3f70c","thread_id":"thread-3286114814ae","node_name":"model","tool_call_id":"call_1","sequence":2,"data":{"name":"search_documents","args":{"query":"agentstage"}}}

event: message_completed
id: 3
data: {"type":"message_completed","run_id":"run-e63052a3f70c","thread_id":"thread-3286114814ae","message_id":"m1","sequence":3}

event: tool_call_completed
id: 4
data: {"type":"tool_call_completed","run_id":"run-e63052a3f70c","thread_id":"thread-3286114814ae","node_name":"tools","tool_call_id":"call_1","sequence":4,"data":{"result":"3 documents found","name":"search_documents"}}

event: message_started
id: 5
data: {"type":"message_started","run_id":"run-e63052a3f70c","thread_id":"thread-3286114814ae","node_name":"model","message_id":"m2","sequence":5}

event: message_delta
id: 6
data: {"type":"message_delta","run_id":"run-e63052a3f70c","thread_id":"thread-3286114814ae","node_name":"model","message_id":"m2","sequence":6,"data":{"text":"I found the answer."}}

event: message_completed
id: 7
data: {"type":"message_completed","run_id":"run-e63052a3f70c","thread_id":"thread-3286114814ae","message_id":"m2","sequence":7,"data":{"text":"I found the answer."}}

event: run_completed
id: 8
data: {"type":"run_completed","run_id":"run-e63052a3f70c","thread_id":"thread-3286114814ae","sequence":8}
```

Notice `thread_id` appears on every event — it's carried through the whole run, not just `run_started` — and the assistant's first turn (`m1`) closes with a bare `message_completed` carrying no `data` at all, because it produced only a tool call, no text. A UI should drop that empty bubble rather than render it; the reference UI does exactly this in `closeAssistantMessage` in [`app.js`](../src/agentstage/static/app.js).

## Guarantees

**Exactly one terminal event per run.** Every run ends in precisely one `run_completed` or `run_failed` — never both, never neither, never zero. The one exception: a run that pauses on `interrupt_created` ends its stream there, with neither terminal event, because the run is paused, not finished or failed. Resuming it via `POST /resume` starts a fresh SSE stream that itself ends in exactly one terminal event.

**Sequence numbers are gapless and monotonic within a run.** Starting from 0, incrementing by exactly 1 per event. A client can use this to detect a dropped SSE frame.

**No chain-of-thought ever appears.** `AgentEvent.sanitized` strips a fixed set of known reasoning-related keys (`reasoning`, `reasoning_content`, `thinking`, `thought`, `chain_of_thought`, `additional_kwargs`) from `data` before an event reaches the SSE encoder — applied at the transport boundary in [`sse.py`](../src/agentstage/runtime/sse.py), so server-side logging can still see the full event while the browser never can.

**Tool call correlation is by id, never by order.** Parallel tool calls can complete in any order; a UI must key its tool cards by `tool_call_id`, not by arrival sequence.

## Backward compatibility

Once published, this is a wire contract other code depends on. A [golden-file contract test](../tests/unit/test_event_contract.py) pins the exact serialized shape of one example per event type — adding an optional field is backward-compatible and won't break it; renaming, removing, or reordering a field will, on purpose. If you change the wire format, regenerate the golden file deliberately with `python scripts/regenerate_golden_events.py` so the change is visible in review.

## Building your own UI

You don't have to use the shipped UI. Anything that can read SSE can consume this contract directly:

```python
from agentstage import AgentApp

app = AgentApp(agent)
router = app.router()  # mount this; skip AgentApp's own UI entirely
```

Then `POST /chat` with `{"message": "..."}` and read the `text/event-stream` response — each `data:` line is one `AgentEvent`, JSON-encoded exactly as shown above.
