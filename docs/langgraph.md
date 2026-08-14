# LangGraph notes

Everything on this page was confirmed by installing the real packages and introspecting or running them — not recalled from documentation or tutorials. Several findings contradict widely-copied example code, which is exactly why they're written down here rather than assumed.

Verified against `langgraph==1.2.11`, `langchain==1.3.15`, `langchain-core==1.5.4`. If you're on different versions, treat these as things to re-verify, not eternal truths — the [regeneration script](../scripts/regenerate_golden_events.py) and the test files cited below are how to check.

## Streaming

**`astream` accepts `version: 'v1' | 'v2'` only.** `version='v3'` exists solely on `astream_events`, a separate method. Passing it to `astream` raises.

**`astream_events(version='v3')` is explicitly experimental**, and raises `TypeError` if you pass `stream_mode` or `subgraphs` — it owns those parameters itself.

**The exact `StreamMode` literals are:** `values`, `updates`, `checkpoints`, `tasks`, `debug`, `messages`, `custom`. Nothing else.

**A single `stream_mode` yields a bare payload; multiple modes yield `(mode, payload)` tuples.** `agent.astream(x, stream_mode="updates")` yields `{node: state}` dicts directly. `agent.astream(x, stream_mode=["updates", "messages"])` yields `(mode, payload)` pairs where you dispatch on `mode`. This distinction is easy to get backward if you test only one mode and assume the other behaves the same — [`LangGraphAdapter._split_chunk`](../src/agentstage/adapters/langgraph.py) exists specifically to handle the multi-mode shape agentstage always uses.

**Token streaming and tool lifecycle arrive on different channels.** A tool call appears on `AIMessage.tool_calls` in the `updates` channel — never in `messages`. The tool's result arrives later, as a separate `updates` chunk carrying a `ToolMessage`. Correlate the two by `tool_call_id`; there is no ordering guarantee between parallel calls' results. See [`StreamNormalizer._start_tool_calls` / `_complete_tool_call`](../src/agentstage/events/normalize.py).

**A non-streaming model still emits a whole message, not chunks, even on the `messages` channel.** The normalizer treats a full message as a single delta plus an immediate completion, so both streaming and non-streaming models render identically to the UI.

## Interrupts and resume

**`Interrupt` has exactly two fields: `value` and `id`.** Tutorials referencing `resumable`, `ns`, or `when` are describing an older or different API — those fields do not exist and code written against them raises `AttributeError`.

**`__interrupt__` holds a *tuple* of `Interrupt` objects**, keyed under that literal reserved string in the `updates` channel: `{'__interrupt__': (Interrupt(...),)}`. A graph can pause on more than one interrupt at once.

**Resuming a thread with no pending interrupt does not raise.** `Command(resume=...)` on a thread whose interrupted node already finished — or that was never interrupted at all — silently **restarts that node from its beginning** and produces a brand-new interrupt, rather than erroring. This is the single most consequential finding in this document: a stale or mistyped `thread_id` looks like a working resume, not a failure. `agentstage` guards against it explicitly — see [`LangGraphAdapter.has_pending_interrupt`](../src/agentstage/adapters/langgraph.py) and its two call sites, one before resuming and one inside `resume()` itself, so the check holds even if a caller skips the first one.

**Approve, reject, and edit are the same mechanism.** `Command(resume=value)` is the only primitive; the interrupted node decides what `value` means. There's no separate LangGraph API for "reject" versus "approve" — see [`ResumeRequest`](../src/agentstage/runtime/fastapi.py), where `decision` is a plain string precisely because of this.

**`interrupt()` has no memoized side effects.** LangGraph re-runs the interrupting node from its start on resume, replaying everything the node did before it called `interrupt()`. Put side effects — sending an email, charging a card — after the `interrupt()` call, not before it.

## Checkpointers

**No checkpointer, no interrupt — silently.** `interrupt()` requires a checkpointer to actually pause a run; without one, LangGraph does not raise a helpful error at graph-compile time. `agentstage` checks explicitly and raises before a run starts — see [`LangGraphAdapter.require_checkpointer`](../src/agentstage/adapters/langgraph.py).

**`aget_state` requires a checkpointer too**, and raises `ValueError("No checkpointer set")` if the agent has none — this is what backs both `has_pending_interrupt` and the thread-history transcript endpoint, which is why `AgentApp.thread_history(enabled=True)` calls the same checkpointer guard as `human_approval`.

**`aget_state` on a thread the checkpointer has never seen returns empty state**, not an error — `values == {}`, `next == ()`, `interrupts == ()`. A UI listing a brand-new or never-opened thread should treat this as a normal empty conversation.

## Message content and citations

**Citations are a real, standard LangChain shape**, not something agentstage invented: `content = [{"type": "text", "text": ..., "annotations": [Citation, ...]}]`, where `Citation` is `langchain_core.messages.content.Citation` with `url`, `title`, `cited_text`, `start_index`, `end_index`. `create_citation(...)` builds one. See [`text_of` / `citations_of`](../src/agentstage/events/normalize.py).

**File and image content also use standard blocks.** `create_image_block`/`create_file_block` from `langchain_core.messages.content` produce the shapes a vision- or document-capable model already knows how to read — agentstage builds these directly rather than inventing a parallel attachment format. See [`content_block_for`](../src/agentstage/files.py).

## Sources

Findings above were checked directly against the installed packages using small probe scripts, not derived from external documentation. Where a claim mattered enough to break something if wrong, it's backed by a test — search `tests/unit/test_normalize.py`, `tests/integration/test_langgraph_adapter.py`, and `tests/unit/test_files.py` for the corresponding assertions.
