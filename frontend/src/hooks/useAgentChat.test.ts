// Regression coverage: resuming a run after a human-in-the-loop interrupt
// legitimately restarts the same tool_call_id in its own bounded run. The
// transcript used to append a second, duplicate card for it instead of
// resetting the original one in place — this is what fixes that.

import { describe, expect, test } from 'vitest'
import { upsertToolCall } from './useAgentChat'
import type { TranscriptItem, ToolCallState } from './useAgentChat'

function tool(overrides: Partial<ToolCallState> = {}): ToolCallState {
  return { id: 'call-1', name: 'send_alert', args: {}, status: 'running', ...overrides }
}

describe('upsertToolCall', () => {
  test('a new tool_call_id is appended', () => {
    const result = upsertToolCall([], tool())
    expect(result).toEqual([{ kind: 'tool', tool: tool() }])
  })

  test('a repeated tool_call_id (a resumed run restarting it) replaces the existing card in place', () => {
    const started: TranscriptItem[] = [{ kind: 'tool', tool: tool() }]

    const resumed = upsertToolCall(started, tool({ status: 'running' }))

    expect(resumed).toHaveLength(1)
    expect(resumed[0]).toEqual({ kind: 'tool', tool: tool({ status: 'running' }) })
  })

  test('other transcript items are left untouched', () => {
    const items: TranscriptItem[] = [
      { kind: 'message', message: { id: 'm1', role: 'assistant', text: 'hi', streaming: false } },
      { kind: 'tool', tool: tool() },
    ]

    const result = upsertToolCall(items, tool({ status: 'failed', error: 'nope' }))

    expect(result).toHaveLength(2)
    expect(result[0]).toBe(items[0])
    expect(result[1]).toEqual({ kind: 'tool', tool: tool({ status: 'failed', error: 'nope' }) })
  })
})
