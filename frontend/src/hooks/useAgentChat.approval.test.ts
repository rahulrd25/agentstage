// Regression coverage: a second human-in-the-loop interrupt in the same
// thread used to completely erase the first one's (already-resolved) card
// from the transcript, since approvals lived in a single overwritable slot
// instead of being positioned entries in the transcript itself.

import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, test, vi } from 'vitest'
import { useAgentChat } from './useAgentChat'
import type { AgentEvent } from '../types'

function sseResponse(events: AgentEvent[]): Response {
  const body = events.map((e) => `event: ${e.type}\nid: 0\ndata: ${JSON.stringify(e)}\n\n`).join('')
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(body))
      controller.close()
    },
  })
  return new Response(stream, { status: 200 })
}

describe('useAgentChat — human-in-the-loop transcript persistence', () => {
  const originalFetch = globalThis.fetch

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  test('a second interrupt in the same thread does not erase the first, already-resolved one', async () => {
    const responses: Response[] = [
      // send('send an alert to Sam') -> pauses on an interrupt
      sseResponse([
        { type: 'run_started', run_id: 'r1', thread_id: 'thread-1' },
        {
          type: 'interrupt_created',
          run_id: 'r1',
          thread_id: 'thread-1',
          data: { interrupt_id: 'int-1', value: { question: 'Approve alert to Sam?' } },
        },
      ]),
      // resolveApproval('approved') -> the resumed run completes
      sseResponse([
        { type: 'run_started', run_id: 'r2', thread_id: 'thread-1' },
        { type: 'message_started', run_id: 'r2', thread_id: 'thread-1', message_id: 'm1' },
        {
          type: 'message_completed',
          run_id: 'r2',
          thread_id: 'thread-1',
          message_id: 'm1',
          data: { text: 'Sent to Sam.' },
        },
        { type: 'run_completed', run_id: 'r2', thread_id: 'thread-1' },
      ]),
      // send('send an alert to Priya') -> a second, later interrupt
      sseResponse([
        { type: 'run_started', run_id: 'r3', thread_id: 'thread-1' },
        {
          type: 'interrupt_created',
          run_id: 'r3',
          thread_id: 'thread-1',
          data: { interrupt_id: 'int-2', value: { question: 'Approve alert to Priya?' } },
        },
      ]),
    ]
    let call = 0
    globalThis.fetch = vi.fn(() => Promise.resolve(responses[call++])) as unknown as typeof fetch

    const { result } = renderHook(() => useAgentChat(false))

    await act(async () => {
      await result.current.send('send an alert to Sam', [])
    })
    await act(async () => {
      await result.current.resolveApproval('approved')
    })
    await act(async () => {
      await result.current.send('send an alert to Priya', [])
    })

    const approvals = result.current.transcript.filter((item) => item.kind === 'approval')

    expect(approvals).toHaveLength(2)
    expect(approvals[0].approval).toMatchObject({ id: 'int-1', decision: 'approved' })
    expect(approvals[1].approval).toMatchObject({ id: 'int-2', decision: 'pending' })
  })

  test('a HumanInTheLoopMiddleware interrupt merges into its tool card instead of a second one', async () => {
    const responses: Response[] = [
      // send('send an alert to Priya') -> the model requests the tool call,
      // then a HITLRequest-shaped interrupt gates it.
      sseResponse([
        { type: 'run_started', run_id: 'r1', thread_id: 'thread-1' },
        {
          type: 'tool_call_started',
          run_id: 'r1',
          thread_id: 'thread-1',
          tool_call_id: 'call-1',
          data: { name: 'send_alert', args: { recipient: 'Priya', message: 'server is down' } },
        },
        {
          type: 'interrupt_created',
          run_id: 'r1',
          thread_id: 'thread-1',
          data: {
            interrupt_id: 'int-1',
            value: {
              action_requests: [
                { name: 'send_alert', args: { recipient: 'Priya', message: 'server is down' } },
              ],
              review_configs: [{ action_name: 'send_alert', allowed_decisions: ['approve', 'reject'] }],
            },
          },
        },
      ]),
      // resolveApproval('rejected') -> the resumed run re-fires
      // tool_call_started for the same id (its own bounded run), then fails it.
      sseResponse([
        { type: 'run_started', run_id: 'r2', thread_id: 'thread-1' },
        {
          type: 'tool_call_started',
          run_id: 'r2',
          thread_id: 'thread-1',
          tool_call_id: 'call-1',
          data: { name: 'send_alert', args: { recipient: 'Priya', message: 'server is down' } },
        },
        {
          type: 'tool_call_failed',
          run_id: 'r2',
          thread_id: 'thread-1',
          tool_call_id: 'call-1',
          data: { error: 'User rejected the tool call.', name: 'send_alert' },
        },
        { type: 'run_completed', run_id: 'r2', thread_id: 'thread-1' },
      ]),
    ]
    let call = 0
    globalThis.fetch = vi.fn(() => Promise.resolve(responses[call++])) as unknown as typeof fetch

    const { result } = renderHook(() => useAgentChat(false))

    await act(async () => {
      await result.current.send('send an alert to Priya', [])
    })

    expect(result.current.transcript.filter((item) => item.kind === 'approval')).toHaveLength(0)
    const toolItem = result.current.transcript.find((item) => item.kind === 'tool')
    expect(toolItem?.kind).toBe('tool')
    expect(toolItem?.tool.approval).toMatchObject({ id: 'int-1', decision: 'pending' })

    await act(async () => {
      await result.current.resolveApproval('rejected')
    })

    expect(result.current.transcript.filter((item) => item.kind === 'approval')).toHaveLength(0)
    const resolved = result.current.transcript.find((item) => item.kind === 'tool')
    expect(resolved?.kind).toBe('tool')
    expect(resolved?.tool).toMatchObject({ status: 'failed', approval: { id: 'int-1', decision: 'rejected' } })
  })
})
