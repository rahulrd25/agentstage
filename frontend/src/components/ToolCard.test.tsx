// Regression coverage for the merged tool+approval card: a
// HumanInTheLoopMiddleware interrupt now renders as part of this card
// (approve/reject, human-readable args) instead of a separate one.

import { fireEvent, render } from '@testing-library/react'
import { describe, expect, test, vi } from 'vitest'
import { ToolCard } from './ToolCard'
import type { ToolCallState } from '../hooks/useAgentChat'

function tool(overrides: Partial<ToolCallState> = {}): ToolCallState {
  return {
    id: 'call-1',
    name: 'send_alert',
    args: { recipient: 'Priya', message: 'server is down' },
    status: 'running',
    ...overrides,
  }
}

describe('ToolCard', () => {
  test('a completed tool renders human-readable arguments, not raw JSON', () => {
    const { getByText, queryByText } = render(
      <ToolCard tool={tool({ status: 'completed', result: 'sent' })} />,
    )

    fireEvent.click(getByText('send_alert'))

    expect(getByText('recipient')).toBeInTheDocument()
    expect(getByText('Priya')).toBeInTheDocument()
    expect(queryByText(/"recipient"/)).not.toBeInTheDocument()
  })

  test('a pending approval shows Approve/Reject and calls onDecideApproval', () => {
    const onDecideApproval = vi.fn()
    const { getByText } = render(
      <ToolCard
        tool={tool({ approval: { id: 'int-1', decision: 'pending' } })}
        onDecideApproval={onDecideApproval}
      />,
    )

    expect(getByText('Awaiting approval')).toBeInTheDocument()
    fireEvent.click(getByText('Reject'))

    expect(onDecideApproval).toHaveBeenCalledWith('rejected')
    expect(getByText('Approve')).toBeDisabled()
  })

  test('a resolved approval shows the decision as the status, with no leftover Approve/Reject', () => {
    // A failed tool starts expanded by default (same as any other failure),
    // so the body — and the absence of Approve/Reject inside it — is already
    // visible without needing to click anything open first.
    const { getByText, queryByText } = render(
      <ToolCard
        tool={tool({
          status: 'failed',
          error: 'User rejected the tool call.',
          approval: { id: 'int-1', decision: 'rejected' },
        })}
      />,
    )

    expect(getByText('Rejected')).toBeInTheDocument()
    expect(getByText('recipient')).toBeInTheDocument()
    expect(queryByText('Approve')).not.toBeInTheDocument()
    expect(queryByText('Reject')).not.toBeInTheDocument()
  })
})
