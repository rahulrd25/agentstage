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
    const { getByText, queryByText } = render(
      <ToolCard
        tool={tool({ approval: { id: 'int-1', decision: 'pending' } })}
        onDecideApproval={onDecideApproval}
      />,
    )

    expect(getByText('Awaiting approval')).toBeInTheDocument()
    fireEvent.click(getByText('Reject'))

    expect(onDecideApproval).toHaveBeenCalledWith('rejected')
    // As soon as a decision is made, the card closes immediately
    expect(queryByText('Approve')).not.toBeInTheDocument()
  })

  test('a resolved approval shows the decision as the status, with no leftover Approve/Reject', () => {
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
    // Resolved cards default to collapsed; click header to expand and inspect body
    fireEvent.click(getByText('send_alert'))
    expect(getByText('recipient')).toBeInTheDocument()
    expect(queryByText('Approve')).not.toBeInTheDocument()
    expect(queryByText('Reject')).not.toBeInTheDocument()
  })
})
