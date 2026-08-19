// Regression coverage for ApprovalCard's design: human-readable action/args
// instead of a raw JSON dump while pending, and — after three separate bugs
// traced back to collapse/expand mechanics (native <details> animation, then
// a manually-toggled one) — a resolved decision renders as a small permanent
// summary that never expands or collapses, so there is no toggle state left
// to get stuck.

import { fireEvent, render } from '@testing-library/react'
import { describe, expect, test } from 'vitest'
import { ApprovalCard } from './ApprovalCard'
import type { ApprovalState } from '../hooks/useAgentChat'

function pending(id: string, value: unknown): ApprovalState {
  return { id, threadId: 't1', value, decision: 'pending' }
}

const hitlValue = {
  action_requests: [
    { name: 'send_alert', args: { recipient: 'Priya', message: 'The server is down.' } },
  ],
  review_configs: [{ action_name: 'send_alert', allowed_decisions: ['approve', 'reject'] }],
}

function Harness({ approval }: { approval: ApprovalState }) {
  return <ApprovalCard key={approval.id} approval={approval} onDecide={() => {}} />
}

describe('ApprovalCard', () => {
  test('a HumanInTheLoopMiddleware request renders human-readable, not raw JSON', () => {
    const { getByText, queryByText } = render(
      <Harness approval={pending('interrupt-1', hitlValue)} />,
    )

    expect(getByText('send_alert')).toBeInTheDocument()
    expect(getByText('recipient')).toBeInTheDocument()
    expect(getByText('Priya')).toBeInTheDocument()
    // The raw JSON fallback block must not also render alongside it.
    expect(queryByText(/action_requests/)).not.toBeInTheDocument()
  })

  test('a hand-rolled interrupt value without action_requests falls back to raw content', () => {
    const { getByText } = render(
      <Harness approval={pending('interrupt-2', { question: 'Approve sending the email?' })} />,
    )

    expect(getByText(/Approve sending the email\?/)).toBeInTheDocument()
  })

  test('a later interrupt in the same thread gets its own enabled buttons', () => {
    const { getByText, rerender } = render(<Harness approval={pending('interrupt-3', hitlValue)} />)

    fireEvent.click(getByText('Approve'))
    expect(getByText('Approve')).toBeDisabled()
    expect(getByText('Reject')).toBeDisabled()

    rerender(<Harness approval={pending('interrupt-4', hitlValue)} />)

    expect(getByText('Approve')).not.toBeDisabled()
    expect(getByText('Reject')).not.toBeDisabled()
  })

  test('resolving a decision replaces the card with a permanent summary, not a collapsible one', () => {
    const approval = pending('interrupt-5', hitlValue)
    const { getByText, queryByText, rerender } = render(
      <ApprovalCard approval={approval} onDecide={() => {}} />,
    )

    rerender(
      <ApprovalCard approval={{ ...approval, decision: 'approved' }} onDecide={() => {}} />,
    )

    expect(getByText('Approved — send_alert')).toBeInTheDocument()
    // No leftover Details/Approve/Reject affordances, and nothing to click to
    // bring them back — this state has no expand/collapse behavior at all.
    expect(queryByText('Approve')).not.toBeInTheDocument()
    expect(queryByText('Reject')).not.toBeInTheDocument()
    expect(queryByText('recipient')).not.toBeInTheDocument()
  })
})
