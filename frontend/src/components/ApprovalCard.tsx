import { useState } from 'react'
import { formatValue } from '../lib/api'
import type { ApprovalState } from '../hooks/useAgentChat'

export function ApprovalCard({
  approval,
  onDecide,
}: {
  approval: ApprovalState
  onDecide: (decision: 'approved' | 'rejected') => void
}) {
  const [deciding, setDeciding] = useState(false)

  const decide = (decision: 'approved' | 'rejected') => {
    setDeciding(true)
    onDecide(decision)
  }

  return (
    <div className="approval" data-state={approval.decision}>
      <div className="approval-heading">
        {approval.decision === 'pending'
          ? 'Approval requested'
          : approval.decision === 'approved'
            ? 'Approved'
            : 'Rejected'}
      </div>
      <div className="approval-body">
        <div className="tool-label">Details</div>
        <pre>{formatValue(approval.value)}</pre>
      </div>
      {approval.decision === 'pending' && (
        <div className="approval-actions">
          <button
            type="button"
            className="primary"
            disabled={deciding}
            onClick={() => decide('approved')}
          >
            Approve
          </button>
          <button type="button" className="ghost" disabled={deciding} onClick={() => decide('rejected')}>
            Reject
          </button>
        </div>
      )}
    </div>
  )
}
