import { useState } from 'react'
import { actionRequestsOf, formatArgValue, formatValue } from '../lib/api'
import type { ApprovalState } from '../hooks/useAgentChat'

// Only reached for a hand-rolled interrupt() call with no associated tool
// card to merge into (see examples/human_approval) — a HumanInTheLoopMiddleware
// interrupt merges into its ToolCard instead, since they describe one action,
// not two (see useAgentChat's interrupt_created handler).
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

  const actions = actionRequestsOf(approval.value)

  // Resolved: a small, permanent, single-line summary — never toggled,
  // never animated, so it cannot get stuck mid-transition the way the
  // collapsible versions of this card did. There is nothing left to decide,
  // so there is nothing that needs expanding back open.
  if (approval.decision !== 'pending') {
    const label = actions?.length ? actions.map((a) => a.name).join(', ') : null
    return (
      <div className="approval-summary" data-state={approval.decision}>
        <span className="approval-summary-icon" aria-hidden="true">
          {approval.decision === 'approved' ? '✓' : '✗'}
        </span>
        <span>
          {approval.decision === 'approved' ? 'Approved' : 'Rejected'}
          {label ? ` — ${label}` : ''}
        </span>
      </div>
    )
  }

  return (
    <div className="approval" data-state="pending">
      <div className="approval-heading">Approval requested</div>
      <div className="approval-body">
        {actions ? (
          <div className="approval-action-list">
            {actions.map((action, index) => (
              <div className="approval-action" key={`${action.name}-${index}`}>
                <div className="approval-action-name">{action.name}</div>
                {action.args && Object.keys(action.args).length > 0 && (
                  <dl className="kv-list">
                    {Object.entries(action.args).map(([key, value]) => (
                      <div className="kv-row" key={key}>
                        <dt>{key}</dt>
                        <dd>{formatArgValue(value)}</dd>
                      </div>
                    ))}
                  </dl>
                )}
              </div>
            ))}
          </div>
        ) : (
          <pre className="approval-raw">{formatValue(approval.value)}</pre>
        )}
        <div className="approval-actions">
          <button
            type="button"
            className="primary"
            disabled={deciding}
            onClick={() => decide('approved')}
          >
            Approve
          </button>
          <button
            type="button"
            className="ghost"
            disabled={deciding}
            onClick={() => decide('rejected')}
          >
            Reject
          </button>
        </div>
      </div>
    </div>
  )
}
