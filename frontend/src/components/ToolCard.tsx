import { useEffect, useState } from 'react'
import { formatArgValue, formatValue, keyValueRows } from '../lib/api'
import type { ToolCallState } from '../hooks/useAgentChat'

export function ToolCard({
  tool,
  onDecideApproval,
}: {
  tool: ToolCallState
  onDecideApproval?: (decision: 'approved' | 'rejected') => void
}) {
  const isPendingApproval = tool.approval?.decision === 'pending'
  const [expanded, setExpanded] = useState(tool.status === 'failed' || isPendingApproval)
  const [deciding, setDeciding] = useState(false)

  // Opens automatically the moment a decision is needed or the call fails —
  // once, not a native <details> toggle animating open/closed on every
  // re-render, which is what actually caused the approval card's repeated
  // sliver bug once this same card started carrying that responsibility too.
  useEffect(() => {
    if (tool.status === 'failed' || isPendingApproval) setExpanded(true)
  }, [tool.status, isPendingApproval])

  const decide = (decision: 'approved' | 'rejected') => {
    setDeciding(true)
    onDecideApproval?.(decision)
  }

  const statusLabel = isPendingApproval
    ? 'Awaiting approval'
    : tool.approval?.decision === 'approved'
      ? 'Approved'
      : tool.approval?.decision === 'rejected'
        ? 'Rejected'
        : tool.status === 'running'
          ? 'Running…'
          : tool.status === 'failed'
            ? 'Failed'
            : 'Done'

  const argRows = keyValueRows(tool.args)

  return (
    <div className="tool" data-state={tool.status} data-pending-approval={isPendingApproval}>
      <button
        type="button"
        className="tool-summary"
        aria-expanded={expanded}
        onClick={() => setExpanded((v) => !v)}
      >
        {tool.status === 'running' && !isPendingApproval ? (
          <span className="spinner" aria-hidden="true" />
        ) : (
          <span
            className="dot"
            data-state={isPendingApproval ? 'awaiting' : tool.status}
            aria-hidden="true"
          />
        )}
        <span className="tool-name">{tool.name}</span>
        <span className="tool-state">{statusLabel}</span>
      </button>
      {expanded && (
        <div className="tool-body">
          <div>
            <div className="tool-label">Arguments</div>
            {argRows ? (
              <dl className="kv-list">
                {argRows.map(([key, value]) => (
                  <div className="kv-row" key={key}>
                    <dt>{key}</dt>
                    <dd>{formatArgValue(value)}</dd>
                  </div>
                ))}
              </dl>
            ) : (
              <pre>{formatValue(tool.args)}</pre>
            )}
          </div>
          {isPendingApproval && (
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
          )}
          {tool.status === 'failed' && (
            <LabelledBlock label="Error" text={tool.error ?? 'The tool failed.'} errorTone />
          )}
          {tool.status === 'completed' && (
            <LabelledBlock label="Result" text={formatValue(tool.result)} />
          )}
        </div>
      )}
    </div>
  )
}

function LabelledBlock({
  label,
  text,
  errorTone,
}: {
  label: string
  text: string
  errorTone?: boolean
}) {
  return (
    <div>
      <div className="tool-label">{label}</div>
      <pre className={errorTone ? 'tool-error' : undefined}>{text}</pre>
    </div>
  )
}
