import { useEffect, useRef, useState } from 'react'
import * as Collapsible from '@radix-ui/react-collapsible'
import { Check, ChevronDown, Lock, X } from 'lucide-react'
import { formatArgValue, formatValue, keyValueRows } from '../lib/api'
import type { ToolCallState } from '../hooks/useAgentChat'

export function ToolCard({
  tool,
  onDecideApproval,
  isLastApprovalInBatch = true,
}: {
  tool: ToolCallState
  onDecideApproval?: (decision: 'approved' | 'rejected') => void
  isLastApprovalInBatch?: boolean
}) {
  const isPendingApproval = tool.approval?.decision === 'pending'
  const isApproved = tool.approval?.decision === 'approved'
  const isRejected = tool.approval?.decision === 'rejected'
  const isResolved = isApproved || isRejected

  const [userExpanded, setUserExpanded] = useState<boolean | null>(null)
  const autoExpand = isPendingApproval || (tool.status === 'failed' && !isResolved)
  const expanded = userExpanded ?? autoExpand
  const [deciding, setDeciding] = useState(false)
  const cardRef = useRef<HTMLDivElement>(null)

  const prevPendingRef = useRef(isPendingApproval)

  useEffect(() => {
    if (isResolved) {
      setUserExpanded(false)
    }
  }, [isResolved])

  useEffect(() => {
    if (isPendingApproval && !prevPendingRef.current) {
      setUserExpanded(null)
    }
    prevPendingRef.current = isPendingApproval

    if (isPendingApproval) {
      const scrollCard = () => {
        cardRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
      }
      const timer1 = setTimeout(scrollCard, 60)
      const timer2 = setTimeout(scrollCard, 220)
      return () => {
        clearTimeout(timer1)
        clearTimeout(timer2)
      }
    }
  }, [isPendingApproval])

  const decide = (decision: 'approved' | 'rejected') => {
    setDeciding(true)
    setUserExpanded(false)
    onDecideApproval?.(decision)
  }

  const toggleExpand = () => {
    setUserExpanded(!expanded)
  }

  const statusLabel = isPendingApproval
    ? 'Awaiting approval'
    : isApproved
      ? 'Approved'
      : isRejected
        ? 'Rejected'
        : tool.status === 'running'
          ? 'Running…'
          : tool.status === 'failed'
            ? 'Failed'
            : 'Done'

  const argRows = keyValueRows(tool.args)

  return (
    <Collapsible.Root
      open={expanded}
      onOpenChange={toggleExpand}
      ref={cardRef}
      className={`tool ${isPendingApproval ? 'border-amber-500/80 shadow-amber-500/20' : ''}`}
      data-state={tool.status}
      data-pending-approval={isPendingApproval}
      data-approved={isApproved}
      data-rejected={isRejected}
    >
      <Collapsible.Trigger asChild>
        <button
          type="button"
          className="tool-summary flex items-center gap-2.5 w-full px-3.5 py-2.5 bg-transparent border-none text-left font-mono cursor-pointer"
          aria-expanded={expanded}
        >
          {tool.status === 'running' && !isPendingApproval ? (
            <span className="spinner" aria-hidden="true" />
          ) : (
            <span
              className="dot"
              data-state={isPendingApproval ? 'awaiting' : tool.approval?.decision ?? tool.status}
              aria-hidden="true"
            />
          )}
          <span className="tool-name font-mono text-xs font-semibold">{tool.name}</span>
          {argRows && argRows.length > 0 && (
            <span className="text-xs text-muted-foreground/80 truncate max-w-[220px] font-mono font-normal">
              ({argRows[0][0]}: {formatArgValue(argRows[0][1])})
            </span>
          )}
          <span
            className="tool-state ml-auto font-mono text-xs uppercase tracking-wider text-muted-foreground"
            data-state={isPendingApproval ? 'awaiting' : tool.approval?.decision ?? tool.status}
          >
            {statusLabel}
          </span>
          <ChevronDown
            className={`w-4 h-4 transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`}
            aria-hidden="true"
          />
        </button>
      </Collapsible.Trigger>

      <Collapsible.Content className="tool-body border-t border-border p-3.5 bg-background space-y-3">
        <div>
          <div className="tool-label text-[11px] font-mono font-bold uppercase tracking-wider text-muted-foreground mb-1.5">
            Arguments
          </div>
          <div className="tool-args-scroll max-h-[180px] overflow-y-auto pr-1">
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
        </div>

        {isPendingApproval && isLastApprovalInBatch && (
          <div className="approval-banner rounded-lg border border-amber-500/40 bg-amber-500/10 p-3.5 space-y-2">
            <div className="approval-banner-header flex items-center gap-2 text-amber-700 dark:text-amber-400 font-mono text-xs font-bold uppercase tracking-wider">
              <Lock className="w-4 h-4 text-amber-600 dark:text-amber-400" />
              <span>Approval Required</span>
            </div>
            <p className="approval-banner-desc text-xs text-muted-foreground leading-relaxed m-0">
              Review the proposed tool action above and choose whether to authorize execution.
            </p>
            <div className="approval-actions flex items-center gap-2.5 pt-1">
              <button
                type="button"
                className="btn-approve flex items-center justify-center gap-1.5 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold rounded-md shadow-sm transition-colors cursor-pointer disabled:opacity-50"
                disabled={deciding}
                onClick={() => decide('approved')}
              >
                <Check className="w-3.5 h-3.5" />
                <span>Approve</span>
              </button>
              <button
                type="button"
                className="btn-reject flex items-center justify-center gap-1.5 px-4 py-2 bg-red-600 hover:bg-red-500 text-white text-xs font-semibold rounded-md shadow-sm transition-colors cursor-pointer disabled:opacity-50"
                disabled={deciding}
                onClick={() => decide('rejected')}
              >
                <X className="w-3.5 h-3.5" />
                <span>Reject</span>
              </button>
            </div>
          </div>
        )}

        {isApproved && !isPendingApproval && (
          <div className="approval-status-banner approved flex items-center gap-1.5 p-2 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-600 dark:text-emerald-400 text-xs font-semibold">
            <Check className="w-3.5 h-3.5" />
            <span>Action authorized by user</span>
          </div>
        )}

        {isRejected && !isPendingApproval && (
          <div className="approval-status-banner rejected flex items-center gap-1.5 p-2 rounded bg-red-500/10 border border-red-500/30 text-red-600 dark:text-red-400 text-xs font-semibold">
            <X className="w-3.5 h-3.5" />
            <span>Action rejected by user</span>
          </div>
        )}

        {tool.status === 'failed' && !isRejected && (
          <LabelledBlock label="Error" text={tool.error ?? 'The tool failed.'} errorTone />
        )}

        {tool.status === 'completed' && (
          <LabelledBlock label="Result" text={formatValue(tool.result)} />
        )}
      </Collapsible.Content>
    </Collapsible.Root>
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
      <div className="tool-label text-[11px] font-mono font-bold uppercase tracking-wider text-muted-foreground mb-1.5">
        {label}
      </div>
      <pre className={errorTone ? 'tool-error text-red-500 bg-red-500/10 border-red-500/30 p-2.5 rounded text-xs' : undefined}>
        {text}
      </pre>
    </div>
  )
}
