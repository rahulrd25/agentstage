import { useEffect, useRef } from 'react'
import { MessageBubble } from './MessageBubble'
import { ToolCard } from './ToolCard'
import { ApprovalCard } from './ApprovalCard'
import type { TranscriptItem } from '../hooks/useAgentChat'

export function Transcript({
  items,
  running,
  onDecideApproval,
}: {
  items: TranscriptItem[]
  running: boolean
  onDecideApproval: (decision: 'approved' | 'rejected') => void
}) {
  const ref = useRef<HTMLElement>(null)
  const wasPinnedRef = useRef(true)

  // Only autoscroll when already at the bottom, so scrolling up to read isn't
  // yanked away by an incoming token. Measured before the DOM updates
  // (useLayoutEffect-style timing would be ideal, but a plain effect after
  // render is sufficient here since scrollHeight already reflects new content
  // by the time this runs, and we want to react to that, not race it).
  useEffect(() => {
    const el = ref.current
    if (el && wasPinnedRef.current) {
      el.scrollTop = el.scrollHeight
    }
  })

  const handleScroll = () => {
    const el = ref.current
    if (!el) return
    wasPinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 60
  }

  const isEmpty = items.length === 0

  return (
    <main
      ref={ref}
      className="transcript"
      aria-live="polite"
      aria-busy={running}
      onScroll={handleScroll}
    >
      {isEmpty && (
        <div className="empty">
          <p className="empty-title">No messages yet</p>
          <p className="empty-body">Ask the agent something to get started.</p>
        </div>
      )}
      {items.map((item) =>
        item.kind === 'message' ? (
          <MessageBubble key={item.message.id} message={item.message} />
        ) : item.kind === 'tool' ? (
          <ToolCard key={item.tool.id} tool={item.tool} onDecideApproval={onDecideApproval} />
        ) : (
          <ApprovalCard
            key={item.approval.id}
            approval={item.approval}
            onDecide={onDecideApproval}
          />
        ),
      )}
    </main>
  )
}
