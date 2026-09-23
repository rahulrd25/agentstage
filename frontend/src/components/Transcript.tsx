import { useEffect, useRef } from 'react'
import * as ScrollArea from '@radix-ui/react-scroll-area'
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
  const viewportRef = useRef<HTMLDivElement>(null)

  const hasPendingApproval = items.some(
    (item) =>
      (item.kind === 'tool' && item.tool.approval?.decision === 'pending') ||
      (item.kind === 'approval' && item.approval.decision === 'pending'),
  )

  useEffect(() => {
    const el = viewportRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
  }, [items, hasPendingApproval])

  const isEmpty = items.length === 0

  return (
    <ScrollArea.Root className="flex-1 w-full overflow-hidden bg-background" aria-busy={running}>
      <ScrollArea.Viewport
        ref={viewportRef}
        className="w-full h-full p-6 md:px-12 flex flex-col items-center gap-4 space-y-4"
      >
        <div className="w-full max-w-[46rem] mx-auto space-y-4 pb-16">
          {isEmpty && (
            <div className="text-center py-24 text-muted-foreground max-w-xs mx-auto">
              <p className="font-sans font-bold text-base text-foreground mb-1">No messages yet</p>
              <p className="text-sm">Ask the agent something to get started.</p>
            </div>
          )}
          {items.map((item, index) => {
            if (item.kind === 'message') {
              return <MessageBubble key={item.message.id} message={item.message} />
            }
            if (item.kind === 'tool') {
              const isPending = item.tool.approval?.decision === 'pending'
              const approvalId = item.tool.approval?.id
              const isLastApprovalInBatch =
                !isPending ||
                !items
                  .slice(index + 1)
                  .some(
                    (nextItem) =>
                      nextItem.kind === 'tool' &&
                      nextItem.tool.approval?.decision === 'pending' &&
                      nextItem.tool.approval?.id === approvalId,
                  )
              return (
                <ToolCard
                  key={item.tool.id}
                  tool={item.tool}
                  onDecideApproval={onDecideApproval}
                  isLastApprovalInBatch={isLastApprovalInBatch}
                />
              )
            }
            return (
              <ApprovalCard
                key={item.approval.id}
                approval={item.approval}
                onDecide={onDecideApproval}
              />
            )
          })}
        </div>
      </ScrollArea.Viewport>
      <ScrollArea.Scrollbar
        className="flex select-none touch-none p-0.5 bg-transparent transition-colors w-2.5"
        orientation="vertical"
      >
        <ScrollArea.Thumb className="flex-1 bg-border rounded-[10px] relative" />
      </ScrollArea.Scrollbar>
    </ScrollArea.Root>
  )
}
