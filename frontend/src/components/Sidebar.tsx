import type { ThreadInfo } from '../types'

export function Sidebar({
  threads,
  activeThreadId,
  onSelect,
  onDelete,
  onNewThread,
  hidden,
}: {
  threads: ThreadInfo[]
  activeThreadId: string | null
  onSelect: (id: string) => void
  onDelete: (id: string) => void
  onNewThread: () => void
  hidden?: boolean
}) {
  return (
    <nav className="sidebar" hidden={hidden}>
      <button type="button" className="ghost new-thread" onClick={onNewThread}>
        + New chat
      </button>
      <div className="thread-list">
        {threads.map((thread) => (
          <div
            key={thread.thread_id}
            className="thread-item"
            data-active={thread.thread_id === activeThreadId}
            onClick={() => onSelect(thread.thread_id)}
          >
            <span className="thread-item-title">{thread.title}</span>
            <button
              type="button"
              className="thread-item-delete"
              aria-label={`Delete ${thread.title}`}
              onClick={(e) => {
                e.stopPropagation()
                onDelete(thread.thread_id)
              }}
            >
              ×
            </button>
          </div>
        ))}
      </div>
    </nav>
  )
}
