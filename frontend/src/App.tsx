import { useEffect, useState } from 'react'
import { fetchHealth } from './lib/api'
import { useAgentChat } from './hooks/useAgentChat'
import { Sidebar } from './components/Sidebar'
import { Transcript } from './components/Transcript'
import { Composer } from './components/Composer'
import { ErrorBanner } from './components/ErrorBanner'
import type { HealthInfo } from './types'

export default function App() {
  const [health, setHealth] = useState<HealthInfo | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const threadsEnabled = health?.threads ?? false

  const chat = useAgentChat(threadsEnabled)

  useEffect(() => {
    fetchHealth().then((info) => {
      setHealth(info)
      if (info?.title) document.title = info.title
    })
  }, [])

  useEffect(() => {
    if (threadsEnabled) chat.refreshThreads()
    // Only re-run when the feature flips on — refreshThreads itself is stable
    // across renders via useCallback, and re-running on every chat state
    // change would refetch the list constantly.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [threadsEnabled])

  return (
    <div className="app-shell">
      {threadsEnabled && (
        <Sidebar
          threads={chat.threads}
          activeThreadId={chat.threadId}
          onSelect={chat.openThread}
          onDelete={chat.removeThread}
          onNewThread={chat.startNewThread}
          hidden={!sidebarOpen}
        />
      )}

      <div className="main">
        <header className="header">
          {threadsEnabled && (
            <button
              type="button"
              className="ghost icon-button"
              title="Conversations"
              onClick={() => setSidebarOpen((v) => !v)}
            >
              ☰
            </button>
          )}
          <h1>{health?.title ?? 'Agent'}</h1>
          <div className="header-actions">
            <span className="status" data-state={chat.status}>
              {chat.statusLabel}
            </span>
            <button type="button" className="ghost" disabled={chat.running} onClick={chat.startNewThread}>
              Clear
            </button>
          </div>
        </header>

        <Transcript
          items={chat.transcript}
          running={chat.running}
          onDecideApproval={chat.resolveApproval}
        />

        {chat.error && <ErrorBanner message={chat.error} onRetry={chat.retry} />}

        <Composer
          running={chat.running}
          attachmentsEnabled={health?.attachments ?? false}
          onSend={chat.send}
          onStop={chat.stop}
          onUploadError={chat.reportError}
        />
      </div>
    </div>
  )
}
