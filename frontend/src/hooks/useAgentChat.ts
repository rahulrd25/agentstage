import { useCallback, useRef, useState } from 'react'
import {
  ApiError,
  apiUrl,
  deleteThread as apiDeleteThread,
  fetchThreadHistory,
  fetchThreads,
  streamChat,
} from '../lib/api'
import type { AgentEvent, Citation, HistoryTurn, ThreadInfo, UploadedAttachment } from '../types'

export interface ToolCallState {
  id: string
  name: string
  args: unknown
  status: 'running' | 'completed' | 'failed'
  result?: unknown
  error?: string
}

export interface MessageState {
  id: string
  role: 'user' | 'assistant'
  text: string
  streaming: boolean
  attachments?: UploadedAttachment[]
  citations?: Citation[]
}

export interface ApprovalState {
  threadId: string
  value: unknown
  decision: 'pending' | 'approved' | 'rejected'
}

// A transcript item is either a chat message or an interleaved tool call —
// order matters, so both live in one array rather than two separate lists the
// UI would have to re-merge for display.
export type TranscriptItem =
  | { kind: 'message'; message: MessageState }
  | { kind: 'tool'; tool: ToolCallState }

type Status = 'idle' | 'running' | 'error' | 'stopped'

export function useAgentChat(threadsEnabled: boolean) {
  const [threadId, setThreadId] = useState<string | null>(null)
  const [transcript, setTranscript] = useState<TranscriptItem[]>([])
  const [approval, setApproval] = useState<ApprovalState | null>(null)
  const [status, setStatus] = useState<Status>('idle')
  const [statusLabel, setStatusLabel] = useState('Ready')
  const [error, setError] = useState<string | null>(null)
  const [threads, setThreads] = useState<ThreadInfo[]>([])
  const [running, setRunning] = useState(false)

  // Refs, not state: these are read inside the streaming loop's closures and
  // must reflect the latest value without re-subscribing the effect/callback
  // to state changes — React state updates inside a fast token loop would
  // otherwise be stale reads.
  const controllerRef = useRef<AbortController | null>(null)
  const lastMessageRef = useRef<{ text: string; attachmentIds: string[] } | null>(null)
  const pendingInterruptRef = useRef(false)

  const refreshThreads = useCallback(async () => {
    if (!threadsEnabled) return
    setThreads(await fetchThreads())
  }, [threadsEnabled])

  const resetTranscript = useCallback(() => {
    setTranscript([])
    setError(null)
    setStatus('idle')
    setStatusLabel('Ready')
  }, [])

  // Applies one AgentEvent to transcript/approval/status state. A message that
  // only produced tool calls closes with empty text and is dropped entirely —
  // an empty bubble would read as a rendering bug, not an intentional silence.
  const applyEvent = useCallback((event: AgentEvent) => {
    switch (event.type) {
      case 'run_started':
        if (event.thread_id) setThreadId(event.thread_id)
        setStatus('running')
        setStatusLabel('Thinking…')
        break

      case 'message_started':
        setTranscript((prev) => [
          ...prev,
          {
            kind: 'message',
            message: { id: event.message_id!, role: 'assistant', text: '', streaming: true },
          },
        ])
        break

      case 'message_delta': {
        const delta = (event.data?.text as string) ?? ''
        setTranscript((prev) =>
          prev.map((item) =>
            item.kind === 'message' && item.message.id === event.message_id
              ? { ...item, message: { ...item.message, text: item.message.text + delta } }
              : item,
          ),
        )
        break
      }

      case 'message_completed': {
        const finalText = event.data?.text as string | undefined
        const citations = event.data?.citations as Citation[] | undefined
        setTranscript((prev) => {
          const next = prev.map((item) =>
            item.kind === 'message' && item.message.id === event.message_id
              ? {
                  ...item,
                  message: {
                    ...item.message,
                    text: finalText ?? item.message.text,
                    streaming: false,
                    citations,
                  },
                }
              : item,
          )
          // Drop an assistant turn that produced no text (tool-call-only turn).
          return next.filter(
            (item) =>
              !(item.kind === 'message' && item.message.id === event.message_id && !item.message.text),
          )
        })
        break
      }

      case 'tool_call_started': {
        const data = event.data ?? {}
        setTranscript((prev) => [
          ...prev,
          {
            kind: 'tool',
            tool: {
              id: event.tool_call_id!,
              name: (data.name as string) ?? 'tool',
              args: data.args,
              status: 'running',
            },
          },
        ])
        setStatusLabel(`Running ${(data.name as string) ?? 'tool'}…`)
        break
      }

      case 'tool_call_completed':
      case 'tool_call_failed': {
        const failed = event.type === 'tool_call_failed'
        const data = event.data ?? {}
        setTranscript((prev) =>
          prev.map((item) =>
            item.kind === 'tool' && item.tool.id === event.tool_call_id
              ? {
                  ...item,
                  tool: {
                    ...item.tool,
                    status: failed ? 'failed' : 'completed',
                    result: data.result,
                    error: data.error as string | undefined,
                  },
                }
              : item,
          ),
        )
        break
      }

      case 'interrupt_created':
        pendingInterruptRef.current = true
        setApproval({
          threadId: event.thread_id!,
          value: event.data?.value,
          decision: 'pending',
        })
        setStatus('running')
        setStatusLabel('Waiting for approval')
        break

      case 'run_failed':
        setError((event.data?.error as string) || 'The agent failed.')
        setStatus('error')
        setStatusLabel('Failed')
        break

      case 'run_completed':
        setStatus('idle')
        setStatusLabel('Ready')
        break

      default:
        break // Unknown future event types are ignored, not fatal.
    }
  }, [])

  const runStream = useCallback(
    async (url: string, body: Parameters<typeof streamChat>[1]): Promise<boolean> => {
      setRunning(true)
      const controller = new AbortController()
      controllerRef.current = controller
      let ok = true

      try {
        for await (const event of streamChat(url, body, controller.signal)) {
          applyEvent(event)
        }
      } catch (err) {
        ok = false
        if (err instanceof DOMException && err.name === 'AbortError') {
          setStatus('stopped')
          setStatusLabel('Stopped')
        } else {
          setError(err instanceof Error ? err.message : 'The connection failed.')
          setStatus('error')
          setStatusLabel('Failed')
        }
      } finally {
        // Any message left mid-stream when the connection ends would keep
        // streaming=true (and its caret) forever.
        setTranscript((prev) =>
          prev.map((item) =>
            item.kind === 'message' && item.message.streaming
              ? { ...item, message: { ...item.message, streaming: false } }
              : item,
          ),
        )
        // A run that paused on an interrupt ends its SSE stream without a
        // run_completed — expected, not a dropped connection — so the composer
        // must stay blocked until the approval card is answered.
        if (!pendingInterruptRef.current) setRunning(false)
        controllerRef.current = null
      }
      return ok
    },
    [applyEvent],
  )

  // The actual /chat call, independent of whether a new bubble is added — send()
  // adds one (a fresh user message), retry() does not (the failed message's
  // bubble is already in the transcript from the attempt that failed).
  const submit = useCallback(
    async (text: string, attachmentIds: string[]) => {
      setError(null)
      const wasNewThread = threadId === null
      lastMessageRef.current = { text, attachmentIds }

      await runStream(apiUrl('chat'), {
        message: text,
        thread_id: threadId,
        attachment_ids: attachmentIds,
      })
      if (wasNewThread || threadsEnabled) refreshThreads()
    },
    [threadId, runStream, refreshThreads, threadsEnabled],
  )

  const send = useCallback(
    async (text: string, attachments: UploadedAttachment[]) => {
      setTranscript((prev) => [
        ...prev,
        {
          kind: 'message',
          message: { id: `local-${Date.now()}`, role: 'user', text, streaming: false, attachments },
        },
      ])
      await submit(text, attachments.map((a) => a.id))
    },
    [submit],
  )

  const resolveApproval = useCallback(
    async (decision: 'approved' | 'rejected') => {
      if (!approval) return
      setError(null)
      pendingInterruptRef.current = false
      const targetThreadId = approval.threadId
      const ok = await runStream(apiUrl('resume'), { thread_id: targetThreadId, decision })
      setApproval((prev) => (prev ? { ...prev, decision: ok ? decision : prev.decision } : prev))
      refreshThreads() // Bumps the thread's position to most-recently-used.
    },
    [approval, runStream, refreshThreads],
  )

  const retry = useCallback(() => {
    if (!lastMessageRef.current || running) return
    // No new bubble: the message that failed is already in the transcript.
    void submit(lastMessageRef.current.text, lastMessageRef.current.attachmentIds)
  }, [running, submit])

  const stop = useCallback(() => {
    controllerRef.current?.abort()
  }, [])

  const startNewThread = useCallback(() => {
    if (running) return
    resetTranscript()
    setThreadId(null)
    setApproval(null)
  }, [running, resetTranscript])

  const openThread = useCallback(
    async (id: string) => {
      if (running || id === threadId) return
      resetTranscript()
      setThreadId(id)
      setApproval(null)
      try {
        const history = await fetchThreadHistory(id)
        setTranscript(historyToTranscript(history))
      } catch (err) {
        setError(err instanceof ApiError ? err.message : 'Could not load that conversation.')
      }
    },
    [running, threadId, resetTranscript],
  )

  const removeThread = useCallback(
    async (id: string) => {
      try {
        await apiDeleteThread(id)
        if (id === threadId) {
          resetTranscript()
          setThreadId(null)
        }
        await refreshThreads()
      } catch (err) {
        setError(err instanceof ApiError ? err.message : 'Could not delete that conversation.')
      }
    },
    [threadId, resetTranscript, refreshThreads],
  )

  return {
    threadId,
    transcript,
    approval,
    status,
    statusLabel,
    error,
    running,
    threads,
    send,
    resolveApproval,
    retry,
    stop,
    startNewThread,
    openThread,
    removeThread,
    refreshThreads,
    clearError: () => setError(null),
    // Exposed so components outside the streaming loop (e.g. a failed upload)
    // can surface through the same single error banner, rather than each
    // owning a separate error UI.
    reportError: setError,
  }
}

// A past turn is rendered once, fully formed — no streaming, no caret —
// reusing the same TranscriptItem shape as the live path so history and a live
// run render identically.
function historyToTranscript(history: HistoryTurn[]): TranscriptItem[] {
  const items: TranscriptItem[] = []
  for (const turn of history) {
    if (turn.role === 'user') {
      items.push({
        kind: 'message',
        message: { id: crypto.randomUUID(), role: 'user', text: turn.text ?? '', streaming: false },
      })
      continue
    }
    if (turn.role === 'tool') {
      items.push({
        kind: 'tool',
        tool: {
          id: turn.tool_call_id ?? crypto.randomUUID(),
          name: 'tool result',
          args: undefined,
          status: 'completed',
          result: turn.text,
        },
      })
      continue
    }
    // assistant
    for (const call of turn.tool_calls ?? []) {
      items.push({
        kind: 'tool',
        tool: {
          id: call.id ?? crypto.randomUUID(),
          name: call.name ?? 'tool',
          args: call.args,
          status: 'completed',
        },
      })
    }
    if (turn.text) {
      items.push({
        kind: 'message',
        message: {
          id: crypto.randomUUID(),
          role: 'assistant',
          text: turn.text,
          streaming: false,
          citations: turn.citations,
        },
      })
    }
  }
  return items
}
