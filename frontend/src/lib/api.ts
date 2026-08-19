import type { AgentEvent, HealthInfo, HistoryTurn, ThreadInfo } from '../types'

// The API prefix is injected by the server into <body data-api-base="...">,
// because the UI is served from "/" (or a host app's mount prefix) while the
// API may live under a different path. Relative fetch() URLs would resolve
// against "/" and 404 — see src/agentstage/app.py's HTML rewrite.
const API_BASE = (document.body.dataset.apiBase || '/api').replace(/\/$/, '')

export function apiUrl(path: string): string {
  return `${API_BASE}/${path}`
}

export class ApiError extends Error {}

async function readErrorDetail(response: Response): Promise<string> {
  try {
    return await response.text()
  } catch {
    return ''
  }
}

export async function fetchHealth(): Promise<HealthInfo | null> {
  try {
    const response = await fetch(apiUrl('health'))
    if (!response.ok) return null
    return (await response.json()) as HealthInfo
  } catch {
    return null
  }
}

export async function uploadFile(file: File): Promise<{ id: string; filename: string }> {
  const form = new FormData()
  form.append('file', file)
  const response = await fetch(apiUrl('upload'), { method: 'POST', body: form })
  if (!response.ok) {
    throw new ApiError((await readErrorDetail(response)) || `Upload failed: ${response.status}`)
  }
  return response.json()
}

export async function fetchThreads(): Promise<ThreadInfo[]> {
  const response = await fetch(apiUrl('threads'))
  if (!response.ok) return []
  return response.json()
}

export async function fetchThreadHistory(threadId: string): Promise<HistoryTurn[]> {
  const response = await fetch(apiUrl(`threads/${encodeURIComponent(threadId)}/messages`))
  if (!response.ok) {
    throw new ApiError(`Server returned ${response.status}`)
  }
  return response.json()
}

export async function deleteThread(threadId: string): Promise<void> {
  const response = await fetch(apiUrl(`threads/${encodeURIComponent(threadId)}`), {
    method: 'DELETE',
  })
  if (!response.ok && response.status !== 204) {
    throw new ApiError(`Server returned ${response.status}`)
  }
}

export interface ChatBody {
  message: string
  thread_id: string | null
  attachment_ids: string[]
}

export interface ResumeBody {
  thread_id: string
  decision: 'approved' | 'rejected'
}

/**
 * POST /chat or /resume and yield each AgentEvent as it arrives over SSE.
 * fetch + ReadableStream is used rather than EventSource because EventSource
 * cannot issue a POST.
 *
 * Parses the SSE framing directly: frames are separated by a blank line, and
 * only `data:` lines carry payload. The `event:` field is redundant here
 * because the JSON body already includes its own `type`.
 */
export async function* streamChat(
  url: string,
  body: ChatBody | ResumeBody,
  signal: AbortSignal,
): AsyncGenerator<AgentEvent> {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })

  if (!response.ok || !response.body) {
    throw new ApiError(
      (await readErrorDetail(response)) || `Server returned ${response.status} ${response.statusText}`,
    )
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''

  const parseFrame = (frame: string): AgentEvent | null => {
    const payloads = frame
      .split('\n')
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trim())
    if (!payloads.length) return null // A comment frame (keepalive).
    return JSON.parse(payloads.join('\n')) as AgentEvent
  }

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += value

    let split: number
    while ((split = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, split)
      buffer = buffer.slice(split + 2)
      const event = parseFrame(frame)
      if (event) yield event
    }
  }
  if (buffer.trim()) {
    const event = parseFrame(buffer)
    if (event) yield event
  }
}

/** Renders any tool argument/result value as readable text for a <pre> block. */
export function formatValue(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'string') return value
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

/** A single value formatted for one row of a key/value list — never raw JSON syntax for a plain string. */
export function formatArgValue(value: unknown): string {
  return typeof value === 'string' ? value : formatValue(value)
}

/** Key/value rows for a plain args-shaped object, or null when it isn't one (caller falls back to formatValue). */
export function keyValueRows(value: unknown): [string, unknown][] | null {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return null
  const entries = Object.entries(value as Record<string, unknown>)
  return entries.length > 0 ? entries : null
}

export interface ActionRequest {
  name: string
  args?: Record<string, unknown>
  description?: string
}

/** HumanInTheLoopMiddleware's interrupt value; a hand-rolled interrupt() call
 * can send anything, so this is a best-effort read, not an assumed contract. */
export function actionRequestsOf(value: unknown): ActionRequest[] | null {
  if (
    typeof value === 'object' &&
    value !== null &&
    Array.isArray((value as { action_requests?: unknown }).action_requests)
  ) {
    return (value as { action_requests: ActionRequest[] }).action_requests
  }
  return null
}
