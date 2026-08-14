// Mirrors agentstage's AgentEvent (src/agentstage/events/models.py) and the
// FastAPI request/response shapes it's paired with. Keep this file's shape in
// sync with the Python side by hand — there is no shared schema generator (yet).

export type EventType =
  | 'run_started'
  | 'run_completed'
  | 'run_failed'
  | 'message_started'
  | 'message_delta'
  | 'message_completed'
  | 'tool_call_started'
  | 'tool_call_delta'
  | 'tool_call_completed'
  | 'tool_call_failed'
  | 'interrupt_created'
  | 'progress_updated'
  | 'state_updated'

export interface AgentEvent {
  type: EventType
  run_id: string
  thread_id?: string
  node_name?: string
  message_id?: string
  tool_call_id?: string
  sequence?: number
  data?: Record<string, unknown>
  metadata?: Record<string, unknown>
}

export interface Citation {
  url?: string
  title?: string
  cited_text?: string
  start_index?: number
  end_index?: number
}

export interface UploadedAttachment {
  id: string
  filename: string
}

export interface ThreadInfo {
  thread_id: string
  title: string
  created_at: number
  updated_at: number
}

export interface HistoryToolCall {
  id?: string
  name?: string
  args?: Record<string, unknown>
}

export interface HistoryTurn {
  role: 'user' | 'assistant' | 'tool'
  text?: string
  tool_call_id?: string
  tool_calls?: HistoryToolCall[]
  citations?: Citation[]
}

export interface HealthInfo {
  status: string
  title: string
  attachments: boolean
  threads: boolean
}
