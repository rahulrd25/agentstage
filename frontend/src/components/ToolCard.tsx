import { formatValue } from '../lib/api'
import type { ToolCallState } from '../hooks/useAgentChat'

export function ToolCard({ tool }: { tool: ToolCallState }) {
  return (
    <details className="tool" data-state={tool.status} open={tool.status === 'failed'}>
      <summary>
        {tool.status === 'running' ? (
          <span className="spinner" aria-hidden="true" />
        ) : (
          <span className="dot" data-state={tool.status} aria-hidden="true" />
        )}
        <span className="tool-name">{tool.name}</span>
        <span className="tool-state">
          {tool.status === 'running' ? 'Running…' : tool.status === 'failed' ? 'Failed' : 'Done'}
        </span>
      </summary>
      <div className="tool-body">
        <LabelledBlock label="Arguments" text={formatValue(tool.args)} />
        {tool.status === 'failed' && (
          <LabelledBlock label="Error" text={tool.error ?? 'The tool failed.'} errorTone />
        )}
        {tool.status === 'completed' && <LabelledBlock label="Result" text={formatValue(tool.result)} />}
      </div>
    </details>
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
