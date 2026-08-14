import { renderMarkdown } from '../lib/markdown'
import { AttachmentChip } from './AttachmentChip'
import { SourcesList } from './SourcesList'
import type { MessageState } from '../hooks/useAgentChat'

export function MessageBubble({ message }: { message: MessageState }) {
  return (
    <div className="msg" data-role={message.role}>
      <div className={`bubble${message.streaming ? ' caret' : ''}`}>
        {message.role === 'user' ? message.text : renderMarkdown(message.text)}
      </div>
      {message.role === 'user' && message.attachments && message.attachments.length > 0 && (
        <div className="attachment-chips">
          {message.attachments.map((a) => (
            <AttachmentChip key={a.id} filename={a.filename} />
          ))}
        </div>
      )}
      {message.role === 'assistant' && message.citations && message.citations.length > 0 && (
        <SourcesList citations={message.citations} />
      )}
    </div>
  )
}
