import { useRef, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'
import { uploadFile, ApiError } from '../lib/api'
import { AttachmentChip } from './AttachmentChip'
import type { UploadedAttachment } from '../types'

interface PendingUpload extends UploadedAttachment {
  uploading?: boolean
}

export function Composer({
  running,
  attachmentsEnabled,
  onSend,
  onStop,
  onUploadError,
}: {
  running: boolean
  attachmentsEnabled: boolean
  onSend: (text: string, attachments: UploadedAttachment[]) => void
  onStop: () => void
  onUploadError: (message: string) => void
}) {
  const [text, setText] = useState('')
  const [pending, setPending] = useState<PendingUpload[]>([])
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const resize = () => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`
  }

  const submit = (e: FormEvent) => {
    e.preventDefault()
    const trimmed = text.trim()
    if (!trimmed || running) return
    const attachments = pending.filter((p) => !p.uploading)
    setText('')
    setPending([])
    onSend(trimmed, attachments)
    requestAnimationFrame(resize)
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter sends; Shift+Enter inserts a newline.
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit(e as unknown as FormEvent)
    }
  }

  const onFileChange = async () => {
    const file = fileInputRef.current?.files?.[0]
    if (fileInputRef.current) fileInputRef.current.value = '' // allow re-selecting the same file
    if (!file) return

    const localId = `uploading-${Date.now()}`
    setPending((prev) => [...prev, { id: localId, filename: file.name, uploading: true }])
    try {
      const uploaded = await uploadFile(file)
      setPending((prev) =>
        prev.map((p) => (p.id === localId ? { id: uploaded.id, filename: uploaded.filename } : p)),
      )
    } catch (err) {
      setPending((prev) => prev.filter((p) => p.id !== localId))
      onUploadError(err instanceof ApiError ? err.message : 'The file could not be uploaded.')
    }
  }

  const removePending = (id: string) => {
    setPending((prev) => prev.filter((p) => p.id !== id))
  }

  return (
    <div className="composer-bar">
      {pending.length > 0 && (
        <div className="pending-attachments">
          {pending.map((p) => (
            <AttachmentChip
              key={p.id}
              filename={p.filename}
              uploading={p.uploading}
              onRemove={p.uploading ? undefined : () => removePending(p.id)}
            />
          ))}
        </div>
      )}
      <form className="composer" onSubmit={submit}>
        {attachmentsEnabled && (
          <>
            <button
              type="button"
              className="ghost"
              title="Attach a file"
              onClick={() => fileInputRef.current?.click()}
            >
              📎
            </button>
            <input ref={fileInputRef} type="file" hidden onChange={onFileChange} />
          </>
        )}
        <textarea
          ref={textareaRef}
          className="input"
          rows={1}
          placeholder="Send a message…"
          autoComplete="off"
          value={text}
          disabled={running}
          onChange={(e) => {
            setText(e.target.value)
            resize()
          }}
          onKeyDown={onKeyDown}
        />
        <button type="submit" className="primary" disabled={running}>
          Send
        </button>
        {running && (
          <button type="button" className="ghost" onClick={onStop}>
            Stop
          </button>
        )}
      </form>
    </div>
  )
}
