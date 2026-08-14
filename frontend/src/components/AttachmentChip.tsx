// Filenames are client-supplied (the browser's own file picker, but still
// attacker-controllable — nothing stops "<script>.txt" as a filename) so this
// renders through JSX text content only, same discipline as every other
// untrusted string in this app.
export function AttachmentChip({
  filename,
  uploading,
  onRemove,
}: {
  filename: string
  uploading?: boolean
  onRemove?: () => void
}) {
  return (
    <span className="attachment-chip" data-state={uploading ? 'uploading' : undefined}>
      📎 {filename}
      {uploading && ' (uploading…)'}
      {onRemove && (
        <button
          type="button"
          className="attachment-chip-remove"
          aria-label={`Remove ${filename}`}
          onClick={onRemove}
        >
          ×
        </button>
      )}
    </span>
  )
}
