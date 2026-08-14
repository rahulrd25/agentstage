export function ErrorBanner({
  message,
  onRetry,
}: {
  message: string
  onRetry: () => void
}) {
  return (
    <div className="error" role="alert">
      <span>{message}</span>
      <button type="button" className="ghost" onClick={onRetry}>
        Retry
      </button>
    </div>
  )
}
