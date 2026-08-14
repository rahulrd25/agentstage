import type { Citation } from '../types'

// Citation url/title/cited_text are model output — untrusted the same as any
// tool result — so links are restricted to http(s), same discipline as the
// markdown renderer's inline links.
export function SourcesList({ citations }: { citations: Citation[] }) {
  return (
    <div className="sources">
      <div className="sources-heading">{citations.length === 1 ? 'Source' : 'Sources'}</div>
      <ol className="sources-list">
        {citations.map((citation, i) => {
          const safe = typeof citation.url === 'string' && /^https?:\/\//i.test(citation.url)
          const label = citation.title || citation.url || `Source ${i + 1}`
          return (
            <li key={i}>
              {safe ? (
                <a href={citation.url} rel="noopener noreferrer" target="_blank">
                  {label}
                </a>
              ) : (
                label
              )}
              {citation.cited_text && (
                <span className="sources-quote"> — &quot;{citation.cited_text}&quot;</span>
              )}
            </li>
          )
        })}
      </ol>
    </div>
  )
}
