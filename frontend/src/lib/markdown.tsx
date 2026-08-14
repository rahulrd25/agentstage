// Ported from the original app.js markdown renderer. Security property carried
// over unchanged: every string here is untrusted (model output, tool results),
// so this returns React elements built from data, never dangerouslySetInnerHTML
// or any other markup-injection path. React's own JSX escaping is the second
// line of defense; the real guarantee is that this code never constructs HTML
// strings in the first place.
//
// A deliberately small markdown subset: fenced code, inline code, bold, italic,
// links, lists, paragraphs.

import type { ReactNode } from 'react'

type Block = { type: 'code'; lang: string; text: string } | { type: 'prose'; text: string }

// Fences are extracted BEFORE splitting on blank lines. Splitting first tears a
// code block that contains a blank line into fragments — the opening fence
// renders as code and the remainder leaks out as prose with a stray ``` visible.
// (This was a real bug in the original implementation; see tests/js/test_markdown_split.mjs
// in the Python package for the regression this guards against.)
function splitBlocks(text: string): Block[] {
  const lines = text.split('\n')
  const blocks: Block[] = []
  let prose: string[] = []
  let code: { lang: string; lines: string[] } | null = null

  const flushProse = () => {
    if (prose.length) blocks.push({ type: 'prose', text: prose.join('\n') })
    prose = []
  }

  for (const line of lines) {
    const fence = line.match(/^\s*```(\w*)\s*$/)
    if (code) {
      if (fence) {
        blocks.push({ type: 'code', lang: code.lang, text: code.lines.join('\n') })
        code = null
      } else {
        code.lines.push(line)
      }
      continue
    }
    if (fence) {
      flushProse()
      code = { lang: fence[1] ?? '', lines: [] }
      continue
    }
    prose.push(line)
  }

  // An unterminated fence is the normal mid-stream state: the closing ``` has
  // not arrived yet, so render what we have as code rather than dumping the
  // raw source.
  if (code) blocks.push({ type: 'code', lang: code.lang, text: code.lines.join('\n') })
  flushProse()
  return blocks
}

const INLINE = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*]+\*)|(\[[^\]]+\]\([^)\s]+\))/

// Inline spans, tokenized in one pass. Link hrefs are restricted to http(s) so a
// javascript: URL cannot be smuggled through a rendered answer.
function renderInline(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = []
  let rest = text
  let i = 0

  while (rest) {
    const match = rest.match(INLINE)
    if (!match || match.index === undefined) {
      nodes.push(rest)
      break
    }
    if (match.index > 0) {
      nodes.push(rest.slice(0, match.index))
    }
    const token = match[0]
    const key = `${keyPrefix}-${i++}`
    if (token.startsWith('`')) {
      nodes.push(<code key={key}>{token.slice(1, -1)}</code>)
    } else if (token.startsWith('**')) {
      nodes.push(<strong key={key}>{token.slice(2, -2)}</strong>)
    } else if (token.startsWith('*')) {
      nodes.push(<em key={key}>{token.slice(1, -1)}</em>)
    } else {
      const link = token.match(/^\[([^\]]+)\]\(([^)\s]+)\)$/)
      const href = link?.[2] ?? ''
      const label = link?.[1] ?? token
      const safe = /^https?:\/\//i.test(href)
      if (safe) {
        nodes.push(
          <a key={key} href={href} rel="noopener noreferrer" target="_blank">
            {label}
          </a>,
        )
      } else {
        // Not a safe scheme: show the markdown source rather than a live link.
        nodes.push(token)
      }
    }
    rest = rest.slice(match.index + token.length)
  }
  return nodes
}

function renderProse(text: string, keyPrefix: string): ReactNode[] {
  const out: ReactNode[] = []
  let blockIndex = 0

  for (const raw of text.split(/\n{2,}/)) {
    const block = raw.replace(/\s+$/, '')
    if (!block) continue
    const key = `${keyPrefix}-p${blockIndex++}`

    const lines = block.split('\n')
    if (lines.every((l) => /^\s*[-*]\s+/.test(l))) {
      out.push(
        <ul key={key}>
          {lines.map((line, i) => (
            <li key={i}>{renderInline(line.replace(/^\s*[-*]\s+/, ''), `${key}-li${i}`)}</li>
          ))}
        </ul>,
      )
      continue
    }
    if (lines.every((l) => /^\s*\d+[.)]\s+/.test(l))) {
      out.push(
        <ol key={key}>
          {lines.map((line, i) => (
            <li key={i}>{renderInline(line.replace(/^\s*\d+[.)]\s+/, ''), `${key}-li${i}`)}</li>
          ))}
        </ol>,
      )
      continue
    }

    out.push(<p key={key}>{renderInline(block.replace(/\n/g, ' '), key)}</p>)
  }
  return out
}

/** Renders a markdown string as React nodes. Pure — no DOM access, safe to call
 * during render on every keystroke of a streaming message. */
export function renderMarkdown(text: string): ReactNode {
  const blocks = splitBlocks(text)
  return (
    <>
      {blocks.map((block, i) => {
        if (block.type === 'code') {
          return (
            <pre key={i}>
              <code data-lang={block.lang || undefined}>{block.text}</code>
            </pre>
          )
        }
        return <span key={i}>{renderProse(block.text, `b${i}`)}</span>
      })}
    </>
  )
}
