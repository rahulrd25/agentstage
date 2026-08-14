// Regression coverage for SourcesList — citation url/title/cited_text are model
// output, untrusted the same as any tool result. Proves it builds real DOM
// nodes (text-node link labels, filtered hrefs) rather than assuming it from
// reading the source. Ported from the original tests/js/test_citations_render.mjs,
// which exercised the equivalent buildSources() in plain app.js before the React
// rewrite.

import { render } from '@testing-library/react'
import { describe, expect, test } from 'vitest'
import { SourcesList } from './SourcesList'

describe('SourcesList', () => {
  test('a citation with a safe url renders as a link with a text label', () => {
    const { container } = render(
      <SourcesList citations={[{ url: 'https://docs.example.com', title: 'Docs' }]} />,
    )

    const link = container.querySelector('a')
    expect(link).not.toBeNull()
    expect(link?.textContent).toBe('Docs')
    expect(link?.getAttribute('href')).toBe('https://docs.example.com')
  })

  test('a javascript: url is not rendered as a link', () => {
    const { container } = render(
      <SourcesList citations={[{ url: 'javascript:alert(1)', title: 'gotcha' }]} />,
    )

    expect(container.querySelector('a')).toBeNull()
    expect(container.textContent).toContain('gotcha')
  })

  test('cited_text with markup-looking content stays a text node', () => {
    const { container } = render(
      <SourcesList
        citations={[{ url: 'https://example.com', cited_text: '<img onerror=alert(1)>' }]}
      />,
    )

    expect(container.querySelector('img')).toBeNull()
    expect(container.querySelector('script')).toBeNull()
    expect(container.textContent).toContain('<img onerror=alert(1)>')
  })

  test('a citation with no url or title falls back to a numbered label', () => {
    const { container } = render(<SourcesList citations={[{ cited_text: 'just a quote' }]} />)

    expect(container.textContent).toContain('Source 1')
  })

  test('heading is singular for one citation, plural for many', () => {
    const one = render(<SourcesList citations={[{ url: 'https://a.example.com' }]} />)
    const many = render(
      <SourcesList
        citations={[{ url: 'https://a.example.com' }, { url: 'https://b.example.com' }]}
      />,
    )

    expect(one.container.querySelector('.sources-heading')?.textContent).toBe('Source')
    expect(many.container.querySelector('.sources-heading')?.textContent).toBe('Sources')
  })
})
