// Regression coverage for splitBlocks (via renderMarkdown): splitting on blank
// lines before extracting fences tears a code block containing a blank line
// into fragments, leaking the tail out as prose with a stray ``` visible.
// Ported from the original tests/js/test_markdown_split.mjs, which exercised
// this logic in plain app.js before the React rewrite.

import { render } from '@testing-library/react'
import { describe, expect, test } from 'vitest'
import { renderMarkdown } from './markdown'

describe('renderMarkdown', () => {
  test('a fenced block containing a blank line stays one code block', () => {
    const text = ['before', '', '```python', 'line one', '', 'line two', '```', '', 'after'].join(
      '\n',
    )

    const { container } = render(<>{renderMarkdown(text)}</>)

    const codeBlocks = container.querySelectorAll('pre > code')
    expect(codeBlocks).toHaveLength(1)
    expect(codeBlocks[0].textContent).toBe('line one\n\nline two')
    expect(codeBlocks[0].getAttribute('data-lang')).toBe('python')
  })

  test('an unterminated fence still renders as code, not leaked source', () => {
    const { container } = render(<>{renderMarkdown('```python\nprint(1)')}</>)

    const codeBlocks = container.querySelectorAll('pre > code')
    expect(codeBlocks).toHaveLength(1)
    expect(codeBlocks[0].textContent).toBe('print(1)')
  })

  test("no stray fence markers appear in any block's text", () => {
    const { container } = render(<>{renderMarkdown('```js\na\n\nb\n```\n\nnote')}</>)

    expect(container.textContent).not.toContain('```')
  })

  test('prose with no code fence is untouched', () => {
    const { container } = render(<>{renderMarkdown('hello\n\nworld')}</>)

    expect(container.querySelectorAll('pre')).toHaveLength(0)
    expect(container.textContent).toBe('helloworld')
  })
})
