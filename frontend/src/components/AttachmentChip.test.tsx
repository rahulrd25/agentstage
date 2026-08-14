// Regression coverage for AttachmentChip — filenames come from the browser's
// file picker but are still attacker-controllable ("<script>.txt" is a valid
// filename), so this proves the chip is built from real DOM nodes rather than
// assuming it from reading the source. Ported from the original
// tests/js/test_attachment_chip.mjs, which exercised the equivalent
// buildAttachmentChip() in plain app.js before the React rewrite.

import { fireEvent, render } from '@testing-library/react'
import { describe, expect, test, vi } from 'vitest'
import { AttachmentChip } from './AttachmentChip'

describe('AttachmentChip', () => {
  test('a plain filename renders as its label', () => {
    const { container } = render(<AttachmentChip filename="report.pdf" />)

    expect(container.textContent).toContain('report.pdf')
  })

  test('a markup-looking filename stays a text node, not an element', () => {
    const filename = '<img src=x onerror=alert(1)>.txt'
    const { container } = render(<AttachmentChip filename={filename} />)

    expect(container.querySelector('img')).toBeNull()
    expect(container.querySelector('script')).toBeNull()
    expect(container.textContent).toContain(filename)
  })

  test('the uploading state shows a distinct label and no remove button', () => {
    const { container } = render(<AttachmentChip filename="f.txt" uploading />)

    expect(container.textContent).toContain('uploading')
    expect(container.querySelector('button')).toBeNull()
  })

  test('a removable chip carries a working remove button', () => {
    const onRemove = vi.fn()
    const { container } = render(<AttachmentChip filename="f.txt" onRemove={onRemove} />)

    const button = container.querySelector('button')
    expect(button).not.toBeNull()
    fireEvent.click(button as HTMLButtonElement)
    expect(onRemove).toHaveBeenCalledTimes(1)
  })

  test('no state means no data-state attribute', () => {
    const { container } = render(<AttachmentChip filename="f.txt" />)

    expect(container.querySelector('.attachment-chip')?.getAttribute('data-state')).toBeNull()
  })
})
