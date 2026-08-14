// Regression coverage for the error banner's visibility contract: it must not
// render until there is actually an error, and must appear (with a working
// retry) once one occurs. Replaces the old server-rendered-HTML check
// (tests/integration/test_http.py::test_the_error_banner_starts_hidden_in_the_markup),
// which asserted a static `hidden` attribute in index.html — no longer
// meaningful now that the banner is a conditionally-mounted React component
// (see App.tsx) rather than always-present markup hidden by CSS.

import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'
import App from './App'

function mockFetch(chatOk: boolean) {
  globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input)
    if (url.endsWith('/api/health')) {
      return new Response(JSON.stringify({ status: 'ok', title: 'Agent', attachments: false }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    }
    if (url.endsWith('/api/chat')) {
      if (!chatOk) return new Response('boom', { status: 500 })
      return new Response('', { status: 200, headers: { 'Content-Type': 'text/event-stream' } })
    }
    return new Response('[]', { status: 200, headers: { 'Content-Type': 'application/json' } })
  }) as typeof fetch
}

describe('App error banner', () => {
  beforeEach(() => {
    document.body.dataset.apiBase = '/api'
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  test('no error banner renders on initial load', async () => {
    mockFetch(true)
    render(<App />)

    await waitFor(() => expect(screen.getByText('Agent')).toBeInTheDocument())
    expect(screen.queryByRole('alert')).toBeNull()
  })

  test('a failed send surfaces the error banner with a working retry', async () => {
    mockFetch(false)
    const user = userEvent.setup()
    render(<App />)

    await waitFor(() => expect(screen.getByText('Agent')).toBeInTheDocument())

    await user.type(screen.getByPlaceholderText('Send a message…'), 'hello')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    const banner = await screen.findByRole('alert')
    expect(banner).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Retry' }))
    // Retrying re-fires the same failing request; the banner stays visible
    // rather than disappearing silently.
    expect(await screen.findByRole('alert')).toBeInTheDocument()
  })
})
