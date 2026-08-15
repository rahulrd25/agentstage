"""Automated smoke test driving app.py over real HTTP, no browser involved.

Starts app.py as a subprocess against a real OpenAI model, then exercises
every AgentApp feature enabled in that app (streaming chat, tool-call
visibility, human-in-the-loop approval, thread history) through its actual
HTTP surface, and reports pass/fail per feature.

    source .venv/bin/activate
    python harness.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

HOST = "127.0.0.1"
PORT = 8010
BASE = f"http://{HOST}:{PORT}"
APP_DIR = Path(__file__).resolve().parent

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))


def parse_sse(response: httpx.Response) -> list[dict]:
    """Parse an SSE stream into a list of {"event": ..., "data": ...} dicts."""
    events = []
    current_event = None
    current_data = None
    for line in response.iter_lines():
        if line.startswith("event:"):
            current_event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            current_data = line.split(":", 1)[1].strip()
        elif line == "" and current_event is not None:
            events.append({"event": current_event, "data": json.loads(current_data)})
            current_event, current_data = None, None
    return events


def wait_for_server(proc: subprocess.Popen, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Server process exited early with code {proc.returncode}")
        try:
            r = httpx.get(f"{BASE}/api/health", timeout=2)
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError("Server did not become ready in time")


def main() -> None:
    env = dict(os.environ)
    env["SMOKE_TEST_HOST"] = HOST
    env["SMOKE_TEST_PORT"] = str(PORT)

    log_path = APP_DIR / "server.log"
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, str(APP_DIR / "app.py")],
        cwd=APP_DIR,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    try:
        wait_for_server(proc)
        record("server starts and /api/health responds", True)

        client = httpx.Client(timeout=60)

        # --- UI is actually served from the wheel ---
        r = client.get(f"{BASE}/")
        record("GET / serves the UI", r.status_code == 200 and "<html" in r.text.lower())

        # --- Feature 1: streaming chat, no tool call ---
        with client.stream(
            "POST",
            f"{BASE}/api/chat",
            json={"message": "Reply with exactly the word: pong", "thread_id": "smoke-chat"},
        ) as r:
            events = parse_sse(r)
        types = [e["event"] for e in events]
        deltas = [e for e in events if e["event"] == "message_delta"]
        record(
            "chat(streaming=True): message streams and completes",
            "run_completed" in types and len(deltas) > 0,
            f"event types seen: {sorted(set(types))}",
        )

        # --- Feature 2: tool call visibility (no approval needed) ---
        with client.stream(
            "POST",
            f"{BASE}/api/chat",
            json={
                "message": "What is the weather in Paris right now? Use the tool.",
                "thread_id": "smoke-weather",
            },
        ) as r:
            events = parse_sse(r)
        types = [e["event"] for e in events]
        tool_started = [e for e in events if e["event"] == "tool_call_started"]
        record(
            "tool_calls(visible=True): get_weather call is visible and run completes",
            "tool_call_started" in types
            and "tool_call_completed" in types
            and "run_completed" in types
            and any(e["data"].get("data", {}).get("name") == "get_weather" for e in tool_started),
            f"event types seen: {sorted(set(types))}",
        )

        # --- Feature 3: human-in-the-loop interrupt is created ---
        with client.stream(
            "POST",
            f"{BASE}/api/chat",
            json={
                "message": "Send an alert to alice@example.com saying the build passed.",
                "thread_id": "smoke-approval",
            },
        ) as r:
            events = parse_sse(r)
        types = [e["event"] for e in events]
        interrupt_events = [e for e in events if e["event"] == "interrupt_created"]
        record(
            "human_approval(enabled=True): send_alert pauses on an interrupt",
            "interrupt_created" in types and len(interrupt_events) == 1,
            f"event types seen: {sorted(set(types))}",
        )

        # --- Feature 3b: resuming that interrupt with "approved" ---
        r = client.post(
            f"{BASE}/api/resume",
            json={"thread_id": "smoke-approval", "decision": "approved"},
        )
        if r.status_code != 200:
            record(
                "resume(decision=approved) completes the run",
                False,
                f"HTTP {r.status_code}: {r.text[:300]}",
            )
        else:
            resume_events = parse_sse(r)
            resume_types = [e["event"] for e in resume_events]
            failed = [e for e in resume_events if e["event"] == "run_failed"]
            record(
                "resume(decision=approved) completes the run",
                "run_completed" in resume_types and not failed,
                (
                    f"event types seen: {sorted(set(resume_types))}"
                    + (f"; run_failed data: {failed[0]['data']}" if failed else "")
                ),
            )

        # --- Feature 4: thread history lists prior conversations ---
        r = client.get(f"{BASE}/api/threads")
        threads = r.json() if r.status_code == 200 else []
        thread_ids = {t.get("thread_id") for t in threads}
        record(
            "thread_history(enabled=True): GET /api/threads lists this session's threads",
            r.status_code == 200
            and {"smoke-chat", "smoke-weather", "smoke-approval"} <= thread_ids,
            f"threads found: {sorted(thread_ids)}",
        )

        # --- Feature 4b: rename and delete a thread ---
        r_rename = client.patch(
            f"{BASE}/api/threads/smoke-chat", json={"title": "Smoke test renamed"}
        )
        r_delete = client.delete(f"{BASE}/api/threads/smoke-chat")
        record(
            "thread_history: rename and delete a thread",
            r_rename.status_code == 200 and r_delete.status_code == 204,
            f"rename={r_rename.status_code} delete={r_delete.status_code}",
        )

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_file.close()

    print()
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"{passed}/{len(results)} checks passed")
    if passed != len(results):
        print(f"Server log: {log_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
