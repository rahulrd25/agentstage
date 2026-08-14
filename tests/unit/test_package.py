"""The package imports cleanly and exposes its version."""

from __future__ import annotations

import agentstage


def test_version_is_exposed():
    assert agentstage.__version__ == "0.1.0"


def test_importing_the_package_does_not_pull_in_a_web_stack():
    """Importing `agentstage` must stay cheap. The FastAPI/SSE runtime and the
    adapter are imported on use, not at package import, so a library consumer
    doesn't pay for a server they never start.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import agentstage, sys; "
            "print(','.join(m for m in ('fastapi', 'langgraph') if m in sys.modules))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "", f"eagerly imported: {result.stdout.strip()}"
