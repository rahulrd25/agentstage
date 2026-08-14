"""Regenerate the event-contract golden file.

Run after an *intentional* change to the wire format, and commit the result
alongside the change so the break is visible in review:

    uv run python scripts/regenerate_golden_events.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# pytest supplies the repo root via `pythonpath`; running this directly does not.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.unit.test_event_contract import GOLDEN_PATH, _sample_events


def main() -> None:
    payload = {name: event.to_dict() for name, event in _sample_events().items()}
    GOLDEN_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {GOLDEN_PATH} ({len(payload)} event types).")


if __name__ == "__main__":
    main()
