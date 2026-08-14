"""Adapters translating agent runtimes into normalized events.

Named ``adapters`` rather than ``langgraph`` so the package does not shadow the
real ``langgraph`` distribution.
"""

from agentstage.adapters.langgraph import LangGraphAdapter

__all__ = ["LangGraphAdapter"]
