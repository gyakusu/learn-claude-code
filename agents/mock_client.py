"""
mock_client.py - Offline mock for the Anthropic Messages API.

Usage:
    MOCK=1 python agents/s01_agent_loop.py

Simulates tool_use / end_turn responses so learners can run tutorials
without an API key or network access.
"""

import json
import re
import dataclasses
from typing import Any


# ---------------------------------------------------------------------------
# Dataclasses that mirror the Anthropic SDK response shapes
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TextBlock:
    type: str = "text"
    text: str = ""


@dataclasses.dataclass
class ToolUseBlock:
    type: str = "tool_use"
    id: str = "mock_tool_0"
    name: str = "bash"
    input: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class MockResponse:
    content: list = dataclasses.field(default_factory=list)
    stop_reason: str = "end_turn"


# ---------------------------------------------------------------------------
# Pattern-based response rules
# ---------------------------------------------------------------------------

_TOOL_COUNTER = 0


def _next_tool_id() -> str:
    global _TOOL_COUNTER
    _TOOL_COUNTER += 1
    return f"mock_tool_{_TOOL_COUNTER}"


def _infer_response(messages: list, tools: list) -> MockResponse:
    """Pick a plausible mock response based on the last user/tool message."""

    last = messages[-1]

    # After a tool_result, just acknowledge with text (end the loop).
    if isinstance(last.get("content"), list) and any(
        isinstance(c, dict) and c.get("type") == "tool_result"
        for c in last["content"]
    ):
        # Summarise what happened
        results = [
            c.get("content", "")
            for c in last["content"]
            if isinstance(c, dict) and c.get("type") == "tool_result"
        ]
        summary = results[0][:200] if results else "(done)"
        return MockResponse(
            content=[TextBlock(text=f"Done. Output:\n{summary}")],
            stop_reason="end_turn",
        )

    # Otherwise it's a fresh user query -- try to pick a bash command.
    query = last.get("content", "") if isinstance(last.get("content"), str) else ""
    cmd = _query_to_command(query)

    if cmd:
        return MockResponse(
            content=[ToolUseBlock(id=_next_tool_id(), input={"command": cmd})],
            stop_reason="tool_use",
        )

    # Fallback: just reply with text.
    return MockResponse(
        content=[TextBlock(text=f"(mock) I would handle: {query[:120]}")],
        stop_reason="end_turn",
    )


def _query_to_command(query: str) -> str | None:
    """Map common tutorial prompts to shell commands."""
    q = query.lower()

    if "hello" in q and ("file" in q or "create" in q or "write" in q):
        return 'echo \'print("Hello, World!")\' > hello.py && echo "Created hello.py"'

    if "list" in q and "python" in q:
        return "ls *.py"

    if "git" in q and "branch" in q:
        return "git branch --show-current"

    if "directory" in q or "mkdir" in q or ("create" in q and "file" in q):
        return (
            "mkdir -p test_output && "
            "echo 'file1' > test_output/a.txt && "
            "echo 'file2' > test_output/b.txt && "
            "echo 'file3' > test_output/c.txt && "
            "echo 'Created test_output/ with 3 files'"
        )

    # Generic: wrap the query into a simple echo so the loop still cycles.
    if query.strip():
        return f"echo '(mock) executed placeholder for: {query[:80]}'"

    return None


# ---------------------------------------------------------------------------
# Drop-in replacement for anthropic.Anthropic
# ---------------------------------------------------------------------------

class _Messages:
    def create(self, *, model: str, system: Any = None,
               messages: list, tools: list = None,
               max_tokens: int = 8000, **kwargs) -> MockResponse:
        return _infer_response(messages, tools or [])


class MockAnthropic:
    """Quacks like anthropic.Anthropic() but needs no network."""

    def __init__(self, **kwargs):
        self.messages = _Messages()
