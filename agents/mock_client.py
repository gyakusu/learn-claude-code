"""
mock_client.py - Offline mock for the Anthropic SDK client.

Usage:
    Set MOCK=1 in your environment (or .env) to swap in this mock.
    Each agent only needs a two-line change at the client-creation site.

The mock returns *scripted* tool-use / text responses so you can observe
the agent's harness behavior (tool dispatch, compression, etc.) without
making real API calls.

Supports two response modes:
  1. tool_use  – the "model" asks to call a tool
  2. text      – the "model" replies with plain text

Scenario scripts are per-agent; see SCENARIOS below.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Lightweight response objects that mirror the Anthropic SDK shapes
# ---------------------------------------------------------------------------

@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class MockResponse:
    content: list
    stop_reason: str
    model: str = "mock-model"
    usage: dict = field(default_factory=lambda: {"input_tokens": 100, "output_tokens": 50})


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------

def _tool(name: str, inp: dict) -> dict:
    """Shorthand for a tool_use step."""
    return {"kind": "tool_use", "name": name, "input": inp}


def _text(msg: str) -> dict:
    """Shorthand for a text step."""
    return {"kind": "text", "text": msg}


def _text_and_tool(msg: str, name: str, inp: dict) -> dict:
    """Text + tool_use in a single response (like a real model does)."""
    return {"kind": "text_and_tool", "text": msg, "name": name, "input": inp}


# ---------------------------------------------------------------------------
# Per-agent scenarios
#
# Each scenario is a list of steps.  The mock replays them in order; after
# the list is exhausted it cycles back from a configurable restart index.
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, list[dict]] = {
    # --- s06: context compact demo ---
    # The script reads many files so the context grows, triggering compression.
    "s06": [
        _text_and_tool("Sure! I'll read every Python file in agents/ one by one.",
                       "read_file", {"path": "agents/s01_agent_loop.py"}),
        _tool("read_file", {"path": "agents/s02_tool_use.py"}),
        _tool("read_file", {"path": "agents/s03_todo_write.py"}),
        _tool("read_file", {"path": "agents/s04_subagent.py"}),
        _tool("read_file", {"path": "agents/s05_skill_loading.py"}),
        _tool("read_file", {"path": "agents/s06_context_compact.py"}),
        _tool("read_file", {"path": "agents/s07_task_system.py"}),
        _tool("read_file", {"path": "agents/s08_background_tasks.py"}),
        _tool("read_file", {"path": "agents/s09_agent_teams.py"}),
        _text(
            "I've finished reading all 9 Python files in agents/. "
            "The context grew quite large — you may have seen micro_compact "
            "replacing old tool results with placeholders, and possibly an "
            "auto_compact summarisation if the token threshold was exceeded."
        ),
        _tool("bash", {"command": "wc -l agents/*.py"}),
        _text("All agents total about 1800 lines of Python."),
        # After this, cycle from index 10 (the bash + text pair)
    ],

    # Fallback: simple echo
    "_default": [
        _text("(mock) I received your message."),
    ],
}

# When a scenario is exhausted, restart from this index (per scenario).
_CYCLE_FROM: dict[str, int] = {
    "s06": 10,
}


# ---------------------------------------------------------------------------
# The mock Messages resource
# ---------------------------------------------------------------------------

class _MockMessages:
    """Drop-in replacement for client.messages.create(...)."""

    def __init__(self) -> None:
        self._cursors: dict[str, int] = {}   # scenario -> current step index

    def create(self, *, model: str = "", messages: list | None = None,
               system: str = "", tools: list | None = None,
               max_tokens: int = 4096, **kwargs: Any) -> MockResponse:
        scenario_key = self._detect_scenario(messages, system, tools)
        steps = SCENARIOS.get(scenario_key, SCENARIOS["_default"])
        cursor = self._cursors.get(scenario_key, 0)

        # --- special case: summarisation request (auto_compact) ---
        if self._is_summarisation_request(messages):
            return self._summarise(messages)

        step = steps[cursor]

        # Advance cursor (with cycling)
        next_cursor = cursor + 1
        if next_cursor >= len(steps):
            next_cursor = _CYCLE_FROM.get(scenario_key, 0)
        self._cursors[scenario_key] = next_cursor

        return self._step_to_response(step, model)

    # -- internal helpers --------------------------------------------------

    @staticmethod
    def _detect_scenario(messages: list | None, system: str,
                         tools: list | None) -> str:
        """Guess which agent is calling based on tools / system prompt."""
        if tools:
            tool_names = {t["name"] for t in tools}
            if "compact" in tool_names:
                return "s06"
        return "_default"

    @staticmethod
    def _is_summarisation_request(messages: list | None) -> bool:
        if not messages or len(messages) != 1:
            return False
        msg = messages[0]
        content = msg.get("content", "")
        if isinstance(content, str) and "Summarize this conversation" in content:
            return True
        return False

    @staticmethod
    def _summarise(messages: list) -> MockResponse:
        return MockResponse(
            content=[TextBlock(
                text=(
                    "Summary: The agent read multiple Python files from agents/. "
                    "Key observations: each file implements one stage of a coding-agent "
                    "tutorial (s01–s09). The agent demonstrated tool use (bash, "
                    "read_file, write_file, edit_file) and context compression. "
                    "Current state: all files have been read successfully."
                )
            )],
            stop_reason="end_turn",
        )

    @staticmethod
    def _step_to_response(step: dict, model: str) -> MockResponse:
        if step["kind"] == "tool_use":
            block = ToolUseBlock(
                id=f"toolu_{uuid.uuid4().hex[:24]}",
                name=step["name"],
                input=step["input"],
            )
            return MockResponse(content=[block], stop_reason="tool_use", model=model)
        elif step["kind"] == "text_and_tool":
            text_block = TextBlock(text=step["text"])
            tool_block = ToolUseBlock(
                id=f"toolu_{uuid.uuid4().hex[:24]}",
                name=step["name"],
                input=step["input"],
            )
            return MockResponse(content=[text_block, tool_block],
                                stop_reason="tool_use", model=model)
        else:
            block = TextBlock(text=step["text"])
            return MockResponse(content=[block], stop_reason="end_turn", model=model)


# ---------------------------------------------------------------------------
# Public: drop-in replacement for Anthropic()
# ---------------------------------------------------------------------------

class MockAnthropicClient:
    """Mimics the subset of the Anthropic SDK client used by the agents."""

    def __init__(self, **kwargs: Any) -> None:
        self.messages = _MockMessages()
