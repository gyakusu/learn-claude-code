"""
Tests for s03_todo_write – TodoManager + nag reminder logic.

Run:  python -m pytest tests/test_s03.py -v
      MOCK=1 python agents/s03_todo_write.py   (interactive demo)
"""

import os
import sys
from pathlib import Path

# Ensure the repo root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Set MOCK so importing s03 doesn't require the anthropic package
os.environ.setdefault("MOCK", "1")

import pytest


# ── TodoManager unit tests ──────────────────────────────────────────

from agents.s03_todo_write import TodoManager


class TestTodoManager:
    def setup_method(self):
        self.tm = TodoManager()

    def test_basic_update(self):
        result = self.tm.update([
            {"id": "1", "text": "task A", "status": "pending"},
            {"id": "2", "text": "task B", "status": "in_progress"},
        ])
        assert "[>] #2: task B" in result
        assert "[ ] #1: task A" in result
        assert "(0/2 completed)" in result

    def test_completed_count(self):
        result = self.tm.update([
            {"id": "1", "text": "done", "status": "completed"},
            {"id": "2", "text": "also done", "status": "completed"},
        ])
        assert "(2/2 completed)" in result

    def test_only_one_in_progress(self):
        with pytest.raises(ValueError, match="Only one task"):
            self.tm.update([
                {"id": "1", "text": "A", "status": "in_progress"},
                {"id": "2", "text": "B", "status": "in_progress"},
            ])

    def test_empty_text_rejected(self):
        with pytest.raises(ValueError, match="text required"):
            self.tm.update([{"id": "1", "text": "", "status": "pending"}])

    def test_invalid_status_rejected(self):
        with pytest.raises(ValueError, match="invalid status"):
            self.tm.update([{"id": "1", "text": "X", "status": "cancelled"}])

    def test_max_20_items(self):
        with pytest.raises(ValueError, match="Max 20"):
            self.tm.update([
                {"id": str(i), "text": f"t{i}", "status": "pending"}
                for i in range(21)
            ])

    def test_render_empty(self):
        assert self.tm.render() == "No todos."


# ── Nag reminder logic tests ───────────────────────────────────────

class TestNagReminder:
    """Test the nag injection logic extracted from agent_loop."""

    @staticmethod
    def simulate_rounds(tool_calls_per_round: list[list[str]]):
        """
        Simulate rounds and return rounds_since_todo after each round,
        plus whether a nag would fire.

        Args:
            tool_calls_per_round: list of lists, each inner list contains
                tool names called in that round.

        Returns:
            list of (rounds_since_todo, nag_fired) tuples.
        """
        rounds_since_todo = 0
        trace = []
        for tools_used in tool_calls_per_round:
            used_todo = "todo" in tools_used
            rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
            nag = rounds_since_todo >= 3
            trace.append((rounds_since_todo, nag))
        return trace

    def test_no_nag_when_todo_used_regularly(self):
        trace = self.simulate_rounds([
            ["todo", "read_file"],   # round 1
            ["edit_file"],           # round 2
            ["todo"],               # round 3 – resets counter
            ["bash"],               # round 4
        ])
        assert trace == [(0, False), (1, False), (0, False), (1, False)]

    def test_nag_fires_after_3_rounds_without_todo(self):
        trace = self.simulate_rounds([
            ["todo"],       # 0
            ["bash"],       # 1
            ["edit_file"],  # 2
            ["bash"],       # 3 → nag!
        ])
        assert trace[-1] == (3, True)

    def test_nag_resets_after_todo(self):
        trace = self.simulate_rounds([
            ["bash"],       # 1
            ["bash"],       # 2
            ["bash"],       # 3 → nag
            ["todo"],       # 0 – reset
            ["bash"],       # 1
        ])
        assert trace[2] == (3, True)
        assert trace[3] == (0, False)
        assert trace[4] == (1, False)


# ── Mock client integration test ───────────────────────────────────

from tests.mock_client import MockAnthropic


class TestMockIntegration:
    """Verify the mock produces a valid multi-step conversation."""

    def test_s03_scenario_runs_to_completion(self):
        mock = MockAnthropic()
        messages = [{"role": "user", "content": "Refactor hello.py: add type hints"}]

        steps = 0
        while steps < 20:  # safety limit
            resp = mock.messages.create(model="mock", messages=messages)
            steps += 1
            if resp.stop_reason != "tool_use":
                break

        # Should reach end_turn (the final scripted response)
        assert resp.stop_reason == "end_turn"
        assert any(hasattr(b, "text") and "done" in b.text.lower()
                    for b in resp.content)

    def test_s03_scenario_has_todo_calls(self):
        """The scenario should include todo tool_use blocks."""
        mock = MockAnthropic()
        messages = [{"role": "user", "content": "Refactor hello.py"}]

        todo_count = 0
        for _ in range(20):
            resp = mock.messages.create(model="mock", messages=messages)
            for block in resp.content:
                if hasattr(block, "name") and block.name == "todo":
                    todo_count += 1
            if resp.stop_reason != "tool_use":
                break

        assert todo_count >= 3, "Scenario should use todo at least 3 times"

    def test_nag_gap_exists_in_scenario(self):
        """Steps 2-4 have no todo call, so the nag should fire."""
        mock = MockAnthropic()
        messages = [{"role": "user", "content": "Refactor hello.py"}]

        rounds_since_todo = 0
        nag_fired = False

        for _ in range(20):
            resp = mock.messages.create(model="mock", messages=messages)
            used_todo = any(
                hasattr(b, "name") and b.name == "todo"
                for b in resp.content
            )
            rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
            if rounds_since_todo >= 3:
                nag_fired = True
            if resp.stop_reason != "tool_use":
                break

        assert nag_fired, "Scenario should trigger the nag reminder"
