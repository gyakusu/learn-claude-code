"""
mock_s12.py – Offline mock for s12_worktree_task_isolation.

Provides a fake Anthropic client that returns scripted tool-use responses
so the s12 tutorial can run without network access.

Usage:
    USE_MOCK=1 python agents/s12_worktree_task_isolation.py
"""

import json
import time


# ---------------------------------------------------------------------------
# Minimal dataclass-like objects that mimic the Anthropic SDK response shape
# ---------------------------------------------------------------------------
class _TextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _ToolUseBlock:
    _counter = 0

    def __init__(self, name: str, input_: dict):
        _ToolUseBlock._counter += 1
        self.type = "tool_use"
        self.name = name
        self.input = input_
        self.id = f"mock_{_ToolUseBlock._counter}_{int(time.time()*1000)}"


class _MockResponse:
    def __init__(self, content: list, stop_reason: str = "end_turn"):
        self.content = content
        self.stop_reason = stop_reason


# ---------------------------------------------------------------------------
# Scenario scripts – each user prompt is pattern-matched to a scripted reply
# ---------------------------------------------------------------------------

def _match(user_text: str, keywords: list[str]) -> bool:
    low = user_text.lower()
    return all(k in low for k in keywords)


def _script(user_text: str, turn: int):
    """Return (content_blocks, stop_reason) for a given user prompt."""

    # Step 2: Create worktrees and bind to tasks (must check BEFORE "create task")
    if _match(user_text, ["worktree"]) and _match(user_text, ["create"]) and _match(user_text, ["bind"]):
        return [
            _ToolUseBlock("worktree_create", {"name": "auth-refactor", "task_id": 1}),
            _ToolUseBlock("worktree_create", {"name": "ui-login", "task_id": 2}),
        ], "tool_use"

    if _match(user_text, ["worktree"]) and _match(user_text, ["create"]) and "auth" in user_text.lower():
        return [
            _ToolUseBlock("worktree_create", {"name": "auth-refactor", "task_id": 1}),
        ], "tool_use"

    if _match(user_text, ["bind"]) and "ui" in user_text.lower():
        return [
            _ToolUseBlock("worktree_create", {"name": "ui-login", "task_id": 2}),
        ], "tool_use"

    # Step 1: Create tasks + list
    if _match(user_text, ["create", "task"]):
        return [
            _ToolUseBlock("task_create", {"subject": "Backend auth refactor", "description": "Refactor the authentication module"}),
            _ToolUseBlock("task_create", {"subject": "Frontend login page", "description": "Build login page UI"}),
            _ToolUseBlock("task_list", {}),
        ], "tool_use"

    # Step 3: Run command in worktree
    if _match(user_text, ["run"]) and _match(user_text, ["auth"]):
        return [
            _ToolUseBlock("worktree_run", {"name": "auth-refactor", "command": "git status --short"}),
        ], "tool_use"

    if _match(user_text, ["run"]) and _match(user_text, ["worktree"]):
        return [
            _ToolUseBlock("worktree_run", {"name": "auth-refactor", "command": "git status --short"}),
        ], "tool_use"

    # Step 4: Keep worktree + list worktrees + events
    if _match(user_text, ["keep"]):
        return [
            _ToolUseBlock("worktree_keep", {"name": "ui-login"}),
            _ToolUseBlock("worktree_list", {}),
            _ToolUseBlock("worktree_events", {"limit": 20}),
        ], "tool_use"

    # Step 5: Remove worktree with complete_task + list tasks/worktrees/events
    # (must check BEFORE generic "list task" since prompt contains both)
    if _match(user_text, ["remove"]) and _match(user_text, ["auth"]):
        return [
            _ToolUseBlock("worktree_remove", {"name": "auth-refactor", "force": True, "complete_task": True}),
            _ToolUseBlock("task_list", {}),
            _ToolUseBlock("worktree_list", {}),
            _ToolUseBlock("worktree_events", {"limit": 20}),
        ], "tool_use"

    # List tasks (generic)
    if _match(user_text, ["list", "task"]):
        return [
            _ToolUseBlock("task_list", {}),
        ], "tool_use"

    # List worktrees
    if _match(user_text, ["list", "worktree"]):
        return [
            _ToolUseBlock("worktree_list", {}),
        ], "tool_use"

    # Events
    if _match(user_text, ["event"]):
        return [
            _ToolUseBlock("worktree_events", {"limit": 20}),
        ], "tool_use"

    # Fallback
    return None, None


# ---------------------------------------------------------------------------
# Follow-up: after tool results, produce a text summary
# ---------------------------------------------------------------------------

def _summarize_tool_results(tool_results: list[dict]) -> str:
    """Generate a human-readable summary from tool results."""
    parts = []
    for tr in tool_results:
        content = tr.get("content", "")
        if len(content) > 300:
            content = content[:300] + "..."
        parts.append(content)
    combined = "\n".join(parts)

    if "task_" in combined or "pending" in combined or "in_progress" in combined:
        return f"Done. Here are the results:\n\n{combined}"
    if "worktree" in combined.lower() or "active" in combined.lower():
        return f"Worktree operation complete:\n\n{combined}"
    if "event" in combined.lower():
        return f"Lifecycle events:\n\n{combined}"
    return f"Completed:\n\n{combined}"


# ---------------------------------------------------------------------------
# MockMessages – replaces client.messages
# ---------------------------------------------------------------------------
class MockMessages:
    def __init__(self):
        self._pending_summary = None

    def create(self, *, model, system, messages, tools, max_tokens):
        # If the last user message contains tool_results, generate summary
        last = messages[-1] if messages else None
        if last and last.get("role") == "user" and isinstance(last.get("content"), list):
            # This is a tool-result turn; produce a text summary
            if self._pending_summary:
                text = self._pending_summary
                self._pending_summary = None
            else:
                text = _summarize_tool_results(last["content"])
            return _MockResponse([_TextBlock(text)], "end_turn")

        # Extract user text from messages
        user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                if isinstance(m["content"], str):
                    user_text = m["content"]
                    break
                elif isinstance(m["content"], list):
                    for part in m["content"]:
                        if isinstance(part, dict) and part.get("type") == "text":
                            user_text = part["text"]
                            break
                    if user_text:
                        break

        turn = len([m for m in messages if m.get("role") == "user"])
        blocks, stop = _script(user_text, turn)

        if blocks is not None:
            return _MockResponse(blocks, stop)

        # No matching script – return a generic text response
        return _MockResponse(
            [_TextBlock(f"(mock) I can help with worktree and task operations. Try creating tasks, worktrees, or running commands.")],
            "end_turn",
        )


class MockAnthropic:
    """Drop-in replacement for anthropic.Anthropic with scripted responses."""

    def __init__(self, **kwargs):
        self.messages = MockMessages()
