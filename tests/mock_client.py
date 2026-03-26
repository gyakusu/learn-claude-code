"""
Mock Anthropic client for offline tutorial use.

Returns scripted tool-use responses so learners can explore
agent behaviour without an API key.

Usage:
    from tests.mock_client import MockAnthropic
    client = MockAnthropic()          # drop-in for Anthropic()
    resp = client.messages.create(...)  # returns scripted steps
"""

from dataclasses import dataclass, field


# ── tiny response objects (mirror the real SDK just enough) ──────────

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
    stop_reason: str = "end_turn"


# ── scenario helpers ────────────────────────────────────────────────

def _todo(id_, items):
    """Shorthand for a todo tool_use block."""
    return ToolUseBlock(id=id_, name="todo", input={"items": items})


def _bash(id_, cmd):
    return ToolUseBlock(id=id_, name="bash", input={"command": cmd})


def _read(id_, path):
    return ToolUseBlock(id=id_, name="read_file", input={"path": path})


def _write(id_, path, content):
    return ToolUseBlock(id=id_, name="write_file", input={"path": path, "content": content})


def _edit(id_, path, old, new):
    return ToolUseBlock(id=id_, name="edit_file", input={"path": path, "old_text": old, "new_text": new})


# ── scripted scenario for s03 ──────────────────────────────────────
#
# Simulates: "Refactor hello.py: add type hints, docstrings, and a main guard"
#
# The script is intentionally designed so that:
#   - Steps 0-1: model creates todos and reads the file
#   - Steps 2-4: model works WITHOUT calling todo  →  nag fires at step 4
#   - Steps 5-6: model resumes updating todos after the nag
#
# This lets learners observe both the TodoManager and the nag reminder.

HELLO_ORIGINAL = """\
def greet(name):
    return "Hello, " + name

print(greet("world"))
"""

HELLO_REFACTORED = '''\
def greet(name: str) -> str:
    """Return a greeting string for *name*."""
    return f"Hello, {name}"


def main() -> None:
    """Entry point."""
    print(greet("world"))


if __name__ == "__main__":
    main()
'''

S03_SCENARIO = [
    # Step 0 – plan: create todo list
    MockResponse(
        stop_reason="tool_use",
        content=[
            TextBlock("Let me plan the refactoring steps first."),
            _todo("t0", [
                {"id": "1", "text": "Read hello.py", "status": "in_progress"},
                {"id": "2", "text": "Add type hints", "status": "pending"},
                {"id": "3", "text": "Add docstrings", "status": "pending"},
                {"id": "4", "text": "Add main guard", "status": "pending"},
            ]),
        ],
    ),
    # Step 1 – read file & mark task 1 done, task 2 in_progress
    MockResponse(
        stop_reason="tool_use",
        content=[
            _todo("t1", [
                {"id": "1", "text": "Read hello.py", "status": "completed"},
                {"id": "2", "text": "Add type hints", "status": "in_progress"},
                {"id": "3", "text": "Add docstrings", "status": "pending"},
                {"id": "4", "text": "Add main guard", "status": "pending"},
            ]),
            _read("r1", "hello.py"),
        ],
    ),
    # Step 2 – edit: add type hints (NO todo call → rounds_since_todo starts counting)
    MockResponse(
        stop_reason="tool_use",
        content=[
            TextBlock("Adding type hints to greet()."),
            _edit("e2", "hello.py",
                  "def greet(name):",
                  "def greet(name: str) -> str:"),
        ],
    ),
    # Step 3 – edit: add docstring (still no todo → count = 2)
    MockResponse(
        stop_reason="tool_use",
        content=[
            TextBlock("Adding a docstring."),
            _edit("e3", "hello.py",
                  'def greet(name: str) -> str:\n    return',
                  'def greet(name: str) -> str:\n    """Return a greeting string for *name*."""\n    return'),
        ],
    ),
    # Step 4 – rewrite with main guard (no todo → count = 3 → NAG fires!)
    MockResponse(
        stop_reason="tool_use",
        content=[
            TextBlock("Rewriting with f-string and adding __main__ guard."),
            _write("w4", "hello.py", HELLO_REFACTORED),
        ],
    ),
    # Step 5 – after nag, model updates todos
    MockResponse(
        stop_reason="tool_use",
        content=[
            TextBlock("Right, let me update my progress."),
            _todo("t5", [
                {"id": "1", "text": "Read hello.py", "status": "completed"},
                {"id": "2", "text": "Add type hints", "status": "completed"},
                {"id": "3", "text": "Add docstrings", "status": "completed"},
                {"id": "4", "text": "Add main guard", "status": "completed"},
            ]),
        ],
    ),
    # Step 6 – verify
    MockResponse(
        stop_reason="tool_use",
        content=[
            _bash("b6", "python hello.py"),
        ],
    ),
    # Step 7 – done
    MockResponse(
        stop_reason="end_turn",
        content=[
            TextBlock(
                "All done! Here's what I did:\n"
                "1. Added type hints (`name: str`, `-> str`)\n"
                "2. Added docstrings to `greet` and `main`\n"
                "3. Wrapped the script in an `if __name__` guard\n\n"
                "Run `python hello.py` to verify."
            ),
        ],
    ),
]


# ── default scenario (generic fallback for any prompt) ─────────────

DEFAULT_SCENARIO = [
    MockResponse(
        stop_reason="tool_use",
        content=[
            TextBlock("Let me plan the steps."),
            _todo("t0", [
                {"id": "1", "text": "Understand the request", "status": "in_progress"},
                {"id": "2", "text": "Execute the task", "status": "pending"},
                {"id": "3", "text": "Verify the result", "status": "pending"},
            ]),
        ],
    ),
    MockResponse(
        stop_reason="tool_use",
        content=[
            _todo("t1", [
                {"id": "1", "text": "Understand the request", "status": "completed"},
                {"id": "2", "text": "Execute the task", "status": "in_progress"},
                {"id": "3", "text": "Verify the result", "status": "pending"},
            ]),
            _bash("b1", "echo 'mock: task executed'"),
        ],
    ),
    MockResponse(
        stop_reason="tool_use",
        content=[
            _todo("t2", [
                {"id": "1", "text": "Understand the request", "status": "completed"},
                {"id": "2", "text": "Execute the task", "status": "completed"},
                {"id": "3", "text": "Verify the result", "status": "in_progress"},
            ]),
            _bash("b2", "echo 'mock: verified'"),
        ],
    ),
    MockResponse(
        stop_reason="end_turn",
        content=[
            TextBlock("Task completed. (This was a mock response.)"),
        ],
    ),
]


# ── mock Messages API ──────────────────────────────────────────────

class MockMessages:
    """Drop-in for client.messages with scripted responses."""

    def __init__(self):
        self._scenarios = {
            "s03": list(S03_SCENARIO),
        }
        self._queue = []

    def _pick_scenario(self, messages):
        """Choose a scenario based on the first user message."""
        for msg in messages:
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                text = msg["content"].lower()
                if "hello.py" in text or "refactor" in text or "type hint" in text:
                    return "s03"
        return None

    def create(self, *, model=None, system=None, messages=None, tools=None,
               max_tokens=None, **kwargs):
        if not self._queue:
            key = self._pick_scenario(messages or [])
            if key and key in self._scenarios:
                self._queue = list(self._scenarios[key])
            else:
                self._queue = list(DEFAULT_SCENARIO)
        return self._queue.pop(0) if self._queue else MockResponse(
            stop_reason="end_turn",
            content=[TextBlock("(mock: no more scripted responses)")],
        )


class MockAnthropic:
    """Drop-in replacement for anthropic.Anthropic()."""

    def __init__(self, **kwargs):
        self.messages = MockMessages()
