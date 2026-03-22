"""
mock_client.py - Offline mock for the Anthropic Messages API.

Usage:
    MOCK=1 python agents/s01_agent_loop.py
    MOCK=1 python agents/s04_subagent.py
    MOCK=1 python agents/s06_context_compact.py

Simulates tool_use / end_turn responses so learners can run tutorials
without an API key or network access.

For s04 (subagents), uses scenario-based responses that distinguish
parent (has task tool) from child (no task tool) calls.

For s06 (context compact), uses a scripted sequence of read_file calls
to grow context and trigger the three-layer compression pipeline.
"""

import re
import dataclasses
import uuid
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
# Generic pattern-based responses (s01, s02, s03, etc.)
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
# Scenario-based responses for s04 subagent pattern
# ---------------------------------------------------------------------------
# Each scenario is a list of steps.  Each step is either:
#   ("tool_use", tool_name, tool_input)   -- the mock calls a tool
#   ("text", message)                     -- the mock returns final text

SCENARIOS = {
    # --- Parent scenarios ---------------------------------------------------
    "parent:test_framework": [
        ("tool_use", "task", {
            "prompt": "Find out what testing framework this project uses. "
                      "Read config files like requirements.txt, setup.cfg, "
                      "pyproject.toml, or package.json.",
            "description": "find testing framework",
        }),
        ("text", "Based on the subagent's investigation, this project uses "
                 "**no dedicated testing framework** in requirements.txt "
                 "(only anthropic and python-dotenv are listed). The GitHub "
                 "Actions workflow references `python tests/test_unit.py`, "
                 "suggesting plain unittest or a simple script-based approach."),
    ],
    "parent:summarize_py": [
        ("tool_use", "task", {
            "prompt": "Read all .py files under agents/ and give a one-line "
                      "summary of each.",
            "description": "summarize agent files",
        }),
        ("text", "Here is the summary from the subagent:\n\n"
                 "- **s01_agent_loop.py** -- Minimal agent loop: send message, "
                 "print response.\n"
                 "- **s02_tool_use.py** -- Adds tool definitions and a "
                 "tool-dispatch loop.\n"
                 "- **s03_todo_write.py** -- Adds a todo/checklist tool for "
                 "task tracking.\n"
                 "- **s04_subagent.py** -- Spawns child agents with isolated "
                 "context.\n"
                 "- **mock_client.py** -- Offline mock of the Anthropic API "
                 "for tutorials."),
    ],
    "parent:default": [
        ("tool_use", "task", {
            "prompt": "Investigate the user's request by reading relevant "
                      "files in the workspace.",
            "description": "investigate request",
        }),
        ("text", "The subagent explored the workspace and here is what it "
                 "found:\n\nThis is a tutorial project called "
                 "**learn-claude-code** that teaches how to build AI coding "
                 "agents step by step (s01 through s12). Each stage adds one "
                 "concept: agent loop, tool use, todo tracking, subagents, "
                 "skill loading, and more."),
    ],

    # --- Child (subagent) scenarios -----------------------------------------
    "child:test_framework": [
        ("tool_use", "read_file", {"path": "requirements.txt"}),
        ("tool_use", "bash", {
            "command": "ls *.toml *.cfg 2>/dev/null || echo 'no config files'",
        }),
        ("text", "This project's requirements.txt lists only "
                 "`anthropic>=0.25.0` and `python-dotenv>=1.0.0`. No pytest, "
                 "unittest2, or nose. The CI workflow runs "
                 "`python tests/test_unit.py`, suggesting plain unittest or "
                 "script-based tests."),
    ],
    "child:summarize_py": [
        ("tool_use", "bash", {"command": "ls agents/*.py"}),
        ("tool_use", "read_file", {"path": "agents/s01_agent_loop.py", "limit": 5}),
        ("tool_use", "read_file", {"path": "agents/s02_tool_use.py", "limit": 5}),
        ("text", "Summary of agent files:\n"
                 "- s01_agent_loop.py: Minimal agent loop\n"
                 "- s02_tool_use.py: Tool definitions and dispatch\n"
                 "- s03_todo_write.py: Todo/checklist tracking\n"
                 "- s04_subagent.py: Subagent context isolation\n"
                 "- mock_client.py: Offline API mock"),
    ],
    "child:default": [
        ("tool_use", "bash", {
            "command": "find . -maxdepth 2 -name '*.py' | head -20",
        }),
        ("text", "The workspace contains a tutorial project with multiple "
                 "agent stages (s01-s12) under agents/, skill definitions "
                 "under skills/, and documentation under docs/."),
    ],
}

# Tracks position within each scenario (shared across calls in one process)
_scenario_step: dict[str, int] = {}


def _pick_scenario(role: str, messages: list) -> str:
    """Choose a scenario key based on role and conversation content."""
    first_msg = ""
    for m in messages:
        if m["role"] == "user":
            content = m["content"]
            if isinstance(content, str):
                first_msg = content.lower()
                break
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        first_msg = part["text"].lower()
                        break
                if first_msg:
                    break

    if re.search(r"test|pytest|framework|testing", first_msg):
        return f"{role}:test_framework"
    if re.search(r"summar|\.py|each|all.*file", first_msg):
        return f"{role}:summarize_py"
    return f"{role}:default"


def _scenario_response(messages: list, tools: list) -> MockResponse:
    """Scenario-based response for s04 subagent calls."""
    has_task_tool = any(t.get("name") == "task" for t in (tools or []))
    role = "parent" if has_task_tool else "child"

    scenario_key = _pick_scenario(role, messages)
    steps = SCENARIOS.get(scenario_key, SCENARIOS[f"{role}:default"])

    idx = _scenario_step.get(scenario_key, 0)
    if idx >= len(steps):
        return MockResponse(
            content=[TextBlock(text="Done.")], stop_reason="end_turn",
        )

    content = []
    stop_reason = "end_turn"

    step = steps[idx]
    if step[0] == "text":
        content.append(TextBlock(text=step[1]))
        _scenario_step[scenario_key] = idx + 1
    else:
        while idx < len(steps) and steps[idx][0] == "tool_use":
            _, tool_name, tool_input = steps[idx]
            content.append(ToolUseBlock(
                id=_next_tool_id(), name=tool_name, input=tool_input,
            ))
            idx += 1
        _scenario_step[scenario_key] = idx
        stop_reason = "tool_use"

    return MockResponse(content=content, stop_reason=stop_reason)


# ---------------------------------------------------------------------------
# s06 context-compact scenario helpers
# ---------------------------------------------------------------------------

def _s06_tool(name: str, inp: dict) -> dict:
    """Shorthand for a tool_use step in s06 scenario."""
    return {"kind": "tool_use", "name": name, "input": inp}


def _s06_text(msg: str) -> dict:
    """Shorthand for a text step in s06 scenario."""
    return {"kind": "text", "text": msg}


def _s06_text_and_tool(msg: str, name: str, inp: dict) -> dict:
    """Text + tool_use in a single response (like a real model does)."""
    return {"kind": "text_and_tool", "text": msg, "name": name, "input": inp}


_S06_STEPS: list[dict] = [
    _s06_text_and_tool("Sure! I'll read every Python file in agents/ one by one.",
                       "read_file", {"path": "agents/s01_agent_loop.py"}),
    _s06_tool("read_file", {"path": "agents/s02_tool_use.py"}),
    _s06_tool("read_file", {"path": "agents/s03_todo_write.py"}),
    _s06_tool("read_file", {"path": "agents/s04_subagent.py"}),
    _s06_tool("read_file", {"path": "agents/s05_skill_loading.py"}),
    _s06_tool("read_file", {"path": "agents/s06_context_compact.py"}),
    _s06_tool("read_file", {"path": "agents/s07_task_system.py"}),
    _s06_tool("read_file", {"path": "agents/s08_background_tasks.py"}),
    _s06_tool("read_file", {"path": "agents/s09_agent_teams.py"}),
    _s06_text(
        "I've finished reading all 9 Python files in agents/. "
        "The context grew quite large — you may have seen micro_compact "
        "replacing old tool results with placeholders, and possibly an "
        "auto_compact summarisation if the token threshold was exceeded."
    ),
    _s06_tool("bash", {"command": "wc -l agents/*.py"}),
    _s06_text("All agents total about 1800 lines of Python."),
]
_S06_CYCLE_FROM = 10  # cycle back to bash + text pair


def _s06_response(messages: list, cursor: int) -> tuple[MockResponse, int]:
    """Return the next scripted s06 response and updated cursor."""
    # Special case: summarisation request from auto_compact
    if (len(messages) == 1
            and isinstance(messages[0].get("content"), str)
            and "Summarize this conversation" in messages[0]["content"]):
        return MockResponse(
            content=[TextBlock(text=(
                "Summary: The agent read multiple Python files from agents/. "
                "Key observations: each file implements one stage of a coding-agent "
                "tutorial (s01-s09). The agent demonstrated tool use (bash, "
                "read_file, write_file, edit_file) and context compression. "
                "Current state: all files have been read successfully."
            ))],
            stop_reason="end_turn",
        ), cursor

    step = _S06_STEPS[cursor]
    next_cursor = cursor + 1
    if next_cursor >= len(_S06_STEPS):
        next_cursor = _S06_CYCLE_FROM

    if step["kind"] == "tool_use":
        block = ToolUseBlock(
            id=f"toolu_{uuid.uuid4().hex[:24]}",
            name=step["name"],
            input=step["input"],
        )
        return MockResponse(content=[block], stop_reason="tool_use"), next_cursor
    elif step["kind"] == "text_and_tool":
        text_block = TextBlock(text=step["text"])
        tool_block = ToolUseBlock(
            id=f"toolu_{uuid.uuid4().hex[:24]}",
            name=step["name"],
            input=step["input"],
        )
        return MockResponse(
            content=[text_block, tool_block], stop_reason="tool_use",
        ), next_cursor
    else:
        block = TextBlock(text=step["text"])
        return MockResponse(content=[block], stop_reason="end_turn"), next_cursor


# ---------------------------------------------------------------------------
# s07 task-system scenario helpers
# ---------------------------------------------------------------------------

_s07_step: dict[str, int] = {}


def _s07_user_text(messages: list) -> str:
    """Extract the original user query (first plain-text user message)."""
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"]
    return ""


def _s07_tool_result_count(messages: list) -> int:
    """Count how many tool_result round-trips have occurred."""
    return sum(
        1 for m in messages
        if isinstance(m.get("content"), list)
        and any(isinstance(b, dict) and b.get("type") == "tool_result"
                for b in m["content"])
    )


def _s07_response(messages: list) -> MockResponse:
    """Scripted responses for s07 task_create/update/list/get tools."""
    user_text = _s07_user_text(messages)
    q = user_text.lower()
    last = messages[-1]
    is_tool_result = (
        isinstance(last.get("content"), list)
        and any(isinstance(b, dict) and b.get("type") == "tool_result"
                for b in last["content"])
    )

    round_count = _s07_tool_result_count(messages)

    # --- "create 3 tasks" with dependency wiring ---
    if re.search(r"create\s+\d+\s+task", q):
        subjects = re.findall(r'"([^"]+)"', user_text)
        if not subjects:
            subjects = ["Task 1", "Task 2", "Task 3"]
        n = len(subjects)

        if not is_tool_result:
            # Step 1: create first task
            return MockResponse(
                content=[
                    TextBlock(text="I'll create the tasks and set up dependencies."),
                    ToolUseBlock(id=_next_tool_id(), name="task_create",
                                input={"subject": subjects[0]}),
                ], stop_reason="tool_use")

        # Steps 2..n: create remaining tasks
        if round_count <= n - 1:
            idx = round_count
            return MockResponse(
                content=[ToolUseBlock(id=_next_tool_id(), name="task_create",
                                     input={"subject": subjects[idx]})],
                stop_reason="tool_use")

        # Steps n+1..2n-1: wire dependencies (task i+1 blocked by task i)
        dep_round = round_count - n
        if dep_round < n - 1:
            task_id = dep_round + 2  # tasks 2, 3, ...
            return MockResponse(
                content=[ToolUseBlock(id=_next_tool_id(), name="task_update",
                                     input={"task_id": task_id,
                                            "addBlockedBy": [task_id - 1]})],
                stop_reason="tool_use")

        # Final: list all tasks
        if dep_round == n - 1:
            return MockResponse(
                content=[ToolUseBlock(id=_next_tool_id(), name="task_list",
                                     input={})],
                stop_reason="tool_use")

        # Done
        results = [b.get("content", "") for b in last["content"]
                   if isinstance(b, dict) and b.get("type") == "tool_result"]
        return MockResponse(
            content=[TextBlock(text=f"Done.\n\n{results[0][:500]}" if results else "Done.")],
            stop_reason="end_turn")

    # --- "task board" with parallel deps ---
    if re.search(r"task.*board|parse.*transform.*emit", q):
        board = ["Parse", "Transform", "Emit", "Test"]

        if not is_tool_result:
            return MockResponse(
                content=[
                    TextBlock(text="I'll create a task board with the dependency structure."),
                    ToolUseBlock(id=_next_tool_id(), name="task_create",
                                input={"subject": board[0]}),
                ], stop_reason="tool_use")

        if round_count <= 3:
            return MockResponse(
                content=[ToolUseBlock(id=_next_tool_id(), name="task_create",
                                     input={"subject": board[round_count]})],
                stop_reason="tool_use")

        # Wire: Transform(2) <- Parse(1), Emit(3) <- Parse(1), Test(4) <- [2,3]
        dep_steps = [
            {"task_id": 2, "addBlockedBy": [1]},
            {"task_id": 3, "addBlockedBy": [1]},
            {"task_id": 4, "addBlockedBy": [2, 3]},
        ]
        dep_idx = round_count - 4
        if dep_idx < len(dep_steps):
            return MockResponse(
                content=[ToolUseBlock(id=_next_tool_id(), name="task_update",
                                     input=dep_steps[dep_idx])],
                stop_reason="tool_use")

        if dep_idx == len(dep_steps):
            return MockResponse(
                content=[ToolUseBlock(id=_next_tool_id(), name="task_list",
                                     input={})],
                stop_reason="tool_use")

        results = [b.get("content", "") for b in last["content"]
                   if isinstance(b, dict) and b.get("type") == "tool_result"]
        return MockResponse(
            content=[TextBlock(text=f"Done.\n\n{results[0][:500]}" if results else "Done.")],
            stop_reason="end_turn")

    # --- "complete task N" then list ---
    m = re.search(r"complete\s+task\s+(\d+)", q)
    if m:
        tid = int(m.group(1))
        if not is_tool_result:
            return MockResponse(
                content=[ToolUseBlock(id=_next_tool_id(), name="task_update",
                                     input={"task_id": tid, "status": "completed"})],
                stop_reason="tool_use")
        if round_count == 1:
            return MockResponse(
                content=[ToolUseBlock(id=_next_tool_id(), name="task_list",
                                     input={})],
                stop_reason="tool_use")
        results = [b.get("content", "") for b in last["content"]
                   if isinstance(b, dict) and b.get("type") == "tool_result"]
        return MockResponse(
            content=[TextBlock(text=f"Task {tid} completed.\n\n{results[0][:500]}" if results else "Done.")],
            stop_reason="end_turn")

    # --- "list tasks" ---
    if re.search(r"list.*task|show.*task|task.*graph|dependency", q):
        if not is_tool_result:
            return MockResponse(
                content=[ToolUseBlock(id=_next_tool_id(), name="task_list",
                                     input={})],
                stop_reason="tool_use")
        results = [b.get("content", "") for b in last["content"]
                   if isinstance(b, dict) and b.get("type") == "tool_result"]
        return MockResponse(
            content=[TextBlock(text=f"Here are the current tasks:\n\n{results[0][:500]}" if results else "No tasks.")],
            stop_reason="end_turn")

    # --- fallback ---
    return MockResponse(
        content=[TextBlock(text=f"(mock) I would handle: {user_text[:120]}")],
        stop_reason="end_turn")


# ---------------------------------------------------------------------------
# Drop-in replacement for anthropic.Anthropic
# ---------------------------------------------------------------------------

class _Messages:
    def __init__(self) -> None:
        self._s06_cursor: int = 0

    def create(self, *, model: str, system: Any = None,
               messages: list, tools: list = None,
               max_tokens: int = 8000, **kwargs) -> MockResponse:
        tool_names = {t.get("name") for t in (tools or [])}

        # s06: compact tool present → use scripted sequence
        if "compact" in tool_names:
            resp, self._s06_cursor = _s06_response(messages, self._s06_cursor)
            return resp

        # s07: task_create/update/list/get tools present
        if "task_create" in tool_names:
            return _s07_response(messages)

        # s04+: scenario-based logic for task tool / subagent children
        has_task_tool = "task" in tool_names
        is_subagent_child = (
            not has_task_tool
            and tools
            and "read_file" in tool_names
            and len(messages) == 1  # fresh context = subagent
        )
        if has_task_tool or is_subagent_child:
            return _scenario_response(messages, tools or [])

        # s01, s02, s03: generic pattern-based responses
        return _infer_response(messages, tools or [])


class MockAnthropic:
    """Quacks like anthropic.Anthropic() but needs no network."""

    def __init__(self, **kwargs):
        self.messages = _Messages()


# Alias so s06 import works too
MockAnthropicClient = MockAnthropic
