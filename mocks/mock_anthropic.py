"""
Mock Anthropic client for offline testing of agent tutorials.

Simulates Claude API responses by pattern-matching user messages
and generating appropriate tool_use / text responses.

Usage:
    from mocks.mock_anthropic import MockAnthropic
    client = MockAnthropic()
    response = client.messages.create(model=..., messages=..., tools=..., ...)
"""

import re
import uuid
from dataclasses import dataclass, field


@dataclass
class TextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class ToolUseBlock:
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = f"toolu_{uuid.uuid4().hex[:24]}"


@dataclass
class MockResponse:
    content: list = field(default_factory=list)
    stop_reason: str = "end_turn"
    model: str = "mock-model"
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})


class MockMessages:
    """Simulates client.messages.create() for s08 background tasks."""

    def __init__(self):
        self._call_count = 0
        self._pending_bg_tasks = set()

    def create(self, *, model, messages, tools=None, system=None, max_tokens=8000, **kwargs):
        self._call_count += 1
        last_user = self._extract_last_user_text(messages)
        last_role = messages[-1]["role"] if messages else "user"
        last_content = messages[-1].get("content", "")

        # If we just got tool results back, decide next action
        if isinstance(last_content, list) and any(
            isinstance(item, dict) and item.get("type") == "tool_result" for item in last_content
        ):
            return self._handle_tool_results(messages, last_user)

        # If background results were injected, acknowledge and continue
        if isinstance(last_content, str) and "Noted background results" in last_content:
            return self._after_bg_notification(messages, last_user)

        # Fresh user request
        return self._handle_user_request(last_user)

    def _extract_last_user_text(self, messages):
        """Walk backwards to find the most recent real user text."""
        for msg in reversed(messages):
            if msg["role"] == "user":
                content = msg["content"]
                if isinstance(content, str):
                    if "<background-results>" not in content:
                        return content.lower()
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool_result":
                            continue
                        if isinstance(item, dict) and "text" in item:
                            return item["text"].lower()
        return ""

    def _handle_user_request(self, text):
        # Background + foreground parallel request
        if "background" in text and ("file" in text or "create" in text or "write" in text):
            return self._bg_and_foreground(text)

        # Multiple background tasks
        if re.search(r"(3|three|multiple)\s*(background)?\s*task", text):
            return self._multiple_bg_tasks()

        # Single background request
        if "background" in text or "bg" in text:
            return self._single_bg_task(text)

        # Status check
        if "status" in text or "check" in text:
            return MockResponse(
                content=[ToolUseBlock(name="check_background", input={})],
                stop_reason="tool_use",
            )

        # File operations
        if "create" in text and "file" in text:
            filename = self._extract_filename(text) or "example.txt"
            return MockResponse(
                content=[ToolUseBlock(
                    name="write_file",
                    input={"path": filename, "content": f"# Created by s08 mock\n"},
                )],
                stop_reason="tool_use",
            )

        if "read" in text:
            filename = self._extract_filename(text) or "example.txt"
            return MockResponse(
                content=[ToolUseBlock(name="read_file", input={"path": filename})],
                stop_reason="tool_use",
            )

        if "list" in text or "ls" in text:
            return MockResponse(
                content=[ToolUseBlock(name="bash", input={"command": "ls -la"})],
                stop_reason="tool_use",
            )

        # pytest in background
        if "pytest" in text or "test" in text:
            return MockResponse(
                content=[
                    TextBlock(text="Running pytest in the background so we can keep working."),
                    ToolUseBlock(name="background_run", input={"command": "python -m pytest --tb=short 2>&1 || echo 'pytest not found'"}),
                ],
                stop_reason="tool_use",
            )

        # Default: echo via bash
        return MockResponse(
            content=[TextBlock(text=f"I'll help with that. Let me run a command.")],
            stop_reason="end_turn",
        )

    def _single_bg_task(self, text):
        cmd = "sleep 3 && echo done"
        # Try to extract a command from quotes
        m = re.search(r'"([^"]+)"', text)
        if m:
            cmd = m.group(1)
        return MockResponse(
            content=[
                TextBlock(text="Starting that in the background."),
                ToolUseBlock(name="background_run", input={"command": cmd}),
            ],
            stop_reason="tool_use",
        )

    def _multiple_bg_tasks(self):
        return MockResponse(
            content=[
                TextBlock(text="Starting 3 background tasks."),
                ToolUseBlock(name="background_run", input={"command": "sleep 2 && echo 'task1 done'"}),
                ToolUseBlock(name="background_run", input={"command": "sleep 4 && echo 'task2 done'"}),
                ToolUseBlock(name="background_run", input={"command": "sleep 6 && echo 'task3 done'"}),
            ],
            stop_reason="tool_use",
        )

    def _bg_and_foreground(self, text):
        cmd = "sleep 5 && echo done"
        m = re.search(r'"([^"]+)"', text)
        if m:
            cmd = m.group(1)
        return MockResponse(
            content=[
                TextBlock(text="I'll run that in the background and create the file in parallel."),
                ToolUseBlock(name="background_run", input={"command": cmd}),
                ToolUseBlock(name="write_file", input={
                    "path": "config.json",
                    "content": '{\n  "name": "s08-demo",\n  "version": "1.0.0"\n}\n',
                }),
            ],
            stop_reason="tool_use",
        )

    def _handle_tool_results(self, messages, last_user):
        """After tool results come back, decide whether to continue or finish."""
        results = messages[-1]["content"]
        result_texts = []
        for item in results:
            if isinstance(item, dict) and item.get("type") == "tool_result":
                result_texts.append(item.get("content", ""))

        combined = " ".join(result_texts)

        # If a background task was just started, report it
        if "Background task" in combined and "started" in combined:
            task_ids = re.findall(r"Background task (\w+)", combined)
            if task_ids:
                summary = ", ".join(task_ids)
                # Check if there's also a foreground result
                has_foreground = any(
                    "Wrote" in item.get("content", "") or "Edited" in item.get("content", "")
                    for item in results
                    if isinstance(item, dict) and item.get("type") == "tool_result"
                )
                if has_foreground:
                    return MockResponse(
                        content=[TextBlock(
                            text=f"Background tasks started (IDs: {summary}). "
                                 f"The file has been created. The background command is still running -- "
                                 f"you can check on it with `check_background`."
                        )],
                        stop_reason="end_turn",
                    )
                if len(task_ids) > 1:
                    return MockResponse(
                        content=[TextBlock(
                            text=f"All {len(task_ids)} background tasks are running (IDs: {summary}). "
                                 f"Use `check_background` to monitor their progress."
                        )],
                        stop_reason="end_turn",
                    )
                return MockResponse(
                    content=[TextBlock(
                        text=f"Background task {summary} is running. I'll continue working while it runs."
                    )],
                    stop_reason="end_turn",
                )

        # Generic tool result acknowledgement
        return MockResponse(
            content=[TextBlock(text="Done.")],
            stop_reason="end_turn",
        )

    def _after_bg_notification(self, messages, last_user):
        """After background results are noted, summarize them."""
        # Find the background-results message
        for msg in reversed(messages):
            if isinstance(msg.get("content"), str) and "<background-results>" in msg["content"]:
                return MockResponse(
                    content=[TextBlock(
                        text="The background task has completed. " + self._summarize_bg(msg["content"])
                    )],
                    stop_reason="end_turn",
                )
        return MockResponse(
            content=[TextBlock(text="Background tasks have completed.")],
            stop_reason="end_turn",
        )

    @staticmethod
    def _summarize_bg(content):
        if "completed" in content:
            return "It finished successfully."
        if "error" in content.lower():
            return "It encountered an error."
        if "timeout" in content.lower():
            return "It timed out."
        return ""

    @staticmethod
    def _extract_filename(text):
        m = re.search(r'["\']([^"\']+\.\w+)["\']', text)
        if m:
            return m.group(1)
        m = re.search(r'(\w+\.\w{1,5})', text)
        if m:
            return m.group(1)
        return None


class MockAnthropic:
    """Drop-in replacement for anthropic.Anthropic() for offline testing."""

    def __init__(self, **kwargs):
        self.messages = MockMessages()
