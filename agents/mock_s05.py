"""
Mock client for s05_skill_loading.py -- offline demonstration.

Simulates the model's behavior:
  Turn 1: "What skills are available?" → text listing skills
  Turn 2: "Load the code-review skill" → load_skill tool call → text summary
  Turn 3: "Review this file: agents/s05_skill_loading.py" → read_file + text review
  Turn 4: (fallback) → generic helpful text

Usage:
    OFFLINE=1 python agents/s05_skill_loading.py
"""

import json
import re
from dataclasses import dataclass, field


# -- Tiny response model mirroring anthropic SDK types --

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


@dataclass
class MockMessages:
    """Stateful responder that walks through a scripted conversation."""

    _turn: int = 0

    def create(self, *, model, system, messages, tools, max_tokens):
        # Inspect the latest user message to decide the response
        last_user = _last_user_text(messages)
        has_tool_result = _has_tool_result(messages)

        # After a tool_result → provide a textual follow-up
        if has_tool_result:
            return self._after_tool(messages)

        # Keyword-based routing (check "load" before "skill" to avoid false match)
        if _matches(last_user, ["load", "読み込", "ロード"]):
            return self._load_skill(last_user)
        if _matches(last_user, ["skill", "available", "スキル", "何"]):
            return self._list_skills(system)
        if _matches(last_user, ["review", "レビュー", "check", "チェック"]):
            return self._review(last_user)
        if _matches(last_user, ["mcp", "MCP", "server", "サーバ"]):
            return self._load_skill_by_name("mcp-builder")
        if _matches(last_user, ["build", "agent", "エージェント"]):
            return self._load_skill_by_name("agent-builder")
        if _matches(last_user, ["pdf", "PDF"]):
            return self._load_skill_by_name("pdf")

        # Fallback
        self._turn += 1
        return MockResponse(content=[TextBlock(
            text=(
                "I'm a coding agent with skill-loading capabilities. "
                "Ask me what skills are available, or ask me to load one!\n\n"
                "Try: 'What skills are available?' or 'Load the code-review skill'"
            )
        )])

    # -- Scripted responses --

    def _list_skills(self, system: str):
        # Extract skill list from system prompt (just like the real model would)
        skills_section = ""
        for line in system.splitlines():
            if line.strip().startswith("- "):
                skills_section += line + "\n"
        self._turn += 1
        return MockResponse(content=[TextBlock(
            text=(
                "Here are the skills I have access to:\n\n"
                f"{skills_section}\n"
                "I can load any of these skills to get detailed instructions. "
                "Just ask me to load the one you need!"
            )
        )])

    def _load_skill(self, text: str):
        # Try to extract a skill name from the user text
        name = _extract_skill_name(text)
        return self._load_skill_by_name(name)

    def _load_skill_by_name(self, name: str):
        self._turn += 1
        return MockResponse(
            content=[
                TextBlock(text=f"Let me load the **{name}** skill for detailed instructions."),
                ToolUseBlock(
                    id=f"mock_tool_{self._turn}",
                    name="load_skill",
                    input={"name": name},
                ),
            ],
            stop_reason="tool_use",
        )

    def _review(self, text: str):
        # First load the code-review skill, then the model will read the file
        self._turn += 1
        return MockResponse(
            content=[
                TextBlock(text="I'll load the code-review skill first to follow best practices."),
                ToolUseBlock(
                    id=f"mock_tool_{self._turn}",
                    name="load_skill",
                    input={"name": "code-review"},
                ),
            ],
            stop_reason="tool_use",
        )

    def _after_tool(self, messages):
        """Respond after receiving a tool_result."""
        # Find what tool was called by looking at the assistant's last message
        tool_name = _last_tool_name(messages)
        tool_result = _last_tool_result_text(messages)
        self._turn += 1

        if tool_name == "load_skill":
            # Summarise the loaded skill
            skill_name = _extract_skill_tag(tool_result)
            return MockResponse(content=[TextBlock(
                text=(
                    f"I've loaded the **{skill_name}** skill. "
                    f"I now have detailed knowledge about {skill_name} workflows.\n\n"
                    "How would you like me to apply this? "
                    "For example, I can follow the skill's instructions on a specific task."
                )
            )])

        if tool_name == "read_file":
            return MockResponse(content=[TextBlock(
                text=(
                    "I've read the file. Based on the code-review skill's checklist:\n\n"
                    "**Security**: Path traversal protection via `safe_path()` ✓\n"
                    "**Correctness**: Tool handlers properly catch exceptions ✓\n"
                    "**Improvement**: Consider adding input validation for `load_skill` name parameter.\n\n"
                    "Overall the code looks solid."
                )
            )])

        # Generic follow-up
        return MockResponse(content=[TextBlock(
            text=f"Done. The `{tool_name}` tool returned successfully. What's next?"
        )])


@dataclass
class MockClient:
    """Drop-in replacement for anthropic.Anthropic with mock messages."""
    messages: MockMessages = field(default_factory=MockMessages)


# -- Helpers --

def _last_user_text(messages: list) -> str:
    """Extract text from the last user message."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg["content"]
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        continue
                    if isinstance(item, dict) and "text" in item:
                        return item["text"]
                    if isinstance(item, str):
                        return item
            return str(content)
    return ""


def _has_tool_result(messages: list) -> bool:
    """Check if the most recent user message contains a tool_result."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg["content"]
            if isinstance(content, list):
                return any(
                    isinstance(item, dict) and item.get("type") == "tool_result"
                    for item in content
                )
            return False
    return False


def _last_tool_name(messages: list) -> str:
    """Find the name of the last tool_use block in assistant messages."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg["content"]
            if isinstance(content, list):
                for block in reversed(content):
                    if hasattr(block, "type") and block.type == "tool_use":
                        return block.name
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        return block.get("name", "")
    return ""


def _last_tool_result_text(messages: list) -> str:
    """Extract text from the most recent tool_result."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg["content"]
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        c = item.get("content", "")
                        return c if isinstance(c, str) else str(c)
    return ""


def _matches(text: str, keywords: list) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _extract_skill_name(text: str) -> str:
    """Try to extract a skill name from user text."""
    known = ["code-review", "pdf", "agent-builder", "mcp-builder"]
    text_lower = text.lower()
    for name in known:
        if name in text_lower:
            return name
    # Fuzzy matches
    if any(w in text_lower for w in ["review", "レビュー"]):
        return "code-review"
    if any(w in text_lower for w in ["pdf"]):
        return "pdf"
    if any(w in text_lower for w in ["mcp", "server"]):
        return "mcp-builder"
    if any(w in text_lower for w in ["agent", "エージェント", "build"]):
        return "agent-builder"
    return "agent-builder"  # default


def _extract_skill_tag(text: str) -> str:
    """Extract skill name from <skill name="..."> tag."""
    m = re.search(r'<skill name="([^"]+)">', text)
    return m.group(1) if m else "unknown"
